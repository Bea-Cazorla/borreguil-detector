#!/usr/bin/env python3
"""
borreguil_pipeline.py — Pipeline unificado para identificación de borreguiles
/ mountain wet meadows en cualquier zona de montaña.

USO BÁSICO:
    python borreguil_pipeline.py --input puntos.kml --output ./out

CON VERDAD-TERRENO (para entrenar / validar el modelo localmente):
    python borreguil_pipeline.py --input puntos.kml \
                                 --truth verdad_campo.kml \
                                 --output ./out

OPCIONES PRINCIPALES:
    --input          KML/Shp/GeoJSON con puntos candidatos (o usa --generate N)
    --generate N     Generar N puntos aleatorios en el área de estudio
    --truth          KML/Shp con borreguiles verificados en campo. PRESENCE-ONLY:
                     todos los puntos se asumen borreguil; no hace falta atributo.
    --study-area / --wdpa / --name   Definición del área de estudio
    --backend        mpc (Microsoft Planetary Computer, sin cuenta) | gee
                     (Google Earth Engine, requiere cuenta y --gee-project)
    --gee-project    Nombre del proyecto Earth Engine (p. ej. ee-tunombre)
    --output         Carpeta destino (default: ./output)
    --years          Años Sentinel-2; admite rangos (default: 2017-2025)
    --skip-imgs      Saltar descarga de imágenes ESRI (más rápido)
    --skip-s2        Saltar Sentinel-2 (más rápido, sin NDVI/Clre/etc.)
    --threshold      Umbral RF para BORREGUIL (default: 0.5)

VERDAD-TERRENO (presence-only):
    El KML de --truth solo necesita puntos (coordenadas) de borreguiles
    confirmados. El pipeline genera automáticamente pseudo-ausencias
    (puntos candidatos lejos de cualquier borreguil verificado) como
    clase negativa para entrenar el Random Forest.

Para una zona nueva, ejecutar al menos una vez sin --truth para generar la
clasificación con el modelo "por defecto" calibrado en Sierra Nevada; después
verificar en campo los puntos POSIBLE BORREGUIL y reentrenar con --truth.

OUTPUTS:
    out/
        ├── classification.csv          — todas las features y predicciones
        ├── Clasificacion_puntos.xlsx   — tabla coloreada con autofiltro
        ├── Informe_puntos.docx         — informe Word con imágenes por punto
        ├── mapa.html                   — mapa Folium interactivo
        └── imgs/                       — imágenes ESRI con marcador (si no --skip-imgs)
"""
import argparse, os, sys, csv, math, time, io, json, tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

# rasterio SOLO hace falta para leer rásters en local (backend MPC y
# PlanetScope). Con el backend GEE el cómputo es en la nube y no se usa. Como sus
# librerías nativas pueden faltar o estar bloqueadas por el sistema (p. ej. el
# Control de aplicaciones de Windows), su ausencia no debe impedir arrancar: se
# sustituye por este testigo, que explica el problema solo si se llega a usar.
class _RasterioNoDisponible:
    def __init__(self, error):
        self._error = error

    def __getattr__(self, nombre):
        raise ImportError(
            'No se pudo cargar «rasterio», necesario para el backend Microsoft '
            'Planetary Computer (MPC) y PlanetScope. Usa el backend '
            'Google Earth Engine, que no lo necesita. '
            f'Causa original: {self._error}')


# scikit-image SOLO hace falta para las texturas de las imágenes ESRI (GLCM y
# morfología). Su DLL también puede estar bloqueada por Windows; sin ella la app
# arranca igual y funciona sin imágenes (modo rápido o «Saltar imágenes»).
class _SkimageNoDisponible:
    def __init__(self, error):
        self._error = error

    def __call__(self, *args, **kwargs):
        raise ImportError(
            'No se pudo cargar «scikit-image», necesario para las texturas de las '
            'imágenes. Usa el modo rápido o «Saltar imágenes», que no lo necesitan. '
            f'Causa original: {self._error}')


# ===== Lazy imports (so --help works without all deps) =====
def _imports():
    global np, gpd, requests, Image, ImageDraw, rasterio
    global Client, planetary_computer, pyproj, ndi
    global graycomatrix, graycoprops, opening, disk, rgb2gray
    global RandomForestClassifier, GroupKFold, StratifiedKFold, cross_val_predict
    global roc_auc_score, confusion_matrix
    import numpy as np
    import geopandas as gpd
    import requests
    from PIL import Image, ImageDraw
    try:
        import rasterio
    except Exception as exc:            # ImportError, DLL bloqueada, GDAL ausente…
        rasterio = _RasterioNoDisponible(exc)
        print(f'  AVISO: rasterio no disponible ({exc}). El backend MPC quedará '
              'inhabilitado; usa Google Earth Engine.')
    from pystac_client import Client
    import planetary_computer
    import pyproj
    from scipy import ndimage as ndi
    try:
        from skimage.feature import graycomatrix, graycoprops
        try:
            from skimage.morphology import opening, disk        # skimage ≥0.26
        except ImportError:
            from skimage.morphology import binary_opening as opening, disk  # versiones antiguas
        from skimage.color import rgb2gray
    except Exception as exc:            # ImportError, DLL bloqueada…
        graycomatrix = graycoprops = opening = disk = rgb2gray = _SkimageNoDisponible(exc)
        print(f'  AVISO: scikit-image no disponible ({exc}). Sin texturas de imagen: '
              'usa el modo rápido o «Saltar imágenes».')
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score, confusion_matrix


# ============================================================
# UTIL
# ============================================================
NS = {'kml': 'http://www.opengis.net/kml/2.2'}

def read_points(path):
    """Leer puntos de KML, Shapefile o GeoJSON. Devuelve lista de dicts."""
    path = str(path)
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if path.lower().endswith('.kml'):
        rows = []
        root = ET.parse(path).getroot()
        # Independiente del namespace: hay KML con xmlns opengis.net y otros con
        # earth.google.com (OruxMaps, etc.). Se compara por el nombre local de la
        # etiqueta (sin el prefijo de namespace).
        _local = lambda tag: tag.rsplit('}', 1)[-1]
        for pm in root.iter():
            if _local(pm.tag) != 'Placemark':
                continue
            d = {}
            coord_el = None; name_txt = None
            for el in pm.iter():
                lt = _local(el.tag)
                if lt == 'SimpleData' and el.get('name'):
                    d[el.get('name')] = el.text
                elif lt == 'coordinates' and coord_el is None:
                    coord_el = el
                elif lt == 'name' and name_txt is None and el.text:
                    name_txt = el.text.strip()
            if coord_el is None or not (coord_el.text and coord_el.text.strip()):
                continue
            parts = coord_el.text.strip().split()[0].split(',')
            d['lon'] = float(parts[0]); d['lat'] = float(parts[1])
            if name_txt:
                d.setdefault('name', name_txt)
            if 'ID' not in d and 'id' in d:
                d['ID'] = d['id']
            elif 'ID' not in d:
                d['ID'] = pm.get('id') or name_txt or f'pt_{len(rows)}'
            rows.append(d)
        return rows
    elif path.lower().endswith('.csv'):
        return _read_csv_points(path)
    else:
        try:
            gdf = gpd.read_file(path).to_crs(4326)
        except ImportError:
            # Sin pyogrio/GDAL (p. ej. su DLL bloqueada por Windows) un GeoJSON se
            # puede leer igual, porque es JSON; un Shapefile no.
            if not path.lower().endswith(('.geojson', '.json')):
                raise
            return _read_geojson_sin_gdal(path)
        rows = []
        for i, row in gdf.iterrows():
            d = dict(row)
            d['lon'] = row.geometry.x
            d['lat'] = row.geometry.y
            if 'ID' not in d and 'id' not in d:
                d['ID'] = f'pt_{i}'
            elif 'id' in d and 'ID' not in d:
                d['ID'] = str(d['id'])
            d.pop('geometry', None)
            rows.append(d)
        return rows


# Lo que admiten los recuadros de puntos de la app: un Shapefile son varios ficheros
# (.shp, .dbf, .shx, .prj, a veces .cpg), o un .zip con todos.
FORMATOS_PUNTOS = ('kml', 'geojson', 'json', 'csv', 'zip', 'shp', 'shx', 'dbf', 'prj', 'cpg')


def guardar_subida(ficheros, en=False):
    """Guarda en una carpeta temporal, con sus nombres, lo subido en un recuadro de
    puntos (objetos con .name y .getvalue(), como los de Streamlit) y devuelve la ruta
    que hay que pasar a read_points: el .shp, el .zip o el único fichero de puntos
    (KML, GeoJSON, CSV); None si no hay nada. ValueError, con lo que hay que hacer (en
    inglés si `en`), si a un .shp le faltan partes o si hay varios ficheros de puntos.
    (Hasta el 24-11-2026 la app guardaba un solo fichero y con otra extensión: un
    Shapefile no se podía leer nunca.)"""
    import shutil
    import tempfile
    if not ficheros:
        return None
    if not isinstance(ficheros, (list, tuple)):
        ficheros = [ficheros]
    d = tempfile.mkdtemp(prefix='puntos_')
    rutas = []
    for f in ficheros:
        ruta = os.path.join(d, os.path.basename(f.name) or 'puntos.json')
        with open(ruta, 'wb') as h:
            h.write(f.getvalue())
        rutas.append(ruta)
    try:
        return _ruta_de_lectura(rutas, en)
    except ValueError:
        shutil.rmtree(d, ignore_errors=True)
        raise


def _ruta_de_lectura(rutas, en=False):
    """De los ficheros guardados por guardar_subida, el que hay que leer."""
    def ext(r):
        return os.path.splitext(r)[1].lower()

    def txt(es, en_):
        return en_ if en else es

    shp = [r for r in rutas if ext(r) == '.shp']
    if shp:
        base = os.path.splitext(shp[0])[0].lower()
        faltan = [e for e in ('.dbf', '.shx')
                  if not any(os.path.splitext(r)[0].lower() == base and ext(r) == e
                             for r in rutas)]
        if faltan:
            raise ValueError(txt(
                f'Al Shapefile le falta {" y ".join(faltan)}. Selecciona a la vez todos '
                'sus ficheros (.shp, .dbf, .shx y .prj) o súbelo en un .zip.',
                f'The Shapefile is missing {" and ".join(faltan)}. Select all its files '
                'at once (.shp, .dbf, .shx and .prj) or upload it as a .zip.'))
        return shp[0]
    zips = [r for r in rutas if ext(r) == '.zip']
    if zips:
        return zips[0]
    puntos = [r for r in rutas if ext(r) in ('.kml', '.geojson', '.json', '.csv')]
    if len(puntos) > 1:
        raise ValueError(txt(
            'Sube un solo fichero de puntos (o todas las partes de un Shapefile).',
            'Upload a single points file (or all the parts of a Shapefile).'))
    if not puntos:
        raise ValueError(txt(
            'No hay ningún fichero de puntos: si es un Shapefile, falta el .shp.',
            'There is no points file: if it is a Shapefile, the .shp is missing.'))
    return puntos[0]


COLUMNAS_LON = ('lon', 'longitude', 'longitud', 'long', 'lng', 'x')
COLUMNAS_LAT = ('lat', 'latitude', 'latitud', 'y')


def _read_csv_points(path):
    """Puntos de un CSV (separado por comas o por punto y coma, como lo guarda Excel
    en español) con la longitud y la latitud en grados (WGS84) en columnas `lon` y
    `lat` (o `longitude`/`latitude`, `x`/`y`…). Admite la coma decimal."""
    with open(path, encoding='utf-8-sig', newline='') as f:
        texto = f.read()
    primera = texto.splitlines()[0] if texto.strip() else ''
    sep = ';' if primera.count(';') > primera.count(',') else ','
    lector = csv.DictReader(io.StringIO(texto), delimiter=sep)
    campos = {c.strip().lower(): c for c in (lector.fieldnames or []) if c}
    c_lon = next((campos[k] for k in COLUMNAS_LON if k in campos), None)
    c_lat = next((campos[k] for k in COLUMNAS_LAT if k in campos), None)
    if c_lon is None or c_lat is None:
        raise ValueError(f'El CSV {Path(path).name} no tiene columnas de coordenadas: '
                         'hacen falta `lon` y `lat` (en grados, WGS84). Columnas: '
                         + ', '.join(lector.fieldnames or []))

    def numero(v):
        try:
            return float(str(v).strip().replace(',', '.'))
        except ValueError:
            return None

    rows = []
    for i, fila in enumerate(lector):
        x, y = numero(fila.get(c_lon)), numero(fila.get(c_lat))
        if x is None or y is None:
            continue
        if abs(x) > 180 or abs(y) > 90:
            raise ValueError(f'En el CSV {Path(path).name} las coordenadas no son grados '
                             f'(fila {i + 2}: {x}, {y}); ¿están en UTM? Hacen falta '
                             '`lon` y `lat` en grados (WGS84).')
        d = {k: v for k, v in fila.items() if k not in (c_lon, c_lat) and k is not None}
        d['lon'], d['lat'] = x, y
        if 'ID' not in d and 'id' in d:
            d['ID'] = str(d['id'])
        elif 'ID' not in d:
            d['ID'] = f'pt_{i}'
        rows.append(d)
    return rows


def _read_geojson_sin_gdal(path):
    """Puntos de un GeoJSON leídos como JSON, sin GDAL. Mismo resultado que la
    lectura con geopandas: un dict por punto con sus atributos, lon/lat en EPSG:4326
    e ID. Admite el miembro «crs» antiguo (p. ej. EPSG:25830) y toma el primer punto
    de un MultiPoint."""
    import re
    with open(path, encoding='utf-8-sig') as f:
        datos = json.load(f)
    feats = datos.get('features', []) if datos.get('type') == 'FeatureCollection' \
        else [datos]
    a_wgs84 = None
    crs = ((datos.get('crs') or {}).get('properties') or {}).get('name', '')
    m = re.search(r'EPSG:+(\d+)', str(crs))
    if m and m.group(1) not in ('4326', '4258'):
        a_wgs84 = pyproj.Transformer.from_crs(int(m.group(1)), 4326, always_xy=True)
    rows = []
    for i, ft in enumerate(feats):
        geom = ft.get('geometry') or {}
        coords = geom.get('coordinates')
        if geom.get('type') == 'MultiPoint' and coords:
            coords = coords[0]
        if geom.get('type') not in ('Point', 'MultiPoint') or not coords:
            continue
        x, y = float(coords[0]), float(coords[1])
        if a_wgs84 is not None:
            x, y = a_wgs84.transform(x, y)
        d = dict(ft.get('properties') or {})
        d['lon'], d['lat'] = x, y
        if 'ID' not in d and 'id' not in d:
            d['ID'] = f'pt_{i}'
        elif 'id' in d and 'ID' not in d:
            d['ID'] = str(d['id'])
        rows.append(d)
    return rows


def bbox_with_buffer(rows, buf=0.02):
    lons = [r['lon'] for r in rows]; lats = [r['lat'] for r in rows]
    return (min(lons)-buf, min(lats)-buf, max(lons)+buf, max(lats)+buf)


def bbox_from_geom(geom, buf=0.02):
    """BBox de un polígono shapely, con buffer en grados."""
    minx, miny, maxx, maxy = geom.bounds
    return (minx-buf, miny-buf, maxx+buf, maxy+buf)


def _utm_epsg_for(rows):
    lat = sum(r['lat'] for r in rows) / len(rows)
    lon = sum(r['lon'] for r in rows) / len(rows)
    zone = int((lon + 180) / 6) + 1
    return 32600 + zone if lat >= 0 else 32700 + zone


def _xy(rows, epsg):
    t = pyproj.Transformer.from_crs(4326, epsg, always_xy=True)
    return [t.transform(r['lon'], r['lat']) for r in rows]


# ---- Rejilla Sentinel-2 / Sentinel-1: snap y huellas de muestreo --------------
# Las imágenes S2 L2A y S1-RTC de MPC están en UTM con la rejilla alineada a
# múltiplos de la resolución (10 m / 20 m). El muestreo del pipeline usa ventanas
# 5×5 px a 10 m (= 50 m) y 3×3 px a 20 m (= 60 m). Estas funciones permiten (a)
# colocar los puntos en el CENTRO de un píxel S2 y (b) dibujar la huella en el mapa.
def _utm_for_lonlat(lon, lat):
    zone = int((lon + 180) / 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def snap_to_s2_grid(pts):
    """pts: lista de (lon, lat). Devuelve la lista movida al CENTRO del píxel
    Sentinel-2 de 10 m que contiene cada punto (rejilla alineada a 10 m en UTM).
    Así el valor extraído corresponde a un píxel S2 bien definido."""
    import pyproj                                    # autónoma (no requiere _imports)
    if not pts:
        return pts
    lon0 = sum(p[0] for p in pts) / len(pts)
    lat0 = sum(p[1] for p in pts) / len(pts)
    epsg = _utm_for_lonlat(lon0, lat0)
    fwd = pyproj.Transformer.from_crs(4326, epsg, always_xy=True)
    inv = pyproj.Transformer.from_crs(epsg, 4326, always_xy=True)
    out = []
    for lon, lat in pts:
        x, y = fwd.transform(lon, lat)
        xc = math.floor(x / 10.0) * 10.0 + 5.0
        yc = math.floor(y / 10.0) * 10.0 + 5.0
        lo, la = inv.transform(xc, yc)
        out.append((lo, la))
    return out


def pixel_footprints(rows):
    """Para cada punto (con 'lon','lat') devuelve un dict de polígonos en formato
    Folium ([[lat,lon],…]) que representan la rejilla de muestreo:
      'pixel10'   : el píxel S2 de 10 m que contiene el punto (NDVI, B02-B04, B08…)
      'win_s2_10' : ventana de muestreo 5×5 px a 10 m (= 50 m) — la zona que se
                    promedia para las bandas de 10 m (vale también para Sentinel-1,
                    que comparte la rejilla de 10 m).
      'win_s2_20' : ventana 3×3 px a 20 m (= 60 m) — bandas de 20 m (B05/B11/B12…,
                    usadas por clre, ndmi, etc.).
    """
    import pyproj                                    # autónoma (no requiere _imports)
    if not rows:
        return []
    lon0 = sum(r['lon'] for r in rows) / len(rows)
    lat0 = sum(r['lat'] for r in rows) / len(rows)
    epsg = _utm_for_lonlat(lon0, lat0)
    fwd = pyproj.Transformer.from_crs(4326, epsg, always_xy=True)
    inv = pyproj.Transformer.from_crs(epsg, 4326, always_xy=True)

    def rect(x0, y0, x1, y1):
        out = []
        for cx, cy in [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]:
            lo, la = inv.transform(cx, cy)
            out.append([la, lo])           # Folium quiere [lat, lon]
        return out

    res = []
    for r in rows:
        x, y = fwd.transform(r['lon'], r['lat'])
        px0 = math.floor(x / 10.0) * 10.0; py0 = math.floor(y / 10.0) * 10.0
        qx0 = math.floor(x / 20.0) * 20.0; qy0 = math.floor(y / 20.0) * 20.0
        res.append({
            'pixel10':   rect(px0, py0, px0 + 10, py0 + 10),
            'win_s2_10': rect(px0 - 20, py0 - 20, px0 + 30, py0 + 30),   # 5×5 px = 50 m
            'win_s2_20': rect(qx0 - 20, qy0 - 20, qx0 + 40, qy0 + 40),   # 3×3 px = 60 m
        })
    return res


# Valores que marcan presencia / ausencia (insensible a mayúsculas y acentos; ver
# valor_si_no). Los nombres de columna que se buscan están en CLAVES_ETIQUETA.
_TRUTH_POS = {'si', 'sí', 'yes', 'y', '1', 'true', 'presencia', 'presence',
              'borreguil', 'positivo', 'pos', 'p'}
_TRUTH_NEG = {'no', 'n', '0', 'false', 'ausencia', 'absence', 'no_borreguil',
              'noborreguil', 'negativo', 'neg', 'ausente'}


# Puntos candidatos: la app guarda la etiqueta en `Borreguil` (si/no) y la duda en
# `Duda` (si), pero los ficheros de otras personas traen otros nombres (`borreguil`
# yes/no, `presence` 1/0, `doubtful`…). leer_etiquetas() los traduce al cargar.
# Sin 'tipo' ni 'y' (en un fichero de candidatos suelen ser el tipo de borreguil o
# una coordenada).
CLAVES_ETIQUETA = ('Borreguil', 'presencia', 'presence', 'presente', 'present',
                   'etiqueta', 'label', 'clase', 'class', 'target', 'verdad', 'truth')
CLAVES_DUDA = ('Duda', 'dudoso', 'dudosa', 'doubtful', 'doubt', 'uncertain',
               'incierto')
_POS_EXTRA = {'presente', 'present', 'verdadero', 's'}
_NEG_EXTRA = {'absent', 'falso', 'f'}
_MIN_ENTENDIDOS = 0.9    # fracción de valores reconocidos para aceptar otra columna


def valor_si_no(v):
    """'si', 'no', '' (vacío) o '?' (un valor que no se entiende) para la etiqueta de
    un punto: acepta si/sí/yes/1/true/presencia…, no/0/false/ausencia…, y también
    booleanos y números (1.0 / 0.0)."""
    import numbers, unicodedata
    if v is None or (isinstance(v, float) and v != v):
        return ''
    if isinstance(v, bool) or type(v).__name__ in ('bool_', 'bool'):   # también numpy
        return 'si' if v else 'no'
    if isinstance(v, numbers.Real):
        v = float(v)
        return '' if v != v else {1.0: 'si', 0.0: 'no'}.get(v, '?')
    s = unicodedata.normalize('NFKD', str(v)).encode('ascii', 'ignore').decode().strip().lower()
    if s == '' or s == 'nan':
        return ''
    if s in _TRUTH_POS or s in _POS_EXTRA:
        return 'si'
    if s in _TRUTH_NEG or s in _NEG_EXTRA:
        return 'no'
    try:
        return {1.0: 'si', 0.0: 'no'}.get(float(s), '?')
    except ValueError:
        return '?'


def _buscar_columna(rows, claves, acepta):
    """La primera de `claves` (sin distinguir mayúsculas) que está en los puntos y que
    `acepta` da por buena. Devuelve (nombre tal cual en el fichero, valores) o None."""
    nombres = {}
    for r in rows:
        for k in r:
            if isinstance(k, str):
                nombres.setdefault(k.lower(), k)
    for i, clave in enumerate(claves):
        k = nombres.get(clave.lower())
        if k is None:
            continue
        valores = [valor_si_no(r.get(k)) for r in rows]
        if acepta(i, valores):
            return k, valores
    return None


def leer_etiquetas(rows):
    """Traduce a `Borreguil` (si/no) y `Duda` (si) la etiqueta de los puntos
    candidatos, venga en la columna que venga (ver CLAVES_ETIQUETA/CLAVES_DUDA), y
    dice qué ha leído. Modifica `rows`:
      - `Borreguil` queda con 'si'/'no' en los puntos con un valor reconocido; los
        vacíos o no reconocidos no se tocan (siguen sin etiqueta).
      - Si la columna original solo se diferencia en mayúsculas (`borreguil`,
        `duda`), se renombra; si tiene otro nombre (`presence`), se conserva.
    Una columna que no sea `Borreguil` solo se usa si casi todos sus valores
    (≥ 90 %) son de sí o no: así no se toma por etiqueta, p. ej., una `clase` con
    tipos de vegetación.

    Devuelve {'columna', 'n_si', 'n_no', 'n_sin', 'no_entendidos' (valor → nº),
    'duda', 'n_duda', 'columnas' (las del fichero), 'descartadas' (columnas con
    nombre de etiqueta pero valores que no son de sí o no)}."""
    info = {'columna': None, 'n_si': 0, 'n_no': 0, 'n_sin': len(rows),
            'no_entendidos': {}, 'duda': None, 'n_duda': 0, 'descartadas': [],
            'columnas': sorted({k for r in rows for k in r if isinstance(k, str)}
                               - {'lon', 'lat', 'ID', 'geometry'})}

    def acepta_etiqueta(i, valores):
        llenos = [v for v in valores if v]
        if not llenos or all(v == '?' for v in llenos):
            if llenos:
                info['descartadas'].append(CLAVES_ETIQUETA[i])
            return False
        if i == 0:                                  # `Borreguil`: la de la app
            return True
        ok = sum(v != '?' for v in llenos) / len(llenos) >= _MIN_ENTENDIDOS
        if not ok:
            info['descartadas'].append(CLAVES_ETIQUETA[i])
        return ok

    hallada = _buscar_columna(rows, CLAVES_ETIQUETA, acepta_etiqueta)
    if hallada:
        k, valores = hallada
        info['columna'] = k
        renombrar = k != 'Borreguil' and k.lower() == 'borreguil'
        raros = {}
        for r, v in zip(rows, valores):
            original = r.pop(k, None) if renombrar else r.get(k)
            if v in ('si', 'no'):
                r['Borreguil'] = v
            elif renombrar:
                r['Borreguil'] = original
            if v == '?':
                raros[_txt(original)] = raros.get(_txt(original), 0) + 1
        info['n_si'] = sum(v == 'si' for v in valores)
        info['n_no'] = sum(v == 'no' for v in valores)
        info['n_sin'] = len(rows) - info['n_si'] - info['n_no']
        info['no_entendidos'] = raros

    def acepta_duda(i, valores):
        return any(v in ('si', 'no') for v in valores)

    hallada = _buscar_columna(rows, CLAVES_DUDA, acepta_duda)
    if hallada:
        k, valores = hallada
        info['duda'] = k
        renombrar = k != 'Duda' and k.lower() == 'duda'
        for r, v in zip(rows, valores):
            if renombrar:
                r.pop(k, None)
            if v == 'si':
                r['Duda'] = 'si'
            elif k != 'Duda':
                r['Duda'] = ''
        info['n_duda'] = sum(v == 'si' for v in valores)
    info['tipos'] = leer_tipos(rows)
    info['revision'] = leer_revision(rows)
    return info


# Tipo de borreguil en un fichero de puntos (plantilla de datos de campo, en inglés,
# o con los nombres de la app): una columna por nivel, y cada una puede faltar o venir
# vacía en parte de los puntos. Lo que trae un punto cuenta como revisado en ESE
# nivel (lo ha visto una persona). Valores en inglés o en español.
CLAVES_TIPO = {
    'ambiente': ('environment', 'ambiente', 'env', 'setting'),
    'humedad':  ('moisture', 'humedad', 'wetness', 'humidity'),
    'pureza':   ('purity', 'pureza'),
}
_SINONIMOS_TIPO = {
    'ambiente': {'arroyo': ('stream', 'arroyo', 'brook', 'river', 'creek', 'rio',
                            'riparian', 'streamside'),
                 'laguna': ('lake', 'laguna', 'pond', 'lagoon', 'lago', 'tarn',
                            'lakeshore', 'lake-shore'),
                 'ladera': ('slope', 'ladera', 'hillslope', 'hill-slope', 'hillside')},
    'humedad':  {'húmedo': ('wet', 'humedo', 'moist', 'humid'),
                 'seco':   ('dry', 'seco')},
    'pureza':   {'puro':       ('pure', 'puro'),
                 'mixto-agua': ('mixed-water', 'mixto-agua', 'mixedwater', 'water-mixed'),
                 'mixto-roca': ('mixed-rock', 'mixto-roca', 'mixedrock', 'rock-mixed')},
}
# Lo que se escribe al guardar (plantilla en inglés)
TIPO_EN = {'arroyo': 'stream', 'laguna': 'lake', 'ladera': 'slope', 'húmedo': 'wet',
           'seco': 'dry', 'puro': 'pure', 'mixto-agua': 'mixed-water',
           'mixto-roca': 'mixed-rock'}
COLUMNA_TIPO_EN = {'ambiente': 'environment', 'humedad': 'moisture', 'pureza': 'purity'}


def valor_tipo(nivel, v):
    """La categoría de la app ('arroyo', 'húmedo'…) para un valor de tipo de un
    fichero, '' si viene vacío o '?' si no se entiende."""
    import unicodedata
    if v is None or (isinstance(v, float) and v != v):
        return ''
    s = (unicodedata.normalize('NFKD', str(v)).encode('ascii', 'ignore').decode()
         .strip().lower().replace('_', '-').replace(' ', '-'))
    if s in ('', 'nan', '-'):
        return ''
    for cat, sin in _SINONIMOS_TIPO[nivel].items():
        if s in sin:
            return cat
    return '?'


def leer_tipos(rows, es_ausencia=None):
    """Lee el tipo de borreguil que traen los puntos (CLAVES_TIPO) y lo da por
    revisado en cada nivel que traiga cada punto. Modifica `rows`. Una columna solo
    se usa si casi todos sus valores (≥ 90 %) son categorías reconocidas. Un punto
    marcado como «no borreguil» no tiene tipo: lo que traiga se ignora.

    Los ficheros que guarda la propia app con las marcas de revisión antiguas
    (`tipo_revisado`, `<nivel>_revisado`) ya dicen qué está revisado: ahí `ambiente`,
    `humedad` y `pureza` pueden ser la propuesta de la app, y no se leen como
    revisados.

    `es_ausencia(punto)` dice si es «no borreguil» (por defecto, Borreguil = no).
    Devuelve {nivel: {'columna', 'n', 'no_entendidos'}, 'en_ausencias': nº,
    'descartadas': [columnas con nombre de tipo pero valores que no lo son]}."""
    es_ausencia = es_ausencia or (lambda r: _txt(r.get('Borreguil')).lower() == 'no')
    info = {'en_ausencias': 0, 'descartadas': []}
    marcas = {'tipo_revisado'} | {n + '_revisado' for n in NIVELES_TIPO}
    if any(k in marcas for r in rows for k in r):
        return info
    nombres = {}
    for r in rows:
        for k in r:
            if isinstance(k, str):
                nombres.setdefault(k.lower(), k)
    por_punto = [{} for _ in rows]
    for nivel, claves in CLAVES_TIPO.items():
        for clave in claves:
            k = nombres.get(clave.lower())
            if k is None:
                continue
            valores = [valor_tipo(nivel, r.get(k)) for r in rows]
            llenos = [v for v in valores if v]
            if not llenos or sum(v != '?' for v in llenos) / len(llenos) < _MIN_ENTENDIDOS:
                if llenos:
                    info['descartadas'].append(k)
                continue
            raros = {}
            for r, v, d in zip(rows, valores, por_punto):
                if v == '?':
                    raros[_txt(r.get(k))] = raros.get(_txt(r.get(k)), 0) + 1
                elif v:
                    d[nivel] = v
            info[nivel] = {'columna': k, 'n': 0, 'no_entendidos': raros}
            break
    for r, d in zip(rows, por_punto):
        if not d:
            continue
        if es_ausencia(r):
            info['en_ausencias'] += 1
            continue
        set_niveles_revisados(r, cuando=_fecha_txt(r.get('date')), **d)
        for nivel in d:
            info[nivel]['n'] += 1
    return info


# ------------------------------------------------------------
# Quién revisa cada punto, su perfil y su certidumbre
# ------------------------------------------------------------
# Cada vez que una persona dice si un punto es borreguil (en «Revisar puntos», al
# pinchar en el mapa o en su fichero de campo) se guarda quién lo dice, su perfil y lo
# segura que está, con las columnas de la plantilla de datos de campo: `observer`,
# `expertise` y `certainty`. Así se sabe de dónde sale cada etiqueta y se puede medir
# cuánto acierta el modelo según la certidumbre o el perfil. «Dudoso» es otra cosa:
# quien no sabe decir si es borreguil marca el punto como dudoso y no cuenta; la
# certidumbre gradúa un sí o un no.
PERFILES = ('student', 'manager', 'researcher')
CERTIDUMBRES = ('high', 'medium', 'low')
PERFIL_ES = {'student': 'estudiante', 'manager': 'gestor/a', 'researcher': 'investigador/a'}
CERTIDUMBRE_ES = {'high': 'alta', 'medium': 'media', 'low': 'baja'}
CAMPOS_QUIEN = ('observer', 'expertise', 'certainty')
_SINONIMOS_PERFIL = {
    'student': ('student', 'estudiante', 'alumno', 'alumna'),
    'manager': ('manager', 'gestor', 'gestora', 'gestor/a', 'gestor/gestora',
                'technician', 'tecnico', 'técnico', 'tecnica', 'técnica'),
    'researcher': ('researcher', 'investigador', 'investigadora', 'investigador/a',
                   'scientist', 'cientifico', 'científico', 'cientifica', 'científica'),
}
_SINONIMOS_CERTIDUMBRE = {
    'high': ('high', 'alta', 'alto'),
    'medium': ('medium', 'media', 'medio', 'moderate', 'moderada'),
    'low': ('low', 'baja', 'bajo'),
}
# Nombres de columna que se aceptan al leer un fichero (el primero es el de la plantilla)
CLAVES_QUIEN = {
    'observer': ('observer', 'observador', 'observadora', 'revisor', 'revisora',
                 'reviewer'),
    'expertise': ('expertise', 'perfil', 'profile'),
    'certainty': ('certainty', 'certidumbre', 'certeza', 'confianza', 'confidence'),
}


def _de_sinonimos(v, sinonimos):
    t = _txt(v).strip().lower()
    if not t:
        return ''
    for codigo, palabras in sinonimos.items():
        if t in palabras:
            return codigo
    return '?'


def perfil_de(v):
    """'student', 'manager', 'researcher', '' (vacío) o '?' (no se entiende). Admite
    inglés y español (estudiante, gestor/a, investigador/a)."""
    return _de_sinonimos(v, _SINONIMOS_PERFIL)


def certidumbre_de(v):
    """'high', 'medium', 'low', '' (vacío) o '?' (no se entiende). Admite inglés y
    español (alta, media, baja)."""
    return _de_sinonimos(v, _SINONIMOS_CERTIDUMBRE)


def anotar_revision(r, observador='', perfil='', certidumbre=''):
    """Apunta en el punto `r` quién lo acaba de revisar, su perfil y su certidumbre.
    Lo que no se indica se quita: la etiqueta es nueva y lo que hubiera era de otra
    revisión. Modifica y devuelve `r`."""
    p, c = perfil_de(perfil), certidumbre_de(certidumbre)
    for k, v in (('observer', _txt(observador).strip()),
                 ('expertise', p if p != '?' else ''),
                 ('certainty', c if c != '?' else '')):
        if v:
            r[k] = v
        else:
            r.pop(k, None)
    return r


def leer_revision(rows):
    """Lee quién revisó cada punto, su perfil y su certidumbre (CLAVES_QUIEN), venga
    en inglés o en español, y lo deja en `observer`, `expertise` y `certainty` con
    los valores de la plantilla. Un perfil o una certidumbre que no se entiende se
    quita (y se cuenta). Modifica `rows`. Devuelve {campo: {'columna', 'n',
    'no_entendidos'}} de los campos que trae el fichero."""
    nombres = {}
    for r in rows:
        for k in r:
            if isinstance(k, str):
                nombres.setdefault(k.lower(), k)
    leer = {'expertise': perfil_de, 'certainty': certidumbre_de}
    info = {}
    for campo, claves in CLAVES_QUIEN.items():
        k = next((nombres[c] for c in claves if c in nombres), None)
        if k is None:
            continue
        datos = {'columna': k, 'n': 0, 'no_entendidos': {}}
        for r in rows:
            original = r.pop(k, None) if k != campo else r.get(campo)
            v = _txt(original).strip()
            if campo in leer:
                v = leer[campo](v)
                if v == '?':
                    t = _txt(original).strip()
                    datos['no_entendidos'][t] = datos['no_entendidos'].get(t, 0) + 1
                    v = ''
            if v:
                r[campo] = v
                datos['n'] += 1
            else:
                r.pop(campo, None)
        info[campo] = datos
    return info


def resumen_etiquetas(info):
    """[(nivel, texto)] para contar lo que leer_etiquetas() ha leído; nivel es 'ok',
    'info' (ha traducido otra columna) o 'aviso' (no hay etiqueta, o hay valores que
    no entiende)."""
    out = []
    k = info['columna']
    if k is None:
        cols = ', '.join(f'`{c}`' for c in info['columnas'][:25]) or '(ninguna)'
        if len(info['columnas']) > 25:
            cols += f' y {len(info["columnas"]) - 25} más'
        raras = (' Las columnas ' + ', '.join(f'`{c}`' for c in info['descartadas'])
                 + ' tienen nombre de etiqueta, pero sus valores no son de sí o no.'
                 if info['descartadas'] else '')
        out.append(('aviso',
                    'No encuentro ninguna columna que diga si cada punto es borreguil, '
                    'así que ningún punto cuenta como etiquetado (ni para entrenar ni en '
                    f'«Mapa Modelo RF»).{raras} Columnas del fichero: {cols}. Para que se '
                    'usen tus etiquetas, pon una columna `Borreguil` (o `presence`, '
                    '`presencia`, `label`, `clase`…) con si/no, yes/no, 1/0, true/false '
                    'o presencia/ausencia. Si solo quieres que la app clasifique los '
                    'puntos, puedes seguir.'))
    else:
        cuenta = f"{info['n_si']} sí · {info['n_no']} no"
        if info['n_sin']:
            cuenta += f" · {info['n_sin']} sin etiqueta"
        if info['duda']:
            cuenta += f" · {info['n_duda']} dudosos (columna `{info['duda']}`)"
        # la columna de la app o la de la plantilla de datos de campo
        if k in ('Borreguil', 'presence') and info['duda'] in (None, 'Duda', 'doubtful'):
            out.append(('ok', f'Etiquetas de la columna `{k}`: {cuenta}.'))
        else:
            out.append(('info', f'He leído si cada punto es borreguil de la columna '
                                f'`{k}` y lo he pasado a `Borreguil` (si/no): {cuenta}.'))
        if info['no_entendidos']:
            n = sum(info['no_entendidos'].values())
            vals = ', '.join(f'«{v}» ({c})' for v, c in sorted(
                info['no_entendidos'].items(), key=lambda x: -x[1])[:8])
            out.append(('aviso', f'{n} puntos tienen en `{k}` un valor que no es de sí o '
                                 f'no ({vals}): se quedan sin etiqueta.'))
    tipos = info.get('tipos') or {}
    leidos = [(n, tipos[n]) for n in NIVELES_TIPO if n in tipos]
    if leidos:
        out.append(('ok', 'Tipo de campo, que cuenta como revisado: ' + ' · '.join(
            f"{n} en {t['n']} puntos (columna `{t['columna']}`)" for n, t in leidos) + '.'))
        for n, t in leidos:
            if t['no_entendidos']:
                vals = ', '.join(f'«{v}» ({c})' for v, c in sorted(
                    t['no_entendidos'].items(), key=lambda x: -x[1])[:8])
                out.append(('aviso', f"{sum(t['no_entendidos'].values())} puntos tienen en "
                                     f"`{t['columna']}` un valor que no es un {n} "
                                     f"reconocido ({vals}): ese nivel queda sin revisar. "
                                     f"Valores admitidos: "
                                     + ', '.join(TIPO_EN[c] for c in NIVELES_TIPO[n]) + '.'))
    if tipos.get('en_ausencias'):
        out.append(('info', f"{tipos['en_ausencias']} puntos marcados como «no borreguil» "
                             'traen un tipo: se ignora, porque lo que no es borreguil no '
                             'tiene tipo.'))
    if tipos.get('descartadas'):
        out.append(('aviso', 'Las columnas ' + ', '.join(f'`{c}`' for c in tipos['descartadas'])
                             + ' tienen nombre de tipo, pero sus valores no son categorías '
                               'reconocidas: no se usan.'))
    rev = info.get('revision') or {}
    nombres_q = {'observer': 'quién revisó', 'expertise': 'perfil',
                 'certainty': 'certidumbre'}
    leidos_q = [(c, rev[c]) for c in CAMPOS_QUIEN if c in rev and rev[c]['n']]
    if leidos_q:
        out.append(('ok', 'Revisión: ' + ' · '.join(
            f"{nombres_q[c]} en {d['n']} puntos (columna `{d['columna']}`)"
            for c, d in leidos_q) + '.'))
    admitidos = {'expertise': 'student, manager, researcher (o estudiante, gestor/a, '
                              'investigador/a)',
                 'certainty': 'high, medium, low (o alta, media, baja)'}
    for c, texto in admitidos.items():
        d = rev.get(c)
        if d and d['no_entendidos']:
            vals = ', '.join(f'«{v}» ({k})' for v, k in sorted(
                d['no_entendidos'].items(), key=lambda x: -x[1])[:8])
            out.append(('aviso', f"{sum(d['no_entendidos'].values())} puntos tienen en "
                                 f"`{d['columna']}` un valor que no se entiende ({vals}): "
                                 f"se queda sin indicar. Valores admitidos: {texto}."))
    return out


def leer_puntos_de_campo(truth_rows):
    """Lee la etiqueta de los puntos del fichero de «puntos de campo» igual que la de
    «Tus puntos» (leer_etiquetas: columna `presence`, `Borreguil`, `presencia`…, sin
    distinguir mayúsculas; valores sí/no, 1/0, 1.0/0.0, yes/no, true/false…; dudosos;
    tipo; quién revisó). Hasta el 24-11-2026 se leía aparte, con nombres y valores
    exactos, y una ausencia escrita `0.0` (Shapefile, GeoPackage) o en una columna
    `Presence` se tomaba como borreguil.

    Si el fichero no trae ninguna columna de etiqueta, cada punto es un borreguil
    (solo presencias), salvo que haya una columna con nombre de etiqueta cuyos
    valores no se entienden: entonces no se usa ninguno (mejor nada que ausencias
    tomadas por borreguiles). Modifica `truth_rows`. Devuelve el info de
    leer_etiquetas() más 'solo_presencias' (bool) y 'n_sin_usar' (puntos sin una
    etiqueta reconocida, que no se usan)."""
    info = leer_etiquetas(truth_rows)
    info['solo_presencias'] = info['columna'] is None and not info['descartadas']
    if info['solo_presencias']:
        for r in truth_rows:
            r['Borreguil'] = 'si'
        # (el tipo ya se leyó bien: sin etiqueta, ninguno era «no borreguil»)
        info['n_si'], info['n_sin'] = len(truth_rows), 0
    info['n_sin_usar'] = sum(1 for r in truth_rows
                             if _txt(r.get('Borreguil')).lower() not in ('si', 'no'))
    return info


def merge_truth_points(rows, truth_rows, match_m=20, info=None):
    """Incorpora los puntos de campo (presencias y, si las hay, ausencias), leídos con
    leer_puntos_de_campo(). Si un punto coincide (< match_m) con un candidato,
    etiqueta ese candidato; si no, se añade. Los que no tienen una etiqueta
    reconocida no se usan. Un dudoso entra como dudoso (no cuenta). Con `info` (un
    dict), deja en él lo que se ha leído (para contarlo en pantalla). Devuelve
    (n_matched, n_added, n_pos, n_neg)."""
    if not truth_rows:
        return 0, 0, 0, 0
    leido = leer_puntos_de_campo(truth_rows)
    if info is not None:
        info.update(leido)
    truth_rows = [r for r in truth_rows if _txt(r.get('Borreguil')).lower() in ('si', 'no')]
    if not truth_rows:
        return 0, 0, 0, 0
    epsg = _utm_epsg_for(rows + truth_rows)
    cand_xy = _xy(rows, epsg)
    truth_xy = _xy(truth_rows, epsg)
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(cand_xy) if cand_xy else None
    except Exception:
        tree = None
    n_match = 0; n_add = 0; n_pos = 0; n_neg = 0
    for k, (tr, txy) in enumerate(zip(truth_rows, truth_xy)):
        lab = _txt(tr.get('Borreguil')).lower()
        duda = 'si' if _txt(tr.get('Duda')).lower() == 'si' else ''
        if lab == 'si': n_pos += 1
        else: n_neg += 1
        nearest_i, d = None, 1e18
        if tree is not None:
            d, nearest_i = tree.query(txy)
        if nearest_i is not None and d <= match_m:
            rows[nearest_i]['Borreguil'] = lab
            rows[nearest_i]['Duda'] = duda
            rows[nearest_i]['is_truth'] = 1
            # Si el punto de campo trae un tipo ya revisado por una persona, se
            # traslada al candidato con el que coincide (antes solo pasaba la
            # presencia y lo revisado se perdía).
            if tipo_revisado(tr):
                for campo in CAMPOS_REVISION:
                    if campo in tr and not campo.endswith('_regla'):
                        rows[nearest_i][campo] = tr[campo]
            # Quién lo dijo, cómo y con qué certidumbre: van con la etiqueta.
            anotar_revision(rows[nearest_i], tr.get('observer'), tr.get('expertise'),
                            tr.get('certainty'))
            for campo in ('method', 'date'):
                if _txt(tr.get(campo)):
                    rows[nearest_i][campo] = _txt(tr.get(campo))
            n_match += 1
        else:
            new = dict(tr)
            new['ID'] = tr.get('ID', tr.get('id', f'truth_{k:04d}'))
            new['Borreguil'] = lab
            new['is_truth'] = 1
            new['source'] = 'truth'
            rows.append(new)
            n_add += 1
    return n_match, n_add, n_pos, n_neg


def parse_years(spec):
    """Acepta '2023,2024', '2017-2025' o combinaciones '2017-2020,2024'.
    Devuelve una tupla de enteros ordenada y sin duplicados."""
    if spec is None:
        return tuple()
    if not isinstance(spec, str):
        return tuple(spec)
    years = set()
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-', 1)
            years.update(range(int(a), int(b) + 1))
        else:
            years.add(int(part))
    return tuple(sorted(years))


def abreviar_anios(years):
    """Lo contrario de parse_years, para mostrar: (2017, …, 2025) → '2017-2025';
    (2017, 2019, 2020) → '2017, 2019-2020'."""
    ys = sorted({int(y) for y in years})
    if not ys:
        return ''
    tramos, ini, ant = [], ys[0], ys[0]
    for y in ys[1:]:
        if y != ant + 1:
            tramos.append((ini, ant))
            ini = y
        ant = y
    tramos.append((ini, ant))
    return ', '.join(str(a) if a == b else f'{a}-{b}' for a, b in tramos)


# ============================================================
# 1. ESRI World Imagery
# ============================================================
def fetch_esri_imgs(rows, out_dir, delta_lat=0.0025, delta_lon=0.0031,
                    size=(500,400), workers=8):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    URL = ('https://services.arcgisonline.com/ArcGIS/rest/services/'
           'World_Imagery/MapServer/export')
    W, H = size

    def fetch_one(i, r):
        fname = out_dir / f"pt_{i:04d}.jpg"
        if fname.exists() and fname.stat().st_size > 5000:
            return i, True
        lon, lat = r['lon'], r['lat']
        bbox = f"{lon-delta_lon},{lat-delta_lat},{lon+delta_lon},{lat+delta_lat}"
        try:
            resp = requests.get(URL, params={
                'bbox': bbox, 'bboxSR': '4326', 'imageSR': '4326',
                'size': f'{W},{H}', 'format': 'jpg', 'f': 'image'
            }, timeout=20)
            if resp.status_code != 200 or len(resp.content) < 1500:
                return i, False
            im = Image.open(io.BytesIO(resp.content)).convert('RGB')
            dr = ImageDraw.Draw(im)
            cx, cy = W//2, H//2
            dr.ellipse([cx-10, cy-10, cx+10, cy+10], outline='white', width=3)
            dr.ellipse([cx-7, cy-7, cx+7, cy+7], outline='red', width=2)
            im.save(fname, 'JPEG', quality=80)
            return i, True
        except Exception:
            return i, False

    print(f'  → {len(rows)} imágenes a {out_dir}…')
    ok = 0; t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_one, i, r): i for i, r in enumerate(rows)}
        for fut in as_completed(futs):
            i, success = fut.result()
            if success: ok += 1
            if (ok % 50) == 0:
                print(f'    {ok}/{len(rows)}  t={time.time()-t0:.0f}s')
    print(f'    Done: {ok}/{len(rows)} en {time.time()-t0:.0f}s')


# ============================================================
# 1b. PNOA IGN + ortofotos autonómicas (España)
# ============================================================
# Catálogo de servicios WMS confirmados por GetCapabilities (junio 2026).
# Cada entrada: (ccaa_id, (lon_min, lat_min, lon_max, lat_max), base_url, layer, nota)
# Se usa CRS:84 en todos (lon/lat BBOX, WMS 1.3.0) → estándar, sin ambigüedad de ejes.
_PNOA_CCAA_WMS = [
    ('andalucia',
     (-7.6, 36.0, -1.6, 38.7),
     'https://www.ideandalucia.es/wms/ortofoto_2022',
     'ortofotografia_2022_rgb',
     'REDIAM Andalucía 2022 0,25 m (Sierra Nevada, Cazorla, Grazalema, Sierra Mágina…)'),
    ('aragon',
     (-2.2, 40.0, 0.8, 42.9),
     'https://idearagon.aragon.es/AragonFotos',
     '2024_pnoa',
     'ICEARAGON Aragón 2024 0,25 m (Ordesa y Monte Perdido, Posets-Maladeta…)'),
    ('cataluna',
     (0.2, 40.5, 3.3, 42.9),
     'https://geoserveis.icgc.cat/servei/catalunya/orto-territorial/wms',
     'ortofoto_color_vigent',
     'ICGC Cataluña vigent 0,25 m (Aigüestortes i Estany de Sant Maurici, Pirineos…)'),
    ('cantabria',
     (-5.1, 42.7, -3.1, 43.5),
     'https://geoservicios.cantabria.es/inspire/services/Ortofoto_2023_Aux/MapServer/WMSServer',
     '1',
     'IDE Cantabria 2023 0,25 m (Picos de Europa lado cántabro, Picos de Urrión…)'),
    ('castilla_leon',
     (-6.9, 40.2, -1.7, 43.4),
     'https://idecyl.jcyl.es/geoserver/oi/wms',
     'oi_2020_cyl',
     'IDECyL Castilla y León 2020 0,25 m (Picos de Europa-CyL, Gredos, Guadarrama…)'),
    ('canarias',
     (-18.2, 27.6, -13.4, 29.4),
     'https://idecan1.grafcan.es/ServicioWMS/OrtoExpress',
     'ortoexpress',
     'GRAFCAN Canarias 0,25 m (Teide, Caldera de Taburiente, Garajonay…)'),
]
# Fallback nacional: IGN PNOA Máxima Actualidad (toda España, 0,25–0,5 m, sin restricción)
_PNOA_NACION_URL   = 'https://www.ign.es/wms-inspire/pnoa-ma'
_PNOA_NACION_LAYER = 'OI.OrthoimageCoverage'


def _pnoa_wms_candidates(center_lon, center_lat):
    """Devuelve lista ordenada de (url, layer) para una ubicación dada.
    Primero el servicio autonómico más específico (más reciente), luego el nacional."""
    cands = []
    for _, (lon_min, lat_min, lon_max, lat_max), url, layer, _ in _PNOA_CCAA_WMS:
        if lon_min <= center_lon <= lon_max and lat_min <= center_lat <= lat_max:
            cands.append((url, layer))
    cands.append((_PNOA_NACION_URL, _PNOA_NACION_LAYER))
    return cands


def fetch_pnoa_imgs(rows, out_dir, delta_lat=0.0025, delta_lon=0.0031,
                    size=(500, 400), workers=8):
    """Drop-in de fetch_esri_imgs usando WMS PNOA IGN (0,25 m, España).

    Prioridad por punto: WMS autonómico específico (más reciente/mayor res) →
    PNOA-MA nacional IGN → fallo silencioso (compute_img_features asigna NaN).

    Servicios autonómicos configurados (actualización: junio 2026):
      Andalucía  — REDIAM 2022 0,25 m  (Sierra Nevada, Cazorla, Grazalema…)
      Aragón     — ICEARAGON 2024 0,25 m  (Ordesa, Posets-Maladeta…)
      Cataluña   — ICGC vigent 0,25 m  (Aigüestortes, Pirineos…)
      Cantabria  — IDE Cantabria 2023 0,25 m  (Picos de Europa lado cántabro)
      CyL        — IDECyL 2020 0,25 m  (Picos de Europa-CyL, Gredos, Guadarrama…)
      Canarias   — GRAFCAN 0,25 m  (Teide, Caldera de Taburiente, Garajonay…)
      Resto      — PNOA-MA nacional (toda España, 0,25–0,5 m)
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    W, H = size

    center_lon = sum(r['lon'] for r in rows) / len(rows)
    center_lat = sum(r['lat'] for r in rows) / len(rows)
    candidates = _pnoa_wms_candidates(center_lon, center_lat)

    # Describe qué servicio(s) se usarán
    labels = [note for _, bbox, _, _, note in _PNOA_CCAA_WMS
              if bbox[0] <= center_lon <= bbox[2] and bbox[1] <= center_lat <= bbox[3]]
    if not labels:
        labels = ['PNOA-MA nacional IGN (toda España)']
    print(f'  → Fuente: {" / ".join(labels)}  +  PNOA-MA fallback')
    print(f'  → {len(rows)} imágenes a {out_dir}…')

    def _try_wms(url, layer, lon, lat, dlat, dlon, w, h):
        """Intenta WMS GetMap con dos variantes de CRS; devuelve bytes de imagen o None."""
        bbox_lonlat = f'{lon-dlon},{lat-dlat},{lon+dlon},{lat+dlat}'
        variants = [
            # 1ª: WMS 1.3.0 + CRS:84 (lon/lat BBOX) — la mayoría de servicios modernos
            {'SERVICE':'WMS','VERSION':'1.3.0','REQUEST':'GetMap','LAYERS':layer,
             'STYLES':'','CRS':'CRS:84','BBOX':bbox_lonlat,
             'WIDTH':str(w),'HEIGHT':str(h),'FORMAT':'image/jpeg'},
            # 2ª: WMS 1.1.1 + EPSG:4326 (lon/lat BBOX) — servicios que no aceptan CRS:84
            {'SERVICE':'WMS','VERSION':'1.1.1','REQUEST':'GetMap','LAYERS':layer,
             'STYLES':'','SRS':'EPSG:4326','BBOX':bbox_lonlat,
             'WIDTH':str(w),'HEIGHT':str(h),'FORMAT':'image/jpeg'},
        ]
        for params in variants:
            try:
                resp = requests.get(url, params=params, timeout=20)
                if resp.status_code != 200 or len(resp.content) < 5000:
                    continue
                try:
                    Image.open(io.BytesIO(resp.content)).verify()
                    return resp.content
                except Exception:
                    continue  # respuesta XML de error
            except Exception:
                continue
        return None

    def fetch_one(i, r):
        fname = out_dir / f"pt_{i:04d}.jpg"
        if fname.exists() and fname.stat().st_size > 5000:
            return i, True
        lon, lat = r['lon'], r['lat']
        for url, layer in candidates:
            raw = _try_wms(url, layer, lon, lat, delta_lat, delta_lon, W, H)
            if raw is None:
                continue
            try:
                im = Image.open(io.BytesIO(raw)).convert('RGB')
                dr = ImageDraw.Draw(im)
                cx, cy = W // 2, H // 2
                dr.ellipse([cx-10, cy-10, cx+10, cy+10], outline='white', width=3)
                dr.ellipse([cx-7,  cy-7,  cx+7,  cy+7],  outline='red',   width=2)
                im.save(fname, 'JPEG', quality=80)
                return i, True
            except Exception:
                continue
        return i, False

    ok = 0; t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_one, i, r): i for i, r in enumerate(rows)}
        for fut in as_completed(futs):
            i, success = fut.result()
            if success: ok += 1
            if (ok % 50) == 0 and ok > 0:
                print(f'    {ok}/{len(rows)}  t={time.time()-t0:.0f}s')
    print(f'    Done: {ok}/{len(rows)} en {time.time()-t0:.0f}s')


# ============================================================
# 2. Descriptores imagen (color + textura + patrón)
# ============================================================
def hsv(rgb):
    arr = rgb.astype(np.float32)/255
    R,G,B = arr[...,0],arr[...,1],arr[...,2]
    mx = arr.max(-1); mn = arr.min(-1); df = mx-mn
    H = np.zeros_like(mx); nz = df>1e-6
    rmask = nz & (mx==R); gmask = nz & (mx==G); bmask = nz & (mx==B)
    H[rmask] = ((G[rmask]-B[rmask])/df[rmask])%6
    H[gmask] = ((B[gmask]-R[gmask])/df[gmask])+2
    H[bmask] = ((R[bmask]-G[bmask])/df[bmask])+4
    H = H/6
    with np.errstate(divide='ignore', invalid='ignore'):
        S = np.where(mx>0, df/mx, 0)
    return H, S, mx

def img_features(img_path):
    im = Image.open(img_path).convert('RGB')
    W, Ht = im.size; cx, cy = W//2, Ht//2

    # Central 120x120
    half = 60
    crop = np.array(im.crop((cx-half, cy-half, cx+half, cy+half)))
    H, S, V = hsv(crop)
    R, G, B = crop[...,0].astype(np.float32), crop[...,1].astype(np.float32), crop[...,2].astype(np.float32)
    green = (H>0.18)&(H<0.45)&(S>0.12)&(V>0.18)
    bright_green = green & (S>0.25)&(V>0.30)
    pale = (S<0.12)&(V>0.35)
    exg = (2*G - R - B)

    gray = (0.299*R + 0.587*G + 0.114*B).astype(np.uint8)
    q = (gray.astype(np.int32) * 32 // 256).astype(np.uint8)
    glcm = graycomatrix(q, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=32, symmetric=True, normed=True)

    from scipy.ndimage import sobel, laplace
    sx = sobel(gray.astype(np.float32)/255, axis=0); sy = sobel(gray.astype(np.float32)/255, axis=1)
    mag = np.hypot(sx, sy)
    thr = np.percentile(mag, 75)*1.25

    # Outer ring 300x300
    outer = np.array(im.crop((cx-150, cy-150, cx+150, cy+150)))
    Ho, So, Vo = hsv(outer)
    Ro, Go, Bo = outer[...,0].astype(np.float32), outer[...,1].astype(np.float32), outer[...,2].astype(np.float32)
    exg_o = 2*Go - Ro - Bo
    h, w = outer.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    out_mask = (np.abs(xx-w/2)>60)|(np.abs(yy-h/2)>60)
    surr_rock = ((So<0.20)&(Vo>0.20)&(Vo<0.55)&out_mask).sum()/out_mask.sum()
    surr_pale = ((So<0.15)&(Vo>0.40)&out_mask).sum()/out_mask.sum()
    surr_snow = ((So<0.10)&(Vo>0.75)&out_mask).sum()/out_mask.sum()
    surr_water = ((Ho>0.45)&(Ho<0.65)&(So>0.20)&out_mask).sum()/out_mask.sum()

    # Patrón (180x180)
    half2 = 90
    crop2 = np.array(im.crop((cx-half2, cy-half2, cx+half2, cy+half2)))
    H2, S2, V2 = hsv(crop2)
    gmask2 = (H2>0.18)&(H2<0.45)&(S2>0.18)&(V2>0.20)
    green_total = gmask2.sum()/gmask2.size
    if green_total < 0.02:
        n_components = 0; largest_frac = 0.0; mean_comp_px = 0.0
        granul_5 = 0.0; matorral_score = 0.0; mat_sig = 'ROCK_LIKE'
    else:
        lbl, ncomp = ndi.label(gmask2, structure=np.ones((3,3)))
        sizes = np.bincount(lbl.ravel())[1:]
        n_components = ncomp
        largest_frac = float(sizes.max()/sizes.sum())
        mean_comp_px = float(sizes.mean())
        op5 = opening(gmask2, footprint=disk(5))
        granul_5 = float(op5.sum()/gmask2.sum())
        def sat(x, lo, hi): return float(np.clip((x-lo)/(hi-lo), 0, 1))
        s = (0.30*sat(ncomp, 5, 60) + 0.25*(1-sat(mean_comp_px, 5, 200)) +
             0.20*(1-sat(largest_frac, 0.2, 0.8)) + 0.25*(1-sat(granul_5, 0.2, 0.8)))
        matorral_score = 100.0*s
        if granul_5 >= 0.55 and largest_frac >= 0.40 and green_total >= 0.08:
            mat_sig = 'BORREGUIL_LIKE'
        elif matorral_score >= 55:
            mat_sig = 'MATORRAL_LIKE'
        elif green_total < 0.05:
            mat_sig = 'ROCK_LIKE'
        else:
            mat_sig = 'MIXED'

    return dict(
        frac_bgreen=float(bright_green.sum()/bright_green.size),
        frac_pale=float(pale.sum()/pale.size),
        exg_mean=float(exg.mean()),
        glcm_contrast=float(graycoprops(glcm,'contrast').mean()),
        glcm_homog=float(graycoprops(glcm,'homogeneity').mean()),
        glcm_energy=float(graycoprops(glcm,'energy').mean()),
        edge_density=float((mag>thr).mean()),
        laplacian_var=float(laplace(gray.astype(np.float32)/255).var()),
        vegcontrast=float(exg.mean() - exg_o[out_mask].mean()),
        surr_rock=float(surr_rock),
        surr_pale=float(surr_pale),
        surr_snow=float(surr_snow),
        surr_water=float(surr_water),
        green_total=green_total,
        n_components=n_components,
        largest_frac=largest_frac,
        granul_5=granul_5,
        matorral_score=matorral_score,
        mat_signature=mat_sig,
    )

def compute_img_features(rows, img_dir):
    print(f'  → Calculando features de imagen para {len(rows)} puntos…')
    for i, r in enumerate(rows):
        p = Path(img_dir) / f"pt_{i:04d}.jpg"
        # Las imágenes se nombran por la POSICIÓN del punto al descargarlas. Se anota
        # en el propio punto cuál es la suya: si luego se quitan puntos de la lista,
        # las posiciones se corren y por posición se mostraría la de otro punto.
        r['_img'] = p.name if p.exists() else ''
        if not p.exists():
            for k in ('frac_bgreen','frac_pale','exg_mean','glcm_contrast','glcm_homog',
                      'glcm_energy','edge_density','laplacian_var','vegcontrast',
                      'surr_rock','surr_pale','surr_snow','surr_water',
                      'green_total','largest_frac','granul_5','matorral_score'):
                r[k] = float('nan')
            r['n_components'] = 0; r['mat_signature'] = 'MIXED'
            continue
        try:
            r.update(img_features(p))
        except Exception as e:
            print(f'    error en pt_{i:04d}: {e}')


# ============================================================
# 3. Cauces de OpenStreetMap (ríos y arroyos), para el ambiente «arroyo»
# ============================================================
# Un borreguil es de «arroyo» si está a 30 m o menos de un río o arroyo de
# OpenStreetMap (waterway = river | stream | brook; no las acequias ni los canales, ni
# las fuentes: decidido con Bea el 23-11-2026). Probado en los 372 borreguiles de Sierra
# Nevada: «llano» (pendiente < 7°) se dejaba casi todos los de junto a un arroyo, y los
# cauces deducidos del relieve de 30 m van en escalera y a veces se separan de la
# vaguada.
# Los de Sierra Nevada van con la app (CAUCES_SN: sin depender de internet, y Overpass
# falla a menudo); en otras zonas se descargan de Overpass al analizar.
CAUCES_SN = 'cauces_sierra_nevada.geojson'
TIPOS_CAUCE = ('river', 'stream', 'brook')
SERVIDORES_OVERPASS = ('https://overpass-api.de/api/interpreter',
                       'https://overpass.private.coffee/api/interpreter',
                       'https://overpass.kumi.systems/api/interpreter',
                       'https://maps.mail.ru/osm/tools/overpass/api/interpreter')
MARGEN_CAUCES_GRADOS = 0.05           # ~5 km alrededor de los puntos
_cauces_descargados = {}              # bbox redondeado → líneas (para los puntos añadidos)


def _ruta_capa_cauces(ruta=None):
    return Path(ruta) if ruta else Path(__file__).resolve().parent / CAUCES_SN


def leer_capa_cauces(ruta=None):
    """La capa de cauces que va con la app: (líneas en grados, cobertura (oeste, sur,
    este, norte)), o (None, None) si no está."""
    import json
    from shapely.geometry import shape
    p = _ruta_capa_cauces(ruta)
    if not p.exists():
        return None, None
    with open(p, encoding='utf-8') as f:
        d = json.load(f)
    lineas = [shape(g['geometry']) for g in d.get('features', [])
              if (g.get('properties') or {}).get('waterway') in TIPOS_CAUCE]
    return lineas, tuple(d.get('cobertura') or ())


def _descargar_cauces(o, s, e, n, timeout=45):
    """Ríos y arroyos de Overpass en ese rectángulo; prueba varios servidores (uno solo
    falla a menudo). None si ninguno responde."""
    import requests
    from shapely.geometry import LineString
    tipos = '|'.join(TIPOS_CAUCE)
    q = (f'[out:json][timeout:{int(timeout)}];way["waterway"~"^({tipos})$"]'
         f'({s},{o},{n},{e});out geom;')
    for url in SERVIDORES_OVERPASS:
        try:
            r = requests.post(url, data={'data': q}, timeout=timeout + 15,
                              headers={'User-Agent': 'borreguil-app/1.0'})
            if r.status_code != 200:
                continue
            lineas = []
            for el in r.json().get('elements', []):
                cs = [(p['lon'], p['lat']) for p in el.get('geometry') or []]
                if len(cs) >= 2:
                    lineas.append(LineString(cs))
            return lineas
        except Exception:                                   # noqa: BLE001
            continue
    return None


def cauces_osm(bbox, ruta_capa=None):
    """Los ríos y arroyos de OpenStreetMap alrededor de `bbox` (oeste, sur, este, norte):
    (líneas en grados, de dónde salen: 'capa' o 'overpass'), o (None, 'sin_conexion') si
    no se han podido leer. De la capa de la app si la zona cae dentro; si no, de
    Overpass."""
    from shapely.geometry import box
    m = MARGEN_CAUCES_GRADOS
    o, s, e, n = bbox[0] - m, bbox[1] - m, bbox[2] + m, bbox[3] + m
    lineas, cob = leer_capa_cauces(ruta_capa)
    if lineas is not None and len(cob) == 4 and \
            cob[0] <= bbox[0] and cob[1] <= bbox[1] and cob[2] >= bbox[2] and cob[3] >= bbox[3]:
        zona = box(o, s, e, n)
        return [g for g in lineas if g.intersects(zona)], 'capa'
    clave = tuple(round(v, 3) for v in (o, s, e, n))
    if clave not in _cauces_descargados:
        lineas = _descargar_cauces(*clave)
        if lineas is None:
            return None, 'sin_conexion'
        _cauces_descargados[clave] = lineas
    return _cauces_descargados[clave], 'overpass'


def attach_cauce_distance(rows, lineas):
    """r['dist_cauce_m'] = metros al río o arroyo más cercano. Sin cauces (no se han
    podido leer, o no hay ninguno en la zona), NaN: el ambiente se decide entonces por
    la pendiente. Devuelve cuántos cauces se han usado."""
    if not rows:
        return 0
    if not lineas:
        for r in rows:
            r['dist_cauce_m'] = float('nan')
        return 0
    import pyproj
    from shapely.geometry import Point
    from shapely.ops import transform
    from shapely.strtree import STRtree
    epsg = _utm_epsg_for(rows)
    tr = pyproj.Transformer.from_crs(4326, epsg, always_xy=True)
    utm = [transform(tr.transform, g) for g in lineas]
    arbol = STRtree(utm)
    for r in rows:
        try:
            p = Point(*tr.transform(float(r['lon']), float(r['lat'])))
        except (TypeError, ValueError, KeyError):
            r['dist_cauce_m'] = float('nan')
            continue
        r['dist_cauce_m'] = float(p.distance(utm[int(arbol.nearest(p))]))
    return len(lineas)


# ============================================================
# 4. DEM + topografía (Copernicus GLO-30 vía MPC)
# ============================================================
def fetch_topo(rows, bbox, work_dir):
    print('  → Descargando Copernicus DEM GLO-30 vía MPC…')
    south, west, north, east = bbox[1], bbox[0], bbox[3], bbox[2]
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                           modifier=planetary_computer.sign_inplace)
    items = list(catalog.search(collections=['cop-dem-glo-30'],
                                  bbox=[west, south, east, north]).items())
    if not items:
        print('    Sin DEM disponible'); return
    print(f'    {len(items)} tiles')

    from rasterio.merge import merge as rio_merge
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    srcs = [rasterio.open(it.assets['data'].href) for it in items]
    mosaic, out_trans = rio_merge(srcs, bounds=(west, south, east, north))
    src0 = srcs[0]

    # Pick local UTM
    lat_med = sum(r['lat'] for r in rows)/len(rows)
    lon_med = sum(r['lon'] for r in rows)/len(rows)
    utm_zone = int((lon_med + 180)/6) + 1
    dst_crs = f'EPSG:{32600 + utm_zone if lat_med >= 0 else 32700 + utm_zone}'

    transform_dst, w_dst, h_dst = calculate_default_transform(
        src0.crs, dst_crs, mosaic.shape[2], mosaic.shape[1],
        west, south, east, north, resolution=30.0)
    dem_utm = np.full((h_dst, w_dst), np.nan, dtype=np.float32)
    for i, src in enumerate(srcs):
        reproject(source=mosaic[i] if mosaic.ndim==3 else mosaic,
                  destination=dem_utm,
                  src_transform=out_trans, src_crs=src0.crs,
                  dst_transform=transform_dst, dst_crs=dst_crs,
                  resampling=Resampling.bilinear)
        break  # mosaic is already merged
    dem_utm[dem_utm < -100] = np.nan
    res_x = abs(transform_dst.a); res_y = abs(transform_dst.e)

    # Derivatives
    dzdx = np.gradient(dem_utm, axis=1) / res_x
    dzdy = np.gradient(dem_utm, axis=0) / res_y
    slope_rad = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
    slope_deg = np.rad2deg(slope_rad)
    aspect_rad = np.arctan2(-dzdx, dzdy)
    aspect_n = np.cos(aspect_rad); aspect_e = np.sin(aspect_rad)
    curvature = ndi.laplace(dem_utm) / (res_x**2)
    # (Hasta el 20-11-2026 se calculaba aquí también el TWI, con una acumulación de
    # flujo D8 sobre todo el relieve: lenta, dependía de lo grande que fuera la zona y
    # Earth Engine no lo hace. Los modelos ya no lo usan.)
    h_, w_ = dem_utm.shape

    trans_pt = pyproj.Transformer.from_crs(4326, dst_crs, always_xy=True)
    left = transform_dst.c; top = transform_dst.f
    for r in rows:
        X, Y = trans_pt.transform(r['lon'], r['lat'])
        col = int((X - left)/res_x); row = int((top - Y)/res_y)
        if 0 <= row < h_ and 0 <= col < w_:
            r['elev_dem_m'] = float(dem_utm[row, col])
            r['slope_deg'] = float(slope_deg[row, col])
            r['aspect_north'] = float(aspect_n[row, col])
            r['aspect_east'] = float(aspect_e[row, col])
            r['curvature'] = float(curvature[row, col])
        else:
            for k in ('elev_dem_m','slope_deg','aspect_north','aspect_east','curvature'):
                r[k] = float('nan')


# ============================================================
# 5. Sentinel-2 multi-temporal vía MPC
# ============================================================
SCL_VALID = {4, 5, 6, 7, 11}

# Timeouts y caché para lecturas COG por HTTP. SIN GDAL_HTTP_TIMEOUT una conexión
# estancada cuelga el proceso PARA SIEMPRE (visto en campo: "parado indefinidamente").
# Se aplican vía os.environ (GDAL las lee directamente; más robusto que rasterio.Env,
# que rechaza algunos valores). Los hilos del ThreadPool heredan el entorno del proceso.
_GDAL_HTTP_ENV = dict(
    GDAL_HTTP_TIMEOUT='30', GDAL_HTTP_CONNECTTIMEOUT='10',
    GDAL_HTTP_MAX_RETRY='2', GDAL_HTTP_RETRY_DELAY='2',
    GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',
    GDAL_CACHEMAX='512',          # MB; cache de bloques → reusar tiles entre puntos
    CPL_VSIL_CURL_USE_HEAD='NO',  # menos round-trips HTTP por COG
    VSI_CACHE='TRUE',
)


def _apply_gdal_http_env():
    """Activa timeouts/caché de GDAL para lecturas COG (idempotente)."""
    for k, v in _GDAL_HTTP_ENV.items():
        os.environ.setdefault(k, v)


def _s2_boa_offset(item):
    """Offset BOA_ADD_OFFSET del processing baseline 04.00 de Sentinel-2 (vigente
    desde 2022-01-25). Para baseline ≥ 04.00 la reflectancia se codifica como DN con
    +1000 de offset → reflectancia real = (DN - 1000) / 10000.

    MPC (`sentinel-2-l2a`) sirve el DN CRUDO, a diferencia de GEE
    `COPERNICUS/S2_SR_HARMONIZED`, que ya lo resta. Sin restarlo, el denominador de
    los índices queda inflado y el NDVI/Clre/… se SUBESTIMAN en escenas ≥2022.

    Devuelve 1000.0 si hay que restar el offset, 0.0 si la escena es pre-baseline.
    """
    try:
        bl = item.properties.get('s2:processing_baseline')
        if bl is not None:
            return 1000.0 if float(bl) >= 4.0 else 0.0
    except Exception:
        pass
    # Fallback si falta la propiedad: el baseline 04.00 entró en vigor el 2022-01-25.
    try:
        dt = item.datetime
        if dt is not None:
            return 1000.0 if (dt.year, dt.month, dt.day) >= (2022, 1, 25) else 0.0
    except Exception:
        pass
    return 0.0


def fetch_s2(rows, bbox, years=(2017,2018,2019,2020,2021,2022,2023,2024,2025),
             max_per_period=None, scene_workers=6, progress=None):
    """progress: callback opcional progress(frac 0..1, mensaje) para una barra en la UI."""
    import warnings
    # 'All-NaN slice' / 'Mean of empty slice' son ESPERADOS (puntos sin cobertura en un
    # periodo) y benignos: nanmedian/nanmean devuelven NaN para esa columna. Se silencian.
    warnings.filterwarnings('ignore', message='.*All-NaN slice.*', category=RuntimeWarning)
    warnings.filterwarnings('ignore', message='.*Mean of empty slice.*', category=RuntimeWarning)
    warnings.filterwarnings('ignore', message='.*Degrees of freedom.*', category=RuntimeWarning)
    _apply_gdal_http_env()

    def _emit(frac, msg):
        if progress:
            try: progress(max(0.0, min(1.0, frac)), msg)
            except Exception: pass

    print('  → Descargando Sentinel-2 L2A multi-temporal vía MPC…')
    # Cap adaptativo: la mediana de NDVI de verano es estable, así que 5-8 escenas/periodo
    # bastan (el modelo se entrenó con ~4). Menos escenas = MENOS aperturas COG = más
    # rápido, sin pérdida real de calidad. (Antes hasta 12 → el doble de lento.)
    if max_per_period is None:
        n_years = max(1, len(years))
        max_per_period = max(4, min(8, 24 // n_years))
    n_periods = 2 * len(years)
    print(f'    Años: {min(years)}–{max(years)} ({len(years)} años) · '
          f'máx {max_per_period} escenas/periodo · {len(rows)} puntos · '
          f'{scene_workers} escenas en paralelo')
    if len(rows) >= 500 and n_periods * max_per_period > 40:
        print(f'    AVISO: {len(rows)} puntos × ~{n_periods * max_per_period} escenas '
              'puede tardar bastante; con muchos puntos conviene un rango de años '
              'más corto (p. ej. 2021-2025).')
    south, west, north, east = bbox[1], bbox[0], bbox[3], bbox[2]
    # timeout en el STAC: sin él, una búsqueda estancada también cuelga indefinidamente
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                           modifier=planetary_computer.sign_inplace,
                           timeout=30)

    lat_med = sum(r['lat'] for r in rows)/len(rows)
    lon_med = sum(r['lon'] for r in rows)/len(rows)
    utm_zone = int((lon_med + 180)/6) + 1
    epsg_pts = 32600 + utm_zone if lat_med >= 0 else 32700 + utm_zone
    trans_pts = pyproj.Transformer.from_crs(4326, epsg_pts, always_xy=True)
    pts_xy = [trans_pts.transform(r['lon'], r['lat']) for r in rows]

    INDS = ['ndvi','ndwi','clre','gndvi','ndmi','evi','nbr']
    # Índices NUEVOS (Indices_Mascaras_S2.txt) con estadísticos sobre el período:
    NEW_INDS = ['wavi','albedo','ndmi','ndwi']      # se guardan como {k}_{mean,min,max,sd}
    SCL_NOSNOW = {4, 5, 6, 7}                        # excluye nubes(3,8,9,10) y nieve(11)
    _ALB = (0.1836, 0.1759, 0.1456, 0.1347, 0.1233, 0.1134, 0.1001, 0.0231, 0.0003)

    def sample(item):
        """Muestrea una escena S2 para TODOS los puntos. Lee UN bloque del bbox de los
        puntos por banda (no una ventana por punto) y muestrea en memoria → ~N× menos
        peticiones HTTP; el tiempo deja de crecer con el nº de puntos. Si hay dispersión,
        trocea en celdas de CELL px para acotar la memoria. Valores idénticos al muestreo
        por punto (mismas ventanas 5×5 a 10 m / 3×3 a 20 m y mismas máscaras)."""
        from rasterio.windows import Window
        try:
            a = item.assets
            srcs = {b: rasterio.open(a[b].href)
                    for b in ('B02','B03','B04','B05','B06','B07','B08','B11','B12','SCL')}
        except Exception:
            return None, None
        out = {k: np.full(len(rows), np.nan, dtype=np.float32) for k in INDS}
        out_new = {k: np.full(len(rows), np.nan, dtype=np.float32) for k in NEW_INDS}
        off = _s2_boa_offset(item)          # BOA_ADD_OFFSET (baseline ≥04.00) o 0
        try:
            b08 = srcs['B08']
            if b08.crs.to_epsg() != epsg_pts:
                t = pyproj.Transformer.from_crs(epsg_pts, b08.crs.to_epsg(), always_xy=True)
                tpts = [t.transform(x, y) for x, y in pts_xy]
            else:
                tpts = pts_xy
            H, W = b08.height, b08.width
            rc = [b08.index(X, Y) for X, Y in tpts]
            valid = [i for i, (r, c) in enumerate(rc) if 3 <= r < H-3 and 3 <= c < W-3]
            if not valid:
                return out, out_new

            CELL = 2048                                   # px (~20 km a 10 m) por bloque
            cells = {}
            for i in valid:
                r, c = rc[i]
                cells.setdefault((r // CELL, c // CELL), []).append(i)
            B10 = ('B02','B03','B04','B08'); B20 = ('B05','B06','B07','B11','B12','SCL')

            for idxs in cells.values():
                rs = [rc[i][0] for i in idxs]; cs = [rc[i][1] for i in idxs]
                r0, r1, c0, c1 = min(rs)-2, max(rs)+3, min(cs)-2, max(cs)+3
                r0b, c0b = max(0, r0//2 - 1), max(0, c0//2 - 1)
                r1b, c1b = (r1+1)//2 + 1, (c1+1)//2 + 1
                try:
                    blk10 = {b: srcs[b].read(1, window=Window(c0, r0, c1-c0, r1-r0)
                                             ).astype(np.float32) for b in B10}
                    blk20 = {b: srcs[b].read(1, window=Window(c0b, r0b, c1b-c0b, r1b-r0b)
                                             ).astype(np.float32) for b in B20}
                except Exception:
                    continue
                # Corrección del offset BOA del baseline ≥04.00 (ver _s2_boa_offset):
                # se resta SOLO a las bandas ópticas (NUNCA a SCL) y se clampa a ≥0,
                # de modo que la máscara de validez (B>0, más abajo) siga detectando
                # nodata (DN=0) y descarte reflectancias físicamente imposibles (<0).
                # Las escenas pre-baseline (off=0) quedan exactamente igual que antes.
                if off:
                    for b in B10:
                        ar = blk10[b]; ar -= off; np.clip(ar, 0, None, out=ar)
                    for b in ('B05', 'B06', 'B07', 'B11', 'B12'):
                        ar = blk20[b]; ar -= off; np.clip(ar, 0, None, out=ar)
                h20, w20 = blk20['SCL'].shape
                for i in idxs:
                    r, c = rc[i]
                    lr, lc = r - r0, c - c0
                    r20, c20 = r//2 - r0b, c//2 - c0b
                    if r20 < 1 or c20 < 1 or r20 >= h20-1 or c20 >= w20-1:
                        continue
                    scl = blk20['SCL'][r20-1:r20+2, c20-1:c20+2]
                    if np.isin(scl.ravel(), list(SCL_VALID)).sum() < 0.5*scl.size: continue
                    B02 = blk10['B02'][lr-2:lr+3, lc-2:lc+3]
                    B03 = blk10['B03'][lr-2:lr+3, lc-2:lc+3]
                    B04 = blk10['B04'][lr-2:lr+3, lc-2:lc+3]
                    B08v = blk10['B08'][lr-2:lr+3, lc-2:lc+3]
                    v10 = (B02>0)&(B03>0)&(B04>0)&(B08v>0)
                    if v10.sum() < 8: continue
                    B05 = blk20['B05'][r20-1:r20+2, c20-1:c20+2]
                    B11 = blk20['B11'][r20-1:r20+2, c20-1:c20+2]
                    B12 = blk20['B12'][r20-1:r20+2, c20-1:c20+2]
                    v20 = (B05>0)&(B11>0)&(B12>0)
                    if v20.sum() < 3: continue
                    b2, b3, b4, b8 = B02[v10].mean(), B03[v10].mean(), B04[v10].mean(), B08v[v10].mean()
                    b5, b11, b12 = B05[v20].mean(), B11[v20].mean(), B12[v20].mean()
                    # --- Índices EXISTENTES (máscara SCL_VALID con nieve) ---
                    out['ndvi'][i] = (b8-b4)/(b8+b4+1e-6)
                    out['ndwi'][i] = (b3-b8)/(b3+b8+1e-6)
                    out['clre'][i] = b8/(b5+1e-6) - 1
                    out['gndvi'][i]= (b8-b3)/(b8+b3+1e-6)
                    out['ndmi'][i] = (b8-b11)/(b8+b11+1e-6)
                    # EVI normal: las bandas van aquí en números digitales (0–10 000),
                    # así que su «+1» de reflectancia es +10 000. (Hasta el 19-11-2026
                    # se sumaba 1, casi nada, y el EVI salía unas 3 veces mayor.)
                    out['evi'][i]  = 2.5*(b8-b4)/(b8+6*b4-7.5*b2+10000+1e-6)
                    out['nbr'][i]  = (b8-b12)/(b8+b12+1e-6)
                    # --- Índices NUEVOS (máscara extra: sin nieve, NDSI<0.4) ---
                    nosnow = np.isin(scl.ravel(), list(SCL_NOSNOW)).sum() >= 0.5*scl.size
                    ndsi = (b3 - b11) / (b3 + b11 + 1e-6)
                    if nosnow and ndsi < 0.4:
                        B06 = blk20['B06'][r20-1:r20+2, c20-1:c20+2]
                        B07 = blk20['B07'][r20-1:r20+2, c20-1:c20+2]
                        b6 = B06[B06>0].mean() if (B06>0).any() else np.nan
                        b7 = B07[B07>0].mean() if (B07>0).any() else np.nan
                        r2,r3,r4,r5,r6,r7,r8,r11,r12 = (b2/1e4,b3/1e4,b4/1e4,b5/1e4,b6/1e4,
                                                        b7/1e4,b8/1e4,b11/1e4,b12/1e4)
                        out_new['ndmi'][i] = (r8-r11)/(r8+r11+1e-6)
                        out_new['ndwi'][i] = (r3-r8)/(r3+r8+1e-6)
                        L = 0.5
                        out_new['wavi'][i] = (1+L)*(r8-r2)/(r8+r2+L+1e-6)
                        if not (np.isnan(r6) or np.isnan(r7)):
                            alb = (_ALB[0]*r2+_ALB[1]*r3+_ALB[2]*r4+_ALB[3]*r5+_ALB[4]*r6+
                                   _ALB[5]*r7+_ALB[6]*r8+_ALB[7]*r11+_ALB[8]*r12)
                            out_new['albedo'][i] = float(np.clip(alb, 0.0, 1.0))
        finally:
            for s in srcs.values():
                try: s.close()
                except Exception: pass
        return out, out_new

    # Search by period (índices antiguos: medianas early/late; índices nuevos:
    # mean/min/max/sd acumulando TODAS las escenas válidas del período de estudio)
    period_results = {}
    new_stack = {k: [] for k in NEW_INDS}
    n_periods = 2 * len(years)
    period_i = 0
    for yr in years:
        for label, start, end in [
            (f'early_{yr}', f'{yr}-06-15', f'{yr}-07-25'),
            (f'late_{yr}',  f'{yr}-08-01', f'{yr}-09-30'),
        ]:
            t0p = time.time()
            _emit(period_i / n_periods, f'Sentinel-2: periodo {label} '
                  f'({period_i+1}/{n_periods})…')
            print(f'    Periodo {label}: buscando escenas…')
            items = []
            for _att in range(5):
                try:
                    items = list(catalog.search(collections=['sentinel-2-l2a'],
                                                 bbox=[west, south, east, north],
                                                 datetime=f'{start}/{end}',
                                                 query={'eo:cloud_cover':{'lt':30}},
                                                 max_items=40).items())
                    break
                except Exception as _e:
                    print(f'      STAC reintento {_att+1}/5 ({_e})')
                    time.sleep(5)

            from collections import deque as _deque
            import itertools as _itertools
            stacks = {k: [] for k in INDS}; n_ok = 0; tried = 0
            it_iter = iter(items)
            pending = _deque()
            with ThreadPoolExecutor(max_workers=scene_workers) as ex:
                for it in _itertools.islice(it_iter, scene_workers):
                    pending.append(ex.submit(sample, it))
                while pending:
                    res, res_new = pending.popleft().result()
                    tried += 1
                    if res is not None and np.sum(~np.isnan(res['ndvi'])) > 0:
                        n_ok += 1
                        for k in INDS:
                            stacks[k].append(res[k])
                        for k in NEW_INDS:
                            new_stack[k].append(res_new[k])
                        print(f'      escena válida {min(n_ok, max_per_period)}/'
                              f'{max_per_period} '
                              f'({int(np.sum(~np.isnan(res["ndvi"])))}/{len(rows)} pts · '
                              f'{time.time()-t0p:.0f}s)')
                    if n_ok >= max_per_period:
                        break                          # objetivo alcanzado: no procesar más
                    nxt = next(it_iter, None)
                    if nxt is not None:
                        pending.append(ex.submit(sample, nxt))
            print(f'    Periodo {label}: {n_ok} escenas válidas de {tried} probadas '
                  f'({time.time()-t0p:.0f}s)')
            period_results[label] = {k: np.nanmedian(np.stack(stacks[k]), axis=0)
                                      if stacks[k] else np.full(len(rows), np.nan)
                                      for k in INDS}
            period_i += 1
            _emit(period_i / n_periods, f'Sentinel-2: {period_i}/{n_periods} periodos')

    early_keys = [k for k in period_results if k.startswith('early_')]
    late_keys  = [k for k in period_results if k.startswith('late_')]

    early = {k: np.nanmedian(np.stack([period_results[lab][k] for lab in early_keys]), axis=0)
             for k in INDS} if early_keys else None
    late  = {k: np.nanmedian(np.stack([period_results[lab][k] for lab in late_keys]), axis=0)
             for k in INDS} if late_keys else None

    # Estadísticos mean/min/max/sd de los índices nuevos sobre todo el período
    new_stats = {}
    for k in NEW_INDS:
        if new_stack[k]:
            arr = np.stack(new_stack[k])  # (n_escenas, n_puntos)
            with np.errstate(all='ignore'):
                new_stats[(k,'mean')] = np.nanmean(arr, axis=0)
                new_stats[(k,'min')]  = np.nanmin(arr, axis=0)
                new_stats[(k,'max')]  = np.nanmax(arr, axis=0)
                new_stats[(k,'sd')]   = np.nanstd(arr, axis=0)
        else:
            for st in ('mean','min','max','sd'):
                new_stats[(k,st)] = np.full(len(rows), np.nan)

    for i, r in enumerate(rows):
        if early:
            for k in INDS:
                r[f'{k}_early'] = float(early[k][i]) if not np.isnan(early[k][i]) else float('nan')
        if late:
            for k in INDS:
                r[f'{k}_late'] = float(late[k][i]) if not np.isnan(late[k][i]) else float('nan')
        if early and late:
            r['ndvi_drop'] = (r.get('ndvi_early', float('nan')) -
                              r.get('ndvi_late', float('nan')))
        for k in NEW_INDS:
            for st in ('mean','min','max','sd'):
                v = new_stats[(k,st)][i]
                r[f'{k}_{st}'] = float(v) if not np.isnan(v) else float('nan')


# ---- Sentinel-1 (radar) ------------------------------------------------------
# Backscatter de verano (jun–sep): una dimensión de humedad y estructura que el
# óptico no capta. Desde el 19-11-2026 se lee SOLO de Earth Engine
# (gee_backend.fetch_s1_gee), en segundos, y los modelos de partida se reentrenaron
# con esos valores (classification_v7.csv). Antes se leía de Planetary Computer
# (sentinel-1-rtc): más de media hora con 20 puntos, y es otro producto. Con el backend
# Planetary Computer no se lee, y se usa la versión del modelo entrenada sin radar
# (carpeta sin_radar/, ver retrain_v7_s1gee.py --sin-radar): el modelo con radar, si
# le falta, acierta bastante menos (AUC 0,793 frente a 0,833 la versión sin radar).
S1_FEATS = ['s1_vv_mean', 's1_vh_mean', 's1_vv_sd', 's1_vh_sd', 's1_ratio_mean']
CARPETA_SIN_RADAR = 'sin_radar'


def hay_radar(rows):
    """¿Se ha leído el radar en algún punto?"""
    for r in rows or []:
        try:
            v = float(r.get('s1_vv_mean'))
        except (TypeError, ValueError):
            continue
        if v == v:
            return True
    return False


def ruta_sin_radar(model_path):
    """Versión sin radar de un modelo que viene con la app (sin_radar/<nombre>_sin_radar
    .joblib junto a él), o None si no la tiene."""
    if not model_path:
        return None
    p = Path(model_path)
    q = p.parent / CARPETA_SIN_RADAR / p.name.replace('.joblib', '_sin_radar.joblib')
    return q if q.exists() else None


# ---- PlanetScope 3 m (8 bandas SR, Orders API) --------------------------------
# Muy alta resolución comercial: requiere API key de Planet con permiso de descarga
# (variable de entorno PL_API_KEY o st.secrets; NUNCA hardcodeada). Consume cuota
# por km² pedido → el AOI se construye como rejilla de celdas que contienen puntos
# (no la bbox completa) y se recorta con la herramienta `clip` de la Orders API.
# Bundle analytic_8b_sr_udm2: 8 bandas surface reflectance + máscara UDM2.
# Bandas PSB.SD: 1=coastal_blue 2=blue 3=green_i 4=green 5=yellow 6=red 7=rededge 8=nir
PS_FEATS = ['ps_ndvi_mean', 'ps_ndvi_sd', 'ps_ndvi_p10',
            'ps_ndre_mean', 'ps_ndre_sd', 'ps_gndvi_mean', 'ps_gndvi_sd',
            'ps_ndwi_mean', 'ps_vegfrac', 'ps_ndvi_tex']

_PLANET_API = 'https://api.planet.com'


def _planet_aoi(rows, cell_km=2.0):
    """AOI compacto: unión de celdas de rejilla (cell_km) que contienen puntos.
    Devuelve (geojson_geom, area_km2). Minimiza la cuota frente a usar la bbox."""
    import math
    from shapely.geometry import box as shp_box, mapping
    from shapely.ops import unary_union
    lat0 = sum(r['lat'] for r in rows) / len(rows)
    dlon = cell_km / (111.32 * math.cos(math.radians(lat0)))
    dlat = cell_km / 110.57
    cells = {(int(math.floor(r['lon']/dlon)), int(math.floor(r['lat']/dlat)))
             for r in rows}
    geoms = [shp_box(cx*dlon, cy*dlat, (cx+1)*dlon, (cy+1)*dlat) for cx, cy in cells]
    try:
        union = unary_union(geoms)
    except TypeError:
        # shapely<2.0.7 + numpy 2.x: bug 'create_collection' → unión binaria en cadena
        from functools import reduce
        union = reduce(lambda a, b: a.union(b), geoms)
    area_km2 = len(cells) * cell_km * cell_km
    return mapping(union), area_km2


def _planet_pick_days(auth, aoi_geom, rows, year, windows, max_days_per_window=1):
    """quick-search por ventana temporal; agrupa por día y elige el día con mayor
    cobertura de puntos (y menos nubes). Devuelve lista de (day, [item_ids], n_cov)."""
    from shapely.geometry import shape as shp_shape, Point as ShpPoint
    pts = [ShpPoint(r['lon'], r['lat']) for r in rows]
    chosen = []
    for w_start, w_end in windows:
        search = {
            'item_types': ['PSScene'],
            'filter': {'type': 'AndFilter', 'config': [
                {'type': 'GeometryFilter', 'field_name': 'geometry', 'config': aoi_geom},
                {'type': 'DateRangeFilter', 'field_name': 'acquired',
                 'config': {'gte': f'{year}-{w_start}T00:00:00Z',
                            'lte': f'{year}-{w_end}T23:59:59Z'}},
                {'type': 'RangeFilter', 'field_name': 'cloud_cover', 'config': {'lte': 0.15}},
                {'type': 'AssetFilter', 'config': ['ortho_analytic_8b_sr']},
            ]}
        }
        feats = []
        url = f'{_PLANET_API}/data/v1/quick-search'
        body = search
        while url:
            rq = requests.post(url, auth=auth, json=body, timeout=30) if body else \
                 requests.get(url, auth=auth, timeout=30)
            rq.raise_for_status()
            j = rq.json()
            feats.extend(j.get('features', []))
            url = j.get('_links', {}).get('_next'); body = None
            if len(feats) > 400: break
        by_day = {}
        for f in feats:
            day = f['properties']['acquired'][:10]
            by_day.setdefault(day, []).append(f)
        best = None
        for day, fs in by_day.items():
            try:
                foot = [shp_shape(f['geometry']) for f in fs]
                cov = sum(1 for p in pts if any(g.contains(p) for g in foot))
            except Exception:
                cov = 0
            cloud = sum(f['properties'].get('cloud_cover', 0) for f in fs) / len(fs)
            key = (cov, -cloud)
            if best is None or key > best[0]:
                best = (key, day, [f['id'] for f in fs], cov)
        if best:
            _, day, ids, cov = best
            chosen.append((day, ids, cov))
            print(f'    ventana {year}-{w_start}→{w_end}: día {day}, '
                  f'{len(ids)} escenas, cubre {cov}/{len(rows)} pts')
        else:
            print(f'    ventana {year}-{w_start}→{w_end}: sin escenas válidas')
    return chosen


def _planet_order_and_download(auth, item_ids, aoi_geom, out_dir, order_name,
                               poll_s=30, timeout_s=3600):
    """POST orden con clip → poll hasta success/partial → descarga los GeoTIFF.
    Devuelve lista de paths descargados. Reanuda si los ficheros ya existen."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    order = {
        'name': order_name,
        'products': [{'item_ids': sorted(set(item_ids)), 'item_type': 'PSScene',
                      'product_bundle': 'analytic_8b_sr_udm2'}],
        'tools': [{'clip': {'aoi': aoi_geom}}],
    }
    ro = requests.post(f'{_PLANET_API}/compute/ops/orders/v2', auth=auth,
                       json=order, timeout=60)
    if ro.status_code not in (200, 201, 202):
        raise RuntimeError(f'Orden rechazada (HTTP {ro.status_code}): {ro.text[:400]}')
    oid = ro.json()['id']
    print(f'    Orden {oid} creada ({len(set(item_ids))} escenas). Esperando…')
    t0 = time.time(); state = ''
    while time.time() - t0 < timeout_s:
        rs = requests.get(f'{_PLANET_API}/compute/ops/orders/v2/{oid}', auth=auth,
                          timeout=30)
        state = rs.json().get('state', '?')
        if state in ('success', 'partial'):
            break
        if state == 'failed':
            raise RuntimeError(f'Orden {oid} FAILED: '
                               f'{rs.json().get("last_message","?")[:300]}')
        print(f'      [{time.time()-t0:5.0f}s] estado: {state}')
        time.sleep(poll_s)
    if state not in ('success', 'partial'):
        raise RuntimeError(f'Orden {oid} no completada en {timeout_s}s (estado {state})')
    results = rs.json().get('_links', {}).get('results', [])
    paths = []
    for res in results:
        name = res['name'].split('/')[-1]
        if not name.endswith('.tif'):
            continue
        dest = out_dir / name
        if dest.exists() and dest.stat().st_size > 0:
            paths.append(dest); continue
        try:
            with requests.get(res['location'], stream=True, timeout=300) as rr:
                rr.raise_for_status()
                with open(dest, 'wb') as fh:
                    for chunk in rr.iter_content(1 << 20):
                        fh.write(chunk)
            paths.append(dest)
        except Exception as e:
            print(f'      ⚠ fallo descargando {name}: {e}')
    print(f'    Descargados {len(paths)} GeoTIFF a {out_dir}')
    return paths


def _planet_sample(rows, tif_paths, win_px=11):
    """Muestrea ventana win_px×win_px (3 m/px) por punto en cada escena SR clip,
    enmascara con UDM2 (banda 1 = clear) y agrega PS_FEATS por punto."""
    from rasterio.windows import Window
    srs = {}; udms = {}
    for p in tif_paths:
        n = p.name
        iid = n.split('_3B_')[0]
        if 'udm2' in n: udms[iid] = p
        elif 'AnalyticMS_SR' in n: srs[iid] = p
    half = win_px // 2
    daily = [{} for _ in rows]   # por punto: {day: reading_dict}

    for iid, sr_path in sorted(srs.items()):
        day = f'{iid[0:4]}-{iid[4:6]}-{iid[6:8]}'
        udm_path = udms.get(iid)
        try:
            ds = rasterio.open(sr_path)
            du = rasterio.open(udm_path) if udm_path else None
        except Exception:
            continue
        try:
            t = pyproj.Transformer.from_crs(4326, ds.crs, always_xy=True)
            for i, r in enumerate(rows):
                X, Y = t.transform(r['lon'], r['lat'])
                row_f, col_f = ds.index(X, Y)
                if (row_f < half or col_f < half or
                        row_f >= ds.height - half or col_f >= ds.width - half):
                    continue
                w = Window(col_f - half, row_f - half, win_px, win_px)
                try:
                    bands = ds.read([4, 6, 7, 8], window=w).astype(np.float64)
                except Exception:
                    continue
                green, red, rededge, nir = bands
                valid = (green + red + nir) > 0
                if du is not None:
                    try:
                        clear = du.read(1, window=w)
                        valid &= (clear == 1)
                    except Exception:
                        pass
                if valid.sum() < 0.35 * win_px * win_px:
                    continue
                g, rd, re_, nr = (b[valid] for b in (green, red, rededge, nir))
                with np.errstate(divide='ignore', invalid='ignore'):
                    ndvi  = (nr - rd) / (nr + rd)
                    ndre  = (nr - re_) / (nr + re_)
                    gndvi = (nr - g) / (nr + g)
                    ndwi  = (g - nr) / (g + nr)
                reading = dict(
                    n_valid=int(valid.sum()),
                    ndvi=float(np.nanmedian(ndvi)), ndre=float(np.nanmedian(ndre)),
                    gndvi=float(np.nanmedian(gndvi)), ndwi=float(np.nanmedian(ndwi)),
                    vegfrac=float(np.mean(ndvi > 0.4)),
                    tex=float(np.nanstd(ndvi)),
                )
                prev = daily[i].get(day)
                if prev is None or reading['n_valid'] > prev['n_valid']:
                    daily[i][day] = reading
        finally:
            ds.close()
            if du is not None: du.close()

    n_ok = 0
    for i, r in enumerate(rows):
        obs = list(daily[i].values())
        if not obs:
            for k in PS_FEATS: r[k] = float('nan')
            continue
        n_ok += 1
        ndvis  = np.array([o['ndvi'] for o in obs])
        ndres  = np.array([o['ndre'] for o in obs])
        gndvis = np.array([o['gndvi'] for o in obs])
        r['ps_ndvi_mean']  = float(ndvis.mean())
        r['ps_ndvi_sd']    = float(ndvis.std()) if len(obs) >= 2 else float('nan')
        r['ps_ndvi_p10']   = float(np.percentile(ndvis, 10))
        r['ps_ndre_mean']  = float(ndres.mean())
        r['ps_ndre_sd']    = float(ndres.std()) if len(obs) >= 2 else float('nan')
        r['ps_gndvi_mean'] = float(gndvis.mean())
        r['ps_gndvi_sd']   = float(gndvis.std()) if len(obs) >= 2 else float('nan')
        r['ps_ndwi_mean']  = float(np.mean([o['ndwi'] for o in obs]))
        r['ps_vegfrac']    = float(np.mean([o['vegfrac'] for o in obs]))
        r['ps_ndvi_tex']   = float(np.mean([o['tex'] for o in obs]))
    print(f'    Puntos con dato Planet: {n_ok}/{len(rows)}')
    return n_ok


def fetch_planet(rows, api_key=None, year=2023, windows=None, cell_km=2.0,
                 work_dir=None, dry_run=False, order_name=None):
    """Rellena PS_FEATS por punto desde PlanetScope 8b SR (Orders API + clip).

    api_key: clave Planet con permiso de descarga; por defecto env PL_API_KEY.
    windows: lista de (mm-dd, mm-dd) dentro de `year`; por defecto 3 ventanas
             de verano (jun-jul / ago / sep) → mean/sd/p10 multi-fecha.
    dry_run: solo busca escenas y estima la cuota (km²); no pide ni descarga.
    Falla con elegancia: sin key o sin cobertura → PS_FEATS quedan NaN."""
    key = api_key or os.environ.get('PL_API_KEY', '')
    if not key:
        print('  ⚠ Sin PL_API_KEY: se omite PlanetScope (features NaN)')
        for r in rows:
            for k in PS_FEATS: r[k] = float('nan')
        return 0
    auth = (key, '')
    if windows is None:
        windows = [('06-20', '07-10'), ('07-25', '08-15'), ('09-05', '09-30')]
    print(f'  → PlanetScope 8b SR · {year} · {len(windows)} ventanas')
    aoi_geom, aoi_km2 = _planet_aoi(rows, cell_km=cell_km)
    print(f'    AOI rejilla {cell_km} km: {aoi_km2:.0f} km²  '
          f'(≈{aoi_km2*len(windows):.0f} km² de cuota para {len(windows)} fechas)')
    chosen = _planet_pick_days(auth, aoi_geom, rows, year, windows)
    if not chosen:
        print('  ⚠ Sin escenas en ninguna ventana; PS_FEATS → NaN')
        for r in rows:
            for k in PS_FEATS: r[k] = float('nan')
        return 0
    all_ids = [iid for _, ids, _ in chosen for iid in ids]
    if dry_run:
        print(f'    DRY-RUN: {len(all_ids)} escenas en {len(chosen)} fechas. '
              'No se realiza el pedido.')
        return 0
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix='planet_'))
    if order_name is None:
        order_name = f'borreguiles_{year}_{len(all_ids)}sc'
    paths = _planet_order_and_download(auth, all_ids, aoi_geom, work_dir, order_name)
    if not paths:
        for r in rows:
            for k in PS_FEATS: r[k] = float('nan')
        return 0
    return _planet_sample(rows, paths)


# ---- PNOA Falso Color Infrarrojo 0,25 m (CIR, NIR-R-G) ------------------------
# Ortofoto infrarroja autonómica (gratis, España). Banda NIR a 25 cm → pseudo-NDVI
# que capta el borde abrupto de la mancha de borreguil que S2 (10 m) promedia. Es
# la mejor mejora individual en Sierra Nevada (+0.046 AUC honesto), gratis. SOLO
# donde hay servicio WMS con capa IR (de momento Andalucía y Cataluña confirmadas).
CIR_FEATS = ['cir_ndvi_mean', 'cir_ndvi_p90', 'cir_ndvi_sd',
             'cir_vegfrac', 'cir_nir_mean', 'cir_ndvi_contrast']

# (ccaa, (lon_min,lat_min,lon_max,lat_max), wms_url, layer_ir, epsg_utm, fmt)
# fmt: 'tiff' (rasterio, bandas crudas NIR-R-G) o 'jpeg' (PIL, composición renderizada).
# En falso color, banda/canal 1 = NIR, 2 = R, 3 = G → pseudo-NDVI = (b1-b2)/(b1+b2).
_CIR_CCAA_WMS = [
    ('andalucia', (-7.6, 36.0, -1.6, 38.7),
     'https://www.ideandalucia.es/wms/ortofoto_2022',
     'ortofotografia_2022_infrarrojo', 25830, 'tiff'),
    ('cataluna', (0.2, 40.5, 3.3, 42.9),
     'https://geoserveis.icgc.cat/servei/catalunya/orto-territorial/wms',
     'ortofoto_infraroig_vigent', 25831, 'tiff'),
    ('canarias', (-18.3, 27.6, -13.3, 29.5),
     'https://idecan1.grafcan.es/ServicioWMS/OrtofotoCIR',
     'ortocir_3000', 32628, 'jpeg'),
]


def _cir_service_for(lon, lat):
    for _, (lo0, la0, lo1, la1), url, layer, epsg, fmt in _CIR_CCAA_WMS:
        if lo0 <= lon <= lo1 and la0 <= lat <= la1:
            return url, layer, epsg, fmt
    return None


def _cir_one(lon, lat, url, layer, epsg, fmt='tiff', half=25.0, px=200):
    """GetMap CIR (NIR-R-G) 50×50 m a 0,25 m → 6 features de pseudo-NDVI.
    fmt 'tiff' → rasterio (bandas crudas); 'jpeg' → PIL (composición renderizada).
    Devuelve dict de CIR_FEATS o NaN si no hay dato/cobertura."""
    tr = pyproj.Transformer.from_crs(4326, epsg, always_xy=True)
    x, y = tr.transform(float(lon), float(lat))
    mime = 'image/jpeg' if fmt == 'jpeg' else 'image/tiff'
    params = {'service': 'WMS', 'version': '1.3.0', 'request': 'GetMap',
              'layers': layer, 'styles': '', 'crs': f'EPSG:{epsg}',
              'bbox': f'{x-half},{y-half},{x+half},{y+half}',
              'width': px, 'height': px, 'format': mime}
    nan = {k: float('nan') for k in CIR_FEATS}
    content = None
    for _att in range(3):
        try:
            r = requests.get(url, params=params, timeout=40)
            ct = r.headers.get('content-type', '')
            ok = (r.status_code == 200 and 'image' in ct and 'xml' not in ct
                  and len(r.content) > 1500)
            if ok:
                content = r.content; break
        except Exception:
            pass
        time.sleep(2)
    if content is None:
        return nan
    try:
        if fmt == 'jpeg':
            arr = np.asarray(Image.open(io.BytesIO(content)).convert('RGB'),
                             dtype=np.float64)           # (h, w, 3): NIR, R, G renderizados
            a = np.transpose(arr, (2, 0, 1))             # → (3, h, w)
        else:
            from rasterio.io import MemoryFile
            with MemoryFile(content) as mf, mf.open() as ds:
                a = ds.read().astype(np.float64)          # (bandas, h, w): NIR, R, G
    except Exception:
        return nan
    if a.shape[0] < 2 or np.nanstd(a[0]) < 1e-6:          # tile en blanco / fuera vuelo
        return nan
    nir, red = a[0], a[1]
    with np.errstate(all='ignore'):
        ndvi = (nir - red) / (nir + red + 1e-6)
    h, w = ndvi.shape
    cy, cx = h // 2, w // 2
    c = max(1, h // 4)                                     # centro 25 m
    cen = ndvi[cy-c:cy+c, cx-c:cx+c]
    nirc = nir[cy-c:cy+c, cx-c:cx+c]
    ring = np.ones_like(ndvi, bool); ring[cy-c:cy+c, cx-c:cx+c] = False
    return {
        'cir_ndvi_mean': float(np.nanmedian(cen)),
        'cir_ndvi_p90': float(np.nanpercentile(cen, 90)),
        'cir_ndvi_sd': float(np.nanstd(cen)),
        'cir_vegfrac': float(np.nanmean(cen > 0.18)),
        'cir_nir_mean': float(np.nanmedian(nirc)),
        'cir_ndvi_contrast': float(np.nanmedian(cen) - np.nanmedian(ndvi[ring])),
    }


def fetch_cir(rows, bbox=None, workers=6):
    """Rellena CIR_FEATS por punto desde la ortofoto infrarroja autonómica (0,25 m).
    Elige el servicio WMS por la ubicación del punto (Andalucía / Cataluña). Fuera
    de cobertura IR → NaN (build_matrix imputa con la mediana). Gratis, sin clave."""
    lon0 = sum(r['lon'] for r in rows) / len(rows)
    lat0 = sum(r['lat'] for r in rows) / len(rows)
    svc = _cir_service_for(lon0, lat0)
    if svc is None:
        print('  ⚠ Sin servicio PNOA-IR WMS para esta zona '
              '(Andalucía/Cataluña/Canarias). CIR → NaN')
        for r in rows:
            for k in CIR_FEATS: r[k] = float('nan')
        return 0
    url, layer, epsg, fmt = svc
    print(f'  → PNOA Falso Color IR 0,25 m · {url.split("/")[2]} · capa {layer} ({fmt})')

    def work(i_r):
        i, r = i_r
        return i, _cir_one(r['lon'], r['lat'], url, layer, epsg, fmt)

    t0 = time.time(); n_ok = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(work, ir) for ir in enumerate(rows)]
        for fut in as_completed(futs):
            i, feats = fut.result()
            rows[i].update(feats)
            if not (isinstance(feats['cir_ndvi_mean'], float)
                    and np.isnan(feats['cir_ndvi_mean'])):
                n_ok += 1
    print(f'    Puntos con dato CIR: {n_ok}/{len(rows)} (t={time.time()-t0:.0f}s)')
    return n_ok


# ============================================================
# 6. Random Forest
# ============================================================
FEATURES_RF = [
    'frac_bgreen','exg_mean','glcm_homog','glcm_contrast','edge_density',
    'laplacian_var','vegcontrast','surr_rock','granul_5','largest_frac',
    'matorral_score','n_components',
    # Relieve. Sin TWI desde el 20-11-2026: Earth Engine no lo calcula y el modelo
    # acierta lo mismo sin él (retrain_v8_sin_twi.py)
    'elev_dem_m','slope_deg','curvature',
    'ndvi_early','ndvi_late','ndvi_drop','ndwi_late',
    'clre_early','clre_late','gndvi_early','gndvi_late',
    'ndmi_early','ndmi_late','evi_early','evi_late','nbr_late',
    # Índices nuevos (Indices_Mascaras_S2.txt): mean/min/max/sd sobre jun–sep
    'wavi_mean','wavi_min','wavi_max','wavi_sd',
    'albedo_mean','albedo_min','albedo_max','albedo_sd',
    'ndmi_mean','ndmi_min','ndmi_max','ndmi_sd',
    'ndwi_mean','ndwi_min','ndwi_max','ndwi_sd',
    # Sentinel-1 (radar, de Earth Engine): humedad y estructura
    's1_vv_mean','s1_vh_mean','s1_vv_sd','s1_vh_sd','s1_ratio_mean',
]

# Variantes opcionales (modelos rf_*_{cir,planet,combo}.joblib). El modelo por
# defecto sigue siendo FEATURES_RF — la inferencia usa bundle['features'], así que
# cada modelo extrae solo lo que necesita.
FEATURES_RF_PLANET = FEATURES_RF + PS_FEATS                 # +PlanetScope 3 m (mundial, key)
FEATURES_RF_CIR    = FEATURES_RF + CIR_FEATS               # +PNOA-IR 0,25 m (España, gratis)
FEATURES_RF_COMBO  = FEATURES_RF + CIR_FEATS + PS_FEATS    # ambos (ganancia marginal)

def build_matrix(rows):
    X = np.full((len(rows), len(FEATURES_RF)), np.nan, dtype=np.float32)
    for i, r in enumerate(rows):
        for j, f in enumerate(FEATURES_RF):
            v = r.get(f)
            if v is None or v == '': continue
            try: X[i,j] = float(v)
            except: pass
    medians = np.nanmedian(X, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)  # columnas vacías → 0
    for j in range(X.shape[1]):
        m = np.isnan(X[:,j]); X[m,j] = medians[j]
    return X, medians


# ---- Serialización de modelos (mismo formato que rf_sierra_nevada.joblib) ----
def save_model_bundle(bundle, path):
    """Guarda un bundle de modelo (dict con 'model','features','medians',meta)."""
    import joblib
    joblib.dump(bundle, path)
    return path

def load_model_bundle(path, intentos=3):
    """Carga y valida un bundle de modelo. Devuelve dict o lanza excepción.

    Lee el fichero completo a memoria de una sola vez y luego lo deserializa. Es
    más robusto que pasarle la ruta a joblib: en Windows, el antivirus o el
    Control de aplicaciones pueden interferir en una lectura y devolver datos
    incompletos, y entonces el pickle aborta con un error críptico
    (`KeyError: 0`) que no dice ni qué fichero era. Se reintenta un par de veces
    y, si aun así falla, el mensaje identifica el fichero y su tamaño.
    """
    import io
    import time
    import joblib
    ruta = Path(path)
    ultimo = None
    for intento in range(intentos):
        try:
            datos = ruta.read_bytes()
            if len(datos) < 1024:
                raise ValueError(f'el fichero solo tiene {len(datos)} bytes; '
                                 'está vacío o incompleto')
            bundle = joblib.load(io.BytesIO(datos))
            if (not isinstance(bundle, dict) or 'model' not in bundle
                    or 'features' not in bundle):
                raise ValueError('El archivo no es un modelo válido de borreguil '
                                 "(faltan claves 'model'/'features').")
            return bundle
        except Exception as exc:                       # noqa: BLE001
            ultimo = exc
            if intento < intentos - 1:
                time.sleep(0.8)
    tam = ruta.stat().st_size if ruta.exists() else 0
    raise RuntimeError(
        f'No se pudo leer el modelo «{ruta.name}» ({tam:,} bytes) tras '
        f'{intentos} intentos: {type(ultimo).__name__}: {ultimo}. '
        'Si el fichero es correcto, suele deberse a que el antivirus o el '
        'Control de aplicaciones de Windows bloquearon la lectura: reintenta '
        'la ejecución en unos segundos.') from ultimo

def _spatial_groups(rows, idxs, cell_deg=0.05):
    """Grupos para GroupKFold: usa cuenca_id si existe, si no una rejilla espacial."""
    groups = []
    for i in idxs:
        r = rows[i]
        c = r.get('cuenca_id')
        if c:
            groups.append(str(c))
        else:
            gx = round(r['lon'] / cell_deg)
            gy = round(r['lat'] / cell_deg)
            groups.append(f'cell_{gx}_{gy}')
    return np.array(groups)


# Para comparar los dos modelos con los puntos de campo hacen falta, como mínimo,
# estos borreguiles y estas ausencias (con menos, el AUC no dice nada).
MIN_COMPARAR = 5
# Para entrenar un modelo propio: mínimo de positivos y de negativos.
MIN_ENTRENAR = 15


def predecir_con_bundle(rows, bundle):
    """Probabilidad de borreguil que da a cada punto el modelo de `bundle` (el de
    partida o uno propio). Lo que le falte a un punto se rellena con la mediana con la
    que se entrenó el modelo."""
    clf = bundle['model']
    feats = bundle['features']
    medians = np.array(bundle['medians'], dtype=np.float32)
    X = np.full((len(rows), len(feats)), np.nan, dtype=np.float32)
    for i, r in enumerate(rows):
        for j, f in enumerate(feats):
            v = r.get(f)
            if v is None or v == '': continue
            try: X[i,j] = float(v)
            except: pass
    for j in range(X.shape[1]):
        m = np.isnan(X[:,j]); X[m,j] = medians[j]
    return clf.predict_proba(X)[:,1]


def _predecir_partida(rows, model_path=None):
    """Probabilidad que da el MODELO DE PARTIDA (el preentrenado elegido en el
    panel) a cada punto. Devuelve (proba, info) o (None, None) si no hay modelo."""
    if model_path is None:
        model_path = Path(__file__).parent / 'rf_sierra_nevada.joblib'
    if not Path(model_path).exists():
        return None, None
    bundle = load_model_bundle(model_path)
    proba = predecir_con_bundle(rows, bundle)
    return proba, {'mode': 'pretrained',
                   'training_zone': bundle.get('training_zone','?'),
                   'training_n': bundle.get('training_n', 0),
                   'auc_groupkfold': bundle.get('auc_groupkfold', None),
                   'model_bundle': bundle}


def etiquetas_de_campo(rows):
    """Lo que ha dicho una PERSONA de cada punto, en campo o mirando la ortofoto:
    1 = borreguil, 0 = no borreguil, -1 = nada. No cuentan los dudosos: con estas
    etiquetas se mide cuánto acierta cada modelo, y eso tiene que ser dato real."""
    y = np.full(len(rows), -1, dtype=np.int8)
    for i, r in enumerate(rows):
        if _txt(r.get('Duda')).lower() == 'si':
            continue
        b = _txt(r.get('Borreguil')).lower()
        if b == 'si':
            y[i] = 1
        elif b == 'no':
            y[i] = 0
    return y


def _evaluacion(y_campo, rows=None, **probas):
    """La etiqueta de campo y la probabilidad de cada modelo, SOLO en los puntos
    con etiqueta de campo. Es pequeño y no depende del orden de la tabla, así que
    se puede guardar y recalcular los aciertos con otro umbral. Con `rows` (los
    mismos puntos que `y_campo`), también la certidumbre y el perfil de quien dio
    cada etiqueta ('' si no se sabe), para el acierto según la revisión."""
    sel = np.asarray(y_campo) >= 0
    ev = {'etiqueta': [int(v) for v in np.asarray(y_campo)[sel]]}
    if rows is not None and len(rows) == len(sel):
        elegidos = [r for r, s in zip(rows, sel) if s]
        ev['certidumbre'] = [_txt(r.get('certainty')) for r in elegidos]
        ev['perfil'] = [_txt(r.get('expertise')) for r in elegidos]
    for k, p in probas.items():
        if p is not None:
            ev[k] = [float(v) for v in np.asarray(p, dtype=float)[sel]]
    return ev


def _auc(y, p):
    try:
        return float(roc_auc_score(y, p))
    except Exception:
        return None


def aciertos_en_campo(ev, umbral):
    """Cuántos puntos de campo acierta cada modelo con el umbral de la decisión.

    Un borreguil de campo está acertado si el modelo lo da por «probable» o
    «posible» (probabilidad ≥ umbral, o ≥ 0,70, como en decide()); una ausencia,
    si lo da por «incierto» o «no». El AUC solo se da con al menos MIN_COMPARAR
    borreguiles y MIN_COMPARAR ausencias. Devuelve {modelo: {...}} para 'partida'
    y, si se entrenó, 'propio' (con la predicción de la validación por zonas, no
    con la del modelo final, que ya ha visto esos puntos)."""
    if not ev or not ev.get('etiqueta'):
        return {}
    corte = min(float(umbral), 0.70)
    y = np.asarray(ev['etiqueta'])
    si, no = y == 1, y == 0
    out = {}
    for k in ('partida', 'propio'):
        p = ev.get(k)
        if p is None:
            continue
        p = np.asarray(p, dtype=float)
        out[k] = {'n_si': int(si.sum()), 'ok_si': int((p[si] >= corte).sum()),
                  'n_no': int(no.sum()), 'ok_no': int((p[no] < corte).sum()),
                  'auc': (_auc(y, p) if si.sum() >= MIN_COMPARAR
                          and no.sum() >= MIN_COMPARAR else None)}
    return out


def aciertos_por_revision(ev, umbral, modelo='partida'):
    """Los aciertos del `modelo` en los puntos de campo, separados por la
    certidumbre y por el perfil de quien dio la etiqueta (aciertos_en_campo() para
    cada grupo). Devuelve {'certidumbre': {valor: {...}}, 'perfil': {valor: {...}}},
    con '' para «sin indicar», o {} si ningún punto trae certidumbre ni perfil."""
    if not ev or not ev.get('etiqueta') or ev.get(modelo) is None:
        return {}
    out = {}
    for clave, orden in (('certidumbre', CERTIDUMBRES), ('perfil', PERFILES)):
        valores = ev.get(clave) or []
        if len(valores) != len(ev['etiqueta']) or not any(valores):
            continue
        grupos = {}
        for v in list(orden) + ['']:
            sel = [i for i, x in enumerate(valores) if x == v]
            if not sel:
                continue
            sub = {'etiqueta': [ev['etiqueta'][i] for i in sel],
                   modelo: [ev[modelo][i] for i in sel]}
            a = aciertos_en_campo(sub, umbral).get(modelo)
            if a:
                grupos[v] = a
        out[clave] = grupos
    return out


def elegir_modelo(ev):
    """Con qué modelo se queda la app cuando ha entrenado uno propio: con el que
    tenga MÁS AUC en los puntos de campo; si empatan, con el de partida. Si no se
    puede comparar (faltan puntos de campo, no hay modelo de partida o falló la
    validación), con el propio, que es lo que se ha pedido, y lo dice el motivo.
    Devuelve {'elegido', 'motivo', 'auc_partida', 'auc_propio', 'n_si', 'n_no'}."""
    y = np.asarray(ev.get('etiqueta', []))
    n_si, n_no = int((y == 1).sum()), int((y == 0).sum())
    res = {'elegido': 'propio', 'motivo': None, 'auc_partida': None,
           'auc_propio': None, 'n_si': n_si, 'n_no': n_no}
    if ev.get('partida') is None:
        res['motivo'] = 'sin_partida'
    elif ev.get('propio') is None:
        res['motivo'] = 'sin_validacion'
    elif n_si < MIN_COMPARAR or n_no < MIN_COMPARAR:
        res['motivo'] = 'pocos_de_campo'
    else:
        res['auc_partida'] = _auc(y, ev['partida'])
        res['auc_propio'] = _auc(y, ev['propio'])
        if res['auc_partida'] is None or res['auc_propio'] is None:
            res['motivo'] = 'sin_validacion'
        else:
            res['motivo'] = 'comparado'
            if res['auc_propio'] <= res['auc_partida']:
                res['elegido'] = 'partida'
    return res


# ------------------------------------------------------------
# Sumar los puntos de quien usa la app a los del modelo de Sierra Nevada
# ------------------------------------------------------------
# La tabla con la que se entrenó el modelo de Sierra Nevada (796 puntos: 372
# borreguiles, 397 no y 27 dudosos) va con la app. Con la casilla «Sumar mis puntos»,
# el modelo se vuelve a entrenar con ella MÁS los puntos con etiqueta de la sesión: un
# solo conjunto, no dos modelos aparte.
TABLA_ENTRENAMIENTO = 'entrenamiento_sierra_nevada.csv'
# Para medir el acierto en tus puntos sin hacer trampa: al predecir un grupo de tus
# puntos, el modelo no ve ni esos puntos ni los de la tabla a menos de esto de ellos.
SEPARACION_CV_M = 500


def leer_tabla_entrenamiento(path):
    """Filas de la tabla de entrenamiento: lon, lat, borreguil ('si'/'no'), duda
    ('si'/''), cuenca_id y las variables como número (NaN si faltan)."""
    out = []
    with open(path, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            d = {}
            for k, v in r.items():
                if k in ('ID', 'borreguil', 'duda', 'cuenca_id'):
                    d[k] = v or ''
                else:
                    try:
                        d[k] = float(v)
                    except (TypeError, ValueError):
                        d[k] = float('nan')
            out.append(d)
    return out


def _matriz(rows, feats):
    X = np.full((len(rows), len(feats)), np.nan, dtype=np.float32)
    for i, r in enumerate(rows):
        for j, f in enumerate(feats):
            v = r.get(f)
            if v is None or v == '':
                continue
            try:
                X[i, j] = float(v)
            except (TypeError, ValueError):
                pass
    return X


def _rf_sierra_nevada(n_estimators=400):
    """Los mismos parámetros con los que se entrenó el modelo de Sierra Nevada."""
    return RandomForestClassifier(n_estimators=n_estimators, max_depth=12,
                                  min_samples_split=4, min_samples_leaf=2,
                                  class_weight='balanced', random_state=42, n_jobs=-1)


def _distancias_m(lon1, lat1, lon2, lat2):
    """Matriz de distancias aproximadas (m) entre dos listas de puntos."""
    lon1, lat1 = np.asarray(lon1, float)[:, None], np.asarray(lat1, float)[:, None]
    lon2, lat2 = np.asarray(lon2, float)[None, :], np.asarray(lat2, float)[None, :]
    dx = (lon2 - lon1) * 111320.0 * np.cos(np.radians((lat1 + lat2) / 2))
    dy = (lat2 - lat1) * 110540.0
    return np.hypot(dx, dy)


def _entrenar_sumando(rows, tabla, feats, proba_p, info_p, y_campo, cv_arboles=150):
    """Entrena UN modelo con la tabla de entrenamiento y los puntos con etiqueta de la
    sesión (Borreguil 'si'/'no', sin los dudosos). Un punto de la tabla a menos de
    CERCA_M de uno de la sesión es el mismo sitio: vale el de la sesión. Devuelve
    (proba, info) como train_rf."""
    lab = [i for i, r in enumerate(rows)
           if not _es_si(r.get('Duda')) and _txt(r.get('Borreguil')).lower() in ('si', 'no')]
    zona = (info_p or {}).get('training_zone') or 'Sierra Nevada'
    if not lab:
        if proba_p is None:
            return None, None
        info_p['motivo'] = 'sin_puntos'
        return proba_p, info_p
    t_ok = [t for t in tabla if t.get('borreguil') in ('si', 'no') and not _es_si(t.get('duda'))]
    s_lon = [rows[i]['lon'] for i in lab]
    s_lat = [rows[i]['lat'] for i in lab]
    D = _distancias_m([t['lon'] for t in t_ok], [t['lat'] for t in t_ok], s_lon, s_lat)
    dmin = D.min(axis=1) if D.size else np.full(len(t_ok), np.inf)
    t_usa = [t for t, d in zip(t_ok, dmin) if d >= CERCA_M]
    n_quitados = len(t_ok) - len(t_usa)

    y_t = np.array([1 if t['borreguil'] == 'si' else 0 for t in t_usa], dtype=np.int8)
    y_s = np.array([1 if _txt(rows[i].get('Borreguil')).lower() == 'si' else 0 for i in lab],
                   dtype=np.int8)
    X_t, X_s = _matriz(t_usa, feats), _matriz([rows[i] for i in lab], feats)
    X = np.vstack([X_t, X_s])
    y = np.concatenate([y_t, y_s])
    with np.errstate(all='ignore'):
        medians = np.nanmedian(X, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)

    def imputa(A):
        A = A.copy()
        for j in range(A.shape[1]):
            m = np.isnan(A[:, j]); A[m, j] = medians[j]
        return A

    n_si_s, n_no_s = int(y_s.sum()), int(len(y_s) - y_s.sum())
    print(f'  → Reentrenando con {len(t_usa)} puntos de {zona.split(" (")[0]} '
          f'({int(y_t.sum())} borreguiles) + {len(lab)} tuyos ({n_si_s} borreguiles, '
          f'{n_no_s} no)' + (f'; {n_quitados} de la tabla coinciden con tuyos'
                             if n_quitados else ''))
    clf = _rf_sierra_nevada().fit(imputa(X), y)
    proba = clf.predict_proba(imputa(_matriz(rows, feats)))[:, 1]

    # Acierto en tus puntos sin hacer trampa: validación por zonas. Cada grupo de tus
    # puntos se predice con un modelo que no ha visto ni esos puntos ni los de la
    # tabla a menos de SEPARACION_CV_M de ellos.
    proba_cv, cv_label = None, None
    grupos = _spatial_groups(rows, lab)
    ug = sorted(set(grupos))
    if len(ug) >= 2:
        proba_cv = np.full(len(rows), np.nan)
        k = min(5, len(ug))
        try:
            for tr, te in GroupKFold(n_splits=k).split(X_s, y_s, grupos):
                lejos = D[:, te].min(axis=1) >= SEPARACION_CV_M
                lejos_t = lejos[dmin >= CERCA_M]
                Xa = np.vstack([X_t[lejos_t], X_s[tr]])
                ya = np.concatenate([y_t[lejos_t], y_s[tr]])
                if len(set(ya.tolist())) < 2:
                    continue
                m = _rf_sierra_nevada(cv_arboles).fit(imputa(Xa), ya)
                proba_cv[[lab[i] for i in te]] = m.predict_proba(imputa(X_s[te]))[:, 1]
            cv_label = f'{k} grupos por zonas'
        except Exception as e:
            print(f'    La validación por zonas falló ({e})')
            proba_cv, cv_label = None, None
    ev = _evaluacion(y_campo, rows, partida=proba_p, propio=proba_cv)
    ac = aciertos_en_campo(ev, 0.5).get('propio') if proba_cv is not None else None
    auc_cv = ac.get('auc') if ac else None

    bundle = {'model': clf, 'features': list(feats),
              'medians': [float(v) for v in medians],
              'training_zone': f'{zona.split(" (")[0]} + tus puntos',
              'training_n': int(len(y)), 'n_positives': int(y.sum()),
              'n_negatives': int(len(y) - y.sum()), 'auc_groupkfold': auc_cv,
              'ref_stats': (info_p or {}).get('model_bundle', {}).get('ref_stats')
              if info_p else None}
    info = {'mode': 'trained', 'sumado': True, 'model_bundle': bundle,
            'evaluacion': ev, 'comparacion': None, 'zona_partida': zona,
            'auc': auc_cv if auc_cv is not None else float('nan'), 'cv': cv_label,
            'n_train': int(len(y)), 'n_pos': n_si_s, 'n_neg': n_no_s,
            'n_tabla': len(t_usa), 'n_tabla_si': int(y_t.sum()),
            'n_tabla_no': int(len(y_t) - y_t.sum()), 'n_tabla_quitados': n_quitados}
    return proba, info


def train_rf(rows, default_model_path=None, neg_buffer_m=250, propio=False,
             entrenamiento=None):
    """Da a cada punto su probabilidad de ser borreguil. Devuelve (proba, info).

    POR DEFECTO (propio=False) usa el MODELO DE PARTIDA (el preentrenado elegido
    en el panel, p. ej. el de Sierra Nevada), aunque los puntos traigan etiquetas
    de campo: con ellas solo se mide cuánto acierta en la zona (info['evaluacion']).
    Así un socio con 15 borreguiles marcados no se queda sin darse cuenta con un
    modelo flojo entrenado solo con ellos.

    Con propio=True entrena ADEMÁS un RF en modo presence-only con los puntos:
    - Positivos: rows con Borreguil=='si'.
    - Negativos: los marcados 'no' y las pseudo-ausencias (candidatos SIN etiqueta
      a más de neg_buffer_m de cualquier positivo).
    - Ambiguos (candidatos a < neg_buffer_m de un positivo) y dudosos: fuera del
      entrenamiento, pero se predicen igualmente.
    Si hay menos de MIN_ENTRENAR positivos o negativos, se queda con el de partida.
    Si no, compara los dos con los puntos de campo (validación por zonas para el
    propio) y se queda con el que más acierte (elegir_modelo).

    info['mode'] es 'pretrained' o 'trained'; info['motivo'] dice por qué se usa
    el de partida cuando se pidió el propio ('sin_borreguiles', 'pocos_datos',
    'comparado'), e info['comparacion'] el resultado de elegir_modelo.

    Si no hay modelo de partida, entrena el propio si puede (como antes).

    Con `entrenamiento` (la tabla de entrenamiento del modelo de Sierra Nevada, ver
    leer_tabla_entrenamiento) y propio=True, NO entrena un modelo aparte con tus
    puntos: vuelve a entrenar el de partida con su tabla MÁS tus puntos con etiqueta
    (_entrenar_sumando), sin pseudo-ausencias ni comparación para elegir.
    """
    truth_attr = 'Borreguil'; duda_attr = 'Duda'
    y_campo = etiquetas_de_campo(rows)
    proba_p, info_p = _predecir_partida(rows, default_model_path)
    if proba_p is not None:
        info_p['evaluacion'] = _evaluacion(y_campo, rows, partida=proba_p)
        info_p['motivo'] = None
        if not propio:
            print(f'  → Modelo de partida ({info_p["training_zone"]})')
            return proba_p, info_p
    if entrenamiento:
        feats = ((info_p or {}).get('model_bundle') or {}).get('features') or FEATURES_RF
        return _entrenar_sumando(rows, entrenamiento, list(feats), proba_p, info_p, y_campo)

    has_pos = any(str(r.get(truth_attr, '')).lower() == 'si' for r in rows)
    if not has_pos:
        if proba_p is not None:
            print(f'  → Sin borreguiles marcados: modelo de partida ({info_p["training_zone"]})')
            info_p['motivo'] = 'sin_borreguiles'
            return proba_p, info_p
        print('  → Sin verdad-terreno y sin modelo preentrenado. RF no se ejecuta.')
        return None, None

    print('  → Entrenando Random Forest (presence-only) con verdad-terreno…')
    X, medians = build_matrix(rows)
    n = len(rows)

    # Etiquetas iniciales
    is_pos = np.array([str(r.get(truth_attr, '')).lower() == 'si' for r in rows])
    is_neg_explicit = np.array([str(r.get(truth_attr, '')).lower() == 'no' for r in rows])
    is_duda = np.array([str(r.get(duda_attr, '')).lower() == 'si' for r in rows])

    # Pseudo-ausencias: candidatos sin etiqueta, lejos de TODO positivo
    epsg = _utm_epsg_for(rows)
    xy = np.array(_xy(rows, epsg))
    pos_xy = xy[is_pos]
    dist_to_pos = np.full(n, np.inf)
    if len(pos_xy):
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(pos_xy)
            dist_to_pos, _ = tree.query(xy)
        except Exception:
            for i in range(n):
                dist_to_pos[i] = np.min(np.hypot(xy[i,0]-pos_xy[:,0], xy[i,1]-pos_xy[:,1]))

    candidate = (~is_pos) & (~is_duda)
    pseudo_neg = candidate & (dist_to_pos > neg_buffer_m)
    y = np.full(n, -1, dtype=np.int8)
    y[is_pos] = 1
    y[is_neg_explicit | pseudo_neg] = 0

    train_mask = (y != -1) & (~is_duda)
    npos = int((y[train_mask] == 1).sum())
    nneg = int((y[train_mask] == 0).sum())
    n_neg_campo = int((is_neg_explicit & train_mask).sum())
    n_pseudo = int((pseudo_neg & ~is_neg_explicit & train_mask).sum())
    print(f'    Positivos (verdad-terreno): {npos}')
    print(f'    Negativos: {nneg} ({n_neg_campo} de campo + {n_pseudo} pseudo-ausencias '
          f'> {neg_buffer_m} m)')
    print(f'    Ambiguos excluidos (candidatos < {neg_buffer_m} m de un positivo): '
          f'{int(candidate.sum() - pseudo_neg.sum())}')
    cuentas = {'n_pos': npos, 'n_neg': nneg, 'n_neg_campo': n_neg_campo,
               'n_pseudo': n_pseudo, 'neg_buffer_m': neg_buffer_m}

    if npos < MIN_ENTRENAR or nneg < MIN_ENTRENAR:
        print(f'    ATENCIÓN: muestra insuficiente (pos={npos}, neg={nneg}). '
              'Mínimo recomendado: 30 positivos + 60 negativos. '
              'Se usa el modelo preentrenado.')
        if proba_p is None:
            return None, None
        info_p.update(cuentas, motivo='pocos_datos')
        return proba_p, info_p

    clf = RandomForestClassifier(n_estimators=400, max_depth=12,
                                  min_samples_split=4, min_samples_leaf=2,
                                  class_weight='balanced', random_state=42, n_jobs=-1)

    groups = _spatial_groups(rows, np.where(train_mask)[0])
    ugroups = set(groups)
    yhat_cv = None
    try:
        if len(ugroups) >= 3:
            yhat_cv = cross_val_predict(clf, X[train_mask], y[train_mask],
                                        cv=GroupKFold(n_splits=min(5, len(ugroups))),
                                        groups=groups, method='predict_proba', n_jobs=-1)[:,1]
            cv_label = f'GroupKFold espacial ({len(ugroups)} grupos)'
        else:
            yhat_cv = cross_val_predict(clf, X[train_mask], y[train_mask],
                                        cv=StratifiedKFold(5, shuffle=True, random_state=42),
                                        method='predict_proba', n_jobs=-1)[:,1]
            cv_label = 'StratifiedKFold'
        auc = roc_auc_score(y[train_mask], yhat_cv)
        print(f'    AUC {cv_label}: {auc:.3f}')
    except Exception as e:
        print(f'    CV falló ({e})'); auc = float('nan'); cv_label = 'n/a'; yhat_cv = None

    # Predicción de la validación por zonas en cada punto de entrenamiento: con
    # ella se mide el acierto del modelo propio en los puntos de campo (el modelo
    # final ya los ha visto y acertaría casi todos).
    proba_cv = None
    if yhat_cv is not None:
        proba_cv = np.full(n, np.nan)
        proba_cv[train_mask] = yhat_cv
    ev = _evaluacion(y_campo, rows, partida=proba_p, propio=proba_cv)
    comp = elegir_modelo(ev)
    if comp is not None:
        print(f'    Comparación con los puntos de campo: {comp}')
    if comp is not None and comp['elegido'] == 'partida':
        print('    Se queda el modelo de partida: acierta más con tus puntos de campo.')
        info_p.update(cuentas, motivo='comparado', comparacion=comp, evaluacion=ev,
                      auc_propio_cv=float(auc) if auc == auc else None, cv=cv_label)
        return proba_p, info_p

    clf.fit(X[train_mask], y[train_mask])
    proba = clf.predict_proba(X)[:,1]
    bundle = {
        'model': clf,
        'features': list(FEATURES_RF),
        'medians': [float(v) for v in medians],
        'training_zone': 'usuario',
        'training_n': int(train_mask.sum()),
        'n_positives': npos,
        'n_negatives': nneg,
        'auc_groupkfold': float(auc) if auc == auc else None,
    }
    info_t = {'mode': 'trained', 'auc': float(auc), 'cv': cv_label,
              'n_train': int(train_mask.sum()), 'model_bundle': bundle,
              'evaluacion': ev, 'comparacion': comp,
              'zona_partida': (info_p or {}).get('training_zone')}
    info_t.update(cuentas)
    return proba, info_t


# ============================================================
# DECISIÓN FINAL
# ============================================================
def corte_borreguil(threshold):
    """Probabilidad a partir de la cual decide() da un punto por borreguil (probable
    o posible): el umbral, o 0,70 si el umbral es más alto."""
    return min(float(threshold), 0.70)


def decide(r, threshold=0.5):
    # _txt: al leer un fichero, los puntos sin etiqueta llegan con NaN (no con
    # cadena vacía) y `.strip()` sobre un NaN detenía toda la ejecución.
    truth = _txt(r.get('Borreguil')).lower()
    duda  = _txt(r.get('Duda')).lower() == 'si'
    rf = r.get('rf_proba', float('nan'))

    if duda:
        return 'DUDOSO (campo)'
    if truth == 'si':
        return 'BORREGUIL VERIFICADO'
    if truth == 'no':
        return 'NO BORREGUIL VERIFICADO'
    if not isinstance(rf, float) or math.isnan(rf):
        return 'SIN PREDICCIÓN'
    if rf >= corte_borreguil(threshold): return ('BORREGUIL PROBABLE' if rf >= 0.70
                                                 else 'POSIBLE BORREGUIL')
    if rf <= 0.20: return 'NO BORREGUIL'
    return 'INCIERTO'


# ------------------------------------------------------------
# Puntos añadidos pinchando en el mapa
# ------------------------------------------------------------
# Quien conoce el sitio (en el taller, los gestores) pincha en el mapa y dice si ahí
# hay borreguil. El punto entra como un dato de campo más, con las columnas de la
# plantilla: `method` dice que viene del conocimiento del sitio, no de una visita.
METODO_MAPA = 'local knowledge'
RESPUESTAS_MAPA = ('si', 'no', 'duda')
# A menos de esto de un punto que ya está, el clic es sobre ese punto (pinchar en un
# marcador también es pinchar en el mapa).
CERCA_M = 15


def distancia_m(lon1, lat1, lon2, lat2):
    """Distancia aproximada en metros (basta para distancias cortas)."""
    dx = (lon2 - lon1) * 111320.0 * math.cos(math.radians((lat1 + lat2) / 2))
    dy = (lat2 - lat1) * 110540.0
    return math.hypot(dx, dy)


def punto_mas_cercano(rows, lon, lat):
    """(índice, distancia en m) del punto más cercano a (lon, lat); (None, inf) si no
    hay puntos."""
    mejor, dmin = None, float('inf')
    for i, r in enumerate(rows):
        try:
            d = distancia_m(lon, lat, float(r['lon']), float(r['lat']))
        except (KeyError, TypeError, ValueError):
            continue
        if d < dmin:
            mejor, dmin = i, d
    return mejor, dmin


def nuevo_id_mapa(rows):
    """mapa_1, mapa_2…: el siguiente número a los que ya hay."""
    import re
    n = 0
    for r in rows:
        m = re.fullmatch(r'mapa_(\d+)', _txt(r.get('ID')))
        if m:
            n = max(n, int(m.group(1)))
    return f'mapa_{n + 1}'


def punto_del_mapa(rows, lon, lat, respuesta, observador='', nota='', fecha=None,
                   perfil='', certidumbre=''):
    """El punto que se añade al pinchar en el mapa. `respuesta`: 'si' (aquí hay
    borreguil), 'no' (aquí no hay) o 'duda' (no está seguro: como un dudoso de campo,
    no entrena ni cuenta en el acierto). `perfil` y `certidumbre`, como en
    anotar_revision(); un dudoso no lleva certidumbre."""
    if respuesta not in RESPUESTAS_MAPA:
        raise ValueError(f'respuesta no válida: {respuesta!r}')
    import datetime as _dt
    r = {'ID': nuevo_id_mapa(rows), 'lon': float(lon), 'lat': float(lat),
         'source': 'mapa',
         'Borreguil': {'si': 'si', 'no': 'no'}.get(respuesta, ''),
         'Duda': 'si' if respuesta == 'duda' else '',
         'origin': 'dudoso' if respuesta == 'duda' else 'verificado',
         'method': METODO_MAPA,
         'date': fecha or _dt.date.today().isoformat()}
    anotar_revision(r, observador, perfil, '' if respuesta == 'duda' else certidumbre)
    if _txt(nota):
        r['notes'] = _txt(nota)
    return r


def respuesta_del_mapa(r):
    """'si', 'no' o 'duda' de un punto añadido en el mapa (None si no lo es)."""
    if _txt(r.get('source')) != 'mapa':
        return None
    if _es_si(r.get('Duda')):
        return 'duda'
    return {'si': 'si', 'no': 'no'}.get(_txt(r.get('Borreguil')).lower())


def opinion_modelo(proba, threshold):
    """'si' si el modelo daría el punto por borreguil (lo mismo que decide()), 'no'
    si no, None si no hay probabilidad."""
    if not isinstance(proba, (int, float)) or proba != proba:
        return None
    return 'si' if proba >= corte_borreguil(threshold) else 'no'


def resumen_mapa(rows, threshold):
    """Los puntos añadidos en el mapa y si el modelo coincidía con quien los marcó.
    La probabilidad es la que dio el modelo AL AÑADIRLOS (`rf_proba_al_anadir`), antes
    de saber la respuesta: así no cuenta como acierto lo que el modelo propio
    aprende de ese mismo punto. Devuelve (filas, n con respuesta sí/no, n en que
    coincide)."""
    filas, n, ok = [], 0, 0
    for r in rows:
        resp = respuesta_del_mapa(r)
        if resp is None:
            continue
        p = r.get('rf_proba_al_anadir')
        op = opinion_modelo(p, threshold)
        coincide = None if (resp == 'duda' or op is None) else (op == resp)
        if coincide is not None:
            n += 1; ok += int(coincide)
        filas.append({'ID': r.get('ID'), 'respuesta': resp,
                      'proba': p if op is not None else None, 'modelo': op,
                      'coincide': coincide, 'observer': _txt(r.get('observer')),
                      'notes': _txt(r.get('notes'))})
    return filas, n, ok


def auto_threshold(probs, lo=0.30, hi=0.80, default=0.50):
    """Umbral RF automático por el método de Otsu sobre la distribución de
    probabilidades: el corte que mejor separa los dos grupos (no-borreguil /
    borreguil). Sin variación o con muy pocos puntos devuelve `default`. Se acota
    a [lo, hi] para evitar cortes extremos."""
    import numpy as np
    vals = [float(p) for p in (probs or [])
            if isinstance(p, (int, float)) and p == p]
    if len(vals) < 8:
        return default
    arr = np.asarray(vals, dtype=float)
    if float(arr.max() - arr.min()) < 1e-3:
        return default
    try:
        from skimage.filters import threshold_otsu
        t = float(threshold_otsu(arr))
    except Exception:
        t = float(np.median(arr))
    return float(min(max(t, lo), hi))


def auto_threshold_truth(rows, lo=0.30, hi=0.80, recall_target=0.90, min_n=5):
    """Umbral RF a partir de la VERDAD-TERRENO. Devuelve None si no hay datos
    suficientes (para poder caer en Otsu). Acota a [lo, hi].
      - Con presencias ('si') Y ausencias ('no'): corte que maximiza el índice de
        Youden (J = sensibilidad + especificidad − 1).
      - Solo con presencias (presence-only): corte que captura `recall_target`
        (90%) de los borreguiles conocidos (percentil de sus probabilidades)."""
    import numpy as np
    def _proba(r):
        v = r.get('rf_proba')
        return float(v) if isinstance(v, (int, float)) and v == v else None
    pos = [p for r in rows if str(r.get('Borreguil', '')).lower() == 'si'
           and (p := _proba(r)) is not None]
    neg = [p for r in rows if str(r.get('Borreguil', '')).lower() == 'no'
           and (p := _proba(r)) is not None]
    if len(pos) < min_n:
        return None
    if len(neg) >= min_n:
        scores = np.array(pos + neg, dtype=float)
        labels = np.array([1] * len(pos) + [0] * len(neg))
        best_t, best_j = 0.5, -1.0
        for t in np.unique(np.round(scores, 3)):
            tp = int(np.sum((scores >= t) & (labels == 1)))
            fn = int(np.sum((scores < t) & (labels == 1)))
            tn = int(np.sum((scores < t) & (labels == 0)))
            fp = int(np.sum((scores >= t) & (labels == 0)))
            sens = tp / (tp + fn) if (tp + fn) else 0.0
            spec = tn / (tn + fp) if (tn + fp) else 0.0
            j = sens + spec - 1.0
            if j > best_j:
                best_j, best_t = j, float(t)
        return float(min(max(best_t, lo), hi))
    # Presence-only: percentil que mantiene recall_target de los positivos
    t = float(np.percentile(pos, (1.0 - recall_target) * 100.0))
    return float(min(max(t, lo), hi))


def decision_color(dec):
    """Color hex para una decisión (compartido por la app y el mapa exportado)."""
    d = (dec or '').lower()
    if 'verificado' in d and 'no ' not in d:         return '#1B5E20'  # verde oscuro
    if 'no borreguil' in d:                          return '#8E0000'  # rojo oscuro
    if 'dudoso' in d:                                return '#E69100'  # ámbar
    if 'probable' in d:                              return '#2E7D32'  # verde
    if 'posible' in d:                               return '#66BB6A'  # verde claro
    if 'incierto' in d:                              return '#FFB74D'  # naranja claro
    return '#999999'


# ============================================================
# Clasificación jerárquica del TIPO de borreguil (reglas sobre las features ya
# calculadas; no necesita puntos etiquetados por subtipo). Tres niveles:
#   1) AMBIENTE : arroyo | laguna | ladera   (siempre una de las tres)
#   2) HUMEDAD  : húmedo | seco              (siempre una de las dos)
#   3) PUREZA   : puro | mixto-agua | mixto-roca  (como en el mapa: assign_pureza)
# Los umbrales del ambiente y la humedad son AJUSTABLES aquí; los de la pureza son los
# del mapa (mapeo_rf.PUREZA_DEFECTO). Un nivel vale '—' si faltan sus variables (p. ej.
# sin topografía no hay ambiente; sin leer el píxel en Earth Engine, no hay pureza).
# ============================================================
HIER_THRESH = {
    'dist_laguna_m':      120.0,  # a < 120 m de una laguna conocida (capa) → laguna
    'ndwi_laguna':        0.05,   # NDWI medio alto → lámina de agua estancada cerca
    'surr_water_laguna':  0.15,   # fracción de agua alrededor (imagen) alta
    'dist_arroyo_m':      30.0,   # a ≤ 30 m de un río o arroyo de OpenStreetMap → arroyo
    'slope_arroyo':       7.0,    # sin cauces de OSM (no se han podido leer): pendiente baja
    'ndmi_humedo':        0.10,   # NDMI (humedad) medio por encima → húmedo
}


def attach_laguna_distance(rows, laguna_pts):
    """Calcula r['dist_laguna_m'] = distancia (m) a la laguna más cercana de una capa
    de referencia. `laguna_pts` es una lista de dicts con 'lat'/'lon' (p. ej. de
    read_points). Devuelve el nº de lagunas usadas (0 si no hay)."""
    laguna_pts = [p for p in (laguna_pts or [])
                  if p.get('lat') is not None and p.get('lon') is not None]
    if not rows or not laguna_pts:
        return 0
    epsg = _utm_epsg_for(rows + laguna_pts)
    rxy = np.array(_xy(rows, epsg))
    lxy = np.array(_xy(laguna_pts, epsg))
    try:
        from scipy.spatial import cKDTree
        d, _ = cKDTree(lxy).query(rxy)
    except Exception:
        d = np.array([float(np.min(np.hypot(x - lxy[:, 0], y - lxy[:, 1])))
                      for x, y in rxy])
    for r, dist in zip(rows, d):
        r['dist_laguna_m'] = float(dist)
    return len(laguna_pts)


def _hier_num(r, *keys):
    """Primer valor numérico válido entre varias claves alternativas (o None)."""
    for k in keys:
        try:
            x = float(r.get(k))
            if x == x:
                return x
        except (TypeError, ValueError):
            pass
    return None


def classify_hierarchy(r, th=None):
    """Clasificación jerárquica de un punto borreguil por reglas sobre sus features.
    Devuelve dict {'ambiente','humedad','pureza','tipo'}; cada nivel es '—' si no
    hay datos suficientes y 'tipo' es la combinación legible de los tres. La pureza
    sale de assign_pureza (como en el mapa): aquí vale '—' hasta que se le pase."""
    th = th or HIER_THRESH
    ndwi       = _hier_num(r, 'ndwi_mean', 'ndwi_late')
    surr_water = _hier_num(r, 'surr_water')
    slope      = _hier_num(r, 'slope_deg')
    dist_c     = _hier_num(r, 'dist_cauce_m')
    ndmi       = _hier_num(r, 'ndmi_mean', 'ndmi_late')
    dist_lag   = _hier_num(r, 'dist_laguna_m')

    # ---- Nivel 1: AMBIENTE ----
    # Prioridad: cercanía a una laguna conocida (capa de referencia, lo más fiable);
    # luego firma espectral de agua (NDWI/agua alrededor); luego cauce/vaguada.
    if dist_lag is not None and dist_lag < th['dist_laguna_m']:
        amb = 'laguna'
    elif (surr_water is not None and surr_water > th['surr_water_laguna']) or \
         (ndwi is not None and ndwi > th['ndwi_laguna']):
        amb = 'laguna'
    elif dist_c is not None:
        # junto a un río o arroyo de OpenStreetMap
        amb = 'arroyo' if dist_c <= th['dist_arroyo_m'] else 'ladera'
    elif slope is not None:
        # sin los cauces (no se han podido leer): solo la pendiente, que se queda corta
        # (junto a un arroyo suele haber pendiente)
        amb = 'arroyo' if slope < th['slope_arroyo'] else 'ladera'
    else:
        amb = '—'

    # ---- Nivel 2: HUMEDAD (NDMI) ----
    if ndmi is not None:
        hum = 'húmedo' if ndmi > th['ndmi_humedo'] else 'seco'
    else:
        hum = '—'

    pur = '—'                     # la pone assign_pureza
    tipo = f'{amb} · {hum} · {pur}'
    return {'ambiente': amb, 'humedad': hum, 'pureza': pur, 'tipo': tipo}


# Lo que se guarda en cada punto para su pureza (gee_backend.fetch_pureza_gee lee los
# dos primeros; assign_pureza pone el tercero). Son variables: no viajan al GeoJSON.
CLAVES_PUREZA_PUNTO = ('ndvi_early_pixel', 'agua_pixel_vecinos', 'verde_estimado')


def assign_pureza(borr_rows, th=None, todo_verde=None):
    """Propone la PUREZA de cada borreguil igual que el mapa («Pureza dentro del
    píxel», mapeo_rf): cuánto verde hay en su píxel de 10 m, estimado con el NDVI de
    inicio de verano de ese píxel (r['ndvi_early_pixel']); puro si llega al corte;
    si no, mixto-agua si el píxel o uno de sus 8 vecinos es agua
    (r['agua_pixel_vecinos'] = 1) y mixto-roca en otro caso. Esos dos datos se leen
    con Earth Engine (gee_backend.fetch_pureza_gee); sin ellos no hay propuesta: '—'.

    `todo_verde`: el NDVI de «todo verde»; si no se da, el del mapa por defecto.
    Hasta el 24-11-2026 la pureza era relativa al conjunto (percentiles de NDWI, albedo
    y NDVI, y la foto aérea de alrededor). Frente a la ortofoto de 0,25 m, en 372
    borreguiles de campo, aquella acertaba el 62 % y esta el 82 %.
    Actualiza r['pureza'], r['verde_estimado'] (0 a 1) y r['tipo_borreguil']."""
    import mapeo_rf as mr
    if not borr_rows:
        return
    try:
        q = mr.params_pureza(None if todo_verde is None else {'todo_verde': todo_verde})
    except ValueError:
        q = mr.params_pureza()
    nombre = {v: k for k, v in mr.codigos('pureza').items()}
    for r in borr_rows:
        nd = _hier_num(r, 'ndvi_early_pixel')
        if nd is None:
            pur, verde = '—', None
        else:
            verde = float(mr.verde_estimado(nd, q['sin_verde'], q['todo_verde']))
            agua = _hier_num(r, 'agua_pixel_vecinos') == 1
            pur = nombre[int(mr.pureza_de(verde, agua, q['puro_desde']))]
        r['pureza'] = pur
        r['verde_estimado'] = verde
        r['tipo_borreguil'] = f"{r.get('ambiente', '—')} · {r.get('humedad', '—')} · {pur}"


# ============================================================
# Tipo de borreguil REVISADO por una persona
# ------------------------------------------------------------
# classify_hierarchy/assign_pureza PROPONEN el tipo con reglas sobre las propias
# variables predictoras (la humedad es un umbral de NDMI, etc.). Esa propuesta no
# sirve como etiqueta de entrenamiento: un modelo entrenado con ella solo
# reaprendería la regla. Cuando una persona revisa el punto (ortofoto, campo) y
# confirma o corrige el tipo, la etiqueta pasa a ser un dato independiente.
# Por eso se guardan por separado, y nunca se pisa lo revisado:
#   <nivel>_regla        lo que propone la app (se recalcula en cada ejecución)
#   <nivel>              el valor vigente: el revisado si lo hay; si no, la propuesta
#   <nivel>_revisado     'si' cuando una persona ha revisado ESE nivel: cada nivel
#                        cuenta por separado (quien solo sabe la humedad de un punto
#                        aporta una muestra al modelo de humedad)
#   tipo_revisado        'si' cuando están revisados los tres (en ficheros anteriores
#                        a las marcas por nivel, quería decir eso mismo)
#   tipo_revisado_fecha  cuándo se revisó por última vez
# ============================================================
NIVELES_TIPO = {
    'ambiente': ('arroyo', 'laguna', 'ladera'),
    'humedad':  ('húmedo', 'seco'),
    'pureza':   ('puro', 'mixto-agua', 'mixto-roca'),
}
CAMPOS_REVISION = (tuple(NIVELES_TIPO) + tuple(n + '_regla' for n in NIVELES_TIPO)
                   + tuple(n + '_revisado' for n in NIVELES_TIPO)
                   + ('tipo_borreguil', 'tipo_revisado', 'tipo_revisado_fecha'))


def _txt(v):
    """Texto limpio de un atributo ('' para None/NaN: al releer un fichero, los
    huecos llegan como NaN y no como cadena vacía)."""
    if v is None or (isinstance(v, float) and v != v):
        return ''
    return str(v).strip()


def _fecha_txt(v):
    """Fecha de revisión como texto ISO. Al releer un fichero en el que todos los
    puntos tienen fecha, la columna llega como fecha (no como texto).

    Con GDAL, una columna `date` de la plantilla (solo el día, «2026-10-01») llega
    como fecha con hora 00:00:00 aunque falte en algún punto: se devuelve solo el
    día, que es lo que se guardó, y no «2026-10-01T00:00:00»."""
    if v is None or v != v:                       # None, NaN, NaT
        return ''
    if not hasattr(v, 'isoformat'):
        return str(v).strip()
    if (callable(getattr(v, 'date', None))
            and not any(getattr(v, a, 0) for a in ('hour', 'minute', 'second',
                                                   'microsecond'))):
        return v.date().isoformat()
    return v.isoformat()


def _es_si(v):
    return _txt(v).lower() in ('si', 'sí')


def niveles_revisados(r):
    """Los niveles del tipo que una persona ha revisado en este punto (en el orden de
    NIVELES_TIPO). `tipo_revisado` = 'si' sin marcas por nivel son los tres."""
    if _es_si(r.get('tipo_revisado')):
        return list(NIVELES_TIPO)
    return [n for n in NIVELES_TIPO if _es_si(r.get(n + '_revisado'))]


def tipo_revisado(r):
    """True si una persona ha revisado el tipo de este punto, en algún nivel."""
    return bool(niveles_revisados(r))


def niveles_invalidos(r, niveles=None):
    """Niveles (de `niveles`, o todos) cuyo valor vigente no es una categoría
    admitida."""
    return [n for n in (niveles or NIVELES_TIPO)
            if _txt(r.get(n)) not in NIVELES_TIPO[n]]


def _marcar_revisados(r, niveles):
    """Deja en el punto las marcas de revisión de `niveles` (y quita las demás)."""
    for n in NIVELES_TIPO:
        r[n + '_revisado'] = 'si' if n in niveles else ''
    r['tipo_revisado'] = 'si' if len(niveles) == len(NIVELES_TIPO) else ''


def _tipo_compuesto(r):
    return ' · '.join(_txt(r.get(n)) or '—' for n in NIVELES_TIPO)


def apply_typology(rows, es_borreguil, th=None):
    """(Re)calcula la tipología de todos los puntos SIN pisar lo revisado.

    Hay que llamarla tras cada cambio de decisiones (ejecución inicial,
    reentrenamiento): antes solo se calculaba una vez, y un
    punto que pasaba a borreguil al reentrenar se quedaba sin tipo.

    es_borreguil(decision) -> bool.  Devuelve los ID de los puntos que venían
    marcados como revisados pero con alguna categoría no admitida: esos niveles
    dejan de contar como revisados y su valor original queda en
    `tipo_revision_invalida` (no se corrige nada en silencio).
    """
    invalidos, poblacion = [], []
    for r in rows:
        rev = niveles_revisados(r)
        if rev:
            malos = niveles_invalidos(r, rev)
            if malos:
                r['tipo_revision_invalida'] = '; '.join(
                    f'{n}={_txt(r.get(n))!r}' for n in malos)
                rev = [n for n in rev if n not in malos]
                invalidos.append(r.get('ID'))
        if rev or es_borreguil(r.get('decision', '')):
            poblacion.append((r, {n: _txt(r.get(n)) for n in rev}))
        else:
            for n in NIVELES_TIPO:
                r[n] = ''
                r[n + '_regla'] = ''
            r['tipo_borreguil'] = ''
            _marcar_revisados(r, [])
            r['tipo_revisado_fecha'] = ''
    # Propuesta de la regla: ambiente y humedad por punto; pureza como en el mapa, con
    # el NDVI de «todo verde» que proponen los puntos revisados como puros (si hay 5).
    for r, _previo in poblacion:
        h = classify_hierarchy(r, th)
        r['ambiente'], r['humedad'], r['pureza'] = h['ambiente'], h['humedad'], h['pureza']
    import mapeo_rf as mr
    todo_verde, _de_puros = mr.todo_verde_de(
        [_hier_num(r, 'ndvi_early_pixel') for r, previo in poblacion
         if previo.get('pureza') == 'puro'])
    assign_pureza([r for r, _ in poblacion], th, todo_verde=todo_verde)
    for r, previo in poblacion:
        for n in NIVELES_TIPO:
            r[n + '_regla'] = r.get(n, '')
            if n in previo:
                r[n] = previo[n]                  # lo revisado manda
        _marcar_revisados(r, list(previo))
        r['tipo_revisado_fecha'] = (_fecha_txt(r.get('tipo_revisado_fecha'))
                                    if previo else '')
        r['tipo_borreguil'] = _tipo_compuesto(r)
    return invalidos


def _revisar(r, nuevos, cuando):
    """Común a set_tipo_revisado y set_niveles_revisados: valores ya validados de los
    niveles que se dan por revisados."""
    for n in NIVELES_TIPO:
        if (n + '_regla') not in r:               # sesión anterior a este campo:
            r[n + '_regla'] = _txt(r.get(n))      # lo vigente era la propuesta
    revisados = niveles_revisados(r)
    for n, v in nuevos.items():
        r[n] = v
    _marcar_revisados(r, [n for n in NIVELES_TIPO if n in revisados or n in nuevos])
    if cuando is None:
        from datetime import datetime
        cuando = datetime.now().isoformat(timespec='seconds')
    r['tipo_revisado_fecha'] = cuando
    r['tipo_borreguil'] = _tipo_compuesto(r)
    r.pop('tipo_revision_invalida', None)
    return r


def _validar(nuevos):
    for n, v in nuevos.items():
        if v is not None and v not in NIVELES_TIPO[n]:
            raise ValueError(f'{n}: «{v}» no es una categoría admitida '
                             f'({", ".join(NIVELES_TIPO[n])})')


def set_tipo_revisado(r, ambiente=None, humedad=None, pureza=None, cuando=None):
    """Registra la revisión humana del tipo ENTERO de un punto: los tres niveles.

    Los niveles que no se indican conservan su valor vigente (así se confirma la
    propuesta sin cambiarla). Lanza ValueError si una categoría no es admitida o si,
    tras aplicar los cambios, queda algún nivel sin categoría. Para revisar solo
    algunos niveles, set_niveles_revisados.
    """
    nuevos = {'ambiente': ambiente, 'humedad': humedad, 'pureza': pureza}
    _validar(nuevos)
    final = {n: (v if v is not None else _txt(r.get(n))) for n, v in nuevos.items()}
    faltan = [n for n, v in final.items() if v not in NIVELES_TIPO[n]]
    if faltan:
        raise ValueError('falta elegir ' + ', '.join(faltan))
    return _revisar(r, final, cuando)


def set_niveles_revisados(r, cuando=None, **valores):
    """Registra la revisión humana de ALGUNOS niveles del tipo: solo los que se dan
    (p. ej. humedad='seco') pasan a contar como revisados; los demás siguen como
    estaban, revisados o no. Lanza ValueError si una categoría no es admitida o si no
    se da ningún nivel."""
    nuevos = {n: v for n, v in valores.items() if v is not None}
    desconocidos = [n for n in nuevos if n not in NIVELES_TIPO]
    if desconocidos:
        raise ValueError('nivel desconocido: ' + ', '.join(desconocidos))
    if not nuevos:
        raise ValueError('no se ha elegido ningún nivel')
    _validar(nuevos)
    return _revisar(r, nuevos, cuando)


def clear_tipo_revisado(r):
    """Deshace la revisión: el punto vuelve a la propuesta de la app."""
    for n in NIVELES_TIPO:
        if (n + '_regla') in r:
            r[n] = r[n + '_regla']
    _marcar_revisados(r, [])
    r['tipo_revisado_fecha'] = ''
    r['tipo_borreguil'] = (_tipo_compuesto(r)
                           if any(_txt(r.get(n)) for n in NIVELES_TIPO) else '')
    return r


def resumen_revision(rows, es_borreguil):
    """Cuántos borreguiles tienen el tipo revisado y en cuántos se corrigió la
    propuesta de la app. Si se revisan muchos y no se corrige casi ninguno, o las
    reglas aciertan o la revisión no es real: conviene tenerlo a la vista."""
    borr = [r for r in rows if es_borreguil(r.get('decision', '')) or tipo_revisado(r)]
    rev = [r for r in borr if tipo_revisado(r)]
    por_nivel = {n: 0 for n in NIVELES_TIPO}
    revisados_nivel = {n: 0 for n in NIVELES_TIPO}
    n_corr = 0
    for r in rev:
        cambiado = False
        for n in niveles_revisados(r):
            revisados_nivel[n] += 1
            regla = _txt(r.get(n + '_regla'))
            if regla in NIVELES_TIPO[n] and regla != _txt(r.get(n)):
                por_nivel[n] += 1
                cambiado = True
        n_corr += cambiado
    return {'n_borreguil': len(borr), 'n_revisados': len(rev),
            'n_corregidos': n_corr, 'por_nivel': por_nivel,
            'revisados_por_nivel': revisados_nivel}


def run_rf_and_decide(rows, threshold=0.5, default_model_path=None, neg_buffer_m=250,
                      propio=False, entrenamiento=None):
    """Entrena/predice el RF y asigna la decisión a cada punto.

    Se usa también tras verificar, añadir o quitar puntos: como las features ya
    están en `rows`, sólo se reentrena el clasificador (si hace falta) y se vuelve
    a predecir, lo que tarda segundos. Devuelve el dict `info` de train_rf.
    """
    proba, info = train_rf(rows, default_model_path=default_model_path,
                           neg_buffer_m=neg_buffer_m, propio=propio,
                           entrenamiento=entrenamiento)
    if proba is not None:
        for i, r in enumerate(rows):
            r['rf_proba'] = float(proba[i])
    else:
        for r in rows:
            r.setdefault('rf_proba', float('nan'))
    for r in rows:
        r['decision'] = decide(r, threshold=threshold)
    return info


# ============================================================
# OUTPUTS
# ============================================================
def save_csv(rows, out):
    out = Path(out)
    if not rows: return
    keys = list(rows[0].keys())
    # Ensure all keys present
    all_keys = set()
    for r in rows: all_keys.update(r.keys())
    for k in all_keys:
        if k not in keys: keys.append(k)
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader(); w.writerows(rows)
    print(f'  → CSV: {out}')

def save_points_geojson(rows, out):
    """Puntos con sus etiquetas en GeoJSON (EPSG:4326), para volver a cargarlos en
    otra sesión sin perder lo hecho a mano (verdad-terreno y tipo revisado).

    Las etiquetas van con las columnas de la plantilla de datos de campo, las mismas
    que se piden a quien sube sus puntos:
      presence      1 borreguil · 0 no borreguil · vacío sin etiqueta
      doubtful      1 si es dudoso
      environment   stream · lake · slope           } solo los niveles que ha
      moisture      wet · dry                         } revisado una persona; la
      purity        pure · mixed-water · mixed-rock   } propuesta de la app no va
      method        field · orthophoto (lo revisado en la app es con ortofoto)
      date          fecha de la observación o de la revisión (AAAA-MM-DD)
    Además, `ID`, `decision` (lo que dijo el detector), la cuenca, la probabilidad y
    los demás atributos de texto del fichero original. Las variables predictoras NO,
    vengan como número o como texto (en un KML todo es texto): al recargar se vuelven
    a extraer, y un valor antiguo guardado aquí podría quedarse en lugar de uno que
    falle al descargarse. Para la tabla completa están el CSV y el Excel.
    """
    import json
    variables = set(FEATURES_RF_COMBO) | set(CLAVES_PUREZA_PUNTO)
    # Las etiquetas internas no salen: van traducidas a las columnas de la plantilla
    internas = set(CAMPOS_REVISION) | {'Borreguil', 'Duda'}
    feats = []
    for r in rows:
        props = {k: v for k, v in r.items()
                 if isinstance(v, str) and k not in ('lon', 'lat') and k not in variables
                 and k not in internas
                 and not k.startswith('_')        # anotaciones internas de la sesión
                 # 'truth' marca los puntos añadidos desde el fichero de campo en ESTA
                 # sesión; recargado, los ocultaría la opción «solo para entrenar».
                 and not (k == 'source' and v == 'truth')}
        props['ID'] = _txt(r.get('ID'))
        props['decision'] = _txt(r.get('decision'))
        origen = _txt(props.pop('origin', '')).lower()
        campo = _txt(r.get('Borreguil')).lower()
        props['presence'] = {'si': 1, 'no': 0}.get(campo)
        props['doubtful'] = 1 if _es_si(r.get('Duda')) else None
        revisados = niveles_revisados(r)
        for n, col in COLUMNA_TIPO_EN.items():
            props[col] = TIPO_EN.get(_txt(r.get(n))) if n in revisados else None
        fecha = _fecha_txt(r.get('tipo_revisado_fecha'))
        if not _txt(r.get('method')) and origen == 'verificado':   # verificado en la app
            props['method'] = 'orthophoto'
        if not _txt(r.get('date')) and fecha:
            props['date'] = fecha[:10]
        cuenca = r.get('cuenca_id')               # agrupa la validación espacial
        if isinstance(cuenca, (int, float)) and not isinstance(cuenca, bool) \
                and cuenca == cuenca:
            props['cuenca_id'] = int(cuenca) if float(cuenca).is_integer() else float(cuenca)
        pr = r.get('rf_proba')
        props['rf_proba'] = (round(float(pr), 4)
                             if isinstance(pr, (int, float)) and pr == pr else None)
        feats.append({'type': 'Feature',
                      'geometry': {'type': 'Point',
                                   'coordinates': [float(r['lon']), float(r['lat'])]},
                      'properties': props})
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'type': 'FeatureCollection', 'features': feats}, f,
                  ensure_ascii=False)
    print(f'  → GeoJSON de puntos: {out}')


def save_xlsx(rows, out, threshold=0.5):
    from openpyxl import Workbook
    from openpyxl.styles import PatternFill, Font, Alignment
    from openpyxl.utils import get_column_letter
    from openpyxl.formatting.rule import ColorScaleRule

    wb = Workbook(); ws = wb.active; ws.title = 'Puntos'
    COLS = [
        ('#',         lambda r,i: i+1),
        ('ID',        lambda r,i: r.get('ID','')),
        ('lon',       lambda r,i: float(r['lon'])),
        ('lat',       lambda r,i: float(r['lat'])),
        ('Cuenca',    lambda r,i: r.get('cuenca_id','')),
        ('Verdad campo', lambda r,i: r.get('Borreguil','')),
        ('Duda',      lambda r,i: r.get('Duda','')),
        ('Revisado por', lambda r,i: r.get('observer','')),
        ('Perfil',    lambda r,i: PERFIL_ES.get(r.get('expertise'), '')),
        ('Certidumbre', lambda r,i: CERTIDUMBRE_ES.get(r.get('certainty'), '')),
        ('DECISIÓN',  lambda r,i: r.get('decision','')),
        ('RF prob.',  lambda r,i: r.get('rf_proba', None)),
        ('Ambiente',  lambda r,i: r.get('ambiente','')),
        ('Humedad',   lambda r,i: r.get('humedad','')),
        ('Pureza',    lambda r,i: r.get('pureza','')),
        ('Tipo revisado', lambda r,i: ', '.join(niveles_revisados(r))),
        ('Fecha revisión', lambda r,i: r.get('tipo_revisado_fecha','')),
        ('Ambiente (propuesta app)', lambda r,i: r.get('ambiente_regla','')),
        ('Humedad (propuesta app)',  lambda r,i: r.get('humedad_regla','')),
        ('Pureza (propuesta app)',   lambda r,i: r.get('pureza_regla','')),
        ('Patrón',    lambda r,i: r.get('mat_signature','')),
        ('Altitud (m)', lambda r,i: r.get('elev_dem_m', None)),
        ('Slope (°)',   lambda r,i: r.get('slope_deg', None)),
        ('Dist. cauce (m)', lambda r,i: r.get('dist_cauce_m', None)),
        ('NDVI fin verano', lambda r,i: r.get('ndvi_late', None)),
        ('Clre fin verano', lambda r,i: r.get('clre_late', None)),
        ('NDMI fin verano', lambda r,i: r.get('ndmi_late', None)),
        ('EVI fin verano',  lambda r,i: r.get('evi_late', None)),
        ('Verde sat. %',    lambda r,i: 100*r['frac_bgreen'] if 'frac_bgreen' in r else None),
        ('Homog. GLCM',     lambda r,i: r.get('glcm_homog', None)),
        ('Roca circund. %', lambda r,i: 100*r['surr_rock'] if 'surr_rock' in r else None),
    ]
    hdr_fill = PatternFill('solid', fgColor='37474F')
    for i, (nm, _) in enumerate(COLS, 1):
        c = ws.cell(row=1, column=i, value=nm)
        c.fill = hdr_fill; c.font = Font(bold=True, color='FFFFFF')
        ws.column_dimensions[get_column_letter(i)].width = 16
    ws.freeze_panes = 'C2'

    anchos = [len(nm) for nm, _ in COLS]
    for ri, r in enumerate(rows, 2):
        for ci, (nm, fn) in enumerate(COLS, 1):
            try:
                v = fn(r, ri-2)
            except Exception:
                v = None
            if isinstance(v, float) and math.isnan(v): v = None
            c = ws.cell(row=ri, column=ci, value=v)
            if isinstance(v, float):
                # Solo cambia lo que se VE (3 decimales; 6 en las coordenadas, que con
                # 3 se moverían unos 100 m): la celda conserva el valor completo.
                nd = 6 if nm in ('lon', 'lat') else 3
                c.number_format = '0.' + '0' * nd
                visto = f'{v:.{nd}f}'
            else:
                visto = '' if v is None else str(v)
            anchos[ci-1] = max(anchos[ci-1], len(visto))
    # Cada columna, tan ancha como su contenido: los nombres largos (p. ej. un ID
    # heredado del nombre de la capa) no quedan cortados.
    for ci, ancho in enumerate(anchos, 1):
        ws.column_dimensions[get_column_letter(ci)].width = min(60, max(8, ancho + 2))

    last_row = len(rows)+1
    def col(name):
        return get_column_letter(next(i for i,(n,_) in enumerate(COLS,1) if n==name))
    ws.conditional_formatting.add(f'{col("RF prob.")}2:{col("RF prob.")}{last_row}',
        ColorScaleRule(start_type='num', start_value=0.1, start_color='8E0000',
                        mid_type='num', mid_value=0.5, mid_color='FFEB3B',
                        end_type='num', end_value=0.9, end_color='1B5E20'))
    ws.auto_filter.ref = f'A1:{get_column_letter(len(COLS))}{last_row}'
    wb.save(out)
    print(f'  → Excel: {out}')

def fmt_num(v, nd=3):
    """Número con `nd` decimales para MOSTRARLO ('—' si falta o no es un número).
    Los valores se guardan con toda su precisión; esto solo cambia lo que se ve."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return '—'
    return '—' if x != x else f'{x:.{nd}f}'


def abreviar_ids(ids, largo=24):
    """Versión corta de unos ID para MOSTRARLOS en una tabla.

    Los ID heredados del nombre de la capa pueden ser muy largos y casi iguales
    ('all_borreguil_lagunas500ptos_epsg25830.41'): ocupan media tabla y lo único
    que los distingue está al final. A los largos se les quita el arranque que
    comparten, cortando en un separador para dejar algo de contexto:
    '…epsg25830.41'. Un ID largo suelto se acorta por el medio. Los cortos no se
    tocan y, si al acortar dos ID distintos quedaran iguales, no se acorta ninguno.
    El ID de verdad no cambia: sigue completo en el punto, en el mapa y en las
    descargas.
    """
    txt = ['' if x is None else str(x) for x in ids]
    largos = sorted({t for t in txt if len(t) > largo})
    if not largos:
        return txt

    def por_el_medio(t):
        return t if len(t) <= largo else t[:6] + '…' + t[-(largo - 7):]

    corte = 0
    if len(largos) >= 2:
        comun = largos[0]
        for t in largos[1:]:
            while not t.startswith(comun):
                comun = comun[:-1]
        # Se corta en un separador anterior al final de lo común, para que quede
        # '…epsg25830.41' y no un escueto '…41'.
        separadores = [i for i, ch in enumerate(comun[:-1]) if ch in '_-. /']
        corte = separadores[-1] + 1 if separadores else 0

    def corta(t):
        if len(t) <= largo:
            return t
        return por_el_medio('…' + t[corte:]) if corte >= 8 else por_el_medio(t)

    cortos = [corta(t) for t in txt]
    return txt if len(set(cortos)) < len(set(txt)) else cortos


def nombres_de_punto(ids, punto='Punto'):
    """Cómo se llama cada punto en el GLOBO del mapa (25-11-2026, la autora: «este
    nombre de la capa cámbialo»).

    Los ID heredados del nombre de la capa ('all_borreguil_lagunas500ptos_epsg25830.663')
    pasan a 'Punto 663': la capa que más se repite entre los puntos se quita y queda
    su número. Los demás ID (cortos, o de otra capa) se quedan como están. Si al
    quitarla dos puntos distintos se llamaran igual, se dejan todos sin cambiar. El ID
    de verdad no cambia: sigue completo en el punto, en la tabla y en las descargas."""
    import re
    from collections import Counter
    txt = ['' if x is None else str(x) for x in ids]
    partes = [re.match(r'^(.{13,})\.(\d+)$', t) for t in txt]
    capas = Counter(m.group(1) for m in partes if m)
    if not capas:
        return txt
    capa = capas.most_common(1)[0][0]
    cortos = [f'{punto} {m.group(2)}' if m and m.group(1) == capa else t
              for t, m in zip(txt, partes)]
    return txt if len(set(cortos)) < len(set(txt)) else cortos


def nombre_html(nombre):
    """Un nombre listo para HTML: escapado y con un punto de corte tras cada
    separador, de modo que uno largo se parta por ahí ('…500ptos_' / 'epsg25830.41')
    y no por cualquier letra. No añade caracteres al texto: si se copia de la
    pantalla, sale el nombre exacto."""
    import html
    import re
    return re.sub(r'([_\-./])', r'\1<wbr>', html.escape(str(nombre)))


def _frase(texto):
    """«BORREGUIL VERIFICADO» → «Borreguil verificado» (solo para verlo)."""
    texto = str(texto)
    return texto[:1].upper() + texto[1:].lower()


def popup_html(r, traduce=None, textos=None, nombre=None, variables=None, num=None):
    """Globo de un punto en el mapa (el de la app y el de mapa.html).

    - Los números, con 3 decimales.
    - Los nombres largos sin espacios (p. ej. un ID heredado del nombre de la capa,
      'all_borreguil_lagunas500ptos_epsg25830.21') se parten en varias líneas en
      lugar de salirse del globo.
    - Los textos que vienen del fichero se escapan: un atributo con «<» no rompe
      el mapa.
    `traduce` traduce los valores (decisión, tipo) y `textos` los rótulos.

    En la app (25-11-2026, la autora): `nombre`, el nombre que se enseña (ver
    nombres_de_punto; si no, el ID); `variables`, [(rótulo, columna)] de las que se
    dan, una por línea (las que más pesan en el modelo, como en la tabla); `num`, cómo
    se escribe un número (con coma en español). Sin `variables`, las de siempre:
    altitud, pendiente, NDVI y Clre.
    """
    import html
    t = {'tipo': 'Tipo', 'revisado': 'revisado', 'altitud': 'Altitud',
         'pendiente': 'Pendiente'}
    t.update(textos or {})
    tv = traduce or (lambda x: x)

    def esc(x):
        return html.escape(str(x))

    rf = r.get('rf_proba')
    rf_txt = f'{rf * 100:.0f}%' if isinstance(rf, (int, float)) and rf == rf else '—'
    tipo = _txt(r.get('tipo_borreguil'))
    tipo_html = ''
    if tipo:
        marca = f" ✓ {t['revisado']}" if tipo_revisado(r) else ''
        tipo_html = f"<br>🌿 {t['tipo']}: <b>{esc(tv(tipo))}</b>{marca}"
    num = num or fmt_num
    if variables:
        # que se sepa por qué salen esas (25-11-2026, la autora: «¿se indica en algún
        # sitio que el globo enseña las 3 variables que más pesan en el modelo?»)
        lineas = (f"<i>{esc(t.get('variables', 'Las que más pesan en el modelo'))}:</i><br>"
                  + '<br>'.join(f'{esc(etq)}: {num(r.get(col))}' for etq, col in variables))
    else:
        # (sin el «Patrón» BORREGUIL_LIKE…: era un código interno)
        lineas = (f"{t['altitud']}: {num(r.get('elev_dem_m'))} m · "
                  f"{t['pendiente']}: {num(r.get('slope_deg'))}°<br>"
                  f"NDVI: {num(r.get('ndvi_late'))} · Clre: {num(r.get('clre_late'))}")
    return (
        '<div style="max-width:280px;overflow-wrap:anywhere;word-break:break-word;">'
        f"<b>{nombre_html(r.get('ID', '?') if nombre is None else nombre)}</b><br>"
        f"<b>{esc(_frase(tv(_txt(r.get('decision')) or 'SIN PREDICCIÓN')))}</b><br>"
        f"RF: {rf_txt}{tipo_html}<br>"
        + lineas +
        '</div>')


def save_map(rows, out, study_geom=None):
    import folium
    from shapely.geometry import mapping
    if not rows: return
    lat_med = sum(r['lat'] for r in rows)/len(rows)
    lon_med = sum(r['lon'] for r in rows)/len(rows)
    m = folium.Map(location=[lat_med, lon_med], zoom_start=12, tiles=None,
                   max_zoom=22, control_scale=True)
    folium.TileLayer('https://services.arcgisonline.com/ArcGIS/rest/services/'
                     'World_Imagery/MapServer/tile/{z}/{y}/{x}',
                     attr='Esri', name='ESRI World Imagery',
                     max_zoom=22, max_native_zoom=19).add_to(m)
    folium.TileLayer('https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
                     attr='Google', name='Google Satélite (más zoom)',
                     max_zoom=22, max_native_zoom=21).add_to(m)
    folium.TileLayer('OpenStreetMap', name='OpenStreetMap', max_zoom=19).add_to(m)
    # Capas de rejilla de muestreo S2/S1 (desactivables)
    try:
        fps = pixel_footprints(rows[:600])
        g_pix = folium.FeatureGroup(name='▫ Píxel S2 (10 m)', show=False)
        g_w10 = folium.FeatureGroup(name='▢ Ventana muestreo 10 m (S2 + S1)', show=False)
        g_w20 = folium.FeatureGroup(name='▢ Ventana muestreo 20 m (S2)', show=False)
        for fp in fps:
            folium.Polygon(fp['pixel10'], color='#FFD600', weight=1, fill=True,
                           fill_color='#FFD600', fill_opacity=0.12).add_to(g_pix)
            folium.Polygon(fp['win_s2_10'], color='#00E5FF', weight=1, fill=False).add_to(g_w10)
            folium.Polygon(fp['win_s2_20'], color='#FF6D00', weight=1, fill=False,
                           dash_array='4').add_to(g_w20)
        for g in (g_pix, g_w10, g_w20): g.add_to(m)
    except Exception:
        pass
    # Study area polygon
    if study_geom is not None:
        folium.GeoJson(mapping(study_geom), name='Área de estudio',
                       style_function=lambda x: {'fillColor':'#1976D2',
                                                  'color':'#0D47A1',
                                                  'weight': 2,
                                                  'fillOpacity': 0.08}
                       ).add_to(m)
    for r in rows:
        color = decision_color(r.get('decision', 'SIN PREDICCIÓN'))
        folium.CircleMarker([r['lat'], r['lon']], radius=5,
                            color=color, fill=True, fill_color=color, fill_opacity=0.8,
                            popup=folium.Popup(popup_html(r), max_width=300)).add_to(m)
    folium.LayerControl().add_to(m)
    m.save(out)
    print(f'  → Mapa: {out}')


# ============================================================
# MAIN
# ============================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', help='KML/Shp/GeoJSON con puntos candidatos. Opcional si se usa --generate.')
    ap.add_argument('--generate', type=int, help='Generar N puntos aleatorios dentro del área de estudio (requiere --study-area/--wdpa/--name)')
    ap.add_argument('--sample-mode', default='stratified_elev', choices=['uniform','poisson_disc','stratified_elev'], help='Estrategia de muestreo (default: stratified_elev)')
    ap.add_argument('--min-elev', type=float, help='Elevación mínima para los puntos generados (m)')
    ap.add_argument('--max-elev', type=float, help='Elevación máxima para los puntos generados (m)')
    ap.add_argument('--min-dist', type=float, default=100, help='Distancia mínima entre puntos para poisson_disc (m)')
    ap.add_argument('--seed', type=int, default=42, help='Semilla aleatoria (default: 42)')
    ap.add_argument('--truth', help='KML/Shp con borreguiles verificados en campo (presence-only: todos se asumen borreguil)')
    ap.add_argument('--reference', help='KML/Shp con inventario existente (opcional)')
    ap.add_argument('--study-area', help='Vector (KML/Shp/GeoJSON) con el polígono del área de estudio. Si no se aporta, se intenta WDPA o nombre.')
    ap.add_argument('--wdpa', help='ID WDPA (protectedplanet.net) — necesita WDPA_TOKEN env var o área conocida en cache')
    ap.add_argument('--name', help='Nombre del área protegida (búsqueda Nominatim sin token)')
    ap.add_argument('--output', default='./output', help='Carpeta destino (default ./output)')
    ap.add_argument('--backend', default='mpc', choices=['mpc','gee'],
                    help='Fuente de datos satelitales: mpc (Microsoft Planetary Computer, sin cuenta) o gee (Google Earth Engine, requiere cuenta)')
    ap.add_argument('--gee-project', help='Nombre del proyecto Google Earth Engine (p. ej. ee-tunombre). Requerido con --backend gee')
    ap.add_argument('--years', default='2017-2025', help='Años Sentinel-2; admite rangos (default 2017-2025)')
    ap.add_argument('--skip-imgs', action='store_true', help='No descargar imágenes ESRI')
    ap.add_argument('--skip-s2', action='store_true', help='No descargar Sentinel-2')
    # [obsoletas] los cauces de OpenStreetMap se usan siempre (ver cauces_osm)
    ap.add_argument('--use-osm', action='store_true', help=argparse.SUPPRESS)
    ap.add_argument('--no-overpass', action='store_true', help=argparse.SUPPRESS)
    ap.add_argument('--threshold', type=float, default=0.5, help='Umbral RF (default 0.5)')
    ap.add_argument('--modelo-propio', action='store_true',
                    help='Entrenar también un modelo con los puntos de campo (--truth) y '
                         'quedarse con el que más acierte; por defecto se usa el modelo de partida')
    args = ap.parse_args()

    _imports()  # lazy import

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    img_dir = out / 'imgs'

    print('=== borreguil_pipeline ===')

    # Validate inputs
    if not args.input and not args.generate:
        ap.error('Debes aportar --input (KML de puntos) o --generate N (puntos aleatorios).')
    if args.generate and not (args.study_area or args.wdpa or args.name):
        ap.error('--generate requiere también --study-area, --wdpa o --name para definir el área.')

    # Resolve study area FIRST (needed by --generate)
    study_geom = None; study_info = None
    if args.study_area or args.wdpa or args.name:
        try:
            import study_area as sa
            study_geom, study_info = sa.resolve(args.study_area, args.wdpa, args.name)
            if study_geom:
                print(f'Área de estudio: {study_info["source"]}')
                if 'name' in study_info: print(f'  {study_info["name"]}')
                print(f'  área aprox: {study_info["area_km2"]:.3f} km²')
        except Exception as e:
            print(f'Resolución de área falló: {e}')

    # Load or generate points
    if args.generate:
        if not study_geom:
            ap.error('No se pudo resolver el área de estudio. No se pueden generar puntos.')
        import random_points
        rows, _gstats = random_points.generate(study_geom, args.generate,
                                       mode=args.sample_mode,
                                       min_elev=args.min_elev,
                                       max_elev=args.max_elev,
                                       min_dist_m=args.min_dist,
                                       seed=args.seed)
        if not rows:
            amax = _gstats.get('area_elev_max'); amin = _gstats.get('area_elev_min')
            if _gstats.get('reason') == 'min_elev_above_area' and amax is not None:
                ap.error(f'El área no alcanza la altitud mínima ({args.min_elev:.0f} m). '
                         f'Altitud máxima del área ≈ {amax:.0f} m. Baja --min-elev.')
            elif _gstats.get('reason') == 'max_elev_below_area' and amin is not None:
                ap.error(f'El área supera la altitud máxima ({args.max_elev:.0f} m). '
                         f'Altitud mínima del área ≈ {amin:.0f} m. Sube --max-elev.')
            elif _gstats.get('reason') == 'no_elevation_data':
                ap.error('No se pudo obtener la altitud del área (¿zona de agua o fallo '
                         'de la API de elevación?). Prueba sin --min-elev/--max-elev.')
            else:
                ap.error('No se generaron puntos. Revisa el área y los filtros de altitud.')
        print(f'  {len(rows)} puntos generados')
    else:
        print(f'Input: {args.input}')
        rows = read_points(args.input)
        print(f'  {len(rows)} puntos cargados')
        for nivel, texto in resumen_etiquetas(leer_etiquetas(rows)):
            print(f"  {'AVISO: ' if nivel == 'aviso' else ''}{texto.replace('`', '')}")

    # Merge ground truth (PRESENCE-ONLY: todos los puntos = borreguil)
    if args.truth:
        tr = read_points(args.truth)
        leido = {}
        n_match, n_add, n_pos, n_neg = merge_truth_points(rows, tr, info=leido)
        if leido.get('solo_presencias'):
            print('  Puntos de campo sin columna de etiqueta: todos se toman como borreguil')
        else:
            for nivel, texto in resumen_etiquetas(leido):
                print(f"  {'AVISO: ' if nivel == 'aviso' else ''}{texto.replace('`', '')}")
        print(f'  Verdad-terreno: {n_pos} borreguiles + {n_neg} no-borreguiles '
              f'({n_match} coinciden con candidatos, {n_add} añadidos como nuevos)')
        print(f'    Total puntos tras añadir verdad-terreno: {len(rows)}')

    # Mark points inside the study area (if defined)
    if study_geom:
        try:
            import study_area as sa
            n_in = sa.points_inside(rows, study_geom)
            print(f'  puntos dentro del área: {n_in}/{len(rows)}')
        except Exception:
            pass

    if study_geom:
        bbox = bbox_from_geom(study_geom)
        print(f'BBox (área de estudio): {bbox}')
    else:
        bbox = bbox_with_buffer(rows)
        print(f'BBox (derivada de puntos): {bbox}')

    # Backend satelital
    backend = args.backend
    if backend == 'gee':
        import gee_backend as geb
        if not geb.gee_available():
            print('  ⚠ earthengine-api no instalado; usando MPC.')
            backend = 'mpc'
        else:
            ok, msg = geb.initialize(args.gee_project)
            print(f'  GEE: {msg}')
            if not ok:
                print('  ⚠ No se pudo inicializar GEE; usando MPC. '
                      '(ejecuta `earthengine authenticate` o revisa --gee-project)')
                backend = 'mpc'

    years = parse_years(args.years)

    # 1. ESRI images (siempre desde ESRI; aporta textura/patrón)
    if not args.skip_imgs:
        print('\n[1/6] Imágenes ESRI…')
        fetch_esri_imgs(rows, img_dir)
        print('[2/6] Features de imagen…')
        compute_img_features(rows, img_dir)
    else:
        print('Imágenes ESRI saltadas (--skip-imgs)')

    # 3. Cauces de OpenStreetMap (ríos y arroyos), para el ambiente «arroyo»
    print('\n[3/6] Cauces (OpenStreetMap)…')
    lineas, origen = cauces_osm(bbox)
    n_c = attach_cauce_distance(rows, lineas)
    if lineas is None:
        print('  No se han podido leer: «arroyo» se decide solo por la pendiente.')
    else:
        print(f'  {n_c} ríos y arroyos ({"incluidos en la app" if origen == "capa" else "de Overpass"})')

    # 4. Topografía
    print(f'\n[4/6] Topografía DEM ({backend})…')
    try:
        if backend == 'gee':
            import gee_backend as geb
            geb.fetch_topo_gee(rows, bbox)
        else:
            fetch_topo(rows, bbox, out)
    except Exception as e:
        print(f'  Topografía falló: {e}')

    # 5. Sentinel-2
    if not args.skip_s2:
        print(f'\n[5/6] Sentinel-2 ({backend})…')
        try:
            if backend == 'gee':
                import gee_backend as geb
                geb.fetch_s2_gee(rows, bbox, years=years)
            else:
                fetch_s2(rows, bbox, years=years)
        except Exception as e:
            print(f'  Sentinel-2 falló: {e}')

        # Sentinel-1 (radar): solo con Earth Engine
        if backend == 'gee':
            try:
                import gee_backend as geb
                geb.fetch_s1_gee(rows, bbox)
            except Exception as e:
                print(f'  Sentinel-1 falló (se continúa sin radar): {e}')
        else:
            print('  Sentinel-1 (radar) solo se lee con Earth Engine: se usa la versión '
                  'del modelo sin radar.')

    # 6. Random Forest (sin radar: la versión del modelo entrenada sin él)
    print('\n[6/6] Random Forest…')
    _modelo = Path(__file__).parent / 'rf_sierra_nevada.joblib'
    if not hay_radar(rows) and ruta_sin_radar(_modelo) is not None:
        _modelo = ruta_sin_radar(_modelo)
    proba, info = train_rf(rows, default_model_path=_modelo, propio=args.modelo_propio)
    if proba is not None:
        for i, r in enumerate(rows):
            r['rf_proba'] = float(proba[i])
    else:
        for r in rows: r['rf_proba'] = float('nan')

    # Decisión
    for r in rows:
        r['decision'] = decide(r, threshold=args.threshold)

    # Outputs
    print('\nGuardando…')
    save_csv(rows, out / 'classification.csv')
    save_xlsx(rows, out / 'Clasificacion_puntos.xlsx', threshold=args.threshold)
    save_map(rows, out / 'mapa.html', study_geom=study_geom)
    # If study area defined: also export it as GeoJSON
    if study_geom is not None:
        from shapely.geometry import mapping
        with open(out / 'area_estudio.geojson', 'w') as f:
            json.dump({'type':'Feature','properties':study_info or {},
                        'geometry': mapping(study_geom)}, f)
        print(f'  → Área estudio: {out / "area_estudio.geojson"}')

    # Resumen final
    print('\n=== RESUMEN ===')
    print(f'Puntos clasificados: {len(rows)}')
    dec_count = Counter(r['decision'] for r in rows)
    for k, v in dec_count.most_common():
        print(f'  {v:>4}  {k}')
    if info:
        if info.get('mode') == 'pretrained':
            print(f'\nModelo: preentrenado en {info.get("training_zone","?")}  '
                  f'(n_train_original = {info.get("training_n","?")}, '
                  f'AUC GroupKFold reportado: {info.get("auc_groupkfold","?")})')
            _ac = aciertos_en_campo(info.get('evaluacion'), args.threshold).get('partida')
            if _ac and (_ac['n_si'] or _ac['n_no']):
                print(f'Con los puntos de campo: {_ac["ok_si"]} de {_ac["n_si"]} borreguiles '
                      f'y {_ac["ok_no"]} de {_ac["n_no"]} ausencias acertados'
                      + (f' (AUC {_ac["auc"]:.3f})' if _ac['auc'] is not None else ''))
        elif info.get('mode') == 'trained':
            print(f'\nModelo: entrenado con verdad-terreno (presence-only)  '
                  f'AUC {info.get("cv","")} = {info["auc"]:.3f}  '
                  f'({info["n_pos"]} positivos; negativos: {info["n_neg_campo"]} de campo '
                  f'+ {info["n_pseudo"]} pseudo-ausencias)')

if __name__ == '__main__':
    main()
