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
    --use-osm        Usar OSM (por defecto se OMITE; Overpass es inestable)

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
    import rasterio
    from pystac_client import Client
    import planetary_computer
    import pyproj
    from scipy import ndimage as ndi
    from skimage.feature import graycomatrix, graycoprops
    try:
        from skimage.morphology import opening, disk        # skimage ≥0.26
    except ImportError:
        from skimage.morphology import binary_opening as opening, disk  # versiones antiguas
    from skimage.color import rgb2gray
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
        for pm in root.findall('.//kml:Placemark', NS):
            d = {sd.get('name'): sd.text for sd in pm.findall('.//kml:SimpleData', NS)}
            coord = pm.find('.//kml:coordinates', NS)
            if coord is None: continue
            parts = coord.text.strip().split(',')
            d['lon'] = float(parts[0]); d['lat'] = float(parts[1])
            if 'ID' not in d and 'id' in d:
                d['ID'] = d['id']
            elif 'ID' not in d:
                d['ID'] = pm.get('id', f'pt_{len(rows)}')
            rows.append(d)
        return rows
    else:
        gdf = gpd.read_file(path).to_crs(4326)
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


# Valores que marcan presencia / ausencia en la verdad-terreno (insensible a
# mayúsculas/acentos). Atributos reconocidos: Borreguil, presencia/presence,
# clase/class, label, tipo, y.
_TRUTH_POS = {'si', 'sí', 'yes', 'y', '1', 'true', 'presencia', 'presence',
              'borreguil', 'positivo', 'pos', 'p'}
_TRUTH_NEG = {'no', 'n', '0', 'false', 'ausencia', 'absence', 'no_borreguil',
              'noborreguil', 'negativo', 'neg', 'ausente'}
_TRUTH_KEYS = ('Borreguil', 'borreguil', 'presencia', 'presence', 'clase',
               'class', 'label', 'tipo', 'y', 'gt', 'target')


def truth_label(d):
    """Devuelve 'si' (presencia) o 'no' (ausencia) para un punto de verdad-terreno.
    Reconoce varios nombres de atributo y valores (ver _TRUTH_POS/_TRUTH_NEG). Si no
    hay ningún atributo de clase, asume PRESENCIA ('si') — compatibilidad con el modo
    presence-only clásico (un fichero solo de borreguiles confirmados)."""
    for key in _TRUTH_KEYS:
        if key in d and d[key] is not None and str(d[key]).strip() != '':
            v = str(d[key]).strip().lower()
            if v in _TRUTH_NEG:
                return 'no'
            if v in _TRUTH_POS:
                return 'si'
    return 'si'


def merge_truth_points(rows, truth_rows, match_m=20):
    """Incorpora verdad-terreno con PRESENCIAS y (opcionalmente) AUSENCIAS.
    Cada punto de truth_rows se etiqueta con truth_label(): 'si' (borreguil) o 'no'
    (no-borreguil). Sin atributo de clase → presencia (modo presence-only clásico).
    Si un punto coincide (< match_m) con un candidato, etiqueta ese candidato; si no,
    se añade. Devuelve (n_matched, n_added, n_pos, n_neg)."""
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
        lab = truth_label(tr)
        if lab == 'si': n_pos += 1
        else: n_neg += 1
        nearest_i, d = None, 1e18
        if tree is not None:
            d, nearest_i = tree.query(txy)
        if nearest_i is not None and d <= match_m:
            rows[nearest_i]['Borreguil'] = lab
            rows[nearest_i]['is_truth'] = 1
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
# 3. OSM (hidrografía + infraestructura)
# ============================================================
def fetch_osm(rows, bbox):
    print('  → Descargando OSM (waterways, springs, aerialways, highways)…')
    south, west, north, east = bbox[1], bbox[0], bbox[3], bbox[2]
    headers = {'User-Agent': 'borreguil-pipeline/1.0'}
    URL = 'https://overpass-api.de/api/interpreter'

    def overpass(q):
        for _ in range(4):
            try:
                r = requests.post(URL, data={'data': q}, headers=headers, timeout=180)
                if r.status_code == 200:
                    return r.json()
                time.sleep(5)
            except Exception:
                time.sleep(5)
        return None

    qw = f'''[out:json][timeout:120];
    (way["waterway"]({south},{west},{north},{east});
     node["natural"="spring"]({south},{west},{north},{east}););
    out geom;'''
    qi = f'''[out:json][timeout:120];
    (way["aerialway"]({south},{west},{north},{east});
     way["highway"~"^(primary|secondary|tertiary|unclassified|residential|track|service)$"]({south},{west},{north},{east});
     way["building"]({south},{west},{north},{east});
     way["leisure"="ski"]({south},{west},{north},{east});
     way["piste:type"]({south},{west},{north},{east}););
    out geom;'''

    from shapely.geometry import Point, LineString
    from shapely.strtree import STRtree

    water_geoms = []; infra_geoms = []; infra_tags = []
    jw = overpass(qw)
    if jw:
        for el in jw['elements']:
            if el['type']=='way' and 'geometry' in el:
                cs = [(p['lon'],p['lat']) for p in el['geometry']]
                if len(cs)>=2: water_geoms.append(LineString(cs))
            elif el['type']=='node':
                water_geoms.append(Point(el['lon'], el['lat']))
    ji = overpass(qi)
    if ji:
        for el in ji['elements']:
            if 'geometry' in el and len(el['geometry'])>=2:
                cs = [(p['lon'],p['lat']) for p in el['geometry']]
                infra_geoms.append(LineString(cs))
                tags = el.get('tags', {})
                if 'aerialway' in tags or 'piste:type' in tags or tags.get('leisure')=='ski':
                    infra_tags.append('ski')
                elif 'building' in tags:
                    infra_tags.append('building')
                else:
                    infra_tags.append('road')
    print(f'    Hidrografía: {len(water_geoms)} elementos')
    print(f'    Infraestructura: {len(infra_geoms)} elementos ({dict(Counter(infra_tags))})')

    # Reproject to a local UTM
    lat_med = sum(r['lat'] for r in rows)/len(rows)
    lon_med = sum(r['lon'] for r in rows)/len(rows)
    utm_zone = int((lon_med + 180)/6) + 1
    epsg_utm = 32600 + utm_zone if lat_med >= 0 else 32700 + utm_zone
    trans = pyproj.Transformer.from_crs(4326, epsg_utm, always_xy=True)

    def reproj(g):
        from shapely.ops import transform as shp_transform
        return shp_transform(lambda x, y, z=None: trans.transform(x, y), g)

    water_utm = [reproj(g) for g in water_geoms]
    infra_utm = [reproj(g) for g in infra_geoms]
    ski_utm   = [g for g, t in zip(infra_utm, infra_tags) if t == 'ski']

    pts_utm = [trans.transform(r['lon'], r['lat']) for r in rows]
    from shapely.geometry import Point
    pts_geom = [Point(x, y) for x, y in pts_utm]

    tree_w = STRtree(water_utm) if water_utm else None
    tree_i = STRtree(infra_utm) if infra_utm else None
    tree_s = STRtree(ski_utm) if ski_utm else None

    for i, p in enumerate(pts_geom):
        rows[i]['dist_water_m'] = (p.distance(water_utm[tree_w.nearest(p)])
                                    if tree_w else float('nan'))
        rows[i]['dist_infra_m'] = (p.distance(infra_utm[tree_i.nearest(p)])
                                    if tree_i else float('nan'))
        rows[i]['dist_ski_m']   = (p.distance(ski_utm[tree_s.nearest(p)])
                                    if tree_s else float('nan'))


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

    # D8 flow accumulation (simplified, slow for very large rasters)
    h_, w_ = dem_utm.shape
    fdem = np.where(np.isnan(dem_utm), 1e6, dem_utm)
    offs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    dz = []
    for dy, dx in offs:
        nb = np.roll(np.roll(fdem, -dy, 0), -dx, 1)
        dz.append((fdem - nb)/math.sqrt(dy*dy+dx*dx))
    dz = np.stack(dz)
    fd = np.argmax(dz, axis=0)
    no_out = dz.max(0) <= 0
    accum = np.ones_like(fdem)
    order = np.argsort(-fdem.flatten())
    for idx in order:
        y, x = idx // w_, idx % w_
        if no_out[y, x]: continue
        dy, dx = offs[fd[y, x]]
        ny, nx = y+dy, x+dx
        if 0 <= ny < h_ and 0 <= nx < w_:
            accum[ny, nx] += accum[y, x]
    area = accum * res_y
    twi = np.log(area / np.tan(np.maximum(slope_rad, np.deg2rad(0.5))))

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
            r['twi'] = float(twi[row, col])
        else:
            for k in ('elev_dem_m','slope_deg','aspect_north','aspect_east','curvature','twi'):
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
                    out['evi'][i]  = 2.5*(b8-b4)/(b8+6*b4-7.5*b2+1+1e-6)
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


# ---- Sentinel-1 RTC (gamma-0, terrain-flattened) -----------------------------
# Backscatter de verano (jun–sep) terrain-corrected (ideal en montaña). Aporta una
# dimensión de humedad/estructura que el óptico no capta (+0.013 AUC honesto sobre
# las 45 features ópticas+topo). Se usa SIEMPRE vía MPC (independiente del backend
# S2) para que entrenamiento e inferencia compartan la misma fuente radiométrica.
S1_FEATS = ['s1_vv_mean', 's1_vh_mean', 's1_vv_sd', 's1_vh_sd', 's1_ratio_mean']

def fetch_s1(rows, bbox, years=(2019,2020,2021,2022,2023,2024), max_per_year=8):
    """Rellena s1_{vv,vh}_{mean,sd} y s1_ratio_mean (dB) por punto desde
    sentinel-1-rtc (MPC). Valores RTC en potencia lineal → se convierten a dB.
    Falla con elegancia: si no hay cobertura, deja las features en NaN (build_matrix
    las imputa con la mediana). El summer-mean de backscatter es temporalmente
    estable, así que el rango exacto de años no es crítico."""
    from rasterio.windows import Window
    _apply_gdal_http_env()
    print('  → Descargando Sentinel-1 RTC (terrain-flattened) vía MPC…')
    south, west, north, east = bbox[1], bbox[0], bbox[3], bbox[2]
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                          modifier=planetary_computer.sign_inplace)
    lat_med = sum(r['lat'] for r in rows)/len(rows)
    lon_med = sum(r['lon'] for r in rows)/len(rows)
    utm_zone = int((lon_med + 180)/6) + 1
    epsg_pts = 32600 + utm_zone if lat_med >= 0 else 32700 + utm_zone
    trans_pts = pyproj.Transformer.from_crs(4326, epsg_pts, always_xy=True)
    pts_xy = [trans_pts.transform(r['lon'], r['lat']) for r in rows]
    n = len(rows)
    vv_series = [[] for _ in range(n)]
    vh_series = [[] for _ in range(n)]

    def sample(item):
        # Lectura POR BLOQUE (igual que fetch_s2): una ventana del bbox por banda, no
        # una ventana por punto → el tiempo no crece con el nº de puntos.
        try:
            sv = rasterio.open(item.assets['vv'].href)
            sh = rasterio.open(item.assets['vh'].href)
        except Exception:
            return
        try:
            if sv.crs.to_epsg() != epsg_pts:
                t = pyproj.Transformer.from_crs(epsg_pts, sv.crs.to_epsg(), always_xy=True)
                tpts = [t.transform(x, y) for x, y in pts_xy]
            else:
                tpts = pts_xy
            H, Wd = sv.height, sv.width
            rc = [sv.index(X, Y) for X, Y in tpts]
            valid = [i for i, (r, c) in enumerate(rc) if 3 <= r < H-3 and 3 <= c < Wd-3]
            if not valid:
                return
            CELL = 2048
            cells = {}
            for i in valid:
                r, c = rc[i]
                cells.setdefault((r // CELL, c // CELL), []).append(i)
            for idxs in cells.values():
                rs = [rc[i][0] for i in idxs]; cs = [rc[i][1] for i in idxs]
                r0, r1, c0, c1 = min(rs)-2, max(rs)+3, min(cs)-2, max(cs)+3
                try:
                    bvv = sv.read(1, window=Window(c0, r0, c1-c0, r1-r0)).astype(np.float64)
                    bvh = sh.read(1, window=Window(c0, r0, c1-c0, r1-r0)).astype(np.float64)
                except Exception:
                    continue
                for i in idxs:
                    r, c = rc[i]; lr, lc = r - r0, c - c0
                    vv = bvv[lr-2:lr+3, lc-2:lc+3]; vh = bvh[lr-2:lr+3, lc-2:lc+3]
                    vvg = vv[(vv > 0) & np.isfinite(vv)]
                    vhg = vh[(vh > 0) & np.isfinite(vh)]
                    if vvg.size >= 8 and vhg.size >= 8:
                        vv_series[i].append(10*np.log10(vvg.mean()))
                        vh_series[i].append(10*np.log10(vhg.mean()))
        finally:
            sv.close(); sh.close()

    for yr in years:
        items = []
        for _att in range(5):
            try:
                items = list(catalog.search(collections=['sentinel-1-rtc'],
                                            bbox=[west, south, east, north],
                                            datetime=f'{yr}-06-01/{yr}-09-30',
                                            max_items=60).items())
                break
            except Exception:
                time.sleep(5)
        used = 0
        for it in items:
            sample(it)
            used += 1
            if used >= max_per_year: break
        nv = sum(1 for s in vv_series if s)
        print(f'    {yr}: {used} escenas · {nv}/{n} pts con dato')

    for i, r in enumerate(rows):
        if vv_series[i]:
            vv = np.array(vv_series[i]); vh = np.array(vh_series[i])
            r['s1_vv_mean'] = float(vv.mean()); r['s1_vh_mean'] = float(vh.mean())
            r['s1_vv_sd'] = float(vv.std()); r['s1_vh_sd'] = float(vh.std())
            r['s1_ratio_mean'] = float(vv.mean() - vh.mean())
        else:
            for k in S1_FEATS: r[k] = float('nan')


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
    'elev_dem_m','slope_deg','twi','curvature',
    'ndvi_early','ndvi_late','ndvi_drop','ndwi_late',
    'clre_early','clre_late','gndvi_early','gndvi_late',
    'ndmi_early','ndmi_late','evi_early','evi_late','nbr_late',
    # Índices nuevos (Indices_Mascaras_S2.txt): mean/min/max/sd sobre jun–sep
    'wavi_mean','wavi_min','wavi_max','wavi_sd',
    'albedo_mean','albedo_min','albedo_max','albedo_sd',
    'ndmi_mean','ndmi_min','ndmi_max','ndmi_sd',
    'ndwi_mean','ndwi_min','ndwi_max','ndwi_sd',
    # Sentinel-1 RTC (gamma-0 terrain-flattened): humedad/estructura SAR
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

def load_model_bundle(path):
    """Carga y valida un bundle de modelo. Devuelve dict o lanza excepción."""
    import joblib
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or 'model' not in bundle or 'features' not in bundle:
        raise ValueError('El archivo no es un modelo válido de borreguil '
                         "(faltan claves 'model'/'features').")
    return bundle

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


def train_rf(rows, default_model_path=None, neg_buffer_m=250):
    """Entrena un RF en modo PRESENCE-ONLY.

    - Positivos: rows con Borreguil=='si' (verdad-terreno; todos son borreguil).
    - Negativos (pseudo-ausencias): candidatos SIN etiqueta que están a más de
      neg_buffer_m de cualquier positivo (y los marcados explícitamente como 'no').
    - Ambiguos (candidatos a < neg_buffer_m de un positivo): excluidos del
      entrenamiento, pero se predicen igualmente.

    Si no hay positivos, carga el modelo preentrenado (default_model_path).
    """
    truth_attr = 'Borreguil'; duda_attr = 'Duda'
    has_pos = any(str(r.get(truth_attr, '')).lower() == 'si' for r in rows)
    if not has_pos:
        # Try to load the default model
        if default_model_path is None:
            default_model_path = Path(__file__).parent / 'rf_sierra_nevada.joblib'
        if Path(default_model_path).exists():
            import joblib
            bundle = joblib.load(default_model_path)
            _zone = bundle.get('training_zone', '?')
            print(f'  → Sin verdad-terreno: usando modelo preentrenado ({_zone})')
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
            proba = clf.predict_proba(X)[:,1]
            return proba, {'mode': 'pretrained',
                            'training_zone': bundle.get('training_zone','?'),
                            'training_n': bundle.get('training_n', 0),
                            'auc_groupkfold': bundle.get('auc_groupkfold', None),
                            'model_bundle': bundle}
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
    print(f'    Positivos (verdad-terreno): {npos}')
    print(f'    Negativos (pseudo-ausencias > {neg_buffer_m} m + explícitos): {nneg}')
    print(f'    Ambiguos excluidos (candidatos < {neg_buffer_m} m de un positivo): '
          f'{int(candidate.sum() - pseudo_neg.sum())}')

    if npos < 15 or nneg < 15:
        print(f'    ATENCIÓN: muestra insuficiente (pos={npos}, neg={nneg}). '
              'Mínimo recomendado: 30 positivos + 60 negativos. '
              'Se usa el modelo preentrenado.')
        return train_rf([{k: v for k, v in r.items() if k != 'Borreguil'} for r in rows],
                        default_model_path=default_model_path, neg_buffer_m=neg_buffer_m)

    clf = RandomForestClassifier(n_estimators=400, max_depth=12,
                                  min_samples_split=4, min_samples_leaf=2,
                                  class_weight='balanced', random_state=42, n_jobs=-1)

    groups = _spatial_groups(rows, np.where(train_mask)[0])
    ugroups = set(groups)
    try:
        if len(ugroups) >= 3:
            yhat_cv = cross_val_predict(clf, X[train_mask], y[train_mask],
                                        cv=GroupKFold(n_splits=min(5, len(ugroups))),
                                        groups=groups, method='predict_proba', n_jobs=-1)[:,1]
            auc = roc_auc_score(y[train_mask], yhat_cv)
            cv_label = f'GroupKFold espacial ({len(ugroups)} grupos)'
        else:
            yhat_cv = cross_val_predict(clf, X[train_mask], y[train_mask],
                                        cv=StratifiedKFold(5, shuffle=True, random_state=42),
                                        method='predict_proba', n_jobs=-1)[:,1]
            auc = roc_auc_score(y[train_mask], yhat_cv)
            cv_label = 'StratifiedKFold'
        print(f'    AUC {cv_label}: {auc:.3f}')
    except Exception as e:
        print(f'    CV falló ({e})'); auc = float('nan'); cv_label = 'n/a'

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
    return proba, {'mode': 'trained', 'auc': float(auc), 'cv': cv_label,
                   'n_pos': npos, 'n_neg': nneg,
                   'n_train': int(train_mask.sum()),
                   'model_bundle': bundle}


# ============================================================
# DECISIÓN FINAL
# ============================================================
def decide(r, threshold=0.5):
    truth = (r.get('Borreguil') or '').strip().lower()
    duda  = (r.get('Duda') or '').strip().lower() == 'si'
    rf = r.get('rf_proba', float('nan'))

    if duda:
        return 'DUDOSO (campo)'
    if truth == 'si':
        return 'BORREGUIL VERIFICADO'
    if truth == 'no':
        return 'NO BORREGUIL VERIFICADO'
    if not isinstance(rf, float) or math.isnan(rf):
        return 'SIN PREDICCIÓN'
    if rf >= 0.70: return 'BORREGUIL PROBABLE'
    if rf >= threshold: return 'POSIBLE BORREGUIL'
    if rf <= 0.20: return 'NO BORREGUIL'
    return 'INCIERTO'


def decision_color(dec):
    """Color hex para una decisión (compartido por la app y el mapa exportado)."""
    d = (dec or '').lower()
    if 'auto' in d:                                  return '#00897B'  # teal: pseudo-positivo (auto-iteración)
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
#   1) AMBIENTE : arroyo | laguna | ladera
#   2) HUMEDAD  : húmedo | seco
#   3) PUREZA   : puro | mixto-agua | mixto-roca | mixto-otros
# Los umbrales son AJUSTABLES aquí. Cada nivel vale '—' si faltan las variables
# necesarias (p. ej. dist_water_m solo existe con OSM activado; surr_* solo si no
# se saltan las imágenes; cir_/ps_vegfrac solo con esas fuentes).
# ============================================================
HIER_THRESH = {
    'ndwi_laguna':        0.05,   # NDWI medio alto → lámina de agua estancada cerca
    'surr_water_laguna':  0.15,   # fracción de agua alrededor (imagen) alta
    'dist_arroyo_m':      30.0,   # a < 30 m de un cauce OSM → arroyo
    'twi_arroyo':         8.0,    # acumulación de flujo alta → fondo de vaguada/arroyo
    'ndmi_humedo':        0.10,   # NDMI (humedad) medio por encima → húmedo
    'twi_humedo':         7.0,    # respaldo de humedad si no hay NDMI
    'ndwi_pixel_agua':    0.0,    # NDWI del píxel positivo → mezcla con agua
    'surr_rock_mixto':    0.20,   # roca alrededor alta → mezcla con roca
    'albedo_roca':        0.22,   # albedo alto + NDVI bajo → roca/suelo desnudo
    'ndvi_puro':          0.45,   # NDVI fin de verano alto → vegetación dominante
    'vegfrac_puro':       0.75,   # fracción vegetal (CIR/Planet) alta → píxel puro
}


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
    hay datos suficientes y 'tipo' es la combinación legible de los tres."""
    th = th or HIER_THRESH
    ndwi       = _hier_num(r, 'ndwi_mean', 'ndwi_late')
    surr_water = _hier_num(r, 'surr_water')
    twi        = _hier_num(r, 'twi')
    slope      = _hier_num(r, 'slope_deg')
    dist_w     = _hier_num(r, 'dist_water_m')
    ndmi       = _hier_num(r, 'ndmi_mean', 'ndmi_late')
    ndvi       = _hier_num(r, 'ndvi_late')
    albedo     = _hier_num(r, 'albedo_mean')
    surr_rock  = _hier_num(r, 'surr_rock')
    vegfrac    = _hier_num(r, 'cir_vegfrac', 'ps_vegfrac')

    # ---- Nivel 1: AMBIENTE ----
    if (surr_water is not None and surr_water > th['surr_water_laguna']) or \
       (ndwi is not None and ndwi > th['ndwi_laguna']):
        amb = 'laguna'
    elif (dist_w is not None and dist_w < th['dist_arroyo_m']) or \
         (twi is not None and twi > th['twi_arroyo']):
        amb = 'arroyo'
    elif slope is not None or twi is not None:
        amb = 'ladera'
    else:
        amb = '—'

    # ---- Nivel 2: HUMEDAD (NDMI directo; respaldo TWI) ----
    if ndmi is not None:
        hum = 'húmedo' if ndmi > th['ndmi_humedo'] else 'seco'
    elif twi is not None:
        hum = 'húmedo' if twi > th['twi_humedo'] else 'seco'
    else:
        hum = '—'

    # ---- Nivel 3: PUREZA (agua > roca > puro > otros) ----
    if ndwi is not None and ndwi > th['ndwi_pixel_agua']:
        pur = 'mixto-agua'
    elif (surr_rock is not None and surr_rock > th['surr_rock_mixto']) or \
         (albedo is not None and ndvi is not None and
          albedo > th['albedo_roca'] and ndvi < th['ndvi_puro']):
        pur = 'mixto-roca'
    elif (vegfrac is not None and vegfrac >= th['vegfrac_puro']) or \
         (ndvi is not None and ndvi >= th['ndvi_puro']):
        pur = 'puro'
    elif ndvi is not None or albedo is not None or ndwi is not None:
        pur = 'mixto-otros'
    else:
        pur = '—'

    tipo = f'{amb} · {hum} · {pur}'
    return {'ambiente': amb, 'humedad': hum, 'pureza': pur, 'tipo': tipo}


def run_rf_and_decide(rows, threshold=0.5, default_model_path=None, neg_buffer_m=250):
    """Entrena/predice el RF y asigna la decisión a cada punto.

    Reutilizable para la EVALUACIÓN RECURSIVA (auto-entrenamiento): como las
    features ya están en `rows`, sólo se reentrena el clasificador y se vuelve
    a predecir, lo que tarda segundos. Devuelve el dict `info` de train_rf.
    """
    proba, info = train_rf(rows, default_model_path=default_model_path,
                           neg_buffer_m=neg_buffer_m)
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
        ('DECISIÓN',  lambda r,i: r.get('decision','')),
        ('RF prob.',  lambda r,i: r.get('rf_proba', None)),
        ('Patrón',    lambda r,i: r.get('mat_signature','')),
        ('Altitud (m)', lambda r,i: r.get('elev_dem_m', None)),
        ('Slope (°)',   lambda r,i: r.get('slope_deg', None)),
        ('TWI',         lambda r,i: r.get('twi', None)),
        ('Dist. cauce (m)', lambda r,i: r.get('dist_water_m', None)),
        ('Dist. ski (m)',   lambda r,i: r.get('dist_ski_m', None)),
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

    for ri, r in enumerate(rows, 2):
        for ci, (nm, fn) in enumerate(COLS, 1):
            try:
                v = fn(r, ri-2)
            except Exception:
                v = None
            if isinstance(v, float) and math.isnan(v): v = None
            ws.cell(row=ri, column=ci, value=v)

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
        dec = r.get('decision','SIN PREDICCIÓN')
        color = decision_color(dec)
        rf = r.get('rf_proba', None)
        rf_txt = f'{rf*100:.0f}%' if isinstance(rf, float) and not math.isnan(rf) else '—'
        popup = (f"<b>{r.get('ID','?')}</b><br>"
                 f"<b>{dec}</b><br>"
                 f"RF: {rf_txt}<br>"
                 f"Patrón: {r.get('mat_signature','—')}<br>"
                 f"Altitud: {r.get('elev_dem_m','—')} m · Slope: {r.get('slope_deg','—')}°<br>"
                 f"NDVI: {r.get('ndvi_late','—')} · Clre: {r.get('clre_late','—')}")
        folium.CircleMarker([r['lat'], r['lon']], radius=5,
                            color=color, fill=True, fill_color=color, fill_opacity=0.8,
                            popup=folium.Popup(popup, max_width=300)).add_to(m)
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
    ap.add_argument('--use-osm', action='store_true', help='Usar OSM (hidrografía/infraestructuras). Por defecto OSM se OMITE (Overpass es inestable).')
    ap.add_argument('--no-overpass', action='store_true', help='[obsoleto] Equivalente a no usar OSM (OSM ya se omite por defecto)')
    ap.add_argument('--threshold', type=float, default=0.5, help='Umbral RF (default 0.5)')
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
                print(f'  área aprox: {study_info["area_km2"]:.1f} km²')
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

    # Merge ground truth (PRESENCE-ONLY: todos los puntos = borreguil)
    if args.truth:
        tr = read_points(args.truth)
        n_match, n_add, n_pos, n_neg = merge_truth_points(rows, tr)
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

    # 3. OSM (omitido por defecto; Overpass es inestable)
    use_osm = args.use_osm and not args.no_overpass
    if use_osm:
        print('\n[3/6] OSM…')
        try:
            fetch_osm(rows, bbox)
        except Exception as e:
            print(f'  OSM falló (se continúa sin OSM): {e}')
    else:
        print('\n[3/6] OSM omitido (usa --use-osm para activarlo)')

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

        # Sentinel-1 RTC (siempre vía MPC, fuente del modelo)
        try:
            fetch_s1(rows, bbox)
        except Exception as e:
            print(f'  Sentinel-1 falló (se continúa sin SAR): {e}')

    # 6. Random Forest
    print('\n[6/6] Random Forest…')
    proba, info = train_rf(rows)
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
        elif info.get('mode') == 'trained':
            print(f'\nModelo: entrenado con verdad-terreno (presence-only)  '
                  f'AUC {info.get("cv","")} = {info["auc"]:.3f}  '
                  f'({info["n_pos"]} positivos, {info["n_neg"]} pseudo-ausencias)')

if __name__ == '__main__':
    main()
