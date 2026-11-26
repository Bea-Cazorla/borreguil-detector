"""
random_points.py — Generación de puntos aleatorios dentro de un área de estudio
para probar el modelo Random Forest pre-entrenado.

Tres modos:
- uniform: muestreo aleatorio uniforme (rejection sampling sobre la bbox).
- stratified_elev: estratificado en bandas altitudinales iguales (más eficiente
  para evaluar el modelo a lo largo del gradiente bioclimático).
- poisson_disc: distancia mínima entre puntos (mejor cobertura espacial; útil
  para evitar redundancia en muestreo de evaluación).

Filtro opcional de elevación mínima/máxima usando Open-Topo-Data SRTM 30 m
(API gratis, sin token, ~100 puntos por petición).

Opcionalmente, los puntos quedan lejos de otros (los de campo): así caen en sitios
que todavía no se han muestreado, para buscar allí ausencias que revisa una persona.

- cuartiles: por cuartiles de una capa de polígonos que sube quien usa la app (p. ej.
  las áreas probables de borreguil de cada parque): una parte de los puntos (75 % de
  entrada) en los polígonos del cuartil 1 y el resto en los de los cuartiles 2 a 4.
"""
import time, math
from pathlib import Path

import numpy as np
import requests
from shapely.geometry import Point

OPEN_TOPO = 'https://api.opentopodata.org/v1/srtm30m'


def fetch_elevations(coords, batch=100, max_retry=5):
    """coords: lista de (lat, lon). Devuelve lista de elevaciones (m) o None.
    Tasa de Open-Topo-Data: 1 req/seg, 100 puntos por req."""
    out = [None] * len(coords)
    for start in range(0, len(coords), batch):
        sub = coords[start:start+batch]
        locs = '|'.join(f"{lat},{lon}" for lat, lon in sub)
        for attempt in range(max_retry):
            try:
                r = requests.get(OPEN_TOPO, params={'locations': locs}, timeout=30)
                if r.status_code == 429:
                    time.sleep(2.0); continue
                r.raise_for_status()
                data = r.json()['results']
                for i, d in enumerate(data):
                    out[start+i] = d.get('elevation')
                break
            except Exception:
                time.sleep(2.0)
        time.sleep(1.1)  # rate limit
    return out


def random_uniform(geom, n, seed=42, max_attempts_factor=200):
    """Rejection sampling: muestrea n puntos aleatorios uniformes dentro de geom."""
    rng = np.random.default_rng(seed)
    minx, miny, maxx, maxy = geom.bounds
    pts = []; attempts = 0
    while len(pts) < n and attempts < n * max_attempts_factor:
        attempts += 1
        x = float(rng.uniform(minx, maxx))
        y = float(rng.uniform(miny, maxy))
        if geom.contains(Point(x, y)) or geom.intersects(Point(x, y)):
            pts.append((x, y))
    if len(pts) < n:
        print(f'  ⚠ Solo se pudieron generar {len(pts)}/{n} puntos en {attempts} intentos')
    return pts


def random_poisson_disc(geom, n, min_dist_m=100, seed=42):
    """Genera puntos con distancia mínima entre ellos (rechazo).
    min_dist_m: distancia mínima entre cualquier par de puntos, en metros."""
    rng = np.random.default_rng(seed)
    minx, miny, maxx, maxy = geom.bounds
    # Aproximación grados → metros usando latitud media
    lat_med = (miny + maxy) / 2
    deg_per_m_y = 1.0 / 111_000
    deg_per_m_x = 1.0 / (111_000 * math.cos(math.radians(lat_med)))
    min_dx = min_dist_m * deg_per_m_x
    min_dy = min_dist_m * deg_per_m_y
    min_d_deg2 = (min_dx)**2  # approx — usamos isotropic en grados

    pts = []
    attempts = 0
    max_attempts = n * 1000
    while len(pts) < n and attempts < max_attempts:
        attempts += 1
        x = float(rng.uniform(minx, maxx))
        y = float(rng.uniform(miny, maxy))
        if not (geom.contains(Point(x, y)) or geom.intersects(Point(x, y))):
            continue
        ok = True
        for px, py in pts:
            if (x-px)**2 + (y-py)**2 < min_d_deg2:
                ok = False; break
        if ok:
            pts.append((x, y))
    if len(pts) < n:
        print(f'  ⚠ Poisson disc: solo {len(pts)}/{n} puntos en {attempts} intentos '
              f'(min_dist demasiado grande para el área)')
    return pts


def lejos_de(pts, excluir, lejos_m):
    """De unos puntos (lon, lat), los que quedan a `lejos_m` metros o más de todos los
    de `excluir` (lon, lat). Las distancias se miden en la UTM del centro de los
    puntos."""
    pts = list(pts)
    if not pts or not excluir or not lejos_m or lejos_m <= 0:
        return pts
    import pyproj
    from scipy.spatial import cKDTree
    lon_c = sum(p[0] for p in pts) / len(pts)
    lat_c = sum(p[1] for p in pts) / len(pts)
    epsg = (32600 if lat_c >= 0 else 32700) + int((lon_c + 180) // 6) + 1
    tr = pyproj.Transformer.from_crs(4326, epsg, always_xy=True)
    xe, ye = tr.transform([float(p[0]) for p in excluir], [float(p[1]) for p in excluir])
    xp, yp = tr.transform([p[0] for p in pts], [p[1] for p in pts])
    distancia, _ = cKDTree(np.column_stack([xe, ye])).query(np.column_stack([xp, yp]))
    return [p for p, d in zip(pts, distancia) if d >= lejos_m]


def filter_by_elevation(pts, min_elev=None, max_elev=None, siempre=False):
    """Filtra puntos por elevación SRTM.

    Devuelve (keep, keep_elevs, pool_elevs) donde `pool_elevs` son TODAS las
    elevaciones válidas del muestreo (para diagnosticar el rango altitudinal real
    del área cuando ningún punto cumple el filtro). Con `siempre`, consulta las
    altitudes aunque no haya límites (el reparto por franjas de altitud las necesita)."""
    if not (min_elev or max_elev or siempre):
        return pts, [None]*len(pts), []
    coords = [(y, x) for x, y in pts]   # (lat, lon)
    elevs = fetch_elevations(coords)
    pool_elevs = [e for e in elevs if e is not None]
    keep = []; keep_elevs = []
    for (x, y), e in zip(pts, elevs):
        if e is None: continue
        if min_elev is not None and e < min_elev: continue
        if max_elev is not None and e > max_elev: continue
        keep.append((x, y)); keep_elevs.append(e)
    return keep, keep_elevs, pool_elevs


def generate(geom, n, mode='uniform', min_elev=None, max_elev=None,
             min_dist_m=100, seed=42, excluir=None, lejos_m=0, capa=None, frac_q1=0.75):
    """
    Genera n puntos aleatorios dentro del área de estudio.

    excluir: puntos (lon, lat) de los que hay que alejarse (los de campo) y lejos_m,
    a cuántos metros como mínimo. Sin ellos, igual que siempre.

    mode='cuartiles': por cuartiles de una capa (ver `por_cuartiles`). `capa` es
    {'geoms', 'valores', 'columna'} y `frac_q1`, la parte de los puntos que va al
    cuartil 1.

    Devuelve (rows, stats):
      - rows: lista de dicts compatible con read_points del pipeline
        [{'ID': 'rand_0000', 'lon': ..., 'lat': ..., 'source': 'random_uniform', ...}, ...]
      - stats: dict con diagnóstico altitudinal del área muestreada:
        {'requested', 'kept', 'area_elev_min', 'area_elev_max', 'reason'}.
        `reason` puede ser 'min_elev_above_area' (el área entera queda por debajo
        de la altitud mínima pedida), 'max_elev_below_area', o
        'empty_after_filter'.
    """
    print(f'Generando {n} puntos aleatorios (modo={mode})…')
    stats = {'requested': n, 'kept': 0, 'reason': None,
             'area_elev_min': None, 'area_elev_max': None, 'excluded_near': 0}
    if mode == 'cuartiles':
        return por_cuartiles(geom, n, capa, frac_q1=frac_q1, min_elev=min_elev,
                             max_elev=max_elev, seed=seed, excluir=excluir,
                             lejos_m=lejos_m, stats=stats)
    # Si hay que alejarse de unos puntos, se generan más para que, quitados los que
    # caen cerca, sigan quedando bastantes.
    lejos = bool(excluir) and bool(lejos_m) and lejos_m > 0
    extra = 3 if lejos else 1
    if mode == 'uniform':
        # Para tener un pool donde filtrar por elevación
        n_pool = (n if not (min_elev or max_elev) else int(n * 3)) * extra
        raw = random_uniform(geom, n_pool, seed=seed)
    elif mode == 'poisson_disc':
        n_pool = (n if not (min_elev or max_elev) else int(n * 2)) * extra
        raw = random_poisson_disc(geom, n_pool, min_dist_m=min_dist_m, seed=seed)
    elif mode == 'stratified_elev':
        # primero genera muchos puntos uniformes, después estratifica por banda
        raw = random_uniform(geom, n * 4 * extra, seed=seed)
    else:
        raise ValueError(f'Modo desconocido: {mode}')
    if lejos:
        antes = len(raw)
        raw = lejos_de(raw, excluir, lejos_m)
        stats['excluded_near'] = antes - len(raw)
        print(f'  {antes - len(raw)} quitados por caer a menos de {lejos_m:.0f} m de los '
              f'{len(excluir)} puntos de campo; quedan {len(raw)}')

    if min_elev or max_elev or mode == 'stratified_elev':
        print(f'  Consultando elevaciones (Open-Topo-Data SRTM 30 m)…')
        kept, elevs, pool_elevs = filter_by_elevation(raw, min_elev=min_elev, max_elev=max_elev,
                                                      siempre=(mode == 'stratified_elev'))
        print(f'  {len(kept)}/{len(raw)} puntos cumplen criterio altitudinal')

        # Diagnóstico del rango altitudinal real del área (del muestreo)
        if pool_elevs:
            stats['area_elev_min'] = float(min(pool_elevs))
            stats['area_elev_max'] = float(max(pool_elevs))
        if len(kept) == 0 and pool_elevs:
            amin, amax = min(pool_elevs), max(pool_elevs)
            if min_elev is not None and amax < min_elev:
                stats['reason'] = 'min_elev_above_area'
            elif max_elev is not None and amin > max_elev:
                stats['reason'] = 'max_elev_below_area'
            else:
                stats['reason'] = 'empty_after_filter'
            print(f'  ⚠ 0 puntos: el área va de {amin:.0f} a {amax:.0f} m '
                  f'(min_elev={min_elev}, max_elev={max_elev})')
            return [], stats
        if len(kept) == 0 and not pool_elevs and (min_elev or max_elev or mode == 'stratified_elev'):
            # SRTM no devolvió altitudes (zona de agua, fuera de cobertura, o
            # fallo/timeout de la API de elevación).
            stats['reason'] = 'no_elevation_data'
            print('  ⚠ 0 puntos: no se obtuvieron altitudes del área '
                  '(¿zona de agua o fallo de la API de elevación?)')
            return [], stats

        if mode == 'stratified_elev' and kept:
            # Estratificar: dividir rango en 5 bandas iguales y tomar n/5 de cada una
            arr = np.array(elevs)
            lo, hi = arr.min(), arr.max()
            n_bands = min(5, len(kept))
            edges = np.linspace(lo, hi, n_bands+1)
            per_band = max(1, n // n_bands)
            rng = np.random.default_rng(seed)
            selected = []
            for b in range(n_bands):
                in_band = [(p, e) for p, e in zip(kept, elevs)
                            if edges[b] <= e <= edges[b+1]]
                if not in_band: continue
                idx = rng.choice(len(in_band), min(per_band, len(in_band)), replace=False)
                for i in idx:
                    selected.append(in_band[i])
            kept = [p for p, e in selected]
            elevs = [e for p, e in selected]
            print(f'  Estratificación: {len(kept)} puntos en {n_bands} bandas '
                  f'({lo:.0f}-{hi:.0f} m)')

        # Recortar a n
        if len(kept) > n:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(kept), n, replace=False)
            kept = [kept[i] for i in idx]
            elevs = [elevs[i] for i in idx]
    else:
        kept = raw[:n]; elevs = [None] * len(kept)

    # Snap al CENTRO del píxel Sentinel-2 (10 m) que contiene cada punto, para que
    # cada punto caiga en un píxel S2 bien definido (no a caballo entre dos).
    if kept:
        try:
            import borreguil_pipeline as _bp
            kept = _bp.snap_to_s2_grid([(x, y) for x, y in kept])
            print('  Puntos ajustados al centro de píxel Sentinel-2 (10 m)')
        except Exception as _e:
            print(f'  (no se pudo ajustar a rejilla S2: {_e})')

    rows = []
    for i, ((x, y), e) in enumerate(zip(kept, elevs)):
        d = {
            'ID': f'rand_{i:04d}',
            'lon': x,
            'lat': y,
            'source': f'random_{mode}',
            'snapped_s2': 1,
        }
        if e is not None:
            d['elev_m_initial'] = e
        rows.append(d)
    print(f'Total generados: {len(rows)}')
    stats['kept'] = len(rows)
    return rows, stats



# ======================================================================
# Por cuartiles de una capa (26-11-2026)
# ======================================================================
# La autora: «quiero lanzar por cuartiles de una capa (que la subirán los usuarios): el
# 75 % de los puntos en el primer cuartil y el 25 % en el resto. Ausencia y presencia».
# Los puntos no llevan etiqueta: se revisan después, y así salen presencias (sobre todo
# en el cuartil 1) y ausencias (sobre todo en el resto). Dentro de cada grupo caen al
# azar sobre la superficie: un polígono grande recibe más puntos que uno pequeño.
#
# La capa se lee sin GDAL cuando es GeoPackage o GeoJSON (en algunos Windows la DLL de
# GDAL está bloqueada); KML, KMZ y Shapefile necesitan GDAL (geopandas).

FORMATOS_CAPA = ('gpkg', 'geojson', 'json', 'kml', 'kmz', 'zip')
_M_POR_GRADO = 111320.0                  # metros por grado (en el ecuador)


def _a_wgs84(geoms, crs):
    """Las geometrías en EPSG:4326 (lon, lat). `crs`: lo que entienda pyproj, o None
    si la capa no lo dice (entonces se dan por grados)."""
    import shapely
    if crs is None:
        return list(geoms)
    import pyproj
    crs = pyproj.CRS.from_user_input(crs)
    if crs.to_epsg() == 4326:
        return list(geoms)
    t = pyproj.Transformer.from_crs(crs, 4326, always_xy=True)

    def _cambia(xy):
        x, y = t.transform(xy[:, 0], xy[:, 1])
        return np.column_stack([x, y])
    return list(shapely.transform(np.asarray(geoms, dtype=object), _cambia))


def _geom_de_gpkg(blob):
    """La geometría de una celda de GeoPackage: cabecera «GP» + WKB."""
    import shapely
    if blob is None:
        return None
    b = bytes(blob)
    if b[:2] != b'GP':
        return shapely.from_wkb(b)
    flags = b[3]
    if flags & 0b10000:                              # geometría vacía
        return None
    sobre = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}.get((flags >> 1) & 0b111, 0)
    return shapely.from_wkb(b[8 + sobre:])


def _leer_gpkg(ruta):
    """(geoms, atributos, crs, capa) de la primera capa de polígonos de un GeoPackage,
    con sqlite3 (un GeoPackage es una base de datos SQLite)."""
    import sqlite3
    import pandas as pd
    con = sqlite3.connect(Path(ruta).resolve().as_uri() + '?mode=ro', uri=True)
    try:
        capas = con.execute(
            'select table_name, column_name, geometry_type_name, srs_id '
            'from gpkg_geometry_columns').fetchall()
        poligonales = [c for c in capas if 'POLYGON' in str(c[2]).upper()
                       or str(c[2]).upper() in ('GEOMETRY', 'SURFACE', 'MULTISURFACE')]
        if not poligonales:
            raise ValueError('el GeoPackage no tiene ninguna capa de polígonos')
        tabla, col_geom, _tipo, srs = poligonales[0]
        df = pd.read_sql_query(f'select * from "{tabla}"', con)
        crs = None
        if srs not in (None, 0, -1):
            fila = con.execute(
                'select organization, organization_coordsys_id, definition '
                'from gpkg_spatial_ref_sys where srs_id = ?', (srs,)).fetchone()
            if fila:
                org, cod, wkt = fila
                crs = (f'EPSG:{cod}' if str(org).upper() == 'EPSG'
                       else (wkt if wkt and wkt != 'undefined' else None))
    finally:
        con.close()
    geoms = [_geom_de_gpkg(b) for b in df[col_geom]]
    atributos = df.drop(columns=[col_geom])
    return geoms, atributos, crs, tabla


def _leer_geojson(ruta):
    import json
    import re
    import pandas as pd
    from shapely.geometry import shape
    with open(ruta, encoding='utf-8-sig') as f:
        datos = json.load(f)
    feats = (datos.get('features', []) if datos.get('type') == 'FeatureCollection'
             else [datos])
    geoms = [shape(ft['geometry']) if ft.get('geometry') else None for ft in feats]
    atributos = pd.DataFrame([dict(ft.get('properties') or {}) for ft in feats])
    nombre = ((datos.get('crs') or {}).get('properties') or {}).get('name', '')
    m = re.search(r'EPSG:+(\d+)', str(nombre))
    return geoms, atributos, (f'EPSG:{m.group(1)}' if m else None), Path(ruta).stem


def _leer_con_gdal(ruta):
    try:
        import geopandas as gpd
        gdf = gpd.read_file(ruta)
    except ImportError as e:
        raise ValueError('este ordenador no puede leer ese formato (le falta GDAL): sube '
                         'la capa en GeoPackage (.gpkg) o GeoJSON') from e
    crs = gdf.crs.to_wkt() if gdf.crs is not None else None
    return (list(gdf.geometry), gdf.drop(columns=gdf.geometry.name), crs,
            Path(ruta).stem)


def _de_un_zip(ruta):
    """La capa que trae un .zip (sin la basura de macOS). Si trae varias, avisa."""
    import tempfile
    import zipfile
    sufijos = ('.gpkg', '.geojson', '.json', '.shp', '.kml', '.kmz')
    with zipfile.ZipFile(ruta) as z:
        nombres = [n for n in z.namelist()
                   if not n.startswith('__MACOSX/') and not Path(n).name.startswith('.')
                   and Path(n).suffix.lower() in sufijos]
        if not nombres:
            raise ValueError('el .zip no trae ninguna capa (GeoPackage, GeoJSON, '
                             'Shapefile o KML)')
        if len(nombres) > 1:
            raise ValueError('el .zip trae varias capas (' + ', '.join(
                Path(n).name for n in nombres[:6]) + ('…' if len(nombres) > 6 else '')
                + '): sube solo la que quieras usar')
        carpeta = Path(tempfile.mkdtemp(prefix='capa_cuartiles_'))
        base = Path(nombres[0]).with_suffix('')
        for n in z.namelist():          # el Shapefile necesita sus ficheros hermanos
            if Path(n).with_suffix('') == base and not n.startswith('__MACOSX/'):
                (carpeta / Path(n).name).write_bytes(z.read(n))
        return carpeta / Path(nombres[0]).name


def _solo_poligonos(g):
    """Lo poligonal de una geometría (o None)."""
    from shapely.geometry import MultiPolygon
    if g is None or g.is_empty:
        return None
    if g.geom_type in ('Polygon', 'MultiPolygon'):
        return g
    if g.geom_type == 'GeometryCollection':
        partes = [p for q in g.geoms for p in
                  ((q,) if q.geom_type == 'Polygon' else
                   tuple(q.geoms) if q.geom_type == 'MultiPolygon' else ())]
        return MultiPolygon(partes) if partes else None
    return None


def leer_capa(ruta):
    """Una capa de polígonos con sus atributos, en EPSG:4326.

    Devuelve {'geoms': [...], 'atributos': DataFrame (una fila por polígono),
    'capa': nombre}. GeoPackage y GeoJSON se leen sin GDAL; lo demás, con geopandas.
    Lanza ValueError con un mensaje para la persona si no se puede usar."""
    import shutil
    import shapely
    ruta = Path(ruta)
    del_zip = None
    if ruta.suffix.lower() == '.zip':
        ruta = _de_un_zip(ruta)
        del_zip = ruta.parent
    try:
        suf = ruta.suffix.lower()
        if suf == '.gpkg':
            geoms, atributos, crs, capa = _leer_gpkg(ruta)
        elif suf in ('.geojson', '.json'):
            geoms, atributos, crs, capa = _leer_geojson(ruta)
        else:
            geoms, atributos, crs, capa = _leer_con_gdal(ruta)
    finally:
        if del_zip is not None:          # lo que se sacó del .zip, ya leído
            shutil.rmtree(del_zip, ignore_errors=True)
    geoms = _a_wgs84([g if g is not None else shapely.Polygon() for g in geoms], crs)
    validas = []
    for g in geoms:
        if g is not None and not g.is_empty and not g.is_valid:
            g = shapely.make_valid(g)
        validas.append(_solo_poligonos(g))
    usar = [g is not None for g in validas]
    if not any(usar):
        raise ValueError('la capa no tiene polígonos (¿es de puntos o de líneas?)')
    atributos = atributos.reset_index(drop=True)[usar].reset_index(drop=True)
    return {'geoms': [g for g in validas if g is not None], 'atributos': atributos,
            'capa': capa}


def es_columna_de_cuartiles(nombre, valores):
    """True si la columna ya trae los cuartiles: se llama «cuartil» (o quartile…) y
    solo tiene los números 1, 2, 3 y 4."""
    import re
    import pandas as pd
    if not re.search(r'cuartil|quartil', str(nombre), re.I):
        return False
    v = pd.to_numeric(pd.Series(list(valores)), errors='coerce').dropna()
    return len(v) > 0 and bool(((v == v.round()) & v.between(1, 4)).all())


def columnas_para_cuartiles(atributos):
    """Las columnas con las que se pueden hacer cuartiles: la de cuartiles, si la hay,
    primero; después las numéricas con al menos 4 valores distintos."""
    import pandas as pd
    ya, numericas = [], []
    for c in atributos.columns:
        if es_columna_de_cuartiles(c, atributos[c]):
            ya.append(c)
            continue
        v = pd.to_numeric(atributos[c], errors='coerce')
        if (str(c).lower() not in ('fid', 'id', 'objectid', 'ogc_fid')
                and v.notna().any() and v.nunique() >= 4):
            numericas.append(c)
    return ya + numericas


def cuartiles_de(valores, nombre):
    """El cuartil (1 a 4) de cada polígono; 0 si no tiene valor.

    Si la columna ya es de cuartiles, se usa tal cual (1 = el primero). Si no, se
    ordenan los polígonos de mayor a menor valor y se reparten en cuatro grupos con el
    mismo número de polígonos: el 1 es el 25 % con los valores más altos."""
    import pandas as pd
    v = pd.to_numeric(pd.Series(list(valores)), errors='coerce')
    if es_columna_de_cuartiles(nombre, v):
        return v.fillna(0).astype(int).to_numpy()
    q = np.zeros(len(v), dtype=int)
    con = v.dropna()
    if len(con):
        orden = np.asarray(con.sort_values(ascending=False, kind='mergesort').index)
        q[orden] = np.arange(len(orden)) * 4 // len(orden) + 1
    return q


def _hectareas(g):
    """Superficie aproximada (ha) de una geometría en grados."""
    if g is None or g.is_empty:
        return 0.0
    return g.area * _M_POR_GRADO ** 2 * math.cos(math.radians(g.centroid.y)) / 1e4


def zonas_por_cuartil(area, geoms, valores, columna):
    """Los polígonos de la capa dentro del área de estudio, en dos grupos: 1 (el
    cuartil 1) y 2 (los cuartiles 2 a 4), cada uno como [(polígono, cuartil)]. Los
    cuartiles se calculan con los polígonos que caen en el área (si la columna ya los
    trae, se usan tal cual). Devuelve (grupos, cuartil de cada polígono del área)."""
    import shapely
    from shapely.strtree import STRtree
    geoms = list(geoms)
    valores = list(valores)
    if area is not None:
        idx = sorted(int(i) for i in STRtree(geoms).query(area, predicate='intersects'))
        shapely.prepare(area)
    else:
        idx = list(range(len(geoms)))
    q = cuartiles_de([valores[i] for i in idx], columna)
    grupos = {1: [], 2: []}
    for i, cu in zip(idx, q):
        if cu not in (1, 2, 3, 4):
            continue
        g = geoms[i]
        if area is not None and not area.contains(g):
            g = _solo_poligonos(g.intersection(area))
        if g is None or g.is_empty:
            continue
        for p in (g.geoms if g.geom_type == 'MultiPolygon' else (g,)):
            grupos[1 if cu == 1 else 2].append((p, int(cu)))
    return grupos, q


def muestrear_poligonos(polis, n, rng):
    """n puntos al azar sobre la superficie de unos polígonos en grados, en proporción
    a la superficie de cada uno: [(lon, lat, cuartil)]."""
    import shapely
    if n <= 0 or not polis:
        return []
    pesos = np.array([p.area * math.cos(math.radians(p.centroid.y)) for p, _ in polis])
    if pesos.sum() <= 0:
        return []
    elegidos = rng.choice(len(polis), size=n, p=pesos / pesos.sum())
    salida = []
    for i, k in zip(*np.unique(elegidos, return_counts=True)):
        p, cu = polis[int(i)]
        shapely.prepare(p)
        minx, miny, maxx, maxy = p.bounds
        dentro = []
        for _ in range(2000):
            if len(dentro) >= k:
                break
            m = max(32, 4 * (int(k) - len(dentro)))
            xs = rng.uniform(minx, maxx, m)
            ys = rng.uniform(miny, maxy, m)
            si = shapely.contains_xy(p, xs, ys)
            dentro.extend(zip(xs[si].tolist(), ys[si].tolist()))
        salida.extend((x, y, cu) for x, y in dentro[:int(k)])
    return [salida[j] for j in rng.permutation(len(salida))]


def por_cuartiles(area, n, capa, frac_q1=0.75, min_elev=None, max_elev=None, seed=42,
                  excluir=None, lejos_m=0, stats=None):
    """n puntos: round(n · frac_q1) en el cuartil 1 de la capa y el resto en los
    cuartiles 2 a 4, al azar sobre la superficie de cada grupo y dentro del área de
    estudio. Cada punto lleva el cuartil del polígono en el que cae. Admite el filtro de
    altitud y alejarse de los puntos de campo, como los demás modos.

    `capa`: {'geoms', 'valores', 'columna'} (de `leer_capa` y la columna elegida).
    Devuelve (rows, stats); stats['cuartiles'] dice, de cada grupo (1 y 2 = cuartiles
    2-4), cuántos puntos se pedían, cuántos salen, cuántos polígonos hay y su superficie
    (ha)."""
    if stats is None:
        stats = {'requested': n, 'kept': 0, 'reason': None, 'area_elev_min': None,
                 'area_elev_max': None, 'excluded_near': 0}
    if not capa or not capa.get('geoms'):
        stats['reason'] = 'sin_capa'
        return [], stats
    rng = np.random.default_rng(seed)
    grupos, q = zonas_por_cuartil(area, capa['geoms'], capa['valores'],
                                  capa.get('columna', ''))
    n1 = int(round(n * float(frac_q1)))
    pedidos = {1: n1, 2: n - n1}
    filtro_alt = bool(min_elev or max_elev)
    lejos = bool(excluir) and bool(lejos_m) and lejos_m > 0
    factor = (3 if filtro_alt else 1) * (3 if lejos else 1)
    resumen, elegidos, todas_elev = {}, [], []
    for grupo in (1, 2):
        polis = grupos[grupo]
        info = {'pedidos': pedidos[grupo], 'salen': 0, 'poligonos': len(polis),
                'ha': round(sum(_hectareas(p) for p, _ in polis), 3)}
        resumen[grupo] = info
        if not pedidos[grupo] or not polis:
            continue
        crudos = muestrear_poligonos(polis, pedidos[grupo] * factor, rng)
        if lejos:
            quedan = set(lejos_de([(x, y) for x, y, _ in crudos], excluir, lejos_m))
            stats['excluded_near'] = stats.get('excluded_near', 0) + len(crudos) - len(quedan)
            crudos = [c for c in crudos if (c[0], c[1]) in quedan]
        elevs = [None] * len(crudos)
        if filtro_alt and crudos:
            kept, k_elev, pool = filter_by_elevation([(x, y) for x, y, _ in crudos],
                                                     min_elev=min_elev, max_elev=max_elev)
            todas_elev += pool
            alt = dict(zip(kept, k_elev))
            crudos = [c for c in crudos if (c[0], c[1]) in alt]
            elevs = [alt[(c[0], c[1])] for c in crudos]
        sel = list(zip(crudos, elevs))[:pedidos[grupo]]
        info['salen'] = len(sel)
        elegidos += sel
    stats['cuartiles'] = resumen
    stats['poligonos_area'] = int(sum(1 for x in q if x in (1, 2, 3, 4)))
    if todas_elev:
        stats['area_elev_min'] = float(min(todas_elev))
        stats['area_elev_max'] = float(max(todas_elev))
    if not elegidos:
        if not (grupos[1] or grupos[2]):
            stats['reason'] = 'capa_fuera_del_area'
        elif filtro_alt and todas_elev:
            amin, amax = min(todas_elev), max(todas_elev)
            stats['reason'] = (
                'min_elev_above_area' if min_elev is not None and amax < min_elev
                else 'max_elev_below_area' if max_elev is not None and amin > max_elev
                else 'empty_after_filter')
        elif filtro_alt:
            stats['reason'] = 'no_elevation_data'
        else:
            stats['reason'] = 'empty_after_filter'
        return [], stats

    pts = [(c[0], c[1]) for c, _ in elegidos]
    try:
        import borreguil_pipeline as _bp
        pts = _bp.snap_to_s2_grid(pts)
    except Exception as _e:                              # noqa: BLE001
        print(f'  (no se pudo ajustar a rejilla S2: {_e})')
    rows = []
    for i, ((x, y), (c, e)) in enumerate(zip(pts, elegidos)):
        d = {'ID': f'rand_{i:04d}', 'lon': x, 'lat': y, 'source': 'random_cuartiles',
             'snapped_s2': 1, 'cuartil': int(c[2])}
        if e is not None:
            d['elev_m_initial'] = e
        rows.append(d)
    stats['kept'] = len(rows)
    print(f'Por cuartiles: {resumen[1]["salen"]} en el cuartil 1 y {resumen[2]["salen"]} '
          f'en los cuartiles 2-4 ({stats["poligonos_area"]} polígonos en el área)')
    return rows, stats


if __name__ == '__main__':
    import argparse, sys, json
    import study_area as sa
    from shapely.geometry import mapping
    ap = argparse.ArgumentParser()
    ap.add_argument('--wdpa')
    ap.add_argument('--name')
    ap.add_argument('--file')
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--mode', default='uniform',
                     choices=['uniform','poisson_disc','stratified_elev'])
    ap.add_argument('--min-elev', type=float)
    ap.add_argument('--max-elev', type=float)
    ap.add_argument('--out', default='random_points.geojson')
    args = ap.parse_args()

    geom, info = sa.resolve(args.file, args.wdpa, args.name)
    if not geom:
        print('No se pudo resolver el área de estudio.', file=sys.stderr); sys.exit(1)
    print(f'Área: {info.get("name", info.get("query","?"))} — {info["area_km2"]:.0f} km²')

    rows, _stats = generate(geom, args.n, mode=args.mode,
                            min_elev=args.min_elev, max_elev=args.max_elev)

    # Save GeoJSON
    fc = {'type':'FeatureCollection','features': []}
    for r in rows:
        fc['features'].append({
            'type':'Feature',
            'properties': {k:v for k,v in r.items() if k not in ('lon','lat')},
            'geometry': {'type':'Point','coordinates':[r['lon'], r['lat']]}
        })
    with open(args.out, 'w') as f:
        json.dump(fc, f)
    print(f'Saved: {args.out}')
