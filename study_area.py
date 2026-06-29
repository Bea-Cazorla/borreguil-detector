"""
study_area.py — Resolución de área de estudio para el pipeline.

Tres vías de entrada (en orden de preferencia):

1. ARCHIVO VECTORIAL local (KML, Shp, GeoJSON, GeoPackage, .zip de shapefile).
2. WDPA ID (World Database on Protected Areas, protectedplanet.net):
   - Token de Protected Planet (arg `wdpa_token` o env WDPA_TOKEN) → API oficial,
     resuelve CUALQUIER ID.
   - Google Earth Engine (si `use_gee=True` y EE inicializado) → capa WCMC/WDPA,
     sin token.
   - Overpass (`ref:WDPA`) y cache local de nombres conocidos → Nominatim.
3. NOMBRE del área protegida (string libre): búsqueda en Nominatim.

Devuelve un objeto shapely (Polygon o MultiPolygon) en EPSG:4326.

NOTA DE ROBUSTEZ: en entornos con shapely/numpy incompatibles, la construcción de
multi-geometrías (`unary_union`, `MultiPolygon`, `shape()` de un MultiPolygon)
puede fallar con "ufunc 'create_collection' not supported". Las utilidades
`_shape_safe` y `_merge_polys` degradan a la caja envolvente (bbox) en ese caso en
lugar de romper. La solución de raíz es `pip install -U "shapely>=2.0.7"`.
"""
import os, json, requests
from pathlib import Path
from shapely.geometry import shape as shp_shape, box as shp_box

HEADERS = {'User-Agent': 'borreguil-pipeline/1.0'}

# Areas conocidas (cache local nombre↔WDPA ID). Se puede ampliar libremente.
WDPA_KNOWN = {
    '555512151': 'Parque Nacional de Sierra Nevada, Granada',
    '555512152': 'Parque Natural de Sierra Nevada, Granada',
    '4514':      'Parque Nacional de Picos de Europa',
    '555588708': 'Parque Nacional de Ordesa y Monte Perdido',
    '7411':      "Parc Nacional d'Aigüestortes i Estany de Sant Maurici",
    '555588709': 'Parque Nacional de la Sierra de Guadarrama',
    '2553':      'Parc national des Écrins',
    '2552':      'Parc national de la Vanoise',
    '11':        'Yellowstone National Park, USA',
    '1234':      'Yosemite National Park, USA',
    '2017':      'Banff National Park, Canada',
    '900754':    'Parco Nazionale del Gran Paradiso',
}


# ======================================================================
# Utilidades robustas de geometría
# ======================================================================
def _geojson_bounds(gj):
    """(minx, miny, maxx, maxy) recorriendo coordenadas de un dict GeoJSON
    (Polygon o MultiPolygon), sin shapely."""
    xs, ys = [], []

    def walk(coords):
        # coords puede anidar a cualquier profundidad hasta pares [x, y]
        if (isinstance(coords, (list, tuple)) and len(coords) >= 2
                and isinstance(coords[0], (int, float))
                and isinstance(coords[1], (int, float))):
            xs.append(coords[0]); ys.append(coords[1])
        elif isinstance(coords, (list, tuple)):
            for c in coords:
                walk(c)

    walk(gj.get('coordinates', []))
    if not xs:
        raise ValueError('GeoJSON sin coordenadas')
    return (min(xs), min(ys), max(xs), max(ys))


def _shape_safe(gj):
    """Convierte un dict GeoJSON a geometría shapely. Si la construcción de
    multi-geometría falla (entorno shapely/numpy roto), devuelve la caja
    envolvente (shapely.box), que mantiene .bounds/.contains/.area/mapping."""
    try:
        return shp_shape(gj)
    except Exception:
        try:
            return shp_box(*_geojson_bounds(gj))
        except Exception:
            return None


def _merge_polys(polys):
    """Fusiona una lista de geometrías shapely (Polygon/MultiPolygon) en una sola.
    1 elemento → tal cual. Varios → unary_union; si falla (entorno roto) → bbox
    envolvente."""
    polys = [p for p in polys if p is not None]
    if not polys:
        return None
    if len(polys) == 1:
        return polys[0]
    try:
        from shapely.ops import unary_union
        return unary_union(polys)
    except Exception:
        xs = [p.bounds for p in polys]
        return shp_box(min(b[0] for b in xs), min(b[1] for b in xs),
                       max(b[2] for b in xs), max(b[3] for b in xs))


def _info(geom, **extra):
    d = {'bounds': list(geom.bounds), 'area_km2': geom.area * 12321}
    d.update(extra)
    return d


# ======================================================================
# Carga desde archivo vectorial
# ======================================================================
def load_vector(path):
    """Lee un vector (KML/Shp/GeoJSON/GeoPackage/.zip) y devuelve un Polygon o
    MultiPolygon fusionado en EPSG:4326."""
    import geopandas as gpd
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs('EPSG:4326')
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    polys = [g for g in gdf.geometry
             if g is not None and g.geom_type in ('Polygon', 'MultiPolygon')]
    if not polys:
        raise ValueError(f'No hay polígonos en {Path(path).name} '
                         '(¿es una capa de puntos o líneas?)')
    geom = _merge_polys(polys)
    if geom is None:
        raise ValueError(f'No se pudo construir la geometría de {Path(path).name}')
    return geom


# ======================================================================
# Búsqueda por nombre (Nominatim / OSM)
# ======================================================================
# Preferencia de tipos de área protegida (mayor = mejor)
_TYPE_RANK = {
    'national_park': 5, 'protected_area': 4, 'nature_reserve': 3,
    'natural_park': 3, 'park': 1,
}
_CLASS_RANK = {'boundary': 3, 'leisure': 2, 'natural': 1}


def fetch_nominatim(name_or_query, limit=8):
    """Busca en Nominatim y devuelve (geom, info) eligiendo el resultado poligonal
    más parecido a un área protegida. Si solo hay puntos, devuelve (None, None)."""
    url = 'https://nominatim.openstreetmap.org/search'
    params = {'q': name_or_query, 'format': 'json', 'polygon_geojson': 1,
              'limit': limit, 'extratags': 1}
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=30)
        r.raise_for_status()
        results = r.json()
    except Exception:
        return None, None

    best, best_score = None, -1
    for res in results:
        gj = res.get('geojson') or {}
        if gj.get('type') not in ('Polygon', 'MultiPolygon'):
            continue
        score = (_TYPE_RANK.get(res.get('type'), 0)
                 + _CLASS_RANK.get(res.get('class'), 0)
                 + float(res.get('importance', 0)))
        # bonus si los extratags lo marcan como protegido
        et = res.get('extratags') or {}
        if et.get('protect_class') or et.get('boundary') == 'protected_area':
            score += 2
        if score > best_score:
            best, best_score = res, score

    if best is None:
        return None, None
    geom = _shape_safe(best['geojson'])
    return (geom, best) if geom is not None else (None, None)


# ======================================================================
# WDPA: API Protected Planet (con token)
# ======================================================================
def fetch_wdpa_api(wdpa_id, token=None):
    """Descarga el polígono del área desde la API de Protected Planet.
    Requiere token (arg o env WDPA_TOKEN). Resuelve cualquier WDPA ID."""
    token = token or os.environ.get('WDPA_TOKEN')
    if not token:
        return None, None
    url = f'https://api.protectedplanet.net/v3/protected_areas/{wdpa_id}'
    try:
        r = requests.get(url, params={'token': token, 'with_geometry': 'true'},
                         headers=HEADERS, timeout=60)
        if r.status_code != 200:
            return None, None
        data = r.json()
    except Exception:
        return None, None
    pa = data.get('protected_area') or data
    # La API v3 devuelve la geometría como Feature GeoJSON bajo 'geojson'.
    gj = pa.get('geojson') or {}
    geom_gj = gj.get('geometry') if isinstance(gj, dict) else None
    geom_gj = geom_gj or pa.get('geometry') or pa.get('geom')
    if not geom_gj:
        return None, None
    geom = _shape_safe(geom_gj)
    if geom is None:
        return None, None
    return geom, {'name': pa.get('name', '')}


# ======================================================================
# WDPA / áreas protegidas vía Overpass (OSM)
# ======================================================================
def fetch_overpass_protected_area(name_or_id):
    """Busca relation con ref:WDPA / protect_id en OSM y reconstruye el polígono."""
    queries = [
        f'[out:json][timeout:60];relation["ref:WDPA"="{name_or_id}"];(._;>;);out geom;',
        f'[out:json][timeout:60];relation["protect_id"="{name_or_id}"];(._;>;);out geom;',
    ]
    URL = 'https://overpass-api.de/api/interpreter'
    from shapely.geometry import Polygon
    for q in queries:
        try:
            r = requests.post(URL, data={'data': q}, headers=HEADERS, timeout=120)
            if r.status_code != 200:
                continue
            js = r.json()
            ways_by_id = {el['id']: el for el in js['elements'] if el['type'] == 'way'}
            polys = []
            for el in js['elements']:
                if el['type'] != 'relation':
                    continue
                for mem in el.get('members', []):
                    if mem.get('role') == 'outer' and mem.get('type') == 'way':
                        w = ways_by_id.get(mem['ref'])
                        if w and 'geometry' in w and len(w['geometry']) >= 4:
                            coords = [(p['lon'], p['lat']) for p in w['geometry']]
                            try:
                                polys.append(Polygon(coords))
                            except Exception:
                                pass
            if polys:
                geom = _merge_polys(polys)
                if geom is not None:
                    return geom, {'source': 'OSM Overpass', 'query': q}
        except Exception:
            continue
    return None, None


# ======================================================================
# Punto de entrada
# ======================================================================
def resolve(study_area_arg, wdpa_id=None, name=None, wdpa_token=None, use_gee=False):
    """Devuelve (geom, info_dict) o (None, None).

    study_area_arg: path a un vector local (preferente).
    wdpa_id: ID WDPA. Resuelve con token Protected Planet, GEE (use_gee), Overpass
             o cache→Nominatim, en ese orden.
    name: nombre del área protegida (Nominatim).
    wdpa_token: token de Protected Planet (opcional).
    use_gee: si True y EE está inicializado, intenta la capa WDPA de Earth Engine.
    """
    # 1) Archivo vectorial
    if study_area_arg:
        p = Path(study_area_arg)
        if p.exists():
            geom = load_vector(p)
            return geom, _info(geom, source='file', name=p.name, path=str(p))

    # 2) WDPA ID
    if wdpa_id:
        wdpa_id = str(wdpa_id).strip()
        # 2a) API Protected Planet (token) — cualquier ID
        geom, meta = fetch_wdpa_api(wdpa_id, token=wdpa_token)
        if geom is not None:
            return geom, _info(geom, source='Protected Planet API', wdpa_id=wdpa_id,
                               name=(meta or {}).get('name', ''))
        # 2b) Google Earth Engine (capa WCMC/WDPA) — sin token
        if use_gee:
            try:
                import gee_backend as geb
                geom, meta = geb.wdpa_geometry(wdpa_id)
                if geom is not None:
                    return geom, _info(geom, source='GEE WDPA', wdpa_id=wdpa_id,
                                       name=(meta or {}).get('name', ''))
            except Exception:
                pass
        # 2c) Overpass por ref:WDPA
        geom, meta = fetch_overpass_protected_area(wdpa_id)
        if geom is not None:
            return geom, _info(geom, source='OSM Overpass via WDPA ID', wdpa_id=wdpa_id)
        # 2d) Cache de nombres conocidos → Nominatim
        known = WDPA_KNOWN.get(wdpa_id)
        if known:
            geom, meta = fetch_nominatim(known)
            if geom is not None:
                return geom, _info(geom, source='Nominatim (via WDPA→nombre)',
                                   wdpa_id=wdpa_id, query=known,
                                   name=(meta or {}).get('display_name', ''))

    # 3) Nombre libre
    if name:
        geom, meta = fetch_nominatim(name)
        if geom is not None:
            return geom, _info(geom, source='Nominatim', query=name,
                               name=(meta or {}).get('display_name', ''),
                               osm_id=(meta or {}).get('osm_id'))

    return None, None


def points_inside(rows, geom):
    """Marca cada row con 'inside_study_area' (True/False)."""
    from shapely.geometry import Point
    if geom is None:
        for r in rows:
            r['inside_study_area'] = ''
        return 0
    n_inside = 0
    for r in rows:
        p = Point(r['lon'], r['lat'])
        ins = geom.contains(p) or geom.intersects(p)
        r['inside_study_area'] = bool(ins)
        if ins:
            n_inside += 1
    return n_inside


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--file')
    ap.add_argument('--wdpa')
    ap.add_argument('--name')
    ap.add_argument('--token')
    ap.add_argument('--gee', action='store_true')
    args = ap.parse_args()
    geom, info = resolve(args.file, args.wdpa, args.name,
                         wdpa_token=args.token, use_gee=args.gee)
    if geom is not None:
        print(f'Encontrado: {info}')
        print(f'  Geometry type: {geom.geom_type}')
        print(f'  Bounds: {geom.bounds}')
        print(f'  Approx area: {geom.area * 12321:.1f} km²')
    else:
        print('No encontrado')
