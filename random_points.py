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
"""
import time, math
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


def filter_by_elevation(pts, min_elev=None, max_elev=None):
    """Filtra puntos por elevación SRTM.

    Devuelve (keep, keep_elevs, pool_elevs) donde `pool_elevs` son TODAS las
    elevaciones válidas del muestreo (para diagnosticar el rango altitudinal real
    del área cuando ningún punto cumple el filtro)."""
    if not (min_elev or max_elev):
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
             min_dist_m=100, seed=42):
    """
    Genera n puntos aleatorios dentro del área de estudio.

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
             'area_elev_min': None, 'area_elev_max': None}
    if mode == 'uniform':
        # Para tener un pool donde filtrar por elevación
        n_pool = n if not (min_elev or max_elev) else int(n * 3)
        raw = random_uniform(geom, n_pool, seed=seed)
    elif mode == 'poisson_disc':
        n_pool = n if not (min_elev or max_elev) else int(n * 2)
        raw = random_poisson_disc(geom, n_pool, min_dist_m=min_dist_m, seed=seed)
    elif mode == 'stratified_elev':
        # primero genera muchos puntos uniformes, después estratifica por banda
        raw = random_uniform(geom, n * 4, seed=seed)
    else:
        raise ValueError(f'Modo desconocido: {mode}')

    if min_elev or max_elev or mode == 'stratified_elev':
        print(f'  Consultando elevaciones (Open-Topo-Data SRTM 30 m)…')
        kept, elevs, pool_elevs = filter_by_elevation(raw, min_elev=min_elev, max_elev=max_elev)
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
        if len(kept) == 0 and not pool_elevs and (min_elev or max_elev):
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
