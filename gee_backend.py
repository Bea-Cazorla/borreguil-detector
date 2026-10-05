"""
gee_backend.py — Backend Google Earth Engine para el pipeline de borreguiles.

Alternativa a Microsoft Planetary Computer (MPC). Si el usuario tiene una cuenta
de Google Earth Engine, todo el cómputo de Sentinel-2 y de topografía se hace
en los servidores de Google (server-side), evitando descargas pesadas y sin
necesidad de MPC.

Autenticación:
- En local: ee.Authenticate() abre el navegador (flujo OAuth).
- Necesita el NOMBRE DEL PROYECTO de Google Cloud asociado a Earth Engine
  (p. ej. "ee-tunombre"), que se pasa a ee.Initialize(project=...).

Funciones públicas:
- gee_available()            -> bool (si earthengine-api está instalado)
- authenticate()            -> lanza el flujo de autenticación por navegador
- initialize(project)       -> inicializa la sesión; devuelve (ok, mensaje)
- is_initialized()          -> bool
- fetch_s2_gee(rows, bbox, years)   -> rellena índices Sentinel-2 en cada row
- fetch_topo_gee(rows, bbox)        -> rellena elev/slope/aspect/curvature

Las mismas variables y nombres de columna que el backend MPC, de modo que el
modelo Random Forest preentrenado es directamente aplicable.
"""
import math


def gee_available():
    try:
        import ee  # noqa
        return True
    except ImportError:
        return False


def is_initialized():
    try:
        import ee
        # Una llamada trivial; si no está inicializado lanza excepción
        ee.Number(1).getInfo()
        return True
    except Exception:
        return False


def wdpa_geometry(wdpa_id):
    """Devuelve (shapely_geom, meta) del área protegida con ese WDPA ID usando la
    capa WCMC/WDPA/current/polygons de Earth Engine. Requiere EE inicializado;
    NO necesita token de Protected Planet. (None, None) si no se encuentra.

    Conversión a shapely tolerante a entornos rotos: si la construcción de
    multi-geometría falla, degrada a la caja envolvente (bbox)."""
    try:
        import ee
    except Exception:
        return None, None
    try:
        wid = int(str(wdpa_id).strip())
    except Exception:
        return None, None
    try:
        fc = (ee.FeatureCollection('WCMC/WDPA/current/polygons')
              .filter(ee.Filter.eq('WDPAID', wid)))
        if not fc.size().getInfo():
            return None, None
        name = ''
        try:
            name = fc.first().get('NAME').getInfo() or ''
        except Exception:
            pass
        # simplify (maxError en metros) para limitar el tamaño del payload
        gj = fc.geometry().simplify(maxError=100).getInfo()
    except Exception:
        return None, None

    from shapely.geometry import shape as shp_shape, box as shp_box
    try:
        geom = shp_shape(gj)
    except Exception:
        xs, ys = [], []

        def _walk(c):
            if (isinstance(c, (list, tuple)) and len(c) >= 2
                    and isinstance(c[0], (int, float)) and isinstance(c[1], (int, float))):
                xs.append(c[0]); ys.append(c[1])
            elif isinstance(c, (list, tuple)):
                for x in c:
                    _walk(x)

        _walk(gj.get('coordinates', []))
        geom = shp_box(min(xs), min(ys), max(xs), max(ys)) if xs else None
    if geom is None:
        return None, None
    return geom, {'name': name, 'wdpaid': wid}


def authenticate(auth_mode=None):
    """Lanza el flujo de autenticación de Earth Engine.
    En local abre el navegador. auth_mode='notebook' da un flujo de copiar/pegar
    código (útil en entornos sin navegador)."""
    import ee
    if auth_mode:
        ee.Authenticate(auth_mode=auth_mode)
    else:
        ee.Authenticate()


def initialize(project=None, service_account_json=None):
    """Inicializa Earth Engine. Devuelve (ok: bool, mensaje: str).

    Para despliegue headless (Streamlit Cloud / HuggingFace), pasar
    service_account_json (ruta a un .json o el propio contenido JSON como str),
    o definir la variable de entorno EE_SERVICE_ACCOUNT_KEY. En local basta con
    haber ejecutado authenticate() una vez (flujo de navegador)."""
    import ee, os, json, tempfile
    # 1) Service account (deploy headless)
    sa = service_account_json or os.environ.get('EE_SERVICE_ACCOUNT_KEY')
    if sa:
        try:
            if sa.strip().startswith('{'):
                info = json.loads(sa)
                email = info['client_email']
                tmp = tempfile.NamedTemporaryFile('w', suffix='.json', delete=False)
                json.dump(info, tmp); tmp.close()
                key_path = tmp.name
            else:
                key_path = sa
                with open(key_path) as f:
                    email = json.load(f)['client_email']
            creds = ee.ServiceAccountCredentials(email, key_path)
            ee.Initialize(creds, project=project)
            ee.Number(1).getInfo()
            return True, f'Earth Engine inicializado (service account, proyecto: {project or "default"})'
        except Exception as e:
            return False, f'Fallo con service account: {e}'
    # 2) Credenciales de usuario (local, tras authenticate())
    try:
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
        # comprobar
        ee.Number(1).getInfo()
        return True, f'Earth Engine inicializado (proyecto: {project or "default"})'
    except Exception as e:
        msg = str(e)
        # Intento de autenticación automática si el problema es de credenciales
        if 'not been used' in msg or 'authenticate' in msg.lower() or 'credentials' in msg.lower():
            return False, ('No autenticado o proyecto sin Earth Engine API. '
                           'Ejecuta authenticate() y verifica el proyecto. Detalle: ' + msg)
        return False, msg


# ============================================================
# Índices Sentinel-2 (server-side)
# ============================================================
INDS = ['ndvi', 'ndwi', 'clre', 'gndvi', 'ndmi', 'evi', 'nbr']
SCL_VALID = [4, 5, 6, 7, 11]   # veg, suelo, agua, sin clasificar, nieve


def _utm_epsg(lat, lon):
    zone = int((lon + 180) / 6) + 1
    return 32600 + zone if lat >= 0 else 32700 + zone


def _mask_scl(img):
    """Máscara de nubes/sombras usando la banda SCL de S2_SR_HARMONIZED."""
    import ee
    scl = img.select('SCL')
    mask = scl.remap(SCL_VALID, [1] * len(SCL_VALID), 0)
    return img.updateMask(mask)


def _indices_image(composite):
    """Calcula los 7 índices sobre un composite de reflectancia (0-1)."""
    import ee
    b2 = composite.select('B2')
    b3 = composite.select('B3')
    b4 = composite.select('B4')
    b5 = composite.select('B5')
    b8 = composite.select('B8')
    b11 = composite.select('B11')
    b12 = composite.select('B12')
    ndvi = b8.subtract(b4).divide(b8.add(b4).add(1e-6)).rename('ndvi')
    ndwi = b3.subtract(b8).divide(b3.add(b8).add(1e-6)).rename('ndwi')
    clre = b8.divide(b5.add(1e-6)).subtract(1).rename('clre')
    gndvi = b8.subtract(b3).divide(b8.add(b3).add(1e-6)).rename('gndvi')
    ndmi = b8.subtract(b11).divide(b8.add(b11).add(1e-6)).rename('ndmi')
    evi = b8.subtract(b4).multiply(2.5).divide(
        b8.add(b4.multiply(6)).subtract(b2.multiply(7.5)).add(1).add(1e-6)).rename('evi')
    nbr = b8.subtract(b12).divide(b8.add(b12).add(1e-6)).rename('nbr')
    return ee.Image.cat([ndvi, ndwi, clre, gndvi, ndmi, evi, nbr])


# ---- Índices NUEVOS (Indices_Mascaras_S2.txt): WAVI/ALBEDO/NDMI/NDWI ----
NEW_INDS = ['wavi', 'albedo', 'ndmi', 'ndwi']
_ALB = [0.1836, 0.1759, 0.1456, 0.1347, 0.1233, 0.1134, 0.1001, 0.0231, 0.0003]


def _s2cloudless_mask(img):
    """Máscara de nubes con s2cloudless (COPERNICUS/S2_CLOUD_PROBABILITY ≤ 30),
    ventana ±1 día (applyCloudFilter1 del spec)."""
    import ee
    prob = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
            .filterDate(img.date().advance(-1, 'day'), img.date().advance(1, 'day'))
            .mean().select('probability'))
    return img.updateMask(prob.lte(30))


def _snow_mask(img):
    """Máscara de nieve: NDSI < 0.4  y  SCL ≠ 11."""
    import ee
    ndsi = img.normalizedDifference(['B3', 'B11'])
    return img.updateMask(ndsi.lt(0.4)).updateMask(img.select('SCL').neq(11))


def _add_new_indices(img):
    """Devuelve una imagen con las 4 bandas nuevas (reflectancia = banda/10000)."""
    import ee
    r = img.divide(10000)
    b2, b3, b4 = r.select('B2'), r.select('B3'), r.select('B4')
    b5, b6, b7 = r.select('B5'), r.select('B6'), r.select('B7')
    b8, b11, b12 = r.select('B8'), r.select('B11'), r.select('B12')
    ndmi = b8.subtract(b11).divide(b8.add(b11).add(1e-6)).rename('ndmi')
    ndwi = b3.subtract(b8).divide(b3.add(b8).add(1e-6)).rename('ndwi')
    L = 0.5
    wavi = b8.subtract(b2).multiply(1 + L).divide(b8.add(b2).add(L).add(1e-6)).rename('wavi')
    albedo = (b2.multiply(_ALB[0]).add(b3.multiply(_ALB[1])).add(b4.multiply(_ALB[2]))
              .add(b5.multiply(_ALB[3])).add(b6.multiply(_ALB[4])).add(b7.multiply(_ALB[5]))
              .add(b8.multiply(_ALB[6])).add(b11.multiply(_ALB[7])).add(b12.multiply(_ALB[8]))
              ).clamp(0, 1).rename('albedo')
    return ee.Image.cat([wavi, albedo, ndmi, ndwi]).copyProperties(img, ['system:time_start'])


def _period_new_stats(region, years):
    """Imagen con 16 bandas: {wavi,albedo,ndmi,ndwi}_{mean,min,max,sd} sobre
    jun–sep de los años indicados, con máscara s2cloudless + nieve."""
    import ee
    col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
           .filterBounds(region)
           .filter(ee.Filter.calendarRange(6, 9, 'month')))
    yf = [ee.Filter.date(f'{y}-06-01', f'{y}-09-30') for y in years]
    combo = yf[0]
    for f in yf[1:]:
        combo = ee.Filter.Or(combo, f)
    col = (col.filter(combo)
           .map(_s2cloudless_mask)
           .map(_snow_mask)
           .map(_add_new_indices))
    red = (ee.Reducer.mean()
           .combine(ee.Reducer.minMax(), sharedInputs=True)
           .combine(ee.Reducer.stdDev(), sharedInputs=True))
    stats = col.reduce(red)
    # _stdDev → _sd para casar con FEATURES_RF
    stats = stats.rename(stats.bandNames().map(
        lambda nm: ee.String(nm).replace('_stdDev', '_sd')))
    return stats


def _season_composite(region, years, season):
    """Composite mediano (median) de reflectancia para una estación.
    season: 'early' (15-jun a 25-jul) o 'late' (1-ago a 30-sep), combinando años."""
    import ee
    col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
           .filterBounds(region)
           .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40)))
    # Filtro temporal: unión de las ventanas de cada año
    filters = []
    for yr in years:
        if season == 'early':
            filters.append(ee.Filter.date(f'{yr}-06-15', f'{yr}-07-25'))
        else:
            filters.append(ee.Filter.date(f'{yr}-08-01', f'{yr}-09-30'))
    combo = filters[0]
    for f in filters[1:]:
        combo = ee.Filter.Or(combo, f)
    col = col.filter(combo).map(_mask_scl)
    # Reflectancia 0-1
    bands = ['B2', 'B3', 'B4', 'B5', 'B8', 'B11', 'B12']
    col = col.select(bands).map(lambda im: im.divide(10000))
    return col.median()


def _s2_stack(region, years):
    """Las bandas de Sentinel-2 que usa la app: índices de inicio y fin de verano,
    la caída del NDVI y las estadísticas de jun–sep. Devuelve (pila, estadísticas).
    Es la ÚNICA definición: la usan el muestreo de los puntos y el mapa."""
    import ee
    early = _indices_image(_season_composite(region, years, 'early'))
    late = _indices_image(_season_composite(region, years, 'late'))
    early = early.rename([f'{k}_early' for k in INDS])
    late = late.rename([f'{k}_late' for k in INDS])
    drop = early.select('ndvi_early').subtract(late.select('ndvi_late')).rename('ndvi_drop')
    # Índices nuevos (mean/min/max/sd sobre jun–sep)
    new_stats = _period_new_stats(region, years)
    return ee.Image.cat([early, late, drop, new_stats]), new_stats


def fetch_s2_gee(rows, bbox, years=(2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025),
                 buffer_m=25, chunk=400, progress=None):
    """Rellena en cada row los índices Sentinel-2 (early/late) usando GEE."""
    import ee
    print('  → Sentinel-2 vía Google Earth Engine (server-side)…')
    try:
        # Límite por petición: sin él, un getInfo pesado puede quedarse colgado
        # MUCHO tiempo. Con deadline, falla a los 3 min y la sub-división reintenta
        # con lotes menores (progreso visible en vez de cuelgue aparente).
        ee.data.setDeadline(300_000)
    except Exception:
        pass
    west, south, east, north = bbox[0], bbox[1], bbox[2], bbox[3]
    region = ee.Geometry.Rectangle([west, south, east, north])

    stack, new_stats = _s2_stack(region, years)
    try:
        new_keys = list(new_stats.bandNames().getInfo())
    except Exception:
        new_keys = [f'{k}_{s}' for k in NEW_INDS for s in ('mean', 'min', 'max', 'sd')]

    # Inicializa NaN
    out_keys = ([f'{k}_early' for k in INDS] + [f'{k}_late' for k in INDS]
                + ['ndvi_drop'] + new_keys)
    for r in rows:
        for k in out_keys:
            r.setdefault(k, float('nan'))

    n = len(rows)
    stats = {'ok': 0, 'failed': 0, 'last_error': None}

    def sample_chunk(sub, base_idx):
        """Muestrea un sub-lote. Si getInfo() falla (límite de tiempo/memoria de GEE,
        típico con muchos puntos), divide el lote en dos y reintenta; así un fallo
        no deja TODO el lote en NaN silenciosamente. Por debajo de un mínimo, registra
        el fallo de forma visible (no se traga el error)."""
        feats = [ee.Feature(ee.Geometry.Point([r['lon'], r['lat']]).buffer(buffer_m),
                            {'idx': base_idx + i}) for i, r in enumerate(sub)]
        fc = ee.FeatureCollection(feats)
        # tileScale=4: GEE divide el cómputo en tiles menores → evita 'User memory limit
        # exceeded' con muchos puntos (esos fallos disparaban reintentos lentos).
        sampled = stack.reduceRegions(collection=fc, reducer=ee.Reducer.mean(),
                                      scale=10, tileScale=4)
        try:
            data = sampled.getInfo()['features']
        except Exception as e:
            if len(sub) > 25:                       # divide y reintenta (lotes menores)
                mid = len(sub) // 2
                sample_chunk(sub[:mid], base_idx)
                sample_chunk(sub[mid:], base_idx + mid)
            else:
                stats['failed'] += len(sub)
                stats['last_error'] = str(e)
                print(f'    sub-lote {base_idx} ({len(sub)} pts): error getInfo ({e})')
            return
        for f in data:
            props = f['properties']
            idx = props.get('idx')
            if idx is None:
                continue
            for k in out_keys:
                v = props.get(k)
                if v is not None:
                    rows[idx][k] = float(v)
        stats['ok'] += len(sub)

    for start in range(0, n, chunk):
        sample_chunk(rows[start:start + chunk], start)
        done = min(start + chunk, n)
        print(f'    {done}/{n} puntos muestreados '
              f'(ok={stats["ok"]}, fallidos={stats["failed"]})')
        if progress:
            try: progress(done / n, f'Sentinel-2 (GEE): {done}/{n} puntos')
            except Exception: pass

    # Si NINGÚN punto se pudo muestrear, propaga el error para que la UI lo informe
    if stats['ok'] == 0 and n > 0:
        raise RuntimeError(
            'Earth Engine no devolvió datos para ningún punto '
            f'(último error: {stats["last_error"] or "desconocido"}). '
            'Suele pasar con MUCHOS puntos a la vez: reduce el nº de puntos o usa '
            'el backend Microsoft Planetary Computer (MPC).')
    return stats['failed']


# ============================================================
# Topografía (server-side)
# ============================================================
# Modelo de elevaciones: Copernicus GLO-30, edición 2024_1. Sustituye a
# COPERNICUS/DEM/GLO30, que Earth Engine marca como obsoleta. Mismas bandas, mismo
# tipo y misma rejilla; en Sierra Nevada los valores son idénticos (comprobado en
# 6 puntos: diferencia 0,000 m).
DEM_ASSET = 'COPERNICUS/DEM/GLO30_2024_1'
TOPO_KEYS = ('elev_dem_m', 'slope_deg', 'aspect_north', 'aspect_east', 'curvature')


def _topo_stack(epsg):
    """Las 5 bandas de topografía, calculadas en la rejilla UTM de 30 m. Es la ÚNICA
    definición: la usan el muestreo de los puntos y el mapa."""
    import ee
    dem = (ee.ImageCollection(DEM_ASSET).select('DEM')
           .mosaic().setDefaultProjection('EPSG:4326', None, 30)
           .reproject(crs=f'EPSG:{epsg}', scale=30))
    # Pendiente y orientación en grados CON decimales. ee.Terrain.products las
    # devuelve como enteros (14°, no 14,4°), y así llegaban a los puntos.
    slope = ee.Terrain.slope(dem).rename('slope_deg')
    aspect = ee.Terrain.aspect(dem)
    aspect_rad = aspect.multiply(math.pi / 180.0)
    aspect_north = aspect_rad.cos().rename('aspect_north')
    aspect_east = aspect_rad.sin().rename('aspect_east')
    # Curvatura: laplaciano aproximado por convolución
    lap_kernel = ee.Kernel.laplacian8(normalize=False)
    curvature = dem.convolve(lap_kernel).rename('curvature')
    elev = dem.rename('elev_dem_m')
    return ee.Image.cat([elev, slope, aspect_north, aspect_east, curvature])


def fetch_topo_gee(rows, bbox, buffer_m=15, chunk=400):
    """Rellena elev/slope/aspect/curvature usando Copernicus DEM GLO30 en GEE."""
    import ee
    print('  → Topografía vía Google Earth Engine (Copernicus GLO30)…')
    west, south, east, north = bbox[0], bbox[1], bbox[2], bbox[3]
    region = ee.Geometry.Rectangle([west, south, east, north])
    lat_med = (south + north) / 2
    lon_med = (west + east) / 2
    epsg = _utm_epsg(lat_med, lon_med)

    stack = _topo_stack(epsg)
    keys = list(TOPO_KEYS)
    for r in rows:
        for k in keys:
            r.setdefault(k, float('nan'))
        r.setdefault('twi', float('nan'))  # TWI no se calcula en GEE (se imputa)

    n = len(rows)
    for start in range(0, n, chunk):
        sub = rows[start:start + chunk]
        feats = [ee.Feature(ee.Geometry.Point([r['lon'], r['lat']]).buffer(buffer_m),
                            {'idx': start + i}) for i, r in enumerate(sub)]
        fc = ee.FeatureCollection(feats)
        sampled = stack.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=30)
        try:
            data = sampled.getInfo()['features']
        except Exception as e:
            print(f'    chunk {start}: error getInfo ({e})')
            continue
        for f in data:
            props = f['properties']
            idx = props.get('idx')
            if idx is None:
                continue
            for k in keys:
                v = props.get(k)
                if v is not None:
                    rows[idx][k] = float(v)
        print(f'    {min(start+chunk, n)}/{n} puntos muestreados')


# ============================================================
# Pila de predictores para el MAPEO (Random Forest de tipos)
# ------------------------------------------------------------
# Todas las variables que existen como mapa continuo, en una sola imagen y sobre una
# única rejilla de referencia: UTM, 10 m, alineada con la de Sentinel-2. De esa
# misma imagen salen el valor de cada punto de entrenamiento (muestrear_pila) y,
# después, los píxeles que se clasifican: así el modelo ve en el mapa exactamente lo
# mismo que vio al entrenar.
# ============================================================
ESCALA_M = 10
# GNDVI = (NIR − verde)/(NIR + verde) es exactamente −NDWI: la misma variable con el
# signo cambiado. El detector las tiene las dos (sus modelos se entrenaron así); en
# el mapeo se deja solo NDWI.
INDS_MAPEO = tuple(k for k in INDS if k != 'gndvi')
BANDAS_S2 = tuple([f'{k}_early' for k in INDS_MAPEO] + [f'{k}_late' for k in INDS_MAPEO]
                  + ['ndvi_drop']
                  + [f'{k}_{s}' for k in NEW_INDS for s in ('mean', 'min', 'max', 'sd')])
BANDAS_PILA = BANDAS_S2 + TOPO_KEYS            # 34 bandas, en ORDEN FIJO


def utm_de(puntos):
    """EPSG de la zona UTM del centro de unos puntos [(lon, lat), …]."""
    lon = sum(p[0] for p in puntos) / len(puntos)
    lat = sum(p[1] for p in puntos) / len(puntos)
    return _utm_epsg(lat, lon)


def rejilla(epsg):
    """Rejilla de referencia del mapeo: los bordes de píxel caen en múltiplos de 10 m
    de las coordenadas UTM, igual que en las teselas de Sentinel-2."""
    return {'crs': f'EPSG:{epsg}', 'crsTransform': [ESCALA_M, 0, 0, 0, -ESCALA_M, 0]}


def pila_predictores(region, years, epsg):
    """ee.Image con las bandas de BANDAS_PILA.

    Sentinel-2 llega en su rejilla (10 m; las bandas de 20 m repiten valor). La
    topografía se calcula a 30 m y se lleva a 10 m por interpolación BILINEAL, que es
    lo indicado para una variable continua (repetir el píxel de 30 m dibujaría
    escalones)."""
    import ee
    s2, _ = _s2_stack(region, years)
    topo = _topo_stack(epsg).resample('bilinear')
    return ee.Image.cat([s2, topo]).select(list(BANDAS_PILA))


def muestrear_pila(pila, puntos, epsg, bandas=None, chunk=300, progress=None):
    """Valor de cada banda en el PÍXEL de 10 m que contiene cada punto (no la media de
    un entorno, como en el detector: aquí tiene que coincidir con lo que luego se
    clasifica).

    puntos: [(lon, lat), …]. Devuelve (valores, fallidos): `valores` es una lista
    paralela; cada elemento es un dict {banda: número | None} —None en la banda sin
    dato en ese píxel— o None si el punto NO SE PUDO CONSULTAR (fallo de Earth
    Engine: no es lo mismo que no tener dato, y hay que poder reintentarlo).
    `fallidos` es cuántos puntos quedaron sin consultar. Si no se pudo consultar
    ninguno, lanza RuntimeError con el motivo.
    """
    import ee
    bandas = list(bandas or BANDAS_PILA)
    g = rejilla(epsg)
    img = pila.select(bandas)
    out = [None] * len(puntos)
    estado = {'ok': 0, 'fallidos': 0, 'error': None}
    try:
        ee.data.setDeadline(300_000)
    except Exception:
        pass

    def lote(sub, base):
        fc = ee.FeatureCollection([
            ee.Feature(ee.Geometry.Point([lo, la]), {'idx': base + i})
            for i, (lo, la) in enumerate(sub)])
        red = img.reduceRegions(collection=fc, reducer=ee.Reducer.first(),
                                crs=g['crs'], crsTransform=g['crsTransform'],
                                tileScale=4)
        try:
            feats = red.getInfo()['features']
        except Exception as e:
            if len(sub) > 20:                   # divide el lote y reintenta
                mid = len(sub) // 2
                lote(sub[:mid], base)
                lote(sub[mid:], base + mid)
            else:
                estado['fallidos'] += len(sub)
                estado['error'] = str(e)
            return
        for f in feats:
            props = f.get('properties', {})
            i = props.get('idx')
            if i is None:
                continue
            out[i] = {b: (float(props[b]) if props.get(b) is not None else None)
                      for b in bandas}
        estado['ok'] += len(sub)

    n = len(puntos)
    for ini in range(0, n, chunk):
        lote(puntos[ini:ini + chunk], ini)
        if progress:
            try:
                progress(min(ini + chunk, n) / n)
            except Exception:
                pass
    if n and estado['ok'] == 0:
        raise RuntimeError('Earth Engine no devolvió datos para ningún punto '
                           f'(último error: {estado["error"] or "desconocido"}).')
    return out, estado['fallidos']


def leer_predictores(puntos, years, epsg=None, progress=None):
    """Lee las variables de la pila en el píxel de cada punto [(lon, lat), …].

    Devuelve (valores, fallidos, info): `valores` y `fallidos` como en
    muestrear_pila; `info` describe de dónde salen los datos (rejilla, años,
    colecciones), para poder reproducir la lectura. `epsg` fija la zona UTM de la
    rejilla; si no se da, se toma la del centro de los puntos.
    """
    import ee
    years = tuple(int(y) for y in years)
    epsg = int(epsg or utm_de(puntos))
    lons = [p[0] for p in puntos]
    lats = [p[1] for p in puntos]
    margen = 0.01                               # ~1 km: solo acota la búsqueda de escenas
    bbox = (min(lons) - margen, min(lats) - margen, max(lons) + margen, max(lats) + margen)
    pila = pila_predictores(ee.Geometry.Rectangle(list(bbox)), years, epsg)
    valores, fallidos = muestrear_pila(pila, puntos, epsg, progress=progress)
    return valores, fallidos, info_pila(years, epsg)


def info_pila(years, epsg):
    """De dónde salen los valores de la pila: lo necesario para repetir la lectura."""
    return {
        'crs': f'EPSG:{int(epsg)}', 'resolucion_m': ESCALA_M,
        'rejilla': 'UTM, bordes de píxel en múltiplos de 10 m (la de Sentinel-2)',
        'anios': [int(y) for y in years],
        'bandas': list(BANDAS_PILA),
        'sentinel2': 'COPERNICUS/S2_SR_HARMONIZED',
        'elevaciones': DEM_ASSET,
        'remuestreo': 'Sentinel-2: vecino más próximo; topografía (30 m): bilineal',
    }


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', help='Proyecto de Google Earth Engine')
    ap.add_argument('--auth', action='store_true', help='Lanzar autenticación')
    args = ap.parse_args()
    if not gee_available():
        print('earthengine-api no está instalado (pip install earthengine-api)')
        raise SystemExit(1)
    if args.auth:
        authenticate()
    ok, msg = initialize(args.project)
    print(msg)
    if ok:
        # Mini test
        rows = [{'lon': -3.27045, 'lat': 37.12285},
                {'lon': -3.40619, 'lat': 37.03503}]
        bbox = (-3.45, 37.0, -3.2, 37.15)
        fetch_topo_gee(rows, bbox)
        fetch_s2_gee(rows, bbox, years=(2023, 2024))
        for r in rows:
            print(r)
