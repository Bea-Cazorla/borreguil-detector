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


def _period_new_stats(region, years, por_anio=False):
    """Imagen con 16 bandas: {wavi,albedo,ndmi,ndwi}_{mean,min,max,sd} sobre
    jun–sep de los años indicados, con máscara s2cloudless + nieve.

    Hay dos maneras de juntar los años:

      · todas las imágenes de todos los años a la vez (por defecto);
      · `por_anio`: la estadística se calcula en CADA año y después se hace la media
        entre los años que tienen dato. Es lo que usa el mapeo. Con todo junto, el
        mínimo y el máximo son los de una sola imagen en nueve veranos (basta una
        sombra de nube), y la variabilidad mezcla el cambio dentro del verano con las
        diferencias entre años; así, una imagen mala pesa la parte de su año.
    """
    import ee
    base = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(region)
            .filter(ee.Filter.calendarRange(6, 9, 'month')))
    red = (ee.Reducer.mean()
           .combine(ee.Reducer.minMax(), sharedInputs=True)
           .combine(ee.Reducer.stdDev(), sharedInputs=True))

    def indices(filtro):
        return (base.filter(filtro)
                .map(_s2cloudless_mask)
                .map(_snow_mask)
                .map(_add_new_indices))

    veranos = [ee.Filter.date(f'{y}-06-01', f'{y}-09-30') for y in years]
    if not por_anio:
        combo = veranos[0]
        for f in veranos[1:]:
            combo = ee.Filter.Or(combo, f)
        stats = indices(combo).reduce(red)
        # _stdDev → _sd para casar con FEATURES_RF
        return stats.rename(stats.bandNames().map(
            lambda nm: ee.String(nm).replace('_stdDev', '_sd')))

    nombres = [f'{k}_{s}' for k in NEW_INDS for s in ('mean', 'min', 'max', 'sd')]
    # Una imagen sin dato en ningún píxel: con ella, un año sin escenas da igualmente
    # sus 16 bandas (vacías) en vez de una imagen sin bandas, y no cuenta en la media.
    # (Para juntarla con las demás, todas tienen que ser del mismo tipo de número.)
    vacia = (ee.Image.constant([0] * len(NEW_INDS)).rename(NEW_INDS).toDouble()
             .updateMask(0))
    red_anual = red.combine(ee.Reducer.count(), sharedInputs=True)
    anuales = []
    for verano in veranos:
        st = (indices(verano).map(lambda im: ee.Image(im).toDouble())
              .merge(ee.ImageCollection([vacia])).reduce(red_anual))
        partes = []
        for k in NEW_INDS:
            partes.append(st.select([f'{k}_mean', f'{k}_min', f'{k}_max']))
            # con una sola imagen válida en el año no hay variabilidad que medir
            partes.append(st.select(f'{k}_stdDev')
                          .updateMask(st.select(f'{k}_count').gte(2)))
        anuales.append(ee.Image.cat(partes).rename(nombres))
    return ee.ImageCollection(anuales).mean()


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


def _s2_stack(region, years, por_anio=False):
    """Las bandas de Sentinel-2 del MAPEO: índices de inicio y fin de verano (mediana
    de todas las imágenes de las ventanas), la caída del NDVI y las estadísticas de
    jun–sep. Devuelve (pila, estadísticas). El mapeo entrena sus propios modelos con
    estas bandas (y año a año, `por_anio`), así que solo tiene que ser coherente
    consigo mismo. El detector NO la usa: ver _pila_detector."""
    import ee
    early = _indices_image(_season_composite(region, years, 'early'))
    late = _indices_image(_season_composite(region, years, 'late'))
    early = early.rename([f'{k}_early' for k in INDS])
    late = late.rename([f'{k}_late' for k in INDS])
    drop = early.select('ndvi_early').subtract(late.select('ndvi_late')).rename('ndvi_drop')
    # Índices nuevos (mean/min/max/sd sobre jun–sep)
    new_stats = _period_new_stats(region, years, por_anio)
    return ee.Image.cat([early, late, drop, new_stats]), new_stats


# ---- Detector: las variables tal como se entrenaron los modelos de partida ----
# Los modelos de Sierra Nevada se entrenaron con variables de Planetary Computer
# (borreguil_pipeline.fetch_s2), y el modelo solo acierta si recibe los números
# calculados de la misma manera. Calcularlas «a la manera de Earth Engine» (mediana de
# todas las imágenes) da otros números: comprobado el 19-11-2026 en los 796 puntos de
# entrenamiento, el EVI salía ×0,3, la variabilidad del NDWI y del NDMI ×2 y la caída
# del NDVI muy distinta (r 0,46), y el modelo bajaba de AUC 0,83 a 0,79 en cuencas que
# no había visto. Con esta imitación las variables coinciden (r ≥ 0,9 en las que más
# pesan) y el acierto es el mismo (AUC 0,83). Lo que imita de fetch_s2:
#   · ventanas 15 jun–25 jul y 1 ago–30 sep de cada año, escenas con nubes < 30 %;
#   · de cada ventana, la ÚLTIMA escena válida (fetch_s2 se queda con las primeras que
#     devuelve el catálogo, que van de la más reciente a la más antigua);
#   · en cada escena, la MEDIA de las bandas en una ventana de 5 × 5 píxeles (10 m) y
#     de 3 × 3 (20 m) alrededor del punto, y el índice con esas medias; la escena vale
#     si al menos la mitad de la ventana 3 × 3 de SCL es vegetación, suelo, agua, sin
#     clasificar o nieve;
#   · inicio y fin de verano: mediana entre años;
#   · el EVI, el normal (reflectancia 0–1 y su «+1»). Hasta el 19-11-2026 los modelos se
#     entrenaron con un EVI calculado en números digitales (0–10 000), donde el «+1» casi
#     no pesa y el EVI salía unas 3 veces mayor (pasaba de 1); ese día se recalculó el EVI
#     normal de los 796 puntos y se reentrenaron los modelos;
#   · estadísticas de WAVI, albedo, NDMI y NDWI: con las mismas escenas elegidas, todos
#     los años juntos, sin nieve (SCL) y con NDSI de la ventana < 0,4.
# Solo se usa para los puntos del detector; el mapeo tiene su propia pila (_s2_stack).
_SCL_VALIDA = [4, 5, 6, 7, 11]          # vegetación, suelo, agua, sin clasificar, nieve
_SCL_SIN_NIEVE = [4, 5, 6, 7]
CLAVES_S2 = ([f'{k}_early' for k in INDS] + [f'{k}_late' for k in INDS] + ['ndvi_drop']
             + [f'{k}_{s}' for k in NEW_INDS for s in ('mean', 'min', 'max', 'sd')])


def _escena_detector(img):
    """Índices de una escena con la media de la ventana de cada punto (ver arriba)."""
    import ee
    p10 = img.select('B2').projection()
    p20 = img.select('B5').projection()
    k5 = ee.Kernel.square(2, 'pixels')
    k3 = ee.Kernel.square(1, 'pixels')
    n10 = ['B2', 'B3', 'B4', 'B8']
    n20 = ['B5', 'B6', 'B7', 'B11', 'B12']
    b10 = img.select(n10)
    b20 = img.select(n20)
    # Como en fetch_s2, un píxel ≤ 0 es «sin dato» y no entra en la media.
    # reproject: la ventana se mide en píxeles de la propia escena (10 m o 20 m)
    m10 = (b10.updateMask(b10.gt(0)).reduceNeighborhood(ee.Reducer.mean(), k5)
           .reproject(p10).rename(n10).divide(10000))
    m20 = (b20.updateMask(b20.gt(0)).reduceNeighborhood(ee.Reducer.mean(), k3)
           .reproject(p20).rename(n20).divide(10000))
    scl = img.select('SCL')
    valida = (scl.remap(_SCL_VALIDA, [1] * len(_SCL_VALIDA), 0)
              .reduceNeighborhood(ee.Reducer.mean(), k3).reproject(p20).gte(0.5))
    sin_nieve = (scl.remap(_SCL_SIN_NIEVE, [1] * len(_SCL_SIN_NIEVE), 0)
                 .reduceNeighborhood(ee.Reducer.mean(), k3).reproject(p20).gte(0.5))
    b2, b3, b4, b8 = (m10.select(b) for b in n10)
    b5, b6, b7, b11, b12 = (m20.select(b) for b in n20)
    e = 1e-6

    def nd(a, b):
        return a.subtract(b).divide(a.add(b).add(e))

    ind = ee.Image.cat([
        nd(b8, b4).rename('ndvi'),
        nd(b3, b8).rename('ndwi'),
        b8.divide(b5.add(e)).subtract(1).rename('clre'),
        nd(b8, b3).rename('gndvi'),
        nd(b8, b11).rename('ndmi'),
        b8.subtract(b4).multiply(2.5).divide(
            b8.add(b4.multiply(6)).subtract(b2.multiply(7.5)).add(1).add(e)
        ).rename('evi'),
        nd(b8, b12).rename('nbr'),
    ]).updateMask(valida)
    alb = (b2.multiply(_ALB[0]).add(b3.multiply(_ALB[1])).add(b4.multiply(_ALB[2]))
           .add(b5.multiply(_ALB[3])).add(b6.multiply(_ALB[4])).add(b7.multiply(_ALB[5]))
           .add(b8.multiply(_ALB[6])).add(b11.multiply(_ALB[7])).add(b12.multiply(_ALB[8]))
           ).clamp(0, 1)
    L = 0.5
    nuevos = (ee.Image.cat([
        b8.subtract(b2).multiply(1 + L).divide(b8.add(b2).add(L).add(e)).rename('n_wavi'),
        alb.rename('n_albedo'),
        nd(b8, b11).rename('n_ndmi'),
        nd(b3, b8).rename('n_ndwi'),
    ]).updateMask(valida).updateMask(sin_nieve).updateMask(nd(b3, b11).lt(0.4)))
    return ee.Image.cat([ind, nuevos]).copyProperties(img, ['system:time_start'])


def _pila_detector(region, years):
    """ee.Image con las 31 bandas Sentinel-2 del detector (CLAVES_S2), calculadas como
    en el entrenamiento de los modelos de partida. Se lee en el píxel del punto: la
    media de su entorno ya va dentro (ventanas 5 × 5 y 3 × 3)."""
    import ee
    base = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(region)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)))
    nuevos = [f'n_{k}' for k in NEW_INDS]
    early, late, escenas = [], [], []
    for y in years:
        for lista, (a, b) in ((early, (f'{y}-06-15', f'{y}-07-26')),
                              (late, (f'{y}-08-01', f'{y}-10-01'))):
            # mosaic() pone encima la última imagen: en cada píxel queda la escena
            # válida más reciente de la ventana
            ultima = (base.filterDate(a, b).map(_escena_detector)
                      .sort('system:time_start').mosaic())
            lista.append(ultima.select(INDS))
            escenas.append(ultima.select(nuevos))
    e = ee.ImageCollection(early).median().rename([f'{k}_early' for k in INDS])
    l = ee.ImageCollection(late).median().rename([f'{k}_late' for k in INDS])
    drop = e.select('ndvi_early').subtract(l.select('ndvi_late')).rename('ndvi_drop')
    red = (ee.Reducer.mean().combine(ee.Reducer.minMax(), sharedInputs=True)
           .combine(ee.Reducer.stdDev(), sharedInputs=True))   # stdDev: poblacional
    st = ee.ImageCollection(escenas).reduce(red)
    de, a = [], []
    for k in NEW_INDS:
        for s, s2 in (('mean', 'mean'), ('min', 'min'), ('max', 'max'), ('stdDev', 'sd')):
            de.append(f'n_{k}_{s}'); a.append(f'{k}_{s2}')
    return ee.Image.cat([e, l, drop, st.select(de, a)]).select(CLAVES_S2)


def fetch_s2_gee(rows, bbox, years=(2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025),
                 chunk=200, progress=None):
    """Rellena en cada row las variables Sentinel-2 del detector usando GEE, calculadas
    como en el entrenamiento de los modelos de partida (ver _pila_detector)."""
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

    stack = _pila_detector(region, years)
    # Se lee el píxel de 10 m que contiene el punto, en la rejilla UTM de Sentinel-2
    epsg = _utm_epsg((south + north) / 2, (west + east) / 2)
    return _leer_en_puntos(stack, rows, list(CLAVES_S2), epsg, chunk, 'Sentinel-2',
                           progress)


def _leer_en_puntos(stack, rows, out_keys, epsg, chunk, nombre, progress=None):
    """Lee las bandas `out_keys` de `stack` en el píxel de 10 m de cada punto y las
    escribe en cada row. Devuelve cuántos puntos fallaron."""
    import ee
    # Inicializa NaN
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
        feats = [ee.Feature(ee.Geometry.Point([r['lon'], r['lat']]),
                            {'idx': base_idx + i}) for i, r in enumerate(sub)]
        fc = ee.FeatureCollection(feats)
        # tileScale=4: GEE divide el cómputo en tiles menores → evita 'User memory limit
        # exceeded' con muchos puntos (esos fallos disparaban reintentos lentos).
        sampled = stack.reduceRegions(collection=fc, reducer=ee.Reducer.mean(),
                                      scale=10, crs=f'EPSG:{epsg}', tileScale=4)
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
            try: progress(done / n, f'{nombre} (GEE): {done}/{n} puntos')
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
# Sentinel-1 (radar) del detector
# ------------------------------------------------------------
# Antes se leía de Planetary Computer (sentinel-1-rtc): más de media hora con 20
# puntos. Desde el 19-11-2026 se lee aquí, en segundos, y los modelos de partida se
# reentrenaron con estos valores (classification_v7.csv).
#   · COPERNICUS/S1_GRD, modo IW, VV y VH, 10 m, jun–sep de S1_ANIOS.
#   · Corrección del relieve: gamma-0 aplanado con el modelo de volumen de Vollrath
#     et al. (2020, Remote Sensing 12: 1867), con el DEM Copernicus GLO-30; fuera las
#     zonas en sombra o con inversión por relieve (layover).
#   · En cada escena, media de la ventana 5 × 5 píxeles (en potencia, no en dB) con al
#     menos 8 píxeles válidos, como se hacía con Planetary Computer.
#   · De todas las escenas: media y desviación típica en dB de VV y VH, y VV − VH.
# ============================================================
S1_ANIOS = (2019, 2020, 2021, 2022, 2023, 2024)
CLAVES_S1 = ('s1_vv_mean', 's1_vh_mean', 's1_vv_sd', 's1_vh_sd', 's1_ratio_mean')


def _escena_s1(img, pendiente, orientacion):
    """gamma-0 aplanado (potencia) de una escena, con la media de la ventana 5 × 5."""
    import ee
    rad = math.pi / 180
    proj = img.select('VV').projection()
    angulo = img.select('angle')
    # Dirección en que mira el radar: hacia donde crece el ángulo de incidencia
    mira = (ee.Terrain.aspect(angulo)
            .reduceRegion(ee.Reducer.mean(), img.geometry(), 1000).get('aspect'))
    theta = angulo.multiply(rad)
    phi_r = ee.Image.constant(mira).multiply(rad).subtract(orientacion)
    alfa_r = pendiente.tan().multiply(phi_r.cos()).atan()      # pendiente hacia el radar
    noventa = ee.Image.constant(math.pi / 2)
    potencia = ee.Image.constant(10).pow(img.select(['VV', 'VH']).divide(10))
    gamma0 = potencia.divide(theta.cos())
    factor = (noventa.subtract(theta).add(alfa_r).tan()
              .divide(noventa.subtract(theta).tan()))
    sin_inversion = alfa_r.lt(theta)
    sin_sombra = alfa_r.gt(theta.subtract(noventa))
    plano = gamma0.divide(factor).updateMask(sin_inversion.And(sin_sombra))
    plano = plano.updateMask(plano.gt(0))
    k5 = ee.Kernel.square(2, 'pixels')
    media = plano.reduceNeighborhood(ee.Reducer.mean(), k5).reproject(proj)
    n = plano.select('VV').reduceNeighborhood(ee.Reducer.count(), k5).reproject(proj)
    db = media.log10().multiply(10).rename(['vv', 'vh'])
    return db.updateMask(n.gte(8)).copyProperties(img, ['system:time_start'])


def _pila_s1(region, anios=S1_ANIOS):
    """ee.Image con las 5 bandas de Sentinel-1 del detector (CLAVES_S1)."""
    import ee
    dem = (ee.ImageCollection(DEM_ASSET).select('DEM').mosaic()
           .setDefaultProjection('EPSG:4326', None, 30))
    rad = math.pi / 180
    pendiente = ee.Terrain.slope(dem).multiply(rad)
    orientacion = ee.Terrain.aspect(dem).multiply(rad)
    filtro = ee.Filter.Or(*[ee.Filter.date(f'{a}-06-01', f'{a}-10-01') for a in anios])
    col = (ee.ImageCollection('COPERNICUS/S1_GRD')
           .filterBounds(region).filter(filtro)
           .filter(ee.Filter.eq('instrumentMode', 'IW'))
           .filter(ee.Filter.eq('resolution_meters', 10))
           .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
           .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
           .map(lambda im: _escena_s1(im, pendiente, orientacion)))
    red = ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True)  # poblacional
    st = col.reduce(red)
    vv, vh = st.select('vv_mean'), st.select('vh_mean')
    return ee.Image.cat([vv.rename('s1_vv_mean'), vh.rename('s1_vh_mean'),
                         st.select('vv_stdDev').rename('s1_vv_sd'),
                         st.select('vh_stdDev').rename('s1_vh_sd'),
                         vv.subtract(vh).rename('s1_ratio_mean')])


def fetch_s1_gee(rows, bbox, anios=S1_ANIOS, chunk=200, progress=None):
    """Rellena en cada row las 5 variables de Sentinel-1 del detector (ver _pila_s1).
    Devuelve cuántos puntos fallaron."""
    import ee
    print('  → Sentinel-1 (radar) vía Google Earth Engine…')
    try:
        ee.data.setDeadline(300_000)
    except Exception:
        pass
    west, south, east, north = bbox[0], bbox[1], bbox[2], bbox[3]
    region = ee.Geometry.Rectangle([west, south, east, north])
    epsg = _utm_epsg((south + north) / 2, (west + east) / 2)
    return _leer_en_puntos(_pila_s1(region, anios), rows, list(CLAVES_S1), epsg, chunk,
                           'Sentinel-1', progress)


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
# Cambia cuando cambia lo que VALE una banda (no su nombre): lo leído con otra
# versión, sean muestras o bloques del mapa, deja de servir y hay que volver a leerlo.
#   1  estadísticas de jun–sep con todas las imágenes de todos los años juntas
#   2  estadísticas de jun–sep año a año y, después, la media entre años
VERSION_PILA = 2
ESTADISTICAS_PILA = ('jun–sep: cada estadística (media, mínimo, máximo, variabilidad) se '
                     'calcula en cada año y después se hace la media entre años')


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

    Sentinel-2 llega en su rejilla (10 m; las bandas de 20 m repiten valor). Sus
    estadísticas de jun–sep se calculan año a año y después se hace la media entre
    años (ver _period_new_stats). La topografía se calcula a 30 m y se lleva a 10 m
    por interpolación BILINEAL, que es lo indicado para una variable continua (repetir
    el píxel de 30 m dibujaría escalones)."""
    import ee
    s2, _ = _s2_stack(region, years, por_anio=True)
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


# Lo que necesita la pureza de cada punto, leído como la «Pureza dentro del píxel» del
# mapa (mapeo_rf): el NDVI de inicio de verano del PÍXEL de 10 m de la pila del mapeo y
# si ese píxel o alguno de sus 8 vecinos es agua (NDWI medio de jun–sep > 0; sin dato no
# cuenta como agua). Los índices del detector no sirven: son la media de una ventana de
# 50 m y salen más bajos (con ellos, frente a la ortofoto, acertaba el 56 %; así, el 82 %).
CLAVES_PUREZA = ('ndvi_early_pixel', 'agua_pixel_vecinos')


def fetch_pureza_gee(rows, years=(2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025),
                     progress=None):
    """Deja en cada punto r['ndvi_early_pixel'] y r['agua_pixel_vecinos'] (1 o 0), o
    None donde no hay dato. Devuelve cuántos puntos quedan con NDVI."""
    import ee
    validos = []
    for r in rows:
        try:
            lo, la = float(r['lon']), float(r['lat'])
        except (KeyError, TypeError, ValueError):
            continue
        if lo == lo and la == la:
            validos.append((r, (lo, la)))
    for r in rows:
        for k in CLAVES_PUREZA:
            r[k] = None
    if not validos:
        return 0
    puntos = [p for _, p in validos]
    years = tuple(int(y) for y in years)
    epsg = int(utm_de(puntos))
    lons, lats = [p[0] for p in puntos], [p[1] for p in puntos]
    margen = 0.01
    region = ee.Geometry.Rectangle([min(lons) - margen, min(lats) - margen,
                                    max(lons) + margen, max(lats) + margen])
    pila = pila_predictores(region, years, epsg)
    g = rejilla(epsg)
    agua = (pila.select('ndwi_mean').gt(0).unmask(0)
            .reproject(crs=g['crs'], crsTransform=g['crsTransform'])
            .focalMax(radius=1, kernelType='square', units='pixels')
            .rename('agua_vecinos'))
    img = ee.Image.cat([pila.select('ndvi_early'), agua])
    valores, _fallidos = muestrear_pila(img, puntos, epsg, bandas=['ndvi_early', 'agua_vecinos'],
                                        progress=progress)
    n = 0
    for (r, _), v in zip(validos, valores):
        v = v or {}
        r['ndvi_early_pixel'] = v.get('ndvi_early')
        r['agua_pixel_vecinos'] = (None if v.get('agua_vecinos') is None
                                   else int(round(v['agua_vecinos'])))
        n += r['ndvi_early_pixel'] is not None
    return n


# ------------------------------------------------------------
# Lectura por BLOQUES, para el mapa
# ------------------------------------------------------------
# Medido con Earth Engine real (34 variables, 2017-2025): un bloque de 256 × 256
# píxeles tarda alrededor de un minuto, uno de 512 × 512 agota el tiempo que da
# Earth Engine, y pedir varios a la vez no acelera (se frenan entre sí). Por eso se
# lee de uno en uno y, si alguno no llega, se parte en cuatro.
SIN_DATO_BLOQUE = -9999.0


def pila_para_mapa(limites_lonlat, years, epsg, bandas):
    """La pila, preparada para leerla por bloques sobre la rejilla del mapeo.

    Solo las bandas pedidas y en ese orden; en float32, que es la precisión con la
    que el Random Forest ve los datos tanto al entrenar como al predecir (no cambia
    ninguna predicción); y con un valor centinela donde no hay dato, porque la
    lectura por bloques no devuelve la máscara. `limites_lonlat` = (oeste, sur, este,
    norte) solo acota qué escenas se buscan: el valor de un píxel no depende de él.
    """
    import ee
    o, s, e, n = (float(v) for v in limites_lonlat)
    margen = 0.01                               # ~1 km, como en leer_predictores
    region = ee.Geometry.Rectangle([o - margen, s - margen, e + margen, n + margen])
    pila = pila_predictores(region, tuple(int(y) for y in years), int(epsg))
    return _para_bloques(pila.select(list(bandas)))


def _para_bloques(img):
    """La imagen en float32 y con el valor centinela donde no hay dato."""
    return img.toFloat().unmask(SIN_DATO_BLOQUE, False)


def leer_bloque(pila, epsg, x0, y0, ancho, alto, bandas, intentos=3, espera=4):
    """Píxeles de una ventana de la rejilla del mapeo: esquina superior izquierda
    (x0, y0) en UTM y `ancho` × `alto` píxeles de 10 m.

    `pila` es la de pila_para_mapa y `bandas`, las suyas en el mismo orden. Devuelve
    una matriz float32 (alto, ancho, nº de bandas) con NaN donde no hay dato. Si
    Earth Engine no devuelve el bloque, lo parte en cuatro y lo intenta por partes;
    si tampoco, lanza RuntimeError: no se devuelve nunca un bloque a medias.
    """
    import time

    import ee
    import numpy as np
    bandas = list(bandas)
    try:
        ee.data.setDeadline(300_000)
    except Exception:
        pass

    def pide(x, y, w, h):
        arr = ee.data.computePixels({
            'expression': pila, 'fileFormat': 'NUMPY_NDARRAY', 'bandIds': bandas,
            'grid': {'dimensions': {'width': int(w), 'height': int(h)},
                     'affineTransform': {'scaleX': ESCALA_M, 'shearX': 0, 'translateX': x,
                                         'shearY': 0, 'scaleY': -ESCALA_M, 'translateY': y},
                     'crsCode': f'EPSG:{int(epsg)}'}})
        return np.stack([arr[b].astype('float32') for b in bandas], axis=-1)

    def lee(x, y, w, h, partido=0):
        ultimo = None
        for i in range(intentos):
            try:
                return pide(x, y, w, h)
            except Exception as e:                  # noqa: BLE001
                ultimo = e
                # Si el bloque le cuesta demasiado, repetirlo igual no lo arregla.
                if any(t in str(e).lower() for t in ('timed out', 'too large', 'limit')):
                    break
                time.sleep(espera * (i + 1))
        if partido < 3 and min(w, h) >= 32:
            w1, h1 = w // 2, h // 2
            out = np.empty((h, w, len(bandas)), dtype='float32')
            for dx, dy, ww, hh in ((0, 0, w1, h1), (w1, 0, w - w1, h1),
                                   (0, h1, w1, h - h1), (w1, h1, w - w1, h - h1)):
                out[dy:dy + hh, dx:dx + ww] = lee(x + dx * ESCALA_M, y - dy * ESCALA_M,
                                                  ww, hh, partido + 1)
            return out
        raise RuntimeError(f'Earth Engine no ha devuelto un bloque del mapa ({ultimo}).')

    datos = lee(x0, y0, int(ancho), int(alto))
    datos[datos == SIN_DATO_BLOQUE] = np.nan
    return datos


def info_pila(years, epsg):
    """De dónde salen los valores de la pila: lo necesario para repetir la lectura."""
    return {
        'crs': f'EPSG:{int(epsg)}', 'resolucion_m': ESCALA_M,
        'rejilla': 'UTM, bordes de píxel en múltiplos de 10 m (la de Sentinel-2)',
        'anios': [int(y) for y in years],
        'bandas': list(BANDAS_PILA),
        'version': VERSION_PILA, 'estadisticas': ESTADISTICAS_PILA,
        'sentinel2': 'COPERNICUS/S2_SR_HARMONIZED',
        'elevaciones': DEM_ASSET,
        'remuestreo': 'Sentinel-2: vecino más próximo; topografía (30 m): bilineal',
    }


# ============================================================
# HUMEDAD TRAS EL DESHIELO (comparable dentro de un parque y entre parques)
# ------------------------------------------------------------
# El NDMI medio de junio a septiembre depende del calendario: el NDMI baja a lo
# largo del verano, así que un píxel que se queda sin nieve en julio y uno que la
# pierde en mayo se miden en momentos distintos de su verano (y entre parques, con
# deshielos de meses distintos, más aún). Aquí se mide en el mismo MOMENTO para cada
# píxel: unos días después de que se le vaya la nieve, cada año.
#
# Probado con las 948 muestras de Sierra Nevada (humedad_tras_deshielo/ en la carpeta
# del proyecto): el NDMI de 15 a 45 días tras el deshielo separa los borreguiles de
# un pasto igual de verde mejor que el de jun–sep (AUC 0,69 frente a 0,58; +0,11,
# IC 95 % +0,09 a +0,13). Descontar el verdor (restar el NDMI de un pasto igual de
# verde) se probó y empeora: en estos prados, seguir verde en verano es parte de
# estar húmedo, y al quitarlo se quita la humedad. Por eso no se descuenta.
# ============================================================
DIAS_TRAS_DESHIELO = (15, 45)       # ventana, en días después del deshielo
# Fin de la primavera: la nieve que se ve después (una nevada de verano) no retrasa el
# deshielo.
FIN_DESHIELO = '08-01'
# Sin nieve vista desde el 1 de marzo, el deshielo es esa fecha (día 60).
DIA_SIN_NIEVE = 60
BANDAS_HUMEDAD = ('deshielo', 'ndmi_tras', 'ndvi_tras', 'n_tras')


def _escenas_temporada(region, anio):
    """Escenas de marzo a octubre de un año, sin nubes ni sombras, con su día del
    año, si hay nieve, NDMI y NDVI."""
    import ee

    def prep(img):
        r = img.select(['B3', 'B4', 'B8', 'B11']).divide(10000)
        scl = img.select('SCL')
        # nieve: SCL 11, o NDSI > 0,4 con el infrarrojo cercano alto (el agua también
        # tiene NDSI alto, pero refleja poco infrarrojo)
        nieve = (scl.eq(11).Or(r.normalizedDifference(['B3', 'B11']).gt(0.4)
                               .And(r.select('B8').gt(0.11))))
        dia = ee.Image.constant(img.date().getRelative('day', 'year').add(1)).toFloat()
        buena = scl.remap(SCL_VALID, [1] * len(SCL_VALID), 0)
        return (ee.Image.cat([dia.rename('dia'), nieve.rename('nieve'),
                              r.normalizedDifference(['B8', 'B11']).rename('ndmi'),
                              r.normalizedDifference(['B8', 'B4']).rename('ndvi')])
                .toFloat().updateMask(buena)
                .copyProperties(img, ['system:time_start']))

    return (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(region)
            .filterDate(f'{anio}-03-01', f'{anio}-11-01')
            .map(_s2cloudless_mask)
            .map(prep))


def deshielo_anio(region, anio, col=None):
    """Día del año en que el píxel se queda sin nieve: la primera escena sin nieve
    después de la última con nieve de la primavera (hasta FIN_DESHIELO). Sin nieve
    vista desde marzo, la primera escena sin nieve."""
    col = col or _escenas_temporada(region, anio)
    ultima = (col.filterDate(f'{anio}-03-01', f'{anio}-{FIN_DESHIELO}')
              .map(lambda im: im.select('dia').updateMask(im.select('nieve')))
              .max().unmask(DIA_SIN_NIEVE))
    return (col.map(lambda im: im.select('dia').updateMask(
                im.select('nieve').Not().And(im.select('dia').gt(ultima))))
            .min().rename('deshielo'))


def humedad_anio(region, anio, dias=DIAS_TRAS_DESHIELO):
    """Un año: deshielo, NDMI y NDVI medianos de las escenas sin nieve entre
    `dias[0]` y `dias[1]` días después del deshielo, y cuántas escenas hay."""
    import ee
    col = _escenas_temporada(region, anio)
    deshielo = deshielo_anio(region, anio, col)

    def en_ventana(im):
        tras = im.select('dia').subtract(deshielo)
        dentro = tras.gte(dias[0]).And(tras.lte(dias[1])).And(im.select('nieve').Not())
        return im.select(['ndmi', 'ndvi']).updateMask(dentro)

    ventana = col.map(en_ventana)
    med = ventana.median().rename(['ndmi_tras', 'ndvi_tras'])
    n = ventana.select('ndmi').count().rename('n_tras').toFloat()
    return ee.Image.cat([deshielo.toFloat(), med, n])


def humedad_tras_deshielo(region, years, dias=DIAS_TRAS_DESHIELO):
    """Imagen con BANDAS_HUMEDAD: la mediana entre años del deshielo y del NDMI y
    NDVI tras el deshielo (`ndmi_tras` es la humedad), y `n_tras`, cuántas escenas
    hubo en la ventana (suma de los años)."""
    import ee
    anuales = [humedad_anio(region, int(y), dias) for y in years]
    # un año sin escenas en la ventana no cuenta en la mediana
    col = ee.ImageCollection([a.updateMask(a.select('n_tras').gt(0)) for a in anuales])
    med = col.select(['deshielo', 'ndmi_tras', 'ndvi_tras']).median()
    n = ee.ImageCollection([a.select('n_tras') for a in anuales]).sum()
    return ee.Image.cat([med, n]).select(list(BANDAS_HUMEDAD))


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
