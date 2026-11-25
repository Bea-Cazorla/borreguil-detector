"""
mapeo_raster.py — Del modelo final al mapa (paso «4 · Mapa» del mapeo por Random Forest).

Sin interfaz y sin Earth Engine: aquí están la zona y su rejilla de salida, el
troceado en bloques, la predicción de cada bloque, la escritura progresiva de los
GeoTIFF, la memoria de los bloques ya leídos, el dibujo de lo que se ve en el visor y
la consulta de un píxel. Quien lee los píxeles se pasa como función (en la app,
gee_backend.leer_bloque), de modo que todo se puede probar sin red
(tests/test_mapeo_mapa.py).

Rásteres que salen, en la MISMA rejilla de la que se leyeron las muestras (UTM, 10 m,
bordes de píxel en múltiplos de 10 m):

  clase.tif           código de la categoría elegida (uint8; 255 = sin dato)
  pmax.tif            probabilidad de esa categoría, de 0 a 1 (float32; NaN = sin dato)
  probabilidades.tif  una banda por categoría, con su nombre (float32; NaN = sin dato)
  probabilidad_borreguil.tif
                      probabilidad de ser borreguil, del tipo que sea: la suma de la
                      de los tipos (float32; solo si el modelo tiene «no borreguil»)

  verde_estimado.tif  cuánto verde hay dentro de cada píxel, de 0 a 1, estimado con
                      su NDVI (float32; NaN = sin dato)
  pureza.tif          en los píxeles de borreguil: 1 puro, 2 mixto-agua, 3 mixto-roca
                      (uint8; 255 = no es borreguil o no hay dato)

  ambiente.tif        solo en «ambiente y humedad»: el ambiente de cada píxel por
                      cercanía (1 arroyo, 2 laguna, 3 ladera; uint8; 255 = sin dato)
  distancias_ambiente.tif
                      las tres distancias, en metros, de las que sale ese ambiente: al
                      agua de una laguna, a una laguna de la capa y a un cauce
                      (float32; NaN = no hay nada de eso cerca)
  lagunas_y_cauces.tif  lo que se ha tomado por laguna y por cauce (1 agua de laguna,
                      2 cauce, 3 laguna de la capa; uint8; 0 = nada)

La categoría se elige con la regla del modelo (mapeo_rf.decidir): borreguil si esa
suma llega al umbral, y entonces su tipo más probable. En el nivel combinado (ambiente
y, dentro, humedad) el modelo solo dice la humedad: el ambiente lo pone el mapa por
cercanía a lagunas y cauces (mapeo_ambiente.py) y la categoría de `clase.tif` es la
compuesta («laguna · húmedo»). Las probabilidades siguen siendo las del modelo (de
«húmedo», de «seco», de «no borreguil»). La pureza dentro del píxel
(los dos últimos ficheros) es opcional y sale de la composición del píxel, no del
modelo: ver mapeo_rf.verde_estimado y mapeo_rf.pureza_de.

Un píxel queda SIN DATO, y no se le asigna ninguna categoría, si le falta alguna
variable, si cae fuera de la zona pedida o si cae fuera de la máscara. Nunca se
rellena ni se inventa nada.

Los GeoTIFF se leen y se escriben con mapeo_tif.py (numpy y tifffile, Python puro) y la
geometría se resuelve con shapely y pyproj. Aquí no se usa GDAL: en los equipos con el
«Control inteligente de aplicaciones» de Windows 11, Windows no siempre deja cargarlo.
"""
import contextlib
import json
import math
import time
from pathlib import Path

ESCALA_M = 10
LADO_BLOQUE = 256          # píxeles: bloques mayores no los devuelve Earth Engine a tiempo
SIN_DATO_CLASE = 255
FICHEROS = {'clase': 'clase.tif', 'pmax': 'pmax.tif', 'prob': 'probabilidades.tif',
            'pborr': 'probabilidad_borreguil.tif',
            'verde': 'verde_estimado.tif', 'pureza': 'pureza.tif',
            'ambiente': 'ambiente.tif', 'distancias': 'distancias_ambiente.tif',
            'referencias': 'lagunas_y_cauces.tif', 'agua': 'agua_de_laguna.tif',
            'codigos': 'codigos.csv', 'meta': 'mapa.json'}

# Un color por categoría, el mismo en el ráster, en su leyenda y en los puntos. En la
# humedad, tono frío para lo húmedo y cálido para lo seco (paleta de Okabe e Ito).
# El nivel «ambiente» lleva los tonos que pidió la usuaria: el arroyo, verde; la
# laguna, azul; la ladera, amarillo verdoso (25-11-2026, antes marrón). El amarillo
# verdoso es mucho más claro que el verde del arroyo: con daltonismo se distinguen por
# la claridad (OKLab × 100: 21,5 con visión normal, 16,4 con protanopía).
# ESTE es el catálogo de colores de toda la app: las gráficas (point_profile) los
# toman de aquí, para que cada categoría tenga siempre el mismo color (25-11-2026, la
# autora). «No borreguil», rojo oscuro, el mismo que en el mapa de los puntos
# (borreguil_pipeline.decision_color): antes aquí era gris; la autora vio los dos y
# eligió el rojo (25-11-2026).
# El nivel combinado lleva la paleta que eligió ella («Natural y contrastada»), tal
# como venía en su imagen: en cada ambiente, lo húmedo en un tono frío (verde azulado,
# azul, verde oliva) y lo seco en uno cálido (mostaza, naranja, teja). No se retoca sin
# que lo pida. Medido entre todas sus parejas (distancia en OKLab × 100; lo
# recomendable es 15 con visión normal y 8 con daltonismo simulado): lo que menos se
# separa es el mostaza del naranja (9,5), el verde azulado del oliva (11,0) y, con
# protanopía, el oliva del teja (2,5).
COLORES = {
    'no borreguil': '#8E0000', 'borreguil': '#33A02C',
    'arroyo': '#188C09', 'laguna': '#2D60A4', 'ladera': '#B0B83A',
    'húmedo': '#009E73', 'seco': '#D55E00',
    'puro': '#009E73', 'mixto-agua': '#0072B2', 'mixto-roca': '#8C6D31',
    'arroyo · húmedo': '#168C73', 'arroyo · seco': '#D9AA37',
    'laguna · húmedo': '#2875BA', 'laguna · seco': '#D98850',
    'ladera · húmedo': '#526A35', 'ladera · seco': '#B85C42',
}
_RESERVA = ('#444444', '#B2DF8A', '#FB9A99', '#CAB2D6', '#FFFF99', '#B15928')
# «no borreguil» suele ocupar casi todo el mapa: en el visor se pinta más transparente
# para que no tape el terreno y destaquen los borreguiles. En el GeoTIFF no cambia nada.
TENUES = {'no borreguil': 0.45}
# Probabilidad de 0 a 1: de oscuro a claro (viridis), que se lee igual en gris.
RAMPA = ('#440154', '#3B528B', '#21918C', '#5EC962', '#FDE725')


def color(categoria, i=0):
    """Color fijo de una categoría (y uno de reserva si no está en el catálogo)."""
    return COLORES.get(categoria, _RESERVA[i % len(_RESERVA)])


def _rgb(hexa):
    h = hexa.lstrip('#')
    return tuple(int(h[k:k + 2], 16) for k in (0, 2, 4))


# ============================================================
# Rejilla y bloques
# ============================================================
def limites_utm(limites_lonlat, epsg):
    """(minx, miny, maxx, maxy) en UTM de un rectángulo (oeste, sur, este, norte) en
    grados. Se recorre todo el borde y no solo las esquinas: en UTM los lados de un
    rectángulo geográfico no son rectos."""
    import pyproj
    o, s, e, n = (float(v) for v in limites_lonlat)
    tr = pyproj.Transformer.from_crs(4326, int(epsg), always_xy=True)
    xs, ys = [], []
    for k in range(21):
        t = k / 20
        for lon, lat in ((o + (e - o) * t, s), (o + (e - o) * t, n),
                         (o, s + (n - s) * t), (e, s + (n - s) * t)):
            x, y = tr.transform(lon, lat)
            xs.append(x)
            ys.append(y)
    return min(xs), min(ys), max(xs), max(ys)


def rejilla(limites, epsg, escala=ESCALA_M):
    """Rejilla de salida que cubre (minx, miny, maxx, maxy) en UTM, con los bordes de
    píxel en múltiplos de `escala`: la misma en la que se leyeron las muestras.
    (x0, y0) es la esquina superior izquierda."""
    minx, miny, maxx, maxy = (float(v) for v in limites)
    if not (maxx > minx and maxy > miny):
        raise ValueError('la zona a cartografiar no tiene superficie')
    x0 = math.floor(minx / escala) * escala
    y0 = math.ceil(maxy / escala) * escala
    return {'epsg': int(epsg), 'escala': escala, 'x0': x0, 'y0': y0,
            'ancho': max(1, math.ceil((maxx - x0) / escala)),
            'alto': max(1, math.ceil((y0 - miny) / escala))}


def bloques(rej, lado=LADO_BLOQUE):
    """Ventanas (columna, fila, ancho, alto) que cubren la rejilla sin solaparse."""
    return [(c, f, min(lado, rej['ancho'] - c), min(lado, rej['alto'] - f))
            for f in range(0, rej['alto'], lado) for c in range(0, rej['ancho'], lado)]


def minutos_estimados(rej, lado=LADO_BLOQUE, ventanas=None):
    """Cuánto tardará Earth Engine en servir la rejilla (o solo las `ventanas` que se
    le pasen), a ojo: unos 20 s fijos por bloque más 0,8 ms por píxel (medido el
    2026-10-05 con las 34 variables y los años 2017-2025: un bloque entero, algo más
    de un minuto). Solo orienta: el mismo bloque puede tardar la mitad o el doble."""
    if ventanas is None:
        ventanas = bloques(rej, lado)
    return sum(20 + 0.0008 * w * h for _, _, w, h in ventanas) / 60


def esquina(rej, col, fila):
    """Coordenadas UTM de la esquina superior izquierda del píxel (col, fila)."""
    return rej['x0'] + col * rej['escala'], rej['y0'] - fila * rej['escala']


def _marco_en(epsg_de, epsg_a, minx, miny, maxx, maxy):
    """El rectángulo que envuelve a otro al pasarlo a otro sistema de coordenadas. Se
    recorre todo el borde y no solo las esquinas: los lados dejan de ser rectos."""
    import pyproj
    tr = pyproj.Transformer.from_crs(int(epsg_de), int(epsg_a), always_xy=True)
    return tr.transform_bounds(minx, miny, maxx, maxy, densify_pts=21)


def limites_lonlat(rej):
    """(oeste, sur, este, norte) en grados del rectángulo que ocupa la rejilla."""
    x1, y1 = esquina(rej, rej['ancho'], rej['alto'])
    return _marco_en(rej['epsg'], 4326, rej['x0'], y1, x1, rej['y0'])


def a_utm(geom, epsg):
    """Una geometría (shapely) en grados, llevada a la UTM de la rejilla."""
    import pyproj
    from shapely.ops import transform
    return transform(pyproj.Transformer.from_crs(4326, int(epsg), always_xy=True).transform,
                     geom)


def dentro(geom_utm, rej, ventana):
    """Matriz booleana (alto, ancho) de la ventana: True en los píxeles cuyo CENTRO
    cae dentro de la geometría."""
    import numpy as np
    import shapely
    col, fila, ancho, alto = ventana
    e = rej['escala']
    x, y = esquina(rej, col, fila)
    shapely.prepare(geom_utm)
    # Lo corriente es que la ventana caiga entera fuera o entera dentro: eso se sabe
    # sin mirar píxel a píxel.
    marco = shapely.box(x, y - alto * e, x + ancho * e, y)
    if not shapely.intersects(geom_utm, marco):
        return np.zeros((alto, ancho), dtype=bool)
    if shapely.contains(geom_utm, marco):
        return np.ones((alto, ancho), dtype=bool)
    xs, ys = np.meshgrid(x + (np.arange(ancho) + 0.5) * e, y - (np.arange(alto) + 0.5) * e)
    return shapely.contains_xy(geom_utm, xs, ys)


def _permitido(rej, ventana, zona_utm, mascara_utm):
    """Píxeles de la ventana que se pueden clasificar: dentro de la zona y dentro de
    la máscara. None si no hay ni zona ni máscara (todos)."""
    permitido = None
    for geom in (zona_utm, mascara_utm):
        if geom is not None:
            m = dentro(geom, rej, ventana)
            permitido = m if permitido is None else (permitido & m)
    return permitido


def bloques_utiles(rej, zona_utm=None, mascara_utm=None, lado=LADO_BLOQUE):
    """Los bloques que tienen algún píxel que clasificar. Los demás ni se leen: en una
    zona irregular, o con máscara, es buena parte del tiempo de Earth Engine."""
    ventanas = bloques(rej, lado)
    if zona_utm is None and mascara_utm is None:
        return ventanas
    return [v for v in ventanas if _permitido(rej, v, zona_utm, mascara_utm).any()]


# ============================================================
# La zona que se pide
# ============================================================
MAX_BLOQUES_REJILLA = 20_000     # ≈ 130 000 km²: por encima ni se calcula


def rect_centrado(lon, lat, ancho_km, alto_km, epsg):
    """(minx, miny, maxx, maxy) en UTM de un rectángulo de ancho × alto km centrado
    en un punto dado en grados."""
    import pyproj
    x, y = pyproj.Transformer.from_crs(4326, int(epsg), always_xy=True).transform(
        float(lon), float(lat))
    a, b = float(ancho_km) * 500, float(alto_km) * 500
    return x - a, y - b, x + a, y + b


def limites_de_puntos(puntos, epsg, margen_m=100):
    """(minx, miny, maxx, maxy) en UTM del rectángulo que ocupan unos puntos
    [(lon, lat), …], con un margen alrededor."""
    import pyproj
    tr = pyproj.Transformer.from_crs(4326, int(epsg), always_xy=True)
    xy = [tr.transform(float(lo), float(la)) for lo, la in puntos]
    xs, ys = [p[0] for p in xy], [p[1] for p in xy]
    return min(xs) - margen_m, min(ys) - margen_m, max(xs) + margen_m, max(ys) + margen_m


def _valida(geom):
    """Un polígono dibujado a mano puede cruzarse consigo mismo; se arregla antes de
    operar con él para que no falle la intersección."""
    if geom is None or geom.is_valid:
        return geom
    from shapely import make_valid
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    g = make_valid(geom)
    if g.geom_type in ('Polygon', 'MultiPolygon'):
        return g
    # al arreglarlo pueden quedar líneas o puntos sueltos: solo cuentan las superficies
    partes = [p for p in getattr(g, 'geoms', [])
              if p.geom_type in ('Polygon', 'MultiPolygon')]
    return unary_union(partes) if partes else Polygon()


def preparar_zona(epsg, limites=None, zona=None, mascara=None, lado=LADO_BLOQUE):
    """Rejilla y bloques de una petición de mapa.

    limites: rectángulo (minx, miny, maxx, maxy) en UTM, si la zona es un rectángulo.
    zona: polígono (shapely, en grados), si la zona es un polígono.
    mascara: polígonos (shapely, en grados) fuera de los cuales no se clasifica.
    Si solo se da la máscara, la zona es todo lo que cubre.

    La rejilla se ciñe a lo que de verdad hay que clasificar (zona ∩ máscara).
    Devuelve un dict con `error` (None, 'sin_superficie', 'mascara_fuera' o
    'demasiado_grande') y, si no hay error: rej, zona_utm, mascara_utm, utiles (los
    bloques con algo que clasificar), bloques (los de toda la rejilla) y km2 (la
    superficie que se va a clasificar).
    """
    from shapely.geometry import box
    zona_utm = _valida(a_utm(zona, epsg)) if zona is not None else None
    mascara_utm = _valida(a_utm(mascara, epsg)) if mascara is not None else None
    if zona_utm is not None:
        marco = zona_utm
    elif limites is not None:
        minx, miny, maxx, maxy = (float(v) for v in limites)
        if not (maxx > minx and maxy > miny):
            return {'error': 'sin_superficie'}
        marco = box(minx, miny, maxx, maxy)
    elif mascara_utm is not None:
        marco = mascara_utm
    else:
        raise ValueError('hay que dar un rectángulo, una zona o una máscara')
    util = marco if (mascara_utm is None or marco is mascara_utm) \
        else marco.intersection(mascara_utm)
    if util.is_empty or not util.area > 0:
        return {'error': 'mascara_fuera' if mascara_utm is not None and marco is not mascara_utm
                else 'sin_superficie'}
    rej = rejilla(util.bounds, epsg)
    total = math.ceil(rej['ancho'] / lado) * math.ceil(rej['alto'] / lado)
    if total > MAX_BLOQUES_REJILLA:
        return {'error': 'demasiado_grande', 'bloques': total, 'km2': util.area / 1e6}
    return {'error': None, 'rej': rej, 'zona_utm': zona_utm, 'mascara_utm': mascara_utm,
            'utiles': bloques_utiles(rej, zona_utm, mascara_utm, lado), 'bloques': total,
            'km2': util.area / 1e6}


def _rect_lonlat(epsg, minx, miny, maxx, maxy):
    import pyproj
    tr = pyproj.Transformer.from_crs(int(epsg), 4326, always_xy=True)
    out = []
    for x, y in ((minx, maxy), (maxx, maxy), (maxx, miny), (minx, miny)):
        lon, lat = tr.transform(x, y)
        out.append([lat, lon])
    return out


def contorno_lonlat(rej):
    """Vértices [[lat, lon], …] del rectángulo de la rejilla, para dibujarlo."""
    x1, y1 = esquina(rej, rej['ancho'], rej['alto'])
    return _rect_lonlat(rej['epsg'], rej['x0'], y1, x1, rej['y0'])


def anillos_para_dibujar(geom, tolerancia=0.0001, maximo=400):
    """Contornos [[lat, lon], …] de los polígonos de una geometría en grados, para
    dibujarla en el visor: simplificada (unos 10 m) y, si tiene muchísimos polígonos,
    solo los mayores. Es solo para verla: los cálculos usan la geometría entera."""
    g = geom.simplify(tolerancia)
    polis = [g] if g.geom_type == 'Polygon' else [
        p for p in getattr(g, 'geoms', []) if p.geom_type == 'Polygon']
    polis = sorted((p for p in polis if not p.is_empty), key=lambda p: -p.area)[:maximo]
    return [[[y, x] for x, y in p.exterior.coords] for p in polis]


def contorno_pixel(rej, col, fila):
    """Vértices [[lat, lon], …] del píxel (col, fila) de la rejilla."""
    x, y = esquina(rej, col, fila)
    return _rect_lonlat(rej['epsg'], x, y - rej['escala'], x + rej['escala'], y)


def encuadre(limites_lonlat, ancho_px=600, alto_px=460):
    """Centro [lat, lon] y nivel de zoom con los que un rectángulo (oeste, sur, este,
    norte) cabe en un visor de ese tamaño. Se calcula aquí, y no se deja al visor,
    porque un visor que se crea en una pestaña oculta todavía no sabe cuánto mide."""
    o, s, e, n = (float(v) for v in limites_lonlat)

    def mercator(lat):
        return math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    dx = max((e - o) / 360.0, 1e-9)
    dy = max((mercator(n) - mercator(s)) / (2 * math.pi), 1e-9)
    zoom = math.floor(min(math.log2(ancho_px / 256 / dx), math.log2(alto_px / 256 / dy)))
    return [(s + n) / 2, (o + e) / 2], max(1, min(18, zoom))


# ============================================================
# Bloques ya leídos
# ------------------------------------------------------------
# Leer un bloque cuesta alrededor de un minuto y predecirlo, menos de un segundo. Lo
# leído se guarda en disco mientras dura la sesión: si el mapa se interrumpe sigue
# donde iba, y el mapa de otro nivel sobre la misma zona no vuelve a pedir nada.
# ============================================================
def clave_lectura(epsg, anios, bandas, version=1):
    """Identifica de dónde sale un bloque: la rejilla, los años, las variables, en su
    orden, y la manera de calcularlas (`version`: gee_backend.VERSION_PILA). Con otra
    combinación, los bloques guardados no valen."""
    import hashlib
    txt = (f"{int(epsg)}|{','.join(str(int(a)) for a in anios)}|{','.join(bandas)}"
           f"|v{int(version)}")
    return hashlib.md5(txt.encode()).hexdigest()[:12]


def _ruta_bloque(carpeta, rej, ventana):
    col, fila, ancho, alto = ventana
    x, y = esquina(rej, col, fila)
    return Path(carpeta) / f'{int(x)}_{int(y)}_{int(ancho)}x{int(alto)}.npy'


def lector_con_memoria(leer, carpeta, rej):
    """Envuelve a un lector de bloques: lo que lee lo guarda en `carpeta`, y lo que ya
    está guardado no lo vuelve a pedir. La carpeta tiene que ser solo de una lectura
    (ver clave_lectura). Un bloque se guarda entero o no se guarda."""
    import os

    import numpy as np
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)

    def lee(col, fila, ancho, alto):
        ruta = _ruta_bloque(carpeta, rej, (col, fila, ancho, alto))
        if ruta.exists():
            try:
                datos = np.load(ruta)
                if datos.shape[:2] == (alto, ancho):
                    return datos
            except Exception:                       # noqa: BLE001
                pass                                # estropeado: se vuelve a pedir
        datos = np.asarray(leer(col, fila, ancho, alto), dtype='float32')
        # No poder guardarlo (disco lleno, fichero ocupado un instante) no detiene el
        # mapa: solo habría que volver a pedir ese bloque otra vez.
        parte = ruta.with_name(ruta.name + '.parte')
        try:
            with open(parte, 'wb') as f:
                np.save(f, datos)
            os.replace(parte, ruta)
        except OSError:
            _borrar([parte], intentos=1)
        return datos
    return lee


def bloques_por_leer(carpeta, rej, ventanas):
    """De esas ventanas, las que todavía no están guardadas en `carpeta`."""
    return [v for v in ventanas if not _ruta_bloque(carpeta, rej, v).exists()]


def bloques_guardados(carpeta, rej, ventanas):
    """Cuántas de esas ventanas están ya leídas y guardadas en `carpeta`."""
    return len(ventanas) - len(bloques_por_leer(carpeta, rej, ventanas))


# ============================================================
# Predicción de un bloque
# ============================================================
def proba_repetible(modelo, X):
    """predict_proba sumando los árboles siempre en el mismo orden.

    Con varios hilos, scikit-learn los suma según van acabando: el último decimal
    cambia de una ejecución a otra y un empate entre dos categorías puede caer a un
    lado o a otro. Con un solo hilo, el mismo modelo y los mismos datos dan siempre
    exactamente el mismo mapa, se trocee como se trocee (un bloque tarda menos de un
    segundo; leerlo de Earth Engine, un minuto)."""
    hilos = getattr(modelo, 'n_jobs', None)
    if hilos in (None, 1):
        return modelo.predict_proba(X)
    modelo.n_jobs = 1
    try:
        return modelo.predict_proba(X)
    finally:
        modelo.n_jobs = hilos


def predecir_bloque(modelo, datos, permitido=None, umbral=None):
    """Clasifica los píxeles de un bloque.

    datos: (alto, ancho, nº de variables), en el orden con que se entrenó el modelo y
    con NaN donde no hay dato. permitido: matriz booleana (alto, ancho) o None.
    umbral: el de la regla de «¿borreguil o no?» (mapeo_rf.decidir); None = sin regla,
    la categoría más probable de todas.

    Devuelve (clase, pmax, prob, pborr): el código de la categoría elegida (uint8;
    255 = sin dato), su probabilidad, la de cada categoría (en el orden de
    modelo.classes_) y la de ser borreguil de algún tipo (la suma de la de los tipos).
    Un píxel al que le falte alguna variable, o no permitido, queda sin dato en las
    cuatro.

    La categoría se decide con las probabilidades ya en float32, que es como se
    guardan: quien lea los ficheros y aplique el mismo umbral obtiene el mismo mapa.
    """
    import numpy as np

    import mapeo_rf as mr
    alto, ancho, p = datos.shape
    if p != modelo.n_features_in_:
        raise ValueError(f'el bloque trae {p} variables y el modelo se entrenó con '
                         f'{modelo.n_features_in_}: no se puede predecir')
    X = datos.reshape(-1, p)
    valido = np.isfinite(X).all(axis=1)
    if permitido is not None:
        valido &= np.asarray(permitido, dtype=bool).reshape(-1)
    k = len(modelo.classes_)
    clase = np.full(alto * ancho, SIN_DATO_CLASE, dtype='uint8')
    pmax = np.full(alto * ancho, np.nan, dtype='float32')
    prob = np.full((k, alto * ancho), np.nan, dtype='float32')
    pborr = np.full(alto * ancho, np.nan, dtype='float32')
    if valido.any():
        pr = proba_repetible(modelo, X[valido]).astype('float32')
        categoria, p_categoria, p_borr = mr.decidir(pr, modelo.classes_, umbral)
        clase[valido] = categoria.astype('uint8')
        pmax[valido] = p_categoria
        prob[:, valido] = pr.T
        pborr[valido] = p_borr
    return (clase.reshape(alto, ancho), pmax.reshape(alto, ancho),
            prob.reshape(k, alto, ancho), pborr.reshape(alto, ancho))


# ============================================================
# El mapa entero, bloque a bloque
# ============================================================
def _con_vecinos(m):
    """Los píxeles de una máscara y los que tocan a alguno de ellos (los ocho de
    alrededor)."""
    r = m.copy()
    r[1:, :] |= m[:-1, :]
    r[:-1, :] |= m[1:, :]
    r[:, 1:] |= m[:, :-1]
    r[:, :-1] |= m[:, 1:]
    r[1:, 1:] |= m[:-1, :-1]
    r[1:, :-1] |= m[:-1, 1:]
    r[:-1, 1:] |= m[1:, :-1]
    r[:-1, :-1] |= m[1:, 1:]
    return r


def hacer_mapa(mod, rej, leer_bloque, carpeta, zona_utm=None, mascara_utm=None,
               lado=LADO_BLOQUE, progreso=None, extra=None, bandas=None, pureza=None,
               ambiente=None, quitar_agua=False):
    """Predice toda la rejilla por bloques y va escribiendo los GeoTIFF.

    mod: lo que devuelve mapeo_rf.entrenar_y_validar (modelo final, categorías con su
         código, predictores en su orden y la regla de «¿borreguil o no?» con su
         umbral; sin regla, cada píxel recibe la categoría más probable de todas).
    leer_bloque(col, fila, ancho, alto) → matriz (alto, ancho, nº de bandas) con las
         variables EN EL ORDEN de `bandas` y NaN donde no hay dato.
    bandas: los nombres de lo que trae cada bloque. Tienen que empezar por
         mod['predictores'], en su orden; detrás pueden ir las que solo hacen falta
         para la pureza. Por defecto, mod['predictores'].
    pureza: None, o los valores con los que calcular la pureza dentro del píxel (ver
         mapeo_rf.params_pureza). Con ellos se guardan además el verde estimado de
         cada píxel y, en los de borreguil, si es puro, mixto-agua o mixto-roca.
    ambiente: None, o cómo poner el ambiente por cercanía (mapeo_ambiente) en un
         nivel combinado, con el modelo de su nivel de dentro (la humedad), o en el
         mapa de borreguil sí o no, con su modelo (nivel mapeo_rf.NIVEL_SI_NO):
           {'nivel': 'ambiente_humedad' (o 'borreguil'),
            'params': las distancias (mapeo_ambiente.params),
            'lagunas': [(x, y), …] puntos de la capa de lagunas, en la UTM de la rejilla,
            'cauces': geometría de una capa de cauces en esa UTM, o None,
            'relieve': función (margen_px) → relieve de la rejilla ampliada con ese
                       margen por cada lado, o None,
            'fuentes': {'lagunas': …, 'cauces': …} de dónde sale cada cosa (texto)}
         Sin capa de cauces, salen del relieve: el de `relieve` (con margen, para
         contar el agua que entra de fuera) o, si tampoco se da, el de los bloques.
         La categoría de cada píxel de borreguil se compone entonces con su ambiente
         y con lo que dice el modelo («laguna · húmedo»); en el de sí o no, es su
         ambiente sin más (las categorías y los códigos del nivel «ambiente»: un
         píxel de borreguil junto a una laguna queda «laguna»). Se guardan además el
         ambiente, las distancias de las que sale y lo tomado por laguna y por cauce.
    quitar_agua: el agua abierta de una laguna no es borreguil. Con True, los píxeles
         de agua de laguna que se ven en Sentinel-2 (los mismos que usa el ambiente:
         mapeo_ambiente.agua_de_laguna) pasan a «no borreguil», y se guarda esa agua en
         su propio fichero. Solo si el modelo tiene «no borreguil»; hacen falta además
         las variables del agua y la pendiente. Las probabilidades se guardan como las
         da el modelo; la de la categoría elegida, la de «no borreguil».
    zona_utm: geometría de la zona pedida (None = todo el rectángulo de la rejilla).
    mascara_utm: geometría fuera de la cual no se clasifica (None = sin máscara).
    progreso(hechos, total): se llama al acabar cada bloque de los que se leen.
    extra: datos de la ejecución que se guardan tal cual en mapa.json (años, etc.).

    De lo leído solo hay en memoria un bloque cada vez. Lo que se va calculando se
    guarda aparte (en memoria o, si el mapa es grande, en ficheros temporales) y los
    GeoTIFF se escriben al final. Los bloques sin ningún píxel dentro de la zona y de la
    máscara no se leen. Si un bloque no se puede leer, se detiene y no deja nada
    escrito: un mapa con huecos sin explicar sería peor que no tenerlo.
    Devuelve el dict que también queda en mapa.json.
    """
    import numpy as np

    import mapeo_rf as mr
    import mapeo_tif as mt
    modelo = mod['modelo']
    codigos = [int(c) for c in modelo.classes_]
    nombre = {int(c['codigo']): c['categoria'] for c in mod['categorias']}
    if sorted(nombre) != codigos:
        raise ValueError('las categorías del modelo no coinciden con sus códigos')
    if modelo.n_features_in_ != len(mod['predictores']):
        raise ValueError('el modelo no se entrenó con esa lista de variables')
    regla = mod.get('regla') or {}
    umbral = regla.get('umbral')
    # La probabilidad de ser borreguil solo dice algo si el modelo sabe decir que no.
    con_pborr = 0 in codigos
    pred = list(mod['predictores'])
    bandas = list(bandas or pred)
    if bandas[:len(pred)] != pred:
        raise ValueError('las bandas de los bloques tienen que empezar por las variables '
                         'del modelo, en su orden')
    par_pur = None
    if pureza is not None:
        par_pur = mr.params_pureza(pureza)
        faltan = [b for b in (mr.VARIABLE_VERDE, mr.VARIABLE_AGUA) if b not in bandas]
        if faltan:
            raise ValueError('para la pureza dentro del píxel hacen falta también las '
                             'variables: ' + ', '.join(faltan))
        i_verde, i_agua = bandas.index(mr.VARIABLE_VERDE), bandas.index(mr.VARIABLE_AGUA)
        cod_pur = {n: c for n, c in mr.codigos('pureza').items() if c != 0}
        # De cada píxel de borreguil, si su verde llega al corte (1) o no (2); y dónde
        # hay agua. La pureza se decide al final, con el mapa entero: un píxel puede
        # tocar agua que está en otro bloque.
        llega = np.zeros((rej['alto'], rej['ancho']), dtype='uint8')
        agua = np.zeros((rej['alto'], rej['ancho']), dtype=bool)
    # Las categorías del MAPA: las del modelo o, con el ambiente por cercanía, las
    # compuestas («laguna · húmedo»): cada ambiente con cada tipo que conoce el modelo.
    cat_mapa = [(c, nombre[c]) for c in codigos]
    par_amb = None
    if ambiente is not None:
        import mapeo_ambiente as ma
        nivel_c = ambiente.get('nivel')
        # En el de sí o no, el ambiente es la categoría de cada píxel de borreguil.
        solo_ambiente = nivel_c == mr.NIVEL_SI_NO == mod.get('nivel') and 0 in codigos
        if not solo_ambiente and (mr.nivel_por_cercania(nivel_c) is None
                                  or mr.nivel_del_modelo(nivel_c) != mod.get('nivel')):
            raise ValueError('el ambiente por cercanía es para un nivel combinado, con el '
                             'modelo de su nivel de dentro, o para el de borreguil sí o no')
        par_amb = ma.params(ambiente.get('params'))
        capa_cauces, leer_relieve = ambiente.get('cauces'), ambiente.get('relieve')
        del_bloque = capa_cauces is None and leer_relieve is None
        hacen_falta = [ma.VARIABLE_AGUA, ma.VARIABLE_PENDIENTE] + (
            [ma.VARIABLE_RELIEVE] if del_bloque else [])
        faltan = [b for b in hacen_falta if b not in bandas]
        if faltan:
            raise ValueError('para el ambiente por cercanía hacen falta también las '
                             'variables: ' + ', '.join(faltan))
        i_ndwi, i_pend = bandas.index(ma.VARIABLE_AGUA), bandas.index(ma.VARIABLE_PENDIENTE)
        if solo_ambiente:
            de_amb = mr.codigos('ambiente')
            cat_mapa = [(0, nombre[0])] + [(de_amb[r], r) for r in mr.NIVELES['ambiente']]
        else:
            de_compuesta = mr.codigos(nivel_c)
            cat_mapa = ([(0, nombre[0])] if 0 in codigos else []) + [
                (de_compuesta[f'{r}{mr.SEP}{nombre[c]}'], f'{r}{mr.SEP}{nombre[c]}')
                for r in mr.NIVELES[mr.nivel_por_cercania(nivel_c)] for c in codigos if c != 0]
        # Lo que dice el modelo en cada píxel, dónde puede haber agua de laguna y, si
        # hace falta, el relieve: el ambiente se decide al final, con el mapa entero
        # (una laguna o un cauce pueden estar en otro bloque).
        forma = (rej['alto'], rej['ancho'])
        tipo_mod = np.full(forma, SIN_DATO_CLASE, dtype='uint8')
        candidata = np.zeros(forma, dtype=bool)
        relieve = np.full(forma, np.nan, dtype='float32') if del_bloque else None
    # El agua de laguna, que no es borreguil: se decide al final, con el mapa entero (una
    # mancha de agua puede repartirse entre bloques).
    quitar = bool(quitar_agua) and con_pborr
    if quitar:
        import mapeo_ambiente as ma
        faltan = [b for b in (ma.VARIABLE_AGUA, ma.VARIABLE_PENDIENTE) if b not in bandas]
        if faltan:
            raise ValueError('para quitar el agua de laguna hacen falta también las '
                             'variables: ' + ', '.join(faltan))
        i_ndwi, i_pend = bandas.index(ma.VARIABLE_AGUA), bandas.index(ma.VARIABLE_PENDIENTE)
        if par_amb is None:
            candidata = np.zeros((rej['alto'], rej['ancho']), dtype=bool)
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    rutas = {k: carpeta / v for k, v in FICHEROS.items()}
    ventanas = bloques(rej, lado)
    utiles = set(bloques_utiles(rej, zona_utm, mascara_utm, lado))
    if not utiles:
        raise ValueError('no hay ningún píxel que clasificar dentro de la zona y la máscara')
    cuenta = {c: 0 for c in codigos}
    suma_pmax, validos, leidos, en_zona, t0 = 0.0, 0, 0, 0, time.time()
    abiertos = []

    def nuevo(clave, **como):
        """Un GeoTIFF de la rejilla, por llenar: lo que no se escriba queda sin dato."""
        f = mt.Escritor(rutas[clave], rej, **como)
        abiertos.append(f)
        return f

    try:
        f_clase = nuevo('clase', tipo='uint8', sin_dato=SIN_DATO_CLASE)
        f_pmax = nuevo('pmax', sin_dato=float('nan'))
        f_prob = nuevo('prob', bandas=len(codigos), sin_dato=float('nan'))
        f_pborr = None
        if con_pborr:
            f_pborr = nuevo('pborr', sin_dato=float('nan'))
            f_pborr.nombre_banda(1, 'probabilidad de borreguil (suma de los tipos)')
        f_verde = None
        if par_pur:
            f_verde = nuevo('verde', sin_dato=float('nan'))
            f_verde.nombre_banda(1, 'verde dentro del pixel, de 0 a 1 (estimado '
                                    f'con {mr.VARIABLE_VERDE})')
        # Los colores y los nombres viajan dentro de los ficheros.
        f_clase.paleta({c: _rgb(color(n, i)) for i, (c, n) in enumerate(cat_mapa)})
        f_clase.etiquetas(CATEGORIAS='; '.join(f'{c} = {n}' for c, n in cat_mapa),
                          SIN_DATO=str(SIN_DATO_CLASE),
                          **({} if umbral is None else {'UMBRAL_BORREGUIL': str(umbral)}),
                          **({} if par_amb is None else {
                              'AMBIENTE_POR_CERCANIA': ma.frase(par_amb)}))
        f_clase.nombre_banda(1, 'categoria (codigo)')
        f_pmax.nombre_banda(1, 'probabilidad de la categoria elegida')
        for i, c in enumerate(codigos, 1):
            f_prob.nombre_banda(i, f'P({nombre[c]})')

        for col, fila, ancho, alto in ventanas:
            win = (col, fila, ancho, alto)
            if win not in utiles:
                # nada que clasificar aquí: queda sin dato y no se pide a Earth Engine
                continue
            datos = np.asarray(leer_bloque(col, fila, ancho, alto), dtype='float32')
            if datos.shape != (alto, ancho, len(bandas)):
                raise ValueError(f'el bloque ({col}, {fila}) ha llegado con forma '
                                 f'{datos.shape}; se esperaba '
                                 f'{(alto, ancho, len(bandas))}')
            permitido = _permitido(rej, (col, fila, ancho, alto), zona_utm, mascara_utm)
            clase, pmax, prob, pborr = predecir_bloque(modelo, datos[..., :len(pred)],
                                                       permitido, umbral)
            leidos += 1
            en_zona += ancho * alto if permitido is None else int(permitido.sum())
            trozo = (slice(fila, fila + alto), slice(col, col + ancho))
            if par_amb is None:
                f_clase.escribir(clase, 1, ventana=win)
            else:
                # la categoría se escribe al final, compuesta con el ambiente
                tipo_mod[trozo] = clase
                if relieve is not None:
                    relieve[trozo] = datos[..., bandas.index(ma.VARIABLE_RELIEVE)]
            if par_amb is not None or quitar:
                candidata[trozo] = ma.candidata_a_agua(datos[..., i_ndwi], datos[..., i_pend])
            f_pmax.escribir(pmax, 1, ventana=win)
            f_prob.escribir(prob, ventana=win)
            if f_pborr:
                f_pborr.escribir(pborr, 1, ventana=win)
            ok = clase != SIN_DATO_CLASE
            if par_pur:
                # el verde, de todo lo clasificado (también de lo que no es borreguil)
                verde = np.where(ok, mr.verde_estimado(
                    datos[..., i_verde], par_pur['sin_verde'], par_pur['todo_verde']),
                    np.nan).astype('float32')
                f_verde.escribir(verde, 1, ventana=win)
                borreguil = ok & (clase != 0) & np.isfinite(verde)
                llega[trozo] = np.where(
                    borreguil, np.where(verde >= np.float32(par_pur['puro_desde']), 1, 2), 0)
                with np.errstate(invalid='ignore'):
                    agua[trozo] = datos[..., i_agua] > mr.NDWI_AGUA
            validos += int(ok.sum())
            if ok.any():
                suma_pmax += float(pmax[ok].sum())
                for c, cuantos in zip(*np.unique(clase[ok], return_counts=True)):
                    cuenta[int(c)] += int(cuantos)
            if progreso:
                progreso(leidos, len(utiles))
        info_agua = None
        if quitar:
            # El agua de laguna pasa a «no borreguil»: en la categoría (la del modelo,
            # antes de componerla con el ambiente), en lo que llega a la pureza y en la
            # probabilidad de la categoría elegida, que pasa a ser la de «no borreguil».
            agua_l = ma.agua_de_laguna(candidata)
            cl = f_clase.matriz() if par_amb is None else tipo_mod
            con_dato_a = cl != SIN_DATO_CLASE
            quita = agua_l & con_dato_a & (cl != 0)
            if quita.any():
                pm, p_no = f_pmax.matriz(), f_prob.matriz(codigos.index(0) + 1)
                suma_pmax += float(p_no[quita].sum(dtype='float64')
                                   - pm[quita].sum(dtype='float64'))
                pm[quita] = p_no[quita]
                cl[quita] = 0
                if par_pur:
                    llega[quita] = 0
                if par_amb is None:
                    cuenta = {c: int((cl[con_dato_a] == c).sum()) for c in codigos}
            regla_agua = (f'agua de laguna (NDWI medio > {ma.NDWI_LAGUNA}, pendiente < '
                          f'{ma.PENDIENTE_LAGUNA:g} grados, manchas de {ma.MIN_AGUA_PX} '
                          'pixeles o mas) = no borreguil')
            f_clase.etiquetas(MASCARA_AGUA=regla_agua)
            f_agua = nuevo('agua', tipo='uint8', sin_dato=SIN_DATO_CLASE)
            f_agua.paleta({0: _rgb('#FFFFFF'), 1: _rgb(ma.COLORES_REFERENCIAS[1])})
            f_agua.etiquetas(CATEGORIAS='0 = no es agua; 1 = agua de laguna',
                             SIN_DATO=str(SIN_DATO_CLASE), REGLA=regla_agua)
            f_agua.nombre_banda(1, 'agua de laguna (Sentinel-2)')
            f_agua.escribir(np.where(con_dato_a, agua_l, SIN_DATO_CLASE).astype('uint8'), 1)
            info_agua = {'regla': regla_agua, 'pixeles_agua': int((agua_l & con_dato_a).sum()),
                         'pixeles_quitados': int(quita.sum())}
        info_amb = None
        if par_amb is not None:
            # El ambiente, con el mapa entero delante (ver mapeo_ambiente): laguna cerca
            # del agua de una laguna o de una laguna de la capa; si no, arroyo cerca de
            # un cauce; si no, ladera.
            e = rej['escala']
            agua = ma.agua_de_laguna(candidata)
            lagunas = [(float(x), float(y)) for x, y in (ambiente.get('lagunas') or [])]
            d_agua, d_capa = ma.distancia(agua, e), ma.distancia_a_puntos(lagunas, rej)
            # Los cauces, con margen alrededor: los de fuera también quedan cerca, y el
            # agua que entra de fuera también hace cauce.
            margen = int(math.ceil(ma.MARGEN_RELIEVE_M / e))
            if capa_cauces is not None:
                cauces_g, de_cauces = ma.rasterizar(capa_cauces, ma.ampliar(rej, margen)), 'capa'
            else:
                if leer_relieve is not None:
                    dem = np.asarray(leer_relieve(margen), dtype='float64')
                    if dem.shape != (rej['alto'] + 2 * margen, rej['ancho'] + 2 * margen):
                        raise ValueError(f'el relieve ha llegado con forma {dem.shape}; se '
                                         'esperaba la rejilla con su margen')
                else:
                    margen, dem = 0, relieve
                cauces_g, de_cauces = ma.cauces_del_relieve(dem, par_amb['cauce_ha'], e), 'relieve'
            dentro_g = (slice(margen, margen + rej['alto']), slice(margen, margen + rej['ancho']))
            d_cauce, cauces = ma.distancia(cauces_g, e)[dentro_g], cauces_g[dentro_g]
            amb = ma.decidir(d_agua, d_capa, d_cauce, par_amb)
            con_dato = tipo_mod != SIN_DATO_CLASE
            compuesta = np.full(tipo_mod.shape, SIN_DATO_CLASE, dtype='uint8')
            if solo_ambiente:
                # ma.CODIGOS y los del nivel «ambiente» son los mismos (1 arroyo, …)
                compuesta[con_dato] = np.where(tipo_mod[con_dato] != 0, amb[con_dato], 0)
            else:
                compuesta[con_dato] = mr.componer(nivel_c, amb[con_dato], tipo_mod[con_dato])
            f_clase.escribir(compuesta, 1)
            cuenta = {c: int((compuesta == c).sum()) for c, _ in cat_mapa}
            amb = np.where(con_dato, amb, SIN_DATO_CLASE).astype('uint8')
            cod_amb = {n: c for n, c in ma.CODIGOS.items()}
            f_amb = nuevo('ambiente', tipo='uint8', sin_dato=SIN_DATO_CLASE)
            f_amb.paleta({c: _rgb(color(n)) for n, c in cod_amb.items()})
            f_amb.etiquetas(CATEGORIAS='; '.join(f'{c} = {n}' for n, c in cod_amb.items()),
                            SIN_DATO=str(SIN_DATO_CLASE), REGLA=ma.frase(par_amb),
                            **{k.upper(): str(v) for k, v in par_amb.items()})
            f_amb.nombre_banda(1, 'ambiente por cercania (codigo)')
            f_amb.escribir(amb, 1)
            f_dist = nuevo('distancias', bandas=3, sin_dato=float('nan'))
            for i, (d, que) in enumerate(((d_agua, 'al agua de una laguna (Sentinel-2)'),
                                          (d_capa, 'a una laguna de la capa'),
                                          (d_cauce, 'a un cauce')), 1):
                f_dist.escribir(np.where(np.isfinite(d), d, np.nan).astype('float32'), i)
                f_dist.nombre_banda(i, f'distancia {que}, en metros')
            # lo que se ha tomado por laguna y por cauce: manda el punto de la capa,
            # después el agua y después el cauce
            refs = np.zeros(tipo_mod.shape, dtype='uint8')
            refs[cauces] = 2
            refs[agua] = 1
            refs[ma.puntos_en_rejilla(lagunas, rej)] = 3
            f_ref = nuevo('referencias', tipo='uint8', sin_dato=0)
            f_ref.paleta({c: _rgb(h) for c, h in ma.COLORES_REFERENCIAS.items()})
            f_ref.etiquetas(CATEGORIAS='; '.join(f'{c} = {n}'
                                                 for c, n in ma.REFERENCIAS.items()))
            f_ref.nombre_banda(1, 'lagunas y cauces (codigo)')
            f_ref.escribir(refs, 1)
            borr = con_dato & (tipo_mod != 0)
            info_amb = {
                'nivel': nivel_c, **par_amb, 'regla': ma.frase(par_amb),
                'cauces': de_cauces, 'margen_m': margen * e,
                'fuentes': dict(ambiente.get('fuentes') or {}),
                'lagunas_capa': len(lagunas),
                'lagunas_capa_en_el_mapa': int((refs == 3).sum()),
                'pixeles_agua': int(agua.sum()), 'pixeles_cauce': int(cauces.sum()),
                'categorias': [{'codigo': c, 'categoria': n, 'color': color(n),
                                'pixeles': int((amb[borr] == c).sum())}
                               for n, c in cod_amb.items()]}
        cuenta_pur = {}
        if par_pur:
            # La pureza, con el mapa entero delante (ver mapeo_rf.pureza_de): puro si
            # el verde llega al corte; si no, mixto-agua si el píxel es agua o toca
            # agua, y mixto-roca en otro caso.
            toca = _con_vecinos(agua)
            pur = np.full(llega.shape, SIN_DATO_CLASE, dtype='uint8')
            pur[llega == 1] = cod_pur['puro']
            pur[(llega == 2) & toca] = cod_pur['mixto-agua']
            pur[(llega == 2) & ~toca] = cod_pur['mixto-roca']
            cuenta_pur = {n: int((pur == c).sum()) for n, c in cod_pur.items()}
            f_pur = nuevo('pureza', tipo='uint8', sin_dato=SIN_DATO_CLASE)
            f_pur.paleta({c: _rgb(color(n)) for n, c in cod_pur.items()})
            f_pur.etiquetas(CATEGORIAS='; '.join(f'{c} = {n}' for n, c in cod_pur.items()),
                            SIN_DATO=str(SIN_DATO_CLASE),
                            **{k.upper(): str(v) for k, v in par_pur.items()})
            f_pur.nombre_banda(1, 'pureza dentro del pixel (codigo)')
            f_pur.escribir(pur, 1)
        # Ahora sí, los ficheros.
        for f in abiertos:
            f.cerrar()
    except BaseException:
        for f in abiertos:
            f.descartar()
        _borrar(rutas.values())
        raise
    # Versiones reducidas dentro de cada GeoTIFF, para dibujarlo deprisa cuando se ve
    # de lejos. Los datos a tamaño real no cambian.
    con_raster = (('clase', 'pmax', 'prob') + (('pborr',) if con_pborr else ())
                  + (('verde', 'pureza') if par_pur else ())
                  + (('ambiente', 'distancias', 'referencias') if par_amb is not None else ())
                  + (('agua',) if info_agua else ()))
    reducidas = [_con_vistas_reducidas(
        rutas[k], categorica=(k in ('clase', 'pureza', 'ambiente', 'referencias', 'agua')))
        for k in con_raster]

    total = rej['ancho'] * rej['alto']
    ha = rej['escala'] ** 2 / 10_000
    meta = {
        'nivel': mod.get('nivel'), 'firma_modelo': mod.get('firma'),
        'cuando': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'segundos': round(time.time() - t0, 1),
        'rejilla': dict(rej), 'crs': f"EPSG:{rej['epsg']}", 'resolucion_m': rej['escala'],
        'limites_lonlat': [round(v, 7) for v in limites_lonlat(rej)],
        'bloques': len(ventanas), 'bloques_leidos': leidos, 'lado_bloque': lado,
        'predictores': list(mod['predictores']),
        # cómo se ha decidido la categoría de cada píxel
        'regla': {'criterio': regla.get('criterio'), 'umbral': umbral},
        'categorias': [{'codigo': c, 'categoria': n, 'color': color(n, i),
                        'pixeles': cuenta[c], 'hectareas': round(cuenta[c] * ha, 2)}
                       for i, (c, n) in enumerate(cat_mapa)],
        'pixeles': total, 'pixeles_clasificados': validos,
        'pixeles_sin_dato': total - validos,
        # dentro de la zona y de la máscara, y de ellos los que no se han podido
        # clasificar porque les falta alguna variable
        'pixeles_zona': en_zona, 'pixeles_sin_variables': en_zona - validos,
        'pmax_media': round(suma_pmax / validos, 6) if validos else None,
        'con_zona': zona_utm is not None, 'con_mascara': mascara_utm is not None,
        'sin_dato_clase': SIN_DATO_CLASE,
        'ficheros': {k: FICHEROS[k] for k in con_raster + ('codigos',)},
        'vistas_reducidas': reducidas[0],
    }
    if par_pur:
        # con qué se ha calculado la pureza dentro del píxel y cuánto hay de cada una
        meta['pureza'] = {
            **par_pur, 'variable_verde': mr.VARIABLE_VERDE, 'variable_agua': mr.VARIABLE_AGUA,
            'categorias': [{'codigo': c, 'categoria': n, 'color': color(n),
                            'pixeles': cuenta_pur[n], 'hectareas': round(cuenta_pur[n] * ha, 2)}
                           for n, c in cod_pur.items()]}
    if info_agua:
        meta['mascara_agua'] = {
            **info_agua, 'hectareas_quitadas': round(info_agua['pixeles_quitados'] * ha, 2),
            'nota': 'las probabilidades son las del modelo, sin quitar el agua'}
    if info_amb is not None:
        # Las probabilidades son las del modelo, que no conoce las categorías
        # compuestas: aquí van las suyas, en el orden de las bandas.
        meta['categorias_modelo'] = [{'codigo': c, 'categoria': nombre[c]} for c in codigos]
        for c in info_amb['categorias']:
            c['hectareas'] = round(c['pixeles'] * ha, 2)
        meta['ambiente'] = info_amb
        # cuánto borreguil hay de cada ambiente, sumando sus categorías
        meta['ramas'] = [{'rama': c['categoria'], 'pixeles': c['pixeles'],
                          'hectareas': c['hectareas']} for c in info_amb['categorias']]
    meta.update(extra or {})
    rutas['codigos'].write_text(codigos_csv(meta), encoding='utf-8-sig')
    rutas['meta'].write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                             encoding='utf-8')
    return meta


def _con_vistas_reducidas(ruta, categorica):
    """Añade a un GeoTIFF sus versiones reducidas (a la mitad, a la cuarta parte…) y
    devuelve los factores. En las categorías, cada celda reducida es la categoría más
    frecuente; en las probabilidades, la media. Solo sirven para dibujar: si no se
    pueden añadir, el mapa vale igual."""
    import mapeo_tif as mt
    try:
        return mt.anadir_reducidas(ruta, categorica)
    except Exception:                               # noqa: BLE001
        return []


def _borrar(rutas, intentos=5):
    """Borra unos ficheros sin que un fallo al borrar tape el error que se está
    tratando. En Windows un fichero recién cerrado puede seguir ocupado un instante
    (el antivirus lo está mirando): se reintenta y, si no se deja, se queda; sin
    mapa.json, que se escribe lo último, esa carpeta no cuenta como mapa."""
    for ruta in rutas:
        for i in range(intentos):
            try:
                Path(ruta).unlink(missing_ok=True)
                break
            except OSError:
                time.sleep(0.2 * (i + 1))


def codigos_csv(meta):
    """La correspondencia código ↔ categoría del mapa, con su color y su superficie."""
    lineas = ['codigo,categoria,color,pixeles,hectareas']
    for c in meta['categorias']:
        lineas.append(f"{c['codigo']},{c['categoria']},{c['color']},{c['pixeles']},"
                      f"{c['hectareas']}")
    lineas.append(f"{meta['sin_dato_clase']},sin dato,,{meta['pixeles_sin_dato']},"
                  f"{round(meta['pixeles_sin_dato'] * meta['resolucion_m'] ** 2 / 10_000, 2)}")
    return '\n'.join(lineas) + '\n'


def leer_meta(carpeta):
    """mapa.json de una carpeta de mapa, o None si no hay un mapa completo en ella."""
    carpeta = Path(carpeta)
    try:
        meta = json.loads((carpeta / FICHEROS['meta']).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    if not all((carpeta / FICHEROS[k]).exists() for k in ('clase', 'pmax', 'prob')):
        return None
    return meta


# ============================================================
# Para el visor
# ------------------------------------------------------------
# El mapa está en la rejilla de Sentinel-2 (UTM, 10 m). El visor usa la proyección de
# los mapas web (Mercator), que es OTRA rejilla: algo girada respecto a la UTM y con
# píxeles de otro tamaño. Si se convirtiera el mapa entero de una vez a esa otra
# rejilla, cada píxel dibujado caería hasta medio píxel fuera de su sitio (5 m), y al
# acercarse se vería que no casa con los píxeles reales.
#
# Por eso se dibuja solo lo que se está viendo y a la resolución de la pantalla: cada
# punto de la pantalla toma el valor del píxel de Sentinel-2 que tiene debajo. Así los
# bordes de los píxeles caen donde son, a cualquier zoom y con mapas de cualquier
# tamaño.
# ============================================================
R_TIERRA = 6378137.0
RES_ZOOM0 = 2 * math.pi * R_TIERRA / 256      # metros de Mercator por punto de pantalla
MERCATOR = 3857
# Dónde cae cada punto de pantalla se calcula de verdad en una malla de nudos, y entre
# nudos se interpola. Los nudos no se separan más de esto: así lo interpolado no se
# aparta ni un centímetro de lo calculado punto a punto, y es cien veces más rápido.
MALLA_M, MALLA_PUNTOS = 500.0, 32
# De lejos, en un punto de pantalla caben varios píxeles: se miran hasta 8 × 8 sitios
# repartidos por el punto.
MAX_MUESTRAS = 8


def _a_mercator(lon, lat):
    lat = max(-85.0, min(85.0, float(lat)))
    return (R_TIERRA * math.radians(float(lon)),
            R_TIERRA * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)))


def _de_mercator(x, y):
    return (math.degrees(x / R_TIERRA),
            math.degrees(2 * math.atan(math.exp(y / R_TIERRA)) - math.pi / 2))


def vista_inicial(centro, zoom, ancho_px=1600, alto_px=900):
    """(oeste, sur, este, norte) de lo que mostraría un visor de ese tamaño centrado en
    [lat, lon] a ese zoom. Se usa hasta que el visor dice lo que muestra de verdad; el
    tamaño es generoso para que cubra cualquier pantalla."""
    res = RES_ZOOM0 / 2 ** zoom
    x, y = _a_mercator(centro[1], centro[0])
    o, s = _de_mercator(x - ancho_px / 2 * res, y - alto_px / 2 * res)
    e, n = _de_mercator(x + ancho_px / 2 * res, y + alto_px / 2 * res)
    return o, s, e, n


def leer_vista(estado):
    """Lo que el visor dice que está mostrando: ((oeste, sur, este, norte), zoom), o
    None si todavía no lo ha dicho. `estado` es lo que devuelve st_folium."""
    try:
        b = estado['bounds']
        vista = (float(b['_southWest']['lng']), float(b['_southWest']['lat']),
                 float(b['_northEast']['lng']), float(b['_northEast']['lat']))
        zoom = float(estado['zoom'])
    except (KeyError, TypeError, ValueError):
        return None
    if not (vista[2] > vista[0] and vista[3] > vista[1]) or not 0 <= zoom <= 24:
        return None
    return vista, zoom


def leer_clic(estado):
    """El último punto pinchado en el visor, (lon, lat), o None."""
    try:
        c = estado['last_clicked']
        return float(c['lng']), float(c['lat'])
    except (KeyError, TypeError, ValueError):
        return None


def _entre_nudos(pos, nudos):
    """Para interpolar: de cada posición, el nudo que tiene antes y cuánto se ha
    apartado de él (0 = está en ese nudo; 1 = en el siguiente)."""
    import numpy as np
    k = np.clip(np.searchsorted(nudos, pos, side='right') - 1, 0, len(nudos) - 2)
    return k, (pos - nudos[k]) / (nudos[k + 1] - nudos[k])


def bajo_cada_punto(epsg, c0, f1, res, ancho, alto, sub=1):
    """Dónde cae, en las coordenadas del mapa (su EPSG), cada punto de una imagen de
    pantalla: la de ancho × alto puntos de `res` metros de Mercator cuya esquina superior
    izquierda está en (c0 · res, f1 · res).

    Devuelve dos matrices (x, y) de (alto · sub, ancho · sub): con sub = 1, el centro de
    cada punto; con más, sub × sub sitios repartidos dentro de cada punto."""
    import numpy as np
    import pyproj
    paso = max(1, min(MALLA_PUNTOS, int(MALLA_M / res)))
    nj = np.unique(np.append(np.arange(0, ancho, paso), ancho)).astype('float64')
    ni = np.unique(np.append(np.arange(0, alto, paso), alto)).astype('float64')
    xm, ym = np.meshgrid((c0 + nj) * res, (f1 - ni) * res)
    xu, yu = pyproj.Transformer.from_crs(MERCATOR, int(epsg), always_xy=True).transform(xm, ym)
    kj, tj = _entre_nudos((np.arange(ancho * sub) + 0.5) / sub, nj)
    ki, ti = _entre_nudos((np.arange(alto * sub) + 0.5) / sub, ni)
    ti = ti[:, None]
    salen = []
    for m in (xu, yu):
        a = m[:, kj] * (1 - tj) + m[:, kj + 1] * tj          # a lo ancho…
        salen.append(a[ki] * (1 - ti) + a[ki + 1] * ti)      # …y a lo alto
    return salen[0], salen[1]


def imagen_encuadre(ruta, banda, vista, zoom, categorica=False, margen=0.5,
                    max_px=4_000_000):
    """La parte de una banda de un GeoTIFF que asoma por el visor, dibujada punto a
    punto de pantalla.

    vista: (oeste, sur, este, norte), en grados, de lo que muestra el visor; zoom: su
    nivel de zoom. Alrededor se dibuja un margen (media pantalla por cada lado) para
    que desplazarse un poco no obligue a dibujar otra vez.

    Devuelve None si el ráster no asoma por ahí. Si no, un dict con:
      matriz   los códigos (uint8, 255 = sin dato) o los valores (float32, NaN = sin
               dato), una celda por punto de pantalla;
      limites  [[sur, oeste], [norte, este]] de esa imagen, en grados;
      cubre    (x0, y0, x1, y1) de esa imagen, en metros de Mercator;
      res      metros de Mercator por punto.

    Cada punto toma el valor del píxel que tiene debajo. Si el mapa se ve tan de lejos
    que en un punto caben varios píxeles, se queda con la categoría más frecuente (o
    con la media, en las probabilidades) de lo que tiene debajo, y se apoya en las
    versiones reducidas del fichero para no leerlo entero.
    """
    import mapeo_tif as mt
    res = RES_ZOOM0 / 2 ** float(zoom)
    x0, y0 = _a_mercator(vista[0], vista[1])
    x1, y1 = _a_mercator(vista[2], vista[3])
    mx, my = (x1 - x0) * margen, (y1 - y0) * margen
    x0, x1, y0, y1 = x0 - mx, x1 + mx, y0 - my, y1 + my
    with mt.abrir(ruta) as src:
        bx0, by0, bx1, by1 = _marco_en(src.epsg, MERCATOR, *src.limites)
        x0, x1, y0, y1 = max(x0, bx0), min(x1, bx1), max(y0, by0), min(y1, by1)
        if not (x1 > x0 and y1 > y0):
            return None
        return _dibujar_encuadre(src, banda, x0, y0, x1, y1, res, categorica, max_px)


def _dibujar_encuadre(src, banda, x0, y0, x1, y1, res, categorica, max_px):
    """Lo que hace imagen_encuadre con el fichero ya abierto y la vista ya recortada a lo
    que ocupa el mapa (en metros de Mercator)."""
    import numpy as np

    import mapeo_tif as mt
    reducidas, pixel_m = src.reducidas, src.escala
    # La rejilla de la imagen, en múltiplos de la resolución: al desplazarse, los
    # puntos caen siempre en los mismos sitios y la imagen no «baila».
    c0, c1 = math.floor(x0 / res), math.ceil(x1 / res)
    f0, f1 = math.floor(y0 / res), math.ceil(y1 / res)
    if (c1 - c0) * (f1 - f0) > max_px:              # una pantalla descomunal: más grueso
        res *= math.sqrt((c1 - c0) * (f1 - f0) / max_px)
        c0, c1 = math.floor(x0 / res), math.ceil(x1 / res)
        f0, f1 = math.floor(y0 / res), math.ceil(y1 / res)
    ancho, alto = c1 - c0, f1 - f0
    # Cuántos píxeles del mapa caben en un punto de pantalla (en Mercator las
    # distancias salen agrandadas 1/cos(latitud)).
    lat = _de_mercator(0.0, (y0 + y1) / 2)[1]
    por_punto = res * math.cos(math.radians(lat)) / pixel_m
    nivel = None
    for i, factor in enumerate(reducidas):
        if factor <= por_punto:
            nivel = i
    factor = 1 if nivel is None else reducidas[nivel]
    por_punto /= factor
    # De cerca, un sitio por punto: el píxel de debajo, tal cual. De lejos, varios
    # repartidos por el punto, para quedarse con lo más frecuente o con la media.
    sub = 1 if por_punto <= 1.0 else min(MAX_MUESTRAS, math.ceil(por_punto))
    while sub > 1 and ancho * alto * sub * sub > 4 * max_px:
        sub -= 1
    xs, ys = bajo_cada_punto(src.epsg, c0, f1, res, ancho, alto, sub)
    lado = pixel_m * factor
    filas, cols = src.forma(nivel)
    col = np.floor((xs - src.x0) / lado).astype('int64')
    fila = np.floor((src.y0 - ys) / lado).astype('int64')
    cae = (col >= 0) & (col < cols) & (fila >= 0) & (fila < filas)
    salida = (np.full(col.shape, SIN_DATO_CLASE, dtype='uint8') if categorica
              else np.full(col.shape, np.nan, dtype='float32'))
    if cae.any():
        f_a, f_b = int(fila[cae].min()), int(fila[cae].max())
        c_a, c_b = int(col[cae].min()), int(col[cae].max())
        datos = src.leer(banda, (c_a, f_a, c_b - c_a + 1, f_b - f_a + 1), nivel)
        valores = datos[fila[cae] - f_a, col[cae] - c_a]
        if src.sin_dato is not None and src.sin_dato == src.sin_dato:
            # el «sin dato» del fichero pasa a ser el de la imagen (255 o NaN)
            falta = valores == src.sin_dato
            valores = valores.astype(salida.dtype)
            valores[falta] = SIN_DATO_CLASE if categorica else np.nan
        salida[cae] = valores
    if sub > 1:
        salida = mt.reducir(salida, sub, categorica, SIN_DATO_CLASE if categorica else None)
    o, s = _de_mercator(c0 * res, f0 * res)
    e, n = _de_mercator(c1 * res, f1 * res)
    return {'matriz': salida, 'limites': [[s, o], [n, e]],
            'cubre': (c0 * res, f0 * res, c1 * res, f1 * res), 'res': res}


def dibujar_vista(memoria, ruta, banda, vista, zoom, pintar, categorica=False):
    """La imagen que hay que poner en el visor para esa vista: (texto «data:image…»,
    límites), o None si el mapa no asoma.

    `memoria` es un dict que se conserva entre llamadas (uno por mapa y capa): si la
    vista nueva sigue cabiendo en lo último que se dibujó, al mismo zoom, se devuelve
    lo mismo y el visor no tiene que cambiar nada. `pintar` convierte la matriz en una
    imagen RGBA (pintar_clases o pintar_probabilidad).
    """
    import mapeo_tif as mt
    if 'marco' not in memoria:                      # lo que ocupa el mapa, en Mercator
        with mt.abrir(ruta) as src:
            memoria['marco'] = _marco_en(src.epsg, MERCATOR, *src.limites)
    bx0, by0, bx1, by1 = memoria['marco']
    x0, y0 = _a_mercator(vista[0], vista[1])
    x1, y1 = _a_mercator(vista[2], vista[3])
    x0, x1, y0, y1 = max(x0, bx0), min(x1, bx1), max(y0, by0), min(y1, by1)
    if not (x1 > x0 and y1 > y0):
        return None
    ult = memoria.get('ultima')
    if ult and ult['zoom'] == zoom:
        ux0, uy0, ux1, uy1 = ult['cubre']
        if ux0 <= x0 and uy0 <= y0 and ux1 >= x1 and uy1 >= y1:
            return ult['url'], ult['limites']
    img = imagen_encuadre(ruta, banda, vista, zoom, categorica=categorica)
    if img is None:
        return None
    memoria['ultima'] = {'zoom': zoom, 'cubre': img['cubre'], 'limites': img['limites'],
                         'url': a_png(pintar(img['matriz']), rapido=True)}
    return memoria['ultima']['url'], img['limites']


# Las capas ráster del visor (categoría, probabilidad máxima y probabilidad de una
# categoría) son FIJAS: están en el recuadro de capas del mapa, donde se encienden y se
# apagan. Lo que cambia cada vez que se mueve el visor es la IMAGEN que llevan, que se
# les pone con un poco de JavaScript. Así el visor no se recarga, no pierde el zoom y no
# olvida qué capas tenía encendidas la persona.
# Un punto transparente: lo que llevan las capas hasta que se les pone su imagen.
PNG_VACIO = ('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAA'
             'DUlEQVR42mNgYGBgAAAABQABeqhXUAAAAABJRU5ErkJggg==')


def guion_registro(mapa, control, capas):
    """JavaScript que apunta en el visor sus capas ráster, para poder cambiarles después
    la imagen. Los argumentos son los NOMBRES que tienen dentro del visor el mapa, el
    recuadro de capas y cada capa: {'clase': …, 'pmax': …, 'pborr': …, 'pureza': …,
    'px': …} ('pborr', la probabilidad de borreguil, y 'pureza', la de dentro del píxel,
    solo si el mapa las tiene)."""
    return ('window.capas_mrf = {mapa: %s, control: %s, %s, elegida: ""};'
            % (mapa, control, ', '.join(f'{k}: {v}' for k, v in capas.items())))


def guion_capas(imagenes, opacidad, elegida='', nombre=''):
    """JavaScript que pone en las capas del visor lo que hay que ver ahora.

    imagenes: {capa: (imagen «data:…», [[sur, oeste], [norte, este]])} para 'clase',
    'pmax', 'pborr', 'pureza' y 'refs' (si el mapa las tiene) y, si se ha elegido una
    categoría, 'px'.
    elegida: qué categoría es esa ('' = ninguna); nombre: el rótulo que llevará en el
    recuadro de capas.

    No enciende ni apaga lo que la persona ha tocado en el recuadro: solo cambia las
    imágenes y la opacidad. Lo único que mueve es la capa de la categoría elegida, y
    solo cuando la elección cambia: entonces la añade al recuadro y la enciende
    (apagando las demás, que una encima de otra no se leen), o la retira y vuelve a
    encender la de categorías."""
    datos = {k: [v[0], v[1]] for k, v in imagenes.items() if v}
    return """(function () {
    var c = window.capas_mrf;
    if (!c) { return; }
    var datos = %s;
    for (var k in datos) {
        if (c[k]) { c[k].setUrl(datos[k][0]); c[k].setBounds(datos[k][1]); c[k].setOpacity(%s); }
    }
    var elegida = %s, nombre = %s;
    if (c.elegida !== elegida) {
        if (c.elegida) { c.control.removeLayer(c.px); }
        if (elegida) {
            c.control.addOverlay(c.px, nombre);
            ["clase", "pmax", "pborr", "pureza"].forEach(function (k) {
                if (c[k]) { c.mapa.removeLayer(c[k]); }
            });
            c.mapa.addLayer(c.px);
        } else {
            c.mapa.removeLayer(c.px); c.mapa.addLayer(c.clase);
        }
        c.elegida = elegida;
    }
})();""" % (json.dumps(datos), json.dumps(float(opacidad)), json.dumps(str(elegida)),
            json.dumps(str(nombre)))


def pintar_clases(matriz, categorias, opacidad=0.75):
    """Matriz de códigos → imagen RGBA (uint8) con el color de cada categoría; lo que
    no tiene dato queda transparente, y las categorías de TENUES, más transparentes
    que las demás."""
    import numpy as np
    rgba = np.zeros(matriz.shape + (4,), dtype='uint8')
    for c in categorias:
        donde = matriz == c['codigo']
        alfa = 255 * opacidad * TENUES.get(c['categoria'], 1.0)
        rgba[donde] = _rgb(c['color']) + (int(round(alfa)),)
    return rgba


def pintar_probabilidad(matriz, opacidad=0.8):
    """Matriz de probabilidades (0–1) → imagen RGBA con la rampa de RAMPA; lo que no
    tiene dato queda transparente."""
    import numpy as np
    paradas = np.array([_rgb(h) for h in RAMPA], dtype='float64')
    ok = np.isfinite(matriz)
    v = np.clip(np.where(ok, matriz, 0.0), 0.0, 1.0) * (len(RAMPA) - 1)
    i = np.minimum(v.astype(int), len(RAMPA) - 2)
    f = (v - i)[..., None]
    rgb = paradas[i] * (1 - f) + paradas[i + 1] * f
    rgba = np.zeros(matriz.shape + (4,), dtype='uint8')
    rgba[..., :3] = np.round(rgb).astype('uint8')
    rgba[..., 3] = np.where(ok, int(round(255 * opacidad)), 0)
    return rgba


def a_png(rgba, rapido=False):
    """Imagen RGBA (uint8) → texto «data:image/png;base64,…», para ponerla en el visor
    sin escribir ningún fichero. `rapido`: comprime menos y tarda mucho menos, para
    las imágenes que se rehacen cada vez que se mueve el visor."""
    import base64
    import io

    from PIL import Image
    buf = io.BytesIO()
    if rapido:
        Image.fromarray(rgba).save(buf, format='PNG', compress_level=3)
    else:
        Image.fromarray(rgba).save(buf, format='PNG', optimize=True)
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')


def consultar_varios(carpeta, puntos):
    """Lo que dice el mapa en el píxel que contiene cada punto [(lon, lat), …]: leído
    de los GeoTIFF completos, no de la imagen del visor.

    Devuelve una lista paralela. Cada elemento es None si el punto cae fuera del mapa;
    si no, un dict con la fila y la columna, el centro del píxel en UTM y en grados,
    el código y la categoría (None si el píxel no tiene dato), su probabilidad, la de
    cada categoría y la de ser borreguil de algún tipo (None si el mapa no la tiene);
    y, si el mapa lleva la pureza dentro del píxel, el verde estimado ('verde') y la
    pureza ('pureza': None si el píxel no es borreguil). Si el ambiente se puso por
    cercanía, también 'ambiente' y 'distancias': los metros hasta el agua de una laguna
    ('agua'), hasta una laguna de la capa ('capa') y hasta un cauce ('cauce'); None =
    no hay nada de eso cerca. Las probabilidades son siempre las del modelo."""
    import pyproj

    import mapeo_tif as mt
    carpeta = Path(carpeta)
    meta = leer_meta(carpeta)
    if meta is None:
        return [None] * len(puntos)
    nombre = {c['codigo']: c['categoria'] for c in meta['categorias']}
    out = []
    ruta_pborr = carpeta / FICHEROS['pborr']
    with contextlib.ExitStack() as pila:
        def abierto(ruta):
            return pila.enter_context(mt.abrir(ruta))
        fc = abierto(carpeta / FICHEROS['clase'])
        fp = abierto(carpeta / FICHEROS['pmax'])
        fq = abierto(carpeta / FICHEROS['prob'])
        fb = abierto(ruta_pborr) if ruta_pborr.exists() else None
        con_pureza = all((carpeta / FICHEROS[k]).exists() for k in ('verde', 'pureza'))
        fv = abierto(carpeta / FICHEROS['verde']) if con_pureza else None
        fu = abierto(carpeta / FICHEROS['pureza']) if con_pureza else None
        de_pureza = {c['codigo']: c['categoria']
                     for c in (meta.get('pureza') or {}).get('categorias', [])}
        con_amb = all((carpeta / FICHEROS[k]).exists() for k in ('ambiente', 'distancias'))
        fa = abierto(carpeta / FICHEROS['ambiente']) if con_amb else None
        fd = abierto(carpeta / FICHEROS['distancias']) if con_amb else None
        de_ambiente = {c['codigo']: c['categoria']
                       for c in (meta.get('ambiente') or {}).get('categorias', [])}
        del_modelo = meta.get('categorias_modelo') or meta['categorias']
        a_mapa = pyproj.Transformer.from_crs(4326, fc.epsg, always_xy=True)
        a_geo = pyproj.Transformer.from_crs(fc.epsg, 4326, always_xy=True)
        for lon, lat in puntos:
            x, y = a_mapa.transform(float(lon), float(lat))
            fila, col = fc.indice(x, y)
            if not (0 <= fila < fc.alto and 0 <= col < fc.ancho):
                out.append(None)
                continue
            win = (col, fila, 1, 1)
            codigo = int(fc.valor(fila, col))
            cx, cy = fc.centro(fila, col)
            clon, clat = a_geo.transform(cx, cy)
            q = {'fila': int(fila), 'columna': int(col), 'x': float(cx), 'y': float(cy),
                 'lon': float(clon), 'lat': float(clat), 'codigo': None, 'categoria': None,
                 'pmax': None, 'p_borreguil': None, 'verde': None, 'pureza': None,
                 'ambiente': None, 'distancias': None, 'probabilidades': []}
            if codigo != SIN_DATO_CLASE:
                p = fq.leer(ventana=win)[:, 0, 0]
                q['codigo'], q['categoria'] = codigo, nombre.get(codigo)
                q['pmax'] = float(fp.valor(fila, col))
                if fb is not None:
                    q['p_borreguil'] = float(fb.valor(fila, col))
                if fv is not None:
                    v = float(fv.valor(fila, col))
                    q['verde'] = v if v == v else None
                    q['pureza'] = de_pureza.get(int(fu.valor(fila, col)))
                if fa is not None:
                    q['ambiente'] = de_ambiente.get(int(fa.valor(fila, col)))
                    d = [float(v) for v in fd.leer(ventana=win)[:, 0, 0]]
                    q['distancias'] = dict(zip(('agua', 'capa', 'cauce'),
                                               (v if v == v else None for v in d)))
                q['probabilidades'] = [{'codigo': c['codigo'], 'categoria': c['categoria'],
                                        'probabilidad': float(p[i])}
                                       for i, c in enumerate(del_modelo)]
            out.append(q)
    return out


def consultar(carpeta, lon, lat):
    """consultar_varios para un solo punto."""
    return consultar_varios(carpeta, [(lon, lat)])[0]
