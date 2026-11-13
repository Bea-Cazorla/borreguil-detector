"""
mapeo_ambiente.py — El AMBIENTE de cada píxel (arroyo, laguna o ladera), por cercanía.

El ambiente no es algo que se vea dentro del píxel: dice DÓNDE está el borreguil (junto
a una laguna, a lo largo de un cauce o en una ladera). Un modelo que solo conoce las
variables de cada píxel no puede saberlo: probado, daba por «laguna» casi todo el
borreguil de una zona, la mitad a más de 600 m del agua. Por eso, en el mapa de
«ambiente y humedad», el ambiente no sale del modelo sino de la distancia de cada píxel
a las lagunas y a los cauces, con la misma idea (y los mismos valores de partida) con
que la app propone el tipo de cada punto (borreguil_pipeline.HIER_THRESH):

  laguna   a menos de `laguna_capa_m` de una laguna conocida (una capa de puntos) o a
           menos de `laguna_agua_m` del agua de una laguna que se ve en Sentinel-2;
  arroyo   si no, a menos de `arroyo_m` de un cauce;
  ladera   el resto.

Qué cuenta como agua de laguna: los píxeles con NDWI medio de verano por encima de
0,05 en terreno casi llano (pendiente menor de 15°), en manchas de 3 píxeles o más. Con
«NDWI > 0» a secas entran las sombras y la nieve de las paredes; una mancha de 1 o 2
píxeles puede ser cualquier cosa, y las lagunas tan pequeñas las pone la capa.

Qué cuenta como cauce: las líneas de una capa de la usuaria o, si no la hay, los cauces
DEDUCIDOS DEL RELIEVE: se rellenan los hoyos del modelo de elevaciones, cada celda
vierte a su vecina más baja y es cauce la que recibe el agua de `cauce_ha` hectáreas o
más. Es una aproximación (el relieve tiene 30 m de detalle): sitúa bien las vaguadas,
no el hilo de agua.

Sin interfaz y sin Earth Engine: todo son funciones sobre matrices, para poder
probarlas solas (tests/test_mapeo_ambiente.py).
"""
import math

# Códigos del ambiente, los del nivel «ambiente» de la app (mapeo_rf.codigos).
CODIGOS = {'arroyo': 1, 'laguna': 2, 'ladera': 3}
SIN_DATO = 255

# Distancias de partida: las que ya usa la app al proponer el tipo de un punto (120 m a
# una laguna de la capa, 30 m a un cauce); 100 m a la orilla del agua que se ve; y 8 ha
# vertiendo para que una vaguada cuente como cauce.
DEFECTO = {'laguna_capa_m': 120.0, 'laguna_agua_m': 100.0, 'arroyo_m': 30.0,
           'cauce_ha': 8.0}
LIMITES = {'laguna_capa_m': (0.0, 1000.0), 'laguna_agua_m': (0.0, 1000.0),
           'arroyo_m': (0.0, 500.0), 'cauce_ha': (0.5, 500.0)}

VARIABLE_AGUA = 'ndwi_mean'           # NDWI medio de jun–sep
VARIABLE_PENDIENTE = 'slope_deg'
VARIABLE_RELIEVE = 'elev_dem_m'
NDWI_LAGUNA = 0.05                    # el mismo que borreguil_pipeline: «lámina de agua»
PENDIENTE_LAGUNA = 15.0               # una laguna está en llano; una pared en sombra, no
MIN_AGUA_PX = 3
# El relieve se lee con este margen alrededor de la zona: el agua que entra de fuera
# también hace cauce. Y hasta aquí se mide la distancia a los puntos de la capa.
MARGEN_RELIEVE_M = 1000
ALCANCE_PUNTOS_M = 1000

# Lo que se pinta en la capa «Lagunas y cauces» del visor: con qué se ha decidido.
REFERENCIAS = {1: 'agua de laguna (Sentinel-2)', 2: 'cauce', 3: 'laguna de la capa'}
COLORES_REFERENCIAS = {1: '#00E5FF', 2: '#1565C0', 3: '#FF4081'}


def params(valores=None):
    """Las distancias de la regla, comprobadas."""
    q = dict(DEFECTO)
    q.update({k: v for k, v in (valores or {}).items() if k in DEFECTO})
    try:
        q = {k: float(v) for k, v in q.items()}
    except (TypeError, ValueError):
        raise ValueError('el ambiente por cercanía necesita números: '
                         + ', '.join(DEFECTO)) from None
    for k, (desde, hasta) in LIMITES.items():
        if not (math.isfinite(q[k]) and desde <= q[k] <= hasta):
            raise ValueError(f'{k} tiene que estar entre {desde:g} y {hasta:g}')
    return q


# ============================================================
# Lagunas
# ============================================================
def candidata_a_agua(ndwi, pendiente):
    """Píxeles que pueden ser agua de laguna: NDWI medio alto y terreno casi llano.
    Donde falta alguno de los dos datos, no."""
    import numpy as np
    with np.errstate(invalid='ignore'):
        return ((np.asarray(ndwi, dtype='float32') > NDWI_LAGUNA)
                & (np.asarray(pendiente, dtype='float32') < PENDIENTE_LAGUNA))


def agua_de_laguna(candidata, minimo=MIN_AGUA_PX):
    """De los píxeles candidatos, los que forman manchas de `minimo` píxeles o más
    (contando también los que se tocan en diagonal)."""
    import numpy as np
    from scipy import ndimage as ndi
    candidata = np.asarray(candidata, dtype=bool)
    if minimo <= 1 or not candidata.any():
        return candidata.copy()
    etiquetas, n = ndi.label(candidata, structure=np.ones((3, 3), bool))
    tam = np.bincount(etiquetas.ravel(), minlength=n + 1)
    tam[0] = 0
    return (tam >= minimo)[etiquetas]


# ============================================================
# Distancias
# ============================================================
def distancia(mascara, escala=10):
    """Metros desde el centro de cada píxel hasta el del píxel marcado más cercano (0 en
    los marcados). Si no hay ninguno marcado, infinito en todas partes."""
    import numpy as np
    from scipy import ndimage as ndi
    mascara = np.asarray(mascara, dtype=bool)
    if not mascara.any():
        return np.full(mascara.shape, np.inf, dtype='float32')
    return (ndi.distance_transform_edt(~mascara) * escala).astype('float32')


def distancia_a_puntos(puntos, rej, hasta_m=ALCANCE_PUNTOS_M):
    """Metros desde el centro de cada píxel de la rejilla hasta el punto más cercano
    de `puntos` [(x, y), …] en la UTM de la rejilla (pueden caer fuera de ella). Exacta
    hasta `hasta_m`; más lejos, infinito (ya no hace falta saber cuánto)."""
    import numpy as np
    alto, ancho, e = rej['alto'], rej['ancho'], rej['escala']
    d = np.full((alto, ancho), np.inf, dtype='float32')
    radio = int(math.ceil(hasta_m / e)) + 1
    for x, y in puntos:
        col, fila = (x - rej['x0']) / e, (rej['y0'] - y) / e       # en píxeles, con decimales
        c0, c1 = max(0, int(col) - radio), min(ancho, int(col) + radio + 1)
        f0, f1 = max(0, int(fila) - radio), min(alto, int(fila) + radio + 1)
        if c0 >= c1 or f0 >= f1:
            continue
        ff, cc = np.mgrid[f0:f1, c0:c1]
        trozo = (np.hypot(cc + 0.5 - col, ff + 0.5 - fila) * e).astype('float32')
        trozo[trozo > hasta_m] = np.inf
        d[f0:f1, c0:c1] = np.minimum(d[f0:f1, c0:c1], trozo)
    return d


def puntos_en_rejilla(puntos, rej):
    """Matriz booleana con el píxel en que cae cada punto (los de fuera, nada)."""
    import numpy as np
    m = np.zeros((rej['alto'], rej['ancho']), dtype=bool)
    for x, y in puntos:
        col = int(math.floor((x - rej['x0']) / rej['escala']))
        fila = int(math.floor((rej['y0'] - y) / rej['escala']))
        if 0 <= col < rej['ancho'] and 0 <= fila < rej['alto']:
            m[fila, col] = True
    return m


# ============================================================
# Cauces
# ============================================================
def rellenar_hoyos(dem):
    """El relieve con los hoyos rellenos hasta el nivel por el que desaguan (y una pizca
    más por cada celda, para que también lo llano —una laguna— tenga hacia dónde
    verter). Es el algoritmo «Priority-Flood» de Barnes y otros (2014): se entra desde
    el borde, siempre por la celda más baja de las que se conocen. Donde no hay dato
    (NaN) se trata como borde: por ahí también sale el agua."""
    import heapq
    from array import array

    import numpy as np
    z = np.asarray(dem, dtype='float64')
    alto, ancho = z.shape
    hueco = ~np.isfinite(z)
    borde = np.zeros((alto, ancho), dtype=bool)
    borde[0, :] = borde[-1, :] = borde[:, 0] = borde[:, -1] = True
    if hueco.any():
        from scipy import ndimage as ndi
        borde |= ndi.binary_dilation(hueco, structure=np.ones((3, 3), bool))
    borde &= ~hueco
    lleno = array('d')
    lleno.frombytes(np.ascontiguousarray(np.where(hueco, 0.0, z)).tobytes())
    cerrado = bytearray(hueco.ravel().tobytes())
    cola = []
    for i in np.flatnonzero(borde).tolist():
        cerrado[i] = 1
        cola.append((lleno[i], i))
    heapq.heapify(cola)
    sacar, meter, siguiente, inf = heapq.heappop, heapq.heappush, math.nextafter, math.inf
    while cola:
        zi, i = sacar(cola)
        f, c = divmod(i, ancho)
        for f2 in (f - 1, f, f + 1):
            if 0 <= f2 < alto:
                base = f2 * ancho
                for c2 in (c - 1, c, c + 1):
                    if 0 <= c2 < ancho:
                        j = base + c2
                        if not cerrado[j]:
                            cerrado[j] = 1
                            zj = lleno[j]
                            if zj <= zi:
                                zj = siguiente(zi, inf)
                                lleno[j] = zj
                            meter(cola, (zj, j))
    fuera = np.frombuffer(lleno, dtype='float64').reshape(alto, ancho).copy()
    fuera[hueco] = np.nan
    return fuera


def superficie_que_vierte(dem, escala=10):
    """Hectáreas que vierten a cada celda (contándola a ella): se rellenan los hoyos y
    cada celda vierte a la más baja de sus ocho vecinas, la de mayor desnivel por metro
    (el método «D8»). Donde no hay dato, NaN."""
    from array import array

    import numpy as np
    lleno = rellenar_hoyos(dem)
    alto, ancho = lleno.shape
    hueco = ~np.isfinite(lleno)
    z = np.where(hueco, np.inf, lleno)
    con_borde = np.full((alto + 2, ancho + 2), np.inf)
    con_borde[1:-1, 1:-1] = z
    mejor = np.zeros((alto, ancho))
    receptor = np.full((alto, ancho), -1, dtype='int64')
    indice = np.arange(alto * ancho, dtype='int64').reshape(alto, ancho)
    for df in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if df == 0 and dc == 0:
                continue
            vecina = con_borde[1 + df:1 + df + alto, 1 + dc:1 + dc + ancho]
            with np.errstate(invalid='ignore'):
                caida = (z - vecina) / (escala * math.hypot(df, dc))
            gana = np.isfinite(caida) & (caida > mejor)
            mejor = np.where(gana, caida, mejor)
            receptor = np.where(gana, indice + df * ancho + dc, receptor)
    receptor[hueco] = -1
    # Lo que llega al borde, sale: fuera no se sabe hacia dónde baja el terreno, y si se
    # obligara al agua a seguir por dentro correría pegada al borde como un cauce falso.
    receptor[0, :] = receptor[-1, :] = receptor[:, 0] = receptor[:, -1] = -1
    # de la más alta a la más baja (los huecos, al final y fuera): cada una pasa a su
    # receptora todo lo que ha recibido
    orden = np.argsort(np.where(hueco, np.inf, -lleno), axis=None, kind='stable')
    en_orden = array('q')
    en_orden.frombytes(orden[: int((~hueco).sum())].astype('int64').tobytes())
    rec = array('q')
    rec.frombytes(np.ascontiguousarray(receptor).tobytes())
    celdas = array('d', [1.0]) * (alto * ancho)
    for i in en_orden:
        r = rec[i]
        if r >= 0:
            celdas[r] += celdas[i]
    ha = np.frombuffer(celdas, dtype='float64').reshape(alto, ancho) * (escala ** 2 / 10_000)
    ha = ha.copy()
    ha[hueco] = np.nan
    return ha


def cauces_del_relieve(dem, cauce_ha, escala=10):
    """Dónde hay cauce según el relieve: las celdas a las que vierten `cauce_ha`
    hectáreas o más."""
    import numpy as np
    with np.errstate(invalid='ignore'):
        return superficie_que_vierte(dem, escala) >= float(cauce_ha)


def leer_cauces(ruta):
    """Lee una capa de cauces (KML, GeoJSON, GeoPackage, shapefile o .zip) y devuelve
    sus líneas, unidas, en EPSG:4326. Si trae polígonos (un río ancho, un embalse),
    cuentan enteros. Los puntos no son cauces: se ignoran."""
    from pathlib import Path

    import geopandas as gpd
    from shapely.ops import unary_union
    try:
        gdf = gpd.read_file(ruta)
    except ImportError:
        # Sin pyogrio/GDAL (p. ej. su DLL bloqueada por Windows) un GeoJSON se
        # puede leer igual, porque es JSON; los demás formatos no.
        if Path(ruta).suffix.lower() not in ('.geojson', '.json'):
            raise
        import study_area
        geometrias = study_area.geometrias_geojson(ruta)
    else:
        if gdf.crs is None:
            gdf = gdf.set_crs('EPSG:4326')
        elif gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(4326)
        geometrias = list(gdf.geometry)
    sirven = [g for g in geometrias
              if g is not None and not g.is_empty and g.geom_type in (
                  'LineString', 'MultiLineString', 'Polygon', 'MultiPolygon')]
    if not sirven:
        raise ValueError(f'No hay líneas en {Path(ruta).name} (¿es una capa de puntos?)')
    return unary_union(sirven)


def rasterizar(geom_utm, rej):
    """Matriz booleana de la rejilla: True en los píxeles que TOCA la geometría (líneas
    o polígonos, ya en la UTM de la rejilla).

    Cada píxel es su cuadrado de la rejilla con su lado de arriba y su lado izquierdo:
    lo que pasa justo por la raya entre dos píxeles cuenta en uno, no en los dos."""
    import numpy as np
    import shapely
    from scipy import ndimage as ndi
    alto, ancho, e = rej['alto'], rej['ancho'], rej['escala']
    x0, y0 = rej['x0'], rej['y0']
    pintado = np.zeros((alto, ancho), dtype=bool)
    if geom_utm is None or geom_utm.is_empty:
        return pintado
    if not geom_utm.is_valid:
        geom_utm = shapely.make_valid(geom_utm)
    # solo lo que cae en la rejilla, con un píxel de holgura
    g = shapely.intersection(geom_utm, shapely.box(x0 - e, y0 - (alto + 1) * e,
                                                   x0 + (ancho + 1) * e, y0 + e))
    if g.is_empty:
        return pintado
    shapely.prepare(g)
    if g.area > 0:
        # de los polígonos, todo píxel con el centro dentro
        minx, miny, maxx, maxy = g.bounds
        c_a, c_b = max(0, int((minx - x0) // e)), min(ancho, int((maxx - x0) // e) + 1)
        f_a, f_b = max(0, int((y0 - maxy) // e)), min(alto, int((y0 - miny) // e) + 1)
        if c_b > c_a and f_b > f_a:
            xs, ys = np.meshgrid(x0 + (np.arange(c_a, c_b) + 0.5) * e,
                                 y0 - (np.arange(f_a, f_b) + 0.5) * e)
            pintado[f_a:f_b, c_a:c_b] = shapely.contains_xy(g, xs, ys)
    # Los que pueden tocar las líneas (o el contorno de los polígonos): los que contienen
    # alguno de sus puntos, tomando uno cada medio píxel, y los de alrededor. De esos se
    # comprueba uno por uno si de verdad los toca.
    xy = shapely.get_coordinates(shapely.segmentize(g, e / 2))
    col = np.floor((xy[:, 0] - x0) / e).astype('int64')
    fila = np.floor((y0 - xy[:, 1]) / e).astype('int64')
    en = (col >= 0) & (col < ancho) & (fila >= 0) & (fila < alto)
    cerca = np.zeros((alto, ancho), dtype=bool)
    cerca[fila[en], col[en]] = True
    cerca = ndi.binary_dilation(cerca, structure=np.ones((3, 3), dtype=bool)) & ~pintado
    ff, cc = np.nonzero(cerca)
    if len(ff):
        pizca = e * 1e-7
        cuadros = shapely.box(x0 + cc * e, y0 - (ff + 1) * e + pizca,
                              x0 + (cc + 1) * e - pizca, y0 - ff * e)
        toca = shapely.intersects(g, cuadros)
        pintado[ff[toca], cc[toca]] = True
    return pintado


def ampliar(rej, margen_px):
    """La misma rejilla con `margen_px` píxeles más por cada lado."""
    e = rej['escala']
    return {**rej, 'x0': rej['x0'] - margen_px * e, 'y0': rej['y0'] + margen_px * e,
            'ancho': rej['ancho'] + 2 * margen_px, 'alto': rej['alto'] + 2 * margen_px}


# ============================================================
# La regla
# ============================================================
def decidir(d_agua, d_capa, d_cauce, par=None):
    """El ambiente de cada píxel, como código (CODIGOS): laguna si está a menos de
    `laguna_agua_m` del agua de una laguna o a menos de `laguna_capa_m` de una laguna de
    la capa; si no, arroyo si está a menos de `arroyo_m` de un cauce; si no, ladera.
    Las distancias, en metros (infinito = no hay nada de eso cerca); «a menos de»
    incluye la distancia justa. Las comparaciones se hacen en float32, que es como se
    guardan las distancias: quien lea los ficheros obtiene lo mismo."""
    import numpy as np
    q = params(par)
    d_agua, d_capa, d_cauce = (np.asarray(d, dtype='float32') for d in (d_agua, d_capa, d_cauce))
    amb = np.full(d_agua.shape, CODIGOS['ladera'], dtype='uint8')
    amb[d_cauce <= np.float32(q['arroyo_m'])] = CODIGOS['arroyo']
    amb[(d_agua <= np.float32(q['laguna_agua_m']))
        | (d_capa <= np.float32(q['laguna_capa_m']))] = CODIGOS['laguna']
    return amb


def frase(par=None, en=False):
    """La regla, dicha en una frase con sus distancias."""
    q = params(par)
    if en:
        return (f"lake: within {q['laguna_capa_m']:g} m of a lake of the layer or "
                f"{q['laguna_agua_m']:g} m of lake water seen by Sentinel-2; stream: "
                f"otherwise, within {q['arroyo_m']:g} m of a watercourse; slope: the rest")
    return (f"laguna: a {q['laguna_capa_m']:g} m o menos de una laguna de la capa, o a "
            f"{q['laguna_agua_m']:g} m o menos del agua de una laguna que ve Sentinel-2; "
            f"arroyo: si no, a {q['arroyo_m']:g} m o menos de un cauce; ladera: el resto")
