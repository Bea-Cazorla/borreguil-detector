"""Pruebas del paso «Mapa» del mapeo por Random Forest: rejilla, bloques, predicción,
GeoTIFF y consulta de un píxel.

Se ejecutan con `python tests/test_mapeo_mapa.py`; no necesitan red. Earth Engine se
sustituye por un «mundo» inventado: cada píxel de 10 m tiene unos valores que dependen
solo de dónde está, así que se puede comprobar que el mapa dice en cada sitio lo que
el modelo predice con las variables de ESE píxel.
"""
import json
import math
import shutil
import sys
import tempfile
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import borreguil_pipeline as bp   # noqa: E402
import gee_backend as geb         # noqa: E402
import mapeo_ambiente as ma       # noqa: E402
import mapeo_raster as mra        # noqa: E402
import mapeo_rf as mr             # noqa: E402
import mapeo_tif as mt            # noqa: E402
from test_mapeo_datos import ausencia, es_borreguil, revisado   # noqa: E402

bp._imports()
import pyproj                     # noqa: E402
from shapely.geometry import box  # noqa: E402

EPSG = 32630
A_UTM = pyproj.Transformer.from_crs(4326, EPSG, always_xy=True)
A_GEO = pyproj.Transformer.from_crs(EPSG, 4326, always_xy=True)
# Zona de prueba: 1,2 km × 0,8 km en Sierra Nevada, con origen en un múltiplo de 10 m.
X0, Y0 = 467_000.0, 4_102_000.0
ANCHO_M, ALTO_M = 1200.0, 800.0
PRED = ['ndvi_late', 'ndmi_mean', 'elev_dem_m', 'slope_deg']     # se dan desordenadas
ORDEN = mr.ordenar_predictores(PRED)


def ruido(*claves):
    return (zlib.crc32('|'.join(str(c) for c in claves).encode()) % 1000) / 1000.0


def mundo(cx, cy):
    """Variables del píxel cuyo centro es (cx, cy): de oeste a este cambia la humedad;
    lo demás es ruido que depende solo del sitio."""
    v = {b: ruido(round(cx), round(cy), b) for b in geb.BANDAS_PILA}
    v['ndmi_mean'] = (cx - X0) / ANCHO_M + 0.02 * (ruido(round(cx), round(cy), 'h') - 0.5)
    return v


def centro_de(lon, lat):
    x, y = A_UTM.transform(lon, lat)
    return math.floor(x / 10) * 10 + 5, math.floor(y / 10) * 10 + 5


def lector(orden=ORDEN, rej=None, hueco=None, falla_en=None, llamadas=None):
    """Lee bloques del «mundo», como haría gee_backend.leer_bloque."""
    def leer(col, fila, ancho, alto):
        if llamadas is not None:
            llamadas.append((col, fila, ancho, alto))
        if falla_en is not None and len(llamadas or [0]) >= falla_en:
            raise RuntimeError('Earth Engine no ha devuelto un bloque del mapa (simulado).')
        out = np.empty((alto, ancho, len(orden)), dtype='float32')
        for f in range(alto):
            for c in range(ancho):
                x, y = mra.esquina(rej, col + c, fila + f)
                v = mundo(x + 5, y - 5)
                out[f, c] = [v[b] for b in orden]
                if hueco and hueco(col + c, fila + f):
                    out[f, c, 0] = np.nan
        return out
    return leer


def entrenado(con_ausencia=True):
    """Muestras repartidas por la zona (su categoría depende de dónde caen) y el
    modelo entrenado con lo que el «mundo» tiene en el píxel de cada una."""
    rows, valores = [], {}
    for k in range(60):
        # con decimales: ninguna cae justo en el borde entre dos píxeles
        x = X0 + 17.3 + (k * 197) % int(ANCHO_M - 30)
        y = Y0 - 13.7 - (k * 131) % int(ALTO_M - 30)
        lon, lat = A_GEO.transform(x, y)
        tercio = (x - X0) / ANCHO_M
        if tercio > 2 / 3 and con_ausencia:
            r = ausencia(k, lon=lon, lat=lat)
        else:
            r = revisado(k, humedad='húmedo' if tercio > 1 / 3 else 'seco', lon=lon, lat=lat)
        r['cuenca_id'] = f'c{k % 5}'
        rows.append(r)
        valores[mr.clave_punto(lon, lat)] = mundo(*centro_de(lon, lat))
    res = mr.muestras(rows, 'humedad', incluir_ausencias=con_ausencia,
                      es_borreguil=es_borreguil)
    tab = mr.construir_tabla(res, valores, PRED)
    mod = mr.entrenar_y_validar(tab, mr.grupos_espaciales(tab, rows)[0],
                                {'n_estimators': 40})
    return mod, tab, rows


MOD, TAB, ROWS = entrenado()
REJ = mra.rejilla((X0, Y0 - ALTO_M, X0 + ANCHO_M, Y0), EPSG)


def carpeta():
    return Path(tempfile.mkdtemp(prefix='mapa_'))


def limpiar(d):
    """Borra una carpeta temporal. No es lo que se prueba: si Windows tarda un instante
    en soltar un fichero recién cerrado, la prueba no falla por eso."""
    shutil.rmtree(d, ignore_errors=True)


def _gdal():
    """rasterio (GDAL), si en este equipo se puede cargar; None si no: en los equipos con
    el «Control inteligente de aplicaciones» de Windows activado, Windows lo bloquea a
    ratos. El mapa no lo usa; aquí solo sirve de segundo lector."""
    try:
        import rasterio
        return rasterio
    except Exception:                               # noqa: BLE001
        return None


GDAL = _gdal()
COTEJADOS = []


def abrir(ruta):
    """Abre un GeoTIFF del mapa. Si en el equipo hay GDAL, antes comprueba que GDAL ve en
    él lo mismo, cosa por cosa: es quien lo va a leer en QGIS."""
    f = mt.abrir(ruta)
    if GDAL is not None:
        with GDAL.open(ruta) as g:
            assert g.crs.to_epsg() == f.epsg
            assert (g.width, g.height, g.count) == (f.ancho, f.alto, f.bandas)
            assert (g.transform.a, g.transform.e, g.transform.c, g.transform.f) == \
                (f.escala, -f.escala, f.x0, f.y0)
            assert g.dtypes[0] == f.tipo and g.overviews(1) == f.reducidas
            assert (g.nodata == f.sin_dato or (g.nodata is not None and f.sin_dato is not None
                                               and np.isnan(g.nodata) and np.isnan(f.sin_dato)))
            assert tuple(n or '' for n in g.descriptions) == f.nombres
            assert all(g.tags().get(k) == v for k, v in f.etiquetas.items())
            if f.paleta:
                assert all(g.colormap(1)[c] == f.paleta[c] for c in range(256))
            assert np.array_equal(g.read(), f.leer(), equal_nan=True)
        COTEJADOS.append(Path(ruta).name)
    return f


def leer(ruta, banda=None):
    with abrir(ruta) as f:
        return f.leer(banda)


# ------------------------------------------------------------ rejilla y bloques
def test_la_rejilla_se_alinea_a_multiplos_de_10_m_y_cubre_la_zona():
    r = mra.rejilla((467_003.2, 4_101_204.9, 468_187.0, 4_101_998.1), EPSG)
    assert (r['x0'], r['y0']) == (467_000, 4_102_000) and r['epsg'] == EPSG
    assert (r['ancho'], r['alto']) == (119, 80)           # hasta 468 190 y 4 101 200
    assert r['x0'] + r['ancho'] * 10 >= 468_187.0 and r['y0'] - r['alto'] * 10 <= 4_101_204.9
    # una zona que ya cae en la rejilla no gana ni pierde píxeles
    assert (REJ['x0'], REJ['y0'], REJ['ancho'], REJ['alto']) == (X0, Y0, 120, 80)
    try:
        mra.rejilla((10, 10, 10, 20), EPSG)
    except ValueError:
        pass
    else:
        raise AssertionError('una zona sin superficie debía rechazarse')


def test_los_bloques_cubren_la_rejilla_sin_huecos_ni_solapes():
    for lado in (32, 50, 256):
        visto = np.zeros((REJ['alto'], REJ['ancho']), dtype=int)
        for col, fila, ancho, alto in mra.bloques(REJ, lado):
            assert 0 < ancho <= lado and 0 < alto <= lado
            visto[fila:fila + alto, col:col + ancho] += 1
        assert (visto == 1).all(), lado
    assert mra.bloques(REJ, 256) == [(0, 0, 120, 80)]
    assert mra.minutos_estimados(REJ) > 0


def test_los_limites_en_utm_contienen_todo_el_rectangulo_geografico():
    o, s = A_GEO.transform(X0, Y0 - ALTO_M)
    e, n = A_GEO.transform(X0 + ANCHO_M, Y0)
    minx, miny, maxx, maxy = mra.limites_utm((o, s, e, n), EPSG)
    for lon, lat in ((o, s), (o, n), (e, s), (e, n), ((o + e) / 2, n), (e, (s + n) / 2)):
        x, y = A_UTM.transform(lon, lat)
        assert minx - 1e-6 <= x <= maxx + 1e-6 and miny - 1e-6 <= y <= maxy + 1e-6
    o2, s2, e2, n2 = mra.limites_lonlat(REJ)
    assert o2 <= o + 1e-9 and e2 >= e - 1e-9 and s2 <= s + 1e-9 and n2 >= n - 1e-9


# -------------------------------------------------------- predicción de un bloque
def test_un_pixel_sin_alguna_variable_queda_sin_dato_en_todos_los_rasteres():
    datos = lector(rej=REJ)(0, 0, 6, 4)
    datos[1, 2, 3] = np.nan                                 # le falta UNA variable
    permitido = np.ones((4, 6), dtype=bool)
    permitido[3, 5] = False
    clase, pmax, prob, pborr = mra.predecir_bloque(MOD['modelo'], datos, permitido)
    assert clase.dtype == np.uint8 and clase.shape == (4, 6) and prob.shape == (3, 4, 6)
    assert pborr.dtype == np.float32 and pborr.shape == (4, 6)
    for f, c in ((1, 2), (3, 5)):
        assert clase[f, c] == mra.SIN_DATO_CLASE and np.isnan(pmax[f, c])
        assert np.isnan(prob[:, f, c]).all() and np.isnan(pborr[f, c])
    ok = clase != mra.SIN_DATO_CLASE
    assert ok.sum() == 22 and set(np.unique(clase[ok])) <= {0, 1, 2}
    assert np.allclose(prob[:, ok].sum(axis=0), 1, atol=1e-6)
    assert np.allclose(pmax[ok], prob[:, ok].max(axis=0))       # sin regla: la más probable
    assert np.allclose(pborr[ok], 1 - prob[0][ok], atol=1e-6)   # lo que no es «no borreguil»
    # y es lo mismo que predecir esos píxeles uno a uno
    X = datos.reshape(-1, 4)[ok.reshape(-1)]
    assert (clase[ok] == MOD['modelo'].predict(X)).all()


def test_la_prediccion_es_repetible_aunque_el_modelo_venga_con_varios_hilos():
    modelo = MOD['modelo']
    assert modelo.n_jobs == 1                    # así sale de entrenar_y_validar
    datos = lector(rej=REJ)(0, 0, 60, 40)
    ref = mra.predecir_bloque(modelo, datos)
    X = datos.reshape(-1, 4)
    assert np.array_equal(mra.proba_repetible(modelo, X), modelo.predict_proba(X))
    # Un modelo que llegue con varios hilos se predice igualmente con uno, y después
    # queda como estaba.
    modelo.n_jobs = -1
    try:
        for _ in range(5):
            otra = mra.predecir_bloque(modelo, datos)
            assert all(np.array_equal(a, b, equal_nan=True) for a, b in zip(ref, otra))
            assert modelo.n_jobs == -1
        # por trozos, lo mismo que de una vez
        mitad = mra.predecir_bloque(modelo, datos[:, :30])
        assert np.array_equal(mitad[0], ref[0][:, :30]) and np.array_equal(mitad[1], ref[1][:, :30])
    finally:
        modelo.n_jobs = 1


def test_no_predice_si_el_bloque_no_trae_las_variables_del_modelo():
    try:
        mra.predecir_bloque(MOD['modelo'], np.zeros((2, 2, 3), dtype='float32'))
    except ValueError as e:
        assert '3 variables' in str(e) and '4' in str(e)
    else:
        raise AssertionError('debía negarse')


# ------------------------------------------------------------------ el mapa entero
def test_el_mapa_dice_en_cada_muestra_lo_que_el_modelo_predice_en_su_pixel():
    d = carpeta()
    avances = []
    meta = mra.hacer_mapa(MOD, REJ, lector(rej=REJ), d, lado=50,
                          progreso=lambda a, b: avances.append((a, b)),
                          extra={'anios': [2024]})
    assert avances[-1] == (6, 6) and len(avances) == 6 and meta['bloques'] == 6
    assert meta['anios'] == [2024] and meta['predictores'] == ORDEN
    with abrir(d / 'clase.tif') as f:
        assert f.epsg == EPSG and (f.ancho, f.alto) == (120, 80)
        assert (f.escala, f.x0, f.y0) == (10, X0, Y0)
        assert f.sin_dato == mra.SIN_DATO_CLASE and f.tipo == 'uint8'
    # con la regla del modelo: borreguil si la suma de sus tipos llega al umbral
    assert meta['regla'] == {'criterio': 'f1', 'umbral': MOD['regla']['umbral']}
    proba = MOD['modelo'].predict_proba(TAB['X'])
    elegida, p_elegida, p_borr = mr.decidir(proba, MOD['modelo'].classes_,
                                            MOD['regla']['umbral'])
    for fila, p, cat, pc, pb in zip(TAB['filas'], proba, elegida, p_elegida, p_borr):
        q = mra.consultar(d, fila['lon'], fila['lat'])
        assert q is not None and q['codigo'] == int(cat)
        assert q['categoria'] == {0: 'no borreguil', 1: 'húmedo', 2: 'seco'}[q['codigo']]
        assert q['pmax'] == float(pc) and q['p_borreguil'] == float(pb)
        assert abs(q['p_borreguil'] - (p[1] + p[2])) < 1e-6
        assert [x['categoria'] for x in q['probabilidades']] == ['no borreguil', 'húmedo', 'seco']
        assert np.allclose([x['probabilidad'] for x in q['probabilidades']], p, atol=1e-6)
        # el centro del píxel consultado es el del píxel que contiene a la muestra
        assert (q['x'], q['y']) == centro_de(fila['lon'], fila['lat'])
    limpiar(d)


def test_el_resultado_no_depende_de_como_se_trocee():
    a, b = carpeta(), carpeta()
    ma = mra.hacer_mapa(MOD, REJ, lector(rej=REJ), a, lado=256)
    mb = mra.hacer_mapa(MOD, REJ, lector(rej=REJ), b, lado=37)
    assert ma['bloques'] == 1 and mb['bloques'] == 12
    for nombre in ('clase.tif', 'pmax.tif', 'probabilidades.tif',
                   'probabilidad_borreguil.tif'):
        assert np.array_equal(leer(a / nombre), leer(b / nombre), equal_nan=True), nombre
    assert ma['categorias'] == mb['categorias']
    limpiar(a)
    limpiar(b)


# ------------------------------------------- la pureza dentro del píxel, en el mapa
# El modelo de estas pruebas no usa ni el NDVI de inicio de verano ni el NDWI medio, que
# son los que hacen falta para la pureza: van detrás de sus variables en cada bloque.
BANDAS_PUREZA = list(ORDEN) + [mr.VARIABLE_VERDE, mr.VARIABLE_AGUA]
LAGUNA = (slice(30, 40), slice(50, 60))          # filas y columnas con agua
PUREZA = {'sin_verde': 0.10, 'todo_verde': 0.60, 'puro_desde': 0.75}


def _verde_y_agua():
    """El NDVI crece de norte a sur (de 0.05 a 0.84: el verde llega al 75 % en la fila
    43) y hay una laguna de 10 × 10 píxeles, justo en el borde de un bloque de 50."""
    ndvi = np.repeat((0.05 + 0.01 * np.arange(REJ['alto'], dtype='float32'))[:, None],
                     REJ['ancho'], axis=1)
    ndwi = np.full((REJ['alto'], REJ['ancho']), -0.3, dtype='float32')
    ndwi[LAGUNA] = 0.2
    return ndvi, ndwi


def lector_con_pureza(hueco_verde=None):
    base = lector(rej=REJ)
    ndvi, ndwi = _verde_y_agua()

    def leer(col, fila, ancho, alto):
        out = np.empty((alto, ancho, len(BANDAS_PUREZA)), dtype='float32')
        out[..., :len(ORDEN)] = base(col, fila, ancho, alto)
        out[..., -2] = ndvi[fila:fila + alto, col:col + ancho]
        out[..., -1] = ndwi[fila:fila + alto, col:col + ancho]
        if hueco_verde:
            for f in range(alto):
                for c in range(ancho):
                    if hueco_verde(col + c, fila + f):
                        out[f, c, -2] = np.nan
        return out
    return leer


def test_el_mapa_dice_si_cada_pixel_de_borreguil_es_puro_o_esta_mezclado():
    d = carpeta()
    meta = mra.hacer_mapa(MOD, REJ, lector_con_pureza(), d, lado=50, bandas=BANDAS_PUREZA,
                          pureza=PUREZA)
    assert meta['ficheros']['verde'] == 'verde_estimado.tif'
    assert meta['ficheros']['pureza'] == 'pureza.tif'
    clase, verde, pur = (leer(d / n, 1) for n in ('clase.tif', 'verde_estimado.tif', 'pureza.tif'))
    ndvi, ndwi = _verde_y_agua()
    # el verde de cada píxel: su NDVI entre el de «sin verde» y el de «todo verde»
    assert verde.dtype == np.float32 and np.array_equal(verde, mr.verde_estimado(ndvi, 0.10, 0.60))
    assert verde[0, 0] == 0 and verde[-1, 0] == 1 and abs(verde[30, 0] - 0.5) < 1e-6
    # la pureza, solo donde el modelo dice borreguil: puro si el verde llega al corte;
    # si no, mixto-agua si es agua o la toca, y mixto-roca en otro caso
    borreguil = clase != 0
    assert borreguil.sum() > 3000 and (~borreguil).sum() > 1000
    assert (pur[~borreguil] == mra.SIN_DATO_CLASE).all()
    toca = np.zeros_like(borreguil)
    toca[29:41, 49:61] = True                          # la laguna y los píxeles de alrededor
    assert np.array_equal(pur[borreguil], mr.pureza_de(verde, toca, 0.75)[borreguil])
    assert set(np.unique(pur[borreguil])) == {1, 2, 3}
    assert (pur[borreguil & (verde >= 0.75)] == 1).all()            # puro
    assert (pur[29:41, 49:61][borreguil[29:41, 49:61]] == 2).all()  # mixto-agua
    assert (pur[35, 49], pur[35, 48]) == (2, 3)        # toca la laguna desde el otro bloque
    assert (pur[29, 49], pur[28, 49]) == (2, 3)        # también en diagonal; más lejos, no
    # las cuentas y lo que se usó para calcularla quedan en la ficha
    pz = meta['pureza']
    assert {k: pz[k] for k in PUREZA} == PUREZA
    assert (pz['variable_verde'], pz['variable_agua']) == ('ndvi_early', 'ndwi_mean')
    assert [(c['codigo'], c['categoria']) for c in pz['categorias']] == [
        (1, 'puro'), (2, 'mixto-agua'), (3, 'mixto-roca')]
    for c in pz['categorias']:
        assert c['pixeles'] == int((pur == c['codigo']).sum())
        assert c['hectareas'] == round(c['pixeles'] / 100, 2) and c['color'] == mra.COLORES[c['categoria']]
    assert sum(c['pixeles'] for c in pz['categorias']) == int(borreguil.sum())
    with abrir(d / 'pureza.tif') as f:
        assert f.sin_dato == mra.SIN_DATO_CLASE and f.tipo == 'uint8'
        assert f.etiquetas['CATEGORIAS'] == '1 = puro; 2 = mixto-agua; 3 = mixto-roca'
        assert f.etiquetas['PURO_DESDE'] == '0.75' and f.paleta[3][:3] == (0x8C, 0x6D, 0x31)
    with abrir(d / 'verde_estimado.tif') as f:
        assert f.tipo == 'float32' and np.isnan(f.sin_dato) and 'ndvi_early' in f.nombres[0]
    # al consultar un píxel: su verde y, si es borreguil, su pureza
    f_b, c_b = np.argwhere(borreguil & (pur == 3))[0]
    q = mra.consultar(d, *A_GEO.transform(X0 + c_b * 10 + 5, Y0 - f_b * 10 - 5))
    assert q['pureza'] == 'mixto-roca' and q['verde'] == float(verde[f_b, c_b])
    f_n, c_n = np.argwhere(~borreguil)[0]
    q = mra.consultar(d, *A_GEO.transform(X0 + c_n * 10 + 5, Y0 - f_n * 10 - 5))
    assert q['pureza'] is None and q['verde'] == float(verde[f_n, c_n]) and q['codigo'] == 0
    # troceado de otra manera sale lo mismo; y las demás capas no cambian por pedirla
    d2, d3 = carpeta(), carpeta()
    mra.hacer_mapa(MOD, REJ, lector_con_pureza(), d2, lado=256, bandas=BANDAS_PUREZA, pureza=PUREZA)
    for n in ('pureza.tif', 'verde_estimado.tif', 'clase.tif'):
        assert np.array_equal(leer(d / n), leer(d2 / n), equal_nan=True), n
    m3 = mra.hacer_mapa(MOD, REJ, lector(rej=REJ), d3)
    assert np.array_equal(leer(d3 / 'clase.tif'), leer(d / 'clase.tif'))
    assert 'pureza' not in m3 and 'verde' not in m3['ficheros'] and 'pureza' not in m3['ficheros']
    assert not (d3 / 'pureza.tif').exists() and not (d3 / 'verde_estimado.tif').exists()
    q = mra.consultar(d3, *A_GEO.transform(X0 + c_b * 10 + 5, Y0 - f_b * 10 - 5))
    assert q['pureza'] is None and q['verde'] is None
    for x in (d, d2, d3):
        limpiar(x)


# ------------------------------------------------- el agua de laguna no es borreguil
BANDAS_AGUA = list(ORDEN) + [ma.VARIABLE_AGUA]       # la pendiente ya va en ORDEN


def lector_con_agua():
    """El «mundo», con el NDWI medio de la laguna de _verde_y_agua (la pendiente del
    mundo es de menos de 1°: todo es llano)."""
    base = lector(rej=REJ)
    _, ndwi = _verde_y_agua()

    def leer(col, fila, ancho, alto):
        out = np.empty((alto, ancho, len(BANDAS_AGUA)), dtype='float32')
        out[..., :len(ORDEN)] = base(col, fila, ancho, alto)
        out[..., -1] = ndwi[fila:fila + alto, col:col + ancho]
        return out
    return leer


def test_el_agua_de_laguna_pasa_a_no_borreguil():
    d, d0 = carpeta(), carpeta()
    meta = mra.hacer_mapa(MOD, REJ, lector_con_agua(), d, lado=50, bandas=BANDAS_AGUA,
                          quitar_agua=True)
    m0 = mra.hacer_mapa(MOD, REJ, lector_con_agua(), d0, lado=50, bandas=BANDAS_AGUA)
    antes, clase = leer(d0 / 'clase.tif', 1), leer(d / 'clase.tif', 1)
    laguna = np.zeros(antes.shape, bool)
    laguna[LAGUNA] = True
    # en la laguna había borreguil, y ahora no; fuera de ella no cambia nada
    assert (antes[laguna] != 0).sum() > 50
    assert (clase[laguna] == 0).all() and np.array_equal(clase[~laguna], antes[~laguna])
    # la probabilidad de lo elegido pasa a ser la de «no borreguil»; las del modelo, igual
    prob = leer(d / 'probabilidades.tif')
    pmax, pmax0 = leer(d / 'pmax.tif', 1), leer(d0 / 'pmax.tif', 1)
    quitados = laguna & (antes != 0)
    assert np.array_equal(pmax[quitados], prob[0][quitados])
    assert np.array_equal(pmax[~quitados], pmax0[~quitados])
    for n in ('probabilidades.tif', 'probabilidad_borreguil.tif'):
        assert np.array_equal(leer(d / n), leer(d0 / n), equal_nan=True), n
    # el agua, en su fichero; las cuentas, ya sin ella
    assert np.array_equal(leer(d / 'agua_de_laguna.tif', 1), laguna.astype('uint8'))
    assert meta['ficheros']['agua'] == 'agua_de_laguna.tif' and 'agua' not in m0['ficheros']
    ma_ = meta['mascara_agua']
    assert (ma_['pixeles_agua'], ma_['pixeles_quitados']) == (100, int(quitados.sum()))
    assert ma_['hectareas_quitadas'] == round(quitados.sum() / 100, 2)
    assert {c['codigo']: c['pixeles'] for c in meta['categorias']} == {
        c: int((clase == c).sum()) for c in (0, 1, 2)}
    with abrir(d / 'clase.tif') as f:
        assert 'NDWI medio > 0.05' in f.etiquetas['MASCARA_AGUA']
    # un modelo sin «no borreguil» no tiene a qué pasar el agua: no se toca
    sin_aus = entrenado(con_ausencia=False)[0]
    d1 = carpeta()
    m1 = mra.hacer_mapa(sin_aus, REJ, lector_con_agua(), d1, bandas=BANDAS_AGUA,
                        quitar_agua=True)
    assert 'mascara_agua' not in m1 and not (d1 / 'agua_de_laguna.tif').exists()
    # sin el NDWI medio no se puede: se dice
    d2 = carpeta()
    try:
        mra.hacer_mapa(MOD, REJ, lector(rej=REJ), d2, quitar_agua=True)
    except ValueError as e:
        assert 'agua de laguna' in str(e) and 'ndwi_mean' in str(e), str(e)
    else:
        raise AssertionError('debía pedir el NDWI medio')
    for x in (d, d0, d1, d2):
        limpiar(x)


# ------------------------------------------- en dos pasos: sí o no, y después el tipo
PRED_SI_NO = ['ndmi_mean', 'albedo_mean']


def entrenado_si_no():
    """El modelo de sí o no con los mismos sitios que entrenado(): lo del este no es
    borreguil y el resto sí, dicho en campo y sin el tipo revisado."""
    from test_mapeo_datos import punto
    rows, valores = [], {}
    for k in range(60):
        x = X0 + 17.3 + (k * 197) % int(ANCHO_M - 30)
        y = Y0 - 13.7 - (k * 131) % int(ALTO_M - 30)
        lon, lat = A_GEO.transform(x, y)
        r = (ausencia(k, lon=lon, lat=lat) if (x - X0) / ANCHO_M > 2 / 3 else
             punto(k, 'BORREGUIL VERIFICADO', Borreguil='si', lon=lon, lat=lat))
        r['cuenca_id'] = f'c{k % 5}'
        rows.append(r)
        valores[mr.clave_punto(lon, lat)] = mundo(*centro_de(lon, lat))
    res = mr.muestras(rows, mr.NIVEL_SI_NO, es_borreguil=es_borreguil)
    assert res['listo'] and res['excluidos'] == {}
    tab = mr.construir_tabla(res, valores, PRED_SI_NO)
    return mr.entrenar_y_validar(tab, mr.grupos_espaciales(tab, rows)[0], {'n_estimators': 40})


def test_el_mapa_en_dos_pasos_decide_el_si_con_un_modelo_y_el_tipo_con_otro():
    si_no, tipo = entrenado_si_no(), entrenado(con_ausencia=False)[0]
    assert si_no['nivel'] == 'borreguil' and si_no['regla']['criterio'] == 'f1'
    assert tipo['regla']['umbral'] is None                 # sin «no borreguil»
    comb = mr.en_dos_pasos(si_no, tipo)
    pred = comb['predictores']
    assert pred == mr.ordenar_predictores(PRED_SI_NO + PRED) and len(pred) == 5
    assert comb['nivel'] == 'humedad' and comb['regla']['umbral'] == si_no['regla']['umbral']
    assert [(c['codigo'], c['categoria']) for c in comb['categorias']] == [
        (0, 'no borreguil'), (1, 'húmedo'), (2, 'seco')]
    assert comb['dos_pasos']['si_no']['firma'] == si_no['firma']
    d = carpeta()
    meta = mra.hacer_mapa(comb, REJ, lector(orden=pred, rej=REJ), d, lado=50)
    assert meta['regla']['umbral'] == si_no['regla']['umbral']
    clase, pborr = leer(d / 'clase.tif', 1), leer(d / 'probabilidad_borreguil.tif', 1)
    X = lector(orden=pred, rej=REJ)(0, 0, REJ['ancho'], REJ['alto']).reshape(-1, len(pred))
    p_si = si_no['modelo'].predict_proba(X[:, [pred.index(v) for v in si_no['predictores']]])[:, 1]
    p_tipo = tipo['modelo'].predict_proba(X[:, [pred.index(v) for v in tipo['predictores']]])
    # la probabilidad de borreguil es la del modelo de sí o no; el tipo, el del otro
    assert np.allclose(pborr.reshape(-1), p_si, atol=1e-6)
    es = clase.reshape(-1) != 0
    assert np.array_equal(es, pborr.reshape(-1) >= np.float32(si_no['regla']['umbral']))
    assert 0 < es.sum() < es.size
    assert np.array_equal(clase.reshape(-1)[es], tipo['modelo'].classes_[p_tipo.argmax(1)][es])
    limpiar(d)
    # no se juntan al revés, ni con un modelo del tipo que ya sabe decir que no
    for a, b, trozo in ((tipo, si_no, 'sí o no'), (si_no, MOD, 'no borreguil'),
                        (si_no, si_no, 'nivel del tipo')):
        try:
            mr.en_dos_pasos(a, b)
        except ValueError as e:
            assert trozo in str(e), str(e)
        else:
            raise AssertionError('no debían juntarse')


def test_la_pureza_necesita_sus_variables_y_no_inventa_donde_faltan():
    # sin el NDVI de inicio de verano o el NDWI medio no se puede calcular: se dice
    for bandas, trozo in ((None, 'ndvi_early, ndwi_mean'),
                          (list(ORDEN) + [mr.VARIABLE_AGUA], 'ndvi_early')):
        d = carpeta()
        try:
            mra.hacer_mapa(MOD, REJ, lector(rej=REJ), d, bandas=bandas, pureza=PUREZA)
        except ValueError as e:
            assert 'hacen falta también' in str(e) and trozo in str(e), str(e)
        else:
            raise AssertionError('debía pedir las variables de la pureza')
        limpiar(d)
    # las bandas de los bloques empiezan por las variables del modelo, en su orden
    d = carpeta()
    try:
        mra.hacer_mapa(MOD, REJ, lector(rej=REJ), d, bandas=list(ORDEN)[::-1])
    except ValueError as e:
        assert 'en su orden' in str(e)
    else:
        raise AssertionError('debía rechazar unas bandas en otro orden')
    # valores sin sentido
    for malo in ({'sin_verde': 0.6, 'todo_verde': 0.5}, {'puro_desde': 0}, {'todo_verde': 'x'}):
        try:
            mra.hacer_mapa(MOD, REJ, lector_con_pureza(), d, bandas=BANDAS_PUREZA, pureza=malo)
        except ValueError:
            pass
        else:
            raise AssertionError(f'debía rechazar {malo}')
    assert not list(d.glob('*'))
    # Donde falta el NDVI, el píxel se clasifica igual (el modelo no lo usa), pero su
    # verde y su pureza quedan sin dato: no se inventan.
    hueco = lambda c, f: 10 <= c < 14 and 50 <= f < 53                    # noqa: E731
    meta = mra.hacer_mapa(MOD, REJ, lector_con_pureza(hueco), d, bandas=BANDAS_PUREZA,
                          pureza=PUREZA)
    clase, verde, pur = (leer(d / n, 1) for n in ('clase.tif', 'verde_estimado.tif', 'pureza.tif'))
    assert meta['pixeles_sin_dato'] == 0 and (clase[50:53, 10:14] != mra.SIN_DATO_CLASE).all()
    assert np.isnan(verde[50:53, 10:14]).all() and (pur[50:53, 10:14] == mra.SIN_DATO_CLASE).all()
    assert np.isfinite(verde).sum() == 9600 - 12
    q = mra.consultar(d, *A_GEO.transform(X0 + 115, Y0 - 515))
    assert q['codigo'] is not None and q['verde'] is None and q['pureza'] is None
    limpiar(d)


def test_los_valores_de_la_pureza_y_lo_que_proponen_las_muestras():
    assert mr.params_pureza() == mr.PUREZA_DEFECTO == {
        'sin_verde': 0.08, 'todo_verde': 0.60, 'puro_desde': 0.75}
    assert mr.params_pureza({'puro_desde': 0.6, 'otra': 1})['puro_desde'] == 0.6
    v = mr.verde_estimado([0.05, 0.08, 0.34, 0.60, 0.9, float('nan')], 0.08, 0.60)
    assert v.dtype == np.float32 and list(v[:5]) == [0.0, 0.0, 0.5, 1.0, 1.0] and np.isnan(v[5])
    assert list(mr.pureza_de(v[:5], [False, True, True, False, False], 0.75)) == [3, 2, 2, 1, 1]
    # el «todo verde» que proponen las muestras puras: su mediana, si hay bastantes
    assert mr.todo_verde_de([0.5, 0.6, None, 0.7]) == (0.60, False)
    assert mr.todo_verde_de([0.50, 0.52, 0.55, 0.61, 0.70, float('nan')]) == (0.55, True)
    # cómo quedarían las muestras cuya pureza ha revisado una persona
    r = mr.comparar_pureza([{'pureza': 'puro', 'ndvi': 0.58}, {'pureza': 'puro', 'ndvi': 0.30},
                            {'pureza': 'mixto-roca', 'ndvi': 0.25},
                            {'pureza': 'mixto-agua', 'ndvi': 0.60},
                            {'pureza': 'puro', 'ndvi': None}, {'pureza': '', 'ndvi': 0.5}])
    assert r == {'n': 4, 'coinciden': 2, 'tabla': {'puro': {'puro': 1, 'mixto': 1},
                                                   'mixto-agua': {'puro': 1, 'mixto': 0},
                                                   'mixto-roca': {'puro': 0, 'mixto': 1}}}
    assert mr.comparar_pureza([{'pureza': 'puro', 'ndvi': 0.30}],
                              {'todo_verde': 0.35})['coinciden'] == 1


# --------------------------------------------- ¿borreguil o no? La regla en el mapa
def _modelo_dudoso():
    """Un modelo que duda: tres categorías que las variables no distinguen, así que la
    probabilidad se reparte y hay muchos píxeles donde «no borreguil» es la más
    probable sin llegar a la mitad."""
    from sklearn.ensemble import RandomForestClassifier
    azar = np.random.default_rng(11)
    X = azar.random((240, len(ORDEN)))
    y = azar.integers(0, 3, 240)
    m = RandomForestClassifier(n_estimators=40, min_samples_leaf=8, random_state=3).fit(X, y)
    m.n_jobs = 1
    return {'modelo': m, 'nivel': 'humedad', 'firma': 'dudoso', 'predictores': list(ORDEN),
            'categorias': [{'codigo': 0, 'categoria': 'no borreguil'},
                           {'codigo': 1, 'categoria': 'húmedo'},
                           {'codigo': 2, 'categoria': 'seco'}]}


def test_el_bloque_aplica_la_regla_de_borreguil_o_no():
    mod = _modelo_dudoso()
    datos = np.random.default_rng(4).random((30, 40, len(ORDEN))).astype('float32')
    simple, _, prob, pborr = mra.predecir_bloque(mod['modelo'], datos)
    # la probabilidad de borreguil es la suma de la de los tipos, en float32
    assert np.array_equal(pborr, prob[1] + prob[2])
    # el caso que motivó la regla existe: «no borreguil» gana sin llegar a la mitad
    perdidos = (simple == 0) & (pborr >= np.float32(0.5))
    assert perdidos.sum() > 20, int(perdidos.sum())
    tipo = np.where(prob[2] > prob[1], 2, 1)                 # si empatan, el de código más bajo
    borreguiles = []
    for u in (0.3, 0.5, 0.65, 0.8):
        clase, pmax, prob_u, pborr_u = mra.predecir_bloque(mod['modelo'], datos, None, u)
        assert np.array_equal(prob_u, prob) and np.array_equal(pborr_u, pborr)   # no cambian
        es = pborr >= np.float32(u)
        assert np.array_equal(clase, np.where(es, tipo, 0))
        assert np.array_equal(pmax, np.where(es, np.maximum(prob[1], prob[2]), prob[0]))
        borreguiles.append(int(es.sum()))
    assert borreguiles == sorted(borreguiles, reverse=True) and borreguiles[0] > borreguiles[-1]
    # con 0.5, lo que el voto simple daba por borreguil lo sigue siendo, y del mismo tipo
    con_05 = mra.predecir_bloque(mod['modelo'], datos, None, 0.5)[0]
    assert (con_05[simple != 0] == simple[simple != 0]).all()
    assert (con_05 != 0).sum() == (simple != 0).sum() + perdidos.sum()


def test_el_mapa_guarda_la_probabilidad_de_borreguil_y_el_umbral():
    mod = dict(_modelo_dudoso(), regla={'criterio': 'manual', 'umbral': 0.55})
    azar = np.random.default_rng(8)
    campo = azar.random((REJ['alto'], REJ['ancho'], len(ORDEN))).astype('float32')
    d = carpeta()
    meta = mra.hacer_mapa(mod, REJ, lambda c, f, w, h: campo[f:f + h, c:c + w], d, lado=50)
    assert meta['regla'] == {'criterio': 'manual', 'umbral': 0.55}
    assert meta['ficheros']['pborr'] == 'probabilidad_borreguil.tif'
    clase, prob = leer(d / 'clase.tif', 1), leer(d / 'probabilidades.tif')
    pborr, pmax = leer(d / 'probabilidad_borreguil.tif', 1), leer(d / 'pmax.tif', 1)
    with abrir(d / 'probabilidad_borreguil.tif') as f:
        assert f.tipo == 'float32' and np.isnan(f.sin_dato)
        assert f.nombres[0] == 'probabilidad de borreguil (suma de los tipos)'
        assert (f.ancho, f.alto) == (120, 80) and f.x0 == X0
    with abrir(d / 'clase.tif') as f:
        assert f.etiquetas['UMBRAL_BORREGUIL'] == '0.55'
    # Quien lea los ficheros y aplique el mismo umbral obtiene EXACTAMENTE el mapa:
    assert np.array_equal(pborr, prob[1] + prob[2])
    es = pborr >= np.float32(0.55)
    assert np.array_equal(clase, np.where(es, np.where(prob[2] > prob[1], 2, 1), 0))
    assert np.array_equal(pmax, np.where(es, np.maximum(prob[1], prob[2]), prob[0]))
    assert 0 < es.sum() < es.size                  # hay de las dos cosas
    por_codigo = {c['codigo']: c['pixeles'] for c in meta['categorias']}
    assert por_codigo[0] == int((~es).sum()) and por_codigo[1] + por_codigo[2] == int(es.sum())
    q = mra.consultar(d, *A_GEO.transform(X0 + 305, Y0 - 205))
    assert q['p_borreguil'] == float(pborr[20, 30]) and q['codigo'] == int(clase[20, 30])
    assert q['pmax'] == float(pmax[20, 30])
    # otro umbral, otro mapa; sin regla, la categoría más probable de todas
    d2, d3 = carpeta(), carpeta()
    m2 = mra.hacer_mapa(dict(mod, regla={'criterio': 'manual', 'umbral': 0.3}), REJ,
                        lambda c, f, w, h: campo[f:f + h, c:c + w], d2)
    assert np.array_equal(leer(d2 / 'clase.tif', 1) != 0, pborr >= np.float32(0.3))
    assert m2['categorias'][0]['pixeles'] < por_codigo[0]
    m3 = mra.hacer_mapa(_modelo_dudoso(), REJ, lambda c, f, w, h: campo[f:f + h, c:c + w], d3)
    assert m3['regla'] == {'criterio': None, 'umbral': None}
    assert np.array_equal(leer(d3 / 'clase.tif', 1), prob.argmax(axis=0))
    with abrir(d3 / 'clase.tif') as f:
        assert 'UMBRAL_BORREGUIL' not in f.etiquetas
    for x in (d, d2, d3):
        limpiar(x)


# ----------------------- nivel combinado: el ambiente por cercanía a lagunas y cauces
PRED_AMB = mr.ordenar_predictores(['ndvi_late', 'ndmi_mean'])
# detrás de las variables del modelo, las que solo hacen falta para el ambiente
BANDAS_AMB = PRED_AMB + [ma.VARIABLE_AGUA, ma.VARIABLE_PENDIENTE, ma.VARIABLE_RELIEVE]
EJE = 60                                            # columna por la que baja la vaguada
LAGO = (slice(10, 16), slice(20, 28))               # filas y columnas con agua que se ve
LAGUNILLO = (90, 60)                                # (columna, fila) de una laguna de la capa


def _modelo_de_humedad():
    """Un modelo del nivel «humedad» que duda (sus variables no separan nada): por el
    campo hay de todo, borreguil húmedo, seco y «no borreguil»."""
    from sklearn.ensemble import RandomForestClassifier
    azar = np.random.default_rng(11)
    X = azar.random((240, len(PRED_AMB)))
    y = azar.integers(0, 3, 240)
    m = RandomForestClassifier(n_estimators=40, min_samples_leaf=8, random_state=3).fit(X, y)
    m.n_jobs = 1
    return {'modelo': m, 'nivel': 'humedad', 'firma': 'humedad', 'predictores': list(PRED_AMB),
            'regla': {'criterio': 'manual', 'umbral': 0.5},
            'categorias': [{'codigo': 0, 'categoria': 'no borreguil'},
                           {'codigo': 1, 'categoria': 'húmedo'},
                           {'codigo': 2, 'categoria': 'seco'}]}


def _mundo_con_ambiente(bandas=BANDAS_AMB):
    """Un campo de 120 × 80 píxeles: una vaguada que baja hacia el sur por la columna
    60, una laguna que se ve (NDWI alto, en llano) y, aparte, un lagunillo que no se
    ve y solo está en la capa de lagunas. Lo demás es ruido. Devuelve el campo y los
    puntos de la capa."""
    azar = np.random.default_rng(8)
    campo = azar.random((REJ['alto'], REJ['ancho'], len(bandas))).astype('float32')
    ff, cc = np.mgrid[0:REJ['alto'], 0:REJ['ancho']]
    if ma.VARIABLE_RELIEVE in bandas:
        campo[..., bandas.index(ma.VARIABLE_RELIEVE)] = 3000.0 - 2.0 * ff + 3.0 * np.abs(cc - EJE)
    campo[..., bandas.index(ma.VARIABLE_PENDIENTE)] = 6.0
    campo[..., bandas.index(ma.VARIABLE_AGUA)] = -0.2
    campo[LAGO + (bandas.index(ma.VARIABLE_AGUA),)] = 0.3
    x, y = mra.esquina(REJ, *LAGUNILLO)
    return campo, [(x + 5.0, y - 5.0)]                # el centro de ese píxel


def _relieve_con_margen(pedidos=None):
    """El mismo relieve del campo, que sigue fuera de la zona: lo que daría la app."""
    def relieve(margen):
        if pedidos is not None:
            pedidos.append(margen)
        ff, cc = np.mgrid[-margen:REJ['alto'] + margen, -margen:REJ['ancho'] + margen]
        return 3000.0 - 2.0 * ff + 3.0 * np.abs(cc - EJE)
    return relieve


def test_el_mapa_pone_el_ambiente_por_cercania_y_compone_la_categoria():
    campo, lagunas = _mundo_con_ambiente()
    par = {'cauce_ha': 1.0}
    d = carpeta()
    meta = mra.hacer_mapa(
        _modelo_de_humedad(), REJ, lambda c, f, w, h: campo[f:f + h, c:c + w], d, lado=50,
        bandas=BANDAS_AMB,
        ambiente={'nivel': 'ambiente_humedad', 'params': par, 'lagunas': lagunas,
                  'fuentes': {'lagunas': 'capa de prueba', 'cauces': 'relieve'}})
    clase, prob = leer(d / 'clase.tif', 1), leer(d / 'probabilidades.tif')
    amb, dist = leer(d / 'ambiente.tif', 1), leer(d / 'distancias_ambiente.tif')
    refs = leer(d / 'lagunas_y_cauces.tif', 1)
    L, A, D = ma.CODIGOS['laguna'], ma.CODIGOS['arroyo'], ma.CODIGOS['ladera']

    # 1) Lo que se ha tomado por laguna y por cauce.
    agua = np.zeros(clase.shape, bool)
    agua[LAGO] = True
    assert np.array_equal(refs == 1, agua)
    assert refs[LAGUNILLO[1], LAGUNILLO[0]] == 3 and (refs == 3).sum() == 1
    cauce = refs == 2
    assert cauce[40:, EJE].all() and not cauce[:5, EJE].any()
    assert not np.delete(cauce, [EJE - 1, EJE, EJE + 1], axis=1).any()

    # 2) Las distancias son a eso (NaN = no hay nada cerca).
    sin_nan = lambda m: np.nan_to_num(m, nan=np.inf)                    # noqa: E731
    assert np.array_equal(sin_nan(dist[0]), ma.distancia(agua))
    assert np.array_equal(sin_nan(dist[1]), ma.distancia_a_puntos(lagunas, REJ))
    assert np.array_equal(sin_nan(dist[2]), ma.distancia(cauce))

    # 3) El ambiente sale de esas distancias con la regla: quien lea los ficheros y la
    #    aplique obtiene EXACTAMENTE el mismo.
    assert np.array_equal(amb, ma.decidir(dist[0], dist[1], dist[2], par))
    assert amb[12, 24] == L and amb[12, 35] == L and amb[12, 37] == L     # ≤ 100 m del agua
    assert amb[12, 38] == D                                               # a 110 m, ya no
    assert amb[60, 90] == L and amb[60, 102] == L and amb[60, 103] == D   # 120 m de la capa
    assert amb[60, EJE + 3] == A and amb[60, EJE + 4] == D                # 30 m del cauce
    assert amb[60, EJE] == A

    # 4) La categoría es la compuesta: ese ambiente con lo que dice el modelo.
    pborr = prob[1] + prob[2]
    tipo = np.where(pborr >= np.float32(0.5), np.where(prob[2] > prob[1], 2, 1), 0)
    assert np.array_equal(clase, mr.componer('ambiente_humedad', amb, tipo))
    assert sorted(np.unique(clase)) == list(range(7))                     # hay de todo

    # 5) Los metadatos lo cuentan.
    assert [c['categoria'] for c in meta['categorias']] == \
        ['no borreguil'] + list(mr.NIVELES['ambiente_humedad'])
    assert [c['codigo'] for c in meta['categorias']] == list(range(7))
    assert [c['pixeles'] for c in meta['categorias']] == [int((clase == c).sum()) for c in range(7)]
    assert [c['color'] for c in meta['categorias']] == \
        [mra.color(c['categoria']) for c in meta['categorias']]
    assert [c['categoria'] for c in meta['categorias_modelo']] == ['no borreguil', 'húmedo', 'seco']
    a = meta['ambiente']
    assert a['nivel'] == 'ambiente_humedad' and a['cauce_ha'] == 1.0
    assert (a['laguna_capa_m'], a['laguna_agua_m'], a['arroyo_m']) == (120.0, 100.0, 30.0)
    assert a['cauces'] == 'relieve' and a['margen_m'] == 0
    assert a['lagunas_capa'] == 1 and a['lagunas_capa_en_el_mapa'] == 1
    assert a['pixeles_agua'] == 48 and a['pixeles_cauce'] == int(cauce.sum())
    assert a['fuentes'] == {'lagunas': 'capa de prueba', 'cauces': 'relieve'}
    assert a['regla'] == ma.frase(par)
    borr = clase != 0
    assert [(r['rama'], r['pixeles']) for r in meta['ramas']] == \
        [(n, int((amb[borr] == c).sum())) for n, c in ma.CODIGOS.items()]
    assert sum(r['pixeles'] for r in meta['ramas']) == int(borr.sum()) > 0
    assert all(r['hectareas'] == round(r['pixeles'] / 100, 2) for r in meta['ramas'])
    assert {'ambiente', 'distancias', 'referencias'} <= set(meta['ficheros'])
    assert json.loads((d / 'mapa.json').read_text(encoding='utf-8'))['ambiente'] == a
    with abrir(d / 'clase.tif') as f:
        assert f.etiquetas['AMBIENTE_POR_CERCANIA'] == ma.frase(par)
        assert f.etiquetas['CATEGORIAS'].startswith(
            '0 = no borreguil; 1 = arroyo · húmedo; 2 = arroyo · seco; 3 = laguna · húmedo')
    with abrir(d / 'probabilidades.tif') as f:       # las del modelo, tal cual
        assert f.nombres == ('P(no borreguil)', 'P(húmedo)', 'P(seco)')
    with abrir(d / 'ambiente.tif') as f:
        assert f.sin_dato == 255 and f.etiquetas['CATEGORIAS'] == '1 = arroyo; 2 = laguna; 3 = ladera'
        assert f.etiquetas['ARROYO_M'] == '30.0' and (f.ancho, f.alto) == (120, 80)
    with abrir(d / 'distancias_ambiente.tif') as f:
        assert f.bandas == 3 and f.tipo == 'float32' and 'cauce' in f.nombres[2]
    with abrir(d / 'lagunas_y_cauces.tif') as f:
        assert f.sin_dato == 0 and f.etiquetas['CATEGORIAS'].startswith('1 = agua de laguna')

    # 6) La consulta de un píxel dice su ambiente y por qué.
    q = mra.consultar(d, *A_GEO.transform(X0 + 355, Y0 - 125))            # columna 35, fila 12
    assert q['ambiente'] == 'laguna' and abs(q['distancias']['agua'] - 80) < 1e-3
    assert abs(q['distancias']['capa'] - 10 * np.hypot(55, 48)) < 1e-2
    assert abs(q['distancias']['cauce'] - 250) < 1e-3
    assert [x['categoria'] for x in q['probabilidades']] == ['no borreguil', 'húmedo', 'seco']
    assert abs(sum(x['probabilidad'] for x in q['probabilidades']) - 1) < 1e-5
    assert q['codigo'] == 0 or q['categoria'].startswith('laguna · ')
    lejos = mra.consultar(d, *A_GEO.transform(X0 + 5, Y0 - 5))            # la esquina de arriba
    assert lejos['ambiente'] == 'ladera' and lejos['distancias']['capa'] is None
    limpiar(d)


def _modelo_si_no_que_duda():
    """Un modelo de borreguil sí o no que duda (sus variables no separan nada): por el
    campo sale borreguil y «no borreguil» un poco de cada."""
    from sklearn.ensemble import RandomForestClassifier
    azar = np.random.default_rng(12)
    X = azar.random((240, len(PRED_AMB)))
    y = azar.integers(0, 2, 240)
    m = RandomForestClassifier(n_estimators=40, min_samples_leaf=8, random_state=3).fit(X, y)
    m.n_jobs = 1
    return {'modelo': m, 'nivel': mr.NIVEL_SI_NO, 'firma': 'si_no',
            'predictores': list(PRED_AMB), 'regla': {'criterio': 'f1', 'umbral': 0.5},
            'categorias': [{'codigo': 0, 'categoria': 'no borreguil'},
                           {'codigo': 1, 'categoria': 'borreguil'}]}


def test_el_mapa_de_si_o_no_pone_el_ambiente_de_cada_pixel_de_borreguil():
    # Sin ningún tipo revisado: el modelo dice si hay borreguil y la cercanía, dónde.
    campo, lagunas = _mundo_con_ambiente()
    par = {'cauce_ha': 1.0}
    mod = _modelo_si_no_que_duda()
    leer_campo = lambda c, f, w, h: campo[f:f + h, c:c + w]               # noqa: E731
    d, d0 = carpeta(), carpeta()
    meta = mra.hacer_mapa(mod, REJ, leer_campo, d, lado=50, bandas=BANDAS_AMB,
                          ambiente={'nivel': mr.NIVEL_SI_NO, 'params': par, 'lagunas': lagunas})
    m0 = mra.hacer_mapa(mod, REJ, leer_campo, d0, lado=50, bandas=BANDAS_AMB)
    clase, clase0 = leer(d / 'clase.tif', 1), leer(d0 / 'clase.tif', 1)
    amb = leer(d / 'ambiente.tif', 1)
    # Lo que es borreguil no cambia; cada píxel de borreguil queda con su ambiente.
    assert np.array_equal(clase != 0, clase0 != 0) and 0 < (clase0 != 0).sum() < clase.size
    assert np.array_equal(clase, np.where(clase0 != 0, amb, 0))
    assert sorted(np.unique(clase)) == [0, 1, 2, 3]                       # hay de todo
    # Las categorías y los códigos son los del nivel «ambiente»; las probabilidades,
    # las del modelo (no borreguil, borreguil), tal cual.
    assert [(c['codigo'], c['categoria']) for c in meta['categorias']] == [
        (0, 'no borreguil'), (1, 'arroyo'), (2, 'laguna'), (3, 'ladera')]
    assert {c: n for n, c in mr.codigos('ambiente').items()} == {
        c['codigo']: c['categoria'] for c in meta['categorias']}
    assert [c['pixeles'] for c in meta['categorias']] == [int((clase == c).sum()) for c in range(4)]
    assert [c['color'] for c in meta['categorias']] == \
        [mra.color(c['categoria']) for c in meta['categorias']]
    assert [c['categoria'] for c in meta['categorias_modelo']] == ['no borreguil', 'borreguil']
    for n in ('probabilidades.tif', 'probabilidad_borreguil.tif', 'pmax.tif'):
        assert np.array_equal(leer(d / n), leer(d0 / n), equal_nan=True), n
    assert meta['ambiente']['nivel'] == mr.NIVEL_SI_NO
    assert [(r['rama'], r['pixeles']) for r in meta['ramas']] == [
        (n, int((clase == c).sum())) for n, c in ma.CODIGOS.items()]
    with abrir(d / 'clase.tif') as f:
        assert f.etiquetas['CATEGORIAS'] == \
            '0 = no borreguil; 1 = arroyo; 2 = laguna; 3 = ladera'
    q = mra.consultar(d, *A_GEO.transform(X0 + 355, Y0 - 125))            # junto al lago
    assert q['ambiente'] == 'laguna' and q['categoria'] in ('no borreguil', 'laguna')
    assert [x['categoria'] for x in q['probabilidades']] == ['no borreguil', 'borreguil']
    # Con el agua de laguna quitada, el agua pasa a «no borreguil» y el resto, igual.
    d1 = carpeta()
    m1 = mra.hacer_mapa(mod, REJ, leer_campo, d1, lado=50, bandas=BANDAS_AMB, quitar_agua=True,
                        ambiente={'nivel': mr.NIVEL_SI_NO, 'params': par, 'lagunas': lagunas})
    c1 = leer(d1 / 'clase.tif', 1)
    agua = np.zeros(c1.shape, bool)
    agua[LAGO] = True
    assert (c1[agua] == 0).all() and np.array_equal(c1[~agua], clase[~agua])
    assert m1['mascara_agua']['pixeles_quitados'] == int((clase[agua] != 0).sum()) > 0
    assert [c['pixeles'] for c in m1['categorias']] == [int((c1 == c).sum()) for c in range(4)]
    # Con el nivel de sí o no, solo vale el modelo de sí o no.
    for malo in (_modelo_de_humedad(), dict(mod, nivel='humedad')):
        d2 = carpeta()
        try:
            mra.hacer_mapa(malo, REJ, leer_campo, d2, bandas=BANDAS_AMB,
                           ambiente={'nivel': mr.NIVEL_SI_NO})
        except ValueError as e:
            assert 'sí o no' in str(e), str(e)
        else:
            raise AssertionError('debía fallar')
        limpiar(d2)
    assert 'ambiente' not in m0
    for x in (d, d0, d1):
        limpiar(x)


def test_los_cauces_salen_de_la_capa_o_del_relieve_con_su_margen():
    campo, lagunas = _mundo_con_ambiente()
    mod = _modelo_de_humedad()
    base = {'nivel': 'ambiente_humedad', 'params': {'cauce_ha': 8.0}, 'lagunas': lagunas}
    d1, d2, d3 = carpeta(), carpeta(), carpeta()
    # a) Sin más, del relieve de los bloques: el cauce empieza donde ya vierten 8 ha.
    sin = mra.hacer_mapa(mod, REJ, lambda c, f, w, h: campo[f:f + h, c:c + w], d1,
                         bandas=BANDAS_AMB, ambiente=dict(base))
    c_sin = leer(d1 / 'lagunas_y_cauces.tif', 1) == 2
    assert sin['ambiente']['margen_m'] == 0
    assert not c_sin[:5, EJE].any() and c_sin[60:, EJE].all()
    # b) Con el relieve leído con margen, el agua que entra de fuera también cuenta: la
    #    vaguada ya es cauce al entrar en la zona. (Y el relieve de los bloques sobra.)
    pedidos = []
    sin_dem = BANDAS_AMB[:-1]
    con = mra.hacer_mapa(mod, REJ, lambda c, f, w, h: campo[f:f + h, c:c + w, :len(sin_dem)],
                         d2, bandas=sin_dem,
                         ambiente=dict(base, relieve=_relieve_con_margen(pedidos)))
    c_con = leer(d2 / 'lagunas_y_cauces.tif', 1) == 2
    assert pedidos == [100] and con['ambiente']['margen_m'] == 1000
    assert con['ambiente']['cauces'] == 'relieve' and c_con[:, EJE].all()
    assert not np.delete(c_con, [EJE - 1, EJE, EJE + 1], axis=1).any()
    # c) Con una capa de cauces, mandan sus líneas, y el relieve ni se pide.
    from shapely.geometry import LineString
    xa, ya = mra.esquina(REJ, 0, 40)
    capa = LineString([(xa - 300, ya - 5), (xa + 1500, ya - 5)])     # por la fila 40
    de_capa = mra.hacer_mapa(mod, REJ, lambda c, f, w, h: campo[f:f + h, c:c + w, :len(sin_dem)],
                             d3, bandas=sin_dem,
                             ambiente=dict(base, cauces=capa, relieve=_relieve_con_margen(pedidos)))
    assert pedidos == [100] and de_capa['ambiente']['cauces'] == 'capa'
    refs, amb = leer(d3 / 'lagunas_y_cauces.tif', 1), leer(d3 / 'ambiente.tif', 1)
    assert (refs[40] == 2).all() and (np.delete(refs, 40, axis=0) != 2).all()
    A, D = ma.CODIGOS['arroyo'], ma.CODIGOS['ladera']
    assert (amb[37:44, 105:] == A).all()                             # a 30 m o menos
    assert (amb[45, 105:] == D).all() and (amb[35, 105:] == D).all() # a 50 m, ladera
    assert amb[60, EJE] == D                                         # la vaguada ya no cuenta
    for x in (d1, d2, d3):
        limpiar(x)


def test_el_ambiente_por_cercania_pide_lo_que_necesita_y_si_no_se_pide_no_esta():
    campo, lagunas = _mundo_con_ambiente()
    mod = _modelo_de_humedad()
    sin_dem = BANDAS_AMB[:-1]
    casos = [
        (PRED_AMB, {'nivel': 'ambiente_humedad'},
         'hacen falta también las variables: ndwi_mean, slope_deg, elev_dem_m'),
        (sin_dem, {'nivel': 'ambiente_humedad'}, 'hacen falta también las variables: elev_dem_m'),
        (BANDAS_AMB, {'nivel': 'humedad'}, 'nivel combinado'),
        (BANDAS_AMB, {'nivel': 'ambiente_humedad', 'params': {'arroyo_m': -5}}, 'arroyo_m'),
        (sin_dem, {'nivel': 'ambiente_humedad', 'relieve': lambda m: np.zeros((3, 3))},
         'el relieve ha llegado con forma (3, 3)'),
    ]
    for bandas, amb, texto in casos:
        d = carpeta()
        try:
            mra.hacer_mapa(mod, REJ,
                           lambda c, f, w, h, n=len(bandas): campo[f:f + h, c:c + w, :n],
                           d, bandas=bandas, ambiente=amb)
        except ValueError as e:
            assert texto in str(e), (texto, str(e))
        else:
            raise AssertionError(f'debía fallar: {texto}')
        assert not list(d.glob('*.tif'))                 # y no deja ficheros a medias
        limpiar(d)
    # un modelo de otro nivel (aquí, uno que no es el de la humedad) tampoco vale
    d = carpeta()
    try:
        mra.hacer_mapa(dict(mod, nivel='pureza'), REJ, lambda c, f, w, h: campo[f:f + h, c:c + w],
                       d, bandas=BANDAS_AMB, ambiente={'nivel': 'ambiente_humedad'})
    except ValueError as e:
        assert 'nivel combinado' in str(e)
    else:
        raise AssertionError('el modelo tiene que ser el de la humedad')
    limpiar(d)
    # Sin pedirlo, el mapa es el de siempre: las categorías del modelo y nada más.
    d = carpeta()
    m = mra.hacer_mapa(mod, REJ, lambda c, f, w, h: campo[f:f + h, c:c + w], d, bandas=BANDAS_AMB)
    assert not {'ambiente', 'categorias_modelo', 'ramas'} & set(m)
    assert [c['categoria'] for c in m['categorias']] == ['no borreguil', 'húmedo', 'seco']
    assert not any((d / mra.FICHEROS[k]).exists() for k in ('ambiente', 'distancias', 'referencias'))
    q = mra.consultar(d, *A_GEO.transform(X0 + 355, Y0 - 125))
    assert q['ambiente'] is None and q['distancias'] is None
    with abrir(d / 'clase.tif') as f:
        assert 'AMBIENTE_POR_CERCANIA' not in f.etiquetas
    limpiar(d)


def test_el_ambiente_convive_con_la_zona_y_con_la_pureza():
    bandas = BANDAS_AMB + [mr.VARIABLE_VERDE]
    campo, lagunas = _mundo_con_ambiente(bandas)
    # una zona que deja fuera la laguna (columnas 20 a 27) pero no lo que la rodea
    zona = box(X0 + 350, Y0 - 700, X0 + 1100, Y0 - 50)
    d = carpeta()
    meta = mra.hacer_mapa(
        _modelo_de_humedad(), REJ, lambda c, f, w, h: campo[f:f + h, c:c + w], d, lado=50,
        zona_utm=zona, bandas=bandas, pureza=PUREZA,
        ambiente={'nivel': 'ambiente_humedad', 'params': {'cauce_ha': 1.0}, 'lagunas': lagunas})
    clase, amb, pur = leer(d / 'clase.tif', 1), leer(d / 'ambiente.tif', 1), leer(d / 'pureza.tif', 1)
    fuera = ~mra.dentro(zona, REJ, (0, 0, REJ['ancho'], REJ['alto']))
    assert 0 < fuera.sum() < fuera.size
    assert (clase[fuera] == 255).all() and (amb[fuera] == 255).all()
    assert (clase[~fuera] != 255).all() and (amb[~fuera] != 255).all()
    # La laguna queda fuera de la zona, pero se ve (su bloque se lee entero): lo que la
    # rodea por dentro de la zona es «laguna».
    assert fuera[12, 24] and (amb[10:16, 35:38] == ma.CODIGOS['laguna']).all()
    assert (leer(d / 'lagunas_y_cauces.tif', 1)[LAGO] == 1).all()
    # Las distancias se guardan de todo el rectángulo; la pureza, solo del borreguil.
    assert np.isfinite(leer(d / 'distancias_ambiente.tif', 1)).all()
    borr = (clase != 0) & (clase != 255)
    assert np.array_equal(pur != 255, borr) and borr.sum() > 0
    assert meta['pureza'] and meta['ambiente']
    assert sum(c['pixeles'] for c in meta['categorias']) == int((~fuera).sum())
    assert sum(r['pixeles'] for r in meta['ramas']) == int(borr.sum())
    assert sum(c['pixeles'] for c in meta['pureza']['categorias']) == int(borr.sum())
    limpiar(d)


def test_los_colores_de_los_ambientes_son_los_que_eligio_la_usuaria():
    def luz(categoria):
        r, g, b = mra._rgb(mra.color(categoria))
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    def tono(categoria):
        import colorsys
        r, g, b = mra._rgb(mra.color(categoria))
        return 360 * colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)[0]

    # Nivel combinado: la paleta «Natural y contrastada» que eligió ella, tal cual (no
    # se retoca sin que lo pida).
    assert {c: mra.color(c) for c in mr.NIVELES['ambiente_humedad']} == {
        'arroyo · húmedo': '#168C73', 'arroyo · seco': '#D9AA37',
        'laguna · húmedo': '#2875BA', 'laguna · seco': '#D98850',
        'ladera · húmedo': '#526A35', 'ladera · seco': '#B85C42'}
    # En cada ambiente, lo húmedo en un tono frío (verdes y azules) y lo seco en uno
    # cálido (del rojo teja al mostaza).
    for amb in mr.NIVELES['ambiente']:
        assert 80 <= tono(f'{amb} · húmedo') <= 250, amb
        assert 0 <= tono(f'{amb} · seco') <= 50, amb
    # Nivel «ambiente» solo: el arroyo verde, la laguna azul y la ladera marrón (un
    # marrón es un naranja oscuro), como pidió; y el marrón, más oscuro que el verde:
    # con daltonismo se distinguen por la claridad, no por el tono.
    pedido = {'arroyo': (90, 150), 'laguna': (200, 250), 'ladera': (15, 45)}
    assert set(pedido) == set(mr.NIVELES['ambiente'])
    for amb, (desde, hasta) in pedido.items():
        assert desde <= tono(amb) <= hasta, (amb, tono(amb))
    assert luz('ladera') < luz('arroyo') - 30


def test_cuentas_metadatos_y_codigos():
    d = carpeta()
    meta = mra.hacer_mapa(MOD, REJ, lector(rej=REJ), d)
    clase = leer(d / 'clase.tif', 1)
    assert meta['pixeles'] == 9600 == meta['pixeles_clasificados'] and meta['pixeles_sin_dato'] == 0
    assert [c['codigo'] for c in meta['categorias']] == [0, 1, 2]
    for c in meta['categorias']:
        assert c['pixeles'] == int((clase == c['codigo']).sum())
        assert c['hectareas'] == round(c['pixeles'] / 100, 2)       # 1 píxel = 100 m²
        assert c['color'] == mra.COLORES[c['categoria']]
    assert abs(meta['pmax_media'] - float(np.nanmean(leer(d / 'pmax.tif', 1)))) < 1e-5
    assert meta['crs'] == 'EPSG:32630' and meta['resolucion_m'] == 10
    assert meta['firma_modelo'] == MOD['firma'] and meta['nivel'] == 'humedad'
    assert mra.leer_meta(d) == json.loads((d / 'mapa.json').read_text(encoding='utf-8'))
    lineas = (d / 'codigos.csv').read_text(encoding='utf-8-sig').strip().split('\n')
    assert lineas[0] == 'codigo,categoria,color,pixeles,hectareas'
    assert lineas[1].startswith('0,no borreguil,#') and lineas[2].startswith('1,húmedo,#')
    assert lineas[-1] == '255,sin dato,,0,0.0'
    # los colores y los nombres van dentro de los ficheros
    with abrir(d / 'clase.tif') as f:
        mapa_color = f.paleta
        assert mapa_color[1][:3] == (0x00, 0x9E, 0x73) and mapa_color[2][:3] == (0xD5, 0x5E, 0x00)
        assert '1 = húmedo' in f.etiquetas['CATEGORIAS']
    with abrir(d / 'probabilidades.tif') as f:
        assert f.bandas == 3 and f.nombres == ('P(no borreguil)', 'P(húmedo)', 'P(seco)')
        assert f.tipo == 'float32' and np.isnan(f.sin_dato)
    limpiar(d)


def test_dentro_de_una_zona_estan_los_pixeles_que_tienen_el_centro_dentro():
    from shapely.geometry import Point, Polygon
    # un cuadrilátero torcido con un hueco triangular en medio
    zona = Polygon([(X0 + 33, Y0 - 41), (X0 + 911, Y0 - 87), (X0 + 705, Y0 - 688),
                    (X0 + 148, Y0 - 512)],
                   holes=[[(X0 + 300, Y0 - 200), (X0 + 500, Y0 - 210), (X0 + 420, Y0 - 400)]])
    for ventana in ((0, 0, 120, 80), (30, 20, 40, 25), (100, 60, 20, 20), (20, 12, 7, 5)):
        col, fila, ancho, alto = ventana
        m = mra.dentro(zona, REJ, ventana)
        uno_a_uno = np.array([[zona.contains(Point(X0 + (col + c) * 10 + 5,
                                                   Y0 - (fila + f) * 10 - 5))
                               for c in range(ancho)] for f in range(alto)])
        assert m.dtype == bool and m.shape == (alto, ancho) and np.array_equal(m, uno_a_uno)
    assert 0 < mra.dentro(zona, REJ, (0, 0, 120, 80)).sum() < 120 * 80
    assert mra.dentro(zona, REJ, (20, 12, 7, 5)).all()            # ventana entera dentro
    assert not mra.dentro(zona, REJ, (100, 60, 20, 20)).any()     # y entera fuera
    assert not mra.dentro(zona, REJ, (38, 26, 4, 3)).any()        # en el hueco


def test_fuera_de_la_zona_o_de_la_mascara_no_se_clasifica():
    zona = box(X0 + 200, Y0 - 600, X0 + 1000, Y0 - 100)        # 80 × 50 píxeles
    mascara = box(X0 + 600, Y0 - 800, X0 + 1200, Y0)           # la mitad este
    d = carpeta()
    llamadas, avances = [], []
    meta = mra.hacer_mapa(MOD, REJ, lector(rej=REJ, llamadas=llamadas), d, zona_utm=zona,
                          mascara_utm=mascara, lado=64,
                          progreso=lambda a, b: avances.append((a, b)))
    clase = leer(d / 'clase.tif', 1)
    ok = clase != mra.SIN_DATO_CLASE
    esperado = np.zeros_like(ok)
    esperado[10:60, 60:100] = True                 # zona ∩ máscara: 40 × 50 píxeles
    assert (ok == esperado).all() and meta['pixeles_clasificados'] == 2000
    # de los cuatro bloques, los dos de abajo no tienen nada que clasificar: ni se piden
    assert llamadas == [(0, 0, 64, 64), (64, 0, 56, 64)] and avances == [(1, 2), (2, 2)]
    assert meta['bloques'] == 4 and meta['bloques_leidos'] == 2
    assert mra.bloques_utiles(REJ, zona, mascara, 64) == llamadas
    assert mra.bloques_utiles(REJ, lado=64) == mra.bloques(REJ, 64)
    assert meta['con_zona'] and meta['con_mascara'] and meta['pixeles_sin_dato'] == 7600
    assert meta['pixeles_zona'] == 2000 and meta['pixeles_sin_variables'] == 0
    assert np.isnan(leer(d / 'pmax.tif', 1)[~ok]).all()
    assert np.isnan(leer(d / 'probabilidades.tif')[:, ~ok]).all()
    q = mra.consultar(d, *A_GEO.transform(X0 + 50, Y0 - 50))       # dentro del mapa, sin dato
    assert q is not None and q['codigo'] is None and q['probabilidades'] == []
    assert mra.consultar(d, *A_GEO.transform(X0 - 500, Y0 - 50)) is None    # fuera del mapa
    limpiar(d)


def test_los_huecos_de_las_variables_quedan_sin_dato():
    d = carpeta()
    hueco = lambda c, f: 30 <= c < 40 and 20 <= f < 25                     # noqa: E731
    meta = mra.hacer_mapa(MOD, REJ, lector(rej=REJ, hueco=hueco), d, lado=64)
    clase = leer(d / 'clase.tif', 1)
    assert (clase[20:25, 30:40] == mra.SIN_DATO_CLASE).all()
    assert meta['pixeles_sin_dato'] == 50 == int((clase == mra.SIN_DATO_CLASE).sum())
    assert meta['pixeles_zona'] == 9600 and meta['pixeles_sin_variables'] == 50
    limpiar(d)


def test_si_falla_un_bloque_no_queda_un_mapa_a_medias():
    d = carpeta()
    llamadas = []
    try:
        mra.hacer_mapa(MOD, REJ, lector(rej=REJ, falla_en=3, llamadas=llamadas), d, lado=50)
    except RuntimeError as e:
        assert 'bloque' in str(e)
    else:
        raise AssertionError('debía detenerse')
    assert len(llamadas) == 3 and not list(d.glob('*')) and mra.leer_meta(d) is None
    limpiar(d)


def test_sin_clase_de_ausencia_el_mapa_solo_tiene_tipos():
    mod, _, _ = entrenado(con_ausencia=False)
    d = carpeta()
    meta = mra.hacer_mapa(mod, REJ, lector(rej=REJ), d)
    assert [c['categoria'] for c in meta['categorias']] == ['húmedo', 'seco']
    assert set(np.unique(leer(d / 'clase.tif', 1))) <= {1, 2}       # los códigos no se corren
    # Sin «no borreguil» no hay nada que decidir en el primer paso: ni umbral ni mapa
    # de probabilidad de borreguil (valdría 1 en todas partes).
    assert mod['regla']['umbral'] is None and meta['regla'] == {'criterio': None, 'umbral': None}
    assert 'pborr' not in meta['ficheros'] and not (d / 'probabilidad_borreguil.tif').exists()
    assert sorted(x.name for x in d.glob('*')) == ['clase.tif', 'codigos.csv', 'mapa.json',
                                                   'pmax.tif', 'probabilidades.tif']
    m0 = mod['muestras'][0]
    q = mra.consultar(d, m0['lon'], m0['lat'])
    assert q['p_borreguil'] is None and q['codigo'] in (1, 2)
    limpiar(d)


def test_si_no_hay_nada_que_clasificar_no_se_hace_el_mapa():
    d = carpeta()
    lejos = box(X0 + 5000, Y0 - 800, X0 + 6000, Y0)
    llamadas = []
    try:
        mra.hacer_mapa(MOD, REJ, lector(rej=REJ, llamadas=llamadas), d, mascara_utm=lejos)
    except ValueError as e:
        assert 'ningún píxel' in str(e)
    else:
        raise AssertionError('debía negarse')
    assert not llamadas and not list(d.glob('*'))
    limpiar(d)


def test_el_modelo_recuerda_donde_estan_sus_muestras():
    assert len(MOD['muestras']) == len(TAB['filas']) == MOD['n_muestras']
    for m, f in zip(MOD['muestras'], TAB['filas']):
        assert m == {'ID': f['ID'], 'lon': f['lon'], 'lat': f['lat'],
                     'categoria': f['categoria'], 'codigo': f['codigo']}


# ---------------------------------------------------------------- la zona pedida
def en_grados(geom):
    from shapely.ops import transform
    return transform(A_GEO.transform, geom)


def test_un_rectangulo_centrado_mide_lo_que_se_pide():
    lon, lat = A_GEO.transform(X0 + 600, Y0 - 400)
    minx, miny, maxx, maxy = mra.rect_centrado(lon, lat, 1.2, 0.8, EPSG)
    assert abs(maxx - minx - 1200) < 1e-6 and abs(maxy - miny - 800) < 1e-6
    assert abs(minx - X0) < 1e-3 and abs(maxy - Y0) < 1e-3
    z = mra.preparar_zona(EPSG, limites=(minx + 0.01, miny + 0.01, maxx - 0.01, maxy - 0.01))
    assert z['error'] is None and z['rej'] == REJ and z['zona_utm'] is None
    assert z['utiles'] == [(0, 0, 120, 80)] and z['bloques'] == 1
    assert abs(z['km2'] - 0.96) < 1e-3
    assert mra.preparar_zona(EPSG, limites=(5, 5, 5, 9))['error'] == 'sin_superficie'


def test_el_rectangulo_de_las_muestras_las_contiene_a_todas():
    puntos = [(m['lon'], m['lat']) for m in MOD['muestras']]
    minx, miny, maxx, maxy = mra.limites_de_puntos(puntos, EPSG, margen_m=100)
    xs, ys = zip(*(A_UTM.transform(lo, la) for lo, la in puntos))
    assert abs(minx - (min(xs) - 100)) < 1e-6 and abs(maxy - (max(ys) + 100)) < 1e-6
    rej = mra.preparar_zona(EPSG, limites=(minx, miny, maxx, maxy))['rej']
    for x, y in zip(xs, ys):
        assert rej['x0'] <= x <= rej['x0'] + rej['ancho'] * 10
        assert rej['y0'] - rej['alto'] * 10 <= y <= rej['y0']


def test_la_rejilla_se_cine_a_la_zona_y_a_la_mascara():
    zona = en_grados(box(X0 + 200, Y0 - 600, X0 + 1000, Y0 - 100))
    mascara = en_grados(box(X0 + 600, Y0 - 800, X0 + 1200, Y0))
    z = mra.preparar_zona(EPSG, zona=zona, mascara=mascara, lado=16)
    assert z['error'] is None
    r = z['rej']                                    # zona ∩ máscara: 400 × 500 m
    assert abs(r['x0'] - (X0 + 600)) <= 10 and abs(r['y0'] - (Y0 - 100)) <= 10
    assert 40 <= r['ancho'] <= 42 and 50 <= r['alto'] <= 52 and abs(z['km2'] - 0.2) < 0.01
    assert z['zona_utm'] is not None and z['mascara_utm'] is not None
    assert z['utiles'] and len(z['utiles']) <= z['bloques'] == 12
    # solo con la máscara: la zona es todo lo que cubre
    solo = mra.preparar_zona(EPSG, mascara=mascara)
    assert solo['error'] is None and solo['zona_utm'] is None
    assert 60 <= solo['rej']['ancho'] <= 62 and 80 <= solo['rej']['alto'] <= 82
    # un rectángulo con máscara
    lim = (X0, Y0 - 800, X0 + 700, Y0)
    rect = mra.preparar_zona(EPSG, limites=lim, mascara=mascara)
    assert rect['error'] is None and 10 <= rect['rej']['ancho'] <= 12
    # la máscara no toca la zona
    lejos = en_grados(box(X0 + 5000, Y0 - 800, X0 + 6000, Y0))
    assert mra.preparar_zona(EPSG, zona=zona, mascara=lejos)['error'] == 'mascara_fuera'
    assert mra.preparar_zona(EPSG, limites=lim, mascara=lejos)['error'] == 'mascara_fuera'
    # una zona desmesurada se rechaza sin calcular nada
    grande = mra.preparar_zona(EPSG, limites=(0, 0, 3_000_000, 3_000_000))
    assert grande['error'] == 'demasiado_grande' and grande['bloques'] > mra.MAX_BLOQUES_REJILLA
    try:
        mra.preparar_zona(EPSG)
    except ValueError:
        pass
    else:
        raise AssertionError('sin zona no hay nada que preparar')


def test_un_poligono_que_se_cruza_consigo_mismo_no_rompe_la_zona():
    from shapely.geometry import Polygon
    lazo = Polygon([(X0, Y0), (X0 + 400, Y0 - 400), (X0 + 400, Y0), (X0, Y0 - 400)])
    assert not lazo.is_valid
    z = mra.preparar_zona(EPSG, limites=(X0, Y0 - 800, X0 + 1200, Y0), mascara=en_grados(lazo))
    assert z['error'] is None and z['mascara_utm'].is_valid and 0.07 < z['km2'] < 0.09
    d = carpeta()
    meta = mra.hacer_mapa(MOD, z['rej'], lector(rej=z['rej']), d, mascara_utm=z['mascara_utm'])
    assert 700 < meta['pixeles_clasificados'] < 900          # dos triángulos: 8 ha
    limpiar(d)


def test_el_contorno_de_la_rejilla_son_sus_cuatro_esquinas():
    c = mra.contorno_lonlat(REJ)
    assert len(c) == 4
    o, s, e, n = mra.limites_lonlat(REJ)
    for lat, lon in c:
        assert s - 1e-9 <= lat <= n + 1e-9 and o - 1e-9 <= lon <= e + 1e-9
    x, y = A_UTM.transform(c[0][1], c[0][0])
    assert abs(x - X0) < 1e-3 and abs(y - Y0) < 1e-3
    # el contorno de un píxel: 10 × 10 m, con su esquina en la rejilla
    px = [A_UTM.transform(lon, lat) for lat, lon in mra.contorno_pixel(REJ, 3, 2)]
    assert abs(px[0][0] - (X0 + 30)) < 1e-3 and abs(px[0][1] - (Y0 - 20)) < 1e-3
    assert abs(px[2][0] - (X0 + 40)) < 1e-3 and abs(px[2][1] - (Y0 - 30)) < 1e-3


def test_el_encuadre_deja_ver_todo_el_rectangulo():
    lim = mra.limites_lonlat(REJ)                     # 1,2 × 0,8 km
    centro, zoom = mra.encuadre(lim, 600, 460)
    assert abs(centro[0] - (lim[1] + lim[3]) / 2) < 1e-9
    assert abs(centro[1] - (lim[0] + lim[2]) / 2) < 1e-9
    # a ese zoom el rectángulo cabe, y al siguiente ya no
    ancho = lambda z: (lim[2] - lim[0]) / 360 * 256 * 2 ** z        # noqa: E731
    assert ancho(zoom) <= 600 and (ancho(zoom + 1) > 600 or zoom == 18)
    assert mra.encuadre((-3.6, 36.9, -2.6, 37.3))[1] < zoom           # una zona mayor
    assert mra.encuadre((-3.3, 37.05, -3.3, 37.05))[1] == 18          # un punto


# ------------------------------------------------------------ bloques ya leídos
def test_lo_leido_se_guarda_y_no_se_vuelve_a_pedir():
    d, a, b = carpeta(), carpeta(), carpeta()
    llamadas = []
    con_memoria = mra.lector_con_memoria(lector(rej=REJ, llamadas=llamadas), d, REJ)
    ventanas = mra.bloques(REJ, 50)
    assert mra.bloques_guardados(d, REJ, ventanas) == 0
    assert mra.bloques_por_leer(d, REJ, ventanas) == ventanas
    ma = mra.hacer_mapa(MOD, REJ, con_memoria, a, lado=50)
    assert len(llamadas) == 6 and mra.bloques_guardados(d, REJ, ventanas) == 6
    # otra vez (otro nivel, o el mismo mapa repetido): no se pide nada
    mb = mra.hacer_mapa(MOD, REJ, con_memoria, b, lado=50)
    assert len(llamadas) == 6 and ma['categorias'] == mb['categorias']
    for nombre in ('clase.tif', 'pmax.tif', 'probabilidades.tif'):
        assert np.array_equal(leer(a / nombre), leer(b / nombre), equal_nan=True), nombre
    # lo guardado es exactamente lo leído
    directo = lector(rej=REJ)(50, 0, 50, 50)
    assert np.array_equal(con_memoria(50, 0, 50, 50), directo) and len(llamadas) == 6
    # un fichero estropeado se descarta y se vuelve a pedir
    ruta = next(iter(sorted(d.glob('*.npy'))))
    ruta.write_bytes(b'esto no es un bloque')
    assert mra.bloques_guardados(d, REJ, ventanas) == 6
    for v in ventanas:
        con_memoria(*v)
    assert len(llamadas) == 7 and not list(d.glob('*.parte'))
    for x in (d, a, b):
        limpiar(x)


def test_no_poder_guardar_un_bloque_no_detiene_el_mapa():
    d, a = carpeta(), carpeta()
    llamadas = []
    con_memoria = mra.lector_con_memoria(lector(rej=REJ, llamadas=llamadas), d, REJ)
    limpiar(d)                                   # la carpeta desaparece: no se puede guardar
    meta = mra.hacer_mapa(MOD, REJ, con_memoria, a, lado=50)
    assert meta['bloques_leidos'] == 6 and len(llamadas) == 6 and not d.exists()
    b = carpeta()
    mra.hacer_mapa(MOD, REJ, lector(rej=REJ), b, lado=50)
    assert np.array_equal(leer(a / 'clase.tif'), leer(b / 'clase.tif'))
    limpiar(a)
    limpiar(b)


def test_si_el_mapa_se_interrumpe_sigue_donde_iba():
    d, a = carpeta(), carpeta()
    llamadas = []
    try:
        mra.hacer_mapa(MOD, REJ, mra.lector_con_memoria(
            lector(rej=REJ, falla_en=4, llamadas=llamadas), d, REJ), a, lado=50)
    except RuntimeError:
        pass
    else:
        raise AssertionError('debía detenerse')
    ventanas = mra.bloques(REJ, 50)
    assert len(llamadas) == 4 and mra.bloques_guardados(d, REJ, ventanas) == 3
    assert mra.bloques_por_leer(d, REJ, ventanas) == ventanas[3:]
    assert not list(a.glob('*'))                      # el mapa a medias no queda
    otras = []
    meta = mra.hacer_mapa(MOD, REJ, mra.lector_con_memoria(
        lector(rej=REJ, llamadas=otras), d, REJ), a, lado=50)
    assert otras == ventanas[3:] and meta['bloques_leidos'] == 6
    b = carpeta()
    mra.hacer_mapa(MOD, REJ, lector(rej=REJ), b, lado=50)
    assert np.array_equal(leer(a / 'clase.tif'), leer(b / 'clase.tif'))
    for x in (d, a, b):
        limpiar(x)


def test_la_clave_de_lectura_cambia_con_los_anios_la_rejilla_o_las_variables():
    base = mra.clave_lectura(EPSG, [2023, 2024], ORDEN)
    assert base == mra.clave_lectura(EPSG, (2023, 2024), list(ORDEN)) and len(base) == 12
    assert base != mra.clave_lectura(EPSG, [2023], ORDEN)
    assert base != mra.clave_lectura(32631, [2023, 2024], ORDEN)
    assert base != mra.clave_lectura(EPSG, [2023, 2024], ORDEN[::-1])     # el orden cuenta
    assert base != mra.clave_lectura(EPSG, [2023, 2024], ORDEN[:-1])
    # …y con la manera de calcular las variables: si cambia, lo guardado no vale
    assert base == mra.clave_lectura(EPSG, [2023, 2024], ORDEN, 1)
    nueva = mra.clave_lectura(EPSG, [2023, 2024], ORDEN, geb.VERSION_PILA)
    assert geb.VERSION_PILA == 2 and nueva != base and len(nueva) == 12


# ---------------------------------------------------------------------- el visor
A_UTM_DE_MERCATOR = pyproj.Transformer.from_crs(3857, EPSG, always_xy=True)


def raster_de_prueba(ancho=120, alto=80, dibujo=None):
    """Un GeoTIFF de categorías en la rejilla de prueba con un dibujo reconocible: la
    categoría de cada píxel es (columna + 2 · fila) % 3, y hay un hueco sin dato."""
    d = carpeta()
    if dibujo is None:
        dibujo = ((np.arange(ancho)[None, :] + 2 * np.arange(alto)[:, None]) % 3).astype('uint8')
        dibujo[10:14, 20:26] = mra.SIN_DATO_CLASE
    rej = dict(REJ, ancho=ancho, alto=alto)
    mt.escribir(d / 'clase.tif', rej, dibujo, sin_dato=mra.SIN_DATO_CLASE)
    return d, dibujo, rej


def vista_en_utm(cx, cy, medio_ancho, medio_alto):
    o, s = A_GEO.transform(cx - medio_ancho, cy - medio_alto)
    e, n = A_GEO.transform(cx + medio_ancho, cy + medio_alto)
    return o, s, e, n


def pixel_bajo_cada_punto(img, dibujo, cuantos=4000, semilla=1):
    """Para puntos de la imagen elegidos al azar: (lo pintado, lo que hay debajo en el
    ráster). Se saltan los que caen a menos de 2 cm de un borde entre píxeles."""
    alto, ancho = img['matriz'].shape
    x0, y0, x1, y1 = img['cubre']
    rng = np.random.default_rng(semilla)
    pintado, debajo = [], []
    for _ in range(cuantos):
        j, i = int(rng.integers(0, ancho)), int(rng.integers(0, alto))
        x, y = A_UTM_DE_MERCATOR.transform(x0 + (j + 0.5) * img['res'], y1 - (i + 0.5) * img['res'])
        col, fila = (x - X0) / 10, (Y0 - y) / 10
        if min(col % 1, 1 - col % 1, fila % 1, 1 - fila % 1) < 0.002:
            continue
        dentro = 0 <= col < dibujo.shape[1] and 0 <= fila < dibujo.shape[0]
        pintado.append(int(img['matriz'][i, j]))
        debajo.append(int(dibujo[int(fila), int(col)]) if dentro else mra.SIN_DATO_CLASE)
    return np.array(pintado), np.array(debajo)


def test_se_sabe_donde_cae_cada_punto_de_pantalla_sin_apartarse_un_centimetro():
    """bajo_cada_punto convierte de verdad solo los nudos de una malla e interpola entre
    ellos: aquí se compara con convertir todos los puntos, uno por uno."""
    cx, cy = mra._a_mercator(*A_GEO.transform(X0 + 600, Y0 - 400))
    ancho, alto = 700, 450
    for zoom in (8, 11, 12, 13, 15, 18, 21):
        res = mra.RES_ZOOM0 / 2 ** zoom
        c0, f1 = math.floor(cx / res) - ancho // 2, math.floor(cy / res) + alto // 2
        for sub in (1, 3):
            xs, ys = mra.bajo_cada_punto(EPSG, c0, f1, res, ancho, alto, sub)
            assert xs.shape == ys.shape == (alto * sub, ancho * sub)
            xm, ym = np.meshgrid((c0 + (np.arange(ancho * sub) + 0.5) / sub) * res,
                                 (f1 - (np.arange(alto * sub) + 0.5) / sub) * res)
            x, y = A_UTM_DE_MERCATOR.transform(xm, ym)
            assert max(np.abs(xs - x).max(), np.abs(ys - y).max()) < 0.01, zoom
    # una imagen de un solo punto también tiene sus dos nudos
    xs, ys = mra.bajo_cada_punto(EPSG, math.floor(cx / 0.5), math.floor(cy / 0.5) + 1, 0.5, 1, 1)
    assert xs.shape == (1, 1) and abs(xs[0, 0] - (X0 + 600)) < 1 and abs(ys[0, 0] - (Y0 - 400)) < 1


def test_de_cerca_el_visor_pinta_en_cada_punto_el_pixel_que_tiene_debajo():
    d, dibujo, _ = raster_de_prueba()
    # Unos 60 × 44 m alrededor de la esquina de cuatro píxeles, a zoom 21: cada píxel
    # de 10 m ocupa más de 150 puntos de pantalla.
    img = mra.imagen_encuadre(d / 'clase.tif', 1, vista_en_utm(X0 + 400, Y0 - 300, 30, 22), 21,
                              categorica=True, margen=0)
    m = img['matriz']
    assert m.dtype == np.uint8 and m.shape[1] > 900 and m.shape[0] > 600
    assert abs((img['cubre'][2] - img['cubre'][0]) / m.shape[1] - img['res']) < 1e-9
    pintado, debajo = pixel_bajo_cada_punto(img, dibujo)
    assert len(pintado) > 3500 and (pintado == debajo).all(), int((pintado != debajo).sum())
    assert set(np.unique(pintado)) == {0, 1, 2}
    # los límites en grados son los de esa misma imagen
    (s, o), (n, e) = img['limites']
    x0, y0, x1, y1 = img['cubre']
    assert np.allclose((o, s), mra._de_mercator(x0, y0)) and np.allclose((e, n), mra._de_mercator(x1, y1))
    # en el borde del mapa y en el hueco no se pinta nada
    borde = mra.imagen_encuadre(d / 'clase.tif', 1, vista_en_utm(X0 + 230, Y0 - 120, 45, 30), 20,
                                categorica=True, margen=0)
    pintado, debajo = pixel_bajo_cada_punto(borde, dibujo)
    assert (pintado == debajo).all() and (debajo == mra.SIN_DATO_CLASE).sum() > 100
    rgba = mra.pintar_clases(borde['matriz'], [{'codigo': c, 'categoria': n, 'color': mra.color(n)}
                                               for c, n in ((0, 'no borreguil'), (1, 'húmedo'), (2, 'seco'))], 1.0)
    assert (rgba[borde['matriz'] == mra.SIN_DATO_CLASE, 3] == 0).all()
    assert tuple(rgba[borde['matriz'] == 2][0]) == (0xD5, 0x5E, 0x00, 255)
    # «no borreguil» va más transparente que los tipos, para no tapar el terreno
    assert rgba[borde['matriz'] == 0][0][3] == round(255 * 0.45) < rgba[borde['matriz'] == 1][0][3]
    limpiar(d)


def test_el_dibujo_no_baila_al_desplazar_la_vista():
    d, dibujo, _ = raster_de_prueba()
    a = mra.imagen_encuadre(d / 'clase.tif', 1, vista_en_utm(X0 + 400, Y0 - 300, 40, 30), 19,
                            categorica=True, margen=0)
    b = mra.imagen_encuadre(d / 'clase.tif', 1, vista_en_utm(X0 + 417.3, Y0 - 291.9, 40, 30), 19,
                            categorica=True, margen=0)
    res = a['res']
    assert b['res'] == res
    # las dos imágenes van sobre la misma rejilla de puntos: donde se solapan, coinciden
    dj = round((b['cubre'][0] - a['cubre'][0]) / res)
    di = round((a['cubre'][3] - b['cubre'][3]) / res)
    assert abs((b['cubre'][0] - a['cubre'][0]) / res - dj) < 1e-6 and dj > 0 and di < 0
    comun_a = a['matriz'][:a['matriz'].shape[0] + di, dj:]
    comun_b = b['matriz'][-di:, :b['matriz'].shape[1] - dj]
    alto, ancho = min(comun_a.shape[0], comun_b.shape[0]), min(comun_a.shape[1], comun_b.shape[1])
    assert alto > 100 and ancho > 100
    assert np.array_equal(comun_a[:alto, :ancho], comun_b[:alto, :ancho])
    # con margen se dibuja media pantalla más por cada lado
    c = mra.imagen_encuadre(d / 'clase.tif', 1, vista_en_utm(X0 + 400, Y0 - 300, 40, 30), 19,
                            categorica=True)
    assert 1.9 < c['matriz'].shape[1] / a['matriz'].shape[1] < 2.1
    limpiar(d)


def test_de_lejos_cada_punto_lleva_la_categoria_mas_frecuente_y_fuera_no_hay_nada():
    mitades = np.zeros((80, 120), dtype='uint8')
    mitades[:, 60:] = 2                                # oeste «no borreguil», este «seco»
    d, _, rej = raster_de_prueba(dibujo=mitades)
    todo = mra.limites_lonlat(rej)
    img = mra.imagen_encuadre(d / 'clase.tif', 1, todo, 12, categorica=True)
    m = img['matriz']                                  # 38 m por punto: unos 40 × 27 puntos
    assert 30 < m.shape[1] < 50 and 18 < m.shape[0] < 35
    con = m != mra.SIN_DATO_CLASE
    cols = np.where(con.any(axis=0))[0]
    assert set(np.unique(m[con])) == {0, 2}
    assert (m[:, cols[:len(cols) // 2 - 2]][con[:, cols[:len(cols) // 2 - 2]]] == 0).all()
    assert (m[:, cols[len(cols) // 2 + 2:]][con[:, cols[len(cols) // 2 + 2:]]] == 2).all()
    # la imagen no se sale del mapa más que un punto
    marco = mra.dibujar_vista({}, d / 'clase.tif', 1, todo, 12, lambda x: np.zeros(x.shape + (4,), 'uint8'),
                              categorica=True)
    assert marco is not None
    lejos = (todo[0] + 1.0, todo[1], todo[2] + 1.0, todo[3])
    assert mra.imagen_encuadre(d / 'clase.tif', 1, lejos, 12, categorica=True) is None
    assert mra.dibujar_vista({}, d / 'clase.tif', 1, lejos, 12, lambda x: x, categorica=True) is None
    # las probabilidades, de lejos, son la media
    mt.escribir(d / 'p.tif', rej, np.where(mitades == 2, 0.9, 0.1).astype('float32'),
                sin_dato=float('nan'))
    p = mra.imagen_encuadre(d / 'p.tif', 1, todo, 12)['matriz']
    assert p.dtype == np.float32 and 0.09 < np.nanmin(p) < 0.11 and 0.89 < np.nanmax(p) < 0.91
    pr = mra.pintar_probabilidad(p)
    assert (pr[np.isnan(p), 3] == 0).all() and (pr[np.isfinite(p), 3] > 0).all()
    limpiar(d)


def test_las_vistas_reducidas_no_cambian_los_datos_y_sirven_para_ver_de_lejos():
    rng = np.random.default_rng(3)
    # manchas de 40 × 40 píxeles, para que la categoría más frecuente tenga sentido
    grande = np.kron(rng.integers(0, 3, (20, 28)), np.ones((40, 40))).astype('uint8')   # 1120 × 800
    d, _, rej = raster_de_prueba(1120, 800, dibujo=grande)
    assert mra._con_vistas_reducidas(d / 'clase.tif', categorica=True) == [2, 4]
    with abrir(d / 'clase.tif') as f:
        assert f.reducidas == [2, 4] and np.array_equal(f.leer(1), grande)
    # un mapa pequeño no las necesita, y un fichero que no se puede abrir no rompe nada
    chico, _, _ = raster_de_prueba()
    assert mra._con_vistas_reducidas(chico / 'clase.tif', categorica=True) == []
    assert mra._con_vistas_reducidas(chico / 'no_existe.tif', categorica=True) == []
    # de lejos (4 píxeles por punto) se lee de la versión reducida y sale lo mismo que
    # mirando el mapa entero: el centro de cada mancha conserva su categoría
    todo = mra.limites_lonlat(rej)
    lejos = mra.imagen_encuadre(d / 'clase.tif', 1, todo, 12, categorica=True)
    pintado, debajo = pixel_bajo_cada_punto(lejos, grande, cuantos=1500)
    assert (pintado == debajo).mean() > 0.85            # solo fallan los bordes de las manchas
    # y de cerca sigue siendo exacto, con los datos a tamaño real
    cerca = mra.imagen_encuadre(d / 'clase.tif', 1, vista_en_utm(X0 + 4000, Y0 - 3000, 60, 40), 19,
                                categorica=True, margen=0)
    pintado, debajo = pixel_bajo_cada_punto(cerca, grande)
    assert (pintado == debajo).all()
    limpiar(d)
    limpiar(chico)


def test_el_mapa_pequeno_no_lleva_vistas_reducidas_y_lo_dice():
    d = carpeta()
    meta = mra.hacer_mapa(MOD, REJ, lector(rej=REJ), d)
    assert meta['vistas_reducidas'] == []
    with abrir(d / 'clase.tif') as f:
        assert f.reducidas == []
    limpiar(d)


def test_el_mapa_se_hace_se_ve_y_se_consulta_sin_rasterio():
    """En los equipos donde Windows bloquea rasterio (GDAL), el mapa tiene que salir
    igual. Aquí se impide cargarlo, esté como esté el equipo, y se hace todo el
    recorrido: zona, mapa con ambiente y pureza, versiones reducidas, visor y consulta."""
    suyos = {k: sys.modules.pop(k) for k in list(sys.modules)
             if k == 'rasterio' or k.startswith('rasterio.')}
    sys.modules['rasterio'] = None                 # «import rasterio» falla desde ahora
    d = carpeta()
    try:
        try:
            import rasterio                         # noqa: F401
            raise AssertionError('rasterio se ha podido cargar: la prueba no vale')
        except ImportError:
            pass
        bandas = BANDAS_AMB + [mr.VARIABLE_VERDE]
        campo, lagunas = _mundo_con_ambiente(bandas)
        zona = box(X0 + 100, Y0 - 700, X0 + 1100, Y0 - 100)
        prep = mra.preparar_zona(EPSG, limites=zona.bounds)
        assert prep['error'] is None
        assert (prep['rej']['ancho'], prep['rej']['alto'], len(prep['utiles'])) == (100, 60, 1)
        meta = mra.hacer_mapa(
            _modelo_de_humedad(), REJ, lambda c, f, w, h: campo[f:f + h, c:c + w], d, lado=50,
            zona_utm=zona, bandas=bandas, pureza=PUREZA, quitar_agua=True,
            ambiente={'nivel': 'ambiente_humedad', 'params': {'cauce_ha': 1.0},
                      'lagunas': lagunas, 'relieve': _relieve_con_margen()})
        assert meta['ambiente']['cauces'] == 'relieve' and meta['pureza']['categorias']
        assert 'mascara_agua' in meta
        # están todos los ficheros del mapa, y no queda ninguno a medias
        assert sorted(p.name for p in d.iterdir()) == sorted(mra.FICHEROS.values())
        assert mra._con_vistas_reducidas(d / 'clase.tif', categorica=True) == []
        with mt.abrir(d / 'clase.tif') as f:
            clase = f.leer(1)
        fuera = ~mra.dentro(zona, REJ, (0, 0, REJ['ancho'], REJ['alto']))
        assert 0 < fuera.sum() < fuera.size and np.array_equal(clase == 255, fuera)
        img = mra.imagen_encuadre(d / 'clase.tif', 1, vista_en_utm(X0 + 600, Y0 - 400, 80, 50),
                                  19, categorica=True, margen=0)
        pintado, debajo = pixel_bajo_cada_punto(img, clase, cuantos=1500)
        assert len(pintado) > 1000 and (pintado == debajo).all()
        q = mra.consultar(d, *A_GEO.transform(X0 + 605, Y0 - 405))
        assert q['codigo'] == int(clase[40, 60]) and q['ambiente'] == 'arroyo'
        assert q['distancias']['cauce'] == 0.0 and (q['pureza'] is None) == (q['codigo'] == 0)
        assert mra.consultar_varios(d, [A_GEO.transform(X0 - 50, Y0)]) == [None]
    finally:
        del sys.modules['rasterio']
        sys.modules.update(suyos)
        limpiar(d)


def test_la_vista_solo_se_vuelve_a_dibujar_cuando_hace_falta():
    d, dibujo, rej = raster_de_prueba()
    veces = []
    de_verdad = mra.imagen_encuadre

    def contando(*a, **k):
        veces.append(a[3])
        return de_verdad(*a, **k)

    pintar = lambda m: mra.pintar_probabilidad(m.astype('float32') / 2, 1.0)        # noqa: E731
    mra.imagen_encuadre = contando
    try:
        memoria = {}
        v = vista_en_utm(X0 + 400, Y0 - 300, 60, 40)
        a = mra.dibujar_vista(memoria, d / 'clase.tif', 1, v, 18, pintar, categorica=True)
        assert a is not None and a[0].startswith('data:image/png;base64,') and veces == [18]
        # un desplazamiento pequeño cabe en lo ya dibujado: no se toca nada
        b = mra.dibujar_vista(memoria, d / 'clase.tif', 1, vista_en_utm(X0 + 420, Y0 - 310, 60, 40),
                              18, pintar, categorica=True)
        assert b[0] is a[0] and b[1] == a[1] and veces == [18]
        # uno grande, o cambiar de zoom, sí obliga a dibujar otra vez
        c = mra.dibujar_vista(memoria, d / 'clase.tif', 1, vista_en_utm(X0 + 700, Y0 - 300, 60, 40),
                              18, pintar, categorica=True)
        assert c[0] is not a[0] and veces == [18, 18]
        mra.dibujar_vista(memoria, d / 'clase.tif', 1, vista_en_utm(X0 + 700, Y0 - 300, 60, 40), 19,
                          pintar, categorica=True)
        assert veces == [18, 18, 19]
        # cuando el mapa entero cabe en la vista, moverse alrededor no cambia nada
        todo = mra.limites_lonlat(rej)
        ancha = (todo[0] - 0.02, todo[1] - 0.02, todo[2] + 0.02, todo[3] + 0.02)
        e1 = mra.dibujar_vista(memoria, d / 'clase.tif', 1, ancha, 13, pintar, categorica=True)
        corrida = tuple(x + 0.01 for x in ancha)
        e2 = mra.dibujar_vista(memoria, d / 'clase.tif', 1, corrida, 13, pintar, categorica=True)
        assert e2[0] is e1[0] and veces == [18, 18, 19, 13]
    finally:
        mra.imagen_encuadre = de_verdad
    limpiar(d)


def test_lo_que_dice_el_visor_se_lee_con_cuidado():
    estado = {'last_clicked': {'lat': 37.05, 'lng': -3.34}, 'zoom': 15,
              'bounds': {'_southWest': {'lat': 37.0, 'lng': -3.4},
                         '_northEast': {'lat': 37.1, 'lng': -3.3}}}
    assert mra.leer_vista(estado) == ((-3.4, 37.0, -3.3, 37.1), 15.0)
    assert mra.leer_clic(estado) == (-3.34, 37.05)
    for roto in (None, {}, {'bounds': None, 'zoom': 3}, {'bounds': {}, 'zoom': 3},
                 {'bounds': {'_southWest': {'lat': None, 'lng': None},
                             '_northEast': {'lat': None, 'lng': None}}, 'zoom': 12},
                 {'bounds': estado['bounds'], 'zoom': None},
                 {'bounds': {'_southWest': {'lat': 37.1, 'lng': -3.3},
                             '_northEast': {'lat': 37.0, 'lng': -3.4}}, 'zoom': 12}):
        assert mra.leer_vista(roto) is None, roto
    for roto in (None, {}, {'last_clicked': None}, {'last_clicked': {'lat': None, 'lng': None}}):
        assert mra.leer_clic(roto) is None, roto
    # la vista de partida está centrada donde se pide y mide lo que una pantalla grande
    o, s, e, n = mra.vista_inicial([37.05, -3.34], 14, 1600, 900)
    assert abs((o + e) / 2 + 3.34) < 1e-9 and s < 37.05 < n
    ancho_m = (mra._a_mercator(e, n)[0] - mra._a_mercator(o, s)[0])
    assert abs(ancho_m - 1600 * mra.RES_ZOOM0 / 2 ** 14) < 1e-6


def test_el_guion_de_las_capas_les_pone_su_imagen_sin_tocar_lo_encendido():
    import re
    limites = [[37.04, -3.36], [37.06, -3.33]]
    js = mra.guion_capas({'clase': ('data:image/png;base64,AAA', limites),
                          'pmax': ('data:image/png;base64,BBB', limites), 'px': None}, 0.5)
    datos = json.loads(re.search(r'var datos = (\{.*?\});\n', js, re.S).group(1))
    assert datos == {'clase': ['data:image/png;base64,AAA', limites],
                     'pmax': ['data:image/png;base64,BBB', limites]}      # sin la que no hay
    assert 'setOpacity(0.5)' in js and 'var elegida = "", nombre = "";' in js
    # las imágenes se cambian siempre; encender y apagar, solo si cambia la categoría elegida
    cuerpo, cambio = js.split('if (c.elegida !== elegida) {')
    assert 'setUrl' in cuerpo and 'setBounds' in cuerpo
    assert 'addLayer' not in cuerpo and 'removeLayer' not in cuerpo
    assert all(x in cambio for x in ('c.control.addOverlay(c.px, nombre)', 'c.control.removeLayer(c.px)',
                                     'c.mapa.addLayer(c.px)', 'c.mapa.addLayer(c.clase)',
                                     'c.elegida = elegida'))
    # las que solo existen si el mapa las tiene (probabilidad de borreguil, pureza) se
    # apagan con cuidado
    assert '["clase", "pmax", "pborr", "pureza"].forEach' in cambio
    assert 'if (c[k]) { c.mapa.removeLayer(c[k]); }' in cambio
    con_pb = mra.guion_capas({'clase': ('a', limites), 'pmax': ('b', limites),
                              'pborr': ('c', limites)}, 0.5)
    assert sorted(json.loads(re.search(r'var datos = (\{.*?\});\n', con_pb, re.S).group(1))) == [
        'clase', 'pborr', 'pmax']
    assert js.count('{') == js.count('}') and js.count('(') == js.count(')')
    # con una categoría elegida: su imagen y su rótulo, bien escritos aunque lleve comillas
    js = mra.guion_capas({'clase': ('a', limites), 'pmax': ('b', limites), 'px': ('c', limites)},
                         0.7, 'p2', 'Probabilidad de «seco» "x"')
    assert sorted(json.loads(re.search(r'var datos = (\{.*?\});\n', js, re.S).group(1))) == [
        'clase', 'pmax', 'px']
    elegida, nombre = re.search(
        r'var elegida = ("(?:[^"\\]|\\.)*"), nombre = ("(?:[^"\\]|\\.)*");', js).groups()
    assert json.loads(elegida) == 'p2' and json.loads(nombre) == 'Probabilidad de «seco» "x"'
    # el registro apunta el mapa, el recuadro y las tres capas con el nombre que se le dé
    reg = mra.guion_registro('map_1', 'ctl_2', {'clase': 'io_3', 'pmax': 'io_4', 'px': 'io_5'})
    assert reg == ('window.capas_mrf = {mapa: map_1, control: ctl_2, clase: io_3, pmax: io_4, '
                   'px: io_5, elegida: ""};')
    # la imagen de partida es un punto transparente
    import base64
    import io

    from PIL import Image
    vacio = np.array(Image.open(io.BytesIO(base64.b64decode(mra.PNG_VACIO.split(',', 1)[1]))))
    assert vacio.shape == (1, 1, 4) and vacio[0, 0, 3] == 0


def test_la_rampa_de_probabilidad_va_de_oscuro_a_claro():
    r = mra.pintar_probabilidad(np.array([[0.0, 0.5, 1.0, np.nan]], dtype='float32'))
    assert tuple(r[0, 0][:3]) == (0x44, 0x01, 0x54) and tuple(r[0, 2][:3]) == (0xFD, 0xE7, 0x25)
    assert tuple(r[0, 1][:3]) == (0x21, 0x91, 0x8C) and r[0, 3, 3] == 0
    luz = [int(r[0, k, :3].astype(int).sum()) for k in range(3)]
    assert luz[0] < luz[1] < luz[2]


def test_cada_categoria_tiene_su_color_y_no_se_repiten_dentro_de_un_nivel():
    for nivel, cats in mr.NIVELES.items():
        usados = [mra.color(c) for c in (mr.AUSENCIA,) + tuple(cats)]
        assert len(set(usados)) == len(usados), nivel
        assert all(c in mra.COLORES for c in cats), nivel
    assert mra.color('otra cosa', 1) not in mra.COLORES.values()


def test_la_imagen_png_del_visor_es_la_misma_imagen():
    import base64
    import io

    from PIL import Image
    rgba = np.zeros((7, 9, 4), dtype='uint8')
    rgba[2:5, 3:8] = (0xD5, 0x5E, 0x00, 255)
    rgba[0, 0] = (1, 2, 3, 128)
    url = mra.a_png(rgba)
    assert url.startswith('data:image/png;base64,')
    vuelta = np.array(Image.open(io.BytesIO(base64.b64decode(url.split(',', 1)[1]))))
    assert vuelta.shape == rgba.shape and np.array_equal(vuelta, rgba)
    rapida = mra.a_png(rgba, rapido=True)               # comprime menos; la imagen es la misma
    otra = np.array(Image.open(io.BytesIO(base64.b64decode(rapida.split(',', 1)[1]))))
    assert rapida.startswith('data:image/png;base64,') and np.array_equal(otra, rgba)
    # folium la acepta tal cual, sin volver a codificarla
    from folium.utilities import image_to_url
    assert image_to_url(url) == url


def test_consultar_varios_puntos_es_lo_mismo_que_consultarlos_uno_a_uno():
    d = carpeta()
    zona = box(X0 + 200, Y0 - 600, X0 + 1000, Y0 - 100)
    mra.hacer_mapa(MOD, REJ, lector(rej=REJ), d, zona_utm=zona)
    puntos = [(m['lon'], m['lat']) for m in MOD['muestras']]
    puntos.append(A_GEO.transform(X0 - 500, Y0 - 50))                 # fuera del mapa
    varios = mra.consultar_varios(d, puntos)
    assert len(varios) == len(puntos) and varios[-1] is None
    assert varios == [mra.consultar(d, lon, lat) for lon, lat in puntos]
    con_dato = [q for q in varios if q and q['codigo'] is not None]
    sin_dato = [q for q in varios if q and q['codigo'] is None]
    assert con_dato and sin_dato                    # hay muestras dentro y fuera de la zona
    assert mra.consultar_varios(carpeta(), puntos[:2]) == [None, None]      # sin mapa
    limpiar(d)


# ------------------------------------------------- lectura de bloques (Earth Engine)
def test_leer_bloque_parte_el_bloque_si_no_llega_y_lo_recompone_en_su_sitio():
    import ee
    bandas = ['ndmi_mean', 'elev_dem_m']
    pedidos = []

    def falso(peticion):
        g = peticion['grid']
        w, h = g['dimensions']['width'], g['dimensions']['height']
        pedidos.append((w, h))
        if w * h > 1500:                                   # «demasiado grande»
            raise RuntimeError('Computation timed out.')
        x, y = g['affineTransform']['translateX'], g['affineTransform']['translateY']
        assert peticion['bandIds'] == bandas and g['crsCode'] == 'EPSG:32630'
        assert g['affineTransform']['scaleX'] == 10 and g['affineTransform']['scaleY'] == -10
        arr = np.zeros((h, w), dtype=[(b, 'float32') for b in bandas])
        for f in range(h):
            for c in range(w):
                arr['ndmi_mean'][f, c] = (x + c * 10) / 1000 + (y - f * 10) / 1e7
                arr['elev_dem_m'][f, c] = geb.SIN_DATO_BLOQUE if (c + f) % 7 == 0 else 2000 + c
        return arr

    original, plazo = ee.data.computePixels, ee.data.setDeadline
    ee.data.computePixels, ee.data.setDeadline = falso, (lambda ms: None)
    try:
        datos = geb.leer_bloque(object(), 32630, X0, Y0, 70, 50, bandas, espera=0)
        entero = geb.leer_bloque(object(), 32630, X0, Y0, 30, 40, bandas, espera=0)
    finally:
        ee.data.computePixels, ee.data.setDeadline = original, plazo
    assert datos.shape == (50, 70, 2) and datos.dtype == np.float32
    assert pedidos[0] == (70, 50) and (35, 25) in pedidos and pedidos[-1] == (30, 40)
    for f, c in ((0, 0), (10, 34), (10, 35), (24, 69), (25, 0), (49, 69), (30, 41)):
        esperado = np.float32((X0 + c * 10) / 1000 + (Y0 - f * 10) / 1e7)
        assert datos[f, c, 0] == esperado, (f, c)
    assert np.isnan(datos[0, 0, 1]) and np.isnan(datos[3, 4, 1]) and datos[1, 1, 1] == 2001
    assert entero.shape == (40, 30, 2)

    # Si ni por partes llega, no devuelve un bloque a medias.
    ee.data.computePixels = lambda p: (_ for _ in ()).throw(RuntimeError('Computation timed out.'))
    ee.data.setDeadline = lambda ms: None
    try:
        geb.leer_bloque(object(), 32630, X0, Y0, 64, 64, bandas, espera=0)
    except RuntimeError as e:
        assert 'bloque' in str(e)
    else:
        raise AssertionError('debía fallar')
    finally:
        ee.data.computePixels, ee.data.setDeadline = original, plazo


if __name__ == '__main__':
    pruebas = [(n, f) for n, f in sorted(globals().items())
               if n.startswith('test_') and callable(f)]
    fallos = 0
    for nombre, f in pruebas:
        try:
            f()
            print(f'  ok     {nombre}')
        except Exception as e:                      # noqa: BLE001
            fallos += 1
            # con la línea que ha fallado: un «assert» sin mensaje no dice cuál es
            import traceback
            paso = traceback.extract_tb(e.__traceback__)[-1]
            print(f'  FALLO  {nombre}: {type(e).__name__}: {e}'
                  f'  [línea {paso.lineno}: {paso.line}]')
    print(f'{len(pruebas) - fallos}/{len(pruebas)} pruebas correctas')
    print(f'  (GDAL ha leído lo mismo en los {len(COTEJADOS)} GeoTIFF abiertos)' if GDAL
          else '  (en este equipo no se puede cargar rasterio: sin cotejar con GDAL)')
    sys.exit(1 if fallos else 0)
