"""Pruebas del ambiente por cercanía a lagunas y cauces (mapeo_ambiente.py).

Se ejecutan con `python tests/test_mapeo_ambiente.py`; no necesitan red. Los relieves
son inventados: vaguadas y hoyos de los que se sabe por dónde tiene que correr el agua.
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import borreguil_pipeline as bp   # noqa: E402
import mapeo_ambiente as ma       # noqa: E402
import mapeo_rf as mr             # noqa: E402

bp._imports()

REJ = {'epsg': 32630, 'escala': 10, 'x0': 467_000.0, 'y0': 4_102_000.0, 'ancho': 60, 'alto': 40}


def vaguada(alto=120, ancho=81, eje=40):
    """Una vaguada en V que baja hacia el sur: 2 m por píxel a lo largo y 3 m por píxel
    hacia el eje."""
    ff, cc = np.mgrid[0:alto, 0:ancho]
    return 3000.0 - 2.0 * ff + 3.0 * np.abs(cc - eje)


# ------------------------------------------------------------------ la regla
def test_los_valores_de_partida_son_los_de_las_reglas_de_la_app():
    q = ma.params()
    assert q == {'laguna_capa_m': 120.0, 'laguna_agua_m': 100.0, 'arroyo_m': 30.0,
                 'cauce_ha': 8.0}
    # las mismas distancias con que la app propone el tipo de un punto
    assert q['laguna_capa_m'] == bp.HIER_THRESH['dist_laguna_m']
    assert q['arroyo_m'] == bp.HIER_THRESH['dist_arroyo_m']
    assert ma.NDWI_LAGUNA == bp.HIER_THRESH['ndwi_laguna']
    # los códigos, los del nivel «ambiente»
    assert ma.CODIGOS == {c: i for c, i in mr.codigos('ambiente').items() if i}
    assert ma.params({'arroyo_m': '50', 'otra': 1})['arroyo_m'] == 50.0
    for malo in ({'arroyo_m': -1}, {'laguna_capa_m': 5000}, {'cauce_ha': 0},
                 {'laguna_agua_m': 'mucho'}, {'arroyo_m': float('nan')}):
        try:
            ma.params(malo)
        except ValueError:
            pass
        else:
            raise AssertionError(f'debía rechazar {malo}')


def test_primero_laguna_despues_arroyo_y_el_resto_ladera():
    inf = np.inf
    #                  agua   capa   cauce
    casos = np.array([[50,    inf,   inf],      # junto al agua → laguna
                      [inf,   110,   inf],      # junto a una laguna de la capa → laguna
                      [80,    inf,   5],        # laguna Y cauce → laguna (manda la laguna)
                      [300,   400,   20],       # junto a un cauce → arroyo
                      [300,   400,   200],      # lejos de todo → ladera
                      [inf,   inf,   inf],      # no hay ni lagunas ni cauces → ladera
                      [100,   inf,   inf],      # justo en el límite → laguna
                      [inf,   120,   inf],
                      [inf,   inf,   30],       # justo en el límite → arroyo
                      [100.5, 120.5, 30.5]],    # un poco más → ladera
                     dtype='float32')
    amb = ma.decidir(casos[:, 0], casos[:, 1], casos[:, 2])
    L, A, D = ma.CODIGOS['laguna'], ma.CODIGOS['arroyo'], ma.CODIGOS['ladera']
    assert list(amb) == [L, L, L, A, D, D, L, L, A, D] and amb.dtype == np.uint8
    # con otras distancias, otro resultado; y vale con matrices de cualquier forma
    otras = ma.decidir(casos[:, 0].reshape(2, 5), casos[:, 1].reshape(2, 5),
                       casos[:, 2].reshape(2, 5),
                       {'laguna_agua_m': 40, 'laguna_capa_m': 40, 'arroyo_m': 250})
    assert otras.shape == (2, 5)
    assert list(otras.ravel()) == [D, D, A, A, A, D, D, D, A, A]
    assert 'laguna' in ma.frase() and '120' in ma.frase() and '30' in ma.frase()
    assert 'lake' in ma.frase(en=True) and 'laguna' not in ma.frase(en=True)


# ------------------------------------------------------------------ lagunas
def test_el_agua_de_laguna_es_ndwi_alto_en_llano_y_en_manchas():
    ndwi = np.full((20, 30), -0.2, dtype='float32')
    pend = np.full((20, 30), 5.0, dtype='float32')
    ndwi[2:6, 3:8] = 0.3                         # una laguna: 20 píxeles en llano
    ndwi[10:14, 20:25] = 0.3
    pend[10:14, 20:25] = 40.0                    # «agua» en una pared: sombra o nieve
    ndwi[16, 5] = ndwi[16, 6] = 0.3              # dos píxeles sueltos
    ndwi[8, 12] = 0.03                           # NDWI positivo pero bajo
    ndwi[18, 28] = np.nan
    cand = ma.candidata_a_agua(ndwi, pend)
    assert cand.sum() == 22 and not cand[10:14, 20:25].any() and not cand[8, 12]
    agua = ma.agua_de_laguna(cand)
    assert agua.sum() == 20 and agua[2:6, 3:8].all() and not agua[16, 5]
    assert ma.agua_de_laguna(cand, minimo=1).sum() == 22
    # tres píxeles que solo se tocan en diagonal también son una mancha
    diag = np.zeros((6, 6), bool)
    diag[1, 1] = diag[2, 2] = diag[3, 3] = True
    assert ma.agua_de_laguna(diag).sum() == 3
    assert not ma.agua_de_laguna(np.zeros((4, 4), bool)).any()


# ------------------------------------------------------------------ distancias
def test_la_distancia_va_en_metros_y_sin_nada_cerca_es_infinita():
    m = np.zeros((10, 20), bool)
    m[4, 5] = True
    d = ma.distancia(m)
    assert d[4, 5] == 0 and d[4, 8] == 30 and d[0, 5] == 40 and d.dtype == np.float32
    assert abs(d[7, 9] - 50) < 1e-4                           # 3 y 4 píxeles: 50 m
    assert np.isinf(ma.distancia(np.zeros((3, 3), bool))).all()
    assert ma.distancia(m, escala=20)[4, 8] == 60


def test_la_distancia_a_los_puntos_es_exacta_tambien_si_caen_fuera():
    x0, y0 = REJ['x0'], REJ['y0']
    dentro = (x0 + 305.0, y0 - 205.0)            # centro del píxel (col 30, fila 20)
    fuera = (x0 - 95.0, y0 - 5.0)                # 100 m al oeste del primer píxel
    d = ma.distancia_a_puntos([dentro, fuera], REJ)
    assert d.shape == (40, 60) and d.dtype == np.float32
    assert d[20, 30] == 0 and abs(d[20, 33] - 30) < 1e-3 and abs(d[23, 34] - 50) < 1e-3
    assert abs(d[0, 0] - 100) < 1e-3 and abs(d[0, 2] - 120) < 1e-3   # manda el de fuera
    # un punto que no está en el centro de un píxel: la distancia es al punto, no al píxel
    d2 = ma.distancia_a_puntos([(x0 + 300.0, y0 - 200.0)], REJ)
    assert abs(d2[20, 30] - np.hypot(5, 5)) < 1e-3
    # más lejos del alcance ya no se mide; y sin puntos, nada cerca
    corto = ma.distancia_a_puntos([dentro], REJ, hasta_m=100)
    assert np.isinf(corto[20, 45]) and abs(corto[20, 39] - 90) < 1e-3
    assert np.isinf(ma.distancia_a_puntos([], REJ)).all()
    en = ma.puntos_en_rejilla([dentro, fuera], REJ)
    assert en.sum() == 1 and en[20, 30]


# ------------------------------------------------------------------ cauces del relieve
def test_el_agua_baja_por_la_vaguada_y_el_cauce_es_su_eje():
    dem = vaguada()
    ha = ma.superficie_que_vierte(dem)
    assert ha.shape == dem.shape and np.nanmin(ha) == 0.01            # cada celda, 100 m²
    # en el eje, cuanto más abajo, más agua; y fuera del eje casi nada
    eje = ha[1:-1, 40]
    assert (np.diff(eje) > 0).all() and eje[-1] > 50
    assert ha[1:-1, 10].max() < 1 and ha[1:-1, 70].max() < 1
    cauce = ma.cauces_del_relieve(dem, 8.0)
    assert cauce[40:-1, 40].all() and not cauce[:10, 40].any()        # empieza con 8 ha
    assert not np.delete(cauce, [39, 40, 41], axis=1).any()           # y solo es el eje
    # con menos superficie el cauce empieza más arriba
    assert ma.cauces_del_relieve(dem, 1.0)[:, 40].sum() > cauce[:, 40].sum()


def test_un_hoyo_no_corta_el_cauce():
    """Una laguna es un hoyo llano: el agua tiene que cruzarla y seguir por su desagüe
    con todo lo que traía."""
    dem = vaguada()
    dem[50:60, 36:45] = dem[50:60, 36:45].min() - 5.0
    lleno = ma.rellenar_hoyos(dem)
    assert (lleno >= dem).all()                                        # solo se rellena
    assert (lleno[50:60, 36:45] > dem[50:60, 36:45]).all()
    fuera = np.ones(dem.shape, bool)
    fuera[50:60, 36:45] = False
    assert np.array_equal(lleno[fuera], dem[fuera])                    # y solo el hoyo
    assert np.ptp(lleno[50:60, 36:45]) < 1e-6                          # queda (casi) llano
    ha = ma.superficie_que_vierte(dem)
    arriba, abajo = ha[45, 40], ha[70, 40]
    assert abajo > arriba + 2.0, (arriba, abajo)      # debajo lleva lo de arriba y más
    assert ma.cauces_del_relieve(dem, 8.0)[62:-1, 40].all()
    # sin rellenar, el agua se quedaría en el hoyo: por debajo empezaría de cero
    assert ha[61, 40] > ha[49, 40]


def test_lo_que_llega_al_borde_sale_y_no_se_pierde_agua():
    dem = vaguada(60, 41, 20)
    ha = ma.superficie_que_vierte(dem)
    # todo lo que cae dentro acaba saliendo por alguna celda del borde
    borde = np.zeros(dem.shape, bool)
    borde[0, :] = borde[-1, :] = borde[:, 0] = borde[:, -1] = True
    # las celdas del borde no reciben de otras del borde: ninguna hace de cauce falso
    assert ha[-1, [0, 5, 10, 30, 40]].max() < 1.0
    assert abs(ha[-2, 20] - ha[-1, 20]) < ha[-1, 20]                   # el eje sí llega
    # donde falta el relieve no hay resultado, y lo demás sigue saliendo
    con_hueco = dem.copy()
    con_hueco[20:25, 5:9] = np.nan
    ha2 = ma.superficie_que_vierte(con_hueco)
    assert np.isnan(ha2[20:25, 5:9]).all() and np.isfinite(ha2[~np.isnan(con_hueco)]).all()
    assert not ma.cauces_del_relieve(con_hueco, 8.0)[20:25, 5:9].any()
    # una sola celda, o todo sin dato: no rompe
    assert ma.superficie_que_vierte(np.array([[100.0]]))[0, 0] == 0.01
    assert np.isnan(ma.superficie_que_vierte(np.full((3, 4), np.nan))).all()


def test_el_calculo_del_relieve_sale_igual_cada_vez_y_no_tarda():
    import time
    azar = np.random.default_rng(1)
    y, x = np.mgrid[0:400, 0:500]
    dem = 2500 + 300 * np.sin(x / 90.0) * np.cos(y / 70.0) + 0.3 * y + azar.normal(0, 0.5, y.shape)
    t0 = time.time()
    a = ma.superficie_que_vierte(dem)
    tarda = time.time() - t0
    assert np.array_equal(a, ma.superficie_que_vierte(dem))
    assert tarda < 20, tarda                    # 20 km² en segundos
    assert np.nanmax(a) > 100                   # el ruido no impide que el agua se junte


# ------------------------------------------------------------------ capa de cauces
def test_una_capa_de_cauces_se_lee_y_se_pinta_en_la_rejilla():
    import pyproj
    a_geo = pyproj.Transformer.from_crs(32630, 4326, always_xy=True)
    x0, y0 = REJ['x0'], REJ['y0']
    linea = [list(a_geo.transform(x0 + 55, y0 - 55)), list(a_geo.transform(x0 + 455, y0 - 55))]
    d = Path(tempfile.mkdtemp(prefix='cauces_'))
    try:
        (d / 'cauces.geojson').write_text(json.dumps({'type': 'FeatureCollection', 'features': [
            {'type': 'Feature', 'properties': {'nombre': 'río de prueba'},
             'geometry': {'type': 'LineString', 'coordinates': linea}},
            {'type': 'Feature', 'properties': {},
             'geometry': {'type': 'Point', 'coordinates': linea[0]}}]}), encoding='utf-8')
        geom = ma.leer_cauces(d / 'cauces.geojson')
        assert geom.geom_type in ('LineString', 'MultiLineString', 'GeometryCollection')
        import mapeo_raster as mra
        pintado = ma.rasterizar(mra.a_utm(geom, 32630), REJ)
        assert pintado.shape == (40, 60) and pintado[5, 5:46].all()   # la fila que pisa
        assert pintado.sum() <= 2 * 41 and not pintado[20].any()
        # una capa solo de puntos no vale, y se dice
        (d / 'puntos.geojson').write_text(json.dumps({'type': 'FeatureCollection', 'features': [
            {'type': 'Feature', 'properties': {},
             'geometry': {'type': 'Point', 'coordinates': linea[0]}}]}), encoding='utf-8')
        try:
            ma.leer_cauces(d / 'puntos.geojson')
        except ValueError as e:
            assert 'No hay líneas' in str(e)
        else:
            raise AssertionError('una capa de puntos no son cauces')
    finally:
        shutil.rmtree(d, ignore_errors=True)
    assert not ma.rasterizar(None, REJ).any()


def test_la_rejilla_ampliada_conserva_los_pixeles_en_su_sitio():
    g = ma.ampliar(REJ, 100)
    assert (g['ancho'], g['alto']) == (260, 240)
    assert g['x0'] == REJ['x0'] - 1000 and g['y0'] == REJ['y0'] + 1000
    assert g['epsg'] == REJ['epsg'] and g['escala'] == 10
    assert ma.ampliar(REJ, 0) == REJ


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
            print(f'  FALLO  {nombre}: {type(e).__name__}: {e}')
    print(f'{len(pruebas) - fallos}/{len(pruebas)} pruebas correctas')
    sys.exit(1 if fallos else 0)
