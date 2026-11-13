"""Pruebas de los GeoTIFF del mapa sin GDAL (mapeo_tif.py).

Se ejecutan con `python tests/test_mapeo_tif.py`; no necesitan red. Lo que se escribe se
vuelve a leer con esta misma pieza, con Pillow (otro lector, independiente) y, si en el
equipo se puede cargar rasterio, también con GDAL, que es quien manda en QGIS: tiene que
ver la misma rejilla, el mismo sistema de coordenadas, el mismo sin dato, los mismos
nombres, etiquetas y colores, y los mismos valores.
"""
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mapeo_tif as mt            # noqa: E402

REJ = {'epsg': 32630, 'escala': 10, 'x0': 467_000.0, 'y0': 4_102_000.0, 'ancho': 520, 'alto': 300}
RNG = np.random.default_rng(7)
CLASES = RNG.integers(0, 4, (300, 520), dtype='uint8')
CLASES[:12, :9] = 255
PROB = RNG.random((3, 300, 520), dtype='float32')
PROB[:, :12, :9] = np.nan
COLORES = {0: (158, 158, 158), 1: (22, 140, 115), 2: (217, 170, 55), 3: (40, 117, 186)}


def carpeta():
    return Path(tempfile.mkdtemp(prefix='tif_'))


def limpiar(d):
    shutil.rmtree(d, ignore_errors=True)


def gdal():
    """rasterio, si en este equipo se puede cargar; None si no (Windows lo bloquea en
    algunos). Las comprobaciones con GDAL se saltan entonces, y se dice."""
    try:
        import rasterio
        return rasterio
    except Exception:                               # noqa: BLE001
        return None


def test_lo_que_se_escribe_se_lee_igual_con_su_rejilla_y_todo_lo_demas():
    d = carpeta()
    mt.escribir(d / 'clase.tif', REJ, CLASES, sin_dato=255, paleta=COLORES,
                etiquetas={'CATEGORIAS': '0 = no borreguil; 1 = arroyo · húmedo', 'UMBRAL': '0.55'},
                nombres=['categoria (codigo)'])
    with mt.abrir(d / 'clase.tif') as f:
        assert (f.ancho, f.alto, f.bandas, f.tipo) == (520, 300, 1, 'uint8')
        assert (f.epsg, f.escala, f.x0, f.y0) == (32630, 10.0, 467_000.0, 4_102_000.0)
        assert f.sin_dato == 255 and isinstance(f.sin_dato, int)
        assert f.etiquetas == {'CATEGORIAS': '0 = no borreguil; 1 = arroyo · húmedo',
                               'UMBRAL': '0.55'}
        assert f.nombres == ('categoria (codigo)',) and f.reducidas == []
        assert f.paleta[2] == (217, 170, 55, 255) and f.paleta[200] == (0, 0, 0, 255)
        assert f.paleta[255] == (0, 0, 0, 0)        # el sin dato, transparente (como en GDAL)
        assert f.limites == (467_000.0, 4_099_000.0, 472_200.0, 4_102_000.0)
        assert f.rej == REJ
        assert np.array_equal(f.leer(1), CLASES) and f.leer(1).dtype == np.uint8
        assert f.leer().shape == (1, 300, 520)
    mt.escribir(d / 'prob.tif', REJ, PROB, sin_dato=float('nan'),
                nombres=['P(no borreguil)', 'P(húmedo)', 'P(seco)'])
    with mt.abrir(d / 'prob.tif') as f:
        assert (f.bandas, f.tipo) == (3, 'float32') and np.isnan(f.sin_dato) and f.paleta is None
        assert f.nombres == ('P(no borreguil)', 'P(húmedo)', 'P(seco)') and f.etiquetas == {}
        assert np.array_equal(f.leer(), PROB, equal_nan=True)
        assert np.array_equal(f.leer(2), PROB[1], equal_nan=True)
    limpiar(d)


def test_se_puede_leer_cualquier_ventana_sin_leer_el_fichero_entero():
    d = carpeta()
    mt.escribir(d / 'prob.tif', REJ, PROB, sin_dato=float('nan'))
    with mt.abrir(d / 'prob.tif') as f:
        # ventanas que cruzan teselas, en las esquinas y de un solo píxel
        for col, fila, ancho, alto in ((0, 0, 1, 1), (250, 250, 12, 12), (500, 290, 20, 10),
                                       (255, 0, 2, 300), (0, 255, 520, 2), (519, 299, 1, 1)):
            trozo = PROB[:, fila:fila + alto, col:col + ancho]
            assert np.array_equal(f.leer(ventana=(col, fila, ancho, alto)), trozo, equal_nan=True)
            assert np.array_equal(f.leer(3, (col, fila, ancho, alto)), trozo[2], equal_nan=True)
        assert f.valor(299, 519, 2) == PROB[1, 299, 519]
        # un píxel suelto solo descomprime su tesela
        f._teselas.clear()
        f.valor(10, 300)
        assert len(f._teselas) == 1
        for mala in ((-1, 0, 5, 5), (0, 0, 521, 1), (0, 299, 1, 2), (5, 5, 0, 1)):
            try:
                f.leer(1, mala)
                raise AssertionError(f'ha leído la ventana {mala}')
            except ValueError:
                pass
        try:
            f.leer(4)
            raise AssertionError('ha leído una banda que no existe')
        except ValueError:
            pass
    limpiar(d)


def test_el_pixel_de_un_punto_y_el_centro_de_un_pixel():
    d = carpeta()
    mt.escribir(d / 'clase.tif', REJ, CLASES, sin_dato=255)
    with mt.abrir(d / 'clase.tif') as f:
        assert f.indice(467_000.0, 4_102_000.0) == (0, 0)             # la esquina es del primero
        assert f.indice(467_009.99, 4_101_990.01) == (0, 0)
        assert f.indice(467_010.0, 4_101_990.0) == (1, 1)             # el borde, del siguiente
        assert f.indice(466_999.9, 4_102_000.1) == (-1, -1)           # fuera: no se recorta
        assert f.centro(0, 0) == (467_005.0, 4_101_995.0)
        assert f.indice(*f.centro(123, 456)) == (123, 456)
    limpiar(d)


def test_se_escribe_por_ventanas_y_lo_no_escrito_queda_sin_dato():
    d = carpeta()
    with mt.Escritor(d / 'clase.tif', REJ, tipo='uint8', sin_dato=255) as f:
        f.escribir(CLASES[:256, :256], 1, ventana=(0, 0, 256, 256))
        f.escribir(CLASES[256:, 256:512], 1, ventana=(256, 256, 256, 44))
    esperado = np.full((300, 520), 255, dtype='uint8')
    esperado[:256, :256] = CLASES[:256, :256]
    esperado[256:, 256:512] = CLASES[256:, 256:512]
    with mt.abrir(d / 'clase.tif') as f:
        assert np.array_equal(f.leer(1), esperado)
    # varias bandas: todas de una vez (bandas, alto, ancho) o de una en una
    with mt.Escritor(d / 'prob.tif', REJ, bandas=3, sin_dato=float('nan')) as f:
        f.escribir(PROB[:, :100, :200], ventana=(0, 0, 200, 100))
        f.escribir(PROB[1, 100:, :], 2, ventana=(0, 100, 520, 200))
    with mt.abrir(d / 'prob.tif') as f:
        m = f.leer()
        assert np.array_equal(m[:, :100, :200], PROB[:, :100, :200], equal_nan=True)
        assert np.array_equal(m[1, 100:], PROB[1, 100:], equal_nan=True)
        assert np.isnan(m[0, 100:]).all() and np.isnan(m[2, :100, 200:]).all()
    # una matriz que no cabe en su ventana se rechaza; y solo uint8 o float32
    f = mt.Escritor(d / 'x.tif', REJ, bandas=3)
    for mala, banda in ((PROB[0, :10, :10], 1), (PROB[:2, :5, :5], None), (PROB[0, :5, :5], None)):
        try:
            f.escribir(mala, banda, ventana=(0, 0, 5, 5))
            raise AssertionError('ha admitido una matriz que no cabe')
        except ValueError:
            pass
    f.descartar()
    assert not (d / 'x.tif').exists()
    try:
        mt.Escritor(d / 'y.tif', REJ, tipo='int16')
        raise AssertionError('ha admitido int16')
    except ValueError:
        pass
    limpiar(d)


def test_si_algo_falla_a_medias_no_queda_ningun_fichero():
    d = carpeta()
    try:
        with mt.Escritor(d / 'clase.tif', REJ, tipo='uint8', sin_dato=255) as f:
            f.escribir(CLASES)
            raise RuntimeError('se ha interrumpido (simulado)')
    except RuntimeError:
        pass
    assert list(d.iterdir()) == []
    limpiar(d)


def test_los_rasteres_grandes_van_a_un_fichero_temporal_y_salen_igual():
    d = carpeta()
    de_verdad = mt.EN_MEMORIA_HASTA
    mt.EN_MEMORIA_HASTA = 1000                       # todo es «grande»
    try:
        f = mt.Escritor(d / 'prob.tif', REJ, bandas=3, sin_dato=float('nan'))
        assert (d / 'prob.tif.parte').exists()
        f.escribir(PROB[:, :, :300], ventana=(0, 0, 300, 300))
        f.cerrar()
        assert not (d / 'prob.tif.parte').exists()   # el temporal desaparece
        with mt.abrir(d / 'prob.tif') as g:
            m = g.leer()
            assert np.array_equal(m[:, :, :300], PROB[:, :, :300], equal_nan=True)
            assert np.isnan(m[:, :, 300:]).all()
        f = mt.Escritor(d / 'otro.tif', REJ, tipo='uint8', sin_dato=255)
        f.descartar()
        assert sorted(p.name for p in d.iterdir()) == ['prob.tif']
    finally:
        mt.EN_MEMORIA_HASTA = de_verdad
    limpiar(d)


def test_reducir_categoria_mas_frecuente_o_media_sin_contar_lo_que_no_tiene_dato():
    m = np.array([[1, 1, 2, 2, 3],
                  [1, 2, 2, 255, 3],
                  [255, 255, 0, 0, 255],
                  [255, 255, 0, 1, 255]], dtype='uint8')
    r = mt.reducir(m, 2, categorica=True, sin_dato=255)
    # 1 gana 3 a 1; 2 gana 3 a 0; la columna que sobra, con lo que tiene; un bloque sin
    # ningún dato queda sin dato; y sin contar el sin dato aunque sea mayoría
    assert r.tolist() == [[1, 2, 3], [255, 0, 255]] and r.dtype == np.uint8
    # en el empate gana el código menor, sea cual sea el orden en que aparecen
    assert mt.reducir(np.array([[2, 1], [1, 2]], 'uint8'), 2, True, 255).tolist() == [[1]]
    # sin valor «sin dato», el 0 cuenta como una categoría más
    assert mt.reducir(np.array([[0, 0], [0, 3]], 'uint8'), 2, True).tolist() == [[0]]
    # con el 0 como «nada» (lagunas y cauces), una línea fina no desaparece de lejos
    linea = np.zeros((4, 4), 'uint8')
    linea[1, :] = 2
    assert mt.reducir(linea, 4, True, sin_dato=0).tolist() == [[2]]
    p = np.array([[0.2, 0.4, np.nan, np.nan],
                  [0.6, np.nan, np.nan, np.nan]], dtype='float32')
    r = mt.reducir(p, 2, categorica=False)
    assert r.dtype == np.float32 and abs(r[0, 0] - 0.4) < 1e-6 and np.isnan(r[0, 1])
    assert mt.factores_reducidos(120, 80) == [] and mt.factores_reducidos(511, 300) == []
    assert mt.factores_reducidos(512, 10) == [2] and mt.factores_reducidos(1120, 800) == [2, 4]


def test_las_versiones_reducidas_se_anaden_sin_tocar_los_datos():
    d = carpeta()
    rej = dict(REJ, ancho=1120, alto=800)
    grande = np.kron(RNG.integers(0, 3, (20, 28)), np.ones((40, 40))).astype('uint8')
    grande[:40, :40] = 255
    mt.escribir(d / 'clase.tif', rej, grande, sin_dato=255, paleta=COLORES,
                etiquetas={'CATEGORIAS': 'x'})
    assert mt.anadir_reducidas(d / 'clase.tif', categorica=True) == [2, 4]
    assert mt.anadir_reducidas(d / 'clase.tif', categorica=True) == [2, 4]     # ya las tiene
    with mt.abrir(d / 'clase.tif') as f:
        assert f.reducidas == [2, 4] and np.array_equal(f.leer(1), grande)
        assert f.forma() == (800, 1120) and f.forma(0) == (400, 560) and f.forma(1) == (200, 280)
        assert np.array_equal(f.leer(1, nivel=0), mt.reducir(grande, 2, True, 255))
        assert np.array_equal(f.leer(1, nivel=1), grande[::4, ::4])     # manchas de 40: exacto
        assert np.array_equal(f.leer(1, (10, 20, 30, 5), nivel=1), grande[80:100:4, 40:160:4])
        assert f.etiquetas == {'CATEGORIAS': 'x'} and f.paleta[1] == (22, 140, 115, 255)
    prob = RNG.random((2, 800, 1120), dtype='float32')
    prob[:, 300:400, 500:700] = np.nan
    mt.escribir(d / 'prob.tif', rej, prob, sin_dato=float('nan'))
    assert mt.anadir_reducidas(d / 'prob.tif', categorica=False) == [2, 4]
    with mt.abrir(d / 'prob.tif') as f:
        assert np.array_equal(f.leer(), prob, equal_nan=True)
        r = f.leer(nivel=1)
        assert r.shape == (2, 200, 280) and np.isnan(r[:, 75:100, 125:175]).all()
        assert abs(float(r[1, 0, 0]) - float(prob[1, :4, :4].mean())) < 1e-6
    # un ráster pequeño no las necesita: se queda como estaba
    mt.escribir(d / 'chico.tif', dict(REJ, ancho=120, alto=80), CLASES[:80, :120], sin_dato=255)
    antes = (d / 'chico.tif').read_bytes()
    assert mt.anadir_reducidas(d / 'chico.tif', categorica=True) == []
    assert (d / 'chico.tif').read_bytes() == antes
    limpiar(d)


def test_pillow_que_es_otro_lector_ve_lo_mismo():
    from PIL import Image
    d = carpeta()
    mt.escribir(d / 'clase.tif', REJ, CLASES, sin_dato=255, paleta=COLORES)
    mt.escribir(d / 'p.tif', REJ, PROB[0], sin_dato=float('nan'))
    with Image.open(d / 'clase.tif') as im:
        assert im.mode == 'P' and im.size == (520, 300) and np.array_equal(np.asarray(im), CLASES)
        assert im.getpalette()[3:6] == [22, 140, 115]
    with Image.open(d / 'p.tif') as im:
        assert im.mode == 'F' and np.array_equal(np.asarray(im), PROB[0], equal_nan=True)
    limpiar(d)


def test_gdal_si_se_puede_cargar_ve_lo_mismo():
    rio = gdal()
    if rio is None:
        print('         (rasterio no se puede cargar en este equipo: sin cotejar con GDAL)')
        return
    d = carpeta()
    rej = dict(REJ, ancho=1120, alto=800)
    grande = np.kron(RNG.integers(0, 3, (20, 28)), np.ones((40, 40))).astype('uint8')
    mt.escribir(d / 'clase.tif', rej, grande, sin_dato=255, paleta=COLORES,
                etiquetas={'CATEGORIAS': '1 = arroyo · húmedo'}, nombres=['categoria (codigo)'])
    mt.anadir_reducidas(d / 'clase.tif', categorica=True)
    mt.escribir(d / 'prob.tif', REJ, PROB, sin_dato=float('nan'), nombres=['a', 'P(húmedo)', 'c'])
    with rio.open(d / 'clase.tif') as f:
        assert f.crs.to_epsg() == 32630 and (f.width, f.height, f.count) == (1120, 800, 1)
        assert (f.transform.a, f.transform.e, f.transform.c, f.transform.f) == \
            (10, -10, 467_000.0, 4_102_000.0)
        assert f.nodata == 255 and f.dtypes[0] == 'uint8' and f.overviews(1) == [2, 4]
        assert f.tags()['CATEGORIAS'] == '1 = arroyo · húmedo'
        assert f.descriptions == ('categoria (codigo)',) and f.colormap(1)[2] == (217, 170, 55, 255)
        assert f.profile['compress'] == 'deflate' and f.block_shapes == [(256, 256)]
        assert np.array_equal(f.read(1), grande)
    with rio.open(d / 'prob.tif') as f:
        assert f.count == 3 and f.dtypes[0] == 'float32' and np.isnan(f.nodata)
        assert f.descriptions == ('a', 'P(húmedo)', 'c')
        assert np.array_equal(f.read(), PROB, equal_nan=True)
    # y al revés: lo que escribe GDAL se lee aquí
    with rio.open(d / 'gdal.tif', 'w', driver='GTiff', width=520, height=300, count=3,
                  dtype='float32', nodata=float('nan'), crs='EPSG:32630',
                  transform=rio.transform.from_origin(467_000.0, 4_102_000.0, 10, 10),
                  tiled=True, blockxsize=256, blockysize=256, compress='deflate') as f:
        f.write(PROB)
        f.set_band_description(2, 'P(húmedo)')
        f.update_tags(UMBRAL='0.55')
    with mt.abrir(d / 'gdal.tif') as f:
        assert f.rej == REJ and np.isnan(f.sin_dato) and f.etiquetas['UMBRAL'] == '0.55'
        assert f.nombres[1] == 'P(húmedo)' and np.array_equal(f.leer(), PROB, equal_nan=True)
    limpiar(d)


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
