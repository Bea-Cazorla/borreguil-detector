"""Subir puntos en la app: un Shapefile son varios ficheros (.shp, .dbf, .shx, .prj) y
tiene que poderse subir a la vez, o en un .zip (24-11-2026; antes la app guardaba un
solo fichero y con otra extensión, y un Shapefile no se leía nunca).

Se ejecutan con `python tests/test_subida_puntos.py` (o con pytest). Sin red.
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sklearn.ensemble  # noqa: F401,E402  (Smart App Control: antes que lo demás)
import borreguil_pipeline as bp   # noqa: E402

bp._imports()
EJEMPLO = Path(__file__).resolve().parent.parent / 'plantilla' / 'example_sierra_nevada.csv'


class Subido:
    """Como un fichero subido en Streamlit: nombre y contenido."""
    def __init__(self, ruta, nombre=None):
        self.name = nombre or os.path.basename(ruta)
        self._datos = Path(ruta).read_bytes()

    def getvalue(self):
        return self._datos


def shapefile():
    """Los 20 puntos del ejemplo como Shapefile (en ETRS89 UTM 30N, como lo daría un
    SIG): devuelve las rutas de sus partes."""
    import geopandas as gpd
    import pandas as pd
    d = tempfile.mkdtemp()
    df = pd.read_csv(EJEMPLO)
    (gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs=4326)
     .to_crs(25830).to_file(os.path.join(d, 'mis_puntos.shp')))
    return {os.path.splitext(f)[1]: os.path.join(d, f) for f in os.listdir(d)}


def test_shapefile_con_todas_sus_partes():
    partes = shapefile()
    ruta = bp.guardar_subida([Subido(p) for p in partes.values()])
    assert ruta.endswith('mis_puntos.shp')
    filas = bp.read_points(ruta)
    assert len(filas) == 20
    assert abs(filas[0]['lon'] - (-3.33104)) < 1e-4          # vuelto a grados
    assert bp.resumen_etiquetas(bp.leer_etiquetas(filas))[0][1].startswith(
        'Etiquetas de la columna `presence`: 12 sí · 8 no')


def test_shapefile_en_un_zip():
    partes = shapefile()
    z = shutil.make_archive(os.path.join(tempfile.mkdtemp(), 'mis_puntos'), 'zip',
                            os.path.dirname(partes['.shp']))
    ruta = bp.guardar_subida([Subido(z)])
    assert ruta.endswith('.zip') and len(bp.read_points(ruta)) == 20


def test_shp_solo_dice_lo_que_falta():
    partes = shapefile()
    for en, texto in ((False, 'Al Shapefile le falta .dbf y .shx'),
                      (True, 'The Shapefile is missing .dbf and .shx')):
        try:
            bp.guardar_subida([Subido(partes['.shp'])], en=en)
        except ValueError as e:
            assert texto in str(e) and '.zip' in str(e), str(e)
        else:
            raise AssertionError('un .shp solo no se puede leer')
    try:
        bp.guardar_subida([Subido(partes['.shp']), Subido(partes['.dbf'])])
    except ValueError as e:
        assert 'le falta .shx' in str(e)
    else:
        raise AssertionError('sin .shx no se puede leer')


def test_sin_shp_o_con_dos_ficheros():
    partes = shapefile()
    try:
        bp.guardar_subida([Subido(partes['.dbf']), Subido(partes['.shx'])])
    except ValueError as e:
        assert 'falta el .shp' in str(e)
    else:
        raise AssertionError('sin .shp no hay puntos')
    try:
        bp.guardar_subida([Subido(EJEMPLO), Subido(EJEMPLO, 'otro.csv')])
    except ValueError as e:
        assert 'un solo fichero' in str(e)
    else:
        raise AssertionError('dos ficheros de puntos a la vez no')


def test_un_fichero_suelto_como_siempre():
    ruta = bp.guardar_subida([Subido(EJEMPLO)])
    assert ruta.endswith('example_sierra_nevada.csv') and len(bp.read_points(ruta)) == 20
    ruta = bp.guardar_subida(Subido(EJEMPLO))              # también sin lista
    assert len(bp.read_points(ruta)) == 20
    assert bp.guardar_subida(None) is None and bp.guardar_subida([]) is None


def test_geojson_con_su_extension():
    import geopandas as gpd
    import pandas as pd
    d = tempfile.mkdtemp()
    df = pd.read_csv(EJEMPLO)
    g = os.path.join(d, 'puntos.geojson')
    gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs=4326).to_file(
        g, driver='GeoJSON')
    ruta = bp.guardar_subida([Subido(g)])
    assert ruta.endswith('.geojson') and len(bp.read_points(ruta)) == 20


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
