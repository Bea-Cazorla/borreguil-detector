"""Pruebas de cómo se lee el fichero de «puntos de campo para comprobar el modelo».

Desde el 24-11-2026 se lee igual que «Tus puntos» (bp.leer_puntos_de_campo →
leer_etiquetas). Antes se leía aparte, con nombres y valores exactos, y una ausencia
escrita `0.0` (lo normal en un Shapefile o un GeoPackage con la columna numérica) o en
una columna `Presence` se tomaba como borreguil sin avisar. Se ejecutan con
`python tests/test_puntos_campo.py` (o con pytest); no necesitan red.
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import borreguil_pipeline as bp   # noqa: E402

bp._imports()

APP = Path(__file__).resolve().parent.parent
TMP = Path(tempfile.mkdtemp(prefix='puntos_campo_'))


def candidato():
    return [{'ID': 'c1', 'lon': -3.300000, 'lat': 37.050000}]


def campo(**kw):
    """Un punto de campo encima del candidato y otro lejos (se añade)."""
    a = {'lon': -3.300010, 'lat': 37.050010}
    b = {'lon': -3.310000, 'lat': 37.060000}
    return [dict(a, **{k: v[0] for k, v in kw.items()}),
            dict(b, **{k: v[1] for k, v in kw.items()})]


def etiquetas(rows):
    return [(r.get('Borreguil'), r.get('Duda') or '') for r in rows]


def test_una_ausencia_numerica_0_0_es_ausencia():
    rows, info = candidato(), {}
    res = bp.merge_truth_points(rows, campo(presence=(0.0, 1.0)), info=info)
    assert res == (1, 1, 1, 1)
    assert etiquetas(rows)[0] == ('no', '') and rows[1]['Borreguil'] == 'si'
    assert info['columna'] == 'presence' and not info['solo_presencias']


def test_la_columna_con_mayusculas_se_lee():
    for col in ('Presence', 'PRESENCE', 'Presencia'):
        rows = candidato()
        bp.merge_truth_points(rows, campo(**{col: ('0', 'Sí')}))
        assert [r['Borreguil'] for r in rows] == ['no', 'si'], col


def test_sin_columna_todos_son_borreguil():
    rows, info = candidato(), {}
    res = bp.merge_truth_points(rows, campo(), info=info)
    assert res == (1, 1, 2, 0) and info['solo_presencias']
    assert [r['Borreguil'] for r in rows] == ['si', 'si']


def test_un_dudoso_entra_como_dudoso_y_no_cuenta():
    rows, info = candidato(), {}
    bp.merge_truth_points(rows, campo(presence=(1, 0), doubtful=(1, None)), info=info)
    assert etiquetas(rows)[0] == ('si', 'si') and info['n_duda'] == 1
    assert list(bp.etiquetas_de_campo(rows)) == [-1, 0]      # el dudoso no cuenta


def test_si_los_valores_no_se_entienden_no_se_usa_ninguno():
    rows, info = candidato(), {}
    res = bp.merge_truth_points(rows, campo(presence=('quizás', 'otro')), info=info)
    assert res == (0, 0, 0, 0) and len(rows) == 1 and 'Borreguil' not in rows[0]
    assert not info['solo_presencias'] and info['n_sin_usar'] == 2


def test_un_punto_sin_valor_no_se_usa():
    rows, info = candidato(), {}
    res = bp.merge_truth_points(rows, campo(presence=(1, None)), info=info)
    assert res == (1, 0, 1, 0) and len(rows) == 1 and info['n_sin_usar'] == 1


def test_desde_un_geojson_con_0_0():
    ruta = TMP / 'campo.geojson'
    ruta.write_text(json.dumps({'type': 'FeatureCollection', 'features': [
        {'type': 'Feature', 'geometry': {'type': 'Point', 'coordinates': [-3.30001, 37.05001]},
         'properties': {'Presence': 0.0}},
        {'type': 'Feature', 'geometry': {'type': 'Point', 'coordinates': [-3.31, 37.06]},
         'properties': {'Presence': 1.0}}]}), encoding='utf-8')
    rows = candidato()
    assert bp.merge_truth_points(rows, bp.read_points(str(ruta))) == (1, 1, 1, 1)
    assert [r['Borreguil'] for r in rows] == ['no', 'si']


def test_el_ejemplo_de_la_plantilla():
    tr = bp.read_points(str(APP / 'plantilla' / 'example_sierra_nevada.csv'))
    info = bp.leer_puntos_de_campo(tr)
    assert (info['columna'], info['n_si'], info['n_no'], info['n_sin_usar']) == \
        ('presence', 12, 8, 0)


def test_la_plantilla_en_mayusculas():
    """Desde el 27-11-2026 la plantilla trae los nombres en MAYÚSCULAS: quién, cómo y
    cuándo pasan al candidato igual que con los nombres en minúsculas."""
    import openpyxl
    cab = [c.value for c in openpyxl.load_workbook(
        APP / 'plantilla' / 'point_data_template.xlsx')['points'][1]]
    assert cab == [c.upper() for c in cab] and 'PRESENCE' in cab and 'DATE' in cab
    ruta = TMP / 'mayusculas.csv'
    ruta.write_text('ID,LON,LAT,PRESENCE,MOISTURE,METHOD,DATE,OBSERVER,EXPERTISE,CERTAINTY\n'
                    'p1,-3.300001,37.050001,1,wet,field,2026-07-15,BPC,researcher,high\n',
                    encoding='utf-8')
    tr = bp.read_points(str(ruta))
    assert not any(k in tr[0] for k in ('PRESENCE', 'DATE', 'OBSERVER'))
    rows = candidato()
    assert bp.merge_truth_points(rows, tr) == (1, 0, 1, 0)
    assert (rows[0]['Borreguil'], rows[0]['method'], rows[0]['date']) == \
        ('si', 'field', '2026-07-15')


class _Subido:
    def __init__(self, name, datos=b'x'):
        self.name, self._d = name, datos

    def getvalue(self):
        return self._d


def test_sin_sistema_de_referencia_no_se_deja_subir():
    """27-11-2026: una capa sin sistema de referencia (un Shapefile sin .prj, suelto o en
    un .zip) no se deja usar; KML, GeoJSON y CSV van siempre en WGS84 y valen."""
    import zipfile
    partes = [_Subido(n) for n in ('pts.shp', 'pts.dbf', 'pts.shx')]
    try:
        bp.guardar_subida(partes)
        assert False, 'sin .prj tenía que fallar'
    except ValueError as e:
        assert 'sistema de referencia' in str(e) and 'QGIS' in str(e), str(e)
    try:
        bp.guardar_subida(partes, en=True)
        assert False
    except ValueError as e:
        assert 'coordinate reference system' in str(e), str(e)
    ruta = bp.guardar_subida(partes + [_Subido('pts.prj', b'GEOGCS["WGS 84"]')])
    assert ruta.endswith('pts.shp') and not bp.falta_sistema_de_referencia(ruta)
    # vacío, el .prj no cuenta
    assert bp.falta_sistema_de_referencia(_shp_con_prj_vacio())
    # en un .zip
    z = TMP / 'capa.zip'
    with zipfile.ZipFile(z, 'w') as f:
        for n in ('zona.shp', 'zona.dbf', 'zona.shx'):
            f.writestr(n, b'x')
    assert bp.falta_sistema_de_referencia(z)
    with zipfile.ZipFile(z, 'a') as f:
        f.writestr('zona.prj', b'PROJCS["ETRS89 / UTM zone 30N"]')
    assert not bp.falta_sistema_de_referencia(z)
    # los formatos que siempre van en grados
    for n in ('p.kml', 'p.geojson', 'p.csv'):
        assert not bp.falta_sistema_de_referencia(TMP / n)


def _shp_con_prj_vacio():
    d = TMP / 'prj_vacio'
    d.mkdir(exist_ok=True)
    for n in ('a.shp', 'a.dbf', 'a.shx'):
        (d / n).write_bytes(b'x')
    (d / 'a.prj').write_bytes(b'')
    return d / 'a.shp'


if __name__ == '__main__':
    pruebas = [(n, f) for n, f in sorted(globals().items())
               if n.startswith('test_') and callable(f)]
    fallos = 0
    try:
        for nombre, f in pruebas:
            try:
                f()
                print(f'  ok     {nombre}')
            except Exception as e:                      # noqa: BLE001
                fallos += 1
                print(f'  FALLO  {nombre}: {type(e).__name__}: {e}')
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    print(f'{len(pruebas) - fallos}/{len(pruebas)} pruebas correctas')
    sys.exit(1 if fallos else 0)
