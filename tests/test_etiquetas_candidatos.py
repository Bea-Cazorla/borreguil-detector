"""Pruebas de cómo se lee la etiqueta de los puntos candidatos.

La app guarda si un punto es borreguil en `Borreguil` (si/no) y si es dudoso en
`Duda` (si), pero los ficheros de otras personas traen otros nombres y valores
(`borreguil` yes/no, `presence` 1/0, `doubtful`…). Al cargarlos se traducen, y si
no hay ninguna columna de etiqueta la app lo dice. Se ejecutan con
`python tests/test_etiquetas_candidatos.py`.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import borreguil_pipeline as bp   # noqa: E402

bp._imports()


def p(i, **attrs):
    return {'ID': f'p{i}', 'lon': -3.3 + i * 1e-3, 'lat': 37.05, **attrs}


def test_los_valores_de_si_o_no_se_entienden_en_cualquier_forma():
    for v in ('si', 'Sí', 'SI', 'yes', 'Y', '1', 1, 1.0, True, 'true', 'presencia',
              'presence', 'borreguil'):
        assert bp.valor_si_no(v) == 'si', v
    for v in ('no', 'No', 'N', '0', 0, 0.0, False, 'false', 'ausencia', 'absence',
              'no_borreguil'):
        assert bp.valor_si_no(v) == 'no', v
    for v in (None, float('nan'), '', '  ', 'nan'):
        assert bp.valor_si_no(v) == '', v
    for v in ('quizás', 2, 0.5, 'húmedo'):
        assert bp.valor_si_no(v) == '?', v


def test_el_fichero_de_la_app_queda_igual():
    rows = [p(0, Borreguil='si', Duda=''), p(1, Borreguil='no', Duda=''),
            p(2, Borreguil='', Duda='si'), p(3, Borreguil=None, Duda=None)]
    antes = [dict(r) for r in rows]
    info = bp.leer_etiquetas(rows)
    assert rows == antes
    assert (info['columna'], info['n_si'], info['n_no'], info['n_sin']) == ('Borreguil', 1, 1, 2)
    assert (info['duda'], info['n_duda']) == ('Duda', 1)
    assert [n for n, _ in bp.resumen_etiquetas(info)] == ['ok']


def test_borreguil_con_tilde_pasa_a_si():
    rows = [p(0, Borreguil='Sí'), p(1, Borreguil='NO')]
    bp.leer_etiquetas(rows)
    assert [r['Borreguil'] for r in rows] == ['si', 'no']


def test_el_fichero_en_ingles_se_traduce_y_se_renombra():
    # Como el paquete para Flore: `borreguil` yes/no y `doubtful` yes/no.
    rows = [p(0, borreguil='yes', doubtful='no'), p(1, borreguil='no', doubtful='no'),
            p(2, borreguil='yes', doubtful='yes')]
    info = bp.leer_etiquetas(rows)
    assert [r['Borreguil'] for r in rows] == ['si', 'no', 'si']
    assert [r['Duda'] for r in rows] == ['', '', 'si']
    assert all('borreguil' not in r for r in rows)       # solo cambia la mayúscula
    assert all(r['doubtful'] in ('yes', 'no') for r in rows)   # otro nombre: se queda
    assert (info['columna'], info['n_si'], info['n_no'], info['n_duda']) == ('borreguil', 2, 1, 1)
    niveles = [n for n, _ in bp.resumen_etiquetas(info)]
    assert niveles == ['info']


def test_presencia_en_unos_y_ceros():
    rows = [p(0, presence=1), p(1, presence=0), p(2, presence=None)]
    info = bp.leer_etiquetas(rows)
    assert [r.get('Borreguil') for r in rows] == ['si', 'no', None]
    assert 'presence' in rows[0]
    assert (info['columna'], info['n_sin']) == ('presence', 1)


def test_sin_columna_de_etiqueta_avisa_y_dice_que_columnas_hay():
    rows = [p(0, altitud=2500, nota='junto al lago'), p(1, altitud=2600, nota='')]
    info = bp.leer_etiquetas(rows)
    assert info['columna'] is None and all('Borreguil' not in r for r in rows)
    (nivel, texto), = bp.resumen_etiquetas(info)
    assert nivel == 'aviso' and '`altitud`' in texto and '`nota`' in texto


def test_una_clase_con_tipos_no_se_toma_por_etiqueta():
    rows = [p(i, clase=c) for i, c in enumerate(['humedo', 'seco', 'humedo', 'roca'])]
    info = bp.leer_etiquetas(rows)
    assert info['columna'] is None and info['descartadas'] == ['clase']
    assert all('Borreguil' not in r for r in rows)
    assert 'clase' in bp.resumen_etiquetas(info)[0][1]


def test_la_columna_tipo_no_se_usa_aunque_diga_borreguil():
    rows = [p(0, tipo='borreguil'), p(1, tipo='no')]
    assert bp.leer_etiquetas(rows)['columna'] is None


def test_valores_raros_quedan_sin_etiqueta_y_se_avisa():
    rows = [p(i, Borreguil=v) for i, v in enumerate(['si', 'no', 'quizás', 'quizás'])]
    info = bp.leer_etiquetas(rows)
    assert [r['Borreguil'] for r in rows] == ['si', 'no', 'quizás', 'quizás']
    assert info['no_entendidos'] == {'quizás': 2} and info['n_sin'] == 2
    niveles = [n for n, _ in bp.resumen_etiquetas(info)]
    assert niveles == ['ok', 'aviso']


def test_una_columna_vacia_de_borreguil_deja_paso_a_otra():
    rows = [p(0, Borreguil='', label='yes'), p(1, Borreguil=None, label='no')]
    info = bp.leer_etiquetas(rows)
    assert info['columna'] == 'label'
    assert [r['Borreguil'] for r in rows] == ['si', 'no']


def test_desde_un_geojson_y_luego_cuenta_para_el_modelo():
    import mapeo_rf as mr
    feats = [{'type': 'Feature', 'geometry': {'type': 'Point', 'coordinates': [r['lon'], r['lat']]},
              'properties': {k: v for k, v in r.items() if k not in ('lon', 'lat')}}
             for r in [p(0, borreguil='yes', doubtful='no'), p(1, borreguil='no', doubtful='no'),
                       p(2, borreguil='yes', doubtful='yes')]]
    with tempfile.TemporaryDirectory() as d:
        ruta = Path(d) / 'puntos.geojson'
        ruta.write_text(json.dumps({'type': 'FeatureCollection', 'features': feats}),
                        encoding='utf-8')
        rows = bp.read_points(ruta)
    bp.leer_etiquetas(rows)
    res = mr.muestras(rows, mr.NIVEL_SI_NO)
    assert sorted((c['categoria'], c['n']) for c in res['clases']) == [
        ('borreguil', 1), ('no borreguil', 1)]


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
