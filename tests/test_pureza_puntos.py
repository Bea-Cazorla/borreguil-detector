"""La pureza que la app propone para cada punto es la misma que la del mapa
(«Pureza dentro del píxel»): verde estimado con el NDVI de inicio de verano del píxel
de 10 m, puro desde el 75 %, y si no, mixto-agua si el píxel es agua o la toca y
mixto-roca si no. Desde el 24-11-2026 (antes era relativa al conjunto).

Se ejecutan con `python tests/test_pureza_puntos.py` (o con pytest). Sin red.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sklearn.ensemble  # noqa: F401,E402  (Smart App Control: antes que lo demás)
import borreguil_pipeline as bp   # noqa: E402
import mapeo_rf as mr             # noqa: E402

bp._imports()


def es_borreguil(dec):
    d = (dec or '').lower()
    return 'no borreguil' not in d and 'borreguil' in d


def punto(i, ndvi=None, agua=None, decision='BORREGUIL PROBABLE'):
    r = {'ID': f'p{i}', 'lon': -3.30 + i * 0.001, 'lat': 37.05, 'decision': decision,
         'ndmi_mean': 0.05, 'slope_deg': 15.0}
    if ndvi is not None:
        r['ndvi_early_pixel'] = ndvi
    if agua is not None:
        r['agua_pixel_vecinos'] = agua
    return r


def test_puro_mixto_agua_y_mixto_roca():
    rows = [punto(0, 0.60, 0), punto(1, 0.30, 1), punto(2, 0.30, 0)]
    bp.apply_typology(rows, es_borreguil)
    assert [r['pureza'] for r in rows] == ['puro', 'mixto-agua', 'mixto-roca']
    assert rows[0]['pureza_regla'] == 'puro'
    assert abs(rows[0]['verde_estimado'] - 1.0) < 1e-6
    assert abs(rows[1]['verde_estimado'] - (0.30 - 0.08) / (0.60 - 0.08)) < 1e-6
    assert rows[1]['tipo_borreguil'].endswith(' · mixto-agua')


def test_el_corte_del_75_por_ciento():
    # verde 0,75 justo: (ndvi - 0,08) / 0,52 = 0,75 → ndvi = 0,47
    rows = [punto(0, 0.47, 0), punto(1, 0.46, 0)]
    bp.apply_typology(rows, es_borreguil)
    assert [r['pureza'] for r in rows] == ['puro', 'mixto-roca']


def test_igual_que_el_mapa():
    q = mr.params_pureza()
    nombre = {v: k for k, v in mr.codigos('pureza').items()}
    ndvis = [-0.2, 0.0, 0.08, 0.2, 0.35, 0.469, 0.47, 0.5, 0.6, 0.9]
    for agua in (0, 1):
        rows = [punto(i, v, agua) for i, v in enumerate(ndvis)]
        bp.apply_typology(rows, es_borreguil)
        mapa = [nombre[int(c)] for c in mr.pureza_de(
            mr.verde_estimado(ndvis, q['sin_verde'], q['todo_verde']), [agua == 1] * len(ndvis),
            q['puro_desde'])]
        assert [r['pureza'] for r in rows] == mapa


def test_no_depende_de_los_demas_puntos():
    # Antes era relativa: el mismo punto cambiaba según con quién se analizara.
    solo = [punto(0, 0.40, 0)]
    bp.apply_typology(solo, es_borreguil)
    grupo = [punto(0, 0.40, 0)] + [punto(i, 0.10 + 0.01 * i, 0) for i in range(1, 30)]
    bp.apply_typology(grupo, es_borreguil)
    assert solo[0]['pureza'] == grupo[0]['pureza'] == 'mixto-roca'


def test_sin_el_pixel_no_hay_propuesta():
    # Sin Earth Engine (o si falla la lectura) no se inventa: '—', y se puede revisar.
    rows = [punto(0), punto(1, 0.60)]
    rows[0].update({'ndwi_mean': 0.3, 'albedo_mean': 0.4, 'ndvi_late': 0.1,
                    'surr_water': 0.5, 'surr_rock': 0.5})     # lo que usaba la regla vieja
    bp.apply_typology(rows, es_borreguil)
    assert rows[0]['pureza'] == '—' and rows[0]['verde_estimado'] is None
    assert rows[0]['tipo_borreguil'].endswith(' · —')
    assert rows[1]['pureza'] == 'puro'          # sin dato de agua, cuenta como sin agua
    bp.set_tipo_revisado(rows[0], pureza='mixto-roca', ambiente='ladera')
    assert rows[0]['pureza'] == 'mixto-roca' and rows[0]['pureza_regla'] == '—'


def test_todo_verde_de_los_puros_revisados():
    # Con 5 o más puntos revisados como puros, el NDVI de «todo verde» es su mediana,
    # como en el mapa. Aquí 0,45: un píxel con NDVI 0,40 pasa a puro (0,865 de verde).
    rows = [punto(i, 0.45, 0) for i in range(5)] + [punto(9, 0.40, 0)]
    bp.apply_typology(rows, es_borreguil)
    assert rows[5]['pureza'] == 'mixto-roca'        # con 0,60: 0,615 de verde
    for r in rows[:5]:
        bp.set_tipo_revisado(r, pureza='puro')
    bp.apply_typology(rows, es_borreguil)
    assert rows[5]['pureza'] == 'puro'
    assert abs(rows[5]['verde_estimado'] - (0.40 - 0.08) / (0.45 - 0.08)) < 1e-6
    # con cuatro no basta
    bp.clear_tipo_revisado(rows[0])
    bp.apply_typology(rows, es_borreguil)
    assert rows[5]['pureza'] == 'mixto-roca'


def test_lo_revisado_no_se_pisa():
    rows = [punto(0, 0.60, 0)]
    bp.apply_typology(rows, es_borreguil)
    bp.set_tipo_revisado(rows[0], pureza='mixto-agua')
    bp.apply_typology(rows, es_borreguil)
    assert rows[0]['pureza'] == 'mixto-agua' and rows[0]['pureza_regla'] == 'puro'


def test_los_no_borreguiles_no_tienen_pureza():
    rows = [punto(0, 0.60, 0, decision='NO BORREGUIL')]
    bp.apply_typology(rows, es_borreguil)
    assert rows[0]['pureza'] == '' and rows[0]['tipo_borreguil'] == ''


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
