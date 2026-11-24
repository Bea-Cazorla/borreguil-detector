"""Pruebas de los puntos que se añaden pinchando en el mapa.

Quien conoce el sitio (en un taller, los gestores) pincha en el mapa y dice si ahí
hay borreguil, si no lo hay o si no está seguro. El punto entra como un dato de campo
más, con las columnas de la plantilla, y la app dice si el modelo coincidía con esa
respuesta (con la probabilidad que le daba ANTES de saberla). Se ejecutan con
`python tests/test_puntos_mapa.py`; no necesitan red y tardan unos segundos.
"""
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                 # noqa: E402

import borreguil_pipeline as bp   # noqa: E402

bp._imports()

TMP = Path(tempfile.mkdtemp(prefix='puntos_mapa_'))
BASE = [{'ID': 'p1', 'lon': -3.30, 'lat': 37.05}, {'ID': 'mapa_2', 'lon': -3.31, 'lat': 37.06},
        {'ID': 'p3', 'lon': -3.32, 'lat': 37.07}]


def test_punto_nuevo_con_las_columnas_de_la_plantilla():
    r = bp.punto_del_mapa(BASE, -3.35, 37.04, 'si', observador=' Ana ', nota='junto al arroyo',
                          fecha='2026-11-27')
    assert r['ID'] == 'mapa_3'                          # el siguiente a los que hay
    assert (r['lon'], r['lat']) == (-3.35, 37.04)
    assert (r['Borreguil'], r['Duda'], r['origin']) == ('si', '', 'verificado')
    assert (r['source'], r['method'], r['date']) == ('mapa', 'local knowledge', '2026-11-27')
    assert (r['observer'], r['notes']) == ('Ana', 'junto al arroyo')
    no = bp.punto_del_mapa(BASE, -3.35, 37.04, 'no')
    assert (no['Borreguil'], no['Duda'], no['origin']) == ('no', '', 'verificado')
    assert 'observer' not in no and 'notes' not in no   # vacíos: no se ponen
    duda = bp.punto_del_mapa(BASE, -3.35, 37.04, 'duda')
    assert (duda['Borreguil'], duda['Duda'], duda['origin']) == ('', 'si', 'dudoso')
    assert len(no['date']) == 10                        # hoy, AAAA-MM-DD
    try:
        bp.punto_del_mapa(BASE, 0, 0, 'quizas')
        raise AssertionError('debía rechazar una respuesta desconocida')
    except ValueError:
        pass


def test_numeracion():
    assert bp.nuevo_id_mapa([]) == 'mapa_1'
    assert bp.nuevo_id_mapa([{'ID': 'mapa_9'}, {'ID': 'mapa_10'}, {'ID': 'mapa_x'},
                             {'ID': 'otro_mapa_50'}, {'ID': float('nan')}]) == 'mapa_11'


def test_pinchar_encima_de_un_punto():
    i, d = bp.punto_mas_cercano(BASE, -3.30005, 37.05004)
    assert i == 0 and d < bp.CERCA_M                    # unos 6 m: es ese punto
    i, d = bp.punto_mas_cercano(BASE, -3.305, 37.05)
    assert i == 0 and 400 < d < 500                     # ~445 m: un sitio nuevo
    assert bp.punto_mas_cercano([], 0, 0) == (None, float('inf'))
    # 1 km hacia el norte y hacia el este
    assert abs(bp.distancia_m(-3.3, 37.0, -3.3, 37.0 + 1 / 110.54) - 1000) < 1
    assert abs(bp.distancia_m(-3.3, 37.0, -3.3 + 1 / (111.32 * np.cos(np.radians(37))), 37.0)
               - 1000) < 1


def test_opinion_del_modelo_igual_que_la_decision():
    # La opinión usa el mismo corte que decide(): el umbral, o 0,70 si es más alto
    assert bp.corte_borreguil(0.5) == 0.5 and bp.corte_borreguil(0.8) == 0.70
    for umbral in (0.3, 0.5, 0.75):
        for p in np.linspace(0, 1, 101):
            dec = bp.decide({'rf_proba': float(p)}, threshold=umbral)
            es = dec in ('BORREGUIL PROBABLE', 'POSIBLE BORREGUIL')
            assert (bp.opinion_modelo(float(p), umbral) == 'si') == es, (umbral, p, dec)
    assert bp.opinion_modelo(float('nan'), 0.5) is None
    assert bp.opinion_modelo(None, 0.5) is None
    # decide() no ha cambiado
    assert bp.decide({'rf_proba': 0.72}, 0.8) == 'BORREGUIL PROBABLE'
    assert bp.decide({'rf_proba': 0.55}, 0.5) == 'POSIBLE BORREGUIL'
    assert bp.decide({'rf_proba': 0.45}, 0.5) == 'INCIERTO'
    assert bp.decide({'rf_proba': 0.10}, 0.5) == 'NO BORREGUIL'


def test_resumen_de_coincidencias():
    rows = [dict(r) for r in BASE[:1]]                  # un punto normal: no cuenta
    for resp, p in (('si', 0.81), ('si', 0.30), ('no', 0.12), ('duda', 0.9), ('no', None)):
        r = bp.punto_del_mapa(rows, -3.3, 37.0 + len(rows) / 100, resp, observador='Luis')
        r['rf_proba_al_anadir'] = p if p is not None else float('nan')
        rows.append(r)
    filas, n, ok = bp.resumen_mapa(rows, 0.5)
    assert [f['ID'] for f in filas] == ['mapa_1', 'mapa_2', 'mapa_3', 'mapa_4', 'mapa_5']
    assert [f['coincide'] for f in filas] == [True, False, True, None, None]
    assert [f['modelo'] for f in filas] == ['si', 'no', 'no', 'si', None]
    assert (n, ok) == (3, 2)                            # los dudosos y sin dato no cuentan
    assert filas[0]['observer'] == 'Luis' and filas[4]['proba'] is None
    assert bp.respuesta_del_mapa(BASE[0]) is None


def test_cuentan_como_dato_de_campo_salvo_el_dudoso():
    rows = []
    for resp in ('si', 'no', 'duda'):
        rows.append(bp.punto_del_mapa(rows, -3.3, 37.0 + len(rows) / 100, resp))
    assert list(bp.etiquetas_de_campo(rows)) == [1, 0, -1]
    for r in rows:
        r['rf_proba'] = 0.5
    assert [bp.decide(r) for r in rows] == ['BORREGUIL VERIFICADO', 'NO BORREGUIL VERIFICADO',
                                            'DUDOSO (campo)']


def test_se_guardan_y_se_vuelven_a_leer():
    rows = [dict(BASE[0], Borreguil='', decision='POSIBLE BORREGUIL', rf_proba=0.6)]
    for resp in ('si', 'no', 'duda'):
        r = bp.punto_del_mapa(rows, -3.3, 37.0 + len(rows) / 100, resp, observador='Ana',
                              nota=f'nota {resp}', fecha='2026-11-27')
        r['rf_proba'] = 0.4
        r['decision'] = bp.decide(r)
        rows.append(r)
    ruta = TMP / 'puntos.geojson'
    bp.save_points_geojson(rows, ruta)
    import json
    props = {f['properties']['ID']: f['properties']
             for f in json.loads(ruta.read_text(encoding='utf-8'))['features']}
    p1, p2, p3 = props['mapa_1'], props['mapa_2'], props['mapa_3']
    assert (p1['presence'], p2['presence'], p3['presence']) == (1, 0, None)
    assert p3['doubtful'] == 1
    for p in (p1, p2, p3):
        assert (p['method'], p['date'], p['observer']) == ('local knowledge', '2026-11-27', 'Ana')
    assert p1['notes'] == 'nota si'
    otra = bp.read_points(str(ruta))
    bp.leer_etiquetas(otra)                             # lo que hace la app al cargar
    o = {r['ID']: r for r in otra}
    assert bp._txt(o['mapa_1']['Borreguil']) == 'si'
    assert bp._txt(o['mapa_2']['Borreguil']) == 'no'
    assert bp._txt(o['mapa_3']['Duda']) == 'si'
    assert bp._txt(o['mapa_1']['method']) == 'local knowledge'


def test_prediccion_con_el_modelo_en_uso():
    from sklearn.ensemble import RandomForestClassifier
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    y = (X[:, 0] > 0).astype(int)
    clf = RandomForestClassifier(n_estimators=30, random_state=0).fit(X, y)
    bundle = {'model': clf, 'features': ['a', 'b'], 'medians': [0.0, 0.0]}
    p = bp.predecir_con_bundle([{'a': 2.0, 'b': 0.0}, {'a': -2.0, 'b': ''},
                                {'b': 1.0}], bundle)
    assert p[0] > 0.8 and p[1] < 0.2
    # lo que falta se rellena con la mediana del modelo (a = 0)
    assert abs(p[2] - clf.predict_proba([[0.0, 1.0]])[0, 1]) < 1e-9


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
