"""Pruebas de qué modelo da la probabilidad de cada punto.

Por defecto, el MODELO DE PARTIDA (el de Sierra Nevada u otro elegido en el panel),
aunque los puntos traigan etiquetas de campo: con ellas solo se mide cuánto
acierta. Con la casilla «Entrenar también un modelo con mis puntos», la app entrena
otro con los puntos, compara los dos con los puntos de campo y se queda con el que
más acierte. Se ejecutan con `python tests/test_modelo_partida.py`; usan datos y
modelos de partida inventados (no hace falta red) y tardan unos segundos.
"""
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                 # noqa: E402

import borreguil_pipeline as bp   # noqa: E402

bp._imports()

TMP = Path(tempfile.mkdtemp(prefix='modelo_partida_'))
SENAL = 'ndvi_late'        # variable del detector (FEATURES_RF) que delata al borreguil
OTRA = 'senal_aparte'      # variable que solo conoce el modelo de partida «bueno»


def puntos(n=240, semilla=0, senal_en_rf=True):
    """Puntos repartidos por unos 15 km (varias zonas para la validación), la mitad
    borreguil. `senal_en_rf`: si la variable del detector delata al borreguil o es
    ruido; OTRA siempre lo delata."""
    rng = np.random.default_rng(semilla)
    rows = []
    for i in range(n):
        es = i % 2 == 0
        r = {'ID': f'p{i}', 'lon': -3.40 + rng.uniform(0, 0.18),
             'lat': 37.00 + rng.uniform(0, 0.12), 'es': es}
        for f in bp.FEATURES_RF:
            r[f] = float(rng.normal())
        if senal_en_rf:
            r[SENAL] = float(rng.normal(1.5 if es else -1.5, 0.5))
        r[OTRA] = float(rng.normal(1.5 if es else -1.5, 0.5))
        rows.append(r)
    return rows


def modelo_de_partida(nombre, variable, zona='Sierra Nevada (prueba)'):
    """Un .joblib como los de la app, entrenado con datos inventados en los que
    `variable` separa borreguil de no borreguil."""
    from sklearn.ensemble import RandomForestClassifier
    rows = puntos(400, semilla=99)
    X = np.array([[r[variable]] for r in rows])
    y = np.array([int(r['es']) for r in rows])
    clf = RandomForestClassifier(n_estimators=50, random_state=0).fit(X, y)
    ruta = TMP / nombre
    bp.save_model_bundle({'model': clf, 'features': [variable], 'medians': [0.0],
                          'training_zone': zona, 'training_n': 400,
                          'auc_groupkfold': 0.9}, ruta)
    return str(ruta)


BUENO = modelo_de_partida('bueno.joblib', OTRA)
# Uno que no sabe nada: su variable es ruido en los puntos de la zona.
MALO = modelo_de_partida('malo.joblib', 'ndvi_early')


def etiquetar(rows, n_si=40, n_no=40):
    """Marca como de campo los primeros n_si borreguiles y n_no no borreguiles."""
    si = no = 0
    for r in rows:
        if r['es'] and si < n_si:
            r['Borreguil'] = 'si'; si += 1
        elif not r['es'] and no < n_no:
            r['Borreguil'] = 'no'; no += 1
    return rows


def test_por_defecto_usa_el_de_partida_aunque_haya_etiquetas():
    rows = etiquetar(puntos())
    proba, info = bp.train_rf(rows, default_model_path=BUENO)
    assert info['mode'] == 'pretrained' and info['motivo'] is None, info['mode']
    ev = info['evaluacion']
    assert ev['etiqueta'].count(1) == 40 and ev['etiqueta'].count(0) == 40
    assert 'propio' not in ev and len(ev['partida']) == 80
    ac = bp.aciertos_en_campo(ev, 0.5)['partida']
    assert ac['n_si'] == 40 and ac['n_no'] == 40 and ac['auc'] > 0.9, ac
    assert 'n_pseudo' not in info        # no se ha entrenado nada


def test_sin_etiquetas_no_hay_nada_que_medir():
    proba, info = bp.train_rf(puntos(), default_model_path=BUENO)
    assert info['mode'] == 'pretrained' and info['evaluacion']['etiqueta'] == []
    assert bp.aciertos_en_campo(info['evaluacion'], 0.5) == {}


def test_aciertos_con_el_umbral_de_la_decision():
    ev = {'etiqueta': [1, 1, 1, 0, 0, 0], 'partida': [0.9, 0.72, 0.55, 0.6, 0.3, 0.1]}
    a = bp.aciertos_en_campo(ev, 0.5)['partida']
    assert (a['ok_si'], a['ok_no']) == (3, 2), a        # 0,6 es «posible»: falla
    # Con un umbral de 0,80 cuenta como borreguil lo «probable» (≥ 0,70), como en decide()
    a = bp.aciertos_en_campo(ev, 0.80)['partida']
    assert (a['ok_si'], a['ok_no']) == (2, 3), a
    assert a['auc'] is None            # con menos de 5 y 5 no se da el AUC


def test_con_la_casilla_se_queda_con_el_propio_si_acierta_mas():
    rows = etiquetar(puntos())
    proba, info = bp.train_rf(rows, default_model_path=MALO, propio=True)
    c = info['comparacion']
    assert info['mode'] == 'trained' and c['elegido'] == 'propio', (info['mode'], c)
    assert c['motivo'] == 'comparado' and c['auc_propio'] > c['auc_partida']
    ac = bp.aciertos_en_campo(info['evaluacion'], 0.5)
    assert set(ac) == {'partida', 'propio'} and ac['propio']['auc'] > 0.9
    assert info['zona_partida'] == 'Sierra Nevada (prueba)'


def test_con_la_casilla_se_queda_con_el_de_partida_si_acierta_mas():
    # La variable del detector es ruido: el modelo propio no aprende nada.
    rows = etiquetar(puntos(senal_en_rf=False))
    proba, info = bp.train_rf(rows, default_model_path=BUENO, propio=True)
    c = info['comparacion']
    assert info['mode'] == 'pretrained' and info['motivo'] == 'comparado', info['mode']
    assert c['elegido'] == 'partida' and c['auc_partida'] > c['auc_propio'], c
    # la probabilidad es la del modelo de partida
    p_partida, _ = bp.train_rf(rows, default_model_path=BUENO)
    assert np.allclose(proba, p_partida)
    # y se sabe con qué negativos se entrenó el que se ha descartado
    assert info['n_neg_campo'] == 40 and info['n_pseudo'] > 0


def test_pseudo_ausencias_contadas_aparte():
    rows = etiquetar(puntos(), n_si=40, n_no=20)
    _, info = bp.train_rf(rows, default_model_path=MALO, propio=True)
    assert info['n_pos'] == 40 and info['n_neg_campo'] == 20
    assert info['n_neg'] == info['n_neg_campo'] + info['n_pseudo'] and info['n_pseudo'] > 0
    # una pseudo-ausencia es un punto SIN etiqueta a más de 250 m de un borreguil
    assert info['neg_buffer_m'] == 250


def test_sin_ausencias_de_campo_no_se_puede_comparar():
    rows = etiquetar(puntos(), n_si=40, n_no=0)
    proba, info = bp.train_rf(rows, default_model_path=BUENO, propio=True)
    c = info['comparacion']
    assert info['mode'] == 'trained' and c['motivo'] == 'pocos_de_campo', c
    assert c['n_si'] == 40 and c['n_no'] == 0
    assert info['n_neg_campo'] == 0 and info['n_pseudo'] >= bp.MIN_ENTRENAR


def test_con_pocos_borreguiles_se_queda_con_el_de_partida():
    rows = etiquetar(puntos(), n_si=10, n_no=40)
    proba, info = bp.train_rf(rows, default_model_path=MALO, propio=True)
    assert info['mode'] == 'pretrained' and info['motivo'] == 'pocos_datos'
    assert info['n_pos'] == 10


def test_sin_ningun_borreguil_se_queda_con_el_de_partida():
    rows = etiquetar(puntos(), n_si=0, n_no=40)
    _, info = bp.train_rf(rows, default_model_path=MALO, propio=True)
    assert info['mode'] == 'pretrained' and info['motivo'] == 'sin_borreguiles'


def test_sin_modelo_de_partida_entrena_el_propio_como_antes():
    rows = etiquetar(puntos())
    _, info = bp.train_rf(rows, default_model_path=str(TMP / 'no_existe.joblib'))
    assert info['mode'] == 'trained' and info['comparacion']['motivo'] == 'sin_partida'


def test_lo_dudoso_no_es_dato_de_campo():
    rows = [{'Borreguil': 'si'}, {'Borreguil': 'no'},
            {'Borreguil': 'si', 'Duda': 'si'}, {'Duda': 'si'}, {}, {'Borreguil': float('nan')}]
    assert list(bp.etiquetas_de_campo(rows)) == [1, 0, -1, -1, -1, -1]


def test_si_empatan_se_queda_el_de_partida():
    ev = {'etiqueta': [1] * 5 + [0] * 5, 'partida': [0.9] * 5 + [0.1] * 5,
          'propio': [0.8] * 5 + [0.2] * 5}
    c = bp.elegir_modelo(ev)
    assert c['elegido'] == 'partida' and c['auc_partida'] == c['auc_propio'] == 1.0


def test_run_rf_and_decide_pasa_la_casilla():
    rows = etiquetar(puntos())
    info = bp.run_rf_and_decide(rows, threshold=0.5, default_model_path=MALO)
    assert info['mode'] == 'pretrained'
    info = bp.run_rf_and_decide(rows, threshold=0.5, default_model_path=MALO, propio=True)
    assert info['mode'] == 'trained'
    assert all(r['decision'] for r in rows)


# ---- Sumar los puntos de la sesión a la tabla de entrenamiento del modelo ----------
# Con la tabla de entrenamiento, la casilla ya no entrena un modelo aparte con tus
# puntos: vuelve a entrenar el de partida con su tabla MÁS tus puntos (un solo modelo).
CON_SENAL = modelo_de_partida('senal.joblib', SENAL)


def tabla(n=200, semilla=11, lon0=-2.80):
    """Una tabla de entrenamiento inventada, lejos de los puntos de la sesión."""
    rng = np.random.default_rng(semilla)
    t = []
    for i in range(n):
        es = i % 2 == 0
        t.append({'ID': f'SN_{i}', 'lon': lon0 + rng.uniform(0, 0.2),
                  'lat': 37.0 + rng.uniform(0, 0.1), 'borreguil': 'si' if es else 'no',
                  'duda': '', SENAL: float(rng.normal(1.5 if es else -1.5, 0.5))})
    return t


def test_sumar_entrena_un_solo_modelo_con_los_dos():
    rows = etiquetar(puntos(60), n_si=10, n_no=10)
    t = tabla()
    # Un punto de la tabla en el mismo sitio que uno de la sesión: cuenta el de la sesión
    t.append(dict(t[0], lon=rows[0]['lon'] + 0.00003, lat=rows[0]['lat']))
    proba, info = bp.train_rf(rows, default_model_path=CON_SENAL, propio=True,
                              entrenamiento=t)
    assert info['mode'] == 'trained' and info['sumado'], info
    assert (info['n_tabla'], info['n_tabla_quitados']) == (200, 1), info
    assert (info['n_pos'], info['n_neg'], info['n_train']) == (10, 10, 220)
    b = info['model_bundle']
    assert b['features'] == [SENAL] and b['training_n'] == 220
    assert b['training_zone'] == 'Sierra Nevada + tus puntos'
    # El acierto en tus puntos sale de la validación por zonas
    ac = bp.aciertos_en_campo(info['evaluacion'], 0.5)
    assert set(ac) == {'partida', 'propio'} and ac['propio']['auc'] > 0.9, ac
    assert len(proba) == len(rows)


def test_sumar_lo_tuyo_cambia_el_modelo():
    # En la tabla, la señal alta es borreguil. En la sesión hay un sitio con la señal
    # MUY alta (10) que no es borreguil: solo lo sabe el modelo si suma tus puntos.
    rows = puntos(40)
    for i, r in enumerate(rows[:20]):
        r[SENAL] = 10.0 + 0.01 * i
        r['Borreguil'] = 'no'
    nuevo = dict(rows[20], ID='nuevo', Borreguil='')
    nuevo[SENAL] = 10.05
    rows.append(nuevo)
    p_partida, _ = bp.train_rf(rows, default_model_path=CON_SENAL)
    p_sumado, info = bp.train_rf(rows, default_model_path=CON_SENAL, propio=True,
                                 entrenamiento=tabla())
    assert info['sumado'] and info['n_neg'] == 20
    assert p_partida[-1] > 0.8 and p_sumado[-1] < 0.3, (p_partida[-1], p_sumado[-1])


def test_sumar_sin_etiquetas_se_queda_el_de_partida():
    rows = puntos(40)
    rows[0]['Borreguil'] = 'si'; rows[0]['Duda'] = 'si'       # el dudoso no cuenta
    proba, info = bp.train_rf(rows, default_model_path=CON_SENAL, propio=True,
                              entrenamiento=tabla())
    assert info['mode'] == 'pretrained' and info['motivo'] == 'sin_puntos'


def test_sumar_dudosos():
    rows = etiquetar(puntos(60), n_si=10, n_no=10)
    rows[0]['Duda'] = 'si'                                     # un borreguil dudoso
    _, info = bp.train_rf(rows, default_model_path=CON_SENAL, propio=True,
                          entrenamiento=tabla())
    # el dudoso no entrena ni cuenta como dato de campo en el acierto
    assert (info['n_pos'], info['n_neg']) == (10 - 1, 10), info
    assert info['evaluacion']['etiqueta'].count(1) == 9


def test_sumar_en_una_sola_zona_no_se_mide():
    rows = etiquetar(puntos(30), n_si=8, n_no=8)
    for r in rows:                                   # todos en la misma celda de 0,05°
        r['lon'] = -3.301 + (r['lon'] + 3.40) * 0.01
        r['lat'] = 37.001 + (r['lat'] - 37.00) * 0.01
    _, info = bp.train_rf(rows, default_model_path=CON_SENAL, propio=True,
                          entrenamiento=tabla())
    assert info['sumado'] and 'propio' not in info['evaluacion']
    assert info['cv'] is None


def test_la_tabla_de_la_app():
    ruta = Path(__file__).resolve().parent.parent / bp.TABLA_ENTRENAMIENTO
    t = bp.leer_tabla_entrenamiento(ruta)
    si = sum(1 for r in t if r['borreguil'] == 'si' and not r['duda'])
    no = sum(1 for r in t if r['borreguil'] == 'no' and not r['duda'])
    assert (len(t), si, no) == (796, 372, 397), (len(t), si, no)
    falta = [f for f in bp.FEATURES_RF_CIR if f not in t[0]]
    assert not falta, falta
    con_dato = np.mean([np.isfinite(r['elev_dem_m']) and np.isfinite(r['evi_early'])
                        for r in t])
    assert con_dato > 0.98, con_dato
    # Con el modelo de verdad: 12 puntos de la tabla como si fueran tuyos
    rows = [dict(r, Borreguil=r['borreguil'], Duda='') for r in t
            if not r['duda']][:12]
    _, info = bp.train_rf(rows, default_model_path=str(ruta.parent / 'rf_sierra_nevada.joblib'),
                          propio=True, entrenamiento=t)
    assert info['sumado'] and info['n_tabla_quitados'] >= 12
    assert info['n_train'] == 769 - info['n_tabla_quitados'] + 12, info['n_train']
    # el modelo sumado usa las variables del de partida: una por grupo de repetidas (27)
    assert len(info['model_bundle']['features']) == 27


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
