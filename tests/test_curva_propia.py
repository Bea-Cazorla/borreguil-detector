"""Pruebas de la curva con los puntos de la zona («3 · Modelo» → ¿Mejoraría con más
puntos?): mapeo_rf.curva_propia y cifras_curva, y su gráfica.

La curva repite la validación por zonas del modelo, con las mismas partes, entrenando
con el 25, el 50 y el 75 % de los puntos. Se ejecutan con
`python tests/test_curva_propia.py` (o con pytest); no necesitan red. Los datos son
los inventados de test_mapeo_modelo.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import sklearn.ensemble  # noqa: F401,E402  (primero, por Smart App Control)

import numpy as np                              # noqa: E402

import mapeo_rf as mr                           # noqa: E402
import point_profile as pp                      # noqa: E402
from test_mapeo_modelo import RAPIDO, tabla     # noqa: E402


def modelo(**kw):
    tab, grupos, _ = tabla(**kw)
    return tab, mr.entrenar_y_validar(tab, grupos, RAPIDO)


def test_la_curva_entrena_con_menos_puntos_y_el_100_es_el_modelo():
    tab, mod = modelo()
    avances = []
    curva = mr.curva_propia(tab, mod, al_avanzar=avances.append)
    partes = len(mod['pliegues'])
    assert mr.entrenamientos_curva(mod) == 3 * 3 * partes == len(avances)
    assert avances == sorted(avances) and avances[-1] == 1.0
    # 3 fracciones × 3 sorteos, más el 100 %
    assert len(curva['corridas']) == 10 and curva['firma'] == mod['firma']
    c = mr.cifras_curva(curva, mod)
    assert [f['fraccion'] for f in c['filas']] == [0.25, 0.5, 0.75, 1.0]
    assert [f['sorteos'] for f in c['filas']] == [3, 3, 3, 1]
    n = [f['n_puntos'] for f in c['filas']]
    assert n == sorted(n) and n[0] < n[-1] / 3
    # el 100 % es la validación del propio modelo: el mismo F1 y los mismos puntos
    assert abs(c['filas'][-1]['f1'] - mod['macro_f1']) < 1e-9
    assert round(n[-1]) == round(np.mean([p['n_entrena'] for p in mod['pliegues']]))
    # con «no borreguil», la medida es el AUC; la media, entre el peor y el mejor sorteo
    assert c['medida'] == 'auc'
    assert all(f['auc_min'] <= f['auc'] <= f['auc_max'] for f in c['filas'])
    assert c['lectura'] in ('sube', 'plana')
    # los datos se separan bien: hasta con un cuarto de los puntos acierta casi todo
    assert c['filas'][0]['auc'] > 0.9


def test_la_curva_sale_igual_cada_vez():
    tab, mod = modelo(n=(12, 12, 12), zonas=4)
    a = mr.cifras_curva(mr.curva_propia(tab, mod), mod)
    b = mr.cifras_curva(mr.curva_propia(tab, mod), mod)
    assert a == b


def test_cada_categoria_queda_en_la_parte_que_aprende():
    y = np.array([0] * 40 + [1] * 8 + [2] * 2)
    azar = np.random.default_rng(1)
    for fraccion in (0.25, 0.5, 0.75):
        elegidas = mr._parte_al_azar(np.arange(len(y)), y, fraccion, azar)
        assert len(set(elegidas)) == len(elegidas)
        assert set(y[elegidas]) == {0, 1, 2}               # al menos una de cada
        assert np.sum(y[elegidas] == 0) == round(40 * fraccion)


def test_no_se_calcula_si_la_tabla_ha_cambiado():
    tab, mod = modelo(n=(12, 12, 12), zonas=4)
    otra, _, _ = tabla(n=(12, 12, 11), zonas=4)
    try:
        mr.curva_propia(otra, mod)
        raise AssertionError('tenía que negarse')
    except ValueError as e:
        assert 'tabla ha cambiado' in str(e)
    viejo = {k: v for k, v in mod.items() if k != '_validacion'}
    try:
        mr.curva_propia(tab, viejo)
        raise AssertionError('tenía que negarse')
    except ValueError as e:
        assert 'versión anterior' in str(e)


def test_el_f1_sigue_la_regla_del_umbral():
    tab, mod = modelo(n=(12, 12, 12), zonas=4)
    curva = mr.curva_propia(tab, mod)
    mr.aplicar_regla(mod, 'manual', 0.3)
    c = mr.cifras_curva(curva, mod)
    assert abs(c['filas'][-1]['f1'] - mod['macro_f1']) < 1e-9
    mr.aplicar_regla(mod, 'tss')
    assert abs(mr.cifras_curva(curva, mod)['filas'][-1]['f1'] - mod['macro_f1']) < 1e-9


def test_sin_no_borreguil_la_medida_es_el_f1():
    tab, mod = modelo(n=(20, 20, 0))
    c = mr.cifras_curva(mr.curva_propia(tab, mod), mod)
    assert c['medida'] == 'f1' and all(f['auc'] is None for f in c['filas'])
    assert abs(c['filas'][-1]['f1'] - mod['macro_f1']) < 1e-9


def test_dice_si_todavia_sube():
    """De la mitad de los puntos a todos (el doble): si sube 0,01 o más, «sube»; si
    no, «plana»."""
    _, mod = modelo(n=(12, 12, 12), zonas=4)
    y = mod['_validacion']['y']
    es = (y != mr.CODIGO_AUSENCIA).astype(np.float32)
    tipo = np.where(y == mr.CODIGO_AUSENCIA, 1, y)

    def corrida(fraccion, p):
        return {'fraccion': fraccion, 'sorteo': 0, 'n_puntos': 100 * fraccion,
                'n_borreguiles': 30 * fraccion, 'p_borr': p.astype(np.float32),
                'tipo': tipo}

    ruido = np.random.default_rng(3).random(len(y)).astype(np.float32)
    flojo = 0.2 * es + 0.8 * ruido                 # acierta bastante, no todo
    bueno = 0.9 * es + 0.1 * ruido                 # separa del todo
    # se compara con la mitad, no con el 75 %, que aquí está tan flojo como la mitad
    sube = mr.cifras_curva({'corridas': [corrida(0.5, flojo), corrida(0.75, bueno),
                                         corrida(1.0, bueno)]}, mod)
    assert sube['mitad']['fraccion'] == 0.5
    assert sube['lectura'] == 'sube' and sube['ganancia'] >= mr.SIGUE_SUBIENDO
    plana = mr.cifras_curva({'corridas': [corrida(0.5, bueno), corrida(0.75, flojo),
                                          corrida(1.0, bueno)]}, mod)
    assert plana['lectura'] == 'plana' and plana['ganancia'] == 0


def test_grafica_de_la_curva_propia():
    tab, mod = modelo(n=(12, 12, 12), zonas=4)
    c = mr.cifras_curva(mr.curva_propia(tab, mod), mod)
    es = pp.chart_curva_propia(c, en=False).to_dict()
    assert len(es['layer']) == 4                     # franja, línea, puntos y rótulos
    texto = str(es)
    assert "replace(format(datum.value, '.3f'), '.', ',')" in texto    # coma decimal
    assert 'AUC en zonas nuevas' in texto and 'Peor – mejor sorteo' in texto
    datos = next(iter(es['datasets'].values()))
    assert [d['parte'] for d in datos] == ['25 %', '50 %', '75 %', '100 %']
    en = str(pp.chart_curva_propia(c, en=True).to_dict())
    assert 'AUC in new areas' in en and "format(datum.value, '.3f')" in en


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
