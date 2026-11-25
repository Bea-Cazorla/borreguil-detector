"""Pruebas de la curva de cuántos puntos hacen falta para el mapa de borreguil.

La curva (mapeo_rf.CURVA_PUNTOS_SN) se calculó con los 948 puntos revisados de Sierra
Nevada y el mismo procedimiento que el mapa de la app. La app la usa en «1 · Datos»
para decir si son bastantes puntos y en la Documentación. Se ejecutan con
`python tests/test_curva_puntos.py` (o con pytest); no necesitan red.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sklearn.ensemble  # noqa: F401,E402  (primero, por Smart App Control)

import mapeo_rf as mr             # noqa: E402
import point_profile as pp       # noqa: E402


def test_la_curva_esta_ordenada_y_sube():
    c = mr.CURVA_PUNTOS_SN
    assert [f[0] for f in c] == sorted(f[0] for f in c)
    assert all(f[3] <= f[2] <= f[4] for f in c)            # la media, dentro de la franja
    assert c[-1][2] > c[0][2] + 0.05                       # con más puntos acierta más
    # el mínimo y lo recomendado están en la curva, y lo recomendado acierta más
    ns = {f[0]: f for f in c}
    assert mr.MINIMO_PUNTOS[0] in ns and mr.RECOMENDADO_PUNTOS[0] in ns
    assert ns[mr.RECOMENDADO_PUNTOS[0]][2] > ns[mr.MINIMO_PUNTOS[0]][2]
    # el mínimo queda a menos de 0,02 de todos los puntos; lo recomendado, a menos de 0,01
    assert c[-1][2] - ns[mr.MINIMO_PUNTOS[0]][2] < 0.02
    assert c[-1][2] - ns[mr.RECOMENDADO_PUNTOS[0]][2] < 0.01


def test_auc_esperado():
    assert mr.auc_esperado(None) is None and mr.auc_esperado(10) is None
    assert mr.auc_esperado(100) == 0.951 and mr.auc_esperado(5000) == mr.CURVA_PUNTOS_SN[-1][2]
    a = mr.auc_esperado(70)                                  # entre 60 y 80
    assert 0.936 < a < 0.945


def test_lectura_puntos():
    lp = mr.lectura_puntos
    assert lp(1, 0)['nivel'] == 'sin_referencia'
    assert (lp(1, 0)['faltan'], lp(1, 0)['faltan_borreguiles']) == (99, 33)
    assert lp(35, 10)['nivel'] == 'pocos'
    assert lp(60, 20)['nivel'] == 'bajo_minimo'
    # con los puntos pero sin los borreguiles no se llega: faltan solo borreguiles
    l = lp(120, 30)
    assert l['nivel'] == 'bajo_minimo' and (l['faltan'], l['faltan_borreguiles']) == (0, 3)
    l = lp(120, 40)
    assert l['nivel'] == 'minimo' and (l['faltan'], l['faltan_borreguiles']) == (130, 43)
    l = lp(900, 300)
    assert l['nivel'] == 'recomendado' and l['faltan'] == 0 and l['auc'] == 0.969
    assert (mr.MINIMO_AUSENCIAS, mr.RECOMENDADO_AUSENCIAS) == (67, 167)


def test_lectura_puntos_cuenta_las_ausencias():
    """25-11-2026: con muchos borreguiles y pocas ausencias ya no se llega al mínimo."""
    lp = mr.lectura_puntos
    l = lp(106, 100)                               # 100 borreguiles y 6 ausencias
    assert l['nivel'] == 'bajo_minimo'
    assert (l['faltan_borreguiles'], l['faltan_ausencias']) == (0, 61)
    l = lp(120, 40)                                # 40 y 80: mínimo; faltan para lo recomendado
    assert l['nivel'] == 'minimo' and (l['faltan_borreguiles'], l['faltan_ausencias']) == (43, 87)
    l = lp(300, 200)                               # 200 borreguiles y 100 ausencias
    assert l['nivel'] == 'minimo' and (l['faltan_borreguiles'], l['faltan_ausencias']) == (0, 67)
    # las ausencias se pueden dar aparte (p. ej. si hubiera muestras de otras categorías)
    assert lp(120, 40, n_ausencia=60)['nivel'] == 'bajo_minimo'
    assert lp(1, 0)['faltan_ausencias'] == 66


def test_grafica_con_y_sin_tus_puntos():
    marcas = [(100, 'Mínimo'), (250, 'Recomendado')]
    sin = pp.chart_curva_puntos(mr.CURVA_PUNTOS_SN, marcas=marcas, en=False).to_dict()
    con = pp.chart_curva_puntos(mr.CURVA_PUNTOS_SN, n=60, marcas=marcas, en=False).to_dict()
    assert len(con['layer']) == len(sin['layer']) + 2          # la línea y su texto
    textos = str(con)
    assert 'Tus puntos: 60' in textos and 'Mínimo: 100 puntos' in textos
    assert "replace(format(datum.value, '.2f'), '.', ',')" in textos   # coma decimal
    # fuera de la escala, la línea se queda en el borde
    lejos = pp.chart_curva_puntos(mr.CURVA_PUNTOS_SN, n=5000, en=True).to_dict()
    assert 'Your points: 5000' in str(lejos)
    assert pp.chart_curva_puntos(mr.CURVA_PUNTOS_SN, n=0, en=True).to_dict()


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
