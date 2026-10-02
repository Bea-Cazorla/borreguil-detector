"""Pruebas de las etiquetas del auto-entrenamiento.

Lo que puso una persona (ausencias de campo, puntos dudosos) no se toca ni se
promueve. Se ejecutan con `python tests/test_autoentrenamiento.py`; no entrenan
ningún modelo: solo comprueban qué etiqueta se le da a cada punto.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import borreguil_pipeline as bp   # noqa: E402

bp._imports()


def puntos():
    def p(i, proba, borreguil='', duda=''):
        return {'ID': f'p{i}', 'lon': -3.30 + i * 0.01, 'lat': 37.05,
                'rf_proba': proba, 'Borreguil': borreguil, 'Duda': duda}
    return [
        p(0, 0.90, borreguil='si'),               # verificado en campo
        p(1, 0.95, borreguil='no'),               # AUSENCIA de campo, pero el modelo la ve borreguil
        p(2, 0.90, duda='si'),                    # dudoso sin etiqueta
        p(3, 0.80),                               # candidato con probabilidad alta
        p(4, 0.30),                               # candidato con probabilidad baja
        p(5, 0.85, borreguil='si', duda='si'),    # borreguil de campo marcado como dudoso
        p(6, 0.40, borreguil='si'),               # pseudo-positivo de una iteración anterior
    ]


BASE = {'p0', 'p5'}                                # verdad-terreno fijada al ejecutar


def etiquetas(rows):
    return {r['ID']: (r['Borreguil'], r['Duda']) for r in rows}


def decidir(rows):
    for r in rows:
        r['decision'] = bp.decide(r, 0.5)


def test_solo_se_promueven_candidatos_sin_etiqueta_de_campo():
    rows = puntos()
    assert [r['ID'] for r in rows if bp.promovible(r, 0.70)] == ['p0', 'p3']
    # p1 (ausencia) y p2/p5 (dudosos) no, por mucha probabilidad que tengan
    assert not bp.promovible(rows[1], 0.70) and not bp.promovible(rows[2], 0.70)
    assert not bp.promovible(rows[5], 0.70)
    assert not bp.promovible({'ID': 'x', 'rf_proba': float('nan')}, 0.70)
    assert not bp.promovible({'ID': 'x'}, 0.70)


def test_acumular_conserva_ausencias_y_dudas():
    rows = puntos()
    s = bp.self_training_labels(rows, 0.70, 'add', BASE)
    assert etiquetas(rows) == {
        'p0': ('si', ''), 'p1': ('no', ''), 'p2': ('', 'si'), 'p3': ('si', ''),
        'p4': ('', ''), 'p5': ('si', 'si'), 'p6': ('si', '')}
    assert s['promoted'] == {'p0', 'p3'} and s['neg_campo'] == {'p1'}
    assert s['dudosos'] == {'p2', 'p5'}
    assert s['promoted'] - s['prev_pos'] == {'p3'}             # el único nuevo


def test_reemplazar_suelta_los_pseudo_positivos_antiguos_y_nada_mas():
    rows = puntos()
    bp.self_training_labels(rows, 0.70, 'replace', BASE)
    e = etiquetas(rows)
    assert e['p6'] == ('', '')                                 # pseudo-positivo antiguo
    assert e['p0'] == ('si', '') and e['p5'] == ('si', 'si')   # la verdad-terreno sigue
    assert e['p1'] == ('no', '') and e['p2'] == ('', 'si')     # y lo de campo, intacto
    assert e['p3'] == ('si', '')


def test_las_decisiones_distinguen_campo_de_automatico():
    rows = puntos()
    s = bp.self_training_labels(rows, 0.70, 'add', BASE)
    decidir(rows)
    bp.self_training_decisions(rows, s)
    dec = {r['ID']: (r['decision'], r['origin']) for r in rows}
    assert dec['p0'] == ('BORREGUIL VERIFICADO', 'verificado')
    assert dec['p1'] == ('NO BORREGUIL VERIFICADO', 'verificado')
    assert dec['p2'] == ('DUDOSO (campo)', 'dudoso')
    assert dec['p5'] == ('DUDOSO (campo)', 'dudoso')           # no pasa a «verificado»
    assert dec['p3'] == ('BORREGUIL (auto)', 'auto')
    assert dec['p6'] == ('BORREGUIL (auto)', 'auto')
    assert dec['p4'] == ('INCIERTO', 'predicho')


def test_repetir_iteraciones_no_degrada_las_etiquetas():
    rows = puntos()
    for _ in range(3):
        s = bp.self_training_labels(rows, 0.70, 'add', BASE)
        decidir(rows)
        bp.self_training_decisions(rows, s)
    e = etiquetas(rows)
    assert e['p1'] == ('no', '') and e['p2'] == ('', 'si') and e['p5'] == ('si', 'si')


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
