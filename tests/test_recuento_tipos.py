"""Barras con el nº de borreguiles de cada ambiente, humedad y pureza («Tipos y
distribuciones»). 25-11-2026, la autora: «esta me gustaría que estuviera para los
tipos de borreguil, antes había gráficos de barras con el n».
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import borreguil_pipeline as bp   # noqa: E402

bp._imports()

import i18n                       # noqa: E402
import point_profile as pp        # noqa: E402


def _p(decision, ambiente, humedad, pureza):
    return {'decision': decision, 'ambiente': ambiente, 'humedad': humedad,
            'pureza': pureza}


def _puntos():
    b = 'BORREGUIL VERIFICADO'
    return ([_p(b, 'ladera', 'seco', '—')] * 5 + [_p(b, 'arroyo', 'húmedo', '—')] * 3
            + [_p(b, 'laguna', '', '—')]
            # los no borreguiles no cuentan, aunque lleven tipo
            + [_p('NO BORREGUIL VERIFICADO', 'arroyo', 'húmedo', 'puro')] * 4)


def test_cuenta_cada_nivel_en_su_orden_y_los_sin_propuesta_aparte():
    cuentas = dict(pp.type_counts(_puntos()))
    assert cuentas['ambiente'] == [('arroyo', 3), ('laguna', 1), ('ladera', 5)]
    assert cuentas['humedad'] == [('húmedo', 3), ('seco', 5), (None, 1)]
    # sin Earth Engine no hay pureza: todos sin propuesta
    assert cuentas['pureza'] == [(None, 9)]
    # cada nivel suma todos los borreguiles
    assert all(sum(n for _, n in c) == 9 for c in cuentas.values())


def test_graficas_de_barras_con_su_numero():
    graficas, n = pp.chart_type_counts(_puntos())
    assert n == 9
    assert [t for t, _ in graficas] == ['Ambiente', 'Humedad', 'Pureza']
    d = graficas[1][1].to_dict()
    marcas = [c['mark']['type'] for c in d['layer']]
    assert marcas == ['bar', 'text']                      # barras y el número
    datos = [x for ds in d['datasets'].values() for x in ds]
    assert {(x['tipo'], x['n']) for x in datos} == {
        ('húmedo', 3), ('seco', 5), ('sin propuesta (—)', 1)}
    # la misma escala en los tres, de 0 a algo más que el total (sitio para el número)
    escalas = {str(g.to_dict()['layer'][0]['encoding']['x']['scale']) for _, g in graficas}
    assert len(escalas) == 1


def test_en_ingles():
    i18n.set_lang('en')
    try:
        graficas, _ = pp.chart_type_counts(_puntos())
        assert [t for t, _ in graficas] == ['Environment', 'Moisture', 'Purity']
        datos = [x for ds in graficas[2][1].to_dict()['datasets'].values() for x in ds]
        assert datos[0]['tipo'] == 'no proposal (—)'
    finally:
        i18n.set_lang('es')


def test_sin_borreguiles_no_hay_graficas():
    assert pp.chart_type_counts([_p('NO BORREGUIL VERIFICADO', 'ladera', 'seco', 'puro')]) == ([], 0)
