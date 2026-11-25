"""Pruebas de cómo se MUESTRAN los datos: 3 decimales y nombres largos sin cortar.

Se ejecutan con `python tests/test_presentacion.py`. Lo que se comprueba es lo que
se ve (globos del mapa, Excel); los valores guardados conservan toda su precisión.
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import borreguil_pipeline as bp   # noqa: E402

bp._imports()

ID_LARGO = 'all_borreguil_lagunas500ptos_epsg25830.21'


def punto(**extra):
    r = {'ID': ID_LARGO, 'lon': -3.3312, 'lat': 37.0493123456,
         'decision': 'BORREGUIL PROBABLE', 'rf_proba': 0.9333333,
         'mat_signature': 'BORREGUIL_LIKE',
         'elev_dem_m': 2935.835205078125, 'slope_deg': 14.577339172363281,
         'ndvi_late': 0.2185155749320984, 'clre_late': 0.28997838497161865,
         'ambiente': 'ladera', 'humedad': 'seco', 'pureza': 'mixto-agua',
         'tipo_borreguil': 'ladera · seco · mixto-agua'}
    r.update(extra)
    return r


def test_numeros_con_tres_decimales():
    assert bp.fmt_num(2935.835205078125) == '2935.835'
    assert bp.fmt_num(0.28997838497161865) == '0.290'
    assert bp.fmt_num('0.2185155749320984') == '0.219'      # en un KML llegan como texto
    assert bp.fmt_num(3) == '3.000'
    for vacio in (None, float('nan'), '', 'abc'):
        assert bp.fmt_num(vacio) == '—'


def test_anios_abreviados_son_lo_contrario_de_parse_years():
    assert bp.abreviar_anios(range(2017, 2026)) == '2017-2025'
    assert bp.abreviar_anios([2023, 2024]) == '2023-2024'
    assert bp.abreviar_anios([2020, 2017, 2019, 2024]) == '2017, 2019-2020, 2024'
    assert bp.abreviar_anios([2024]) == '2024' and bp.abreviar_anios([]) == ''
    for texto in ('2017-2025', '2017, 2019-2020, 2024', '2024'):
        assert bp.abreviar_anios(bp.parse_years(texto)) == texto


def test_ids_largos_abreviados_conservando_lo_que_los_distingue():
    ids = [f'all_borreguil_lagunas500ptos_epsg25830.{n}' for n in (1, 21, 41, 441)]
    cortos = bp.abreviar_ids(ids)
    assert cortos == ['…epsg25830.1', '…epsg25830.21', '…epsg25830.41', '…epsg25830.441']
    assert len(set(cortos)) == len(set(ids))                # siguen distinguiéndose
    # Los cortos no se tocan, tampoco cuando conviven con los largos.
    assert bp.abreviar_ids(['p1', 'campo_22']) == ['p1', 'campo_22']
    assert bp.abreviar_ids(['rand_0001'] + ids)[:2] == ['rand_0001', '…epsg25830.1']
    assert bp.abreviar_ids([7, None]) == ['7', '']
    # Un nombre largo suelto se acorta por el medio, conservando el final.
    assert bp.abreviar_ids(['p1', ids[1]]) == ['p1', 'all_bo…ptos_epsg25830.21']
    # Si al acortar dos nombres distintos quedaran iguales, no se acorta ninguno.
    parecidos = ['A' * 20 + '1' + 'B' * 20, 'A' * 20 + '2' + 'B' * 20]
    assert bp.abreviar_ids(parecidos) == parecidos


def test_globo_del_mapa_con_tres_decimales():
    h = bp.popup_html(punto())
    assert 'Altitud: 2935.835 m' in h and 'Pendiente: 14.577°' in h
    assert 'NDVI: 0.219' in h and 'Clre: 0.290' in h and 'RF: 93%' in h
    assert '2935.83520' not in h and '0.21851' not in h     # nada de colas largas


def test_nombre_con_puntos_de_corte_en_los_separadores():
    h = bp.nombre_html(ID_LARGO)
    assert h == 'all_<wbr>borreguil_<wbr>lagunas500ptos_<wbr>epsg25830.<wbr>21'
    assert h.replace('<wbr>', '') == ID_LARGO          # el texto es el nombre exacto
    assert bp.nombre_html('a<b>&c') == 'a&lt;b&gt;&amp;c'
    assert bp.nombre_html(17) == '17'


def test_globo_parte_los_nombres_largos():
    h = bp.popup_html(punto())
    assert bp.nombre_html(ID_LARGO) in h
    # Sin esta regla el nombre, que no tiene espacios, se salía del globo.
    assert 'overflow-wrap:anywhere' in h and 'max-width:280px' in h


def test_globo_muestra_el_tipo_y_si_esta_revisado():
    h = bp.popup_html(punto())
    assert 'Tipo: <b>ladera · seco · mixto-agua</b>' in h and 'revisado' not in h
    h = bp.popup_html(punto(tipo_revisado='si'))
    assert '✓ revisado' in h
    assert 'Tipo' not in bp.popup_html(punto(tipo_borreguil='', decision='NO BORREGUIL'))


def test_decision_en_texto_normal():
    # En la pantalla (tarjetas, Tabla, ficha y globos) la decisión va en texto normal;
    # en los puntos y las descargas sigue en mayúsculas, que es como la lee la app.
    assert '<b>Borreguil probable</b>' in bp.popup_html(punto())
    assert '<b>No borreguil verificado</b>' in bp.popup_html(
        punto(decision='NO BORREGUIL VERIFICADO'))
    assert '<b>Borreguil (auto)</b>' in bp.popup_html(punto(decision='BORREGUIL (auto)'))
    import i18n
    try:
        i18n._LANG = 'es'
        assert i18n.tv_decision('BORREGUIL VERIFICADO') == 'Borreguil verificado'
        assert i18n.tv_decision('POSIBLE BORREGUIL') == 'Posible borreguil'
        i18n._LANG = 'en'
        assert i18n.tv_decision('NO BORREGUIL VERIFICADO') == 'Verified non-borreguil'
        assert i18n.tv_decision('NO BORREGUIL') == 'Not a borreguil'
        assert i18n.tv_decision('') == '' and i18n.tv_decision(None) is None
    finally:
        i18n._LANG = 'es'


def test_globo_en_otro_idioma():
    en = {'BORREGUIL PROBABLE': 'LIKELY BORREGUIL',
          'ladera · seco · mixto-agua': 'slope · dry · mixed-water'}
    h = bp.popup_html(punto(tipo_revisado='si'), lambda v: en.get(v, v),
                      {'tipo': 'Type', 'revisado': 'reviewed', 'altitud': 'Elevation',
                       'pendiente': 'Slope'})
    assert '<b>Likely borreguil</b>' in h and 'Type: <b>slope · dry · mixed-water</b>' in h
    assert '✓ reviewed' in h and 'Elevation: 2935.835 m' in h and 'Slope: 14.577°' in h
    # el «patrón» (BORREGUIL_LIKE…) era un código interno: ya no sale
    assert 'BORREGUIL_LIKE' not in h and 'Pattern' not in h


def test_en_el_globo_el_punto_se_llama_por_su_numero():
    """25-11-2026, la autora: «este nombre de la capa cámbialo»."""
    ids = [f'all_borreguil_lagunas500ptos_epsg25830.{n}' for n in (1, 21, 663)]
    assert bp.nombres_de_punto(ids) == ['Punto 1', 'Punto 21', 'Punto 663']
    assert bp.nombres_de_punto(ids, 'Point')[2] == 'Point 663'
    # Los cortos y los de otra capa se quedan como están
    assert bp.nombres_de_punto(['SN_001', 'campo.5', None] + ids[:1]) == [
        'SN_001', 'campo.5', '', 'Punto 1']
    otra = 'otra_capa_de_puntos_de_campo.7'
    assert bp.nombres_de_punto(ids + [otra])[-1] == otra
    # Si dos puntos distintos se llamaran igual, no se cambia ninguno
    dobles = [ids[0], 'Punto 1']
    assert bp.nombres_de_punto(dobles) == dobles
    assert bp.nombres_de_punto(['p1', 'p2']) == ['p1', 'p2']
    # En la tabla de «Revisar puntos», solo el número
    assert bp.nombres_de_punto(ids + ['SN_001'], punto='') == ['1', '21', '663', 'SN_001']


def test_globo_con_las_variables_que_mas_pesan():
    """25-11-2026, la autora: «las variables que aparezcan que sean las tres que más
    peso tienen en el RF, como en el otro sitio» (la tabla de Revisar puntos)."""
    h = bp.popup_html(punto(), nombre='Punto 21',
                      variables=[('Altitud (m)', 'elev_dem_m'),
                                 ('Caída NDVI estacional', 'ndvi_drop'),
                                 ('NDVI fin verano', 'ndvi_late')],
                      num=lambda v: '—' if v is None else f'{v:.3f}'.replace('.', ','))
    assert '<b>Punto 21</b>' in h and ID_LARGO not in h
    assert '<i>Las que más pesan en el modelo:</i><br>Altitud (m)' in h      # y lo dice
    assert 'Altitud (m): 2935,835<br>' in h and 'Caída NDVI estacional: —' in h
    assert 'NDVI fin verano: 0,219' in h
    assert 'Clre' not in h and 'Pendiente' not in h      # solo las que se piden


def test_cajas_con_las_variables_que_mas_pesan_y_su_peso():
    """25-11-2026, la autora: «aquí deberían mostrarse las que más peso tienen en el
    modelo y que lo indique» (Tipos y distribuciones)."""
    import point_profile as pp
    imp = {'elev_dem_m': 0.30, 'ndvi_drop': 0.20, 'ndvi_late': 0.12, 'evi_late': 0.10,
           'slope_deg': 0.05}
    v = pp.vars_boxplots(imp, n=3)
    assert [f for f, _ in v][0] == 'elev_dem_m' and len(v) == 3
    assert v[0][1] == f"{pp.etiqueta('elev_dem_m')} · 39 %"      # 0,30 de 0,77
    assert pp.vars_boxplots(None) == list(pp.BOX_VARS)
    rows = [dict(punto(), decision='BORREGUIL PROBABLE'),
            dict(punto(), decision='NO BORREGUIL', elev_dem_m=2500.0)]
    ch, n_borr = pp.chart_boxplots(rows, v, en=False)
    d = ch.to_dict()
    spec = str(d)
    assert n_borr == 1 and "replace(format(datum.value, '~r'), '.', ',')" in spec
    assert "'axis': None" in spec               # sin los nombres que se pisaban debajo
    assert v[0][1] in spec
    # Al pasar el ratón: en palabras y con 3 decimales, sin «min of null» (25-11-2026)
    assert 'boxplot' not in spec and 'of null' not in spec
    datos = [x for ds in d['datasets'].values() for x in ds]
    alt_b = [x for x in datos if x['parte'] == 'caja' and x['variable'] == v[0][1]
             and x['clase'] == 'Borreguil'][0]
    assert alt_b['med_txt'] == '2935,835' and alt_b['n'] == '1'
    assert "'title': 'Mediana'" in spec and "'title': 'Bigotes'" in spec


def _puntos_vistas():
    """30 borreguiles (húmedos de arroyo y secos de ladera) y 30 del resto, con la
    altitud y la caída del NDVI más altas en los borreguiles."""
    import random
    azar = random.Random(3)
    rows = []
    for i in range(30):
        hum = i % 2 == 0
        rows.append({'ID': f'b{i}', 'decision': 'BORREGUIL PROBABLE',
                     'humedad': 'húmedo' if hum else 'seco',
                     'ambiente': 'arroyo' if hum else 'ladera',
                     'elev_dem_m': 2800 + azar.gauss(0, 80), 'ndvi_drop': 0.12 + azar.gauss(0, 0.02),
                     'slope_deg': 12 + azar.gauss(0, 3), 'rf_proba': 0.8})
    for i in range(30):
        rows.append({'ID': f'n{i}', 'decision': 'NO BORREGUIL',
                     'elev_dem_m': 2500 + azar.gauss(0, 80), 'ndvi_drop': 0.04 + azar.gauss(0, 0.02),
                     'slope_deg': 20 + azar.gauss(0, 3), 'rf_proba': 0.1})
    rows[0]['humedad'] = ''                      # un borreguil sin humedad
    return rows


VARS_VISTAS = [('elev_dem_m', 'Altitud (m) · 40 %'), ('ndvi_drop', 'Caída NDVI · 30 %'),
               ('slope_deg', 'Pendiente (°) · 10 %')]


def test_grupos_para_comparar():
    """25-11-2026, la autora: «sí, hazlo con comparar por tipo»."""
    import point_profile as pp
    rows = _puntos_vistas()
    g, orden, resto = pp.grupos_puntos(rows, 'borreguil', en=False)
    assert [o[0] for o in orden] == ['Borreguil', 'No borreguil'] and resto == 'No borreguil'
    assert g.count('Borreguil') == 30
    g, orden, resto = pp.grupos_puntos(rows, 'humedad', en=False)
    assert [o[0] for o in orden] == ['húmedo', 'seco', 'sin tipo', 'No borreguil']
    assert g[0] == 'sin tipo' and g.count('húmedo') == 14 and g.count('No borreguil') == 30
    g, orden, _ = pp.grupos_puntos(rows, 'ambiente', en=True,
                                   traduce={'arroyo': 'stream', 'ladera': 'slope'}.get)
    assert [o[0] for o in orden] == ['stream', 'slope', 'Not a borreguil']
    # «No borreguil» es solo eso: los inciertos y los dudosos se quedan fuera
    # (25-11-2026, la autora: «¿el resto aquí quiere decir no borreguil? Pon no borreguil»)
    otros = rows + [{'decision': 'INCIERTO', 'elev_dem_m': 2000.0},
                    {'decision': 'DUDOSO (campo)', 'elev_dem_m': 2000.0}]
    for comparar in pp.COMPARAR:
        g, orden, _ = pp.grupos_puntos(otros, comparar, en=False)
        assert g[-2:] == [None, None] and None not in [o[0] for o in orden]
        assert g.count('No borreguil') == 30
    ch, _ = pp.chart_boxplots(otros, VARS_VISTAS, en=False)
    clases = {x.get('clase') for ds in ch.to_dict()['datasets'].values() for x in ds}
    assert clases == {'Borreguil', 'No borreguil'}, clases
    try:
        pp.grupos_puntos(rows, 'pureza')
        raise AssertionError('tenía que negarse')
    except ValueError:
        pass


def test_las_cuatro_vistas_se_dibujan_comparando_por_tipo():
    import point_profile as pp
    rows = _puntos_vistas()
    for comparar in pp.COMPARAR:
        ch, n_borr = pp.chart_boxplots(rows, VARS_VISTAS, en=False, comparar=comparar)
        assert n_borr == 30 and ch.to_dict()
        assert pp.chart_densidades(rows, VARS_VISTAS, en=False, comparar=comparar).to_dict()
        assert pp.chart_diferencias(rows, VARS_VISTAS, en=False, comparar=comparar).to_dict()
        ch = pp.chart_dos_variables(rows, 'elev_dem_m', 'ndvi_drop', 'Altitud (m)',
                                    'Caída NDVI', en=False, comparar=comparar)
        assert "'title': 'Probabilidad del modelo'" in str(ch.to_dict())
    # las curvas: 80 puntos por grupo y variable, y cada una encierra un área de 1
    d = pp.chart_densidades(rows, VARS_VISTAS, en=False).to_dict()
    datos = [x for ds in d['datasets'].values() for x in ds]
    curva = [x for x in datos if x['variable'] == 'Altitud (m) · 40 %'
             and x['grupo'] == 'Borreguil']
    assert len(curva) == 80
    paso = curva[1]['x'] - curva[0]['x']
    assert 0.5 < sum(x['densidad'] for x in curva) * paso <= 1.01


def test_diferencias_en_desviaciones_tipicas():
    import point_profile as pp
    rows = _puntos_vistas()
    f = pp.tabla_diferencias(rows, VARS_VISTAS, en=False)
    por = {x['variable']: x for x in f}
    assert por['Altitud (m) · 40 %']['d'] > 1 and por['Pendiente (°) · 10 %']['d'] < -1
    assert [x['variable'] for x in f] == [lab for _, lab in VARS_VISTAS]
    f = pp.tabla_diferencias(rows, VARS_VISTAS, en=False, comparar='humedad')
    # «sin tipo» tiene 1 punto: menos de MIN_GRUPO, no se dibuja (su mediana engañaría)
    assert {x['grupo'] for x in f} == {'húmedo', 'seco'} and pp.MIN_GRUPO == 5
    d = pp.chart_densidades(rows, VARS_VISTAS, en=False, comparar='humedad').to_dict()
    assert {x['grupo'] for ds in d['datasets'].values() for x in ds} == {'húmedo', 'seco',
                                                                         'No borreguil'}
    assert 'sin tipo' not in str(d)                            # tampoco en la leyenda
    # sin puntos del resto no hay con qué comparar
    assert pp.tabla_diferencias(rows[:30], VARS_VISTAS, en=False) == []
    assert pp.chart_diferencias(rows[:30], VARS_VISTAS, en=False) is None


def test_variables_para_dos_a_la_vez():
    import point_profile as pp
    rows = _puntos_vistas()
    lv = pp.lista_variables(rows, {'ndvi_drop': 0.6, 'elev_dem_m': 0.4, 'evi_late': 0.1},
                            en=False, todas=['elev_dem_m', 'slope_deg', 'ndvi_late'])
    assert [f for f, _ in lv] == ['ndvi_drop', 'elev_dem_m', 'slope_deg']   # evi: sin valor
    assert lv[0][1].endswith('· 55 %') and lv[2][1].endswith('(no la usa el modelo)')


def test_globo_no_se_rompe_con_textos_raros_ni_huecos():
    h = bp.popup_html({'ID': '<b>punto</b> & co', 'lon': 0, 'lat': 0})
    assert '&lt;b&gt;punto&lt;/<wbr>b&gt; &amp; co' in h and '<b>punto</b>' not in h
    assert '<b>Sin predicción</b>' in h and 'RF: —' in h and 'Altitud: — m' in h
    nan = float('nan')
    h = bp.popup_html(punto(decision=nan, mat_signature=nan, rf_proba=nan, elev_dem_m=nan))
    assert '<b>Sin predicción</b>' in h and 'Patrón' not in h and 'nan' not in h.lower()


def test_mapa_html_guardado_usa_el_mismo_globo():
    with tempfile.TemporaryDirectory() as d:
        ruta = os.path.join(d, 'mapa.html')
        bp.save_map([punto(), punto(ID='otro', lon=-3.32, tipo_revisado='si')], ruta)
        html = Path(ruta).read_text(encoding='utf-8')
    assert '2935.835' in html and '2935.8352' not in html
    assert 'overflow-wrap:anywhere' in html and 'revisado' in html


def test_excel_tres_decimales_y_columnas_a_lo_ancho_del_nombre():
    from openpyxl import load_workbook
    from openpyxl.utils import get_column_letter
    with tempfile.TemporaryDirectory() as d:
        ruta = os.path.join(d, 'puntos.xlsx')
        bp.save_xlsx([punto(), punto(ID='corto')], ruta)
        ws = load_workbook(ruta).active
        cab = {c.value: i for i, c in enumerate(ws[1], 1)}
        alt = ws.cell(row=2, column=cab['Altitud (m)'])
        assert alt.value == 2935.835205078125           # el dato, entero
        assert alt.number_format == '0.000'             # lo que se ve, 3 decimales
        assert ws.cell(row=2, column=cab['lat']).number_format == '0.000000'
        ancho_id = ws.column_dimensions[get_column_letter(cab['ID'])].width
        assert ancho_id >= len(ID_LARGO), ancho_id
        ancho_cab = ws.column_dimensions[
            get_column_letter(cab['Ambiente (propuesta app)'])].width
        assert ancho_cab >= len('Ambiente (propuesta app)')


# ---- Ficha del punto: este punto frente a borreguiles y no borreguiles -------------
import i18n                         # noqa: E402
import point_profile as pp          # noqa: E402


def _q(p10, p25, p50, p75, p90):
    return {'p10': p10, 'p25': p25, 'p50': p50, 'p75': p75, 'p90': p90}


def test_variables_clave_una_por_grupo_y_por_peso_del_grupo():
    # tres índices de verdor (mismo grupo) que por separado pesan poco, juntos más que
    # la altitud; la pendiente sola
    imp = {'ndvi_late': 0.08, 'evi_early': 0.10, 'clre_late': 0.07,
           'elev_dem_m': 0.20, 'slope_deg': 0.05}
    c = pp.variables_clave(imp, n=3)
    assert [k['feat'] for k in c] == ['evi_early', 'elev_dem_m', 'slope_deg'], c
    assert abs(c[0]['peso'] - 0.25 / 0.50) < 1e-9 and c[0]['parecidas'] == 2
    assert c[1]['parecidas'] == 0
    # solo las que tienen valores de referencia
    c = pp.variables_clave(imp, disponibles={'ndvi_late', 'slope_deg'}, n=8)
    assert [k['feat'] for k in c] == ['ndvi_late', 'slope_deg']


def test_variables_clave_sin_pesos_usa_las_de_por_defecto():
    c = pp.variables_clave(None, n=3)
    assert [k['feat'] for k in c] == pp.CLAVE_POR_DEFECTO[:3]
    assert all(k['peso'] is None for k in c)


def test_los_grupos_de_parecidas_no_se_repiten():
    vistas = [f for g in pp.GRUPOS_PARECIDAS for f in g]
    assert len(vistas) == len(set(vistas))
    assert all(f in pp.FEATURE_LABELS and f in pp.FEATURE_LABELS_EN for f in vistas)


def test_posicion_del_punto_entre_los_borreguiles():
    si = _q(0.1, 0.3, 0.4, 0.5, 0.7)
    assert pp.percentil_entre_borreguiles(0.4, si) == 50
    assert pp.percentil_entre_borreguiles(0.3, si) == 25
    assert abs(pp.percentil_entre_borreguiles(0.45, si) - 62.5) < 1e-9
    assert pp.percentil_entre_borreguiles(0.05, si) == 5      # más bajo que casi todos
    assert pp.percentil_entre_borreguiles(0.9, si) == 95
    assert pp.percentil_entre_borreguiles(None, si) is None


def test_grafica_del_punto_con_el_modelo_de_sierra_nevada():
    b = bp.load_model_bundle(Path(__file__).resolve().parent.parent / 'rf_sierra_nevada.joblib')
    rs = dict(b['ref_stats'], importances=b['importances'], model_features=b['features'])
    sel = punto(evi_early=0.5, ndvi_drop=0.1, ndwi_sd=None)
    ch, recs = pp.chart_punto_vs_borreguiles(sel, rs, n=8)
    assert ch is not None and len(recs) == 8
    feats = [r['feat'] for r in recs]
    # modelo del 20-11-2026, con una variable por grupo de repetidas (27, sin TWI) y el
    # radar de Earth Engine: la altitud, la caída del NDVI y el cociente VV − VH del radar; la
    # variabilidad del NDWI y el NDVI de inicio, entre las ocho
    assert feats[:3] == ['elev_dem_m', 'ndvi_drop', 's1_ratio_mean'], feats
    assert {'ndvi_early', 'ndwi_sd'} <= set(feats) and 'evi_early' not in feats
    # una por grupo: ninguna pareja de parecidas
    assert len({pp.grupo_de(f) for f in feats}) == 8
    assert all(f in b['features'] for f in feats) and len(b['features']) == 27
    falta = next(r for r in recs if r['feat'] == 'ndwi_sd')
    assert falta['val'] is None and falta['pct'] is None and falta['texto'] == 'sin dato'
    alt_ = next(r for r in recs if r['feat'] == 'elev_dem_m')
    si = b['ref_stats']['features']['elev_dem_m']['si']
    # (en español, con coma decimal: lo pidió Bea el 24-11-2026)
    assert alt_['rango'] == f"{si['p25']:.3f}–{si['p75']:.3f}".replace('.', ',')
    tabla = pp.tabla_punto_vs_borreguiles(recs)
    fila = tabla[tabla['Variable'] == 'Altitud (m)'].iloc[0]
    assert fila['Este punto'] == '2935,835' and fila['Borreguiles (mitad central)'] == alt_['rango']
    assert list(tabla['Variable']) == [r['label'] for r in recs]
    assert tabla[tabla['Variable'] == pp.etiqueta('ndwi_sd')].iloc[0]['Este punto'] == '—'
    # el no borreguil típico (24-11-2026): la mediana de los no borreguiles, en la misma
    # escala de 0 a 100 y en la tabla
    no = b['ref_stats']['features']['elev_dem_m']['no']
    assert alt_['no_med'] == no['p50']
    assert alt_['pct_no'] == pp.percentil_entre_borreguiles(no['p50'], si)
    assert fila['No borreguil típico'] == f"{no['p50']:.3f}".replace('.', ',')
    assert falta['pct_no'] is not None                     # aunque el punto no tenga dato
    assert '"diamond"' in ch.to_json()
    ch.to_dict()                                           # Vega-Lite válido
    i18n.set_lang('en')
    try:
        _, recs_en = pp.chart_punto_vs_borreguiles(sel, rs, n=3)
        assert recs_en[0]['label'] == 'Elevation (m)'
    finally:
        i18n.set_lang('es')


def test_leyenda_de_la_grafica_del_punto():
    html = pp.leyenda_punto_vs_borreguiles()
    # los dos colores del punto y la franja/línea de los borreguiles, cada uno explicado
    for color in (pp.COL_PT, pp.COL_OUT, pp.COL_BORR, pp.COL_NO):
        assert color in html, color
    for texto in ('Este punto: dentro de lo habitual', 'más bajo o más alto que casi todos',
                  'Mitad central de los borreguiles verdaderos (25–75)',
                  'El borreguil típico (50)', 'El no borreguil típico, en la misma escala',
                  'Fila sin punto'):
        assert texto in html, texto
    i18n.set_lang('en')
    try:
        en = pp.leyenda_punto_vs_borreguiles()
        assert 'This point: within the usual' in en and 'Este punto' not in en
        assert 'The typical non-borreguil' in en
    finally:
        i18n.set_lang('es')


def test_grafica_del_punto_sin_valores_de_referencia():
    assert pp.chart_punto_vs_borreguiles(punto(), {}) == (None, [])


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


def test_recuento_por_decision_en_texto_normal_y_con_los_colores_del_mapa():
    """25-11-2026: el recuento va en «Resumen de los puntos», debajo de la
    probabilidad; salía en mayúsculas y cortado («NO BORREGUIL V…»), con colores que no
    eran los del mapa y ordenado por tamaño."""
    import point_profile as pp
    rows = ([{'decision': 'NO BORREGUIL VERIFICADO'}] * 5 + [{'decision': 'NO BORREGUIL'}]
            + [{'decision': 'BORREGUIL PROBABLE'}] * 3 + [{'decision': 'BORREGUIL VERIFICADO'}])
    d = pp.chart_decision_counts(rows).to_dict()
    datos = [x for ds in d['datasets'].values() for x in ds]
    # de más a menos borreguil, no por tamaño
    assert [x['decision'] for x in datos] == ['Borreguil verificado', 'Borreguil probable',
                                              'No borreguil', 'No borreguil verificado']
    por = {x['decision']: x for x in datos}
    assert por['Borreguil probable']['n'] == 3 and por['Borreguil probable']['texto'] == '3 (30 %)'
    assert por['Borreguil probable']['color'] == bp.decision_color('BORREGUIL PROBABLE')
    assert "'labelLimit': 0" in str(d)                  # el nombre entero, sin cortar
    assert pp.chart_decision_counts([{'decision': 'INCIERTO'}] + [{'decision': 'NO BORREGUIL'}]
                                    * 300).to_dict()['datasets']   # y «<1 %» no rompe nada
    raros = [x for ds in pp.chart_decision_counts(
        [{'decision': 'INCIERTO'}] + [{'decision': 'NO BORREGUIL'}] * 300).to_dict()[
        'datasets'].values() for x in ds]
    assert raros[0]['texto'] == '1 (<1 %)'


def _muestras_categoria():
    """Muestras de una tabla de entrenamiento del mapa (nivel «humedad»)."""
    import random
    azar = random.Random(3)
    filas = []
    for cat, base, n in (('húmedo', 0.4, 12), ('seco', 0.1, 10), ('no borreguil', -0.2, 15)):
        for i in range(n):
            filas.append({'ID': f'capa_de_prueba_larga.{len(filas) + 1}', 'categoria': cat,
                          'ndmi_early': base + azar.gauss(0, 0.05),
                          'ndvi_early': 0.5 + base + azar.gauss(0, 0.05)})
    return filas


def test_graficos_por_categoria_del_modelo_con_los_colores_del_mapa():
    """25-11-2026, la autora: en «5 · Peso» de Mapa Modelo RF, «además de peso, podría
    mostrar gráficos de las variables por tipo de borreguil como en los puntos de
    verdad terreno»."""
    import json
    import mapeo_raster as mra
    import point_profile as pp
    filas = _muestras_categoria()
    g, orden, resto = pp.grupos_puntos(filas, 'categoria', en=False)
    # el orden del catálogo del mapa, y «no borreguil» al final: es con lo que se compara
    assert [o[0] for o in orden] == ['húmedo', 'seco', 'no borreguil'] and resto == 'no borreguil'
    assert [o[1] for o in orden] == [mra.color(c) for c in ('húmedo', 'seco', 'no borreguil')]
    assert g.count('húmedo') == 12
    g, orden, resto = pp.grupos_puntos(filas, 'categoria', en=True,
                                       traduce={'húmedo': 'wet', 'seco': 'dry',
                                                'no borreguil': 'not a borreguil'}.get)
    assert [o[0] for o in orden] == ['wet', 'dry', 'not a borreguil']
    vars_ = [('ndmi_early', 'NDMI'), ('ndvi_early', 'NDVI')]
    for ch in (pp.chart_boxplots(filas, vars_, en=False, comparar='categoria')[0],
               pp.chart_densidades(filas, vars_, en=False, comparar='categoria'),
               pp.chart_diferencias(filas, vars_, en=False, comparar='categoria'),
               pp.chart_dos_variables(filas, 'ndmi_early', 'ndvi_early', 'NDMI', 'NDVI',
                                      en=False, comparar='categoria')):
        d = json.dumps(ch.to_dict())
        assert mra.color('húmedo') in d and mra.color('seco') in d
    # las diferencias, frente a «no borreguil»: lo húmedo, por encima en NDMI
    dif = {(x['variable'], x['grupo']): x['d'] for x in
           pp.tabla_diferencias(filas, vars_, en=False, comparar='categoria')}
    assert dif[('NDMI', 'húmedo')] > dif[('NDMI', 'seco')] > 0
    # un nivel combinado: «ladera · seco», con su color, y traducido por partes
    comb = [{'categoria': 'ladera · seco', 'x': 1.0}, {'categoria': 'arroyo · húmedo', 'x': 2.0}]
    g, orden, resto = pp.grupos_puntos(comb, 'categoria', en=True,
                                       traduce={'ladera': 'slope', 'seco': 'dry',
                                                'arroyo': 'stream', 'húmedo': 'wet'}.get)
    assert orden == [('stream · wet', mra.color('arroyo · húmedo')),
                     ('slope · dry', mra.color('ladera · seco'))] and resto is None
    # sin «no borreguil» no hay con qué comparar las diferencias
    assert pp.chart_diferencias(comb * 5, [('x', 'x')], en=False, comparar='categoria') is None
