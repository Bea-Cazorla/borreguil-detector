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
