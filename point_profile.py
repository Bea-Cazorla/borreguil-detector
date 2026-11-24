"""
point_profile.py — Gráficas Altair del perfil espectral de un punto frente al rango
típico de los borreguiles de entrenamiento. Código puro (altair/numpy/pandas, SIN
Streamlit) para que sea reutilizable y testeable sin lanzar la app.

Las bandas de referencia salen de ref_stats['features'][feat]['si'|'no'] = cuantiles
p10/p25/p50/p75/p90 (los guarda train_v5.py en el bundle del modelo).
"""
import numpy as np
import pandas as pd
import altair as alt

import i18n

PCTS = [('p10', 10), ('p25', 25), ('p50', 50), ('p75', 75), ('p90', 90)]
COL_BORR = '#2e7d32'   # verde: rango de borreguiles
COL_PT = '#1565c0'     # azul: este punto (dentro del rango)
COL_OUT = '#e65100'    # naranja: el punto cae fuera del rango habitual
COL_NO = '#757575'     # gris: el no borreguil típico (mediana de los no borreguiles)

# Etiquetas legibles de TODAS las variables del modelo (para el ranking de importancia)
FEATURE_LABELS = {
    # topografía
    'elev_dem_m': 'Altitud (m)', 'slope_deg': 'Pendiente (°)',
    'twi': 'Humedad topográfica (TWI)', 'curvature': 'Curvatura',
    # Sentinel-2 estacional
    'ndvi_early': 'NDVI inicio verano', 'ndvi_late': 'NDVI fin verano',
    'ndvi_drop': 'Caída NDVI estacional', 'ndwi_late': 'NDWI fin (agua)',
    'clre_early': 'Clorofila red-edge inicio', 'clre_late': 'Clorofila red-edge fin',
    'gndvi_early': 'GNDVI inicio', 'gndvi_late': 'GNDVI fin',
    'ndmi_early': 'NDMI humedad inicio', 'ndmi_late': 'NDMI humedad fin',
    'evi_early': 'EVI inicio', 'evi_late': 'EVI fin', 'nbr_late': 'NBR fin',
    'ndwi_early': 'NDWI inicio (agua)', 'nbr_early': 'NBR inicio',
    'aspect_north': 'Orientación (norte)', 'aspect_east': 'Orientación (este)',
    # la pureza (como en el mapa): el píxel de 10 m del punto
    'ndvi_early_pixel': 'NDVI inicio verano (su píxel)',
    'agua_pixel_vecinos': 'Agua en su píxel o al lado', 'verde_estimado': 'Verde estimado',
    # Sentinel-2 multianual (mean/min/max/sd)
    'wavi_mean': 'WAVI media', 'wavi_min': 'WAVI mínimo', 'wavi_max': 'WAVI máximo',
    'wavi_sd': 'WAVI variabilidad', 'albedo_mean': 'Albedo medio', 'albedo_min': 'Albedo mínimo',
    'albedo_max': 'Albedo máximo', 'albedo_sd': 'Albedo variabilidad',
    'ndmi_mean': 'NDMI media (multianual)', 'ndmi_min': 'NDMI mínimo', 'ndmi_max': 'NDMI máximo',
    'ndmi_sd': 'NDMI variabilidad', 'ndwi_mean': 'NDWI media (multianual)', 'ndwi_min': 'NDWI mínimo',
    'ndwi_max': 'NDWI máximo', 'ndwi_sd': 'NDWI variabilidad',
    # Sentinel-1 (radar; «radar», como en el resto de la app, no «SAR»)
    's1_vv_mean': 'Radar VV media (dB)', 's1_vh_mean': 'Radar VH media (dB)',
    's1_vv_sd': 'Radar VV variabilidad', 's1_vh_sd': 'Radar VH variabilidad',
    's1_ratio_mean': 'Radar VV–VH (dB)',
    # textura/imagen
    'frac_bgreen': 'Fracción verde-azulada (imagen)', 'exg_mean': 'Exceso de verde (imagen)',
    'glcm_homog': 'Homogeneidad textura', 'glcm_contrast': 'Contraste textura',
    'edge_density': 'Densidad de bordes', 'laplacian_var': 'Nitidez (Laplaciano)',
    'vegcontrast': 'Contraste vegetación', 'surr_rock': 'Roca alrededor',
    'granul_5': 'Granulometría', 'largest_frac': 'Mayor parche conexo',
    'matorral_score': 'Índice matorral', 'n_components': 'Nº de parches',
    # CIR (PNOA IR 0,25 m)
    'cir_ndvi_mean': 'CIR NDVI media 0,25 m', 'cir_ndvi_p90': 'CIR NDVI p90 0,25 m',
    'cir_ndvi_sd': 'CIR NDVI variabilidad', 'cir_vegfrac': 'CIR % vegetación',
    'cir_nir_mean': 'CIR NIR media', 'cir_ndvi_contrast': 'CIR contraste 0,25 m',
}
FEATURE_LABELS_EN = {
    'elev_dem_m': 'Elevation (m)', 'slope_deg': 'Slope (°)',
    'twi': 'Topographic wetness (TWI)', 'curvature': 'Curvature',
    'ndvi_early': 'NDVI early summer', 'ndvi_late': 'NDVI late summer',
    'ndvi_drop': 'Seasonal NDVI drop', 'ndwi_late': 'NDWI late (water)',
    'clre_early': 'Red-edge chlorophyll early', 'clre_late': 'Red-edge chlorophyll late',
    'gndvi_early': 'GNDVI early', 'gndvi_late': 'GNDVI late',
    'ndmi_early': 'NDMI moisture early', 'ndmi_late': 'NDMI moisture late',
    'evi_early': 'EVI early', 'evi_late': 'EVI late', 'nbr_late': 'NBR late',
    'ndwi_early': 'NDWI early (water)', 'nbr_early': 'NBR early',
    'aspect_north': 'Aspect (north)', 'aspect_east': 'Aspect (east)',
    'ndvi_early_pixel': 'NDVI early summer (its pixel)',
    'agua_pixel_vecinos': 'Water in its pixel or next to it', 'verde_estimado': 'Estimated green',
    'wavi_mean': 'WAVI mean', 'wavi_min': 'WAVI minimum', 'wavi_max': 'WAVI maximum',
    'wavi_sd': 'WAVI variability', 'albedo_mean': 'Mean albedo', 'albedo_min': 'Minimum albedo',
    'albedo_max': 'Maximum albedo', 'albedo_sd': 'Albedo variability',
    'ndmi_mean': 'NDMI mean (multi-year)', 'ndmi_min': 'NDMI minimum', 'ndmi_max': 'NDMI maximum',
    'ndmi_sd': 'NDMI variability', 'ndwi_mean': 'NDWI mean (multi-year)',
    'ndwi_min': 'NDWI minimum', 'ndwi_max': 'NDWI maximum', 'ndwi_sd': 'NDWI variability',
    's1_vv_mean': 'Radar VV mean (dB)', 's1_vh_mean': 'Radar VH mean (dB)',
    's1_vv_sd': 'Radar VV variability', 's1_vh_sd': 'Radar VH variability',
    's1_ratio_mean': 'Radar VV–VH (dB)',
    'frac_bgreen': 'Blue-green fraction (image)', 'exg_mean': 'Excess green (image)',
    'glcm_homog': 'Texture homogeneity', 'glcm_contrast': 'Texture contrast',
    'edge_density': 'Edge density', 'laplacian_var': 'Sharpness (Laplacian)',
    'vegcontrast': 'Vegetation contrast', 'surr_rock': 'Surrounding rock',
    'granul_5': 'Granulometry', 'largest_frac': 'Largest connected patch',
    'matorral_score': 'Shrub index', 'n_components': 'Number of patches',
    'cir_ndvi_mean': 'CIR NDVI mean 0.25 m', 'cir_ndvi_p90': 'CIR NDVI p90 0.25 m',
    'cir_ndvi_sd': 'CIR NDVI variability', 'cir_vegfrac': 'CIR % vegetation',
    'cir_nir_mean': 'CIR NIR mean', 'cir_ndvi_contrast': 'CIR contrast 0.25 m',
}


def fnum(v):
    """float o None (descarta NaN/strings)."""
    try:
        x = float(v)
        return x if x == x else None
    except (TypeError, ValueError):
        return None


def _fmt(v):
    """Un valor con 3 decimales, como en el resto de la app (lo pidió la autora el 18-11):
    con coma decimal en español."""
    t = f'{v:.3f}'
    return t if i18n.get_lang() == 'en' else t.replace('.', ',')


def ref_stats_from_csv(candidate_paths):
    """Fallback: calcula cuantiles por clase desde el primer classification_*.csv que
    exista en candidate_paths (lista de pathlib.Path). Devuelve dict o None."""
    for cand in candidate_paths:
        if not cand.exists():
            continue
        try:
            df = pd.read_csv(cand)
        except Exception:
            continue
        if 'gt_borreguil' not in df.columns:
            continue
        if 'gt_duda' in df.columns:
            df = df[df['gt_duda'].astype(str).str.lower() != 'si']
        feats = {}
        for f in FEATURE_LABELS:
            if f not in df.columns:
                continue
            d = {}
            for cls in ('si', 'no'):
                s = pd.to_numeric(df.loc[df['gt_borreguil'] == cls, f], errors='coerce').dropna()
                if len(s) >= 10:
                    d[cls] = {p: float(s.quantile(q / 100)) for p, q in PCTS}
                    d[cls]['n'] = int(len(s))
            if d:
                feats[f] = d
        return {'features': feats,
                'n_si': int((df['gt_borreguil'] == 'si').sum()),
                'n_no': int((df['gt_borreguil'] == 'no').sum()),
                'source': f'{cand.name} (local)'}
    return None


# ============================================================
# Este punto frente a borreguiles y no borreguiles
# ============================================================
# Grupos de variables casi iguales. Salen de la tabla con la que se entrenó el modelo de
# Sierra Nevada (classification_v5.csv, 769 puntos; en la v6 solo cambia la escala del
# EVI, que se parece aún más al NDVI): correlación de Spearman y
# agrupamiento por la media (dentro de cada grupo, |r| medio ≥ 0,8). Ver una de cada
# grupo basta: las demás dicen lo mismo, y el peso del modelo se reparte entre ellas.
GRUPOS_PARECIDAS = [
    # verdor de verano (Sentinel-2) y el NDVI de la ortofoto infrarroja
    ('evi_early', 'ndvi_early', 'ndvi_late', 'ndwi_late', 'clre_early', 'clre_late',
     'gndvi_early', 'gndvi_late', 'evi_late', 'wavi_mean', 'wavi_max', 'ndwi_mean',
     'ndwi_min', 'cir_ndvi_mean', 'cir_vegfrac'),
    ('ndmi_early', 'ndmi_mean', 'ndmi_max'),          # humedad de la vegetación
    ('ndmi_late', 'ndmi_min'),                         # humedad al final del verano
    ('frac_bgreen', 'exg_mean'),                       # verde en la imagen de 0,5 m
    ('glcm_homog', 'glcm_contrast', 'laplacian_var'),  # textura de la imagen
    ('matorral_score', 'n_components'),
    ('wavi_min', 'ndwi_max'),
    ('ndwi_sd', 'wavi_sd'),                            # cuánto cambia entre años
    ('albedo_mean', 'albedo_max'),
]
_GRUPO_DE = {f: g for g in GRUPOS_PARECIDAS for f in g}

# Sin pesos del modelo (un modelo sin `importances`), estas: una por grupo, de las que
# más separan borreguil de no borreguil en Sierra Nevada.
CLAVE_POR_DEFECTO = ['elev_dem_m', 'ndvi_drop', 'evi_early', 'ndwi_sd', 'slope_deg',
                     's1_ratio_mean', 'surr_rock', 'ndmi_early']



def etiqueta(f):
    """Nombre legible de una variable, en el idioma de la app."""
    if i18n.get_lang() == 'en':
        return FEATURE_LABELS_EN.get(f, f)
    return FEATURE_LABELS.get(f, f)


def grupo_de(f):
    """Las variables casi iguales a `f` (ella incluida)."""
    return _GRUPO_DE.get(f, (f,))


def variables_clave(importances, disponibles=None, n=8):
    """Las `n` variables que más pesan en el modelo, UNA POR GRUPO de parecidas.

    Los grupos se ordenan por el peso SUMADO de sus variables (el Random Forest reparte
    el peso entre las que dicen lo mismo, y por separado cada una parecería poco
    importante) y de cada grupo se coge la que más pesa. `disponibles`: solo esas
    variables (las que tienen valores de referencia). Devuelve una lista de dicts
    {feat, peso (0–1, del grupo), parecidas (cuántas más hay en el grupo)}; sin pesos,
    CLAVE_POR_DEFECTO con peso None."""
    imp = {}
    for f, w in (importances or {}).items():
        try:
            w = float(w)
        except (TypeError, ValueError):
            continue
        if w == w and (disponibles is None or f in disponibles):
            imp[f] = w
    if not imp:
        return [{'feat': f, 'peso': None, 'parecidas': 0} for f in CLAVE_POR_DEFECTO
                if disponibles is None or f in disponibles][:n]
    total = sum(imp.values()) or 1.0
    grupos = {}
    for f, w in imp.items():
        grupos.setdefault(grupo_de(f), []).append((f, w))
    filas = []
    for miembros in grupos.values():
        rep = max(miembros, key=lambda m: (m[1], m[0]))[0]
        filas.append({'feat': rep, 'peso': sum(w for _, w in miembros) / total,
                      'parecidas': len(miembros) - 1})
    filas.sort(key=lambda d: (-d['peso'], d['feat']))
    return filas[:n]


def percentil_entre_borreguiles(v, si):
    """En qué percentil de los borreguiles verdaderos cae el valor `v` (0–100),
    interpolando entre sus cuantiles p10…p90. Por debajo del p10 da 5 y por encima
    del p90, 95: ahí solo se sabe que es más bajo (o más alto) que casi todos."""
    if v is None or not si:
        return None
    xs = [si['p10'], si['p25'], si['p50'], si['p75'], si['p90']]
    for i in range(1, len(xs)):                 # estrictamente creciente para interp
        if xs[i] <= xs[i - 1]:
            xs[i] = xs[i - 1] + 1e-9
    if v < xs[0]:
        return 5.0
    if v > xs[-1]:
        return 95.0
    return float(np.interp(v, xs, [10, 25, 50, 75, 90]))


def _tres(v):
    return '—' if v is None else _fmt(v)


def chart_punto_vs_borreguiles(sel, ref_stats, n=8):
    """Dónde cae el punto entre los BORREGUILES VERDADEROS (los de entrenamiento del
    modelo) en las variables que más pesan (variables_clave, una por grupo de
    parecidas, de arriba abajo por peso). Eje común de 0 a 100: la posición del punto
    entre los borreguiles (50 = como el borreguil mediano); franja verde = la mitad
    central de los borreguiles (25–75). Punto azul dentro del 10–90 y naranja fuera.
    Rombo gris: el NO borreguil típico (la mediana de los no borreguiles verdaderos),
    en la misma escala, para ver si el punto se parece más a uno o a otro (añadido el
    24-11-2026). A la derecha, el valor del punto y el rango habitual de los
    borreguiles (p25–p75), con 3 decimales. Devuelve (chart|None, recs)."""
    feats = (ref_stats or {}).get('features') or {}
    con_ref = {f for f, d in feats.items() if (d or {}).get('si')}
    modelo = (ref_stats or {}).get('model_features')
    if modelo:
        con_ref &= set(modelo)
    en = i18n.get_lang() == 'en'
    recs = []
    for k in variables_clave((ref_stats or {}).get('importances'), con_ref, n):
        f = k['feat']
        si = feats[f]['si']
        v = fnum(sel.get(f))
        pct = percentil_entre_borreguiles(v, si)
        rango = f"{_tres(si['p25'])}–{_tres(si['p75'])}"
        no = (feats[f] or {}).get('no') or {}
        no_med = fnum(no.get('p50'))
        recs.append({
            'feat': f, 'label': etiqueta(f), 'val': v, 'pct': pct, 'peso': k['peso'],
            'no_med': no_med, 'pct_no': percentil_entre_borreguiles(no_med, si),
            'fuera': pct is not None and (pct < 10 or pct > 90),
            'rango': rango,
            'texto': (f"{_tres(v)} ({'borreguiles' if not en else 'borreguiles'}: {rango})"
                      if v is not None else ('no data' if en else 'sin dato')),
            'peso_txt': '—' if k['peso'] is None else f"{100 * k['peso']:.0f} %",
        })
    if not recs:
        return None, []
    df = pd.DataFrame(recs)
    orden = [r['label'] for r in recs]
    xsc = alt.Scale(domain=[0, 100])
    titulo_x = ('position among the real borreguiles (0–100)' if en
                else 'posición entre los borreguiles verdaderos (0–100)')
    franja = alt.Chart(pd.DataFrame({'a': [25], 'b': [75]})).mark_rect(
        opacity=0.18, color=COL_BORR).encode(
        x=alt.X('a:Q', scale=xsc, title=titulo_x), x2='b:Q')
    mediana = alt.Chart(pd.DataFrame({'x': [50]})).mark_rule(
        color=COL_BORR, strokeDash=[4, 3]).encode(x=alt.X('x:Q', scale=xsc))
    con_valor = df[df['pct'].notna()]
    # Todas las variables en su fila, también las que no tienen dato en este punto.
    y = alt.Y('label:N', sort=orden, title=None, scale=alt.Scale(domain=orden),
              axis=alt.Axis(labelLimit=260, labelFontSize=12))
    puntos = alt.Chart(con_valor).mark_point(filled=True, size=150).encode(
        x=alt.X('pct:Q', scale=xsc), y=y,
        color=alt.condition('datum.fuera', alt.value(COL_OUT), alt.value(COL_PT)),
        tooltip=[alt.Tooltip('label:N', title='variable'),
                 alt.Tooltip('val:Q', title=('this point' if en else 'este punto'),
                             format='.3f'),
                 alt.Tooltip('texto:N', title=''),
                 alt.Tooltip('peso_txt:N', title=('weight in the model' if en
                                                  else 'peso en el modelo'))])
    # El no borreguil típico, debajo del punto para que no lo tape
    con_no = df[df['pct_no'].notna()]
    capas = [franja, mediana]
    if len(con_no):
        capas.append(alt.Chart(con_no).mark_point(
            shape='diamond', filled=True, size=110, color=COL_NO, opacity=0.85).encode(
            x=alt.X('pct_no:Q', scale=xsc), y=y,
            tooltip=[alt.Tooltip('label:N', title='variable'),
                     alt.Tooltip('no_med:Q', title=('typical non-borreguil' if en
                                                    else 'no borreguil típico'),
                                 format='.3f')]))
    capas.append(puntos)
    grafica = alt.layer(*capas).properties(height=34 * len(recs))
    return grafica.configure_view(stroke=None).configure_axis(grid=False), recs


# La franja de la gráfica es COL_BORR con opacidad 0,18 sobre blanco
_FRANJA_HTML = '#d6e8d7'


def leyenda_punto_vs_borreguiles():
    """Leyenda (HTML) de la gráfica chart_punto_vs_borreguiles: qué es cada color."""
    en = i18n.get_lang() == 'en'

    def punto(color):
        return (f'<span style="display:inline-block;width:12px;height:12px;'
                f'border-radius:50%;background:{color};vertical-align:middle"></span>')

    franja = (f'<span style="display:inline-block;width:22px;height:12px;'
              f'background:{_FRANJA_HTML};vertical-align:middle"></span>')
    linea = (f'<span style="display:inline-block;width:0;height:14px;'
             f'border-left:2px dashed {COL_BORR};margin:0 10px 0 9px;'
             'vertical-align:middle"></span>')
    vacio = ('<span style="display:inline-block;width:12px;height:12px;'
             'border:1px dashed #999;border-radius:50%;vertical-align:middle"></span>')
    rombo = (f'<span style="display:inline-block;width:9px;height:9px;margin:0 2px;'
             f'background:{COL_NO};transform:rotate(45deg);vertical-align:middle"></span>')
    items = [
        (punto(COL_PT), 'This point: within the usual for borreguiles' if en else
         'Este punto: dentro de lo habitual en los borreguiles'),
        (punto(COL_OUT), 'This point: lower or higher than almost all borreguiles '
         '(below 10 or above 90)' if en else
         'Este punto: más bajo o más alto que casi todos los borreguiles (por debajo de '
         '10 o por encima de 90)'),
        (franja, 'Middle half of the real borreguiles (25–75)' if en else
         'Mitad central de los borreguiles verdaderos (25–75)'),
        (linea, 'The typical borreguil (50)' if en else 'El borreguil típico (50)'),
        (rombo, 'The typical non-borreguil, on the same scale: if this point is closer to '
         'it than to the green band, it looks more like a non-borreguil' if en else
         'El no borreguil típico, en la misma escala: si este punto está más cerca de él '
         'que de la franja verde, se parece más a un no borreguil'),
        (vacio, 'Row without a dot: this point has no value for that variable' if en else
         'Fila sin punto: este punto no tiene dato en esa variable'),
    ]
    filas = ''.join(f'<div style="margin:3px 0">{s}&nbsp;&nbsp;{t}</div>' for s, t in items)
    return f'<div style="font-size:0.9em;line-height:1.3">{filas}</div>'


def tabla_punto_vs_borreguiles(recs):
    """Los números de la gráfica, con 3 decimales: el valor del punto y la mitad
    central de los borreguiles verdaderos (p25–p75) en cada variable."""
    en = i18n.get_lang() == 'en'
    return pd.DataFrame([{
        'Variable': r['label'],
        ('This point' if en else 'Este punto'): _tres(r['val']),
        ('Borreguiles (middle half)' if en else 'Borreguiles (mitad central)'):
            r['rango'],
        ('Typical non-borreguil' if en else 'No borreguil típico'): _tres(r.get('no_med')),
        ('Weight' if en else 'Peso'): r['peso_txt'],
    } for r in recs])


def chart_curva_puntos(curva, n=None, marcas=(), en=None):
    """La curva de cuántos puntos hacen falta (mapeo_rf.CURVA_PUNTOS_SN): AUC en zonas
    que el modelo no ha visto frente al número de puntos revisados, en escala
    logarítmica, con la franja de 8 de cada 10 repeticiones. `marcas`: [(n, texto)]
    que se señalan (mínimo, recomendado). Con `n`, una línea vertical con «tus
    puntos» (fuera de la escala, en su borde). Devuelve el gráfico de Altair."""
    en = i18n.get_lang() == 'en' if en is None else en
    df = pd.DataFrame([{'n': c[0], 'borr': c[1], 'auc': c[2], 'p10': c[3], 'p90': c[4]}
                       for c in curva])
    df['auc_txt'] = df['auc'].map(_fmt)
    df['franja'] = df['p10'].map(_fmt) + ' – ' + df['p90'].map(_fmt)
    xmin, xmax = 15, int(df['n'].max() * 1.15)
    x = alt.X('n:Q', scale=alt.Scale(type='log', domain=[xmin, xmax], nice=False),
              axis=alt.Axis(values=[20, 30, 50, 100, 250, 500, int(df['n'].max())],
                            format='d', grid=False),
              title=('Reviewed points the model learns from (log scale)' if en else
                     'Puntos revisados con los que aprende el modelo (escala logarítmica)'))
    # La misma escala en todas las capas (la franja también): si no, empieza en 0 y la
    # curva queda aplastada arriba.
    escala = alt.Scale(domain=[0.86, 0.98], zero=False, clamp=True)
    y = alt.Y('auc:Q', scale=escala,
              axis=alt.Axis(labelExpr="format(datum.value, '.2f')" if en else
                            "replace(format(datum.value, '.2f'), '.', ',')", tickCount=6),
              title='AUC in new areas' if en else 'AUC en zonas nuevas')
    pts = 'Points' if en else 'Puntos'
    tip = [alt.Tooltip('n:Q', title=pts),
           alt.Tooltip('borr:Q', title='Borreguiles'),
           alt.Tooltip('auc_txt:N', title='AUC'),
           alt.Tooltip('franja:N', title='8 of 10 repetitions' if en else
                       '8 de cada 10 repeticiones')]
    base = alt.Chart(df)
    capas = [base.mark_area(color=COL_PT, opacity=0.15).encode(
                 x=x, y=alt.Y('p10:Q', scale=escala), y2='p90:Q'),
             base.mark_line(color=COL_PT, strokeWidth=2).encode(x=x, y=y),
             base.mark_point(color=COL_PT, filled=True, size=40, opacity=1).encode(
                 x=x, y=y, tooltip=tip)]
    tope = df['auc'].iloc[-1]
    capas.append(alt.Chart(pd.DataFrame({'auc': [tope]})).mark_rule(
        color=COL_NO, strokeDash=[4, 3]).encode(y='auc:Q'))
    if marcas:
        dm = df[df['n'].isin([m for m, _ in marcas])].copy()
        dm['txt'] = dm['n'].map(dict(marcas)) + ': ' + dm['n'].astype(str) + (
            ' points' if en else ' puntos')
        capas.append(alt.Chart(dm).mark_point(color=COL_PT, filled=True, size=130,
                                              opacity=1, stroke='white', strokeWidth=2)
                     .encode(x=x, y=y, tooltip=tip))
        # La primera etiqueta, debajo y a la derecha; la segunda, encima y a la
        # izquierda: la curva sube hacia la derecha y así no se pisan.
        for k, (m, _) in enumerate(marcas):
            capas.append(alt.Chart(dm[dm['n'] == m]).mark_text(
                align='left' if k == 0 else 'right', dx=8 if k == 0 else -8,
                dy=18 if k == 0 else -14, fontSize=11, color='#37474f')
                .encode(x=x, y=y, text='txt:N'))
    if n:
        nv = min(max(int(n), xmin), xmax)
        dn = pd.DataFrame({'n': [nv], 'txt': [(f'Your points: {int(n)}' if en
                                               else f'Tus puntos: {int(n)}')]})
        capas.append(alt.Chart(dn).mark_rule(color=COL_OUT, strokeWidth=2).encode(x=x))
        capas.append(alt.Chart(dn).mark_text(
            align='left' if nv < 300 else 'right', dx=6 if nv < 300 else -6, dy=-6,
            fontSize=11, fontWeight='bold', color=COL_OUT)
            .encode(x=x, y=alt.value(12), text='txt:N'))
    return alt.layer(*capas).properties(height=280)


def chart_importance(rs, topn=12):
    """Ranking de las variables más importantes del modelo (barras horizontales), con el
    rango típico de borreguil (p25–p75) a la derecha de cada barra. Sirve de guía jerárquica
    para leer las demás gráficas. Devuelve (chart|None, recs)."""
    imp = (rs or {}).get('importances')
    if not imp:
        return None, []
    feats = rs.get('features', {})
    items = sorted(imp.items(), key=lambda kv: -kv[1])[:topn]
    recs = []
    for f, w in items:
        si = (feats.get(f) or {}).get('si') or {}
        rng = f"{_fmt(si['p25'])}–{_fmt(si['p75'])}" if 'p25' in si and 'p75' in si else ''
        recs.append({'feat': f, 'label': FEATURE_LABELS.get(f, f),
                     'imp': round(w * 100, 3), 'rango': rng})
    df = pd.DataFrame(recs)
    order = [r['label'] for r in recs]
    xsc = alt.Scale(domain=[0, max(r['imp'] for r in recs) * 1.5])
    bars = alt.Chart(df).mark_bar(color=COL_PT, opacity=0.85).encode(
        x=alt.X('imp:Q', scale=xsc, title='importancia en el modelo (%)'),
        y=alt.Y('label:N', sort=order, title=None,
                axis=alt.Axis(labelLimit=240, labelOverlap=False)),
        tooltip=[alt.Tooltip('label:N', title='variable'),
                 alt.Tooltip('imp:Q', title='importancia %', format='.3f'),
                 alt.Tooltip('rango:N', title='rango borreguil p25–p75')])
    txt = alt.Chart(df).mark_text(align='left', dx=4, fontSize=10, color='#2e7d32').encode(
        x=alt.X('imp:Q', scale=xsc), y=alt.Y('label:N', sort=order), text='rango:N')
    return (bars + txt).properties(height=max(150, 26 * len(recs))), recs


# ============================================================
# Distribuciones del conjunto de puntos (no de un punto suelto): boxplots por
# clase, dispersión y recuento. Operan sobre la lista `rows` completa.
# ============================================================

# Variables más interpretables para comparar borreguil vs. resto.
BOX_VARS = [
    ('elev_dem_m', 'Altitud (m)'), ('slope_deg', 'Pendiente (°)'),
    ('ndvi_late', 'NDVI fin verano'),
    ('ndmi_late', 'NDMI (humedad)'), ('ndwi_late', 'NDWI (agua)'),
    ('evi_late', 'EVI fin'), ('clre_late', 'Clorofila red-edge'),
    ('ndvi_drop', 'Caída NDVI'), ('dist_cauce_m', 'Distancia al cauce (m)'),
]
_CLS_BORR = 'Borreguil'
_CLS_REST = 'No / resto'
_CLS_SCALE = alt.Scale(domain=[_CLS_BORR, _CLS_REST], range=['#2e7d32', '#8e0000'])


def is_borreguil_decision(dec):
    """True si la decisión corresponde a (posible/probable/auto/verificado) borreguil."""
    d = (dec or '').lower()
    if 'no borreguil' in d:
        return False
    return any(k in d for k in ('borreguil', 'probable', 'posible', 'verificado'))


def _class_of(r):
    return _CLS_BORR if is_borreguil_decision(r.get('decision', '')) else _CLS_REST


def chart_boxplots(rows, vars=BOX_VARS, ncols=5):
    """Boxplots de cada variable comparando puntos BORREGUIL vs. resto. Cada panel
    tiene su propia escala Y (rangos muy distintos). Devuelve (chart|None, n_borr)."""
    recs = []
    for r in rows:
        klass = _class_of(r)
        for key, lab in vars:
            v = fnum(r.get(key))
            if v is None:
                continue
            recs.append({'clase': klass, 'variable': lab, 'valor': v})
    if not recs:
        return None, 0
    df = pd.DataFrame(recs)
    order = [lab for _, lab in vars if lab in set(df['variable'])]
    n_borr = sum(1 for r in rows if _class_of(r) == _CLS_BORR)
    box = alt.Chart(df).mark_boxplot(size=26, outliers={'size': 6}).encode(
        x=alt.X('clase:N', title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y('valor:Q', title=None, scale=alt.Scale(zero=False)),
        color=alt.Color('clase:N', scale=_CLS_SCALE,
                        legend=alt.Legend(orient='top', title=None)),
    ).properties(width=120, height=170)
    chart = box.facet(
        facet=alt.Facet('variable:N', title=None, sort=order, header=alt.Header(labelFontWeight='bold')),
        columns=ncols,
    ).resolve_scale(y='independent')
    return chart, n_borr


def chart_scatter(rows, xkey='elev_dem_m', ykey='ndvi_late',
                  xlab='Altitud (m)', ylab='NDVI fin verano'):
    """Dispersión de dos variables clave, coloreada por clase. Muestra la separación
    altitud–verdor entre borreguil y el resto. Devuelve chart|None."""
    recs = []
    for r in rows:
        x, y = fnum(r.get(xkey)), fnum(r.get(ykey))
        if x is None or y is None:
            continue
        recs.append({'clase': _class_of(r), xlab: x, ylab: y,
                     'ID': r.get('ID', ''), 'rf': fnum(r.get('rf_proba'))})
    if not recs:
        return None
    df = pd.DataFrame(recs)
    return alt.Chart(df).mark_circle(size=55, opacity=0.6).encode(
        x=alt.X(f'{xlab}:Q', scale=alt.Scale(zero=False)),
        y=alt.Y(f'{ylab}:Q', scale=alt.Scale(zero=False)),
        color=alt.Color('clase:N', scale=_CLS_SCALE,
                        legend=alt.Legend(orient='top', title=None)),
        tooltip=['ID:N', 'clase:N', alt.Tooltip(f'{xlab}:Q', format='.3f'),
                 alt.Tooltip(f'{ylab}:Q', format='.3f'),
                 alt.Tooltip('rf:Q', title='prob. RF', format='.3f')],
    ).properties(height=380)


def chart_decision_counts(rows):
    """Nº de puntos por categoría de decisión (barras horizontales). Devuelve chart|None."""
    from collections import Counter
    c = Counter(r.get('decision', '—') or '—' for r in rows)
    if not c:
        return None
    df = pd.DataFrame([{'decision': i18n.tv(k), 'n': v} for k, v in c.items()])
    return alt.Chart(df).mark_bar().encode(
        x=alt.X('n:Q', title='Nº of points' if i18n.get_lang() == 'en'
                        else 'Nº de puntos'),
        y=alt.Y('decision:N', sort='-x', title=None),
        color=alt.Color('decision:N', legend=None),
        tooltip=['decision:N', 'n:Q'],
    ).properties(height=max(120, 30 * len(df)))


_AMB_ORDER = ['arroyo', 'laguna', 'ladera']
_HUM_ORDER = ['húmedo', 'seco']
_PUR_ORDER = ['mixto-agua', 'mixto-roca', 'puro']
_PUR_SCALE = alt.Scale(domain=_PUR_ORDER, range=['#1565c0', '#8d6e63', '#2e7d32'])


def chart_hierarchy(rows):
    """Vista ANIDADA de la clasificación jerárquica sobre los borreguiles detectados:
    para cada AMBIENTE (filas) y cada HUMEDAD (columnas), barras apiladas por PUREZA.
    Devuelve (chart|None, n)."""
    sub = [r for r in rows if is_borreguil_decision(r.get('decision', ''))]
    recs = [{'ambiente': r.get('ambiente'), 'humedad': r.get('humedad'),
             'pureza': r.get('pureza')}
            for r in sub
            if r.get('ambiente') not in ('', '—', None)
            and r.get('humedad') not in ('', '—', None)
            and r.get('pureza') not in ('', '—', None)]
    if not recs:
        return None, len(sub)
    df = pd.DataFrame(recs)
    # Los valores llegan en español; se traducen solo para dibujar, junto con el
    # orden de las categorías y la escala de color (que van por valor).
    for col in ('ambiente', 'humedad', 'pureza'):
        df[col] = df[col].map(i18n.tv)
    amb = [i18n.tv(v) for v in _AMB_ORDER]
    hum = [i18n.tv(v) for v in _HUM_ORDER]
    pur = [i18n.tv(v) for v in _PUR_ORDER]
    escala = alt.Scale(domain=pur, range=['#1565c0', '#8d6e63', '#2e7d32'])
    en = i18n.get_lang() == 'en'
    chart = alt.Chart(df).mark_bar().encode(
        y=alt.Y('ambiente:N', sort=amb, title=None),
        x=alt.X('count():Q',
                title='nº of borreguil points' if en else 'nº puntos de borreguil'),
        color=alt.Color('pureza:N', scale=escala, sort=pur,
                        legend=alt.Legend(orient='bottom',
                                          title='Purity' if en else 'Pureza')),
        column=alt.Column('humedad:N', sort=hum, title=None),
        tooltip=['ambiente:N', 'humedad:N', 'pureza:N',
                 alt.Tooltip('count():Q', title='nº')],
    ).properties(width=240, height=180)
    return chart, len(sub)


def hierarchy_table(rows):
    """Tabla con el recuento de cada combinación completa ambiente·humedad·pureza
    (los 'tipos' de borreguil), ordenada por frecuencia. Devuelve un DataFrame|None."""
    from collections import Counter
    sub = [r for r in rows if is_borreguil_decision(r.get('decision', ''))]
    c = Counter(r.get('tipo_borreguil') for r in sub
                if r.get('tipo_borreguil') and '—' not in r.get('tipo_borreguil'))
    if not c:
        return None
    en = i18n.get_lang() == 'en'
    df = pd.DataFrame([{('Borreguil type' if en else 'Tipo de borreguil'): i18n.tv(k),
                        ('Nº points' if en else 'Nº puntos'): v}
                       for k, v in c.most_common()])
    return df
