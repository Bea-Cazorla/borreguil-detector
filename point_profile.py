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
COL_NO = '#8E0000'     # rojo oscuro: el no borreguil típico (mediana de los no
#                        borreguiles), el mismo «no borreguil» de toda la app (25-11-2026)
COL_GRIS = '#757575'   # gris neutro: líneas de referencia

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
    Rombo rojo oscuro: el NO borreguil típico (la mediana de los no borreguiles verdaderos),
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
        color=COL_GRIS, strokeDash=[4, 3]).encode(y='auc:Q'))
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


def chart_curva_propia(cifras, en=None):
    """La curva con los puntos de la zona (mapeo_rf.cifras_curva): el acierto en zonas
    que el modelo no ha visto frente a los puntos con los que aprende, con la franja
    entre el peor y el mejor sorteo y, bajo cada punto, qué parte de los puntos es.
    Devuelve el gráfico de Altair."""
    en = i18n.get_lang() == 'en' if en is None else en
    m = cifras['medida']
    df = pd.DataFrame([{'n': f['n_puntos'], 'borr': f['n_borreguiles'], 'v': f[m],
                        'lo': f[m + '_min'], 'hi': f[m + '_max'],
                        'parte': f"{round(f['fraccion'] * 100)} %"}
                       for f in cifras['filas']])
    df['n_txt'] = df['n'].round().astype(int).astype(str)
    df['borr_txt'] = df['borr'].round().astype(int).astype(str)
    df['v_txt'] = df['v'].map(_fmt)
    df['franja'] = df['lo'].map(_fmt) + ' – ' + df['hi'].map(_fmt)
    titulo_y = (('AUC in new areas' if en else 'AUC en zonas nuevas') if m == 'auc' else
                ('Mean F1 in new areas' if en else 'F1 medio en zonas nuevas'))
    x = alt.X('n:Q', scale=alt.Scale(domain=[0, float(df['n'].max()) * 1.08], nice=False),
              axis=alt.Axis(format='d', grid=False),
              title=('Points the model learns from (mean of the folds)' if en else
                     'Puntos con los que aprende el modelo (media de las partes)'))
    abajo = max(0.0, float(df['lo'].min()) - 0.01)
    arriba = min(1.0, float(df['hi'].max()) + 0.01)
    escala = alt.Scale(domain=[abajo, arriba], zero=False, clamp=True)
    y = alt.Y('v:Q', scale=escala, title=titulo_y,
              axis=alt.Axis(labelExpr="format(datum.value, '.3f')" if en else
                            "replace(format(datum.value, '.3f'), '.', ',')", tickCount=5))
    tip = [alt.Tooltip('parte:N', title='Share of your points' if en else
                       'Parte de tus puntos'),
           alt.Tooltip('n_txt:N', title='Points' if en else 'Puntos'),
           alt.Tooltip('borr_txt:N', title='Borreguiles'),
           alt.Tooltip('v_txt:N', title=titulo_y),
           alt.Tooltip('franja:N', title='Worst – best draw' if en else
                       'Peor – mejor sorteo')]
    base = alt.Chart(df)
    capas = [base.mark_area(color=COL_PT, opacity=0.15).encode(
                 x=x, y=alt.Y('lo:Q', scale=escala), y2='hi:Q'),
             base.mark_line(color=COL_PT, strokeWidth=2).encode(x=x, y=y),
             base.mark_point(color=COL_PT, filled=True, size=70, opacity=1,
                             stroke='white', strokeWidth=2).encode(x=x, y=y, tooltip=tip),
             base.mark_text(dy=16, fontSize=11, color='#37474f').encode(
                 x=x, y=y, text='parte:N')]
    return alt.layer(*capas).properties(height=260)


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


def vars_boxplots(importances, disponibles=None, n=8):
    """Las variables de las cajas de «Tipos y distribuciones»: las `n` que más pesan en
    el modelo, una por grupo de parecidas, de más a menos (variables_clave), con su
    peso en el título: [(variable, 'Altitud (m) · 12 %'), …]. Sin pesos, BOX_VARS
    (25-11-2026, la autora: «aquí deberían mostrarse las que más peso tienen en el
    modelo y que lo indique»)."""
    if not importances:
        return list(BOX_VARS)
    return [(k['feat'], f"{etiqueta(k['feat'])} · {round(100 * k['peso'])} %"
             if k['peso'] is not None else etiqueta(k['feat']))
            for k in variables_clave(importances, disponibles, n)]


# ============================================================
# Comparar grupos de puntos («Tipos y distribuciones», 25-11-2026, la autora: «sí, hazlo
# con comparar por tipo»): borreguil frente al resto, o los borreguiles por su humedad
# o por su ambiente frente al resto. Los colores son los del catálogo de toda la app
# (mapeo_raster.COLORES), también el rojo oscuro de «no borreguil».
# ============================================================
COMPARAR = ('borreguil', 'humedad', 'ambiente')
_ORDEN_TIPO = {'humedad': ('húmedo', 'seco'), 'ambiente': ('arroyo', 'laguna', 'ladera')}
# Los colores de cada categoría, del catálogo de toda la app (mapeo_raster.COLORES):
# laguna azul, arroyo verde, ladera amarillo verdoso… siempre los mismos (25-11-2026,
# la autora: «que siempre sean los mismos cuando nombramos cada categoría en la app»).
from mapeo_raster import COLORES as _COLORES_APP
_COLOR_TIPO = {k: _COLORES_APP[k] for k in
               ('húmedo', 'seco', 'arroyo', 'laguna', 'ladera',
                'puro', 'mixto-agua', 'mixto-roca')}
# «No borreguil», el rojo oscuro de toda la app (antes, gris al comparar tipos)
_COLOR_RESTO = _COLORES_APP['no borreguil']
# Por debajo, un grupo no se dibuja en las curvas ni en las diferencias: la mediana o
# la forma de 1 o 2 puntos engañan (en las cajas sí sale, con su nº de puntos).
MIN_GRUPO = 5
_COLOR_SIN_TIPO = '#d0cfc8'


def es_no_borreguil(r):
    """True si la app dice que el punto NO es borreguil (verificado o por el modelo).
    Los inciertos, los dudosos y los que no tienen predicción no son ni una cosa ni
    otra."""
    return 'no borreguil' in str(r.get('decision') or '').lower()


def grupos_puntos(rows, comparar='borreguil', en=None, traduce=None):
    """El grupo de cada punto para comparar, y los grupos en orden con su color.

    comparar: 'borreguil' (borreguil / no borreguil), 'humedad' (húmedo, seco y no
    borreguil) o 'ambiente' (arroyo, laguna, ladera y no borreguil). Un borreguil sin
    ese nivel del tipo va a «sin tipo». Los que no son ni borreguil ni no borreguil
    (inciertos, dudosos, sin predicción) se quedan fuera: su grupo es None (25-11-2026,
    la autora: antes se juntaban con los no borreguiles en «Resto»; ahora el grupo se
    llama «No borreguil» y tiene solo eso). Devuelve (grupo de cada fila, [(grupo,
    color)], resto): solo los grupos que tienen algún punto, en su orden; `resto`, el
    nombre del grupo con el que se comparan los demás (el de los no borreguiles)."""
    en = i18n.get_lang() == 'en' if en is None else en
    tv = traduce or (lambda x: x)
    if comparar == 'categoria':
        return grupos_categoria(rows, en, traduce)
    if comparar not in COMPARAR:
        raise ValueError(f'comparar: uno de {COMPARAR}')
    resto = 'Not a borreguil' if en else 'No borreguil'
    if comparar == 'borreguil':
        grupos = [_CLS_BORR if _class_of(r) == _CLS_BORR
                  else resto if es_no_borreguil(r) else None for r in rows]
        orden = [(_CLS_BORR, '#2e7d32'), (resto, _COLOR_RESTO)]
    else:
        sin = 'no type' if en else 'sin tipo'
        nombres = {v: str(tv(v)) for v in _ORDEN_TIPO[comparar]}
        grupos = []
        for r in rows:
            if _class_of(r) != _CLS_BORR:
                grupos.append(resto if es_no_borreguil(r) else None)
                continue
            v = str(r.get(comparar) or '').strip()
            grupos.append(nombres.get(v, sin))
        orden = ([(nombres[v], _COLOR_TIPO[v]) for v in _ORDEN_TIPO[comparar]]
                 + [(sin, _COLOR_SIN_TIPO), (resto, _COLOR_RESTO)])
    hay = set(grupos)
    return grupos, [(g, c) for g, c in orden if g in hay], resto


def grupos_categoria(rows, en=None, traduce=None):
    """Como grupos_puntos, pero con la CATEGORÍA de cada muestra de la tabla de
    entrenamiento del mapa (`categoria`: «húmedo», «ladera · seco», «no borreguil»…):
    las clases que aprende el modelo de ese nivel. Para «5 · Peso» de Mapa Modelo RF
    (25-11-2026, la autora: «además de peso, podría mostrar gráficos de las variables
    por tipo de borreguil como en los puntos de verdad terreno»).

    Cada categoría lleva su color del mapa (mapeo_raster.COLORES), en el orden del
    catálogo, y «no borreguil» va la última: es el grupo con el que se comparan los
    demás (`resto`; None si el nivel no lo tiene)."""
    en = i18n.get_lang() == 'en' if en is None else en
    tv = traduce or (lambda x: x)

    def nombre(c):                       # «ladera · seco» → «slope · dry»
        return ' · '.join(str(tv(p)) for p in str(c).split(' · '))

    from mapeo_raster import color as _color
    cats = [str(r.get('categoria') or '').strip() or None for r in rows]
    hay = {c for c in cats if c}
    catalogo = list(_COLORES_APP)
    orden_cats = (sorted((c for c in hay if c != 'no borreguil'),
                         key=lambda c: (catalogo.index(c) if c in catalogo else 99, c))
                  + (['no borreguil'] if 'no borreguil' in hay else []))
    grupos = [nombre(c) if c else None for c in cats]
    orden = [(nombre(c), _color(c, i)) for i, c in enumerate(orden_cats)]
    resto = nombre('no borreguil') if 'no borreguil' in hay else None
    return grupos, orden, resto


def _escala(orden):
    return alt.Scale(domain=[g for g, _ in orden], range=[c for _, c in orden])


def _f3(v, en):
    t = f'{float(v):.3f}'
    return t if en else t.replace('.', ',')


def _eje_num(en):
    """Números del eje sin separador de miles y, en español, con coma decimal."""
    return alt.Axis(labelExpr="format(datum.value, '~r')" if en else
                    "replace(format(datum.value, '~r'), '.', ',')")


def _valores(rows, grupos, vars):
    """{(rótulo de la variable, grupo): [valores]}."""
    out = {}
    for r, g in zip(rows, grupos):
        if g is None:                    # ni borreguil ni no borreguil: fuera
            continue
        for key, lab in vars:
            v = fnum(r.get(key))
            if v is not None:
                out.setdefault((lab, g), []).append(v)
    return out


def chart_boxplots(rows, vars=BOX_VARS, ncols=4, en=None, comparar='borreguil',
                   traduce=None):
    """Boxplots de cada variable comparando los grupos de `comparar` (ver
    grupos_puntos). Cada panel tiene su propia escala Y (rangos muy distintos).
    Devuelve (chart|None, n_borr).

    Debajo de las cajas no va el nombre de cada grupo (no cabían y se pisaban: lo dice
    la leyenda, arriba) y los números del eje van sin separador de miles y, en
    español, con coma decimal («3200», «0,4»; antes «3,200» se leía como 3,2).

    Las cajas se dibujan a mano (caja, bigotes, mediana y atípicos) con sus cifras
    calculadas aquí, para que al pasar el ratón salgan en palabras y con 3 decimales
    (25-11-2026, la autora: con mark_boxplot salía «min of null», «max of null» y
    muchos decimales). Como siempre: la caja va del p25 al p75, la raya blanca es la
    mediana, los bigotes llegan al valor más extremo a menos de 1,5 veces el ancho de
    la caja, y lo que queda fuera son los atípicos."""
    en = i18n.get_lang() == 'en' if en is None else en
    grupos, orden, _ = grupos_puntos(rows, comparar, en, traduce)
    valores = _valores(rows, grupos, vars)
    if not valores:
        return None, 0
    recs = []
    for (lab, g), xs in valores.items():
        a = np.asarray(xs, dtype=float)
        q1, med, q3 = (float(q) for q in np.percentile(a, [25, 50, 75]))
        dentro = a[(a >= q1 - 1.5 * (q3 - q1)) & (a <= q3 + 1.5 * (q3 - q1))]
        lo, hi = float(dentro.min()), float(dentro.max())
        recs.append({'parte': 'caja', 'variable': lab, 'clase': g, 'q1': q1,
                     'med': med, 'q3': q3, 'lo': lo, 'hi': hi, 'n': str(len(a)),
                     'med_txt': _f3(med, en), 'cajas_txt': f'{_f3(q1, en)} – {_f3(q3, en)}',
                     'bigotes_txt': f'{_f3(lo, en)} – {_f3(hi, en)}'})
        recs += [{'parte': 'atipico', 'variable': lab, 'clase': g, 'valor': float(v),
                  'valor_txt': _f3(v, en)} for v in a[(a < lo) | (a > hi)]]
    df = pd.DataFrame(recs)
    order = [lab for _, lab in vars if lab in set(df['variable'])]
    n_borr = sum(1 for r in rows if _class_of(r) == _CLS_BORR)
    ancho = 26 if len(orden) <= 2 else 18
    x = alt.X('clase:N', title=None, axis=None, sort=[g for g, _ in orden])
    escala = alt.Scale(zero=False)
    color = alt.Color('clase:N', scale=_escala(orden),
                      legend=alt.Legend(orient='top', title=None))
    t_cls = alt.Tooltip('clase:N', title='Group' if en else 'Grupo')
    tip = [t_cls,
           alt.Tooltip('med_txt:N', title='Median' if en else 'Mediana'),
           alt.Tooltip('cajas_txt:N', title='Box (p25–p75)' if en else 'Caja (p25–p75)'),
           alt.Tooltip('bigotes_txt:N', title='Whiskers' if en else 'Bigotes'),
           alt.Tooltip('n:N', title='Points' if en else 'Puntos')]
    base = alt.Chart(df)
    caja = base.transform_filter(alt.datum.parte == 'caja')
    capas = alt.layer(
        caja.mark_rule(color='#757575', strokeWidth=1.5).encode(
            x=x, y=alt.Y('lo:Q', title=None, scale=escala, axis=_eje_num(en)), y2='hi:Q',
            tooltip=tip),
        caja.mark_bar(size=ancho).encode(
            x=x, y=alt.Y('q1:Q', title=None), y2='q3:Q', color=color, tooltip=tip),
        caja.mark_tick(color='white', size=ancho, thickness=2).encode(
            x=x, y=alt.Y('med:Q', title=None), tooltip=tip),
        base.transform_filter(alt.datum.parte == 'atipico').mark_circle(size=22).encode(
            x=x, y=alt.Y('valor:Q', title=None), color=color,
            tooltip=[t_cls, alt.Tooltip('valor_txt:N', title='Value' if en else 'Valor')]),
    ).properties(width=120 if len(orden) <= 2 else 150, height=170)
    chart = capas.facet(
        facet=alt.Facet('variable:N', title=None, sort=order,
                        header=alt.Header(labelFontWeight='bold')),
        columns=ncols,
    ).resolve_scale(y='independent')
    return chart, n_borr


def _kde(xs, malla):
    """Densidad (núcleo gaussiano, ancho de banda de Scott) de `xs` en `malla`; con
    menos de MIN_GRUPO valores o sin variación, None."""
    a = np.asarray(xs, dtype=float)
    if len(a) < MIN_GRUPO or np.std(a) == 0:
        return None
    h = 1.06 * np.std(a, ddof=1) * len(a) ** (-1 / 5)
    z = (malla[:, None] - a[None, :]) / h
    return np.exp(-0.5 * z ** 2).sum(axis=1) / (len(a) * h * np.sqrt(2 * np.pi))


def chart_densidades(rows, vars=BOX_VARS, ncols=4, en=None, comparar='borreguil',
                     traduce=None):
    """Curvas de distribución de cada variable, una por grupo, superpuestas: si apenas
    se tocan, la variable separa bien los grupos. Cada curva encierra un área de 1
    (compara formas, no cuántos puntos hay) y se calcula aquí, en una malla común a
    todos los grupos de la variable. Devuelve chart|None."""
    en = i18n.get_lang() == 'en' if en is None else en
    grupos, orden, _ = grupos_puntos(rows, comparar, en, traduce)
    valores = _valores(rows, grupos, vars)
    recs = []
    for _, lab in vars:
        todos = [v for (l2, _g), xs in valores.items() if l2 == lab for v in xs]
        if len(todos) < 3:
            continue
        lo, hi = min(todos), max(todos)
        if hi <= lo:
            continue
        malla = np.linspace(lo, hi, 80)
        for g, _c in orden:
            d = _kde(valores.get((lab, g), []), malla)
            if d is None:
                continue
            recs += [{'variable': lab, 'grupo': g, 'x': float(x), 'densidad': float(y),
                      'n': str(len(valores[(lab, g)]))} for x, y in zip(malla, d)]
    if not recs:
        return None
    df = pd.DataFrame(recs)
    order = [lab for _, lab in vars if lab in set(df['variable'])]
    orden = [(g, c) for g, c in orden if g in set(df['grupo'])]   # solo los dibujados
    color = alt.Color('grupo:N', scale=_escala(orden),
                      legend=alt.Legend(orient='top', title=None))
    tip = [alt.Tooltip('grupo:N', title='Group' if en else 'Grupo'),
           alt.Tooltip('n:N', title='Points' if en else 'Puntos')]
    base = alt.Chart(df).encode(
        x=alt.X('x:Q', title=None, axis=_eje_num(en)),
        y=alt.Y('densidad:Q', title=None, axis=None),
        color=color, detail='grupo:N', tooltip=tip)
    capas = alt.layer(base.mark_area(opacity=0.15), base.mark_line(strokeWidth=2)
                      ).properties(width=150, height=110)
    return capas.facet(
        facet=alt.Facet('variable:N', title=None, sort=order,
                        header=alt.Header(labelFontWeight='bold')),
        columns=ncols,
    ).resolve_scale(x='independent', y='independent')


def tabla_diferencias(rows, vars=BOX_VARS, en=None, comparar='borreguil', traduce=None):
    """Cuánto se aparta cada grupo del resto en cada variable: (mediana del grupo −
    mediana del resto) / desviación típica de todos los puntos. 1 = una desviación
    típica más que el resto; −1, una menos. Devuelve una lista de dicts (variable,
    grupo, med_grupo, med_resto, d), en el orden de `vars`; sin resto, []."""
    en = i18n.get_lang() == 'en' if en is None else en
    grupos, orden, resto = grupos_puntos(rows, comparar, en, traduce)
    valores = _valores(rows, grupos, vars)
    filas = []
    for _, lab in vars:
        xr = valores.get((lab, resto))
        if xr is not None and len(xr) < MIN_GRUPO:
            xr = None
        todos = [v for (l2, _g), xs in valores.items() if l2 == lab for v in xs]
        if not xr or len(todos) < 3:
            continue
        sd = float(np.std(todos, ddof=1))
        if sd == 0:
            continue
        mr = float(np.median(xr))
        for g, _c in orden:
            if g == resto or len(valores.get((lab, g)) or []) < MIN_GRUPO:
                continue
            mg = float(np.median(valores[(lab, g)]))
            filas.append({'variable': lab, 'grupo': g, 'med_grupo': mg, 'med_resto': mr,
                          'd': (mg - mr) / sd, 'n': len(valores[(lab, g)])})
    return filas


def chart_diferencias(rows, vars=BOX_VARS, en=None, comparar='borreguil', traduce=None):
    """Diferencia estandarizada de medianas: una fila por variable y un punto por grupo, a la
    derecha si el grupo tiene más que el resto y a la izquierda si menos (en
    desviaciones típicas; tabla_diferencias). Devuelve chart|None."""
    en = i18n.get_lang() == 'en' if en is None else en
    filas = tabla_diferencias(rows, vars, en, comparar, traduce)
    if not filas:
        return None
    _, orden, resto = grupos_puntos(rows, comparar, en, traduce)
    df = pd.DataFrame(filas)
    orden = [(g, c) for g, c in orden if g != resto and g in set(df['grupo'])]
    df['cero'] = 0.0
    df['med_grupo_txt'] = df['med_grupo'].map(lambda v: _f3(v, en))
    df['med_resto_txt'] = df['med_resto'].map(lambda v: _f3(v, en))
    df['d_txt'] = df['d'].map(lambda v: ('+' if v > 0 else '') + _f3(v, en))
    order = [lab for _, lab in vars if lab in set(df['variable'])]
    y = alt.Y('variable:N', sort=order, title=None, axis=alt.Axis(labelLimit=260))
    lim = max(1.0, float(df['d'].abs().max()) * 1.1)
    # Con el nombre del grupo con el que se compara («no borreguil»), no «el resto»
    x = alt.X('d:Q', scale=alt.Scale(domain=[-lim, lim]), axis=_eje_num(en),
              title=(f'← less · more than «{resto}» → (standard deviations)' if en else
                     f'← menos · más que «{resto}» → (desviaciones típicas)'))
    color = alt.Color('grupo:N', scale=_escala(orden),
                      legend=alt.Legend(orient='top', title=None))
    tip = [alt.Tooltip('grupo:N', title='Group' if en else 'Grupo'),
           alt.Tooltip('variable:N', title='Variable'),
           alt.Tooltip('med_grupo_txt:N', title='Median of the group' if en
                       else 'Mediana del grupo'),
           alt.Tooltip('med_resto_txt:N', title=f'Median of «{resto}»' if en
                       else f'Mediana de «{resto}»'),
           alt.Tooltip('d_txt:N', title='Difference' if en else 'Diferencia')]
    off = alt.YOffset('grupo:N', sort=[g for g, _ in orden]) if len(orden) > 1 else None
    enc_l = dict(y=y, x=alt.X('cero:Q'), x2='d:Q', color=color, tooltip=tip)
    enc_p = dict(y=y, x=x, color=color, tooltip=tip)
    if off is not None:
        enc_l['yOffset'] = off
        enc_p['yOffset'] = off
    base = alt.Chart(df)
    cero = alt.Chart(pd.DataFrame({'d': [0.0]})).mark_rule(color='#9e9e9e').encode(
        x=alt.X('d:Q'))
    return alt.layer(
        cero,
        base.mark_rule(strokeWidth=2).encode(**enc_l),
        base.mark_circle(size=90, opacity=1).encode(**enc_p),
    # En la app, la altura incluye la leyenda de arriba y el eje de abajo (unos 90 px):
    # con 8 variables las filas se aplastaban y solo se leían dos nombres (25-11-2026).
    ).properties(height=90 + len(order) * (22 + 10 * max(1, len(orden))))


def lista_variables(rows, importances=None, en=None, todas=None):
    """Las variables para elegir en «Dos variables a la vez»: primero las del modelo,
    de más a menos peso («Altitud (m) · 12 %»), y después las demás que tienen valor
    en los puntos («… (no la usa el modelo)»). `todas`: los nombres posibles (de
    entrada, FEATURE_LABELS). Devuelve [(variable, rótulo)]."""
    en = i18n.get_lang() == 'en' if en is None else en
    imp = {}
    for f, w in (importances or {}).items():
        try:
            w = float(w)
        except (TypeError, ValueError):
            continue
        if w == w:
            imp[f] = w
    total = sum(imp.values()) or 1.0

    def con_valor(f):
        return any(fnum(r.get(f)) is not None for r in rows)

    del_modelo = [f for f in sorted(imp, key=lambda f: (-imp[f], f)) if con_valor(f)]
    otras = [f for f in (todas if todas is not None else FEATURE_LABELS)
             if f not in imp and con_valor(f)]
    fuera = ' (not used by the model)' if en else ' (no la usa el modelo)'
    return ([(f, f'{etiqueta(f)} · {round(100 * imp[f] / total)} %') for f in del_modelo]
            + [(f, etiqueta(f) + fuera) for f in otras])


def chart_dos_variables(rows, xkey, ykey, xlab, ylab, en=None, comparar='borreguil',
                        traduce=None, nombres=None):
    """Dos variables a la vez: cada punto en el plano, del color de su grupo. Al pasar
    el ratón: el punto (`nombres`: su nombre corto), el grupo, los dos valores y la
    probabilidad del modelo, con 3 decimales. Devuelve chart|None."""
    en = i18n.get_lang() == 'en' if en is None else en
    grupos, orden, _ = grupos_puntos(rows, comparar, en, traduce)
    recs = []
    for i, (r, g) in enumerate(zip(rows, grupos)):
        xv, yv = fnum(r.get(xkey)), fnum(r.get(ykey))
        if xv is None or yv is None or g is None:
            continue
        rf = fnum(r.get('rf_proba'))
        recs.append({'x': xv, 'y': yv, 'grupo': g,
                     'punto': str(nombres[i] if nombres else r.get('ID', '')),
                     'x_txt': _f3(xv, en), 'y_txt': _f3(yv, en),
                     'rf_txt': '—' if rf is None else _f3(rf, en)})
    if not recs:
        return None
    df = pd.DataFrame(recs)
    return alt.Chart(df).mark_circle(size=55, opacity=0.7).encode(
        x=alt.X('x:Q', title=xlab, scale=alt.Scale(zero=False), axis=_eje_num(en)),
        y=alt.Y('y:Q', title=ylab, scale=alt.Scale(zero=False), axis=_eje_num(en)),
        color=alt.Color('grupo:N', scale=_escala(orden),
                        legend=alt.Legend(orient='top', title=None)),
        tooltip=[alt.Tooltip('punto:N', title='Point' if en else 'Punto'),
                 alt.Tooltip('grupo:N', title='Group' if en else 'Grupo'),
                 alt.Tooltip('x_txt:N', title=xlab), alt.Tooltip('y_txt:N', title=ylab),
                 alt.Tooltip('rf_txt:N', title='Model probability' if en
                             else 'Probabilidad del modelo')],
    ).properties(height=380)


def chart_scatter(rows, xkey='elev_dem_m', ykey='ndvi_late',
                  xlab='Altitud (m)', ylab='NDVI fin verano'):
    """(De antes; la app usa chart_dos_variables.) Dispersión de dos variables,
    borreguil frente al resto. Devuelve chart|None."""
    return chart_dos_variables(rows, xkey, ykey, xlab, ylab)


def chart_decision_counts(rows):
    """Nº de puntos por categoría de decisión (barras horizontales). Devuelve chart|None."""
    from collections import Counter
    c = Counter(r.get('decision', '—') or '—' for r in rows)
    if not c:
        return None
    # 25-11-2026, la autora: salía en mayúsculas y cortado («NO BORREGUIL V…»), con
    # colores al azar y ordenado por tamaño. Ahora: en texto normal y entero; de más a
    # menos borreguil; cada barra con el color de su decisión en el mapa de los puntos,
    # y al final, el número y el porcentaje.
    from borreguil_pipeline import decision_color
    total = sum(c.values())
    en = i18n.get_lang() == 'en'

    def _orden(dec):
        """De más a menos borreguil (lo de «no borreguil» se mira antes: también
        contiene «borreguil verificado»)."""
        d = str(dec).lower()
        for clave, rango in (('no borreguil verificado', 6), ('no borreguil', 5),
                             ('borreguil verificado', 0), ('probable', 1),
                             ('posible', 2), ('incierto', 3), ('dudoso', 4)):
            if clave in d:
                return rango
        return 99

    recs = sorted(({'decision': i18n.tv_decision(k), 'n': v, 'color': decision_color(k),
                    'texto': f'{v} (' + ('<1' if 100 * v / total < 0.5 else
                                         f'{100 * v / total:.0f}') + ' %)',
                    'orden': _orden(k)}
                   for k, v in c.items()), key=lambda x: (x['orden'], x['decision']))
    df = pd.DataFrame(recs)
    tope = float(df['n'].max()) * 1.25          # sitio para «186 (48 %)»
    base = alt.Chart(df).encode(
        x=alt.X('n:Q', scale=alt.Scale(domain=[0, tope]),
                title='Nº of points' if en else 'Nº de puntos'),
        y=alt.Y('decision:N', sort=list(df['decision']), title=None,
                axis=alt.Axis(labelLimit=0, labelFontSize=12)),
        tooltip=[alt.Tooltip('decision:N', title='Result' if en else 'Resultado'),
                 alt.Tooltip('texto:N', title='nº')])
    barras = base.mark_bar(cornerRadiusEnd=4, size=20).encode(
        color=alt.Color('color:N', scale=None, legend=None))
    numeros = base.mark_text(align='left', dx=5, color='#424242',
                             fontSize=12).encode(text='texto:N')
    return (barras + numeros).properties(height=60 + 32 * len(df))


_AMB_ORDER = ['arroyo', 'laguna', 'ladera']
_HUM_ORDER = ['húmedo', 'seco']
_PUR_ORDER = ['mixto-agua', 'mixto-roca', 'puro']
# (los colores de la pureza, los del mapa: mixto-agua azul, mixto-roca marrón, puro
# verde azulado; antes aquí eran otros)
_PUR_SCALE = alt.Scale(domain=_PUR_ORDER, range=[_COLOR_TIPO[v] for v in _PUR_ORDER])


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
    escala = alt.Scale(domain=pur, range=[_COLOR_TIPO[v] for v in _PUR_ORDER])
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


_NIVELES_TIPO = (('ambiente', _AMB_ORDER, 'Ambiente', 'Environment'),
                 ('humedad', _HUM_ORDER, 'Humedad', 'Moisture'),
                 ('pureza', _PUR_ORDER, 'Pureza', 'Purity'))


def type_counts(rows):
    """Cuántos borreguiles hay de cada ambiente, de cada humedad y de cada pureza.
    Lista de (clave, [(valor, n), …]) en el orden de los niveles; el valor None reúne
    los que no tienen propuesta en ese nivel (—), y va al final. Valores en español."""
    from collections import Counter
    sub = [r for r in rows if is_borreguil_decision(r.get('decision', ''))]
    out = []
    for clave, orden, _, _ in _NIVELES_TIPO:
        c = Counter(r.get(clave) if r.get(clave) not in ('', '—', None) else None
                    for r in sub)
        cats = ([v for v in orden if c.get(v)]
                + sorted(k for k in c if k is not None and k not in orden))
        out.append((clave, [(k, c[k]) for k in cats]
                    + ([(None, c[None])] if c.get(None) else [])))
    return out


def chart_type_counts(rows):
    """Nº de borreguiles de cada ambiente, humedad y pureza: una gráfica de barras
    horizontales por nivel, con el número al final de cada barra, como el recuento por
    categoría de decisión (25-11-2026, la autora: «esta me gustaría que estuviera para
    los tipos de borreguil»). Los que no tienen propuesta en un nivel salen en gris como
    «sin propuesta (—)». Las tres van en la misma escala (de 0 al total de borreguiles),
    así que se pueden comparar. Devuelve ([(título, chart), …], n) o ([], 0)."""
    sub = [r for r in rows if is_borreguil_decision(r.get('decision', ''))]
    if not sub:
        return [], 0
    en = i18n.get_lang() == 'en'
    sin = 'no proposal (—)' if en else 'sin propuesta (—)'
    titulos = {k: (t_en if en else t_es) for k, _, t_es, t_en in _NIVELES_TIPO}
    tope = len(sub) * 1.15            # sitio para el número al final de la barra
    out = []
    for clave, cuentas in type_counts(rows):
        # Cada barra, del color de su categoría en toda la app (laguna azul, arroyo
        # verde, ladera amarillo verdoso…; 25-11-2026, la autora); sin propuesta, gris.
        df = pd.DataFrame([{'tipo': sin if k is None else i18n.tv(k), 'n': n,
                            'sin': k is None,
                            'color': _COLOR_TIPO.get(k, '#bdbdbd')} for k, n in cuentas])
        base = alt.Chart(df).encode(
            y=alt.Y('tipo:N', sort=list(df['tipo']), title=None),
            x=alt.X('n:Q', scale=alt.Scale(domain=[0, tope]),
                    title='nº of borreguil points' if en else 'nº puntos de borreguil'),
            tooltip=[alt.Tooltip('tipo:N', title=titulos[clave]),
                     alt.Tooltip('n:Q', title='nº')])
        # Barras de 18 px de grueso, haya una categoría o tres (con una sola, el ancho
        # automático la dejaba en una raya).
        barras = base.mark_bar(cornerRadiusEnd=4, size=18).encode(
            color=alt.Color('color:N', scale=None, legend=None))
        numeros = base.mark_text(align='left', dx=4, color='#424242').encode(text='n:Q')
        # En la app, la altura incluye el eje de abajo y su título (unos 60 px): sin
        # sumarlos, las barras se quedaban sin sitio y los nombres se montaban.
        out.append((titulos[clave],
                    (barras + numeros).properties(height=60 + 30 * len(df))))
    return out, len(sub)


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
