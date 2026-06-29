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

# Conjunto curado de features (deben existir en el punto y en ref_stats).
CHART_FEATS = [
    'ndvi_early', 'ndvi_late', 'clre_early', 'clre_late', 'ndmi_early', 'ndmi_late',
    'evi_early', 'evi_late', 'gndvi_early', 'gndvi_late', 'ndvi_drop',
    'ndwi_late', 'nbr_late',
    'ndmi_mean', 'ndmi_min', 'ndmi_max', 'ndwi_mean', 'ndwi_min', 'ndwi_max',
    'wavi_mean', 'wavi_min', 'wavi_max', 'albedo_mean', 'albedo_min', 'albedo_max',
    'cir_ndvi_mean', 'cir_ndvi_p90',
]
PCTS = [('p10', 10), ('p25', 25), ('p50', 50), ('p75', 75), ('p90', 90)]
COL_BORR = '#2e7d32'   # verde: rango de borreguiles
COL_PT = '#1565c0'     # azul: este punto (dentro del rango)
COL_OUT = '#e65100'    # naranja: el punto cae fuera del rango habitual

# Perfil estacional inicio→fin de verano (clave base, etiqueta)
SEASONAL = [('ndvi', 'NDVI (verdor)'), ('clre', 'Clorofila red-edge'),
            ('ndmi', 'NDMI (humedad)'), ('evi', 'EVI'), ('gndvi', 'GNDVI')]
# Índices "fin de verano" para la gráfica de percentil-posición
KEY = [('ndvi_late', 'NDVI fin'), ('clre_late', 'Clorofila fin'),
       ('ndmi_late', 'NDMI humedad'), ('ndwi_late', 'NDWI agua'),
       ('evi_late', 'EVI fin'), ('ndvi_drop', 'Caída NDVI'),
       ('cir_ndvi_mean', 'CIR NDVI 0,25 m')]
# Índices multianuales (mean con whisker min–max)
MULTI = [('ndmi', 'NDMI (humedad)'), ('ndwi', 'NDWI (agua)'),
         ('wavi', 'WAVI'), ('albedo', 'Albedo')]

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
    # Sentinel-2 multianual (mean/min/max/sd)
    'wavi_mean': 'WAVI media', 'wavi_min': 'WAVI mínimo', 'wavi_max': 'WAVI máximo',
    'wavi_sd': 'WAVI variabilidad', 'albedo_mean': 'Albedo medio', 'albedo_min': 'Albedo mínimo',
    'albedo_max': 'Albedo máximo', 'albedo_sd': 'Albedo variabilidad',
    'ndmi_mean': 'NDMI media (multianual)', 'ndmi_min': 'NDMI mínimo', 'ndmi_max': 'NDMI máximo',
    'ndmi_sd': 'NDMI variabilidad', 'ndwi_mean': 'NDWI media (multianual)', 'ndwi_min': 'NDWI mínimo',
    'ndwi_max': 'NDWI máximo', 'ndwi_sd': 'NDWI variabilidad',
    # Sentinel-1 SAR
    's1_vv_mean': 'SAR VV media (dB)', 's1_vh_mean': 'SAR VH media (dB)',
    's1_vv_sd': 'SAR VV variabilidad', 's1_vh_sd': 'SAR VH variabilidad',
    's1_ratio_mean': 'SAR VV–VH (dB)',
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


def fnum(v):
    """float o None (descarta NaN/strings)."""
    try:
        x = float(v)
        return x if x == x else None
    except (TypeError, ValueError):
        return None


def _fmt(v):
    """Formatea un valor con precisión adaptada a su magnitud (m, dB, índices…)."""
    a = abs(v)
    if a >= 100:
        return f'{v:.0f}'
    if a >= 10:
        return f'{v:.1f}'
    if a >= 1:
        return f'{v:.2f}'
    return f'{v:.3f}'


def pctile_in_borr(v, q):
    """Percentil aproximado de v dentro de la distribución borreguil, interpolando sus
    cuantiles p10..p90 (fuera de rango se acota a 5/95)."""
    if v is None or not q:
        return None
    xs = [q['p10'], q['p25'], q['p50'], q['p75'], q['p90']]
    ys = [10, 25, 50, 75, 90]
    for i in range(1, len(xs)):           # monotonía estricta para np.interp
        if xs[i] <= xs[i - 1]:
            xs[i] = xs[i - 1] + 1e-9
    return float(max(0.0, min(100.0, np.interp(v, xs, ys, left=5.0, right=95.0))))


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
        for f in CHART_FEATS:
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


def chart_seasonal(sel, feats, ncols=2):
    """G1 — perfil estacional inicio→fin de verano vs. banda de borreguiles. Un panel por
    índice repartidos en una rejilla de `ncols` columnas (varias filas, paneles grandes)."""
    panels = []
    for key, label in SEASONAL:
        recs = []
        for season, sname in (('early', 'inicio'), ('late', 'fin')):
            f = f'{key}_{season}'
            si = feats.get(f, {}).get('si') or {}
            v = fnum(sel.get(f))
            if not si and v is None:
                continue
            recs.append({'season': sname, 'lo': si.get('p25'), 'hi': si.get('p75'),
                         'med': si.get('p50'), 'val': v})
        if not recs:
            continue
        df = pd.DataFrame(recs)
        x = alt.X('season:N', sort=['inicio', 'fin'], title=None,
                  axis=alt.Axis(labelAngle=0, labelFontSize=11))
        band = alt.Chart(df).mark_area(opacity=0.22, color=COL_BORR).encode(
            x=x, y=alt.Y('lo:Q', title=None), y2='hi:Q')
        med = alt.Chart(df).mark_line(color=COL_BORR, strokeDash=[4, 3]).encode(x=x, y='med:Q')
        ln = alt.Chart(df).mark_line(color=COL_PT, strokeWidth=2.5).encode(x=x, y='val:Q')
        pt = alt.Chart(df).mark_point(filled=True, color=COL_PT, size=95).encode(
            x=x, y='val:Q', tooltip=[alt.Tooltip('season:N', title='periodo'),
                                     alt.Tooltip('val:Q', title='valor', format='.3f')])
        panels.append((band + med + ln + pt).properties(width=260, height=200, title=label))
    if not panels:
        return None
    return alt.concat(*panels, columns=ncols).resolve_scale(y='independent')


def chart_percentile(sel, feats):
    """G2 — percentil de cada índice (fin de verano) dentro de los borreguiles.
    Devuelve (chart|None, recs)."""
    recs = []
    for f, label in KEY:
        si = feats.get(f, {}).get('si')
        v = fnum(sel.get(f))
        if si is None or v is None:
            continue
        p = pctile_in_borr(v, si)
        recs.append({'label': label, 'pctile': p, 'val': v, 'valtxt': f'{v:.3f}',
                     'inside': bool(25 <= p <= 75)})
    if not recs:
        return None, []
    df = pd.DataFrame(recs)
    order = [r['label'] for r in recs]
    xsc = alt.Scale(domain=[0, 100])
    zone = alt.Chart(pd.DataFrame({'x0': [25], 'x1': [75]})).mark_rect(
        opacity=0.15, color=COL_BORR).encode(
        x=alt.X('x0:Q', scale=xsc, title='percentil dentro de los borreguiles (0–100)'), x2='x1:Q')
    med = alt.Chart(pd.DataFrame({'x': [50]})).mark_rule(
        color=COL_BORR, strokeDash=[4, 3]).encode(x=alt.X('x:Q', scale=xsc))
    pts = alt.Chart(df).mark_point(filled=True, size=130).encode(
        x=alt.X('pctile:Q', scale=xsc), y=alt.Y('label:N', sort=order, title=None),
        color=alt.condition('datum.inside', alt.value(COL_PT), alt.value(COL_OUT)),
        tooltip=[alt.Tooltip('label:N', title='índice'),
                 alt.Tooltip('val:Q', title='valor', format='.3f'),
                 alt.Tooltip('pctile:Q', title='percentil', format='.0f')])
    txt = alt.Chart(df).mark_text(align='left', dx=9, fontSize=10, color='#444').encode(
        x=alt.X('pctile:Q', scale=xsc), y=alt.Y('label:N', sort=order), text='valtxt:N')
    return (zone + med + pts + txt).properties(height=max(120, 30 * len(recs))), recs


def chart_multiyear(sel, feats, ncols=2):
    """G3 — variabilidad multianual: un panel por índice (NDMI/NDWI/WAVI/albedo) con la
    media (●) y el rango mín–máx (│) del punto frente a la banda de borreguiles, en
    unidades reales. Rejilla de `ncols` columnas."""
    panels = []
    for key, label in MULTI:
        si = feats.get(f'{key}_mean', {}).get('si')
        vmean = fnum(sel.get(f'{key}_mean'))
        if not si or vmean is None:
            continue
        vmin = fnum(sel.get(f'{key}_min'))
        vmax = fnum(sel.get(f'{key}_max'))
        inside = si['p25'] <= vmean <= si['p75']
        bd = pd.DataFrame({'lo': [si['p25']], 'hi': [si['p75']], 'med': [si['p50']]})
        pdf = pd.DataFrame({'x': ['punto'], 'mean': [vmean],
                            'min': [vmin if vmin is not None else vmean],
                            'max': [vmax if vmax is not None else vmean]})
        xpos = alt.X('x:N', title=None, axis=alt.Axis(labels=False, ticks=False))
        band = alt.Chart(bd).mark_rect(opacity=0.22, color=COL_BORR).encode(
            y=alt.Y('lo:Q', title=None), y2='hi:Q')
        med = alt.Chart(bd).mark_rule(color=COL_BORR, strokeDash=[4, 3]).encode(y='med:Q')
        whisk = alt.Chart(pdf).mark_rule(color=COL_PT, strokeWidth=2).encode(
            x=xpos, y='min:Q', y2='max:Q')
        dot = alt.Chart(pdf).mark_point(filled=True, size=120,
                                        color=COL_PT if inside else COL_OUT).encode(
            x=xpos, y='mean:Q', tooltip=[alt.Tooltip('mean:Q', title='media', format='.3f'),
                                         alt.Tooltip('min:Q', title='mín', format='.3f'),
                                         alt.Tooltip('max:Q', title='máx', format='.3f')])
        panels.append((band + med + whisk + dot).properties(width=150, height=200, title=label))
    if not panels:
        return None
    return alt.concat(*panels, columns=ncols).resolve_scale(y='independent')


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
                     'imp': round(w * 100, 1), 'rango': rng})
    df = pd.DataFrame(recs)
    order = [r['label'] for r in recs]
    xsc = alt.Scale(domain=[0, max(r['imp'] for r in recs) * 1.5])
    bars = alt.Chart(df).mark_bar(color=COL_PT, opacity=0.85).encode(
        x=alt.X('imp:Q', scale=xsc, title='importancia en el modelo (%)'),
        y=alt.Y('label:N', sort=order, title=None,
                axis=alt.Axis(labelLimit=240, labelOverlap=False)),
        tooltip=[alt.Tooltip('label:N', title='variable'),
                 alt.Tooltip('imp:Q', title='importancia %', format='.1f'),
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
    ('twi', 'TWI (humedad topo.)'), ('ndvi_late', 'NDVI fin verano'),
    ('ndmi_late', 'NDMI (humedad)'), ('ndwi_late', 'NDWI (agua)'),
    ('evi_late', 'EVI fin'), ('clre_late', 'Clorofila red-edge'),
    ('ndvi_drop', 'Caída NDVI'), ('dist_water_m', 'Distancia al agua (m)'),
]
_CLS_BORR = 'Borreguil'
_CLS_REST = 'No / resto'
_CLS_SCALE = alt.Scale(domain=[_CLS_BORR, _CLS_REST], range=['#2e7d32', '#8e0000'])


def is_borreguil_decision(dec):
    """True si la decisión corresponde a (posible/probable/auto/verificado) borreguil."""
    d = (dec or '').lower()
    if 'no borreguil' in d:
        return False
    return any(k in d for k in ('borreguil', 'probable', 'posible', 'auto', 'verificado'))


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
        tooltip=['ID:N', 'clase:N', alt.Tooltip(f'{xlab}:Q', format='.2f'),
                 alt.Tooltip(f'{ylab}:Q', format='.3f'),
                 alt.Tooltip('rf:Q', title='prob. RF', format='.2f')],
    ).properties(height=380)


def chart_decision_counts(rows):
    """Nº de puntos por categoría de decisión (barras horizontales). Devuelve chart|None."""
    from collections import Counter
    c = Counter(r.get('decision', '—') or '—' for r in rows)
    if not c:
        return None
    df = pd.DataFrame([{'decision': k, 'n': v} for k, v in c.items()])
    return alt.Chart(df).mark_bar().encode(
        x=alt.X('n:Q', title='Nº de puntos'),
        y=alt.Y('decision:N', sort='-x', title=None),
        color=alt.Color('decision:N', legend=None),
        tooltip=['decision:N', 'n:Q'],
    ).properties(height=max(120, 30 * len(df)))
