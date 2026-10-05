"""Prueba de PANTALLA: revisión del tipo y etiquetas de campo, en la app real.

Abre la app directamente en la pantalla de resultados con 40 puntos reales ya
clasificados (más uno sin variables) y hace lo que haría una persona: seleccionar
filas, cambiar el tipo, guardar, deshacer, verificar y reentrenar, lanzar una
iteración de auto-entrenamiento y quitar puntos. Después recorre la pestaña «Mapa
Modelo RF» (muestras, tabla de entrenamiento y modelo) y, por último, abre la
app sin puntos. Se repite en español y en inglés.

    python tests/pantalla_revision.py        (unos 3 minutos, sin red)

Necesita `classification_v5.csv` (los puntos de entrenamiento, que NO están en el
repositorio) en la carpeta que contiene a `borreguil_app`. Si no está, se omite.

No es una prueba de pytest: ejecuta la app al arrancar. Se apoya en detalles
internos de `streamlit.testing` para imitar la selección de filas de la tabla.
"""
import csv
import json
import os
import sys
import tempfile
import warnings
from collections import Counter
from pathlib import Path

APP = Path(__file__).resolve().parent.parent
DATOS = APP.parent / 'classification_v5.csv'
if not DATOS.exists():
    print(f'Se omite: no está {DATOS} (los puntos de prueba no van en el repositorio).')
    sys.exit(0)
os.chdir(APP)
sys.path.insert(0, str(APP))
os.environ['BORREGUIL_DATA_DIR'] = tempfile.mkdtemp(prefix='bt_')
warnings.simplefilter('ignore')

import borreguil_pipeline as bp   # noqa: E402
import mapeo_rf as mr             # noqa: E402
import point_profile as pp        # noqa: E402

bp._imports()
from streamlit.testing.v1 import AppTest   # noqa: E402

FALLOS = []


def comprobar(cond, texto, detalle=''):
    print(('     ok     ' if cond else '     FALLO  ') + texto
          + (f'  [{detalle}]' if detalle and not cond else ''))
    if not cond:
        FALLOS.append(texto)
    return cond


# ---------------------------------------------------------------- datos de prueba
with open(DATOS, encoding='utf-8') as f:
    todas = list(csv.DictReader(f))
base = []
for i, r in enumerate(todas[::20][:40]):
    d = {k: v for k, v in r.items() if k not in ('Borreguil', 'Duda')}
    d['lat'] = float(r['lat'])
    d['lon'] = float(r['lon'])
    d['ID'] = r.get('ID') or f'p{i}'
    base.append(d)
proba, info = bp.train_rf(base, default_model_path=str(APP / 'rf_sierra_nevada.joblib'))
for r, p in zip(base, proba):
    r['rf_proba'] = float(p)
# Etiquetas de campo que el modelo contradice: una AUSENCIA y dos puntos DUDOSOS,
# los tres con probabilidad alta. Nada debe borrarlas ni promoverlos.
_altos = [i for i, r in enumerate(base) if r['rf_proba'] >= 0.70]
NO_CAMPO, D1, D2 = _altos[-1], _altos[-2], _altos[-3]
base[NO_CAMPO]['Borreguil'] = 'no'
base[D1]['Duda'] = 'si'
base[D2]['Duda'] = 'si'
for r in base:
    r['decision'] = bp.decide(r, 0.5)
# Un punto «borreguil» al que le han fallado todas las descargas: no tiene ninguna
# variable, así que la app no puede proponerle tipo (los tres niveles quedan en «—»).
base.append({'ID': 'sin_datos', 'lat': base[0]['lat'] + 0.01, 'lon': base[0]['lon'] + 0.01,
             'rf_proba': 0.81, 'decision': 'POSIBLE BORREGUIL'})
bp.apply_typology(base, pp.is_borreguil_decision)
print('puntos de prueba:', len(base), dict(Counter(r['decision'] for r in base)))

BORR = [i for i, r in enumerate(base)
        if pp.is_borreguil_decision(r['decision']) and r['ID'] != 'sin_datos']
NOB = [i for i, r in enumerate(base) if not pp.is_borreguil_decision(r['decision'])
       and i not in (NO_CAMPO, D1, D2)]
SIN = len(base) - 1
assert len(BORR) >= 4 and len(NOB) >= 3, (len(BORR), len(NOB))
print(f'borreguiles: {len(BORR)} · no borreguiles: {len(NOB)} · sin datos: índice {SIN}')


# ------------------------------------------------------------------- utilidades
def textos(at):
    out = []
    for grupo in (at.markdown, at.caption, at.success, at.error, at.info, at.warning):
        out += [str(x.value) for x in grupo]
    return out


def hay(at, trozo):
    return any(trozo in t for t in textos(at))


# En el navegador, la tabla reenvía su selección en cada ejecución. AppTest no lo
# hace (la tabla no es un «widget» de su árbol), así que se imita aquí: se añade el
# estado de la tabla a lo que AppTest envía, igual que haría el navegador.
from streamlit.proto.WidgetStates_pb2 import WidgetState          # noqa: E402
from streamlit.testing.v1 import element_tree as _et              # noqa: E402

SELECCION = {'rows': []}
_estados_originales = _et.ElementTree.get_widget_states


def _estados_con_seleccion(self):
    from streamlit.proto.WidgetStates_pb2 import WidgetStates
    ws = WidgetStates()
    for node in self:
        try:
            w = _et.get_widget_state(node)
        except KeyError:
            # Nodo de una ejecución interrumpida por st.rerun(): su widget ya no
            # existe (cambió de clave). El navegador lo habría retirado.
            continue
        if w is not None:
            ws.widgets.append(w)
    for d in self.dataframe:
        if d.key == 'results_table' and d.proto.id:
            w = WidgetState()
            w.id = d.proto.id
            w.string_value = json.dumps({'selection': {
                'rows': sorted(SELECCION['rows']), 'columns': [], 'cells': []}})
            ws.widgets.append(w)
    return ws


_et.ElementTree.get_widget_states = _estados_con_seleccion


def sincroniza(at):
    """Si la app ha fijado la selección por programa (p. ej. la vacía al quitar
    puntos), el navegador la aplica y deja de reenviar la antigua."""
    for d in at.dataframe:
        if d.key == 'results_table' and d.proto.selection_state:
            SELECCION['rows'] = list(
                json.loads(d.proto.selection_state)['selection']['rows'])


def seleccionar(at, indices):
    SELECCION['rows'] = list(indices)
    at.run()
    sincroniza(at)
    return at


def desplegable(at, nivel):
    cs = [s for s in at.selectbox if (s.key or '').startswith(f'rev_{nivel}_')]
    return cs[0] if cs else None


def boton(at, clave):
    bs = [b for b in at.button if b.key == clave]
    return bs[0] if bs else None


def filas(at):
    return at.session_state['rows']


def sin_error(at, paso, exc=None):
    exc = list(at.exception) if exc is None else exc
    comprobar(not exc, f'{paso}: sin excepciones',
              ' | '.join(str(e.value)[:200] for e in exc[:2]))


def pulsar(at, b, paso):
    """Pulsa un botón y devuelve los textos de esa ejecución.

    Tras un st.rerun(), AppTest conserva en su árbol los elementos de la ejecución
    interrumpida además de los de la nueva (botones duplicados), cosa que el
    navegador no hace. Por eso se ejecuta una vez más: deja el árbol limpio, como
    la pantalla que ve la persona."""
    b.click()
    at.run()
    sincroniza(at)
    msgs, exc = textos(at), list(at.exception)
    sin_error(at, paso, exc)
    at.run()
    sincroniza(at)
    return msgs


def tabla_principal(at):
    for d in at.dataframe:
        if d.key == 'results_table':
            return d.value
    return None


def tabla_con_columna(at, *nombres):
    """La primera tabla de la pantalla que tiene una columna con ese nombre (en
    cualquiera de los dos idiomas)."""
    for d in at.dataframe:
        try:
            df = d.value
        except Exception:
            continue
        if any(n in df.columns for n in nombres):
            return df
    return None


def clases_del_mapeo(at):
    """{categoría: nº de muestras} de la pestaña «Mapa Modelo RF»."""
    df = tabla_con_columna(at, 'Categoría', 'Category')
    if df is None:
        return None
    return dict(zip(df.iloc[:, 0], df.iloc[:, 2]))


def control(at, tipo, clave):
    cs = [w for w in getattr(at, tipo) if w.key == clave]
    return cs[0] if cs else None


# ------------------------------------------------------------------------ guion
def guion(lang):
    en = lang == 'English'
    print(f'\n================ {lang} ================')
    SELECCION['rows'] = []
    work = Path(tempfile.mkdtemp(prefix='bw_'))
    at = AppTest.from_file(str(APP / 'app.py'), default_timeout=600)
    at.session_state['rows'] = [dict(r) for r in base]
    at.session_state['work'] = work
    at.session_state['info'] = {k: v for k, v in (info or {}).items() if k != 'model_bundle'}
    at.session_state['ui_lang'] = lang
    at.run()

    # --- A · pantalla inicial de resultados
    print('  A · pantalla de resultados')
    sin_error(at, 'A')
    df = tabla_principal(at)
    col_rev = 'type reviewed' if en else 'tipo_revisado'
    comprobar(df is not None and col_rev in df.columns,
              f'la tabla tiene la columna «{col_rev}»',
              str(list(df.columns)) if df is not None else 'sin tabla')
    n_b = len(BORR) + 1
    comprobar(hay(at, f'Type reviewed: 0 of {n_b}' if en else f'Tipo revisado: 0 de {n_b}'),
              f'contador a cero sobre {n_b} borreguiles')
    comprobar(boton(at, 'rev_save') is None, 'sin selección no aparece el panel de revisión')
    T = (lambda es_, en_: en_ if en else es_)
    # «Mapa Modelo RF» está en la fila de arriba, junto a «Verdad terreno», con sus
    # pasos
    # en pestañas propias; los resultados se quedan con seis.
    etiq = [t.label for t in at.tabs]
    comprobar(T('🌿  Mapa Modelo RF', '🌿  RF model map') in etiq
              and etiq.index(T('🌿  Mapa Modelo RF', '🌿  RF model map'))
              < etiq.index(T('📖  Documentación y recomendaciones',
                             '📖  Documentation & recommendations'))
              and not any('Mapeo RF' in e or 'RF mapping' in e for e in etiq),
              '«Mapa Modelo RF» es una pestaña de la fila de arriba', str(etiq))
    comprobar(all(p in etiq for p in (T('1 · Datos', '1 · Data'),
                                      T('2 · Tabla de entrenamiento', '2 · Training table'),
                                      T('3 · Modelo', '3 · Model')))
              and len(etiq) == 3 + 6 + 3,
              'sus tres pasos son pestañas, y los resultados tienen seis', str(etiq))
    comprobar(hay(at, T('**🌿 Mapa Modelo RF**, arriba del todo',
                        '**🌿 RF model map** tab, at the very top')),
              'la pestaña «Tabla» dice dónde sigue lo que se revisa')
    comprobar(all(hay(at, x) for x in (
        T('1 · Datos · faltan muestras', '1 · Data · not enough samples'),
        T('2 · Tabla · sin leer', '2 · Table · not read yet'),
        T('3 · Modelo · sin entrenar', '3 · Model · not trained yet'),
        T('4 · Mapa · todavía no disponible', '4 · Map · not available yet'))),
        'la línea de estado dice por dónde va cada paso')
    # Sin tipos revisados solo cuenta la ausencia de campo, y con eso no se puede
    # entrenar.
    comprobar(clases_del_mapeo(at) == {T('no borreguil', 'not a borreguil'): 1,
                                       T('arroyo', 'stream'): 0, T('laguna', 'lake'): 0,
                                       T('ladera', 'slope'): 0},
              'mapeo: recuento por categoría antes de revisar nada',
              str(clases_del_mapeo(at)))
    comprobar(hay(at, T('**1 muestra** de los 41 puntos', '**1 sample** out of the 41 points'))
              and hay(at, T('Solo hay muestras de una categoría',
                            'Only one category has samples'))
              and hay(at, T('el mínimo es 5', 'the minimum is 5')),
              'mapeo: avisa de que con eso no se puede entrenar')
    comprobar(hay(at, T('34 de 34 variables seleccionadas', '34 of 34 variables selected')),
              'mapeo: las 34 variables disponibles vienen seleccionadas')
    # Cada parámetro del modelo lleva su ayuda: qué es y qué valor conviene.
    ayudas = {k: (control(at, tipo, k).help or '') for tipo, k in (
        ('number_input', 'mrf_p_arboles'), ('number_input', 'mrf_p_prof'),
        ('number_input', 'mrf_p_semilla'), ('number_input', 'mrf_p_hoja'),
        ('selectbox', 'mrf_p_maxfeat'), ('selectbox', 'mrf_p_balance'))}
    comprobar(all(T('ecomendado', 'ecommended') in a and len(a) > 150
                  for a in ayudas.values()),
              'cada parámetro del modelo explica qué es y qué valor se recomienda',
              str({k: a[:30] for k, a in ayudas.items()}))
    comprobar(T('raíz cuadrada y logaritmo prueban las dos 5',
                'square root and logarithm both try 5') in ayudas['mrf_p_maxfeat'],
              'la ayuda de «variables por división» hace la cuenta con las 34 elegidas',
              ayudas['mrf_p_maxfeat'][-200:])
    tarjetas = [m.label for m in at.metric]
    comprobar(('LIKELY BORREGUIL' in tarjetas) if en else ('BORREGUIL PROBABLE' in tarjetas),
              'las tarjetas de recuento van en el idioma elegido', str(tarjetas))
    ids = [r['ID'] for r in base]
    comprobar(df is not None and list(df['ID']) == bp.abreviar_ids(ids)
              and max(len(str(x)) for x in df['ID']) <= 24 < max(len(x) for x in ids),
              'los nombres largos se muestran abreviados en la tabla',
              str(list(df['ID'])[:3]) if df is not None else 'sin tabla')
    comprobar(len(set(df['ID'])) == len(set(ids)), 'y siguen distinguiéndose unos de otros')

    # --- B · seleccionar un borreguil
    print('  B · seleccionar un borreguil')
    i1 = BORR[0]
    seleccionar(at, [i1])
    sin_error(at, 'B')
    etiquetas = [e.label for e in at.expander]
    esperado = ('🌿 Review the borreguil type of the selected points' if en else
                '🌿 Revisar el tipo de borreguil de los puntos seleccionados')
    comprobar(esperado in etiquetas, 'aparece el panel de revisión', str(etiquetas))
    r1 = dict(filas(at)[i1])
    for niv in bp.NIVELES_TIPO:
        d = desplegable(at, niv)
        comprobar(d is not None and d.value == r1[niv],
                  f'el desplegable de {niv} arranca en el tipo del punto ({r1[niv]})',
                  repr(d.value) if d is not None else 'no existe')
    comprobar(hay(at, 'Orthophoto background' if en else 'Ortofoto de fondo'),
              'el mapa del punto anuncia la ortofoto y el píxel de 10 m')
    comprobar(hay(at, "app's proposal, not reviewed" if en else
                  'propuesta de la app, sin revisar'),
              'la ficha del punto dice que el tipo está sin revisar')
    comprobar(hay(at, bp.nombre_html(r1['ID'])),
              'la ficha muestra el nombre completo, con cortes en los separadores')
    comprobar(hay(at, f"**{bp.fmt_num(r1.get('elev_dem_m'))} m**")
              and not hay(at, str(r1.get('elev_dem_m')) + ' m'),
              'los números de la ficha van con 4 decimales')
    d = desplegable(at, 'ambiente')
    opciones = list(d.options) if d is not None else []
    comprobar(opciones == (['(no change)', 'stream', 'lake', 'slope'] if en else
                           ['(sin cambio)', 'arroyo', 'laguna', 'ladera']),
              'opciones de ambiente en el idioma elegido', str(opciones))
    comprobar([b.label for b in at.button if b.key == 'rev_save']
              == ['✓ Save reviewed type' if en else '✓ Guardar tipo revisado'],
              'botón de guardar en el idioma elegido')

    # --- C · corregir la humedad y guardar
    print('  C · corregir la humedad y guardar')
    nueva_hum = 'seco' if r1['humedad'] == 'húmedo' else 'húmedo'
    desplegable(at, 'humedad').select(nueva_hum)
    msgs = pulsar(at, boton(at, 'rev_save'), 'C')
    r = filas(at)[i1]
    comprobar(bp.tipo_revisado(r), 'el punto queda marcado como revisado')
    comprobar(r['humedad'] == nueva_hum and r['humedad_regla'] == r1['humedad'],
              'humedad corregida y propuesta de la app conservada',
              f"{r.get('humedad')} / {r.get('humedad_regla')}")
    comprobar(r['ambiente'] == r1['ambiente'] and r['pureza'] == r1['pureza'],
              'los niveles no tocados quedan confirmados tal cual')
    comprobar(bool(r.get('tipo_revisado_fecha')), 'queda la fecha de la revisión')
    comprobar(any(('Reviewed type saved for 1 point(s).' if en else
                   'Tipo revisado guardado en 1 punto(s).') in m for m in msgs),
              'mensaje de confirmación')
    comprobar(hay(at, f'Type reviewed: 1 of {n_b}' if en else f'Tipo revisado: 1 de {n_b}'),
              'el contador pasa a 1 revisado')
    comprobar(hay(at, 'moisture 1' if en else 'humedad 1'),
              'el contador registra 1 corrección de humedad')
    df = tabla_principal(at)
    comprobar(df is not None and df[col_rev].iloc[i1] == ('yes' if en else 'sí')
              and (df[col_rev] != '').sum() == 1, 'la tabla marca solo ese punto')
    for nombre in ('classification.csv', 'puntos.geojson', 'Clasificacion_puntos.xlsx',
                   'mapa.html'):
        comprobar((work / nombre).exists(), f'se ha guardado {nombre}')
    gj = json.loads((work / 'puntos.geojson').read_text(encoding='utf-8'))
    props = {f['properties']['ID']: f['properties'] for f in gj['features']}
    comprobar(props[r['ID']]['tipo_revisado'] == 'si'
              and props[r['ID']]['humedad'] == nueva_hum, 'puntos.geojson lleva la revisión')
    d = desplegable(at, 'humedad')
    comprobar(d is not None and d.value == nueva_hum,
              'tras guardar, el desplegable muestra el valor guardado',
              repr(d.value) if d is not None else 'no existe')
    comprobar(hay(at, '**reviewed**' if en else '**revisado**'),
              'la ficha del punto pasa a decir «revisado»')
    comprobar(SELECCION['rows'] == [i1], 'la selección se conserva al guardar')

    # --- D · deshacer
    print('  D · deshacer la revisión')
    msgs = pulsar(at, boton(at, 'rev_undo'), 'D')
    r = filas(at)[i1]
    comprobar(not bp.tipo_revisado(r) and r['humedad'] == r1['humedad'],
              'el punto vuelve a la propuesta de la app',
              f"{r.get('tipo_revisado')!r} / {r.get('humedad')}")
    comprobar(any(('Review undone for 1 point(s).' if en else
                   'Revisión deshecha en 1 punto(s).') in m for m in msgs),
              'mensaje de deshacer')
    d = desplegable(at, 'humedad')
    comprobar(d is not None and d.value == r1['humedad'],
              'tras deshacer, el desplegable vuelve a mostrar la propuesta',
              repr(d.value) if d is not None else 'no existe')

    # --- E · varios puntos a la vez (dos borreguiles y un no borreguil)
    print('  E · varios puntos a la vez')
    i2, j = BORR[1], NOB[0]
    antes = {i: dict(filas(at)[i]) for i in (i1, i2, j)}
    seleccionar(at, [i1, i2, j])
    sin_error(at, 'E (selección)')
    comprobar(hay(at, '2 point(s) to review · 1 are not borreguil and are skipped' if en else
                  '2 punto(s) a revisar · 1 no son borreguil y se ignoran'),
              'avisa de cuántos se revisan y cuántos se ignoran')
    comprobar(all(desplegable(at, n).value == '(sin cambio)' for n in bp.NIVELES_TIPO),
              'con varios puntos los desplegables arrancan en «(sin cambio)»')
    desplegable(at, 'pureza').select('mixto-roca')
    pulsar(at, boton(at, 'rev_save'), 'E (guardar)')
    comprobar(all(desplegable(at, n).value == '(sin cambio)' for n in bp.NIVELES_TIPO),
              'tras guardar varios, los desplegables vuelven a «(sin cambio)»')
    for i in (i1, i2):
        r = filas(at)[i]
        comprobar(bp.tipo_revisado(r) and r['pureza'] == 'mixto-roca'
                  and r['ambiente'] == antes[i]['ambiente']
                  and r['humedad'] == antes[i]['humedad'],
                  f"{r['ID']}: pureza corregida, ambiente y humedad confirmados")
    r = filas(at)[j]
    comprobar(not bp.tipo_revisado(r) and r.get('ambiente', '') == '',
              'el no borreguil seleccionado no se toca')

    # --- F · solo no borreguiles
    print('  F · selección sin borreguiles')
    seleccionar(at, NOB[:2])
    sin_error(at, 'F')
    comprobar(hay(at, 'The selected points are not classified as borreguil' if en else
                  'Los puntos seleccionados no están clasificados como borreguil'),
              'explica que esos puntos no tienen tipo')
    comprobar(boton(at, 'rev_save') is None, 'no ofrece guardar')

    # --- G · todo o nada: un punto sin datos y sin elegir sus niveles
    print('  G · todo o nada')
    i3 = BORR[2]
    seleccionar(at, [i3, SIN])
    sin_error(at, 'G (selección)')
    desplegable(at, 'humedad').select('seco')
    boton(at, 'rev_save').click()       # sin st.rerun(): el error se queda en pantalla
    at.run()
    sin_error(at, 'G (guardar)')
    comprobar(hay(at, 'Nothing was saved.' if en else 'No se ha guardado nada.'),
              'avisa de que no se ha guardado nada')
    comprobar(hay(at, 'sin_datos'), 'el aviso dice qué punto lo impide')
    comprobar(not bp.tipo_revisado(filas(at)[i3]) and not bp.tipo_revisado(filas(at)[SIN]),
              'ninguno de los dos queda revisado')
    for niv, val in (('ambiente', 'ladera'), ('humedad', 'seco'), ('pureza', 'puro')):
        desplegable(at, niv).select(val)
    pulsar(at, boton(at, 'rev_save'), 'G (guardar eligiendo los tres)')
    r = filas(at)[SIN]
    comprobar(bp.tipo_revisado(r) and (r['ambiente'], r['humedad'], r['pureza'])
              == ('ladera', 'seco', 'puro'), 'eligiendo los tres niveles sí se guarda')

    # --- G2 · la pestaña «Mapa Modelo RF» cuenta lo revisado
    print('  G2 · mapeo: muestras por categoría')
    control(at, 'selectbox', 'mrf_nivel').select('pureza')
    at.run()
    sin_error(at, 'G2')
    # Revisados hasta aquí: dos «mixto-roca» (paso E) y dos «puro» (paso G), más la
    # ausencia de campo. Los dos dudosos no cuentan.
    comprobar(clases_del_mapeo(at) == {T('no borreguil', 'not a borreguil'): 1,
                                       T('puro', 'pure'): 2,
                                       T('mixto-agua', 'mixed-water'): 0,
                                       T('mixto-roca', 'mixed-rock'): 2},
              'el recuento sigue a lo que se ha revisado en la tabla',
              str(clases_del_mapeo(at)))
    comprobar(hay(at, T('**5 muestras** de los 41 puntos', '**5 samples** out of the 41 points')),
              'cinco muestras de 41 puntos')
    motivos = tabla_con_columna(at, 'Motivo', 'Reason')
    mot = dict(zip(motivos.iloc[:, 0], motivos.iloc[:, 1])) if motivos is not None else {}
    comprobar(mot.get(mr.motivo('dudoso', en)) == 2
              and mot.get(mr.motivo('tipo_sin_revisar', en)) == len(BORR) - 3,
              'explica por qué no entra el resto (dudosos, tipo sin revisar…)', str(mot))
    comprobar(sum(mot.values()) + 5 == 41, 'muestras y descartes suman todos los puntos')
    # La humedad con el ambiente dentro: seis categorías, compuestas con lo que se
    # revisó en la tabla (más la ausencia de campo).
    control(at, 'selectbox', 'mrf_nivel').select('humedad_ambiente')
    at.run()
    sin_error(at, 'G2 (humedad y ambiente)')
    _en = {'húmedo': 'wet', 'seco': 'dry', 'arroyo': 'stream', 'laguna': 'lake',
           'ladera': 'slope', 'no borreguil': 'not a borreguil'}
    comb = (lambda c: ' · '.join(_en.get(p, p) for p in c.split(' · ')) if en else c)
    esperado = mr.muestras(filas(at), 'humedad_ambiente',
                           es_borreguil=pp.is_borreguil_decision)
    vistas = clases_del_mapeo(at)
    comprobar(vistas == {comb(c['categoria']): c['n'] for c in esperado['clases']}
              and len(vistas) == 7 and sum(vistas.values()) == 5,
              'humedad y ambiente: seis categorías combinadas y la ausencia', str(vistas))
    comprobar(T('húmedo · arroyo', 'wet · stream') in vistas
              and T('seco · ladera', 'dry · slope') in vistas,
              'con el nombre compuesto, en el idioma elegido', str(list(vistas)))
    comprobar(hay(at, T('1 · Datos · faltan muestras', '1 · Data · not enough samples')),
              'y, con tan pocas, la línea de estado dice que faltan muestras')
    control(at, 'selectbox', 'mrf_nivel').select('pureza')
    at.run()
    control(at, 'checkbox', 'mrf_ausencias').uncheck()
    at.run()
    sin_error(at, 'G2 (sin ausencias)')
    comprobar(T('no borreguil', 'not a borreguil') not in clases_del_mapeo(at)
              and hay(at, T('no demuestra presencia frente a ausencia',
                            'does not show presence versus absence')),
              'sin la clase de ausencia, avisa de lo que eso implica')
    control(at, 'checkbox', 'mrf_ausencias').check()
    control(at, 'multiselect', 'mrf_vars_4').set_value(['elev_dem_m', 'slope_deg'])
    at.run()
    comprobar(hay(at, T('31 de 34 variables seleccionadas', '31 of 34 variables selected')),
              'se pueden quitar variables')
    for i in range(5):
        control(at, 'multiselect', f'mrf_vars_{i}').set_value([])
    at.run()
    comprobar(hay(at, T('Elige al menos una variable', 'Choose at least one variable')),
              'sin variables, lo dice')
    for i, g in enumerate(mr.GRUPOS):
        control(at, 'multiselect', f'mrf_vars_{i}').set_value(mr.nombres_predictores([g]))
    control(at, 'selectbox', 'mrf_nivel').select('ambiente')
    at.run()
    sin_error(at, 'G2 (restaurar)')

    # --- H · verificar y reentrenar conserva lo revisado
    print('  H · marcar como verificado y reentrenar')
    revisados = {r['ID']: (r['ambiente'], r['humedad'], r['pureza'])
                 for r in filas(at) if bp.tipo_revisado(r)}
    comprobar(len(revisados) == 4, 'hay 4 puntos revisados antes de reentrenar',
              str(len(revisados)))
    k = NOB[1]
    seleccionar(at, [i1, k, D2])
    pulsar(at, boton(at, 'mark_verified'), 'H')
    r = filas(at)[D2]
    comprobar(r['Duda'] == '' and r['decision'] == 'BORREGUIL VERIFICADO',
              'verificar a mano un punto dudoso resuelve la duda',
              f"{r.get('Duda')!r} / {r['decision']}")
    r = filas(at)[D1]
    comprobar(r['Duda'] == 'si' and r['decision'] == 'DUDOSO (campo)',
              'el dudoso no seleccionado sigue siendo dudoso', r['decision'])
    r = filas(at)[NO_CAMPO]
    comprobar(r['Borreguil'] == 'no' and r['decision'] == 'NO BORREGUIL VERIFICADO',
              'la ausencia de campo sigue siendo ausencia', r['decision'])
    despues = {r['ID']: (r['ambiente'], r['humedad'], r['pureza'])
               for r in filas(at) if bp.tipo_revisado(r)}
    comprobar(despues == revisados, 'los tipos revisados sobreviven al reentrenamiento',
              f'{len(despues)} revisados')
    r = filas(at)[k]
    comprobar(r['decision'] == 'BORREGUIL VERIFICADO'
              and r['ambiente'] in bp.NIVELES_TIPO['ambiente'] and not bp.tipo_revisado(r),
              'el punto recién verificado recibe una propuesta de tipo (sin revisar)',
              f"{r['decision']} / {r.get('ambiente')!r}")

    # --- I · iteración de auto-entrenamiento
    print('  I · iteración de auto-entrenamiento')
    its = [b for b in at.button if 'iteración' in b.label or 'iteration' in b.label.lower()]
    comprobar(len(its) == 1, 'existe el botón de la siguiente iteración',
              str([b.label for b in at.button]))
    if its:
        pulsar(at, its[0], 'I')
        despues = {r['ID']: (r['ambiente'], r['humedad'], r['pureza'])
                   for r in filas(at) if bp.tipo_revisado(r)}
        comprobar(despues == revisados, 'los tipos revisados sobreviven a la iteración')
        sin_tipo = [r['ID'] for r in filas(at)
                    if pp.is_borreguil_decision(r['decision']) and r['ID'] != 'sin_datos'
                    and r.get('ambiente') not in bp.NIVELES_TIPO['ambiente']]
        comprobar(not sin_tipo, 'todos los borreguiles tras la iteración tienen tipo',
                  str(sin_tipo[:5]))
        r = filas(at)[NO_CAMPO]
        comprobar(r['Borreguil'] == 'no' and r['decision'] == 'NO BORREGUIL VERIFICADO',
                  f"la ausencia de campo (prob. {base[NO_CAMPO]['rf_proba']:.2f}) no se "
                  'promueve ni se borra', f"{r['Borreguil']!r} / {r['decision']}")
        r = filas(at)[D1]
        comprobar(r['Duda'] == 'si' and r['Borreguil'] == ''
                  and r['decision'] == 'DUDOSO (campo)',
                  'el punto dudoso sigue dudoso tras la iteración',
                  f"{r.get('Duda')!r} / {r['decision']}")
        autos = [r for r in filas(at) if r['decision'] == 'BORREGUIL (auto)']
        comprobar(len(autos) > 0, 'la iteración sí promueve candidatos sin etiqueta',
                  str(len(autos)))

    # --- J · quitar un punto no revisado
    print('  J · quitar puntos del análisis')
    n0 = len(filas(at))
    quitar = NOB[2]
    id_quitado = filas(at)[quitar]['ID']
    seleccionar(at, [quitar])
    pulsar(at, boton(at, 'remove_pts'), 'J')
    comprobar(len(filas(at)) == n0 - 1
              and id_quitado not in {r['ID'] for r in filas(at)}, 'el punto desaparece')
    comprobar(SELECCION['rows'] == [] and boton(at, 'remove_pts') is None
              and boton(at, 'rev_save') is None,
              'tras quitar, no queda ningún punto seleccionado', str(SELECCION['rows']))
    despues = {r['ID']: (r['ambiente'], r['humedad'], r['pureza'])
               for r in filas(at) if bp.tipo_revisado(r)}
    comprobar(despues == revisados, 'los revisados siguen intactos')

    # --- K · el fichero guardado se puede volver a cargar
    print('  K · volver a cargar puntos.geojson')
    otra = bp.read_points(str(work / 'puntos.geojson'))
    comprobar(len(otra) == n0 - 1, 'el fichero tiene los mismos puntos que la pantalla',
              str(len(otra)))
    vuelta = {r['ID']: (r['ambiente'], r['humedad'], r['pureza'])
              for r in otra if bp.tipo_revisado(r)}
    comprobar(vuelta == revisados, 'al recargarlo vuelven los 4 tipos revisados',
              f'{len(vuelta)} revisados')
    comprobar(not any('ndmi_mean' in r for r in otra),
              'el fichero no arrastra variables predictoras')
    por_id = {r['ID']: r for r in otra}
    comprobar(por_id[base[NO_CAMPO]['ID']]['Borreguil'] == 'no'
              and por_id[base[D1]['ID']]['Duda'] == 'si',
              'el fichero conserva la ausencia y la duda de campo')
    comprobar(all(por_id[r['ID']]['Borreguil'] == '' for r in filas(at)
                  if r['decision'] == 'BORREGUIL (auto)')
              and all(por_id[r['ID']]['Borreguil'] == 'si' for r in filas(at)
                      if r['decision'] == 'BORREGUIL VERIFICADO'),
              'en el fichero solo es verdad-terreno lo verificado, no lo promovido')

    # --- L · nada del texto nuevo queda en español en la versión inglesa
    if en:
        print('  L · textos nuevos en inglés')
        seleccionar(at, [BORR[3]])
        visibles = textos(at) + [e.label for e in at.expander] + [b.label for b in at.button] \
            + [s.label for s in at.selectbox]
        restos = [t[:70] for t in visibles
                  if any(p in t for p in ('Revisar el tipo', 'Guardar tipo', 'Deshacer',
                                          'a revisar', 'Tipo revisado', 'sin cambio',
                                          'Ambiente', 'Humedad', 'Pureza'))]
        comprobar(not restos, 'sin restos en español en el panel de revisión', str(restos))


# ------------------------------------------------- tabla de entrenamiento
# Con suficientes puntos revisados, leer las variables y montar la tabla X/y.
# Earth Engine se sustituye por un lector simulado (misma firma que el de verdad):
# la primera vez deja un punto sin consultar y a otro le falta una variable.
import zlib                                  # noqa: E402

import gee_backend as geb                    # noqa: E402

LLAMADAS = []


def _lector_simulado(puntos, years, epsg=None, progress=None):
    LLAMADAS.append({'n': len(puntos), 'years': tuple(years), 'epsg': epsg})
    primera = len(LLAMADAS) == 1
    valores = []
    for k, (lon, lat) in enumerate(puntos):
        if primera and k == 3:               # fallo de Earth Engine: sin consultar
            valores.append(None)
            continue
        v = {b: (zlib.crc32(f'{lon:.5f},{lat:.5f},{j}'.encode()) % 1000) / 1000.0
             for j, b in enumerate(geb.BANDAS_PILA)}
        if primera and k == 5:               # píxel sin dato de una variable
            v['ndmi_mean'] = None
        valores.append(v)
    if progress:
        progress(1.0)
    return valores, sum(v is None for v in valores), geb.info_pila(years, epsg or 32630)


def guion_tabla(lang):
    en = lang == 'English'
    T = (lambda es_, en_: en_ if en else es_)
    print(f'\n================ {lang} · tabla de entrenamiento ================')
    LLAMADAS.clear()
    SELECCION['rows'] = []
    datos = Path(tempfile.mkdtemp(prefix='bt_'))
    os.environ['BORREGUIL_DATA_DIR'] = str(datos)
    geb.initialize = lambda project=None, service_account_json=None: (True, 'simulado')
    geb.leer_predictores = _lector_simulado

    # 14 borreguiles con el tipo revisado (7 húmedos, 7 secos) y 8 ausencias de campo
    # (7 que se marcan aquí más la que ya traían los datos de prueba): 22 muestras.
    rows = [dict(r) for r in base if r['ID'] != 'sin_datos']
    borr = [r for r in rows if pp.is_borreguil_decision(r['decision'])][:14]
    for k, r in enumerate(borr):
        bp.set_tipo_revisado(r, humedad=('húmedo' if k % 2 else 'seco'))
    for r in [r for r in rows if r['decision'] == 'NO BORREGUIL'][:7]:
        r['Borreguil'] = 'no'
        r['decision'] = bp.decide(r, 0.5)
    bp.apply_typology(rows, pp.is_borreguil_decision)

    def nueva():
        a = AppTest.from_file(str(APP / 'app.py'), default_timeout=600)
        a.session_state['rows'] = [dict(r) for r in rows]
        a.session_state['work'] = Path(tempfile.mkdtemp(prefix='bw_'))
        a.session_state['info'] = {k: v for k, v in (info or {}).items()
                                   if k != 'model_bundle'}
        a.session_state['ui_lang'] = lang
        a.run()
        return a

    at = nueva()
    control(at, 'selectbox', 'mrf_nivel').select('humedad')
    at.run()
    sin_error(at, 'T1')
    comprobar(clases_del_mapeo(at) == {T('no borreguil', 'not a borreguil'): 8,
                                       T('húmedo', 'wet'): 7, T('seco', 'dry'): 7},
              'hay tres categorías: 8 ausencias, 7 húmedos y 7 secos', str(clases_del_mapeo(at)))
    b = control(at, 'button', 'mrf_leer')
    comprobar(b is not None and b.disabled and '22' in b.label,
              'sin proyecto de Earth Engine, el botón de leer está desactivado',
              f'{b.label if b else None}')
    comprobar(hay(at, T('1 · Datos · 22 muestras', '1 · Data · 22 samples'))
              and hay(at, T('2 · Tabla · sin leer', '2 · Table · not read yet')),
              'estado: datos listos, tabla sin leer')

    print('  T2 · primera lectura (un punto falla, a otro le falta una variable)')
    control(at, 'text_input', 'mrf_proyecto').set_value('ee-proyecto-de-prueba')
    at.run()
    msgs = pulsar(at, control(at, 'button', 'mrf_leer'), 'T2')
    comprobar(LLAMADAS and LLAMADAS[0]['n'] == 22 and LLAMADAS[0]['years'] == tuple(range(2017, 2026))
              and LLAMADAS[0]['epsg'] == 32630,
              'se piden las 22 muestras, con los años del panel lateral y la zona UTM',
              str(LLAMADAS[:1]))
    comprobar(any(T('21 muestras leídas; 1 no se pudieron consultar',
                    '21 samples read; 1 could not be queried') in m for m in msgs),
              'dice cuántas se han leído y cuántas han fallado')
    comprobar(hay(at, T('**20 muestras con todas sus variables** (34 variables)',
                        '**20 samples with all their variables** (34 variables)')),
              'la tabla tiene 20 muestras: 22 menos la que falló y la que no tiene dato')
    comprobar(hay(at, mr.motivo_tabla('sin_leer', en))
              and hay(at, mr.motivo_tabla('sin_dato', en)),
              'explica por qué faltan esas dos')
    b = control(at, 'button', 'mrf_leer')
    comprobar(b is not None and not b.disabled and ' 1 ' in b.label,
              'el botón ofrece reintentar solo la muestra que falló',
              f'{b.label if b else None}')
    comprobar(hay(at, T('2 · Tabla · falta 1 muestra por leer',
                        '2 · Table · 1 sample still to read')),
              'estado: a la tabla le falta una muestra por leer')

    print('  T3 · reintento')
    msgs = pulsar(at, control(at, 'button', 'mrf_leer'), 'T3')
    comprobar(len(LLAMADAS) == 2 and LLAMADAS[1]['n'] == 1, 'solo se vuelve a pedir esa muestra',
              str(LLAMADAS[1:]))
    comprobar(hay(at, T('**21 muestras con todas sus variables**',
                        '**21 samples with all their variables**')),
              'ahora son 21: sigue fuera la que no tiene dato')
    comprobar(control(at, 'button', 'mrf_leer') is None
              and control(at, 'button', 'mrf_releer') is not None,
              'ya no queda nada por leer')
    comprobar(hay(at, T('2 · Tabla · 21 muestras, 34 variables',
                        '2 · Table · 21 samples, 34 variables'))
              and hay(at, T('3 · Modelo · sin entrenar', '3 · Model · not trained yet')),
              'estado: tabla completa, modelo sin entrenar')
    tabla = tabla_con_columna(at, 'En la tabla', 'In the table')
    comprobar(tabla is not None and sorted(tabla.iloc[:, 3]) == [6, 7, 8]
              and list(tabla.iloc[:, 2]) == [8, 7, 7],
              'la tabla de categorías compara las muestras con las que entran',
              str(tabla.values.tolist()) if tabla is not None else 'no está')
    comprobar(hay(at, T('Hay más variables (34) que muestras (21)',
                        'There are more variables (34) than samples (21)')),
              'el control de calidad avisa de que hay más variables que muestras')
    xy = tabla_con_columna(at, 'ndvi_early')
    comprobar(xy is not None and len(xy) == 21
              and list(xy.columns[3:]) == mr.nombres_predictores(),
              'la tabla X/y tiene 21 filas y las 34 variables en el orden del catálogo',
              str(list(xy.columns[:5])) if xy is not None else 'no está')

    print('  M1 · entrenar y validar')
    b = control(at, 'button', 'mrf_entrenar')
    comprobar(b is not None and not b.disabled, 'con la tabla lista, se puede entrenar')
    b.click()
    at.run()
    sin_error(at, 'M1')
    mod = at.session_state['mrf_modelos'].get('humedad') if 'mrf_modelos' in at.session_state \
        else None
    comprobar(mod is not None and mod['n_muestras'] == 21 and len(mod['predictores']) == 34
              and mod['params'] == mr.PARAMS_DEFECTO and mod['balance']['estrategia'] == 'pesos',
              'queda un modelo de humedad con 21 muestras, 34 variables y los parámetros '
              'por defecto', str({k: mod[k] for k in ('n_muestras', 'params')}) if mod else 'no')
    comprobar(mod is not None and sum(map(sum, mod['matriz'])) == 21
              and [c['categoria'] for c in mod['categorias']] == ['no borreguil', 'húmedo', 'seco'],
              'la matriz de confusión reparte las 21 muestras entre las tres categorías')
    comprobar(mod is not None and mod['validacion']['tipo'] == 'espacial'
              and hay(at, T('**Validación por zonas**', '**Validation by zones**')),
              'la validación se ha hecho por zonas, y se dice',
              str(mod['validacion']) if mod else 'no')
    etiquetas = [m.label for m in at.metric]
    comprobar(T('F1 medio entre categorías (macro-F1)', 'Mean F1 across categories (macro-F1)')
              in etiquetas and T('Exactitud equilibrada', 'Balanced accuracy') in etiquetas,
              'se muestran el macro-F1 y la exactitud equilibrada, no solo los aciertos',
              str(etiquetas[-4:]))
    por_cat = tabla_con_columna(at, 'Precisión', 'Precision')
    comprobar(por_cat is not None and len(por_cat) == 3 and 'F1' in por_cat.columns,
              'hay precisión, sensibilidad y F1 de cada categoría')
    matriz = tabla_con_columna(at, 'es ↓ · dice →', 'is ↓ · says →')
    comprobar(matriz is not None and matriz.shape == (3, 4)
              and int(matriz.iloc[:, 1:].to_numpy().sum()) == 21,
              'se muestra la matriz de confusión con sus categorías')
    comprobar(hay(at, T('**Modelo final** entrenado con 21 de las 21 muestras y 34 variables',
                        '**Final model** trained with 21 of the 21 samples and 34 variables')),
              'el modelo final se entrena con todas las muestras de la tabla')
    comprobar(mod is not None
              and hay(at, T('3 · Modelo · F1 medio ', '3 · Model · mean F1 ')
                      + bp.fmt_num(mod['macro_f1'])),
              'estado: modelo entrenado, con su F1 medio')

    print('  T4 · quitar variables no obliga a leer otra vez')
    control(at, 'multiselect', 'mrf_vars_3').set_value(['wavi_mean', 'albedo_mean'])
    at.run()
    sin_error(at, 'T4')
    comprobar(hay(at, T('**22 muestras con todas sus variables** (20 variables)',
                        '**22 samples with all their variables** (20 variables)'))
              and len(LLAMADAS) == 2,
              'sin «ndmi_mean», la muestra que no la tenía vuelve a entrar; no se relee')
    comprobar(not hay(at, T('Hay más variables', 'There are more variables')),
              'y ya no hay más variables que muestras')

    print('  M2 · el modelo avisa si la tabla ha cambiado; reentrenar con otros parámetros')
    aviso_tabla = T('La tabla ha cambiado desde que se entrenó este modelo',
                    'The table has changed since this model was trained')
    comprobar(hay(at, aviso_tabla), 'avisa de que el modelo es de la tabla anterior')
    comprobar(hay(at, T('3 · Modelo · es de una tabla anterior',
                        '3 · Model · from a previous table')),
              'estado: lo dice también la línea de arriba')
    control(at, 'number_input', 'mrf_p_arboles').set_value(100)
    control(at, 'number_input', 'mrf_p_prof').set_value(6)
    control(at, 'selectbox', 'mrf_p_balance').select('submuestreo')
    at.run()
    comprobar(T('Recomendado: 3', 'Recommended: 3')
              in (control(at, 'number_input', 'mrf_p_razon').help or ''),
              'el tope del submuestreo también explica qué es y qué valor conviene')
    control(at, 'button', 'mrf_entrenar').click()
    at.run()
    sin_error(at, 'M2')
    mod = at.session_state['mrf_modelos']['humedad']
    comprobar(mod['params']['n_estimators'] == 100 and mod['params']['max_depth'] == 6
              and mod['modelo'].n_estimators == 100 and mod['modelo'].max_depth == 6,
              'el modelo nuevo usa los parámetros elegidos', str(mod['params']))
    comprobar(mod['balance']['estrategia'] == 'submuestreo'
              and mod['modelo'].class_weight is None and mod['n_muestras'] == 22
              and len(mod['predictores']) == 20,
              'y la estrategia de balance elegida, con las 22 muestras y 20 variables',
              str(mod['balance']))
    comprobar(not hay(at, aviso_tabla)
              and hay(at, T('3 · Modelo · F1 medio ', '3 · Model · mean F1 ')
                      + bp.fmt_num(mod['macro_f1'])),
              'ya no avisa: el modelo es de la tabla actual')

    print('  T5 · cambiar los años invalida lo leído')
    [t for t in at.text_input if t.label.startswith(T('Años Sentinel-2', 'Sentinel-2 years'))
     ][0].set_value('2024')
    at.run()
    sin_error(at, 'T5')
    b = control(at, 'button', 'mrf_leer')
    comprobar(b is not None and '22' in b.label
              and not hay(at, T('con todas sus variables', 'with all their variables')),
              'con otros años hay que leer de nuevo las 22', f'{b.label if b else None}')
    comprobar(hay(at, T('2 · Tabla · sin leer', '2 · Table · not read yet'))
              and hay(at, T('3 · Modelo · es de una tabla anterior',
                            '3 · Model · from a previous table')),
              'estado: tabla sin leer y modelo de una tabla anterior')

    print('  T6 · el proyecto se recuerda')
    guardado = json.loads((datos / 'ajustes.json').read_text(encoding='utf-8'))
    comprobar(guardado == {'gee_project': 'ee-proyecto-de-prueba'},
              'queda guardado en la carpeta de datos', str(guardado))
    otra = nueva()
    comprobar(control(otra, 'text_input', 'mrf_proyecto').value == 'ee-proyecto-de-prueba'
              and any(t.value == 'ee-proyecto-de-prueba' for t in otra.text_input
                      if 'Earth Engine' in t.label and t.key != 'mrf_proyecto'),
              'al abrir de nuevo, aparece ya escrito en la pestaña y en el panel lateral')


# ------------------------------------------------------------------ sin puntos
# Antes de ejecutar nada, «Mapa Modelo RF» explica qué hace falta. Y si una
# ejecución se detiene a medias (aquí, generar puntos en un área que no se
# encuentra), la pestaña, que se dibuja al final del guion, no queda en blanco.
def guion_sin_puntos(lang):
    en = lang == 'English'
    T = (lambda es_, en_: en_ if en else es_)
    print(f'\n================ {lang} · sin puntos ================')
    SELECCION['rows'] = []
    import study_area as sa
    at = AppTest.from_file(str(APP / 'app.py'), default_timeout=600)
    at.session_state['ui_lang'] = lang
    at.run()
    sin_error(at, 'Z1')
    etiq = [t.label for t in at.tabs]
    comprobar(etiq == [T('📍  Verdad terreno', '📍  Ground truth'),
                       T('🌿  Mapa Modelo RF', '🌿  RF model map'),
                       T('📖  Documentación y recomendaciones',
                         '📖  Documentation & recommendations')],
              'sin puntos, la fila de arriba ya tiene las tres pestañas', str(etiq))
    comprobar(any(T('Todavía no hay puntos en la sesión',
                    'There are no points in the session yet') in str(i.value)
                  for i in at.info)
              and control(at, 'selectbox', 'mrf_nivel') is None,
              '«Mapa Modelo RF» explica qué hace falta, y nada más')

    print('  Z2 · una ejecución que se detiene a medias')
    resolver = sa.resolve
    sa.resolve = lambda *a, **k: (None, None)          # el área no se encuentra
    try:
        [r for r in at.radio if str(r.value).startswith('📁')][0].set_value(
            '🎲  Generar aleatoriamente en el área')
        [r for r in at.radio if str(r.value).startswith('⨯')][0].set_value(
            '🔤  Nombre del área protegida')
        at.run()
        [t for t in at.text_input if t.label == T('Nombre del área', 'Area name')
         ][0].set_value('un sitio que no existe')
        at.run()
        b = [b for b in at.button if b.label == T('▶ Ejecutar pipeline', '▶ Run pipeline')][0]
        comprobar(not b.disabled, 'con un nombre de área se puede lanzar la generación')
        b.click()
        at.run()
    finally:
        sa.resolve = resolver
    sin_error(at, 'Z2')
    comprobar(len(at.error) >= 1 and 'rows' not in at.session_state,
              'la ejecución se detiene con su aviso y sin puntos',
              str([str(e.value)[:60] for e in at.error]))
    comprobar(any(T('se ha detenido antes de terminar', 'stopped before finishing')
                  in str(i.value) for i in at.info),
              '«Mapa Modelo RF» no queda en blanco: remite al aviso',
              str([str(i.value)[:60] for i in at.info]))


for idioma in ('Español', 'English'):
    guion(idioma)
for idioma in ('Español', 'English'):
    guion_tabla(idioma)
for idioma in ('Español', 'English'):
    guion_sin_puntos(idioma)

print()
print('RESULTADO:', 'todo correcto' if not FALLOS else f'{len(FALLOS)} FALLO(S)')
for f_ in FALLOS:
    print('   -', f_)
sys.exit(1 if FALLOS else 0)
