"""Prueba de PANTALLA: revisión del tipo y etiquetas de campo, en la app real.

Abre la app directamente en la pantalla de resultados con 40 puntos reales ya
clasificados (más uno sin variables) y hace lo que haría una persona: seleccionar
filas, cambiar el tipo, guardar, deshacer, verificar y reentrenar, lanzar una
iteración de auto-entrenamiento y quitar puntos. Se repite en español y en inglés.

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
    tarjetas = [m.label for m in at.metric]
    comprobar(('LIKELY BORREGUIL' in tarjetas) if en else ('BORREGUIL PROBABLE' in tarjetas),
              'las tarjetas de recuento van en el idioma elegido', str(tarjetas))

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


for idioma in ('Español', 'English'):
    guion(idioma)

print()
print('RESULTADO:', 'todo correcto' if not FALLOS else f'{len(FALLOS)} FALLO(S)')
for f_ in FALLOS:
    print('   -', f_)
sys.exit(1 if FALLOS else 0)
