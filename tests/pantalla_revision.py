"""Prueba de PANTALLA: revisión del tipo y etiquetas de campo, en la app real.

Abre la app directamente en la pantalla de resultados con 40 puntos reales ya
clasificados (más uno sin variables) y hace lo que haría una persona: seleccionar
filas, cambiar el tipo, guardar, deshacer, verificar, reentrenar sumando sus puntos
y quitar puntos. Después recorre la pestaña «Mapa
Modelo RF» (muestras, tabla de entrenamiento, modelo y mapa) y, por último, abre la
app sin puntos. Se repite en español y en inglés.

    python tests/pantalla_revision.py        (unos 6 minutos, sin red)
    python tests/pantalla_revision.py mapa   (solo una parte: revision, tabla,
                                              mapa o sin_puntos)

Necesita `classification_v7.csv` (los puntos de entrenamiento, que NO están en el
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
DATOS = APP.parent / 'classification_v7.csv'     # con el radar de Earth Engine
if not DATOS.exists():
    print(f'Se omite: no está {DATOS} (los puntos de prueba no van en el repositorio).')
    sys.exit(0)
os.chdir(APP)
sys.path.insert(0, str(APP))
os.environ['BORREGUIL_DATA_DIR'] = tempfile.mkdtemp(prefix='bt_')
# sin buscar modelos en GitHub al abrir la app (modelos_remotos.py)
os.environ.setdefault('BORREGUIL_MODELOS_URL', '')
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
    # El píxel con el que se propone la pureza (como en el mapa; en la app se lee con
    # Earth Engine). Simulado: el NDVI del detector sale unos 0,12 más bajo que el del
    # píxel, y el agua, del NDWI medio.
    try:
        d['ndvi_early_pixel'] = float(r['ndvi_early']) + 0.12
        d['agua_pixel_vecinos'] = 1 if float(r['ndwi_mean']) > 0 else 0
    except (KeyError, TypeError, ValueError):
        pass
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


def N(v, en, nd=3, signo=False):
    """Un número como lo escribe la app: con coma decimal en español."""
    v = float(v)
    t = f'{v:+.{nd}f}' if signo else f'{v:.{nd}f}'
    return t if en else t.replace('.', ',')


def V(n, en):
    """El nombre legible de una variable, como lo escribe la app."""
    return (pp.FEATURE_LABELS_EN if en else pp.FEATURE_LABELS).get(n, n)


def poner_proyecto(at, valor):
    """El proyecto de Earth Engine se escribe en el panel lateral (desde el 24-11-2026
    la pestaña del mapa ya no lo pide)."""
    cs = [w for w in at.text_input if 'Earth Engine' in (w.label or '')]
    cs[0].set_value(valor)


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
    cualquiera de los dos idiomas), sin contar la Tabla de resultados: desde el 20-11
    sus cabeceras son legibles («Probabilidad»…) y se confundiría con las demás."""
    for d in at.dataframe:
        if d.key == 'results_table':
            continue
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


def grafico_del_peso(at):
    """Lo que la app manda dibujar en «5 · Peso»: (especificación del gráfico, tabla
    con sus datos). (None, None) si no está en pantalla."""
    from streamlit.dataframe_util import convert_arrow_bytes_to_pandas_df
    for g in at.get('vega_lite_chart'):
        for d in g.proto.datasets:
            df = convert_arrow_bytes_to_pandas_df(d.data.data)
            if 'variable' in df.columns and 'pierde' in df.columns:
                return json.loads(g.proto.spec), df
    return None, None


def grafico_del_umbral(at):
    """Lo que la app manda dibujar en «¿Borreguil o no?»: (especificación, tabla de
    las líneas TSS y F1, tabla con el umbral elegido). (None, None, None) si no está."""
    from streamlit.dataframe_util import convert_arrow_bytes_to_pandas_df
    for g in at.get('vega_lite_chart'):
        tablas = [convert_arrow_bytes_to_pandas_df(d.data.data) for d in g.proto.datasets]
        lineas = [t for t in tablas if 'medida' in t.columns]
        if lineas:
            elegido = [t for t in tablas if 'rotulo' in t.columns]
            return json.loads(g.proto.spec), lineas[0], (elegido[0] if elegido else None)
    return None, None, None


def peso_en_pantalla(at, mod, en, paso):
    """Comprueba que la pestaña enseña el peso de las variables de ese modelo: el
    gráfico, las tablas y la descarga dicen lo mismo que el modelo guardado."""
    T = (lambda es_, en_: en_ if en else es_)
    imp = mod['importancia']
    v, n = imp['variables'], len(imp['variables'])
    spec, datos = grafico_del_peso(at)
    comprobar(datos is not None and list(datos['variable']) == [V(x['variable'], en) for x in v]
              and all(abs(a - x['permutacion']) < 1e-12 for a, x in zip(datos['valor'], v)),
              f'{paso}: el gráfico tiene una barra por variable ({n}), de la que más pesa '
              'a la que menos', str(list(datos['variable'])[:4]) if datos is not None else 'no está')
    if spec is None:
        return
    capas = spec['layer']
    barras = [c for c in capas if c['mark'].get('type') == 'bar' and c['mark'].get('size') == 14]
    orden = [c['encoding']['y']['sort'] for c in capas if 'y' in c['encoding']]
    comprobar(len(barras) == 2 and len({b['mark']['color'] for b in barras}) == 1
              and all(o == [V(x['variable'], en) for x in v] for o in orden)
              and spec['height'] == 20 * n,
              f'{paso}: un solo color, las barras finas y en el orden del peso',
              str([b['mark'] for b in barras]))
    comprobar(any(c['mark'].get('type') == 'rule' and c['encoding'].get('x', {}).get('field')
                  == 'menos' and c['encoding'].get('x2', {}).get('field') == 'mas'
                  for c in capas)
              and all(abs(a - (x['permutacion'] - x['desviacion'])) < 1e-12
                      for a, x in zip(datos['menos'], v)),
              f'{paso}: cada barra lleva su «±»')
    ayuda = [c for c in capas if 'tooltip' in c['encoding']]
    comprobar(len(ayuda) == 1 and ayuda[0] is capas[-1]
              and ayuda[0]['encoding']['x2'] == {'value': 'width'}
              and [t['field'] for t in ayuda[0]['encoding']['tooltip']]
              == ['variable', 'que', 'grupo', 'pierde', 'uso', 'puesto']
              and all(datos['que']) and all(datos['grupo']),
              f'{paso}: al pasar el ratón por una fila dice qué mide y sus dos pesos',
              str(datos.iloc[0].to_dict()))
    comprobar(datos['pierde'][0] == (f"{N(v[0]['permutacion'], en, signo=True)} ± "
                                     f"{N(v[0]['desviacion'], en)}")
              and datos['uso'][0] == f"{N(100 * v[0]['impureza'], en)} %",
              f'{paso}: con tres decimales', f"{datos['pierde'][0]} | {datos['uso'][0]}")
    tabla = tabla_con_columna(at, 'Bajada del F1 sin ella', 'F1 drop without it')
    comprobar(tabla is not None and len(tabla) == n
              and list(tabla.columns[:6]) == [
                  'Variable', T('Puesto', 'Rank'),
                  T('Bajada del F1 sin ella', 'F1 drop without it'), '±',
                  T('Uso interno (%)', 'Internal use (%)'), T('Puesto por uso', 'Rank by use')]
              and list(tabla['Variable']) == [V(x['variable'], en) for x in v]
              and list(tabla.iloc[:, 1]) == list(range(1, n + 1))
              and all(abs(a - x['permutacion']) < 1e-12 for a, x in zip(tabla.iloc[:, 2], v))
              and abs(float(tabla.iloc[:, 4].sum()) - 100) < 1e-6
              and sorted(tabla.iloc[:, 5]) == list(range(1, n + 1))
              and all(tabla.iloc[:, 6]) and all(tabla.iloc[:, 8]),
              f'{paso}: la tabla da las dos medidas y el puesto de cada variable, con los '
              'números delante', str(list(tabla.columns)) if tabla is not None else 'no está')
    fijas = {t.value.index.name: t.value for t in at.table}
    for clave, nombre, lista, campo, rotulo in (
            ('familias', T('Según lo que miden', 'By what they measure'), imp['familias'],
             'familia', mr.nombre_familia),
            ('grupos', T('Según la época o el origen', 'By season or source'), imp['grupos'],
             'grupo', mr.nombre_grupo)):
        t = fijas.get(nombre)
        comprobar(t is not None and list(t.index) == [rotulo(g[campo], en) for g in lista]
                  and sum(int(c) for c in t.iloc[:, 0]) == n
                  and list(t.iloc[:, 1]) == [
                      f"{N(g['permutacion'], en, signo=True)}\u00a0±\u00a0"
                      f"{N(g['desviacion'], en)}" for g in lista]
                  and list(t.iloc[:, 2]) == [f"{N(100 * g['impureza'], en)}\u00a0%"
                                             for g in lista],
                  f'{paso}: y el peso de las variables juntas, por {clave} ({len(lista)}), en '
                  'una tabla que no corta los nombres',
                  str(t.reset_index().values.tolist()[:2]) if t is not None else str(list(fijas)))
    desc = [d for d in at.get('download_button') if d.proto.label.startswith('⬇ peso_variables_')]
    comprobar(len(desc) == 1 and desc[0].proto.label == f"⬇ peso_variables_{mod['nivel']}.csv",
              f'{paso}: se puede descargar en CSV', str([d.proto.label for d in desc]))
    comprobar(hay(at, T('que es **', 'which is **') + N(mod['macro_f1'], en) + '**')
              and hay(at, T(f"Se hace {imp['repeticiones']} veces",
                            f"This is done {imp['repeticiones']} times")),
              f'{paso}: explica las dos medidas con el F1 y las repeticiones de este modelo')
    claras = sum(1 for x in v if x['permutacion'] > 0
                 and x['permutacion'] > 2 * x['desviacion'])
    comprobar((hay(at, T(f'**{claras} de las {n} variables**',
                         f'**{claras} of the {n} variables**')) if claras else
               hay(at, T(f'**ninguna de las {n} variables**', f'**none of the {n} variables**')))
              and (hay(at, T(f"**La que más pesa es «{V(v[0]['variable'], en)}»**",
                             f"**The one that weighs most is “{V(v[0]['variable'], en)}”**"))
                   if v[0]['permutacion'] > 0 else
                   hay(at, T('**Ninguna variable hace bajar el F1',
                             '**No variable lowers the F1'))),
              f'{paso}: resume en una frase cuál pesa más y cuántas pesan con claridad')
    comprobar(hay(at, T('5 · Peso · la primera: ', '5 · Weights · top: ')
                  + V(v[0]['variable'], en)) if v[0]['permutacion'] > 0 else
              hay(at, T('5 · Peso · ninguna destaca', '5 · Weights · none stands out')),
              f'{paso}: estado: el peso está calculado')


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
    col_rev = 'Type reviewed' if en else 'Tipo revisado'
    comprobar(df is not None and col_rev in df.columns,
              f'la tabla tiene la columna «{col_rev}»',
              str(list(df.columns)) if df is not None else 'sin tabla')
    n_b = len(BORR) + 1
    comprobar(hay(at, f'Type reviewed: 0 of {n_b}' if en else f'Tipo revisado: 0 de {n_b}'),
              f'contador a cero sobre {n_b} borreguiles')
    comprobar(boton(at, 'rev_save') is None, 'sin selección no aparece el panel de revisión')
    T = (lambda es_, en_: en_ if en else es_)
    # Por defecto, el modelo de partida, y la pantalla lo dice
    comprobar(hay(at, T('Modelo usado: el de partida', 'Model used: the starting model'))
              and at.checkbox(key='entrenar_propio').value is False,
              'por defecto se usa el modelo de partida (casilla del modelo propio sin marcar)')
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
    # Pestañas de resultados: nombres y orden de la autora (24-11), Descargas al final.
    _res = [T('Mapa de los puntos', 'Points map'), T('Revisar puntos', 'Review points'),
            T('Probabilidades', 'Probabilities'),
            T('Peso de las variables', 'Variable weights'),
            T('Tipos y distribuciones', 'Types and distributions'),
            T('Descargas', 'Downloads')]
    _i = etiq.index(_res[0]) if _res[0] in etiq else -1
    comprobar(_i >= 0 and etiq[_i:_i + 6] == _res,
              'las pestañas de resultados se llaman como pidió la autora y en su orden',
              str(etiq))
    pasos = [T('1 · Datos', '1 · Data'),
             T('2 · Tabla de entrenamiento', '2 · Training table'),
             T('3 · Modelo', '3 · Model'), T('4 · Mapa', '4 · Map'),
             T('5 · Peso', '5 · Weights')]
    comprobar(all(p in etiq for p in pasos) and len(etiq) == 3 + 6 + 5
              and [e for e in etiq if e in pasos] == pasos,
              'sus cinco pasos son pestañas, en orden, y los resultados tienen seis',
              str(etiq))
    comprobar(hay(at, T('**🌿 Mapa Modelo RF**, arriba del todo',
                        '**🌿 RF model map** tab, at the very top')),
              'la pestaña «Tabla» dice dónde sigue lo que se revisa')
    comprobar(all(hay(at, x) for x in (
        T('1 · Datos · faltan muestras', '1 · Data · not enough samples'),
        T('2 · Tabla · sin leer', '2 · Table · not read yet'),
        T('3 · Modelo · sin entrenar', '3 · Model · not trained yet'),
        T('4 · Mapa · sin hacer', '4 · Map · not made yet'))),
        'la línea de estado dice por dónde va cada paso')
    # Sin nada marcado como borreguil en campo solo cuenta la ausencia de campo, y con
    # eso no se puede entrenar. (Viene elegido el primer paso, «borreguil sí o no».)
    comprobar(clases_del_mapeo(at) == {T('no borreguil', 'not a borreguil'): 1,
                                       'borreguil': 0},
              'mapeo: recuento por categoría antes de revisar nada',
              str(clases_del_mapeo(at)))
    # ¿Son bastantes puntos? (curva de Sierra Nevada, 24-11): con 1 muestra no hay
    # referencia, y dice cuánto falta para el mínimo; la Documentación explica la curva.
    comprobar(hay(at, T('**¿Son bastantes puntos?**', '**Are they enough points?**'))
              and hay(at, T('Con menos de 20 no hay referencia',
                            'Below 20 there is no reference'))
              and hay(at, T('(0 borreguiles y 1 ausencia)', '(0 borreguiles and 1 absence)'))
              and hay(at, T('Para el mínimo (100 puntos: 33 borreguiles y 67 ausencias), '
                            'te faltan 33 borreguiles y 66 ausencias.',
                            'For the minimum (100 points: 33 borreguiles and 67 absences), '
                            'you need 33 borreguiles and 66 absences more.'))
              and hay(at, T('Las ausencias son los sitios que has comprobado que **no** son '
                            'borreguil', 'Absences are the places you have checked are '
                            '**not** a borreguil')),
              'mapeo: dice si son bastantes puntos y cuántos borreguiles y cuántas ausencias '
              'faltan para el mínimo, y cómo se ponen las ausencias (25-11)')
    cuentas = tabla_con_columna(at, 'Tienes', 'You have')
    comprobar(cuentas is not None and len(cuentas) == 3
              and list(cuentas.iloc[:, 1]) == ['1', '0', '1']
              and list(cuentas.iloc[:, 2]) == [T('100 (faltan 99)', '100 (99 to go)'),
                                               T('33 (faltan 33)', '33 (33 to go)'),
                                               T('67 (faltan 66)', '67 (66 to go)')]
              and list(cuentas.iloc[:, 3])[0] == T('250 (faltan 249)', '250 (249 to go)')
              and T('· ausencias (no borreguil)', '· absences (not a borreguil)')
              in list(cuentas.iloc[:, 0])
              and hay(at, T('El mínimo de la gráfica, 100 puntos, es la suma de 33 '
                            'borreguiles y 67 ausencias',
                            'The minimum in the chart, 100 points, is the sum of 33 '
                            'borreguiles and 67 absences')),
              'tabla siempre a la vista: el total de la gráfica y, debajo, borreguiles y '
              'ausencias, con el mínimo, lo recomendado y lo que falta; la frase dice que '
              '100 = 33 + 67', str(cuentas.values.tolist())
              if cuentas is not None else 'no está')
    comprobar(hay(at, T('¿Cuántos puntos necesito para el mapa?',
                        'How many points do I need for the map?'))
              and hay(at, T('Mínimo: unos 100 puntos revisados, con unos 33 borreguiles y '
                            '67 ausencias',
                            'Minimum: about 100 reviewed points, with about 33 '
                            'borreguiles and 67 absences')),
              'la Documentación explica cuántos puntos hacen falta')
    comprobar(hay(at, T('Sugerir un cambio o informar de un error',
                        'Suggest a change or report an error'))
              and hay(at, 'github.com/Bea-Cazorla/borreguil-detector/issues/new/choose?version=')
              and hay(at, 'github.com/Bea-Cazorla/borreguil-detector/wiki'),
              'manual (wiki) y sugerencias: el enlace a GitHub lleva la versión')
    # El formulario anónimo propio de esta app (25-11-2026), con la versión ya puesta
    comprobar(hay(at, 'docs.google.com/forms/d/e/'
                      '1FAIpQLScq7UehnDlaNaYsq9DZNhXc-CxRo57KgdT9e98U0q1xGglRPg/viewform?')
              and hay(at, 'entry.666897268=')
              and hay(at, T('Formulario anónimo', 'Anonymous form')),
              'y el formulario anónimo de la app, con la versión ya puesta')
    comprobar(any(T('aprende **solo si hay borreguil**',
                    'learns **only whether there is a borreguil**') in str(i.value)
                  for i in at.info),
              'y dice que en ese nivel el modelo aprende solo si hay borreguil, con lo '
              'marcado en campo')
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
    comprobar(('Likely borreguil' in tarjetas) if en else ('Borreguil probable' in tarjetas),
              'las tarjetas de recuento van en el idioma elegido y como texto normal, no '
              'en mayúsculas', str(tarjetas))
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
        # sin revisar, arranca en «(sin cambio)» (guardar no lo da por revisado sin
        # mirarlo) y la propuesta de la app se lee en la etiqueta
        comprobar(d is not None and d.value == '(sin cambio)'
                  and i18n_valor(r1[niv], en) in d.label,
                  f'el desplegable de {niv} arranca sin revisar y enseña la propuesta '
                  f'({r1[niv]})', f'{d.value!r} · {d.label!r}' if d is not None else 'no existe')
    comprobar(hay(at, 'Orthophoto background' if en else 'Ortofoto de fondo'),
              'el mapa del punto anuncia la ortofoto y el píxel de 10 m')
    comprobar(hay(at, "app's proposal, not reviewed" if en else
                  'propuesta de la app, sin revisar'),
              'la ficha del punto dice que el tipo está sin revisar')
    comprobar(hay(at, bp.nombre_html(r1['ID'])),
              'la ficha muestra el nombre completo, con cortes en los separadores')
    _alt = N(r1.get('elev_dem_m'), en)
    comprobar(hay(at, f"**{_alt}**") and len(_alt.split('.' if en else ',')[1]) == 3
              and not hay(at, str(r1.get('elev_dem_m'))),
              'los números de la ficha van con 3 decimales', _alt)
    comprobar(hay(at, T('Lo que más pesa en el modelo:', 'What weighs most in the model:')),
              'la ficha dice el valor de las variables que más pesan')
    comprobar(hay(at, T('Este punto: dentro de lo habitual en los borreguiles',
                        'This point: within the usual for borreguiles'))
              and hay(at, T('Cada punto de color es **el punto que estás mirando**',
                            'Each coloured dot is **the point you are looking at**')),
              'la gráfica de la ficha lleva su leyenda de colores')
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
    comprobar(bp.niveles_revisados(r) == ['humedad'],
              'el punto queda revisado solo en la humedad', str(bp.niveles_revisados(r)))
    comprobar(r['humedad'] == nueva_hum and r['humedad_regla'] == r1['humedad'],
              'humedad corregida y propuesta de la app conservada',
              f"{r.get('humedad')} / {r.get('humedad_regla')}")
    comprobar(r['ambiente'] == r1['ambiente'] and r['pureza'] == r1['pureza'],
              'los niveles no tocados siguen con la propuesta, sin revisar')
    comprobar(bool(r.get('tipo_revisado_fecha')) and r.get('method') == 'orthophoto',
              'queda la fecha de la revisión, y que se hizo con la ortofoto')
    comprobar(any(('Reviewed type saved for 1 point(s): moisture.' if en else
                   'Tipo revisado guardado en 1 punto(s): humedad.') in m for m in msgs),
              'mensaje de confirmación, con el nivel revisado')
    comprobar(hay(at, f'Type reviewed: 1 of {n_b}' if en else f'Tipo revisado: 1 de {n_b}'),
              'el contador pasa a 1 revisado')
    comprobar(hay(at, 'moisture 1' if en else 'humedad 1'),
              'el contador registra 1 corrección de humedad')
    df = tabla_principal(at)
    comprobar(df is not None and df[col_rev].iloc[i1] == ('moisture' if en else 'humedad')
              and (df[col_rev] != '').sum() == 1,
              'la tabla marca solo ese punto, con el nivel revisado')
    for nombre in ('classification.csv', 'puntos.geojson', 'Clasificacion_puntos.xlsx',
                   'mapa.html'):
        comprobar((work / nombre).exists(), f'se ha guardado {nombre}')
    gj = json.loads((work / 'puntos.geojson').read_text(encoding='utf-8'))
    props = {f['properties']['ID']: f['properties'] for f in gj['features']}
    comprobar(props[r['ID']]['moisture'] == bp.TIPO_EN[nueva_hum]
              and props[r['ID']]['environment'] is None
              and props[r['ID']]['method'] == 'orthophoto',
              'puntos.geojson lleva la revisión con las columnas de la plantilla, solo el '
              'nivel revisado')
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
    comprobar(d is not None and d.value == '(sin cambio)',
              'tras deshacer, el desplegable vuelve a «(sin cambio)»',
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
        comprobar(bp.niveles_revisados(r) == ['pureza'] and r['pureza'] == 'mixto-roca'
                  and r['ambiente'] == antes[i]['ambiente']
                  and r['humedad'] == antes[i]['humedad'],
                  f"{r['ID']}: pureza revisada; ambiente y humedad, sin tocar")
    r = filas(at)[j]
    comprobar(not bp.tipo_revisado(r) and r.get('ambiente', '') == '',
              'el no borreguil seleccionado no se toca')
    # --- E2 · completar uno a uno el ambiente y la humedad (dando por buena la
    # propuesta, eligiéndola): el nivel revisado antes se conserva
    print('  E2 · completar los otros niveles de cada punto')
    for i in (i1, i2):
        seleccionar(at, [i])
        sin_error(at, 'E2 (selección)')
        comprobar(desplegable(at, 'pureza').value == 'mixto-roca'
                  and desplegable(at, 'humedad').value == '(sin cambio)',
                  f"{filas(at)[i]['ID']}: arranca con lo revisado y el resto sin revisar")
        desplegable(at, 'ambiente').select(antes[i]['ambiente'])
        desplegable(at, 'humedad').select(antes[i]['humedad'])
        pulsar(at, boton(at, 'rev_save'), 'E2 (guardar)')
        r = filas(at)[i]
        comprobar(bp.niveles_revisados(r) == list(bp.NIVELES_TIPO)
                  and (r['ambiente'], r['humedad'], r['pureza'])
                  == (antes[i]['ambiente'], antes[i]['humedad'], 'mixto-roca'),
                  f"{r['ID']}: los tres niveles revisados, sin perder la pureza")

    # --- F · solo no borreguiles
    print('  F · selección sin borreguiles')
    seleccionar(at, NOB[:2])
    sin_error(at, 'F')
    comprobar(hay(at, 'The selected points are not classified as borreguil' if en else
                  'Los puntos seleccionados no están clasificados como borreguil'),
              'explica que esos puntos no tienen tipo')
    comprobar(boton(at, 'rev_save') is None, 'no ofrece guardar')

    # --- G · sin elegir nada no se guarda; un punto sin datos se revisa por niveles
    print('  G · guardar sin elegir nada, y un punto sin datos')
    i3 = BORR[2]
    seleccionar(at, [i3, SIN])
    sin_error(at, 'G (selección)')
    boton(at, 'rev_save').click()       # sin st.rerun(): el aviso se queda en pantalla
    at.run()
    sin_error(at, 'G (guardar sin elegir)')
    comprobar(hay(at, 'You have not chosen any level' if en else
                  'No has elegido ningún nivel'),
              'sin elegir ningún nivel, avisa de que no hay nada que guardar')
    comprobar(not bp.tipo_revisado(filas(at)[i3]) and not bp.tipo_revisado(filas(at)[SIN]),
              'ninguno de los dos queda revisado')
    desplegable(at, 'humedad').select('seco')
    pulsar(at, boton(at, 'rev_save'), 'G (guardar la humedad)')
    comprobar(all(bp.niveles_revisados(filas(at)[k]) == ['humedad']
                  and filas(at)[k]['humedad'] == 'seco' for k in (i3, SIN)),
              'la humedad se guarda en los dos, también en el que no tiene datos')
    for niv, val in (('ambiente', 'ladera'), ('humedad', 'seco'), ('pureza', 'puro')):
        desplegable(at, niv).select(val)
    pulsar(at, boton(at, 'rev_save'), 'G (guardar eligiendo los tres)')
    r = filas(at)[SIN]
    comprobar(bp.niveles_revisados(r) == list(bp.NIVELES_TIPO)
              and (r['ambiente'], r['humedad'], r['pureza']) == ('ladera', 'seco', 'puro'),
              'eligiendo los tres niveles quedan los tres revisados')

    # --- G2 · la pestaña «Mapa Modelo RF» cuenta lo revisado
    print('  G2 · mapeo: muestras por categoría')
    comprobar(control(at, 'selectbox', 'mrf_nivel').value == mr.NIVEL_SI_NO,
              'viene elegido el primer paso: borreguil sí o no',
              str(control(at, 'selectbox', 'mrf_nivel').value))
    # sí o no: cuenta lo dicho en campo, tenga o no el tipo revisado
    esperado = mr.muestras(filas(at), mr.NIVEL_SI_NO, es_borreguil=pp.is_borreguil_decision)
    comprobar(clases_del_mapeo(at) == {T(c['categoria'], i18n_valor(c['categoria'], True)): c['n']
                                       for c in esperado['clases']}
              and control(at, 'radio', 'mrf_quien_decide') is None,
              'sí o no: cuenta los puntos de campo, sin preguntar quién decide',
              str(clases_del_mapeo(at)))
    opciones = list(control(at, 'selectbox', 'mrf_nivel').options)
    comprobar(len(opciones) == 3
              and not any(o.startswith(('Ambiente (', 'Pureza (', 'Environment (', 'Purity ('))
                          for o in opciones),
              'el desplegable ofrece sí o no, humedad, y ambiente y humedad (sin «Ambiente» '
              'ni «Pureza» solos)', str(opciones))
    control(at, 'selectbox', 'mrf_nivel').select('humedad')
    at.run()
    sin_error(at, 'G2')
    quien = control(at, 'radio', 'mrf_quien_decide')
    comprobar(quien is not None and quien.value == 'si_no',
              'en un nivel del tipo, de entrada el sí o no lo decide el modelo de sí o no',
              str(quien.value if quien else None))
    # Lo que sigue prueba el modelo que aprende también «no borreguil».
    quien.set_value('este')
    at.run()
    # Las muestras de la humedad revisada, más la ausencia de campo; los dudosos no cuentan.
    esperado = mr.muestras(filas(at), 'humedad', es_borreguil=pp.is_borreguil_decision)
    n_us = esperado['n_usadas']
    comprobar(clases_del_mapeo(at) == {T(c['categoria'], i18n_valor(c['categoria'], True)): c['n']
                                       for c in esperado['clases']}
              and T('no borreguil', 'not a borreguil') in clases_del_mapeo(at) and n_us >= 3,
              'el recuento sigue a lo que se ha revisado en la tabla',
              str(clases_del_mapeo(at)))
    comprobar(hay(at, T(f'**{n_us} muestras** de los 41 puntos',
                        f'**{n_us} samples** out of the 41 points')),
              f'{n_us} muestras de 41 puntos')
    motivos = tabla_con_columna(at, 'Motivo', 'Reason')
    mot = dict(zip(motivos.iloc[:, 0], motivos.iloc[:, 1])) if motivos is not None else {}
    comprobar(mot.get(mr.motivo('dudoso', en)) == 2
              and mot == {mr.motivo(k, en): v for k, v in esperado['excluidos'].items()},
              'explica por qué no entra el resto (dudosos, tipo sin revisar…)', str(mot))
    comprobar(sum(mot.values()) + n_us == 41, 'muestras y descartes suman todos los puntos')
    # «Ambiente y humedad»: el modelo aprende la humedad, así que las muestras se
    # cuentan como en «Humedad»; el ambiente lo pondrá el mapa por cercanía.
    control(at, 'selectbox', 'mrf_nivel').select('ambiente_humedad')
    at.run()
    sin_error(at, 'G2 (ambiente y humedad)')
    esperado = mr.muestras(filas(at), 'humedad', es_borreguil=pp.is_borreguil_decision)
    vistas = clases_del_mapeo(at)
    comprobar(vistas == {T(c['categoria'], i18n_valor(c['categoria'], True)): c['n']
                         for c in esperado['clases']}
              and list(vistas) == [T('no borreguil', 'not a borreguil'), T('húmedo', 'wet'),
                                   T('seco', 'dry')] and sum(vistas.values()) == 5,
              'ambiente y humedad: se cuentan húmedo, seco y la ausencia, como en «Humedad»',
              str(vistas))
    comprobar(any(T('el modelo aprende **húmedo o seco**', 'the model learns **wet or dry**')
                  in str(i.value) for i in at.info),
              'y se explica que el ambiente no sale del modelo')
    control(at, 'selectbox', 'mrf_nivel').select('ambiente_humedad')
    at.run()
    comprobar(hay(at, T('1 · Datos · faltan muestras', '1 · Data · not enough samples')),
              'y, con tan pocas, la línea de estado dice que faltan muestras')
    control(at, 'selectbox', 'mrf_nivel').select('humedad')
    at.run()
    control(at, 'radio', 'mrf_quien_decide').set_value('si_no')
    at.run()
    sin_error(at, 'G2 (sin ausencias)')
    comprobar(T('no borreguil', 'not a borreguil') not in clases_del_mapeo(at)
              and any(T('aprende **solo el tipo**', 'learns **only the type**') in str(i.value)
                      and T('entrénalo también', 'train it too') in str(i.value)
                      for i in at.info),
              'si decide el modelo de sí o no, este aprende solo el tipo, y pide entrenar el otro')
    control(at, 'radio', 'mrf_quien_decide').set_value('este')
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
    control(at, 'selectbox', 'mrf_nivel').select('humedad')
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
    comprobar(not any(p in boton(at, 'mark_verified').label for p in ('reentrenar', 'retrain')),
              'sin la casilla del modelo propio, verificar no promete reentrenar',
              boton(at, 'mark_verified').label)
    # quién revisa, su perfil y la certidumbre se apuntan en los puntos verificados
    control(at, 'text_input', 'revisor_nombre').input('Ana')
    control(at, 'selectbox', 'revisor_perfil').set_value('researcher')
    control(at, 'radio', 'rev_certidumbre').set_value('medium')
    msgs_h = pulsar(at, boton(at, 'mark_verified'), 'H')
    _ids = {filas(at)[i].get('ID') for i in (i1, k, D2)}
    _ver = [r for r in filas(at) if r.get('ID') in _ids]
    comprobar(len(_ver) == 3 and all((r.get('observer'), r.get('expertise'),
                                      r.get('certainty')) == ('Ana', 'researcher', 'medium')
                                     for r in _ver),
              'los puntos verificados guardan quién revisa, su perfil y la certidumbre',
              str([(r.get('ID'), r.get('observer'), r.get('expertise'), r.get('certainty'))
                   for r in _ver]))
    comprobar(T('Acierto según la certidumbre y el perfil de quien revisa',
                'Accuracy by the certainty and profile of the reviewer')
              in [e.label for e in at.expander],
              'el acierto se separa según la certidumbre y el perfil de quien revisa')
    comprobar(any(T('3 puntos marcados como borreguil verificado',
                    '3 points marked as verified borreguil') in m for m in msgs_h)
              and not any(T('Modelo reentrenado', 'Model retrained') in m for m in msgs_h),
              'al verificar se ve un aviso con los puntos, sin decir que ha reentrenado '
              '(sin la casilla no reentrena)',
              str([m[:90] for m in msgs_h if 'erifi' in m][:3]))
    df = tabla_principal(at)
    cab = list(df.columns) if df is not None else []
    comprobar(all(c in cab for c in (T('Decisión', 'Decision'),
                                     T('Probabilidad', 'Probability'),
                                     T('Tipo revisado', 'Type reviewed'),
                                     T('Altitud (m)', 'Elevation (m)'),
                                     T('Pendiente (°)', 'Slope (°)')))
              and not any(c in cab for c in ('decision', 'rf_proba', 'mat_signature', 'twi',
                                             'ndvi_late', 'clre_late', 'ndmi_late',
                                             'evi_late', 'elev_dem_m', 'slope_deg')),
              'la tabla tiene cabeceras legibles y las variables que más pesan en el '
              'modelo (no las 4 de fin de verano, el TWI ni el patrón)', str(cab))
    _col_b = T('Borreguil (tu etiqueta)', 'Borreguil (your label)')
    comprobar(_col_b in cab and set(df[_col_b]) <= {T('sí', 'yes'), 'no', ''}
              and '' in set(df[_col_b]),
              'la columna de la etiqueta dice sí, no o nada (no «None»)',
              str(sorted(set(map(str, df[_col_b])))) if _col_b in cab else str(cab))
    comprobar(hay(at, T('Modelo usado: el de partida', 'Model used: the starting model'))
              and hay(at, T('Borreguiles de campo acertados', 'Field borreguiles right')),
              'tras verificar sigue el modelo de partida, y mide su acierto con lo verificado')
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

    # --- I · reentrenar sumando los puntos (la casilla, con los resultados en pantalla)
    print('  I · marcar la casilla de sumar y reentrenar')
    _recursiva = [e.label for e in at.expander
                  if 'recursiv' in e.label.lower() or 'self-training' in e.label.lower()]
    comprobar(not _recursiva and not hay(at, T('auto-entrenamiento', 'self-training')),
              'ya no hay auto-entrenamiento (evaluación recursiva)', str(_recursiva))
    comprobar(boton(at, 'aplicar_casilla') is None,
              'sin la casilla de sumar no se ofrece reentrenar')
    _guardar = T('💾 Guardar el modelo reentrenado', '💾 Save the retrained model')
    comprobar(_guardar not in [e.label for e in at.expander]
              and not hay(at, 'local + git'),
              'con el modelo de partida no se ofrece guardarlo (ya está en el desplegable)')
    comprobar(hay(at, T('Excel para **mirar los resultados**',
                        'Excel to **look at the results**')),
              'la pestaña Descargas explica para qué sirve cada fichero')
    # se marca la casilla: aparece sin volver a ejecutar
    at.checkbox(key='entrenar_propio').check()
    at.run()
    sincroniza(at)
    sin_error(at, 'I')
    _ap = boton(at, 'aplicar_casilla')
    comprobar(_ap is not None and any(p in _ap.label for p in ('Reentrenar ahora',
                                                               'Retrain now'))
              and hay(at, T('pulsa el botón de abajo', 'press the button below')),
              'al marcar la casilla con los resultados en pantalla, se ofrece reentrenar '
              'en el momento', _ap.label if _ap is not None else 'sin botón')
    if _ap is not None:
        pulsar(at, _ap, 'I')
        inf = at.session_state['iter_history'][-1].get('info') or {}
        comprobar(inf.get('sumado') and inf.get('n_tabla', 0) > 700
                  and hay(at, T('reentrenado sumando tus puntos',
                                'retrained adding your points')),
                  'reentrena UN modelo: la tabla de entrenamiento más tus puntos',
                  str({k: inf.get(k) for k in ('mode', 'sumado', 'n_tabla', 'n_pos')}))
        comprobar(boton(at, 'aplicar_casilla') is None,
                  'ya reentrenado sumando, no se ofrece reentrenar otra vez')
        comprobar(_guardar in [e.label for e in at.expander]
                  and not hay(at, 'local + git')
                  and not hay(at, T('publicar el modelo', 'publish the model')),
                  'tras reentrenar se ofrece guardar el modelo, sin git ni publicación online',
                  str([e.label for e in at.expander]))
        despues = {r['ID']: (r['ambiente'], r['humedad'], r['pureza'])
                   for r in filas(at) if bp.tipo_revisado(r)}
        comprobar(despues == revisados, 'los tipos revisados sobreviven al reentrenar')
        sin_tipo = [r['ID'] for r in filas(at)
                    if pp.is_borreguil_decision(r['decision']) and r['ID'] != 'sin_datos'
                    and r.get('ambiente') not in bp.NIVELES_TIPO['ambiente']]
        comprobar(not sin_tipo, 'todos los borreguiles tras reentrenar tienen tipo',
                  str(sin_tipo[:5]))
        r = filas(at)[NO_CAMPO]
        comprobar(r['Borreguil'] == 'no' and r['decision'] == 'NO BORREGUIL VERIFICADO',
                  f"la ausencia de campo (prob. {base[NO_CAMPO]['rf_proba']:.3f}) no se "
                  'borra al reentrenar', f"{r['Borreguil']!r} / {r['decision']}")
        r = filas(at)[D1]
        comprobar(r['Duda'] == 'si' and bp._txt(r.get('Borreguil')) == ''
                  and r['decision'] == 'DUDOSO (campo)',
                  'el punto dudoso sigue dudoso tras reentrenar',
                  f"{r.get('Duda')!r} / {r['decision']}")
        comprobar(not any('auto' in r['decision'].lower() for r in filas(at)),
                  'reentrenar no convierte ningún punto en borreguil por su cuenta')
    else:
        comprobar(False, 'al marcar la casilla se ofrece reentrenar')

    # --- J · quitar un punto no revisado
    print('  J · quitar puntos del análisis')
    n0 = len(filas(at))
    quitar = NOB[2]
    id_quitado = filas(at)[quitar]['ID']
    seleccionar(at, [quitar])
    msgs_j = pulsar(at, boton(at, 'remove_pts'), 'J')
    comprobar(any(f"{id_quitado} {T('quitado del análisis', 'removed from the analysis')}"
                  in m for m in msgs_j),
              'al quitar un punto se ve un aviso con su nombre',
              str([m[:90] for m in msgs_j if 'quit' in m or 'remov' in m][:3]))
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
    bp.leer_etiquetas(otra)                 # lo que hace la app al cargarlo
    comprobar(len(otra) == n0 - 1, 'el fichero tiene los mismos puntos que la pantalla',
              str(len(otra)))
    # vuelve cada nivel revisado, con su valor (los no revisados no viajan: son la
    # propuesta de la app, que se recalcula)
    vuelta = {r['ID']: {n: r[n] for n in bp.niveles_revisados(r)}
              for r in otra if bp.tipo_revisado(r)}
    esperado_k = {r['ID']: {n: r[n] for n in bp.niveles_revisados(r)}
                  for r in filas(at) if bp.tipo_revisado(r)}
    comprobar(vuelta == esperado_k, 'al recargarlo vuelven los tipos revisados, nivel a nivel',
              f'{len(vuelta)} revisados')
    comprobar(not any('ndmi_mean' in r for r in otra),
              'el fichero no arrastra variables predictoras')
    por_id = {r['ID']: r for r in otra}
    comprobar(por_id[base[NO_CAMPO]['ID']].get('Borreguil') == 'no'
              and por_id[base[D1]['ID']].get('Duda') == 'si',
              'el fichero conserva la ausencia y la duda de campo')
    comprobar(all(por_id[r['ID']].get('Borreguil') == 'si' for r in filas(at)
                  if r['decision'] == 'BORREGUIL VERIFICADO')
              and all(bp._txt(por_id[r['ID']].get('Borreguil')) == '' for r in filas(at)
                      if bp._txt(r.get('Borreguil')) == ''),
              'en el fichero es verdad-terreno lo verificado, y solo eso')

    # --- M · un punto mirado que NO es borreguil (uno al azar, por ejemplo) pasa a
    # ser ausencia verificada, como las de campo
    print('  M · marcar como no borreguil verificado y reentrenar')
    ids = [r['ID'] for r in filas(at)]
    # uno sin etiqueta de campo ni tipo revisado; mejor si el modelo lo da por borreguil
    sin_etiqueta = [r for r in filas(at) if bp._txt(r.get('Borreguil')) == ''
                    and bp._txt(r.get('Duda')) != 'si' and not bp.tipo_revisado(r)]
    sin_mirar = next((r['ID'] for r in sin_etiqueta if pp.is_borreguil_decision(r['decision'])),
                     sin_etiqueta[0]['ID'])
    con_tipo = next(r['ID'] for r in filas(at) if bp.tipo_revisado(r))
    seleccionar(at, [ids.index(sin_mirar), ids.index(con_tipo)])
    comprobar(boton(at, 'mark_absent') is not None and boton(at, 'mark_verified') is not None
              and boton(at, 'remove_pts') is not None,
              'con puntos seleccionados está también el botón de marcarlos como no borreguil, '
              'además de los que ya había')
    msgs_m = pulsar(at, boton(at, 'mark_absent'), 'M')
    comprobar(any(T('2 puntos marcados como no borreguil verificado',
                    '2 points marked as verified non-borreguil') in m
                  and T('Modelo reentrenado sumando tus puntos',
                        'Model retrained adding your points') in m for m in msgs_m),
              'al marcar como no borreguil con la casilla, el aviso dice que ha reentrenado',
              str([m[:120] for m in msgs_m if 'erifi' in m][:3]))
    por_id = {r['ID']: r for r in filas(at)}
    comprobar(all(por_id[i]['Borreguil'] == 'no'
                  and por_id[i]['decision'] == 'NO BORREGUIL VERIFICADO'
                  and por_id[i]['origin'] == 'verificado' for i in (sin_mirar, con_tipo)),
              'los dos quedan como ausencias verificadas, aunque el modelo dijera borreguil',
              str([(por_id[i]['Borreguil'], por_id[i]['decision']) for i in (sin_mirar, con_tipo)]))
    comprobar(not bp.tipo_revisado(por_id[con_tipo]),
              'al que tenía un tipo revisado se le quita: un no borreguil no tiene tipo')
    comprobar({r['ID'] for r in filas(at) if bp.tipo_revisado(r)} == set(revisados) - {con_tipo},
              'los demás revisados siguen intactos')
    res = mr.muestras(filas(at), 'humedad', incluir_ausencias=True,
                      es_borreguil=pp.is_borreguil_decision)
    como = {f['ID']: f['categoria'] for f in res['tabla'] if f['usada']}
    comprobar(como.get(sin_mirar) == mr.AUSENCIA and como.get(con_tipo) == mr.AUSENCIA,
              'y entran en el mapa de tipos como ausencias, igual que las de campo')
    guardado = {r['ID']: r for r in bp.read_points(str(work / 'puntos.geojson'))}
    # (con las columnas de la plantilla: presence = 0). Un hueco vuelve como None sin
    # GDAL y como NaN con GDAL: los dos son «sin tipo».
    comprobar(guardado[sin_mirar].get('presence') == 0
              and guardado[con_tipo].get('presence') == 0
              and bp._txt(guardado[con_tipo].get('moisture')) == '',
              'y quedan guardadas en puntos.geojson, sin tipo',
              repr(guardado[con_tipo].get('moisture')))

    # --- N · desmarcar la casilla: volver al modelo de partida (y otra vez sumar)
    print('  N · desmarcar y volver a marcar la casilla de sumar')
    at.checkbox(key='entrenar_propio').uncheck()
    at.run()
    sincroniza(at)
    sin_error(at, 'N1')
    _ap = boton(at, 'aplicar_casilla')
    comprobar(_ap is not None and any(p in _ap.label for p in ('Volver', 'Back')),
              'al desmarcarla se ofrece volver al modelo de partida',
              _ap.label if _ap is not None else 'sin botón')
    if _ap is not None:
        msgs_n = pulsar(at, _ap, 'N2')
        inf = at.session_state['iter_history'][-1].get('info') or {}
        comprobar(not inf.get('sumado')
                  and any(T('con el modelo de partida, sin tus puntos',
                            'with the starting model, without your points') in m
                          for m in msgs_n)
                  and hay(at, T('Modelo usado: el de partida', 'Model used: the starting model'))
                  and boton(at, 'aplicar_casilla') is None,
                  'vuelve al modelo de partida y lo dice')
    at.checkbox(key='entrenar_propio').check()
    at.run()
    sincroniza(at)
    _ap = boton(at, 'aplicar_casilla')
    if _ap is not None:
        msgs_n = pulsar(at, _ap, 'N3')
        inf = at.session_state['iter_history'][-1].get('info') or {}
        comprobar(inf.get('sumado') and inf.get('n_tabla', 0) > 700
                  and any(T('Modelo reentrenado sumando tus puntos',
                            'Model retrained adding your points') in m for m in msgs_n),
                  'al volver a marcarla y pulsar, reentrena sumando y lo dice')
    else:
        comprobar(False, 'al volver a marcarla se ofrece reentrenar')

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
        # el modelo del tipo aprende también «no borreguil» (el de dos pasos, en «mapa»)
        a.session_state['mrf_quien_decide'] = 'este'
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
    comprobar(any(T('Todavía no hay modelo en este nivel', 'There is no model at this level yet')
                  in str(i.value) for i in at.info)
              and hay(at, T('5 · Peso · sin calcular', '5 · Weights · not computed yet'))
              and grafico_del_peso(at) == (None, None)
              and control(at, 'radio', 'mrf_peso_medida') is None,
              'sin modelo, «5 · Peso» dice que hay que entrenar, y nada más')

    print('  T2 · primera lectura (un punto falla, a otro le falta una variable)')
    poner_proyecto(at, 'ee-proyecto-de-prueba')
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
    comprobar(any(T('lo de junio a septiembre se calcula en cada año y después se hace la '
                    'media entre años',
                    'the June to September figures are computed in each year and then '
                    'averaged across years')
                  in str(c.value) for c in at.caption)
              and any(T('se calculan **en cada año** y después se hace la **media entre años**',
                        'are computed **in each year** and then **averaged across years**')
                      in str(c.value) for c in at.caption),
              'dice cómo se calculan las estadísticas de jun–sep: donde se eligen las '
              'variables y bajo la tabla')

    print('  T3b · lo leído con la manera anterior de calcular las variables no vale')
    lectura = at.session_state['mrf_lectura']
    comprobar(lectura['info']['version'] == geb.VERSION_PILA == 2,
              'la lectura guarda la versión de la pila', str(lectura['info'].get('version')))
    lectura['info']['version'] = 1                      # como si fuera de la app anterior
    at.run()
    sin_error(at, 'T3b')
    b = control(at, 'button', 'mrf_leer')
    comprobar(b is not None and '22' in b.label
              and not hay(at, T('con todas sus variables', 'with all their variables'))
              and hay(at, T('2 · Tabla · sin leer', '2 · Table · not read yet')),
              'una lectura hecha con las estadísticas de antes se descarta: hay que leer '
              'las 22 otra vez', f'{b.label if b else None}')
    lectura['info']['version'] = geb.VERSION_PILA       # se deja como estaba
    at.run()
    comprobar(control(at, 'button', 'mrf_leer') is None
              and hay(at, T('**21 muestras con todas sus variables**',
                            '**21 samples with all their variables**')),
              'con la versión actual, lo leído vuelve a valer')
    tabla = tabla_con_columna(at, 'En la tabla', 'In the table')
    comprobar(tabla is not None and sorted(tabla.iloc[:, 3]) == [6, 7, 8]
              and list(tabla.iloc[:, 2]) == [8, 7, 7],
              'la tabla de categorías compara las muestras con las que entran',
              str(tabla.values.tolist()) if tabla is not None else 'no está')
    comprobar(hay(at, T('Hay más variables (34) que muestras (21)',
                        'There are more variables (34) than samples (21)')),
              'el control de calidad avisa de que hay más variables que muestras')
    xy = tabla_con_columna(at, V('ndvi_early', en))
    comprobar(xy is not None and len(xy) == 21
              and list(xy.columns[3:]) == [V(n, en) for n in mr.nombres_predictores()],
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
    por_cat = [d.value for d in at.dataframe
               if T('Categoría', 'Category') in d.value.columns and 'F1' in d.value.columns]
    por_cat = por_cat[0] if por_cat else None
    comprobar(por_cat is not None and len(por_cat) == 3
              and T('Precisión', 'Precision') in por_cat.columns,
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
                      + N(en=en, v=mod['macro_f1'])),
              'estado: modelo entrenado, con su F1 medio')

    # ¿Mejoraría con más puntos? (25-11-2026, la autora: «¿los puntos mínimos son solo
    # para Sierra Nevada?»): la curva con los puntos de la zona
    print('  M1b · la curva con tus puntos')
    b = control(at, 'button', 'mrf_curva')
    comprobar(b is not None and hay(at, T('#### ¿Mejoraría con más puntos?',
                                          '#### Would more points help?'))
              and hay(at, T(f'Entrena {mr.entrenamientos_curva(mod)} modelos',
                            f'It trains {mr.entrenamientos_curva(mod)} models')),
              'con el modelo entrenado se puede pedir la curva con tus puntos, y dice '
              'cuántos modelos entrena')
    b.click()
    at.run()
    sin_error(at, 'M1b')
    mod = at.session_state['mrf_modelos'].get('humedad')
    cc = mr.cifras_curva(mod['_curva'], mod) if mod and '_curva' in mod else None
    comprobar(cc is not None and [f['fraccion'] for f in cc['filas']] == [0.25, 0.5, 0.75, 1.0]
              and abs(cc['filas'][-1]['f1'] - mod['macro_f1']) < 1e-9,
              'la curva queda en el modelo: 25, 50, 75 y 100 %, y el 100 % es su validación')
    dicho = T('Todavía sube', 'still rising') if cc and cc['lectura'] == 'sube' else \
        T('Ya casi no sube', 'almost stopped rising')
    comprobar(cc is not None
              and hay(at, T('De la mitad de tus puntos a todos (el modelo aprende con unos ',
                            'From half of your points to all of them (the model learns '))
              and hay(at, dicho)
              and hay(at, N(cc['filas'][-1][cc['medida']], en))
              and hay(at, N(cc['mitad'][cc['medida']], en)),
              'dice cuánto acierta con la mitad de los puntos y con todos, y si todavía sube')
    curva = tabla_con_columna(at, 'Parte de tus puntos', 'Share of your points')
    comprobar(curva is not None and len(curva) == 4
              and list(curva.iloc[:, 0]) == ['25 %', '50 %', '75 %', '100 %'],
              'la tabla de la curva tiene las cuatro partes de los puntos')
    from streamlit.dataframe_util import convert_arrow_bytes_to_pandas_df
    partes = [list(convert_arrow_bytes_to_pandas_df(d.data.data)['parte'])
              for g in at.get('vega_lite_chart') for d in g.proto.datasets
              if 'parte' in convert_arrow_bytes_to_pandas_df(d.data.data).columns]
    comprobar(['25 %', '50 %', '75 %', '100 %'] in partes, 'y se dibuja la curva')

    print('  V1 · el peso de las variables')
    imp = mod['importancia']
    comprobar(len(imp['variables']) == 34 and len(imp['familias']) == 9
              and len(imp['grupos']) == 5 and abs(imp['base'] - mod['macro_f1']) < 1e-12
              and imp['repeticiones'] == mr.REPETICIONES_MAX,
              'el modelo trae el peso de sus 34 variables, por familias y por grupos')
    peso_en_pantalla(at, mod, en, 'V1')
    r = control(at, 'radio', 'mrf_peso_medida')
    comprobar(r is not None and r.value == 'permutacion' and len(r.options) == 2
              and T('Recomendado: la primera', 'Recommended: the first one') in (r.help or ''),
              'se puede elegir la medida que se dibuja; la ayuda dice cuál conviene',
              str(r.options) if r else 'no está')
    comprobar(any(T('**No se quita ninguna variable**', '**No variable is removed**')
                  in str(c.value) for c in at.caption)
              and hay(at, T('**Las variables parecidas se reparten el peso.**',
                            '**Similar variables share the weight.**')),
              'dice que no se quita ninguna variable y cómo leer el resultado con cuidado')

    print('  V2 · dibujar la otra medida')
    r.set_value('impureza')
    at.run()
    sin_error(at, 'V2')
    spec, dibujado = grafico_del_peso(at)
    interno = sorted(imp['variables'], key=lambda x: x['orden_impureza'])
    comprobar(dibujado is not None
              and list(dibujado['variable']) == [V(x['variable'], en) for x in interno]
              and all(abs(a - 100 * x['impureza']) < 1e-9 for a, x in zip(dibujado['valor'], interno))
              and abs(float(dibujado['valor'].sum()) - 100) < 1e-6
              and list(dibujado['valor']) == sorted(dibujado['valor'], reverse=True),
              'el gráfico pasa a ordenarse por el uso interno, en % (suman 100)',
              str(list(dibujado['variable'])[:4]) if dibujado is not None else 'no está')
    comprobar(spec is not None
              and not any(c['encoding'].get('x', {}).get('field') == 'menos'
                          for c in spec['layer'])
              and all(c['encoding']['x'].get('title')
                      == T('uso interno (% del total)', 'internal use (% of the total)')
                      for c in spec['layer'] if 'field' in c['encoding'].get('x', {})),
              'sin «±», que esa medida no lo tiene, y con su propio rótulo')
    tabla = tabla_con_columna(at, 'Bajada del F1 sin ella', 'F1 drop without it')
    comprobar(tabla is not None
              and list(tabla['Variable']) == [V(x['variable'], en) for x in imp['variables']],
              'la tabla no cambia: sigue en el orden de lo que pierde el modelo')
    control(at, 'radio', 'mrf_peso_medida').set_value('permutacion')
    at.run()

    print('  U1 · ¿borreguil o no?: el umbral se elige con la validación')
    regla = mod['regla']
    n_borr = sum(1 for m in mod['muestras'] if m['codigo'] != 0)
    modo = control(at, 'radio', 'mrf_umbral_modo')
    comprobar(modo is not None and modo.value == 'f1' and len(modo.options) == 3
              and T('Recomendado: el de mayor F1',
                    'Recommended: the one with the highest F1') in (modo.help or ''),
              'se elige cómo fijar el umbral; de entrada, el de mayor F1 en la validación, '
              'y la ayuda lo explica', str(modo.options) if modo else 'no está')
    comprobar(regla['criterio'] == 'f1' and regla['umbral'] in mr.UMBRALES
              and regla['umbral'] == regla['mejor_f1'] and len(regla['barrido']) == 39
              and regla['en_umbral']['f1'] == max(b['f1'] for b in regla['barrido']),
              'el modelo guarda el umbral de mayor F1 y lo que da cada uno de los 39',
              str({k: regla[k] for k in ('criterio', 'umbral')}))
    eu = regla['en_umbral']
    comprobar(eu['vp'] + eu['fn'] == n_borr == 13 and eu['vn'] + eu['fp'] == 8
              and hay(at, T(f"llega a **{N(regla['umbral'], en)}**",
                            f"add up to **{N(regla['umbral'], en)}**"))
              and hay(at, T(f"encuentra **{eu['vp']} de {n_borr}** borreguiles",
                            f"it finds **{eu['vp']} of {n_borr}** borreguiles"))
              and hay(at, T(f"descarta bien **{eu['vn']} de 8**",
                            f"it correctly rejects **{eu['vn']} of 8**"))
              and hay(at, 'TSS **' + N(eu['tss'], en) + '**'),
              'dice el umbral y lo que da en la validación: borreguiles que encuentra y '
              'lo que descarta bien', str(eu))
    comprobar(any(T('algo optimistas', 'somewhat optimistic') in str(c.value) for c in at.caption),
              'y avisa de que, al elegirlo con las mismas muestras, salen algo optimistas')
    u = np.float32(regla['umbral'])
    comprobar(all((q['predicha'] == 'no borreguil') == (np.float32(q['p_borreguil']) < u)
                  for q in mod['predicciones'])
              and sum(q['predicha'] != 'no borreguil' for q in mod['predicciones'])
              == eu['vp'] + eu['fp'],
              'cada muestra se da por borreguil si la suma de sus tipos llega al umbral')
    spec_u, lineas_u, elegido_u = grafico_del_umbral(at)
    comprobar(lineas_u is not None and len(lineas_u) == 78
              and sorted(set(lineas_u['medida'])) == ['F1', 'TSS']
              and list(lineas_u['umbral'][::2]) == list(mr.UMBRALES)
              and elegido_u is not None and float(elegido_u['umbral'][0]) == regla['umbral']
              and elegido_u['rotulo'][0] == T('umbral', 'threshold') + f" {N(regla['umbral'], en)}",
              'una gráfica enseña el TSS y el F1 con cada umbral, y marca el que se usa',
              str(elegido_u.to_dict('records')) if elegido_u is not None else 'no está')
    escala = next((c['encoding']['color']['scale'] for c in (spec_u or {}).get('layer', [])
                   if 'color' in c.get('encoding', {})), {})
    comprobar(escala.get('domain') == ['TSS', 'F1'] and len(set(escala.get('range', []))) == 2,
              'con un color fijo para cada medida y su leyenda', str(escala))
    barr = tabla_con_columna(at, 'Borreguiles encontrados', 'Borreguiles found')
    comprobar(barr is not None and len(barr) == 39
              and [round(float(x), 3) for x in barr.iloc[:, 0]] == list(mr.UMBRALES)
              and all(abs(a - b['tss']) < 1e-12 for a, b in zip(barr['TSS'], regla['barrido']))
              and all(int(a) + int(b_) == n_borr for a, b_ in zip(barr.iloc[:, 6], barr.iloc[:, 7])),
              'y la misma información va en una tabla, umbral a umbral',
              str(list(barr.columns)) if barr is not None else 'no está')

    print('  U2 · con otro criterio o a mano, las cifras se rehacen sin volver a entrenar')
    bosque, matriz_tss = mod['modelo'], [list(f) for f in mod['matriz']]
    modo.set_value('manual')
    at.run()
    sin_error(at, 'U2 · a mano')
    mod = at.session_state['mrf_modelos']['humedad']
    desl = control(at, 'slider', 'mrf_umbral_valor')
    comprobar(desl is not None and desl.value == 0.5
              and T('Recomendado', 'Recommended') in (desl.help or '')
              and mod['regla']['criterio'] == 'manual' and mod['regla']['umbral'] == 0.5
              and mod['modelo'] is bosque
              and hay(at, T('llega a **0,500**', 'add up to **0.500**'))
              and not any(T('algo optimistas', 'somewhat optimistic') in str(c.value)
                          for c in at.caption),
              'a mano aparece el deslizador, en 0.5, y se aplica sin entrenar otra vez',
              str(mod['regla'].get('umbral')))
    desl.set_value(0.9)
    at.run()
    sin_error(at, 'U2 · 0.9')
    mod = at.session_state['mrf_modelos']['humedad']
    eu = mod['regla']['en_umbral']
    comprobar(mod['regla']['umbral'] == 0.9 and sum(map(sum, mod['matriz'])) == 21
              and sum(q['predicha'] != 'no borreguil' for q in mod['predicciones'])
              == eu['vp'] + eu['fp']
              and abs(mod['importancia']['base'] - mod['macro_f1']) < 1e-12
              and hay(at, T('3 · Modelo · F1 medio ', '3 · Model · mean F1 ')
                      + N(en=en, v=mod['macro_f1']))
              and N(en=en, v=mod['macro_f1']) in [m.value for m in at.metric],
              'con 0.9, la validación, el estado y el peso de las variables son los de ese '
              'umbral', str(eu))
    control(at, 'radio', 'mrf_umbral_modo').set_value('tss')
    at.run()
    mod = at.session_state['mrf_modelos']['humedad']
    comprobar(mod['regla']['criterio'] == 'tss' and mod['regla']['umbral'] == mod['regla']['mejor_tss']
              and control(at, 'slider', 'mrf_umbral_valor') is None,
              'con el TSS se usa su mejor umbral, y el deslizador desaparece')
    control(at, 'radio', 'mrf_umbral_modo').set_value('f1')
    at.run()
    sin_error(at, 'U2 · vuelta')
    mod = at.session_state['mrf_modelos']['humedad']
    comprobar(mod['regla']['criterio'] == 'f1' and mod['regla']['umbral'] == regla['umbral']
              and [list(f) for f in mod['matriz']] == matriz_tss and mod['modelo'] is bosque,
              'al volver al de partida, todo queda como estaba')

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
    comprobar(sum(aviso_tabla in str(w.value) for w in at.warning) == 2
              and hay(at, T('5 · Peso · es de una tabla anterior',
                            '5 · Weights · from a previous table'))
              and len(grafico_del_peso(at)[1]) == 34,
              'el peso de las variables también avisa: sigue siendo el del modelo anterior')
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
                      + N(en=en, v=mod['macro_f1'])),
              'ya no avisa: el modelo es de la tabla actual')

    print('  V3 · el peso es el del modelo nuevo, con sus 20 variables')
    imp = mod['importancia']
    comprobar(len(imp['variables']) == 20 and len(imp['familias']) == 9
              and sorted(f['n_variables'] for f in imp['familias']) == [1, 1, 2, 2, 2, 2, 2, 3, 5]
              and [g['n_variables'] for g in sorted(imp['grupos'], key=lambda g: mr.GRUPOS.index(
                  g['grupo']))] == [6, 6, 1, 2, 5],
              'se mide solo con las variables que entran en el modelo',
              str([(f['familia'], f['n_variables']) for f in imp['familias']]))
    peso_en_pantalla(at, mod, en, 'V3')

    print('  V4 · un modelo de antes de que se midiera el peso')
    guardado_imp = mod.pop('importancia')
    at.run()
    sin_error(at, 'V4')
    comprobar(any(T('se entrenó antes de que la app midiera el peso',
                    'was trained before the app measured the weight') in str(i.value)
                  for i in at.info)
              and hay(at, T('5 · Peso · vuelve a entrenar', '5 · Weights · train again'))
              and grafico_del_peso(at) == (None, None),
              'no falla: dice que hay que volver a entrenar')
    mod['importancia'] = guardado_imp
    at.run()

    print('  T7 · quitar las variables repetidas con el VIF, y comparar el acierto')
    caja = control(at, 'radio', 'mrf_seleccion')
    comprobar(caja is not None and caja.value == 'ninguno'
              and list(caja.options) == [T('No quitar ninguna', 'Remove none'),
                                         T('Con el VIF', 'With the VIF'),
                                         T('Con un PCA', 'With a PCA')]
              and control(at, 'number_input', 'mrf_vif_umbral') is None
              and T('Recomendado:', 'Recommended:') in (caja.help or ''),
              'la opción de las variables repetidas (VIF o PCA) está; de entrada no se '
              'quita nada')

    def pestana_datos():
        return [x for x in at.tabs if x.label == T('1 · Datos', '1 · Data')]

    comprobar(pestana_datos()
              and any(c.key == 'mrf_seleccion' for c in pestana_datos()[0].radio)
              and hay(at, T('Para quitar las variables repetidas, elige el VIF o un PCA en '
                            '«1 · Datos»',
                            'To remove repeated variables, choose the VIF or a PCA under '
                            '"1 · Data"')),
              'el VIF y el PCA se eligen en «1 · Datos», junto a las variables, y «2 · Tabla» '
              'lo dice')
    caja.set_value('vif')
    at.run()
    sin_error(at, 'T7')
    umbral = control(at, 'number_input', 'mrf_vif_umbral')
    fijas = control(at, 'multiselect', 'mrf_vif_fijas')
    comprobar(umbral is not None and umbral.value == 10.0 and fijas is not None
              and fijas.value == ['elev_dem_m']
              and T('Recomendado: 10', 'Recommended: 10') in (umbral.help or '')
              and T('Recomendado: la altitud', 'Recommended: elevation') in (fijas.help or ''),
              'con el umbral 10 y la altitud fija, cada uno con lo que es y lo recomendado')
    # Desde el 25-11-2026 se aplica con un botón, y lo que sale se ve en «1 · Datos»
    boton = control(at, 'button', 'mrf_aplicar_sel')
    comprobar(boton is not None and boton.label == T('✔ Aplicar el VIF', '✔ Apply the VIF')
              and not boton.disabled
              and hay(at, T('Pulsa «✔ Aplicar el VIF»', 'Press "✔ Apply the VIF"'))
              and hay(at, T('pero no está aplicado: se usan las 20 variables',
                            'but it is not applied: the 20 variables are used'))
              and hay(at, T('2 · Tabla · 22 muestras, 20 variables',
                            '2 · Table · 22 samples, 20 variables'))
              and tabla_con_columna(at, 'Qué pasa', 'What happens') is None,
              'al elegir el VIF no se quita nada hasta pulsar «Aplicar» (lo dicen «1 · Datos» '
              'y «2 · Tabla»)')
    # Lo elegido en otros pasos no se pierde al aplicar (25-11-2026: con st.rerun() en
    # el botón, Streamlit olvidaba los controles que esa pasada no llegaba a dibujar)
    control(at, 'number_input', 'mrf_p_semilla').set_value(43)
    at.run()
    boton = control(at, 'button', 'mrf_aplicar_sel')
    boton.click()
    at.run()
    sin_error(at, 'T7 · aplicar')
    semilla = control(at, 'number_input', 'mrf_p_semilla')
    comprobar(semilla is not None and semilla.value == 43,
              'al aplicar el VIF, lo elegido en «3 · Modelo» se queda como estaba',
              str(semilla.value) if semilla else 'no está')
    semilla.set_value(42)
    at.run()
    vif_t = tabla_con_columna(at, 'Qué pasa', 'What happens')
    de_nombre = {V(n, en): n for n in mr.nombres_predictores()}   # nombre legible → código
    quedan = [] if vif_t is None else [
        de_nombre[v] for v, q in zip(vif_t['Variable'], vif_t.iloc[:, 2])
        if not q.startswith(T('se quita', 'removed'))]
    comprobar(vif_t is not None and len(vif_t) == 20 and 0 < len(quedan) < 20
              and 'elev_dem_m' in quedan
              and hay(at, T(f'**Con el VIF se quedan {len(quedan)} de 20 variables:**',
                            f'**With the VIF, {len(quedan)} of 20 variables stay:**'))
              and hay(at, T(f'2 · Tabla · 22 muestras, {len(quedan)} variables (VIF)',
                            f'2 · Table · 22 samples, {len(quedan)} variables (VIF)'))
              and hay(at, T(f'**Variables repetidas (VIF):** se quedan {len(quedan)} de 20; '
                            'cuáles y por qué, en «1 · Datos»',
                            f'**Repeated variables (VIF):** {len(quedan)} of 20 stay; which '
                            'ones and why, under "1 · Data"'))
              and hay(at, T('el VIF no es fiable', 'the VIF is not reliable')),
              'al aplicarlo dice cuáles se quedan (la altitud, siempre) y cuáles se quitan; '
              '«2 · Tabla» sigue con esas; con 22 muestras y 20 variables, avisa de que el '
              'VIF no es fiable', str(quedan))

    def marcadas():
        return sorted(v for i in range(len(mr.GRUPOS))
                      for v in control(at, 'multiselect', f'mrf_vars_{i}').value)

    # 25-11-2026, la autora: «al aplicar el vif o pca me gustaría que en variables
    # predictoras se queden las que han quedado»
    comprobar(marcadas() == sorted(quedan)
              and hay(at, T(f'{len(quedan)} de {len(mr.PREDICTORES)} variables seleccionadas',
                            f'{len(quedan)} of {len(mr.PREDICTORES)} variables selected'))
              and hay(at, T('Están marcadas las que dejó el VIF. Antes de aplicarlo había 20 '
                            'marcadas.', 'The ones the VIF kept are selected. Before applying '
                            'it there were 20 selected.'))
              and control(at, 'button', 'mrf_vars_volver') is not None,
              'al aplicarlo, en «Variables predictoras» quedan marcadas solo las que se '
              'quedan, y se puede volver a las 20', str(marcadas()))
    _datos = pestana_datos()
    _md_datos = [str(m.value) for m in _datos[0].markdown] if _datos else []
    comprobar(_datos
              and any(c.key == 'mrf_vif_umbral' for c in _datos[0].number_input)
              and any(T(f'Con el VIF se quedan {len(quedan)} de 20 variables',
                        f'With the VIF, {len(quedan)} of 20 variables stay')
                      in str(m.value) for m in _datos[0].success)
              and any(T('Qué pasa', 'What happens') in d.value.columns
                      and T('Las demás la explican', 'The others explain it') in d.value.columns
                      for d in _datos[0].dataframe)
              and any(T('Las demás explicaban el', 'The others explained') in t
                      and T('que se queda.', 'which stays.') in t for t in _md_datos),
              'en «1 · Datos», junto al umbral y las fijas: cuáles se quedan, la tabla del '
              'VIF y, variable a variable, por qué (cuánto la explican las demás y a cuál '
              'se parece de las que se quedan)')
    control(at, 'number_input', 'mrf_vif_umbral').set_value(5.0)
    at.run()
    sin_error(at, 'T7 · otro umbral')
    comprobar(hay(at, T('Has cambiado algo desde que lo aplicaste',
                        'Something has changed since you applied it'))
              and hay(at, T(f'2 · Tabla · 22 muestras, {len(quedan)} variables',
                            f'2 · Table · 22 samples, {len(quedan)} variables'))
              and not hay(at, f'{len(quedan)} variables (VIF)')
              and marcadas() == sorted(quedan)
              and tabla_con_columna(at, 'Qué pasa', 'What happens') is None,
              'si cambia el umbral, hay que volver a aplicarlo; mientras, se usan las '
              'marcadas')
    control(at, 'number_input', 'mrf_vif_umbral').set_value(10.0)
    at.run()
    sin_error(at, 'T7 · umbral de antes')
    comprobar(hay(at, T(f'2 · Tabla · 22 muestras, {len(quedan)} variables (VIF)',
                        f'2 · Table · 22 samples, {len(quedan)} variables (VIF)')),
              'con el umbral con el que se aplicó, lo aplicado vuelve a valer')
    comprobar(hay(at, aviso_tabla), 'el modelo de antes pasa a ser de una tabla anterior')
    control(at, 'button', 'mrf_entrenar').click()
    at.run()
    sin_error(at, 'T7 · entrenar')
    mod_v = at.session_state['mrf_modelos']['humedad']
    comprobar(mod_v['predictores'] == mr.ordenar_predictores(quedan)
              and mod_v['variables']['umbral'] == 10.0 and mod_v['variables']['de'] == 20
              and mod_v['variables']['metodo'] == 'vif'
              and len(grafico_del_peso(at)[1]) == len(quedan) and not hay(at, aviso_tabla),
              'el modelo nuevo, y su peso, usan solo las que se quedan')
    comp = tabla_con_columna(at, 'F1 medio', 'Mean F1')
    filas = [] if comp is None else list(comp.iloc[:, 0])
    comprobar(comp is not None and len(comp) == 3
              and filas[0].endswith('(34)') and filas[1].endswith('(20)')
              and not filas[0].startswith('▶') and not filas[1].startswith('▶')
              and filas[2].startswith('▶') and 'VIF ≤ 10' in filas[2]
              and list(comp.iloc[:, 1]) == [21, 22, mod_v['n_muestras']]
              and list(comp.iloc[:, 2])[1:] == [N(en=en, v=mod['macro_f1']),
                                                N(en=en, v=mod_v['macro_f1'])],
              'al final de «3 · Modelo», el acierto de los tres modelos con las mismas '
              'muestras (34, 20 y las del VIF), marcado el de ahora',
              str(comp.values.tolist()) if comp is not None else 'no está')
    control(at, 'radio', 'mrf_seleccion').set_value('ninguno')
    at.run()
    sin_error(at, 'T7 · sin VIF')
    comprobar(hay(at, T(f'2 · Tabla · 22 muestras, {len(quedan)} variables',
                        f'2 · Table · 22 samples, {len(quedan)} variables'))
              and not hay(at, f'{len(quedan)} variables (VIF)')
              and marcadas() == sorted(quedan)
              and control(at, 'number_input', 'mrf_vif_umbral') is None
              and not hay(at, aviso_tabla),
              'sin la opción siguen marcadas las que dejó el VIF (es la misma tabla: el '
              'modelo sigue al día)')
    control(at, 'button', 'mrf_vars_volver').click()
    at.run()
    sin_error(at, 'T7 · volver a las 20')
    comprobar(len(marcadas()) == 20
              and hay(at, T('2 · Tabla · 22 muestras, 20 variables',
                            '2 · Table · 22 samples, 20 variables'))
              and control(at, 'button', 'mrf_vars_volver') is None and hay(at, aviso_tabla),
              '«↺ Volver a marcar las 20 de antes» devuelve las 20, y el modelo del VIF queda '
              'como de otra tabla')

    print('  T7b · elegir las variables con un PCA')
    control(at, 'radio', 'mrf_seleccion').set_value('pca')
    at.run()
    sin_error(at, 'T7b')
    comprobar(hay(at, T('Pulsa «✔ Aplicar el PCA»', 'Press "✔ Apply the PCA"'))
              and tabla_con_columna(at, 'Eje', 'Axis') is None,
              'el PCA tampoco quita nada hasta pulsar «Aplicar»')
    control(at, 'button', 'mrf_aplicar_sel').click()
    at.run()
    sin_error(at, 'T7b · aplicar')
    carga = control(at, 'number_input', 'mrf_pca_carga')
    ejes_t = tabla_con_columna(at, 'Eje', 'Axis')
    elegidas = [] if ejes_t is None else [
        de_nombre[v] for v in ejes_t.iloc[:, 4] if v != '—']
    quedan_p = mr.ordenar_predictores(elegidas + ['elev_dem_m'])
    comprobar(carga is not None and abs(carga.value - 0.65) < 1e-9
              and control(at, 'number_input', 'mrf_vif_umbral') is None
              and control(at, 'multiselect', 'mrf_vif_fijas').value == ['elev_dem_m']
              and T('Recomendado: 0,65', 'Recommended: 0.65') in (carga.help or ''),
              'con el PCA: la correlación mínima con el eje (0,65) y la altitud fija')
    comprobar(ejes_t is not None and len(ejes_t) >= 1 and elegidas
              and 'elev_dem_m' not in elegidas
              and hay(at, T(f'2 · Tabla · 22 muestras, {len(quedan_p)} variables (PCA)',
                            f'2 · Table · 22 samples, {len(quedan_p)} variables (PCA)'))
              and any(T(f'Con el PCA se quedan {len(quedan_p)} de 20 variables',
                        f'With the PCA, {len(quedan_p)} of 20 variables stay')
                      in str(m.value) for m in pestana_datos()[0].success)
              and hay(at, T('el PCA no es fiable', 'the PCA is not reliable')),
              'una variable por eje del PCA, más la altitud; lo dicen «1 · Datos», «2 · Tabla» '
              'y la línea de estado; con 22 muestras avisa de que no es fiable',
              str(elegidas))
    comprobar(marcadas() == sorted(quedan_p)
              and hay(at, T('Están marcadas las que dejó el PCA. Antes de aplicarlo había 20 '
                            'marcadas.', 'The ones the PCA kept are selected. Before applying '
                            'it there were 20 selected.')),
              'con el PCA también quedan marcadas solo las que se quedan', str(marcadas()))
    var_t = tabla_con_columna(at, 'Su eje', 'Its axis')
    quedan_t = [] if var_t is None else sorted(
        de_nombre[v] for v, q in zip(var_t['Variable'], var_t.iloc[:, 1])
        if not q.startswith(T('se quita', 'removed')))
    comprobar(var_t is not None and len(var_t) == 20 and quedan_t == sorted(quedan_p)
              and hay(at, T('Valor propio de cada eje del PCA', 'Eigenvalue of each PCA axis'))
              and any(T('es la que mejor separa', 'it is the one that best separates')
                      in str(m.value) for m in pestana_datos()[0].markdown),
              'en «1 · Datos»: el gráfico de los ejes, la tabla variable a variable (casa con '
              'las que se quedan) y por qué', str(quedan_t))
    control(at, 'button', 'mrf_entrenar').click()
    at.run()
    sin_error(at, 'T7b · entrenar')
    mod_p = at.session_state['mrf_modelos']['humedad']
    comp = tabla_con_columna(at, 'F1 medio', 'Mean F1')
    filas = [] if comp is None else list(comp.iloc[:, 0])
    comprobar(mod_p['predictores'] == quedan_p and mod_p['variables']['metodo'] == 'pca'
              and any(f.startswith('▶') and T('una por eje del PCA', 'one per PCA axis') in f
                      for f in filas),
              'el modelo usa solo las del PCA y sale en la comparación', str(filas))
    # Con el PCA aplicado, el VIF se vuelve a aplicar desde las 20 de antes, no desde
    # las que dejó el PCA
    control(at, 'radio', 'mrf_seleccion').set_value('vif')
    at.run()
    sin_error(at, 'T7b · otra vez el VIF')
    comprobar(hay(at, T('Has cambiado algo desde que lo aplicaste',
                        'Something has changed since you applied it'))
              or hay(at, T('Pulsa «✔ Aplicar el VIF»', 'Press "✔ Apply the VIF"')),
              'al volver al VIF, lo del PCA no cuenta como aplicado con el VIF')
    control(at, 'button', 'mrf_aplicar_sel').click()
    at.run()
    sin_error(at, 'T7b · el VIF desde las 20')
    comprobar(marcadas() == sorted(quedan)
              and hay(at, T(f'**Con el VIF se quedan {len(quedan)} de 20 variables:**',
                            f'**With the VIF, {len(quedan)} of 20 variables stay:**')),
              'el VIF parte de las 20 de antes y deja las mismas que la primera vez',
              str(marcadas()))
    # Volver a las 20 con el VIF elegido: el VIF sigue elegido, con su umbral y su botón
    control(at, 'button', 'mrf_vars_volver').click()
    at.run()
    sin_error(at, 'T7b · volver a las 20')
    comprobar(len(marcadas()) == 20
              and control(at, 'radio', 'mrf_seleccion').value == 'vif'
              and control(at, 'number_input', 'mrf_vif_umbral') is not None
              and control(at, 'button', 'mrf_aplicar_sel') is not None
              and hay(at, T('Pulsa «✔ Aplicar el VIF»', 'Press "✔ Apply the VIF"'))
              and hay(at, T('2 · Tabla · 22 muestras, 20 variables',
                            '2 · Table · 22 samples, 20 variables')),
              'al volver a las 20, el VIF sigue elegido y hay que volver a aplicarlo')
    control(at, 'radio', 'mrf_seleccion').set_value('ninguno')
    at.run()
    sin_error(at, 'T7b · sin PCA')

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
    comprobar(control(otra, 'text_input', 'mrf_proyecto') is None
              and any(t.value == 'ee-proyecto-de-prueba' for t in otra.text_input
                      if 'Earth Engine' in t.label),
              'al abrir de nuevo, aparece ya escrito en el panel lateral (y la pestaña del '
              'mapa no lo vuelve a pedir)')


# ------------------------------------------------------------------- el mapa
# Con el modelo entrenado, el paso «4 · Mapa»: elegir la zona, hacer el mapa, verlo,
# pinchar un píxel y descargarlo. Earth Engine se sustituye por un «mundo» en el que
# cada píxel de 10 m tiene unos valores que dependen solo de dónde está: así lo que
# se lee para las muestras y lo que se lee para el mapa es, como en la realidad, lo
# mismo. El visor (que en el navegador es un componente aparte) también se sustituye,
# para poder mirar qué se le manda y simular un clic.
import math                                  # noqa: E402

import numpy as np                           # noqa: E402
import pyproj                                # noqa: E402
import streamlit_folium                      # noqa: E402

import mapeo_ambiente as mam
import mapeo_raster as mra                   # noqa: E402
import mapeo_tif as mt                       # noqa: E402

BLOQUES, PILAS, VISORES, CLICS = [], [], {}, {}
FALLA_BLOQUE = {'activo': False}
# Para el ambiente por cercanía hace falta un «mundo» con sentido: si se rellena, los
# bloques traen una laguna (NDWI alto en un círculo de 25 m), terreno casi llano y un
# relieve en vaguada que baja hacia el sur por una coordenada X.
MUNDO = {'ambiente': None}


def _valor_pixel(cx, cy, j):
    return (zlib.crc32(f'{int(cx)},{int(cy)},{j}'.encode()) % 1000) / 1000.0


def _lector_de_pixeles(puntos, years, epsg=None, progress=None):
    tr = pyproj.Transformer.from_crs(4326, epsg, always_xy=True)
    valores = []
    for lon, lat in puntos:
        x, y = tr.transform(lon, lat)
        cx, cy = math.floor(x / 10) * 10 + 5, math.floor(y / 10) * 10 + 5
        valores.append({b: _valor_pixel(cx, cy, j) for j, b in enumerate(geb.BANDAS_PILA)})
    if progress:
        progress(1.0)
    return valores, 0, geb.info_pila(years, epsg)


def _pila_simulada(limites, years, epsg, bandas):
    PILAS.append({'years': tuple(years), 'epsg': epsg, 'bandas': list(bandas)})
    return 'pila simulada'


def _bloque_simulado(pila, epsg, x0, y0, ancho, alto, bandas, **kw):
    BLOQUES.append((x0, y0, ancho, alto))
    if FALLA_BLOQUE['activo']:
        raise RuntimeError('Earth Engine no ha devuelto un bloque del mapa (simulado).')
    idx = {b: j for j, b in enumerate(geb.BANDAS_PILA)}
    out = np.empty((alto, ancho, len(bandas)), dtype='float32')
    for f in range(alto):
        for c in range(ancho):
            out[f, c] = [_valor_pixel(x0 + c * 10 + 5, y0 - f * 10 - 5, idx[b]) for b in bandas]
    m = MUNDO['ambiente']
    if m:
        X, Y = np.meshgrid(x0 + np.arange(ancho) * 10 + 5, y0 - np.arange(alto) * 10 - 5)
        for k, banda in enumerate(bandas):
            if banda == 'ndwi_mean':
                out[..., k] = np.where(
                    np.hypot(X - m['lago'][0], Y - m['lago'][1]) <= 25, 0.3, -0.2)
            elif banda == 'slope_deg':
                out[..., k] = 8.0
            elif banda == 'elev_dem_m':
                out[..., k] = 3000 + 0.2 * (Y - m['lago'][1]) + 0.3 * np.abs(X - m['eje'])
    return out


def _leer_guion(js):
    """Lo que el JavaScript de «lo que cambia» va a poner en el visor: la imagen y los
    límites de cada capa, la opacidad, y la categoría elegida con su rótulo."""
    import re
    datos = json.loads(re.search(r'var datos = (\{.*?\});\n', js, re.S).group(1))
    elegida, nombre = re.search(
        r'var elegida = ("(?:[^"\\]|\\.)*"), nombre = ("(?:[^"\\]|\\.)*");', js).groups()
    return {'datos': datos, 'opacidad': float(re.search(r'setOpacity\(([0-9.]+)\)', js).group(1)),
            'elegida': json.loads(elegida), 'nombre': json.loads(nombre)}


def _visor_simulado(fig, key=None, returned_objects=None, feature_group_to_add=None, **kw):
    """Hace de st_folium. Guarda lo que se le manda (lo fijo del visor y, aparte, lo que
    cambia) y devuelve lo que diga CLICS para esa clave: un punto pinchado (lon, lat) o
    un dict con lo que el visor contaría (clic, zoom y lo que muestra). Como el visor de
    verdad, deja además su valor en la sesión, con su clave: la app lo lee de ahí al
    empezar la pasada siguiente."""
    import streamlit as st
    hijos = list(feature_group_to_add._children.values()) if feature_group_to_add is not None else []
    guiones = [h for h in hijos if type(h).__name__ == '_Guion']
    VISORES[key] = {'html': fig.get_root().render(), 'capa': feature_group_to_add,
                    'devuelve': list(returned_objects or []), 'opciones': dict(kw),
                    'cambia': _leer_guion(guiones[0].hacer()) if guiones else None,
                    'otros': [h for h in hijos if type(h).__name__ != '_Guion']}
    valor = {'last_clicked': None}
    for prefijo, dato in CLICS.items():
        if key and key.startswith(prefijo):
            if isinstance(dato, dict):
                valor.update(dato)
            else:
                valor['last_clicked'] = {'lat': dato[1], 'lng': dato[0]}
            st.session_state[key] = valor
    return valor


def _fijo(html):
    """Lo fijo del visor sin los nombres al azar que folium da a cada elemento: si no
    cambia de una pasada a otra, el visor no se recarga."""
    import re
    return re.sub(r'[0-9a-f]{32}', '', html)


def _capa(nombre):
    """El nombre de una capa tal como va escrito dentro del visor (texto JSON)."""
    return json.dumps(nombre)[1:-1]


def _visor(prefijo):
    claves = [k for k in VISORES if k and k.startswith(prefijo)]
    return VISORES[claves[-1]] if claves else None


def guion_mapa(lang):
    en = lang == 'English'
    T = (lambda es_, en_: en_ if en else es_)
    print(f'\n================ {lang} · mapa ================')
    SELECCION['rows'] = []
    for x in (BLOQUES, PILAS):
        x.clear()
    VISORES.clear()
    CLICS.clear()
    FALLA_BLOQUE['activo'] = False
    os.environ['BORREGUIL_DATA_DIR'] = tempfile.mkdtemp(prefix='bt_')
    geb.initialize = lambda project=None, service_account_json=None: (True, 'simulado')
    geb.leer_predictores = _lector_de_pixeles
    geb.pila_para_mapa = _pila_simulada
    geb.leer_bloque = _bloque_simulado
    visor_de_verdad = streamlit_folium.st_folium
    streamlit_folium.st_folium = _visor_simulado

    # Las mismas 22 muestras de la tabla: 7 húmedos, 7 secos y 8 ausencias de campo.
    rows = [dict(r) for r in base if r['ID'] != 'sin_datos']
    borr = [r for r in rows if pp.is_borreguil_decision(r['decision'])][:14]
    for k, r in enumerate(borr):
        bp.set_tipo_revisado(r, humedad=('húmedo' if k % 2 else 'seco'))
    for r in [r for r in rows if r['decision'] == 'NO BORREGUIL'][:7]:
        r['Borreguil'] = 'no'
        r['decision'] = bp.decide(r, 0.5)
    bp.apply_typology(rows, pp.is_borreguil_decision)

    at = AppTest.from_file(str(APP / 'app.py'), default_timeout=600)
    at.session_state['rows'] = [dict(r) for r in rows]
    at.session_state['work'] = Path(tempfile.mkdtemp(prefix='bw_'))
    at.session_state['info'] = {k: v for k, v in (info or {}).items() if k != 'model_bundle'}
    at.session_state['ui_lang'] = lang
    at.session_state['mrf_quien_decide'] = 'este'
    at.run()
    control(at, 'selectbox', 'mrf_nivel').select('humedad')
    at.run()
    try:
        _pasos_del_mapa(at, T, en)
    finally:
        streamlit_folium.st_folium = visor_de_verdad


def _pasos_del_mapa(at, T, en):
    sin_error(at, 'P1')
    comprobar(hay(at, T('Antes hace falta un modelo entrenado («3 · Modelo»)',
                        'A trained model is needed first ("3 · Model")'))
              and control(at, 'button', 'mrf_hacer_mapa') is None,
              'sin modelo, el paso del mapa dice qué falta y no ofrece hacerlo')

    poner_proyecto(at, 'ee-proyecto-de-prueba')
    at.run()
    pulsar(at, control(at, 'button', 'mrf_leer'), 'P1 · leer')
    control(at, 'button', 'mrf_entrenar').click()
    at.run()
    sin_error(at, 'P1 · entrenar')
    mod = at.session_state['mrf_modelos']['humedad']
    comprobar(mod.get('lectura') == {'anios': list(range(2017, 2026)), 'epsg': 32630,
                                     'version': geb.VERSION_PILA}
              and len(mod['muestras']) == 22,
              'el modelo recuerda de qué años y rejilla salió su tabla, cómo se calcularon '
              'las variables, y sus muestras', str(mod.get('lectura')))

    print('  P2 · la zona: un rectángulo de 2 × 2 km por defecto')
    zona = control(at, 'radio', 'mrf_zona')
    comprobar(zona is not None and zona.value == 'rect'
              and list(zona.options) == [T('Un rectángulo que elijo yo', 'A rectangle I choose'),
                                         T('El rectángulo que ocupan las muestras',
                                           'The rectangle the samples occupy')],
              'se puede elegir un rectángulo propio o el de las muestras',
              str(zona.options) if zona else 'no está')
    comprobar(T('Recomendado para empezar', 'Recommended to start') in (zona.help or ''),
              'la zona explica qué conviene elegir para empezar')
    ancho, alto = control(at, 'number_input', 'mrf_rect_ancho'), control(
        at, 'number_input', 'mrf_rect_alto')
    lon0 = control(at, 'number_input', 'mrf_rect_lon').value
    lat0 = control(at, 'number_input', 'mrf_rect_lat').value
    comprobar(ancho.value == 2.0 and alto.value == 2.0
              and any(abs(m['lon'] - lon0) < 1e-5 and abs(m['lat'] - lat0) < 1e-5
                      for m in mod['muestras']),
              'el rectángulo arranca en 2 × 2 km, centrado en una muestra', f'{lon0}, {lat0}')
    comprobar(all(T('Recomendado', 'Recommended') in (c.help or '') for c in (ancho, alto)),
              'el ancho y el alto dicen qué valor conviene')
    comprobar(hay(at, T('**4,000 km²** que clasificar', '**4.000 km²** to classify'))
              and hay(at, T('en 1 trozo de Earth Engine', 'in 1 Earth Engine piece'))
              and hay(at, T('años 2017-2025', 'years 2017-2025')),
              'antes de lanzarlo dice cuánto es, con qué años y cuánto tardará')
    v = _visor('mrf_mapa_zona')
    comprobar(v is not None and v['devuelve'] == ['last_clicked'] and v['capa'] is not None
              and len(v['capa']._children) == 1 + 22,
              'la zona se enseña en un mapa con el rectángulo y las 22 muestras',
              str(len(v['capa']._children)) if v else 'no está')
    hz = v['html'] if v else ''
    comprobar(all(x in hz for x in (
        T('Muestras del modelo', 'Model samples'),
        T('Lo que se va a cartografiar', 'What will be mapped'),
        mra.COLORES['húmedo'], mra.COLORES['seco'], mra.COLORES['no borreguil'],
        T('>húmedo<', '>wet<'), T('>seco<', '>dry<'), T('>no borreguil<', '>not a borreguil<')))
        and T('Máscara', 'Mask') + '</div>' not in hz,
        'ese mapa lleva la leyenda: el color de cada categoría de punto y qué es el recuadro')
    b = control(at, 'button', 'mrf_hacer_mapa')
    comprobar(b is not None and not b.disabled, 'con modelo, zona y proyecto se puede hacer')
    comprobar(not hay(at, T('no tiene la clase «no borreguil»', 'has no "not a borreguil" class')),
              'con ausencias de campo no se exige máscara')
    agua = control(at, 'checkbox', 'mrf_quitar_agua')
    comprobar(agua is not None and agua.value is True
              and T('Recomendado: marcado', 'Recommended: ticked') in (agua.help or ''),
              'se ofrece quitar el agua de laguna, marcado de entrada')
    # Lo que sigue compara el mapa con lo que dice el modelo, píxel a píxel: sin quitar
    # el agua (eso lo prueba tests/test_mapeo_mapa.py).
    agua.uncheck()
    at.run()

    print('  P3 · pinchar en el mapa de la zona mueve el rectángulo')
    otra = mod['muestras'][3]
    CLICS['mrf_mapa_zona'] = (otra['lon'], otra['lat'])
    at.run()
    at.run()
    sin_error(at, 'P3')
    comprobar(abs(control(at, 'number_input', 'mrf_rect_lon').value - round(otra['lon'], 5)) < 1e-9
              and abs(control(at, 'number_input', 'mrf_rect_lat').value - round(otra['lat'], 5)) < 1e-9,
              'el centro pasa a ser el punto pinchado')
    CLICS.clear()
    control(at, 'number_input', 'mrf_rect_ancho').set_value(0.5)
    control(at, 'number_input', 'mrf_rect_alto').set_value(0.4)
    at.run()
    sin_error(at, 'P3 · tamaño')
    comprobar(hay(at, T('**0,200 km²** que clasificar', '**0.200 km²** to classify')),
              'al cambiar el tamaño cambia la superficie')

    print('  P3b · la pureza dentro del píxel: sus valores, antes de hacer el mapa')
    si = control(at, 'checkbox', 'mrf_pureza_si')
    sin_v = control(at, 'number_input', 'mrf_pureza_sin')
    todo_v = control(at, 'number_input', 'mrf_pureza_todo')
    corte = control(at, 'slider', 'mrf_pureza_corte')
    comprobar(si is not None and si.value is True and sin_v is not None and todo_v is not None
              and corte is not None and abs(sin_v.value - 0.08) < 1e-9 and corte.value == 75
              and all(T('Recomendado', 'Recommended') in (w.help or '')
                      for w in (si, sin_v, todo_v, corte)),
              'se puede pedir la pureza de cada píxel de borreguil, con tres valores de '
              'partida y su ayuda', str([getattr(w, 'value', None) for w in (si, sin_v, corte)]))
    # lo que proponen las muestras: el NDVI de las de borreguil puro (su mediana)
    filas_p = at.session_state['rows']
    lect = at.session_state['mrf_lectura']['valores']
    de_borr = [(bp._txt(filas_p[k].get('pureza')),
                lect[mr.clave_punto(filas_p[k]['lon'], filas_p[k]['lat'])]['ndvi_early'])
               for k in range(len(filas_p)) if filas_p[k].get('tipo_revisado') == 'si']
    puras = [v for c, v in de_borr if c == 'puro']
    propuesto, de_muestras = mr.todo_verde_de(puras)
    comprobar(len(de_borr) == 14 and abs(todo_v.value - round(propuesto, 3)) < 1e-9
              and any(T(f'{len(puras)} muestras de borreguil puro',
                        f'{len(puras)} pure borreguil samples') in str(c.value)
                      for c in at.caption),
              'el «todo verde» que viene puesto es el de las muestras de borreguil puro, y '
              'se dice de cuántas sale', f'{todo_v.value} · {propuesto:.3f} · {len(puras)} puras')
    par = {'sin_verde': sin_v.value, 'todo_verde': todo_v.value, 'puro_desde': corte.value / 100}
    cmp_ = mr.comparar_pureza([{'pureza': c, 'ndvi': v} for c, v in de_borr], par)
    rev = tabla_con_columna(at, 'Pureza revisada', 'Reviewed purity')
    comprobar(rev is not None and len(rev) == 3
              and int(rev.iloc[:, 1].sum() + rev.iloc[:, 2].sum()) == cmp_['n'] == 14
              and hay(at, T(f"**{cmp_['coinciden']} de tus 14 muestras**",
                            f"**{cmp_['coinciden']} of your 14 borreguil samples**")),
              'y se ve cómo quedan con esos valores las muestras cuya pureza se revisó',
              str(rev.values.tolist()) if rev is not None else 'no está')

    # con valores que no valen, se dice a la vista (no solo dentro del desplegable) y no
    # se puede hacer el mapa
    valor_todo = todo_v.value
    control(at, 'number_input', 'mrf_pureza_sin').set_value(0.45)
    control(at, 'number_input', 'mrf_pureza_todo').set_value(0.30)
    at.run()
    sin_error(at, 'P3b · valores que no valen')
    comprobar(any(T('el NDVI de «todo verde» tiene que ser mayor', 'the "all green" NDVI must be greater')
                  in str(e.value) for e in at.error)
              and control(at, 'button', 'mrf_hacer_mapa').disabled,
              'si «todo verde» no es mayor que «sin verde», lo dice y no deja hacer el mapa')
    control(at, 'number_input', 'mrf_pureza_sin').set_value(0.08)
    control(at, 'number_input', 'mrf_pureza_todo').set_value(valor_todo)
    at.run()
    comprobar(not control(at, 'button', 'mrf_hacer_mapa').disabled
              and not any(T('«todo verde»', '"all green"') in str(e.value) for e in at.error),
              'al corregirlo, se puede hacer el mapa')

    print('  P4 · hacer el mapa')
    msgs = pulsar(at, control(at, 'button', 'mrf_hacer_mapa'), 'P4')
    mapa = at.session_state['mrf_mapas'].get('humedad') if 'mrf_mapas' in at.session_state \
        else None
    comprobar(mapa is not None and len(BLOQUES) == 1 and len(PILAS) == 1,
              'se ha leído un bloque y hay un mapa de humedad', f'{len(BLOQUES)} bloques')
    if mapa is None:
        return
    meta, carpeta = mapa['meta'], Path(mapa['carpeta'])
    comprobar(PILAS[0] == {'years': tuple(range(2017, 2026)), 'epsg': 32630,
                           'bandas': mod['predictores']},
              'los píxeles se piden con los años, la rejilla y las variables del modelo, '
              'en su orden', str(PILAS[0])[:120])
    rej = meta['rejilla']
    comprobar(BLOQUES[0] == (rej['x0'], rej['y0'], rej['ancho'], rej['alto'])
              and rej['x0'] % 10 == 0 and rej['y0'] % 10 == 0
              and 50 <= rej['ancho'] <= 51 and 40 <= rej['alto'] <= 41,
              'el bloque es la rejilla de 10 m, alineada, de unos 50 × 40 píxeles', str(rej))
    comprobar(all((carpeta / n).exists() for n in ('clase.tif', 'pmax.tif',
                                                   'probabilidades.tif',
                                                   'probabilidad_borreguil.tif',
                                                   'pureza.tif', 'verde_estimado.tif',
                                                   'codigos.csv', 'mapa.json')),
              'quedan los seis GeoTIFF, los códigos y la ficha del mapa')
    u_mod = mod['regla']['umbral']
    comprobar(meta.get('regla') == {'criterio': 'f1', 'umbral': u_mod} and u_mod in mr.UMBRALES
              and hay(at, T(f'Borreguil si la suma de la probabilidad de sus tipos llega a '
                            f'{N(u_mod, en)}.',
                            f'Borreguil if the probabilities of its types add up to '
                            f'{N(u_mod, en)} or more.')),
              'el mapa se hace con el umbral del modelo, y lo dice', str(meta.get('regla')))
    def leer_tif(ruta, banda=None):
        """Los valores de un GeoTIFF del mapa: una banda (alto, ancho) o todas."""
        with mt.abrir(ruta) as f:
            return f.leer(banda)

    r_clase = leer_tif(carpeta / 'clase.tif', 1)
    r_pb = leer_tif(carpeta / 'probabilidad_borreguil.tif', 1)
    r_prob = leer_tif(carpeta / 'probabilidades.tif')
    pz = meta.get('pureza') or {}
    r_pur = leer_tif(carpeta / 'pureza.tif', 1)
    r_verde = leer_tif(carpeta / 'verde_estimado.tif', 1)
    borr = r_clase != 0
    comprobar({k: pz.get(k) for k in par} == par
              and [c['categoria'] for c in pz['categorias']] == ['puro', 'mixto-agua', 'mixto-roca']
              and sum(c['pixeles'] for c in pz['categorias']) == int(borr.sum()),
              'el mapa lleva la pureza de cada píxel de borreguil, con los valores elegidos',
              str({k: pz.get(k) for k in par}))
    comprobar(np.isfinite(r_verde).all() and float(r_verde.min()) >= 0 and float(r_verde.max()) <= 1
              and (r_pur[~borr] == 255).all()
              and np.array_equal(r_pur[borr] == 1, r_verde[borr] >= np.float32(par['puro_desde']))
              and set(np.unique(r_pur[borr])) <= {1, 2, 3},
              'puro donde el verde estimado llega al corte, mixto donde no, y nada donde no '
              'es borreguil')
    comprobar(np.array_equal(r_pb, r_prob[1] + r_prob[2])
              and np.array_equal(r_clase != 0, r_pb >= np.float32(u_mod))
              and np.array_equal(r_clase[r_clase != 0],
                                 np.where(r_prob[2] > r_prob[1], 2, 1)[r_clase != 0]),
              'en cada píxel: borreguil si la suma de sus tipos llega al umbral, y entonces '
              'su tipo más probable')
    comprobar(meta['pixeles_clasificados'] == meta['pixeles'] == rej['ancho'] * rej['alto']
              and [c['categoria'] for c in meta['categorias']] == ['no borreguil', 'húmedo', 'seco']
              and meta['firma_modelo'] == mod['firma'] and meta['anios'] == list(range(2017, 2026))
              and meta['sin_clase_no_borreguil'] is False,
              'la ficha dice qué se ha clasificado, con qué modelo y qué años')
    comprobar(meta.get('pila') == {'version': geb.VERSION_PILA,
                                   'estadisticas': geb.ESTADISTICAS_PILA},
              'y cómo se calcularon las estadísticas de jun–sep: año a año',
              str(meta.get('pila')))
    comprobar(any(T('Mapa hecho en', 'Map made in') in m for m in msgs),
              'avisa de que el mapa está hecho')
    ha = meta['pixeles_clasificados'] / 100
    comprobar(hay(at, T(f'4 · Mapa · {N(ha, en)} ha clasificadas',
                        f'4 · Map · {N(ha, en)} ha classified')),
              'estado: mapa hecho, con su superficie')
    # el mapa dice en el píxel de cada muestra lo que el modelo predice para ella
    dentro = [m for m in mapa['muestras'] if m['dentro']]
    por_id = {m['ID']: m for m in mapa['muestras']}
    nombres = {c['codigo']: c['categoria'] for c in mod['categorias']}
    coincide = True
    for m in dentro:
        tr = pyproj.Transformer.from_crs(4326, 32630, always_xy=True)
        x, y = tr.transform(m['lon'], m['lat'])
        cx, cy = math.floor(x / 10) * 10 + 5, math.floor(y / 10) * 10 + 5
        fila = [[_valor_pixel(cx, cy, list(geb.BANDAS_PILA).index(b))
                 for b in mod['predictores']]]
        pr = mod['modelo'].predict_proba(np.asarray(fila, dtype='float32'))
        cat, p_cat, _ = mr.decidir(pr, mod['modelo'].classes_, mod['regla']['umbral'])
        coincide &= (por_id[m['ID']]['mapa'] == nombres[int(cat[0])]
                     and por_id[m['ID']]['p_mapa'] == float(p_cat[0]))
    comprobar(len(dentro) >= 1 and coincide,
              'en el píxel de cada muestra, el mapa dice lo que el modelo predice para ella',
              f'{len(dentro)} muestras dentro')

    print('  P5 · ver el mapa')
    etiquetas = [m.label for m in at.metric]
    comprobar(all(x in etiquetas for x in (
        T('Superficie clasificada', 'Classified area'),
        T('Probabilidad media de la categoría elegida', 'Mean probability of the chosen category'),
        T('Píxeles de la zona sin dato', 'Pixels of the zone without data'))),
        'se resumen la superficie, la confianza media y lo que queda sin dato',
        str(etiquetas[-3:]))
    sup = tabla_con_columna(at, 'Hectáreas', 'Hectares')
    comprobar(sup is not None and len(sup) == 3
              and abs(float(sup.iloc[:, 2].sum()) - ha) < 0.02
              and abs(float(sup.iloc[:, 3].sum()) - 100) < 0.01,
              'la tabla da las hectáreas y el porcentaje de cada categoría')
    tp = tabla_con_columna(at, '% del borreguil', '% of the borreguil')
    comprobar(tp is not None and len(tp) == 3
              and list(tp.iloc[:, 0]) == [T('puro', 'pure'), T('mixto-agua', 'mixed-water'),
                                          T('mixto-roca', 'mixed-rock')]
              and abs(float(tp.iloc[:, 2].sum()) - int(borr.sum()) / 100) < 0.02
              and abs(float(tp.iloc[:, 3].sum()) - 100) < 0.01
              and any(T('puro desde el 75 % de verde', 'pure from 75 % green') in str(c.value)
                      for c in at.caption),
              'otra tabla da las hectáreas de borreguil puro y mixto, y dice con qué valores',
              str(tp.values.tolist()) if tp is not None else 'no está')
    v = _visor('mrf_visor_')
    comprobar(v is not None and v['devuelve'] == ['last_clicked', 'bounds', 'zoom']
              and v['opciones'].get('pixelated') is True,
              'el visor avisa de dónde se pincha y de lo que está mostrando',
              str(v['devuelve']) if v else 'no está')
    h = v['html'] if v else ''
    # Lo fijo: las capas, vacías, en el recuadro del mapa, y su leyenda. Las muestras
    # ya no van en este visor (pidió quitarlas).
    comprobar(all(_capa(x) in h for x in (T('Categoría', 'Category'),
                                          T('Probabilidad máxima', 'Maximum probability'),
                                          T('Probabilidad de borreguil',
                                            'Probability of borreguil'),
                                          T('Pureza dentro del píxel',
                                            'Purity within the pixel')))
              and '"' + _capa(T('Muestras', 'Samples')) + '"' not in h,
              'en el recuadro del mapa están la categoría, la probabilidad máxima, la de '
              'borreguil y la pureza; las muestras, no')
    # (entre comillas: así es como van escritos los nombres en el recuadro de capas, y
    # no se confunden con los rótulos de la leyenda)
    orden = [h.index('"' + _capa(x) + '"') for x in (
        T('Categoría', 'Category'), T('Probabilidad máxima', 'Maximum probability'),
        T('Probabilidad de borreguil', 'Probability of borreguil'),
        T('Pureza dentro del píxel', 'Purity within the pixel'))]
    comprobar(orden == sorted(orden),
              'las capas nuevas van detrás de las que ya había, que siguen donde estaban',
              str(orden))
    comprobar(h.count(mra.PNG_VACIO) == 5 and h.count('data:image/png') == 5
              and 'window.capas_mrf = {mapa: map_' in h and ', pborr: image_overlay_' in h
              and ', pureza: image_overlay_' in h,
              'las capas ráster son fijas y nacen vacías: su imagen se pone aparte',
              f"{h.count('data:image/png')} imágenes en lo fijo")
    comprobar(all(x in h for x in (
        '>' + T('Categoría', 'Category') + '</div>',
        mra.COLORES['húmedo'], mra.COLORES['seco'], mra.COLORES['no borreguil'],
        T('>Probabilidad<', '>Probability<'), 'linear-gradient(to right,#440154')),
        'con su leyenda: los colores de las categorías y la escala de probabilidad')
    comprobar(all(x in h for x in (
        T('>Pureza dentro del píxel<', '>Purity within the pixel<'),
        mra.COLORES['mixto-roca'], mra.COLORES['mixto-agua'],
        '</span>' + T('mixto-roca', 'mixed-rock') + '</div>')),
        'y los colores de la pureza')
    # (la frase de debajo del visor; la del mapa pequeño de la zona sí habla de sus puntos)
    debajo = [str(c.value) for c in at.caption
              if T('Lo pintado son los píxeles', 'What is drawn are the') in str(c.value)]
    comprobar('circle_marker' not in h and T('mapa y muestras', 'map and samples') not in h
              and len(debajo) == 1
              and T('Los puntos son las muestras', 'The points are the samples') not in debajo[0]
              and T('Que el mapa coincida con las muestras',
                    'The map agreeing with the samples') in debajo[0]
              and 'Esri' in h and 'Google' in h,
              'el visor ya no pinta las muestras (ni las nombra la leyenda), sigue el aviso '
              'de que coincidir con ellas no mide el acierto, y va sobre la ortofoto con su '
              'atribución')
    # Lo que cambia: la imagen de cada capa para lo que se está viendo.
    cambia = v['cambia'] if v else None
    comprobar(cambia is not None
              and sorted(cambia['datos']) == ['clase', 'pborr', 'pmax', 'pureza']
              and all(x[0].startswith('data:image/png;base64,') and x[0] != mra.PNG_VACIO
                      for x in cambia['datos'].values())
              and len({x[0] for x in cambia['datos'].values()}) == 4
              and cambia['opacidad'] == 0.7 and cambia['elegida'] == '' and not v['otros'],
              'se les pone la imagen de la categoría, la de la probabilidad máxima, la de '
              'la probabilidad de borreguil y la de la pureza',
              str({k: cambia[k] for k in ('opacidad', 'elegida')}) if cambia else 'no hay')
    (s_i, o_i), (n_i, e_i) = cambia['datos']['clase'][1]
    o_m, s_m, e_m, n_m = meta['limites_lonlat']
    comprobar(max(abs(o_i - o_m), abs(s_i - s_m), abs(e_i - e_m), abs(n_i - n_m)) < 2e-4
              and cambia['datos']['pmax'][1] == cambia['datos']['clase'][1],
              'de entrada se pinta el mapa entero, en su sitio',
              str([o_i - o_m, s_i - s_m, e_i - e_m, n_i - n_m]))
    capa = control(at, 'selectbox', 'mrf_ver_prob')
    comprobar(capa is not None and capa.value == '' and list(capa.options) == [
        T('— ninguna —', '— none —'), T('no borreguil', 'not a borreguil'),
        T('húmedo', 'wet'), T('seco', 'dry')]
        and T('recuadro de capas del mapa', 'layer box of the map') in (capa.help or ''),
        'además se puede pedir la probabilidad de una categoría concreta',
        str(capa.options) if capa else 'no está')
    comprobar(any(all(x in str(c.value) for x in (
        T('**Probabilidad máxima**', '**Maximum probability**'),
        T('sea la que sea', 'whichever it is'),
        T('está seguro de que **no** lo es', 'sure that it is **not** one')))
        for c in at.caption),
        'debajo se explica la probabilidad máxima: en «no borreguil», seguro de que no lo es')
    comprobar(any(all(x in str(c.value) for x in (
        T('**Probabilidad de borreguil** es la suma de la de sus tipos',
          '**Probability of borreguil** is the sum of those of its types'),
        N(u_mod, en))) for c in at.caption),
        'y la probabilidad de borreguil: la suma de los tipos, con su umbral')
    comprobar(any(T('**Pureza dentro del píxel** dice, de cada píxel de borreguil',
                    '**Purity within the pixel** tells, for each borreguil pixel')
                  in str(c.value) for c in at.caption),
              'y la pureza: si cada píxel de borreguil es puro o está mezclado')
    comprobar(any(T('Pincha en el mapa', 'Click on the map') in str(i.value) for i in at.info),
              'invita a pinchar un píxel')
    desc = {d.proto.label.split(' (')[0]: d for d in at.get('download_button')}
    comprobar(all(f'⬇ mapa_humedad_{x}' in desc for x in (
        'categoria.tif', 'probabilidad_maxima.tif', 'probabilidad_por_categoria.tif',
        'probabilidad_borreguil.tif', 'pureza.tif', 'verde_estimado.tif', 'codigos.csv')),
        'se pueden descargar los seis GeoTIFF y los códigos',
        str([k for k in desc if 'mapa_' in k]))

    print('  P6 · pinchar un píxel')
    m0 = dentro[0]
    CLICS['mrf_visor_'] = (m0['lon'], m0['lat'])
    at.run()
    at.run()
    sin_error(at, 'P6')
    comprobar(hay(at, T(f"**Píxel pinchado:** «{m0['mapa']}»",
                        f"**Clicked pixel:** \"{i18n_valor(m0['mapa'], en)}\""))
              and hay(at, N(m0['p_mapa'], en)),
              'dice la categoría de ese píxel y su probabilidad')
    q0 = mra.consultar(carpeta, m0['lon'], m0['lat'])
    comprobar(q0['p_borreguil'] is not None
              and hay(at, T('Probabilidad de borreguil (la suma de sus tipos): **',
                            'Probability of borreguil (the sum of its types): **')
                      + N(q0['p_borreguil'], en) + '**')
              and (q0['codigo'] != 0) == (np.float32(q0['p_borreguil']) >= np.float32(u_mod)),
              'y su probabilidad de borreguil, junto al umbral', str(q0.get('p_borreguil')))
    comprobar(q0['verde'] is not None
              and hay(at, T('Verde dentro del píxel (estimado): **',
                            'Green within the pixel (estimated): **')
                      + f"{100 * q0['verde']:.0f} %**")
              and (q0['pureza'] is None) == (q0['codigo'] == 0)
              and (q0['pureza'] is None
                   or hay(at, T('pureza por ese verde: **', 'purity from that green: **')
                          + i18n_valor(q0['pureza'], en) + '**')),
              'y el verde que se estima dentro del píxel; si es borreguil, su pureza',
              f"{q0.get('verde')} · {q0.get('pureza')}")
    pr = tabla_con_columna(at, 'Probabilidad', 'Probability')
    comprobar(pr is not None and len(pr) == 3 and abs(float(pr.iloc[:, 2].sum()) - 1) < 1e-4
              and list(pr.iloc[:, 2]) == sorted(pr.iloc[:, 2], reverse=True),
              'y la probabilidad de cada categoría, de mayor a menor; suman 1')
    v = _visor('mrf_visor_')
    comprobar(v is not None and len(v['otros']) == 1
              and sorted(v['cambia']['datos']) == ['clase', 'pborr', 'pmax', 'pureza'],
              'el píxel pinchado queda marcado en el visor, encima de las capas')
    CLICS.clear()

    print('  P7 · la probabilidad de una categoría entra en el recuadro del mapa')
    antes = _visor('mrf_visor_')
    fijo = _fijo(antes['html'])
    url_clase, url_pmax = antes['cambia']['datos']['clase'][0], antes['cambia']['datos']['pmax'][0]
    control(at, 'selectbox', 'mrf_ver_prob').select('seco')
    control(at, 'slider', 'mrf_opacidad').set_value(0.5)
    at.run()
    sin_error(at, 'P7 · probabilidad de seco')
    v = _visor('mrf_visor_')
    cambia = v['cambia']
    comprobar(sorted(cambia['datos']) == ['clase', 'pborr', 'pmax', 'pureza', 'px']
              and cambia['elegida'] == 'p2'
              and cambia['nombre'] == T('Probabilidad de «seco»', 'Probability of "dry"')
              and cambia['datos']['px'][0] not in (url_clase, url_pmax, mra.PNG_VACIO),
              'se añade la capa «Probabilidad de «seco»», con su imagen',
              str([sorted(cambia['datos']), cambia['elegida'], cambia['nombre']]))
    comprobar(cambia['opacidad'] == 0.5 and cambia['datos']['clase'][0] == url_clase
              and cambia['datos']['pmax'][0] == url_pmax and len(v['otros']) == 1,
              'cambia la opacidad, y las otras dos capas y el píxel marcado siguen igual')
    comprobar(_fijo(v['html']) == fijo,
              'lo fijo del visor no cambia: no se recarga ni pierde el zoom ni lo encendido')
    control(at, 'selectbox', 'mrf_ver_prob').select('')
    at.run()
    sin_error(at, 'P7 · ninguna')
    v = _visor('mrf_visor_')
    comprobar(sorted(v['cambia']['datos']) == ['clase', 'pborr', 'pmax', 'pureza']
              and v['cambia']['elegida'] == '' and _fijo(v['html']) == fijo
              and sorted(mapa['vistas']) == ['clase', 'p2', 'pborr', 'pmax', 'pureza'],
              'al volver a «ninguna» esa capa se retira; lo ya dibujado se reutiliza',
              str(sorted(mapa['vistas'])))

    print('  P7b · al acercarse se pinta lo que se ve, a la resolución de la pantalla')
    clave = [k for k in VISORES if k and k.startswith('mrf_visor_')][-1]
    CLICS['mrf_visor_'] = {
        'last_clicked': {'lat': m0['lat'], 'lng': m0['lon']}, 'zoom': 19,
        'bounds': {'_southWest': {'lat': m0['lat'] - 0.0004, 'lng': m0['lon'] - 0.0006},
                   '_northEast': {'lat': m0['lat'] + 0.0004, 'lng': m0['lon'] + 0.0006}}}
    at.run()
    at.run()
    sin_error(at, 'P7b')
    v = _visor('mrf_visor_')
    vistas = at.session_state['mrf_mapas']['humedad']['vistas']
    ult = vistas['clase']['ultima']
    (s_i, o_i), (n_i, e_i) = v['cambia']['datos']['clase'][1]
    comprobar(ult['zoom'] == 19 and v['cambia']['datos']['clase'][0] == ult['url'] != url_clase
              and (e_i - o_i) < 0.8 * (e_m - o_m) and o_i <= m0['lon'] <= e_i
              and s_i <= m0['lat'] <= n_i,
              'la imagen de la categoría pasa a ser la de la zona que se ve, a zoom 19',
              str([ult['zoom'], e_i - o_i, e_m - o_m]))
    comprobar(vistas['pmax']['ultima']['zoom'] == 19
              and v['cambia']['datos']['pmax'][0] == vistas['pmax']['ultima']['url'] != url_pmax
              and v['cambia']['datos']['pmax'][1] == v['cambia']['datos']['clase'][1],
              'y la de la probabilidad máxima también, esté encendida o no')
    comprobar(vistas['pborr']['ultima']['zoom'] == 19
              and v['cambia']['datos']['pborr'][0] == vistas['pborr']['ultima']['url']
              and v['cambia']['datos']['pborr'][1] == v['cambia']['datos']['clase'][1],
              'y la de la probabilidad de borreguil')
    # A esa resolución cada punto de la imagen mide unos 24 cm: los bordes de los píxeles
    # de 10 m caen donde son. Se comprueba descodificando la imagen.
    import base64
    import io

    from PIL import Image
    img = np.array(Image.open(io.BytesIO(base64.b64decode(ult['url'].split(',', 1)[1]))))
    x0, y0, x1, y1 = ult['cubre']
    res = (x1 - x0) / img.shape[1]
    a_utm = pyproj.Transformer.from_crs(3857, 32630, always_xy=True)
    rng, bien, vistos = np.random.default_rng(5), 0, 0
    clase = leer_tif(carpeta / 'clase.tif', 1)
    color_de = {c['codigo']: tuple(int(c['color'][k:k + 2], 16) for k in (1, 3, 5))
                for c in meta['categorias']}
    for _ in range(1500):
        j, i = int(rng.integers(0, img.shape[1])), int(rng.integers(0, img.shape[0]))
        x, y = a_utm.transform(x0 + (j + 0.5) * res, y1 - (i + 0.5) * res)
        col, fila = (x - rej['x0']) / 10, (rej['y0'] - y) / 10
        if min(col % 1, 1 - col % 1, fila % 1, 1 - fila % 1) < 0.005:
            continue
        if not (0 <= col < rej['ancho'] and 0 <= fila < rej['alto']):
            continue
        vistos += 1
        bien += tuple(int(v_) for v_ in img[i, j][:3]) == color_de[int(clase[int(fila), int(col)])]
    comprobar(abs(res - 0.2986) < 0.001 and vistos > 1000 and bien == vistos,
              'cada punto de la imagen lleva el color del píxel de 10 m que tiene debajo',
              f'{bien} de {vistos} · {res:.4f} m por punto')
    comprobar(clave in at.session_state and len(v['otros']) == 1 and _fijo(v['html']) == fijo,
              'el píxel pinchado sigue marcado y lo fijo del visor sigue sin cambiar')
    CLICS.clear()

    print('  P8 · repetir el mapa no vuelve a pedir lo ya leído')
    comprobar(hay(at, T('1 ya leído', '1 already read'))
              and hay(at, T('menos de 1 min', 'less than 1 min')),
              'la previsión cuenta con el bloque que ya está leído')
    pulsar(at, control(at, 'button', 'mrf_hacer_mapa'), 'P8')
    nuevo = at.session_state['mrf_mapas']['humedad']
    comprobar(len(BLOQUES) == 1 and nuevo['carpeta'] != mapa['carpeta']
              and not Path(mapa['carpeta']).exists() and Path(nuevo['carpeta']).exists(),
              'el mapa se rehace sin leer nada, y el anterior se borra', f'{len(BLOQUES)} bloques')

    print('  P8b · si después se cambia el umbral, el mapa avisa y se rehace sin leer nada')
    u_otro = 0.9 if u_mod < 0.7 else 0.1
    control(at, 'radio', 'mrf_umbral_modo').set_value('manual')
    at.run()
    control(at, 'slider', 'mrf_umbral_valor').set_value(u_otro)
    at.run()
    sin_error(at, 'P8b')
    comprobar(at.session_state['mrf_modelos']['humedad']['regla']['umbral'] == u_otro
              and any(T('Este mapa se hizo con otro umbral', 'This map was made with another')
                      in str(w.value) and N(u_mod, en) in str(w.value)
                      and N(u_otro, en) in str(w.value) for w in at.warning)
              and hay(at, T('4 · Mapa · es de otro umbral',
                            '4 · Map · made with another threshold')),
              'el mapa dice que se hizo con otro umbral, y cuáles son los dos')
    pulsar(at, control(at, 'button', 'mrf_hacer_mapa'), 'P8b · rehacer')
    m_u = at.session_state['mrf_mapas']['humedad']
    c_u = leer_tif(Path(m_u['carpeta']) / 'clase.tif', 1)
    pb_u = leer_tif(Path(m_u['carpeta']) / 'probabilidad_borreguil.tif', 1)
    comprobar(len(BLOQUES) == 1 and m_u['meta']['regla'] == {'criterio': 'manual', 'umbral': u_otro}
              and np.array_equal(pb_u, r_pb) and np.array_equal(c_u != 0, pb_u >= np.float32(u_otro))
              and int((c_u != 0).sum()) != int((r_clase != 0).sum())
              and not any(T('Este mapa se hizo con otro umbral', 'This map was made with another')
                          in str(w.value) for w in at.warning),
              'se rehace con el umbral nuevo sin pedir nada a Earth Engine: cambia qué es '
              'borreguil, no las probabilidades',
              f"{int((c_u != 0).sum())} px de borreguil; antes {int((r_clase != 0).sum())}")
    control(at, 'radio', 'mrf_umbral_modo').set_value('f1')
    at.run()
    pulsar(at, control(at, 'button', 'mrf_hacer_mapa'), 'P8b · vuelta')
    nuevo = at.session_state['mrf_mapas']['humedad']
    comprobar(len(BLOQUES) == 1 and nuevo['meta']['regla'] == {'criterio': 'f1', 'umbral': u_mod},
              'y al volver al umbral de la validación, el mapa vuelve a ser el de antes')

    print('  P8c · la pureza es opcional: sin pedirla, el mapa no la lleva')
    control(at, 'checkbox', 'mrf_pureza_si').set_value(False)
    at.run()
    sin_error(at, 'P8c')
    comprobar(control(at, 'number_input', 'mrf_pureza_sin').disabled
              and control(at, 'slider', 'mrf_pureza_corte').disabled,
              'al desmarcarla, sus valores quedan desactivados')
    pulsar(at, control(at, 'button', 'mrf_hacer_mapa'), 'P8c · sin pureza')
    m_sp = at.session_state['mrf_mapas']['humedad']
    v = _visor('mrf_visor_')
    desc = [d.proto.label for d in at.get('download_button')]
    comprobar(len(BLOQUES) == 1 and 'pureza' not in m_sp['meta']
              and not (Path(m_sp['carpeta']) / 'pureza.tif').exists()
              and not (Path(m_sp['carpeta']) / 'verde_estimado.tif').exists()
              and v['html'].count(mra.PNG_VACIO) == 4
              and _capa(T('Pureza dentro del píxel', 'Purity within the pixel')) not in v['html']
              and sorted(v['cambia']['datos']) == ['clase', 'pborr', 'pmax']
              and not any('pureza' in x or 'verde_estimado' in x for x in desc)
              and tabla_con_columna(at, '% del borreguil', '% of the borreguil') is None,
              'ese mapa no tiene ni la capa, ni la tabla, ni los ficheros de la pureza; y lo '
              'demás, igual')
    comprobar(np.array_equal(leer_tif(Path(m_sp['carpeta']) / 'clase.tif', 1), r_clase),
              'las categorías no cambian por pedirla o no')
    control(at, 'checkbox', 'mrf_pureza_si').set_value(True)
    control(at, 'slider', 'mrf_pureza_corte').set_value(60)
    at.run()
    pulsar(at, control(at, 'button', 'mrf_hacer_mapa'), 'P8c · con otro corte')
    nuevo = at.session_state['mrf_mapas']['humedad']
    pur60 = leer_tif(Path(nuevo['carpeta']) / 'pureza.tif', 1)
    comprobar(len(BLOQUES) == 1 and nuevo['meta']['pureza']['puro_desde'] == 0.6
              and np.array_equal(pur60[borr] == 1, r_verde[borr] >= np.float32(0.6))
              and int((pur60 == 1).sum()) >= int((r_pur == 1).sum())
              and any(T('puro desde el 60 % de verde', 'pure from 60 % green') in str(c.value)
                      for c in at.caption),
              'con un corte más bajo hay más píxeles puros, sin leer nada otra vez',
              f"{int((pur60 == 1).sum())} puros; con 75 %, {int((r_pur == 1).sum())}")
    control(at, 'slider', 'mrf_pureza_corte').set_value(75)
    at.run()
    pulsar(at, control(at, 'button', 'mrf_hacer_mapa'), 'P8c · vuelta')
    nuevo = at.session_state['mrf_mapas']['humedad']

    print('  P9 · si Earth Engine falla, no queda un mapa a medias')
    control(at, 'number_input', 'mrf_rect_ancho').set_value(0.6)
    at.run()
    FALLA_BLOQUE['activo'] = True
    control(at, 'button', 'mrf_hacer_mapa').click()
    at.run()
    FALLA_BLOQUE['activo'] = False
    sin_error(at, 'P9')
    comprobar(any(T('El mapa no se ha podido terminar', 'The map could not be finished')
                  in str(e.value) for e in at.error)
              and at.session_state['mrf_mapas']['humedad']['carpeta'] == nuevo['carpeta']
              and len(list(Path(at.session_state['mrf_dir']).glob('mapa_*'))) == 1,
              'lo dice, conserva el mapa anterior y no deja ficheros a medias')
    at.run()

    print('  P10 · una zona grande avisa de lo que tarda')
    control(at, 'radio', 'mrf_zona').set_value('muestras')
    at.run()
    sin_error(at, 'P10')
    zc = at.session_state['mrf_zona_calc']
    nb = len(zc.get('utiles') or [])
    comprobar(zc['error'] is None and nb > 20
              and hay(at, T(f'en {nb} trozos de Earth Engine', f'in {nb} Earth Engine pieces'))
              and hay(at, T('Va para largo', 'This will take a while')),
              'el rectángulo de las muestras son muchos bloques: lo dice y avisa de que '
              'va para largo', f'{nb} bloques')
    v = _visor('mrf_mapa_zona')
    comprobar(v is not None and v['devuelve'] == [],
              'y en esa zona pinchar en el mapa no mueve nada')
    control(at, 'radio', 'mrf_zona').set_value('rect')
    at.run()

    print('  P10b · el área de estudio como zona: solo se clasifica lo de dentro')
    # Un triángulo de 400 m de base y 300 m de alto (6 ha) alrededor de una muestra.
    from shapely.geometry import Polygon
    s0 = mod['muestras'][0]
    tr = pyproj.Transformer.from_crs(4326, 32630, always_xy=True)
    inv = pyproj.Transformer.from_crs(32630, 4326, always_xy=True)
    x, y = tr.transform(s0['lon'], s0['lat'])
    x, y = math.floor(x / 10) * 10, math.floor(y / 10) * 10
    at.session_state['study_geom'] = Polygon([inv.transform(x - 200, y - 150),
                                              inv.transform(x + 200, y - 150),
                                              inv.transform(x, y + 150)])
    at.run()
    sin_error(at, 'P10b')
    zona = control(at, 'radio', 'mrf_zona')
    comprobar(T('El área de estudio', 'The study area') in zona.options,
              'con un área de estudio, aparece como zona posible', str(zona.options))
    zona.set_value('area')
    at.run()
    sin_error(at, 'P10b · zona')
    comprobar(hay(at, T('**0,060 km²** que clasificar', '**0.060 km²** to classify'))
              and hay(at, T('en 1 trozo de Earth Engine', 'in 1 Earth Engine piece')),
              'la previsión es la del triángulo, no la de su rectángulo')
    pulsar(at, control(at, 'button', 'mrf_hacer_mapa'), 'P10b · hacer')
    m_area = at.session_state['mrf_mapas']['humedad']['meta']
    comprobar(m_area['con_zona'] and m_area['zona']['tipo'] == 'area'
              and 560 <= m_area['pixeles_clasificados'] <= 640
              and m_area['pixeles'] >= 1200
              and m_area['pixeles_zona'] == m_area['pixeles_clasificados'],
              'el mapa clasifica las 6 ha del triángulo y deja sin dato el resto del rectángulo',
              str({k: m_area.get(k) for k in ('pixeles', 'pixeles_clasificados', 'con_zona')}))
    del at.session_state['study_geom']
    at.run()
    control(at, 'radio', 'mrf_zona').set_value('rect')
    at.run()
    sin_error(at, 'P10b · vuelta')

    print('  P11 · sin la clase «no borreguil» (ni el modelo de sí o no) hace falta una máscara')
    control(at, 'radio', 'mrf_quien_decide').set_value('si_no')
    at.run()
    control(at, 'button', 'mrf_entrenar').click()
    at.run()
    sin_error(at, 'P11')
    mod2 = at.session_state['mrf_modelos']['humedad']
    comprobar([c['categoria'] for c in mod2['categorias']] == ['húmedo', 'seco'],
              'el modelo nuevo solo tiene húmedo y seco')
    comprobar(mod2['regla']['umbral'] is None
              and control(at, 'radio', 'mrf_umbral_modo') is None
              and any(T('no tiene la categoría «no borreguil»',
                        'has no "not a borreguil" category') in str(c.value)
                      for c in at.caption),
              'sin «no borreguil» no hay umbral que elegir, y se dice por qué')
    comprobar(hay(at, T('no tiene la clase «no borreguil»', 'has no "not a borreguil" class'))
              and any(T('Falta la máscara de zonas candidatas',
                        'The mask of candidate zones is missing') in str(e.value)
                      for e in at.error)
              and control(at, 'button', 'mrf_hacer_mapa').disabled,
              'sin máscara no deja hacer el mapa, y explica por qué')
    comprobar(hay(at, T('Este mapa se hizo con un modelo anterior',
                        'This map was made with a model older'))
              and hay(at, T('4 · Mapa · es de un modelo anterior',
                            '4 · Map · from a previous model')),
              'y el mapa que había avisa de que es de un modelo anterior')

    print('  P12 · con la máscara, solo se clasifica dentro')
    # Una máscara de 200 × 200 m alrededor de una muestra de borreguil.
    mb = mod2['muestras'][0]
    tr = pyproj.Transformer.from_crs(4326, 32630, always_xy=True)
    inv = pyproj.Transformer.from_crs(32630, 4326, always_xy=True)
    x, y = tr.transform(mb['lon'], mb['lat'])
    x, y = math.floor(x / 10) * 10, math.floor(y / 10) * 10
    anillo = [list(inv.transform(x + dx, y + dy))
              for dx, dy in ((-100, -100), (100, -100), (100, 100), (-100, 100), (-100, -100))]
    geojson = json.dumps({'type': 'FeatureCollection', 'features': [{
        'type': 'Feature', 'properties': {},
        'geometry': {'type': 'Polygon', 'coordinates': [anillo]}}]}).encode()
    control(at, 'file_uploader', 'mrf_mascara_fich').set_value(
        ('zonas_candidatas.geojson', geojson, 'application/geo+json'))
    at.run()
    sin_error(at, 'P12 · subir')
    comprobar(hay(at, T('Máscara leída: 0,040 km²', 'Mask read: 0.040 km²'))
              and not any(T('Falta la máscara', 'The mask of candidate zones is missing')
                          in str(e.value) for e in at.error),
              'la máscara se lee y se dice lo que mide')
    zona = control(at, 'radio', 'mrf_zona')
    comprobar(T('Todo lo que cubre la máscara', 'Everything the mask covers') in zona.options,
              'aparece la opción de cartografiar todo lo que cubre la máscara',
              str(zona.options))
    zona.set_value('mascara')
    at.run()
    sin_error(at, 'P12 · zona')
    comprobar(hay(at, T('**0,040 km²** que clasificar', '**0.040 km²** to classify'))
              and not control(at, 'button', 'mrf_hacer_mapa').disabled,
              'la zona se ciñe a la máscara y ya se puede hacer el mapa')
    v = _visor('mrf_mapa_zona')
    comprobar(v is not None and len(v['capa']._children) == 1 + 1 + len(mod2['muestras']),
              'el mapa de la zona dibuja también la máscara')
    antes = len(BLOQUES)
    pulsar(at, control(at, 'button', 'mrf_hacer_mapa'), 'P12 · hacer')
    m3 = at.session_state['mrf_mapas']['humedad']
    comprobar(len(BLOQUES) == antes + 1 and m3['meta']['pixeles_clasificados'] == 400
              and m3['meta']['con_mascara'] and m3['meta']['sin_clase_no_borreguil'] is True
              and m3['meta']['mascara'] == 'zonas_candidatas.geojson'
              and [c['categoria'] for c in m3['meta']['categorias']] == ['húmedo', 'seco'],
              'el mapa clasifica las 4 ha de la máscara, solo entre húmedo y seco',
              str({k: m3['meta'].get(k) for k in ('pixeles_clasificados', 'mascara')}))
    comprobar(any(T('no tiene la clase «no borreguil»', 'has no "not a borreguil" class')
                  in str(i.value) for i in at.info)
              and hay(at, T('4 · Mapa · 4,000 ha clasificadas', '4 · Map · 4.000 ha classified')),
              'y recuerda que ese mapa no demuestra que haya borreguil')
    v = _visor('mrf_visor_')
    desc = [d.proto.label for d in at.get('download_button')]
    comprobar(m3['meta']['regla'] == {'criterio': None, 'umbral': None}
              and not (Path(m3['carpeta']) / 'probabilidad_borreguil.tif').exists()
              and v['html'].count(mra.PNG_VACIO) == 4
              and _capa(T('Probabilidad de borreguil', 'Probability of borreguil')) not in v['html']
              and sorted(v['cambia']['datos']) == ['clase', 'pmax', 'pureza']
              and not any('probabilidad_borreguil' in x for x in desc)
              and not any(T('**Probabilidad de borreguil**', '**Probability of borreguil**')
                          in str(c.value) for c in at.caption),
              'ese mapa no tiene probabilidad de borreguil: ni capa, ni fichero, ni umbral')
    comprobar(sum(c['pixeles'] for c in m3['meta']['pureza']['categorias']) == 400
              and _capa(T('Pureza dentro del píxel', 'Purity within the pixel')) in v['html'],
              'pero sí la pureza: ahí todo lo clasificado es borreguil')

    print('  P13 · una máscara que no toca la zona')
    control(at, 'radio', 'mrf_zona').set_value('rect')
    at.run()
    control(at, 'number_input', 'mrf_rect_lon').set_value(round(mb['lon'] + 0.5, 5))
    at.run()
    sin_error(at, 'P13')
    comprobar(any(T('La máscara no toca la zona elegida',
                    'The mask does not touch the chosen zone') in str(e.value)
                  for e in at.error)
              and control(at, 'button', 'mrf_hacer_mapa').disabled,
              'lo dice y no deja hacer el mapa')

    print('  P14 · la pureza sola ya no se cartografía: el desplegable no la ofrece')
    comprobar('pureza' not in list(control(at, 'selectbox', 'mrf_nivel').options)
              and not any(T('Estás cartografiando la **pureza**', 'You are mapping **purity**')
                          in str(i.value) for i in at.info),
              'no hay nivel «Pureza», ni su aviso de las dos purezas')

    print('  P15 · el mapa completo: la humedad del modelo y el ambiente por cercanía')
    # Las mismas muestras, con el ambiente y la humedad revisados (7 «arroyo · húmedo»
    # y 7 «ladera · seco»), y un mundo con una laguna y una vaguada cerca de una de ellas.
    filas_p, k = [dict(r) for r in at.session_state['rows']], 0
    for r in filas_p:
        if bp.tipo_revisado(r):
            bp.set_tipo_revisado(r, ambiente=('arroyo' if k % 2 else 'ladera'),
                                 humedad=('húmedo' if k % 2 else 'seco'))
            k += 1
    at.session_state['rows'] = filas_p
    control(at, 'selectbox', 'mrf_nivel').select('ambiente_humedad')
    at.run()
    if control(at, 'radio', 'mrf_quien_decide') is not None:
        control(at, 'radio', 'mrf_quien_decide').set_value('este')
        at.run()
    sin_error(at, 'P15 · nivel')
    control(at, 'button', 'mrf_entrenar').click()
    at.run()
    sin_error(at, 'P15 · entrenar')
    mod_c = at.session_state['mrf_modelos'].get('ambiente_humedad') or {}
    comprobar(mod_c.get('nivel') == 'humedad'
              and [c['categoria'] for c in mod_c.get('categorias') or []]
              == ['no borreguil', 'húmedo', 'seco'],
              'el modelo de «ambiente y humedad» aprende la humedad: no borreguil, húmedo y seco',
              str([c['categoria'] for c in mod_c.get('categorias') or []]))
    comprobar(any(T('El **ambiente** (arroyo, laguna o ladera) no entra aquí',
                    'The **environment** (stream, lake or slope) is not part of them')
                  in str(c.value) for c in at.caption)
              and not hay(at, T('por pasos', 'in steps')),
              '«3 · Modelo» dice que sus cifras son de la humedad y que el ambiente va aparte')
    tr = pyproj.Transformer.from_crs(4326, 32630, always_xy=True)
    xb, yb = tr.transform(mb['lon'], mb['lat'])
    MUNDO['ambiente'] = {'lago': (xb + 150.0, yb), 'eje': xb - 150.0}
    cauces_osm_de_verdad = bp.cauces_osm
    bp.cauces_osm = lambda bbox, ruta_capa=None: (None, 'sin_conexion')
    try:
        control(at, 'radio', 'mrf_zona').set_value('rect')
        at.run()
        control(at, 'number_input', 'mrf_rect_lon').set_value(round(mb['lon'], 5))
        control(at, 'number_input', 'mrf_rect_lat').set_value(round(mb['lat'], 5))
        control(at, 'number_input', 'mrf_rect_ancho').set_value(0.7)     # otra rejilla: se lee de nuevo
        control(at, 'number_input', 'mrf_rect_alto').set_value(0.7)
        at.run()
        sin_error(at, 'P15 · zona')
        dist = [control(at, 'number_input', k) for k in
                ('mrf_amb_capa', 'mrf_amb_agua', 'mrf_amb_arroyo')]
        comprobar(all(d is not None for d in dist)
                  and [d.value for d in dist] == [120.0, 100.0, 30.0]
                  and all(T('Recomendado', 'Recommended') in (d.help or '') for d in dist)
                  and control(at, 'number_input', 'mrf_amb_ha') is None
                  and control(at, 'file_uploader', 'mrf_cauces_fich') is not None,
                  'el ambiente tiene su apartado: tres distancias y una capa de cauces '
                  'opcional, cada cosa con su recomendación (ya sin la superficie de los '
                  'cauces del relieve)', str([getattr(d, 'value', None) for d in dist]))
        comprobar(any(T('Lagunas: 65 puntos de la capa de la app', 'Lakes: 65 points of the '
                        'layer of the app') in str(c.value)
                      and T('los ríos y arroyos de OpenStreetMap',
                            'the OpenStreetMap rivers and streams') in str(c.value)
                      for c in at.caption),
                  'y dice de dónde salen las lagunas y los cauces (OpenStreetMap)')
        vz = _visor('mrf_mapa_zona')
        de_ambiente = '<span style="display:inline-block;min-width:3.6em;">{}</span>'
        comprobar(vz is not None
                  and de_ambiente.format(T('arroyo', 'stream')) in vz['html']
                  and de_ambiente.format(T('ladera', 'slope')) in vz['html']
                  and vz['html'].count('<span style="margin-right:7px;">') == 2
                  and mra.color('arroyo · húmedo') in vz['html']
                  and mra.color('ladera · seco') in vz['html'],
                  'las muestras se siguen pintando con lo que se revisó entero (ambiente y '
                  'humedad), con la leyenda por ambientes')
        pilas_antes, bloques_antes = len(PILAS), len(BLOQUES)
        pulsar(at, control(at, 'button', 'mrf_hacer_mapa'), 'P15 · hacer')
        m_c = at.session_state['mrf_mapas'].get('ambiente_humedad') or {}
        meta_c = m_c.get('meta') or {}
        amb_c = meta_c.get('ambiente') or {}
        comprobar([c['categoria'] for c in meta_c.get('categorias') or []]
                  == ['no borreguil'] + list(mr.NIVELES['ambiente_humedad'])
                  and [c['categoria'] for c in meta_c.get('categorias_modelo') or []]
                  == ['no borreguil', 'húmedo', 'seco'],
                  'el mapa lleva las categorías compuestas, y aparte las del modelo',
                  str([c['categoria'] for c in meta_c.get('categorias') or []]))
        comprobar(amb_c.get('cauces') == 'relieve' and amb_c.get('margen_m') == 1000
                  and amb_c.get('lagunas_capa') == 65
                  and (amb_c.get('laguna_capa_m'), amb_c.get('laguna_agua_m'),
                       amb_c.get('arroyo_m'), amb_c.get('cauce_ha')) == (120.0, 100.0, 30.0, 8.0)
                  and 15 <= amb_c.get('pixeles_agua', 0) <= 30
                  and amb_c.get('pixeles_cauce', 0) >= 60,
                  'el ambiente se ha puesto por cercanía: ve la laguna y saca el cauce del '
                  'relieve', str({k: amb_c.get(k) for k in ('cauces', 'margen_m', 'pixeles_agua',
                                                            'pixeles_cauce')}))
        comprobar(len(PILAS) == pilas_antes + 2 and PILAS[-1]['bandas'] == ['elev_dem_m']
                  and len(BLOQUES) > bloques_antes + 1,
                  'el relieve se ha leído aparte, con margen alrededor de la zona',
                  str(PILAS[-1]['bandas']))
        carp_c = Path(m_c['carpeta'])
        clase_c = leer_tif(carp_c / 'clase.tif', 1)
        ambi_c = leer_tif(carp_c / 'ambiente.tif', 1)
        dist_c = leer_tif(carp_c / 'distancias_ambiente.tif')
        prob_c = leer_tif(carp_c / 'probabilidades.tif')
        u_c = mod_c['regla']['umbral']
        tipo_c = np.where(prob_c[1] + prob_c[2] >= np.float32(u_c),
                          np.where(prob_c[2] > prob_c[1], 2, 1), 0)
        # el agua de laguna, si se quitó (la casilla viene marcada), no es borreguil
        if (carp_c / mra.FICHEROS['agua']).exists():
            tipo_c = np.where(leer_tif(carp_c / mra.FICHEROS['agua'], 1) == 1, 0, tipo_c)
        comprobar(np.array_equal(ambi_c, mam.decidir(dist_c[0], dist_c[1], dist_c[2]))
                  and np.array_equal(clase_c, mr.componer('ambiente_humedad', ambi_c, tipo_c))
                  and len(np.unique(ambi_c)) == 3,
                  'los ficheros casan: el ambiente sale de las distancias y la categoría, de '
                  'ese ambiente con la humedad del modelo', str(np.unique(ambi_c)))
        ramas_c = meta_c.get('ramas') or []
        comprobar([r['rama'] for r in ramas_c] == ['arroyo', 'laguna', 'ladera']
                  and all(r['pixeles'] > 0 for r in ramas_c)
                  and hay(at, T('**Por ambiente** (húmedo más seco): ',
                                '**By environment** (wet plus dry): ')
                          + ' · '.join(f"{i18n_valor(r['rama'], en)} **{N(r['hectareas'], en)} ha**"
                                       for r in ramas_c) + '.'),
                  'hay borreguil de los tres ambientes, y debajo de la tabla va cuánto de cada uno',
                  str(ramas_c))
        comprobar(any(T('El ambiente no sale del modelo, sino de la cercanía',
                        'The environment does not come from the model') in str(c.value)
                      and T('deducidos del relieve', 'derived from the relief') in str(c.value)
                      and T('OpenStreetMap no respondió', 'OpenStreetMap did not answer')
                      in str(c.value)
                      for c in at.caption),
                  'y se dice con qué regla y con qué cauces se ha puesto: del relieve, porque '
                  'OpenStreetMap no respondió')
        dentro_c = [m for m in m_c['muestras'] if m.get('ambiente') and m.get('ambiente_mapa')]
        rev = tabla_con_columna(at, 'Ambiente revisado', 'Reviewed environment')
        comprobar(dentro_c and rev is not None and len(rev) == 3
                  and int(rev.iloc[:, 1:].to_numpy().sum()) == len(dentro_c)
                  and hay(at, T(f'De las **{len(dentro_c)} muestras** de borreguil',
                                f'Of the **{len(dentro_c)} borreguil samples**')),
                  'compara el ambiente por cercanía con el que se revisó en las muestras del mapa',
                  str(len(dentro_c)))
        v = _visor('mrf_visor_')
        titulo = '<span style="display:inline-block;min-width:3.6em;">{}</span>'
        comprobar(v is not None
                  and all(titulo.format(i18n_valor(a, en)) in v['html']
                          for a in ('arroyo', 'laguna', 'ladera'))
                  and v['html'].count('<span style="margin-right:7px;">') == 6
                  and all(mra.color(c) in v['html'] for c in mr.NIVELES['ambiente_humedad']),
                  'la leyenda del visor va por ambientes, cada uno con su húmedo y su seco')
        comprobar(_capa(T('Lagunas y cauces', 'Lakes and watercourses')) in v['html']
                  and sorted(v['cambia']['datos']) == ['clase', 'pborr', 'pmax', 'pureza', 'refs']
                  and T('laguna de la capa', 'lake of the layer') not in v['html']
                  and 'agua de laguna' not in v['html'] and 'lake water' not in v['html']
                  and any(T('en turquesa, el agua de laguna', 'turquoise: the lake water')
                          in str(c.value) for c in at.caption),
                  'y el visor tiene la capa «Lagunas y cauces»; no ocupa sitio en la leyenda '
                  'del mapa: sus colores se dicen debajo')
        opciones = list(control(at, 'selectbox', 'mrf_ver_prob').options)
        comprobar(len(opciones) == 4,
                  'la probabilidad por categoría es la de las tres del modelo', str(opciones))
        desc = [d.proto.label for d in at.get('download_button')]
        comprobar(all(any(x in d for d in desc) for x in (
            'mapa_ambiente_humedad_ambiente.tif', 'mapa_ambiente_humedad_distancias_ambiente.tif',
            'mapa_ambiente_humedad_lagunas_y_cauces.tif')),
            'se pueden descargar el ambiente, las distancias y lo tomado por laguna y cauce')
        # pinchar el píxel de la muestra: dice su ambiente y a cuánto está de cada cosa
        q0 = mra.consultar(carp_c, mb['lon'], mb['lat'])
        CLICS['mrf_visor_'] = (q0['lon'], q0['lat'])
        at.run()
        at.run()
        sin_error(at, 'P15 · píxel')
        comprobar(q0['ambiente'] in ('arroyo', 'laguna', 'ladera')
                  and hay(at, T('Ambiente por cercanía: **', 'Environment by closeness: **')
                          + i18n_valor(q0['ambiente'], en) + '** (')
                  and hay(at, T(f"a {q0['distancias']['cauce']:.0f} m de un cauce",
                                f"{q0['distancias']['cauce']:.0f} m from a watercourse"))
                  and 110 <= q0['distancias']['agua'] <= 135
                  and 140 <= q0['distancias']['cauce'] <= 160,
                  'al pinchar un píxel dice su ambiente y a cuántos metros está de la laguna y '
                  'del cauce', str(q0.get('distancias')))
        CLICS.clear()

        print('  P15b · con OpenStreetMap, sus ríos y arroyos (y no se lee el relieve)')
        inv_o = pyproj.Transformer.from_crs(32630, 4326, always_xy=True)
        from shapely.geometry import LineString as _Linea
        arroyo_osm = _Linea([inv_o.transform(xb - 150.0, yb - 900.0),
                             inv_o.transform(xb - 150.0, yb + 900.0)])
        bp.cauces_osm = lambda bbox, ruta_capa=None: ([arroyo_osm], 'capa')
        pilas_antes = len(PILAS)
        pulsar(at, control(at, 'button', 'mrf_hacer_mapa'), 'P15b · con OpenStreetMap')
        m_o = at.session_state['mrf_mapas']['ambiente_humedad']
        a_o = m_o['meta']['ambiente']
        q_o = mra.consultar(m_o['carpeta'], mb['lon'], mb['lat'])
        comprobar(a_o['cauces'] == 'capa'
                  and a_o['fuentes']['cauces'] == 'OpenStreetMap (capa de la app)'
                  and all(b['bandas'] != ['elev_dem_m'] for b in PILAS[pilas_antes:])
                  and 140 <= q_o['distancias']['cauce'] <= 160
                  and any(T('Cauces: los ríos y arroyos de OpenStreetMap.',
                            'Watercourses: the OpenStreetMap rivers and streams.')
                          in str(c.value) for c in at.caption),
                  'el mapa usa los ríos y arroyos de OpenStreetMap (la muestra, a 150 m del '
                  'arroyo), sin pedir el relieve, y lo dice', str(q_o.get('distancias')))

        print('  P16 · cambiar una distancia avisa; con una capa de cauces, mandan sus líneas')
        control(at, 'number_input', 'mrf_amb_arroyo').set_value(160.0)
        at.run()
        sin_error(at, 'P16 · distancia')
        comprobar(any(T('Este mapa se hizo con otras distancias de ambiente',
                        'This map was made with other environment distances') in str(w.value)
                      for w in at.warning),
                  'si cambian las distancias, el mapa avisa de que es de otras')
        leidos = len(BLOQUES)
        pulsar(at, control(at, 'button', 'mrf_hacer_mapa'), 'P16 · rehacer')
        m_d = at.session_state['mrf_mapas']['ambiente_humedad']
        q1 = mra.consultar(m_d['carpeta'], mb['lon'], mb['lat'])
        comprobar(len(BLOQUES) == leidos and m_d['meta']['ambiente']['arroyo_m'] == 160.0
                  and q1['ambiente'] == ('laguna' if q0['ambiente'] == 'laguna' else 'arroyo')
                  and not any(T('otras distancias de ambiente', 'other environment distances')
                              in str(w.value) for w in at.warning),
                  'rehacerlo aplica la distancia nueva sin volver a leer nada: la muestra, a '
                  '150 m del cauce, pasa a ser de arroyo', f"{q0['ambiente']} → {q1['ambiente']}")
        control(at, 'number_input', 'mrf_amb_arroyo').set_value(30.0)
        at.run()
        # una capa que no son cauces (solo puntos): se dice a la vista y no deja hacer el mapa
        inv = pyproj.Transformer.from_crs(32630, 4326, always_xy=True)
        puntos = json.dumps({'type': 'FeatureCollection', 'features': [{
            'type': 'Feature', 'properties': {},
            'geometry': {'type': 'Point', 'coordinates': list(inv.transform(xb, yb))}}]}).encode()
        control(at, 'file_uploader', 'mrf_cauces_fich').set_value(
            ('fuentes.geojson', puntos, 'application/geo+json'))
        at.run()
        sin_error(at, 'P16 · capa que no vale')
        comprobar(any(T('no se ha podido leer la capa de cauces', 'the watercourse layer could '
                        'not be read') in str(e.value) for e in at.error)
                  and control(at, 'button', 'mrf_hacer_mapa').disabled,
                  'una capa sin líneas no vale: lo dice y no deja hacer el mapa')
        # una capa de verdad: una línea de oeste a este que pasa por la muestra
        linea = json.dumps({'type': 'FeatureCollection', 'features': [{
            'type': 'Feature', 'properties': {},
            'geometry': {'type': 'LineString', 'coordinates': [
                list(inv.transform(xb - 900, yb)), list(inv.transform(xb + 900, yb))]}}]}).encode()
        control(at, 'file_uploader', 'mrf_cauces_fich').set_value(
            ('cauces.geojson', linea, 'application/geo+json'))
        at.run()
        sin_error(at, 'P16 · capa')
        comprobar(not control(at, 'button', 'mrf_hacer_mapa').disabled
                  and any(T('Cauces: los de tu capa.', 'Watercourses: those of your layer.')
                          in str(c.value) for c in at.caption),
                  'con una capa de líneas se puede, y dice que los cauces son los suyos')
        pilas_antes = len(PILAS)
        pulsar(at, control(at, 'button', 'mrf_hacer_mapa'), 'P16 · con capa')
        m_e = at.session_state['mrf_mapas']['ambiente_humedad']
        a_e = m_e['meta']['ambiente']
        q2 = mra.consultar(m_e['carpeta'], mb['lon'], mb['lat'])
        comprobar(a_e['cauces'] == 'capa' and a_e['fuentes']['cauces'] == 'cauces.geojson'
                  and all(b['bandas'] != ['elev_dem_m'] for b in PILAS[pilas_antes:])
                  and q2['distancias']['cauce'] <= 10
                  and q2['ambiente'] in ('arroyo', 'laguna')
                  and any(T('Cauces: los de tu capa.', 'Watercourses: those of your layer.')
                          in str(c.value) for c in at.caption),
                  'el mapa usa esas líneas (la muestra queda sobre el cauce) y ni pide el relieve',
                  str(q2.get('distancias')))
    finally:
        MUNDO['ambiente'] = None
        bp.cauces_osm = cauces_osm_de_verdad


def i18n_valor(valor, en):
    import i18n
    return i18n.VALORES.get(valor, valor) if en else valor


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
    t0 = at.tabs[0]                    # la pantalla de inicio (sin el panel lateral)
    inicio = ' '.join(str(x.value) for x in
                      list(t0.markdown) + list(t0.info) + list(t0.caption))
    comprobar(T('Encuentra borreguiles en tu zona', 'Find borreguiles in your area') in inicio
              and T('No te preocupes', "Don't worry") in inicio
              and all(T(a, b) in inicio for a, b in (('1 · Tus puntos', '1 · Your points'),
                                                      ('2 · Analiza', '2 · Analyse'),
                                                      ('3 · Qué obtienes', '3 · What you get')))
              and T('Random Forest entrenado en Sierra Nevada',
                    'Random Forest trained in Sierra Nevada') in inicio
              and T('(AUC 0,833)', '(AUC 0.833)') in inicio
              and T('Sin cuenta, la app usa **Planetary Computer**',
                    'Without an account, the app uses **Planetary Computer**') in inicio
              and T('reentrenar el modelo de detección de puntos',
                    'retrain the point detection model') in inicio
              and T('mapeas el borreguil de tu zona de estudio',
                    'you map the borreguil of your study area') in inicio
              and not any(x in inicio for x in ('presence-only', 'GroupKFold', 'Backend',
                                                'inputs', 'MPC', 'rásters', 'rasters'))
              and not any('Placemark' in str(c.value) for c in t0.code),
              'la pantalla de inicio: qué hacer en tres pasos (para qué son los puntos y '
              'que, si no tienes, la app los genera) y qué se obtiene, sin jerga, '
              'con lo que acierta con y sin radar, y sin el recuadro del KML', inicio[:200])
    b_ej = boton(at, 'probar_ejemplo')
    comprobar(b_ej is not None
              and b_ej.label == T('▶ Probar con 20 puntos de ejemplo de Sierra Nevada',
                                  '▶ Try with 20 example points from Sierra Nevada'),
              'y un botón para probar con los 20 puntos de ejemplo')
    bajar = [d for d in at.get('download_button') if getattr(d, 'key', None) == 'bajar_plantilla'
             or 'bajar_plantilla' in str(getattr(d, 'proto', ''))]
    comprobar(len(bajar) == 1, 'y otro para descargar la plantilla de datos de campo',
              str(len(at.get('download_button'))))

    print('  Z1b · el botón de ejemplo analiza los 20 puntos de la plantilla')
    fuente = max((base[i] for i in BORR), key=lambda r: r['rf_proba'])
    leidos = []
    s1 = set(bp.S1_FEATS)
    de_imagen = [k for k in bp.FEATURES_RF if k not in geb.CLAVES_S2
                 and k not in geb.TOPO_KEYS and k not in s1]

    def copia(que, claves):
        def lector(rows, *a, **kw):
            leidos.append((que, len(rows)))
            for r in rows:
                for k in claves:
                    if k in fuente:
                        r[k] = fuente[k]
        return lector

    # Al analizar, la app inicia Earth Engine: aquí no hay proyecto, así que también
    # se simula (si no, pasaría a Planetary Computer y bajaría datos de verdad).
    cambios = {(geb, 'is_initialized'): lambda: True,
               (geb, 'initialize'): lambda project=None, service_account_json=None:
                   (True, 'simulado'),
               (geb, 'fetch_s2_gee'): copia('s2', geb.CLAVES_S2),
               (geb, 'fetch_topo_gee'): copia('topo', geb.TOPO_KEYS),
               (geb, 'fetch_s1_gee'): copia('s1', bp.S1_FEATS),
               (geb, 'fetch_pureza_gee'): copia('pureza', geb.CLAVES_PUREZA),
               (bp, 'fetch_esri_imgs'): copia('imagen', []),
               (bp, 'fetch_pnoa_imgs'): copia('imagen', []),
               (bp, 'compute_img_features'): copia('texturas', de_imagen),
               (streamlit_folium, 'st_folium'): _visor_simulado}
    originales = {k: getattr(*k) for k in cambios}
    for (mod, nombre), f in cambios.items():
        setattr(mod, nombre, f)
    try:
        ae = AppTest.from_file(str(APP / 'app.py'), default_timeout=600)
        ae.session_state['ui_lang'] = lang
        ae.run()
        [r for r in ae.radio if r.label == 'Backend'][0].set_value(
            '🌍  Google Earth Engine (requiere cuenta) — más rápido')
        ae.run()
        boton(ae, 'probar_ejemplo').click()
        ae.run()
        sin_error(ae, 'Z1b')
        filas_e = ae.session_state['rows'] if 'rows' in ae.session_state else []
        campo = Counter(bp._txt(r.get('Borreguil')) for r in filas_e)
        comprobar(len(filas_e) == 20 and campo == Counter({'si': 12, 'no': 8}),
                  'se analizan los 20 puntos de ejemplo, con su etiqueta de campo '
                  '(12 borreguiles y 8 no)', f'{len(filas_e)} · {dict(campo)}')
        comprobar({q for q, _ in leidos} >= {'s2', 'topo', 's1', 'pureza', 'imagen'}
                  and all(n == 20 for _, n in leidos),
                  'con la fuente del panel lateral: Sentinel-2, relieve, radar, imagen y '
                  'el píxel de la pureza', str(leidos))
        comprobar(any(r.get('pureza_regla') in bp.NIVELES_TIPO['pureza'] for r in filas_e),
                  'los borreguiles salen con su pureza propuesta')
        comprobar(not any(T('Encuentra borreguiles en tu zona', 'Find borreguiles in your '
                            'area') in str(m.value) for m in ae.tabs[0].markdown),
                  'y la pantalla de inicio ya no se ve')
    finally:
        for (mod, nombre), f in originales.items():
            setattr(mod, nombre, f)

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


def guion_anadir(lang):
    """Añadir puntos pinchando en el mapa de resultados (pestaña «Mapa de los puntos»), como en un
    taller con quien conoce el sitio. El visor y las lecturas son simulados: cada
    lectura copia las variables de un punto real (un borreguil o un no borreguil)."""
    en = lang == 'English'
    T = (lambda es_, en_: en_ if en else es_)
    print(f'\n================ {lang} · añadir en el mapa ================')
    SELECCION['rows'] = []
    VISORES.clear()
    CLICS.clear()
    os.environ['BORREGUIL_DATA_DIR'] = tempfile.mkdtemp(prefix='bt_')
    fuente = {'r': max((base[i] for i in BORR), key=lambda r: r['rf_proba'])}
    leidos = []
    s1 = set(bp.S1_FEATS)
    de_imagen = [k for k in bp.FEATURES_RF if k not in geb.CLAVES_S2
                 and k not in geb.TOPO_KEYS and k not in s1]

    def copia(que, claves):
        def lector(rows, *a, **kw):
            leidos.append((que, len(rows)))
            for r in rows:
                for k in claves:
                    if k in fuente['r']:
                        r[k] = fuente['r'][k]
        return lector

    cambios = {(geb, 'is_initialized'): lambda: True,
               (geb, 'fetch_s2_gee'): copia('s2', geb.CLAVES_S2),
               (geb, 'fetch_topo_gee'): copia('topo', geb.TOPO_KEYS),
               (geb, 'fetch_s1_gee'): copia('s1', bp.S1_FEATS),
               (geb, 'fetch_pureza_gee'): copia('pureza', geb.CLAVES_PUREZA),
               (bp, 'fetch_esri_imgs'): copia('imagen', []),
               (bp, 'fetch_pnoa_imgs'): copia('imagen', []),
               (bp, 'compute_img_features'): copia('texturas', de_imagen),
               (streamlit_folium, 'st_folium'): _visor_simulado}
    originales = {k: getattr(*k) for k in cambios}
    for (mod, nombre), f in cambios.items():
        setattr(mod, nombre, f)
    work = Path(tempfile.mkdtemp(prefix='bw_'))
    try:
        at = AppTest.from_file(str(APP / 'app.py'), default_timeout=600)
        at.session_state['rows'] = [dict(r) for r in base]
        at.session_state['work'] = work
        at.session_state['info'] = {k: v for k, v in (info or {}).items()
                                    if k != 'model_bundle'}
        at.session_state['ui_lang'] = lang
        at.session_state['active_backend'] = 'gee'
        at.run()
        sin_error(at, 'M1')
        tg = control(at, 'toggle', 'modo_anadir')
        comprobar(tg is not None and tg.value is False and boton(at, 'mapa_si') is None
                  and 'last_clicked' not in VISORES['mainmap']['devuelve'],
                  'el modo de añadir viene apagado: el mapa no recoge clics ni hay botones')

        # 1 · se enciende y se pincha en un sitio sin puntos: «aquí hay borreguil»
        lon0, lat0 = base[0]['lon'] + 0.004, base[0]['lat'] + 0.003
        assert bp.punto_mas_cercano(base, lon0, lat0)[1] > 100
        CLICS['mainmap'] = (lon0, lat0)
        tg.set_value(True)
        at.run()
        sin_error(at, 'M2')
        v = VISORES['mainmap']
        comprobar('last_clicked' in v['devuelve'] and 'center' not in v['opciones'],
                  'con el modo de añadir el mapa recoge el clic y no se recoloca en cada uno')
        comprobar(all(boton(at, k) for k in ('mapa_si', 'mapa_no', 'mapa_duda',
                                              'mapa_cancelar'))
                  and hay(at, N(lat0, en, 5) + ('; ' if not en else ', ') + N(lon0, en, 5)),
                  'al pinchar dice dónde y aparecen los botones para contestar')
        at.run()                       # la pasada siguiente ya lee el clic de la sesión
        comprobar(len(VISORES['mainmap']['otros']) >= 1,
                  'el sitio pinchado se marca en el mapa (sin volver a dibujarlo)')
        html = VISORES['mainmap']['html']
        comprobar('"openstreetmap"' not in html and html.count('"OpenStreetMap"') == 1,
                  'OpenStreetMap sale una sola vez en la lista de capas del mapa')
        # Los puntos de los que aprendió el modelo NO se dibujan (lo pidió el 19-11)
        comprobar(T('Entrenamiento del modelo', 'Model training') not in html
                  and T('Puntos de entrenamiento', 'training points') not in html,
                  'los puntos de los que aprendió el modelo no salen en el mapa')
        comprobar(T('Verificado (campo)', 'Verified (field)') not in html,
                  'la leyenda dice «Verificado», sin «(campo)»')
        comprobar(hay(at, T('Para que los puntos que añadas se sumen',
                            'For the points you add to be added')),
                  'dice que para sumar los puntos al modelo hace falta la casilla')
        comprobar(control(at, 'text_input', 'mapa_observador') is None
                  and control(at, 'radio', 'mapa_certidumbre') is not None,
                  'quién lo dice va en «Quién revisa»; aquí, la certidumbre')
        control(at, 'text_input', 'revisor_nombre').input('Ana')
        control(at, 'selectbox', 'revisor_perfil').set_value('manager')
        control(at, 'radio', 'mapa_certidumbre').set_value('high')
        at.run()
        n0 = len(filas(at))
        msgs = pulsar(at, boton(at, 'mapa_si'), 'M3')
        r1 = [r for r in filas(at) if r.get('ID') == 'mapa_1']
        comprobar(len(filas(at)) == n0 + 1 and len(r1) == 1, 'se añade el punto mapa_1')
        r1 = r1[0] if r1 else {}
        comprobar(r1.get('Borreguil') == 'si' and r1.get('decision') == 'BORREGUIL VERIFICADO'
                  and r1.get('method') == 'local knowledge' and r1.get('observer') == 'Ana'
                  and r1.get('expertise') == 'manager' and r1.get('certainty') == 'high'
                  and r1.get('source') == 'mapa'
                  and abs(r1.get('lat', 0) - lat0) < 1e-6,
                  'entra como borreguil verificado, con quién lo dice y el método',
                  str({k: r1.get(k) for k in ('Borreguil', 'decision', 'method',
                                               'observer', 'expertise', 'certainty')}))
        comprobar(leidos and all(n == 1 for _, n in leidos)
                  and {q for q, _ in leidos} >= {'s2', 'topo', 'imagen', 's1', 'pureza'},
                  'solo se leen las variables del punto nuevo, radar y píxel de la pureza '
                  'incluidos', str(leidos))
        comprobar(r1.get('pureza_regla') in bp.NIVELES_TIPO['pureza']
                  and r1.get('verde_estimado') is not None,
                  'y la app le propone la pureza, como en el mapa',
                  f"{r1.get('pureza_regla')} · {r1.get('verde_estimado')}")
        p = r1.get('rf_proba_al_anadir')
        comprobar(isinstance(p, float) and 0 <= p <= 1
                  and abs(p - r1.get('rf_proba', -1)) < 1e-9 and p >= 0.5,
                  'guarda lo que opinaba el modelo antes de saber la respuesta (el de '
                  'partida da lo mismo después)', f'{p} / {r1.get("rf_proba")}')
        comprobar(any(T('**mapa_1** añadido: hay borreguil', '**mapa_1** added: there is a '
                        'borreguil') in m and T('**coincide** ✓', '**it agrees** ✓') in m
                      for m in msgs),
                  'dice qué opinaba el modelo y que coincide', str([m[:90] for m in msgs
                                                                     if 'mapa_1' in m]))
        comprobar(hay(at, T('Puntos añadidos en el mapa: 1', 'Points added on the map: 1'))
                  and hay(at, T('el modelo coincidía en **1 de 1**',
                                'the model agreed on **1 of 1**')),
                  'resumen de lo añadido y de cuántos coincide el modelo')
        comprobar('mapa_1' in at.session_state['base_truth_ids']
                  and bp.etiquetas_de_campo([r1])[0] == 1,
                  'cuenta como borreguil de campo (y en el acierto del modelo)')
        gj = json.loads((work / 'puntos.geojson').read_text(encoding='utf-8'))
        g1 = [f['properties'] for f in gj['features'] if f['properties']['ID'] == 'mapa_1']
        comprobar(g1 and g1[0].get('presence') == 1 and g1[0].get('method') == 'local knowledge'
                  and g1[0].get('observer') == 'Ana',
                  'va en puntos.geojson con las columnas de la plantilla',
                  str(g1[0] if g1 else None))
        comprobar(boton(at, 'mapa_si') is None, 'el mismo clic no se vuelve a ofrecer')

        # 2 · con la casilla de sumar, otro sitio: «aquí no hay»
        at.checkbox(key='entrenar_propio').check()
        at.run()
        comprobar(hay(at, T('se **suma a los 769 de los que aprendió el modelo**',
                            'is **added to the 769 the model learnt from**')),
                  'con la casilla, avisa de que cada punto se suma a los del modelo')
        fuente['r'] = min((base[i] for i in NOB), key=lambda r: r['rf_proba'])
        leidos.clear()
        CLICS['mainmap'] = (lon0 + 0.004, lat0)
        at.run()
        pulsar(at, boton(at, 'mapa_no'), 'M4')
        r2 = ([r for r in filas(at) if r.get('ID') == 'mapa_2'] or [{}])[0]
        comprobar(r2.get('Borreguil') == 'no' and r2.get('decision') == 'NO BORREGUIL VERIFICADO'
                  and 'mapa_2' not in at.session_state['base_truth_ids'],
                  'el segundo entra como ausencia verificada')
        inf = at.session_state['iter_history'][-1].get('info') or {}
        # tuyos con etiqueta: mapa_1 (sí), la ausencia de campo y mapa_2 (no)
        comprobar(inf.get('sumado') and (inf.get('n_pos'), inf.get('n_neg')) == (1, 2)
                  and inf.get('n_tabla', 0) + inf.get('n_tabla_quitados', 0) == 769,
                  'el modelo se reentrena con UN conjunto: los 769 de entrenamiento y los '
                  'tuyos', str({k: inf.get(k) for k in ('sumado', 'n_pos', 'n_neg', 'n_tabla',
                                                       'n_tabla_quitados')}))
        df = tabla_con_columna(at, '¿Coincide?', 'Agrees?')
        comprobar(df is not None and len(df) == 2
                  and hay(at, T('Puntos añadidos en el mapa: 2', 'Points added on the map: 2')),
                  'la tabla de lo añadido tiene los dos', str(df))

        # 3 · pinchar encima de un punto que ya está no añade nada
        CLICS['mainmap'] = (base[3]['lon'] + 0.00003, base[3]['lat'])
        at.run()
        comprobar(boton(at, 'mapa_si') is None
                  and hay(at, T('que ya está en el análisis', 'which is already in the '
                                'analysis')),
                  'pinchar en un marcador no ofrece añadir otro punto encima')

        # 4 · deshacer quita el último
        pulsar(at, boton(at, 'mapa_deshacer'), 'M5')
        comprobar([r['ID'] for r in filas(at) if str(r.get('ID')).startswith('mapa_')]
                  == ['mapa_1'], 'deshacer quita el último añadido (mapa_2)')
        comprobar('mapa_2' not in (work / 'puntos.geojson').read_text(encoding='utf-8'),
                  'y desaparece de las descargas')

        # 5 · cancelar un clic
        CLICS['mainmap'] = (lon0 - 0.004, lat0)
        at.run()
        n1 = len(filas(at))
        pulsar(at, boton(at, 'mapa_cancelar'), 'M6')
        comprobar(len(filas(at)) == n1 and boton(at, 'mapa_si') is None,
                  'cancelar no añade nada y quita los botones')

        # 6 · se apaga el modo: queda el resumen, sin deshacer ni clics
        control(at, 'toggle', 'modo_anadir').set_value(False)
        at.run()
        sin_error(at, 'M7')
        comprobar(boton(at, 'mapa_deshacer') is None
                  and 'last_clicked' not in VISORES['mainmap']['devuelve']
                  and hay(at, T('Puntos añadidos en el mapa: 1', 'Points added on the map: 1')),
                  'apagado, el mapa ya no recoge clics y el resumen sigue a la vista')
        comprobar(T('Añadido en el mapa', 'Added on the map') in VISORES['mainmap']['html']
                  and T('>Verificado</', '>Verified</') in VISORES['mainmap']['html'],
                  'la leyenda del mapa explica el borde blanco de los añadidos y dice '
                  '«Verificado»')
    finally:
        for (mod, nombre), f in originales.items():
            setattr(mod, nombre, f)


# Sin argumentos se ejecutan todos; con nombres, solo esos (p. ej. `… mapa`).
GUIONES = {'revision': guion, 'tabla': guion_tabla, 'mapa': guion_mapa,
           'sin_puntos': guion_sin_puntos, 'anadir': guion_anadir}
for nombre in [a for a in sys.argv[1:] if a in GUIONES] or list(GUIONES):
    for idioma in ('Español', 'English'):
        GUIONES[nombre](idioma)

print()
print('RESULTADO:', 'todo correcto' if not FALLOS else f'{len(FALLOS)} FALLO(S)')
for f_ in FALLOS:
    print('   -', f_)
sys.exit(1 if FALLOS else 0)
