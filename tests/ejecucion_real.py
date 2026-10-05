"""Ejecución REAL del análisis, de principio a fin, con 8 puntos.

Sube un GeoJSON como haría una persona (con etiquetas de campo, huecos y un tipo ya
revisado), ejecuta el pipeline en modo rápido y comprueba lo que queda en la sesión
y en los ficheros guardados. Es la única prueba que pasa por el bloque que descarga,
clasifica y guarda.

    python tests/ejecucion_real.py        (unos 3 minutos, NECESITA RED)

Por defecto usa Planetary Computer, que no pide cuenta. Para probar el camino de
Earth Engine, pon el nombre de tu proyecto en BORREGUIL_GEE_PROJECT:

    BORREGUIL_GEE_PROJECT=mi-proyecto python tests/ejecucion_real.py

Los 8 puntos son una rejilla inventada en la alta montaña de Sierra Nevada: lo que
se comprueba son las etiquetas y los ficheros, no lo que prediga el modelo.
No es una prueba de pytest: ejecuta la app al arrancar.
"""
import json
import os
import sys
import tempfile
import time
import warnings
from pathlib import Path

APP = Path(__file__).resolve().parent.parent
os.chdir(APP)
sys.path.insert(0, str(APP))
DATOS = Path(tempfile.mkdtemp(prefix='bt_'))
os.environ['BORREGUIL_DATA_DIR'] = str(DATOS)
PROYECTO_GEE = os.environ.get('BORREGUIL_GEE_PROJECT', '').strip()
warnings.simplefilter('ignore')

import borreguil_pipeline as bp   # noqa: E402

bp._imports()
from streamlit.testing.v1 import AppTest   # noqa: E402

FALLOS = []


def comprobar(cond, texto, detalle=''):
    print(('  ok     ' if cond else '  FALLO  ') + texto
          + (f'  [{detalle}]' if detalle and not cond else ''))
    if not cond:
        FALLOS.append(texto)


# ------------------------------------------------------------ fichero de entrada
feats = []
for i in range(8):                       # rejilla de unos 450 m en la alta montaña
    props = {'ID': f'campo_{i}', 'Borreguil': None, 'Duda': None, 'nota': f'punto {i}'}
    feats.append({'type': 'Feature', 'properties': props,
                  'geometry': {'type': 'Point',
                               'coordinates': [-3.345 + (i % 4) * 0.005,
                                               37.050 + (i // 4) * 0.004]}})
P = [f['properties'] for f in feats]
P[0].update(Borreguil='si', tipo_revisado='si', ambiente='laguna', humedad='seco',
            pureza='mixto-roca', tipo_revisado_fecha='2026-09-30T09:00:00')
P[1].update(Borreguil='si')
P[2].update(Duda='si')
P[3].update(tipo_revisado='si', ambiente='rio', humedad='seco', pureza='puro')  # inválido
P[5].update(Borreguil='no')
contenido = json.dumps({'type': 'FeatureCollection', 'features': feats}).encode('utf-8')
print(f'fichero de prueba: {len(feats)} puntos '
      '(2 «si», 1 «no», 1 dudoso, 1 tipo revisado, 1 revisado con categoría inválida)')

# ---------------------------------------------------------------- ejecución real
at = AppTest.from_file(str(APP / 'app.py'), default_timeout=1500)
at.run()
at.file_uploader[0].set_value(('puntos_campo.geojson', contenido, 'application/geo+json'))
if PROYECTO_GEE:
    print('origen de los datos: Google Earth Engine')
    [t for t in at.text_input if t.label == 'Proyecto Google Earth Engine'][0].set_value(
        PROYECTO_GEE)
else:
    print('origen de los datos: Microsoft Planetary Computer')
    [r for r in at.radio if r.label == 'Backend'][0].set_value(
        '🛰  Microsoft Planetary Computer (sin cuenta)')
[t for t in at.text_input if t.label.startswith('Años Sentinel-2')][0].set_value('2024')
[c for c in at.checkbox if c.label.startswith('⚡ Modo rápido')][0].set_value(True)
at.run()
boton = [b for b in at.button if b.label == '▶ Ejecutar pipeline'][0]
comprobar(not boton.disabled, 'con el fichero subido, el botón de ejecutar está activo')
t0 = time.time()
boton.click()
at.run()
print(f'  (ejecución: {time.time() - t0:.0f} s)')

exc = list(at.exception)
comprobar(not exc, 'la ejecución termina sin excepciones',
          ' | '.join(str(e.value)[:300] for e in exc[:2]))
if 'rows' not in at.session_state:
    print('  la ejecución no ha dejado resultados; mensajes:')
    for grupo in (at.error, at.warning):
        for x in grupo:
            print('     ', str(x.value)[:200])
    sys.exit(1)

rows = at.session_state['rows']
work = at.session_state['work']
if PROYECTO_GEE:
    comprobar(at.session_state['active_backend'] == 'gee',
              'los datos han salido de Earth Engine (no ha recurrido al otro origen)',
              str(at.session_state['active_backend']))
    recordado = (json.loads((DATOS / 'ajustes.json').read_text(encoding='utf-8'))
                 if (DATOS / 'ajustes.json').exists() else {})
    comprobar(recordado.get('gee_project') == PROYECTO_GEE,
              'el proyecto de Earth Engine queda recordado para la próxima vez')
por_id = {r['ID']: r for r in rows}
comprobar(len(rows) == 8, 'están los 8 puntos', str(len(rows)))

# ¿qué se ha podido descargar? (informativo: depende de la red)
for etiqueta, var in (('topografía', 'slope_deg'), ('Sentinel-2', 'ndvi_late'),
                      ('NDMI', 'ndmi_mean')):
    n = sum(1 for r in rows if isinstance(r.get(var), (int, float)) and r[var] == r[var])
    print(f'  (dato de {etiqueta}: {n}/8 puntos)')

# decisiones: lo que puso la persona manda
comprobar(por_id['campo_0']['decision'] == 'BORREGUIL VERIFICADO'
          and por_id['campo_1']['decision'] == 'BORREGUIL VERIFICADO',
          'los dos «si» de campo salen como verificados')
comprobar(por_id['campo_5']['decision'] == 'NO BORREGUIL VERIFICADO',
          'el «no» de campo sale como ausencia verificada', por_id['campo_5']['decision'])
comprobar(por_id['campo_2']['decision'] == 'DUDOSO (campo)',
          'el dudoso sale como dudoso', por_id['campo_2']['decision'])
sin_etq = [por_id[f'campo_{i}']['decision'] for i in (3, 4, 6, 7)]
comprobar(all(d in ('BORREGUIL PROBABLE', 'POSIBLE BORREGUIL', 'INCIERTO', 'NO BORREGUIL',
                    'SIN PREDICCIÓN') for d in sin_etq),
          'los puntos sin etiqueta (huecos en el fichero) se deciden por el modelo',
          str(sin_etq))

# tipo revisado que venía en el fichero
r0 = por_id['campo_0']
comprobar(bp.tipo_revisado(r0) and (r0['ambiente'], r0['humedad'], r0['pureza'])
          == ('laguna', 'seco', 'mixto-roca'),
          'el tipo revisado que traía el fichero se conserva',
          f"{r0.get('ambiente')}/{r0.get('humedad')}/{r0.get('pureza')}")
comprobar(r0['tipo_revisado_fecha'] == '2026-09-30T09:00:00', 'y su fecha también',
          repr(r0.get('tipo_revisado_fecha')))
comprobar('ambiente_regla' in r0, 'la propuesta de la app se guarda aparte')
r3 = por_id['campo_3']
comprobar(not bp.tipo_revisado(r3) and 'rio' in str(r3.get('tipo_revision_invalida')),
          'el revisado con categoría inválida no cuenta y queda anotado',
          repr(r3.get('tipo_revision_invalida')))
avisos = [str(x.value) for x in at.warning]
comprobar(any('campo_3' in a and 'no admitida' in a for a in avisos),
          'la pantalla avisa de ese punto', str(avisos)[:200])
comprobar(por_id['campo_1'].get('nota') == 'punto 1', 'los atributos propios se conservan')

# ficheros
for nombre in ('classification.csv', 'puntos.geojson', 'Clasificacion_puntos.xlsx', 'mapa.html'):
    comprobar((work / nombre).exists(), f'se ha guardado {nombre}')
gj = json.loads((work / 'puntos.geojson').read_text(encoding='utf-8'))
props = {f['properties']['ID']: f['properties'] for f in gj['features']}
comprobar(props['campo_0']['tipo_revisado'] == 'si' and props['campo_0']['ambiente'] == 'laguna'
          and props['campo_5']['Borreguil'] == 'no' and props['campo_2']['Duda'] == 'si'
          and props['campo_3']['tipo_revisado'] == '',
          'puntos.geojson lleva las etiquetas tal como quedaron')
comprobar(not any(set(p) & set(bp.FEATURES_RF_COMBO) for p in props.values()),
          'puntos.geojson no lleva variables')

# la pantalla de resultados se dibuja tras la ejecución
comprobar(any(h.value == 'Resultados' for h in at.header), 'se muestra la pantalla de resultados')
comprobar(any('Tipo revisado: 1 de' in str(m.value) for m in at.markdown),
          'el contador arranca con el revisado que venía en el fichero',
          str([str(m.value)[:60] for m in at.markdown if 'Tipo revisado' in str(m.value)]))

# segunda vuelta: lo guardado se puede cargar otra vez y sigue igual
otra = bp.read_points(str(work / 'puntos.geojson'))
o = {r['ID']: r for r in otra}
comprobar(bp.tipo_revisado(o['campo_0']) and bp._txt(o['campo_5']['Borreguil']) == 'no'
          and bp._txt(o['campo_2']['Duda']) == 'si',
          'el fichero guardado se vuelve a leer con las mismas etiquetas')

print()
print('RESULTADO:', 'todo correcto' if not FALLOS else f'{len(FALLOS)} FALLO(S)')
for f_ in FALLOS:
    print('   -', f_)
sys.exit(1 if FALLOS else 0)
