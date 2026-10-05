"""Prueba REAL contra Earth Engine de la pila de predictores del mapeo.

Lo que tiene que cumplirse para que el mapa sea coherente con el entrenamiento:

  1. La pila tiene las bandas del catálogo (34), en el mismo orden.
  2. El valor que se lee en un punto (lo que entra en la tabla de entrenamiento) es
     EXACTAMENTE el del píxel que después se clasificará: se compara con un recorte
     de la misma pila descargado sobre la misma rejilla.
  3. La topografía (30 m) llega interpolada a la rejilla de 10 m, y la pendiente
     tiene decimales (no grados enteros).
  4. GNDVI no está en la pila porque es NDWI con el signo cambiado.

    BORREGUIL_GEE_PROJECT=mi-proyecto python tests/earth_engine_real.py

Necesita red, una cuenta de Earth Engine ya autenticada en el equipo y el nombre del
proyecto en la variable de entorno BORREGUIL_GEE_PROJECT. Sin ella, se omite. Tarda
alrededor de un minuto.
"""
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROYECTO = os.environ.get('BORREGUIL_GEE_PROJECT', '').strip()
if not PROYECTO:
    print('Se omite: define BORREGUIL_GEE_PROJECT con el nombre de tu proyecto de '
          'Earth Engine.')
    sys.exit(0)

import gee_backend as geb   # noqa: E402
import mapeo_rf as mr       # noqa: E402

FALLOS = []


def comprobar(cond, texto, detalle=''):
    print(('  ok     ' if cond else '  FALLO  ') + texto
          + (f'  [{detalle}]' if detalle and not cond else ''))
    if not cond:
        FALLOS.append(texto)


ok, msg = geb.initialize(PROYECTO)
comprobar(ok, 'Earth Engine se inicializa con el proyecto', msg)
if not ok:
    sys.exit(1)
import ee                    # noqa: E402
import pyproj                # noqa: E402

# Alta montaña de Sierra Nevada: prado, roca, laguna y ladera.
PUNTOS = [(-3.27045, 37.12285), (-3.40619, 37.03503), (-3.3312, 37.0493),
          (-3.345, 37.050), (-3.335, 37.054), (-3.38546, 37.04938)]
ANIOS = (2023, 2024)
epsg = geb.utm_de(PUNTOS)
comprobar(epsg == 32630, 'la rejilla es UTM 30N', str(epsg))

t0 = time.time()
valores, fallidos, info = geb.leer_predictores(PUNTOS, ANIOS)
print(f'  (lectura de {len(PUNTOS)} puntos: {time.time() - t0:.0f} s)')
comprobar(fallidos == 0 and all(v is not None for v in valores),
          'se consultan todos los puntos', f'fallidos={fallidos}')
comprobar(all(list(v) == mr.nombres_predictores() for v in valores),
          f'cada punto trae las {len(geb.BANDAS_PILA)} variables, en el orden del catálogo')
huecos = [(i, b) for i, v in enumerate(valores) for b, x in v.items() if x is None]
comprobar(not huecos, 'ningún punto se queda sin dato', str(huecos[:6]))
comprobar(info['crs'] == 'EPSG:32630' and info['resolucion_m'] == 10
          and info['anios'] == list(ANIOS), 'la lectura deja constancia de rejilla y años')
comprobar(len(geb.BANDAS_PILA) == 34 and not [b for b in geb.BANDAS_PILA if 'gndvi' in b],
          'la pila tiene 34 bandas y no incluye GNDVI')
comprobar(any(abs(v['slope_deg'] - round(v['slope_deg'])) > 1e-6 for v in valores),
          'la pendiente llega con decimales, no en grados enteros',
          str([v['slope_deg'] for v in valores]))
comprobar(geb.DEM_ASSET == 'COPERNICUS/DEM/GLO30_2024_1',
          'se usa la edición vigente del modelo de elevaciones', geb.DEM_ASSET)

# --- el píxel que se clasificará es el mismo que se ha leído en el punto
pila = geb.pila_predictores(ee.Geometry.Rectangle([-3.45, 37.0, -3.2, 37.15]), ANIOS, epsg)
comprobar(pila.bandNames().getInfo() == list(geb.BANDAS_PILA),
          'la pila de Earth Engine tiene las bandas del catálogo, en orden')
i = 2
x, y = pyproj.Transformer.from_crs(4326, epsg, always_xy=True).transform(*PUNTOS[i])
col, fila = math.floor(x / 10), math.floor(y / 10)
t0 = time.time()
recorte = ee.data.computePixels({
    'expression': pila, 'fileFormat': 'NUMPY_NDARRAY',
    'grid': {'dimensions': {'width': 3, 'height': 3},
             'affineTransform': {'scaleX': 10, 'shearX': 0, 'translateX': (col - 1) * 10,
                                 'shearY': 0, 'scaleY': -10, 'translateY': (fila + 2) * 10},
             'crsCode': f'EPSG:{epsg}'}})
print(f'  (recorte de 3×3 píxeles: {time.time() - t0:.0f} s)')
distintas = [b for b in geb.BANDAS_PILA
             if abs(float(recorte[b][1, 1]) - valores[i][b]) > 1e-6 * max(1.0, abs(valores[i][b]))]
comprobar(not distintas,
          'el píxel del recorte coincide con el valor leído en el punto, en todas las bandas',
          str(distintas[:6]))
comprobar(len(set(recorte['elev_dem_m'].ravel().tolist())) > 1,
          'la altitud cambia de un píxel de 10 m a otro (interpolada, no repetida)')

# --- por qué GNDVI no está: en la pila del detector es exactamente −NDWI
s2, _ = geb._s2_stack(ee.Geometry.Rectangle([-3.45, 37.0, -3.2, 37.15]), ANIOS)
par = s2.select(['ndwi_late', 'gndvi_late']).reduceRegion(
    ee.Reducer.first(), ee.Geometry.Point(list(PUNTOS[i])), 10).getInfo()
comprobar(abs(par['ndwi_late'] + par['gndvi_late']) < 1e-9,
          'GNDVI es exactamente NDWI con el signo cambiado', str(par))

print()
print('RESULTADO:', 'todo correcto' if not FALLOS else f'{len(FALLOS)} FALLO(S)')
for f_ in FALLOS:
    print('   -', f_)
sys.exit(1 if FALLOS else 0)
