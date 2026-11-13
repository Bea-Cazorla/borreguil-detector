"""Prueba REAL contra Earth Engine de la pila de predictores del mapeo.

Lo que tiene que cumplirse para que el mapa sea coherente con el entrenamiento:

  1. La pila tiene las bandas del catálogo (34), en el mismo orden.
  2. El valor que se lee en un punto (lo que entra en la tabla de entrenamiento) es
     EXACTAMENTE el del píxel que después se clasificará: se compara con un recorte
     de la misma pila descargado sobre la misma rejilla.
  3. La topografía (30 m) llega interpolada a la rejilla de 10 m, y la pendiente
     tiene decimales (no grados enteros).
  4. GNDVI no está en la pila porque es NDWI con el signo cambiado.
  5. El mapa, que lee por bloques, ve en cada píxel EXACTAMENTE lo que se leyó para
     la muestra que cae en él, y donde no hay dato recibe un hueco, no un cero.
  6. Las estadísticas de jun–sep del mapeo son la media entre años de la estadística
     de cada año (se rehacen aquí imagen a imagen), y las del detector siguen siendo
     las de todos los años juntos.

    BORREGUIL_GEE_PROJECT=mi-proyecto python tests/earth_engine_real.py

Necesita red, una cuenta de Earth Engine ya autenticada en el equipo y el nombre del
proyecto en la variable de entorno BORREGUIL_GEE_PROJECT. Sin ella, se omite. Tarda
alrededor de cuatro minutos.
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

# --- las estadísticas de jun–sep del mapeo: cada año por separado y, después, la
# media entre años. Se rehacen aquí por otro camino: Earth Engine da el valor de cada
# imagen en ese píxel y las cuentas se hacen en este ordenador.
import datetime              # noqa: E402
import statistics            # noqa: E402

region = ee.Geometry.Rectangle([-3.45, 37.0, -3.2, 37.15])
veranos = [ee.Filter.date(f'{a}-06-01', f'{a}-09-30') for a in ANIOS]
serie = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(region)
         .filter(ee.Filter.calendarRange(6, 9, 'month'))
         .filter(ee.Filter.Or(*veranos))
         .map(geb._s2cloudless_mask).map(geb._snow_mask).map(geb._add_new_indices))
t0 = time.time()
tabla = serie.getRegion(ee.Geometry.Point(list(PUNTOS[i])), None, f'EPSG:{epsg}',
                        [10, 0, 0, 0, -10, 0]).getInfo()
print(f'  (serie de imágenes en el píxel: {time.time() - t0:.0f} s)')
cab = tabla[0]
por_anio = {a: {k: [] for k in geb.NEW_INDS} for a in ANIOS}
for fila_ in tabla[1:]:
    anio = datetime.datetime.fromtimestamp(fila_[cab.index('time')] / 1000,
                                           datetime.timezone.utc).year
    for k in geb.NEW_INDS:
        if fila_[cab.index(k)] is not None:
            por_anio[anio][k].append(fila_[cab.index(k)])
n_img = {a: len(por_anio[a]['ndmi']) for a in ANIOS}
comprobar(all(n >= 5 for n in n_img.values()),
          'hay imágenes despejadas de cada verano en ese píxel', str(n_img))
ESTAD = {'mean': statistics.fmean, 'min': min, 'max': max, 'sd': statistics.pstdev}


def cerca(a, b):
    return abs(a - b) <= 1e-6 * max(1.0, abs(b))


distintas = []
for k in geb.NEW_INDS:
    for s, f in ESTAD.items():
        anuales = [f(por_anio[a][k]) for a in ANIOS
                   if len(por_anio[a][k]) >= (2 if s == 'sd' else 1)]
        esperado = statistics.fmean(anuales)
        if not cerca(valores[i][f'{k}_{s}'], esperado):
            distintas.append((f'{k}_{s}', valores[i][f'{k}_{s}'], esperado))
comprobar(not distintas,
          'las 16 estadísticas de jun–sep son la media entre años de la de cada año '
          '(rehechas imagen a imagen)', str(distintas[:4]))
# El detector no cambia: sus modelos se entrenaron con todos los años juntos.
juntas = geb._s2_stack(region, ANIOS)[1].reduceRegion(
    ee.Reducer.first(), ee.Geometry.Point(list(PUNTOS[i])), crs=f'EPSG:{epsg}',
    crsTransform=[10, 0, 0, 0, -10, 0]).getInfo()
distintas = []
for k in geb.NEW_INDS:
    todas = [v for a in ANIOS for v in por_anio[a][k]]
    for s, f in ESTAD.items():
        if not cerca(juntas[f'{k}_{s}'], f(todas)):
            distintas.append((f'{k}_{s}', juntas[f'{k}_{s}'], f(todas)))
comprobar(not distintas,
          'el detector sigue con todas las imágenes de todos los años juntas',
          str(distintas[:4]))
comprobar(all(valores[i][f'{k}_min'] >= juntas[f'{k}_min'] - 1e-9
              and valores[i][f'{k}_max'] <= juntas[f'{k}_max'] + 1e-9 for k in geb.NEW_INDS)
          and any(not cerca(valores[i][f'{k}_min'], juntas[f'{k}_min']) for k in geb.NEW_INDS),
          'año a año, el mínimo y el máximo quedan dentro de los de todos los años juntos',
          str({k: (valores[i][f'{k}_min'], juntas[f'{k}_min']) for k in geb.NEW_INDS}))
comprobar(info.get('version') == geb.VERSION_PILA == 2 and 'cada año' in info.get('estadisticas', ''),
          'la lectura deja constancia de cómo se calculan las estadísticas')

# --- el MAPA lee por bloques: tiene que ver en cada píxel lo mismo que vio la muestra
# Un bloque de 12 × 8 píxeles con el punto en la columna 5, fila 3. El Random Forest
# trabaja en float32, así que la comparación se hace con esa precisión, y es exacta.
import numpy as np           # noqa: E402

bandas = list(geb.BANDAS_PILA)
x0, y0 = (col - 5) * 10, (fila + 1 + 3) * 10
lon0, lat0 = PUNTOS[i]
limites = (lon0 - 0.01, lat0 - 0.01, lon0 + 0.01, lat0 + 0.01)
pila_mapa = geb.pila_para_mapa(limites, ANIOS, epsg, bandas)
t0 = time.time()
bloque = geb.leer_bloque(pila_mapa, epsg, x0, y0, 12, 8, bandas)
print(f'  (bloque de 12×8 píxeles: {time.time() - t0:.0f} s)')
comprobar(bloque.shape == (8, 12, 34) and bloque.dtype == np.float32,
          'el bloque llega como (filas, columnas, variables), en float32',
          f'{bloque.shape} {bloque.dtype}')
esperado = np.array([valores[i][b] for b in bandas], dtype='float32')
distintas = [b for b, a, e in zip(bandas, bloque[3, 5], esperado) if a != e]
comprobar(not distintas,
          'el píxel del bloque es EXACTAMENTE el valor de la muestra (en float32), '
          'en las 34 variables', str(distintas[:6]))
comprobar(np.isfinite(bloque).all() and not (bloque == geb.SIN_DATO_BLOQUE).any(),
          'en esta zona el bloque no tiene huecos ni restos del valor centinela')
# con menos variables y en otro orden, cada columna sigue siendo la suya
pocas = ['elev_dem_m', 'ndvi_late', 'ndmi_mean']
menor = geb.leer_bloque(geb.pila_para_mapa(limites, ANIOS, epsg, pocas), epsg, x0, y0, 12, 8,
                        pocas)
comprobar(menor.shape == (8, 12, 3)
          and all(np.array_equal(menor[..., k], bloque[..., bandas.index(b)])
                  for k, b in enumerate(pocas)),
          'con tres variables en otro orden, cada columna trae la variable pedida')
# Donde no hay dato tiene que llegar NaN, no un cero. Se tapa a propósito la mitad
# derecha del bloque (como haría la nube o la nieve de todos los años).
tapada = pila.select(pocas).updateMask(
    ee.Image.pixelCoordinates(f'EPSG:{epsg}').select('x').lt(x0 + 60))
mitad = geb.leer_bloque(geb._para_bloques(tapada), epsg, x0, y0, 12, 8, pocas)
comprobar(np.isnan(mitad[:, 6:]).all() and np.array_equal(mitad[:, :6], menor[:, :6]),
          'los píxeles sin dato llegan como NaN, y los demás no cambian',
          f'{int(np.isnan(mitad).sum())} NaN de {mitad.size}')

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
