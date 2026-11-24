"""Prueba REAL contra Earth Engine de las variables Sentinel-2 del DETECTOR.

Los modelos de partida (Sierra Nevada) se entrenaron con variables de Planetary
Computer, y solo aciertan si Earth Engine les da los mismos números. Aquí se piden a
Earth Engine (geb.fetch_s2_gee, lo mismo que hace la app) 16 puntos de entrenamiento
de 9 cuencas, 8 borreguiles y 8 no borreguiles, y se comparan con los valores de la
tabla con la que se entrenó el modelo (classification_v7.csv, copiados aquí):

  1. Earth Engine devuelve las 31 variables Sentinel-2 en los 16 puntos.
  2. En las variables que más pesan en el modelo, la diferencia típica con la tabla es
     menor que el 40 % del rango central (p25–p75) de esa variable en el
     entrenamiento. Antes del 19-11-2026, con la mediana de todas las imágenes, la de
     la variabilidad del NDWI era casi el doble de ese rango y la del EVI, más de él.
  3. El EVI sale en la misma escala que en el entrenamiento (cociente de medianas
     0,9–1,1) y es el EVI normal, por debajo de 1. (Hasta el 19-11-2026 el modelo se
     entrenó con un EVI en números digitales que pasaba de 1.)
  4. El radar (geb.fetch_s1_gee) da en los 16 puntos las 5 variables de Sentinel-1
     con los mismos valores con los que se reentrenaron los modelos el 19-11-2026
     (classification_v7.csv), con menos de 0,01 dB de diferencia.

    BORREGUIL_GEE_PROJECT=mi-proyecto python tests/earth_engine_detector.py

Necesita red, una cuenta de Earth Engine ya autenticada en el equipo y el nombre del
proyecto en BORREGUIL_GEE_PROJECT. Sin ella, se omite. Tarda un minuto y medio.
"""
import math
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROYECTO = os.environ.get('BORREGUIL_GEE_PROJECT', '').strip()
if not PROYECTO:
    print('Se omite: define BORREGUIL_GEE_PROJECT con el nombre de tu proyecto de '
          'Earth Engine.')
    sys.exit(0)

import gee_backend as geb   # noqa: E402

FALLOS = []


def comprobar(cond, texto, detalle=''):
    print(('  ok     ' if cond else '  FALLO  ') + texto
          + (f'  [{detalle}]' if detalle and not cond else ''))
    if not cond:
        FALLOS.append(texto)


CLAVES = ['ndvi_early', 'clre_early', 'ndmi_early', 'evi_early', 'ndvi_drop',
          'wavi_sd', 'ndmi_sd', 'ndwi_sd', 'ndwi_min', 'ndmi_max']
# p75 − p25 de cada variable en los 769 puntos de entrenamiento sin duda
RANGO = {'ndvi_early': 0.2063, 'clre_early': 0.4701, 'ndmi_early': 0.0895,
         'evi_early': 0.1524, 'ndvi_drop': 0.0719, 'wavi_sd': 0.0189, 'ndmi_sd': 0.0252,
         'ndwi_sd': 0.0191, 'ndwi_min': 0.2003, 'ndmi_max': 0.1029}
# (lon, lat, borreguil, valores de la tabla de entrenamiento en el orden de CLAVES)
PUNTOS = [
    (-3.296979, 37.067829, 'si', (0.17, 0.1564, 0.0784, 0.1196, 0.0647, 0.0301, 0.0465, 0.0383, -0.2805, 0.1943)),
    (-3.365203, 37.064646, 'si', (0.5227, 1.0862, 0.2038, 0.3741, 0.1461, 0.0353, 0.0531, 0.0437, -0.5978, 0.291)),
    (-3.270446, 37.122837, 'si', (0.4385, 0.7542, 0.1491, 0.3012, 0.0674, 0.0282, 0.033, 0.0287, -0.4954, 0.1766)),
    (-3.420477, 37.040588, 'si', (0.3298, 0.4764, 0.1203, 0.2182, 0.2947, 0.0303, 0.0308, 0.0415, -0.4385, 0.1795)),
    (-3.410138, 37.048047, 'si', (0.4463, 0.8759, 0.1619, 0.3156, 0.1315, 0.0389, 0.0563, 0.0419, -0.5149, 0.2028)),
    (-3.385845, 37.049607, 'si', (0.4764, 0.7506, 0.1496, 0.3264, 0.1151, 0.0419, 0.0434, 0.0455, -0.5007, 0.1818)),
    (-3.059086, 37.08874, 'si', (0.3702, 0.6512, 0.1368, 0.2487, 0.0796, 0.0365, 0.0472, 0.0366, -0.46, 0.1772)),
    (-3.383317, 37.050475, 'si', (0.1589, 0.1688, 0.0405, 0.1051, 0.047, 0.0152, 0.0153, 0.0201, -0.2469, 0.0617)),
    (-3.106887, 37.081873, 'no', (0.1909, 0.2883, 0.0375, 0.12, 0.0448, 0.0205, 0.0198, 0.0232, -0.2727, 0.0579)),
    (-3.370374, 37.030131, 'no', (0.3225, 0.5269, 0.1064, 0.2261, 0.0483, 0.0294, 0.0352, 0.0296, -0.4348, 0.1461)),
    (-3.205143, 37.104008, 'no', (0.2669, 0.3631, 0.06, 0.1621, 0.0021, 0.0182, 0.0232, 0.0283, -0.3834, 0.0911)),
    (-3.411397, 37.052215, 'no', (0.1583, 0.1739, 0.0158, 0.0936, 0.0335, 0.0149, 0.028, 0.0207, -0.2482, 0.057)),
    (-3.382776, 37.065706, 'no', (0.3248, 0.524, 0.118, 0.2102, 0.0665, 0.0493, 0.0339, 0.0653, -0.3538, 0.1508)),
    (-3.2184, 37.111987, 'no', (0.3122, 0.4399, 0.1097, 0.1935, -0.0206, 0.0272, 0.0298, 0.0371, -0.3966, 0.1636)),
    (-3.236058, 37.15058, 'no', (0.3626, 0.5989, 0.0357, 0.2398, 0.0308, 0.0413, 0.0399, 0.0667, -0.4829, 0.0726)),
    (-3.460174, 37.023424, 'no', (0.3276, 0.5618, 0.1091, 0.2386, 0.0453, 0.0315, 0.0359, 0.0358, -0.4393, 0.1524)),
]

# Radar de la tabla de entrenamiento (classification_v7.csv), en el orden de PUNTOS:
# s1_vv_mean, s1_vh_mean, s1_vv_sd, s1_vh_sd, s1_ratio_mean (dB)
RADAR = [
    (-3.9388, -13.9871, 3.3684, 5.4399, 10.0483),
    (-9.7917, -19.3446, 2.3257, 1.1889, 9.5528),
    (-9.8450, -19.1534, 2.0396, 1.1007, 9.3085),
    (-9.0522, -18.7839, 1.5104, 1.1789, 9.7317),
    (-9.9591, -19.2716, 1.3798, 1.0713, 9.3125),
    (-9.6839, -19.4332, 0.9760, 1.5353, 9.7493),
    (-9.7648, -17.8651, 1.7322, 1.2533, 8.1003),
    (-10.0449, -18.9090, 0.9617, 1.1386, 8.8640),
    (-8.3879, -15.9052, 1.6356, 1.6343, 7.5173),
    (-8.2244, -16.1183, 2.1950, 1.4488, 7.8939),
    (-11.4157, -18.5976, 1.1337, 1.1708, 7.1819),
    (-9.3618, -18.8632, 1.2708, 1.3272, 9.5014),
    (-10.4782, -20.0763, 1.0355, 1.3261, 9.5982),
    (-10.9745, -18.5130, 1.6247, 1.6392, 7.5385),
    (-12.4488, -20.5759, 1.1777, 1.6242, 8.1271),
    (-10.1815, -19.5472, 1.1715, 1.1790, 9.3657),
]

ok, msg = geb.initialize(PROYECTO)
comprobar(ok, 'Earth Engine se inicializa con el proyecto', msg)
if not ok:
    sys.exit(1)

rows = [{'lon': lon, 'lat': lat} for lon, lat, _, _ in PUNTOS]
lons = [r['lon'] for r in rows]
lats = [r['lat'] for r in rows]
bbox = (min(lons) - 0.02, min(lats) - 0.02, max(lons) + 0.02, max(lats) + 0.02)
fallidos = geb.fetch_s2_gee(rows, bbox)


def numero(v):
    return isinstance(v, (int, float)) and not math.isnan(v)


sin_dato = [(i, k) for i, r in enumerate(rows) for k in geb.CLAVES_S2 if not numero(r.get(k))]
comprobar(fallidos == 0 and not sin_dato and len(geb.CLAVES_S2) == 31,
          'Earth Engine devuelve las 31 variables Sentinel-2 en los 16 puntos',
          f'fallidos={fallidos}, sin dato={sin_dato[:5]}')

for j, k in enumerate(CLAVES):
    difs = [abs(r[k] - p[3][j]) for r, p in zip(rows, PUNTOS) if numero(r.get(k))]
    rel = statistics.median(difs) / RANGO[k] if difs else float('inf')
    comprobar(rel < 0.4, f'{k}: diferencia típica con la tabla de entrenamiento '
              f'{rel:.0%} de su rango central', f'{rel:.2f}')

evi_ee = statistics.median(r['evi_early'] for r in rows if numero(r.get('evi_early')))
evi_tabla = statistics.median(p[3][CLAVES.index('evi_early')] for p in PUNTOS)
comprobar(0.9 <= evi_ee / evi_tabla <= 1.1,
          f'el EVI sale en la escala del entrenamiento (cociente {evi_ee / evi_tabla:.3f})',
          f'{evi_ee:.3f} frente a {evi_tabla:.3f}')
evis = [r[k] for r in rows for k in ('evi_early', 'evi_late') if numero(r.get(k))]
comprobar(evis and max(evis) < 1,
          f'es el EVI normal: ninguno pasa de 1 (el mayor, {max(evis):.3f})', str(max(evis)))

fallidos_s1 = geb.fetch_s1_gee(rows, bbox)
dif_s1 = max((abs(r[k] - v[j]) for r, v in zip(rows, RADAR)
              for j, k in enumerate(geb.CLAVES_S1) if numero(r.get(k))), default=float('inf'))
sin_s1 = [i for i, r in enumerate(rows) if not all(numero(r.get(k)) for k in geb.CLAVES_S1)]
comprobar(fallidos_s1 == 0 and not sin_s1 and dif_s1 < 0.01,
          f'el radar da las 5 variables en los 16 puntos, iguales a las del entrenamiento '
          f'(diferencia máxima {dif_s1:.4f} dB)', f'fallidos={fallidos_s1}, sin dato={sin_s1}')

print()
if FALLOS:
    print(f'RESULTADO: {len(FALLOS)} fallo(s)')
    sys.exit(1)
print('RESULTADO: todo correcto')
