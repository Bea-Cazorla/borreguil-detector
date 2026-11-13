"""Pruebas del generador de puntos al azar (random_points.py): que, si se pide, los
puntos quedan lejos de los de campo, que sin pedirlo sale lo mismo que antes, y que el
reparto por franjas de altitud funciona sin límites de altitud.

Se ejecutan con `python tests/test_puntos_azar.py`; no necesitan red (la altitud, que
se consulta a un servicio de internet, se inventa donde hace falta).
"""
import sys
from pathlib import Path

import numpy as np
import pyproj
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import random_points as rp       # noqa: E402

# Un rectángulo de unos 9 × 6 km en Sierra Nevada y unos «puntos de campo» en su
# mitad oeste.
ZONA = box(-3.40, 37.03, -3.30, 37.08)
A_UTM = pyproj.Transformer.from_crs(4326, 32630, always_xy=True)
CAMPO = [(-3.39 + 0.004 * i, 37.035 + 0.0035 * j) for i in range(10) for j in range(12)]


def distancia_minima(puntos, otros):
    xp, yp = A_UTM.transform([p[0] for p in puntos], [p[1] for p in puntos])
    xo, yo = A_UTM.transform([p[0] for p in otros], [p[1] for p in otros])
    d = np.hypot(np.subtract.outer(np.array(xp), np.array(xo)),
                 np.subtract.outer(np.array(yp), np.array(yo)))
    return d.min(axis=1)


def test_lejos_de_quita_los_que_caen_cerca_de_los_de_campo():
    pts = [(-3.39, 37.035), (-3.35, 37.06), (-3.3895, 37.0352)]
    quedan = rp.lejos_de(pts, CAMPO, 300)
    assert quedan == [(-3.35, 37.06)]
    assert rp.lejos_de(pts, CAMPO, 0) == pts and rp.lejos_de(pts, [], 300) == pts


def test_los_puntos_al_azar_quedan_lejos_de_los_de_campo():
    for modo in ('uniform', 'poisson_disc'):
        filas, stats = rp.generate(ZONA, 40, mode=modo, min_dist_m=200, seed=7,
                                   excluir=CAMPO, lejos_m=300)
        pts = [(f['lon'], f['lat']) for f in filas]
        assert len(pts) == 40, (modo, len(pts))
        # se ajustan al centro del píxel de 10 m: pueden acercarse hasta 7,1 m más
        assert distancia_minima(pts, CAMPO).min() >= 300 - 7.1, modo
        assert stats['excluded_near'] > 0 and stats['kept'] == 40
        assert all(ZONA.buffer(0.0002).contains(_p(p)) for p in pts)
        assert [f['ID'] for f in filas] == [f'rand_{i:04d}' for i in range(40)]


def _p(p):
    from shapely.geometry import Point
    return Point(p)


def test_por_franjas_de_altitud_sin_limites_de_altitud():
    """El reparto por franjas de altitud consulta las altitudes aunque no se pongan
    límites (antes fallaba si se dejaba vacía la altitud mínima). La altitud de
    internet se cambia aquí por una inventada: casi toda la zona baja y solo el borde
    norte alto, para ver que las franjas altas reciben más puntos de los que les tocarían
    por superficie."""
    real = rp.fetch_elevations
    try:
        rp.fetch_elevations = lambda coords, **kw: [
            2000 + 1000 * ((lat - 37.03) / 0.05) ** 4 for lat, lon in coords]
        filas, stats = rp.generate(ZONA, 30, mode='stratified_elev', seed=5)
        alt = [f['elev_m_initial'] for f in filas]
        assert 20 <= len(filas) <= 30 and stats['reason'] is None, (len(filas), stats)
        assert all(2000 <= a <= 3000 for a in alt)
        # por encima de 2600 m solo hay un 12 % de la zona: al azar caerían unos 4
        assert sum(a > 2600 for a in alt) >= 8, sorted(alt)
        # si el servicio no da altitudes, ningún punto, y se dice por qué
        rp.fetch_elevations = lambda coords, **kw: [None] * len(coords)
        filas, stats = rp.generate(ZONA, 30, mode='stratified_elev', seed=5)
        assert filas == [] and stats['reason'] == 'no_elevation_data'
    finally:
        rp.fetch_elevations = real


def test_sin_pedirlo_sale_lo_mismo_que_antes():
    a, sa = rp.generate(ZONA, 30, mode='uniform', seed=3)
    b, sb = rp.generate(ZONA, 30, mode='uniform', seed=3, excluir=CAMPO, lejos_m=0)
    c, _ = rp.generate(ZONA, 30, mode='uniform', seed=3, excluir=None, lejos_m=300)
    assert a == b == c and sa['excluded_near'] == sb['excluded_near'] == 0
    # y alejándose, cerca de los de campo no queda ninguno de los de antes
    lejos, _ = rp.generate(ZONA, 30, mode='uniform', seed=3, excluir=CAMPO, lejos_m=300)
    assert distancia_minima([(f['lon'], f['lat']) for f in lejos], CAMPO).min() >= 292.9
    assert distancia_minima([(f['lon'], f['lat']) for f in a], CAMPO).min() < 292.9


if __name__ == '__main__':
    pruebas = [(n, f) for n, f in sorted(globals().items())
               if n.startswith('test_') and callable(f)]
    fallos = 0
    for nombre, f in pruebas:
        try:
            f()
            print(f'  ok     {nombre}')
        except Exception as e:                      # noqa: BLE001
            fallos += 1
            print(f'  FALLO  {nombre}: {type(e).__name__}: {e}')
    print(f'{len(pruebas) - fallos}/{len(pruebas)} pruebas correctas')
    sys.exit(1 if fallos else 0)
