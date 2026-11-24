"""Pruebas de elegir las variables del mapa con un PCA (mapeo_rf.elegir_por_pca).

Del PCA de la matriz de correlación, los ejes con valor propio mayor que 1; de cada eje,
entre las variables muy ligadas a él (|correlación| ≥ 0,65), la que mejor separa las
categorías por sí sola; las fijas (la altitud) quedan fuera del PCA y se añaden. Se
ejecutan con `python tests/test_pca.py` (o con pytest); no necesitan red.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sklearn.ensemble  # noqa: F401,E402  (primero, por Smart App Control)
import numpy as np                 # noqa: E402

import mapeo_rf as mr             # noqa: E402


def datos(n=300, semilla=0):
    """Dos grupos de variables casi iguales (verdor y agua) y la altitud. La categoría
    depende sobre todo de `ndvi_early`, y algo de `ndwi_mean`."""
    rng = np.random.default_rng(semilla)
    verde, agua = rng.normal(size=n), rng.normal(size=n)
    y = (verde + 0.3 * agua + rng.normal(scale=0.5, size=n) > 0).astype(int)
    cols = {
        'ndvi_early': verde + rng.normal(scale=0.05, size=n),     # la que más separa
        'evi_early': verde + rng.normal(scale=0.4, size=n),
        'clre_early': verde + rng.normal(scale=0.4, size=n),
        'ndwi_mean': agua + rng.normal(scale=0.05, size=n),
        'ndwi_min': agua + rng.normal(scale=0.4, size=n),
        'elev_dem_m': rng.normal(2500, 200, size=n),
    }
    return np.column_stack(list(cols.values())), y, list(cols)


def test_una_por_eje_y_la_que_mejor_separa():
    X, y, nombres = datos()
    s = mr.elegir_por_pca(X, y, nombres)
    assert len(s['ejes']) == 2                                 # verdor y agua
    # del verdor, la que más separa; del agua (casi no separa), una de las dos
    agua = s['ejes'][1]['elegida']
    assert s['ejes'][0]['elegida'] == 'ndvi_early' and agua in ('ndwi_mean', 'ndwi_min')
    assert s['quedan'] == [n for n in nombres if n in ('ndvi_early', agua, 'elev_dem_m')]
    assert s['fijas'] == ['elev_dem_m'] and 50 < s['varianza_pct'] <= 100
    # la altitud no entra en el PCA
    assert all(d['variable'] != 'elev_dem_m' for e in s['ejes'] for d in e['variables'])
    assert not s['pocas_muestras']


def test_sin_fijas_y_con_correlacion_mas_exigente():
    X, y, nombres = datos()
    s = mr.elegir_por_pca(X, y, nombres, fijas=())
    assert 'elev_dem_m' not in s['quedan'] or any(
        e['elegida'] == 'elev_dem_m' for e in s['ejes'])
    s2 = mr.elegir_por_pca(X, y, nombres, carga=0.95)
    assert all(abs(d['carga']) >= 0.95 for e in s2['ejes'] for d in e['variables'])
    assert len(s2['quedan']) >= 2                              # siempre queda alguna


def test_una_variable_que_no_varia_se_aparta():
    X, y, nombres = datos()
    X = np.column_stack([X, np.full(len(y), 3.0)])
    s = mr.elegir_por_pca(X, y, nombres + ['curvature'])
    assert s['sin_variar'] == ['curvature'] and 'curvature' not in s['quedan']


def test_separacion_con_dos_y_con_mas_categorias():
    x = np.arange(20, dtype=float)
    assert mr.separacion(x, (x >= 10).astype(int)) == 1.0      # separa del todo
    assert mr.separacion(np.ones(20), (x >= 10).astype(int)) == 0.0
    y3 = np.repeat([0, 1, 2], [7, 7, 6])
    assert mr.separacion(x, y3) > mr.separacion(np.random.default_rng(0).normal(size=20), y3)


def test_pocas_muestras_y_errores():
    X, y, nombres = datos(n=8)
    assert mr.elegir_por_pca(X, y, nombres)['pocas_muestras']
    for mal in (dict(carga=0), dict(carga=1.2)):
        try:
            mr.elegir_por_pca(*datos()[:2], datos()[2], **mal)
            assert False, mal
        except ValueError:
            pass
    X, y, nombres = datos()
    X[0, 0] = np.nan
    try:
        mr.elegir_por_pca(X, y, nombres)
        assert False
    except ValueError:
        pass


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
