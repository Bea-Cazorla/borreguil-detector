"""Por qué se queda o se quita cada variable con el VIF o el PCA (25-11-2026).

La autora pidió que, al aplicar el VIF o el PCA en «1 · Datos», la app enseñe qué
variables se quedan «y algún output del análisis para entender por qué».
mapeo_rf.por_que_vif / por_que_pca dan una fila por variable y texto_por_que, la frase.

    python tests/test_por_que_seleccion.py     (o con pytest)
"""
import os
import sys

import sklearn  # noqa: F401  (antes que nada que cargue DLL: Smart App Control)
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mapeo_rf as mr  # noqa: E402

NOMBRES = ['elev_dem_m', 'ndvi_early', 'ndvi_drop', 'ndwi_sd', 'albedo_sd', 'slope_deg',
           'cte']


def _datos():
    """ndvi_drop casi igual que ndvi_early; albedo_sd = ndvi_early − ndwi_sd (exacta);
    slope_deg suelta; cte no varía."""
    rng = np.random.default_rng(0)
    n = 60
    a = rng.normal(size=n)
    b = a + rng.normal(scale=0.1, size=n)
    c = rng.normal(size=n)
    d = a - c
    e = rng.normal(size=n)
    elev = rng.normal(size=n) + a
    X = np.column_stack([elev, a, b, c, d, e, np.ones(n)])
    y = (a + rng.normal(scale=0.5, size=n) > 0).astype(int)
    return X, y


def test_vif_una_fila_por_variable_y_su_motivo():
    X, _ = _datos()
    s = mr.elegir_por_vif(X, NOMBRES, 10, ['elev_dem_m'])
    filas = mr.por_que_vif(X, NOMBRES, s)
    por = {f['variable']: f for f in filas}
    assert sorted(por) == sorted(NOMBRES)
    # primero las que se quedan, en su orden; después las quitadas por pasos
    assert [f['variable'] for f in filas[:len(s['quedan'])]] == s['quedan']
    assert por['elev_dem_m']['que'] == 'fija'
    assert por['cte']['que'] == 'no_varia' and por['cte']['vif'] is None
    drop = por['ndvi_drop']
    assert drop['que'] == 'quitada' and drop['vif'] > 10 and drop['explicado_pct'] > 90
    # se parece sobre todo a la variable de la que es casi una copia, que se queda
    assert drop['parecida'] == 'ndvi_early' and drop['r'] > 0.95
    # cada variable que se queda está por debajo del umbral
    for n in s['quedan']:
        if n not in s['fijas']:
            assert por[n]['que'] == 'queda' and por[n]['vif'] <= 10


def test_vif_frases():
    X, _ = _datos()
    s = mr.elegir_por_vif(X, NOMBRES, 10, ['elev_dem_m'])
    por = {f['variable']: f for f in mr.por_que_vif(X, NOMBRES, s)}
    t = mr.texto_por_que(por['ndvi_drop'], 'vif', umbral=10)
    pct = f"{por['ndvi_drop']['explicado_pct']:.1f}"          # 98.9: con un decimal
    assert t.startswith(f'Las demás explicaban el {pct} %') and \
        'más que el umbral (90 %)' in t
    # cerca del umbral se ve la diferencia (90,2 % frente al 90 %)
    casi = dict(por['ndvi_drop'], vif=10.234, explicado_pct=100 * (1 - 1 / 10.234))
    assert 'explicaban el 90.2 % de ella' in mr.texto_por_que(casi, 'vif', umbral=10)
    assert 'umbral (86.7 %)' in mr.texto_por_que(casi, 'vif', umbral=7.5)
    assert 'Se parece sobre todo a ndvi_early (r 0.994), que se queda.' in t
    assert mr.texto_por_que(por['elev_dem_m'], 'vif', umbral=10).startswith(
        'Se queda siempre (la elegiste así).')
    assert mr.texto_por_que(por['cte'], 'vif') == \
        'No varía en las muestras: no aporta nada.'
    infinita = [f for f in por.values() if f['vif'] == float('inf')]
    assert infinita and mr.texto_por_que(infinita[0], 'vif', umbral=10).startswith(
        'Se obtiene exactamente de otras (VIF ∞).')
    # en inglés y con la coma del español a través de `num`
    en = mr.texto_por_que(por['ndvi_drop'], 'vif', en=True, umbral=10)
    assert en.startswith(f'The others explained {pct} % of it')
    coma = mr.texto_por_que(por['ndvi_drop'], 'vif', umbral=10,
                            num=lambda v, nd=3: f'{v:.{nd}f}'.replace('.', ','))
    assert '(r 0,994)' in coma


def test_pca_devuelve_todos_los_ejes_y_las_cargas():
    X, y = _datos()
    s = mr.elegir_por_pca(X, y, NOMBRES, ['elev_dem_m'], 0.65)
    vp = s['valores_propios']
    assert len(vp) == 5 and vp == sorted(vp, reverse=True)       # 5 que entran y varían
    assert len(s['ejes']) == sum(v > 1 for v in vp)
    assert set(s['cargas']) == {'ndvi_early', 'ndvi_drop', 'ndwi_sd', 'albedo_sd',
                                'slope_deg'}
    assert all(len(c) == len(s['ejes']) for c in s['cargas'].values())
    assert set(s['separa']) == set(s['cargas'])


def test_pca_una_fila_por_variable_y_su_motivo():
    X, y = _datos()
    s = mr.elegir_por_pca(X, y, NOMBRES, ['elev_dem_m'], 0.65)
    filas = mr.por_que_pca(s, NOMBRES)
    por = {f['variable']: f for f in filas}
    assert sorted(por) == sorted(NOMBRES)
    assert por['elev_dem_m']['que'] == 'fija' and por['cte']['que'] == 'no_varia'
    quedan = sorted(f['variable'] for f in filas if f['que'] in ('fija', 'elegida'))
    assert quedan == sorted(s['quedan'])
    assert por['slope_deg']['que'] == 'suelta' and abs(por['slope_deg']['carga']) < 0.65
    # una ligada sabe cuál se queda en su eje y lo que separa cada una
    lig = [f for f in filas if f['que'] == 'ligada']
    assert lig and all(f['ganadora'] in s['quedan'] for f in lig)
    assert all(f['separa_ganadora'] >= f['separa'] for f in lig)
    # primero las que se quedan
    assert [f['que'] for f in filas[:len(quedan)]] and \
        all(f['que'] in ('fija', 'elegida') for f in filas[:len(quedan)])


def test_pca_frases():
    X, y = _datos()
    s = mr.elegir_por_pca(X, y, NOMBRES, ['elev_dem_m'], 0.65)
    por = {f['variable']: f for f in mr.por_que_pca(s, NOMBRES)}
    elegida = next(f for f in por.values() if f['que'] == 'elegida')
    assert 'es la que mejor separa las categorías por sí sola' in \
        mr.texto_por_que(elegida, 'pca', carga=0.65)
    t = mr.texto_por_que(por['slope_deg'], 'pca', carga=0.65)
    assert t.startswith('No está muy ligada a ningún eje importante') and \
        'hace falta 0.650' in t
    for f in por.values():
        if f['que'] == 'ligada':
            t = mr.texto_por_que(f, 'pca', carga=0.65)
            assert f'se queda {f["ganadora"]}' in t
            # en un empate no dice «separa mejor»
            if f'{f["separa"]:.3f}' == f'{f["separa_ganadora"]:.3f}':
                assert 'separa igual de bien' in t
            else:
                assert 'que separa mejor' in t
    assert mr.texto_por_que(por['elev_dem_m'], 'pca', en=True) == \
        'Always stays (you chose so): it is left out of the PCA.'


if __name__ == '__main__':
    for nombre, f in list(globals().items()):
        if nombre.startswith('test_'):
            f()
            print('ok', nombre)
