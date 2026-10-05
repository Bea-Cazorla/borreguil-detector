"""Pruebas del Random Forest del mapeo: validación por zonas y modelo final.

Se ejecutan con `python tests/test_mapeo_modelo.py`; no necesitan red. Los datos son
inventados: tres categorías que una variable separa bien (o no separa, según la
prueba), repartidas por zonas.
"""
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import borreguil_pipeline as bp   # noqa: E402
import gee_backend as geb         # noqa: E402
import mapeo_rf as mr             # noqa: E402
from test_mapeo_datos import ausencia, es_borreguil, revisado   # noqa: E402

bp._imports()

RAPIDO = {'n_estimators': 60}     # pocos árboles: las pruebas van de la lógica


def ruido(*claves):
    return (zlib.crc32('|'.join(str(c) for c in claves).encode()) % 1000) / 1000.0


def datos(n=(20, 20, 20), zonas=6, cuencas=True, separable=True, zona_de=None):
    """Puntos de tres categorías (seco, húmedo, no borreguil) y lo que «leería»
    Earth Engine en cada uno. `n` son las muestras de cada categoría; `zona_de`
    permite fijar a mano la zona de una categoría."""
    rows, valores, k = [], {}, 0
    centro = {'seco': (0.05, 0.60), 'húmedo': (0.30, 0.60), 'no borreguil': (-0.10, 0.15)}
    for cat, cuantas in zip(('seco', 'húmedo', 'no borreguil'), n):
        for j in range(cuantas):
            z = (zona_de or {}).get(cat, j % zonas)
            lon, lat = -3.40 + z * 0.10, 37.00 + k * 0.0005      # una celda por zona
            r = (ausencia(k, lon=lon, lat=lat) if cat == 'no borreguil'
                 else revisado(k, humedad=cat, lon=lon, lat=lat))
            if cuencas:
                r['cuenca_id'] = f'c{z}'
            rows.append(r)
            v = {b: ruido(k, b) for b in geb.BANDAS_PILA}
            if separable:
                v['ndmi_mean'] = centro[cat][0] + 0.04 * (ruido(k, 'a') - 0.5)
                v['ndvi_late'] = centro[cat][1] + 0.04 * (ruido(k, 'b') - 0.5)
            valores[mr.clave_punto(lon, lat)] = v
            k += 1
    return rows, valores


def tabla(predictores=None, **kw):
    rows, valores = datos(**kw)
    res = mr.muestras(rows, 'humedad', es_borreguil=es_borreguil)
    tab = mr.construir_tabla(res, valores, predictores or mr.nombres_predictores())
    return tab, mr.grupos_espaciales(tab, rows)[0], rows


# ------------------------------------------------------------------ zonas
def test_la_zona_es_la_cuenca_o_si_no_la_celda():
    tab, _, rows = tabla(n=(6, 6, 6), zonas=3)
    grupos, info = mr.grupos_espaciales(tab, rows)
    assert sorted(set(grupos)) == ['cuenca c0', 'cuenca c1', 'cuenca c2']
    assert info == {'n_zonas': 3, 'con_cuenca': 18, 'con_celda': 0, 'celda_grados': 0.05}
    for r in rows[:9]:                                   # a la mitad se les quita
        r['cuenca_id'] = float('nan') if r['ID'] != 'p0' else ''
    grupos, info = mr.grupos_espaciales(tab, rows)
    assert info['con_cuenca'] == 9 and info['con_celda'] == 9
    assert grupos[0].startswith('celda ') and grupos[-1].startswith('cuenca ')
    tab, grupos, _ = tabla(n=(6, 6, 6), zonas=3, cuencas=False)
    assert len(set(grupos)) == 3 and all(g.startswith('celda ') for g in grupos)


# ------------------------------------------------- validar y modelo final
def test_valida_por_zonas_y_deja_el_modelo_final():
    tab, grupos, _ = tabla()
    r = mr.entrenar_y_validar(tab, grupos, RAPIDO)
    assert r['validacion'] == {'tipo': 'espacial', 'particiones': 5, 'n_zonas': 6,
                               'avisos': []}
    assert [c['categoria'] for c in r['categorias']] == ['no borreguil', 'húmedo', 'seco']
    # matriz de confusión: filas = lo real, columnas = lo predicho; suma = muestras
    assert len(r['matriz']) == 3 and all(len(f) == 3 for f in r['matriz'])
    assert sum(map(sum, r['matriz'])) == 60 and [sum(f) for f in r['matriz']] == [20, 20, 20]
    assert [c['n_validacion'] for c in r['por_clase']] == [20, 20, 20]
    # los datos son separables: tiene que acertar casi todo
    assert r['macro_f1'] > 0.95 and r['exactitud_equilibrada'] > 0.95, r['macro_f1']
    assert all(0.9 < c['precision'] <= 1 and 0.9 < c['recall'] <= 1 and 0.9 < c['f1'] <= 1
               for c in r['por_clase'])
    # cada muestra se predice UNA vez, cuando le toca ser validada
    assert len(r['predicciones']) == 60
    assert sorted(set(p['parte'] for p in r['predicciones'])) == [1, 2, 3, 4, 5]
    assert sum(p['n_valida'] for p in r['pliegues']) == 60
    assert all(p['n_entrena'] + p['n_valida'] == 60 for p in r['pliegues'])
    # la confianza es la probabilidad de la categoría elegida: con tres, más de 1/3
    assert all(1 / 3 < p['confianza'] <= 1 for p in r['predicciones'])
    # el modelo final se entrena con todas
    assert r['n_muestras'] == 60 and r['n_entrenadas'] == 60
    assert list(r['modelo'].classes_) == [0, 1, 2]
    assert r['modelo'].n_features_in_ == len(tab['predictores']) == len(r['predictores'])


def test_una_zona_nunca_esta_a_la_vez_en_entrenamiento_y_validacion():
    tab, grupos, _ = tabla()
    r = mr.entrenar_y_validar(tab, grupos, RAPIDO)
    vistas = [set(p['zonas_valida']) for p in r['pliegues']]
    for a in range(len(vistas)):
        for b in range(a + 1, len(vistas)):
            assert not vistas[a] & vistas[b], (a, b)          # ninguna zona repetida
    assert set().union(*vistas) == set(grupos)                 # y todas se validan
    for p in r['predicciones']:                                # cada muestra, con su zona
        assert p['zona'] in vistas[p['parte'] - 1]


def test_con_pocas_zonas_no_puede_validar_por_zonas_y_lo_dice():
    tab, grupos, _ = tabla(zonas=2)
    r = mr.entrenar_y_validar(tab, grupos, RAPIDO)
    assert r['validacion']['tipo'] == 'estratificada' and r['validacion']['n_zonas'] == 2
    assert r['validacion']['avisos'] == [('no_espacial', 2)]
    texto = mr.mensaje_validacion(('no_espacial', 2))
    assert 'no se ha podido' in texto and 'mejor de lo que es' in texto
    assert 'by zones' in mr.mensaje_validacion(('no_espacial', 2), en=True)


def test_categoria_que_vive_en_una_sola_zona():
    tab, grupos, _ = tabla(zona_de={'húmedo': 0})
    r = mr.entrenar_y_validar(tab, grupos, RAPIDO)
    assert ('categoria_en_una_zona', 'húmedo') in r['validacion']['avisos']
    humedo = [c for c in r['por_clase'] if c['categoria'] == 'húmedo'][0]
    assert humedo['recall'] == 0.0                # al apartar su zona, nadie se la enseñó
    assert any('húmedo' in p['categorias_sin_entrenar'] for p in r['pliegues'])
    assert r['confusiones'][0][0] == 'húmedo' and r['confusiones'][0][2] > 0
    assert 'no ha visto nunca' in mr.mensaje_validacion(('categoria_en_una_zona', 'húmedo'))
    # El modelo final sí la conoce: se entrena con todas las zonas.
    assert 1 in r['modelo'].classes_


def test_sin_senal_las_metricas_lo_reflejan():
    tab, grupos, _ = tabla(separable=False)
    r = mr.entrenar_y_validar(tab, grupos, RAPIDO)
    assert r['macro_f1'] < 0.6                    # con ruido no se puede acertar
    assert r['confusiones'] and sum(c[2] for c in r['confusiones']) > 20


def test_es_reproducible():
    tab, grupos, _ = tabla()
    a = mr.entrenar_y_validar(tab, grupos, RAPIDO)
    b = mr.entrenar_y_validar(tab, grupos, RAPIDO)
    assert a['matriz'] == b['matriz'] and a['macro_f1'] == b['macro_f1']
    assert [p['predicha'] for p in a['predicciones']] == [p['predicha'] for p in b['predicciones']]
    # (los árboles se suman en paralelo: puede bailar el último decimal, nada más)
    dif = abs(a['modelo'].predict_proba(tab['X']) - b['modelo'].predict_proba(tab['X']))
    assert dif.max() < 1e-12
    assert a['firma'] == b['firma'] == mr.firma_tabla(tab)


# -------------------------------------------------------------- balance
def test_submuestreo_recorta_solo_las_abundantes_y_es_reproducible():
    y = [0] * 60 + [1] * 10 + [2] * 12
    q = mr.submuestrear(y, 3.0, 42)
    assert [sum(1 for i in q if y[i] == c) for c in (0, 1, 2)] == [30, 10, 12]
    assert q == mr.submuestrear(y, 3.0, 42) and q != mr.submuestrear(y, 3.0, 7)
    assert q == sorted(q) and len(set(q)) == len(q)
    assert mr.submuestrear([0] * 5 + [1] * 5, 3.0, 42) == list(range(10))   # nada que recortar


def test_estrategias_de_balance_quedan_registradas():
    tab, grupos, _ = tabla(n=(60, 10, 10))
    r = mr.entrenar_y_validar(tab, grupos, RAPIDO)                 # por defecto: pesos
    assert r['balance'] == {
        'estrategia': 'pesos', 'razon_maxima': None, 'class_weight': 'balanced',
        'original': {'no borreguil': 10, 'húmedo': 10, 'seco': 60},
        'final': {'no borreguil': 10, 'húmedo': 10, 'seco': 60}}
    assert r['modelo'].class_weight == 'balanced' and r['n_entrenadas'] == 80
    r = mr.entrenar_y_validar(tab, grupos, RAPIDO, estrategia='submuestreo')
    assert r['balance']['final'] == {'no borreguil': 10, 'húmedo': 10, 'seco': 30}
    assert r['balance']['razon_maxima'] == 3.0 and r['modelo'].class_weight is None
    assert r['n_muestras'] == 80 and r['n_entrenadas'] == 50
    assert sum(map(sum, r['matriz'])) == 80        # se valida con TODAS, no con el recorte
    r = mr.entrenar_y_validar(tab, grupos, RAPIDO, estrategia='ambas', razon_maxima=2.0)
    assert r['balance']['final']['seco'] == 20 and r['modelo'].class_weight == 'balanced'
    r = mr.entrenar_y_validar(tab, grupos, RAPIDO, estrategia='ninguna')
    assert r['balance']['final']['seco'] == 60 and r['modelo'].class_weight is None


# ------------------------------------------------------------ parámetros
def test_parametros_configurables_y_validados():
    tab, grupos, _ = tabla()
    r = mr.entrenar_y_validar(tab, grupos, {'n_estimators': 25, 'max_depth': 4,
                                            'min_samples_leaf': 3, 'max_features': 'log2',
                                            'random_state': 7})
    m = r['modelo']
    assert (m.n_estimators, m.max_depth, m.min_samples_leaf, m.max_features,
            m.random_state) == (25, 4, 3, 'log2', 7)
    assert r['params'] == {'n_estimators': 25, 'max_features': 'log2', 'max_depth': 4,
                           'min_samples_leaf': 3, 'random_state': 7}
    r = mr.entrenar_y_validar(tab, grupos, RAPIDO)             # el resto, por defecto
    assert r['params'] == {**mr.PARAMS_DEFECTO, 'n_estimators': 60}
    for malo in ({'n_estimators': 3}, {'max_features': 'todas'}, {'max_depth': 0},
                 {'min_samples_leaf': 0}, {'random_state': 'x'}):
        try:
            mr.entrenar_y_validar(tab, grupos, malo)
        except ValueError as e:
            assert list(malo)[0] in str(e)
        else:
            raise AssertionError(f'debía rechazar {malo}')


def test_se_niega_a_entrenar_con_una_tabla_que_no_vale():
    tab, grupos, _ = tabla(n=(20, 3, 20))                      # «húmedo» por debajo del mínimo
    for llamada, trozo in (
            (lambda: mr.entrenar_y_validar(tab, grupos, RAPIDO), 'mínimo es 5'),
            (lambda: mr.entrenar_y_validar(*tabla()[:1], ['z1'], RAPIDO), 'zona de cada muestra'),
            (lambda: mr.entrenar_y_validar(*tabla()[:2], RAPIDO, estrategia='smote'),
             'estrategia desconocida')):
        try:
            llamada()
        except ValueError as e:
            assert trozo in str(e), str(e)
        else:
            raise AssertionError(f'debía negarse: {trozo}')
    tab, grupos, _ = tabla()
    tab['X'][3][2] = float('nan')
    try:
        mr.entrenar_y_validar(tab, grupos, RAPIDO)
    except ValueError as e:
        assert 'no válidos' in str(e)
    else:
        raise AssertionError('debía negarse con un hueco en la tabla')


def test_la_firma_cambia_si_cambia_la_tabla():
    tab, _, _ = tabla()
    otra, _, _ = tabla(predictores=mr.nombres_predictores()[:10])
    menos, _, _ = tabla(n=(20, 20, 19))
    assert len({mr.firma_tabla(tab), mr.firma_tabla(otra), mr.firma_tabla(menos)}) == 3
    assert mr.firma_tabla(tab) == mr.firma_tabla(tabla()[0])


def test_el_modelo_ve_las_variables_en_el_orden_de_la_tabla():
    usar = ['slope_deg', 'ndmi_mean', 'ndvi_late']             # se dan desordenadas
    tab, grupos, _ = tabla(predictores=usar)
    r = mr.entrenar_y_validar(tab, grupos, RAPIDO)
    assert r['predictores'] == ['ndvi_late', 'ndmi_mean', 'slope_deg']   # orden del catálogo
    imp = dict(zip(r['predictores'], r['modelo'].feature_importances_))
    assert imp['slope_deg'] < 0.1 < imp['ndmi_mean']           # la que es ruido no pesa
    prob = r['modelo'].predict_proba(tab['X'][:4])
    assert prob.shape == (4, 3) and abs(prob.sum(axis=1) - 1).max() < 1e-9


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
