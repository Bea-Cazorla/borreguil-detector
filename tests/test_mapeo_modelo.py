"""Pruebas del Random Forest del mapeo: validación por zonas y modelo final.

Se ejecutan con `python tests/test_mapeo_modelo.py`; no necesitan red. Los datos son
inventados: tres categorías que una variable separa bien (o no separa, según la
prueba), repartidas por zonas.
"""
import sys
import zlib
from collections import Counter
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
    # la confianza es la probabilidad de la categoría elegida
    assert all(0 < p['confianza'] <= 1 for p in r['predicciones'])
    assert all(0 <= p['p_borreguil'] <= 1 for p in r['predicciones'])
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


# ---------------------------------- nivel combinado: ambiente con la humedad dentro
def test_el_nivel_combinado_entrena_con_sus_seis_categorias():
    cats = list(mr.NIVELES['ambiente_humedad'])
    rows, valores, k = [], {}, 0
    for c, cat in enumerate(['no borreguil'] + cats):
        for j in range(10):
            z = j % 5
            lon, lat = -3.40 + z * 0.10, 37.00 + k * 0.0005
            if cat == 'no borreguil':
                r = ausencia(k, lon=lon, lat=lat)
            else:
                ambiente, humedad = cat.split(' · ')
                r = revisado(k, humedad=humedad, ambiente=ambiente, lon=lon, lat=lat)
            r['cuenca_id'] = f'c{z}'
            rows.append(r)
            v = {b: ruido(k, b) for b in geb.BANDAS_PILA}
            v['ndmi_mean'] = 0.10 * c + 0.02 * (ruido(k, 'a') - 0.5)   # las separa una variable
            valores[mr.clave_punto(lon, lat)] = v
            k += 1
    res = mr.muestras(rows, 'ambiente_humedad', es_borreguil=es_borreguil)
    assert res['listo'] and res['n_usadas'] == 70
    tab = mr.construir_tabla(res, valores, mr.nombres_predictores())
    # probando todas las variables en cada división, la que las separa no se pierde
    # entre las 33 que son ruido
    r = mr.entrenar_y_validar(tab, mr.grupos_espaciales(tab, rows)[0],
                              {**RAPIDO, 'max_features': None})
    assert r['nivel'] == 'ambiente_humedad'
    assert [c['categoria'] for c in r['categorias']] == ['no borreguil'] + cats
    assert [c['codigo'] for c in r['categorias']] == list(range(7))
    assert len(r['matriz']) == 7 and sum(map(sum, r['matriz'])) == 70
    assert r['validacion']['tipo'] == 'espacial' and r['macro_f1'] > 0.9
    assert [int(c) for c in r['modelo'].classes_] == list(range(7))
    # Lo que más se confunde, si algo, se nombra con la categoría entera.
    assert all(' · ' in a or a == 'no borreguil' for a, _, _ in r['confusiones'])


# ------------------------------------------------- importancia de las variables
def test_el_f1_medio_propio_da_lo_mismo_que_el_de_scikit_learn():
    import numpy as np
    from sklearn.metrics import f1_score
    azar = np.random.default_rng(3)
    y = azar.integers(0, 4, 200)
    presentes = np.array(sorted(set(int(c) for c in y)))
    pred = azar.integers(0, 5, (6, 7, 200))       # la 4 no existe; a veces falta alguna
    pred[0, 0] = 2                                # todo a una categoría
    pred[0, 1] = y                                # todo acertado
    f1 = mr._macro_f1(y, pred, presentes)
    assert f1.shape == (6, 7) and f1[0, 1] == 1.0
    for a in range(6):
        for b in range(7):
            esperado = f1_score(y, pred[a, b], average='macro', labels=presentes,
                                zero_division=0)
            assert abs(f1[a, b] - esperado) < 1e-12, (a, b)
    assert abs(float(mr._macro_f1(y, pred[3, 3], presentes)) - f1[3, 3]) < 1e-15


def test_la_importancia_senala_la_variable_que_separa_las_categorias():
    # `ndmi_mean` separa las tres categorías; `ndvi_late`, solo el «no borreguil» del
    # resto; las otras 32 son ruido.
    tab, grupos, _ = tabla()
    r = mr.entrenar_y_validar(tab, grupos, RAPIDO)
    imp = r['importancia']
    assert imp['medida'] == 'macro_f1' and abs(imp['base'] - r['macro_f1']) < 1e-12
    assert imp['repeticiones'] == mr.REPETICIONES_MAX          # pocas muestras: el máximo
    v = imp['variables']
    assert len(v) == 34 and sorted(x['variable'] for x in v) == sorted(r['predictores'])
    assert [x['orden'] for x in v] == list(range(1, 35))       # de la que más a la que menos
    assert all(a['permutacion'] >= b['permutacion'] for a, b in zip(v, v[1:]))
    assert sorted(x['orden_impureza'] for x in v) == list(range(1, 35))
    por = {x['variable']: x for x in v}
    # Sin `ndmi_mean` el modelo deja de distinguir seco de húmedo: el F1 se hunde.
    assert v[0]['variable'] == 'ndmi_mean' and v[0]['permutacion'] > 0.3, v[0]
    assert por['ndmi_mean']['grupo'] == mr.G_PERIODO
    # Sin `ndvi_late` pierde menos: solo le ayudaba a apartar el «no borreguil».
    assert v[1]['variable'] == 'ndvi_late'
    assert 0.1 < v[1]['permutacion'] < v[0]['permutacion'] - 0.1, v[1]
    # Por dentro, el bosque usa las dos más que ninguna otra.
    assert (por['ndmi_mean']['orden_impureza'], por['ndvi_late']['orden_impureza']) == (1, 2)
    assert por['ndvi_late']['impureza'] > 0.1
    # El ruido no pesa (y puede salir un pelo negativo: es azar, no un error).
    ruidos = [x for x in v if x['variable'] not in ('ndmi_mean', 'ndvi_late')]
    assert all(abs(x['permutacion']) < 0.05 and x['impureza'] < 0.05 for x in ruidos)
    assert all(x['desviacion'] >= 0 for x in v)
    assert abs(sum(x['impureza'] for x in v) - 1) < 1e-9
    assert [x['impureza'] for x in sorted(v, key=lambda x: x['orden_impureza'])] == sorted(
        (x['impureza'] for x in v), reverse=True)
    # coincide con lo que guarda el modelo final
    interno = dict(zip(r['predictores'], r['modelo'].feature_importances_))
    assert all(abs(x['impureza'] - interno[x['variable']]) < 1e-15 for x in v)


def test_cada_variable_tiene_su_familia():
    assert set(mr.familia(n) for n in mr.nombres_predictores()) == set(mr.FAMILIAS)
    assert [mr.familia(n) for n in ('ndvi_early', 'ndvi_late', 'ndvi_drop')] == ['ndvi'] * 3
    assert mr.familia('ndmi_sd') == 'ndmi' and mr.familia('albedo_max') == 'albedo'
    assert all(mr.familia(n) == 'topo' for n in mr.nombres_predictores([mr.G_TOPO]))
    cuantas = Counter(mr.familia(n) for n in mr.nombres_predictores())
    assert cuantas == {'ndvi': 3, 'ndwi': 6, 'clre': 2, 'ndmi': 6, 'evi': 2, 'nbr': 2,
                       'wavi': 4, 'albedo': 4, 'topo': 5}, cuantas
    assert mr.nombre_familia('ndmi') == 'NDMI · humedad de la vegetación'
    assert mr.nombre_familia('topo', en=True) == 'Topography'


def test_la_importancia_tambien_se_da_por_familias_y_por_grupos():
    tab, grupos, _ = tabla()
    imp = mr.entrenar_y_validar(tab, grupos, RAPIDO)['importancia']
    por = {x['variable']: x for x in imp['variables']}
    assert por['ndmi_mean']['familia'] == 'ndmi' and por['slope_deg']['familia'] == 'topo'
    # por grupos del catálogo (la época o el origen)
    g = {x['grupo']: x for x in imp['grupos']}
    assert set(g) == set(mr.GRUPOS)
    assert [g[k]['n_variables'] for k in mr.GRUPOS] == [6, 6, 1, 16, 5]
    assert imp['grupos'][0]['grupo'] == mr.G_PERIODO            # el de `ndmi_mean`
    assert g[mr.G_PERIODO]['permutacion'] >= por['ndmi_mean']['permutacion'] - 0.05
    # por familias (el mismo índice en distintos momentos)
    f = {x['familia']: x for x in imp['familias']}
    assert set(f) == set(mr.FAMILIAS) and f['ndmi']['n_variables'] == 6
    assert [x['familia'] for x in imp['familias'][:2]] == ['ndmi', 'ndvi']
    assert f['ndmi']['permutacion'] >= por['ndmi_mean']['permutacion'] - 0.05
    assert f['ndvi']['permutacion'] >= por['ndvi_late']['permutacion'] - 0.05
    for conjuntos in (imp['grupos'], imp['familias']):
        assert sum(x['n_variables'] for x in conjuntos) == 34
        assert abs(sum(x['impureza'] for x in conjuntos) - 1) < 1e-9
        assert all(a['permutacion'] >= b['permutacion']
                   for a, b in zip(conjuntos, conjuntos[1:]))
    # La topografía es a la vez un grupo y una familia: se mide una sola vez.
    assert (g[mr.G_TOPO]['permutacion'], g[mr.G_TOPO]['desviacion']) == (
        f['topo']['permutacion'], f['topo']['desviacion'])
    # Un grupo de una sola variable es esa variable.
    assert (g[mr.G_CAMBIO]['permutacion'], g[mr.G_CAMBIO]['desviacion']) == (
        por['ndvi_drop']['permutacion'], por['ndvi_drop']['desviacion'])
    # Con tres variables de tres grupos y tres familias, cada conjunto es su variable.
    tab, grupos, _ = tabla(predictores=['slope_deg', 'ndmi_mean', 'ndvi_late'])
    imp = mr.entrenar_y_validar(tab, grupos, RAPIDO)['importancia']
    assert len(imp['variables']) == len(imp['grupos']) == len(imp['familias']) == 3
    assert imp['repeticiones'] == mr.REPETICIONES_MAX
    for campo, conjuntos in (('grupo', imp['grupos']), ('familia', imp['familias'])):
        por = {x[campo]: x for x in imp['variables']}
        for x in conjuntos:
            assert x['n_variables'] == 1
            assert (x['permutacion'], x['desviacion'], x['impureza']) == (
                por[x[campo]]['permutacion'], por[x[campo]]['desviacion'],
                por[x[campo]]['impureza'])


def test_dos_variables_gemelas_se_reparten_el_peso_y_el_grupo_lo_junta():
    # La única señal está repetida en dos variables del mismo grupo: al desordenar una
    # sola, la otra la cubre en parte; al desordenar el grupo entero, el modelo se
    # queda a ciegas. El peso de cada gemela se queda corto; el del grupo, no.
    rows, valores = datos()
    for k, v in valores.items():
        v['ndvi_late'] = ruido(k, 'sin señal')
        v['ndmi_min'] = v['ndmi_mean']
    res = mr.muestras(rows, 'humedad', es_borreguil=es_borreguil)
    tab = mr.construir_tabla(res, valores, mr.nombres_predictores())
    r = mr.entrenar_y_validar(tab, mr.grupos_espaciales(tab, rows)[0], RAPIDO)
    imp = r['importancia']
    por = {x['variable']: x for x in imp['variables']}
    grupo = [x for x in imp['grupos'] if x['grupo'] == mr.G_PERIODO][0]
    fam = imp['familias'][0]
    assert r['macro_f1'] > 0.9
    assert grupo['permutacion'] > 0.5, grupo
    assert fam['familia'] == 'ndmi' and fam['permutacion'] > 0.5, fam
    for gemela in ('ndmi_mean', 'ndmi_min'):
        assert 0.1 < por[gemela]['permutacion'] < grupo['permutacion'] - 0.15, por[gemela]
        assert por[gemela]['permutacion'] < fam['permutacion'] - 0.15
    assert {por['ndmi_mean']['orden'], por['ndmi_min']['orden']} == {1, 2}
    assert {por['ndmi_mean']['orden_impureza'], por['ndmi_min']['orden_impureza']} == {1, 2}
    # y la que antes separaba y ahora es ruido ya no pesa
    assert abs(por['ndvi_late']['permutacion']) < 0.05


def test_la_importancia_sale_igual_cada_vez_y_como_se_trocee():
    tab, grupos, _ = tabla(n=(30, 12, 12), separable=False)     # ruido: todo son empates
    a = mr.entrenar_y_validar(tab, grupos, RAPIDO)['importancia']
    assert a == mr.entrenar_y_validar(tab, grupos, RAPIDO)['importancia']
    trozo, lote = mr.TROZO_PERMUTACION, mr.LOTE_PERMUTACION
    try:
        mr.TROZO_PERMUTACION, mr.LOTE_PERMUTACION = 700, 5_000  # muchos trozos y tandas
        assert a == mr.entrenar_y_validar(tab, grupos, RAPIDO)['importancia']
    finally:
        mr.TROZO_PERMUTACION, mr.LOTE_PERMUTACION = trozo, lote
    # con otra semilla cambian el bosque y los desórdenes
    otra = mr.entrenar_y_validar(tab, grupos, {**RAPIDO, 'random_state': 7})['importancia']
    assert otra != a
    # con muchas muestras se repite menos veces, pero nunca menos del mínimo
    filas = mr.FILAS_PERMUTADAS
    try:
        mr.FILAS_PERMUTADAS = 10_000
        pocas = mr.entrenar_y_validar(tab, grupos, RAPIDO)['importancia']
    finally:
        mr.FILAS_PERMUTADAS = filas
    assert pocas['repeticiones'] == mr.REPETICIONES_MIN and a['repeticiones'] > 5


def test_la_importancia_se_puede_guardar_en_csv():
    import csv
    import io
    tab, grupos, _ = tabla()
    r = mr.entrenar_y_validar(tab, grupos, RAPIDO)
    filas = list(csv.DictReader(io.StringIO(mr.importancia_csv(r))))
    assert len(filas) == 34
    assert list(filas[0]) == ['orden', 'variable', 'que_mide', 'familia', 'grupo',
                              'importancia_permutacion', 'desviacion',
                              'importancia_interna', 'orden_interno']
    assert [int(f['orden']) for f in filas] == list(range(1, 35))
    assert filas[0]['variable'] == 'ndmi_mean' and filas[0]['grupo'] == mr.G_PERIODO
    assert filas[0]['que_mide'] == 'NDMI, humedad: media'
    assert filas[0]['familia'] == 'NDMI · humedad de la vegetación'
    v = r['importancia']['variables'][0]
    assert filas[0]['importancia_permutacion'] == f"{v['permutacion']:.6f}"
    assert abs(sum(float(f['importancia_interna']) for f in filas) - 1) < 1e-4
    en = list(csv.DictReader(io.StringIO(mr.importancia_csv(r, en=True))))
    assert en[0]['que_mide'] == 'NDMI, moisture: mean'
    assert en[0]['familia'] == 'NDMI · vegetation moisture'
    assert en[0]['grupo'] == mr.GRUPOS_EN[mr.G_PERIODO]


# ------------------------------------------- ¿borreguil o no? La regla y su umbral
def test_la_regla_decide_en_dos_pasos_borreguil_o_no_y_despues_el_tipo():
    import numpy as np
    clases = [0, 1, 2]
    # El caso que motivó la regla: 0.40 de «no borreguil» y 0.60 repartido en dos tipos.
    p_borr, tipo, p_tipo, p_no = mr.repartir([[0.40, 0.35, 0.25]], clases)
    assert p_borr.dtype == np.float32 and p_tipo.dtype == np.float32
    assert abs(float(p_borr[0]) - 0.60) < 1e-6 and int(tipo[0]) == 1
    assert abs(float(p_tipo[0]) - 0.35) < 1e-6 and abs(float(p_no[0]) - 0.40) < 1e-6
    cat, p_cat, pb = mr.decidir([[0.40, 0.35, 0.25]], clases, 0.5)
    assert int(cat[0]) == 1 and abs(float(p_cat[0]) - 0.35) < 1e-6      # borreguil, húmedo
    assert int(mr.decidir([[0.40, 0.35, 0.25]], clases, 0.7)[0][0]) == 0   # no llega a 0.7
    # sin regla, la más probable de todas: «no borreguil»
    cat, p_cat, _ = mr.decidir([[0.40, 0.35, 0.25]], clases)
    assert int(cat[0]) == 0 and abs(float(p_cat[0]) - 0.40) < 1e-6
    # Con seis tipos es peor: 0.75 de ser borreguil y el voto simple diría que no.
    seis = [[0.25] + [0.125] * 6]
    assert int(mr.decidir(seis, range(7))[0][0]) == 0
    assert int(mr.decidir(seis, range(7), 0.5)[0][0]) == 1              # el de código más bajo
    # justo en el umbral cuenta como borreguil; los empates, al código más bajo
    assert int(mr.decidir([[0.5, 0.25, 0.25]], clases, 0.5)[0][0]) == 1
    assert int(mr.decidir([[0.5, 0.5, 0.0]], clases)[0][0]) == 0
    assert int(mr.decidir([[0.2, 0.4, 0.4]], clases, 0.5)[0][0]) == 1
    # sin la categoría «no borreguil» no hay umbral que valga: siempre el tipo
    cat, p_cat, pb = mr.decidir([[0.3, 0.7]], [1, 2], 0.99)
    assert int(cat[0]) == 2 and abs(float(p_cat[0]) - 0.7) < 1e-6
    assert mr.repartir([[0.3, 0.7]], [1, 2])[3] is None
    # muchas a la vez, con cualquier forma
    azar = np.random.default_rng(5)
    prob = azar.dirichlet([1, 1, 1, 1], size=(3, 50))
    cat, p_cat, pb = mr.decidir(prob, [0, 1, 2, 3], 0.5)
    assert cat.shape == p_cat.shape == pb.shape == (3, 50)
    assert np.allclose(pb, 1 - prob[..., 0], atol=1e-6)
    assert ((cat == 0) == (pb < np.float32(0.5))).all()
    assert np.allclose(p_cat, np.take_along_axis(prob, cat[..., None], axis=-1)[..., 0],
                       atol=1e-6)
    # Con 0.5, lo que el voto simple daba por borreguil lo sigue siendo, y del mismo tipo.
    simple = mr.decidir(prob, [0, 1, 2, 3])[0]
    assert (cat[simple != 0] == simple[simple != 0]).all() and (cat != 0).sum() > (simple != 0).sum()
    try:
        mr.repartir([[0.5, 0.5]], clases)
    except ValueError as e:
        assert 'una probabilidad por categoría' in str(e)
    else:
        raise AssertionError('debía rechazar una matriz que no casa con las categorías')


def test_el_barrido_cuenta_aciertos_y_fallos_con_cada_umbral():
    assert len(mr.UMBRALES) == 39 and mr.UMBRALES[0] == 0.025 and mr.UMBRALES[-1] == 0.975
    assert 0.5 in mr.UMBRALES
    # dos aciertos y dos fallos, uno de cada clase
    b = mr._binarias([0.2, 0.6, 0.4, 0.8], [False, False, True, True], 0.5)
    assert (b['vp'], b['fp'], b['fn'], b['vn']) == (1, 1, 1, 1)
    assert b['sensibilidad'] == b['especificidad'] == b['precision'] == b['f1'] == 0.5
    assert b['tss'] == 0.0 and b['umbral'] == 0.5
    # si no hay nada que dividir, la medida vale 0 y no falla
    b = mr._binarias([0.1, 0.2], [False, False], 0.5)
    assert (b['sensibilidad'], b['precision'], b['f1'], b['especificidad']) == (0, 0, 0, 1.0)
    # los dos grupos se separan del todo con cualquier umbral entre 0.3 y 0.6
    p = [0.1, 0.2, 0.3, 0.6, 0.7, 0.9]
    es = [False, False, False, True, True, True]
    barrido = mr.barrido_umbral(p, es)
    assert [x['umbral'] for x in barrido] == list(mr.UMBRALES)
    perfectos = [x['umbral'] for x in barrido if x['tss'] == 1.0]
    assert perfectos[0] == 0.325 and perfectos[-1] == 0.6 and len(perfectos) == 12
    # de los que empatan, el del centro (y de los dos del centro, el más cercano a 0.5)
    mejor = mr.elegir_umbral(barrido, 'tss')
    assert mejor['umbral'] == 0.475 and (mejor['vp'], mejor['vn'], mejor['fp'], mejor['fn']) == (3, 3, 0, 0)
    assert mr.elegir_umbral(barrido, 'f1')['umbral'] == 0.475
    # con más «no borreguil» que borreguil, el F1 y el TSS no tienen por qué coincidir
    p = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.45, 0.55, 0.4, 0.6, 0.9]
    es = [False] * 9 + [True] * 3
    barrido = mr.barrido_umbral(p, es)
    u_tss, u_f1 = (mr.elegir_umbral(barrido, c) for c in ('tss', 'f1'))
    assert u_tss['tss'] == max(x['tss'] for x in barrido)
    assert u_f1['f1'] == max(x['f1'] for x in barrido)
    assert u_tss['umbral'] < u_f1['umbral']          # el TSS prefiere no perder borreguiles


def _con_tipos_que_se_confunden():
    """Borreguiles que se distinguen de lo que no lo es solo a medias (una variable
    que se solapa), y dos tipos que no se distinguen entre sí: lo que hace que la
    probabilidad de borreguil se reparta."""
    rows, valores = datos(n=(25, 25, 25), zonas=8)
    for k, r in enumerate(rows):
        v = valores[mr.clave_punto(r['lon'], r['lat'])]
        es_no = bp._txt(r.get('Borreguil')).lower() == 'no'
        v['ndmi_mean'] = ruido(k, 'a')                              # ya no separa los tipos
        v['ndvi_late'] = (0.30 if es_no else 0.55) + 0.5 * (ruido(k, 'b') - 0.5)
    res = mr.muestras(rows, 'humedad', es_borreguil=es_borreguil)
    tab = mr.construir_tabla(res, valores, mr.nombres_predictores())
    return tab, mr.grupos_espaciales(tab, rows)[0]


def test_el_modelo_elige_el_umbral_con_la_validacion_y_lo_aplica():
    import numpy as np
    tab, grupos = _con_tipos_que_se_confunden()
    r = mr.entrenar_y_validar(tab, grupos, RAPIDO)
    regla = r['regla']
    # de entrada, el de mayor F1 (el borreguil es escaso: con el TSS se pasa)
    assert regla['criterio'] == 'f1' and regla['umbral'] in mr.UMBRALES
    assert regla['umbral'] == regla['mejor_f1'] and regla['mejor_tss'] in mr.UMBRALES
    assert [x['umbral'] for x in regla['barrido']] == list(mr.UMBRALES)
    assert regla['en_umbral']['f1'] == max(x['f1'] for x in regla['barrido'])
    # cada muestra: «no borreguil» si su probabilidad de borreguil no llega al umbral
    u = np.float32(regla['umbral'])
    for q in r['predicciones']:
        assert (q['predicha'] == 'no borreguil') == (np.float32(q['p_borreguil']) < u), q
    # las cuentas de borreguil sí/no son las de la matriz de confusión, plegada
    m = np.array(r['matriz'])                      # filas: lo que es; columnas: lo que dice
    e = regla['en_umbral']
    assert (e['vn'], e['fp'], e['fn'], e['vp']) == (
        m[0, 0], m[0, 1:].sum(), m[1:, 0].sum(), m[1:, 1:].sum())
    assert e['vp'] + e['fn'] == 50 and e['vn'] + e['fp'] == 25
    assert e['tss'] > 0.3                          # se distinguen, aunque no del todo
    # el peso de las variables parte del mismo F1 medio
    assert abs(r['importancia']['base'] - r['macro_f1']) < 1e-12
    assert all('macro_f1' in pl and 'exactitud' in pl for pl in r['pliegues'])


def test_cambiar_el_umbral_rehace_las_cifras_sin_volver_a_entrenar():
    import copy
    tab, grupos = _con_tipos_que_se_confunden()
    r = mr.entrenar_y_validar(tab, grupos, RAPIDO)
    modelo = r['modelo']
    antes = copy.deepcopy({k: r[k] for k in ('matriz', 'macro_f1', 'exactitud', 'por_clase',
                                             'predicciones', 'pliegues', 'importancia',
                                             'regla', 'confusiones')})

    def borreguiles(mod):
        return sum(q['predicha'] != 'no borreguil' for q in mod['predicciones'])

    cuantos = []
    for u in (0.1, 0.3, 0.5, 0.7, 0.9):
        assert mr.aplicar_regla(r, 'manual', u) is r and r['modelo'] is modelo
        assert r['regla']['criterio'] == 'manual' and r['regla']['umbral'] == u
        assert r['regla']['en_umbral']['umbral'] == u
        assert sum(map(sum, r['matriz'])) == 75
        assert abs(r['importancia']['base'] - r['macro_f1']) < 1e-12
        cuantos.append(borreguiles(r))
    # cuanto más alto el umbral, menos se da por borreguil
    assert cuantos == sorted(cuantos, reverse=True) and cuantos[0] > cuantos[-1], cuantos
    assert r['matriz'] != antes['matriz']
    # con el TSS se puede elegir otro umbral; y volver al de partida deja todo igual
    mr.aplicar_regla(r, 'tss')
    assert r['regla']['criterio'] == 'tss' and r['regla']['umbral'] == antes['regla']['mejor_tss']
    mr.aplicar_regla(r, 'f1')
    assert {k: r[k] for k in antes} == antes
    for malo, trozo in ((('manual',), 'hay que dar el umbral'), (('manual', 1.5), 'entre 0 y 1'),
                        (('manual', 0), 'entre 0 y 1'), (('otro',), 'criterio desconocido')):
        try:
            mr.aplicar_regla(r, *malo)
        except ValueError as e:
            assert trozo in str(e), str(e)
        else:
            raise AssertionError(f'debía rechazar {malo}')
    assert {k: r[k] for k in antes} == antes       # un intento fallido no cambia nada


def test_sin_no_borreguil_no_hay_umbral_cada_caso_es_de_su_tipo():
    tab, grupos, _ = tabla(n=(20, 20, 0))
    r = mr.entrenar_y_validar(tab, grupos, RAPIDO)
    assert [c['categoria'] for c in r['categorias']] == ['húmedo', 'seco']
    assert r['regla'] == {'criterio': None, 'umbral': None, 'barrido': []}
    assert all(q['predicha'] in ('húmedo', 'seco') for q in r['predicciones'])
    assert all(abs(q['p_borreguil'] - 1) < 1e-6 for q in r['predicciones'])
    antes = (r['matriz'], r['macro_f1'])
    mr.aplicar_regla(r, 'manual', 0.9)             # no hay nada que aplicar
    assert (r['matriz'], r['macro_f1']) == antes and r['regla']['umbral'] is None


# ------------------------------------------------------- variables repetidas (VIF)
def test_el_vif_es_lo_que_las_demas_explican_de_cada_variable():
    import numpy as np
    azar = np.random.default_rng(5)
    a, b = azar.normal(size=400), azar.normal(size=400)
    c = a + 0.5 * azar.normal(size=400)
    v = mr.vif(np.column_stack([a, b, c]))
    # las demás explican el 80 % de «a» y de «c» (VIF 5) y nada de «b» (VIF 1)
    assert 4.0 < v[0] < 6.2 and 4.0 < v[2] < 6.2 and 0.99 < v[1] < 1.1
    # la resta exacta de otras dos: infinito (y las otras dos, también); la que no
    # varía no tiene VIF
    v = mr.vif(np.column_stack([a, b, a - b, np.full(400, 3.0)]))
    assert np.isinf(v[:3]).all() and np.isnan(v[3])


def _repetidas(n=300, semilla=6):
    """Variables del catálogo con repeticiones conocidas: la caída del NDVI es la resta
    exacta de los dos NDVI, el NDWI de inicio es casi el NDVI de inicio al revés, la
    pendiente casi la altitud, y un albedo que no varía."""
    import numpy as np
    azar = np.random.default_rng(semilla)
    ini, fin, alt, cur = (azar.normal(size=n) for _ in range(4))
    col = {'ndvi_early': ini, 'ndwi_early': -ini + 0.05 * azar.normal(size=n),
           'ndvi_late': fin, 'ndvi_drop': ini - fin, 'albedo_min': np.full(n, 0.1),
           'elev_dem_m': alt, 'slope_deg': alt + 0.05 * azar.normal(size=n),
           'curvature': cur}
    nombres = mr.ordenar_predictores(list(col))
    return np.column_stack([col[k] for k in nombres]), nombres


def test_el_vif_quita_de_una_en_una_la_mas_repetida_y_nunca_la_altitud():
    X, nombres = _repetidas()
    r = mr.elegir_por_vif(X, nombres)
    quitadas = [q['variable'] for q in r['quitadas']]
    # primero la que no varía; después, entre las tres que se explican del todo, la
    # que va más atrás en el catálogo: la caída del NDVI
    assert r['quitadas'][0] == {'variable': 'albedo_min', 'vif': None, 'paso': 1}
    assert r['quitadas'][1]['variable'] == 'ndvi_drop' and r['quitadas'][1]['vif'] == float('inf')
    assert [q['paso'] for q in r['quitadas']] == list(range(1, len(quitadas) + 1))
    # la altitud se queda siempre: se va la pendiente, su casi copia
    assert 'elev_dem_m' in r['quedan'] and 'slope_deg' in quitadas
    assert ('ndvi_early' in r['quedan']) != ('ndwi_early' in r['quedan'])
    assert {'ndvi_late', 'curvature'} <= set(r['quedan'])
    assert sorted(r['quedan'] + quitadas) == sorted(nombres)
    assert r['quedan'] == mr.ordenar_predictores(r['quedan'])          # en su orden
    assert all(v <= 10 for v in r['vif'].values()) and r['fijas'] == ['elev_dem_m']
    assert not r['pocas_muestras'] and r['umbral'] == 10.0
    # lo que quedaba por encima del umbral se ha ido con el VIF que tenía entonces
    assert all(q['vif'] is None or q['vif'] > 10 for q in r['quitadas'])
    # sin fijas, de la altitud y la pendiente se queda una de las dos
    r2 = mr.elegir_por_vif(X, nombres, fijas=())
    assert ('elev_dem_m' in r2['quedan']) != ('slope_deg' in r2['quedan']) and r2['fijas'] == []
    # siempre sale lo mismo
    assert mr.elegir_por_vif(X, nombres) == r


def test_el_umbral_del_vif_y_lo_que_no_se_admite():
    import numpy as np
    X, nombres = _repetidas()
    estricto = mr.elegir_por_vif(X, nombres, umbral=1.5)
    assert set(estricto['quedan']) <= set(mr.elegir_por_vif(X, nombres)['quedan'])
    assert all(v <= 1.5 for v in estricto['vif'].values() if v is not None)
    # sin repetidas no se quita nada
    azar = np.random.default_rng(9)
    libres = azar.normal(size=(200, 4))
    r = mr.elegir_por_vif(libres, ['ndvi_early', 'albedo_mean', 'elev_dem_m', 'curvature'])
    assert r['quitadas'] == [] and len(r['quedan']) == 4
    # con menos del doble de muestras que variables, se avisa
    assert mr.elegir_por_vif(libres[:7], ['ndvi_early', 'albedo_mean', 'elev_dem_m',
                                          'curvature'])['pocas_muestras']
    for malo in (dict(umbral=1.0), dict(umbral=0.5)):
        try:
            mr.elegir_por_vif(X, nombres, **malo)
            raise AssertionError(f'ha admitido {malo}')
        except ValueError:
            pass
    for X_malo, nombres_malos in ((X[:, :3], nombres), (np.where(X > 2, np.nan, X), nombres)):
        try:
            mr.elegir_por_vif(X_malo, nombres_malos)
            raise AssertionError('ha admitido una tabla que no casa o con huecos')
        except ValueError:
            pass


def test_con_las_variables_del_vif_se_entrena_y_el_modelo_usa_solo_esas():
    rows, valores = datos(n=(12, 12, 12), zonas=4)
    res = mr.muestras(rows, 'humedad', es_borreguil=es_borreguil)
    tab = mr.construir_tabla(res, valores, mr.nombres_predictores())
    sel = mr.elegir_por_vif(tab['X'], tab['predictores'], umbral=3.0)
    assert 'elev_dem_m' in sel['quedan'] and len(sel['quedan']) < len(tab['predictores'])
    corta = mr.construir_tabla(res, valores, sel['quedan'])
    assert corta['predictores'] == sel['quedan'] and corta['listo']
    assert mr.firma_tabla(corta) != mr.firma_tabla(tab)       # el modelo de antes queda viejo
    mod = mr.entrenar_y_validar(corta, mr.grupos_espaciales(corta, rows)[0], RAPIDO)
    assert mod['predictores'] == sel['quedan'] and mod['modelo'].n_features_in_ == len(sel['quedan'])
    assert {v['variable'] for v in mod['importancia']['variables']} == set(sel['quedan'])


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
