"""Pruebas del bloque «Datos» del mapeo por Random Forest.

Qué puntos sirven como muestra de entrenamiento, con qué categoría y código, y qué
variables existen como mapa. Se ejecutan con `python tests/test_mapeo_datos.py`; no
necesitan red ni Earth Engine.
"""
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import borreguil_pipeline as bp   # noqa: E402
import gee_backend as geb         # noqa: E402
import mapeo_rf as mr             # noqa: E402

bp._imports()


def es_borreguil(dec):
    d = (dec or '').lower()
    return 'no borreguil' not in d and any(
        k in d for k in ('borreguil', 'probable', 'posible', 'auto', 'verificado'))


def punto(i, decision='POSIBLE BORREGUIL', **extra):
    # Rejilla de ~1 km: cada punto cae en su propio píxel de 10 m.
    r = {'ID': f'p{i}', 'lon': -3.40 + (i % 10) * 0.01, 'lat': 37.00 + (i // 10) * 0.01,
         'decision': decision, 'ndmi_mean': 0.05, 'slope_deg': 15.0, 'twi': 5.0,
         'ndwi_mean': -0.40, 'albedo_mean': 0.15, 'ndvi_late': 0.50}
    r.update(extra)
    return r


def revisado(i, ambiente='ladera', humedad='seco', pureza='puro', **extra):
    r = punto(i, **extra)
    bp.apply_typology([r], es_borreguil)
    bp.set_tipo_revisado(r, ambiente=ambiente, humedad=humedad, pureza=pureza,
                         cuando='2026-10-02T09:00:00')
    return r


def ausencia(i, **extra):
    return punto(i, decision='NO BORREGUIL VERIFICADO', Borreguil='no', **extra)


def por_id(res):
    return {f['ID']: f for f in res['tabla']}


# ------------------------------------------------------------------ códigos
def test_los_codigos_son_estables_y_el_cero_es_no_borreguil():
    assert mr.codigos('ambiente') == {'no borreguil': 0, 'arroyo': 1, 'laguna': 2, 'ladera': 3}
    assert mr.codigos('humedad') == {'no borreguil': 0, 'húmedo': 1, 'seco': 2}
    assert mr.codigos('pureza') == {'no borreguil': 0, 'puro': 1, 'mixto-agua': 2,
                                    'mixto-roca': 3}
    # No dependen de los datos: «ladera» es 3 aunque no haya ninguna «laguna».
    res = mr.muestras([revisado(0, ambiente='ladera')], 'ambiente', es_borreguil=es_borreguil)
    assert por_id(res)['p0']['codigo'] == 3


# ------------------------------------------------- qué entra como muestra
def test_solo_cuentan_los_puntos_con_el_tipo_revisado():
    rows = [revisado(0, humedad='húmedo'), revisado(1, humedad='seco'),
            punto(2, 'BORREGUIL PROBABLE'),                  # borreguil sin revisar
            punto(3, 'BORREGUIL VERIFICADO', Borreguil='si'),  # verificado, tipo sin revisar
            punto(4, 'NO BORREGUIL'), punto(5, 'INCIERTO')]  # lo dice el modelo, nadie más
    bp.apply_typology(rows, es_borreguil)                    # la app propone tipo a 2 y 3
    res = mr.muestras(rows, 'humedad', es_borreguil=es_borreguil)
    t = por_id(res)
    assert (t['p0']['categoria'], t['p0']['origen_etiqueta']) == ('húmedo', 'tipo revisado')
    assert t['p0']['fecha_revision'] == '2026-10-02T09:00:00' and t['p0']['usada']
    assert t['p2']['motivo'] == 'tipo_sin_revisar' and not t['p2']['usada']
    assert t['p2']['categoria'] == ''                        # la propuesta NO es etiqueta
    assert t['p2']['propuesta_app'] in mr.NIVELES['humedad']  # pero se deja constancia
    assert t['p3']['motivo'] == 'tipo_sin_revisar'
    assert t['p4']['motivo'] == 'sin_etiqueta' and t['p5']['motivo'] == 'sin_etiqueta'
    assert res['n_inicial'] == 6 and res['n_usadas'] == 2
    assert res['excluidos'] == {'tipo_sin_revisar': 2, 'sin_etiqueta': 2}


def test_la_categoria_es_la_que_dejo_la_persona_no_la_de_la_app():
    r = revisado(0, humedad='húmedo')                        # la app proponía «seco»
    assert r['humedad_regla'] == 'seco'
    f = por_id(mr.muestras([r], 'humedad', es_borreguil=es_borreguil))['p0']
    assert f['categoria'] == 'húmedo' and f['propuesta_app'] == 'seco'


def test_las_ausencias_son_solo_las_de_campo_y_no_se_inventan():
    rows = [revisado(0), revisado(1, ambiente='arroyo'),
            ausencia(2), ausencia(3),
            punto(4, 'NO BORREGUIL')]                        # «no» del modelo: no es ausencia
    res = mr.muestras(rows, 'ambiente', es_borreguil=es_borreguil)
    t = por_id(res)
    assert (t['p2']['categoria'], t['p2']['codigo'], t['p2']['origen_etiqueta']) == \
        ('no borreguil', 0, 'ausencia de campo')
    assert t['p4']['motivo'] == 'sin_etiqueta' and not t['p4']['usada']
    assert {c['categoria']: c['n'] for c in res['clases']} == \
        {'no borreguil': 2, 'arroyo': 1, 'laguna': 0, 'ladera': 1}
    # Sin la clase de ausencia, esos puntos quedan fuera y se dice por qué.
    res = mr.muestras(rows, 'ambiente', incluir_ausencias=False, es_borreguil=es_borreguil)
    assert por_id(res)['p2']['motivo'] == 'ausencia_no_incluida'
    assert [c['categoria'] for c in res['clases']] == ['arroyo', 'laguna', 'ladera']
    assert ('sin_ausencia',) in res['avisos']


def test_dudosos_e_incoherentes_no_entran():
    dudoso = revisado(0, Duda='si')
    incoherente = revisado(1)
    incoherente['Borreguil'] = 'no'                          # campo dice «no», pero tiene tipo
    ausencia_dudosa = ausencia(2, Duda='si')
    t = por_id(mr.muestras([dudoso, incoherente, ausencia_dudosa], 'pureza',
                           es_borreguil=es_borreguil))
    assert t['p0']['motivo'] == 'dudoso'
    assert t['p1']['motivo'] == 'incoherente'
    assert t['p2']['motivo'] == 'dudoso'
    assert not any(f['usada'] for f in t.values())


def test_dos_muestras_en_el_mismo_pixel():
    a = revisado(0)
    c = revisado(2, ambiente='arroyo')
    # Se colocan en el CENTRO de su píxel de 10 m, para que un vecino a ~3 m caiga
    # con seguridad en el mismo píxel (y no al otro lado de un borde).
    for r, (lon, lat) in zip((a, c), bp.snap_to_s2_grid([(a['lon'], a['lat']),
                                                         (c['lon'], c['lat'])])):
        r['lon'], r['lat'] = lon, lat
    b = revisado(1, lon=a['lon'] + 0.00003, lat=a['lat'])
    d = revisado(3, ambiente='ladera', lon=c['lon'] + 0.00003, lat=c['lat'])
    e = revisado(4)                                          # lejos de todos
    t = por_id(mr.muestras([a, b, c, d, e], 'ambiente', es_borreguil=es_borreguil))
    # misma categoría: se queda la primera
    assert t['p0']['usada'] and t['p1']['motivo'] == 'duplicado'
    # distinta categoría: no se puede saber cuál es la buena, fuera las dos
    assert t['p2']['motivo'] == 'conflicto' and t['p3']['motivo'] == 'conflicto'
    assert t['p4']['usada']
    # En otro nivel las dos del conflicto coinciden (ambas «seco»): ahí es un duplicado.
    t = por_id(mr.muestras([c, d], 'humedad', es_borreguil=es_borreguil))
    assert t['p2']['usada'] and t['p3']['motivo'] == 'duplicado'


def test_punto_sin_coordenadas_no_rompe_ni_desplaza_a_los_demas():
    rows = [revisado(0), revisado(1, humedad='húmedo'),
            dict(revisado(2), lon=None), dict(revisado(3), lat=float('nan'))]
    res = mr.muestras(rows, 'humedad', es_borreguil=es_borreguil)
    t = por_id(res)
    assert t['p2']['motivo'] == 'sin_coordenadas' and t['p3']['motivo'] == 'sin_coordenadas'
    assert t['p0']['usada'] and t['p1']['usada']


def test_no_modifica_ningun_punto():
    rows = [revisado(0), punto(1), ausencia(2), revisado(3, Duda='si')]
    antes = copy.deepcopy(rows)
    for nivel in mr.NIVELES:
        mr.muestras(rows, nivel, es_borreguil=es_borreguil)
        mr.muestras(rows, nivel, incluir_ausencias=False, es_borreguil=es_borreguil)
    assert rows == antes


# --------------------------------------------------------- errores y avisos
def test_sin_muestras_o_con_una_sola_categoria_no_se_puede_seguir():
    res = mr.muestras([punto(0), punto(1)], 'humedad', es_borreguil=es_borreguil)
    assert not res['listo'] and res['errores'] == [('pocas_categorias', 0)]
    assert 'pestaña Tabla' in mr.mensaje(res['errores'][0])
    assert res['avisos'] == []               # sin muestras, sobran los demás avisos
    rows = [revisado(i, humedad='seco') for i in range(6)]
    res = mr.muestras(rows, 'humedad', es_borreguil=es_borreguil)
    assert not res['listo'] and ('pocas_categorias', 1) in res['errores']


def test_minimo_y_recomendado_por_categoria():
    rows = ([revisado(i, humedad='seco') for i in range(6)]
            + [revisado(10 + i, humedad='húmedo') for i in range(3)])
    res = mr.muestras(rows, 'humedad', es_borreguil=es_borreguil)
    assert ('pocas_muestras', 'húmedo', 3) in res['errores'] and not res['listo']
    assert 'mínimo es 5' in mr.mensaje(('pocas_muestras', 'húmedo', 3))
    rows += [revisado(20 + i, humedad='húmedo') for i in range(2)]
    res = mr.muestras(rows, 'humedad', es_borreguil=es_borreguil)
    assert res['listo'] and res['errores'] == []
    assert ('muestras_justas', 'seco', 6) in res['avisos']   # se puede, pero es poco
    assert ('sin_ausencia',) in res['avisos']                # y se avisa de lo que implica
    assert 'no demuestra presencia' in mr.mensaje(('sin_ausencia',))


def test_categoria_sin_muestras_se_avisa():
    rows = ([revisado(i, ambiente='ladera') for i in range(5)]
            + [revisado(10 + i, ambiente='arroyo') for i in range(5)])
    res = mr.muestras(rows, 'ambiente', es_borreguil=es_borreguil)
    assert res['listo'] and ('categoria_sin_muestras', 'laguna') in res['avisos']


def test_mensajes_en_ingles_con_la_categoria_traducida():
    en = {'húmedo': 'wet', 'laguna': 'lake'}
    tr = lambda v: en.get(v, v)                                           # noqa: E731
    assert mr.mensaje(('pocas_muestras', 'húmedo', 3), en=True, traduce=tr) == \
        '"wet" has 3 samples; the minimum is 5.'
    assert mr.mensaje(('pocas_muestras', 'húmedo', 1)) == \
        '«húmedo» tiene 1 muestra; el mínimo es 5.'
    assert '"lake"' in mr.mensaje(('categoria_sin_muestras', 'laguna'), en=True, traduce=tr)
    assert 'at least two' in mr.mensaje(('pocas_categorias', 1), en=True)


# ------------------------------------------------------------- exportación
def test_tabla_csv_una_fila_por_punto_con_su_motivo():
    rows = [revisado(0), punto(1), ausencia(2)]
    csv = mr.tabla_csv(mr.muestras(rows, 'pureza', es_borreguil=es_borreguil))
    lineas = csv.strip().split('\n')
    assert lineas[0] == ('ID,lon,lat,categoria,codigo,origen_etiqueta,fecha_revision,'
                         'propuesta_app,fuente,usada,motivo')
    assert len(lineas) == 4
    assert lineas[1].startswith('p0,') and ',puro,1,tipo revisado,' in lineas[1]
    assert lineas[1].endswith(',si,')
    assert lineas[2].endswith(',no,tipo_sin_revisar')
    assert ',no borreguil,0,ausencia de campo,' in lineas[3]


# ------------------------------------------------------------- predictores
def test_los_predictores_son_las_bandas_que_calcula_earth_engine():
    nombres = mr.nombres_predictores()
    assert len(nombres) == 34 and len(set(nombres)) == 34
    # Las que calcula Earth Engine para el detector, menos GNDVI: es −NDWI.
    indices = [k for k in geb.INDS if k != 'gndvi']
    assert len(indices) == len(geb.INDS) - 1
    esperados = ([f'{k}_early' for k in indices] + [f'{k}_late' for k in indices]
                 + ['ndvi_drop']
                 + [f'{k}_{s}' for k in geb.NEW_INDS for s in ('mean', 'min', 'max', 'sd')]
                 + ['elev_dem_m', 'slope_deg', 'aspect_north', 'aspect_east', 'curvature'])
    assert nombres == esperados
    por_grupo = {g: len(mr.nombres_predictores([g])) for g in mr.GRUPOS}
    assert list(por_grupo.values()) == [6, 6, 1, 16, 5]
    # Nada que no sea una variable del terreno: ni ID, ni etiquetas, ni la decisión.
    prohibidos = {'ID', 'decision', 'rf_proba', 'Borreguil', 'ambiente', 'humedad', 'pureza',
                  'tipo_borreguil', 'tipo_revisado', 'lon', 'lat', 'cuenca_id'}
    assert not prohibidos & set(nombres)
    # Las que no existen como mapa no están: ni texturas, ni radar, ni TWI.
    assert not {'twi', 'glcm_homog', 's1_vv_mean', 'surr_rock'} & set(nombres)
    # GNDVI tampoco: sería repetir NDWI. Y se explica en la lista de ausentes.
    assert not [n for n in nombres if 'gndvi' in n]
    assert any('GNDVI' in q and 'signo cambiado' in p_ for q, p_, _q, _p in mr.NO_DISPONIBLES)


def test_cada_predictor_tiene_descripcion_y_grupo_en_los_dos_idiomas():
    for nombre, grupo, es, en in mr.PREDICTORES:
        assert es and en and mr.descripcion(nombre) == es
        assert mr.descripcion(nombre, en=True) == en
        assert grupo in mr.GRUPOS and mr.nombre_grupo(grupo, en=True) != grupo
    assert mr.descripcion('ndmi_sd') == 'NDMI, humedad: variabilidad'
    assert mr.descripcion('no_existe') == ''
    assert set(mr.MOTIVOS) == set(mr.MOTIVOS_EN)
    assert mr.motivo('duplicado', en=True).startswith('same 10 m pixel')


def test_la_seleccion_de_predictores_conserva_el_orden_del_catalogo():
    assert mr.ordenar_predictores(['slope_deg', 'ndvi_late', 'ndvi_early', 'ndvi_late']) == \
        ['ndvi_early', 'ndvi_late', 'slope_deg']
    try:
        mr.ordenar_predictores(['ndvi_late', 'twi'])
    except ValueError as e:
        assert 'twi' in str(e)
    else:
        raise AssertionError('debía rechazar una variable que no existe como mapa')


# ----------------------------------------------------- tabla de entrenamiento
def dos_clases(n_seco=6, n_humedo=6):
    return ([revisado(i, humedad='seco') for i in range(n_seco)]
            + [revisado(20 + i, humedad='húmedo') for i in range(n_humedo)])


def lectura(rows, f=None, sin=None):
    """Imita lo que devuelve Earth Engine: un valor por banda y punto. `f(i, j)` da
    el valor de la banda j en el punto i; `sin` quita bandas a algunos puntos."""
    f = f or (lambda i, j: 0.01 * (i + 1) + 0.001 * j * (1 + i % 3))
    valores = {}
    for i, r in enumerate(rows):
        v = {b: f(i, j) for j, b in enumerate(geb.BANDAS_PILA)}
        for b in (sin or {}).get(r['ID'], ()):
            v[b] = None
        valores[mr.clave_punto(r['lon'], r['lat'])] = v
    return valores


def test_la_tabla_tiene_una_fila_por_muestra_y_las_variables_en_orden():
    rows = dos_clases() + [punto(40, 'BORREGUIL PROBABLE'), ausencia(41)]
    res = mr.muestras(rows, 'humedad', es_borreguil=es_borreguil)
    tab = mr.construir_tabla(res, lectura(rows), ['slope_deg', 'ndvi_late', 'wavi_sd'])
    assert tab['predictores'] == ['ndvi_late', 'wavi_sd', 'slope_deg']   # orden del catálogo
    assert len(tab['filas']) == 13 and len(tab['X']) == 13 and len(tab['y']) == 13
    assert all(len(x) == 3 for x in tab['X'])
    assert sorted(set(tab['y'])) == [0, 1, 2]                 # no borreguil, húmedo, seco
    f0 = tab['filas'][0]
    assert (f0['ID'], f0['categoria'], f0['codigo'], f0['indice']) == ('p0', 'seco', 2, 0)
    assert tab['X'][0] == [f0['ndvi_late'], f0['wavi_sd'], f0['slope_deg']]
    assert tab['apartadas'] == []
    # Con una sola ausencia de campo esa clase no llega al mínimo, y se dice.
    assert tab['errores'] == [('pocas_muestras', 'no borreguil', 1)] and not tab['listo']
    # El punto sin revisar (p40) ni se lee ni aparece.
    assert 'p40' not in {f['ID'] for f in tab['filas']}


def test_las_muestras_sin_dato_se_apartan_y_se_dice_por_que():
    rows = dos_clases(7, 7)
    res = mr.muestras(rows, 'humedad', es_borreguil=es_borreguil)
    valores = lectura(rows, sin={'p1': ['ndmi_mean'], 'p2': list(geb.BANDAS_PILA)})
    valores[mr.clave_punto(rows[3]['lon'], rows[3]['lat'])]['slope_deg'] = float('inf')
    del valores[mr.clave_punto(rows[4]['lon'], rows[4]['lat'])]
    tab = mr.construir_tabla(res, valores, mr.nombres_predictores())
    ap = {a['ID']: (a['motivo'], a['faltan']) for a in tab['apartadas']}
    assert ap['p1'] == ('sin_dato', ['ndmi_mean'])
    assert ap['p2'][0] == 'sin_cobertura' and len(ap['p2'][1]) == 34
    assert ap['p3'] == ('sin_dato', ['slope_deg'])            # un infinito no es un dato
    assert ap['p4'] == ('sin_leer', [])
    assert len(tab['filas']) == 10 and not set(ap) & {f['ID'] for f in tab['filas']}
    # Si la variable que falta no está entre las elegidas, la muestra sí vale.
    tab = mr.construir_tabla(res, valores, ['ndvi_late', 'elev_dem_m'])
    assert 'p1' in {f['ID'] for f in tab['filas']}


def test_si_al_apartar_una_categoria_se_queda_corta_no_se_puede_seguir():
    rows = dos_clases(6, 5)
    res = mr.muestras(rows, 'humedad', es_borreguil=es_borreguil)
    assert res['listo']
    tab = mr.construir_tabla(res, lectura(rows, sin={'p20': ['evi_late']}),
                             mr.nombres_predictores())
    assert not tab['listo'] and ('pocas_muestras', 'húmedo', 4) in tab['errores']
    c = {x['categoria']: (x['n_antes'], x['n']) for x in tab['clases']}
    assert c['húmedo'] == (5, 4) and c['seco'] == (6, 6)


def test_sin_predictores_no_hay_tabla():
    rows = dos_clases()
    res = mr.muestras(rows, 'humedad', es_borreguil=es_borreguil)
    try:
        mr.construir_tabla(res, lectura(rows), [])
    except ValueError as e:
        assert 'predictor' in str(e)
    else:
        raise AssertionError('debía negarse a construir una tabla sin variables')


def test_tabla_csv_de_entrenamiento():
    rows = dos_clases()
    res = mr.muestras(rows, 'humedad', es_borreguil=es_borreguil)
    tab = mr.construir_tabla(res, lectura(rows), ['ndvi_late', 'slope_deg'])
    lineas = mr.tabla_entrenamiento_csv(tab).strip().split('\n')
    assert lineas[0] == 'ID,categoria,codigo,ndvi_late,slope_deg'
    assert len(lineas) == 13 and lineas[1].startswith('p0,seco,2,')


def test_control_de_calidad_avisa_pero_no_quita_nada():
    rows = dos_clases(15, 15)
    res = mr.muestras(rows, 'humedad', es_borreguil=es_borreguil)
    pos = {b: j for j, b in enumerate(geb.BANDAS_PILA)}

    def f(i, j):
        if j == pos['nbr_late']:
            return 0.5                                   # constante
        if j == pos['evi_late']:
            return 9.0 if i == 0 else 1.0                # casi constante (29 de 30)
        if j in (pos['ndvi_early'], pos['ndvi_late']):
            return ((i * 31) % 17) / 17.0                # dos columnas idénticas
        if j == pos['clre_late']:
            return 2.0 * i + 0.3                         # correlación perfecta con ndwi_late
        if j == pos['ndwi_late']:
            return -1.0 * i
        if j == pos['ndmi_late']:
            return 0.1 + ((i * 53) % 19) / 19.0
        if j == pos['ndmi_early']:
            return -(0.1 + ((i * 53) % 19) / 19.0)       # ndmi_late con el signo cambiado
        return ((i * 7919 + j * 104729) % 97) / 97.0     # ruido sin relación
    usar = ['ndvi_early', 'ndvi_late', 'ndwi_late', 'clre_late', 'evi_late', 'nbr_late',
            'ndmi_early', 'ndmi_late']
    tab = mr.construir_tabla(res, lectura(rows, f), usar)
    h = mr.calidad(tab)
    claves = [x[0] for x in h]
    assert ('constante', 'nbr_late') in h
    assert any(x[:2] == ('casi_constante', 'evi_late') and x[2] > 0.95 for x in h)
    assert ('duplicadas', 'ndvi_early', 'ndvi_late') in h
    # Como NDWI y GNDVI en los datos reales: no se repite además como «correlación».
    assert ('opuestas', 'ndmi_early', 'ndmi_late') in h
    assert not [x for x in h if x[0] == 'correlacion' and {x[1], x[2]} == {'ndmi_early',
                                                                         'ndmi_late'}]
    assert 'signo cambiado' in mr.mensaje_calidad(('opuestas', 'a', 'b'))
    corr = [x for x in h if x[0] == 'correlacion']
    assert any({x[1], x[2]} == {'ndwi_late', 'clre_late'} and x[3] < -0.99 for x in corr)
    assert 'mas_variables_que_muestras' not in claves
    assert tab['predictores'] == mr.ordenar_predictores(usar)   # no se ha quitado ninguna
    assert 'no puede ayudar' in mr.mensaje_calidad(('constante', 'nbr_late'))
    assert 'r = -1.00' in mr.mensaje_calidad(('correlacion', 'a', 'b', -1.0))
    assert 'identical' in mr.mensaje_calidad(('duplicadas', 'a', 'b'), en=True)


def test_control_de_calidad_muestras_frente_a_variables():
    rows = dos_clases(6, 6)
    res = mr.muestras(rows, 'humedad', es_borreguil=es_borreguil)
    ruido = lambda i, j: ((i * 7919 + j * 104729) % 97) / 97.0             # noqa: E731
    tab = mr.construir_tabla(res, lectura(rows, ruido), mr.nombres_predictores())
    assert ('mas_variables_que_muestras', 12, 34) in mr.calidad(tab)      # 12 < 34
    tab = mr.construir_tabla(res, lectura(rows, ruido), mr.nombres_predictores()[:4])
    assert ('pocas_muestras_por_variable', 12, 4) in mr.calidad(tab)      # 12 < 5 × 4
    tab = mr.construir_tabla(res, lectura(rows, ruido), mr.nombres_predictores()[:2])
    assert not [x for x in mr.calidad(tab) if 'muestras' in x[0]]         # 12 ≥ 5 × 2


def test_desequilibrio_entre_categorias():
    clases = [{'categoria': 'seco', 'n': 60}, {'categoria': 'húmedo', 'n': 6},
              {'categoria': 'no borreguil', 'n': 0}]
    razon, mayor, menor = mr.desequilibrio(clases)
    assert razon == 10 and mayor['categoria'] == 'seco' and menor['categoria'] == 'húmedo'
    assert mr.desequilibrio([{'categoria': 'seco', 'n': 9}]) is None


def test_las_bandas_de_earth_engine_son_las_del_catalogo():
    # El orden importa: la tabla, el modelo y el mapa tienen que ver lo mismo.
    assert list(geb.BANDAS_PILA) == mr.nombres_predictores()
    info = geb.info_pila((2023, 2024), 32630)
    assert info['crs'] == 'EPSG:32630' and info['resolucion_m'] == 10
    assert info['bandas'] == mr.nombres_predictores() and info['anios'] == [2023, 2024]
    assert geb.rejilla(32630) == {'crs': 'EPSG:32630', 'crsTransform': [10, 0, 0, 0, -10, 0]}
    assert geb.utm_de([(-3.3, 37.05), (-3.4, 37.0)]) == 32630


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
