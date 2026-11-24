"""Pruebas del tipo de borreguil revisado por una persona.

Se ejecutan con `python tests/test_tipo_revisado.py`. No necesitan Earth Engine ni
conexión: usan puntos sintéticos con las pocas variables que consultan las reglas.
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import borreguil_pipeline as bp   # noqa: E402

bp._imports()


def es_borreguil(dec):
    d = (dec or '').lower()
    return 'no borreguil' not in d and any(
        k in d for k in ('borreguil', 'probable', 'posible', 'auto', 'verificado'))


def punto(i, decision, ndmi=0.05, slope=15.0, twi=5.0, ndwi=-0.40,
          albedo=0.15, ndvi=0.50):
    # ndvi_early_pixel y agua_pixel_vecinos: con lo que se propone la pureza (como en
    # el mapa); 0,50 sin agua al lado es «puro»
    return {'ID': f'p{i}', 'lon': -3.30 + i * 0.001, 'lat': 37.05, 'decision': decision,
            'ndmi_mean': ndmi, 'slope_deg': slope, 'twi': twi, 'ndwi_mean': ndwi,
            'albedo_mean': albedo, 'ndvi_late': ndvi, 'ndvi_early_pixel': ndvi,
            'agua_pixel_vecinos': 0}


def conjunto():
    return [
        punto(0, 'POSIBLE BORREGUIL', ndmi=0.20),              # húmedo, ladera
        punto(1, 'BORREGUIL PROBABLE', ndmi=0.02),             # seco, ladera
        punto(2, 'BORREGUIL VERIFICADO', ndmi=0.15, slope=3),  # húmedo, arroyo
        punto(3, 'NO BORREGUIL'),
        punto(4, 'INCIERTO'),
    ]


def test_propuesta_inicial():
    rows = conjunto()
    assert bp.apply_typology(rows, es_borreguil) == []
    assert rows[0]['humedad'] == 'húmedo' and rows[0]['humedad_regla'] == 'húmedo'
    assert rows[1]['humedad'] == 'seco'
    assert rows[2]['ambiente'] == 'arroyo'
    assert rows[0]['tipo_borreguil'].count(' · ') == 2
    for r in rows[3:]:                         # los no borreguiles no tienen tipo
        assert r['ambiente'] == '' and r['tipo_borreguil'] == ''
    assert not any(bp.tipo_revisado(r) for r in rows)


def test_revisar_cambia_valor_y_conserva_la_propuesta():
    rows = conjunto()
    bp.apply_typology(rows, es_borreguil)
    r = rows[0]
    bp.set_tipo_revisado(r, humedad='seco', cuando='2026-10-01T10:00:00')
    assert r['humedad'] == 'seco'              # lo que dejó la persona
    assert r['humedad_regla'] == 'húmedo'      # lo que propuso la app
    assert r['ambiente'] == r['ambiente_regla']  # nivel no tocado: se confirma
    assert bp.tipo_revisado(r) and r['tipo_revisado_fecha'] == '2026-10-01T10:00:00'
    assert r['tipo_borreguil'].split(' · ')[1] == 'seco'


def test_categoria_no_admitida_se_rechaza():
    rows = conjunto()
    bp.apply_typology(rows, es_borreguil)
    try:
        bp.set_tipo_revisado(rows[0], ambiente='río')
    except ValueError as e:
        assert 'río' in str(e)
    else:
        raise AssertionError('debía rechazar una categoría inventada')
    assert not bp.tipo_revisado(rows[0])       # y no deja el punto a medias


def test_nivel_sin_dato_obliga_a_elegir():
    r = {'ID': 'x', 'lon': -3.3, 'lat': 37.0, 'decision': 'POSIBLE BORREGUIL'}
    bp.apply_typology([r], es_borreguil)       # sin variables: todo '—'
    assert r['ambiente'] == '—'
    try:
        bp.set_tipo_revisado(r, humedad='seco')
    except ValueError as e:
        assert 'ambiente' in str(e) and 'pureza' in str(e)
    else:
        raise AssertionError('no puede darse por revisado con niveles vacíos')
    bp.set_tipo_revisado(r, ambiente='ladera', humedad='seco', pureza='puro')
    assert bp.tipo_revisado(r)


def test_recalcular_no_pisa_lo_revisado():
    rows = conjunto()
    bp.apply_typology(rows, es_borreguil)
    bp.set_tipo_revisado(rows[0], ambiente='laguna', humedad='seco', pureza='mixto-roca')
    # Se reentrena: cambian decisiones y se vuelve a calcular la tipología.
    rows[4]['decision'] = 'POSIBLE BORREGUIL'  # un incierto pasa a borreguil
    rows[0]['decision'] = 'INCIERTO'           # el modelo ya no ve borreguil el revisado
    bp.apply_typology(rows, es_borreguil)
    assert (rows[0]['ambiente'], rows[0]['humedad'], rows[0]['pureza']) == \
        ('laguna', 'seco', 'mixto-roca')
    assert rows[0]['ambiente_regla'] == 'ladera' and bp.tipo_revisado(rows[0])
    assert rows[4]['ambiente'] in bp.NIVELES_TIPO['ambiente']   # antes se quedaba vacío
    assert not bp.tipo_revisado(rows[4])


def test_deshacer_vuelve_a_la_propuesta():
    rows = conjunto()
    bp.apply_typology(rows, es_borreguil)
    antes = rows[1]['tipo_borreguil']
    bp.set_tipo_revisado(rows[1], ambiente='laguna', humedad='húmedo')
    bp.clear_tipo_revisado(rows[1])
    assert rows[1]['tipo_borreguil'] == antes and not bp.tipo_revisado(rows[1])


def test_resumen_cuenta_revisados_y_corregidos():
    rows = conjunto()
    bp.apply_typology(rows, es_borreguil)
    bp.set_tipo_revisado(rows[0])                       # confirmado sin cambios
    bp.set_tipo_revisado(rows[1], humedad='húmedo')     # corregido en un nivel
    s = bp.resumen_revision(rows, es_borreguil)
    assert s['n_borreguil'] == 3 and s['n_revisados'] == 2 and s['n_corregidos'] == 1
    assert s['por_nivel'] == {'ambiente': 0, 'humedad': 1, 'pureza': 0}


def test_fichero_con_categoria_invalida_se_avisa_y_no_cuenta():
    rows = conjunto()
    rows[0].update({'tipo_revisado': 'si', 'ambiente': 'rio', 'humedad': 'seco',
                    'pureza': 'puro'})
    invalidos = bp.apply_typology(rows, es_borreguil)
    # el nivel no válido deja de contar; los otros dos, revisados bien, siguen
    assert invalidos == ['p0'] and bp.niveles_revisados(rows[0]) == ['humedad', 'pureza']
    assert 'rio' in rows[0]['tipo_revision_invalida']
    assert rows[0]['ambiente'] == rows[0]['ambiente_regla']   # vuelve a la propuesta
    assert rows[0]['tipo_revisado'] == ''                       # ya no son los tres


def test_guardar_y_volver_a_cargar_conserva_lo_revisado():
    rows = conjunto()
    rows[2]['Borreguil'] = 'si'
    bp.apply_typology(rows, es_borreguil)
    bp.set_tipo_revisado(rows[0], ambiente='laguna', humedad='seco', pureza='mixto-agua',
                         cuando='2026-10-01T10:00:00')
    with tempfile.TemporaryDirectory() as d:
        ruta = os.path.join(d, 'puntos.geojson')
        bp.save_points_geojson(rows, ruta)
        otra = bp.read_points(ruta)
    bp.leer_etiquetas(otra)                        # lo que hace la app al cargar
    assert len(otra) == len(rows)
    por_id = {r['ID']: r for r in otra}
    assert 'ndmi_mean' not in por_id['p0']        # las variables no viajan
    for r in otra:                                 # lo que hace la app al ejecutarse
        r.update(punto(int(r['ID'][1:]), ''))      # (re-extrae las variables…)
        r['decision'] = bp.decide(r)               # …y decide: no debe fallar con huecos
        r['decision'] = por_id[r['ID']].get('decision') or r['decision']
    por_id['p0'].update(decision='POSIBLE BORREGUIL', ndmi_mean=0.20)
    bp.apply_typology(otra, es_borreguil)
    p0 = por_id['p0']
    assert (p0['ambiente'], p0['humedad'], p0['pureza']) == ('laguna', 'seco', 'mixto-agua')
    # vuelve la fecha (la plantilla guarda el día: `date`)
    assert bp.tipo_revisado(p0) and p0['tipo_revisado_fecha'] == '2026-10-01'
    assert p0['humedad_regla'] == 'húmedo'         # la propuesta se recalcula
    assert por_id['p2']['Borreguil'] == 'si'       # la verdad-terreno también vuelve
    assert not bp.tipo_revisado(por_id['p1'])


def test_recargar_con_todos_los_puntos_revisados_conserva_la_fecha():
    # Si todos los puntos tienen fecha, al leer el fichero la columna llega como
    # fecha y no como texto: debe volver al mismo texto que se guardó.
    rows = [punto(i, 'POSIBLE BORREGUIL', ndmi=0.20) for i in range(6)]
    bp.apply_typology(rows, es_borreguil)
    for r in rows:
        bp.set_tipo_revisado(r, cuando='2026-10-01T10:00:00')
    with tempfile.TemporaryDirectory() as d:
        ruta = os.path.join(d, 'puntos.geojson')
        bp.save_points_geojson(rows, ruta)
        otra = bp.read_points(ruta)
    bp.leer_etiquetas(otra)
    for r in otra:
        r.update(punto(int(r['ID'][1:]), ''))
        r['decision'] = 'POSIBLE BORREGUIL'
    assert bp.apply_typology(otra, es_borreguil) == []
    assert all(bp.niveles_revisados(r) == list(bp.NIVELES_TIPO) for r in otra)
    assert {bp._fecha_txt(r['tipo_revisado_fecha'])[:10] for r in otra} == {'2026-10-01'}


def test_geojson_no_arrastra_variables_ni_marcas_de_sesion():
    import json
    rows = conjunto()
    # En un KML todos los atributos llegan como texto, también los que tienen
    # nombre de variable: tampoco deben viajar.
    rows[0].update(ndmi_mean='0.20', twi='5.0', nota='visto en campo', cuenca_id=7)
    rows[1].update(source='truth', cuenca_id=3.0,
                   ndvi_early_pixel='0.47', agua_pixel_vecinos='1')   # los de la pureza
    rows[2].update(source='random_uniform', cuenca_id='Lanjarón')
    # p2: verificado en la app
    rows[2].update(Borreguil='si', origin='verificado')
    rows[3].update(Borreguil='no', decision='NO BORREGUIL VERIFICADO')
    bp.apply_typology(rows, es_borreguil)
    with tempfile.TemporaryDirectory() as d:
        ruta = os.path.join(d, 'puntos.geojson')
        bp.save_points_geojson(rows, ruta)
        with open(ruta, encoding='utf-8') as f:
            props = [x['properties'] for x in json.load(f)['features']]
    for p in props:
        assert not set(p) & set(bp.FEATURES_RF_COMBO), sorted(set(p) & set(bp.FEATURES_RF_COMBO))
        assert not set(p) & set(bp.CLAVES_PUREZA_PUNTO), sorted(set(p) & set(bp.CLAVES_PUREZA_PUNTO))
    assert props[0]['nota'] == 'visto en campo'           # los atributos propios sí
    assert [p.get('cuenca_id') for p in props[:3]] == [7, 3, 'Lanjarón']
    assert 'source' not in props[1]                       # marca de esta sesión
    assert props[2]['source'] == 'random_uniform'
    # La verdad-terreno vuelve como tal, con las columnas de la plantilla: presence 1/0.
    assert [p['presence'] for p in props[:4]] == [None, None, 1, 0]
    assert all('origin' not in p for p in props)
    assert props[2]['method'] == 'orthophoto'             # verificado en la app
    # las etiquetas internas no salen: van en las columnas de la plantilla
    internas = set(bp.CAMPOS_REVISION) | {'Borreguil', 'Duda'}
    assert not any(set(p) & internas for p in props)
    # la propuesta de la app no sale como si fuera un tipo revisado
    assert all(p['environment'] is None and p['moisture'] is None for p in props)


def test_decidir_no_falla_con_etiquetas_vacias_de_un_fichero():
    # Un GeoJSON con "Borreguil": null en los puntos sin etiquetar llega como NaN.
    nan = float('nan')
    r = dict(punto(0, ''), Borreguil=nan, Duda=nan, rf_proba=0.9)
    assert bp.decide(r, 0.5) == 'BORREGUIL PROBABLE'
    assert bp.decide(dict(r, Borreguil=None, Duda=None, rf_proba=0.1), 0.5) == 'NO BORREGUIL'
    assert bp.decide(dict(r, Borreguil=' No '), 0.5) == 'NO BORREGUIL VERIFICADO'
    assert bp.decide(dict(r, Borreguil='si', Duda='SI'), 0.5) == 'DUDOSO (campo)'


def test_la_imagen_va_con_el_punto_y_no_con_su_posicion():
    # Quien revisa decide mirando la imagen: tiene que ser la del punto, también
    # después de quitar otros de la lista.
    import json
    from PIL import Image
    rows = conjunto()
    with tempfile.TemporaryDirectory() as d:
        for i in (0, 2):                              # el punto 1 se queda sin imagen
            Image.new('RGB', (500, 400), (60 + 40 * i, 120, 70)).save(
                os.path.join(d, f'pt_{i:04d}.jpg'))
        bp.compute_img_features(rows, d)
        assert [r['_img'] for r in rows] == ['pt_0000.jpg', '', 'pt_0002.jpg', '', '']
        quedan = rows[1:]                             # se quita el primero
        assert quedan[1]['ID'] == 'p2' and quedan[1]['_img'] == 'pt_0002.jpg'
        ruta = os.path.join(d, 'puntos.geojson')
        bp.save_points_geojson(rows, ruta)            # la anotación no sale de la sesión
        with open(ruta, encoding='utf-8') as f:
            assert all('_img' not in x['properties'] for x in json.load(f)['features'])


def test_verdad_terreno_traslada_el_tipo_revisado_al_candidato():
    cand = [punto(0, ''), punto(1, '')]
    campo = [dict(punto(0, ''), ID='campo_a', Borreguil='si', tipo_revisado='si',
                  ambiente='arroyo', humedad='húmedo', pureza='puro',
                  tipo_revisado_fecha='2026-09-30T09:00:00')]
    n_match, n_add, n_pos, n_neg = bp.merge_truth_points(cand, campo)
    assert (n_match, n_add) == (1, 0)
    assert cand[0]['Borreguil'] == 'si' and bp.tipo_revisado(cand[0])
    cand[0]['decision'] = 'BORREGUIL VERIFICADO'
    bp.apply_typology(cand, es_borreguil)
    assert (cand[0]['ambiente'], cand[0]['humedad'], cand[0]['pureza']) == \
        ('arroyo', 'húmedo', 'puro')


def test_verdad_terreno_con_la_plantilla_traslada_cada_nivel():
    cand = [punto(0, ''), punto(1, '')]
    campo = [dict(punto(0, ''), ID='socio_1', presence='1', moisture='dry'),
             dict(punto(5, ''), ID='socio_2', presence='0', moisture='wet')]   # ausencia
    bp.merge_truth_points(cand, campo)
    assert bp.niveles_revisados(cand[0]) == ['humedad'] and cand[0]['humedad'] == 'seco'
    nuevo = [r for r in cand if r.get('ID') == 'socio_2'][0]
    assert nuevo['Borreguil'] == 'no' and not bp.tipo_revisado(nuevo)


def test_cada_nivel_cuenta_por_separado():
    import mapeo_rf as mr
    rows = conjunto()
    bp.apply_typology(rows, es_borreguil)
    bp.set_niveles_revisados(rows[1], humedad='húmedo', cuando='2026-11-18')
    r = rows[1]
    assert bp.niveles_revisados(r) == ['humedad'] and r['humedad'] == 'húmedo'
    assert r['tipo_revisado'] == '' and bp.tipo_revisado(r)   # alguno, no los tres
    assert r['ambiente'] == r['ambiente_regla']               # el resto, la propuesta
    # sirve para el modelo de humedad, no para el de ambiente
    hum = {f['ID']: f for f in mr.muestras(rows, 'humedad', es_borreguil=es_borreguil)['tabla']}
    amb = {f['ID']: f for f in mr.muestras(rows, 'ambiente', es_borreguil=es_borreguil)['tabla']}
    assert hum['p1']['usada'] and hum['p1']['categoria'] == 'húmedo'
    assert not amb['p1']['usada'] and amb['p1']['motivo'] == 'tipo_sin_revisar'
    # recalcular la tipología no lo pisa, y otro nivel se suma al que había
    bp.apply_typology(rows, es_borreguil)
    assert bp.niveles_revisados(r) == ['humedad'] and r['humedad'] == 'húmedo'
    bp.set_niveles_revisados(r, ambiente='arroyo')
    assert bp.niveles_revisados(r) == ['ambiente', 'humedad']
    try:
        bp.set_niveles_revisados(r)
    except ValueError as e:
        assert 'ningún nivel' in str(e)
    else:
        raise AssertionError('sin ningún nivel no hay nada que revisar')


def _geojson(ruta, puntos):
    import json
    with open(ruta, 'w', encoding='utf-8') as f:
        json.dump({'type': 'FeatureCollection', 'features': [
            {'type': 'Feature', 'properties': p,
             'geometry': {'type': 'Point', 'coordinates': [-3.30 + i * 0.001, 37.05]}}
            for i, p in enumerate(puntos)]}, f)


def test_plantilla_de_campo_con_el_tipo_en_ingles_y_en_parte():
    with tempfile.TemporaryDirectory() as d:
        ruta = os.path.join(d, 'socio.geojson')
        _geojson(ruta, [
            {'id': 'a', 'presence': 1, 'moisture': 'wet', 'method': 'field',
             'date': '2026-07-15'},                              # solo la humedad
            {'id': 'b', 'presence': 1, 'environment': 'Stream', 'moisture': 'dry',
             'purity': 'mixed_rock'},                            # mayúsculas, guion bajo
            {'id': 'c', 'presence': 1},                          # sin tipo
            {'id': 'd', 'presence': 0, 'moisture': 'dry'},       # ausencia: sin tipo
            {'id': 'e', 'presence': 1, 'moisture': 'damp'},      # valor no reconocido
            {'id': 'f', 'presence': 1, 'environment': 'ladera', 'moisture': 'húmedo'},
        ] + [{'id': f'g{i}', 'presence': 1, 'moisture': 'wet'} for i in range(10)])
        rows = bp.read_points(ruta)
    info = bp.leer_etiquetas(rows)
    por = {r['ID']: r for r in rows}
    assert bp.niveles_revisados(por['a']) == ['humedad'] and por['a']['humedad'] == 'húmedo'
    assert por['a']['tipo_revisado_fecha'] == '2026-07-15'
    assert bp.niveles_revisados(por['b']) == list(bp.NIVELES_TIPO)
    assert (por['b']['ambiente'], por['b']['pureza']) == ('arroyo', 'mixto-roca')
    assert not bp.tipo_revisado(por['c']) and not bp.tipo_revisado(por['d'])
    assert not bp.tipo_revisado(por['e'])
    assert (por['f']['ambiente'], por['f']['humedad']) == ('ladera', 'húmedo')  # español
    t = info['tipos']
    assert t['humedad']['n'] == 13 and t['ambiente']['n'] == 2 and t['pureza']['n'] == 1
    assert t['humedad']['no_entendidos'] == {'damp': 1} and t['en_ausencias'] == 1
    textos = ' '.join(x for _, x in bp.resumen_etiquetas(info))
    assert 'damp' in textos and 'wet, dry' in textos and 'no borreguil' in textos
    # y lo revisado se conserva al recalcular la tipología
    for r in rows:
        r.update(decision='BORREGUIL VERIFICADO', ndmi_mean=0.02, slope_deg=15.0,
                 twi=5.0, ndwi_mean=-0.4, albedo_mean=0.15, ndvi_late=0.5)
    bp.apply_typology(rows, es_borreguil)
    assert por['a']['humedad'] == 'húmedo' and por['a']['humedad_regla'] == 'seco'


def test_un_fichero_de_la_app_con_marcas_antiguas_no_toma_la_propuesta_por_revisada():
    with tempfile.TemporaryDirectory() as d:
        ruta = os.path.join(d, 'antiguo.geojson')
        _geojson(ruta, [
            {'ID': 'r', 'Borreguil': 'si', 'ambiente': 'laguna', 'humedad': 'seco',
             'pureza': 'puro', 'tipo_revisado': 'si'},
            {'ID': 'p', 'Borreguil': 'si', 'ambiente': 'ladera', 'humedad': 'húmedo',
             'pureza': 'puro', 'tipo_revisado': ''},             # propuesta de la app
        ])
        rows = bp.read_points(ruta)
    bp.leer_etiquetas(rows)
    por = {r['ID']: r for r in rows}
    assert bp.niveles_revisados(por['r']) == list(bp.NIVELES_TIPO)
    assert not bp.tipo_revisado(por['p'])


def test_guardar_y_volver_a_cargar_un_nivel_suelto():
    rows = conjunto()
    bp.apply_typology(rows, es_borreguil)
    bp.set_niveles_revisados(rows[0], pureza='mixto-agua', cuando='2026-11-18T09:00:00')
    rows[0]['method'] = 'orthophoto'
    with tempfile.TemporaryDirectory() as d:
        ruta = os.path.join(d, 'puntos.geojson')
        bp.save_points_geojson(rows, ruta)
        import json
        with open(ruta, encoding='utf-8') as f:
            p0 = json.load(f)['features'][0]['properties']
        otra = bp.read_points(ruta)
    assert (p0['purity'], p0['environment'], p0['moisture']) == ('mixed-water', None, None)
    assert (p0['method'], p0['date']) == ('orthophoto', '2026-11-18')
    bp.leer_etiquetas(otra)
    assert bp.niveles_revisados(otra[0]) == ['pureza'] and otra[0]['pureza'] == 'mixto-agua'


def test_csv_con_lon_y_lat():
    with tempfile.TemporaryDirectory() as d:
        ruta = os.path.join(d, 'puntos.csv')
        with open(ruta, 'w', encoding='utf-8') as f:     # como lo guarda Excel en español
            f.write('id;Lon;Lat;presence;moisture\n'
                    'a;-3,3011;37,0512;1;wet\n'
                    'b;-3,3022;37,0523;0;\n'
                    'c;;37,05;1;dry\n')                  # sin longitud: se salta
        rows = bp.read_points(ruta)
        assert [r['ID'] for r in rows] == ['a', 'b']
        assert (rows[0]['lon'], rows[0]['lat']) == (-3.3011, 37.0512)
        info = bp.leer_etiquetas(rows)
        assert info['n_si'] == 1 and info['n_no'] == 1
        assert bp.niveles_revisados(rows[0]) == ['humedad']
        utm = os.path.join(d, 'utm.csv')
        with open(utm, 'w', encoding='utf-8') as f:
            f.write('x,y,presence\n456000,4100000,1\n')
        try:
            bp.read_points(utm)
        except ValueError as e:
            assert 'UTM' in str(e)
        else:
            raise AssertionError('unas coordenadas UTM no son grados')
        sin = os.path.join(d, 'sin.csv')
        with open(sin, 'w', encoding='utf-8') as f:
            f.write('nombre,presence\na,1\n')
        try:
            bp.read_points(sin)
        except ValueError as e:
            assert '`lon` y `lat`' in str(e)
        else:
            raise AssertionError('sin coordenadas no se puede leer')


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
