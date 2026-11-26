"""Pruebas de los puntos por cuartiles de una capa (random_points.py, 26-11-2026): leer
la capa sin GDAL (GeoPackage con sqlite3, GeoJSON, un .zip con una capa), reconocer una
columna que ya trae los cuartiles o calcularlos de una numérica, el reparto (75 % en el
cuartil 1 de entrada, ajustable), que cada punto cae en un polígono de su cuartil y
dentro del área de estudio, que dentro de cada grupo se reparte según la superficie y
qué pasa si la capa no cae en el área.

Se ejecutan con `python tests/test_puntos_cuartiles.py`; no necesitan red ni GDAL.
"""
import json
import sqlite3
import struct
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pyproj
import shapely
from shapely.geometry import Point, box

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import random_points as rp       # noqa: E402

A_UTM = pyproj.Transformer.from_crs(4326, 32630, always_xy=True)
A_GRADOS = pyproj.Transformer.from_crs(32630, 4326, always_xy=True)
X0, Y0 = 465000.0, 4100000.0                     # Sierra Nevada, en UTM 30N

# Ocho cuadrados de 500 m en fila, en UTM: el 0 vale más (cuartil 1) y el 7 menos.
CUADRADOS = [box(X0 + 600 * i, Y0, X0 + 600 * i + 500, Y0 + 500) for i in range(8)]
VALORES = [80.0, 70.0, 6.0, 5.0, 0.4, 0.3, 0.02, 0.01]
CUARTIL = [1, 1, 2, 2, 3, 3, 4, 4]
CARPETA = Path(tempfile.mkdtemp(prefix='prueba_cuartiles_'))


def _blob(geom, srs=32630, sobre=False):
    """Una geometría como la guarda un GeoPackage: «GP», versión, banderas, srs y
    (si se pide) el sobre, y después el WKB."""
    flags = 0b1 | ((1 << 1) if sobre else 0)            # little endian (+ sobre xy)
    cab = b'GP' + bytes([0, flags]) + struct.pack('<i', srs)
    if sobre:
        cab += struct.pack('<4d', *[geom.bounds[i] for i in (0, 2, 1, 3)])
    return cab + shapely.to_wkb(geom, byte_order=1)


def gpkg(nombre, geoms, columnas, srs=32630, sobre=False):
    """Un GeoPackage mínimo con una capa de polígonos, escrito con sqlite3."""
    ruta = CARPETA / nombre
    ruta.unlink(missing_ok=True)
    con = sqlite3.connect(ruta)
    con.execute('pragma application_id = 1196444487')        # «GPKG»
    con.executescript("""
        create table gpkg_spatial_ref_sys (srs_name text, srs_id integer primary key,
            organization text, organization_coordsys_id integer, definition text,
            description text);
        create table gpkg_contents (table_name text primary key, data_type text,
            identifier text, description text, last_change text, min_x real,
            min_y real, max_x real, max_y real, srs_id integer);
        create table gpkg_geometry_columns (table_name text, column_name text,
            geometry_type_name text, srs_id integer, z integer, m integer);
    """)
    con.execute("insert into gpkg_spatial_ref_sys values ('WGS 84 / UTM 30N', 32630, "
                "'EPSG', 32630, 'undefined', '')")
    con.execute("insert into gpkg_contents values ('area_probable', 'features', "
                "'area_probable', '', '', 0, 0, 0, 0, ?)", (srs,))
    con.execute("insert into gpkg_geometry_columns values ('area_probable', 'geom', "
                "'POLYGON', ?, 0, 0)", (srs,))
    defs = ', '.join(f'"{c}" real' for c in columnas)
    con.execute(f'create table area_probable (fid integer primary key, geom blob, {defs})')
    for i, g in enumerate(geoms):
        vals = [columnas[c][i] for c in columnas]
        con.execute(f'insert into area_probable values (?, ?, {", ".join("?" * len(vals))})',
                    [i + 1, _blob(g, srs, sobre)] + vals)
    con.commit()
    con.close()
    return ruta


def en_grados(g):
    """Una geometría de UTM 30N a grados."""
    return shapely.transform(
        g, lambda xy: np.column_stack(A_GRADOS.transform(xy[:, 0], xy[:, 1])))


AREA = en_grados(box(X0 - 200, Y0 - 200, X0 + 600 * 8 + 200, Y0 + 700))
CUADRADOS_GRADOS = [en_grados(c) for c in CUADRADOS]


def capa_de(columna='cuartil'):
    ruta = gpkg('capa.gpkg', CUADRADOS, {'cuartil': CUARTIL, 'valor': VALORES})
    c = rp.leer_capa(ruta)
    return {'geoms': c['geoms'], 'valores': list(c['atributos'][columna]),
            'columna': columna}


def cuadrado_de(p):
    """El índice del cuadrado en el que cae un punto (o None). Los puntos se ajustan al
    centro del píxel de 10 m: pueden moverse hasta 7,1 m."""
    pt = Point(p['lon'], p['lat'])
    for i, c in enumerate(CUADRADOS_GRADOS):
        if c.buffer(0.0001).contains(pt):
            return i
    return None


# ------------------------------------------------------------------ leer la capa
def test_lee_un_geopackage_sin_gdal_y_lo_pasa_a_grados():
    for sobre in (False, True):
        ruta = gpkg('capa.gpkg', CUADRADOS, {'cuartil': CUARTIL, 'valor': VALORES},
                    sobre=sobre)
        c = rp.leer_capa(ruta)
        assert c['capa'] == 'area_probable' and len(c['geoms']) == 8
        assert list(c['atributos'].columns) == ['fid', 'cuartil', 'valor']
        assert list(c['atributos']['valor']) == VALORES
        minx, miny, maxx, maxy = c['geoms'][0].bounds
        assert -3.5 < minx < maxx < -3.3 and 37.0 < miny < maxy < 37.1, c['geoms'][0].bounds
        assert abs(c['geoms'][0].area - CUADRADOS_GRADOS[0].area) < 1e-12


def test_lee_un_geojson_y_un_zip_con_una_capa():
    fc = {'type': 'FeatureCollection', 'features': [
        {'type': 'Feature', 'properties': {'cuartil': q, 'valor': v},
         'geometry': shapely.geometry.mapping(g)}
        for g, q, v in zip(CUADRADOS_GRADOS, CUARTIL, VALORES)]}
    ruta = CARPETA / 'capa.geojson'
    ruta.write_text(json.dumps(fc), encoding='utf-8')
    c = rp.leer_capa(ruta)
    assert len(c['geoms']) == 8 and list(c['atributos']['cuartil']) == CUARTIL
    # un .zip con una capa (y la basura que deja macOS) se lee igual
    zp = CARPETA / 'capa.zip'
    with zipfile.ZipFile(zp, 'w') as z:
        z.write(gpkg('una.gpkg', CUADRADOS, {'cuartil': CUARTIL}), 'carpeta/una.gpkg')
        z.writestr('__MACOSX/carpeta/._una.gpkg', b'basura')
    assert len(rp.leer_capa(zp)['geoms']) == 8


def test_un_zip_con_varias_capas_pide_que_se_suba_solo_una():
    zp = CARPETA / 'varias.zip'
    with zipfile.ZipFile(zp, 'w') as z:
        z.write(gpkg('a.gpkg', CUADRADOS, {'cuartil': CUARTIL}), 'a.gpkg')
        z.write(gpkg('b.gpkg', CUADRADOS, {'cuartil': CUARTIL}), 'b.gpkg')
    try:
        rp.leer_capa(zp)
    except ValueError as e:
        assert 'varias capas' in str(e) and 'a.gpkg' in str(e)
    else:
        raise AssertionError('tenía que avisar de que hay varias capas')


# ------------------------------------------------------------------ los cuartiles
def test_la_columna_de_cuartiles_va_primero_y_el_id_no_cuenta():
    import pandas as pd
    at = pd.DataFrame({'fid': range(8), 'id': range(8), 'valor': VALORES,
                       'cuartil': CUARTIL, 'nombre': list('abcdefgh')})
    assert rp.columnas_para_cuartiles(at) == ['cuartil', 'valor']
    assert rp.es_columna_de_cuartiles('cuartil', CUARTIL)
    assert rp.es_columna_de_cuartiles('Quartile', [1, 2, 3, 4])
    assert not rp.es_columna_de_cuartiles('cuartil', [1, 2, 5])      # no son cuartiles
    assert not rp.es_columna_de_cuartiles('valor', [1, 2, 3, 4])     # no se llama así


def test_cuartiles_ya_hechos_se_usan_tal_cual_y_si_no_se_calculan():
    assert list(rp.cuartiles_de(CUARTIL, 'cuartil')) == CUARTIL
    # de una numérica: el 1 es el 25 % de los polígonos con los valores más altos
    assert list(rp.cuartiles_de(VALORES, 'valor')) == CUARTIL
    assert list(rp.cuartiles_de(VALORES[::-1], 'valor')) == CUARTIL[::-1]
    # sin valor, fuera (0)
    assert list(rp.cuartiles_de([5, None, 3, 2, 1], 'valor')) == [1, 0, 2, 3, 4]


# ------------------------------------------------------------------ el reparto
def test_75_en_el_cuartil_1_y_25_en_el_resto_cada_uno_en_su_poligono():
    filas, st = rp.generate(AREA, 200, mode='cuartiles', capa=capa_de(), seed=5)
    assert len(filas) == 200 and st['kept'] == 200
    assert sum(f['cuartil'] == 1 for f in filas) == 150
    assert st['cuartiles'][1]['salen'] == 150 and st['cuartiles'][2]['salen'] == 50
    assert st['cuartiles'][1]['poligonos'] == 2 and st['cuartiles'][2]['poligonos'] == 6
    assert abs(st['cuartiles'][1]['ha'] - 50) < 1 and abs(st['cuartiles'][2]['ha'] - 150) < 2
    for f in filas:
        i = cuadrado_de(f)
        assert i is not None and CUARTIL[i] == f['cuartil'], f
        assert f['source'] == 'random_cuartiles' and f['ID'].startswith('rand_')
    # los cuartiles 2, 3 y 4 salen todos (tienen la misma superficie)
    for q in (2, 3, 4):
        assert 8 <= sum(f['cuartil'] == q for f in filas) <= 28, q


def test_el_reparto_se_puede_cambiar_y_sale_igual_con_la_misma_semilla():
    todo, _ = rp.generate(AREA, 60, mode='cuartiles', capa=capa_de(), frac_q1=1.0, seed=2)
    assert all(f['cuartil'] == 1 for f in todo) and len(todo) == 60
    mitad, _ = rp.generate(AREA, 60, mode='cuartiles', capa=capa_de(), frac_q1=0.5, seed=2)
    assert sum(f['cuartil'] == 1 for f in mitad) == 30
    otra, _ = rp.generate(AREA, 60, mode='cuartiles', capa=capa_de(), frac_q1=0.5, seed=2)
    assert otra == mitad


def test_con_una_columna_numerica_sale_lo_mismo_que_con_los_cuartiles():
    filas, st = rp.generate(AREA, 100, mode='cuartiles', capa=capa_de('valor'), seed=5)
    assert sum(f['cuartil'] == 1 for f in filas) == 75
    assert all(CUARTIL[cuadrado_de(f)] == f['cuartil'] for f in filas)


def test_dentro_de_cada_grupo_segun_la_superficie():
    # dos polígonos del cuartil 1: uno con 4 veces la superficie del otro
    grande = box(X0, Y0, X0 + 1000, Y0 + 1000)
    chico = box(X0 + 2000, Y0, X0 + 2500, Y0 + 500)
    resto = box(X0 + 3000, Y0, X0 + 3500, Y0 + 500)
    ruta = gpkg('superficie.gpkg', [grande, chico, resto], {'cuartil': [1, 1, 2]})
    c = rp.leer_capa(ruta)
    capa = {'geoms': c['geoms'], 'valores': list(c['atributos']['cuartil']),
            'columna': 'cuartil'}
    filas, _ = rp.generate(None, 500, mode='cuartiles', capa=capa, frac_q1=1.0, seed=11)
    g = en_grados(grande).buffer(0.0001)
    en_grande = sum(g.contains(Point(f['lon'], f['lat'])) for f in filas)
    assert 3.0 < en_grande / (500 - en_grande) < 5.5, en_grande


def test_solo_cuentan_los_poligonos_del_area_de_estudio():
    # un área con los cuatro últimos cuadrados (valores 0,4 a 0,01): con la columna
    # numérica, los cuartiles se calculan entre ellos (el 0,4 pasa a ser el cuartil 1)
    area = en_grados(box(X0 + 600 * 4 - 50, Y0 - 50, X0 + 600 * 8, Y0 + 550))
    filas, st = rp.generate(area, 40, mode='cuartiles', capa=capa_de('valor'), seed=3)
    assert st['poligonos_area'] == 4 and len(filas) == 40
    assert {cuadrado_de(f) for f in filas} <= {4, 5, 6, 7}
    assert all(area.buffer(0.0001).contains(Point(f['lon'], f['lat'])) for f in filas)
    assert {cuadrado_de(f) for f in filas if f['cuartil'] == 1} == {4}
    # con la columna de cuartiles de la capa, se respetan los suyos (3 y 4): no hay
    # cuartil 1 en esa área, así que solo salen los del resto
    filas, st = rp.generate(area, 40, mode='cuartiles', capa=capa_de('cuartil'), seed=3)
    assert st['cuartiles'][1]['poligonos'] == 0 and st['cuartiles'][1]['salen'] == 0
    assert len(filas) == 10 and {f['cuartil'] for f in filas} <= {3, 4}


def test_una_capa_fuera_del_area_no_da_puntos_y_lo_dice():
    lejos = box(-5.0, 40.0, -4.9, 40.1)
    filas, st = rp.generate(lejos, 50, mode='cuartiles', capa=capa_de(), seed=1)
    assert filas == [] and st['reason'] == 'capa_fuera_del_area'
    filas, st = rp.generate(AREA, 50, mode='cuartiles', capa=None)
    assert filas == [] and st['reason'] == 'sin_capa'


def test_el_area_de_estudio_tambien_se_lee_de_un_geopackage():
    import study_area as sa
    g = sa.load_vector(gpkg('area.gpkg', CUADRADOS[:3], {'cuartil': [1, 2, 3]}))
    assert g.geom_type in ('Polygon', 'MultiPolygon')
    assert abs(g.area - sum(c.area for c in CUADRADOS_GRADOS[:3])) < 1e-10


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
