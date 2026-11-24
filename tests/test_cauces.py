"""Pruebas del ambiente «arroyo» con los ríos y arroyos de OpenStreetMap.

Un borreguil es de arroyo si está a 30 m o menos de un río o arroyo de OpenStreetMap
(no las acequias, los canales ni las fuentes). Los de Sierra Nevada van con la app
(cauces_sierra_nevada.geojson); en otras zonas se descargan de Overpass, y si no
responde, «arroyo» se decide por la pendiente. Se ejecutan con
`python tests/test_cauces.py`; no usan internet (Overpass va simulado).
"""
import json
import math
import os
import sys
import tempfile
from pathlib import Path

APP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP))
os.chdir(APP)

import borreguil_pipeline as bp  # noqa: E402

TH = bp.HIER_THRESH
# El Parque Nacional de Sierra Nevada (Granada y Almería), aproximado
PARQUE = (-3.60, 36.92, -2.60, 37.20)


def punto(**kw):
    r = {'lon': -3.30, 'lat': 37.05, 'slope_deg': 15.0, 'ndwi_mean': -0.40,
         'ndmi_mean': 0.05, 'dist_laguna_m': 5000.0}
    r.update(kw)
    return r


# ---------------- la regla
def test_junto_a_un_rio_es_arroyo():
    assert bp.classify_hierarchy(punto(dist_cauce_m=20.0))['ambiente'] == 'arroyo'
    assert bp.classify_hierarchy(punto(dist_cauce_m=30.0))['ambiente'] == 'arroyo'
    assert bp.classify_hierarchy(punto(dist_cauce_m=30.5))['ambiente'] == 'ladera'


def test_con_cauces_el_llano_no_basta():
    # antes, pendiente < 7° era arroyo aunque no hubiera ningún cauce cerca
    assert bp.classify_hierarchy(punto(dist_cauce_m=400.0, slope_deg=3.0))['ambiente'] == 'ladera'


def test_la_laguna_va_antes_que_el_arroyo():
    r = punto(dist_cauce_m=5.0, dist_laguna_m=60.0)
    assert bp.classify_hierarchy(r)['ambiente'] == 'laguna'


def test_sin_cauces_decide_la_pendiente():
    nan = float('nan')
    assert bp.classify_hierarchy(punto(dist_cauce_m=nan, slope_deg=5.0))['ambiente'] == 'arroyo'
    assert bp.classify_hierarchy(punto(dist_cauce_m=nan, slope_deg=12.0))['ambiente'] == 'ladera'
    assert bp.classify_hierarchy(punto(slope_deg=12.0))['ambiente'] == 'ladera'
    assert bp.classify_hierarchy(punto(slope_deg=None))['ambiente'] == '—'


# ---------------- la distancia
def test_distancia_al_cauce_mas_cercano():
    from shapely.geometry import LineString
    # un arroyo norte-sur en lon -3.30; los puntos, a 0, 25 y 100 m al este
    rio = LineString([(-3.30, 37.00), (-3.30, 37.10)])
    m_por_grado = 111_320 * math.cos(math.radians(37.05))
    rows = [{'lon': -3.30 + d / m_por_grado, 'lat': 37.05} for d in (0.0, 25.0, 100.0)]
    assert bp.attach_cauce_distance(rows, [rio]) == 1
    for r, d in zip(rows, (0.0, 25.0, 100.0)):
        assert abs(r['dist_cauce_m'] - d) < 1.0, (r['dist_cauce_m'], d)


def test_sin_cauces_la_distancia_queda_vacia():
    rows = [punto()]
    assert bp.attach_cauce_distance(rows, None) == 0
    assert math.isnan(rows[0]['dist_cauce_m'])
    assert bp.attach_cauce_distance(rows, []) == 0
    assert math.isnan(rows[0]['dist_cauce_m'])


# ---------------- de dónde salen
class Overpass:
    """Sustituye a _descargar_cauces: cuenta las llamadas y devuelve un arroyo (o nada)."""
    def __init__(self, responde=True):
        self.llamadas, self.responde = 0, responde

    def __call__(self, o, s, e, n, timeout=45):
        from shapely.geometry import LineString
        self.llamadas += 1
        if not self.responde:
            return None
        return [LineString([(o, s), (e, n)])]


def con_overpass(falso, f):
    viejo = bp._descargar_cauces
    bp._descargar_cauces = falso
    bp._cauces_descargados.clear()
    try:
        return f()
    finally:
        bp._descargar_cauces = viejo
        bp._cauces_descargados.clear()


def test_en_sierra_nevada_no_hace_falta_internet():
    ov = Overpass()
    lineas, origen = con_overpass(ov, lambda: bp.cauces_osm((-3.40, 37.02, -3.30, 37.10)))
    assert origen == 'capa' and len(lineas) > 50 and ov.llamadas == 0, (origen, ov.llamadas)


def test_fuera_de_sierra_nevada_se_descargan_una_vez():
    ov = Overpass()
    pirineo = (0.90, 42.55, 1.00, 42.65)

    def dos_veces():
        a = bp.cauces_osm(pirineo)
        b = bp.cauces_osm(pirineo)
        return a, b
    (la, oa), (lb, ob) = con_overpass(ov, dos_veces)
    assert oa == ob == 'overpass' and len(la) == 1 and lb is la
    assert ov.llamadas == 1                     # la segunda vez, de lo ya descargado


def test_si_overpass_no_responde_se_avisa():
    ov = Overpass(responde=False)
    lineas, origen = con_overpass(ov, lambda: bp.cauces_osm((0.90, 42.55, 1.00, 42.65)))
    assert lineas is None and origen == 'sin_conexion'


def test_una_zona_que_sale_de_la_capa_se_descarga():
    # medio dentro y medio fuera de la capa de Sierra Nevada: no sirve la capa
    lineas, cob = bp.leer_capa_cauces()
    ov = Overpass()
    zona = (cob[2] - 0.05, 37.00, cob[2] + 0.05, 37.10)
    _, origen = con_overpass(ov, lambda: bp.cauces_osm(zona))
    assert origen == 'overpass' and ov.llamadas == 1


def test_lee_una_capa_cualquiera():
    capa = {'type': 'FeatureCollection', 'cobertura': [-1, -1, 1, 1], 'features': [
        {'type': 'Feature', 'properties': {'waterway': 'stream'},
         'geometry': {'type': 'LineString', 'coordinates': [[0, 0], [0.5, 0.5]]}},
        {'type': 'Feature', 'properties': {'waterway': 'canal'},          # acequia: no
         'geometry': {'type': 'LineString', 'coordinates': [[0, 0], [-0.5, 0.5]]}}]}
    with tempfile.TemporaryDirectory() as d:
        ruta = Path(d) / 'c.geojson'
        ruta.write_text(json.dumps(capa), encoding='utf-8')
        lineas, cob = bp.leer_capa_cauces(ruta)
        assert len(lineas) == 1 and cob == (-1, -1, 1, 1)
        lineas, origen = bp.cauces_osm((-0.1, -0.1, 0.1, 0.1), ruta)
        assert origen == 'capa' and len(lineas) == 1
    assert bp.leer_capa_cauces(Path(d) / 'no_esta.geojson') == (None, None)


# ---------------- la capa que va con la app
def test_la_capa_de_sierra_nevada():
    ruta = APP / bp.CAUCES_SN
    d = json.loads(ruta.read_text(encoding='utf-8'))
    tipos = {f['properties']['waterway'] for f in d['features']}
    assert tipos <= set(bp.TIPOS_CAUCE) and 'stream' in tipos, tipos
    assert 'OpenStreetMap' in d['licencia'] and 'ODbL' in d['licencia']
    o, s, e, n = d['cobertura']
    assert o <= PARQUE[0] and s <= PARQUE[1] and e >= PARQUE[2] and n >= PARQUE[3]
    # cubre también todos los puntos de los que aprendió el modelo
    import csv
    with open(APP / bp.TABLA_ENTRENAMIENTO, encoding='utf-8') as f:
        pts = [(float(r['lon']), float(r['lat'])) for r in csv.DictReader(f)]
    assert all(o <= x <= e and s <= y <= n for x, y in pts)
    assert len(d['features']) > 500


def test_el_panel_ya_no_tiene_la_casilla_de_osm():
    # scikit-learn antes que la app: si se carga después del intento de rasterio (que
    # Windows bloquea en este equipo), Windows bloquea también una de sus DLL
    import sklearn.ensemble  # noqa: F401
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(APP / 'app.py'), default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    assert not [c.label for c in at.checkbox if 'OSM' in c.label]


if __name__ == '__main__':
    fallos = 0
    pruebas = [(n, f) for n, f in sorted(globals().items())
               if n.startswith('test_') and callable(f)]
    for nombre, f in pruebas:
        try:
            f()
            print(f'  ok     {nombre}')
        except Exception as e:                    # noqa: BLE001
            fallos += 1
            print(f'  FALLO  {nombre}: {type(e).__name__}: {e}')
    print(f'{len(pruebas) - fallos}/{len(pruebas)} pruebas correctas')
    sys.exit(1 if fallos else 0)
