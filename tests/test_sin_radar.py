"""Pruebas de las versiones SIN radar de los modelos de Sierra Nevada.

Con Planetary Computer (o «Saltar Sentinel-2») el radar (Sentinel-1) no se lee. El
modelo con radar, si le falta, acierta bastante menos (AUC 0,793) que uno entrenado
desde el principio sin él (0,833). Por eso cada modelo que viene con la app tiene una
versión sin radar en la carpeta sin_radar/, que la app usa sola cuando el radar no se
ha leído, y lo dice. Se ejecutan con `python tests/test_sin_radar.py`; no hace falta red
y tardan unos segundos.
"""
import ast
import os
import shutil
import sys
import tempfile
from pathlib import Path

APP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP))
# sin buscar modelos en GitHub al abrir la app (modelos_remotos.py)
os.environ.setdefault('BORREGUIL_MODELOS_URL', '')

import borreguil_pipeline as bp   # noqa: E402
import i18n                        # noqa: E402

bp._imports()

TMP = Path(tempfile.mkdtemp(prefix='sin_radar_'))
S1 = set(bp.S1_FEATS)
MODELOS = {'rf_sierra_nevada.joblib': (22, 0.82), 'rf_sierra_nevada_cir.joblib': (27, 0.93)}


def funciones_de_app(*nombres):
    """Las funciones de app.py con esos nombres, sin ejecutar la app entera."""
    src = (APP / 'app.py').read_text(encoding='utf-8')
    arbol = ast.parse(src)
    trozos = [ast.get_source_segment(src, n) for n in arbol.body
              if isinstance(n, ast.FunctionDef) and n.name in nombres]
    ns = {'Path': Path, 'APP_DIR': APP, 'bp': bp, 'i18n': i18n,
          'DESCARGADOS': TMP / 'modelos_descargados'}
    exec('\n\n'.join(trozos), ns)
    return ns


def test_las_versiones_sin_radar_no_usan_radar():
    for nombre, (n_vars, auc_min) in MODELOS.items():
        ruta = bp.ruta_sin_radar(APP / nombre)
        assert ruta is not None and ruta.parent.name == 'sin_radar', (nombre, ruta)
        b = bp.load_model_bundle(ruta)
        assert not S1 & set(b['features']), b['features']
        assert len(b['features']) == n_vars, len(b['features'])
        assert b['auc_groupkfold'] >= auc_min, b['auc_groupkfold']
        assert 'sin radar' in b['training_zone'].lower(), b['training_zone']
        # el de la app sí usa radar (si no, no haría falta la versión sin él)
        assert S1 & set(bp.load_model_bundle(APP / nombre)['features'])


def test_no_salen_en_el_desplegable():
    # list_models busca *.joblib solo en la carpeta de la app, no en sin_radar/
    assert not list(APP.glob('*_sin_radar.joblib'))


def test_solo_los_modelos_de_la_app_tienen_version_sin_radar():
    otro = TMP / 'rf_mi_zona.joblib'
    shutil.copy2(APP / 'rf_sierra_nevada.joblib', otro)
    assert bp.ruta_sin_radar(otro) is None
    assert bp.ruta_sin_radar(None) is None
    # uno guardado por quien usa la app con el mismo nombre, en otra carpeta: tampoco
    ns = funciones_de_app('modelo_sin_radar', 'modelo_segun_radar')
    copia = TMP / 'rf_sierra_nevada.joblib'
    shutil.copy2(APP / 'rf_sierra_nevada.joblib', copia)
    assert ns['modelo_sin_radar'](str(copia)) is None
    assert ns['modelo_segun_radar'](str(copia), False) == str(copia)


def test_la_app_elige_segun_haya_radar():
    ns = funciones_de_app('modelo_sin_radar', 'modelo_segun_radar')
    for nombre in MODELOS:
        ruta = str(APP / nombre)
        assert ns['modelo_segun_radar'](ruta, True) == ruta
        sin = ns['modelo_segun_radar'](ruta, False)
        assert Path(sin) == APP / 'sin_radar' / nombre.replace('.joblib', '_sin_radar.joblib')


def test_hay_radar():
    assert not bp.hay_radar([])
    assert not bp.hay_radar([{'s1_vv_mean': float('nan')}, {'s1_vv_mean': ''}, {}])
    assert bp.hay_radar([{'s1_vv_mean': float('nan')}, {'s1_vv_mean': -12.5}])
    assert bp.hay_radar([{'s1_vv_mean': '-12.5'}])


def test_sin_radar_dice_que_modelo_se_ha_usado():
    """Puntos sin radar → la versión sin radar da la probabilidad, y el título de
    «Modelo usado» lo dice (en los dos idiomas)."""
    tabla = bp.leer_tabla_entrenamiento(APP / bp.TABLA_ENTRENAMIENTO)
    rows = [{k: v for k, v in r.items() if k not in S1} for r in tabla[:40]]
    assert not bp.hay_radar(rows)
    ns = funciones_de_app('_zona_partida', '_titulo_modelo', '_es_sin_radar')
    ruta = bp.ruta_sin_radar(APP / 'rf_sierra_nevada.joblib')
    proba, info = bp.train_rf(rows, default_model_path=ruta)
    assert proba is not None and len(proba) == 40 and all(0 <= p <= 1 for p in proba)
    assert ns['_es_sin_radar'](info)
    try:
        i18n._LANG = 'es'
        t = ns['_titulo_modelo'](info)
        assert 'versión sin radar' in t and 'el radar no se ha leído' in t, t
        i18n._LANG = 'en'
        t = ns['_titulo_modelo'](info)
        assert 'version without radar' in t and 'radar was not read' in t, t
    finally:
        i18n._LANG = 'es'
    # con radar, el título de siempre
    proba, info = bp.train_rf(tabla[:40], default_model_path=str(APP / 'rf_sierra_nevada.joblib'))
    assert not ns['_es_sin_radar'](info)
    assert 'sin radar' not in ns['_titulo_modelo'](info)


def test_aviso_en_el_panel_con_planetary_computer():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(APP / 'app.py'), default_timeout=120)
    at.run()
    assert not at.exception, [str(e.value)[:200] for e in at.exception]

    def aviso():
        return [str(i.value) for i in at.info if 'entrenada sin radar' in str(i.value)]

    assert not aviso(), 'con Earth Engine no debe salir'
    radio = [r for r in at.radio if r.label == 'Backend'][0]
    radio.set_value([o for o in radio.options if 'Planetary' in o][0])
    at.run()
    assert not at.exception, [str(e.value)[:200] for e in at.exception]
    a = aviso()
    assert len(a) == 1 and '(AUC 0,833)' in a[0], a


if __name__ == '__main__':
    pruebas = [(n, f) for n, f in sorted(globals().items())
               if n.startswith('test_') and callable(f)]
    fallos = 0
    try:
        for nombre, f in pruebas:
            try:
                f()
                print(f'  ok     {nombre}')
            except Exception as e:                      # noqa: BLE001
                fallos += 1
                print(f'  FALLO  {nombre}: {type(e).__name__}: {e}')
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    print(f'{len(pruebas) - fallos}/{len(pruebas)} pruebas correctas')
    sys.exit(1 if fallos else 0)
