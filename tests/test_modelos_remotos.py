"""Pruebas de los modelos que la app descarga de GitHub (modelos_remotos.py y app.py).

Al abrirse, la app lee modelos.json y descarga a la carpeta del usuario los modelos que
no tiene o que han cambiado, comprobando su huella. Aquí la «web» es una carpeta local
(direcciones file://), así que no hace falta red. Se ejecutan con
`python tests/test_modelos_remotos.py` (o con pytest).
"""
import ast
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

APP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP))

import sklearn.ensemble  # noqa: F401,E402  (primero, por Smart App Control)

import borreguil_pipeline as bp   # noqa: E402
import i18n                        # noqa: E402
import modelos_remotos as mrm      # noqa: E402


def _modelo(ruta, semilla, n=150_000):
    """Un fichero que pasa por modelo (los de verdad pesan más de 100 kB)."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_bytes(bytes((semilla + i * 7) % 251 for i in range(n)))
    return ruta


def _web(modelos, sin_radar=(), extra=None, formato=1):
    """Una «web» local con esos modelos y su modelos.json; devuelve (carpeta, url)."""
    web = Path(tempfile.mkdtemp(prefix='web_'))
    lista = []
    for nombre, semilla in modelos:
        d = mrm.entrada(_modelo(web / nombre, semilla), nombre='Prueba')
        if nombre in sin_radar:
            sr = nombre.replace('.joblib', '_sin_radar.joblib')
            d['sin_radar'] = mrm.entrada(_modelo(web / 'sin_radar' / sr, semilla + 1),
                                         url=f'sin_radar/{sr}')
        lista.append(d)
    lista += extra or []
    (web / 'modelos.json').write_text(
        json.dumps({'formato': formato, 'modelos': lista}), encoding='utf-8')
    return web, (web / 'modelos.json').as_uri()


def _carpetas():
    base = Path(tempfile.mkdtemp(prefix='modelos_'))
    (base / 'app').mkdir()
    return base / 'app', base / 'descargados'


def test_descarga_un_modelo_nuevo_y_su_version_sin_radar():
    web, url = _web([('rf_alpes.joblib', 3)], sin_radar={'rf_alpes.joblib'})
    app, desc = _carpetas()
    r = mrm.actualizar(desc, incluidos=app, url=url)
    assert r['estado'] == 'descargados' and r['nuevos'] == ['rf_alpes.joblib'], r
    assert not r['actualizados'] and not r['fallos']
    assert mrm.huella(desc / 'rf_alpes.joblib') == mrm.huella(web / 'rf_alpes.joblib')
    sr = desc / 'sin_radar' / 'rf_alpes_sin_radar.joblib'
    assert sr.exists() and bp.ruta_sin_radar(desc / 'rf_alpes.joblib') == sr
    assert not [p for p in desc.rglob('*') if p.name.endswith('.part')]
    # la segunda vez ya está al día
    assert mrm.actualizar(desc, incluidos=app, url=url)['estado'] == 'al_dia'


def test_si_la_app_ya_lo_trae_igual_no_lo_descarga():
    web, url = _web([('rf_sierra_nevada.joblib', 5)])
    app, desc = _carpetas()
    shutil.copy2(web / 'rf_sierra_nevada.joblib', app)
    r = mrm.actualizar(desc, incluidos=app, url=url)
    assert r['estado'] == 'al_dia' and not desc.exists(), r


def test_un_modelo_cambiado_se_actualiza_y_volver_atras_quita_la_copia():
    web, url = _web([('rf_sierra_nevada.joblib', 5)])
    app, desc = _carpetas()
    _modelo(app / 'rf_sierra_nevada.joblib', 9)                    # el del instalador, otro
    r = mrm.actualizar(desc, incluidos=app, url=url)
    assert r['estado'] == 'descargados' and r['actualizados'] == ['rf_sierra_nevada.joblib']
    # el equipo vuelve a publicar el del instalador: la copia descargada sobra
    shutil.copy2(app / 'rf_sierra_nevada.joblib', web / 'rf_sierra_nevada.joblib')
    d = json.loads((web / 'modelos.json').read_text(encoding='utf-8'))
    d['modelos'][0].update(mrm.entrada(web / 'rf_sierra_nevada.joblib'))
    (web / 'modelos.json').write_text(json.dumps(d), encoding='utf-8')
    r = mrm.actualizar(desc, incluidos=app, url=url)
    assert r['estado'] == 'al_dia' and not (desc / 'rf_sierra_nevada.joblib').exists(), r


def test_una_huella_que_no_coincide_no_deja_nada():
    web, url = _web([('rf_alpes.joblib', 3)])
    _modelo(web / 'rf_alpes.joblib', 4)                    # cambiado sin poner la lista al día
    app, desc = _carpetas()
    r = mrm.actualizar(desc, incluidos=app, url=url)
    assert r['estado'] == 'error' and 'huella' in r['fallos'][0], r
    assert not desc.exists() or not list(desc.iterdir())


def test_las_entradas_mal_escritas_se_descartan_sin_parar_las_demas():
    buena = {'sha256': 'a' * 64, 'bytes': 10}
    web, url = _web([('rf_alpes.joblib', 3)], extra=[
        dict(buena, archivo='../fuera.joblib'),
        dict(buena, archivo='otro.joblib', url='https://otro.sitio/otro.joblib'),
        dict(buena, archivo='corto.joblib', sha256='abc'),
        dict(buena, archivo='enorme.joblib', bytes=10 ** 10),
        dict(buena, archivo='script.py'),
        {'archivo': 'rf_alpes.joblib', 'sha256': 'b' * 64, 'bytes': 5},      # repetido
    ])
    app, desc = _carpetas()
    r = mrm.actualizar(desc, incluidos=app, url=url)
    assert r['nuevos'] == ['rf_alpes.joblib'] and len(r['descartados']) == 6, r
    assert sorted(p.name for p in desc.iterdir()) == ['rf_alpes.joblib']


def test_sin_conexion_desactivado_y_listas_que_no_sirven():
    app, desc = _carpetas()
    r = mrm.actualizar(desc, incluidos=app, url='https://127.0.0.1:9/modelos.json',
                       timeout=2)
    assert r['estado'] == 'sin_conexion' and r['fallos'], r
    assert mrm.actualizar(desc, incluidos=app, url='')['estado'] == 'desactivado'
    r = mrm.actualizar(desc, incluidos=app, url='http://ejemplo.org/modelos.json')
    assert r['estado'] == 'error' and 'HTTPS' in r['fallos'][0], r
    _, url = _web([], formato=99)
    r = mrm.actualizar(desc, incluidos=app, url=url)
    assert r['estado'] == 'error' and 'más nueva' in r['fallos'][0], r
    assert not desc.exists()


def test_la_variable_de_entorno_cambia_o_apaga_la_lista():
    antes = os.environ.pop(mrm.VARIABLE_URL, None)
    try:
        assert mrm.url_manifiesto() == mrm.URL_POR_DEFECTO
        assert mrm.URL_POR_DEFECTO.startswith('https://raw.githubusercontent.com/')
        os.environ[mrm.VARIABLE_URL] = ' https://otro.repo/modelos.json '
        assert mrm.url_manifiesto() == 'https://otro.repo/modelos.json'
        os.environ[mrm.VARIABLE_URL] = ''
        assert mrm.url_manifiesto() == ''
    finally:
        os.environ.pop(mrm.VARIABLE_URL, None)
        if antes is not None:
            os.environ[mrm.VARIABLE_URL] = antes


def test_la_lista_del_repositorio_coincide_con_sus_modelos():
    import publicar_modelo as pm
    assert pm.diferencias() == [], pm.diferencias()
    lista = mrm.interpretar(json.loads((APP / 'modelos.json').read_text(encoding='utf-8')),
                            mrm.URL_POR_DEFECTO)
    assert not lista['descartados'], lista['descartados']
    nombres = [m['archivo'] for m in lista['modelos']]
    assert nombres[0] == 'rf_sierra_nevada.joblib' and 'rf_sierra_nevada_cir.joblib' in nombres
    assert all(m['sin_radar'] for m in lista['modelos'])
    assert all(m['url'].startswith(mrm.REPO_RAW) for m in lista['modelos'])


def funciones_de_app(ns, *nombres):
    """Las funciones de app.py con esos nombres, sin ejecutar la app entera."""
    src = (APP / 'app.py').read_text(encoding='utf-8')
    trozos = [ast.get_source_segment(src, n) for n in ast.parse(src).body
              if isinstance(n, ast.FunctionDef) and n.name in nombres]
    assert len(trozos) == len(nombres)
    ns = dict(ns, Path=Path, bp=bp, i18n=i18n)
    exec('\n\n'.join(trozos), ns)
    return ns


def test_la_app_usa_el_descargado_antes_que_el_suyo():
    base = Path(tempfile.mkdtemp(prefix='app_'))
    app, datos = base / 'app', base / 'datos'
    desc = datos / mrm.CARPETA
    _modelo(app / 'rf_sierra_nevada.joblib', 1)
    _modelo(app / 'sin_radar' / 'rf_sierra_nevada_sin_radar.joblib', 2)
    _modelo(desc / 'rf_sierra_nevada.joblib', 3)                       # versión nueva
    _modelo(desc / 'rf_alpes.joblib', 4)
    _modelo(desc / 'sin_radar' / 'rf_alpes_sin_radar.joblib', 5)
    _modelo(datos / 'rf_mi_zona.joblib', 6)                            # guardado por quien usa la app
    ns = funciones_de_app({'APP_DIR': app, 'DATA_DIR': datos, 'DESCARGADOS': desc},
                          'carpetas_de_modelos', 'model_path', 'list_models',
                          '_es_modelo_real', 'modelo_sin_radar')
    assert ns['list_models']() == ['rf_sierra_nevada.joblib', 'rf_alpes.joblib',
                                   'rf_mi_zona.joblib']
    assert ns['model_path']('rf_sierra_nevada.joblib') == desc / 'rf_sierra_nevada.joblib'
    assert ns['model_path']('rf_mi_zona.joblib') == datos / 'rf_mi_zona.joblib'
    # sin radar: el descargado con la suya; el nuevo de Sierra Nevada no trae: ninguna
    assert ns['modelo_sin_radar'](desc / 'rf_alpes.joblib') == \
        desc / 'sin_radar' / 'rf_alpes_sin_radar.joblib'
    assert ns['modelo_sin_radar'](desc / 'rf_sierra_nevada.joblib') is None
    assert ns['modelo_sin_radar'](app / 'rf_sierra_nevada.joblib') == \
        app / 'sin_radar' / 'rf_sierra_nevada_sin_radar.joblib'
    assert ns['modelo_sin_radar'](datos / 'rf_mi_zona.joblib') is None
    # ejecutando desde la carpeta de la app (sin carpeta de datos aparte)
    ns = funciones_de_app({'APP_DIR': app, 'DATA_DIR': app, 'DESCARGADOS': app / mrm.CARPETA},
                          'carpetas_de_modelos', 'model_path')
    assert ns['carpetas_de_modelos']() == [app / mrm.CARPETA, app]
    assert ns['model_path']('rf_sierra_nevada.joblib') == app / 'rf_sierra_nevada.joblib'


def test_lo_que_dice_el_panel():
    ns = funciones_de_app({}, 'texto_modelos_github')
    t = ns['texto_modelos_github']
    i18n.set_lang('es')
    assert t({'estado': 'al_dia'}) == 'Modelos al día con GitHub.'
    assert t({'estado': 'descargados', 'nuevos': ['rf_alpes.joblib'],
              'actualizados': ['rf_sierra_nevada.joblib']}) == (
        'Modelos descargados de GitHub al abrir la app: rf_alpes.joblib (nuevo), '
        'rf_sierra_nevada.joblib (actualizado).')
    assert 'Sin conexión' in t({'estado': 'sin_conexion', 'fallos': ['x']})
    assert 'huella' in t({'estado': 'error', 'fallos': ['rf_x.joblib: la huella no coincide']})
    assert t({'estado': 'desactivado'}) == '' and t(None) == ''
    i18n.set_lang('en')
    try:
        assert t({'estado': 'al_dia'}) == 'Models up to date with GitHub.'
    finally:
        i18n.set_lang('es')


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
