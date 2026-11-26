"""
launcher.py — Punto de entrada de la versión STANDALONE (ejecutable).

Copyright (C) 2026 los autores de este programa.
Este programa es software libre bajo la Licencia Pública General GNU, versión 3
o posterior. Se distribuye SIN NINGUNA GARANTÍA. Ver el fichero LICENSE o
<https://www.gnu.org/licenses/>.

Arranca el servidor Streamlit embebido dentro del propio ejecutable y abre el
navegador del usuario. No requiere instalar Python ni ninguna librería.

Diseño:
  · Los recursos empaquetados (app.py, modelos .joblib, KML de lagunas…) viven en
    una carpeta temporal de solo lectura que PyInstaller crea al arrancar
    (`sys._MEIPASS`) y borra al salir.
  · Todo lo que el usuario genera o guarda (modelos entrenados, salidas) va a una
    carpeta PERSISTENTE en su cuenta: ~/Borreguil  (BORREGUIL_DATA_DIR).
    Así «Guardar el modelo entrenado» sigue funcionando entre sesiones.

  · Al abrirse busca en GitHub si hay una versión nueva del código y, si la hay, la
    descarga, la comprueba y arranca con ella (actualizador.py; 25-11-2026, para las
    aulas). Sin conexión, sigue con el código que haya.

  · VERSIÓN CON VENTANA PROPIA (rama «pywebview», 26-11-2026, la autora: «hazme una
    versión de la app en pywebview»): la app se abre en una ventana del programa
    (pywebview, con el motor web del sistema: Edge WebView2 en Windows, WebKit en Mac)
    en lugar de en el navegador. El servidor de Streamlit va en otro proceso (este mismo
    programa con --servidor), porque pywebview y Streamlit quieren los dos el hilo
    principal. Al cerrar la ventana se cierra también el servidor. Si la ventana no se
    puede abrir (Linux sin GTK ni Qt, por ejemplo), se abre el navegador como siempre.

Ejecutar en desarrollo (sin compilar):  python launcher.py
Opciones:
  --navegador              abrir la app en el navegador, como la versión normal.
  --servidor               (interno) solo el servidor de Streamlit, sin ventana: lo
                           lanza la ventana en otro proceso.
  --probar-ventana         abrir la ventana, comprobar que la app se ve en ella y
                           cerrarla (lo usa la compilación).
  --selftest               autodiagnóstico del ejecutable (lo usa la compilación).
  --probar-codigo CARPETA  el mismo autodiagnóstico con el código de CARPETA (lo usa
                           el actualizador antes de dar por buena una versión nueva).
  --sin-actualizar         arrancar sin buscar versiones nuevas (o BORREGUIL_CODIGO_URL
                           vacía).
"""
import importlib
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

import actualizador

# La consola de Windows usa cp1252 y tanto este lanzador como el pipeline imprimen
# acentos y símbolos Unicode; sin esto se ven como «detecci�n» o abortan.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

APP_NAME = 'Borreguil'


def resource_dir() -> Path:
    """Carpeta con los recursos empaquetados (o la del código si no está compilado)."""
    if getattr(sys, 'frozen', False):
        return Path(getattr(sys, '_MEIPASS', Path(sys.executable).parent))
    return Path(__file__).resolve().parent


def data_dir() -> Path:
    """Carpeta persistente y escribible del usuario para modelos y salidas."""
    env = os.environ.get('BORREGUIL_DATA_DIR')
    base = Path(env) if env else Path.home() / APP_NAME
    base.mkdir(parents=True, exist_ok=True)
    return base


def free_port(preferred: int = 8501) -> int:
    """Devuelve `preferred` si está libre; si no, un puerto libre cualquiera."""
    for port in (preferred, 8502, 8503, 8504, 8505):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('127.0.0.1', port)) != 0:
                return port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return int(s.getsockname()[1])


def open_browser_when_ready(url: str, port: int, timeout: float = 90.0):
    """Abre el navegador en cuanto el servidor acepta conexiones."""
    def _wait():
        deadline = time.time() + timeout
        while time.time() < deadline:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(('127.0.0.1', port)) == 0:
                    try:
                        webbrowser.open(url)
                    except Exception:
                        pass
                    return
            time.sleep(0.4)
    threading.Thread(target=_wait, daemon=True).start()


def selftest(res: Path) -> int:
    """Autodiagnóstico: comprueba que TODO lo que la app necesita está realmente
    dentro del ejecutable. Se usa en la compilación automática (CI).

    Es imprescindible porque el servidor web responde «200 OK» aunque el script de
    la app falle: sin esta prueba se podría publicar un binario que abre una
    pantalla de error. Devuelve 0 si todo va bien.
    """
    print('Autodiagnóstico del ejecutable')
    print('-' * 50)
    fallos = []

    sys.path.insert(0, str(res))

    def comprobar(desc, fn):
        try:
            fn()
            print(f'  OK      {desc}')
        except Exception as exc:                 # noqa: BLE001
            print(f'  FALLO   {desc}: {type(exc).__name__}: {exc}')
            fallos.append(desc)

    # 1) Dependencias científicas y geoespaciales (aquí es donde suelen aparecer
    #    los módulos que el empaquetado se ha dejado fuera).
    # Los módulos de la app se importan por su nombre (importlib), no con «import»:
    # así PyInstaller no los mete compilados dentro del ejecutable y siempre se usan
    # los de la carpeta del código, que el actualizador puede cambiar.
    def _pipeline():
        bp = importlib.import_module('borreguil_pipeline')
        bp._imports()
    comprobar('importar el pipeline y sus dependencias', _pipeline)
    comprobar('módulos de la interfaz (point_profile, i18n, mapeo_rf, mapeo_raster, '
              'mapeo_tif, mapeo_ambiente, modelos_remotos)',
              lambda: (__import__('point_profile'), __import__('i18n'),
                       __import__('mapeo_rf'), __import__('mapeo_raster'),
                       __import__('mapeo_tif'), __import__('mapeo_ambiente'),
                       __import__('modelos_remotos')))

    # El mapa de tipos usa partes de shapely, pyproj y Pillow que nada más toca, y
    # escribe sus GeoTIFF con tifffile (sin rasterio: Windows lo bloquea en algunos
    # equipos).
    def _mapa_de_tipos():
        import tempfile
        import numpy as np
        from shapely.geometry import box
        mra = importlib.import_module('mapeo_raster')
        mt = importlib.import_module('mapeo_tif')
        rej = mra.rejilla((500000, 4100000, 500040, 4100040), 32630)
        dentro = mra.dentro(box(500000, 4100000, 500020, 4100040), rej, (0, 0, 4, 4))
        if int(dentro.sum()) != 8:
            raise RuntimeError(f'la máscara de una zona sale mal ({int(dentro.sum())} de 8)')
        mra.limites_lonlat(rej)
        with tempfile.TemporaryDirectory() as tmp:
            ruta = Path(tmp) / 'prueba.tif'
            mt.escribir(ruta, rej, np.arange(16, dtype='uint8').reshape(4, 4), sin_dato=255)
            with mt.abrir(ruta) as f:
                if f.rej != rej or int(f.valor(3, 3)) != 15:
                    raise RuntimeError('un GeoTIFF no se lee como se escribió')
        if not mra.a_png(np.zeros((2, 2, 4), dtype='uint8')).startswith('data:image/png'):
            raise RuntimeError('no se pudo crear una imagen PNG')
    comprobar('piezas del mapa de tipos (rejilla, máscara, GeoTIFF, PNG)', _mapa_de_tipos)

    # El ambiente por cercanía usa scipy.ndimage (distancias, manchas) y shapely (pintar
    # los cauces en la rejilla).
    def _ambiente_por_cercania():
        import numpy as np
        from shapely.geometry import LineString
        ma = importlib.import_module('mapeo_ambiente')
        ff, cc = np.mgrid[0:40, 0:31]
        cauce = ma.cauces_del_relieve(3000.0 - 2.0 * ff + 3.0 * np.abs(cc - 15), 1.0)
        if not cauce[30:, 15].all() or cauce[:, 5].any():
            raise RuntimeError('los cauces del relieve salen mal')
        rej = {'epsg': 32630, 'escala': 10, 'x0': 500000.0, 'y0': 4100400.0,
               'ancho': 31, 'alto': 40}
        linea = ma.rasterizar(LineString([(500005, 4100205), (500305, 4100205)]), rej)
        if int(linea.sum()) != 31 or float(ma.distancia(linea)[10, 3]) != 90.0:
            raise RuntimeError('la distancia a un cauce sale mal')
    comprobar('piezas del ambiente por cercanía (relieve, cauces, distancias)',
              _ambiente_por_cercania)
    comprobar('streamlit', lambda: __import__('streamlit'))
    comprobar('scipy.ndimage', lambda: __import__('scipy.ndimage', fromlist=['ndimage']))
    comprobar('sklearn', lambda: __import__('sklearn.ensemble', fromlist=['ensemble']))
    comprobar('pyproj', lambda: __import__('pyproj'))
    comprobar('pywebview (la ventana del programa)', lambda: __import__('webview'))
    # La cuenta compartida de MonitorEO (gee_backend, 26-11-2026): sus permisos de una
    # hora van con google.oauth2 y se piden con requests.
    comprobar('cuenta compartida de Earth Engine (google.oauth2, requests)',
              lambda: (__import__('google.oauth2.credentials'), __import__('requests')))

    # 2) Recursos que deben viajar dentro del ejecutable.
    comprobar('app.py incluido', lambda: (res / 'app.py').read_text(encoding='utf-8'))
    modelos = sorted(res.glob('*.joblib'))
    comprobar(f'modelos incluidos ({len(modelos)})',
              lambda: (_ for _ in ()).throw(FileNotFoundError('no hay .joblib'))
              if not modelos else None)

    # 3) Un modelo debe poder cargarse y predecir con la scikit-learn empaquetada.
    def _cargar_modelo():
        import warnings
        import joblib
        import numpy as np
        from sklearn.exceptions import InconsistentVersionWarning
        pref = [m for m in modelos if m.name == 'rf_sierra_nevada.joblib'] or modelos
        with warnings.catch_warnings(record=True) as avisos:
            warnings.simplefilter('always')
            bundle = joblib.load(pref[0])
            if [a for a in avisos if issubclass(a.category, InconsistentVersionWarning)]:
                raise RuntimeError(
                    f'{pref[0].name} se guardó con otra versión de scikit-learn; '
                    'las predicciones podrían ser inválidas')
        n = len(bundle['features'])
        p = bundle['model'].predict_proba(np.array([bundle['medians']], dtype=np.float32))
        assert 0.0 <= float(p[0][1]) <= 1.0, 'probabilidad fuera de rango'
        print(f'          ({n} variables, predicción de prueba = {float(p[0][1]):.3f})')
    comprobar('cargar un modelo y predecir', _cargar_modelo)

    print('-' * 50)
    if fallos:
        print(f'RESULTADO: {len(fallos)} FALLO(S) -> ' + '; '.join(fallos))
        return 1
    print('RESULTADO: correcto, el ejecutable está completo.')
    return 0


def probar_en_otro_proceso(carpeta):
    """El autodiagnóstico con el código de `carpeta`, en otro proceso (el mismo
    ejecutable, o este fichero sin compilar): así un código que no funciona con las
    librerías del instalador no llega a cargarse en este. Devuelve (ok, texto)."""
    if getattr(sys, 'frozen', False):
        cmd = [sys.executable, '--probar-codigo', str(carpeta)]
    else:
        cmd = [sys.executable, str(Path(__file__).resolve()), '--probar-codigo',
               str(carpeta)]
    entorno = dict(os.environ, **{actualizador.VARIABLE_URL: ''})
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8',
                           errors='replace', timeout=600, env=entorno)
    except Exception as exc:                     # noqa: BLE001
        return False, f'{type(exc).__name__}: {exc}'
    texto = (r.stdout or '') + (r.stderr or '')
    return r.returncode == 0, texto[-2000:]


def buscar_version_nueva(res, out):
    """Busca en GitHub una versión nueva del código y la deja lista (actualizador.py).
    Escribe en la consola lo que pasa; nunca para el arranque."""
    if '--sin-actualizar' in sys.argv or not actualizador.url_manifiesto():
        return None
    print('  Buscando una versión nueva de la app…', flush=True)
    r = actualizador.actualizar(res, out, probar=probar_en_otro_proceso,
                                aviso=lambda t: print(t, flush=True))
    linea = actualizador.texto_estado(r)
    if linea:
        print(linea, flush=True)
    if r.get('estado') == 'no_pasa_prueba' and r.get('detalle'):
        print('  Lo que dijo la comprobación (últimas líneas):')
        print('\n'.join('    ' + x for x in r['detalle'].strip().splitlines()[-8:]))
    return r


TITULO_VENTANA = f'{APP_NAME} · Wet Meadow Finder'

_PAGINA = """<!doctype html><html lang="es"><head><meta charset="utf-8">
<title>{titulo}</title><style>
html,body{{height:100%;margin:0}}
body{{display:flex;align-items:center;justify-content:center;background:#F1F8E9;
color:#1B1B1B;font-family:"Segoe UI",system-ui,-apple-system,sans-serif}}
.caja{{max-width:34em;padding:2em;text-align:center}}
h1{{color:#2E7D32;font-size:1.6em;margin:0 0 .5em}} p{{line-height:1.5;margin:.4em 0}}
.gira{{width:2.2em;height:2.2em;margin:1.4em auto 0;border-radius:50%;
border:.3em solid #C5E1A5;border-top-color:#2E7D32;animation:g 1s linear infinite}}
@keyframes g{{to{{transform:rotate(360deg)}}}}
</style></head><body><div class="caja">{cuerpo}</div></body></html>"""

CARGANDO = _PAGINA.format(
    titulo=TITULO_VENTANA,
    cuerpo='<h1>Borreguil · Wet Meadow Finder</h1><p>Buscador y mapa de borreguiles</p>'
           '<p>Abriendo la app…</p>'
           '<p>La primera vez puede tardar hasta un minuto.</p><div class="gira"></div>')

NO_ARRANCA = _PAGINA.format(
    titulo=TITULO_VENTANA,
    cuerpo='<h1>No se ha podido abrir la app</h1>'
           '<p>El motor de la app no ha arrancado. El motivo está en la ventana de '
           'consola.</p><p>Cierra esta ventana y vuelve a abrir el programa. Si sigue '
           'igual, copia el texto de la consola y avisa en '
           'github.com/Bea-Cazorla/borreguil-detector/issues.</p>')


def lanzar_servidor(port: int):
    """El servidor de Streamlit en otro proceso: este mismo programa con --servidor.
    No vuelve a buscar versiones nuevas: ya lo ha hecho este proceso."""
    if getattr(sys, 'frozen', False):
        cmd = [sys.executable]
    else:
        cmd = [sys.executable, str(Path(__file__).resolve())]
    cmd += ['--servidor', '--sin-actualizar']
    return subprocess.Popen(cmd, env=dict(os.environ, BORREGUIL_PUERTO=str(port)))


def esperar_servidor(port: int, hijo, timeout: float = 180.0) -> bool:
    """True en cuanto el servidor responde; False si el proceso termina o se agota el
    tiempo."""
    url = f'http://localhost:{port}/_stcore/health'
    fin = time.time() + timeout
    while time.time() < fin:
        if hijo.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.4)
    return False


def parar_servidor(hijo):
    if hijo is None or hijo.poll() is not None:
        return
    hijo.terminate()
    try:
        hijo.wait(timeout=10)
    except Exception:
        hijo.kill()


def abrir_en_ventana(url: str, port: int, out: Path, probar: bool = False):
    """La app en su propia ventana (pywebview). Devuelve (estado, proceso del servidor):

    - ('cerrada', None): la persona ha cerrado la ventana (y con ella, el servidor).
    - ('sin_ventana', hijo o None): no se puede abrir la ventana; si el servidor ya
      está en marcha, se devuelve para seguir con él en el navegador.
    - con probar: ('ok' | 'no_arranca' | 'no_se_ve', None)."""
    try:
        import webview
    except Exception as exc:                     # noqa: BLE001
        print(f'  (sin ventana propia: {type(exc).__name__}: {exc})', flush=True)
        return 'sin_ventana', None

    hijo = lanzar_servidor(port)
    resultado = {'estado': 'cerrada'}

    def cargar(ventana):
        if not esperar_servidor(port, hijo):
            resultado['estado'] = 'no_arranca'
            ventana.load_html(NO_ARRANCA)
            if probar:
                ventana.destroy()
            return
        ventana.load_url(url)
        if not probar:
            return
        # Prueba: la app tiene que verse en la ventana (el panel lateral escrito por
        # app.py, no solo el servidor).
        resultado['estado'] = 'no_se_ve'
        fin = time.time() + 150
        while time.time() < fin:
            time.sleep(2)
            try:
                texto = ventana.evaluate_js(
                    'document.body ? document.body.innerText : ""') or ''
            except Exception:                    # noqa: BLE001
                texto = ''
            if 'Área de estudio' in texto:
                resultado['estado'] = 'ok'
                break
        ventana.destroy()

    webview.settings['ALLOW_DOWNLOADS'] = True               # botones de descarga
    webview.settings['OPEN_EXTERNAL_LINKS_IN_BROWSER'] = True  # GitHub, formulario…
    ventana = webview.create_window(TITULO_VENTANA, html=CARGANDO, width=1400,
                                    height=900, min_size=(900, 600),
                                    maximized=not probar, text_select=True)
    try:
        # private_mode=False y una carpeta propia: la ventana recuerda lo que el
        # navegador recordaría (preferencias de Streamlit) entre sesiones.
        webview.start(cargar, (ventana,), private_mode=False,
                      storage_path=str(out / '.ventana'))
    except KeyboardInterrupt:
        pass
    except Exception as exc:                     # noqa: BLE001
        print(f'  (no se ha podido abrir la ventana: {type(exc).__name__}: {exc})',
              flush=True)
        return 'sin_ventana', hijo
    parar_servidor(hijo)
    return resultado['estado'], None


def arrancar_streamlit(app_path: Path, port: int):
    """El servidor de Streamlit en este proceso (hasta que se cierre)."""
    # Opciones de Streamlit. OJO: las claves usan guion bajo (la propia librería las
    # convierte a 'server.address', etc.) y hay que cargarlas con load_config_options
    # ANTES de run() — pasarlas solo a run() no surte efecto.
    # server_address='localhost' evita exponer la app a toda la red local.
    flag_options = {
        # Dentro del ejecutable, Streamlit no reconoce su propia instalación y activa
        # el «modo desarrollo», que es incompatible con fijar el puerto. Hay que
        # desactivarlo explícitamente o el arranque falla.
        'global_developmentMode': False,
        'server_address': 'localhost',
        'server_port': port,
        'server_headless': True,
        'server_fileWatcherType': 'none',
        'server_maxUploadSize': 50,
        'browser_gatherUsageStats': False,
    }
    from streamlit import config as _st_config
    from streamlit.web import bootstrap
    _st_config._main_script_path = str(app_path)   # como hace la CLI de Streamlit
    bootstrap.load_config_options(flag_options=flag_options)
    try:
        bootstrap.run(str(app_path), is_hello=False, args=[], flag_options=flag_options)
    except KeyboardInterrupt:
        print('\nCerrando…')


def main():
    res = resource_dir()

    if '--selftest' in sys.argv:
        sys.exit(selftest(res))

    if '--probar-codigo' in sys.argv:
        i = sys.argv.index('--probar-codigo')
        if i + 1 >= len(sys.argv):
            sys.exit('ERROR: falta la carpeta del código que hay que probar')
        carpeta = Path(sys.argv[i + 1]).resolve()
        actualizador.usar_codigo_de(carpeta)
        sys.exit(selftest(carpeta))

    out = data_dir()

    # El código: el que trae el instalador o, si se ha descargado una versión más
    # nueva y está completa, esa (en la carpeta del usuario).
    buscar_version_nueva(res, out)
    codigo, descargada = actualizador.carpeta_a_usar(res, out)

    # El script de la app y sus módulos hermanos viajan en la carpeta del código.
    app_path = codigo / 'app.py'
    if not app_path.exists():
        codigo, descargada = res, None
        app_path = res / 'app.py'
    if not app_path.exists():
        sys.exit(f'ERROR: no se encuentra app.py en {res}')
    if descargada:
        actualizador.usar_codigo_de(codigo)
        print(f'  Código de la versión {descargada["version"]} '
              f'(commit {descargada["commit"][:7]}), descargada de GitHub.', flush=True)
    else:
        sys.path.insert(0, str(res))

    # Carpeta escribible para lo que genere el usuario (modelos, descargas).
    os.environ['BORREGUIL_DATA_DIR'] = str(out)
    try:
        os.chdir(out)          # las salidas relativas caen en la carpeta del usuario
    except Exception:
        pass

    # Configuración de Streamlit por entorno (evita depender de .streamlit/config.toml
    # y del asistente de primer arranque que pide un email).
    servidor = '--servidor' in sys.argv
    port = (int(os.environ['BORREGUIL_PUERTO'])
            if servidor and os.environ.get('BORREGUIL_PUERTO') else free_port())
    os.environ.setdefault('STREAMLIT_SERVER_HEADLESS', 'true')          # no abre él el navegador
    os.environ.setdefault('STREAMLIT_BROWSER_GATHER_USAGE_STATS', 'false')
    os.environ.setdefault('STREAMLIT_SERVER_PORT', str(port))
    os.environ.setdefault('STREAMLIT_SERVER_ADDRESS', 'localhost')
    os.environ.setdefault('STREAMLIT_SERVER_FILE_WATCHER_TYPE', 'none')  # innecesario y frágil al compilar
    os.environ.setdefault('STREAMLIT_SERVER_MAX_UPLOAD_SIZE', '50')
    os.environ.setdefault('STREAMLIT_THEME_PRIMARY_COLOR', '#2E7D32')
    os.environ.setdefault('STREAMLIT_THEME_BACKGROUND_COLOR', '#FFFFFF')
    os.environ.setdefault('STREAMLIT_THEME_SECONDARY_BACKGROUND_COLOR', '#F1F8E9')
    os.environ.setdefault('STREAMLIT_THEME_TEXT_COLOR', '#1B1B1B')

    url = f'http://localhost:{port}'
    if servidor:                          # el proceso que lanza la ventana
        arrancar_streamlit(app_path, port)
        return

    sin_navegador = os.environ.get('BORREGUIL_NO_BROWSER', '').lower() in ('1', 'true',
                                                                              'yes')
    probar = '--probar-ventana' in sys.argv
    con_ventana = probar or not ('--navegador' in sys.argv or sin_navegador)

    print('=' * 64)
    print(f'  {APP_NAME} · Wet Meadow Finder — buscador y mapa de borreguiles')
    print('=' * 64)
    if con_ventana:
        print('  La app se abre en su propia ventana.')
    else:
        print(f'  Abriendo el navegador en:  {url}')
    print(f'  Tus modelos y salidas:     {out}')
    print()
    print('  Deja ESTA ventana (la consola) abierta mientras uses el programa.')
    if con_ventana:
        print('  Para cerrarlo: cierra la ventana de la app.')
    else:
        print('  Para cerrarlo: cierra esta ventana o pulsa Ctrl+C.')
    print('=' * 64, flush=True)

    if con_ventana:
        estado, hijo = abrir_en_ventana(url, port, out, probar=probar)
        if probar:
            print(f'  Prueba de la ventana: {estado}', flush=True)
            sys.exit(0 if estado == 'ok' else 1)
        if estado != 'sin_ventana':
            return
        print('  No se ha podido abrir la ventana de la app: se abre el navegador en',
              url, flush=True)
        if hijo is not None:              # el servidor ya está en marcha
            open_browser_when_ready(url, port)
            try:
                hijo.wait()
            except KeyboardInterrupt:
                parar_servidor(hijo)
            return

    if not sin_navegador:
        open_browser_when_ready(url, port)
    arrancar_streamlit(app_path, port)

if __name__ == '__main__':
    # En el ejecutable, un proceso hijo que lance una librería (multiprocessing,
    # joblib) vuelve a ejecutar este mismo programa. Sin esta línea arrancaría la app
    # entera otra vez: en Mac el autodiagnóstico se quedaba colgado (25-11-2026).
    import multiprocessing
    multiprocessing.freeze_support()
    main()
