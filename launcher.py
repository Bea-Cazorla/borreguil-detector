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

  · LA APP SE ABRE EN SU PROPIA VENTANA (26-11-2026, la autora: «hazme una versión de
    la app en pywebview»; desde el mismo día es la versión principal, en main): una
    ventana del programa (pywebview, con el motor web del sistema: Edge WebView2 en
    Windows, WebKit en Mac) en lugar del navegador. El servidor de Streamlit va en otro
    proceso (este mismo programa con --servidor), porque pywebview y Streamlit quieren
    los dos el hilo principal. Al cerrar la ventana se cierra también el servidor. Si la
    ventana no se puede abrir (Windows sin el motor de Edge, Linux sin GTK ni Qt…), se
    abre el navegador como antes.

  · ZOOM DE LA VENTANA (27-11-2026, la autora: «dale más zoom a la app para que la
    vean en una videollamada»): Ctrl + rueda del ratón y Ctrl + / Ctrl − / Ctrl 0
    agrandan, achican o devuelven al 100 %, y la ventana recuerda el zoom para la
    próxima vez (carpeta del usuario, .ventana/zoom.txt). pywebview lo traía apagado.

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
import contextlib
import html
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
              'mapeo_tif, mapeo_ambiente, modelos_remotos, registros)',
              lambda: (__import__('point_profile'), __import__('i18n'),
                       __import__('mapeo_rf'), __import__('mapeo_raster'),
                       __import__('mapeo_tif'), __import__('mapeo_ambiente'),
                       __import__('modelos_remotos'), __import__('registros')))

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


def buscar_version_nueva(res, out, mostrar=None):
    """Busca en GitHub una versión nueva del código y la deja lista (actualizador.py).
    Escribe en la consola lo que pasa y, con `mostrar`, se lo pasa también a la ventana
    (la página de carga); nunca para el arranque. Devuelve lo que dice el actualizador, o
    None si no se busca."""
    if '--sin-actualizar' in sys.argv or not actualizador.url_manifiesto():
        return None

    def avisar(texto):
        print(texto, flush=True)
        if mostrar is not None:
            mostrar(texto.strip())

    avisar('  Buscando una versión nueva de la app…')
    r = actualizador.actualizar(res, out, probar=probar_en_otro_proceso, aviso=avisar)
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
body{{display:flex;align-items:center;justify-content:center;background:#221F20;
color:#ECEFF1;font-family:"Segoe UI",system-ui,-apple-system,sans-serif}}
.caja{{max-width:34em;padding:2em;text-align:center}}
h1{{color:#F5F7F8;font-size:1.6em;margin:0 0 .5em}} p{{line-height:1.5;margin:.4em 0;color:#AEB6BE}}
.gira{{width:2.2em;height:2.2em;margin:1.4em auto 0;border-radius:50%;
border:.3em solid #403A3C;border-top-color:#7CC496;animation:g 1s linear infinite}}
@keyframes g{{to{{transform:rotate(360deg)}}}}
</style></head><body><div class="caja">{cuerpo}</div></body></html>"""

def cargando(texto='Abriendo la app…'):
    """La página de la ventana mientras arranca: qué se está haciendo (buscar una versión
    nueva, descargarla y comprobarla, abrir la app). Sin consola (la app del Escritorio
    del instalador de Mac), es lo único que se ve mientras tanto (27-11-2026)."""
    return _PAGINA.format(
        titulo=TITULO_VENTANA,
        cuerpo='<h1>Borreguil · Wet Meadow Finder</h1><p>Buscador y mapa de borreguiles</p>'
               f'<p>{html.escape(texto)}</p>'
               # sin repetirlo si el texto ya dice lo que tarda
               + ('' if 'minuto' in texto else
                  '<p>La primera vez puede tardar hasta un minuto.</p>')
               + '<div class="gira"></div>')


# Dónde se explica cómo instalar el programa (lo que hay que hacer cuando una versión
# nueva necesita un instalador nuevo).
INSTRUCCIONES_MAC = ('https://github.com/monitoreo-obsnev/borreguil-detector/blob/main/'
                     'TUTORIAL.md#22-descomprimir-y-ejecutar')


def aviso_de_la_actualizacion(r):
    """Lo que la persona tiene que saber de la búsqueda de versiones nuevas, para enseñarlo
    en la ventana: (título, texto, enlace), o None si no hay nada que hacer. En Windows
    lo dice también la consola; en Mac, la app del Escritorio no tiene, y sin esto nadie
    se enteraría de que hace falta volver a instalar (27-11-2026)."""
    if not r:
        return None
    estado, version = r.get('estado'), r.get('version') or ''
    if sys.platform == 'darwin':
        reinstalar = ('vuelve a ejecutar el instalador: en la app Terminal, pega la línea '
                      'del instalador (TUTORIAL, apartado 2.2) y pulsa Intro. Tus modelos '
                      'y salidas se conservan.')
        enlace = INSTRUCCIONES_MAC
    else:
        reinstalar = ('descarga el programa nuevo e instálalo como la primera vez. Tus '
                      'modelos y salidas se conservan.')
        enlace = actualizador.DESCARGAS_RELEASE
    if estado == 'necesita_instalador':
        return ('Hay una versión nueva de la app',
                f'La versión {version} necesita un programa nuevo, porque han cambiado '
                'Python o sus librerías. Mientras, la app sigue con la versión que '
                f'tenías.\n\nPara tenerla, cierra la app y {reinstalar}', enlace)
    if estado == 'no_pasa_prueba':
        return ('La versión nueva no se ha podido usar',
                f'Se ha descargado la versión {version} de la app, pero no ha pasado la '
                'comprobación: la app sigue con la versión que tenías.\n\nSi vuelve a '
                f'pasar, cierra la app y {reinstalar}', enlace)
    return None


def mostrar_aviso(ventana, aviso):
    """Enseña el aviso en un cuadro de diálogo de la ventana; si la persona acepta, abre
    las instrucciones en el navegador."""
    titulo, texto, enlace = aviso
    mensaje = texto + '\n\n¿Abrir las instrucciones en el navegador?'
    if sys.platform == 'darwin':                 # en Mac, pywebview no enseña el título
        mensaje = f'{titulo}\n\n{mensaje}'
    try:
        abrir = ventana.create_confirmation_dialog(titulo, mensaje)
    except Exception as exc:                     # noqa: BLE001
        print(f'  (no se ha podido enseñar el aviso: {type(exc).__name__}: {exc})',
              flush=True)
        return
    if abrir and enlace:
        with contextlib.suppress(Exception):
            webbrowser.open(enlace)

def no_arranca():
    """La página de «no se ha podido abrir la app». El motivo está en la consola o, si el
    programa se abre desde la app del Escritorio que crea el instalador de Mac (sin
    consola, 27-11-2026), en su fichero de salida, BORREGUIL_SALIDA."""
    salida = os.environ.get('BORREGUIL_SALIDA', '').strip()
    donde = (f'en el fichero {html.escape(salida)}' if salida
             else 'en la ventana de consola')
    return _PAGINA.format(
        titulo=TITULO_VENTANA,
        cuerpo='<h1>No se ha podido abrir la app</h1>'
               f'<p>El motor de la app no ha arrancado. El motivo está {donde}.</p>'
               '<p>Cierra esta ventana y vuelve a abrir el programa. Si sigue igual, '
               'copia las últimas líneas de ese texto y avisa en '
               'github.com/monitoreo-obsnev/borreguil-detector/issues.</p>')


def lanzar_servidor(port: int):
    """El servidor de Streamlit en otro proceso: este mismo programa con --servidor.
    No vuelve a buscar versiones nuevas: ya lo ha hecho este proceso (y elige el código,
    el descargado o el del instalador, al arrancar)."""
    if getattr(sys, 'frozen', False):
        cmd = [sys.executable]
    else:
        cmd = [sys.executable, str(Path(__file__).resolve())]
    cmd += ['--servidor', '--sin-actualizar']
    return subprocess.Popen(cmd, env=dict(os.environ, BORREGUIL_PUERTO=str(port),
                                          STREAMLIT_SERVER_PORT=str(port)))


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


_WEBVIEW2 = '{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}'   # Microsoft Edge WebView2 Runtime


def hay_motor_de_edge() -> bool:
    """Windows: si está el motor de Edge (WebView2 Runtime) que usa la ventana. Viene
    con Windows 11 y con casi todo Windows 10 al día. Sin él, pywebview usaría el de
    Internet Explorer, en el que la app sale en blanco: entonces mejor el navegador.
    BORREGUIL_SIN_WEBVIEW2=1 hace como si no estuviera (para probarlo)."""
    if os.environ.get('BORREGUIL_SIN_WEBVIEW2', '').lower() in ('1', 'true', 'yes'):
        return False
    try:
        import winreg
    except ImportError:                          # no es Windows
        return True
    claves = [(winreg.HKEY_CURRENT_USER, rf'SOFTWARE\Microsoft\EdgeUpdate\Clients\{_WEBVIEW2}'),
              (winreg.HKEY_LOCAL_MACHINE,
               rf'SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{_WEBVIEW2}'),
              (winreg.HKEY_LOCAL_MACHINE, rf'SOFTWARE\Microsoft\EdgeUpdate\Clients\{_WEBVIEW2}')]
    for raiz, ruta in claves:
        try:
            with winreg.OpenKey(raiz, ruta) as clave:
                version, _ = winreg.QueryValueEx(clave, 'pv')
        except OSError:
            continue
        if str(version or '').strip() not in ('', '0.0.0.0'):
            return True
    return False


# Zoom de la ventana: los mismos pasos que un navegador.
ZOOM_PASOS = (0.5, 0.67, 0.75, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0)

# pywebview apaga las teclas propias del navegador (bien: Ctrl+R o F5 recargarían la app
# y se perdería el análisis), y con ellas Ctrl + / Ctrl − / Ctrl 0. Las atiende la
# página y le pide el zoom a Python (zoom_ventana). Sin AltGr (Ctrl+Alt en Windows).
TECLAS_ZOOM_JS = """
if (!window.__zoomBorreguil) {
  window.__zoomBorreguil = true;
  window.addEventListener('keydown', function (e) {
    if (!(e.ctrlKey || e.metaKey) || e.altKey) return;
    var paso = null;
    if (e.key === '+' || e.key === '=' || e.code === 'NumpadAdd') paso = 1;
    else if (e.key === '-' || e.code === 'NumpadSubtract') paso = -1;
    else if (e.key === '0' || e.code === 'Numpad0') paso = 0;
    if (paso === null || !window.pywebview || !window.pywebview.api) return;
    e.preventDefault();
    window.pywebview.api.zoom_ventana(paso);
  }, true);
}
"""


def siguiente_zoom(actual: float, paso: int) -> float:
    """El zoom tras Ctrl + (paso 1), Ctrl − (-1) o Ctrl 0 (0, el 100 %)."""
    if paso == 0:
        return 1.0
    if paso > 0:
        return next((z for z in ZOOM_PASOS if z > actual + 1e-3), ZOOM_PASOS[-1])
    return next((z for z in reversed(ZOOM_PASOS) if z < actual - 1e-3), ZOOM_PASOS[0])


def leer_zoom(fichero: Path) -> float:
    try:
        z = float(fichero.read_text(encoding='utf-8').strip())
    except Exception:                            # noqa: BLE001
        return 1.0
    return min(max(z, ZOOM_PASOS[0]), ZOOM_PASOS[-1])


def preparar_zoom(ventana, fichero: Path):
    """Zoom de la ventana con Ctrl + rueda y Ctrl + / − / 0, recordado en `fichero`.

    Solo Windows (Edge WebView2): el zoom se fija en el control con ZoomFactor, que es
    un zoom de navegador de verdad (los mapas y gráficos siguen respondiendo bien al
    ratón, al contrario que un zoom por CSS). En Mac queda el gesto del trackpad."""
    if sys.platform != 'win32':
        return
    estado = {'factor': leer_zoom(fichero), 'enganchado': False}

    def guardar(factor):
        estado['factor'] = factor
        try:
            fichero.parent.mkdir(parents=True, exist_ok=True)
            fichero.write_text(f'{factor:.3f}', encoding='utf-8')
        except Exception:                        # noqa: BLE001
            pass

    def en_la_ventana(funcion):
        """Ejecuta `funcion(control WebView2)` en el hilo de la ventana."""
        from System import Func, Type            # pythonnet, lo carga pywebview
        form = ventana.native
        if form is None or getattr(form, 'browser', None) is None:
            return
        form.Invoke(Func[Type](lambda: funcion(form.browser.webview)))

    def fijar(factor):
        def _fijar(control):
            control.ZoomFactor = factor
        try:
            en_la_ventana(_fijar)
        except Exception as exc:                 # noqa: BLE001
            print(f'  (zoom: {type(exc).__name__}: {exc})', flush=True)
            return
        # Cuando lo cambia el programa, WebView2 no avisa (ZoomFactorChanged solo salta
        # con la rueda): se guarda aquí.
        guardar(factor)

    def actual():
        leido = {}

        def _leer(control):
            leido['z'] = float(control.ZoomFactor)
        try:
            en_la_ventana(_leer)
        except Exception:                        # noqa: BLE001
            pass
        return leido.get('z', estado['factor'])

    def zoom_ventana(paso):
        """Lo llama la página con Ctrl + / Ctrl − / Ctrl 0."""
        fijar(siguiente_zoom(actual(), int(paso)))

    def al_cargar():
        if not estado['enganchado']:
            estado['enganchado'] = True

            def _enganchar(control):
                # Ctrl + rueda: guardar el zoom para la próxima vez.
                control.ZoomFactorChanged += (
                    lambda _sender, _args: guardar(float(control.ZoomFactor)))
            try:
                en_la_ventana(_enganchar)
            except Exception as exc:             # noqa: BLE001
                print(f'  (zoom: {type(exc).__name__}: {exc})', flush=True)
            if abs(estado['factor'] - 1.0) > 1e-3:
                fijar(estado['factor'])
        try:
            ventana.evaluate_js(TECLAS_ZOOM_JS)
        except Exception:                        # noqa: BLE001
            pass

    ventana.expose(zoom_ventana)
    ventana.events.loaded += al_cargar


def abrir_en_ventana(url: str, port: int, out: Path, probar: bool = False, buscar=None):
    """La app en su propia ventana (pywebview). La ventana se abre enseguida, con la
    página de carga; después, si se da `buscar`, se buscan versiones nuevas de la app
    (`buscar(mostrar)` dice en la página lo que va pasando y devuelve lo que dice el
    actualizador), luego se arranca el servidor y se carga la app. Si de la búsqueda sale
    algo que la persona tiene que hacer, se le enseña en un cuadro (27-11-2026: sin
    consola, en Mac, era lo único que no se veía).

    Devuelve (estado, proceso del servidor, si la ventana se ocupó de buscar, puerto):

    - ('cerrada', None, …): la persona ha cerrado la ventana (y con ella, el servidor).
    - ('sin_ventana', hijo o None, …): no se puede abrir la ventana; si el servidor ya
      está en marcha, se devuelve para seguir con él en el navegador.
    - con probar: ('ok' | 'no_arranca' | 'no_se_ve', None, …).
    El puerto es el del servidor: el elegido al principio puede haberlo cogido otro
    programa mientras se buscaba la versión nueva."""
    if sys.platform == 'win32' and not hay_motor_de_edge():
        print('  (sin ventana propia: falta el motor de Edge, WebView2 Runtime; se puede '
              'instalar desde https://developer.microsoft.com/microsoft-edge/webview2/)',
              flush=True)
        return 'sin_ventana', None, False, port
    try:
        import webview
    except Exception as exc:                     # noqa: BLE001
        print(f'  (sin ventana propia: {type(exc).__name__}: {exc})', flush=True)
        return 'sin_ventana', None, False, port

    # Lo que comparten este hilo y el de la ventana (cargar), siempre con el cerrojo:
    # 'cerrada' lo pone este hilo cuando la ventana se cierra; 'buscado' y 'hijo', cargar
    # al empezar la búsqueda y al arrancar el servidor. Así, una ventana cerrada antes de
    # tiempo no deja ni una búsqueda que nadie espera ni un servidor sin nadie.
    resultado = {'estado': 'cerrada', 'hijo': None, 'cerrada': False, 'buscado': False,
                 'r': None, 'puerto': port}
    cerrojo = threading.Lock()
    fin_busqueda = threading.Event()

    def mostrar(ventana, texto):
        if resultado['cerrada']:
            return
        with contextlib.suppress(Exception):
            ventana.load_html(cargando(texto))

    def busqueda(ventana):
        try:
            resultado['r'] = buscar(lambda texto: mostrar(ventana, texto))
        except Exception as exc:                 # noqa: BLE001  (la app abre igual)
            print(f'  (no se ha podido buscar la versión nueva: {exc})', flush=True)
        finally:
            fin_busqueda.set()

    def cargar(ventana):
        r = None
        if buscar is not None:
            with cerrojo:
                if resultado['cerrada']:
                    return
                resultado['buscado'] = True
            # La búsqueda (descargas, la comprobación de la versión nueva: hasta minutos)
            # va en un hilo aparte que no retiene el programa: pywebview espera a este
            # hilo al cerrarse, y si se cierra la ventana a medias tiene que terminar ya.
            threading.Thread(target=busqueda, args=(ventana,), daemon=True).start()
            while not fin_busqueda.wait(0.3):
                if resultado['cerrada']:
                    return
            r = resultado['r']
            if r and r.get('estado') == 'actualizado':
                mostrar(ventana, f'App actualizada a la versión {r.get("version")}. '
                                 'Abriendo la app…')
            elif r is not None:
                mostrar(ventana, 'Abriendo la app…')
        with cerrojo:
            if resultado['cerrada']:
                return
            puerto = resultado['puerto'] = free_port(port)
            hijo = resultado['hijo'] = lanzar_servidor(puerto)
        if not esperar_servidor(puerto, hijo):
            if resultado['cerrada']:              # se cerró la ventana: no hay que avisar
                return
            resultado['estado'] = 'no_arranca'
            ventana.load_html(no_arranca())
            if probar:
                ventana.destroy()
            return
        ventana.load_url(f'http://localhost:{puerto}')
        aviso = aviso_de_la_actualizacion(r)
        if aviso:
            print(f'  Aviso en la ventana: {aviso[0]}', flush=True)
        if not probar:
            if aviso and not resultado['cerrada']:
                # En otro hilo que no retiene el programa: en Mac y Linux el cuadro espera
                # sin fin si la ventana se cierra antes de que salga.
                threading.Thread(target=mostrar_aviso, args=(ventana, aviso),
                                 daemon=True).start()
            return
        # Prueba: la app tiene que verse en la ventana (lo que escribe app.py, no solo
        # el servidor).
        resultado['estado'] = 'no_se_ve'
        fin = time.time() + 150
        while time.time() < fin:
            time.sleep(2)
            try:
                texto = ventana.evaluate_js(
                    'document.body ? document.body.innerText : ""') or ''
            except Exception:                    # noqa: BLE001
                texto = ''
            # El Inicio (título) o, en versiones de antes, el panel lateral.
            if 'Encuentra los borreguiles' in texto or 'Área de estudio' in texto:
                resultado['estado'] = 'ok'
                break
        ventana.destroy()

    def cerrar():
        with cerrojo:
            resultado['cerrada'] = True
            return resultado['hijo'], resultado['buscado']

    webview.settings['ALLOW_DOWNLOADS'] = True               # botones de descarga
    webview.settings['OPEN_EXTERNAL_LINKS_IN_BROWSER'] = True  # GitHub, formulario…
    primera = 'Buscando una versión nueva de la app…' if buscar else 'Abriendo la app…'
    ventana = webview.create_window(TITULO_VENTANA, html=cargando(primera), width=1400,
                                    height=900, min_size=(900, 600),
                                    background_color='#221F20',   # sin destello blanco
                                    maximized=not probar, text_select=True,
                                    zoomable=True)   # Ctrl + rueda (pywebview lo apaga)
    preparar_zoom(ventana, out / '.ventana' / 'zoom.txt')
    try:
        # private_mode=False y una carpeta propia: la ventana recuerda lo que el
        # navegador recordaría (preferencias de Streamlit) entre sesiones. Los botones
        # de los cuadros, en español (en Mac y Linux los pone pywebview, en inglés).
        webview.start(cargar, (ventana,), private_mode=False,
                      storage_path=str(out / '.ventana'),
                      localization={'global.ok': 'Aceptar', 'global.cancel': 'Cancelar'})
    except KeyboardInterrupt:
        pass
    except Exception as exc:                     # noqa: BLE001
        print(f'  (no se ha podido abrir la ventana: {type(exc).__name__}: {exc})',
              flush=True)
        hijo, buscado = cerrar()
        if buscado:
            # Se sigue en el navegador, que elige el código a usar: antes, que acabe la
            # búsqueda que empezó la ventana (podría cambiarlo o quitarlo).
            fin_busqueda.wait()
        return 'sin_ventana', hijo, buscado, resultado['puerto']
    hijo, buscado = cerrar()
    parar_servidor(hijo)
    return resultado['estado'], None, buscado, resultado['puerto']


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


def codigo_de_la_app(res, out):
    """El app.py que hay que ejecutar: el del código que trae el instalador o, si se ha
    descargado una versión más nueva y está completa, el de esa (en la carpeta del
    usuario). Deja que sus módulos hermanos se importen de esa misma carpeta."""
    codigo, descargada = actualizador.carpeta_a_usar(res, out)
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
    return app_path


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
    servidor = '--servidor' in sys.argv
    sin_navegador = os.environ.get('BORREGUIL_NO_BROWSER', '').lower() in ('1', 'true',
                                                                              'yes')
    probar = '--probar-ventana' in sys.argv
    con_ventana = not servidor and (probar or not ('--navegador' in sys.argv
                                                   or sin_navegador))

    # Versiones nuevas del código: con la ventana, las busca ella, después de abrirse
    # (así se ve qué pasa mientras tanto); sin ella, ahora, en la consola. El servidor
    # (--servidor) no busca: lo ha hecho quien lo lanza.
    buscado = False
    if not servidor and not con_ventana:
        buscar_version_nueva(res, out)
        buscado = True

    # Carpeta escribible para lo que genere el usuario (modelos, descargas).
    os.environ['BORREGUIL_DATA_DIR'] = str(out)
    try:
        os.chdir(out)          # las salidas relativas caen en la carpeta del usuario
    except Exception:
        pass

    # Configuración de Streamlit por entorno (evita depender de .streamlit/config.toml
    # y del asistente de primer arranque que pide un email).
    port = (int(os.environ['BORREGUIL_PUERTO'])
            if servidor and os.environ.get('BORREGUIL_PUERTO') else free_port())
    os.environ.setdefault('STREAMLIT_SERVER_HEADLESS', 'true')          # no abre él el navegador
    os.environ.setdefault('STREAMLIT_BROWSER_GATHER_USAGE_STATS', 'false')
    os.environ.setdefault('STREAMLIT_SERVER_PORT', str(port))
    os.environ.setdefault('STREAMLIT_SERVER_ADDRESS', 'localhost')
    os.environ.setdefault('STREAMLIT_SERVER_FILE_WATCHER_TYPE', 'none')  # innecesario y frágil al compilar
    os.environ.setdefault('STREAMLIT_SERVER_MAX_UPLOAD_SIZE', '50')
    # El tema (oscuro) lo pone app.py: estas variables no llegaban a Streamlit sin la
    # orden «streamlit run», y app.py se actualiza sola.

    url = f'http://localhost:{port}'
    if servidor:                          # el proceso que lanza la ventana
        arrancar_streamlit(codigo_de_la_app(res, out), port)
        return

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
        sin_buscar = '--sin-actualizar' in sys.argv or not actualizador.url_manifiesto()
        estado, hijo, buscado, port = abrir_en_ventana(
            url, port, out, probar=probar,
            buscar=None if sin_buscar else (
                lambda mostrar: buscar_version_nueva(res, out, mostrar)))
        url = f'http://localhost:{port}'
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

    if not buscado:                      # la ventana no llegó a buscar
        buscar_version_nueva(res, out)
    app_path = codigo_de_la_app(res, out)
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
