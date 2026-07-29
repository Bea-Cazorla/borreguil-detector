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

Ejecutar en desarrollo (sin compilar):  python launcher.py
"""
import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

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
    def _pipeline():
        import borreguil_pipeline as bp
        bp._imports()
    comprobar('importar el pipeline y sus dependencias', _pipeline)
    comprobar('módulos de la interfaz (point_profile, i18n)',
              lambda: (__import__('point_profile'), __import__('i18n')))
    comprobar('streamlit', lambda: __import__('streamlit'))
    comprobar('scipy.ndimage', lambda: __import__('scipy.ndimage', fromlist=['ndimage']))
    comprobar('sklearn', lambda: __import__('sklearn.ensemble', fromlist=['ensemble']))
    comprobar('pyproj', lambda: __import__('pyproj'))

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


def main():
    res = resource_dir()

    if '--selftest' in sys.argv:
        sys.exit(selftest(res))

    out = data_dir()

    # El script de la app y sus módulos hermanos viajan en la carpeta de recursos.
    app_path = res / 'app.py'
    if not app_path.exists():
        sys.exit(f'ERROR: no se encuentra app.py en {res}')
    sys.path.insert(0, str(res))

    # Carpeta escribible para lo que genere el usuario (modelos, descargas).
    os.environ['BORREGUIL_DATA_DIR'] = str(out)
    try:
        os.chdir(out)          # las salidas relativas caen en la carpeta del usuario
    except Exception:
        pass

    # Configuración de Streamlit por entorno (evita depender de .streamlit/config.toml
    # y del asistente de primer arranque que pide un email).
    port = free_port()
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
    print('=' * 64)
    print(f'  {APP_NAME} — detección de borreguiles / wet-meadow finder')
    print('=' * 64)
    print(f'  Abriendo el navegador en:  {url}')
    print(f'  Tus modelos y salidas:     {out}')
    print()
    print('  Deja ESTA ventana abierta mientras uses el programa.')
    print('  Para cerrarlo: cierra esta ventana o pulsa Ctrl+C.')
    print('=' * 64, flush=True)

    if os.environ.get('BORREGUIL_NO_BROWSER', '').lower() not in ('1', 'true', 'yes'):
        open_browser_when_ready(url, port)

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


if __name__ == '__main__':
    main()
