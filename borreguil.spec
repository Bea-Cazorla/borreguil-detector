# -*- mode: python ; coding: utf-8 -*-
"""
borreguil.spec — Receta de PyInstaller para la versión STANDALONE.

Genera una carpeta autocontenida (modo onedir) con el ejecutable `Borreguil`.
Se compila por separado en cada sistema operativo (PyInstaller no puede compilar
para otro SO distinto del que lo ejecuta); de eso se encarga GitHub Actions.

Uso manual:
    pyinstaller borreguil.spec --noconfirm
"""
import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files, copy_metadata

# El .spec se ejecuta con exec(), así que __file__ no siempre está definido:
# SPECPATH lo aporta PyInstaller.
APP_DIR = Path(globals().get('SPECPATH', os.getcwd())).resolve()
REPO_DIR = APP_DIR.parent

datas, binaries, hiddenimports = [], [], []


def add_package(name, with_metadata=True):
    """Incluye datos/binarios/submódulos de un paquete. Si no está instalado, se
    avisa y se continúa: así el mismo .spec sirve aunque falte una dependencia
    opcional (p. ej. rasterio solo hace falta para el backend MPC)."""
    global datas, binaries, hiddenimports
    try:
        d, b, h = collect_all(name)
        datas += d
        binaries += b
        # Las baterías de tests de numpy/scipy/pandas/sklearn suman cientos de MB y
        # no se usan nunca. Se filtran AQUÍ y no con `excludes`, porque excluir un
        # módulo que collect_all ya ha declarado como hidden import provoca errores
        # "Hidden import not found" en la compilación.
        hiddenimports += [m for m in h
                          if '.tests' not in m and '.test_' not in m
                          and not m.endswith('.conftest')]
        if with_metadata:
            try:
                datas += copy_metadata(name)
            except Exception:
                pass
        print(f'[spec] incluido: {name}')
    except Exception as exc:                     # noqa: BLE001
        print(f'[spec] OMITIDO (no instalado): {name} -> {exc}')


# --- Núcleo de la interfaz -------------------------------------------------
# Streamlit necesita sus ficheros estáticos y sus metadatos (comprueba su propia
# versión en tiempo de ejecución mediante importlib.metadata).
add_package('streamlit')
datas += collect_data_files('streamlit', include_py_files=True)
for meta in ('streamlit', 'altair', 'pandas', 'numpy', 'pyarrow', 'packaging',
             'click', 'tornado', 'watchdog', 'gitpython', 'pydeck', 'protobuf'):
    try:
        datas += copy_metadata(meta)
    except Exception:
        pass

# --- Visualización y datos --------------------------------------------------
for pkg in ('altair', 'folium', 'streamlit_folium', 'branca', 'jinja2', 'pydeck'):
    add_package(pkg)

# --- Geoespacial (pyproj/rasterio/fiona traen datos PROJ y GDAL) ------------
for pkg in ('pyproj', 'geopandas', 'shapely', 'rasterio', 'fiona', 'pyogrio'):
    add_package(pkg)

# --- Ciencia de datos -------------------------------------------------------
for pkg in ('sklearn', 'scipy', 'skimage', 'joblib', 'numpy', 'pandas'):
    add_package(pkg)

# --- Servicios remotos ------------------------------------------------------
for pkg in ('ee', 'google', 'google_auth_httplib2', 'googleapiclient',
            'pystac_client', 'planetary_computer', 'requests', 'certifi',
            'urllib3', 'charset_normalizer', 'docx', 'openpyxl', 'PIL'):
    add_package(pkg)

# Submódulos que se cargan de forma dinámica y el análisis estático no ve.
hiddenimports += [
    'streamlit.runtime.scriptrunner.magic_funcs',
    'streamlit.web.bootstrap',
    'sklearn.ensemble._forest',
    'sklearn.tree._partitioner',
    'sklearn.utils._typedefs',
    'sklearn.neighbors._partition_nodes',
    'scipy.spatial.transform._rotation_groups',
    'pandas._libs.tslibs.base',
    'PIL._tkinter_finder',
    'encodings.idna',
    # numpy.f2py arrastra estos módulos de la biblioteca estándar por importación
    # perezosa; sin ellos el arranque falla con ModuleNotFoundError.
    'fileinput',
    'numpy.f2py',
]

# --- Ficheros propios de la aplicación --------------------------------------
# Los módulos .py van como DATOS (no como imports) porque Streamlit ejecuta
# app.py como un script; launcher.py añade esta carpeta a sys.path para que los
# `import borreguil_pipeline`, `import i18n`, etc. funcionen.
for name in ('app.py', 'borreguil_pipeline.py', 'point_profile.py', 'i18n.py',
             'gee_backend.py', 'random_points.py', 'study_area.py'):
    p = APP_DIR / name
    if p.exists():
        datas.append((str(p), '.'))

# Modelos entrenados y capas auxiliares que la app espera encontrar a su lado.
for pattern in ('*.joblib', 'lagunas_sierra_nevada.kml'):
    for p in APP_DIR.glob(pattern):
        # Los ficheros «._algo» son restos de macOS (AppleDouble), no modelos:
        # empaquetarlos los haria aparecer en el desplegable del ejecutable.
        if p.name.startswith('.'):
            continue
        datas.append((str(p), '.'))

# Datos de referencia para las gráficas comparativas (opcional pero recomendable).
for cand in (APP_DIR / 'classification_v5.csv', REPO_DIR / 'classification_v5.csv'):
    if cand.exists():
        datas.append((str(cand), '.'))
        break

# Documentación que viaja con el programa.
for name in ('MANUAL.md', 'TUTORIAL.md', 'LICENSE', 'CITATION.cff'):
    for base in (APP_DIR, REPO_DIR):
        p = base / name
        if p.exists():
            datas.append((str(p), '.'))
            break

# Paquetes de terceros que la app no usa. NO excluir aquí submódulos internos de
# numpy/scipy (p. ej. numpy.f2py): numpy los importa de forma perezosa desde
# `from numpy import *` que hace scipy, y su ausencia aborta la app al arrancar.
excludes = [
    'tkinter', 'matplotlib', 'notebook', 'jupyter', 'IPython', 'ipywidgets',
    'pytest', 'sphinx', 'torch', 'tensorflow', 'numba', 'dask', 'cupy',
]

block_cipher = None

a = Analysis(
    [str(APP_DIR / 'launcher.py')],
    pathex=[str(APP_DIR)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Borreguil',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    # console=True es IMPORTANTE: el pipeline escribe su progreso por consola y,
    # en modo ventana, sys.stdout sería None y los print() abortarían la ejecución.
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(APP_DIR / 'icon.ico') if (APP_DIR / 'icon.ico').exists() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='Borreguil',
)
