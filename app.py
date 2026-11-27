"""
app.py — Interfaz web Streamlit para el pipeline de identificación de borreguiles.

Copyright (C) 2026 los autores de este programa.
Este programa es software libre bajo la Licencia Pública General GNU, versión 3
o posterior. Se distribuye SIN NINGUNA GARANTÍA. Ver el fichero LICENSE o
<https://www.gnu.org/licenses/>.

USO:
    pip install streamlit folium streamlit-folium openpyxl rasterio scikit-learn \
                scikit-image scipy pyproj shapely planetary-computer pystac-client \
                geopandas requests pillow joblib python-docx
    streamlit run app.py

La app permite a un experto de cualquier zona protegida:
1. Subir un KML/Shapefile/GeoJSON con puntos candidatos
2. Opcionalmente subir un fichero de verdad-terreno
3. Lanzar el pipeline (descarga ESRI, cauces de OSM, DEM, Sentinel-2; clasifica con RF)
4. Inspeccionar resultados en un mapa interactivo Folium
5. Descargar Excel + CSV + mapa HTML
"""
import io, os, re, shutil, tempfile, sys, threading, time, unicodedata
# Windows: la consola por defecto es cp1252 y los print() del pipeline llevan
# caracteres Unicode (→, emojis); sin esto, escribir en stdout lanza
# UnicodeEncodeError y aborta la ejecución. Forzamos UTF-8 tolerante a fallos.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
from pathlib import Path
import streamlit as st
import pandas as pd
from collections import Counter

# Inject our pipeline
APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))
import borreguil_pipeline as bp
bp._imports()  # eager-import everything (Streamlit shows spinners during this)

import folium
from streamlit_folium import st_folium
import point_profile as pp
import i18n
import modelos_remotos as mrm
import registros

# Carpeta ESCRIBIBLE del usuario. En la versión standalone (ejecutable) APP_DIR es
# una carpeta temporal de solo lectura que se borra al salir, así que lo que el
# usuario guarda (modelos entrenados) debe ir aquí. Sin la variable de entorno
# —modo desarrollo— coincide con APP_DIR y todo sigue igual que antes.
DATA_DIR = Path(os.environ.get('BORREGUIL_DATA_DIR') or APP_DIR)
try:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    DATA_DIR = APP_DIR

# Modelos públicos que la app descarga de GitHub al abrirse (modelos_remotos.py), para
# tener los nuevos sin reinstalar. Pasan delante de los que trae la app.
DESCARGADOS = DATA_DIR / mrm.CARPETA


# Ajustes que conviene recordar de una sesión a otra (el proyecto de Earth Engine,
# para no escribirlo cada vez). Solo preferencias: nunca claves ni contraseñas.
_AJUSTES = DATA_DIR / 'ajustes.json'


def leer_ajuste(clave, defecto=''):
    try:
        import json
        return json.loads(_AJUSTES.read_text(encoding='utf-8')).get(clave, defecto)
    except Exception:
        return defecto


def guardar_ajuste(clave, valor):
    try:
        import json
        try:
            ajustes = json.loads(_AJUSTES.read_text(encoding='utf-8'))
        except Exception:
            ajustes = {}
        if ajustes.get(clave) != valor:
            ajustes[clave] = valor
            _AJUSTES.write_text(json.dumps(ajustes, ensure_ascii=False, indent=1),
                                encoding='utf-8')
    except Exception:
        pass                    # no poder recordar una preferencia no detiene nada


def carpetas_de_modelos():
    """Dónde se buscan los modelos, por orden: los que guarda quien usa la app, los
    descargados de GitHub y los que trae la app."""
    try:
        propia = DATA_DIR.resolve() != APP_DIR.resolve()
    except OSError:
        propia = False
    return ([DATA_DIR] if propia else []) + [DESCARGADOS, APP_DIR]


def model_path(name):
    """Ruta de un modelo por nombre, en el orden de carpetas_de_modelos()."""
    for d in carpetas_de_modelos():
        p = d / name
        if p.exists():
            return p
    return APP_DIR / name


def modelo_sin_radar(ruta):
    """Versión SIN radar de un modelo incluido en la app o descargado de GitHub (su
    carpeta sin_radar/), o None.

    Con Microsoft Planetary Computer (o «Saltar Sentinel-2») el radar no se lee. El modelo con
    radar, si le falta, acierta bastante menos (AUC 0,793 frente a 0,847) que uno
    entrenado desde el principio sin él (0,833). Solo los modelos que vienen con la app
    o de GitHub tienen esta versión; los que guarda quien usa la app, no."""
    if not ruta:
        return None
    p = Path(ruta)
    try:
        if p.resolve().parent not in (APP_DIR.resolve(), DESCARGADOS.resolve()):
            return None
    except OSError:
        return None
    return bp.ruta_sin_radar(p)


def modelo_segun_radar(ruta, hay_radar):
    """El modelo que se usa de verdad: el elegido o, si no se ha leído el radar, su
    versión sin radar (si la tiene)."""
    if hay_radar:
        return ruta
    q = modelo_sin_radar(ruta)
    return str(q) if q is not None else ruta


FUENTES_VARIABLES = ('img', 'topo', 's2', 's1', 'cir')
_TOPO_VARIABLES = ('elev_dem_m', 'slope_deg', 'curvature', 'twi', 'aspect_north',
                   'aspect_east')


def fuente_de_variable(f):
    """De qué fuente sale una variable del modelo: imagen aérea, relieve, Óptico
    Sentinel-2, Radar Sentinel-1 o infrarrojo (CIR)."""
    if f in bp.S1_FEATS:
        return 's1'
    if f in bp.CIR_FEATS:
        return 'cir'
    if f in _TOPO_VARIABLES:
        return 'topo'
    if f in bp.FEATURES_RF[:12]:
        return 'img'
    return 's2'


@st.cache_data(show_spinner=False)
def _variables_modelo(ruta, mtime):
    """Las variables de un modelo, en su orden; `mtime` renueva la caché."""
    try:
        return list(bp.load_model_bundle(ruta).get('features', []))
    except Exception:                                   # noqa: BLE001
        return []


@st.cache_data(show_spinner=False)
def _auc_modelo(ruta, mtime):
    """AUC (validación por cuencas) guardado en un modelo; `mtime` renueva la caché."""
    try:
        v = bp.load_model_bundle(ruta).get('auc_groupkfold')
        return float(v) if isinstance(v, (int, float)) else None
    except Exception:
        return None


def _es_sin_radar(info):
    """¿Ha dado la probabilidad la versión sin radar del modelo?"""
    info = info or {}
    return 'sin radar' in (str(info.get('training_zone', '')) + ' '
                           + str(info.get('zona_partida', ''))).lower()


@st.cache_resource(show_spinner=False)
def tabla_entrenamiento():
    """Los puntos con los que se entrenó el modelo de Sierra Nevada (796, con sus
    variables), o None si la tabla no está. Sirven para verlos en el mapa y para
    reentrenar el modelo sumando los puntos de quien usa la app. No se modifican."""
    p = APP_DIR / bp.TABLA_ENTRENAMIENTO
    try:
        return bp.leer_tabla_entrenamiento(p) if p.exists() else None
    except Exception:
        return None


def _es_modelo_real(p):
    """Descarta ficheros que parecen modelos pero no lo son.

    Al copiar el proyecto desde macOS aparecen ficheros «AppleDouble» con el
    prefijo `._` (p. ej. `._rf_sierra_nevada.joblib`, de 4 KB). Coinciden con
    `*.joblib` y llegaban al desplegable, de modo que elegir uno abortaba la
    ejecución al intentar cargarlo. También se ignoran otros ocultos y los
    ficheros demasiado pequeños para ser un Random Forest.
    """
    if p.name.startswith('.'):
        return False
    try:
        return p.stat().st_size > 100_000
    except OSError:
        return False


def list_models():
    """Modelos .joblib disponibles (los del usuario, los descargados de GitHub y los
    incluidos; SN primero). Son nombres: model_path() dice de qué carpeta sale cada uno."""
    found = set()
    for d in carpetas_de_modelos():
        if d.is_dir():
            found.update(p.name for p in d.glob('*.joblib') if _es_modelo_real(p))
    return sorted(found, key=lambda n: (n != 'rf_sierra_nevada.joblib', n))


@st.cache_resource(show_spinner='Buscando modelos nuevos en GitHub… / '
                                 'Checking GitHub for new models…')
def modelos_de_github():
    """Una vez por arranque de la app: descarga de GitHub los modelos públicos nuevos o
    cambiados (modelos_remotos.py). Sin conexión, sigue con los que haya."""
    return mrm.actualizar(DESCARGADOS, incluidos=APP_DIR)


def texto_modelos_github(res):
    """Una línea sobre los modelos de GitHub para el panel ('' si no se buscan)."""
    estado = (res or {}).get('estado')
    if estado == 'al_dia':
        return i18n.pick('Modelos al día con GitHub.', 'Models up to date with GitHub.')
    if estado == 'descargados':
        partes = ([i18n.pick(f'{n} (nuevo)', f'{n} (new)') for n in res['nuevos']]
                  + [i18n.pick(f'{n} (actualizado)', f'{n} (updated)')
                     for n in res['actualizados']])
        return i18n.pick('Modelos descargados de GitHub al abrir la app: ',
                         'Models downloaded from GitHub when the app opened: ') + \
            ', '.join(partes) + '.'
    if estado == 'sin_conexion':
        return i18n.pick('Sin conexión con GitHub: se usan los modelos que ya tienes.',
                         'No connection to GitHub: the models you already have are used.')
    if estado == 'error':
        motivo = (res.get('fallos') or [''])[0]
        return i18n.pick(f'No se pudieron traer los modelos de GitHub ({motivo}). Se usan '
                         'los que ya tienes.',
                         f'The models could not be fetched from GitHub ({motivo}). The '
                         'ones you already have are used.')
    return ''


def area_to_filename(area_name: str) -> str:
    """Convierte el nombre de un área protegida a un nombre de fichero seguro.

    Ejemplos:
        'Parque Nacional de Sierra Nevada'  → 'rf_parque_nacional_de_sierra_nevada'
        'Aigüestortes i Estany de Sant Maurici' → 'rf_aiguëstortes_i_estany…'
    Criterio: se normaliza Unicode (NFKD → ASCII), se reemplaza cualquier
    carácter no alfanumérico por '_', se pone en minúsculas y se añade el
    prefijo 'rf_'.
    """
    if not area_name:
        return 'rf_mi_zona'
    nfkd = unicodedata.normalize('NFKD', area_name)
    ascii_str = nfkd.encode('ascii', 'ignore').decode('ascii')
    safe = re.sub(r'[^a-zA-Z0-9]+', '_', ascii_str).strip('_').lower()
    return f'rf_{safe}' if safe else 'rf_mi_zona'


def get_secret(key, default=None):
    """Lee un secreto de forma robusta al directorio de trabajo y sin provocar el
    warning 'No secrets found'. Orden: (1) parseo directo del secrets.toml que vive
    JUNTO a la app o en ~/.streamlit (funciona aunque se lance desde otro cwd);
    (2) st.secrets, solo si Streamlit tiene un archivo en SUS rutas (caso Streamlit
    Cloud), evitando así el warning; (3) variable de entorno."""
    # (1) Parseo directo del archivo (junto a la app o en ~) — robusto al cwd
    for p in (os.path.join(os.path.dirname(__file__), '.streamlit', 'secrets.toml'),
              os.path.expanduser('~/.streamlit/secrets.toml')):
        try:
            if os.path.exists(p):
                import tomllib
                with open(p, 'rb') as fh:
                    data = tomllib.load(fh)
                if key in data:
                    return data[key]
        except Exception:
            pass
    # (2) st.secrets (Streamlit Cloud) — solo si existe un secrets.toml en las rutas
    # que Streamlit consulta; así nunca se dispara el warning 'No secrets found'.
    st_paths = (os.path.join(os.getcwd(), '.streamlit', 'secrets.toml'),
                os.path.expanduser('~/.streamlit/secrets.toml'))
    if any(os.path.exists(p) for p in st_paths):
        try:
            if key in st.secrets:
                return st.secrets[key]
        except Exception:
            pass
    # (3) Variable de entorno
    return os.environ.get(key, default)


# ── Manual y sugerencias (24-11, como en MonitorEO) ────────────────────────────
# La wiki del repositorio, en español e inglés, y sus plantillas de issues (error y
# sugerencia, en .github/ISSUE_TEMPLATE). Para quien no tiene cuenta de GitHub, un
# formulario anónimo propio de esta app (25-11-2026, la autora: «quiero que este
# formulario de sugerencias también aparezca en la app», «que sea de esta app»), hecho
# como el de MonitorEO: qué quieres hacer, en qué parte de la app, versión, ordenador,
# descripción, enlace a pantallazos y texto de la consola. No recoge correos y lo puede
# rellenar cualquiera con el enlace. Se abre con la versión ya puesta.
REPO_URL = 'https://github.com/monitoreo-obsnev/borreguil-detector'
WIKI_URL = REPO_URL + '/wiki'
FORMULARIO_URL = registros.FORMULARIO
FORMULARIO_CAMPO_VERSION = registros.CAMPOS['version']


def version_app():
    """La versión de CITATION.cff ('' si no se puede leer)."""
    import re
    try:
        texto = (APP_DIR / 'CITATION.cff').read_text(encoding='utf-8')
    except OSError:
        return ''
    m = re.search(r'^version:\s*"?([^"\s#]+)', texto, re.M)
    return m.group(1) if m else ''


def fecha_app():
    """La fecha de la versión (date-released de CITATION.cff) como 26-11-2026 ('' si no)."""
    import re
    try:
        texto = (APP_DIR / 'CITATION.cff').read_text(encoding='utf-8')
    except OSError:
        return ''
    m = re.search(r'^date-released:\s*"?(\d{4})-(\d{2})-(\d{2})', texto, re.M)
    return f'{m.group(3)}-{m.group(2)}-{m.group(1)}' if m else ''


# Borreguil sí o no, siempre verde o rojo (la autora, 26-11-2026: «rojo y verde siempre
# que sea borreguil sí o no»), y la probabilidad en grande («pon el % más grande»).
COLOR_SI, COLOR_NO = '#43A047', '#E53935'


def _es_borreguil(r):
    """True si el punto es borreguil: por tu revisión o, sin ella, por el modelo."""
    v = bp._txt(r.get('Borreguil')).lower()
    if v in ('si', 'no'):
        return v == 'si'
    return pp.is_borreguil_decision(r.get('decision'))


def bloque_probabilidad(r, umbral, pie=''):
    """El resultado del punto y su probabilidad de borreguil, en grande, con la barra
    del corte (el umbral del análisis), como en la demo (Revisar.dc.html): el % en
    Montserrat de 68 px y, debajo de la barra, `pie` (la línea del tipo), todo con
    una raya debajo, que lo separa de las preguntas."""
    import html as _html
    si = _es_borreguil(r)
    color = COLOR_SI if si else COLOR_NO
    claro = '#7FD483' if si else '#F28B88'
    dec = _html.escape(i18n.tv_decision(r.get('decision') or '—'))
    cab = ('<div style="display:flex;align-items:center;gap:10px">'
           f'<span style="width:16px;height:16px;border-radius:50%;background:{color};'
           'display:inline-block;flex:none"></span><span style="font-size:16px;'
           f'font-weight:700;color:#F5F7F8">{dec}</span></div>')

    def caja(dentro):
        return ('<div style="display:flex;flex-direction:column;gap:8px;'
                'border-bottom:1px solid #403A3C;padding-bottom:16px">' + dentro + pie
                + '</div>')
    try:
        p = float(r.get('rf_proba'))
    except (TypeError, ValueError):
        return caja(cab)
    if p != p:
        return caja(cab)
    u = umbral if isinstance(umbral, (int, float)) and 0 < umbral < 1 else None
    barra = ''
    if u is not None:
        barra = (
            '<div style="position:relative;height:12px;border-radius:999px;'
            f'background:linear-gradient(90deg,#4A2424 0 {u * 100:.1f}%,'
            f'#1F3A27 {u * 100:.1f}% 100%)" aria-hidden="true">'
            f'<span style="position:absolute;left:{u * 100:.1f}%;top:-3px;width:2px;'
            'height:18px;background:#ECEFF1"></span>'
            f'<span style="position:absolute;left:calc({p * 100:.1f}% - 9px);top:-3px;'
            f'width:18px;height:18px;border-radius:50%;background:{color};'
            'border:3px solid #221F20;box-sizing:border-box"></span></div>'
            '<div style="display:flex;justify-content:space-between;font-size:11px;'
            'color:#98A2AB">'
            f'<span>{i18n.pick("0 · no borreguil", "0 · not borreguil")}</span>'
            f'<span>{i18n.pick("corte", "cut")} {_num(u)}</span>'
            f'<span>{i18n.pick("1 · borreguil", "1 · borreguil")}</span></div>')
    return caja(
        cab
        + '<span style="font-size:14px;font-weight:600;color:#CDD3D9;margin-top:4px">'
        + i18n.pick('Probabilidad de borreguil', 'Probability of borreguil') + '</span>'
        + '<div style="display:flex;flex-wrap:wrap;align-items:baseline;gap:4px 12px">'
        + '<span style="font-family:\'Montserrat\',Inter,sans-serif;font-size:68px;'
        + 'font-weight:700;line-height:1;letter-spacing:-0.02em;'
        + f'color:{claro}">{p * 100:.0f} %</span>'
        + f'<span style="font-size:14px;color:#AEB6BE">{_num(p)} '
        + i18n.pick('de 1', 'of 1') + '</span></div>' + barra)


# Pantalla de inicio: el Observatorio, la foto del borreguil y los logos del pie (en la
# carpeta imagenes/, que llega con la actualización; si falta, no se dibujan).
OBSNEV_URL = 'https://obsnev.es/ecosistemas-borreguiles/'
OBSNEV_URL_EN = 'https://obsnev.es/en/ecosistemas-borreguiles/'   # la misma página en inglés
OBSNEV_TITULO = ('Ecosistemas Borreguiles · Observatorio de Cambio Global de Sierra Nevada '
                 '(obsnev.es)')
OBSNEV_TITULO_EN = ('Borreguiles ecosystems · Sierra Nevada Global Change Observatory '
                    '(obsnev.es)')
IMAGENES = APP_DIR / 'imagenes'


@st.cache_data(show_spinner=False)
def _imagen_datos(nombre):
    """La imagen de imagenes/ como data: URI para meterla en HTML ('' si no está)."""
    import base64
    p = IMAGENES / nombre
    try:
        datos = p.read_bytes()
    except OSError:
        return ''
    tipo = 'jpeg' if p.suffix.lower() in ('.jpg', '.jpeg') else 'png'
    return f'data:image/{tipo};base64,' + base64.b64encode(datos).decode('ascii')


def foto_inicio():
    """La foto de los borreguiles de la ladera del Veleta, con su autoría (CC BY 4.0).

    Elegida por la autora el 27-11-2026 entre las de Wikimedia Commons («Borreguiles -
    Sierra Nevada - Spain 2.tif», de Bouke ten Cate), guardada a 1600 px de ancho. Antes,
    la del Cerro del Caballo (Granadaruralsur, CC BY-SA 4.0)."""
    src = _imagen_datos('borreguil_veleta.jpg')
    if not src:
        return ''
    alt = i18n.pick('Borreguil verde con pozas de agua y un arroyo en la ladera del Veleta, '
                    'Sierra Nevada, con niebla',
                    'Green alpine wet meadow with pools and a stream on the slopes of the '
                    'Veleta, Sierra Nevada, in the mist')
    pie = i18n.pick('Borreguiles en la ladera del Veleta, Sierra Nevada · Foto: Bouke ten '
                    'Cate, CC BY 4.0, Wikimedia Commons',
                    'Alpine wet meadows on the slopes of the Veleta, Sierra Nevada · Photo: '
                    'Bouke ten Cate, CC BY 4.0, Wikimedia Commons')
    return ('<style>.foto-inicio{position:relative;height:340px;border-radius:18px;'
            'overflow:hidden;border:1px solid #403A3C;margin:0}'
            '.foto-inicio img{width:100%;height:100%;object-fit:cover;display:block;'
            'object-position:center 60%}'
            '.foto-inicio figcaption{position:absolute;left:0;right:0;bottom:0;'
            'padding:28px 14px 10px;font-size:11px;color:#CDD3D9;'
            'background:linear-gradient(transparent,rgba(0,0,0,.72))}</style>'
            f'<figure class="foto-inicio"><img src="{src}" alt="{alt}">'
            f'<figcaption>{pie}</figcaption></figure>')


def pie_inicio(backend, gee_project):
    """El pie del Inicio: con qué calcula la app y la versión, y los logos de quién hace
    la app (Observatorio, UGR) y quién la financia (Biodiversa+ por PRESINMED;
    Ministerio, UE y AEI por EVEREST), todos sobre el mismo blanco."""
    import html as _html
    # Sin la línea de con qué calcula la app («Earth Engine con la cuenta compartida de
    # MonitorEO · te quedan … h»; la autora, 27-11-2026: «esto quítalo»): solo la versión.
    v, f = version_app(), fecha_app()
    version = ' · '.join(x for x in (i18n.pick(f'Versión {v}', f'Version {v}') if v else '',
                                     f) if x)
    hacen = [('logo_obsnev.png', 54, 'Observatorio de Cambio Global de Sierra Nevada'),
             ('logo_ugr.png', 68, 'Universidad de Granada')]
    financian = [
        ('logo_biodiversa.png', 40, 250,
         'Biodiversa+, European Biodiversity Partnership',
         i18n.pick('Proyecto PRESINMED (PCI 2025-163175), de la convocatoria '
                   'Biodiversa+NBS 2023.',
                   'PRESINMED project (PCI 2025-163175), Biodiversa+NBS 2023 call.')),
        ('logo_miciu_ue_aei.png', 40, 330,
         'Ministerio de Ciencia, Innovación y Universidades; Cofinanciado por la Unión '
         'Europea; Agencia Estatal de Investigación',
         i18n.pick('Proyecto EVEREST (PID2023-151939OB-I00), financiado por '
                   'MICIU/AEI/10.13039/501100011033 y por FEDER/UE.',
                   'EVEREST project (PID2023-151939OB-I00), funded by '
                   'MICIU/AEI/10.13039/501100011033 and by ERDF/EU.'))]
    izq = ''.join(f'<img src="{_imagen_datos(n)}" alt="{_html.escape(a)}" '
                  f'style="height:{h}px;width:auto;display:block">'
                  for n, h, a in hacen if _imagen_datos(n))
    der = ''.join(f'<div style="max-width:{w}px;display:flex;flex-direction:column;gap:6px">'
                  f'<img src="{_imagen_datos(n)}" alt="{_html.escape(a)}" '
                  f'style="height:{h}px;width:auto;display:block;align-self:flex-start">'
                  f'<span>{_html.escape(t)}</span></div>'
                  for n, h, w, a, t in financian if _imagen_datos(n))
    logos = ('' if not (izq or der) else
             f'<div class="pie-logos" aria-label="'
             + i18n.pick('Quién hace la app y quién la financia',
                         'Who makes the app and who funds it')
             + f'"><div class="grupo">{izq}</div><div class="grupo">{der}</div></div>')
    return ('<style>.pie-inicio{margin-top:40px;border-top:1px solid #403A3C}'
            '.pie-estado{display:flex;justify-content:flex-end;flex-wrap:wrap;gap:8px;'
            'background:#282526;padding:14px 32px;font-size:13px;color:#AEB6BE}'
            '.pie-estado .punto{display:inline-block;width:8px;height:8px;border-radius:50%;'
            'background:#7CC496;margin-right:8px;vertical-align:middle}'
            '.pie-estado .version{color:#CDD3D9}'
            '.pie-logos{display:flex;justify-content:space-between;align-items:center;'
            'flex-wrap:wrap;gap:20px 40px;background:#FFFFFF;padding:20px 32px}'
            '.pie-logos .grupo{display:flex;align-items:center;flex-wrap:wrap;gap:20px 40px}'
            '.pie-logos span{font-size:11px;line-height:1.35;color:#61585B}</style>'
            '<div class="pie-inicio"><div class="pie-estado">'
            f'<span class="version">{_html.escape(version)}</span></div>{logos}</div>')


def enlaces_sugerencias(version=None):
    """{'issue': una issue nueva de GitHub (elegir error o sugerencia) con la versión
    ya puesta, 'form': el formulario anónimo con la versión ya puesta ('' si no hay),
    'wiki': el manual}."""
    import urllib.parse
    version = version_app() if version is None else version
    issue = REPO_URL + '/issues/new/choose'
    if version:
        issue += '?' + urllib.parse.urlencode({'version': version})
    form = FORMULARIO_URL
    if form and version:
        form += '?' + urllib.parse.urlencode({'usp': 'pp_url',
                                              FORMULARIO_CAMPO_VERSION: version})
    return {'issue': issue, 'form': form, 'wiki': WIKI_URL}


# ── Registros (26-11-2026, la autora: «en las incidencias que suban los log de la app,
# tal cual en monitoreo está hecho en el asistente») ───────────────────────────────
# Desde la primera pasada, lo que sale por la consola se apunta también en registro.log
# de la carpeta Borreguil (registros.py). «Enviar los registros», en «Sugerir un cambio o
# informar de un error» y debajo de cada error de la app, enseña sus últimas líneas, sin
# lo que diría quién es, para leerlas, cambiarlas o borrarlas, y las envía con el
# formulario anónimo (en «Texto de la consola»), como el asistente de MonitorEO.
registros.empezar(version_app())


def _textos_registros():
    return registros.TEXTOS[i18n.get_lang()]


def _abrir_registros():
    """on_click de «Enviar los registros»: corre antes que el guion, así que vale también
    si el botón ya no se dibuja en la pasada siguiente (el error que lo llevaba)."""
    st.session_state['_abrir_registros'] = True
    st.session_state['registros_texto'] = registros.recoger(
        version_app(), ocultar=[leer_ajuste('gee_project', '')])
    for clave in ('registros_enviados', 'registros_fallo'):
        st.session_state.pop(clave, None)


def abrir_dialogo_registros():
    """La ventana de «Enviar los registros»: qué es, qué estabas haciendo (opcional), en
    qué parte de la app, los registros (para leerlos, cambiarlos o borrarlos) y Enviar."""
    t = _textos_registros()
    es = i18n.get_lang() != 'en'

    @st.dialog(t['boton'], width='large')
    def _dialogo():
        st.caption(t['ayuda'])
        partes = list(registros.PARTES)
        parte = st.selectbox(t['parte'], partes, index=partes.index('otra'),
                             format_func=lambda k: registros.PARTES[k][0 if es else 1],
                             key='registros_parte')
        descripcion = st.text_area(t['descripcion'], key='registros_descripcion', height=90)
        texto = st.text_area(t['registros'], key='registros_texto', height=340)
        if st.session_state.get('registros_enviados'):
            st.success(t['enviados'])
            return
        cancelar, enviar = st.columns(2)
        if cancelar.button(t['cancelar'], key='registros_cancelar', width='stretch'):
            st.rerun()                      # cierra la ventana sin enviar nada
        if enviar.button(t['enviar'], type='primary', key='registros_enviar', width='stretch'):
            if not (texto or '').strip():
                st.warning(t['vacio'])
                return
            try:
                with st.spinner(i18n.pick('Enviando…', 'Sending…')):
                    registros.enviar(texto, version_app(), descripcion, parte)
            except Exception as ex:         # sin conexión, el formulario no los coge…
                st.session_state['registros_fallo'] = str(ex)
            else:
                st.session_state['registros_enviados'] = True
                st.success(t['enviados'])
        if st.session_state.get('registros_fallo'):
            # st.warning y no st.error: st.error pondría otro «Enviar los registros»
            st.warning(st.session_state['registros_fallo'])

    _dialogo()


def _vigilar_errores():
    """Cada st.error se apunta en el registro (con la traza, si viene de un error de
    Python) y lleva debajo «Enviar los registros», como el botón junto a cada error del
    asistente de MonitorEO. Una vez por programa (después de i18n)."""
    original = st.error
    if getattr(original, '_registra', False):
        return

    def error(body='', *args, **kwargs):
        registros.anotar_error(body, sys.exc_info())
        salida = original(body, *args, **kwargs)
        import hashlib
        clave = 'registros_error_' + hashlib.md5(str(body).encode('utf-8')).hexdigest()[:12]
        try:
            st.button(_textos_registros()['boton'], key=clave, type='tertiary',
                      on_click=_abrir_registros)
        except Exception:     # el mismo error dos veces en una pasada, o dentro de un st.form
            pass
        return salida

    error._registra = True
    st.error = error


def mostrar_sugerencias(donde='lateral'):
    """El manual y cómo sugerir un cambio o informar de un error. DONDE distingue sus
    botones (el panel lateral y la documentación)."""
    e = enlaces_sugerencias()
    st.markdown(i18n.pick(
        f'📖 [Manual (wiki), en español e inglés]({e["wiki"]})\n\n'
        f'**Sugerir un cambio o informar de un error**:\n'
        f'- [En GitHub]({e["issue"]}): público, con una cuenta de GitHub; puedes pegar '
        'pantallazos y seguir la respuesta.'
        + (f'\n- [Formulario anónimo]({e["form"]}): sin cuenta, nombre ni correo, así que '
           'no te podrán contestar. La versión ya va puesta.'
           if e['form'] else ''),
        f'📖 [Manual (wiki), in Spanish and English]({e["wiki"]})\n\n'
        f'**Suggest a change or report an error**:\n'
        f'- [On GitHub]({e["issue"]}): public, with a GitHub account; you can paste '
        'screenshots and follow the answer.'
        + (f'\n- [Anonymous form]({e["form"]}): no account, no name and no email, so '
           'nobody can answer you. The version is already filled in.'
           if e['form'] else '')))
    if e['form']:       # los registros de este ordenador, con el formulario anónimo
        st.button('→ ' + _textos_registros()['menu'], key=f'registros_{donde}',
                  type='tertiary', on_click=_abrir_registros)
    # Sin el aviso «No pegues coordenadas…» (la autora, 27-11-2026: «quita siempre que
    # aparezca»).


def render_docs(con_sugerencias=True):
    """Pestaña de documentación y recomendaciones (cuerpo ancho, no el sidebar)."""
    st.markdown('## 📖 Documentación y recomendaciones')
    st.markdown(
        'Esta herramienta identifica **borreguiles** (cervunales higroturbosos / '
        '*mountain wet meadows*) combinando teledetección multi-fuente con un '
        'clasificador Random Forest, validado de forma **honesta** (GroupKFold por '
        'cuenca: entrena en unas cuencas y evalúa en otras nunca vistas).')

    # ---- 1. Estrategia de modelos -------------------------------------------
    st.markdown('### 🧭 ¿Qué modelo elegir?')
    st.markdown(
        'Todos los modelos parten de las **mismas variables base** (imagen aérea + '
        'topografía + Óptico Sentinel-2 + Radar Sentinel-1); de cada grupo de variables que dicen '
        'lo mismo usan una sola (27 en el modelo «General»). Las variantes añaden una '
        'fuente de **muy alta resolución** que sube el AUC. **La app extrae '
        'automáticamente la fuente extra** que el modelo elegido necesite — tú solo '
        'eliges el modelo.')
    # st.table no pasa por la traducción automática: cada idioma, su tabla
    st.table(i18n.pick({
        'Tu zona': ['🇪🇸 Andalucía · Cataluña · Canarias',
                    '🌍 Cualquier otra zona'],
        'Modelo recomendado': ['Con infrarrojo', 'General'],
        'Variables': ['32', '27'],
        'AUC honesto': ['0,945', '0,847'],
        'Requisitos': ['Gratis (PNOA IR 0,25 m)', 'Ninguno — universal'],
    }, {
        'Your area': ['🇪🇸 Andalusia · Catalonia · Canary Islands', '🌍 Any other area'],
        'Recommended model': ['With infrared', 'General'],
        'Variables': ['32', '27'],
        'Honest AUC': ['0.945', '0.847'],
        'Requirements': ['Free (PNOA IR 0.25 m)', 'None — universal'],
    }))
    st.markdown(i18n.pick(
        '- **El mejor resultado es el del modelo «Con infrarrojo», y es gratis** (AUC 0,945). '
        'Úsalo siempre que trabajes en Andalucía, Cataluña o Canarias.\n'
        '- **En el resto de España y fuera** usa el modelo «General» '
        '(AUC 0,847).\n'
        '- **Con Microsoft Planetary Computer** no se lee el radar y la app usa sola '
        'la versión de cada modelo entrenada sin él: AUC 0,833 el «General» y 0,937 el '
        '«Con infrarrojo».\n'
        '- Al abrirse, la app trae de GitHub los modelos nuevos o actualizados; la última '
        'línea de «Esta instalación», en la Ayuda, dice si están al día.',
        '- **The best result is that of the "With infrared" model, and it is free** (AUC '
        '0.945). Use it whenever you work in Andalusia, Catalonia or the Canary '
        'Islands.\n'
        '- **Elsewhere in Spain and abroad** use the "General" model '
        '(AUC 0.847).\n'
        '- **With Microsoft Planetary Computer** the radar is not read and the '
        'app automatically uses the version of each model trained without it: AUC 0.833 '
        'for "General" and 0.937 for "With infrared".\n'
        '- When it opens, the app fetches new or updated models from GitHub; the last '
        'line of "This installation", in Help, says whether they are up to date.'))

    # ---- 2. Cobertura geográfica del CIR ------------------------------------
    st.markdown(i18n.pick('### 🗺️ Cobertura de la ortofoto infrarroja (modelo «Con infrarrojo», gratis)',
                           '### 🗺️ Coverage of the infrared orthophoto ("With infrared" model, free)'))
    c1, c2 = st.columns(2)
    with c1:
        st.success(
            '**✅ Con WMS infrarrojo (0,25 m)**\n\n'
            '- **Andalucía** → Sierra Nevada\n'
            '- **Cataluña** → Aigüestortes, Pirineos\n'
            '- **Canarias** → Teide, Garajonay, Caldera de Taburiente')
    with c2:
        st.warning(i18n.pick(
            '**❌ Sin WMS infrarrojo** (usa el modelo «General»)\n\n'
            '- **Aragón** → Ordesa y Monte Perdido\n'
            '- **Cantabria / Castilla y León** → Picos de Europa\n'
            '- **Madrid / CyL** → Sierra de Guadarrama',
            '**❌ Without infrared WMS** (use the "General" model)\n\n'
            '- **Aragón** → Ordesa y Monte Perdido\n'
            '- **Cantabria / Castile-León** → Picos de Europa\n'
            '- **Madrid / Castile-León** → Sierra de Guadarrama'))
    st.caption(
        'El IGN tiene PNOA Falso Color Infrarrojo a 0,25 m para **toda España**, pero solo '
        'como descarga COG por hojas MTN25 (sin WMS) → conectarlo cubriría también Ordesa, '
        'Picos y Guadarrama. El modelo «Con infrarrojo» se calibró con Andalucía; en otras CCAA las '
        'variables IR son comparables pero no idénticas → para máxima precisión, entrena '
        'con verdad-terreno local.')

    # ---- 4. Fuentes de datos y variables ------------------------------------
    # Qué variables usa cada modelo, con radar y sin él, leídas de los propios modelos
    # (la autora, 27-11-2026: «poner en data source los detalles de modelos y variables»).
    st.markdown(i18n.pick('### 🛰️ Fuentes de datos y variables de cada modelo',
                          '### 🛰️ Data sources and variables of each model'))
    _nombre_f = {
        'img': i18n.pick('Imagen aérea (ESRI / PNOA)', 'Aerial image (ESRI / PNOA)'),
        'topo': i18n.pick('Relieve (Copernicus DEM 30 m)', 'Terrain (Copernicus DEM 30 m)'),
        's2': i18n.pick('Óptico Sentinel-2 (2017-2025)', 'Optical Sentinel-2 (2017-2025)'),
        's1': 'Radar Sentinel-1 (2019-2024)',
        'cir': i18n.pick('Infrarrojo PNOA Falso Color 0,25 m',
                         'PNOA False Colour infrared 0.25 m')}
    _aporta = {
        'img': i18n.pick('Textura, color, patrón y contexto alpino (% de roca)',
                         'Texture, colour, pattern and alpine context (% rock)'),
        'topo': i18n.pick('Altitud, pendiente y curvatura', 'Elevation, slope and curvature'),
        's2': i18n.pick('Verdor, humedad de la vegetación y agua (NDVI, NDMI, NDWI…) de '
                        'inicio (15 jun – 25 jul) y fin de verano (1 ago – 30 sep), y cómo '
                        'varían en el verano (1 jun – 30 sep)',
                        'Greenness, vegetation moisture and water (NDVI, NDMI, NDWI…) of '
                        'early (15 Jun – 25 Jul) and late summer (1 Aug – 30 Sep), and how '
                        'they vary over the summer (1 Jun – 30 Sep)'),
        's1': i18n.pick('Humedad y estructura del suelo, con el relieve corregido; atraviesa '
                        'las nubes', 'Soil moisture and structure, with the relief '
                        'corrected; it goes through clouds'),
        'cir': i18n.pick('Verdor a 25 cm: lo nítido que es el borde de la mancha',
                         'Greenness at 25 cm: how sharp the edge of the patch is')}
    _cols_m = []
    for _nm, _rot in (('rf_sierra_nevada.joblib', i18n.pick('General', 'General')),
                      ('rf_sierra_nevada_cir.joblib',
                       i18n.pick('Con infrarrojo', 'With infrared'))):
        _rc = model_path(_nm)
        _rs = modelo_sin_radar(str(_rc)) if _rc.exists() else None
        for _r, _suf in ((_rc, i18n.pick('con radar', 'with radar')),
                         (_rs, i18n.pick('sin radar', 'without radar'))):
            if _r is not None and Path(_r).exists():
                _cols_m.append((f'{_rot} · {_suf}',
                                _variables_modelo(str(_r), Path(_r).stat().st_mtime)))
    _por_fuente = {c: [[f for f in v if fuente_de_variable(f) == c] for _, v in _cols_m]
                   for c in FUENTES_VARIABLES}
    _k_f = i18n.pick('Fuente', 'Source')
    _k_a = i18n.pick('Qué aporta', 'What it adds')
    _tabla_f = [{_k_f: _nombre_f[c],
                 **{t: (str(len(n)) if n else '—') for (t, _), n in zip(_cols_m,
                                                                       _por_fuente[c])},
                 _k_a: _aporta[c]} for c in FUENTES_VARIABLES]
    _tabla_f.append({_k_f: '**' + i18n.pick('Total', 'Total') + '**',
                     **{t: f'**{len(v)}**' for t, v in _cols_m}, _k_a: ''})
    st.table(pd.DataFrame(_tabla_f).set_index(_k_f))
    st.caption(i18n.pick(
        'Cada modelo usa una variable por grupo de las que dicen lo mismo. Con Microsoft '
        'Planetary Computer no se lee el Radar Sentinel-1 y la app usa sola la versión «sin '
        'radar». El «Con infrarrojo» añade la ortofoto infrarroja: su variable más '
        'informativa es el **contraste del verdor a 25 cm**, justo el borde nítido del '
        'borreguil contra la roca, que el Óptico Sentinel-2 (10 m) promedia y pierde.',
        'Each model uses one variable per group of those that say the same. With Microsoft '
        'Planetary Computer the Radar Sentinel-1 is not read and the app uses the '
        '"without radar" version on its own. "With infrared" adds the infrared orthophoto: '
        'its most informative variable is the **greenness contrast at 25 cm**, precisely the '
        'sharp edge of the alpine wet meadow against the rock, which the Optical Sentinel-2 '
        '(10 m) averages out and loses.'))
    if _cols_m:
        with st.expander(i18n.pick('Ver las variables de cada fuente',
                                   'See the variables of each source')):
            _todas = []
            for _, _v in _cols_m:
                _todas += [f for f in _v if f not in _todas]
            _solo = {'s1': i18n.pick(' · no están en las versiones sin radar',
                                     ' · not in the versions without radar'),
                     'cir': i18n.pick(' · solo en «Con infrarrojo»',
                                      ' · only in "With infrared"')}
            st.markdown('\n'.join(
                f'- **{_nombre_f[c]}** ({len(_vs)}{_solo.get(c, "")}): '
                + ', '.join(pp.etiqueta(f) for f in _vs)
                for c in FUENTES_VARIABLES
                for _vs in [[f for f in _todas if fuente_de_variable(f) == c]] if _vs))

    # ---- 4b. Importancia de variables y lectura jerárquica ------------------
    st.markdown('### 🔑 Qué variables pesan más y cómo leer las gráficas')
    st.markdown(i18n.pick(
        'El modelo es un **Random Forest**: cada variable recibe una **importancia** '
        '(0–100 %) según cuánto ayuda a separar *borreguil* de *no-borreguil*. Muchas '
        'variables son casi iguales (por ejemplo, 13 índices de verdor de Sentinel-2 con '
        'correlación de 0,89 o más), así que el modelo usa **una sola de cada grupo**: '
        'si usara todas, se repartiría el peso entre ellas y cada una parecería poco '
        'importante. El acierto es el mismo.\n\n'
        'Al seleccionar un punto (**2 · Revisar**), el panel **📈 Este punto frente a '
        'los borreguiles verdaderos** muestra las variables que más pesan, **una por '
        'grupo** de parecidas, y para cada una dónde cae el punto entre los borreguiles '
        'con los que se entrenó el modelo: en la franja verde está la mitad central de '
        'ellos.',
        'The model is a **Random Forest**: each variable receives an **importance** '
        '(0–100 %) according to how much it helps separate *borreguil* from '
        '*non-borreguil*. Many variables are nearly identical (for example, 13 Sentinel-2 '
        'greenness indices with a correlation of 0.89 or more), so the model uses **only '
        'one from each group**: if it used them all, the weight would be shared among them '
        'and each would look unimportant. The accuracy is the same.\n\n'
        'When you select a point (**2 · Review**), the panel **📈 This point vs the real '
        'borreguiles** shows the variables that weigh most, **one per group** of similar '
        'ones, and for each one where the point falls among the borreguiles the model was '
        'trained with: the green band holds the middle half of them.'))
    _mp = st.session_state.get('start_model_path') or str(model_path('rf_sierra_nevada_cir.joblib'))
    _rs = load_ref_stats(_mp)
    _ic, _ = pp.chart_importance(_rs, topn=15) if _rs else (None, [])
    if _ic is not None:
        st.altair_chart(_ic, width='stretch')
        st.caption(i18n.pick(
            f'Ranking del modelo seleccionado («{_rotulo_modelo(Path(_mp).name)}»), variable a '
            'variable. '
            'Cambia según el modelo: en el «Con infrarrojo» manda el **contraste del NDVI a 0,25 m**; '
            'en el «General», la **altitud**, la **caída estacional del NDVI**, la '
            '**diferencia VV–VH del radar**, el **NDVI de inicio de verano** (15 jun – 25 jul) y la '
            '**variación del NDWI**. Cada modelo usa una '
            'sola variable de cada grupo de las que dicen lo mismo, así que el peso no se '
            'reparte entre variables correlacionadas.',
            f'Ranking of the selected model ("{_rotulo_modelo(Path(_mp).name)}"), variable by '
            'variable. It '
            'changes with the model: in "With infrared" the **NDVI contrast at 0.25 m** '
            'leads; in "General", **elevation**, the **seasonal NDVI drop**, the '
            '**radar VV–VH difference**, **early-summer NDVI** (15 Jun – 25 Jul) and the **NDWI '
            'variability**. Each model uses a single '
            'variable from each group of those that say the same, so the weight is not '
            'shared among correlated ones.'))
    else:
        st.caption(i18n.pick(
            'Este modelo no trae el peso de sus variables, así que no se puede mostrar '
            'el ranking.',
            'This model does not carry the weight of its variables, so the ranking cannot '
            'be shown.'))

    # ---- 4c. Cuántos puntos hacen falta (curva de Sierra Nevada, 24-11) ------
    import mapeo_rf as mr      # aquí, como en «Mapa Modelo RF»: la app lo carga tarde
    st.markdown(i18n.pick('### 📏 ¿Cuántos puntos necesito para el mapa?',
                          '### 📏 How many points do I need for the map?'))
    (_mn, _mb), (_rn, _rb) = mr.MINIMO_PUNTOS, mr.RECOMENDADO_PUNTOS
    _ma, _ra = mr.MINIMO_AUSENCIAS, mr.RECOMENDADO_AUSENCIAS
    _todos = mr.CURVA_PUNTOS_SN[-1]
    st.markdown(i18n.pick(
        'El mapa de borreguil sí o no (**3 · Hacer el mapa**) aprende de tus puntos '
        'revisados, borreguiles y no borreguiles. Para saber cuántos hacen falta, se '
        'repitió el procedimiento de la app con los 948 puntos revisados de Sierra Nevada '
        '(315 borreguiles): se entrenó con 20, 30… hasta '
        f'{_todos[0]} puntos elegidos al azar y se midió el acierto en zonas que el modelo '
        'no había visto, 10 veces para cada número.\n\n'
        f'- **Mínimo: unos {_mn} puntos revisados, con unos {_mb} borreguiles y {_ma} '
        f'ausencias** (AUC {_num(mr.auc_esperado(_mn))}; con todos, {_num(_todos[2])}).\n'
        f'- **Recomendado: unos {_rn}, con unos {_rb} borreguiles y {_ra} ausencias** (AUC '
        f'{_num(mr.auc_esperado(_rn))}). Con más, el acierto apenas cambia.\n'
        f'- **Con menos de {mr.POCOS_PUNTOS}**, el acierto baja y depende mucho de qué '
        'puntos sean.\n\n'
        'Las **ausencias** son los sitios que has comprobado que no son borreguil: '
        '`presence` = 0 en tu fichero, «No» en «¿Es borreguil?» al revisar un '
        'punto o «✗ Aquí no hay» al pinchar en el mapa. La app no se las inventa. Para buscarlas en sitios que no has '
        'muestreado, genera puntos al azar «lejos de tus puntos de campo» y revísalos.\n\n'
        'Es la referencia de Sierra Nevada: en una montaña más variada pueden hacer falta '
        'más. Los puntos repartidos por toda la zona enseñan más que los concentrados en '
        'pocos valles. La curva se hizo con 1 borreguil de cada 3 (por cada borreguil, 2 '
        'ausencias); de ahí salen los borreguiles y las ausencias de cada nivel, y no se '
        'probaron otras proporciones. En **3 · Hacer el mapa → 1 · Datos** la app te dice '
        'dónde estás en esta curva y cuántos borreguiles y ausencias te faltan.\n\n'
        'En tu zona, lo que vale es la curva con **tus** puntos: en **3 · Hacer el mapa → '
        '3 · Modelo**, después de entrenar, **📈 Calcular la curva con mis puntos** repite '
        'la validación del modelo aprendiendo con el 25, el 50 y el 75 % de tus puntos. Si '
        'de la mitad a todos (el doble) el acierto todavía sube, más puntos mejorarán el '
        'mapa; si ya casi no sube, más puntos parecidos apenas lo cambiarán.',
        'The borreguil yes-or-no map (**3 · Make the map**) learns from your reviewed '
        'points, borreguiles and non-borreguiles. To find out how many are needed, the '
        "app's procedure was repeated with the 948 reviewed points of Sierra Nevada (315 "
        f'borreguiles): it was trained with 20, 30… up to {_todos[0]} points chosen at '
        'random and its accuracy was measured in areas the model had not seen, 10 times '
        'for each number.\n\n'
        f'- **Minimum: about {_mn} reviewed points, with about {_mb} borreguiles and {_ma} '
        f'absences** (AUC {_num(mr.auc_esperado(_mn))}; with all of them, '
        f'{_num(_todos[2])}).\n'
        f'- **Recommended: about {_rn}, with about {_rb} borreguiles and {_ra} absences** '
        f'(AUC {_num(mr.auc_esperado(_rn))}). With more, the accuracy hardly changes.\n'
        f'- **With fewer than {mr.POCOS_PUNTOS}**, the accuracy drops and depends a lot on '
        'which points they are.\n\n'
        '**Absences** are the places you have checked are not a borreguil: `presence` = 0 '
        'in your file, "No" under "Is it a borreguil?" when reviewing a point or "✗ '
        'There is none" when clicking on the map. The app never invents them. To look for them in places you '
        'have not sampled, generate random points "far from your field points" and review '
        'them.\n\n'
        'It is the Sierra Nevada reference: a more varied mountain may need more. Points '
        'spread over the whole area teach more than points concentrated in a few valleys. '
        'The curve was made with 1 borreguil in 3 (2 absences for each borreguil); that is '
        'where the borreguiles and absences of each level come from, and other proportions '
        'were not tested. Under **3 · Make the map → 1 · Data** the app tells you where you '
        'are on this curve and how many borreguiles and absences you still need.\n\n'
        'In your area, what counts is the curve with **your** points: under **🌿 RF model '
        'map → 3 · Model**, after training, **📈 Compute the curve with my points** repeats '
        "the model's validation learning from 25, 50 and 75 % of your points. If accuracy "
        'still rises from half of them to all (twice as many), more points will improve the '
        'map; if it has almost stopped rising, more similar points will hardly change it.'))
    st.altair_chart(pp.chart_curva_puntos(
        mr.CURVA_PUNTOS_SN, marcas=[(_mn, i18n.pick('Mínimo', 'Minimum')),
                                    (_rn, i18n.pick('Recomendado', 'Recommended'))]),
        width='stretch')
    st.caption(i18n.pick(
        'Línea: media de 10 repeticiones; franja: 8 de cada 10. Línea discontinua: con '
        'todos los puntos. Procedimiento de la app con «Quitar las variables correlacionadas: '
        'Con VIF» (umbral 10), Random Forest y validación por zonas.',
        'Line: mean of 10 repetitions; band: 8 out of 10. Dashed line: with all the '
        'points. Same procedure as the app with "Remove correlated variables: With '
        'VIF" (threshold 10), Random Forest and spatial validation.'))

    # ---- 5. Cómo usar la app ------------------------------------------------
    st.markdown('### ▶️ Flujo de trabajo')
    st.markdown(i18n.pick(
        '1. **Tus puntos**: súbelos o deja que la app los genere al azar en el área '
        '(vector, WDPA ID o nombre).\n'
        '2. Elige la **fuente de datos** (Earth Engine: más rápido y con radar; Planetary '
        'Computer: sin cuenta y sin radar), el **modelo** según la tabla de arriba y la '
        '**fuente de imágenes** (PNOA en España, ESRI fuera).\n'
        '3. Pulsa **Analizar** (en «1 · Analizar»). La app descarga cada fuente, extrae las '
        'variables y predice con el **modelo de partida**. Si tus puntos traen '
        'etiquetas de campo, te dice cuántas acierta.\n'
        '4. En **2 · Revisar**, verifica puntos como borreguil o no '
        '(con quién revisa, su perfil y la certidumbre) y revisa su tipo; en **Mapa de '
        'los puntos** puedes añadir puntos pinchando. Descarga Excel, CSV o '
        '`puntos.geojson` en **Descargas**.\n'
        '5. *(Opcional)* si tienes puntos de campo de tu zona, marca **«Sumar mis puntos a los del modelo y reentrenarlo»**: '
        'la app vuelve a entrenar el modelo con sus 769 puntos de Sierra Nevada y los '
        'tuyos juntos, y te dice cuánto acierta en tus puntos.\n'
        f'6. En **3 · Hacer el mapa**, con tus puntos revisados (mínimo unos {_mn}, con '
        f'{_mb} borreguiles), entrena y valida un modelo de tu zona y haz el mapa de '
        'borreguil a 10 m. Necesita Earth Engine.',
        '1. **Your points**: upload them or let the app generate them at random in the '
        'area (vector, WDPA ID or name).\n'
        '2. Choose the **data source** (Earth Engine: faster and with radar; Planetary '
        'Computer: no account and no radar), the **model** according to the table above '
        'and the **image source** (PNOA in Spain, ESRI elsewhere).\n'
        '3. Click **Analyse** (in "1 · Analyse"). The app downloads each source, extracts the variables '
        'and predicts with the **starting model**. If your points carry field labels, '
        'it tells you how many it gets right.\n'
        '4. Under **2 · Review**, verify points as borreguil or not '
        '(with who is reviewing, their profile and the certainty) and review their type; '
        'on the **Results** map you can add points by clicking. Download Excel, CSV or '
        '`puntos.geojson` under **Downloads**.\n'
        '5. *(Optional)* if you have field points from your area, tick **"Add my points to the model\'s and retrain it"**: '
        'the app retrains the model with its 769 Sierra Nevada points and yours '
        'together, and tells you how well it does on your points.\n'
        f'6. Under **3 · Make the map**, with your reviewed points (at least about {_mn}, '
        f'with {_mb} borreguiles), train and validate a model for your area and make the '
        '10 m borreguil map. It needs Earth Engine.'))

    # ---- 6. Manual y sugerencias --------------------------------------------
    if con_sugerencias:
        st.markdown(i18n.pick('### 💬 Manual y sugerencias', '### 💬 Manual and suggestions'))
        mostrar_sugerencias('documentacion')


# ---- Qué modelo ha dado la probabilidad y cuánto acierta con los puntos de campo ----
def _zona_partida(info):
    z = (info or {}).get('training_zone')
    if not z or z == 'usuario':
        z = (info or {}).get('zona_partida') or '?'
    return str(z).split(' (')[0]


def nombre_modelo(zona_entrenamiento):
    """Cómo se llama un modelo en la app (la autora, 27-11-2026: por lo que es, no por
    dónde se entrenó). Los que trae la app (entrenados en Sierra Nevada), «General» o
    «Con infrarrojo»; los que guardas tú, el nombre de su zona."""
    z = str(zona_entrenamiento or '?')
    if 'sierra nevada (andaluc' in z.lower():
        return (i18n.pick('Con infrarrojo', 'With infrared') if 'CIR' in z
                else i18n.pick('General', 'General'))
    return z.split(' (')[0]


def _nombre_partida(info):
    """El nombre del modelo de partida de un análisis (ver nombre_modelo)."""
    z = (info or {}).get('training_zone')
    if not z or z == 'usuario':
        z = (info or {}).get('zona_partida') or '?'
    z = str(z)
    if z.endswith(' + tus puntos'):
        z = z[:-len(' + tus puntos')]
    return nombre_modelo(z)


def _titulo_modelo(info):
    """Una línea: qué modelo ha dado la probabilidad de los puntos."""
    if not info:
        return i18n.pick('No se ha podido ejecutar el modelo.', 'The model could not be run.')
    # Sin radar (Microsoft Planetary Computer): se usa la versión del modelo entrenada sin él
    sr = _es_sin_radar(info)
    if info.get('sumado'):
        z = _nombre_partida(info)
        return i18n.pick(
            f'**Modelo usado: «{z}»' + (' (versión sin radar)' if sr else '')
            + ' reentrenado sumando tus puntos**',
            f'**Model used: "{z}"' + (' (version without radar)' if sr else '')
            + ' retrained adding your points**')
    if info.get('mode') == 'trained':
        return i18n.pick('**Modelo usado: el entrenado con tus puntos**',
                         '**Model used: the one trained with your points**')
    z = _nombre_partida(info)
    return i18n.pick(
        f'**Modelo usado: el de partida, «{z}»' + (' (versión sin radar)' if sr else '')
        + '**' + (', porque el radar no se ha leído' if sr else ''),
        f'**Model used: the starting model, "{z}"' + (' (version without radar)' if sr else '')
        + '**' + (', because the radar was not read' if sr else ''))


def _de(ok, n):
    return '—' if not n else i18n.pick(f'{ok} de {n}', f'{ok} of {n}')


def _nombre_columna(c):
    """Cabecera de la Tabla de resultados en el idioma de la app. Las variables, con
    el mismo nombre que en la ficha del punto."""
    nombres = {
        'cuenca_id': ('Cuenca', 'Basin'),
        'source': ('Fuente', 'Source'),
        'cuartil': ('Cuartil', 'Quartile'),
        'origin': ('Origen', 'Origin'),
        # (25-11-2026, la autora: «no entiendo borreguil (tu etiqueta)» → «mejor revisado»)
        'Borreguil': ('¿Borreguil? (revisado)', 'Borreguil? (reviewed)'),
        # (25-11-2026, la autora: «no entiendo decisión» → «Sí, «Resultado»»)
        'decision': ('Resultado', 'Result'),
        'rf_proba': ('Probabilidad', 'Probability'),
        'ambiente': ('Ambiente', 'Environment'),
        'humedad': ('Humedad', 'Moisture'),
        'pureza': ('Pureza', 'Purity'),
        'tipo_revisado': ('Tipo revisado', 'Type reviewed'),
        'dist_cauce_m': ('Distancia al cauce (m)', 'Distance to stream (m)'),
    }
    if c in nombres:
        return i18n.pick(*nombres[c])
    if c in ('ID', 'lat', 'lon'):
        return c
    return pp.etiqueta(c)


@st.cache_resource(show_spinner=False)
def _capa_cauces_app():
    """La capa de ríos y arroyos de OpenStreetMap que va con la app (Sierra Nevada): se
    lee una vez; pesa 5 MB y leerla en cada pasada hacía lento el mapa del punto."""
    return bp.leer_capa_cauces(APP_DIR / bp.CAUCES_SN)


def _cauces_cerca(lon, lat, radio_m=3000):
    """Los ríos y arroyos de OpenStreetMap a menos de `radio_m` (aprox.) del punto: los
    de la capa de la app si el punto cae en ella; si no, los que se descargaron al
    analizar. None si no hay ninguno leído para esa zona (no se descarga nada aquí:
    el mapa del punto no tiene que esperar a internet)."""
    from shapely.geometry import box
    try:
        lineas, cob = _capa_cauces_app()
    except Exception:                                         # noqa: BLE001
        lineas, cob = None, None
    if lineas and cob and len(cob) == 4 and cob[0] <= lon <= cob[2] and cob[1] <= lat <= cob[3]:
        fuente = lineas
    else:
        fuente = [g for ls in bp._cauces_descargados.values() for g in (ls or [])]
    if not fuente:
        return None
    d = radio_m / 111000.0
    zona = box(lon - d * 1.3, lat - d, lon + d * 1.3, lat + d)
    return [g for g in fuente if g.intersects(zona)]


def _texto_ids(ids, n_max=5):
    """Los nombres de unos puntos, como mucho `n_max` (y «y N más»)."""
    ids = [str(i) for i in ids]
    txt = ', '.join(ids[:n_max])
    if len(ids) > n_max:
        txt += i18n.pick(f' y {len(ids) - n_max} más', f' and {len(ids) - n_max} more')
    return txt


def _nombres_cortos(ids, punto=''):
    """Los nombres de unos puntos como en la tabla: los ID heredados del nombre de
    la capa ('all_…_epsg25830.17'), con su número ('17', o 'Punto 17' con `punto`);
    los demás, igual. Se calculan con todos los puntos (y estos), para que un número
    no pueda ser de dos capas a la vez."""
    ids = [str(i) for i in ids]
    todos = [str(r.get('ID')) for r in (st.session_state.get('rows') or [])]
    todos += [i for i in ids if i not in set(todos)]
    mapa = dict(zip(todos, bp.nombres_de_punto(todos, punto=punto)))
    return [mapa.get(i, i) for i in ids]


def _msg_puntos(ids, es_1, es_n, en_1, en_n):
    """Aviso tras verificar o quitar puntos: «Punto 17 marcado como …» con uno, o «3
    puntos marcados como …: 1, 17, 289» con varios (los ID heredados del nombre de la
    capa, con su número, como en la tabla; los demás, como son: «SN_005»)."""
    ids = list(ids)
    if len(ids) == 1:
        _uno = _nombres_cortos(ids, i18n.pick('Punto', 'Point'))[0]
        return i18n.pick(f'{_uno} {es_1}.', f'{_uno} {en_1}.')
    _varios = _texto_ids(_nombres_cortos(ids))
    return i18n.pick(f'{len(ids)} puntos {es_n}: {_varios}.',
                     f'{len(ids)} points {en_n}: {_varios}.')


def _num(v, nd=3, signo=False):
    """Número para MOSTRARLO, con `nd` decimales: con coma decimal en español y con
    punto en inglés ('—' si falta o no es un número). Sin separador de miles: en
    español «20.910» se leería como veinte mil. Lo que se guarda y se descarga no
    cambia. `signo`: con «+» delante de los positivos."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return '—'
    if x != x:
        return '—'
    t = f'{x:+.{nd}f}' if signo else f'{x:.{nd}f}'
    return t if i18n.get_lang() == 'en' else t.replace('.', ',')


@st.cache_data(ttl=300, show_spinner=False)
def _uso_cuenta_compartida():
    """Las horas de cálculo que le quedan a este ordenador en la cuenta compartida de
    MonitorEO (gee_backend.uso_compartida), preguntadas como mucho cada 5 minutos; None
    si su servicio no responde o no hay earthengine-api."""
    try:
        import gee_backend as geb
        return geb.uso_compartida()
    except Exception:                                   # noqa: BLE001
        return None


def _n3(v):
    """Número con 3 decimales: con coma en español y con punto en inglés."""
    return _num(v)


def _con_coma(df, nd=3, decimales=None):
    """La tabla para st.dataframe con sus columnas de decimales a `nd` cifras (o las
    de `decimales`, {columna: cifras}) y, en español, con coma. Siguen siendo números
    (se ordenan bien): solo cambia lo que se ve. Sus columnas no llevan `format` en
    el column_config, que mandaría sobre esto."""
    decimales = decimales or {}
    flo = [c for c in df.columns if pd.api.types.is_float_dtype(df[c])]
    if not flo or not len(df):
        return df
    return df.style.format(
        {c: (lambda v, k=decimales.get(c, nd): _num(v, k)) for c in flo}, na_rep='—')


def _var(n):
    """Nombre legible de una variable (el de la Tabla y la ficha), no el del código."""
    return pp.etiqueta(n)


def _auc_txt(v):
    if v is None:
        return '—'
    return _n3(v)


def mostrar_modelo(info, threshold, casilla=None):
    """Qué modelo se ha usado y por qué; cuánto acierta con los puntos de campo (y, si
    se entrenó uno propio, los dos lado a lado); y con cuántas pseudo-ausencias se
    entrenó el propio. `casilla`: «Sumar mis puntos…» marcada sin haber reentrenado
    aún ('boton' si hay puntos con etiqueta para sumar, 'sin_puntos' si no)."""
    if not info:
        st.warning(_titulo_modelo(info))
        return
    pick = i18n.pick
    modo, motivo, comp = info.get('mode'), info.get('motivo'), info.get('comparacion')
    titulo, z = _titulo_modelo(info), _zona_partida(info)
    sumado = bool(info.get('sumado'))
    # Como en la demo de Resultados (27-11-2026): en frases sin recuadro de color; los
    # avisos (pocos puntos, sin borreguiles…) siguen en su recuadro.
    sr = _es_sin_radar(info)
    _z_es = z + (', versión sin radar, porque el radar no se ha leído' if sr else '')
    _z_en = z + (', version without radar, because the radar was not read' if sr else '')
    if sumado:
        _n_t, _n_s = info.get('n_tabla', 0), info.get('n_pos', 0) + info.get('n_neg', 0)
        _q = info.get('n_tabla_quitados', 0)
        _nm = _nombre_partida(info)
        st.markdown(pick(
            f'«{_nm}»' + (' (versión sin radar)' if sr else '')
            + ' reentrenado sumando tus puntos. ',
            f'"{_nm}"' + (' (version without radar)' if sr else '')
            + ' retrained adding your points. ') + pick(
            f'Un solo modelo con todos los puntos: {_n_t} de {z} '
            f'({info.get("n_tabla_si", 0)} borreguiles y {info.get("n_tabla_no", 0)} no) y '
            f'{_n_s} tuyos ({info.get("n_pos", 0)} borreguiles y {info.get("n_neg", 0)} '
            'ausencias).'
            + (f' {_q} de los de {z} estaban en el mismo sitio que alguno tuyo: cuenta el '
               'tuyo.' if _q else ''),
            f'A single model with all the points: {_n_t} from {z} '
            f'({info.get("n_tabla_si", 0)} borreguiles and {info.get("n_tabla_no", 0)} not) '
            f'and {_n_s} of yours ({info.get("n_pos", 0)} borreguiles and '
            f'{info.get("n_neg", 0)} absences).'
            + (f' {_q} of the {z} ones were at the same place as one of yours: yours '
               'counts.' if _q else '')))
    elif modo == 'pretrained' and motivo == 'sin_puntos':
        st.warning(titulo + '. ' + pick(
            'Has marcado «Sumar mis puntos a los del modelo», pero ninguno de tus puntos '
            'dice si es borreguil o no: se usa el modelo tal cual. Verifica puntos en la '
            'pantalla «2 · Revisar» o añádelos pinchando en el mapa de Resultados.',
            'You ticked "Add my points to the model\'s", but none of your points says '
            'whether it is a borreguil: the model is used as it is. Verify points in the '
            '"2 · Review" screen or add them by clicking on the Results map.'))
    elif modo == 'pretrained':
        if motivo == 'comparado':
            st.info(titulo + '. ' + pick(
                'También se ha entrenado uno con tus puntos, pero acierta menos con tus '
                'puntos de campo, así que se queda el de partida.',
                'A model was also trained with your points, but it gets fewer of your field '
                'points right, so the starting model is kept.'))
        elif motivo == 'pocos_datos':
            st.warning(titulo + '. ' + pick(
                f'Has pedido entrenar también un modelo con tus puntos, pero hay pocos: '
                f'{info.get("n_pos", 0)} borreguiles y {info.get("n_neg", 0)} negativos '
                f'({info.get("n_neg_campo", 0)} ausencias de campo y '
                f'{info.get("n_pseudo", 0)} pseudo-ausencias). Hacen falta al menos '
                f'{bp.MIN_ENTRENAR} de cada; mejor 30 o más.',
                f'You asked to also train a model with your points, but there are too few: '
                f'{info.get("n_pos", 0)} borreguiles and {info.get("n_neg", 0)} negatives '
                f'({info.get("n_neg_campo", 0)} field absences and '
                f'{info.get("n_pseudo", 0)} pseudo-absences). At least {bp.MIN_ENTRENAR} '
                'of each are needed; 30 or more is better.'))
        elif motivo == 'sin_borreguiles':
            st.warning(titulo + '. ' + pick(
                'Has pedido entrenar también un modelo con tus puntos, pero no traen '
                'ningún borreguil marcado.',
                'You asked to also train a model with your points, but none of them is '
                'marked as borreguil.'))
        elif casilla == 'boton':
            # La casilla está marcada pero el modelo aún no se ha reentrenado: el
            # botón para hacerlo va justo debajo.
            st.markdown(pick(
                f'El de partida ({_z_es}). Has marcado «Sumar mis puntos a los del modelo y '
                'reentrenarlo»: pulsa el botón de abajo para reentrenarlo ahora.',
                f'The starting one ({_z_en}). You ticked "Add my points to the model\'s and '
                'retrain it": press the button below to retrain it now.'))
        elif casilla == 'sin_puntos':
            st.markdown(pick(
                f'El de partida ({_z_es}). Has marcado «Sumar mis puntos a los del modelo y '
                'reentrenarlo», pero ninguno de tus puntos dice todavía si es borreguil o '
                'no. Revisa puntos en «2 · Revisar» o añádelos pinchando en el mapa.',
                f'The starting one ({_z_en}). You ticked "Add my points to the model\'s and '
                'retrain it", but none of your points says yet whether it is a borreguil. '
                'Review points in "2 · Review" or add them by clicking on the map.'))
        else:
            st.markdown(pick(
                f'El de partida ({_z_es}), como hace la app de entrada. Para sumar tus '
                'puntos y reentrenarlo: en Analizar, panel «Modelo».',
                f'The starting one ({_z_en}), as the app does by default. To add your '
                'points and retrain it: in Analyse, "Model" panel.'))
    elif comp is None:
        st.info(titulo + '.')
    elif comp.get('motivo') == 'comparado':
        st.success(titulo + '. ' + pick(
            f'Acierta más con tus puntos de campo que el de partida ({z}).',
            f'It gets more of your field points right than the starting model ({z}).'))
    else:
        razon = {
            'pocos_de_campo': pick(
                f'hacen falta al menos {bp.MIN_COMPARAR} borreguiles y {bp.MIN_COMPARAR} '
                f'ausencias de campo, y hay {comp.get("n_si", 0)} y {comp.get("n_no", 0)}',
                f'at least {bp.MIN_COMPARAR} field borreguiles and {bp.MIN_COMPARAR} field '
                f'absences are needed, and there are {comp.get("n_si", 0)} and '
                f'{comp.get("n_no", 0)}'),
            'sin_partida': pick('no hay modelo de partida', 'there is no starting model'),
        }.get(comp.get('motivo'), pick('la validación por zonas no ha funcionado',
                                       'the spatial validation did not work'))
        st.warning(titulo + '. ' + pick(
            f'No se ha podido comparar con el de partida: {razon}. Su acierto no está '
            'comprobado con datos de campo.',
            f'It could not be compared with the starting model: {razon}. How well it '
            'does has not been checked against field data.'))

    ac = bp.aciertos_en_campo(info.get('evaluacion'), threshold)
    if ac:
        usado = 'propio' if modo == 'trained' else 'partida'
        # Los nombres de las filas, como en la demo: la zona del modelo («Sierra Nevada»)
        nombres = {'partida': z,
                   'propio': pick('Con tus puntos (validación por zonas)',
                                  'With your points (spatial validation)')}
        if sumado:
            nombres = {'partida': pick(f'{z}, sin tus puntos', f'{z}, without your points'),
                       'propio': pick('Sumando tus puntos (validación por zonas)',
                                      'Adding your points (spatial validation)')}
        tabla = (f"| {pick('Modelo', 'Model')} | "
                 + pick('Borreguiles acertados', 'Borreguiles right')
                 + ' | ' + pick('Ausencias acertadas', 'Absences right')
                 + ' | AUC |\n|---|---|---|---|\n')
        for k in ('partida', 'propio'):
            if k in ac:
                a = ac[k]
                marca = ' ✓' if (k == usado and len(ac) > 1) else ''
                tabla += (f'| {nombres[k]}{marca} | {_de(a["ok_si"], a["n_si"])} | '
                          f'{_de(a["ok_no"], a["n_no"])} | {_auc_txt(a["auc"])} |\n')
        st.markdown(tabla)
        st.markdown(pick(
            f'Acertado: un borreguil que el modelo da por probable o posible (umbral '
            f'{_num(threshold)}); una ausencia que no da por borreguil.',
            f'Right: a borreguil that the model calls probable or possible (threshold '
            f'{_num(threshold)}); an absence that it does not call borreguil.'))
        # El acierto según la certidumbre y el perfil de quien dio cada etiqueta
        # (24-11): si el modelo falla sobre todo donde la revisión era dudosa, el fallo
        # puede ser de la etiqueta y no del modelo. Las dos tablas, una al lado de otra.
        _rev = bp.aciertos_por_revision(info.get('evaluacion'), threshold, usado)
        if _rev:
            _txt_val = {
                'certidumbre': {'high': pick('Alta', 'High'), 'medium': pick('Media', 'Medium'),
                                'low': pick('Baja', 'Low'), '': pick('Sin indicar', 'Not stated')},
                'perfil': {'student': pick('Estudiante', 'Student'),
                           'manager': pick('Gestor/a', 'Manager'),
                           'researcher': pick('Investigador/a', 'Researcher'),
                           '': pick('Sin indicar', 'Not stated')}}
            _cab = {'certidumbre': pick('Certidumbre', 'Certainty'),
                    'perfil': pick('Perfil de quien revisa', 'Reviewer profile')}

            def _ambar(txt, si):        # lo que falla, en ámbar (la demo)
                return f'<span style="color:#F5B83D">{txt}</span>' if si else txt

            with st.expander(pick('Acierto según la certidumbre y el perfil de quien revisa',
                                  'Accuracy by the certainty and profile of the reviewer'),
                             key='res_cert'):
                _cols_rev = st.columns(len(_rev), gap='small')
                for _col_r, (clave, grupos) in zip(_cols_rev, _rev.items()):
                    tb = (f'| {_cab[clave]} | '
                          + pick('Borreguiles acertados', 'Borreguiles right') + ' | '
                          + pick('Ausencias acertadas', 'Absences right')
                          + ' |\n|---|---|---|\n')
                    for v, a_ in grupos.items():
                        f_si = a_['ok_si'] < a_['n_si']
                        f_no = a_['ok_no'] < a_['n_no']
                        tb += (f"| {_ambar(_txt_val[clave][v], f_si or f_no)} | "
                               f"{_ambar(_de(a_['ok_si'], a_['n_si']), f_si)} | "
                               f"{_ambar(_de(a_['ok_no'], a_['n_no']), f_no)} |\n")
                    _col_r.markdown(tb, unsafe_allow_html=True)
                # La última frase, según los datos: dónde falla
                _gc = _rev.get('certidumbre') or {}
                _f_si = sum(a_['n_si'] - a_['ok_si'] for a_ in _gc.values())
                _f_no = sum(a_['n_no'] - a_['ok_no'] for a_ in _gc.values())
                _bs = _gc.get('low', {})
                _b_si = _bs.get('n_si', 0) - _bs.get('ok_si', 0)
                _b_no = _bs.get('n_no', 0) - _bs.get('ok_no', 0)
                _aqui = ''
                if _gc and _f_si + _f_no == 0:
                    _aqui = pick(' Aquí acierta todos.', ' Here it gets them all right.')
                elif _gc and _b_si + _b_no == _f_si + _f_no:
                    if (_f_si, _f_no) == (1, 0):
                        _aqui = pick(' Aquí, el único borreguil que no acierta es el de '
                                     'certidumbre baja.',
                                     ' Here, the only borreguil it gets wrong is the '
                                     'low-certainty one.')
                    elif (_f_si, _f_no) == (0, 1):
                        _aqui = pick(' Aquí, la única ausencia que no acierta es la de '
                                     'certidumbre baja.',
                                     ' Here, the only absence it gets wrong is the '
                                     'low-certainty one.')
                    else:
                        _aqui = pick(f' Aquí, los {_f_si + _f_no} puntos que no acierta son '
                                     'de certidumbre baja.',
                                     f' Here, the {_f_si + _f_no} points it gets wrong are '
                                     'all low-certainty.')
                elif _gc and _b_si + _b_no:
                    _aqui = pick(f' Aquí, de los {_f_si + _f_no} puntos que no acierta, '
                                 f'{_b_si + _b_no} son de certidumbre baja.',
                                 f' Here, of the {_f_si + _f_no} points it gets wrong, '
                                 f'{_b_si + _b_no} are low-certainty.')
                st.markdown(pick(
                    'Del modelo que se está usando. Si falla sobre todo en los puntos de '
                    'certidumbre baja, puede que el error sea de la etiqueta y no del modelo.',
                    'For the model in use. If it fails mostly on low-certainty points, the '
                    'error may be in the label rather than in the model.') + _aqui)
        if sumado and 'propio' in ac:
            st.caption(pick(
                'Sumando tus puntos: cada grupo de tus puntos se predice con un modelo que '
                f'no los ha visto, ni a los de {z} a menos de '
                f'{bp.SEPARACION_CV_M} m de ellos, para no hacer trampa.',
                'Adding your points: each group of your points is predicted with a model '
                f'that has not seen them, nor the {z} points within '
                f'{bp.SEPARACION_CV_M} m of them, so as not to cheat.'))
        elif sumado:
            st.caption(pick(
                'Tus puntos están todos en la misma zona, así que no se puede medir sin '
                'hacer trampa cuánto acierta el modelo sumándolos.',
                'Your points are all in the same zone, so how well the model does when '
                'adding them cannot be measured without cheating.'))
    elif modo == 'pretrained':
        st.caption(pick(
            'Tus puntos no traen etiquetas de campo, así que no se puede medir cuánto '
            'acierta en tu zona: revisa los puntos que propone (en «2 · Revisar»).',
            'Your points carry no field labels, so how well it does in your area cannot '
            'be measured: review the points it proposes (in "2 · Review").'))

    if info.get('n_pseudo'):
        st.warning(pick(
            f'Para entrenar el modelo con tus puntos se usaron {info.get("n_neg_campo", 0)} '
            f'ausencias de campo y **{info["n_pseudo"]} pseudo-ausencias**: puntos sin '
            f'etiqueta a más de {info.get("neg_buffer_m", 250)} m de un borreguil, que la '
            'app da por no borreguil. Algunas pueden ser borreguiles que nadie ha '
            'visitado. Cuantas más ausencias de campo (`presence` = 0) subas, más fiable '
            'será.',
            f'To train the model with your points, {info.get("n_neg_campo", 0)} field '
            f'absences and **{info["n_pseudo"]} pseudo-absences** were used: unlabelled '
            f'points more than {info.get("neg_buffer_m", 250)} m from a borreguil, which '
            'the app takes as non-borreguil. Some may be borreguiles nobody has visited. '
            'The more field absences (`presence` = 0) you upload, the more reliable it '
            'will be.'))


def leer_variables_puntos(nuevos, work, backend, years, img_src='esri', skip_imgs=False,
                          skip_s2=False, con_cir=False, gee_project='',
                          consola=None):
    """Lee las variables del detector SOLO para los puntos `nuevos` (los añadidos
    pinchando en el mapa), con las mismas fuentes que la ejecución. Con Earth Engine
    tarda unos segundos, radar incluido; con Microsoft Planetary Computer el radar no se lee (si
    el análisis tampoco lo leyó, se usa la versión del modelo sin radar).
    `consola`: dónde se ven en vivo los mensajes de la lectura (un st.status).
    Devuelve la lista de fuentes que fallaron (vacía si todo fue bien)."""
    fallos = []
    work = Path(work)
    bbox = bp.bbox_with_buffer(nuevos)
    if backend == 'gee':
        import gee_backend as geb
        if not geb.is_initialized():
            ok, msg = geb.initialize(gee_project or leer_ajuste('gee_project', '') or None)
            if not ok:
                fallos.append(f'Earth Engine ({msg})')
                return fallos
    with live_logs(consola if consola is not None else st.container()):
        if not skip_imgs:
            # Las imágenes se nombran por su posición en la lista: se descargan
            # aparte y cada una pasa a llamarse como su punto.
            tmp = work / 'imgs' / '_nuevos'
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir(parents=True, exist_ok=True)
            try:
                (bp.fetch_pnoa_imgs if img_src == 'pnoa' else bp.fetch_esri_imgs)(nuevos, tmp)
                bp.compute_img_features(nuevos, tmp)
                for i, r in enumerate(nuevos):
                    p = tmp / f'pt_{i:04d}.jpg'
                    if p.exists():
                        destino = work / 'imgs' / f"{r['ID']}.jpg"
                        shutil.move(str(p), str(destino))
                        r['_img'] = destino.name
            except Exception as e:
                fallos.append(i18n.pick(f'imagen aérea ({e})', f'aerial image ({e})'))
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
        # Distancia al río o arroyo más cercano (OpenStreetMap), para el ambiente
        try:
            _lin, _ = bp.cauces_osm(bbox, APP_DIR / bp.CAUCES_SN)
            bp.attach_cauce_distance(nuevos, _lin)
            if _lin is None:
                fallos.append(i18n.pick('cauces de OpenStreetMap (sin conexión)',
                                        'OpenStreetMap watercourses (no connection)'))
        except Exception as e:
            fallos.append(i18n.pick(f'cauces de OpenStreetMap ({e})',
                                    f'OpenStreetMap watercourses ({e})'))
        try:
            if backend == 'gee':
                geb.fetch_topo_gee(nuevos, bbox)
            else:
                bp.fetch_topo(nuevos, bbox, work)
        except Exception as e:
            fallos.append(i18n.pick(f'topografía ({e})', f'topography ({e})'))
        if not skip_s2:
            try:
                if backend == 'gee':
                    geb.fetch_s2_gee(nuevos, bbox, years=years)
                else:
                    bp.fetch_s2(nuevos, bbox, years=years)
            except Exception as e:
                fallos.append(f'Sentinel-2 ({e})')
            if backend == 'gee':
                try:
                    geb.fetch_s1_gee(nuevos, bbox)
                except Exception as e:
                    fallos.append(f'Sentinel-1 ({e})')
                try:                     # para la pureza, como en el mapa
                    geb.fetch_pureza_gee(nuevos, years=years)
                except Exception as e:
                    fallos.append(i18n.pick(f'pureza, el píxel de 10 m ({e})',
                                            f'purity, the 10 m pixel ({e})'))
        if con_cir:
            try:
                bp.fetch_cir(nuevos, bbox)
            except Exception as e:
                fallos.append(f'CIR ({e})')
    try:
        _lag = APP_DIR / 'lagunas_sierra_nevada.kml'
        if _lag.exists():
            bp.attach_laguna_distance(nuevos, bp.read_points(str(_lag)))
    except Exception:
        pass
    return fallos


def _coverage(rows, feats):
    """Nº de puntos con AL MENOS un valor presente (no NaN, no vacío) en `feats`."""
    import math
    n = 0
    for r in rows:
        for f in feats:
            v = r.get(f)
            if v is None or v == '':
                continue
            try:
                if not math.isnan(float(v)):
                    n += 1
                    break
            except (TypeError, ValueError):
                n += 1  # valor no numérico (p. ej. mat_signature) cuenta como presente
                break
    return n


def render_extraction_report(rows, skip_s2=False, skip_imgs=False, skip_s1=False):
    """Informe SIEMPRE visible de qué fuentes se extrajeron y con qué cobertura.
    Resuelve el 'no sé qué ha funcionado': muestra puntos con dato por grupo y
    avisa con causa probable cuando una fuente crítica quedó vacía. Las fuentes
    omitidas a propósito (casillas «Saltar…», o el radar sin Earth Engine) se marcan
    como omitidas, sin alarma."""
    N = len(rows)
    if not N:
        return
    img_feats = list(bp.FEATURES_RF[:12])
    topo_feats = ['elev_dem_m', 'slope_deg', 'curvature']
    s1_feats = list(bp.S1_FEATS)
    s2_feats = [f for f in bp.FEATURES_RF
                if f not in img_feats and f not in topo_feats and f not in s1_feats]
    # (etiqueta, features, id, ¿omitida a propósito?)
    groups = [('🖼️ Imagen aérea (textura/patrón)', img_feats, 'img', skip_imgs),
              ('⛰️ Topografía (DEM)', topo_feats, 'topo', False),
              ('🛰️ Óptico Sentinel-2 (NDVI, Clre…)', s2_feats, 's2', skip_s2),
              ('📡 Radar Sentinel-1', s1_feats, 's1', skip_s2 or skip_s1)]
    sel_feats = st.session_state.get('sel_model_feats', [])
    if any(f.startswith('cir_') for f in sel_feats):
        groups.append(('🇪🇸 PNOA Falso Color IR (CIR)', list(bp.CIR_FEATS), 'cir', False))
    if any(f.startswith('ps_') for f in sel_feats):
        groups.append(('🌍 PlanetScope 3 m', list(bp.PS_FEATS), 'ps', False))

    table = []
    problems = []
    for name, feats, gid, skipped in groups:
        if skipped:
            _por_que = ''
            if gid == 's1' and not skip_s2:
                _por_que = i18n.pick(' (solo con Earth Engine)', ' (Earth Engine only)')
            table.append({'Fuente': name, 'Puntos con dato': '—',
                          '%': '—', 'Estado': i18n.pick('⚪ omitida', '⚪ skipped')
                          + _por_que})
            continue
        cov = _coverage(rows, feats)
        pct = 100 * cov / N
        flag = '🟢' if pct >= 80 else ('🟡' if pct >= 30 else '🔴')
        table.append({'Fuente': name, 'Puntos con dato': f'{cov}/{N}',
                      '%': f'{pct:.0f}%', 'Estado': flag})
        if pct < 30:
            problems.append((name, cov))

    crit_empty = (not skip_s2) and any('Sentinel-2' in n and c == 0 for n, c in problems)
    if not problems:
        st.success(f'🔍 **Diagnóstico de extracción** — todas las fuentes con buena '
                   f'cobertura sobre los {N} puntos.')
    else:
        st.error('🔍 **Diagnóstico de extracción** — alguna fuente quedó casi vacía '
                 '(filas 🔴). Las variables que falten se rellenan con la mediana, así '
                 'que la predicción puede ser poco fiable. Causas y solución abajo.')
    st.table(table)

    if crit_empty:
        _backend = st.session_state.get('active_backend', '?')
        st.warning(
            '**Sentinel-2 quedó sin datos en (casi) todos los puntos.** Es la causa de '
            'que NDVI/Clre salgan vacíos y de que no se detecten borreguiles. '
            'Causa más probable y solución:\n\n'
            f'- **Backend Google Earth Engine con muchos puntos** (usaste: '
            f'`{_backend.upper()}`). Earth Engine corta la consulta por límite de '
            'tiempo/memoria y, antes, el fallo no se mostraba. → **Reduce el nº de '
            'puntos** (p. ej. 200–300 por ejecución) **o cambia el backend a Microsoft '
            'Microsoft Planetary Computer (MPC)** en «1 · Analizar», «Datos de satélite».\n'
            '- Caída temporal del servicio satelital → **reintenta** en unos minutos.\n'
            '- Años sin escenas con poca nube → amplía el **rango de años**.')
    elif problems:
        st.caption(i18n.pick(
            'Sugerencia: reintenta (caídas temporales de los servicios), reduce el nº de '
            'puntos, o cambia la fuente de datos (Earth Engine ↔ Microsoft Planetary Computer) en '
            '«1 · Analizar», «Datos de satélite».',
            'Tip: retry (temporary service outages), reduce the number of points, or switch '
            'the data source (Earth Engine ↔ Microsoft Planetary Computer) in "1 · Analyse", '
            '"Satellite data".'))


# Lo que admiten los recuadros de puntos: un Shapefile son varios ficheros, o un .zip
FORMATOS_PUNTOS = list(bp.FORMATOS_PUNTOS)


def _guardar_subida(ficheros):
    """bp.guardar_subida con los avisos en el idioma de la app."""
    return bp.guardar_subida(ficheros, en=i18n.get_lang() == 'en')


class _Subido:
    """Un fichero subido, reducido a su nombre y su contenido (para la caché)."""
    def __init__(self, name, datos):
        self.name, self._datos = name, datos

    def getvalue(self):
        return self._datos


def _resumen_campo(info):
    """[(nivel, texto)] de lo que bp.leer_puntos_de_campo() ha leído del fichero de
    puntos de campo: como resumen_etiquetas(), pero sin columna de etiqueta cada punto
    es un borreguil, y se avisa de los puntos que no se usan."""
    if info.get('solo_presencias'):
        return [('info', i18n.pick(
            'El fichero de puntos de campo no trae ninguna columna que diga si hay '
            'borreguil: cada punto se toma como borreguil. Si incluye ausencias, ponlas '
            'en una columna `presence` (1 = borreguil, 0 = no).',
            'The field-points file has no column saying whether there is borreguil: '
            'every point is taken as a borreguil. If it includes absences, put them in a '
            '`presence` column (1 = borreguil, 0 = no).'))]
    out = bp.resumen_etiquetas(info)
    if info.get('n_sin_usar'):
        out.append(('aviso', i18n.pick(
            f"{info['n_sin_usar']} puntos de campo no tienen una etiqueta que se entienda "
            'y no se usan.',
            f"{info['n_sin_usar']} field points have no label that can be understood and "
            'are not used.')))
    return out


def _etiquetas_de_subida(ficheros, campo=False):
    """Lo que leer_etiquetas() leería de unos puntos recién subidos (con `campo`, el
    fichero de puntos de campo, leído con leer_puntos_de_campo()), para avisar antes
    de ejecutar; si falta algo (un .shp sin su .dbf), el aviso de qué hacer. None si
    no se puede leer aquí: al ejecutar ya se verá."""
    if not ficheros:
        return None
    if not isinstance(ficheros, (list, tuple)):
        ficheros = [ficheros]
    return _etiquetas_de_partes(tuple((f.name, f.getvalue()) for f in ficheros), campo,
                                i18n.get_lang())


def _linea_ok(texto):
    """Lo leído de un fichero, en una línea verde clara, como en la demo."""
    import html as _html
    st.markdown('<p class="leido-ok">' + _html.escape(texto).replace('`', '') + '</p>',
                unsafe_allow_html=True)


def _linea_fichero(nombre, n, info, idioma):
    """Como en la demo: «puntos.geojson · 20 puntos · 12 borreguiles y 8 sin borreguil
    (columna «presence»)»."""
    def _pl(k, uno, varios):
        return f'{k} {uno if k == 1 else varios}'
    if idioma == 'en':
        txt = (f"{nombre} · {_pl(n, 'point', 'points')} · "
               f"{_pl(info['n_si'], 'borreguil', 'borreguiles')} and {info['n_no']} without "
               f"borreguil («{info['columna']}» column)")
        if info.get('n_sin'):
            txt += f" · {info['n_sin']} unlabelled"
        if info.get('duda'):
            txt += f" · {_pl(info.get('n_duda', 0), 'doubtful', 'doubtful')}"
        return txt
    txt = (f"{nombre} · {_pl(n, 'punto', 'puntos')} · "
           f"{_pl(info['n_si'], 'borreguil', 'borreguiles')} y {info['n_no']} sin borreguil "
           f"(columna «{info['columna']}»)")
    if info.get('n_sin'):
        txt += f" · {info['n_sin']} sin etiqueta"
    if info.get('duda'):
        txt += f" · {_pl(info.get('n_duda', 0), 'dudoso', 'dudosos')}"
    return txt


@st.cache_data(show_spinner=False)
def _etiquetas_de_partes(partes, campo=False, idioma='es'):
    ruta = None
    try:
        ruta = _guardar_subida([_Subido(n, d) for n, d in partes])
        if campo:
            return _resumen_campo(bp.leer_puntos_de_campo(bp.read_points(ruta)))
        _filas = bp.read_points(ruta)
        _info = bp.leer_etiquetas(_filas)
        _res = bp.resumen_etiquetas(_info)
        # La primera línea, como en la demo (en vez de «Etiquetas de la columna…»).
        if _res and _res[0][0] == 'ok' and _info.get('columna'):
            _nombre = next((n for n, _ in partes if n.lower().endswith('.shp')), partes[0][0])
            _res[0] = ('ok', _linea_fichero(_nombre, len(_filas), _info, idioma))
        return _res
    except ValueError as e:                  # falta algo, o el fichero no se entiende
        return [('falta', str(e))]
    except Exception:
        return None
    finally:
        if ruta:
            shutil.rmtree(os.path.dirname(ruta), ignore_errors=True)


@st.cache_data(show_spinner=False)
@st.cache_data(show_spinner=False)
def _sin_crs_subida(partes, idioma='es'):
    """El aviso si la capa subida (`partes`: sus ficheros, (nombre, bytes)) no tiene
    sistema de referencia (datum); '' si lo tiene o no hace falta (KML, GeoJSON, CSV)."""
    carpeta = Path(tempfile.mkdtemp(prefix='capa_'))
    try:
        rutas = []
        for n, d in partes:
            r = carpeta / Path(n).name
            r.write_bytes(d)
            rutas.append(r)
        principal = next((r for e in ('.shp', '.zip', '.gpkg') for r in rutas
                          if r.suffix.lower() == e), None)
        if principal is not None and bp.falta_sistema_de_referencia(principal):
            return bp.aviso_sin_sistema_de_referencia(principal.name, idioma == 'en')
        return ''
    finally:
        shutil.rmtree(carpeta, ignore_errors=True)


def _leer_capa_cuartiles(datos, nombre):
    """La capa de polígonos para lanzar puntos por cuartiles (random_points.leer_capa),
    leída una sola vez por fichero. Devuelve (capa, error)."""
    import random_points
    carpeta = Path(tempfile.mkdtemp(prefix='capa_cuartiles_'))
    try:
        ruta = carpeta / Path(nombre).name
        ruta.write_bytes(datos)
        if bp.falta_sistema_de_referencia(ruta):
            return None, bp.aviso_sin_sistema_de_referencia(ruta.name,
                                                            i18n.get_lang() == 'en')
        return random_points.leer_capa(ruta), None
    except ValueError as e:
        return None, str(e)
    except Exception as e:                               # noqa: BLE001
        return None, f'{type(e).__name__}: {e}'
    finally:
        shutil.rmtree(carpeta, ignore_errors=True)


# ── Perfil espectral por punto vs. referencia de borreguiles ─────────────────
# Las gráficas viven en point_profile.py (puro altair/numpy/pandas, testeable sin la app).
@st.cache_data(show_spinner=False)
def _ref_stats_from_csv():
    """Fallback: cuantiles por clase desde classification_v7.csv si está local (la v5
    no sirve: su EVI va en otra escala)."""
    return pp.ref_stats_from_csv([APP_DIR / 'classification_v7.csv',
                                  APP_DIR.parent / 'classification_v7.csv'])


@st.cache_data(show_spinner=False)
def load_ref_stats(model_path):
    """ref_stats (cuantiles por clase) + importancias del modelo; fallback a CSV; None si nada."""
    try:
        b = bp.load_model_bundle(model_path)
        rs = b.get('ref_stats')
        if rs and rs.get('features'):
            rs = dict(rs)
            rs['importances'] = b.get('importances')
            rs['model_features'] = b.get('features')
            return rs
    except Exception:
        pass
    return _ref_stats_from_csv()


def render_point_profile(sel, ref_stats):
    """«Este punto frente a los borreguiles verdaderos», como en la demo (Revisar.dc.html):
    una tarjeta que se pliega, abierta de entrada, con la gráfica del punto en HTML
    (point_profile.html_punto_vs_borreguiles: dónde cae entre los borreguiles verdaderos
    en las variables que más pesan en el modelo), su tabla de valores y «Cómo se lee esta
    gráfica y para qué sirve». Los estilos (.rev-perfil, .rev-leer) van con los de
    2 · Revisar."""
    import html as _html
    titulo = i18n.pick('Este punto frente a los borreguiles verdaderos',
                       'This point vs the real borreguiles')
    if not ref_stats or not ref_stats.get('features'):
        vals = ''.join(
            f'<span>{_html.escape(pp.etiqueta(k["feat"]))}: '
            f'<span style="font-weight:700">{_num(sel.get(k["feat"]))}</span></span>'
            for k in pp.variables_clave(None))
        cuerpo = ('<span style="line-height:1.5">' + i18n.pick(
            'El modelo elegido no trae los valores de los borreguiles verdaderos, así que '
            'no se puede comparar el punto con ellos. Estos son sus valores en las '
            'variables que más separan en Sierra Nevada.',
            'The chosen model does not carry the values of the real borreguiles, so the '
            'point cannot be compared with them. These are its values for the variables '
            'that separate best in Sierra Nevada.') + '</span>'
            '<div style="display:flex;flex-wrap:wrap;gap:6px 18px">' + vals + '</div>')
    else:
        _ch, recs = pp.chart_punto_vs_borreguiles(sel, ref_stats, n=8)
        if not recs:
            cuerpo = ('<span>' + i18n.pick(
                'Sin variables con valores de referencia para este punto.',
                'No variables with reference values for this point.') + '</span>')
        else:
            cuerpo = pp.html_punto_vs_borreguiles(recs, ref_stats.get('n_si', '?'))
    st.html('<details class="rev-perfil" open><summary>' + titulo + '</summary>'
            '<div class="rev-perfil-cuerpo">' + cuerpo + '</div></details>')


def _ctx_actual():
    """Contexto de Streamlit del hilo actual: el de la ejecución de la página que corre
    en él, o None en un hilo auxiliar (los de un ThreadPool no lo tienen y no deben
    tocar la página)."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        # sin aviso: preguntarlo desde un hilo auxiliar es normal aquí
        return get_script_run_ctx(suppress_warning=True)
    except Exception:
        return None


class _StatusWriter(io.TextIOBase):
    """Redirige los print() del pipeline a una 'consola' en vivo (st.code) dentro del
    st.status, mostrando las últimas líneas. Si no, el paso parece congelado.
    Solo toca la página desde la MISMA ejecución que la creó: si se para un análisis y
    empieza otro, escribir aquí desde el nuevo apuntaría a un hueco de la página que ya
    no existe, y la página se quedaba en blanco."""
    def __init__(self, container, max_lines=16):
        self.slot = container.empty()
        self.ctx = _ctx_actual()
        self.lines = []
        self.buf = ''
        self.max_lines = max_lines

    def write(self, s):
        self.buf += s
        updated = False
        while '\n' in self.buf:
            line, self.buf = self.buf.split('\n', 1)
            if line.strip():
                self.lines.append(line.rstrip())
                self.lines = self.lines[-self.max_lines:]
                updated = True
        if updated and self.ctx is not None and _ctx_actual() is self.ctx:
            try:
                self.slot.code('\n'.join(self.lines), language=None)
            except Exception:
                pass
        return len(s)

    def flush(self):
        return None


class _SalidaPorHilo(io.TextIOBase):
    """sys.stdout / sys.stderr de todo el programa, que reparte lo que se escribe según
    el hilo: cada análisis, en la consola del bloque live_logs abierto en SU hilo; los
    hilos auxiliares (sin contexto de Streamlit), en la última consola abierta, que solo
    acumula sus líneas; el resto, en la salida de siempre.
    Antes, live_logs cambiaba sys.stdout para todo el programa y lo devolvía al acabar.
    Si se paraba un análisis y empezaba otro mientras el primero seguía esperando a un
    servidor, cada uno dejaba puesta al acabar la consola del otro: el análisis nuevo
    escribía en huecos de la página del viejo y la página se quedaba en blanco."""
    _es_salida_por_hilo = True

    def __init__(self, original):
        self.original = original
        self.hilos = threading.local()
        self.ultima = None

    def _destino(self):
        w = getattr(self.hilos, 'consola', None)
        if w is None and _ctx_actual() is None:
            w = self.ultima
        return w if w is not None else self.original

    def write(self, s):
        destino = self._destino()
        if destino is not self.original:   # lo del análisis, también al registro
            registros.anotar(s)
        return destino.write(s)

    def flush(self):
        try:
            return self._destino().flush()
        except Exception:
            return None

    @property
    def encoding(self):
        return getattr(self.original, 'encoding', 'utf-8')

    def isatty(self):
        try:
            return self.original.isatty()
        except Exception:
            return False

    def fileno(self):
        return self.original.fileno()


def _salidas():
    """Pone (una vez) el reparto por hilos en sys.stdout y sys.stderr y los devuelve."""
    for nombre in ('stdout', 'stderr'):
        actual = getattr(sys, nombre)
        if not getattr(actual, '_es_salida_por_hilo', False):
            setattr(sys, nombre, _SalidaPorHilo(actual))
    return sys.stdout, sys.stderr


from contextlib import contextmanager

@contextmanager
def live_logs(container):
    """Context manager: dentro del bloque, los print() del pipeline (los de este hilo y
    los de sus hilos auxiliares) se ven en vivo en una consola del st.status."""
    w = _StatusWriter(container)
    salidas = _salidas()
    antes = [getattr(x.hilos, 'consola', None) for x in salidas]
    for x in salidas:
        x.hilos.consola = w
        x.ultima = w
    try:
        yield w
    finally:
        for x, previa in zip(salidas, antes):
            x.hilos.consola = previa
            if x.ultima is w:
                x.ultima = previa


# ============================================================
# UI
# ============================================================
st.set_page_config(page_title='Borreguil · Wet Meadow Finder',
                    page_icon='🌿',
                    layout='wide',
                    initial_sidebar_state='collapsed')

# --- Tema oscuro («C · Sobria», elegido por la autora el 26-11-2026) -----------------
# Gris pizarra y verde bosque, con Inter y Montserrat. El ejecutable no aplica ningún
# tema (las STREAMLIT_THEME_* del lanzador no llegan a Streamlit y .streamlit/config.toml
# no viaja con él), así que lo pone la app: llega también con la actualización automática,
# sin instalador nuevo. Streamlit manda el tema al navegador al empezar cada pasada, así
# que la primera vez que se pone hay que repetir la pasada (una sola vez por sesión).
_FUENTE = ('Inter:https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700'
           '&display=swap, system-ui, sans-serif')
_FUENTE_TITULOS = ('Montserrat:https://fonts.googleapis.com/css2?family=Montserrat:'
                   'wght@500;600;700&display=swap, Inter, system-ui, sans-serif')
TEMA = {
    'theme.base': 'dark',
    'theme.primaryColor': '#7CC496',             # casillas, deslizadores, barras (demo)
    'theme.backgroundColor': '#221F20',          # fondo
    'theme.secondaryBackgroundColor': '#2E2A2B',  # tarjetas y cajas
    'theme.textColor': '#ECEFF1',
    'theme.linkColor': '#7CC496',
    'theme.borderColor': '#494245',
    'theme.dataframeBorderColor': '#403A3C',
    'theme.dataframeHeaderBackgroundColor': '#282526',
    'theme.codeBackgroundColor': '#282526',
    'theme.greenColor': '#43A047',               # borreguil
    'theme.redColor': '#E53935',                 # no borreguil
    'theme.font': _FUENTE,
    'theme.headingFont': _FUENTE_TITULOS,
    'theme.baseRadius': '10px',
    'theme.buttonRadius': '12px',
    'theme.showSidebarBorder': True,
    'theme.sidebar.backgroundColor': '#342F31',  # el panel, más claro que el fondo
    'theme.sidebar.secondaryBackgroundColor': '#3D383A',
    'theme.sidebar.borderColor': '#554D50',
}


def _poner_tema():
    """Pone el tema oscuro en la configuración de este proceso. True si ha cambiado algo
    (entonces la pasada en curso aún se ha enviado con el tema de antes)."""
    from streamlit import config as _cfg
    cambiado = False
    for clave, valor in TEMA.items():
        try:
            if _cfg.get_option(clave) != valor:
                _cfg.set_option(clave, valor)
                cambiado = True
        except Exception:                                   # noqa: BLE001
            pass                                            # una opción que no exista
    return cambiado


if _poner_tema() and not st.session_state.get('_tema_puesto'):
    st.session_state['_tema_puesto'] = True
    st.rerun()

# Lo que el tema no cubre: el panel lateral con sombra (la autora: «se confunde con la
# pantalla principal»), la letra blanca de los botones verdes, las cifras grandes en
# Montserrat y las tarjetas del Inicio.
st.html('<style>' + '''
section[data-testid="stSidebar"]{box-shadow:6px 0 18px rgba(0,0,0,.35)}
button[data-testid^="stBaseButton-primary"]{background-color:#2A7048;border-color:#2A7048}
button[data-testid^="stBaseButton-primary"]:hover{background-color:#24613E;border-color:#24613E}
button[data-testid^="stBaseButton-primary"],
button[data-testid^="stBaseButton-primary"] p{color:#FFFFFF;font-weight:700}
/* Las «?» de Streamlit, como las de la demo: un círculo de 22 px con «?». */
[data-testid="stTooltipIcon"] button:not([data-testid]){width:22px;height:22px;min-height:0;border-radius:50%;
  border:1px solid #60575A;background:#393335;display:inline-flex;align-items:center;
  justify-content:center;padding:0;margin-left:4px}
[data-testid="stTooltipIcon"] button:not([data-testid]) svg{display:none}
[data-testid="stTooltipIcon"] button:not([data-testid])::after{content:"?";font:700 12px/1 Inter,system-ui,sans-serif;
  color:#CDD3D9}
[data-testid="stTooltipIcon"] button:not([data-testid]):hover{border-color:#7CC496}
[data-testid="stTooltipIcon"] button:not([data-testid]):hover::after{color:#7CC496}
[data-testid="stTooltipContent"]{font-size:13px;line-height:1.5;color:#CDD3D9;
  background:#292627;border:1px solid #3E6B4F;border-radius:10px}
[data-testid="stHeaderActionElements"] a{display:none}
/* Casillas, en el verde claro de la demo. */
label[data-baseweb="checkbox"]:has(input:checked)>span{background-color:#7CC496;
  border-color:#7CC496}
.separador{border:0;border-top:1px solid #403A3C;margin:4px 0}
.leido-ok{font-size:13px;color:#A3D9B5;margin:0}
.ayuda-aviso{font-size:12px;color:#F5B83D}
/* 1 · Analizar, como en la demo: etiquetas de 13 px, títulos con su «?», enlaces verdes. */
.st-key-panel_analizar [data-testid="stWidgetLabel"] p{font-size:13px;color:#CDD3D9}
.st-key-subir_campo [data-testid="stWidgetLabel"] p{font-weight:600;color:#ECEFF1}
[class*="st-key-tit_"] p{font-size:13px;color:#CDD3D9;margin:0}
[class*="st-key-titb_"] p{font-size:13px;font-weight:600;color:#ECEFF1;margin:0}
.st-key-cab_puntos{justify-content:space-between}
.st-key-cab_puntos h3{padding:0 !important}
.st-key-plantilla_panel button,.st-key-ir_puntos_campo button,.st-key-ir_modelos button{
  padding:0;min-height:0}
.st-key-plantilla_panel button p,.st-key-ir_puntos_campo button p,
.st-key-ir_modelos button p{font-size:13px;color:#7CC496;text-decoration:underline;
  text-underline-offset:3px;white-space:nowrap}
.st-key-nota_modelo{flex-wrap:wrap;gap:2px 6px}
.st-key-nota_modelo [data-testid="stCaptionContainer"]{width:auto}
.st-key-sugerencias_panel details{border:1px solid #6A6063;background:#484143;
  border-radius:12px}
.st-key-sugerencias_panel summary p{font-size:14px;font-weight:600}
.st-key-tit_principal h3{font-size:22px !important;font-weight:700 !important;padding:0 !important}
.st-key-tit_principal [data-testid="stCaptionContainer"] p{font-size:13px;color:#AEB6BE}
[data-testid="stColumn"]:has(.st-key-analizar) [data-testid="stCaptionContainer"]{
  text-align:center}
.st-key-origen_puntos [role="radiogroup"],.st-key-zona_modo [role="radiogroup"],
.st-key-imagen_ambito [role="radiogroup"]{display:grid !important;
  grid-template-columns:repeat(2,minmax(0,1fr))}
.st-key-cuenta [role="radiogroup"],.st-key-modo_umbral [role="radiogroup"]{
  display:grid !important;grid-template-columns:repeat(3,minmax(0,1fr))}
.st-key-origen_puntos label[data-baseweb="radio"],.st-key-zona_modo label[data-baseweb="radio"],
.st-key-imagen_ambito label[data-baseweb="radio"],.st-key-cuenta label[data-baseweb="radio"],
.st-key-modo_umbral label[data-baseweb="radio"]{min-width:0 !important;text-align:center}
.st-key-origen_puntos,.st-key-zona_modo,.st-key-imagen_ambito,.st-key-cuenta,.st-key-modo_umbral,
.st-key-origen_puntos>div,.st-key-zona_modo>div,.st-key-imagen_ambito>div,.st-key-cuenta>div,
.st-key-modo_umbral>div,.st-key-origen_puntos [role="radiogroup"],
.st-key-zona_modo [role="radiogroup"],.st-key-imagen_ambito [role="radiogroup"],
.st-key-cuenta [role="radiogroup"],.st-key-modo_umbral [role="radiogroup"]{width:100%}
[data-testid="stMetricValue"]{font-family:Montserrat,Inter,system-ui,sans-serif}
[data-testid="stMetric"]{background:#2E2A2B;border:1px solid #403A3C;border-radius:16px;padding:14px 18px}
[class*="st-key-tarjeta_"]{background:#2E2A2B;border:1px solid #403A3C;
  border-radius:16px;padding:18px 20px}
[class*="st-key-tarjeta_"] h4{color:#7CC496;font-size:15px}
/* Las tres tarjetas del Inicio, con la letra y el espacio de la demo. */
[class*="st-key-tarjeta_inicio_"]{padding:20px !important;gap:8px !important;height:100%;
  box-sizing:border-box}
/* las tres, igual de altas aunque una tenga más texto */
[data-testid="stLayoutWrapper"]:has(>[class*="st-key-tarjeta_inicio_"]){flex:1 1 auto}
[class*="st-key-tarjeta_inicio_"] h4{font-weight:700 !important;padding:0 !important;
  line-height:1.3 !important}
[class*="st-key-tarjeta_inicio_"] p{margin:0 !important}
[class*="st-key-tarjeta_inicio_"] [data-testid="stMarkdownContainer"]{margin:0 !important}
[class*="st-key-tarjeta_inicio_"] [data-testid="stMarkdownContainer"] p{font-size:16px;
  color:#F5F7F8}
[class*="st-key-tarjeta_inicio_"] strong{font-weight:600}
[class*="st-key-tarjeta_inicio_"] [data-testid="stCaptionContainer"],
[class*="st-key-tarjeta_inicio_"] [data-testid="stCaptionContainer"] p{font-size:14px !important;
  line-height:1.5 !important;color:#AEB6BE !important;opacity:1}
div.inicio .ceja{display:inline-block;font-size:12px;font-weight:600;letter-spacing:.08em;
  text-transform:uppercase;color:#7CC496;background:#1E3327;border:1px solid #3E6B4F;
  border-radius:999px;padding:4px 12px}
div.inicio h1 [data-testid="stHeaderActionElements"]{display:none}
div.inicio h1{font-size:clamp(28px,3.1vw,40px) !important;line-height:1.08 !important;letter-spacing:-.02em;
  color:#F5F7F8;margin:16px 0 12px !important;padding:0 !important}
div.inicio .entrada{font-size:18px;line-height:1.55;color:#B9C1C8;margin:0 0 14px}
div.inicio a.obsnev{display:inline-block;color:#7CC496;font-size:15px;font-weight:600;
  text-decoration:none;border-bottom:1px solid #3E6B4F;margin-bottom:18px}
div.inicio a.obsnev:hover{color:#A3D9B5}
.st-key-panel_analizar{background:#342F31;border:1px solid #534B4E;border-radius:16px;
  padding:16px;box-shadow:6px 0 18px rgba(0,0,0,.35)}
[class*="st-key-seccion_"]{background:#3D383A;border:1px solid #554D50;
  border-radius:14px;padding:14px 16px}
[class*="st-key-seccion_"] h3{font-family:Inter,system-ui,sans-serif !important;font-size:15px !important;
  font-weight:700 !important;padding:0 0 2px !important}
[class*="st-key-tarjeta_"] h3{font-size:17px !important}
.mini{font-size:11px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;
  color:#98A2AB;margin-top:6px}
/* Opciones cortas como botones segmentados (la demo), sin cambiar los radios. */
.st-key-panel_analizar [data-testid="stRadio"] [role="radiogroup"]{gap:6px;flex-wrap:wrap}
.st-key-panel_analizar [data-testid="stRadio"] label[data-baseweb="radio"]{
  flex:1 1 0;min-width:max-content;justify-content:center;margin:0;min-height:44px;
  padding:8px 12px;border:1px solid #62585B;background:#484143;border-radius:10px}
.st-key-panel_analizar [data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked){
  border:1.5px solid #7CC496;background:#244A33}
.st-key-panel_analizar [data-testid="stRadio"] label[data-baseweb="radio"]>div:first-of-type{
  display:none}
.st-key-panel_analizar [data-testid="stRadio"] label[data-baseweb="radio"] p{
  font-size:13px;font-weight:500;color:#CDD3D9}
.st-key-panel_analizar [data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) p{
  font-weight:600;color:#ECEFF1}
.st-key-ir_inicio button p{white-space:nowrap}
/* Ayuda, como en la demo. */
.ayuda-tarjetas{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}
@media (max-width:760px){.ayuda-tarjetas{grid-template-columns:1fr}}
.ayuda-tarjeta{text-decoration:none;background:#2E2A2B;border:1px solid #403A3C;
  border-radius:16px;padding:18px;display:flex;flex-direction:column;gap:6px;color:#ECEFF1}
.ayuda-tarjeta:hover{border-color:#3E6B4F}
.ayuda-tarjeta .t{font-weight:700;font-size:16px;color:#ECEFF1}
.ayuda-tarjeta .d{font-size:13px;color:#AEB6BE}
.ayuda-caja,.ayuda-sug,.ayuda-inst{display:flex;flex-direction:column;gap:10px;
  font-size:14px;color:#CDD3D9}
.ayuda-caja{background:#2E2A2B;border:1px solid #403A3C;border-radius:16px;padding:20px}
.ayuda-caja .t,.ayuda-sug .t,.ayuda-inst .t{font-weight:700;font-size:16px;color:#ECEFF1}
.ayuda-caja table{border-collapse:collapse;font-size:13px;min-width:560px}
.ayuda-caja th{color:#AEB6BE;text-align:left;font-weight:500;padding:6px 16px 6px 0}
.ayuda-caja td{border-top:1px solid #403A3C;padding:6px 16px 6px 0;vertical-align:top}
/* la tabla de modelos: con radar y sin radar, cada uno en su línea */
.tabla-modelos{overflow-x:auto}
@media (min-width:760px){.tabla-modelos{overflow:visible}}
.ayuda-caja td.r{color:#AEB6BE;white-space:nowrap}
.ayuda-caja tr.sin td{border-top:0;padding-top:0}
.ayuda-caja td strong{color:#ECEFF1;font-weight:600}
/* una «?» como las de ayuda_q, dentro de un texto: el globo sale al pasar o al pinchar */
.q-html{position:relative;display:inline-flex;width:22px;height:22px;border-radius:50%;
  border:1px solid #60575A;background:#393335;color:#CDD3D9;font-size:12px;font-weight:700;
  align-items:center;justify-content:center;cursor:pointer;vertical-align:middle;
  margin-left:4px;outline:none}
.q-html:hover,.q-html:focus{border-color:#7CC496;color:#7CC496}
.q-html .globo{display:none;position:absolute;top:28px;left:-10px;z-index:50;width:30em;
  max-width:80vw;background:#282526;border:1px solid #3E6B4F;border-radius:8px;
  padding:10px 12px;font-size:13px;font-weight:400;line-height:1.5;color:#ECEFF1;
  text-align:left;white-space:normal;cursor:auto;box-shadow:0 8px 24px rgba(0,0,0,.4)}
.q-html:hover .globo,.q-html:focus .globo,.q-html:focus-within .globo{display:block}
.ayuda-inst{gap:8px}
.ayuda-inst .nota{font-size:13px;color:#AEB6BE}
.ayuda-sug .enlace{display:block;border:1px solid #60575A;background:#393335;
  border-radius:12px;padding:12px 14px;text-decoration:none;color:#ECEFF1}
.ayuda-sug .enlace:hover{border-color:#7CC496}
.ayuda-sug .enlace span{font-size:13px;color:#AEB6BE}
.ayuda-aviso{font-size:12px;color:#F5B83D}
.st-key-tarjeta_ayuda_sugerir,.st-key-tarjeta_ayuda_instalacion{background:#2E2A2B;
  border:1px solid #403A3C;border-radius:16px;padding:20px;height:100%}
.st-key-registros_documentacion button{display:block;width:100%;text-align:left;
  border:1px solid #60575A;background:#393335;border-radius:12px;padding:12px 14px}
.st-key-registros_documentacion button:hover{border-color:#7CC496}
.st-key-registros_documentacion button p{margin:0;text-align:left;color:#ECEFF1;font-size:14px}
.st-key-registros_documentacion button p+p{font-size:13px;color:#AEB6BE;font-weight:400}
/* Las «?» de la demo (ayuda_q): un botón redondo de 22 px que abre el texto. */
[class*="st-key-q_"] button{width:22px;height:22px;min-height:0;padding:0;border-radius:50%;
  border:1px solid #60575A;background:#393335;justify-content:center}
[class*="st-key-q_"] button p{font-size:12px;font-weight:700;color:#CDD3D9;margin:0}
[class*="st-key-q_"] button:hover{border-color:#7CC496}
[class*="st-key-q_"] button:hover p{color:#7CC496}
[class*="st-key-q_"] button [data-testid="stIconMaterial"]{display:none}
/* Sin el hueco de la flecha de Streamlit, la «?» queda en el centro del círculo. */
[class*="st-key-q_"] button>div{margin:0 !important;width:100%;justify-content:center}
[class*="st-key-q_"] button>div>div:last-child:not(:first-child){display:none}
[class*="st-key-q_"] button p{line-height:1}
[data-testid="stPopoverBody"]{max-width:34em;font-size:13px;line-height:1.5;
  background:#282526;border:1px solid #3E6B4F}
/* Los dos botones grandes del Inicio, de dos líneas y con el texto a la izquierda. */
.st-key-botones_inicio{flex-wrap:wrap;gap:12px}
/* la «?» del ejemplo, siempre al lado del botón (aunque la columna sea estrecha) */
.st-key-fila_ejemplo{flex-wrap:nowrap}
.st-key-fila_ejemplo>div:first-child{min-width:0;flex:0 1 auto}
.st-key-probar_ejemplo button,.st-key-analizar_mis_puntos button{min-height:64px;
  padding:14px 22px;border-radius:12px;justify-content:flex-start}
.st-key-probar_ejemplo button p,.st-key-analizar_mis_puntos button p{margin:0;
  font-size:16px;line-height:1.35;text-align:left}
.st-key-probar_ejemplo [data-testid="stMarkdownContainer"],
.st-key-analizar_mis_puntos [data-testid="stMarkdownContainer"]{text-align:left}
.st-key-bajar_plantilla button,.st-key-bajar_ejemplo button{padding:0;min-height:0}
.st-key-bajar_plantilla button p,.st-key-bajar_ejemplo button p{font-size:14px;
  color:#7CC496;text-decoration:underline;text-underline-offset:3px}
.st-key-bajar_ejemplo{margin-bottom:10px}
.st-key-probar_ejemplo button p+p{font-size:13px;font-weight:400;color:#E2F0E6}
.st-key-analizar_mis_puntos button{background:#2E2A2B;border:1.5px solid #60575A}
.st-key-analizar_mis_puntos button p+p{font-size:13px;font-weight:400;color:#98A2AB}
/* Revisar: «¿Es borreguil?» y la certeza como botones grandes (la demo). */
[class*="st-key-rev_es_"] [role="radiogroup"],.st-key-rev_certidumbre [role="radiogroup"]{
  gap:6px;flex-wrap:wrap}
[class*="st-key-rev_es_"] label[data-baseweb="radio"],
.st-key-rev_certidumbre label[data-baseweb="radio"]{flex:1 1 0;justify-content:center;
  margin:0;min-height:46px;padding:8px 12px;border:1.5px solid #60575A;background:#393335;
  border-radius:12px}
[class*="st-key-rev_es_"] label[data-baseweb="radio"]:has(input:checked){
  background:#2A7048;border:2px solid #7CC496}
.st-key-rev_certidumbre label[data-baseweb="radio"]:has(input:checked){
  background:#1E3327;border:1.5px solid #7CC496}
[class*="st-key-rev_es_"] label[data-baseweb="radio"]>div:first-of-type,
.st-key-rev_certidumbre label[data-baseweb="radio"]>div:first-of-type{display:none}
[class*="st-key-rev_es_"] label[data-baseweb="radio"] p{font-size:15px;font-weight:700}
''' + '</style>')

# --- Idioma / Language ---------------------------------------------------
# El selector ES | EN está en la cabecera; aquí se lee lo elegido (de la pasada
# anterior) ANTES de instalar la traducción, que fija el idioma de todo lo demás.
if st.session_state.get('ui_lang') not in ('Español', 'English'):
    st.session_state['ui_lang'] = 'Español'
_lang_choice = st.session_state['ui_lang']
i18n.set_lang('en' if _lang_choice == 'English' else 'es')
i18n.install(st)   # traduce todos los st.* a partir de aquí (fallback a ES)
_vigilar_errores()  # cada st.error, al registro, y con «Enviar los registros» si es de Python
if st.session_state.pop('_abrir_registros', False):
    abrir_dialogo_registros()

# --- Modelos nuevos de GitHub ------------------------------------------------
# Una vez por arranque (antes del desplegable de modelos, para que ya salgan en él).
_modelos_gh = modelos_de_github()
if _modelos_gh.get('estado') == 'descargados' and not st.session_state.get('_aviso_gh'):
    st.session_state['_aviso_gh'] = True
    st.toast(i18n.pick('⬇️ Modelos nuevos de GitHub: ', '⬇️ New models from GitHub: ')
             + ', '.join(_modelos_gh['nuevos'] + _modelos_gh['actualizados'])
             + i18n.pick('. Ya están en «Modelo de partida».',
                         '. They are already under "Starting model".'))

# ============================================================
# CABECERA Y PANTALLAS (rediseño de la autora, 26-11-2026)
# ============================================================
# La app son pantallas por pasos, como en la demo: Inicio · 1 · Analizar · Resultados ·
# 2 · Revisar · 3 · Hacer el mapa · Ayuda. Por dentro son las pestañas de un st.tabs con
# key='pantalla' cuya barra se oculta: todo se sigue ejecutando en cada pasada, como
# antes, y la cabecera cambia de pantalla con st.session_state['pantalla'].
PANTALLAS = ('inicio', 'analizar', 'resultados', 'revisar', 'mapa', 'ayuda')
if st.session_state.get('_ir_a') in PANTALLAS:
    st.session_state['pantalla'] = st.session_state.pop('_ir_a')
if st.session_state.get('pantalla') not in PANTALLAS:
    st.session_state['pantalla'] = 'inicio'
PANTALLA = st.session_state['pantalla']

# Un análisis que no llegó al final. Un clic mientras se analiza vuelve a ejecutar la app
# desde arriba y el análisis se pierde sin avisar (27-11-2026, la autora: «si le doy al
# ejemplo del principio no se me abre revisar»: pulsaba «2 · Revisar» mientras se
# analizaba y la pantalla salía vacía). Por eso, mientras se analiza, la cabecera no
# responde (_CSS_BLOQUEO). Si aun así se corta (se cambia algo en el panel, «Stop», un
# fallo), al empezar la pasada siguiente sigue apuntado y se avisa.
if st.session_state.pop('_analisis_en_marcha', False):
    st.session_state['_aviso_cortado'] = True
_AVISO_CORTADO = None     # dónde está el aviso, para quitarlo si se vuelve a analizar
_BLOQUEO = None           # el estilo que bloquea la cabecera durante el análisis
_CSS_BLOQUEO = ('.st-key-cabecera button,.st-key-cabecera [data-testid="stButtonGroup"] *,'
                '.st-key-analizar button{pointer-events:none;cursor:default}'
                '.st-key-pasos,.st-key-cab_der,.st-key-analizar{opacity:.45}')


def ir_a(pantalla):
    """Para on_click: cambia de pantalla antes de la pasada siguiente."""
    st.session_state['pantalla'] = pantalla


def ir_a_luego(pantalla):
    """Desde el guion, con las pantallas ya dibujadas: en la pasada siguiente."""
    st.session_state['_ir_a'] = pantalla
    st.rerun()


def ayuda_q(key, texto_es, texto_en):
    """La «?» de la demo junto a un botón o un enlace (que no tienen la «?» de Streamlit):
    un botón redondo que, al pincharlo, abre el texto de ayuda."""
    with st.popover('?', key=key):
        st.markdown(i18n.pick(texto_es, texto_en))


# El logo de la demo (hierba verde sobre agua). st.html quita las <svg> en línea, así
# que va como imagen (data: URI).
LOGO_SVG = ('<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" '
            'viewBox="0 0 36 36">'
            '<rect width="36" height="36" rx="9" fill="#1E3327"></rect>'
            '<ellipse cx="18" cy="27" rx="12" ry="4" fill="#4FC3F7"></ellipse>'
            '<path d="M12 27 L9 10 M16 27 L15 6 M20 27 L22 8 M24 27 L28 12" '
            'stroke="#7CC496" stroke-width="2.5" stroke-linecap="round" fill="none"></path>'
            '</svg>')
LOGO_SRC = ('data:image/svg+xml;base64,'
            + __import__('base64').b64encode(LOGO_SVG.encode()).decode())

# Los pasos: pendiente, el actual (verde) o hecho (✓), como en la demo.
_PASOS = (('analizar', i18n.pick('1 · Analizar', '1 · Analyse')),
          ('revisar', i18n.pick('2 · Revisar', '2 · Review')),
          ('mapa', i18n.pick('3 · Hacer el mapa', '3 · Make the map')))
_ESTADO_PASOS = {
    'analizar': {'analizar': 'actual'},
    'resultados': {'analizar': 'hecho', 'revisar': 'actual'},
    'revisar': {'analizar': 'hecho', 'revisar': 'actual'},
    'mapa': {'analizar': 'hecho', 'revisar': 'hecho', 'mapa': 'actual'},
}.get(PANTALLA, {})
_DESTINO_PASOS = {'analizar': ('resultados' if PANTALLA in ('revisar', 'mapa')
                               else 'analizar'),
                  'revisar': 'revisar', 'mapa': 'mapa'}
_CSS_PASOS = {
    'actual': 'background:#2A7048;border-color:#2A7048;color:#FFFFFF;font-weight:700',
    'hecho': 'background:#1E3327;border-color:#1E3327;color:#A3D9B5;font-weight:600'}
# Como en la demo: la cabecera es una barra de lado a lado, arriba del todo (sin la barra
# de Streamlit ni su botón «Deploy»; el menú ⋮ sigue arriba a la derecha), y cada
# pantalla va centrada con un ancho de 1180 px como mucho.
st.html('<style>' + '''
header[data-testid="stHeader"]{background:transparent;height:0;overflow:visible}
/* La barra de Streamlit ocupa todo el ancho de arriba, encima de la cabecera: que deje
   pasar los clics (los pasos, Ayuda, ES|EN), salvo en sus propios botones (el menú ⋮). */
header[data-testid="stHeader"],header[data-testid="stHeader"] *{pointer-events:none}
header[data-testid="stHeader"] button,header[data-testid="stHeader"] a,
header[data-testid="stHeader"] [role="button"],[data-testid="stMainMenu"],
[data-testid="stMainMenu"] *,[data-testid="stStatusWidget"],
[data-testid="stStatusWidget"] *{pointer-events:auto}
[data-testid="stAppDeployButton"]{display:none}
[data-testid="stToolbar"]{top:14px;right:10px}
[data-testid="stMainBlockContainer"]{padding:0;max-width:none}
.st-key-pantalla{max-width:1180px;width:100%;margin:0 auto;padding:16px 32px 48px;
  box-sizing:border-box}
.st-key-pantalla>div>[data-baseweb="tab-border"]{display:none}
.st-key-cabecera{background:#282526;border-bottom:1px solid #403A3C;
  padding:12px 64px 12px 32px;flex-wrap:wrap;justify-content:space-between;
  row-gap:10px;column-gap:24px}
.st-key-cabecera>div:has(>.st-key-marca){flex:1 1 260px}
.st-key-ir_inicio button{padding:0;min-height:0;border:none;background:none}
.st-key-ir_inicio button p{font-family:Montserrat,Inter,system-ui,sans-serif;
  font-size:18px;font-weight:700;letter-spacing:-.01em;color:#ECEFF1;white-space:nowrap}
.st-key-subtitulo p{font-size:12px;color:#98A2AB;margin:-4px 0 0;white-space:nowrap}
.st-key-pasos{flex-wrap:wrap}
.st-key-pasos>div,.st-key-cab_der>div{flex-shrink:0}
.st-key-pasos button{border-radius:999px;min-height:0;padding:8px 14px;
  border:1px solid #494245;background:#2E2A2B;color:#CDD3D9}
.st-key-pasos button p{font-size:14px;font-weight:500;color:inherit;white-space:nowrap}
.st-key-pasos .sep{color:#5E6871;font-size:16px}
.st-key-ir_sugerencias button,.st-key-ir_ayuda button{padding:0;min-height:0}
.st-key-ir_sugerencias button p,.st-key-ir_ayuda button p{font-size:14px;font-weight:500;
  color:#7CC496;white-space:nowrap}
.st-key-ir_sugerencias button:hover p,.st-key-ir_ayuda button:hover p{color:#A3D9B5}
.st-key-ui_lang [data-testid="stButtonGroup"]>div{gap:0;border:1px solid #494245;
  border-radius:8px;overflow:hidden}
.st-key-ui_lang button{border:0;border-radius:0;min-height:0;padding:6px 10px;
  background:#2E2A2B}
.st-key-ui_lang button p{font-size:13px;font-weight:600;color:#CDD3D9}
.st-key-ui_lang button[kind$="Active"]{background:#2A7048}
.st-key-ui_lang button[kind$="Active"] p{color:#FFFFFF;font-weight:700}
.st-key-pantalla>div>div>[data-baseweb="tab-list"],
.st-key-pantalla>div>div>[data-baseweb="tab-border"]{display:none}
.st-key-aviso_cortado{max-width:1180px;width:100%;margin:12px auto 0;padding:0 32px;
  box-sizing:border-box}
''' + ''.join(f'.st-key-paso_{k} button,.st-key-paso_{k} button p{{{_CSS_PASOS[e]}}}'
              for k, e in _ESTADO_PASOS.items())
        + ('.st-key-ir_ayuda button{background:#2A7048;border-radius:999px;'
           'padding:6px 12px}.st-key-ir_ayuda button p{color:#FFFFFF;font-weight:700}'
           if PANTALLA == 'ayuda' else '')
        + '</style>')

# Mientras la app trabaja (la autora, 27-11-2026: «que cuando esté cargando algo, en vez de
# un icono pequeño arriba se ponga en medio de la página, en grande, y a poder ser una barra
# que cargue el % del progreso o el tiempo que le queda»). Streamlit pone su icono
# (stStatusWidget, arriba a la derecha) solo mientras ejecuta la app: con él, en el centro
# sale «Trabajando…» si tarda más de un instante, o, si lo que se hace tiene barra
# (barra_progreso), la barra grande con el % y lo que falta. No bloquea la página.
_N_BARRA = [0]          # para la clave de cada barra en esta pasada


def _tiempo_que_falta(segundos):
    """«quedan unos 3 min»: lo que falta, redondeado hacia arriba (mejor acabar antes de
    lo anunciado que después)."""
    import math
    if segundos < 45:
        return i18n.pick('queda menos de 1 min', 'less than 1 min left')
    minutos = math.ceil(segundos / 60)
    if minutos < 90:
        return i18n.pick(f'quedan unos {minutos} min', f'about {minutos} min left')
    return i18n.pick(f'quedan unas {minutos // 60} h {minutos % 60} min',
                     f'about {minutos // 60} h {minutos % 60} min left')


class barra_progreso:
    """Una barra de progreso grande, en medio de la pantalla mientras la app trabaja, con
    lo que se está haciendo, el % hecho y el tiempo que falta (calculado con lo que se ha
    tardado hasta ahora). Se usa como st.progress: .progress(fracción, text=…) y .empty().
    Al acabar no se ve (solo sale mientras la app trabaja)."""

    def __init__(self, texto=''):
        _N_BARRA[0] += 1
        self._hueco = st.container(key=f'barra_grande_{_N_BARRA[0]}')
        self._texto = texto
        self._t0 = time.time()
        self._barra = self._hueco.progress(0.0, text=texto or None)

    def progress(self, fraccion, text=None):
        f = max(0.0, min(1.0, float(fraccion)))
        if text is not None:
            self._texto = text
        partes = [self._texto] if self._texto else []
        if f < 1:
            partes.append(f'{round(f * 100)} %')
            pasado = time.time() - self._t0
            if f >= 0.02 and pasado >= 3:
                partes.append(_tiempo_que_falta(pasado * (1 - f) / f))
        try:
            self._barra.progress(f, text=' · '.join(partes) or None)
        except Exception:                               # noqa: BLE001
            pass

    def empty(self):
        self._hueco.empty()


st.html('<style>' + '''
[data-testid="stStatusWidget"]{visibility:hidden}
.st-key-cargando{position:fixed;inset:0;z-index:999980;display:flex;align-items:center;
  justify-content:center;pointer-events:none;background:rgba(18,16,17,.35);opacity:0;
  visibility:hidden}
body:has([data-testid="stStatusWidget"]) .st-key-cargando{visibility:visible;
  animation:cargando-aparece .25s ease .7s forwards}
@keyframes cargando-aparece{to{opacity:1}}
.st-key-cargando>div{width:auto !important;flex:0 0 auto}
.st-key-cargando .caja{background:#2E2A2B;border:1px solid #494245;border-radius:16px;
  padding:26px 34px;display:flex;align-items:center;gap:18px;width:max-content;
  max-width:90vw;margin:0 auto;
  box-shadow:0 12px 40px rgba(0,0,0,.45)}
.st-key-cargando .rueda{width:46px;height:46px;border-radius:50%;border:5px solid #403A3C;
  border-top-color:#7CC496;animation:cargando-gira .9s linear infinite;flex-shrink:0}
@keyframes cargando-gira{to{transform:rotate(360deg)}}
.st-key-cargando .t1{font:700 20px Montserrat,Inter,sans-serif;color:#ECEFF1}
.st-key-cargando .t2{font-size:14px;color:#AEB6BE;margin-top:4px}
[class*="st-key-barra_grande_"]{position:fixed;top:50%;left:50%;
  transform:translate(-50%,-50%);z-index:999990;width:min(560px,90vw);background:#2E2A2B;
  border:1px solid #494245;border-radius:16px;padding:24px 30px !important;
  box-sizing:border-box;
  box-shadow:0 0 0 100vmax rgba(18,16,17,.45),0 12px 40px rgba(0,0,0,.45) !important}
[class*="st-key-barra_grande_"] [data-testid="stProgress"] p,
[class*="st-key-barra_grande_"] [data-testid="stProgress"] [data-testid="stMarkdownContainer"]{
  font-size:16px !important;color:#ECEFF1;line-height:1.45}
/* la barra entera en gris (lo que falta) y lo hecho en verde */
[class*="st-key-barra_grande_"] [data-testid="stProgressBarTrack"]{height:14px !important;
  background:#403A3C !important;border-radius:7px !important;overflow:hidden;margin-top:6px}
[class*="st-key-barra_grande_"] [data-testid="stProgressBarTrack"]>div{height:100% !important;
  background:#7CC496 !important;border-radius:7px !important}
body:not(:has([data-testid="stStatusWidget"])) [class*="st-key-barra_grande_"]{display:none}
body:has([class*="st-key-barra_grande_"]:not([data-stale="true"])) .st-key-cargando{
  display:none}
''' + '</style>')
with st.container(key='cargando'):
    st.html('<div class="caja"><div class="rueda"></div><div><div class="t1">'
            + i18n.pick('Trabajando…', 'Working…') + '</div><div class="t2">'
            + i18n.pick('Un momento, por favor.', 'One moment, please.')
            + '</div></div></div>')

with st.container(key='cabecera', horizontal=True, vertical_alignment='center'):
    with st.container(key='marca', horizontal=True, vertical_alignment='center',
                      gap='small'):
        st.html(f'<img src="{LOGO_SRC}" alt="" width="36" height="36" '
                'style="display:block">', width='content')
        with st.container(gap=None, width='content'):
            st.button('Borreguil · Wet Meadow Finder', key='ir_inicio', type='tertiary',
                      on_click=ir_a, args=('inicio',))
            with st.container(key='subtitulo'):
                st.markdown(i18n.pick('Buscador y mapa de borreguiles',
                                      'Find and map borreguiles'))
    with st.container(horizontal=True, vertical_alignment='center', key='pasos',
                      width='content', gap='small'):
        for _i, (_k, _et) in enumerate(_PASOS):
            if _i:
                st.html('<span class="sep">›</span>', width='content')
            st.button(('✓ ' if _ESTADO_PASOS.get(_k) == 'hecho' else '') + _et,
                      key=f'paso_{_k}', on_click=ir_a, args=(_DESTINO_PASOS[_k],))
    with st.container(horizontal=True, vertical_alignment='center', key='cab_der',
                      width='content', gap='medium'):
        st.button(i18n.pick('Sugerencias y errores', 'Suggestions and errors'),
                  key='ir_sugerencias', type='tertiary', on_click=ir_a, args=('ayuda',),
                  help=i18n.pick('Sugerir un cambio o informar de un error',
                                 'Suggest a change or report an error'))
        st.button(i18n.pick('Ayuda', 'Help'), key='ir_ayuda', type='tertiary',
                  on_click=ir_a, args=('ayuda',))
        st.segmented_control('Idioma / Language', ['Español', 'English'],
                             format_func={'Español': 'ES', 'English': 'EN'}.get,
                             key='ui_lang', required=True, label_visibility='collapsed')

if st.session_state.get('_aviso_cortado') and PANTALLA in ('analizar', 'resultados',
                                                             'revisar', 'mapa'):
    _AVISO_CORTADO = st.container(key='aviso_cortado').empty()
    _AVISO_CORTADO.warning(i18n.pick(
        '**El último análisis no ha terminado.** Se paró antes de acabar (por ejemplo, al '
        'cambiar algo en el panel mientras se hacía). Vuelve a pulsar «Analizar» o «Probar '
        'un ejemplo».',
        '**The last analysis did not finish.** It stopped before the end (for instance, '
        'because something in the panel was changed while it was running). Press "Analyse" '
        'or "Try an example" again.'))

pant_inicio, pant_analizar, pant_resultados, pant_revisar, pant_mapa, pant_ayuda = st.tabs(
    list(PANTALLAS), key='pantalla', on_change='rerun')
# El pie del Inicio va fuera de las pantallas, de lado a lado como la cabecera (la demo).
pie_pantalla = st.container(key='pie_pantalla')
with pant_analizar:
    col_panel, col_principal = st.columns([0.38, 0.62], gap='large')

def _rotulo_modelo(nombre):
    """Cómo se ve un modelo en el desplegable, como en la demo: «Sierra Nevada (27
    variables · AUC 0,847)»."""
    try:
        ruta = model_path(nombre)
        return _rotulo_modelo_de(str(ruta), ruta.stat().st_mtime, i18n.get_lang())
    except Exception:                                   # noqa: BLE001
        return str(nombre)


@st.cache_data(show_spinner=False)
def _rotulo_modelo_de(ruta, mtime, idioma):
    mb = bp.load_model_bundle(ruta)
    feats = mb.get('features', [])
    zona = nombre_modelo(mb.get('training_zone'))
    if (any(f.startswith('cir_') for f in feats)
            and 'sierra nevada (andaluc' not in str(mb.get('training_zone', '')).lower()):
        zona += ' with infrared' if idioma == 'en' else ' con infrarrojo'
    auc = mb.get('auc_groupkfold')
    datos = f'{len(feats)} variables' + (f' · AUC {_n3(auc)}'
                                         if isinstance(auc, (int, float)) else '')
    return f'{zona} ({datos})'


def _css_caja_subir(key, titulo, detalle='', pequena=False):
    """Las cajas de subir ficheros como en la demo: borde discontinuo y su texto (en el
    idioma de la app) en vez del de Streamlit; toda la caja abre el selector."""
    s = f'.st-key-{key} [data-testid="stFileUploaderDropzone"]'
    i = f'{s} [data-testid="stFileUploaderDropzoneInstructions"]'
    return (f'{s}{{position:relative;border:1px dashed #60575A;border-radius:'
            f'{10 if pequena else 12}px;padding:{10 if pequena else 14}px;background:#292627;'
            'display:flex;flex-direction:column;align-items:flex-start;cursor:pointer}'
            f'{s}>span{{position:absolute;inset:0;opacity:0;z-index:1}}'
            f'{s}>span button{{width:100%;height:100%}}'
            f'{i}{{pointer-events:none;margin:0}}'
            f'{i} span{{display:none}}'
            f'{i}>div::before{{content:"{titulo}";display:block;'
            + ('font-size:13px;color:#CDD3D9' if pequena else
               'font-size:14px;font-weight:600;color:#ECEFF1') + '}'
            + (f'{i}>div::after{{content:"{detalle}";display:block;font-size:12px;'
               'color:#AEB6BE;margin-top:4px}' if detalle else ''))


# Como en la demo, en «1 · Analizar» el panel va pegado a la izquierda y de arriba abajo.
_CSS_ANALIZAR_ANCHO = '''
.st-key-pantalla{max-width:none !important;padding:0 !important}
[data-testid="stHorizontalBlock"]:has(>[data-testid="stColumn"]>div>.st-key-panel_analizar),
[data-testid="stHorizontalBlock"]:has(.st-key-panel_analizar){gap:0 !important}
[data-testid="stColumn"]:has(.st-key-panel_analizar){flex:0 1 420px !important;
  min-width:340px;max-width:440px}
.st-key-panel_analizar{border-radius:0 !important;border:0 !important;
  border-right:1px solid #534B4E !important;padding:22px !important;
  min-height:calc(100vh - 62px)}
[data-testid="stColumn"]:has(.st-key-analizar){padding:24px}
'''


def texto_puntos_campo():
    """Cómo preparar los puntos de campo. Antes era un desplegable del panel de Analizar;
    como en la demo, el panel enlaza a la Ayuda, donde está este texto."""
    return i18n.pick(
                'Usa la **plantilla de datos de puntos** (se descarga en la pantalla de '
                'inicio). Una fila por punto; solo son obligatorios el punto y `presence`:\n'
                '- `presence`: `1` borreguil · `0` no borreguil. `doubtful`: `1` si no sabes '
                'decirlo (no cuenta).\n'
                '- **Tipo**, opcional y solo si `presence` = 1; cada nivel cuenta por '
                'separado: `environment` (`stream` · `lake` · `slope`), `moisture` (`wet` · '
                '`dry`), `purity` (`pure` · `mixed-water` · `mixed-rock`). En español: '
                '`ambiente` (`arroyo` · `laguna` · `ladera`), `humedad` (`húmedo` · `seco`), '
                '`pureza` (`puro` · `mixto-agua` · `mixto-roca`).\n'
                '- Recomendado: `method` (`field` · `orthophoto`) y `date` (AAAA-MM-DD). '
                'Opcional: `observer`, `expertise` (`student` · `manager` · `researcher`), '
                '`certainty` (`high` · `medium` · `low`) y `notes`.\n'
                '- CSV con columnas `lon` y `lat` en grados, o una capa (GeoJSON, GeoPackage, '
                'Shapefile o KML) con esos nombres. La capa puede estar en cualquier '
                'proyección, pero tiene que tenerla (un Shapefile, con su .prj): la app la '
                'pasa sola a WGS84.\n\n'
                '**Incluye ausencias.** Los sitios donde has comprobado que no hay borreguil '
                '(`presence` = 0) valen tanto como los borreguiles. Sin ellos no se mide bien '
                'cuánto acierta el modelo ni se puede hacer el mapa de borreguil sí o no: la '
                'app no se inventa ausencias.\n\n'
                '**Otros nombres.** Si tu fichero trae la etiqueta con otro nombre '
                '(`Borreguil`, `presencia`, `label`, `clase`…) y valores sí/no, 1/0, yes/no o '
                'true/false, la app la reconoce y, al subirlo, te dice qué columna ha leído.\n\n'
                '**Sin ninguna columna** (solo en «puntos de campo»): cada punto se toma como '
                'borreguil. Sirve para ver cuántos borreguiles acierta el modelo, pero no '
                'aporta ausencias.',
                'Use the **point data template** (downloadable on the start screen). One row '
                'per point; only the point and `presence` are required:\n'
                '- `presence`: `1` borreguil · `0` not a borreguil. `doubtful`: `1` if you '
                'cannot tell (it does not count).\n'
                '- **Type**, optional and only if `presence` = 1; each level counts '
                'separately: `environment` (`stream` · `lake` · `slope`), `moisture` (`wet` · '
                '`dry`), `purity` (`pure` · `mixed-water` · `mixed-rock`). Spanish values also '
                'work.\n'
                '- Recommended: `method` (`field` · `orthophoto`) and `date` (YYYY-MM-DD). '
                'Optional: `observer`, `expertise` (`student` · `manager` · `researcher`), '
                '`certainty` (`high` · `medium` · `low`) and `notes`.\n'
                '- CSV with `lon` and `lat` columns in degrees, or a layer (GeoJSON, '
                'GeoPackage, Shapefile or KML) with those names. The layer can be in any '
                'projection, but it must have one (a Shapefile, with its .prj): the app '
                'converts it to WGS84 on its own.\n\n'
                '**Include absences.** The places where you have checked there is no '
                'borreguil (`presence` = 0) are worth as much as the borreguiles. Without '
                'them, how well the model does cannot be measured properly and the borreguil '
                'yes-or-no map cannot be made: the app never invents absences.\n\n'
                '**Other names.** If your file carries the label under another name '
                '(`Borreguil`, `presencia`, `label`, `clase`…) with yes/no, 1/0, sí/no or '
                'true/false values, the app recognises it and, when you upload it, tells you '
                'which column it has read.\n\n'
                '**No column at all** (only in "field points"): every point is taken as a '
                'borreguil. It shows how many borreguiles the model gets right, but it brings '
                'no absences.')


# ============================================================
# SIDEBAR – inputs y opciones
# ============================================================
with col_panel, st.container(key='panel_analizar'):
    # Las cajas de subir ficheros con el texto de la demo (en el idioma elegido) y, en esta
    # pantalla, el panel pegado a la izquierda y de arriba abajo, como en la demo.
    st.html('<style>'
            + _css_caja_subir('subir_puntos',
                              i18n.pick('Arrastra aquí tu fichero o elígelo',
                                        'Drag your file here or choose it'),
                              i18n.pick('KML, GeoJSON, CSV con lon y lat, o Shapefile',
                                        'KML, GeoJSON, CSV with lon and lat, or Shapefile'))
            + _css_caja_subir('subir_campo', i18n.pick('Añadir fichero de campo',
                                                       'Add a field file'), pequena=True)
            + _css_caja_subir('area_file', i18n.pick(
                'Arrastra tu polígono: KML, GeoJSON, GeoPackage o Shapefile (selecciona a la '
                'vez .shp, .shx, .dbf y .prj, o en un .zip)',
                'Drag your polygon: KML, GeoJSON, GeoPackage or Shapefile (select .shp, .shx, '
                '.dbf and .prj at once, or in a .zip)'), pequena=True)
            + _css_caja_subir('subir_capa_cuartiles',
                              i18n.pick('Arrastra aquí tu capa o elígela',
                                        'Drag your layer here or choose it'),
                              i18n.pick('GeoPackage o GeoJSON; KML, KMZ o un Shapefile en .zip',
                                        'GeoPackage or GeoJSON; KML, KMZ or a zipped Shapefile'))
            + (_CSS_ANALIZAR_ANCHO if PANTALLA == 'analizar' else '') + '</style>')
    # «Tu zona» lo primero (la autora, 27-11-2026: «que lo primero que aparezca sea
    # Your area»).
    with st.container(key='seccion_zona'):
        st.subheader(i18n.pick('Tu zona', 'Your area'))

        _modos_zona = {
            '🔤  Nombre del área protegida': i18n.pick('Un espacio protegido',
                                                      'A protected area'),
            '🆔  WDPA ID (protectedplanet.net)': i18n.pick('Código WDPA', 'WDPA code'),
            '🗺  Vector (subir KML/Shp/GeoJSON)': i18n.pick('Subir un polígono',
                                                           'Upload a polygon'),
            '⨯  Sin área (bbox de los puntos)': i18n.pick('Alrededor de los puntos',
                                                         'Around the points')}
        area_mode = st.radio('Modo', list(_modos_zona), index=3, horizontal=True,
                             key='zona_modo', format_func=_modos_zona.get,
                             label_visibility='collapsed')

        f_area = None; wdpa_id = ''; pa_name = ''
        _zona_sin_crs = False
        if area_mode.startswith('🗺'):
            # Con su «?»: tiene que tener proyección (la autora, 27-11-2026).
            f_area = st.file_uploader(
                i18n.pick('Tu polígono', 'Your polygon'),
                type=['kml','geojson','json','gpkg', 'shp','shx','dbf','prj','cpg','zip'],
                accept_multiple_files=True, key='area_file',
                help=i18n.pick(
                    'Tiene que tener proyección (sistema de referencia). Vale cualquiera, por '
                    'ejemplo ETRS89 / UTM 30N o WGS84: la app la pasa sola a grados. Si es un '
                    'Shapefile, sube también su .prj (o todo en un .zip); sin él no se deja '
                    'usar. KML y GeoJSON van siempre en grados (WGS84).',
                    'It must have a projection (coordinate reference system). Any projection '
                    'is fine, for instance ETRS89 / UTM 30N or WGS84: the app converts it to '
                    'degrees on its own. If it is a Shapefile, upload its .prj too (or '
                    'everything in a .zip); without it, it cannot be used. KML and GeoJSON are '
                    'always in degrees (WGS84).'))
            # Sin sistema de referencia (datum) no se deja usar (bp.falta_sistema_de_referencia)
            if f_area:
                _aviso_crs = _sin_crs_subida(tuple((f.name, f.getvalue()) for f in f_area),
                                             i18n.get_lang())
                if _aviso_crs:
                    st.error(_aviso_crs)
                    _zona_sin_crs = True
        elif area_mode.startswith('🆔'):
            # Se consulta la base mundial de espacios protegidos (WDPA) en Earth Engine con
            # la cuenta compartida de MonitorEO, sin clave de Protected Planet (la autora,
            # 27-11-2026: «que el campo WDPA ID haga la consulta por monitoreo»).
            wdpa_id = st.text_input(
                i18n.pick('Código WDPA', 'WDPA code'), placeholder='389011',
                help=i18n.pick('Código numérico del espacio protegido en la base mundial de '
                               'espacios protegidos (WDPA, protectedplanet.net). La app lo '
                               'busca en Earth Engine con la cuenta compartida de MonitorEO: '
                               'no hace falta clave ni cuenta. Ejemplos: 389011 Parque '
                               'Nacional de Sierra Nevada, 71213 Picos de Europa, 893 Ordesa y '
                               'Monte Perdido.',
                               'Numeric code of the protected area in the World Database on '
                               'Protected Areas (WDPA, protectedplanet.net). The app looks it '
                               'up in Earth Engine with the MonitorEO shared account: no key '
                               'or account is needed. Examples: 389011 Sierra Nevada National '
                               'Park, 71213 Picos de Europa, 893 Ordesa y Monte Perdido.'))
        elif area_mode.startswith('🔤'):
            pa_name = st.text_input(
                i18n.pick('Nombre del espacio protegido', 'Name of the protected area'),
                placeholder='Parque Nacional de Sierra Nevada', key='nombre_zona',
                help=i18n.pick('Búsqueda en Nominatim (OpenStreetMap). Cuanto más específico, '
                               'mejor: incluir «Parque Nacional» o «Parque Natural» suele ayudar.',
                               'Search in Nominatim (OpenStreetMap). The more specific, the '
                               'better: including "National Park" or "Natural Park" usually '
                               'helps.'))
        else:
            st.caption(i18n.pick('La zona será el rectángulo que rodea a tus puntos.',
                                 'The area will be the rectangle around your points.'))

    with st.container(key='seccion_puntos'):
        _plantilla_xlsx = APP_DIR / 'plantilla' / 'point_data_template.xlsx'
        with st.container(horizontal=True, vertical_alignment='center', key='cab_puntos'):
            st.subheader(i18n.pick('Tus puntos', 'Your points'))
            if _plantilla_xlsx.exists():
                st.download_button(
                    i18n.pick('Plantilla de datos de puntos', 'Point data template'),
                    _plantilla_xlsx.read_bytes(), file_name=_plantilla_xlsx.name,
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    key='plantilla_panel', type='tertiary')

        input_mode = st.radio('Origen de los puntos', [
            '📁  Subir archivo (KML/Shp/GeoJSON)',
            '🎲  Generar aleatoriamente en el área',
        ], index=0, label_visibility='collapsed', horizontal=True, key='origen_puntos',
            format_func=lambda o: (i18n.pick('Subir un fichero', 'Upload a file')
                                   if o.startswith('📁') else
                                   i18n.pick('Generar puntos al azar',
                                             'Generate random points')))

        f_input = None; gen_n = 0; gen_mode = 'stratified_elev'
        gen_min_elev = None; gen_max_elev = None; gen_seed = 42; gen_min_dist = 50
        gen_lejos = 0
        gen_capa = None; gen_frac_q1 = 0.75
        if input_mode.startswith('📁'):
            # Varios ficheros a la vez: un Shapefile son varios (.shp, .dbf, .shx, .prj)
            f_input = st.file_uploader(
                i18n.pick('Puntos por verificar', 'Points to be verified'),
                type=FORMATOS_PUNTOS,
                accept_multiple_files=True, key='subir_puntos',
                help=i18n.pick(
                    'KML, GeoJSON o CSV (con columnas lon y lat). Si es un Shapefile, '
                    'selecciona a la vez todos sus ficheros (.shp, .dbf, .shx y .prj) o '
                    'súbelo en un .zip.',
                    'KML, GeoJSON or CSV (with `lon` and `lat` columns). If it is a '
                    'Shapefile, select all its files at once (.shp, .dbf, .shx and .prj) or '
                    'upload it as a .zip.')) or None
            if f_input is not None:
                for _niv_et, _txt_et in (_etiquetas_de_subida(f_input) or []):
                    {'ok': _linea_ok, 'info': st.info, 'aviso': st.warning,
                     'falta': st.error}[_niv_et](_txt_et)
        else:
            gen_n = st.slider(i18n.pick('Cuántos puntos', 'How many points'), 30, 1000,
                              200, 10, key='cuantos_puntos')
            # Los valores internos no cambian (van a random_points); solo cómo se leen.
            # En este orden y con estos nombres (la autora, 27-11-2026).
            _nombre_estrategia = {
                'uniform': i18n.pick('Puntos al azar', 'Random points'),
                'poisson_disc': i18n.pick('Al azar, separados una distancia mínima',
                                          'Random but separated by a minimum distance'),
                'stratified_elev': i18n.pick('Al azar, estratificados por altitud',
                                             'Random but stratified by altitude'),
                'cuartiles': i18n.pick('Al azar por clase de probabilidad',
                                       'Random by probability class'),
            }
            gen_mode = st.selectbox(
                i18n.pick('Cómo se reparten', 'How they are spread'), list(_nombre_estrategia),
                index=0, format_func=_nombre_estrategia.get, key='estrategia',
                help=i18n.pick(
                    '**Puntos al azar**: sin ninguna regla.\n\n**Al azar, separados una '
                    'distancia mínima**: ningún par de puntos queda más cerca de la distancia '
                    'que elijas, para que no salgan puntos pegados.\n\n**Al azar, '
                    'estratificados por altitud**: el rango de altitud del área (o el que '
                    'pongas) se divide en 5 franjas iguales y se toma el mismo número de puntos '
                    'en cada una, para cubrir todo el gradiente.\n\n**Al azar por clase de '
                    'probabilidad**: subes una capa de polígonos con clases de probabilidad '
                    '(por ejemplo, los cuartiles de las áreas probables de borreguil) y la '
                    'mayoría de los puntos caen en los polígonos de la clase 1 y el resto en las '
                    'demás.',
                    '**Random points**: no rule.\n\n**Random but separated by a minimum '
                    'distance**: no pair of points is closer than the distance you choose, so '
                    'that no points end up side by side.\n\n**Random but stratified by '
                    'altitude**: the elevation range of the area (or the one you set) is split '
                    'into 5 equal bands and the same number of points is taken in each, to '
                    'cover the whole gradient.\n\n**Random by probability class**: you upload '
                    'a polygon layer with probability classes (for instance, the quartiles of '
                    'the probable alpine wet meadow areas) and most points fall in the polygons '
                    'of class 1 and the rest in the others.'))
            if gen_mode == 'cuartiles':
                # 26-11-2026, la autora: «quiero lanzar por cuartiles de una capa (que la
                # subirán los usuarios): el 75 % de los puntos en el primer cuartil y el 25 %
                # en el resto». Los puntos van sin etiqueta: al revisarlos salen presencias
                # y ausencias. El reparto lo hace random_points.por_cuartiles.
                import random_points
                _f_capa = st.file_uploader(
                    i18n.pick('Capa de polígonos con los cuartiles',
                              'Polygon layer with the quartiles'),
                    type=list(random_points.FORMATOS_CAPA), key='subir_capa_cuartiles',
                    help=i18n.pick(
                        'Una capa de polígonos con una columna «cuartil» (1, 2, 3 o 4; el 1 es '
                        'el primero) o con una columna numérica de la que la app calcula los '
                        'cuartiles. GeoPackage (.gpkg) o GeoJSON; KML, KMZ o un Shapefile en '
                        '.zip, si el ordenador puede leerlos. Si es un .zip, que traiga una '
                        'sola capa.',
                        'A polygon layer with a "cuartil" column (1, 2, 3 or 4; 1 is the '
                        'first) or with a numeric column from which the app computes the '
                        'quartiles. GeoPackage (.gpkg) or GeoJSON; KML, KMZ or a zipped '
                        'Shapefile, if the computer can read them. If it is a .zip, it must '
                        'hold a single layer.'))
                if _f_capa is None:
                    st.caption(i18n.pick(
                        'Sube la capa para elegir la columna y el reparto. Los puntos caen '
                        'dentro de sus polígonos y del área de estudio.',
                        'Upload the layer to choose the column and the split. The points fall '
                        'inside its polygons and inside the study area.'))
                else:
                    _capa_c, _err_c = _leer_capa_cuartiles(_f_capa.getvalue(), _f_capa.name)
                    _cols_c = (random_points.columnas_para_cuartiles(_capa_c['atributos'])
                               if _capa_c else [])
                    if _err_c:
                        st.error(i18n.pick(f'No se puede usar la capa: {_err_c}.',
                                           f'The layer cannot be used: {_err_c}.'))
                    elif not _cols_c:
                        st.error(i18n.pick(
                            'La capa no tiene ninguna columna «cuartil» ni numérica con la que '
                            'hacer cuartiles.',
                            'The layer has no "cuartil" column and no numeric column to make '
                            'quartiles with.'))
                    else:
                        # 26-11-2026, la autora: «el desplegable "columna de los cuartiles"
                        # no me encanta… que automáticamente lea la columna cuartil sin
                        # preguntar» o pida el campo con el que calcularlos. Si la capa trae
                        # la columna «cuartil», se usa sin preguntar (con una casilla para
                        # calcularlos con otro campo); si no, se elige ese campo.
                        _at_c = _capa_c['atributos']
                        _ya_cols = [c for c in _cols_c
                                    if random_points.es_columna_de_cuartiles(c, _at_c[c])]
                        _num_cols = [c for c in _cols_c if c not in _ya_cols]
                        _n_pol = len(_capa_c['geoms'])
                        _otro_c = bool(_ya_cols) and bool(_num_cols) and st.checkbox(
                            i18n.pick('Calcular los cuartiles con otro campo',
                                      'Compute the quartiles from another field'),
                            key='capa_cuartiles_otro',
                            help=i18n.pick(
                                f'La capa ya trae los cuartiles (columna «{_ya_cols[0]}») y '
                                'se usan tal cual. Marca la casilla para calcularlos con otro '
                                'campo de la capa.',
                                f'The layer already holds the quartiles ("{_ya_cols[0]}" '
                                'column) and they are used as they are. Tick the box to compute '
                                'them from another field of the layer.') if _ya_cols else None)
                        if _ya_cols and not _otro_c:
                            _col_c = _ya_cols[0]
                            st.caption(i18n.pick(
                                f"«{_capa_c['capa']}»: {_n_pol} polígonos. Se usan los "
                                f'cuartiles que trae la capa (columna «{_col_c}»).',
                                f"\"{_capa_c['capa']}\": {_n_pol} polygons. The quartiles "
                                f'that the layer holds are used ("{_col_c}" column).'))
                        else:
                            _col_c = st.selectbox(
                                i18n.pick('Campo para calcular los cuartiles',
                                          'Field to compute the quartiles from'), _num_cols,
                                key='capa_cuartiles_campo',
                                help=i18n.pick(
                                    'Un campo numérico de la capa (por ejemplo, el borreguil '
                                    'esperado). La app ordena los polígonos que caen en el área '
                                    'de estudio de mayor a menor valor y los reparte en cuatro '
                                    'grupos con el mismo número de polígonos: el cuartil 1 es '
                                    'el 25 % con los valores más altos.',
                                    'A numeric field of the layer (for instance, the expected '
                                    'borreguil). The app sorts the polygons that fall in the '
                                    'study area from the highest to the lowest value and splits '
                                    'them into four groups with the same number of polygons: '
                                    'quartile 1 is the 25 % with the highest values.'))
                            st.caption(i18n.pick(
                                f"«{_capa_c['capa']}»: {_n_pol} polígonos. Los cuartiles se "
                                f'calculan con «{_col_c}» y los polígonos que caen en el área '
                                'de estudio: el 1 es el 25 % con los valores más altos.',
                                f"\"{_capa_c['capa']}\": {_n_pol} polygons. The quartiles are "
                                f'computed from "{_col_c}" and the polygons that fall in the '
                                'study area: 1 is the 25 % with the highest values.'))
                        _pct_c = st.slider(
                            i18n.pick('Puntos en el cuartil 1 (%)',
                                      'Points in quartile 1 (%)'),
                            50, 100, 75, 5, key='capa_cuartiles_pct',
                            help=i18n.pick(
                                'El resto cae en los cuartiles 2, 3 y 4 juntos. Dentro de cada '
                                'grupo, al azar según la superficie: un polígono grande recibe '
                                'más puntos que uno pequeño. Los puntos no llevan etiqueta: al '
                                'revisarlos salen borreguiles (sobre todo en el cuartil 1) y '
                                'ausencias (sobre todo en el resto).',
                                'The rest falls in quartiles 2, 3 and 4 together. Within each '
                                'group, at random by area: a large polygon gets more points '
                                'than a small one. The points carry no label: when you review '
                                'them you get borreguiles (mostly in quartile 1) and absences '
                                '(mostly in the rest).'))
                        gen_frac_q1 = _pct_c / 100
                        _n1_c = int(round(gen_n * gen_frac_q1))
                        st.caption(i18n.pick(
                            f'{_n1_c} puntos en el cuartil 1 y {gen_n - _n1_c} en los cuartiles '
                            '2 a 4. Cada punto lleva su cuartil (columna «Cuartil» de la tabla y '
                            'de las descargas).',
                            f'{_n1_c} points in quartile 1 and {gen_n - _n1_c} in quartiles 2 to '
                            '4. Each point carries its quartile ("Quartile" column of the table '
                            'and the downloads).'))
                        gen_capa = {'geoms': _capa_c['geoms'],
                                    'valores': list(_capa_c['atributos'][_col_c]),
                                    'columna': _col_c}
            if gen_mode == 'poisson_disc':
                gen_min_dist = st.number_input(
                    i18n.pick('Distancia mínima entre puntos (m)',
                              'Minimum distance between points (m)'),
                    min_value=10, max_value=500, value=50, step=10,
                    help=i18n.pick(
                        'Ningún par de puntos quedará más cerca de esta distancia. Los puntos '
                        'se ajustan además al centro del píxel Sentinel-2 (10 m).',
                        'No pair of points will be closer than this distance. Points are '
                        'also snapped to the centre of the Sentinel-2 pixel (10 m).'))
            # Como en la demo: una sola etiqueta con su «?» y dos cajas (mínima y máxima);
            # después «Lejos de tus puntos de campo» y la semilla.
            # La altitud mínima y máxima, solo en «estratificados por altitud» (la autora,
            # 27-11-2026: «quita la elevación de todas las opciones menos la de altitude»);
            # en las demás, sin límite de altitud.
            gen_min_elev = gen_max_elev = None
            if gen_mode == 'stratified_elev':
                with st.container(key='tit_altitud'):
                    st.markdown(i18n.pick('Altitud mínima y máxima (m)',
                                          'Minimum and maximum elevation (m)'),
                                help=i18n.pick(
                                    'Opcional: el rango de altitud que se reparte en franjas, '
                                    'con el modelo de elevación SRTM de 30 m. Vacío, sin límite.',
                                    'Optional: the elevation range split into bands, with the '
                                    '30 m SRTM elevation model. Empty, no limit.'))
                col1, col2 = st.columns(2)
                with col1:
                    gen_min_elev_str = st.text_input(
                        i18n.pick('Altitud mínima (m)', 'Minimum elevation (m)'), value='2000',
                        placeholder=i18n.pick('mínima: sin límite', 'minimum: no limit'),
                        label_visibility='collapsed')
                with col2:
                    gen_max_elev_str = st.text_input(
                        i18n.pick('Altitud máxima (m)', 'Maximum elevation (m)'), value='',
                        placeholder=i18n.pick('máxima: sin límite', 'maximum: no limit'),
                        label_visibility='collapsed')
                gen_min_elev = float(gen_min_elev_str) if gen_min_elev_str.strip() else None
                gen_max_elev = float(gen_max_elev_str) if gen_max_elev_str.strip() else None
            gen_lejos = st.number_input(
                i18n.pick('Lejos de tus puntos ya verificados (m)',
                          'Away from your already verified points (m)'),
                min_value=0, max_value=5000, value=0, step=50,
                help=i18n.pick(
                    'Si subes tus puntos ya verificados (debajo), los puntos al azar quedan al menos a '
                    'esta distancia de ellos: caen en sitios que todavía no has muestreado, para '
                    'buscar allí ausencias (o algún borreguil nuevo) que revisas tú en la tabla. '
                    '0 = sin límite.\n\nRecomendado: 0 para probar el modelo por toda el área; '
                    '300 m para buscar ausencias fuera de tus puntos.',
                    'If you upload your already verified points (below), the random points stay at least '
                    'this far from them: they fall in places you have not sampled yet, to look '
                    'there for absences (or some new borreguil) that you review in the table. '
                    '0 = no limit.\n\nRecommended: 0 to test the model over the whole area; '
                    '300 m to look for absences away from your points.'))
            gen_seed = st.number_input(
                i18n.pick('Semilla (para repetirlo igual)', 'Seed (to repeat it exactly)'),
                0, 99999, 42,
                help=i18n.pick(
                    'La semilla es el número con el que empieza el sorteo de los puntos al '
                    'azar. Con la misma semilla, la misma zona y las mismas opciones, la app '
                    'saca exactamente los mismos puntos, así que el análisis se puede repetir o '
                    'lo puede comprobar otra persona. Cámbiala para sacar otros puntos.',
                    'The seed is the number the random draw of the points starts from. With '
                    'the same seed, the same area and the same options, the app draws exactly '
                    'the same points again, so the analysis can be repeated or checked by '
                    'someone else. Change it to get a different set of points.'))

        st.html('<hr class="separador">')
        f_truth = st.file_uploader(
            i18n.pick('Puntos ya verificados para la validación externa :gray[(opcional)]',
                      'Points already verified for external validation :gray[(optional)]'),
            type=FORMATOS_PUNTOS, accept_multiple_files=True, key='subir_campo',
            help=i18n.pick(
                'Sitios donde ya sabes si hay borreguil, en un fichero aparte. Sirven para '
                'medir cuánto acierta el modelo y, si quieres, para reentrenarlo. Si esa '
                'información ya va en «Tus puntos» (columna presence: 1 = borreguil, 0 = no), '
                'no subas nada aquí.\n\nKML, GeoJSON o CSV (con columnas lon y lat). Si es '
                'un Shapefile, selecciona a la vez todos sus ficheros o súbelo en un .zip.',
                'Places where you already know whether there is borreguil, in a separate '
                'file. They are used to measure how well the model does and, if you want, to '
                'retrain it. If that information is already in "Your points" (presence '
                'column: 1 = borreguil, 0 = no), do not upload anything here.\n\nKML, '
                'GeoJSON or CSV (with lon and lat columns). If it is a Shapefile, select all '
                'its files at once or upload it as a .zip.')) or None
        if f_truth is not None:
            # Qué columna ha leído y cuántos sí y no (24-11), o lo que falta (un .shp sin .dbf…)
            for _niv_et, _txt_et in (_etiquetas_de_subida(f_truth, campo=True) or []):
                {'ok': _linea_ok, 'info': st.info, 'aviso': st.warning,
                 'falta': st.error}[_niv_et](_txt_et)
        truth_train_only = st.checkbox(
            i18n.pick('No mostrarlos en el mapa, la tabla ni las descargas',
                      'Do not show them on the map, table or downloads'),
            value=False, disabled=(f_truth is None),
            help=i18n.pick(
                'Con la casilla o sin ella, tus puntos ya verificados sirven igual para medir '
                'cuánto acierta el modelo (y, si marcas «Sumar mis puntos a los del modelo y '
                'reentrenarlo», para entrenarlo). Marcada, además no salen en el mapa, la '
                'tabla ni las descargas: solo verás los puntos que estás revisando. Útil '
                'cuando subes muchos borreguiles verificados y no quieres que tapen el '
                'resultado. Los puntos tuyos que caigan a menos de 20 m de uno verificado sí '
                'se marcan como verificados y se muestran.',
                'With or without the box, your already verified points are used just the same '
                'to measure how well the model does (and, if you tick "Add my points to the '
                'model\'s and retrain it", to train it). Ticked, they are also left out of the '
                'map, table and downloads: you will only see the points you are reviewing. '
                'Useful when you upload many verified alpine wet meadows and do not want them '
                'to swamp the result. Your points that fall within 20 m of a verified one are '
                'marked as verified and shown.'))
        # «Cómo preparar tus puntos» es un enlace a la Ayuda, como en la demo (el
        # texto está allí, en texto_puntos_campo).
        st.button(i18n.pick('Cómo preparar tus puntos',
                            'How to prepare your points'),
                  key='ir_puntos_campo', type='tertiary', on_click=ir_a, args=('ayuda',))

    # El modelo, antes de «Datos de satélite» (la autora, 27-11-2026). Lo que necesita
    # de allí (si se lee el radar) sale de lo que ya está elegido.
    _backend_elegido = 'mpc' if st.session_state.get('cuenta') == 'mpc' else 'gee'
    with st.container(key='seccion_modelo'):
        # Como en la demo: la «?» en el título, el desplegable con el nombre de cada modelo
        # y una sola nota gris con el enlace a «Qué modelo elegir».
        st.subheader(i18n.pick('Modelos para obtener puntos candidatos de entrenamiento',
                                'Models to get candidate training points'), help=i18n.pick(
            'Es el que da la probabilidad de cada punto, aunque tus puntos traigan etiquetas de '
            'campo (con ellas la app mide cuánto acierta). Para entrenar además uno con tus '
            'puntos, marca la casilla de abajo. Un modelo entrenado en tu zona se puede guardar '
            '(en Resultados, Descargas) para reutilizarlo aquí.',
            'It is the one that gives each point its probability, even if your points carry '
            'field labels (with them the app measures how many it gets right). To also train '
            'one with your points, tick the box below. A model trained on your area can be '
            'saved (in Results, Downloads) to reuse it here.'))
        _models = list_models()
        if _models:
            sel_model_name = st.selectbox(
                i18n.pick('Modelo', 'Model'), _models, index=0, key='modelo_partida',
                format_func=_rotulo_modelo, label_visibility='collapsed')
            start_model_path = str(model_path(sel_model_name))
            _sel_feats = []
            try:
                _mb = bp.load_model_bundle(start_model_path)
                _sel_feats = _mb.get('features', [])
                _zona_m = str(_mb.get('training_zone', '?')).split(' (')[0]
                _n_m = _mb.get('training_n')
                # Sin Earth Engine no se lee el radar: la app usa la versión del modelo
                # entrenada sin él, y la nota dice cuánto acierta.
                _sr = modelo_sin_radar(start_model_path) if _backend_elegido != 'gee' else None
                _auc_sr = (_auc_modelo(str(_sr), _sr.stat().st_mtime)
                           if _sr is not None else None)
                with st.container(horizontal=True, vertical_alignment='center',
                                  key='nota_modelo'):
                    st.caption(i18n.pick(
                        (f'Entrenado con {_n_m} puntos de {_zona_m}. ' if _n_m else
                         f'Entrenado en {_zona_m}. ')
                        + 'En otra zona, revisa los puntos antes de fiarte.'
                        + ('' if _sr is None else ' Con Microsoft Planetary Computer se usa su versión '
                           'entrenada sin radar' + ('' if _auc_sr is None else
                                                    f' (AUC {_n3(_auc_sr)})') + '.'),
                        (f'Trained with {_n_m} points from {_zona_m}. ' if _n_m else
                         f'Trained in {_zona_m}. ')
                        + 'In another area, review the points before trusting them.'
                        + ('' if _sr is None else ' With Microsoft Planetary Computer its version '
                           'trained without radar is used' + ('' if _auc_sr is None else
                                                              f' (AUC {_n3(_auc_sr)})') + '.')))
                    # Abre la wiki, en el apartado «Qué modelo elegir» (la autora, 27-11-2026)
                    with st.container(key='ir_modelos'):
                        st.markdown(
                            '[' + i18n.pick('Qué modelo elegir', 'Which model to choose')
                            + ' ↗](' + i18n.pick(WIKI_URL + '#qué-modelo-elegir',
                                                  WIKI_URL + '/English#which-model-to-choose')
                            + ')')
                # Un modelo con PlanetScope sin la clave de Planet: sus variables irían vacías.
                if (any(f.startswith('ps_') for f in _sel_feats)
                        and not (get_secret('PL_API_KEY', None)
                                 or os.environ.get('PL_API_KEY', ''))):
                    st.warning(i18n.pick(
                        'Este modelo necesita PlanetScope 3 m, pero en este ordenador no hay '
                        'clave de Planet (PL_API_KEY). Sin ella, esas variables van vacías y la '
                        'probabilidad es menos fiable. Elige otro modelo.',
                        'This model needs PlanetScope 3 m, but there is no Planet key '
                        '(PL_API_KEY) on this computer. Without it those variables are empty and '
                        'the probability is less reliable. Choose another model.'))
            except Exception as _e:
                st.caption(f'(no se pudieron leer metadatos: {_e})')
            st.session_state.sel_model_feats = _sel_feats
        else:
            start_model_path = None
            st.session_state.sel_model_feats = []
            st.caption('No hay modelos .joblib en la carpeta.')
        st.session_state.start_model_path = start_model_path
        # Por defecto se usa SIEMPRE el modelo de partida, aunque los puntos traigan
        # etiquetas de campo (con ellas se mide cuánto acierta). Entrenar uno propio es
        # una decisión de quien usa la app, no algo que pase solo al subir 15 borreguiles.
        st.checkbox(
            i18n.pick('Sumar mis puntos a los del modelo y reentrenarlo', "Add my points to the model's and retrain it"),
            value=False, key='entrenar_propio',
            help=i18n.pick(
                'Sin marcar, la app usa el modelo de arriba tal cual y, si tus puntos traen '
                'etiquetas de campo, te dice cuántas acierta. Marcada, vuelve a entrenarlo '
                'con los 769 puntos de Sierra Nevada con los que se hizo MÁS todos tus puntos '
                'con etiqueta: los de campo, los que verificas en «Revisar» y los que añades '
                'pinchando en el mapa. Es un solo modelo con todos los puntos, y la app te dice '
                'cuánto acierta en los tuyos (validación por zonas). Se aplica al analizar y al '
                'verificar o añadir puntos; no hace falta volver a analizar.',
                'Unticked, the app uses the model above as it is and, if your points carry '
                'field labels, tells you how many it gets right. Ticked, it retrains it with '
                'the 769 Sierra Nevada points it was made with PLUS all your labelled points: '
                'field points, the ones you verify in «Review» and the ones you add by '
                'clicking on the map. It is a single model with all the points, and the app '
                'tells you how well it does on yours (spatial validation). It applies when '
                'running and when verifying or adding points; there is no need to run again.'))

        # El umbral es el corte del modelo: va en esta tarjeta, «Modelo» (27-11-2026).
        st.html('<hr class="separador">')
        with st.container(key='titb_umbral'):
            st.markdown(i18n.pick('Umbral para BORREGUIL', 'Threshold for BORREGUIL'),
                        help=i18n.pick(
                'Para qué: el modelo da a cada punto una probabilidad de ser borreguil, de 0 a '
                '1, y el umbral es el corte. Por encima, el punto sale como borreguil; por '
                'debajo, como no borreguil. Decide el resultado de cada punto, los recuentos y '
                'el acierto con tus puntos de campo; las probabilidades no cambian.\n\n'
                'Automático (Otsu, recomendado y de entrada): se adapta a cada análisis. Al '
                'aplicar el modelo a una zona nueva las probabilidades se recalibran, así que '
                'un umbral fijo puede dejar casi todo fuera; Otsu evita ese problema. Calcula '
                'el corte a partir de la forma de la distribución de las probabilidades (el '
                'método de Otsu, el clásico para binarizar imágenes): el valor que mejor separa '
                'los dos grupos, probable no borreguil y borreguil, haciendo mínima la '
                'variación dentro de cada grupo. No necesita puntos de campo.\n\n'
                'A mano: eliges el valor con el deslizador.\n\n'
                'Según mis puntos (verdad-terreno): usa tus puntos de campo. Con presencias y '
                'ausencias, el corte que maximiza el índice de Youden (sensibilidad + '
                'especificidad − 1). Solo con presencias, el corte que captura el 90 % de los '
                'borreguiles conocidos. Si no hay suficientes puntos de campo, usa Otsu.',
                'What for: the model gives each point a probability of being borreguil, from 0 '
                'to 1, and the threshold is the cut. Above it, the point comes out as '
                'borreguil; below it, as not borreguil. It decides the result of each point, '
                'the counts and the accuracy against your field points; the probabilities do '
                'not change.\n\n'
                'Automatic (Otsu, recommended and the default): it adapts to each analysis. '
                'When the model is applied to a new area the probabilities are recalibrated, '
                'so a fixed threshold can leave almost everything out; Otsu avoids that '
                'problem. It computes the cut from the shape of the distribution of the '
                'probabilities (the Otsu method, the classic one to binarise images): the value '
                'that best separates the two groups, probably not borreguil and borreguil, '
                'minimising the variation within each group. It needs no field points.\n\n'
                'By hand: you choose the value with the slider.\n\n'
                'From my points (ground truth): it uses your field points. With presences and '
                'absences, the cut that maximises the Youden index (sensitivity + specificity '
                '− 1). With presences only, the cut that captures 90 % of the known '
                'borreguiles. If there are not enough field points, it uses Otsu.'))
        st.caption(i18n.pick(
            'El modelo da a cada punto una probabilidad de ser borreguil, de 0 a 1. El umbral '
            'es el corte: por encima, el punto sale como borreguil (verde); por debajo, como no '
            'borreguil (rojo). Decide los recuentos y el acierto, pero no cambia las '
            'probabilidades.',
            'The model gives each point a probability of being borreguil, from 0 to 1. The '
            'threshold is the cut: above it, the point comes out as borreguil (green); below '
            'it, as not borreguil (red). It decides the counts and the accuracy, but it does '
            'not change the probabilities.'))
        _umbrales = {'Automático · Otsu': i18n.pick('Automático (Otsu)', 'Automatic (Otsu)'),
                     'Manual (deslizador)': i18n.pick('A mano', 'By hand'),
                     'Automático · verdad-terreno': i18n.pick('Según mis puntos',
                                                              'From my points')}
        thr_mode = st.radio(
            i18n.pick('Umbral para BORREGUIL', 'Threshold for BORREGUIL'), list(_umbrales),
            index=0, horizontal=True, format_func=_umbrales.get, key='modo_umbral',
            label_visibility='collapsed')
        if thr_mode.startswith('Manual'):
            threshold = st.slider(i18n.pick('Valor del umbral', 'Threshold value'),
                                  0.30, 0.80, 0.50, 0.01, key='umbral_manual', format='%.3f')
        else:
            threshold = st.session_state.get('umbral_manual', 0.50)
    with st.container(key='seccion_satelite'):
        st.subheader(i18n.pick('Datos de satélite', 'Satellite data'), help=i18n.pick('Cuenta compartida: la app calcula en Earth Engine con la cuenta compartida de MonitorEO (cuenta UGR), sin iniciar sesión ni tener cuenta propia. Es para uso no comercial.\n\nMi cuenta: con tu proyecto de Google Cloud registrado en Earth Engine (por ejemplo ee-tunombre), calcula con tu cuenta de Google y gasta la cuota de ese proyecto. Se recuerda para la próxima vez. «Entrar con Google» abre el navegador; solo hace falta una vez en cada ordenador. «Probar conexión» comprueba que funciona; no es obligatorio.\n\nMicrosoft Planetary Computer: sin cuenta, más lento y sin radar.', 'Shared account: the app computes in Earth Engine with the MonitorEO shared account (UGR account), with no sign-in and no account of your own. It is for non-commercial use.\n\nMy account: with your Google Cloud project registered in Earth Engine (for instance ee-yourname), it computes with your Google account and uses the quota of that project. It is remembered for the next time. "Sign in with Google" opens the browser; it is only needed once on each computer. "Test connection" checks that it works; it is not required.\n\nMicrosoft Planetary Computer: no account, slower and without radar.'))
        # Tres maneras de calcular (la autora, 26-11-2026: «que use la cuenta de servicio de
        # monitoreo, que esté por defecto y que el usuario pueda elegir su propia cuenta»):
        # la cuenta compartida de MonitorEO, la cuenta de Google de quien usa la app con su
        # proyecto (arranca en ella si es la última que funcionó) o Microsoft Planetary Computer.
        _cuentas = {'compartida': i18n.pick('Cuenta compartida', 'Shared account'),
                    'mia': i18n.pick('Mi cuenta', 'My account'),
                    'mpc': 'Microsoft Planetary Computer'}
        cuenta = st.radio(i18n.pick('Cuenta', 'Account'), list(_cuentas), horizontal=True,
                          index=1 if leer_ajuste('gee_project', '') else 0, key='cuenta',
                          format_func=_cuentas.get, label_visibility='collapsed')
        backend = 'mpc' if cuenta == 'mpc' else 'gee'
        gee_project = ''
        if cuenta == 'mpc':
            st.caption(i18n.pick(
                'Microsoft Planetary Computer: sin cuenta, más lento y sin radar. La app usa la versión '
                'del modelo sin radar.',
                'Microsoft Planetary Computer: no account, slower and without radar. The app uses the '
                'version of the model without radar.'))
        elif cuenta == 'compartida':
            if leer_ajuste('gee_project', ''):
                guardar_ajuste('gee_project', '')        # ha vuelto a la cuenta compartida
            _uso = _uso_cuenta_compartida()
            _txt_uso = i18n.pick(
                'Earth Engine con la cuenta compartida de MonitorEO (cuenta UGR), sin iniciar '
                'sesión.',
                'Earth Engine with the MonitorEO shared account (UGR account), with no '
                'sign-in.')
            st.caption(_txt_uso)
            if _uso and _uso.get('restricted'):
                st.warning(i18n.pick(
                    'Earth Engine ha limitado ahora la cuenta compartida (ha gastado su '
                    'cuota de este mes): los cálculos irán lentos y alguno puede fallar. '
                    'Si tienes cuenta propia, elige «Mi cuenta».',
                    'Earth Engine is now limiting the shared account (it has spent its '
                    'quota for this month): computations will be slow and some may fail. '
                    'If you have your own account, choose "My account".'))
        else:
            gee_project = st.text_input(
                i18n.pick('Tu proyecto de Google Cloud', 'Your Google Cloud project'),
                value=leer_ajuste('gee_project', ''), placeholder='ee-tunombre',
                key='gee_proyecto')
            c1, c2 = st.columns(2)
            with c1:
                if st.button(i18n.pick('Entrar con Google', 'Sign in with Google'),
                             key='entrar_google', width='stretch'):
                    try:
                        import gee_backend as geb
                        geb.authenticate()
                        st.success('Autenticación lanzada (revisa el navegador).')
                    except Exception as e:
                        st.error(f'Error de autenticación: {e}')
            with c2:
                if st.button(i18n.pick('Probar conexión', 'Test connection'),
                             key='probar_conexion', width='stretch'):
                    try:
                        import gee_backend as geb
                        ok, msg = geb.initialize(gee_project.strip() or None)
                        (st.success if ok else st.error)(msg)
                        if ok and gee_project.strip():
                            guardar_ajuste('gee_project', gee_project.strip())
                    except Exception as e:
                        st.error(str(e))
        # Los años de las imágenes van en esta tarjeta (la autora, 27-11-2026: «si ahí pone
        # datos de satélite, debería estar con la de abajo»); se rellena más abajo.
        # Los años, aquí, en «Datos de satélite» (antes se escribían desde la
        # tarjeta del modelo en un hueco de esta; ahora el modelo va antes).
        with st.container():
            st.html('<hr class="separador">')
            # Como en la demo: el título con su «?», la descripción y después el control.
            with st.container(key='titb_anios'):
                st.markdown(i18n.pick('Años de las imágenes de satélite',
                                      'Years of the satellite images'),
                            help=i18n.pick(
                    'Para qué: las imágenes de Óptico Sentinel-2 de los veranos (1 jun – 30 sep) de estos años son con las '
                    'que se calculan las variables de satélite de cada punto: el verdor, la humedad '
                    'de la vegetación, el agua. Cada variable junta todos los veranos (por ejemplo, '
                    'la mediana de las imágenes del inicio de verano (15 jun – 25 jul) y del fin de '
                    'verano (1 ago – 30 sep) de todos los años).\n\n'
                    'Con más años los valores son más estables (pesan menos las nubes, la nieve o '
                    'un verano raro), pero un cambio reciente del borreguil se diluye. Los mismos '
                    'años se usan para leer las variables del mapa, en «3 · Hacer el mapa». El '
                    'Radar Sentinel-1 usa siempre los veranos (1 jun – 30 sep) de 2019 a 2024.\n\n'
                    'Calidad de las imágenes: solo entran escenas de Sentinel-2 con menos del 30 % '
                    'de nubes, y en cada píxel se quitan las nubes, las sombras de nube, los cirros '
                    'y los píxeles oscuros o saturados (con la clasificación SCL de Sentinel-2, '
                    'mirando también los píxeles de alrededor). En las variables de junio a '
                    'septiembre (WAVI, albedo, NDMI y NDWI: media, mínimo, máximo y variabilidad) '
                    'se quita también la nieve; en las de inicio (15 jun – 25 jul) y fin de verano (1 ago – 30 sep), no. El radar no '
                    'lo necesita: atraviesa las nubes.\n\n'
                    'Escribe «2017-actualidad» (o «2017-present») para llegar hasta el último '
                    'verano terminado, o años sueltos y rangos: «2017-2025», «2019,2021».\n\n'
                    'Recomendado: 2017-2025, los años con los que se entrenó el modelo de Sierra '
                    'Nevada; con otros, sus valores pueden no ser comparables.',
                    'What for: the Optical Sentinel-2 images of the summers (1 Jun – 30 Sep) of these years are the ones '
                    'used to compute the satellite variables of each point: greenness, vegetation '
                    'moisture, water. Each variable pools all the summers (for instance, the median '
                    'of the images of early summer (15 Jun – 25 Jul) and late summer (1 Aug – '
                    '30 Sep) of all the years).\n\n'
                    'With more years the values are more stable (clouds, snow or an odd summer '
                    'weigh less), but a recent change of the borreguil gets diluted. The same years '
                    'are used to read the map variables, in "3 · Make the map". The Radar '
                    'Sentinel-1 always uses the summers (1 Jun – 30 Sep) of 2019 to 2024.\n\n'
                    'Image quality: only Sentinel-2 scenes with less than 30 % clouds are used, '
                    'and in each pixel clouds, cloud shadows, cirrus and dark or saturated pixels '
                    'are removed (with the Sentinel-2 SCL classification, also looking at the '
                    'surrounding pixels). In the June to September variables (WAVI, albedo, NDMI '
                    'and NDWI: mean, minimum, maximum and variability) snow is removed too; in the '
                    'early (15 Jun – 25 Jul) and late summer (1 Aug – 30 Sep) ones, it is not. The radar does not need it: it goes '
                    'through clouds.\n\n'
                    'Write "2017-present" to go up to the last summer that has ended, or single '
                    'years and ranges: "2017-2025", "2019,2021".\n\n'
                    'Recommended: 2017-2025, the years the Sierra Nevada model was trained with; '
                    'with others, its values may not be comparable.'))
            st.caption(i18n.pick(
                'De qué veranos (1 jun – 30 sep) se toman las imágenes de Óptico Sentinel-2 con las que se calculan las '
                'variables de satélite de cada punto (el verdor, la humedad, el agua). Antes se '
                'quitan las nubes, sus sombras y, en algunas variables, la nieve. Son también los '
                'del mapa; el Radar Sentinel-1 usa siempre 2019-2024.',
                'Which summers (1 Jun – 30 Sep) the Optical Sentinel-2 images come from, used to compute the satellite '
                'variables of each point (greenness, moisture, water). Clouds, their shadows and, '
                'in some variables, snow are removed first. They are also the map years; the '
                'Radar Sentinel-1 always uses 2019-2024.'))
            # «2017-present» (la autora, 27-11-2026): hasta el último verano terminado
            # (bp.anio_presente); también vale «2017-2025», «2019,2021»…
            years_str = st.text_input(
                i18n.pick('Años (p. ej. de 2017 a la actualidad)',
                          'Years (e.g. from 2017 to present)'),
                value=i18n.pick('2017-actualidad', '2017-present'), key='anios')

    with st.container(key='seccion_imagenes'):
        st.subheader(i18n.pick('Imágenes aéreas', 'Aerial images'))
        with st.container(key='titb_imagen'):
            st.markdown(i18n.pick('Imagen aérea', 'Aerial image'), help=i18n.pick(
                'Para qué: de cada punto se descarga una foto de su zona (de unos 500 m de '
                'lado, centrada en él), y de ella se calculan las variables de la imagen aérea '
                'que usa el modelo: la roca de alrededor, la densidad de bordes, la fracción '
                'verde-azulada, el contraste de la vegetación, la textura. La ortofoto que ves '
                'al revisar se elige aparte, en las capas del mapa.\n\n'
                'Mundial: ESRI World Imagery, ortofoto de todo el mundo, de unos 0,5 m; o '
                'PlanetScope (3 m), si tienes cuenta de Planet: añade sus imágenes de 8 '
                'bandas (sirve con los modelos de la variante Planet; tarda 10-30 minutos '
                'más y gasta tu cuota).\n\n'
                'Por país: España (PNOA del IGN) usa la ortofoto autonómica cuando está disponible '
                '(Andalucía 2022, Aragón 2024, Cataluña vigente, Cantabria 2023, Castilla y León '
                '2020, Canarias) y el PNOA nacional para el resto de España. Fuera de España usa '
                'la mundial.',
                'What for: for each point a photo of its area is downloaded (about 500 m '
                'across, centred on it), and from it the aerial-image variables used by the '
                'model are computed: the surrounding rock, the edge density, the blue-green '
                'fraction, the vegetation contrast, the texture. The orthophoto you see when '
                'reviewing is chosen separately, in the map layers.\n\n'
                'Worldwide: ESRI World Imagery, orthophoto of the whole world, about 0.5 m; or '
                'PlanetScope (3 m), if you have a Planet account: it adds its 8-band images '
                '(it works with the Planet variant models; it takes 10-30 minutes longer and '
                'uses your quota).\n\n'
                'By country: Spain (IGN PNOA) uses the regional orthophoto when available (Andalusia '
                '2022, Aragon 2024, Catalonia current, Cantabria 2023, Castile and León 2020, '
                'Canary Islands) and the national PNOA for the rest of Spain. Outside Spain it '
                'uses the worldwide one.'))
        st.caption(i18n.pick(
            'De qué ortofoto se descarga la foto de cada punto, de la que el modelo saca las '
            'variables de la imagen aérea: la roca de alrededor, los bordes, el verde, la '
            'textura.',
            'Which orthophoto the photo of each point is downloaded from; the model takes the '
            'aerial-image variables from it: the surrounding rock, the edges, the green, the '
            'texture.'))
        # Mundial (ESRI o, con cuenta de Planet, PlanetScope) o por país (de momento, el
        # PNOA de España), cada uno con su desplegable (la autora, 27-11-2026).
        _pl_key = get_secret('PL_API_KEY', None) or os.environ.get('PL_API_KEY', '')
        # Las etiquetas (ocultas) no cambian con el idioma: si cambian, Streamlit lo toma
        # por otro control y se queda sin nada elegido.
        _ambito = st.radio(
            'Imagen aérea / Aerial image', ['mundial', 'pais'], index=0,
            horizontal=True, key='imagen_ambito', label_visibility='collapsed',
            format_func=lambda a_: (i18n.pick('Mundial', 'Worldwide') if a_ == 'mundial'
                                    else i18n.pick('Por país', 'By country')))
        use_planet = False
        if _ambito != 'pais':          # «Mundial» (también si no hay nada elegido)
            _mundial = st.selectbox(
                'Mundial / Worldwide', ['esri', 'planet'], index=0,
                key='imagen_mundial', label_visibility='collapsed',
                format_func=lambda o_: (
                    i18n.pick('ESRI World Imagery (todo el mundo, ~0,5 m)',
                              'ESRI World Imagery (whole world, ~0.5 m)') if o_ == 'esri'
                    else 'PlanetScope (3 m)' if _pl_key
                    else i18n.pick('PlanetScope (3 m) · necesita cuenta de Planet',
                                   'PlanetScope (3 m) · needs a Planet account')))
            img_src = 'esri'
            if _mundial == 'planet':
                if _pl_key:
                    use_planet = True
                    st.caption(i18n.pick(
                        'La foto de cada punto sale de ESRI y además se piden las imágenes '
                        'PlanetScope (3 m, 8 bandas) del área: tarda 10–30 minutos más y gasta '
                        'la cuota de tu suscripción.',
                        'The photo of each point comes from ESRI and the PlanetScope images '
                        '(3 m, 8 bands) of the area are also requested: it takes 10–30 minutes '
                        'longer and uses the quota of your subscription.'))
                else:
                    st.warning(i18n.pick(
                        'PlanetScope necesita una cuenta de Planet (su clave, PL_API_KEY, en '
                        'este ordenador). Sin ella se usa ESRI.',
                        'PlanetScope needs a Planet account (its key, PL_API_KEY, on this '
                        'computer). Without it, ESRI is used.'))
            else:
                st.caption(i18n.pick('Ortofoto de todo el mundo, de unos 0,5 m.',
                                     'Orthophoto of the whole world, about 0.5 m.'))
        else:
            st.selectbox(
                'País / Country', ['es'], index=0, key='imagen_pais',
                label_visibility='collapsed',
                format_func=lambda o_: i18n.pick('España · PNOA (IGN)', 'Spain · PNOA (IGN)'))
            img_src = 'pnoa'
            st.caption(i18n.pick(
                'Ortofoto de España de 0,25 m (la autonómica cuando la hay). Fuera de España '
                'se usa la mundial.',
                'Orthophoto of Spain of 0.25 m (the regional one when there is one). Outside '
                'Spain the worldwide one is used.'))
        # «Modo rápido» y «Saltar imágenes» se quitaron con el diseño nuevo (la autora,
        # 26-11-2026): el análisis es siempre completo, con la foto aérea de cada punto.
        fast_mode = False
        skip_imgs = False
        # Sin scikit-image (p. ej. su DLL bloqueada por Windows) no hay texturas aunque se
        # descarguen las imágenes.
        _sin_texturas = isinstance(bp.graycomatrix, bp._SkimageNoDisponible)
        _modelo_con_img = any(f in st.session_state.get('sel_model_feats', [])
                              for f in bp.FEATURES_RF[:12])
        if _modelo_con_img and _sin_texturas:
            st.warning(i18n.pick(
                'Este ordenador no puede calcular las texturas de la imagen (Windows bloquea la '
                'librería scikit-image). El modelo de partida se entrenó con ellas: aquí se '
                'rellenan con un valor típico de Sierra Nevada y su probabilidad es **menos '
                'fiable**. Úsala para proponer puntos y revísalos.',
                'This computer cannot compute the image textures (Windows blocks the '
                'scikit-image library). The starting model was trained with them: here they are '
                'filled with a typical Sierra Nevada value and its probability is **less '
                'reliable**. Use it to propose points and review them.'))
        skip_s2 = False
    # Como en la demo: se despliega dentro del panel (▸ / ▾), no flota.
    with st.expander(i18n.pick('💬 Sugerir un cambio o informar de un error',
                               '💬 Suggest a change or report an error'),
                     key='sugerencias_panel'):
        mostrar_sugerencias()
    _v = version_app()
    if _v:
        st.caption(' · '.join(x for x in (i18n.pick(f'Versión {_v}', f'Version {_v}'),
                                          fecha_app()) if x))


# --- 1 · Analizar: tus puntos y tu zona, y el botón de analizar ---------------------
@st.cache_data(show_spinner=False)
def _puntos_vista(partes):
    """(lon, lat) de unos puntos recién subidos, para verlos en el mapa antes de
    analizar; [] si aquí no se pueden leer (al analizar ya se verá por qué)."""
    ruta = None
    try:
        ruta = _guardar_subida([_Subido(n, d) for n, d in partes])
        return [(float(r['lon']), float(r['lat'])) for r in bp.read_points(ruta)
                if r.get('lon') is not None and r.get('lat') is not None]
    except Exception:                                   # noqa: BLE001
        return []
    finally:
        if ruta:
            shutil.rmtree(os.path.dirname(ruta), ignore_errors=True)


SIERRA_NEVADA_CAJA = (-3.55, 36.95, -2.85, 37.20)      # oeste, sur, este, norte


def capas_oscuras(fm, abiertas=True, nota_id=None, nota_capa=None):
    """El control de capas como el panel «Capas» de la demo (Revisar.dc.html): en
    oscuro, con su título («▾ Capas», que pliega y despliega la lista) y los dos
    grupos, FONDO (las capas de fondo) y ENCIMA (las que se encienden encima); los
    créditos, en una etiqueta oscura y sin el «Leaflet» delante; y los botones del
    zoom, oscuros. Se llama después de añadir el folium.LayerControl(collapsed=False).
    `nota_id`/`nota_capa`: un elemento del mapa que se ve solo mientras esté
    encendida la capa cuyo nombre contiene `nota_capa` (la parte de los ríos en la
    leyenda del mapa de 2 · Revisar)."""
    import json as _json
    from branca.element import MacroElement
    from jinja2 import Template
    fondo, encima = i18n.pick('Fondo', 'Base'), i18n.pick('Encima', 'On top')
    fm.get_root().header.add_child(folium.Element(
        '<style>'
        '.leaflet-container{background:#2E2A2B;font-family:Inter,system-ui,sans-serif}'
        '.leaflet-control-layers,.leaflet-control-layers-expanded{'
        'background:#282526F2;border:1px solid #524A4D!important;border-radius:12px;'
        'padding:6px 12px 8px;color:#CDD3D9;font:12px/1.25 Inter,system-ui,sans-serif;'
        'box-shadow:0 6px 18px rgba(0,0,0,.35)!important;max-width:260px}'
        # (sin la barra de desplazamiento que Leaflet le pone si al crearse no cabe)
        '.leaflet-control-layers-list,.leaflet-control-layers-scrollbar{margin:0;'
        'height:auto!important;overflow:visible!important;padding-right:0!important}'
        '.leaflet-control-layers .capas-tit{display:flex;align-items:center;gap:8px;'
        'border:0;background:transparent;color:#ECEFF1;font:700 13px Inter,system-ui,'
        'sans-serif;padding:4px 0;cursor:pointer;width:100%;text-align:left}'
        '.leaflet-control-layers .capas-tit span{color:#7CC496;width:12px}'
        '.leaflet-control-layers-base::before,.leaflet-control-layers-overlays::before{'
        'display:block;font-size:11px;font-weight:700;letter-spacing:.06em;'
        'text-transform:uppercase;color:#98A2AB;margin:2px 0 1px}'
        f'.leaflet-control-layers-base::before{{content:{_json.dumps(fondo)}}}'
        '.leaflet-control-layers-overlays::before{'
        f'content:{_json.dumps(encima)};margin-top:6px}}'
        '.leaflet-control-layers-separator{display:none}'
        # sin capas «encima» (el mapa de Analizar), sin su título
        '.leaflet-control-layers-overlays:empty{display:none}'
        '.leaflet-control-layers label{margin:0;cursor:pointer}'
        '.leaflet-control-layers label>span,.leaflet-control-layers label>div{'
        'display:flex;align-items:center;gap:8px;min-height:21px}'
        '.leaflet-control-layers-selector{accent-color:#7CC496;width:14px;height:14px;'
        'margin:0!important;top:0!important;flex:none}'
        '.leaflet-bar{border:0!important;border-radius:10px!important;'
        'box-shadow:0 4px 12px rgba(0,0,0,.35)!important}'
        '.leaflet-bar a,.leaflet-bar a:hover{width:40px!important;height:40px!important;'
        'line-height:38px!important;background:#2E2A2B!important;color:#ECEFF1!important;'
        'border:1px solid #60575A!important;font-size:20px!important}'
        '.leaflet-bar a:first-child{border-radius:10px 10px 0 0!important}'
        '.leaflet-bar a:last-child{border-radius:0 0 10px 10px!important;'
        'border-top:0!important}'
        '.leaflet-control-attribution{background:#221F20B3!important;'
        'color:#AEB6BE!important;border-radius:6px;font-size:11px!important;'
        'padding:3px 8px!important;margin:0 12px 14px 0!important}'
        '.leaflet-control-attribution a{color:#AEB6BE!important}'
        '</style>'))
    js = (
        'setTimeout(function(){'
        f'var mapa={fm.get_name()};'
        'try{mapa.attributionControl.setPrefix(false);}catch(e){}'
        'var c=document.querySelector(".leaflet-control-layers");'
        'if(c&&!c.querySelector(".capas-tit")){'
        'var l=c.querySelector(".leaflet-control-layers-list");'
        'var b=document.createElement("button");b.type="button";b.className="capas-tit";'
        f'var abierta={"true" if abiertas else "false"};'
        f'var t={_json.dumps(i18n.pick("Capas", "Layers"))};'
        'var pon=function(){l.style.display=abierta?"":"none";'
        'b.innerHTML="<span>"+(abierta?"\\u25be":"\\u25b8")+"</span>"+t;'
        'b.setAttribute("aria-expanded",abierta?"true":"false");};'
        'b.addEventListener("click",function(e){e.preventDefault();e.stopPropagation();'
        'abierta=!abierta;pon();});'
        'pon();c.insertBefore(b,c.firstChild);'
        'if(window.L){L.DomEvent.disableClickPropagation(c);'
        'L.DomEvent.disableScrollPropagation(c);}}'
        + ('' if not (nota_id and nota_capa) else
           f'var n=document.getElementById({_json.dumps(nota_id)});'
           f'var k={_json.dumps(nota_capa)};'
           'if(n){mapa.on("overlayadd",function(e){if(e.name.indexOf(k)>=0)'
           'n.style.display="";});'
           'mapa.on("overlayremove",function(e){if(e.name.indexOf(k)>=0)'
           'n.style.display="none";});}')
        + '},0);')

    class _Js(MacroElement):
        _template = Template('{% macro script(this, kwargs) %}' + js
                             + '{% endmacro %}')
    _Js().add_to(fm)


def _zoom_para(caja, ancho=800, alto=680, margen=60):
    """El zoom de Leaflet con el que la caja (oeste, sur, este, norte) cabe en un mapa de
    ancho × alto píxeles, con un margen."""
    import math
    def _y(lat):
        s = math.sin(math.radians(max(-85.0, min(85.0, lat))))
        return math.log((1 + s) / (1 - s)) / (4 * math.pi)
    dx = max(caja[2] - caja[0], 1e-4) / 360
    dy = max(abs(_y(caja[3]) - _y(caja[1])), 1e-5)
    z = min(math.log2((ancho - margen) / 256 / dx), math.log2((alto - margen) / 256 / dy))
    return int(max(3, min(16, math.floor(z))))


def _mapa_previo(puntos, rectangulo):
    """El mapa de «Tus puntos y tu zona»: la ortofoto de fondo (ESRI de entrada; PNOA,
    Google u OpenStreetMap en el panel «Capas»), tus puntos (círculos blancos), tu zona si es
    el rectángulo que los rodea (verde, discontinuo) y la leyenda dentro del mapa. Sin
    puntos, Sierra Nevada."""
    if puntos:
        lons = [p[0] for p in puntos]
        lats = [p[1] for p in puntos]
        caja = (min(lons), min(lats), max(lons), max(lats))
    else:
        caja = SIERRA_NEVADA_CAJA
    # El zoom se calcula aquí y no con fit_bounds: el mapa se dibuja también con la
    # pantalla escondida (tamaño 0), y fit_bounds lo dejaría en el zoom máximo.
    m = folium.Map(location=[(caja[1] + caja[3]) / 2, (caja[0] + caja[2]) / 2],
                   zoom_start=_zoom_para(caja), tiles=None, control_scale=False)
    # Fondos (la autora, 27-11-2026): los del panel «Capas» de Resultados, en su orden
    # (Ortofoto PNOA, ESRI World Imagery, Google Satélite, OpenStreetMap), con la de ESRI
    # de entrada; sin ríos ni lagunas.
    folium.raster_layers.WmsTileLayer(
        url='https://www.ign.es/wms-inspire/pnoa-ma', layers='OI.OrthoimageCoverage',
        fmt='image/jpeg', transparent=False, attr='IGN · PNOA',
        name=i18n.pick('Ortofoto PNOA', 'PNOA orthophoto'),
        overlay=False, control=True, show=False, max_zoom=22).add_to(m)
    folium.TileLayer(
        'https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/'
        'tile/{z}/{y}/{x}', attr='Esri World Imagery', name='ESRI World Imagery',
        max_zoom=22, max_native_zoom=19, show=True).add_to(m)
    folium.TileLayer('https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}', attr='Google',
                     name=i18n.pick('Google Satélite (más zoom)', 'Google Satellite (more zoom)'),
                     max_zoom=22, max_native_zoom=21, show=False).add_to(m)
    folium.TileLayer('OpenStreetMap', name='OpenStreetMap', max_zoom=19, show=False).add_to(m)
    if puntos and rectangulo and len(puntos) > 1:
        folium.Rectangle([[caja[1], caja[0]], [caja[3], caja[2]]],
                         color='#7CC496', weight=2.5, dash_array='8 6', fill=True,
                         fill_color='#7CC496', fill_opacity=0.07).add_to(m)
    for lon, lat in puntos:
        folium.CircleMarker([lat, lon], radius=6, color='#221F20', weight=2,
                            fill=True, fill_color='#ECEFF1', fill_opacity=1).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    capas_oscuras(m)                # el panel «Capas» oscuro, como en Resultados
    # La leyenda, dentro del mapa (abajo a la izquierda), como en la demo.
    filas = []
    if puntos:
        filas.append('<span>○ ' + i18n.pick(f'Tus {len(puntos)} puntos',
                                            f'Your {len(puntos)} points') + '</span>')
        if rectangulo:
            filas.append('<span style="color:#7CC496">┄ ' + i18n.pick('Tu zona', 'Your area')
                         + '</span>')
    if filas:                       # sin puntos, sin leyenda
        m.get_root().html.add_child(folium.Element(
            '<div style="position:fixed;left:16px;bottom:16px;z-index:9999;'
            'background:rgba(46,42,43,.9);border:1px solid #494245;border-radius:10px;'
            'padding:10px 12px;display:flex;flex-direction:column;gap:6px;'
            'font:12px Inter,system-ui,sans-serif;color:#CDD3D9">' + ''.join(filas)
            + '</div>'))
    return m


with col_principal:
    _pts_vista = (_puntos_vista(tuple((f.name, f.getvalue()) for f in f_input))
                  if f_input else [])
    if f_input:
        _n_analizar = len(_pts_vista)
    else:
        _n_analizar = gen_n if input_mode.startswith('🎲') else 0
    # Se puede analizar con puntos subidos, o generándolos si hay zona
    can_run = (f_input is not None) or (input_mode.startswith('🎲') and gen_n > 0
                                          and (f_area or wdpa_id or pa_name))
    # Una capa sin sistema de referencia (datum) no se deja usar: ni la zona ni los puntos
    if _zona_sin_crs or any(
            _sin_crs_subida(tuple((f.name, f.getvalue()) for f in _fs), i18n.get_lang())
            for _fs in (f_input, f_truth) if _fs):
        can_run = False
    c_tit, c_boton = st.columns([3, 2], vertical_alignment='center')
    with c_tit, st.container(key='tit_principal'):
        st.subheader(i18n.pick('Tus puntos y tu zona', 'Your points and your area'))
        st.caption(i18n.pick('Comprueba que todo cae donde esperas y pulsa Analizar.',
                             'Check that everything falls where you expect and press '
                             'Analyse.'))
    with c_boton:
        run_btn = st.button(
            (i18n.pick(f'Analizar los {_n_analizar} puntos',
                       f'Analyse the {_n_analizar} points') if _n_analizar else
             i18n.pick('Analizar', 'Analyse')),
            type='primary', disabled=not can_run, key='analizar', width='stretch')
        if backend != 'gee':
            _txt_cuenta = i18n.pick('Microsoft Planetary Computer: más lento y sin radar.',
                                    'Microsoft Planetary Computer: slower and without radar.')
        elif gee_project:
            _txt_cuenta = i18n.pick('Earth Engine con tu proyecto.',
                                    'Earth Engine with your project.')
        else:
            _uso_v = _uso_cuenta_compartida()
            _txt_cuenta = (i18n.pick(
                f"Earth Engine con la cuenta compartida: te quedan "
                f"{_num(_uso_v['left_hours'])} h este mes.",
                f"Earth Engine with the shared account: {_num(_uso_v['left_hours'])} h "
                'left this month.')
                if _uso_v and _uso_v.get('left_hours') is not None else
                i18n.pick('Earth Engine con la cuenta compartida.',
                          'Earth Engine with the shared account.'))
        if _n_analizar:
            # Unos 3 s por punto con Earth Engine (20 puntos, un minuto); el triple sin él.
            _min = max(1, round(_n_analizar * (3 if backend == 'gee' else 9) / 60))
            _txt_cuenta += ' ' + (i18n.pick('Alrededor de un minuto.', 'About a minute.')
                                  if _min <= 1 else
                                  i18n.pick(f'Unos {_min} minutos.',
                                            f'About {_min} minutes.'))
        st.caption(_txt_cuenta)
    # El mapa siempre a la vista, como en la demo (sin puntos, Sierra Nevada). Solo se
    # dibuja con esta pantalla delante: escondido, Leaflet se queda sin tamaño y no se ve.
    if PANTALLA == 'analizar':
        st_folium(_mapa_previo(tuple(_pts_vista), area_mode.startswith('⨯')), height=680,
                  use_container_width=True, returned_objects=[], key='mapa_previo')
    if input_mode.startswith('🎲'):
        st.caption(i18n.pick('Los puntos se generan al analizar, dentro de tu zona.',
                             'The points are generated when analysing, inside your area.'))
    elif not f_input and not st.session_state.get('_pedir_ejemplo'):
        st.caption(i18n.pick('Sube tu fichero de puntos en «Tus puntos», a la izquierda, o '
                             'genéralos al azar en tu zona. Para probar sin nada, usa el '
                             'ejemplo del Inicio.',
                             'Upload your points file under "Your points", on the left, or '
                             'generate them at random in your area. To try without anything, '
                             'use the example on the start screen.'))
    elif _pts_vista and not area_mode.startswith('⨯'):
        st.caption(i18n.pick('Tu zona se dibuja al analizar.',
                             'Your area is drawn when analysing.'))

# ============================================================
# HELP / WELCOME
# ============================================================
# ============================================================
# CUERPO PRINCIPAL — pestañas: Verdad terreno  |  Mapa Modelo RF  |  Documentación
# «Verdad terreno» es la detección y la revisión de los puntos (tab_analisis);
# «Mapa Modelo RF», el mapa de tipos (tab_tipos).
# ============================================================
# La fila principal tiene que verse más que las pestañas de dentro: letra grande,
# fondo y todo el ancho. Solo ella (selectores de hijo directo). Si no cabe en una
# línea pasa a dos, en vez de cortarse; por eso la barra deslizante de Streamlit se
# cambia por un borde bajo la pestaña activa, que sí funciona con dos líneas.
# Las pestañas de antes ahora son pantallas (cabecera y st.tabs key=pantalla).
tab_tipos = pant_mapa


def parar_ejecucion():
    """Detiene esta ejecución (st.stop). La pestaña «Mapa Modelo RF» se dibuja al
    final del guion: si se para antes, hay que dejarle algo escrito."""
    with tab_tipos:
        st.info(i18n.pick(
            'El último análisis se ha detenido antes de terminar. Mira el aviso en '
            '«1 · Analizar».',
            'The last analysis stopped before finishing. See the message in '
            '"1 · Analyse".'))
    # Parada con su propio aviso: no es un análisis cortado, y la cabecera vuelve a
    # responder.
    st.session_state.pop('_analisis_en_marcha', None)
    if _BLOQUEO is not None:
        _BLOQUEO.empty()
    st.stop()


def render_ayuda():
    """La pantalla Ayuda, como en la demo: el manual, los primeros pasos y los puntos de
    campo; qué modelo elegir según tu zona; sugerir un cambio o informar de un error; esta
    instalación; y debajo, la documentación completa (antes, su pestaña)."""
    import html as _html
    e = enlaces_sugerencias()
    st.title(i18n.pick('Ayuda', 'Help'))
    _tarjetas = [
        (i18n.pick('Manual (wiki)', 'Manual (wiki)'),
         i18n.pick('Cada opción explicada, en español e inglés.',
                   'Every option explained, in Spanish and English.'), WIKI_URL),
        (i18n.pick('Primeros pasos', 'Getting started'),
         i18n.pick('Instalar, abrir y hacer la primera prueba.',
                   'Install, open and run the first test.'),
         WIKI_URL + i18n.pick('/Primeros-pasos', '/Getting-started')),
        (i18n.pick('Cómo preparar tus puntos', 'How to prepare your points'),
         i18n.pick('La plantilla, las columnas y por qué incluir ausencias.',
                   'The template, the columns and why to include absences.'),
         WIKI_URL + i18n.pick('/Puntos-de-campo', '/Field-points'))]
    st.html('<div class="ayuda-tarjetas">' + ''.join(
        f'<a class="ayuda-tarjeta" href="{_html.escape(u)}" target="_blank">'
        f'<span class="t">{_html.escape(ti)}</span><span class="d">{_html.escape(tx)}</span>'
        '</a>' for ti, tx, u in _tarjetas) + '</div>')

    # Qué modelo elegir según tu zona: el resumen; el detalle, en la documentación de abajo.
    # Columnas (la autora, 27-11-2026, en inglés): Model · Variables · AUC (performance),
    # con su «?» · Trained in · Usable in. Cada modelo, en dos líneas: con radar y sin radar
    # (la versión sin radar tiene 5 variables menos, las del radar; la autora: «el número
    # de variables cambiará con o sin radar»). Comprobado en los modelos el 27-11-2026.
    _entrenado = i18n.pick('Sierra Nevada (España)', 'Sierra Nevada (Spain)')
    _modelos_tabla = [
        (i18n.pick('General', 'General'), (27, 0.847), (22, 0.833),
         i18n.pick('En cualquier sitio', 'Anywhere')),
        (i18n.pick('Con infrarrojo', 'With infrared'), (32, 0.945), (27, 0.937),
         i18n.pick('Andalucía, Cataluña y Canarias',
                   'Andalusia, Catalonia and the Canary Islands'))]
    _con_r, _sin_r = i18n.pick('Con radar', 'With radar'), i18n.pick('Sin radar', 'Without radar')
    _cuerpo = ''.join(
        f'<tr><td rowspan="2"><strong>{_html.escape(m)}</strong></td>'
        f'<td class="r">{_con_r}</td><td>{con[0]}</td><td>{_num(con[1])}</td>'
        f'<td rowspan="2">{_html.escape(_entrenado)}</td><td rowspan="2">{_html.escape(u)}</td>'
        f'</tr><tr class="sin"><td class="r">{_sin_r}</td><td>{sin[0]}</td>'
        f'<td>{_num(sin[1])}</td></tr>'
        for m, con, sin, u in _modelos_tabla)
    _q_auc = (
        '<span class="q-html" tabindex="0">?<span class="globo">'
        + i18n.pick(
            'El AUC dice lo bien que el modelo distingue un borreguil de lo que no lo es: '
            '0,5 es como echarlo a cara o cruz y 1 es acertar siempre. Se ha medido en '
            'zonas que el modelo no había visto al entrenar. «Sin radar» es lo que acierta '
            'cuando no se puede leer el radar (Sentinel-1), por ejemplo con Planetary '
            'Computer: entonces la app usa sola una versión del modelo sin él.',
            'The AUC tells how well the model separates an alpine wet meadow from what is '
            'not one: '
            '0.5 is like tossing a coin and 1 is always right. It was measured on areas '
            'the model had not seen while training. "Without radar" is how well it does '
            'when the radar (Sentinel-1) cannot be read, for instance with Planetary '
            'Computer: the app then uses a version of the model without it on its own.')
        + '</span></span>')
    _cab = [_html.escape(i18n.pick('Modelo', 'Model')), '',
            _html.escape(i18n.pick('Variables', 'Variables')),
            _html.escape(i18n.pick('AUC (acierto)', 'AUC (performance)')) + ' ' + _q_auc,
            _html.escape(i18n.pick('Entrenado en', 'Trained in')),
            _html.escape(i18n.pick('Se puede usar en', 'Usable in'))]
    st.html(
        '<section class="ayuda-caja"><span class="t">'
        + i18n.pick('Qué modelo elegir según tu zona', 'Which model to choose for your area')
        + '</span><span>'
        + i18n.pick('La guía de los modelos, cuánto acierta cada uno, las recomendaciones '
                    'por parque y cuántos puntos hacen falta para el mapa están más abajo, '
                    'en «Documentación y recomendaciones».',
                    'The guide to the models, how accurate each one is, the recommendations '
                    'per park and how many points the map needs are further down, under '
                    '"Documentation and recommendations".')
        + '</span><div class="tabla-modelos"><table><tr>'
        + ''.join(f'<th>{c}</th>' for c in _cab) + '</tr>' + _cuerpo
        + '</table></div></section>')

    # El título, del mismo tamaño que «Ayuda» (la autora, 27-11-2026: «poner este título
    # igual de grande que help»): encima de sus dos tarjetas, no dentro (no cabría).
    st.title('💬 ' + i18n.pick('Sugerir un cambio o informar de un error',
                               'Suggest a change or report an error'), anchor=False)
    c_sug, c_inst = st.columns(2, gap='small')
    with c_sug.container(key='tarjeta_ayuda_sugerir'):
        st.html(
            '<div class="ayuda-sug">'
            + f'<a class="enlace" href="{_html.escape(e["issue"])}" target="_blank">'
            + '<strong>' + i18n.pick('En GitHub', 'On GitHub') + '</strong><br><span>'
            + i18n.pick('Público, con una cuenta de GitHub; puedes pegar pantallazos y '
                        'seguir la respuesta.',
                        'Public, with a GitHub account; you can paste screenshots and '
                        'follow the answer.') + '</span></a>'
            + (f'<a class="enlace" href="{_html.escape(e["form"])}" target="_blank">'
               + '<strong>' + i18n.pick('Formulario anónimo', 'Anonymous form')
               + '</strong><br><span>'
               + i18n.pick('Sin cuenta, nombre ni correo, así que no te podrán contestar. '
                           'La versión ya va puesta.',
                           'No account, no name and no email, so nobody can answer you. '
                           'The version is already filled in.') + '</span></a>'
               if e['form'] else '') + '</div>')
        if e['form']:       # los registros de este ordenador, con el formulario anónimo
            st.button(i18n.pick('→ Enviar los registros de este ordenador',
                                '→ Send the logs of this computer') + '\n\n' + i18n.pick(
                'Con el formulario anónimo: los ves antes, sin tu nombre, correo ni '
                'carpetas. También sale debajo de cada error de la app.',
                'With the anonymous form: you see them first, without your name, email or '
                'folders. It also appears under every error of the app.'),
                key='registros_documentacion', type='tertiary', on_click=_abrir_registros,
                width='stretch')
    with c_inst.container(key='tarjeta_ayuda_instalacion'):
        _v = version_app()
        _lineas = [' · '.join(x for x in (i18n.pick(f'Versión {_v}', f'Version {_v}')
                                          if _v else '', fecha_app()) if x)]
        _txt_gh = texto_modelos_github(_modelos_gh)
        if _txt_gh:
            _lineas.append(_txt_gh)
        if backend != 'gee':
            _lineas.append(i18n.pick('Microsoft Planetary Computer, sin cuenta',
                                     'Microsoft Planetary Computer, no account'))
        elif gee_project:
            _lineas.append(i18n.pick(f'Earth Engine con tu proyecto {gee_project}',
                                     f'Earth Engine with your project {gee_project}'))
        else:
            _ee = i18n.pick('Earth Engine: cuenta compartida de MonitorEO',
                            'Earth Engine: MonitorEO shared account')
            _uso = _uso_cuenta_compartida()
            if _uso and _uso.get('left_hours') is not None:
                _ee += i18n.pick(f" · te quedan {_num(_uso['left_hours'])} h este mes",
                                 f" · {_num(_uso['left_hours'])} h left this month")
            _lineas.append(_ee)
        _lineas.append(i18n.pick('Tus modelos y resultados: carpeta Borreguil de tu usuario',
                                 'Your models and results: the Borreguil folder of your '
                                 'user'))
        st.html('<div class="ayuda-inst"><span class="t">'
                + i18n.pick('Esta instalación', 'This installation') + '</span>'
                + ''.join(f'<span>{_html.escape(x)}</span>' for x in _lineas if x)
                + '<span class="nota">'
                + i18n.pick('Se actualiza sola al abrirla. Para abrirla en el navegador: '
                            'Borreguil --navegador.',
                            'It updates itself when opened. To open it in the browser: '
                            'Borreguil --navegador.')
                + '</span></div>')
    st.write('')
    st.markdown('### ' + i18n.pick('Cómo preparar tus puntos',
                                   'How to prepare your points'))
    st.markdown(texto_puntos_campo())
    render_docs(con_sugerencias=False)


with pant_ayuda:
    render_ayuda()

# El botón «Probar con 20 puntos de ejemplo» de la pantalla de inicio deja aquí su
# petición (on_click corre antes que el guion): así esta pasada ya no dibuja la
# pantalla de inicio y lanza el análisis como «Ejecutar pipeline», con los puntos de
# la plantilla, sin área y con lo elegido en el panel lateral.
EJEMPLO_PUNTOS = APP_DIR / 'plantilla' / 'example_sierra_nevada.csv'
PLANTILLA_EXCEL = APP_DIR / 'plantilla' / 'point_data_template.xlsx'
EJEMPLO_EXCEL = APP_DIR / 'plantilla' / 'example_point_data_sierra_nevada.xlsx'
con_ejemplo = bool(st.session_state.pop('_pedir_ejemplo', False)) and EJEMPLO_PUNTOS.exists()


def _pedir_ejemplo():
    st.session_state['_pedir_ejemplo'] = True
    st.session_state['pantalla'] = 'analizar'      # el análisis se ve allí


with pant_inicio:
    with st.container():           # la pantalla de inicio
        # Pantalla de inicio (la de la demo, aprobada por la autora el 26-11-2026): qué
        # hace la app, el enlace al Observatorio y la foto de un borreguil, el ejemplo que
        # se prueba con un clic, los tres pasos y el pie con quién la hace y la financia.
        # La autora (27-11-2026, tarde): debajo del título, los tres pasos (con sus textos,
        # que escribió en inglés); sin la frase de entrada; la foto, abajo. En inglés,
        # «alpine wet meadows» en vez de «borreguil» y «area» en vez de «place». El título
        # en español no cambia: el lanzador lo busca para saber que la app ya ha cargado.
        st.markdown(i18n.pick(
            '<div class="inicio"><span class="ceja">Prados húmedos de alta montaña</span>'
            '<h1>Encuentra los borreguiles de tu zona y haz su mapa</h1></div>',
            '<div class="inicio"><span class="ceja">Alpine wet meadows</span>'
            '<h1>Find the alpine wet meadows in your area and map them</h1></div>'),
            unsafe_allow_html=True)
        _pasos_inicio = [
            (i18n.pick('1 · Prepara tu análisis', '1 · Set your analysis'),
             i18n.pick('Zonas o puntos, modelo y datos para obtener puntos candidatos de '
                       'entrenamiento',
                       'Areas/points, model and data to get candidate training points'),
             i18n.pick('Con relieve, imágenes de satélite y aéreas, un modelo da la '
                       'probabilidad de encontrar los distintos tipos de borreguil en muchos '
                       'puntos.',
                       'With terrain, satellite and aerial imagery a model gives the '
                       'probability to find different types of alpine wet meadows in '
                       'multiple points.')),
            (i18n.pick('2 · Revisa los puntos de entrenamiento', '2 · Review the training points'),
             i18n.pick('Confirma o corrige a la vista cada punto candidato de entrenamiento',
                       'Visually confirm or correct each candidate training point'),
             i18n.pick('Sobre las ortofotos, indica en cada punto si es borreguil (sí / no / '
                       'no estoy seguro/a) y de qué tipo.',
                       'On the orthophotos, indicate for each point whether (yes/no/not sure) '
                       'it corresponds to alpine wet meadows and what type it is.')),
            (i18n.pick('3 · Haz el mapa', '3 · Create the map'),
             i18n.pick('El modelo usa los puntos verificados para hacer el mapa de tu zona.',
                       'The model uses the verified points to create a map of your area.'),
             i18n.pick('En cada píxel de 10 m de tu zona, el modelo da la probabilidad de '
                       'que haya borreguil y su tipo.',
                       'For each 10-metre pixel in your area, the model provides a '
                       'probability of the presence of alpine wet meadows and its type.'))]
        for _k, (_col, (_paso, _tit, _txt)) in enumerate(zip(st.columns(3), _pasos_inicio)):
            with _col.container(key=f'tarjeta_inicio_{_k + 1}'):
                st.markdown(f'#### {_paso}')
                st.markdown(f'**{_tit}**')
                st.caption(_txt)
        st.write('')
        # Los botones, debajo del bloque 1; la foto, centrada debajo de los bloques 2 y 3
        # (la autora, 27-11-2026: «centraría la foto entre los dos bloques de arriba»).
        h_izq, h_der = st.columns([1, 2])
        with h_izq:
            st.markdown(
                f'<div class="inicio"><a class="obsnev" '
                f'href="{i18n.pick(OBSNEV_URL, OBSNEV_URL_EN)}" target="_blank" '
                f'title="{i18n.pick(OBSNEV_TITULO, OBSNEV_TITULO_EN)}">'
                # La autora (27-11-2026): la frase de más información, debajo de la pregunta.
                + i18n.pick('¿Qué es un borreguil?<br>Más información en el Observatorio de '
                            'Cambio Global de Sierra Nevada',
                            'What are alpine wet meadows?<br>More information at the Sierra '
                            'Nevada Global Change Observatory')
                + ' ↗</a></div>', unsafe_allow_html=True)
            # La autora (27-11-2026, tarde; textos suyos en inglés): «Try an example» con
            # la hoja del ejemplo rellena debajo, y «Analyse my points» con la plantilla
            # debajo (con sus tres filas grises de ejemplo: le gustan, 27-11-2026).
            _xlsx = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            with st.container(key='botones_inicio'):
                if EJEMPLO_PUNTOS.exists():
                    with st.container(horizontal=True, vertical_alignment='center',
                                      gap='small', key='fila_ejemplo'):
                        st.button(
                            i18n.pick('**Probar un ejemplo**\n\n20 puntos de Sierra Nevada · '
                                      'un minuto',
                                      '**Try an example**\n\n20 points from Sierra Nevada · '
                                      'one minute'),
                            key='probar_ejemplo', on_click=_pedir_ejemplo, type='primary')
                        ayuda_q('q_ejemplo',
                                'Usa un modelo y datos ambientales (por ejemplo, de satélite) '
                                'para analizar 20 puntos con presencia (12 puntos) y ausencia '
                                '(8 puntos) de borreguil y obtener un mapa continuo de tu '
                                'zona.',
                                'It uses a model and environmental (e.g. satellite) data to '
                                'analyse 20 points with presence (12 points) and absence (8 '
                                'points) of alpine wet meadows to produce a continuous map for '
                                'your area.')
                    if EJEMPLO_EXCEL.exists():
                        st.download_button(
                            i18n.pick('Descargar la hoja de datos de puntos del ejemplo',
                                      'Download the point data sheet for the example'),
                            EJEMPLO_EXCEL.read_bytes(), file_name=EJEMPLO_EXCEL.name,
                            mime=_xlsx, key='bajar_ejemplo', type='tertiary')
                st.button(
                    i18n.pick('**Analizar mis puntos**\n\nKML, GeoJSON, Shapefile o CSV',
                              '**Analyse my points**\n\nKML, GeoJSON, Shapefile or CSV'),
                    key='analizar_mis_puntos', on_click=ir_a, args=('analizar',))
                if PLANTILLA_EXCEL.exists():
                    with st.container(horizontal=True, vertical_alignment='center',
                                      gap='small'):
                        st.download_button(
                            i18n.pick('Descargar la plantilla de datos de puntos',
                                      'Download the point data template'),
                            PLANTILLA_EXCEL.read_bytes(), file_name=PLANTILLA_EXCEL.name,
                            mime=_xlsx, key='bajar_plantilla', type='tertiary')
                        ayuda_q('q_plantilla',
                                'Una fila por punto: coordenadas (longitud y latitud en grados '
                                'decimales), presencia (1 = sí, 0 = no) e información adicional '
                                'opcional. Cuando la completes, súbela como archivo CSV (usa el '
                                'punto como separador decimal).',
                                'One row per point: Coordinates (longitude and latitude in '
                                'decimal degrees), Presence (1=yes, 0=no), and optional '
                                'additional information. Once completed, upload it as a CSV '
                                'file (use point as decimal separator).')
        with h_der:
            st.html(foto_inicio())
        if PANTALLA == 'inicio':
            with pie_pantalla:
                st.html(pie_inicio(backend, gee_project))

    # ============================================================
    # PIPELINE EXECUTION
    # ============================================================
with col_principal:
    def save_uploaded(f, suffix):
        if f is None: return None
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(f.getvalue()); tmp.close()
        return tmp.name


    def save_uploaded_area(files):
        """Guarda los ficheros del área (lista de UploadedFile, p. ej. las partes de un
    shapefile) en un directorio temporal conservando nombres y devuelve la ruta a
    leer: el .shp si existe (los .shx/.dbf/.prj se resuelven solos), el .zip, o el
    único KML/GeoJSON/GeoPackage."""
        if not files:
            return None
        if not isinstance(files, (list, tuple)):
            files = [files]
        if not files:
            return None
        d = tempfile.mkdtemp(prefix='area_')
        saved = []
        for f in files:
            dest = os.path.join(d, os.path.basename(f.name))
            with open(dest, 'wb') as out:
                out.write(f.getvalue())
            saved.append(dest)
        for ext in ('.shp', '.zip'):
            for s in saved:
                if s.lower().endswith(ext):
                    return s
        return saved[0]

    if (run_btn and can_run) or con_ejemplo:
        # Mientras se analiza, la cabecera (pasos, Ayuda, ES|EN) y «Analizar» no
        # responden: un clic cortaría el análisis (ver _CSS_BLOQUEO). Al terminar, la
        # pasada siguiente los vuelve a dibujar como siempre. Se apunta que está en
        # marcha: si no llega al final, la pasada siguiente lo avisa.
        st.session_state['_analisis_en_marcha'] = True
        st.session_state.pop('_aviso_cortado', None)
        if _AVISO_CORTADO is not None:
            _AVISO_CORTADO.empty()
        _BLOQUEO = st.empty()
        _BLOQUEO.markdown(f'<style>{_CSS_BLOQUEO}</style>', unsafe_allow_html=True)

        # Workspace
        work = Path(tempfile.mkdtemp(prefix='borreguil_'))
        st.session_state.work = work

        if con_ejemplo:
            # El ejemplo: los puntos de la plantilla (llevan su etiqueta de campo), sin
            # verdad-terreno aparte ni área, aunque el panel lateral tenga algo puesto.
            st.info(i18n.pick(
                'Analizando los **20 puntos de ejemplo** de Sierra Nevada (12 borreguiles '
                'y 8 zonas sin borreguil)…',
                'Analysing the **20 example points** from Sierra Nevada (12 alpine wet '
                'meadows and 8 areas without them)…'))
            input_path, truth_path, area_path = str(EJEMPLO_PUNTOS), None, None
        else:
            try:                       # con su nombre: un Shapefile son varios ficheros
                input_path = _guardar_subida(f_input)
                truth_path = _guardar_subida(f_truth)
            except ValueError as e:
                st.error(str(e))
                parar_ejecucion()
            area_path  = save_uploaded_area(f_area) if f_area else None
            if area_path and bp.falta_sistema_de_referencia(area_path):
                st.error(bp.aviso_sin_sistema_de_referencia(os.path.basename(area_path),
                                                            i18n.get_lang() == 'en'))
                parar_ejecucion()
        st.caption(i18n.pick('Cuando termine, se abre «Resultados». Hasta entonces, los '
                             'botones de arriba no responden.',
                             'When it finishes, "Results" opens. Until then, the buttons at '
                             'the top do not respond.'))

        # Resolve study area FIRST (may be needed for generation)
        study_geom = None; study_info = None
        if not con_ejemplo and (area_path or wdpa_id or pa_name):
            with st.status('Resolviendo área de estudio…', expanded=True) as status:
                try:
                    import study_area as sa
                    # El código WDPA, en la capa WCMC/WDPA de Earth Engine con la cuenta
                    # compartida de MonitorEO, elija quien usa la app lo que elija para
                    # calcular (después, al leer los datos, se vuelve a su cuenta).
                    _use_gee = False
                    if wdpa_id:
                        try:
                            import gee_backend as geb
                            _use_gee = geb.initialize()[0]
                        except Exception:                   # noqa: BLE001
                            _use_gee = False
                    study_geom, study_info = sa.resolve(area_path,
                                                        wdpa_id.strip() if wdpa_id else None,
                                                        pa_name.strip() if pa_name else None,
                                                        wdpa_token=None, use_gee=_use_gee)
                    if study_geom:
                        st.write(f"**Origen**: {study_info['source']}")
                        if study_info.get('name'):
                            st.write(f"**Nombre**: {study_info['name']}")
                        st.write(f"**Área aprox**: {_num(study_info['area_km2'])} km²")
                        status.update(label='✓ Área de estudio resuelta', state='complete')
                    else:
                        if wdpa_id:
                            st.warning(i18n.pick(
                                'No se ha encontrado ese código WDPA en la base mundial de '
                                'espacios protegidos (o no se ha podido conectar con Earth '
                                'Engine). Comprueba el número en protectedplanet.net, o busca '
                                'el espacio por su nombre o sube su polígono. Se usará el '
                                'rectángulo que rodea a tus puntos.',
                                'That WDPA code was not found in the World Database on '
                                'Protected Areas (or Earth Engine could not be reached). '
                                'Check the number at protectedplanet.net, or search the area '
                                'by name or upload its polygon. The rectangle around your '
                                'points will be used.'))
                        elif pa_name:
                            st.warning('No se encontró un polígono para ese nombre. Prueba a '
                                       'ser más específico (incluye "Parque Nacional/Natural") '
                                       'o sube un vector. Se usará bbox de los puntos.')
                        else:
                            st.warning('No se pudo resolver el área. Se usará bbox de los puntos.')
                        status.update(label='⚠ Área no resuelta', state='error')
                except Exception as e:
                    st.error(f'Error: {e}')
                    status.update(label='⚠ Error', state='error')

        # Load or generate points
        if input_mode.startswith('🎲') and not con_ejemplo:
            if not study_geom:
                st.error('Para generar puntos hace falta un área de estudio válida.')
                parar_ejecucion()
            if gen_mode == 'cuartiles' and not gen_capa:
                st.error(i18n.pick(
                    'Para lanzar puntos por cuartiles, sube la capa de polígonos en el panel '
                    'lateral (Datos de entrada) y elige su columna.',
                    'To place points by quartiles, upload the polygon layer in the side '
                    'panel (Inputs) and choose its column.'))
                parar_ejecucion()
            with st.status(f'Generando {gen_n} puntos aleatorios ({gen_mode})…', expanded=True) as status:
                import random_points
                # Lejos de los puntos de campo, si se ha pedido y están subidos
                _excluir = []
                if gen_lejos and truth_path:
                    _excluir = [(float(r['lon']), float(r['lat']))
                                for r in bp.read_points(truth_path)]
                elif gen_lejos:
                    st.warning('Para alejar los puntos de tus puntos de campo hace falta '
                               'subir la verdad-terreno: se generan sin ese límite.')
                rows, gen_stats = random_points.generate(study_geom, gen_n,
                                                mode=gen_mode,
                                                min_elev=gen_min_elev,
                                                max_elev=gen_max_elev,
                                                min_dist_m=int(gen_min_dist),
                                                seed=int(gen_seed),
                                                excluir=_excluir, lejos_m=float(gen_lejos),
                                                capa=gen_capa, frac_q1=gen_frac_q1)
                status.update(label=f'{len(rows)} puntos generados'
                              + (f' a {gen_lejos:.0f} m o más de tus {len(_excluir)} '
                                 'puntos de campo' if _excluir else ''), state='complete')

            # El filtro altitudinal dejó 0 puntos → avisar con el rango real del área
            if not rows:
                reason = gen_stats.get('reason')
                amax = gen_stats.get('area_elev_max')
                amin = gen_stats.get('area_elev_min')
                if reason == 'capa_fuera_del_area':
                    st.warning(i18n.pick(
                        '⚠ Ningún polígono de la capa cae dentro del área de estudio. '
                        'Comprueba que la capa y el área son de la misma zona.',
                        '⚠ No polygon of the layer falls inside the study area. Check that '
                        'the layer and the area are of the same place.'))
                elif reason == 'min_elev_above_area' and amax is not None:
                    st.warning(
                        f'⚠ El área de estudio **no alcanza la altitud mínima** de '
                        f'**{gen_min_elev:.0f} m** que has indicado. La altitud máxima del '
                        f'área es **≈ {amax:.0f} m**. Baja la **altitud mínima** por debajo '
                        f'de ese valor y vuelve a ejecutar.')
                elif reason == 'max_elev_below_area' and amin is not None:
                    st.warning(
                        f'⚠ El área de estudio **supera la altitud máxima** de '
                        f'**{gen_max_elev:.0f} m** indicada. La altitud mínima del área es '
                        f'**≈ {amin:.0f} m**. Sube la **altitud máxima** por encima de ese '
                        f'valor y vuelve a ejecutar.')
                elif amin is not None and amax is not None:
                    st.warning(
                        f'⚠ Ningún punto cumple el filtro altitudinal. El área va de '
                        f'**≈ {amin:.0f}** a **≈ {amax:.0f} m**. Ajusta los límites de '
                        f'altitud.')
                elif reason == 'no_elevation_data':
                    st.warning(
                        '⚠ No se pudo obtener la altitud del área (¿zona de agua, fuera de '
                        'cobertura SRTM, o fallo temporal de la API de elevación?). '
                        'Prueba sin filtro de altitud o repite en unos minutos.')
                else:
                    st.warning('⚠ No se generaron puntos. Revisa el área y los filtros de '
                               'altitud.')
                parar_ejecucion()

            st.write(f'**{len(rows)} puntos generados**')
            _cu = gen_stats.get('cuartiles')
            if _cu:
                st.write(i18n.pick(
                    f"{_cu[1]['salen']} en el cuartil 1 ({_cu[1]['poligonos']} polígonos, "
                    f"≈ {_num(_cu[1]['ha'])} ha) y {_cu[2]['salen']} en los cuartiles 2 a 4 "
                    f"({_cu[2]['poligonos']} polígonos, ≈ {_num(_cu[2]['ha'])} ha), dentro "
                    'del área de estudio.',
                    f"{_cu[1]['salen']} in quartile 1 ({_cu[1]['poligonos']} polygons, "
                    f"≈ {_num(_cu[1]['ha'])} ha) and {_cu[2]['salen']} in quartiles 2 to 4 "
                    f"({_cu[2]['poligonos']} polygons, ≈ {_num(_cu[2]['ha'])} ha), inside "
                    'the study area.'))
                for _g, _nom_g in ((1, i18n.pick('el cuartil 1', 'quartile 1')),
                                   (2, i18n.pick('los cuartiles 2 a 4',
                                                 'quartiles 2 to 4'))):
                    if _cu[_g]['salen'] < _cu[_g]['pedidos']:
                        st.warning(i18n.pick(
                            f"⚠ En {_nom_g} solo han salido {_cu[_g]['salen']} de "
                            f"{_cu[_g]['pedidos']} puntos"
                            + (': no hay polígonos de ese grupo en el área de estudio.'
                               if not _cu[_g]['poligonos'] else
                               ': los demás no cumplen la altitud o la distancia a tus '
                               'puntos de campo.'),
                            f"⚠ In {_nom_g} only {_cu[_g]['salen']} of "
                            f"{_cu[_g]['pedidos']} points came out"
                            + (': there are no polygons of that group in the study area.'
                               if not _cu[_g]['poligonos'] else
                               ': the others do not meet the elevation or the distance to '
                               'your field points.')))
            if rows and 'elev_m_initial' in rows[0]:
                elevs = [r.get('elev_m_initial') for r in rows if r.get('elev_m_initial')]
                if elevs:
                    st.write(f'Rango altitudinal: {min(elevs):.0f} – {max(elevs):.0f} m '
                              f'(mediana {sorted(elevs)[len(elevs)//2]:.0f} m)')
        else:
            with st.status('Leyendo puntos…', expanded=True) as status:
                rows = bp.read_points(input_path)
                st.write(f'{len(rows)} puntos cargados')
                for _niv_et, _txt_et in bp.resumen_etiquetas(bp.leer_etiquetas(rows)):
                    {'ok': st.write, 'info': st.info, 'aviso': st.warning}[_niv_et](_txt_et)
                status.update(label='✓ Lectura completada', state='complete')

        # Puntos de campo: se leen igual que «Tus puntos» (24-11) y se dice qué se ha
        # leído; antes, una ausencia escrita 0.0 o en una columna `Presence` se tomaba
        # por borreguil sin avisar.
        if truth_path:
            tr = bp.read_points(truth_path)
            _leido = {}
            n_match, n_add, n_pos, n_neg = bp.merge_truth_points(rows, tr, info=_leido)
            for _niv_et, _txt_et in (_resumen_campo(_leido) if _leido else []):
                {'ok': st.write, 'info': st.info, 'aviso': st.warning}[_niv_et](_txt_et)
            _dud = _leido.get('n_duda') or 0
            st.write(i18n.pick(
                f'**Puntos de campo**: {n_pos} borreguiles y {n_neg} no borreguiles'
                + (f', {_dud} de ellos dudosos (no cuentan)' if _dud else '')
                + f' ({n_match} coinciden con tus puntos, {n_add} añadidos). '
                f'Total: {len(rows)} puntos.',
                f'**Field points**: {n_pos} borreguiles and {n_neg} non-borreguiles'
                + (f', {_dud} of them doubtful (they do not count)' if _dud else '')
                + f' ({n_match} match your points, {n_add} added). '
                f'Total: {len(rows)} points.'))

        if study_geom:
            import study_area as sa
            n_in = sa.points_inside(rows, study_geom)
            st.write(f"**Puntos dentro del área**: {n_in}/{len(rows)} "
                      f"({100*n_in/len(rows):.0f}%)")

        bbox = bp.bbox_from_geom(study_geom) if study_geom else bp.bbox_with_buffer(rows)
        st.session_state.bbox = bbox
        st.session_state.study_geom = study_geom
        st.session_state.study_info = study_info

        # Backend satelital
        active_backend = backend
        if backend == 'gee':
            with st.status('Inicializando Google Earth Engine…', expanded=True) as status:
                try:
                    import gee_backend as geb
                    # En deploy online, las credenciales de service account pueden venir
                    # de st.secrets["EE_SERVICE_ACCOUNT_KEY"] (en local no hace falta).
                    sa = get_secret('EE_SERVICE_ACCOUNT_KEY', None)
                    proj = gee_project.strip() or get_secret('GEE_PROJECT', None)
                    ok, msg = geb.initialize(proj, service_account_json=sa)
                    if ok:
                        if gee_project.strip():
                            guardar_ajuste('gee_project', gee_project.strip())
                        st.write(f'✓ {msg}')
                        status.update(label='✓ GEE inicializado', state='complete')
                    else:
                        st.warning(f'{msg}\n\nUsando Microsoft Planetary Computer en su lugar.')
                        active_backend = 'mpc'
                        status.update(label='⚠ GEE no disponible → MPC', state='error')
                except Exception as e:
                    st.warning(f'GEE falló ({e}). Usando MPC.')
                    active_backend = 'mpc'
                    status.update(label='⚠ GEE error → MPC', state='error')

        # Barra de progreso global del pipeline (hitos por fase). La fase Sentinel-2,
        # la más larga, se actualiza en continuo vía callback.
        pbar = barra_progreso('Iniciando extracción de variables…')
        def set_prog(frac, text):
            try: pbar.progress(max(0.0, min(1.0, frac)), text=text)
            except Exception: pass

        # Imágenes aéreas (ESRI o PNOA IGN)
        if not skip_imgs:
            set_prog(0.04, 'Imágenes aéreas (textura/patrón)…')
            _img_label = 'PNOA IGN 0,25 m (España)' if img_src == 'pnoa' else 'ESRI World Imagery'
            with st.status(f'Descargando imágenes {_img_label}…', expanded=False) as status:
                t0 = time.time()
                try:
                    with live_logs(status):
                        if img_src == 'pnoa':
                            bp.fetch_pnoa_imgs(rows, work / 'imgs')
                        else:
                            bp.fetch_esri_imgs(rows, work / 'imgs')
                        bp.compute_img_features(rows, work / 'imgs')
                    status.update(label=f'✓ Imágenes + texturas ({time.time()-t0:.0f}s)',
                                  state='complete')
                except Exception as e:
                    status.update(label=f'⚠ Imágenes fallaron ({e}), continuando sin texturas',
                                  state='error')

        # Ríos y arroyos de OpenStreetMap, para el ambiente «arroyo» (los de Sierra
        # Nevada van con la app; en otras zonas se descargan)
        with st.status(i18n.pick('Cauces (OpenStreetMap)…', 'Watercourses (OpenStreetMap)…'),
                       expanded=False) as status:
            try:
                _lin, _orig = bp.cauces_osm(bbox, APP_DIR / bp.CAUCES_SN)
                _n_c = bp.attach_cauce_distance(rows, _lin)
                if _lin is None:
                    status.update(label=i18n.pick(
                        '⚠ No se han podido leer los ríos y arroyos de OpenStreetMap (sin '
                        'conexión con Overpass): «arroyo» se decide solo por la pendiente.',
                        '⚠ The OpenStreetMap rivers and streams could not be read (no '
                        'connection to Overpass): "stream" is decided by the slope only.'),
                        state='error')
                else:
                    status.update(label=i18n.pick(
                        f'✓ Cauces: {_n_c} ríos y arroyos de OpenStreetMap'
                        + (' (incluidos en la app)' if _orig == 'capa' else ''),
                        f'✓ Watercourses: {_n_c} OpenStreetMap rivers and streams'
                        + (' (included in the app)' if _orig == 'capa' else '')),
                        state='complete')
            except Exception as e:
                bp.attach_cauce_distance(rows, None)
                status.update(label=i18n.pick(
                    f'⚠ Los cauces de OpenStreetMap han fallado ({e}): «arroyo» se decide '
                    'solo por la pendiente.',
                    f'⚠ The OpenStreetMap watercourses failed ({e}): "stream" is decided '
                    'by the slope only.'), state='error')

        # DEM / topografía
        set_prog(0.20, 'Topografía (DEM)…')
        with st.status(f'Topografía ({active_backend.upper()})…', expanded=False) as status:
            try:
                with live_logs(status):
                    if active_backend == 'gee':
                        import gee_backend as geb
                        geb.fetch_topo_gee(rows, bbox)
                    else:
                        bp.fetch_topo(rows, bbox, work)
                status.update(label='✓ Topografía completada', state='complete')
            except Exception as e:
                status.update(label=f'⚠ Topografía falló: {e}', state='error')

        # Sentinel-2
        st.session_state.active_backend = active_backend
        if not skip_s2:
            set_prog(0.28, 'Óptico Sentinel-2 (NDVI, Clre…)…')
            with st.status(f'Óptico Sentinel-2 multitemporal ({active_backend.upper()})…', expanded=True) as status:
                try:
                    years = bp.parse_years(years_str)
                    # callback: mapea el progreso interno de S2 al tramo 0,28–0,68
                    def s2_prog(frac, msg):
                        set_prog(0.28 + 0.40 * frac, msg)
                    with live_logs(status):
                        if active_backend == 'gee':
                            import gee_backend as geb
                            geb.fetch_s2_gee(rows, bbox, years=years, progress=s2_prog)
                        else:
                            bp.fetch_s2(rows, bbox, years=years, progress=s2_prog)
                    cov = _coverage(rows, ['ndvi_early', 'ndvi_late'])
                    if cov == 0:
                        status.update(label=f'🔴 Óptico Sentinel-2: 0/{len(rows)} puntos con NDVI '
                                      '(ver Diagnóstico abajo)', state='error')
                    elif cov < len(rows):
                        status.update(label=f'🟡 Óptico Sentinel-2: {cov}/{len(rows)} puntos con '
                                      f'NDVI ({len(years)} años)', state='complete')
                    else:
                        status.update(label=f'✓ Óptico Sentinel-2: {cov}/{len(rows)} puntos '
                                      f'({len(years)} años)', state='complete')
                except Exception as e:
                    status.update(label=f'⚠ Óptico Sentinel-2 falló: {e}', state='error')
                    st.error(f'**Sentinel-2 falló:** {e}')

        # La pureza de cada punto se propone como en el mapa: con el NDVI de inicio de
        # verano de SU píxel de 10 m y si ese píxel o uno de sus vecinos es agua. Solo
        # con Earth Engine; sin ello, los puntos se quedan sin propuesta de pureza.
        if not skip_s2 and active_backend == 'gee':
            set_prog(0.69, i18n.pick('Pureza: el píxel de 10 m de cada punto…',
                                     'Purity: the 10 m pixel of each point…'))
            with st.status(i18n.pick('Pureza: el píxel de 10 m de cada punto (Earth Engine)…',
                                     'Purity: the 10 m pixel of each point (Earth Engine)…'),
                           expanded=False) as status:
                try:
                    import gee_backend as geb
                    _n_pz = geb.fetch_pureza_gee(rows, years=bp.parse_years(years_str))
                    status.update(label=i18n.pick(
                        f'✓ Pureza: {_n_pz}/{len(rows)} puntos con el NDVI de su píxel',
                        f'✓ Purity: {_n_pz}/{len(rows)} points with the NDVI of their pixel'),
                        state='complete')
                except Exception as e:
                    status.update(label=i18n.pick(
                        f'⚠ Pureza: no se pudo leer el píxel ({e}); los puntos se quedan '
                        'sin propuesta de pureza',
                        f'⚠ Purity: the pixel could not be read ({e}); the points get no '
                        'purity proposal'), state='error')

        # Sentinel-1 (radar): solo con Earth Engine, en segundos (los modelos de partida
        # se entrenaron con esos valores). Con Microsoft Planetary Computer no se lee y se usa la
        # versión del modelo sin radar (ver modelo_segun_radar). Si falla, igual.
        skip_s1 = skip_s2 or active_backend != 'gee'
        if not skip_s1:
            set_prog(0.70, 'Radar Sentinel-1…')
            with st.status('Radar Sentinel-1 (Earth Engine)…', expanded=False) as status:
                try:
                    with live_logs(status):
                        import gee_backend as geb
                        geb.fetch_s1_gee(rows, bbox)
                    cov = _coverage(rows, bp.S1_FEATS)
                    status.update(label=f'✓ Radar Sentinel-1: {cov}/{len(rows)} puntos',
                                  state='complete')
                except Exception as e:
                    status.update(label=f'⚠ Radar Sentinel-1 falló (se continúa sin radar): {e}',
                                  state='error')

        # Detectar qué fuentes extra necesita el modelo de partida elegido
        _sel_feats = st.session_state.get('sel_model_feats', [])
        model_needs_cir = any(f.startswith('cir_') for f in _sel_feats)
        model_needs_ps  = any(f.startswith('ps_')  for f in _sel_feats)

        # PNOA Falso Color IR 0,25 m (CIR): gratis. Se extrae si el modelo lo usa.
        if model_needs_cir and not fast_mode:
            set_prog(0.84, 'PNOA Falso Color IR 0,25 m (CIR)…')
            with st.status('PNOA Falso Color IR 0,25 m (CIR, gratis)…', expanded=False) as status:
                try:
                    with live_logs(status):
                        n_cir = bp.fetch_cir(rows, bbox)
                    if n_cir:
                        status.update(label=f'✓ CIR 0,25 m ({n_cir} puntos con dato)',
                                      state='complete')
                    else:
                        status.update(label='⚠ Sin ortofoto IR WMS para esta zona '
                                      '(Andalucía/Cataluña/Canarias): mejor el modelo «General»',
                                      state='error')
                except Exception as e:
                    status.update(label=f'⚠ CIR falló (se continúa sin IR): {e}',
                                  state='error')

        # PlanetScope 3 m: si el usuario lo pidió (checkbox) o el modelo lo requiere.
        if (use_planet or model_needs_ps) and _pl_key and not fast_mode:
            with st.status('PlanetScope 8b SR (Orders API, 10-30 min)…', expanded=True) as status:
                try:
                    with live_logs(status):
                        n_ps = bp.fetch_planet(rows, api_key=_pl_key)
                    status.update(label=f'✓ PlanetScope 3 m ({n_ps} puntos con dato)',
                                  state='complete')
                except Exception as e:
                    status.update(label=f'⚠ PlanetScope falló (se continúa sin 3 m): {e}',
                                  state='error')
        elif model_needs_ps and not _pl_key:
            st.warning('El modelo elegido usa PlanetScope 3 m pero no hay PL_API_KEY: '
                       'esas 10 variables irán vacías y la predicción será menos precisa.')

        # Informe de extracción (SIEMPRE visible): qué fuentes funcionaron y cobertura
        set_prog(0.92, 'Clasificando (Random Forest)…')
        render_extraction_report(rows, skip_s2=skip_s2, skip_imgs=skip_imgs,
                                 skip_s1=skip_s1)

        # RF
        with st.status('Random Forest…', expanded=True) as status:
            # Sin radar (Microsoft Planetary Computer, o falló): la versión del modelo sin radar
            st.session_state.radar_leido = bp.hay_radar(rows)
            _smp = modelo_segun_radar(
                st.session_state.get('start_model_path') or str(model_path('rf_sierra_nevada.joblib')),
                st.session_state.radar_leido)
            proba, info = bp.train_rf(rows, default_model_path=_smp,
                                      propio=bool(st.session_state.get('entrenar_propio')),
                                      entrenamiento=tabla_entrenamiento())
            if proba is not None:
                for i, r in enumerate(rows):
                    r['rf_proba'] = float(proba[i])
                if info:
                    st.markdown(_titulo_modelo(info))
                    if info.get('mode') == 'pretrained':
                        st.write(f'Entrenamiento original: n = {info["training_n"]} · '
                                  f'AUC GroupKFold = {info["auc_groupkfold"]}')
                    elif info.get('mode') == 'trained':
                        auc = info.get('auc', float('nan'))
                        if auc == auc and info.get('sumado'):
                            st.write(i18n.pick(
                                f'AUC en tus puntos con validación por zonas '
                                f'({info.get("cv", "")}): **{_num(auc)}**',
                                f'AUC on your points with spatial validation '
                                f'({info.get("cv", "")}): **{_num(auc)}**'))
                        elif auc == auc:  # not NaN
                            st.write(i18n.pick(
                                f'AUC con validación por zonas ({info.get("cv", "")}), contra '
                                f'ausencias y pseudo-ausencias: **{_num(auc)}**',
                                f'AUC with spatial validation ({info.get("cv", "")}), against '
                                f'absences and pseudo-absences: **{_num(auc)}**'))
                        if not info.get('sumado') and info["n_pos"] < 30:
                            st.warning('⚠ Pocos positivos (< 30). Recomendado: 30–150 para '
                                        'un modelo fiable en tu zona.')
                    st.caption(i18n.pick(
                        'Por qué este modelo y cuánto acierta con tus puntos de campo: '
                        'en «Modelo usado», debajo de los resultados.',
                        'Why this model and how well it does on your field points: in '
                        '"Model used", below the results.'))
                status.update(label='✓ RF completado', state='complete')
            else:
                for r in rows: r['rf_proba'] = float('nan')
                status.update(label='⚠ RF no ejecutado', state='error')

        set_prog(1.0, '✓ Proceso completado')

        # Umbral RF: manual, automático por Otsu, o automático por verdad-terreno.
        _probas = [r.get('rf_proba') for r in rows]
        if thr_mode.startswith('Automático · Otsu'):
            threshold = bp.auto_threshold(_probas)
            st.info(i18n.pick(
                f'⚙ Umbral automático (Otsu): **{_n3(threshold)}** '
                '(corte que mejor separa los dos grupos de probabilidad).',
                f'⚙ Automatic threshold (Otsu): **{_n3(threshold)}** '
                '(the cut that best separates the two groups of probability).'))
        elif thr_mode.startswith('Automático · verdad'):
            _t = bp.auto_threshold_truth(rows)
            if _t is not None:
                threshold = _t
                st.info(i18n.pick(
                    f'⚙ Umbral por verdad-terreno: **{_n3(threshold)}** '
                    '(óptimo según tus puntos de campo).',
                    f'⚙ Ground-truth threshold: **{_n3(threshold)}** '
                    '(the best one for your field points).'))
            else:
                threshold = bp.auto_threshold(_probas)
                st.warning(i18n.pick(
                    'No hay suficiente verdad-terreno etiquetada para fijar el '
                    f'umbral; se usa Otsu: **{_n3(threshold)}**.',
                    'There is not enough labelled ground truth to set the '
                    f'threshold; Otsu is used: **{_n3(threshold)}**.'))
        st.session_state.eff_threshold = threshold

        # Decision
        for r in rows:
            r['decision'] = bp.decide(r, threshold=threshold)

        # Distancia a las lagunas conocidas (capa incluida de Sierra Nevada).
        # Mejora la detección del ambiente «laguna» en la clasificación jerárquica.
        try:
            _bundled = APP_DIR / 'lagunas_sierra_nevada.kml'
            _lag_path = str(_bundled) if _bundled.exists() else None
            if _lag_path:
                _n_lag = bp.attach_laguna_distance(rows, bp.read_points(_lag_path))
                if _n_lag:
                    st.caption(f'🌊 {_n_lag} lagunas de referencia usadas para el ambiente.')
        except Exception as _e:
            st.caption(f'(no se pudo usar la capa de lagunas: {_e})')

        # Tipo de borreguil (ambiente/humedad/pureza). La app lo PROPONE por reglas
        # y respeta lo que una persona haya revisado (p. ej. al volver a cargar unos
        # puntos ya trabajados en otra sesión).
        _rev_malos = bp.apply_typology(rows, pp.is_borreguil_decision)
        if _rev_malos:
            _lista = ', '.join(str(x) for x in _rev_malos[:8])
            st.warning(i18n.pick(
                f'{len(_rev_malos)} punto(s) venían marcados como revisados pero con '
                f'una categoría no admitida, y no cuentan como revisados: {_lista}.',
                f'{len(_rev_malos)} point(s) were marked as reviewed but carry a '
                f'category that is not allowed, so they do not count as reviewed: '
                f'{_lista}.'))

        # Verdad-terreno "solo entrenar": el modelo ya se entrenó con ella arriba;
        # ahora se quitan del conjunto los puntos AÑADIDOS desde el fichero de campo
        # (source=='truth') para no saturar mapa/tabla/descargas. Los candidatos que
        # coincidieron con un punto de campo conservan su source y siguen visibles.
        if truth_train_only:
            _n0 = len(rows)
            rows = [r for r in rows if r.get('source') != 'truth']
            _n_hidden = _n0 - len(rows)
            if _n_hidden:
                st.info(i18n.pick(
                    f'🎓 {_n_hidden} puntos ya verificados no se muestran en el mapa, la tabla '
                    f'ni las descargas (sí cuentan para medir el acierto). Se muestran tus '
                    f'{len(rows)} puntos.',
                    f'🎓 {_n_hidden} already verified points are not shown on the map, table '
                    f'or downloads (they do count to measure the accuracy). Your {len(rows)} '
                    f'points are shown.'))

        # Save outputs
        bp.save_csv(rows, work / 'classification.csv')
        bp.save_points_geojson(rows, work / 'puntos.geojson')
        bp.save_xlsx(rows, work / 'Clasificacion_puntos.xlsx', threshold=threshold)
        bp.save_map(rows, work / 'mapa.html', study_geom=study_geom)
        if study_geom is not None:
            from shapely.geometry import mapping
            import json as _json
            with open(work / 'area_estudio.geojson', 'w') as fh:
                _json.dump({'type':'Feature','properties':study_info or {},
                              'geometry': mapping(study_geom)}, fh)

        # --- Estado de la sesión: lo verificado y cada vez que se recalcula ---
        # IDs de los borreguiles VERIFICADOS en campo (inmutables; siempre positivos).
        st.session_state.base_truth_ids = {
            r.get('ID') for r in rows if str(r.get('Borreguil','')).lower() == 'si'
        }
        counts0 = dict(Counter(r['decision'] for r in rows))
        n_pos0 = sum(1 for r in rows if str(r.get('Borreguil','')).lower() == 'si')
        st.session_state.iter_history = [{
            'iter': 1,
            'mode': 'inicial',
            'cutoff': None,
            'info': info,
            'counts': counts0,
            'n_train_pos': n_pos0,
            'n_new': None,
        }]
        st.session_state.iter_n = 1
        st.session_state.sel_point = None
        st.session_state.sel_idx = None
        st.session_state.prev_sel_set = set()
        st.session_state['_vaciar_seleccion'] = True  # la selección era de la tabla anterior
        st.session_state.model_bundle = (info or {}).get('model_bundle')
        st.session_state.rows = rows
        st.session_state.info = info
        # Cómo ha ido la lectura, para el resumen de Resultados («Todo ha ido bien…» y
        # «Ver detalles», 27-11-2026, la demo): antes solo se veía aquí mientras corría.
        # Lo que se puede contar con los puntos (cuántos tienen dato de cada fuente) se
        # cuenta allí; aquí, lo que solo se sabe durante el análisis.
        _g = globals()
        try:
            _imgs = [] if skip_imgs else list((work / 'imgs').glob('*.jpg'))
            _seg_imgs = (max(p.stat().st_mtime for p in _imgs) - _g['t0']
                         if _imgs and 't0' in _g else None)
        except Exception:                                   # noqa: BLE001
            _seg_imgs = None
        try:
            _n_anios = len(bp.parse_years(years_str))
        except Exception:                                   # noqa: BLE001
            _n_anios = None
        st.session_state['diag_analisis'] = {
            'ejemplo': bool(con_ejemplo),
            'fichero': (Path(str(_g.get('input_path'))).name
                        if _g.get('input_path') and not con_ejemplo else None),
            'area': (study_info or {}).get('name'),
            'backend': active_backend, 'cuenta': _g.get('cuenta'),
            'skip_imgs': bool(skip_imgs), 'skip_s2': bool(skip_s2),
            'skip_s1': bool(skip_s1), 'seg_imgs': _seg_imgs,
            'cauces_ok': _g.get('_lin') is not None, 'n_cauces': _g.get('_n_c'),
            'cauces_capa': _g.get('_orig') == 'capa', 'anios': _n_anios,
            'umbral': threshold,
            'umbral_modo': ('manual' if not thr_mode.startswith('Automático')
                            else 'otsu' if thr_mode.startswith('Automático · Otsu')
                            or _g.get('_t') is None else 'verdad'),
            'n_lagunas': _g.get('_n_lag'),
            'cir': bool(model_needs_cir and not fast_mode),
            'ps': bool((use_planet or model_needs_ps) and _pl_key and not fast_mode)}
        st.session_state.pop('_analisis_en_marcha', None)   # ha llegado al final
        ir_a_luego('resultados')      # el análisis ha terminado: a Resultados

with pant_resultados:
    # ============================================================
    # RESULTS DISPLAY
    # ============================================================
    if 'rows' in st.session_state:
        import math as _m
        rows = st.session_state.rows
        work = st.session_state.work
        sg = st.session_state.get('study_geom')
        # Umbral efectivo (el automático calculado en la ejecución manda sobre el slider)
        threshold = st.session_state.get('eff_threshold', threshold)
        # El modelo elegido o, si en el análisis no se leyó el radar, su versión sin radar
        MODEL_PATH = modelo_segun_radar(
            st.session_state.get('start_model_path') or str(model_path('rf_sierra_nevada.joblib')),
            st.session_state.get('radar_leido', True))

        # La selección de la tabla va por POSICIÓN. Cuando cambia la lista de puntos
        # (ejecución nueva, o se quita alguno) las posiciones se corren y la selección
        # pasaría a apuntar a puntos que nadie ha elegido: «Quitar» borraría otro y
        # el tipo revisado se guardaría en otro. Se vacía antes de dibujar la tabla.
        if st.session_state.pop('_vaciar_seleccion', False):
            st.session_state['results_table'] = {
                'selection': {'rows': [], 'columns': [], 'cells': []}}

        # En pantallas estrechas los rótulos de las tarjetas se cortaban con «…»
        # («POSIBLE BORRE…»): se dejan partir en varias líneas.
        st.markdown('<style>[data-testid="stMetricLabel"] p{white-space:normal!important;'
                    'overflow:visible!important;text-overflow:clip!important;'
                    'overflow-wrap:anywhere;}</style>', unsafe_allow_html=True)

        # ----------------------------------------------------------------
        # Modelo usado y recálculo tras verificar, añadir o quitar puntos
        # ----------------------------------------------------------------
        iter_n = st.session_state.get('iter_n', 1)
        hist = st.session_state.get('iter_history', [])
        base_truth_ids = st.session_state.get('base_truth_ids', set())
        # La casilla del panel: sumar los puntos de la sesión a los del modelo.
        _propio = bool(st.session_state.get('entrenar_propio'))

        def _recalcular_y_guardar(rows2, new_base, modo, n_new):
            """Tras verificar, añadir o quitar puntos, o al cambiar la casilla de sumar:
            probabilidad y decisión de todos los puntos, descargas guardadas y estado de
            la sesión al día. Devuelve la información del modelo usado."""
            info2 = bp.run_rf_and_decide(rows2, threshold=threshold,
                                         default_model_path=MODEL_PATH, propio=_propio,
                                         entrenamiento=tabla_entrenamiento())
            for r in rows2:
                if (r.get('ID') in new_base
                        and bp._txt(r.get('Duda')).lower() != 'si'):
                    r['decision'] = 'BORREGUIL VERIFICADO'; r['origin'] = 'verificado'
            bp.apply_typology(rows2, pp.is_borreguil_decision)
            bp.save_csv(rows2, work / 'classification.csv')
            bp.save_points_geojson(rows2, work / 'puntos.geojson')
            bp.save_xlsx(rows2, work / 'Clasificacion_puntos.xlsx', threshold=threshold)
            bp.save_map(rows2, work / 'mapa.html', study_geom=sg)
            st.session_state.rows = rows2
            st.session_state.base_truth_ids = new_base
            st.session_state.iter_n = iter_n + 1
            st.session_state.model_bundle = (info2 or {}).get('model_bundle')
            st.session_state.iter_history = hist + [{
                'iter': iter_n + 1, 'mode': modo, 'cutoff': None, 'info': info2,
                'counts': dict(Counter(r['decision'] for r in rows2)),
                'n_train_pos': sum(1 for r in rows2
                                   if bp._txt(r.get('Borreguil')).lower() == 'si'
                                   and bp._txt(r.get('Duda')).lower() != 'si'),
                'n_new': n_new,
            }]
            return info2

        # Qué modelo ha dado la probabilidad (el de la última vez que se ejecutó o se
        # reentrenó) y cuánto acierta con los puntos de campo.
        _info_modelo = (hist[-1].get('info') if hist else None) or st.session_state.get('info')
        _sumado = bool((_info_modelo or {}).get('sumado'))

        # ================================================================
        # Pantalla «Resultados» como en la demo (27-11-2026): título, resumen del
        # proceso, cuatro tarjetas, el modelo usado (al pulsar su tarjeta), quién
        # revisa y, debajo, las vistas en píldoras con «Siguiente paso» a la derecha.
        # Los estilos van aquí, solo para las claves de esta pantalla (res_…).
        # ================================================================
        st.html('<style>' + '''
.res-titulo{display:flex;flex-wrap:wrap;align-items:baseline;gap:12px;margin:2px 0 0}
.res-titulo h1{margin:0!important;padding:0!important;font-family:Montserrat,Inter,
  sans-serif!important;font-size:28px!important;font-weight:700!important;
  letter-spacing:-.01em;color:#F5F7F8!important;line-height:1.25!important}
.res-titulo span{font-size:14px;color:#AEB6BE}
.st-key-res_proceso details,.st-key-res_proceso_aviso details{border-radius:12px;
  border:1px solid #3E6B4F;background:#1B2B21}
.st-key-res_proceso_aviso details{border-color:#6B5A2A;background:#2A2416}
.st-key-res_proceso summary,.st-key-res_proceso_aviso summary{padding:11px 14px;
  min-height:44px}
.st-key-res_proceso summary p,.st-key-res_proceso_aviso summary p{font-size:14px;
  color:#ECEFF1}
.st-key-res_proceso summary p::before,.st-key-res_proceso_aviso summary p::before{
  content:"●";color:#7CC496;margin-right:10px}
.st-key-res_proceso_aviso summary p::before{color:#F5B83D}
.st-key-res_proceso summary svg,.st-key-res_proceso_aviso summary svg,
.st-key-res_proceso summary [data-testid="stIconMaterial"],
.st-key-res_proceso_aviso summary [data-testid="stIconMaterial"]{display:none}
.st-key-res_proceso summary::after,.st-key-res_proceso_aviso summary::after{
  content:var(--res-ver);margin-left:auto;padding-left:12px;font-size:13px;
  font-weight:600;color:#7CC496;white-space:nowrap}
.st-key-res_proceso_aviso summary::after{color:#F5B83D}
.st-key-res_proceso details[open] summary::after,
.st-key-res_proceso_aviso details[open] summary::after{content:var(--res-ocultar)}
.res-proceso{display:flex;flex-wrap:wrap;gap:24px;font-size:13px;color:#CDD3D9;
  padding:0 0 4px 24px}
.res-proceso ul{margin:0;padding-left:16px;display:flex;flex-direction:column;gap:4px;
  flex:1 1 300px}
.res-proceso li{font-size:13px;margin:0;padding:0}
.res-proceso table{font-size:13px;flex:1 1 300px;border-collapse:collapse}
.res-proceso th{text-align:left;padding:4px 10px 4px 0;color:#AEB6BE;font-weight:500}
.res-proceso th+th,.res-proceso td+td{text-align:right;padding-right:0}
.res-proceso td{padding:4px 10px 4px 0}
.res-proceso .nota{flex:1 1 100%;font-size:12px;color:#E9DDBF}
.res-tarjeta{background:#2E2A2B;border:1px solid #403A3C;border-radius:16px;padding:18px;
  display:flex;flex-direction:column;gap:4px;min-height:132px;box-sizing:border-box}
.res-tarjeta .r{font-size:13px;color:#AEB6BE;font-weight:500}
.res-tarjeta .v{font-family:Montserrat,Inter,sans-serif;font-size:40px;
  font-weight:700;line-height:1.1;white-space:nowrap}
.res-tarjeta .s{font-size:13px;color:#AEB6BE}
.st-key-res_modelo button{width:100%;min-height:132px;display:flex;
  flex-direction:column;align-items:flex-start;justify-content:flex-start;
  background:#2E2A2B;border:1px solid #403A3C;border-radius:16px;padding:18px;
  text-align:left}
.st-key-res_modelo button:hover{border-color:#3E6B4F;background:#2E2A2B}
.st-key-res_modelo button>div{width:100%;justify-content:flex-start}
.st-key-res_modelo button>div>span{width:100%}
.st-key-res_modelo button [data-testid="stMarkdownContainer"]{text-align:left;width:100%}
.st-key-res_modelo button p{margin:0 0 4px!important;text-align:left;font-size:13px;
  font-weight:500;color:#AEB6BE;line-height:1.4}
.st-key-res_modelo button p:nth-of-type(2){font-family:Montserrat,Inter,sans-serif;
  font-size:22px;font-weight:700;color:#ECEFF1;line-height:1.25}
.st-key-res_modelo button p:nth-of-type(3){color:#7CC496;font-weight:400}
.st-key-res_modelo_abierto details:not([open]){display:none}
.st-key-res_modelo_abierto details{background:#2E2A2B;border:1px solid #403A3C;
  border-radius:16px}
.st-key-res_modelo_abierto summary p{font-weight:700;font-size:15px;color:#ECEFF1}
.st-key-res_modelo_abierto [data-testid="stMarkdownContainer"] p,
.st-key-res_modelo_abierto [data-testid="stMarkdownContainer"] li{font-size:13px;
  color:#CDD3D9}
.st-key-res_modelo_abierto table{border-collapse:collapse;font-size:13px;width:auto}
.st-key-res_modelo_abierto th{color:#AEB6BE;font-weight:500;text-align:left;
  background:none;border:0;padding:6px 16px 6px 0}
.st-key-res_modelo_abierto td{border:0;padding:6px 16px 6px 0;color:#ECEFF1}
.st-key-res_cert details{border:0;background:none}
.st-key-res_cert summary{padding:6px 0;min-height:32px}
.st-key-res_cert summary p{font-size:13px!important;font-weight:600!important;
  color:#A3D9B5!important}
.st-key-res_cert table{border:1px solid #403A3C;border-radius:10px;width:100%}
.st-key-res_cert th{padding:7px 10px;border-bottom:1px solid #403A3C}
.st-key-res_cert td{padding:7px 10px;border-bottom:1px solid #393335}
.st-key-res_reentrenar{border-top:1px solid #403A3C;padding-top:10px}
.st-key-res_quien{background:#282526;border:1px solid #403A3C;border-radius:14px;
  padding:12px 16px}
.st-key-res_quien [data-testid="stCaptionContainer"] p{font-size:12px;color:#AEB6BE}
.st-key-vistas_resultados [data-baseweb="tab-list"]{gap:6px;flex-wrap:wrap}
.st-key-vistas_resultados [data-baseweb="tab-highlight"],
.st-key-vistas_resultados [data-baseweb="tab-border"]{display:none}
.st-key-vistas_resultados button[data-baseweb="tab"]{border:1px solid #494245;
  background:#2E2A2B;border-radius:999px;padding:8px 16px;min-height:40px;height:auto;
  margin:0}
.st-key-vistas_resultados button[data-baseweb="tab"] p{font-size:14px;font-weight:500;
  color:#CDD3D9}
.st-key-vistas_resultados button[data-baseweb="tab"][aria-selected="true"]{
  background:#ECEFF1;border-color:#ECEFF1}
.st-key-vistas_resultados button[data-baseweb="tab"][aria-selected="true"] p{
  color:#221F20;font-weight:700}
.st-key-res_aside{margin-top:var(--res-aside, 58px)}
.st-key-res_siguiente{background:#1B2B21;border:1.5px solid #7CC496;border-radius:16px;
  padding:20px}
.res-sig .e{font-size:13px;font-weight:600;color:#7CC496}
.res-sig .t{font-family:Montserrat,Inter,sans-serif;font-size:20px;font-weight:700;
  color:#F5F7F8;margin:8px 0 6px}
.res-sig .d{font-size:14px;line-height:1.5;color:#CDD3D9}
.st-key-ir_revisar button{min-height:48px;border-radius:12px}
.st-key-ir_revisar button p{font-size:15px}
.st-key-res_despues{background:#2E2A2B;border:1px solid #403A3C;border-radius:16px;
  padding:18px}
.res-des .t{font-weight:700;font-size:15px;color:#ECEFF1;margin-bottom:6px}
.res-des .d{font-size:13px;color:#CDD3D9}
.st-key-ir_mapa button,.st-key-res_revisarlo button,.st-key-remove_map_pt button,
.st-key-res_revisar_tabla button{padding:0;min-height:0;border:0;background:none}
.st-key-ir_mapa button p,.st-key-res_revisarlo button p,.st-key-res_revisar_tabla button p{
  font-size:13px;font-weight:600;color:#7CC496}
.st-key-remove_map_pt button p{font-size:13px;font-weight:600;color:#E8A3A3}
.st-key-ir_mapa button:hover p,.st-key-res_revisarlo button:hover p{color:#A3D9B5}
[class*="st-key-res_tarjeta_"]{background:#2E2A2B;border:1px solid #403A3C;
  border-radius:16px;padding:16px}
.res-t{font-weight:700;font-size:14px;color:#ECEFF1}
.res-t15{font-weight:700;font-size:15px;color:#ECEFF1}
.res-ej{display:block;font-size:12px;line-height:1.5;color:#98A2AB}
.res-muted{font-size:12px;line-height:1.5;color:#AEB6BE}
.st-key-res_anadir{background:#2E2A2B;border:1px solid #403A3C;border-radius:14px;
  padding:14px 16px}
.st-key-modo_anadir label p{font-size:14px;font-weight:600}
.st-key-res_anadir [data-testid="stMarkdownContainer"] p{font-size:13px;color:#CDD3D9}
.st-key-res_respuestas button{min-height:44px;border-radius:10px;border:1px solid #60575A;
  background:#393335}
.st-key-res_respuestas button p{font-size:13px;font-weight:600;color:#ECEFF1}
.st-key-mapa_si button{border:1.5px solid #7CC496;background:#1E3327}
.st-key-mapa_cancelar button,.st-key-mapa_deshacer button{min-height:40px;
  border-radius:10px;border:1px solid #60575A;background:#393335}
.st-key-mapa_cancelar button p,.st-key-mapa_deshacer button p{font-size:13px;
  font-weight:500;color:#CDD3D9}
.st-key-dist_vista [data-testid="stButtonGroup"]>div,
.st-key-dist_comparar [data-testid="stButtonGroup"]>div{gap:6px;flex-wrap:wrap}
.st-key-dist_vista button,.st-key-dist_comparar button{border:1px solid #494245!important;
  background:#393335;border-radius:10px!important;min-height:44px;padding:9px 10px}
.st-key-dist_vista button p,.st-key-dist_comparar button p{font-size:13px;font-weight:500;
  color:#CDD3D9}
.st-key-dist_vista button[kind$="Active"],.st-key-dist_comparar button[kind$="Active"]{
  border:1.5px solid #7CC496!important;background:#1E3327}
.st-key-dist_vista button[kind$="Active"] p,.st-key-dist_comparar button[kind$="Active"] p{
  color:#ECEFF1;font-weight:600}
.st-key-dist_n input{font-size:15px;font-weight:700}
.st-key-res_tarjeta_descargas [data-testid="stDownloadButton"] button{padding:0;
  min-height:0;border:0;background:none;justify-content:flex-start;text-align:left}
.st-key-res_tarjeta_descargas [data-testid="stDownloadButton"] button p{text-align:left}
.st-key-res_tarjeta_descargas [data-testid="stDownloadButton"] button p{font-size:14px;
  color:#7CC496;text-decoration:underline;text-underline-offset:3px}
.st-key-res_tarjeta_descargas [data-testid="stDownloadButton"] button:hover p{color:#A3D9B5}
.st-key-res_tarjeta_guardar [data-testid="stCaptionContainer"] p{font-size:13px}
''' + '</style>'
            + '<style>:root{--res-ver:"' + i18n.pick('Ver detalles', 'See details')
            + '";--res-ocultar:"' + i18n.pick('Ocultar', 'Hide') + '"}</style>')

        _N = len(rows)
        _diag = st.session_state.get('diag_analisis') or {}

        # --- Título y de qué ejecución se trata («20 puntos de ejemplo · Sierra Nevada»)
        import html as _hres
        if _diag.get('ejemplo'):
            _desc = i18n.pick(f'{_N} puntos de ejemplo · Sierra Nevada',
                              f'{_N} example points · Sierra Nevada')
        else:
            _lugar = str(_diag.get('area') or '').split(',')[0].strip() or _diag.get('fichero')
            _desc = (i18n.pick(f'{_N} puntos', f'{_N} points')
                     + (f' · {_lugar}' if _lugar else ''))
        st.html('<div class="res-titulo"><h1>'
                + i18n.pick('Análisis terminado', 'Analysis finished')
                + f'</h1><span>{_hres.escape(_desc)}</span></div>')

        # --- Cómo ha ido la lectura: una línea y, al abrirla, el detalle (lo que antes
        # solo se veía en Analizar mientras corría: «🔍 Diagnóstico de extracción»).
        def _lista_y(cosas):
            cosas = list(cosas)
            if len(cosas) < 2:
                return ''.join(cosas)
            return ', '.join(cosas[:-1]) + i18n.pick(' y ', ' and ') + cosas[-1]

        def _resumen_proceso():
            """(frase, ¿todo bien?, html del detalle)."""
            _img = list(bp.FEATURES_RF[:12])
            _topo = ['elev_dem_m', 'slope_deg', 'curvature']
            _s1 = list(bp.S1_FEATS)
            _s2 = [f for f in bp.FEATURES_RF if f not in _img + _topo + _s1]
            sin_diag = not _diag
            radar_no = (_diag.get('skip_s1') if not sin_diag
                        else not st.session_state.get('radar_leido', True))
            fuentes = [
                ('img', i18n.pick('Imagen aérea', 'Aerial image'),
                 i18n.pick('imagen aérea', 'aerial image'), _img, _diag.get('skip_imgs')),
                ('topo', i18n.pick('Relieve', 'Terrain'), i18n.pick('relieve', 'terrain'),
                 _topo, False),
                ('s2', i18n.pick('Óptico Sentinel-2', 'Optical Sentinel-2'),
                 'Sentinel-2', _s2, _diag.get('skip_s2')),
                ('s1', 'Radar Sentinel-1', 'radar', _s1, radar_no)]
            if _diag.get('cir'):
                fuentes.append(('cir', 'CIR (PNOA)', 'CIR', list(bp.CIR_FEATS), False))
            if _diag.get('ps'):
                fuentes.append(('ps', 'PlanetScope', 'PlanetScope', list(bp.PS_FEATS), False))
            filas, bien, mal, problema = [], [], [], False
            for _k, nombre, corto, feats, omitida in fuentes:
                cov = _coverage(rows, feats)
                if omitida or (sin_diag and cov == 0):
                    filas.append((nombre, i18n.pick('no se ha leído', 'not read'), '#98A2AB'))
                    continue
                color = ('#7CC496' if cov == _N else '#F5B83D' if cov >= 0.3 * _N
                         else '#F28B88')
                problema = problema or cov < 0.3 * _N
                filas.append((nombre, i18n.pick(f'{cov} de {_N}', f'{cov} of {_N}'), color))
                (bien if cov == _N else mal).append((corto, cov))
            if not mal:
                frase = (i18n.pick('**Todo ha ido bien:** ', '**All went well:** ')
                         + _lista_y(c for c, _ in bien)
                         + i18n.pick(f' en los {_N} puntos.', f' in the {_N} points.'))
            else:
                frase = (i18n.pick('**Revisa la lectura:** ', '**Check the reading:** ')
                         + '; '.join(i18n.pick(f'{c} en {n} de {_N} puntos',
                                               f'{c} in {n} of {_N} points')
                                     for c, n in mal) + '.')
            # El detalle, paso a paso
            pasos = []
            if not sin_diag:
                if _diag.get('backend') == 'gee':
                    pasos.append(i18n.pick(
                        'Puntos leídos y Earth Engine conectado ('
                        + ('tu cuenta' if _diag.get('cuenta') == 'mia'
                           else 'cuenta compartida') + ')',
                        'Points read and Earth Engine connected ('
                        + ('your account' if _diag.get('cuenta') == 'mia'
                           else 'shared account') + ')'))
                else:
                    pasos.append(i18n.pick(
                        'Puntos leídos; satélite de Microsoft Planetary Computer',
                        'Points read; satellite data from Microsoft Planetary Computer'))
                if _diag.get('skip_imgs'):
                    pasos.append(i18n.pick('Imagen aérea: no se ha leído',
                                           'Aerial image: not read'))
                elif _diag.get('seg_imgs') is not None:
                    pasos.append(i18n.pick(
                        f"Imagen aérea y texturas: {max(1, round(_diag['seg_imgs']))} s",
                        f"Aerial image and textures: {max(1, round(_diag['seg_imgs']))} s"))
                if _diag.get('cauces_ok'):
                    pasos.append(i18n.pick(
                        f"Ríos y arroyos: {_diag.get('n_cauces')} de OpenStreetMap"
                        + (' (incluidos en la app)' if _diag.get('cauces_capa') else ''),
                        f"Rivers and streams: {_diag.get('n_cauces')} from OpenStreetMap"
                        + (' (included in the app)' if _diag.get('cauces_capa') else '')))
                else:
                    pasos.append(i18n.pick(
                        'Ríos y arroyos: no se han podido leer («arroyo» se decide solo por '
                        'la pendiente)',
                        'Rivers and streams: could not be read ("stream" is decided by the '
                        'slope only)'))
            # Relieve, Sentinel-2, pureza y radar: cuántos puntos los tienen todos
            _grupos, _nom = [_topo], [i18n.pick('Relieve', 'Terrain')]
            if not _diag.get('skip_s2'):
                _grupos.append(_s2)
                _an = _diag.get('anios')
                _nom.append(i18n.pick('Óptico Sentinel-2', 'Optical Sentinel-2')
                            + (i18n.pick(f' ({_an} años)', f' ({_an} years)') if _an else ''))
                if _diag.get('backend') == 'gee' or (sin_diag and _coverage(
                        rows, ['ndvi_early_pixel'])):
                    _grupos.append(['ndvi_early_pixel'])
                    _nom.append(i18n.pick('pureza del píxel', 'pixel purity'))
            if not radar_no:
                _grupos.append(_s1)
                _nom.append('Radar Sentinel-1')
            _todos = sum(1 for r in rows if all(_coverage([r], g) for g in _grupos))
            pasos.append(_lista_y(_nom) + ': ' + i18n.pick(f'{_todos} de {_N}',
                                                              f'{_todos} of {_N}'))
            _u = _num(_diag.get('umbral', threshold))
            _modo = _diag.get('umbral_modo')
            _txt_u = (i18n.pick(f'Umbral automático (Otsu): {_u}',
                                f'Automatic threshold (Otsu): {_u}') if _modo == 'otsu'
                      else i18n.pick(f'Umbral según tus puntos de campo: {_u}',
                                     f'Threshold from your field points: {_u}')
                      if _modo == 'verdad'
                      else i18n.pick(f'Umbral: {_u}', f'Threshold: {_u}'))
            if _diag.get('n_lagunas'):
                _txt_u += i18n.pick(
                    f" · {_diag['n_lagunas']} lagunas de referencia para el ambiente",
                    f" · {_diag['n_lagunas']} reference lakes for the environment")
            pasos.append(_txt_u)
            esc = _hres.escape
            det = ('<div class="res-proceso"><ul>'
                   + ''.join(f'<li>{esc(p)}</li>' for p in pasos) + '</ul><table><tr><th>'
                   + i18n.pick('Fuente', 'Source') + '</th><th>'
                   + i18n.pick('Puntos con dato', 'Points with data') + '</th></tr>'
                   + ''.join(f'<tr><td>{esc(n)}</td><td style="color:{c}">{esc(v)}</td></tr>'
                             for n, v, c in filas) + '</table>'
                   + (('<span class="nota">' + esc(i18n.pick(
                       'Alguna fuente ha quedado casi vacía: lo que falta se rellena con la '
                       'mediana, así que la predicción puede ser poco fiable. Reintenta, '
                       'usa menos puntos o cambia la fuente de datos en «1 · Analizar», '
                       '«Datos de satélite».',
                       'Some source came out almost empty: what is missing is filled with '
                       'the median, so the prediction may be unreliable. Retry, use fewer '
                       'points or change the data source in "1 · Analyse", "Satellite '
                       'data".')) + '</span>') if problema else '')
                   + '</div>')
            return frase, not mal, det

        _frase_proc, _proc_bien, _det_proc = _resumen_proceso()
        with st.expander(_frase_proc,
                         key='res_proceso' if _proc_bien else 'res_proceso_aviso'):
            st.html(_det_proc)

        # --- Las cuatro tarjetas del resumen
        _ac_todo = bp.aciertos_en_campo((_info_modelo or {}).get('evaluacion'),
                                        threshold) or {}
        _usado_m = 'propio' if (_info_modelo or {}).get('mode') == 'trained' else 'partida'
        _ac = _ac_todo.get(_usado_m) or next(iter(_ac_todo.values()), None)

        def _tarjeta(rotulo, valor, sub, color):
            return (f'<div class="res-tarjeta"><span class="r">{_hres.escape(rotulo)}</span>'
                    f'<span class="v" style="color:{color}">{_hres.escape(str(valor))}</span>'
                    f'<span class="s">{_hres.escape(sub)}</span></div>')

        def _acierta(ok, n, fem):
            if not n:
                return i18n.pick('ninguna en tus puntos' if fem else 'ninguno en tus puntos',
                                 'none in your points')
            if ok == n and n == 1:
                return i18n.pick('el modelo la acierta' if fem else 'el modelo lo acierta',
                                 'the model gets it right')
            if ok == n:
                return i18n.pick(f"el modelo acierta {'las' if fem else 'los'} {n}",
                                 f'the model gets all {n} right')
            return i18n.pick(f'el modelo acierta {ok}', f'the model gets {ok} right')

        if _ac and (_ac['n_si'] or _ac['n_no']):
            _t1 = _tarjeta(i18n.pick('Borreguiles de campo', 'Field borreguiles'),
                           _ac['n_si'], _acierta(_ac['ok_si'], _ac['n_si'], False), '#7CC496')
            _t2 = _tarjeta(i18n.pick('Sin borreguil', 'Without borreguil'), _ac['n_no'],
                           _acierta(_ac['ok_no'], _ac['n_no'], True), '#F28B88')
            _t3 = _tarjeta(i18n.pick('Acierto con tus puntos', 'Accuracy on your points'),
                           i18n.pick(f"{_ac['ok_si'] + _ac['ok_no']} de "
                                     f"{_ac['n_si'] + _ac['n_no']}",
                                     f"{_ac['ok_si'] + _ac['ok_no']} of "
                                     f"{_ac['n_si'] + _ac['n_no']}"),
                           'AUC ' + _auc_txt(_ac.get('auc')), '#F5F7F8')
        else:
            # Sin puntos de campo, lo que dice el modelo
            _n_si_m = sum(1 for r in rows if _es_borreguil(r))
            _n_no_m = sum(1 for r in rows if 'no borreguil' in str(r.get('decision')).lower())
            _de_n = i18n.pick(f'según el modelo, de {_N}', f'according to the model, of {_N}')
            _t1 = _tarjeta(i18n.pick('Borreguiles', 'Borreguiles'), _n_si_m, _de_n, '#7CC496')
            _t2 = _tarjeta(i18n.pick('Sin borreguil', 'Without borreguil'), _n_no_m, _de_n,
                           '#F28B88')
            _t3 = _tarjeta(i18n.pick('Acierto con tus puntos', 'Accuracy on your points'),
                           '—', i18n.pick('sin puntos de campo: revisa algunos',
                                          'no field points: review some'), '#F5F7F8')
        # «Modelo usado»: la tarjeta entera es un botón que abre (y cierra) el acierto
        _im = _info_modelo or {}
        _zm = _nombre_partida(_im)
        if _sumado:
            _tit_m = i18n.pick(f'{_zm} + tus puntos', f'{_zm} + your points')
            _n_m = (_im.get('n_tabla') or 0) + (_im.get('n_pos') or 0) + (_im.get('n_neg') or 0)
            _auc_m = _im.get('auc')
        elif _im.get('mode') == 'trained':
            _tit_m = i18n.pick('Tus puntos', 'Your points')
            _n_m = (_im.get('n_pos') or 0) + (_im.get('n_neg') or 0)
            _auc_m = _im.get('auc')
        else:
            _tit_m = _zm + (i18n.pick(' (sin radar)', ' (no radar)')
                            if _es_sin_radar(_im) else '')
            _n_m, _auc_m = _im.get('training_n'), _im.get('auc_groupkfold')
        try:
            _auc_m = float(_auc_m)
            _auc_m = None if _auc_m != _auc_m else _auc_m
        except (TypeError, ValueError):
            _auc_m = None
        _abierto_m = bool(st.session_state.get('res_modelo_abierto'))
        _linea_m = (i18n.pick('Ocultar el acierto', 'Hide the accuracy') if _abierto_m else
                    ' · '.join(([i18n.pick(f'{_n_m} puntos', f'{_n_m} points')] if _n_m else [])
                               + ([f'AUC {_auc_txt(_auc_m)}'] if _auc_m is not None else [])
                               + [i18n.pick('ver acierto', 'see accuracy')]))

        def _abrir_modelo():
            st.session_state['res_modelo_abierto'] = not st.session_state.get(
                'res_modelo_abierto', False)

        _c1, _c2, _c3, _c4 = st.columns(4, gap='small')
        _c1.html(_t1)
        _c2.html(_t2)
        _c3.html(_t3)
        _c4.button(i18n.pick('Modelo usado', 'Model used') + '\n\n' + _tit_m + '\n\n'
                   + _linea_m, key='res_modelo', on_click=_abrir_modelo, width='stretch',
                   help=None)

        # --- El modelo usado y su acierto con tus puntos de campo: cerrado de entrada;
        # se abre con la tarjeta «Modelo usado» (y se cierra con ella o con su título).
        with st.expander(i18n.pick('Modelo usado y acierto con tus puntos de campo',
                                   'Model used and accuracy on your field points'),
                         key='res_modelo_abierto', on_change='rerun'):
            _msg_modelo = st.session_state.pop('_msg_modelo', None)
            if _msg_modelo:
                st.success(_msg_modelo)
            # La casilla «Sumar mis puntos…» se puede marcar o desmarcar con los
            # resultados ya en pantalla: antes no pasaba nada hasta verificar otro
            # punto. Ahora se ofrece aplicarlo en el momento.
            _n_etq = sum(1 for r in rows
                         if bp._txt(r.get('Borreguil')).lower() in ('si', 'no')
                         and bp._txt(r.get('Duda')).lower() != 'si')
            _pendiente = bool(_info_modelo) and _propio != _sumado
            mostrar_modelo(_info_modelo, threshold,
                           casilla=(('boton' if _n_etq else 'sin_puntos')
                                    if _pendiente and _propio else None))
            # Como en la demo, siempre a la vista; el botón solo se puede pulsar
            # cuando la casilla no coincide con el modelo que se está usando.
            with st.container(key='res_reentrenar', horizontal=True,
                              vertical_alignment='center'):
                st.markdown(i18n.pick(
                    'Si cambias la casilla de sumar tus puntos con los resultados en '
                    'pantalla, se aplica sin volver a analizar:',
                    'If you change the box to add your points with the results on screen, '
                    'it is applied without analysing again:'))
                if st.button(
                        i18n.pick(f'Reentrenar ahora sumando mis {_n_etq} puntos',
                                  f'Retrain now adding my {_n_etq} points') if not _sumado
                        else i18n.pick('Volver al modelo de partida, sin mis puntos',
                                       'Back to the starting model, without my points'),
                        type='primary', key='aplicar_casilla', width='content',
                        disabled=not (_pendiente and (_n_etq or not _propio))):
                    with st.spinner(i18n.pick('Reentrenando…', 'Retraining…') if _propio
                                    else i18n.pick('Calculando…', 'Computing…')):
                        _inf = _recalcular_y_guardar(
                            [dict(r) for r in rows], set(base_truth_ids),
                            'sumar tus puntos' if _propio else 'modelo de partida', 0)
                    st.session_state['_msg_modelo'] = (
                        i18n.pick('Modelo reentrenado sumando tus puntos.',
                                  'Model retrained adding your points.')
                        if (_inf or {}).get('sumado') else
                        i18n.pick('Probabilidades calculadas de nuevo con el modelo de '
                                  'partida, sin tus puntos.',
                                  'Probabilities computed again with the starting model, '
                                  'without your points.'))
                    st.rerun()

        # ----------------------------------------------------------------
        # Helpers de mapa
        # ----------------------------------------------------------------
        def _sin_prediccion(r):
            return 'SIN PRED' in str(r.get('decision', 'SIN PREDICCIÓN'))

        def _add_points(fmap, pts, highlight_id=None, radius=7, hueco=False):
            # Como en la demo (27-11-2026): círculos de 18 px, relleno y borde del color
            # del punto (verde si es borreguil, rojo si no; gris sin predicción), con un
            # halo oscuro para que se vean sobre la ortofoto, y el globo oscuro.
            # hueco=True: el punto resaltado se dibuja sin relleno, para que el
            # marcador no tape justo el terreno que hay que mirar.
            _rotulos = i18n.pick(
                {'tipo': 'Tipo', 'revisado': 'revisado',
                 'variables': 'Las que más pesan en el modelo',
                 'prob': 'Probabilidad de borreguil', 'propuesta': '(propuesta de la app)'},
                {'tipo': 'Type', 'revisado': 'reviewed',
                 'variables': 'Those that weigh most in the model',
                 'prob': 'Probability of borreguil', 'propuesta': "(the app's proposal)"})
            # En el globo (25-11-2026, la autora): el punto, por su número («Punto 663»)
            # y no por el nombre de la capa, y las 3 variables que más pesan en el
            # modelo usado, una por grupo de parecidas, como en la tabla de «Revisar
            # puntos» (antes: altitud, pendiente, NDVI y Clre, fijas).
            _ids_pop = [r.get('ID') for r in pts]
            _nombre_pop = dict(zip(map(str, _ids_pop), bp.nombres_de_punto(
                _ids_pop, i18n.pick('Punto', 'Point'))))
            _rs_pop = load_ref_stats(MODEL_PATH)
            _vars_pop = [(_var(k['feat']), k['feat']) for k in pp.variables_clave(
                (_rs_pop or {}).get('importances'),
                set((_rs_pop or {}).get('model_features') or []) or None, 3)]
            for r in pts:
                _si = _es_borreguil(r)
                color = '#98A2AB' if _sin_prediccion(r) else COLOR_SI if _si else COLOR_NO
                # Números con 3 decimales y nombres largos partidos en líneas.
                popup = bp.popup_html(r, i18n.tv, _rotulos,
                                      nombre=_nombre_pop.get(str(r.get('ID'))),
                                      variables=_vars_pop, num=_num, oscuro=True,
                                      es_borreguil=_si)
                is_sel = (highlight_id is not None and r.get('ID') == highlight_id)
                if is_sel:
                    folium.CircleMarker([r['lat'], r['lon']], radius=radius + 6,
                                        color='#FFFFFF', weight=3, fill=False).add_to(fmap)
                folium.CircleMarker([r['lat'], r['lon']], radius=radius + 3, color='#000000',
                                    weight=2, opacity=0.45, fill=False,
                                    interactive=False).add_to(fmap)
                _relleno = not (hueco and is_sel)
                folium.CircleMarker([r['lat'], r['lon']], radius=radius, color=color,
                                    weight=3, fill=True, fill_color=color,
                                    fill_opacity=1.0 if _relleno else 0.0,
                                    popup=folium.Popup(popup, max_width=280)).add_to(fmap)

        def _add_legend(fmap, rows=None):
            """La leyenda (abajo a la izquierda), con los mismos colores que los puntos:
            «Borreguil» y «No borreguil» (y «Sin predicción» si hay alguno). Además, el
            estilo oscuro de la demo para el globo y el panel de capas («Capas»)."""
            filas = [(COLOR_SI, i18n.pick('Borreguil', 'Borreguil')),
                     (COLOR_NO, i18n.pick('No borreguil', 'Not a borreguil'))]
            if rows is not None and any(_sin_prediccion(r) for r in rows):
                filas.append(('#98A2AB', i18n.pick('Sin predicción', 'No prediction')))
            html = (
                '<div class="leyenda-res" style="position:fixed;bottom:58px;left:16px;'
                'z-index:9999;background:rgba(46,42,43,0.92);color:#CDD3D9;'
                'padding:10px 12px;border-radius:10px;border:1px solid #494245;'
                'font:12px Inter,system-ui,sans-serif;display:flex;flex-direction:column;'
                'gap:5px;box-shadow:0 4px 12px rgba(0,0,0,0.35)">'
                + ''.join('<span style="display:flex;align-items:center;gap:7px">'
                          f'<span style="width:11px;height:11px;border-radius:50%;'
                          f'background:{c};flex:none"></span>{t}</span>' for c, t in filas)
                + '</div>')
            fmap.get_root().html.add_child(folium.Element(html))
            _capas = i18n.pick('Capas', 'Layers')
            fmap.get_root().header.add_child(folium.Element(
                '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
                'family=Inter:wght@400;500;600;700&family=Montserrat:wght@700'
                '&display=swap"><style>'
                '.leaflet-popup-content-wrapper{background:rgba(46,42,43,.96);'
                'color:#CDD3D9;border:1px solid #60575A;border-radius:12px;'
                'box-shadow:0 8px 22px rgba(0,0,0,.45)}'
                '.leaflet-popup-content{margin:10px 12px;font-family:Inter,system-ui,'
                'sans-serif}'
                '.leaflet-popup-tip{background:#2E2A2B;border:1px solid #60575A}'
                '.leaflet-container a.leaflet-popup-close-button{color:#AEB6BE;'
                'font:700 15px Inter,sans-serif;top:8px;right:8px}'
                '.leaflet-container a.leaflet-popup-close-button:hover{color:#ECEFF1}'
                '.leaflet-control-layers{background:rgba(46,42,43,.9)!important;'
                'border:1px solid #494245!important;border-radius:10px!important;'
                'color:#CDD3D9;font:12px Inter,system-ui,sans-serif;'
                'padding:10px 12px!important;box-shadow:0 4px 12px rgba(0,0,0,.35)!important}'
                '.leaflet-control-layers-list::before{content:"' + _capas + '";'
                'display:block;font-weight:600;color:#ECEFF1;margin-bottom:6px}'
                '.leaflet-control-layers-separator{border-top:1px solid #494245!important}'
                '.leaflet-control-layers input{accent-color:#7CC496}'
                '.leaflet-control-layers label{margin:3px 0}'
                '.leaflet-control-layers-list,.leaflet-control-layers-scrollbar{'
                'max-height:none!important;height:auto!important;overflow:visible!important;'
                'padding-right:0!important}'
                '</style>'))

        def _base_map(center, zoom, ortofoto=False, escala=True):
            # max_zoom alto para acercar bastante a los puntos (over-zoom de las
            # ortofotos); Google Satélite tiene resolución nativa mayor que ESRI.
            # tiles=None: las capas de fondo se añaden aquí abajo. Sin él, folium ponía
            # además su OpenStreetMap y salía dos veces en la lista de capas.
            # escala=False: sin la barra de escala (el mapa de 2 · Revisar, como la demo).
            fm = folium.Map(location=center, zoom_start=zoom, max_zoom=22,
                            control_scale=escala, tiles=None)
            esri = dict(tiles='https://services.arcgisonline.com/ArcGIS/rest/services/'
                              'World_Imagery/MapServer/tile/{z}/{y}/{x}',
                        attr='Esri World Imagery', name='ESRI World Imagery', max_zoom=22,
                        max_native_zoom=19)
            google = dict(tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
                          attr='Google', name=i18n.pick('Google Satélite (más zoom)',
                                                        'Google Satellite (more zoom)'),
                          max_zoom=22, max_native_zoom=21)
            osm = dict(tiles='OpenStreetMap', name='OpenStreetMap', max_zoom=19)
            if not ortofoto:
                # Se ve la última capa de fondo (en el mapa de resultados, OpenStreetMap);
                # las demás quedan en el control de capas.
                capas = [esri, google, osm]
                for _c in capas[:-1]:
                    _c['show'] = False
                for _c in capas:
                    folium.TileLayer(**_c).add_to(fm)
            else:
                # Para revisar un punto hay que ver el terreno: se ve la ortofoto PNOA y
                # las demás quedan en el control de capas, en el orden de la demo de
                # 2 · Revisar (27-11-2026): Ortofoto PNOA, ESRI, Google, OpenStreetMap y
                # LiDAR (en el control salen en el orden en que se añaden).
                folium.raster_layers.WmsTileLayer(
                    url='https://www.ign.es/wms-inspire/pnoa-ma',
                    layers='OI.OrthoimageCoverage', fmt='image/jpeg', transparent=False,
                    attr='IGN · PNOA', name=i18n.pick('Ortofoto PNOA', 'PNOA orthophoto'),
                    overlay=False, control=True, show=True, max_zoom=22).add_to(fm)
                # Para decidir si un borreguil es de arroyo o de ladera hay que ver
                # la forma del terreno: el sombreado del LiDAR del IGN (PNOA, ~2 m)
                # deja ver regueros y cauces pequeños que la ortofoto no enseña.
                lidar = dict(
                    tiles='https://wmts-mapa-lidar.idee.es/lidar?request=GetTile'
                          '&service=WMTS&version=1.0.0&layer=EL.GridCoverageDSM'
                          '&style=default&tilematrixset=GoogleMapsCompatible'
                          '&format=image/png&TileMatrix={z}&TileRow={y}&TileCol={x}',
                    attr='IGN · LiDAR PNOA', name=i18n.pick('Relieve LiDAR (IGN, sombreado)',
                                                            'LiDAR relief (IGN, shaded)'),
                    max_zoom=22, max_native_zoom=19)
                for _c in (esri, google, osm, lidar):
                    folium.TileLayer(show=False, **_c).add_to(fm)
            # Encima: el área de estudio y, con la ortofoto, las curvas de nivel (en este
            # orden, el de la demo).
            if sg is not None:
                from shapely.geometry import mapping
                folium.GeoJson(mapping(sg), name=i18n.pick('Área de estudio', 'Study area'),
                               style_function=lambda x: {'fillColor':'#1976D2','color':'#0D47A1',
                                                          'weight':2,'fillOpacity':0.08}).add_to(fm)
            if ortofoto:
                folium.raster_layers.WmsTileLayer(
                    url='https://servicios.idee.es/wms-inspire/mdt',
                    layers='EL.ContourLine', fmt='image/png', transparent=True,
                    attr='IGN · MDT', name=i18n.pick('Curvas de nivel (IGN, cada 50 m)',
                                                     'Contour lines (IGN, every 50 m)'),
                    overlay=True, show=False, max_zoom=22).add_to(fm)
            return fm

        _capas_oscuras = capas_oscuras      # (ahora también en el mapa de Analizar)

        def _add_pixel_layers(fm, pts, show=False, cap=600):
            """Capas (desactivables) con la rejilla de muestreo S2/S1 por punto:
            el píxel S2 de 10 m, la ventana de muestreo de 10 m (S2 bandas 10 m + S1)
            y la de 20 m (S2 bandas 20 m). Así se ve sobre el satélite qué cubre cada
            píxel. Se limita a `cap` puntos para no saturar el render."""
            sub = pts[:cap]
            try:
                fps = bp.pixel_footprints(sub)
            except Exception:
                return
            # Los nombres de la demo (27-11-2026): las dos ventanas, en una sola capa
            g_pix = folium.FeatureGroup(name=i18n.pick('Píxel Sentinel-2 (10 m)',
                                                       'Sentinel-2 pixel (10 m)'), show=show)
            g_win = folium.FeatureGroup(name=i18n.pick('Ventana de muestreo 10 m y 20 m',
                                                       'Sampling window 10 m and 20 m'),
                                        show=show)
            for fp in fps:
                folium.Polygon(fp['pixel10'], color='#FFD600', weight=1,
                               fill=True, fill_color='#FFD600', fill_opacity=0.12).add_to(g_pix)
                folium.Polygon(fp['win_s2_10'], color='#00E5FF', weight=1,
                               fill=False).add_to(g_win)
                folium.Polygon(fp['win_s2_20'], color='#FF6D00', weight=1,
                               fill=False, dash_array='4').add_to(g_win)
            g_pix.add_to(fm); g_win.add_to(fm)
            if len(pts) > cap:
                st.caption(f'ℹ Rejilla de píxeles mostrada para los primeros {cap} de '
                           f'{len(pts)} puntos (para no saturar el mapa).')

        # Leer la selección de la tabla desde el estado del widget (evita el desfase
        # de un rerun en el mapa principal de la pestaña Mapa de los puntos). Streamlit devuelve la
        # selección ORDENADA POR ÍNDICE (no por orden de clic); para enfocar el ÚLTIMO
        # punto seleccionado detectamos cuál se acaba de añadir comparando con la
        # selección previa guardada en sesión.
        def _table_sel_rows():
            tw = st.session_state.get('results_table')
            if tw is None:
                return []
            try:
                selo = tw.get('selection') if isinstance(tw, dict) else getattr(tw, 'selection', None)
                return list((selo.get('rows') if isinstance(selo, dict)
                             else getattr(selo, 'rows', None)) or [])
            except Exception:
                return []

        def _focus_idx(sel_rows):
            cur = set(sel_rows)
            prev = set(st.session_state.get('prev_sel_set', set()))
            st.session_state.prev_sel_set = cur
            if not cur:
                return None
            added = cur - prev
            if len(added) == 1:
                return next(iter(added))          # el punto recién clicado
            if added:
                return max(added)                 # varios añadidos (rango): el de mayor índice
            # No se añadió ninguno (se deseleccionó o sin cambios): mantener el foco
            # previo si sigue seleccionado.
            prev_focus = st.session_state.get('sel_idx')
            if prev_focus is not None and prev_focus in cur:
                return prev_focus
            return max(cur)

        _all_sel = _table_sel_rows()
        _focus = _focus_idx(_all_sel)
        if _focus is not None and _focus < len(rows):
            st.session_state.sel_idx = _focus
            st.session_state.sel_point = dict(rows[_focus])
        elif not _all_sel:
            st.session_state.sel_idx = None
            st.session_state.sel_point = None
        sel = st.session_state.get('sel_point')

        # --- Puntos añadidos pinchando en el mapa ---------------------------------
        # Quién revisa, su perfil y su certidumbre (24-11, idea de la autora): se
        # apuntan en cada punto que se verifica o se añade en el mapa y van en las
        # descargas con las columnas de la plantilla (observer, expertise, certainty).
        _PERFIL_TXT = {'student': i18n.pick('Estudiante', 'Student'),
                       'manager': i18n.pick('Gestor/a', 'Manager'),
                       'researcher': i18n.pick('Investigador/a', 'Researcher')}
        _CERT_TXT = {'high': i18n.pick('Alta', 'High'),
                     'medium': i18n.pick('Media', 'Medium'),
                     'low': i18n.pick('Baja', 'Low')}
        _CERT_AYUDA = i18n.pick(
            'Lo segura que es la respuesta. **Alta**: no hay duda. **Media**: hay alguna '
            'duda. **Baja**: es lo más probable, pero hay bastante duda. Si no sabes decir '
            'si es borreguil, no lo verifiques. Si no eliges ninguna, se queda sin indicar.',
            'How sure the answer is. **High**: no doubt. **Medium**: some doubt. **Low**: '
            'it is the most likely answer, but there is quite a lot of doubt. If you cannot '
            'tell whether it is a borreguil, do not verify it. If you choose none, it is '
            'left unstated.')

        def _quien():
            """(nombre, perfil) de quien revisa, tal como está en la sesión."""
            return (st.session_state.get('revisor_nombre') or '',
                    st.session_state.get('revisor_perfil') or '')

        _RESP_MAPA = {'si': i18n.pick('hay borreguil', 'there is a borreguil'),
                      'no': i18n.pick('no hay borreguil', 'there is no borreguil'),
                      'duda': i18n.pick('no está seguro', 'not sure')}
        _OPINION = {'si': i18n.pick('borreguil', 'borreguil'),
                    'no': i18n.pick('no borreguil', 'not a borreguil')}

        def _anadir_del_mapa(resp, lat, lng):
            _nom, _per = _quien()
            nuevo = bp.punto_del_mapa(rows, lng, lat, resp, observador=_nom, perfil=_per,
                                      certidumbre=st.session_state.get('mapa_certidumbre')
                                      or '',
                                      nota=st.session_state.get('mapa_nota', ''))
            _be = st.session_state.get('active_backend') or backend
            _feats = st.session_state.get('sel_model_feats', []) or []
            try:
                _anios = bp.parse_years(years_str)
            except Exception:
                _anios = tuple(range(2017, 2026))
            with st.status(i18n.pick(f'Leyendo las variables de {nuevo["ID"]}…',
                                     f'Reading the variables of {nuevo["ID"]}…'),
                           expanded=True) as _stt:
                _t0 = time.time()
                fallos = leer_variables_puntos(
                    [nuevo], work, _be, _anios, img_src=img_src, skip_imgs=skip_imgs,
                    skip_s2=skip_s2,
                    con_cir=any(f.startswith('cir_') for f in _feats) and not fast_mode,
                    gee_project=gee_project, consola=_stt)
                _stt.update(label=i18n.pick(
                    f'Variables de {nuevo["ID"]} leídas ({time.time() - _t0:.0f} s)',
                    f'Variables of {nuevo["ID"]} read ({time.time() - _t0:.0f} s)'),
                    state='error' if fallos else 'complete')
            # Lo que opinaba el modelo ANTES de saber la respuesta (el modelo propio,
            # si lo hay, aprenderá después de este mismo punto).
            try:
                _bun = st.session_state.get('model_bundle') or bp.load_model_bundle(MODEL_PATH)
                p = float(bp.predecir_con_bundle([nuevo], _bun)[0])
            except Exception:
                p = float('nan')
            nuevo['rf_proba_al_anadir'] = p
            rows2 = [dict(r) for r in rows] + [nuevo]
            new_base = set(base_truth_ids) | ({nuevo['ID']} if resp == 'si' else set())
            with st.spinner(i18n.pick('Recalculando…', 'Recalculating…')):
                _recalcular_y_guardar(rows2, new_base,
                                      f'añadido en el mapa ({nuevo["ID"]})', 1)
            st.session_state['_clic_usado'] = (lat, lng)
            st.session_state['_ultimo_anadido'] = (lat, lng)
            op = bp.opinion_modelo(p, threshold)
            if op is None:
                texto = i18n.pick(f'**{nuevo["ID"]}** añadido ({_RESP_MAPA[resp]}).',
                                  f'**{nuevo["ID"]}** added ({_RESP_MAPA[resp]}).')
            elif resp == 'duda':
                texto = i18n.pick(
                    f'**{nuevo["ID"]}** añadido como dudoso: no entrena ni cuenta en el '
                    f'acierto. El modelo le daba **{_num(p)}** ({_OPINION[op]}).',
                    f'**{nuevo["ID"]}** added as doubtful: it does not train and does not '
                    f'count in the accuracy. The model gave it **{_num(p)}** ({_OPINION[op]}).')
            else:
                _corte = bp.corte_borreguil(threshold)
                texto = i18n.pick(
                    f'**{nuevo["ID"]}** añadido: {_RESP_MAPA[resp]}. El modelo le daba '
                    f'**{_num(p)}** (da borreguil desde {_num(_corte)}), así que decía '
                    f'«{_OPINION[op]}»: '
                    + ('**coincide** ✓' if op == resp else '**no coincide** ✗') + '.',
                    f'**{nuevo["ID"]}** added: {_RESP_MAPA[resp]}. The model gave it '
                    f'**{_num(p)}** (borreguil from {_num(_corte)}), so it said '
                    f'"{_OPINION[op]}": '
                    + ('**it agrees** ✓' if op == resp else '**it disagrees** ✗') + '.')
            if fallos:
                texto += ' ' + i18n.pick(
                    'No se pudo leer: ' + '; '.join(fallos) + '. Para eso el modelo usa su '
                    'valor típico, así que su opinión vale menos.',
                    'Could not read: ' + '; '.join(fallos) + '. For that the model uses '
                    'its typical value, so its opinion is less reliable.')
            st.session_state['_msg_mapa'] = ('aviso' if fallos else 'ok', texto)
            st.rerun()

        # La «?» de la certidumbre al añadir puntos, en líneas como en la demo
        _CERT_AYUDA_MAPA = i18n.pick(
            'Lo segura que es la respuesta.\n\n**Alta**: no hay duda.  \n**Media**: hay '
            'alguna duda.  \n**Baja**: es lo más probable, pero hay bastante duda.\n\nSi no '
            'sabes decir si es borreguil, no lo verifiques. Si no eliges ninguna, se queda '
            'sin indicar.',
            'How sure the answer is.\n\n**High**: no doubt.  \n**Medium**: some doubt.  \n'
            '**Low**: it is the most likely answer, but there is quite a lot of doubt.\n\n'
            'If you cannot tell whether it is a borreguil, do not verify it. If you choose '
            'none, it is left unstated.')

        def _deshacer_ultimo():
            """«Quitar el último añadido»: quita el último punto añadido en el mapa."""
            _anadidos = [r.get('ID') for r in rows if r.get('source') == 'mapa']
            if not _anadidos:
                return
            _ult_id = _anadidos[-1]
            rows2 = [dict(r) for r in rows if r.get('ID') != _ult_id]
            with st.spinner(i18n.pick('Recalculando…', 'Recalculating…')):
                _recalcular_y_guardar(rows2, set(base_truth_ids) - {_ult_id},
                                      f'quitado del mapa ({_ult_id})', 0)
            st.session_state.sel_point = None
            st.session_state.sel_idx = None
            st.session_state.prev_sel_set = set()
            st.session_state['_vaciar_seleccion'] = True
            st.session_state['_msg_mapa'] = ('ok', i18n.pick(
                f'**{_ult_id}** quitado.', f'**{_ult_id}** removed.'))
            st.rerun()

        def _panel_anadir(clic):
            """Lo de debajo de la casilla «Añadir puntos pinchando en el mapa», como en la
            demo: el texto, los tres botones de respuesta (se pueden pulsar en cuanto se
            pincha en el mapa) y la fila de la certidumbre, la nota, «Cancelar» y «Quitar
            el último añadido»."""
            st.markdown(i18n.pick(
                'Pincha donde sepas si hay borreguil. La app lee solo ese punto y te dice '
                'qué opinaba el modelo antes de saber tu respuesta.',
                'Click where you know whether there is a borreguil. The app reads only that '
                'point and tells you what the model thought before knowing your answer.'))
            valido = None
            if clic:
                lat, lng = clic
                _i, _d = bp.punto_mas_cercano(rows, lng, lat)
                if _i is not None and _d < bp.CERCA_M:
                    st.info(i18n.pick(
                        f'Has pinchado en **{rows[_i].get("ID")}**, que ya está en el '
                        'análisis. Para añadir un punto, pincha fuera de los marcadores.',
                        f'You clicked on **{rows[_i].get("ID")}**, which is already in '
                        'the analysis. To add a point, click away from the markers.'))
                else:
                    valido = clic
                    st.caption(i18n.pick(
                        f'Has pinchado en **{_num(lat, 5)}; {_num(lng, 5)}** (círculo rojo). '
                        '¿Hay borreguil aquí?',
                        f'You clicked on **{_num(lat, 5)}, {_num(lng, 5)}** (red circle). '
                        'Is there a borreguil here?'))
            resp = None
            with st.container(key='res_respuestas', horizontal=True, gap='small'):
                if st.button(i18n.pick('Aquí hay borreguil', 'There is a borreguil here'),
                             key='mapa_si', disabled=not valido, width='stretch'):
                    resp = 'si'
                if st.button(i18n.pick('Aquí no hay', 'There is none here'), key='mapa_no',
                             disabled=not valido, width='stretch'):
                    resp = 'no'
                if st.button(i18n.pick('No estoy seguro/a', 'Not sure'), key='mapa_duda',
                             disabled=not valido, width='stretch'):
                    resp = 'duda'
            _anadidos = [r.get('ID') for r in rows if r.get('source') == 'mapa']
            _f1, _f2, _f3, _f4 = st.columns([1.3, 2, 0.9, 1.5], gap='small',
                                            vertical_alignment='bottom')
            _f1.selectbox(i18n.pick('Certidumbre', 'Certainty'), list(bp.CERTIDUMBRES),
                          index=None, format_func=_CERT_TXT.get, key='mapa_certidumbre',
                          placeholder=i18n.pick('Elige…', 'Choose…'), help=_CERT_AYUDA_MAPA)
            _f2.text_input(i18n.pick('Nota (opcional)', 'Note (optional)'), key='mapa_nota',
                           placeholder=i18n.pick('Lo que veas', 'What you see'))
            if _f3.button(i18n.pick('Cancelar', 'Cancel'), key='mapa_cancelar',
                          disabled=not valido, width='stretch'):
                st.session_state['_clic_usado'] = clic
                st.rerun()
            if _f4.button(i18n.pick('Quitar el último añadido', 'Remove the last one added'),
                          key='mapa_deshacer', disabled=not _anadidos, width='stretch',
                          help=(i18n.pick(f'Quita {_anadidos[-1]}', f'Removes {_anadidos[-1]}')
                                if _anadidos else None)):
                _deshacer_ultimo()
            if resp and valido:
                _anadir_del_mapa(resp, *valido)

        def _revisar_puntos(indices):
            """«Revisarlo» (en el mapa) y «Revisar» (en la Tabla): esos puntos quedan
            elegidos en la tabla de «2 · Revisar» y se pasa a esa pantalla."""
            st.session_state['results_table'] = {
                'selection': {'rows': list(indices), 'columns': [], 'cells': []}}
            st.session_state['pantalla'] = 'revisar'

        # --- Quién revisa: una franja, como en la demo
        with st.container(key='res_quien'):
            _q1, _q2, _q3 = st.columns([1.1, 1, 2], gap='small', vertical_alignment='bottom')
            _q1.text_input(i18n.pick('Quién revisa (opcional)', 'Who is reviewing (optional)'),
                           key='revisor_nombre', placeholder=i18n.pick('Tu nombre', 'Your name'))
            _q2.selectbox(i18n.pick('Perfil (opcional)', 'Profile (optional)'),
                          list(bp.PERFILES), index=None, format_func=_PERFIL_TXT.get,
                          placeholder=i18n.pick('Elige…', 'Choose…'), key='revisor_perfil')
            _q3.caption(i18n.pick(
                'Se apunta en cada punto que revises o añadas, con su certidumbre, y sale en '
                'las descargas.',
                'It is recorded in each point you review or add, with its certainty, and it '
                'goes into the downloads.'))

        # --- Las vistas (píldoras) a la izquierda y, a la derecha, siempre a la vista,
        # «Siguiente paso» y «Después» (la demo). Nombres y orden de la autora; desde el
        # 27-11-2026, con «Tabla» (solo para mirar: la que se edita está en «2 · Revisar»).
        _col_vistas, _col_lado = st.columns([0.775, 0.225], gap='small')
        with _col_vistas:
            tab1, tab_tabla, tab3, tab5, tab4 = st.tabs([
                i18n.pick('Mapa', 'Map'),
                i18n.pick('Tabla', 'Table'),
                i18n.pick('Tipos de borreguil', 'Borreguil types'),
                i18n.pick('En qué se fija el modelo', 'What the model looks at'),
                i18n.pick('Descargas', 'Downloads')], key='vistas_resultados')
        tab2 = pant_revisar            # «Revisar puntos» es ahora la pantalla «2 · Revisar»
        with _col_lado, st.container(key='res_aside', gap='small'):
            _corte_sig = bp.corte_borreguil(threshold)
            _n_dudosos = sum(
                1 for r in rows if bp._txt(r.get('Borreguil')).lower() not in ('si', 'no')
                and isinstance(r.get('rf_proba'), (int, float))
                and abs(r['rf_proba'] - _corte_sig) < 0.1)
            _empieza = ('' if not _n_dudosos else
                        i18n.pick(' Empieza por el dudoso.', ' Start with the doubtful one.')
                        if _n_dudosos == 1 else
                        i18n.pick(f' Empieza por los {_n_dudosos} dudosos.',
                                  f' Start with the {_n_dudosos} doubtful ones.'))
            with st.container(key='res_siguiente', gap='small'):
                st.html('<div class="res-sig"><div class="e">'
                        + i18n.pick('Siguiente paso', 'Next step') + '</div><div class="t">'
                        + i18n.pick('Revisa los puntos', 'Review the points')
                        + '</div><div class="d">'
                        + i18n.pick('Mira cada punto sobre la ortofoto y di si es borreguil.',
                                    'Look at each point on the orthophoto and say whether it '
                                    'is a borreguil.') + _empieza + '</div></div>')
                st.button(i18n.pick('Revisar puntos', 'Review points'), type='primary',
                          key='ir_revisar', on_click=ir_a, args=('revisar',), width='stretch')
            with st.container(key='res_despues', gap='small'):
                st.html('<div class="res-des"><div class="t">'
                        + i18n.pick('Después', 'Then') + '</div><div class="d">'
                        + i18n.pick('Con los puntos revisados, haz el mapa de borreguil de '
                                    'toda tu zona.',
                                    'With the reviewed points, make the borreguil map of your '
                                    'whole area.') + '</div></div>')
                st.button(i18n.pick('Ir a «Hacer el mapa»', 'Go to "Make the map"'),
                          key='ir_mapa', type='tertiary', on_click=ir_a, args=('mapa',))

        with tab1:
            lat_med = sum(r['lat'] for r in rows)/len(rows)
            lon_med = sum(r['lon'] for r in rows)/len(rows)
            # La casilla de añadir puntos va debajo del mapa (la demo), pero el mapa ya
            # tiene que saber si está marcada: su valor ya está en la sesión.
            _anadir = bool(st.session_state.get('modo_anadir'))

            # El clic que trae el mapa (el componente guarda su último valor con su
            # clave): se lee antes de dibujarlo para marcar dónde se ha pinchado.
            _clic = None
            _ms = st.session_state.get('mainmap')
            _ms = _ms if isinstance(_ms, dict) else {}
            if _anadir:
                _lc = _ms.get('last_clicked')
                if _lc and _lc.get('lat') is not None:
                    _c = (round(float(_lc['lat']), 7), round(float(_lc['lng']), 7))
                    if _c != st.session_state.get('_clic_usado'):
                        _clic = _c

            def _punto_pinchado(obj):
                """La posición del punto del análisis en el que se ha pinchado (None si
                no hay ninguno): last_object_clicked trae las coordenadas exactas del
                marcador; se toleran ~50 m por redondeos."""
                if not obj or obj.get('lat') is None:
                    return None
                ni, nd = None, 1e18
                for i, r in enumerate(rows):
                    d = (r['lat'] - obj['lat'])**2 + (r['lon'] - obj['lng'])**2
                    if d < nd:
                        nd, ni = d, i
                return ni if ni is not None and nd < (0.00045)**2 else None

            # Con el modo de añadir, el mapa no se recoloca en cada clic: solo se
            # centra en el último punto añadido cuando se vuelve a dibujar.
            # De fondo, la ortofoto PNOA (la demo); las demás, en «Capas».
            _ult = st.session_state.get('_ultimo_anadido') if _anadir else None
            if _ult:
                m = _base_map(list(_ult), 16, ortofoto=True)
            else:
                m = _base_map([lat_med, lon_med], 12, ortofoto=True)
            _add_points(m, rows)
            _add_pixel_layers(m, rows, show=False)
            _add_legend(m, rows)
            folium.LayerControl(collapsed=False).add_to(m)
            # Lo que cambia sin volver a dibujar el mapa: el aro blanco del punto elegido
            # (el último en el que se ha pinchado o, si no, el elegido en «2 · Revisar»)
            # y, al añadir puntos, dónde se ha pinchado.
            _g_clic = folium.FeatureGroup(name=i18n.pick('Donde has pinchado',
                                                         'Where you clicked'))
            _i_pinchado = None if _anadir else _punto_pinchado(_ms.get('last_object_clicked'))
            _elegido = rows[_i_pinchado] if _i_pinchado is not None else sel
            if _elegido and _elegido.get('lat') is not None:
                folium.CircleMarker([_elegido['lat'], _elegido['lon']], radius=12,
                                    color='#FFFFFF', weight=5, fill=False,
                                    interactive=False).add_to(_g_clic)
            if _clic:
                folium.CircleMarker(list(_clic), radius=11, color='#FF1744', weight=3,
                                    fill=False).add_to(_g_clic)
                folium.CircleMarker(list(_clic), radius=2, color='#FF1744',
                                    fill=True, fill_opacity=1).add_to(_g_clic)
            if _anadir:
                # Con el modo de añadir, el cursor es una cruz. Además así cambia el guion
                # del mapa y st_folium lo vuelve a crear: si no, seguía con lo que devolvía
                # al dibujarse la primera vez (sin los clics) y pinchar no hacía nada.
                from branca.element import MacroElement as _Macro
                from jinja2 import Template as _Plantilla
                _cruz = _Macro()
                _cruz._template = _Plantilla(
                    '{% macro script(this, kwargs) %}' + m.get_name()
                    + ".getContainer().style.cursor = 'crosshair';{% endmacro %}")
                _cruz.add_to(m)
                map_state = st_folium(m, width=None, height=540, feature_group_to_add=_g_clic,
                                      returned_objects=['last_object_clicked', 'last_clicked'],
                                      key='mainmap')
            elif sel and _i_pinchado is None:
                map_state = st_folium(m, width=None, height=540, feature_group_to_add=_g_clic,
                                      returned_objects=['last_object_clicked'],
                                      center=[sel['lat'], sel['lon']], zoom=18, key='mainmap')
            else:
                map_state = st_folium(m, width=None, height=540, feature_group_to_add=_g_clic,
                                      returned_objects=['last_object_clicked'], key='mainmap')

            if _anadir:
                # Manda lo que devuelve el mapa al dibujarlo: es su último clic (al
                # empezar la pasada, la sesión podía tener aún el anterior).
                _lc = (map_state or {}).get('last_clicked') if isinstance(map_state, dict) else None
                if _lc and _lc.get('lat') is not None:
                    _c = (round(float(_lc['lat']), 7), round(float(_lc['lng']), 7))
                    _clic = None if _c == st.session_state.get('_clic_usado') else _c

            # El punto en el que se ha pinchado: revisarlo o quitarlo del análisis (no con
            # el modo de añadir: ahí pinchar es para añadir, y lo añadido se deshace con
            # su botón). En la demo son enlaces del globo; el globo del mapa no puede
            # pulsar botones de la app, así que van justo debajo del mapa.
            clicked = (map_state or {}).get('last_object_clicked') \
                if isinstance(map_state, dict) and not _anadir else None
            ni = _punto_pinchado(clicked)
            if ni is not None:
                rr = rows[ni]
                _nom_rr = bp.nombres_de_punto([rr.get('ID')], i18n.pick('Punto', 'Point'))[0]
                with st.container(key='res_sel_punto', horizontal=True,
                                  vertical_alignment='center', gap='medium'):
                    st.markdown(f"**{_nom_rr}** · "
                                f"{i18n.tv_decision(rr.get('decision') or '—')}",
                                width='content')
                    st.button(i18n.pick('Revisarlo', 'Review it'), key='res_revisarlo',
                              type='tertiary', on_click=_revisar_puntos, args=([ni],))
                    if st.button(i18n.pick('Quitar este punto del análisis',
                                           'Remove this point from the analysis'),
                                 key='remove_map_pt', type='tertiary'):
                        keep = [r for j, r in enumerate(rows) if j != ni]
                        if not keep:
                            st.warning(i18n.pick('No puedes quitar todos los puntos.',
                                                 'You cannot remove all the points.'))
                        else:
                            bp.save_csv(keep, work / 'classification.csv')
                            bp.save_points_geojson(keep, work / 'puntos.geojson')
                            bp.save_xlsx(keep, work / 'Clasificacion_puntos.xlsx',
                                         threshold=threshold)
                            bp.save_map(keep, work / 'mapa.html', study_geom=sg)
                            st.session_state.rows = keep
                            st.session_state.sel_point = None
                            st.session_state.sel_idx = None
                            st.session_state.prev_sel_set = set()
                            st.session_state['_vaciar_seleccion'] = True
                            # Para la pasada siguiente (antes de st.rerun no se veía)
                            st.session_state['_msg_mapa'] = ('ok', _msg_puntos(
                                [rr.get('ID', '?')], 'quitado del análisis', '',
                                'removed from the analysis', ''))
                            st.rerun()
            st.caption(i18n.pick(
                'Pincha en un punto para ver su globo. Verde: es borreguil (lo dice el modelo '
                'o tu revisión); rojo: no lo es. El aro blanco marca el punto elegido.',
                'Click a point to see its pop-up. Green: it is a borreguil (the model or '
                'your review says so); red: it is not. The white ring marks the chosen '
                'point.'))

            # --- Añadir puntos pinchando en el mapa (p. ej. en un taller con quien
            # conoce el sitio). El punto entra como un dato de campo más.
            with st.container(key='res_anadir', gap='small'):
                st.checkbox(
                    i18n.pick('Añadir puntos pinchando en el mapa',
                              'Add points by clicking on the map'),
                    key='modo_anadir',
                    help=i18n.pick(
                        'Para quien conoce el sitio: pincha donde sepas si hay borreguil y '
                        'contesta debajo del mapa. El punto queda como un dato de campo más '
                        '(method = local knowledge) y se guarda en las descargas.',
                        'For people who know the site: click where you know whether there '
                        'is a borreguil and answer below the map. The point becomes one more '
                        'field point (method = local knowledge) and is saved in the '
                        'downloads.'))
                _msg_mapa = st.session_state.pop('_msg_mapa', None)
                if _msg_mapa:
                    (st.success if _msg_mapa[0] == 'ok' else st.warning)(_msg_mapa[1])
                if _anadir:
                    _panel_anadir(_clic)

        with tab2:
            # ── 2 · Revisar, como en la demo (Revisar.dc.html, 27-11-2026) ───────────────
            # A la izquierda, la lista de puntos con sus filtros (Sin revisar · Dudosos
            # primero · Todos). En el centro, el punto abierto: su nombre y su frase, la
            # ortofoto con el píxel, por qué miramos dentro del recuadro amarillo, lo que más
            # pesa en el modelo y la gráfica frente a los borreguiles verdaderos; a su lado,
            # la revisión en una tarjeta; debajo, la tabla con todos los puntos. Siempre hay
            # un punto abierto: el que se elige en la lista o en la tabla (en la tabla se
            # pueden marcar varios y revisarlos juntos) o, al entrar, el primero de la lista.
            # Los estilos van aquí, con las claves rev_… (y las clases rev-…).
            _CSS_REVISAR = '''
.st-key-rev_lado{background:#342F31;border:1px solid #534B4E;border-radius:16px;padding:20px;
  box-shadow:6px 0 18px rgba(0,0,0,.35);gap:12px}
.rev-cuenta{display:flex;flex-direction:column;gap:6px}
.rev-cuenta b{font-size:15px;font-weight:700;color:#ECEFF1}
.rev-barra{height:8px;border-radius:999px;background:#403A3C;overflow:hidden}
.rev-barra span{display:block;height:8px;background:#7CC496}
.st-key-rev_filtro [data-testid="stButtonGroup"]>div{gap:6px;flex-wrap:wrap}
.st-key-rev_filtro button{border:1px solid #62585B;background:#484143;border-radius:999px;
  padding:6px 12px;min-height:0}
.st-key-rev_filtro button p{font-size:13px;font-weight:500;color:#CDD3D9}
.st-key-rev_filtro button[kind$="Active"]{border-color:#ECEFF1;background:#ECEFF1}
.st-key-rev_filtro button[kind$="Active"] p{color:#221F20;font-weight:700}
.st-key-rev_lista_caja{height:auto!important;max-height:620px}
.st-key-rev_lista,.st-key-rev_lista [data-testid="stRadio"],
.st-key-rev_lista [role="radiogroup"]{width:100%}
.st-key-rev_lista [role="radiogroup"]{gap:6px;flex-direction:column;flex-wrap:nowrap}
.st-key-rev_lista label[data-baseweb="radio"]{width:100%;margin:0;padding:10px 12px;
  border-radius:10px;background:#3D383A;border:1px solid #524A4D;box-sizing:border-box}
.st-key-rev_lista label[data-baseweb="radio"]:hover{border-color:#5E6871}
.st-key-rev_lista label[data-baseweb="radio"]:has(input:checked){background:#1B2B21;
  border:1.5px solid #7CC496}
.st-key-rev_lista label[data-baseweb="radio"]>div:first-of-type{display:none}
.st-key-rev_lista label[data-baseweb="radio"]>div:last-child,
.st-key-rev_tarjeta [class*="st-key-rev_es_"] label[data-baseweb="radio"]>div:last-child,
.st-key-rev_tarjeta .st-key-rev_certidumbre label[data-baseweb="radio"]>div:last-child{
  flex:1;min-width:0;padding:0}
.st-key-rev_lista label [data-testid="stMarkdownContainer"] p{display:flex;align-items:center;
  gap:10px;margin:0;font-size:14px;color:#ECEFF1}
.st-key-rev_lista label p strong{flex:1;min-width:0;font-weight:600;overflow:hidden;
  text-overflow:ellipsis;white-space:nowrap}
.st-key-rev_lista label p em{font-style:normal;font-weight:700;font-size:15px;color:#ECEFF1}
.st-key-rev_lado_nota p{font-size:12px;line-height:1.45;color:#98A2AB}
.rev-titulo{display:flex;flex-wrap:wrap;align-items:baseline;gap:10px}
.rev-titulo h1{margin:0;padding:0;font-family:Montserrat,Inter,sans-serif;font-size:26px;
  font-weight:700;line-height:1.2;color:#F5F7F8;overflow-wrap:anywhere}
.rev-titulo span{font-size:14px;color:#AEB6BE}
.st-key-rev_mapa iframe{border-radius:18px;border:1px solid #403A3C!important;
  background:#2E2A2B;display:block}
details.rev-amarillo,details.rev-perfil{border-radius:12px;overflow:hidden}
details.rev-amarillo{border:1px solid #5C5320;background:#221F12}
details.rev-perfil{border:1px solid #403A3C;background:#2E2A2B}
details.rev-amarillo>summary,details.rev-perfil>summary{list-style:none;display:flex;
  align-items:center;gap:8px;padding:11px 14px;font:600 14px Inter,system-ui,sans-serif;
  min-height:44px;box-sizing:border-box;cursor:pointer}
details.rev-amarillo>summary{color:#F5E7B0}
details.rev-perfil>summary{color:#ECEFF1}
details.rev-amarillo>summary::-webkit-details-marker,
details.rev-perfil>summary::-webkit-details-marker,
details.rev-leer>summary::-webkit-details-marker{display:none}
details.rev-amarillo>summary::before,details.rev-perfil>summary::before{content:"\\25B8";
  width:14px;flex:none}
details.rev-amarillo[open]>summary::before,details.rev-perfil[open]>summary::before{
  content:"\\25BE"}
details.rev-amarillo>summary::before{color:#F5C842}
details.rev-perfil>summary::before{color:#7CC496}
.rev-amarillo-cuerpo{padding:0 16px 14px 36px;display:flex;flex-direction:column;gap:8px;
  font-size:13px;line-height:1.55;color:#E9E2C4}
.rev-perfil-cuerpo{padding:4px 16px 16px;display:flex;flex-direction:column;gap:12px;
  font-size:13px;color:#CDD3D9}
details.rev-leer>summary{list-style:none;color:#A3D9B5;font:600 13px Inter,system-ui,sans-serif;
  padding:6px 0;min-height:32px;box-sizing:border-box;cursor:pointer}
details.rev-leer>summary::before{content:"\\25B8  "}
details.rev-leer[open]>summary::before{content:"\\25BE  "}
details.rev-leer>summary:hover{color:#CFEBD9}
.rev-leer-cuerpo{font-size:12.5px;line-height:1.55;color:#CDD3D9;background:#221F20;
  border:1px solid #3E6B4F;border-radius:10px;padding:10px 12px;display:flex;
  flex-direction:column;gap:6px}
.rev-clave{display:flex;flex-direction:column;gap:8px}
.rev-clave>.t{font-size:13px;color:#AEB6BE}
.rev-tarjetas{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}
.rev-tarjetas>div{background:#2E2A2B;border:1px solid #403A3C;border-radius:12px;padding:12px;
  display:flex;flex-direction:column;gap:3px}
.rev-tarjetas .e{font-size:12px;color:#AEB6BE}
.rev-tarjetas .v{font-weight:700;font-size:18px;color:#F5F7F8}
.rev-tarjetas .r{font-size:12px;color:#98A2AB}
@media (max-width:700px){.rev-tarjetas{grid-template-columns:1fr}}
.st-key-rev_linea p{font-size:13px;color:#CDD3D9;margin:0}
.st-key-rev_tarjeta{background:#2E2A2B;border:1px solid #403A3C;border-radius:18px;
  padding:20px;gap:16px}
.st-key-rev_tarjeta [data-testid="stCaptionContainer"] p{font-size:12px;color:#98A2AB}
.rev-preg{font-family:Montserrat,Inter,sans-serif;font-weight:700;font-size:18px;
  color:#F5F7F8}
.rev-nota{font-size:13px;line-height:1.45;color:#E9E2C4;background:#221F12;
  border:1px solid #5C5320;border-radius:10px;padding:8px 10px}
.st-key-rev_pregunta{gap:10px}
.st-key-rev_tarjeta [class*="st-key-rev_es_"] [role="radiogroup"]{display:grid;
  grid-template-columns:minmax(0,1fr) minmax(0,1fr) minmax(0,1.35fr);gap:8px}
.st-key-rev_tarjeta [class*="st-key-rev_es_"] label[data-baseweb="radio"]{min-height:48px;
  padding:12px 6px;margin:0}
.st-key-rev_tarjeta [class*="st-key-rev_es_"] label[data-baseweb="radio"] p{text-align:center;
  line-height:1.2}
.st-key-rev_tarjeta [class*="st-key-rev_es_"] label[data-baseweb="radio"]:not(:has(input:checked)) p{
  font-size:14px;font-weight:600;color:#ECEFF1}
.st-key-rev_tarjeta [class*="st-key-rev_es_"] label[data-baseweb="radio"]:has(input:checked) p{
  color:#FFFFFF}
.st-key-rev_certidumbre [data-testid="stWidgetLabel"] p{font-size:14px;font-weight:600;
  color:#ECEFF1}
.st-key-rev_certidumbre [data-testid="stWidgetLabel"] em{font-style:normal;font-weight:400;
  color:#AEB6BE}
.st-key-rev_tarjeta .st-key-rev_certidumbre [role="radiogroup"]{display:grid;
  grid-template-columns:repeat(3,minmax(0,1fr));gap:6px}
.st-key-rev_tarjeta .st-key-rev_certidumbre label[data-baseweb="radio"]{min-height:44px;
  padding:9px 6px;margin:0;border:1px solid #494245;border-radius:10px;background:#393335}
.st-key-rev_tarjeta .st-key-rev_certidumbre label[data-baseweb="radio"] p{font-size:13px;
  font-weight:500;color:#CDD3D9;text-align:center}
.st-key-rev_tarjeta .st-key-rev_certidumbre label[data-baseweb="radio"]:has(input:checked){
  border:1.5px solid #7CC496;background:#1E3327}
.st-key-rev_tarjeta .st-key-rev_certidumbre label[data-baseweb="radio"]:has(input:checked) p{
  font-weight:600;color:#ECEFF1}
.st-key-rev_tipo{border-top:1px solid #403A3C;padding-top:14px;gap:10px}
.rev-tipo-tit{font-weight:600;font-size:14px;color:#ECEFF1}
.rev-tipo-tit span{font-weight:400;color:#AEB6BE}
.st-key-rev_tipo [data-testid="stWidgetLabel"] p{font-size:13px;color:#CDD3D9}
.st-key-rev_tipo [data-baseweb="select"]>div{background:#221F20;border:1px solid #60575A;
  border-radius:10px;min-height:44px}
.rev-dudoso{font-size:13px;color:#AEB6BE;border-top:1px solid #403A3C;padding-top:14px}
.st-key-rev_fila_guardar,.st-key-rev_fila_deshacer,.st-key-rev_fila_quitar{gap:6px;
  flex-wrap:nowrap}
.st-key-rev_fila_guardar>*,.st-key-rev_fila_deshacer>*,.st-key-rev_fila_quitar>*{min-width:0}
.st-key-rev_guardar button{min-height:48px;border-radius:12px;padding:13px}
.st-key-rev_guardar button p{font-size:15px;font-weight:700}
.st-key-rev_deshacer button,.st-key-remove_pts button{min-height:44px;border-radius:12px;
  padding:11px}
.st-key-rev_deshacer button{border:1px solid #494245;background:#393335}
.st-key-rev_deshacer button p{font-size:14px;font-weight:500;color:#CDD3D9}
.st-key-remove_pts button{border:1px solid #5A2E2E;background:#241616}
.st-key-remove_pts button:hover{border-color:#8C4A4A;background:#2E1B1B}
.st-key-remove_pts button p{font-size:14px;font-weight:500;color:#F0B4B4;line-height:1.2}
.rev-tabla-tit{display:flex;flex-wrap:wrap;align-items:baseline;gap:6px 12px}
.rev-tabla-tit h2{margin:0;padding:0;font-family:Montserrat,Inter,sans-serif;
  font-size:20px;font-weight:700;color:#F5F7F8}
.rev-tabla-tit span{font-size:13px;color:#AEB6BE}
.st-key-rev_tabla [data-testid="stCaptionContainer"] p{font-size:12.5px;line-height:1.5;
  color:#AEB6BE}
'''
            # La pantalla, más ancha que las demás: en la demo va de lado a lado.
            st.html('<style>' + _CSS_REVISAR
                    + ('.st-key-pantalla{max-width:1440px}' if PANTALLA == 'revisar' else '')
                    + '</style>')
            _ids = [r.get('ID') for r in rows]
            _nombres_sel = bp.nombres_de_punto(_ids, i18n.pick('Punto', 'Point'))
            _corte_rev = bp.corte_borreguil(threshold)

            # Revisados: los que se han revisado aquí, en este análisis (y los añadidos en
            # el mapa), como en la demo («Revisados 3 de 20», con los 20 puntos con su
            # etiqueta de campo): un punto que llega con etiqueta de tu fichero sigue en
            # «Sin revisar» hasta que lo miras. Se apuntan en la sesión, por análisis (su
            # carpeta), sin tocar los puntos.
            _hechos_rev = st.session_state.get('rev_hechos')
            if not isinstance(_hechos_rev, dict) or _hechos_rev.get('work') != str(work):
                _hechos_rev = {'work': str(work), 'ids': set()}
                st.session_state['rev_hechos'] = _hechos_rev

            def _revisado(r):
                return (r.get('ID') in _hechos_rev['ids']
                        or bp._txt(r.get('source')) == 'mapa')

            def _prob(r):
                try:
                    p = float(r.get('rf_proba'))
                except (TypeError, ValueError):
                    return None
                return p if p == p else None

            def _estado(r):
                """'si', 'no' o 'duda' según lo revisado del punto (None si nada)."""
                if bp._txt(r.get('Duda')).lower() == 'si':
                    return 'duda'
                b = bp._txt(r.get('Borreguil')).lower()
                return b if b in ('si', 'no') else None

            # El resultado de cada punto: la frase bajo su nombre y la etiqueta corta de la
            # lista (las de la demo; «Dudoso (campo)» y «Sin predicción» no están en ella).
            _FRASE = {
                'BORREGUIL VERIFICADO': ('Ya revisado: lo diste por borreguil.',
                                         'Already reviewed: you said it is a borreguil.'),
                'NO BORREGUIL VERIFICADO': ('Ya revisado: lo diste por no borreguil.',
                                            'Already reviewed: you said it is not a '
                                            'borreguil.'),
                'BORREGUIL PROBABLE': ('El modelo lo ve como borreguil.',
                                       'The model sees it as a borreguil.'),
                'POSIBLE BORREGUIL': ('El modelo se inclina por borreguil, sin mucha '
                                      'seguridad.',
                                      'The model leans towards borreguil, without much '
                                      'certainty.'),
                'NO BORREGUIL': ('El modelo no lo ve como borreguil.',
                                 'The model does not see it as a borreguil.'),
                'DUDOSO (campo)': ('Marcado como dudoso: no se usa para entrenar.',
                                   'Marked as doubtful: it is not used for training.'),
                'SIN PREDICCIÓN': ('El modelo no ha podido darle una probabilidad.',
                                   'The model could not give it a probability.'),
            }
            _CORTA = {'BORREGUIL VERIFICADO': ('Verificado', 'Verified'),
                      'NO BORREGUIL VERIFICADO': ('No (verif.)', 'No (verified)'),
                      'BORREGUIL PROBABLE': ('Probable', 'Likely'),
                      'POSIBLE BORREGUIL': ('Posible', 'Possible'),
                      'INCIERTO': ('Incierto', 'Uncertain'),
                      'NO BORREGUIL': ('No borreguil', 'Not borreguil'),
                      'DUDOSO (campo)': ('Dudoso', 'Doubtful'),
                      'SIN PREDICCIÓN': ('Sin dato', 'No data')}

            def _frase(r):
                dec = r.get('decision')
                if dec in ('BORREGUIL VERIFICADO', 'NO BORREGUIL VERIFICADO') \
                        and not _revisado(r):
                    # la etiqueta viene de tu fichero: no lo has revisado aquí
                    return (i18n.pick('Tu fichero de campo dice que es borreguil.',
                                      'Your field file says it is a borreguil.')
                            if dec == 'BORREGUIL VERIFICADO' else
                            i18n.pick('Tu fichero de campo dice que no es borreguil.',
                                      'Your field file says it is not a borreguil.'))
                if dec == 'INCIERTO':
                    p = _prob(r)
                    if p is not None and abs(p - _corte_rev) <= 0.05:
                        return i18n.pick('El modelo duda: está cerca del corte entre '
                                         'borreguil y no borreguil.',
                                         'The model is in doubt: it is close to the cut '
                                         'between borreguil and not borreguil.')
                    return i18n.pick('El modelo no lo tiene claro: algo por debajo del '
                                     'corte.',
                                     'The model is not sure: somewhat below the cut.')
                return i18n.pick(*_FRASE[dec]) if dec in _FRASE else ''

            # --- La lista de la izquierda (arriba, «Revisados N de M» y su barra)
            _n_rev = sum(1 for _r in rows if _revisado(_r))
            col_lado, col_main = st.columns([0.27, 0.73], gap='medium')
            with col_lado, st.container(key='rev_lado'):
                st.html('<div class="rev-cuenta"><b>'
                        + i18n.pick(f'Revisados {_n_rev} de {len(rows)}',
                                    f'Reviewed {_n_rev} of {len(rows)}')
                        + '</b><div class="rev-barra"><span style="width:'
                        f'{100 * _n_rev / max(1, len(rows)):.1f}%"></span></div></div>')
                _FILTROS = {'sin': i18n.pick('Sin revisar', 'Not reviewed'),
                            'dudosos': i18n.pick('Dudosos primero', 'Doubtful first'),
                            'todos': i18n.pick('Todos', 'All')}
                _filtro = st.pills(i18n.pick('Filtrar', 'Filter'), list(_FILTROS),
                                   default='dudosos', required=True,
                                   format_func=_FILTROS.get, key='rev_filtro',
                                   label_visibility='collapsed') or 'dudosos'
                _caja_lista = st.container(height=620, border=False, key='rev_lista_caja')
                with st.container(key='rev_lado_nota'):
                    st.caption(i18n.pick(
                        'Verde: es borreguil (lo dice el modelo o tu revisión); rojo: no lo '
                        'es. Puedes marcar varios a la vez en la tabla «Todos los puntos» (la '
                        'casilla de la izquierda de cada fila) y revisarlos juntos.',
                        'Green: it is a borreguil (the model or your review says so); red: it '
                        'is not. You can tick several at once in the "All the points" table '
                        '(the box on the left of each row) and review them together.'))
            # «Sin revisar»: los que no tienen el sí o el no de una persona; «Dudosos
            # primero»: esos mismos, del más cercano al corte al más lejano; «Todos»: todos,
            # en el orden de la tabla.
            if _filtro == 'todos':
                _lista = list(range(len(rows)))
            else:
                _lista = [i for i, _r in enumerate(rows) if not _revisado(_r)]
                if _filtro == 'dudosos':
                    _lista.sort(key=lambda i: (
                        abs(_prob(rows[i]) - _corte_rev) if _prob(rows[i]) is not None
                        else 9.0, i))

            with col_main:
                _aviso_rev = st.session_state.pop('_rev_msg', None)
                if _aviso_rev:
                    st.success(_aviso_rev)
                _rev_punto = st.container(key='rev_punto')   # arriba, el punto que se revisa
                _rev_tabla = st.container(key='rev_tabla')   # debajo, todos los puntos
            # --- El punto (o los puntos) que se revisa: lo marcado en la tabla o, si no hay
            # nada marcado, el abierto en la lista (al entrar, el primero de la lista). Lo
            # marcado se lee del estado de la tabla antes de dibujarla, para que su fila
            # salga resaltada (la demo).
            sel_rows = _table_sel_rows()
            _sel_ok = [i for i in sel_rows if isinstance(i, int) and 0 <= i < len(rows)]
            _de_tabla = bool(_sel_ok)
            if _de_tabla:
                focus_idx = st.session_state.get('sel_idx')
                idx = (focus_idx if focus_idx is not None and focus_idx < len(rows)
                       else _sel_ok[-1])
            else:
                _abierto = st.session_state.get('rev_punto_id')
                idx = (_ids.index(_abierto) if _abierto in _ids
                       else (_lista[0] if _lista else 0))
                _sel_ok = [idx]
            st.session_state['rev_punto_id'] = _ids[idx]
            sel_ids = [rows[i].get('ID') for i in _sel_ok]
            _n = len(sel_ids)
            sel = dict(rows[idx])

            with _rev_tabla:
                st.html('<div class="rev-tabla-tit"><h2>'
                        + i18n.pick('Todos los puntos', 'All the points') + '</h2><span>'
                        + i18n.pick(
                            'La probabilidad es la de ser borreguil según el modelo, de 0 a 1 '
                            f'(por encima de {_num(_corte_rev)}, borreguil). Pincha en el '
                            'nombre de una columna para ordenar (otra vez, al revés) y en una '
                            'fila para revisar ese punto en el mapa.',
                            'The probability is that of being borreguil according to the '
                            f'model, from 0 to 1 (above {_num(_corte_rev)}, borreguil). Click '
                            'a column name to sort (again, the other way) and a row to review '
                            'that point on the map.')
                        + '</span></div>')
                # Variables: las 3 que más pesan en el modelo usado, una por grupo de
                # parecidas (como en la ficha del punto), y la pendiente. Antes salían 4 de
                # fin de verano que dicen casi lo mismo y que el modelo ya no usa, el TWI
                # (vacío con Earth Engine; desde el 20-11-2026 los modelos no lo usan) y el
                # «patrón», un código interno.
                _rs_tab = load_ref_stats(MODEL_PATH)
                _clave_tab = pp.variables_clave(
                    (_rs_tab or {}).get('importances'),
                    set((_rs_tab or {}).get('model_features') or []) or None, 3)
                _vars_tab = [k['feat'] for k in _clave_tab]
                if 'slope_deg' not in _vars_tab:
                    _vars_tab.append('slope_deg')
                # (Sin la cuenca: para revisar no aporta y salía vacía en los puntos que no
                # la traen; quitada el 25-11-2026 a petición de la autora. Sigue en el
                # punto, para la validación por zonas, y en las descargas.)
                # Las columnas de la demo (27-11-2026): Punto, Origen (de dónde viene: «tu
                # fichero»…), ¿Borreguil? (revisado), Resultado, Probabilidad, el tipo, las
                # variables con su peso, la pendiente, la distancia al cauce y las
                # coordenadas. Sin «origin» (verificado o dudoso), que ya lo dice el
                # resultado; el cuartil, solo si los puntos lo traen.
                cols_show = (['ID', 'source', 'cuartil', 'Borreguil', 'decision',
                              'rf_proba', 'ambiente', 'humedad', 'pureza', 'tipo_revisado']
                             + _vars_tab + ['dist_cauce_m', 'lat', 'lon'])
                df = pd.DataFrame(rows)
                if 'source' not in df.columns:
                    df['source'] = ''
                cols_show = [c for c in cols_show if c in df.columns]
                # Las columnas opcionales que están vacías en todos los puntos (p. ej.
                # la cuenca, si el fichero no la trae) no se muestran: solo ocupaban
                # ancho y empujaban fuera de la pantalla la decisión y el tipo.
                _opcionales = ('cuenca_id', 'cuartil', 'Borreguil', 'dist_cauce_m')
                cols_show = [c for c in cols_show if c not in _opcionales
                             or any(bp._txt(v) != '' for v in df[c])]
                # Los valores del pipeline son españoles; se traducen SOLO para
                # mostrarlos (df_show), sin tocar `rows`, del que depende la lógica.
                df_show = df[cols_show].copy()
                # Las variables que llegan como texto (de un CSV, p. ej. «0.10313319351…»),
                # como números: si no, la tabla las enseñaba con todos sus decimales en vez
                # de con 3 (25-11-2026). Solo si todo lo que hay en la columna es un número.
                for _c in df_show.columns:
                    if _c in ('ID', 'source') or pd.api.types.is_numeric_dtype(df_show[_c]):
                        continue
                    _llenos = [v for v in df_show[_c]
                               if not (v is None or (isinstance(v, float) and v != v)
                                       or (isinstance(v, str) and v.strip().lower()
                                           in ('', 'none', 'nan')))]
                    if _llenos and not any(isinstance(v, bool) for v in _llenos):
                        _num_c = pd.to_numeric(pd.Series(_llenos), errors='coerce')
                        if _num_c.notna().all():
                            df_show[_c] = pd.to_numeric(df_show[_c], errors='coerce')
                # Celdas sin dato en las columnas de texto: vacías, no «None» (la columna
                # «Borreguil» de los puntos sin etiqueta de campo lo enseñaba así).
                # (las de texto no siempre son «object»: con pandas 3 son «str», y lo que
                # falta es NaN)
                for _c in df_show.columns:
                    if not pd.api.types.is_numeric_dtype(df_show[_c]):
                        df_show[_c] = df_show[_c].map(
                            lambda v: '' if v is None or (isinstance(v, float) and v != v)
                            or (isinstance(v, str) and v.strip().lower() in ('none', 'nan'))
                            else v)
                if 'Borreguil' in df_show.columns:
                    df_show['Borreguil'] = df_show['Borreguil'].map(
                        lambda v: {'si': i18n.pick('sí', 'yes'), 'no': 'no'}.get(
                            bp._txt(v).lower(), ''))
                # De dónde viene el punto («truth» era un nombre interno: los puntos de
                # campo que no coincidían con ninguno de los tuyos, 25-11-2026).
                df_show['source'] = df_show['source'].map(
                    lambda v: {'': i18n.pick('tu fichero', 'your file'),
                               'truth': i18n.pick('campo', 'field'),
                               'mapa': i18n.pick('mapa', 'map')}.get(bp._txt(v), bp._txt(v)))
                for _c in ('decision', 'ambiente', 'humedad', 'pureza', 'tipo_borreguil'):
                    if _c in df_show.columns:
                        # la decisión, en texto normal («Borreguil verificado»)
                        df_show[_c] = df_show[_c].map(i18n.tv_decision if _c == 'decision'
                                                      else i18n.tv)
                if 'tipo_revisado' in df_show.columns:
                    # «sí» con los tres niveles revisados; si no, cuáles
                    def _rev_txt(r):
                        nv = bp.niveles_revisados(r)
                        if len(nv) == len(bp.NIVELES_TIPO):
                            return i18n.pick('sí', 'yes')
                        return ', '.join(i18n.pick(n, bp.COLUMNA_TIPO_EN[n]) for n in nv)
                    df_show['tipo_revisado'] = [_rev_txt(r) for r in rows]
                # Los ID muy largos se abrevian SOLO en la tabla, conservando el final, que
                # es lo que los distingue; si no, la columna se come el ancho y hay que
                # desplazarse para ver la decisión y el tipo. Además queda fija al
                # desplazar la tabla hacia la derecha.
                # Y los heredados del nombre de la capa ('all_…_epsg25830.17'), solo con
                # su número ('17'), como en el globo del mapa (25-11-2026, la autora).
                _cfg_tabla = {}
                # Cabeceras legibles («Resultado», «Altitud (m)»…), no los nombres internos;
                # el punto se llama «Punto» y la fuente, «Origen» (la demo), y las 3
                # variables que más pesan llevan su peso: «Altitud (m) (13 %)».
                _peso_de = {k['feat']: k['peso'] for k in _clave_tab}
                _col = {c: _nombre_columna(c) for c in df_show.columns}
                _col['ID'] = i18n.pick('Punto', 'Point')
                _col['source'] = i18n.pick('Origen', 'Origin')
                for _f, _p in _peso_de.items():
                    if _f in _col and _p is not None:
                        _col[_f] = f'{_col[_f]} ({100 * _p:.0f} %)'
                _ids_t = [str(x) for x in df_show['ID'].tolist()]
                _cortos_t = bp.nombres_de_punto(_ids_t, punto='')
                df_show['ID'] = bp.abreviar_ids(_cortos_t)
                df_show = df_show.rename(columns=_col)
                _cfg_tabla[_col['ID']] = st.column_config.Column(pinned=True, help=(i18n.pick(
                    'El número del punto: el nombre de la capa de la que viene se ha '
                    'quitado para que se lea mejor. El nombre completo sale en las '
                    'descargas.',
                    'The number of the point: the name of the layer it comes from has '
                    'been removed so that it reads better. The full name is in the '
                    'downloads.') if _cortos_t != _ids_t else i18n.pick(
                    'Los nombres largos se muestran abreviados («…»). El nombre '
                    'completo aparece al abrir el punto y en las descargas.',
                    'Long names are shown abbreviated ("…"). The full name appears '
                    'when you open the point and in the downloads.')))
                # Los números, con 3 decimales; las coordenadas con 5 (con 3 se moverían
                # unos 100 m). Solo cambia lo que se ve: `rows` conserva el valor completo.
                _coords = {_col.get('lat'), _col.get('lon')} & set(df_show.columns)
                # Al pasar el ratón por el nombre de una de las 3 variables que más pesan,
                # se dice por qué está en la tabla y cuánto pesa.
                for _f, _p in _peso_de.items():
                    _c = _col.get(_f)
                    if _c not in df_show.columns:
                        continue
                    _p_es = '' if _p is None else f' ({100 * _p:.0f} % del peso)'
                    _p_en = '' if _p is None else f' ({100 * _p:.0f} % of the weight)'
                    _cfg_tabla[_c] = st.column_config.NumberColumn(help=i18n.pick(
                        f'Una de las 3 variables que más pesan en el modelo{_p_es}.',
                        f'One of the 3 variables that weigh most in the model{_p_en}.'))
                for _c in df_show.columns:
                    if _c not in _cfg_tabla and pd.api.types.is_float_dtype(df_show[_c]):
                        _cfg_tabla[_c] = st.column_config.NumberColumn()
                # «Resultado» (antes «Decisión»), al pasar el ratón: si el punto está revisado,
                # manda lo revisado; si no,
                # la probabilidad (25-11-2026, la autora: «¿por qué pone decisión borreguil
                # verificado?» → «Sí, explícalo»). Con los cortes de bp.decide().
                _col_dec = _col.get('decision')
                if _col_dec in df_show.columns:
                    _d = i18n.tv_decision
                    _corte = bp.corte_borreguil(threshold)
                    _posible = _corte < 0.7 - 1e-9
                    _cfg_tabla[_col_dec] = st.column_config.TextColumn(help=i18n.pick(
                        'Si el punto está revisado (en la app o ya en tu fichero), manda lo '
                        f'revisado: «sí» → «{_d("BORREGUIL VERIFICADO")}», «no» → '
                        f'«{_d("NO BORREGUIL VERIFICADO")}» y dudoso → «{_d("DUDOSO (campo)")}». '
                        'Los puntos sin revisar los decide el modelo con su probabilidad: '
                        + (f'desde {_n3(_corte)} (el umbral), «{_d("POSIBLE BORREGUIL")}», y '
                           if _posible else '')
                        + f'desde {_n3(0.7)}, «{_d("BORREGUIL PROBABLE")}»; hasta {_n3(0.2)}, '
                        f'«{_d("NO BORREGUIL")}»; entre medias, «{_d("INCIERTO")}». La '
                        'probabilidad se calcula siempre, también en los revisados.',
                        'If the point is reviewed (in the app or already in your file), the '
                        f'review decides: "yes" → "{_d("BORREGUIL VERIFICADO")}", "no" → '
                        f'"{_d("NO BORREGUIL VERIFICADO")}" and doubtful → '
                        f'"{_d("DUDOSO (campo)")}". Points not reviewed are decided by the '
                        'model with its probability: '
                        + (f'from {_n3(_corte)} (the threshold), "{_d("POSIBLE BORREGUIL")}", and '
                           if _posible else '')
                        + f'from {_n3(0.7)}, "{_d("BORREGUIL PROBABLE")}"; up to {_n3(0.2)}, '
                        f'"{_d("NO BORREGUIL")}"; in between, "{_d("INCIERTO")}". The '
                        'probability is always computed, also for reviewed points.'))
                # Las demás columnas, al pasar el ratón por su nombre (25-11-2026, la autora:
                # «que explique cada campo de la tabla al pasar el ratón por encima»; el
                # inventario de la demo las conserva). Las 3 variables que más pesan ya
                # llevan la suya (arriba).
                _ayudas = {
                    'source': (
                        'De dónde viene el punto. «tu fichero»: de tu fichero «Tus puntos». '
                        '«campo»: de tu fichero de puntos de campo, en un sitio donde no había '
                        'ninguno de tus puntos. «mapa»: lo añadiste pinchando en el mapa.',
                        'Where the point comes from. "your file": from your "Your points" '
                        'file. "field": from your field points file, at a place where none of '
                        'your points was. "map": you added it by clicking on the map.'),
                    'cuartil': (
                        'Si el punto se lanzó por cuartiles de una capa, el cuartil del polígono '
                        'en el que cayó: 1 es el primero (el de los valores más altos).',
                        'If the point was placed by quartiles of a layer, the quartile of the '
                        'polygon it fell in: 1 is the first (the one with the highest values).'),
                    'Borreguil': (
                        'Lo que ha comprobado una persona: «sí», es borreguil; «no», no lo es. '
                        'Viene de tu fichero (columna presence o Borreguil), de tu revisión en '
                        '«2 · Revisar» o de un punto añadido en el mapa. Vacío: sin revisar, y '
                        'entonces decide el modelo.',
                        'What a person has checked: "yes", it is a borreguil; "no", it is not. '
                        'It comes from your file (presence or Borreguil column), from your review '
                        'in "2 · Review" or from a point added on the map. Empty: not reviewed, so '
                        'the model decides.'),
                    'rf_proba': (
                        'Probabilidad de que sea borreguil según el modelo, de 0 a 1. Se calcula '
                        'siempre, también en los puntos revisados, para comparar.',
                        'Probability of being a borreguil according to the model, from 0 to 1. '
                        'It is always computed, also for reviewed points, to compare.'),
                    'ambiente': (
                        'Dónde está el borreguil. laguna: a menos de 120 m de una laguna conocida '
                        'o con agua cerca en la imagen; arroyo: a 30 m o menos de un río o arroyo '
                        'de OpenStreetMap; ladera: el resto. Es la propuesta de la app, salvo que '
                        'lo hayas revisado.',
                        'Where the borreguil is. lake: less than 120 m from a known lake or with '
                        'water nearby in the image; stream: 30 m or less from an OpenStreetMap '
                        'river or stream; slope: the rest. It is the app\'s proposal unless you '
                        'reviewed it.'),
                    'humedad': (
                        'húmedo si el NDMI (humedad de la vegetación) de junio a septiembre pasa '
                        'de 0,100; si no, seco. Es la propuesta de la app, salvo que lo hayas '
                        'revisado.',
                        'wet if the NDMI (vegetation moisture) from June to September is above '
                        '0.100; otherwise dry. It is the app\'s proposal unless you reviewed it.'),
                    'pureza': (
                        'Cuánto verde hay en el píxel de 10 m del punto: puro si llega al 75 %; '
                        'si no, mixto-agua si el píxel es agua o toca agua, y mixto-roca si no. Se '
                        'lee con Earth Engine: sin él, sale «—».',
                        'How much green there is in the 10 m pixel of the point: pure if it '
                        'reaches 75 %; otherwise mixed-water if the pixel is water or touches '
                        'water, and mixed-rock if not. It is read with Earth Engine: without it, '
                        'it shows "—".'),
                    'tipo_revisado': (
                        '«sí» si una persona ha revisado los tres niveles del tipo; si solo '
                        'algunos, cuáles. Vacío: el tipo es la propuesta de la app.',
                        '"yes" if a person reviewed the three levels of the type; if only some, '
                        'which ones. Empty: the type is the app\'s proposal.'),
                    'slope_deg': (
                        'Pendiente del terreno, en grados, del Copernicus DEM. Va siempre, como '
                        'referencia del terreno.',
                        'Terrain slope, in degrees, from the Copernicus DEM. It is always there, '
                        'as a reference for the terrain.'),
                    'dist_cauce_m': (
                        'Metros al río o arroyo de OpenStreetMap más cercano. Con 30 m o menos, '
                        'el ambiente propuesto es «arroyo».',
                        'Metres to the nearest OpenStreetMap river or stream. With 30 m or less, '
                        'the proposed environment is "stream".'),
                    'lat': ('Latitud del punto, en grados (WGS84), con 5 decimales: '
                            'alrededor de 1 m.',
                            'Latitude of the point, in degrees (WGS84), with 5 decimals: '
                            'about 1 m.'),
                    'lon': ('Longitud del punto, en grados (WGS84), con 5 decimales: '
                            'alrededor de 1 m.',
                            'Longitude of the point, in degrees (WGS84), with 5 decimals: '
                            'about 1 m.'),
                }
                for _k, (_a_es, _a_en) in _ayudas.items():
                    _c = _col.get(_k)
                    if _c not in df_show.columns or _k in _peso_de:
                        continue
                    _ayuda = i18n.pick(_a_es, _a_en)
                    if pd.api.types.is_numeric_dtype(df_show[_c]):
                        _cfg_tabla[_c] = st.column_config.NumberColumn(help=_ayuda)
                    else:
                        _cfg_tabla[_c] = st.column_config.TextColumn(help=_ayuda)
                # Los colores de la demo: el punto en verde claro, el resultado del color de
                # su etiqueta, la probabilidad en verde o en rojo (borreguil o no) y la fila
                # del punto abierto, con fondo verde oscuro. (Cambiar los colores no
                # desmarca la tabla: su estado va por la clave y el modo de selección.)
                _tabla = _con_coma(df_show, decimales={c: 5 for c in _coords})
                from pandas.io.formats.style import Styler as _Styler
                _sty = _tabla if isinstance(_tabla, _Styler) else df_show.style
                _si_no = [_es_borreguil(r) for r in rows]
                _sty = _sty.map(lambda v: 'color:#A3D9B5;font-weight:600',
                                subset=[_col['ID']])
                if _col_dec in df_show.columns:
                    _sty = _sty.apply(lambda s: [
                        'color:' + ('#A5D6A7' if _si_no[i] else '#F5A3A1') + ';font-weight:600'
                        for i in range(len(s))], subset=[_col_dec], axis=0)
                _col_p = _col.get('rf_proba')
                if _col_p in df_show.columns:
                    _sty = _sty.apply(lambda s: [
                        'color:' + ('#7FD483' if _si_no[i] else '#F28B88') + ';font-weight:700'
                        for i in range(len(s))], subset=[_col_p], axis=0)
                _sty = _sty.apply(lambda s: ['background-color:#1B2B21' if i == idx else ''
                                             for i in range(len(s))], axis=0)
                # Todas las filas a la vista, como en la demo (con muchos puntos, desplazable)
                st.dataframe(_sty, hide_index=True, width='stretch',
                             height='content' if len(rows) <= 30 else 640,
                             on_select='rerun', selection_mode='multi-row',
                             column_config=_cfg_tabla, key='results_table')
                # Por qué salen esas variables, en una frase junto a la tabla (25-11-2026, la
                # autora: «que eso se explique brevemente cerca de la tabla»), con el texto
                # de la demo. El peso, sin decimales, como en la cabecera.
                _lista_tab = [k for k in _clave_tab if _col.get(k['feat']) in df_show.columns]
                if _lista_tab:
                    def _con_peso(k):
                        if k['peso'] is None:
                            return _nombre_columna(k['feat'])
                        return f"{_nombre_columna(k['feat'])} ({100 * k['peso']:.0f} %)"
                    _nombres_tab = [_con_peso(k) for k in _lista_tab]
                    _y = i18n.pick(' y ', ' and ')
                    _txt_vars = (_nombres_tab[0] if len(_nombres_tab) == 1 else
                                 ', '.join(_nombres_tab[:-1]) + _y + _nombres_tab[-1])
                    _con_pesos = all(k['peso'] is not None for k in _lista_tab)
                    _cir_tab = any(str(k['feat']).startswith('cir_') for k in _lista_tab)
                    st.caption(i18n.pick(
                        'Por qué estas variables: '
                        + (f'son las {len(_lista_tab)} que más pesan en el modelo que usas, una '
                           'por cada grupo de variables que dicen casi lo mismo (el peso es el del '
                           f'grupo): {_txt_vars}. ' if _con_pesos else
                           f'{_txt_vars}: el modelo no trae el peso de sus variables, así que '
                           'salen las que más separan los borreguiles en Sierra Nevada. ')
                        + 'La pendiente está siempre, como referencia del terreno. Si cambias de '
                        'modelo, cambian.'
                        + (' Las CIR salen de la ortofoto en falso color infrarrojo a 0,25 m, de '
                           'un recorte de 50 × 50 m alrededor de cada punto.' if _cir_tab else ''),
                        'Why these variables: '
                        + (f'they are the {len(_lista_tab)} that weigh most in the model in use, '
                           'one for each group of variables that say almost the same thing (the '
                           f'weight is the group\'s): {_txt_vars}. ' if _con_pesos else
                           f'{_txt_vars}: the model does not carry the weight of its variables, '
                           'so these are the ones that best separate borreguiles in Sierra '
                           'Nevada. ')
                        + 'The slope is always there, as a reference for the terrain. If you '
                        'change the model, they change.'
                        + (' The CIR ones come from the false-colour infrared orthophoto at '
                           '0.25 m, from a 50 × 50 m clip around each point.' if _cir_tab else '')))

            def _siguiente():
                """El ID del punto que se abre tras guardar: el siguiente de la lista de la
                izquierda (con «Todos», el siguiente de la tabla) que no se acabe de
                guardar."""
                if _lista:
                    if idx in _lista:
                        _k = _lista.index(idx)
                        _cand = _lista[_k + 1:] + _lista[:_k]
                    else:
                        _cand = ([i for i in _lista if i > idx]
                                 + [i for i in _lista if i <= idx])
                else:
                    _cand = list(range(idx + 1, len(rows))) + list(range(0, idx))
                _libres = [i for i in _cand if _ids[i] not in sel_ids]
                return _ids[(_libres or _cand or [idx])[0]]

            # La lista: un botón por punto (un st.radio con aspecto de lista), con su
            # color, su nombre, su probabilidad y su resultado. El que está abierto, marcado.
            def _abrir_de_lista():
                _v = st.session_state.get('rev_lista')
                if _v is not None:
                    st.session_state['rev_punto_id'] = _v
                    st.session_state['_vaciar_seleccion'] = True   # la tabla deja de mandar
                    st.session_state.sel_point = None

            def _md_rev(t):
                return ''.join('\\' + ch if ch in '\\`*_{}[]()#+-.!|<>~:' else ch
                               for ch in str(t))

            # Las etiquetas, hechas ya (un dict, no una función que mire variables del
            # guion: aquí todo es global y otra pantalla puede reutilizar los nombres).
            _etiqueta_lista = {}
            for _i in _lista:
                _r = rows[_i]
                _c = 'green' if _es_borreguil(_r) else 'red'
                _p = _prob(_r)
                _corta = i18n.pick(*_CORTA.get(_r.get('decision'), (
                    i18n.tv_decision(_r.get('decision') or '—'),) * 2))
                _etiqueta_lista[_ids[_i]] = (
                    f':{_c}[●] **{_md_rev(_nombres_sel[_i])}** '
                    f'*{"—" if _p is None else f"{_p * 100:.0f} %"}* '
                    f':{_c}-badge[{_md_rev(_corta)}]')

            _ids_lista = [_ids[i] for i in _lista]
            st.session_state['rev_lista'] = _ids[idx] if _ids[idx] in _ids_lista else None
            with _caja_lista:
                if _ids_lista:
                    st.radio(i18n.pick('Puntos', 'Points'), _ids_lista, index=None,
                             format_func=_etiqueta_lista.get, key='rev_lista',
                             on_change=_abrir_de_lista, label_visibility='collapsed',
                             width='stretch')
                else:
                    st.caption(i18n.pick('No queda ningún punto sin revisar.',
                                         'There are no points left to review.'))

            with _rev_punto:
                _espera = (i18n.pick('Guardando y reentrenando…', 'Saving and retraining…')
                           if _propio else i18n.pick('Guardando…', 'Saving…'))
                _reentrenado = i18n.pick(' Modelo reentrenado sumando tus puntos.',
                                         ' Model retrained adding your points.')
                colL, colR = st.columns([0.64, 0.36], gap='medium')
                with colL:
                    # El nombre del punto y la frase de su resultado (la demo)
                    import html as _html_rev
                    st.html('<div class="rev-titulo"><h1>'
                            + bp.nombre_html(_nombres_sel[idx]) + '</h1><span>'
                            + _html_rev.escape(_frase(sel)) + '</span></div>')
                    # Mapa para MIRAR el punto (y revisar su tipo): ortofoto de fondo, el
                    # punto y, en amarillo, el píxel Sentinel-2 de 10 m al que se refiere
                    # la pureza.
                    fm = _base_map([sel['lat'], sel['lon']], 17, ortofoto=True, escala=False)
                    # Solo el punto y su píxel (la autora, 26-11-2026: «solo quiero que
                    # salga el punto de verdad y el píxel sentinel en amarillo»): el
                    # píxel, amarillo con un halo oscuro para que se vea en cualquier
                    # fondo; el punto, pequeño, verde si es borreguil y rojo si no. (Sin
                    # textos al pasar el ratón: la leyenda del mapa dice qué es cada cosa.)
                    try:
                        _px10 = bp.pixel_footprints([sel])[0]['pixel10']
                        folium.Polygon(_px10, color='#000000', weight=7, opacity=0.45,
                                       fill=False).add_to(fm)
                        folium.Polygon(_px10, color='#F5C842', weight=3, fill=False).add_to(fm)
                    except Exception:
                        pass                      # sin la huella el mapa sigue sirviendo
                    folium.CircleMarker(
                        [sel['lat'], sel['lon']], radius=6, weight=3,
                        color='rgba(0,0,0,0.6)', fill=True, fill_opacity=1,
                        fill_color=COLOR_SI if _es_borreguil(sel) else COLOR_NO).add_to(fm)
                    # La ventana de unos 50 m que promedia el detector de puntos (5 × 5
                    # píxeles de 10 m), apagada de entrada; y los ríos y arroyos de
                    # OpenStreetMap, los mismos con los que se propone «arroyo» (a 30 m
                    # o menos), encendidos (25-11-2026, la autora).
                    try:
                        _g_win = folium.FeatureGroup(
                            name=i18n.pick('▢ Ventana del detector de puntos (50 m)',
                                           '▢ Point detector window (50 m)'), show=False)
                        folium.Polygon(bp.pixel_footprints([sel])[0]['win_s2_10'],
                                       color='#00E5FF', weight=2, fill=False,
                                       dash_array='6').add_to(_g_win)
                        _g_win.add_to(fm)
                    except Exception:
                        pass
                    _dist_osm = None
                    try:
                        _cerca = _cauces_cerca(float(sel['lon']), float(sel['lat']))
                    except Exception:                 # noqa: BLE001
                        _cerca = None
                    _capa_rios = i18n.pick('〰 Ríos y arroyos (OpenStreetMap)',
                                           '〰 Rivers and streams (OpenStreetMap)')
                    if _cerca is not None:
                        if _cerca:
                            _tmp = [{'lat': sel['lat'], 'lon': sel['lon']}]
                            bp.attach_cauce_distance(_tmp, _cerca)
                            _dist_osm = _tmp[0].get('dist_cauce_m')
                        _g_osm = folium.FeatureGroup(name=_capa_rios, show=True)
                        for _l in _cerca:
                            for _p in getattr(_l, 'geoms', [_l]):
                                folium.PolyLine(
                                    [(y, x) for x, y, *_ in _p.coords], color='#1565C0',
                                    weight=3, opacity=0.9).add_to(_g_osm)
                        _g_osm.add_to(fm)
                    folium.LayerControl(collapsed=False).add_to(fm)
                    # La leyenda, sobre el mapa, abajo a la izquierda (la demo): qué es el
                    # recuadro, qué es el punto y, mientras estén encendidos los ríos, a
                    # cuánto está el más cercano.
                    if _dist_osm is not None and _dist_osm == _dist_osm:
                        _rios_txt = i18n.pick(
                            f' · en azul, ríos y arroyos; el más cercano, a {_dist_osm:.0f} m',
                            f' · in blue, rivers and streams; the nearest, {_dist_osm:.0f} m '
                            'away')
                    elif _cerca is not None:
                        _rios_txt = i18n.pick(
                            ' · no hay ningún río ni arroyo de OpenStreetMap a menos de unos '
                            '3 km',
                            ' · there is no OpenStreetMap river or stream within about 3 km')
                    else:
                        _rios_txt = ''
                    _leyenda_mapa = (i18n.pick(
                        'Recuadro amarillo: el píxel Sentinel-2 de 10 m · punto: dónde está el '
                        'punto, verde si es borreguil y rojo si no',
                        'Yellow square: the 10 m Sentinel-2 pixel · dot: where the point is, '
                        'green if it is a borreguil and red if not')
                        + f'<span id="rev-rios">{_rios_txt}</span>')
                    fm.get_root().html.add_child(folium.Element(
                        '<div class="rev-leyenda" style="position:fixed;left:16px;bottom:14px;'
                        'max-width:60%;z-index:9999;background:#221F20E6;border:1px solid '
                        '#494245;border-radius:8px;padding:6px 10px;color:#CDD3D9;'
                        'font:12px/1.35 Inter,system-ui,sans-serif">' + _leyenda_mapa
                        + '</div>'))
                    _capas_oscuras(fm, nota_id='rev-rios', nota_capa=_capa_rios)
                    # (para comprobarlo sin navegador: lo que dice la leyenda del mapa)
                    st.session_state['_rev_leyenda_mapa'] = _leyenda_mapa
                    with st.container(key='rev_mapa'):
                        st_folium(fm, width=None, height=420, returned_objects=[],
                                  center=[sel['lat'], sel['lon']], zoom=17, key='focusmap')
                    # Por qué miramos dentro del recuadro amarillo (el recuadro de la demo,
                    # abierto de entrada).
                    _por_que = i18n.pick(
                        ('Por qué miramos dentro del recuadro amarillo',
                         'El recuadro es el <b>píxel de 10 × 10 m</b> de Sentinel-2 en el que '
                         'cae el punto. Es lo que mira el satélite: el verdor, la humedad y el '
                         'agua que lee el modelo son la mezcla de todo lo que hay dentro de ese '
                         'cuadrado, no del sitio exacto del punto.',
                         'Y es con lo que se hace el mapa: el Random Forest de «Hacer el mapa» '
                         'aprende de ese píxel y después clasifica la zona píxel a píxel. Por '
                         'eso la pregunta no es «¿hay borreguil justo en el punto?», sino '
                         '<b>«¿este cuadrado es borreguil?»</b>. Si el punto cae en un '
                         'borreguil pero el cuadrado es casi todo roca o agua, sus valores no '
                         'son los de un borreguil, y enseñarle al modelo que sí lo es lo '
                         'confundiría.',
                         'Además, el punto puede estar unos metros desplazado (el GPS, la '
                         'ortofoto), y el detector de puntos promedia una ventana de unos 50 m '
                         'alrededor (enciéndela en «Capas» para verla).',
                         'Si el cuadrado es una mezcla, dilo en la pureza (mixto con agua o con '
                         'roca) o marca «No estoy seguro/a».'),
                        ('Why we look inside the yellow square',
                         'The square is the <b>10 × 10 m pixel</b> of Sentinel-2 the point falls '
                         'in. It is what the satellite sees: the greenness, moisture and water '
                         'the model reads are the mix of everything inside that square, not of '
                         'the exact spot of the point.',
                         'And it is what the map is made with: the Random Forest of "Make the '
                         'map" learns from that pixel and then classifies the area pixel by '
                         'pixel. That is why the question is not "is there borreguil exactly at '
                         'the point?", but <b>"is this square borreguil?"</b>. If the point '
                         'falls on a borreguil but the square is almost all rock or water, its '
                         'values are not those of a borreguil, and teaching the model that it '
                         'is would confuse it.',
                         'Also, the point may be a few metres off (the GPS, the orthophoto), and '
                         'the point detector averages a window of about 50 m around it (turn it '
                         'on in "Layers" to see it).',
                         'If the square is a mix, say so in the purity (mixed with water or '
                         'with rock) or choose "Not sure".'))
                    st.html('<details class="rev-amarillo" open><summary>' + _por_que[0]
                            + '</summary><div class="rev-amarillo-cuerpo">'
                            + ''.join(f'<span>{t}</span>' for t in _por_que[1:])
                            + '</div></details>')
                    # Lo que más pesa en el modelo: las 3 variables que más pesan, una por
                    # grupo de parecidas, en tres tarjetas con el valor del punto y la mitad
                    # central de los borreguiles (la demo). Debajo, la pendiente, las
                    # coordenadas y el enlace a Google Maps.
                    _rs_sel = load_ref_stats(MODEL_PATH)
                    _mf = set((_rs_sel or {}).get('model_features') or []) or None
                    _clave3 = pp.variables_clave((_rs_sel or {}).get('importances'), _mf, 3)
                    _feats_ref = (_rs_sel or {}).get('features') or {}
                    _tarj = ''
                    for k in _clave3:
                        _si = ((_feats_ref.get(k['feat']) or {}).get('si')) or {}
                        _rango = (f"{_num(_si.get('p25'))}–{_num(_si.get('p75'))}"
                                  if _si else '')
                        _tarj += (
                            '<div><span class="e">' + _html_rev.escape(pp.etiqueta(k['feat']))
                            + ('' if k['peso'] is None else
                               i18n.pick(' · pesa ', ' · weighs ') + f"{100 * k['peso']:.0f} %")
                            + f'</span><span class="v">{_num(sel.get(k["feat"]))}</span>'
                            + (f'<span class="r">{i18n.pick("borreguiles", "borreguiles")}: '
                               f'{_rango}</span>' if _rango else '') + '</div>')
                    st.html('<div class="rev-clave"><span class="t">' + i18n.pick(
                        'Lo que más pesa en el modelo (las 3 que más pesan, una por cada grupo '
                        'de variables que dicen casi lo mismo)',
                        'What weighs most in the model (the 3 that weigh most, one for each '
                        'group of variables that say almost the same thing)')
                        + '</span><div class="rev-tarjetas">' + _tarj + '</div></div>')
                    gmaps = ('https://www.google.com/maps/search/?api=1&query='
                             f"{sel.get('lat')},{sel.get('lon')}")
                    with st.container(key='rev_linea'):
                        st.markdown(
                            f"{pp.etiqueta('slope_deg')}: **{_num(sel.get('slope_deg'))}** · "
                            f"Coords: {_num(sel.get('lat'), 5)}{i18n.pick('; ', ', ')}"
                            f"{_num(sel.get('lon'), 5)} · "
                            f"[{i18n.pick('Abrir en Google Maps', 'Open in Google Maps')}]"
                            f"({gmaps})")
                    # Este punto frente a los borreguiles verdaderos (abierto de entrada)
                    render_point_profile(sel, _rs_sel)

                with colR, st.container(key='rev_tarjeta'):
                    import hashlib as _hl
                    # Lo que dice la app del punto, en grande (25-11-2026, la autora): el
                    # resultado, la probabilidad con su barra y el tipo.
                    _pie = ''
                    if sel.get('tipo_borreguil'):
                        _pie = ('<span style="font-size:13px;color:#CDD3D9">'
                                + i18n.pick('Tipo: ', 'Type: ')
                                + '<span style="font-weight:700;color:#ECEFF1">'
                                + _html_rev.escape(i18n.tv(sel['tipo_borreguil'])) + '</span> · '
                                + (i18n.pick('revisado', 'reviewed') if bp.tipo_revisado(sel)
                                   else i18n.pick('propuesta de la app, sin revisar',
                                                  "app's proposal, not reviewed"))
                                + '</span>')
                    st.html(bloque_probabilidad(sel, st.session_state.get('eff_threshold'),
                                                pie=_pie))
                    if _n > 1:
                        st.caption(i18n.pick(
                            f'**{_n} puntos seleccionados** en la tabla: lo que guardes se '
                            'aplica a todos.',
                            f'**{_n} points selected** in the table: what you save applies to '
                            'all of them.'))
                    # La clave de los controles depende de QUÉ puntos se revisan (por ID: al
                    # quitar uno las posiciones se corren) y de cuántas veces se ha guardado
                    # o deshecho: arrancan con lo vigente de esos puntos, nunca con lo
                    # elegido para otro.
                    _firma = _hl.md5('|'.join(
                        [str(x) for x in sel_ids] + [str(st.session_state.get('_rev_n', 0))]
                    ).encode()).hexdigest()[:10]
                    # De entrada, lo revisado o, si no, lo que dice el modelo: «Sí» si lo da
                    # por borreguil y «No» si no (la demo). Con varios puntos que no
                    # coinciden, nada.
                    _inis = {(_estado(rows[i]) or ('si' if _es_borreguil(rows[i]) else 'no'))
                             for i in _sel_ok}
                    _es_ini = next(iter(_inis)) if len(_inis) == 1 else None
                    _ES_TXT = {'si': i18n.pick('Sí', 'Yes'), 'no': 'No',
                               'duda': i18n.pick('No estoy seguro/a', 'Not sure')}
                    with st.container(horizontal=True, vertical_alignment='center',
                                      key='rev_pregunta'):
                        st.html('<span class="rev-preg">'
                                + i18n.pick('¿Es borreguil?', 'Is it a borreguil?')
                                + '</span>', width='content')
                        ayuda_q('q_rev_es',
                                'Míralo en la ortofoto (o con tus datos de campo), en todo el '
                                'recuadro amarillo y no solo en el punto: el mapa del Random '
                                'Forest aprende de ese píxel de 10 m, y el detector de puntos '
                                'promedia unos 50 m alrededor.\n\n'
                                'Sí o No lo dejan verificado. No estoy seguro/a lo deja como '
                                'dudoso: no entrena ni cuenta en el acierto, y no tiene tipo.',
                                'Look at it on the orthophoto (or with your field data), in '
                                'the whole yellow square and not only at the point: the Random '
                                'Forest map learns from that 10 m pixel, and the point detector '
                                'averages about 50 m around it.\n\n'
                                'Yes or No leave it verified. Not sure leaves it as doubtful: it '
                                'does not train or count in the accuracy, and has no type.')
                    # (25-11-2026, la autora: «habría que indicar en algún sitio de la
                    # revisión que deben mirar lo que hay dentro del píxel, no solamente
                    # el punto»): entre la pregunta y los botones, en amarillo (la demo).
                    st.html('<div class="rev-nota">' + i18n.pick(
                        'Decide por lo que hay <b>dentro del recuadro amarillo</b> (el píxel de '
                        '10 m), no solo por el punto. Si el píxel es casi todo roca o agua, '
                        'dilo en la pureza o marca «No estoy seguro/a».',
                        'Decide by what is <b>inside the yellow square</b> (the 10 m pixel), '
                        'not only by the point. If the pixel is almost all rock or water, say so '
                        'in the purity or choose "Not sure".') + '</div>')
                    _es = st.radio(
                        i18n.pick('¿Es borreguil?', 'Is it a borreguil?'),
                        list(_ES_TXT), horizontal=True, format_func=_ES_TXT.get,
                        index=list(_ES_TXT).index(_es_ini) if _es_ini else None,
                        key=f'rev_es_{_firma}', label_visibility='collapsed')
                    st.radio(i18n.pick('¿Cuánta certeza tienes? *(opcional)*',
                                       'How sure are you? *(optional)*'),
                             list(bp.CERTIDUMBRES), index=0, horizontal=True,
                             format_func=_CERT_TXT.get, key='rev_certidumbre',
                             help=i18n.pick(
                                 'Lo segura que es la respuesta.\n\n'
                                 'Alta: no hay duda.  \nMedia: hay alguna duda.  \n'
                                 'Baja: es lo más probable, pero hay bastante duda.\n\n'
                                 'Si no sabes decir si es borreguil, no lo verifiques. Si no '
                                 'eliges ninguna, se queda sin indicar.',
                                 'How sure the answer is.\n\n'
                                 'High: no doubt.  \nMedium: some doubt.  \n'
                                 'Low: it is the most likely answer, but there is quite a lot '
                                 'of doubt.\n\n'
                                 'If you cannot tell whether it is a borreguil, do not verify '
                                 'it. If you choose none, it is left unstated.'),
                             disabled=_es == 'duda')
                    _nom, _per = _quien()
                    _cert = '' if _es == 'duda' else (st.session_state.get('rev_certidumbre')
                                                      or '')
                    # El tipo, solo si es borreguil (la demo): cada nivel con la propuesta
                    # de la app marcada «(propuesta)» y, al final, «No lo he podido ver»
                    # (no cuenta como revisado).
                    _SIN = '(sin cambio)'
                    _uno = rows[_sel_ok[0]] if _n == 1 else None
                    _ORDEN_TIPO = {'ambiente': ('laguna', 'arroyo', 'ladera'),
                                   'humedad': ('húmedo', 'seco'),
                                   'pureza': ('puro', 'mixto-agua', 'mixto-roca')}
                    _NOMBRE_TIPO = {'laguna': ('Laguna', 'Lake'), 'arroyo': ('Arroyo', 'Stream'),
                                    'ladera': ('Ladera', 'Slope'), 'húmedo': ('Húmedo', 'Wet'),
                                    'seco': ('Seco', 'Dry'), 'puro': ('Puro', 'Pure'),
                                    'mixto-agua': ('Mixto con agua', 'Mixed with water'),
                                    'mixto-roca': ('Mixto con roca', 'Mixed with rock')}
                    if _es == 'si':
                        with st.container(key='rev_tipo'):
                            st.html('<div class="rev-tipo-tit">' + i18n.pick(
                                'Tipo <span>(lo propone la app; cámbialo si no lo ves '
                                'así)</span>',
                                'Type <span>(proposed by the app; change it if you do not see '
                                'it that way)</span>') + '</div>')
                            for _niv, _etq in (
                                    ('ambiente', i18n.pick('Ambiente', 'Environment')),
                                    ('humedad', i18n.pick('Humedad', 'Moisture')),
                                    ('pureza', i18n.pick('Pureza del píxel', 'Pixel purity'))):
                                _ops = ([v for v in _ORDEN_TIPO[_niv]
                                         if v in bp.NIVELES_TIPO[_niv]]
                                        + [v for v in bp.NIVELES_TIPO[_niv]
                                           if v not in _ORDEN_TIPO[_niv]] + [_SIN])
                                # Con un punto: lo revisado o, si no, la propuesta de la app,
                                # para aceptarla o corregirla. Con varios, «No lo he podido
                                # ver»: cada uno se queda con lo suyo.
                                _rev = (_uno.get(_niv) if _uno
                                        and _niv in bp.niveles_revisados(_uno) else None)
                                _prop = bp._txt(_uno.get(_niv + '_regla')) if _uno else ''
                                _ini = (_rev if _rev in _ops else
                                        (_prop if _prop in _ops else _SIN))
                                st.selectbox(
                                    _etq, _ops, index=_ops.index(_ini),
                                    key=f'rev_{_niv}_{_firma}',
                                    # (un dict hecho ya, como en la lista)
                                    format_func={o: (
                                        i18n.pick('No lo he podido ver', "I couldn't see it")
                                        if o == _SIN else
                                        i18n.pick(*_NOMBRE_TIPO.get(o, (i18n.tv(o),) * 2))
                                        + (i18n.pick(' (propuesta)', ' (proposal)')
                                           if o == _prop else '')) for o in _ops}.get)
                    elif _es == 'duda':
                        st.html('<div class="rev-dudoso">' + i18n.pick(
                            'Queda como dudoso: no se usa para entrenar.',
                            'It stays as doubtful: it is not used for training.') + '</div>')
                    _snap = st.session_state.get('_rev_deshacer')
                    with st.container(horizontal=True, vertical_alignment='center',
                                      key='rev_fila_guardar'):
                        _guardar = st.button(
                            i18n.pick('Guardar y siguiente punto', 'Save and next point'),
                            type='primary', key='rev_guardar', width='stretch')
                        ayuda_q('q_rev_guardar',
                                'Guarda a la vez si es borreguil, la certidumbre, quién revisa '
                                'y el tipo. Si has marcado «Sumar mis puntos a los del modelo y '
                                'reentrenarlo» (en el panel de Analizar, en «Modelo») y cambia '
                                'el sí o el no, el modelo se reentrena.',
                                'Saves at once whether it is a borreguil, the certainty, who '
                                'reviews and the type. If you ticked "Add my points to the '
                                'model\'s and retrain it" (in the Analyse panel, under "Model") '
                                'and the yes or no changes, the model is retrained.')
                    _b1, _b2 = st.columns(2, gap='small')
                    with _b1, st.container(horizontal=True, vertical_alignment='center',
                                           key='rev_fila_deshacer'):
                        _deshacer = st.button(
                            i18n.pick('Deshacer', 'Undo'), key='rev_deshacer',
                            width='stretch', disabled=not _snap)
                        ayuda_q('q_rev_deshacer', 'Deshace la última revisión guardada.',
                                'Undoes the last saved review.')
                    with _b2, st.container(horizontal=True, vertical_alignment='center',
                                           key='rev_fila_quitar'):
                        _quitar = st.button(i18n.pick('Quitar del análisis',
                                                      'Remove from the analysis'),
                                            key='remove_pts', width='stretch')
                        ayuda_q('q_rev_quitar',
                                'Elimina los puntos seleccionados del mapa, la tabla y las '
                                'descargas (no reentrena el modelo).',
                                'Removes the selected points from the map, the table and the '
                                'downloads (it does not retrain the model).')
                    if _quitar:
                        drop = set(_sel_ok)
                        keep = [r for i, r in enumerate(rows) if i not in drop]
                        if not keep:
                            st.warning(i18n.pick('No puedes quitar todos los puntos.',
                                                 'You cannot remove all the points.'))
                        else:
                            bp.save_csv(keep, work / 'classification.csv')
                            bp.save_points_geojson(keep, work / 'puntos.geojson')
                            bp.save_xlsx(keep, work / 'Clasificacion_puntos.xlsx',
                                         threshold=threshold)
                            bp.save_map(keep, work / 'mapa.html', study_geom=sg)
                            st.session_state.rows = keep
                            st.session_state.sel_point = None
                            st.session_state.sel_idx = None
                            st.session_state.prev_sel_set = set()
                            st.session_state['_vaciar_seleccion'] = True
                            # se abre el siguiente de la lista
                            _sig = _siguiente()
                            if _sig in sel_ids:
                                st.session_state.pop('rev_punto_id', None)
                            else:
                                st.session_state['rev_punto_id'] = _sig
                            st.session_state['_rev_msg'] = _msg_puntos(
                                sel_ids, 'quitado del análisis', 'quitados del análisis',
                                'removed from the analysis', 'removed from the analysis')
                            st.rerun()

                    if _guardar and _es is None:
                        st.warning(i18n.pick('Elige primero si es borreguil.',
                                             'First choose whether it is a borreguil.'))
                    elif _guardar:
                        _elegidos = {}
                        if _es == 'si':
                            _elegidos = {n: st.session_state.get(f'rev_{n}_{_firma}')
                                         for n in bp.NIVELES_TIPO}
                            _elegidos = {n: v for n, v in _elegidos.items()
                                         if v and v != _SIN}
                        rows2 = [dict(r) for r in rows]
                        sset = set(sel_ids)
                        _errores, _sin_tipo = [], 0
                        for r in rows2:
                            if r.get('ID') not in sset:
                                continue
                            if _es == 'si':
                                r['Borreguil'] = 'si'; r['origin'] = 'verificado'
                                # verificarlo resuelve una duda de campo: si siguiera
                                # «dudoso», no entraría en el entrenamiento
                                r['Duda'] = ''
                                bp.anotar_revision(r, _nom, _per, _cert)
                                if _elegidos:
                                    try:
                                        bp.set_niveles_revisados(r, **_elegidos)
                                        # aquí se revisa con la ortofoto (plantilla:
                                        # `method`); lo que traía el fichero se respeta
                                        if not bp._txt(r.get('method')):
                                            r['method'] = 'orthophoto'
                                    except ValueError as _e:
                                        _errores.append(f"{r.get('ID', '?')}: {_e}")
                            else:
                                # no borreguil (verificado) o dudoso: sin tipo revisado
                                r['Borreguil'] = 'no' if _es == 'no' else ''
                                r['Duda'] = 'si' if _es == 'duda' else ''
                                r['origin'] = 'verificado' if _es == 'no' else 'dudoso'
                                bp.anotar_revision(r, _nom, _per, _cert)
                                if bp.tipo_revisado(r):
                                    bp.clear_tipo_revisado(r)
                                    _sin_tipo += 1
                        if _errores:
                            # Todo o nada: con un solo punto inválido no se guarda ninguno
                            st.error(i18n.pick('No se ha guardado nada. ',
                                               'Nothing was saved. ')
                                     + ' · '.join(_errores[:5]))
                        else:
                            # Para «Deshacer»: cómo estaba todo antes de guardar (y qué
                            # puntos se revisaron, para volver a abrirlos)
                            st.session_state['_rev_deshacer'] = {
                                'rows': [dict(r) for r in rows],
                                'base': list(base_truth_ids),
                                'ids': list(sel_ids),
                                'hechos': set(_hechos_rev['ids'])}
                            _hechos_rev['ids'] = set(_hechos_rev['ids']) | sset
                            _cambia = any(_estado(rows[i]) != _es for i in _sel_ok)
                            info2 = None
                            if _cambia:
                                with st.spinner(_espera):
                                    _base = set(base_truth_ids)
                                    _nb = (_base | sset if _es == 'si' else _base - sset)
                                    info2 = _recalcular_y_guardar(
                                        rows2, _nb, {'si': 'verdad-terreno',
                                                     'no': 'ausencias verificadas',
                                                     'duda': 'dudosos'}[_es]
                                        + f' (+{len(sset)})', len(sset))
                            else:
                                # el sí o el no no cambia: solo el tipo, sin reentrenar
                                bp.apply_typology(rows2, pp.is_borreguil_decision)
                                bp.save_csv(rows2, work / 'classification.csv')
                                bp.save_points_geojson(rows2, work / 'puntos.geojson')
                                bp.save_xlsx(rows2, work / 'Clasificacion_puntos.xlsx',
                                             threshold=threshold)
                                bp.save_map(rows2, work / 'mapa.html', study_geom=sg)
                                st.session_state.rows = rows2
                            # Y al siguiente punto (la demo: «Guardar y siguiente punto»);
                            # lo que estaba marcado en la tabla se desmarca.
                            st.session_state['rev_punto_id'] = _siguiente()
                            if _de_tabla:
                                st.session_state['_vaciar_seleccion'] = True
                                st.session_state.prev_sel_set = set()
                            st.session_state.sel_point = None
                            st.session_state['_rev_n'] = st.session_state.get('_rev_n', 0) + 1
                            _que = {
                                'si': ('marcado como borreguil verificado',
                                       'marcados como borreguil verificado',
                                       'marked as verified borreguil',
                                       'marked as verified borreguil'),
                                'no': ('marcado como no borreguil verificado',
                                       'marcados como no borreguil verificado',
                                       'marked as verified non-borreguil',
                                       'marked as verified non-borreguil'),
                                'duda': ('marcado como dudoso (no entrena)',
                                         'marcados como dudosos (no entrenan)',
                                         'marked as doubtful (it does not train)',
                                         'marked as doubtful (they do not train)')}[_es]
                            _extra = ''
                            if _elegidos:
                                _extra = i18n.pick(
                                    ' Tipo revisado: ' + ', '.join(_elegidos) + '.',
                                    ' Type reviewed: ' + ', '.join(
                                        bp.COLUMNA_TIPO_EN[n] for n in _elegidos) + '.')
                            elif _sin_tipo:
                                _extra = (i18n.pick(' Se le ha quitado el tipo revisado.',
                                                    ' Its reviewed type was removed.')
                                          if len(sel_ids) == 1 else
                                          i18n.pick(' Se les ha quitado el tipo revisado.',
                                                    ' Their reviewed type was removed.'))
                            st.session_state['_rev_msg'] = (
                                _msg_puntos(sel_ids, *_que) + _extra
                                + (_reentrenado if (info2 or {}).get('sumado') else ''))
                            st.rerun()
                    if _deshacer and _snap:
                        st.session_state.pop('_rev_deshacer', None)
                        with st.spinner(_espera):
                            _recalcular_y_guardar([dict(r) for r in _snap['rows']],
                                                  set(_snap['base']), 'deshacer', 0)
                        if 'hechos' in _snap:
                            _hechos_rev['ids'] = set(_snap['hechos'])
                        # vuelve a abrirse el punto que se había revisado
                        if _snap.get('ids'):
                            st.session_state['rev_punto_id'] = _snap['ids'][0]
                            st.session_state['_vaciar_seleccion'] = True
                            st.session_state.prev_sel_set = set()
                        st.session_state.sel_point = None
                        st.session_state['_rev_n'] = st.session_state.get('_rev_n', 0) + 1
                        st.session_state['_rev_msg'] = i18n.pick(
                            'Revisión deshecha: los puntos vuelven a como estaban.',
                            'Review undone: the points are back to how they were.')
                        st.rerun()
                    st.html('<div style="font-size:12px;color:#98A2AB">' + i18n.pick(
                        'Lo que revises cuenta para medir el acierto y, si lo pides, para '
                        'reentrenar el modelo. Solo se reentrena si cambia el sí o el no.',
                        'What you review counts to measure the accuracy and, if you ask for it, '
                        'to retrain the model. It is only retrained if the yes or no changes.')
                        + '</div>')

        # ================================================================
        # Vistas «Tabla», «Tipos de borreguil», «En qué se fija el modelo» y
        # «Descargas», como en la demo (27-11-2026)
        # ================================================================
        import altair as alt
        import numpy as _np
        import textwrap as _tw

        def _n_mas(v):
            """Con signo y 3 decimales: +0,123 / −0,123."""
            return _num(v, signo=True).replace('-', '−')

        # Los rótulos largos de la demo (con las fechas: memoria «textos de las
        # variables»); los demás, los de siempre.
        _ETQ_LARGA = {
            'ndvi_drop': ('Caída del NDVI, del inicio de verano (15 jun – 25 jul) al fin de '
                          'verano (1 ago – 30 sep)',
                          'NDVI drop, from early summer (15 Jun – 25 Jul) to late summer '
                          '(1 Aug – 30 Sep)'),
            'ndmi_early': ('NDMI inicio de verano (15 jun – 25 jul)',
                           'NDMI early summer (15 Jun – 25 Jul)'),
            'ndmi_late': ('NDMI fin de verano (1 ago – 30 sep)',
                          'NDMI late summer (1 Aug – 30 Sep)'),
        }

        def _etq_larga(f):
            return i18n.pick(*_ETQ_LARGA[f]) if f in _ETQ_LARGA else pp.etiqueta(f)

        with tab_tabla:
            # Solo para mirar (la que se edita está en «2 · Revisar»), con las columnas de
            # la demo: las 3 variables que más pesan llevan su peso en el nombre.
            _rs_t = load_ref_stats(MODEL_PATH)
            _clave_t = pp.variables_clave((_rs_t or {}).get('importances'),
                                          set((_rs_t or {}).get('model_features') or [])
                                          or None, 3)
            _ids_t = [r.get('ID') for r in rows]
            _org = {'truth': i18n.pick('campo', 'field'), 'mapa': i18n.pick('mapa', 'map')}

            def _rev_t(r):
                nv = bp.niveles_revisados(r)
                if len(nv) == len(bp.NIVELES_TIPO):
                    return i18n.pick('sí', 'yes')
                return ', '.join(i18n.pick(n, bp.COLUMNA_TIPO_EN[n]) for n in nv) or '—'

            _col_var = {}
            for k in _clave_t:
                _col_var[k['feat']] = (_nombre_columna(k['feat'])
                                       + ('' if k['peso'] is None
                                          else f" ({100 * k['peso']:.0f} %)"))
            _extra_t = [f for f in ('slope_deg', 'dist_cauce_m') if f not in _col_var]
            _filas_t = []
            for r, nom in zip(rows, bp.nombres_de_punto(_ids_t, i18n.pick('Punto', 'Point'))):
                d = {i18n.pick('Punto', 'Point'): nom,
                     i18n.pick('Origen', 'Origin'): _org.get(
                         bp._txt(r.get('source')), i18n.pick('tu fichero', 'your file')),
                     _nombre_columna('Borreguil'): {'si': i18n.pick('sí', 'yes'),
                                                    'no': 'no'}.get(
                         bp._txt(r.get('Borreguil')).lower(), ''),
                     _nombre_columna('decision'): i18n.tv_decision(
                         r.get('decision') or 'SIN PREDICCIÓN'),
                     _nombre_columna('rf_proba'): pp.fnum(r.get('rf_proba'))}
                for c in ('ambiente', 'humedad', 'pureza'):
                    d[_nombre_columna(c)] = i18n.tv(bp._txt(r.get(c))) or '—'
                d[_nombre_columna('tipo_revisado')] = _rev_t(r)
                for f, nombre in _col_var.items():
                    d[nombre] = pp.fnum(r.get(f))
                for f in _extra_t:
                    d[_nombre_columna(f)] = pp.fnum(r.get(f))
                d['lat'], d['lon'] = pp.fnum(r.get('lat')), pp.fnum(r.get('lon'))
                _filas_t.append(d)
            df_t = pd.DataFrame(_filas_t)
            _c_res, _c_pro = _nombre_columna('decision'), _nombre_columna('rf_proba')
            _si_t = [_es_borreguil(r) for r in rows]
            _sty = _con_coma(df_t, decimales={'lat': 5, 'lon': 5})
            from pandas.io.formats.style import Styler as _Styler
            if not isinstance(_sty, _Styler):
                _sty = df_t.style
            _sty = _sty.apply(lambda s: [
                f"color:{'#A5D6A7' if si else '#F5A3A1'};font-weight:600" for si in _si_t],
                subset=[_c_res]).apply(lambda s: ['color:#F5F7F8;font-weight:700'] * len(s),
                                       subset=[_c_pro])
            # Al pasar el ratón por el nombre de cada columna, se explica (las ayudas de
            # la tabla de «2 · Revisar», si están; las 3 variables, su peso).
            _ay_rev = globals().get('_ayudas')
            _ay_rev = _ay_rev if isinstance(_ay_rev, dict) else {}
            _cfg_t = {}
            for _k_int in ('Borreguil', 'decision', 'rf_proba', 'ambiente', 'humedad',
                           'pureza', 'tipo_revisado', 'slope_deg', 'dist_cauce_m', 'lat',
                           'lon'):
                _c = _nombre_columna(_k_int) if _k_int not in ('lat', 'lon') else _k_int
                if _c in df_t.columns and _k_int in _ay_rev:
                    _cfg_t[_c] = st.column_config.Column(help=i18n.pick(*_ay_rev[_k_int]))
            _cfg_t[i18n.pick('Origen', 'Origin')] = st.column_config.Column(help=i18n.pick(
                'De dónde viene el punto: tu fichero, tu fichero de puntos de campo («campo») '
                'o pinchando en el mapa («mapa»).',
                'Where the point comes from: your file, your field-points file ("field") or '
                'clicking on the map ("map").'))
            _cfg_t[i18n.pick('Punto', 'Point')] = st.column_config.Column(pinned=True)
            for k in _clave_t:
                _cfg_t[_col_var[k['feat']]] = st.column_config.Column(help=i18n.pick(
                    'Una de las 3 variables que más pesan en el modelo, con el peso de su '
                    'grupo de variables parecidas.',
                    'One of the 3 variables that weigh most in the model, with the weight '
                    'of its group of similar variables.'))
            with st.container(key='res_tarjeta_tabla', gap='small'):
                _ev_t = st.dataframe(_sty, hide_index=True, width='stretch',
                                     on_select='rerun', selection_mode='multi-row',
                                     column_config=_cfg_t, key='tabla_resultados')
                try:
                    _sel_t = list(_ev_t.selection.rows)
                except Exception:                           # noqa: BLE001
                    _sel_t = []
                with st.container(gap=None):
                    st.caption(i18n.pick(
                        'La probabilidad es la de ser borreguil según el modelo, de 0 a 1 '
                        f'(por encima de {_num(bp.corte_borreguil(threshold))}, borreguil). '
                        'Por qué estas variables: son las 3 que más pesan en el modelo que '
                        'usas, una por cada grupo de variables que dicen casi lo mismo (el '
                        'peso es el del grupo). La pendiente está siempre, como referencia '
                        'del terreno. Al pasar el ratón por el nombre de cada columna, se '
                        'explica. Selecciona filas y pulsa Revisar.',
                        'The probability is that of being a borreguil according to the '
                        f'model, from 0 to 1 (above {_num(bp.corte_borreguil(threshold))}, '
                        'borreguil). Why these variables: they are the 3 that weigh most in '
                        'the model you use, one for each group of variables that say almost '
                        'the same thing (the weight is the group\'s). The slope is always '
                        'there, as a reference for the terrain. Hover over the name of each '
                        'column to see what it is. Select rows and press Review.'))
                    st.button(i18n.pick('Revisar', 'Review'), key='res_revisar_tabla',
                              type='tertiary', on_click=_revisar_puntos, args=(_sel_t,))

        with tab3:
            # Dos tarjetas: el reparto de las probabilidades y, al lado, los puntos por
            # resultado y los tipos de borreguil.
            _cut3 = bp.corte_borreguil(threshold)
            rf_vals = [float(r['rf_proba']) for r in rows
                       if isinstance(r.get('rf_proba'), (int, float))
                       and not _m.isnan(r.get('rf_proba'))]
            _k1, _k2 = st.columns(2, gap='small')
            with _k1, st.container(key='res_tarjeta_reparto', gap='small'):
                st.markdown('<span class="res-t">' + i18n.pick(
                    'Reparto de las probabilidades', 'How the probabilities are spread')
                    + '</span>', unsafe_allow_html=True)
                if rf_vals:
                    # Barras rojas por debajo del corte y verdes por encima (como los
                    # puntos): los tramos de 0,05 se parten en el corte.
                    _bordes = sorted({round(0.05 * i, 6) for i in range(21)}
                                     | {round(_cut3, 6)})
                    _cuenta = _np.histogram(rf_vals, bins=_bordes)[0]
                    _dfh = pd.DataFrame([{
                        'a': a + min(0.004, (b - a) / 4), 'b': b - min(0.004, (b - a) / 4),
                        'n': int(n), 'lado': 'si' if a >= _cut3 - 1e-9 else 'no',
                        'tramo': f'{_num(a)} – {_num(b)}'}
                        for a, b, n in zip(_bordes[:-1], _bordes[1:], _cuenta) if n])
                    _eje_x = alt.Axis(values=[0, 0.25, 0.5, 0.75, 1], title=None, grid=False,
                                      labelExpr="format(datum.value, '.2~f')"
                                      if i18n.get_lang() == 'en' else
                                      "replace(format(datum.value, '.2~f'), '.', ',')")
                    _barras = alt.Chart(_dfh).mark_bar(
                        cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                        x=alt.X('a:Q', scale=alt.Scale(domain=[0, 1]), axis=_eje_x),
                        x2='b:Q', y2=alt.datum(0),
                        y=alt.Y('n:Q', title=None,
                                axis=alt.Axis(tickMinStep=1, format='d', grid=False)),
                        color=alt.Color('lado:N', legend=None, scale=alt.Scale(
                            domain=['no', 'si'], range=[COLOR_NO, COLOR_SI])),
                        tooltip=[alt.Tooltip('tramo:N', title=i18n.pick('Probabilidad',
                                                                         'Probability')),
                                 alt.Tooltip('n:Q', title=i18n.pick('Puntos', 'Points'))])
                    _raya = alt.Chart(pd.DataFrame({'u': [_cut3]})).mark_rule(
                        color='#ECEFF1', strokeDash=[4, 3]).encode(x='u:Q')
                    st.altair_chart((_barras + _raya).properties(
                        height=150, background='transparent').configure_view(stroke=None),
                                    width='stretch')
                    # Una frase según cómo salen
                    _nn = len(rf_vals)
                    _bajo = sum(1 for v in rf_vals if v < _cut3)
                    _cerca = sum(1 for v in rf_vals if abs(v - _cut3) < 0.15)
                    if _bajo == 0:
                        _fr = i18n.pick('Todos por encima del umbral: el modelo los ve '
                                        'todos como borreguil.',
                                        'All above the threshold: the model sees them all '
                                        'as borreguil.')
                    elif _bajo == _nn:
                        _fr = i18n.pick('Todos por debajo del umbral: el modelo no ve '
                                        'ningún borreguil.',
                                        'All below the threshold: the model sees no '
                                        'borreguil.')
                    elif _cerca <= 0.2 * _nn:
                        _fr = i18n.pick('Dos grupos separados: el modelo distingue bien.',
                                        'Two separate groups: the model tells them apart '
                                        'well.')
                    elif _cerca >= 0.5 * _nn:
                        _fr = i18n.pick('Muchos puntos cerca del umbral: el modelo duda; '
                                        'conviene revisarlos y reentrenar.',
                                        'Many points near the threshold: the model is '
                                        'unsure; review them and retrain.')
                    else:
                        _fr = i18n.pick('Hay puntos cerca del umbral: son los primeros que '
                                        'conviene revisar.',
                                        'There are points near the threshold: they are the '
                                        'first ones to review.')
                    st.caption(_fr + i18n.pick(f' Raya del umbral en {_num(_cut3)}.',
                                               f' Threshold line at {_num(_cut3)}.'))
                else:
                    st.caption(i18n.pick('Sin probabilidades del modelo.',
                                         'No model probabilities.'))
            with _k2, st.container(key='res_tarjeta_recuento', gap='small'):
                # Puntos por resultado: una fila por resultado, de más a menos borreguil
                _orden_dec = ['BORREGUIL VERIFICADO', 'BORREGUIL PROBABLE',
                              'POSIBLE BORREGUIL', 'BORREGUIL (auto)', 'DUDOSO (campo)',
                              'INCIERTO', 'NO BORREGUIL', 'NO BORREGUIL VERIFICADO',
                              'SIN PREDICCIÓN']
                _cnt_dec = Counter(r.get('decision') or 'SIN PREDICCIÓN' for r in rows)
                _decs = ([d for d in _orden_dec if d in _cnt_dec]
                         + sorted(d for d in _cnt_dec if d not in _orden_dec))
                st.markdown(
                    '<div style="display:flex;flex-direction:column;gap:8px;font-size:13px">'
                    '<span class="res-t">' + i18n.pick('Puntos por resultado',
                                                       'Points by result') + '</span>'
                    + ''.join('<span style="display:flex;justify-content:space-between">'
                              f'{_hres.escape(i18n.tv_decision(d))}'
                              f'<strong>{_cnt_dec[d]}</strong></span>' for d in _decs)
                    + '</div>', unsafe_allow_html=True)
                # Tipos de borreguil: una barra apilada por nivel, con los colores de la
                # demo (los del mapa, aclarados para el fondo oscuro)
                _COL_TIPO_RES = {'arroyo': '#7CC496', 'laguna': '#4FC3F7',
                                 'ladera': '#C5D94A', 'húmedo': '#7FE0C2', 'seco': '#F5B48A',
                                 'puro': '#7FE0C2', 'mixto-agua': '#4FC3F7',
                                 'mixto-roca': '#C9A86A'}
                _borr = [r for r in rows if pp.is_borreguil_decision(r.get('decision', ''))]
                _html_t = ('<div style="display:flex;flex-direction:column;gap:8px;'
                           'font-size:13px;margin-top:8px"><span class="res-t">'
                           + i18n.pick('Tipos de borreguil', 'Borreguil types') + '</span>')
                if not _borr:
                    _html_t += ('<span class="res-muted">' + i18n.pick(
                        'Ningún punto es borreguil: no hay tipos.',
                        'No point is a borreguil: there are no types.') + '</span>')
                for _niv, _vals in bp.NIVELES_TIPO.items():
                    if not _borr:
                        break
                    _cn = Counter(bp._txt(r.get(_niv)) for r in _borr)
                    _seg = [(v, _cn.get(v, 0), _COL_TIPO_RES[v]) for v in _vals
                            if _cn.get(v)]
                    _sin = sum(n for v, n in _cn.items() if v not in _vals)
                    if _sin:
                        _seg.append((i18n.pick('sin tipo', 'no type'), _sin, '#5E6871'))
                    _tot = sum(n for _, n, _ in _seg) or 1
                    _nombre_niv = i18n.pick(_niv.capitalize(),
                                            bp.COLUMNA_TIPO_EN[_niv].capitalize())
                    _html_t += (
                        '<span style="display:flex;gap:6px;align-items:center">'
                        f'<span style="width:70px;flex:none;color:#AEB6BE">{_nombre_niv}'
                        '</span><span style="flex:1;display:flex;gap:6px">'
                        + ''.join(f'<span title="{_hres.escape(str(i18n.tv(v)))}: {n}" '
                                  f'style="height:12px;width:{100 * n / _tot:.1f}%;'
                                  f'background:{c};border-radius:3px"></span>'
                                  for v, n, c in _seg)
                        + '</span></span><span style="display:flex;flex-wrap:wrap;'
                        'gap:2px 12px;padding-left:76px;font-size:12px;color:#AEB6BE">'
                        + ''.join('<span style="display:flex;align-items:center;gap:5px">'
                                  f'<span style="width:9px;height:9px;border-radius:2px;'
                                  f'background:{c}"></span>'
                                  f'{_hres.escape(str(i18n.tv(v)))} {n}</span>'
                                  for v, n, c in _seg) + '</span>')
                _html_t += '</div>'
                st.markdown(_html_t, unsafe_allow_html=True)
                if _borr:
                    st.caption(i18n.pick(
                        'arroyo verde · laguna azul · ladera amarillo verdoso. Igual para la '
                        'humedad y la pureza.',
                        'stream green · lake blue · slope yellow-green. The same for moisture '
                        'and purity.'))

        with tab4:
            # Dos tarjetas: los ficheros de los puntos y guardar el modelo reentrenado
            _dl1, _dl2 = st.columns(2, gap='small')
            with _dl1, st.container(key='res_tarjeta_descargas', gap='small'):
                st.markdown('<span class="res-t">' + i18n.pick('Los puntos', 'The points')
                            + '</span>', unsafe_allow_html=True)
                for name, etiqueta, que in (
                    ('Clasificacion_puntos.xlsx',
                     i18n.pick('Excel con todos los puntos y sus variables',
                               'Excel with all the points and their variables'),
                     i18n.pick('Para mirar los resultados: cada punto con su probabilidad, '
                               'su resultado y el tipo de borreguil.',
                               'To look at the results: each point with its probability, '
                               'its result and the type of borreguil.')),
                    ('classification.csv',
                     i18n.pick('CSV (para R, Python o QGIS)', 'CSV (for R, Python or QGIS)'),
                     i18n.pick('Todas las variables de cada punto, para analizarlas.',
                               'All the variables of each point, to analyse them.')),
                    ('puntos.geojson',
                     i18n.pick('puntos.geojson (para volver a cargar tu trabajo)',
                               'puntos.geojson (to load your work again)'),
                     i18n.pick('Los puntos con todo lo revisado: ábrelo en QGIS, o vuelve a '
                               'subirlo a la app otro día para seguir donde lo dejaste.',
                               'The points with everything reviewed: open it in QGIS, or '
                               'upload it to the app again another day to carry on.')),
                    ('mapa.html',
                     i18n.pick('Mapa para abrir sin la app (HTML)',
                               'Map to open without the app (HTML)'),
                     i18n.pick('El mapa de los puntos, para abrirlo en el navegador.',
                               'The map of the points, to open in the browser.')),
                    ('area_estudio.geojson',
                     i18n.pick('Área de estudio (GeoJSON, para QGIS)',
                               'Study area (GeoJSON, for QGIS)'),
                     i18n.pick('El área de estudio que usaste.',
                               'The study area you used.')),
                ):
                    p = work / name
                    if p.exists():
                        with open(p, 'rb') as fh:
                            st.download_button(label=etiqueta, data=fh.read(), file_name=name,
                                               mime='application/octet-stream',
                                               key=f'dl_{name}', type='tertiary', help=que)

            # --- Guardar el modelo reentrenado: solo cuando se ha reentrenado sumando
            # los puntos de la sesión (el de partida ya está en el desplegable y
            # guardarlo otra vez solo haría una copia); si no, se ve pero no se puede.
            bundle = st.session_state.get('model_bundle')
            _info_ult = (hist[-1].get('info') if hist else None) or st.session_state.get('info') or {}
            _saved_msg = st.session_state.pop('_model_saved_msg', None)
            _puede_guardar = bool(bundle) and _info_ult.get('mode') == 'trained'
            with _dl2, st.container(key='res_tarjeta_guardar', gap='small'):
                st.markdown('<span class="res-t">' + i18n.pick(
                    'Guardar el modelo reentrenado', 'Save the retrained model')
                    + '</span>', unsafe_allow_html=True)
                # Mensaje PERSISTENTE: sobrevive al st.rerun que sigue a guardar.
                if _saved_msg:
                    st.success(_saved_msg)
                st.caption(i18n.pick(
                    'Solo si has sumado tus puntos al modelo. Lo podrás elegir otro día en '
                    '«Modelo».',
                    'Only if you have added your points to the model. You will be able to '
                    'choose it another day in "Model".'))
                # Con el nombre del área de estudio, sin sufijos largos tipo «, Comarca
                # de…, Andalucía, España»
                _si = st.session_state.get('study_info') or {}
                area_default = (_si.get('name') or _si.get('query') or '').strip()
                if ',' in area_default:
                    area_default = area_default.split(',')[0].strip()
                area_name = st.text_input(
                    i18n.pick('Nombre de la zona', 'Name of the area'), value=area_default,
                    placeholder=i18n.pick('Por ejemplo, Picos de Europa',
                                          'For example, Picos de Europa'),
                    disabled=not _puede_guardar)
                area_name = area_name.strip() or i18n.pick('Mi zona de montaña',
                                                           'My mountain area')
                fname = area_to_filename(area_name)
                out_path = DATA_DIR / f'{fname}.joblib'
                if _puede_guardar and out_path.exists():
                    st.caption(i18n.pick(
                        f'Ya hay un modelo llamado **{fname}**: se sustituye.',
                        f'There is already a model called **{fname}**: it is replaced.'))
                buf_bytes = b''
                b_save = None
                if _puede_guardar:
                    import io as _io, joblib as _jl
                    b_save = dict(bundle); b_save['training_zone'] = area_name
                    buf = _io.BytesIO()
                    try:
                        _jl.dump(b_save, buf)
                        buf_bytes = buf.getvalue()
                    except Exception as _je:
                        st.error(i18n.pick(f'No se pudo preparar el modelo: {_je}',
                                           f'The model could not be prepared: {_je}'))
                with st.container(horizontal=True, gap='small'):
                    if st.button(i18n.pick('Guardar en la app', 'Save in the app'),
                                 key='guardar_modelo', disabled=not _puede_guardar):
                        try:
                            bp.save_model_bundle(b_save, out_path)
                            if out_path.exists() and out_path.stat().st_size > 0:
                                st.session_state['_model_saved_msg'] = i18n.pick(
                                    f'✓ Modelo guardado: **{fname}**. Ya aparece en '
                                    '«Modelo».',
                                    f'✓ Model saved: **{fname}**. It is already in "Model".')
                                st.rerun()
                            else:
                                st.error(i18n.pick(
                                    'No se pudo guardar. Usa «Descargar el modelo».',
                                    'It could not be saved. Use "Download the model".'))
                        except Exception as e:
                            st.error(i18n.pick(
                                f'No se pudo guardar ({e}). Usa «Descargar el modelo».',
                                f'It could not be saved ({e}). Use "Download the model".'))
                    st.download_button(
                        i18n.pick('Descargar el modelo', 'Download the model'),
                        data=buf_bytes, file_name=f'{fname}.joblib',
                        mime='application/octet-stream', key='dl_model',
                        disabled=not (_puede_guardar and buf_bytes))

        with tab5:
            # Prioriza el modelo recién reentrenado (en sesión); si no, el de partida.
            _mb = st.session_state.get('model_bundle')
            rs_imp = None
            if _mb and _mb.get('importances'):
                rs_imp = dict(_mb.get('ref_stats') or {})
                rs_imp['importances'] = _mb['importances']
            if not (rs_imp and rs_imp.get('importances')):
                rs_imp = load_ref_stats(MODEL_PATH)
            _imp_box = (rs_imp or {}).get('importances')
            _disp_box = set((rs_imp or {}).get('model_features') or []) or None
            with st.container(key='res_tarjeta_peso', gap='small'):
                # --- Lo que más pesa: 12 barras, una por grupo de variables parecidas
                _top = pp.variables_clave(_imp_box, _disp_box, 12) if _imp_box else []
                st.markdown('<span class="res-t15">' + i18n.pick(
                    'En qué se fija el modelo: lo que más pesa',
                    'What the model looks at: what weighs most') + '</span>',
                    unsafe_allow_html=True)
                if _top:
                    _zona_txt = _tit_m
                    st.markdown('<span class="res-ej">' + _hres.escape(i18n.pick(
                        f'Las {len(_top)} variables que más pesan en el modelo que usas '
                        f'({_zona_txt}), una por cada grupo de variables que dicen casi lo '
                        'mismo, con el peso del grupo. Las de satélite se calculan sin nubes '
                        'ni sombras de nube (y, las de junio a septiembre, sin nieve): el '
                        'detalle, en el «?» de los años, en Analizar.',
                        f'The {len(_top)} variables that weigh most in the model you use '
                        f'({_zona_txt}), one for each group of variables that say almost '
                        "the same thing, with the group's weight. The satellite ones are "
                        'computed without clouds or cloud shadows (and, those from June to '
                        'September, without snow): the details, in the "?" of the years, in '
                        'Analyse.')) + '</span>', unsafe_allow_html=True)
                    _max_p = max(k['peso'] for k in _top) or 1
                    st.markdown(
                        '<div style="display:flex;flex-direction:column;gap:7px;'
                        'font-size:13px">'
                        + ''.join(
                            '<span style="display:flex;gap:10px;align-items:center">'
                            '<span style="width:300px;flex:none;line-height:1.3;'
                            f'color:#ECEFF1">{_hres.escape(_etq_larga(k["feat"]))}</span>'
                            '<span style="flex:1;display:flex;align-items:center;gap:8px">'
                            '<span style="display:block;flex:none;height:12px;'
                            f'width:{85 * k["peso"] / _max_p:.1f}%;background:#7CC496;'
                            'border-radius:3px"></span><span style="font-size:12px;'
                            f'color:#AEB6BE">{100 * k["peso"]:.0f} %</span></span></span>'
                            for k in _top) + '</div>', unsafe_allow_html=True)
                else:
                    st.info(i18n.pick(
                        'Este modelo no trae el peso de sus variables, así que no se puede '
                        'mostrar.',
                        'This model does not carry the weight of its variables, so it '
                        'cannot be shown.'))

                # --- Las que más pesan, grupo a grupo: cuatro maneras de verlas,
                # comparando borreguil con no borreguil o los borreguiles por su humedad
                # o su ambiente con los no borreguiles (25-11-2026, la autora), con los
                # controles y los gráficos de la demo (27-11-2026).
                st.markdown('<div style="border-top:1px solid #403A3C;margin-top:4px;'
                            'padding-top:12px"><span class="res-t15">' + i18n.pick(
                                'Las que más pesan, grupo a grupo',
                                'The ones that weigh most, group by group')
                            + '</span></div>', unsafe_allow_html=True)
                _vistas = {'cajas': i18n.pick('Cajas', 'Boxes'),
                           'curvas': i18n.pick('Curvas de distribución',
                                               'Distribution curves'),
                           'diferencias': i18n.pick('Diferencia estandarizada de medianas',
                                                    'Standardized median difference'),
                           'dos': i18n.pick('Dos variables a la vez', 'Two variables at once')}
                # (El grupo con el que se compara es «No borreguil» y tiene solo eso; antes
                # se llamaba «Resto» y llevaba también los inciertos y los dudosos.
                # 25-11-2026, la autora: «¿el resto aquí quiere decir no borreguil?».)
                _comp_txt = {
                    'borreguil': i18n.pick('Borreguil frente a no borreguil',
                                           'Borreguil versus not a borreguil'),
                    'humedad': i18n.pick('Por humedad (húmedo, seco y no borreguil)',
                                         'By moisture (wet, dry and not a borreguil)'),
                    'ambiente': i18n.pick('Por ambiente (arroyo, laguna, ladera y no '
                                          'borreguil)',
                                          'By environment (stream, lake, slope and not a '
                                          'borreguil)')}
                _todas_vb = pp.vars_boxplots(_imp_box, _disp_box, 1000)
                _max_v = max(1, len(_todas_vb))
                if st.session_state.get('dist_n', 0) > _max_v:
                    st.session_state['dist_n'] = _max_v
                with st.container(horizontal=True, vertical_alignment='bottom',
                                  gap='medium'):
                    _vista = st.segmented_control(
                        i18n.pick('Cómo verlo', 'How to see it'), list(_vistas),
                        format_func=_vistas.get, key='dist_vista', default='cajas',
                        required=True, width='content')
                    _comparar = st.segmented_control(
                        i18n.pick('Comparar', 'Compare'), list(pp.COMPARAR),
                        format_func=_comp_txt.get, key='dist_comparar',
                        default='borreguil', required=True, width='content')
                    if _vista != 'dos':
                        _n_vars = st.number_input(
                            i18n.pick(f'Cuántas variables (de 1 a {_max_v})',
                                      f'How many variables (1 to {_max_v})'),
                            min_value=1, max_value=_max_v, value=min(8, _max_v), step=1,
                            key='dist_n', width=200)
                _grp, _ord, _rest = pp.grupos_puntos(rows, _comparar, None, i18n.tv)
                _cnt = Counter(_grp)

                def _cap(g):
                    return g[:1].upper() + g[1:]

                _cuantos = ' · '.join(f'{_cap(g)}: {_cnt[g]}' for g, _ in _ord)
                if _cnt.get(None):
                    _cuantos += i18n.pick(
                        f' (y {_cnt[None]} inciertos o dudosos, que no se dibujan)',
                        f' (and {_cnt[None]} uncertain or doubtful, not drawn)')
                # Relleno y borde de cada grupo (la demo: los colores de cada categoría
                # en la app, aclarados para el fondo oscuro)
                _PAR = {'borreguil': ('#2E8B4A', '#8FD9A8'),
                        'no borreguil': ('#B23B3B', '#F4A3A3'),
                        'not a borreguil': ('#B23B3B', '#F4A3A3'),
                        'húmedo': ('#0E8063', '#7FE0C2'), 'wet': ('#0E8063', '#7FE0C2'),
                        'seco': ('#B4561C', '#F5B48A'), 'dry': ('#B4561C', '#F5B48A'),
                        'arroyo': ('#23801B', '#9BE08F'), 'stream': ('#23801B', '#9BE08F'),
                        'laguna': ('#2D5FA3', '#9CC2F2'), 'lake': ('#2D5FA3', '#9CC2F2'),
                        'ladera': ('#8C9226', '#E2E68E'), 'slope': ('#8C9226', '#E2E68E')}
                _gr = [g for g, _ in _ord]
                _rell = {g: _PAR.get(g.lower(), ('#5E6871', '#AEB6BE'))[0] for g in _gr}
                _bord = {g: _PAR.get(g.lower(), ('#5E6871', '#AEB6BE'))[1] for g in _gr}
                _vals_g = {}               # {(variable, grupo): [valores]}

                def _valores_de(feats):
                    out = {}
                    for r, g in zip(rows, _grp):
                        if g is None:
                            continue
                        for f in feats:
                            v = pp.fnum(r.get(f))
                            if v is not None:
                                out.setdefault((f, g), []).append(v)
                    return out

                def _rotulo(f, lab):
                    """El rótulo del panel: el largo, con el peso del grupo si lo trae
                    («Altitud (m) · 13 %»), partido en líneas («|»)."""
                    peso = lab.split(' · ')[-1] if ' · ' in lab else ''
                    txt = _etq_larga(f) + (f' · {peso}' if peso else '')
                    return '|'.join(_tw.wrap(txt, 34)) or txt

                _eje_num = ("replace(format(datum.value, '~r'), '.', ',')"
                            if i18n.get_lang() != 'en' else "format(datum.value, '~r')")
                _cabecera = alt.Header(title=None, labelExpr="split(datum.label, '|')",
                                       labelAlign='left', labelAnchor='start',
                                       labelFontWeight='bold', labelFontSize=12.5,
                                       labelColor='#F5F7F8', labelLimit=0,
                                       labelPadding=6)
                _ch, _como, _nota_min = None, '', False
                if _vista != 'dos':
                    _vars_box = _todas_vb[:int(_n_vars)]
                    st.caption(i18n.pick(
                        f'Las {len(_vars_box)} variables que más pesan en el modelo, de más a '
                        'menos, con el peso junto al nombre cuando se sabe. Muestras: '
                        f'{_cuantos}.',
                        f'The {len(_vars_box)} variables that weigh most in the model, from '
                        'most to least, with the weight next to the name when it is known. '
                        f'Samples: {_cuantos}.'))
                    _rot = {f: _rotulo(f, lab) for f, lab in _vars_box}
                    _orden_rot = [_rot[f] for f, _ in _vars_box]
                    _vals_g = _valores_de([f for f, _ in _vars_box])
                if _vista == 'cajas':
                    # Cajas verticales, un panel por variable con su escala: la caja, del
                    # 25 % al 75 %; las líneas, del 10 % al 90 %; la mediana, en blanco.
                    _recs = []
                    for f, _lab in _vars_box:
                        for g in _gr:
                            v = _vals_g.get((f, g))
                            if not v:
                                continue
                            q = _np.percentile(v, [10, 25, 50, 75, 90])
                            _recs.append({
                                'variable': _rot[f], 'grupo': g, 'p10': q[0], 'p25': q[1],
                                'p50': q[2], 'p75': q[3], 'p90': q[4],
                                'tt': i18n.pick(
                                    f'{_cap(g)} · mediana {_num(q[2])} · la mitad central, '
                                    f'de {_num(q[1])} a {_num(q[3])} · casi todas (del 10 % '
                                    f'al 90 %), de {_num(q[0])} a {_num(q[4])}',
                                    f'{_cap(g)} · median {_num(q[2])} · the central half, '
                                    f'from {_num(q[1])} to {_num(q[3])} · almost all (10 % '
                                    f'to 90 %), from {_num(q[0])} to {_num(q[4])}')})
                    if _recs:
                        _dfc = pd.DataFrame(_recs)
                        _x = alt.X('grupo:N', sort=_gr, axis=None,
                                   scale=alt.Scale(paddingOuter=0.35, paddingInner=0.3))
                        _y = alt.Y('p10:Q', title=None, scale=alt.Scale(zero=False),
                                   axis=alt.Axis(labelExpr=_eje_num, tickCount=4,
                                                 gridColor='#393335', domainColor='#60575A',
                                                 labelColor='#AEB6BE'))
                        _c_b = alt.Color('grupo:N', legend=None, scale=alt.Scale(
                            domain=_gr, range=[_bord[g] for g in _gr]))
                        _base = alt.Chart()
                        _ch = alt.layer(
                            _base.mark_rule(strokeWidth=2).encode(x=_x, y=_y, y2='p90:Q',
                                                                  color=_c_b),
                            _base.mark_tick(thickness=2, size=14).encode(x=_x, y='p10:Q',
                                                                         color=_c_b),
                            _base.mark_tick(thickness=2, size=14).encode(x=_x, y='p90:Q',
                                                                         color=_c_b),
                            _base.mark_bar(size=26, cornerRadius=4, strokeWidth=2).encode(
                                x=_x, y='p25:Q', y2='p75:Q',
                                fill=alt.Color('grupo:N', legend=None, scale=alt.Scale(
                                    domain=_gr, range=[_rell[g] for g in _gr])),
                                stroke=_c_b, tooltip=[alt.Tooltip('tt:N', title=None)]),
                            _base.mark_tick(thickness=4, size=30, color='#FFFFFF',
                                            stroke='#000000', strokeWidth=0.5).encode(
                                x=_x, y='p50:Q', tooltip=[alt.Tooltip('tt:N', title=None)]),
                            data=_dfc).properties(width=190, height=190).facet(
                            facet=alt.Facet('variable:N', sort=_orden_rot, header=_cabecera),
                            columns=3, spacing=18).resolve_scale(y='independent',
                                                                 x='independent')
                    _como = i18n.pick(
                        'Cada caja va de la mitad central de las muestras de un grupo (del 25 % '
                        'al 75 %), con la mediana en blanco; las líneas llegan del 10 % al '
                        '90 %. Si las cajas de dos grupos no se tocan, esa variable los separa '
                        'bien. Pasa el ratón por una caja para ver sus cifras.',
                        'Each box spans the central half of the samples of a group (25 % to '
                        '75 %), with the median in white; the lines reach from 10 % to 90 %. '
                        'If the boxes of two groups do not touch, that variable separates '
                        'them well. Hover over a box to see its figures.')
                elif _vista == 'curvas':
                    # Una curva por grupo (de densidad, con la misma área), por variable
                    _recs = []
                    for f, _lab in _vars_box:
                        _todos_v = [x for g in _gr for x in _vals_g.get((f, g), [])]
                        if len(_todos_v) < 2:
                            continue
                        lo, hi = min(_todos_v), max(_todos_v)
                        pad = (hi - lo) * 0.08 or 1.0
                        malla = _np.linspace(lo - pad, hi + pad, 80)
                        for g in _gr:
                            v = _np.asarray(_vals_g.get((f, g), []), dtype=float)
                            if len(v) < pp.MIN_GRUPO:
                                _nota_min = _nota_min or len(v) > 0
                                continue
                            sd = float(_np.std(v, ddof=1))
                            iqr = float(_np.subtract(*_np.percentile(v, [75, 25])))
                            esc = min(x for x in (sd, iqr / 1.34) if x > 0) \
                                if (sd > 0 or iqr > 0) else (hi - lo) / 10 or 1.0
                            h = 1.06 * esc * len(v) ** (-0.2)
                            dens = _np.exp(-0.5 * ((malla[:, None] - v[None, :]) / h) ** 2
                                           ).sum(axis=1) / (len(v) * h * (2 * _np.pi) ** 0.5)
                            _recs += [{'variable': _rot[f], 'grupo': g, 'x': float(x),
                                       'd': float(d)} for x, d in zip(malla, dens)]
                    if _recs:
                        _dfd = pd.DataFrame(_recs)
                        _ch = alt.Chart(_dfd).mark_area(
                            fillOpacity=0.22, interpolate='monotone',
                            line={'strokeWidth': 2}).encode(
                            x=alt.X('x:Q', title=None, axis=alt.Axis(
                                labelExpr=_eje_num, tickCount=3, grid=False,
                                domainColor='#60575A', labelColor='#AEB6BE')),
                            y=alt.Y('d:Q', axis=None, stack=None),
                            color=alt.Color('grupo:N', legend=None, scale=alt.Scale(
                                domain=_gr, range=[_bord[g] for g in _gr])),
                            detail='grupo:N').properties(width=330, height=70).facet(
                            facet=alt.Facet('variable:N', sort=_orden_rot, header=_cabecera),
                            columns=2, spacing=18).resolve_scale(x='independent',
                                                                 y='independent')
                    _como = i18n.pick(
                        'Cada curva muestra cómo se reparten los valores de un grupo; todas '
                        'encierran la misma área. Si apenas se tocan, esa variable separa '
                        'bien los grupos.',
                        'Each curve shows how the values of a group are spread; they all '
                        'enclose the same area. If they hardly touch, that variable '
                        'separates the groups well.')
                elif _vista == 'diferencias':
                    # Cuánto se aparta la mediana de cada grupo de borreguil de la de «no
                    # borreguil», en desviaciones típicas (del ancho de las cajas / 1,349);
                    # escala fija de −2,5 a +2,5, con la franja de ±1.
                    _recs, _filas_d = [], []
                    for f, _lab in _vars_box:
                        no = _np.asarray(_vals_g.get((f, _rest), []), dtype=float)
                        _filas_d.append({'variable': _rot[f], 'a': -1.0, 'b': 1.0})
                        if len(no) < pp.MIN_GRUPO:
                            continue
                        for g in _gr:
                            if g == _rest:
                                continue
                            v = _np.asarray(_vals_g.get((f, g), []), dtype=float)
                            if len(v) < pp.MIN_GRUPO:
                                _nota_min = _nota_min or len(v) > 0
                                continue
                            iqr_g = float(_np.subtract(*_np.percentile(v, [75, 25])))
                            iqr_n = float(_np.subtract(*_np.percentile(no, [75, 25])))
                            den = (iqr_g + iqr_n) / 2 / 1.349
                            if den <= 0:
                                continue
                            dz = (float(_np.median(v)) - float(_np.median(no))) / den
                            _recs.append({'variable': _rot[f], 'grupo': g,
                                          'dz': max(-2.5, min(2.5, dz)),
                                          'tt': f'{_cap(g)}: {_n_mas(dz)}'})
                    if _recs:
                        _dfz, _dff = pd.DataFrame(_recs), pd.DataFrame(_filas_d)
                        _yv = alt.Y('variable:N', sort=_orden_rot, title=None,
                                    axis=alt.Axis(labelExpr="split(datum.label, '|')",
                                                  labelLimit=250, labelFontSize=12.5,
                                                  labelColor='#ECEFF1', domain=False,
                                                  ticks=False, labelPadding=262,
                                                  labelAlign='left', labelOverlap=False,
                                                  labelBaseline='middle'))
                        _sx = alt.Scale(domain=[-2.5, 2.5], clamp=True, nice=False)
                        _ax = alt.Axis(values=[-2.5, -1, 0, 1, 2.5], title=None, grid=False,
                                       domainColor='#494245', labelColor='#AEB6BE',
                                       labelLimit=0, labelOverlap=False, labelFontSize=11,
                                       labelExpr=(
                                           "datum.value == 0 ? '" + i18n.pick(
                                               '0 · como no borreguil',
                                               '0 · like not a borreguil')
                                           + "' : (datum.value > 0 ? '+' : '−') + "
                                           + ("replace(format(abs(datum.value), '.1~f'), "
                                              "'.', ',')" if i18n.get_lang() != 'en' else
                                              "format(abs(datum.value), '.1~f')")))
                        _ch = alt.layer(
                            alt.Chart(_dff).mark_bar(size=12, color='#363133',
                                                     cornerRadius=3).encode(
                                y=_yv, x=alt.X('a:Q', scale=_sx, axis=_ax), x2='b:Q'),
                            alt.Chart(pd.DataFrame({'cero': [0.0]})).mark_rule(
                                color='#98A2AB', strokeWidth=2).encode(
                                x=alt.X('cero:Q', scale=_sx, axis=_ax)),
                            alt.Chart(_dfz).mark_circle(size=200, opacity=1,
                                                        strokeWidth=2).encode(
                                y=_yv, x=alt.X('dz:Q', scale=_sx, axis=_ax),
                                fill=alt.Color('grupo:N', legend=None, scale=alt.Scale(
                                    domain=_gr, range=[_rell[g] for g in _gr])),
                                stroke=alt.Color('grupo:N', legend=None, scale=alt.Scale(
                                    domain=_gr, range=[_bord[g] for g in _gr])),
                                tooltip=[alt.Tooltip('tt:N', title=None)]),
                        ).properties(height=alt.Step(52))
                    _como = i18n.pick(
                        'Cada punto dice cuánto se aparta la mediana del grupo de la de «no '
                        'borreguil», en desviaciones típicas de todas las muestras: a la '
                        'derecha, más; a la izquierda, menos. Pasar de 1 (o de −1) es una '
                        'diferencia clara. Pasa el ratón por un punto para ver su valor.',
                        'Each dot says how far the median of the group is from that of '
                        '"not a borreguil", in standard deviations of all the samples: to the '
                        'right, more; to the left, less. Beyond 1 (or −1) is a clear '
                        'difference. Hover over a dot to see its value.')
                else:
                    # Dos variables a la vez: las del modelo, de más a menos peso
                    _lv = pp.lista_variables(rows, _imp_box, todas=[] if _imp_box else None)
                    _op_lv = [f for f, _ in _lv]
                    _top2 = [k['feat'] for k in pp.variables_clave(_imp_box, _disp_box, 2)
                             if k['feat'] in _op_lv]
                    _ix = _op_lv.index(_top2[0]) if _top2 else 0
                    _iy = (_op_lv.index(_top2[1]) if len(_top2) > 1
                           else min(1, len(_op_lv) - 1))
                    if len(_op_lv) >= 2:
                        _d1, _d2 = st.columns(2, gap='small')
                        _vx = _d1.selectbox(i18n.pick('Eje horizontal', 'Horizontal axis'),
                                            _op_lv, index=_ix, format_func=_etq_larga,
                                            key='dist_x')
                        _vy = _d2.selectbox(i18n.pick('Eje vertical', 'Vertical axis'),
                                            _op_lv, index=_iy, format_func=_etq_larga,
                                            key='dist_y')
                        st.caption(i18n.pick(
                            'Las variables del modelo, de más a menos peso; de entrada, las '
                            f'dos que más pesan. Muestras: {_cuantos}.',
                            "The model's variables, from most to least weight; by default, "
                            f'the two that weigh most. Samples: {_cuantos}.'))
                        _nombres_d = bp.nombres_de_punto([r.get('ID') for r in rows],
                                                         i18n.pick('Punto', 'Point'))
                        _recs = []
                        for r, g, nom in zip(rows, _grp, _nombres_d):
                            xv, yv = pp.fnum(r.get(_vx)), pp.fnum(r.get(_vy))
                            if g is None or xv is None or yv is None:
                                continue
                            _recs.append({'x': xv, 'y': yv, 'grupo': g, 'punto': nom,
                                          'gtxt': _cap(g), 'xt': _num(xv), 'yt': _num(yv)})
                        if _recs:
                            _ch = alt.Chart(pd.DataFrame(_recs)).mark_circle(
                                size=110, opacity=0.85).encode(
                                x=alt.X('x:Q', title=_tw.wrap(_etq_larga(_vx), 70),
                                        scale=alt.Scale(
                                    zero=False, nice=True), axis=alt.Axis(
                                    labelExpr=_eje_num, gridColor='#3E383A',
                                    domainColor='#5E6871', labelColor='#AEB6BE')),
                                # (partido en líneas: un título más largo que el gráfico lo
                                # aplastaba)
                                y=alt.Y('y:Q', title=_tw.wrap(_etq_larga(_vy), 42),
                                        scale=alt.Scale(
                                    zero=False, nice=True), axis=alt.Axis(
                                    labelExpr=_eje_num, gridColor='#3E383A',
                                    domainColor='#5E6871', labelColor='#AEB6BE')),
                                color=alt.Color('grupo:N', legend=None, scale=alt.Scale(
                                    domain=_gr, range=[_bord[g] for g in _gr])),
                                tooltip=[alt.Tooltip('punto:N', title=i18n.pick('Punto',
                                                                                 'Point')),
                                         alt.Tooltip('gtxt:N', title=i18n.pick('Grupo',
                                                                                'Group')),
                                         alt.Tooltip('xt:N', title=_etq_larga(_vx)),
                                         alt.Tooltip('yt:N', title=_etq_larga(_vy))]
                            ).properties(height=380)
                    _como = i18n.pick(
                        'Cada punto es una muestra, colocada según las dos variables. Si las '
                        'nubes de colores se separan, combinar esas dos variables ayuda a '
                        'distinguir los grupos. Pasa el ratón por un punto para ver sus '
                        'valores.',
                        'Each dot is a sample, placed by the two variables. If the coloured '
                        'clouds separate, combining those two variables helps to tell the '
                        'groups apart. Hover over a dot to see its values.')
                # La leyenda de los grupos (cuadro con relleno y borde, y cuántos hay)
                st.markdown(
                    '<div style="display:flex;flex-wrap:wrap;gap:8px 18px;font-size:12.5px;'
                    'color:#ECEFF1">' + ''.join(
                        '<span style="display:flex;align-items:center;gap:7px">'
                        '<span style="width:14px;height:14px;border-radius:3px;'
                        f'box-sizing:border-box;background:{_rell[g]};border:2px solid '
                        f'{_bord[g]}"></span>{_hres.escape(_cap(g))} ({_cnt[g]})</span>'
                        for g in _gr) + '</div>', unsafe_allow_html=True)
                if _ch is not None:
                    # Sobre el fondo de la tarjeta; cada panel, en su recuadro (la demo)
                    _ch = _ch.properties(background='transparent').configure_view(
                        fill='#282526' if _vista in ('cajas', 'curvas') else None,
                        stroke='#403A3C' if _vista in ('cajas', 'curvas') else None,
                        cornerRadius=8)
                    st.altair_chart(_ch, width='stretch' if _vista in ('diferencias', 'dos')
                                    else 'content')
                    if _nota_min:
                        _como += i18n.pick(
                            f' Los grupos con menos de {pp.MIN_GRUPO} puntos no se dibujan.',
                            f' Groups with fewer than {pp.MIN_GRUPO} points are not drawn.')
                    st.caption(_como + i18n.pick(
                        ' Valores de tus puntos: borreguil o no según tu revisión o, si no '
                        'está revisado, según el modelo.',
                        ' Values of your points: borreguil or not according to your review '
                        'or, if not reviewed, according to the model.'))
                else:
                    st.info(i18n.pick(
                        'No hay datos suficientes para este gráfico (por ejemplo, sin puntos '
                        'que no sean borreguil no hay con qué comparar).',
                        'There is not enough data for this chart (for instance, without '
                        'points that are not a borreguil there is nothing to compare '
                        'with).'))

    else:
        st.info(i18n.pick(
            'Todavía no hay resultados. Analiza tus puntos en «1 · Analizar», o prueba con '
            'el ejemplo del Inicio.',
            'There are no results yet. Analyse your points in "1 · Analyse", or try the '
            'example on the start screen.'))
        with pant_revisar:
            st.info(i18n.pick(
                'Aquí revisarás los puntos uno a uno cuando los hayas analizado, en '
                '«1 · Analizar».',
                'Here you will review the points one by one once you have analysed them, '
                'in "1 · Analyse".'))

# ============================================================
# MAPA MODELO RF — pestaña propia, al mismo nivel que «Verdad terreno»
# De los puntos con el tipo revisado a un mapa de tipos de borreguil, con Random
# Forest y por pasos: datos, tabla de entrenamiento, modelo, mapa y peso de las
# variables. La lógica está en mapeo_rf.py y mapeo_raster.py; aquí solo se enseña.
# No toca los puntos ni el resto de la app. Va al final del guion porque usa los
# puntos tal como los deja la pestaña «Verdad terreno» en esta misma ejecución.
# ============================================================
# Lo propio de esta pantalla, como en la demo (27-11-2026): título, pasos, tarjetas,
# botones segmentados, avisos y colores. Todo va acotado a sus claves (mapa_…, mrf_…,
# tarjeta_mapa_…), para no tocar las demás pantallas.
_CSS_MAPA = '''
.st-key-mapa_pantalla{gap:16px}
.st-key-mapa_titulo h1{font-family:Montserrat,Inter,system-ui,sans-serif;font-size:28px !important;
  font-weight:700;color:#F5F7F8;padding:0 !important;margin:0 !important;line-height:1.25}
.st-key-mapa_pantalla h1 [data-testid="stHeaderActionElements"],
.st-key-mapa_pantalla h5 [data-testid="stHeaderActionElements"]{display:none}
.st-key-mapa_titulo p{font-size:14px;color:#AEB6BE;margin:0}
.st-key-mapa_pantalla h5{font-family:Inter,system-ui,sans-serif;font-size:15px !important;
  font-weight:700;color:#ECEFF1;padding:0 !important;margin:0 !important}
[class*="st-key-tarjeta_mapa_"]{gap:10px}
[class*="st-key-tarjeta_mapa_"] [data-testid="stMarkdownContainer"] p,
[class*="st-key-tarjeta_mapa_"] [data-testid="stMarkdownContainer"] li{font-size:13px;color:#CDD3D9}
[class*="st-key-tarjeta_mapa_"] [data-testid="stCaptionContainer"] p{font-size:12px;color:#98A2AB;line-height:1.5}
/* Los pasos: cinco botones con su estado, como en la demo; las pestañas, ocultas */
.st-key-mrf_paso>div>[data-baseweb="tab-border"],
.st-key-mrf_paso>div>div>[data-baseweb="tab-list"],
.st-key-mrf_paso>div>div>[data-baseweb="tab-border"]{display:none}
[class*="st-key-mrf_ir_"] button{width:100%;min-height:64px;padding:10px 12px;border-radius:12px;
  border:1px solid #494245;background:#2E2A2B;justify-content:flex-start;align-items:flex-start}
[class*="st-key-mrf_ir_"] button [data-testid="stMarkdownContainer"]{text-align:left;width:100%}
[class*="st-key-mrf_ir_"] button>div,[class*="st-key-mrf_ir_"] button>div>span{width:100%;
  justify-content:flex-start;text-align:left}
.st-key-mapa_pantalla [data-testid="stMarkdownContainer"]:has(h1),
.st-key-mapa_pantalla [data-testid="stMarkdownContainer"]:has(h5){margin-bottom:0 !important}
[class*="st-key-mrf_ir_"] button p{margin:0;font-size:14px;font-weight:600;color:#CDD3D9;
  line-height:1.35;text-align:left}
[class*="st-key-mrf_ir_"] button p:nth-child(2){font-size:14px !important;font-weight:700}
[class*="st-key-mrf_ir_"] button p:first-child{font-size:12px !important;opacity:.8}
[class*="st-key-mrf_ir_"] button p:last-child{font-size:12px !important;font-weight:500;opacity:.85}
[class*="st-key-mrf_ir_"] button:hover{border-color:#7CC496}
/* Opciones cortas como botones segmentados (la demo) */
.st-key-mrf_seleccion [role="radiogroup"],.st-key-mrf_umbral_modo [role="radiogroup"],
.st-key-mrf_peso_medida [role="radiogroup"],.st-key-mrf_peso_por [role="radiogroup"],
.st-key-mrf_dist_vista [role="radiogroup"]{gap:6px;flex-wrap:wrap}
.st-key-mrf_seleccion label[data-baseweb="radio"],.st-key-mrf_umbral_modo label[data-baseweb="radio"],
.st-key-mrf_peso_medida label[data-baseweb="radio"],.st-key-mrf_peso_por label[data-baseweb="radio"],
.st-key-mrf_dist_vista label[data-baseweb="radio"]{flex:1 1 0;min-width:max-content;
  justify-content:center;margin:0;min-height:44px;padding:9px 10px;border:1px solid #494245;
  background:#393335;border-radius:10px}
.st-key-mrf_peso_por label[data-baseweb="radio"],.st-key-mrf_dist_vista label[data-baseweb="radio"],
.st-key-mrf_peso_medida label[data-baseweb="radio"]{flex:0 1 auto}
.st-key-mrf_seleccion label[data-baseweb="radio"]:has(input:checked),
.st-key-mrf_umbral_modo label[data-baseweb="radio"]:has(input:checked),
.st-key-mrf_peso_medida label[data-baseweb="radio"]:has(input:checked),
.st-key-mrf_peso_por label[data-baseweb="radio"]:has(input:checked),
.st-key-mrf_dist_vista label[data-baseweb="radio"]:has(input:checked){
  border:1.5px solid #7CC496;background:#1E3327}
.st-key-mrf_seleccion label[data-baseweb="radio"]>div:first-of-type,
.st-key-mrf_umbral_modo label[data-baseweb="radio"]>div:first-of-type,
.st-key-mrf_peso_medida label[data-baseweb="radio"]>div:first-of-type,
.st-key-mrf_peso_por label[data-baseweb="radio"]>div:first-of-type,
.st-key-mrf_dist_vista label[data-baseweb="radio"]>div:first-of-type{display:none}
.st-key-mrf_seleccion label[data-baseweb="radio"] p,.st-key-mrf_umbral_modo label[data-baseweb="radio"] p,
.st-key-mrf_peso_medida label[data-baseweb="radio"] p,.st-key-mrf_peso_por label[data-baseweb="radio"] p,
.st-key-mrf_dist_vista label[data-baseweb="radio"] p{font-size:13px;font-weight:500;color:#CDD3D9}
.st-key-mrf_seleccion label[data-baseweb="radio"]:has(input:checked) p,
.st-key-mrf_umbral_modo label[data-baseweb="radio"]:has(input:checked) p,
.st-key-mrf_peso_medida label[data-baseweb="radio"]:has(input:checked) p,
.st-key-mrf_peso_por label[data-baseweb="radio"]:has(input:checked) p,
.st-key-mrf_dist_vista label[data-baseweb="radio"]:has(input:checked) p{font-weight:600;color:#ECEFF1}
.st-key-mrf_seleccion [data-testid="stWidgetLabel"] p{font-size:14px;font-weight:700;color:#ECEFF1}
.st-key-mrf_umbral_modo [data-testid="stWidgetLabel"] p{font-size:13px;font-weight:600;color:#ECEFF1}
/* Las variables como fichas (st.pills): marcadas en verde, las demás tachadas */
[class*="st-key-mrf_vars_"] [data-testid="stButtonGroup"]>div{gap:6px}
[class*="st-key-mrf_vars_"] [data-testid="stButtonGroup"] button{border-radius:999px;
  min-height:0;padding:5px 10px;border:1px solid #494245;background:transparent}
[class*="st-key-mrf_vars_"] [data-testid="stButtonGroup"] button p{font-size:13px;
  font-weight:600;color:#7E8891;text-decoration:line-through}
[class*="st-key-mrf_vars_"] [data-testid="stButtonGroup"] button[kind$="Active"]{
  border:1px solid #3E6B4F;background:#1E3327}
[class*="st-key-mrf_vars_"] [data-testid="stButtonGroup"] button[kind$="Active"] p{
  color:#ECEFF1;text-decoration:none}
[class*="st-key-mrf_vars_"] [data-testid="stButtonGroup"] button[kind$="Active"] p::after{
  content:" ✓"}
.st-key-mapa_pantalla [data-testid="stMarkdownContainer"]:has(.mapa-grupo),
.st-key-mapa_pantalla [data-testid="stMarkdownContainer"]:has(.mapa-cifras),
.st-key-mapa_pantalla [data-testid="stMarkdownContainer"]:has(.mapa-f1),
.st-key-mapa_pantalla [data-testid="stMarkdownContainer"]:has(.mapa-pixel){margin-bottom:0 !important}
.st-key-mapa_ok_tabla{background:#1B2B21;border:1px solid #3E6B4F;border-radius:12px;padding:12px 14px;gap:8px}
.st-key-mapa_nota_bastantes{background:#282526;border:1px solid #403A3C;border-radius:10px;
  padding:9px 12px}
.st-key-mapa_correl{border-top:1px solid #403A3C;padding-top:12px}
.st-key-mapa_vars_cabeza{flex-wrap:wrap}
.st-key-mapa_curva{border-top:1px solid #403A3C;padding-top:10px}
.st-key-mapa_comparar{border-top:1px solid #403A3C;padding-top:10px}
.st-key-mapa_grupo_a_grupo{border-top:1px solid #403A3C;padding-top:12px}
.mapa-grupo{font-weight:600;color:#ECEFF1;font-size:13px}
.mapa-grupo span{font-weight:400;color:#98A2AB}
/* Las tres cifras de las muestras */
.mapa-cifras{display:grid;grid-template-columns:repeat(var(--n,3),minmax(0,1fr));gap:10px;
  max-width:calc(var(--n,3) * 207px)}
.mapa-cifras div{border-radius:12px;padding:12px 14px;display:flex;flex-direction:column;gap:2px}
.mapa-cifras .et{font-size:12px}
.mapa-cifras .n{font-family:Montserrat,Inter,sans-serif;font-size:30px;font-weight:700;line-height:1.1}
.mapa-cifras .si{background:#1B2B21;border:1px solid #2F5A3C}
.mapa-cifras .si .et{color:#A5D6A7}.mapa-cifras .si .n{color:#7FD483}
.mapa-cifras .no{background:#2A1A1A;border:1px solid #5A2E2E}
.mapa-cifras .no .et{color:#F5A3A1}.mapa-cifras .no .n{color:#F28B88}
.mapa-cifras .tot{background:#282526;border:1px solid #403A3C}
.mapa-cifras .tot .et{color:#AEB6BE}.mapa-cifras .tot .n{color:#F5F7F8}
/* El F1 grande */
.mapa-f1{display:flex;flex-wrap:wrap;align-items:baseline;gap:14px;font-size:13px;color:#CDD3D9}
.mapa-f1 b{font-family:Montserrat,Inter,sans-serif;font-size:30px;font-weight:700;color:#7CC496}
/* Los avisos con los colores de la demo */
.st-key-mapa_pantalla [data-testid="stAlertContainer"]{border-radius:12px;color:#CDD3D9}
.st-key-mapa_pantalla [data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]){
  background:#1B2B21;border:1px solid #3E6B4F}
.st-key-mapa_pantalla [data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]){
  background:#2A2416;border:1px solid #6B5A2A}
.st-key-mapa_pantalla [data-testid="stAlertContentWarning"] p,
.st-key-mapa_pantalla [data-testid="stAlertContentWarning"] li{color:#E9DDBF}
.st-key-mapa_pantalla [data-testid="stAlertContentWarning"] strong{color:#F5D58A}
.st-key-mapa_pantalla [data-testid="stAlertContainer"]:has([data-testid="stAlertContentInfo"]){
  background:#14222C;border:1px solid #2F5A78}
.st-key-mapa_pantalla [data-testid="stAlertContentSuccess"] p,
.st-key-mapa_pantalla [data-testid="stAlertContentInfo"] p{color:#CDD3D9}
.st-key-mapa_pantalla [data-testid="stAlertContentSuccess"] strong,
.st-key-mapa_pantalla [data-testid="stAlertContentInfo"] strong{color:#ECEFF1}
/* Enlaces de descarga y botones que son enlaces */
.st-key-mrf_dl_muestras button,.st-key-mrf_dl_tabla button,.st-key-mrf_dl_peso button,
.st-key-mrf_vars_volver button{padding:0;min-height:0;border:none;background:none}
.st-key-mrf_dl_muestras button p,.st-key-mrf_dl_tabla button p,.st-key-mrf_dl_peso button p{
  font-size:13px;color:#7CC496;text-decoration:underline;text-underline-offset:3px}
.st-key-mrf_vars_volver button p{font-size:13px;font-weight:600;color:#A3D9B5}
.st-key-mrf_vars_volver button:disabled p{color:#5E6871}
/* Las descargas del mapa, como fichas */
[class*="st-key-mrf_dl_mapa_"] button{min-height:36px;padding:6px 12px;border:1px solid #60575A;
  background:#393335;border-radius:10px}
[class*="st-key-mrf_dl_mapa_"] button:hover{border-color:#7CC496}
[class*="st-key-mrf_dl_mapa_"] button p{font-size:13px;font-weight:600;color:#ECEFF1}
.st-key-mapa_descargas{flex-wrap:wrap;gap:8px}
/* El hueco del mapa antes de hacerlo */
.mapa-vacio{height:440px;border-radius:18px;border:1px solid #403A3C;background:#262324;
  display:flex;align-items:center;justify-content:center;color:#98A2AB;font-size:14px}
.st-key-mapa_visor iframe{border-radius:18px;border:1px solid #403A3C}
/* El píxel pinchado */
.mapa-pixel{background:rgba(34,31,32,.94);border:1px solid #7CC496;border-radius:12px;
  padding:12px;display:flex;flex-direction:column;gap:4px;font-size:12.5px;color:#CDD3D9;max-width:360px}
.mapa-pixel .t{font-weight:700;color:#ECEFF1}
.mapa-pixel b{color:#ECEFF1;font-weight:600}
/* La leyenda del bloque de gráficos de «5 · Peso» */
.mapa-leyenda{display:flex;flex-wrap:wrap;gap:8px 18px;font-size:12.5px;color:#ECEFF1}
.mapa-leyenda span{display:flex;align-items:center;gap:7px}
.mapa-leyenda i{width:14px;height:14px;border-radius:3px;display:inline-block;box-sizing:border-box}
.mapa-barras{display:flex;flex-direction:column;gap:7px;font-size:13px;color:#CDD3D9}
.mapa-barras .f{display:flex;gap:10px;align-items:center}
.mapa-barras .nom{width:300px;flex:none;line-height:1.3}
.mapa-barras .b{flex:1;display:flex;align-items:center;gap:8px}
.mapa-barras .b i{display:block;flex:none;height:12px;background:#7CC496;border-radius:3px}
.mapa-barras .b span{font-size:12px;color:#AEB6BE}
'''


with tab_tipos, st.container(key='mapa_pantalla'):
    import mapeo_rf as mr
    _en = i18n.get_lang() == 'en'
    st.html('<style>' + _CSS_MAPA + '</style>')

    def _var_m(n):
        """El nombre de una variable DEL MAPA, con las fechas del mapa (15 jun – 24 jul,
        1 ago – 29 sep y 1 jun – 29 sep: Earth Engine deja fuera el último día). No es el
        de los puntos del detector (_var, con 25 jul y 30 sep)."""
        return mr.etiqueta(n, _en)

    def _cat(c):
        """El nombre de una categoría, con mayúscula, como en la demo («Borreguil», «No
        borreguil», «Arroyo · húmedo»)."""
        t = str(i18n.tv(c) or '')
        return t[:1].upper() + t[1:]

    # Los colores de las gráficas de point_profile en esta pantalla, los de la demo: verde
    # #7CC496 para la curva, gris #98A2AB para la raya, naranja #FF8A3D para «Tus puntos»
    # y rótulos #CDD3D9. (point_profile es de las demás pantallas: aquí se cambian sobre la
    # especificación del gráfico, sin tocarlo.)
    _COLORES_DEMO = {'#1565c0': '#7CC496', '#757575': '#98A2AB', '#e65100': '#FF8A3D',
                     '#AEB6BE': '#CDD3D9', '#2e7d32': '#7CC496'}

    def _recolorear(chart, cambios=None):
        """La especificación de un gráfico de Altair con otros colores (para
        st.vega_lite_chart)."""
        import json as _json
        s = _json.dumps(chart.to_dict())
        for a, b in (cambios or _COLORES_DEMO).items():
            s = s.replace(f'"{a}"', f'"{b}"').replace(f'"{a.upper()}"', f'"{b}"')
        return _json.loads(s)

    def _cajas_mapa(filas, variables, grupos):
        """Las cajas de «5 · Peso», como en la demo: un panel por variable, con su escala;
        en cada grupo, la caja va del 25 % al 75 % de sus muestras (relleno y borde de su
        color), la mediana en blanco y las líneas del 10 % al 90 %, sin puntos sueltos.
        `variables`: [(clave, rótulo)]; `grupos`: [(categoría, nombre, relleno, borde)].
        None si no hay datos."""
        import altair as alt
        import numpy as np

        def _t(v):
            return _num(v)
        recs = []
        for k, lab in variables:
            for c, nom, _rel, _bor in grupos:
                xs = []
                for f in filas:
                    if f.get('categoria') != c:
                        continue
                    try:
                        x = float(f.get(k))
                    except (TypeError, ValueError):
                        continue
                    if x == x:
                        xs.append(x)
                if not xs:
                    continue
                p10, p25, p50, p75, p90 = (float(q) for q in np.percentile(xs, [10, 25, 50,
                                                                                 75, 90]))
                recs.append({'variable': lab, 'grupo': nom, 'p10': p10, 'p25': p25,
                             'p50': p50, 'p75': p75, 'p90': p90,
                             'med': _t(p50), 'mitad': f'{_t(p25)} – {_t(p75)}',
                             'casi': f'{_t(p10)} – {_t(p90)}', 'n': str(len(xs))})
        if not recs:
            return None
        df = pd.DataFrame(recs)
        nombres = [g[1] for g in grupos if g[1] in set(df['grupo'])]
        relleno = alt.Scale(domain=nombres, range=[g[2] for g in grupos if g[1] in nombres])
        borde = alt.Scale(domain=nombres, range=[g[3] for g in grupos if g[1] in nombres])
        x = alt.X('grupo:N', title=None, axis=None, sort=nombres)
        coma = ("format(datum.value, '~r')" if _en
                else "replace(format(datum.value, '~r'), '.', ',')")
        y = alt.Y('p10:Q', title=None, scale=alt.Scale(zero=False),
                  axis=alt.Axis(labelExpr=coma, labelColor='#AEB6BE', gridColor='#393335',
                                domainColor='#60575A', tickCount=4))
        tip = [alt.Tooltip('grupo:N', title=i18n.pick('Grupo', 'Group')),
               alt.Tooltip('med:N', title=i18n.pick('Mediana', 'Median')),
               alt.Tooltip('mitad:N', title=i18n.pick('La mitad central',
                                                      'The middle half')),
               alt.Tooltip('casi:N', title=i18n.pick('Casi todas (del 10 % al 90 %)',
                                                     'Almost all (10 % to 90 %)')),
               alt.Tooltip('n:N', title=i18n.pick('Muestras', 'Samples'))]
        b = alt.Chart(df)
        ancho = 26 if len(nombres) <= 2 else 18
        capas = alt.layer(
            b.mark_rule(strokeWidth=2).encode(
                x=x, y=y, y2='p90:Q', color=alt.Color('grupo:N', scale=borde, legend=None),
                tooltip=tip),
            b.mark_tick(thickness=2, size=ancho * 0.55).encode(
                x=x, y='p10:Q', color=alt.Color('grupo:N', scale=borde, legend=None)),
            b.mark_tick(thickness=2, size=ancho * 0.55).encode(
                x=x, y='p90:Q', color=alt.Color('grupo:N', scale=borde, legend=None)),
            b.mark_bar(size=ancho, strokeWidth=2, cornerRadius=3).encode(
                x=x, y='p25:Q', y2='p75:Q',
                fill=alt.Fill('grupo:N', scale=relleno, legend=None),
                stroke=alt.Stroke('grupo:N', scale=borde, legend=None), tooltip=tip),
            b.mark_tick(color='#FFFFFF', thickness=3, size=ancho + 8).encode(
                x=x, y='p50:Q', tooltip=tip),
        ).properties(width=150 if len(nombres) <= 2 else 170, height=190)
        return capas.facet(
            facet=alt.Facet('variable:N', title=None, sort=[lab for _, lab in variables],
                            header=alt.Header(
                                labelFontWeight='bold', labelColor='#F5F7F8',
                                labelFontSize=12.5, labelLimit=260,
                                # en dos líneas, las fechas debajo (la demo)
                                labelExpr="lastindexof(datum.label, ' (') > 0 ? "
                                          "[substring(datum.label, 0, lastindexof(datum.label, "
                                          "' (')), substring(datum.label, lastindexof("
                                          "datum.label, ' (') + 1)] : datum.label")),
            columns=4).resolve_scale(y='independent').configure_view(
            fill='#282526', stroke='#403A3C')

    # El título y, a la derecha, qué se quiere cartografiar (la demo)
    _cab_mapa = st.container(key='mapa_cabeza', horizontal=True,
                             vertical_alignment='bottom', gap='medium')
    with _cab_mapa, st.container(key='mapa_titulo', gap='small'):
        st.markdown('# ' + i18n.pick('El mapa de borreguil de tu zona',
                                     'The borreguil map of your area'))
        st.markdown(i18n.pick(
            'Con tus puntos revisados, la app aprende tu zona y clasifica cada píxel de '
            '10 m. Necesita Earth Engine.',
            'With your reviewed points, the app learns your area and classifies every 10 m '
            'pixel. It needs Earth Engine.'))
    if 'rows' not in st.session_state:
        st.info(i18n.pick(
            'Aquí se entrena un modelo con tus puntos y se obtiene el **mapa de '
            'borreguiles**: dónde hay borreguil y, si quieres, de qué tipo. Todavía no '
            'hay puntos en la sesión:\n\n'
            '1. Ejecuta el análisis (**1 · Analizar**, botón **Analizar**). El '
            'resultado sale en la pantalla **Resultados**.\n'
            '2. En **2 · Revisar**, marca puntos como **borreguil o no borreguil '
            'verificado**, o añádelos pinchando en **Resultados → Mapa**: con eso se '
            'hace el mapa de borreguil sí o no. Para el tipo, revisa además el tipo de '
            'los borreguiles (ambiente, humedad y pureza).\n'
            '3. Vuelve a esta pantalla.',
            'Here a model is trained with your points to produce the **borreguil '
            'map**: where there is borreguil and, if you want, of which type. There are '
            'no points in the session yet:\n\n'
            '1. Run the analysis (**1 · Analyse**, **Analyse** button). The result appears in '
            'the **Results** screen.\n'
            '2. Under **2 · Review**, mark points as **verified borreguil or '
            'verified not-borreguil**, or add them by clicking under **Results → Map**: '
            'that is enough for the borreguil yes-or-no map. For the type, also review '
            'the type of the borreguiles (environment, moisture and purity).\n'
            '3. Come back to this screen.'))
    else:
        rows = st.session_state.rows
        _niv_txt = {
            mr.NIVEL_SI_NO: i18n.pick('Borreguil sí o no', 'Borreguil yes or no'),
            'humedad': i18n.pick('Humedad (húmedo o seco)', 'Moisture (wet or dry)'),
            # (la autora, 27-11-2026: «humedad está repetido»: este es el tipo completo,
            # que incluye la humedad)
            'ambiente_humedad': i18n.pick('Tipo completo (ambiente y humedad)',
                                          'Full type (environment and moisture)'),
        }
        # Lo que se ofrece (24-11-2026, la autora): sin «Ambiente», en el que el modelo no ve
        # dónde están las lagunas ni los cauces y puede poner «laguna» lejos de
        # cualquiera («Ambiente y humedad» lo pone por cercanía), ni «Pureza», que
        # repetía la «Pureza dentro del píxel» del mapa (salían dos purezas distintas en
        # un mismo mapa). mapeo_rf los sigue sabiendo hacer.
        _NIVELES_APP = [n for n in mr.NIVELES if n in _niv_txt]
        # El nivel va encima de los pasos: lo que se elige aquí vale para todos. Viene
        # elegido el primer paso: borreguil sí o no.
        if st.session_state.get('mrf_nivel') not in _NIVELES_APP:
            st.session_state['mrf_nivel'] = mr.NIVEL_SI_NO
        with _cab_mapa, st.container(key='mapa_nivel', width=340):
            _nivel = st.selectbox(
                i18n.pick('Qué quieres cartografiar', 'What to map'),
                _NIVELES_APP, key='mrf_nivel', format_func=lambda n: _niv_txt[n],
                help=i18n.pick(
                    'Se entrena un modelo por cada opción.\n\n'
                    'Borreguil sí o no decide solo si hay borreguil. Aprende de todos los '
                    'puntos marcados en campo como borreguil o no, tengan o no el tipo '
                    'revisado. Su mapa ya es el de borreguil sí o no.\n\n'
                    'Para el tipo, entrena después una de las otras opciones: aprende el '
                    'tipo solo de los borreguiles con el tipo revisado, y en su mapa el sí '
                    'o no lo sigue decidiendo el modelo de Borreguil sí o no.\n\n'
                    'Humedad hace el mapa solo de húmedo o seco. Tipo completo (ambiente y '
                    'humedad) añade el ambiente: el modelo dice si es húmedo o seco, y el '
                    'ambiente (arroyo, laguna o ladera) sale de la cercanía de cada píxel a '
                    'lagunas y cauces, que se ajusta en 4 · Mapa.\n\n'
                    'Recomendado: primero Borreguil sí o no, y después Tipo completo.\n\n'
                    'No se ofrece la combinación de los tres niveles (18 categorías): no '
                    'habría puntos para casi ninguna. La pureza se añade al mapa de otra '
                    'manera, en 4 · Mapa.',
                    'One model is trained for each option.\n\n'
                    'Borreguil yes or no only decides whether there is a borreguil. It '
                    'learns from every point marked in the field as a borreguil or not, '
                    'whether or not its type has been reviewed. Its map is already the '
                    'borreguil yes-or-no map.\n\n'
                    'For the type, train one of the other options afterwards: it learns '
                    'the type only from the borreguiles whose type has been reviewed, and '
                    'in its map the yes or no is still decided by the Borreguil yes or no '
                    'model.\n\n'
                    'Moisture maps only wet or dry. Full type (environment and moisture) '
                    'adds the environment: the model says whether it is wet or dry, and '
                    'the environment (stream, lake or slope) comes from how close each '
                    'pixel is to lakes and watercourses, set under 4 · Map.\n\n'
                    'Recommended: first Borreguil yes or no, then Full type.\n\n'
                    'The combination of the three levels (18 categories) is not offered: '
                    'there would be no points for most of them. Purity is added to the '
                    'map in another way, under 4 · Map.'))
        # En «Ambiente y humedad» el modelo aprende solo el nivel de dentro (la humedad);
        # el de fuera (el ambiente) lo pone el mapa por cercanía a lagunas y cauces.
        _nivel_mod = mr.nivel_del_modelo(_nivel)
        _cercania = mr.nivel_por_cercania(_nivel)

        def _leyenda_cats(cats, marca, color_de=lambda c, i: c['color']):
            """Las filas de una leyenda (HTML): una por categoría, con su marca de
            color. En el nivel combinado, una por AMBIENTE: su nombre y, al lado, sus
            categorías de dentro (húmedo, seco), cada una con su color."""
            import html as _h
            trozos, i, abierta = [], 0, False
            for f in mr.filas_de_leyenda(cats):
                if 'titulo' in f:
                    trozos.append(('</div>' if abierta else '')
                                  + '<div style="margin:3px 0;white-space:nowrap;">'
                                  '<span style="display:inline-block;min-width:3.6em;">'
                                  f'{_h.escape(i18n.tv(f["titulo"]))}</span>')
                    abierta = True
                    continue
                if f['rama']:
                    trozos.append('<span style="margin-right:7px;">' + marca(color_de(f, i))
                                  + f'{_h.escape(i18n.tv(f["etiqueta"]))}</span>')
                else:
                    trozos.append(('</div>' if abierta else '') + '<div style="margin:2px 0;">'
                                  + marca(color_de(f, i))
                                  + f'{_h.escape(i18n.tv(f["etiqueta"]))}</div>')
                    abierta = False
                i += 1
            return ''.join(trozos) + ('</div>' if abierta else '')
        def _para_el_mapa(mod):
            """(modelo con el que se hace el mapa, por qué no se ha podido juntar). Un
            modelo del tipo sin «no borreguil» lo hace en dos pasos, con el de sí o no
            si está entrenado (mapeo_rf.en_dos_pasos); cualquier otro, él solo."""
            si_no = st.session_state.get('mrf_modelos', {}).get(mr.NIVEL_SI_NO)
            if (not mod or not si_no or _nivel == mr.NIVEL_SI_NO
                    or any(c['categoria'] == mr.AUSENCIA for c in mod['categorias'])):
                return mod, None
            try:
                return mr.en_dos_pasos(si_no, mod), None
            except ValueError as e:
                return mod, str(e)

        def _mostrar_seleccion(ap, metodo, param):
            """Lo que sale en «1 · Datos» al aplicar el VIF o el PCA, como en la demo: qué
            variables se quedan (arriba quedan marcadas solo esas) y, variable a variable,
            por qué (25-11-2026). `ap` es lo guardado al aplicar; `param`, el umbral del VIF
            o la correlación mínima del PCA."""
            s, filas, de = ap['sel'], ap['filas'], ap['de']
            nq = len(s['quedan'])
            fijas = [v for v in (s.get('fijas') or [])]
            if fijas == ['elev_dem_m']:
                f_es, f_en = 'la altitud', 'elevation'
            else:
                f_es = f_en = ', '.join(_var_m(v) for v in fijas)
            if metodo == 'vif':
                _p = (f'{float(param):g}' if float(param).is_integer()
                      else _num(param, 1))
                st.success('**' + i18n.pick(
                    f'Se quedan {nq} de las {de} variables (umbral {_p}'
                    + (f'; {f_es}, siempre' if fijas else '')
                    + '). Arriba quedan marcadas solo esas.',
                    f'{nq} of the {de} variables stay (threshold {_p}'
                    + (f'; {f_en}, always' if fijas else '')
                    + '). Only those stay selected above.') + '**')
            else:
                _ne = len(s['ejes'])
                st.success('**' + i18n.pick(
                    f'Se quedan {nq}: una variable por cada uno de los {_ne} ejes '
                    'importantes' + (f', y {f_es}' if fijas else '')
                    + '. Arriba quedan marcadas solo esas.',
                    f'{nq} stay: one variable for each of the {_ne} important axes'
                    + (f', and {f_en}' if fijas else '')
                    + '. Only those stay selected above.') + '**')
            if s['pocas_muestras']:
                _m = 'VIF' if metodo == 'vif' else 'PCA'
                st.warning(i18n.pick(
                    f"Con {ap['n']} muestras y {de} variables el {_m} no es fiable: hacen "
                    'falta bastantes más muestras que variables.',
                    f"With {ap['n']} samples and {de} variables the {_m} is not reliable: "
                    'many more samples than variables are needed.'))
            k_q = i18n.pick('Qué pasa', 'What happens')

            def que_pasa(f):
                q = f['que']
                if q == 'fija':
                    return i18n.pick('Se queda siempre', 'Always stays')
                if q in ('queda', 'elegida'):
                    if metodo == 'pca' and f['eje']:
                        return i18n.pick(f"Se queda (eje {f['eje']})",
                                         f"Stays (axis {f['eje']})")
                    return i18n.pick('Se queda', 'Stays')
                if q == 'no_varia':
                    return i18n.pick('Se quita: no varía', 'Removed: it does not vary')
                base = (i18n.pick('Se quita la primera', 'Removed first')
                        if f.get('paso') == 1 else i18n.pick('Se quita', 'Removed'))
                if metodo == 'vif' and f.get('vif') == float('inf'):
                    if f['variable'] == 'ndvi_drop':
                        return base + i18n.pick(': es el NDVI de inicio menos el de fin',
                                                ': it is the early NDVI minus the late one')
                    return base + i18n.pick(': se obtiene exactamente de otras',
                                            ': it is obtained exactly from others')
                return base

            def por_que(f):
                return mr.texto_por_que(f, metodo, _en, nombre=_var_m, num=_num,
                                        umbral=param if metodo == 'vif' else None,
                                        carga=param if metodo == 'pca' else None)

            with st.expander(i18n.pick('Por qué, variable a variable',
                                       'Why, variable by variable'), expanded=True):
                if metodo == 'vif':
                    def txt_vif(v):
                        return '—' if v is None else ('∞' if v == float('inf') else _num(v))
                    st.dataframe(pd.DataFrame([{
                        'Variable': _var_m(f['variable']), 'VIF': txt_vif(f['vif']),
                        k_q: que_pasa(f)} for f in filas]), hide_index=True, width='stretch')
                    st.caption(i18n.pick(
                        'Se quita de una en una la de VIF más alto y se vuelve a calcular. El '
                        'VIF que sale es el que tenía al quitarla.',
                        'The one with the highest VIF is removed, one at a time, and it is '
                        'computed again. The VIF shown is the one it had when removed.'))
                else:
                    # Cada eje, con lo que resume: el grupo de EBVs que más pesa entre sus
                    # variables (verdor, agua, relieve…), como en la demo
                    _resume = {'productividad': ('Verdor', 'Greenness'),
                               'agua_vegetacion': ('Humedad de la vegetación',
                                                   'Vegetation moisture'),
                               'clorofila': ('Clorofila', 'Chlorophyll'),
                               'fenologia': ('Fenología', 'Phenology'),
                               'agua': ('Agua', 'Water'),
                               'suelo': ('Brillo del suelo', 'Ground brightness'),
                               'perturbaciones': ('Perturbaciones', 'Disturbance'),
                               'relieve': ('Relieve', 'Terrain')}

                    def _que_resume(e):
                        peso = {}
                        for d in e['variables']:
                            g = mr.grupo_ebv(d['variable'])
                            peso[g] = peso.get(g, 0.0) + abs(d['carga'] or 0.0)
                        if not peso:
                            return ''
                        g = max(peso, key=peso.get)
                        return ' · ' + _resume.get(g, (g, g))[1 if _en else 0]

                    _fil_eje = []
                    for i, e in enumerate(s['ejes']):
                        _c = next((d['carga'] for d in e['variables']
                                   if d['variable'] == e['elegida']), None)
                        if e['elegida']:
                            _pq = (i18n.pick(f'correlación {_num(abs(_c))} con el eje; ',
                                             f'correlation {_num(abs(_c))} with the axis; ')
                                   if i == 0 else
                                   i18n.pick(f'correlación {_num(abs(_c))}; ',
                                             f'correlation {_num(abs(_c))}; ')) if _c is not None else ''
                            _pq += (i18n.pick('la que mejor separa borreguil y no',
                                              'the one that best separates borreguil and not')
                                    if i == 0 and _nivel == mr.NIVEL_SI_NO else
                                    i18n.pick('la que mejor separa', 'the one that separates best'))
                        else:
                            _pq = i18n.pick('ninguna llega a la correlación mínima',
                                            'none reaches the minimum correlation')
                        _fil_eje.append({
                            i18n.pick('Eje (lo que resume)', 'Axis (what it sums up)'):
                                f"{e['eje']}{_que_resume(e)}",
                            i18n.pick('Variable que se queda', 'Variable that stays'):
                                _var_m(e['elegida']) if e['elegida'] else '—',
                            i18n.pick('Por qué esa', 'Why that one'): _pq})
                    for v in fijas:
                        _fil_eje.append({
                            i18n.pick('Eje (lo que resume)', 'Axis (what it sums up)'): '—',
                            i18n.pick('Variable que se queda', 'Variable that stays'): _var_m(v),
                            i18n.pick('Por qué esa', 'Why that one'):
                                i18n.pick('Se queda siempre (fuera del PCA)',
                                          'Always stays (outside the PCA)')})
                    st.dataframe(pd.DataFrame(_fil_eje), hide_index=True, width='stretch')
                    # El detalle de cada eje y de cada variable (lo que había antes de la
                    # demo), para quien quiera las cifras
                    k_lig = i18n.pick(f"Ligadas al eje (|r| ≥ {_num(param)})",
                                      f"Tied to the axis (|r| ≥ {_num(param)})")
                    st.dataframe(pd.DataFrame([{
                        i18n.pick('Eje', 'Axis'): e['eje'],
                        i18n.pick('Valor propio', 'Eigenvalue'): _num(e['valor_propio']),
                        i18n.pick('Varianza (%)', 'Variance (%)'): _num(e['varianza_pct'], 1),
                        k_lig: ', '.join(f"{_var_m(d['variable'])} ({_num(d['carga'])})"
                                         for d in e['variables']) or '—',
                        i18n.pick('Se queda', 'Stays'):
                            _var_m(e['elegida']) if e['elegida'] else '—'}
                        for e in s['ejes']]), hide_index=True, width='stretch')
                    st.dataframe(pd.DataFrame([{
                        'Variable': _var_m(f['variable']),
                        k_q: que_pasa(f),
                        i18n.pick('Su eje', 'Its axis'): str(f['eje']) if f['eje'] else '—',
                        i18n.pick('Correlación con su eje', 'Correlation with its axis'):
                            '—' if f['carga'] is None else _num(f['carga']),
                        i18n.pick('Separa sola', 'Separates on its own'):
                            '—' if f['separa'] is None else _num(f['separa'])}
                        for f in filas]), hide_index=True, width='stretch')
                    st.caption(i18n.pick(
                        'Cada eje del PCA resume variables que van juntas. Correlación con el '
                        'eje: lo ligada que está la variable a él (cerca de 1 o de −1, mucho). '
                        'Separa sola: lo bien que distingue las categorías ella sola; con dos, '
                        'de 0 (nada) a 1 (del todo), y con más, la F del ANOVA (cuanto más '
                        'alta, mejor).',
                        'Each PCA axis summarises variables that go together. Correlation with '
                        'the axis: how tied the variable is to it (close to 1 or −1, a lot). '
                        'Separates on its own: how well it tells the categories apart by '
                        'itself; with two, from 0 (not at all) to 1 (completely), and with '
                        'more, the ANOVA F (the higher, the better).'))
                st.markdown('\n'.join(
                    f"- **{_var_m(f['variable'])}** · {que_pasa(f)}. {por_que(f)}"
                    for f in filas))

        # Por dónde va el trabajo en el nivel elegido: los cinco pasos, como en la demo,
        # cada uno con lo que dice de él. El hueco se rellena al final, cuando los pasos han
        # hecho sus cuentas, pero queda encima de ellos. Los pasos son las pestañas de un
        # st.tabs con la barra oculta (todo se sigue ejecutando en cada pasada); los botones
        # de arriba cambian de pestaña con st.session_state['mrf_paso'].
        _estado = st.container(key='mapa_pasos')
        _PASOS_MAPA = (('datos', i18n.pick('Datos', 'Data')),
                       ('tabla', i18n.pick('Tabla', 'Table')),
                       ('modelo', i18n.pick('Modelo', 'Model')),
                       ('mapa', i18n.pick('Mapa', 'Map')),
                       ('peso', i18n.pick('Peso', 'Weights')))
        _rot_pasos = [f'{_i} · {_t}' for _i, (_, _t) in enumerate(_PASOS_MAPA, 1)]
        # El número del paso no depende del idioma: al cambiarlo se sigue en el mismo.
        if st.session_state.get('mrf_paso') not in _rot_pasos:
            st.session_state['mrf_paso'] = _rot_pasos[
                min(4, max(0, int(st.session_state.get('_mrf_paso_n', 0))))]
        _t_datos, _t_tabla, _t_modelo, _t_mapa, _t_peso = st.tabs(
            _rot_pasos, key='mrf_paso', on_change='rerun')

        # ------------------------------------------------------------ 1 · Datos
        with _t_datos:
            # Ausencias de campo: lo único que puede formar la clase «no borreguil».
            _n_aus = sum(1 for r in rows if bp._txt(r.get('Borreguil')).lower() == 'no'
                         and bp._txt(r.get('Duda')).lower() != 'si')
            # Quién decide si un píxel es borreguil (en un nivel del tipo): se elige al
            # final de la tarjeta «Muestras», como en la demo, pero hace falta antes.
            if _nivel == mr.NIVEL_SI_NO:
                # El de sí o no siempre lleva las ausencias: sin ellas no hay «no».
                _incluir = True
            elif _n_aus:
                _incluir = st.session_state.get('mrf_quien_decide', 'si_no') == 'este'
            else:
                _incluir = False
            _res = mr.muestras(rows, _nivel_mod, incluir_ausencias=bool(_incluir and _n_aus),
                               es_borreguil=pp.is_borreguil_decision)
            _nm = _res['n_usadas']
            _n_cl = {c['categoria']: c['n'] for c in _res['clases']}
            _nb = sum(n for c, n in _n_cl.items() if c != mr.AUSENCIA)
            _na_m = _n_cl.get(mr.AUSENCIA)          # None: el nivel no lleva ausencias

            with st.container(key='tarjeta_mapa_muestras'):
                st.markdown('##### ' + i18n.pick('Muestras', 'Samples'))
                # Las cifras: presencias (en un nivel del tipo, una por categoría),
                # ausencias y el total
                import html as _hm
                if _nivel == mr.NIVEL_SI_NO:
                    _cif = [('si', i18n.pick('Presencias (borreguil)', 'Presences (borreguil)'),
                             _nb)]
                else:
                    _cif = [('si', f"{_cat(c['categoria'])} (borreguil)", c['n'])
                            for c in _res['clases'] if c['categoria'] != mr.AUSENCIA]
                if _na_m is not None:
                    _cif.append(('no', i18n.pick('Ausencias (no borreguil)',
                                                 'Absences (not a borreguil)'), _na_m))
                _cif.append(('tot', i18n.pick('Total de muestras', 'Total samples'), _nm))
                st.markdown(f'<div class="mapa-cifras" style="--n:{len(_cif)}">' + ''.join(
                    f'<div class="{k}"><span class="et">{_hm.escape(t)}</span>'
                    f'<span class="n">{n}</span></div>' for k, t, n in _cif) + '</div>',
                    unsafe_allow_html=True)
                if _nivel == mr.NIVEL_SI_NO:
                    _nna = _na_m or 0
                    st.markdown(i18n.pick(
                        f"{_nm} muestra{'' if _nm == 1 else 's'}: {_nb} "
                        f"borreguil{'' if _nb == 1 else 'es'} y {_nna} sin borreguil (de campo, "
                        'revisados en la app o añadidos en el mapa).',
                        f"{_nm} sample{'' if _nm == 1 else 's'}: {_nb} "
                        f"borreguil{'' if _nb == 1 else 'es'} and {_nna} without borreguil "
                        '(from the field, reviewed in the app or added on the map).'))
                elif _cercania:
                    st.markdown(i18n.pick(
                        'La humedad, de los borreguiles con la humedad revisada; el ambiente '
                        'lo pone el mapa por cercanía a lagunas y cauces.',
                        'The moisture, from the borreguiles whose moisture has been reviewed; '
                        'the environment is set by the map from closeness to lakes and '
                        'watercourses.'))
                else:
                    st.markdown(i18n.pick(
                        'Solo los borreguiles con la humedad revisada'
                        + (', y las ausencias.' if _na_m is not None else '.'),
                        'Only the borreguiles whose moisture has been reviewed'
                        + (', and the absences.' if _na_m is not None else '.')))
                if _nivel != mr.NIVEL_SI_NO and not _n_aus:
                    st.caption(i18n.pick(
                        'No hay ausencias de campo en estos puntos, así que no habrá clase '
                        '«no borreguil» (no se inventan ausencias).',
                        'There are no field absences among these points, so there will be '
                        'no "not a borreguil" class (absences are never invented).'))
                _sin_rev = _res['excluidos'].get('tipo_sin_revisar', 0)
                if _sin_rev:
                    st.caption(i18n.pick(
                        f'Los {_sin_rev} borreguiles con el tipo sin revisar pasarán a ser '
                        'muestras en cuanto los revises en **2 · Revisar**.',
                        f'The {_sin_rev} borreguiles whose type has not been reviewed will '
                        'become samples as soon as you review them under **2 · Review**.'))
                if _res['errores']:
                    st.error(i18n.pick('**Con estos puntos todavía no se puede entrenar:**',
                                       '**These points are not enough to train yet:**')
                             + ''.join('\n- ' + mr.mensaje(e, _en, i18n.tv, nivel=_nivel_mod)
                                       for e in _res['errores']))
                _otros = [a for a in _res['avisos'] if a[0] != 'sin_ausencia']
                if _otros:
                    st.warning(''.join('- ' + mr.mensaje(a, _en, i18n.tv) + '\n'
                                       for a in _otros))
                if ('sin_ausencia',) in _res['avisos']:
                    if _nivel != mr.NIVEL_SI_NO and _n_aus and not _incluir:
                        _hay_si_no = mr.NIVEL_SI_NO in st.session_state.get('mrf_modelos', {})
                        st.info(i18n.pick(
                            'Este modelo aprende **solo el tipo**. Si un píxel es borreguil '
                            'lo decide en el mapa el modelo **«Borreguil sí o no»**'
                            + (', que ya está entrenado.' if _hay_si_no else
                               ': entrénalo también (elígelo arriba, en «Qué quieres '
                               'cartografiar»).'),
                            'This model learns **only the type**. Whether a pixel is a '
                            'borreguil is decided in the map by the **"Borreguil yes or no"** '
                            'model'
                            + (', which is already trained.' if _hay_si_no else
                               ': train it too (choose it above, under "What to map").')))
                    else:
                        st.info(mr.mensaje(('sin_ausencia',), _en))

                # Punto por punto: de dónde viene, si es borreguil y si se usa (y si no,
                # por qué), como en la demo. Todo lo demás va en el CSV.
                with st.expander(i18n.pick(
                        'Ver punto por punto (cuáles se usan y por qué no el resto)',
                        'See point by point (which are used and why not the rest)')):
                    _ids_m = bp.abreviar_ids([f['ID'] for f in _res['tabla']])

                    def _de_donde(r):
                        if bp._txt(r.get('source')) == 'mapa':
                            return 'mapa', i18n.pick('Añadido en el mapa', 'Added on the map')
                        _o = bp._txt(r.get('origin')).lower()
                        if _o == 'dudoso':
                            return 'dudoso', i18n.pick('Revisado: «No estoy seguro»',
                                                       'Reviewed: "Not sure"')
                        if _o == 'verificado' or bp.niveles_revisados(r):
                            return 'app', i18n.pick('Revisado en la app', 'Reviewed in the app')
                        if (bp._txt(r.get('Borreguil')).lower() in ('si', 'no')
                                or bp._txt(r.get('Duda')).lower() == 'si'):
                            return 'campo', i18n.pick('Campo', 'Field')
                        return 'nada', i18n.pick('Sin revisar', 'Not reviewed')

                    _SI_TXT, _NO_TXT = i18n.pick('Sí', 'Yes'), 'No'
                    _filas_p, _usos = [], Counter()
                    for _f, _idc in zip(_res['tabla'], _ids_m):
                        _r = rows[_f['indice']] if 0 <= _f['indice'] < len(rows) else {}
                        _k, _txt = _de_donde(_r)
                        _campo = bp._txt(_r.get('Borreguil')).lower()
                        _dec = bp._txt(_r.get('decision'))
                        if _campo == 'si' or (_k == 'app' and bp.niveles_revisados(_r)):
                            _bo = _SI_TXT
                        elif _campo == 'no':
                            _bo = _NO_TXT
                        elif pp.is_borreguil_decision(_dec):
                            _bo = i18n.pick('Probable (detector)', 'Likely (detector)')
                        else:
                            _bo = '—'
                        if _f['usada']:
                            _usos[_k] += 1
                            if _f['origen_etiqueta'] == mr.ORIGEN_REVISADO and \
                                    _nivel != mr.NIVEL_SI_NO:
                                _usa = i18n.pick('Sí: tipo revisado', 'Yes: reviewed type')
                            elif _k == 'mapa':
                                _usa = i18n.pick('Sí: conocimiento del sitio',
                                                 'Yes: knowledge of the site')
                            elif _k == 'app':
                                _usa = i18n.pick('Sí: verificado', 'Yes: verified')
                            elif _f['categoria'] == mr.AUSENCIA:
                                _usa = i18n.pick('Sí: ausencia de campo', 'Yes: field absence')
                            else:
                                _usa = _SI_TXT
                        elif _f['motivo'] in ('sin_si_no', 'sin_etiqueta') and \
                                pp.is_borreguil_decision(_dec):
                            _usa = i18n.pick('No: lo dice solo el detector',
                                             'No: only the detector says so')
                        elif _f['motivo'] == 'dudoso':
                            _usa = i18n.pick('No: dudoso', 'No: doubtful')
                        else:
                            _usa = 'No: ' + mr.motivo(_f['motivo'], _en)
                        _filas_p.append({
                            i18n.pick('Punto', 'Point'): _idc,
                            i18n.pick('De dónde viene', 'Where it comes from'): _txt,
                            'Borreguil': _bo,
                            i18n.pick('¿Se usa?', 'Used?'): _usa})
                    _k_usa = i18n.pick('¿Se usa?', 'Used?')
                    _dfp_m = pd.DataFrame(_filas_p)
                    if len(_dfp_m):
                        _dfp_m = _dfp_m.style.map(
                            lambda v: 'color:#F5B83D' if str(v).startswith('No')
                            else 'color:#A3D9B5', subset=[_k_usa])
                    st.dataframe(_dfp_m, hide_index=True, width='stretch')
                    _partes = [(_usos['campo'], 'de campo', 'from the field'),
                               (_usos['app'], 'revisados en la app', 'reviewed in the app'),
                               (_usos['mapa'], 'añadidos en el mapa', 'added on the map')]
                    _partes = [p for p in _partes if p[0]]

                    def _juntar(trozos, y):
                        return (', '.join(trozos[:-1]) + f' {y} ' + trozos[-1]
                                if len(trozos) > 1 else ''.join(trozos))
                    _de_es = _juntar([f'{n} {es}' for n, es, _ in _partes], 'y')
                    _de_en = _juntar([f'{n} {en_}' for n, _, en_ in _partes], 'and')
                    st.caption(i18n.pick(
                        f"Los {_res['n_inicial']} puntos de la sesión. Se "
                        f"usa{'' if _nm == 1 else 'n'} {_nm}"
                        + (f': {_de_es}' if _de_es else '') + '.',
                        f"The {_res['n_inicial']} points in the session. {_nm} "
                        f"{'is' if _nm == 1 else 'are'} used"
                        + (f': {_de_en}' if _de_en else '') + '.'))
                st.download_button(
                    label=i18n.pick('Descargar muestras.csv', 'Download muestras.csv'),
                    data=mr.tabla_csv(_res).encode('utf-8-sig'),
                    file_name=f'muestras_{_nivel}.csv', mime='text/csv',
                    key='mrf_dl_muestras', type='tertiary')
                if _nivel != mr.NIVEL_SI_NO and _n_aus:
                    # En un nivel del tipo, lo normal es que el sí o no lo decida el modelo
                    # de sí o no, y que este aprenda solo los tipos (sin «no borreguil»).
                    _quien_txt = {
                        'si_no': i18n.pick('El modelo «Borreguil sí o no» (recomendado)',
                                           'The "Borreguil yes or no" model (recommended)'),
                        'este': i18n.pick('Este mismo modelo, con las ausencias de campo',
                                          'This same model, with the field absences')}
                    st.radio(
                        i18n.pick('Quién decide si un píxel es borreguil',
                                  'Who decides whether a pixel is a borreguil'),
                        list(_quien_txt), key='mrf_quien_decide',
                        format_func=lambda q: _quien_txt[q],
                        help=i18n.pick(
                            'El modelo «Borreguil sí o no»: este modelo aprende solo el '
                            'tipo, de los borreguiles con el tipo revisado. En el mapa, si un '
                            'píxel es borreguil lo decide el modelo de «Borreguil sí o no», '
                            'que aprende de todos los puntos de campo (también de los '
                            'borreguiles sin el tipo revisado), y este le pone el tipo. Hay '
                            'que entrenar los dos.\n\n'
                            'Este mismo modelo: aprende también «no borreguil», con las '
                            'ausencias de campo, y decide él solo. Solo cuentan los '
                            'borreguiles con el tipo revisado.\n\n'
                            'Recomendado: el modelo «Borreguil sí o no».',
                            'The "Borreguil yes or no" model: this model learns only the type, '
                            'from the borreguiles whose type has been reviewed. In the map, '
                            'whether a pixel is a borreguil is decided by the "Borreguil yes '
                            'or no" model, which learns from every field point (including '
                            'borreguiles whose type has not been reviewed), and this one gives '
                            'it its type. Both have to be trained.\n\n'
                            'This same model: it also learns "not a borreguil", with the field '
                            'absences, and decides on its own. Only the borreguiles whose type '
                            'has been reviewed count.\n\n'
                            'Recommended: the "Borreguil yes or no" model.'))

            # ¿Son bastantes puntos? Dónde está la persona en la curva de Sierra Nevada
            # (24-11, idea de la autora). Sale en todos los niveles (la demo): la curva se
            # midió con el de sí o no, y en los demás sirve de referencia.
            with st.container(key='tarjeta_mapa_bastantes'):
                st.markdown('##### ' + i18n.pick('¿Son bastantes puntos?',
                                                 'Are they enough points?'))
                # Borreguiles y ausencias por separado (las ausencias, desde el 25-11-2026:
                # antes, 100 borreguiles y 6 ausencias «llegaban al mínimo»). En un nivel
                # del tipo sin ausencias propias, cuentan las de campo, que son las del
                # modelo de sí o no.
                _na = _na_m if _na_m is not None else _n_aus
                _nq = _nb + _na
                if _nivel != mr.NIVEL_SI_NO:
                    with st.container(key='mapa_nota_bastantes'):
                        st.markdown(i18n.pick(
                            'Estos mínimos son los del mapa de «Borreguil sí o no», que es '
                            'donde se midió la curva de Sierra Nevada. Para la humedad o el '
                            'ambiente sirven de referencia: hacen falta además muestras de '
                            'cada tipo, y cuantas más, mejor.',
                            'These minimums are those of the "Borreguil yes or no" map, which '
                            'is where the Sierra Nevada curve was measured. For moisture or '
                            'environment they are a reference: samples of each type are also '
                            'needed, and the more the better.'))
                _lp = mr.lectura_puntos(_nq, _nb, _na)
                (_mn, _mb), (_rn, _rb) = mr.MINIMO_PUNTOS, mr.RECOMENDADO_PUNTOS
                _ma, _ra = mr.MINIMO_AUSENCIAS, mr.RECOMENDADO_AUSENCIAS
                _fb, _fx = _lp['faltan_borreguiles'], _lp['faltan_ausencias']

                def _bor(k, en=False):
                    return f"{k} borreguil{'' if k == 1 else 'es'}"

                def _aus(k, en=False):
                    return (f"{k} absence{'' if k == 1 else 's'}" if en
                            else f"{k} ausencia{'' if k == 1 else 's'}")

                if _fb and _fx:
                    _falta_es = f'te faltan {_bor(_fb)} y {_aus(_fx)}'
                    _falta_en = f'you need {_bor(_fb, True)} and {_aus(_fx, True)} more'
                elif _fx:
                    _falta_es = f'te faltan {_aus(_fx)}'
                    _falta_en = f'you need {_aus(_fx, True)} more'
                else:
                    _falta_es = f'te faltan {_bor(_fb)}'
                    _falta_en = f'you need {_bor(_fb, True)} more'
                _tienes_es = (f"**{_nq} muestra{'' if _nq == 1 else 's'} ({_bor(_nb)} y "
                              f"{_aus(_na)})**")
                _tienes_en = (f"**{_nq} sample{'' if _nq == 1 else 's'} ({_bor(_nb, True)} "
                              f"and {_aus(_na, True)})**")
                _min_es = f'{_mn} puntos: {_mb} borreguiles y {_ma} ausencias'
                _min_en = f'{_mn} points: {_mb} borreguiles and {_ma} absences'
                _auc = _num(_lp['auc'])
                if _lp['nivel'] == 'sin_referencia':
                    st.warning(i18n.pick(
                        f'Tienes {_tienes_es}. Con menos de 20 no hay referencia: el acierto '
                        f'será bajo y muy variable. Para el mínimo ({_min_es}), '
                        f'{_falta_es}.',
                        f'You have {_tienes_en}. Below 20 there is no reference: accuracy '
                        f'will be low and very variable. For the minimum ({_min_en}), '
                        f'{_falta_en}.'))
                elif _lp['nivel'] in ('pocos', 'bajo_minimo'):
                    _var_es = (', y cambiaba mucho según qué puntos fueran'
                               if _lp['nivel'] == 'pocos' else '')
                    _var_en = (', and it changed a lot depending on which points they were'
                               if _lp['nivel'] == 'pocos' else '')
                    st.warning(i18n.pick(
                        f'Tienes {_tienes_es}. En Sierra Nevada, con {_nq} puntos el AUC en '
                        f'zonas nuevas era de {_auc}{_var_es}. Para el mínimo ({_min_es}), '
                        f'{_falta_es}.',
                        f'You have {_tienes_en}. In Sierra Nevada, with {_nq} points the AUC '
                        f'in new areas was {_auc}{_var_en}. For the minimum ({_min_en}), '
                        f'{_falta_en}.'))
                elif _lp['nivel'] == 'minimo':
                    st.info(i18n.pick(
                        f'Tienes {_tienes_es}: llegas al mínimo. En Sierra Nevada, con '
                        f'{_nq} puntos el AUC en zonas nuevas era de {_auc}. Con más, hasta '
                        f'unos {_rn} ({_rb} borreguiles y {_ra} ausencias), el mapa todavía '
                        'mejora algo.',
                        f'You have {_tienes_en}: you reach the minimum. In Sierra Nevada, '
                        f'with {_nq} points the AUC in new areas was {_auc}. With more, up '
                        f'to about {_rn} ({_rb} borreguiles and {_ra} absences), the map '
                        'still improves a little.'))
                else:
                    st.success(i18n.pick(
                        f'Tienes {_tienes_es}: llegas a lo recomendado. En Sierra Nevada, con '
                        f'{_nq} puntos el AUC en zonas nuevas era de {_auc}; con más puntos '
                        'apenas cambia.',
                        f'You have {_tienes_en}: you reach the recommended number. In Sierra '
                        f'Nevada, with {_nq} points the AUC in new areas was {_auc}; with more '
                        'points it hardly changes.'))

                # Borreguiles y ausencias, siempre a la vista (25-11-2026, la autora: «no
                # veo lo de las ausencias»: si se llegaba, solo salían entre paréntesis)
                def _meta(tienes, hace_falta):
                    return (f'{hace_falta} ✓' if tienes >= hace_falta else
                            i18n.pick(f'{hace_falta} (faltan {hace_falta - tienes})',
                                      f'{hace_falta} ({hace_falta - tienes} to go)'))
                # El total va primero: es el número de la gráfica (100 = 33 + 67); sin él,
                # «mínimo 33» y «mínimo 100 puntos» parecían contradecirse (25-11-2026)
                _k_mu, _k_ti = i18n.pick('Muestras', 'Samples'), i18n.pick('Tienes', 'You have')
                _k_mi, _k_re = i18n.pick('Mínimo', 'Minimum'), i18n.pick('Recomendado',
                                                                        'Recommended')
                _dfq = pd.DataFrame([
                    {_k_mu: i18n.pick('Total (los puntos de la gráfica)',
                                      'Total (the points in the chart)'),
                     _k_ti: str(_nq), _k_mi: _meta(_nq, _mn), _k_re: _meta(_nq, _rn)},
                    {_k_mu: i18n.pick('· borreguiles (presencias)', '· borreguiles (presences)'),
                     _k_ti: str(_nb), _k_mi: _meta(_nb, _mb), _k_re: _meta(_nb, _rb)},
                    {_k_mu: i18n.pick('· ausencias (no borreguil)',
                                      '· absences (not a borreguil)'),
                     _k_ti: str(_na), _k_mi: _meta(_na, _ma), _k_re: _meta(_na, _ra)}])
                st.dataframe(
                    _dfq.style.apply(lambda c: ['font-weight:700',
                                                'font-weight:700;color:#7FD483',
                                                'font-weight:700;color:#F28B88'],
                                     subset=[_k_ti])
                    .map(lambda v: '' if '✓' in str(v) else 'color:#F5D58A',
                         subset=[_k_mi, _k_re]),
                    hide_index=True, width='stretch')
                st.caption(i18n.pick(
                    f'El mínimo de la gráfica, {_mn} puntos, es la suma de {_mb} borreguiles y '
                    f'{_ma} ausencias; lo recomendado, {_rn} = {_rb} + {_ra}. Hacen falta los '
                    'dos: muchos borreguiles no suplen las ausencias que faltan. '
                    'Las ausencias son los sitios que has comprobado que no son borreguil: '
                    'presence = 0 en tu fichero, «No» en «¿Es borreguil?» al revisar un '
                    'punto o «Aquí no hay» al pinchar en el mapa. '
                    'La app no se las inventa. Para buscarlas donde no has muestreado, genera '
                    'puntos al azar «lejos de tus puntos de campo» y revísalos.',
                    f'The minimum in the chart, {_mn} points, is the sum of {_mb} borreguiles '
                    f'and {_ma} absences; the recommended one, {_rn} = {_rb} + {_ra}. Both '
                    'are needed: many borreguiles do not make up for missing absences. '
                    'Absences are the places you have checked are not a borreguil: '
                    'presence = 0 in your file, "No" under "Is it a borreguil?" when '
                    'reviewing a point or "There is none" when clicking on the map. The app '
                    'never invents them. To look for them where you have not sampled, '
                    'generate random points "far from your field points" and review them.'))
                st.vega_lite_chart(spec=_recolorear(pp.chart_curva_puntos(
                    mr.CURVA_PUNTOS_SN, n=_nq,
                    marcas=[(_mn, i18n.pick('Mínimo', 'Minimum')),
                            (_rn, i18n.pick('Recomendado', 'Recommended'))])),
                    width='stretch')
                st.caption(i18n.pick(
                    'Referencia de Sierra Nevada: el mapa entrenado con distintos números de '
                    'puntos revisados y medido en zonas que no había visto (franja: 8 de cada '
                    '10 repeticiones; raya discontinua: con todos los puntos). La curva se '
                    'hizo con 1 borreguil por cada 2 ausencias; de ahí salen los borreguiles '
                    'y las ausencias del mínimo y de lo recomendado. En una montaña más '
                    'variada pueden hacer falta más, y los puntos repartidos por toda la zona '
                    'enseñan más que los concentrados en pocos valles. La curva con tus '
                    'propios puntos se calcula en «3 · Modelo», después de entrenar.',
                    'Sierra Nevada reference: the map trained with different numbers of '
                    'reviewed points and measured in areas it had not seen (band: 8 out of '
                    '10 repetitions; dashed line: with all the points). The curve was made '
                    'with 1 borreguil for every 2 absences; that is where the borreguiles and '
                    'absences of the minimum and the recommended numbers come from. A more '
                    'varied mountain may need more, and points spread over the whole area '
                    'teach more than points concentrated in a few valleys. The curve with '
                    'your own points is computed under "3 · Model", after training.'))

            # Lo que ya está leído en Earth Engine de cada muestra («2 · Tabla»). Vale
            # mientras no cambien los años, la zona UTM de la rejilla ni la manera de
            # calcular las variables (geb.VERSION_PILA). Se calcula aquí porque el VIF y
            # el PCA se aplican en este paso, con esos valores (y antes de las fichas
            # de las variables, que el VIF o el PCA cambian al aplicarse).
            import gee_backend as geb
            try:
                _anios = list(bp.parse_years(years_str))
            except Exception:
                _anios = []
            _usadas = [f for f in _res['tabla'] if f['usada']]
            _epsg = (geb.utm_de([(float(f['lon']), float(f['lat'])) for f in _usadas])
                     if _usadas else None)
            _lect = st.session_state.get('mrf_lectura')
            if _lect and (_lect.get('anios') != _anios or _lect.get('epsg') != _epsg
                          or (_lect.get('info') or {}).get('version') != geb.VERSION_PILA):
                _lect = None
            _leidos = (_lect or {}).get('valores', {})
            _pend = [f for f in _usadas
                     if mr.clave_punto(f['lon'], f['lat']) not in _leidos]

            def _firma_sel(metodo, param, fijas, tab_base):
                """De qué sale lo aplicado con el VIF o el PCA: vale mientras no cambien
                el método, su ajuste, las fijas, las variables de partida, las muestras
                ni su lectura."""
                return (metodo, round(float(param), 6), tuple(sorted(fijas)),
                        tuple(tab_base['predictores']),
                        tuple((round(float(f['lon']), 7), round(float(f['lat']), 7),
                               f['codigo']) for f in tab_base['filas']),
                        (_lect or {}).get('cuando'))

            def _marcar(variables):
                """Deja marcadas en las fichas solo `variables`. Solo vale antes de
                dibujarlas (o en el on_click de un botón)."""
                for gi, g in enumerate(mr.GRUPOS):
                    st.session_state[f'mrf_vars_{gi}'] = [
                        v for v in mr.nombres_predictores([g]) if v in variables]

            # ---- Aplicar el VIF o el PCA (25-11-2026, la autora: «al aplicar el vif o
            # pca me gustaría que en variables predictoras se queden las que han
            # quedado»). El botón (más abajo) solo deja el aviso '_mrf_aplicar'; se aplica
            # aquí, en la pasada siguiente y antes de dibujar las fichas, para poder
            # dejar marcadas solo las que quedan. (Con st.rerun() en el botón, Streamlit
            # olvidaba lo elegido en los controles de debajo, que esa pasada no llegaba
            # a dibujar.) Se parte de las que había marcadas antes de aplicarlo, si
            # siguen marcadas las que quedaron; si no, de las marcadas ahora.
            if st.session_state.pop('_mrf_aplicar', False):
                _m0 = st.session_state.get('mrf_seleccion', 'ninguno')
                _ahora0 = mr.ordenar_predictores([
                    v for gi, g in enumerate(mr.GRUPOS)
                    for v in (st.session_state.get(f'mrf_vars_{gi}',
                                                   mr.nombres_predictores([g])) or [])])
                _a0 = st.session_state.get('mrf_vars_antes')
                _b0 = (list(_a0['vars']) if _a0 and set(_ahora0) == set(_a0['resultado'])
                       else _ahora0)
                _tb0 = mr.construir_tabla(_res, _leidos, _b0) if _b0 else None
                if (_m0 in ('vif', 'pca') and _tb0 is not None and _tb0['listo']
                        and len(_tb0['filas']) >= 3):
                    _p0 = float(st.session_state.get(
                        'mrf_vif_umbral' if _m0 == 'vif' else 'mrf_pca_carga',
                        mr.VIF_UMBRAL if _m0 == 'vif' else mr.PCA_CARGA))
                    _f0 = [v for v in st.session_state.get('mrf_vif_fijas', mr.VIF_FIJAS)
                           if v in _tb0['predictores']]
                    if _m0 == 'vif':
                        _s0 = mr.elegir_por_vif(_tb0['X'], _tb0['predictores'], _p0, _f0)
                        _fl0 = mr.por_que_vif(_tb0['X'], _tb0['predictores'], _s0)
                    else:
                        _s0 = mr.elegir_por_pca(_tb0['X'], _tb0['y'], _tb0['predictores'],
                                                _f0, _p0)
                        _fl0 = mr.por_que_pca(_s0, _tb0['predictores'])
                    st.session_state.setdefault('mrf_sel_aplicada', {})[f'{_nivel}|{_m0}'] = {
                        'firma': _firma_sel(_m0, _p0, _f0, _tb0), 'sel': _s0,
                        'filas': _fl0, 'de': len(_tb0['predictores']),
                        'n': len(_tb0['filas'])}
                    _q0 = mr.ordenar_predictores(_s0['quedan'])
                    st.session_state['mrf_vars_antes'] = {
                        'vars': list(_tb0['predictores']), 'metodo': _m0, 'resultado': _q0}
                    _marcar(_q0)

            def _volver_a_las_de_antes():
                """«Volver a marcar las de antes»: las que había antes de aplicar el VIF o
                el PCA; lo aplicado se deshace."""
                a = st.session_state.pop('mrf_vars_antes', None)
                if a:
                    _marcar(a['vars'])
                st.session_state.pop('mrf_sel_aplicada', None)

            with st.container(key='tarjeta_mapa_variables'):
                # Las marcadas, antes de dibujar las fichas (para la cabecera): las fichas
                # no llevan `default`, su valor de partida (todas) se pone en
                # session_state, que es lo que cambian el VIF o el PCA al aplicarse.
                # (st.pills devuelve a veces el rótulo de la ficha en vez de la variable,
                # por ejemplo al cambiar de idioma: se vuelve a la variable, o se deja)
                for _gi, _g in enumerate(mr.GRUPOS):
                    _k_v, _ops_v = f'mrf_vars_{_gi}', mr.nombres_predictores([_g])
                    # (en cada grupo, cada rótulo es de una sola variable)
                    _de_rotulo = {mr.etiqueta(n, e_, 'corta'): n
                                  for n in _ops_v for e_ in (False, True)}
                    if _k_v not in st.session_state:
                        st.session_state[_k_v] = list(_ops_v)
                    else:
                        _ahora_v = list(st.session_state[_k_v] or [])
                        _limpio = [x if x in _ops_v else _de_rotulo.get(str(x).replace(' ✓', ''))
                                   for x in _ahora_v]
                        _limpio = [x for x in dict.fromkeys(_limpio) if x in _ops_v]
                        if _limpio != _ahora_v:
                            st.session_state[_k_v] = _limpio
                _predictores = mr.ordenar_predictores([
                    v for _gi in range(len(mr.GRUPOS))
                    for v in (st.session_state.get(f'mrf_vars_{_gi}') or [])])
                # Las que había marcadas antes de aplicar el VIF o el PCA: de ellas parte
                # el VIF o el PCA si se vuelve a aplicar (con otro umbral, o el otro
                # método), y se pueden volver a marcar. Si la persona cambia las fichas a
                # mano, lo que marca pasa a ser el punto de partida.
                _antes = st.session_state.get('mrf_vars_antes')
                if _antes and set(_predictores) != set(_antes['resultado']):
                    st.session_state.pop('mrf_vars_antes', None)
                    _antes = None
                _n_tot = len(mr.PREDICTORES)
                _aplicado = _antes['metodo'] if (_antes and set(_antes['vars'])
                                                 != set(_predictores)) else None
                if _aplicado == 'vif':
                    _txt_vars = i18n.pick(
                        f'{len(_predictores)} de {_n_tot} marcadas · quitadas las '
                        'correlacionadas con el VIF',
                        f'{len(_predictores)} of {_n_tot} selected · correlated ones removed '
                        'with the VIF')
                elif _aplicado == 'pca':
                    _txt_vars = i18n.pick(
                        f'{len(_predictores)} de {_n_tot} marcadas · elegidas con el PCA',
                        f'{len(_predictores)} of {_n_tot} selected · chosen with the PCA')
                elif len(_predictores) == _n_tot:
                    _txt_vars = i18n.pick(
                        f'{_n_tot} de {_n_tot} marcadas · Sentinel-2 y relieve, las que '
                        'existen como mapa continuo',
                        f'{_n_tot} of {_n_tot} selected · Sentinel-2 and terrain, those that '
                        'exist as a continuous map')
                else:
                    _txt_vars = i18n.pick(f'{len(_predictores)} de {_n_tot} marcadas',
                                          f'{len(_predictores)} of {_n_tot} selected')
                with st.container(horizontal=True, vertical_alignment='center', gap='small',
                                  key='mapa_vars_cabeza'):
                    st.markdown('##### ' + i18n.pick('Variables predictoras',
                                                     'Predictor variables'), width='content')
                    st.markdown(_txt_vars, width='content')
                    st.space('stretch')
                    st.button(i18n.pick('Volver a marcar las de antes',
                                        'Select the previous ones again'),
                              key='mrf_vars_volver', type='tertiary',
                              on_click=_volver_a_las_de_antes, disabled=_aplicado is None)
                    ayuda_q('q_mapa_volver_marcar',
                            'Vuelve a marcar las variables que había antes de aplicar el VIF '
                            'o el PCA, por ejemplo para entrenar con todas y comparar. Lo '
                            'aplicado se deshace.',
                            'Selects again the variables there were before applying the VIF '
                            'or the PCA, for instance to train with all of them and compare. '
                            'What was applied is undone.')
                # Las variables, como fichas que se marcan y se desmarcan (la demo); el
                # grupo lleva las fechas y cuántas hay marcadas.
                _elegidas = []
                for _gi, _g in enumerate(mr.GRUPOS):
                    _ops = mr.nombres_predictores([_g])
                    _ya = set(st.session_state.get(f'mrf_vars_{_gi}') or [])
                    st.markdown(
                        f'<div class="mapa-grupo">{_hm.escape(mr.nombre_grupo(_g, _en))} '
                        f'<span>· ' + i18n.pick(
                            f'{len(_ya & set(_ops))} de {len(_ops)} marcadas',
                            f'{len(_ya & set(_ops))} of {len(_ops)} selected')
                        + '</span></div>', unsafe_allow_html=True)
                    # (la ✓ de las marcadas la pone el CSS: así el rótulo no cambia al
                    # marcarla)
                    _elegidas += st.pills(
                        mr.nombre_grupo(_g, _en), _ops, selection_mode='multi',
                        key=f'mrf_vars_{_gi}', label_visibility='collapsed',
                        format_func=lambda v: mr.etiqueta(v, _en, 'corta')) or []
                _predictores = mr.ordenar_predictores(_elegidas)
                if not _predictores:
                    st.error(i18n.pick('Elige al menos una variable.',
                                       'Choose at least one variable.'))
                with st.expander(i18n.pick('Qué mide cada variable',
                                           'What each variable measures')):
                    _anios_txt = bp.abreviar_anios(_anios).replace('-', '–')
                    _k_marc = i18n.pick('¿Marcada?', 'Selected?')
                    _dfv = pd.DataFrame([{
                        'Variable': mr.etiqueta(n, _en, 'tabla'),
                        i18n.pick('Cuándo', 'When'): mr.cuando_grupo(g, _anios_txt, _en),
                        i18n.pick('Qué mide', 'What it measures'): mr.que_mide(n, _en),
                        i18n.pick('Qué se quita de las imágenes', 'What is removed from the '
                                  'images'): mr.filtro_grupo(g, _en),
                        _k_marc: (i18n.pick('Sí', 'Yes') if n in _predictores
                                  else i18n.pick('No (quitada)', 'No (removed)'))}
                        for n, g, _es, _en_ in mr.PREDICTORES])
                    st.dataframe(_dfv.style.map(
                        lambda v: 'color:#98A2AB' if str(v).startswith('No') else
                        'color:#A3D9B5', subset=[_k_marc]), hide_index=True, width='stretch')
                    _n_s2 = sum(1 for p in mr.PREDICTORES if p[1] != mr.G_TOPO)
                    st.caption(i18n.pick(
                        f'Las {_n_tot} ({_n_s2} de Sentinel-2 y {_n_tot - _n_s2} del relieve). '
                        'Los años son los del panel de Analizar (de entrada, de 2017 a 2025). '
                        'Inicio de verano (15 jun – 24 jul) y fin de verano (1 ago – 29 sep): '
                        'la mediana de todas las imágenes sin nubes de esas fechas de todos '
                        'los años. Junio a septiembre: la media, el mínimo, el máximo o la '
                        'variabilidad (desviación típica) de cada año, y después la media '
                        'entre años. El relieve no tiene fecha: es el modelo de elevaciones '
                        'Copernicus (30 m). Antes de calcular nada se quitan de las imágenes '
                        'las nubes y, según la variable, sus sombras o la nieve: lo dice la '
                        'columna «Qué se quita de las imágenes».',
                        f'The {_n_tot} ({_n_s2} from Sentinel-2 and {_n_tot - _n_s2} from the '
                        'terrain). The years are those of the Analyse panel (by default, 2017 '
                        'to 2025). Early summer (15 Jun – 24 Jul) and late summer (1 Aug – 29 '
                        'Sep): the median of all the cloud-free images of those dates in all '
                        'the years. June to September: the mean, minimum, maximum or '
                        'variability (standard deviation) of each year, and then the mean '
                        'across years. The terrain has no date: it is the Copernicus '
                        'elevation model (30 m). Before computing anything, clouds and, '
                        'depending on the variable, their shadows or snow are removed from '
                        'the images: the column "What is removed from the images" says so.'))
                with st.expander(i18n.pick('Variables que usa el detector y aquí no están',
                                           'Variables the detector uses that are missing '
                                           'here')):
                    st.markdown('\n'.join(f'- **{i18n.pick(_q, _q_en)}**: {i18n.pick(_p, _p_en)}.'
                                          for _q, _p, _q_en, _p_en in mr.NO_DISPONIBLES))

                # La tabla de entrenamiento con todas las variables elegidas (la de «2 ·
                # Tabla» antes de quitar las correlacionadas), si ya hay algo leído
                _tab_todas = (mr.construir_tabla(_res, _leidos, _predictores)
                              if _predictores and any(mr.clave_punto(f['lon'], f['lat'])
                                                      in _leidos for f in _usadas) else None)

                # ---- Quitar las variables correlacionadas: con el VIF o con un PCA. Desde
                # el 24-11 se decide aquí, junto a las variables (lo pidió la autora), y
                # desde el 25-11 se aplica aquí con un botón, que enseña qué variables se
                # quedan y por qué. Se calcula con los valores de las muestras (la lectura
                # de «2 · Tabla» trae siempre todas las variables). De entrada no se quita
                # nada.
                with st.container(key='mapa_correl'):
                    _met_txt = {'ninguno': i18n.pick('No quitar ninguna', 'Remove none'),
                                'vif': i18n.pick('Con VIF', 'With VIF'),
                                'pca': i18n.pick('Con PCA', 'With PCA')}
                    _metodo = st.radio(
                        i18n.pick('Quitar las variables correlacionadas',
                                  'Remove the correlated variables'),
                        list(_met_txt), format_func=_met_txt.get, key='mrf_seleccion',
                        horizontal=True,
                        help=i18n.pick(
                            'Muchas variables dicen casi lo mismo (los índices de verdor, por '
                            'ejemplo). Hay dos formas de quedarse con menos:\n\n'
                            'Con VIF: el VIF de una variable dice qué parte de ella se '
                            'puede adivinar con todas las demás juntas (5 quiere decir el '
                            '80 %; 10, el 90 %). Se quita la más correlacionada con las '
                            'demás, se vuelve a calcular y se sigue mientras alguna pase del '
                            'umbral.\n\n'
                            'Con PCA: el PCA resume las variables en unos pocos ejes '
                            'independientes. De cada eje importante (valor propio mayor que '
                            '1) se queda una variable: entre las muy ligadas a él, la que '
                            'mejor separa las categorías por sí sola.\n\n'
                            'Los dos se calculan con los valores de las muestras, que se leen '
                            'en «2 · Tabla». Al pulsar Aplicar, sale aquí mismo qué variables '
                            'se quedan y por qué, y arriba se quedan marcadas solo esas. En '
                            'Sierra Nevada, el VIF dejó 14 variables y el PCA con la altitud, '
                            '4, con el mismo acierto (F1 0,865 y 0,861); el del PCA '
                            'cartografió un 38 % más de borreguil.\n\n'
                            'Recomendado: entrenar con cada opción y quedarse con la que mejor '
                            'valide (la comparación sale al final de «3 · Modelo»).',
                            'Many variables say almost the same (the greenness indices, for '
                            'instance). There are two ways of keeping fewer:\n\n'
                            'With VIF: the VIF of a variable says how much of it can be '
                            'guessed from all the others together (5 means 80 %; 10, 90 %). '
                            'The one most correlated with the others is removed, the VIF is '
                            'computed again, and so on while any is above the threshold.\n\n'
                            'With PCA: the PCA summarises the variables in a few '
                            'independent axes. One variable is kept from each important axis '
                            '(eigenvalue above 1): among those strongly tied to it, the one '
                            'that best separates the categories on its own.\n\n'
                            'Both are computed with the values of the samples, which are read '
                            'under "2 · Table". When you press Apply, which variables stay and '
                            'why is shown right here, and only those stay selected above. In '
                            'Sierra Nevada, the VIF kept 14 variables and the PCA with '
                            'elevation 4, with the same accuracy (F1 0.865 and 0.861); the PCA '
                            'one mapped 38 % more borreguil.\n\n'
                            'Recommended: train with each option and keep the one that '
                            'validates better (the comparison is at the end of "3 · Model").'))
                    _vif_si, _pca_si = _metodo == 'vif', _metodo == 'pca'
                    _vif_u, _vif_f = float(mr.VIF_UMBRAL), list(mr.VIF_FIJAS)
                    # _sel_vig: lo aplicado con el botón, si sigue valiendo (si no, None)
                    _pca_c, _sel_vig = float(mr.PCA_CARGA), None
                    if _metodo != 'ninguno':
                        # Los valores de partida se ponen justo antes de dibujar los
                        # controles (si no, el navegador no los enseña); las fijas, solo
                        # las elegidas.
                        st.session_state.setdefault('mrf_vif_umbral', float(mr.VIF_UMBRAL))
                        st.session_state.setdefault('mrf_pca_carga', float(mr.PCA_CARGA))
                        _fv = st.session_state.get('mrf_vif_fijas')
                        if _fv is None:
                            st.session_state['mrf_vif_fijas'] = [
                                v for v in mr.VIF_FIJAS if v in _predictores]
                        elif any(v not in _predictores for v in _fv):
                            st.session_state['mrf_vif_fijas'] = [v for v in _fv
                                                                 if v in _predictores]
                        _nom_met = (i18n.pick('el VIF', 'the VIF') if _vif_si
                                    else i18n.pick('el PCA', 'the PCA'))
                        # De qué variables parte: las que había marcadas antes de
                        # aplicarlo (si ya se aplicó y siguen marcadas las que quedaron) o
                        # las marcadas ahora.
                        _base = list(_antes['vars']) if _antes else list(_predictores)
                        _tab_base = (None if _tab_todas is None else _tab_todas
                                     if _base == list(_predictores)
                                     else mr.construir_tabla(_res, _leidos, _base))
                        _lista = bool(_tab_base is not None and _tab_base['listo']
                                      and len(_tab_base['filas']) >= 3)
                        _vq1, _vq2, _vq3 = st.columns([1.1, 2.2, 1.3],
                                                      vertical_alignment='bottom')
                        if _vif_si:
                            _vif_u = _vq1.number_input(
                                i18n.pick('Umbral del VIF', 'VIF threshold'),
                                min_value=1.5, max_value=100.0, step=0.5, key='mrf_vif_umbral',
                                help=i18n.pick(
                                    'Se quitan variables mientras alguna tenga un VIF mayor '
                                    'que este. Con 10, ninguna de las que quedan se explica '
                                    'en más del 90 % con las demás; con 5, en más del 80 %. '
                                    'Cuanto más bajo, más variables se quitan.\n\n'
                                    'Recomendado: 10; 5 si quieres quedarte con menos.',
                                    'Variables are removed while any has a VIF above this '
                                    'one. With 10, none of those left is explained more than '
                                    '90 % by the others; with 5, more than 80 %. The lower it '
                                    'is, the more variables are removed.\n\n'
                                    'Recommended: 10; 5 if you want to keep fewer.'))
                        else:
                            _pca_c = _vq1.number_input(
                                i18n.pick('Correlación mínima con el eje',
                                          'Minimum correlation with the axis'),
                                min_value=0.3, max_value=0.95, step=0.05, key='mrf_pca_carga',
                                help=i18n.pick(
                                    'De cada eje se elige entre las variables cuya '
                                    'correlación con él (en valor absoluto) llega a esta. '
                                    'Cuanto más alta, menos candidatas y más claro qué '
                                    'representa cada eje.\n\n'
                                    'Recomendado: 0,65, el de la comparación de Sierra Nevada.',
                                    'From each axis, the choice is among the variables whose '
                                    'correlation with it (in absolute value) reaches this one. '
                                    'The higher it is, the fewer candidates and the clearer '
                                    'what each axis stands for.\n\n'
                                    'Recommended: 0.65, the one of the Sierra Nevada '
                                    'comparison.'))
                        _vif_f = _vq2.multiselect(
                            i18n.pick('Variables que se quedan siempre',
                                      'Variables that always stay'),
                            _predictores, key='mrf_vif_fijas', format_func=_var_m,
                            help=i18n.pick(
                                'Estas no se quitan nunca (con el VIF, sí cuentan al calcular '
                                'el de las demás; con el PCA, quedan fuera de él y se '
                                'añaden). Viene la altitud: en cualquier sitio, los '
                                'borreguiles son prados húmedos de alta montaña.\n\n'
                                'Recomendado: la altitud.',
                                'These are never removed (with the VIF, they do count when '
                                'computing that of the others; with the PCA, they are left '
                                'out of it and added). Elevation comes selected: anywhere, '
                                'borreguiles are wet high-mountain meadows.\n\n'
                                'Recommended: elevation.'))
                        _fijas_t = ([v for v in _vif_f if v in _tab_base['predictores']]
                                    if _tab_base is not None else [])
                        # ---- Aplicar (25-11-2026, la autora: «que haya un botón de aplicar
                        # y te muestre en esta misma pestaña qué variables se han quedado y
                        # algún output del análisis para entender por qué»). Lo aplicado se
                        # guarda por nivel y método, y vale mientras no cambien sus ajustes,
                        # las variables elegidas ni las muestras (o sus valores). Si
                        # cambian, hay que volver a aplicarlo; mientras, se usan todas las
                        # variables elegidas.
                        _firma = (_firma_sel(_metodo, _vif_u if _vif_si else _pca_c, _fijas_t,
                                             _tab_base) if _lista else None)
                        _aplicadas = st.session_state.setdefault('mrf_sel_aplicada', {})
                        _clave_ap = f'{_nivel}|{_metodo}'
                        with _vq3.container(horizontal=True, vertical_alignment='center',
                                            gap='small'):
                            st.button(i18n.pick(f'Aplicar {_nom_met}', f'Apply {_nom_met}'),
                                      key='mrf_aplicar_sel', type='primary', disabled=not _lista,
                                      on_click=lambda: st.session_state.__setitem__(
                                          '_mrf_aplicar', True))
                            ayuda_q('q_mapa_aplicar',
                                    'Calcula qué variables se quedan con los valores de las '
                                    'muestras y deja marcadas arriba solo esas. El modelo, el '
                                    'mapa y el peso usarán solo esas. Para volver a las de '
                                    'antes, «Volver a marcar las de antes», junto a las '
                                    'variables.',
                                    'Works out which variables stay from the values of the '
                                    'samples and leaves only those selected above. The model, '
                                    'the map and the weights will use only those. To go back '
                                    'to the previous ones, "Select the previous ones again", '
                                    'next to the variables.')
                        _ap = _aplicadas.get(_clave_ap)
                        if (_ap and _firma is not None and _ap['firma'] == _firma
                                and mr.ordenar_predictores(_ap['sel']['quedan']) == _predictores):
                            _sel_vig = _ap
                        _n_todas = len(_predictores)
                        if _sel_vig is None:
                            if _tab_todas is None:
                                st.caption(i18n.pick(
                                    'Para aplicarlo hacen falta los valores de las muestras: '
                                    'léelos en «2 · Tabla» (botón «Leer las variables…») y '
                                    f'vuelve aquí. Mientras, se usan las {_n_todas} variables.',
                                    'Applying it needs the values of the samples: read them '
                                    'under "2 · Table" ("Read the variables…" button) and come '
                                    f'back here. Meanwhile, the {_n_todas} variables are used.'))
                            elif not _lista:
                                st.caption(i18n.pick(
                                    'Se podrá aplicar cuando la tabla de entrenamiento esté '
                                    f'lista (mira «2 · Tabla»). Mientras, se usan las '
                                    f'{_n_todas} variables.',
                                    'It can be applied once the training table is ready (see '
                                    f'"2 · Table"). Meanwhile, the {_n_todas} variables are '
                                    'used.'))
                            elif _ap:
                                st.warning(i18n.pick(
                                    'Has cambiado algo desde que lo aplicaste (las variables, '
                                    'el ajuste, las que se quedan siempre o las muestras): '
                                    f'vuelve a pulsar «Aplicar {_nom_met}». Mientras, se usan '
                                    f'las {_n_todas} variables.',
                                    'Something has changed since you applied it (the '
                                    'variables, the setting, those that always stay or the '
                                    f'samples): press "Apply {_nom_met}" again. Meanwhile, the '
                                    f'{_n_todas} variables are used.'))
                            else:
                                st.markdown(i18n.pick(
                                    'Al aplicarlo sale aquí qué variables se quedan y por qué, '
                                    'variable a variable.',
                                    'When you apply it, which variables stay and why shows up '
                                    'here, variable by variable.'))
                        else:
                            _mostrar_seleccion(_sel_vig, 'vif' if _vif_si else 'pca',
                                               _vif_u if _vif_si else _pca_c)

        # ------------------------------------------ 2 · Tabla de entrenamiento
        # El valor de cada variable se lee en el PÍXEL de 10 m de cada muestra, de la
        # misma pila de Earth Engine que después se clasificará. (Lo ya leído de cada
        # muestra, _lect/_leidos/_pend, se calcula en «1 · Datos», que lo usa para el VIF
        # y el PCA.)
        with _t_tabla, st.container(key='tarjeta_mapa_tabla'):
            st.markdown('##### ' + i18n.pick('Tabla de entrenamiento', 'Training table'))
            st.markdown(i18n.pick(
                'Lee el valor de cada variable en el píxel de 10 m de cada muestra: el mismo '
                'píxel que después se clasifica. Usa la cuenta de Earth Engine y los años del '
                'panel de analizar.',
                'Reads the value of each variable at the 10 m pixel of each sample: the same '
                'pixel that is classified afterwards. It uses the Earth Engine account and the '
                'years of the analysis panel.'))
            # El proyecto de Earth Engine es el del panel lateral (ya no se vuelve a pedir
            # aquí, 24-11-2026). Si allí está elegido Microsoft Planetary Computer, el último con
            # el que Earth Engine funcionó. Vacío: la cuenta compartida de MonitorEO
            # (26-11-2026), que también vale para leer las variables y hacer el mapa.
            _proy = gee_project.strip() or leer_ajuste('gee_project', '')
            _rm = st.session_state.pop('_mrf_msg', None)
            if _rm:
                (st.success if _rm[0] == 'ok' else st.warning)(_rm[1])

            _puede = bool(_res['listo'] and _predictores and _anios)
            if not _res['listo']:
                st.caption(i18n.pick(
                    'Antes hay que resolver lo que se indica en «1 · Datos».',
                    'First resolve what is flagged under "1 · Data".'))
            elif not _anios:
                st.error(i18n.pick('No entiendo los años de «1 · Analizar».',
                                   'The years in "1 · Analyse" could not be understood.'))
            # Los dos botones, siempre a la vista (la demo): leer lo que falta y volver a
            # leerlo todo.
            _n_leer = len(_pend) or len(_usadas)
            with st.container(horizontal=True, vertical_alignment='center', gap='small',
                              key='mapa_botones_leer'):
                _leer = st.button(
                    i18n.pick(f'Leer las variables de {_n_leer} muestra'
                              f"{'' if _n_leer == 1 else 's'} en Earth Engine",
                              f'Read the variables of {_n_leer} sample'
                              f"{'' if _n_leer == 1 else 's'} from Earth Engine"),
                    key='mrf_leer', type='primary', disabled=not (_puede and _pend))
                _releer = st.button(i18n.pick('Volver a leer todas', 'Read them all again'),
                                    key='mrf_releer',
                                    disabled=not (_puede and _usadas and _lect))
                ayuda_q('q_mapa_volver_leer',
                        'Solo hace falta si crees que una lectura salió mal: los valores no '
                        'cambian mientras no cambien los años.',
                        'Only needed if you think a reading went wrong: the values do not '
                        'change unless the years do.')
            if _releer:
                st.session_state.pop('mrf_lectura', None)
                st.rerun()
            if _leer:
                with st.status(i18n.pick('Leyendo en Earth Engine…',
                                         'Reading from Earth Engine…'), expanded=True) as _stt:
                    _ok, _msg = geb.initialize(
                        _proy.strip(), service_account_json=get_secret('EE_SERVICE_ACCOUNT_KEY',
                                                                      None))
                    if not _ok:
                        _stt.update(label=i18n.pick('Earth Engine no está disponible',
                                                    'Earth Engine is not available'),
                                    state='error')
                        st.error(_msg)
                    else:
                        guardar_ajuste('gee_project', _proy.strip())
                        _barra = barra_progreso(i18n.pick(
                            'Leyendo las variables en Earth Engine…',
                            'Reading the variables from Earth Engine…'))
                        try:
                            _vals, _fall, _inf = geb.leer_predictores(
                                [(float(f['lon']), float(f['lat'])) for f in _pend],
                                _anios, epsg=_epsg,
                                progress=lambda fr: _barra.progress(max(0.0, min(1.0, fr))))
                        except Exception as _e:
                            _stt.update(label=i18n.pick('La lectura ha fallado',
                                                        'The reading failed'), state='error')
                            st.error(i18n.pick(
                                f'Earth Engine no ha devuelto los datos: {_e}',
                                f'Earth Engine did not return the data: {_e}'))
                        else:
                            from datetime import datetime as _dt
                            _nuevo = _lect or {'anios': _anios, 'epsg': _epsg, 'valores': {}}
                            _nuevo['info'] = _inf
                            _nuevo['cuando'] = _dt.now().isoformat(timespec='seconds')
                            for _f, _v in zip(_pend, _vals):
                                if _v is not None:      # los fallidos quedan pendientes
                                    _nuevo['valores'][mr.clave_punto(_f['lon'], _f['lat'])] = _v
                            st.session_state['mrf_lectura'] = _nuevo
                            _n_ok = len(_pend) - _fall
                            st.session_state['_mrf_msg'] = (
                                ('ok', i18n.pick(f'{_n_ok} muestras leídas.',
                                                 f'{_n_ok} samples read.'))
                                if not _fall else
                                ('aviso', i18n.pick(
                                    f'{_n_ok} muestras leídas; {_fall} no se pudieron '
                                    'consultar. Pulsa de nuevo para reintentarlas.',
                                    f'{_n_ok} samples read; {_fall} could not be queried. '
                                    'Press again to retry them.')))
                            st.rerun()

            # La tabla, con lo que ya está leído
            _tab = None
            if _tab_todas is not None:          # hecha en «1 · Datos», con lo leído
                _tab = _tab_todas
                _nt = len(_tab['filas'])
                # El ID, con solo su número, como en «Revisar puntos» (25-11-2026, la
                # autora: «aquí también debería poner id 1,2,3»). El completo sigue en el
                # CSV que se descarga.
                _ids_tab = ([str(f['ID']) for f in _tab['filas']]
                            + [str(a['ID']) for a in _tab['apartadas']])
                _corto_tab = dict(zip(_ids_tab, bp.abreviar_ids(
                    bp.nombres_de_punto(_ids_tab, punto=''))))
                # (las que fallaron al leerlas están entre las apartadas, «sin leer»)
                _n_fallan = len(_tab['apartadas'])
                with st.container(key='mapa_ok_tabla'):
                    st.markdown('**' + i18n.pick(
                        f"{_nt} muestra{'' if _nt == 1 else 's'} con sus "
                        f"{len(_tab['predictores'])} variables",
                        f"{_nt} sample{'' if _nt == 1 else 's'} with their "
                        f"{len(_tab['predictores'])} variables") + '**'
                        + (i18n.pick(f' · {_n_fallan} no entran: mira «Ver cuáles fallaron»',
                                     f' · {_n_fallan} left out: see "See which ones failed"')
                           if _n_fallan else ''))
                    with st.expander(i18n.pick('Ver cuáles fallaron',
                                               'See which ones failed')):
                        if not _n_fallan:
                            st.markdown(i18n.pick(
                                'Ninguna. Si alguna fallara (por ejemplo, porque en esos años '
                                'no hay imágenes sin nubes en su píxel), saldría aquí con el '
                                'motivo, y «Volver a leer todas» la intentaría otra vez.',
                                'None. If any failed (for instance, because in those years '
                                'there are no cloud-free images at its pixel), it would show '
                                'here with the reason, and "Read them all again" would try it '
                                'again.'))
                        else:
                            if _tab['apartadas']:
                                _cuenta = Counter(a['motivo'] for a in _tab['apartadas'])
                                st.markdown(i18n.pick('**Muestras que no entran en la tabla:**',
                                                      '**Samples left out of the table:**')
                                            + ''.join(f'\n- {n} — {mr.motivo_tabla(k, _en)}'
                                                      for k, n in _cuenta.most_common()))
                                st.dataframe(pd.DataFrame([{
                                    'ID': _corto_tab.get(str(a['ID']), str(a['ID'])),
                                    i18n.pick('Categoría', 'Category'): _cat(a['categoria']),
                                    i18n.pick('Motivo', 'Reason'):
                                        mr.motivo_tabla(a['motivo'], _en),
                                    i18n.pick('Variables sin dato', 'Variables without data'):
                                        ', '.join(_var_m(x) for x in a['faltan'][:6])
                                        + (' …' if len(a['faltan']) > 6 else '')}
                                    for a in _tab['apartadas']]), hide_index=True,
                                    width='stretch')
                            if _pend:
                                st.markdown(i18n.pick(
                                    f"{len(_pend)} muestra{'' if len(_pend) == 1 else 's'} "
                                    'sin leer en Earth Engine: pulsa «Leer las variables…» '
                                    'para intentarlo otra vez.',
                                    f"{len(_pend)} sample{'' if len(_pend) == 1 else 's'} not "
                                    'read from Earth Engine yet: press "Read the variables…" '
                                    'to try again.'))
                    with st.expander(i18n.pick('Ver la tabla de entrenamiento',
                                               'See the training table')):
                        # Qué variables salen, según el caso: todas, las que dejó el VIF o
                        # el PCA, o las marcadas a mano (25-11-2026, la autora: «y en la otra
                        # que muestra las 34, las del vif, o las del pca pues que en algún
                        # sitio lo diga según el caso»).
                        _n_pred, _n_todas = len(_tab['predictores']), len(mr.PREDICTORES)
                        _ant_t = st.session_state.get('mrf_vars_antes')
                        if _ant_t and set(_tab['predictores']) == set(_ant_t['resultado']):
                            _met_t = (i18n.pick('el VIF', 'the VIF')
                                      if _ant_t['metodo'] == 'vif'
                                      else i18n.pick('el PCA', 'the PCA'))
                            _cuales = i18n.pick(
                                f"Salen las **{_n_pred} variables que dejó {_met_t}**, de "
                                f"las {len(_ant_t['vars'])} que había marcadas (se aplicó en "
                                '«1 · Datos»).',
                                f"These are the **{_n_pred} variables {_met_t} kept**, out of "
                                f"the {len(_ant_t['vars'])} that were selected (applied under "
                                '"1 · Data").')
                        elif _n_pred == _n_todas:
                            _cuales = i18n.pick(
                                f'Salen **las {_n_todas} variables**: no se ha quitado '
                                'ninguna, ni con el VIF ni con el PCA.',
                                f'These are **all {_n_todas} variables**: none has been '
                                'removed, neither with the VIF nor with the PCA.')
                        else:
                            _cuales = i18n.pick(
                                f'Salen las **{_n_pred} variables que tienes marcadas** en '
                                f'«1 · Datos», de las {_n_todas} que hay. No las ha elegido el '
                                'VIF ni el PCA.',
                                f'These are the **{_n_pred} variables you have selected** '
                                f'under "1 · Data", out of the {_n_todas} there are. Neither '
                                'the VIF nor the PCA chose them.')
                        st.caption(_cuales + i18n.pick(
                            ' Cada fila es una muestra: el número del punto, si es borreguil, '
                            'su zona (la de la validación) y el valor de cada variable en su '
                            'píxel de 10 m.',
                            ' Each row is a sample: the point number, whether it is a '
                            'borreguil, its zone (the one of the validation) and the value of '
                            'each variable at its 10 m pixel.'))
                        _dft = pd.DataFrame(_tab['filas'])
                        if len(_dft):
                            _zonas_t, _ = mr.grupos_espaciales(_tab, rows)
                            _k_mues = i18n.pick('Muestra', 'Sample')
                            _k_bor = ('Borreguil' if _nivel == mr.NIVEL_SI_NO
                                      else i18n.pick('Categoría', 'Category'))
                            _k_zon = i18n.pick('Zona', 'Zone')
                            _cab_t = pd.DataFrame({
                                _k_mues: [_corto_tab.get(str(i), str(i)) for i in _dft['ID']],
                                _k_bor: [(i18n.pick('Sí', 'Yes') if c != mr.AUSENCIA else 'No')
                                         if _nivel == mr.NIVEL_SI_NO else _cat(c)
                                         for c in _dft['categoria']],
                                _k_zon: [_cat(z.replace('cuenca', i18n.pick('cuenca', 'basin'))
                                              .replace('celda', i18n.pick('celda', 'cell')))
                                         for z in _zonas_t]})
                            _vals_t = _dft[list(_tab['predictores'])].rename(
                                columns={v: _var_m(v) for v in _tab['predictores']})
                            st.dataframe(_con_coma(pd.concat([_cab_t, _vals_t], axis=1)),
                                         hide_index=True, width='stretch',
                                         column_config={_k_mues: st.column_config.Column(
                                             pinned=True)})
                    st.download_button(
                        label=i18n.pick('Descargar tabla_entrenamiento.csv',
                                        'Download tabla_entrenamiento.csv'),
                        data=mr.tabla_entrenamiento_csv(_tab).encode('utf-8-sig'),
                        file_name=f'tabla_entrenamiento_{_nivel}.csv', mime='text/csv',
                        key='mrf_dl_tabla', type='tertiary')
                if _tab['errores']:
                    st.error(i18n.pick('**Con esta tabla todavía no se puede entrenar:**',
                                       '**This table is not enough to train yet:**')
                             + ''.join('\n- ' + mr.mensaje(e, _en, i18n.tv, nivel=_nivel_mod)
                                       for e in _tab['errores']))

                # Control de calidad de las variables: avisa, no quita nada
                _hall = mr.calidad(_tab)
                if _hall:
                    _max = 10
                    _nh = len(_hall)
                    st.warning(i18n.pick(
                        f"**Control de calidad de las variables: {_nh} "
                        f"aviso{'' if _nh == 1 else 's'}**\n\nNo se ha quitado ninguna: si "
                        'quieres prescindir de alguna, desmárcala en «1 · Datos».',
                        f"**Quality check of the variables: {_nh} "
                        f"warning{'' if _nh == 1 else 's'}**\n\nNone has been removed: if "
                        'you want to drop one, deselect it under "1 · Data".')
                        + ''.join('\n- ' + mr.mensaje_calidad(h, _en, nombre=_var_m, num=_num)
                                  for h in _hall[:_max])
                        + (i18n.pick(f'\n- … y {len(_hall) - _max} más.',
                                     f'\n- … and {len(_hall) - _max} more.')
                           if len(_hall) > _max else ''))
                elif _nt:
                    st.markdown('<span style="color:#A3D9B5">' + i18n.pick(
                        'Control de calidad de las variables: sin incidencias.',
                        'Quality check of the variables: nothing to report.') + '</span>',
                        unsafe_allow_html=True)

                # ---- Variables correlacionadas (VIF o PCA). Se eligen y se aplican en
                # «1 · Datos», que enseña cuáles se quedan y por qué (25-11-2026); aquí se
                # sigue con la tabla de las que se quedan, que es la que usan el modelo, el
                # mapa y el peso.
                if _metodo != 'ninguno' and _sel_vig is not None:
                    _s_ap = _sel_vig['sel']
                    # La tabla con la que se sigue: solo con las que se quedan (puede
                    # recuperar muestras a las que solo les faltaba una de las quitadas).
                    _tab = mr.construir_tabla(_res, _leidos, _s_ap['quedan'])
                    if _vif_si:
                        _tab['vif'] = {'metodo': 'vif', 'umbral': float(_vif_u),
                                       'fijas': list(_s_ap['fijas']),
                                       'de': _sel_vig['de'],
                                       'quitadas': [q['variable'] for q in _s_ap['quitadas']]}
                    else:
                        _tab['pca'] = {'metodo': 'pca', 'carga': float(_pca_c),
                                       'fijas': list(_s_ap['fijas']), 'de': _sel_vig['de'],
                                       'ejes': len(_s_ap['ejes'])}

        # ------------------------------------------------------ 3 · Modelo
        # Validar (por zonas) y entrenar el modelo final son dos procesos
        # distintos: las métricas salen del primero; el mapa, del segundo. Dos tarjetas
        # lado a lado, como en la demo: entrenar, y el resultado de la validación.
        with _t_modelo:
            _est_txt = {
                'pesos': i18n.pick('Más peso a las escasas (recomendado)',
                                   'More weight to the scarce ones (recommended)'),
                'submuestreo': i18n.pick('Menos muestras de las abundantes',
                                         'Fewer samples of the abundant ones'),
                'ambas': i18n.pick('Las dos cosas', 'Both'),
                'ninguna': i18n.pick('No compensar', 'Do not compensate'),
            }
            _mf_txt = {
                'sqrt': i18n.pick('Raíz cuadrada (recomendado)', 'Square root (recommended)'),
                'log2': i18n.pick('Logaritmo', 'Logarithm'),
                'todas': i18n.pick('Todas', 'All'),
            }
            _col_ent, _col_res = st.columns(2, gap='small')
            with _col_ent, st.container(key='tarjeta_mapa_entrenar'):
                st.markdown('##### ' + i18n.pick('Entrenar y validar por zonas',
                                                 'Train and validate by zones'))
                st.markdown(i18n.pick(
                    'Cada cuenca (o celda de unos 5 km) se predice con un modelo que no ha '
                    'visto ningún punto de ella. Después se entrena el modelo final con todas '
                    'las muestras.',
                    'Each basin (or cell of about 5 km) is predicted by a model that has seen '
                    'no point from it. The final model is then trained with all the '
                    'samples.'))
                # Qué es cada parámetro y qué valor conviene: va en la ayuda (?) de cada
                # uno, para poder decidir sin salir de la app.
                with st.expander(i18n.pick('Parámetros del modelo', 'Model parameters')):
                    _pa, _pb = st.columns(2)
                    _p_arboles = _pa.number_input(
                        i18n.pick('Nº de árboles', 'Nº of trees'), 50, 2000,
                        mr.PARAMS_DEFECTO['n_estimators'], 50, key='mrf_p_arboles',
                        help=i18n.pick(
                            'El modelo es un conjunto de árboles de decisión que votan. '
                            'Con más árboles el resultado es más estable (cambia menos de '
                            'una ejecución a otra), pero tarda más; a partir de unos '
                            'cientos apenas mejora.\n\n'
                            'Recomendado: 500. Entre 300 y 1000 va bien. Con menos de '
                            '100 las cifras bailan.',
                            'The model is a set of decision trees that vote. More trees '
                            'give a steadier result (it changes less from one run to the '
                            'next) but take longer; beyond a few hundred it hardly '
                            'improves.\n\n'
                            'Recommended: 500. Between 300 and 1000 works well. Below '
                            '100 the figures wobble.'))
                    _p_prof = _pb.number_input(
                        i18n.pick('Profundidad máxima (0 = sin límite)',
                                  'Maximum depth (0 = no limit)'), 0, 50, 0, 1,
                        key='mrf_p_prof',
                        help=i18n.pick(
                            'Cuántas preguntas seguidas puede hacer cada árbol antes de '
                            'decidir. Sin límite, el árbol crece hasta separar del todo las '
                            'muestras, que es lo habitual en un Random Forest.\n\n'
                            'Recomendado: 0 (sin límite). Ponle un límite, entre 5 y 10, '
                            'solo si tienes muy pocas muestras y la validación sale mal: '
                            'los árboles más cortos se fijan menos en los casos sueltos.',
                            'How many questions in a row each tree may ask before '
                            'deciding. With no limit, the tree grows until it separates '
                            'the samples completely, which is the usual way in a Random '
                            'Forest.\n\n'
                            'Recommended: 0 (no limit). Set a limit, between 5 and 10, '
                            'only if you have very few samples and the validation comes '
                            'out badly: shorter trees pay less attention to odd cases.'))
                    _pa, _pb = st.columns(2)
                    _p_hoja = _pa.number_input(
                        i18n.pick('Muestras mínimas por hoja', 'Minimum samples per leaf'),
                        1, 20, mr.PARAMS_DEFECTO['min_samples_leaf'], 1, key='mrf_p_hoja',
                        help=i18n.pick(
                            'Cada árbol va repartiendo las muestras en grupos cada vez más '
                            'pequeños. Este es el tamaño mínimo del grupo final («hoja»). '
                            'Con 1, el árbol puede aprenderse de memoria cada muestra, '
                            'también las raras o mal etiquetadas; al subirlo, generaliza '
                            'más.\n\n'
                            'Recomendado: 2 si tienes decenas de muestras por categoría; '
                            'entre 3 y 5 si tienes cientos. Si lo subes mucho, las '
                            'categorías con pocas muestras dejan de distinguirse.',
                            'Each tree keeps splitting the samples into smaller and '
                            'smaller groups. This is the minimum size of the final group '
                            '("leaf"). With 1, the tree can learn each sample by heart, '
                            'including odd or mislabelled ones; raising it makes the tree '
                            'generalise more.\n\n'
                            'Recommended: 2 if you have tens of samples per category; '
                            'between 3 and 5 if you have hundreds. If you raise it a lot, '
                            'categories with few samples can no longer be told apart.'))
                    _p_semilla = _pb.number_input(
                        i18n.pick('Semilla', 'Seed'), 0, 99999,
                        mr.PARAMS_DEFECTO['random_state'], 1, key='mrf_p_semilla',
                        help=i18n.pick(
                            'El modelo usa el azar para elegir qué muestras y qué '
                            'variables ve cada árbol. La semilla fija ese azar: con la '
                            'misma semilla y los mismos datos sale exactamente el mismo '
                            'modelo, y otra persona puede repetirlo.\n\n'
                            'Recomendado: déjala en 42 y anótala con tus resultados. '
                            'Ningún número es mejor que otro. Si al cambiarla las cifras '
                            'cambian mucho, es que hay pocas muestras.',
                            'The model uses chance to choose which samples and which '
                            'variables each tree sees. The seed fixes that chance: the '
                            'same seed and the same data give exactly the same model, and '
                            'someone else can repeat it.\n\n'
                            'Recommended: leave it at 42 and note it down with your '
                            'results. No number is better than another. If the figures '
                            'change a lot when you change it, there are too few samples.'))
                    _pa, _pb = st.columns(2)
                    _p_mf = _pa.selectbox(
                        i18n.pick('Variables probadas en cada división',
                                  'Variables tried at each split'),
                        list(_mf_txt), key='mrf_p_maxfeat', format_func=lambda o: _mf_txt[o],
                        help=i18n.pick(
                            'En cada división, el árbol no mira todas las variables: '
                            'prueba solo unas cuantas, elegidas al azar, y se queda con la '
                            'mejor. Así los árboles salen distintos entre sí, que es lo '
                            'que hace que juntos acierten más.\n\n'
                            'Recomendado: raíz cuadrada, lo habitual para clasificar. Con '
                            '«todas», los árboles se parecen mucho entre sí y el resultado '
                            'suele ser peor.',
                            'At each split the tree does not look at every variable: it '
                            'tries only a few, chosen at random, and keeps the best. That '
                            'makes the trees differ from one another, which is what makes '
                            'them more accurate together.\n\n'
                            'Recommended: square root, the usual choice for '
                            'classification. With "all", the trees look very much alike '
                            'and the result is usually worse.'))
                    _p_bal = _pb.selectbox(
                        i18n.pick('Si unas categorías tienen muchas más muestras',
                                  'If some categories have many more samples'),
                        list(mr.ESTRATEGIAS), key='mrf_p_balance',
                        format_func=lambda o: _est_txt[o],
                        help=i18n.pick(
                            'Si una categoría tiene muchas más muestras que las demás, el '
                            'modelo tiende a elegirla siempre.\n\n'
                            'Dar más peso a las escasas (recomendado): cada muestra de una '
                            'categoría escasa cuenta más. No se pierde ninguna.\n'
                            'Usar menos muestras de las abundantes: se aparta al azar parte '
                            'de ellas.\n'
                            'Las dos cosas: para desequilibrios muy fuertes.\n'
                            'No compensar: solo si todas tienen un número parecido de '
                            'muestras.\n\n'
                            'No se generan muestras artificiales en ningún caso, y se valida '
                            'siempre con todas.',
                            'If one category has many more samples than the others, the '
                            'model tends to choose it every time.\n\n'
                            'Give more weight to the scarce ones (recommended): each sample '
                            'of a scarce category counts more. None is lost.\n'
                            'Use fewer samples of the abundant ones: part of them is set '
                            'aside at random.\n'
                            'Both: for very strong imbalances.\n'
                            'Do not compensate: only if all of them have a similar number '
                            'of samples.\n\n'
                            'No artificial samples are generated in any case, and '
                            'validation always uses all of them.'))
                    _p_razon = st.number_input(
                        i18n.pick('Como mucho, cuántas veces la categoría más escasa (si usas '
                                  'menos muestras de las abundantes)',
                                  'At most, how many times the scarcest category (if you use '
                                  'fewer samples of the abundant ones)'),
                        1.0, 10.0, mr.RAZON_MAXIMA, 0.5, key='mrf_p_razon',
                        disabled=_p_bal not in ('submuestreo', 'ambas'), width=200,
                        help=i18n.pick(
                            'Tope de muestras de cada categoría abundante, como múltiplo '
                            'de la más escasa. Con 3 y una categoría escasa de 10 '
                            'muestras, de las demás se usan como mucho 30, elegidas al '
                            'azar con la semilla.\n\n'
                            'Recomendado: 3.',
                            'Cap on the samples of each abundant category, as a multiple '
                            'of the scarcest one. With 3 and a scarce category of 10 '
                            'samples, at most 30 of each of the others are used, chosen '
                            'at random with the seed.\n\n'
                            'Recommended: 3.'))
                _params = {'n_estimators': int(_p_arboles),
                           'max_features': None if _p_mf == 'todas' else _p_mf,
                           'max_depth': int(_p_prof) or None,
                           'min_samples_leaf': int(_p_hoja), 'random_state': int(_p_semilla)}

                _listo_m = bool(_tab and _tab['listo'])
                if not _listo_m:
                    st.caption(i18n.pick(
                        'Antes hace falta una tabla de entrenamiento válida («2 · Tabla»).',
                        'A valid training table is needed first ("2 · Table").'))
                if st.button(i18n.pick('Entrenar y validar', 'Train and validate'),
                             key='mrf_entrenar', type='primary', disabled=not _listo_m,
                             width='stretch'):
                    with st.spinner(i18n.pick(
                            'Validando por zonas y entrenando el modelo final…',
                            'Validating by zones and training the final model…')):
                        try:
                            _grupos, _ginfo = mr.grupos_espaciales(_tab, rows)
                            _nuevo_m = mr.entrenar_y_validar(_tab, _grupos, _params,
                                                             estrategia=_p_bal,
                                                             razon_maxima=float(_p_razon))
                            _nuevo_m['zonas'] = _ginfo
                            # De qué lectura salió la tabla: el mapa tiene que leer los
                            # píxeles de los mismos años, en la misma rejilla y con las
                            # variables calculadas de la misma manera.
                            _nuevo_m['lectura'] = {'anios': [int(a) for a in _anios],
                                                   'epsg': int(_epsg),
                                                   'version': geb.VERSION_PILA}
                            # y si las variables salieron del VIF o del PCA (None = las de
                            # «1 · Datos»)
                            _nuevo_m['variables'] = _tab.get('vif') or _tab.get('pca')
                            # uno por nivel: se pueden tener los cuatro entrenados a la vez
                            st.session_state.setdefault('mrf_modelos', {})[_nivel] = _nuevo_m
                        except ValueError as _e:
                            st.error(i18n.pick(f'No se ha podido entrenar: {_e}',
                                               f'Training was not possible: {_e}'))


            _mod = st.session_state.get('mrf_modelos', {}).get(_nivel)
            with _col_res, st.container(key='tarjeta_mapa_validacion'):
                st.markdown('##### ' + i18n.pick('Resultado de la validación',
                                                 'Validation result'))
                if not _mod:
                    st.markdown(i18n.pick('Pulsa «Entrenar y validar» para verlo.',
                                          'Press "Train and validate" to see it.'))
                else:
                    if _tab is None or _mod['firma'] != mr.firma_tabla(_tab):
                        st.warning(i18n.pick(
                            'La tabla ha cambiado desde que se entrenó este modelo (otras '
                            'muestras u otras variables). Lo de abajo corresponde al modelo '
                            'anterior: vuelve a entrenar para ponerlo al día.',
                            'The table has changed since this model was trained (other '
                            'samples or other variables). What follows belongs to the '
                            'previous model: train again to bring it up to date.'))
                    _val, _zi = _mod['validacion'], _mod.get('zonas', {})
                    # La cifra grande va arriba, pero depende del umbral, que se elige
                    # debajo: se deja su hueco y se rellena después.
                    _hueco_f1 = st.container()

                    # ---- ¿borreguil o no? La regla y su umbral. De ella salen las cifras
                    # de arriba, el mapa y el peso de las variables. Cambiarla no vuelve a
                    # entrenar nada.
                    _regla = _mod.get('regla') or {}
                    _crit = None
                    if '_validacion' not in _mod:
                        st.info(i18n.pick(
                            'Este modelo se entrenó antes de que la app decidiera «¿borreguil '
                            'o no?» con un umbral. Pulsa otra vez **Entrenar y validar**.',
                            'This model was trained before the app decided "borreguil or '
                            'not?" with a threshold. Press **Train and validate** again.'))
                    elif _regla.get('umbral') is None:
                        st.caption(i18n.pick(
                            'Este modelo no tiene la categoría «no borreguil», así que no '
                            'decide si algo es borreguil o no: a cada caso le pone su tipo '
                            'más probable.',
                            'This model has no "not a borreguil" category, so it does not '
                            'decide whether something is a borreguil or not: it gives each '
                            'case its most probable type.'))
                    else:
                        _crit_txt = {
                            'f1': i18n.pick('Mayor F1 (recomendado)',
                                            'Highest F1 (recommended)'),
                            'tss': i18n.pick('Mayor TSS', 'Highest TSS'),
                            'manual': i18n.pick('Lo fijo yo', 'I set it myself')}
                        _crit = st.radio(
                            i18n.pick('¿Borreguil o no? Cómo se elige el umbral',
                                      'Borreguil or not? How the threshold is chosen'),
                            list(mr.CRITERIOS), key='mrf_umbral_modo', horizontal=True,
                            format_func=lambda c: _crit_txt[c],
                            help=i18n.pick(
                                'El modelo da a cada píxel la probabilidad de cada '
                                'categoría. Para decidir si es borreguil se suman las de los '
                                'tipos (es lo mismo que 1 − la de «no borreguil») y se '
                                'compara con un umbral. Si llega, es borreguil, y después se '
                                'elige su tipo.\n\n'
                                'El de mayor F1: de 39 umbrales, de 0,025 a 0,975, el que '
                                'mejor equilibra en la validación encontrar borreguiles y no '
                                'dar por borreguil lo que no lo es.\n\n'
                                'El que mejor separa (TSS): el que da más TSS en la '
                                'validación (sensibilidad + especificidad − 1). Como hay '
                                'mucho más «no borreguil» que borreguil, suele elegir un '
                                'umbral bajo, y el mapa se llena de pasto que no es '
                                'borreguil.\n\n'
                                'Lo fijo yo: el valor que pongas.\n\n'
                                'Recomendado: el de mayor F1.',
                                'The model gives each pixel the probability of each '
                                'category. To decide whether it is a borreguil, those of the '
                                'types are added up (the same as 1 − that of "not a '
                                'borreguil") and compared with a threshold. If it reaches '
                                'it, it is a borreguil, and its type is then chosen.\n\n'
                                'The one with the highest F1: out of 39 thresholds, from '
                                '0.025 to 0.975, the one that in the validation best '
                                'balances finding borreguiles and not calling borreguil what '
                                'is not.\n\n'
                                'The one that separates best (TSS): the one with the '
                                'highest TSS in the validation (sensitivity + specificity − '
                                '1). As there is far more "not a borreguil" than borreguil, '
                                'it tends to pick a low threshold, and the map fills up with '
                                'grassland that is not borreguil.\n\n'
                                'I set it myself: the value you enter.\n\n'
                                'Recommended: the one with the highest F1.'))
                        _u_manual = None
                        if _crit == 'manual':
                            _u_manual = float(st.slider(
                                i18n.pick('Umbral', 'Threshold'), 0.10, 0.90, 0.5, 0.025,
                                key='mrf_umbral_valor', format='%.3f',
                                help=i18n.pick(
                                    'Probabilidad de ser borreguil a partir de la cual un '
                                    'píxel se da por borreguil. Más bajo: se encuentran más '
                                    'borreguiles y se cuelan más cosas que no lo son. Más '
                                    'alto: al revés.\n\n'
                                    'Recomendado: mira antes la gráfica del acierto con el '
                                    'umbral y elige un valor donde el F1 sea alto. 0,5 es '
                                    '«más probable que sí que que no».',
                                    'Probability of being a borreguil from which a pixel '
                                    'counts as one. Lower: more borreguiles are found and '
                                    'more things that are not get through. Higher: the '
                                    'opposite.\n\n'
                                    'Recommended: look at the chart of accuracy against the '
                                    'threshold first and choose a value where F1 is high. '
                                    '0.5 is "more likely than not".')))
                        if _regla.get('criterio') != _crit or (
                                _crit == 'manual'
                                and abs(_regla['umbral'] - _u_manual) > 1e-9):
                            mr.aplicar_regla(_mod, _crit, _u_manual)
                            _regla = _mod['regla']
                        _u_txt = f"{_num(_regla['umbral'])}"
                        with st.expander(i18n.pick('Cómo cambia el acierto con el umbral',
                                                   'How accuracy changes with the '
                                                   'threshold')):
                            import altair as alt
                            _bar = _regla['barrido']
                            _k_um = i18n.pick('umbral', 'threshold')
                            _larga = pd.DataFrame(
                                [{'umbral': b['umbral'], 'medida': m, 'valor': b[k]}
                                 for b in _bar for m, k in (('F1', 'f1'), ('TSS', 'tss'))])
                            _ancha = pd.DataFrame([{
                                'umbral': b['umbral'], 'TSS': b['tss'], 'F1': b['f1'],
                                'sens': b['sensibilidad'], 'espec': b['especificidad']}
                                for b in _bar])
                            _coma = ("replace(format(datum.value, '~r'), '.', ',')" if not _en
                                     else "format(datum.value, '~r')")
                            _ex = alt.X('umbral:Q', title=_k_um,
                                        scale=alt.Scale(domain=[0, 1], nice=False),
                                        axis=alt.Axis(values=[0, 0.25, 0.5, 0.75, 1],
                                                      labelExpr=_coma, grid=False))
                            _ey = alt.Y('valor:Q', title=None,
                                        scale=alt.Scale(domain=[0.1, 1], clamp=True,
                                                        nice=False),
                                        axis=alt.Axis(values=[0.1, 0.5, 1], labelExpr=_coma))
                            # F1 en verde, continua; TSS en azul, discontinua (la demo)
                            _lineas = alt.Chart(_larga).mark_line(strokeWidth=2.5).encode(
                                x=_ex, y=_ey,
                                color=alt.Color('medida:N', legend=None, scale=alt.Scale(
                                    domain=['F1', 'TSS'], range=['#7CC496', '#4FC3F7'])),
                                strokeDash=alt.StrokeDash('medida:N', legend=None,
                                                          scale=alt.Scale(
                                                              domain=['F1', 'TSS'],
                                                              range=[[1, 0], [6, 4]])))
                            # el nombre de cada línea, junto a ella (sin leyenda)
                            _nombres_l = pd.DataFrame([
                                {'umbral': 0.12, 'valor': float(min(
                                    0.97, max(0.15, next(b['f1'] for b in _bar
                                                         if b['umbral'] >= 0.12) + 0.06))),
                                 'medida': 'F1'},
                                {'umbral': 0.18, 'valor': float(min(
                                    0.97, max(0.15, next(b['tss'] for b in _bar
                                                         if b['umbral'] >= 0.18) - 0.06))),
                                 'medida': 'TSS'}])
                            _rotulos = alt.Chart(_nombres_l).mark_text(
                                fontSize=12, fontWeight=600, align='left').encode(
                                x=_ex, y=_ey, text='medida:N',
                                color=alt.Color('medida:N', legend=None, scale=alt.Scale(
                                    domain=['F1', 'TSS'], range=['#7CC496', '#4FC3F7'])))
                            _elegido = pd.DataFrame({'umbral': [_regla['umbral']],
                                                     'rotulo': [f'{_k_um} {_u_txt}']})
                            _raya = alt.Chart(_elegido).mark_rule(
                                color='#F5B83D', strokeWidth=2).encode(x=_ex)
                            # Al pasar el ratón: el umbral más cercano, con todas sus cifras.
                            _cerca = alt.selection_point(
                                name='umbral_cerca', nearest=True, on='pointerover',
                                clear='pointerout', fields=['umbral'], empty=False)
                            _puntos = alt.Chart(_ancha).mark_rule(
                                color='#98A2AB', strokeWidth=1).encode(
                                x=_ex, opacity=alt.condition(_cerca, alt.value(0.6),
                                                             alt.value(0.0)),
                                tooltip=[
                                    alt.Tooltip('umbral:Q', title=_k_um, format='.3f'),
                                    alt.Tooltip('F1:Q', format='.3f'),
                                    alt.Tooltip('TSS:Q', format='.3f'),
                                    alt.Tooltip('sens:Q', format='.3f', title=i18n.pick(
                                        'sensibilidad', 'sensitivity')),
                                    alt.Tooltip('espec:Q', format='.3f', title=i18n.pick(
                                        'especificidad', 'specificity'))]
                            ).add_params(_cerca)
                            st.altair_chart(
                                alt.layer(_lineas, _rotulos, _raya, _puntos).properties(
                                    height=210), width='stretch')
                            st.caption(i18n.pick(
                                'En amarillo, el umbral que se usa. Más bajo: encuentra más '
                                'borreguiles y se cuelan más cosas que no lo son.',
                                'In yellow, the threshold in use. Lower: it finds more '
                                'borreguiles and more things that are not get through.'))

                    # ---- la cifra grande y cómo se ha validado (en su hueco, arriba)
                    with _hueco_f1:
                        _eu = _regla.get('en_umbral') if _regla.get('umbral') is not None \
                            else None
                        if _eu:
                            _f1_txt = f"F1 {_num(_eu['f1'])}"
                            _met_txt = i18n.pick(
                                f"Precisión {_num(_eu['precision'])} · Sensibilidad "
                                f"{_num(_eu['sensibilidad'])} · TSS {_num(_eu['tss'])}",
                                f"Precision {_num(_eu['precision'])} · Sensitivity "
                                f"{_num(_eu['sensibilidad'])} · TSS {_num(_eu['tss'])}")
                            if _nivel != mr.NIVEL_SI_NO:
                                _met_txt += i18n.pick(
                                    f" · F1 medio entre categorías {_num(_mod['macro_f1'])}",
                                    f" · mean F1 across categories {_num(_mod['macro_f1'])}")
                        else:
                            _f1_txt = i18n.pick(f"F1 medio {_num(_mod['macro_f1'])}",
                                                f"Mean F1 {_num(_mod['macro_f1'])}")
                            _met_txt = i18n.pick(
                                f"Exactitud equilibrada {_num(_mod['exactitud_equilibrada'])}"
                                f" · Aciertos sobre el total {_num(_mod['exactitud'])}",
                                f"Balanced accuracy {_num(_mod['exactitud_equilibrada'])}"
                                f" · Overall accuracy {_num(_mod['exactitud'])}")
                        st.markdown(f'<div class="mapa-f1"><b>{_f1_txt}</b>'
                                    f'<span>{_met_txt}</span></div>', unsafe_allow_html=True)
                        _np_v = _val['particiones']
                        if _val['tipo'] == 'espacial':
                            _nz = _val['n_zonas']
                            if _nz == _np_v:
                                if not _zi.get('con_celda'):
                                    _cada = i18n.pick('cada una con una cuenca apartada',
                                                      'each with one basin held out')
                                elif not _zi.get('con_cuenca'):
                                    _cada = i18n.pick(
                                        'cada una con una celda de unos 5 km apartada',
                                        'each with one cell of about 5 km held out')
                                else:
                                    _cada = i18n.pick(
                                        'cada una con una zona apartada (una cuenca o una '
                                        'celda de unos 5 km)',
                                        'each with one zone held out (a basin or a cell of '
                                        'about 5 km)')
                            else:
                                _cada = i18n.pick(
                                    f'con {_nz} zonas apartadas por turnos (cuencas o celdas '
                                    'de unos 5 km)',
                                    f'with {_nz} zones held out in turn (basins or cells of '
                                    'about 5 km)')
                            _validado = i18n.pick(f'Validado por zonas: {_np_v} partes, {_cada}.',
                                                  f'Validated by zones: {_np_v} folds, {_cada}.')
                        else:
                            _validado = i18n.pick(
                                f'Validado sin zonas: {_np_v} partes al azar, respetando la '
                                'proporción de cada categoría.',
                                f'Validated without zones: {_np_v} random folds, keeping the '
                                'proportion of each category.')
                        if _crit:
                            _de_umbral = {'f1': i18n.pick('el de mayor F1', 'the one with the '
                                                          'highest F1'),
                                          'tss': i18n.pick('el de mayor TSS', 'the one with '
                                                           'the highest TSS'),
                                          'manual': i18n.pick('puesto a mano', 'set by hand')}
                            _validado += i18n.pick(f' Umbral: {_u_txt} ({_de_umbral[_crit]}).',
                                                   f' Threshold: {_u_txt} ({_de_umbral[_crit]}).')
                        st.markdown(_validado)
                        if _val['avisos']:
                            st.warning(''.join('- ' + mr.mensaje_validacion(a, _en, i18n.tv)
                                               + '\n' for a in _val['avisos']))
                        if _cercania:
                            st.caption(i18n.pick(
                                'Estas cifras son de lo que hace el modelo: decidir si es '
                                'borreguil y si es húmedo o seco. El ambiente (arroyo, laguna '
                                'o ladera) no entra aquí: lo pone el mapa por cercanía a '
                                'lagunas y cauces (4 · Mapa).',
                                'These figures are for what the model does: deciding whether '
                                'it is a borreguil and whether it is wet or dry. The '
                                'environment (stream, lake or slope) is not part of them: the '
                                'map sets it by closeness to lakes and watercourses (4 · '
                                'Map).'))

                    with st.expander(i18n.pick('Matriz de confusión', 'Confusion matrix')):
                        st.markdown(i18n.pick('Filas: lo que es; columnas: lo que dice el '
                                              'modelo.',
                                              'Rows: what it is; columns: what the model '
                                              'says.'))
                        _cats = [_cat(c['categoria']) for c in _mod['categorias']]
                        _k_real = i18n.pick('es ↓ · dice →', 'is ↓ · says →')
                        _dfc = pd.DataFrame(
                            [{_k_real: _cats[a], **{_cats[b]: _mod['matriz'][a][b]
                                                    for b in range(len(_cats))}}
                             for a in range(len(_cats))])

                        def _pinta_matriz(df):
                            out = pd.DataFrame('', index=df.index, columns=df.columns)
                            for a in range(len(_cats)):
                                for b in range(len(_cats)):
                                    if a == b:
                                        out.iloc[a, b + 1] = ('background-color:#1E3327;'
                                                              'font-weight:700')
                                    elif _mod['matriz'][a][b]:
                                        out.iloc[a, b + 1] = 'color:#F5B83D;font-weight:700'
                            return out
                        st.dataframe(_dfc.style.apply(_pinta_matriz, axis=None),
                                     hide_index=True, width='stretch')
                        if _mod['confusiones']:
                            _ca, _cb, _cn = _mod['confusiones'][0]
                            st.markdown(i18n.pick(
                                f'Lo que más se confunde: «{_cat(_ca)}» tomado por '
                                f'«{_cat(_cb)}»: {_cn}.',
                                f'Most frequent confusion: "{_cat(_ca)}" taken for '
                                f'"{_cat(_cb)}": {_cn}.'))
                        else:
                            st.markdown(i18n.pick(
                                'Ninguna muestra mal clasificada en la validación.',
                                'No sample misclassified in the validation.'))

                    _c4 = st.column_config.NumberColumn()
                    with st.expander(i18n.pick('Resultado de cada parte',
                                               'Result of each fold')):
                        _k_f1 = 'macro-F1'
                        _k_ba = i18n.pick('Exactitud equilibrada', 'Balanced accuracy')
                        st.dataframe(_con_coma(pd.DataFrame([{
                            i18n.pick('Parte', 'Fold'): p['parte'],
                            i18n.pick('Zona apartada', 'Held-out zone'):
                                ', '.join(_cat(z.replace('cuenca', i18n.pick('cuenca', 'basin'))
                                               .replace('celda', i18n.pick('celda', 'cell')))
                                          for z in p['zonas_valida']),
                            i18n.pick('Entrena', 'Trains'): p['n_entrena'],
                            i18n.pick('Valida', 'Validates'): p['n_valida'],
                            _k_f1: p['macro_f1'], _k_ba: p['exactitud_equilibrada']}
                            for p in _mod['pliegues']])), hide_index=True, width='stretch',
                            column_config={_k_f1: _c4, _k_ba: _c4})
                    _mal = [p for p in _mod['predicciones'] if not p['acierta']]
                    if _mal:
                        with st.expander(i18n.pick(
                                f'Ver las {len(_mal)} muestras mal clasificadas',
                                f'See the {len(_mal)} misclassified samples')):
                            _k_conf = i18n.pick('Confianza', 'Confidence')
                            st.dataframe(_con_coma(pd.DataFrame({
                                'ID': bp.abreviar_ids([p['ID'] for p in _mod['predicciones']]),
                                i18n.pick('Zona', 'Zone'):
                                    [_cat(str(p['zona']).replace(
                                        'cuenca', i18n.pick('cuenca', 'basin')).replace(
                                        'celda', i18n.pick('celda', 'cell')))
                                     for p in _mod['predicciones']],
                                i18n.pick('Es', 'Is'):
                                    [_cat(p['real']) for p in _mod['predicciones']],
                                i18n.pick('El modelo dice', 'The model says'):
                                    [_cat(p['predicha']) for p in _mod['predicciones']],
                                _k_conf: [p['confianza'] for p in _mod['predicciones']],
                            })[[not p['acierta'] for p in _mod['predicciones']]]),
                                hide_index=True, width='stretch', column_config={_k_conf: _c4})
                            st.caption(i18n.pick(
                                'Merece la pena mirarlas: una muestra mal clasificada puede '
                                'ser un fallo del modelo o una etiqueta que conviene revisar.',
                                'They are worth a look: a misclassified sample may be a model '
                                'error or a label worth reviewing.'))

                    # ---- ¿Mejoraría con más puntos? La curva con los puntos de esta zona
                    # (25-11-2026, la autora: «¿los puntos mínimos son solo para Sierra
                    # Nevada?»). La de Sierra Nevada, en «1 · Datos», es una referencia;
                    # esta repite la validación de este modelo con menos puntos
                    # (mapeo_rf.curva_propia). Se guarda en el propio modelo: al volver a
                    # entrenar, desaparece.
                    if ('_validacion' in _mod and _tab is not None
                            and _mod['firma'] == mr.firma_tabla(_tab)):
                        with st.container(key='mapa_curva'):
                            st.markdown('**' + i18n.pick('¿Mejoraría con más puntos?',
                                                         'Would more points help?') + '**')
                            _calc = st.button(i18n.pick('Calcular la curva con mis puntos',
                                                        'Compute the curve with my points'),
                                              key='mrf_curva', width='stretch')
                            if _calc:
                                _barra = barra_progreso(i18n.pick(
                                    'Entrenando con menos puntos…',
                                    'Training with fewer points…'))
                                try:
                                    _mod['_curva'] = mr.curva_propia(
                                        _tab, _mod, al_avanzar=lambda f: _barra.progress(f))
                                except ValueError as _e:
                                    st.error(i18n.pick(f'No se ha podido calcular: {_e}',
                                                       f'It could not be computed: {_e}'))
                                _barra.empty()
                            _curva = _mod.get('_curva')
                            if not (_curva and _curva['firma'] == _mod['firma']):
                                st.markdown(i18n.pick(
                                    'Repite la validación con el 25, el 50 y el 75 % de tus '
                                    'puntos y dice si «todavía sube» o «ya casi no sube».',
                                    'It repeats the validation with 25, 50 and 75 % of your '
                                    'points and tells whether it "is still rising" or "has '
                                    'almost stopped rising".'))
                            else:
                                _cc = mr.cifras_curva(_curva, _mod)
                                _md = _cc['medida']
                                _todo, _mit = _cc['filas'][-1], _cc['mitad']
                                st.vega_lite_chart(spec=_recolorear(pp.chart_curva_propia(_cc)),
                                                   width='stretch')
                                _k_parte = i18n.pick('Parte de tus puntos',
                                                     'Share of your points')
                                _k_np = i18n.pick('Puntos con los que aprende',
                                                  'Points it learns from')
                                _k_nb = i18n.pick('de ellos, borreguiles',
                                                  'of which borreguiles')
                                _k_lo = i18n.pick('peor sorteo', 'worst draw')
                                _k_hi = i18n.pick('mejor sorteo', 'best draw')
                                _k_f1m = i18n.pick('F1 medio', 'Mean F1')
                                st.dataframe(_con_coma(pd.DataFrame([{
                                    _k_parte: f"{round(f['fraccion'] * 100)} %",
                                    _k_np: int(round(f['n_puntos'])),
                                    _k_nb: int(round(f['n_borreguiles'])),
                                    **({'AUC': f['auc'], f'AUC · {_k_lo}': f['auc_min'],
                                        f'AUC · {_k_hi}': f['auc_max']}
                                       if _md == 'auc' else {}),
                                    _k_f1m: f['f1']} for f in _cc['filas']])),
                                    hide_index=True, width='stretch')
                                st.caption(i18n.pick(
                                    'Franja: entre el peor y el mejor sorteo de puntos.',
                                    'Band: between the worst and the best draw of points.'))
                                _que = 'AUC' if _md == 'auc' else 'F1'
                                _dicho = i18n.pick(
                                    'De la mitad de tus puntos a todos (el modelo aprende de '
                                    f"unos {round(_mit['n_puntos'])} y de unos "
                                    f"{round(_todo['n_puntos'])} en cada parte), el {_que} "
                                    f"pasa de {_num(_mit[_md])} a {_num(_todo[_md])} "
                                    f"({_num(_cc['ganancia'], signo=True)}).",
                                    'From half of your points to all of them (the model '
                                    f"learns from about {round(_mit['n_puntos'])} and about "
                                    f"{round(_todo['n_puntos'])} in each fold), the {_que} "
                                    f"goes from {_num(_mit[_md])} to {_num(_todo[_md])} "
                                    f"({_num(_cc['ganancia'], signo=True)}).")
                                if _cc['lectura'] == 'sube':
                                    st.info(_dicho + i18n.pick(
                                        ' **Todavía sube**: si vuelves a doblar los puntos, '
                                        'cabe esperar que suba algo parecido, o algo menos. '
                                        'Busca sobre todo en las zonas y los ambientes donde '
                                        'tienes pocos, y revisa tanto borreguiles como '
                                        'ausencias.',
                                        ' **It is still rising**: if you double the points '
                                        'again, expect it to rise by about as much, or a '
                                        'little less. Look above all in the areas and '
                                        'settings where you have few, and review both '
                                        'borreguiles and absences.'))
                                else:
                                    st.success(_dicho + i18n.pick(
                                        ' **Ya casi no sube**: aunque dobles los puntos, si '
                                        'son parecidos a los que tienes, el mapa apenas '
                                        'cambiará. Si el acierto te parece bajo, lo que falta '
                                        'no es cantidad: revisa las muestras mal '
                                        'clasificadas, añade puntos de sitios distintos '
                                        '(otros valles, otras altitudes) o prueba otras '
                                        'variables.',
                                        ' **It has almost stopped rising**: even if you '
                                        'double the points, if they are like the ones you '
                                        'have, the map will hardly change. If the accuracy '
                                        'seems low to you, what is missing is not quantity: '
                                        'review the misclassified samples, add points from '
                                        'different places (other valleys, other altitudes) '
                                        'or try other variables.'))

                    # ---- Con qué variables acierta más. Cada modelo entrenado en la sesión
                    # deja aquí su acierto en la validación por zonas; se comparan los del
                    # mismo nivel y con las mismas muestras (las variables pueden ser otras).
                    _f_mu = mr.firma_muestras(_res)
                    if _tab is not None and _mod['firma'] == mr.firma_tabla(_tab):
                        st.session_state.setdefault('mrf_variantes', {})[
                            (_nivel, _f_mu, tuple(_mod['predictores']))] = {
                            'vif': (_mod.get('variables') or {}).get('umbral'),
                            'pca': (_mod.get('variables') or {}).get('metodo') == 'pca',
                            'n_var': len(_mod['predictores']),
                            'n_muestras': _mod['n_muestras'], 'f1': _mod['macro_f1']}
                    _variantes = sorted(
                        ((k, v) for k, v in (st.session_state.get('mrf_variantes')
                                             or {}).items()
                         if k[0] == _nivel and k[1] == _f_mu), key=lambda kv: -kv[1]['n_var'])
                    if _variantes:
                        with st.container(key='mapa_comparar'):
                            st.markdown('**' + i18n.pick('Con qué variables acierta más',
                                                         'Which variables work best') + '**')
                            _ahora = tuple(_mod.get('predictores') or ())
                            _n_vif = sum(1 for _, v in _variantes if v['vif'])

                            def _nombre_var(v):
                                if v.get('pca'):
                                    return i18n.pick('Con PCA', 'With PCA')
                                if v['vif']:
                                    return (i18n.pick('Con VIF', 'With VIF')
                                            + (f" (≤ {v['vif']:g})" if _n_vif > 1 else ''))
                                if v['n_var'] == len(mr.PREDICTORES):
                                    return i18n.pick('Con todas las variables',
                                                     'With all the variables')
                                return i18n.pick('Con las marcadas a mano',
                                                 'With the ones selected by hand')
                            st.dataframe(pd.DataFrame([{
                                ' ': '▶' if k[2] == _ahora else '',
                                i18n.pick('Modelo', 'Model'): _nombre_var(v),
                                i18n.pick('Variables', 'Variables'): v['n_var'],
                                i18n.pick('F1 medio', 'Mean F1'): _num(v['f1'])}
                                for k, v in _variantes]), hide_index=True, width='stretch')
                            st.caption(i18n.pick(
                                'Los modelos del mismo nivel entrenados en esta sesión con '
                                'las mismas muestras; ▶ el de ahora, el que hace el mapa.',
                                'The models of the same level trained in this session with '
                                'the same samples; ▶ the current one, which makes the map.'))

        # ------------------------------------------------------------ 4 · Mapa
        # El modelo final se aplica a todos los píxeles de una zona. Se leen de la
        # misma pila, con los mismos años y en la misma rejilla que las muestras, por
        # bloques, y el resultado se va escribiendo en GeoTIFF. Como en la demo: los
        # ajustes a la izquierda (zona, máscara, pureza, ambiente, agua) y el mapa a la
        # derecha, con lo que ocupa cada categoría y las descargas debajo.
        with _t_mapa:
            import math
            import shutil
            import mapeo_ambiente as mam
            import mapeo_raster as mra
            _mm = st.session_state.pop('_mrf_mapa_msg', None)
            if _mm:
                st.success(_mm)

            def _dura(minutos, prevision=True):
                """Una duración en minutos, dicha como se dice. Las previsiones se
                redondean hacia arriba: es mejor que el mapa acabe antes de lo
                anunciado que después."""
                if minutos < 0.5:
                    return i18n.pick('menos de 1 min', 'less than 1 min')
                minutos = math.ceil(minutos) if prevision else max(1, round(minutos))
                if minutos < 90:
                    return f'{minutos} min'
                return f'{minutos // 60} h {minutos % 60} min'

            def _tardara(minutos):
                """Lo que tardará, como en la demo: «menos de 1 minuto», «algo más de 1
                minuto», «unos 3 minutos», «unas 2 h y 10 min»."""
                if minutos < 1:
                    return i18n.pick('menos de 1 minuto', 'less than 1 minute')
                if minutos < 1.5:
                    return i18n.pick('algo más de 1 minuto', 'a little over 1 minute')
                m = math.ceil(minutos)
                if m < 90:
                    return i18n.pick(f'unos {m} minutos', f'about {m} minutes')
                return i18n.pick(f'unas {m // 60} h y {m % 60} min',
                                 f'about {m // 60} h {m % 60} min')

            _MAX_BLOQUES = 400           # ≈ 2 600 km² y unas 7 h: más, por partes
            _lec_m = (_mod or {}).get('lectura') or {}
            _al_dia = bool(_mod and _tab is not None and _lec_m and _mod.get('muestras')
                           and _lec_m.get('version') == geb.VERSION_PILA
                           and _mod['firma'] == mr.firma_tabla(_tab))
            _col_aj, _col_vi = st.columns([0.37, 0.63], gap='small')
            with _col_aj, st.container(key='tarjeta_mapa_ajustes'):
                if not _mod:
                    st.caption(i18n.pick(
                        'Antes hace falta un modelo entrenado («3 · Modelo»).',
                        'A trained model is needed first ("3 · Model").'))
                elif not _al_dia:
                    st.warning(i18n.pick(
                        'El modelo es de una tabla anterior. Vuelve a entrenarlo en '
                        '«3 · Modelo» antes de hacer un mapa nuevo.',
                        'The model comes from a previous table. Train it again under '
                        '"3 · Model" before making a new map.'))
                else:
                    _m_epsg, _m_anios = int(_lec_m['epsg']), list(_lec_m['anios'])
                    _pts_m = [(m['lon'], m['lat']) for m in _mod['muestras']]
                    _con_aus = any(c['categoria'] == mr.AUSENCIA for c in _mod['categorias'])
                    _mod_si_no = st.session_state.get('mrf_modelos', {}).get(mr.NIVEL_SI_NO)
                    _mod_mapa, _dos_mal = _para_el_mapa(_mod)
                    if _mod_mapa is not _mod:
                        _con_aus = True
                        st.info(i18n.pick(
                            'El mapa se hace **en dos pasos**: si un píxel es borreguil lo '
                            'decide el modelo **«Borreguil sí o no»** (umbral '
                            f"{_num(_mod_si_no['regla']['umbral'])}, entrenado el "
                            f"{_mod_si_no['cuando'].replace('T', ' a las ')}), y su tipo, "
                            'este modelo.',
                            'The map is made **in two steps**: whether a pixel is a '
                            'borreguil is decided by the **"Borreguil yes or no"** model '
                            f"(threshold {_num(_mod_si_no['regla']['umbral'])}, trained on "
                            f"{_mod_si_no['cuando'].replace('T', ' at ')}), and its type, by "
                            'this model.'))
                    elif _dos_mal:
                        st.warning(i18n.pick(
                            f'No se puede juntar con el modelo «Borreguil sí o no»: {_dos_mal}.',
                            f'It cannot be combined with the "Borreguil yes or no" model: '
                            f'{_dos_mal}.'))
                    elif _nivel != mr.NIVEL_SI_NO and not _con_aus:
                        st.info(i18n.pick(
                            'Para que el mapa diga también dónde **no** hay borreguil, '
                            'entrena el modelo **«Borreguil sí o no»** (elígelo arriba, en '
                            '«Qué quieres cartografiar»): el mapa se hará entonces en dos '
                            'pasos.',
                            'For the map to also say where there is **no** borreguil, train '
                            'the **"Borreguil yes or no"** model (choose it above, under '
                            '"What to map"): the map will then be made in two steps.'))
                    # Con el ambiente por cercanía, el modelo aprende de cada muestra solo la
                    # humedad; pero la persona revisó también su ambiente. Aquí queda lo que
                    # revisó en cada una: el ambiente y la categoría completa («laguna ·
                    # húmedo»), para compararlo después con el mapa.
                    _de_revisado = {}
                    if _cercania:
                        for _r in rows:
                            if (all(p in bp.niveles_revisados(_r)
                                    for p in mr.COMBINADOS.get(_nivel, (_nivel,)))
                                    and mr.categoria(_r, _nivel)):
                                try:
                                    _k_r = mr.clave_punto(_r['lon'], _r['lat'])
                                except Exception:
                                    continue
                                _de_revisado[_k_r] = {'ambiente': mr.categoria(_r, _cercania),
                                                      'compuesta': mr.categoria(_r, _nivel)}

                    def _cat_muestra(m):
                        """La categoría de una muestra: la completa que revisó la persona,
                        si la hay; si no, la del modelo."""
                        return (_de_revisado.get(mr.clave_punto(m['lon'], m['lat'])) or {}).get(
                            'compuesta') or m['categoria']

                    # La zona va la primera (la demo), pero sus opciones y lo que tardará
                    # dependen de lo de debajo (la máscara y lo que hay que leer): su hueco
                    # se rellena después.
                    _hueco_zona = st.container(gap='small')

                    # ---- máscara de zonas candidatas
                    # Sin clase «no borreguil», el modelo le pondría un tipo a cualquier
                    # píxel: solo se clasifica dentro de una máscara, y sin ella no hay mapa.
                    if _con_aus:
                        _caja_masc = st.expander(i18n.pick(
                            'Clasificar solo dentro de una máscara (opcional)',
                            'Classify only inside a mask (optional)'))
                    else:
                        st.warning(i18n.pick(
                            'Este modelo **no tiene la clase «no borreguil»**: distingue '
                            'tipos de borreguil, pero no sabe decir que algo no lo es, así '
                            'que le pondría un tipo a cualquier píxel (roca, pinar, agua…). '
                            'Por eso solo clasifica **dentro de una máscara de zonas '
                            'candidatas**, que tienes que subir; el resto queda sin dato. El '
                            'mapa dirá de qué tipo sería cada sitio, no que allí haya '
                            'borreguil.',
                            'This model **has no "not a borreguil" class**: it tells '
                            'borreguil types apart, but it cannot say that something is not '
                            'one, so it would give a type to any pixel (rock, pine forest, '
                            'water…). That is why it only classifies **inside a mask of '
                            'candidate zones**, which you have to upload; the rest is left '
                            'without data. The map will say which type each place would be, '
                            'not that there is a borreguil there.'))
                        _caja_masc = st.container()
                    _masc, _mc = None, None
                    with _caja_masc:
                        _f_masc = st.file_uploader(
                            i18n.pick('Máscara de zonas candidatas (polígonos)',
                                      'Mask of candidate zones (polygons)'),
                            type=['kml', 'geojson', 'json', 'gpkg', 'shp', 'shx', 'dbf', 'prj',
                                  'cpg', 'zip'],
                            accept_multiple_files=True, key='mrf_mascara_fich',
                            help=i18n.pick(
                                'Polígonos de las zonas donde puede haber borreguil (por '
                                'ejemplo, una cartografía de hábitats). Un único KML, '
                                'GeoJSON o GeoPackage, o un shapefile (.shp + .shx + .dbf + '
                                '.prj juntos, o un .zip). Solo se clasifican los píxeles '
                                'cuyo centro cae dentro.',
                                'Polygons of the zones where there may be borreguil (for '
                                'example, a habitat map). A single KML, GeoJSON or '
                                'GeoPackage, or a shapefile (.shp + .shx + .dbf + .prj '
                                'together, or a .zip). Only pixels whose centre falls inside '
                                'are classified.'))
                        if _f_masc:
                            _clave_masc = tuple(sorted((f.name, f.size) for f in _f_masc))
                            _mc = st.session_state.get('mrf_mascara')
                            if not _mc or _mc['clave'] != _clave_masc:
                                try:
                                    import study_area as sa
                                    _mc = {'clave': _clave_masc, 'error': None,
                                           'geom': sa.load_vector(save_uploaded_area(_f_masc))}
                                except Exception as _e:
                                    _mc = {'clave': _clave_masc, 'error': str(_e), 'geom': None}
                                st.session_state['mrf_mascara'] = _mc
                            if _mc['error']:
                                st.error(i18n.pick(
                                    f"No se ha podido leer la máscara: {_mc['error']}",
                                    f"The mask could not be read: {_mc['error']}"))
                            else:
                                _masc = _mc['geom']
                                if _mc.get('epsg') != _m_epsg:
                                    _mc['km2'] = mra.a_utm(_masc, _m_epsg).area / 1e6
                                    _mc['epsg'] = _m_epsg
                                st.caption(i18n.pick(
                                    f"Máscara leída: {_num(_mc['km2'])} km².",
                                    f"Mask read: {_num(_mc['km2'])} km²."))

                    # ---- pureza dentro del píxel: puro, mixto-agua o mixto-roca. No sale
                    # del modelo, sino de cuánto verde hay en cada píxel de borreguil
                    # (mapeo_rf.verde_estimado). Los valores de partida se ajustan aquí.
                    _m_pur = []                 # muestras de borreguil: su pureza y su NDVI
                    for _f in _usadas:
                        _v = _leidos.get(mr.clave_punto(_f['lon'], _f['lat'])) or {}
                        if (_f['categoria'] != mr.AUSENCIA
                                and _v.get(mr.VARIABLE_VERDE) is not None):
                            _r = rows[_f['indice']] if 0 <= _f['indice'] < len(rows) else {}
                            # solo la pureza que ha revisado una persona, no la propuesta
                            _m_pur.append({'pureza': (bp._txt(_r.get('pureza'))
                                                      if 'pureza' in bp.niveles_revisados(_r)
                                                      else ''),
                                           'ndvi': _v[mr.VARIABLE_VERDE]})
                    _n_puras = sum(1 for m in _m_pur if m['pureza'] == 'puro')
                    _tv_prop, _tv_muestras = mr.todo_verde_de(
                        [m['ndvi'] for m in _m_pur if m['pureza'] == 'puro'])
                    _par_pur, _pur_mal = None, False
                    with st.container(key='mapa_pureza', gap='small'):
                        _pur_si = st.checkbox(
                            i18n.pick('Calcular la pureza de cada píxel de borreguil',
                                      'Compute the purity of each borreguil pixel'),
                            value=True, key='mrf_pureza_si',
                            help=i18n.pick(
                                'Añade al mapa una capa y dos ficheros más: la pureza de cada '
                                'píxel de borreguil y el verde estimado de cada píxel. Solo '
                                'usa Sentinel-2 y no alarga el mapa.\n\n'
                                'Recomendado: marcado.',
                                'Adds one layer and two more files to the map: the purity of '
                                'each borreguil pixel and the estimated green of each pixel. '
                                'It only uses Sentinel-2 and does not lengthen the map.\n\n'
                                'Recommended: ticked.'))
                        if _pur_si:
                            _q1, _q2, _q3 = st.columns(3)
                            _pur_sin = _q1.number_input(
                                i18n.pick('NDVI sin verde', 'NDVI no green'),
                                0.0, 0.5, mr.PUREZA_DEFECTO['sin_verde'], 0.01, format='%.3f',
                                key='mrf_pureza_sin',
                                help=i18n.pick(
                                    'El NDVI que tiene la roca o el suelo desnudo. Por debajo '
                                    'de este valor se cuenta un 0 % de verde.\n\n'
                                    'Recomendado: 0,080, lo habitual en roca y suelo de alta '
                                    'montaña.',
                                    'The NDVI of rock or bare ground. Below this value the '
                                    'pixel counts as 0 % green.\n\n'
                                    'Recommended: 0.080, the usual value for high-mountain '
                                    'rock and ground.'))
                            _pur_todo = _q2.number_input(
                                i18n.pick('NDVI todo verde', 'NDVI all green'),
                                0.2, 1.0, min(1.0, max(0.2, round(_tv_prop, 3))), 0.01,
                                format='%.3f', key='mrf_pureza_todo',
                                help=i18n.pick(
                                    'El NDVI de un píxel cubierto entero por borreguil. A '
                                    'partir de este valor se cuenta un 100 % de verde; entre '
                                    'los dos, en proporción.\n\n'
                                    'Recomendado: el de tus muestras de borreguil puro (su '
                                    'mediana), que es el que viene puesto si hay al menos 5; '
                                    'si no, 0,600.',
                                    'The NDVI of a pixel entirely covered by borreguil. From '
                                    'this value on the pixel counts as 100 % green; between '
                                    'the two, in proportion.\n\n'
                                    'Recommended: that of your pure borreguil samples (their '
                                    'median), which is the preset value if there are at least '
                                    '5; otherwise 0.600.'))
                            _pur_corte = _q3.number_input(
                                i18n.pick('Puro desde (%)', 'Pure from (%)'),
                                50, 95, int(round(100 * mr.PUREZA_DEFECTO['puro_desde'])), 5,
                                key='mrf_pureza_corte',
                                help=i18n.pick(
                                    'Con menos verde que este, el píxel es mixto.\n\n'
                                    'Recomendado: 75 %. El verde se estima con un error de '
                                    'unos 10 puntos, así que un corte muy alto dejaría casi '
                                    'todo como mixto.',
                                    'With less green than this, the pixel is mixed.\n\n'
                                    'Recommended: 75 %. The green is estimated with an error '
                                    'of about 10 points, so a very high cut-off would leave '
                                    'almost everything as mixed.'))
                            _pur_mal = bool(_pur_todo <= _pur_sin)
                            if not _pur_mal:
                                _par_pur = {'sin_verde': float(_pur_sin),
                                            'todo_verde': float(_pur_todo),
                                            'puro_desde': _pur_corte / 100}
                            else:
                                st.error(i18n.pick(
                                    'El NDVI de «todo verde» tiene que ser mayor que el de '
                                    '«sin verde». Corrígelo, o desmarca la pureza, para poder '
                                    'hacer el mapa.',
                                    'The "all green" NDVI must be greater than the "no green" '
                                    'one. Fix it, or untick the purity, to be able to make '
                                    'the map.'))
                            # Cómo quedan tus muestras con estos valores (lo que había antes
                            # de la demo, plegado)
                            if _m_pur or _n_puras:
                                with st.expander(i18n.pick('Cómo sale con tus muestras',
                                                           'How it comes out with your '
                                                           'samples')):
                                    st.caption(i18n.pick(
                                        f'Tus {_n_puras} muestras de borreguil puro tienen un '
                                        'NDVI de inicio de verano (15 jun – 24 jul) de '
                                        f'{_num(_tv_prop)} de mediana.' if _tv_muestras else
                                        f'Hay {_n_puras} muestras de borreguil puro: con menos '
                                        f'de {mr.MIN_PUROS} no se propone su NDVI, sino '
                                        f'{_num(_tv_prop)}.',
                                        f'Your {_n_puras} pure borreguil samples have a median '
                                        'early-summer NDVI (15 Jun – 24 Jul) of '
                                        f'{_num(_tv_prop)}.' if _tv_muestras else
                                        f'There are {_n_puras} pure borreguil samples: with '
                                        f'fewer than {mr.MIN_PUROS} their NDVI is not '
                                        f'proposed, but {_num(_tv_prop)}.'))
                                    if _par_pur and _m_pur:
                                        _cmp = mr.comparar_pureza(_m_pur, _par_pur)
                                        st.markdown(i18n.pick(
                                            f"Con estos valores, **{_cmp['coinciden']} de tus "
                                            f"{_cmp['n']} muestras** de borreguil quedan como "
                                            'las revisaste (puro frente a mixto):',
                                            f"With these values, **{_cmp['coinciden']} of your "
                                            f"{_cmp['n']} borreguil samples** come out as you "
                                            'reviewed them (pure versus mixed):'))
                                        st.dataframe(pd.DataFrame([{
                                            i18n.pick('Pureza revisada', 'Reviewed purity'):
                                                _cat(c),
                                            i18n.pick('Sale puro', 'Comes out pure'):
                                                _cmp['tabla'][c]['puro'],
                                            i18n.pick('Sale mixto', 'Comes out mixed'):
                                                _cmp['tabla'][c]['mixto']}
                                            for c in mr.NIVELES['pureza']]), hide_index=True,
                                            width='stretch')
                                        st.caption(i18n.pick(
                                            'En un punto suelto no se sabe si toca agua: por '
                                            'eso aquí solo se compara puro frente a mixto.',
                                            'For a single point it is not known whether it '
                                            'touches water: that is why only pure versus mixed '
                                            'is compared here.'))

                    # ---- ambiente por cercanía (en «Ambiente y humedad» y, si se pide, en
                    # «Borreguil sí o no»). No sale del modelo, sino de la distancia de cada
                    # píxel a lagunas y cauces (mapeo_ambiente): por eso no necesita tipos
                    # revisados. Las distancias de partida se ajustan aquí.
                    _par_amb, _amb_mal, _capa_cauces, _f_cauces, _lagunas_ll = (
                        None, None, None, None, [])
                    _amb_si_no = False
                    _ay_amb = (
                        'El modelo decide si hay borreguil; el ambiente lo pone el mapa por la '
                        'cercanía de cada píxel de borreguil a lagunas y cauces, así que no hace '
                        'falta ningún punto con el tipo revisado. En el mapa, cada píxel de '
                        'borreguil sale como «arroyo», «laguna» o «ladera». Los cauces son los '
                        'ríos y arroyos de OpenStreetMap, los mismos que en los puntos.\n\n'
                        'Recomendado: marcado.',
                        'The model decides whether there is borreguil; the environment is set '
                        'by the map from how close each borreguil pixel is to lakes and '
                        'watercourses, so no point with a reviewed type is needed. On the map, '
                        'each borreguil pixel comes out as "stream", "lake" or "slope". The '
                        'watercourses are the OpenStreetMap rivers and streams, the same as for '
                        'the points.\n\nRecommended: ticked.')
                    if _cercania or (_nivel == mr.NIVEL_SI_NO and _con_aus):
                        with st.container(key='mapa_ambiente', gap='small'):
                            if _nivel == mr.NIVEL_SI_NO:
                                _amb_si_no = st.checkbox(
                                    i18n.pick('Decir el ambiente de cada píxel (arroyo, laguna '
                                              'o ladera)',
                                              'Give the environment of each pixel (stream, '
                                              'lake or slope)'),
                                    value=True, key='mrf_amb_si_no',
                                    help=i18n.pick(*_ay_amb))
                            else:
                                with st.container(horizontal=True, vertical_alignment='center',
                                                  gap='small'):
                                    st.markdown('**' + i18n.pick(
                                        'El ambiente de cada píxel (arroyo, laguna o ladera)',
                                        'The environment of each pixel (stream, lake or '
                                        'slope)') + '**', width='content')
                                    ayuda_q('q_mapa_ambiente', *_ay_amb)
                            if _cercania or _amb_si_no:
                                if 'mrf_lagunas' not in st.session_state:
                                    try:
                                        st.session_state['mrf_lagunas'] = [
                                            (float(q['lon']), float(q['lat']))
                                            for q in bp.read_points(
                                                str(APP_DIR / 'lagunas_sierra_nevada.kml'))]
                                    except Exception:
                                        st.session_state['mrf_lagunas'] = []
                                _lagunas_ll = st.session_state['mrf_lagunas']
                                _a1, _a2, _a3 = st.columns(3)
                                _d_capa = _a1.number_input(
                                    # La capa de lagunas solo es de Sierra Nevada (la autora,
                                    # 27-11-2026: «ponlo de alguna manera»).
                                    i18n.pick('Laguna de la capa · solo Sierra Nevada (m)',
                                              'Lake of the layer · Sierra Nevada only (m)'),
                                    0.0, 1000.0, mam.DEFECTO['laguna_capa_m'], 10.0,
                                    format='%.0f', key='mrf_amb_capa',
                                    help=i18n.pick(
                                        'Un píxel es de laguna si está a esta distancia o '
                                        'menos de un punto de la capa de lagunas de la app '
                                        '(65 lagunas de Sierra Nevada). El punto es el centro '
                                        'de la laguna, no su orilla: sirve sobre todo para las '
                                        'lagunas pequeñas, que Sentinel-2 no ve.\n\n'
                                        'Recomendado: 120 m, lo que usa la app al proponer el '
                                        'tipo de cada punto.\n\n'
                                        'Solo sirve en Sierra Nevada: fuera de allí la capa no '
                                        'tiene lagunas y el agua sale solo de Sentinel-2.',
                                        'A pixel is a lake one if it is within this distance '
                                        'of a point of the lake layer of the app (65 lakes of '
                                        'Sierra Nevada). The point is the centre of the lake, '
                                        'not its shore: it mostly helps with small lakes, which '
                                        'Sentinel-2 does not see.\n\n'
                                        'Recommended: 120 m, what the app uses when it proposes '
                                        'the type of each point.\n\n'
                                        'It only works in Sierra Nevada: elsewhere the layer has '
                                        'no lakes and water comes from Sentinel-2 only.'))
                                _d_agua = _a2.number_input(
                                    i18n.pick('Agua de Sentinel-2 (m)', 'Sentinel-2 water (m)'),
                                    0.0, 1000.0, mam.DEFECTO['laguna_agua_m'], 10.0,
                                    format='%.0f', key='mrf_amb_agua',
                                    help=i18n.pick(
                                        'Un píxel es de laguna si está a esta distancia o '
                                        'menos de la orilla de una laguna que se ve en '
                                        'Sentinel-2: píxeles con NDWI medio de verano (1 jun – 29 sep) mayor de '
                                        '0,05, en terreno casi llano y en manchas de 3 o más. '
                                        'Vale en cualquier sitio, aunque no haya capa de '
                                        'lagunas.\n\n'
                                        'Recomendado: 100 m.',
                                        'A pixel is a lake one if it is within this distance '
                                        'of the shore of a lake that shows in Sentinel-2: '
                                        'pixels with a mean summer (1 Jun – 29 Sep) NDWI above 0.05, on nearly '
                                        'flat ground and in patches of 3 or more. It works '
                                        'anywhere, even with no lake layer.\n\n'
                                        'Recommended: 100 m.'))
                                _d_arroyo = _a3.number_input(
                                    i18n.pick('Arroyo: a un cauce (m)',
                                              'Stream: to a watercourse (m)'),
                                    0.0, 500.0, mam.DEFECTO['arroyo_m'], 10.0, format='%.0f',
                                    key='mrf_amb_arroyo',
                                    help=i18n.pick(
                                        'Un píxel que no es de laguna es de arroyo si está a '
                                        'esta distancia o menos de un río o arroyo de '
                                        'OpenStreetMap (o de tu capa de cauces, si la '
                                        'subes).\n\n'
                                        'Recomendado: 30 m, lo que usa la app al proponer el '
                                        'tipo de cada punto.',
                                        'A pixel that is not a lake one is a stream one if it '
                                        'is within this distance of an OpenStreetMap river or '
                                        'stream (or of your watercourse layer, if you upload '
                                        'one).\n\n'
                                        'Recommended: 30 m, what the app uses when it proposes '
                                        'the type of each point.'))
                                # Los cauces son los ríos y arroyos de OpenStreetMap, como en
                                # los puntos (24-11-2026, la autora). Los del relieve quedan
                                # solo de reserva, si OpenStreetMap no responde fuera de
                                # Sierra Nevada, con su ajuste de siempre.
                                _cauce_ha = mam.DEFECTO['cauce_ha']
                                _f_cauces = st.file_uploader(
                                    i18n.pick('Capa de cauces propia (líneas; opcional)',
                                              'Your own watercourse layer (lines; optional)'),
                                    type=['kml', 'geojson', 'json', 'gpkg', 'shp', 'shx', 'dbf',
                                          'prj', 'cpg', 'zip'],
                                    accept_multiple_files=True, key='mrf_cauces_fich',
                                    help=i18n.pick(
                                        'Las líneas de los cauces (por ejemplo, la red '
                                        'hidrográfica oficial). Un único KML, GeoJSON o '
                                        'GeoPackage, o un shapefile (.shp + .shx + .dbf + .prj '
                                        'juntos, o un .zip). Si la subes, se usa en lugar de '
                                        'los ríos y arroyos de OpenStreetMap.\n\n'
                                        'Recomendado: no subir nada, salvo que tengas una red '
                                        'de cauces de la zona más completa que la de '
                                        'OpenStreetMap.',
                                        'The lines of the watercourses (for example, the '
                                        'official hydrographic network). A single KML, GeoJSON '
                                        'or GeoPackage, or a shapefile (.shp + .shx + .dbf + '
                                        '.prj together, or a .zip). If you upload it, it is '
                                        'used instead of the OpenStreetMap rivers and '
                                        'streams.\n\n'
                                        'Recommended: upload nothing, unless you have a '
                                        'watercourse network of the zone more complete than '
                                        'OpenStreetMap.'))
                                if _f_cauces:
                                    _clave_cau = tuple(sorted((f.name, f.size)
                                                              for f in _f_cauces))
                                    _cc = st.session_state.get('mrf_cauces')
                                    if not _cc or _cc['clave'] != _clave_cau:
                                        try:
                                            _cc = {'clave': _clave_cau, 'error': None,
                                                   'geom': mam.leer_cauces(
                                                       save_uploaded_area(_f_cauces))}
                                        except Exception as _e:
                                            _cc = {'clave': _clave_cau, 'error': str(_e),
                                                   'geom': None}
                                        st.session_state['mrf_cauces'] = _cc
                                    if _cc['error']:
                                        _amb_mal = i18n.pick(
                                            'no se ha podido leer la capa de cauces '
                                            f"({_cc['error']})",
                                            'the watercourse layer could not be read '
                                            f"({_cc['error']})")
                                    else:
                                        _capa_cauces = _cc['geom']
                                        st.caption(i18n.pick('Cauces: los de tu capa.',
                                                             'Watercourses: those of your '
                                                             'layer.'))
                                _par_amb = {'laguna_capa_m': float(_d_capa),
                                            'laguna_agua_m': float(_d_agua),
                                            'arroyo_m': float(_d_arroyo),
                                            'cauce_ha': float(_cauce_ha)}
                                if _amb_mal:
                                    st.error(i18n.pick(
                                        f'**Ambiente por cercanía:** {_amb_mal}. Quita ese '
                                        'fichero o sube otro para poder hacer el mapa.',
                                        f'**Environment by closeness:** {_amb_mal}. Remove '
                                        'that file or upload another one to be able to make '
                                        'the map.'))
                    # ---- el agua de laguna, que no es borreguil
                    _agua_si = False
                    if _con_aus:
                        with st.container(key='mapa_agua'):
                            _agua_si = st.checkbox(
                                i18n.pick('Quitar el agua de laguna (no es borreguil)',
                                          'Remove lake water (it is not borreguil)'),
                                value=True, key='mrf_quitar_agua',
                                help=i18n.pick(
                                    'El agua abierta de una laguna se parece a un borreguil '
                                    'muy húmedo, y el modelo la puede dar por borreguil. Con '
                                    'esto, los píxeles de agua que se ven en Sentinel-2 (NDWI '
                                    'medio de verano (1 jun – 29 sep) mayor de 0,05, en terreno casi llano y en '
                                    'manchas de 3 o más píxeles) pasan a «no borreguil». El '
                                    'agua se guarda además en su propio fichero. Las '
                                    'probabilidades no cambian.\n\n'
                                    'Recomendado: marcado.',
                                    'The open water of a lake looks like a very wet borreguil, '
                                    'and the model may call it one. With this, the water '
                                    'pixels that show in Sentinel-2 (mean summer (1 Jun – 29 Sep) NDWI above '
                                    '0.05, on nearly flat ground and in patches of 3 or more '
                                    'pixels) become "not a borreguil". The water is also saved '
                                    'in its own file. The probabilities do not change.\n\n'
                                    'Recommended: ticked.'))
                    # Lo que hay que leer de cada bloque: las variables del modelo y, detrás,
                    # las que solo hacen falta para la pureza, el ambiente o el agua, si el
                    # modelo no las lleva. (El relieve de los cauces se lee aparte, con su
                    # margen.)
                    _extra_b = ([mr.VARIABLE_VERDE, mr.VARIABLE_AGUA] if _par_pur else []) + (
                        [mam.VARIABLE_AGUA, mam.VARIABLE_PENDIENTE]
                        if _par_amb or _agua_si else [])
                    _bandas_m = list(_mod_mapa['predictores']) + [
                        b for b in dict.fromkeys(_extra_b) if b not in _mod_mapa['predictores']]

                    # ---- zona (arriba, en su hueco)
                    with _hueco_zona:
                        _zonas = {}
                        if sg is not None:
                            _zonas['area'] = i18n.pick('El área de estudio', 'The study area')
                        _zonas['muestras'] = i18n.pick('El rectángulo que ocupan las muestras',
                                                       'The rectangle the samples occupy')
                        _zonas['rect'] = i18n.pick('Un rectángulo que elijo yo',
                                                   'A rectangle I choose')
                        if _masc is not None:
                            _zonas['mascara'] = i18n.pick('Todo lo que cubre la máscara',
                                                          'Everything the mask covers')
                        if st.session_state.get('mrf_zona') not in _zonas:
                            st.session_state.pop('mrf_zona', None)
                        # En el orden de la demo, pero de entrada el rectángulo de 2 × 2 km
                        # (lo que recomienda la ayuda): el área de estudio o las muestras
                        # pueden ser cientos de km² y horas de Earth Engine.
                        _zona = st.selectbox(
                            i18n.pick('Zona a cartografiar', 'Zone to map'), list(_zonas),
                            index=list(_zonas).index('rect'),
                            key='mrf_zona', format_func=lambda z: _zonas[z],
                            help=i18n.pick(
                                'Earth Engine sirve unos 6 km² por minuto. Conviene empezar '
                                'por un rectángulo pequeño, ver cómo sale y ampliar después. '
                                'Debajo se indica lo que tardará lo que elijas.\n\n'
                                'Recomendado para empezar: un rectángulo de 2 × 2 km (un '
                                'minuto).',
                                'Earth Engine serves about 6 km² per minute. It is best to '
                                'start with a small rectangle, see how it turns out and '
                                'enlarge it afterwards. What you choose shows below how long '
                                'it will take.\n\n'
                                'Recommended to start: a 2 × 2 km rectangle (one minute).'))
                        _lim_z, _pol_z = None, None
                        if _zona == 'rect':
                            # Centro por defecto: la muestra más cercana al centro de todas,
                            # para que el primer rectángulo tenga alguna dentro. Los valores
                            # se inician en el estado (y no con `value`) para poder
                            # cambiarlos al pinchar en el mapa de la zona.
                            _lon_c = sorted(p[0] for p in _pts_m)[len(_pts_m) // 2]
                            _lat_c = sorted(p[1] for p in _pts_m)[len(_pts_m) // 2]
                            _cerca = min(_pts_m, key=lambda p: (p[0] - _lon_c) ** 2
                                         + (p[1] - _lat_c) ** 2)
                            _antes_r = st.session_state.get('_mrf_rect') or (
                                round(_cerca[0], 5), round(_cerca[1], 5), 2.0, 2.0)
                            _pend_c = st.session_state.pop('_mrf_centro', None)
                            if _pend_c:
                                _antes_r = (_pend_c[0], _pend_c[1]) + tuple(_antes_r[2:])
                                st.session_state['mrf_rect_lon'] = _pend_c[0]
                                st.session_state['mrf_rect_lat'] = _pend_c[1]
                            for _k, _v in zip(('mrf_rect_lon', 'mrf_rect_lat', 'mrf_rect_ancho',
                                               'mrf_rect_alto'), _antes_r):
                                st.session_state.setdefault(_k, float(_v))
                            _r1, _r2 = st.columns(2)
                            _r_lon = _r1.number_input(
                                i18n.pick('Longitud del centro', 'Centre longitude'),
                                -180.0, 180.0, step=0.005, format='%.5f', key='mrf_rect_lon',
                                help=i18n.pick(
                                    'En grados decimales (oeste, negativo). También puedes '
                                    'pinchar en el mapa.',
                                    'In decimal degrees (west is negative). You can also '
                                    'click on the map.'))
                            _r_lat = _r2.number_input(
                                i18n.pick('Latitud del centro', 'Centre latitude'),
                                -90.0, 90.0, step=0.005, format='%.5f', key='mrf_rect_lat',
                                help=i18n.pick(
                                    'En grados decimales. También puedes pinchar en el mapa.',
                                    'In decimal degrees. You can also click on the map.'))
                            _r_w = _r1.number_input(
                                i18n.pick('Ancho (km)', 'Width (km)'), 0.1, 200.0, step=0.5,
                                key='mrf_rect_ancho',
                                help=i18n.pick(
                                    'De oeste a este. Recomendado para empezar: 2 km.',
                                    'From west to east. Recommended to start: 2 km.'))
                            _r_h = _r2.number_input(
                                i18n.pick('Alto (km)', 'Height (km)'), 0.1, 200.0, step=0.5,
                                key='mrf_rect_alto',
                                help=i18n.pick(
                                    'De sur a norte. Recomendado para empezar: 2 km.',
                                    'From south to north. Recommended to start: 2 km.'))
                            st.session_state['_mrf_rect'] = (_r_lon, _r_lat, _r_w, _r_h)
                            _lim_z = mra.rect_centrado(_r_lon, _r_lat, _r_w, _r_h, _m_epsg)
                            _clave_z = ('rect', _r_lon, _r_lat, _r_w, _r_h)
                        elif _zona == 'muestras':
                            _lim_z = mra.limites_de_puntos(_pts_m, _m_epsg, margen_m=100)
                            _clave_z = ('muestras', _mod['firma'])
                        elif _zona == 'area':
                            _pol_z = sg
                            _clave_z = ('area',) + tuple(sg.bounds) + (sg.area,)
                        else:
                            _clave_z = ('mascara',)

                        # La rejilla y sus bloques: se calculan una vez por zona, no en
                        # cada pasada del guion.
                        _clave_z = (_clave_z, _m_epsg,
                                    _mc['clave'] if _masc is not None else None)
                        _zc = st.session_state.get('mrf_zona_calc')
                        if not _zc or _zc['clave'] != _clave_z:
                            try:
                                _zc = mra.preparar_zona(_m_epsg, limites=_lim_z, zona=_pol_z,
                                                        mascara=_masc)
                            except Exception as _e:
                                _zc = {'error': 'fallo', 'detalle': str(_e)}
                            _zc['clave'] = _clave_z
                            st.session_state['mrf_zona_calc'] = _zc

                        _rej = _zc.get('rej')
                        _utiles = _zc.get('utiles') or []
                        _pega = None                        # por qué no se puede hacer el mapa
                        if _zc['error'] == 'mascara_fuera':
                            _pega = i18n.pick(
                                'La máscara no toca la zona elegida: no hay nada que '
                                'clasificar.',
                                'The mask does not touch the chosen zone: there is nothing to '
                                'classify.')
                        elif _zc['error'] == 'sin_superficie':
                            _pega = i18n.pick('La zona elegida no tiene superficie.',
                                              'The chosen zone has no area.')
                        elif _zc['error'] == 'demasiado_grande' or len(_utiles) > _MAX_BLOQUES:
                            _nbq = len(_utiles) or _zc.get('bloques', 0)
                            _pega = i18n.pick(
                                f"La zona es demasiado grande para hacerla de una vez "
                                f"({_num(_zc.get('km2', 0))} km², {_nbq} bloques; el máximo "
                                f"son {_MAX_BLOQUES}, unas 7 h). Elige una zona menor y hazla "
                                'por partes.',
                                f"The zone is too large to do in one go "
                                f"({_num(_zc.get('km2', 0))} km², {_nbq} blocks; the maximum "
                                f"is {_MAX_BLOQUES}, about 7 h). Choose a smaller zone and do "
                                'it in parts.')
                        elif _zc['error']:
                            _pega = i18n.pick(
                                f"No se ha podido preparar la zona: {_zc.get('detalle', '')}",
                                f"The zone could not be prepared: {_zc.get('detalle', '')}")
                        elif not _utiles:
                            _pega = i18n.pick(
                                'En la zona elegida no cae el centro de ningún píxel: no hay '
                                'nada que clasificar.',
                                'No pixel centre falls inside the chosen zone: there is '
                                'nothing to classify.')

                        if _rej and not _pega:
                            _dir_m = st.session_state.get('mrf_dir')
                            _lect_m = mra.clave_lectura(_m_epsg, _m_anios, _bandas_m,
                                                        geb.VERSION_PILA)
                            _por_leer = (mra.bloques_por_leer(
                                Path(_dir_m) / 'bloques' / _lect_m, _rej, _utiles)
                                if _dir_m else list(_utiles))
                            _min_m = mra.minutos_estimados(_rej, ventanas=_por_leer)
                            st.markdown('<span style="color:#AEB6BE">' + i18n.pick(
                                f"{_num(_zc['km2'])} km² · tardará {_tardara(_min_m)} (Earth "
                                'Engine sirve unos 6 km² por minuto)',
                                f"{_num(_zc['km2'])} km² · it will take {_tardara(_min_m)} "
                                '(Earth Engine serves about 6 km² per minute)') + '</span>',
                                unsafe_allow_html=True)
                            if _min_m > 15:
                                st.warning(i18n.pick(
                                    'Va para largo. Mientras se hace el mapa **no uses la '
                                    'app**: cualquier clic lo interrumpe. Si se interrumpe no '
                                    'se pierde lo leído: vuelve a pulsar el botón y sigue '
                                    'donde iba. Los mapas grandes también gastan más horas de '
                                    'cálculo de Earth Engine (las de la cuenta compartida o la '
                                    'cuota de tu proyecto).',
                                    'This will take a while. While the map is being made **do '
                                    'not use the app**: any click interrupts it. If it is '
                                    'interrupted, what has been read is not lost: press the '
                                    'button again and it carries on where it was. Large maps '
                                    'also use more Earth Engine compute hours (those of the '
                                    'shared account or the quota of your project).'))

                        # ---- dónde cae: la zona sobre la ortofoto (la demo: el área de
                        # estudio en discontinua y lo que se cartografía en verde)
                        if _rej and st.checkbox(i18n.pick('Ver la zona en un mapa',
                                                          'Show the zone on a map'),
                                                value=False, key='mrf_ver_zona'):
                            _lz = mra.limites_lonlat(_rej)
                            _lo = [p[0] for p in _pts_m] + [_lz[0], _lz[2]]
                            _la = [p[1] for p in _pts_m] + [_lz[1], _lz[3]]
                            # El encuadre no sigue al rectángulo: así el mapa no se recarga
                            # (ni pierde el zoom) cada vez que se mueve. Lo que cambia va en
                            # una capa que se añade aparte.
                            _clave_enc = (_mod['firma'], _zona, _clave_z[2])
                            _enc = st.session_state.get('_mrf_encuadre')
                            if not _enc or _enc[0] != _clave_enc:
                                _enc = (_clave_enc, mra.encuadre(
                                    (min(_lo), min(_la), max(_lo), max(_la)), 420, 240))
                                st.session_state['_mrf_encuadre'] = _enc
                            _pm = _base_map(_enc[1][0], _enc[1][1], ortofoto=True)
                            _gz = folium.FeatureGroup(name=i18n.pick('Zona y muestras',
                                                                     'Zone and samples'))
                            if sg is not None:
                                from shapely.geometry import mapping as _mapping
                                folium.GeoJson(
                                    _mapping(sg), style_function=lambda x: {
                                        'color': '#98A2AB', 'weight': 2, 'dashArray': '6 5',
                                        'fillOpacity': 0}).add_to(_gz)
                            if _masc is not None:
                                if 'dibujo' not in _mc:
                                    _mc['dibujo'] = mra.anillos_para_dibujar(_masc)
                                for _anillo in _mc['dibujo']:
                                    folium.Polygon(_anillo, color='#4FC3F7', weight=1.5,
                                                   dash_array='4', fill=False).add_to(_gz)
                            folium.Polygon(mra.contorno_lonlat(_rej), color='#7CC496',
                                           weight=2, fill=True, fill_color='#7CC496',
                                           fill_opacity=0.22).add_to(_gz)
                            # las muestras, solo en verde (borreguil) y rojo (no)
                            for _mu in _mod['muestras']:
                                _es_no = _mu['categoria'] == mr.AUSENCIA
                                folium.CircleMarker(
                                    [_mu['lat'], _mu['lon']], radius=4, color='#111111',
                                    weight=1, fill=True,
                                    fill_color='#E53935' if _es_no else '#43A047',
                                    fill_opacity=1.0,
                                    tooltip=_cat(_cat_muestra(_mu))).add_to(_gz)
                            _rp = st_folium(_pm, width=None, height=240,
                                            feature_group_to_add=_gz,
                                            returned_objects=(['last_clicked']
                                                              if _zona == 'rect' else []),
                                            key='mrf_mapa_zona')
                            st.caption(i18n.pick(
                                'Discontinua: el área de estudio · verde: lo que se cartografía'
                                + (' · pincha en el mapa para llevar allí el centro del '
                                   'rectángulo' if _zona == 'rect' else ''),
                                'Dashed: the study area · green: what is mapped'
                                + (' · click on the map to move the centre of the rectangle '
                                   'there' if _zona == 'rect' else '')))
                            _c = (_rp or {}).get('last_clicked') if isinstance(_rp, dict) \
                                else None
                            if _zona == 'rect' and _c and _c.get('lat') is not None:
                                _cc = (round(float(_c['lng']), 5), round(float(_c['lat']), 5))
                                if _cc != st.session_state.get('_mrf_clic_zona'):
                                    st.session_state['_mrf_clic_zona'] = _cc
                                    st.session_state['_mrf_centro'] = _cc
                                    st.rerun()

                    # ---- hacer el mapa
                    if _pega:
                        st.error(_pega)
                    if not _con_aus and _masc is None:
                        st.error(i18n.pick(
                            '**Falta la máscara de zonas candidatas.** Sin ella, este modelo '
                            'no puede hacer el mapa. Súbela arriba o entrena un modelo que '
                            'incluya las ausencias de campo («1 · Datos»).',
                            '**The mask of candidate zones is missing.** Without it, this '
                            'model cannot make the map. Upload it above or train a model '
                            'that includes the field absences ("1 · Data").'))
                    _puede_mapa = bool(_rej and not _pega and not _pur_mal
                                       and not _amb_mal
                                       and (_con_aus or _masc is not None))
                    if st.button(i18n.pick('Hacer el mapa', 'Make the map'),
                                 key='mrf_hacer_mapa', type='primary', width='stretch',
                                 disabled=not _puede_mapa):
                        with st.status(i18n.pick('Haciendo el mapa…', 'Making the map…'),
                                       expanded=True) as _stm:
                            _ok, _msg = geb.initialize(
                                _proy.strip(),
                                service_account_json=get_secret('EE_SERVICE_ACCOUNT_KEY', None))
                            if not _ok:
                                _stm.update(label=i18n.pick('Earth Engine no está disponible',
                                                            'Earth Engine is not available'),
                                            state='error')
                                st.error(_msg)
                            else:
                                guardar_ajuste('gee_project', _proy.strip())
                                if 'mrf_dir' not in st.session_state:
                                    st.session_state['mrf_dir'] = tempfile.mkdtemp(
                                        prefix='borreguil_mapa_')
                                _dir_m = Path(st.session_state['mrf_dir'])
                                # Lo leído se guarda mientras dura la sesión; lo de otra
                                # lectura (otros años u otras variables) ya no sirve.
                                _dir_bl = _dir_m / 'bloques' / _lect_m
                                for _otra in (_dir_m / 'bloques').glob('*'):
                                    if _otra.name != _lect_m:
                                        shutil.rmtree(_otra, ignore_errors=True)
                                # Cada mapa en su carpeta: el anterior se borra solo cuando
                                # el nuevo está entero.
                                _carp = _dir_m / f"mapa_{_nivel}_{time.strftime('%Y%m%d_%H%M%S')}"
                                while _carp.exists():
                                    _carp = _carp.with_name(_carp.name + 'b')
                                _barra = barra_progreso(i18n.pick(
                                    'Preparando la lectura en Earth Engine…',
                                    'Preparing the Earth Engine reading…'))

                                def _avance(hechos, total):
                                    _barra.progress(hechos / total, text=i18n.pick(
                                        f'Haciendo el mapa: bloque {hechos} de {total}',
                                        f'Making the map: block {hechos} of {total}'))

                                try:
                                    _pila = geb.pila_para_mapa(
                                        mra.limites_lonlat(_rej), _m_anios, _m_epsg, _bandas_m)
                                    _lee = mra.lector_con_memoria(
                                        lambda c, f, w, h: geb.leer_bloque(
                                            _pila, _m_epsg, *mra.esquina(_rej, c, f), w, h,
                                            _bandas_m),
                                        _dir_bl, _rej)
                                    _amb_arg = None
                                    if _par_amb:
                                        import pyproj as _pyproj

                                        def _leer_relieve(margen_px):
                                            """El relieve de la zona con su margen: se lee
                                            aparte (una sola variable, va deprisa) y también
                                            se guarda, por si el mapa se repite."""
                                            import numpy as _np
                                            _barra.progress(1.0, text=i18n.pick(
                                                'Leyendo el relieve de alrededor, para los '
                                                'cauces…',
                                                'Reading the surrounding relief, for the '
                                                'watercourses…'))
                                            _rej_g = mam.ampliar(_rej, margen_px)
                                            _b_dem = [mam.VARIABLE_RELIEVE]
                                            _pila_d = geb.pila_para_mapa(
                                                mra.limites_lonlat(_rej_g), _m_anios, _m_epsg, _b_dem)
                                            _lee_d = mra.lector_con_memoria(
                                                lambda c, f, w, h: geb.leer_bloque(
                                                    _pila_d, _m_epsg, *mra.esquina(_rej_g, c, f),
                                                    w, h, _b_dem),
                                                _dir_m / 'relieve' / mra.clave_lectura(
                                                    _m_epsg, _m_anios, _b_dem, geb.VERSION_PILA),
                                                _rej_g)
                                            _dem = _np.full((_rej_g['alto'], _rej_g['ancho']),
                                                            _np.nan, dtype='float32')
                                            for _c, _f, _w, _h in mra.bloques(_rej_g):
                                                _dem[_f:_f + _h, _c:_c + _w] = _lee_d(
                                                    _c, _f, _w, _h)[..., 0]
                                            return _dem

                                        _a_utm_m = _pyproj.Transformer.from_crs(
                                            4326, _m_epsg, always_xy=True)
                                        # Los cauces: los de tu capa o, si no, los ríos y
                                        # arroyos de OpenStreetMap (bp.cauces_osm: los de
                                        # Sierra Nevada van con la app, y lee unos 5 km más
                                        # alrededor). Si OpenStreetMap no responde, los del
                                        # relieve, y se avisa con el mapa.
                                        _cau_utm, _de_cau = None, 'relieve'
                                        if _capa_cauces is not None:
                                            _cau_utm = mra.a_utm(_capa_cauces, _m_epsg)
                                            _de_cau = ', '.join(f.name for f in _f_cauces)
                                        else:
                                            from shapely.ops import unary_union as _union
                                            _barra.progress(1.0, text=i18n.pick(
                                                'Leyendo los ríos y arroyos de OpenStreetMap…',
                                                'Reading the OpenStreetMap rivers and '
                                                'streams…'))
                                            _lin_osm, _de_osm = bp.cauces_osm(
                                                mra.limites_lonlat(_rej))
                                            if _lin_osm is not None:
                                                _cau_utm = mra.a_utm(_union(_lin_osm), _m_epsg)
                                                _de_cau = ('OpenStreetMap ('
                                                           + ('capa de la app' if _de_osm == 'capa'
                                                              else 'Overpass') + ')')
                                        _amb_arg = {
                                            'nivel': _nivel, 'params': _par_amb,
                                            'lagunas': [_a_utm_m.transform(lo, la)
                                                        for lo, la in _lagunas_ll],
                                            'cauces': _cau_utm,
                                            'relieve': None if _cau_utm is not None else _leer_relieve,
                                            'fuentes': {
                                                'lagunas': 'lagunas_sierra_nevada.kml '
                                                           f'({len(_lagunas_ll)}) + Sentinel-2',
                                                'cauces': _de_cau}}
                                    _meta = mra.hacer_mapa(
                                        _mod_mapa, _rej, _lee, _carp, zona_utm=_zc['zona_utm'],
                                        mascara_utm=_zc['mascara_utm'], progreso=_avance,
                                        bandas=_bandas_m, pureza=_par_pur, ambiente=_amb_arg,
                                        quitar_agua=_agua_si,
                                        extra={
                                            'anios': _m_anios,
                                            'pila': {'version': geb.VERSION_PILA,
                                                     'estadisticas': geb.ESTADISTICAS_PILA},
                                            'zona': {'tipo': _zona,
                                                     'rectangulo_utm': ([round(v, 2) for v in _lim_z]
                                                                        if _lim_z else None)},
                                            'mascara': (', '.join(f.name for f in _f_masc)
                                                        if _masc is not None else None),
                                            'sin_clase_no_borreguil': not _con_aus,
                                            'modelo': {
                                                'entrenado': _mod['cuando'],
                                                'parametros': _mod['params'],
                                                'balance': _mod['balance']['estrategia'],
                                                'muestras': _mod['n_muestras'],
                                                'macro_f1_validacion': _mod['macro_f1']},
                                            'dos_pasos': _mod_mapa.get('dos_pasos')})
                                    _cons = mra.consultar_varios(_carp, _pts_m)
                                except Exception as _e:
                                    shutil.rmtree(_carp, ignore_errors=True)
                                    _stm.update(label=i18n.pick(
                                        'El mapa no se ha podido terminar',
                                        'The map could not be finished'), state='error')
                                    st.error(i18n.pick(
                                        f'El mapa no se ha podido terminar: {_e}\n\nLos '
                                        'bloques ya leídos están guardados: si vuelves a '
                                        'pulsar el botón, sigue donde iba.',
                                        f'The map could not be finished: {_e}\n\nThe blocks '
                                        'already read are saved: if you press the button '
                                        'again, it carries on where it was.'))
                                else:
                                    _viejo = st.session_state.setdefault('mrf_mapas', {}).get(_nivel)
                                    st.session_state['mrf_mapas'][_nivel] = {
                                        'carpeta': str(_carp), 'meta': _meta, 'vistas': {},
                                        # qué dice el mapa en el píxel de cada muestra
                                        'muestras': [
                                            {**m, 'dentro': q is not None,
                                             'mapa': (q or {}).get('categoria'),
                                             'p_mapa': (q or {}).get('pmax'),
                                             # con el ambiente por cercanía: el que dice el
                                             # mapa en ese píxel y el que revisó la persona
                                             'ambiente_mapa': (q or {}).get('ambiente'),
                                             **_de_revisado.get(
                                                 mr.clave_punto(m['lon'], m['lat']), {})}
                                            for m, q in zip(_mod['muestras'], _cons)]}
                                    if _viejo and _viejo['carpeta'] != str(_carp):
                                        shutil.rmtree(_viejo['carpeta'], ignore_errors=True)
                                    _ha_m = _meta['pixeles_clasificados'] / 100
                                    st.session_state['_mrf_mapa_msg'] = i18n.pick(
                                        f"Mapa hecho en {_dura(_meta['segundos'] / 60, False)}: "
                                        f"{_num(_ha_m)} ha clasificadas.",
                                        f"Map made in {_dura(_meta['segundos'] / 60, False)}: "
                                        f"{_num(_ha_m)} ha classified.")
                                    st.rerun()


            # ---------------------------------------------------- el mapa ya hecho
            _mapa = st.session_state.get('mrf_mapas', {}).get(_nivel)
            if _mapa and mra.leer_meta(_mapa['carpeta']) is None:
                _mapa = None                        # los ficheros ya no están
            _txt_pincha = i18n.pick(
                'Pincha en el mapa para ver la categoría, la probabilidad, la pureza y el '
                'ambiente de un píxel.',
                'Click on the map to see the category, the probability, the purity and the '
                'environment of a pixel.')
            with _col_vi:
                if not _mapa:
                    st.html('<div class="mapa-vacio">' + i18n.pick(
                        'Aquí sale el mapa al hacerlo', 'The map shows here once it is made')
                        + '</div>')
                    st.markdown(f'<span style="color:#AEB6BE">{_txt_pincha}</span>',
                                unsafe_allow_html=True)
            if _mapa:
                _meta = _mapa['meta']
                _cats_m = _meta['categorias']
                _carp_m = Path(_mapa['carpeta'])
                _rej_m = _meta['rejilla']
                # con qué umbral se decidió «¿borreguil o no?» (None: sin esa regla)
                _u_mapa = (_meta.get('regla') or {}).get('umbral')
                _con_pb = (_carp_m / mra.FICHEROS['pborr']).exists()
                # el ambiente por cercanía, si el mapa se hizo con él
                _am = _meta.get('ambiente') or {}
                _con_am = bool(_am) and (_carp_m / mra.FICHEROS['referencias']).exists()
                # las probabilidades son siempre las del modelo
                _cats_p = _meta.get('categorias_modelo') or _cats_m
                # la pureza dentro del píxel, si el mapa se hizo con ella
                _pz = _meta.get('pureza') or {}
                _con_pz = bool(_pz) and (_carp_m / mra.FICHEROS['pureza']).exists()
                # el agua de laguna quitada, si el mapa la tiene
                _con_ag = (_carp_m / mra.FICHEROS['agua']).exists()
                # Los colores del visor, los de la demo (mapeo_raster.COLORES_VISOR): «no
                # borreguil» sin pintar
                # (en el de sí o no, todo el borreguil del mismo verde, sea del ambiente que
                # sea, como en la demo; el ambiente lo dice el píxel pinchado)
                _meta_si_no = _meta.get('nivel') == mr.NIVEL_SI_NO
                _cats_v = [{**c, 'color': (mra.COLORES_VISOR['borreguil']
                                           if _meta_si_no and c['categoria'] != mr.AUSENCIA
                                           else mra.color_visor(c['categoria'], i))}
                           for i, c in enumerate(_cats_m)]
                with _col_vi:
                    _mod_m = _para_el_mapa(_mod)[0]          # en dos pasos, los dos juntos
                    if not _mod_m or _meta.get('firma_modelo') != _mod_m['firma']:
                        st.warning(i18n.pick(
                            'Este mapa se hizo con un modelo anterior al que hay ahora en '
                            '«3 · Modelo». Vuelve a hacerlo para ponerlo al día.',
                            'This map was made with a model older than the one now under '
                            '"3 · Model". Make it again to bring it up to date.'))
                    elif _u_mapa != (_mod_m.get('regla') or {}).get('umbral'):
                        _u_ahora = (_mod_m.get('regla') or {}).get('umbral')
                        st.warning(i18n.pick(
                            'Este mapa se hizo con otro umbral de «¿borreguil o no?» ('
                            + ('ninguno' if _u_mapa is None else f'{_num(_u_mapa)}')
                            + '); el de «3 · Modelo» es ahora '
                            + ('ninguno' if _u_ahora is None else f'{_num(_u_ahora)}')
                            + '. Vuelve a hacerlo para aplicarlo: lo ya leído se aprovecha, '
                            'así que tarda poco.',
                            'This map was made with another "borreguil or not?" threshold ('
                            + ('none' if _u_mapa is None else f'{_num(_u_mapa)}')
                            + '); the one under "3 · Model" is now '
                            + ('none' if _u_ahora is None else f'{_num(_u_ahora)}')
                            + '. Make it again to apply it: what has been read is reused, so '
                            'it takes little time.'))
                    if _meta.get('sin_clase_no_borreguil'):
                        st.info(i18n.pick(
                            'El modelo de este mapa no tiene la clase «no borreguil»: el mapa '
                            'dice de qué tipo sería cada píxel de la máscara, pero no '
                            'demuestra que allí haya borreguil.',
                            'The model of this map has no "not a borreguil" class: the map '
                            'says which type each pixel of the mask would be, but it does '
                            'not prove that there is a borreguil there.'))

                import html as _html

                # ---- visor: el mapa, con su leyenda y lo que dice el píxel pinchado
                from branca.element import MacroElement as _Macro
                from jinja2 import Template as _Plantilla

                class _Guion(_Macro):
                    """Un trozo de JavaScript dentro del visor. Lo escribe `hacer` en el
                    momento de dibujar, cuando cada capa tiene ya su nombre definitivo."""
                    _template = _Plantilla(
                        '{% macro script(this, kwargs) %}{{ this.hacer() }}{% endmacro %}')

                    def __init__(self, hacer):
                        super().__init__()
                        self._name = 'Guion'
                        self.hacer = hacer

                # Lo que pinta la capa «Lagunas y cauces»: con qué se decidió el ambiente.
                # No lleva bloque en la leyenda del mapa (ella pidió quitarlo): sus
                # colores se dicen en «Qué es cada capa».
                _refs_cats = [{'codigo': c, 'categoria': n, 'color': mam.COLORES_REFERENCIAS[c]}
                              for c, n in mam.REFERENCIAS.items()]

                # Va en un fragmento: al moverse por el mapa, pinchar un píxel o cambiar la
                # opacidad se repite solo esta parte, no el guion entero (que con cientos
                # de puntos son segundos).
                @st.fragment
                def _visor_del_mapa():
                    _nombres_m = [c['categoria'] for c in _cats_p]
                    # Los controles van debajo del mapa (la demo), pero el mapa los
                    # necesita antes: se leen de su estado.
                    _prob_de = st.session_state.get('mrf_ver_prob') or ''
                    if _prob_de not in _nombres_m:
                        _prob_de = ''
                    _opac = float(st.session_state.get('mrf_opacidad', 0.7))

                    # ---- lo que el visor dice que está mostrando, y dónde se ha pinchado
                    # (su estado queda en st.session_state con su clave)
                    _clave_v = f'mrf_visor_{_carp_m.name}'
                    _est = st.session_state.get(_clave_v)
                    _centro_m, _zoom_m = mra.encuadre(_meta['limites_lonlat'], 600, 440)
                    _vista, _zoom = mra.leer_vista(_est) or (
                        mra.vista_inicial(_centro_m, _zoom_m), _zoom_m)
                    _clic_m = mra.leer_clic(_est)
                    _q = mra.consultar(_carp_m, *_clic_m) if _clic_m else None

                    # ---- lo fijo del visor. No cambia de una pasada a otra, así que el
                    # visor no se recarga, no pierde el zoom y no olvida qué capas tenía
                    # encendidas la persona. Las capas ráster nacen vacías: la imagen se
                    # les pone después (ver «lo que cambia»).
                    _fm = _base_map(_centro_m, _zoom_m, ortofoto=True)
                    _om, _sm, _em, _nm = _meta['limites_lonlat']
                    _fijas = {
                        'clase': folium.raster_layers.ImageOverlay(
                            image=mra.PNG_VACIO, bounds=[[_sm, _om], [_nm, _em]], opacity=0.7,
                            name=i18n.pick('El mapa de borreguil', 'The borreguil map'),
                            show=True, pixelated=True),
                        'pmax': folium.raster_layers.ImageOverlay(
                            image=mra.PNG_VACIO, bounds=[[_sm, _om], [_nm, _em]], opacity=0.7,
                            name=i18n.pick('Probabilidad máxima', 'Maximum probability'),
                            show=False, pixelated=True),
                        # la de ser borreguil (la suma de los tipos), si el mapa la tiene
                        **({'pborr': folium.raster_layers.ImageOverlay(
                            image=mra.PNG_VACIO, bounds=[[_sm, _om], [_nm, _em]], opacity=0.7,
                            name=i18n.pick('Probabilidad de borreguil',
                                           'Probability of borreguil'),
                            show=False, pixelated=True)} if _con_pb else {}),
                        # la pureza dentro del píxel, si el mapa la tiene
                        **({'pureza': folium.raster_layers.ImageOverlay(
                            image=mra.PNG_VACIO, bounds=[[_sm, _om], [_nm, _em]], opacity=0.7,
                            name=i18n.pick('Pureza dentro del píxel',
                                           'Purity within the pixel'),
                            show=False, pixelated=True)} if _con_pz else {}),
                        # lo tomado por laguna y por cauce, si el ambiente es por cercanía
                        **({'refs': folium.raster_layers.ImageOverlay(
                            image=mra.PNG_VACIO, bounds=[[_sm, _om], [_nm, _em]], opacity=0.7,
                            name=i18n.pick('Lagunas y cauces', 'Lakes and watercourses'),
                            show=False, pixelated=True)} if _con_am else {}),
                        # la de la categoría elegida entra en el recuadro solo si se elige
                        'px': folium.raster_layers.ImageOverlay(
                            image=mra.PNG_VACIO, bounds=[[_sm, _om], [_nm, _em]], opacity=0.7,
                            control=False, show=False, pixelated=True)}
                    for _ov in _fijas.values():
                        _ov.add_to(_fm)
                    folium.Polygon(mra.contorno_lonlat(_rej_m), color='#98A2AB', weight=1.5,
                                   dash_array='6', fill=False).add_to(_fm)
                    # Las muestras no se pintan en este visor: lo pidió ella («quita los
                    # puntos del mapa»), para ver solo lo que dice el modelo.
                    _marca = ('<span style="display:inline-block;width:12px;height:12px;'
                              'border-radius:3px;background:{0};margin-right:6px;'
                              'vertical-align:middle;"></span>')
                    # La leyenda de la demo: «Borreguil» (o «Borreguil húmedo» y «Borreguil
                    # seco») y el agua de laguna quitada; en «Ambiente y humedad», cada
                    # ambiente con su húmedo y su seco.
                    _cats_ley = [c for c in _cats_v if c['categoria'] != mr.AUSENCIA]
                    if _meta_si_no:
                        _ley_html = ('<div style="margin:2px 0;">'
                                     + _marca.format(mra.COLORES_VISOR['borreguil'])
                                     + 'Borreguil</div>')
                    elif all(c['categoria'] in mra.COLORES_VISOR for c in _cats_ley):
                        _ley_html = ''.join(
                            '<div style="margin:2px 0;">' + _marca.format(c['color'])
                            + _html.escape(i18n.pick(f"Borreguil {i18n.tv(c['categoria'])}",
                                                     f"{_cat(c['categoria'])} borreguil"))
                            + '</div>' for c in _cats_ley)
                    else:
                        _ley_html = _leyenda_cats(_cats_ley, lambda col: _marca.format(col))
                    # (el agua de laguna quitada ya no se pinta ni sale en la leyenda: la
                    # autora, 27-11-2026, «quítala del todo»; sigue en las descargas)
                    _fm.get_root().html.add_child(folium.Element(
                        # Los píxeles, con los bordes nítidos; el recuadro de capas y los
                        # botones de zoom, oscuros como la app (la demo); y el recuadro de
                        # capas sin una barra de desplazamiento que aquí nunca hace falta.
                        '<style>.leaflet-image-layer{image-rendering:crisp-edges;'
                        'image-rendering:pixelated}'
                        '.leaflet-control-layers-scrollbar{overflow-y:visible;'
                        'padding-right:0;height:auto!important}'
                        '.leaflet-control-layers{background:rgba(34,31,32,.92);color:#ECEFF1;'
                        'border:1px solid #494245!important;border-radius:10px;'
                        'font:12px Inter,system-ui,sans-serif;box-shadow:0 4px 12px '
                        'rgba(0,0,0,.35)!important}'
                        '.leaflet-control-layers-separator{border-top-color:#494245}'
                        '.leaflet-bar{border:1px solid #60575A!important;border-radius:10px;'
                        'overflow:hidden}'
                        '.leaflet-bar a{background:#2E2A2B;color:#ECEFF1;border-bottom-color:'
                        '#60575A}.leaflet-bar a:hover{background:#393335;color:#7CC496}'
                        '.leaflet-control-attribution{background:rgba(34,31,32,.75)!important;'
                        'color:#AEB6BE}.leaflet-control-attribution a{color:#7CC496}'
                        '.leaflet-control-scale-line{background:rgba(34,31,32,.75);'
                        'color:#CDD3D9;border-color:#98A2AB}</style>'
                        # la leyenda, abajo a la izquierda, por encima de la escala
                        '<div style="position:fixed;bottom:40px;left:12px;z-index:9999;'
                        'background:rgba(34,31,32,0.88);padding:8px 10px;border-radius:10px;'
                        'border:1px solid #494245;font:12px Inter,system-ui,sans-serif;'
                        'line-height:1.25;color:#ECEFF1;pointer-events:none;">'
                        + _ley_html
                        + ('' if not _con_pz else (
                            '<div style="font-weight:600;margin:7px 0 4px;color:#98A2AB;">'
                            + i18n.pick('Pureza dentro del píxel', 'Purity within the pixel')
                            + '</div>' + ''.join(
                                f'<div style="margin:2px 0;">{_marca.format(c["color"])}'
                                f'{_html.escape(_cat(c["categoria"]))}</div>'
                                for c in _pz['categorias'])))
                        + '<div style="font-weight:600;margin:7px 0 3px;color:#98A2AB;">'
                        + i18n.pick('Probabilidad', 'Probability') + '</div>'
                        '<div style="width:130px;height:10px;border-radius:3px;'
                        f'background:linear-gradient(to right,{",".join(mra.RAMPA)});"></div>'
                        '<div style="width:132px;display:flex;justify-content:space-between;'
                        'color:#AEB6BE;">'
                        f'<span>0</span><span>{_num(0.5, 1)}</span><span>1</span></div></div>'))
                    _ctl = folium.LayerControl(collapsed=False)
                    _ctl.add_to(_fm)
                    _Guion(lambda: mra.guion_registro(
                        _fm.get_name(), _ctl.get_name(),
                        {k: v.get_name() for k, v in _fijas.items()})).add_to(_fm)

                    # ---- lo que cambia: la imagen de cada capa, dibujada para lo que se
                    # está viendo y con los píxeles en su sitio (mapeo_raster.
                    # imagen_encuadre), la opacidad y el píxel consultado
                    _vistas = _mapa.setdefault('vistas', {})
                    _imgs = {
                        'clase': mra.dibujar_vista(
                            _vistas.setdefault('clase_v', {}), _carp_m / mra.FICHEROS['clase'],
                            1, _vista, _zoom,
                            lambda m: mra.pintar_clases(m, _cats_v, 1.0,
                                                        tenues=mra.TENUES_VISOR),
                            categorica=True,
                            encima=None),
                        'pmax': mra.dibujar_vista(
                            _vistas.setdefault('pmax', {}), _carp_m / mra.FICHEROS['pmax'], 1,
                            _vista, _zoom, lambda m: mra.pintar_probabilidad(m, 1.0))}
                    if _con_pb:
                        _imgs['pborr'] = mra.dibujar_vista(
                            _vistas.setdefault('pborr', {}), _carp_m / mra.FICHEROS['pborr'],
                            1, _vista, _zoom, lambda m: mra.pintar_probabilidad(m, 1.0))
                    if _con_pz:
                        _imgs['pureza'] = mra.dibujar_vista(
                            _vistas.setdefault('pureza', {}), _carp_m / mra.FICHEROS['pureza'],
                            1, _vista, _zoom,
                            lambda m: mra.pintar_clases(m, _pz['categorias'], 1.0),
                            categorica=True)
                    if _con_am:
                        _imgs['refs'] = mra.dibujar_vista(
                            _vistas.setdefault('refs', {}),
                            _carp_m / mra.FICHEROS['referencias'], 1, _vista, _zoom,
                            lambda m: mra.pintar_clases(m, _refs_cats, 1.0), categorica=True)
                    _elegida, _rotulo = '', ''
                    if _prob_de:
                        _ip = _nombres_m.index(_prob_de)
                        _elegida = f'p{_ip}'
                        _rotulo = i18n.pick(f'Probabilidad de «{_cat(_prob_de)}»',
                                            f'Probability of "{_cat(_prob_de)}"')
                        _imgs['px'] = mra.dibujar_vista(
                            _vistas.setdefault(_elegida, {}), _carp_m / mra.FICHEROS['prob'],
                            _ip + 1, _vista, _zoom, lambda m: mra.pintar_probabilidad(m, 1.0))
                    _g_din = folium.FeatureGroup(name='lo que cambia')
                    _Guion(lambda: mra.guion_capas(_imgs, _opac, _elegida, _rotulo)).add_to(_g_din)
                    if _q:
                        folium.Polygon(mra.contorno_pixel(_rej_m, _q['columna'], _q['fila']),
                                       color='#7CC496', weight=3, fill=False).add_to(_g_din)
                    # El visor avisa de cada movimiento y de cada clic, y entonces se
                    # repite este fragmento.
                    with st.container(key='mapa_visor'):
                        st_folium(_fm, width=None, height=440, feature_group_to_add=_g_din,
                                  returned_objects=['last_clicked', 'bounds', 'zoom'],
                                  pixelated=True, key=_clave_v)

                    # ---- debajo del mapa: la probabilidad de una categoría y la opacidad
                    _v1, _v2 = st.columns(2)
                    if st.session_state.get('mrf_ver_prob') not in [''] + _nombres_m:
                        st.session_state.pop('mrf_ver_prob', None)
                    _v1.selectbox(
                        i18n.pick('Ver también la probabilidad de una categoría',
                                  'Also show the probability of a category'),
                        [''] + _nombres_m, key='mrf_ver_prob',
                        format_func=lambda c: _cat(c) if c else i18n.pick('Ninguna', 'None'),
                        help=i18n.pick(
                            'Añade al recuadro de capas del mapa una capa más, con la '
                            'probabilidad que el modelo da a esa categoría en cada píxel, la '
                            'elija o no. Sirve para ver dónde ve ese tipo de borreguil y las '
                            'zonas de transición. La categoría, la probabilidad máxima y la '
                            'probabilidad de borreguil están siempre en ese recuadro.',
                            'Adds one more layer to the layer box of the map, with the '
                            'probability the model gives to that category at each pixel, '
                            'whether it chooses it or not. Useful to see where it sees that '
                            'type of borreguil and the transition zones. The category, the '
                            'maximum probability and the probability of borreguil are always '
                            'in that box.'))
                    _v2.slider(
                        i18n.pick('Opacidad de las capas', 'Layer opacity'), 0.2, 1.0, 0.7, 0.1,
                        key='mrf_opacidad',
                        help=i18n.pick(
                            'Con menos opacidad se ve la ortofoto por debajo.\n\n'
                            'Recomendado: 0,7.',
                            'With less opacity the orthophoto shows through.\n\n'
                            'Recommended: 0.7.'))
                    st.markdown(f'<span style="color:#AEB6BE">{_txt_pincha}</span>',
                                unsafe_allow_html=True)

                    # ---- lo que dice el mapa en el píxel pinchado (la demo)
                    if _q is None and _clic_m:
                        st.markdown(i18n.pick('Has pinchado fuera del mapa.',
                                              'You clicked outside the map.'))
                    elif _q is not None:
                        def _gr(v, pos, neg):
                            return f"{_num(abs(v), 5)} {pos if v >= 0 else neg}"
                        _lin = [f'<span class="t">' + i18n.pick(
                            f"Píxel en {_gr(_q['lat'], 'N', 'S')} · {_gr(_q['lon'], 'E', 'O')}",
                            f"Pixel at {_gr(_q['lat'], 'N', 'S')} · {_gr(_q['lon'], 'E', 'W')}")
                            + '</span>']
                        if _q['codigo'] is None:
                            _lin.append(i18n.pick(
                                'Sin dato: está fuera de la zona o de la máscara, o le falta '
                                'alguna variable.',
                                'No data: it is outside the zone or the mask, or it lacks '
                                'some variable.'))
                        else:
                            # (en el de sí o no, «Borreguil»; su ambiente va más abajo)
                            _cat_q = ('Borreguil' if _meta_si_no
                                      and _q['categoria'] != mr.AUSENCIA
                                      else _cat(_q['categoria']))
                            _lin.append(i18n.pick('Categoría: ', 'Category: ')
                                        + f"<b>{_html.escape(_cat_q)}</b>")
                            if _q.get('p_borreguil') is not None and _u_mapa is not None:
                                _lin.append(i18n.pick('Probabilidad de borreguil: ',
                                                      'Probability of borreguil: ')
                                            + _num(_q['p_borreguil']))
                            else:
                                _lin.append(i18n.pick('Probabilidad: ', 'Probability: ')
                                            + _num(_q['pmax']))
                            if len(_q['probabilidades']) > 2:
                                _lin.append(i18n.pick('De cada categoría: ', 'Of each '
                                                      'category: ') + ' · '.join(
                                    f"{_html.escape(_cat(p['categoria']))} "
                                    f"{_num(p['probabilidad'])}"
                                    for p in sorted(_q['probabilidades'],
                                                    key=lambda p: -p['probabilidad'])))
                            if _q.get('verde') is not None:
                                _lin.append(i18n.pick('Pureza: ', 'Purity: ')
                                            + i18n.pick(
                                                f"{100 * _q['verde']:.0f} % de verde",
                                                f"{100 * _q['verde']:.0f} % green")
                                            + (f" ({_html.escape(i18n.tv(_q['pureza']))})"
                                               if _q.get('pureza') else ''))
                            if _q.get('ambiente'):
                                _d = _q.get('distancias') or {}
                                _amb = _q['ambiente']
                                if _amb == 'arroyo' and _d.get('cauce') is not None:
                                    _dd = i18n.pick(f", a {_d['cauce']:.0f} m de un cauce",
                                                    f", {_d['cauce']:.0f} m from a watercourse")
                                elif _amb == 'laguna':
                                    _cand = [(v, k) for k, v in _d.items()
                                             if k in ('agua', 'capa') and v is not None]
                                    _v, _k = min(_cand) if _cand else (None, None)
                                    _dd = '' if _v is None else (i18n.pick(
                                        f', a {_v:.0f} m del agua de una laguna',
                                        f', {_v:.0f} m from lake water') if _k == 'agua'
                                        else i18n.pick(f', a {_v:.0f} m de una laguna de la '
                                                       'capa',
                                                       f', {_v:.0f} m from a lake of the '
                                                       'layer'))
                                else:
                                    # ladera: lejos de todo; se dice a cuánto está de cada cosa
                                    def _una(v, es_, en_, nada_es, nada_en):
                                        if v is None:
                                            return i18n.pick(nada_es, nada_en)
                                        return i18n.pick(f'a {v:.0f} m {es_}', f'{v:.0f} m {en_}')
                                    _dd = ' (' + '; '.join((
                                        _una(_d.get('agua'), 'del agua de una laguna',
                                             'from lake water', 'sin agua de laguna a la vista',
                                             'no lake water in sight'),
                                        _una(_d.get('capa'), 'de una laguna de la capa',
                                             'from a lake of the layer',
                                             'a más de 1 km de las lagunas de la capa',
                                             'more than 1 km from the lakes of the layer'),
                                        _una(_d.get('cauce'), 'de un cauce', 'from a watercourse',
                                             'sin cauces cerca', 'no watercourse nearby'))) + ')'
                                _lin.append(i18n.pick('Ambiente: ', 'Environment: ')
                                            + _html.escape(str(i18n.tv(_amb))) + _dd)
                        st.markdown('<div class="mapa-pixel">' + ''.join(
                            f'<span>{x}</span>' for x in _lin) + '</div>',
                            unsafe_allow_html=True)

                    # Una línea por capa, con lo que es (la autora, 27-11-2026: «demasiado
                    # troncho de texto, ordenado una línea por capa con su explicación»).
                    with st.expander(i18n.pick('Qué es cada capa', 'What each layer is')):
                        _lineas_c = [i18n.pick(
                            '**El mapa de borreguil**: la categoría de cada píxel («no '
                            'borreguil» no se pinta, para ver el terreno).',
                            '**The borreguil map**: the category of each pixel ("not a '
                            'borreguil" is not painted, so the ground shows).'),
                            i18n.pick(
                            '**Probabilidad máxima**: lo seguro que está el modelo de la '
                            'categoría que ha puesto, sea la que sea; cuanto más clara, más '
                            'seguro.',
                            '**Maximum probability**: how sure the model is of the category '
                            'it has assigned, whichever it is; the lighter, the surer.')]
                        if _u_mapa is not None and _con_pb:
                            _lineas_c.append(i18n.pick(
                                '**Probabilidad de borreguil**: la suma de la de sus tipos; '
                                f'desde {_num(_u_mapa)}, el mapa pone borreguil.',
                                '**Probability of borreguil**: the sum of those of its types; '
                                f'from {_num(_u_mapa)}, the map says borreguil.'))
                        if _con_am:
                            _lineas_c.append(i18n.pick(
                                '**Lagunas y cauces**: lo que se ha tomado por agua para poner '
                                'el ambiente (turquesa, el agua que ve Sentinel-2; azul, los '
                                'cauces; rosa, las lagunas de la capa).',
                                '**Lakes and watercourses**: what was taken as water to set '
                                'the environment (turquoise, the water Sentinel-2 sees; blue, '
                                'watercourses; pink, the lakes of the layer).'))
                        if _con_pz:
                            _lineas_c.append(i18n.pick(
                                '**Pureza dentro del píxel**: si el borreguil es puro o está '
                                'mezclado con agua o con roca.',
                                '**Purity within the pixel**: whether the borreguil is pure or '
                                'mixed with water or rock.'))
                        st.markdown('\n'.join('- ' + x for x in _lineas_c))
                        st.caption(i18n.pick(
                            'Se encienden en el recuadro «Capas» del mapa. Cada cuadrado es un '
                            'píxel de 10 m de Sentinel-2. El acierto del mapa está en la '
                            'validación por zonas de «3 · Modelo».',
                            'Switch them on in the "Layers" box of the map. Each square is a '
                            '10 m Sentinel-2 pixel. The accuracy of the map is in the '
                            'validation by zones under "3 · Model".'))

                with _col_vi:
                    _visor_del_mapa()

                    # ---- debajo: lo que ocupa cada categoría y las descargas
                    _c_sup, _c_des = st.columns([1.15, 1], gap='small')
                    _n_cl = _meta['pixeles_clasificados']
                    _n_zo = _meta.get('pixeles_zona') or _n_cl
                    with _c_sup:
                        st.markdown('**' + i18n.pick('Superficie de cada categoría',
                                                     'Area of each category') + '**')
                        _k_ha = i18n.pick('Superficie (ha)', 'Area (ha)')
                        _k_pc = i18n.pick('De la zona', 'Of the zone')
                        st.dataframe(_con_coma(pd.DataFrame([{
                            i18n.pick('Categoría', 'Category'): _cat(c['categoria']),
                            _k_ha: float(c['hectareas']),
                            _k_pc: f"{_num(100 * c['pixeles'] / _n_zo, 1)} %" if _n_zo else '—'}
                            for c in sorted(_cats_m, key=lambda c: c['categoria'] == mr.AUSENCIA)
                        ])), hide_index=True, width='stretch')
                        _ag = _meta.get('mascara_agua') or {}
                        _sin = _n_zo - _n_cl
                        st.caption(i18n.pick(
                            f'Zona de {_num(_n_zo / 100)} ha.'
                            + (f" El agua de laguna ({_num(_ag.get('pixeles_agua', 0) / 100)} "
                               'ha) ya está quitada del borreguil.' if _ag else '')
                            + (f' {_num(_sin / 100)} ha de la zona se quedan sin dato: les '
                               'falta alguna variable (nubes o nieve todos los años, borde de '
                               'las imágenes).' if _sin > 0 else ''),
                            f'Zone of {_num(_n_zo / 100)} ha.'
                            + (f" The lake water ({_num(_ag.get('pixeles_agua', 0) / 100)} ha) "
                               'has already been removed from the borreguil.' if _ag else '')
                            + (f' {_num(_sin / 100)} ha of the zone are left without data: they '
                               'lack some variable (cloud or snow every year, edge of the '
                               'images).' if _sin > 0 else '')))
                        # (en el de sí o no, la tabla de arriba ya va por ambiente)
                        if _meta.get('ramas') and _meta.get('nivel') != mr.NIVEL_SI_NO:
                            st.caption(i18n.pick('Por ambiente (húmedo más seco): ',
                                                 'By environment (wet plus dry): ')
                                       + ' · '.join(
                                           f"{_cat(r['rama'])} {_num(r['hectareas'])} ha"
                                           for r in _meta['ramas']) + '.')
                        # ---- la pureza dentro del píxel, de lo que es borreguil
                        if _con_pz:
                            with st.expander(i18n.pick('Pureza dentro del píxel',
                                                       'Purity within the pixel')):
                                _n_bo = sum(c['pixeles'] for c in _pz['categorias'])
                                _k_pb = i18n.pick('% del borreguil', '% of the borreguil')
                                st.dataframe(_con_coma(pd.DataFrame([{
                                    i18n.pick('Pureza', 'Purity'): _cat(c['categoria']),
                                    _k_ha: float(c['hectareas']),
                                    _k_pb: (100 * c['pixeles'] / _n_bo) if _n_bo else 0.0}
                                    for c in _pz['categorias']])), hide_index=True,
                                    width='stretch')
                                st.caption(i18n.pick(
                                    'Estimada con el NDVI de inicio de verano (15 jun – 24 '
                                    f"jul) de cada píxel: sin verde {_num(_pz['sin_verde'])}, "
                                    f"todo verde {_num(_pz['todo_verde'])}; puro desde el "
                                    f"{100 * _pz['puro_desde']:.0f} % de verde. Mixto-agua, si "
                                    'el píxel es agua o toca un píxel de agua.',
                                    'Estimated from the early-summer NDVI (15 Jun – 24 Jul) of '
                                    f"each pixel: no green {_num(_pz['sin_verde'])}, all green "
                                    f"{_num(_pz['todo_verde'])}; pure from "
                                    f"{100 * _pz['puro_desde']:.0f} % green. Mixed-water, if "
                                    'the pixel is water or touches a water pixel.'))
                        if _con_am:
                            with st.expander(i18n.pick('Ambiente por cercanía',
                                                       'Environment by closeness')):
                                # los cauces de OpenStreetMap, o los de una capa subida
                                _am_osm = str((_am.get('fuentes') or {}).get(
                                    'cauces', '')).startswith('OpenStreetMap')
                                st.caption(i18n.pick(
                                    'El ambiente no sale del modelo, sino de la cercanía de '
                                    'cada píxel a lagunas y cauces. '
                                    f"{mam.frase(_am)[:1].upper()}{mam.frase(_am)[1:]}. "
                                    + (('Cauces: los ríos y arroyos de OpenStreetMap.'
                                        if _am_osm else 'Cauces: los de tu capa.')
                                       if _am['cauces'] == 'capa' else
                                       'Cauces: deducidos del relieve (los que reciben el agua '
                                       f"de {_num(_am['cauce_ha'])} ha o más), porque "
                                       'OpenStreetMap no respondió; son una aproximación.')
                                    + ' Lo que se ha tomado por laguna y por cauce se ve en la '
                                    'capa **Lagunas y cauces** del mapa.',
                                    'The environment does not come from the model but from '
                                    'how close each pixel is to lakes and watercourses. '
                                    f"{mam.frase(_am, True)[:1].upper()}"
                                    f"{mam.frase(_am, True)[1:]}. "
                                    + (('Watercourses: the OpenStreetMap rivers and streams.'
                                        if _am_osm else 'Watercourses: those of your layer.')
                                       if _am['cauces'] == 'capa' else
                                       'Watercourses: derived from the relief (those receiving '
                                       f"the water of {_num(_am['cauce_ha'])} ha or more), "
                                       'because OpenStreetMap did not answer; they are an '
                                       'approximation.')
                                    + ' What was taken as lake and as watercourse shows in '
                                    'the **Lakes and watercourses** layer of the map.'))
                                # Lo que revisaste frente a lo que dice la cercanía, en las
                                # muestras de borreguil que caen dentro del mapa.
                                _cmp_a = [(m['ambiente'], m['ambiente_mapa'])
                                          for m in _mapa['muestras']
                                          if m.get('ambiente') and m.get('ambiente_mapa')]
                                if _cmp_a:
                                    _ambs = list(mr.NIVELES[_cercania or 'ambiente'])
                                    st.markdown(i18n.pick(
                                        f'De las **{len(_cmp_a)} muestras** de borreguil que '
                                        'caen en este mapa, el ambiente por cercanía coincide '
                                        'con el que revisaste en '
                                        f'**{sum(a == b for a, b in _cmp_a)}**:',
                                        f'Of the **{len(_cmp_a)} borreguil samples** that fall '
                                        'on this map, the environment by closeness matches the '
                                        'one you reviewed in '
                                        f'**{sum(a == b for a, b in _cmp_a)}**:'))
                                    st.dataframe(pd.DataFrame([{
                                        i18n.pick('Ambiente revisado', 'Reviewed environment'):
                                            i18n.tv(a),
                                        **{i18n.pick(f'Sale {b}', f'Comes out {i18n.tv(b)}'):
                                           sum(1 for x, y in _cmp_a if x == a and y == b)
                                           for b in _ambs}}
                                        for a in _ambs]), hide_index=True, width='stretch')
                        # (las distancias de ahora, de los controles: si no están en
                        # pantalla, no hay nada que comparar)
                        if _con_am:
                            _ahora_d = {'laguna_capa_m': 'mrf_amb_capa',
                                        'laguna_agua_m': 'mrf_amb_agua',
                                        'arroyo_m': 'mrf_amb_arroyo'}
                            if any(st.session_state.get(w) is not None
                                   and abs(_am.get(k, 0.0) - float(st.session_state[w])) > 1e-9
                                   for k, w in _ahora_d.items()):
                                st.warning(i18n.pick(
                                    'Este mapa se hizo con otras distancias de ambiente. '
                                    'Vuelve a hacerlo para aplicar las de ahora: lo ya leído '
                                    'se aprovecha, así que tarda poco.',
                                    'This map was made with other environment distances. Make '
                                    'it again to apply the current ones: what has been read is '
                                    'reused, so it takes little time.'))

                    # ---- los ficheros: los de la demo a la vista y el resto, plegados
                    with _c_des:
                        st.markdown('**' + i18n.pick('Descargas', 'Downloads') + '**')

                        def _fichero(nombre):
                            return lambda: (_carp_m / nombre).read_bytes()

                        def _boton(cl, rotulo, nombre, mime, ayuda=None, datos=None):
                            st.download_button(
                                label=rotulo,
                                data=datos or _fichero(mra.FICHEROS[cl]), file_name=nombre,
                                mime=mime, key=f'mrf_dl_mapa_{cl}', help=ayuda,
                                on_click='ignore')

                        _prob_cl = 'pborr' if _con_pb else 'pmax'
                        with st.container(key='mapa_descargas', horizontal=True, gap='small'):
                            _boton('clase', i18n.pick('Mapa (GeoTIFF)', 'Map (GeoTIFF)'),
                                   f'mapa_{_nivel}_categoria.tif', 'image/tiff')
                            _boton(_prob_cl, i18n.pick('Probabilidad (GeoTIFF)',
                                                       'Probability (GeoTIFF)'),
                                   f'mapa_{_nivel}_probabilidad_borreguil.tif' if _con_pb
                                   else f'mapa_{_nivel}_probabilidad_maxima.tif', 'image/tiff')
                            if _con_pz:
                                _boton('pureza', i18n.pick('Pureza (GeoTIFF)', 'Purity (GeoTIFF)'),
                                       f'mapa_{_nivel}_pureza.tif', 'image/tiff')
                            if _con_ag:
                                _boton('agua', i18n.pick('Agua de laguna (GeoTIFF)',
                                                         'Lake water (GeoTIFF)'),
                                       f'mapa_{_nivel}_agua_de_laguna.tif', 'image/tiff')
                            _boton('png', i18n.pick('Imagen PNG', 'PNG image'),
                                   f'mapa_{_nivel}.png', 'image/png',
                                   datos=lambda: mra.png_del_mapa(_carp_m, _cats_v))
                            _boton('codigos', i18n.pick('Superficie (CSV)', 'Area (CSV)'),
                                   f'mapa_{_nivel}_codigos.csv', 'text/csv')
                        # Los demás ficheros del mapa (los de antes de la demo), plegados
                        _mas = (
                            *([('pmax', f'mapa_{_nivel}_probabilidad_maxima.tif', i18n.pick(
                                'La probabilidad, de 0 a 1, de la categoría elegida en cada '
                                'píxel.',
                                'The probability, from 0 to 1, of the category chosen at each '
                                'pixel.'))] if _con_pb else []),
                            ('prob', f'mapa_{_nivel}_probabilidad_por_categoria.tif', i18n.pick(
                                'Una banda por categoría, con su nombre: la probabilidad de '
                                'cada una en cada píxel.',
                                'One band per category, with its name: the probability of '
                                'each one at each pixel.')),
                            *([('verde', f'mapa_{_nivel}_verde_estimado.tif', i18n.pick(
                                'Cuánto verde hay dentro de cada píxel, de 0 a 1, estimado con '
                                'su NDVI de inicio de verano (15 jun – 24 jul). De todos los '
                                'píxeles, sean o no borreguil.',
                                'How much green there is within each pixel, from 0 to 1, '
                                'estimated from its early-summer NDVI (15 Jun – 24 Jul). For '
                                'every pixel, borreguil or not.'))] if _con_pz else []),
                            *([('ambiente', f'mapa_{_nivel}_ambiente.tif', i18n.pick(
                                'El ambiente de cada píxel por cercanía a lagunas y cauces, '
                                'como código: 1 arroyo, 2 laguna, 3 ladera (255 = sin dato). '
                                'De todos los píxeles, sean o no borreguil.',
                                'The environment of each pixel by closeness to lakes and '
                                'watercourses, as a code: 1 stream, 2 lake, 3 slope (255 = no '
                                'data). For every pixel, borreguil or not.')),
                               ('distancias', f'mapa_{_nivel}_distancias_ambiente.tif',
                                i18n.pick('Tres bandas, en metros: la distancia de cada píxel '
                                          'al agua de una laguna, a una laguna de la capa y a '
                                          'un cauce. De ellas sale el ambiente.',
                                          'Three bands, in metres: the distance from each '
                                          'pixel to lake water, to a lake of the layer and to '
                                          'a watercourse. The environment comes from them.')),
                               ('referencias', f'mapa_{_nivel}_lagunas_y_cauces.tif',
                                i18n.pick('Lo que se ha tomado por laguna y por cauce: 1 agua '
                                          'de laguna (Sentinel-2), 2 cauce, 3 laguna de la '
                                          'capa.',
                                          'What was taken as lake and as watercourse: 1 lake '
                                          'water (Sentinel-2), 2 watercourse, 3 lake of the '
                                          'layer.'))] if _con_am else []))
                        with st.expander(i18n.pick('Más ficheros', 'More files')):
                            # Uno debajo de otro: los nombres son largos.
                            for _cl, _nom, _ay in _mas:
                                _mb = (_carp_m / mra.FICHEROS[_cl]).stat().st_size / 1e6
                                _boton(_cl, f'{_nom} ({_num(_mb, 1)} MB)' if _mb >= 0.1
                                       else _nom, _nom, 'image/tiff', ayuda=_ay)

        # ------------------------------------------- 5 · Peso (de las variables)
        # Cuánto pesa cada variable en el modelo. Se mide al entrenar, de dos
        # maneras (mapeo_rf.entrenar_y_validar); aquí solo se enseña. No se quita
        # ninguna variable: eso lo decide quien usa la app, en «1 · Datos».
        with _t_peso, st.container(key='tarjeta_mapa_peso'):
            import altair as alt
            _imp = (_mod or {}).get('importancia')
            _med_txt = {
                'permutacion': i18n.pick('Lo que pierde el modelo sin ella',
                                         'What the model loses without it'),
                'impureza': i18n.pick('Uso interno', 'Internal use')}
            _med = 'permutacion'
            with st.container(horizontal=True, vertical_alignment='bottom', gap='medium',
                              key='mapa_peso_cabeza'):
                st.markdown('##### ' + i18n.pick(
                    'Cuánto pesa cada variable en el modelo del mapa',
                    'How much each variable weighs in the map model'))
                if _mod and _imp:
                    _med = st.radio(
                        i18n.pick('Qué medida dibujar', 'Measure to plot'), list(_med_txt),
                        key='mrf_peso_medida', format_func=lambda m: _med_txt[m],
                        horizontal=True, width='content',
                        help=i18n.pick(
                            'Lo que pierde el modelo sin ella se mide con muestras que el '
                            'modelo no ha visto (las de la validación): dice qué variables '
                            'necesita para acertar en sitios nuevos.\n\n'
                            'Uso interno se mide con las mismas muestras con las que se '
                            'entrena: da algo de peso a todas las variables que el modelo usa, '
                            'también a las que no le ayudan a acertar fuera.\n\n'
                            'Recomendado: la primera para decidir qué variables importan. La '
                            'segunda es la que suele darse en los artículos.',
                            'What the model loses without it is measured with samples the '
                            'model has not seen (those of the validation): it tells which '
                            'variables it needs to get new places right.\n\n'
                            'Internal use is measured with the same samples used for '
                            'training: it gives some weight to every variable the model uses, '
                            'including those that do not help it elsewhere.\n\n'
                            'Recommended: the first one to decide which variables matter. The '
                            'second is the one usually reported in papers.'))
            if not _mod:
                st.info(i18n.pick(
                    'Todavía no hay modelo en este nivel. Entrénalo en «3 · Modelo» y '
                    'vuelve aquí.',
                    'There is no model at this level yet. Train it under "3 · Model" and '
                    'come back here.'))
            elif not _imp:
                st.info(i18n.pick(
                    'Este modelo se entrenó antes de que la app midiera el peso de las '
                    'variables. Pulsa otra vez **Entrenar y validar** en «3 · Modelo»: son '
                    'unos segundos y no hay que volver a leer nada en Earth Engine.',
                    'This model was trained before the app measured the weight of the '
                    'variables. Press **Train and validate** again under "3 · Model": it '
                    'takes a few seconds and nothing has to be read from Earth Engine '
                    'again.'))
            else:
                if _tab is None or _mod['firma'] != mr.firma_tabla(_tab):
                    st.warning(i18n.pick(
                        'La tabla ha cambiado desde que se entrenó este modelo. Lo de abajo '
                        'corresponde al modelo anterior: vuelve a entrenar en «3 · Modelo» '
                        'para ponerlo al día.',
                        'The table has changed since this model was trained. What follows '
                        'belongs to the previous model: train again under "3 · Model" to '
                        'bring it up to date.'))
                _vs = _imp['variables']
                _perm = _med == 'permutacion'
                # Las barras van encima de «Mirar por» (la demo), pero dependen de él: se
                # deja su hueco y se rellena después.
                _hueco_barras = st.container()
                # Mirar por variable o por grupo de EBVs (la autora lo confirmó el
                # 26-11-2026; sustituye a «Familias» y «Grupos de variables parecidas»).
                _por = st.radio(
                    i18n.pick('Mirar por', 'Look by'), ['variable', 'ebv'],
                    horizontal=True, key='mrf_peso_por',
                    format_func=lambda k: (i18n.pick('Variable', 'Variable')
                                           if k == 'variable' else
                                           i18n.pick('Grupo de EBVs', 'EBV group')),
                    help=i18n.pick(
                        'Variable: cada variable del modelo por separado (por ejemplo, el NDVI '
                        'de inicio de verano (15 jun – 24 jul) o la altitud), con su peso.\n\n'
                        'Grupo de EBVs: las variables juntas según la variable esencial de '
                        'biodiversidad (EBV, Essential Biodiversity Variables, de GEO BON) que '
                        'miden de la vegetación o del agua: productividad y vigor (NDVI, EVI, '
                        'WAVI), contenido de agua de la vegetación (NDMI), clorofila (borde del '
                        'rojo), fenología (la caída del NDVI del inicio de verano, 15 jun – 24 jul, al fin de verano, 1 ago – 29 sep), agua en '
                        'superficie (NDWI), cubierta y brillo del suelo (albedo) y '
                        'perturbaciones (NBR). El relieve va aparte: no es una EBV, es el '
                        'contexto físico. El peso de un grupo es la suma del de sus '
                        'variables.\n\n'
                        'Para qué: las variables de un mismo grupo dicen casi lo mismo y se '
                        'reparten el peso, así que cada una parece pesar poco; juntas dicen '
                        'cuánto importa esa característica del borreguil.\n\n'
                        'Recomendado: mirar los dos.',
                        'Variable: each variable of the model on its own (for instance, '
                        'early-summer NDVI (15 Jun – 24 Jul) or elevation), with its weight.\n\n'
                        'EBV group: the variables together according to the essential '
                        'biodiversity variable (EBV, from GEO BON) they measure of the '
                        'vegetation or the water: productivity and vigour (NDVI, EVI, WAVI), '
                        'vegetation water content (NDMI), chlorophyll (red edge), phenology '
                        '(the NDVI drop from early summer, 15 Jun – 24 Jul, to late summer, 1 Aug – 29 Sep), surface water (NDWI), ground cover '
                        'and brightness (albedo) and disturbance (NBR). Terrain goes apart: it '
                        'is not an EBV, it is the physical context. The weight of a group is '
                        'the sum of its variables.\n\n'
                        'What for: the variables of the same group say almost the same and '
                        'share the weight, so each one seems to weigh little; together they '
                        'say how much that feature of the borreguil matters.\n\n'
                        'Recommended: look at both.'))

                def _barras(df, campo, titulo):
                    """Barras horizontales en verde, con su valor (3 decimales) al lado, como
                    en la demo: sin eje, de la que más pesa a la que menos."""
                    df = df.copy()
                    df['txt'] = [_num(v) for v in df['valor']]
                    df['pos'] = [max(0.0, float(v)) for v in df['valor']]
                    _y = alt.Y(f'{campo}:N', sort=list(df[campo]), title=None,
                               axis=alt.Axis(labelLimit=440, labelOverlap=False, ticks=False,
                                             domain=False, labelPadding=10, labelFontSize=13,
                                             labelColor='#CDD3D9'))
                    _x = alt.X('valor:Q', title=titulo, axis=None,
                               scale=alt.Scale(domainMin=min(0.0, float(df['valor'].min())),
                                               domainMax=float(df['valor'].max()) * 1.18
                                               or 1.0))
                    _b = alt.Chart(df)
                    _tip = [alt.Tooltip(f'{campo}:N', title=campo)] + [
                        alt.Tooltip(f'{c}:N', title=t) for c, t in (
                            ('que', i18n.pick('qué mide', 'what it measures')),
                            ('pierde', i18n.pick('pierde el modelo sin ella',
                                                 'the model loses without it')),
                            ('uso', i18n.pick('uso interno', 'internal use'))) if c in df]
                    return alt.layer(
                        _b.mark_bar(size=12, color='#7CC496', cornerRadiusEnd=3).encode(
                            x=_x, y=_y, tooltip=_tip),
                        _b.mark_text(align='left', dx=6, fontSize=12, color='#AEB6BE').encode(
                            x=alt.X('pos:Q', axis=None), y=_y, text='txt:N')
                    ).properties(height=24 * len(df))

                with _hueco_barras:
                    if _por == 'ebv':
                        _suma = {}
                        for v in _vs:
                            _g = mr.grupo_ebv(v['variable'])
                            _acu = _suma.setdefault(_g, [0.0, 0])
                            _acu[0] += v['permutacion'] if _perm else v['impureza']
                            _acu[1] += 1
                        _dfe = pd.DataFrame([
                            {'grupo': f'{mr.nombre_ebv(g, _en)} · {n}', 'valor': x, 'n': n}
                            for g, (x, n) in sorted(_suma.items(), key=lambda e: -e[1][0])])
                        st.altair_chart(_barras(_dfe, 'grupo', _med_txt[_med]), width='stretch')
                        st.caption(i18n.pick(
                            'El peso de cada grupo es la suma del de sus variables; tras el '
                            'nombre, cuántas tiene el modelo.',
                            'The weight of each group is the sum of its variables; after the '
                            'name, how many the model has.'))
                    else:
                        _fil = sorted(_vs, key=lambda v: v['orden' if _perm else
                                                          'orden_impureza'])
                        _dfp = pd.DataFrame([{
                            'variable': _var_m(v['variable']),
                            'que': mr.que_mide(v['variable'], _en),
                            'grupo': mr.nombre_grupo(v['grupo'], _en),
                            'valor': v['permutacion'] if _perm else v['impureza'],
                            'pierde': f"{_num(v['permutacion'], signo=True)} ± "
                                      f"{_num(v['desviacion'])}",
                            'uso': f"{_num(100 * v['impureza'])} %",
                            'puesto': f"{v['orden']} · {v['orden_impureza']}"} for v in _fil])
                        st.altair_chart(_barras(_dfp, 'variable', _med_txt[_med]),
                                        width='stretch')

                # ---- Las que más pesan, grupo a grupo: cómo son en las muestras de cada
                # categoría del modelo (25-11-2026, la autora: «además de peso, podría
                # mostrar gráficos de las variables por tipo de borreguil como en los
                # puntos de verdad terreno»). Con las muestras de la tabla de
                # entrenamiento: sus categorías son las que aprende el modelo.
                if _tab and _tab.get('filas'):
                    with st.container(key='mapa_grupo_a_grupo', gap='small'):
                        st.markdown('##### ' + i18n.pick('Las que más pesan, grupo a grupo',
                                                         'The ones that weigh most, group by '
                                                         'group'))
                        _pred_tab = set(_tab.get('predictores') or [])
                        _ord_k = 'orden' if _perm else 'orden_impureza'
                        _orden_m = [v['variable'] for v in sorted(_vs, key=lambda v: v[_ord_k])
                                    if v['variable'] in _pred_tab]
                        # el peso junto al nombre, cuando se sabe (la medida elegida arriba,
                        # como parte del total)
                        _tot_p = sum(max(0.0, v['permutacion'] if _perm else v['impureza'])
                                     for v in _vs) or 1.0
                        _peso_de = {v['variable']: max(0.0, v['permutacion'] if _perm
                                                       else v['impureza']) / _tot_p
                                    for v in _vs}

                        def _con_peso(f):
                            p = _peso_de.get(f, 0.0)
                            return (_var_m(f) + (f' · {round(100 * p)} %' if p >= 0.005
                                                 else ''))
                        _vistas_m = {
                            'cajas': i18n.pick('Cajas', 'Boxes'),
                            'curvas': i18n.pick('Curvas de distribución', 'Distribution curves'),
                            'diferencias': i18n.pick('Diferencia estandarizada de medianas',
                                                     'Standardized median difference'),
                            'dos': i18n.pick('Dos variables a la vez', 'Two variables at once')}
                        _filas_m = _tab['filas']
                        if _cercania:
                            # En «Ambiente y humedad» los grupos son los ambientes (arroyo,
                            # laguna, ladera) y «no borreguil», como en la demo: el ambiente
                            # que revisó la persona en cada muestra (el modelo aprende solo
                            # la humedad). Las que no lo tienen revisado no entran.
                            def _amb_de(f):
                                if f.get('categoria') == mr.AUSENCIA:
                                    return mr.AUSENCIA
                                r = rows[f['indice']] if 0 <= f['indice'] < len(rows) else {}
                                return (mr.categoria(r, _cercania)
                                        if _cercania in bp.niveles_revisados(r) else None)
                            _filas_m = [{**f, 'categoria': _amb_de(f)} for f in _filas_m
                                        if _amb_de(f)]
                        # Los grupos: las categorías del modelo, en el orden del catálogo y
                        # «no borreguil» la última, con los colores de la demo (relleno y
                        # borde, aclarados para el fondo oscuro)
                        _DEMO_GR = {'borreguil': ('#2E8B4A', '#8FD9A8'),
                                    'no borreguil': ('#B23B3B', '#F4A3A3'),
                                    'húmedo': ('#0E8063', '#7FE0C2'),
                                    'seco': ('#B4561C', '#F5B48A'),
                                    'arroyo': ('#23801B', '#9BE08F'),
                                    'laguna': ('#2D5FA3', '#9CC2F2'),
                                    'ladera': ('#8C9226', '#E2E68E')}
                        _cnt_m = Counter(f.get('categoria') for f in _filas_m)
                        _cats_g = [c for c in ([c['categoria'] for c in _res['clases']]
                                               if not _cercania else
                                               list(mr.NIVELES[_cercania]))
                                   if c != mr.AUSENCIA and _cnt_m.get(c)]
                        _cats_g += [c for c in _cnt_m if c and c not in _cats_g
                                    and c != mr.AUSENCIA]
                        if _cnt_m.get(mr.AUSENCIA):
                            _cats_g.append(mr.AUSENCIA)
                        _gr_m = [(c, _cat(c)) + _DEMO_GR.get(c, (mra.color(c, i),
                                                                 mra.color(c, i)))
                                 for i, c in enumerate(_cats_g)]
                        _cuantos_m = ' · '.join(f'{n}: {_cnt_m[c]}' for c, n, _, _ in _gr_m)
                        # «Cómo verlo» en su fila y, debajo, en una sola línea, los dos ejes
                        # (o cuántas variables) (la autora, 27-11-2026).
                        _vista_m = st.radio(i18n.pick('Cómo verlo', 'How to see it'),
                                            list(_vistas_m), format_func=_vistas_m.get,
                                            key='mrf_dist_vista', horizontal=True,
                                            width='content')
                        if _vista_m != 'dos':
                            _max_n = max(1, len(_orden_m))
                            _n_m = int(st.number_input(
                                i18n.pick(f'Cuántas variables (de 1 a {_max_n})',
                                          f'How many variables (1 to {_max_n})'),
                                1, _max_n, min(8, _max_n), 1, key='mrf_dist_n',
                                width=220))
                        elif len(_orden_m) >= 2:
                            # (lo que no sea una variable de este modelo, fuera)
                            for _k_xy in ('mrf_dist_x', 'mrf_dist_y'):
                                if st.session_state.get(_k_xy) not in _orden_m:
                                    st.session_state.pop(_k_xy, None)
                            _e_x, _e_y = st.columns(2, gap='small')
                            _vxm = _e_x.selectbox(i18n.pick('Eje horizontal', 'Horizontal axis'),
                                                  _orden_m, index=0, format_func=_var_m,
                                                  key='mrf_dist_x')
                            _vym = _e_y.selectbox(i18n.pick('Eje vertical', 'Vertical axis'),
                                                  _orden_m, index=1, format_func=_var_m,
                                                  key='mrf_dist_y')
                        _los_grupos = i18n.pick(
                            ' Los grupos son las categorías de lo que eliges arriba, en «Qué '
                            'quieres cartografiar».',
                            ' The groups are the categories of what you choose above, under '
                            '"What to map".')
                        if _vista_m != 'dos':
                            _vars_m = [(f, _con_peso(f)) for f in _orden_m[:_n_m]]
                            st.caption(i18n.pick(
                                f'Las {len(_vars_m)} variables que más pesan en el modelo, de '
                                'más a menos, con el peso junto al nombre cuando se sabe. '
                                f'Muestras: {_cuantos_m}.',
                                f'The {len(_vars_m)} variables that weigh most in the model, '
                                'from most to least, with the weight next to the name when it '
                                f'is known. Samples: {_cuantos_m}.') + _los_grupos)
                        else:
                            st.caption(i18n.pick(
                                'Las variables del modelo, de más a menos peso; de entrada, '
                                f'las dos que más pesan. Muestras: {_cuantos_m}.',
                                "The model's variables, from most to least weight; by default, "
                                f'the two that weigh most. Samples: {_cuantos_m}.')
                                + _los_grupos)
                        st.html('<div class="mapa-leyenda">' + ''.join(
                            f'<span><i style="background:{rel};border:2px solid {bor}"></i>'
                            f'{n} ({_cnt_m[c]})</span>' for c, n, rel, bor in _gr_m)
                            + '</div>')
                        # Las gráficas de point_profile, con los colores de la demo (el borde
                        # de cada grupo) y sin su leyenda (va encima, la de la demo)
                        _a_demo = {mra.color(c, i): bor for i, (c, _, _, bor) in enumerate(_gr_m)}

                        def _sin_leyenda(spec):
                            spec.setdefault('config', {}).setdefault('legend', {})['disable'] = True
                            return spec
                        _ch_m, _spec_m, _como_m = None, None, ''
                        if _vista_m == 'cajas':
                            _ch_m = _cajas_mapa(_filas_m, _vars_m, _gr_m)
                            _como_m = i18n.pick(
                                'Cada caja va de la mitad central de las muestras de un grupo '
                                '(del 25 % al 75 %), con la mediana en blanco; las líneas '
                                'llegan del 10 % al 90 %. Si las cajas de dos grupos no se '
                                'tocan, esa variable los separa bien. Pasa el ratón por una '
                                'caja para ver sus cifras.',
                                'Each box spans the middle half of the samples of a group '
                                '(from 25 % to 75 %), with the median in white; the lines reach '
                                'from 10 % to 90 %. If the boxes of two groups do not touch, '
                                'that variable separates them well. Hover over a box to see its '
                                'figures.')
                        elif _vista_m == 'curvas':
                            _c0 = pp.chart_densidades(_filas_m, _vars_m, comparar='categoria',
                                                      traduce=_cat)
                            _spec_m = _sin_leyenda(_recolorear(_c0, _a_demo)) if _c0 else None
                            _como_m = i18n.pick(
                                'Cada curva muestra cómo se reparten los valores de un grupo; '
                                'todas encierran la misma área. Si apenas se tocan, esa '
                                'variable separa bien los grupos.',
                                'Each curve shows how the values of a group are spread; they '
                                'all enclose the same area. If they hardly touch, that '
                                'variable separates the groups well.')
                        elif _vista_m == 'diferencias':
                            _c0 = pp.chart_diferencias(_filas_m, _vars_m, comparar='categoria',
                                                       traduce=_cat)
                            _spec_m = _sin_leyenda(_recolorear(_c0, _a_demo)) if _c0 else None
                            _como_m = i18n.pick(
                                'Cada punto dice cuánto se aparta la mediana del grupo de la '
                                'de «no borreguil», en desviaciones típicas de todas las '
                                'muestras: a la derecha, más; a la izquierda, menos. Pasar de '
                                '1 (o de −1) es una diferencia clara. Pasa el ratón por un '
                                'punto para ver su valor.',
                                'Each dot says how far the median of the group is from that of '
                                '"not a borreguil", in standard deviations of all the samples: '
                                'to the right, more; to the left, less. Beyond 1 (or −1) is a '
                                'clear difference. Hover over a dot to see its value.')
                        elif len(_orden_m) >= 2:
                            _c0 = pp.chart_dos_variables(
                                _filas_m, _vxm, _vym, _var_m(_vxm), _var_m(_vym),
                                comparar='categoria', traduce=_cat,
                                nombres=bp.nombres_de_punto([f.get('ID') for f in _filas_m],
                                                            i18n.pick('Punto', 'Point')))
                            _spec_m = _sin_leyenda(_recolorear(_c0, _a_demo)) if _c0 else None
                            _como_m = i18n.pick(
                                'Cada punto es una muestra, colocada según las dos variables. '
                                'Si las nubes de colores se separan, combinar esas dos '
                                'variables ayuda a distinguir los grupos. Pasa el ratón por un '
                                'punto para ver sus valores.',
                                'Each dot is a sample, placed by the two variables. If the '
                                'coloured clouds separate, combining those two variables helps '
                                'to tell the groups apart. Hover over a dot to see its '
                                'values.')
                        if _ch_m is not None:
                            st.altair_chart(_ch_m, width='stretch')
                            st.caption(_como_m)
                        elif _spec_m is not None:
                            st.vega_lite_chart(spec=_spec_m, width='stretch')
                            st.caption(_como_m)
                        else:
                            st.info(i18n.pick(
                                'No hay datos suficientes para este gráfico (por ejemplo, sin '
                                'muestras de «no borreguil» en este nivel no hay con qué '
                                'comparar las diferencias).',
                                'There is not enough data for this chart (for instance, '
                                'without "not a borreguil" samples at this level the '
                                'differences have nothing to be compared with).'))

                with st.expander(i18n.pick('Cómo leerlo', 'How to read it')):
                    st.markdown(i18n.pick(
                        'Una variable con mucho peso es la que más le sirve al modelo para '
                        'acertar, no la causa de que haya borreguil.\n\n'
                        'Si dos variables dicen casi lo mismo (los índices de verdor, por '
                        'ejemplo), se reparten el peso, y cada una parece pesar menos de lo que '
                        'pesa su información. Por eso se puede mirar también por grupo de '
                        'EBVs.\n\n'
                        'Con pocas muestras, el orden cambia de un entrenamiento a otro: si al '
                        'cambiar la semilla en «3 · Modelo» cambia mucho, no te fíes del orden '
                        'fino, solo de las que van siempre arriba.',
                        'A variable with a lot of weight is the one that most helps the model '
                        'to be right, not the cause of there being borreguil.\n\n'
                        'If two variables say almost the same (the greenness indices, for '
                        'instance), they share the weight, and each one seems to weigh less '
                        'than its information does. That is why it can also be looked at by '
                        'EBV group.\n\n'
                        'With few samples, the order changes from one training to the next: '
                        'if it changes a lot when you change the seed under "3 · Model", do not '
                        'trust the fine order, only those that are always at the top.'))
                st.download_button(
                    label=i18n.pick('Descargar peso_variables.csv',
                                    'Download peso_variables.csv'),
                    data=mr.importancia_csv(_mod, _en).encode('utf-8-sig'),
                    file_name=f'peso_variables_{_nivel}.csv', mime='text/csv',
                    key='mrf_dl_peso', type='tertiary')
                # ---- todas, una por una, con sus números (lo que había antes de la demo,
                # plegado): lo mismo que el gráfico, con el «±» y el puesto en cada medida
                with st.expander(i18n.pick('Ver todas las cifras', 'See all the figures')):
                    _c4v = st.column_config.NumberColumn()
                    _c2v = st.column_config.NumberColumn()
                    _k_pv = i18n.pick('Bajada del F1 sin ella', 'F1 drop without it')
                    _k_ui = i18n.pick('Uso interno (%)', 'Internal use (%)')
                    st.dataframe(_con_coma(pd.DataFrame([{
                        'Variable': _var_m(v['variable']),
                        i18n.pick('Puesto', 'Rank'): v['orden'],
                        _k_pv: v['permutacion'], '±': v['desviacion'],
                        _k_ui: 100 * v['impureza'],
                        i18n.pick('Puesto por uso', 'Rank by use'): v['orden_impureza'],
                        i18n.pick('Qué mide', 'What it measures'):
                            mr.que_mide(v['variable'], _en),
                        i18n.pick('Grupo de EBVs', 'EBV group'):
                            mr.nombre_ebv(mr.grupo_ebv(v['variable']), _en),
                        i18n.pick('Época u origen', 'Season or source'):
                            mr.nombre_grupo(v['grupo'], _en)}
                        for v in _vs])), hide_index=True, width='stretch',
                        column_config={_k_pv: _c4v, '±': _c4v, _k_ui: _c2v,
                                       'Variable': st.column_config.Column(pinned=True)})
                    st.caption(i18n.pick(
                        'Bajada del F1 sin ella: cuánto baja el F1 medio de la validación al '
                        'desordenar esa variable (se repite varias veces; «±» es lo que cambia '
                        'de una vez a otra). Si la bajada es menor que su «±», no se distingue '
                        'de cero.',
                        'F1 drop without it: how much the mean F1 of the validation drops '
                        'when that variable is shuffled (it is repeated several times; "±" is '
                        'how much it changes from one time to the next). If the drop is '
                        'smaller than its "±", it cannot be told apart from zero.'))

        # ------------------------------------------------ por dónde va el trabajo
        # Un rótulo por paso, con icono y texto además del color.
        _hecho, _ojo, _falta = 'green', 'orange', 'gray'
        if not _predictores:
            _e1 = (_ojo, i18n.pick('elige alguna variable', 'choose some variable'))
        elif not _res['listo']:
            _e1 = (_ojo, i18n.pick('faltan muestras', 'not enough samples'))
        else:
            _e1 = (_hecho, i18n.pick(f'{_nm} muestras', f'{_nm} samples'))
        if _tab is None:
            _e2 = (_falta, i18n.pick('sin leer', 'not read yet'))
        elif _pend:
            _e2 = (_ojo, i18n.pick(
                f"falta{'' if len(_pend) == 1 else 'n'} {len(_pend)} "
                f"muestra{'' if len(_pend) == 1 else 's'} por leer",
                f"{len(_pend)} sample{'' if len(_pend) == 1 else 's'} still to read"))
        elif not _tab['listo']:
            _e2 = (_ojo, i18n.pick('no basta para entrenar', 'not enough to train'))
        else:
            _e2 = (_hecho, i18n.pick(
                f"{len(_tab['filas'])} muestras, {len(_tab['predictores'])} variables",
                f"{len(_tab['filas'])} samples, {len(_tab['predictores'])} variables")
                + (' (VIF)' if _tab.get('vif') else ' (PCA)' if _tab.get('pca') else ''))
        if not _mod:
            _e3 = (_falta, i18n.pick('sin entrenar', 'not trained yet'))
        elif _tab is None or _mod['firma'] != mr.firma_tabla(_tab):
            _e3 = (_ojo, i18n.pick('es de una tabla anterior', 'from a previous table'))
        else:
            _e3 = (_hecho, i18n.pick(f"F1 medio {_num(_mod['macro_f1'])}",
                                     f"mean F1 {_num(_mod['macro_f1'])}"))
        _mod_e4 = _para_el_mapa(_mod)[0]           # en dos pasos, los dos juntos
        if not _mapa:
            _e4 = (_falta, i18n.pick('sin hacer', 'not made yet'))
        elif not _mod_e4 or _mapa['meta'].get('firma_modelo') != _mod_e4['firma']:
            _e4 = (_ojo, i18n.pick('es de un modelo anterior', 'from a previous model'))
        elif _e3[0] != _hecho:
            _e4 = (_ojo, i18n.pick('es de una tabla anterior', 'from a previous table'))
        elif ((_mapa['meta'].get('regla') or {}).get('umbral')
              != (_mod_e4.get('regla') or {}).get('umbral')):
            _e4 = (_ojo, i18n.pick('es de otro umbral', 'made with another threshold'))
        else:
            _e4 = (_hecho, i18n.pick(
                f"{_num(_mapa['meta']['pixeles_clasificados'] / 100)} ha clasificadas",
                f"{_num(_mapa['meta']['pixeles_clasificados'] / 100)} ha classified"))
        _imp_e = (_mod or {}).get('importancia')
        if not _mod:
            _e5 = (_falta, i18n.pick('tras entrenar', 'after training'))
        elif not _imp_e:
            _e5 = (_ojo, i18n.pick('vuelve a entrenar', 'train again'))
        elif _e3[0] != _hecho:
            _e5 = (_ojo, i18n.pick('es de una tabla anterior', 'from a previous table'))
        elif _imp_e['variables'][0]['permutacion'] <= 0:
            _e5 = (_ojo, i18n.pick('ninguna destaca', 'none stands out'))
        else:
            _e5 = (_hecho, i18n.pick('listo', 'ready'))
        # Los cinco pasos, como en la demo: «Paso 1 · Datos · 22 muestras», el que se ve en
        # verde. Lo que dice cada uno, con su color: hecho (verde claro), a medias o con
        # algo que mirar (ámbar) o sin empezar (gris).
        _tono = {_hecho: '#A3D9B5', _ojo: '#F5B83D', _falta: '#98A2AB'}
        _activo = (_rot_pasos.index(st.session_state['mrf_paso'])
                   if st.session_state.get('mrf_paso') in _rot_pasos else 0)

        def _ir_paso(i):
            st.session_state['_mrf_paso_n'] = i
            st.session_state['mrf_paso'] = _rot_pasos[i]

        with _estado:
            st.html('<style>' + ''.join(
                f'.st-key-mrf_ir_{_k} button p:last-child{{color:{_tono[_c]}}}'
                for (_k, _), (_c, _) in zip(_PASOS_MAPA, (_e1, _e2, _e3, _e4, _e5)))
                + f'.st-key-mrf_ir_{_PASOS_MAPA[_activo][0]} button{{border:1.5px solid '
                '#7CC496;background:#1E3327}'
                f'.st-key-mrf_ir_{_PASOS_MAPA[_activo][0]} button p{{color:#ECEFF1}}'
                + '</style>')
            for _i, (_col_p, (_k, _t), (_c, _txt)) in enumerate(zip(
                    st.columns(5, gap='small'), _PASOS_MAPA, (_e1, _e2, _e3, _e4, _e5))):
                _col_p.button(f"{i18n.pick('Paso', 'Step')} {_i + 1}\n\n{_t}\n\n{_txt}",
                              key=f'mrf_ir_{_k}', on_click=_ir_paso, args=(_i,),
                              width='stretch')


# Abajo del todo en Resultados, un botón verde para seguir (la autora, 27-11-2026).
if PANTALLA == 'resultados' and 'rows' in st.session_state:
    with pant_resultados, st.container(key='siguiente_revisar'):
        st.html('<style>.st-key-siguiente_revisar{margin-top:28px}'
                '.st-key-siguiente_revisar button{min-height:54px}'
                '.st-key-siguiente_revisar button p{font-size:17px;font-weight:700}</style>')
        st.button(i18n.pick('Siguiente → 2 · Revisar', 'Next → 2 · Review'), type='primary',
                  key='boton_siguiente_revisar', on_click=ir_a, args=('revisar',),
                  width='stretch')
