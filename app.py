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

    Con Planetary Computer (o «Saltar Sentinel-2») el radar no se lee. El modelo con
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
FORMULARIO_URL = ('https://docs.google.com/forms/d/e/'
                  '1FAIpQLScq7UehnDlaNaYsq9DZNhXc-CxRo57KgdT9e98U0q1xGglRPg/viewform')
FORMULARIO_CAMPO_VERSION = 'entry.666897268'


def version_app():
    """La versión de CITATION.cff ('' si no se puede leer)."""
    import re
    try:
        texto = (APP_DIR / 'CITATION.cff').read_text(encoding='utf-8')
    except OSError:
        return ''
    m = re.search(r'^version:\s*"?([^"\s#]+)', texto, re.M)
    return m.group(1) if m else ''


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


def mostrar_sugerencias():
    """El manual y cómo sugerir un cambio o informar de un error."""
    e = enlaces_sugerencias()
    st.markdown(i18n.pick(
        f'📖 [Manual (wiki), en español e inglés]({e["wiki"]})\n\n'
        f'💬 **Sugerir un cambio o informar de un error**:\n'
        f'- [En GitHub]({e["issue"]}): público, con una cuenta de GitHub; puedes pegar '
        'pantallazos y seguir la respuesta.'
        + (f'\n- [Formulario anónimo]({e["form"]}): sin cuenta, nombre ni correo, así que '
           'no te podrán contestar. La versión ya va puesta.'
           if e['form'] else ''),
        f'📖 [Manual (wiki), in Spanish and English]({e["wiki"]})\n\n'
        f'💬 **Suggest a change or report an error**:\n'
        f'- [On GitHub]({e["issue"]}): public, with a GitHub account; you can paste '
        'screenshots and follow the answer.'
        + (f'\n- [Anonymous form]({e["form"]}): no account, no name and no email, so '
           'nobody can answer you. The version is already filled in.'
           if e['form'] else '')))
    st.caption(i18n.pick(
        'No pegues coordenadas de borreguiles: son un hábitat sensible.',
        'Do not paste the coordinates of borreguiles: they are a sensitive habitat.'))


def render_docs():
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
        'topografía + Sentinel-2 + Sentinel-1); de cada grupo de variables que dicen '
        'lo mismo usan una sola (27 en el modelo base). Las variantes añaden una '
        'fuente de **muy alta resolución** que sube el AUC. **La app extrae '
        'automáticamente la fuente extra** que el modelo elegido necesite — tú solo '
        'eliges el modelo.')
    # st.table no pasa por la traducción automática: cada idioma, su tabla
    st.table(i18n.pick({
        'Tu zona': ['🇪🇸 Andalucía · Cataluña · Canarias',
                    '🌍 Cualquier otra zona'],
        'Modelo recomendado': ['rf_sierra_nevada_cir', 'rf_sierra_nevada'],
        'Variables': ['32', '27'],
        'AUC honesto': ['0,945', '0,847'],
        'Requisitos': ['Gratis (PNOA IR 0,25 m)', 'Ninguno — universal'],
    }, {
        'Your area': ['🇪🇸 Andalusia · Catalonia · Canary Islands', '🌍 Any other area'],
        'Recommended model': ['rf_sierra_nevada_cir', 'rf_sierra_nevada'],
        'Variables': ['32', '27'],
        'Honest AUC': ['0.945', '0.847'],
        'Requirements': ['Free (PNOA IR 0.25 m)', 'None — universal'],
    }))
    st.markdown(i18n.pick(
        '- **El mejor resultado es el del modelo `_cir`, y es gratis** (AUC 0,945). '
        'Úsalo siempre que trabajes en Andalucía, Cataluña o Canarias.\n'
        '- **En el resto de España y fuera** usa el modelo base, `rf_sierra_nevada` '
        '(AUC 0,847).\n'
        '- **Sin Earth Engine** (Planetary Computer) no se lee el radar y la app usa sola '
        'la versión de cada modelo entrenada sin él: AUC 0,833 el base y 0,937 el '
        '`_cir`.\n'
        '- Al abrirse, la app trae de GitHub los modelos nuevos o actualizados; la última '
        'línea del panel lateral dice si están al día.',
        '- **The best result is that of the `_cir` model, and it is free** (AUC '
        '0.945). Use it whenever you work in Andalusia, Catalonia or the Canary '
        'Islands.\n'
        '- **Elsewhere in Spain and abroad** use the base model, `rf_sierra_nevada` '
        '(AUC 0.847).\n'
        '- **Without Earth Engine** (Planetary Computer) the radar is not read and the '
        'app automatically uses the version of each model trained without it: AUC 0.833 '
        'for the base one and 0.937 for the `_cir`.\n'
        '- When it opens, the app fetches new or updated models from GitHub; the last '
        'line of the side panel says whether they are up to date.'))

    # ---- 2. Cobertura geográfica del CIR ------------------------------------
    st.markdown('### 🗺️ Cobertura de la ortofoto infrarroja (modelo `_cir`, gratis)')
    c1, c2 = st.columns(2)
    with c1:
        st.success(
            '**✅ Con WMS infrarrojo (0,25 m)**\n\n'
            '- **Andalucía** → Sierra Nevada\n'
            '- **Cataluña** → Aigüestortes, Pirineos\n'
            '- **Canarias** → Teide, Garajonay, Caldera de Taburiente')
    with c2:
        st.warning(i18n.pick(
            '**❌ Sin WMS infrarrojo** (usa el modelo base)\n\n'
            '- **Aragón** → Ordesa y Monte Perdido\n'
            '- **Cantabria / Castilla y León** → Picos de Europa\n'
            '- **Madrid / CyL** → Sierra de Guadarrama',
            '**❌ Without infrared WMS** (use the base model)\n\n'
            '- **Aragón** → Ordesa y Monte Perdido\n'
            '- **Cantabria / Castile-León** → Picos de Europa\n'
            '- **Madrid / Castile-León** → Sierra de Guadarrama'))
    st.caption(
        'El IGN tiene PNOA Falso Color Infrarrojo a 0,25 m para **toda España**, pero solo '
        'como descarga COG por hojas MTN25 (sin WMS) → conectarlo cubriría también Ordesa, '
        'Picos y Guadarrama. El modelo `_cir` se calibró con Andalucía; en otras CCAA las '
        'variables IR son comparables pero no idénticas → para máxima precisión, entrena '
        'con verdad-terreno local.')

    # ---- 4. Fuentes de datos y variables ------------------------------------
    st.markdown('### 🛰️ Fuentes de datos y variables del modelo')
    st.table(i18n.pick({
        'Fuente': ['Imagen aérea (ESRI / PNOA)', 'Copernicus DEM 30 m',
                   'Sentinel-2 L2A (2017-2025)', 'Sentinel-1, radar (2019-2024)',
                   'PNOA Falso Color IR 0,25 m'],
        'Variables': ['8', '3', '11', '5', '5 (solo `_cir`)'],
        'Qué aporta': [
            'Textura, color, patrón y contexto alpino (% roca)',
            'Altitud, pendiente y curvatura',
            'NDVI/NDWI/NDMI… de inicio y fin de verano y estadísticos de verano',
            'Humedad y estructura del suelo por radar, con el relieve corregido',
            'Pseudo-NDVI a 25 cm: nitidez del borde de la mancha'],
    }, {
        'Source': ['Aerial image (ESRI / PNOA)', 'Copernicus DEM 30 m',
                   'Sentinel-2 L2A (2017-2025)', 'Sentinel-1, radar (2019-2024)',
                   'PNOA False Colour IR 0.25 m'],
        'Variables': ['8', '3', '11', '5', '5 (`_cir` only)'],
        'What it adds': [
            'Texture, colour, pattern and alpine context (% rock)',
            'Elevation, slope and curvature',
            'NDVI/NDWI/NDMI… of early and late summer and summer statistics',
            'Soil moisture and structure from radar, with the relief corrected',
            'Pseudo-NDVI at 25 cm: sharpness of the patch edge'],
    }))
    st.caption(i18n.pick(
        'Las variables que usa el modelo: 27 en el base y 32 en el `_cir` (una por grupo '
        'de las que dicen lo mismo). La última fila es la que añade el `_cir`: su variable '
        'más informativa es el **contraste del NDVI a 25 cm**, justo el borde nítido del '
        'borreguil contra el sustrato pétreo, que Sentinel-2 (10 m) promedia y pierde.',
        'The variables the model uses: 27 in the base one and 32 in the `_cir` one (one '
        'per group of those that say the same). The last row is what the `_cir` adds: its '
        'most informative variable is the **NDVI contrast at 25 cm**, precisely the sharp '
        'edge of the borreguil against the rocky substrate, which Sentinel-2 (10 m) '
        'averages out and loses.'))

    # ---- 4b. Importancia de variables y lectura jerárquica ------------------
    st.markdown('### 🔑 Qué variables pesan más y cómo leer las gráficas')
    st.markdown(i18n.pick(
        'El modelo es un **Random Forest**: cada variable recibe una **importancia** '
        '(0–100 %) según cuánto ayuda a separar *borreguil* de *no-borreguil*. Muchas '
        'variables son casi iguales (por ejemplo, 13 índices de verdor de Sentinel-2 con '
        'correlación de 0,89 o más), así que el modelo usa **una sola de cada grupo**: '
        'si usara todas, se repartiría el peso entre ellas y cada una parecería poco '
        'importante. El acierto es el mismo.\n\n'
        'Al seleccionar un punto (pestaña **Revisar puntos**), el panel **📈 Este punto frente a '
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
        'When you select a point (**Review points** tab), the panel **📈 This point vs the real '
        'borreguiles** shows the variables that weigh most, **one per group** of similar '
        'ones, and for each one where the point falls among the borreguiles the model was '
        'trained with: the green band holds the middle half of them.'))
    _mp = st.session_state.get('start_model_path') or str(model_path('rf_sierra_nevada_cir.joblib'))
    _rs = load_ref_stats(_mp)
    _ic, _ = pp.chart_importance(_rs, topn=15) if _rs else (None, [])
    if _ic is not None:
        st.altair_chart(_ic, width='stretch')
        st.caption(i18n.pick(
            f'Ranking del modelo seleccionado (`{Path(_mp).name}`), variable a variable. '
            'Cambia según el modelo: en los `_cir` manda el **contraste del NDVI a 0,25 m**; '
            'en el modelo base, la **altitud**, la **caída estacional del NDVI**, la '
            '**diferencia VV–VH del radar**, el **NDVI de inicio de verano** y la '
            '**variación del NDWI**. Cada modelo usa una '
            'sola variable de cada grupo de las que dicen lo mismo, así que el peso no se '
            'reparte entre variables correlacionadas.',
            f'Ranking of the selected model (`{Path(_mp).name}`), variable by variable. It '
            'changes with the model: in the `_cir` ones the **NDVI contrast at 0.25 m** '
            'leads; in the base model, **elevation**, the **seasonal NDVI drop**, the '
            '**radar VV–VH difference**, **early-summer NDVI** and the **NDWI '
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
        'El mapa de borreguil sí o no (**🌿 Mapa Modelo RF**) aprende de tus puntos '
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
        'probaron otras proporciones. En **🌿 Mapa Modelo RF → 1 · Datos** la app te dice '
        'dónde estás en esta curva y cuántos borreguiles y ausencias te faltan.\n\n'
        'En tu zona, lo que vale es la curva con **tus** puntos: en **🌿 Mapa Modelo RF → '
        '3 · Modelo**, después de entrenar, **📈 Calcular la curva con mis puntos** repite '
        'la validación del modelo aprendiendo con el 25, el 50 y el 75 % de tus puntos. Si '
        'de la mitad a todos (el doble) el acierto todavía sube, más puntos mejorarán el '
        'mapa; si ya casi no sube, más puntos parecidos apenas lo cambiarán.',
        'The borreguil yes-or-no map (**🌿 RF model map**) learns from your reviewed '
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
        'were not tested. Under **🌿 RF model map → 1 · Data** the app tells you where you '
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
        'Con el VIF» (umbral 10), Random Forest y validación por zonas.',
        'Line: mean of 10 repetitions; band: 8 out of 10. Dashed line: with all the '
        'points. Same procedure as the app with "Remove correlated variables: With the '
        'VIF" (threshold 10), Random Forest and spatial validation.'))

    # ---- 5. Cómo usar la app ------------------------------------------------
    st.markdown('### ▶️ Flujo de trabajo')
    st.markdown(i18n.pick(
        '1. **Tus puntos**: súbelos o deja que la app los genere al azar en el área '
        '(vector, WDPA ID o nombre).\n'
        '2. Elige la **fuente de datos** (Earth Engine: más rápido y con radar; Planetary '
        'Computer: sin cuenta y sin radar), el **modelo** según la tabla de arriba y la '
        '**fuente de imágenes** (PNOA en España, ESRI fuera).\n'
        '3. Pulsa **Ejecutar pipeline**. La app descarga cada fuente, extrae las '
        'variables y predice con el **modelo de partida**. Si tus puntos traen '
        'etiquetas de campo, te dice cuántas acierta.\n'
        '4. En **📍 Verdad terreno → Revisar puntos**, verifica puntos como borreguil o no '
        '(con quién revisa, su perfil y la certidumbre) y revisa su tipo; en **Mapa de '
        'los puntos** puedes añadir puntos pinchando. Descarga Excel, CSV o '
        '`puntos.geojson` en **Descargas**.\n'
        '5. *(Opcional)* si tienes puntos de campo de tu zona, marca **«Sumar mis puntos a los del modelo y reentrenarlo»**: '
        'la app vuelve a entrenar el modelo con sus 769 puntos de Sierra Nevada y los '
        'tuyos juntos, y te dice cuánto acierta en tus puntos.\n'
        f'6. En **🌿 Mapa Modelo RF**, con tus puntos revisados (mínimo unos {_mn}, con '
        f'{_mb} borreguiles), entrena y valida un modelo de tu zona y haz el mapa de '
        'borreguil a 10 m. Necesita Earth Engine.',
        '1. **Your points**: upload them or let the app generate them at random in the '
        'area (vector, WDPA ID or name).\n'
        '2. Choose the **data source** (Earth Engine: faster and with radar; Planetary '
        'Computer: no account and no radar), the **model** according to the table above '
        'and the **image source** (PNOA in Spain, ESRI elsewhere).\n'
        '3. Click **Run pipeline**. The app downloads each source, extracts the variables '
        'and predicts with the **starting model**. If your points carry field labels, '
        'it tells you how many it gets right.\n'
        '4. Under **📍 Ground truth → Review points**, verify points as borreguil or not '
        '(with who is reviewing, their profile and the certainty) and review their type; '
        'under **Points map** you can add points by clicking. Download Excel, CSV or '
        '`puntos.geojson` under **Downloads**.\n'
        '5. *(Optional)* if you have field points from your area, tick **"Add my points to the model\'s and retrain it"**: '
        'the app retrains the model with its 769 Sierra Nevada points and yours '
        'together, and tells you how well it does on your points.\n'
        f'6. Under **🌿 RF model map**, with your reviewed points (at least about {_mn}, '
        f'with {_mb} borreguiles), train and validate a model for your area and make the '
        '10 m borreguil map. It needs Earth Engine.'))

    # ---- 6. Manual y sugerencias --------------------------------------------
    st.markdown(i18n.pick('### 💬 Manual y sugerencias', '### 💬 Manual and suggestions'))
    mostrar_sugerencias()


# ---- Qué modelo ha dado la probabilidad y cuánto acierta con los puntos de campo ----
def _zona_partida(info):
    z = (info or {}).get('training_zone')
    if not z or z == 'usuario':
        z = (info or {}).get('zona_partida') or '?'
    return str(z).split(' (')[0]


def _titulo_modelo(info):
    """Una línea: qué modelo ha dado la probabilidad de los puntos."""
    if not info:
        return i18n.pick('No se ha podido ejecutar el modelo.', 'The model could not be run.')
    # Sin radar (Planetary Computer): se usa la versión del modelo entrenada sin él
    sr = _es_sin_radar(info)
    if info.get('sumado'):
        z = _zona_partida(info)
        return i18n.pick(
            f'**Modelo usado: el de {z}' + (' (versión sin radar)' if sr else '')
            + ' reentrenado sumando tus puntos**',
            f'**Model used: the {z} model' + (' (version without radar)' if sr else '')
            + ' retrained adding your points**')
    if info.get('mode') == 'trained':
        return i18n.pick('**Modelo usado: el entrenado con tus puntos**',
                         '**Model used: the one trained with your points**')
    z = _zona_partida(info)
    return i18n.pick(
        f'**Modelo usado: el de partida ({z}' + (', versión sin radar' if sr else '') + ')**'
        + (', porque el radar no se ha leído' if sr else ''),
        f'**Model used: the starting model ({z}' + (', version without radar' if sr else '')
        + ')**' + (', because the radar was not read' if sr else ''))


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
    if sumado:
        _n_t, _n_s = info.get('n_tabla', 0), info.get('n_pos', 0) + info.get('n_neg', 0)
        _q = info.get('n_tabla_quitados', 0)
        st.success(titulo + '. ' + pick(
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
            'pestaña Revisar puntos o añádelos pinchando en el mapa.',
            'You ticked "Add my points to the model\'s", but none of your points says '
            'whether it is a borreguil: the model is used as it is. Verify points in the '
            'Review points tab or add them by clicking on the map.'))
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
            # botón para hacerlo va justo debajo de este recuadro.
            st.info(titulo + '. ' + pick(
                'Has marcado «Sumar mis puntos a los del modelo y reentrenarlo»: pulsa el '
                'botón de abajo para reentrenarlo ahora.',
                'You ticked "Add my points to the model\'s and retrain it": press the '
                'button below to retrain it now.'))
        elif casilla == 'sin_puntos':
            st.info(titulo + '. ' + pick(
                'Has marcado «Sumar mis puntos a los del modelo y reentrenarlo», pero '
                'ninguno de tus puntos dice todavía si es borreguil o no. Verifica puntos '
                'en la pestaña Revisar puntos o añádelos pinchando en el mapa.',
                'You ticked "Add my points to the model\'s and retrain it", but none of '
                'your points says yet whether it is a borreguil. Verify points in the '
                'Review points tab or add them by clicking on the map.'))
        else:
            st.info(titulo + '. ' + pick(
                'Es lo que hace la app por defecto. Para que tus puntos con etiqueta se '
                'sumen a los del modelo y reentrenarlo, marca «Sumar mis puntos a los del '
                'modelo y reentrenarlo» en el panel.',
                'This is what the app does by default. To add your labelled points to the '
                'model\'s and retrain it, tick "Add my points to the model\'s and retrain '
                'it" in the panel.'))
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
        nombres = {'partida': pick(f'Modelo de partida ({z})', f'Starting model ({z})'),
                   'propio': pick('Modelo con tus puntos (validación por zonas)',
                                  'Model with your points (spatial validation)')}
        if sumado:
            nombres = {'partida': pick(f'Modelo de {z}, sin tus puntos',
                                       f'{z} model, without your points'),
                       'propio': pick('Sumando tus puntos (validación por zonas)',
                                      'Adding your points (spatial validation)')}
        tabla = ('| | ' + pick('Borreguiles de campo acertados', 'Field borreguiles right')
                 + ' | ' + pick('Ausencias de campo acertadas', 'Field absences right')
                 + ' | AUC |\n|---|---|---|---|\n')
        for k in ('partida', 'propio'):
            if k in ac:
                a = ac[k]
                marca = ' ✓' if (k == usado and len(ac) > 1) else ''
                tabla += (f'| {nombres[k]}{marca} | {_de(a["ok_si"], a["n_si"])} | '
                          f'{_de(a["ok_no"], a["n_no"])} | {_auc_txt(a["auc"])} |\n')
        st.markdown(tabla)
        _umbral_es = f'{threshold:.3f}'.replace('.', ',')
        st.caption(pick(
            f'Acertado: un borreguil de campo que el modelo da por probable o posible '
            f'(umbral {_umbral_es}); una ausencia que no da por borreguil. AUC: 1 = '
            'separa perfectamente borreguil de no borreguil; 0,5 = como tirar una moneda '
            f'(se da con {bp.MIN_COMPARAR} borreguiles y {bp.MIN_COMPARAR} ausencias o '
            'más). Si estos puntos se usaron para entrenar el modelo de partida, su '
            'acierto sale más alto de lo real.',
            f'Right: a field borreguil that the model calls probable or possible '
            f'(threshold {_num(threshold)}); an absence that it does not call borreguil. '
            'AUC: 1 = separates borreguil from non-borreguil perfectly; 0.5 = like '
            f'tossing a coin (given with {bp.MIN_COMPARAR} borreguiles and '
            f'{bp.MIN_COMPARAR} absences or more). If these points were used to train the '
            'starting model, its accuracy comes out higher than it really is.'))
        # El acierto según la certidumbre y el perfil de quien dio cada etiqueta
        # (24-11): si el modelo falla sobre todo donde la revisión era dudosa, el fallo
        # puede ser de la etiqueta y no del modelo.
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
            with st.expander(pick('Acierto según la certidumbre y el perfil de quien revisa',
                                  'Accuracy by the certainty and profile of the reviewer')):
                for clave, grupos in _rev.items():
                    tb = (f'| {_cab[clave]} | '
                          + pick('Borreguiles acertados', 'Borreguiles right') + ' | '
                          + pick('Ausencias acertadas', 'Absences right')
                          + ' |\n|---|---|---|\n')
                    for v, a_ in grupos.items():
                        tb += (f'| {_txt_val[clave][v]} | {_de(a_["ok_si"], a_["n_si"])} | '
                               f'{_de(a_["ok_no"], a_["n_no"])} |\n')
                    st.markdown(tb)
                st.caption(pick(
                    'Del modelo que se está usando. Si falla sobre todo en los puntos de '
                    'certidumbre baja, puede que el error sea de la etiqueta y no del modelo.',
                    'For the model in use. If it fails mostly on low-certainty points, the '
                    'error may be in the label rather than in the model.'))
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
            'acierta en tu zona: revisa los puntos que propone (pestaña Revisar puntos).',
            'Your points carry no field labels, so how well it does in your area cannot '
            'be measured: review the points it proposes (Review points tab).'))

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
    tarda unos segundos, radar incluido; con Planetary Computer el radar no se lee (si
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
              ('🛰️ Sentinel-2 (NDVI, Clre…)', s2_feats, 's2', skip_s2),
              ('📡 Sentinel-1 (radar)', s1_feats, 's1', skip_s2 or skip_s1)]
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
            'Planetary Computer (MPC)** en el panel lateral.\n'
            '- Caída temporal del servicio satelital → **reintenta** en unos minutos.\n'
            '- Años sin escenas con poca nube → amplía el **rango de años**.')
    elif problems:
        st.caption('Sugerencia: reintenta (caídas temporales de los servicios), reduce '
                   'el nº de puntos, o cambia de backend (MPC ↔ GEE) en el panel lateral.')


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
    return _etiquetas_de_partes(tuple((f.name, f.getvalue()) for f in ficheros), campo)


@st.cache_data(show_spinner=False)
def _etiquetas_de_partes(partes, campo=False):
    ruta = None
    try:
        ruta = _guardar_subida([_Subido(n, d) for n, d in partes])
        if campo:
            return _resumen_campo(bp.leer_puntos_de_campo(bp.read_points(ruta)))
        return bp.resumen_etiquetas(bp.leer_etiquetas(bp.read_points(ruta)))
    except ValueError as e:                  # falta algo, o el fichero no se entiende
        return [('falta', str(e))]
    except Exception:
        return None
    finally:
        if ruta:
            shutil.rmtree(os.path.dirname(ruta), ignore_errors=True)


@st.cache_data(show_spinner=False)
def _leer_capa_cuartiles(datos, nombre):
    """La capa de polígonos para lanzar puntos por cuartiles (random_points.leer_capa),
    leída una sola vez por fichero. Devuelve (capa, error)."""
    import random_points
    carpeta = Path(tempfile.mkdtemp(prefix='capa_cuartiles_'))
    try:
        ruta = carpeta / Path(nombre).name
        ruta.write_bytes(datos)
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
    """Dónde cae el punto entre los borreguiles verdaderos en las variables que más
    pesan en el modelo (point_profile.chart_punto_vs_borreguiles)."""
    if not ref_stats or not ref_stats.get('features'):
        st.info(i18n.pick(
            'El modelo elegido no trae los valores de los borreguiles verdaderos, así que '
            'no se puede comparar el punto con ellos. Estos son sus valores en las '
            'variables que más separan en Sierra Nevada.',
            'The chosen model does not carry the values of the real borreguiles, so the '
            'point cannot be compared with them. These are its values for the variables '
            'that separate best in Sierra Nevada.'))
        vals = {pp.etiqueta(k['feat']): _num(sel.get(k['feat']))
                for k in pp.variables_clave(None)}
        st.dataframe(pd.DataFrame([vals]), width='stretch', hide_index=True)
        return
    ch, recs = pp.chart_punto_vs_borreguiles(sel, ref_stats, n=8)
    if ch is None:
        st.caption(i18n.pick('Sin variables con valores de referencia para este punto.',
                             'No variables with reference values for this point.'))
        return
    nsi = ref_stats.get('n_si', '?')
    st.caption(i18n.pick(
        f'Cada punto de color es **el punto que estás mirando**, en una variable: dónde '
        f'cae entre los **{nsi} borreguiles verdaderos** de los que aprendió el modelo, en '
        'las variables que más pesan (de arriba abajo). 0 = más bajo que todos, 50 = como '
        'el borreguil típico, 100 = más alto que todos. Debajo, los valores.',
        f'Each coloured dot is **the point you are looking at**, in one variable: where it '
        f'falls among the **{nsi} real borreguiles** the model learnt from, for the '
        'variables that weigh most (top to bottom). 0 = lower than all, 50 = like the '
        'typical borreguil, 100 = higher than all. Below, the values.'))
    st.markdown(pp.leyenda_punto_vs_borreguiles(), unsafe_allow_html=True)
    st.altair_chart(ch, width='stretch')
    st.dataframe(pp.tabla_punto_vs_borreguiles(recs), hide_index=True, width='stretch')


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
        return self._destino().write(s)

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
                    initial_sidebar_state='expanded')

# --- Idioma / Language ---------------------------------------------------
# Selector bilingüe (se dibuja ANTES de instalar la traducción, así su propia
# etiqueta no depende del idioma). Fija el idioma de todo el resto de la app.
_lang_choice = st.sidebar.selectbox(
    'Idioma / Language', ['Español', 'English'],
    index=0, key='ui_lang')
i18n.set_lang('en' if _lang_choice == 'English' else 'es')
i18n.install(st)   # traduce todos los st.* a partir de aquí (fallback a ES)

# --- Modelos nuevos de GitHub ------------------------------------------------
# Una vez por arranque (antes del desplegable de modelos, para que ya salgan en él).
_modelos_gh = modelos_de_github()
if _modelos_gh.get('estado') == 'descargados' and not st.session_state.get('_aviso_gh'):
    st.session_state['_aviso_gh'] = True
    st.toast(i18n.pick('⬇️ Modelos nuevos de GitHub: ', '⬇️ New models from GitHub: ')
             + ', '.join(_modelos_gh['nuevos'] + _modelos_gh['actualizados'])
             + i18n.pick('. Ya están en «Modelo de partida».',
                         '. They are already under "Starting model".'))

st.title('🌿 Borreguil · Wet Meadow Finder')
st.markdown(i18n.pick(
    'Identificación automática de borreguiles en zonas de montaña con imagen aérea, '
    'relieve (Copernicus DEM), Sentinel-2, radar Sentinel-1 y Random Forest. '
    f'[Manual (wiki)]({WIKI_URL})',
    'Automatic detection of borreguiles (high-mountain wet meadows) with aerial '
    'imagery, terrain (Copernicus DEM), Sentinel-2, Sentinel-1 radar and Random '
    f'Forest. [Manual (wiki)]({WIKI_URL})'))

# ============================================================
# SIDEBAR – inputs y opciones
# ============================================================
with st.sidebar:
    st.header(i18n.pick('Datos de entrada', 'Inputs'))

    input_mode = st.radio('Origen de los puntos', [
        '📁  Subir archivo (KML/Shp/GeoJSON)',
        '🎲  Generar aleatoriamente en el área',
    ], index=0, label_visibility='collapsed')

    f_input = None; gen_n = 0; gen_mode = 'stratified_elev'
    gen_min_elev = None; gen_max_elev = None; gen_seed = 42; gen_min_dist = 50
    gen_lejos = 0
    gen_capa = None; gen_frac_q1 = 0.75
    if input_mode.startswith('📁'):
        # Varios ficheros a la vez: un Shapefile son varios (.shp, .dbf, .shx, .prj)
        f_input = st.file_uploader(
            i18n.pick('Tus puntos', 'Your points'), type=FORMATOS_PUNTOS,
            accept_multiple_files=True, key='subir_puntos',
            help=i18n.pick(
                'KML, GeoJSON o CSV (con columnas `lon` y `lat`). Si es un Shapefile, '
                'selecciona a la vez todos sus ficheros (.shp, .dbf, .shx y .prj) o '
                'súbelo en un .zip.',
                'KML, GeoJSON or CSV (with `lon` and `lat` columns). If it is a '
                'Shapefile, select all its files at once (.shp, .dbf, .shx and .prj) or '
                'upload it as a .zip.')) or None
        st.caption(i18n.pick(
            'Los sitios en los que la app mira si hay borreguil. Mira los requisitos en '
            'la plantilla de la pantalla de inicio.',
            'The places where the app checks whether there is borreguil. See the '
            'requirements in the template on the start screen.'))
        if f_input is not None:
            for _niv_et, _txt_et in (_etiquetas_de_subida(f_input) or []):
                {'ok': st.caption, 'info': st.info, 'aviso': st.warning,
                 'falta': st.warning}[_niv_et](
                    ('✓ ' if _niv_et == 'ok' else '') + _txt_et)
    else:
        st.caption('Requiere definir el área de estudio (panel siguiente).')
        gen_n = st.slider('Nº puntos a generar', 30, 1000, 200, 10)
        # Los valores internos no cambian (van a random_points); solo cómo se leen
        _nombre_estrategia = {
            'stratified_elev': i18n.pick('Repartidos por altitud', 'Spread across elevation'),
            'uniform': i18n.pick('Al azar', 'At random'),
            'poisson_disc': i18n.pick('Al azar, separados una distancia mínima',
                                      'At random, a minimum distance apart'),
            'cuartiles': i18n.pick('Por cuartiles de una capa', 'By quartiles of a layer'),
        }
        gen_mode = st.selectbox(
            i18n.pick('Estrategia', 'Strategy'), list(_nombre_estrategia), index=0,
            format_func=_nombre_estrategia.get,
            help=i18n.pick(
                '**Repartidos por altitud**: el rango de altitud del área se divide en 5 '
                'franjas iguales y se toma el mismo número de puntos en cada una, para '
                'cubrir todo el gradiente. **Al azar**: sin ninguna regla. **Al azar, '
                'separados una distancia mínima**: ningún par de puntos queda más cerca '
                'de la distancia que elijas, para que no salgan puntos pegados. '
                '**Por cuartiles de una capa**: subes una capa de polígonos (por ejemplo, '
                'las áreas probables de borreguil) y la mayoría de los puntos caen en los '
                'polígonos del cuartil 1 y el resto en los demás.',
                '**Spread across elevation**: the elevation range of the area is split '
                'into 5 equal bands and the same number of points is taken in each, to '
                'cover the whole gradient. **At random**: no rule. **At random, a '
                'minimum distance apart**: no pair of points is closer than the distance '
                'you choose, so that no points end up side by side. **By quartiles of a '
                'layer**: you upload a polygon layer (for instance, the probable borreguil '
                'areas) and most points fall in the polygons of quartile 1 and the rest in '
                'the others.'))
        if gen_mode == 'cuartiles':
            # 26-11-2026, la autora: «quiero lanzar por cuartiles de una capa (que la
            # subirán los usuarios): el 75 % de los puntos en el primer cuartil y el 25 %
            # en el resto». Los puntos van sin etiqueta: al revisarlos salen presencias
            # y ausencias. El reparto lo hace random_points.por_cuartiles.
            import random_points
            _f_capa = st.file_uploader(
                i18n.pick('Capa de polígonos', 'Polygon layer'),
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
                        i18n.pick('% de los puntos en el cuartil 1',
                                  '% of the points in quartile 1'),
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
        col1, col2 = st.columns(2)
        with col1:
            # Por cuartiles, vacía de entrada: la capa ya dice dónde, y en parques más
            # bajos que 2000 m no saldría ningún punto.
            gen_min_elev_str = st.text_input(
                'Elev. mín (m)', value='' if gen_mode == 'cuartiles' else '2000',
                help='Opcional: filtra por SRTM 30m')
        with col2:
            gen_max_elev_str = st.text_input('Elev. máx (m)', value='',
                                                help='Opcional')
        gen_min_elev = float(gen_min_elev_str) if gen_min_elev_str.strip() else None
        gen_max_elev = float(gen_max_elev_str) if gen_max_elev_str.strip() else None
        gen_seed = st.number_input('Semilla', 0, 99999, 42)
        gen_lejos = st.number_input(
            'Lejos de tus puntos de campo (m)', min_value=0, max_value=5000, value=0,
            step=50,
            help='Si subes tu verdad-terreno (debajo), los puntos al azar quedan al '
                 'menos a esta distancia de tus puntos de campo: caen en sitios que '
                 'todavía no has muestreado, para buscar allí ausencias (o algún '
                 'borreguil nuevo) que revisas tú en la tabla. 0 = sin límite.\n\n'
                 '**Recomendado: 0** para probar el modelo por toda el área; **300 m** '
                 'para buscar ausencias fuera de tus puntos.')

    f_truth = st.file_uploader(
        i18n.pick('Opcional: puntos de campo para comprobar el modelo',
                  'Optional: field points to check the model'),
        type=FORMATOS_PUNTOS, accept_multiple_files=True, key='subir_campo',
        help=i18n.pick(
            'KML, GeoJSON o CSV (con columnas `lon` y `lat`). Si es un Shapefile, '
            'selecciona a la vez todos sus ficheros (.shp, .dbf, .shx y .prj) o súbelo '
            'en un .zip.',
            'KML, GeoJSON or CSV (with `lon` and `lat` columns). If it is a Shapefile, '
            'select all its files at once (.shp, .dbf, .shx and .prj) or upload it as '
            'a .zip.')) or None
    st.caption(i18n.pick(
        'Sitios donde ya sabes si hay borreguil, **en un fichero aparte**. Sirven para '
        'medir cuánto acierta el modelo y, si quieres, para reentrenarlo. Si esa '
        'información ya va en «Tus puntos» (columna `presence`: 1 = borreguil, 0 = no), '
        'no subas nada aquí.',
        'Places where you already know whether there is borreguil, **in a separate '
        'file**. They are used to measure how well the model does and, if you want, to '
        'retrain it. If that information is already in "Your points" (`presence` '
        'column: 1 = borreguil, 0 = no), do not upload anything here.'))
    if f_truth is not None:
        # Qué columna ha leído y cuántos sí y no (24-11), o lo que falta (un .shp sin .dbf…)
        for _niv_et, _txt_et in (_etiquetas_de_subida(f_truth, campo=True) or []):
            {'ok': st.caption, 'info': st.info, 'aviso': st.warning,
             'falta': st.warning}[_niv_et](('✓ ' if _niv_et == 'ok' else '') + _txt_et)
    truth_train_only = st.checkbox(
        i18n.pick('Usar los puntos de campo solo para medir y entrenar (no mostrarlos en '
                  'el mapa)',
                  'Use the field points only to measure and train (do not show them on '
                  'the map)'),
        value=False, disabled=(f_truth is None),
        help=i18n.pick(
            'Tus puntos de campo sirven para medir cuánto acierta el modelo (y, si marcas '
            '«Sumar mis puntos a los del modelo y reentrenarlo», para entrenarlo), pero NO se '
            'añaden al mapa, la tabla ni las descargas: solo verás tus puntos. '
            'Útil cuando subes muchos borreguiles verificados y no quieres que saturen el '
            'resultado. Los candidatos que coincidan (<20 m) con un punto de campo sí se '
            'marcan como verificados y se muestran.',
            'Your field points are used to measure how well the model does (and, if you '
            'tick "Add my points to the model\'s and retrain it", to train it), but they are NOT '
            'added to the map, table or downloads: you will only see your candidate '
            'points. Useful when you upload many verified borreguiles and do not want '
            'them to swamp the result. Candidates that match (<20 m) a field point are '
            'marked as verified and shown.'))
    # Reescrito con la autora el 24-11: primero la plantilla; los puntos sin columnas,
    # al final y con sus límites.
    with st.expander(i18n.pick('ℹ ¿Cómo preparar tus puntos de campo?',
                               'ℹ How to prepare your field points?')):
        st.markdown(i18n.pick(
            'Usa la **plantilla de datos de campo** (se descarga en la pantalla de '
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
            '- CSV con columnas `lon` y `lat` (grados, WGS84), o GeoJSON, Shapefile o KML '
            'con esos nombres.\n\n'
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
            'Use the **field data template** (downloadable on the start screen). One row '
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
            '- CSV with `lon` and `lat` columns (degrees, WGS84), or GeoJSON, Shapefile or '
            'KML with those names.\n\n'
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
            'no absences.'))

    st.divider()
    st.header('Área de estudio')
    st.caption('Define el alcance espacial. Si no se aporta, se usa la bbox de los puntos.')

    area_mode = st.radio('Modo', [
        '🗺  Vector (subir KML/Shp/GeoJSON)',
        '🆔  WDPA ID (protectedplanet.net)',
        '🔤  Nombre del área protegida',
        '⨯  Sin área (bbox de los puntos)'
    ], index=3, label_visibility='collapsed')

    f_area = None; wdpa_id = ''; pa_name = ''; wdpa_token = ''
    if area_mode.startswith('🗺'):
        f_area = st.file_uploader('Polígono del área de estudio',
                                     type=['kml','geojson','json','gpkg',
                                           'shp','shx','dbf','prj','cpg','zip'],
                                     accept_multiple_files=True, key='area_file')
        st.caption('Un único **KML / GeoJSON / GeoPackage**, o un **shapefile** '
                   '(selecciona .shp + .shx + .dbf + .prj juntos, o sube un .zip).')
    elif area_mode.startswith('🆔'):
        wdpa_id = st.text_input('WDPA ID', placeholder='555512151',
                                  help='ID numérico de protectedplanet.net.')
        wdpa_token = st.text_input('Token Protected Planet (opcional)', type='password',
                                  help=('Gratuito en protectedplanet.net/api. Necesario '
                                        'para IDs fuera del cache, salvo que uses el '
                                        'backend Google Earth Engine (resuelve sin token).'))
        st.caption('Ejemplos: `555512151` Sierra Nevada · `4514` Picos de Europa · '
                    '`11` Yellowstone. IDs arbitrarios requieren token o backend GEE.')
    elif area_mode.startswith('🔤'):
        pa_name = st.text_input('Nombre del área',
                                  placeholder='Parque Nacional de Sierra Nevada',
                                  help=('Búsqueda en Nominatim (OSM). Cuanto más '
                                        'específico, mejor (incluir "Parque Nacional/'
                                        'Natural" suele ayudar).'))

    st.divider()
    st.header('Fuente de datos satelitales')
    backend_label = st.radio('Backend', [
        '🌍  Google Earth Engine — más rápido (cuenta compartida o la tuya)',
        '🛰  Microsoft Planetary Computer (sin cuenta)',
    ], index=0, label_visibility='collapsed')
    backend = 'gee' if backend_label.startswith('🌍') else 'mpc'
    st.caption(i18n.pick(
        'Earth Engine calcula las variables en sus servidores, en segundos; de entrada, '
        'con la cuenta compartida de MonitorEO, sin cuenta propia. Planetary Computer es '
        'más lento y no lee el radar (Sentinel-1): con él, la app usa una versión del '
        'modelo entrenada sin radar, que acierta algo menos. Si Earth Engine no está '
        'disponible, la app usa Planetary Computer.',
        'Earth Engine computes the variables on its servers, in seconds; by default with '
        'the MonitorEO shared account, with no account of your own. Planetary Computer '
        'is slower and does not read the radar (Sentinel-1): with it, the app uses a '
        'version of the model trained without radar, which is somewhat less accurate. If '
        'Earth Engine is not available, the app uses Planetary Computer.'))

    gee_project = ''
    if backend == 'gee':
        # Vacío: la cuenta compartida de MonitorEO, como en sus herramientas (26-11-2026,
        # la autora: «que use la cuenta de servicio de monitoreo, que esté por defecto y
        # que el usuario pueda elegir su propia cuenta»). Con un proyecto, la cuenta de
        # Google de quien usa la app. Arranca con el último proyecto con el que la
        # conexión funcionó (vacío si se eligió la compartida).
        gee_project = st.text_input(
            i18n.pick('Proyecto de Google Cloud (vacío: cuenta compartida)',
                      'Google Cloud project (empty: shared account)'),
            value=leer_ajuste('gee_project', ''),
            placeholder=i18n.pick('vacío: cuenta compartida de MonitorEO · o tu proyecto',
                                  'empty: MonitorEO shared account · or your project'),
            help=i18n.pick(
                '**Vacío**: la app calcula en Earth Engine con la **cuenta compartida de '
                'MonitorEO**, sin iniciar sesión ni tener cuenta propia. Cada ordenador '
                'tiene unas horas de cálculo de Earth Engine al mes, para uso no '
                'comercial; quien las gasta puede usar su propia cuenta hasta el mes '
                'siguiente.\n\n'
                '**Con tu proyecto de Google Cloud** registrado en Earth Engine (por '
                'ejemplo `ee-tunombre`), calcula con tu cuenta de Google y gasta la cuota '
                'de ese proyecto. Se recuerda para la próxima vez. Para registrar uno: '
                'https://code.earthengine.google.com/register.\n\n'
                '**🔑 Autenticar GEE** abre el navegador para entrar con tu cuenta de '
                'Google: solo hace falta con tu proyecto, y una vez en cada ordenador. '
                '**✓ Probar conexión** comprueba que funciona; no es obligatorio, porque '
                'la app se conecta sola al calcular.',
                '**Empty**: the app computes in Earth Engine with the **MonitorEO shared '
                'account**, with no sign-in and no account of your own. Each computer has '
                'some Earth Engine compute hours a month, for non-commercial use; whoever '
                'spends them can use their own account until the next month.\n\n'
                '**With your Google Cloud project** registered in Earth Engine (for '
                'instance `ee-yourname`), it computes with your Google account and uses '
                'the quota of that project. It is remembered for the next time. To '
                'register one: https://code.earthengine.google.com/register.\n\n'
                '**🔑 Authenticate GEE** opens the browser to sign in with your Google '
                'account: it is only needed with your project, and once on each computer. '
                '**✓ Test connection** checks that it works; it is not required, because '
                'the app connects by itself when it computes.'))
        if not gee_project.strip() and leer_ajuste('gee_project', ''):
            guardar_ajuste('gee_project', '')        # ha vuelto a la cuenta compartida
        if gee_project.strip():
            st.caption(i18n.pick(
                'Con **tu cuenta de Google** y la cuota de ese proyecto. La primera vez en '
                'este ordenador, pulsa **🔑 Autenticar GEE**.',
                'With **your Google account** and the quota of that project. The first '
                'time on this computer, press **🔑 Authenticate GEE**.'))
        else:
            _uso = _uso_cuenta_compartida()
            _txt_uso = i18n.pick(
                'Con la **cuenta compartida de MonitorEO**: sin iniciar sesión ni cuenta '
                'propia.',
                'With the **MonitorEO shared account**: no sign-in and no account of your '
                'own.')
            if _uso and _uso.get('left_hours') is not None:
                _txt_uso += ' ' + i18n.pick(
                    f"Te quedan **{_num(_uso['left_hours'])} h** de cálculo este mes "
                    f"(de {_num(_uso.get('limit_hours'))}; se renuevan el "
                    f"{_uso.get('resets', 'día 1')}).",
                    f"You have **{_num(_uso['left_hours'])} h** of compute left this month "
                    f"(of {_num(_uso.get('limit_hours'))}; they renew on "
                    f"{_uso.get('resets', 'the 1st')}).")
            st.caption(_txt_uso)
            if _uso and _uso.get('restricted'):
                st.warning(i18n.pick(
                    'Earth Engine ha limitado ahora la cuenta compartida (ha gastado su '
                    'cuota de este mes): los cálculos irán lentos y alguno puede fallar. '
                    'Si tienes cuenta propia, escribe tu proyecto.',
                    'Earth Engine is now limiting the shared account (it has spent its '
                    'quota for this month): computations will be slow and some may fail. '
                    'If you have your own account, write your project.'))
        c1, c2 = st.columns(2)
        with c1:
            if st.button('🔑 Autenticar GEE'):
                try:
                    import gee_backend as geb
                    geb.authenticate()
                    st.success('Autenticación lanzada (revisa el navegador).')
                except Exception as e:
                    st.error(f'Error de autenticación: {e}')
        with c2:
            if st.button('✓ Probar conexión'):
                try:
                    import gee_backend as geb
                    ok, msg = geb.initialize(gee_project.strip() or None)
                    (st.success if ok else st.error)(msg)
                    if ok and gee_project.strip():
                        guardar_ajuste('gee_project', gee_project.strip())
                except Exception as e:
                    st.error(str(e))
        st.caption('Sentinel-2, el radar (Sentinel-1) y el relieve se calculan en los '
                    'servidores de Google (rápido, sin descargas).')

    st.divider()
    st.header('Modelo de partida')

    st.caption('🧭 ¿Qué modelo elegir según tu zona? Mira la pestaña '
               '**📖 Documentación** (arriba) para la guía completa y las '
               'recomendaciones por parque.')

    _models = list_models()
    if _models:
        sel_model_name = st.selectbox(
            'Modelo base Random Forest', _models, index=0, key='modelo_partida',
            help=i18n.pick(
                'Modelos .joblib de la carpeta de la app. Es el que da la probabilidad '
                'de cada punto, aunque tus puntos traigan etiquetas de campo (con ellas '
                'la app mide cuánto acierta). Para entrenar además uno con tus puntos, '
                'marca la casilla de abajo. Un modelo entrenado en tu zona se puede '
                'guardar aquí (panel de resultados) para reutilizarlo.',
                '.joblib models from the app folder. It is the one that gives each '
                'point its probability, even if your points carry field labels (with '
                'them the app measures how many it gets right). To also train one with '
                'your points, tick the box below. A model trained on your area can be '
                'saved here (results panel) to reuse it.'))
        start_model_path = str(model_path(sel_model_name))
        _sel_feats = []
        try:
            _mb = bp.load_model_bundle(start_model_path)
            _sel_feats = _mb.get('features', [])
            _auc_m = _mb.get('auc_groupkfold')
            st.caption(f"Zona: {_mb.get('training_zone','?')} · "
                       f"n={_mb.get('training_n','?')} · "
                       f"AUC={_num(_auc_m)}" if isinstance(_auc_m, (int, float))
                       else f"Zona: {_mb.get('training_zone','?')} · "
                            f"n={_mb.get('training_n','?')} · AUC=?")
            # Planetary Computer no lee el radar: se usará la versión sin radar
            _sr = modelo_sin_radar(start_model_path) if backend != 'gee' else None
            if _sr is not None:
                _auc_sr = _auc_modelo(str(_sr), _sr.stat().st_mtime)
                _txt_sr = '' if _auc_sr is None else f' (AUC {_n3(_auc_sr)})'
                st.info(i18n.pick(
                    '📡 Con Planetary Computer no se lee el radar, así que la app usará '
                    f'la versión de este modelo entrenada sin radar{_txt_sr}. Acierta más '
                    'que este modelo cuando le falta el radar.',
                    '📡 Planetary Computer does not read the radar, so the app will use '
                    f'the version of this model trained without radar{_txt_sr}. It does '
                    'better than this model when the radar is missing.'))
            # Badge contextual: qué fuente extra usará y si está disponible
            _needs_cir = any(f.startswith('cir_') for f in _sel_feats)
            _needs_ps  = any(f.startswith('ps_')  for f in _sel_feats)
            if _needs_cir:
                st.success('🇪🇸 Usa **PNOA Falso Color IR 0,25 m** (gratis). Cobertura WMS '
                           'IR en Andalucía, Cataluña y Canarias; se extrae '
                           'automáticamente (en otras zonas, mejor el modelo base).')
            if _needs_ps:
                _k = get_secret('PL_API_KEY', None) or os.environ.get('PL_API_KEY', '')
                if _k:
                    st.success('🌍 Usa **PlanetScope 3 m**. Key detectada; se extrae '
                               'automáticamente (consume cuota Planet).')
                else:
                    st.warning('🌍 Este modelo necesita **PlanetScope 3 m**, pero no hay '
                               '`PL_API_KEY`. Sin ella, esas variables van vacías '
                               '(predicción degradada). Configúrala o elige otro modelo.')
            if not _needs_cir and not _needs_ps:
                st.caption(i18n.pick(
                    f'Modelo de {_mb.get("training_zone", "?").split(" (")[0]} '
                    f'({len(_sel_feats)} variables): no requiere fuentes extra ni claves.',
                    f'Model of {_mb.get("training_zone", "?").split(" (")[0]} '
                    f'({len(_sel_feats)} variables): needs no extra sources or keys.'))
            st.warning(i18n.pick(
                'Este modelo se entrenó **solo con puntos de Sierra Nevada**. En otra zona, '
                'úsalo para **proponer** puntos y revísalos (pestaña Revisar puntos) antes de fiarte '
                'del resultado. Si tus puntos traen etiquetas de campo, la app te dirá '
                'cuántas acierta.',
                'This model was trained **only with points from Sierra Nevada**. In '
                'another area, use it to **propose** points and review them (Review points tab) '
                'before trusting the result. If your points carry field labels, the app '
                'will tell you how many it gets right.')
                if 'sierra nevada' in str(_mb.get('training_zone', '')).lower() else
                i18n.pick(f"Entrenado en: {_mb.get('training_zone', '?')}. En otra zona, "
                          'revisa los puntos antes de fiarte del resultado.',
                          f"Trained in: {_mb.get('training_zone', '?')}. In another area, "
                          'review the points before trusting the result.'))
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
            'con etiqueta: los de campo, los que verificas en «Revisar puntos» y los que añades '
            'pinchando en el mapa. Es un solo modelo con todos los puntos, y la app te dice '
            'cuánto acierta en los tuyos (validación por zonas). Se aplica al ejecutar y al '
            'verificar o añadir puntos; no hace falta volver a ejecutar.',
            'Unticked, the app uses the model above as it is and, if your points carry '
            'field labels, tells you how many it gets right. Ticked, it retrains it with '
            'the 769 Sierra Nevada points it was made with PLUS all your labelled points: '
            'field points, the ones you verify in «Review points» and the ones you add by '
            'clicking on the map. It is a single model with all the points, and the app '
            'tells you how well it does on yours (spatial validation). It applies when '
            'running and when verifying or adding points; there is no need to run again.'))

    st.divider()
    st.header('Opciones')
    years_str = st.text_input('Años Sentinel-2 (admite rangos, p. ej. 2017-2025)',
                                value='2017-2025')
    thr_mode = st.radio(
        'Umbral RF para BORREGUIL', [
            'Automático · Otsu',
            'Manual (deslizador)',
            'Automático · verdad-terreno',
        ], index=0,
        help=('**Otsu (recomendado, por defecto)**: se adapta a cada ejecución. '
              'Al aplicar el modelo a una zona nueva las probabilidades se '
              'recalibran, así que un umbral manual fijo puede dejar casi todo '
              'fuera; Otsu evita ese problema.\n\n'
              '**Manual**: eliges el valor con el deslizador.\n\n'
              '**Otsu**: calcula el corte automáticamente a partir de la *forma* '
              'de la distribución de probabilidades RF (método de Otsu, el clásico '
              'para binarizar imágenes). Encuentra el valor que mejor separa los '
              'dos grupos —probable no-borreguil vs. borreguil— minimizando la '
              'varianza dentro de cada grupo. **No necesita verdad-terreno.**\n\n'
              '**Verdad-terreno**: usa tus puntos de campo. Con presencias y '
              'ausencias, el corte que maximiza el índice de Youden '
              '(sensibilidad + especificidad − 1). Solo con presencias, el corte '
              'que captura el 90% de los borreguiles conocidos. Si no hay '
              'suficiente verdad-terreno, recurre a Otsu.'))
    threshold = st.slider('Valor del umbral', 0.30, 0.80, 0.50, 0.05,
                          disabled=not thr_mode.startswith('Manual'))
    img_src_label = st.radio('Fuente de imágenes', [
        '🌐  ESRI World Imagery (global, ~0,5 m)',
        '🇪🇸  PNOA IGN + autonómica (España, 0,25 m)',
    ], index=0,
    help=('PNOA IGN usa ortofoto autonómica cuando está disponible (Andalucía 2022, '
          'Aragón 2024, Cataluña vigente, Cantabria 2023, CyL 2020, Canarias) y cae '
          'en el PNOA-MA nacional para el resto de España. Fuera de España usa ESRI.'))
    img_src = 'pnoa' if 'PNOA' in img_src_label else 'esri'
    fast_mode = st.checkbox(
        '⚡ Modo rápido (vista previa)', value=False,
        help='Salta las descargas lentas que se hacen punto a punto en tu ordenador: '
             'las imágenes aéreas (y sus texturas) y CIR/Planet. Con Earth Engine, el '
             'resto (relieve, Sentinel-2 y radar) se calcula en la nube en segundos. '
             'Da resultados APROXIMADOS (sin texturas): sirve para probar rápido y '
             'después afinar desmarcando esta casilla.')
    if fast_mode:
        st.caption('⚡ Vista previa: sin imágenes aéreas ni CIR/Planet. '
                   'Resultados aproximados.')
    skip_imgs = st.checkbox('Saltar imágenes (más rápido, sin texturas)',
                            value=False, disabled=fast_mode) or fast_mode
    # Sin scikit-image (p. ej. su DLL bloqueada por Windows) no hay texturas aunque
    # se descarguen las imágenes.
    _sin_texturas = isinstance(bp.graycomatrix, bp._SkimageNoDisponible)
    _modelo_con_img = any(f in st.session_state.get('sel_model_feats', [])
                          for f in bp.FEATURES_RF[:12])
    if _modelo_con_img and (skip_imgs or _sin_texturas):
        st.warning(i18n.pick(
            ('Este ordenador no puede calcular las texturas de la imagen (Windows bloquea '
             'la librería scikit-image). ' if _sin_texturas and not skip_imgs else
             'Sin imágenes no hay texturas. ')
            + 'El modelo de partida se entrenó con ellas: aquí se rellenan con un valor '
              'típico de Sierra Nevada y su probabilidad es **menos fiable**. Úsala para '
              'proponer puntos y revísalos.',
            ('This computer cannot compute the image textures (Windows blocks the '
             'scikit-image library). ' if _sin_texturas and not skip_imgs else
             'Without images there are no textures. ')
            + 'The starting model was trained with them: here they are filled with a '
              'typical Sierra Nevada value and its probability is **less reliable**. Use '
              'it to propose points and review them.'))
    skip_s2   = st.checkbox('Saltar Sentinel-2 (más rápido, sin NDVI)', value=False)

    _pl_key = get_secret('PL_API_KEY', None) or os.environ.get('PL_API_KEY', '')
    use_planet = st.checkbox(
        'PlanetScope 3 m (8 bandas SR) — requiere PL_API_KEY',
        value=False, disabled=not _pl_key,
        help=('Pide imágenes PlanetScope vía Orders API (clip al área de los puntos). '
              'CONSUME CUOTA de tu suscripción Planet (km² pedidos) y tarda 10-30 min '
              'extra. Útil con el modelo variante rf_*_planet.joblib (60 variables). '
              'Configura PL_API_KEY como variable de entorno o en st.secrets.'))
    if not _pl_key:
        st.caption('PlanetScope desactivado: no hay PL_API_KEY en el entorno/secrets.')

    st.divider()
    st.header('Acción')
    # Habilitar botón si hay puntos cargados O si vamos a generar y hay área
    can_run = (f_input is not None) or (input_mode.startswith('🎲') and gen_n > 0
                                          and (f_area or wdpa_id or pa_name))
    run_btn = st.button('▶ Ejecutar pipeline', type='primary', disabled=not can_run)

    st.divider()
    with st.popover(i18n.pick('💬 Sugerir un cambio o informar de un error',
                              '💬 Suggest a change or report an error'),
                    width='stretch'):
        mostrar_sugerencias()
    _v = version_app()
    if _v:
        st.caption(i18n.pick(f'Versión {_v}', f'Version {_v}'))
    _txt_gh = texto_modelos_github(_modelos_gh)
    if _txt_gh:
        st.caption(_txt_gh)

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
st.html('<style>' + '''
FILA{gap:.5rem;row-gap:.4rem;flex-wrap:wrap}
FILA>button[role="tab"]{flex:1 1 auto;justify-content:center;height:auto;
  padding:.7rem 1.1rem;border-radius:.6rem .6rem 0 0;background:rgba(128,128,128,.13)}
FILA>button[role="tab"] p{font-size:1.2rem;font-weight:600}
FILA>button[role="tab"][aria-selected="true"]{
  background:color-mix(in srgb,currentColor 14%,transparent);
  box-shadow:inset 0 -4px 0 currentColor}
FILA>[data-baseweb="tab-highlight"]{display:none}
'''.replace('FILA', '.st-key-pestanas_principales>div>div>[data-baseweb="tab-list"]')
        + '</style>')
tab_analisis, tab_tipos, tab_docs = st.tabs(
    ['📍  Verdad terreno', '🌿  Mapa Modelo RF', '📖  Documentación y recomendaciones'],
    key='pestanas_principales')


def parar_ejecucion():
    """Detiene esta ejecución (st.stop). La pestaña «Mapa Modelo RF» se dibuja al
    final del guion: si se para antes, hay que dejarle algo escrito."""
    with tab_tipos:
        st.info(i18n.pick(
            'La última ejecución se ha detenido antes de terminar. Mira el aviso en la '
            'pestaña **📍 Verdad terreno**.',
            'The last run stopped before finishing. See the message in the '
            '**📍 Ground truth** tab.'))
    st.stop()


with tab_docs:
    render_docs()

# El botón «Probar con 20 puntos de ejemplo» de la pantalla de inicio deja aquí su
# petición (on_click corre antes que el guion): así esta pasada ya no dibuja la
# pantalla de inicio y lanza el análisis como «Ejecutar pipeline», con los puntos de
# la plantilla, sin área y con lo elegido en el panel lateral.
EJEMPLO_PUNTOS = APP_DIR / 'plantilla' / 'example_sierra_nevada.csv'
PLANTILLA_EXCEL = APP_DIR / 'plantilla' / 'field_points_template.xlsx'
con_ejemplo = bool(st.session_state.pop('_pedir_ejemplo', False)) and EJEMPLO_PUNTOS.exists()


def _pedir_ejemplo():
    st.session_state['_pedir_ejemplo'] = True


with tab_analisis:
    if 'rows' not in st.session_state and not run_btn and not con_ejemplo:
        # Pantalla de inicio (revisada con la autora el 24-11-2026): qué hace la app, qué hay
        # que hacer y qué se obtiene, y un ejemplo que se prueba con un clic.
        st.markdown(i18n.pick('### Encuentra borreguiles en tu zona y haz su mapa',
                              '### Find borreguiles in your area and map them'))
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(i18n.pick('#### 1 · Tus puntos', '#### 1 · Your points'))
            st.markdown(i18n.pick(
                'Son los sitios en los que la app mira si hay borreguil.\n\n'
                '**¿No tienes puntos? No te preocupes:** la app los genera al azar en tu '
                'zona. En el panel lateral 👈, elige «🎲 Generar aleatoriamente en el área» '
                'e indica el área.\n\n'
                '**¿Tienes puntos?** Súbelos en el panel lateral (KML, GeoJSON, Shapefile o '
                'CSV). Mira los requisitos en la plantilla de abajo.',
                'They are the places where the app checks whether there is borreguil.\n\n'
                "**No points? Don't worry:** the app generates them at random in your "
                'area. In the side panel 👈, choose «🎲 Generate randomly within the area» and '
                'give the area.\n\n'
                '**Have points?** Upload them in the side panel (KML, GeoJSON, Shapefile or '
                'CSV). See the requirements in the template below.'))
        with col2:
            st.markdown(i18n.pick('#### 2 · Analiza', '#### 2 · Analyse'))
            st.markdown(i18n.pick(
                # Texto de la autora (24-11-2026), acortado a petición suya
                'Pulsa **▶ Ejecutar pipeline**, abajo del todo en el panel lateral. En cada '
                'punto, la app calcula variables de Sentinel-1, Sentinel-2, topografía y '
                'textura de la imagen aérea, y un Random Forest entrenado en Sierra Nevada '
                'estima la probabilidad de que sea borreguil. Con 20 puntos tarda alrededor '
                'de un minuto.\n\n'
                'Con una cuenta de **Google Earth Engine** es más rápido y lee el radar '
                '(AUC 0,847). Sin cuenta, la app usa **Planetary Computer**: más lento y '
                'sin radar (AUC 0,833).',
                'Press **▶ Run pipeline**, at the very bottom of the side panel. At each '
                'point, the app computes variables from Sentinel-1, Sentinel-2, terrain and '
                'aerial-image texture, and a Random Forest trained in Sierra Nevada '
                'estimates the probability that it is borreguil. With 20 points it takes '
                'about a minute.\n\n'
                'With a **Google Earth Engine** account it is faster and reads the radar '
                '(AUC 0.847). Without an account, the app uses **Planetary Computer**: '
                'slower and without radar (AUC 0.833).'))
        with col3:
            st.markdown(i18n.pick('#### 3 · Qué obtienes', '#### 3 · What you get'))
            st.markdown(i18n.pick(
                '- Para cada punto, si es borreguil, con qué probabilidad y de qué tipo '
                '(ambiente, humedad y pureza)\n'
                # Texto de la autora (24-11-2026)
                '- Puedes revisarlos para validarlos, añadir puntos y reentrenar el modelo '
                'de detección de puntos\n'
                '- En la pestaña **🌿 Mapa Modelo RF**, con los puntos revisados, mapeas el '
                'borreguil de tu zona de estudio',
                '- For each point, whether it is borreguil, with what probability and of '
                'what type (environment, moisture and purity)\n'
                '- You can review them to validate them, add points and retrain the point '
                'detection model\n'
                '- In the **🌿 RF model map** tab, with the reviewed points, you map the '
                'borreguil of your study area'))
        b_ej, b_pl = st.columns(2)
        if EJEMPLO_PUNTOS.exists():
            b_ej.button(i18n.pick('▶ Probar con 20 puntos de ejemplo de Sierra Nevada',
                                  '▶ Try with 20 example points from Sierra Nevada'),
                        key='probar_ejemplo', on_click=_pedir_ejemplo,
                        help=i18n.pick(
                            'Analiza los 20 puntos de la plantilla de datos de campo (12 '
                            'borreguiles y 8 sitios sin borreguil) con la fuente de datos y '
                            'el modelo elegidos en el panel lateral. No hace falta subir nada.',
                            'Analyses the 20 points of the field data template (12 '
                            'borreguiles and 8 places with no borreguil) with the data '
                            'source and the model chosen in the side panel. Nothing to '
                            'upload.'))
        if PLANTILLA_EXCEL.exists():
            b_pl.download_button(
                i18n.pick('⬇ Descargar la plantilla de datos de campo (Excel)',
                          '⬇ Download the field data template (Excel)'),
                PLANTILLA_EXCEL.read_bytes(), file_name=PLANTILLA_EXCEL.name,
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                key='bajar_plantilla',
                help=i18n.pick(
                    'Una fila por punto: sus coordenadas (lon y lat, en grados), `presence` '
                    '(1 borreguil, 0 no) y, si quieres, el tipo. La segunda hoja explica '
                    'cada columna. Rellénala y guárdala como CSV para subirla.',
                    'One row per point: its coordinates (lon and lat, in degrees), '
                    '`presence` (1 borreguil, 0 not) and, if you want, the type. The second '
                    'sheet explains each column. Fill it in and save it as CSV to upload it.'))
        st.caption(i18n.pick(
            'AUC con validación cruzada por cuencas. El modelo se entrenó solo en Sierra '
            'Nevada: en otras montañas, revisa los resultados antes de fiarte de ellos.',
            'AUC with cross-validation by catchment. The model was trained only in Sierra '
            'Nevada: in other mountains, review the results before relying on them.'))

    # ============================================================
    # PIPELINE EXECUTION
    # ============================================================
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
        # Workspace
        work = Path(tempfile.mkdtemp(prefix='borreguil_'))
        st.session_state.work = work

        if con_ejemplo:
            # El ejemplo: los puntos de la plantilla (llevan su etiqueta de campo), sin
            # verdad-terreno aparte ni área, aunque el panel lateral tenga algo puesto.
            st.info(i18n.pick(
                'Analizando los **20 puntos de ejemplo** de Sierra Nevada (12 borreguiles '
                'y 8 sitios sin borreguil)…',
                'Analysing the **20 example points** from Sierra Nevada (12 borreguiles '
                'and 8 places with no borreguil)…'))
            input_path, truth_path, area_path = str(EJEMPLO_PUNTOS), None, None
        else:
            try:                       # con su nombre: un Shapefile son varios ficheros
                input_path = _guardar_subida(f_input)
                truth_path = _guardar_subida(f_truth)
            except ValueError as e:
                st.error(str(e))
                parar_ejecucion()
            area_path  = save_uploaded_area(f_area) if f_area else None

        # Resolve study area FIRST (may be needed for generation)
        study_geom = None; study_info = None
        if not con_ejemplo and (area_path or wdpa_id or pa_name):
            with st.status('Resolviendo área de estudio…', expanded=True) as status:
                try:
                    import study_area as sa
                    # Token Protected Planet (campo, secret o env) y uso opcional de GEE
                    _wtok = ((wdpa_token.strip() if wdpa_token else None)
                             or get_secret('WDPA_TOKEN', None) or os.environ.get('WDPA_TOKEN'))
                    _use_gee = False
                    if backend == 'gee' and wdpa_id and not _wtok:
                        # Inicializar EE para resolver el WDPA ID vía la capa WCMC/WDPA
                        try:
                            import gee_backend as geb
                            if not geb.is_initialized():
                                geb.initialize(gee_project.strip() or get_secret('GEE_PROJECT', None))
                            _use_gee = geb.is_initialized()
                        except Exception:
                            _use_gee = False
                    study_geom, study_info = sa.resolve(area_path,
                                                        wdpa_id.strip() if wdpa_id else None,
                                                        pa_name.strip() if pa_name else None,
                                                        wdpa_token=_wtok, use_gee=_use_gee)
                    if study_geom:
                        st.write(f"**Origen**: {study_info['source']}")
                        if study_info.get('name'):
                            st.write(f"**Nombre**: {study_info['name']}")
                        st.write(f"**Área aprox**: {_num(study_info['area_km2'])} km²")
                        status.update(label='✓ Área de estudio resuelta', state='complete')
                    else:
                        if wdpa_id:
                            st.warning('No se pudo resolver el WDPA ID. Para IDs fuera del '
                                       'cache hace falta un **token de Protected Planet** o '
                                       'el **backend GEE**. Alternativa: busca por nombre o '
                                       'sube un vector. Se usará bbox de los puntos.')
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
        pbar = st.progress(0.0, text='Iniciando extracción de variables…')
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
            set_prog(0.28, 'Sentinel-2 (NDVI, Clre…)…')
            with st.status(f'Sentinel-2 multi-temporal ({active_backend.upper()})…', expanded=True) as status:
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
                        status.update(label=f'🔴 Sentinel-2: 0/{len(rows)} puntos con NDVI '
                                      '(ver Diagnóstico abajo)', state='error')
                    elif cov < len(rows):
                        status.update(label=f'🟡 Sentinel-2: {cov}/{len(rows)} puntos con '
                                      f'NDVI ({len(years)} años)', state='complete')
                    else:
                        status.update(label=f'✓ Sentinel-2: {cov}/{len(rows)} puntos '
                                      f'({len(years)} años)', state='complete')
                except Exception as e:
                    status.update(label=f'⚠ Sentinel-2 falló: {e}', state='error')
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
        # se entrenaron con esos valores). Con Planetary Computer no se lee y se usa la
        # versión del modelo sin radar (ver modelo_segun_radar). Si falla, igual.
        skip_s1 = skip_s2 or active_backend != 'gee'
        if not skip_s1:
            set_prog(0.70, 'Sentinel-1 (radar)…')
            with st.status('Sentinel-1 (radar, Earth Engine)…', expanded=False) as status:
                try:
                    with live_logs(status):
                        import gee_backend as geb
                        geb.fetch_s1_gee(rows, bbox)
                    cov = _coverage(rows, bp.S1_FEATS)
                    status.update(label=f'✓ Sentinel-1: {cov}/{len(rows)} puntos',
                                  state='complete')
                except Exception as e:
                    status.update(label=f'⚠ Sentinel-1 falló (se continúa sin radar): {e}',
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
                                      '(Andalucía/Cataluña/Canarias): mejor el modelo base',
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
            # Sin radar (Planetary Computer, o falló): la versión del modelo sin radar
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
                st.info(f'🎓 {_n_hidden} puntos de verdad-terreno usados **solo para '
                        f'entrenar** (ocultos del mapa, la tabla y las descargas). '
                        f'Se muestran tus {len(rows)} candidatos.')

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
        st.success(i18n.pick(
            'Análisis terminado: los resultados están abajo y los ficheros, en la '
            'pestaña Descargas.',
            'Analysis finished: the results are below and the files, in the Downloads '
            'tab.'))

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

        st.divider()
        st.header('Resultados')

        # Summary
        # En pantallas estrechas los rótulos de las tarjetas se cortaban con «…»
        # («POSIBLE BORRE…»): se dejan partir en varias líneas.
        st.markdown('<style>[data-testid="stMetricLabel"] p{white-space:normal!important;'
                    'overflow:visible!important;text-overflow:clip!important;'
                    'overflow-wrap:anywhere;}</style>', unsafe_allow_html=True)
        counts = Counter(r['decision'] for r in rows)
        # Alineadas por abajo: las cifras quedan a la misma altura aunque un rótulo
        # ocupe dos líneas y el de al lado una.
        cols = st.columns(min(6, len(counts)+1), vertical_alignment='bottom')
        cols[0].metric('Total', len(rows))
        for i, (k, v) in enumerate(counts.most_common(5)):
            # Rótulo como texto normal («No borreguil verificado»), no en mayúsculas
            cols[i+1].metric(i18n.tv_decision(k), v)

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
        with st.expander(i18n.pick('🧭 Modelo usado y acierto con tus puntos de campo',
                                   '🧭 Model used and accuracy on your field points'),
                         expanded=True):
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
            if _pendiente and (_n_etq or not _propio):
                if st.button(
                        i18n.pick(f'🔁 Reentrenar ahora sumando mis {_n_etq} puntos',
                                  f'🔁 Retrain now adding my {_n_etq} points') if _propio
                        else i18n.pick('↩ Volver al modelo de partida, sin mis puntos',
                                       '↩ Back to the starting model, without my points'),
                        type='primary', key='aplicar_casilla'):
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
        def _add_points(fmap, pts, highlight_id=None, radius=6, hueco=False):
            # hueco=True: el punto resaltado se dibuja sin relleno, para que el
            # marcador no tape justo el terreno que hay que mirar.
            _rotulos = i18n.pick(
                {'tipo': 'Tipo', 'revisado': 'revisado', 'altitud': 'Altitud',
                 'pendiente': 'Pendiente', 'variables': 'Las que más pesan en el modelo'},
                {'tipo': 'Type', 'revisado': 'reviewed', 'altitud': 'Elevation',
                 'pendiente': 'Slope', 'variables': 'Those that weigh most in the model'})
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
                color = bp.decision_color(r.get('decision', 'SIN PREDICCIÓN'))
                # Números con 3 decimales y nombres largos partidos en líneas.
                popup = bp.popup_html(r, i18n.tv, _rotulos,
                                      nombre=_nombre_pop.get(str(r.get('ID'))),
                                      variables=_vars_pop, num=_num)
                is_sel = (highlight_id is not None and r.get('ID') == highlight_id)
                if is_sel:
                    folium.CircleMarker([r['lat'], r['lon']], radius=radius+8,
                                        color='#FF1744', weight=4, fill=False).add_to(fmap)
                _relleno = not (hueco and is_sel)
                # Los añadidos pinchando en el mapa llevan borde blanco
                _del_mapa = r.get('source') == 'mapa'
                folium.CircleMarker([r['lat'], r['lon']],
                                    radius=radius + 2 if _del_mapa else radius,
                                    color='#FFFFFF' if _del_mapa else color,
                                    weight=3,
                                    fill=True, fill_color=color,
                                    fill_opacity=0.85 if _relleno else 0.0,
                                    popup=folium.Popup(popup, max_width=300)).add_to(fmap)

        # Leyenda de colores de los puntos (decisión del Random Forest).
        _LEGEND_ITEMS = [
            ('BORREGUIL VERIFICADO', 'Verificado'),
            ('BORREGUIL PROBABLE', 'Probable'),
            ('POSIBLE BORREGUIL', 'Posible'),
            ('DUDOSO', 'Dudoso'),
            ('INCIERTO', 'Incierto'),
            ('NO BORREGUIL', 'No borreguil'),
        ]

        def _add_legend(fmap, rows=None):
            """Añade una leyenda HTML fija (abajo-izda) con los colores de decisión.
            Si se pasan `rows`, solo muestra las categorías presentes."""
            present = None
            if rows is not None:
                present = {bp.decision_color(r.get('decision', '')) for r in rows}
            filas = ''
            for dec, _lab in _LEGEND_ITEMS:
                lab = i18n.tv(_lab)
                col = bp.decision_color(dec)
                if present is not None and col not in present:
                    continue
                filas += (f'<div style="margin:2px 0;"><span style="display:inline-block;'
                          f'width:12px;height:12px;border-radius:50%;background:{col};'
                          f'margin-right:6px;vertical-align:middle;border:1px solid #555;">'
                          f'</span>{lab}</div>')
            if rows is not None and any(r.get('source') == 'mapa' for r in rows):
                filas += ('<div style="margin:2px 0;"><span style="display:inline-block;'
                          'width:12px;height:12px;border-radius:50%;background:#777;'
                          'margin-right:6px;vertical-align:middle;border:3px solid #FFF;'
                          'box-shadow:0 0 0 1px #555;"></span>'
                          + i18n.pick('Añadido en el mapa', 'Added on the map') + '</div>')
            html = (
                '<div style="position:fixed;bottom:24px;left:12px;z-index:9999;'
                'background:rgba(255,255,255,0.92);padding:8px 10px;border-radius:6px;'
                'border:1px solid #999;font-size:12px;line-height:1.1;'
                'box-shadow:0 1px 4px rgba(0,0,0,0.3);">'
                '<div style="font-weight:bold;margin-bottom:4px;">Clasificación</div>'
                f'{filas}</div>')
            fmap.get_root().html.add_child(folium.Element(html))

        def _base_map(center, zoom, ortofoto=False):
            # max_zoom alto para acercar bastante a los puntos (over-zoom de las
            # ortofotos); Google Satélite tiene resolución nativa mayor que ESRI.
            # tiles=None: las capas de fondo se añaden aquí abajo. Sin él, folium ponía
            # además su OpenStreetMap y salía dos veces en la lista de capas.
            fm = folium.Map(location=center, zoom_start=zoom, max_zoom=22,
                            control_scale=True, tiles=None)
            capas = [
                dict(tiles='https://services.arcgisonline.com/ArcGIS/rest/services/'
                           'World_Imagery/MapServer/tile/{z}/{y}/{x}',
                     attr='Esri', name='ESRI World Imagery', max_zoom=22,
                     max_native_zoom=19),
                dict(tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
                     attr='Google', name='Google Satélite (más zoom)', max_zoom=22,
                     max_native_zoom=21),
                dict(tiles='OpenStreetMap', name='OpenStreetMap', max_zoom=19),
            ]
            if ortofoto:
                # Para decidir si un borreguil es de arroyo o de ladera hay que ver
                # la forma del terreno: el sombreado del LiDAR del IGN (PNOA, ~2 m)
                # deja ver regueros y cauces pequeños que la ortofoto no enseña.
                capas.append(dict(
                    tiles='https://wmts-mapa-lidar.idee.es/lidar?request=GetTile'
                          '&service=WMTS&version=1.0.0&layer=EL.GridCoverageDSM'
                          '&style=default&tilematrixset=GoogleMapsCompatible'
                          '&format=image/png&TileMatrix={z}&TileRow={y}&TileCol={x}',
                    attr='IGN · LiDAR PNOA', name='Relieve LiDAR (IGN, sombreado)',
                    max_zoom=22, max_native_zoom=19))
                # Para revisar un punto hay que ver el terreno: la ortofoto pasa a ser
                # la capa visible (la última capa base que se añade) y las otras
                # quedan disponibles en el control de capas.
                capas = capas[1:] + capas[:1]
            # Se ve la última capa de fondo (en el mapa de resultados, OpenStreetMap);
            # las demás quedan en el control de capas.
            for _c in capas[:-1]:
                _c['show'] = False
            for _c in capas:
                folium.TileLayer(**_c).add_to(fm)
            if ortofoto:
                folium.raster_layers.WmsTileLayer(
                    url='https://servicios.idee.es/wms-inspire/mdt',
                    layers='EL.ContourLine', fmt='image/png', transparent=True,
                    attr='IGN · MDT', name='Curvas de nivel (IGN, cada 50 m)',
                    overlay=True, show=False, max_zoom=22).add_to(fm)
            if sg is not None:
                from shapely.geometry import mapping
                folium.GeoJson(mapping(sg), name='Área de estudio',
                               style_function=lambda x: {'fillColor':'#1976D2','color':'#0D47A1',
                                                          'weight':2,'fillOpacity':0.08}).add_to(fm)
            return fm

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
            g_pix = folium.FeatureGroup(name='▫ Píxel S2 (10 m)', show=show)
            g_w10 = folium.FeatureGroup(name='▢ Ventana muestreo 10 m (S2 + S1)', show=show)
            g_w20 = folium.FeatureGroup(name='▢ Ventana muestreo 20 m (S2)', show=False)
            for fp in fps:
                folium.Polygon(fp['pixel10'], color='#FFD600', weight=1,
                               fill=True, fill_color='#FFD600', fill_opacity=0.12).add_to(g_pix)
                folium.Polygon(fp['win_s2_10'], color='#00E5FF', weight=1,
                               fill=False).add_to(g_w10)
                folium.Polygon(fp['win_s2_20'], color='#FF6D00', weight=1,
                               fill=False, dash_array='4').add_to(g_w20)
            g_pix.add_to(fm); g_w10.add_to(fm); g_w20.add_to(fm)
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

        def _resumen_anadidos(con_deshacer):
            filas, n, ok = bp.resumen_mapa(rows, threshold)
            if not filas:
                return
            st.markdown(
                i18n.pick(f'**Puntos añadidos en el mapa: {len(filas)}**',
                          f'**Points added on the map: {len(filas)}**')
                + (i18n.pick(f' · el modelo coincidía en **{ok} de {n}**',
                             f' · the model agreed on **{ok} of {n}**') if n else ''))
            _c_quien = i18n.pick('Quién', 'Who')
            _c_nota = i18n.pick('Nota', 'Note')
            _c_p = i18n.pick('Probabilidad del modelo', 'Model probability')
            df_m = pd.DataFrame([{
                i18n.pick('Punto', 'Point'): f['ID'],
                i18n.pick('Dijo', 'Said'): _RESP_MAPA[f['respuesta']],
                _c_quien: f['observer'],
                _c_p: f['proba'],
                i18n.pick('El modelo decía', 'The model said'): _OPINION.get(f['modelo'], '—'),
                i18n.pick('¿Coincide?', 'Agrees?'): ('✓' if f['coincide']
                                                     else '✗' if f['coincide'] is False
                                                     else '—'),
                _c_nota: f['notes'],
            } for f in filas])
            for _c in (_c_quien, _c_nota):
                if not df_m[_c].astype(str).str.strip().any():
                    df_m = df_m.drop(columns=_c)
            st.dataframe(_con_coma(df_m), hide_index=True, width='stretch')
            st.caption(i18n.pick(
                'La probabilidad es la que daba el modelo al añadir el punto, antes de saber '
                'la respuesta. Los dudosos no cuentan. Los puntos añadidos van en las '
                'descargas como datos de campo (`method` = local knowledge).',
                'The probability is the one the model gave when the point was added, '
                'before knowing the answer. Doubtful points do not count. The added '
                'points go into the downloads as field points (`method` = local '
                'knowledge).'))
            if con_deshacer:
                _ult_id = filas[-1]['ID']
                if st.button(i18n.pick(f'↩ Quitar el último añadido ({_ult_id})',
                                       f'↩ Remove the last one added ({_ult_id})'),
                             key='mapa_deshacer'):
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
                    st.markdown(i18n.pick(
                        f'📍 Has pinchado en **{_num(lat, 5)}; {_num(lng, 5)}** (círculo rojo). '
                        '¿Hay borreguil aquí?',
                        f'📍 You clicked on **{_num(lat, 5)}, {_num(lng, 5)}** (red circle). '
                        'Is there a borreguil here?'))
                    st.radio(i18n.pick('Certidumbre de tu respuesta (opcional)',
                                       'Certainty of your answer (optional)'),
                             list(bp.CERTIDUMBRES), index=None, horizontal=True,
                             format_func=_CERT_TXT.get, key='mapa_certidumbre',
                             help=_CERT_AYUDA)
                    _b1, _b2, _b3, _b4 = st.columns([3, 3, 3, 2])
                    resp = None
                    if _b1.button(i18n.pick('✓ Aquí hay borreguil', '✓ There is a borreguil'),
                                  type='primary', key='mapa_si'):
                        resp = 'si'
                    if _b2.button(i18n.pick('✗ Aquí no hay', '✗ There is none'),
                                  key='mapa_no'):
                        resp = 'no'
                    if _b3.button(i18n.pick('? No estoy seguro', '? Not sure'),
                                  key='mapa_duda'):
                        resp = 'duda'
                    if _b4.button(i18n.pick('Cancelar', 'Cancel'), key='mapa_cancelar'):
                        st.session_state['_clic_usado'] = clic
                        st.rerun()
                    if resp:
                        _anadir_del_mapa(resp, lat, lng)
            else:
                st.caption(i18n.pick('👆 Pincha en el mapa donde sepas si hay borreguil.',
                                     '👆 Click on the map where you know whether there '
                                     'is a borreguil.'))

        _q1, _q2 = st.columns([3, 2])
        _q1.text_input(i18n.pick('Quién revisa (opcional)', 'Who is reviewing (optional)'),
                       key='revisor_nombre')
        _q2.selectbox(i18n.pick('Perfil (opcional)', 'Profile (optional)'),
                      list(bp.PERFILES), index=None, format_func=_PERFIL_TXT.get,
                      placeholder=i18n.pick('Elige…', 'Choose…'), key='revisor_perfil')
        st.caption(i18n.pick(
            'Se apunta, con la certidumbre que indiques, en cada punto que verifiques en '
            '«Revisar puntos» o añadas en el mapa, y sale en las descargas.',
            'It is recorded, with the certainty you give, in each point you verify in '
            '"Review points" or add on the map, and it goes into the downloads.'))

        # Tabs
        # Nombres y orden de la autora (24-11): Descargas, al final.
        # «Probabilidades», «Peso de las variables» y «Tipos y distribuciones» se unen
        # en dos (25-11-2026, la autora): lo que ha encontrado la app en los puntos
        # (decisiones, probabilidades y tipos) y en qué se fija el modelo (el peso de
        # cada variable y cómo cambian entre grupos).
        tab1, tab2, tab3, tab5, tab4 = st.tabs([
            i18n.pick('Mapa de los puntos', 'Points map'),
            i18n.pick('Revisar puntos', 'Review points'),
            i18n.pick('Resumen de los puntos', 'Points summary'),
            i18n.pick('Variables del modelo', 'Model variables'),
            i18n.pick('Descargas', 'Downloads')])

        with tab1:
            st.markdown('Mapa interactivo. Haz click en un marcador para ver sus detalles y '
                        'poder **quitarlo** del análisis. Al seleccionar filas en la pestaña '
                        '**Revisar puntos**, el mapa se centra en el **último punto seleccionado**. '
                        'La **leyenda de colores** está abajo a la izquierda.')
            lat_med = sum(r['lat'] for r in rows)/len(rows)
            lon_med = sum(r['lon'] for r in rows)/len(rows)
            st.caption('Capas activables (control ▤ arriba a la derecha): **Píxel S2 '
                       '(10 m)**, **ventana de muestreo 10 m** (S2 y radar) y **20 m** '
                       '(S2). Cambia a **Google Satélite** para más zoom.')
            # --- Añadir puntos pinchando en el mapa (p. ej. en un taller con quien
            # conoce el sitio). El punto entra como un dato de campo más.
            _anadir = st.toggle(
                i18n.pick('➕ Añadir puntos pinchando en el mapa',
                          '➕ Add points by clicking on the map'),
                key='modo_anadir',
                help=i18n.pick(
                    'Para quien conoce el sitio: pincha donde sepas si hay borreguil y '
                    'contesta debajo del mapa. El punto queda como un dato de campo más '
                    '(con `method` = local knowledge) y se guarda en las descargas.',
                    'For people who know the site: click where you know whether there is '
                    'a borreguil and answer below the map. The point becomes one more '
                    'field point (with `method` = local knowledge) and is saved in the '
                    'downloads.'))
            _msg_mapa = st.session_state.pop('_msg_mapa', None)
            if _msg_mapa:
                (st.success if _msg_mapa[0] == 'ok' else st.warning)(_msg_mapa[1])
            if _anadir:
                # cuántos puntos tiene el modelo para aprender (sin los dudosos)
                _n_e = sum(1 for t in (tabla_entrenamiento() or [])
                           if t.get('borreguil') in ('si', 'no')
                           and not bp._es_si(t.get('duda')))
                st.caption(i18n.pick(
                    'Pincha en el mapa y contesta **debajo** si ahí hay borreguil. '
                    'La app lee solo el punto nuevo (con Earth Engine, unos segundos) y '
                    'dice qué opinaba el modelo **antes** de saber tu respuesta.',
                    'Click on the map and answer **below** whether there is a borreguil '
                    'there. The app reads only the new point (with Earth Engine, a few seconds) '
                    'and tells what the model thought **before** knowing your answer.'))
                if _propio and _n_e:
                    st.success(i18n.pick(
                        f'Cada punto que añadas se **suma a los {_n_e} de los que aprendió '
                        'el modelo** y el modelo se reentrena con todos (unos segundos).',
                        f'Each point you add is **added to the {_n_e} the model learnt '
                        'from** and the model is retrained with all of them (a few '
                        'seconds).'))
                elif _n_e:
                    st.info(i18n.pick(
                        'Para que los puntos que añadas se sumen a los del modelo y lo '
                        'reentrenen, marca **«Sumar mis puntos a los del modelo y '
                        'reentrenarlo»** en el panel (Modelo de partida). Sin ella, solo '
                        'sirven para medir cuánto acierta.',
                        'For the points you add to be added to the model\'s and retrain '
                        'it, tick **"Add my points to the model\'s and retrain it"** in '
                        'the panel (Starting model). Without it, they only serve to '
                        'measure how well it does.'))
                # Quién lo dice va en «Quién revisa», encima de las pestañas.
                st.text_input(i18n.pick('Nota (opcional)', 'Note (optional)'),
                              key='mapa_nota')

            # El clic que trae el mapa (el componente guarda su último valor con su
            # clave): se lee antes de dibujarlo para marcar dónde se ha pinchado.
            _clic = None
            if _anadir:
                _lc = (st.session_state.get('mainmap') or {}).get('last_clicked') \
                    if isinstance(st.session_state.get('mainmap'), dict) else None
                if _lc and _lc.get('lat') is not None:
                    _c = (round(float(_lc['lat']), 7), round(float(_lc['lng']), 7))
                    if _c != st.session_state.get('_clic_usado'):
                        _clic = _c

            # Con el modo de añadir, el mapa no se recoloca en cada clic: solo se
            # centra en el último punto añadido cuando se vuelve a dibujar.
            _ult = st.session_state.get('_ultimo_anadido') if _anadir else None
            if _ult:
                m = _base_map(list(_ult), 16)
            else:
                m = _base_map([lat_med, lon_med], 12)
            _add_points(m, rows, highlight_id=(sel.get('ID') if sel else None))
            _add_pixel_layers(m, rows, show=False)
            _add_legend(m, rows)
            folium.LayerControl(collapsed=False).add_to(m)
            _g_clic = folium.FeatureGroup(name=i18n.pick('Donde has pinchado',
                                                         'Where you clicked'))
            if _clic:
                folium.CircleMarker(list(_clic), radius=11, color='#FF1744', weight=3,
                                    fill=False).add_to(_g_clic)
                folium.CircleMarker(list(_clic), radius=2, color='#FF1744',
                                    fill=True, fill_opacity=1).add_to(_g_clic)
            if _anadir:
                map_state = st_folium(m, width=None, height=600, feature_group_to_add=_g_clic,
                                      returned_objects=['last_object_clicked', 'last_clicked'],
                                      key='mainmap')
            elif sel:
                map_state = st_folium(m, width=None, height=600,
                                      returned_objects=['last_object_clicked'],
                                      center=[sel['lat'], sel['lon']], zoom=18, key='mainmap')
            else:
                map_state = st_folium(m, width=None, height=600,
                                      returned_objects=['last_object_clicked'], key='mainmap')

            if _anadir:
                # Manda lo que devuelve el mapa al dibujarlo: es su último clic (al
                # empezar la pasada, la sesión podía tener aún el anterior).
                _lc = (map_state or {}).get('last_clicked') if isinstance(map_state, dict) else None
                if _lc and _lc.get('lat') is not None:
                    _c = (round(float(_lc['lat']), 7), round(float(_lc['lng']), 7))
                    _clic = None if _c == st.session_state.get('_clic_usado') else _c
                _panel_anadir(_clic)

            # Quitar un punto pinchando directamente en su marcador (no con el modo de
            # añadir: ahí pinchar es para añadir, y lo añadido se deshace con su botón)
            clicked = (map_state or {}).get('last_object_clicked') \
                if isinstance(map_state, dict) and not _anadir else None
            if clicked and clicked.get('lat') is not None:
                clat, clon = clicked['lat'], clicked['lng']
                ni, nd = None, 1e18
                for i, r in enumerate(rows):
                    d = (r['lat'] - clat)**2 + (r['lon'] - clon)**2
                    if d < nd:
                        nd, ni = d, i
                # last_object_clicked devuelve las coords exactas del marcador → match casi
                # perfecto; toleramos ~50 m por redondeos.
                if ni is not None and nd < (0.00045)**2:
                    rr = rows[ni]
                    _rfp = rr.get('rf_proba')
                    _rftxt = f'{_rfp*100:.0f}%' if isinstance(_rfp, (int, float)) and not _m.isnan(_rfp) else '—'
                    ci, cb = st.columns([3, 2])
                    with ci:
                        st.markdown(i18n.pick('📍 Marcador seleccionado: ',
                                              '📍 Selected marker: ')
                                    + f"**{rr.get('ID','?')}** — "
                                    f"{i18n.tv_decision(rr.get('decision') or '—')} "
                                    f"(RF {_rftxt})")
                    with cb:
                        if st.button('🗑 Quitar este punto del análisis', key='remove_map_pt'):
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

            _resumen_anadidos(con_deshacer=_anadir)

        with tab2:
            st.caption(i18n.pick(
                'Selecciona una o varias filas (casilla izquierda). Debajo sale el mapa del '
                '**último punto seleccionado** y, a su lado, su revisión en dos pasos: '
                '**¿es borreguil?** y, si lo es, **su tipo**, con un solo botón para '
                'guardarlo todo (con varias filas, se aplica a todas). Si has marcado «Sumar '
                'mis puntos a los del modelo y reentrenarlo» (panel de la izquierda, en '
                '«Modelo de partida»), al cambiar el sí o el no se reentrena el modelo. '
                'Debajo de la tabla se explica qué variables salen y por qué.',
                'Select one or several rows (left checkbox). Below appears the map of the '
                '**last selected point** and, next to it, its review in two steps: **is it '
                'a borreguil?** and, if it is, **its type**, with a single button to save '
                'it all (with several rows, it applies to all of them). If you ticked "Add '
                'my points to the model\'s and retrain it" (left panel, under "Modelo de '
                'partida"), the model is retrained when the yes or no changes. Below the '
                'table it says which variables appear and why.'))
            _aviso_rev = st.session_state.pop('_rev_msg', None)
            if _aviso_rev:
                st.success(_aviso_rev)
            _rs = bp.resumen_revision(rows, pp.is_borreguil_decision)
            if _rs['n_borreguil']:
                _pn = _rs['por_nivel']
                _rn = _rs['revisados_por_nivel']
                st.markdown(i18n.pick(
                    f"🌿 **Tipo revisado: {_rs['n_revisados']} de {_rs['n_borreguil']}** "
                    f"borreguiles (ambiente {_rn['ambiente']} · humedad {_rn['humedad']} · "
                    f"pureza {_rn['pureza']}) · corregidos respecto a la propuesta de la "
                    f"app: **{_rs['n_corregidos']}** (ambiente {_pn['ambiente']} · humedad "
                    f"{_pn['humedad']} · pureza {_pn['pureza']})",
                    f"🌿 **Type reviewed: {_rs['n_revisados']} of {_rs['n_borreguil']}** "
                    f"borreguiles (environment {_rn['ambiente']} · moisture "
                    f"{_rn['humedad']} · purity {_rn['pureza']}) · corrected against the "
                    f"app's proposal: **{_rs['n_corregidos']}** (environment "
                    f"{_pn['ambiente']} · moisture {_pn['humedad']} · purity "
                    f"{_pn['pureza']})"))
                st.caption(i18n.pick(
                    'Los tipos revisados son las muestras del mapa de tipos: pestaña '
                    '**🌿 Mapa Modelo RF**, arriba del todo.',
                    'The reviewed types are the samples for the type map: **🌿 RF model '
                    'map** tab, at the very top.'))
                if _rs['n_revisados'] >= 10 and _rs['n_corregidos'] == 0:
                    st.caption(i18n.pick(
                        'No se ha corregido ninguno: o la propuesta acierta siempre o se '
                        'está aceptando sin mirar la ortofoto. Un tipo aceptado sin mirar '
                        'sigue siendo la regla de la app.',
                        "None has been corrected: either the proposal is always right or "
                        "it is being accepted without looking at the orthophoto. A type "
                        "accepted without looking is still the app's rule."))
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
            cols_show = (['ID', 'source', 'cuartil', 'origin', 'Borreguil', 'decision',
                          'rf_proba', 'ambiente', 'humedad', 'pureza', 'tipo_revisado']
                         + _vars_tab + ['dist_cauce_m', 'lat', 'lon'])
            df = pd.DataFrame(rows)
            cols_show = [c for c in cols_show if c in df.columns]
            # Las columnas opcionales que están vacías en todos los puntos (p. ej.
            # la cuenca, si el fichero no la trae) no se muestran: solo ocupaban
            # ancho y empujaban fuera de la pantalla la decisión y el tipo.
            _opcionales = ('cuenca_id', 'source', 'cuartil', 'origin', 'Borreguil',
                           'dist_cauce_m')
            cols_show = [c for c in cols_show if c not in _opcionales
                         or any(bp._txt(v) != '' for v in df[c])]
            # Los valores del pipeline son españoles; se traducen SOLO para
            # mostrarlos (df_show), sin tocar `rows`, del que depende la lógica.
            df_show = df[cols_show].copy()
            # Las variables que llegan como texto (de un CSV, p. ej. «0.10313319351…»),
            # como números: si no, la tabla las enseñaba con todos sus decimales en vez
            # de con 3 (25-11-2026). Solo si todo lo que hay en la columna es un número.
            for _c in df_show.columns:
                if _c == 'ID' or pd.api.types.is_numeric_dtype(df_show[_c]):
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
            if 'source' in df_show.columns:
                # «truth» era un nombre interno: los puntos de campo que no coincidían
                # con ninguno de los tuyos (25-11-2026)
                df_show['source'] = df_show['source'].map(
                    lambda v: {'truth': i18n.pick('campo', 'field'),
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
            if 'ID' in df_show.columns:
                _ids_t = [str(x) for x in df_show['ID'].tolist()]
                _cortos_t = bp.nombres_de_punto(_ids_t, punto='')
                df_show['ID'] = bp.abreviar_ids(_cortos_t)
                _cfg_tabla['ID'] = st.column_config.Column(pinned=True, help=(i18n.pick(
                    'El número del punto: el nombre de la capa de la que viene se ha '
                    'quitado para que se lea mejor. El nombre completo sale en las '
                    'descargas.',
                    'The number of the point: the name of the layer it comes from has '
                    'been removed so that it reads better. The full name is in the '
                    'downloads.') if _cortos_t != _ids_t else i18n.pick(
                    'Los nombres largos se muestran abreviados («…»). El nombre '
                    'completo aparece al seleccionar el punto y en las descargas.',
                    'Long names are shown abbreviated ("…"). The full name appears '
                    'when you select the point and in the downloads.')))
            # Cabeceras legibles («Decisión», «Altitud (m)»…), no los nombres internos
            df_show = df_show.rename(columns={c: _nombre_columna(c) for c in df_show.columns})
            # Los números, con 3 decimales; las coordenadas con 5 (con 3 se moverían
            # unos 100 m). Solo cambia lo que se ve: `rows` conserva el valor completo.
            _coords = {_nombre_columna('lat'), _nombre_columna('lon')}
            # Al pasar el ratón por el nombre de una de las 3 variables que más pesan,
            # se dice por qué está en la tabla y cuánto pesa.
            _peso_tab = {_nombre_columna(k['feat']): k['peso'] for k in _clave_tab}
            for _c in df_show.columns:
                if _c in _peso_tab:
                    _p = _peso_tab[_c]
                    _p_es = '' if _p is None else f' ({100 * _p:.0f} % del peso)'
                    _p_en = '' if _p is None else f' ({100 * _p:.0f} % of the weight)'
                    _cfg_tabla[_c] = st.column_config.NumberColumn(help=i18n.pick(
                        f'Una de las 3 variables que más pesan en el modelo{_p_es}.',
                        f'One of the 3 variables that weigh most in the model{_p_en}.'))
                elif _c not in _cfg_tabla and pd.api.types.is_float_dtype(df_show[_c]):
                    _cfg_tabla[_c] = st.column_config.NumberColumn()
            # «Resultado» (antes «Decisión»), al pasar el ratón: si el punto está revisado,
            # manda lo revisado; si no,
            # la probabilidad (25-11-2026, la autora: «¿por qué pone decisión borreguil
            # verificado?» → «Sí, explícalo»). Con los cortes de bp.decide().
            _col_dec = _nombre_columna('decision')
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
            # «que explique cada campo de la tabla al pasar el ratón por encima»). Las 3
            # variables que más pesan ya llevan la suya (arriba).
            _ayudas = {
                'source': (
                    'De dónde viene el punto. Vacío: de tu fichero «Tus puntos». «campo»: de '
                    'tu fichero de puntos de campo, en un sitio donde no había ninguno de tus '
                    'puntos. «mapa»: lo añadiste pinchando en el mapa.',
                    'Where the point comes from. Empty: from your "Your points" file. '
                    '"field": from your field points file, at a place where none of your '
                    'points was. "map": you added it by clicking on the map.'),
                'cuartil': (
                    'Si el punto se lanzó por cuartiles de una capa, el cuartil del polígono '
                    'en el que cayó: 1 es el primero (el de los valores más altos).',
                    'If the point was placed by quartiles of a layer, the quartile of the '
                    'polygon it fell in: 1 is the first (the one with the highest values).'),
                'origin': (
                    'Si se revisó en la app. «verificado»: una persona lo revisó aquí o lo '
                    'añadió en el mapa. «dudoso»: lo marcó como dudoso. Vacío: no se ha '
                    'revisado en la app; si sale revisado, venía así en tu fichero.',
                    'Whether it was reviewed in the app. "verificado": a person reviewed it '
                    'here or added it on the map. "dudoso": marked as doubtful. Empty: not '
                    'reviewed in the app; if it shows as reviewed, it came so in your file.'),
                'Borreguil': (
                    'Lo que ha comprobado una persona: «sí», es borreguil; «no», no lo es. '
                    'Viene de tu fichero (columna presence o Borreguil), de tu revisión en '
                    'esta pestaña o de un punto añadido en el mapa. Vacío: sin revisar, y '
                    'entonces decide el modelo.',
                    'What a person has checked: "yes", it is a borreguil; "no", it is not. '
                    'It comes from your file (presence or Borreguil column), from your review '
                    'in this tab or from a point added on the map. Empty: not reviewed, so '
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
                _c = _nombre_columna(_k)
                if _c not in df_show.columns or _c in _peso_tab:
                    continue
                _ayuda = i18n.pick(_a_es, _a_en)
                if pd.api.types.is_numeric_dtype(df_show[_c]):
                    _cfg_tabla[_c] = st.column_config.NumberColumn(help=_ayuda)
                else:
                    _cfg_tabla[_c] = st.column_config.TextColumn(help=_ayuda)
            event = st.dataframe(_con_coma(df_show, decimales={c: 5 for c in _coords}),
                                 hide_index=True, width='stretch',
                                 on_select='rerun', selection_mode='multi-row',
                                 column_config=_cfg_tabla, key='results_table')
            # Por qué salen esas variables, en una frase junto a la tabla (25-11-2026, la
            # autora: «que eso se explique brevemente cerca de la tabla»). El peso, sin
            # decimales, como al pasar el ratón por la columna.
            _lista_tab = [k for k in _clave_tab
                          if _nombre_columna(k['feat']) in df_show.columns]
            if _lista_tab:
                def _con_peso(k):
                    if k['peso'] is None:
                        return _nombre_columna(k['feat'])
                    _np = k['parecidas']
                    _par = ('' if not _np else
                            i18n.pick(f", con {_np} parecida{'' if _np == 1 else 's'}",
                                      f", with {_np} similar one{'' if _np == 1 else 's'}"))
                    return f"{_nombre_columna(k['feat'])} ({100 * k['peso']:.0f} %{_par})"
                _nombres_tab = [_con_peso(k) for k in _lista_tab]
                _y = i18n.pick(' y ', ' and ')
                _txt_vars = (_nombres_tab[0] if len(_nombres_tab) == 1 else
                             ', '.join(_nombres_tab[:-1]) + _y + _nombres_tab[-1])
                _con_pesos = all(k['peso'] is not None for k in _lista_tab)
                _cir_tab = any(str(k['feat']).startswith('cir_') for k in _lista_tab)
                st.caption(i18n.pick(
                    '**Por qué salen estas variables:** '
                    + (f'son las {len(_lista_tab)} que más pesan en el modelo que usas, una '
                       'por cada grupo de variables que dicen casi lo mismo (el peso es el del '
                       f'grupo): {_txt_vars}. ' if _con_pesos else
                       f'{_txt_vars}: el modelo no trae el peso de sus variables, así que '
                       'salen las que más separan los borreguiles en Sierra Nevada. ')
                    + 'La pendiente va siempre, como referencia del terreno. Si cambias de '
                    'modelo, cambian.'
                    + (' Las CIR salen de la ortofoto en falso color infrarrojo a 0,25 m, de '
                       'un recorte de 50 × 50 m alrededor de cada punto.' if _cir_tab else '')
                    + ' El modelo del mapa (🌿 Mapa Modelo RF) usa otras: solo las que '
                    'existen como mapa de toda la zona.',
                    '**Why these variables:** '
                    + (f'they are the {len(_lista_tab)} that weigh most in the model in use, '
                       'one for each group of variables that say almost the same thing (the '
                       f'weight is the group\'s): {_txt_vars}. ' if _con_pesos else
                       f'{_txt_vars}: the model does not carry the weight of its variables, '
                       'so these are the ones that best separate borreguiles in Sierra '
                       'Nevada. ')
                    + 'The slope is always there, as a reference for the terrain. If you '
                    'change the model, they change.'
                    + (' The CIR ones come from the false-colour infrared orthophoto at '
                       '0.25 m, from a 50 × 50 m clip around each point.' if _cir_tab else '')
                    + ' The map model (🌿 RF model map) uses others: only those that exist '
                    'as a map of the whole area.'))
            sel_rows = []
            if event is not None and getattr(event, 'selection', None):
                _selo = event.selection
                sel_rows = (_selo.get('rows') if isinstance(_selo, dict)
                            else getattr(_selo, 'rows', [])) or []

            # --- Revisar los puntos seleccionados: una sola revisión en dos pasos -----
            # (25-11-2026, la autora: «¿no te parece que está muy desconectado borreguil sí
            # o no con el tipo?» → «haz este cambio que me propones»). 1 · ¿Es borreguil?
            # (sí, no o no estoy seguro) y la certidumbre; 2 · si es sí, el tipo, con la
            # propuesta de la app puesta. Un solo botón guarda todo, y el formulario va al
            # lado del mapa del punto, para mirar y decidir a la vez. Antes: botones de
            # verificar arriba y, aparte, «Revisar el tipo» con su propio botón.
            focus_idx = st.session_state.get('sel_idx')
            _sel_ok = [i for i in sel_rows if i < len(rows)]
            if _sel_ok:
                sel_ids = [rows[i].get('ID') for i in _sel_ok]
                _n = len(sel_ids)

                def _estado(r):
                    """'si', 'no' o 'duda' según lo revisado del punto (None si nada)."""
                    if bp._txt(r.get('Duda')).lower() == 'si':
                        return 'duda'
                    b = bp._txt(r.get('Borreguil')).lower()
                    return b if b in ('si', 'no') else None

                _ya = sum(1 for i in _sel_ok if _estado(rows[i]) in ('si', 'no'))
                _ya_txt = ('' if not _ya else
                           i18n.pick(' · ya verificado', ' · already verified') if _n == 1 else
                           i18n.pick(f" · {_ya} ya verificado{'s' if _ya > 1 else ''}",
                                     f' · {_ya} already verified'))
                _espera = (i18n.pick('Guardando y reentrenando…', 'Saving and retraining…')
                           if _propio else i18n.pick('Guardando…', 'Saving…'))
                _reentrenado = i18n.pick(' Modelo reentrenado sumando tus puntos.',
                                         ' Model retrained adding your points.')
                _nombres_sel = bp.nombres_de_punto([r.get('ID') for r in rows],
                                                   i18n.pick('Punto', 'Point'))
                st.markdown(i18n.pick(
                    f"**{_n} punto{'' if _n == 1 else 's'} seleccionado"
                    f"{'' if _n == 1 else 's'}**{_ya_txt}"
                    + ('' if _n == 1 else ': lo que guardes se aplica a todos.'),
                    f"**{_n} point{'' if _n == 1 else 's'} selected**{_ya_txt}"
                    + ('' if _n == 1 else ': what you save applies to all of them.')))
                idx = (focus_idx if focus_idx is not None and focus_idx < len(rows)
                       else _sel_ok[-1])
                sel = st.session_state.get('sel_point') or dict(rows[idx])
                colL, colR = st.columns([3, 2])
                with colL:
                    # Mapa para MIRAR el punto (y revisar su tipo): ortofoto de fondo,
                    # el marcador sin relleno y, en amarillo, el píxel Sentinel-2 de
                    # 10 m al que se refiere la pureza.
                    fm = _base_map([sel['lat'], sel['lon']], 17, ortofoto=True)
                    _add_points(fm, rows, highlight_id=sel.get('ID'), hueco=True)
                    try:
                        folium.Polygon(bp.pixel_footprints([sel])[0]['pixel10'],
                                       color='#FFD600', weight=2, fill=False,
                                       tooltip=i18n.pick('Píxel Sentinel-2 (10 m)',
                                                         'Sentinel-2 pixel (10 m)')
                                       ).add_to(fm)
                    except Exception:
                        pass                      # sin la huella el mapa sigue sirviendo
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
                    if _cerca is not None:
                        if _cerca:
                            _tmp = [{'lat': sel['lat'], 'lon': sel['lon']}]
                            bp.attach_cauce_distance(_tmp, _cerca)
                            _dist_osm = _tmp[0].get('dist_cauce_m')
                        _g_osm = folium.FeatureGroup(
                            name=i18n.pick('〰 Ríos y arroyos (OpenStreetMap)',
                                           '〰 Rivers and streams (OpenStreetMap)'),
                            show=True)
                        for _l in _cerca:
                            for _p in getattr(_l, 'geoms', [_l]):
                                folium.PolyLine(
                                    [(y, x) for x, y, *_ in _p.coords], color='#1565C0',
                                    weight=3, opacity=0.9,
                                    tooltip=i18n.pick('Río o arroyo (OpenStreetMap)',
                                                      'River or stream (OpenStreetMap)')
                                ).add_to(_g_osm)
                        _g_osm.add_to(fm)
                    folium.LayerControl(collapsed=True).add_to(fm)
                    st_folium(fm, width=None, height=420, returned_objects=[],
                              center=[sel['lat'], sel['lon']], zoom=17, key='focusmap')
                    st.caption(i18n.pick(
                        'Ortofoto de fondo; el recuadro amarillo es el píxel de 10 m del '
                        'punto, y en azul van los ríos y arroyos de OpenStreetMap. Con el '
                        'control de capas (arriba a la derecha) puedes cambiar a Google '
                        'Satélite, que admite más zoom, o a Relieve LiDAR, que enseña la '
                        'forma del terreno (regueros y cauces pequeños: arroyo o ladera), '
                        'y encender las curvas de nivel o la ventana de unos 50 m que '
                        'promedia el detector de puntos.',
                        'Orthophoto background; the yellow square is the 10 m pixel of '
                        'the point, and the OpenStreetMap rivers and streams are in blue. '
                        'Use the layer control (top right) to switch to Google Satellite, '
                        'which allows more zoom, or to LiDAR relief, which shows the shape '
                        'of the terrain (gullies and small streams: stream or slope), and '
                        'to turn on the contour lines or the window of about 50 m that the '
                        'point detector averages.'))
                    if _dist_osm is not None and _dist_osm == _dist_osm:
                        st.caption(i18n.pick(
                            'El río o arroyo de OpenStreetMap más cercano está a '
                            f'**{_dist_osm:.0f} m** (para proponer «arroyo», 30 m o menos). '
                            'Si ves un cauce que OpenStreetMap no tiene, corrige el '
                            'ambiente.',
                            'The nearest OpenStreetMap river or stream is '
                            f'**{_dist_osm:.0f} m** away (to propose "stream", 30 m or '
                            'less). If you see a stream OpenStreetMap does not have, '
                            'correct the environment.'))
                    elif _cerca is not None:
                        st.caption(i18n.pick(
                            'No hay ningún río ni arroyo de OpenStreetMap a menos de unos '
                            '3 km del punto.',
                            'There is no OpenStreetMap river or stream within about 3 km '
                            'of the point.'))
                with colR:
                    # El punto que se ve en el mapa
                    st.markdown(
                        "<h4 style='margin:0 0 .3rem 0;padding:0;font-size:1.1rem;"
                        "line-height:1.3;overflow-wrap:anywhere;'>"
                        f"{bp.nombre_html(_nombres_sel[idx])}</h4>",
                        unsafe_allow_html=True)
                    # Lo que dice la app del punto, en grande como antes de unir la
                    # revisión (25-11-2026, la autora): la decisión, el tipo y la
                    # probabilidad como cifra destacada.
                    st.markdown(f"**{i18n.tv_decision(sel.get('decision') or '—')}**")
                    if sel.get('tipo_borreguil'):
                        st.markdown(
                            f"🌿 {i18n.tv(sel['tipo_borreguil'])} — "
                            + (i18n.pick('**revisado**', '**reviewed**')
                               if bp.tipo_revisado(sel)
                               else i18n.pick('propuesta de la app, sin revisar',
                                              "app's proposal, not reviewed")))
                    rfp = sel.get('rf_proba')
                    if isinstance(rfp, (int, float)) and not _m.isnan(rfp):
                        st.metric(i18n.pick('Probabilidad RF', 'RF probability'),
                                  f'{rfp * 100:.0f} %')
                    import hashlib as _hl
                    # La clave de los controles depende de QUÉ puntos hay seleccionados
                    # (por ID: al quitar uno las posiciones se corren) y de cuántas veces
                    # se ha guardado o deshecho: arrancan con lo vigente de la selección,
                    # nunca con lo elegido para otro punto.
                    _firma = _hl.md5('|'.join(
                        [str(x) for x in sel_ids] + [str(st.session_state.get('_rev_n', 0))]
                    ).encode()).hexdigest()[:10]
                    _estados = {_estado(rows[i]) for i in _sel_ok}
                    _es_ini = next(iter(_estados)) if len(_estados) == 1 else None
                    _ES_TXT = {'si': i18n.pick('Sí', 'Yes'), 'no': 'No',
                               'duda': i18n.pick('No estoy seguro', 'Not sure')}
                    _es = st.radio(
                        i18n.pick('1 · ¿Es borreguil?', '1 · Is it a borreguil?'),
                        list(_ES_TXT), horizontal=True, format_func=_ES_TXT.get,
                        index=list(_ES_TXT).index(_es_ini) if _es_ini else None,
                        key=f'rev_es_{_firma}',
                        help=i18n.pick(
                            'Míralo en la ortofoto (o con tus datos de campo), **en todo el '
                            'recuadro amarillo** y no solo en el punto: el mapa del Random '
                            'Forest aprende de ese píxel de 10 m, y el detector de puntos '
                            'promedia unos 50 m alrededor. **Sí** o **No** lo dejan '
                            'verificado; **No estoy seguro** lo deja como dudoso: no '
                            'entrena ni cuenta en el acierto, y no tiene tipo.',
                            'Look at it on the orthophoto (or with your field data), **in the '
                            'whole yellow square** and not only at the point: the Random '
                            'Forest map learns from that 10 m pixel, and the point detector '
                            'averages about 50 m around it. **Yes** or **No** leave it '
                            'verified; **Not sure** leaves it as doubtful: it does not train '
                            'or count in the accuracy, and has no type.'))
                    # (25-11-2026, la autora: «habría que indicar en algún sitio de la
                    # revisión que deben mirar lo que hay dentro del píxel, no solamente
                    # el punto»)
                    st.caption(i18n.pick(
                        '👁 Decide por lo que hay **dentro del recuadro amarillo** (el píxel '
                        'de 10 m), no solo por el punto. Si el píxel es casi todo roca o '
                        'agua, dilo en la pureza o marca **No estoy seguro**.',
                        '👁 Decide by what is **inside the yellow square** (the 10 m pixel), '
                        'not only by the point. If the pixel is almost all rock or water, '
                        'say so in the purity or choose **Not sure**.'))
                    st.radio(i18n.pick('Certidumbre (opcional)', 'Certainty (optional)'),
                             list(bp.CERTIDUMBRES), index=None, horizontal=True,
                             format_func=_CERT_TXT.get, key='rev_certidumbre',
                             help=_CERT_AYUDA, disabled=_es == 'duda')
                    _nom, _per = _quien()
                    _cert = '' if _es == 'duda' else (st.session_state.get('rev_certidumbre')
                                                      or '')
                    st.markdown(i18n.pick('**2 · Tipo**', '**2 · Type**')
                                + (i18n.pick(
                                    ' · elige «Sí» para revisarlo',
                                    ' · choose "Yes" to review it') if _es is None else
                                   i18n.pick(' · un punto que no es borreguil no tiene tipo',
                                             ' · a point that is not a borreguil has no '
                                             'type') if _es == 'no' else
                                   i18n.pick(' · un punto dudoso no tiene tipo',
                                             ' · a doubtful point has no type')
                                   if _es == 'duda' else ''))
                    _SIN = '(sin cambio)'
                    _uno = rows[_sel_ok[0]] if _n == 1 else None
                    for _niv, _etq in (('ambiente', 'Ambiente'), ('humedad', 'Humedad'),
                                       ('pureza', 'Pureza')):
                        _ops = [_SIN] + list(bp.NIVELES_TIPO[_niv])
                        # Con un punto: lo revisado o, si no, la propuesta de la app, para
                        # aceptarla o corregirla. Con varios, «sin cambio»: cada uno se
                        # queda con lo suyo.
                        _rev = (_uno.get(_niv) if _uno and _niv in bp.niveles_revisados(_uno)
                                else None)
                        _prop = bp._txt(_uno.get(_niv + '_regla')) if _uno else ''
                        _ini = _rev if _rev in _ops else (_prop if _prop in _ops else _SIN)
                        st.selectbox(
                            i18n.tr(_etq), _ops, index=_ops.index(_ini),
                            key=f'rev_{_niv}_{_firma}', disabled=_es != 'si',
                            format_func=lambda o, p=_prop: (
                                i18n.pick('No lo he podido ver (sin cambio)',
                                          "I couldn't see it (no change)") if o == _SIN
                                else i18n.tv(o) + (i18n.pick(' (propuesta de la app)',
                                                             " (app's proposal)")
                                                   if o == p else '')))
                    st.caption(i18n.pick(
                        'Decide con la ortofoto o con datos de campo, no por lo que propone '
                        'la app. **Ambiente**: dónde está el prado (junto a un arroyo, junto '
                        'a una laguna o en ladera). **Humedad**: aplica siempre el mismo '
                        'criterio. **Pureza**: «mixto-agua» si hay agua dentro del píxel de '
                        '10 m (el recuadro amarillo), «mixto-roca» si hay roca o suelo '
                        'desnudo, «puro» si todo es prado. Cada nivel cuenta por separado: lo '
                        'que dejes en «No lo he podido ver» no cuenta como revisado.',
                        'Decide with the orthophoto or with field data, not by what the app '
                        'proposes. **Environment**: where the meadow is (by a stream, by a '
                        'lake or on a slope). **Moisture**: always apply the same criterion. '
                        '**Purity**: "mixed-water" if there is water inside the 10 m pixel '
                        '(the yellow square), "mixed-rock" if there is rock or bare soil, '
                        '"pure" if it is all meadow. Each level counts separately: what you '
                        'leave at "I couldn\'t see it" does not count as reviewed.'))
                    _snap = st.session_state.get('_rev_deshacer')
                    # Si al guardar se reentrena o no, y dónde se elige: la casilla está
                    # en el panel de la izquierda y desde aquí no se veía (25-11-2026, la
                    # autora: «no sé dónde se selecciona eso»).
                    if st.session_state.get('entrenar_propio'):
                        st.caption(i18n.pick(
                            '🔁 **«Sumar mis puntos a los del modelo y reentrenarlo»** está '
                            'marcada: si cambias el sí o el no, al guardar el modelo se '
                            'reentrena con tus puntos (unos segundos).',
                            '🔁 **"Add my points to the model\'s and retrain it"** is ticked: '
                            'if you change the yes or no, saving retrains the model with your '
                            'points (a few seconds).'))
                    else:
                        st.caption(i18n.pick(
                            'Lo que guardes queda en el punto y sirve para medir cuánto '
                            'acierta el modelo, pero no lo reentrena. Para eso, marca '
                            '**«Sumar mis puntos a los del modelo y reentrenarlo»** en el '
                            'panel de la izquierda, en «Modelo de partida».',
                            'What you save stays in the point and is used to measure how '
                            'well the model does, but it does not retrain it. For that, tick '
                            '**"Add my points to the model\'s and retrain it"** in the left '
                            'panel, under "Modelo de partida" (Starting model).'))
                    _guardar = st.button(
                        i18n.pick('💾 Guardar la revisión', '💾 Save the review'),
                        type='primary', key='rev_guardar', width='stretch',
                        help=i18n.pick(
                            'Guarda a la vez si es borreguil, la certidumbre, quién revisa y '
                            'el tipo. Si has marcado «Sumar mis puntos a los del modelo y '
                            'reentrenarlo» (panel de la izquierda, en «Modelo de partida») y '
                            'cambia el sí o el no, el modelo se reentrena.',
                            'Saves at once whether it is a borreguil, the certainty, who '
                            'reviews and the type. If you ticked "Add my points to the '
                            "model's and retrain it\" (left panel, under \"Modelo de "
                            'partida") and the yes or no changes, the model is retrained.'))
                    _b1, _b2 = st.columns(2)
                    with _b1:
                        _deshacer = st.button(
                            i18n.pick('↩ Deshacer', '↩ Undo'), key='rev_deshacer',
                            width='stretch', disabled=not _snap,
                            help=i18n.pick('Deshace la última revisión guardada.',
                                           'Undoes the last saved review.'))
                    with _b2:
                        if st.button('🗑 Quitar del análisis', key='remove_pts',
                                     help='Elimina los puntos seleccionados del mapa, la tabla '
                                          'y las descargas (no reentrena el modelo).'):
                            drop = set(sel_rows)
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
                            # Para «↩ Deshacer»: cómo estaba todo antes de guardar
                            st.session_state['_rev_deshacer'] = {
                                'rows': [dict(r) for r in rows],
                                'base': list(base_truth_ids)}
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
                        st.session_state.sel_point = None
                        st.session_state['_rev_n'] = st.session_state.get('_rev_n', 0) + 1
                        st.session_state['_rev_msg'] = i18n.pick(
                            'Revisión deshecha: los puntos vuelven a como estaban.',
                            'Review undone: the points are back to how they were.')
                        st.rerun()
                # Debajo del mapa y del formulario: los valores del punto y su ficha
                _n = _num                       # 3 decimales, con coma en español
                # Las 3 variables que más pesan en el modelo, una por grupo de
                # parecidas (antes eran NDVI, clorofila y EVI de fin de verano, que
                # dicen casi lo mismo: correlación 0,94–0,98).
                _rs_sel = load_ref_stats(MODEL_PATH)
                _mf = set((_rs_sel or {}).get('model_features') or []) or None
                _clave3 = pp.variables_clave((_rs_sel or {}).get('importances'), _mf, 3)
                _ya = {k['feat'] for k in _clave3}
                # (sin las que no tienen dato)
                _terreno = ' · '.join(
                    f"{pp.etiqueta(f)}: **{_n(sel.get(f))}**"
                    for f in ('elev_dem_m', 'slope_deg')
                    if f not in _ya and _n(sel.get(f)) != '—')
                _mas_peso = ' · '.join(
                    f"{pp.etiqueta(k['feat'])}: **{_n(sel.get(k['feat']))}**"
                    + ('' if k['peso'] is None else f" ({100 * k['peso']:.0f} %)")
                    for k in _clave3)
                st.markdown(
                    i18n.pick('Lo que más pesa en el modelo: ',
                              'What weighs most in the model: ') + _mas_peso + '\n\n'
                    + (_terreno + '\n\n' if _terreno else '')
                    # (sin el «Patrón»: era un código interno, BORREGUIL_LIKE…)
                    + f"Coords: {_num(sel.get('lat'), 5)}{i18n.pick('; ', ', ')}{_num(sel.get('lon'), 5)}")
                gmaps = f"https://www.google.com/maps/search/?api=1&query={sel.get('lat')},{sel.get('lon')}"
                st.markdown(f'[Abrir en Google Maps]({gmaps})')
                rs = load_ref_stats(MODEL_PATH)
                with st.expander(i18n.pick('📈 Este punto frente a los borreguiles verdaderos',
                                           '📈 This point vs the real borreguiles'),
                                 expanded=True):
                    render_point_profile(sel, rs)
            elif sel:
                if st.button('✕ Quitar selección'):
                    st.session_state.sel_point = None
                    st.rerun()

        with tab3:
            rf_vals = [r.get('rf_proba') for r in rows
                        if isinstance(r.get('rf_proba'), (int,float)) and not _m.isnan(r.get('rf_proba'))]
            if rf_vals:
                import altair as alt
                st.markdown(i18n.pick('#### Probabilidad de ser borreguil',
                                      '#### Probability of being a borreguil'))
                st.markdown(i18n.pick(
                    'Distribución de probabilidad Random Forest (nº de puntos por intervalo '
                    'de probabilidad):',
                    'Random Forest probability distribution (no. of points per probability '
                    'interval):'))
                dfh = pd.DataFrame({'rf_proba': rf_vals})
                # Histograma real: agrupa las probabilidades en bins de 0,05 y cuenta
                # cuántos puntos caen en cada uno (antes pintaba una barra por punto).
                graf_hist = alt.Chart(dfh).mark_bar(color='#2e7d32', opacity=0.85).encode(
                    x=alt.X('rf_proba:Q', bin=alt.Bin(extent=[0, 1], step=0.05),
                            title='Probabilidad RF'),
                    y=alt.Y('count():Q', title='Nº de puntos'),
                    tooltip=[alt.Tooltip('count():Q', title='puntos')])
                rules = alt.Chart(
                    pd.DataFrame({'umbral': [threshold, 0.7],
                                  'etiqueta': [i18n.pick('umbral ', 'threshold ')
                                               + _n3(threshold), _n3(0.7)]})
                ).mark_rule(color='#c62828', strokeDash=[4, 4]).encode(
                    x='umbral:Q',
                    tooltip=[alt.Tooltip('etiqueta:N', title='línea')])
                st.altair_chart((graf_hist + rules).properties(height=320),
                                width='stretch')
                _med = sorted(rf_vals)[len(rf_vals)//2]
                # «N ≥» cuenta con el umbral que se usa (la raya roja), no con un 0,5 fijo:
                # con Otsu o con la verdad-terreno el umbral puede ser otro.
                _n_borr = sum(1 for v in rf_vals if v >= threshold)
                _n_muy = sum(1 for v in rf_vals if v >= 0.7)
                st.markdown(i18n.pick('**Mediana**', '**Median**') + f': {_n3(_med)}  ·  '
                            + i18n.pick(f'**Borreguil** (≥ {_n3(threshold)}, tu umbral): {_n_borr}',
                                        f'**Borreguil** (≥ {_n3(threshold)}, your threshold): {_n_borr}')
                            + '  ·  '
                            + i18n.pick(f'**Muy probable** (≥ {_n3(0.7)}): {_n_muy}',
                                        f'**Very likely** (≥ {_n3(0.7)}): {_n_muy}')
                            + f'  ·  **Total**: {len(rf_vals)}')
                # 25-11-2026, la autora: «como se interpreta esto y para que es util» → la
                # explicación, abierta, debajo de la gráfica. «Si mueves el umbral 0,15» cuenta
                # los puntos que podrían cambiar de lado: los que están a menos de 0,15 de él.
                _margen = 0.15
                _n_cerca = sum(1 for v in rf_vals
                               if threshold - _margen <= v < threshold + _margen)
                with st.expander(i18n.pick('ℹ ¿Cómo se lee esta gráfica y para qué sirve?',
                                           'ℹ How to read this chart and what it is for'),
                                 expanded=True):
                    st.markdown(i18n.pick(
                        f'Cada barra cuenta los puntos cuya probabilidad de ser borreguil, '
                        f'según el Random Forest, cae en ese tramo de {_n3(0.05)}.\n\n'
                        f'- **A la izquierda** (cerca de 0) están los puntos que el modelo ve '
                        f'claramente como **no** borreguil; **a la derecha** (cerca de 1), los '
                        f'que ve claramente como borreguil; **en el centro**, los que le hacen '
                        f'dudar.\n'
                        f'- **Forma de U**, con el centro casi vacío: el modelo separa bien los '
                        f'dos grupos. **Una joroba en el centro**: duda en muchos puntos, por '
                        f'ejemplo porque la zona es distinta de aquella con la que se entrenó; '
                        f'conviene revisar puntos y reentrenar.\n'
                        f'- **Rayas rojas**: {_n3(threshold)} es el umbral que usas (por encima, '
                        f'el punto cuenta como borreguil; se elige en el panel de la izquierda). '
                        f'{_n3(0.7)} es una referencia fija: lo que pasa de ahí es *muy '
                        f'probable*.\n'
                        f'- **Para el umbral**: si cae en un valle con pocas barras, moverlo un '
                        f'poco apenas cambia nada; si cae donde hay muchas, un cambio pequeño '
                        f'cambia muchos puntos. El umbral automático por Otsu busca ese valle. '
                        f'Ahora, si lo mueves hasta {_n3(_margen)} arriba o abajo, cambian como '
                        f'mucho **{_n_cerca}** de tus {len(rf_vals)} puntos.\n'
                        f'- **Para revisar**: los puntos del centro son los primeros que conviene '
                        f'mirar en «Revisar puntos»; son los que más enseñan al modelo cuando lo '
                        f'reentrenas.\n'
                        f'- Si el modelo se entrenó con estos mismos puntos, la U sale más limpia '
                        f'de lo que saldría con puntos nuevos.',
                        f'Each bar counts the points whose probability of being borreguil, '
                        f'according to the Random Forest, falls in that {_n3(0.05)} band.\n\n'
                        f'- **On the left** (near 0) are the points the model clearly sees as '
                        f'**not** borreguil; **on the right** (near 1), the ones it clearly sees '
                        f'as borreguil; **in the middle**, the ones it is unsure about.\n'
                        f'- **A U shape**, with an almost empty middle: the model separates the '
                        f'two groups well. **A hump in the middle**: it is unsure about many '
                        f'points, for example because the area differs from the one it was '
                        f'trained on; review points and retrain.\n'
                        f'- **Red lines**: {_n3(threshold)} is the threshold in use (above it, '
                        f'a point counts as borreguil; it is chosen in the left panel). '
                        f'{_n3(0.7)} is a fixed reference: anything above it is *very '
                        f'likely*.\n'
                        f'- **For the threshold**: if it falls in a valley with few bars, moving '
                        f'it a little barely changes anything; if it falls where there are many, '
                        f'a small change moves many points. The automatic Otsu threshold looks '
                        f'for that valley. Right now, moving it up to {_n3(_margen)} up or down '
                        f'changes at most **{_n_cerca}** of your {len(rf_vals)} points.\n'
                        f'- **For reviewing**: the points in the middle are the first ones to '
                        f'check in «Review points»; they teach the model the most when you '
                        f'retrain it.\n'
                        f'- If the model was trained on these same points, the U looks cleaner '
                        f'than it would with new points.'))
            else:
                st.info('Sin probabilidades RF disponibles.')

            # Debajo de la probabilidad (25-11-2026, la autora: «esta quiero que salga
            # debajo de probabilidad de ser borreguil, y si se te ocurre otra manera de
            # mostrarla más bonita, mejor»): de más a menos borreguil, con el color de
            # cada decisión en el mapa de los puntos, su número y su porcentaje.
            st.divider()
            st.markdown(i18n.pick('#### Recuento de puntos por resultado',
                                  '#### Point count by result'))
            dc = pp.chart_decision_counts(rows)
            if dc is not None:
                st.altair_chart(dc, width='stretch')
                st.caption(i18n.pick(
                    'Cada color es el del punto en «Mapa de los puntos». De arriba abajo, '
                    'de más a menos borreguil.',
                    'Each colour is that of the point in «Points map». From top to '
                    'bottom, from most to least borreguil.'))

            st.divider()
            st.markdown(i18n.pick('#### 🌿 Tipos de borreguil', '#### 🌿 Borreguil types'))
            # Tipos de borreguil. Primero, cuántos hay de cada ambiente, humedad y pureza:
            # barras con su número, como el recuento por categoría de decisión (25-11-2026,
            # la autora: «esta me gustaría que estuviera para los tipos de borreguil, antes
            # había gráficos de barras con el n»). Salen siempre que haya borreguiles,
            # aunque falte un nivel (sin Earth Engine no hay pureza); la vista anidada de
            # debajo necesita los tres y, si falta uno, no sale.
            _tipos_ch, _n_tipos = pp.chart_type_counts(rows)
            hier_chart, n_hier = pp.chart_hierarchy(rows)
            if _tipos_ch:
                st.markdown(i18n.pick(
                    f'Cuántos de los {_n_tipos} borreguiles hay de cada **ambiente**, de '
                    f'cada **humedad** y de cada **pureza**. Cada categoría lleva su color, '
                    f'el mismo en toda la app.',
                    f'How many of the {_n_tipos} borreguiles there are of each '
                    f'**environment**, each **moisture** and each **purity**. Each category '
                    f'has its colour, the same throughout the app.'))
                # Una debajo de otra y a todo lo ancho, como el recuento por decisión: en
                # tres columnas no cabían los nombres ni las barras.
                for _tit, _ch in _tipos_ch:
                    st.markdown(f'**{_tit}**')
                    st.altair_chart(_ch, width='stretch')
                _rsd = bp.resumen_revision(rows, pp.is_borreguil_decision)
                st.caption(i18n.pick(
                    f"{_rsd['n_revisados']} de {_rsd['n_borreguil']} tienen el tipo revisado "
                    f"por una persona; en el resto se muestra la propuesta de la app.",
                    f"{_rsd['n_revisados']} of {_rsd['n_borreguil']} have their type "
                    f"reviewed by a person; the rest show the app's proposal."))
            if hier_chart is not None:
                st.markdown(i18n.pick(
                    f'**Los tres niveles juntos** (clasificación jerárquica de los {n_hier} '
                    f'detectados): **ambiente** → **humedad** → **pureza**. Cada columna es '
                    f'una humedad; cada fila, un ambiente; el color reparte la pureza.',
                    f'**The three levels together** (hierarchical classification of the '
                    f'{n_hier} detected): **environment** → **moisture** → **purity**. Each '
                    f'column is a moisture; each row, an environment; the colour splits the '
                    f'purity.'))
                st.altair_chart(hier_chart)
                hier_df = pp.hierarchy_table(rows)
                if hier_df is not None:
                    st.markdown('Recuento por **combinación completa** (ambiente · humedad · pureza):')
                    st.dataframe(hier_df, hide_index=True, width='stretch')
            if _tipos_ch or hier_chart is not None:
                st.caption(i18n.pick(
                    'Cómo se propone el tipo: el **ambiente**, por la cercanía a lagunas y a '
                    'los ríos y arroyos de OpenStreetMap (arroyo: a 30 m o menos); la '
                    '**humedad**, por el NDMI; la **pureza**, igual que en el mapa: cuánto '
                    'verde hay en el píxel de 10 m del punto, estimado con su NDVI de inicio '
                    'de verano (puro si llega al 75 %; si no, mixto-agua si el píxel es agua '
                    'o toca agua, y mixto-roca si no). La pureza se lee con Earth Engine: sin '
                    'él, se queda sin propuesta (—).',
                    'How the type is proposed: the **environment**, from the closeness to '
                    'lakes and to the OpenStreetMap rivers and streams (stream: 30 m or '
                    'less); the **moisture**, from the NDMI; the **purity**, as in the map: '
                    'how much green there is in the 10 m pixel of the point, estimated from '
                    'its early-summer NDVI (pure if it reaches 75 %; otherwise mixed-water if '
                    'the pixel is water or touches water, and mixed-rock if not). Purity is '
                    'read with Earth Engine: without it, there is no proposal (—).'))

        with tab4:
            st.markdown(i18n.pick(
                'Los ficheros de este análisis, con los últimos cambios (puntos '
                'revisados, añadidos o reentrenados):',
                'The files of this analysis, with the latest changes (points reviewed, '
                'added or retrained):'))
            for name, que in (
                ('Clasificacion_puntos.xlsx', i18n.pick(
                    'Excel para **mirar los resultados**: cada punto con su probabilidad, '
                    'su resultado y el tipo de borreguil.',
                    'Excel to **look at the results**: each point with its probability, '
                    'its result and the type of borreguil.')),
                ('puntos.geojson', i18n.pick(
                    'Los puntos con **todo lo revisado**: ábrelo en QGIS, o vuelve a '
                    'subirlo a la app otro día para seguir donde lo dejaste.',
                    'The points with **everything reviewed**: open it in QGIS, or upload '
                    'it to the app again another day to carry on where you left off.')),
                ('mapa.html', i18n.pick(
                    'El **mapa** de los puntos para abrirlo en el navegador, sin la app.',
                    'The **map** of the points to open in the browser, without the app.')),
                ('classification.csv', i18n.pick(
                    'Tabla con **todas las variables** de cada punto (para analizarlas en '
                    'R, Python o una hoja de cálculo).',
                    'Table with **all the variables** of each point (to analyse them in '
                    'R, Python or a spreadsheet).')),
                ('area_estudio.geojson', i18n.pick(
                    'El **área de estudio** que usaste, para abrirla en QGIS.',
                    'The **study area** you used, to open it in QGIS.')),
            ):
                p = work / name
                if p.exists():
                    _c1, _c2 = st.columns([1, 2], vertical_alignment='center')
                    with open(p, 'rb') as fh:
                        _c1.download_button(label=f'⬇ {name}',
                                            data=fh.read(),
                                            file_name=name,
                                            mime='application/octet-stream',
                                            key=f'dl_{name}')
                    _c2.markdown(que)

        with tab5:
            st.markdown(i18n.pick('#### El peso de cada variable',
                                  '#### The weight of each variable'))
            st.markdown(i18n.pick(
                'Las variables más decisivas del modelo Random Forest usado en esta '
                'estimación, de arriba (más peso) a abajo. A la derecha, el **rango típico '
                'de borreguil** (p25–p75) cuando el modelo trae valores de referencia.',
                'The most decisive variables of the Random Forest model used in this '
                'estimate, from top (most weight) to bottom. On the right, the **typical '
                'borreguil range** (p25–p75) when the model carries reference values.'))
            # Prioriza el modelo recién reentrenado (en sesión); si no, el de partida.
            _mb = st.session_state.get('model_bundle')
            rs_imp = None
            if _mb and _mb.get('importances'):
                rs_imp = dict(_mb.get('ref_stats') or {})
                rs_imp['importances'] = _mb['importances']
            if not (rs_imp and rs_imp.get('importances')):
                rs_imp = load_ref_stats(MODEL_PATH)
            imp_chart, _ = pp.chart_importance(rs_imp, topn=20) if rs_imp else (None, [])
            if imp_chart is not None:
                st.altair_chart(imp_chart, width='stretch')
            else:
                st.info(i18n.pick(
                    'Este modelo no trae el peso de sus variables, así que no se puede '
                    'mostrar.',
                    'This model does not carry the weight of its variables, so it cannot '
                    'be shown.'))

            st.divider()

            # ---- Las variables que más pesan en el modelo usado, vistas de cuatro
            # maneras y comparando borreguil con no borreguil, o los borreguiles por su
            # humedad o su ambiente con los no borreguiles (25-11-2026, la autora: «aquí deberían
            # mostrarse las que más peso tienen en el modelo y que lo indique», «podríamos
            # hacer un desplegable para que elijan distintos tipos de gráficos», «sí,
            # hazlo con comparar por tipo»). Antes: 9 variables fijas en cajas y una
            # dispersión fija de altitud y NDVI.
            _rs_box = load_ref_stats(MODEL_PATH)
            _imp_box = (_rs_box or {}).get('importances')
            _disp_box = set((_rs_box or {}).get('model_features') or []) or None
            st.markdown(i18n.pick('#### Las que más pesan, grupo a grupo',
                                  '#### The ones that weigh most, group by group'))
            _vistas = {'cajas': i18n.pick('Cajas', 'Boxes'),
                       'curvas': i18n.pick('Curvas de distribución', 'Distribution curves'),
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
                'ambiente': i18n.pick('Por ambiente (arroyo, laguna, ladera y no borreguil)',
                                      'By environment (stream, lake, slope and not a '
                                      'borreguil)')}
            _c1, _c2, _c3 = st.columns([1.1, 1.4, 0.9])
            _vista = _c1.selectbox(i18n.pick('Cómo verlo', 'How to see it'), list(_vistas),
                                   format_func=_vistas.get, key='dist_vista')
            _comparar = _c2.selectbox(i18n.pick('Comparar', 'Compare'), list(pp.COMPARAR),
                                      format_func=_comp_txt.get, key='dist_comparar')
            _grp, _ord, _rest = pp.grupos_puntos(rows, _comparar, None, i18n.tv)
            from collections import Counter as _Cnt
            _cnt = _Cnt(_grp)
            _cuantos = ' · '.join(f'{g}: {_cnt[g]}' for g, _ in _ord)
            if _cnt.get(None):
                _cuantos += i18n.pick(
                    f' (y {_cnt[None]} inciertos o dudosos, que no se dibujan)',
                    f' (and {_cnt[None]} uncertain or doubtful, not drawn)')
            if _vista != 'dos':
                _n_vars = _c3.selectbox(i18n.pick('Cuántas variables', 'How many variables'),
                                        [4, 8, 12], index=1, key='dist_n')
                _vars_box = pp.vars_boxplots(_imp_box, _disp_box, _n_vars)
                st.caption(i18n.pick(
                    (f'Las {len(_vars_box)} variables que más pesan en el modelo usado, de '
                     'más a menos: una por cada grupo de variables parecidas, con el peso '
                     'del grupo en el modelo junto al nombre. ' if _imp_box else '')
                    + f'Puntos: {_cuantos}.',
                    (f'The {len(_vars_box)} variables that weigh most in the model in use, '
                     'from most to least: one for each group of similar variables, with the '
                     "group's weight in the model next to the name. " if _imp_box else '')
                    + f'Points: {_cuantos}.'))
            if _vista == 'cajas':
                _ch, _ = pp.chart_boxplots(rows, _vars_box, comparar=_comparar,
                                           traduce=i18n.tv)
                _como = i18n.pick(
                    'Cada caja va del p25 al p75, con la mediana en blanco; los bigotes '
                    'llegan hasta los valores normales y los puntos sueltos son atípicos. '
                    'Al pasar el ratón salen las cifras.',
                    'Each box goes from p25 to p75, with the median in white; the whiskers '
                    'reach the normal values and the loose dots are outliers. Hover to see '
                    'the figures.')
            elif _vista == 'curvas':
                _ch = pp.chart_densidades(rows, _vars_box, comparar=_comparar,
                                          traduce=i18n.tv)
                _como = i18n.pick(
                    'Cada curva muestra cómo se reparten los valores de un grupo. Todas '
                    'encierran la misma área, así que se comparan formas, no cuántos '
                    'puntos hay. Si las curvas apenas se tocan, esa variable separa bien '
                    'los grupos; si se montan una encima de otra, poco. Los grupos con '
                    f'menos de {pp.MIN_GRUPO} puntos no se dibujan.',
                    'Each curve shows how the values of a group are spread. They all '
                    'enclose the same area, so shapes are compared, not how many points '
                    'there are. If the curves hardly touch, that variable separates the '
                    'groups well; if they overlap, it does not. Groups with fewer than '
                    f'{pp.MIN_GRUPO} points are not drawn.')
            elif _vista == 'diferencias':
                _ch = pp.chart_diferencias(rows, _vars_box, comparar=_comparar,
                                           traduce=i18n.tv)
                _como = i18n.pick(
                    'Cada punto dice cuánto se aparta la mediana del grupo de la de los no '
                    'borreguiles, en desviaciones típicas de todos los puntos: a la '
                    'derecha, más que los no borreguiles; a la izquierda, menos. Pasar de 1 '
                    '(o de −1) es una diferencia clara. Los grupos con menos de '
                    f'{pp.MIN_GRUPO} puntos no se dibujan.',
                    'Each dot says how far the median of the group is from that of the '
                    'points that are not a borreguil, in standard deviations of all the '
                    'points: to the right, more than those; to the left, less. Beyond 1 '
                    f'(or −1) is a clear difference. Groups with fewer than {pp.MIN_GRUPO} '
                    'points are not drawn.')
            else:
                _lv = pp.lista_variables(rows, _imp_box, todas=list(pp.FEATURE_LABELS))
                _et_lv = dict(_lv)
                _op_lv = [f for f, _ in _lv]
                _top2 = [k['feat'] for k in pp.variables_clave(_imp_box, _disp_box, 2)
                         if k['feat'] in _op_lv]
                _ix = _op_lv.index(_top2[0]) if _top2 else 0
                _iy = (_op_lv.index(_top2[1]) if len(_top2) > 1
                       else min(1, len(_op_lv) - 1))
                if len(_op_lv) < 2:
                    _ch = None
                else:
                    _d1, _d2 = st.columns(2)
                    _vx = _d1.selectbox(i18n.pick('Eje horizontal', 'Horizontal axis'),
                                        _op_lv, index=_ix, format_func=_et_lv.get,
                                        key='dist_x')
                    _vy = _d2.selectbox(i18n.pick('Eje vertical', 'Vertical axis'),
                                        _op_lv, index=_iy, format_func=_et_lv.get,
                                        key='dist_y')
                    st.caption(i18n.pick(
                        'Todas las variables de los puntos: primero las del modelo, de más '
                        'a menos peso, y después las que el modelo no usa. De entrada, las '
                        f'dos que más pesan. Puntos: {_cuantos}.',
                        "All the variables of the points: first the model's, from most to "
                        "least weight, then those the model does not use. By default, the "
                        f'two that weigh most. Points: {_cuantos}.'))
                    _ids_d = [r.get('ID') for r in rows]
                    _ch = pp.chart_dos_variables(
                        rows, _vx, _vy, _var(_vx), _var(_vy), comparar=_comparar,
                        traduce=i18n.tv,
                        nombres=bp.nombres_de_punto(_ids_d, i18n.pick('Punto', 'Point')))
                _como = i18n.pick(
                    'Cada punto es uno de tus puntos, colocado según las dos variables. Si '
                    'las nubes de colores se separan, combinar esas dos variables ayuda a '
                    'distinguir los grupos. Al pasar el ratón: el punto, sus valores y la '
                    'probabilidad del modelo.',
                    'Each dot is one of your points, placed by the two variables. If the '
                    'coloured clouds separate, combining those two variables helps to tell '
                    'the groups apart. Hover to see the point, its values and the model '
                    'probability.')
            if _ch is not None:
                st.altair_chart(_ch, width='stretch')
                st.caption(_como)
            else:
                st.info(i18n.pick(
                    'No hay datos suficientes para este gráfico (por ejemplo, sin puntos '
                    'que no sean borreguil no hay con qué comparar).',
                    'There is not enough data for this chart (for instance, without points '
                    'that are not a borreguil there is nothing to compare with).'))

        # --- Guardar el modelo reentrenado ---
        # Solo cuando se ha reentrenado sumando los puntos de la sesión: el de partida
        # ya está en el desplegable y guardarlo otra vez solo haría una copia.
        bundle = st.session_state.get('model_bundle')
        _info_ult = (hist[-1].get('info') if hist else None) or st.session_state.get('info') or {}
        _saved_msg = st.session_state.pop('_model_saved_msg', None)
        if bundle and _info_ult.get('mode') == 'trained':
            with st.expander(i18n.pick('💾 Guardar el modelo reentrenado',
                                       '💾 Save the retrained model'),
                             expanded=bool(_saved_msg)):
                # Mensaje PERSISTENTE: sobrevive al st.rerun que sigue a guardar.
                if _saved_msg:
                    st.success(_saved_msg)
                st.markdown(i18n.pick(
                    'Guarda el modelo reentrenado con tus puntos para usarlo otro día: '
                    'aparecerá en el desplegable «Modelo de partida». También puedes '
                    'descargarlo para pasárselo a otra persona.',
                    'Save the model retrained with your points to use it another day: it '
                    'will appear in the "Starting model" list. You can also download it '
                    'to give it to someone else.'))
                # Pre-rellenar con el nombre del área de estudio, sin sufijos largos
                # tipo «, Comarca de…, Andalucía, España»
                _si = st.session_state.get('study_info') or {}
                area_default = (_si.get('name') or _si.get('query') or '').strip()
                if ',' in area_default:
                    area_default = area_default.split(',')[0].strip()
                area_name = st.text_input(
                    i18n.pick('Nombre de la zona (será el nombre del modelo)',
                              'Name of the area (it will be the name of the model)'),
                    value=area_default or i18n.pick('Mi zona de montaña', 'My mountain area'))
                fname = area_to_filename(area_name)
                out_path = DATA_DIR / f'{fname}.joblib'
                st.caption(i18n.pick(
                    f'Se guardará como **{fname}**'
                    + (' (sustituye al que ya hay con ese nombre)' if out_path.exists() else ''),
                    f'It will be saved as **{fname}**'
                    + (' (it replaces the one with that name)' if out_path.exists() else '')))

                import io as _io, joblib as _jl
                b_save = dict(bundle); b_save['training_zone'] = area_name
                buf = _io.BytesIO()
                try:
                    _jl.dump(b_save, buf)
                    buf_bytes = buf.getvalue()
                except Exception as _je:
                    buf_bytes = None
                    st.error(i18n.pick(f'No se pudo preparar el modelo: {_je}',
                                       f'The model could not be prepared: {_je}'))

                cS1, cS2 = st.columns(2)
                with cS1:
                    if st.button(i18n.pick('💾 Guardar en la app', '💾 Save in the app'),
                                 type='primary'):
                        try:
                            bp.save_model_bundle(b_save, out_path)
                            if out_path.exists() and out_path.stat().st_size > 0:
                                st.session_state['_model_saved_msg'] = i18n.pick(
                                    f'✓ Modelo guardado: **{fname}**. Ya aparece en el '
                                    'desplegable «Modelo de partida».',
                                    f'✓ Model saved: **{fname}**. It is already in the '
                                    '"Starting model" list.')
                                st.rerun()
                            else:
                                st.error(i18n.pick(
                                    'No se pudo guardar. Usa «Descargar el modelo».',
                                    'It could not be saved. Use "Download the model".'))
                        except Exception as e:
                            st.error(i18n.pick(
                                f'No se pudo guardar ({e}). Usa «Descargar el modelo».',
                                f'It could not be saved ({e}). Use "Download the model".'))
                with cS2:
                    if buf_bytes:
                        st.download_button(
                            i18n.pick('⬇ Descargar el modelo', '⬇ Download the model'),
                            data=buf_bytes, file_name=f'{fname}.joblib',
                            mime='application/octet-stream', key='dl_model')


# ============================================================
# MAPA MODELO RF — pestaña propia, al mismo nivel que «Verdad terreno»
# De los puntos con el tipo revisado a un mapa de tipos de borreguil, con Random
# Forest y por pasos: datos, tabla de entrenamiento, modelo, mapa y peso de las
# variables. La lógica está en mapeo_rf.py y mapeo_raster.py; aquí solo se enseña.
# No toca los puntos ni el resto de la app. Va al final del guion porque usa los
# puntos tal como los deja la pestaña «Verdad terreno» en esta misma ejecución.
# ============================================================
with tab_tipos:
    import mapeo_rf as mr
    _en = i18n.get_lang() == 'en'
    st.subheader(i18n.pick('Mapa de borreguiles — Random Forest',
                           'Borreguil map — Random Forest'))
    if 'rows' not in st.session_state:
        st.info(i18n.pick(
            'Aquí se entrena un modelo con tus puntos y se obtiene el **mapa de '
            'borreguiles**: dónde hay borreguil y, si quieres, de qué tipo. Todavía no '
            'hay puntos en la sesión:\n\n'
            '1. Ejecuta el análisis (panel lateral → **▶ Ejecutar pipeline**). El '
            'resultado sale en la pestaña **📍 Verdad terreno**.\n'
            '2. En **📍 Verdad terreno → Revisar puntos**, marca puntos como **borreguil o no borreguil '
            'verificado**, o añádelos pinchando en **📍 Verdad terreno → Mapa de los puntos**: con eso se '
            'hace el mapa de borreguil sí o no. Para el tipo, revisa además el tipo de '
            'los borreguiles (ambiente, humedad y pureza).\n'
            '3. Vuelve a esta pestaña.',
            'Here a model is trained with your points to produce the **borreguil '
            'map**: where there is borreguil and, if you want, of which type. There are '
            'no points in the session yet:\n\n'
            '1. Run the analysis (sidebar → **▶ Run pipeline**). The result appears in '
            'the **📍 Ground truth** tab.\n'
            '2. Under **📍 Ground truth → Review points**, mark points as **verified borreguil or '
            'verified not-borreguil**, or add them by clicking under **📍 Ground truth → Points map**: '
            'that is enough for the borreguil yes-or-no map. For the type, also review '
            'the type of the borreguiles (environment, moisture and purity).\n'
            '3. Come back to this tab.'))
    else:
        rows = st.session_state.rows
        st.markdown(i18n.pick(
            'Entrena un modelo de **borreguil sí o no** con los puntos marcados en campo y, '
            'si quieres el tipo, otro con los puntos cuyo **tipo has revisado** (en **📍 '
            'Verdad terreno → Revisar puntos**), y obtén el **mapa de borreguiles**. Los pasos van '
            'en orden, cada uno en su pestaña.',
            'Train a **borreguil yes-or-no** model with the points marked in the field and, '
            'if you want the type, another one with the points whose **type you have '
            'reviewed** (under **📍 Ground truth → Review points**), and get the **borreguil map**. '
            'The steps go in order, each in its own tab.'))
        _niv_txt = {
            mr.NIVEL_SI_NO: i18n.pick('Borreguil sí o no', 'Borreguil yes or no'),
            'humedad': i18n.pick('Humedad (húmedo · seco)', 'Moisture (wet · dry)'),
            # corto: en el panel estrecho el desplegable no parte el texto, lo corta
            'ambiente_humedad': i18n.pick(
                'Ambiente y humedad (en cada ambiente, húmedo o seco)',
                'Environment and moisture (in each one, wet or dry)'),
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
        _nivel = st.selectbox(
            i18n.pick('Qué quieres cartografiar', 'What to map'),
            _NIVELES_APP, key='mrf_nivel', format_func=lambda n: _niv_txt[n],
            help=i18n.pick(
                'Se entrena un modelo por cada opción.\n\n'
                '**Borreguil sí o no** decide solo si hay borreguil. Aprende de **todos '
                'los puntos marcados en campo** como borreguil o no, tengan o no el tipo '
                'revisado. Su mapa ya es el de borreguil sí o no.\n\n'
                'Para el **tipo**, entrena después una de las otras opciones: aprende el '
                'tipo solo de los borreguiles con el tipo revisado, y en su mapa el sí o '
                'no lo sigue decidiendo el modelo de **Borreguil sí o no**.\n\n'
                '**Ambiente y humedad** da el tipo completo: el modelo dice si es húmedo '
                'o seco, y el ambiente (arroyo, laguna o ladera) sale de la **cercanía de '
                'cada píxel a lagunas y cauces**, que se ajusta en **4 · Mapa**.\n\n'
                '**Recomendado: primero Borreguil sí o no, y después Ambiente y '
                'humedad.**\n\n'
                'No se ofrece la combinación de los tres niveles (18 categorías): no '
                'habría puntos para casi ninguna. La pureza se añade al mapa de otra '
                'manera: «Pureza dentro del píxel», en **4 · Mapa**.',
                'One model is trained for each option.\n\n'
                '**Borreguil yes or no** only decides whether there is a borreguil. It '
                'learns from **every point marked in the field** as a borreguil or not, '
                'whether or not its type has been reviewed. Its map is already the '
                'borreguil yes-or-no map.\n\n'
                'For the **type**, train one of the other options afterwards: it learns '
                'the type only from the borreguiles whose type has been reviewed, and in '
                'its map the yes or no is still decided by the **Borreguil yes or no** '
                'model.\n\n'
                '**Environment and moisture** gives the full type: the model says whether '
                'it is wet or dry, and the environment (stream, lake or slope) comes from '
                '**how close each pixel is to lakes and watercourses**, set under **4 · '
                'Map**.\n\n'
                '**Recommended: first Borreguil yes or no, then Environment and '
                'moisture.**\n\n'
                'The combination of the three levels (18 categories) is not offered: '
                'there would be no points for most of them. Purity is added to the map '
                'in another way: "Purity within the pixel", under **4 · Map**.'))
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
            """Lo que sale en «1 · Datos» al aplicar el VIF o el PCA: qué variables se
            quedan, la tabla del análisis y, variable a variable, por qué (25-11-2026).
            `ap` es lo guardado al aplicar; `param`, el umbral del VIF o la correlación
            mínima del PCA."""
            s, filas, de = ap['sel'], ap['filas'], ap['de']
            quedan = ', '.join(_var(v) for v in s['quedan'])
            nq = len(s['quedan'])
            if metodo == 'vif':
                st.success(i18n.pick(
                    f'**Con el VIF se quedan {nq} de {de} variables:** {quedan}.',
                    f'**With the VIF, {nq} of {de} variables stay:** {quedan}.'))
            else:
                st.success(i18n.pick(
                    f'**Con el PCA se quedan {nq} de {de} variables:** {quedan}. Una por '
                    f"cada eje importante ({len(s['ejes'])} ejes, el "
                    f"{_num(s['varianza_pct'], 1)} % de la varianza), más las que se quedan "
                    'siempre.',
                    f'**With the PCA, {nq} of {de} variables stay:** {quedan}. One for each '
                    f"important axis ({len(s['ejes'])} axes, {_num(s['varianza_pct'], 1)} % "
                    'of the variance), plus those that always stay.'))
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
                    return i18n.pick('se queda siempre', 'always stays')
                if q in ('queda', 'elegida'):
                    if metodo == 'pca' and f['eje']:
                        return i18n.pick(f"se queda (eje {f['eje']})",
                                         f"stays (axis {f['eje']})")
                    return i18n.pick('se queda', 'stays')
                paso = (i18n.pick(f" (paso {f['paso']})", f" (step {f['paso']})")
                        if f.get('paso') else '')
                return (i18n.pick('se quita', 'removed') + paso
                        + (i18n.pick(': no varía', ': it does not vary')
                           if q == 'no_varia' else ''))

            def por_que(f):
                return mr.texto_por_que(f, metodo, _en, nombre=_var, num=_num,
                                        umbral=param if metodo == 'vif' else None,
                                        carga=param if metodo == 'pca' else None)

            if metodo == 'vif':
                def txt_vif(v):
                    return '—' if v is None else ('∞' if v == float('inf') else _num(v))
                st.dataframe(pd.DataFrame([{
                    'Variable': _var(f['variable']),
                    'VIF': txt_vif(f['vif']),
                    k_q: que_pasa(f),
                    i18n.pick('Las demás la explican', 'The others explain it'):
                        '—' if f['explicado_pct'] is None else f"{_num(f['explicado_pct'], 1)} %",
                    i18n.pick('Más parecida de las que se quedan',
                              'Most similar of those that stay'):
                        f"{_var(f['parecida'])} (r {_num(f['r'])})" if f['parecida'] else '—'}
                    for f in filas]), hide_index=True, width='stretch')
                lim = 100 * (1 - 1 / param)
                lim = _num(lim, 0 if abs(lim - round(lim)) < 0.05 else 1)
                st.caption(i18n.pick(
                    'El VIF de una variable es 1 / (1 − R²), con R² lo que las demás juntas '
                    f'explican de ella: con el umbral {_num(param, 1)}, se quitan las que las '
                    f'demás explican en más de un {lim} %. Se '
                    'quita la de VIF más alto, se vuelve a calcular y se sigue mientras alguna '
                    'pase del umbral. De las que se quedan, su VIF entre ellas; de las que se '
                    'quitan, el que tenían al quitarlas. ∞: se obtiene exactamente con otras. '
                    'r es la correlación entre dos variables: cerca de 1 o de −1, dicen casi '
                    'lo mismo.',
                    'The VIF of a variable is 1 / (1 − R²), with R² how much of it the others '
                    f'together explain: with the threshold {_num(param, 1)}, those that the '
                    f'others explain more than {lim} % are removed. '
                    'The one with the highest VIF is removed, the VIF is computed again, and so '
                    'on while any is above the threshold. For those that stay, their VIF among '
                    'themselves; for those removed, the one they had when removed. ∞: it is '
                    'obtained exactly from others. r is the correlation between two '
                    'variables: close to 1 or −1, they say almost the same.'))
            else:
                import altair as alt
                vp = s.get('valores_propios') or []
                if vp:
                    t_eje, t_vp = i18n.pick('Eje', 'Axis'), i18n.pick('Valor propio',
                                                                     'Eigenvalue')
                    dfv = pd.DataFrame({'eje': list(range(1, len(vp) + 1)), 'vp': vp,
                                        'imp': [v > mr.PCA_VALOR_PROPIO for v in vp]})
                    barras = alt.Chart(dfv).mark_bar().encode(
                        x=alt.X('eje:O', title=t_eje, axis=alt.Axis(labelAngle=0)),
                        y=alt.Y('vp:Q', title=t_vp),
                        color=alt.Color('imp:N', legend=None, scale=alt.Scale(
                            domain=[True, False], range=['#2e7d32', '#bdbdbd'])),
                        tooltip=[alt.Tooltip('eje:O', title=t_eje),
                                 alt.Tooltip('vp:Q', title=t_vp, format='.3f')])
                    regla = alt.Chart(pd.DataFrame({'y': [mr.PCA_VALOR_PROPIO]})).mark_rule(
                        color='#c62828', strokeDash=[4, 3]).encode(y='y:Q')
                    st.altair_chart((barras + regla).properties(height=180), width='stretch')
                    st.caption(i18n.pick(
                        'Valor propio de cada eje del PCA: cuántas variables «vale» el eje. '
                        'Los verdes pasan de 1 (la línea roja): resumen más que una variable '
                        'sola, y de cada uno se queda una variable.',
                        'Eigenvalue of each PCA axis: how many variables the axis is "worth". '
                        'The green ones are above 1 (the red line): they summarise more than a '
                        'single variable, and one variable stays for each.'))
                k_lig = i18n.pick(f"Ligadas al eje (|r| ≥ {_num(param)})",
                                  f"Tied to the axis (|r| ≥ {_num(param)})")
                st.dataframe(pd.DataFrame([{
                    i18n.pick('Eje', 'Axis'): e['eje'],
                    i18n.pick('Valor propio', 'Eigenvalue'): _num(e['valor_propio']),
                    i18n.pick('Varianza (%)', 'Variance (%)'): _num(e['varianza_pct'], 1),
                    k_lig: ', '.join(f"{_var(d['variable'])} ({_num(d['carga'])})"
                                     for d in e['variables']) or '—',
                    i18n.pick('Se queda', 'Stays'):
                        _var(e['elegida']) if e['elegida'] else '—'}
                    for e in s['ejes']]), hide_index=True, width='stretch')
                st.dataframe(pd.DataFrame([{
                    'Variable': _var(f['variable']),
                    k_q: que_pasa(f),
                    i18n.pick('Su eje', 'Its axis'): str(f['eje']) if f['eje'] else '—',
                    i18n.pick('Correlación con su eje', 'Correlation with its axis'):
                        '—' if f['carga'] is None else _num(f['carga']),
                    i18n.pick('Separa sola', 'Separates on its own'):
                        '—' if f['separa'] is None else _num(f['separa'])}
                    for f in filas]), hide_index=True, width='stretch')
                st.caption(i18n.pick(
                    'Cada eje del PCA resume variables que van juntas. Correlación con el eje: '
                    'lo ligada que está la variable a él (cerca de 1 o de −1, mucho). Separa '
                    'sola: lo bien que distingue las categorías ella sola; con dos, de 0 (nada) '
                    'a 1 (del todo), y con más, la F del ANOVA (cuanto más alta, mejor). Las '
                    'que se quedan siempre (' + (', '.join(_var(v) for v in s['fijas'])
                                                  or 'ninguna') + ') no entran en el PCA.',
                    'Each PCA axis summarises variables that go together. Correlation with the '
                    'axis: how tied the variable is to it (close to 1 or −1, a lot). Separates '
                    'on its own: how well it tells the categories apart by itself; with two, '
                    'from 0 (not at all) to 1 (completely), and with more, the ANOVA F (the '
                    'higher, the better). Those that always stay (' + (', '.join(
                        _var(v) for v in s['fijas']) or 'none') + ') are left out of the PCA.'))
            with st.expander(i18n.pick('Por qué, variable a variable',
                                       'Why, variable by variable'), expanded=True):
                st.markdown('\n'.join(
                    f"- **{_var(f['variable'])}** · {que_pasa(f)}. {por_que(f)}"
                    for f in filas))

        # Por dónde va el trabajo en el nivel elegido. El hueco se rellena al final,
        # cuando las pestañas han hecho sus cuentas, pero queda encima de ellas.
        _estado = st.container()
        _t_datos, _t_tabla, _t_modelo, _t_mapa, _t_peso = st.tabs([
            i18n.pick('1 · Datos', '1 · Data'),
            i18n.pick('2 · Tabla de entrenamiento', '2 · Training table'),
            i18n.pick('3 · Modelo', '3 · Model'),
            i18n.pick('4 · Mapa', '4 · Map'),
            i18n.pick('5 · Peso', '5 · Weights')])

        # ------------------------------------------------------------ 1 · Datos
        with _t_datos:
            st.caption(i18n.pick(
                'Qué puntos sirven como muestra en este nivel y qué variables se van a '
                'usar.',
                'Which points count as samples at this level and which variables will '
                'be used.'))
            if _cercania:
                st.info(i18n.pick(
                    'En «Ambiente y humedad» el modelo aprende **húmedo o seco** (y «no '
                    'borreguil»): por eso aquí se cuentan esas categorías. El **ambiente** '
                    'no sale del modelo: lo pone el mapa por cercanía a lagunas y cauces, '
                    'en **4 · Mapa**.',
                    'Under "Environment and moisture" the model learns **wet or dry** (and '
                    '"not a borreguil"): that is why those categories are counted here. '
                    'The **environment** does not come from the model: the map sets it by '
                    'closeness to lakes and watercourses, under **4 · Map**.'))
            # Ausencias de campo: lo único que puede formar la clase «no borreguil».
            _n_aus = sum(1 for r in rows if bp._txt(r.get('Borreguil')).lower() == 'no'
                         and bp._txt(r.get('Duda')).lower() != 'si')
            if _nivel == mr.NIVEL_SI_NO:
                # El de sí o no siempre lleva las ausencias: sin ellas no hay «no».
                _incluir = True
                st.info(i18n.pick(
                    'Este modelo aprende **solo si hay borreguil**: cuenta cada punto '
                    'marcado en campo como borreguil o como «no borreguil», **tenga o no '
                    'el tipo revisado**. Un borreguil con el tipo revisado también cuenta. '
                    'Lo que propone el detector, no.',
                    'This model learns **only whether there is a borreguil**: it counts '
                    'every point marked in the field as a borreguil or as "not a '
                    'borreguil", **whether or not its type has been reviewed**. A '
                    'borreguil whose type has been reviewed also counts. What the detector '
                    'proposes does not.'))
            elif _n_aus:
                # En un nivel del tipo, lo normal es que el sí o no lo decida el modelo
                # de sí o no, y que este aprenda solo los tipos (sin «no borreguil»).
                _quien_txt = {
                    'si_no': i18n.pick('El modelo «Borreguil sí o no» (recomendado)',
                                       'The "Borreguil yes or no" model (recommended)'),
                    'este': i18n.pick('Este mismo modelo, con las ausencias de campo',
                                      'This same model, with the field absences')}
                _quien = st.radio(
                    i18n.pick('Quién decide si un píxel es borreguil',
                              'Who decides whether a pixel is a borreguil'),
                    list(_quien_txt), key='mrf_quien_decide',
                    format_func=lambda q: _quien_txt[q],
                    help=i18n.pick(
                        '- **El modelo «Borreguil sí o no»:** este modelo aprende **solo '
                        'el tipo**, de los borreguiles con el tipo revisado. En el mapa, '
                        'si un píxel es borreguil lo decide el modelo de «Borreguil sí o '
                        'no», que aprende de todos los puntos de campo (también de los '
                        'borreguiles sin el tipo revisado), y este le pone el tipo. Hay '
                        'que entrenar los dos.\n'
                        f'- **Este mismo modelo:** aprende también «no borreguil», con las '
                        f'{_n_aus} ausencias de campo, y decide él solo. Solo cuentan los '
                        'borreguiles con el tipo revisado.\n\n'
                        '**Recomendado: el modelo «Borreguil sí o no».**',
                        '- **The "Borreguil yes or no" model:** this model learns **only '
                        'the type**, from the borreguiles whose type has been reviewed. In '
                        'the map, whether a pixel is a borreguil is decided by the '
                        '"Borreguil yes or no" model, which learns from every field point '
                        '(including borreguiles whose type has not been reviewed), and '
                        'this one gives it its type. Both have to be trained.\n'
                        f'- **This same model:** it also learns "not a borreguil", with the '
                        f'{_n_aus} field absences, and decides on its own. Only the '
                        'borreguiles whose type has been reviewed count.\n\n'
                        '**Recommended: the "Borreguil yes or no" model.**'))
                _incluir = _quien == 'este'
            else:
                _incluir = False
                st.caption(i18n.pick(
                    'No hay ausencias de campo en estos puntos, así que no habrá clase '
                    '«no borreguil» (no se inventan ausencias).',
                    'There are no field absences among these points, so there will be '
                    'no "not a borreguil" class (absences are never invented).'))
            _res = mr.muestras(rows, _nivel_mod, incluir_ausencias=bool(_incluir and _n_aus),
                               es_borreguil=pp.is_borreguil_decision)

            # Categoría | Código | Nº de muestras
            st.dataframe(pd.DataFrame([{
                i18n.pick('Categoría', 'Category'): i18n.tv(c['categoria']),
                i18n.pick('Código', 'Code'): c['codigo'],
                i18n.pick('Nº de muestras', 'Nº of samples'): c['n']}
                for c in _res['clases']]), hide_index=True, width='stretch')
            _nm = _res['n_usadas']
            st.markdown(i18n.pick(
                f"**{_nm} muestra{'' if _nm == 1 else 's'}** de los {_res['n_inicial']} "
                f"puntos de la sesión.",
                f"**{_nm} sample{'' if _nm == 1 else 's'}** out of the "
                f"{_res['n_inicial']} points in the session."))
            if _res['errores']:
                st.error(i18n.pick('**Con estos puntos todavía no se puede entrenar:**',
                                   '**These points are not enough to train yet:**')
                         + ''.join('\n- ' + mr.mensaje(e, _en, i18n.tv, nivel=_nivel_mod)
                                   for e in _res['errores']))
            _otros = [a for a in _res['avisos'] if a[0] != 'sin_ausencia']
            if _otros:
                st.warning(''.join('- ' + mr.mensaje(a, _en, i18n.tv) + '\n' for a in _otros))
            if ('sin_ausencia',) in _res['avisos']:
                if _nivel != mr.NIVEL_SI_NO and _n_aus and not _incluir:
                    _hay_si_no = mr.NIVEL_SI_NO in st.session_state.get('mrf_modelos', {})
                    st.info(i18n.pick(
                        'Este modelo aprende **solo el tipo**. Si un píxel es borreguil lo '
                        'decide en el mapa el modelo **«Borreguil sí o no»**'
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

            # ¿Son bastantes puntos? Dónde está la persona en la curva de Sierra Nevada
            # (24-11, idea de la autora). Solo para el sí o no: la curva se midió ahí.
            if _nivel == mr.NIVEL_SI_NO:
                # Borreguiles y ausencias por separado (las ausencias, desde el 25-11-2026:
                # antes, 100 borreguiles y 6 ausencias «llegaban al mínimo»)
                _nb = sum(c['n'] for c in _res['clases']
                          if c['codigo'] != mr.CODIGO_AUSENCIA)
                _na = sum(c['n'] for c in _res['clases']
                          if c['codigo'] == mr.CODIGO_AUSENCIA)
                _lp = mr.lectura_puntos(_nm, _nb, _na)
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
                _tienes_es = (f"**{_nm} muestra{'' if _nm == 1 else 's'} ({_bor(_nb)} y "
                              f"{_aus(_na)})**")
                _tienes_en = (f"**{_nm} sample{'' if _nm == 1 else 's'} ({_bor(_nb, True)} "
                              f"and {_aus(_na, True)})**")
                _min_es = f'{_mn} puntos: {_mb} borreguiles y {_ma} ausencias'
                _min_en = f'{_mn} points: {_mb} borreguiles and {_ma} absences'
                _auc = _num(_lp['auc'])
                st.markdown(i18n.pick('**¿Son bastantes puntos?**', '**Are they enough points?**'))
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
                        f'Tienes {_tienes_es}. En Sierra Nevada, con {_nm} puntos el AUC en '
                        f'zonas nuevas era de {_auc}{_var_es}. Para el mínimo ({_min_es}), '
                        f'{_falta_es}.',
                        f'You have {_tienes_en}. In Sierra Nevada, with {_nm} points the AUC '
                        f'in new areas was {_auc}{_var_en}. For the minimum ({_min_en}), '
                        f'{_falta_en}.'))
                elif _lp['nivel'] == 'minimo':
                    st.info(i18n.pick(
                        f'Tienes {_tienes_es}: llegas al mínimo. En Sierra Nevada, con '
                        f'{_nm} puntos el AUC en zonas nuevas era de {_auc}. Con más, hasta '
                        f'unos {_rn} ({_rb} borreguiles y {_ra} ausencias), el mapa todavía '
                        'mejora algo.',
                        f'You have {_tienes_en}: you reach the minimum. In Sierra Nevada, '
                        f'with {_nm} points the AUC in new areas was {_auc}. With more, up '
                        f'to about {_rn} ({_rb} borreguiles and {_ra} absences), the map '
                        'still improves a little.'))
                else:
                    st.success(i18n.pick(
                        f'Tienes {_tienes_es}: llegas a lo recomendado. En Sierra Nevada, con '
                        f'{_nm} puntos el AUC en zonas nuevas era de {_auc}; con más puntos '
                        'apenas cambia.',
                        f'You have {_tienes_en}: you reach the recommended number. In Sierra '
                        f'Nevada, with {_nm} points the AUC in new areas was {_auc}; with more '
                        'points it hardly changes.'))
                # Borreguiles y ausencias, siempre a la vista (25-11-2026, la autora: «no
                # veo lo de las ausencias»: si se llegaba, solo salían entre paréntesis)
                def _meta(tienes, hace_falta):
                    return (f'{hace_falta} ✓' if tienes >= hace_falta else
                            i18n.pick(f'{hace_falta} (faltan {hace_falta - tienes})',
                                      f'{hace_falta} ({hace_falta - tienes} to go)'))
                # El total va primero: es el número de la gráfica (100 = 33 + 67); sin él,
                # «mínimo 33» y «mínimo 100 puntos» parecían contradecirse (25-11-2026)
                st.dataframe(pd.DataFrame([
                    {i18n.pick('Muestras', 'Samples'):
                         i18n.pick('Total (los puntos de la gráfica)',
                                   'Total (the points in the chart)'),
                     i18n.pick('Tienes', 'You have'): str(_nm),
                     i18n.pick('Mínimo', 'Minimum'): _meta(_nm, _mn),
                     i18n.pick('Recomendado', 'Recommended'): _meta(_nm, _rn)},
                    {i18n.pick('Muestras', 'Samples'): i18n.pick('· borreguiles',
                                                                 '· borreguiles'),
                     i18n.pick('Tienes', 'You have'): str(_nb),
                     i18n.pick('Mínimo', 'Minimum'): _meta(_nb, _mb),
                     i18n.pick('Recomendado', 'Recommended'): _meta(_nb, _rb)},
                    {i18n.pick('Muestras', 'Samples'):
                         i18n.pick('· ausencias (no borreguil)',
                                   '· absences (not a borreguil)'),
                     i18n.pick('Tienes', 'You have'): str(_na),
                     i18n.pick('Mínimo', 'Minimum'): _meta(_na, _ma),
                     i18n.pick('Recomendado', 'Recommended'): _meta(_na, _ra)}]),
                    hide_index=True, width='stretch')
                st.caption(i18n.pick(
                    f'El mínimo de la gráfica, {_mn} puntos, es la suma de {_mb} borreguiles y '
                    f'{_ma} ausencias; lo recomendado, {_rn} = {_rb} + {_ra}. Hacen falta los '
                    'dos: muchos borreguiles no suplen las ausencias que faltan. '
                    'Las ausencias son los sitios que has comprobado que **no** son '
                    'borreguil: `presence` = 0 en tu fichero, «No» en «¿Es borreguil?» al '
                    'revisar un punto o «✗ Aquí no hay» al pinchar en el mapa. '
                    'La app no se las inventa. Para buscarlas donde no has muestreado, genera '
                    'puntos al azar «lejos de tus puntos de campo» y revísalos.',
                    f'The minimum in the chart, {_mn} points, is the sum of {_mb} borreguiles '
                    f'and {_ma} absences; the recommended one, {_rn} = {_rb} + {_ra}. Both '
                    'are needed: many borreguiles do not make up for missing absences. '
                    'Absences are the places you have checked are **not** a borreguil: '
                    '`presence` = 0 in your file, "No" under "Is it a borreguil?" when '
                    'reviewing a point or "✗ There is none" when clicking on the map. The app '
                    'never invents them. To look for them where you have not sampled, '
                    'generate random points "far from your field points" and review them.'))
                st.altair_chart(pp.chart_curva_puntos(
                    mr.CURVA_PUNTOS_SN, n=_nm,
                    marcas=[(_mn, i18n.pick('Mínimo', 'Minimum')),
                            (_rn, i18n.pick('Recomendado', 'Recommended'))]),
                    width='stretch')
                st.caption(i18n.pick(
                    'Referencia de Sierra Nevada: el mapa entrenado con distintos números de '
                    'puntos revisados y medido en zonas que no había visto. La curva se hizo '
                    'con 1 borreguil por cada 2 ausencias; de ahí salen los borreguiles y las '
                    'ausencias del mínimo y de lo recomendado (no se probaron otras '
                    'proporciones). En una montaña más variada pueden hacer falta más, y los '
                    'puntos repartidos por toda la zona enseñan más que los concentrados en '
                    'pocos valles. La curva con tus propios puntos se calcula en «3 · '
                    'Modelo», después de entrenar. Más en 📖 Documentación.',
                    'Sierra Nevada reference: the map trained with different numbers of '
                    'reviewed points and measured in areas it had not seen. The curve was '
                    'made with 1 borreguil for every 2 absences; that is where the '
                    'borreguiles and absences of the minimum and the recommended numbers come '
                    'from (other proportions were not tested). A more varied mountain may '
                    'need more, and points spread over the whole area teach more than points '
                    'concentrated in a few valleys. The curve with your own points is '
                    'computed under "3 · Model", after training. More under 📖 '
                    'Documentation.'))

            # Qué puntos no entran, y por qué
            if _res['excluidos']:
                st.markdown(i18n.pick('**Puntos que no entran como muestra**',
                                      '**Points not used as samples**'))
                st.dataframe(pd.DataFrame([{
                    i18n.pick('Motivo', 'Reason'): mr.motivo(k, _en),
                    i18n.pick('Nº de puntos', 'Nº of points'): v}
                    for k, v in sorted(_res['excluidos'].items(), key=lambda kv: -kv[1])]),
                    hide_index=True, width='stretch')
                _sin_rev = _res['excluidos'].get('tipo_sin_revisar', 0)
                if _sin_rev:
                    st.caption(i18n.pick(
                        f'Los {_sin_rev} borreguiles con el tipo sin revisar pasarán a ser '
                        'muestras en cuanto los revises en **📍 Verdad terreno → Revisar puntos**.',
                        f'The {_sin_rev} borreguiles whose type has not been reviewed will '
                        'become samples as soon as you review them under **📍 Ground truth → '
                        'Review points**.'))
            with st.expander(i18n.pick(
                    'Ver punto por punto (cuáles se usan y por qué no el resto)',
                    'See point by point (which are used and why not the rest)')):
                _dfm = pd.DataFrame(_res['tabla'])
                _dfm['ID'] = bp.abreviar_ids(_dfm['ID'].tolist())
                _origen_en = {mr.ORIGEN_REVISADO: 'reviewed type',
                              mr.ORIGEN_AUSENCIA: 'field absence'}
                _dfm['categoria'] = _dfm['categoria'].map(i18n.tv)
                # como texto: la columna mezcla códigos y huecos (puntos sin categoría)
                _dfm['codigo'] = _dfm['codigo'].map(str)
                _dfm['propuesta_app'] = _dfm['propuesta_app'].map(i18n.tv)
                _dfm['origen_etiqueta'] = _dfm['origen_etiqueta'].map(
                    lambda o: _origen_en.get(o, o) if _en else o)
                _dfm['usada'] = _dfm['usada'].map(
                    lambda u: i18n.pick('sí', 'yes') if u else 'no')
                _dfm['motivo'] = _dfm['motivo'].map(lambda k: mr.motivo(k, _en) if k else '')
                _cab = {'categoria': i18n.pick('categoría', 'category'),
                        'codigo': i18n.pick('código', 'code'),
                        'origen_etiqueta': i18n.pick('origen de la etiqueta', 'label source'),
                        'fecha_revision': i18n.pick('fecha de revisión', 'review date'),
                        'propuesta_app': i18n.pick('propuesta de la app', "app's proposal"),
                        'usada': i18n.pick('se usa', 'used'),
                        'motivo': i18n.pick('motivo si no se usa', 'reason if not used')}
                st.dataframe(
                    _dfm[['ID', 'usada', 'categoria', 'codigo', 'origen_etiqueta',
                          'propuesta_app', 'fecha_revision', 'motivo']].rename(columns=_cab),
                    hide_index=True, width='stretch')
                st.download_button(
                    label=f'⬇ muestras_{_nivel}.csv',
                    data=mr.tabla_csv(_res).encode('utf-8-sig'),
                    file_name=f'muestras_{_nivel}.csv', mime='text/csv',
                    key='mrf_dl_muestras',
                    help=i18n.pick(
                        'Todos los puntos de la sesión, con el nombre completo, la '
                        'categoría, si se usan y, si no, por qué.',
                        'Every point in the session, with its full name, category, '
                        'whether it is used and, if not, why.'))

            # Variables predictoras disponibles como mapa
            st.markdown(i18n.pick('#### Variables predictoras', '#### Predictor variables'))
            st.caption(i18n.pick(
                'Para predecir sobre el territorio solo sirven las variables que existen '
                f'como **mapa continuo**. Hoy son {len(mr.PREDICTORES)}: las que la app '
                'calcula en Earth Engine con Sentinel-2 y el modelo de elevaciones. Vienen '
                'todas marcadas; quita las que no quieras usar.',
                'Only variables that exist as a **continuous map** can be used to predict '
                f'over the territory. There are {len(mr.PREDICTORES)} today: those the app '
                'computes in Earth Engine from Sentinel-2 and the elevation model. They '
                'are all selected; remove the ones you do not want to use.'))
            # Lo que ya está leído en Earth Engine de cada muestra («2 · Tabla»). Vale
            # mientras no cambien los años, la zona UTM de la rejilla ni la manera de
            # calcular las variables (geb.VERSION_PILA). Se calcula aquí porque el VIF y
            # el PCA se aplican en esta pestaña, con esos valores (y antes de las casillas
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
                """Deja marcadas en las casillas solo `variables`. Solo vale antes de
                dibujarlas (o en el on_click de un botón)."""
                for gi, g in enumerate(mr.GRUPOS):
                    st.session_state[f'mrf_vars_{gi}'] = [
                        v for v in mr.nombres_predictores([g]) if v in variables]

            # ---- Aplicar el VIF o el PCA (25-11-2026, la autora: «al aplicar el vif o
            # pca me gustaría que en variables predictoras se queden las que han
            # quedado»). El botón (más abajo) solo deja el aviso '_mrf_aplicar'; se aplica
            # aquí, en la pasada siguiente y antes de dibujar las casillas, para poder
            # dejar marcadas solo las que quedan. (Con st.rerun() en el botón, Streamlit
            # olvidaba lo elegido en los controles de debajo, que esa pasada no llegaba
            # a dibujar.) Se parte de las que había marcadas antes de aplicarlo, si
            # siguen marcadas las que quedaron; si no, de las marcadas ahora.
            if st.session_state.pop('_mrf_aplicar', False):
                _m0 = st.session_state.get('mrf_seleccion', 'ninguno')
                _ahora0 = mr.ordenar_predictores([
                    v for gi, g in enumerate(mr.GRUPOS)
                    for v in st.session_state.get(f'mrf_vars_{gi}',
                                                  mr.nombres_predictores([g]))])
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
                """«↺ Volver a marcar…»: las que había antes de aplicar el VIF o el PCA;
                lo aplicado se deshace."""
                a = st.session_state.pop('mrf_vars_antes', None)
                if a:
                    _marcar(a['vars'])
                st.session_state.pop('mrf_sel_aplicada', None)

            # Las casillas no llevan `default`: el valor de partida (todas) se pone en
            # session_state, que es lo que cambian el VIF o el PCA al aplicarse.
            _elegidas = []
            for _gi, _g in enumerate(mr.GRUPOS):
                _ops = mr.nombres_predictores([_g])
                st.session_state.setdefault(f'mrf_vars_{_gi}', list(_ops))
                # Solo el nombre: con la descripción al lado las etiquetas se cortaban.
                # Qué mide cada una está en la tabla de debajo.
                _elegidas += st.multiselect(
                    mr.nombre_grupo(_g, _en), _ops, key=f'mrf_vars_{_gi}',
                    format_func=_var,
                    placeholder=i18n.pick('Ninguna marcada', 'None selected'))
            _predictores = mr.ordenar_predictores(_elegidas)
            st.markdown(i18n.pick(
                f'**{len(_predictores)} de {len(mr.PREDICTORES)} variables seleccionadas.**',
                f'**{len(_predictores)} of {len(mr.PREDICTORES)} variables selected.**'))
            if not _predictores:
                st.error(i18n.pick('Elige al menos una variable.',
                                   'Choose at least one variable.'))
            # Las que había marcadas antes de aplicar el VIF o el PCA: de ellas parte el
            # VIF o el PCA si se vuelve a aplicar (con otro umbral, o el otro método), y
            # se pueden volver a marcar con un botón. Si la persona cambia las casillas a
            # mano, lo que marca pasa a ser el punto de partida.
            _antes = st.session_state.get('mrf_vars_antes')
            if _antes and set(_predictores) != set(_antes['resultado']):
                st.session_state.pop('mrf_vars_antes', None)
                _antes = None
            if _antes and set(_antes['vars']) != set(_predictores):
                _nom_antes = (i18n.pick('el VIF', 'the VIF') if _antes['metodo'] == 'vif'
                              else i18n.pick('el PCA', 'the PCA'))
                _na = len(_antes['vars'])
                st.caption(i18n.pick(
                    f'Están marcadas las que dejó {_nom_antes}. Antes de aplicarlo había '
                    f'{_na} marcadas.',
                    f'The ones {_nom_antes} kept are selected. Before applying it there '
                    f'were {_na} selected.'))
                st.button(i18n.pick(f'↺ Volver a marcar las {_na} de antes',
                                    f'↺ Select the {_na} from before again'),
                          key='mrf_vars_volver', on_click=_volver_a_las_de_antes,
                          help=i18n.pick(
                              'Vuelve a marcar las variables que había antes de aplicar el '
                              'VIF o el PCA, por ejemplo para entrenar con todas y '
                              'comparar. Lo aplicado se deshace.',
                              'Selects again the variables there were before applying the '
                              'VIF or the PCA, for instance to train with all of them and '
                              'compare. What was applied is undone.'))

            # La tabla de entrenamiento con todas las variables elegidas (la de «2 ·
            # Tabla» antes de quitar las correlacionadas), si ya hay algo leído
            _tab_todas = (mr.construir_tabla(_res, _leidos, _predictores)
                          if _predictores and any(mr.clave_punto(f['lon'], f['lat'])
                                                  in _leidos for f in _usadas) else None)

            # ---- Variables correlacionadas: con el VIF o con un PCA. Desde el 24-11 se decide
            # aquí, junto a las variables (lo pidió la autora), y desde el 25-11 se
            # aplica aquí con un botón, que enseña qué variables se quedan y por qué. Se
            # calcula con los valores de las muestras (la lectura de «2 · Tabla» trae
            # siempre todas las variables). De entrada no se quita nada.
            st.markdown(i18n.pick('**Variables correlacionadas**', '**Correlated variables**'))
            _met_txt = {'ninguno': i18n.pick('No quitar ninguna', 'Remove none'),
                        'vif': i18n.pick('Con el VIF', 'With the VIF'),
                        'pca': i18n.pick('Con un PCA', 'With a PCA')}
            _metodo = st.radio(
                i18n.pick('Quitar las variables correlacionadas', 'Remove correlated variables'),
                list(_met_txt), format_func=_met_txt.get, key='mrf_seleccion',
                horizontal=True,
                help=i18n.pick(
                    'Muchas variables dicen casi lo mismo (los índices de verdor, por '
                    'ejemplo). Hay dos formas de quedarse con menos:\n\n'
                    '- **Con el VIF**: el VIF de una variable dice qué parte de ella se '
                    'puede adivinar con todas las demás juntas (5 quiere decir el 80 %; 10, '
                    'el 90 %). Se quita la más correlacionada con las demás, se vuelve a calcular y se '
                    'sigue mientras alguna pase del umbral.\n'
                    '- **Con un PCA**: el PCA resume las variables en unos pocos ejes '
                    'independientes. De cada eje importante (valor propio mayor que 1) se '
                    'queda una variable: entre las muy ligadas a él, la que mejor separa '
                    'las categorías por sí sola.\n\n'
                    'Los dos se calculan con los valores de las muestras, que se leen en '
                    '«2 · Tabla». Después, al pulsar **Aplicar**, sale aquí mismo qué '
                    'variables se quedan y por qué, y arriba se quedan marcadas solo esas '
                    '(«↺ Volver a marcar…» devuelve las de antes). El Random Forest no los '
                    'necesita para acertar, pero con menos variables el modelo es más '
                    'sencillo y su peso, en «5 · Peso», se lee mejor. En Sierra Nevada, el '
                    'VIF dejó 14 variables y el PCA con la altitud, 4, con el mismo acierto '
                    '(F1 0,865 y 0,861); el del PCA cartografió un 38 % más de borreguil.\n\n'
                    '**Recomendado:** entrenar con cada opción y quedarse con la que mejor '
                    'valide (la comparación sale al final de «3 · Modelo»).',
                    'Many variables say almost the same (the greenness indices, for '
                    'instance). There are two ways of keeping fewer:\n\n'
                    '- **With the VIF**: the VIF of a variable says how much of it can be '
                    'guessed from all the others together (5 means 80 %; 10, 90 %). The '
                    'one most correlated with the others is removed, the VIF is computed again, and so on '
                    'while any is above the threshold.\n'
                    '- **With a PCA**: the PCA summarises the variables in a few '
                    'independent axes. One variable is kept from each important axis '
                    '(eigenvalue above 1): among those strongly tied to it, the one that '
                    'best separates the categories on its own.\n\n'
                    'Both are computed with the values of the samples, which are read under '
                    '"2 · Table". Then, when you press **Apply**, which variables stay and '
                    'why is shown right here, and only those stay selected above ("↺ Select '
                    '… again" brings back the previous ones). The Random Forest '
                    'does not need them to be right, but with fewer variables the model is '
                    'simpler and its weights, under "5 · Weights", are easier to read. In '
                    'Sierra Nevada, the VIF kept 14 variables and the PCA with elevation 4, '
                    'with the same accuracy (F1 0.865 and 0.861); the PCA one mapped 38 % '
                    'more borreguil.\n\n'
                    '**Recommended:** train with each option and keep the one that '
                    'validates better (the comparison is at the end of "3 · Model").'))
            _vif_si, _pca_si = _metodo == 'vif', _metodo == 'pca'
            _vif_u, _vif_f = float(mr.VIF_UMBRAL), list(mr.VIF_FIJAS)
            # _sel_vig: lo aplicado con el botón, si sigue valiendo (si no, None)
            _pca_c, _sel_vig = float(mr.PCA_CARGA), None
            if _metodo != 'ninguno':
                # Los valores de partida se ponen justo antes de dibujar los controles
                # (si no, el navegador no los enseña); las fijas, solo las elegidas.
                st.session_state.setdefault('mrf_vif_umbral', float(mr.VIF_UMBRAL))
                st.session_state.setdefault('mrf_pca_carga', float(mr.PCA_CARGA))
                _fv = st.session_state.get('mrf_vif_fijas')
                if _fv is None:
                    st.session_state['mrf_vif_fijas'] = [
                        v for v in mr.VIF_FIJAS if v in _predictores]
                elif any(v not in _predictores for v in _fv):
                    st.session_state['mrf_vif_fijas'] = [v for v in _fv if v in _predictores]
                _vq1, _vq2 = st.columns([1, 2])
                if _vif_si:
                    _vif_u = _vq1.number_input(
                        i18n.pick('Umbral del VIF', 'VIF threshold'),
                        min_value=1.5, max_value=100.0, step=0.5, key='mrf_vif_umbral',
                        help=i18n.pick(
                            'Se quitan variables mientras alguna tenga un VIF mayor que '
                            'este. Con 10, ninguna de las que quedan se explica en más del '
                            '90 % con las demás; con 5, en más del 80 %. Cuanto más bajo, '
                            'más variables se quitan.\n\n'
                            '**Recomendado: 10**; 5 si quieres quedarte con menos.',
                            'Variables are removed while any has a VIF above this one. '
                            'With 10, none of those left is explained more than 90 % by '
                            'the others; with 5, more than 80 %. The lower it is, the more '
                            'variables are removed.\n\n'
                            '**Recommended: 10**; 5 if you want to keep fewer.'))
                else:
                    _pca_c = _vq1.number_input(
                        i18n.pick('Correlación mínima con el eje',
                                  'Minimum correlation with the axis'),
                        min_value=0.3, max_value=0.95, step=0.05, key='mrf_pca_carga',
                        help=i18n.pick(
                            'De cada eje se elige entre las variables cuya correlación con '
                            'él (en valor absoluto) llega a esta. Cuanto más alta, menos '
                            'candidatas y más claro qué representa cada eje.\n\n'
                            '**Recomendado: 0,65**, el de la comparación de Sierra Nevada.',
                            'From each axis, the choice is among the variables whose '
                            'correlation with it (in absolute value) reaches this one. The '
                            'higher it is, the fewer candidates and the clearer what each '
                            'axis stands for.\n\n'
                            '**Recommended: 0.65**, the one of the Sierra Nevada '
                            'comparison.'))
                _vif_f = _vq2.multiselect(
                    i18n.pick('Variables que se quedan siempre', 'Variables that always stay'),
                    _predictores, key='mrf_vif_fijas', format_func=_var,
                    help=i18n.pick(
                        'Estas no se quitan nunca (con el VIF, sí cuentan al calcular el de '
                        'las demás; con el PCA, quedan fuera de él y se añaden). Viene la '
                        'altitud: en cualquier sitio, los borreguiles son prados húmedos de '
                        'alta montaña.\n\n'
                        '**Recomendado: la altitud.**',
                        'These are never removed (with the VIF, they do count when '
                        'computing that of the others; with the PCA, they are left out of it '
                        'and added). Elevation comes selected: anywhere, borreguiles are wet '
                        'high-mountain meadows.\n\n'
                        '**Recommended: elevation.**'))
                # ---- Aplicar (25-11-2026, la autora: «que haya un botón de aplicar y te
                # muestre en esta misma pestaña qué variables se han quedado y algún
                # output del análisis para entender por qué»). Lo aplicado se guarda por
                # nivel y método, y vale mientras no cambien sus ajustes, las variables
                # elegidas ni las muestras (o sus valores). Si cambian, hay que volver a
                # aplicarlo; mientras, se usan todas las variables elegidas.
                _nom_met = (i18n.pick('el VIF', 'the VIF') if _vif_si
                            else i18n.pick('el PCA', 'the PCA'))
                # De qué variables parte: las que había marcadas antes de aplicarlo (si
                # ya se aplicó y siguen marcadas las que quedaron) o las marcadas ahora.
                _base = list(_antes['vars']) if _antes else list(_predictores)
                _tab_base = (None if _tab_todas is None else _tab_todas
                             if _base == list(_predictores)
                             else mr.construir_tabla(_res, _leidos, _base))
                _lista = bool(_tab_base is not None and _tab_base['listo']
                              and len(_tab_base['filas']) >= 3)
                _fijas_t = ([v for v in _vif_f if v in _tab_base['predictores']]
                            if _tab_base is not None else [])
                # Lo aplicado vale mientras no cambien el ajuste, las fijas, las variables
                # de partida, las muestras ni su lectura, y sigan marcadas las que quedaron.
                _firma = (_firma_sel(_metodo, _vif_u if _vif_si else _pca_c, _fijas_t,
                                     _tab_base) if _lista else None)
                _aplicadas = st.session_state.setdefault('mrf_sel_aplicada', {})
                _clave_ap = f'{_nivel}|{_metodo}'
                st.button(i18n.pick(f'✔ Aplicar {_nom_met}', f'✔ Apply {_nom_met}'),
                          key='mrf_aplicar_sel', type='primary', disabled=not _lista,
                          on_click=lambda: st.session_state.__setitem__('_mrf_aplicar', True),
                          help=i18n.pick(
                              'Calcula qué variables se quedan con los valores de las '
                              'muestras y deja marcadas arriba solo esas. El modelo, el '
                              'mapa y el peso usarán solo esas. Para volver a las de antes, '
                              '«↺ Volver a marcar…», debajo de las variables.',
                              'Works out which variables stay from the values of the '
                              'samples and leaves only those selected above. The model, the '
                              'map and the weights will use only those. To go back to the '
                              'previous ones, "↺ Select … again", below the variables.'))
                _ap = _aplicadas.get(_clave_ap)
                if (_ap and _firma is not None and _ap['firma'] == _firma
                        and mr.ordenar_predictores(_ap['sel']['quedan']) == _predictores):
                    _sel_vig = _ap
                _n_todas = len(_predictores)
                if _sel_vig is None:
                    if _tab_todas is None:
                        st.caption(i18n.pick(
                            'Para aplicarlo hacen falta los valores de las muestras: léelos '
                            'en «2 · Tabla» (botón «📥 Leer las variables…») y vuelve aquí. '
                            f'Mientras, se usan las {_n_todas} variables.',
                            'Applying it needs the values of the samples: read them under '
                            '"2 · Table" ("📥 Read the variables…" button) and come back '
                            f'here. Meanwhile, the {_n_todas} variables are used.'))
                    elif not _lista:
                        st.caption(i18n.pick(
                            'Se podrá aplicar cuando la tabla de entrenamiento esté lista '
                            f'(mira «2 · Tabla»). Mientras, se usan las {_n_todas} variables.',
                            'It can be applied once the training table is ready (see '
                            f'"2 · Table"). Meanwhile, the {_n_todas} variables are used.'))
                    elif _ap:
                        st.warning(i18n.pick(
                            'Has cambiado algo desde que lo aplicaste (las variables, el '
                            'ajuste, las que se quedan siempre o las muestras): vuelve a '
                            f'pulsar «✔ Aplicar {_nom_met}». Mientras, se usan las '
                            f'{_n_todas} variables.',
                            'Something has changed since you applied it (the variables, the '
                            'setting, those that always stay or the samples): press '
                            f'"✔ Apply {_nom_met}" again. Meanwhile, the {_n_todas} '
                            'variables are used.'))
                    else:
                        st.info(i18n.pick(
                            f'Pulsa «✔ Aplicar {_nom_met}» para ver qué variables se quedan '
                            f'y por qué. Mientras no lo apliques, se usan las {_n_todas} '
                            'variables.',
                            f'Press "✔ Apply {_nom_met}" to see which variables stay and '
                            f'why. Until you apply it, the {_n_todas} variables are used.'))
                else:
                    _mostrar_seleccion(_sel_vig, 'vif' if _vif_si else 'pca',
                                       _vif_u if _vif_si else _pca_c)
            with st.expander(i18n.pick('Qué mide cada variable', 'What each variable measures')):
                st.caption(i18n.pick(
                    'Las **estadísticas de jun – sep** (media, mínimo, máximo y '
                    'variabilidad) se calculan **en cada año** y después se hace la '
                    '**media entre años**. Así, el mínimo es lo bajo que llega la variable '
                    'en un verano normal, no el valor de una sola imagen (una sombra de '
                    'nube, por ejemplo) en todos los años.',
                    'The **Jun – Sep statistics** (mean, minimum, maximum and variability) '
                    'are computed **in each year** and then **averaged across years**. So '
                    'the minimum is how low the variable gets in a normal summer, not the '
                    'value of a single image (a cloud shadow, for instance) in all the '
                    'years.'))
                st.dataframe(pd.DataFrame([{
                    'Variable': _var(n),
                    i18n.pick('Qué mide', 'What it measures'): (en_ if _en else es_),
                    i18n.pick('Grupo', 'Group'): mr.nombre_grupo(g, _en),
                    i18n.pick('Seleccionada', 'Selected'):
                        i18n.pick('sí', 'yes') if n in _predictores else 'no',
                    i18n.pick('Nombre en los ficheros', 'Name in the files'): n}
                    for n, g, es_, en_ in mr.PREDICTORES]), hide_index=True, width='stretch')
            with st.expander(i18n.pick('Variables que usa el detector y aquí no están',
                                       'Variables the detector uses that are missing here')):
                for _q, _p, _q_en, _p_en in mr.NO_DISPONIBLES:
                    st.markdown(f'- **{i18n.pick(_q, _q_en)}** — {i18n.pick(_p, _p_en)}.')

        # ------------------------------------------ 2 · Tabla de entrenamiento
        # El valor de cada variable se lee en el PÍXEL de 10 m de cada muestra, de
        # la misma pila de Earth Engine que después se clasificará.
        with _t_tabla:
            st.caption(i18n.pick(
                'Lee en Earth Engine el valor de cada variable en el **píxel de 10 m** de '
                'cada muestra: el mismo píxel que después se clasificará. Por eso no se '
                'usan los valores que ya tienen los puntos, que son medias de una ventana '
                'de 50–60 m alrededor del punto.',
                'Reads from Earth Engine the value of each variable at the **10 m pixel** '
                'of each sample: the same pixel that will later be classified. That is why '
                'the values the points already have, which are means over a 50–60 m '
                'window around the point, are not used.'))
            # (lo ya leído de cada muestra, _lect/_leidos/_pend, se calcula en
            # «1 · Datos», que lo usa para el VIF y el PCA)
            import gee_backend as geb

            # El proyecto de Earth Engine es el del panel lateral (ya no se vuelve a pedir
            # aquí, 24-11-2026). Si allí está elegido Planetary Computer, el último con
            # el que Earth Engine funcionó. Vacío: la cuenta compartida de MonitorEO
            # (26-11-2026), que también vale para leer las variables y hacer el mapa.
            _proy = gee_project.strip() or leer_ajuste('gee_project', '')
            st.caption(i18n.pick(
                f'Años de Sentinel-2: **{years_str}** · Earth Engine: '
                + (f'proyecto **{_proy}**' if _proy else '**cuenta compartida de MonitorEO**')
                + ' (los dos, del panel lateral).',
                f'Sentinel-2 years: **{years_str}** · Earth Engine: '
                + (f'project **{_proy}**' if _proy else '**MonitorEO shared account**')
                + ' (both from the sidebar).'))
            _rm = st.session_state.pop('_mrf_msg', None)
            if _rm:
                (st.success if _rm[0] == 'ok' else st.warning)(_rm[1])

            _puede = bool(_res['listo'] and _predictores and _anios)
            if not _res['listo']:
                st.caption(i18n.pick(
                    'Antes hay que resolver lo que se indica en «1 · Datos».',
                    'First resolve what is flagged under "1 · Data".'))
            elif not _anios:
                st.error(i18n.pick('No entiendo los años del panel lateral.',
                                   'The years in the sidebar could not be understood.'))
            _leer = False
            if _pend:
                _leer = st.button(
                    i18n.pick(f'📥 Leer las variables de {len(_pend)} muestra'
                              f"{'' if len(_pend) == 1 else 's'} en Earth Engine",
                              f'📥 Read the variables of {len(_pend)} sample'
                              f"{'' if len(_pend) == 1 else 's'} from Earth Engine"),
                    key='mrf_leer', type='primary', disabled=not _puede)
            elif _usadas and _lect:
                if st.button(i18n.pick('↻ Volver a leer todas las muestras',
                                       '↻ Read all the samples again'), key='mrf_releer',
                             disabled=not _puede,
                             help=i18n.pick(
                                 'Solo hace falta si crees que una lectura salió mal: los '
                                 'valores no cambian mientras no cambien los años.',
                                 'Only needed if you think a reading went wrong: the '
                                 'values do not change unless the years do.')):
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
                        _barra = st.progress(0.0)
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
                st.markdown(i18n.pick(
                    f"**{_nt} muestra{'' if _nt == 1 else 's'} con todas sus variables** "
                    f"({len(_tab['predictores'])} variables).",
                    f"**{_nt} sample{'' if _nt == 1 else 's'} with all their variables** "
                    f"({len(_tab['predictores'])} variables)."))
                st.dataframe(pd.DataFrame([{
                    i18n.pick('Categoría', 'Category'): i18n.tv(c['categoria']),
                    i18n.pick('Código', 'Code'): c['codigo'],
                    i18n.pick('Muestras', 'Samples'): c['n_antes'],
                    i18n.pick('En la tabla', 'In the table'): c['n']}
                    for c in _tab['clases']]), hide_index=True, width='stretch')
                # El ID, con solo su número, como en «Revisar puntos» (25-11-2026, la
                # autora: «aquí también debería poner id 1,2,3»). El completo sigue en el
                # CSV que se descarga.
                _ids_tab = ([str(f['ID']) for f in _tab['filas']]
                            + [str(a['ID']) for a in _tab['apartadas']])
                _corto_tab = dict(zip(_ids_tab, bp.abreviar_ids(
                    bp.nombres_de_punto(_ids_tab, punto=''))))
                if _tab['apartadas']:
                    _cuenta = Counter(a['motivo'] for a in _tab['apartadas'])
                    st.warning(i18n.pick('**Muestras que no entran en la tabla:**',
                                         '**Samples left out of the table:**')
                               + ''.join(f'\n- {n} — {mr.motivo_tabla(k, _en)}'
                                         for k, n in _cuenta.most_common()))
                    with st.expander(i18n.pick('Ver cuáles son', 'See which ones')):
                        st.dataframe(pd.DataFrame([{
                            'ID': _corto_tab.get(str(a['ID']), str(a['ID'])),
                            i18n.pick('categoría', 'category'): i18n.tv(a['categoria']),
                            i18n.pick('motivo', 'reason'): mr.motivo_tabla(a['motivo'], _en),
                            i18n.pick('variables sin dato', 'variables without data'):
                                ', '.join(_var(x) for x in a['faltan'][:6])
                                + (' …' if len(a['faltan']) > 6 else '')}
                            for a in _tab['apartadas']]), hide_index=True, width='stretch')
                if _tab['errores']:
                    st.error(i18n.pick('**Con esta tabla todavía no se puede entrenar:**',
                                       '**This table is not enough to train yet:**')
                             + ''.join('\n- ' + mr.mensaje(e, _en, i18n.tv, nivel=_nivel_mod)
                                       for e in _tab['errores']))

                # Control de calidad de las variables: avisa, no quita nada
                _hall = mr.calidad(_tab)
                if _hall:
                    _max = 10
                    st.warning(i18n.pick(
                        '**Control de calidad de las variables.** No se ha quitado ninguna: '
                        'si quieres prescindir de alguna, desmárcala en «1 · Datos».',
                        '**Quality check of the variables.** None has been removed: if you '
                        'want to drop one, deselect it under "1 · Data".')
                        + ''.join('\n- ' + mr.mensaje_calidad(h, _en, nombre=_var, num=_num)
                                  for h in _hall[:_max])
                        + (i18n.pick(f'\n- … y {len(_hall) - _max} más.',
                                     f'\n- … and {len(_hall) - _max} more.')
                           if len(_hall) > _max else ''))
                elif _nt:
                    st.success(i18n.pick(
                        'Control de calidad de las variables: sin incidencias.',
                        'Quality check of the variables: nothing to report.'))

                # Balance entre categorías
                _des = mr.desequilibrio(_tab['clases'])
                if _des and _des[0] >= 3:
                    st.info(i18n.pick(
                        f"**Categorías desequilibradas:** «{i18n.tv(_des[1]['categoria'])}» "
                        f"tiene {_des[1]['n']} muestras y «{i18n.tv(_des[2]['categoria'])}», "
                        f"{_des[2]['n']} ({_num(_des[0])} veces menos). Al entrenar se "
                        'compensará dando más peso a las categorías escasas.',
                        f"**Unbalanced categories:** \"{i18n.tv(_des[1]['categoria'])}\" has "
                        f"{_des[1]['n']} samples and \"{i18n.tv(_des[2]['categoria'])}\" has "
                        f"{_des[2]['n']} ({_num(_des[0])} times fewer). Training will "
                        'compensate by giving more weight to the scarce categories.'))

                with st.expander(i18n.pick('Ver la tabla de entrenamiento',
                                           'See the training table')):
                    # Qué variables salen, según el caso: todas, las que dejó el VIF o el
                    # PCA, o las marcadas a mano (25-11-2026, la autora: «y en la otra que
                    # muestra las 34, las del vif, o las del pca pues que en algún sitio lo
                    # diga según el caso»).
                    _n_pred, _n_todas = len(_tab['predictores']), len(mr.PREDICTORES)
                    _ant_t = st.session_state.get('mrf_vars_antes')
                    if _ant_t and set(_tab['predictores']) == set(_ant_t['resultado']):
                        _met_t = (i18n.pick('el VIF', 'the VIF') if _ant_t['metodo'] == 'vif'
                                  else i18n.pick('el PCA', 'the PCA'))
                        _cuales = i18n.pick(
                            f"Salen las **{_n_pred} variables que dejó {_met_t}**, de las "
                            f"{len(_ant_t['vars'])} que había marcadas (se aplicó en "
                            '«1 · Datos»).',
                            f"These are the **{_n_pred} variables {_met_t} kept**, out of the "
                            f"{len(_ant_t['vars'])} that were selected (applied under "
                            '"1 · Data").')
                    elif _n_pred == _n_todas:
                        _cuales = i18n.pick(
                            f'Salen **las {_n_todas} variables**: no se ha quitado ninguna, ni '
                            'con el VIF ni con el PCA.',
                            f'These are **all {_n_todas} variables**: none has been removed, '
                            'neither with the VIF nor with the PCA.')
                    else:
                        _cuales = i18n.pick(
                            f'Salen las **{_n_pred} variables que tienes marcadas** en «1 · '
                            f'Datos», de las {_n_todas} que hay. No las ha elegido el VIF ni '
                            'el PCA.',
                            f'These are the **{_n_pred} variables you have selected** under '
                            f'"1 · Data", out of the {_n_todas} there are. Neither the VIF nor '
                            'the PCA chose them.')
                    st.caption(_cuales + i18n.pick(
                        ' Cada fila es una muestra: el número del punto, su categoría, el '
                        'código que aprende el modelo y el valor de cada variable en su píxel '
                        'de 10 m. Es justo lo que recibe el Random Forest; las coordenadas no '
                        'van, porque no son variables.',
                        ' Each row is a sample: the point number, its category, the code the '
                        'model learns and the value of each variable in its 10 m pixel. It is '
                        'exactly what the Random Forest receives; the coordinates are left '
                        'out, because they are not variables.'))
                    _dft = pd.DataFrame(_tab['filas'])
                    if len(_dft):
                        # solo lo que es la tabla X/y: las coordenadas no son variables
                        _dft = _dft.drop(columns=['indice', 'lon', 'lat'])
                        _dft['ID'] = [_corto_tab.get(str(i), str(i)) for i in _dft['ID']]
                        _dft['categoria'] = _dft['categoria'].map(i18n.tv)
                        st.dataframe(_con_coma(_dft.rename(columns={
                            **{v: _var(v) for v in _tab['predictores']},
                            'categoria': i18n.pick('categoría', 'category'),
                            'codigo': i18n.pick('código', 'code')})),
                            hide_index=True, width='stretch',
                            column_config={'ID': st.column_config.Column(pinned=True)})
                    st.download_button(
                        label=f'⬇ tabla_entrenamiento_{_nivel}.csv',
                        data=mr.tabla_entrenamiento_csv(_tab).encode('utf-8-sig'),
                        file_name=f'tabla_entrenamiento_{_nivel}.csv', mime='text/csv',
                        key='mrf_dl_tabla')
                _inf = (_lect or {}).get('info') or {}
                if _inf:
                    st.caption(i18n.pick(
                        f"Leído el {(_lect.get('cuando') or '').replace('T', ' ')} · píxeles "
                        f"de {_inf.get('resolucion_m')} m · años "
                        f"{bp.abreviar_anios(_inf.get('anios', []))} · lo de junio a "
                        'septiembre se calcula en cada año y después se hace la media entre '
                        'años · el relieve viene a 30 m y se suaviza a 10 m.',
                        f"Read on {(_lect.get('cuando') or '').replace('T', ' ')} · "
                        f"{_inf.get('resolucion_m')} m pixels · years "
                        f"{bp.abreviar_anios(_inf.get('anios', []))} · the June to September "
                        'figures are computed in each year and then averaged across years · '
                        'the relief comes at 30 m and is smoothed to 10 m.'))

                # ---- Variables correlacionadas (VIF o PCA). Se eligen y se aplican en
                # «1 · Datos», que enseña cuáles se quedan y por qué (25-11-2026); aquí se
                # sigue con la tabla de las que se quedan, que es la que usan el modelo, el
                # mapa y el peso.
                if _metodo == 'ninguno':
                    st.caption(i18n.pick(
                        'Para quitar las variables correlacionadas, elige el VIF o un PCA en '
                        '«1 · Datos», debajo de las variables.',
                        'To remove correlated variables, choose the VIF or a PCA under '
                        '"1 · Data", below the variables.'))
                elif _sel_vig is None:
                    st.info(i18n.pick(
                        f'Has elegido quitar las variables correlacionadas con {_nom_met}, pero no '
                        f"está aplicado: se usan las {len(_tab['predictores'])} variables. "
                        f'Para aplicarlo, pulsa «✔ Aplicar {_nom_met}» en «1 · Datos».',
                        f'You chose to remove correlated variables with {_nom_met}, but it is '
                        f"not applied: the {len(_tab['predictores'])} variables are used. To "
                        f'apply it, press "✔ Apply {_nom_met}" under "1 · Data".'))
                else:
                    _s_ap = _sel_vig['sel']
                    _m_ap = 'VIF' if _vif_si else 'PCA'
                    st.markdown(i18n.pick(
                        f"**Variables correlacionadas ({_m_ap}):** se quedan "
                        f"{len(_s_ap['quedan'])} de {_sel_vig['de']}; cuáles y por qué, en "
                        '«1 · Datos». El modelo, el mapa y el peso usan solo esas.',
                        f"**Correlated variables ({_m_ap}):** {len(_s_ap['quedan'])} of "
                        f"{_sel_vig['de']} stay; which ones and why, under \"1 · Data\". "
                        'The model, the map and the weights use only those.'))
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
        # distintos: las métricas salen del primero; el mapa, del segundo.
        with _t_modelo:
            st.caption(i18n.pick(
                'Entrena un Random Forest con la tabla y lo **valida por zonas**: cada '
                'zona (una cuenca, o una celda de unos 5 km) se predice con un modelo que '
                'no ha visto ningún punto de ella. Después entrena el **modelo final** con '
                'todas las muestras, que es el que hará el mapa.',
                'Trains a Random Forest with the table and **validates it by zones**: each '
                'zone (a basin, or a cell of about 5 km) is predicted by a model that has '
                'seen no point from it. It then trains the **final model** with all the '
                'samples, which is the one that will make the map.'))
            _est_txt = {
                'pesos': i18n.pick('Dar más peso a las categorías escasas (recomendado)',
                                   'Give more weight to the scarce categories (recommended)'),
                'submuestreo': i18n.pick('Usar menos muestras de las abundantes',
                                         'Use fewer samples of the abundant ones'),
                'ambas': i18n.pick('Las dos cosas', 'Both'),
                'ninguna': i18n.pick('No compensar', 'Do not compensate'),
            }
            _est_corto = {
                'pesos': i18n.pick('más peso a las categorías escasas',
                                   'more weight to the scarce categories'),
                'submuestreo': i18n.pick('menos muestras de las abundantes',
                                         'fewer samples of the abundant ones'),
                'ambas': i18n.pick('más peso a las escasas y menos muestras de las '
                                   'abundantes', 'more weight to the scarce ones and fewer '
                                   'samples of the abundant ones'),
                'ninguna': i18n.pick('sin compensar el desequilibrio',
                                     'without compensating the imbalance'),
            }
            # Cortas: van en media anchura y, si no caben, Streamlit las corta con «…».
            # Que son «del nº de variables» lo dice la ayuda, con la cuenta hecha.
            _mf_txt = {
                'sqrt': i18n.pick('raíz cuadrada (recomendado)', 'square root (recommended)'),
                'log2': i18n.pick('logaritmo', 'logarithm'),
                'todas': i18n.pick('todas', 'all'),
            }
            # Qué es cada parámetro y qué valor conviene: va en la ayuda (?) de cada
            # uno, para poder decidir sin salir de la app.
            _nv = len(_predictores)
            _mf_raiz, _mf_log = max(1, int(_nv ** 0.5)), max(1, _nv.bit_length() - 1)
            if not _nv:
                _mf_cuenta = ''
            elif _mf_raiz == _mf_log:
                _mf_cuenta = i18n.pick(
                    f'Con las {_nv} variables elegidas ahora, raíz cuadrada y logaritmo '
                    f'prueban las dos {_mf_raiz} en cada división; «todas», las {_nv}.',
                    f'With the {_nv} variables selected now, square root and logarithm '
                    f'both try {_mf_raiz} at each split; "all" tries the {_nv}.')
            else:
                _mf_cuenta = i18n.pick(
                    f'Con las {_nv} variables elegidas ahora, raíz cuadrada prueba '
                    f'{_mf_raiz} en cada división; logaritmo, {_mf_log}; «todas», las '
                    f'{_nv}.',
                    f'With the {_nv} variables selected now, square root tries '
                    f'{_mf_raiz} at each split; logarithm, {_mf_log}; "all", the {_nv}.')
            with st.expander(i18n.pick('Parámetros del modelo', 'Model parameters')):
                st.caption(i18n.pick(
                    'Para empezar no hace falta tocar nada: los valores que vienen puestos '
                    'son los recomendados. El **?** de cada parámetro explica qué es y qué '
                    'valor conviene.',
                    'There is no need to change anything to get started: the preset '
                    'values are the recommended ones. The **?** next to each parameter '
                    'explains what it is and which value to use.'))
                _pa, _pb = st.columns(2)
                with _pa:
                    _p_arboles = st.number_input(
                        i18n.pick('Nº de árboles', 'Nº of trees'), 50, 2000,
                        mr.PARAMS_DEFECTO['n_estimators'], 50, key='mrf_p_arboles',
                        help=i18n.pick(
                            'El modelo es un conjunto de árboles de decisión que votan. '
                            'Con más árboles el resultado es más estable (cambia menos de '
                            'una ejecución a otra), pero tarda más; a partir de unos '
                            'cientos apenas mejora.\n\n'
                            '**Recomendado: 500.** Entre 300 y 1000 va bien. Con menos de '
                            '100 las cifras bailan.',
                            'The model is a set of decision trees that vote. More trees '
                            'give a steadier result (it changes less from one run to the '
                            'next) but take longer; beyond a few hundred it hardly '
                            'improves.\n\n'
                            '**Recommended: 500.** Between 300 and 1000 works well. Below '
                            '100 the figures wobble.'))
                    _p_prof = st.number_input(
                        i18n.pick('Profundidad máxima (0 = sin límite)',
                                  'Maximum depth (0 = no limit)'), 0, 50, 0, 1,
                        key='mrf_p_prof',
                        help=i18n.pick(
                            'Cuántas preguntas seguidas puede hacer cada árbol antes de '
                            'decidir. Sin límite, el árbol crece hasta separar del todo las '
                            'muestras, que es lo habitual en un Random Forest.\n\n'
                            '**Recomendado: 0 (sin límite).** Ponle un límite, entre 5 y '
                            '10, solo si tienes muy pocas muestras y la validación sale '
                            'mal: los árboles más cortos se fijan menos en los casos '
                            'sueltos.',
                            'How many questions in a row each tree may ask before '
                            'deciding. With no limit, the tree grows until it separates '
                            'the samples completely, which is the usual way in a Random '
                            'Forest.\n\n'
                            '**Recommended: 0 (no limit).** Set a limit, between 5 and 10, '
                            'only if you have very few samples and the validation comes '
                            'out badly: shorter trees pay less attention to odd cases.'))
                    _p_semilla = st.number_input(
                        i18n.pick('Semilla', 'Seed'), 0, 99999,
                        mr.PARAMS_DEFECTO['random_state'], 1, key='mrf_p_semilla',
                        help=i18n.pick(
                            'El modelo usa el azar para elegir qué muestras y qué '
                            'variables ve cada árbol. La semilla fija ese azar: con la '
                            'misma semilla y los mismos datos sale exactamente el mismo '
                            'modelo, y otra persona puede repetirlo.\n\n'
                            '**Recomendado: déjala en 42** y anótala con tus resultados. '
                            'Ningún número es mejor que otro. Si al cambiarla las cifras '
                            'cambian mucho, es que hay pocas muestras.',
                            'The model uses chance to choose which samples and which '
                            'variables each tree sees. The seed fixes that chance: the '
                            'same seed and the same data give exactly the same model, and '
                            'someone else can repeat it.\n\n'
                            '**Recommended: leave it at 42** and note it down with your '
                            'results. No number is better than another. If the figures '
                            'change a lot when you change it, there are too few samples.'))
                with _pb:
                    _p_hoja = st.number_input(
                        i18n.pick('Muestras mínimas por hoja', 'Minimum samples per leaf'),
                        1, 20, mr.PARAMS_DEFECTO['min_samples_leaf'], 1, key='mrf_p_hoja',
                        help=i18n.pick(
                            'Cada árbol va repartiendo las muestras en grupos cada vez más '
                            'pequeños. Este es el tamaño mínimo del grupo final («hoja»). '
                            'Con 1, el árbol puede aprenderse de memoria cada muestra, '
                            'también las raras o mal etiquetadas; al subirlo, generaliza '
                            'más.\n\n'
                            '**Recomendado: 2** si tienes decenas de muestras por '
                            'categoría; **entre 3 y 5** si tienes cientos. Si lo subes '
                            'mucho, las categorías con pocas muestras dejan de '
                            'distinguirse.',
                            'Each tree keeps splitting the samples into smaller and '
                            'smaller groups. This is the minimum size of the final group '
                            '("leaf"). With 1, the tree can learn each sample by heart, '
                            'including odd or mislabelled ones; raising it makes the tree '
                            'generalise more.\n\n'
                            '**Recommended: 2** if you have tens of samples per category; '
                            '**between 3 and 5** if you have hundreds. If you raise it a '
                            'lot, categories with few samples can no longer be told '
                            'apart.'))
                    _p_mf = st.selectbox(
                        i18n.pick('Variables probadas en cada división',
                                  'Variables tried at each split'),
                        list(_mf_txt), key='mrf_p_maxfeat', format_func=lambda o: _mf_txt[o],
                        help=i18n.pick(
                            'En cada división, el árbol no mira todas las variables: '
                            'prueba solo unas cuantas, elegidas al azar, y se queda con la '
                            'mejor. Así los árboles salen distintos entre sí, que es lo '
                            'que hace que juntos acierten más.\n\n'
                            '**Recomendado: raíz cuadrada**, lo habitual para clasificar. ',
                            'At each split the tree does not look at every variable: it '
                            'tries only a few, chosen at random, and keeps the best. That '
                            'makes the trees differ from one another, which is what makes '
                            'them more accurate together.\n\n'
                            '**Recommended: square root**, the usual choice for '
                            'classification. ')
                        + _mf_cuenta + i18n.pick(
                            ' Con «todas», los árboles se parecen mucho entre sí y el '
                            'resultado suele ser peor.',
                            ' With "all", the trees look very much alike and the result '
                            'is usually worse.'))
                _p_bal = st.selectbox(
                    i18n.pick('Si unas categorías tienen muchas más muestras que otras',
                              'If some categories have many more samples than others'),
                    list(mr.ESTRATEGIAS), key='mrf_p_balance',
                    format_func=lambda o: _est_txt[o],
                    help=i18n.pick(
                        'Si una categoría tiene muchas más muestras que las demás, el '
                        'modelo tiende a elegirla siempre.\n\n'
                        '- **Dar más peso a las escasas (recomendado):** cada muestra de '
                        'una categoría escasa cuenta más. No se pierde ninguna.\n'
                        '- **Usar menos muestras de las abundantes:** se aparta al azar '
                        'parte de ellas. Útil si una categoría tiene muchísimas más que '
                        'el resto.\n'
                        '- **Las dos cosas:** para desequilibrios muy fuertes.\n'
                        '- **No compensar:** solo si todas tienen un número parecido de '
                        'muestras.\n\n'
                        'No se generan muestras artificiales en ningún caso, y se valida '
                        'siempre con todas.',
                        'If one category has many more samples than the others, the '
                        'model tends to choose it every time.\n\n'
                        '- **Give more weight to the scarce ones (recommended):** each '
                        'sample of a scarce category counts more. None is lost.\n'
                        '- **Use fewer samples of the abundant ones:** part of them is '
                        'set aside at random. Useful if one category has far more than '
                        'the rest.\n'
                        '- **Both:** for very strong imbalances.\n'
                        '- **Do not compensate:** only if all of them have a similar '
                        'number of samples.\n\n'
                        'No artificial samples are generated in any case, and validation '
                        'always uses all of them.'))
                _p_razon = mr.RAZON_MAXIMA
                if _p_bal in ('submuestreo', 'ambas'):
                    _p_razon = st.number_input(
                        i18n.pick('Como mucho, cuántas veces la categoría más escasa',
                                  'At most, how many times the scarcest category'),
                        1.0, 10.0, mr.RAZON_MAXIMA, 0.5, key='mrf_p_razon',
                        help=i18n.pick(
                            'Tope de muestras de cada categoría abundante, como múltiplo '
                            'de la más escasa. Con 3 y una categoría escasa de 10 '
                            'muestras, de las demás se usan como mucho 30, elegidas al '
                            'azar con la semilla.\n\n'
                            '**Recomendado: 3.** Con 1 quedan todas iguales, pero se '
                            'desaprovechan muchas muestras.',
                            'Cap on the samples of each abundant category, as a multiple '
                            'of the scarcest one. With 3 and a scarce category of 10 '
                            'samples, at most 30 of each of the others are used, chosen '
                            'at random with the seed.\n\n'
                            '**Recommended: 3.** With 1 they all end up equal, but many '
                            'samples go unused.'))
            _params = {'n_estimators': int(_p_arboles),
                       'max_features': None if _p_mf == 'todas' else _p_mf,
                       'max_depth': int(_p_prof) or None,
                       'min_samples_leaf': int(_p_hoja), 'random_state': int(_p_semilla)}

            _listo_m = bool(_tab and _tab['listo'])
            if not _listo_m:
                st.caption(i18n.pick(
                    'Antes hace falta una tabla de entrenamiento válida («2 · Tabla de '
                    'entrenamiento»).',
                    'A valid training table is needed first ("2 · Training table").'))
            if st.button(i18n.pick('🌲 Entrenar y validar', '🌲 Train and validate'),
                         key='mrf_entrenar', type='primary', disabled=not _listo_m):
                with st.spinner(i18n.pick('Validando por zonas y entrenando el modelo final…',
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
            if _mod:
                if _tab is None or _mod['firma'] != mr.firma_tabla(_tab):
                    st.warning(i18n.pick(
                        'La tabla ha cambiado desde que se entrenó este modelo (otras '
                        'muestras u otras variables). Lo de abajo corresponde al modelo '
                        'anterior: vuelve a entrenar para ponerlo al día.',
                        'The table has changed since this model was trained (other samples '
                        'or other variables). What follows belongs to the previous model: '
                        'train again to bring it up to date.'))
                _val, _zi = _mod['validacion'], _mod.get('zonas', {})
                if _val['tipo'] == 'espacial':
                    st.markdown(i18n.pick(
                        f"**Validación por zonas**: {_val['particiones']} partes sobre "
                        f"{_val['n_zonas']} zonas ({_zi.get('con_cuenca', 0)} muestras con "
                        f"cuenca, {_zi.get('con_celda', 0)} por celda de ~5 km).",
                        f"**Validation by zones**: {_val['particiones']} folds over "
                        f"{_val['n_zonas']} zones ({_zi.get('con_cuenca', 0)} samples with a "
                        f"basin, {_zi.get('con_celda', 0)} by ~5 km cell)."))
                else:
                    st.markdown(i18n.pick(
                        f"**Validación sin zonas** ({_val['particiones']} partes al azar, "
                        'respetando la proporción de cada categoría).',
                        f"**Validation without zones** ({_val['particiones']} random folds, "
                        'keeping the proportion of each category).'))
                if _val['avisos']:
                    st.warning(''.join('- ' + mr.mensaje_validacion(a, _en, i18n.tv) + '\n'
                                       for a in _val['avisos']))

                # ---- ¿borreguil o no? La regla y su umbral. Va aquí, antes de las
                # cifras, porque de ella salen las de debajo, el mapa y el peso de las
                # variables. Cambiarla no vuelve a entrenar nada.
                _regla = _mod.get('regla') or {}
                if '_validacion' not in _mod:
                    st.info(i18n.pick(
                        'Este modelo se entrenó antes de que la app decidiera «¿borreguil '
                        'o no?» con un umbral. Pulsa otra vez **Entrenar y validar**.',
                        'This model was trained before the app decided "borreguil or '
                        'not?" with a threshold. Press **Train and validate** again.'))
                elif _regla.get('umbral') is None:
                    st.caption(i18n.pick(
                        'Este modelo no tiene la categoría «no borreguil», así que no '
                        'decide si algo es borreguil o no: a cada caso le pone su tipo más '
                        'probable.',
                        'This model has no "not a borreguil" category, so it does not '
                        'decide whether something is a borreguil or not: it gives each '
                        'case its most probable type.'))
                else:
                    st.markdown(i18n.pick('**¿Borreguil o no?**', '**Borreguil or not?**'))
                    _crit_txt = {
                        'f1': i18n.pick('El de mayor F1 (recomendado)',
                                        'The one with the highest F1 (recommended)'),
                        'tss': i18n.pick('El que mejor separa en la validación, TSS',
                                         'The one that separates best in the validation, '
                                         'TSS'),
                        'manual': i18n.pick('Lo fijo yo', 'I set it myself')}
                    _crit = st.radio(
                        i18n.pick('Cómo se elige el umbral', 'How the threshold is chosen'),
                        list(mr.CRITERIOS), key='mrf_umbral_modo',
                        format_func=lambda c: _crit_txt[c],
                        help=i18n.pick(
                            'El modelo da a cada píxel la probabilidad de cada categoría. '
                            'Para decidir si es borreguil se **suman las de los tipos** (es '
                            'lo mismo que 1 − la de «no borreguil») y se compara con un '
                            '**umbral**. Si llega, es borreguil, y después se elige su '
                            'tipo.\n\n'
                            '- **El de mayor F1:** de 39 umbrales, de 0.025 a 0.975, el '
                            'que mejor equilibra en la validación encontrar borreguiles y '
                            'no dar por borreguil lo que no lo es.\n'
                            '- **El que mejor separa (TSS):** el que da más TSS en la '
                            'validación. El TSS es sensibilidad + especificidad − 1: lo que '
                            'acierta con los borreguiles más lo que acierta con lo que no lo '
                            'es. Como hay mucho más «no borreguil» que borreguil, suele '
                            'elegir un umbral bajo, y el mapa se llena de pasto que no es '
                            'borreguil.\n'
                            '- **Lo fijo yo:** el valor que pongas.\n\n'
                            '**Recomendado: el de mayor F1.** Si varios umbrales empatan, '
                            'se toma el del centro.',
                            'The model gives each pixel the probability of each category. '
                            'To decide whether it is a borreguil, **those of the types are '
                            'added up** (the same as 1 − that of "not a borreguil") and '
                            'compared with a **threshold**. If it reaches it, it is a '
                            'borreguil, and its type is then chosen.\n\n'
                            '- **The one with the highest F1:** out of 39 thresholds, from '
                            '0.025 to 0.975, the one that in the validation best balances '
                            'finding borreguiles and not calling borreguil what is not.\n'
                            '- **The one that separates best (TSS):** the one with the '
                            'highest TSS in the validation. TSS is sensitivity + '
                            'specificity − 1: what it gets right among borreguiles plus '
                            'what it gets right among the rest. As there is far more "not a '
                            'borreguil" than borreguil, it tends to pick a low threshold, '
                            'and the map fills up with grassland that is not borreguil.\n'
                            '- **I set it myself:** the value you enter.\n\n'
                            '**Recommended: the one with the highest F1.** If several '
                            'thresholds tie, the middle one is taken.'))
                    _u_manual = None
                    if _crit == 'manual':
                        _u_manual = float(st.slider(
                            i18n.pick('Umbral', 'Threshold'), 0.05, 0.95, 0.5, 0.025,
                            key='mrf_umbral_valor', format='%.3f',
                            help=i18n.pick(
                                'Probabilidad de ser borreguil a partir de la cual un '
                                'píxel se da por borreguil. Más bajo: se encuentran más '
                                'borreguiles y se cuelan más cosas que no lo son. Más '
                                'alto: al revés.\n\n'
                                '**Recomendado:** mira antes la gráfica de debajo y elige '
                                'un valor donde el F1 sea alto. 0.5 es «más probable que '
                                'sí que que no».',
                                'Probability of being a borreguil from which a pixel '
                                'counts as one. Lower: more borreguiles are found and more '
                                'things that are not get through. Higher: the opposite.\n\n'
                                '**Recommended:** look at the chart below first and choose '
                                'a value where F1 is high. 0.5 is "more likely than '
                                'not".')))
                    if _regla.get('criterio') != _crit or (
                            _crit == 'manual' and abs(_regla['umbral'] - _u_manual) > 1e-9):
                        mr.aplicar_regla(_mod, _crit, _u_manual)
                        _regla = _mod['regla']
                    _eu, _u_txt = _regla['en_umbral'], f"{_num(_regla['umbral'])}"
                    _su_tipo = i18n.pick('su tipo es el más probable',
                                         'its type is then the most probable one')
                    if _nivel == mr.NIVEL_SI_NO:
                        _cuando = i18n.pick(
                            f'su probabilidad de serlo llega a **{_u_txt}**',
                            f'its probability of being one reaches **{_u_txt}**')
                    else:
                        _cuando = i18n.pick(
                            'la suma de la probabilidad de sus tipos llega a '
                            f'**{_u_txt}**; entonces {_su_tipo}',
                            f'the probabilities of its types add up to **{_u_txt}** or '
                            f'more; {_su_tipo}')
                    st.markdown(i18n.pick(
                        f'Un píxel se da por **borreguil** si {_cuando}. '
                        'Con ese umbral, en la validación:\n\n'
                        f"- encuentra **{_eu['vp']} de {_eu['vp'] + _eu['fn']}** "
                        f"borreguiles (sensibilidad {_num(_eu['sensibilidad'])});\n"
                        f"- descarta bien **{_eu['vn']} de {_eu['vn'] + _eu['fp']}** de lo "
                        f"que no lo es (especificidad {_num(_eu['especificidad'])});\n"
                        f"- TSS **{_num(_eu['tss'])}** · F1 de borreguil "
                        f"{_num(_eu['f1'])}.",
                        f'A pixel counts as a **borreguil** if {_cuando}. '
                        'With that threshold, in the validation:\n\n'
                        f"- it finds **{_eu['vp']} of {_eu['vp'] + _eu['fn']}** borreguiles "
                        f"(sensitivity {_num(_eu['sensibilidad'])});\n"
                        f"- it correctly rejects **{_eu['vn']} of {_eu['vn'] + _eu['fp']}** "
                        f"of what is not one (specificity {_num(_eu['especificidad'])});\n"
                        f"- TSS **{_num(_eu['tss'])}** · F1 of borreguil "
                        f"{_num(_eu['f1'])}."))
                    if _crit != 'manual':
                        st.caption(i18n.pick(
                            'El umbral se ha elegido con estas mismas muestras, así que '
                            'estas cifras salen algo optimistas.',
                            'The threshold was chosen with these same samples, so these '
                            'figures come out somewhat optimistic.'))
                    with st.expander(i18n.pick('Ver cómo cambia el acierto con el umbral',
                                               'See how accuracy changes with the threshold')):
                        import altair as alt
                        _bar = _regla['barrido']
                        try:
                            _oscuro = st.context.theme.type == 'dark'
                        except Exception:
                            _oscuro = False
                        # dos series, dos colores fijos (TSS, F1); la tinta, para el umbral
                        _c_tss, _c_f1, _tinta = (('#3987e5', '#d95926', '#c3c2b7') if _oscuro
                                                 else ('#2a78d6', '#eb6834', '#52514e'))
                        _k_um = i18n.pick('umbral', 'threshold')
                        _larga = pd.DataFrame(
                            [{'umbral': b['umbral'], 'medida': m, 'valor': b[k]}
                             for b in _bar for m, k in (('TSS', 'tss'), ('F1', 'f1'))])
                        _ancha = pd.DataFrame([{
                            'umbral': b['umbral'], 'TSS': b['tss'], 'F1': b['f1'],
                            'sens': b['sensibilidad'], 'espec': b['especificidad']}
                            for b in _bar])
                        _ex = alt.X('umbral:Q', title=_k_um, scale=alt.Scale(domain=[0, 1]))
                        _lineas = alt.Chart(_larga).mark_line(strokeWidth=2).encode(
                            x=_ex, y=alt.Y('valor:Q', title=None),
                            color=alt.Color('medida:N', scale=alt.Scale(
                                domain=['TSS', 'F1'], range=[_c_tss, _c_f1]),
                                legend=alt.Legend(title=None, orient='top')))
                        _elegido = pd.DataFrame({'umbral': [_regla['umbral']],
                                                 'rotulo': [f'{_k_um} {_u_txt}']})
                        _raya = alt.Chart(_elegido).mark_rule(
                            color=_tinta, strokeWidth=1.5).encode(x=_ex)
                        # el rótulo, abajo: arriba pasan las líneas
                        _letra = alt.Chart(_elegido).mark_text(
                            color=_tinta, align='left', dx=5, dy=-5,
                            baseline='bottom').encode(
                            x=_ex, y=alt.value('height'), text='rotulo:N')
                        # Al pasar el ratón: el umbral más cercano, con todas sus cifras.
                        _cerca = alt.selection_point(
                            name='umbral_cerca', nearest=True, on='pointerover',
                            clear='pointerout', fields=['umbral'], empty=False)
                        _puntos = alt.Chart(_ancha).mark_rule(
                            color=_tinta, strokeWidth=1).encode(
                            x=_ex, opacity=alt.condition(_cerca, alt.value(0.6),
                                                         alt.value(0.0)),
                            tooltip=[
                                alt.Tooltip('umbral:Q', title=_k_um, format='.3f'),
                                alt.Tooltip('TSS:Q', format='.3f'),
                                alt.Tooltip('F1:Q', format='.3f'),
                                alt.Tooltip('sens:Q', format='.3f', title=i18n.pick(
                                    'sensibilidad', 'sensitivity')),
                                alt.Tooltip('espec:Q', format='.3f', title=i18n.pick(
                                    'especificidad', 'specificity'))]
                        ).add_params(_cerca)
                        st.altair_chart(
                            alt.layer(_lineas, _raya, _letra, _puntos).properties(height=260),
                            width='stretch')
                        st.caption(i18n.pick(
                            'Cada punto de las líneas es lo que saldría en la validación '
                            'con ese umbral. La raya vertical es el que se está usando.',
                            'Each point of the lines is what the validation would give '
                            'with that threshold. The vertical line is the one in use.'))
                        _c4u = st.column_config.NumberColumn()
                        _k_se = i18n.pick('Sensibilidad', 'Sensitivity')
                        _k_es = i18n.pick('Especificidad', 'Specificity')
                        _k_pr = i18n.pick('Precisión', 'Precision')
                        st.dataframe(_con_coma(pd.DataFrame([{
                            i18n.pick('Umbral', 'Threshold'): b['umbral'],
                            'TSS': b['tss'], 'F1': b['f1'], _k_se: b['sensibilidad'],
                            _k_es: b['especificidad'], _k_pr: b['precision'],
                            i18n.pick('Borreguiles encontrados', 'Borreguiles found'): b['vp'],
                            i18n.pick('Borreguiles perdidos', 'Borreguiles missed'): b['fn'],
                            i18n.pick('Dados por borreguil sin serlo',
                                      'Wrongly called borreguil'): b['fp'],
                            i18n.pick('Bien descartados', 'Correctly rejected'): b['vn']}
                            for b in _bar])), hide_index=True, width='stretch',
                            column_config={
                                i18n.pick('Umbral', 'Threshold'):
                                    st.column_config.NumberColumn(pinned=True),
                                'TSS': _c4u, 'F1': _c4u, _k_se: _c4u, _k_es: _c4u,
                                _k_pr: _c4u})

                if _cercania:
                    st.caption(i18n.pick(
                        'Estas cifras son de lo que hace el modelo: decidir si es borreguil '
                        'y si es **húmedo o seco**. El **ambiente** (arroyo, laguna o '
                        'ladera) no entra aquí: no lo decide el modelo, lo pone el mapa '
                        'por cercanía a lagunas y cauces (**4 · Mapa**).',
                        'These figures are for what the model does: deciding whether it is '
                        'a borreguil and whether it is **wet or dry**. The **environment** '
                        '(stream, lake or slope) is not part of them: the model does not '
                        'decide it, the map sets it by closeness to lakes and '
                        'watercourses (**4 · Map**).'))

                _m1, _m2, _m3 = st.columns(3, vertical_alignment='bottom')
                _m1.metric(i18n.pick('F1 medio entre categorías (macro-F1)',
                                     'Mean F1 across categories (macro-F1)'),
                           _num(_mod['macro_f1']))
                _m2.metric(i18n.pick('Exactitud equilibrada', 'Balanced accuracy'),
                           _num(_mod['exactitud_equilibrada']))
                _m3.metric(i18n.pick('Aciertos sobre el total', 'Overall accuracy'),
                           _num(_mod['exactitud']))

                _c4 = st.column_config.NumberColumn()
                st.markdown(i18n.pick('**Por categoría**', '**By category**'))
                _k_pre = i18n.pick('Precisión', 'Precision')
                _k_rec = i18n.pick('Sensibilidad (recall)', 'Recall')
                st.dataframe(_con_coma(pd.DataFrame([{
                    i18n.pick('Categoría', 'Category'): i18n.tv(c['categoria']),
                    _k_pre: c['precision'], _k_rec: c['recall'], 'F1': c['f1'],
                    i18n.pick('Muestras validadas', 'Validated samples'): c['n_validacion']}
                    for c in _mod['por_clase']])), hide_index=True, width='stretch',
                    column_config={_k_pre: _c4, _k_rec: _c4, 'F1': _c4})
                st.caption(i18n.pick(
                    '**Precisión**: de lo que el modelo llama así, cuánto lo es. '
                    '**Sensibilidad**: de lo que lo es, cuánto encuentra. **F1** resume '
                    'las dos.',
                    '**Precision**: of what the model calls this, how much is. '
                    '**Recall**: of what is this, how much it finds. **F1** summarises '
                    'both.'))

                st.markdown(i18n.pick('**Matriz de confusión** (filas: lo que es; columnas: '
                                      'lo que dice el modelo)',
                                      '**Confusion matrix** (rows: what it is; columns: what '
                                      'the model says)'))
                _cats = [i18n.tv(c['categoria']) for c in _mod['categorias']]
                _k_real = i18n.pick('es ↓ · dice →', 'is ↓ · says →')
                st.dataframe(pd.DataFrame(
                    [{_k_real: _cats[a], **{_cats[b]: _mod['matriz'][a][b]
                                            for b in range(len(_cats))}}
                     for a in range(len(_cats))]), hide_index=True, width='stretch')
                if _mod['confusiones']:
                    st.markdown(i18n.pick('**Lo que más se confunde**',
                                          '**Most frequent confusions**')
                                + ''.join(i18n.pick(
                                    f'\n- «{i18n.tv(a)}» tomado por «{i18n.tv(b)}»: {n}',
                                    f'\n- "{i18n.tv(a)}" taken for "{i18n.tv(b)}": {n}')
                                    for a, b, n in _mod['confusiones'][:5]))
                else:
                    st.caption(i18n.pick('Ninguna muestra mal clasificada en la validación.',
                                         'No sample misclassified in the validation.'))

                with st.expander(i18n.pick('Resultado de cada parte',
                                           'Result of each fold')):
                    _k_f1 = 'macro-F1'
                    _k_ba = i18n.pick('exactitud equilibrada', 'balanced accuracy')
                    st.dataframe(pd.DataFrame([{
                        i18n.pick('parte', 'fold'): p['parte'],
                        i18n.pick('zonas apartadas', 'held-out zones'):
                            ', '.join(p['zonas_valida']),
                        i18n.pick('entrena', 'trains'): p['n_entrena'],
                        i18n.pick('valida', 'validates'): p['n_valida'],
                        _k_f1: p['macro_f1'], _k_ba: p['exactitud_equilibrada'],
                        i18n.pick('categorías que no había visto', 'categories not seen'):
                            ', '.join(i18n.tv(c) for c in p['categorias_sin_entrenar'])}
                        for p in _mod['pliegues']]), hide_index=True, width='stretch',
                        column_config={_k_f1: _c4, _k_ba: _c4})
                _mal = [p for p in _mod['predicciones'] if not p['acierta']]
                if _mal:
                    with st.expander(i18n.pick(
                            f'Ver las {len(_mal)} muestras mal clasificadas',
                            f'See the {len(_mal)} misclassified samples')):
                        _k_conf = i18n.pick('confianza', 'confidence')
                        st.dataframe(pd.DataFrame({
                            'ID': bp.abreviar_ids([p['ID'] for p in _mod['predicciones']]),
                            i18n.pick('zona', 'zone'):
                                [p['zona'] for p in _mod['predicciones']],
                            i18n.pick('es', 'is'):
                                [i18n.tv(p['real']) for p in _mod['predicciones']],
                            i18n.pick('el modelo dice', 'the model says'):
                                [i18n.tv(p['predicha']) for p in _mod['predicciones']],
                            _k_conf: [p['confianza'] for p in _mod['predicciones']],
                        })[[not p['acierta'] for p in _mod['predicciones']]],
                            hide_index=True, width='stretch', column_config={_k_conf: _c4})
                        st.caption(i18n.pick(
                            'Merece la pena mirarlas: una muestra mal clasificada puede '
                            'ser un fallo del modelo o una etiqueta que conviene revisar.',
                            'They are worth a look: a misclassified sample may be a model '
                            'error or a label worth reviewing.'))

                _bal = _mod['balance']
                _pm = _mod['params']
                st.success(i18n.pick(
                    f"**Modelo final** entrenado con {_mod['n_entrenadas']} de las "
                    f"{_mod['n_muestras']} muestras y {len(_mod['predictores'])} variables "
                    f"({_pm['n_estimators']} árboles, semilla {_pm['random_state']}; "
                    f"{_est_corto[_bal['estrategia']]}). Es el que hace el mapa, en "
                    '«4 · Mapa»; las cifras de arriba son de la validación.',
                    f"**Final model** trained with {_mod['n_entrenadas']} of the "
                    f"{_mod['n_muestras']} samples and {len(_mod['predictores'])} variables "
                    f"({_pm['n_estimators']} trees, seed {_pm['random_state']}; "
                    f"{_est_corto[_bal['estrategia']]}). It is the one that makes the "
                    'map, under "4 · Map"; the figures above come from the validation.'))
                if _bal['final'] != _bal['original']:
                    st.dataframe(pd.DataFrame([{
                        i18n.pick('Categoría', 'Category'): i18n.tv(c),
                        i18n.pick('Muestras', 'Samples'): _bal['original'][c],
                        i18n.pick('Usadas al entrenar', 'Used to train'): _bal['final'][c]}
                        for c in _bal['original']]), hide_index=True, width='stretch')

                # ---- ¿Mejoraría con más puntos? La curva con los puntos de esta zona
                # (25-11-2026, la autora: «¿los puntos mínimos son solo para Sierra
                # Nevada?»). La de Sierra Nevada, en «1 · Datos», es una referencia; esta
                # repite la validación de este modelo con menos puntos
                # (mapeo_rf.curva_propia). Se guarda en el propio modelo: al volver a
                # entrenar, desaparece.
                if ('_validacion' in _mod and _tab is not None
                        and _mod['firma'] == mr.firma_tabla(_tab)):
                    st.markdown(i18n.pick('#### ¿Mejoraría con más puntos?',
                                          '#### Would more points help?'))
                    _n_ent = mr.entrenamientos_curva(_mod)
                    st.caption(i18n.pick(
                        'El mínimo de puntos de «1 · Datos» sale de Sierra Nevada: en otra '
                        'zona es solo una referencia. Esta curva se hace con **tus puntos**. '
                        'Repite la validación por zonas de este modelo, con las mismas '
                        'partes, pero el modelo aprende solo con el 25, el 50 o el 75 % de '
                        'los puntos, elegidos al azar (3 sorteos de cada uno). Las zonas '
                        'apartadas se predicen siempre enteras. Si de la mitad de los '
                        'puntos a todos (el doble) el acierto todavía sube, más puntos '
                        'mejorarán el mapa. '
                        f'Entrena {_n_ent} modelos: con unos mil puntos, medio minuto.',
                        'The minimum number of points under "1 · Data" comes from Sierra '
                        'Nevada: elsewhere it is only a reference. This curve is made with '
                        '**your points**. It repeats this model\'s validation by zones, '
                        'with the same folds, but the model learns from only 25, 50 or 75 % '
                        'of the points, chosen at random (3 draws of each). The held-out '
                        'zones are always predicted in full. If accuracy still rises from '
                        'half of the points to all of them (twice as many), more points '
                        'will improve the map. '
                        f'It trains {_n_ent} models: with about a thousand points, half a '
                        'minute.'))
                    if st.button(i18n.pick('📈 Calcular la curva con mis puntos',
                                           '📈 Compute the curve with my points'),
                                 key='mrf_curva'):
                        _barra = st.progress(0.0, text=i18n.pick(
                            'Entrenando con menos puntos…', 'Training with fewer points…'))
                        try:
                            _mod['_curva'] = mr.curva_propia(
                                _tab, _mod, al_avanzar=lambda f: _barra.progress(
                                    min(1.0, f), text=i18n.pick(
                                        f'Entrenando con menos puntos… {round(f * 100)} %',
                                        f'Training with fewer points… {round(f * 100)} %')))
                        except ValueError as _e:
                            st.error(i18n.pick(f'No se ha podido calcular: {_e}',
                                               f'It could not be computed: {_e}'))
                        _barra.empty()
                    _curva = _mod.get('_curva')
                    if _curva and _curva['firma'] == _mod['firma']:
                        _cc = mr.cifras_curva(_curva, _mod)
                        _md = _cc['medida']
                        _todo, _mit = _cc['filas'][-1], _cc['mitad']
                        _que = (i18n.pick('el AUC en zonas nuevas', 'the AUC in new areas')
                                if _md == 'auc' else
                                i18n.pick('el F1 medio en zonas nuevas',
                                          'the mean F1 in new areas'))
                        _dicho = i18n.pick(
                            'De la mitad de tus puntos a todos (el modelo aprende con unos '
                            f"{round(_mit['n_puntos'])} y con unos "
                            f"{round(_todo['n_puntos'])} en cada parte), {_que} pasa de "
                            f"**{_num(_mit[_md])}** a **{_num(_todo[_md])}** "
                            f"({_num(_cc['ganancia'], signo=True)}).",
                            'From half of your points to all of them (the model learns '
                            f"from about {round(_mit['n_puntos'])} and about "
                            f"{round(_todo['n_puntos'])} in each fold), {_que} goes from "
                            f"**{_num(_mit[_md])}** to **{_num(_todo[_md])}** "
                            f"({_num(_cc['ganancia'], signo=True)}).")
                        if _cc['lectura'] == 'sube':
                            st.info(_dicho + i18n.pick(
                                ' **Todavía sube**: si vuelves a doblar los puntos, cabe '
                                'esperar que suba algo parecido, o algo menos. Busca sobre '
                                'todo en las zonas y los ambientes donde tienes pocos, y '
                                'revisa tanto borreguiles como ausencias.',
                                ' **It is still rising**: if you double the points again, '
                                'expect it to rise by about as much, or a little less. Look '
                                'above all in the areas and settings where you have few, '
                                'and review both borreguiles and absences.'))
                        else:
                            st.success(_dicho + i18n.pick(
                                ' **Ya casi no sube**: aunque dobles los puntos, si son '
                                'parecidos a los que tienes, el mapa apenas cambiará. Si el '
                                'acierto te parece bajo, lo que falta no es cantidad: '
                                'revisa las muestras mal clasificadas, añade puntos de '
                                'sitios distintos (otros valles, otras altitudes) o prueba '
                                'otras variables.',
                                ' **It has almost stopped rising**: even if you double the '
                                'points, if they are like the ones you have, the map will '
                                'hardly change. If the accuracy seems low to you, what is '
                                'missing is not quantity: review the misclassified '
                                'samples, add points from different places (other valleys, '
                                'other altitudes) or try other variables.'))
                        st.altair_chart(pp.chart_curva_propia(_cc), width='stretch')
                        _k_parte = i18n.pick('Parte de tus puntos', 'Share of your points')
                        _k_np = i18n.pick('Puntos con los que aprende',
                                          'Points it learns from')
                        _k_nb = i18n.pick('de ellos, borreguiles', 'of which borreguiles')
                        _k_lo = i18n.pick('peor sorteo', 'worst draw')
                        _k_hi = i18n.pick('mejor sorteo', 'best draw')
                        _k_f1m = i18n.pick('F1 medio', 'Mean F1')
                        st.dataframe(_con_coma(pd.DataFrame([{
                            _k_parte: f"{round(f['fraccion'] * 100)} %",
                            _k_np: int(round(f['n_puntos'])),
                            _k_nb: int(round(f['n_borreguiles'])),
                            **({'AUC': f['auc'], f'AUC · {_k_lo}': f['auc_min'],
                                f'AUC · {_k_hi}': f['auc_max']} if _md == 'auc' else {}),
                            _k_f1m: f['f1']} for f in _cc['filas']])),
                            hide_index=True, width='stretch')
                        st.caption(i18n.pick(
                            'En cada parte de la validación, el modelo aprende con los '
                            'puntos de las otras zonas (unas 4 de cada 5 partes del total) '
                            'y predice los de la zona apartada. El 100 % es la validación '
                            'de este modelo, con un solo resultado; las demás son la media '
                            'de 3 sorteos, y la franja va del peor al mejor. Con pocos '
                            'puntos la curva baila: mira la franja antes de sacar '
                            'conclusiones. El F1 medio usa el umbral de «¿Borreguil o no?» '
                            'elegido de la misma manera.',
                            'In each fold of the validation, the model learns from the '
                            'points of the other zones (about 4 out of 5 parts of the '
                            'total) and predicts those of the held-out zone. 100 % is this '
                            'model\'s validation, a single result; the others are the mean '
                            'of 3 draws, and the band goes from the worst to the best. With '
                            'few points the curve wobbles: look at the band before drawing '
                            'conclusions. The mean F1 uses the "Borreguil or not?" '
                            'threshold chosen in the same way.'))

            # ---- Con qué variables acierta más. Cada modelo entrenado en la sesión deja
            # aquí su acierto en la validación por zonas; se comparan los del mismo nivel
            # y con las mismas muestras (las variables pueden ser otras).
            _f_mu = mr.firma_muestras(_res)
            if _mod and _tab is not None and _mod['firma'] == mr.firma_tabla(_tab):
                st.session_state.setdefault('mrf_variantes', {})[
                    (_nivel, _f_mu, tuple(_mod['predictores']))] = {
                    'vif': (_mod.get('variables') or {}).get('umbral'),
                    'pca': (_mod.get('variables') or {}).get('metodo') == 'pca',
                    'n_var': len(_mod['predictores']), 'n_muestras': _mod['n_muestras'],
                    'f1': _mod['macro_f1']}
            _variantes = sorted(
                ((k, v) for k, v in (st.session_state.get('mrf_variantes') or {}).items()
                 if k[0] == _nivel and k[1] == _f_mu), key=lambda kv: -kv[1]['n_var'])
            if len(_variantes) > 1:
                st.markdown(i18n.pick('#### Con qué variables acierta más',
                                      '#### Which variables work best'))
                _ahora = tuple((_mod or {}).get('predictores') or ())
                st.dataframe(pd.DataFrame([{
                    i18n.pick('Variables', 'Variables'):
                        ('▶ ' if k[2] == _ahora else '')
                        + (i18n.pick(f"una por eje del PCA ({v['n_var']})",
                                     f"one per PCA axis ({v['n_var']})")
                           if v.get('pca') else
                           i18n.pick(f"sin las correlacionadas: VIF ≤ {v['vif']:g} ({v['n_var']})",
                                     f"without correlated ones: VIF ≤ {v['vif']:g} ({v['n_var']})")
                           if v['vif'] else
                           i18n.pick(f"las de «1 · Datos» ({v['n_var']})",
                                     f"those of \"1 · Data\" ({v['n_var']})")),
                    i18n.pick('Muestras', 'Samples'): v['n_muestras'],
                    i18n.pick('F1 medio', 'Mean F1'): _num(v['f1'])}
                    for k, v in _variantes]), hide_index=True, width='stretch')
                st.caption(i18n.pick(
                    'Cada fila es un modelo entrenado en esta sesión con las mismas '
                    'muestras; ▶ es el de ahora, el que hace el mapa. El F1 medio es el de '
                    'la validación por zonas. Una diferencia de unas centésimas puede ser '
                    'azar: con pocas muestras, una sola cambia la cifra. Para quedarte con '
                    'otro, cambia la opción de las variables correlacionadas en «1 · Datos» y '
                    'vuelve a entrenar.',
                    'Each row is a model trained in this session with the same samples; ▶ '
                    'is the current one, which makes the map. The mean F1 is that of the '
                    'validation by zones. A difference of a few hundredths may be chance: '
                    'with few samples, a single one changes the figure. To keep another '
                    'one, change the correlated-variables option under "1 · Data" and '
                    'train again.'))

        # ------------------------------------------------------------ 4 · Mapa
        # El modelo final se aplica a todos los píxeles de una zona. Se leen de la
        # misma pila, con los mismos años y en la misma rejilla que las muestras, por
        # bloques, y el resultado se va escribiendo en tres GeoTIFF.
        with _t_mapa:
            import math
            import shutil
            import mapeo_ambiente as mam
            import mapeo_raster as mra
            st.caption(i18n.pick(
                'Aplica el **modelo final** a todos los píxeles de 10 m de una zona y '
                'guarda sus mapas: la **categoría** de cada píxel, la **probabilidad** '
                'de esa categoría (lo seguro que está el modelo), la probabilidad de '
                'cada categoría por separado y, si el modelo distingue «no borreguil», '
                'la **probabilidad de ser borreguil**. La categoría se decide con la '
                'regla y el umbral de «3 · Modelo». Además puede decir, de cada píxel de '
                'borreguil, si es **puro** o está **mezclado** con agua o con roca.',
                'Applies the **final model** to every 10 m pixel of a zone and saves '
                'its maps: the **category** of each pixel, the **probability** of that '
                'category (how sure the model is), the probability of each category '
                'separately and, if the model tells "not a borreguil" apart, the '
                '**probability of being a borreguil**. The category is decided with the '
                'rule and the threshold of "3 · Model". It can also tell, for each '
                'borreguil pixel, whether it is **pure** or **mixed** with water or rock.'))
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

            _MAX_BLOQUES = 400           # ≈ 2 600 km² y unas 7 h: más, por partes
            _lec_m = (_mod or {}).get('lectura') or {}
            _al_dia = bool(_mod and _tab is not None and _lec_m and _mod.get('muestras')
                           and _lec_m.get('version') == geb.VERSION_PILA
                           and _mod['firma'] == mr.firma_tabla(_tab))
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
                        'The map is made **in two steps**: whether a pixel is a borreguil '
                        'is decided by the **"Borreguil yes or no"** model (threshold '
                        f"{_num(_mod_si_no['regla']['umbral'])}, trained on "
                        f"{_mod_si_no['cuando'].replace('T', ' at ')}), and its type, by "
                        'this model.'))
                elif _dos_mal:
                    st.warning(i18n.pick(
                        f'No se puede juntar con el modelo «Borreguil sí o no»: {_dos_mal}.',
                        f'It cannot be combined with the "Borreguil yes or no" model: '
                        f'{_dos_mal}.'))
                elif _nivel != mr.NIVEL_SI_NO and not _con_aus:
                    st.info(i18n.pick(
                        'Para que el mapa diga también dónde **no** hay borreguil, entrena '
                        'el modelo **«Borreguil sí o no»** (elígelo arriba, en «Qué quieres '
                        'cartografiar»): el mapa se hará entonces en dos pasos.',
                        'For the map to also say where there is **no** borreguil, train '
                        'the **"Borreguil yes or no"** model (choose it above, under "What '
                        'to map"): the map will then be made in two steps.'))
                # Con el ambiente por cercanía, el modelo aprende de cada muestra solo la
                # humedad; pero la persona revisó también su ambiente. Aquí queda lo que
                # revisó en cada una: el ambiente y la categoría completa («laguna ·
                # húmedo»), para pintarlas así y para compararlo después con el mapa.
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
                    """La categoría con la que se pinta una muestra: la completa que
                    revisó la persona, si la hay; si no, la del modelo."""
                    return (_de_revisado.get(mr.clave_punto(m['lon'], m['lat'])) or {}).get(
                        'compuesta') or m['categoria']

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

                # ---- pureza dentro del píxel: puro, mixto-agua o mixto-roca. No sale del
                # modelo, sino de cuánto verde hay en cada píxel de borreguil (mapeo_rf.
                # verde_estimado). Los valores de partida se ajustan aquí.
                _m_pur = []                 # muestras de borreguil: su pureza y su NDVI
                for _f in _usadas:
                    _v = _leidos.get(mr.clave_punto(_f['lon'], _f['lat'])) or {}
                    if _f['categoria'] != mr.AUSENCIA and _v.get(mr.VARIABLE_VERDE) is not None:
                        _r = rows[_f['indice']] if 0 <= _f['indice'] < len(rows) else {}
                        # solo la pureza que ha revisado una persona, no la propuesta
                        _m_pur.append({'pureza': (bp._txt(_r.get('pureza'))
                                                  if 'pureza' in bp.niveles_revisados(_r)
                                                  else ''),
                                       'ndvi': _v[mr.VARIABLE_VERDE]})
                _n_puras = sum(1 for m in _m_pur if m['pureza'] == 'puro')
                _tv_prop, _tv_muestras = mr.todo_verde_de(
                    [m['ndvi'] for m in _m_pur if m['pureza'] == 'puro'])
                with st.expander(i18n.pick(
                        'Pureza dentro del píxel: puro, mixto-agua o mixto-roca',
                        'Purity within the pixel: pure, mixed-water or mixed-rock')):
                    st.caption(i18n.pick(
                        'Un píxel de 10 m puede ser borreguil entero o mezclarse con roca '
                        'o con agua. En cada píxel que el modelo da por **borreguil** se '
                        'estima cuánto verde hay dentro con su NDVI de inicio de verano, '
                        'y con eso se le pone: **puro** si el verde llega al corte; si no, '
                        '**mixto-agua** si es agua o toca un píxel de agua, y '
                        '**mixto-roca** en otro caso. No cambia qué es borreguil ni su '
                        'tipo.',
                        'A 10 m pixel may be all borreguil or mixed with rock or water. '
                        'In each pixel the model calls **borreguil**, the amount of green '
                        'inside is estimated from its early-summer NDVI, and from that it '
                        'is labelled: **pure** if the green reaches the cut-off; '
                        'otherwise **mixed-water** if it is water or touches a water '
                        'pixel, and **mixed-rock** in any other case. It does not change '
                        'what is borreguil or its type.'))
                    if _nivel == 'pureza':
                        st.info(i18n.pick(
                            'Estás cartografiando la **pureza**: la categoría del mapa será la '
                            'que el modelo aprende de tus muestras, y esta es **otra '
                            'estimación**, hecha solo con el NDVI de cada píxel. No tienen por '
                            'qué coincidir; donde no coincidan, mira la ortofoto.',
                            'You are mapping **purity**: the map category will be the one the '
                            'model learns from your samples, and this is **another estimate**, '
                            'made only from the NDVI of each pixel. They need not agree; '
                            'where they do not, check the orthophoto.'))
                    _pur_si = st.checkbox(
                        i18n.pick('Calcular la pureza de cada píxel de borreguil',
                                  'Compute the purity of each borreguil pixel'),
                        value=True, key='mrf_pureza_si',
                        help=i18n.pick(
                            'Añade al mapa una capa y dos ficheros más: la pureza de cada '
                            'píxel de borreguil y el verde estimado de cada píxel. Solo '
                            'usa Sentinel-2 y no alarga el mapa.\n\n'
                            '**Recomendado: marcado.**',
                            'Adds one layer and two more files to the map: the purity of '
                            'each borreguil pixel and the estimated green of each pixel. '
                            'It only uses Sentinel-2 and does not lengthen the map.\n\n'
                            '**Recommended: ticked.**'))
                    _q1, _q2 = st.columns(2)
                    _pur_sin = _q1.number_input(
                        i18n.pick('NDVI de un píxel sin verde', 'NDVI of a pixel with no green'),
                        0.0, 0.5, mr.PUREZA_DEFECTO['sin_verde'], 0.01, format='%.3f',
                        key='mrf_pureza_sin', disabled=not _pur_si,
                        help=i18n.pick(
                            'El NDVI que tiene la roca o el suelo desnudo. Por debajo de '
                            'este valor se cuenta un 0 % de verde.\n\n'
                            '**Recomendado: 0.080**, lo habitual en roca y suelo de alta '
                            'montaña.',
                            'The NDVI of rock or bare ground. Below this value the pixel '
                            'counts as 0 % green.\n\n'
                            '**Recommended: 0.080**, the usual value for high-mountain '
                            'rock and ground.'))
                    _pur_todo = _q2.number_input(
                        i18n.pick('NDVI de un píxel todo verde', 'NDVI of an all-green pixel'),
                        0.2, 1.0, min(1.0, max(0.2, round(_tv_prop, 3))), 0.01, format='%.3f',
                        key='mrf_pureza_todo', disabled=not _pur_si,
                        help=i18n.pick(
                            'El NDVI de un píxel cubierto entero por borreguil. A partir '
                            'de este valor se cuenta un 100 % de verde; entre los dos, en '
                            'proporción.\n\n'
                            '**Recomendado:** el de tus muestras de borreguil puro (su '
                            'mediana), que es el que viene puesto si hay al menos 5; si '
                            'no, 0.60.',
                            'The NDVI of a pixel entirely covered by borreguil. From this '
                            'value on the pixel counts as 100 % green; between the two, in '
                            'proportion.\n\n'
                            '**Recommended:** that of your pure borreguil samples (their '
                            'median), which is the preset value if there are at least 5; '
                            'otherwise 0.60.'))
                    _pur_corte = st.slider(
                        i18n.pick('Verde a partir del cual el píxel es puro (%)',
                                  'Green from which the pixel is pure (%)'),
                        50, 95, int(round(100 * mr.PUREZA_DEFECTO['puro_desde'])), 5,
                        key='mrf_pureza_corte', disabled=not _pur_si,
                        help=i18n.pick(
                            'Con menos verde que este, el píxel es mixto.\n\n'
                            '**Recomendado: 75 %.** El verde se estima con un error de '
                            'unos 10 puntos, así que un corte muy alto dejaría casi todo '
                            'como mixto.',
                            'With less green than this, the pixel is mixed.\n\n'
                            '**Recommended: 75 %.** The green is estimated with an error '
                            'of about 10 points, so a very high cut-off would leave almost '
                            'everything as mixed.'))
                    _par_pur, _pur_mal = None, bool(_pur_si and _pur_todo <= _pur_sin)
                    if _pur_si and not _pur_mal:
                        _par_pur = {'sin_verde': float(_pur_sin), 'todo_verde': float(_pur_todo),
                                    'puro_desde': _pur_corte / 100}
                    st.caption(i18n.pick(
                        f'Tus {_n_puras} muestras de borreguil puro tienen un NDVI de inicio '
                        f'de verano de {_num(_tv_prop)} de mediana.' if _tv_muestras else
                        f'Hay {_n_puras} muestras de borreguil puro: con menos de '
                        f'{mr.MIN_PUROS} no se propone su NDVI, sino {_num(_tv_prop)}.',
                        f'Your {_n_puras} pure borreguil samples have a median early-summer '
                        f'NDVI of {_num(_tv_prop)}.' if _tv_muestras else
                        f'There are {_n_puras} pure borreguil samples: with fewer than '
                        f'{mr.MIN_PUROS} their NDVI is not proposed, but {_num(_tv_prop)}.'))
                    if _par_pur and _m_pur:
                        _cmp = mr.comparar_pureza(_m_pur, _par_pur)
                        st.markdown(i18n.pick(
                            f"Con estos valores, **{_cmp['coinciden']} de tus {_cmp['n']} "
                            'muestras** de borreguil quedan como las revisaste (puro frente '
                            'a mixto):',
                            f"With these values, **{_cmp['coinciden']} of your {_cmp['n']} "
                            'borreguil samples** come out as you reviewed them (pure versus '
                            'mixed):'))
                        st.dataframe(pd.DataFrame([{
                            i18n.pick('Pureza revisada', 'Reviewed purity'): i18n.tv(c),
                            i18n.pick('Sale puro', 'Comes out pure'): _cmp['tabla'][c]['puro'],
                            i18n.pick('Sale mixto', 'Comes out mixed'): _cmp['tabla'][c]['mixto']}
                            for c in mr.NIVELES['pureza']]), hide_index=True, width='stretch')
                        st.caption(i18n.pick(
                            'En un punto suelto no se sabe si toca agua: por eso aquí solo '
                            'se compara puro frente a mixto.',
                            'For a single point it is not known whether it touches water: '
                            'that is why only pure versus mixed is compared here.'))
                if _pur_mal:
                    # fuera del desplegable, que puede estar plegado
                    st.error(i18n.pick(
                        '**Pureza dentro del píxel:** el NDVI de «todo verde» tiene que ser '
                        'mayor que el de «sin verde». Corrígelo en el desplegable de arriba, '
                        'o desmarca la pureza, para poder hacer el mapa.',
                        '**Purity within the pixel:** the "all green" NDVI must be greater '
                        'than the "no green" one. Fix it in the expander above, or untick '
                        'the purity, to be able to make the map.'))
                # ---- ambiente por cercanía (en «Ambiente y humedad» y, si se pide, en
                # «Borreguil sí o no»). No sale del modelo, sino de la distancia de cada
                # píxel a lagunas y cauces (mapeo_ambiente): por eso no necesita tipos
                # revisados. Las distancias de partida se ajustan aquí.
                _par_amb, _amb_mal, _capa_cauces, _f_cauces, _lagunas_ll = None, None, None, None, []
                _amb_si_no = False
                if _nivel == mr.NIVEL_SI_NO and _con_aus:
                    _amb_si_no = st.checkbox(
                        i18n.pick('Decir el ambiente de cada píxel de borreguil (arroyo, '
                                  'laguna o ladera)',
                                  'Give the environment of each borreguil pixel (stream, '
                                  'lake or slope)'),
                        value=True, key='mrf_amb_si_no',
                        help=i18n.pick(
                            'El modelo decide si hay borreguil; el ambiente lo pone el mapa '
                            'por la **cercanía** de cada píxel de borreguil a lagunas y '
                            'cauces, así que no hace falta ningún punto con el tipo '
                            'revisado. En el mapa, cada píxel de borreguil sale como '
                            '«arroyo», «laguna» o «ladera». Las distancias se ajustan en el '
                            'desplegable de debajo. Los cauces son los ríos y arroyos de '
                            'OpenStreetMap, los mismos que en los puntos.\n\n'
                            '**Recomendado: marcado.**',
                            'The model decides whether there is borreguil; the environment '
                            'is set by the map from how **close** each borreguil pixel is to '
                            'lakes and watercourses, so no point with a reviewed type is '
                            'needed. On the map, each borreguil pixel comes out as "stream", '
                            '"lake" or "slope". The distances are set in the expander below. '
                            'The watercourses are the OpenStreetMap rivers and streams, the '
                            'same as for the points.\n\n'
                            '**Recommended: ticked.**'))
                if _cercania or _amb_si_no:
                    if 'mrf_lagunas' not in st.session_state:
                        try:
                            st.session_state['mrf_lagunas'] = [
                                (float(q['lon']), float(q['lat'])) for q in bp.read_points(
                                    str(APP_DIR / 'lagunas_sierra_nevada.kml'))]
                        except Exception:
                            st.session_state['mrf_lagunas'] = []
                    _lagunas_ll = st.session_state['mrf_lagunas']
                    with st.expander(i18n.pick(
                            'Ambiente (arroyo, laguna o ladera): por cercanía a lagunas y '
                            'cauces',
                            'Environment (stream, lake or slope): by closeness to lakes and '
                            'watercourses')):
                        st.caption(i18n.pick(
                            'El ambiente dice **dónde está** el borreguil, y eso no se ve '
                            'dentro del píxel: por eso no lo decide el modelo, sino la '
                            'distancia de cada píxel a las lagunas y a los cauces. Un píxel '
                            'es de **laguna** si está cerca de una; si no, de **arroyo** si '
                            'está cerca de un cauce; si no, de **ladera**. El modelo sigue '
                            'diciendo si es borreguil'
                            + ('.' if _amb_si_no else ' y si es húmedo o seco.'),
                            'The environment tells **where** the borreguil is, and that '
                            'cannot be seen inside the pixel: so it is not decided by the '
                            'model but by the distance from each pixel to lakes and '
                            'watercourses. A pixel is a **lake** one if it is close to a '
                            'lake; otherwise a **stream** one if it is close to a '
                            'watercourse; otherwise a **slope** one. The model still says '
                            'whether it is a borreguil'
                            + ('.' if _amb_si_no else ' and whether it is wet or dry.')))
                        _a1, _a2 = st.columns(2)
                        _d_capa = _a1.number_input(
                            i18n.pick('Laguna: metros a una laguna de la capa',
                                      'Lake: metres to a lake of the layer'),
                            0.0, 1000.0, mam.DEFECTO['laguna_capa_m'], 10.0, format='%.0f',
                            key='mrf_amb_capa',
                            help=i18n.pick(
                                'Un píxel es de laguna si está a esta distancia o menos de '
                                'un punto de la capa de lagunas de la app (65 lagunas de '
                                'Sierra Nevada). El punto es el centro de la laguna, no su '
                                'orilla: sirve sobre todo para las lagunas pequeñas, que '
                                'Sentinel-2 no ve.\n\n'
                                '**Recomendado: 120 m**, lo que usa la app al proponer el '
                                'tipo de cada punto.',
                                'A pixel is a lake one if it is within this distance of a '
                                'point of the lake layer of the app (65 lakes of Sierra '
                                'Nevada). The point is the centre of the lake, not its '
                                'shore: it mostly helps with small lakes, which Sentinel-2 '
                                'does not see.\n\n'
                                '**Recommended: 120 m**, what the app uses when it proposes '
                                'the type of each point.'))
                        _d_agua = _a2.number_input(
                            i18n.pick('Laguna: metros al agua que ve Sentinel-2',
                                      'Lake: metres to the water Sentinel-2 sees'),
                            0.0, 1000.0, mam.DEFECTO['laguna_agua_m'], 10.0, format='%.0f',
                            key='mrf_amb_agua',
                            help=i18n.pick(
                                'Un píxel es de laguna si está a esta distancia o menos de '
                                'la orilla de una laguna que se ve en Sentinel-2: píxeles '
                                'con NDWI medio de verano mayor de 0,05, en terreno casi '
                                'llano y en manchas de 3 o más. Vale en cualquier sitio, '
                                'aunque no haya capa de lagunas.\n\n'
                                '**Recomendado: 100 m.**',
                                'A pixel is a lake one if it is within this distance of '
                                'the shore of a lake that shows in Sentinel-2: pixels '
                                'with a mean summer NDWI above 0.05, on nearly flat ground '
                                'and in patches of 3 or more. It works anywhere, even with '
                                'no lake layer.\n\n'
                                '**Recommended: 100 m.**'))
                        _d_arroyo = _a1.number_input(
                            i18n.pick('Arroyo: metros a un cauce', 'Stream: metres to a watercourse'),
                            0.0, 500.0, mam.DEFECTO['arroyo_m'], 10.0, format='%.0f',
                            key='mrf_amb_arroyo',
                            help=i18n.pick(
                                'Un píxel que no es de laguna es de arroyo si está a esta '
                                'distancia o menos de un río o arroyo de OpenStreetMap (o '
                                'de tu capa de cauces, si la subes).\n\n'
                                '**Recomendado: 30 m**, lo que usa la app al proponer el '
                                'tipo de cada punto.',
                                'A pixel that is not a lake one is a stream one if it is '
                                'within this distance of an OpenStreetMap river or stream '
                                '(or of your watercourse layer, if you upload one).\n\n'
                                '**Recommended: 30 m**, what the app uses when it proposes '
                                'the type of each point.'))
                        # Los cauces son los ríos y arroyos de OpenStreetMap, como en los
                        # puntos (24-11-2026, la autora). Los del relieve quedan solo de reserva,
                        # si OpenStreetMap no responde fuera de Sierra Nevada, con su ajuste
                        # de siempre.
                        _cauce_ha = mam.DEFECTO['cauce_ha']
                        _f_cauces = st.file_uploader(
                            i18n.pick('Capa de cauces propia (líneas; opcional)',
                                      'Your own watercourse layer (lines; optional)'),
                            type=['kml', 'geojson', 'json', 'gpkg', 'shp', 'shx', 'dbf', 'prj',
                                  'cpg', 'zip'],
                            accept_multiple_files=True, key='mrf_cauces_fich',
                            help=i18n.pick(
                                'Las líneas de los cauces (por ejemplo, la red hidrográfica '
                                'oficial). Un único KML, GeoJSON o GeoPackage, o un '
                                'shapefile (.shp + .shx + .dbf + .prj juntos, o un .zip). '
                                'Si la subes, se usa en lugar de los ríos y arroyos de '
                                'OpenStreetMap.\n\n'
                                '**Recomendado: no subir nada**, salvo que tengas una red '
                                'de cauces de la zona más completa que la de OpenStreetMap.',
                                'The lines of the watercourses (for example, the official '
                                'hydrographic network). A single KML, GeoJSON or '
                                'GeoPackage, or a shapefile (.shp + .shx + .dbf + .prj '
                                'together, or a .zip). If you upload it, it is used '
                                'instead of the OpenStreetMap rivers and streams.\n\n'
                                '**Recommended: upload nothing**, unless you have a '
                                'watercourse network of the zone more complete than '
                                'OpenStreetMap.'))
                        if _f_cauces:
                            _clave_cau = tuple(sorted((f.name, f.size) for f in _f_cauces))
                            _cc = st.session_state.get('mrf_cauces')
                            if not _cc or _cc['clave'] != _clave_cau:
                                try:
                                    _cc = {'clave': _clave_cau, 'error': None,
                                           'geom': mam.leer_cauces(save_uploaded_area(_f_cauces))}
                                except Exception as _e:
                                    _cc = {'clave': _clave_cau, 'error': str(_e), 'geom': None}
                                st.session_state['mrf_cauces'] = _cc
                            if _cc['error']:
                                _amb_mal = i18n.pick(
                                    f"no se ha podido leer la capa de cauces ({_cc['error']})",
                                    f"the watercourse layer could not be read ({_cc['error']})")
                            else:
                                _capa_cauces = _cc['geom']
                        st.caption(i18n.pick(
                            f'Lagunas: {len(_lagunas_ll)} puntos de la capa de la app '
                            '(Sierra Nevada) y el agua que se vea en Sentinel-2. Cauces: '
                            + ('los de tu capa.' if _capa_cauces is not None else
                               'los ríos y arroyos de OpenStreetMap, los mismos que en los '
                               'puntos (los de Sierra Nevada van con la app; en otras zonas '
                               'se descargan al hacer el mapa).'),
                            f'Lakes: {len(_lagunas_ll)} points of the layer of the app '
                            '(Sierra Nevada) and the water that shows in Sentinel-2. '
                            'Watercourses: '
                            + ('those of your layer.' if _capa_cauces is not None else
                               'the OpenStreetMap rivers and streams, the same as for the '
                               'points (those of Sierra Nevada come with the app; elsewhere '
                               'they are downloaded when the map is made).')))
                    _par_amb = {'laguna_capa_m': float(_d_capa), 'laguna_agua_m': float(_d_agua),
                                'arroyo_m': float(_d_arroyo), 'cauce_ha': float(_cauce_ha)}
                if _amb_mal:
                    # fuera del desplegable, que puede estar plegado
                    st.error(i18n.pick(
                        f'**Ambiente por cercanía:** {_amb_mal}. Quita ese fichero o sube '
                        'otro, en el desplegable de arriba, para poder hacer el mapa.',
                        f'**Environment by closeness:** {_amb_mal}. Remove that file or '
                        'upload another one, in the expander above, to be able to make '
                        'the map.'))
                # ---- el agua de laguna, que no es borreguil
                _agua_si = False
                if _con_aus:
                    _agua_si = st.checkbox(
                        i18n.pick('Quitar el agua de laguna (no es borreguil)',
                                  'Remove lake water (it is not borreguil)'),
                        value=True, key='mrf_quitar_agua',
                        help=i18n.pick(
                            'El agua abierta de una laguna se parece a un borreguil muy '
                            'húmedo, y el modelo la puede dar por borreguil. Con esto, los '
                            'píxeles de agua que se ven en Sentinel-2 (NDWI medio de verano '
                            'mayor de 0,05, en terreno casi llano y en manchas de 3 o más '
                            'píxeles) pasan a «no borreguil». El agua se guarda además en '
                            'su propio fichero. Las probabilidades no cambian.\n\n'
                            '**Recomendado: marcado.**',
                            'The open water of a lake looks like a very wet borreguil, and '
                            'the model may call it one. With this, the water pixels that '
                            'show in Sentinel-2 (mean summer NDWI above 0.05, on nearly flat '
                            'ground and in patches of 3 or more pixels) become "not a '
                            'borreguil". The water is also saved in its own file. The '
                            'probabilities do not change.\n\n'
                            '**Recommended: ticked.**'))
                # Lo que hay que leer de cada bloque: las variables del modelo y, detrás, las
                # que solo hacen falta para la pureza, el ambiente o el agua, si el modelo
                # no las lleva. (El relieve de los cauces se lee aparte, con su margen.)
                _extra_b = ([mr.VARIABLE_VERDE, mr.VARIABLE_AGUA] if _par_pur else []) + (
                    [mam.VARIABLE_AGUA, mam.VARIABLE_PENDIENTE] if _par_amb or _agua_si else [])
                _bandas_m = list(_mod_mapa['predictores']) + [
                    b for b in dict.fromkeys(_extra_b) if b not in _mod_mapa['predictores']]

                # ---- zona
                _zonas = {
                    'rect': i18n.pick('Un rectángulo que elijo yo',
                                      'A rectangle I choose'),
                    'muestras': i18n.pick('El rectángulo que ocupan las muestras',
                                          'The rectangle the samples occupy')}
                if sg is not None:
                    _zonas['area'] = i18n.pick('El área de estudio', 'The study area')
                if _masc is not None:
                    _zonas['mascara'] = i18n.pick('Todo lo que cubre la máscara',
                                                  'Everything the mask covers')
                _zona = st.radio(
                    i18n.pick('Zona a cartografiar', 'Zone to map'), list(_zonas),
                    key='mrf_zona', format_func=lambda z: _zonas[z],
                    help=i18n.pick(
                        'Earth Engine sirve unos 6 km² por minuto. Conviene empezar por '
                        'un rectángulo pequeño, ver cómo sale y ampliar después. Debajo '
                        'se indica lo que tardará lo que elijas.\n\n'
                        '**Recomendado para empezar: un rectángulo de 2 × 2 km** (un '
                        'minuto).',
                        'Earth Engine serves about 6 km² per minute. It is best to start '
                        'with a small rectangle, see how it turns out and enlarge it '
                        'afterwards. What you choose shows below how long it will '
                        'take.\n\n'
                        '**Recommended to start: a 2 × 2 km rectangle** (one minute).'))
                _lim_z, _pol_z = None, None
                if _zona == 'rect':
                    # Centro por defecto: la muestra más cercana al centro de todas,
                    # para que el primer rectángulo tenga alguna dentro. Los valores
                    # se inician en el estado (y no con `value`) para poder cambiarlos
                    # al pinchar en el mapa de debajo.
                    _lon_c = sorted(p[0] for p in _pts_m)[len(_pts_m) // 2]
                    _lat_c = sorted(p[1] for p in _pts_m)[len(_pts_m) // 2]
                    _cerca = min(_pts_m, key=lambda p: (p[0] - _lon_c) ** 2
                                 + (p[1] - _lat_c) ** 2)
                    _antes = st.session_state.get('_mrf_rect') or (
                        round(_cerca[0], 5), round(_cerca[1], 5), 2.0, 2.0)
                    _pend_c = st.session_state.pop('_mrf_centro', None)
                    if _pend_c:
                        _antes = (_pend_c[0], _pend_c[1]) + tuple(_antes[2:])
                        st.session_state['mrf_rect_lon'] = _pend_c[0]
                        st.session_state['mrf_rect_lat'] = _pend_c[1]
                    for _k, _v in zip(('mrf_rect_lon', 'mrf_rect_lat', 'mrf_rect_ancho',
                                       'mrf_rect_alto'), _antes):
                        st.session_state.setdefault(_k, float(_v))
                    _r1, _r2 = st.columns(2)
                    _r_lon = _r1.number_input(
                        i18n.pick('Longitud del centro', 'Centre longitude'), -180.0, 180.0,
                        step=0.005, format='%.5f', key='mrf_rect_lon',
                        help=i18n.pick(
                            'En grados decimales (oeste, negativo). También puedes '
                            'pinchar en el mapa de debajo.',
                            'In decimal degrees (west is negative). You can also click '
                            'on the map below.'))
                    _r_lat = _r2.number_input(
                        i18n.pick('Latitud del centro', 'Centre latitude'), -90.0, 90.0,
                        step=0.005, format='%.5f', key='mrf_rect_lat',
                        help=i18n.pick(
                            'En grados decimales. También puedes pinchar en el mapa de '
                            'debajo.',
                            'In decimal degrees. You can also click on the map below.'))
                    _r_w = _r1.number_input(
                        i18n.pick('Ancho (km)', 'Width (km)'), 0.1, 200.0, step=0.5,
                        key='mrf_rect_ancho',
                        help=i18n.pick(
                            'De oeste a este.\n\n**Recomendado para empezar: 2 km.**',
                            'From west to east.\n\n**Recommended to start: 2 km.**'))
                    _r_h = _r2.number_input(
                        i18n.pick('Alto (km)', 'Height (km)'), 0.1, 200.0, step=0.5,
                        key='mrf_rect_alto',
                        help=i18n.pick(
                            'De sur a norte.\n\n**Recomendado para empezar: 2 km.**',
                            'From south to north.\n\n**Recommended to start: 2 km.**'))
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

                # La rejilla y sus bloques: se calculan una vez por zona, no en cada
                # pasada del guion.
                _clave_z = (_clave_z, _m_epsg, _mc['clave'] if _masc is not None else None)
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
                        'La máscara no toca la zona elegida: no hay nada que clasificar.',
                        'The mask does not touch the chosen zone: there is nothing to '
                        'classify.')
                elif _zc['error'] == 'sin_superficie':
                    _pega = i18n.pick('La zona elegida no tiene superficie.',
                                      'The chosen zone has no area.')
                elif _zc['error'] == 'demasiado_grande' or len(_utiles) > _MAX_BLOQUES:
                    _nb = len(_utiles) or _zc.get('bloques', 0)
                    _pega = i18n.pick(
                        f"La zona es demasiado grande para hacerla de una vez "
                        f"({_zc.get('km2', 0):.0f} km², {_nb} bloques; el máximo son "
                        f"{_MAX_BLOQUES}, unas 7 h). Elige una zona menor y hazla por "
                        'partes.',
                        f"The zone is too large to do in one go "
                        f"({_zc.get('km2', 0):.0f} km², {_nb} blocks; the maximum is "
                        f"{_MAX_BLOQUES}, about 7 h). Choose a smaller zone and do it in "
                        'parts.')
                elif _zc['error']:
                    _pega = i18n.pick(
                        f"No se ha podido preparar la zona: {_zc.get('detalle', '')}",
                        f"The zone could not be prepared: {_zc.get('detalle', '')}")
                elif not _utiles:
                    _pega = i18n.pick(
                        'En la zona elegida no cae el centro de ningún píxel: no hay '
                        'nada que clasificar.',
                        'No pixel centre falls inside the chosen zone: there is nothing '
                        'to classify.')

                if _rej and not _pega:
                    _dir_m = st.session_state.get('mrf_dir')
                    _lect_m = mra.clave_lectura(_m_epsg, _m_anios, _bandas_m,
                                                geb.VERSION_PILA)
                    _por_leer = (mra.bloques_por_leer(Path(_dir_m) / 'bloques' / _lect_m,
                                                      _rej, _utiles)
                                 if _dir_m else list(_utiles))
                    _min_m = mra.minutos_estimados(_rej, ventanas=_por_leer)
                    _nu, _nl = len(_utiles), len(_por_leer)
                    st.markdown(i18n.pick(
                        f"**{_num(_zc['km2'])} km²** que clasificar · tardará "
                        f"**{_dura(_min_m)}** aproximadamente.",
                        f"**{_num(_zc['km2'])} km²** to classify · it will take about "
                        f"**{_dura(_min_m)}**."))
                    # los detalles, para quien los quiera, en pequeño
                    st.caption(i18n.pick(
                        f"Detalles: {_rej['ancho']} × {_rej['alto']} píxeles de 10 m "
                        f"(EPSG:{_m_epsg}) · años {bp.abreviar_anios(_m_anios)} · se lee "
                        f"en {_nu} trozo{'' if _nu == 1 else 's'} de Earth Engine"
                        + (f', {_nu - _nl} ya leído{"" if _nu - _nl == 1 else "s"}'
                           if _nl < _nu else '') + '.',
                        f"Details: {_rej['ancho']} × {_rej['alto']} pixels of 10 m "
                        f"(EPSG:{_m_epsg}) · years {bp.abreviar_anios(_m_anios)} · read "
                        f"in {_nu} Earth Engine piece{'' if _nu == 1 else 's'}"
                        + (f', {_nu - _nl} already read' if _nl < _nu else '') + '.'))
                    if _min_m > 15:
                        st.warning(i18n.pick(
                            'Va para largo. Mientras se hace el mapa **no uses la app**: '
                            'cualquier clic lo interrumpe. Si se interrumpe no se pierde '
                            'lo leído: vuelve a pulsar el botón y sigue donde iba. Los '
                            'mapas grandes también gastan más horas de cálculo de Earth '
                            'Engine (las de la cuenta compartida o la cuota de tu '
                            'proyecto).',
                            'This will take a while. While the map is being made **do '
                            'not use the app**: any click interrupts it. If it is '
                            'interrupted, what has been read is not lost: press the '
                            'button again and it carries on where it was. Large maps '
                            'also use more Earth Engine compute hours (those of the '
                            'shared account or the quota of your project).'))

                # ---- dónde cae: la zona y las muestras sobre la ortofoto
                if _rej and st.toggle(i18n.pick('Ver la zona en un mapa',
                                                'Show the zone on a map'),
                                      value=True, key='mrf_ver_zona'):
                    _lz = mra.limites_lonlat(_rej)
                    _lo = [p[0] for p in _pts_m] + [_lz[0], _lz[2]]
                    _la = [p[1] for p in _pts_m] + [_lz[1], _lz[3]]
                    # El encuadre no sigue al rectángulo: así el mapa no se recarga (ni
                    # pierde el zoom) cada vez que se mueve. Lo que cambia va en una
                    # capa que se añade aparte.
                    _clave_enc = (_mod['firma'], _zona, _clave_z[2])
                    _enc = st.session_state.get('_mrf_encuadre')
                    if not _enc or _enc[0] != _clave_enc:
                        _enc = (_clave_enc, mra.encuadre(
                            (min(_lo), min(_la), max(_lo), max(_la)), 600, 320))
                        st.session_state['_mrf_encuadre'] = _enc
                    _pm = _base_map(_enc[1][0], _enc[1][1], ortofoto=True)
                    # La leyenda: qué categoría es cada color de punto y qué es cada
                    # recuadro. Abajo a la derecha, donde no tapa ni el zoom ni la escala.
                    import html as _html
                    # las categorías con las que se pintan las muestras, en su orden
                    _de_muestras = {_cat_muestra(m) for m in _mod['muestras']}
                    _cats_zona = ([c for c in _mod['categorias']
                                   if c['categoria'] in _de_muestras]
                                  + [{'categoria': c} for c in mr.NIVELES[_nivel]
                                     if c in _de_muestras and _cercania])
                    _marco = ('<div style="margin:3px 0 0;"><span style="display:inline-block;'
                              'width:14px;height:9px;margin-right:6px;vertical-align:middle;'
                              'border:2px solid {0};background:{1};"></span>{2}</div>')
                    _pm.get_root().html.add_child(folium.Element(
                        '<div style="position:fixed;bottom:28px;right:10px;z-index:9999;'
                        'background:rgba(255,255,255,0.92);padding:7px 9px;border-radius:6px;'
                        'border:1px solid #999;font-size:12px;line-height:1.15;color:#111;'
                        'box-shadow:0 1px 4px rgba(0,0,0,0.3);">'
                        '<div style="font-weight:bold;margin-bottom:4px;">'
                        + i18n.pick('Muestras del modelo', 'Model samples') + '</div>'
                        + _leyenda_cats(
                            _cats_zona,
                            lambda col: (
                                '<span style="display:inline-block;'
                                'width:11px;height:11px;border-radius:50%;margin-right:6px;'
                                'vertical-align:middle;border:1px solid #111;background:'
                                f'{col};"></span>'),
                            lambda c, i: mra.color(c['categoria'], i))
                        + '<div style="height:4px;"></div>'
                        + _marco.format('#FFEB3B', 'rgba(255,235,59,.15)', i18n.pick(
                            'Lo que se va a cartografiar', 'What will be mapped'))
                        + (_marco.format('#00E5FF', 'rgba(0,229,255,.15)', i18n.pick(
                            'Máscara', 'Mask')) if _masc is not None else '')
                        + (_marco.format('#0D47A1', 'rgba(25,118,210,.10)', i18n.pick(
                            'Área de estudio', 'Study area')) if sg is not None else '')
                        + '</div>'))
                    _gz = folium.FeatureGroup(name=i18n.pick('Zona y muestras',
                                                             'Zone and samples'))
                    if _masc is not None:
                        if 'dibujo' not in _mc:
                            _mc['dibujo'] = mra.anillos_para_dibujar(_masc)
                        for _anillo in _mc['dibujo']:
                            folium.Polygon(_anillo, color='#00E5FF', weight=1.5, fill=True,
                                           fill_color='#00E5FF',
                                           fill_opacity=0.10).add_to(_gz)
                    folium.Polygon(mra.contorno_lonlat(_rej), color='#FFEB3B', weight=3,
                                   fill=True, fill_color='#FFEB3B',
                                   fill_opacity=0.08).add_to(_gz)
                    for _mu in _mod['muestras']:
                        folium.CircleMarker(
                            [_mu['lat'], _mu['lon']], radius=5, color='#111111', weight=1,
                            fill=True, fill_color=mra.color(_cat_muestra(_mu)),
                            fill_opacity=1.0,
                            tooltip=i18n.tv(_cat_muestra(_mu))).add_to(_gz)
                    _rp = st_folium(_pm, width=None, height=320, feature_group_to_add=_gz,
                                    returned_objects=(['last_clicked'] if _zona == 'rect'
                                                      else []),
                                    key='mrf_mapa_zona')
                    st.caption(i18n.pick(
                        'Los puntos son las muestras con las que se entrenó el modelo; su '
                        'color dice su categoría (leyenda, abajo a la derecha).'
                        + (' **Pincha en el mapa** para llevar allí el centro del '
                           'rectángulo.' if _zona == 'rect' else ''),
                        'The points are the samples the model was trained with; their '
                        'colour tells their category (legend, bottom right).'
                        + (' **Click on the map** to move the centre of the rectangle '
                           'there.' if _zona == 'rect' else '')))
                    _c = (_rp or {}).get('last_clicked') if isinstance(_rp, dict) else None
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
                if st.button(i18n.pick('🗺️ Hacer el mapa', '🗺️ Make the map'),
                             key='mrf_hacer_mapa', type='primary',
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
                            _barra = st.progress(0.0, text=i18n.pick(
                                'Preparando la lectura en Earth Engine…',
                                'Preparing the Earth Engine reading…'))
                            _t0 = time.time()

                            def _avance(hechos, total):
                                _queda = (time.time() - _t0) / hechos * (total - hechos) / 60
                                _barra.progress(hechos / total, text=i18n.pick(
                                    f'Bloque {hechos} de {total}'
                                    + (f' · falta alrededor de {_dura(_queda)}'
                                       if hechos < total else ''),
                                    f'Block {hechos} of {total}'
                                    + (f' · about {_dura(_queda)} left'
                                       if hechos < total else '')))

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
            if _mapa:
                _meta = _mapa['meta']
                _cats_m = _meta['categorias']
                _carp_m = Path(_mapa['carpeta'])
                _rej_m = _meta['rejilla']
                # con qué umbral se decidió «¿borreguil o no?» (None: sin esa regla)
                _u_mapa = (_meta.get('regla') or {}).get('umbral')
                _con_pb = (_carp_m / mra.FICHEROS['pborr']).exists()
                # la pureza dentro del píxel, si el mapa se hizo con ella
                # el ambiente por cercanía, si el mapa se hizo con él
                _am = _meta.get('ambiente') or {}
                _con_am = bool(_am) and (_carp_m / mra.FICHEROS['referencias']).exists()
                # las probabilidades son siempre las del modelo
                _cats_p = _meta.get('categorias_modelo') or _cats_m
                _pz = _meta.get('pureza') or {}
                _con_pz = bool(_pz) and (_carp_m / mra.FICHEROS['pureza']).exists()
                st.divider()
                st.markdown(i18n.pick('#### El mapa', '#### The map'))
                st.caption(i18n.pick(
                    f"Hecho el {_meta['cuando'].replace('T', ' ')} · años "
                    f"{bp.abreviar_anios(_meta.get('anios', []))} · píxeles de "
                    f"{_meta['resolucion_m']} m.",
                    f"Made on {_meta['cuando'].replace('T', ' ')} · years "
                    f"{bp.abreviar_anios(_meta.get('anios', []))} · "
                    f"{_meta['resolucion_m']} m pixels.")
                    + ('' if _u_mapa is None else i18n.pick(
                        f' Borreguil si la suma de la probabilidad de sus tipos llega a '
                        f'{_num(_u_mapa)}.',
                        f' Borreguil if the probabilities of its types add up to '
                        f'{_num(_u_mapa)} or more.')))
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

                _n_cl = _meta['pixeles_clasificados']
                _n_zo = _meta.get('pixeles_zona') or _n_cl
                _k1, _k2, _k3 = st.columns(3, vertical_alignment='bottom')
                _k1.metric(i18n.pick('Superficie clasificada', 'Classified area'),
                           f'{_num(_n_cl / 100)} ha')
                _k2.metric(i18n.pick('Probabilidad media de la categoría elegida',
                                     'Mean probability of the chosen category'),
                           _num(_meta['pmax_media']))
                _k3.metric(i18n.pick('Píxeles de la zona sin dato',
                                     'Pixels of the zone without data'),
                           f"{_num(100 * (_n_zo - _n_cl) / _n_zo)} %" if _n_zo else '—',
                           help=i18n.pick(
                               'Píxeles de la zona a los que les falta alguna variable '
                               '(nubes o nieve todos los años, borde de las imágenes): '
                               'no se clasifican ni se rellenan.',
                               'Pixels of the zone that lack some variable (cloud or '
                               'snow every year, edge of the images): they are neither '
                               'classified nor filled in.'))

                # La leyenda: los mismos nombres y colores en el mapa y en las muestras.
                import html as _html

                def _cuadro(col):
                    return ('<span style="display:inline-block;'
                            f'width:.85rem;height:.85rem;background:{col};'
                            'border:1px solid rgba(128,128,128,.9);border-radius:2px;'
                            'margin-right:.4rem;vertical-align:-.1rem"></span>')
                # En el nivel combinado, cada ambiente con sus dos humedades al lado.
                _ley, _abierto = [], False
                for _f in mr.filas_de_leyenda(_cats_m):
                    if 'titulo' in _f or not _f['rama']:
                        _ley.append('</span>' if _abierto else '')
                        _abierto = 'titulo' in _f
                    if 'titulo' in _f:
                        _ley.append('<span style="white-space:nowrap"><b>'
                                    f'{_html.escape(i18n.tv(_f["titulo"]))}:</b>')
                    elif _f['rama']:
                        _ley.append('<span style="margin-left:.55rem">' + _cuadro(_f['color'])
                                    + f'{_html.escape(i18n.tv(_f["etiqueta"]))}</span>')
                    else:
                        _ley.append('<span style="white-space:nowrap">' + _cuadro(_f['color'])
                                    + f'{_html.escape(i18n.tv(_f["etiqueta"]))}</span>')
                st.html(
                    '<div style="display:flex;flex-wrap:wrap;gap:.35rem 1.1rem;'
                    'font-size:.9rem;line-height:1.3">' + ''.join(_ley)
                    + ('</span>' if _abierto else '') + '</div>')
                _k_ha, _k_pc = i18n.pick('Hectáreas', 'Hectares'), i18n.pick(
                    '% de lo clasificado', '% of what is classified')
                st.dataframe(_con_coma(pd.DataFrame([{
                    i18n.pick('Categoría', 'Category'): i18n.tv(c['categoria']),
                    i18n.pick('Código', 'Code'): c['codigo'],
                    _k_ha: c['hectareas'],
                    _k_pc: (100 * c['pixeles'] / _n_cl) if _n_cl else 0.0}
                    for c in _cats_m])), hide_index=True, width='stretch')
                # (en el de sí o no, la tabla de arriba ya va por ambiente)
                if _meta.get('ramas') and _meta.get('nivel') != mr.NIVEL_SI_NO:
                    st.markdown(i18n.pick('**Por ambiente** (húmedo más seco): ',
                                          '**By environment** (wet plus dry): ')
                                + ' · '.join(
                                    f"{i18n.tv(r['rama'])} **{_num(r['hectareas'])} ha**"
                                    for r in _meta['ramas']) + '.')
                if _con_am:
                    # los cauces de OpenStreetMap, o los de una capa subida
                    _am_osm = str((_am.get('fuentes') or {}).get('cauces', '')).startswith(
                        'OpenStreetMap')
                    st.caption(i18n.pick(
                        'El ambiente no sale del modelo, sino de la cercanía de cada píxel '
                        f"a lagunas y cauces. {mam.frase(_am)[:1].upper()}{mam.frase(_am)[1:]}. "
                        + (('Cauces: los ríos y arroyos de OpenStreetMap.'
                            if _am_osm else 'Cauces: los de tu capa.')
                           if _am['cauces'] == 'capa' else
                           f"Cauces: deducidos del relieve (los que reciben el agua de "
                           f"{_num(_am['cauce_ha'])} ha o más), porque OpenStreetMap no "
                           'respondió; son una aproximación.')
                        + ' Lo que se ha tomado por laguna y por cauce se ve en la capa '
                        '**Lagunas y cauces** del mapa.',
                        'The environment does not come from the model but from how close '
                        'each pixel is to lakes and watercourses. '
                        f"{mam.frase(_am, True)[:1].upper()}{mam.frase(_am, True)[1:]}. "
                        + (('Watercourses: the OpenStreetMap rivers and streams.'
                            if _am_osm else 'Watercourses: those of your layer.')
                           if _am['cauces'] == 'capa' else
                           f"Watercourses: derived from the relief (those receiving the "
                           f"water of {_num(_am['cauce_ha'])} ha or more), because "
                           'OpenStreetMap did not answer; they are an approximation.')
                        + ' What was taken as lake and as watercourse shows in the '
                        '**Lakes and watercourses** layer of the map.'))
                    # (las distancias de ahora, de los controles: si no están en
                    # pantalla, no hay nada que comparar)
                    _ahora = {'laguna_capa_m': 'mrf_amb_capa', 'laguna_agua_m': 'mrf_amb_agua',
                              'arroyo_m': 'mrf_amb_arroyo'}
                    if any(st.session_state.get(w) is not None
                           and abs(_am.get(k, 0.0) - float(st.session_state[w])) > 1e-9
                           for k, w in _ahora.items()):
                        st.warning(i18n.pick(
                            'Este mapa se hizo con otras distancias de ambiente. Vuelve a '
                            'hacerlo para aplicar las de ahora: lo ya leído se aprovecha, '
                            'así que tarda poco.',
                            'This map was made with other environment distances. Make it '
                            'again to apply the current ones: what has been read is '
                            'reused, so it takes little time.'))
                    # Lo que revisaste frente a lo que dice la cercanía, en las muestras
                    # de borreguil que caen dentro del mapa.
                    _cmp_a = [(m['ambiente'], m['ambiente_mapa']) for m in _mapa['muestras']
                              if m.get('ambiente') and m.get('ambiente_mapa')]
                    if _cmp_a:
                        _ambs = list(mr.NIVELES[_cercania or 'ambiente'])
                        st.markdown(i18n.pick(
                            f'De las **{len(_cmp_a)} muestras** de borreguil que caen en '
                            'este mapa, el ambiente por cercanía coincide con el que '
                            f'revisaste en **{sum(a == b for a, b in _cmp_a)}**:',
                            f'Of the **{len(_cmp_a)} borreguil samples** that fall on this '
                            'map, the environment by closeness matches the one you '
                            f'reviewed in **{sum(a == b for a, b in _cmp_a)}**:'))
                        st.dataframe(pd.DataFrame([{
                            i18n.pick('Ambiente revisado', 'Reviewed environment'): i18n.tv(a),
                            **{i18n.pick(f'Sale {b}', f'Comes out {i18n.tv(b)}'):
                               sum(1 for x, y in _cmp_a if x == a and y == b) for b in _ambs}}
                            for a in _ambs]), hide_index=True, width='stretch')

                # ---- la pureza dentro del píxel, de lo que es borreguil
                if _con_pz:
                    _n_bo = sum(c['pixeles'] for c in _pz['categorias'])
                    st.markdown(i18n.pick('**Pureza dentro del píxel** (de lo que es borreguil)',
                                          '**Purity within the pixel** (of what is borreguil)'))
                    _k_pb = i18n.pick('% del borreguil', '% of the borreguil')
                    st.dataframe(_con_coma(pd.DataFrame([{
                        i18n.pick('Pureza', 'Purity'): i18n.tv(c['categoria']),
                        i18n.pick('Código', 'Code'): c['codigo'],
                        _k_ha: c['hectareas'],
                        _k_pb: (100 * c['pixeles'] / _n_bo) if _n_bo else 0.0}
                        for c in _pz['categorias']])), hide_index=True, width='stretch')
                    st.caption(i18n.pick(
                        'Estimada con el NDVI de inicio de verano de cada píxel: sin verde '
                        f"{_num(_pz['sin_verde'])}, todo verde {_num(_pz['todo_verde'])}; puro "
                        f"desde el {100 * _pz['puro_desde']:.0f} % de verde. Mixto-agua, si "
                        'el píxel es agua o toca un píxel de agua.',
                        'Estimated from the early-summer NDVI of each pixel: no green '
                        f"{_num(_pz['sin_verde'])}, all green {_num(_pz['todo_verde'])}; pure "
                        f"from {100 * _pz['puro_desde']:.0f} % green. Mixed-water, if the "
                        'pixel is water or touches a water pixel.'))
                    if _nivel == 'pureza':
                        st.caption(i18n.pick(
                            'No es la pureza de la tabla de arriba: aquella es la que predice '
                            'el modelo a partir de tus muestras; esta es una estimación aparte, '
                            'hecha solo con el NDVI. No tienen por qué coincidir.',
                            'This is not the purity in the table above: that one is what the '
                            'model predicts from your samples; this one is a separate estimate, '
                            'made only from the NDVI. They need not agree.'))

                # ---- visor: el mapa, la confianza y las muestras encima
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
                # colores se dicen en el texto de debajo.
                _refs_cats = [{'codigo': c, 'categoria': n, 'color': mam.COLORES_REFERENCIAS[c]}
                              for c, n in mam.REFERENCIAS.items()]

                # Va en un fragmento: al moverse por el mapa, pinchar un píxel o cambiar la
                # opacidad se repite solo esta parte, no el guion entero (que con cientos
                # de puntos son segundos).
                @st.fragment
                def _visor_del_mapa():
                    _v1, _v2 = st.columns([3, 2], vertical_alignment='bottom')
                    _nombres_m = [c['categoria'] for c in _cats_p]
                    _prob_de = _v1.selectbox(
                        i18n.pick('Ver también la probabilidad de una categoría',
                                  'Also show the probability of a category'),
                        [''] + _nombres_m, key='mrf_ver_prob',
                        format_func=lambda c: i18n.tv(c) if c else i18n.pick('— ninguna —',
                                                                             '— none —'),
                        help=i18n.pick(
                            'Añade al recuadro de capas del mapa una capa más, con la '
                            'probabilidad que el modelo da a esa categoría en cada píxel, '
                            'la elija o no. Sirve para ver dónde ve ese tipo de borreguil '
                            'y las zonas de transición.\n\n'
                            'La **categoría**, la **probabilidad máxima** y la '
                            '**probabilidad de borreguil** están siempre en ese recuadro, '
                            'arriba a la derecha del mapa.',
                            'Adds one more layer to the layer box of the map, with the '
                            'probability the model gives to that category at each pixel, '
                            'whether it chooses it or not. Useful to see where it sees '
                            'that type of borreguil and the transition zones.\n\n'
                            'The **category**, the **maximum probability** and the '
                            '**probability of borreguil** are always in that box, at the '
                            'top right of the map.'))
                    if _prob_de not in _nombres_m:
                        _prob_de = ''
                    _opac = _v2.slider(
                        i18n.pick('Opacidad de las capas', 'Layer opacity'), 0.2, 1.0, 0.7, 0.1,
                        key='mrf_opacidad',
                        help=i18n.pick(
                            'Con menos opacidad se ve la ortofoto por debajo.\n\n'
                            '**Recomendado: 0.7.**',
                            'With less opacity the orthophoto shows through.\n\n'
                            '**Recommended: 0.7.**'))

                    # ---- lo que el visor dice que está mostrando, y dónde se ha pinchado
                    # (su estado queda en st.session_state con su clave)
                    _clave_v = f'mrf_visor_{_carp_m.name}'
                    _est = st.session_state.get(_clave_v)
                    _centro_m, _zoom_m = mra.encuadre(_meta['limites_lonlat'], 600, 520)
                    _vista, _zoom = mra.leer_vista(_est) or (
                        mra.vista_inicial(_centro_m, _zoom_m), _zoom_m)
                    _clic_m = mra.leer_clic(_est)
                    _q = mra.consultar(_carp_m, *_clic_m) if _clic_m else None

                    # ---- lo fijo del visor. No cambia de una pasada a otra, así que el
                    # visor no se recarga, no pierde el zoom y no olvida qué capas tenía
                    # encendidas la persona. Las tres capas ráster nacen vacías: la
                    # imagen se les pone después (ver «lo que cambia»).
                    _fm = _base_map(_centro_m, _zoom_m, ortofoto=True)
                    _om, _sm, _em, _nm = _meta['limites_lonlat']
                    _fijas = {
                        'clase': folium.raster_layers.ImageOverlay(
                            image=mra.PNG_VACIO, bounds=[[_sm, _om], [_nm, _em]], opacity=0.7,
                            name=i18n.pick('Categoría', 'Category'), show=True,
                            pixelated=True),
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
                            name=i18n.pick('Pureza dentro del píxel', 'Purity within the pixel'),
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
                    folium.Polygon(mra.contorno_lonlat(_rej_m), color='#FFFFFF', weight=1.5,
                                   dash_array='6', fill=False).add_to(_fm)
                    # Las muestras no se pintan en este visor: lo pidió ella («quita los
                    # puntos del mapa»), para ver solo lo que dice el modelo. Siguen en el
                    # mapa pequeño de la zona, el de antes de hacer el mapa.
                    _fm.get_root().html.add_child(folium.Element(
                        # Los píxeles, con los bordes nítidos; y el recuadro de capas, sin
                        # una barra de desplazamiento que aquí nunca hace falta (sale si el
                        # visor se crea antes de tener tamaño).
                        '<style>.leaflet-image-layer{image-rendering:crisp-edges;'
                        'image-rendering:pixelated}'
                        '.leaflet-control-layers-scrollbar{overflow-y:visible;'
                        'padding-right:0;height:auto!important}</style>'
                        # por encima de la escala gráfica, que también va abajo a la izquierda
                        '<div style="position:fixed;bottom:58px;left:12px;z-index:9999;'
                        'background:rgba(255,255,255,0.92);padding:8px 10px;border-radius:6px;'
                        'border:1px solid #999;font-size:12px;line-height:1.15;color:#111;'
                        'box-shadow:0 1px 4px rgba(0,0,0,0.3);">'
                        '<div style="font-weight:bold;margin-bottom:4px;">'
                        + i18n.pick('Categoría', 'Category')
                        + '</div>' + _leyenda_cats(
                            _cats_m, lambda col: (
                                '<span style="display:inline-block;'
                                f'width:12px;height:12px;background:{col};margin-right:6px;'
                                'vertical-align:middle;border:1px solid #555;"></span>'))
                        + ('' if not _con_pz else (
                            '<div style="font-weight:bold;margin:7px 0 4px;">'
                            + i18n.pick('Pureza dentro del píxel', 'Purity within the pixel')
                            + '</div>' + ''.join(
                                '<div style="margin:2px 0;"><span style="display:inline-block;'
                                f'width:12px;height:12px;background:{c["color"]};'
                                'margin-right:6px;vertical-align:middle;border:1px solid #555;">'
                                f'</span>{_html.escape(i18n.tv(c["categoria"]))}</div>'
                                for c in _pz['categorias'])))
                        + '<div style="font-weight:bold;margin:7px 0 3px;">'
                        + i18n.pick('Probabilidad', 'Probability') + '</div>'
                        '<div style="width:130px;height:10px;border:1px solid #555;'
                        f'background:linear-gradient(to right,{",".join(mra.RAMPA)});"></div>'
                        '<div style="width:132px;display:flex;justify-content:space-between;">'
                        '<span>0</span><span>0.5</span><span>1</span></div></div>'))
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
                            _vistas.setdefault('clase', {}), _carp_m / mra.FICHEROS['clase'], 1,
                            _vista, _zoom, lambda m: mra.pintar_clases(m, _cats_m, 1.0),
                            categorica=True),
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
                        _rotulo = i18n.pick(f'Probabilidad de «{i18n.tv(_prob_de)}»',
                                            f'Probability of "{i18n.tv(_prob_de)}"')
                        _imgs['px'] = mra.dibujar_vista(
                            _vistas.setdefault(_elegida, {}), _carp_m / mra.FICHEROS['prob'],
                            _ip + 1, _vista, _zoom, lambda m: mra.pintar_probabilidad(m, 1.0))
                    _g_din = folium.FeatureGroup(name='lo que cambia')
                    _Guion(lambda: mra.guion_capas(_imgs, _opac, _elegida, _rotulo)).add_to(_g_din)
                    if _q:
                        folium.Polygon(mra.contorno_pixel(_rej_m, _q['columna'], _q['fila']),
                                       color='#FF1744', weight=3, fill=False).add_to(_g_din)
                    # El visor avisa de cada movimiento y de cada clic, y entonces se
                    # repite este fragmento.
                    st_folium(_fm, width=None, height=560, feature_group_to_add=_g_din,
                              returned_objects=['last_clicked', 'bounds', 'zoom'],
                              pixelated=True, key=_clave_v)
                    st.caption(i18n.pick(
                        'Las capas se encienden y apagan en el recuadro de arriba a la '
                        'derecha del mapa. **Probabilidad máxima** es la de la categoría que '
                        'el modelo ha puesto en cada píxel, **sea la que sea**: cuanto más '
                        'clara, más seguro está. Donde pone «no borreguil», clara quiere '
                        'decir que está seguro de que **no** lo es.',
                        'Layers are switched on and off in the box at the top right of the '
                        'map. **Maximum probability** is that of the category the model has '
                        'assigned to each pixel, **whichever it is**: the lighter, the surer '
                        'it is. Where it says "not a borreguil", light means it is sure '
                        'that it is **not** one.')
                        + ('' if _u_mapa is None or not _con_pb else ' ' + i18n.pick(
                            '**Probabilidad de borreguil** es la suma de la de sus tipos: '
                            f'donde llega a {_num(_u_mapa)}, el mapa pone borreguil, del tipo '
                            'más probable. Es la capa para ver dónde hay borreguil.',
                            '**Probability of borreguil** is the sum of those of its '
                            f'types: where it reaches {_num(_u_mapa)}, the map says '
                            'borreguil, of the most probable type. It is the layer to see '
                            'where there is borreguil.'))
                        + ('' if not _con_am else ' ' + i18n.pick(
                            '**Lagunas y cauces** enseña lo que se ha tomado por laguna y '
                            'por cauce para poner el ambiente (en turquesa, el agua de '
                            'laguna que ve Sentinel-2; en azul, los cauces; en rosa, las '
                            'lagunas de la capa): si ahí falta una laguna o sobra un '
                            'cauce, el ambiente de alrededor saldrá mal.',
                            '**Lakes and watercourses** shows what was taken as lake and '
                            'as watercourse to set the environment (turquoise: the lake '
                            'water Sentinel-2 sees; blue: watercourses; pink: the lakes of '
                            'the layer): if a lake is missing there or a watercourse '
                            'should not be, the environment around it will come out wrong.'))
                        + ('' if not _con_pz else ' ' + i18n.pick(
                            '**Pureza dentro del píxel** dice, de cada píxel de borreguil, '
                            'si es puro o está mezclado con agua o con roca, según el verde '
                            'que se estima que tiene dentro.',
                            '**Purity within the pixel** tells, for each borreguil pixel, '
                            'whether it is pure or mixed with water or rock, according to '
                            'the green estimated inside it.')))
                    st.caption(i18n.pick(
                        'Lo pintado son los píxeles de 10 m de Sentinel-2, cada uno en su '
                        'sitio: acércate para verlos. Que el mapa coincida con las muestras '
                        'con las que se entrenó el modelo no mide su acierto; para eso está '
                        'la validación por zonas de «3 · Modelo».',
                        'What is drawn are the 10 m Sentinel-2 pixels, each in its place: '
                        'zoom in to see them. The map agreeing with the samples the model '
                        'was trained with does not measure its accuracy; that is what the '
                        'validation by zones under "3 · Model" is for.'))

                    def _a_cuanto(d, en):
                        """Las tres distancias de las que sale el ambiente de un píxel."""
                        def _una(v, es_, en_, nada_es, nada_en):
                            if v is None:
                                return nada_en if en else nada_es
                            return (f'{v:.0f} m {en_}' if en else f'a {v:.0f} m {es_}')
                        return '; '.join((
                            _una(d.get('agua'), 'del agua de una laguna', 'from lake water',
                                 'sin agua de laguna a la vista', 'no lake water in sight'),
                            _una(d.get('capa'), 'de una laguna de la capa',
                                 'from a lake of the layer',
                                 'a más de 1 km de las lagunas de la capa',
                                 'more than 1 km from the lakes of the layer'),
                            _una(d.get('cauce'), 'de un cauce', 'from a watercourse',
                                 'sin cauces cerca', 'no watercourse nearby')))

                    # ---- lo que dice el mapa en el píxel pinchado
                    if _q is None:
                        st.info(i18n.pick(
                            '**Pincha en el mapa** para ver qué dice el modelo en ese píxel: '
                            'la categoría, su probabilidad y la del resto.',
                            '**Click on the map** to see what the model says at that pixel: '
                            'the category, its probability and that of the others.')
                            if not _clic_m else i18n.pick(
                                'Has pinchado fuera del mapa.', 'You clicked outside the map.'))
                    else:
                        _donde = i18n.pick(
                            f"lon {_num(_q['lon'], 5)}; lat {_num(_q['lat'], 5)}",
                            f"lon {_num(_q['lon'], 5)}, lat {_num(_q['lat'], 5)}")
                        _donde_mas = i18n.pick(
                            f"fila {_q['fila'] + 1}, columna {_q['columna'] + 1} del mapa · "
                            f"centro en X {_q['x']:.0f}, Y {_q['y']:.0f} "
                            f"({_meta['crs']})",
                            f"row {_q['fila'] + 1}, column {_q['columna'] + 1} of the map · "
                            f"centre at X {_q['x']:.0f}, Y {_q['y']:.0f} "
                            f"({_meta['crs']})")
                        if _q['codigo'] is None:
                            st.info(i18n.pick(
                                f'**Píxel sin dato** ({_donde}). Está fuera de la zona o de '
                                'la máscara, o le falta alguna variable.',
                                f'**Pixel without data** ({_donde}). It is outside the zone '
                                'or the mask, or it lacks some variable.'))
                        else:
                            st.markdown(i18n.pick(
                                f"**Píxel pinchado:** «{i18n.tv(_q['categoria'])}» (código "
                                f"{_q['codigo']}), con probabilidad "
                                f"**{_num(_q['pmax'])}**.",
                                f"**Clicked pixel:** \"{i18n.tv(_q['categoria'])}\" (code "
                                f"{_q['codigo']}), with probability "
                                f"**{_num(_q['pmax'])}**.")
                                + ('' if _q.get('p_borreguil') is None or _u_mapa is None
                                   else ' ' + i18n.pick(
                                       'Probabilidad de borreguil (la suma de sus tipos): '
                                       f"**{_num(_q['p_borreguil'])}**; el umbral es "
                                       f'{_num(_u_mapa)}.',
                                       'Probability of borreguil (the sum of its types): '
                                       f"**{_num(_q['p_borreguil'])}**; the "
                                       f'threshold is {_num(_u_mapa)}.'))
                                + ('' if not _q.get('ambiente') else ' ' + i18n.pick(
                                    f"Ambiente por cercanía: **{i18n.tv(_q['ambiente'])}** ("
                                    + _a_cuanto(_q['distancias'], False) + ').',
                                    'Environment by closeness: '
                                    f"**{i18n.tv(_q['ambiente'])}** ("
                                    + _a_cuanto(_q['distancias'], True) + ').'))
                                + ('' if _q.get('verde') is None else ' ' + i18n.pick(
                                    'Verde dentro del píxel (estimado): '
                                    f"**{100 * _q['verde']:.0f} %**"
                                    + (f"; pureza por ese verde: **{i18n.tv(_q['pureza'])}**."
                                       if _q.get('pureza') else '.'),
                                    'Green within the pixel (estimated): '
                                    f"**{100 * _q['verde']:.0f} %**"
                                    + (f"; purity from that green: **{i18n.tv(_q['pureza'])}**."
                                       if _q.get('pureza') else '.'))))
                            _k_pr = i18n.pick('Probabilidad', 'Probability')
                            st.dataframe(_con_coma(pd.DataFrame([{
                                i18n.pick('Categoría', 'Category'): i18n.tv(p['categoria']),
                                i18n.pick('Código', 'Code'): p['codigo'],
                                _k_pr: p['probabilidad']}
                                for p in sorted(_q['probabilidades'],
                                                key=lambda p: -p['probabilidad'])])),
                                hide_index=True, width='stretch')
                            st.caption(i18n.pick('Dónde: ', 'Where: ') + _donde + ' · '
                                       + _donde_mas + '.')

                _visor_del_mapa()

                # ---- los ficheros
                st.markdown(i18n.pick('**Descargar el mapa** (GeoTIFF, para abrir en QGIS)',
                                      '**Download the map** (GeoTIFF, to open in QGIS)'))

                def _fichero(nombre):
                    return lambda: (_carp_m / nombre).read_bytes()

                _desc = (
                    ('clase', f'mapa_{_nivel}_categoria.tif', 'image/tiff', i18n.pick(
                        'La categoría de cada píxel, como código (255 = sin dato). Lleva '
                        'dentro los colores y la lista de códigos.',
                        'The category of each pixel, as a code (255 = no data). It '
                        'carries the colours and the list of codes.')),
                    ('pmax', f'mapa_{_nivel}_probabilidad_maxima.tif', 'image/tiff', i18n.pick(
                        'La probabilidad, de 0 a 1, de la categoría elegida en cada píxel.',
                        'The probability, from 0 to 1, of the category chosen at each '
                        'pixel.')),
                    ('prob', f'mapa_{_nivel}_probabilidad_por_categoria.tif', 'image/tiff',
                     i18n.pick('Una banda por categoría, con su nombre: la probabilidad de '
                               'cada una en cada píxel.',
                               'One band per category, with its name: the probability of '
                               'each one at each pixel.')),
                    *([('pborr', f'mapa_{_nivel}_probabilidad_borreguil.tif', 'image/tiff',
                        i18n.pick('La probabilidad, de 0 a 1, de ser borreguil del tipo que '
                                  'sea: la suma de la de los tipos. Con el umbral, da el '
                                  'mapa de borreguil sí o no.',
                                  'The probability, from 0 to 1, of being a borreguil of '
                                  'any type: the sum of those of the types. With the '
                                  'threshold, it gives the borreguil yes-or-no map.'))]
                      if _con_pb else []),
                    *([('pureza', f'mapa_{_nivel}_pureza.tif', 'image/tiff', i18n.pick(
                        'La pureza dentro de cada píxel de borreguil, como código: 1 puro, '
                        '2 mixto-agua, 3 mixto-roca (255 = no es borreguil o no hay dato).',
                        'The purity within each borreguil pixel, as a code: 1 pure, 2 '
                        'mixed-water, 3 mixed-rock (255 = not a borreguil or no data).')),
                       ('verde', f'mapa_{_nivel}_verde_estimado.tif', 'image/tiff', i18n.pick(
                           'Cuánto verde hay dentro de cada píxel, de 0 a 1, estimado con '
                           'su NDVI de inicio de verano. De todos los píxeles, sean o no '
                           'borreguil.',
                           'How much green there is within each pixel, from 0 to 1, '
                           'estimated from its early-summer NDVI. For every pixel, '
                           'borreguil or not.'))]
                      if _con_pz else []),
                    *([('ambiente', f'mapa_{_nivel}_ambiente.tif', 'image/tiff', i18n.pick(
                        'El ambiente de cada píxel por cercanía a lagunas y cauces, como '
                        'código: 1 arroyo, 2 laguna, 3 ladera (255 = sin dato). De todos '
                        'los píxeles, sean o no borreguil.',
                        'The environment of each pixel by closeness to lakes and '
                        'watercourses, as a code: 1 stream, 2 lake, 3 slope (255 = no '
                        'data). For every pixel, borreguil or not.')),
                       ('distancias', f'mapa_{_nivel}_distancias_ambiente.tif', 'image/tiff',
                        i18n.pick('Tres bandas, en metros: la distancia de cada píxel al '
                                  'agua de una laguna, a una laguna de la capa y a un '
                                  'cauce. De ellas sale el ambiente.',
                                  'Three bands, in metres: the distance from each pixel to '
                                  'lake water, to a lake of the layer and to a '
                                  'watercourse. The environment comes from them.')),
                       ('referencias', f'mapa_{_nivel}_lagunas_y_cauces.tif', 'image/tiff',
                        i18n.pick('Lo que se ha tomado por laguna y por cauce: 1 agua de '
                                  'laguna (Sentinel-2), 2 cauce, 3 laguna de la capa.',
                                  'What was taken as lake and as watercourse: 1 lake water '
                                  '(Sentinel-2), 2 watercourse, 3 lake of the layer.'))]
                      if _con_am else []),
                    ('codigos', f'mapa_{_nivel}_codigos.csv', 'text/csv', i18n.pick(
                        'Qué categoría es cada código, con su color y su superficie.',
                        'Which category each code is, with its colour and its area.')))
                # Uno debajo de otro: los nombres son largos y en dos columnas se cortan.
                for _cl, _nom, _mime, _ay in _desc:
                    _mb = (_carp_m / mra.FICHEROS[_cl]).stat().st_size / 1e6
                    st.download_button(
                        label=f'⬇ {_nom} ({_num(_mb, 1)} MB)' if _mb >= 0.1 else f'⬇ {_nom}',
                        data=_fichero(mra.FICHEROS[_cl]), file_name=_nom, mime=_mime,
                        key=f'mrf_dl_mapa_{_cl}', help=_ay, on_click='ignore')

        # ------------------------------------------- 5 · Peso (de las variables)
        # Cuánto pesa cada variable en el modelo. Se mide al entrenar, de dos
        # maneras (mapeo_rf.entrenar_y_validar); aquí solo se enseña. No se quita
        # ninguna variable: eso lo decide quien usa la app, en «1 · Datos».
        with _t_peso:
            st.caption(i18n.pick(
                '**Cuánto pesa cada variable en el modelo.** Se mide al entrenar, en '
                '«3 · Modelo»; aquí se consulta. **No se quita ninguna variable**: para '
                'probar sin alguna, desmárcala en «1 · Datos» y vuelve a entrenar.',
                '**How much each variable weighs in the model.** It is measured when '
                'training, under "3 · Model"; here it is only shown. **No variable is '
                'removed**: to try without one, deselect it under "1 · Data" and train '
                'again.'))
            _imp = (_mod or {}).get('importancia')
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
                _vs, _f1, _nrep = _imp['variables'], _imp['base'], _imp['repeticiones']
                _valid = (i18n.pick('la validación por zonas', 'the validation by zones')
                          if _mod['validacion']['tipo'] == 'espacial'
                          else i18n.pick('la validación', 'the validation'))
                st.markdown(i18n.pick(
                    'Se mide de dos maneras, las dos al entrenar:\n\n'
                    '- **Lo que pierde el modelo sin ella** (importancia por permutación). '
                    f'Se repite {_valid} con los valores de esa variable **desordenados** '
                    'entre las muestras, como si dejara de informar, y se mira cuánto baja '
                    f'el F1 medio, que es **{_num(_f1)}**. Se hace {_nrep} veces; el '
                    '«±» es lo que cambia de una vez a otra.\n'
                    '- **Uso interno** (importancia por impureza). Cuánto ha usado el '
                    'modelo final esa variable para separar las muestras al entrenar. Es '
                    'un reparto: entre todas suman 100 %.',
                    'It is measured in two ways, both when training:\n\n'
                    '- **What the model loses without it** (permutation importance). '
                    f'{_valid[:1].upper() + _valid[1:]} is repeated with the values of that '
                    'variable **shuffled** among the samples, as if it stopped telling '
                    'anything, and the drop in the mean F1, which is '
                    f'**{_num(_f1)}**, is recorded. This is done {_nrep} times; the '
                    '"±" is how much it changes from one time to the next.\n'
                    '- **Internal use** (impurity importance). How much the final model '
                    'used that variable to split the samples during training. It is a '
                    'share: all of them add up to 100 %.'))

                # Lo primero que hay que saber, en una frase.
                _1a = _vs[0]
                _claras = [v for v in _vs if v['permutacion'] > 0
                           and v['permutacion'] > 2 * v['desviacion']]
                _nc = len(_claras)
                if _1a['permutacion'] > 0:
                    _titular = i18n.pick(
                        f"**La que más pesa es «{_var(_1a['variable'])}»** "
                        f"({mr.descripcion(_1a['variable'])}): con ella desordenada, el F1 "
                        f"medio baja de {_num(_f1)} a "
                        f"{_num(_f1 - _1a['permutacion'])}.",
                        f"**The one that weighs most is “{_var(_1a['variable'])}”** "
                        f"({mr.descripcion(_1a['variable'], True)}): with it shuffled, the "
                        f"mean F1 drops from {_num(_f1)} to "
                        f"{_num(_f1 - _1a['permutacion'])}.")
                else:
                    _titular = i18n.pick(
                        '**Ninguna variable hace bajar el F1 al desordenarla**: el modelo '
                        'no está sacando partido de ninguna en particular.',
                        '**No variable lowers the F1 when shuffled**: the model is not '
                        'drawing on any one in particular.')
                if _nc:
                    _titular += ' ' + i18n.pick(
                        f"**{_nc} de las {len(_vs)} variables** "
                        f"hace{'' if _nc == 1 else 'n'} bajar el F1 con claridad (más del "
                        'doble de su «±»). En las demás, desordenarlas apenas cambia el '
                        'resultado: o no aportan, o lo que aportan también lo dice otra '
                        'variable parecida.',
                        f"**{_nc} of the {len(_vs)} variables** clearly "
                        f"lower{'s' if _nc == 1 else ''} the F1 (by more than twice their "
                        '"±"). For the rest, shuffling them hardly changes the result: '
                        'either they add nothing, or what they add is also told by a '
                        'similar variable.')
                else:
                    _titular += ' ' + i18n.pick(
                        f'Pero **ninguna de las {len(_vs)} variables** lo hace bajar con '
                        'claridad (más del doble de su «±»): las diferencias entre ellas '
                        'son del tamaño del azar, así que el orden de abajo no es de fiar. '
                        'Suele pasar con pocas muestras o cuando el modelo acierta poco.',
                        f'But **none of the {len(_vs)} variables** lowers it clearly (by '
                        'more than twice its "±"): the differences between them are the '
                        'size of chance, so the order below is not to be trusted. This '
                        'usually happens with few samples or when the model gets little '
                        'right.')
                st.markdown(_titular)

                _med_txt = {
                    'permutacion': i18n.pick('Lo que pierde el modelo sin ella',
                                             'What the model loses without it'),
                    'impureza': i18n.pick('Uso interno', 'Internal use')}

                # Aparte del resto de la página: cambiar de medida solo redibuja esto.
                @st.fragment
                def _grafico_del_peso():
                    import altair as alt
                    _med = st.radio(
                        i18n.pick('Qué medida dibujar', 'Measure to plot'), list(_med_txt),
                        key='mrf_peso_medida', format_func=lambda m: _med_txt[m],
                        horizontal=True,
                        help=i18n.pick(
                            '**Lo que pierde el modelo sin ella** se mide con muestras que '
                            'el modelo no ha visto (las de la validación): dice qué '
                            'variables necesita para acertar en sitios nuevos.\n\n'
                            '**Uso interno** se mide con las mismas muestras con las que '
                            'se entrena: da algo de peso a todas las variables que el '
                            'modelo usa, también a las que no le ayudan a acertar fuera.\n\n'
                            '**Recomendado: la primera** para decidir qué variables '
                            'importan. La segunda es la que suele darse en los artículos.',
                            '**What the model loses without it** is measured with samples '
                            'the model has not seen (those of the validation): it tells '
                            'which variables it needs to get new places right.\n\n'
                            '**Internal use** is measured with the same samples used for '
                            'training: it gives some weight to every variable the model '
                            'uses, including those that do not help it elsewhere.\n\n'
                            '**Recommended: the first one** to decide which variables '
                            'matter. The second is the one usually reported in papers.'))
                    _perm = _med == 'permutacion'
                    _fil = sorted(_vs, key=lambda v: v['orden' if _perm else 'orden_impureza'])
                    try:
                        _oscuro = st.context.theme.type == 'dark'
                    except Exception:
                        _oscuro = False
                    # Una sola serie, un solo color; la tinta del «±» y la línea del
                    # cero, discretas. Cada uno con su versión para fondo oscuro.
                    _azul, _tinta, _linea = (('#3987e5', '#c3c2b7', '#383835') if _oscuro
                                             else ('#2a78d6', '#52514e', '#c3c2b7'))
                    _dfp = pd.DataFrame([{
                        'variable': _var(v['variable']),
                        'que': mr.descripcion(v['variable'], _en),
                        'grupo': mr.nombre_grupo(v['grupo'], _en),
                        'valor': v['permutacion'] if _perm else 100 * v['impureza'],
                        'menos': v['permutacion'] - v['desviacion'],
                        'mas': v['permutacion'] + v['desviacion'],
                        'pierde': f"{_num(v['permutacion'], signo=True)} ± {_num(v['desviacion'])}",
                        'uso': f"{_num(100 * v['impureza'])} %",
                        'puesto': f"{v['orden']} · {v['orden_impureza']}"} for v in _fil])
                    _tx = (i18n.pick('bajada del F1 medio al desordenar la variable',
                                     'drop in mean F1 when the variable is shuffled')
                           if _perm else i18n.pick('uso interno (% del total)',
                                                   'internal use (% of the total)'))

                    def _x(campo):
                        return alt.X(f'{campo}:Q', title=_tx, axis=alt.Axis(orient='top'))

                    _y = alt.Y('variable:N', sort=[_var(v['variable']) for v in _fil], title=None,
                               axis=alt.Axis(labelLimit=220, labelOverlap=False, ticks=False,
                                             domain=False, labelPadding=8))
                    _b = alt.Chart(_dfp)
                    # La barra crece desde el cero: redonda en la punta, recta en la base.
                    _capas = [
                        alt.Chart(pd.DataFrame({'cero': [0.0]})).mark_rule(
                            color=_linea).encode(x=_x('cero')),
                        _b.transform_filter('datum.valor >= 0').mark_bar(
                            size=14, color=_azul, cornerRadiusTopRight=4,
                            cornerRadiusBottomRight=4).encode(x=_x('valor'), y=_y),
                        _b.transform_filter('datum.valor < 0').mark_bar(
                            size=14, color=_azul, cornerRadiusTopLeft=4,
                            cornerRadiusBottomLeft=4).encode(x=_x('valor'), y=_y)]
                    if _perm:
                        _capas.append(_b.mark_rule(color=_tinta, strokeWidth=1.5).encode(
                            x=_x('menos'), x2='mas:Q', y=_y))
                    # Encima, una franja invisible por fila: el ratón no tiene que
                    # acertar en una barra de dos píxeles para ver sus datos.
                    _sobre = alt.selection_point(name='fila_peso', on='pointerover',
                                                 clear='pointerout', fields=['variable'],
                                                 empty=False)
                    _capas.append(_b.mark_bar(size=20, color=_tinta).encode(
                        x=alt.value(0), x2=alt.value('width'), y=_y,
                        opacity=alt.condition(_sobre, alt.value(0.12), alt.value(0.0)),
                        tooltip=[
                            alt.Tooltip('variable:N', title='variable'),
                            alt.Tooltip('que:N', title=i18n.pick('qué mide',
                                                                 'what it measures')),
                            alt.Tooltip('grupo:N', title=i18n.pick('época u origen',
                                                                   'season or source')),
                            alt.Tooltip('pierde:N', title=i18n.pick(
                                'pierde el modelo sin ella', 'the model loses without it')),
                            alt.Tooltip('uso:N', title=i18n.pick('uso interno',
                                                                 'internal use')),
                            alt.Tooltip('puesto:N', title=i18n.pick(
                                'puesto (pérdida · uso)', 'rank (loss · use)'))]
                    ).add_params(_sobre))
                    st.altair_chart(alt.layer(*_capas).properties(height=20 * len(_fil)),
                                    width='stretch')
                    st.caption(i18n.pick(
                        'Cada barra es una variable, de la que más pesa a la que menos. ',
                        'Each bar is a variable, from the one that weighs most to the one '
                        'that weighs least. ')
                        + (i18n.pick(
                            'La raya fina es el «±»: si llega al cero, esa variable no se '
                            'distingue de no aportar nada. ',
                            'The thin line is the "±": if it reaches zero, that variable '
                            'cannot be told apart from adding nothing. ') if _perm else '')
                        + i18n.pick('Pasa el ratón por una fila para ver qué mide.',
                                    'Hover over a row to see what it measures.'))

                _grafico_del_peso()

                with st.expander(i18n.pick('Cómo leerlo, y con qué cuidado',
                                           'How to read it, and with what caution')):
                    st.markdown(i18n.pick(
                        '- **Las variables parecidas se reparten el peso.** El NDMI de '
                        'inicio de verano, el de fin y su media dicen cosas muy parecidas: '
                        'si se desordena una, las otras la cubren, y todas salen con poco '
                        'peso. Por eso están las tablas de «Variables juntas».\n'
                        '- **Cerca de cero, o negativo, no es un error.** Quiere decir que '
                        'desordenar esa variable no empeora el resultado (o lo mejora un '
                        'poco, por azar). Si la bajada es menor que su «±», no se distingue '
                        'de cero.\n'
                        '- **Las dos medidas no tienen por qué coincidir.** El uso interno '
                        'reparte peso entre todas las variables que el modelo usa al '
                        'entrenar; lo que pierde sin ella se mide con muestras que no ha '
                        'visto.\n'
                        '- **Con pocas muestras, el orden baila.** Entre variables de peso '
                        'parecido, el puesto puede cambiar con otra semilla o al añadir '
                        'muestras. Fíate de las diferencias grandes.\n'
                        '- **Que pese no quiere decir que sea la causa.** Dice que al '
                        'modelo le sirve para separar tus categorías con estas muestras, '
                        'no por qué un borreguil es de un tipo u otro.\n'
                        '- **No se quita ninguna variable.** Para probar sin una, '
                        'desmárcala en «1 · Datos» (no hay que volver a leer nada en Earth '
                        'Engine), entrena otra vez y compara el F1 medio.',
                        '- **Similar variables share the weight.** Early-summer NDMI, '
                        'late-summer NDMI and their mean tell very similar things: if one '
                        'is shuffled, the others cover for it, and all of them come out '
                        'with little weight. That is what the "Variables together" tables '
                        'are for.\n'
                        '- **Close to zero, or negative, is not an error.** It means that '
                        'shuffling that variable does not make the result worse (or makes '
                        'it slightly better, by chance). If the drop is smaller than its '
                        '"±", it cannot be told apart from zero.\n'
                        '- **The two measures need not agree.** Internal use shares the '
                        'weight among every variable the model uses when training; what '
                        'it loses without it is measured with samples it has not seen.\n'
                        '- **With few samples, the order wobbles.** Among variables of '
                        'similar weight, the rank may change with another seed or when '
                        'samples are added. Trust the large differences.\n'
                        '- **Weighing a lot does not mean being the cause.** It says the '
                        'variable helps the model separate your categories with these '
                        'samples, not why a borreguil is of one type or another.\n'
                        '- **No variable is removed.** To try without one, deselect it '
                        'under "1 · Data" (nothing has to be read from Earth Engine '
                        'again), train again and compare the mean F1.'))

                # ---- juntas: por familia y por grupo. Tablas fijas: son cortas, y así
                # los nombres largos se parten en dos líneas en vez de cortarse.
                _k_nv = i18n.pick('Variables', 'Variables')
                _k_pj = i18n.pick('Bajada del F1 sin ellas', 'F1 drop without them')
                _k_uj = i18n.pick('Uso interno', 'Internal use')
                st.markdown(i18n.pick('**Variables juntas**', '**Variables together**'))
                st.caption(i18n.pick(
                    'Aquí se desordenan a la vez todas las variables de cada fila. Dos '
                    'variables parecidas se cubren la una a la otra y sueltas parecen '
                    'pesar poco; juntas se ve lo que el modelo pierde sin toda esa '
                    'información.',
                    'Here all the variables in each row are shuffled at once. Two similar '
                    'variables cover for each other and look unimportant on their own; '
                    'together they show what the model loses without all that '
                    'information.'))
                for _k_cj, _campo, _nombre, _lista in (
                        (i18n.pick('Según lo que miden', 'By what they measure'),
                         'familia', mr.nombre_familia, _imp['familias']),
                        (i18n.pick('Según la época o el origen', 'By season or source'),
                         'grupo', mr.nombre_grupo, _imp['grupos'])):
                    # (\u00a0: espacio que no se parte; la cifra y su «±» van juntos)
                    st.table(pd.DataFrame([{
                        _k_cj: _nombre(g[_campo], _en), _k_nv: str(g['n_variables']),
                        _k_pj: f"{_num(g['permutacion'], signo=True)}\u00a0±\u00a0{_num(g['desviacion'])}",
                        _k_uj: f"{_num(100 * g['impureza'])}\u00a0%"} for g in _lista]
                    ).set_index(_k_cj))

                # ---- todas, una por una (lo mismo que el gráfico, con sus números).
                # Los números, delante: lo que mide cada una es largo y va al final.
                st.markdown(i18n.pick('**Todas las variables**', '**All the variables**'))
                _c4v = st.column_config.NumberColumn()
                _c2v = st.column_config.NumberColumn()
                _k_pv = i18n.pick('Bajada del F1 sin ella', 'F1 drop without it')
                _k_ui = i18n.pick('Uso interno (%)', 'Internal use (%)')
                st.dataframe(_con_coma(pd.DataFrame([{
                    'Variable': _var(v['variable']), i18n.pick('Puesto', 'Rank'): v['orden'],
                    _k_pv: v['permutacion'], '±': v['desviacion'],
                    _k_ui: 100 * v['impureza'],
                    i18n.pick('Puesto por uso', 'Rank by use'): v['orden_impureza'],
                    i18n.pick('Qué mide', 'What it measures'):
                        mr.descripcion(v['variable'], _en),
                    i18n.pick('Familia', 'Family'): mr.nombre_familia(v['familia'], _en),
                    i18n.pick('Época u origen', 'Season or source'):
                        mr.nombre_grupo(v['grupo'], _en)}
                    for v in _vs])), hide_index=True, width='stretch',
                    column_config={_k_pv: _c4v, '±': _c4v, _k_ui: _c2v,
                                   'Variable': st.column_config.Column(pinned=True)})
                st.download_button(
                    label=f'⬇ peso_variables_{_nivel}.csv',
                    data=mr.importancia_csv(_mod, _en).encode('utf-8-sig'),
                    file_name=f'peso_variables_{_nivel}.csv', mime='text/csv',
                    key='mrf_dl_peso',
                    help=i18n.pick(
                        'Una fila por variable, con las dos medidas y su puesto en cada '
                        'una. Los números van con seis decimales.',
                        'One row per variable, with both measures and its rank in each. '
                        'The numbers have six decimals.'))

                # ---- Cómo son las que más pesan en cada categoría del modelo, con los
                # mismos gráficos que «Variables del modelo» de Verdad terreno y los
                # colores del mapa (25-11-2026, la autora: «además de peso, podría
                # mostrar gráficos de las variables por tipo de borreguil como en los
                # puntos de verdad terreno»). Con las muestras de la tabla de
                # entrenamiento: sus categorías son las que aprende el modelo.
                if _tab and _tab.get('filas'):
                    st.divider()
                    st.markdown(i18n.pick(
                        '#### Las que más pesan, categoría a categoría',
                        '#### The ones that weigh most, category by category'))
                    _pred_tab = set(_tab.get('predictores') or [])
                    _orden_m = [v['variable'] for v in sorted(_vs, key=lambda v: v['orden'])
                                if v['variable'] in _pred_tab]
                    _vistas_m = {
                        'cajas': i18n.pick('Cajas', 'Boxes'),
                        'curvas': i18n.pick('Curvas de distribución', 'Distribution curves'),
                        'diferencias': i18n.pick('Diferencia estandarizada de medianas',
                                                 'Standardized median difference'),
                        'dos': i18n.pick('Dos variables a la vez', 'Two variables at once')}
                    _m1, _m2 = st.columns([1.4, 0.9])
                    _vista_m = _m1.selectbox(i18n.pick('Cómo verlo', 'How to see it'),
                                             list(_vistas_m), format_func=_vistas_m.get,
                                             key='mrf_dist_vista')
                    _filas_m = _tab['filas']
                    _grp_m, _ord_m, _rest_m = pp.grupos_puntos(_filas_m, 'categoria', None,
                                                               i18n.tv)
                    from collections import Counter as _CntM
                    _cnt_m = _CntM(_grp_m)
                    _cuantos_m = ' · '.join(f'{g}: {_cnt_m[g]}' for g, _ in _ord_m)
                    _ch_m, _como_m = None, ''
                    if _vista_m != 'dos':
                        _n_m = _m2.selectbox(i18n.pick('Cuántas variables',
                                                       'How many variables'),
                                             [4, 8, 12], index=1, key='mrf_dist_n')
                        _vars_m = [(f, _var(f)) for f in _orden_m[:_n_m]]
                        st.caption(i18n.pick(
                            f'Las {len(_vars_m)} variables que más pesan en este modelo '
                            '(lo que pierde sin ellas), de más a menos, en las muestras de la '
                            'tabla de entrenamiento, cada categoría con su color del mapa. '
                            f'Muestras: {_cuantos_m}.',
                            f'The {len(_vars_m)} variables that weigh most in this model '
                            '(what it loses without them), from most to least, in the '
                            'samples of the training table, each category with its map '
                            f'colour. Samples: {_cuantos_m}.'))
                        if _vista_m == 'cajas':
                            _ch_m, _ = pp.chart_boxplots(_filas_m, _vars_m, comparar='categoria',
                                                         traduce=i18n.tv)
                            _como_m = i18n.pick(
                                'Cada caja va del p25 al p75, con la mediana en blanco; los '
                                'bigotes llegan hasta los valores normales y los puntos '
                                'sueltos son atípicos. Si las cajas de dos categorías no se '
                                'tocan, esa variable las separa bien.',
                                'Each box goes from p25 to p75, with the median in white; the '
                                'whiskers reach the normal values and the loose dots are '
                                'outliers. If the boxes of two categories do not touch, that '
                                'variable separates them well.')
                        elif _vista_m == 'curvas':
                            _ch_m = pp.chart_densidades(_filas_m, _vars_m, comparar='categoria',
                                                        traduce=i18n.tv)
                            _como_m = i18n.pick(
                                'Cada curva muestra cómo se reparten los valores de una '
                                'categoría; todas encierran la misma área. Si apenas se '
                                'tocan, esa variable separa bien las categorías. Las de '
                                f'menos de {pp.MIN_GRUPO} muestras no se dibujan.',
                                'Each curve shows how the values of a category are spread; '
                                'they all enclose the same area. If they hardly touch, that '
                                'variable separates the categories well. Those with fewer '
                                f'than {pp.MIN_GRUPO} samples are not drawn.')
                        else:
                            _ch_m = pp.chart_diferencias(_filas_m, _vars_m,
                                                         comparar='categoria',
                                                         traduce=i18n.tv)
                            _como_m = i18n.pick(
                                'Cada punto dice cuánto se aparta la mediana de la categoría '
                                'de la de «no borreguil», en desviaciones típicas de todas '
                                'las muestras: a la derecha, más; a la izquierda, menos. '
                                'Pasar de 1 (o de −1) es una diferencia clara.',
                                'Each dot says how far the median of the category is from '
                                'that of "not a borreguil", in standard deviations of all '
                                'the samples: to the right, more; to the left, less. Beyond '
                                '1 (or −1) is a clear difference.')
                    elif len(_orden_m) >= 2:
                        _d1m, _d2m = st.columns(2)
                        _vxm = _d1m.selectbox(i18n.pick('Eje horizontal', 'Horizontal axis'),
                                              _orden_m, index=0, format_func=_var,
                                              key='mrf_dist_x')
                        _vym = _d2m.selectbox(i18n.pick('Eje vertical', 'Vertical axis'),
                                              _orden_m, index=1, format_func=_var,
                                              key='mrf_dist_y')
                        st.caption(i18n.pick(
                            'Las variables de este modelo, de más a menos peso; de entrada, '
                            f'las dos que más pesan. Muestras: {_cuantos_m}.',
                            "This model's variables, from most to least weight; by "
                            f'default, the two that weigh most. Samples: {_cuantos_m}.'))
                        _ch_m = pp.chart_dos_variables(
                            _filas_m, _vxm, _vym, _var(_vxm), _var(_vym),
                            comparar='categoria', traduce=i18n.tv,
                            nombres=bp.nombres_de_punto([f.get('ID') for f in _filas_m],
                                                        i18n.pick('Punto', 'Point')))
                        _como_m = i18n.pick(
                            'Cada punto es una muestra de la tabla, colocada según las dos '
                            'variables. Si las nubes de colores se separan, combinar esas '
                            'dos variables ayuda a distinguir las categorías.',
                            'Each dot is a sample of the table, placed by the two variables. '
                            'If the coloured clouds separate, combining those two variables '
                            'helps to tell the categories apart.')
                    if _ch_m is not None:
                        st.altair_chart(_ch_m, width='stretch')
                        st.caption(_como_m)
                    else:
                        st.info(i18n.pick(
                            'No hay datos suficientes para este gráfico (por ejemplo, sin '
                            'muestras de «no borreguil» en este nivel no hay con qué '
                            'comparar las diferencias).',
                            'There is not enough data for this chart (for instance, without '
                            '"not a borreguil" samples at this level the differences have '
                            'nothing to be compared with).'))

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
            _e5 = (_falta, i18n.pick('sin calcular', 'not computed yet'))
        elif not _imp_e:
            _e5 = (_ojo, i18n.pick('vuelve a entrenar', 'train again'))
        elif _e3[0] != _hecho:
            _e5 = (_ojo, i18n.pick('es de una tabla anterior', 'from a previous table'))
        elif _imp_e['variables'][0]['permutacion'] <= 0:
            _e5 = (_ojo, i18n.pick('ninguna destaca', 'none stands out'))
        else:
            _e5 = (_hecho, i18n.pick(
                f"la primera: {_var(_imp_e['variables'][0]['variable'])}",
                f"top: {_var(_imp_e['variables'][0]['variable'])}"))
        _icono = {_hecho: ':material/check_circle:', _ojo: ':material/error:',
                  _falta: ':material/radio_button_unchecked:'}
        _pasos = (i18n.pick('1 · Datos', '1 · Data'), i18n.pick('2 · Tabla', '2 · Table'),
                  i18n.pick('3 · Modelo', '3 · Model'), i18n.pick('4 · Mapa', '4 · Map'),
                  i18n.pick('5 · Peso', '5 · Weights'))
        _estado.markdown(' '.join(
            f':{_c}-badge[{_icono[_c]} {_p} · {_txt}]'
            for _p, (_c, _txt) in zip(_pasos, (_e1, _e2, _e3, _e4, _e5))))
