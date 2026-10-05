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
3. Lanzar el pipeline (descarga ESRI, OSM, DEM, Sentinel-2; clasifica con RF)
4. Inspeccionar resultados en un mapa interactivo Folium
5. Descargar Excel + CSV + mapa HTML
"""
import io, os, re, tempfile, sys, threading, time, unicodedata
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

# Carpeta ESCRIBIBLE del usuario. En la versión standalone (ejecutable) APP_DIR es
# una carpeta temporal de solo lectura que se borra al salir, así que lo que el
# usuario guarda (modelos entrenados) debe ir aquí. Sin la variable de entorno
# —modo desarrollo— coincide con APP_DIR y todo sigue igual que antes.
DATA_DIR = Path(os.environ.get('BORREGUIL_DATA_DIR') or APP_DIR)
try:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    DATA_DIR = APP_DIR


def model_path(name):
    """Ruta de un modelo por nombre: primero la carpeta del usuario, luego la de la app."""
    p = DATA_DIR / name
    return p if p.exists() else APP_DIR / name


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
    """Modelos .joblib disponibles (los del usuario + los incluidos; SN primero)."""
    found = {}
    for d in (APP_DIR, DATA_DIR):
        for p in d.glob('*.joblib'):
            if _es_modelo_real(p):
                found.setdefault(p.name, p)   # el de DATA_DIR no pisa al de APP_DIR
    return sorted(found, key=lambda n: (n != 'rf_sierra_nevada.joblib', n))


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


def render_docs():
    """Pestaña de documentación y recomendaciones (cuerpo ancho, no el sidebar)."""
    _has_key = bool(get_secret('PL_API_KEY', None) or os.environ.get('PL_API_KEY', ''))

    st.markdown('## 📖 Documentación y recomendaciones')
    st.markdown(
        'Esta herramienta identifica **borreguiles** (cervunales higroturbosos / '
        '*mountain wet meadows*) combinando teledetección multi-fuente con un '
        'clasificador Random Forest, validado de forma **honesta** (GroupKFold por '
        'cuenca: entrena en unas cuencas y evalúa en otras nunca vistas).')

    # ---- 1. Estrategia de modelos -------------------------------------------
    st.markdown('### 🧭 ¿Qué modelo elegir?')
    st.markdown(
        'Todos los modelos parten de las **mismas 50 variables base** (imagen aérea + '
        'topografía + Sentinel-2 + Sentinel-1). Las variantes añaden una fuente de '
        '**muy alta resolución** que sube el AUC. **La app extrae automáticamente la '
        'fuente extra** que el modelo elegido necesite — tú solo eliges el modelo.')
    st.table({
        'Tu zona': ['🇪🇸 Andalucía · Cataluña · Canarias',
                    '🌍 Fuera de España / resto de CCAA',
                    '🗺️ Cualquier zona, sin claves',
                    '🎯 Andalucía/Cat./Canarias + clave Planet'],
        'Modelo recomendado': ['rf_sierra_nevada_cir',
                               'rf_sierra_nevada_planet',
                               'rf_sierra_nevada',
                               'rf_sierra_nevada_combo'],
        'Variables': ['56', '60', '50', '66'],
        'AUC honesto': ['0,914', '0,888', '0,868', '0,918'],
        'Requisitos': ['Gratis (PNOA IR 0,25 m)',
                       'Clave Planet (PL_API_KEY)',
                       'Ninguno — universal',
                       'Ambos (mejora marginal +0,004)'],
    })
    st.markdown(
        '- **El mejor resultado en Sierra Nevada es el modelo `_cir`, y es gratis** '
        '(0,914). Úsalo siempre que trabajes en Andalucía, Cataluña o Canarias.\n'
        '- **Fuera de España** (o en CCAA sin ortofoto infrarroja) usa `_planet`: '
        'cobertura mundial a 3 m, pero consume cuota de tu suscripción Planet.\n'
        '- El modelo `_combo` (66) es el más alto en términos absolutos, pero la mejora '
        'sobre `_cir` (+0,004) está dentro del ruido: **no compensa** la doble '
        'dependencia salvo casos límite.')

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
        st.warning(
            '**❌ Sin WMS infrarrojo** (usa `_planet` o las 50 base)\n\n'
            '- **Aragón** → Ordesa y Monte Perdido\n'
            '- **Cantabria / Castilla y León** → Picos de Europa\n'
            '- **Madrid / CyL** → Sierra de Guadarrama')
    st.caption(
        'El IGN tiene PNOA Falso Color Infrarrojo a 0,25 m para **toda España**, pero solo '
        'como descarga COG por hojas MTN25 (sin WMS) → conectarlo cubriría también Ordesa, '
        'Picos y Guadarrama. El modelo `_cir` se calibró con Andalucía; en otras CCAA las '
        'variables IR son comparables pero no idénticas → para máxima precisión, entrena '
        'con verdad-terreno local.')

    # ---- 3. Configurar la clave de Planet -----------------------------------
    st.markdown('### 🔑 Configurar el acceso a Planet (PlanetScope 3 m)')
    if _has_key:
        st.success('✓ Clave Planet **detectada**. El modelo `_planet` (y `_combo`) '
                   'extraerá PlanetScope automáticamente al ejecutar el pipeline.')
    else:
        st.info('Aún **no hay clave Planet** configurada. Sigue los pasos de abajo.')
    st.markdown('**Uso local (tu ordenador):** crea el archivo '
                '`borreguil_app/.streamlit/secrets.toml` con este contenido:')
    st.code('PL_API_KEY = "TU_CLAVE_PLANET_AQUI"', language='toml')
    st.markdown(
        'Ese archivo **no se sube a git** (está en `.gitignore`), así que tu clave queda '
        'privada. Reinicia la app y el acceso a Planet se activa solo.\n\n'
        '**Deploy en Streamlit Cloud:** no subas el archivo. En el panel de tu app entra '
        'en **Settings → Secrets** y pega ahí la misma línea `PL_API_KEY = "…"`.')
    st.warning('⚠ La clave da acceso a tu cuenta Planet de pago (consume cuota de km²). '
               'No la compartas en texto plano ni la subas al repositorio. Si crees que '
               'se ha expuesto, rótala en planet.com.')

    # ---- 4. Fuentes de datos y variables ------------------------------------
    st.markdown('### 🛰️ Fuentes de datos y variables del modelo')
    st.table({
        'Fuente': ['Imagen aérea (ESRI / PNOA)', 'Copernicus DEM 30 m',
                   'Sentinel-2 L2A (2017-2025)', 'Sentinel-1 RTC',
                   'PNOA Falso Color IR 0,25 m', 'PlanetScope 8b SR 3 m'],
        'Variables': ['12', '4', '29', '5', '6 (CIR)', '10 (Planet)'],
        'Qué aporta': [
            'Textura, color, patrón y contexto alpino (% roca)',
            'Pendiente, curvatura e índice de humedad topográfica (TWI)',
            'NDVI/NDWI/Clre/NDMI… bi-temporales y estadísticos de verano',
            'Humedad/estructura por radar (terrain-flattened, montaña)',
            'Pseudo-NDVI a 25 cm: nitidez del borde de la mancha',
            'NDVI/NDRE/textura a 3 m: borde abrupto que S2 promedia'],
    })
    st.caption('Las dos últimas filas son las que usan los modelos `_cir` / `_planet`. '
               'La variable más informativa de ambas es la **textura/contraste del NDVI a '
               'alta resolución** — justo el borde nítido del borreguil contra el sustrato '
               'pétreo, que Sentinel-2 (10 m) promedia y pierde.')

    # ---- 4b. Importancia de variables y lectura jerárquica ------------------
    st.markdown('### 🔑 Qué variables pesan más y cómo leer las gráficas')
    st.markdown(
        'El modelo es un **Random Forest**: cada variable recibe una **importancia** '
        '(0–100 %) según cuánto ayuda a separar *borreguil* de *no-borreguil*. En el panel '
        '**📈 Perfil espectral** de cada punto (pestaña **Tabla**) verás este mismo ranking '
        'arriba del todo: **léelo de arriba abajo** e interpreta primero las variables con más '
        'peso. A la derecha de cada barra está el **rango típico de los borreguiles** (p25–p75): '
        'si el valor del punto cae dentro de ese rango, se parece a un borreguil en esa variable.')
    _mp = st.session_state.get('start_model_path') or str(model_path('rf_sierra_nevada_cir.joblib'))
    _rs = load_ref_stats(_mp)
    _ic, _ = pp.chart_importance(_rs, topn=15) if _rs else (None, [])
    if _ic is not None:
        st.altair_chart(_ic, width='stretch')
        st.caption(f'Ranking del modelo seleccionado (`{Path(_mp).name}`). Cambia según el modelo: '
                   'en los `_cir` mandan la **textura/contraste del NDVI a 0,25 m** y la **altitud**; '
                   'en el modelo base (50), la **altitud**, la **caída estacional de NDVI** y el '
                   '**EVI de inicio de verano**. Altitud, pendiente, SAR y textura pesan pero no son '
                   'índices espectrales, por eso no salen en los perfiles estacionales.')
    else:
        st.caption('Selecciona un modelo con `ref_stats` (reentrenado con `train_v5.py`) para ver el ranking.')

    # ---- 5. Cómo usar la app ------------------------------------------------
    st.markdown('### ▶️ Flujo de trabajo')
    st.markdown(
        '1. **Define el área** (vector, WDPA ID o nombre) o sube tus puntos candidatos.\n'
        '2. Elige el **modelo** según la tabla de arriba y la **fuente de imágenes** '
        '(PNOA en España, ESRI fuera).\n'
        '3. Pulsa **Ejecutar pipeline**. La app descarga cada fuente, extrae las '
        'variables y predice.\n'
        '4. Revisa el **mapa** y la **tabla** de resultados; ajusta el umbral; descarga '
        'Excel/CSV/GeoJSON.\n'
        '5. *(Opcional)* marca borreguiles verificados en campo y **reentrena** en tu '
        'zona (presence-only, mínimo ~30 positivos) para un modelo propio.')


def run_self_training_iteration(base_rows, promote_cutoff, mode, base_truth_ids,
                                threshold, model_path):
    """Una iteración de auto-entrenamiento (self-training / pseudo-labeling).

    - Promueve a positivo cada punto con rf_proba >= promote_cutoff.
    - mode='add'     → acumula: positivos = positivos previos ∪ promovidos.
    - mode='replace' → refresca: positivos = verdad-terreno verificada ∪ promovidos
      del modelo actual (no acumula pseudo-positivos antiguos).
    Reentrena el RF (las features ya están), repredice y devuelve (rows2, info, stats).
    """
    rows2 = [dict(r) for r in base_rows]

    # Etiquetas para el reentrenamiento. Las que puso una persona (ausencias de
    # campo, puntos dudosos) se respetan y no se promueven.
    sets = bp.self_training_labels(rows2, promote_cutoff, mode, base_truth_ids)

    info = bp.run_rf_and_decide(rows2, threshold=threshold, default_model_path=model_path)

    # Distinguir en la decisión: verificado de campo vs pseudo-positivo automático
    bp.self_training_decisions(rows2, sets)

    # Las decisiones han cambiado: se recalcula el tipo (lo revisado se conserva).
    bp.apply_typology(rows2, pp.is_borreguil_decision)

    stats = {
        'n_promoted': len(sets['promoted']),
        'n_prev_pos': len(sets['prev_pos']),
        'n_new': len(sets['promoted'] - sets['prev_pos']),
        # los dudosos no entrenan aunque estén etiquetados como borreguil
        'n_train_pos': len(sets['new_pos'] - sets['dudosos']),
        'n_base': len(sets['base']),
    }
    return rows2, info, stats


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


def render_extraction_report(rows, skip_s2=False, skip_imgs=False):
    """Informe SIEMPRE visible de qué fuentes se extrajeron y con qué cobertura.
    Resuelve el 'no sé qué ha funcionado': muestra puntos con dato por grupo y
    avisa con causa probable cuando una fuente crítica quedó vacía. Las fuentes
    omitidas a propósito (casillas «Saltar…») se marcan como omitidas, sin alarma."""
    N = len(rows)
    if not N:
        return
    img_feats = list(bp.FEATURES_RF[:12])
    topo_feats = ['elev_dem_m', 'slope_deg', 'twi', 'curvature']
    s1_feats = list(bp.S1_FEATS)
    s2_feats = [f for f in bp.FEATURES_RF
                if f not in img_feats and f not in topo_feats and f not in s1_feats]
    # (etiqueta, features, id, ¿omitida a propósito?)
    groups = [('🖼️ Imagen aérea (textura/patrón)', img_feats, 'img', skip_imgs),
              ('⛰️ Topografía (DEM)', topo_feats, 'topo', False),
              ('🛰️ Sentinel-2 (NDVI, Clre…)', s2_feats, 's2', skip_s2),
              ('📡 Sentinel-1 RTC (SAR)', s1_feats, 's1', skip_s2)]
    sel_feats = st.session_state.get('sel_model_feats', [])
    if any(f.startswith('cir_') for f in sel_feats):
        groups.append(('🇪🇸 PNOA Falso Color IR (CIR)', list(bp.CIR_FEATS), 'cir', False))
    if any(f.startswith('ps_') for f in sel_feats):
        groups.append(('🌍 PlanetScope 3 m', list(bp.PS_FEATS), 'ps', False))

    table = []
    problems = []
    for name, feats, gid, skipped in groups:
        if skipped:
            table.append({'Fuente': name, 'Puntos con dato': '—',
                          '%': '—', 'Estado': '⚪ omitida'})
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


# ── Perfil espectral por punto vs. referencia de borreguiles ─────────────────
# Las gráficas viven en point_profile.py (puro altair/numpy/pandas, testeable sin la app).
@st.cache_data(show_spinner=False)
def _ref_stats_from_csv():
    """Fallback: cuantiles por clase desde classification_v5.csv si está local."""
    return pp.ref_stats_from_csv([APP_DIR / 'classification_v5.csv',
                                  APP_DIR.parent / 'classification_v5.csv'])


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
    """Gráficas comparativas del punto seleccionado vs. el rango típico de borreguiles."""
    if not ref_stats or not ref_stats.get('features'):
        st.info('El modelo seleccionado no trae valores de referencia (`ref_stats`). '
                'Reentrena/guarda con `train_v5.py` para activar la comparación. '
                'Se muestran solo los valores del punto.')
        vals = {lab: sel.get(f) for f, lab in pp.KEY if pp.fnum(sel.get(f)) is not None}
        if vals:
            st.dataframe(pd.DataFrame([vals]), width='stretch', hide_index=True)
        return
    feats = ref_stats['features']
    nsi, nno = ref_stats.get('n_si', '?'), ref_stats.get('n_no', '?')
    st.caption(f'Zona **verde** = rango habitual de los borreguiles de entrenamiento '
               f'(p25–p75, mediana a trazos; n={nsi} borreguil / {nno} no-borreguil). '
               f'Marcador **azul** = este punto (**naranja** si cae fuera del rango habitual).')

    imp_chart, _ = pp.chart_importance(ref_stats, topn=12)
    if imp_chart is not None:
        st.markdown('**🔑 Variables más decisivas del modelo** (guía jerárquica) — de arriba '
                    '(más peso) a abajo; a la derecha, el **rango típico de borreguil** (p25–p75).')
        st.altair_chart(imp_chart, width='stretch')
        st.caption('Lee los gráficos de abajo en este orden de importancia. Altitud, pendiente, '
                   'SAR y textura de imagen pesan en el modelo pero no son índices espectrales, '
                   'por eso no aparecen en los perfiles estacionales.')

    st.markdown('**1 · Perfil estacional** — inicio→fin de verano frente a la banda de borreguiles (un panel por índice)')
    c1 = pp.chart_seasonal(sel, feats)
    if c1 is not None:
        st.altair_chart(c1)
    else:
        st.caption('Sin datos estacionales para este punto.')

    st.markdown('**2 · Posición vs. distribución de borreguiles** — percentil de cada índice (fin de verano)')
    c2, _ = pp.chart_percentile(sel, feats)
    if c2 is not None:
        st.altair_chart(c2, width='stretch')
    else:
        st.caption('Sin índices clave disponibles para este punto.')

    st.markdown('**3 · Variabilidad multianual** — media (●) y rango mín–máx (│) entre escenas vs. banda de borreguiles')
    c3 = pp.chart_multiyear(sel, feats)
    if c3 is not None:
        st.altair_chart(c3)
    else:
        st.caption('Sin índices multianuales disponibles para este punto.')


def _has_script_ctx():
    """True si el hilo actual puede actualizar la UI de Streamlit (tiene contexto de
    ScriptRunner). En Streamlit el SCRIPT corre en un hilo propio (NO el principal),
    así que NO vale comprobar main_thread; hay que comprobar el contexto real. Los
    hilos worker de un ThreadPool NO lo tienen → no deben tocar la UI."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return True  # si no se puede determinar, intentarlo (se captura cualquier fallo)


class _StatusWriter(io.TextIOBase):
    """Redirige los print() del pipeline a una 'consola' en vivo (st.code) dentro del
    st.status, mostrando las últimas líneas. Si no, el paso parece congelado."""
    def __init__(self, container, max_lines=16):
        self.slot = container.empty()
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
        if updated and _has_script_ctx():
            try:
                self.slot.code('\n'.join(self.lines), language=None)
            except Exception:
                pass
        return len(s)

    def flush(self):
        return None


from contextlib import contextmanager

@contextmanager
def live_logs(container):
    """Context manager: dentro del bloque, los print() del pipeline se ven en vivo
    en una consola del st.status. Restaura stdout siempre. Tolera escritura
    concurrente desde hilos worker (esos no tienen contexto → solo bufferizan)."""
    w = _StatusWriter(container)
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = w
    sys.stderr = w
    try:
        yield w
    finally:
        sys.stdout, sys.stderr = old_out, old_err


# ============================================================
# UI
# ============================================================
st.set_page_config(page_title='Borreguil Pipeline',
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

st.title('🌿 Borreguil / Wet Meadow Pipeline')
st.markdown('Identificación automática de borreguiles en zonas de montaña — '
            'ESRI + OSM + Copernicus DEM + Sentinel-2 (MPC) + Random Forest. '
            '[Documentación metodológica](Metodologia_borreguiles.docx)')

# ============================================================
# SIDEBAR – inputs y opciones
# ============================================================
with st.sidebar:
    st.header('Inputs')

    input_mode = st.radio('Origen de los puntos', [
        '📁  Subir archivo (KML/Shp/GeoJSON)',
        '🎲  Generar aleatoriamente en el área',
    ], index=0, label_visibility='collapsed')

    f_input = None; gen_n = 0; gen_mode = 'stratified_elev'
    gen_min_elev = None; gen_max_elev = None; gen_seed = 42; gen_min_dist = 50
    if input_mode.startswith('📁'):
        f_input = st.file_uploader('Puntos candidatos',
                                      type=['kml','geojson','shp','json'])
    else:
        st.caption('Requiere definir el área de estudio (panel siguiente).')
        gen_n = st.slider('Nº puntos a generar', 30, 1000, 200, 10)
        gen_mode = st.selectbox('Estrategia', [
            'stratified_elev',
            'uniform',
            'poisson_disc',
        ], index=0, help=(
            'stratified_elev: reparto equilibrado por bandas de altitud. '
            'uniform: aleatorio puro. '
            'poisson_disc: separación mínima entre puntos.'))
        if gen_mode == 'poisson_disc':
            gen_min_dist = st.number_input(
                'Distancia mínima entre puntos (m)', min_value=10, max_value=500,
                value=50, step=10,
                help='Solo en poisson_disc: ningún par de puntos quedará más cerca '
                     'de esta distancia. Los puntos se ajustan además al centro del '
                     'píxel Sentinel-2 (10 m).')
        col1, col2 = st.columns(2)
        with col1:
            gen_min_elev_str = st.text_input('Elev. mín (m)', value='2000',
                                                help='Opcional: filtra por SRTM 30m')
        with col2:
            gen_max_elev_str = st.text_input('Elev. máx (m)', value='',
                                                help='Opcional')
        gen_min_elev = float(gen_min_elev_str) if gen_min_elev_str.strip() else None
        gen_max_elev = float(gen_max_elev_str) if gen_max_elev_str.strip() else None
        gen_seed = st.number_input('Semilla', 0, 99999, 42)

    f_truth = st.file_uploader('Opcional: verdad-terreno (presencias y/o ausencias)',
                                  type=['kml','geojson','shp','json'])
    truth_train_only = st.checkbox(
        'Usar verdad-terreno solo para entrenar (no mostrarla en el mapa)',
        value=False, disabled=(f_truth is None),
        help='El modelo aprende de tus puntos de campo, pero estos NO se añaden al '
             'mapa, la tabla ni las descargas: solo verás tus puntos candidatos. '
             'Útil cuando subes muchos borreguiles verificados y no quieres que '
             'saturen el resultado. Los candidatos que coincidan (<20 m) con un punto '
             'de campo sí se marcan como verificados y se muestran.')
    with st.expander('ℹ ¿Cómo estructurar la verdad-terreno?'):
        st.markdown(
            '**Solo presencias** (lo más simple): un KML/GeoJSON/Shapefile con puntos '
            'de borreguiles confirmados, **sin ningún atributo**. Todos se toman como '
            'borreguil; las ausencias se generan solas (pseudo-ausencias lejanas).\n\n'
            '**Presencias + ausencias** (recomendado si las tienes): añade un **atributo '
            'de clase** a cada punto. La app reconoce, sin distinguir mayúsculas:\n'
            '- Atributo llamado `Borreguil`, `presencia`, `clase`, `tipo` o `label`.\n'
            '- Valor de **presencia**: `si` · `1` · `presencia` · `borreguil`.\n'
            '- Valor de **ausencia**: `no` · `0` · `ausencia` · `no_borreguil`.\n\n'
            'Ejemplo GeoJSON (un punto presencia y uno ausencia):')
        st.code('''{ "type": "FeatureCollection", "features": [
  { "type":"Feature",
    "properties": { "Borreguil": "si" },
    "geometry": { "type":"Point", "coordinates": [-3.222, 37.114] } },
  { "type":"Feature",
    "properties": { "Borreguil": "no" },
    "geometry": { "type":"Point", "coordinates": [-3.210, 37.108] } }
]}''', language='json')
        st.caption('En KML, añade un `<SimpleData name="Borreguil">no</SimpleData>` por '
                   'Placemark. Un punto sin atributo de clase se toma como presencia.')

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
        '🌍  Google Earth Engine (requiere cuenta) — más rápido',
        '🛰  Microsoft Planetary Computer (sin cuenta)',
    ], index=0, label_visibility='collapsed')
    backend = 'gee' if backend_label.startswith('🌍') else 'mpc'
    st.caption('GEE calcula los índices en el servidor (mucho más rápido, sobre todo '
               'con el modelo de 45 variables). Si GEE no está disponible, la app usa '
               'MPC automáticamente.')

    gee_project = ''
    if backend == 'gee':
        gee_project = st.text_input('Proyecto Google Earth Engine',
                                      placeholder='ee-tunombre',
                                      help=('Nombre del proyecto de Google Cloud con la '
                                            'Earth Engine API habilitada.'))
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
                except Exception as e:
                    st.error(str(e))
        st.caption('El cómputo S2 + topografía se hace en los servidores de Google '
                    '(rápido, sin descargas). No se usa MPC.')

    st.divider()
    st.header('Modelo de partida')

    st.caption('🧭 ¿Qué modelo elegir según tu zona? Mira la pestaña '
               '**📖 Documentación** (arriba) para la guía completa y las '
               'recomendaciones por parque.')

    _models = list_models()
    if _models:
        sel_model_name = st.selectbox(
            'Modelo base Random Forest', _models, index=0,
            help='Modelos .joblib de la carpeta de la app. Se usa para predecir '
                 'cuando no hay suficiente verdad-terreno local. Tras entrenar en '
                 'tu zona puedes guardarlo aquí (panel de resultados) para reutilizarlo.')
        start_model_path = str(model_path(sel_model_name))
        _sel_feats = []
        try:
            _mb = bp.load_model_bundle(start_model_path)
            _sel_feats = _mb.get('features', [])
            st.caption(f"Zona: {_mb.get('training_zone','?')} · "
                       f"n={_mb.get('training_n','?')} · "
                       f"AUC={_mb.get('auc_groupkfold','?')}")
            # Badge contextual: qué fuente extra usará y si está disponible
            _needs_cir = any(f.startswith('cir_') for f in _sel_feats)
            _needs_ps  = any(f.startswith('ps_')  for f in _sel_feats)
            if _needs_cir:
                st.success('🇪🇸 Usa **PNOA Falso Color IR 0,25 m** (gratis). Cobertura WMS '
                           'IR en Andalucía, Cataluña y Canarias; se extrae '
                           'automáticamente (en otras zonas usa las 50 base).')
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
                st.caption('Modelo universal (50 variables): no requiere fuentes extra '
                           'ni claves.')
        except Exception as _e:
            st.caption(f'(no se pudieron leer metadatos: {_e})')
        st.session_state.sel_model_feats = _sel_feats
    else:
        start_model_path = None
        st.session_state.sel_model_feats = []
        st.caption('No hay modelos .joblib en la carpeta.')
    st.session_state.start_model_path = start_model_path

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
        help='Salta las descargas lentas que se hacen punto a punto en tu PC: '
             'imágenes ESRI/PNOA, Sentinel-1 (SAR) y CIR/Planet. Con backend GEE, '
             'el resto (topografía y Sentinel-2) se calcula en la nube en segundos. '
             'Da resultados APROXIMADOS (sin texturas ni SAR): ideal para iterar '
             'rápido y luego afinar desmarcando esta casilla.')
    if fast_mode:
        st.caption('⚡ Vista previa: sin imágenes, sin Sentinel-1 ni CIR/Planet. '
                   'Resultados aproximados.')
    skip_imgs = st.checkbox('Saltar imágenes (más rápido, sin texturas)',
                            value=False, disabled=fast_mode) or fast_mode
    skip_s2   = st.checkbox('Saltar Sentinel-2 (más rápido, sin NDVI)', value=False)
    skip_s1   = fast_mode  # Sentinel-1 se baja de MPC punto a punto (lento aunque uses GEE)
    use_osm   = st.checkbox('Usar OSM (hidrografía/infraestructuras) — Overpass puede fallar',
                             value=False)
    no_osm = not use_osm

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

# ============================================================
# HELP / WELCOME
# ============================================================
# ============================================================
# CUERPO PRINCIPAL — pestañas: Análisis  |  Documentación
# ============================================================
tab_analisis, tab_docs = st.tabs(['🗺️  Análisis', '📖  Documentación y recomendaciones'])

with tab_docs:
    render_docs()

with tab_analisis:
    if 'rows' not in st.session_state and not run_btn:
        st.info('👈 Configura los inputs en el panel lateral y pulsa **Ejecutar pipeline**. '
                'Puedes subir tus puntos o generarlos aleatoriamente dentro del área.')

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('### 📍 Sin verdad-terreno')
            st.markdown(
                '- Se usa el modelo **preentrenado en Sierra Nevada**\n'
                '- AUC GroupKFold: **0,86**\n'
                '- Funciona en zonas comparables; precaución en climas distintos'
            )
        with col2:
            st.markdown('### 🎯 Con verdad-terreno')
            st.markdown(
                '- **Presencias** (borreguiles) y, opcional, **ausencias**\n'
                '- Sin ausencias: se generan pseudo-ausencias solas\n'
                '- Atributo de clase `Borreguil`=si/no (ver panel lateral)\n'
                '- Mínimo: 30 positivos · Recomendado: 80-150'
            )
        with col3:
            st.markdown('### 🛰 Backend')
            st.markdown(
                '- **MPC**: sin cuenta, descarga rásters (más lento)\n'
                '- **GEE**: requiere cuenta, cómputo en la nube (rápido)\n'
                '- Sentinel-2 por defecto **2017-2025**'
            )

        st.divider()
        st.markdown('### Verdad-terreno: solo coordenadas (presence-only)')
        st.markdown('El KML de borreguiles verificados **no necesita ningún atributo**: '
                    'cada punto se asume borreguil. La clase negativa se genera sola.')
        st.code('''<Placemark>
  <Point><coordinates>-3.22206,37.11353</coordinates></Point>
</Placemark>''', language='xml')

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

    if run_btn and can_run:
        # Workspace
        work = Path(tempfile.mkdtemp(prefix='borreguil_'))
        st.session_state.work = work

        input_path = (save_uploaded(f_input, '.kml' if f_input.name.endswith('.kml') else '.json')
                       if f_input else None)
        truth_path = save_uploaded(f_truth, '.kml' if f_truth and f_truth.name.endswith('.kml') else '.json')
        area_path  = save_uploaded_area(f_area) if f_area else None

        # Resolve study area FIRST (may be needed for generation)
        study_geom = None; study_info = None
        if area_path or wdpa_id or pa_name:
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
                        st.write(f"**Área aprox**: {study_info['area_km2']:.1f} km²")
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
        if input_mode.startswith('🎲'):
            if not study_geom:
                st.error('Para generar puntos hace falta un área de estudio válida.')
                st.stop()
            with st.status(f'Generando {gen_n} puntos aleatorios ({gen_mode})…', expanded=True) as status:
                import random_points
                rows, gen_stats = random_points.generate(study_geom, gen_n,
                                                mode=gen_mode,
                                                min_elev=gen_min_elev,
                                                max_elev=gen_max_elev,
                                                min_dist_m=int(gen_min_dist),
                                                seed=int(gen_seed))
                status.update(label=f'{len(rows)} puntos generados', state='complete')

            # El filtro altitudinal dejó 0 puntos → avisar con el rango real del área
            if not rows:
                reason = gen_stats.get('reason')
                amax = gen_stats.get('area_elev_max')
                amin = gen_stats.get('area_elev_min')
                if reason == 'min_elev_above_area' and amax is not None:
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
                st.stop()

            st.write(f'**{len(rows)} puntos generados**')
            if rows and 'elev_m_initial' in rows[0]:
                elevs = [r.get('elev_m_initial') for r in rows if r.get('elev_m_initial')]
                if elevs:
                    st.write(f'Rango altitudinal: {min(elevs):.0f} – {max(elevs):.0f} m '
                              f'(mediana {sorted(elevs)[len(elevs)//2]:.0f} m)')
        else:
            with st.status('Leyendo puntos…', expanded=True) as status:
                rows = bp.read_points(input_path)
                st.write(f'{len(rows)} puntos cargados')
                status.update(label='✓ Lectura completada', state='complete')

        # Merge ground truth: presencias y (opcionalmente) ausencias explícitas
        if truth_path:
            tr = bp.read_points(truth_path)
            n_match, n_add, n_pos, n_neg = bp.merge_truth_points(rows, tr)
            if n_neg > 0:
                st.write(f'**Verdad-terreno (presencia/ausencia)**: {n_pos} borreguiles + '
                          f'{n_neg} no-borreguiles ({n_match} coinciden, {n_add} añadidos). '
                          f'Total: {len(rows)} puntos.')
            else:
                st.write(f'**Verdad-terreno (presence-only)**: {n_pos} borreguiles '
                          f'({n_match} coinciden, {n_add} añadidos). Total: {len(rows)} '
                          'puntos. Las ausencias se generarán como pseudo-ausencias.')

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

        # OSM (omitido por defecto)
        if not no_osm:
            with st.status('Descargando OSM (hidrografía + infraestructuras)…', expanded=False) as status:
                try:
                    bp.fetch_osm(rows, bbox)
                    status.update(label='✓ OSM completado', state='complete')
                except Exception as e:
                    status.update(label=f'⚠ OSM falló (se continúa sin OSM): {e}', state='error')

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

        # Sentinel-1 RTC (siempre vía MPC, independiente del backend S2: el modelo se
        # entrenó con esta fuente terrain-flattened). Falla con elegancia → features NaN.
        # En modo rápido se salta: es una de las descargas punto-a-punto más lentas.
        if not skip_s2 and not skip_s1:
            set_prog(0.70, 'Sentinel-1 RTC (SAR)…')
            with st.status('Sentinel-1 RTC (humedad SAR, MPC)…', expanded=False) as status:
                try:
                    with live_logs(status):
                        bp.fetch_s1(rows, bbox)
                    status.update(label='✓ Sentinel-1 RTC', state='complete')
                except Exception as e:
                    status.update(label=f'⚠ Sentinel-1 falló (se continúa sin SAR): {e}',
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
                                      '(Andalucía/Cataluña/Canarias); se usan las 50 base',
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
        render_extraction_report(rows, skip_s2=skip_s2, skip_imgs=skip_imgs)

        # RF
        with st.status('Random Forest…', expanded=True) as status:
            _smp = st.session_state.get('start_model_path') or str(model_path('rf_sierra_nevada.joblib'))
            proba, info = bp.train_rf(rows, default_model_path=_smp)
            if proba is not None:
                for i, r in enumerate(rows):
                    r['rf_proba'] = float(proba[i])
                if info:
                    if info.get('mode') == 'pretrained':
                        st.write(f'**Modo: modelo preentrenado en {info["training_zone"]}**')
                        st.write(f'Entrenamiento original: n = {info["training_n"]} · '
                                  f'AUC GroupKFold = {info["auc_groupkfold"]}')
                        st.info('Para mejor precisión en tu zona, sube borreguiles verificados '
                                'en campo (presence-only) y vuelve a ejecutar.')
                    elif info.get('mode') == 'trained':
                        st.write('**Modo: entrenado con tu verdad-terreno (presence-only)**')
                        st.write(f'{info["n_pos"]} positivos · {info["n_neg"]} pseudo-ausencias')
                        auc = info.get('auc', float('nan'))
                        if auc == auc:  # not NaN
                            st.write(f'AUC honesto ({info.get("cv","")}): **{auc:.3f}**')
                        if info["n_pos"] < 30:
                            st.warning('⚠ Pocos positivos (< 30). Recomendado: 30–150 para '
                                        'un modelo fiable en tu zona.')
                status.update(label='✓ RF completado', state='complete')
            else:
                for r in rows: r['rf_proba'] = float('nan')
                status.update(label='⚠ RF no ejecutado', state='error')

        set_prog(1.0, '✓ Proceso completado')

        # Umbral RF: manual, automático por Otsu, o automático por verdad-terreno.
        _probas = [r.get('rf_proba') for r in rows]
        if thr_mode.startswith('Automático · Otsu'):
            threshold = bp.auto_threshold(_probas)
            st.info(f'⚙ Umbral automático (Otsu): **{threshold:.2f}** '
                    '(corte que mejor separa los dos grupos de probabilidad).')
        elif thr_mode.startswith('Automático · verdad'):
            _t = bp.auto_threshold_truth(rows)
            if _t is not None:
                threshold = _t
                st.info(f'⚙ Umbral por verdad-terreno: **{threshold:.2f}** '
                        '(óptimo según tus puntos de campo).')
            else:
                threshold = bp.auto_threshold(_probas)
                st.warning('No hay suficiente verdad-terreno etiquetada para fijar el '
                           f'umbral; se usa Otsu: **{threshold:.2f}**.')
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

        # --- Estado para la evaluación recursiva (auto-entrenamiento) ---
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
        st.success(f'Pipeline terminado. Workspace: {work}')

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
        MODEL_PATH = st.session_state.get('start_model_path') or str(model_path('rf_sierra_nevada.joblib'))

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
            cols[i+1].metric(i18n.tv(k), v)

        # ----------------------------------------------------------------
        # EVALUACIÓN RECURSIVA (auto-entrenamiento / self-training)
        # ----------------------------------------------------------------
        iter_n = st.session_state.get('iter_n', 1)
        hist = st.session_state.get('iter_history', [])
        base_truth_ids = st.session_state.get('base_truth_ids', set())

        with st.expander('🔁 Evaluación recursiva (auto-entrenamiento)', expanded=True):
            st.markdown(
                'En cada iteración, los puntos de **posible borreguil** (probabilidad ≥ '
                'umbral de promoción) se convierten en verdad-terreno y se **recalibra** '
                'el Random Forest para volver a predecir sobre el resto. Útil para '
                'descubrir borreguiles no inventariados de forma progresiva.')

            # Historial de iteraciones
            if hist:
                hrows = []
                for h in hist:
                    info_h = h.get('info') or {}
                    auc = info_h.get('auc')
                    auc_txt = f"{auc:.3f}" if isinstance(auc, float) and auc == auc else '—'
                    cc = h.get('counts', {})
                    n_borr = sum(v for k, v in cc.items()
                                  if 'BORREGUIL' in k.upper() and 'NO ' not in k.upper())
                    hrows.append({
                        'Iteración': h['iter'],
                        'Modo': h['mode'],
                        'Umbral promoción': ('—' if h['cutoff'] is None else f"{h['cutoff']:.2f}"),
                        'Positivos entren.': h.get('n_train_pos', '—'),
                        'Nuevos hallados': ('—' if h.get('n_new') is None else h['n_new']),
                        'AUC (CV)': auc_txt,
                        'Σ Borreguil (mapa)': n_borr,
                    })
                st.dataframe(pd.DataFrame(hrows), hide_index=True, width='stretch')

                last = hist[-1]
                if last.get('n_new') == 0:
                    st.success('✓ **Convergencia**: la última iteración no encontró nuevos '
                               'borreguiles. Puedes detener el proceso.')

            # Controles para la siguiente iteración
            n_pos_now = sum(1 for r in rows if str(r.get('Borreguil','')).lower() == 'si')
            st.markdown(f'**Iteración actual: {iter_n}**  ·  positivos en el modelo actual: '
                        f'{n_pos_now}  ·  verdad-terreno verificada (fija): {len(base_truth_ids)}')

            c1, c2 = st.columns(2)
            with c1:
                promote_cutoff = st.slider('Umbral de promoción a borreguil (probabilidad RF)',
                                            0.50, 0.95, 0.70, 0.05,
                                            help='Sólo se promueven puntos muy probables, '
                                                 'para limitar la propagación de errores.')
            with c2:
                mode_label = st.radio('¿Cómo usar los posibles borreguiles?', [
                    'Añadir (acumular a la verdad-terreno)',
                    'Reemplazar (refrescar desde el modelo actual)',
                ], help='Añadir: el conjunto de positivos crece en cada iteración.\n'
                        'Reemplazar: se recalculan los pseudo-positivos cada vez desde el '
                        'último modelo, manteniendo siempre tus puntos verificados.')
            mode = 'add' if mode_label.startswith('Añadir') else 'replace'

            # ¿Cuántos se promoverían?
            prev_pos = {r.get('ID') for r in rows if str(r.get('Borreguil','')).lower() == 'si'}
            # Mismo criterio que la iteración: no cuentan las ausencias de campo ni
            # los puntos dudosos, que no se promueven.
            promotable = [r for r in rows if bp.promovible(r, promote_cutoff)]
            n_new_would = len([r for r in promotable if r.get('ID') not in prev_pos])
            st.caption(f'Con umbral {promote_cutoff:.2f}: {len(promotable)} puntos ≥ umbral '
                       f'({n_new_would} nuevos respecto a los positivos actuales).')

            if st.button(f'▶ Ejecutar iteración {iter_n + 1}', type='primary'):
                with st.spinner('Reentrenando Random Forest…'):
                    rows2, info2, stats = run_self_training_iteration(
                        rows, promote_cutoff, mode, base_truth_ids,
                        threshold=threshold, model_path=MODEL_PATH)
                    # Re-guardar salidas
                    bp.save_csv(rows2, work / 'classification.csv')
                    bp.save_points_geojson(rows2, work / 'puntos.geojson')
                    bp.save_xlsx(rows2, work / 'Clasificacion_puntos.xlsx', threshold=threshold)
                    bp.save_map(rows2, work / 'mapa.html', study_geom=sg)
                    # Actualizar estado + historial
                    st.session_state.rows = rows2
                    st.session_state.iter_n = iter_n + 1
                    st.session_state.model_bundle = (info2 or {}).get('model_bundle')
                    st.session_state.iter_history = hist + [{
                        'iter': iter_n + 1,
                        'mode': mode,
                        'cutoff': promote_cutoff,
                        'info': info2,
                        'counts': dict(Counter(r['decision'] for r in rows2)),
                        'n_train_pos': stats['n_train_pos'],
                        'n_new': stats['n_new'],
                    }]
                    st.session_state.sel_point = None
                st.success(f'Iteración {iter_n + 1} completada: {stats["n_new"]} nuevos '
                           f'posibles borreguiles ({stats["n_train_pos"]} positivos en total).')
                st.rerun()

            # --- Guardar el modelo entrenado ---
            bundle = st.session_state.get('model_bundle')
            last_mode = (st.session_state.get('info') or {}).get('mode')
            hist_last_info = (hist[-1].get('info') if hist else None) or {}
            is_trained = (last_mode == 'trained' or hist_last_info.get('mode') == 'trained')
            st.divider()
            st.markdown('### 💾 Guardar el modelo entrenado')
            # Mensaje de guardado PERSISTENTE: sobrevive al st.rerun (si no, el éxito se
            # perdía al reejecutarse la app y parecía que no se había guardado).
            _saved_msg = st.session_state.pop('_model_saved_msg', None)
            if _saved_msg:
                st.success(_saved_msg)
            if not bundle:
                st.caption('Aún no hay un modelo en memoria. Ejecuta el pipeline; para un '
                           'modelo propio de tu zona, sube borreguiles verificados o marca '
                           'filas en la tabla como borreguil y reentrena.')
            else:
                if not is_trained:
                    st.info('El modelo en memoria es **preentrenado** (no se ha reentrenado '
                            'con tu verdad-terreno). Puedes guardarlo igualmente con otro '
                            'nombre, aunque para tu zona conviene marcar borreguiles '
                            'verificados y reentrenar antes de guardar.')
                # Pre-rellenar con el nombre del área de estudio
                _si = st.session_state.get('study_info') or {}
                area_default = (_si.get('name') or _si.get('query') or '').strip()
                # Quitar sufijos largos tipo ", Comarca de…, Andalucía, España"
                if ',' in area_default:
                    area_default = area_default.split(',')[0].strip()

                area_name = st.text_input(
                    'Nombre del área protegida (será el nombre del fichero)',
                    value=area_default or 'Mi zona de montaña',
                    help='Usa el nombre oficial del espacio protegido. '
                         'Se guarda como rf_<nombre>.joblib en la carpeta de la app.')

                fname = area_to_filename(area_name)
                out_path = DATA_DIR / f'{fname}.joblib'
                already_exists = out_path.exists()

                st.caption(f'Se guardará como: **{fname}.joblib**'
                            + (' (sobreescribirá el existente)' if already_exists else ''))

                # Preparar bundle con el nombre de zona correcto
                import io as _io, joblib as _jl
                b_save = dict(bundle); b_save['training_zone'] = area_name
                buf = _io.BytesIO()
                try:
                    _jl.dump(b_save, buf); buf.seek(0)
                    buf_bytes = buf.getvalue()
                except Exception as _je:
                    buf_bytes = None
                    st.error(f'Error serializando modelo: {_je}')

                cS1, cS2 = st.columns(2)
                with cS1:
                    # Guardar en carpeta local → aparece en selector y se puede git-push
                    if st.button('💾 Guardar en la carpeta (local + git)', type='primary'):
                        try:
                            bp.save_model_bundle(b_save, out_path)
                            # Confirmación PERSISTENTE (se muestra tras el rerun) + verificación
                            if out_path.exists() and out_path.stat().st_size > 0:
                                st.session_state['_model_saved_msg'] = (
                                    f'✓ Modelo guardado: **{out_path.name}** '
                                    f'({out_path.stat().st_size//1024} KB) en `{DATA_DIR}`. '
                                    'Ya aparece en el desplegable «Modelo de partida».')
                                st.rerun()
                            else:
                                st.error('El fichero no se creó. Usa **⬇ Descargar .joblib**.')
                        except Exception as e:
                            st.error(
                                f'No se pudo guardar en la carpeta: `{e}`.\n\n'
                                '• Si usas la app **online** (Streamlit Cloud), la carpeta no '
                                'es escribible → usa **⬇ Descargar .joblib** y súbelo al repo.\n'
                                '• En local, revisa permisos de escritura en la carpeta de la app.')

                with cS2:
                    # Descarga directa: SIEMPRE funciona (local y deploy online efímero)
                    if buf_bytes:
                        st.download_button(
                            '⬇ Descargar .joblib',
                            data=buf_bytes,
                            file_name=f'{fname}.joblib',
                            mime='application/octet-stream',
                            key='dl_model',
                            help='Alternativa robusta al guardado en carpeta: descarga el '
                                 'fichero y cópialo a borreguil_app/ (o súbelo al repo).')

                # Instrucciones para publicar el modelo en el repositorio.
                # NOTA: no se usa st.expander aquí porque este bloque ya está dentro
                # del expander "🔁 Evaluación recursiva" y Streamlit no permite anidar
                # expanders. Se usa una casilla para mostrar/ocultar (sin restricción).
                if st.checkbox('📤 Cómo publicar el modelo para todos los usuarios online',
                               key='show_publish_help'):
                    st.markdown(f"""
**Opción A — Ejecución local (más sencilla):**
1. Pulsa "💾 Guardar en la carpeta" → se crea `borreguil_app/{fname}.joblib`.
2. En la terminal:
```bash
cd ruta/a/tu/repositorio
git add borreguil_app/{fname}.joblib
git commit -m "Añadir modelo RF: {area_name}"
git push
```
3. Streamlit Cloud redesplegará la app automáticamente. El modelo aparecerá
   en el desplegable "Modelo de partida" para todos los usuarios.

**Opción B — App online (carpeta efímera):**
1. Pulsa "⬇ Descargar .joblib".
2. Copia el fichero descargado a `borreguil_app/` en tu copia local del
   repositorio.
3. Sigue los pasos 2-3 de la opción A.

> Los modelos son ficheros de ~3-4 MB. Para muchos modelos considera
> usar [Git LFS](https://git-lfs.github.com/).
""")


        # ----------------------------------------------------------------
        # Helpers de mapa
        # ----------------------------------------------------------------
        def _add_points(fmap, pts, highlight_id=None, radius=6, hueco=False):
            # hueco=True: el punto resaltado se dibuja sin relleno, para que el
            # marcador no tape justo el terreno que hay que mirar.
            _rotulos = i18n.pick(
                {'tipo': 'Tipo', 'revisado': 'revisado', 'patron': 'Patrón',
                 'altitud': 'Altitud'},
                {'tipo': 'Type', 'revisado': 'reviewed', 'patron': 'Pattern',
                 'altitud': 'Elevation'})
            for r in pts:
                color = bp.decision_color(r.get('decision', 'SIN PREDICCIÓN'))
                # Números con 4 decimales y nombres largos partidos en líneas.
                popup = bp.popup_html(r, i18n.tv, _rotulos)
                is_sel = (highlight_id is not None and r.get('ID') == highlight_id)
                if is_sel:
                    folium.CircleMarker([r['lat'], r['lon']], radius=radius+8,
                                        color='#FF1744', weight=4, fill=False).add_to(fmap)
                _relleno = not (hueco and is_sel)
                folium.CircleMarker([r['lat'], r['lon']], radius=radius,
                                    color=color, fill=True, fill_color=color,
                                    fill_opacity=0.85 if _relleno else 0.0,
                                    popup=folium.Popup(popup, max_width=300)).add_to(fmap)

        # Leyenda de colores de los puntos (decisión del Random Forest).
        _LEGEND_ITEMS = [
            ('BORREGUIL VERIFICADO', 'Verificado (campo)'),
            ('BORREGUIL PROBABLE', 'Probable'),
            ('POSIBLE BORREGUIL', 'Posible'),
            ('BORREGUIL (auto)', 'Auto (iteración)'),
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
            fm = folium.Map(location=center, zoom_start=zoom, max_zoom=22,
                            control_scale=True, **({'tiles': None} if ortofoto else {}))
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
                # Para revisar un punto hay que ver el terreno: la ortofoto pasa a ser
                # la capa visible (la última capa base que se añade) y las otras
                # quedan disponibles en el control de capas.
                capas = capas[1:] + capas[:1]
                for _c in capas[:-1]:
                    _c['show'] = False
            for _c in capas:
                folium.TileLayer(**_c).add_to(fm)
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
        # de un rerun en el mapa principal de la pestaña Mapa). Streamlit devuelve la
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

        # Tabs
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
            ['Mapa', 'Tabla', 'Histograma RF', 'Descargas', '🔑 Variables',
             '📊 Distribuciones', '🌿 Mapeo RF'])

        with tab1:
            st.markdown('Mapa interactivo. Haz click en un marcador para ver sus detalles y '
                        'poder **quitarlo** del análisis. Al seleccionar filas en la pestaña '
                        '**Tabla**, el mapa se centra en el **último punto seleccionado**. '
                        'La **leyenda de colores** está abajo a la izquierda.')
            lat_med = sum(r['lat'] for r in rows)/len(rows)
            lon_med = sum(r['lon'] for r in rows)/len(rows)
            st.caption('Capas activables (control ▤ arriba a la derecha): **Píxel S2 '
                       '(10 m)**, **ventana de muestreo 10 m** (S2 + Sentinel-1) y **20 m** '
                       '(S2). Cambia a **Google Satélite** para más zoom. Sentinel-1 RTC '
                       'comparte la rejilla de 10 m con Sentinel-2.')
            m = _base_map([lat_med, lon_med], 12)
            _add_points(m, rows, highlight_id=(sel.get('ID') if sel else None))
            _add_pixel_layers(m, rows, show=False)
            _add_legend(m, rows)
            folium.LayerControl(collapsed=False).add_to(m)
            if sel:
                map_state = st_folium(m, width=None, height=600,
                                      returned_objects=['last_object_clicked'],
                                      center=[sel['lat'], sel['lon']], zoom=18, key='mainmap')
            else:
                map_state = st_folium(m, width=None, height=600,
                                      returned_objects=['last_object_clicked'], key='mainmap')

            # Quitar un punto pinchando directamente en su marcador
            clicked = (map_state or {}).get('last_object_clicked') if isinstance(map_state, dict) else None
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
                        st.markdown(f"📍 Marcador seleccionado: **{rr.get('ID','?')}** — "
                                    f"{rr.get('decision','—')} (RF {_rftxt})")
                    with cb:
                        if st.button('🗑 Quitar este punto del análisis', key='remove_map_pt'):
                            keep = [r for j, r in enumerate(rows) if j != ni]
                            if not keep:
                                st.warning('No puedes quitar todos los puntos.')
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
                                st.success(f"Punto {rr.get('ID','?')} quitado del análisis.")
                                st.rerun()

        with tab2:
            st.caption('Selecciona una o varias filas (casilla izquierda). El **último punto '
                       'seleccionado** centra el mapa y muestra su imagen, sin perder los '
                       'anteriores. Con varias seleccionadas puedes **marcarlas como '
                       'borreguil verificado** y reentrenar.')
            _aviso_rev = st.session_state.pop('_rev_msg', None)
            if _aviso_rev:
                st.success(_aviso_rev)
            _rs = bp.resumen_revision(rows, pp.is_borreguil_decision)
            if _rs['n_borreguil']:
                _pn = _rs['por_nivel']
                st.markdown(i18n.pick(
                    f"🌿 **Tipo revisado: {_rs['n_revisados']} de {_rs['n_borreguil']}** "
                    f"borreguiles · corregidos respecto a la propuesta de la app: "
                    f"**{_rs['n_corregidos']}** (ambiente {_pn['ambiente']} · humedad "
                    f"{_pn['humedad']} · pureza {_pn['pureza']})",
                    f"🌿 **Type reviewed: {_rs['n_revisados']} of {_rs['n_borreguil']}** "
                    f"borreguiles · corrected against the app's proposal: "
                    f"**{_rs['n_corregidos']}** (environment {_pn['ambiente']} · moisture "
                    f"{_pn['humedad']} · purity {_pn['pureza']})"))
                if _rs['n_revisados'] >= 10 and _rs['n_corregidos'] == 0:
                    st.caption(i18n.pick(
                        'No se ha corregido ninguno: o la propuesta acierta siempre o se '
                        'está aceptando sin mirar la ortofoto. Un tipo aceptado sin mirar '
                        'sigue siendo la regla de la app.',
                        "None has been corrected: either the proposal is always right or "
                        "it is being accepted without looking at the orthophoto. A type "
                        "accepted without looking is still the app's rule."))
            cols_show = ['ID','cuenca_id','source','origin','Borreguil','decision','rf_proba',
                          'ambiente','humedad','pureza','tipo_revisado',
                          'mat_signature','elev_dem_m','slope_deg','twi','dist_water_m',
                          'ndvi_late','clre_late','ndmi_late','evi_late','lat','lon']
            df = pd.DataFrame(rows)
            cols_show = [c for c in cols_show if c in df.columns]
            # Las columnas opcionales que están vacías en todos los puntos (p. ej.
            # la cuenca, si el fichero no la trae) no se muestran: solo ocupaban
            # ancho y empujaban fuera de la pantalla la decisión y el tipo.
            _opcionales = ('cuenca_id', 'source', 'origin', 'Borreguil', 'mat_signature',
                           'dist_water_m')
            cols_show = [c for c in cols_show if c not in _opcionales
                         or any(bp._txt(v) != '' for v in df[c])]
            # Los valores del pipeline son españoles; se traducen SOLO para
            # mostrarlos (df_show), sin tocar `rows`, del que depende la lógica.
            df_show = df[cols_show].copy()
            for _c in ('decision', 'ambiente', 'humedad', 'pureza', 'tipo_borreguil'):
                if _c in df_show.columns:
                    df_show[_c] = df_show[_c].map(i18n.tv)
            if 'tipo_revisado' in df_show.columns:
                df_show['tipo_revisado'] = df_show['tipo_revisado'].map(
                    lambda v: i18n.pick('sí', 'yes')
                    if bp.tipo_revisado({'tipo_revisado': v}) else '')
            # Los ID muy largos se abrevian SOLO en la tabla, conservando el final, que
            # es lo que los distingue; si no, la columna se come el ancho y hay que
            # desplazarse para ver la decisión y el tipo. Además queda fija al
            # desplazar la tabla hacia la derecha.
            _cfg_tabla = {}
            if 'ID' in df_show.columns:
                df_show['ID'] = bp.abreviar_ids(df_show['ID'].tolist())
                _cfg_tabla['ID'] = st.column_config.Column(pinned=True, help=i18n.pick(
                    'Los nombres largos se muestran abreviados («…»). El nombre '
                    'completo aparece al seleccionar el punto y en las descargas.',
                    'Long names are shown abbreviated ("…"). The full name appears '
                    'when you select the point and in the downloads.'))
            df_show = df_show.rename(columns={c: i18n.tcol(c) for c in df_show.columns})
            event = st.dataframe(df_show, hide_index=True, width='stretch',
                                 on_select='rerun', selection_mode='multi-row',
                                 column_config=_cfg_tabla, key='results_table')
            sel_rows = []
            if event is not None and getattr(event, 'selection', None):
                _selo = event.selection
                sel_rows = (_selo.get('rows') if isinstance(_selo, dict)
                            else getattr(_selo, 'rows', [])) or []

            # --- Aprendizaje activo: marcar seleccionados como verdad-terreno ---
            if sel_rows:
                sel_ids = [rows[i].get('ID') for i in sel_rows if i < len(rows)]
                already = sum(1 for i in sel_rows
                              if i < len(rows) and str(rows[i].get('Borreguil','')).lower() == 'si')
                cA, cB, cC = st.columns([3, 2, 2])
                with cA:
                    st.markdown(f'**{len(sel_ids)} punto(s) seleccionado(s)** '
                                f'({already} ya son verdad-terreno).')
                with cB:
                    if st.button('✓ Marcar como borreguil verificado y reentrenar',
                                 type='primary', key='mark_verified'):
                        with st.spinner('Incorporando verdad-terreno y reentrenando…'):
                            rows2 = [dict(r) for r in rows]
                            sset = set(sel_ids)
                            for r in rows2:
                                if r.get('ID') in sset:
                                    r['Borreguil'] = 'si'; r['origin'] = 'verificado'
                                    # Verificarlo a mano resuelve la duda de campo: si
                                    # siguiera «dudoso», no entraría en el entrenamiento.
                                    r['Duda'] = ''
                            new_base = set(base_truth_ids) | sset
                            info2 = bp.run_rf_and_decide(rows2, threshold=threshold,
                                                         default_model_path=MODEL_PATH)
                            # Los verificados se muestran como tales (los que siguen
                            # dudosos, no: tampoco entrenan).
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
                            n_train_pos = sum(1 for r in rows2
                                              if bp._txt(r.get('Borreguil')).lower() == 'si'
                                              and bp._txt(r.get('Duda')).lower() != 'si')
                            st.session_state.iter_history = hist + [{
                                'iter': iter_n + 1,
                                'mode': f'verdad-terreno (+{len(sset - set(base_truth_ids))})',
                                'cutoff': None,
                                'info': info2,
                                'counts': dict(Counter(r['decision'] for r in rows2)),
                                'n_train_pos': n_train_pos,
                                'n_new': len(sset - set(base_truth_ids)),
                            }]
                            st.session_state.sel_point = None
                        st.success(f'{len(sset - set(base_truth_ids))} nuevos verificados '
                                   'incorporados. Modelo reentrenado.')
                        st.rerun()
                with cC:
                    if st.button('🗑 Quitar del análisis', key='remove_pts',
                                 help='Elimina los puntos seleccionados del mapa, la tabla '
                                      'y las descargas (no reentrena el modelo).'):
                        drop = set(sel_rows)
                        keep = [r for i, r in enumerate(rows) if i not in drop]
                        if not keep:
                            st.warning('No puedes quitar todos los puntos.')
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
                            st.success(f'{len(drop)} punto(s) quitado(s) del análisis.')
                            st.rerun()

            # --- Revisión del TIPO de borreguil (ambiente / humedad / pureza) ---
            # La app propone el tipo por reglas; aquí una persona lo confirma o lo
            # corrige. Solo los puntos revisados servirán para entrenar el modelo de
            # tipos: una etiqueta sin revisar es la propia regla.
            if sel_rows:
                _sel_ok = [i for i in sel_rows if i < len(rows)]
                _idx_tipo = [i for i in _sel_ok
                             if pp.is_borreguil_decision(rows[i].get('decision', ''))
                             or bp.tipo_revisado(rows[i])]
                with st.expander('🌿 Revisar el tipo de borreguil de los puntos seleccionados',
                                 expanded=True):
                    if not _idx_tipo:
                        st.info('Los puntos seleccionados no están clasificados como '
                                'borreguil, así que no tienen tipo. Si alguno lo es, '
                                'márcalo primero como borreguil verificado.')
                    else:
                        st.caption('Decide con la ortofoto o con datos de campo, no por lo '
                                   'que propone la app. **Ambiente**: dónde está el prado '
                                   '(junto a un arroyo, junto a una laguna o en ladera). '
                                   '**Humedad**: aplica siempre el mismo criterio a todos '
                                   'los puntos. **Pureza**: «mixto-agua» si hay agua dentro '
                                   'del píxel de 10 m, «mixto-roca» si hay roca o suelo '
                                   'desnudo, «puro» si todo es prado. Deja «(sin cambio)» '
                                   'para confirmar la propuesta.')
                        import hashlib as _hl
                        # La clave de los desplegables depende de QUÉ puntos hay
                        # seleccionados (por ID, no por posición: al quitar un punto
                        # las posiciones se corren) y de cuántas veces se ha guardado
                        # o deshecho. Así arrancan siempre con el tipo vigente de la
                        # selección, nunca con lo elegido para otro punto.
                        _firma = _hl.md5('|'.join(
                            [str(rows[i].get('ID')) for i in _idx_tipo]
                            + [str(st.session_state.get('_rev_n', 0))]
                        ).encode()).hexdigest()[:10]
                        _SIN = '(sin cambio)'
                        _uno = rows[_idx_tipo[0]] if len(_idx_tipo) == 1 else None
                        _eleccion = {}
                        for _col, (_niv, _etq) in zip(st.columns(3), (
                                ('ambiente', 'Ambiente'), ('humedad', 'Humedad'),
                                ('pureza', 'Pureza'))):
                            _ops = [_SIN] + list(bp.NIVELES_TIPO[_niv])
                            _act = _uno.get(_niv) if _uno else None
                            with _col:
                                _eleccion[_niv] = st.selectbox(
                                    _etq, _ops,
                                    index=_ops.index(_act) if _act in _ops else 0,
                                    key=f'rev_{_niv}_{_firma}',
                                    format_func=lambda o: (i18n.tr(o) if o == '(sin cambio)'
                                                           else i18n.tv(o)))
                        _b1, _b2, _b3 = st.columns([2, 2, 3])
                        with _b1:
                            _guardar = st.button('✓ Guardar tipo revisado', type='primary',
                                                 key='rev_save')
                        with _b2:
                            _deshacer = st.button(
                                '↩ Deshacer revisión', key='rev_undo',
                                help='Devuelve los puntos seleccionados a la propuesta '
                                     'de la app.')
                        with _b3:
                            _n_fuera = len(_sel_ok) - len(_idx_tipo)
                            st.caption(i18n.pick(
                                f'{len(_idx_tipo)} punto(s) a revisar'
                                + (f' · {_n_fuera} no son borreguil y se ignoran'
                                   if _n_fuera else ''),
                                f'{len(_idx_tipo)} point(s) to review'
                                + (f' · {_n_fuera} are not borreguil and are skipped'
                                   if _n_fuera else '')))
                        if _guardar or _deshacer:
                            rows2 = [dict(r) for r in rows]
                            _errores = []
                            for i in _idx_tipo:
                                try:
                                    if _guardar:
                                        bp.set_tipo_revisado(rows2[i], **{
                                            n: (None if v == _SIN else v)
                                            for n, v in _eleccion.items()})
                                    else:
                                        bp.clear_tipo_revisado(rows2[i])
                                except ValueError as _e:
                                    _errores.append(f"{rows2[i].get('ID', '?')}: {_e}")
                            if _errores:
                                # Todo o nada: con un solo punto inválido no se guarda
                                # ninguno, para no dejar la selección a medias.
                                st.error(i18n.pick('No se ha guardado nada. ',
                                                   'Nothing was saved. ')
                                         + ' · '.join(_errores[:5]))
                            else:
                                bp.apply_typology(rows2, pp.is_borreguil_decision)
                                bp.save_csv(rows2, work / 'classification.csv')
                                bp.save_points_geojson(rows2, work / 'puntos.geojson')
                                bp.save_xlsx(rows2, work / 'Clasificacion_puntos.xlsx',
                                             threshold=threshold)
                                bp.save_map(rows2, work / 'mapa.html', study_geom=sg)
                                st.session_state.rows = rows2
                                st.session_state['_rev_n'] = \
                                    st.session_state.get('_rev_n', 0) + 1
                                st.session_state['_rev_msg'] = (
                                    i18n.pick(f'Tipo revisado guardado en {len(_idx_tipo)} '
                                              'punto(s).',
                                              f'Reviewed type saved for {len(_idx_tipo)} '
                                              'point(s).')
                                    if _guardar else
                                    i18n.pick(f'Revisión deshecha en {len(_idx_tipo)} '
                                              'punto(s).',
                                              f'Review undone for {len(_idx_tipo)} '
                                              'point(s).'))
                                st.rerun()

            focus_idx = st.session_state.get('sel_idx')
            if focus_idx is not None and focus_idx < len(rows):
                idx = focus_idx
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
                    folium.LayerControl(collapsed=True).add_to(fm)
                    st_folium(fm, width=None, height=420, returned_objects=[],
                              center=[sel['lat'], sel['lon']], zoom=17, key='focusmap')
                    st.caption(i18n.pick(
                        'Ortofoto de fondo; el recuadro amarillo es el píxel de 10 m del '
                        'punto. Con el control de capas (arriba a la derecha) puedes '
                        'cambiar a Google Satélite, que admite más zoom.',
                        'Orthophoto background; the yellow square is the 10 m pixel of '
                        'the point. Use the layer control (top right) to switch to '
                        'Google Satellite, which allows more zoom.'))
                with colR:
                    # El nombre, en un tamaño contenido y partiéndose por los
                    # separadores: los ID largos quedaban troceados por cualquier
                    # letra en tres líneas de titular.
                    st.markdown(
                        "<h4 style='margin:0 0 .4rem 0;padding:0;font-size:1.15rem;"
                        "line-height:1.3;overflow-wrap:anywhere;'>"
                        f"{bp.nombre_html(sel.get('ID', '?'))}</h4>",
                        unsafe_allow_html=True)
                    st.markdown(f"**{i18n.tv(sel.get('decision','—'))}**")
                    if sel.get('tipo_borreguil'):
                        st.markdown(
                            f"🌿 {i18n.tv(sel['tipo_borreguil'])} — "
                            + (i18n.pick('**revisado**', '**reviewed**')
                               if bp.tipo_revisado(sel)
                               else i18n.pick('propuesta de la app, sin revisar',
                                              "app's proposal, not reviewed")))
                    rfp = sel.get('rf_proba')
                    if isinstance(rfp, (int, float)) and not _m.isnan(rfp):
                        st.metric('Probabilidad RF', f'{rfp*100:.0f}%')
                    # La imagen es la que se anotó en el punto al descargarla; por
                    # posición solo si el punto no lleva la anotación.
                    _img = sel.get('_img', f'pt_{idx:04d}.jpg')
                    img_p = (work / 'imgs' / _img) if _img else None
                    if img_p is not None and img_p.exists():
                        st.image(str(img_p), caption='ESRI World Imagery (~550 m)',
                                 width='stretch')

                    _n = bp.fmt_num                 # 4 decimales
                    st.markdown(
                        f"Altitud: **{_n(sel.get('elev_dem_m'))} m** · Slope: "
                        f"**{_n(sel.get('slope_deg'))}°** · TWI: "
                        f"**{_n(sel.get('twi'))}**\n\n"
                        f"NDVI fin: **{_n(sel.get('ndvi_late'))}** · Clre: "
                        f"**{_n(sel.get('clre_late'))}** · EVI: "
                        f"**{_n(sel.get('evi_late'))}**\n\n"
                        f"Patrón: **{sel.get('mat_signature','—')}** · "
                        f"Coords: {sel.get('lat'):.5f}, {sel.get('lon'):.5f}")
                    gmaps = f"https://www.google.com/maps/search/?api=1&query={sel.get('lat')},{sel.get('lon')}"
                    st.markdown(f'[Abrir en Google Maps]({gmaps})')
                rs = load_ref_stats(st.session_state.get('start_model_path')
                                    or str(model_path('rf_sierra_nevada.joblib')))
                with st.expander('📈 Perfil espectral del punto vs. borreguiles de referencia',
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
                st.markdown('Distribución de probabilidad Random Forest '
                            '(nº de puntos por intervalo de probabilidad):')
                dfh = pd.DataFrame({'rf_proba': rf_vals})
                # Histograma real: agrupa las probabilidades en bins de 0,05 y cuenta
                # cuántos puntos caen en cada uno (antes pintaba una barra por punto).
                hist = alt.Chart(dfh).mark_bar(color='#2e7d32', opacity=0.85).encode(
                    x=alt.X('rf_proba:Q', bin=alt.Bin(extent=[0, 1], step=0.05),
                            title='Probabilidad RF'),
                    y=alt.Y('count():Q', title='Nº de puntos'),
                    tooltip=[alt.Tooltip('count():Q', title='puntos')])
                rules = alt.Chart(
                    pd.DataFrame({'umbral': [threshold, 0.7],
                                  'etiqueta': [f'umbral {threshold:.2f}', '0.70']})
                ).mark_rule(color='#c62828', strokeDash=[4, 4]).encode(
                    x='umbral:Q',
                    tooltip=[alt.Tooltip('etiqueta:N', title='línea')])
                st.altair_chart((hist + rules).properties(height=320),
                                width='stretch')
                _med = sorted(rf_vals)[len(rf_vals)//2]
                st.markdown(f'**Mediana**: {_med:.3f}  ·  '
                             f'**N** ≥ 0.5: {sum(1 for v in rf_vals if v>=0.5)}  ·  '
                             f'**N** ≥ 0.7: {sum(1 for v in rf_vals if v>=0.7)}  ·  '
                             f'**Total**: {len(rf_vals)}')
            else:
                st.info('Sin probabilidades RF disponibles.')

        with tab4:
            st.markdown('Resultados generados (reflejan la última iteración):')
            for name in ('classification.csv','Clasificacion_puntos.xlsx',
                          'puntos.geojson','mapa.html','area_estudio.geojson'):
                p = work / name
                if p.exists():
                    with open(p, 'rb') as fh:
                        st.download_button(label=f'⬇ {name}',
                                           data=fh.read(),
                                           file_name=name,
                                           mime='application/octet-stream',
                                           key=f'dl_{name}')

        with tab5:
            st.markdown('**Variables más decisivas del modelo Random Forest** usado en esta '
                        'estimación — de arriba (más peso) a abajo. A la derecha, el **rango '
                        'típico de borreguil** (p25–p75) cuando el modelo trae valores de '
                        'referencia.')
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
                st.info('El modelo actual no expone importancias de variables '
                        '(reentrena o usa un modelo guardado con `train_v5.py`).')

        with tab6:
            n_borr = sum(1 for r in rows if pp.is_borreguil_decision(r.get('decision', '')))

            # Clasificación jerárquica del tipo de borreguil (vista anidada)
            hier_chart, n_hier = pp.chart_hierarchy(rows)
            if hier_chart is not None:
                st.markdown(f'**🌿 Tipos de borreguil** (clasificación jerárquica de los '
                            f'{n_hier} detectados): **ambiente** → **humedad** → **pureza**. '
                            f'Cada columna es una humedad; cada fila, un ambiente; el color '
                            f'reparte la pureza.')
                st.altair_chart(hier_chart)
                _rsd = bp.resumen_revision(rows, pp.is_borreguil_decision)
                st.caption(i18n.pick(
                    f"{_rsd['n_revisados']} de {_rsd['n_borreguil']} tienen el tipo revisado "
                    f"por una persona; en el resto se muestra la propuesta de la app.",
                    f"{_rsd['n_revisados']} of {_rsd['n_borreguil']} have their type "
                    f"reviewed by a person; the rest show the app's proposal."))
                hier_df = pp.hierarchy_table(rows)
                if hier_df is not None:
                    st.markdown('Recuento por **combinación completa** (ambiente · humedad · pureza):')
                    st.dataframe(hier_df, hide_index=True, width='stretch')
                st.caption('Reglas sobre las variables ya calculadas (cercanía a lagunas, '
                           'TWI, pendiente, NDMI, NDWI, roca/agua…). Arroyo se detecta por '
                           'topografía (TWI alto o pendiente baja) cuando no hay capa OSM. '
                           'La **pureza es relativa** a esta población: marca como mixto-agua/'
                           'mixto-roca el ~15% con más firma de agua (NDWI) o roca (albedo/'
                           'NDVI bajo), porque a 10 m no se mide la fracción sub-píxel '
                           'absoluta (para eso harían falta PlanetScope o CIR). Umbrales '
                           'en `borreguil_pipeline.py` (HIER_THRESH).')
                st.divider()

            st.markdown(
                f'Distribución de las variables comparando los **{n_borr} puntos '
                f'clasificados como borreguil** (verde) frente al **resto** '
                f'({len(rows) - n_borr}, rojo). Cada caja muestra mediana, cuartiles '
                f'(p25–p75) y bigotes; los puntos sueltos son valores atípicos.')
            box_chart, _ = pp.chart_boxplots(rows)
            if box_chart is not None:
                st.altair_chart(box_chart, width='stretch')
            else:
                st.info('No hay variables numéricas suficientes para los boxplots.')

            st.divider()
            st.markdown('**Dispersión Altitud × NDVI** — separación entre clases. '
                        'Pasa el ratón por un punto para ver su ID y probabilidad RF.')
            sc = pp.chart_scatter(rows)
            if sc is not None:
                st.altair_chart(sc, width='stretch')

            st.divider()
            st.markdown('**Recuento de puntos por categoría de decisión.**')
            dc = pp.chart_decision_counts(rows)
            if dc is not None:
                st.altair_chart(dc, width='stretch')

        # ----------------------------------------------------------------
        # MAPEO DE BORREGUILES — RANDOM FOREST (por pasos; ahora: «Datos»)
        # De los puntos con el tipo revisado a un mapa de tipos. La lógica está en
        # mapeo_rf.py; aquí solo se enseña. No toca los puntos ni el resto de la app.
        # ----------------------------------------------------------------
        with tab7:
            import mapeo_rf as mr
            _en = i18n.get_lang() == 'en'
            st.subheader(i18n.pick('Mapeo de borreguiles — Random Forest',
                                   'Borreguil mapping — Random Forest'))
            st.markdown(i18n.pick(
                'Usa los puntos cuyo **tipo has revisado** para entrenar un modelo y obtener '
                'un **mapa de tipos de borreguil**. Se construye por pasos: ahora está '
                'disponible el primero, **Datos**. El modelo y el mapa llegarán en los '
                'siguientes.',
                'Uses the points whose **type you have reviewed** to train a model and '
                'produce a **map of borreguil types**. It is being built step by step: the '
                'first one, **Data**, is available now. The model and the map come next.'))

            st.markdown(i18n.pick('#### 1 · Datos', '#### 1 · Data'))
            _niv_txt = {
                'ambiente': i18n.pick('Ambiente (arroyo · laguna · ladera)',
                                      'Environment (stream · lake · slope)'),
                'humedad': i18n.pick('Humedad (húmedo · seco)', 'Moisture (wet · dry)'),
                'pureza': i18n.pick('Pureza (puro · mixto-agua · mixto-roca)',
                                    'Purity (pure · mixed-water · mixed-rock)'),
            }
            # Ausencias de campo: lo único que puede formar la clase «no borreguil».
            _n_aus = sum(1 for r in rows if bp._txt(r.get('Borreguil')).lower() == 'no'
                         and bp._txt(r.get('Duda')).lower() != 'si')
            # Uno debajo del otro: en un panel estrecho, a media anchura el texto
            # del desplegable no cabía.
            _nivel = st.selectbox(
                i18n.pick('Qué quieres cartografiar', 'What to map'),
                list(mr.NIVELES), key='mrf_nivel', format_func=lambda n: _niv_txt[n],
                help=i18n.pick(
                    'Se entrena un modelo por nivel (tres mapas), no uno con las 18 '
                    'combinaciones: no habría puntos para casi ninguna.',
                    'One model per level is trained (three maps), not one with the 18 '
                    'combinations: there would be no points for most of them.'))
            if _n_aus:
                _incluir = st.checkbox(
                    i18n.pick(
                        f'Incluir la clase «no borreguil» ({_n_aus} ausencias de campo)',
                        f'Include the "not a borreguil" class ({_n_aus} field absences)'),
                    value=True, key='mrf_ausencias',
                    help=i18n.pick(
                        'Son los puntos que alguien marcó en campo como «no es '
                        'borreguil». No se inventan ausencias: sin esos puntos, la '
                        'clase no existe.',
                        'These are the points someone marked in the field as "not a '
                        'borreguil". Absences are never invented: without those '
                        'points the class does not exist.'))
            else:
                _incluir = False
                st.caption(i18n.pick(
                    'No hay ausencias de campo en estos puntos, así que no habrá clase '
                    '«no borreguil» (no se inventan ausencias).',
                    'There are no field absences among these points, so there will be '
                    'no "not a borreguil" class (absences are never invented).'))
            _res = mr.muestras(rows, _nivel, incluir_ausencias=bool(_incluir and _n_aus),
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
                         + ''.join('\n- ' + mr.mensaje(e, _en, i18n.tv)
                                   for e in _res['errores']))
            _otros = [a for a in _res['avisos'] if a[0] != 'sin_ausencia']
            if _otros:
                st.warning(''.join('- ' + mr.mensaje(a, _en, i18n.tv) + '\n' for a in _otros))
            if ('sin_ausencia',) in _res['avisos']:
                st.info(mr.mensaje(('sin_ausencia',), _en))

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
                        'muestras en cuanto los revises en la pestaña **Tabla**.',
                        f'The {_sin_rev} borreguiles whose type has not been reviewed will '
                        'become samples as soon as you review them in the **Table** tab.'))
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
            _elegidas = []
            for _gi, _g in enumerate(mr.GRUPOS):
                _ops = mr.nombres_predictores([_g])
                # Solo el nombre: con la descripción al lado las etiquetas se cortaban.
                # Qué mide cada una está en la tabla de debajo.
                _elegidas += st.multiselect(
                    mr.nombre_grupo(_g, _en), _ops, default=_ops, key=f'mrf_vars_{_gi}',
                    format_func=lambda n: n)
            _predictores = mr.ordenar_predictores(_elegidas)
            st.markdown(i18n.pick(
                f'**{len(_predictores)} de {len(mr.PREDICTORES)} variables seleccionadas.**',
                f'**{len(_predictores)} of {len(mr.PREDICTORES)} variables selected.**'))
            if not _predictores:
                st.error(i18n.pick('Elige al menos una variable.',
                                   'Choose at least one variable.'))
            with st.expander(i18n.pick('Qué mide cada variable', 'What each variable measures')):
                st.dataframe(pd.DataFrame([{
                    'Variable': n,
                    i18n.pick('Qué mide', 'What it measures'): (en_ if _en else es_),
                    i18n.pick('Grupo', 'Group'): mr.nombre_grupo(g, _en),
                    i18n.pick('Seleccionada', 'Selected'):
                        i18n.pick('sí', 'yes') if n in _predictores else 'no'}
                    for n, g, es_, en_ in mr.PREDICTORES]), hide_index=True, width='stretch')
            with st.expander(i18n.pick('Variables que usa el detector y aquí no están',
                                       'Variables the detector uses that are missing here')):
                for _q, _p, _q_en, _p_en in mr.NO_DISPONIBLES:
                    st.markdown(f'- **{i18n.pick(_q, _q_en)}** — {i18n.pick(_p, _p_en)}.')

            st.divider()
            st.info(i18n.pick(
                '**Siguiente paso (todavía no disponible):** leer en Earth Engine el valor '
                'de estas variables en el píxel de cada muestra, comprobar la calidad de la '
                'tabla y entrenar el modelo.',
                '**Next step (not available yet):** read from Earth Engine the value of '
                'these variables at the pixel of each sample, check the quality of the '
                'table and train the model.'))
