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

# Carpeta ESCRIBIBLE del usuario. En la versión standalone (ejecutable) APP_DIR es
# una carpeta temporal de solo lectura que se borra al salir, así que lo que el
# usuario guarda (modelos entrenados) debe ir aquí. Sin la variable de entorno
# —modo desarrollo— coincide con APP_DIR y todo sigue igual que antes.
DATA_DIR = Path(os.environ.get('BORREGUIL_DATA_DIR') or APP_DIR)
try:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    DATA_DIR = APP_DIR


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


@st.cache_data(show_spinner=False)
def _etiquetas_de_subida(nombre, datos):
    """Lo que leer_etiquetas() leería de un fichero de puntos candidatos recién
    subido, para avisar antes de ejecutar. None si no se puede leer aquí (p. ej. un
    .shp sin sus .dbf/.shx): al ejecutar ya se verá."""
    d = tempfile.mkdtemp(prefix='candidatos_')
    ruta = os.path.join(d, os.path.basename(nombre) or 'puntos.json')
    try:
        with open(ruta, 'wb') as f:
            f.write(datos)
        return bp.resumen_etiquetas(bp.leer_etiquetas(bp.read_points(ruta)))
    except Exception:
        return None
    finally:
        shutil.rmtree(d, ignore_errors=True)


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
    gen_lejos = 0
    if input_mode.startswith('📁'):
        f_input = st.file_uploader('Puntos candidatos',
                                      type=['kml','geojson','shp','json'])
        if f_input is not None:
            for _niv_et, _txt_et in (_etiquetas_de_subida(f_input.name, f_input.getvalue())
                                     or []):
                {'ok': st.caption, 'info': st.info, 'aviso': st.warning}[_niv_et](
                    ('✓ ' if _niv_et == 'ok' else '') + _txt_et)
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
        gen_lejos = st.number_input(
            'Lejos de tus puntos de campo (m)', min_value=0, max_value=5000, value=0,
            step=50,
            help='Si subes tu verdad-terreno (debajo), los puntos al azar quedan al '
                 'menos a esta distancia de tus puntos de campo: caen en sitios que '
                 'todavía no has muestreado, para buscar allí ausencias (o algún '
                 'borreguil nuevo) que revisas tú en la tabla. 0 = sin límite.\n\n'
                 '**Recomendado: 0** para probar el modelo por toda el área; **300 m** '
                 'para buscar ausencias fuera de tus puntos.')

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
            '- Valor de **presencia**: `si` · `yes` · `1` · `presencia` · `borreguil`.\n'
            '- Valor de **ausencia**: `no` · `0` · `ausencia` · `no_borreguil`.\n\n'
            'En los **puntos candidatos** también vale: la etiqueta puede venir en '
            '`Borreguil`, `presence`, `presencia`, `label`, `clase`… (si/no, yes/no, '
            '1/0, true/false) y los dudosos en `Duda`, `dudoso` o `doubtful`. Al '
            'subirlos, la app te dice qué columna ha leído o te avisa si no encuentra '
            'ninguna.\n\n'
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
        # Arranca con el último proyecto con el que la conexión funcionó.
        gee_project = st.text_input('Proyecto Google Earth Engine',
                                      value=leer_ajuste('gee_project', ''),
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
                    if ok and gee_project.strip():
                        guardar_ajuste('gee_project', gee_project.strip())
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
                                                excluir=_excluir, lejos_m=float(gen_lejos))
                status.update(label=f'{len(rows)} puntos generados'
                              + (f' a {gen_lejos:.0f} m o más de tus {len(_excluir)} '
                                 'puntos de campo' if _excluir else ''), state='complete')

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
                parar_ejecucion()

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
                for _niv_et, _txt_et in bp.resumen_etiquetas(bp.leer_etiquetas(rows)):
                    {'ok': st.write, 'info': st.info, 'aviso': st.warning}[_niv_et](_txt_et)
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
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
            ['Mapa', 'Tabla', 'Histograma RF', 'Descargas', '🔑 Variables',
             '📊 Distribuciones'])

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
                # Lo contrario de verificar como borreguil: un punto que una persona ha
                # mirado y NO lo es (por ejemplo, uno generado al azar). Queda como
                # ausencia verificada, igual que las de campo, y el modelo la usa para
                # aprender qué no es borreguil. Un no borreguil no tiene tipo: si tenía
                # uno revisado, se le quita.
                if st.button('✗ Marcar como no borreguil verificado y reentrenar',
                             key='mark_absent',
                             help='Para los puntos que has mirado (con la ortofoto o en campo) '
                                  'y no son borreguil. Quedan como «NO BORREGUIL '
                                  'VERIFICADO», igual que las ausencias de tu verdad-terreno: '
                                  'sirven de ausencias para el modelo y para el mapa de '
                                  'tipos.'):
                    with st.spinner('Incorporando las ausencias y reentrenando…'):
                        rows2 = [dict(r) for r in rows]
                        sset = set(sel_ids)
                        for r in rows2:
                            if r.get('ID') in sset:
                                r['Borreguil'] = 'no'; r['origin'] = 'verificado'
                                r['Duda'] = ''
                                if bp.tipo_revisado(r):
                                    bp.clear_tipo_revisado(r)
                        new_base = set(base_truth_ids) - sset
                        info2 = bp.run_rf_and_decide(rows2, threshold=threshold,
                                                     default_model_path=MODEL_PATH)
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
                            'mode': f'ausencias verificadas (+{len(sset)})',
                            'cutoff': None,
                            'info': info2,
                            'counts': dict(Counter(r['decision'] for r in rows2)),
                            'n_train_pos': n_train_pos,
                            'n_new': len(sset),
                        }]
                        st.session_state.sel_point = None
                    st.success(f'{len(sset)} ausencia(s) verificada(s) incorporada(s). '
                               'Modelo reentrenado.')
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
                        'cambiar a Google Satélite, que admite más zoom, o a Relieve '
                        'LiDAR, que enseña la forma del terreno (regueros y cauces '
                        'pequeños: arroyo o ladera), y encender las curvas de nivel.',
                        'Orthophoto background; the yellow square is the 10 m pixel of '
                        'the point. Use the layer control (top right) to switch to '
                        'Google Satellite, which allows more zoom, or to LiDAR relief, '
                        'which shows the shape of the terrain (gullies and small '
                        'streams: stream or slope), and to turn on the contour lines.'))
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
    st.subheader(i18n.pick('Mapa de tipos de borreguil — Random Forest',
                           'Map of borreguil types — Random Forest'))
    if 'rows' not in st.session_state:
        st.info(i18n.pick(
            'Aquí se entrena un modelo con los borreguiles cuyo **tipo has revisado** y '
            'se obtiene un **mapa de tipos**. Todavía no hay puntos en la sesión:\n\n'
            '1. Ejecuta el análisis (panel lateral → **▶ Ejecutar pipeline**). El '
            'resultado sale en la pestaña **📍 Verdad terreno**.\n'
            '2. En **Resultados → Tabla**, revisa el tipo de los borreguiles: ambiente, '
            'humedad y pureza.\n'
            '3. Vuelve a esta pestaña.',
            'Here a model is trained with the borreguiles whose **type you have '
            'reviewed** to produce a **type map**. There are no points in the session '
            'yet:\n\n'
            '1. Run the analysis (sidebar → **▶ Run pipeline**). The result appears in '
            'the **📍 Ground truth** tab.\n'
            '2. Under **Results → Table**, review the type of the borreguiles: '
            'environment, moisture and purity.\n'
            '3. Come back to this tab.'))
    else:
        rows = st.session_state.rows
        st.markdown(i18n.pick(
            'Entrena un modelo de **borreguil sí o no** con los puntos marcados en campo y, '
            'si quieres el tipo, otro con los puntos cuyo **tipo has revisado** (en **📍 '
            'Verdad terreno → Tabla**), y obtén el **mapa de borreguiles**. Los pasos van '
            'en orden, cada uno en su pestaña.',
            'Train a **borreguil yes-or-no** model with the points marked in the field and, '
            'if you want the type, another one with the points whose **type you have '
            'reviewed** (under **📍 Ground truth → Table**), and get the **borreguil map**. '
            'The steps go in order, each in its own tab.'))
        _niv_txt = {
            mr.NIVEL_SI_NO: i18n.pick('Borreguil sí o no', 'Borreguil yes or no'),
            'ambiente': i18n.pick('Ambiente (arroyo · laguna · ladera)',
                                  'Environment (stream · lake · slope)'),
            'humedad': i18n.pick('Humedad (húmedo · seco)', 'Moisture (wet · dry)'),
            'pureza': i18n.pick('Pureza (puro · mixto-agua · mixto-roca)',
                                'Purity (pure · mixed-water · mixed-rock)'),
            # corto: en el panel estrecho el desplegable no parte el texto, lo corta
            'ambiente_humedad': i18n.pick(
                'Ambiente y humedad (en cada ambiente, húmedo o seco)',
                'Environment and moisture (in each one, wet or dry)'),
        }
        # El nivel va encima de los pasos: lo que se elige aquí vale para todos. Viene
        # elegido el primer paso: borreguil sí o no.
        if st.session_state.get('mrf_nivel') not in mr.NIVELES:
            st.session_state['mrf_nivel'] = mr.NIVEL_SI_NO
        _nivel = st.selectbox(
            i18n.pick('Qué quieres cartografiar', 'What to map'),
            list(mr.NIVELES), key='mrf_nivel', format_func=lambda n: _niv_txt[n],
            help=i18n.pick(
                'Se entrena un modelo por cada opción.\n\n'
                '**Borreguil sí o no** decide solo si hay borreguil. Aprende de **todos '
                'los puntos marcados en campo** como borreguil o no, tengan o no el tipo '
                'revisado. Su mapa ya es el de borreguil sí o no.\n\n'
                'Para el **tipo**, entrena después uno de los otros niveles: aprende el '
                'tipo solo de los borreguiles con el tipo revisado, y en su mapa el sí o '
                'no lo sigue decidiendo el modelo de **Borreguil sí o no**.\n\n'
                '**Ambiente y humedad** da el tipo completo: el modelo dice si es húmedo '
                'o seco, y el ambiente (arroyo, laguna o ladera) sale de la **cercanía de '
                'cada píxel a lagunas y cauces**, que se ajusta en **4 · Mapa**.\n\n'
                '**Recomendado: primero Borreguil sí o no, y después Ambiente y '
                'humedad.**\n\n'
                'En **Ambiente**, en cambio, el ambiente lo decide el modelo, que no ve '
                'dónde están las lagunas ni los cauces: puede poner «laguna» lejos de '
                'cualquier laguna.\n\n'
                'No se ofrece la combinación de los tres niveles (18 categorías): no '
                'habría puntos para casi ninguna. La pureza se añade al mapa de otra '
                'manera: «Pureza dentro del píxel», en **4 · Mapa**.',
                'One model is trained for each option.\n\n'
                '**Borreguil yes or no** only decides whether there is a borreguil. It '
                'learns from **every point marked in the field** as a borreguil or not, '
                'whether or not its type has been reviewed. Its map is already the '
                'borreguil yes-or-no map.\n\n'
                'For the **type**, train one of the other levels afterwards: it learns '
                'the type only from the borreguiles whose type has been reviewed, and in '
                'its map the yes or no is still decided by the **Borreguil yes or no** '
                'model.\n\n'
                '**Environment and moisture** gives the full type: the model says whether '
                'it is wet or dry, and the environment (stream, lake or slope) comes from '
                '**how close each pixel is to lakes and watercourses**, set under **4 · '
                'Map**.\n\n'
                '**Recommended: first Borreguil yes or no, then Environment and '
                'moisture.**\n\n'
                'Under **Environment**, by contrast, the model decides the environment, '
                'and it cannot see where lakes and watercourses are: it may say "lake" '
                'far from any lake.\n\n'
                'The combination of the three levels (18 categories) is not offered: '
                'there would be no points for most of them. Purity is added to the map '
                'in another way: "Purity within the pixel", under **4 · Map**.'))
        # En «Ambiente y humedad» el modelo aprende solo el nivel de dentro (la humedad);
        # el de fuera (el ambiente) lo pone el mapa por cercanía a lagunas y cauces.
        _nivel_mod = mr.nivel_del_modelo(_nivel)
        _cercania = mr.nivel_por_cercania(_nivel)
        if _nivel == 'ambiente':
            st.warning(i18n.pick(
                'En este nivel el ambiente lo decide **el modelo**, que solo conoce lo '
                'que hay en cada píxel y no ve dónde están las lagunas ni los cauces: '
                'puede poner «laguna» lejos de cualquier laguna. Para que el ambiente '
                'salga de la **cercanía a lagunas y cauces**, elige «Ambiente y humedad».',
                'At this level the environment is decided by **the model**, which only '
                'knows what is in each pixel and cannot see where lakes and watercourses '
                'are: it may say "lake" far from any lake. For the environment to come '
                'from **closeness to lakes and watercourses**, choose "Environment and '
                'moisture".'))

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
                         + ''.join('\n- ' + mr.mensaje(e, _en, i18n.tv)
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
                        'muestras en cuanto los revises en **📍 Verdad terreno → Tabla**.',
                        f'The {_sin_rev} borreguiles whose type has not been reviewed will '
                        'become samples as soon as you review them under **📍 Ground truth → '
                        'Table**.'))
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

        # ------------------------------------------ 2 · Tabla de entrenamiento
        # El valor de cada variable se lee en el PÍXEL de 10 m de cada muestra, de
        # la misma pila de Earth Engine que después se clasificará.
        with _t_tabla:
            st.caption(i18n.pick(
                'Lee en Earth Engine el valor de cada variable en el **píxel de 10 m** de '
                'cada muestra: el mismo píxel que después se clasificará. Por eso no se '
                'usan los valores que ya tienen los puntos, que son medias en un radio de '
                '25 m.',
                'Reads from Earth Engine the value of each variable at the **10 m pixel** '
                'of each sample: the same pixel that will later be classified. That is why '
                'the values the points already have, which are means over a 25 m radius, '
                'are not used.'))
            import gee_backend as geb
            try:
                _anios = list(bp.parse_years(years_str))
            except Exception:
                _anios = []
            _usadas = [f for f in _res['tabla'] if f['usada']]
            _epsg = (geb.utm_de([(float(f['lon']), float(f['lat'])) for f in _usadas])
                     if _usadas else None)
            # Lo ya leído vale mientras no cambien los años, la zona UTM de la rejilla
            # ni la manera de calcular las variables (geb.VERSION_PILA).
            _lect = st.session_state.get('mrf_lectura')
            if _lect and (_lect.get('anios') != _anios or _lect.get('epsg') != _epsg
                          or (_lect.get('info') or {}).get('version') != geb.VERSION_PILA):
                _lect = None
            _leidos = (_lect or {}).get('valores', {})
            _pend = [f for f in _usadas
                     if mr.clave_punto(f['lon'], f['lat']) not in _leidos]

            _proy = st.text_input(
                i18n.pick('Proyecto de Google Earth Engine', 'Google Earth Engine project'),
                value=(gee_project.strip() or leer_ajuste('gee_project', '')),
                key='mrf_proyecto', placeholder='ee-tunombre',
                help=i18n.pick(
                    'El mismo que usas en el panel lateral. Se recuerda para la próxima vez.',
                    'The same one you use in the sidebar. It is remembered for next time.'))
            st.caption(i18n.pick(
                f'Años de Sentinel-2: **{years_str}** (los del panel lateral).',
                f'Sentinel-2 years: **{years_str}** (those in the sidebar).'))
            _rm = st.session_state.pop('_mrf_msg', None)
            if _rm:
                (st.success if _rm[0] == 'ok' else st.warning)(_rm[1])

            _puede = bool(_res['listo'] and _predictores and _anios and _proy.strip())
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
            if _predictores and any(mr.clave_punto(f['lon'], f['lat']) in _leidos
                                    for f in _usadas):
                _tab = mr.construir_tabla(_res, _leidos, _predictores)
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
                if _tab['apartadas']:
                    _cuenta = Counter(a['motivo'] for a in _tab['apartadas'])
                    st.warning(i18n.pick('**Muestras que no entran en la tabla:**',
                                         '**Samples left out of the table:**')
                               + ''.join(f'\n- {n} — {mr.motivo_tabla(k, _en)}'
                                         for k, n in _cuenta.most_common()))
                    with st.expander(i18n.pick('Ver cuáles son', 'See which ones')):
                        st.dataframe(pd.DataFrame([{
                            'ID': a['ID'],
                            i18n.pick('categoría', 'category'): i18n.tv(a['categoria']),
                            i18n.pick('motivo', 'reason'): mr.motivo_tabla(a['motivo'], _en),
                            i18n.pick('variables sin dato', 'variables without data'):
                                ', '.join(a['faltan'][:6])
                                + (' …' if len(a['faltan']) > 6 else '')}
                            for a in _tab['apartadas']]), hide_index=True, width='stretch')
                if _tab['errores']:
                    st.error(i18n.pick('**Con esta tabla todavía no se puede entrenar:**',
                                       '**This table is not enough to train yet:**')
                             + ''.join('\n- ' + mr.mensaje(e, _en, i18n.tv)
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
                        + ''.join('\n- ' + mr.mensaje_calidad(h, _en) for h in _hall[:_max])
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
                        f"{_des[2]['n']} ({_des[0]:.1f} veces menos). Al entrenar se "
                        'compensará dando más peso a las categorías escasas.',
                        f"**Unbalanced categories:** \"{i18n.tv(_des[1]['categoria'])}\" has "
                        f"{_des[1]['n']} samples and \"{i18n.tv(_des[2]['categoria'])}\" has "
                        f"{_des[2]['n']} ({_des[0]:.1f} times fewer). Training will "
                        'compensate by giving more weight to the scarce categories.'))

                with st.expander(i18n.pick('Ver la tabla de entrenamiento',
                                           'See the training table')):
                    _dft = pd.DataFrame(_tab['filas'])
                    if len(_dft):
                        # solo lo que es la tabla X/y: las coordenadas no son variables
                        _dft = _dft.drop(columns=['indice', 'lon', 'lat'])
                        _dft['ID'] = bp.abreviar_ids(_dft['ID'].tolist())
                        _dft['categoria'] = _dft['categoria'].map(i18n.tv)
                        st.dataframe(_dft.rename(columns={
                            'categoria': i18n.pick('categoría', 'category'),
                            'codigo': i18n.pick('código', 'code')}),
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
                        f"Leído el {(_lect.get('cuando') or '').replace('T', ' ')} · rejilla "
                        f"{_inf.get('crs')} de {_inf.get('resolucion_m')} m · años "
                        f"{bp.abreviar_anios(_inf.get('anios', []))} · estadísticas de "
                        'jun – sep calculadas en cada año y promediadas entre años · '
                        'topografía de 30 m interpolada (bilineal).',
                        f"Read on {(_lect.get('cuando') or '').replace('T', ' ')} · "
                        f"{_inf.get('crs')} grid at {_inf.get('resolucion_m')} m · years "
                        f"{bp.abreviar_anios(_inf.get('anios', []))} · Jun – Sep statistics "
                        'computed in each year and averaged across years · 30 m topography '
                        'interpolated (bilinear).'))

                # ---- Variables repetidas (VIF). Es una opción: de entrada no se quita
                # nada. Se calcula con los valores de las muestras de esta tabla; el
                # modelo, el mapa y el peso usan después la tabla con las que se quedan.
                st.markdown(i18n.pick('#### Variables repetidas (VIF)',
                                      '#### Repeated variables (VIF)'))
                _vif_si = st.checkbox(
                    i18n.pick('Quitar las variables repetidas con el VIF',
                              'Remove repeated variables with the VIF'),
                    key='mrf_vif_si',
                    help=i18n.pick(
                        'El **VIF** de una variable dice qué parte de ella se puede '
                        'adivinar con todas las demás juntas: 5 quiere decir el 80 %; 10, '
                        'el 90 %. Con esta opción se quita la que más se repite (la de VIF '
                        'más alto), se vuelve a calcular y se sigue mientras alguna pase '
                        'del umbral. Se calcula con los valores de las muestras de esta '
                        'tabla.\n\n'
                        'El Random Forest no lo necesita para acertar, pero con menos '
                        'variables el modelo es más sencillo y su peso, en «5 · Peso», se '
                        'lee mejor. Si acierta igual o no lo dice la comparación del final '
                        'de «3 · Modelo».\n\n'
                        '**Recomendado:** entrenar una vez sin él y otra con él, y quedarse '
                        'con el que mejor valide.',
                        'The **VIF** of a variable says how much of it can be guessed from '
                        'all the others together: 5 means 80 %; 10, 90 %. With this option '
                        'the most repeated one (the highest VIF) is removed, the VIF is '
                        'computed again, and so on while any is above the threshold. It is '
                        'computed with the values of the samples of this table.\n\n'
                        'The Random Forest does not need it to be right, but with fewer '
                        'variables the model is simpler and its weights, under "5 · '
                        'Weights", are easier to read. Whether it does as well is shown by '
                        'the comparison at the end of "3 · Model".\n\n'
                        '**Recommended:** train once without it and once with it, and keep '
                        'the one that validates better.'))
                if _vif_si:
                    # Los valores de partida se ponen justo antes de dibujar los controles
                    # (si no, el navegador no los enseña); las fijas, solo las que existen.
                    st.session_state.setdefault('mrf_vif_umbral', float(mr.VIF_UMBRAL))
                    _fv = st.session_state.get('mrf_vif_fijas')
                    if _fv is None:
                        st.session_state['mrf_vif_fijas'] = [
                            v for v in mr.VIF_FIJAS if v in _tab['predictores']]
                    elif any(v not in _tab['predictores'] for v in _fv):
                        st.session_state['mrf_vif_fijas'] = [
                            v for v in _fv if v in _tab['predictores']]
                    _vif_u = st.number_input(
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
                    _vif_f = st.multiselect(
                        i18n.pick('Variables que se quedan siempre', 'Variables that always stay'),
                        _tab['predictores'], key='mrf_vif_fijas', format_func=lambda n: n,
                        help=i18n.pick(
                            'Estas no se quitan nunca, aunque su VIF pase del umbral (sí '
                            'cuentan al calcular el de las demás). Viene la altitud: en '
                            'cualquier sitio, los borreguiles son prados húmedos de alta '
                            'montaña.\n\n'
                            '**Recomendado: la altitud.**',
                            'These are never removed, even if their VIF is above the '
                            'threshold (they do count when computing that of the others). '
                            'Elevation comes selected: anywhere, borreguiles are wet '
                            'high-mountain meadows.\n\n'
                            '**Recommended: elevation.**'))
                    if not _tab['listo'] or len(_tab['filas']) < 3:
                        st.info(i18n.pick(
                            'El VIF se calcula cuando la tabla esté lista para entrenar.',
                            'The VIF is computed once the table is ready for training.'))
                    else:
                        _sel = mr.elegir_por_vif(_tab['X'], _tab['predictores'], _vif_u, _vif_f)
                        _de_vif = len(_tab['predictores'])
                        st.markdown(i18n.pick(
                            f"**Se quedan {len(_sel['quedan'])} de {_de_vif} variables.** "
                            'El modelo, el mapa y el peso usan solo estas.',
                            f"**{len(_sel['quedan'])} of {_de_vif} variables stay.** The "
                            'model, the map and the weights use only these.'))
                        if _sel['pocas_muestras']:
                            st.warning(i18n.pick(
                                f"Con {len(_tab['filas'])} muestras y {_de_vif} variables el "
                                'VIF no es fiable: hacen falta bastantes más muestras que '
                                'variables.',
                                f"With {len(_tab['filas'])} samples and {_de_vif} variables "
                                'the VIF is not reliable: many more samples than variables '
                                'are needed.'))

                        def _txt_vif(v):
                            return '—' if v is None else ('∞' if v == float('inf')
                                                          else bp.fmt_num(v))
                        _k_q = i18n.pick('Qué pasa', 'What happens')
                        st.dataframe(pd.DataFrame(
                            [{'Variable': n, 'VIF': _txt_vif(_sel['vif'][n]),
                              _k_q: (i18n.pick('se queda siempre', 'always stays')
                                     if n in _sel['fijas'] else i18n.pick('se queda', 'stays'))}
                             for n in _sel['quedan']]
                            + [{'Variable': q['variable'], 'VIF': _txt_vif(q['vif']),
                                _k_q: i18n.pick(f"se quita (paso {q['paso']})",
                                                f"removed (step {q['paso']})")
                                + (i18n.pick(': no varía', ': it does not vary')
                                   if q['vif'] is None else '')}
                               for q in _sel['quitadas']]),
                            hide_index=True, width='stretch')
                        st.caption(i18n.pick(
                            'De las que se quedan, su VIF entre ellas; de las que se quitan, '
                            'el que tenían al quitarlas. ∞: se obtiene exactamente con '
                            'otras (la caída del NDVI, por ejemplo, es el NDVI de inicio '
                            'menos el de fin de verano).',
                            'For those that stay, their VIF among themselves; for those '
                            'removed, the one they had when removed. ∞: it is obtained '
                            'exactly from others (the NDVI drop, for instance, is the '
                            'early-summer NDVI minus the late-summer one).'))
                        # La tabla con la que se sigue: solo con las que se quedan (puede
                        # recuperar muestras a las que solo les faltaba una de las quitadas).
                        _tab = mr.construir_tabla(_res, _leidos, _sel['quedan'])
                        _tab['vif'] = {'umbral': float(_vif_u), 'fijas': list(_sel['fijas']),
                                       'de': _de_vif,
                                       'quitadas': [q['variable'] for q in _sel['quitadas']]}

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
                        # y si las variables salieron del VIF (None = las de «1 · Datos»)
                        _nuevo_m['variables'] = _tab.get('vif')
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
                    _eu, _u_txt = _regla['en_umbral'], f"{_regla['umbral']:.3f}"
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
                        f"borreguiles (sensibilidad {bp.fmt_num(_eu['sensibilidad'])});\n"
                        f"- descarta bien **{_eu['vn']} de {_eu['vn'] + _eu['fp']}** de lo "
                        f"que no lo es (especificidad {bp.fmt_num(_eu['especificidad'])});\n"
                        f"- TSS **{bp.fmt_num(_eu['tss'])}** · F1 de borreguil "
                        f"{bp.fmt_num(_eu['f1'])}.",
                        f'A pixel counts as a **borreguil** if {_cuando}. '
                        'With that threshold, in the validation:\n\n'
                        f"- it finds **{_eu['vp']} of {_eu['vp'] + _eu['fn']}** borreguiles "
                        f"(sensitivity {bp.fmt_num(_eu['sensibilidad'])});\n"
                        f"- it correctly rejects **{_eu['vn']} of {_eu['vn'] + _eu['fp']}** "
                        f"of what is not one (specificity {bp.fmt_num(_eu['especificidad'])});\n"
                        f"- TSS **{bp.fmt_num(_eu['tss'])}** · F1 of borreguil "
                        f"{bp.fmt_num(_eu['f1'])}."))
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
                                alt.Tooltip('TSS:Q', format='.4f'),
                                alt.Tooltip('F1:Q', format='.4f'),
                                alt.Tooltip('sens:Q', format='.4f', title=i18n.pick(
                                    'sensibilidad', 'sensitivity')),
                                alt.Tooltip('espec:Q', format='.4f', title=i18n.pick(
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
                        _c4u = st.column_config.NumberColumn(format='%.4f')
                        _k_se = i18n.pick('Sensibilidad', 'Sensitivity')
                        _k_es = i18n.pick('Especificidad', 'Specificity')
                        _k_pr = i18n.pick('Precisión', 'Precision')
                        st.dataframe(pd.DataFrame([{
                            i18n.pick('Umbral', 'Threshold'): b['umbral'],
                            'TSS': b['tss'], 'F1': b['f1'], _k_se: b['sensibilidad'],
                            _k_es: b['especificidad'], _k_pr: b['precision'],
                            i18n.pick('Borreguiles encontrados', 'Borreguiles found'): b['vp'],
                            i18n.pick('Borreguiles perdidos', 'Borreguiles missed'): b['fn'],
                            i18n.pick('Dados por borreguil sin serlo',
                                      'Wrongly called borreguil'): b['fp'],
                            i18n.pick('Bien descartados', 'Correctly rejected'): b['vn']}
                            for b in _bar]), hide_index=True, width='stretch',
                            column_config={
                                i18n.pick('Umbral', 'Threshold'):
                                    st.column_config.NumberColumn(format='%.3f', pinned=True),
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
                           bp.fmt_num(_mod['macro_f1']))
                _m2.metric(i18n.pick('Exactitud equilibrada', 'Balanced accuracy'),
                           bp.fmt_num(_mod['exactitud_equilibrada']))
                _m3.metric(i18n.pick('Aciertos sobre el total', 'Overall accuracy'),
                           bp.fmt_num(_mod['exactitud']))

                _c4 = st.column_config.NumberColumn(format='%.4f')
                st.markdown(i18n.pick('**Por categoría**', '**By category**'))
                _k_pre = i18n.pick('Precisión', 'Precision')
                _k_rec = i18n.pick('Sensibilidad (recall)', 'Recall')
                st.dataframe(pd.DataFrame([{
                    i18n.pick('Categoría', 'Category'): i18n.tv(c['categoria']),
                    _k_pre: c['precision'], _k_rec: c['recall'], 'F1': c['f1'],
                    i18n.pick('Muestras validadas', 'Validated samples'): c['n_validacion']}
                    for c in _mod['por_clase']]), hide_index=True, width='stretch',
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

            # ---- Con qué variables acierta más. Cada modelo entrenado en la sesión deja
            # aquí su acierto en la validación por zonas; se comparan los del mismo nivel
            # y con las mismas muestras (las variables pueden ser otras).
            _f_mu = mr.firma_muestras(_res)
            if _mod and _tab is not None and _mod['firma'] == mr.firma_tabla(_tab):
                st.session_state.setdefault('mrf_variantes', {})[
                    (_nivel, _f_mu, tuple(_mod['predictores']))] = {
                    'vif': (_mod.get('variables') or {}).get('umbral'),
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
                        + (i18n.pick(f"sin repetidas: VIF ≤ {v['vif']:g} ({v['n_var']})",
                                     f"no repeats: VIF ≤ {v['vif']:g} ({v['n_var']})")
                           if v['vif'] else
                           i18n.pick(f"las de «1 · Datos» ({v['n_var']})",
                                     f"those of \"1 · Data\" ({v['n_var']})")),
                    i18n.pick('Muestras', 'Samples'): v['n_muestras'],
                    i18n.pick('F1 medio', 'Mean F1'): bp.fmt_num(v['f1'])}
                    for k, v in _variantes]), hide_index=True, width='stretch')
                st.caption(i18n.pick(
                    'Cada fila es un modelo entrenado en esta sesión con las mismas '
                    'muestras; ▶ es el de ahora, el que hace el mapa. El F1 medio es el de '
                    'la validación por zonas. Una diferencia de unas centésimas puede ser '
                    'azar: con pocas muestras, una sola cambia la cifra. Para quedarte con '
                    'otro, cambia la opción del VIF en «2 · Tabla de entrenamiento» y '
                    'vuelve a entrenar.',
                    'Each row is a model trained in this session with the same samples; ▶ '
                    'is the current one, which makes the map. The mean F1 is that of the '
                    'validation by zones. A difference of a few hundredths may be chance: '
                    'with few samples, a single one changes the figure. To keep another '
                    'one, change the VIF option under "2 · Training table" and train again.'))

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
                        f"{_mod_si_no['regla']['umbral']:.3f}, entrenado el "
                        f"{_mod_si_no['cuando'].replace('T', ' a las ')}), y su tipo, "
                        'este modelo.',
                        'The map is made **in two steps**: whether a pixel is a borreguil '
                        'is decided by the **"Borreguil yes or no"** model (threshold '
                        f"{_mod_si_no['regla']['umbral']:.3f}, trained on "
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
                        if bp.tipo_revisado(_r) and mr.categoria(_r, _nivel):
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
                                f"Máscara leída: {_mc['km2']:.2f} km².",
                                f"Mask read: {_mc['km2']:.2f} km²."))

                # ---- pureza dentro del píxel: puro, mixto-agua o mixto-roca. No sale del
                # modelo, sino de cuánto verde hay en cada píxel de borreguil (mapeo_rf.
                # verde_estimado). Los valores de partida se ajustan aquí.
                _m_pur = []                 # muestras de borreguil: su pureza y su NDVI
                for _f in _usadas:
                    _v = _leidos.get(mr.clave_punto(_f['lon'], _f['lat'])) or {}
                    if _f['categoria'] != mr.AUSENCIA and _v.get(mr.VARIABLE_VERDE) is not None:
                        _r = rows[_f['indice']] if 0 <= _f['indice'] < len(rows) else {}
                        _m_pur.append({'pureza': bp._txt(_r.get('pureza')),
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
                        0.0, 0.5, mr.PUREZA_DEFECTO['sin_verde'], 0.01, format='%.2f',
                        key='mrf_pureza_sin', disabled=not _pur_si,
                        help=i18n.pick(
                            'El NDVI que tiene la roca o el suelo desnudo. Por debajo de '
                            'este valor se cuenta un 0 % de verde.\n\n'
                            '**Recomendado: 0.08**, lo habitual en roca y suelo de alta '
                            'montaña.',
                            'The NDVI of rock or bare ground. Below this value the pixel '
                            'counts as 0 % green.\n\n'
                            '**Recommended: 0.08**, the usual value for high-mountain '
                            'rock and ground.'))
                    _pur_todo = _q2.number_input(
                        i18n.pick('NDVI de un píxel todo verde', 'NDVI of an all-green pixel'),
                        0.2, 1.0, min(1.0, max(0.2, round(_tv_prop, 2))), 0.01, format='%.2f',
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
                        f'de verano de {_tv_prop:.2f} de mediana.' if _tv_muestras else
                        f'Hay {_n_puras} muestras de borreguil puro: con menos de '
                        f'{mr.MIN_PUROS} no se propone su NDVI, sino {_tv_prop:.2f}.',
                        f'Your {_n_puras} pure borreguil samples have a median early-summer '
                        f'NDVI of {_tv_prop:.2f}.' if _tv_muestras else
                        f'There are {_n_puras} pure borreguil samples: with fewer than '
                        f'{mr.MIN_PUROS} their NDVI is not proposed, but {_tv_prop:.2f}.'))
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
                            'desplegable de debajo. Tarda algo más: lee también el relieve '
                            'de alrededor, para los cauces.\n\n'
                            '**Recomendado: marcado.**',
                            'The model decides whether there is borreguil; the environment '
                            'is set by the map from how **close** each borreguil pixel is to '
                            'lakes and watercourses, so no point with a reviewed type is '
                            'needed. On the map, each borreguil pixel comes out as "stream", '
                            '"lake" or "slope". The distances are set in the expander below. '
                            'It takes a little longer: it also reads the surrounding relief, '
                            'for the watercourses.\n\n'
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
                                'distancia o menos de un cauce. Con cauces deducidos del '
                                'relieve, que sitúan la vaguada con unos 30 m de error, no '
                                'conviene bajar de 30 m.\n\n'
                                '**Recomendado: 30 m**, lo que usa la app al proponer el '
                                'tipo de cada punto.',
                                'A pixel that is not a lake one is a stream one if it is '
                                'within this distance of a watercourse. With watercourses '
                                'derived from the relief, which place the valley line with '
                                'about 30 m of error, it is best not to go below 30 m.\n\n'
                                '**Recommended: 30 m**, what the app uses when it proposes '
                                'the type of each point.'))
                        _cauce_ha = _a2.number_input(
                            i18n.pick('Cauces del relieve: hectáreas que vierten',
                                      'Relief watercourses: hectares draining'),
                            0.5, 500.0, mam.DEFECTO['cauce_ha'], 1.0, format='%.1f',
                            key='mrf_amb_ha',
                            help=i18n.pick(
                                'Si no subes una capa de cauces, salen del relieve: se '
                                'calcula por dónde corre el agua y es cauce la celda que '
                                'recibe la de esta superficie o más. Con menos hectáreas '
                                'salen más cauces y más arriba; con más, solo los '
                                'principales.\n\n'
                                '**Recomendado: 8 ha.** Es una aproximación: sitúa bien las '
                                'vaguadas, no el hilo de agua.',
                                'If you do not upload a watercourse layer, they come from '
                                'the relief: where the water runs is computed, and a cell '
                                'is a watercourse if it receives the water of this area or '
                                'more. Fewer hectares give more watercourses, starting '
                                'higher up; more, only the main ones.\n\n'
                                '**Recommended: 8 ha.** It is an approximation: it places '
                                'valley lines well, not the thread of water.'))
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
                                'Si la subes, se usa en lugar de los cauces del relieve.\n\n'
                                '**Recomendado: subirla si tienes una de la que te fíes**; '
                                'es más precisa que el relieve.',
                                'The lines of the watercourses (for example, the official '
                                'hydrographic network). A single KML, GeoJSON or '
                                'GeoPackage, or a shapefile (.shp + .shx + .dbf + .prj '
                                'together, or a .zip). If you upload it, it is used '
                                'instead of the relief watercourses.\n\n'
                                '**Recommended: upload it if you have one you trust**; it '
                                'is more precise than the relief.'))
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
                               'deducidos del relieve (se lee 1 km más alrededor de la zona, '
                               'para contar el agua que entra de fuera).'),
                            f'Lakes: {len(_lagunas_ll)} points of the layer of the app '
                            '(Sierra Nevada) and the water that shows in Sentinel-2. '
                            'Watercourses: '
                            + ('those of your layer.' if _capa_cauces is not None else
                               'derived from the relief (1 km more is read around the zone, '
                               'to count the water coming in from outside).')))
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
                        f"**{_zc['km2']:.2f} km²** que clasificar · rejilla de "
                        f"{_rej['ancho']} × {_rej['alto']} píxeles de 10 m "
                        f"(EPSG:{_m_epsg}) · años {bp.abreviar_anios(_m_anios)} · "
                        f"**{_nu} bloque{'' if _nu == 1 else 's'}** de Earth Engine"
                        + (f', {_nu - _nl} ya leído{"" if _nu - _nl == 1 else "s"}'
                           if _nl < _nu else '')
                        + f' · tardará **{_dura(_min_m)}** aproximadamente.',
                        f"**{_zc['km2']:.2f} km²** to classify · grid of "
                        f"{_rej['ancho']} × {_rej['alto']} pixels of 10 m "
                        f"(EPSG:{_m_epsg}) · years {bp.abreviar_anios(_m_anios)} · "
                        f"**{_nu} Earth Engine block{'' if _nu == 1 else 's'}**"
                        + (f', {_nu - _nl} already read' if _nl < _nu else '')
                        + f' · it will take about **{_dura(_min_m)}**.'))
                    if _min_m > 15:
                        st.warning(i18n.pick(
                            'Va para largo. Mientras se hace el mapa **no uses la app**: '
                            'cualquier clic lo interrumpe. Si se interrumpe no se pierde '
                            'lo leído: vuelve a pulsar el botón y sigue donde iba. Los '
                            'mapas grandes también gastan más cuota de tu proyecto de '
                            'Earth Engine.',
                            'This will take a while. While the map is being made **do '
                            'not use the app**: any click interrupts it. If it is '
                            'interrupted, what has been read is not lost: press the '
                            'button again and it carries on where it was. Large maps '
                            'also use more of your Earth Engine project quota.'))

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
                if not _proy.strip():
                    st.caption(i18n.pick(
                        'Escribe el proyecto de Earth Engine en «2 · Tabla de '
                        'entrenamiento».',
                        'Enter the Earth Engine project under "2 · Training table".'))
                _puede_mapa = bool(_rej and not _pega and _proy.strip() and not _pur_mal
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
                                    _amb_arg = {
                                        'nivel': _nivel, 'params': _par_amb,
                                        'lagunas': [_a_utm_m.transform(lo, la)
                                                    for lo, la in _lagunas_ll],
                                        'cauces': (mra.a_utm(_capa_cauces, _m_epsg)
                                                   if _capa_cauces is not None else None),
                                        'relieve': (None if _capa_cauces is not None
                                                    else _leer_relieve),
                                        'fuentes': {
                                            'lagunas': 'lagunas_sierra_nevada.kml '
                                                       f'({len(_lagunas_ll)}) + Sentinel-2',
                                            'cauces': (', '.join(f.name for f in _f_cauces)
                                                       if _capa_cauces is not None
                                                       else 'relieve')}}
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
                                    f"{_ha_m:.1f} ha clasificadas.",
                                    f"Map made in {_dura(_meta['segundos'] / 60, False)}: "
                                    f"{_ha_m:.1f} ha classified.")
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
                    f"{bp.abreviar_anios(_meta.get('anios', []))} · {_meta['crs']}, "
                    f"{_meta['resolucion_m']} m · {_rej_m['ancho']} × {_rej_m['alto']} "
                    'píxeles.',
                    f"Made on {_meta['cuando'].replace('T', ' ')} · years "
                    f"{bp.abreviar_anios(_meta.get('anios', []))} · {_meta['crs']}, "
                    f"{_meta['resolucion_m']} m · {_rej_m['ancho']} × {_rej_m['alto']} "
                    'pixels.')
                    + ('' if _u_mapa is None else i18n.pick(
                        f' Borreguil si la suma de la probabilidad de sus tipos llega a '
                        f'{_u_mapa:.3f}.',
                        f' Borreguil if the probabilities of its types add up to '
                        f'{_u_mapa:.3f} or more.')))
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
                        + ('ninguno' if _u_mapa is None else f'{_u_mapa:.3f}')
                        + '); el de «3 · Modelo» es ahora '
                        + ('ninguno' if _u_ahora is None else f'{_u_ahora:.3f}')
                        + '. Vuelve a hacerlo para aplicarlo: lo ya leído se aprovecha, '
                        'así que tarda poco.',
                        'This map was made with another "borreguil or not?" threshold ('
                        + ('none' if _u_mapa is None else f'{_u_mapa:.3f}')
                        + '); the one under "3 · Model" is now '
                        + ('none' if _u_ahora is None else f'{_u_ahora:.3f}')
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
                           f'{_n_cl / 100:.1f} ha')
                _k2.metric(i18n.pick('Probabilidad media de la categoría elegida',
                                     'Mean probability of the chosen category'),
                           bp.fmt_num(_meta['pmax_media']))
                _k3.metric(i18n.pick('Píxeles de la zona sin dato',
                                     'Pixels of the zone without data'),
                           f"{100 * (_n_zo - _n_cl) / _n_zo:.2f} %" if _n_zo else '—',
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
                st.dataframe(pd.DataFrame([{
                    i18n.pick('Categoría', 'Category'): i18n.tv(c['categoria']),
                    i18n.pick('Código', 'Code'): c['codigo'],
                    _k_ha: c['hectareas'],
                    _k_pc: (100 * c['pixeles'] / _n_cl) if _n_cl else 0.0}
                    for c in _cats_m]), hide_index=True, width='stretch',
                    column_config={_k_ha: st.column_config.NumberColumn(format='%.2f'),
                                   _k_pc: st.column_config.NumberColumn(format='%.1f')})
                # (en el de sí o no, la tabla de arriba ya va por ambiente)
                if _meta.get('ramas') and _meta.get('nivel') != mr.NIVEL_SI_NO:
                    st.markdown(i18n.pick('**Por ambiente** (húmedo más seco): ',
                                          '**By environment** (wet plus dry): ')
                                + ' · '.join(
                                    f"{i18n.tv(r['rama'])} **{r['hectareas']:.2f} ha**"
                                    for r in _meta['ramas']) + '.')
                if _con_am:
                    st.caption(i18n.pick(
                        'El ambiente no sale del modelo, sino de la cercanía de cada píxel '
                        f"a lagunas y cauces. {mam.frase(_am)[:1].upper()}{mam.frase(_am)[1:]}. "
                        + ('Cauces: los de tu capa.' if _am['cauces'] == 'capa' else
                           f"Cauces: deducidos del relieve (los que reciben el agua de "
                           f"{_am['cauce_ha']:g} ha o más); son una aproximación.")
                        + ' Lo que se ha tomado por laguna y por cauce se ve en la capa '
                        '**Lagunas y cauces** del mapa.',
                        'The environment does not come from the model but from how close '
                        'each pixel is to lakes and watercourses. '
                        f"{mam.frase(_am, True)[:1].upper()}{mam.frase(_am, True)[1:]}. "
                        + ('Watercourses: those of your layer.' if _am['cauces'] == 'capa' else
                           f"Watercourses: derived from the relief (those receiving the "
                           f"water of {_am['cauce_ha']:g} ha or more); they are an "
                           'approximation.')
                        + ' What was taken as lake and as watercourse shows in the '
                        '**Lakes and watercourses** layer of the map.'))
                    # (las distancias de ahora, de los controles: si no están en
                    # pantalla, no hay nada que comparar)
                    _ahora = {'laguna_capa_m': 'mrf_amb_capa', 'laguna_agua_m': 'mrf_amb_agua',
                              'arroyo_m': 'mrf_amb_arroyo', 'cauce_ha': 'mrf_amb_ha'}
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
                    st.dataframe(pd.DataFrame([{
                        i18n.pick('Pureza', 'Purity'): i18n.tv(c['categoria']),
                        i18n.pick('Código', 'Code'): c['codigo'],
                        _k_ha: c['hectareas'],
                        _k_pb: (100 * c['pixeles'] / _n_bo) if _n_bo else 0.0}
                        for c in _pz['categorias']]), hide_index=True, width='stretch',
                        column_config={_k_ha: st.column_config.NumberColumn(format='%.2f'),
                                       _k_pb: st.column_config.NumberColumn(format='%.1f')})
                    st.caption(i18n.pick(
                        'Estimada con el NDVI de inicio de verano de cada píxel: sin verde '
                        f"{_pz['sin_verde']:.2f}, todo verde {_pz['todo_verde']:.2f}; puro "
                        f"desde el {100 * _pz['puro_desde']:.0f} % de verde. Mixto-agua, si "
                        'el píxel es agua o toca un píxel de agua.',
                        'Estimated from the early-summer NDVI of each pixel: no green '
                        f"{_pz['sin_verde']:.2f}, all green {_pz['todo_verde']:.2f}; pure "
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
                            f'donde llega a {_u_mapa:.3f}, el mapa pone borreguil, del tipo '
                            'más probable. Es la capa para ver dónde hay borreguil.',
                            '**Probability of borreguil** is the sum of those of its '
                            f'types: where it reaches {_u_mapa:.3f}, the map says '
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
                            f"fila {_q['fila'] + 1}, columna {_q['columna'] + 1} · centro en "
                            f"X {_q['x']:.0f}, Y {_q['y']:.0f} ({_meta['crs']}) · "
                            f"lon {_q['lon']:.5f}, lat {_q['lat']:.5f}",
                            f"row {_q['fila'] + 1}, column {_q['columna'] + 1} · centre at "
                            f"X {_q['x']:.0f}, Y {_q['y']:.0f} ({_meta['crs']}) · "
                            f"lon {_q['lon']:.5f}, lat {_q['lat']:.5f}")
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
                                f"**{bp.fmt_num(_q['pmax'])}**.",
                                f"**Clicked pixel:** \"{i18n.tv(_q['categoria'])}\" (code "
                                f"{_q['codigo']}), with probability "
                                f"**{bp.fmt_num(_q['pmax'])}**.")
                                + ('' if _q.get('p_borreguil') is None or _u_mapa is None
                                   else ' ' + i18n.pick(
                                       'Probabilidad de borreguil (la suma de sus tipos): '
                                       f"**{bp.fmt_num(_q['p_borreguil'])}**; el umbral es "
                                       f'{_u_mapa:.3f}.',
                                       'Probability of borreguil (the sum of its types): '
                                       f"**{bp.fmt_num(_q['p_borreguil'])}**; the "
                                       f'threshold is {_u_mapa:.3f}.'))
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
                            st.dataframe(pd.DataFrame([{
                                i18n.pick('Categoría', 'Category'): i18n.tv(p['categoria']),
                                i18n.pick('Código', 'Code'): p['codigo'],
                                _k_pr: p['probabilidad']}
                                for p in sorted(_q['probabilidades'],
                                                key=lambda p: -p['probabilidad'])]),
                                hide_index=True, width='stretch',
                                column_config={_k_pr: st.column_config.NumberColumn(format='%.4f')})
                            st.caption(_donde[:1].upper() + _donde[1:] + '.')

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
                        label=f'⬇ {_nom} ({_mb:.1f} MB)' if _mb >= 0.1 else f'⬇ {_nom}',
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
                    f'el F1 medio, que es **{bp.fmt_num(_f1)}**. Se hace {_nrep} veces; el '
                    '«±» es lo que cambia de una vez a otra.\n'
                    '- **Uso interno** (importancia por impureza). Cuánto ha usado el '
                    'modelo final esa variable para separar las muestras al entrenar. Es '
                    'un reparto: entre todas suman 100 %.',
                    'It is measured in two ways, both when training:\n\n'
                    '- **What the model loses without it** (permutation importance). '
                    f'{_valid[:1].upper() + _valid[1:]} is repeated with the values of that '
                    'variable **shuffled** among the samples, as if it stopped telling '
                    'anything, and the drop in the mean F1, which is '
                    f'**{bp.fmt_num(_f1)}**, is recorded. This is done {_nrep} times; the '
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
                        f"**La que más pesa es `{_1a['variable']}`** "
                        f"({mr.descripcion(_1a['variable'])}): con ella desordenada, el F1 "
                        f"medio baja de {bp.fmt_num(_f1)} a "
                        f"{bp.fmt_num(_f1 - _1a['permutacion'])}.",
                        f"**The one that weighs most is `{_1a['variable']}`** "
                        f"({mr.descripcion(_1a['variable'], True)}): with it shuffled, the "
                        f"mean F1 drops from {bp.fmt_num(_f1)} to "
                        f"{bp.fmt_num(_f1 - _1a['permutacion'])}.")
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
                            'importan. La segunda es la que suele darse en los artículos '
                            '(en R, «MeanDecreaseGini»; la primera equivale a '
                            '«MeanDecreaseAccuracy», aquí con el F1 medio).',
                            '**What the model loses without it** is measured with samples '
                            'the model has not seen (those of the validation): it tells '
                            'which variables it needs to get new places right.\n\n'
                            '**Internal use** is measured with the same samples used for '
                            'training: it gives some weight to every variable the model '
                            'uses, including those that do not help it elsewhere.\n\n'
                            '**Recommended: the first one** to decide which variables '
                            'matter. The second is the one usually reported in papers (in '
                            'R, "MeanDecreaseGini"; the first corresponds to '
                            '"MeanDecreaseAccuracy", here with the mean F1).'))
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
                        'variable': v['variable'],
                        'que': mr.descripcion(v['variable'], _en),
                        'grupo': mr.nombre_grupo(v['grupo'], _en),
                        'valor': v['permutacion'] if _perm else 100 * v['impureza'],
                        'menos': v['permutacion'] - v['desviacion'],
                        'mas': v['permutacion'] + v['desviacion'],
                        'pierde': f"{v['permutacion']:+.4f} ± {v['desviacion']:.4f}",
                        'uso': f"{100 * v['impureza']:.2f} %",
                        'puesto': f"{v['orden']} · {v['orden_impureza']}"} for v in _fil])
                    _tx = (i18n.pick('bajada del F1 medio al desordenar la variable',
                                     'drop in mean F1 when the variable is shuffled')
                           if _perm else i18n.pick('uso interno (% del total)',
                                                   'internal use (% of the total)'))

                    def _x(campo):
                        return alt.X(f'{campo}:Q', title=_tx, axis=alt.Axis(orient='top'))

                    _y = alt.Y('variable:N', sort=[v['variable'] for v in _fil], title=None,
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
                        _k_pj: f"{g['permutacion']:+.4f}\u00a0±\u00a0{g['desviacion']:.4f}",
                        _k_uj: f"{100 * g['impureza']:.2f}\u00a0%"} for g in _lista]
                    ).set_index(_k_cj))

                # ---- todas, una por una (lo mismo que el gráfico, con sus números).
                # Los números, delante: lo que mide cada una es largo y va al final.
                st.markdown(i18n.pick('**Todas las variables**', '**All the variables**'))
                _c4v = st.column_config.NumberColumn(format='%.4f')
                _c2v = st.column_config.NumberColumn(format='%.2f')
                _k_pv = i18n.pick('Bajada del F1 sin ella', 'F1 drop without it')
                _k_ui = i18n.pick('Uso interno (%)', 'Internal use (%)')
                st.dataframe(pd.DataFrame([{
                    'Variable': v['variable'], i18n.pick('Puesto', 'Rank'): v['orden'],
                    _k_pv: v['permutacion'], '±': v['desviacion'],
                    _k_ui: 100 * v['impureza'],
                    i18n.pick('Puesto por uso', 'Rank by use'): v['orden_impureza'],
                    i18n.pick('Qué mide', 'What it measures'):
                        mr.descripcion(v['variable'], _en),
                    i18n.pick('Familia', 'Family'): mr.nombre_familia(v['familia'], _en),
                    i18n.pick('Época u origen', 'Season or source'):
                        mr.nombre_grupo(v['grupo'], _en)}
                    for v in _vs]), hide_index=True, width='stretch',
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
                + (' (VIF)' if _tab.get('vif') else ''))
        if not _mod:
            _e3 = (_falta, i18n.pick('sin entrenar', 'not trained yet'))
        elif _tab is None or _mod['firma'] != mr.firma_tabla(_tab):
            _e3 = (_ojo, i18n.pick('es de una tabla anterior', 'from a previous table'))
        else:
            _e3 = (_hecho, i18n.pick(f"F1 medio {bp.fmt_num(_mod['macro_f1'])}",
                                     f"mean F1 {bp.fmt_num(_mod['macro_f1'])}"))
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
                f"{_mapa['meta']['pixeles_clasificados'] / 100:.1f} ha clasificadas",
                f"{_mapa['meta']['pixeles_clasificados'] / 100:.1f} ha classified"))
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
                f"la primera: {_imp_e['variables'][0]['variable']}",
                f"top: {_imp_e['variables'][0]['variable']}"))
        _icono = {_hecho: ':material/check_circle:', _ojo: ':material/error:',
                  _falta: ':material/radio_button_unchecked:'}
        _pasos = (i18n.pick('1 · Datos', '1 · Data'), i18n.pick('2 · Tabla', '2 · Table'),
                  i18n.pick('3 · Modelo', '3 · Model'), i18n.pick('4 · Mapa', '4 · Map'),
                  i18n.pick('5 · Peso', '5 · Weights'))
        _estado.markdown(' '.join(
            f':{_c}-badge[{_icono[_c]} {_p} · {_txt}]'
            for _p, (_c, _txt) in zip(_pasos, (_e1, _e2, _e3, _e4, _e5))))
