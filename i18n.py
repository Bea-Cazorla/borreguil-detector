"""
i18n.py — Traducción de la interfaz sin tocar los ~200 puntos de llamada.

Idea: se envuelven (monkeypatch) las funciones de Streamlit que muestran texto.
Cuando el idioma activo es 'en', el texto se busca en el diccionario EN; si no
está, se devuelve el original en español (fallback → la app nunca se rompe).

Para los desplegables/radios se inyecta un `format_func` que traduce SOLO lo que
se ve; el valor devuelto por el widget sigue siendo el español original, así que
la lógica de la app (p. ej. `thr_mode.startswith('Automático')`) no cambia.
"""

# Idioma activo del proceso (Streamlit reejecuta el script de arriba abajo en
# cada interacción, así que basta una variable de módulo fijada al principio).
_LANG = 'es'


def set_lang(lang):
    global _LANG
    _LANG = 'en' if str(lang).lower().startswith('en') else 'es'


def get_lang():
    return _LANG


def tr(s):
    """Traduce una cadena si procede; deja intacto lo que no sea texto conocido."""
    if _LANG == 'en' and isinstance(s, str):
        return EN.get(s, s)
    return s


def _wrap_text(fn):
    """Envuelve una función de Streamlit que muestra texto (title/header/markdown/
    caption/write/info/warning/error/success/button/expander/metric/widgets…).
    Traduce el primer argumento (etiqueta o texto) y el `help`/`label`."""
    def inner(*args, **kwargs):
        if args:
            first = args[0]
            if isinstance(first, str):
                args = (tr(first),) + args[1:]
            elif isinstance(first, (list, tuple)):  # st.tabs(['A','B'])
                args = ([tr(x) if isinstance(x, str) else x for x in first],) + args[1:]
        for k in ('help', 'label', 'placeholder'):
            if isinstance(kwargs.get(k), str):
                kwargs[k] = tr(kwargs[k])
        return fn(*args, **kwargs)
    return inner


def _wrap_choice(fn):
    """Como _wrap_text, pero además inyecta `format_func` para traducir las
    opciones visibles sin alterar el valor devuelto (radio/selectbox/multiselect)."""
    text = _wrap_text(fn)

    def inner(*args, **kwargs):
        if 'format_func' not in kwargs:
            kwargs['format_func'] = lambda o: tr(o) if isinstance(o, str) else o
        return text(*args, **kwargs)
    return inner


def install(st):
    """Parchea las funciones de texto de Streamlit. Idempotente por ejecución."""
    if getattr(st, '_i18n_installed', False):
        return
    text_fns = ['title', 'header', 'subheader', 'markdown', 'caption', 'write',
                'info', 'warning', 'error', 'success', 'button', 'download_button',
                'form_submit_button', 'metric', 'text_input', 'number_input',
                'slider', 'checkbox', 'file_uploader', 'expander', 'tabs', 'toggle']
    choice_fns = ['radio', 'selectbox', 'multiselect']
    for name in text_fns:
        if hasattr(st, name):
            setattr(st, name, _wrap_text(getattr(st, name)))
    for name in choice_fns:
        if hasattr(st, name):
            setattr(st, name, _wrap_choice(getattr(st, name)))
    st._i18n_installed = True


# ============================================================
# Diccionario español → inglés de la interfaz (cadenas estáticas).
# Los mensajes con valores dinámicos (f-strings) se traducen aparte, en app.py.
# ============================================================
EN = {
    "🌿 Borreguil / Wet Meadow Pipeline": "🌿 Borreguil / Wet Meadow Pipeline",
    "Identificación automática de borreguiles en zonas de montaña — ESRI + OSM + Copernicus DEM + Sentinel-2 (MPC) + Random Forest. [Documentación metodológica](Metodologia_borreguiles.docx)":
        "Automatic detection of borreguiles (high-mountain wet meadows) — ESRI + OSM + Copernicus DEM + Sentinel-2 (MPC) + Random Forest. [Methodology documentation](Metodologia_borreguiles.docx)",
    "🗺️  Análisis": "🗺️  Analysis",
    "📖  Documentación y recomendaciones": "📖  Documentation & recommendations",
    "## 📖 Documentación y recomendaciones": "## 📖 Documentation & recommendations",
    "Esta herramienta identifica **borreguiles** (cervunales higroturbosos / *mountain wet meadows*) combinando teledetección multi-fuente con un clasificador Random Forest, validado de forma **honesta** (GroupKFold por cuenca: entrena en unas cuencas y evalúa en otras nunca vistas).":
        "This tool identifies **borreguiles** (high-mountain peaty wet meadows / *mountain wet meadows*) by combining multi-source remote sensing with a Random Forest classifier, validated **honestly** (basin GroupKFold: trains on some basins and evaluates on others never seen).",
    "### 🧭 ¿Qué modelo elegir?": "### 🧭 Which model to choose?",
    "Todos los modelos parten de las **mismas 50 variables base** (imagen aérea + topografía + Sentinel-2 + Sentinel-1). Las variantes añaden una fuente de **muy alta resolución** que sube el AUC. **La app extrae automáticamente la fuente extra** que el modelo elegido necesite — tú solo eliges el modelo.":
        "All models start from the **same 50 base variables** (aerial imagery + topography + Sentinel-2 + Sentinel-1). The variants add a **very-high-resolution** source that raises the AUC. **The app automatically extracts the extra source** the chosen model needs — you only pick the model.",
    "- **El mejor resultado en Sierra Nevada es el modelo `_cir`, y es gratis** (0,914). Úsalo siempre que trabajes en Andalucía, Cataluña o Canarias.\n- **Fuera de España** (o en CCAA sin ortofoto infrarroja) usa `_planet`: cobertura mundial a 3 m, pero consume cuota de tu suscripción Planet.\n- El modelo `_combo` (66) es el más alto en términos absolutos, pero la mejora sobre `_cir` (+0,004) está dentro del ruido: **no compensa** la doble dependencia salvo casos límite.":
        "- **The best result in Sierra Nevada is the `_cir` model, and it is free** (0.914). Use it whenever you work in Andalusia, Catalonia or the Canary Islands.\n- **Outside Spain** (or in regions without infrared orthophoto) use `_planet`: worldwide coverage at 3 m, but it consumes quota from your Planet subscription.\n- The `_combo` model (66) is the highest in absolute terms, but its gain over `_cir` (+0.004) is within the noise: **not worth** the double dependency except in edge cases.",
    "### 🗺️ Cobertura de la ortofoto infrarroja (modelo `_cir`, gratis)":
        "### 🗺️ Infrared orthophoto coverage (`_cir` model, free)",
    "El IGN tiene PNOA Falso Color Infrarrojo a 0,25 m para **toda España**, pero solo como descarga COG por hojas MTN25 (sin WMS) → conectarlo cubriría también Ordesa, Picos y Guadarrama. El modelo `_cir` se calibró con Andalucía; en otras CCAA las variables IR son comparables pero no idénticas → para máxima precisión, entrena con verdad-terreno local.":
        "The IGN provides PNOA False-Colour Infrared at 0.25 m for **all of Spain**, but only as per-sheet MTN25 COG downloads (no WMS) → wiring it in would also cover Ordesa, Picos and Guadarrama. The `_cir` model was calibrated on Andalusia; in other regions the IR variables are comparable but not identical → for maximum accuracy, train with local ground truth.",
    "### 🔑 Configurar el acceso a Planet (PlanetScope 3 m)":
        "### 🔑 Set up Planet access (PlanetScope 3 m)",
    "**Uso local (tu ordenador):** crea el archivo `borreguil_app/.streamlit/secrets.toml` con este contenido:":
        "**Local use (your computer):** create the file `borreguil_app/.streamlit/secrets.toml` with this content:",
    "Ese archivo **no se sube a git** (está en `.gitignore`), así que tu clave queda privada. Reinicia la app y el acceso a Planet se activa solo.\n\n**Deploy en Streamlit Cloud:** no subas el archivo. En el panel de tu app entra en **Settings → Secrets** y pega ahí la misma línea `PL_API_KEY = \"…\"`.":
        "That file **is not committed to git** (it is in `.gitignore`), so your key stays private. Restart the app and Planet access turns on by itself.\n\n**Deploy on Streamlit Cloud:** do not upload the file. In your app panel go to **Settings → Secrets** and paste the same line `PL_API_KEY = \"…\"` there.",
    "⚠ La clave da acceso a tu cuenta Planet de pago (consume cuota de km²). No la compartas en texto plano ni la subas al repositorio. Si crees que se ha expuesto, rótala en planet.com.":
        "⚠ The key grants access to your paid Planet account (consumes km² quota). Do not share it in plain text or upload it to the repository. If you think it has been exposed, rotate it at planet.com.",
    "### 🛰️ Fuentes de datos y variables del modelo":
        "### 🛰️ Data sources and model variables",
    "Las dos últimas filas son las que usan los modelos `_cir` / `_planet`. La variable más informativa de ambas es la **textura/contraste del NDVI a alta resolución** — justo el borde nítido del borreguil contra el sustrato pétreo, que Sentinel-2 (10 m) promedia y pierde.":
        "The last two rows are those used by the `_cir` / `_planet` models. The most informative variable of both is the **high-resolution NDVI texture/contrast** — precisely the sharp edge of the borreguil against the rocky substrate, which Sentinel-2 (10 m) averages out and loses.",
    "### 🔑 Qué variables pesan más y cómo leer las gráficas":
        "### 🔑 Which variables matter most and how to read the charts",
    "El modelo es un **Random Forest**: cada variable recibe una **importancia** (0–100 %) según cuánto ayuda a separar *borreguil* de *no-borreguil*. En el panel **📈 Perfil espectral** de cada punto (pestaña **Tabla**) verás este mismo ranking arriba del todo: **léelo de arriba abajo** e interpreta primero las variables con más peso. A la derecha de cada barra está el **rango típico de los borreguiles** (p25–p75): si el valor del punto cae dentro de ese rango, se parece a un borreguil en esa variable.":
        "The model is a **Random Forest**: each variable receives an **importance** (0–100 %) according to how much it helps separate *borreguil* from *non-borreguil*. In each point's **📈 Spectral profile** panel (the **Table** tab) you will see this same ranking at the very top: **read it top to bottom** and interpret the highest-weight variables first. To the right of each bar is the **typical borreguil range** (p25–p75): if the point's value falls within that range, it resembles a borreguil in that variable.",
    "### ▶️ Flujo de trabajo": "### ▶️ Workflow",
    "1. **Define el área** (vector, WDPA ID o nombre) o sube tus puntos candidatos.\n2. Elige el **modelo** según la tabla de arriba y la **fuente de imágenes** (PNOA en España, ESRI fuera).\n3. Pulsa **Ejecutar pipeline**. La app descarga cada fuente, extrae las variables y predice.\n4. Revisa el **mapa** y la **tabla** de resultados; ajusta el umbral; descarga Excel/CSV/GeoJSON/Word.\n5. *(Opcional)* marca borreguiles verificados en campo y **reentrena** en tu zona (presence-only, mínimo ~30 positivos) para un modelo propio.":
        "1. **Define the area** (vector, WDPA ID or name) or upload your candidate points.\n2. Choose the **model** according to the table above and the **image source** (PNOA in Spain, ESRI elsewhere).\n3. Click **Run pipeline**. The app downloads each source, extracts the variables and predicts.\n4. Review the results **map** and **table**; adjust the threshold; download Excel/CSV/GeoJSON/Word.\n5. *(Optional)* mark field-verified borreguiles and **retrain** on your area (presence-only, minimum ~30 positives) for your own model.",
    "**1 · Perfil estacional** — inicio→fin de verano frente a la banda de borreguiles (un panel por índice)":
        "**1 · Seasonal profile** — start→end of summer against the borreguil band (one panel per index)",
    "**2 · Posición vs. distribución de borreguiles** — percentil de cada índice (fin de verano)":
        "**2 · Position vs. borreguil distribution** — percentile of each index (end of summer)",
    "**3 · Variabilidad multianual** — media (●) y rango mín–máx (│) entre escenas vs. banda de borreguiles":
        "**3 · Multi-year variability** — mean (●) and min–max range (│) across scenes vs. borreguil band",
    "Inputs": "Inputs",
    "Origen de los puntos": "Point source",
    "📁  Subir archivo (KML/Shp/GeoJSON)": "📁  Upload file (KML/Shp/GeoJSON)",
    "🎲  Generar aleatoriamente en el área": "🎲  Generate randomly within the area",
    "Opcional: verdad-terreno (presencias y/o ausencias)":
        "Optional: ground truth (presences and/or absences)",
    "Usar verdad-terreno solo para entrenar (no mostrarla en el mapa)":
        "Use ground truth only for training (do not show it on the map)",
    "El modelo aprende de tus puntos de campo, pero estos NO se añaden al mapa, la tabla ni las descargas: solo verás tus puntos candidatos. Útil cuando subes muchos borreguiles verificados y no quieres que saturen el resultado. Los candidatos que coincidan (<20 m) con un punto de campo sí se marcan como verificados y se muestran.":
        "The model learns from your field points, but they are NOT added to the map, table or downloads: you will only see your candidate points. Useful when you upload many verified borreguiles and do not want them to swamp the result. Candidates that match (<20 m) a field point are marked as verified and shown.",
    "🌊 Capa de lagunas (opcional, mejora el ambiente «laguna»)":
        "🌊 Lakes layer (optional, improves the «lake» environment)",
    "Puntos de lagunas conocidas para la clasificación jerárquica. Si no subes ninguna, se usa la capa incluida de Sierra Nevada. Un borreguil a < 120 m de una laguna se clasifica como ambiente «laguna».":
        "Known lake points for the hierarchical classification. If you upload none, the bundled Sierra Nevada layer is used. A borreguil within < 120 m of a lake is classified as «lake» environment.",
    "Área de estudio": "Study area",
    "Define el alcance espacial. Si no se aporta, se usa la bbox de los puntos.":
        "Defines the spatial extent. If not provided, the bounding box of the points is used.",
    "Modo": "Mode",
    "🗺  Vector (subir KML/Shp/GeoJSON)": "🗺  Vector (upload KML/Shp/GeoJSON)",
    "🆔  WDPA ID (protectedplanet.net)": "🆔  WDPA ID (protectedplanet.net)",
    "🔤  Nombre del área protegida": "🔤  Protected area name",
    "⨯  Sin área (bbox de los puntos)": "⨯  No area (bounding box of the points)",
    "Fuente de datos satelitales": "Satellite data source",
    "Backend": "Backend",
    "🌍  Google Earth Engine (requiere cuenta) — más rápido":
        "🌍  Google Earth Engine (requires account) — faster",
    "🛰  Microsoft Planetary Computer (sin cuenta)":
        "🛰  Microsoft Planetary Computer (no account)",
    "GEE calcula los índices en el servidor (mucho más rápido, sobre todo con el modelo de 45 variables). Si GEE no está disponible, la app usa MPC automáticamente.":
        "GEE computes the indices on the server (much faster, especially with the 45-variable model). If GEE is unavailable, the app falls back to MPC automatically.",
    "Modelo de partida": "Starting model",
    "🧭 ¿Qué modelo elegir según tu zona? Mira la pestaña **📖 Documentación** (arriba) para la guía completa y las recomendaciones por parque.":
        "🧭 Which model to choose for your area? See the **📖 Documentation** tab (above) for the full guide and per-park recommendations.",
    "Opciones": "Options",
    "Años Sentinel-2 (admite rangos, p. ej. 2017-2025)":
        "Sentinel-2 years (accepts ranges, e.g. 2017-2025)",
    "Umbral RF para BORREGUIL": "RF threshold for BORREGUIL",
    "Automático · Otsu": "Automatic · Otsu",
    "Manual (deslizador)": "Manual (slider)",
    "Automático · verdad-terreno": "Automatic · ground truth",
    "**Otsu (recomendado, por defecto)**: se adapta a cada ejecución. Al aplicar el modelo a una zona nueva las probabilidades se recalibran, así que un umbral manual fijo puede dejar casi todo fuera; Otsu evita ese problema.\n\n**Manual**: eliges el valor con el deslizador.\n\n**Otsu**: calcula el corte automáticamente a partir de la *forma* de la distribución de probabilidades RF (método de Otsu, el clásico para binarizar imágenes). Encuentra el valor que mejor separa los dos grupos —probable no-borreguil vs. borreguil— minimizando la varianza dentro de cada grupo. **No necesita verdad-terreno.**\n\n**Verdad-terreno**: usa tus puntos de campo. Con presencias y ausencias, el corte que maximiza el índice de Youden (sensibilidad + especificidad − 1). Solo con presencias, el corte que captura el 90% de los borreguiles conocidos. Si no hay suficiente verdad-terreno, recurre a Otsu.":
        "**Otsu (recommended, default)**: adapts to each run. When the model is applied to a new area the probabilities recalibrate, so a fixed manual threshold may leave almost everything out; Otsu avoids that problem.\n\n**Manual**: you pick the value with the slider.\n\n**Otsu**: computes the cut automatically from the *shape* of the RF probability distribution (Otsu's method, the classic for binarising images). It finds the value that best separates the two groups —likely non-borreguil vs. borreguil— by minimising the variance within each group. **No ground truth needed.**\n\n**Ground truth**: uses your field points. With presences and absences, the cut that maximises Youden's index (sensitivity + specificity − 1). With presences only, the cut that captures 90% of the known borreguiles. If there is not enough ground truth, it falls back to Otsu.",
    "Valor del umbral": "Threshold value",
    "Fuente de imágenes": "Image source",
    "🌐  ESRI World Imagery (global, ~0,5 m)": "🌐  ESRI World Imagery (global, ~0.5 m)",
    "🇪🇸  PNOA IGN + autonómica (España, 0,25 m)":
        "🇪🇸  PNOA IGN + regional (Spain, 0.25 m)",
    "PNOA IGN usa ortofoto autonómica cuando está disponible (Andalucía 2022, Aragón 2024, Cataluña vigente, Cantabria 2023, CyL 2020, Canarias) y cae en el PNOA-MA nacional para el resto de España. Fuera de España usa ESRI.":
        "PNOA IGN uses the regional orthophoto when available (Andalusia 2022, Aragón 2024, Catalonia current, Cantabria 2023, Castile-León 2020, Canary Islands) and falls back to the national PNOA-MA for the rest of Spain. Outside Spain it uses ESRI.",
    "⚡ Modo rápido (vista previa)": "⚡ Fast mode (preview)",
    "Salta las descargas lentas que se hacen punto a punto en tu PC: imágenes ESRI/PNOA, Sentinel-1 (SAR) y CIR/Planet. Con backend GEE, el resto (topografía y Sentinel-2) se calcula en la nube en segundos. Da resultados APROXIMADOS (sin texturas ni SAR): ideal para iterar rápido y luego afinar desmarcando esta casilla.":
        "Skips the slow per-point downloads done on your PC: ESRI/PNOA imagery, Sentinel-1 (SAR) and CIR/Planet. With the GEE backend, the rest (topography and Sentinel-2) is computed in the cloud in seconds. Gives APPROXIMATE results (no textures or SAR): ideal to iterate quickly and then refine by unchecking this box.",
    "Saltar Sentinel-2 (más rápido, sin NDVI)": "Skip Sentinel-2 (faster, no NDVI)",
    "Usar OSM (hidrografía/infraestructuras) — Overpass puede fallar":
        "Use OSM (hydrography/infrastructure) — Overpass may fail",
    "PlanetScope 3 m (8 bandas SR) — requiere PL_API_KEY":
        "PlanetScope 3 m (8 SR bands) — requires PL_API_KEY",
    "Pide imágenes PlanetScope vía Orders API (clip al área de los puntos). CONSUME CUOTA de tu suscripción Planet (km² pedidos) y tarda 10-30 min extra. Útil con el modelo variante rf_*_planet.joblib (60 variables). Configura PL_API_KEY como variable de entorno o en st.secrets.":
        "Requests PlanetScope imagery via the Orders API (clipped to the points' area). CONSUMES QUOTA from your Planet subscription (km² ordered) and takes 10-30 min extra. Useful with the rf_*_planet.joblib variant model (60 variables). Set PL_API_KEY as an environment variable or in st.secrets.",
    "Acción": "Action",
    "▶ Ejecutar pipeline": "▶ Run pipeline",
    "**✅ Con WMS infrarrojo (0,25 m)**\n\n- **Andalucía** → Sierra Nevada\n- **Cataluña** → Aigüestortes, Pirineos\n- **Canarias** → Teide, Garajonay, Caldera de Taburiente":
        "**✅ With infrared WMS (0.25 m)**\n\n- **Andalusia** → Sierra Nevada\n- **Catalonia** → Aigüestortes, Pyrenees\n- **Canary Islands** → Teide, Garajonay, Caldera de Taburiente",
    "**❌ Sin WMS infrarrojo** (usa `_planet` o las 50 base)\n\n- **Aragón** → Ordesa y Monte Perdido\n- **Cantabria / Castilla y León** → Picos de Europa\n- **Madrid / CyL** → Sierra de Guadarrama":
        "**❌ Without infrared WMS** (use `_planet` or the 50 base)\n\n- **Aragón** → Ordesa y Monte Perdido\n- **Cantabria / Castile-León** → Picos de Europa\n- **Madrid / Castile-León** → Sierra de Guadarrama",
    "✓ Clave Planet **detectada**. El modelo `_planet` (y `_combo`) extraerá PlanetScope automáticamente al ejecutar el pipeline.":
        "✓ Planet key **detected**. The `_planet` (and `_combo`) model will extract PlanetScope automatically when running the pipeline.",
    "Aún **no hay clave Planet** configurada. Sigue los pasos de abajo.":
        "There is **no Planet key** configured yet. Follow the steps below.",
    "Selecciona un modelo con `ref_stats` (reentrenado con `train_v5.py`) para ver el ranking.":
        "Select a model with `ref_stats` (retrained with `train_v5.py`) to see the ranking.",
    "🔍 **Diagnóstico de extracción** — alguna fuente quedó casi vacía (filas 🔴). Las variables que falten se rellenan con la mediana, así que la predicción puede ser poco fiable. Causas y solución abajo.":
        "🔍 **Extraction diagnostics** — some source came back almost empty (🔴 rows). Missing variables are filled with the median, so the prediction may be unreliable. Causes and fix below.",
    "El modelo seleccionado no trae valores de referencia (`ref_stats`). Reentrena/guarda con `train_v5.py` para activar la comparación. Se muestran solo los valores del punto.":
        "The selected model has no reference values (`ref_stats`). Retrain/save with `train_v5.py` to enable the comparison. Only the point's values are shown.",
    "**🔑 Variables más decisivas del modelo** (guía jerárquica) — de arriba (más peso) a abajo; a la derecha, el **rango típico de borreguil** (p25–p75).":
        "**🔑 Model's most decisive variables** (hierarchical guide) — from top (highest weight) to bottom; on the right, the **typical borreguil range** (p25–p75).",
    "Lee los gráficos de abajo en este orden de importancia. Altitud, pendiente, SAR y textura de imagen pesan en el modelo pero no son índices espectrales, por eso no aparecen en los perfiles estacionales.":
        "Read the charts below in this order of importance. Elevation, slope, SAR and image texture matter in the model but are not spectral indices, which is why they do not appear in the seasonal profiles.",
    "Sin datos estacionales para este punto.": "No seasonal data for this point.",
    "Sin índices clave disponibles para este punto.": "No key indices available for this point.",
    "Sin índices multianuales disponibles para este punto.":
        "No multi-year indices available for this point.",
    "Puntos candidatos": "Candidate points",
    "Requiere definir el área de estudio (panel siguiente).":
        "Requires defining the study area (next panel).",
    "Nº puntos a generar": "Nº of points to generate",
    "Estrategia": "Strategy",
    "stratified_elev: reparto equilibrado por bandas de altitud. uniform: aleatorio puro. poisson_disc: separación mínima entre puntos.":
        "stratified_elev: balanced distribution across elevation bands. uniform: pure random. poisson_disc: minimum spacing between points.",
    "Semilla": "Seed",
    "ℹ ¿Cómo estructurar la verdad-terreno?": "ℹ How to structure the ground truth?",
    "**Solo presencias** (lo más simple): un KML/GeoJSON/Shapefile con puntos de borreguiles confirmados, **sin ningún atributo**. Todos se toman como borreguil; las ausencias se generan solas (pseudo-ausencias lejanas).\n\n**Presencias + ausencias** (recomendado si las tienes): añade un **atributo de clase** a cada punto. La app reconoce, sin distinguir mayúsculas:\n- Atributo llamado `Borreguil`, `presencia`, `clase`, `tipo` o `label`.\n- Valor de **presencia**: `si` · `1` · `presencia` · `borreguil`.\n- Valor de **ausencia**: `no` · `0` · `ausencia` · `no_borreguil`.\n\nEjemplo GeoJSON (un punto presencia y uno ausencia):":
        "**Presences only** (simplest): a KML/GeoJSON/Shapefile with confirmed borreguil points, **without any attribute**. All are taken as borreguil; absences are generated automatically (distant pseudo-absences).\n\n**Presences + absences** (recommended if you have them): add a **class attribute** to each point. The app recognises, case-insensitively:\n- An attribute named `Borreguil`, `presencia`, `clase`, `tipo` or `label`.\n- **Presence** value: `si` · `1` · `presencia` · `borreguil`.\n- **Absence** value: `no` · `0` · `ausencia` · `no_borreguil`.\n\nGeoJSON example (one presence point and one absence):",
    "En KML, añade un `<SimpleData name=\"Borreguil\">no</SimpleData>` por Placemark. Un punto sin atributo de clase se toma como presencia.":
        "In KML, add a `<SimpleData name=\"Borreguil\">no</SimpleData>` per Placemark. A point without a class attribute is taken as a presence.",
    "Polígono del área de estudio": "Study-area polygon",
    "Un único **KML / GeoJSON / GeoPackage**, o un **shapefile** (selecciona .shp + .shx + .dbf + .prj juntos, o sube un .zip).":
        "A single **KML / GeoJSON / GeoPackage**, or a **shapefile** (select .shp + .shx + .dbf + .prj together, or upload a .zip).",
    "Proyecto Google Earth Engine": "Google Earth Engine project",
    "Nombre del proyecto de Google Cloud con la Earth Engine API habilitada.":
        "Name of the Google Cloud project with the Earth Engine API enabled.",
    "El cómputo S2 + topografía se hace en los servidores de Google (rápido, sin descargas). No se usa MPC.":
        "The S2 + topography computation runs on Google's servers (fast, no downloads). MPC is not used.",
    "Modelo base Random Forest": "Base Random Forest model",
    "Modelos .joblib de la carpeta de la app. Se usa para predecir cuando no hay suficiente verdad-terreno local. Tras entrenar en tu zona puedes guardarlo aquí (panel de resultados) para reutilizarlo.":
        ".joblib models from the app folder. Used to predict when there is not enough local ground truth. After training on your area you can save it here (results panel) to reuse it.",
    "No hay modelos .joblib en la carpeta.": "There are no .joblib models in the folder.",
    "⚡ Vista previa: sin imágenes, sin Sentinel-1 ni CIR/Planet. Resultados aproximados.":
        "⚡ Preview: no imagery, no Sentinel-1 or CIR/Planet. Approximate results.",
    "Saltar imágenes (más rápido, sin texturas)": "Skip imagery (faster, no textures)",
    "PlanetScope desactivado: no hay PL_API_KEY en el entorno/secrets.":
        "PlanetScope disabled: no PL_API_KEY in the environment/secrets.",
    "👈 Configura los inputs en el panel lateral y pulsa **Ejecutar pipeline**. Puedes subir tus puntos o generarlos aleatoriamente dentro del área.":
        "👈 Configure the inputs in the sidebar and click **Run pipeline**. You can upload your points or generate them randomly within the area.",
    "### Verdad-terreno: solo coordenadas (presence-only)":
        "### Ground truth: coordinates only (presence-only)",
    "El KML de borreguiles verificados **no necesita ningún atributo**: cada punto se asume borreguil. La clase negativa se genera sola.":
        "The verified-borreguiles KML **needs no attribute**: every point is assumed to be a borreguil. The negative class is generated automatically.",
    "Resultados": "Results",
    "Total": "Total",
    "Mapa": "Map",
    "Tabla": "Table",
    "Histograma RF": "RF histogram",
    "Descargas": "Downloads",
    "🔑 Variables": "🔑 Variables",
    "📊 Distribuciones": "📊 Distributions",
    "Sugerencia: reintenta (caídas temporales de los servicios), reduce el nº de puntos, o cambia de backend (MPC ↔ GEE) en el panel lateral.":
        "Tip: retry (temporary service outages), reduce the number of points, or switch backend (MPC ↔ GEE) in the sidebar.",
    "Distancia mínima entre puntos (m)": "Minimum distance between points (m)",
    "Solo en poisson_disc: ningún par de puntos quedará más cerca de esta distancia. Los puntos se ajustan además al centro del píxel Sentinel-2 (10 m).":
        "Only in poisson_disc: no pair of points will be closer than this distance. Points are also snapped to the centre of the Sentinel-2 pixel (10 m).",
    "Elev. mín (m)": "Min elev. (m)",
    "Opcional: filtra por SRTM 30m": "Optional: filter by SRTM 30m",
    "Elev. máx (m)": "Max elev. (m)",
    "Opcional": "Optional",
    "WDPA ID": "WDPA ID",
    "ID numérico de protectedplanet.net.": "Numeric ID from protectedplanet.net.",
    "Token Protected Planet (opcional)": "Protected Planet token (optional)",
    "Gratuito en protectedplanet.net/api. Necesario para IDs fuera del cache, salvo que uses el backend Google Earth Engine (resuelve sin token).":
        "Free at protectedplanet.net/api. Needed for IDs outside the cache, unless you use the Google Earth Engine backend (resolves without a token).",
    "Ejemplos: `555512151` Sierra Nevada · `4514` Picos de Europa · `11` Yellowstone. IDs arbitrarios requieren token o backend GEE.":
        "Examples: `555512151` Sierra Nevada · `4514` Picos de Europa · `11` Yellowstone. Arbitrary IDs require a token or the GEE backend.",
    "🔑 Autenticar GEE": "🔑 Authenticate GEE",
    "✓ Probar conexión": "✓ Test connection",
    "### 📍 Sin verdad-terreno": "### 📍 Without ground truth",
    "- Se usa el modelo **preentrenado en Sierra Nevada**\n- AUC GroupKFold: **0,86**\n- Funciona en zonas comparables; precaución en climas distintos":
        "- The model **pretrained on Sierra Nevada** is used\n- GroupKFold AUC: **0.86**\n- Works in comparable areas; use caution in different climates",
    "### 🎯 Con verdad-terreno": "### 🎯 With ground truth",
    "- **Presencias** (borreguiles) y, opcional, **ausencias**\n- Sin ausencias: se generan pseudo-ausencias solas\n- Atributo de clase `Borreguil`=si/no (ver panel lateral)\n- Mínimo: 30 positivos · Recomendado: 80-150":
        "- **Presences** (borreguiles) and, optionally, **absences**\n- Without absences: pseudo-absences are generated automatically\n- Class attribute `Borreguil`=si/no (see sidebar)\n- Minimum: 30 positives · Recommended: 80-150",
    "### 🛰 Backend": "### 🛰 Backend",
    "- **MPC**: sin cuenta, descarga rásters (más lento)\n- **GEE**: requiere cuenta, cómputo en la nube (rápido)\n- Sentinel-2 por defecto **2017-2025**":
        "- **MPC**: no account, downloads rasters (slower)\n- **GEE**: requires account, cloud computation (fast)\n- Sentinel-2 by default **2017-2025**",
    "🔁 Evaluación recursiva (auto-entrenamiento)": "🔁 Recursive evaluation (self-training)",
    "En cada iteración, los puntos de **posible borreguil** (probabilidad ≥ umbral de promoción) se convierten en verdad-terreno y se **recalibra** el Random Forest para volver a predecir sobre el resto. Útil para descubrir borreguiles no inventariados de forma progresiva.":
        "At each iteration, the **possible borreguil** points (probability ≥ promotion threshold) become ground truth and the Random Forest is **recalibrated** to predict again over the rest. Useful for progressively discovering uninventoried borreguiles.",
    "### 💾 Guardar el modelo entrenado": "### 💾 Save the trained model",
    "Mapa interactivo. Haz click en un marcador para ver sus detalles y poder **quitarlo** del análisis. Al seleccionar filas en la pestaña **Tabla**, el mapa se centra en el **último punto seleccionado**. La **leyenda de colores** está abajo a la izquierda.":
        "Interactive map. Click a marker to see its details and be able to **remove it** from the analysis. When selecting rows in the **Table** tab, the map centres on the **last selected point**. The **colour legend** is at the bottom left.",
    "Capas activables (control ▤ arriba a la derecha): **Píxel S2 (10 m)**, **ventana de muestreo 10 m** (S2 + Sentinel-1) y **20 m** (S2). Cambia a **Google Satélite** para más zoom. Sentinel-1 RTC comparte la rejilla de 10 m con Sentinel-2.":
        "Toggleable layers (▤ control top right): **S2 pixel (10 m)**, **10 m sampling window** (S2 + Sentinel-1) and **20 m** (S2). Switch to **Google Satellite** for more zoom. Sentinel-1 RTC shares the 10 m grid with Sentinel-2.",
    "Selecciona una o varias filas (casilla izquierda). El **último punto seleccionado** centra el mapa y muestra su imagen, sin perder los anteriores. Con varias seleccionadas puedes **marcarlas como borreguil verificado** y reentrenar.":
        "Select one or several rows (left checkbox). The **last selected point** centres the map and shows its image, without losing the previous ones. With several selected you can **mark them as verified borreguil** and retrain.",
    "Resultados generados (reflejan la última iteración):":
        "Generated results (reflect the last iteration):",
    "**Variables más decisivas del modelo Random Forest** usado en esta estimación — de arriba (más peso) a abajo. A la derecha, el **rango típico de borreguil** (p25–p75) cuando el modelo trae valores de referencia.":
        "**Most decisive variables of the Random Forest model** used in this estimation — from top (highest weight) to bottom. On the right, the **typical borreguil range** (p25–p75) when the model provides reference values.",
    "**Dispersión Altitud × NDVI** — separación entre clases. Pasa el ratón por un punto para ver su ID y probabilidad RF.":
        "**Elevation × NDVI scatter** — separation between classes. Hover over a point to see its ID and RF probability.",
    "**Recuento de puntos por categoría de decisión.**": "**Point count by decision category.**",
    "Nombre del área": "Area name",
    "Búsqueda en Nominatim (OSM). Cuanto más específico, mejor (incluir \"Parque Nacional/Natural\" suele ayudar).":
        "Nominatim (OSM) search. The more specific, the better (including \"Parque Nacional/Natural\" usually helps).",
    "🇪🇸 Usa **PNOA Falso Color IR 0,25 m** (gratis). Cobertura WMS IR en Andalucía, Cataluña y Canarias; se extrae automáticamente (en otras zonas usa las 50 base).":
        "🇪🇸 Uses **PNOA False-Colour IR 0.25 m** (free). IR WMS coverage in Andalusia, Catalonia and the Canary Islands; extracted automatically (elsewhere it uses the 50 base).",
    "Modelo universal (50 variables): no requiere fuentes extra ni claves.":
        "Universal model (50 variables): requires no extra sources or keys.",
    "Para generar puntos hace falta un área de estudio válida.":
        "A valid study area is required to generate points.",
    "El modelo elegido usa PlanetScope 3 m pero no hay PL_API_KEY: esas 10 variables irán vacías y la predicción será menos precisa.":
        "The chosen model uses PlanetScope 3 m but there is no PL_API_KEY: those 10 variables will be empty and the prediction will be less accurate.",
    "Umbral de promoción a borreguil (probabilidad RF)":
        "Promotion-to-borreguil threshold (RF probability)",
    "Sólo se promueven puntos muy probables, para limitar la propagación de errores.":
        "Only highly probable points are promoted, to limit error propagation.",
    "¿Cómo usar los posibles borreguiles?": "How to use the possible borreguiles?",
    "Añadir (acumular a la verdad-terreno)": "Add (accumulate to the ground truth)",
    "Reemplazar (refrescar desde el modelo actual)": "Replace (refresh from the current model)",
    "Añadir: el conjunto de positivos crece en cada iteración.\nReemplazar: se recalculan los pseudo-positivos cada vez desde el último modelo, manteniendo siempre tus puntos verificados.":
        "Add: the positive set grows at each iteration.\nReplace: pseudo-positives are recomputed each time from the latest model, always keeping your verified points.",
    "Aún no hay un modelo en memoria. Ejecuta el pipeline; para un modelo propio de tu zona, sube borreguiles verificados o marca filas en la tabla como borreguil y reentrena.":
        "There is no model in memory yet. Run the pipeline; for your own area model, upload verified borreguiles or mark table rows as borreguil and retrain.",
    "Nombre del área protegida (será el nombre del fichero)":
        "Protected area name (will be the file name)",
    "Usa el nombre oficial del espacio protegido. Se guarda como rf_<nombre>.joblib en la carpeta de la app.":
        "Use the official name of the protected area. It is saved as rf_<name>.joblib in the app folder.",
    "📤 Cómo publicar el modelo para todos los usuarios online":
        "📤 How to publish the model for all online users",
    "Distribución de probabilidad Random Forest (nº de puntos por intervalo de probabilidad):":
        "Random Forest probability distribution (nº of points per probability interval):",
    "Sin probabilidades RF disponibles.": "No RF probabilities available.",
    "El modelo actual no expone importancias de variables (reentrena o usa un modelo guardado con `train_v5.py`).":
        "The current model does not expose variable importances (retrain or use a model saved with `train_v5.py`).",
    "Reglas sobre las variables ya calculadas (cercanía a lagunas, TWI, pendiente, NDMI, NDWI, roca/agua…). Arroyo se detecta por topografía (TWI alto o pendiente baja) cuando no hay capa OSM. La **pureza es relativa** a esta población: marca como mixto-agua/mixto-roca el ~15% con más firma de agua (NDWI) o roca (albedo/NDVI bajo), porque a 10 m no se mide la fracción sub-píxel absoluta (para eso harían falta PlanetScope o CIR). Umbrales en `borreguil_pipeline.py` (HIER_THRESH).":
        "Rules over the already-computed variables (proximity to lakes, TWI, slope, NDMI, NDWI, rock/water…). Stream is detected by topography (high TWI or low slope) when there is no OSM layer. **Purity is relative** to this population: it marks as mixed-water/mixed-rock the ~15% with the strongest water (NDWI) or rock (albedo/low NDVI) signature, because at 10 m the absolute sub-pixel fraction is not measured (that would need PlanetScope or CIR). Thresholds in `borreguil_pipeline.py` (HIER_THRESH).",
    "No hay variables numéricas suficientes para los boxplots.":
        "Not enough numeric variables for the boxplots.",
    "Autenticación lanzada (revisa el navegador).": "Authentication launched (check your browser).",
    "🌍 Usa **PlanetScope 3 m**. Key detectada; se extrae automáticamente (consume cuota Planet).":
        "🌍 Uses **PlanetScope 3 m**. Key detected; extracted automatically (consumes Planet quota).",
    "🌍 Este modelo necesita **PlanetScope 3 m**, pero no hay `PL_API_KEY`. Sin ella, esas variables van vacías (predicción degradada). Configúrala o elige otro modelo.":
        "🌍 This model needs **PlanetScope 3 m**, but there is no `PL_API_KEY`. Without it, those variables are empty (degraded prediction). Set it up or choose another model.",
    "✓ **Convergencia**: la última iteración no encontró nuevos borreguiles. Puedes detener el proceso.":
        "✓ **Convergence**: the last iteration found no new borreguiles. You can stop the process.",
    "El modelo en memoria es **preentrenado** (no se ha reentrenado con tu verdad-terreno). Puedes guardarlo igualmente con otro nombre, aunque para tu zona conviene marcar borreguiles verificados y reentrenar antes de guardar.":
        "The model in memory is **pretrained** (it has not been retrained with your ground truth). You can still save it under another name, though for your area it is best to mark verified borreguiles and retrain before saving.",
    "💾 Guardar en la carpeta (local + git)": "💾 Save to the folder (local + git)",
    "✓ Marcar como borreguil verificado y reentrenar": "✓ Mark as verified borreguil and retrain",
    "🗑 Quitar del análisis": "🗑 Remove from the analysis",
    "Elimina los puntos seleccionados del mapa, la tabla y las descargas (no reentrena el modelo).":
        "Removes the selected points from the map, table and downloads (does not retrain the model).",
    "📈 Perfil espectral del punto vs. borreguiles de referencia":
        "📈 Spectral profile of the point vs. reference borreguiles",
    "✕ Quitar selección": "✕ Clear selection",
    "Recuento por **combinación completa** (ambiente · humedad · pureza):":
        "Count by **full combination** (environment · moisture · purity):",
    "Para mejor precisión en tu zona, sube borreguiles verificados en campo (presence-only) y vuelve a ejecutar.":
        "For better accuracy in your area, upload field-verified borreguiles (presence-only) and run again.",
    "⬇ Descargar .joblib": "⬇ Download .joblib",
    "Alternativa robusta al guardado en carpeta: descarga el fichero y cópialo a borreguil_app/ (o súbelo al repo).":
        "Robust alternative to saving in the folder: download the file and copy it to borreguil_app/ (or upload it to the repo).",
    "🗑 Quitar este punto del análisis": "🗑 Remove this point from the analysis",
    "Probabilidad RF": "RF probability",
    "No se pudo resolver el WDPA ID. Para IDs fuera del cache hace falta un **token de Protected Planet** o el **backend GEE**. Alternativa: busca por nombre o sube un vector. Se usará bbox de los puntos.":
        "The WDPA ID could not be resolved. IDs outside the cache require a **Protected Planet token** or the **GEE backend**. Alternative: search by name or upload a vector. The points' bounding box will be used.",
    "**Modo: entrenado con tu verdad-terreno (presence-only)**":
        "**Mode: trained with your ground truth (presence-only)**",
    "No puedes quitar todos los puntos.": "You cannot remove all the points.",
    "No se encontró un polígono para ese nombre. Prueba a ser más específico (incluye \"Parque Nacional/Natural\") o sube un vector. Se usará bbox de los puntos.":
        "No polygon was found for that name. Try being more specific (include \"Parque Nacional/Natural\") or upload a vector. The points' bounding box will be used.",
    "No se pudo resolver el área. Se usará bbox de los puntos.":
        "The area could not be resolved. The points' bounding box will be used.",
    "⚠ No se pudo obtener la altitud del área (¿zona de agua, fuera de cobertura SRTM, o fallo temporal de la API de elevación?). Prueba sin filtro de altitud o repite en unos minutos.":
        "⚠ Could not obtain the area's elevation (water zone, outside SRTM coverage, or a temporary failure of the elevation API?). Try without an elevation filter or retry in a few minutes.",
    "⚠ No se generaron puntos. Revisa el área y los filtros de altitud.":
        "⚠ No points were generated. Check the area and the elevation filters.",
    "⚠ Pocos positivos (< 30). Recomendado: 30–150 para un modelo fiable en tu zona.":
        "⚠ Few positives (< 30). Recommended: 30–150 for a reliable model in your area.",
    "El fichero no se creó. Usa **⬇ Descargar .joblib**.":
        "The file was not created. Use **⬇ Download .joblib**.",
}
