# Manual de usuario — Detector de borreguiles

Referencia completa del programa: qué hace cada opción, qué significa cada
variable y cómo interpretar los resultados.

> ¿Es tu primera vez? Empieza por el **[TUTORIAL.md](TUTORIAL.md)**, que te lleva
> paso a paso desde la instalación hasta tu primer mapa.

---

## Índice

1. [Qué hace el programa](#1-qué-hace-el-programa)
2. [Instalación y arranque](#2-instalación-y-arranque)
3. [Dónde se guardan tus datos](#3-dónde-se-guardan-tus-datos)
4. [Panel lateral: todas las opciones](#4-panel-lateral-todas-las-opciones)
5. [Formato de los ficheros de entrada](#5-formato-de-los-ficheros-de-entrada)
6. [Los modelos](#6-los-modelos)
7. [El umbral de decisión](#7-el-umbral-de-decisión)
8. [Las variables del modelo](#8-las-variables-del-modelo)
9. [Resultados: las seis pestañas](#9-resultados-las-seis-pestañas)
10. [Clasificación jerárquica del tipo](#10-clasificación-jerárquica-del-tipo)
11. [Entrenar un modelo propio](#11-entrenar-un-modelo-propio)
12. [Rendimiento](#12-rendimiento)
13. [Solución de problemas](#13-solución-de-problemas)
14. [Limitaciones conocidas](#14-limitaciones-conocidas)

---

## 1. Qué hace el programa

Identifica **borreguiles** (cervunales higroturbosos, *high-mountain wet
meadows*): prados húmedos de alta montaña sostenidos por agua permanente.

El proceso, para cada punto:

```
Punto (lat, lon)
   ├─ Sentinel-2  → NDVI, NDWI, NDMI, EVI, albedo… (inicio y fin de verano, 2017-2025)
   ├─ Sentinel-1  → retrodispersión radar VV/VH (humedad del suelo)
   ├─ DEM         → altitud, pendiente, curvatura, TWI (acumulación de agua)
   ├─ Ortofoto    → textura, contraste, fracción verde, roca alrededor
   └─ OSM         → distancia a cauces y lagunas
                          ↓
              Random Forest (50-66 variables)
                          ↓
              Probabilidad 0-1 → decisión → mapa
```

El modelo incluido se entrenó con **796 puntos de Sierra Nevada** verificados en
campo (390 borreguiles / 406 no-borreguiles), con validación espacial honesta
(GroupKFold por cuenca: entrena en unas cuencas y evalúa en otras nunca vistas).

---

## 2. Instalación y arranque

No hay instalación: se descomprime y se ejecuta. Todo va dentro (Python,
librerías y modelos).

| Sistema | Fichero | Ejecutar |
|---|---|---|
| Windows 10/11 | `Borreguil-windows.zip` | `Borreguil.exe` |
| macOS Apple Silicon | `Borreguil-macos-arm64.zip` | `Borreguil` |
| macOS Intel | `Borreguil-macos-intel.zip` | `Borreguil` |
| Linux x86-64 | `Borreguil-linux.tar.gz` | `./Borreguil` |

**Avisos de seguridad del sistema.** Los ejecutables no llevan firma comercial de
pago, así que el sistema advierte la primera vez:

- **Windows**: «Windows protegió su PC» → *Más información* → *Ejecutar de todas
  formas*.
- **macOS**: clic derecho → *Abrir* → *Abrir*. Si insiste:
  `xattr -dr com.apple.quarantine /ruta/a/Borreguil`

Al arrancar se abre una **ventana de consola** (el motor: no cerrarla) y el
navegador en **http://localhost:8501**. El primer arranque tarda 30–60 s.

**Para cerrar el programa**: cierra la ventana de consola o pulsa `Ctrl+C` en ella.
Cerrar solo la pestaña del navegador no lo detiene.

> **Privacidad**: el servidor escucha únicamente en `localhost`; no es accesible
> desde otros ordenadores de la red.

---

## 3. Dónde se guardan tus datos

| Sistema | Carpeta |
|---|---|
| Windows | `C:\Users\TU_USUARIO\Borreguil` |
| macOS / Linux | `~/Borreguil` |

Ahí van los **modelos que entrenes** y las salidas. Se conserva entre sesiones y
sobrevive a las actualizaciones del programa (la carpeta interna del ejecutable es
temporal y se borra al salir).

Para usar otra ubicación, define la variable de entorno `BORREGUIL_DATA_DIR`.

---

## 4. Panel lateral: todas las opciones

### Idioma / Language
Español o inglés. Cambia toda la interfaz al instante.

### Inputs

**Origen de los puntos**

- **📁 Subir archivo** — tus candidatos en KML, GeoJSON o shapefile.
- **🎲 Generar aleatoriamente** — el programa los crea dentro del área:
  - *Nº puntos*: 30–1000.
  - *Estrategia*:
    - `stratified_elev` — reparto equilibrado por franjas de altitud **(recomendado)**.
    - `uniform` — aleatorio puro.
    - `poisson_disc` — con **distancia mínima** entre puntos (evita duplicados).
  - *Distancia mínima* (solo `poisson_disc`): metros entre puntos.
  - *Elev. mín/máx*: filtro de altitud. Para borreguiles, **mín ≈ 2000 m** en
    Sierra Nevada; ajústalo a la altitud de tu macizo.
  - *Semilla*: fija el azar para poder **reproducir** exactamente el mismo
    conjunto de puntos. Importante para publicar.

> Los puntos generados se ajustan al **centro del píxel Sentinel-2 (10 m)**, de
> modo que las variables extraídas corresponden exactamente a ese píxel.

**Verdad-terreno (opcional)** — tus puntos confirmados en campo. Ver
[§5](#5-formato-de-los-ficheros-de-entrada) y [§11](#11-entrenar-un-modelo-propio).

**Usar verdad-terreno solo para entrenar** — el modelo aprende de ellos pero **no
se muestran** en mapa, tabla ni descargas. Útil cuando aportas muchos y no quieres
que tapen a tus candidatos.

**Capa de lagunas (opcional)** — puntos de lagunas conocidas, para clasificar mejor
el ambiente «laguna». Sin ella se usa la capa incluida de Sierra Nevada (65 lagunas).

### Área de estudio

| Modo | Uso |
|---|---|
| 🗺 **Vector** | Sube tu polígono (KML/GeoJSON/GeoPackage/shapefile). |
| 🆔 **WDPA ID** | Identificador de protectedplanet.net. Ej.: `555512151` Sierra Nevada, `4514` Picos de Europa, `11` Yellowstone. |
| 🔤 **Nombre** | Búsqueda por nombre (Nominatim/OSM). Cuanto más específico, mejor. |
| ⨯ **Sin área** | Usa el rectángulo que engloba tus puntos. |

### Fuente de datos satelitales (backend)

| Backend | Cuenta | Velocidad | Cuándo |
|---|---|---|---|
| 🌍 **Google Earth Engine** | Sí (gratis académico) | **Rápido** (cálculo en Google) | Recomendado. |
| 🛰 **Planetary Computer** | No | Lento (descarga a tu PC) | Sin cuenta GEE. |

Alta en GEE: ver [TUTORIAL §3](TUTORIAL.md#3-darte-de-alta-en-google-earth-engine).

### Modelo de partida
Ver [§6](#6-los-modelos).

### Opciones

- **Años Sentinel-2** — periodo analizado (por defecto `2017-2025`). Más años =
  más robusto frente a un verano anómalo, pero más lento.
- **Umbral RF** — ver [§7](#7-el-umbral-de-decisión).
- **Fuente de imágenes** — ESRI (mundial, ~0,5 m) o PNOA/autonómica (España, 0,25 m).
- **⚡ Modo rápido (vista previa)** — salta las descargas lentas punto a punto
  (imágenes, Sentinel-1, CIR/Planet). Resultados **aproximados**: sirve para
  iterar, no para resultados definitivos.
- **Saltar Sentinel-2** — mucho más rápido pero sin índices de vegetación.
  Prácticamente inutiliza el modelo.
- **Usar OSM** — hidrografía e infraestructuras. El servicio Overpass falla a veces.
- **PlanetScope 3 m** — requiere clave de pago `PL_API_KEY`.

---

## 5. Formato de los ficheros de entrada

Formatos aceptados: **KML**, **GeoJSON**, **shapefile** (`.shp`+`.shx`+`.dbf`+`.prj`,
o un `.zip`), **GeoPackage**.

### Solo presencias (lo más simple)
Un fichero de puntos **sin ningún atributo**. Todos se toman como borreguil; las
ausencias se generan solas (pseudo-ausencias alejadas de los positivos).

### Presencias + ausencias (recomendado si las tienes)
Añade un **atributo de clase** a cada punto. Se reconocen (sin distinguir
mayúsculas):

- **Nombre del atributo**: `Borreguil`, `presencia`, `clase`, `tipo` o `label`.
- **Presencia**: `si` · `1` · `presencia` · `borreguil`
- **Ausencia**: `no` · `0` · `ausencia` · `no_borreguil`

Ejemplo GeoJSON:

```json
{
  "type": "FeatureCollection",
  "features": [
    {"type": "Feature",
     "geometry": {"type": "Point", "coordinates": [-3.3117, 37.0534]},
     "properties": {"Borreguil": "si"}},
    {"type": "Feature",
     "geometry": {"type": "Point", "coordinates": [-3.3201, 37.0489]},
     "properties": {"Borreguil": "no"}}
  ]
}
```

En KML, dentro de cada `Placemark`:
`<SimpleData name="Borreguil">no</SimpleData>`

> Un punto **sin** atributo de clase se interpreta como **presencia**.
> Las coordenadas deben ser **geográficas (WGS84, EPSG:4326)**.

---

## 6. Los modelos

Todos parten de las **mismas 50 variables base**. Las variantes añaden una fuente
de muy alta resolución.

| Modelo | Variables | AUC | Requiere | Cuándo usarlo |
|---|---|---|---|---|
| `rf_sierra_nevada.joblib` | 50 | **0,83** | nada | **Universal.** Cualquier zona del mundo. |
| `rf_sierra_nevada_cir.joblib` | 56 | **0,93** | ortofoto IR (gratis) | **El mejor en España** con cobertura IR: Andalucía, Cataluña, Canarias. |
| `rf_*_planet.joblib` | 60 | — | `PL_API_KEY` (pago) | Fuera de España, si tienes suscripción Planet. |

**AUC** (área bajo la curva ROC) mide la capacidad de distinguir borreguil de
no-borreguil: 0,5 = azar, 1,0 = perfecto. Se midió con **validación espacial**
(entrenando en unas cuencas y evaluando en otras), que es más exigente y honesta
que una validación aleatoria.

> La variable más informativa de las variantes de alta resolución es la
> **textura/contraste del NDVI**: el borde nítido del borreguil contra la roca,
> que Sentinel-2 a 10 m promedia y pierde.

### Transferir a otras montañas
El modelo aprendió en Sierra Nevada. Funciona en macizos comparables, pero:

- La **altitud** es una de las variables de más peso, y su rango absoluto cambia
  entre cordilleras (un borreguil pirenaico o alpino puede estar más bajo).
- Las probabilidades **se recalibran** en cada zona → usa el umbral **Otsu**.
- Para máxima precisión, [entrena con verdad-terreno local](#11-entrenar-un-modelo-propio).

---

## 7. El umbral de decisión

El modelo da una **probabilidad** (0–1). El umbral la convierte en decisión:

| Decisión | Regla |
|---|---|
| BORREGUIL VERIFICADO | Confirmado en campo (verdad-terreno) |
| BORREGUIL PROBABLE | probabilidad **≥ 0,70** |
| POSIBLE BORREGUIL | entre el **umbral** y 0,70 |
| INCIERTO | entre 0,20 y el umbral |
| NO BORREGUIL | **≤ 0,20** |

### Los tres modos

| Modo | Cómo calcula | Cuándo |
|---|---|---|
| **Automático · Otsu** *(por defecto)* | Busca el corte que mejor separa los dos grupos en **esta ejecución** (método de Otsu, el clásico de binarización de imágenes). No necesita datos de campo. | **Recomendado**, sobre todo en zonas nuevas. |
| **Automático · verdad-terreno** | Con presencias y ausencias, maximiza el índice de Youden (sensibilidad + especificidad − 1). Solo con presencias, el corte que captura el 90 % de los borreguiles conocidos. | Si aportas verdad-terreno suficiente (≥ 5 por clase). |
| **Manual** | El valor del deslizador. | Solo si sabes exactamente qué corte quieres. |

> ⚠️ **El error más común**: dejar el umbral en **Manual** al analizar una zona
> nueva. Como las probabilidades se recalibran por región, un corte fijo puede
> dejar **casi todos los puntos fuera** (síntoma: «de 500 puntos solo detecta 1»).
> **Solución: Otsu.**

---

## 8. Las variables del modelo

### Espectrales (Sentinel-2, 10-20 m)
| Variable | Qué mide |
|---|---|
| `ndvi_early` / `ndvi_late` | Vigor de la vegetación al inicio / final del verano. |
| `ndvi_drop` | Cuánto se seca de junio a septiembre. **Un borreguil se seca poco**: es de las variables más decisivas. |
| `ndmi_*` | Humedad de la vegetación. |
| `ndwi_*` | Presencia de agua. |
| `evi_*` | Vigor, robusto en vegetación densa. |
| `clre_*`, `gndvi_*` | Clorofila. |
| `albedo_*` | Reflectividad (alta = roca o suelo desnudo). |
| `wavi_*` | Índice adaptado a zonas húmedas. |

Sufijos: `_early`/`_late` (inicio/fin de verano), `_mean`/`_min`/`_max`/`_sd`
(estadísticos entre años).

### Topográficas (DEM Copernicus, 30 m)
| Variable | Qué mide |
|---|---|
| `elev_dem_m` | Altitud. **Muy determinante**: los borreguiles son de alta montaña. |
| `slope_deg` | Pendiente. Los borreguiles prefieren zonas llanas. |
| `twi` | Índice topográfico de humedad: dónde se acumula el agua. |
| `curvature` | Cóncavo (acumula agua) vs. convexo (drena). |

### Radar (Sentinel-1 SAR)
`s1_vv_mean`, `s1_vh_mean`, `s1_ratio_mean`, `s1_*_sd` — sensibles a la humedad
del suelo y atraviesan las nubes.

### Textura de imagen (ortofoto 0,25-0,5 m)
`glcm_contrast`, `glcm_homog`, `edge_density`, `laplacian_var`, `frac_bgreen`,
`exg_mean`, `surr_rock`, `matorral_score`, `n_components`, `largest_frac` —
capturan que el borreguil es una **mancha verde compacta y homogénea** rodeada de
roca.

> En la pestaña **🔑 Variables** ves el peso real de cada una en el modelo activo.

---

## 9. Resultados: las seis pestañas

### 🗺️ Mapa
Puntos coloreados por decisión (leyenda abajo a la izquierda). Clic en un punto →
ficha y opción de **quitarlo** del análisis.

Capas activables (control ▤ arriba a la derecha): píxel S2 de 10 m, ventanas de
muestreo de 10 m y 20 m, y **Google Satélite** para más zoom.

### 📋 Tabla
Todos los puntos con coordenadas, probabilidad, decisión, tipo y variables.
Seleccionar una fila centra el mapa y despliega el **📈 perfil espectral**, que
compara el punto con el rango típico (p25–p75) de los borreguiles de referencia.

Con varias filas seleccionadas puedes **marcarlas como borreguil verificado y
reentrenar** (aprendizaje activo).

Con las filas de borreguil seleccionadas aparece también el panel **🌿 Revisar el
tipo de borreguil**, para confirmar o corregir el ambiente, la humedad y la pureza
de cada punto (ver [sección 10](#10-clasificación-jerárquica-del-tipo)).

Debajo, el mapa del punto seleccionado usa la **ortofoto** de fondo, dibuja el
punto sin relleno y marca en amarillo su **píxel Sentinel-2 de 10 m**. Con el
control de capas puedes cambiar a Google Satélite, que admite más zoom.

Al **quitar puntos del análisis** la selección se vacía: así ninguna acción
posterior recae sobre un punto que no has elegido.

**Nombres largos y decimales.** Si los nombres de tus puntos son muy largos (por
ejemplo, heredados del nombre de la capa: `all_borreguil_lagunas500ptos_epsg25830.41`),
en la tabla se muestran abreviados conservando el final, que es lo que los
distingue: `…epsg25830.41`. El nombre completo aparece al seleccionar el punto, en
el mapa y en todas las descargas. Los números se muestran con **4 decimales** en la
ficha del punto, en los globos del mapa y en el Excel; los ficheros guardan el
valor completo.

### 📊 Histograma RF
Distribución de probabilidades. **Dos grupos separados** = el modelo discrimina
bien. **Una sola masa central** = no está distinguiendo; revisa el modelo o aporta
verdad-terreno.

### 💾 Descargas
- `classification.csv` y `Clasificacion_puntos.xlsx` — la tabla completa, con
  todas las variables.
- `puntos.geojson` — los puntos con sus **etiquetas** (decisión, verdad-terreno,
  tipo y si está revisado), sin las variables. Es el fichero para **volver a
  cargar tu trabajo** en otra sesión o abrirlo en QGIS.
- `mapa.html` — mapa autónomo que se abre en cualquier navegador.
- `area_estudio.geojson` — el polígono del área, si la definiste.

### 🔑 Variables
Ranking de importancia del modelo activo, con el rango típico de borreguil.

### 📊 Distribuciones
Diagramas de caja por variable, dispersión altitud × NDVI, recuento por decisión y
la **clasificación jerárquica**.

---

## 10. Clasificación jerárquica del tipo

Cada borreguil detectado se etiqueta en **tres niveles anidados**:

```
AMBIENTE (dónde está)          HUMEDAD           PUREZA (qué hay en el píxel)
├── arroyo   (cauce, TWI alto)   ├── húmedo        ├── puro
├── laguna   (< 120 m de laguna) └── seco          ├── mixto-agua
└── ladera   (resto)                               └── mixto-roca
```

- **Ambiente** — por cercanía a lagunas conocidas, distancia a cauces OSM y
  topografía (TWI alto o pendiente baja → arroyo, aunque no haya capa OSM).
- **Humedad** — por NDMI (respaldo: TWI).
- **Pureza** — **relativa a los borreguiles de esa ejecución**: se marca como
  *mixto-agua* el ~15 % con más firma de agua (NDWI) y como *mixto-roca* el ~15 %
  con más albedo o menos NDVI. El resto, *puro*.

> ⚠️ **Por qué la pureza es relativa**: a 10 m de resolución **no se puede medir
> la fracción real de agua o roca dentro del píxel**. Haría falta PlanetScope (3 m)
> o CIR (0,25 m). Es una ordenación dentro de tu población, no una medida absoluta.

Umbrales ajustables en `borreguil_pipeline.py` → `HIER_THRESH`.

### Revisar el tipo: de propuesta de la app a dato revisado

El tipo que calcula la app es una **propuesta**: sale de reglas sobre las mismas
variables que usa el modelo (la humedad, por ejemplo, es un umbral de NDMI). Sirve
para orientarse, pero **no como etiqueta de entrenamiento**: un modelo entrenado
con ella solo volvería a aprender la regla. Para que el tipo sea un dato hay que
revisarlo mirando el terreno.

**Cómo se revisa**

1. Pestaña **📋 Tabla** → selecciona uno o varios borreguiles.
2. Mira el punto en el mapa de debajo (ortofoto y píxel de 10 m en amarillo) o
   usa tus datos de campo.
3. En **🌿 Revisar el tipo de borreguil** elige ambiente, humedad y pureza. Deja
   **(sin cambio)** en lo que la app ya proponía bien.
4. **✓ Guardar tipo revisado**. El punto queda marcado como revisado, con su fecha.

**↩ Deshacer revisión** devuelve los puntos seleccionados a la propuesta de la app.

Con **varios puntos** seleccionados solo cambian los niveles que elijas; el resto
se confirma tal como estaba en cada punto. Es todo o nada: si a alguno le falta un
nivel (la app no pudo proponerlo por falta de datos) y no lo eliges, no se guarda
ninguno y se indica cuál es.

**Criterios** — decide siempre con la misma regla para todos los puntos:

| Nivel | Categorías | Cómo decidir |
|---|---|---|
| Ambiente | arroyo · laguna · ladera | Dónde está el prado: junto a un cauce, junto a una laguna o en ladera |
| Humedad | húmedo · seco | Un único criterio para toda la campaña (anótalo) |
| Pureza | puro · mixto-agua · mixto-roca | Qué hay **dentro del píxel de 10 m**: solo prado, agua, o roca/suelo desnudo |

**Qué se guarda de cada punto**

| Columna | Contenido |
|---|---|
| `ambiente`, `humedad`, `pureza` | El valor vigente: el revisado si lo hay; si no, la propuesta |
| `ambiente_regla`, `humedad_regla`, `pureza_regla` | Lo que propuso la app (se recalcula en cada ejecución) |
| `tipo_revisado` | `si` cuando una persona ha revisado los tres niveles |
| `tipo_revisado_fecha` | Cuándo se revisó |

Lo revisado **no se pisa nunca**: ni al reentrenar, ni en las iteraciones de
auto-entrenamiento, ni aunque el modelo deje de clasificar el punto como borreguil.

Encima de la tabla hay un contador: cuántos borreguiles están revisados y en
cuántos se **corrigió** la propuesta. Si revisas muchos y no corriges ninguno, la
app lo avisa: un tipo aceptado sin mirar sigue siendo la regla de la app.

**Conservar la revisión entre sesiones** — descarga `puntos.geojson` y, la próxima
vez, cárgalo como *puntos candidatos* o como *verdad-terreno*. Los tipos revisados
vuelven con sus puntos. Si un punto llega marcado como revisado pero con una
categoría que no existe (por ejemplo, editada a mano), la app lo avisa y no lo
cuenta como revisado.

---

## 11. Entrenar un modelo propio

El modelo incluido aprendió en Sierra Nevada. En tu zona mejorará bastante si le
das ejemplos locales.

### Requisitos
| | Mínimo | Recomendado |
|---|---|---|
| Borreguiles confirmados | 30 | **80–150** |
| Ausencias | 0 (se generan) | 30+ |

Con menos de 30 positivos el programa avisa y sigue con el modelo preentrenado.

### Procedimiento
1. Marca borreguiles confirmados (campo o fotointerpretación).
2. Guárdalos en KML/GeoJSON (ver [§5](#5-formato-de-los-ficheros-de-entrada)).
3. Súbelos en **Opcional: verdad-terreno**.
4. Ejecuta. Verás *«Modo: entrenado con tu verdad-terreno»*.
5. En resultados, **💾 Guardar el modelo entrenado** con el nombre de tu espacio.
6. Quedará disponible en «Modelo de partida» para las siguientes sesiones.

### Cómo entrena (presence-only)
- **Positivos**: tus puntos `si`.
- **Negativos**: candidatos sin etiqueta a **más de 250 m** de cualquier positivo
  (pseudo-ausencias) más los marcados `no`.
- **Excluidos**: candidatos a menos de 250 m de un positivo (ambiguos).
- **Validación**: GroupKFold espacial por cuenca cuando hay grupos suficientes.

### Evaluación recursiva (auto-entrenamiento)
En cada iteración, los puntos que superan el **umbral de promoción** se convierten
en verdad-terreno y el modelo se recalibra.

- *Añadir*: el conjunto de positivos crece acumulativamente.
- *Reemplazar*: se recalculan desde el último modelo, conservando tus verificados.

**Lo que tú has etiquetado no se toca.** Una ausencia de campo (`Borreguil = no`)
sigue entrenando como ausencia y un punto dudoso (`Duda = si`) sigue fuera del
entrenamiento, y ninguno de los dos se promueve aunque el modelo le dé una
probabilidad alta. Solo se promueven candidatos sin etiqueta de campo. Si marcas a
mano un punto dudoso como *borreguil verificado*, la duda queda resuelta y el punto
pasa a entrenar como positivo.

Los puntos promovidos aparecen como `BORREGUIL (auto)`. En `puntos.geojson` **no**
se guardan como verdad-terreno: al recargar el fichero solo vuelve como tal lo que
verificó una persona.

> ⚠️ Un umbral de promoción bajo propaga errores. Mantenlo **alto** (≥ 0,8).

---

## 12. Rendimiento

Tiempos orientativos para **150 puntos**:

| Configuración | Tiempo |
|---|---|
| GEE + ⚡ Modo rápido | **1-2 min** |
| GEE completo | 5-15 min |
| Planetary Computer completo | 20-45 min |

**Qué es lento y por qué**: las imágenes (ESRI/PNOA), Sentinel-1 y CIR/Planet se
descargan **punto a punto en tu ordenador**, aunque uses GEE. Sentinel-2 y la
topografía sí se calculan en los servidores de Google.

**Para acelerar**: usa GEE, marca **⚡ Modo rápido** para iterar, reduce el número
de puntos y acota los años de Sentinel-2.

---

## 13. Solución de problemas

| Síntoma | Causa probable | Solución |
|---|---|---|
| **Detecta muy pocos puntos** (1 de 500) | Umbral en **Manual** | Cambiar a **Otsu**. |
| Windows bloquea el ejecutable | Sin firma comercial | *Más información* → *Ejecutar de todas formas*. |
| macOS: «no se puede comprobar» | Cuarentena de Gatekeeper | Clic derecho → *Abrir*, o `xattr -dr com.apple.quarantine`. |
| El navegador no abre | — | Ir a **http://localhost:8501**. |
| Muy lento | Backend MPC | Cambiar a GEE y/o **⚡ Modo rápido**. |
| Error de autenticación GEE | Proyecto mal escrito o alta pendiente | Revisar `ee-tunombre` y volver a autenticar. |
| Faltan texturas / SAR | **⚡ Modo rápido** activo | Desmarcarlo. |
| Histograma en una sola masa | El modelo no discrimina en esa zona | Aportar verdad-terreno local. |
| No se puede resolver el WDPA ID | ID fuera del caché | Usar token de Protected Planet, backend GEE, o buscar por nombre. |
| Overpass/OSM falla | Servicio caído | Desmarcar **Usar OSM** y reintentar. |
| No aparece mi modelo guardado | Guardado en otra carpeta | Comprobar `~/Borreguil` (ver [§3](#3-dónde-se-guardan-tus-datos)). |

Para reportar un fallo, incluye tu sistema operativo, la configuración usada y
**el texto de la ventana de consola**.

---

## 14. Limitaciones conocidas

- **Sesgo geográfico**: el modelo se entrenó en Sierra Nevada. Fuera de macizos
  comparables su fiabilidad baja; conviene verdad-terreno local.
- **Resolución**: a 10 m, un borreguil menor de ~100 m² no se distingue con
  fiabilidad. La *pureza* del píxel es relativa, no absoluta ([§10](#10-clasificación-jerárquica-del-tipo)).
- **La altitud pesa mucho** y su rango absoluto cambia entre cordilleras.
- **Dependencia de servicios externos**: Earth Engine, Planetary Computer, OSM e
  IGN pueden fallar o cambiar.
- **Verificación en campo**: los resultados son **candidatos**, no un inventario
  definitivo. Toda cartografía publicada debería validarse en campo.
- **Cuota**: Earth Engine tiene límites de uso por cuenta.

---

## Créditos y cita

Si usas este programa en un trabajo científico, cítalo mediante el **DOI** de la
versión utilizada (ver `CITATION.cff` y la página de Zenodo del repositorio).

**Datos**: Copernicus Sentinel-1 y Sentinel-2 (ESA), Copernicus DEM,
Microsoft Planetary Computer, Google Earth Engine, OpenStreetMap, PNOA/IGN,
ESRI World Imagery, Protected Planet (WDPA).
