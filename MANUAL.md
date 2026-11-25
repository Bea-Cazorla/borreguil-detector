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
9. [Resultados: las cinco pestañas](#9-resultados-las-cinco-pestañas)
10. [Clasificación jerárquica del tipo](#10-clasificación-jerárquica-del-tipo)
11. [Sumar tus puntos al modelo](#11-sumar-tus-puntos-al-modelo)
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
   ├─ DEM         → altitud, pendiente y curvatura
   ├─ Ortofoto    → textura, contraste, fracción verde, roca alrededor
   └─ OSM         → distancia a ríos y arroyos (para el tipo, no para el modelo)
                          ↓
              Random Forest (27 variables; 32 con la ortofoto infrarroja)
                          ↓
              Probabilidad 0-1 → decisión → mapa
```

El modelo incluido se entrenó con **769 puntos de Sierra Nevada** verificados en
campo (372 borreguiles y 397 no borreguiles; otros 27 dudosos no entran), con
validación espacial honesta
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
navegador en **http://localhost:8501** (si ese puerto está ocupado, 8502-8505: la
dirección exacta sale en la ventana de consola). El primer arranque tarda 30–60 s.

**Para cerrar el programa**: cierra la ventana de consola o pulsa `Ctrl+C` en ella.
Cerrar solo la pestaña del navegador no lo detiene.

**Conexión a internet.** Todo lo que la app hace por internet va por **HTTPS (puerto
443)**: Earth Engine (`*.googleapis.com`), los modelos nuevos de GitHub
(`raw.githubusercontent.com`), Planetary Computer, las ortofotos (IGN, servicios
autonómicos, ESRI y Google Satélite), OpenStreetMap (Overpass y Nominatim) y Protected
Planet. En una red con cortafuegos (un aula, por ejemplo) basta con dejar salir
el puerto 443 hacia esos sitios. La dirección `localhost:8501` de la ventana de la app
es del propio ordenador: no sale a internet.

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

- **📁 Subir archivo** — **Tus puntos**: los sitios en los que la app mira si hay
  borreguil, en KML, GeoJSON, CSV (columnas `lon` y `lat`) o Shapefile. Un Shapefile
  son varios ficheros: selecciona a la vez el .shp, el .dbf, el .shx y el .prj, o
  súbelo en un .zip (si falta una parte, la app dice cuál).
- **🎲 Generar aleatoriamente** — el programa los crea dentro del área:
  - *Nº puntos*: 30–1000.
  - *Estrategia*:
    - **Repartidos por altitud** — el rango de altitud del área se divide en 5
      franjas iguales y se toma el mismo número de puntos en cada una
      **(recomendado; viene elegido)**.
    - **Al azar** — sin ninguna regla.
    - **Al azar, separados una distancia mínima** — ningún par de puntos queda más
      cerca de esa distancia (evita puntos pegados).
  - *Distancia mínima* (solo en la última): metros entre puntos.
  - *Elev. mín/máx*: filtro de altitud. Para borreguiles, **mín ≈ 2000 m** en
    Sierra Nevada; ajústalo a la altitud de tu macizo.
  - *Semilla*: fija el azar para poder **reproducir** exactamente el mismo
    conjunto de puntos. Importante para publicar.
  - *Lejos de tus puntos de campo (m)*: si subes tu verdad-terreno, los puntos al
    azar quedan al menos a esa distancia de tus puntos de campo. Así caen en sitios
    que todavía no has muestreado, para buscar allí **ausencias** (ver «Ausencias
    fuera de tus puntos», en la pestaña Revisar puntos). 0 = sin límite, que es como
    viene.

> Los puntos generados se ajustan al **centro del píxel Sentinel-2 (10 m)**, de
> modo que las variables extraídas corresponden exactamente a ese píxel.

**Opcional: puntos de campo para comprobar el modelo** — sitios donde ya sabes si
hay borreguil (por el campo o la ortofoto), **en un fichero aparte**. Sirven para
medir cuánto acierta el modelo y, si quieres, para reentrenarlo. Si esa información
ya va en *Tus puntos* (columna `presence`: 1 = borreguil, 0 = no), no subas nada
aquí. Admite los mismos formatos que *Tus puntos*. Ver
[§5](#5-formato-de-los-ficheros-de-entrada) y [§11](#11-sumar-tus-puntos-al-modelo).

**Usar los puntos de campo solo para medir y entrenar (no mostrarlos en el mapa)** —
sirven para medir cuánto acierta el modelo y, si marcas «Sumar mis puntos a los del
modelo y reentrenarlo», para reentrenarlo, pero **no se muestran** en mapa, tabla ni
descargas. Tus puntos que coinciden (a menos de 20 m) con uno de campo sí quedan como
verificados. Útil cuando aportas muchos y no quieres que tapen a tus puntos.

**Lagunas** — para el ambiente «laguna», la app usa siempre la capa de lagunas de
Sierra Nevada que trae (65 lagunas); no se puede subir otra. Fuera de Sierra Nevada, un
punto es de laguna solo si se ve agua cerca (ver la sección 10).

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
El nombre del proyecto de Earth Engine se recuerda: cuando la conexión funciona
una vez, la próxima aparece ya escrito.

Los dos dan **los mismos números** en Sentinel-2 y el relieve: con Earth Engine, las
variables de Sentinel-2 se calculan imitando lo que hace Planetary Computer, que es
con lo que se entrenó el modelo de Sierra Nevada (ver la sección 8). La diferencia es
el **radar (Sentinel-1)**: solo se lee con Earth Engine, en segundos. Con Planetary
Computer la app usa una versión del modelo entrenada sin radar, que acierta algo menos
(AUC 0,833 en lugar de 0,847 con el modelo base; ver la sección 6).

### Modelo de partida
Ver [§6](#6-los-modelos). Es el que da la probabilidad de cada punto, **aunque tus
puntos traigan etiquetas de campo**: con ellas la app mide cuántas acierta (recuadro
«🧭 Modelo usado y acierto con tus puntos de campo», debajo del recuento de resultados
y encima de las cinco pestañas).

- **Sumar mis puntos a los del modelo y reentrenarlo** (sin marcar por defecto): la app vuelve a entrenar el modelo con
  los 769 puntos de Sierra Nevada con los que se hizo MÁS todos tus puntos con
  etiqueta (de campo, verificados en «Revisar puntos» o añadidos en el mapa): un solo modelo
  con todos. Ver [§11](#11-sumar-tus-puntos-al-modelo).

### Opciones

- **Años Sentinel-2** — periodo analizado (por defecto `2017-2025`). Más años =
  más robusto frente a un verano anómalo, pero más lento.
- **Umbral RF** — ver [§7](#7-el-umbral-de-decisión).
- **Fuente de imágenes** — ESRI (mundial, ~0,5 m) o PNOA/autonómica (España, 0,25 m).
- **⚡ Modo rápido (vista previa)** — salta las descargas lentas punto a punto
  (imágenes aéreas y sus texturas, CIR/Planet). Resultados **aproximados**: sirve
  para iterar, no para resultados definitivos.
- **Saltar imágenes (más rápido, sin texturas)** — no descarga las imágenes aéreas. El
  modelo se entrenó con sus texturas: sin ellas se rellenan con un valor típico de
  Sierra Nevada y la probabilidad es **menos fiable** (la app lo avisa). Con el modo
  rápido va siempre marcada.
- **Saltar Sentinel-2** — mucho más rápido pero sin índices de vegetación.
  Prácticamente inutiliza el modelo.
- **PlanetScope 3 m** — requiere clave de pago `PL_API_KEY`.

---

## 5. Formato de los ficheros de entrada

Formatos aceptados para los puntos: **KML**, **GeoJSON**, **CSV** (columnas `lon` y
`lat`) y **shapefile** (`.shp`+`.shx`+`.dbf`+`.prj`, o un `.zip`). El **GeoPackage**
solo vale para el área de estudio, la máscara y la capa de cauces.

### Puntos de campo: presencias y ausencias (recomendado)
Usa la [plantilla de datos de campo](#plantilla-de-datos-de-campo): una fila por
punto, con `presence` = `1` (borreguil) o `0` (no borreguil); el resto de columnas
son opcionales. **Incluye ausencias**: los sitios donde has comprobado que no hay
borreguil valen tanto como los borreguiles. Sin ellos no se mide bien cuánto acierta
el modelo ni se puede hacer el mapa de borreguil sí o no: la app no se inventa
ausencias.

Si tu fichero trae la etiqueta con otro nombre, también se reconoce, sin distinguir
mayúsculas (se lee igual que en *Tus puntos*; ver «Etiquetas en tus puntos», abajo):

- **Nombre de la columna**: `presence`, `Borreguil`, `presencia`, `label`, `clase`,
  `etiqueta`…
- **Presencia**: `1` (o `1.0`) · `si` · `yes` · `true` · `presencia` · `borreguil`
- **Ausencia**: `0` (o `0.0`) · `no` · `false` · `ausencia` · `no_borreguil`
- Los **dudosos** (`doubtful` = `1`) entran como dudosos y no cuentan.

Al subir el fichero, debajo aparece qué columna ha leído y cuántos borreguiles y no
borreguiles hay. Si la columna existe pero sus valores no se entienden, no se usa
ningún punto (mejor eso que tomar ausencias por borreguiles), y los puntos sin valor
no se usan.

Ejemplo GeoJSON:

```json
{
  "type": "FeatureCollection",
  "features": [
    {"type": "Feature",
     "geometry": {"type": "Point", "coordinates": [-3.3117, 37.0534]},
     "properties": {"presence": 1}},
    {"type": "Feature",
     "geometry": {"type": "Point", "coordinates": [-3.3201, 37.0489]},
     "properties": {"presence": 0}}
  ]
}
```

En KML, dentro de cada `Placemark`: `<SimpleData name="presence">0</SimpleData>`

### Solo presencias
En *puntos de campo para comprobar el modelo*, un fichero **sin ninguna columna** de
etiqueta: cada punto se toma como borreguil. Sirve para ver cuántos borreguiles
acierta el modelo, pero no aporta ausencias. En *Tus puntos*, en cambio, un punto
sin etiqueta es solo un sitio donde la app mira si hay borreguil.

> Las coordenadas deben ser **geográficas (WGS84, EPSG:4326)**.

### Etiquetas en tus puntos
Lo anterior es para los **puntos de campo**. **Tus puntos** también
pueden traer ya su etiqueta (por ejemplo, tus puntos revisados o los de otra
persona), y entonces el modelo de la pestaña «Mapa Modelo RF» los usa como muestras. La app
la guarda en `Borreguil` (`si`/`no`) y la duda en `Duda` (`si`), pero al cargar el
fichero **traduce** otros nombres y valores:

| | Columnas reconocidas (sin distinguir mayúsculas) | Valores |
|---|---|---|
| ¿Es borreguil? | `Borreguil`, `presencia`, `presence`, `presente`, `etiqueta`, `label`, `clase`, `class`, `target`, `verdad`, `truth` | sí: `si`, `sí`, `yes`, `1`, `true`, `presencia`, `presence`… · no: `no`, `0`, `false`, `ausencia`, `absence`… |
| ¿Es dudoso? | `Duda`, `dudoso`, `dudosa`, `doubtful`, `doubt`, `uncertain`, `incierto` | los mismos |
| ¿Quién lo revisó? | `observer`, `observador`, `observadora`, `revisor`, `revisora`, `reviewer` | texto |
| Su perfil | `expertise`, `perfil`, `profile` | `student`, `manager`, `researcher` (o `estudiante`, `gestor/a`, `investigador/a`) |
| Su certidumbre | `certainty`, `certidumbre`, `certeza`, `confianza`, `confidence` | `high`, `medium`, `low` (o `alta`, `media`, `baja`) |

- Se usa la **primera** columna de la lista que tenga algún valor de sí o no.
  Una que no sea `Borreguil` solo se acepta si casi todos sus valores (≥ 90 %)
  son de sí o no; así una `clase` con tipos de vegetación no se toma por
  etiqueta. `tipo` no se mira nunca (suele ser el tipo de borreguil).
- Si la columna solo cambia en las mayúsculas (`borreguil`, `duda`) se renombra;
  si tiene otro nombre (`presence`, `doubtful`) se conserva y se añade `Borreguil`
  o `Duda` al lado.
- **Al subir el fichero**, debajo aparece qué columna ha leído y cuántos sí, no y
  dudosos hay. Si **no encuentra ninguna**, avisa y lista las columnas del
  fichero: los puntos se clasifican igual, pero ninguno cuenta como etiquetado.
  Si hay valores que no entiende (`quizás`), se quedan sin etiqueta y también
  lo dice.

---

## 6. Los modelos

Todos parten de las **mismas variables base**, y de cada grupo de variables que dicen
lo mismo usan **una sola** (ver la sección 8). Las variantes añaden una fuente de muy
alta resolución.

| Modelo | Variables | AUC | Requiere | Cuándo usarlo |
|---|---|---|---|---|
| `rf_sierra_nevada.joblib` | 27 | **0,847** | nada | **Universal.** Cualquier zona del mundo. |
| `rf_sierra_nevada_cir.joblib` | 32 | **0,945** | ortofoto IR (gratis) | **El mejor en España** con cobertura IR: Andalucía, Cataluña, Canarias. |

Cada uno tiene además una **versión sin radar** (carpeta `sin_radar/`), que no sale en
el desplegable: la app la usa **sola** cuando el radar no se ha leído (con Planetary
Computer, o con «Saltar Sentinel-2»), y lo dice en «Modelo usado». Con validación por
cuencas:

| Modelo | Con radar | Sin radar: el mismo modelo | Sin radar: la versión sin radar |
|---|---|---|---|
| Base | 0,847 | 0,793 | **0,833** (22 variables) |
| Con ortofoto IR | 0,945 | 0,927 | **0,937** (27 variables) |

**AUC** (área bajo la curva ROC) mide la capacidad de distinguir borreguil de
no-borreguil: 0,5 = azar, 1,0 = perfecto. Se midió con **validación espacial**
(entrenando en unas cuencas y evaluando en otras), que es más exigente y honesta
que una validación aleatoria.

> La variable más informativa de las variantes de alta resolución es la
> **textura/contraste del NDVI**: el borde nítido del borreguil contra la roca,
> que Sentinel-2 a 10 m promedia y pierde.

### Modelos nuevos sin reinstalar
Al abrirse, la app mira en GitHub la lista de modelos públicos del proyecto
(`modelos.json`) y **descarga los que no tiene o los que han cambiado** a la carpeta
`Borreguil/modelos_descargados` de tu cuenta. Salen en el desplegable **Modelo de
partida** como los demás, y uno descargado pasa delante del que trae la app con el
mismo nombre. Al final del panel lateral, una línea dice cómo ha ido: «Modelos al día
con GitHub», los que ha descargado, o que no hay conexión. Sin internet, la app usa los
que ya tiene.

Cada fichero se comprueba con su **huella (SHA-256)** antes de guardarlo: si llega
incompleto o no es el publicado, se descarta. Solo se descarga del repositorio del
proyecto y por HTTPS. Para usar otro repositorio, o ninguno (un aula sin internet),
define antes de abrir la app la variable de entorno `BORREGUIL_MODELOS_URL` con la
dirección de su `modelos.json`, o vacía. Cómo se publica un modelo: `PUBLICAR.md`.

### Transferir a otras montañas
El modelo aprendió en Sierra Nevada. Funciona en macizos comparables, pero:

- La **altitud** es una de las variables de más peso, y su rango absoluto cambia
  entre cordilleras (un borreguil pirenaico o alpino puede estar más bajo).
- Las probabilidades **se recalibran** en cada zona → usa el umbral **Otsu**.
- Para máxima precisión, [suma tu verdad-terreno local al modelo](#11-sumar-tus-puntos-al-modelo).

---

## 7. El umbral de decisión

El modelo da una **probabilidad** (0–1). El umbral la convierte en decisión:

| Decisión | Regla |
|---|---|
| BORREGUIL VERIFICADO | Confirmado (de campo, en «Revisar puntos» o añadido en el mapa) |
| BORREGUIL PROBABLE | probabilidad **≥ 0,700** |
| POSIBLE BORREGUIL | entre el **umbral** y 0,700 |
| INCIERTO | entre 0,200 y el umbral |
| NO BORREGUIL | **≤ 0,200** |
| NO BORREGUIL VERIFICADO | Ausencia confirmada (de campo, en «Revisar puntos» o añadida en el mapa) |
| DUDOSO (campo) | Marcado como dudoso: no cuenta |

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
| `nbr_late` | Índice con el infrarrojo de onda corta (NBR) a final del verano. |

**El modelo no usa todas.** Muchas variables dicen lo mismo: por ejemplo, 13 índices
de Sentinel-2 de verdor (NDVI, EVI, Clre, GNDVI, NDWI…) se parecen entre sí con una
correlación de al menos 0,89. De cada grupo de variables en el que **todas las
parejas** tienen una correlación de 0,8 o más, el modelo usa solo la que mejor separa
borreguil de no borreguil por sí sola (del grupo del verdor, `ndvi_early`). Quedan
**27 variables** en el modelo base y **32** en el `_cir`. Comprobado con tus 769
puntos y validación por cuencas: el acierto no baja (AUC 0,827 → 0,838 y 0,934 →
0,940) y la decisión es la misma en el 94 % y el 97 % de los puntos. La app sigue
calculando todas: algunas sirven para proponer el tipo de borreguil. Después, el
radar pasó a leerse de Earth Engine y los dos modelos se reentrenaron con las mismas
variables (AUC 0,846 y 0,946), y el 20-11-2026 se quitó el **TWI** (índice
topográfico de humedad): Earth Engine no lo calcula, así que la app lo rellenaba
siempre con el valor típico, y sin él el modelo acierta lo mismo (AUC 0,847 y 0,945;
la misma decisión en el 98,6 % y el 99,6 % de los puntos).

Sufijos: `_early`/`_late` (inicio de verano, del 15 de junio al 25 de julio / fin
de verano, del 1 de agosto al 30 de septiembre), `_mean`/`_min`/`_max`/`_sd`
(estadísticos de esas imágenes de verano).

Cómo se calculan, igual que cuando se entrenó el modelo de Sierra Nevada:

- **Qué imagen.** De cada ventana y cada año se toma la **última imagen sin nubes**
  sobre el punto (con menos del 30 % de nubes en la escena). Inicio y fin de verano
  son la **mediana entre años**. Los estadísticos usan esas mismas imágenes, todos los
  años juntos, quitando las que tienen nieve en el punto.
- **Qué zona.** Cada valor es la **media de una ventana alrededor del punto**: 5 × 5
  píxeles (50 m) en las bandas de 10 m y 3 × 3 píxeles (60 m) en las de 20 m. Son las
  ventanas de muestreo que se pueden ver en el mapa.

En el mapa de tipos se calculan de otra manera: año a año y en cada píxel (ver la
sección 10).

### Topográficas (DEM Copernicus, 30 m)
| Variable | Qué mide |
|---|---|
| `elev_dem_m` | Altitud. **Muy determinante**: los borreguiles son de alta montaña. |
| `slope_deg` | Pendiente. Los borreguiles prefieren zonas llanas. |
| `curvature` | Cóncavo (acumula agua) vs. convexo (drena). |

### Radar (Sentinel-1)
`s1_vv_mean`, `s1_vh_mean`, `s1_ratio_mean`, `s1_*_sd` — sensibles a la humedad
del suelo y atraviesan las nubes. Media y variabilidad de todas las pasadas de junio
a septiembre de 2019 a 2024, con el efecto del relieve corregido. **Solo se leen con
Earth Engine** (en segundos); con Planetary Computer la app usa la versión del
modelo entrenada sin radar (sección 6). Hasta noviembre de 2026 se leían de Planetary Computer: más de
media hora con solo 20 puntos.

### Textura de imagen (ortofoto 0,25-0,5 m)
`frac_bgreen`, `glcm_contrast`, `edge_density`, `vegcontrast`, `surr_rock`,
`granul_5`, `largest_frac` y `matorral_score` (las que usa el modelo; la app calcula
además `glcm_homog`, `laplacian_var`, `exg_mean` y `n_components`) — capturan que el
borreguil es una **mancha verde compacta y homogénea** rodeada de roca.

### Ortofoto infrarroja (solo el `_cir`, 0,25 m)
`cir_ndvi_mean`, `cir_ndvi_p90`, `cir_ndvi_sd`, `cir_nir_mean` y `cir_ndvi_contrast` —
un NDVI a 25 cm en los 25 m centrales del punto: su mediana, su percentil 90, su
variabilidad, el infrarrojo cercano y el contraste con el anillo de alrededor. Es la
variable que más pesa en ese modelo: el borde nítido del borreguil contra la roca.

> En la pestaña **Variables del modelo** ves el peso real de cada una en el modelo activo.

---

## 9. Resultados: las cinco pestañas

Arriba del todo, la app tiene tres pestañas grandes: **📍 Verdad terreno**, donde
se ejecuta la detección y se ven y revisan los puntos; **🌿 Mapa Modelo RF**, para
el mapa de borreguiles (sí o no y, si quieres, su tipo); y **📖 Documentación y recomendaciones**. Los resultados de una
ejecución salen en la primera, repartidos en cinco pestañas: **Mapa de los puntos**,
**Revisar puntos**, **Resumen de los puntos**, **Variables del modelo** y **Descargas**.

**Quién revisa y con qué certidumbre.** Encima de las cinco pestañas puedes escribir
tu nombre y elegir tu perfil (estudiante, gestor/a o investigador/a). Se apuntan en
cada punto que verifiques en «Revisar puntos» o añadas en el mapa, junto con la
**certidumbre** que indiques al hacerlo: **alta** (no hay duda), **media** (hay
alguna duda) o **baja** (es lo más probable, pero hay bastante duda). Si no sabes
decir si es borreguil, no lo verifiques; en el mapa, contesta **? No estoy seguro**
y el punto queda como dudoso, que no cuenta. Todo es opcional: lo que no indiques se
queda vacío. Va en las descargas con las columnas de la plantilla (`observer`,
`expertise` y `certainty`) y, debajo del acierto del modelo, el desplegable
**Acierto según la certidumbre y el perfil de quien revisa** separa los aciertos por
grupos: si el modelo falla sobre todo en los puntos de certidumbre baja, puede que
el error esté en la etiqueta y no en el modelo.

### Mapa de los puntos
Puntos coloreados por decisión (leyenda abajo a la izquierda). Clic en un punto →
ficha y opción de **quitarlo** del análisis.

Capas activables (control ▤ arriba a la derecha): píxel S2 de 10 m, ventanas de
muestreo de 10 m y 20 m, y **Google Satélite** para más zoom.

**➕ Añadir puntos pinchando en el mapa.** Para quien conoce el sitio (por ejemplo,
en un taller con los gestores). Con el interruptor encendido:

Si marcas **Sumar mis puntos a los del modelo y reentrenarlo**, cada punto que
añadas se suma a los 769 de los que aprendió el modelo y se reentrena con todos
([§11](#11-sumar-tus-puntos-al-modelo)).

1. Pincha en el mapa donde sepas si hay borreguil. Sale un círculo rojo y, debajo
   del mapa, tres botones: **✓ Aquí hay borreguil**, **✗ Aquí no hay** y **? No estoy
   seguro** (y **Cancelar**), y encima de ellos la **certidumbre** de tu respuesta
   (opcional). Si pinchas encima de un punto que ya está, la app lo dice y no añade
   otro.
2. Al contestar, la app lee **solo ese punto** (con Earth Engine, unos segundos) y
   dice qué opinaba el modelo **antes** de saber tu respuesta: su probabilidad y si
   coincide contigo.
3. El punto queda como un dato de campo más (`mapa_1`, `mapa_2`…, con borde blanco
   en el mapa): borreguil o ausencia verificada, o dudoso. Cuenta en el acierto del
   modelo y, con la casilla del modelo propio, también para entrenarlo. Va en las
   descargas con las columnas de la plantilla: `method` = `local knowledge`, la
   fecha, la **nota** si la escribes (`notes`) y, si los indicas, quién revisa
   (`observer`), su perfil (`expertise`) y la certidumbre (`certainty`).

Debajo del mapa, una tabla resume los puntos añadidos: qué dijo quien los marcó, la
probabilidad que les daba el modelo y si coincidía («el modelo coincidía en 4 de
5»). Los dudosos no cuentan. **↩ Quitar el último añadido** deshace el último.

### Revisar puntos
Todos los puntos con coordenadas, probabilidad, decisión, tipo y variables.
Seleccionar una fila centra el mapa y despliega la ficha del punto:

- **Lo que más pesa en el modelo**: el valor del punto en las 3 variables que más
  pesan (una por grupo de variables parecidas, con el peso del grupo), y debajo la
  altitud y la pendiente.
- **📈 Este punto frente a los borreguiles verdaderos**: las 8 variables que más
  pesan, una por grupo, de arriba abajo. En cada una, dónde cae el punto entre los
  borreguiles con los que se entrenó el modelo, de 0 (más bajo que todos) a 100 (más
  alto que todos); la franja **verde** es su mitad central y el punto sale **naranja**
  si es más bajo o más alto que casi todos (fuera del 10–90). El **rombo rojo** es el
  **no borreguil típico** (la mediana de los no borreguiles verdaderos), en la misma
  escala: si el punto está más cerca de él que de la franja verde, se parece más a un
  no borreguil. Debajo, el valor del punto, el de la mitad central de los borreguiles
  y el del no borreguil típico.

Muchas variables son casi iguales: por ejemplo, 13 índices de verdor de Sentinel-2
tienen una correlación de 0,89 o más entre sí. El modelo usa una de cada grupo (ver
la sección 8), y la ficha enseña una de cada grupo: las demás dirían lo mismo.

**Revisar un punto: una sola revisión en dos pasos.** Al seleccionar uno o varios
puntos en la tabla, debajo sale el **mapa del punto** (ortofoto, con su píxel de 10 m
en amarillo) y, a su lado, el formulario de revisión:

1. **¿Es borreguil?** — **Sí**, **No** o **No estoy seguro**, y la **certidumbre**
   (alta, media o baja), que se guarda con quién revisa y su perfil (ver el principio
   de esta sección).
2. **Tipo** — solo si dices **Sí**: ambiente, humedad y pureza, con la propuesta de la
   app ya puesta para aceptarla o corregirla. Lo que dejes en **No lo he podido ver**
   no cuenta como revisado. Con **No** o **No estoy seguro** el tipo se apaga.

Un solo botón, **💾 Guardar la revisión**, guarda todo a la vez:

| Respuesta | Queda como | Tipo |
|---|---|---|
| Sí | «BORREGUIL VERIFICADO» | los niveles elegidos, revisados |
| No | «NO BORREGUIL VERIFICADO», igual que las ausencias de tu verdad-terreno: sirve de ausencia para el mapa | sin tipo (si tenía uno revisado, se le quita) |
| No estoy seguro | dudoso: no entrena ni cuenta en el acierto | sin tipo |

Con **varios puntos** seleccionados, lo que guardes se aplica a todos; el tipo
arranca en «No lo he podido ver», y cada uno se queda con lo suyo en los niveles que
no elijas. **↩ Deshacer** deshace la última revisión guardada, y **🗑 Quitar del
análisis** quita los puntos seleccionados. Guardar sin decir si es borreguil no
guarda nada: la app pide elegirlo primero. (Hasta el 25-11-2026 el sí o no y el tipo
se guardaban por separado, con dos botones.)

Lo que verificas cuenta en el recuadro «🧭 Modelo usado» para medir cuánto acierta
el modelo. Con el modelo de partida, la probabilidad de los demás puntos no cambia.
Con la casilla **Sumar mis puntos a los del modelo y reentrenarlo** marcada, al
guardar una revisión que cambia el sí o el no, el modelo se vuelve a entrenar con lo
verificado (aprendizaje activo), sumado a los puntos de Sierra Nevada; si solo
cambia el tipo, no se reentrena.

**Ausencias fuera de tus puntos.** El modelo solo sabe decir «no» en sitios
parecidos a tus ausencias. Si todas están en la misma parte de la sierra, en las
demás (otra vertiente, pinar, matorral) puede ver borreguil donde no lo hay. Para
tener ausencias repartidas por toda la zona:

1. En el panel lateral, **🎲 Generar aleatoriamente en el área**, con el área de
   estudio (por ejemplo, el Espacio Natural por encima de 2000 m) y tu
   verdad-terreno subida. Unos **200 puntos**, estrategia **Al azar, separados una
   distancia mínima**, con
   **500 m** de distancia mínima. *Lejos de tus puntos de campo* puede quedarse en
   0: algunos puntos caerán junto a los tuyos (en Sierra Nevada, unos 30 de cada 200),
   y basta con dejarlos sin marcar.
2. Ejecuta el análisis.
3. En **Revisar puntos**, mira cada punto `rand_…` con la ortofoto y contesta
   **¿Es borreguil?**: **No** si no lo es; **Sí** si lo es, revisando su tipo en el
   mismo sitio; **No estoy seguro**, o nada, si no lo tienes claro. Después,
   **💾 Guardar la revisión**.
4. Al terminar, descarga `puntos.geojson`: guarda todo lo que has marcado.

Son ausencias revisadas por ti, no inventadas: la app no da ningún punto por
ausencia sin que una persona lo haya mirado.

El tipo (ambiente, humedad y pureza) se revisa en el mismo formulario, después de
decir que sí (ver [sección 10](#10-clasificación-jerárquica-del-tipo)). Encima de la
tabla, un contador dice cuántos tipos llevas revisados: son las muestras del mapa de
tipos, que se hace en la pestaña **🌿 Mapa Modelo RF**.

El mapa del punto seleccionado usa la **ortofoto** de fondo, dibuja el punto sin
relleno y marca en amarillo su **píxel Sentinel-2 de 10 m**. Con el control de capas
puedes cambiar a Google Satélite, que admite más zoom.

Al **quitar puntos del análisis** la selección se vacía: así ninguna acción
posterior recae sobre un punto que no has elegido.

**Nombres largos y decimales.** Si los nombres de tus puntos vienen del nombre de
la capa (`all_borreguil_lagunas500ptos_epsg25830.41`), en la tabla y en los avisos
salen solo con su número (`41`), y en el mapa y junto a él, como «Punto 41». Otros
nombres muy largos se abrevian en la tabla conservando el final, que es lo que los
distingue. El nombre completo está en todas las descargas. Los números se muestran
con **3 decimales** en la ficha del punto, en la tabla, en los globos del mapa y en
el Excel (las coordenadas, con 5, porque con 3 se moverían unos 100 m); los ficheros
guardan el valor completo.

**Las columnas de la tabla.** Además de la decisión, la probabilidad y el tipo, a la
derecha están las **3 variables que más pesan en el modelo** (al pasar el ratón por
su nombre se ve cuánto pesa cada una) y la pendiente. La cuenca no sale en la tabla:
sigue en el punto, para la validación por zonas, y en las descargas.

### Resumen de los puntos
Lo que ha encontrado la app en tus puntos, de arriba abajo (hasta el 25-11-2026 eran
dos pestañas, «Probabilidades» y «Tipos y distribuciones»):

1. **Probabilidad de ser borreguil**: el histograma (ver abajo).
2. **Recuento de puntos por categoría de decisión**: cuántos hay de cada decisión, de
   más a menos borreguil (borreguil verificado, probable, posible, incierto, dudoso, no
   borreguil, no borreguil verificado), con el número y el porcentaje al final de cada
   barra. Cada barra lleva el color de esa decisión en «Mapa de los puntos».
3. **🌿 Tipos de borreguil**: ver más abajo.

**La probabilidad.** Histograma de la probabilidad de ser borreguil que da el Random
Forest a cada punto, en tramos de 0,050. Debajo, la pestaña explica cómo leerlo
(abierto por defecto):

- **A la izquierda**, los puntos que el modelo ve claramente como no borreguil; **a la
  derecha**, los que ve claramente como borreguil; **en el centro**, los dudosos.
- **Forma de U** (dos grupos separados, centro casi vacío) = el modelo discrimina bien.
  **Una joroba en el centro** = no está distinguiendo; revisa puntos y reentrena, o
  aporta verdad-terreno.
- **Rayas rojas**: el umbral en uso (por encima, borreguil) y 0,700, una referencia
  fija de «muy probable».
- **Cifras**: la mediana, cuántos puntos salen borreguil con tu umbral, cuántos pasan de
  0,700 y el total.
- **Para el umbral**: si cae en un valle vacío, moverlo apenas cambia nada; la pestaña
  dice cuántos puntos, como mucho, cambiarían si lo movieras 0,150 arriba o abajo. Otsu
  busca ese valle.
- **Para revisar**: los puntos del centro son los primeros que conviene mirar en
  «Revisar puntos».
- Si el modelo se entrenó con esos mismos puntos, la U sale más limpia de lo que
  saldría con puntos nuevos.

**Los tipos.** Primero, tres gráficas de barras: cuántos borreguiles hay de cada
**ambiente**, de cada **humedad** y de cada **pureza**, con el número al final de cada
barra y las tres en la misma escala. Cada categoría lleva **su color, el mismo en toda
la app** (gráficas y mapa): arroyo verde, laguna azul, ladera amarillo verdoso; húmedo
verde azulado y seco naranja; puro verde azulado, mixto-agua azul y mixto-roca marrón;
y **no borreguil, rojo oscuro**, también en el mapa del Random Forest.
Los que no tienen propuesta en un nivel salen en gris como «sin propuesta (—)»; sin
Earth Engine, por ejemplo, no hay pureza. Debajo, los **tres niveles juntos** (la
clasificación jerárquica) y el recuento por combinación completa; esa vista solo sale
si están los tres niveles. Al final, cómo propone la app cada nivel.

### Variables del modelo
En qué se fija el modelo (hasta el 25-11-2026, «Peso de las variables» y la parte de
abajo de «Tipos y distribuciones»):

1. **El peso de cada variable**: el ranking de importancia del modelo activo, con el
   rango típico de borreguil.
2. **Las que más pesan, grupo a grupo**: las variables que más pesan en el modelo
   usado, una por cada grupo de variables parecidas, de más a menos, con el peso del
   grupo junto al nombre («Altitud (m) · 8 %»), comparadas entre grupos de puntos.

Encima del gráfico se elige:

| Control | Opciones |
|---|---|
| **Cómo verlo** | **Cajas** (mediana, cuartiles p25–p75, bigotes y atípicos) · **Curvas de distribución** (cómo se reparten los valores de cada grupo: si apenas se tocan, la variable separa bien) · **Diferencia estandarizada de medianas** (cuánto se aparta la mediana de cada grupo de la de los no borreguiles, en desviaciones típicas: a la derecha, más; a la izquierda, menos) · **Dos variables a la vez** (cada punto en el plano de dos variables) |
| **Comparar** | **Borreguil frente a no borreguil** · **Por humedad** (húmedo, seco y no borreguil) · **Por ambiente** (arroyo, laguna, ladera y no borreguil) |
| **Cuántas variables** | 4, 8 (de entrada) o 12 |

En **Dos variables a la vez** se eligen los dos ejes entre **todas** las variables de los
puntos: primero las del modelo, de más a menos peso, y después las que el modelo no usa.
De entrada, las dos que más pesan. Al pasar el ratón por cualquier gráfico salen sus
cifras con 3 decimales. En las curvas y en las diferencias no se dibujan los grupos con
menos de 5 puntos: su mediana o su forma engañarían. El grupo **No borreguil** tiene
solo los puntos que la app da como no borreguil (verificados o no); los inciertos y los
dudosos no se dibujan, y debajo del selector se dice cuántos son.

### Descargas
- `classification.csv` y `Clasificacion_puntos.xlsx` — la tabla completa, con
  todas las variables.
- `puntos.geojson` — los puntos con sus **etiquetas** (decisión, verdad-terreno y
  tipo revisado, con las columnas de la plantilla de datos de campo), sin las
  variables. Es el fichero para **volver a cargar tu trabajo** en otra sesión o
  abrirlo en QGIS.
- `mapa.html` — mapa autónomo que se abre en cualquier navegador.
- `area_estudio.geojson` — el polígono del área, si la definiste.

> 🌿 El **mapa de borreguiles** no es una de estas cinco: tiene su propia pestaña,
> **🌿 Mapa Modelo RF**, arriba del todo, junto a **📍 Verdad terreno**. Ver
> [Mapa de tipos con Random Forest](#mapa-de-tipos-con-random-forest).

---

## 10. Clasificación jerárquica del tipo

Cada borreguil detectado se etiqueta en **tres niveles anidados**:

```
AMBIENTE (dónde está)          HUMEDAD           PUREZA (qué hay en el píxel)
├── arroyo   (junto a río)       ├── húmedo        ├── puro
├── laguna   (< 120 m de laguna) └── seco          ├── mixto-agua
└── ladera   (resto)                               └── mixto-roca
```

- **Ambiente** — **laguna** si está a menos de 120 m de una laguna conocida o se
  ve agua cerca; si no, **arroyo** si está a 30 m o menos de un **río o arroyo de
  OpenStreetMap** (`waterway` = `river` o `stream`; no cuentan las acequias, los
  canales ni las fuentes); si no, **ladera**. Los ríos y arroyos de Sierra Nevada van
  con la app (`cauces_sierra_nevada.geojson`), así que allí no hace falta internet;
  en otras zonas se descargan de OpenStreetMap (servicio Overpass) al analizar. Si
  no se pueden descargar, la app lo avisa y «arroyo» se decide solo por la pendiente
  (menos de 7°), que se queda corta: junto a un arroyo suele haber pendiente.
  Probado en los 372 borreguiles de Sierra Nevada (23-11-2026): de los que están
  junto a un cauce de OpenStreetMap, la pendiente solo llamaba arroyo a uno de cada
  siete, y los cauces deducidos del relieve de 30 m van en escalera y a veces se
  separan de la vaguada.
- **Humedad** — por NDMI.
- **Pureza** — **igual que la pureza dentro del píxel del mapa**: la app estima
  cuánto **verde** hay en el píxel de 10 m de Sentinel-2 que contiene el punto, con
  su NDVI de inicio de verano: 0 % si vale 0,080 o menos, 100 % si vale 0,600 o más
  (o la mediana de los puntos que has revisado como *puro*, si hay al menos 5) y en
  proporción entre los dos. Es **puro** si llega al 75 %; si no, **mixto-agua** si
  el píxel es agua o toca un píxel de agua (NDWI medio de junio a septiembre mayor
  que 0), y **mixto-roca** en otro caso. Esos datos se leen con **Earth Engine**;
  con Planetary Computer, o si la lectura falla, la pureza se queda sin propuesta
  (—) y la pones tú al revisar.

> ℹ️ **Cuánto acierta.** Comparada con la ortofoto infrarroja de Andalucía de
> 0,25 m en 372 borreguiles de campo de Sierra Nevada (24-11-2026), la propuesta
> acierta la pureza en el **82 %** de los puntos. La manera de antes (relativa a los
> borreguiles de cada ejecución) acertaba el 62 %, menos que decir «puro» a todos
> (72 %), y se cambió por esta. Donde más falla es con el agua: casi nunca propone
> mixto-agua. Sigue siendo una propuesta: la que cuenta para el modelo es la que
> revisas tú.

Los umbrales del ambiente y la humedad se ajustan en `borreguil_pipeline.py` →
`HIER_THRESH`; los de la pureza son los del mapa por defecto.

### Revisar el tipo: de propuesta de la app a dato revisado

El tipo que calcula la app es una **propuesta**: sale de reglas sobre las mismas
variables que usa el modelo (la humedad, por ejemplo, es un umbral de NDMI). Sirve
para orientarse, pero **no como etiqueta de entrenamiento**: un modelo entrenado
con ella solo volvería a aprender la regla. Para que el tipo sea un dato hay que
revisarlo mirando el terreno.

**Cómo se revisa**

1. Pestaña **Revisar puntos** → selecciona uno o varios borreguiles.
2. Mira el punto en el mapa de la izquierda (ortofoto y píxel de 10 m en amarillo) o
   usa tus datos de campo.
3. En el formulario de la derecha, **¿Es borreguil? → Sí**. Se enciende el tipo, con
   la propuesta de la app puesta en cada nivel («arroyo (propuesta de la app)»).
   Déjala si la ves bien, cámbiala si no, o elige **No lo he podido ver**.
4. **💾 Guardar la revisión**. El punto queda como borreguil verificado y quedan
   revisados **los niveles elegidos**, con su fecha, y se anota que se revisó con la
   ortofoto (`method = orthophoto`).

**Cada nivel cuenta por separado.** Lo que dejes en **No lo he podido ver** no se da
por revisado: si solo sabes la humedad de un punto, deja los otros dos así; ese punto
ya sirve para el modelo de humedad, aunque no para el de ambiente ni el de pureza. Un
nivel ya revisado aparece elegido al volver al punto, y otro nivel se suma a los que
había. Ojo: con un solo punto, la propuesta viene puesta; si guardas sin mirarla, se
da por revisada.

**↩ Deshacer** deshace la última revisión guardada: el punto vuelve a como estaba.

Con **varios puntos** seleccionados, lo que elijas se guarda en todos; lo que dejes en
«No lo he podido ver» se queda como estaba en cada uno.

**Criterios** — decide siempre con la misma regla para todos los puntos:

| Nivel | Categorías | Cómo decidir |
|---|---|---|
| Ambiente | arroyo · laguna · ladera | Dónde está el prado: junto a un cauce, junto a una laguna o en ladera |
| Humedad | húmedo · seco | Un único criterio para toda la campaña (anótalo) |
| Pureza | puro · mixto-agua · mixto-roca | Qué hay **dentro del píxel de 10 m**: solo prado, agua, o roca/suelo desnudo |

**Qué se guarda de cada punto** (en la sesión, en la tabla, el CSV y el Excel)

| Columna | Contenido |
|---|---|
| `ambiente`, `humedad`, `pureza` | El valor vigente: el revisado si lo hay; si no, la propuesta |
| `ambiente_regla`, `humedad_regla`, `pureza_regla` | Lo que propuso la app (se recalcula en cada ejecución) |
| `ambiente_revisado`, `humedad_revisado`, `pureza_revisado` | `si` en cada nivel que ha revisado una persona |
| `tipo_revisado` | `si` cuando están revisados los tres |
| `tipo_revisado_fecha` | Cuándo se revisó por última vez |
| `ndvi_early_pixel`, `agua_pixel_vecinos` | Con lo que se propone la pureza: el NDVI de inicio de verano del píxel de 10 m del punto y `1` si ese píxel o uno de sus vecinos es agua |
| `verde_estimado` | Cuánto verde hay en ese píxel, de 0 a 1, según su NDVI |

En la tabla, la columna **tipo revisado** dice «sí» con los tres niveles y, si no,
cuáles («humedad»).

Lo revisado **no se pisa nunca**: ni al reentrenar ni aunque el modelo deje de
clasificar el punto como borreguil.

Encima de la tabla hay un contador: cuántos borreguiles están revisados (y cuántos en
cada nivel) y en cuántos se **corrigió** la propuesta. Si revisas muchos y no corriges
ninguno, la app lo avisa: un tipo aceptado sin mirar sigue siendo la regla de la app.

**Conservar la revisión entre sesiones** — descarga `puntos.geojson` y, la próxima
vez, cárgalo en *Tus puntos* o en *puntos de campo para comprobar el modelo*. Va con las columnas de
la [plantilla de datos de campo](#plantilla-de-datos-de-campo): `presence`,
`doubtful` y, de cada nivel **revisado**, `environment`, `moisture` y `purity` (la
propuesta de la app no se guarda: se recalcula), más `method`, `date` y, si se
indicaron, `observer`, `expertise` y `certainty`. Al volver a
cargarlo, cada nivel revisado vuelve como tal. Los ficheros guardados con versiones
anteriores de la app (con `Borreguil` y `tipo_revisado`) se siguen leyendo igual. Si
un punto llega marcado como revisado pero con una categoría que no existe (por
ejemplo, editada a mano), la app lo avisa y ese nivel no cuenta como revisado.

### Plantilla de datos de campo

Para que los datos de distintos equipos se puedan juntar, la app propone una
plantilla común. En la pantalla de inicio, **⬇ Descargar la plantilla de datos de
campo (Excel)** la descarga (`field_points_template.xlsx`, con la explicación de cada
columna en la segunda hoja), y **▶ Probar con 20 puntos de ejemplo de Sierra Nevada**
analiza sus 20 puntos (12 borreguiles y 8 sin borreguil) sin subir nada. En la carpeta
`plantilla` del código van además `example_sierra_nevada.csv` y `.geojson` y
`definitions_EN.md` (qué significa cada valor; **borrador**).

Una fila por punto. Solo son obligatorios el punto y `presence`; el resto se rellena
si se sabe:

| Columna | Valores | ¿Obligatoria? |
|---|---|---|
| `lon`, `lat` | grados decimales, WGS84 | sí |
| `presence` | `1` borreguil · `0` no borreguil | sí |
| `id` | cualquiera (si falta, la app pone uno) | no |
| `doubtful` | `1` si es dudoso | no |
| `environment` | `stream` · `lake` · `slope` | no (solo si `presence` = 1) |
| `moisture` | `wet` · `dry` | no (solo si `presence` = 1) |
| `purity` | `pure` · `mixed-water` · `mixed-rock` | no (solo si `presence` = 1) |
| `method` | `field` · `orthophoto` | recomendada |
| `date` | `AAAA-MM-DD` | recomendada |
| `observer` | texto: quién observó o revisó el punto | no |
| `expertise` | `student` · `manager` · `researcher` | no |
| `certainty` | `high` · `medium` · `low`: lo segura que es la respuesta de `presence` | no |
| `notes` | texto | no |

- Los valores del tipo valen también en español (`arroyo`, `laguna`, `ladera`,
  `húmedo`, `seco`, `puro`, `mixto-agua`, `mixto-roca`) y la columna puede llamarse
  `ambiente`, `humedad` o `pureza`. Al cargar, la app dice qué ha leído y avisa de
  los valores que no entiende.
- Cada nivel del tipo que traiga un punto cuenta como **revisado** en ese nivel. Lo
  que traiga un punto con `presence` = 0 se ignora: lo que no es borreguil no tiene
  tipo.
- Se puede subir como CSV (columnas `lon` y `lat` en grados; separado por comas o por
  punto y coma, y con coma decimal, como lo guarda Excel en español), GeoJSON,
  Shapefile o KML.

### Mapa de tipos con Random Forest

La pestaña **🌿 Mapa Modelo RF** está arriba del todo, junto a **📍 Verdad
terreno**, y lleva de tus puntos al mapa de borreguiles y, si quieres, de su tipo.
Antes de ejecutar un análisis solo explica qué hace falta. Después enseña, de arriba
abajo:

- **Qué quieres cartografiar**: viene elegido **borreguil sí o no**, el primer
  paso. Para el tipo, después, **ambiente y humedad**, que es el tipo completo
  (arroyo, laguna o ladera y, dentro de cada uno, húmedo o seco), o solo la
  **humedad**. Lo que elijas vale para todos los pasos; hay un modelo para cada
  opción.
- **Una línea de estado**, con un rótulo por paso: hecho (verde), pendiente (gris)
  o algo que atender (naranja), siempre con su texto.
- **Los pasos, cada uno en su pestaña**, en orden: **1 · Datos**, que responde a
  dos preguntas antes de entrenar nada, **2 · Tabla de entrenamiento**,
  **3 · Modelo**, **4 · Mapa** y **5 · Peso**, que dice cuánto pesa cada variable en
  el modelo. Queda por llegar la exportación completa de cada ejecución.

**¿Qué se puede cartografiar?** Tres cosas, cada una con su modelo:

| Opción | Categorías |
|---|---|
| **Borreguil sí o no** *(la que viene elegida)* | borreguil · no borreguil |
| Humedad | húmedo · seco |
| **Ambiente y humedad** | arroyo, laguna o ladera y, dentro de cada uno, húmedo o seco: `arroyo · húmedo`, `arroyo · seco`, `laguna · húmedo`, `laguna · seco`, `ladera · húmedo`, `ladera · seco` |

Desde el 24-11-2026 no se ofrecen el **ambiente** ni la **pureza** solos. En el
ambiente solo, el modelo no ve dónde están las lagunas ni los cauces y podía poner
«laguna» lejos de cualquier laguna; en «Ambiente y humedad» el ambiente sale de la
cercanía a lagunas y cauces. La pureza sola repetía la **pureza dentro del píxel**
que se añade al mapa en **4 · Mapa**, y en un mismo mapa salían dos purezas
distintas.

**El mapa en dos pasos: sí o no, y después el tipo.** El modelo de **borreguil sí o
no** aprende de **todos los puntos marcados en campo** como borreguil o no
(`Borreguil = si` o `no`), **tengan o no el tipo revisado**; un borreguil con el tipo
revisado también cuenta. Su mapa ya es el de borreguil sí o no. Para el tipo se
entrena después otro nivel; en **1 · Datos** se elige **quién decide si un píxel es
borreguil**:

| Quién decide | Qué aprende el modelo del tipo |
|---|---|
| **El modelo «Borreguil sí o no»** *(recomendado; viene elegido)* | Solo el tipo, de los borreguiles con el tipo revisado. En el mapa, si un píxel es borreguil lo decide el modelo de sí o no (con su umbral) y este le pone el tipo. Hay que entrenar los dos |
| **Este mismo modelo, con las ausencias de campo** | También `no borreguil`, y decide él solo. Solo cuentan los borreguiles con el tipo revisado |

Así, para saber si hay borreguil se aprovechan todos los puntos de campo, aunque
todavía no se haya revisado su tipo. Las probabilidades del mapa en dos pasos son
`1 − p` para `no borreguil` y `p ×` la del tipo para cada tipo, con `p` la
probabilidad de borreguil del modelo de sí o no.

**Ambiente y humedad** da el tipo completo, y en él cada cosa sale de donde puede
salir:

- **El modelo** decide si cada píxel es borreguil y si es **húmedo o seco**. Por eso
  en **1 · Datos**, **2 · Tabla** y **3 · Modelo** se cuentan y se validan esas
  categorías (`no borreguil`, `húmedo`, `seco`), y solo hacen falta muestras de
  húmedo y de seco, no de las seis combinaciones.
- **El ambiente** (arroyo, laguna o ladera) **no lo decide el modelo**: lo pone el
  mapa por **cercanía de cada píxel a lagunas y cauces** (ver «El ambiente, por
  cercanía», en el apartado del mapa). El ambiente dice *dónde está* el borreguil, y
  eso no se ve dentro del píxel: un modelo que solo conoce las variables de cada
  píxel daba por «laguna» borreguiles que estaban a cientos de metros de cualquier
  laguna.

La categoría del mapa es la compuesta («laguna · húmedo»). Las leyendas van por
ambientes, cada uno con sus dos colores, **lo húmedo en un tono frío** (verde azulado
en el arroyo, azul en la laguna, verde oliva en la ladera) **y lo seco en uno cálido**
(mostaza, naranja y teja); y debajo de la tabla de hectáreas va la suma de cada
ambiente.

**¿Qué puntos sirven como muestra?** Para lo que se haya elegido, la app cuenta:

| Entra como muestra | Con qué categoría |
|---|---|
| Un punto con **ese nivel del tipo revisado** (los demás niveles pueden estar sin revisar; en un nivel combinado, revisadas sus dos partes) | La que dejó la persona en ese nivel (en **borreguil sí o no**, `borreguil`) |
| En **borreguil sí o no**, un punto marcado en campo como borreguil (`Borreguil = si`), aunque no tenga el tipo revisado | `borreguil` |
| Una **ausencia de campo** (`Borreguil = no`), si se incluye | `no borreguil` |

El resto no entra, y se dice por qué: borreguil con ese nivel del tipo sin revisar, punto sin
etiqueta de campo (lo que diga el modelo no cuenta), dudoso, ausencia de campo que
además tiene un tipo revisado, o dos muestras en el mismo píxel de 10 m (si
coinciden en categoría se queda una; si no, se apartan todas). **No se inventan
ausencias ni se cambia la categoría de ningún punto.** La tabla punto por punto se
puede descargar (`muestras_<nivel>.csv`).

Cada categoría tiene un **código fijo**, el que llevará el mapa: 0 es siempre
`no borreguil` y el resto sigue el orden de la app (por ejemplo, en borreguil sí o
no con el ambiente: 1 arroyo, 2 laguna, 3 ladera; en ambiente y humedad: 1 y 2 el arroyo húmedo y el
seco, 3 y 4 la laguna, 5 y 6 la ladera). No depende de qué categorías haya en cada
ejecución.

Para poder entrenar hacen falta **al menos dos categorías**, y que cada categoría
que tenga alguna muestra llegue a **5**; con menos de 30 por categoría la app avisa
de que el resultado será poco fiable. Con seis categorías es más fácil que alguna
se quede corta: la app dice cuál y cuántas muestras tiene.

**¿Son bastantes puntos?** En **borreguil sí o no**, debajo del recuento, la app dice
dónde están tus muestras en la curva de Sierra Nevada. Esa curva se hizo con los 948
puntos revisados del mapa de Sierra Nevada (315 borreguiles) y este mismo
procedimiento: se entrenó con 20, 30… hasta 682 puntos elegidos al azar y se midió el
acierto en zonas que el modelo no había visto, 10 veces para cada número.

| Puntos revisados | Borreguiles | Ausencias | AUC en zonas nuevas | |
|---|---|---|---|---|
| 20 | 7 | 13 | 0,896 | por debajo de 50, el acierto cambia mucho según qué puntos sean |
| 50 | 17 | 33 | 0,930 | |
| **100** | **33** | **67** | **0,951** | **mínimo**: a menos de 0,02 del de todos los puntos |
| **250** | **83** | **167** | **0,961** | **recomendado**: a menos de 0,01 |
| 682 (todos) | 226 | 456 | 0,969 | |

Se llega a un nivel con los borreguiles **y** las ausencias de esa fila; si falta
algo, la app dice cuántos borreguiles y cuántas ausencias faltan (las ausencias se
cuentan aparte desde el 25-11-2026: antes, 100 borreguiles con 6 ausencias «llegaban
al mínimo»). La curva se hizo con 1 borreguil de cada 3 y no se probaron otras
proporciones: los borreguiles y las ausencias de cada nivel son los que había en ella.
Las **ausencias** son los sitios que has comprobado que no son borreguil: `presence` = 0
en tu fichero, **No** en «¿Es borreguil?» al revisar un punto o **✗ Aquí
no hay** al pinchar en el mapa; la app no se las inventa. Para buscarlas en sitios que
no has muestreado, genera puntos al azar con **Lejos de tus puntos de campo** (300 m) y
revísalos. Si faltan ausencias, el aviso lo recuerda.

Es una referencia: en una montaña más variada pueden hacer falta más, y los
puntos repartidos por toda la zona enseñan más que los concentrados en pocos valles.
Lo que vale en cada zona es la curva con sus propios puntos, al final de **3 · Modelo**
(**¿Mejoraría con más puntos?**, más abajo). La curva de Sierra Nevada, con su
explicación, también está en **📖 Documentación y recomendaciones**. El cálculo está
en `claude-borreguiles/curva_puntos/` (fuera de la app).

> ⚠️ Sin clase `no borreguil`, el modelo solo distingue tipos: cualquier píxel que
> se le dé recibirá uno de ellos. Por eso, para hacer el mapa con un modelo así, la
> app exige una **máscara de zonas candidatas** (ver «El mapa», más abajo); por sí
> solo no demuestra presencia frente a ausencia.

**¿Qué variables se pueden usar?** Solo las que existen como **mapa continuo**, que
son las que luego permiten predecir sobre el territorio: 34, calculadas en Earth
Engine a partir de Sentinel-2 (29) y del modelo de elevaciones (5). Vienen todas
marcadas y se pueden quitar. Las texturas de la imagen aérea y el radar Sentinel-1,
que el detector sí usa, no están porque hoy solo se calculan punto a punto.

Las **estadísticas de junio a septiembre** (media, mínimo, máximo y variabilidad de
WAVI, albedo, NDMI y NDWI) se calculan **en cada año** y después se hace la **media
entre los años** que tienen dato. Así, el mínimo es lo bajo que llega la variable en
un verano normal, y la variabilidad, lo que cambia dentro del verano. Con todos los
años juntos, el mínimo y el máximo serían los de una sola imagen en nueve veranos
(basta una sombra de nube), y la variabilidad mezclaría el cambio dentro del verano
con las diferencias entre años. Una variabilidad necesita al menos dos imágenes
válidas en el año; si no, ese año no cuenta para ella.

> ℹ️ El **detector** (pestaña **📍 Verdad terreno**) calcula sus variables de otra
> manera, la misma con la que se entrenó el modelo de Sierra Nevada (ver la sección
> 8): cambiarla sería darle números distintos de los que aprendió. Por eso el valor
> de, por ejemplo, `albedo_min` de un punto no es el mismo en la tabla del detector
> que en la tabla de entrenamiento del mapa.

**La tabla de entrenamiento.** El botón **📥 Leer las variables… en Earth Engine**
lee el valor de cada variable en el **píxel de 10 m** de cada muestra. Es el mismo
píxel que después se clasificará: por eso no se usan los valores que ya tienen los
puntos, que son medias de una ventana de 50–60 m alrededor del punto. Usa el
proyecto de Earth Engine y los años de Sentinel-2 del panel lateral (si allí está
elegido Planetary Computer, el último proyecto con el que Earth Engine funcionó). Con 21 muestras y los años 2017-2025 tarda alrededor de un minuto.

Lo leído se conserva mientras no cambien los años: añadir muestras solo lee las
nuevas, y quitar o poner variables no obliga a leer nada. (Si una versión nueva de la
app cambia la manera de calcular alguna variable, lo leído antes deja de valer y la
app pide leerlo otra vez: muestras y mapa tienen que salir del mismo cálculo.) La app
enseña:

- **Cuántas muestras entran en la tabla** y cuáles no. Una muestra entra solo si
  tiene dato en todas las variables elegidas; las demás se apartan diciendo por qué
  (píxel sin dato de alguna variable, fuera de las capas, o lectura fallida que se
  puede reintentar). No se rellena ningún hueco.
- **Control de calidad de las variables**: constantes o casi constantes, repetidas,
  la misma con el signo cambiado, muy correlacionadas (|r| ≥ 0,95) y pocas muestras
  para tantas variables. **Solo avisa; no quita ninguna.** Quitar o no es decisión
  tuya: desmarcándola en **1 · Datos** o con la opción **Quitar las variables
  repetidas** (VIF o PCA; ver más abajo).
- **Desequilibrio entre categorías**, cuando la más abundante triplica a la más
  escasa.

La tabla se descarga como `tabla_entrenamiento_<nivel>.csv`: una fila por muestra y
una columna por variable, en el orden exacto que verá el modelo.

**Variables repetidas (VIF o PCA).** En **1 · Datos**, debajo de las variables, está
la opción **Quitar las variables repetidas**: *No quitar ninguna* (la que viene
elegida), *Con el VIF* o *Con un PCA*, con sus ajustes y las variables que se quedan
siempre. Se calcula con los valores de las muestras, así que primero hay que leerlos
en **2 · Tabla de entrenamiento** (la lectura trae siempre todas las variables).
Después, el botón **✔ Aplicar el VIF** (o **✔ Aplicar el PCA**) enseña ahí mismo, en
**1 · Datos**, qué variables se quedan, la tabla del análisis y, en **Por qué,
variable a variable**, una frase para cada una. Hasta que se aplica no se quita
ninguna.

Al aplicarlo, en **Variables predictoras** quedan **marcadas solo las que se quedan**
(«14 de 34 variables seleccionadas»), y debajo la app dice cuántas había antes. El
botón **↺ Volver a marcar las 34 de antes** las devuelve y deshace lo aplicado, por
ejemplo para entrenar con todas y comparar.

- Si cambias el umbral o pasas al otro método y vuelves a aplicar, se calcula con las
  que había marcadas **antes** de aplicarlo, no con las que dejó el anterior.
- Si cambias las casillas a mano, lo que marques pasa a ser el punto de partida.
- Si cambias el ajuste, las variables que se quedan siempre o las muestras, la app pide
  volver a aplicarlo y, mientras, usa las marcadas.

Así se puede aplicar el VIF y entrenar, volver a las de antes, aplicar el PCA y
entrenar otra vez. La tabla del final de **3 · Modelo** compara los modelos.

*Con el VIF.*
El VIF de una variable dice qué parte de ella se puede adivinar con todas las demás
juntas: 5 quiere decir el 80 %; 10, el 90 %; infinito, que se obtiene exactamente con
otras (la caída del NDVI es el NDVI de inicio menos el de fin de verano). Con la
opción activada, la app quita la variable de VIF más alto, vuelve a calcular y sigue
mientras alguna pase del umbral. Se calcula con los valores de las muestras de esa
tabla.

| Control | Qué es | Recomendado |
|---|---|---|
| Umbral del VIF | Se quitan variables mientras alguna lo pase. Cuanto más bajo, más se quitan | **10**; 5 para quedarte con menos |
| Variables que se quedan siempre | No se quitan nunca, aunque pasen del umbral (sí cuentan en el VIF de las demás) | **La altitud**, que viene marcada |

Al aplicarlo, **1 · Datos** dice cuáles se quedan («Con el VIF se quedan 14 de 34
variables: …») y enseña una tabla con cada variable: su VIF, qué pasa con ella (se
queda, se queda siempre o se quita en tal paso), qué parte de ella explican las demás
y a cuál de las que se quedan se parece más (con su correlación, r). La frase de cada
una lo resume; por ejemplo, de una que se quita: «Las demás explicaban el 98,9 % de ella
(VIF 88,324), más que el umbral (90 %). Se parece sobre todo a NDVI inicio
(r 0,994), que se queda». Desde ahí, el
modelo, el mapa y el peso usan **solo las que se quedan**, y la línea de estado lo
dice: «2 · Tabla · … variables (VIF)». Si hay menos del doble de muestras que
variables, avisa de que el VIF no es fiable. Con las 161 muestras del ejemplo y las
34 variables, el umbral 10 deja 14; el 5, 12.

*Con un PCA.* El PCA de la matriz de correlación resume las variables en unos pocos
ejes independientes. De cada eje con valor propio mayor que 1 (criterio de Kaiser) se
queda **una variable**: entre las que tienen con él una correlación de al menos 0,65
(en valor absoluto), la que mejor separa las categorías por sí sola (con dos
categorías, por su AUC; con más, por la F del ANOVA). Las variables que se quedan
siempre (la altitud) no entran en el PCA y se añaden. Al aplicarlo, **1 · Datos**
enseña un gráfico con el valor propio de cada eje (en verde los que pasan de 1), una
tabla con cada eje importante (su valor propio, su parte de la varianza, las variables
ligadas a él y la que se queda) y otra con cada variable: qué pasa con ella, su eje, su
correlación con él y lo bien que separa las categorías ella sola. La frase de cada una
dice por qué: es la que mejor separa de su eje, de su eje se queda otra que separa
mejor, o no está muy ligada a ningún eje importante. La línea de estado dice «…
variables (PCA)».

| Control | Qué es | Recomendado |
|---|---|---|
| Correlación mínima con el eje | De cada eje se elige entre las variables que llegan a ella. Más alta, menos candidatas | **0,65** |
| Variables que se quedan siempre | No entran en el PCA y se añaden | **La altitud** |

Es el procedimiento de la comparación de Sierra Nevada del 22-10-2026
(`modelo_borreguil_sn2000m_v3/comparacion_variables`), con una diferencia: allí el
PCA se hizo con 30 000 píxeles de la zona y aquí con las muestras, como el VIF. Con
las 948 muestras de aquel mapa, la app elige `ndwi_sd`, `albedo_sd` y `albedo_min`
más la altitud (5 ejes, el 85 % de la varianza); con los píxeles salieron `ndwi_sd`,
`wavi_sd` y `albedo_sd`. En aquella comparación, el PCA con la altitud (4 variables)
acertó como el VIF (14): F1 0,861 frente a 0,865 y TSS 0,792 los dos, pero
cartografió un 38 % más de borreguil (1392 ha frente a 1014). Sin la altitud, el PCA
acertaba bastante menos (F1 0,768).

El Random Forest no necesita el VIF ni el PCA para acertar: tolera bien las variables
parecidas. Sirve para tener un modelo más sencillo y un peso de las variables que
se lea mejor (dos variables que dicen lo mismo se reparten el peso y, sueltas,
parecen pesar poco). Si con menos variables acierta igual, lo dice la comparación
del final de **3 · Modelo**, que enseña los modelos entrenados con todas las
variables, con las del VIF y con las del PCA.

> 💡 `gndvi` no está entre las variables del mapeo: es `ndwi` con el signo cambiado
> (las dos se calculan con las mismas dos bandas), así que tenerlas ambas sería
> repetir la misma variable. Los modelos del detector sí las llevan las dos,
> porque se entrenaron así; no les perjudica.

La rejilla de todo el mapeo es la de Sentinel-2: UTM, 10 m. La topografía, que
está a 30 m, se interpola (bilineal) al pasarla a esa rejilla.

**El modelo.** El botón **🌲 Entrenar y validar** hace dos cosas distintas, en este
orden:

1. **Validar por zonas.** La tabla se reparte en 5 partes y cada una se predice
   con un modelo entrenado solo con las otras. Las partes se hacen por **zonas**
   (la `cuenca_id` del punto si la tiene; si no, una celda de unos 5 km), de modo
   que todos los puntos de una zona caen juntos. Los puntos vecinos se parecen
   mucho: si unos entrenaran y otros validaran, el resultado saldría mejor de lo
   que es. De aquí salen todas las cifras.
2. **Entrenar el modelo final** con todas las muestras. Es el que hace el mapa.
   No tiene cifras propias: las suyas son las de la validación.

Qué enseña la validación:

| Medida | Qué dice |
|---|---|
| **Precisión** (por categoría) | De lo que el modelo llama así, cuánto lo es |
| **Sensibilidad** o *recall* (por categoría) | De lo que lo es, cuánto encuentra |
| **F1** (por categoría) | Resume las dos anteriores |
| **F1 medio (macro-F1)** | La media del F1 de las categorías: todas pesan igual, tengan muchas muestras o pocas |
| **Exactitud equilibrada** | La media de la sensibilidad de las categorías |
| **Aciertos sobre el total** | La cifra más conocida y la más engañosa si las categorías están desequilibradas; por eso va la última |
| **Matriz de confusión** | Filas: lo que es. Columnas: lo que dice el modelo. Fuera de la diagonal, los errores |

Debajo aparecen **lo que más se confunde**, el resultado de cada una de las partes
y las **muestras mal clasificadas**. Estas últimas merecen una mirada: puede ser un
fallo del modelo o una etiqueta que conviene revisar.

**¿Borreguil o no? El umbral.** El modelo da a cada caso la probabilidad de cada
categoría. La categoría se decide en dos pasos:

1. **¿Es borreguil?** Sí, si la probabilidad de serlo llega a un **umbral**. Esa
   probabilidad es la **suma de la de sus tipos** (lo mismo que 1 − la de
   `no borreguil`).
2. **¿De qué tipo?** El más probable entre los tipos.

No se usa sin más «la categoría más probable» porque perjudica a los borreguiles: su
probabilidad se reparte entre los tipos y `no borreguil` conserva la suya entera. Un
píxel con 0,400 de `no borreguil`, 0,350 de `húmedo` y 0,250 de `seco` saldría
`no borreguil` teniendo 0,600 de serlo; y con seis tipos, peor.

En **ambiente y humedad** estos dos pasos son los del modelo (borreguil o no, y
húmedo o seco); el ambiente se añade después, en el mapa, por cercanía a lagunas y
cauces. Las cifras de la validación y el peso de las variables son, por tanto, los
de la humedad: el ambiente no entra en ellos.

El umbral se elige encima de las cifras de la validación, porque de él salen esas
cifras, el mapa y el peso de las variables. Cambiarlo no vuelve a entrenar nada:

| Cómo se elige | Qué hace |
|---|---|
| **El de mayor F1** *(recomendado; viene elegido)* | De 39 umbrales, de 0,025 a 0,975, el que mejor equilibra en la validación encontrar borreguiles y no dar por borreguil lo que no lo es |
| **El que mejor separa en la validación, TSS** | El de mayor TSS: sensibilidad + especificidad − 1, es decir, lo que acierta con los borreguiles más lo que acierta con lo que no lo es. Como hay mucho más `no borreguil` que borreguil, suele elegir un umbral bajo y el mapa se llena de pasto que no es borreguil (pasó en Sierra Nevada por encima de 2000 m) |
| **Lo fijo yo** | El valor del deslizador. 0,500 es «más probable que sí que que no» |

Debajo dice lo que da ese umbral en la validación: cuántos borreguiles encuentra
(sensibilidad), cuánto de lo que no lo es descarta bien (especificidad), el TSS y el
F1. Una gráfica y una tabla enseñan las mismas medidas con cada umbral.

- Con pocas muestras suelen **empatar** muchos umbrales seguidos. Se toma el del
  centro del tramo, que es el que más margen deja a los dos lados.
- El umbral se elige y se mide con las mismas muestras, así que esas cifras salen
  **algo optimistas**. La app lo recuerda.
- Si el modelo **no tiene la clase `no borreguil`**, no hay umbral: a cada caso le
  pone su tipo más probable.

Avisos que puede dar:

- **No se ha podido validar por zonas**: las muestras caen en menos de 3 zonas. Se
  valida al azar y el resultado saldrá mejor de lo que es; hay que repetirlo
  cuando haya muestras de más sitios.
- **Una categoría está en una sola zona**: al apartar esa zona, el modelo no ha
  visto nunca esa categoría y no puede acertarla.

**Parámetros** (plegados). Para empezar no hace falta tocarlos: los valores que
vienen puestos son los recomendados, un punto de partida razonable y no el
resultado de una búsqueda. En la app, el **?** que hay junto a cada uno explica qué
es y qué valor conviene:

| Parámetro | Qué es | Recomendado |
|---|---|---|
| Nº de árboles | El modelo es un conjunto de árboles de decisión que votan. Con más, el resultado es más estable, pero tarda más | **500** (entre 300 y 1000 va bien) |
| Profundidad máxima | Cuántas preguntas seguidas puede hacer cada árbol antes de decidir | **Sin límite** (0). Un límite de 5 a 10, solo con muy pocas muestras y una validación mala |
| Muestras mínimas por hoja | El tamaño mínimo del grupo final en que cada árbol reparte las muestras. Con 1 puede aprenderse cada muestra de memoria | **2** con decenas de muestras por categoría; de 3 a 5 con cientos |
| Variables probadas en cada división | Cada árbol prueba en cada división solo unas cuantas variables elegidas al azar; así salen árboles distintos, que juntos aciertan más | **Raíz cuadrada** del nº de variables (5 con las 34) |
| Semilla | Fija el azar: mismos datos y misma semilla, mismo modelo | **42**; ningún número es mejor que otro. Anótala con los resultados |
| Si las categorías están desequilibradas | Cómo evitar que el modelo elija siempre la categoría con más muestras | **Dar más peso a las escasas** |

Para el desequilibrio hay cuatro opciones: dar más peso a las categorías escasas,
usar menos muestras de las abundantes (como mucho 3 veces la más escasa, elegidas
con la semilla), las dos cosas, o no compensar. **No se generan muestras
artificiales.** Con cualquiera de ellas se valida con todas las muestras.

Hay un modelo por cada opción: puedes tener entrenados a la vez el de borreguil sí o
no, el de humedad y el de ambiente y humedad. Si después cambias las
muestras o las variables, la app avisa de que el modelo es de la tabla anterior
hasta que vuelvas a entrenar.

**¿Mejoraría con más puntos?** El mínimo y lo recomendado de **1 · Datos** salen de
Sierra Nevada. Debajo del modelo final, **📈 Calcular la curva con mis puntos** hace
esa curva con **tus** puntos y tu modelo:

1. Repite la validación por zonas del modelo, con las mismas partes.
2. En cada parte, el modelo aprende solo con el **25, el 50 o el 75 %** de los puntos
   de las otras zonas, elegidos al azar con la misma proporción de cada categoría. Se
   hacen 3 sorteos de cada fracción.
3. La zona apartada se predice siempre entera: lo único que cambia es con cuántos
   puntos aprende. El **100 %** es la validación del propio modelo.

Entrena 45 modelos con 5 partes. Con unos mil puntos tarda medio minuto.

La app enseña una gráfica y una tabla: los puntos con los que aprende el modelo, los
borreguiles entre ellos, el **AUC en zonas nuevas** (sin «no borreguil», el F1 medio)
con el peor y el mejor sorteo, y el F1 medio. Después mira lo que pasa **de la mitad de
los puntos a todos**, es decir, con el doble:

| Si de la mitad a todos el acierto… | La app dice | Qué hacer |
|---|---|---|
| sube **0,010 o más** | **Todavía sube** | Con el doble de puntos que ahora, cabe esperar que suba algo parecido, o algo menos. Busca sobre todo en las zonas y los ambientes con pocos puntos, borreguiles y ausencias |
| sube menos de 0,010 | **Ya casi no sube** | Más puntos parecidos apenas cambiarán el mapa. Si el acierto es bajo, lo que falta no es cantidad: revisa las muestras mal clasificadas, busca puntos de sitios distintos o prueba otras variables |

Con los 948 puntos del mapa de Sierra Nevada, el AUC pasa de 0,959 (un cuarto de los
puntos) a 0,968 (todos), y de la mitad a todos sube 0,004: **ya casi no sube**, como en
la curva de referencia.

- Las variables son las del modelo, como en su validación. El VIF no mira si los puntos
  son borreguil o no, así que elegirlas con todos no da ventaja a ninguna fracción.
- Con el criterio de mayor F1 o de mayor TSS, cada sorteo elige su propio umbral, igual
  que el modelo con todos los puntos; con un umbral fijado a mano, se usa ese.
- Con pocos puntos la curva baila: mira la franja entre el peor y el mejor sorteo antes
  de sacar conclusiones.
- La curva se guarda con el modelo: al volver a entrenarlo, desaparece y hay que
  calcularla de nuevo.

**Con qué variables acierta más.** Al final de **3 · Modelo**, en cuanto hay dos
modelos del mismo nivel entrenados en la sesión con las mismas muestras y distintas
variables (por ejemplo, con todas y con las que deja el VIF), una tabla los compara:
las variables de cada uno, sus muestras y su **F1 medio** de la validación por zonas.
El de ahora, el que hace el mapa, va marcado con ▶. Una diferencia de unas
centésimas puede ser azar: con pocas muestras, una sola cambia la cifra. Para quedarte
con otro, cambia la opción de las variables repetidas en **1 · Datos** y vuelve a
entrenar.

**El mapa.** El paso **4 · Mapa** aplica el modelo final a todos los píxeles de
10 m de una zona. Los lee de Earth Engine con los mismos años y en la misma rejilla
que las muestras, de modo que el modelo ve en cada píxel exactamente lo que vio al
entrenar. Hace falta que el modelo esté al día (si la tabla ha cambiado, pide volver
a entrenar).

*Qué zona.* Hay hasta cuatro opciones:

| Zona | Cuándo aparece |
|---|---|
| **Un rectángulo que elijo yo** | Siempre. Se da el centro (también pinchando en el mapa de debajo) y el ancho y el alto en km |
| **El rectángulo que ocupan las muestras** | Siempre. Con 100 m de margen |
| **El área de estudio** | Si la ejecución tenía un área de estudio. Solo se clasifica lo que cae dentro |
| **Todo lo que cubre la máscara** | Si has subido una máscara |

Antes de lanzarlo, la app dice **cuánta superficie es y cuánto tardará** (y, en
pequeño, en cuántos trozos se lee). Earth Engine sirve unos 6 km² por minuto y no va más deprisa pidiendo
varios trozos a la vez, así que conviene **empezar por un rectángulo de 2 × 2 km**
(un minuto), ver cómo sale y ampliar después. Como orientación: 100 km² son unos
20 minutos y el Parque Nacional entero, más de dos horas. El máximo de una sola vez
son 400 bloques (unos 2 600 km²). Los mapas grandes también gastan más cuota de tu
proyecto de Earth Engine.

*Si el mapa es largo.* Se lee por bloques de 2,5 km de lado, de uno en uno, y cada
bloque leído se guarda en el ordenador mientras la app esté abierta. Por eso:

- **Mientras se hace el mapa no uses la app**: cualquier clic lo interrumpe.
- Si se interrumpe, o Earth Engine falla, **no se pierde lo leído**: vuelve a pulsar
  el botón y sigue donde iba. No queda nunca un mapa a medias.
- El mapa de **otro nivel sobre la misma zona** (por ejemplo, humedad después de
  borreguil sí o no) no vuelve a pedir nada si usa las mismas variables: sale en segundos.
- Los bloques que quedan enteros fuera de la zona o de la máscara ni se piden.

*La máscara de zonas candidatas.* Es un fichero de polígonos (KML, GeoJSON,
GeoPackage o shapefile) con las zonas donde puede haber borreguil; solo se
clasifican los píxeles cuyo centro cae dentro.

- Si el modelo **tiene la clase `no borreguil`**, la máscara es opcional.
- Si **no la tiene**, es **obligatoria**: sin ella no se puede hacer el mapa. Ese
  modelo solo distingue tipos, así que le pondría uno a cualquier píxel (roca,
  pinar, agua…). Con la máscara, el mapa dice de qué tipo sería cada sitio, **pero no
  demuestra que allí haya borreguil**. La app no trae ninguna máscara ni la inventa.
- Salvo si está entrenado el modelo de **borreguil sí o no**: entonces el mapa se
  hace en dos pasos (ver «El mapa en dos pasos»), con `no borreguil`, y la máscara
  vuelve a ser opcional.

*El agua de laguna no es borreguil.* El agua abierta de una laguna se parece a un
borreguil muy húmedo, y el modelo la puede dar por borreguil. Con **Quitar el agua de
laguna** (viene marcado; solo si el mapa tiene `no borreguil`), los píxeles de agua que
se ven en Sentinel-2 (NDWI medio de verano mayor de 0,05, pendiente menor de 15° y
manchas de 3 píxeles o más: la misma agua que usa el ambiente por cercanía) pasan a
`no borreguil`. El agua se guarda en `agua_de_laguna.tif` y la ficha del mapa dice
cuántas hectáreas se han quitado. Las probabilidades del modelo no cambian; la de la
categoría elegida pasa a ser la de `no borreguil`.

*La pureza dentro del píxel.* Un píxel de 10 m puede ser borreguil entero o
mezclarse con roca o con agua. En el desplegable **Pureza dentro del píxel**, antes
de hacer el mapa, se puede pedir que el mapa lo diga (viene marcado). No sale del
modelo ni cambia qué es borreguil o su tipo: se calcula **solo en los píxeles que el
modelo da por borreguil**, y solo con Sentinel-2:

1. Se estima **cuánto verde hay dentro** del píxel con su NDVI de inicio de verano:
   0 % si vale lo que la roca, 100 % si vale lo que un borreguil puro, y en proporción
   entre los dos.
2. Con ese verde: **puro** si llega al corte; si no, **mixto-agua** si el píxel es
   agua o toca un píxel de agua, y **mixto-roca** en otro caso.

| Valor | Qué es | Recomendado |
|---|---|---|
| NDVI de un píxel sin verde | El de la roca o el suelo desnudo | **0,080** |
| NDVI de un píxel todo verde | El de un píxel cubierto entero por borreguil | El de tus **muestras de borreguil puro** (su mediana): viene puesto si hay al menos 5; si no, 0,600 |
| Verde a partir del cual el píxel es puro | Con menos, es mixto | **75 %** |

Debajo, una tabla enseña cómo quedan con esos valores **tus muestras de borreguil**,
según la pureza que revisaste: sirve para ajustarlos. En un punto suelto no se sabe
si toca agua, así que ahí solo se compara puro frente a mixto. Si el NDVI de «todo
verde» no es mayor que el de «sin verde», la app lo dice y no deja hacer el mapa hasta
que lo corrijas o desmarques la pureza.

Qué fiabilidad tiene. Se comparó con la ortofoto de 0,25 m en una zona de prueba de
39 ha: el verde estimado se equivoca unos 6 puntos de media entre todos los píxeles
y unos 12 dentro de los de borreguil, y puro frente a mixto coincide en 3 de cada 4
píxeles de borreguil. Lo que peor se ve es el **agua**: Sentinel-2 no distingue una
orilla que ocupa una parte pequeña del píxel, y por eso el mixto-agua se deduce de
los píxeles vecinos. Tampoco distingue poco verde muy vivo de mucho verde apagado.

> ℹ️ Un píxel que el modelo da por `no borreguil` **no recibe pureza**, aunque
> tenga algo de verde: es lo que pasa con un borreguil estrecho de arroyo, que
> ocupa una cuarta parte de sus píxeles. Su verde sí se puede ver, pinchando el píxel
> o en el fichero del verde estimado.

*El ambiente, por cercanía a lagunas y cauces.* En **ambiente y humedad** y en
**borreguil sí o no**. En el de sí o no lo da la casilla **Decir el ambiente de cada
píxel de borreguil (arroyo, laguna o ladera)**, marcada de entrada: el modelo decide
si hay borreguil y cada píxel de borreguil sale en el mapa como **arroyo**,
**laguna** o **ladera** (códigos 1, 2 y 3; 0 sigue siendo «no borreguil»). No hace falta ningún punto con el tipo revisado, y la
superficie de borreguil es la misma que sin la casilla: solo se reparte por ambiente.
Las probabilidades siguen siendo las del modelo (no borreguil, borreguil). En el
desplegable **Ambiente (arroyo, laguna o ladera): por cercanía a lagunas y cauces**,
antes de hacer el mapa, están las distancias de la regla:

1. **laguna**, si el píxel está cerca de una laguna;
2. si no, **arroyo**, si está cerca de un cauce;
3. si no, **ladera**.

| Valor | Qué es | Recomendado |
|---|---|---|
| Laguna: metros a una laguna de la capa | Distancia al punto de una laguna de la capa de la app (65 lagunas de Sierra Nevada). El punto es el centro de la laguna: sirve sobre todo para las pequeñas, que Sentinel-2 no ve | **120 m**, lo que usa la app al proponer el tipo de un punto |
| Laguna: metros al agua que ve Sentinel-2 | Distancia a la orilla de una laguna que se ve: píxeles con NDWI medio de verano mayor de 0,05, en terreno casi llano (menos de 15°) y en manchas de 3 o más. Vale en cualquier sitio, aunque no haya capa | **100 m** |
| Arroyo: metros a un cauce | Distancia a un río o arroyo de OpenStreetMap (o de tu capa de cauces) | **30 m**, lo que usa la app en los puntos |
| Capa de cauces propia | Un fichero de líneas (KML, GeoJSON, GeoPackage o shapefile) con los cauces. Si se sube, se usa en lugar de los de OpenStreetMap | No subir nada, salvo que tengas una red de cauces de la zona más completa |

De dónde salen los **cauces**: son los **ríos y arroyos de OpenStreetMap**, los mismos
que usa la app para el tipo de los puntos (`waterway` = `river` o `stream`; no
cuentan las acequias, los canales ni las fuentes). Los de Sierra Nevada van con la
app; en otras zonas se descargan al hacer el mapa, con unos 5 km de margen alrededor.
Si OpenStreetMap no responde, los cauces se deducen del relieve, como se hacía antes
del 24-11-2026 (es cauce la celda que recibe el agua de 8 ha o más), y el mapa lo
dice debajo. Esos cauces del relieve son una aproximación: van en escalera y a veces
se separan de la vaguada.

Qué fiabilidad tiene. La laguna es lo más seguro: sale solo junto a una laguna de la
capa o al agua que se ve. Los cauces son los que hay dibujados en OpenStreetMap: en
Sierra Nevada están bien, pero un arroyo que no esté dibujado no cuenta. Por eso el mapa lleva una capa, **Lagunas y cauces**, con
lo que se ha tomado por laguna y por cauce: si ahí falta una laguna o sobra un cauce,
el ambiente de alrededor saldrá mal. Todo lo que no queda cerca de una laguna ni de un
cauce es **ladera**, también lo que el modelo dé por borreguil sin serlo.

Debajo del mapa, una tabla compara, en las muestras de borreguil que caen dentro, el
ambiente por cercanía con el que revisaste: sirve para ajustar las distancias. Si las
cambias, el mapa avisa de que es de otras; rehacerlo no vuelve a leer nada.

> ℹ️ Una laguna que quede justo fuera del rectángulo del mapa solo cuenta si está en
> la capa de lagunas: el agua se busca en lo que se lee, que es la zona.

*Qué sale.* Varios GeoTIFF en la rejilla de Sentinel-2 (UTM, 10 m), que se abren
directamente en QGIS:

| Fichero | Contenido |
|---|---|
| `mapa_<nivel>_categoria.tif` | El **código** de la categoría de cada píxel (255 = sin dato), decidida con la regla y el umbral de **3 · Modelo**. Lleva dentro los colores, la lista de códigos y el umbral. En ambiente y humedad es la categoría compuesta (el ambiente por cercanía con la humedad del modelo) y lleva también la regla del ambiente; en borreguil sí o no con el ambiente, 0 no borreguil, 1 arroyo, 2 laguna, 3 ladera |
| `mapa_<nivel>_probabilidad_maxima.tif` | La **probabilidad**, de 0 a 1, de la categoría elegida. Es la confianza del modelo en ese píxel |
| `mapa_<nivel>_probabilidad_por_categoria.tif` | Una banda por categoría, con su nombre: la probabilidad de cada una |
| `mapa_<nivel>_probabilidad_borreguil.tif` | La **probabilidad de ser borreguil**, del tipo que sea: la suma de la de los tipos. Con el umbral, da el mapa de borreguil sí o no. Solo si el modelo tiene la clase `no borreguil` |
| `mapa_<nivel>_pureza.tif` | La **pureza dentro de cada píxel de borreguil**, como código: 1 puro, 2 mixto-agua, 3 mixto-roca (255 = no es borreguil o no hay dato). Solo si se pidió |
| `mapa_<nivel>_verde_estimado.tif` | Cuánto **verde** hay dentro de cada píxel, de 0 a 1, estimado con su NDVI. De todos los píxeles, sean o no borreguil. Solo si se pidió la pureza |
| `mapa_<nivel>_ambiente.tif` | El **ambiente por cercanía** de cada píxel, como código: 1 arroyo, 2 laguna, 3 ladera (255 = sin dato). De todos los píxeles, sean o no borreguil. Solo con el ambiente por cercanía (ambiente y humedad, o borreguil sí o no con su casilla) |
| `mapa_<nivel>_distancias_ambiente.tif` | Tres bandas, en metros: la distancia de cada píxel al agua de una laguna, a una laguna de la capa (hasta 1 km) y a un cauce. De ellas sale el ambiente. Solo con el ambiente por cercanía (ambiente y humedad, o borreguil sí o no con su casilla) |
| `mapa_<nivel>_lagunas_y_cauces.tif` | Lo que se ha tomado por laguna y por cauce: 1 agua de laguna (Sentinel-2), 2 cauce, 3 laguna de la capa. Solo con el ambiente por cercanía (ambiente y humedad, o borreguil sí o no con su casilla) |
| `mapa_<nivel>_codigos.csv` | Qué categoría es cada código, con su color y su superficie |

Los ficheros casan entre sí exactamente: un píxel es borreguil en el mapa de
categorías si, y solo si, su valor en el de probabilidad de borreguil llega al umbral.
En ambiente y humedad, además, el ambiente es el que sale de aplicar la regla a las
distancias guardadas, y la categoría es ese ambiente con la humedad del modelo. Ahí
las probabilidades siguen siendo las del modelo: de `no borreguil`, de `húmedo` y de
`seco`.

Un píxel queda **sin dato**, y no recibe ninguna categoría, si cae fuera de la zona o
de la máscara o si le falta alguna variable (por ejemplo, nube o nieve todos los
años). No se rellena nada.

Todos los GeoTIFF están en la **rejilla de píxeles de Sentinel-2**: la misma zona UTM,
píxeles de 10 m y bordes en múltiplos de 10 m. Cada píxel del mapa es un píxel de
Sentinel-2, el mismo en el que se leyeron las muestras. Los mapas grandes llevan
dentro versiones reducidas, que QGIS y el visor usan para dibujarlos deprisa cuando
se ven de lejos; los datos a tamaño real no cambian.

*Verlo.* Debajo aparecen la superficie clasificada, la probabilidad media, las
hectáreas de cada categoría (en ambiente y humedad, también la suma de cada
ambiente; y, si se pidió, las de borreguil puro y mixto) y un visor sobre la
ortofoto. Las capas se encienden y se
apagan en el **recuadro de arriba a la derecha del mapa**:

| Capa | Qué enseña |
|---|---|
| **Categoría** | La que el modelo pone en cada píxel, con los mismos nombres y colores que las muestras. `no borreguil` va más transparente para que no tape el terreno |
| **Probabilidad máxima** | La probabilidad de la categoría elegida en cada píxel, **sea la que sea**. De oscuro (el modelo duda) a claro (está seguro) |
| **Probabilidad de borreguil** | La suma de la probabilidad de los tipos: donde llega al umbral, el mapa pone borreguil. **Es la capa para ver dónde hay borreguil.** Solo si el modelo tiene la clase `no borreguil` |
| **Pureza dentro del píxel** | De cada píxel de borreguil, si es puro o está mezclado con agua o con roca. Solo si se pidió al hacer el mapa |
| **Lagunas y cauces** | Lo que se ha tomado por laguna y por cauce para poner el ambiente: en turquesa, el agua de laguna que ve Sentinel-2; en azul, los cauces; en rosa, las lagunas de la capa. No tiene bloque en la leyenda del mapa: sus colores van en el texto de debajo. Solo con el ambiente por cercanía (ambiente y humedad, o borreguil sí o no con su casilla). **Es la capa para comprobar el ambiente** |
| **Probabilidad de «…»** | Aparece al elegir una categoría en **Ver también la probabilidad de una categoría**, encima del mapa: la probabilidad que el modelo le da en cada píxel, la elija o no. Sirve para ver dónde ve ese tipo de borreguil y las zonas de transición |

La leyenda, abajo a la izquierda, lleva los colores de las categorías y la escala de
la probabilidad (de 0 a 1). Las muestras no se pintan en este visor, para ver solo lo
que dice el modelo; están en el mapa pequeño de la zona, el de antes de lanzar el mapa,
que lleva la misma leyenda de colores para los puntos. En ambiente y humedad la leyenda
va **por ambientes**: una línea para cada uno (arroyo, laguna, ladera) y, al lado, sus
dos colores, el frío para húmedo y el cálido para seco.

> 💡 La **probabilidad máxima no dice dónde hay borreguil**: dice lo seguro que está
> el modelo de lo que ha puesto. En una zona de `no borreguil` sale clara (cerca de
> 1) porque el modelo está seguro de que **no** lo es. Para ver dónde ve borreguil,
> mira la capa **Probabilidad de borreguil**, la de **Categoría** o la probabilidad
> de un tipo concreto. En un píxel que es borreguil por la suma de sus tipos, la
> probabilidad «máxima» es la de su tipo, que puede ser menor que la de
> `no borreguil` (0,350 en el ejemplo de antes).

**Lo pintado son los píxeles de 10 m de Sentinel-2, cada uno en su sitio.** El visor
dibuja en cada momento lo que se está viendo, a la resolución de la pantalla: al
acercarse se ven los píxeles reales, con los bordes donde son, sea el mapa pequeño o
grande. Moverse, cambiar la opacidad o pedir la probabilidad de una categoría no
recarga el mapa: conserva el zoom y las capas que tuvieras encendidas.

**Pinchando en cualquier píxel** del mapa aparece debajo su categoría, su
probabilidad, la de ser borreguil (junto al umbral), el verde que se estima que tiene
dentro (y la pureza que sale de ese verde, si es borreguil) y la probabilidad de las
demás categorías, y el píxel queda marcado en rojo. Con el ambiente por cercanía dice además
su ambiente y a cuántos metros está del agua de una laguna, de una laguna
de la capa y de un cauce: así se ve por qué ha salido de ese ambiente.

> ⚠️ Que el mapa coincida con las muestras **no mide su acierto**: el modelo final
> se entrenó con ellas. El acierto esperable es el de la validación por zonas del
> paso 3. Y un mapa hecho con pocas muestras es una primera aproximación: mira la
> probabilidad antes de fiarte de una mancha.

El único umbral que se aplica es el de «¿borreguil o no?». No se filtra por
confianza: ningún píxel se queda sin categoría por tener una probabilidad baja; se ve
el resultado completo, con sus probabilidades. Si vuelves a entrenar, la app avisa de
que el mapa es de un modelo anterior hasta que lo rehagas; y si cambias el umbral,
de que se hizo con otro. Rehacerlo entonces tarda poco: lo ya leído en Earth Engine
se aprovecha.

El mapa es **exactamente repetible**: con las mismas muestras, las mismas variables,
los mismos parámetros, la misma semilla y el mismo umbral sale el mismo modelo, y ese
modelo da siempre el mismo mapa, píxel a píxel. Si dos tipos empatan en un píxel, se
queda el de código más bajo; un píxel justo en el umbral cuenta como borreguil.

**El peso de las variables.** La pestaña **5 · Peso** dice cuánto cuenta cada
variable para el modelo. Se calcula al entrenar (alarga el entrenamiento un par de
segundos) y aquí solo se consulta. Se mide de dos maneras:

| Medida | Cómo se mide | Qué dice |
|---|---|---|
| **Lo que pierde el modelo sin ella** (importancia por permutación) | Se repite la validación por zonas con los valores de esa variable **desordenados** entre las muestras, como si dejara de informar, y se mira cuánto baja el F1 medio. Se hace entre 5 y 30 veces (más cuantas menos muestras haya); el «±» es lo que cambia de una vez a otra | Qué variables necesita el modelo para acertar en sitios que no ha visto |
| **Uso interno** (importancia por impureza) | Cuánto ha usado el modelo final esa variable para separar las muestras al entrenar. Es un reparto: entre todas suman 100 % | Con qué variables se han construido los árboles |

Por ejemplo: si el F1 medio de la validación es 0,650 y una variable tiene 0,040, con
esa variable desordenada el F1 se queda en 0,610. En los artículos, y en R, las dos
medidas suelen llamarse *MeanDecreaseAccuracy* (aquí con el F1 medio, que trata igual
a todas las categorías) y *MeanDecreaseGini*. La app no usa esos nombres: dice «lo que
pierde el modelo sin ella» y «uso interno».

Qué enseña la pestaña:

- **Una frase de resumen**: cuál es la variable que más pesa y cuántas hacen bajar el
  F1 **con claridad**, es decir, más del doble de su «±». Si no lo hace ninguna, lo
  dice: el orden entonces no es de fiar.
- **Un gráfico ordenado**, de la que más pesa a la que menos, con el «±» de cada una.
  Encima se elige cuál de las dos medidas se dibuja; al pasar el ratón por una fila
  dice qué mide esa variable y sus dos pesos.
- **Variables juntas**: lo mismo, pero desordenando a la vez todas las variables de
  una **familia** (el mismo índice en distintos momentos: el NDMI de inicio de
  verano, el de fin, su media…) o de una **época u origen** (inicio de verano, fin de
  verano, estadísticas de junio a septiembre, topografía).
- **Todas las variables**, con las dos medidas y su puesto en cada una. Se descarga
  como `peso_variables_<nivel>.csv`.
- **Las que más pesan, categoría a categoría** (desde el 25-11-2026): las variables
  que más pesan en este modelo (lo que pierde sin ellas), en las muestras de la tabla
  de entrenamiento, separadas por las **categorías que aprende el modelo** (por
  ejemplo, húmedo, seco y no borreguil). Son los mismos gráficos que «Las que más
  pesan, grupo a grupo» de **Variables del modelo** en Verdad terreno: **Cajas**,
  **Curvas de distribución**, **Diferencia estandarizada de medianas** (frente a «no borreguil»;
  sin esa categoría en el nivel no hay con qué comparar) y **Dos variables a la vez**.
  Cada categoría lleva su color del mapa. Sirve para ver por qué el modelo separa (o
  no) las categorías: si las cajas de húmedo y seco no se tocan en el NDMI, esa
  variable las distingue.

Cómo leerlo, y con qué cuidado:

- **Las variables parecidas se reparten el peso.** Si se desordena una, las otras la
  cubren, y todas salen con poco peso aunque juntas sean imprescindibles. Para eso
  están las tablas de variables juntas.
- **Cerca de cero, o negativo, no es un error**: desordenar esa variable no empeora
  el resultado (o lo mejora un poco, por azar). Si la bajada es menor que su «±», no
  se distingue de cero.
- **Las dos medidas no tienen por qué coincidir.** El uso interno reparte peso entre
  todas las variables que el modelo usa al entrenar, también las que no le ayudan a
  acertar fuera; lo que pierde sin ella se mide con muestras que no ha visto.
- **Con pocas muestras, el orden baila**: entre variables de peso parecido, el puesto
  puede cambiar con otra semilla o al añadir muestras. Fíate de las diferencias
  grandes.
- **Que pese no quiere decir que sea la causa.** Dice que al modelo le sirve para
  separar tus categorías con estas muestras.

**La app no quita ninguna variable por su cuenta.** Para probar sin una, desmárcala
en **1 · Datos** (no hay que volver a leer nada en Earth Engine), entrena otra vez y
compara el F1 medio. Para quitar de golpe las repetidas está la opción del VIF o
del PCA, en **1 · Datos**, que tampoco viene activada. Las cifras son repetibles: mismos datos, mismos parámetros y
misma semilla dan los mismos pesos.

---

## 11. Sumar tus puntos al modelo

El modelo incluido aprendió en Sierra Nevada, con **769 puntos de campo** (372
borreguiles y 397 no; además hay 27 dudosos que no entrenan). Su tabla de
entrenamiento va con la app (`entrenamiento_sierra_nevada.csv`: coordenadas,
etiqueta, cuenca y variables).

**Por defecto la app usa el modelo tal cual**, aunque tus puntos traigan etiquetas
de campo, y con ellas te dice cuántas acierta. Con la casilla **Sumar mis puntos a los del modelo y reentrenarlo** (panel,
«Modelo de partida»), en cambio, **vuelve a entrenarlo con sus 769 puntos MÁS los
tuyos**: un solo modelo con todos los puntos, no dos modelos aparte.

### Qué puntos tuyos se suman
Todos los que dicen si hay borreguil o no:
- los de tu fichero de campo (o tus puntos con etiqueta);
- los que verificas en **Revisar puntos** (borreguil o no borreguil verificado);
- los que añades **pinchando en el mapa** (pestaña Mapa de los puntos, «➕ Añadir puntos…»).

No se suman los dudosos. Si uno de tus puntos está en el mismo sitio (a menos de
15 m) que uno de los de Sierra Nevada, cuenta el tuyo. Con la casilla, el modelo se
reentrena al ejecutar y al verificar o añadir puntos, sin volver a ejecutar; tarda
unos segundos. Si marcas o desmarcas la casilla con los resultados ya en pantalla, en
el recuadro del modelo aparece **🔁 Reentrenar ahora sumando mis N puntos** (o
**↩ Volver al modelo de partida, sin mis puntos**) para aplicarlo sin volver a
ejecutar. No hay pseudo-ausencias: las ausencias que
faltan las ponen los 397 no borreguiles de Sierra Nevada.

### Cuánto acierta con tus puntos de campo
Si tus puntos traen etiquetas de campo (o has verificado puntos desde la tabla), el
recuadro **🧭 Modelo usado y acierto con tus puntos de campo**, encima de las cinco
pestañas, dice:

| | Borreguiles de campo acertados | Ausencias de campo acertadas | AUC |
|---|---|---|---|
| Modelo de Sierra Nevada, sin tus puntos | 18 de 25 | 12 de 15 | 0,820 |
| Sumando tus puntos (validación por zonas) ✓ | 21 de 25 | 13 de 15 | 0,880 |

- La primera fila es el modelo tal cual; la segunda, solo con la casilla.
- Un borreguil de campo está **acertado** si el modelo lo da por probable o posible
  (con el umbral en uso); una ausencia, si no lo da por borreguil.
- El **AUC** sale con 5 borreguiles y 5 ausencias o más.
- Sumando tus puntos, el acierto **no hace trampa**: cada grupo de tus puntos (por
  cuenca, o por cuadrados de unos 5 km) se predice con un modelo que no ha visto ni
  esos puntos ni los de Sierra Nevada a menos de 500 m de ellos. Si todos tus puntos
  están en la misma zona, no se puede medir así, y la app lo dice.
- Solo cuenta lo que ha dicho una persona: no los dudosos.
- Si tus puntos son de los que entrenaron el modelo de Sierra Nevada, su acierto en
  la primera fila sale más alto de lo real.

### Procedimiento
1. Marca borreguiles confirmados y, si puedes, ausencias (campo o
   fotointerpretación). Usa la [plantilla de datos de campo](#plantilla-de-datos-de-campo).
2. Súbelos en **Opcional: puntos de campo para comprobar el modelo** (o en **Tus
   puntos**, con etiqueta),
   verifícalos en «Revisar puntos» o añádelos pinchando en el mapa.
3. Marca **Sumar mis puntos a los del modelo y reentrenarlo** en el panel.
4. En **🧭 Modelo usado** verás con cuántos puntos se ha entrenado (los de Sierra
   Nevada y los tuyos) y cuánto acierta en los tuyos.
5. Si quieres reutilizarlo, en resultados abre **💾 Guardar el modelo reentrenado**,
   escribe el nombre de tu espacio y pulsa **💾 Guardar en la app**: quedará en
   «Modelo de partida» para las siguientes sesiones. **⬇ Descargar el modelo** te da
   el `.joblib` para pasárselo a otra persona.

### Cómo entrena
- **Puntos**: los 769 de Sierra Nevada (sin los que coinciden con tuyos) y los tuyos
  con etiqueta.
- **Variables**: las del modelo elegido (27, o 32 con el `_cir`; si el radar no se ha
  leído, las de su versión sin radar: 22 o 27).
- **Parámetros**: los mismos con los que se entrenó el de Sierra Nevada (400
  árboles, profundidad 12, clases equilibradas).

---

## 12. Rendimiento

Tiempos orientativos para **150 puntos**:

| Configuración | Tiempo |
|---|---|
| GEE + ⚡ Modo rápido | **1-2 min** |
| GEE completo | 5-15 min |
| Planetary Computer completo | 20-45 min |

**Qué es lento y por qué**: las imágenes (ESRI/PNOA) y CIR/Planet se descargan
**punto a punto en tu ordenador**, aunque uses GEE. Sentinel-2, el radar
(Sentinel-1) y la topografía se calculan en los servidores de Google.

**Para acelerar**: usa GEE, marca **⚡ Modo rápido** para iterar, reduce el número
de puntos y acota los años de Sentinel-2.

---

## 13. Solución de problemas

| Síntoma | Causa probable | Solución |
|---|---|---|
| **Detecta muy pocos puntos** (1 de 500) | Umbral en **Manual** | Cambiar a **Otsu**. |
| Windows bloquea el ejecutable | Sin firma comercial | *Más información* → *Ejecutar de todas formas*. |
| macOS: «no se puede comprobar» | Cuarentena de Gatekeeper | Clic derecho → *Abrir*, o `xattr -dr com.apple.quarantine`. |
| El navegador no abre | — | Ir a la dirección que da la ventana de consola, normalmente **http://localhost:8501**. |
| Muy lento | Backend MPC | Cambiar a GEE y/o **⚡ Modo rápido**. |
| Error de autenticación GEE | Proyecto mal escrito o alta pendiente | Revisar `ee-tunombre` y volver a autenticar. |
| Faltan texturas | **⚡ Modo rápido** activo | Desmarcarlo. |
| El radar sale «omitida» | Backend Planetary Computer | El radar solo se lee con GEE. |
| Histograma en una sola masa | El modelo no discrimina en esa zona | Aportar verdad-terreno local. |
| No se puede resolver el WDPA ID | ID fuera del caché | Usar token de Protected Planet, backend GEE, o buscar por nombre. |
| «No se han podido leer los ríos y arroyos de OpenStreetMap» | Fuera de Sierra Nevada, el servicio Overpass no responde | El análisis sigue, con «arroyo» por la pendiente. Repetirlo más tarde. |
| No aparece mi modelo guardado | Guardado en otra carpeta | Comprobar `~/Borreguil` (ver [§3](#3-dónde-se-guardan-tus-datos)). |

Para reportar un fallo, incluye tu sistema operativo, la configuración usada y
**el texto de la ventana de consola**.

---

## 14. Limitaciones conocidas

- **Sesgo geográfico**: el modelo se entrenó en Sierra Nevada. Fuera de macizos
  comparables su fiabilidad baja; conviene verdad-terreno local.
- **Resolución**: a 10 m, un borreguil menor de ~100 m² no se distingue con
  fiabilidad. La *pureza* del píxel es una estimación con el NDVI: acierta en unos
  8 de cada 10 borreguiles y ve mal el agua ([§10](#10-clasificación-jerárquica-del-tipo)).
- **La altitud pesa mucho** y su rango absoluto cambia entre cordilleras.
- **Dependencia de servicios externos**: Earth Engine, Planetary Computer, OSM e
  IGN pueden fallar o cambiar.
- **Verificación en campo**: los resultados son **candidatos**, no un inventario
  definitivo. Toda cartografía publicada debería validarse en campo.
- **Cuota**: Earth Engine tiene límites de uso por cuenta.

---

## Créditos y cita

**Manual en línea y sugerencias.** La [wiki del repositorio](https://github.com/Bea-Cazorla/borreguil-detector/wiki),
en español e inglés, resume cómo usar la app. Para proponer un cambio o avisar de un
fallo, usa **💬 Sugerir un cambio o informar de un error**, al final del panel lateral
(o el apartado **💬 Manual y sugerencias**, al final de la pestaña de Documentación):
enlaza con la wiki y abre una *issue* en GitHub con la versión ya puesta (pública,
con cuenta de GitHub; puedes pegar pantallazos y seguir la respuesta). Sin cuenta, el
**formulario anónimo** de la app: qué quieres hacer, en qué parte de la app, la versión
(ya puesta), el ordenador, la descripción y, si quieres, un enlace a pantallazos y el
texto de la consola. No pide nombre ni correo, así que no te podrán contestar. No
pegues coordenadas de borreguiles: son un hábitat sensible.

Si usas este programa en un trabajo científico, cítalo mediante el **DOI** de la
versión utilizada (ver `CITATION.cff` y la página de Zenodo del repositorio).

**Datos**: Copernicus Sentinel-1 y Sentinel-2 (ESA), Copernicus DEM,
Microsoft Planetary Computer, Google Earth Engine, OpenStreetMap, PNOA/IGN,
ESRI World Imagery, Protected Planet (WDPA). Los ríos y arroyos de
`cauces_sierra_nevada.geojson` son © colaboradores de OpenStreetMap, con licencia
ODbL (openstreetmap.org/copyright).
