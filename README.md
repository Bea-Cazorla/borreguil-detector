# 🌿 Borreguil · Wet Meadow Finder

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24571286.svg)](https://doi.org/10.5281/zenodo.24571286)

*Buscador y mapa de borreguiles*

> **Se abre en su propia ventana.** El programa abre la app en una ventana suya, sin
> navegador, con [pywebview](https://pywebview.flowrl.com/) y el motor web del sistema:
> Edge WebView2 en Windows 10/11 y WebKit en Mac. Al cerrar la ventana se cierra el
> programa. Si la ventana no se puede abrir (Windows sin el motor de Edge, Linux sin
> GTK ni Qt), se abre el navegador. Con `Borreguil --navegador` se abre siempre en el
> navegador.

App (Streamlit) para identificar **borreguiles** (mountain wet meadows /
cervunales higroturbosos) en cualquier zona de alta montaña y hacer su mapa.

Combina ortofoto (ESRI World Imagery o, en España, PNOA), modelo digital del terreno
(Copernicus DEM 30 m), Óptico Sentinel-2 multitemporal y Radar Sentinel-1, con dos fuentes
de datos posibles:

- **Google Earth Engine (GEE)** — la que viene elegida: el cálculo se hace en la nube
  de Google (rápido, sin descargas). De entrada, con la **cuenta compartida de
  MonitorEO**, sin cuenta propia; o con tu propio proyecto.
- **Microsoft Planetary Computer (MPC)** — sin cuenta, más lento y sin radar. Si
  Earth Engine no está disponible, la app usa esta.

Un **Random Forest preentrenado en Sierra Nevada** sirve como modelo de partida. Se
entrenó con 769 puntos revisados en campo (372 borreguiles y 397 no borreguiles; de
los 796, los 27 dudosos quedaron fuera). Su acierto con validación espacial por
cuencas (AUC) es **0,847** (27 variables) y **0,945** en la variante con ortofoto
infrarroja (32 variables). Sin radar (Planetary Computer), la app usa sola sus
versiones sin radar (0,833 y 0,937). Por defecto el modelo se usa tal cual; con
**«Sumar mis puntos a los del modelo y reentrenarlo»** se vuelve a entrenar con sus
769 puntos más tus puntos con etiqueta (borreguil sí o no).

---

## ⬇️ Descargar el programa (recomendado)

**No necesitas instalar Python ni ninguna librería.**

**macOS (Apple Silicon e Intel).** Abre la app **Terminal**, pega esta línea y pulsa
Intro:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/monitoreo-obsnev/borreguil-detector/main/instalador/macos/instalar.command)"
```

Como los instaladores de MonitorEO y eHabitat+: instala en `~/Borreguil`, sin permisos
de administrador, Python y las librerías de la app (unos 0,3 GB de descarga, solo la
primera vez), deja la app **Borreguil** en el Escritorio y la abre. macOS no la bloquea,
porque la crea tu propio Mac. Más detalles en el [TUTORIAL (2.2)](TUTORIAL.md#22-descomprimir-y-ejecutar).

**Windows y Linux.** Descarga el archivo de tu sistema, descomprímelo y ejecuta el
programa:

| Sistema | Archivo |
|---|---|
| Windows 10/11 | `Borreguil-windows.zip` |
| Linux x86-64 | `Borreguil-linux.tar.gz` |

**Dónde está:**

- **Si hay una versión publicada** en [**Releases**](../../releases), descárgala de ahí. No
  hace falta cuenta.
- **Si Releases está vacía**, descárgalo de la última compilación.
  Hace falta una cuenta de GitHub (es gratis) y haber entrado con ella:
  1. Abre la [lista de compilaciones](https://github.com/monitoreo-obsnev/borreguil-detector/actions/workflows/build-standalone.yml?query=is%3Asuccess)
     y entra en la primera, que es la más reciente.
  2. Baja hasta el final de la página, a **Artifacts**, y pulsa el de tu sistema (por
     ejemplo `Borreguil-windows`).
  3. Se descarga un `.zip` que tiene **otro dentro** (`Borreguil-windows.zip`).
     Descomprime los dos.

  Cada compilación se puede descargar durante 30 días. Si la primera no tiene tu sistema
  o pone que ha caducado (*expired*), prueba con la siguiente o avisa en [Issues](https://github.com/monitoreo-obsnev/borreguil-detector/issues).

Al abrirlo, el programa se actualiza solo con la última versión de la app (ver el
[MANUAL](MANUAL.md#2-instalación-y-arranque)).

📖 **[TUTORIAL.md](TUTORIAL.md)** — primeros pasos, incluido el alta en Google Earth Engine
📘 **[MANUAL.md](MANUAL.md)** — referencia completa de todas las opciones
🌐 **[Wiki](https://github.com/monitoreo-obsnev/borreguil-detector/wiki)** — manual en línea, en español e inglés
🏷️ **[PUBLICAR.md](PUBLICAR.md)** — cómo publicar una versión, un modelo y obtener el DOI

¿Un error o una idea? En la app, **«💬 Sugerir un cambio o informar de un error»**
(abajo en el panel lateral): una *issue* en GitHub o, sin cuenta, el formulario anónimo
de la app, con la versión ya puesta. Si algo falla, **«→ Enviar los registros de este
ordenador»** (o «Enviar los registros», debajo del error) enseña las últimas líneas de lo
que hizo la app, sin tu nombre, correo ni carpetas, y las envía con ese formulario, como
en el asistente de MonitorEO.

> La primera vez, Windows avisa de que el programa no está firmado (es habitual en
> software científico). El tutorial explica cómo continuar.

---

## Ejecutar desde el código fuente (desarrollo)

```bash
pip install -r requirements.txt
streamlit run app.py
```

> **Si al subir un vector ves** `ufunc 'create_collection' not supported…`: es una
> incompatibilidad de versiones antiguas de shapely con numpy 2. Soluciónalo con
> `pip install -U "shapely>=2.0.7"`. La app, de todos modos, degrada a la caja
> envolvente del área en vez de fallar.

Se abre en `http://localhost:8501`.

## Cómo se usa

1. **Origen de los puntos**:
   - Subir tus puntos en **Tus puntos** (KML, GeoJSON, Shapefile o CSV con `lon` y
     `lat`), **o**
   - Generar puntos aleatorios dentro del área de estudio.
2. **Área de estudio** (recomendado), por tres vías:
   - **Subir un vector**: un único **KML / GeoJSON / GeoPackage**, o un
     **shapefile** (selecciona `.shp` + `.shx` + `.dbf` + `.prj` juntos, o sube
     un `.zip`).
   - **WDPA ID** de [protectedplanet.net](https://www.protectedplanet.net):
     resuelve **cualquier** ID si pegas un **token gratuito** de Protected Planet
     (campo opcional, o secret/`env WDPA_TOKEN`), o **sin token** si usas
     **GEE** (capa `WCMC/WDPA` de Earth Engine). Sin token ni GEE, solo
     funcionan los IDs del cache local.
   - **Nombre** del parque (búsqueda Nominatim/OSM; incluye "Parque Nacional/
     Natural" para mejores resultados).
3. **Fuente de datos**: GEE (por defecto) o MPC (sin cuenta, más lento y sin radar).
4. **Puntos de campo** (opcional, fichero aparte): sitios donde ya sabes si hay
   borreguil, con `presence` (1 = borreguil, 0 = no) y, si quieres, `doubtful`, el
   tipo, `observer`, `expertise` y `certainty`. Plantilla:
   `plantilla/point_data_template.xlsx` (botón «⬇ Descargar la plantilla de datos
   de puntos (Excel)» en la pantalla de inicio). **Incluye ausencias**: la app no se
   las inventa. Un fichero sin ninguna columna se toma como todo borreguil: sirve para
   medir cuántos acierta el modelo, pero no aporta ausencias.
5. **▶ Ejecutar pipeline** → cinco pestañas: Mapa de los puntos, Revisar puntos,
   Resumen de los puntos (decisiones, probabilidades y tipos), Variables del modelo
   (su peso y cómo cambian entre grupos) y Descargas (Excel, CSV, `puntos.geojson`,
   mapa HTML y área de estudio).

## Revisar puntos y aportar puntos de campo

En **Revisar puntos**, selecciona una o varias filas. El **último punto
seleccionado** centra el mapa y abre su ficha: la ortofoto con, en amarillo, el
píxel de 10 m del punto (capas Google Satélite, Relieve LiDAR y curvas de nivel), las
variables que más pesan, la gráfica «📈 Este punto frente a los borreguiles
verdaderos» y un enlace a Google Maps. Junto al mapa del punto está la **revisión, en
dos pasos**: **¿Es borreguil?** (sí, no o no estoy seguro, con su **certidumbre**) y,
si es sí, su **tipo**, con la propuesta de la app puesta; un solo botón,
**«💾 Guardar la revisión»**, lo guarda todo, y **«🗑 Quitar del análisis»** quita
puntos. Encima de las pestañas se indica **quién revisa** y su **perfil**
(estudiante, gestor/a, investigador/a); todo sale en las descargas.

En **Mapa de los puntos**, **«➕ Añadir puntos pinchando en el mapa»** añade sitios
que conoces como un dato de campo más. Con **«Sumar mis puntos a los del modelo y
reentrenarlo»** marcada, cada punto verificado o añadido reentrena el modelo al
momento, sin volver a ejecutar.

El tipo lo propone la app por reglas (ambiente, humedad y pureza), y en la revisión una
persona lo confirma o corrige mirando la ortofoto. Lo revisado queda marcado, no
se pisa al reentrenar y se guarda en `puntos.geojson`, que puede volver a cargarse
en otra sesión. Detalles en el [manual](MANUAL.md#10-clasificación-jerárquica-del-tipo).

## Mapa de borreguiles («🌿 Mapa Modelo RF»)

Con los puntos revisados, la app entrena otro Random Forest y lo aplica a cada píxel
de 10 m (necesita **Earth Engine**). Eliges qué cartografiar (borreguil sí o no,
humedad, o ambiente y humedad) y sigues los pasos: **1 · Datos** (aviso
**«¿Son bastantes puntos?»**: mínimo 100 puntos con 33 borreguiles y 67 ausencias,
recomendado 250 con 83 y 167; **«Quitar las variables correlacionadas»**: ninguna, con el
VIF o con un PCA, que se aplican con un botón y dicen qué variables se quedan y por qué),
**2 · Tabla**, **3 · Modelo** (validación por zonas y curva con tus puntos: si más
puntos mejorarían el mapa en tu zona), **4 · Mapa** (GeoTIFF) y
**5 · Peso**. Ver el [manual](MANUAL.md#mapa-de-tipos-con-random-forest).

## Guardar y reutilizar modelos

- **Modelo de partida** (panel lateral, «Modelo base Random Forest»): tus modelos
  guardados, los descargados de GitHub y los incluidos (primero
  `rf_sierra_nevada.joblib`). Da **siempre** la probabilidad, salvo que marques
  «Sumar mis puntos…». Las versiones de `sin_radar/` no salen en la lista: se usan
  solas con Planetary Computer.
- **Guardar**: tras reentrenar, debajo de las pestañas aparece **«💾 Guardar el
  modelo reentrenado»**, con «💾 Guardar en la app» y «⬇ Descargar el modelo». El
  fichero se llama `rf_<nombre de la zona>.joblib`; el nombre se propone a partir
  del área y se puede cambiar.

### Publicar un modelo para todos los usuarios

La app, al abrirse, lee `modelos.json` en este repositorio y **descarga sola los
modelos nuevos o cambiados**, sin reinstalarla. Para publicar uno:

```bash
py publicar_modelo.py rf_<tu_zona>.joblib --nombre "<Nombre del área>"
git add modelos.json rf_<tu_zona>.joblib
git commit -m "Añadir modelo RF: <Nombre del área>"
git push
```

Detalles (versión sin radar, quitar un modelo, otro repositorio): `PUBLICAR.md`.

> **Tamaño**: cada modelo pesa ~3,5-5 MB. Para repositorios con muchos modelos
> considera usar [Git LFS](https://git-lfs.github.com/).

## Probar sin datos propios

Lo más rápido: en la pantalla de inicio, **«▶ Probar con 20 puntos de ejemplo de
Sierra Nevada»** (12 borreguiles y 8 no).

O genera puntos aleatorios y usa el modelo de Sierra Nevada:
- Origen: 🎲 Generar aleatoriamente
- Área: 🆔 WDPA ID `389011` (Parque Nacional de Sierra Nevada) o el de tu zona
- Nº puntos: 200, altitud mín 2000 m → Ejecutar

## Google Earth Engine

**De entrada, la cuenta compartida de MonitorEO.** Con el campo **Proyecto de Google
Cloud** vacío, la app calcula como las herramientas de MonitorEO: pide a su servicio un
permiso de una hora (ninguna clave viaja con el programa) y cada ordenador tiene unas
horas de cálculo de Earth Engine al mes, para uso no comercial. No hay que hacer nada.

**Con tu propia cuenta** (para mapas grandes, o si se acaban esas horas):

1. Crea un proyecto en Google Cloud con la **Earth Engine API** habilitada
   (gratis para uso no comercial): https://earthengine.google.com
2. En la app, escribe el proyecto (p. ej. `ee-tunombre`) en **Proyecto de Google
   Cloud** y pulsa **🔑 Autenticar GEE** (abre el navegador; una vez por ordenador).
3. Pulsa **✓ Probar conexión** para verificar.

Para volver a la cuenta compartida, deja el campo vacío.

## Archivos

| Archivo | Función |
|---|---|
| `app.py` | Interfaz Streamlit |
| `borreguil_pipeline.py` | Pipeline (ortofoto, ríos y arroyos de OSM, DEM, Óptico Sentinel-2, Radar Sentinel-1, Random Forest) |
| `gee_backend.py` | Google Earth Engine |
| `mapeo_rf.py`, `mapeo_raster.py`, `mapeo_tif.py`, `mapeo_ambiente.py` | Pestaña «🌿 Mapa Modelo RF» |
| `point_profile.py`, `i18n.py` | Gráficas por punto e idioma |
| `study_area.py` | Resolución de área (vector / WDPA / nombre) |
| `random_points.py` | Generación de puntos aleatorios |
| `modelos_remotos.py`, `publicar_modelo.py`, `modelos.json` | Modelos nuevos desde GitHub |
| `launcher.py`, `borreguil.spec` | Ejecutable |
| `rf_sierra_nevada.joblib` | Modelo preentrenado (27 variables) |
| `rf_sierra_nevada_cir.joblib` | Variante con ortofoto infrarroja (32 variables) |
| `sin_radar/` | Versiones sin radar (22 y 27 variables) |
| `entrenamiento_sierra_nevada.csv` | Los 796 puntos de entrenamiento, para reentrenar sumando los tuyos |
| `lagunas_sierra_nevada.kml`, `cauces_sierra_nevada.geojson` | Lagunas, ríos y arroyos de Sierra Nevada |
| `plantilla/` | Plantilla de puntos de campo y ejemplo de 20 puntos |
| `examples/` | Datos de ejemplo (borreguiles verificados de Sierra Nevada) |

## Uso en un servidor

Lo recomendado es el ejecutable o el código en tu ordenador. `DEPLOY.md` explica cómo
montarla en Streamlit Community Cloud o Hugging Face Spaces (Earth Engine con
*service account*), pero no se mantiene al día. En un servidor compartido, los
modelos guardados y los ajustes son comunes a todos, y el cálculo de Earth Engine se
carga a esa cuenta.

## Notas

- Todo lo que la app hace por internet va por **HTTPS (puerto 443)**: Earth Engine,
  GitHub, Planetary Computer, ortofotos y OpenStreetMap.
- **MPC** descarga rásters: más lento, más memoria y sin radar.
- El periodo Sentinel-2 por defecto es **2017-2025** (más robusto frente a
  nubes y variabilidad interanual).
- Los **ríos y arroyos de OpenStreetMap** (para el ambiente «arroyo») de Sierra
  Nevada van con la app (`cauces_sierra_nevada.geojson`, © colaboradores de
  OpenStreetMap, ODbL); en otras zonas se descargan del servicio Overpass al
  analizar, y si falla, «arroyo» se decide por la pendiente.

## Agradecimientos

Texto de agradecimiento de los proyectos (el mismo que va en Zenodo):

> This software was funded by the Spanish Multi-regional Program, through the Ministerio de Ciencia, Innovación y Universidades, and by the Project Preserving the singularity of Mediterranean high-mountain biodiversity hotspots: a NBS approach (PRESINMED). PCI 2025-163175. Biodiversa+NBS 2023. Principal investigator: Regino Zamora (University of Granada).
>
> EVEREST project (PID2023-151939OB-I00), funded by MICIU/AEI/10.13039/501100011033 and by the ERDF/EU. Principal investigators: Javier Martínez-López and Domingo Alcaraz-Segura (University of Granada).

## Cómo citar

Si usas este programa en un trabajo científico, cítalo con su DOI de Zenodo
([ficha](https://zenodo.org/records/24571287)). Para la versión 1.0.5:

> Cazorla, B. P., Alcaraz-Segura, D., & Martínez-López, J. (2026). *Borreguil · Wet Meadow Finder: remote sensing and model-assisted ground-truth generation for mapping high-mountain wet meadows* (Version v1.0.5) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.24571287

El DOI **10.5281/zenodo.24571286** vale para todas las versiones (lleva siempre a la última);
**10.5281/zenodo.24571287** es el de la versión 1.0.5. GitHub da la misma cita en
«Cite this repository» (fichero [`CITATION.cff`](CITATION.cff)).

## Licencia

Copyright (C) 2026 los autores de este programa.

Este programa es software libre: puedes redistribuirlo y/o modificarlo bajo los
términos de la **Licencia Pública General GNU** publicada por la Free Software
Foundation, en su **versión 3** o (a tu elección) cualquier versión posterior.

Se distribuye con la esperanza de que sea útil, pero **SIN NINGUNA GARANTÍA**; ni
siquiera la garantía implícita de COMERCIABILIDAD o IDONEIDAD PARA UN PROPÓSITO
PARTICULAR. Consulta la Licencia Pública General GNU para más detalles.

Deberías haber recibido una copia de la Licencia junto con este programa (fichero
[`LICENSE`](LICENSE)); si no, visita <https://www.gnu.org/licenses/>.

> En la práctica: puedes usarlo, estudiarlo y modificarlo libremente; si
> distribuyes una versión modificada, debes publicarla también bajo GPL-3.0 y
> facilitar su código fuente.
