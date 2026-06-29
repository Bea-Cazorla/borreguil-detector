# 🌿 Borreguil / Wet Meadow Finder

App web (Streamlit) para identificar **borreguiles** (mountain wet meadows /
cervunales higroturbosos) en cualquier zona de alta montaña.

Combina imagen aérea (ESRI World Imagery), modelo digital del terreno
(Copernicus DEM 30 m) e imagen satelital multi-temporal Sentinel-2, con dos
backends posibles:

- **Microsoft Planetary Computer (MPC)** — sin cuenta, sin API key.
- **Google Earth Engine (GEE)** — si tienes cuenta, el cómputo se hace en la
  nube de Google (rápido, sin descargas).

Un **Random Forest preentrenado en Sierra Nevada** (AUC GroupKFold = 0,86)
sirve como modelo por defecto, y puede **reentrenarse con tus propios puntos
de campo** (modo *presence-only*: solo necesitas las coordenadas de
borreguiles confirmados).

---

## Ejecutar en local

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
   - Subir un KML/Shapefile/GeoJSON con puntos candidatos, **o**
   - Generar puntos aleatorios dentro del área de estudio.
2. **Área de estudio** (recomendado), por tres vías:
   - **Subir un vector**: un único **KML / GeoJSON / GeoPackage**, o un
     **shapefile** (selecciona `.shp` + `.shx` + `.dbf` + `.prj` juntos, o sube
     un `.zip`).
   - **WDPA ID** de [protectedplanet.net](https://www.protectedplanet.net):
     resuelve **cualquier** ID si pegas un **token gratuito** de Protected Planet
     (campo opcional, o secret/`env WDPA_TOKEN`), o **sin token** si usas el
     backend **GEE** (capa `WCMC/WDPA` de Earth Engine). Sin token ni GEE, solo
     funcionan los IDs del cache local.
   - **Nombre** del parque (búsqueda Nominatim/OSM; incluye "Parque Nacional/
     Natural" para mejores resultados).
3. **Backend satelital**: MPC (sin cuenta) o GEE (con proyecto + autenticación).
4. **Verdad-terreno** (opcional): subir un KML con borreguiles confirmados.
   **No hace falta ningún atributo** — todos los puntos se asumen borreguil
   (*presence-only*). La clase negativa se genera automáticamente con
   pseudo-ausencias.
5. **Ejecutar** → mapa interactivo + tabla + descargas (CSV / Excel / mapa HTML).

## Evaluación recursiva (auto-entrenamiento)

Tras la primera estimación, en el panel **🔁 Evaluación recursiva** puedes
refinar el modelo de forma iterativa (técnica de *self-training*):

1. Los puntos de **posible borreguil** (probabilidad RF ≥ umbral de promoción,
   por defecto 0,70) se convierten en verdad-terreno.
2. Se **recalibra** el Random Forest y se vuelve a predecir sobre el resto.
3. Eliges cómo incorporarlos:
   - **Añadir (acumular)**: el conjunto de positivos crece en cada iteración.
   - **Reemplazar (refrescar)**: se recalculan los pseudo-positivos desde el
     último modelo, manteniendo siempre tus puntos verificados de campo.
4. Se muestra el **historial de iteraciones** (positivos usados, nuevos
   hallados, AUC) y, si una iteración no encuentra nuevos borreguiles, avisa de
   **convergencia**. Puedes lanzar tantas iteraciones como quieras.

Los nuevos borreguiles descubiertos automáticamente aparecen en el mapa en
**verde azulado** (categoría `BORREGUIL (auto)`), distinguibles de los
verificados en campo (verde oscuro).

## Inspeccionar puntos y aportar verdad-terreno (aprendizaje activo)

En la pestaña **Tabla**, **selecciona una o varias filas** (casillas a la izquierda):
- la **primera** seleccionada centra el mapa principal, abre un **mini-mapa con
  zoom** sobre el punto y muestra su **imagen ESRI**, métricas (altitud, slope,
  TWI, NDVI, Clre, EVI, patrón) y un enlace a Google Maps;
- con una o varias seleccionadas, el botón **«✓ Marcar como borreguil verificado
  y reentrenar»** las añade a la verdad-terreno y **recalibra el modelo al
  instante** (las features ya están calculadas).

Así puedes hacer **tantas iteraciones como quieras** combinando dos vías:
auto-promoción (panel recursivo) y confirmación manual de puntos en la tabla.
Cada confirmación mejora el modelo en tu zona.

## Guardar y reutilizar modelos

- **Modelo de partida** (sidebar): un desplegable lista todos los modelos `.joblib`
  de la carpeta de la app (empezando por `rf_sierra_nevada.joblib`). El elegido
  se usa para predecir cuando no hay verdad-terreno local suficiente.
- **Guardar el modelo entrenado**: en el panel **🔁 Evaluación recursiva**, cuando
  el modelo se ha entrenado con verdad-terreno local, aparece la opción de
  guardarlo. El fichero se llama **`rf_<nombre del área protegida>.joblib`** (p. ej.
  `rf_parque_nacional_picos_de_europa.joblib`), donde el nombre se deriva
  automáticamente del área de estudio.

### Publicar el modelo para todos los usuarios del deploy online

Los modelos `.joblib` están **incluidos en el repositorio** (no aparecen en
`.gitignore`) para que cualquier usuario del deploy online pueda usarlos sin
necesitar entrenar desde cero.

Flujo para publicar un modelo nuevo:

1. **Localmente**: pulsa "💾 Guardar en la carpeta" → se crea
   `borreguil_app/rf_<tu_zona>.joblib`.
2. Súbelo al repositorio:
   ```bash
   git add borreguil_app/rf_<tu_zona>.joblib
   git commit -m "Añadir modelo RF: <Nombre del área>"
   git push
   ```
3. Streamlit Cloud redesplegará la app automáticamente. El nuevo modelo
   aparecerá en el desplegable **"Modelo de partida"** para todos los usuarios.

**App online** (la carpeta es efímera entre reinicios): usa el botón
"⬇ Descargar .joblib", copia el fichero a `borreguil_app/` en tu repositorio
local y sigue los pasos 2-3.

> **Tamaño**: cada modelo pesa ~3-4 MB. Para repositorios con muchos modelos
> considera usar [Git LFS](https://git-lfs.github.com/).

## Probar sin datos propios

Genera puntos aleatorios y usa el modelo Sierra Nevada:
- Origen: 🎲 Generar aleatoriamente
- Área: 🆔 WDPA ID `555512151` (Sierra Nevada) o el de tu zona
- Backend: MPC
- Nº puntos: 200, altitud mín 2000 m → Ejecutar

## Backend Google Earth Engine

1. Crea un proyecto en Google Cloud con la **Earth Engine API** habilitada
   (gratis para uso no comercial): https://earthengine.google.com
2. En la app, elige backend **GEE**, escribe el nombre del proyecto
   (p. ej. `ee-tunombre`) y pulsa **🔑 Autenticar GEE** (abre el navegador).
3. Pulsa **✓ Probar conexión** para verificar.

> En local la autenticación es por navegador. Para un despliegue online
> headless se usa un *service account* (ver `DEPLOY.md`).

## Archivos

| Archivo | Función |
|---|---|
| `app.py` | Interfaz Streamlit |
| `borreguil_pipeline.py` | Pipeline (ESRI, OSM, DEM, S2, Random Forest) |
| `gee_backend.py` | Backend Google Earth Engine |
| `study_area.py` | Resolución de área (vector / WDPA / nombre) |
| `random_points.py` | Generación de puntos aleatorios |
| `rf_sierra_nevada.joblib` | Modelo Random Forest preentrenado |
| `Metodologia_borreguiles.docx` | Documentación metodológica completa |
| `examples/` | Datos de ejemplo (borreguiles verificados de Sierra Nevada) |

## Despliegue online

Ver **`DEPLOY.md`** — incluye Streamlit Community Cloud y Hugging Face Spaces,
y la configuración de Earth Engine con service account.

## Notas

- El backend **MPC** descarga rásters: más lento y con más uso de RAM.
  Recomendado para deploy: **GEE** (cómputo en la nube, app ligera).
- El periodo Sentinel-2 por defecto es **2017-2025** (más robusto frente a
  nubes y variabilidad interanual).
- OSM (hidrografía/infraestructuras) está **desactivado por defecto** porque
  el servicio Overpass es inestable; actívalo con la casilla si lo necesitas.
