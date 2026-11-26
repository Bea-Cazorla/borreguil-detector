# Tutorial — primera detección de borreguiles paso a paso

Este tutorial te lleva desde cero (sin nada instalado) hasta tu primer mapa de
borreguiles detectados. **No necesitas saber programar ni instalar Python.**

Tiempo estimado: **40–60 minutos**, de los cuales ~30 son la espera del alta en
Google Earth Engine (que se hace una sola vez).

> ¿Prefieres una referencia completa de cada opción? → [MANUAL.md](MANUAL.md)

---

## Índice

1. [Qué es un borreguil y qué hace este programa](#1-qué-es-un-borreguil-y-qué-hace-este-programa)
2. [Instalar el programa](#2-instalar-el-programa)
3. [Darte de alta en Google Earth Engine](#3-darte-de-alta-en-google-earth-engine)
4. [Conectar el programa con Earth Engine](#4-conectar-el-programa-con-earth-engine)
5. [Tu primera detección](#5-tu-primera-detección)
6. [Interpretar los resultados](#6-interpretar-los-resultados)
7. [Descargar y guardar el trabajo](#7-descargar-y-guardar-el-trabajo)
8. [Siguientes pasos](#8-siguientes-pasos)

---

## 1. Qué es un borreguil y qué hace este programa

Un **borreguil** (o cervunal higroturboso) es un prado húmedo de alta montaña:
vegetación densa y verde alimentada por agua permanente, rodeada de roca o pasto
seco. Son hábitats singulares, pequeños y sensibles al cambio climático, y están
poco cartografiados.

El programa los busca automáticamente:

1. Toma **puntos** sobre el terreno (los tuyos, o generados al azar).
2. Para cada punto descarga **datos de satélite y de terreno**: índices de
   vegetación y humedad (Sentinel-2), radar (Sentinel-1), altitud, pendiente,
   curvatura del terreno y textura de la imagen aérea.
3. Un modelo de **inteligencia artificial** (Random Forest), entrenado con 769
   puntos revisados en campo en Sierra Nevada, estima para cada punto la
   **probabilidad de ser borreguil**.
4. Te lo muestra en un **mapa**, con tablas y gráficas, y te lo deja descargar.
5. Revisas los puntos y, con ellos, haces el **mapa de borreguiles** de tu zona.

---

## 2. Instalar el programa

> No hay instalación como tal: se descarga, se descomprime y se ejecuta.

### 2.1 Descargar

Ve a la página de **Releases** del repositorio (cuando haya una versión publicada) y
descarga el archivo de tu sistema:

| Tu ordenador | Archivo a descargar |
|---|---|
| Windows 10/11 | `Borreguil-windows.zip` |
| macOS (Apple Silicon: M1/M2/M3/M4), con macOS 15 o posterior | `Borreguil-macos-arm64.zip` |
| macOS (Intel) | `Borreguil-macos-intel.zip` (solo si se ha compilado aparte) |
| Linux (Ubuntu, Debian…) | `Borreguil-linux.tar.gz` |

> Ocupa bastante (varios cientos de MB) porque **lleva dentro todo lo necesario**:
> Python, las librerías científicas y los modelos ya entrenados.

### 2.2 Descomprimir y ejecutar

#### 🪟 Windows

1. Clic derecho en el `.zip` → **Extraer todo…**
2. Entra en la carpeta `Borreguil` y haz doble clic en **`Borreguil.exe`**.
3. Windows puede mostrar un aviso azul: **«Windows protegió su PC»**. Es normal en
   programas científicos sin firma comercial de pago. Pulsa **Más información** →
   **Ejecutar de todas formas**.

#### 🍎 macOS

1. Doble clic en el `.zip` para descomprimirlo.
2. **La primera vez**, clic derecho (o Control+clic) sobre **`Borreguil`** →
   **Abrir** → **Abrir**. *(Si haces doble clic normal, macOS lo bloqueará.)*
3. Si aparece «no se puede comprobar que no contenga malware», abre la app
   **Terminal** y pega esto, sustituyendo la ruta por la de tu carpeta:

   ```bash
   xattr -dr com.apple.quarantine /ruta/a/Borreguil
   ```

#### 🐧 Linux

```bash
tar -xzf Borreguil-linux.tar.gz
cd Borreguil
./Borreguil
```

### 2.3 Qué debe ocurrir

Se abre una **ventana negra de consola** (no la cierres: es el motor del programa)
y, a los pocos segundos, la **ventana de la app**. Para cerrar el programa, cierra la
ventana de la app.

Si la ventana no se puede abrir (en Windows, si falta el motor de Edge), la app se abre
en tu **navegador**. Si tampoco, entra en la dirección que muestra la consola,
normalmente **http://localhost:8501**.

Al abrirse, el programa mira también si hay una **versión nueva de la app**. Si la hay,
la descarga, la comprueba y arranca con ella; la consola dice «App actualizada». Sin
internet, abre con la que ya tenía. No hace falta volver a descargar el programa.

Al abrirse, la app trae de GitHub los modelos nuevos o actualizados; la última línea
del panel lateral dice si están al día. Todo lo que hace por internet va por HTTPS
(puerto 443).

> ⏳ El primer arranque puede tardar **30-60 segundos**. Los siguientes son rápidos.

---

## 3. Darte de alta en Google Earth Engine

**¿Por qué?** Earth Engine es el servicio de Google que procesa las imágenes de
satélite *en sus servidores*. Sin él, tu ordenador tendría que descargar gigabytes
de imágenes y una ejecución pasaría de segundos a decenas de minutos.

Es **gratuito para uso académico y de investigación**.

> 💡 **Para empezar no hace falta.** De entrada, la app calcula con la **cuenta
> compartida de MonitorEO**, sin cuenta propia (ver el paso 4). Date de alta si vas a
> usarla mucho, si quieres hacer mapas grandes o si se acaban las horas de cálculo de
> la cuenta compartida.

### 3.1 Crear el proyecto

1. Necesitas una **cuenta de Google** (vale la institucional).
2. Entra en **https://earthengine.google.com/** y pulsa **Get Started** (o
   *Sign Up*).
3. Regístrate indicando el uso **no comercial / académico** y describe brevemente
   tu investigación (p. ej. *"Mapping high-mountain wet meadows in Sierra Nevada,
   University of Granada"*).
4. Google te pedirá crear o elegir un **proyecto de Google Cloud**. Anota su
   **identificador**, con la forma `ee-tunombre`. **Lo necesitarás en el paso 4.**
5. La aprobación suele llegar en **minutos**, a veces en 24-48 h. Recibirás un
   correo.

### 3.2 Comprobar que está activo

Entra en **https://code.earthengine.google.com/**. Si te aparece un editor de
código, ya tienes acceso.

---

## 4. Conectar el programa con Earth Engine

**Con la cuenta compartida de MonitorEO no hay nada que hacer.** En el panel lateral,
**Fuente de datos satelitales** → **🌍 Google Earth Engine** (viene elegido) y el campo
**Proyecto de Google Cloud** vacío. Debajo, la app dice cuántas horas de cálculo te
quedan este mes. Si quieres comprobarlo, pulsa **✓ Probar conexión**.

**Con tu propia cuenta** (después del paso 3). Esto se hace **una sola vez** por
ordenador:

1. En **Proyecto de Google Cloud**, escribe tu identificador (`ee-tunombre`).
2. Pulsa **🔑 Autenticar GEE**.
3. Se abre el navegador: elige tu cuenta de Google y **acepta los permisos**.
   Copia el código que te da y pégalo si te lo pide la consola.
4. Pulsa **✓ Probar conexión**. Debe salir un mensaje de éxito.

Para volver a la cuenta compartida, deja el campo vacío.

> 🔐 **Sobre los permisos:** con tu cuenta, autorizas a *tu propio programa, en tu
> ordenador*, a usar *tu* cuota de Earth Engine. Las credenciales se guardan solo en
> tu equipo. Nadie más las recibe. Con la cuenta compartida no das ningún permiso: la
> app pide al servicio de MonitorEO un permiso de una hora, y tus cálculos se cuentan
> con un identificador anónimo de tu ordenador.

---

## 5. Tu primera detección

Vamos a buscar borreguiles en **Sierra Nevada** sin necesidad de aportar datos
propios. Sigue el panel lateral **de arriba abajo**.

### Paso 1 · Idioma

Arriba del todo, **Idioma / Language**: elige *Español* o *English*.

### Paso 2 · Origen de los puntos

Elige **🎲 Generar aleatoriamente en el área** y configura:

| Opción | Valor para esta prueba | Por qué |
|---|---|---|
| Nº puntos a generar | **150** | Suficiente para ver resultados; rápido. |
| Estrategia | **Repartidos por altitud** | Reparte los puntos por franjas de altitud. |
| Elev. mín (m) | **2000** | Los borreguiles son de alta montaña. |
| Elev. máx (m) | *(vacío)* | Sin límite superior. |

> ¿Tienes una capa de áreas probables de borreguil? Con la estrategia **Por cuartiles
> de una capa**, la mayoría de los puntos (75 % de entrada) caen en los polígonos del
> cuartil 1 y el resto en los demás. Ver el MANUAL, «Datos de entrada».
>
> ¿Tienes tus propios puntos? Elige **📁 Subir archivo** y súbelos en **Tus puntos**
> (KML, GeoJSON, Shapefile o CSV con `lon` y `lat`; plantilla en la pantalla de
> inicio). ¿Solo quieres probar? En la pantalla de inicio, **▶ Probar con 20 puntos
> de ejemplo de Sierra Nevada**.

### Paso 3 · Área de estudio

- **Modo**: **🆔 WDPA ID**
- **WDPA ID**: `555512151`  *(Sierra Nevada)*

> Otros: `4514` Picos de Europa · `11` Yellowstone. También puedes buscar por
> **🔤 Nombre** o subir un **🗺 Vector** con tu polígono.

### Paso 4 · Backend

**🌍 Google Earth Engine** (viene elegido; lo configuraste en el paso 4). Sin cuenta,
elige **🛰 Microsoft Planetary Computer (sin cuenta)**: más lento y sin radar; la app
usa sola la versión del modelo sin radar (AUC 0,833 frente a 0,847).

### Paso 5 · Modelo

En **Modelo de partida** deja `rf_sierra_nevada.joblib` (universal, 27 variables) y
**sin marcar** «Sumar mis puntos a los del modelo y reentrenarlo».

> En Andalucía, Cataluña y Canarias (ortofoto infrarroja PNOA),
> `rf_sierra_nevada_cir.joblib` (32 variables, AUC 0,945) es **más preciso**.

### Paso 6 · Umbral

Deja **Automático · Otsu** (viene por defecto).

> ⚠️ **Importante**: no lo pongas en *Manual* al trabajar en zonas nuevas. Un
> umbral fijo puede dejar fuera casi todo, porque las probabilidades se
> recalibran en cada región.

### Paso 7 · ¡Ejecutar!

Pulsa **▶ Ejecutar pipeline**.

Verás el progreso por fases. **Tiempos orientativos** (150 puntos):

| Configuración | Tiempo |
|---|---|
| GEE + ⚡ Modo rápido | **1-2 min** |
| GEE completo | 5-15 min |
| Planetary Computer completo | 20-45 min |

> ☕ Para una primera prueba, marca **⚡ Modo rápido (vista previa)**: da
> resultados aproximados en una fracción del tiempo. Para resultados definitivos,
> desmárcalo.

---

## 6. Interpretar los resultados

Al terminar, dentro de **📍 Verdad terreno**, aparecen el recuento por decisión, el
recuadro **🧭 Modelo usado y acierto con tus puntos de campo** y cinco pestañas. En la
fila de arriba hay además **🌿 Mapa Modelo RF** (apartado 8) y **📖 Documentación y
recomendaciones**.

### Mapa de los puntos

Cada punto es un candidato, coloreado por su resultado (la columna «Resultado» de «Revisar
puntos»):

| Color | Resultado | Qué significa |
|---|---|---|
| 🟢 Verde oscuro | BORREGUIL VERIFICADO | Una persona lo dio por borreguil (fichero de campo, Revisar puntos o añadido en el mapa). |
| 🟢 Verde | BORREGUIL PROBABLE | Probabilidad **≥ 0,700**. Los más fiables. |
| 🟩 Verde claro | POSIBLE BORREGUIL | Entre el umbral y 0,700. Merecen revisión. |
| 🟧 Naranja | INCIERTO | Entre 0,200 y el umbral. Zona dudosa. |
| 🟨 Ámbar | DUDOSO | Marcado como dudoso en campo: no entrena. |
| 🔴 Rojo | NO BORREGUIL / NO BORREGUIL VERIFICADO | Probabilidad ≤ 0,200, o una persona lo dio por no borreguil. |

Haz **clic en un punto** para ver su ficha y poder **quitarlo** del análisis.
La **leyenda** está abajo a la izquierda. Con **➕ Añadir puntos pinchando en el
mapa** puedes añadir sitios que conoces: pinchas, contestas debajo si hay borreguil y
el punto entra como un dato de campo más (borde blanco).

### Revisar puntos

Todos los puntos con sus coordenadas, probabilidad y variables. Al seleccionar una
fila, el mapa se centra en ese punto y se despliega su ficha: el valor de las
variables que más pesan y la gráfica **📈 Este punto frente a los borreguiles
verdaderos**, que dice en cada una dónde cae el punto entre los borreguiles con los
que se entrenó el modelo.

Con puntos seleccionados, junto al mapa del punto (ortofoto, con el píxel de 10 m en
amarillo) sale la revisión, en dos pasos:

1. **¿Es borreguil?** Sí, No o No estoy seguro, con la certidumbre.
2. **Tipo**, solo si es sí: la app propone ambiente, humedad y pureza, y tú los
   aceptas, los corriges o eliges «No lo he podido ver».

**💾 Guardar la revisión** guarda las dos cosas a la vez; **↩ Deshacer** deshace la
última, y **🗑 Quitar del análisis** quita los puntos. Con «Sumar mis puntos…» marcada,
si cambia el sí o el no, el modelo se reentrena al momento.

Encima de las pestañas puedes poner **quién revisa** y su **perfil** (estudiante,
gestor/a o investigador/a). Al revisar un punto, indica también la **certidumbre**
(alta, media o baja). Todo se guarda en el punto y sale en las descargas.

### Resumen de los puntos

Lo que ha encontrado la app, de arriba abajo:

1. **Probabilidad de ser borreguil.** **Lo ideal son dos "montañas"** separadas: una
   baja (no-borreguiles) y otra alta (borreguiles). Si sale una sola masa central, el
   modelo no está distinguiendo bien en esa zona. Debajo de la gráfica, la app explica
   cómo leerla, qué son las dos rayas rojas y cuántos puntos cambiarían si movieras el
   umbral; los del centro son los primeros que conviene revisar.
2. **Recuento de puntos por categoría de decisión**, de más a menos borreguil, con el
   número y el porcentaje, y los colores del mapa de los puntos.
3. **🌿 Tipos de borreguil**: cuántos hay de cada categoría, cada una con su color (el
   mismo en toda la app: arroyo verde, laguna azul, ladera amarillo verdoso), y la
   **clasificación jerárquica del tipo de borreguil**:

- **Ambiente**: arroyo · laguna · ladera
- **Humedad**: húmedo · seco
- **Pureza**: puro · mixto-agua · mixto-roca

> La *pureza* se propone como en el mapa: cuánto verde hay en el píxel de 10 m del
> punto, según su NDVI de inicio de verano. Es **puro** si llega al 75 %; si no,
> **mixto-agua** si el píxel es agua o toca agua, y **mixto-roca** si no. Necesita
> Earth Engine; sin él, la pureza se queda sin propuesta (—).

### Variables del modelo

Qué pesa más en la decisión del modelo (léelo de arriba abajo) y, debajo, cómo son
esas variables en cada grupo de puntos: borreguil frente a no borreguil, o por humedad
o por ambiente, en cajas, curvas, diferencias o dos variables a la vez.

### Descargas

Excel, CSV, GeoJSON de los puntos y mapa HTML.

---

## 7. Descargar y guardar el trabajo

1. Ve a la pestaña **Descargas**.
2. Recomendado: **`puntos.geojson`** (para QGIS/ArcGIS) y **Excel** (para revisar).
3. El **mapa HTML** se abre en cualquier navegador sin el programa: ideal para
   enviarlo a alguien.

> 💡 `puntos.geojson` guarda lo que has hecho a mano: los puntos verificados y los
> tipos revisados. Para seguir otro día, cárgalo en *Tus puntos* al
> empezar y no tendrás que repetirlo.

Los ficheros del análisis **no se guardan solos**: descárgalos en **Descargas**. En
esta carpeta quedan los modelos guardados con «💾 Guardar en la app», los modelos
nuevos de GitHub y el proyecto de Earth Engine:

| Sistema | Carpeta |
|---|---|
| Windows | `C:\Users\TU_USUARIO\Borreguil` |
| macOS / Linux | `~/Borreguil` |

---

## 8. Siguientes pasos

### Mejorar la precisión en TU zona

El modelo aprendió en Sierra Nevada; mejora si le enseñas ejemplos locales:

1. Descarga la **plantilla de datos de campo** (pantalla de inicio). Una fila por
   punto: `lon`, `lat` (grados) y `presence` (**1** borreguil, **0** no; `doubtful`
   = 1 si no lo sabes). **Incluye ausencias**: la app no se las inventa.
2. Guárdala como CSV y súbela en **Tus puntos** o en **Opcional: puntos de campo
   para comprobar el modelo**.
3. Ejecuta. Por defecto se usa el modelo tal cual, y **🧭 Modelo usado y acierto con
   tus puntos de campo** dice cuántos acierta.
4. Marca **Sumar mis puntos a los del modelo y reentrenarlo**: se reentrena con sus
   769 puntos **más** todos los tuyos con etiqueta (fichero, **Revisar puntos** y los
   añadidos en el mapa). Desde entonces, cada punto verificado o añadido lo reentrena
   al momento. El acierto en tus puntos se mide por zonas.
5. Guárdalo en **💾 Guardar el modelo reentrenado** (debajo de las pestañas):
   «💾 Guardar en la app» o «⬇ Descargar el modelo».

### Hacer el mapa de borreguiles

La pestaña **🌿 Mapa Modelo RF** (necesita Earth Engine) entrena otro modelo con tus
puntos revisados y lo aplica a cada píxel de 10 m: **1 · Datos** («¿Son bastantes
puntos?»: mínimo 100 puntos con 33 borreguiles y 67 ausencias, recomendado 250 con 83
y 167; «Quitar las variables correlacionadas»: ninguna, con el VIF o con un PCA, con el
botón «Aplicar», que dice qué variables se quedan y por qué), **2 · Tabla**, **3 · Modelo**
(con «📈 Calcular la curva con mis puntos», que dice si más puntos mejorarían el mapa
en tu zona), **4 · Mapa** (GeoTIFF) y **5 · Peso**. Ver el
[manual](MANUAL.md#mapa-de-tipos-con-random-forest).

---

## Problemas frecuentes

| Síntoma | Solución |
|---|---|
| Windows bloquea el programa | **Más información** → **Ejecutar de todas formas**. |
| macOS: «no se puede comprobar» | Clic derecho → **Abrir**, o el comando `xattr` del punto 2.2. |
| La ventana de la app no se abre o sale en blanco | El programa abre entonces el navegador; si tampoco, entra en la dirección que da la consola, normalmente **http://localhost:8501**. También puedes abrirlo con `Borreguil --navegador`. |
| **Solo detecta 1 o 2 puntos** | Casi seguro tienes el umbral en **Manual**. Cámbialo a **Otsu**. |
| Va lentísimo | Usa **GEE** en vez de MPC y marca **⚡ Modo rápido**. |
| Error de Earth Engine | Con tu proyecto: revisa su identificador y vuelve a **🔑 Autenticar GEE**. Con la cuenta compartida (campo vacío): prueba más tarde o usa tu proyecto. |
| «Has gastado las horas de cálculo de este mes» | Se han acabado las de la cuenta compartida de MonitorEO: usa tu propio proyecto (paso 4) o espera al día 1. |
| No aparecen texturas | Es normal con **⚡ Modo rápido**: desmárcalo. |
| El radar (Sentinel-1) sale «omitida» | Solo se lee con **GEE**; con Planetary Computer la app usa sola la versión del modelo sin radar. |
| Sin texturas aunque no uses el modo rápido | Windows puede bloquear scikit-image; la app avisa y rellena las texturas con un valor típico (probabilidad menos fiable). |
| «Sin conexión con GitHub» o red con cortafuegos (aula) | Todo va por HTTPS, puerto 443: hay que dejarlo salir hacia Earth Engine y `raw.githubusercontent.com`. |

¿Sigue sin funcionar? **💬 Sugerir un cambio o informar de un error** (abajo en el
panel lateral) abre una *issue* o, sin cuenta de GitHub, un formulario anónimo: indica
tu sistema y copia el texto de la consola; no pegues coordenadas de borreguiles. Manual en la
[wiki](https://github.com/monitoreo-obsnev/borreguil-detector/wiki).
