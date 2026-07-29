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
   acumulación de agua y textura de la imagen aérea.
3. Un modelo de **inteligencia artificial** (Random Forest), entrenado con cientos
   de borreguiles verificados en campo en Sierra Nevada, estima para cada punto la
   **probabilidad de ser borreguil**.
4. Te lo muestra en un **mapa**, con tablas y gráficas, y te lo deja descargar.

---

## 2. Instalar el programa

> No hay instalación como tal: se descarga, se descomprime y se ejecuta.

### 2.1 Descargar

Ve a la página de **Releases** del repositorio y descarga el archivo de tu sistema:

| Tu ordenador | Archivo a descargar |
|---|---|
| Windows 10/11 | `Borreguil-windows.zip` |
| macOS (Apple Silicon: M1/M2/M3/M4) | `Borreguil-macos-arm64.zip` |
| macOS (Intel) | `Borreguil-macos-intel.zip` |
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
y, a los pocos segundos, tu **navegador** con la interfaz.

Si el navegador no se abre solo, entra tú a **http://localhost:8501**.

> ⏳ El primer arranque puede tardar **30-60 segundos**. Los siguientes son rápidos.

---

## 3. Darte de alta en Google Earth Engine

**¿Por qué?** Earth Engine es el servicio de Google que procesa las imágenes de
satélite *en sus servidores*. Sin él, tu ordenador tendría que descargar gigabytes
de imágenes y una ejecución pasaría de segundos a decenas de minutos.

Es **gratuito para uso académico y de investigación**.

> 💡 ¿Tienes prisa? Puedes saltarte este paso: el programa funciona con el backend
> **Planetary Computer**, que no requiere cuenta — pero es bastante más lento.

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

Esto se hace **una sola vez** por ordenador.

1. En el programa, panel lateral izquierdo → **Fuente de datos satelitales**.
2. Elige **🌍 Google Earth Engine**.
3. En **Proyecto Google Earth Engine**, escribe tu identificador (`ee-tunombre`).
4. Pulsa **🔑 Autenticar GEE**.
5. Se abre el navegador: elige tu cuenta de Google y **acepta los permisos**.
   Copia el código que te da y pégalo si te lo pide la consola.
6. Pulsa **✓ Probar conexión**. Debe salir un mensaje de éxito.

> 🔐 **Sobre los permisos:** autorizas a *tu propio programa, en tu ordenador*, a
> usar *tu* cuota de Earth Engine. Las credenciales se guardan solo en tu equipo.
> Nadie más las recibe.

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
| Estrategia | **stratified_elev** | Reparte los puntos por franjas de altitud. |
| Elev. mín (m) | **2000** | Los borreguiles son de alta montaña. |
| Elev. máx (m) | *(vacío)* | Sin límite superior. |

> ¿Tienes tus propios puntos? Elige **📁 Subir archivo** (KML, GeoJSON o
> shapefile). Ver [MANUAL.md](MANUAL.md) para el formato.

### Paso 3 · Área de estudio

- **Modo**: **🆔 WDPA ID**
- **WDPA ID**: `555512151`  *(Sierra Nevada)*

> Otros: `4514` Picos de Europa · `11` Yellowstone. También puedes buscar por
> **🔤 Nombre** o subir un **🗺 Vector** con tu polígono.

### Paso 4 · Backend

**🌍 Google Earth Engine** (el que configuraste en el paso 4). Si no lo tienes,
deja **Microsoft Planetary Computer** y ármate de paciencia.

### Paso 5 · Modelo

En **Modelo de partida** deja `rf_sierra_nevada.joblib` (universal, 50 variables).

> En España peninsular con cobertura infrarroja (Andalucía, Cataluña, Canarias),
> `rf_sierra_nevada_cir.joblib` es **más preciso**.

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

Al terminar aparecen seis pestañas.

### 🗺️ Mapa

Cada punto es un candidato, coloreado por su decisión:

| Color | Decisión | Qué significa |
|---|---|---|
| 🟢 Verde oscuro | BORREGUIL VERIFICADO | Confirmado en campo (lo aportaste tú). |
| 🟢 Verde | BORREGUIL PROBABLE | Probabilidad **≥ 0,70**. Los más fiables. |
| 🟩 Verde claro | POSIBLE BORREGUIL | Entre el umbral y 0,70. Merecen revisión. |
| 🟧 Naranja | INCIERTO | Zona dudosa. |
| 🔴 Rojo | NO BORREGUIL | Probabilidad baja. |

Haz **clic en un punto** para ver su ficha y poder **quitarlo** del análisis.
La **leyenda** está abajo a la izquierda.

### 📋 Tabla

Todos los puntos con sus coordenadas, probabilidad y variables. Al seleccionar una
fila, el mapa se centra en ese punto y se despliega su **📈 perfil espectral**,
que compara el punto con el rango típico de un borreguil variable a variable.

### 📊 Histograma RF

Reparto de probabilidades. **Lo ideal son dos "montañas"** separadas: una baja
(no-borreguiles) y otra alta (borreguiles). Si sale una sola masa central, el
modelo no está distinguiendo bien en esa zona.

### 🔑 Variables

Qué pesa más en la decisión del modelo. Léelo de arriba abajo.

### 📊 Distribuciones

Cajas y bigotes por variable, y la **clasificación jerárquica del tipo de
borreguil**:

- **Ambiente**: arroyo · laguna · ladera
- **Humedad**: húmedo · seco
- **Pureza**: puro · mixto-agua · mixto-roca

> La *pureza* es **relativa** a los borreguiles detectados en esa ejecución: marca
> como mixtos el ~15 % con más firma de agua o de roca. A 10 m de resolución no se
> puede medir la fracción real dentro del píxel.

### 💾 Descargas

Excel, CSV, GeoJSON, mapa HTML e informe Word.

---

## 7. Descargar y guardar el trabajo

1. Ve a la pestaña **Descargas**.
2. Recomendado: **GeoJSON** (para QGIS/ArcGIS) y **Excel** (para revisar).
3. El **mapa HTML** se abre en cualquier navegador sin el programa: ideal para
   enviarlo a alguien.

Tus modelos entrenados y las salidas se guardan en:

| Sistema | Carpeta |
|---|---|
| Windows | `C:\Users\TU_USUARIO\Borreguil` |
| macOS / Linux | `~/Borreguil` |

---

## 8. Siguientes pasos

### Mejorar la precisión en TU zona

El modelo incluido aprendió en Sierra Nevada. En otras montañas funciona, pero
**mejora mucho** si le enseñas ejemplos locales:

1. Sal al campo (o fotointerpreta) y marca **borreguiles confirmados**.
2. Guárdalos como KML o GeoJSON — **no hace falta ningún atributo**: cada punto se
   asume borreguil.
3. Súbelos en **Opcional: verdad-terreno**.
4. Ejecuta de nuevo. Con **≥ 30** borreguiles (ideal 80–150) el programa entrena un
   modelo **propio de tu zona**.
5. Guárdalo desde el panel de resultados: aparecerá en «Modelo de partida» para
   futuras sesiones.

### Descubrir borreguiles progresivamente

En **🔁 Evaluación recursiva**, los puntos muy probables se van convirtiendo en
ejemplos de entrenamiento y el modelo se recalibra en cada vuelta.

---

## Problemas frecuentes

| Síntoma | Solución |
|---|---|
| Windows bloquea el programa | **Más información** → **Ejecutar de todas formas**. |
| macOS: «no se puede comprobar» | Clic derecho → **Abrir**, o el comando `xattr` del punto 2.2. |
| El navegador no se abre | Entra manualmente a **http://localhost:8501**. |
| **Solo detecta 1 o 2 puntos** | Casi seguro tienes el umbral en **Manual**. Cámbialo a **Otsu**. |
| Va lentísimo | Usa **GEE** en vez de MPC y marca **⚡ Modo rápido**. |
| Error de Earth Engine | Revisa el identificador del proyecto y vuelve a **🔑 Autenticar GEE**. |
| No aparecen texturas ni SAR | Es normal con **⚡ Modo rápido**: desmárcalo. |

¿Sigue sin funcionar? Abre una *issue* en el repositorio indicando tu sistema
operativo y **copiando el texto de la ventana de consola**.
