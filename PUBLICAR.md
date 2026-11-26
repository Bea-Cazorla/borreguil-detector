# Publicar el programa y obtener un DOI

Guía para subir el proyecto a GitHub, compilar los ejecutables automáticamente y
conseguir un **DOI citable** a través de Zenodo.

Solo hay que hacerlo **una vez**. Después, cada nueva versión obtiene su DOI sola.

---

## Resumen del circuito

```
   Publicas una etiqueta          GitHub Actions compila            Zenodo archiva
   (git tag v1.0.0)      ───►     Windows / macOS / Linux    ───►   y asigna el DOI
                                  y los adjunta a la Release
```

---

## Paso 1 · Revisar los metadatos

Antes de publicar nada, edita **`CITATION.cff`** y completa lo marcado como
«REVISAR»:

- [ ] **Autoría completa** — incluye a quien desarrolló la versión original de la
      aplicación. *Omitir a un coautor es un problema serio de integridad
      científica: acuérdalo con esa persona antes de publicar.*
- [ ] **ORCID** de cada autor (si tenéis).
- [ ] **URL del repositorio**.
- [ ] **Versión y fecha**.

> **`.zenodo.json`**: si existe (lo hay desde el 26-11-2026), Zenodo toma de ahí los
> datos del DOI —autores con su filiación, investigadores principales y
> agradecimientos— y **no** lee `CITATION.cff`. Si cambias autores o filiaciones,
> cámbialos en los dos. Debe ser JSON válido: si no, Zenodo no archiva la versión.

> **Licencia**: el proyecto se publica bajo **GPL-3.0-or-later** (fichero
> `LICENSE`). En la práctica: cualquiera puede usarlo, estudiarlo y modificarlo,
> pero si distribuye una versión modificada debe publicarla también bajo GPL y con
> el código fuente. Las dependencias son compatibles (Streamlit y earthengine-api
> Apache-2.0; scikit-learn, geopandas, rasterio y shapely BSD; pyproj y folium MIT).
>
> Como los ejecutables incluyen código GPL, **la GPL obliga a ofrecer el código
> fuente correspondiente**: mantener el repositorio público y enlazado desde la
> Release es suficiente para cumplirlo.

**La tabla de entrenamiento va dentro de la app** (decidido el 19-11-2026):
`entrenamiento_sierra_nevada.csv`, los 796 puntos con los que aprendió el modelo de
Sierra Nevada (de `All_borreguil_lagunas+500ptos`), para poder sumarles los puntos de
quien usa la app. Contiene **localizaciones precisas de hábitats sensibles** dentro de
un espacio protegido: quedan a la vista de quien tenga la app o vea el repositorio.
Conviene comentarlo con la dirección del Parque antes de publicar la Release.
(`classification_v7.csv`, la tabla completa, sigue fuera del repositorio.)

---

## Paso 2 · Subir a GitHub

1. Crea un repositorio **público** en https://github.com/new (Zenodo solo archiva
   repositorios públicos). Sin README ni .gitignore: ya los tienes.
2. Desde la carpeta `borreguil_app`:

```bash
git add -A
git commit -m "Version standalone con manual, tutorial y compilacion automatica"
git branch -M main
git remote add origin https://github.com/TU_USUARIO/TU_REPO.git
git push -u origin main
```

> El `.gitignore` ya excluye tus credenciales (`.streamlit/secrets.toml`).
> **Comprueba que tu clave de Planet no se sube**: `git ls-files | grep secrets`
> solo debe devolver el fichero `.example`.

---

## Paso 3 · Conectar Zenodo (una sola vez)

1. Entra en **https://zenodo.org** e inicia sesión **con tu cuenta de GitHub**.
2. Arriba a la derecha: tu nombre → **GitHub**.
3. Pulsa **Sync now** si tu repositorio no aparece.
4. Activa el **interruptor** del repositorio.

> Debe hacerse **antes** de crear la primera Release: Zenodo solo archiva las
> versiones publicadas después de activarlo.

> **Este repositorio es de la organización `monitoreo-obsnev`** (desde el 26-11-2026).
> El interruptor lo ve quien administra la organización, entrando en Zenodo con su
> cuenta de GitHub. Para saber si ya está conectado, en GitHub: el repositorio →
> **Settings → Webhooks**. Si hay uno de `zenodo.org`, cada Release crea un DOI, y un
> DOI no se puede borrar. Si todavía no queréis DOI, apagad el interruptor en Zenodo
> antes de publicar.

> **Decidido el 26-11-2026: el DOI lo sube Bea a mano desde su propia cuenta de
> Zenodo** (zenodo.org → **New upload**), no con el interruptor. **No activéis el
> interruptor de Zenodo para este repositorio**: cada Release crearía otra ficha con
> otro DOI. En cada versión nueva: en la ficha de Zenodo, **New version**, subir el
> zip del código de la etiqueta (`git -c core.autocrlf=false archive --format=zip
> --prefix=borreguil-detector-X.Y.Z/ -o borreguil-detector-vX.Y.Z.zip vX.Y.Z`, o
> «Source code (zip)» de la Release), cambiar la versión y publicar. Los textos del
> formulario (autores, filiaciones, resumen, agradecimientos) están en
> `.zenodo.json`, en inglés.

---

## Paso 4 · Publicar la primera versión

```bash
git tag -a v1.0.0 -m "Primera version publica"
git push origin v1.0.0
```

Esto dispara automáticamente:

1. **GitHub Actions** compila los ejecutables de Windows, Mac (Apple Silicon) y Linux
   (unos 10 min). Puedes seguirlo en la pestaña **Actions**.
2. Si los tres salen bien, crea la **Release** `v1.0.0` con los tres archivos **como
   borrador** (*Draft*). Desde el 26-11-2026 es así para poder probarla antes: solo
   la ven quienes tienen permiso en el repositorio, y Zenodo no archiva los
   borradores.
3. Para hacerla **pública**: **Releases** → la versión → **Edit** → **Publish
   release**. Entonces cualquiera la descarga sin cuenta y, si Zenodo está
   conectado, **Zenodo la archiva y asigna el DOI**.

> **La `v1.0.5` es pública desde el 26-11-2026** (*Latest*, sin cuenta); la `v1.0.3`
> sigue como borrador y se puede borrar (al abrirse, el programa se actualiza solo al
> último código publicado). El 26-11-2026 quedó decidido
> el DOI: autores los tres de `CITATION.cff`, con sus filiaciones; agradecimientos a
> EVEREST y PRESINMED y a sus investigadores principales (`.zenodo.json` y README); y
> la tabla de entrenamiento puede ir a Zenodo. El DOI lo sube Bea a mano (Paso 3).
> La Release **`v1.0.5`** de GitHub se publica aparte (la `v1.0.3` se puede borrar
> entonces); antes, que quien administra el repositorio compruebe en **Settings →
> Webhooks** que no hay uno de `zenodo.org`, para no crear un segundo DOI. Hecho:
> ya se quitó «(ahora todavía lo está)» / «(it still is for now)» del README, el
> TUTORIAL (2.1) y la wiki. En cada versión nueva, pon antes en `date-released` la
> fecha del día, y súbela a Zenodo a mano (Paso 3).

---

## Paso 5 · Recoger el DOI

> **Publicado el 26-11-2026** desde la cuenta de Zenodo de Bea: https://zenodo.org/records/24571287.
> DOI de todas las versiones **10.5281/zenodo.24571286**; de la versión 1.0.5 **10.5281/zenodo.24571287**.
> Ya están en `CITATION.cff` y en el README (insignia y «Cómo citar»).

En **https://zenodo.org/me/uploads** aparecerá tu depósito con dos DOI:

| DOI | Para qué |
|---|---|
| **DOI de concepto** (*all versions*) | Apunta siempre a la última versión. **Úsalo en artículos** cuando quieras citar el software en general. |
| **DOI de versión** | Fija a `v1.0.0`. Úsalo cuando la reproducibilidad exija la versión exacta. |

Después:

1. Añade el DOI a `CITATION.cff` (campo `doi:`).
2. Añade la insignia al `README.md`:

```markdown
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
```

---

## Publicar un modelo para que la app lo descargue

La app, al abrirse, lee `modelos.json` en GitHub y descarga los modelos que no tiene o
que han cambiado: **no hace falta reinstalarla** para tener un modelo nuevo. Para
publicar uno, desde la carpeta `borreguil_app`:

```bash
py publicar_modelo.py rf_mi_zona.joblib --nombre "Mi zona (Sentinel-2 y relieve)"
git add modelos.json rf_mi_zona.joblib
git commit -m "Modelo de mi zona"
git push
```

El script comprueba que el fichero es un modelo de la app, lo copia a la carpeta si no
está ya y escribe en la lista su tamaño y su huella (SHA-256). Otros usos:

| Para | Orden |
|---|---|
| Añadir también su versión sin radar (`<nombre>_sin_radar.joblib`) | `--sin-radar ruta/al/fichero` |
| Poner al día la lista tras reentrenar un modelo que ya estaba | `py publicar_modelo.py` |
| Que deje de descargarse (quien ya lo tiene lo conserva) | `--quitar rf_mi_zona.joblib` |
| Comprobar que la lista coincide con los ficheros | `--comprobar` (también lo hacen las pruebas) |

> **Quién puede publicar.** Un `.joblib` ejecuta código al cargarse. La app solo
> descarga de este repositorio y comprueba la huella de cada fichero, pero quien pueda
> subir ficheros al repositorio puede publicar un modelo: da permiso de escritura solo
> al equipo. Los modelos se suben ya a `main`, así que se descargan en cuanto se hace
> `git push`.

Para usar otro repositorio de modelos (sin tocar el código), se define en cada equipo
la variable de entorno `BORREGUIL_MODELOS_URL` con la dirección «raw» de su
`modelos.json`. Vacía, la app no busca modelos.

---

## Publicar el código para que los programas instalados se actualicen solos

Al abrirse, el programa instalado lee `codigo.json` en GitHub. Si el código publicado es
distinto del que tiene, descarga solo los ficheros que han cambiado, comprueba cada uno
con su huella y pasa el autodiagnóstico a la versión nueva. Si pasa, arranca con ella;
si no, sigue con la de antes. **No hace falta reinstalar** en las aulas para tener la
última versión de la app.

Para publicar, desde la carpeta `borreguil_app`:

1. Pasa las pruebas (`py -m pytest tests` y la de pantalla).
2. Sube la versión en `CITATION.cff` (por ejemplo, `version: 1.0.1`): es la que se ve en
   la app.
3. Haz commit y push del código.
4. Publica:

```bash
py publicar_codigo.py
git add codigo.json
git commit -m "Publicar el código 1.0.1"
git push
```

El script se niega a publicar si hay cambios sin commit, si el commit no está en GitHub
o si algún fichero no es Python 3.12 válido (el instalador lleva Python 3.12).
Los programas instalados se bajan el código **de ese commit**, no de `main`: lo que
subas después no les llega hasta que vuelvas a publicar.

| Para | Orden |
|---|---|
| Comprobar que la lista coincide con lo publicado | `py publicar_codigo.py --comprobar` |
| Dar marcha atrás | Vuelve al commit bueno (`git checkout <commit>`), publica y sube `codigo.json` |
| Código que necesita una librería nueva | Sube `INSTALADOR` en `actualizador.py`, compila el instalador nuevo y publica con `--instalador-minimo <ese número>` |

Qué se actualiza: los módulos de la app (`app.py`, `borreguil_pipeline.py`…; la lista
está en `actualizador.py`), el manual, el tutorial, `CITATION.cff`, la plantilla y las
capas de lagunas y cauces. **No** se actualizan Python ni las librerías (para eso hace
falta un instalador nuevo) ni los modelos, que tienen su propia lista (`modelos.json`).

Cuando compiles un instalador nuevo, publica también su código. Si no lo haces y la
lista publicada es más vieja que el instalador (según `CITATION.cff`), el programa usa
el código del instalador.

> **Quién puede publicar.** El código descargado se ejecuta. El programa solo lo baja
> de este repositorio, por HTTPS (puerto 443), y comprueba la huella de cada fichero,
> pero quien pueda subir ficheros al repositorio puede publicar código: da permiso de
> escritura solo al equipo.

En cada equipo, la variable de entorno `BORREGUIL_CODIGO_URL` cambia la dirección de
`codigo.json`; vacía, el programa no busca versiones nuevas. También se puede arrancar
una vez sin buscarlas con `Borreguil --sin-actualizar`.

---

## Versiones siguientes

```bash
git add -A
git commit -m "Descripcion de los cambios"
git tag -a v1.1.0 -m "Que cambia en esta version"
git push && git push origin v1.1.0
```

Se recompila, se publica y Zenodo asigna un **DOI nuevo** manteniendo el de
concepto. Numeración recomendada (versionado semántico):

| Cambio | Ejemplo |
|---|---|
| Corrección de errores | `v1.0.0` → `v1.0.1` |
| Función nueva compatible | `v1.0.0` → `v1.1.0` |
| Cambio que rompe la compatibilidad (p. ej. modelo nuevo) | `v1.0.0` → `v2.0.0` |

---

## Cómo citarlo en un artículo

> Cazorla, B. P., Alcaraz-Segura, D., & Martínez-López, J. (2026). *Borreguil · Wet Meadow Finder: remote sensing and model-assisted ground-truth generation for mapping high-mountain wet meadows* (Version v1.0.5) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.24571287

En el apartado de métodos conviene indicar: **versión del programa**, **modelo
empleado** (`rf_sierra_nevada.joblib` o `_cir`), **backend** (GEE o MPC), **años de
Sentinel-2**, **modo de umbral** (Otsu/manual/verdad-terreno) y, si generaste los
puntos, la **semilla** — con eso cualquiera puede reproducir tus resultados.

---

## Problemas frecuentes

| Síntoma | Solución |
|---|---|
| Zenodo no muestra el repositorio | Pulsa **Sync now**; debe ser **público**. |
| La Release no tiene DOI | El interruptor se activó *después* de publicarla: crea una versión nueva. |
| Falla la compilación | Pestaña **Actions** → abre el job en rojo y lee el paso que falló. |
| Falta un ejecutable | `fail-fast: false` permite que los demás sigan: revisa solo el sistema que falló. |
| El ejecutable no arranca en otro ordenador | El workflow ya comprueba que responde HTTP 200; si falla en destino, pide el texto de la consola. |
