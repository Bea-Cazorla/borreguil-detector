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

> **Licencia**: el proyecto se publica bajo **GPL-3.0-or-later** (fichero
> `LICENSE`). En la práctica: cualquiera puede usarlo, estudiarlo y modificarlo,
> pero si distribuye una versión modificada debe publicarla también bajo GPL y con
> el código fuente. Las dependencias son compatibles (Streamlit y earthengine-api
> Apache-2.0; scikit-learn, geopandas, rasterio y shapely BSD; pyproj y folium MIT).
>
> Como los ejecutables incluyen código GPL, **la GPL obliga a ofrecer el código
> fuente correspondiente**: mantener el repositorio público y enlazado desde la
> Release es suficiente para cumplirlo.

Decisión aparte: **¿publicas también `classification_v5.csv`** (los 796 puntos de
entrenamiento)? Publicarlo hace el trabajo reproducible, pero contiene
**localizaciones precisas de hábitats sensibles** dentro de un espacio protegido.
Consúltalo con la dirección del Parque antes de subirlo.

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

---

## Paso 4 · Publicar la primera versión

```bash
git tag -a v1.0.0 -m "Primera version publica"
git push origin v1.0.0
```

Esto dispara automáticamente:

1. **GitHub Actions** compila los cuatro ejecutables (~15-30 min). Puedes seguirlo
   en la pestaña **Actions**.
2. Los adjunta a la **Release** `v1.0.0`.
3. **Zenodo** detecta la Release, la archiva y **asigna el DOI**.

Si prefieres hacerlo desde la web: **Releases → Draft a new release → Create new
tag `v1.0.0` → Publish release**.

---

## Paso 5 · Recoger el DOI

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

> Cazorla, B. *et al.* (2026). *Detector de borreguiles: identificación automática
> de prados húmedos de alta montaña mediante teledetección y Random Forest*
> (v1.0.0) [Software]. Zenodo. https://doi.org/10.5281/zenodo.XXXXXXX

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
