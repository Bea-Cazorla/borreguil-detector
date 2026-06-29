# Despliegue online de la app

Esta guía cubre cómo subir la app a un repositorio de GitHub y desplegarla.
Se incluyen dos opciones y una recomendación según el backend que vayas a usar.

---

## TL;DR — ¿qué plataforma elegir?

| Si usas… | Recomendación | Por qué |
|---|---|---|
| **Backend GEE** (Earth Engine) | **Streamlit Community Cloud** (gratis) | El cómputo pesado ocurre en los servidores de Google; la app es ligera y entra de sobra en 1 GB RAM. |
| **Backend MPC** (Planetary Computer) | **Hugging Face Spaces** (gratis, 16 GB RAM / 2 vCPU) | MPC descarga y lee rásters en el propio servidor: necesita más RAM y CPU que el free tier de Streamlit. |
| **Uso intensivo / muchos usuarios** | VM pequeña (Fly.io, Render, una instancia cloud de 2-4 GB) o ejecutar en local | Control total de recursos y tiempos. |

> Regla práctica: **GEE = app ligera → Streamlit Cloud.  MPC = app pesada → Hugging Face Spaces o VM.**

---

## 1. Subir a GitHub

```bash
cd borreguil_app
git init
git add .
git commit -m "Borreguil wet-meadow finder app"
git branch -M main
git remote add origin https://github.com/TU_USUARIO/borreguil-app.git
git push -u origin main
```

El `.gitignore` ya excluye `__pycache__`, `output/`, `*.tif` y, muy importante,
`.streamlit/secrets.toml` (credenciales). El modelo `rf_sierra_nevada.joblib`
(~3,6 MB) sí se sube (está por debajo del límite de 100 MB de GitHub).

---

## 2A. Streamlit Community Cloud  (recomendado con backend GEE)

1. Entra en https://share.streamlit.io y conecta tu cuenta de GitHub.
2. **New app** → elige el repo, branch `main`, archivo `app.py`.
3. Deploy. La primera build instala `requirements.txt` (unos minutos).

### Earth Engine en Streamlit Cloud (service account)

La autenticación por navegador **no** funciona en un servidor headless. Usa un
*service account* de Earth Engine:

1. En Google Cloud, crea un **service account**, habilita Earth Engine para él
   y registra su email en https://signup.earthengine.google.com/#!/service_accounts
2. Descarga su clave JSON.
3. En la app desplegada: **Manage app → Settings → Secrets**, y pega:

   ```toml
   GEE_PROJECT = "ee-tunombre"
   EE_SERVICE_ACCOUNT_KEY = '''
   {
     "type": "service_account",
     "project_id": "ee-tunombre",
     "private_key_id": "…",
     "private_key": "-----BEGIN PRIVATE KEY-----\n…\n-----END PRIVATE KEY-----\n",
     "client_email": "borreguil@ee-tunombre.iam.gserviceaccount.com",
     …
   }
   '''
   ```

   La app lee `st.secrets["EE_SERVICE_ACCOUNT_KEY"]` automáticamente e inicializa
   Earth Engine sin navegador.

> Si solo vas a usar **MPC**, no necesitas secrets: la app funciona tal cual.
> Pero ten en cuenta los límites de RAM/tiempo del free tier (ver 2B).

---

## 2B. Hugging Face Spaces  (recomendado con backend MPC)

1. Crea un Space en https://huggingface.co/spaces → SDK **Streamlit**.
2. Sube los archivos del repo (o conecta el repo de GitHub).
3. Renombra/usa `app.py` como entrypoint (Spaces lo detecta).
4. El free tier "CPU basic" ofrece **16 GB RAM / 2 vCPU**, suficiente para el
   backend MPC con unos cientos de puntos.

Para GEE en Spaces, define el secret `EE_SERVICE_ACCOUNT_KEY` en
**Settings → Variables and secrets** (mismo contenido JSON que arriba).

---

## 3. Consideraciones de recursos

- **Descarga de imágenes ESRI**: ~1 imagen por punto. Para 300 puntos son
  ~300 peticiones HTTP (varios minutos). Si el deploy tiene límites de tiempo
  de request, marca **«Saltar imágenes ESRI»** (se pierden las texturas, pero
  el modelo sigue usando topografía + Sentinel-2).
- **Backend MPC**: lee ventanas de muchos COG de Sentinel-2; con el periodo
  2017-2025 y cientos de puntos puede tardar 10-30 min. El backend **GEE** lo
  hace en segundos (server-side).
- **Memoria**: el merge del DEM y los arrays de Sentinel-2 caben en <1 GB para
  zonas de tamaño de un parque nacional. Áreas enormes (>5000 km²) conviene
  recortarlas.

## 4. Prueba local antes de desplegar

```bash
pip install -r requirements.txt
streamlit run app.py
```

Verifica al menos:
- Generación aleatoria + WDPA ID de tu zona.
- Backend MPC (sin cuenta) ejecuta de principio a fin.
- Si usas GEE: `python gee_backend.py --auth --project ee-tunombre` autentica
  y un test de muestreo funciona.
