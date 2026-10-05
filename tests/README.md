# Pruebas

Se ejecutan desde la carpeta `borreguil_app`. En Windows, `py -X utf8` en lugar de
`python`.

| Fichero | Qué comprueba | Necesita | Tarda |
|---|---|---|---|
| `test_tipo_revisado.py` | La lógica del tipo revisado: propuesta, revisión, deshacer, guardar y volver a cargar `puntos.geojson` | Nada | segundos |
| `test_autoentrenamiento.py` | Que el auto-entrenamiento no borra ni promueve ausencias y dudas de campo | Nada | segundos |
| `test_presentacion.py` | Cómo se muestran los datos: 4 decimales y nombres largos sin cortar (globos del mapa, tabla, Excel) | Nada | segundos |
| `test_mapeo_datos.py` | El mapeo por Random Forest sin red: qué puntos sirven como muestra, códigos de categoría, variables disponibles como mapa, tabla de entrenamiento y su control de calidad | Nada | segundos |
| `pantalla_revision.py` | La app real en la pantalla de resultados: seleccionar, revisar, deshacer, verificar, iterar y quitar puntos, y la pestaña de mapeo (muestras, lectura de variables con Earth Engine simulado, tabla de entrenamiento), en español e inglés | `classification_v5.csv` en la carpeta que contiene a `borreguil_app` (si falta, se omite) | ~3 min |
| `ejecucion_real.py` | Una ejecución completa con 8 puntos: descarga, clasificación, tipología y ficheros guardados. Con `BORREGUIL_GEE_PROJECT` pasa por Earth Engine; sin ella, por Planetary Computer | Conexión a internet | ~3 min |
| `earth_engine_real.py` | La pila de predictores del mapeo contra Earth Engine real: 34 bandas en orden, pendiente con decimales, y que el valor leído en un punto es el del píxel que se clasificará | Internet, Earth Engine autenticado y `BORREGUIL_GEE_PROJECT` (si falta, se omite) | ~2 min |

```bash
python tests/test_tipo_revisado.py
python tests/test_autoentrenamiento.py
python tests/test_presentacion.py
python tests/test_mapeo_datos.py
python tests/pantalla_revision.py
python tests/ejecucion_real.py
BORREGUIL_GEE_PROJECT=mi-proyecto python tests/earth_engine_real.py
```

No hace falta instalar nada más: cada fichero se ejecuta directamente. Las cuatro
primeras están escritas como funciones `test_*`, de modo que un ejecutor como
pytest también las recogería; las otras ejecutan la app o consultan servicios al
arrancar, y por eso no se llaman `test_*.py`.

Todas terminan con una línea `RESULTADO: …` o `N/N pruebas correctas`, y con código
de salida distinto de cero si algo falla.
