# Pruebas

Se ejecutan desde la carpeta `borreguil_app`. En Windows, `py -X utf8` en lugar de
`python`.

| Fichero | Qué comprueba | Necesita | Tarda |
|---|---|---|---|
| `test_tipo_revisado.py` | La lógica del tipo revisado: propuesta, revisión, deshacer, guardar y volver a cargar `puntos.geojson` | Nada | segundos |
| `test_etiquetas_candidatos.py` | Que la etiqueta de los puntos candidatos se lee venga como venga (`borreguil` yes/no, `presence` 1/0, `doubtful`…), que un fichero de la app queda igual, que una `clase` con tipos no se toma por etiqueta y que, si no hay ninguna columna de etiqueta, se avisa con las columnas que trae | Nada | segundos |
| `test_puntos_azar.py` | El generador de puntos al azar: que, si se pide, quedan lejos de los puntos de campo (para buscar ausencias donde no se ha muestreado) y que sin pedirlo sale lo mismo que antes | Nada | segundos |
| `test_autoentrenamiento.py` | Que el auto-entrenamiento no borra ni promueve ausencias y dudas de campo | Nada | segundos |
| `test_presentacion.py` | Cómo se muestran los datos: 4 decimales y nombres largos sin cortar (globos del mapa, tabla, Excel) | Nada | segundos |
| `test_mapeo_datos.py` | El mapeo por Random Forest sin red: qué puntos sirven como muestra (también en el nivel «borreguil sí o no», con lo dicho en campo aunque no haya tipo revisado), códigos de categoría, variables disponibles como mapa, tabla de entrenamiento y su control de calidad | Nada | segundos |
| `test_mapeo_modelo.py` | El Random Forest del mapeo: validación por zonas, métricas y matriz de confusión, estrategias de balance, parámetros, modelo final, la regla de «¿borreguil o no?» (suma de los tipos y umbral elegido con la validación, de entrada el de mayor F1; cambiarlo sin volver a entrenar) peso de las variables (por permutación y uso interno, una a una, por familias y por grupos; que sale igual cada vez; su CSV) y quitar las variables repetidas con el VIF (qué es el VIF, de una en una la más repetida, la altitud siempre se queda, y entrenar solo con las que quedan) | Nada | ~25 s |
| `test_mapeo_mapa.py` | El mapa del mapeo, con un «mundo» inventado en lugar de Earth Engine: zona y rejilla, bloques, que el mapa diga en cada píxel lo que el modelo predice con su regla y su umbral (y que los ficheros casen entre sí exactamente), el mapa de «ambiente y humedad» (el ambiente por cercanía a lagunas y cauces, compuesto con la humedad del modelo, y la suma de cada ambiente), la pureza dentro de cada píxel de borreguil (puro, mixto-agua, mixto-roca) y su verde estimado, el agua de laguna que pasa a «no borreguil», el mapa de sí o no con el ambiente de cada píxel de borreguil (arroyo, laguna o ladera, sin tipos revisados), el mapa en dos pasos (el sí o no de un modelo y el tipo de otro), sin dato, máscara, GeoTIFF y sus códigos, reanudar un mapa interrumpido, imágenes del visor y consulta de un píxel; y que todo eso sale igual sin rasterio (GDAL), que Windows bloquea en algunos equipos. Donde rasterio sí se puede cargar, cada GeoTIFF que se abre se coteja además con él | Nada | ~30 s |
| `test_mapeo_tif.py` | Los GeoTIFF del mapa, que se escriben y se leen sin GDAL: rejilla, sistema de coordenadas, sin dato, nombres de las bandas, etiquetas y colores; leer por ventanas; escribir por ventanas (en memoria o en un fichero temporal); versiones reducidas (categoría más frecuente o media). Lo escrito se vuelve a leer con Pillow y, donde se puede cargar, con GDAL | Nada | segundos |
| `test_mapeo_ambiente.py` | El ambiente por cercanía a lagunas y cauces, con relieves inventados: la regla y sus distancias, el agua de laguna (NDWI alto, en llano y en manchas), las distancias, los cauces deducidos del relieve (hoyos rellenos, vertido a la vecina más baja, superficie que vierte) y la lectura de una capa de cauces | Nada | segundos |
| `pantalla_revision.py` | La app real en la pantalla de resultados: seleccionar, revisar, deshacer, verificar como borreguil o como no borreguil, iterar y quitar puntos, la pestaña «Mapa Modelo RF» (muestras, lectura de variables con Earth Engine simulado, tabla de entrenamiento, quitar las variables repetidas con el VIF y comparar el acierto, entrenar y validar, el peso de las variables, y hacer, ver y consultar el mapa) y la app sin puntos, en español e inglés. Con un nombre detrás (`revision`, `tabla`, `mapa` o `sin_puntos`) ejecuta solo esa parte | `classification_v5.csv` en la carpeta que contiene a `borreguil_app` (si falta, se omite) | ~6 min |
| `ejecucion_real.py` | Una ejecución completa con 8 puntos: descarga, clasificación, tipología y ficheros guardados. Con `BORREGUIL_GEE_PROJECT` pasa por Earth Engine; sin ella, por Planetary Computer | Conexión a internet | ~3 min |
| `earth_engine_real.py` | La pila de predictores del mapeo contra Earth Engine real: 34 bandas en orden, pendiente con decimales, que el valor leído en un punto es el del píxel que se clasificará, que las estadísticas de jun–sep del mapeo son la media entre años de la de cada año (rehechas imagen a imagen) mientras el detector sigue con todos los años juntos, y que el mapa, al leer por bloques, recibe exactamente ese mismo valor (y un hueco donde no hay dato) | Internet, Earth Engine autenticado y `BORREGUIL_GEE_PROJECT` (si falta, se omite) | ~4 min |

```bash
python tests/test_tipo_revisado.py
python tests/test_autoentrenamiento.py
python tests/test_etiquetas_candidatos.py
python tests/test_puntos_azar.py
python tests/test_presentacion.py
python tests/test_mapeo_datos.py
python tests/test_mapeo_modelo.py
python tests/test_mapeo_mapa.py
python tests/test_mapeo_tif.py
python tests/test_mapeo_ambiente.py
python tests/pantalla_revision.py
python tests/ejecucion_real.py
BORREGUIL_GEE_PROJECT=mi-proyecto python tests/earth_engine_real.py
```

No hace falta instalar nada más: cada fichero se ejecuta directamente. Las nueve
primeras están escritas como funciones `test_*`, de modo que un ejecutor como
pytest también las recogería; las otras ejecutan la app o consultan servicios al
arrancar, y por eso no se llaman `test_*.py`.

Todas terminan con una línea `RESULTADO: …` o `N/N pruebas correctas`, y con código
de salida distinto de cero si algo falla.
