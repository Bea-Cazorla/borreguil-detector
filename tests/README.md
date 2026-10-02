# Pruebas

Se ejecutan desde la carpeta `borreguil_app`. En Windows, `py -X utf8` en lugar de
`python`.

| Fichero | Qué comprueba | Necesita | Tarda |
|---|---|---|---|
| `test_tipo_revisado.py` | La lógica del tipo revisado: propuesta, revisión, deshacer, guardar y volver a cargar `puntos.geojson` | Nada | segundos |
| `test_autoentrenamiento.py` | Que el auto-entrenamiento no borra ni promueve ausencias y dudas de campo | Nada | segundos |
| `pantalla_revision.py` | La app real en la pantalla de resultados: seleccionar, revisar, deshacer, verificar, iterar y quitar puntos, en español e inglés | `classification_v5.csv` en la carpeta que contiene a `borreguil_app` (si falta, se omite) | ~3 min |
| `ejecucion_real.py` | Una ejecución completa con 8 puntos: descarga, clasificación, tipología y ficheros guardados | Conexión a internet | ~3 min |

```bash
python tests/test_tipo_revisado.py
python tests/test_autoentrenamiento.py
python tests/pantalla_revision.py
python tests/ejecucion_real.py
```

No hace falta instalar nada más: cada fichero se ejecuta directamente. Las dos
primeras están escritas como funciones `test_*`, de modo que un ejecutor como
pytest también las recogería; las dos últimas ejecutan la app al arrancar, y por
eso no se llaman `test_*.py`.

Todas terminan con una línea `RESULTADO: …` o `N/N pruebas correctas`, y con código
de salida distinto de cero si algo falla.
