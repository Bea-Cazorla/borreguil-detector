"""
mapeo_rf.py — Mapeo de borreguiles con Random Forest (por pasos).

De los puntos cuyo TIPO ha revisado una persona a un mapa de tipos de borreguil.
Se construye de forma incremental; este módulo contiene, por ahora, el bloque
«Datos»:

  · qué puntos de la sesión sirven como muestra de entrenamiento y por qué no
    sirven los demás (nunca se cambia ni se inventa la categoría de un punto);
  · cuántas muestras hay de cada categoría y con qué código entero se guardará
    cada una;
  · qué variables predictoras existen como mapa continuo (las únicas con las que
    después se puede predecir sobre el territorio).

Sin interfaz: todo son funciones puras sobre la lista de puntos, para poder
probarlas solas (tests/test_mapeo_datos.py).

Decisiones acordadas con la usuaria:
  · Las clases son los tres niveles del tipo (ambiente, humedad, pureza) y se
    entrena UN modelo por nivel, no uno con las 18 combinaciones.
  · Solo cuentan los puntos con el tipo REVISADO por una persona: el tipo que
    propone la app sale de reglas sobre los propios predictores, y entrenar con él
    sería enseñarle al modelo la regla.
  · No se inventan ausencias. La clase «no borreguil» solo existe si hay puntos
    marcados así en campo (Borreguil = no).
"""
import math

import borreguil_pipeline as bp

NIVELES = bp.NIVELES_TIPO                  # {'ambiente': (...), 'humedad': (...), ...}
AUSENCIA = 'no borreguil'

# Por debajo de MIN_POR_CLASE no se puede entrenar ni validar una categoría (con
# validación en 5 partes hace falta al menos una muestra por parte). Por debajo de
# RECOMENDADO_POR_CLASE se puede, pero el resultado será poco fiable.
MIN_POR_CLASE = 5
RECOMENDADO_POR_CLASE = 30


def codigos(nivel):
    """Correspondencia ESTABLE categoría → código entero para un nivel.

    0 es siempre «no borreguil»; el resto sigue el orden del catálogo de la app.
    Los códigos no dependen de qué categorías haya en los datos de una ejecución:
    «ladera» vale lo mismo en todos los mapas, haya o no «laguna» esta vez."""
    return {AUSENCIA: 0, **{c: i for i, c in enumerate(NIVELES[nivel], 1)}}


# Por qué un punto no entra como muestra. La clave es estable (va a los ficheros);
# el texto es lo que se enseña.
MOTIVOS = {
    'tipo_sin_revisar': 'borreguil con el tipo sin revisar',
    'sin_etiqueta': 'ni borreguil revisado ni ausencia de campo',
    'dudoso': 'marcado como dudoso en campo',
    'incoherente': 'ausencia de campo que además tiene un tipo revisado',
    'categoria_no_valida': 'categoría que no existe en este nivel',
    'ausencia_no_incluida': 'ausencia de campo (no se ha incluido la clase «no borreguil»)',
    'duplicado': 'mismo píxel de 10 m que otra muestra de su misma categoría',
    'conflicto': 'mismo píxel de 10 m que otra muestra de distinta categoría',
    'sin_coordenadas': 'sin coordenadas válidas',
}
MOTIVOS_EN = {
    'tipo_sin_revisar': 'borreguil whose type has not been reviewed',
    'sin_etiqueta': 'neither a reviewed borreguil nor a field absence',
    'dudoso': 'flagged as doubtful in the field',
    'incoherente': 'field absence that also has a reviewed type',
    'categoria_no_valida': 'category that does not exist at this level',
    'ausencia_no_incluida': 'field absence (the "not a borreguil" class is not included)',
    'duplicado': 'same 10 m pixel as another sample of its own category',
    'conflicto': 'same 10 m pixel as another sample of a different category',
    'sin_coordenadas': 'no valid coordinates',
}
ORIGEN_REVISADO = 'tipo revisado'
ORIGEN_AUSENCIA = 'ausencia de campo'


def motivo(clave, en=False):
    """Texto del motivo por el que un punto no entra como muestra."""
    return (MOTIVOS_EN if en else MOTIVOS).get(clave, clave)


def _num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _clasifica(r, nivel, incluir_ausencias, es_borreguil):
    """(categoría, origen, motivo): la categoría con la que el punto entraría como
    muestra y de dónde sale, o el motivo por el que no entra."""
    if _num(r.get('lon')) is None or _num(r.get('lat')) is None:
        return None, None, 'sin_coordenadas'
    revisado = bp.tipo_revisado(r)
    campo = bp._txt(r.get('Borreguil')).lower()
    if revisado and campo == 'no':
        return None, None, 'incoherente'
    if bp._txt(r.get('Duda')).lower() == 'si':
        return None, None, 'dudoso'
    if revisado:
        valor = bp._txt(r.get(nivel))
        if valor in NIVELES[nivel]:
            return valor, ORIGEN_REVISADO, None
        return None, None, 'categoria_no_valida'
    if campo == 'no':
        if incluir_ausencias:
            return AUSENCIA, ORIGEN_AUSENCIA, None
        return None, None, 'ausencia_no_incluida'
    if campo == 'si' or es_borreguil(bp._txt(r.get('decision'))):
        return None, None, 'tipo_sin_revisar'
    return None, None, 'sin_etiqueta'


def muestras(rows, nivel, incluir_ausencias=True, es_borreguil=None):
    """Selecciona las muestras de entrenamiento de un nivel y explica el resto.

    Entra como muestra:
      · un punto con el tipo REVISADO, con la categoría que dejó la persona;
      · una AUSENCIA de campo (Borreguil = no), como clase «no borreguil», si
        `incluir_ausencias`.
    No entra (y se dice por qué): tipo sin revisar, sin etiqueta, dudoso,
    incoherente, o repetido en el mismo píxel Sentinel-2 de 10 m (dos muestras en
    un píxel tendrían exactamente los mismos predictores: si coinciden en categoría
    se queda la primera; si no, se apartan todas, porque no se puede saber cuál es
    la buena).

    No cambia ningún punto. Devuelve un dict:
      nivel, codigos, tabla (una fila por punto de la sesión, entre o no),
      clases ([{categoria, codigo, n}] en el orden del catálogo),
      n_inicial, n_usadas, excluidos ({motivo: n}), errores, avisos, listo.
    """
    if nivel not in NIVELES:
        raise ValueError(f'nivel desconocido: {nivel!r} (use {", ".join(NIVELES)})')
    es_borreguil = es_borreguil or (lambda dec: False)
    cod = codigos(nivel)

    # Píxel Sentinel-2 de 10 m de cada punto (solo de los que tienen coordenadas:
    # uno sin ellas falsearía la zona UTM con la que se calcula la rejilla).
    validos = [i for i, r in enumerate(rows)
               if _num(r.get('lon')) is not None and _num(r.get('lat')) is not None]
    coords = [(float(rows[i]['lon']), float(rows[i]['lat'])) for i in validos]
    try:
        centros = bp.snap_to_s2_grid(coords) if coords else []
    except Exception:                       # sin rejilla, se compara la coordenada
        centros = coords
    pixeles = dict(zip(validos, centros))

    tabla = []
    for i, r in enumerate(rows):
        cat, origen, motivo = _clasifica(r, nivel, incluir_ausencias, es_borreguil)
        tabla.append({
            'indice': i,                    # posición del punto en la sesión
            'ID': r.get('ID', f'pt_{i}'),
            'lon': r.get('lon'), 'lat': r.get('lat'),
            'categoria': cat or '',
            'codigo': cod[cat] if cat else '',
            'origen_etiqueta': origen or '',
            'fecha_revision': (bp._fecha_txt(r.get('tipo_revisado_fecha'))
                               if origen == ORIGEN_REVISADO else ''),
            'propuesta_app': bp._txt(r.get(nivel + '_regla')),
            'fuente': bp._txt(r.get('source')),
            'usada': cat is not None,
            'motivo': motivo or '',
            '_pixel': ((round(pixeles[i][0], 7), round(pixeles[i][1], 7))
                       if i in pixeles else None),
        })

    # Varias muestras en el mismo píxel de 10 m
    por_pixel = {}
    for fila in tabla:
        if fila['usada']:
            por_pixel.setdefault(fila['_pixel'], []).append(fila)
    for filas in por_pixel.values():
        if len(filas) < 2:
            continue
        if len({f['categoria'] for f in filas}) > 1:
            for f in filas:
                f['usada'], f['motivo'] = False, 'conflicto'
        else:
            for f in filas[1:]:
                f['usada'], f['motivo'] = False, 'duplicado'
    for fila in tabla:
        del fila['_pixel']

    n_por = {c: 0 for c in cod}
    excluidos = {}
    for fila in tabla:
        if fila['usada']:
            n_por[fila['categoria']] += 1
        else:
            excluidos[fila['motivo']] = excluidos.get(fila['motivo'], 0) + 1
    clases = [{'categoria': c, 'codigo': cod[c], 'n': n_por[c]}
              for c in sorted(cod, key=cod.get)
              if c != AUSENCIA or incluir_ausencias]

    con_muestras = [c for c in clases if c['n'] > 0]
    errores, avisos = [], []
    if len(con_muestras) < 2:
        errores.append(('pocas_categorias', len(con_muestras)))
    for c in con_muestras:
        if c['n'] < MIN_POR_CLASE:
            errores.append(('pocas_muestras', c['categoria'], c['n']))
        elif c['n'] < RECOMENDADO_POR_CLASE:
            avisos.append(('muestras_justas', c['categoria'], c['n']))
    if con_muestras:                        # sin ninguna muestra, basta con el error
        for c in clases:
            if c['n'] == 0 and c['categoria'] != AUSENCIA:
                avisos.append(('categoria_sin_muestras', c['categoria']))
        if not any(c['categoria'] == AUSENCIA and c['n'] > 0 for c in clases):
            avisos.append(('sin_ausencia',))

    return {
        'nivel': nivel, 'codigos': cod, 'tabla': tabla, 'clases': clases,
        'n_inicial': len(rows), 'n_usadas': sum(c['n'] for c in clases),
        'excluidos': excluidos, 'errores': errores, 'avisos': avisos,
        'listo': not errores,
    }


def mensaje(item, en=False, traduce=None):
    """Texto de un error o aviso de `muestras()`, en español o en inglés.
    `traduce` traduce el nombre de la categoría (los datos lo guardan en español)."""
    clave = item[0]
    if len(item) > 1 and isinstance(item[1], str) and traduce:
        item = (clave, traduce(item[1])) + tuple(item[2:])
    if clave == 'pocas_categorias':
        if item[1] == 0:
            return ('There are no samples yet. Review the type of some borreguiles in the '
                    'Table tab: only reviewed points count.' if en else
                    'Todavía no hay ninguna muestra. Revisa el tipo de algunos borreguiles '
                    'en la pestaña Tabla: solo cuentan los puntos revisados.')
        return ('Only one category has samples; at least two are needed.' if en else
                'Solo hay muestras de una categoría; hacen falta al menos dos.')
    if clave == 'pocas_muestras':
        s = '' if item[2] == 1 else 's'
        return (f'"{item[1]}" has {item[2]} sample{s}; the minimum is {MIN_POR_CLASE}.' if en
                else f'«{item[1]}» tiene {item[2]} muestra{s}; el mínimo es {MIN_POR_CLASE}.')
    if clave == 'muestras_justas':
        return (f'"{item[1]}" has {item[2]} samples; with fewer than '
                f'{RECOMENDADO_POR_CLASE} the result will be unreliable.' if en else
                f'«{item[1]}» tiene {item[2]} muestras; con menos de '
                f'{RECOMENDADO_POR_CLASE} el resultado será poco fiable.')
    if clave == 'categoria_sin_muestras':
        return (f'No samples of "{item[1]}": it will not appear on the map.' if en else
                f'Ninguna muestra de «{item[1]}»: no aparecerá en el mapa.')
    if clave == 'sin_ausencia':
        return ('There is no "not a borreguil" class: the model will only tell types apart. '
                'Every pixel it is given will receive one of them, so the map must be '
                'limited to areas already known to be borreguil. On its own it does not '
                'show presence versus absence.' if en else
                'No hay clase «no borreguil»: el modelo solo distinguirá tipos. Cualquier '
                'píxel que se le dé recibirá uno de ellos, así que el mapa habrá que '
                'limitarlo a zonas donde ya se sepa que hay borreguil. Por sí solo no '
                'demuestra presencia frente a ausencia.')
    return str(item)


def tabla_csv(resultado):
    """La tabla de muestras como texto CSV (qué puntos se usan y por qué no el
    resto), para poder revisarla fuera y reproducir una ejecución."""
    import csv
    import io
    campos = ['ID', 'lon', 'lat', 'categoria', 'codigo', 'origen_etiqueta',
              'fecha_revision', 'propuesta_app', 'fuente', 'usada', 'motivo']
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=campos, lineterminator='\n',
                       extrasaction='ignore')
    w.writeheader()
    for f in resultado['tabla']:
        w.writerow({**f, 'usada': 'si' if f['usada'] else 'no'})
    return buf.getvalue()


# ============================================================
# Tabla de entrenamiento X / y
# ------------------------------------------------------------
# Una fila por muestra: su categoría y el valor de cada predictor en SU píxel de la
# pila ráster (gee_backend.muestrear_pila), que es el mismo píxel que después se
# clasifica. No valen los valores que ya tienen los puntos del detector: son medias
# en un radio de 25 m e incluyen variables que no existen como mapa.
# ============================================================
def clave_punto(lon, lat):
    """Identifica el punto en la caché de valores leídos (7 decimales ≈ 1 cm)."""
    return (round(float(lon), 7), round(float(lat), 7))


def construir_tabla(res, valores, predictores):
    """Tabla de entrenamiento de las muestras de `res` (salida de `muestras()`).

    valores: {clave_punto: {banda: número | None}} con lo leído de la pila.
    predictores: nombres elegidos (se ponen en el orden fijo del catálogo).

    Una muestra entra solo si tiene valor finito en TODOS los predictores elegidos.
    Las demás se apartan con su motivo, sin rellenar ni inventar nada:
      'sin_leer'       todavía no se ha leído en la pila;
      'sin_cobertura'  la pila no tiene dato de ninguna variable en ese píxel
                       (fuera de las capas);
      'sin_dato'       le falta alguna variable (se dice cuáles).

    Devuelve un dict: predictores, filas (dicts con ID, indice, categoria, codigo y
    los predictores), X, y, apartadas, clases, errores, avisos, listo.
    """
    orden = ordenar_predictores(predictores)
    if not orden:
        raise ValueError('no hay ningún predictor seleccionado')
    filas, apartadas = [], []
    for f in res['tabla']:
        if not f['usada']:
            continue
        base = {'ID': f['ID'], 'indice': f['indice'], 'categoria': f['categoria'],
                'codigo': f['codigo']}
        v = valores.get(clave_punto(f['lon'], f['lat']))
        if v is None:
            apartadas.append({**base, 'motivo': 'sin_leer', 'faltan': []})
            continue
        faltan = [p for p in orden if _num(v.get(p)) is None]
        if faltan:
            sin_nada = all(_num(x) is None for x in v.values())
            apartadas.append({**base, 'motivo': 'sin_cobertura' if sin_nada else 'sin_dato',
                              'faltan': faltan})
            continue
        filas.append({**base, **{p: float(v[p]) for p in orden}})

    cod = res['codigos']
    n_por = {}
    for f in filas:
        n_por[f['categoria']] = n_por.get(f['categoria'], 0) + 1
    clases = [{'categoria': c['categoria'], 'codigo': c['codigo'],
               'n': n_por.get(c['categoria'], 0), 'n_antes': c['n']}
              for c in res['clases']]
    con = [c for c in clases if c['n'] > 0]
    errores, avisos = [], []
    if len(con) < 2:
        errores.append(('pocas_categorias', len(con)))
    for c in con:
        if c['n'] < MIN_POR_CLASE:
            errores.append(('pocas_muestras', c['categoria'], c['n']))
        elif c['n'] < RECOMENDADO_POR_CLASE:
            avisos.append(('muestras_justas', c['categoria'], c['n']))
    return {
        'nivel': res['nivel'], 'codigos': cod, 'predictores': orden, 'filas': filas,
        'X': [[f[p] for p in orden] for f in filas],
        'y': [f['codigo'] for f in filas],
        'apartadas': apartadas, 'clases': clases,
        'errores': errores, 'avisos': avisos, 'listo': not errores,
    }


MOTIVOS_TABLA = {
    'sin_leer': 'todavía no se ha leído en Earth Engine',
    'sin_cobertura': 'fuera de las capas: ninguna variable tiene dato en ese píxel',
    'sin_dato': 'le falta el valor de alguna variable',
}
MOTIVOS_TABLA_EN = {
    'sin_leer': 'not read from Earth Engine yet',
    'sin_cobertura': 'outside the layers: no variable has data at that pixel',
    'sin_dato': 'the value of some variable is missing',
}


def motivo_tabla(clave, en=False):
    return (MOTIVOS_TABLA_EN if en else MOTIVOS_TABLA).get(clave, clave)


def tabla_entrenamiento_csv(tab):
    """La tabla X/y como texto CSV: ID, categoría, código y un predictor por
    columna, en el orden exacto que verá el modelo."""
    import csv
    import io
    campos = ['ID', 'categoria', 'codigo'] + list(tab['predictores'])
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=campos, lineterminator='\n',
                       extrasaction='ignore')
    w.writeheader()
    w.writerows(tab['filas'])
    return buf.getvalue()


# Umbrales del control de calidad de las variables. Solo sirven para AVISAR: no se
# quita ninguna variable automáticamente; decide quien usa la app.
CASI_CONSTANTE = 0.95        # una variable en la que el 95 % de las muestras coincide
CORRELACION_ALTA = 0.95      # |r| de Pearson a partir del cual dos variables se avisan
MUESTRAS_POR_VARIABLE = 5    # menos de 5 muestras por variable: tabla escasa


def calidad(tab):
    """Revisa las variables de la tabla sin cambiarla. Devuelve una lista de
    hallazgos, cada uno una tupla (clave, …):

      ('constante', var)                  no varía: no puede ayudar a distinguir nada
      ('casi_constante', var, fracción)   casi todas las muestras comparten el valor
      ('duplicadas', var_a, var_b)        dos columnas idénticas
      ('opuestas', var_a, var_b)          la misma columna con el signo cambiado
      ('correlacion', var_a, var_b, r)    |r| ≥ CORRELACION_ALTA
      ('mas_variables_que_muestras', n, p)
      ('pocas_muestras_por_variable', n, p)
    """
    import numpy as np
    nombres = list(tab['predictores'])
    X = np.asarray(tab['X'], dtype=float)
    out = []
    n = X.shape[0] if X.ndim == 2 else 0
    p = len(nombres)
    if n == 0:
        return out
    if n < p:
        out.append(('mas_variables_que_muestras', n, p))
    elif n < MUESTRAS_POR_VARIABLE * p:
        out.append(('pocas_muestras_por_variable', n, p))

    constantes = set()
    for j, nombre in enumerate(nombres):
        col = np.round(X[:, j], 6)
        valores_, cuentas = np.unique(col, return_counts=True)
        if len(valores_) == 1:
            constantes.add(j)
            out.append(('constante', nombre))
        elif n >= 20 and cuentas.max() / n >= CASI_CONSTANTE:
            out.append(('casi_constante', nombre, float(cuentas.max() / n)))

    duplicadas = set()
    for a in range(p):
        for b in range(a + 1, p):
            if a in constantes or b in constantes:
                continue
            if np.array_equal(X[:, a], X[:, b]):
                duplicadas.add((a, b))
                out.append(('duplicadas', nombres[a], nombres[b]))
            elif np.allclose(X[:, a], -X[:, b], rtol=0, atol=1e-9):
                # la misma variable con el signo cambiado (p. ej. NDWI = −GNDVI)
                duplicadas.add((a, b))
                out.append(('opuestas', nombres[a], nombres[b]))
    if n >= 3:
        vivos = [j for j in range(p) if j not in constantes]
        if len(vivos) >= 2:
            r = np.corrcoef(X[:, vivos], rowvar=False)
            pares = []
            for ia in range(len(vivos)):
                for ib in range(ia + 1, len(vivos)):
                    a, b = vivos[ia], vivos[ib]
                    if (a, b) in duplicadas:
                        continue
                    val = float(r[ia, ib])
                    if val == val and abs(val) >= CORRELACION_ALTA:
                        pares.append((abs(val), nombres[a], nombres[b], val))
            for _abs, na, nb, val in sorted(pares, reverse=True):
                out.append(('correlacion', na, nb, val))
    return out


def mensaje_calidad(item, en=False):
    """Texto de un hallazgo de `calidad()`."""
    k = item[0]
    if k == 'constante':
        return (f'"{item[1]}" has the same value in every sample: it cannot help.' if en else
                f'«{item[1]}» vale lo mismo en todas las muestras: no puede ayudar.')
    if k == 'casi_constante':
        return (f'"{item[1]}" has the same value in {item[2]:.0%} of the samples.' if en else
                f'«{item[1]}» vale lo mismo en el {item[2]:.0%} de las muestras.')
    if k == 'duplicadas':
        return (f'"{item[1]}" and "{item[2]}" are identical.' if en else
                f'«{item[1]}» y «{item[2]}» son idénticas.')
    if k == 'opuestas':
        return (f'"{item[1]}" and "{item[2]}" are the same variable with the sign '
                'flipped: they carry the same information.' if en else
                f'«{item[1]}» y «{item[2]}» son la misma variable con el signo cambiado: '
                'aportan lo mismo.')
    if k == 'correlacion':
        return (f'"{item[1]}" and "{item[2]}" move together (r = {item[3]:+.2f}).' if en else
                f'«{item[1]}» y «{item[2]}» van juntas (r = {item[3]:+.2f}).')
    if k == 'mas_variables_que_muestras':
        return (f'There are more variables ({item[2]}) than samples ({item[1]}).' if en else
                f'Hay más variables ({item[2]}) que muestras ({item[1]}).')
    if k == 'pocas_muestras_por_variable':
        return (f'{item[1]} samples for {item[2]} variables: fewer than '
                f'{MUESTRAS_POR_VARIABLE} per variable.' if en else
                f'{item[1]} muestras para {item[2]} variables: menos de '
                f'{MUESTRAS_POR_VARIABLE} por variable.')
    return str(item)


def desequilibrio(clases):
    """Cuánto pesa la categoría más abundante frente a la más escasa (solo entre las
    que tienen muestras). Devuelve (razón, mayor, menor) o None si no hay dos."""
    con = [c for c in clases if c['n'] > 0]
    if len(con) < 2:
        return None
    mayor = max(con, key=lambda c: c['n'])
    menor = min(con, key=lambda c: c['n'])
    return mayor['n'] / menor['n'], mayor, menor


# ============================================================
# Variables predictoras disponibles como MAPA continuo
# ------------------------------------------------------------
# Para predecir sobre el territorio, cada variable tiene que existir en todos los
# píxeles, no solo en los puntos. Hoy lo cumplen las que la app calcula en Earth
# Engine (gee_backend.py): los compuestos de Sentinel-2 y la topografía. El nombre
# es el de la banda en esa pila, que es también el de la columna en los puntos.
# ============================================================
G_INICIO = 'Sentinel-2 · inicio de verano (15 jun – 25 jul)'
G_FIN = 'Sentinel-2 · fin de verano (1 ago – 30 sep)'
G_CAMBIO = 'Sentinel-2 · cambio estacional'
G_PERIODO = 'Sentinel-2 · estadísticas de jun – sep'
G_TOPO = 'Topografía (Copernicus DEM, 30 m)'
GRUPOS = (G_INICIO, G_FIN, G_CAMBIO, G_PERIODO, G_TOPO)
GRUPOS_EN = {
    G_INICIO: 'Sentinel-2 · early summer (15 Jun – 25 Jul)',
    G_FIN: 'Sentinel-2 · late summer (1 Aug – 30 Sep)',
    G_CAMBIO: 'Sentinel-2 · seasonal change',
    G_PERIODO: 'Sentinel-2 · Jun – Sep statistics',
    G_TOPO: 'Topography (Copernicus DEM, 30 m)',
}

# (clave, descripción en español, descripción en inglés)
# No está GNDVI: es exactamente −NDWI (las dos se calculan con el verde y el
# infrarrojo cercano), así que sería la misma variable dos veces.
_INDICES = (('ndvi', 'NDVI, vigor de la vegetación', 'NDVI, vegetation vigour'),
            ('ndwi', 'NDWI, agua en superficie', 'NDWI, surface water'),
            ('clre', 'clorofila (borde del rojo)', 'chlorophyll (red edge)'),
            ('ndmi', 'NDMI, humedad de la vegetación', 'NDMI, vegetation moisture'),
            ('evi', 'EVI, vigor corregido', 'EVI, corrected vigour'),
            ('nbr', 'NBR, infrarrojo de onda corta', 'NBR, short-wave infrared'))
_NUEVOS = (('wavi', 'WAVI, vegetación sobre agua', 'WAVI, vegetation over water'),
           ('albedo', 'albedo, brillo de la superficie', 'albedo, surface brightness'),
           ('ndmi', 'NDMI, humedad', 'NDMI, moisture'),
           ('ndwi', 'NDWI, agua', 'NDWI, water'))
_ESTADISTICOS = (('mean', 'media', 'mean'), ('min', 'mínimo', 'minimum'),
                 ('max', 'máximo', 'maximum'), ('sd', 'variabilidad', 'variability'))

# (nombre, grupo, descripción en español, descripción en inglés), en ORDEN FIJO
PREDICTORES = tuple(
    [(f'{k}_early', G_INICIO, es, en) for k, es, en in _INDICES]
    + [(f'{k}_late', G_FIN, es, en) for k, es, en in _INDICES]
    + [('ndvi_drop', G_CAMBIO, 'caída del NDVI entre inicio y fin de verano',
        'NDVI drop between early and late summer')]
    + [(f'{k}_{s}', G_PERIODO, f'{es}: {ses}', f'{en}: {sen}')
       for k, es, en in _NUEVOS for s, ses, sen in _ESTADISTICOS]
    + [('elev_dem_m', G_TOPO, 'altitud', 'elevation'),
       ('slope_deg', G_TOPO, 'pendiente', 'slope'),
       ('aspect_north', G_TOPO, 'orientación: componente norte', 'aspect: north component'),
       ('aspect_east', G_TOPO, 'orientación: componente este', 'aspect: east component'),
       ('curvature', G_TOPO, 'curvatura del terreno', 'terrain curvature')])

# Variables que el detector actual sí usa pero que NO existen como mapa continuo, y
# por qué: (qué, por qué, what, why). No se pueden usar para cartografiar hasta que
# tengan su propia capa.
NO_DISPONIBLES = (
    ('Textura de la imagen aérea (12 variables)',
     'salen de un recorte de ortofoto descargado para cada punto, no de un mapa',
     'Aerial image texture (12 variables)',
     'they come from an orthophoto chip downloaded for each point, not from a map'),
    ('Sentinel-1, radar (5 variables)',
     'se descarga punto a punto; traerlo como mapa desde Earth Engine es código nuevo',
     'Sentinel-1, radar (5 variables)',
     'it is downloaded point by point; bringing it as a map from Earth Engine is new code'),
    ('TWI, índice topográfico de humedad',
     'Earth Engine no lo calcula; solo existe con el otro origen de datos',
     'TWI, topographic wetness index',
     'Earth Engine does not compute it; it only exists with the other data source'),
    ('GNDVI (2 variables)',
     'es el NDWI con el signo cambiado: tenerlas las dos sería repetir la misma variable',
     'GNDVI (2 variables)',
     'it is NDWI with the sign flipped: keeping both would repeat the same variable'),
)


def nombre_grupo(grupo, en=False):
    return GRUPOS_EN.get(grupo, grupo) if en else grupo


def descripcion(nombre, en=False):
    """Qué mide un predictor, en español o en inglés ('' si no existe)."""
    for n, _g, es, eng in PREDICTORES:
        if n == nombre:
            return eng if en else es
    return ''


def nombres_predictores(grupos=None):
    """Nombres de los predictores, en el ORDEN fijo del catálogo (el orden importa:
    el modelo y el mapa tienen que ver las variables en la misma posición)."""
    return [p[0] for p in PREDICTORES if grupos is None or p[1] in grupos]


def ordenar_predictores(nombres):
    """Deja una selección de predictores en el orden del catálogo y sin repetidos.
    Lanza ValueError si alguno no existe como mapa."""
    validos = nombres_predictores()
    desconocidos = [n for n in nombres if n not in validos]
    if desconocidos:
        raise ValueError('no existen como mapa: ' + ', '.join(sorted(set(desconocidos))))
    elegidos = set(nombres)
    return [n for n in validos if n in elegidos]
