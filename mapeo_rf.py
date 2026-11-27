"""
mapeo_rf.py — Mapeo de borreguiles con Random Forest (por pasos).

De los puntos cuyo TIPO ha revisado una persona a un mapa de tipos de borreguil.
Se construye de forma incremental; este módulo contiene los datos y el modelo (el
mapa en sí, píxel a píxel, está en mapeo_raster.py):

  · qué puntos de la sesión sirven como muestra de entrenamiento y por qué no
    sirven los demás (nunca se cambia ni se inventa la categoría de un punto);
  · cuántas muestras hay de cada categoría y con qué código entero se guardará
    cada una;
  · qué variables predictoras existen como mapa continuo (las únicas con las que
    después se puede predecir sobre el territorio);
  · la tabla de entrenamiento y su control de calidad;
  · la validación por zonas, el modelo final y el peso de cada variable.

Sin interfaz: todo son funciones puras sobre la lista de puntos, para poder
probarlas solas (tests/test_mapeo_datos.py y tests/test_mapeo_modelo.py).

Decisiones acordadas con la usuaria:
  · Las clases son los tres niveles del tipo (ambiente, humedad, pureza) y se
    entrena UN modelo por nivel, no uno con las 18 combinaciones.
  · Hay además un nivel combinado, pedido por ella: el ambiente con la humedad
    dentro (arroyo, laguna o ladera y, en cada uno, húmedo o seco): 6 categorías.
    En él el modelo solo aprende la HUMEDAD (y «no borreguil»). El ambiente no
    sale del modelo: lo pone el mapa por cercanía a lagunas y cauces
    (mapeo_ambiente.py), porque dice dónde está el píxel y eso no se ve en sus
    variables. Lo pidió así al ver que el modelo daba por «laguna» borreguiles
    que estaban a cientos de metros de cualquier laguna.
  · Solo cuentan los puntos con el tipo REVISADO por una persona: el tipo que
    propone la app sale de reglas sobre los propios predictores, y entrenar con él
    sería enseñarle al modelo la regla.
  · No se inventan ausencias. La clase «no borreguil» solo existe si hay puntos
    marcados así en campo (Borreguil = no).
  · Hay además un nivel «borreguil» (sí o no), pedido por ella: aprende solo si hay
    borreguil, con TODOS los puntos marcados en campo como borreguil o no, tengan o
    no el tipo revisado. El tipo lo aprende aparte el modelo de un nivel del tipo,
    solo con los borreguiles revisados, y el mapa junta los dos (en_dos_pasos): el
    de sí o no decide si hay borreguil y el del tipo, cuál.
"""
import itertools
import math

import borreguil_pipeline as bp

# Lo que se puede cartografiar: los tres niveles del tipo y, además, los niveles
# COMBINADOS. Un nivel combinado no es un dato nuevo del punto: su categoría se
# compone con las de los niveles de los que sale («ladera · húmedo»), que son las
# que revisa la persona. El orden es fijo: de él salen los códigos. El primer nivel
# es el que manda (la RAMA: el ambiente) y el segundo va dentro de él (la humedad).
SEP = ' · '
COMBINADOS = {'ambiente_humedad': ('ambiente', 'humedad')}
# El nivel de sí o no va el primero: es el primer paso del mapa.
NIVEL_SI_NO = 'borreguil'
NIVELES = {NIVEL_SI_NO: ('borreguil',), **bp.NIVELES_TIPO}
for _nivel, _partes in COMBINADOS.items():
    NIVELES[_nivel] = tuple(SEP.join(c) for c in itertools.product(
        *(bp.NIVELES_TIPO[p] for p in _partes)))
AUSENCIA = 'no borreguil'


def rama_y_hoja(categoria):
    """Una categoría de un nivel combinado, partida: («ladera», «húmedo»). En un nivel
    sencillo (o «no borreguil») no hay rama: (None, la categoría)."""
    rama, sep, hoja = str(categoria).partition(SEP)
    return (rama, hoja) if sep else (None, str(categoria))


def nivel_del_modelo(nivel):
    """El nivel cuyas categorías aprende el modelo. En un nivel sencillo, él mismo. En
    un nivel combinado, el de DENTRO (la humedad): el de fuera (el ambiente) no lo
    aprende el modelo, lo pone el mapa por cercanía (mapeo_ambiente.py)."""
    return COMBINADOS[nivel][-1] if nivel in COMBINADOS else nivel


def nivel_por_cercania(nivel):
    """En un nivel combinado, el nivel que el mapa pone por cercanía (el ambiente). None
    en un nivel sencillo: ahí todo sale del modelo."""
    return COMBINADOS[nivel][0] if nivel in COMBINADOS else None


def componer(nivel, cod_rama, cod_hoja):
    """Código de la categoría compuesta de un nivel combinado a partir del código de su
    rama (el ambiente: 1 arroyo, 2 laguna, 3 ladera) y del de su hoja (la humedad que
    da el modelo: 1 húmedo, 2 seco). Donde la hoja es 0 («no borreguil») el resultado
    es 0. Vale con números sueltos o con matrices."""
    import numpy as np
    rama, hoja = COMBINADOS[nivel]
    n_hojas = len(NIVELES[hoja])
    cod_rama = np.asarray(cod_rama).astype('int64')
    cod_hoja = np.asarray(cod_hoja).astype('int64')
    if ((cod_hoja < 0) | (cod_hoja > n_hojas)).any() or (
            ((cod_rama < 1) | (cod_rama > len(NIVELES[rama])))[cod_hoja > 0]).any():
        raise ValueError(f'códigos fuera de las categorías de «{nivel}»')
    return np.where(cod_hoja > 0, (cod_rama - 1) * n_hojas + cod_hoja, 0)


def filas_de_leyenda(categorias):
    """Las categorías de un modelo o de un mapa, ordenadas para una leyenda.

    categorias: [{'categoria': …, …}] en orden de código. Devuelve una fila por cada
    una, con lo que traía y además 'etiqueta' (lo que se escribe) y 'rama'. En un
    nivel combinado, delante de las categorías de cada rama va una fila que solo trae
    {'titulo': rama}, y la etiqueta de cada categoría es la de dentro:

        arroyo                 ← {'titulo': 'arroyo'}
          húmedo               ← {'etiqueta': 'húmedo', 'rama': 'arroyo', …}
          seco
        ladera
          húmedo

    En un nivel sencillo (y para «no borreguil») no hay títulos ni rama."""
    filas, abierta = [], None
    for c in categorias:
        rama, hoja = rama_y_hoja(c['categoria'])
        if rama and rama != abierta:
            filas.append({'titulo': rama})
        abierta = rama
        filas.append({**c, 'etiqueta': hoja, 'rama': rama})
    return filas

# Por debajo de MIN_POR_CLASE no se puede entrenar ni validar una categoría (con
# validación en 5 partes hace falta al menos una muestra por parte). Por debajo de
# RECOMENDADO_POR_CLASE se puede, pero el resultado será poco fiable.
MIN_POR_CLASE = 5
RECOMENDADO_POR_CLASE = 30

# ¿Cuántos puntos hacen falta para el mapa de borreguil sí o no? Curva de aprendizaje
# calculada el 24-11-2026 (claude-borreguiles/curva_puntos/curva_puntos.py) con las 948
# muestras revisadas del mapa final de Sierra Nevada (315 borreguiles) y el mismo
# procedimiento que aquí: VIF ≤ 10, Random Forest de 500 árboles con pesos y validación
# por zonas (58). Para cada tamaño se entrena con N puntos al azar (con la proporción de
# borreguiles de la tabla, 1 de cada 3) y se predicen siempre zonas que el modelo no ha
# visto; 10 repeticiones. Columnas: puntos, borreguiles, AUC media, AUC p10, AUC p90 y
# TSS (umbral 0,5). La última fila son todos los puntos de las zonas de entrenamiento.
CURVA_PUNTOS_SN = (
    (20, 7, 0.896, 0.879, 0.917, 0.621),
    (30, 10, 0.903, 0.876, 0.923, 0.620),
    (40, 13, 0.923, 0.914, 0.937, 0.672),
    (50, 17, 0.930, 0.919, 0.938, 0.690),
    (60, 20, 0.936, 0.923, 0.943, 0.709),
    (80, 26, 0.945, 0.938, 0.952, 0.730),
    (100, 33, 0.951, 0.946, 0.955, 0.752),
    (125, 41, 0.951, 0.947, 0.956, 0.751),
    (150, 50, 0.955, 0.948, 0.961, 0.772),
    (200, 66, 0.957, 0.955, 0.960, 0.771),
    (250, 83, 0.961, 0.958, 0.964, 0.786),
    (300, 99, 0.961, 0.957, 0.964, 0.781),
    (400, 132, 0.964, 0.960, 0.966, 0.792),
    (500, 166, 0.966, 0.964, 0.968, 0.791),
    (600, 199, 0.968, 0.966, 0.970, 0.802),
    (682, 226, 0.969, 0.967, 0.971, 0.807),
)
# (puntos, borreguiles). Mínimo: el AUC queda a menos de 0,02 del de todos los puntos
# y casi no cambia entre repeticiones. Recomendado: a menos de 0,01.
MINIMO_PUNTOS = (100, 33)
RECOMENDADO_PUNTOS = (250, 83)
# Las ausencias (no borreguiles) de esos puntos: la curva se hizo con 1 borreguil de cada
# 3, así que el resto lo eran. Se piden aparte desde el 25-11-2026 (la autora: «¿cómo sé
# cuál es el mínimo de ausencias?»): antes, 100 borreguiles y 6 ausencias «llegaban al
# mínimo». La curva no probó otras proporciones: son las que había en ella.
MINIMO_AUSENCIAS = MINIMO_PUNTOS[0] - MINIMO_PUNTOS[1]                  # 67
RECOMENDADO_AUSENCIAS = RECOMENDADO_PUNTOS[0] - RECOMENDADO_PUNTOS[1]   # 167
POCOS_PUNTOS = 50               # por debajo, el acierto cae y depende de qué puntos sean


def auc_esperado(n):
    """AUC que daba en Sierra Nevada el mapa de sí o no con `n` puntos revisados
    (interpolado en escala logarítmica). None por debajo de lo medido (20 puntos); con
    más de lo medido, el de todos los puntos."""
    curva = CURVA_PUNTOS_SN
    if n is None or n < curva[0][0]:
        return None
    if n >= curva[-1][0]:
        return curva[-1][2]
    for (n0, _, a0, *_r0), (n1, _b, a1, *_r1) in zip(curva, curva[1:]):
        if n0 <= n <= n1:
            f = (math.log(n) - math.log(n0)) / (math.log(n1) - math.log(n0))
            return a0 + f * (a1 - a0)
    return None


def lectura_puntos(n, n_borreguil, n_ausencia=None):
    """Dónde están `n` muestras (`n_borreguil` de ellas borreguiles y `n_ausencia`
    ausencias; si no se da, el resto) frente a la curva de Sierra Nevada. Devuelve
    {'nivel', 'auc', 'faltan', 'faltan_borreguiles', 'faltan_ausencias'}; nivel:
    'sin_referencia' (< 20), 'pocos' (< 50), 'bajo_minimo', 'minimo' o 'recomendado'.
    Para llegar a un nivel hacen falta los puntos, los borreguiles Y las ausencias;
    'faltan…' cuenta lo que falta para el siguiente."""
    n, nb = int(n or 0), int(n_borreguil or 0)
    na = max(0, n - nb) if n_ausencia is None else int(n_ausencia)
    aus = {MINIMO_PUNTOS: MINIMO_AUSENCIAS, RECOMENDADO_PUNTOS: RECOMENDADO_AUSENCIAS}
    if (n >= RECOMENDADO_PUNTOS[0] and nb >= RECOMENDADO_PUNTOS[1]
            and na >= RECOMENDADO_AUSENCIAS):
        nivel, meta = 'recomendado', None
    elif n >= MINIMO_PUNTOS[0] and nb >= MINIMO_PUNTOS[1] and na >= MINIMO_AUSENCIAS:
        nivel, meta = 'minimo', RECOMENDADO_PUNTOS
    elif n < CURVA_PUNTOS_SN[0][0]:
        nivel, meta = 'sin_referencia', MINIMO_PUNTOS
    elif n < POCOS_PUNTOS:
        nivel, meta = 'pocos', MINIMO_PUNTOS
    else:
        nivel, meta = 'bajo_minimo', MINIMO_PUNTOS
    return {'nivel': nivel, 'auc': auc_esperado(n),
            'faltan': max(0, meta[0] - n) if meta else 0,
            'faltan_borreguiles': max(0, meta[1] - nb) if meta else 0,
            'faltan_ausencias': max(0, aus[meta] - na) if meta else 0}


def codigos(nivel):
    """Correspondencia ESTABLE categoría → código entero para un nivel.

    0 es siempre «no borreguil»; el resto sigue el orden del catálogo de la app.
    Los códigos no dependen de qué categorías haya en los datos de una ejecución:
    «ladera» vale lo mismo en todos los mapas, haya o no «laguna» esta vez."""
    return {AUSENCIA: 0, **{c: i for i, c in enumerate(NIVELES[nivel], 1)}}


# Por qué un punto no entra como muestra. La clave es estable (va a los ficheros);
# el texto es lo que se enseña.
MOTIVOS = {
    'tipo_sin_revisar': 'borreguil con este nivel del tipo sin revisar',
    'sin_etiqueta': 'ni borreguil revisado ni ausencia de campo',
    'sin_si_no': 'nadie ha dicho si es borreguil (ni en campo ni revisando su tipo)',
    'dudoso': 'marcado como dudoso en campo',
    'incoherente': 'ausencia de campo que además tiene un tipo revisado',
    'categoria_no_valida': 'categoría que no existe en este nivel',
    'ausencia_no_incluida': 'ausencia de campo (no se ha incluido la clase «no borreguil»)',
    'duplicado': 'mismo píxel de 10 m que otra muestra de su misma categoría',
    'conflicto': 'mismo píxel de 10 m que otra muestra de distinta categoría',
    'sin_coordenadas': 'sin coordenadas válidas',
}
MOTIVOS_EN = {
    'tipo_sin_revisar': 'borreguil whose type has not been reviewed at this level',
    'sin_etiqueta': 'neither a reviewed borreguil nor a field absence',
    'sin_si_no': 'nobody has said whether it is a borreguil (neither in the field nor '
                 'by reviewing its type)',
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
ORIGEN_CAMPO = 'borreguil de campo'


def motivo(clave, en=False):
    """Texto del motivo por el que un punto no entra como muestra."""
    return (MOTIVOS_EN if en else MOTIVOS).get(clave, clave)


def _num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def categoria(r, nivel, sufijo=''):
    """Categoría de un punto en un nivel. En un nivel combinado, la de sus partes
    unida («ladera · húmedo»), o vacía si le falta alguna. Con sufijo='_regla', lo
    que propuso la app."""
    partes = COMBINADOS.get(nivel)
    if not partes:
        return bp._txt(r.get(nivel + sufijo))
    valores = [bp._txt(r.get(p + sufijo)) for p in partes]
    return SEP.join(valores) if all(valores) else ''


def _clasifica(r, nivel, incluir_ausencias, es_borreguil):
    """(categoría, origen, motivo): la categoría con la que el punto entraría como
    muestra y de dónde sale, o el motivo por el que no entra."""
    if _num(r.get('lon')) is None or _num(r.get('lat')) is None:
        return None, None, 'sin_coordenadas'
    revisados = bp.niveles_revisados(r)
    revisado = bool(revisados)
    campo = bp._txt(r.get('Borreguil')).lower()
    if revisado and campo == 'no':
        return None, None, 'incoherente'
    if bp._txt(r.get('Duda')).lower() == 'si':
        return None, None, 'dudoso'
    if nivel == NIVEL_SI_NO:
        # Solo si es borreguil o no: vale lo dicho en campo, o haber revisado su tipo
        # (quien lo revisa lo da por borreguil). Lo que propone el detector, no.
        if campo == 'si' or revisado:
            return 'borreguil', ORIGEN_REVISADO if revisado else ORIGEN_CAMPO, None
        if campo == 'no':
            if incluir_ausencias:
                return AUSENCIA, ORIGEN_AUSENCIA, None
            return None, None, 'ausencia_no_incluida'
        return None, None, 'sin_si_no'
    # Cada nivel cuenta por separado: entra si están revisados los niveles de los que
    # sale su categoría (en uno combinado, todas sus partes).
    if all(p in revisados for p in COMBINADOS.get(nivel, (nivel,))):
        valor = categoria(r, nivel)
        if valor in NIVELES[nivel]:
            return valor, ORIGEN_REVISADO, None
        return None, None, 'categoria_no_valida'
    if campo == 'no':
        if incluir_ausencias:
            return AUSENCIA, ORIGEN_AUSENCIA, None
        return None, None, 'ausencia_no_incluida'
    if campo == 'si' or revisado or es_borreguil(bp._txt(r.get('decision'))):
        return None, None, 'tipo_sin_revisar'
    return None, None, 'sin_etiqueta'


def muestras(rows, nivel, incluir_ausencias=True, es_borreguil=None):
    """Selecciona las muestras de entrenamiento de un nivel y explica el resto.

    Entra como muestra:
      · un punto con el tipo REVISADO, con la categoría que dejó la persona;
      · una AUSENCIA de campo (Borreguil = no), como clase «no borreguil», si
        `incluir_ausencias`.
    En el nivel de sí o no (NIVEL_SI_NO), un punto es «borreguil» si en campo se dijo
    que lo es (Borreguil = si) o si se ha revisado su tipo, aunque el tipo no se use.
    En los niveles de tipo cuenta cada nivel por separado: basta con que esté revisado
    ese nivel (en uno combinado, todas sus partes), aunque los demás no.
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
            'propuesta_app': categoria(r, nivel, '_regla'),
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


def mensaje(item, en=False, traduce=None, nivel=None):
    """Texto de un error o aviso de `muestras()`, en español o en inglés.
    `traduce` traduce el nombre de la categoría (los datos lo guardan en español).
    `nivel`: el de las muestras; en el de sí o no, «sin muestras» dice cómo
    conseguirlas en ese nivel (ahí no hace falta revisar el tipo)."""
    clave = item[0]
    if len(item) > 1 and isinstance(item[1], str) and traduce:
        item = (clave, traduce(item[1])) + tuple(item[2:])
    if clave == 'pocas_categorias':
        if item[1] == 0 and nivel == NIVEL_SI_NO:
            return ('There are no samples yet. Mark some points as verified borreguil and '
                    'others as verified not-borreguil in the Review points tab, or add them by '
                    'clicking on the map (Points map tab).' if en else
                    'Todavía no hay ninguna muestra. Marca algunos puntos como borreguil '
                    'verificado y otros como no borreguil verificado en la pestaña Revisar puntos, o '
                    'añádelos pinchando en el mapa (pestaña Mapa de los puntos).')
        if item[1] == 0:
            return ('There are no samples yet. Review the type of some borreguiles in the '
                    'Review points tab: only reviewed points count.' if en else
                    'Todavía no hay ninguna muestra. Revisa el tipo de algunos borreguiles '
                    'en la pestaña Revisar puntos: solo cuentan los puntos revisados.')
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
        base = {'ID': f['ID'], 'indice': f['indice'], 'lon': float(f['lon']),
                'lat': float(f['lat']), 'categoria': f['categoria'], 'codigo': f['codigo']}
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


def mensaje_calidad(item, en=False, nombre=None, num=None):
    """Texto de un hallazgo de `calidad()`. `nombre` da el nombre legible de una
    variable y `num(v, signo=…)` escribe un número (con coma en español); sin ellos,
    el código de la variable y el número con punto."""
    k = item[0]
    if nombre:
        # las variables: la 1.ª de cada hallazgo y, en los de dos variables, la 2.ª
        n_var = {'constante': 1, 'casi_constante': 1, 'duplicadas': 2, 'opuestas': 2,
                 'correlacion': 2}.get(k, 0)
        item = (k,) + tuple(nombre(x) if i < n_var else x for i, x in enumerate(item[1:]))
    num = num or (lambda v, signo=False: f'{v:+.3f}' if signo else f'{v:.3f}')
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
        return (f'"{item[1]}" and "{item[2]}" move together (r = {num(item[3], signo=True)}).'
                if en else
                f'«{item[1]}» y «{item[2]}» van juntas (r = {num(item[3], signo=True)}).')
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
# Variables repetidas: el VIF (opcional)
# ------------------------------------------------------------
# El VIF (factor de inflación de la varianza) de una variable es 1 / (1 − R²), con R²
# lo que se explica de ella con todas las demás juntas (regresión lineal): 1 = nada,
# 5 = el 80 %, 10 = el 90 %. Para quedarse con variables que no se repitan, se quita
# la de VIF más alto, se vuelve a calcular y se sigue mientras alguna pase del umbral.
# El Random Forest no lo necesita para acertar; sirve para tener un modelo más sencillo
# y un peso de las variables que se lea mejor. Si acierta igual, lo dice la validación.
# ============================================================
VIF_UMBRAL = 10.0
# La altitud no se quita nunca: lo pidió la usuaria («en cualquier sitio del mundo los
# borreguiles son prados húmedos de alta montaña»). Sí cuenta en el VIF de las demás.
VIF_FIJAS = ('elev_dem_m',)


def vif(X):
    """El VIF de cada columna de X (muestras × variables): infinito si las demás la
    explican entera (es una suma o una resta de otras) y NaN si no varía."""
    import numpy as np
    X = np.asarray(X, dtype=float)
    n, k = X.shape
    sd = X.std(axis=0)
    varia = sd > 1e-12 * np.maximum(1.0, np.abs(X).max(axis=0))
    Z = np.where(varia, (X - X.mean(axis=0)) / np.where(varia, sd, 1.0), 0.0)
    salida = np.full(k, np.nan)
    for j in range(k):
        if not varia[j]:
            continue
        otras = [i for i in range(k) if i != j and varia[i]]
        if not otras:
            salida[j] = 1.0
            continue
        A = np.column_stack([np.ones(n), Z[:, otras]])
        coef = np.linalg.lstsq(A, Z[:, j], rcond=None)[0]
        resto = Z[:, j] - A @ coef
        r2 = 1.0 - float(resto @ resto) / float(Z[:, j] @ Z[:, j])
        salida[j] = np.inf if r2 >= 1.0 - 1e-10 else 1.0 / (1.0 - r2)
    return salida


def elegir_por_vif(X, nombres, umbral=VIF_UMBRAL, fijas=VIF_FIJAS):
    """Quita variables repetidas de una en una: en cada paso calcula el VIF de las que
    quedan y quita la de VIF más alto si pasa del `umbral` (las que no varían, antes que
    ninguna); para cuando todas quedan por debajo. Las de `fijas` no se quitan nunca,
    pero cuentan al calcular el VIF de las demás. Si empatan dos, se quita la que va
    más atrás en el catálogo: la caída del NDVI antes que los NDVI de los que sale.

    Devuelve un dict: quedan (en su orden), quitadas [{variable, vif, paso}] en el
    orden en que se quitaron (vif None = no variaba), vif {variable: VIF final de las
    que quedan}, umbral, fijas y pocas_muestras (con menos del doble de muestras que
    variables el VIF no es fiable).
    """
    import numpy as np
    X = np.asarray(X, dtype=float)
    nombres = list(nombres)
    if X.ndim != 2 or X.shape[1] != len(nombres):
        raise ValueError('la tabla y la lista de variables no casan')
    if not np.isfinite(X).all():
        raise ValueError('la tabla contiene valores no válidos (huecos o infinitos)')
    umbral = float(umbral)
    if not umbral > 1.0:
        raise ValueError('el umbral del VIF tiene que ser mayor que 1')
    fijas = [f for f in dict.fromkeys(fijas or ()) if f in nombres]
    catalogo = {n: i for i, n in enumerate(nombres_predictores())}
    orden = {n: catalogo.get(n, len(catalogo) + i) for i, n in enumerate(nombres)}
    quedan, quitadas = list(range(len(nombres))), []
    while True:
        v = vif(X[:, quedan])
        # primero las que no varían; después, la de VIF más alto; y en el empate, la
        # que va más atrás en el catálogo
        candidatas = [(bool(np.isnan(v[i])), np.inf if np.isnan(v[i]) else v[i],
                       orden[nombres[c]], c)
                      for i, c in enumerate(quedan) if nombres[c] not in fijas]
        if not candidatas:
            break
        constante, peor, _, c = max(candidatas)
        if not peor > umbral:
            break
        quitadas.append({'variable': nombres[c], 'vif': None if constante else float(peor),
                         'paso': len(quitadas) + 1})
        quedan.remove(c)
    final = vif(X[:, quedan])
    return {'quedan': [nombres[c] for c in quedan], 'quitadas': quitadas,
            'vif': {nombres[c]: (None if np.isnan(x) else float(x))
                    for c, x in zip(quedan, final)},
            'umbral': umbral, 'fijas': fijas,
            'pocas_muestras': X.shape[0] < 2 * X.shape[1]}


# Elegir variables con un PCA (24-11-2026), como en la comparación de la autora del
# 22-10 (modelo_borreguil_sn2000m_v3/comparacion_variables): del PCA de la matriz de
# correlación, los ejes con valor propio mayor que 1 (criterio de Kaiser); de cada eje,
# entre las variables muy ligadas a él (|correlación| ≥ PCA_CARGA), la que mejor separa
# las categorías por sí sola. Las fijas (la altitud) quedan fuera del PCA y se añaden
# siempre. En Sierra Nevada, el PCA con la altitud dejó 4 variables con el mismo acierto
# que el VIF con 14 (F1 0,861 frente a 0,865), pero cartografió un 38 % más de borreguil.
# Allí se hizo con 30 000 píxeles de la zona; aquí, con las muestras, como el VIF.
PCA_VALOR_PROPIO = 1.0
PCA_CARGA = 0.65


def separacion(x, y):
    """Lo bien que separa la variable `x` sola las categorías `y`: con dos, |AUC − 0,5|
    × 2 (0 no separa, 1 separa del todo); con más, la F del ANOVA. Solo sirve para
    ordenar variables entre sí."""
    import numpy as np
    x, y = np.asarray(x, dtype=float), np.asarray(y)
    clases = np.unique(y)
    if len(clases) < 2 or np.nanstd(x) == 0:
        return 0.0
    if len(clases) == 2:
        from sklearn.metrics import roc_auc_score
        return float(abs(roc_auc_score(y == clases[1], x) - 0.5) * 2)
    from sklearn.feature_selection import f_classif
    f = f_classif(x.reshape(-1, 1), y)[0][0]
    return float(f) if np.isfinite(f) else 0.0


def elegir_por_pca(X, y, nombres, fijas=VIF_FIJAS, carga=PCA_CARGA,
                   valor_propio=PCA_VALOR_PROPIO):
    """Una variable por eje importante de un PCA (ver arriba). X: muestras × variables;
    y: la categoría de cada muestra.

    Devuelve un dict: quedan (en el orden de `nombres`), ejes [{eje, valor_propio,
    varianza_pct, variables [{variable, carga, separa}], elegida}] de los ejes con valor
    propio > `valor_propio`, varianza_pct (la de esos ejes juntos), carga, fijas,
    sin_variar (las que no varían: fuera), pocas_muestras (con menos del doble de
    muestras que variables el PCA no es fiable), valores_propios (los de TODOS los ejes,
    de mayor a menor), cargas {variable: [correlación con cada eje importante]} y
    separa {variable: lo bien que separa las categorías sola} de las que entran."""
    import numpy as np
    X = np.asarray(X, dtype=float)
    nombres = list(nombres)
    if X.ndim != 2 or X.shape[1] != len(nombres):
        raise ValueError('la tabla y la lista de variables no casan')
    if not np.isfinite(X).all():
        raise ValueError('la tabla contiene valores no válidos (huecos o infinitos)')
    carga = float(carga)
    if not 0 < carga < 1:
        raise ValueError('la correlación mínima con el eje tiene que estar entre 0 y 1')
    fijas = [f for f in dict.fromkeys(fijas or ()) if f in nombres]
    varia = X.std(axis=0) > 1e-12 * np.maximum(1.0, np.abs(X).max(axis=0))
    sin_variar = [n for n, v in zip(nombres, varia) if not v and n not in fijas]
    usar = [i for i, n in enumerate(nombres) if varia[i] and n not in fijas]
    ejes, elegidas = [], []
    total, valores, cargas_de, sep = 0.0, [], {}, {}
    if len(usar) >= 2 and X.shape[0] >= 3:
        Z = X[:, usar]
        Z = (Z - Z.mean(axis=0)) / Z.std(axis=0)
        val, vec = np.linalg.eigh(np.corrcoef(Z, rowvar=False))
        orden = np.argsort(val)[::-1]
        val, vec = np.clip(val[orden], 0, None), vec[:, orden]
        cargas = vec * np.sqrt(val)            # correlación de cada variable con cada eje
        sep = {nombres[i]: separacion(X[:, i], y) for i in usar}
        for k in range(len(val)):
            if not val[k] > valor_propio:
                break
            ligadas = sorted(
                ({'variable': nombres[usar[j]], 'carga': float(cargas[j, k]),
                  'separa': sep[nombres[usar[j]]]}
                 for j in range(len(usar)) if abs(cargas[j, k]) >= carga),
                key=lambda d: -abs(d['carga']))
            libres = [d for d in ligadas if d['variable'] not in elegidas]
            elegida = max(libres, key=lambda d: d['separa'])['variable'] if libres else None
            if elegida:
                elegidas.append(elegida)
            ejes.append({'eje': k + 1, 'valor_propio': float(val[k]),
                         'varianza_pct': float(100 * val[k] / val.sum()),
                         'variables': ligadas, 'elegida': elegida})
        total = float(sum(e['varianza_pct'] for e in ejes))
        valores = [float(v) for v in val]
        cargas_de = {nombres[usar[j]]: [float(cargas[j, k]) for k in range(len(ejes))]
                     for j in range(len(usar))}
        if not elegidas:
            # ningún eje tiene variables tan ligadas: la más ligada al primero
            j = int(np.argmax(np.abs(cargas[:, 0])))
            elegidas.append(nombres[usar[j]])
    elif len(usar) == 1:
        elegidas = [nombres[usar[0]]]
    quedan = [n for n in nombres if n in elegidas or n in fijas]
    return {'quedan': quedan, 'ejes': ejes, 'varianza_pct': total, 'carga': carga,
            'fijas': fijas, 'sin_variar': sin_variar,
            'pocas_muestras': X.shape[0] < 2 * X.shape[1],
            'valores_propios': valores, 'cargas': cargas_de, 'separa': sep}


# Por qué se queda o se quita cada variable (25-11-2026, lo pidió la autora: «que te
# muestre qué variables se han quedado y algún output del análisis para entender por
# qué»). Una fila por variable, con lo que hace falta para contarlo en una frase.
def por_que_vif(X, nombres, sel):
    """Una fila por variable a partir de elegir_por_vif(X, nombres) = `sel`, primero las
    que se quedan (en su orden) y después las quitadas (por pasos): {variable, que
    ('fija', 'queda', 'quitada' o 'no_varia'), vif (None si no varía; inf si se obtiene
    exactamente de otras), explicado_pct (lo que las demás explican de ella: 1 − 1/VIF),
    paso (al quitarla), parecida y r (la que más se le parece, por la correlación, de
    entre las que se quedan; None si no hay)}."""
    import numpy as np
    X = np.asarray(X, dtype=float)
    col = {n: i for i, n in enumerate(nombres)}
    with np.errstate(invalid='ignore', divide='ignore'):
        R = np.corrcoef(X, rowvar=False)
    R = np.atleast_2d(R)

    def _explicado(v):
        if v is None:
            return None
        return 100.0 if v == float('inf') else max(0.0, 100.0 * (1.0 - 1.0 / v))

    def _parecida(n):
        mejor, r = None, None
        for m in sel['quedan']:
            if m == n:
                continue
            x = R[col[n], col[m]]
            if np.isfinite(x) and (r is None or abs(x) > abs(r)):
                mejor, r = m, float(x)
        return mejor, r

    filas = []
    for n in sel['quedan']:
        v = sel['vif'].get(n)
        p, r = _parecida(n)
        filas.append({'variable': n, 'que': 'fija' if n in sel['fijas'] else 'queda',
                      'vif': v, 'explicado_pct': _explicado(v), 'paso': None,
                      'parecida': p, 'r': r})
    for q in sel['quitadas']:
        p, r = _parecida(q['variable'])
        filas.append({'variable': q['variable'],
                      'que': 'no_varia' if q['vif'] is None else 'quitada',
                      'vif': q['vif'], 'explicado_pct': _explicado(q['vif']),
                      'paso': q['paso'], 'parecida': p, 'r': r})
    return filas


def por_que_pca(sel, nombres):
    """Una fila por variable a partir de elegir_por_pca(…, nombres) = `sel`, primero las
    que se quedan: {variable, que ('fija', 'elegida', 'ligada', 'suelta' o 'no_varia'),
    eje (el suyo: el que la elige, o al que más ligada está), carga (su correlación con
    ese eje), separa (lo bien que separa sola), ganadora y separa_ganadora (la que se
    queda en ese eje, para las 'ligada')}. 'suelta' = no llega a la correlación mínima
    con ningún eje importante."""
    ejes = sel['ejes']
    elegida_en = {e['elegida']: e for e in ejes if e['elegida']}
    filas = []
    for n in nombres:
        fila = {'variable': n, 'eje': None, 'carga': None,
                'separa': sel.get('separa', {}).get(n), 'ganadora': None,
                'separa_ganadora': None}
        cargas = sel.get('cargas', {}).get(n) or []
        if n in sel['fijas']:
            fila['que'] = 'fija'
        elif n in sel['sin_variar']:
            fila['que'] = 'no_varia'
        elif n in elegida_en:
            e = elegida_en[n]
            fila.update(que='elegida', eje=e['eje'],
                        carga=next(d['carga'] for d in e['variables'] if d['variable'] == n))
        elif n in sel['quedan']:
            # ningún eje tenía variables tan ligadas: la más ligada al primero
            fila.update(que='elegida', eje=1 if ejes else None,
                        carga=cargas[0] if cargas else None)
        else:
            ligado = [(abs(d['carga']), e) for e in ejes for d in e['variables']
                      if d['variable'] == n]
            if ligado:
                _, e = max(ligado, key=lambda t: t[0])
                g = e['elegida']
                fila.update(que='ligada', eje=e['eje'],
                            carga=next(d['carga'] for d in e['variables']
                                       if d['variable'] == n),
                            ganadora=g,
                            separa_ganadora=next((d['separa'] for d in e['variables']
                                                  if d['variable'] == g), None))
            else:
                fila['que'] = 'suelta'
                if cargas:
                    k = max(range(len(cargas)), key=lambda i: abs(cargas[i]))
                    fila.update(eje=k + 1, carga=cargas[k])
        filas.append(fila)
    orden = {'fija': 0, 'elegida': 0, 'ligada': 1, 'suelta': 2, 'no_varia': 3}
    return sorted(filas, key=lambda f: orden[f['que']])


def texto_por_que(fila, metodo, en=False, nombre=str, num=None, umbral=None, carga=None):
    """La frase de por qué se queda o se quita una variable (una fila de por_que_vif o
    por_que_pca). `nombre` escribe el nombre de una variable y `num(v, nd)` un número
    (con coma en español); `umbral` es el del VIF y `carga`, la correlación mínima del
    PCA."""
    if num is None:
        def num(v, nd=3):
            return f'{v:.{nd}f}'

    def t(es, en_):
        return en_ if en else es

    q = fila['que']
    if q == 'no_varia':
        return t('No varía en las muestras: no aporta nada.',
                 'It does not vary across the samples: it adds nothing.')
    if metodo == 'vif':
        v, pct = fila['vif'], fila['explicado_pct']
        lim = None if umbral is None else 100.0 * (1.0 - 1.0 / umbral)
        # el porcentaje con un decimal (90,2 % frente al umbral del 90 %; 99,9 % y no
        # «100 %» si no es exacto); el del umbral, entero si lo es
        p_txt = None if pct is None else num(pct, 1)
        l_txt = None if lim is None else num(lim, 0 if abs(lim - round(lim)) < 0.05 else 1)
        if v == float('inf'):
            cuanto = t('Se obtiene exactamente de otras (VIF ∞)',
                       'It is obtained exactly from others (VIF ∞)')
        elif q == 'quitada':
            cuanto = t(f'Las demás explicaban el {p_txt} % de ella (VIF {num(v)})',
                       f'The others explained {p_txt} % of it (VIF {num(v)})')
        else:
            cuanto = t(f'Las demás explican el {p_txt} % de ella (VIF {num(v)})',
                       f'The others explain {p_txt} % of it (VIF {num(v)})')
        if lim is None or v == float('inf'):
            frente = ''
        elif q == 'quitada':
            frente = t(f', más que el umbral ({l_txt} %)',
                       f', more than the threshold ({l_txt} %)')
        elif q == 'queda':
            frente = t(f', no más que el umbral ({l_txt} %)',
                       f', not more than the threshold ({l_txt} %)')
        else:
            frente = ''
        par = ''
        if fila['parecida']:
            if q == 'quitada':
                par = t(f' Se parece sobre todo a {nombre(fila["parecida"])} '
                        f'(r {num(fila["r"])}), que se queda.',
                        f' It is most similar to {nombre(fila["parecida"])} '
                        f'(r {num(fila["r"])}), which stays.')
            else:
                par = t(f' La que más se le parece de las que se quedan: '
                        f'{nombre(fila["parecida"])} (r {num(fila["r"])}).',
                        f' The most similar one among those that stay: '
                        f'{nombre(fila["parecida"])} (r {num(fila["r"])}).')
        if q == 'fija':
            return t('Se queda siempre (la elegiste así). ',
                     'Always stays (you chose so). ') + cuanto + '.' + par
        return cuanto + frente + '.' + par
    # PCA
    sep = '—' if fila['separa'] is None else num(fila['separa'])
    if q == 'fija':
        return t('Se queda siempre (la elegiste así): no entra en el PCA.',
                 'Always stays (you chose so): it is left out of the PCA.')
    if q == 'elegida':
        return t(f'De las ligadas al eje {fila["eje"]}, es la que mejor separa las '
                 f'categorías por sí sola ({sep}). Su correlación con el eje: '
                 f'{num(fila["carga"])}.',
                 f'Among those tied to axis {fila["eje"]}, it is the one that best separates '
                 f'the categories on its own ({sep}). Its correlation with the axis: '
                 f'{num(fila["carga"])}.')
    if q == 'ligada':
        if fila['ganadora']:
            sg = '—' if fila['separa_ganadora'] is None else num(fila['separa_ganadora'])
            if sg == sep:       # empate: se queda la más ligada al eje
                return t(f'Está ligada al eje {fila["eje"]} (correlación '
                         f'{num(fila["carga"])}), pero de ese eje se queda '
                         f'{nombre(fila["ganadora"])}: separa igual de bien ({sep}) y está '
                         'más ligada al eje.',
                         f'It is tied to axis {fila["eje"]} (correlation '
                         f'{num(fila["carga"])}), but {nombre(fila["ganadora"])} stays for '
                         f'that axis: it separates just as well ({sep}) and is more tied to '
                         'the axis.')
            return t(f'Está ligada al eje {fila["eje"]} (correlación {num(fila["carga"])}), '
                     f'pero de ese eje se queda {nombre(fila["ganadora"])}, que separa mejor '
                     f'({sg} frente a {sep}).',
                     f'It is tied to axis {fila["eje"]} (correlation {num(fila["carga"])}), '
                     f'but {nombre(fila["ganadora"])} stays for that axis, as it separates '
                     f'better ({sg} against {sep}).')
        return t(f'Está ligada al eje {fila["eje"]} (correlación {num(fila["carga"])}), pero '
                 'ese eje ya está representado por otra variable.',
                 f'It is tied to axis {fila["eje"]} (correlation {num(fila["carga"])}), but '
                 'that axis is already represented by another variable.')
    # suelta
    if fila['eje'] is None:
        return t('No está muy ligada a ningún eje importante.',
                 'It is not strongly tied to any important axis.')
    c = '' if carga is None else t(f'; hace falta {num(carga)}', f'; {num(carga)} is needed')
    return t(f'No está muy ligada a ningún eje importante (como mucho '
             f'{num(abs(fila["carga"]))}, con el eje {fila["eje"]}{c}).',
             f'It is not strongly tied to any important axis (at most '
             f'{num(abs(fila["carga"]))}, with axis {fila["eje"]}{c}).')


# ============================================================
# Random Forest multiclase: validación por zonas y modelo final
# ------------------------------------------------------------
# Son dos procesos separados:
#   1. VALIDAR: se reparte la tabla en partes; cada parte se predice con un modelo
#      entrenado solo con las demás. De ahí salen la matriz de confusión y las
#      métricas. Las partes se hacen por ZONAS (cuenca o celda de ~5 km): los puntos
#      cercanos se parecen mucho, y si vecinos del mismo sitio cayeran a la vez en
#      entrenamiento y en validación, el resultado saldría mejor de lo que es.
#   2. MODELO FINAL: se entrena otra vez con TODAS las muestras válidas. Ese es el
#      que después hace el mapa; sus métricas son las del paso 1.
# ============================================================
PARAMS_DEFECTO = {
    'n_estimators': 500,        # árboles
    'max_features': 'sqrt',     # variables que se prueban en cada división
    'max_depth': None,          # sin límite de profundidad
    'min_samples_leaf': 2,      # muestras mínimas por hoja
    'random_state': 42,         # semilla: mismo dato, mismo modelo
}
# Qué se hace cuando unas categorías tienen muchas más muestras que otras.
ESTRATEGIAS = ('pesos', 'submuestreo', 'ambas', 'ninguna')
RAZON_MAXIMA = 3.0              # submuestreo: la abundante, como mucho 3 × la escasa
CELDA_GRADOS = 0.05             # ~5 km: lado de la celda cuando el punto no tiene cuenca
PARTICIONES = 5
MIN_ZONAS = 3                   # con menos zonas no se puede validar por zonas


# Importancia de las variables por permutación: cuántas veces se desordena cada una
# (más con pocas muestras, que es cuando más baila), cuántas filas se predicen como
# mucho en total, cuántas se tienen a la vez en memoria y de cuántas es cada trozo de
# los que se predicen a la vez.
REPETICIONES_MIN, REPETICIONES_MAX = 5, 30
FILAS_PERMUTADAS = 400_000
LOTE_PERMUTACION = 400_000
TROZO_PERMUTACION = 16_000


def _macro_f1(y, pred, presentes):
    """Media del F1 de las categorías `presentes`. `pred` puede traer muchas
    predicciones a la vez (…, nº de muestras): devuelve una cifra por cada una. Da lo
    mismo que f1_score(average='macro', labels=presentes, zero_division=0)."""
    import numpy as np
    f1s = []
    for c in presentes:
        es, dice = (y == c), (pred == c)
        aciertos = (dice & es).sum(axis=-1)
        total = es.sum() + dice.sum(axis=-1)
        f1s.append(np.where(total > 0, 2 * aciertos / np.maximum(total, 1), 0.0))
    return np.mean(f1s, axis=0)


# ============================================================
# ¿Borreguil o no? La regla de decisión
# ------------------------------------------------------------
# El modelo da, para cada caso, la probabilidad de cada categoría. La categoría se
# decide en dos pasos:
#   1. ¿es borreguil? Sí, si la probabilidad de serlo —la suma de la de sus tipos, que
#      es lo mismo que 1 − la de «no borreguil»— llega a un UMBRAL;
#   2. ¿de qué tipo? El más probable entre los tipos.
# Quedarse sin más con la categoría más probable perjudica a los borreguiles: su
# probabilidad se reparte entre los tipos mientras «no borreguil» conserva la suya
# entera (con 0.40 de «no borreguil», 0.35 de «húmedo» y 0.25 de «seco» saldría «no
# borreguil» habiendo 0.60 de que lo sea), y más cuantos más tipos haya.
# El umbral se elige con la validación: se prueban muchos y se mira cuál separa mejor
# los borreguiles de lo que no lo es.
# ============================================================
CODIGO_AUSENCIA = 0
UMBRALES = tuple(round(0.025 * k, 3) for k in range(1, 40))     # 0.025, 0.050 … 0.975
# Cómo se elige el umbral: el de mayor F1 de «borreguil», el de mayor TSS
# (sensibilidad + especificidad − 1), o el que fije la persona. De entrada, el F1: el
# borreguil es escaso, y con el TSS el umbral baja tanto que el mapa se llena de pasto
# que no lo es (lo vio la usuaria en Sierra Nevada, por encima de 2000 m).
CRITERIOS = ('f1', 'tss', 'manual')
CRITERIO_DEFECTO = 'f1'


def repartir(prob, clases):
    """De la probabilidad de cada categoría a lo que hace falta para decidir.

    prob: matriz (…, nº de categorías), con las columnas en el orden de `clases` (los
    códigos del modelo). Devuelve (p_borr, tipo, p_tipo, p_no):

      p_borr  probabilidad de ser borreguil, del tipo que sea: la suma de la de los
              tipos (lo mismo que 1 − la de «no borreguil»)
      tipo    código del tipo más probable (si empatan, el de código más bajo)
      p_tipo  probabilidad de ese tipo
      p_no    probabilidad de «no borreguil», o None si el modelo no tiene esa
              categoría

    Las cuentas se hacen en float32, que es como se guardan las probabilidades en el
    mapa: lo que se decide aquí es exactamente lo que se decidiría leyendo el fichero.
    """
    import numpy as np
    pr = np.asarray(prob, dtype=np.float32)
    clases = [int(c) for c in clases]
    if pr.shape[-1] != len(clases):
        raise ValueError('hay que dar una probabilidad por categoría')
    tipos = [j for j, c in enumerate(clases) if c != CODIGO_AUSENCIA]
    p_no = pr[..., clases.index(CODIGO_AUSENCIA)] if CODIGO_AUSENCIA in clases else None
    forma = pr.shape[:-1]
    if not tipos:                        # un modelo que solo ha visto «no borreguil»
        return (np.zeros(forma, np.float32), np.full(forma, -1, dtype=int),
                np.zeros(forma, np.float32), p_no)
    p_borr = np.zeros(forma, dtype=np.float32)
    for j in tipos:                      # siempre en el mismo orden: la suma no baila
        p_borr = p_borr + pr[..., j]
    de_tipos = pr[..., tipos]
    mejor = de_tipos.argmax(axis=-1)
    tipo = np.asarray([clases[j] for j in tipos], dtype=int)[mejor]
    p_tipo = np.take_along_axis(de_tipos, mejor[..., None], axis=-1)[..., 0]
    return p_borr, tipo, p_tipo, p_no


def decidir(prob, clases, umbral=None):
    """La categoría de cada caso y con qué probabilidad.

    Con `umbral`, en dos pasos: «no borreguil» si la probabilidad de ser borreguil no
    llega al umbral; si llega, su tipo más probable. Si el modelo no tiene la
    categoría «no borreguil», no hay nada que decidir en el primer paso: cada caso es
    de su tipo más probable. Con `umbral=None` no se aplica la regla y se devuelve,
    sin más, la categoría más probable de todas (si empatan, la de código más bajo).

    Devuelve (categoria, p_categoria, p_borr): el código elegido, su probabilidad y
    la probabilidad de ser borreguil de algún tipo.
    """
    import numpy as np
    p_borr, tipo, p_tipo, p_no = repartir(prob, clases)
    if p_no is None:
        return tipo, p_tipo, p_borr
    es = (p_tipo > p_no) if umbral is None else (p_borr >= np.float32(umbral))
    return (np.where(es, tipo, CODIGO_AUSENCIA), np.where(es, p_tipo, p_no), p_borr)


def _binarias(p_borr, es_borr, umbral):
    """Aciertos y fallos de «¿borreguil o no?» con un umbral, y sus medidas. Si no se
    puede calcular alguna (nada que dividir), vale 0."""
    import numpy as np
    dice = np.asarray(p_borr, dtype=np.float32) >= np.float32(umbral)
    es_borr = np.asarray(es_borr, dtype=bool)
    vp, fp = int(np.sum(dice & es_borr)), int(np.sum(dice & ~es_borr))
    fn, vn = int(np.sum(~dice & es_borr)), int(np.sum(~dice & ~es_borr))
    precision = vp / (vp + fp) if vp + fp else 0.0
    sensibilidad = vp / (vp + fn) if vp + fn else 0.0
    especificidad = vn / (vn + fp) if vn + fp else 0.0
    f1 = (2 * precision * sensibilidad / (precision + sensibilidad)
          if precision + sensibilidad else 0.0)
    return {'umbral': float(umbral), 'vp': vp, 'fp': fp, 'fn': fn, 'vn': vn,
            'sensibilidad': sensibilidad, 'especificidad': especificidad,
            'precision': precision, 'f1': f1, 'tss': sensibilidad + especificidad - 1}


def barrido_umbral(p_borr, es_borr, umbrales=UMBRALES):
    """Qué pasaría con cada umbral: una fila por umbral, con los borreguiles que se
    encuentran (vp) y los que se escapan (fn), lo que se toma por borreguil sin serlo
    (fp) y lo que se descarta bien (vn), y las medidas que salen de ahí."""
    return [_binarias(p_borr, es_borr, u) for u in umbrales]


def elegir_umbral(barrido, criterio=CRITERIO_DEFECTO):
    """La fila del barrido con el mejor `criterio` ('tss' o 'f1').

    Con pocas muestras suelen empatar muchos umbrales seguidos (dan exactamente los
    mismos aciertos). Se toma el del CENTRO del tramo más largo de empatados: es el
    que más margen deja a los dos lados. Entre tramos igual de largos, el más cercano
    a 0.5."""
    valores = [b[criterio] for b in barrido]
    mejor = max(valores)
    tramos, actual = [], []
    for i, v in enumerate(valores):
        if v >= mejor - 1e-12:
            actual.append(i)
        elif actual:
            tramos.append(actual)
            actual = []
    if actual:
        tramos.append(actual)

    def lejos(i):
        return abs(barrido[i]['umbral'] - 0.5)

    def centro(tramo):
        return min((tramo[(len(tramo) - 1) // 2], tramo[len(tramo) // 2]),
                   key=lambda i: (lejos(i), i))

    return barrido[centro(min(tramos, key=lambda t: (-len(t), lejos(centro(t)))))]


def aplicar_regla(mod, criterio=CRITERIO_DEFECTO, umbral=None):
    """Fija cómo se decide «¿borreguil o no?» en un modelo y rehace con esa regla las
    cifras de su validación y el peso de sus variables. No vuelve a entrenar nada: usa
    las probabilidades que el modelo guardó al validar.

    criterio: 'f1' (el umbral de mayor F1 en la validación), 'tss' (el de mayor TSS) o
    'manual' (el de `umbral`, entre 0 y 1). Si el modelo no tiene la categoría «no
    borreguil», no hay umbral: cada caso es de su tipo más probable.

    Cambia `mod` y lo devuelve. La regla queda en mod['regla'].
    """
    import warnings

    import numpy as np
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 confusion_matrix, f1_score,
                                 precision_recall_fscore_support)
    v, perm = mod['_validacion'], mod['_permutacion']
    y, presentes = v['y'], v['presentes']
    nombre = {int(c['codigo']): c['categoria'] for c in mod['categorias']}
    if CODIGO_AUSENCIA in presentes:
        if criterio not in CRITERIOS:
            raise ValueError(f'criterio desconocido: {criterio!r} (use {", ".join(CRITERIOS)})')
        es_borr = y != CODIGO_AUSENCIA
        barrido = barrido_umbral(v['p_borr'], es_borr)
        mejores = {c: elegir_umbral(barrido, c)['umbral'] for c in ('tss', 'f1')}
        if criterio == 'manual':
            try:
                u = float(umbral)
            except (TypeError, ValueError):
                raise ValueError('con el criterio manual hay que dar el umbral') from None
            if not 0 < u < 1:
                raise ValueError('el umbral tiene que estar entre 0 y 1')
        else:
            u = mejores[criterio]
        es = v['p_borr'] >= np.float32(u)
        predicho = np.where(es, v['tipo'], CODIGO_AUSENCIA)
        confianza = np.where(es, v['p_tipo'], v['p_no'])
        desordenado = np.where(perm['p_borr'] >= np.float32(u), perm['tipo'],
                               CODIGO_AUSENCIA)
        regla = {'criterio': criterio, 'umbral': float(u),
                 'mejor_tss': mejores['tss'], 'mejor_f1': mejores['f1'],
                 'barrido': barrido, 'en_umbral': _binarias(v['p_borr'], es_borr, u)}
    else:
        predicho, confianza, desordenado = v['tipo'], v['p_tipo'], perm['tipo']
        regla = {'criterio': None, 'umbral': None, 'barrido': []}

    # ---- la validación, con esa regla
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='y_pred contains classes not in y_true')
        for pl in mod['pliegues']:
            de_la_parte = v['parte'] == pl['parte']
            ya, pa = y[de_la_parte], predicho[de_la_parte]
            pl['exactitud'] = float(accuracy_score(ya, pa))
            pl['exactitud_equilibrada'] = float(balanced_accuracy_score(ya, pa))
            pl['macro_f1'] = float(f1_score(ya, pa, average='macro',
                                            labels=sorted(set(int(c) for c in ya)),
                                            zero_division=0))
        matriz = confusion_matrix(y, predicho, labels=presentes)
        prec, rec, f1, sop = precision_recall_fscore_support(
            y, predicho, labels=presentes, zero_division=0)
        mod['exactitud'] = float(accuracy_score(y, predicho))
        mod['exactitud_equilibrada'] = float(balanced_accuracy_score(y, predicho))
        mod['macro_f1'] = float(f1_score(y, predicho, average='macro', labels=presentes,
                                         zero_division=0))
    mod['matriz'] = [[int(x) for x in fila] for fila in matriz]
    mod['por_clase'] = [{'categoria': nombre[c], 'codigo': c, 'precision': float(prec[i]),
                         'recall': float(rec[i]), 'f1': float(f1[i]),
                         'n_validacion': int(sop[i])} for i, c in enumerate(presentes)]
    mod['confusiones'] = sorted(
        [(nombre[presentes[a]], nombre[presentes[b]], int(matriz[a, b]))
         for a in range(len(presentes)) for b in range(len(presentes))
         if a != b and matriz[a, b] > 0], key=lambda t: -t[2])
    mod['predicciones'] = [{
        'ID': f['ID'], 'indice': f['indice'], 'zona': f['zona'], 'parte': f['parte'],
        'real': nombre[int(y[i])], 'predicha': nombre[int(predicho[i])],
        'acierta': bool(y[i] == predicho[i]), 'confianza': float(confianza[i]),
        'p_borreguil': float(v['p_borr'][i])} for i, f in enumerate(v['filas'])]

    # ---- el peso de las variables, con esa regla
    y_presentes = np.asarray(presentes)
    base = float(_macro_f1(y, predicho, y_presentes))
    caidas = base - _macro_f1(y, desordenado, y_presentes)          # (reps, conjuntos)
    media, desv = caidas.mean(axis=0), caidas.std(axis=0, ddof=1)
    impureza, lugar, variables_ = perm['impureza'], perm['lugar'], mod['predictores']
    nv = len(variables_)
    por_peso = sorted(range(nv), key=lambda j: (-media[j], -impureza[j], j))
    puesto = {j: k for k, j in enumerate(por_peso, 1)}
    puesto_interno = {j: k for k, j in enumerate(
        sorted(range(nv), key=lambda j: (-impureza[j], j)), 1)}

    def juntas(columnas, campo):
        """El peso de cada conjunto de variables, del que más al que menos."""
        return sorted(({campo: clave, 'n_variables': len(cols),
                        'permutacion': float(media[lugar[tuple(cols)]]),
                        'desviacion': float(desv[lugar[tuple(cols)]]),
                        'impureza': float(sum(impureza[j] for j in cols))}
                       for clave, cols in columnas.items()),
                      key=lambda g: -g['permutacion'])

    mod['importancia'] = {
        'medida': 'macro_f1', 'base': base, 'repeticiones': int(perm['repeticiones']),
        'variables': [{
            'variable': variables_[j], 'familia': familia(variables_[j]),
            'grupo': _GRUPO_DE.get(variables_[j], ''),
            'permutacion': float(media[j]), 'desviacion': float(desv[j]),
            'orden': puesto[j], 'impureza': float(impureza[j]),
            'orden_impureza': puesto_interno[j]} for j in por_peso],
        'familias': juntas(perm['de_familia'], 'familia'),
        'grupos': juntas(perm['de_grupo'], 'grupo')}
    mod['regla'] = regla
    return mod


# ============================================================
# El mapa en dos pasos: sí o no, y después el tipo
# ------------------------------------------------------------
# El modelo de sí o no (nivel «borreguil») aprende de todos los puntos de los que se
# sabe si son borreguil, y el de un nivel del tipo, solo de los borreguiles con el tipo
# revisado y sin «no borreguil». Juntos se comportan como un único modelo con «no
# borreguil» y los tipos, y así el mapa se hace igual que con uno solo:
#   P(no borreguil) = 1 − p        P(tipo) = p × P(tipo si es borreguil)
# con p la probabilidad de ser borreguil del modelo de sí o no. La suma de los tipos
# es p: lo que decide el umbral es el modelo de sí o no, y el tipo, el otro.
# ============================================================
class DosPasos:
    """Dos Random Forest que predicen como uno: el de sí o no y el del tipo. Cada uno
    lee sus columnas de la misma tabla de variables (las de los dos, en el orden del
    catálogo)."""

    def __init__(self, si_no, tipo, cols_si_no, cols_tipo, n_variables):
        import numpy as np
        if CODIGO_AUSENCIA in [int(c) for c in tipo.classes_]:
            raise ValueError('el modelo del tipo no puede tener «no borreguil»')
        self.si_no, self.tipo = si_no, tipo
        self.cols_si_no, self.cols_tipo = list(cols_si_no), list(cols_tipo)
        self.n_features_in_ = int(n_variables)
        self.classes_ = np.asarray([CODIGO_AUSENCIA] + [int(c) for c in tipo.classes_])
        self._si = [int(c) for c in si_no.classes_].index(1)

    def predict_proba(self, X):
        import numpy as np
        X = np.asarray(X)
        p = self.si_no.predict_proba(X[:, self.cols_si_no])[:, self._si]
        de_tipo = self.tipo.predict_proba(X[:, self.cols_tipo])
        return np.column_stack([1 - p, p[:, None] * de_tipo])


def en_dos_pasos(mod_si_no, mod_tipo):
    """El modelo con el que se hace el mapa en dos pasos (ver DosPasos), con lo mismo
    que trae un modelo de entrenar_y_validar para hacer el mapa: nivel, categorías,
    predictores, modelo, regla (la del de sí o no, con su umbral), firma y lectura.

    mod_si_no: el del nivel «borreguil», con «no borreguil» y su umbral.
    mod_tipo: el de un nivel del tipo, entrenado sin «no borreguil».
    Lanza ValueError si no casan (otro nivel, otra lectura de la pila…).
    """
    import hashlib
    if mod_si_no.get('nivel') != NIVEL_SI_NO:
        raise ValueError('el primer paso tiene que ser el modelo de sí o no')
    if mod_tipo.get('nivel') == NIVEL_SI_NO:
        raise ValueError('el segundo paso tiene que ser el modelo de un nivel del tipo')
    umbral = (mod_si_no.get('regla') or {}).get('umbral')
    if umbral is None:
        raise ValueError('el modelo de sí o no no tiene umbral')
    if any(int(c['codigo']) == CODIGO_AUSENCIA for c in mod_tipo['categorias']):
        raise ValueError('el modelo del tipo se entrenó con «no borreguil»: el sí o no ya '
                         'lo decide él solo')
    lec_s, lec_t = mod_si_no.get('lectura'), mod_tipo.get('lectura')
    if lec_s and lec_t and lec_s != lec_t:
        raise ValueError('los dos modelos se entrenaron con lecturas distintas de la pila '
                         '(otros años, otra rejilla u otra versión): vuelve a entrenar uno')
    pred = ordenar_predictores(list(mod_si_no['predictores']) + list(mod_tipo['predictores']))
    modelo = DosPasos(mod_si_no['modelo'], mod_tipo['modelo'],
                      [pred.index(v) for v in mod_si_no['predictores']],
                      [pred.index(v) for v in mod_tipo['predictores']], len(pred))
    firma = hashlib.md5(f"{mod_si_no['firma']}|{mod_tipo['firma']}".encode()).hexdigest()
    return {
        'nivel': mod_tipo['nivel'], 'codigos': dict(mod_tipo['codigos']),
        'categorias': ([{'categoria': AUSENCIA, 'codigo': CODIGO_AUSENCIA}]
                       + [dict(c) for c in mod_tipo['categorias']]),
        'predictores': pred, 'modelo': modelo, 'firma': firma,
        'regla': {'criterio': mod_si_no['regla'].get('criterio'), 'umbral': float(umbral)},
        'lectura': lec_t or lec_s,
        'dos_pasos': {paso: {'nivel': m['nivel'], 'firma': m['firma'],
                             'cuando': m.get('cuando'), 'muestras': m.get('n_muestras'),
                             'predictores': list(m['predictores'])}
                      for paso, m in (('si_no', mod_si_no), ('tipo', mod_tipo))},
    }


# ============================================================
# Pureza dentro del píxel (por composición, solo con Sentinel-2)
# ------------------------------------------------------------
# Un píxel de 10 m puede ser borreguil entero o mezclarse con roca o con agua. Cuánto
# verde hay dentro se estima con su NDVI de inicio de verano: vale lo de la roca si no
# hay verde, lo de un borreguil puro si lo hay todo, y en proporción entre los dos.
# (Comparado con la ortofoto de 0,25 m en una zona de prueba, el verde así estimado se
# equivoca unos 6 puntos de media entre todos los píxeles y unos 12 dentro de los de
# borreguil.) La pureza se asigna SOLO a los píxeles que el modelo da por borreguil:
#   · puro        el verde llega al corte (por defecto, 75 %);
#   · mixto-agua  no llega y el píxel es agua o toca un píxel de agua;
#   · mixto-roca  no llega y no hay agua al lado.
# Sentinel-2 no ve una orilla que ocupe una parte pequeña del píxel: por eso el agua
# se deduce de los píxeles vecinos.
# ============================================================
VARIABLE_VERDE = 'ndvi_early'       # NDVI de inicio de verano: cuando más verde está
VARIABLE_AGUA = 'ndwi_mean'         # NDWI medio de jun–sep: > 0 es agua
NDWI_AGUA = 0.0
# NDVI de un píxel sin verde (roca, suelo), NDVI de uno todo verde (se propone el de
# las muestras de borreguil puro) y verde a partir del cual el píxel cuenta como puro.
PUREZA_DEFECTO = {'sin_verde': 0.08, 'todo_verde': 0.60, 'puro_desde': 0.75}
MIN_PUROS = 5                       # muestras puras para fiarse de su NDVI


def params_pureza(params=None):
    """Los tres valores de la pureza dentro del píxel, comprobados."""
    q = dict(PUREZA_DEFECTO)
    q.update({k: v for k, v in (params or {}).items() if k in PUREZA_DEFECTO})
    try:
        q = {k: float(v) for k, v in q.items()}
    except (TypeError, ValueError):
        raise ValueError('la pureza necesita tres números: sin_verde, todo_verde y '
                         'puro_desde') from None
    if not -0.5 <= q['sin_verde'] < q['todo_verde'] <= 1:
        raise ValueError('el NDVI de «todo verde» tiene que ser mayor que el de «sin verde»')
    if not 0 < q['puro_desde'] <= 1:
        raise ValueError('el corte de «puro» tiene que estar entre 0 y 1')
    return q


def verde_estimado(ndvi, sin_verde, todo_verde):
    """Fracción (de 0 a 1) del píxel cubierta de verde, estimada con su NDVI: 0 si vale
    `sin_verde` o menos, 1 si vale `todo_verde` o más, y en proporción entre los dos.
    Donde no hay dato, tampoco lo hay en el resultado. En float32, como se guarda."""
    import numpy as np
    v = np.asarray(ndvi, dtype=np.float32)
    return np.clip((v - np.float32(sin_verde)) / np.float32(todo_verde - sin_verde),
                   0, 1).astype(np.float32)


def pureza_de(verde, toca_agua, puro_desde):
    """El código de pureza de cada píxel de borreguil (los del nivel «pureza»: 1 puro,
    2 mixto-agua, 3 mixto-roca), con su fracción de verde y si es agua o toca agua."""
    import numpy as np
    cod = codigos('pureza')
    return np.where(np.asarray(verde, dtype=np.float32) >= np.float32(puro_desde),
                    cod['puro'],
                    np.where(toca_agua, cod['mixto-agua'], cod['mixto-roca']))


def todo_verde_de(ndvi_puros):
    """El NDVI de «todo verde» que proponen las muestras: la mediana del de las de
    borreguil puro. Devuelve (valor, de_las_muestras): con menos de MIN_PUROS muestras
    puras con dato no hay de qué fiarse y se propone el valor por defecto."""
    import statistics
    validos = [float(v) for v in ndvi_puros if v is not None and float(v) == float(v)]
    if len(validos) < MIN_PUROS:
        return PUREZA_DEFECTO['todo_verde'], False
    return statistics.median(validos), True


def comparar_pureza(muestras, params=None):
    """Cómo quedarían, por su verde estimado, las muestras de borreguil cuya pureza ha
    revisado una persona. Sirve para ajustar los valores con datos.

    muestras: [{'pureza': 'puro' | 'mixto-agua' | 'mixto-roca', 'ndvi': número}].
    En un punto suelto no se sabe si toca agua, así que se compara «puro» frente a
    «mixto». Devuelve {'n', 'coinciden', 'tabla'}: en `tabla`, para cada pureza
    revisada, cuántas salen puras y cuántas mixtas.
    """
    q = params_pureza(params)
    tabla = {c: {'puro': 0, 'mixto': 0} for c in NIVELES['pureza']}
    n = coinciden = 0
    for m in muestras:
        if m.get('pureza') not in tabla or m.get('ndvi') is None:
            continue
        sale = ('puro' if float(verde_estimado(m['ndvi'], q['sin_verde'], q['todo_verde']))
                >= q['puro_desde'] else 'mixto')
        tabla[m['pureza']][sale] += 1
        n += 1
        coinciden += (sale == 'puro') == (m['pureza'] == 'puro')
    return {'n': n, 'coinciden': coinciden, 'tabla': tabla}


def grupos_espaciales(tab, rows, celda=CELDA_GRADOS):
    """Zona de cada muestra de la tabla, para validar por zonas.

    Es la `cuenca_id` del punto si la tiene; si no, la celda de una rejilla de
    `celda` grados (~5 km). Mismo criterio que usa el detector. Devuelve
    (grupos, info): `grupos` es paralela a tab['filas'].
    """
    grupos, con_cuenca = [], 0
    for f in tab['filas']:
        r = rows[f['indice']] if 0 <= f['indice'] < len(rows) else {}
        cuenca = bp._txt(r.get('cuenca_id'))
        if cuenca and cuenca.lower() != 'nan':
            grupos.append(f'cuenca {cuenca}')
            con_cuenca += 1
        else:
            grupos.append(f"celda {round(f['lon'] / celda)}_{round(f['lat'] / celda)}")
    return grupos, {'n_zonas': len(set(grupos)), 'con_cuenca': con_cuenca,
                    'con_celda': len(grupos) - con_cuenca, 'celda_grados': celda}


def submuestrear(y, razon_maxima=RAZON_MAXIMA, semilla=42):
    """Posiciones que se quedan al recortar las categorías abundantes a
    `razon_maxima` veces la más escasa. Reproducible: misma semilla, mismas
    muestras. No inventa nada: solo deja fuera parte de las abundantes."""
    import numpy as np
    y = np.asarray(y)
    clases, cuentas = np.unique(y, return_counts=True)
    tope = max(int(cuentas.min()), int(math.floor(cuentas.min() * razon_maxima)))
    rng = np.random.RandomState(semilla)
    quedan = []
    for c in clases:
        pos = np.where(y == c)[0]
        if len(pos) > tope:
            pos = rng.choice(pos, tope, replace=False)
        quedan.extend(int(i) for i in pos)
    return sorted(quedan)


def _params_validos(params):
    p = dict(PARAMS_DEFECTO)
    p.update({k: v for k, v in (params or {}).items() if k in PARAMS_DEFECTO})
    if not (isinstance(p['n_estimators'], int) and p['n_estimators'] >= 10):
        raise ValueError('n_estimators: tiene que ser un entero de 10 o más')
    if not (p['max_features'] in ('sqrt', 'log2', None)
            or (isinstance(p['max_features'], int) and p['max_features'] >= 1)):
        raise ValueError("max_features: 'sqrt', 'log2', None (todas) o un entero ≥ 1")
    if not (p['max_depth'] is None or (isinstance(p['max_depth'], int) and p['max_depth'] >= 1)):
        raise ValueError('max_depth: None (sin límite) o un entero ≥ 1')
    if not (isinstance(p['min_samples_leaf'], int) and p['min_samples_leaf'] >= 1):
        raise ValueError('min_samples_leaf: tiene que ser un entero ≥ 1')
    if not isinstance(p['random_state'], int):
        raise ValueError('random_state: tiene que ser un entero')
    return p


def firma_tabla(tab):
    """Huella de la tabla (muestras, categorías, variables y valores): sirve para
    saber si un modelo se entrenó con la tabla que hay ahora en pantalla."""
    import hashlib
    h = hashlib.md5()
    h.update('|'.join(tab['predictores']).encode())
    for f, x in zip(tab['filas'], tab['X']):
        h.update(f"{f['ID']}|{f['codigo']}|".encode())
        h.update(','.join(f'{v:.6g}' for v in x).encode())
    return h.hexdigest()


def firma_muestras(res):
    """Huella de las muestras de un nivel (cuáles se usan y con qué categoría), sin las
    variables: dos modelos con la misma se entrenaron con las mismas muestras aunque
    usaran variables distintas, y se pueden comparar."""
    import hashlib
    h = hashlib.md5()
    for i, c in sorted((str(f['ID']), str(f['codigo'])) for f in res['tabla'] if f['usada']):
        h.update(f'{i}|{c}\n'.encode())
    return h.hexdigest()


def entrenar_y_validar(tab, grupos, params=None, estrategia='pesos',
                       razon_maxima=RAZON_MAXIMA, particiones=PARTICIONES):
    """Valida un Random Forest multiclase por zonas y entrena el modelo final.

    tab: salida de construir_tabla(); grupos: zona de cada muestra (paralela a
    tab['filas']); params: ver PARAMS_DEFECTO; estrategia: una de ESTRATEGIAS.

    Lanza ValueError, con un mensaje claro, si no se puede entrenar (menos de dos
    categorías, alguna por debajo del mínimo, tabla y zonas que no casan…).
    Devuelve un dict con la validación (matriz de confusión, métricas por categoría
    y globales, resultado de cada parte, predicción de cada muestra cuando le tocó
    ser validada), con el modelo final entrenado con todas las muestras y con la
    importancia de cada variable (`importancia`).

    La categoría de cada caso se decide con la regla de «¿borreguil o no?» (`regla`,
    ver aplicar_regla): de entrada, con el umbral de mayor F1 en la validación.
    El modelo guarda las probabilidades de la validación, así que la regla se puede
    cambiar después sin volver a entrenar.

    La importancia de cada variable:

      permutacion  cuánto baja el F1 medio de la validación si esa variable se
                   desordena entre las muestras con las que se valida (media de varias
                   repeticiones; `desviacion` es lo que cambia de una a otra). Mide lo
                   que el modelo PIERDE sin ella al predecir sitios que no ha visto.
      impureza     la importancia interna del Random Forest final: cuánto ha usado la
                   variable para separar las muestras al entrenar. Suma 1 entre todas.

    Lo mismo se mide desordenando a la vez todas las variables de cada familia (el
    mismo índice en distintos momentos: `familias`) y de cada grupo del catálogo (la
    misma época u origen: `grupos`): dos variables muy parecidas se cubren la una a la
    otra, y sueltas parecen pesar poco.
    """
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 confusion_matrix, f1_score,
                                 precision_recall_fscore_support)
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    if estrategia not in ESTRATEGIAS:
        raise ValueError(f'estrategia desconocida: {estrategia!r} (use {", ".join(ESTRATEGIAS)})')
    p = _params_validos(params)
    if not tab.get('listo'):
        raise ValueError('la tabla de entrenamiento no cumple los mínimos: '
                         + '; '.join(mensaje(e) for e in tab.get('errores', [])))
    X = np.asarray(tab['X'], dtype=float)
    y = np.asarray(tab['y'], dtype=int)
    if X.ndim != 2 or len(X) != len(y) or X.shape[1] != len(tab['predictores']):
        raise ValueError('la tabla está mal formada: filas, categorías y variables no casan')
    if not np.isfinite(X).all():
        raise ValueError('la tabla contiene valores no válidos (huecos o infinitos)')
    if len(grupos) != len(y):
        raise ValueError('hay que dar la zona de cada muestra, ni más ni menos')
    nombre = {c: n for n, c in tab['codigos'].items()}
    presentes = sorted(int(c) for c in np.unique(y))
    if len(presentes) < 2:
        raise ValueError('hacen falta al menos dos categorías con muestras')
    grupos = np.asarray([str(g) for g in grupos])
    semilla = p['random_state']
    con_pesos = estrategia in ('pesos', 'ambas')
    con_recorte = estrategia in ('submuestreo', 'ambas')

    def nuevo_modelo():
        return RandomForestClassifier(
            n_estimators=p['n_estimators'], max_features=p['max_features'],
            max_depth=p['max_depth'], min_samples_leaf=p['min_samples_leaf'],
            class_weight='balanced' if con_pesos else None,
            random_state=semilla, n_jobs=-1)

    def entrenado(pos):
        """Un modelo entrenado con esas muestras. Se entrena con todos los
        procesadores (con la semilla sale siempre el mismo bosque), pero para PREDECIR
        se deja con uno solo: con varios, scikit-learn suma los árboles según van
        acabando y el último decimal de la probabilidad cambia de una vez a otra, lo
        justo para que un empate entre dos categorías caiga a un lado o a otro. Con
        uno, el mismo modelo da siempre exactamente el mismo resultado."""
        m = nuevo_modelo().fit(X[pos], y[pos])
        m.n_jobs = 1
        return m

    def para_entrenar(pos):
        """Las posiciones con las que de verdad se entrena (tras el recorte)."""
        pos = np.asarray(pos)
        if not con_recorte:
            return pos
        return pos[submuestrear(y[pos], razon_maxima, semilla)]

    # ---- 1) validación
    n_zonas = len(set(grupos))
    minimo = int(min(np.sum(y == c) for c in presentes))
    avisos = []
    if n_zonas >= MIN_ZONAS:
        k = min(particiones, n_zonas)
        tipo = 'espacial'
        partes = list(StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=semilla)
                      .split(X, y, grupos))
    else:
        k = max(2, min(particiones, minimo))
        tipo = 'estratificada'
        avisos.append(('no_espacial', n_zonas))
        partes = list(StratifiedKFold(n_splits=k, shuffle=True, random_state=semilla)
                      .split(X, y))
    for c in presentes:
        if tipo == 'espacial' and len(set(grupos[y == c])) == 1:
            avisos.append(('categoria_en_una_zona', nombre[c]))

    # Lo que cada muestra recibe cuando le toca ser validada: no la categoría, sino
    # las probabilidades de las que sale (ver repartir). La categoría se decide
    # después, con la regla de «¿borreguil o no?» (aplicar_regla), y se puede cambiar
    # de umbral sin volver a entrenar.
    n = len(y)
    v_pborr, v_ptipo, v_pno = (np.zeros(n, np.float32) for _ in range(3))
    v_tipo = np.full(n, -1, dtype=int)
    parte_de = np.full(n, -1, dtype=int)
    pliegues = []
    # Para la importancia por permutación: qué se desordena cada vez. Primero cada
    # variable por separado; después, juntas, las de cada familia (el mismo índice en
    # distintos momentos) y las de cada grupo del catálogo (la misma época u origen).
    nv = X.shape[1]

    def agrupadas(clave_de):
        columnas = {}
        for j, variable in enumerate(tab['predictores']):
            columnas.setdefault(clave_de(variable), []).append(j)
        return columnas

    de_familia = agrupadas(familia)
    de_grupo = agrupadas(lambda variable: _GRUPO_DE.get(variable, ''))
    conjuntos, lugar = [], {}             # sin repetir: un conjunto de una es esa variable
    for cols in ([[j] for j in range(nv)] + list(de_familia.values())
                 + list(de_grupo.values())):
        if tuple(cols) not in lugar:
            lugar[tuple(cols)] = len(conjuntos)
            conjuntos.append(cols)
    reps = max(REPETICIONES_MIN, min(REPETICIONES_MAX,
                                     FILAS_PERMUTADAS // (len(conjuntos) * n)))
    azar = np.random.default_rng(semilla)
    d_pborr = np.zeros((reps, len(conjuntos), n), np.float32)
    d_tipo = np.full((reps, len(conjuntos), n), -1, dtype=np.int16)

    def lo_que_ve(m, filas):
        """(probabilidad de borreguil, tipo más probable) que da `m` a cada fila. Las
        filas se reparten en trozos que se predicen a la vez, pero cada trozo suma los
        árboles en su orden (el modelo predice con un solo procesador): sale lo mismo
        en cualquier ordenador."""
        from joblib import Parallel, delayed
        cortes = np.linspace(0, len(filas), max(1, len(filas) // TROZO_PERMUTACION) + 1,
                             dtype=int)

        def trozo(a, b):
            return repartir(m.predict_proba(filas[a:b]), m.classes_)[:2]

        partes = Parallel(n_jobs=-1, prefer='threads')(
            delayed(trozo)(a, b) for a, b in zip(cortes[:-1], cortes[1:]))
        return (np.concatenate([p_[0] for p_ in partes]),
                np.concatenate([p_[1] for p_ in partes]))

    def predecir_desordenado(m, valida):
        """Lo que da `m` a las muestras de `valida` con cada conjunto de variables
        desordenado entre ellas, `reps` veces."""
        Xv = X[valida].astype(np.float32)        # como ve los datos el bosque por dentro
        nc, nval = len(conjuntos), len(valida)
        de_una_vez = max(1, LOTE_PERMUTACION // (nc * nval))    # repeticiones por tanda
        for r0 in range(0, reps, de_una_vez):
            tanda = range(r0, min(reps, r0 + de_una_vez))
            filas = np.repeat(Xv[None, :, :], len(tanda) * nc, axis=0)
            for k in range(len(tanda)):
                for c, cols in enumerate(conjuntos):
                    # el mismo desorden para todo el conjunto: dentro de un grupo, las
                    # variables siguen yendo juntas
                    filas[k * nc + c][:, cols] = Xv[azar.permutation(nval)][:, cols]
            pb, tp = lo_que_ve(m, filas.reshape(-1, nv))
            pb, tp = pb.reshape(len(tanda), nc, nval), tp.reshape(len(tanda), nc, nval)
            for k, r in enumerate(tanda):
                d_pborr[r][:, valida] = pb[k]
                d_tipo[r][:, valida] = tp[k]

    for i, (entrena, valida) in enumerate(partes, 1):
        usadas = para_entrenar(entrena)
        m = entrenado(usadas)
        pb, tp, pt, pn = repartir(m.predict_proba(X[valida]), m.classes_)
        v_pborr[valida], v_tipo[valida], v_ptipo[valida] = pb, tp, pt
        # un modelo que no ha visto ningún «no borreguil» no le da ninguna probabilidad
        v_pno[valida] = 0.0 if pn is None else pn
        parte_de[valida] = i
        predecir_desordenado(m, valida)
        vistas = set(int(c) for c in m.classes_)
        pliegues.append({
            'parte': i, 'n_entrena': int(len(usadas)), 'n_valida': int(len(valida)),
            'zonas_valida': sorted(set(grupos[valida])),
            'categorias_sin_entrenar': [nombre[c] for c in presentes if c not in vistas],
        })

    # ---- 2) modelo final, con todas las muestras
    todas = para_entrenar(np.arange(n))
    modelo = entrenado(todas)

    def cuenta(pos):
        return {nombre[c]: int(np.sum(y[pos] == c)) for c in presentes}

    from datetime import datetime
    mod = {
        'nivel': tab['nivel'], 'codigos': dict(tab['codigos']),
        'categorias': [{'categoria': nombre[c], 'codigo': c} for c in presentes],
        'predictores': list(tab['predictores']), 'params': p,
        'balance': {'estrategia': estrategia,
                    'razon_maxima': razon_maxima if con_recorte else None,
                    'class_weight': 'balanced' if con_pesos else None,
                    'original': cuenta(np.arange(n)), 'final': cuenta(todas)},
        'validacion': {'tipo': tipo, 'particiones': k, 'n_zonas': n_zonas,
                       'avisos': avisos},
        'pliegues': pliegues,
        # dónde está cada muestra y qué es: para compararlas después con el mapa
        'muestras': [{'ID': f['ID'], 'lon': f['lon'], 'lat': f['lat'],
                      'categoria': f['categoria'], 'codigo': f['codigo']}
                     for f in tab['filas']],
        'modelo': modelo, 'n_muestras': int(n), 'n_entrenadas': int(len(todas)),
        'firma': firma_tabla(tab),
        'cuando': datetime.now().isoformat(timespec='seconds'),
        # Con esto se rehacen las cifras de la validación y el peso de las variables
        # para cualquier umbral, sin volver a entrenar (aplicar_regla).
        '_validacion': {
            'y': y, 'presentes': presentes, 'parte': parte_de,
            'p_borr': v_pborr, 'tipo': v_tipo, 'p_tipo': v_ptipo, 'p_no': v_pno,
            'filas': [{'ID': f['ID'], 'indice': f['indice'], 'zona': str(grupos[i]),
                       'parte': int(parte_de[i])} for i, f in enumerate(tab['filas'])]},
        '_permutacion': {
            'p_borr': d_pborr, 'tipo': d_tipo, 'repeticiones': int(reps),
            'lugar': lugar, 'de_familia': de_familia, 'de_grupo': de_grupo,
            'impureza': modelo.feature_importances_},
    }
    # ---- 3) la regla de «¿borreguil o no?», las cifras de la validación y el peso
    # de las variables. De entrada, el umbral de mayor F1 en la validación.
    return aplicar_regla(mod, CRITERIO_DEFECTO)


def mensaje_validacion(item, en=False, traduce=None):
    """Texto de un aviso de la validación."""
    k = item[0]
    if k == 'no_espacial':
        return (f'The samples fall in only {item[1]} zone(s), so the validation could not '
                'be done by zones: it mixes neighbouring points and will look better '
                'than it is. It should be repeated by zones when there are samples from '
                'more places.' if en else
                f'Las muestras caen en solo {item[1]} zona(s), así que no se ha podido '
                'validar por zonas: se mezclan puntos vecinos y el resultado saldrá mejor '
                'de lo que es. Habrá que repetirla por zonas cuando haya muestras de más '
                'sitios.')
    if k == 'categoria_en_una_zona':
        cat = traduce(item[1]) if traduce else item[1]
        return (f'All the samples of "{cat}" are in a single zone: when that zone is held '
                'out, the model has never seen that category, so it cannot get it right.'
                if en else
                f'Todas las muestras de «{cat}» están en una sola zona: cuando esa zona se '
                'aparta para validar, el modelo no ha visto nunca esa categoría y no puede '
                'acertarla.')
    return str(item)


# ============================================================
# ¿Mejoraría con más puntos? La curva con los puntos de la zona
# ------------------------------------------------------------
# La curva de Sierra Nevada (CURVA_PUNTOS_SN) es solo una referencia: en otra montaña
# pueden hacer falta más puntos, o menos (25-11-2026, la autora: «¿los puntos mínimos
# son solo para Sierra Nevada?»). Esta se calcula con los puntos y el modelo de cada
# persona: se repite su validación por zonas, con las MISMAS partes, pero el modelo
# aprende solo con una parte de los puntos de las otras zonas (un 25, un 50 y un 75 %),
# elegidos al azar y con la misma proporción de cada categoría. La parte apartada se
# predice siempre entera: lo único que cambia es con cuántos puntos aprende. El 100 %
# es la validación del propio modelo.
#
# De la mitad de los puntos a todos, el modelo aprende con el doble. Si en ese paso el
# acierto todavía sube (0,01 o más), es de esperar que con el doble de puntos que ahora
# suba algo parecido, o algo menos: merece la pena buscar más. Si ya no sube, más puntos
# parecidos a los que hay apenas cambiarán el mapa. (Del 75 al 100 % el paso es más
# corto y sube poco aunque la curva siga subiendo: con las 376 muestras de un ejemplo,
# +0,004 entre el 75 y el 100 % y +0,012 entre el 50 y el 100 %.)
#
# Las variables son las del modelo, como en su validación. El VIF no mira si los puntos
# son borreguil o no, así que elegirlas con todos no da ventaja a ninguna fracción.
# ============================================================
CURVA_FRACCIONES = (0.25, 0.5, 0.75)
CURVA_REPETICIONES = 3          # cada fracción, con 3 sorteos distintos de puntos
SIGUE_SUBIENDO = 0.01           # de la mitad a todos, lo que tiene que subir para «sube»


def _parte_al_azar(pos, y, fraccion, azar):
    """Una `fraccion` de las posiciones `pos`, al azar y con la misma proporción de
    cada categoría (al menos una de cada una de las que hay)."""
    import numpy as np
    elegidas = []
    for c in np.unique(y[pos]):
        de_c = pos[y[pos] == c]
        n = min(len(de_c), max(1, int(round(fraccion * len(de_c)))))
        elegidas.append(azar.choice(de_c, n, replace=False))
    return np.sort(np.concatenate(elegidas))


def entrenamientos_curva(mod, fracciones=CURVA_FRACCIONES,
                         repeticiones=CURVA_REPETICIONES):
    """Cuántos modelos entrena curva_propia (para decirlo antes de pulsar)."""
    partes = len(set(int(k) for k in mod['_validacion']['parte']))
    return len(fracciones) * repeticiones * partes


def curva_propia(tab, mod, fracciones=CURVA_FRACCIONES, repeticiones=CURVA_REPETICIONES,
                 al_avanzar=None):
    """La curva de aprendizaje con los puntos de la zona (ver arriba).

    tab: la tabla con la que se entrenó `mod` (salida de construir_tabla); mod: un
    modelo de entrenar_y_validar. al_avanzar(f): se llama tras cada entrenamiento con
    la fracción hecha (0–1), para una barra de progreso.

    Devuelve {'firma', 'corridas'}: una corrida por fracción y sorteo, más la del 100 %
    (la validación del modelo), con los puntos con los que aprende el modelo en cada
    parte (media) y lo que da a cada muestra cuando le toca ser validada. Las cifras
    salen de cifras_curva, con la regla de «¿borreguil o no?» que tenga el modelo.
    Lanza ValueError si el modelo no casa con la tabla."""
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    if '_validacion' not in mod:
        raise ValueError('el modelo es de una versión anterior de la app: vuelve a '
                         'entrenarlo')
    if mod['firma'] != firma_tabla(tab):
        raise ValueError('la tabla ha cambiado desde que se entrenó el modelo: vuelve a '
                         'entrenarlo')
    X = np.asarray(tab['X'], dtype=float)
    v = mod['_validacion']
    y, parte = np.asarray(v['y'], dtype=int), np.asarray(v['parte'], dtype=int)
    p, bal = mod['params'], mod['balance']
    semilla = p['random_state']
    con_pesos = bal.get('class_weight') == 'balanced'
    razon = bal.get('razon_maxima')             # None: sin recortar las abundantes
    partes = sorted(set(int(k) for k in parte))

    def para_entrenar(pos):
        """Como en entrenar_y_validar: tras el recorte, si lo hay."""
        return pos if razon is None else pos[submuestrear(y[pos], razon, semilla)]

    def cuantos(usadas):
        return (float(np.mean([len(u) for u in usadas])),
                float(np.mean([np.sum(y[u] != CODIGO_AUSENCIA) for u in usadas])))

    total, hechos, corridas = len(fracciones) * repeticiones * len(partes), 0, []
    for fraccion in fracciones:
        for r in range(repeticiones):
            # cada fracción y sorteo, su azar: sale lo mismo cada vez
            azar = np.random.default_rng([semilla, int(round(fraccion * 1000)), r])
            p_borr = np.zeros(len(y), np.float32)
            tipo = np.full(len(y), -1, dtype=int)
            usadas = []
            for k in partes:
                aprende = para_entrenar(_parte_al_azar(np.where(parte != k)[0], y,
                                                       fraccion, azar))
                m = RandomForestClassifier(
                    n_estimators=p['n_estimators'], max_features=p['max_features'],
                    max_depth=p['max_depth'], min_samples_leaf=p['min_samples_leaf'],
                    class_weight='balanced' if con_pesos else None,
                    random_state=semilla, n_jobs=-1).fit(X[aprende], y[aprende])
                m.n_jobs = 1                    # predecir con uno: siempre lo mismo
                valida = np.where(parte == k)[0]
                p_borr[valida], tipo[valida] = repartir(m.predict_proba(X[valida]),
                                                        m.classes_)[:2]
                usadas.append(aprende)
                hechos += 1
                if al_avanzar:
                    al_avanzar(hechos / total)
            n, nb = cuantos(usadas)
            corridas.append({'fraccion': float(fraccion), 'sorteo': r, 'n_puntos': n,
                             'n_borreguiles': nb, 'p_borr': p_borr, 'tipo': tipo})
    n, nb = cuantos([para_entrenar(np.where(parte != k)[0]) for k in partes])
    corridas.append({'fraccion': 1.0, 'sorteo': 0, 'n_puntos': n, 'n_borreguiles': nb,
                     'p_borr': np.asarray(v['p_borr'], np.float32),
                     'tipo': np.asarray(v['tipo'], dtype=int)})
    return {'firma': mod['firma'], 'corridas': corridas}


def cifras_curva(curva, mod):
    """Las cifras de la curva de curva_propia, con la regla de «¿borreguil o no?» que
    tenga ahora el modelo (con el F1 o el TSS, cada sorteo elige su umbral, igual que
    el modelo con todos los puntos; con un umbral fijo, ese).

    Devuelve {'medida': 'auc' o 'f1', 'filas': [...], 'mitad', 'ganancia', 'lectura':
    'sube' o 'plana'}. Cada fila es una fracción: los puntos con los que aprende el
    modelo y los borreguiles entre ellos (media de las partes), el AUC de «¿borreguil o
    no?» en zonas que no ha visto (None si el modelo no tiene «no borreguil») y el F1
    medio entre categorías; de cada cifra, la media de los sorteos y la menor y la
    mayor. La medida es el AUC si lo hay (no depende del umbral; es la de la curva de
    Sierra Nevada) y si no el F1. 'mitad' es la fila de la mitad de los puntos (la más
    cercana) y 'ganancia', lo que sube la medida de ella a la de todos: lo que da
    doblar los puntos."""
    import numpy as np
    from sklearn.metrics import roc_auc_score
    v, regla = mod['_validacion'], mod.get('regla') or {}
    y = np.asarray(v['y'], dtype=int)
    presentes = np.asarray(v['presentes'])
    con_no = CODIGO_AUSENCIA in set(int(c) for c in presentes)
    es_borr = y != CODIGO_AUSENCIA
    hay_auc = con_no and 0 < es_borr.sum() < len(y)
    por_fraccion = {}
    for c in curva['corridas']:
        if con_no:
            if regla.get('criterio') == 'manual':
                u = regla['umbral']
            else:
                u = elegir_umbral(barrido_umbral(c['p_borr'], es_borr),
                                  regla.get('criterio') or CRITERIO_DEFECTO)['umbral']
            pred = np.where(c['p_borr'] >= np.float32(u), c['tipo'], CODIGO_AUSENCIA)
        else:
            pred = c['tipo']
        por_fraccion.setdefault(c['fraccion'], []).append({
            'n_puntos': c['n_puntos'], 'n_borreguiles': c['n_borreguiles'],
            'auc': float(roc_auc_score(es_borr, c['p_borr'])) if hay_auc else None,
            'f1': float(_macro_f1(y, pred, presentes))})
    filas = []
    for fraccion in sorted(por_fraccion):
        rs = por_fraccion[fraccion]
        fila = {'fraccion': fraccion, 'sorteos': len(rs),
                'n_puntos': float(np.mean([r['n_puntos'] for r in rs])),
                'n_borreguiles': float(np.mean([r['n_borreguiles'] for r in rs]))}
        for k in ('auc', 'f1'):
            vals = [r[k] for r in rs if r[k] is not None]
            fila[k], fila[k + '_min'], fila[k + '_max'] = (
                (float(np.mean(vals)), float(min(vals)), float(max(vals))) if vals
                else (None, None, None))
        filas.append(fila)
    medida = 'auc' if hay_auc else 'f1'
    mitad = (min(filas[:-1], key=lambda f: abs(f['fraccion'] - filas[-1]['fraccion'] / 2))
             if len(filas) > 1 else filas[-1])
    ganancia = filas[-1][medida] - mitad[medida]
    return {'medida': medida, 'filas': filas, 'mitad': mitad, 'ganancia': float(ganancia),
            'lectura': 'sube' if ganancia >= SIGUE_SUBIENDO else 'plana'}


# ============================================================
# Variables predictoras disponibles como MAPA continuo
# ------------------------------------------------------------
# Para predecir sobre el territorio, cada variable tiene que existir en todos los
# píxeles, no solo en los puntos. Hoy lo cumplen las que la app calcula en Earth
# Engine (gee_backend.py): los compuestos de Sentinel-2 y la topografía. El nombre
# es el de la banda en esa pila, que es también el de la columna en los puntos.
# ============================================================
G_INICIO = 'Sentinel-2 · inicio de verano (15 jun – 24 jul)'
G_FIN = 'Sentinel-2 · fin de verano (1 ago – 29 sep)'
G_CAMBIO = 'Sentinel-2 · cambio estacional'
G_PERIODO = 'Sentinel-2 · estadísticas de junio a septiembre (1 jun – 29 sep)'
G_TOPO = 'Topografía (Copernicus DEM, 30 m)'
GRUPOS = (G_INICIO, G_FIN, G_CAMBIO, G_PERIODO, G_TOPO)
GRUPOS_EN = {
    G_INICIO: 'Sentinel-2 · early summer (15 Jun – 24 Jul)',
    G_FIN: 'Sentinel-2 · late summer (1 Aug – 29 Sep)',
    G_CAMBIO: 'Sentinel-2 · seasonal change',
    G_PERIODO: 'Sentinel-2 · June to September statistics (1 Jun – 29 Sep)',
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
                 ('max', 'máximo', 'maximum'), ('sd', 'variabilidad (desviación típica)', 'variability (standard deviation)'))

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
    ('GNDVI (2 variables)',
     'es el NDWI con el signo cambiado; tener las dos sería repetir la misma variable',
     'GNDVI (2 variables)',
     'it is NDWI with the sign flipped: keeping both would repeat the same variable'),
)


_GRUPO_DE = {p[0]: p[1] for p in PREDICTORES}        # variable → su grupo


# ============================================================
# Los nombres de las variables DEL MAPA, con sus fechas exactas
# ------------------------------------------------------------
# Earth Engine deja fuera el último día de un filtro de fechas (ee.Filter.date), así
# que las ventanas del mapa son 15 jun – 24 jul, 1 ago – 29 sep y 1 jun – 29 sep (la
# autora insiste: son las fechas de verdad). Los puntos del detector usan otras ventanas
# (15 jun – 25 jul, 1 ago – 30 sep): sus nombres están en point_profile.FEATURE_LABELS.
# Estos son los de la pantalla «3 · Hacer el mapa», como en la demo:
#   · corta: la de las fichas de «1 · Datos» (el grupo ya lleva las fechas);
#   · tabla: la de «Qué mide cada variable» (la columna «Cuándo» lleva las fechas);
#   · larga: la de todo lo demás, con sus fechas.
# ============================================================
F_INICIO = ('15 jun – 24 jul', '15 Jun – 24 Jul')
F_FIN = ('1 ago – 29 sep', '1 Aug – 29 Sep')
F_PERIODO = ('1 jun – 29 sep', '1 Jun – 29 Sep')
_INDICE_NOMBRE = {'ndvi': ('NDVI', 'NDVI'), 'ndwi': ('NDWI', 'NDWI'),
                  'clre': ('Clorofila', 'Chlorophyll'), 'ndmi': ('NDMI', 'NDMI'),
                  'evi': ('EVI', 'EVI'), 'nbr': ('NBR', 'NBR'), 'wavi': ('WAVI', 'WAVI'),
                  'albedo': ('Albedo', 'Albedo')}
_INDICE_QUE = {'ndvi': ('vigor de la vegetación', 'vegetation vigour'),
               'ndwi': ('agua en superficie', 'surface water'),
               'clre': ('clorofila (borde del rojo)', 'chlorophyll (red edge)'),
               'ndmi': ('humedad de la vegetación', 'vegetation moisture'),
               'evi': ('vigor corregido', 'corrected vigour'),
               'nbr': ('infrarrojo de onda corta', 'short-wave infrared')}
_PERIODO_QUE = {'wavi': ('vegetación sobre agua', 'vegetation over water'),
                'albedo': ('brillo de la superficie', 'surface brightness'),
                'ndmi': ('humedad', 'moisture'), 'ndwi': ('agua', 'water')}
_ESTAD_NOMBRE = {s: (es, en) for s, es, en in _ESTADISTICOS}
_ESTAD_QUE = {'mean': ('media', 'mean'), 'min': ('el valor más bajo', 'the lowest value'),
              'max': ('el valor más alto', 'the highest value'),
              'sd': ('cuánto varía', 'how much it varies')}
# relieve: (corta, tabla, larga, qué mide), cada una (es, en)
_TOPO_NOMBRE = {
    'elev_dem_m': (('Altitud', 'Elevation'), ('Altitud', 'Elevation'),
                   ('Altitud (m)', 'Elevation (m)'),
                   ('altitud del terreno', 'terrain elevation')),
    'slope_deg': (('Pendiente', 'Slope'), ('Pendiente', 'Slope'), ('Pendiente (°)', 'Slope (°)'),
                  ('inclinación del terreno', 'terrain steepness')),
    'aspect_north': (('Orientación norte', 'Aspect north'),
                     ('Orientación: componente norte', 'Aspect: north component'),
                     ('Orientación: componente norte', 'Aspect: north component'),
                     ('cuánto mira al norte', 'how much it faces north')),
    'aspect_east': (('Orientación este', 'Aspect east'),
                    ('Orientación: componente este', 'Aspect: east component'),
                    ('Orientación: componente este', 'Aspect: east component'),
                    ('cuánto mira al este', 'how much it faces east')),
    'curvature': (('Curvatura', 'Curvature'), ('Curvatura del terreno', 'Terrain curvature'),
                  ('Curvatura del terreno', 'Terrain curvature'),
                  ('si el terreno es una hondonada o una loma',
                   'whether the ground is a hollow or a ridge')),
}
_CAIDA = {'corta': ('Caída del NDVI', 'NDVI drop'),
          'larga': (f'Caída del NDVI, del inicio de verano ({F_INICIO[0]}) al fin de verano '
                    f'({F_FIN[0]})',
                    f'NDVI drop, from early summer ({F_INICIO[1]}) to late summer '
                    f'({F_FIN[1]})'),
          'que': (f'cuánto baja el verdor del inicio de verano ({F_INICIO[0]}) al fin de '
                  f'verano ({F_FIN[0]})',
                  f'how much the greenness drops from early summer ({F_INICIO[1]}) to late '
                  f'summer ({F_FIN[1]})')}
# Qué se quita de las imágenes en cada grupo (gee_backend: _season_composite con la SCL;
# _period_new_stats con s2cloudless y la máscara de nieve)
_FILTRO_SCL = ('Escenas con menos del 40 % de nubes; en cada píxel se quitan nubes, sombras '
               'de nube, cirros y píxeles oscuros o saturados (clasificación SCL de '
               'Sentinel-2). La nieve no se quita.',
               'Scenes with less than 40 % cloud; in each pixel, clouds, cloud shadows, '
               'cirrus and dark or saturated pixels are removed (Sentinel-2 SCL '
               'classification). Snow is not removed.')
FILTRO_GRUPO = {
    G_INICIO: _FILTRO_SCL, G_FIN: _FILTRO_SCL,
    G_CAMBIO: ('Como el inicio y el fin de verano, de los que sale.',
               'Like early and late summer, which it comes from.'),
    G_PERIODO: ('Se quitan las nubes (s2cloudless: probabilidad de nube mayor del 30 %) y '
                'la nieve (NDSI de 0,4 o más, o nieve en la SCL). Las sombras de nube no se '
                'quitan.',
                'Clouds (s2cloudless: cloud probability above 30 %) and snow (NDSI of 0.4 '
                'or more, or snow in the SCL) are removed. Cloud shadows are not removed.'),
    G_TOPO: ('No hace falta: es el relieve.', 'Not needed: it is the relief.'),
}


def etiqueta(nombre, en=False, forma='larga'):
    """El nombre de una variable del mapa, con las fechas del mapa (ver arriba).
    `forma`: 'larga' (con sus fechas), 'tabla' o 'corta'."""
    i = 1 if en else 0
    if nombre in _TOPO_NOMBRE:
        return _TOPO_NOMBRE[nombre][{'corta': 0, 'tabla': 1}.get(forma, 2)][i]
    if nombre == 'ndvi_drop':
        return _CAIDA['corta' if forma != 'larga' else 'larga'][i]
    k, _, s = nombre.rpartition('_')
    if k not in _INDICE_NOMBRE:
        return nombre
    idx = _INDICE_NOMBRE[k][i]
    if s in ('early', 'late'):
        if forma == 'corta':
            return idx
        if s == 'early':
            return (f'{idx} early summer ({F_INICIO[1]})' if en
                    else f'{idx} inicio de verano ({F_INICIO[0]})')
        return (f'{idx} late summer ({F_FIN[1]})' if en
                else f'{idx} fin de verano ({F_FIN[0]})')
    if s in _ESTAD_NOMBRE:
        base = f'{idx} {_ESTAD_NOMBRE[s][i]}'
        return base if forma != 'larga' else f'{base} ({F_PERIODO[i]})'
    return nombre


def que_mide(nombre, en=False):
    """Qué mide una variable del mapa, como en la tabla «Qué mide cada variable»."""
    i = 1 if en else 0
    if nombre in _TOPO_NOMBRE:
        return _TOPO_NOMBRE[nombre][3][i]
    if nombre == 'ndvi_drop':
        return _CAIDA['que'][i]
    k, _, s = nombre.rpartition('_')
    if s in ('early', 'late') and k in _INDICE_QUE:
        return _INDICE_QUE[k][i]
    if s in _ESTAD_QUE and k in _PERIODO_QUE:
        return f'{_PERIODO_QUE[k][i]}: {_ESTAD_QUE[s][i]}'
    return descripcion(nombre, en)


def cuando_grupo(grupo, anios='', en=False):
    """Las fechas de un grupo de variables del mapa (la columna «Cuándo»). `anios`:
    los años del panel de Analizar, ya escritos («2017–2025»)."""
    a = f', {anios}' if anios else ''
    i = 1 if en else 0
    if grupo == G_INICIO:
        return (f'Early summer: {F_INICIO[1]}{a}' if en
                else f'Inicio de verano: {F_INICIO[0]}{a}')
    if grupo == G_FIN:
        return (f'Late summer: {F_FIN[1]}{a}' if en else f'Fin de verano: {F_FIN[0]}{a}')
    if grupo == G_CAMBIO:
        return (f'Early ({F_INICIO[1]}) minus late ({F_FIN[1]})' if en
                else f'Inicio ({F_INICIO[0]}) menos fin ({F_FIN[0]})')
    if grupo == G_PERIODO:
        return (f'{F_PERIODO[1]} of each year{a}' if en
                else f'{F_PERIODO[0]} de cada año{a}')
    return ('No date: the relief (Copernicus DEM)', 'Sin fecha: el relieve (Copernicus DEM)')[1 - i]


def filtro_grupo(grupo, en=False):
    """Qué se quita de las imágenes en un grupo de variables del mapa."""
    return FILTRO_GRUPO.get(grupo, ('', ''))[1 if en else 0]


def grupo_de(nombre):
    """El grupo (época u origen) de una variable del mapa."""
    return _GRUPO_DE.get(nombre)

# Familias: variables que salen del mismo índice (el NDVI de inicio de verano, el de
# fin y su caída) o de la topografía. Se parecen mucho entre sí, así que su peso se
# mira también juntas. (clave: nombre en español, nombre en inglés)
FAMILIAS = {
    'ndvi': ('NDVI · vigor de la vegetación', 'NDVI · vegetation vigour'),
    'ndwi': ('NDWI · agua en superficie', 'NDWI · surface water'),
    'clre': ('Clorofila · borde del rojo', 'Chlorophyll · red edge'),
    'ndmi': ('NDMI · humedad de la vegetación', 'NDMI · vegetation moisture'),
    'evi': ('EVI · vigor corregido', 'EVI · corrected vigour'),
    'nbr': ('NBR · infrarrojo de onda corta', 'NBR · short-wave infrared'),
    'wavi': ('WAVI · vegetación sobre agua', 'WAVI · vegetation over water'),
    'albedo': ('Albedo · brillo de la superficie', 'Albedo · surface brightness'),
    'topo': ('Topografía', 'Topography'),
}


def familia(nombre):
    """La familia de un predictor: el índice del que sale, o la topografía."""
    return 'topo' if _GRUPO_DE.get(nombre) == G_TOPO else nombre.split('_')[0]


# Grupos de EBVs (variables esenciales de biodiversidad, Essential Biodiversity Variables,
# de GEO BON): las variables juntas según lo que miden de la vegetación o del agua. La
# autora confirmó la agrupación el 26-11-2026. El relieve va aparte: no es una EBV, es el
# contexto físico. El peso de un grupo es la suma del de sus variables.
EBV = {
    'productividad': ('Productividad y vigor (NDVI, EVI, WAVI)',
                      'Productivity and vigour (NDVI, EVI, WAVI)'),
    'agua_vegetacion': ('Contenido de agua de la vegetación (NDMI)',
                        'Vegetation water content (NDMI)'),
    'clorofila': ('Clorofila (borde del rojo)', 'Chlorophyll (red edge)'),
    'fenologia': ('Fenología (caída del NDVI)', 'Phenology (NDVI drop)'),
    'agua': ('Agua en superficie (NDWI)', 'Surface water (NDWI)'),
    'suelo': ('Cubierta y brillo del suelo (albedo)', 'Ground cover and brightness (albedo)'),
    'perturbaciones': ('Perturbaciones (NBR)', 'Disturbance (NBR)'),
    'relieve': ('Relieve (no es una EBV)', 'Terrain (not an EBV)'),
}
_EBV_DE_FAMILIA = {'ndvi': 'productividad', 'evi': 'productividad', 'wavi': 'productividad',
                   'ndmi': 'agua_vegetacion', 'clre': 'clorofila', 'ndwi': 'agua',
                   'albedo': 'suelo', 'nbr': 'perturbaciones', 'topo': 'relieve'}


def grupo_ebv(nombre):
    """El grupo de EBVs de un predictor (la caída del NDVI, fenología)."""
    if nombre == 'ndvi_drop':
        return 'fenologia'
    return _EBV_DE_FAMILIA.get(familia(nombre), 'relieve')


def nombre_ebv(clave, en=False):
    es, eng = EBV.get(clave, (clave, clave))
    return eng if en else es


def nombre_familia(clave, en=False):
    es, eng = FAMILIAS.get(clave, (clave, clave))
    return eng if en else es


def nombre_grupo(grupo, en=False):
    return GRUPOS_EN.get(grupo, grupo) if en else grupo


def descripcion(nombre, en=False):
    """Qué mide un predictor, en español o en inglés ('' si no existe)."""
    for n, _g, es, eng in PREDICTORES:
        if n == nombre:
            return eng if en else es
    return ''


def importancia_csv(mod, en=False):
    """La importancia de las variables de un modelo, como texto CSV: una fila por
    variable, de la que más pesa a la que menos."""
    import csv
    import io
    campos = ['orden', 'variable', 'que_mide', 'familia', 'grupo',
              'importancia_permutacion', 'desviacion', 'importancia_interna',
              'orden_interno']
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=campos, lineterminator='\n')
    w.writeheader()
    for v in mod['importancia']['variables']:
        w.writerow({
            'orden': v['orden'], 'variable': v['variable'],
            'que_mide': descripcion(v['variable'], en),
            'familia': nombre_familia(v['familia'], en),
            'grupo': nombre_grupo(v['grupo'], en),
            'importancia_permutacion': f"{v['permutacion']:.6f}",
            'desviacion': f"{v['desviacion']:.6f}",
            'importancia_interna': f"{v['impureza']:.6f}",
            'orden_interno': v['orden_impureza']})
    return buf.getvalue()


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
