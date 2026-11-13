"""
mapeo_tif.py — Los GeoTIFF del mapa, sin GDAL.

Lee y escribe los rásteres del paso «4 · Mapa» solo con numpy y tifffile, que es Python
puro. Antes se hacía con rasterio (GDAL), que trae bibliotecas compiladas; en los
equipos con el «Control inteligente de aplicaciones» de Windows 11 activado, Windows
bloquea a ratos algunas de ellas y entonces el mapa no se podía ni hacer ni ver. Lo que
sale de aquí son GeoTIFF corrientes: QGIS y GDAL los abren con su georreferencia, su
sistema de coordenadas, su valor sin dato, el nombre de cada banda, sus etiquetas, su
paleta de colores y sus versiones reducidas (comprobado con GDAL 3.12).

Solo cubre lo que el mapa necesita: rejillas en una proyección con código EPSG, con el
norte arriba y píxeles cuadrados (las de mapeo_raster.rejilla); datos uint8 o float32;
teselas de 256 píxeles comprimidas con deflate.
"""
import math
from pathlib import Path

TESELA = 256
# Por encima de este tamaño, lo que se va escribiendo no se guarda en memoria sino en
# un fichero temporal junto al GeoTIFF.
EN_MEMORIA_HASTA = 64 * 1024 * 1024
TIPOS = ('uint8', 'float32')

# Etiquetas TIFF de un GeoTIFF y las dos propias de GDAL.
_ESCALA, _ANCLAJE, _GEOCLAVES = 33550, 33922, 34735
_GDAL_ETIQUETAS, _GDAL_SIN_DATO = 42112, 42113
_CLAVE_PROYECTADO, _CLAVE_GEOGRAFICO = 3072, 2048
# Valores de dos etiquetas TIFF: los colores van en una paleta; los valores de cada
# píxel van juntos (y no una banda detrás de otra).
_CON_PALETA, _BANDAS_JUNTAS = 3, 1


# ============================================================
# Etiquetas, nombres de banda y sin dato, como los guarda GDAL
# ============================================================
def _xml_etiquetas(etiquetas, nombres):
    """El texto que GDAL espera en su etiqueta de metadatos: las etiquetas del fichero
    y el nombre de cada banda. None si no hay nada que guardar."""
    from xml.sax.saxutils import escape, quoteattr
    lineas = [f'  <Item name={quoteattr(str(k))}>{escape(str(v))}</Item>'
              for k, v in etiquetas.items()]
    lineas += [f'  <Item name="DESCRIPTION" sample="{i}" role="description">{escape(n)}</Item>'
               for i, n in enumerate(nombres) if n]
    if not lineas:
        return None
    return ('<GDALMetadata>\n' + '\n'.join(lineas) + '\n</GDALMetadata>\n').encode('utf-8')


def _texto(valor):
    if isinstance(valor, bytes):
        valor = valor.decode('utf-8', 'replace')
    return str(valor).rstrip('\x00').strip()


def _leer_xml(valor, bandas):
    """(etiquetas, nombres de banda) de la etiqueta de metadatos de GDAL."""
    import xml.etree.ElementTree as ET
    etiquetas, nombres = {}, [''] * bandas
    try:
        raiz = ET.fromstring(_texto(valor))
    except ET.ParseError:
        return etiquetas, tuple(nombres)
    for it in raiz.iter('Item'):
        nombre, banda = it.get('name'), it.get('sample')
        if banda is None:
            etiquetas[nombre] = it.text or ''
        elif nombre == 'DESCRIPTION' and banda.isdigit() and int(banda) < bandas:
            nombres[int(banda)] = it.text or ''
    return etiquetas, tuple(nombres)


def _texto_sin_dato(sin_dato):
    v = float(sin_dato)
    if v != v:
        return 'nan'
    return str(int(v)) if v == int(v) else repr(v)


def _geoclaves(valor):
    """{clave: valor} del directorio de claves de un GeoTIFF (las que van en él)."""
    v = [int(x) for x in valor]
    return {v[k]: v[k + 3] for k in range(4, min(len(v), 4 + 4 * v[3]) - 3, 4) if v[k + 1] == 0}


# ============================================================
# Versiones reducidas
# ============================================================
def factores_reducidos(ancho, alto):
    """A cuánto conviene reducir un ráster de ese tamaño para dibujarlo de lejos: a la
    mitad, a la cuarta parte… mientras el lado mayor no baje de una tesela."""
    factores, k = [], 2
    while max(ancho, alto) / k >= TESELA:
        factores.append(k)
        k *= 2
    return factores


def _en_celdas(m, k, relleno):
    """La matriz (alto, ancho) repartida en celdas de k × k: (filas, k, columnas, k).
    Lo que falta para completar las celdas del borde se rellena con `relleno`."""
    import numpy as np
    alto, ancho = m.shape
    filas, cols = -(-alto // k), -(-ancho // k)
    if (filas * k, cols * k) != (alto, ancho):
        lleno = np.full((filas * k, cols * k), relleno, dtype=m.dtype)
        lleno[:alto, :ancho] = m
        m = lleno
    return m.reshape(filas, k, cols, k)


def reducir(m, k, categorica, sin_dato=None):
    """Una matriz (alto, ancho) reducida k veces. Cada celda reducida lleva, de los k × k
    píxeles que resume, la categoría más frecuente (`categorica`) o la media; los
    píxeles sin dato no cuentan, y una celda sin ninguno con dato queda sin dato. Si
    empatan dos categorías, se queda con la de código menor."""
    import numpy as np
    if not categorica:
        celdas = _en_celdas(np.asarray(m, dtype='float32'), k, np.nan)
        con = np.isfinite(celdas)
        n = con.sum(axis=(1, 3))
        suma = np.where(con, celdas, 0).sum(axis=(1, 3), dtype='float64')
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.where(n > 0, suma / n, np.nan).astype('float32')
    m = np.asarray(m)
    vacio = sin_dato if sin_dato is not None else 0
    celdas = _en_celdas(m, k, vacio)
    mejor = np.full(celdas.shape[::2], vacio, dtype=m.dtype)
    veces = np.zeros(celdas.shape[::2], dtype='int32')
    for c in np.unique(m):
        if sin_dato is not None and c == sin_dato:
            continue
        n = (celdas == c).sum(axis=(1, 3), dtype='int32')
        gana = n > veces
        mejor[gana] = c
        veces[gana] = n[gana]
    return mejor


# ============================================================
# Escribir
# ============================================================
def _opciones(bandas, paleta):
    """Cómo se guarda cada página: teselas comprimidas y, si hay paleta, con ella."""
    import numpy as np
    op = dict(tile=(TESELA, TESELA), compression='zlib', metadata=None)
    if paleta:
        tabla = np.zeros((3, 256), dtype='uint16')
        for codigo, rgb in paleta.items():
            tabla[:, int(codigo)] = [int(v) * 257 for v in rgb[:3]]
        op.update(photometric='palette', colormap=tabla)
    else:
        op.update(photometric='minisblack')
        if bandas > 1:
            op.update(planarconfig='contig')
    return op


class Escritor:
    """Un GeoTIFF de la rejilla `rej` (la de mapeo_raster.rejilla) que se va llenando por
    ventanas y se escribe entero al cerrarlo.

    Mientras está abierto, los datos están en memoria o, si son muchos, en un fichero
    temporal al lado (`<nombre>.parte`). Lo que no se llegue a escribir queda sin dato.
    tipo: 'uint8' o 'float32'. sin_dato: el valor que significa «sin dato» (None = no
    hay). Con varias bandas, cada píxel guarda sus valores juntos.
    """

    def __init__(self, ruta, rej, bandas=1, tipo='float32', sin_dato=None):
        import numpy as np
        if str(tipo) not in TIPOS:
            raise ValueError(f'solo se escriben rásteres {" o ".join(TIPOS)}, no {tipo}')
        self.ruta, self.rej = Path(ruta), dict(rej)
        self.bandas, self.tipo, self.sin_dato = int(bandas), str(tipo), sin_dato
        self._etiquetas, self._nombres, self._paleta = {}, [''] * self.bandas, None
        forma = (self.rej['alto'], self.rej['ancho']) + ((self.bandas,) if self.bandas > 1 else ())
        self._bytes = int(np.prod(forma)) * np.dtype(self.tipo).itemsize
        self._parte = None
        relleno = 0 if sin_dato is None else sin_dato
        if self._bytes > EN_MEMORIA_HASTA:
            self._parte = self.ruta.with_name(self.ruta.name + '.parte')
            self._datos = np.memmap(self._parte, dtype=self.tipo, mode='w+', shape=forma)
            self._datos[...] = relleno
        else:
            self._datos = np.full(forma, relleno, dtype=self.tipo)

    def escribir(self, matriz, banda=None, ventana=None):
        """Pone una matriz en su sitio. ventana: (columna, fila, ancho, alto); None = todo
        el ráster. Con `banda` (1, 2…), la matriz es (alto, ancho); sin ella, trae todas
        las bandas: (bandas, alto, ancho)."""
        import numpy as np
        col, fila, ancho, alto = ventana or (0, 0, self.rej['ancho'], self.rej['alto'])
        m = np.asarray(matriz)
        trozo = (slice(fila, fila + alto), slice(col, col + ancho))
        if banda is None and m.ndim == 3:
            if m.shape != (self.bandas, alto, ancho):
                raise ValueError(f'se esperaba una matriz {(self.bandas, alto, ancho)} y '
                                 f'ha llegado {m.shape}')
            m = m[0] if self.bandas == 1 else np.moveaxis(m, 0, -1)
        else:
            if m.shape != (alto, ancho):
                raise ValueError(f'se esperaba una matriz {(alto, ancho)} y ha llegado {m.shape}')
            if self.bandas > 1:
                if banda is None or not 1 <= int(banda) <= self.bandas:
                    raise ValueError('hay que decir en qué banda va esa matriz')
                trozo += (int(banda) - 1,)
        self._datos[trozo] = m

    def matriz(self, banda=None):
        """Lo escrito hasta ahora, para corregirlo antes de cerrar: lo que se cambie en
        ella queda escrito. Con varias bandas, la de `banda` (1, 2…)."""
        if self.bandas == 1:
            return self._datos
        if banda is None or not 1 <= int(banda) <= self.bandas:
            raise ValueError('hay que decir qué banda')
        return self._datos[..., int(banda) - 1]

    def paleta(self, colores):
        """El color de cada código, {codigo: (r, g, b)}: viaja dentro del fichero. Solo
        en rásteres uint8 de una banda."""
        if self.tipo != 'uint8' or self.bandas != 1:
            raise ValueError('la paleta es para rásteres uint8 de una banda')
        self._paleta = {int(c): tuple(int(v) for v in rgb[:3]) for c, rgb in colores.items()}

    def etiquetas(self, **pares):
        """Etiquetas del fichero (texto), que se añaden a las que ya tuviera."""
        self._etiquetas.update({str(k): str(v) for k, v in pares.items()})

    def nombre_banda(self, banda, texto):
        self._nombres[int(banda) - 1] = str(texto)

    def _soltar(self):
        """Deja de usar los datos y borra el fichero temporal, si lo había."""
        datos, self._datos = self._datos, None
        mapa = getattr(datos, '_mmap', None)
        del datos
        if mapa is not None:
            try:
                mapa.close()
            except (BufferError, ValueError):
                pass
        if self._parte is not None:
            try:
                self._parte.unlink(missing_ok=True)
            except OSError:
                pass

    def descartar(self):
        """Cierra sin escribir nada (cuando el mapa no se ha podido terminar)."""
        if self._datos is not None:
            self._soltar()

    def cerrar(self):
        """Escribe el GeoTIFF."""
        import tifffile
        if self._datos is None:
            return
        rej = self.rej
        e = float(rej['escala'])
        extra = [(_ESCALA, 'd', 3, (e, e, 0.0), True),
                 (_ANCLAJE, 'd', 6, (0.0, 0.0, 0.0, float(rej['x0']), float(rej['y0']), 0.0), True),
                 # versión de las claves; modelo proyectado; el valor es de todo el píxel;
                 # y el código EPSG de la proyección
                 (_GEOCLAVES, 'H', 16, (1, 1, 0, 3, 1024, 0, 1, 1, 1025, 0, 1, 1,
                                        _CLAVE_PROYECTADO, 0, 1, int(rej['epsg'])), True)]
        if self.sin_dato is not None:
            extra.append((_GDAL_SIN_DATO, 's', 0, _texto_sin_dato(self.sin_dato), True))
        xml = _xml_etiquetas(self._etiquetas, self._nombres)
        if xml:
            extra.append((_GDAL_ETIQUETAS, 's', 0, xml, True))
        try:
            # con sitio para las versiones reducidas, que se añaden después
            with tifffile.TiffWriter(self.ruta, bigtiff=self._bytes * 1.4 > 2 ** 32 - 2 ** 25) as tw:
                tw.write(self._datos, extratags=extra, **_opciones(self.bandas, self._paleta))
        finally:
            self._soltar()

    def __enter__(self):
        return self

    def __exit__(self, tipo, valor, traza):
        if tipo is None:
            self.cerrar()
        else:
            self.descartar()
        return False


def escribir(ruta, rej, matriz, sin_dato=None, paleta=None, etiquetas=None, nombres=None):
    """Escribe de una vez un GeoTIFF con una matriz (alto, ancho) o (bandas, alto, ancho)
    que ocupa toda la rejilla."""
    import numpy as np
    m = np.asarray(matriz)
    bandas = 1 if m.ndim == 2 else m.shape[0]
    with Escritor(ruta, rej, bandas=bandas, tipo=m.dtype.name, sin_dato=sin_dato) as f:
        if paleta:
            f.paleta(paleta)
        f.etiquetas(**(etiquetas or {}))
        for i, n in enumerate(nombres or (), 1):
            f.nombre_banda(i, n)
        f.escribir(m)


def anadir_reducidas(ruta, categorica):
    """Añade a un GeoTIFF sus versiones reducidas (ver `factores_reducidos` y `reducir`)
    y devuelve los factores. Si ya las tenía, no toca nada. Los datos a tamaño real no
    cambian: las reducidas van detrás, y QGIS las usa para dibujar de lejos."""
    import numpy as np
    import tifffile
    with Lector(ruta) as f:
        factores = factores_reducidos(f.ancho, f.alto)
        if not factores or f.reducidas:
            return list(f.reducidas)
        paleta, sin_dato, bandas = f.paleta, f.sin_dato, f.bandas
        capas = [f.leer(b) for b in range(1, bandas + 1)]
    op = _opciones(bandas, paleta)
    with tifffile.TiffWriter(ruta, append=True) as tw:
        for k in factores:
            menores = [reducir(c, k, categorica, sin_dato) for c in capas]
            tw.write(menores[0] if bandas == 1 else np.stack(menores, axis=-1),
                     subfiletype=1, **op)
    return factores


# ============================================================
# Leer
# ============================================================
class Lector:
    """Un GeoTIFF abierto para leer: su rejilla, lo que lleva dentro y sus píxeles, por
    ventanas (solo se descomprimen las teselas que hacen falta).

    ancho, alto, bandas, tipo, sin_dato; epsg, x0, y0 (esquina superior izquierda) y
    escala (metros por píxel); etiquetas (dict), nombres (el de cada banda), paleta
    ({codigo: (r, g, b, a)} o None; a = 255, salvo el código sin dato, que es
    transparente) y reducidas (los factores de sus versiones reducidas, de menor a
    mayor).
    """
    MAX_TESELAS = 96            # las últimas descomprimidas, por si se vuelven a pedir

    def __init__(self, ruta):
        import tifffile
        self.ruta = Path(ruta)
        self._tif = tifffile.TiffFile(self.ruta)
        try:
            self._tif.pages.useframes = False
            paginas = list(self._tif.pages)
            p = paginas[0]
            self.ancho, self.alto = int(p.imagewidth), int(p.imagelength)
            self.bandas, self.tipo = int(p.samplesperpixel), p.dtype.name
            if _ESCALA not in p.tags or _ANCLAJE not in p.tags:
                raise ValueError(f'{self.ruta.name} no lleva georreferencia')
            sx, sy = (float(v) for v in p.tags[_ESCALA].value[:2])
            i, j, _, x, y = (float(v) for v in p.tags[_ANCLAJE].value[:5])
            if abs(sx - sy) > 1e-9 * max(sx, sy):
                raise ValueError(f'los píxeles de {self.ruta.name} no son cuadrados')
            self.escala, self.x0, self.y0 = sx, x - i * sx, y + j * sy
            claves = _geoclaves(p.tags[_GEOCLAVES].value) if _GEOCLAVES in p.tags else {}
            self.epsg = claves.get(_CLAVE_PROYECTADO) or claves.get(_CLAVE_GEOGRAFICO)
            self.sin_dato = None
            if _GDAL_SIN_DATO in p.tags:
                v = float(_texto(p.tags[_GDAL_SIN_DATO].value))
                self.sin_dato = v if self.tipo.startswith('float') else int(v)
            self.etiquetas, self.nombres = ({}, ('',) * self.bandas)
            if _GDAL_ETIQUETAS in p.tags:
                self.etiquetas, self.nombres = _leer_xml(p.tags[_GDAL_ETIQUETAS].value, self.bandas)
            self.paleta = None
            if int(p.photometric) == _CON_PALETA and p.colormap is not None:
                tabla = p.colormap
                # opaco todo menos el código que significa «sin dato», como lo ve GDAL
                self.paleta = {c: (int(tabla[0, c]) >> 8, int(tabla[1, c]) >> 8,
                                   int(tabla[2, c]) >> 8, 0 if c == self.sin_dato else 255)
                               for c in range(tabla.shape[1])}
            self._paginas, self.reducidas = [p], []
            for q in paginas[1:]:
                k = int(round(self.ancho / q.imagewidth)) if q.is_reduced else 0
                if (k > 1 and (q.imagewidth, q.imagelength) == (-(-self.ancho // k), -(-self.alto // k))):
                    self._paginas.append(q)
                    self.reducidas.append(k)
        except BaseException:
            self._tif.close()
            raise
        self._teselas = {}

    def close(self):
        self._tif.close()
        self._teselas = {}

    cerrar = close

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
        return False

    # ---- la rejilla
    @property
    def rej(self):
        return {'epsg': self.epsg, 'escala': self.escala, 'x0': self.x0, 'y0': self.y0,
                'ancho': self.ancho, 'alto': self.alto}

    @property
    def limites(self):
        """(minx, miny, maxx, maxy) de lo que ocupa, en sus coordenadas."""
        return (self.x0, self.y0 - self.alto * self.escala,
                self.x0 + self.ancho * self.escala, self.y0)

    def indice(self, x, y):
        """(fila, columna) del píxel que contiene el punto (x, y); puede caer fuera."""
        return (math.floor((self.y0 - y) / self.escala), math.floor((x - self.x0) / self.escala))

    def centro(self, fila, col):
        """(x, y) del centro del píxel (fila, columna)."""
        return (self.x0 + (col + 0.5) * self.escala, self.y0 - (fila + 0.5) * self.escala)

    def forma(self, nivel=None):
        """(alto, ancho) del ráster, o de una de sus versiones reducidas (0, 1…)."""
        p = self._paginas[0 if nivel is None else nivel + 1]
        return int(p.imagelength), int(p.imagewidth)

    # ---- los píxeles
    def _tesela(self, n_pag, indice):
        clave = (n_pag, indice)
        t = self._teselas.get(clave)
        if t is None:
            p = self._paginas[n_pag]
            cuanto = p.databytecounts[indice]
            t = None
            if cuanto:
                fh = self._tif.filehandle
                fh.seek(p.dataoffsets[indice])
                t = p.decode(fh.read(cuanto), indice, _fullsize=True)[0]
            if t is None:                           # tesela que no se llegó a escribir
                import numpy as np
                t = np.full((1, p.tilelength, p.tilewidth, self.bandas),
                            0 if self.sin_dato is None else self.sin_dato, dtype=self.tipo)
            if len(self._teselas) >= self.MAX_TESELAS:
                self._teselas.pop(next(iter(self._teselas)))
            self._teselas[clave] = t
        return t

    def leer(self, banda=None, ventana=None, nivel=None):
        """Los valores de una ventana (columna, fila, ancho, alto); None = todo. Con
        `banda` (1, 2…) devuelve (alto, ancho); sin ella, todas: (bandas, alto, ancho).
        nivel: None = a tamaño real; 0, 1… = de esa versión reducida (la ventana va
        entonces en píxeles de esa versión)."""
        import numpy as np
        n_pag = 0 if nivel is None else nivel + 1
        p = self._paginas[n_pag]
        alto_p, ancho_p = int(p.imagelength), int(p.imagewidth)
        col, fila, ancho, alto = ventana or (0, 0, ancho_p, alto_p)
        if not (0 <= col and 0 <= fila and col + ancho <= ancho_p and fila + alto <= alto_p
                and ancho > 0 and alto > 0):
            raise ValueError(f'la ventana {(col, fila, ancho, alto)} se sale del ráster')
        if banda is not None and not 1 <= int(banda) <= self.bandas:
            raise ValueError(f'{self.ruta.name} no tiene banda {banda}')
        juntas = int(p.planarconfig) == _BANDAS_JUNTAS
        if not p.is_tiled or not juntas:
            # un GeoTIFF que no es de los nuestros: se lee entero
            todo = self._teselas.get((n_pag, 'todo'))
            if todo is None:
                todo = p.asarray()
                if todo.ndim == 2:
                    todo = todo[..., None]
                elif not juntas:
                    todo = np.moveaxis(todo, 0, -1)
                self._teselas[(n_pag, 'todo')] = todo
            out = todo[fila:fila + alto, col:col + ancho]
        else:
            tl, tw = int(p.tilelength), int(p.tilewidth)
            por_fila = -(-ancho_p // tw)
            out = np.empty((alto, ancho, self.bandas), dtype=self.tipo)
            for tf in range(fila // tl, (fila + alto - 1) // tl + 1):
                for tc in range(col // tw, (col + ancho - 1) // tw + 1):
                    t = self._tesela(n_pag, tf * por_fila + tc)[0]
                    f0, f1 = max(fila, tf * tl), min(fila + alto, (tf + 1) * tl)
                    c0, c1 = max(col, tc * tw), min(col + ancho, (tc + 1) * tw)
                    out[f0 - fila:f1 - fila, c0 - col:c1 - col] = \
                        t[f0 - tf * tl:f1 - tf * tl, c0 - tc * tw:c1 - tc * tw]
        if banda is not None:
            return np.ascontiguousarray(out[..., int(banda) - 1])
        return np.ascontiguousarray(np.moveaxis(out, -1, 0))

    def valor(self, fila, col, banda=1):
        """El valor de un píxel."""
        return self.leer(banda, (int(col), int(fila), 1, 1))[0, 0]


def abrir(ruta):
    """Abre un GeoTIFF para leerlo (ver Lector). Úsese con `with`."""
    return Lector(ruta)
