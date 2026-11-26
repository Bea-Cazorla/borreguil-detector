"""Los registros de Borreguil en este ordenador y cómo enviarlos al equipo cuando algo falla.

Hecho como en el asistente de MonitorEO (monitoreo.logs), 26-11-2026, la autora: «en las
incidencias que suban los log de la app, tal cual en monitoreo está hecho en el asistente».

La app escribe en registro.log, en la carpeta Borreguil del usuario (BORREGUIL_DATA_DIR, si
no ~/Borreguil), todo lo que sale por la consola: lo que va haciendo el análisis, los avisos
y los errores de Python con su traza, también los que la app enseña en pantalla con st.error
y, si Python se cae del todo, lo que deja faulthandler. Guarda sus últimos MAX_BYTES.

«Enviar los registros» enseña a quien usa la app lo que reúne recoger(), que puede cambiar
o borrar, y enviar() lo manda al formulario anónimo de la app (FORMULARIO) como un error, en
la pregunta «Texto de la consola», con lo que haya escrito: sin cuenta, nombre ni correo.
Antes de que lo vea, a los registros se les quita el ruido (compactar: avisos repetidos,
reintentos de peticiones partidos en muchas líneas) y lo que diría quién es (ocultar_datos):
su carpeta personal, su nombre de usuario y el del ordenador, correos, claves y tokens, el
identificador de la instalación y su proyecto de Earth Engine. Lo que reúne recoger() cabe
en una petición al formulario con una descripción, así que lo que ve es lo que se envía."""

import collections
import contextlib
import datetime as dt
import faulthandler
import getpass
import http.client
import io
import os
import platform
import re
import socket
import sys
import threading
import traceback
import urllib.error
import urllib.parse
import urllib.request

FICHERO = 'registro.log'
MAX_BYTES = 2 * 2 ** 20          # un registro más grande se queda con su mitad más nueva
LEER = MAX_BYTES                 # de cada registro, los bytes más nuevos que se compactan
# Bytes de la petición al formulario como mucho (codificada; ver tamano). Los formularios de
# Google aceptaron 32.673 bytes y contestaron HTTP 413 a 32.829 y más (MonitorEO, 19-11-2026).
BODY_BYTES = 30000
LOGS_BYTES = 24000               # para los registros: el resto, para la descripción
DESCRIPCION_BYTES = 4000         # de una descripción larga, lo que se guarda aunque se acorten
CABEZA = 2000                    # bytes del principio que guarda acortar() (versión y sistema)
SIN_DESCRIPCION = '(Solo los registros: no se escribió nada.)'
# Palabras que no son un nombre de usuario (así se llaman algunas cuentas): cambiarlas
# escondería el texto de los registros.
COMUNES = {'user', 'users', 'admin', 'root', 'test', 'guest', 'usuario', 'borreguil',
           'python', 'streamlit'}

# El formulario anónimo de la app y los identificadores de sus preguntas
# (FB_PUBLIC_LOAD_DATA_ de su página, 26-11-2026).
FORMULARIO = ('https://docs.google.com/forms/d/e/'
              '1FAIpQLScq7UehnDlaNaYsq9DZNhXc-CxRo57KgdT9e98U0q1xGglRPg/viewform')
FORMULARIO_RESPUESTA = FORMULARIO.rsplit('/', 1)[0] + '/formResponse'
CAMPOS = {'que': 'entry.848699343', 'parte': 'entry.2000956224', 'version': 'entry.666897268',
          'ordenador': 'entry.146765963', 'descripcion': 'entry.185558525',
          'pantallazos': 'entry.1817049282', 'consola': 'entry.1440064022'}
ERROR = 'Report an error / Informar de un error'     # la opción de «¿Qué quieres hacer?»
PARTES = {   # «¿En qué parte de la app?»: lo que se enseña y la opción del formulario
    'arranque': ('Instalación o arranque', 'Installation or start',
                 'Installation or start / Instalación o arranque'),
    'analisis': ('Análisis de los puntos', 'Analysis of the points',
                 'Analysis of the points / Análisis de los puntos'),
    'revisar': ('Revisar puntos', 'Reviewing points', 'Reviewing points / Revisar puntos'),
    'mapa': ('Mapa de borreguiles (Mapa Modelo RF)', 'Borreguil map (RF model map)',
             'Borreguil map (RF model map) / Mapa de borreguiles (Mapa Modelo RF)'),
    'manual': ('Documentación o manual', 'Documentation or manual',
               'Documentation or manual / Documentación o manual'),
    'otra': ('Otra', 'Other', 'Other / Otra'),
}
ORDENADORES = {'Windows': 'Windows', 'Darwin': 'macOS', 'Linux': 'Linux'}
NO_SE = "I don't know / No lo sé"
REPOSITORIO = 'https://github.com/monitoreo-obsnev/borreguil-detector'
TEXTOS = {   # los de MonitorEO (about.LOGS), para esta app
    'es': {'menu': 'Enviar los registros de este ordenador con el formulario anónimo: los ves '
                   'antes, sin tu nombre, correo ni carpetas',
           'boton': 'Enviar los registros',
           'ayuda': 'Estas son las últimas líneas de los registros de Borreguil en este '
                    'ordenador: lo que hizo y los errores que tuvo. Van al equipo con el '
                    'formulario anónimo (sin cuenta, nombre ni correo), sin tus carpetas, '
                    'nombre de usuario, correos, claves ni tu proyecto de Earth Engine. '
                    'Léelos, y borra o cambia lo que prefieras no enviar.',
           'descripcion': '¿Qué estabas haciendo y qué pasó? (opcional)',
           'parte': '¿En qué parte de la app?',
           'registros': 'Registros',
           'cancelar': 'Cancelar',
           'enviar': 'Enviar',
           'vacio': 'No hay registros que enviar.',
           'enviados': 'Enviados: gracias. El equipo los lee con las respuestas del '
                       'formulario anónimo.'},
    'en': {'menu': 'Send the logs of this computer with the anonymous form: you see them '
                   'first, without your name, email or folders',
           'boton': 'Send the logs',
           'ayuda': 'These are the newest lines of the logs of Borreguil on this computer: '
                    'what it did and the errors it had. They go to the team with the anonymous '
                    'form (no account, no name, no email), without your folders, user name, '
                    'email addresses, keys or your Earth Engine project. Read them, and delete '
                    'or change anything you prefer not to send.',
           'descripcion': 'What were you doing, and what happened? (optional)',
           'parte': 'Which part of the app?',
           'registros': 'Logs',
           'cancelar': 'Cancel',
           'enviar': 'Send',
           'vacio': 'There are no logs to send.',
           'enviados': 'Sent: thank you. The team reads them with the answers of the '
                       'anonymous form.'},
}

# Lo que diría quién es y con qué se cambia
_SECRETOS = [
    (re.compile(r'-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----', re.S),
     '<clave privada>'),
    (re.compile(r'\bya29\.[\w.-]+'), '<token>'),              # tokens de acceso de Google
    (re.compile(r'\b1//[\w.-]{20,}'), '<token>'),             # tokens de renovación de Google
    (re.compile(r'\bsk-ant-[\w-]+'), '<clave>'),              # claves de Anthropic
    (re.compile(r'\bAIza[\w-]{30,}'), '<clave>'),             # claves de Google
    (re.compile(r'\bPLAK[\w]{20,}'), '<clave>'),              # claves de Planet
    (re.compile(r'(?i)\b(bearer\s+)[\w.~+/-]+=*'), r'\1<token>'),
    (re.compile(r'(?i)(?<![\w-])((?:access_token|refresh_token|id_token|client_secret|'
                r'private_key_id|private_key|api[_-]?key|pl_api_key|password|token|secret|'
                r'x-token)["\']?\s*[:=]\s*["\']?)[^"\'\s,&}]+'), r'\1<oculto>'),
    (re.compile(r'(?i)([?&](?:t|token|key|access_token|code)=)[^&\s"\'<>]+'), r'\1<oculto>'),
    (re.compile(r'[\w.+-]+@(?![\w-]+\.iam\.gserviceaccount\.com\b)[\w-]+(?:\.[\w-]+)+'),
     '<correo>'),
    (re.compile(r'\bmo-[0-9a-f]{16}\b'), 'mo-<instalación>'),  # el identificador anónimo
    # las carpetas de otros usuarios (la de este ya es ~)
    (re.compile(r'(?i)((?:/Users/|/home/|[A-Z]:\\{1,2}Users\\{1,2}))[^/\\\s"\']+'), r'\1<usuario>'),
]

# Lo que compactar() no quita ni cambia nunca: las líneas de los errores y de las trazas,
# también las de faulthandler cuando Python se cae (Fatal Python error, Windows fatal
# exception, Current thread …, File "…"), y las sangradas tras ellas.
_GUARDAR = re.compile(r'(?i)error|exception|traceback|fatal|current thread|thread 0x|'
                      r'file "|most recent call')
_AVISO = re.compile(r'\S.*:\d+: \w*Warning: ')     # un aviso de Python (warnings.warn)
_REINTENTO = re.compile(r'\bSleeping \d+(?:\.\d+)? seconds\b')   # googleapiclient, en WARNING
_REINTENTO_TEXTO = re.compile(r'Sleeping [\d.]+ seconds before retry \d+ of \d+ for '
                              r'(?:request|media upload): (?P<method>[A-Z]+) (?P<uri>.+?), '
                              r'after (?P<after>\S.*)$')
_HORA = re.compile(r'\[([^\]]+)\]\s')
_SEGURO = frozenset('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-~ ')


# ── Dónde están ────────────────────────────────────────────────────────────────────

def carpeta():
    """La carpeta de los registros: la de los modelos y salidas de la app."""
    env = os.environ.get('BORREGUIL_DATA_DIR')
    return env if env else os.path.join(os.path.expanduser('~'), 'Borreguil')


def ruta(donde=None):
    return os.path.join(donde or carpeta(), FICHERO)


# ── Escribir ───────────────────────────────────────────────────────────────────────

def _ultimos_bytes(fichero, cuantos):
    """El texto de los últimos CUANTOS bytes de FICHERO, desde su primera línea entera."""
    with open(fichero, 'rb') as f:
        f.seek(0, os.SEEK_END)
        largo = f.tell()
        f.seek(max(largo - cuantos, 0))
        texto = f.read().decode('utf-8', errors='replace')
    if largo > cuantos:
        texto = texto.split('\n', 1)[1] if '\n' in texto else ''
    return texto


class _Registro:
    """El fichero del registro, para todos los hilos. Escribir nunca falla (disco lleno,
    carpeta sin permiso): entonces, simplemente, no se apunta."""

    def __init__(self, fichero):
        self.fichero = fichero
        self.cerrojo = threading.RLock()
        self.f = None
        self.escritos = 0
        self._abrir()

    def _abrir(self):
        with contextlib.suppress(OSError, ValueError):
            os.makedirs(os.path.dirname(self.fichero) or '.', exist_ok=True)
            if os.path.exists(self.fichero) and os.path.getsize(self.fichero) > MAX_BYTES:
                guardado = _ultimos_bytes(self.fichero, MAX_BYTES // 2)
                with open(self.fichero, 'w', encoding='utf-8', newline='\n') as f:
                    f.write(guardado)
            self.f = open(self.fichero, 'a', encoding='utf-8', errors='replace', newline='\n')
            self.escritos = os.path.getsize(self.fichero)
            with contextlib.suppress(Exception):   # lo que deja Python si se cae del todo
                faulthandler.enable(file=self.f, all_threads=True)

    def escribir(self, texto):
        if not texto:
            return
        with self.cerrojo:
            if self.f is None:
                return
            try:
                self.f.write(texto)
                self.f.flush()
                self.escritos += len(texto)
                if self.escritos > MAX_BYTES:            # se queda con su mitad más nueva
                    with contextlib.suppress(Exception):
                        faulthandler.disable()
                    self.f.close()
                    self.f = None
                    self._abrir()
            except (OSError, ValueError):
                pass


class _Copia(io.TextIOBase):
    """sys.stdout o sys.stderr: lo que se escribe va a la consola de siempre y, además,
    al registro."""
    _es_copia_al_registro = True

    def __init__(self, original, registro):
        self.original = original
        self.registro = registro

    def write(self, s):
        if self.original is not None:
            with contextlib.suppress(Exception):
                self.original.write(s)
        self.registro.escribir(s)
        return len(s)

    def flush(self):
        if self.original is not None:
            with contextlib.suppress(Exception):
                self.original.flush()

    @property
    def encoding(self):
        return getattr(self.original, 'encoding', None) or 'utf-8'

    def isatty(self):
        try:
            return bool(self.original and self.original.isatty())
        except Exception:
            return False

    def fileno(self):
        return self.original.fileno()


_registro = None


def empezar(version='', donde=None):
    """Desde ahora (y hasta que se cierre el programa), lo que sale por la consola se
    apunta también en el registro. Se puede llamar en cada pasada de la app: solo hace
    algo la primera vez. Si sys.stdout ya reparte por hilos (app.py, _SalidaPorHilo), la
    copia va debajo, en su salida de siempre."""
    global _registro
    if _registro is not None or os.environ.get('BORREGUIL_SIN_REGISTRO'):
        return _registro          # las pruebas que abren la app sin carpeta propia
    _registro = _Registro(ruta(donde))
    for nombre in ('stdout', 'stderr'):
        actual = getattr(sys, nombre)
        de_hilos = hasattr(actual, 'hilos') and hasattr(actual, 'original')
        base = actual.original if de_hilos else actual
        if getattr(base, '_es_copia_al_registro', False):
            continue
        copia = _Copia(base, _registro)
        if de_hilos:
            actual.original = copia
        else:
            setattr(sys, nombre, copia)
        if nombre == 'stderr':
            _a_los_mensajes_de_streamlit(copia, base)
    _registro.escribir(f'\n=== Borreguil {version} · {_ahora()} · {platform.platform()} '
                       f'· Python {platform.python_version()} ===\n')
    return _registro


def _a_los_mensajes_de_streamlit(copia, base):
    """Los avisos y errores de Streamlit (p. ej. «Uncaught app execution», con su traza)
    salen por sus propios manejadores, atados al sys.stderr de cuando se crearon (BASE):
    se les pone la COPIA, para que también lleguen al registro."""
    import logging
    for nombre, logger in list(logging.Logger.manager.loggerDict.items()):
        if not nombre.startswith('streamlit') or not isinstance(logger, logging.Logger):
            continue
        for h in logger.handlers:
            if type(h) is logging.StreamHandler and h.stream in (base, sys.__stderr__):
                with contextlib.suppress(Exception):
                    h.setStream(copia)


def anotar(texto):
    """Apunta TEXTO en el registro (sin sacarlo por la consola)."""
    if _registro is not None:
        _registro.escribir(texto)


def anotar_error(mensaje, exc_info=None):
    """Apunta un error que la app enseña en pantalla (st.error) y, si viene de una
    excepción de Python, su traza."""
    if _registro is None:
        return
    texto = f'[{_ahora()}] ERROR en pantalla: {mensaje}\n'
    if exc_info and exc_info[0] is not None:
        with contextlib.suppress(Exception):
            texto += ''.join(traceback.format_exception(*exc_info))
    _registro.escribir(texto)


def _ahora():
    return dt.datetime.now().isoformat(timespec='seconds')


# ── Compactar y ocultar ────────────────────────────────────────────────────────────

def tamano(texto):
    """Los bytes de TEXTO en la petición al formulario (codificado): 1 por cada letra,
    cifra, espacio o _.-~, y 3 por cada byte en UTF-8 de cualquier otro carácter."""
    return len(urllib.parse.quote_plus(texto))


def _bytes(c):
    return 1 if c in _SEGURO else 3 * len(c.encode('utf-8'))


def _principio(texto, cuantos):
    usados = 0
    for i, c in enumerate(texto):
        usados += _bytes(c)
        if usados > cuantos:
            return texto[:i]
    return texto


def _lo_mas_nuevo(texto, cuantos):
    """El final de TEXTO que ocupa como mucho CUANTOS bytes, desde una línea entera."""
    usados = 0
    for i in range(len(texto) - 1, -1, -1):
        usados += _bytes(texto[i])
        if usados > cuantos:
            parte = texto[i + 1:]
            corte = parte.find('\n', 0, len(parte) - 1)
            return parte if texto[i] == '\n' or corte < 0 else parte[corte + 1:]
    return texto


def compactar(texto):
    """TEXTO (las líneas de un registro) sin su ruido, lo más nuevo al final como estaba:
    cada aviso de Python (con su línea de código) solo la última vez, con cuántas veces
    salió; cada reintento de una petición de googleapiclient en una línea, y los seguidos en
    una, con cuántos fueron; y las líneas repetidas seguidas, una vez con cuántas veces. Las
    líneas de los errores y de las trazas (_GUARDAR) no cambian nunca."""
    lineas, unidades = texto.replace('\r\n', '\n').split('\n'), []
    ultima, guardada, i = None, False, 0
    while i < len(lineas):
        linea = lineas[i]
        reint = _REINTENTO.search(linea)
        reint = reint if reint and 'WARNING' in linea[:reint.start()] else None
        guardada = bool(_GUARDAR.search(linea)) or (
            guardada and not reint and linea[:1] in (' ', '\t', '│', '╭', '╰'))
        propia = _HORA.match(linea)
        ultima = propia.group(1) if propia else ultima
        if reint and not guardada:
            col, j = reint.start(), i + 1
            while j < len(lineas) and lineas[j].startswith(' ' * col) and lineas[j].strip():
                j += 1
            primera = re.sub(r'\s+\S+\.py:\d+\s*$', '', linea[col:])
            mensaje = ' '.join([primera.strip()] + [x.strip() for x in lineas[i + 1:j]])
            m = _REINTENTO_TEXTO.match(mensaje)
            if m and not _GUARDAR.search(mensaje):
                uri = re.sub(r'\s+', '', m['uri'])
                unidades.append(['reintento', lineas[i:j], {
                    'hora': ultima if propia or linea[:1].isspace() else None,
                    'peticion': f"{m['method']} {uri}", 'tras': m['after'],
                    'texto': mensaje[:m.start('uri')] + uri + mensaje[m.end('uri'):]}])
                i = j
                continue
        if not guardada and _AVISO.match(linea):
            bloque = lineas[i:i + 1]
            if i + 1 < len(lineas) and lineas[i + 1].startswith('  ') and lineas[i + 1].strip() \
                    and not _GUARDAR.search(lineas[i + 1]):
                bloque.append(lineas[i + 1])
            unidades.append(['aviso', bloque, None])
            i += len(bloque)
            continue
        repetible = linea.strip() and not guardada
        unidades.append(['linea' if repetible else 'guardar', [linea], None])
        i += 1
    veces, vistos = collections.Counter(tuple(u[1]) for u in unidades if u[0] == 'aviso'), set()
    for u in reversed(unidades):          # cada aviso, la última vez
        if u[0] == 'aviso':
            clave = tuple(u[1])
            if clave in vistos:
                u[0] = None
            elif veces[clave] > 1:
                u[1] = [f'{u[1][0]} ({veces[clave]} veces; solo se ve la última)'] + u[1][1:]
            vistos.add(clave)
    unidades, salida, i = [u for u in unidades if u[0]], [], 0
    while i < len(unidades):
        tipo, bloque = unidades[i][:2]
        j = i + 1
        while j < len(unidades) and unidades[j][0] == tipo and tipo in ('reintento', 'linea') \
                and (tipo == 'reintento' or unidades[j][1] == bloque):
            j += 1
        if tipo == 'reintento':
            salida.append(_reintentos([u[2] for u in unidades[i:j]]))
        elif tipo == 'linea':
            salida.append(bloque[0] + (f' (repetida {j - i} veces)' if j > i + 1 else ''))
        else:
            salida += bloque
        i = j
    return '\n'.join(salida)


def _reintentos(reints):
    primero, ultimo = reints[0], reints[-1]
    inicio = f"[{primero['hora']}] WARNING  " if primero['hora'] else 'WARNING  '
    if len(reints) == 1:
        return inicio + primero['texto']
    tras = collections.Counter(r['tras'] for r in reints)
    peticiones = list(dict.fromkeys(r['peticion'] for r in reints))
    return (f'{inicio}googleapiclient reintentó peticiones {len(reints)} veces' +
            (f" hasta [{ultimo['hora']}]" if ultimo['hora'] else '') + ': ' +
            ', '.join(f'{n} tras {a}' for a, n in tras.items()) + '; ' +
            '; '.join(peticiones[:3]) + ('; …' if len(peticiones) > 3 else ''))


def ocultar_datos(texto, ocultar=()):
    """TEXTO sin lo que diría quién es: su carpeta personal (~), su nombre de usuario
    (<usuario>) y el del ordenador (<ordenador>), correos (<correo>; no los de las cuentas
    de servicio de Google Cloud), claves y tokens, el identificador de la instalación y cada
    texto de OCULTAR (p. ej. su proyecto de Earth Engine: <proyecto>)."""
    casa = os.path.expanduser('~')
    if len(casa) > 3:
        for h in sorted({casa, casa.replace('\\', '/'), casa.replace('\\', '\\\\')},
                        key=len, reverse=True):
            texto = texto.replace(h, '~')
    for patron, cambio in _SECRETOS:   # antes que los nombres: un correo o un token los llevan
        texto = patron.sub(cambio, texto)
    for valor in ocultar:              # antes que el usuario, que un proyecto puede llevar
        if valor and len(str(valor)) >= 3:
            texto = texto.replace(str(valor), '<proyecto>')
    for nombre, que in ((_usuario(), '<usuario>'), (_ordenador(), '<ordenador>')):
        if nombre and len(nombre) >= 4 and nombre.lower() not in COMUNES:
            texto = re.sub(rf'(?i)(?<![\w-]){re.escape(nombre)}(?![\w-])', que, texto)
    return texto


def _usuario():
    try:
        return getpass.getuser()
    except Exception:
        return None


def _ordenador():
    with contextlib.suppress(OSError):
        return socket.gethostname().split('.')[0]
    return None


def _paquete(nombre):
    try:
        from importlib import metadata
        return metadata.version(nombre)
    except Exception:
        return None


# ── Recoger y enviar ───────────────────────────────────────────────────────────────

def recoger(version='', ocultar=(), donde=None, max_bytes=LOGS_BYTES):
    """Lo que «Enviar los registros» enseña y envía: la versión de la app, el sistema y
    las últimas líneas del registro de la carpeta DONDE (por defecto, carpeta()) sin su
    ruido (compactar), en como mucho MAX_BYTES bytes de la petición al formulario, sin lo
    que diría quién es (ocultar_datos, con OCULTAR)."""
    paquetes = [f'{n} {v}' for n, v in (('earthengine-api', _paquete('earthengine-api')),
                                         ('streamlit', _paquete('streamlit'))) if v]
    cabeza = [f'Registros de Borreguil · Wet Meadow Finder {version}'.rstrip(),
              f'Recogidos: {_ahora()}',
              f'Sistema: {platform.platform()}; Python {platform.python_version()}'
              + ''.join(f'; {p}' for p in paquetes)]
    texto = ocultar_datos('\n'.join(cabeza) + '\n', ocultar)
    fichero = ruta(donde)
    if not os.path.isfile(fichero) or not os.path.getsize(fichero):
        return texto + '\n(Todavía no hay registros de Borreguil en este ordenador.)\n'
    parte = compactar(_ultimos_bytes(fichero, LEER))
    if len(parte) > 4 * max_bytes:          # solo se oculta lo que puede caber
        parte = parte[-4 * max_bytes:].split('\n', 1)[-1]
    parte = ocultar_datos(parte, ocultar)
    titulo = f'\n== {FICHERO} (sus últimas líneas) ==\n'
    return texto + titulo + _lo_mas_nuevo(parte, max_bytes - tamano(texto) - tamano(titulo))


def acortar(texto, max_bytes=LOGS_BYTES):
    """TEXTO en como mucho MAX_BYTES bytes de la petición: sus primeras líneas (la versión
    y el sistema, hasta la primera línea vacía, en como mucho CABEZA bytes) y las últimas."""
    if tamano(texto) <= max_bytes:
        return texto
    cabeza = _principio(texto, min(CABEZA, max_bytes // 4))
    cabeza = (cabeza[:cabeza.index('\n\n') + 1] if '\n\n' in cabeza
              else cabeza[:cabeza.rfind('\n') + 1])
    hueco = '[…]\n'
    return cabeza + hueco + _lo_mas_nuevo(texto[len(cabeza):],
                                          max_bytes - tamano(cabeza) - tamano(hueco))


def ordenador_del_formulario():
    """La opción de «Ordenador» del formulario para este sistema."""
    return ORDENADORES.get(platform.system(), NO_SE)


def enviar(texto, version='', descripcion='', parte='otra', timeout=30):
    """Envía los registros TEXTO (como los vio y quizá cambió quien usa la app) al
    formulario anónimo, como un error de la parte PARTE (una clave de PARTES) de la app en
    VERSION, con DESCRIPCION (lo que escribió). La petición ocupa como mucho BODY_BYTES:
    una descripción larga guarda su principio, y unos registros más largos que lo que deja
    (acortar), su principio y sus últimas líneas. ValueError si no hay nada que enviar;
    RuntimeError, con qué hacer en su lugar, si el formulario no los coge."""
    texto = (texto or '').strip()
    if not texto:
        raise ValueError('no hay registros que enviar')
    descripcion = (descripcion or '').strip() or SIN_DESCRIPCION
    datos = {CAMPOS['que']: ERROR, CAMPOS['parte']: PARTES.get(parte, PARTES['otra'])[2],
             CAMPOS['ordenador']: ordenador_del_formulario(), CAMPOS['descripcion']: '',
             CAMPOS['consola']: ''}
    if version:
        datos[CAMPOS['version']] = str(version)
    sitio = BODY_BYTES - len(urllib.parse.urlencode(datos))   # descripción y registros
    texto = acortar(texto, sitio - min(tamano(descripcion), DESCRIPCION_BYTES))
    if tamano(descripcion) > sitio - tamano(texto):
        hueco = ' […]'
        descripcion = _principio(descripcion, sitio - tamano(texto) - tamano(hueco)) + hueco
    datos[CAMPOS['descripcion']], datos[CAMPOS['consola']] = descripcion, texto
    peticion = urllib.request.Request(
        FORMULARIO_RESPUESTA, data=urllib.parse.urlencode(datos).encode('utf-8'),
        headers={'User-Agent': 'borreguil', 'Content-Type': 'application/x-www-form-urlencoded'})
    en_su_lugar = (f'guárdalos y usa el formulario anónimo ({FORMULARIO}) o una issue de GitHub '
                   f'({REPOSITORIO}/issues) en su lugar')
    try:
        # la dirección del formulario, https (el aviso de Bandit sobre file:/ no se aplica)
        with urllib.request.urlopen(peticion, timeout=timeout) as r:   # nosec B310
            final = r.geturl()
    except urllib.error.HTTPError as e:      # antes que OSError: un HTTPError también lo es
        raise RuntimeError(f'El formulario no ha cogido los registros (HTTP {e.code}): '
                           f'{en_su_lugar}') from None
    except (OSError, http.client.HTTPException) as e:
        raise RuntimeError(f'No se han podido enviar los registros ({e}): mira la conexión, '
                           f'o {en_su_lugar}') from None
    if 'closedform' in (final or ''):
        raise RuntimeError(f'El formulario no admite respuestas ahora: {en_su_lugar}')
