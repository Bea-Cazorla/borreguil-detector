"""Modelos públicos que la app descarga de GitHub, sin reinstalarla.

Los modelos Random Forest que el equipo quiere compartir se suben al repositorio junto
con `modelos.json`: la lista de ficheros, con el tamaño y la huella (SHA-256) de cada
uno. Al abrirse, la app lee esa lista y descarga a la carpeta del usuario los modelos
que no tiene o que han cambiado. Un modelo descargado pasa delante del que trae el
instalador con el mismo nombre. Sin conexión, se usan los que ya hay.

Para publicar un modelo: `py publicar_modelo.py <modelo.joblib>` (ver PUBLICAR.md).

Seguridad: un .joblib ejecuta código al cargarse. Por eso solo se descarga de la
dirección configurada (por defecto, el repositorio del proyecto), por HTTPS y del mismo
sitio que la lista, y cada fichero se comprueba con su huella antes de guardarlo: uno
incompleto o cambiado por el camino se descarta. La huella no protege de quien pueda
subir ficheros al repositorio: eso depende de quién tenga permiso para escribir en él.

Todo va por HTTPS (puerto 443) a raw.githubusercontent.com.
"""
import hashlib
import json
import os
import re
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO_RAW = 'https://raw.githubusercontent.com/monitoreo-obsnev/borreguil-detector/main/'
MANIFIESTO = 'modelos.json'
URL_POR_DEFECTO = REPO_RAW + MANIFIESTO
# Variable de entorno para usar otro repositorio (su dirección de modelos.json) o, vacía,
# para no buscar modelos (aulas sin internet, pruebas).
VARIABLE_URL = 'BORREGUIL_MODELOS_URL'
CARPETA = 'modelos_descargados'            # dentro de la carpeta de datos del usuario
CARPETA_SIN_RADAR = 'sin_radar'            # la misma que usa borreguil_pipeline
TIEMPO_S = 8                               # espera máxima de cada conexión
MAX_BYTES = 200 * 1024 * 1024              # ningún modelo pesa tanto
FORMATO = 1                                # versión del formato de modelos.json
_NOMBRE_OK = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]*\.joblib$')
_SHA_OK = re.compile(r'^[0-9a-f]{64}$')


def url_manifiesto():
    """La dirección de modelos.json: la de BORREGUIL_MODELOS_URL si está definida (vacía:
    no buscar) o la del repositorio del proyecto."""
    v = os.environ.get(VARIABLE_URL)
    return URL_POR_DEFECTO if v is None else v.strip()


def huella(ruta):
    """SHA-256 de un fichero, en hexadecimal."""
    h = hashlib.sha256()
    with open(ruta, 'rb') as f:
        for trozo in iter(lambda: f.read(1 << 20), b''):
            h.update(trozo)
    return h.hexdigest()


def _abrir(url, timeout):
    req = urllib.request.Request(url, headers={'User-Agent': 'borreguil-detector'})
    return urllib.request.urlopen(req, timeout=timeout)


def _mismo_sitio(url, base):
    a, b = urllib.parse.urlsplit(url), urllib.parse.urlsplit(base)
    return a.scheme == b.scheme and a.netloc == b.netloc


def fichero(d, base):
    """Una entrada de la lista ({archivo, sha256, bytes, url?}) comprobada, con la
    dirección completa. `url` es relativa a la lista (por defecto, `archivo`)."""
    if not isinstance(d, dict):
        raise ValueError('entrada que no es un objeto')
    archivo = str(d.get('archivo', ''))
    if not _NOMBRE_OK.match(archivo):
        raise ValueError(f'nombre de fichero no válido: {archivo!r}')
    sha = str(d.get('sha256', '')).lower()
    if not _SHA_OK.match(sha):
        raise ValueError(f'{archivo}: huella SHA-256 no válida')
    n = d.get('bytes')
    if isinstance(n, bool) or not isinstance(n, int) or not 0 < n <= MAX_BYTES:
        raise ValueError(f'{archivo}: tamaño no válido')
    url = urllib.parse.urljoin(base, str(d.get('url') or archivo))
    if not _mismo_sitio(url, base):
        raise ValueError(f'{archivo}: su dirección no es del mismo sitio que la lista')
    return {'archivo': archivo, 'sha256': sha, 'bytes': n, 'url': url}


def interpretar(datos, base):
    """{'modelos': [...], 'descartados': [motivo]} a partir del contenido de modelos.json.
    Cada modelo: archivo, sha256, bytes, url, nombre, fecha y, si la tiene, sin_radar
    (otra entrada igual). Las entradas mal escritas se descartan sin parar las demás."""
    if not isinstance(datos, dict) or not isinstance(datos.get('modelos'), list):
        raise ValueError('la lista de modelos no tiene el formato esperado')
    formato = datos.get('formato', 1)
    if not isinstance(formato, int) or formato > FORMATO:
        raise ValueError('la lista de modelos es de una versión más nueva de la app: '
                         'actualízala para recibir modelos nuevos')
    modelos, descartados, vistos = [], [], set()
    for d in datos['modelos']:
        try:
            m = fichero(d, base)
            if m['archivo'] in vistos:
                raise ValueError(f"{m['archivo']}: repetido en la lista")
            m['nombre'] = str(d.get('nombre') or '')
            m['fecha'] = str(d.get('fecha') or '')
            sr = d.get('sin_radar')
            m['sin_radar'] = fichero(sr, base) if sr else None
            vistos.add(m['archivo'])
            modelos.append(m)
        except ValueError as e:
            descartados.append(str(e))
    return {'modelos': modelos, 'descartados': descartados}


def leer_manifiesto(url=None, timeout=TIEMPO_S):
    """Lee y comprueba modelos.json (ver `interpretar`). Solo por HTTPS (o un fichero
    local, para las pruebas)."""
    url = url_manifiesto() if url is None else url
    if urllib.parse.urlsplit(url).scheme not in ('https', 'file'):
        raise ValueError('la lista de modelos tiene que ir por HTTPS')
    with _abrir(url, timeout) as r:
        texto = r.read(1_000_001)
    if len(texto) > 1_000_000:
        raise ValueError('la lista de modelos es demasiado grande')
    try:
        datos = json.loads(texto.decode('utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ValueError(f'la lista de modelos no se puede leer: {e}') from None
    return interpretar(datos, url)


def descargar(f, destino, timeout=TIEMPO_S):
    """Descarga una entrada a destino/archivo comprobando su tamaño y su huella. Mientras
    baja va a un fichero oculto aparte, así que un corte no deja un modelo a medias."""
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{f['archivo']}.", suffix='.part', dir=destino)
    h, n = hashlib.sha256(), 0
    try:
        with os.fdopen(fd, 'wb') as out, _abrir(f['url'], timeout) as r:
            while True:
                trozo = r.read(1 << 20)
                if not trozo:
                    break
                n += len(trozo)
                if n > f['bytes']:
                    raise ValueError(f"{f['archivo']}: más grande de lo que dice la lista")
                h.update(trozo)
                out.write(trozo)
        if n != f['bytes'] or h.hexdigest() != f['sha256']:
            raise ValueError(f"{f['archivo']}: la huella no coincide con la de la lista "
                             '(descarga incompleta o fichero distinto)')
        final = destino / f['archivo']
        os.replace(tmp, final)
        return final
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _igual(ruta, sha):
    try:
        return ruta.is_file() and huella(ruta) == sha
    except OSError:
        return False


def actualizar(destino, incluidos=None, url=None, timeout=TIEMPO_S):
    """Pone al día la carpeta `destino` (la de los modelos descargados) con la lista.

    `incluidos` es la carpeta de los modelos que trae la app, que no se toca. Para cada
    modelo de la lista (y su versión sin radar, en la subcarpeta sin_radar):
      - si la app ya lo trae igual (misma huella), no se descarga, y una copia
        descargada distinta se quita: la lista manda, así se puede volver atrás;
      - si no, se descarga cuando falta o ha cambiado.

    Devuelve {'estado', 'nuevos', 'actualizados', 'fallos', 'descartados', 'url'}:
    estado es 'al_dia', 'descargados', 'sin_conexion', 'error' o 'desactivado';
    nuevos y actualizados, los nombres de los modelos (no los sin radar) que han bajado.
    Nunca lanza una excepción: la app sigue con los modelos que haya."""
    url = url_manifiesto() if url is None else url
    res = {'estado': 'desactivado', 'nuevos': [], 'actualizados': [], 'fallos': [],
           'descartados': [], 'url': url}
    if not url:
        return res
    try:
        lista = leer_manifiesto(url, timeout)
    except ValueError as e:
        res.update(estado='error', fallos=[str(e)])
        return res
    except urllib.error.HTTPError as e:          # hay conexión, pero no está la lista
        res.update(estado='error',
                   fallos=[f'no se encuentra la lista de modelos, error {e.code}'])
        return res
    except Exception as e:                       # noqa: BLE001  (red, DNS, cortafuegos…)
        res.update(estado='sin_conexion', fallos=[f'{type(e).__name__}: {e}'])
        return res
    res['descartados'] = lista['descartados']
    destino = Path(destino)
    incluidos = Path(incluidos) if incluidos else None
    for m in lista['modelos']:
        for f, sub in ((m, ''), (m.get('sin_radar'), CARPETA_SIN_RADAR)):
            if not f:
                continue
            copia = destino / sub / f['archivo']
            de_la_app = incluidos / sub / f['archivo'] if incluidos else None
            try:
                if de_la_app is not None and _igual(de_la_app, f['sha256']):
                    if copia.exists() and not _igual(copia, f['sha256']):
                        copia.unlink()
                    continue
                if _igual(copia, f['sha256']):
                    continue
                habia = copia.exists() or (de_la_app is not None and de_la_app.exists())
                descargar(f, destino / sub, timeout)
                if not sub:
                    res['actualizados' if habia else 'nuevos'].append(f['archivo'])
            except Exception as e:               # noqa: BLE001
                res['fallos'].append(f'{type(e).__name__}: {e}' if not isinstance(
                    e, ValueError) else str(e))
    if res['nuevos'] or res['actualizados']:
        res['estado'] = 'descargados'
    elif res['fallos']:
        res['estado'] = 'error'
    else:
        res['estado'] = 'al_dia'
    return res


def entrada(ruta, url=None, nombre='', fecha=''):
    """La entrada de modelos.json de un fichero (para publicar_modelo.py)."""
    ruta = Path(ruta)
    d = {'archivo': ruta.name}
    if url and url != ruta.name:
        d['url'] = url
    if nombre:
        d['nombre'] = nombre
    if fecha:
        d['fecha'] = fecha
    d.update(sha256=huella(ruta), bytes=ruta.stat().st_size)
    return d
