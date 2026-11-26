"""Actualización del código de la app al abrirla, sin reinstalarla.

25-11-2026, la autora: «haz el lanzador para que se actualice sola» (lo pidieron Javi y
Domingo para las aulas, como eHabitat+ y MonitorEO).

Cómo funciona
-------------
El instalador trae Python, las librerías y una copia del código de la app. Al abrirse,
el lanzador (launcher.py) lee en el repositorio `codigo.json`: la lista de los ficheros
del código publicado, con el commit del que salen y la huella (SHA-256) de cada uno.

- Si el código publicado es el mismo que ya se usa, no se descarga nada.
- Si es otro, se prepara una carpeta nueva en la del usuario (~/Borreguil/codigo/…):
  lo que no ha cambiado se copia de lo que ya hay y solo se descarga lo nuevo,
  comprobando cada fichero con su huella. Después se pasa el autodiagnóstico del
  ejecutable a esa carpeta, en otro proceso: si algo falla (una librería que el
  instalador no trae, un error al importar), se descarta y se sigue con el código de
  antes. Si pasa, la app arranca ya con el código nuevo.
- Sin conexión, o si la lista pide un instalador más nuevo, se usa el código que haya.
- Si la lista vuelve al código que trae el instalador, se deja la copia descargada y se
  vuelve al del instalador: así se puede dar marcha atrás publicando otra vez.

Lo que se actualiza es el código y la documentación. Los modelos van por su lado
(modelos_remotos.py) y Python y las librerías solo cambian con un instalador nuevo.

Para publicar: `py publicar_codigo.py` (ver PUBLICAR.md).

Seguridad: el código descargado se ejecuta, así que solo se baja por HTTPS de la
dirección configurada (por defecto, el repositorio del proyecto), del mismo sitio que la
lista, y cada fichero tiene que dar la huella de la lista. La huella no protege de quien
pueda subir ficheros al repositorio: eso depende de quién tenga permiso para escribir en
él (igual que con los modelos).

Todo va por HTTPS (puerto 443) a raw.githubusercontent.com. Para no buscar
actualizaciones (aulas sin internet, pruebas): BORREGUIL_CODIGO_URL vacía.
"""
import datetime
import hashlib
import importlib.abc
import importlib.util
import json
import os
import re
import shutil
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO = 'https://raw.githubusercontent.com/monitoreo-obsnev/borreguil-detector/'
MANIFIESTO = 'codigo.json'
URL_POR_DEFECTO = REPO + 'main/' + MANIFIESTO
VARIABLE_URL = 'BORREGUIL_CODIGO_URL'
CARPETA = 'codigo'                     # dentro de la carpeta de datos del usuario
ACTIVO = 'activo.json'                 # qué versión descargada se usa
COPIA_LISTA = '.version.json'          # la lista, guardada dentro de cada versión
TIEMPO_S = 5                           # espera máxima de cada conexión
TIEMPO_FICHERO_S = 30
MAX_BYTES = 20 * 1024 * 1024           # ningún fichero del código pesa tanto
FORMATO = 1                            # versión del formato de codigo.json
DESCARGAS_RELEASE = ('https://github.com/monitoreo-obsnev/borreguil-detector/'
                     'releases/latest')

# Nivel de las librerías de ESTE instalador. Se sube cuando el instalador cambia lo que
# trae (una librería nueva o una versión distinta que el código necesite); el código que
# lo necesite se publica con «instalador_minimo» igual a ese número, y los instaladores
# antiguos no lo descargan: avisan de que hay que instalar el nuevo.
INSTALADOR = 1

# Los módulos de la app: viajan como ficheros junto al ejecutable (los usa borreguil.spec)
# y son los que se publican y se actualizan.
MODULOS_APP = ('app.py', 'borreguil_pipeline.py', 'point_profile.py', 'i18n.py',
               'gee_backend.py', 'random_points.py', 'study_area.py', 'mapeo_rf.py',
               'mapeo_raster.py', 'mapeo_tif.py', 'mapeo_ambiente.py',
               'modelos_remotos.py', 'registros.py')
# Documentación y datos pequeños que cambian con el código. Los modelos (.joblib) y la
# tabla de entrenamiento no: van con su modelo.
DOCS_APP = ('MANUAL.md', 'TUTORIAL.md', 'CITATION.cff', 'LICENSE')
DATOS_APP = ('lagunas_sierra_nevada.kml', 'cauces_sierra_nevada.geojson')
SUBCARPETAS_APP = ('plantilla', 'sin_radar')
# Lo que no se deja cambiar desde fuera: el lanzador y este módulo viajan dentro del
# ejecutable y solo cambian con un instalador nuevo.
PROPIOS = ('launcher', 'actualizador')

_TEXTO = {'.py', '.md', '.cff', '.json', '.kml', '.geojson', '.csv', '.txt', '.toml'}
_RUTA_OK = re.compile(r'^(?:[A-Za-z0-9_][A-Za-z0-9_-]*/)?[A-Za-z0-9_][A-Za-z0-9_.-]*$')
_SHA_OK = re.compile(r'^[0-9a-f]{64}$')
_COMMIT_OK = re.compile(r'^[0-9a-f]{7,40}$')
_SUFIJOS_APP = {'.py', '.joblib', '.kml', '.geojson', '.csv', '.md', '.cff'}


# ------------------------------------------------------------------ huellas
def huella_bytes(datos):
    return hashlib.sha256(datos).hexdigest()


def huella(ruta):
    """SHA-256 de un fichero, tal cual."""
    h = hashlib.sha256()
    with open(ruta, 'rb') as f:
        for trozo in iter(lambda: f.read(1 << 20), b''):
            h.update(trozo)
    return h.hexdigest()


def es_texto(ruta):
    return Path(str(ruta)).suffix.lower() in _TEXTO or Path(str(ruta)).name == 'LICENSE'


def huella_lf(ruta):
    """La huella para comparar contenidos: en los ficheros de texto, con los saltos de
    línea de Windows (CRLF) como LF. Git los guarda con LF, pero al compilar en Windows
    pueden quedar con CRLF dentro del instalador sin que cambie nada más."""
    if not es_texto(ruta):
        return huella(ruta)
    return huella_bytes(Path(ruta).read_bytes().replace(b'\r\n', b'\n'))


def huellas_de(datos, ruta):
    """(sha256, sha256_lf) de unos bytes que serán el fichero `ruta`."""
    lf = datos.replace(b'\r\n', b'\n') if es_texto(ruta) else datos
    return huella_bytes(datos), huella_bytes(lf)


# --------------------------------------------------------------- la lista
def url_manifiesto():
    """La dirección de codigo.json: la de BORREGUIL_CODIGO_URL si está definida (vacía:
    no buscar actualizaciones) o la del repositorio del proyecto."""
    v = os.environ.get(VARIABLE_URL)
    return URL_POR_DEFECTO if v is None else v.strip()


def _abrir(url, timeout):
    req = urllib.request.Request(url, headers={'User-Agent': 'borreguil-detector'})
    return urllib.request.urlopen(req, timeout=timeout)


def _mismo_sitio(url, base):
    a, b = urllib.parse.urlsplit(url), urllib.parse.urlsplit(base)
    return a.scheme == b.scheme and a.netloc == b.netloc


def interpretar(datos, url):
    """La lista comprobada: {version, commit, fecha, instalador_minimo, ficheros}, cada
    fichero {ruta, sha256, sha256_lf, bytes, url}. Los ficheros se bajan del commit
    publicado, no de la rama: así lo que se suba después no llega hasta que se publique.
    Lanza ValueError si algo no tiene el formato esperado (entonces no se usa nada)."""
    if not isinstance(datos, dict):
        raise ValueError('la lista del código no tiene el formato esperado')
    formato = datos.get('formato', 1)
    if isinstance(formato, bool) or not isinstance(formato, int) or formato > FORMATO:
        raise ValueError('la lista del código es de un formato más nuevo: hace falta '
                         'el instalador nuevo')
    commit = str(datos.get('commit', '')).lower()
    if not _COMMIT_OK.match(commit):
        raise ValueError('la lista del código no dice de qué commit sale')
    minimo = datos.get('instalador_minimo', 1)
    if isinstance(minimo, bool) or not isinstance(minimo, int) or minimo < 1:
        raise ValueError('instalador_minimo no válido')
    base = str(datos.get('base') or urllib.parse.urljoin(url, f'../{commit}/'))
    if not _mismo_sitio(base, url):
        raise ValueError('los ficheros del código no son del mismo sitio que la lista')
    if not isinstance(datos.get('ficheros'), list) or not datos['ficheros']:
        raise ValueError('la lista del código no trae ficheros')
    ficheros, vistos = [], set()
    for d in datos['ficheros']:
        if not isinstance(d, dict):
            raise ValueError('entrada de la lista que no es un objeto')
        ruta = str(d.get('ruta', ''))
        if not _RUTA_OK.match(ruta) or '..' in ruta.split('/'):
            raise ValueError(f'nombre de fichero no válido: {ruta!r}')
        if ruta in vistos:
            raise ValueError(f'{ruta}: repetido en la lista')
        sha = str(d.get('sha256', '')).lower()
        lf = str(d.get('sha256_lf') or sha).lower()
        if not _SHA_OK.match(sha) or not _SHA_OK.match(lf):
            raise ValueError(f'{ruta}: huella SHA-256 no válida')
        n = d.get('bytes')
        if isinstance(n, bool) or not isinstance(n, int) or not 0 <= n <= MAX_BYTES:
            raise ValueError(f'{ruta}: tamaño no válido')
        u = urllib.parse.urljoin(base, urllib.parse.quote(ruta))
        if not _mismo_sitio(u, url):
            raise ValueError(f'{ruta}: su dirección no es del mismo sitio que la lista')
        vistos.add(ruta)
        ficheros.append({'ruta': ruta, 'sha256': sha, 'sha256_lf': lf, 'bytes': n,
                         'url': u})
    if 'app.py' not in vistos:
        raise ValueError('la lista del código no trae app.py')
    return {'version': str(datos.get('version') or ''), 'commit': commit,
            'fecha': str(datos.get('fecha') or ''), 'instalador_minimo': minimo,
            'ficheros': ficheros}


def leer_manifiesto(url=None, timeout=TIEMPO_S):
    """Lee y comprueba codigo.json. Solo por HTTPS (o un fichero local, en las pruebas).
    Devuelve (lista comprobada, datos tal cual)."""
    url = url_manifiesto() if url is None else url
    if urllib.parse.urlsplit(url).scheme not in ('https', 'file'):
        raise ValueError('la lista del código tiene que ir por HTTPS')
    with _abrir(url, timeout) as r:
        texto = r.read(1_000_001)
    if len(texto) > 1_000_000:
        raise ValueError('la lista del código es demasiado grande')
    try:
        datos = json.loads(texto.decode('utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ValueError(f'la lista del código no se puede leer: {e}') from None
    return interpretar(datos, url), datos


# ------------------------------------------------------- versiones en disco
def carpeta_codigo(datos_dir):
    return Path(datos_dir) / CARPETA


def _igual(ruta, f):
    try:
        return ruta.is_file() and huella_lf(ruta) == f['sha256_lf']
    except OSError:
        return False


def completa(carpeta, lista):
    """True si `carpeta` tiene todos los ficheros de la lista, con su contenido."""
    carpeta = Path(carpeta)
    return all(_igual(carpeta / f['ruta'], f) for f in lista['ficheros'])


def tupla_version(texto):
    """'1.0.2' → (1, 0, 2); () si no se entiende."""
    partes = re.findall(r'\d+', str(texto or '').split('-')[0])
    return tuple(int(p) for p in partes[:4])


def version_instalador(res):
    """La versión del código que trae el instalador (la de su CITATION.cff)."""
    try:
        texto = (Path(res) / 'CITATION.cff').read_text(encoding='utf-8')
    except OSError:
        return ''
    m = re.search(r'^version:\s*"?([^"\s#]+)', texto, re.M)
    return m.group(1) if m else ''


def mas_vieja(version, res):
    """True si `version` es anterior a la del código del instalador: pasa al instalar un
    instalador nuevo cuando todavía no se ha publicado su código. Entonces manda el
    instalador. Sin versiones que comparar, False."""
    a, b = tupla_version(version), tupla_version(version_instalador(res))
    return bool(a and b and a < b)


def version_activa(datos_dir, res=None):
    """La versión descargada que se usa: {'carpeta', 'version', 'commit', 'lista'}, o
    None si no hay ninguna, está incompleta, necesita otro instalador o es más vieja que
    el código del instalador `res` (entonces se usa el del instalador)."""
    base = carpeta_codigo(datos_dir)
    try:
        activo = json.loads((base / ACTIVO).read_text(encoding='utf-8'))
        carpeta = base / str(activo['carpeta'])
        if carpeta.parent != base or not carpeta.is_dir():
            return None
        guardada = json.loads((carpeta / COPIA_LISTA).read_text(encoding='utf-8'))
        lista = interpretar(guardada['datos'], guardada['url'])
    except Exception:                                   # noqa: BLE001
        return None
    if lista['instalador_minimo'] > INSTALADOR or not completa(carpeta, lista):
        return None
    if res is not None and mas_vieja(lista['version'], res):
        return None
    return {'carpeta': carpeta, 'version': lista['version'], 'commit': lista['commit'],
            'lista': lista}


def carpeta_a_usar(res, datos_dir):
    """(carpeta con el código que hay que ejecutar, versión descargada o None)."""
    act = version_activa(datos_dir, res)
    return (act['carpeta'], act) if act else (Path(res), None)


def ficheros_de_la_app(res):
    """Rutas relativas de los ficheros de la app que trae el instalador (no las
    librerías): los de arriba con sus extensiones y los de sus subcarpetas."""
    res = Path(res)
    rutas = []
    for p in sorted(res.iterdir()):
        if p.is_file() and not p.name.startswith('.') and (
                p.suffix.lower() in _SUFIJOS_APP or p.name == 'LICENSE'):
            rutas.append(p.name)
    for sub in SUBCARPETAS_APP:
        d = res / sub
        if d.is_dir():
            rutas += [f'{sub}/{p.name}' for p in sorted(d.iterdir())
                      if p.is_file() and not p.name.startswith('.')]
    return rutas


def _copiar(origen, destino):
    destino.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(origen, destino)


def descargar(f, destino, timeout=TIEMPO_FICHERO_S):
    """Descarga un fichero de la lista a destino/ruta comprobando su tamaño y su huella."""
    final = Path(destino) / f['ruta']
    final.parent.mkdir(parents=True, exist_ok=True)
    h, n, partes = hashlib.sha256(), 0, []
    with _abrir(f['url'], timeout) as r:
        while True:
            trozo = r.read(1 << 20)
            if not trozo:
                break
            n += len(trozo)
            if n > f['bytes']:
                raise ValueError(f"{f['ruta']}: más grande de lo que dice la lista")
            h.update(trozo)
            partes.append(trozo)
    if n != f['bytes'] or h.hexdigest() != f['sha256']:
        raise ValueError(f"{f['ruta']}: la huella no coincide con la de la lista "
                         '(descarga incompleta o fichero distinto)')
    final.write_bytes(b''.join(partes))
    return final


def preparar(lista, datos_url, res, destino, anterior=None, timeout=TIEMPO_FICHERO_S):
    """Monta en `destino` la versión de la lista: cada fichero se copia de la versión
    anterior o del instalador si ya está igual, y si no se descarga. Después se copian
    del instalador los demás ficheros de la app (modelos, tabla de entrenamiento…), que
    la app busca a su lado. Devuelve cuántos ficheros se han descargado."""
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    fuentes = [Path(anterior)] if anterior else []
    fuentes.append(Path(res))
    bajados = 0
    for f in lista['ficheros']:
        hecho = False
        for fu in fuentes:
            if _igual(fu / f['ruta'], f):
                _copiar(fu / f['ruta'], destino / f['ruta'])
                hecho = True
                break
        if not hecho:
            descargar(f, destino, timeout)
            bajados += 1
    publicados = {f['ruta'] for f in lista['ficheros']}
    for ruta in ficheros_de_la_app(res):
        if ruta not in publicados:
            _copiar(Path(res) / ruta, destino / ruta)
    (destino / COPIA_LISTA).write_text(
        json.dumps({'url': datos_url['url'], 'datos': datos_url['datos']},
                   ensure_ascii=False, indent=1), encoding='utf-8')
    if not completa(destino, lista):
        raise ValueError('la versión descargada no está completa')
    return bajados


def _escribir_json(ruta, datos):
    fd, tmp = tempfile.mkstemp(prefix='.activo.', suffix='.json', dir=ruta.parent)
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        json.dump(datos, f, ensure_ascii=False, indent=1)
    os.replace(tmp, ruta)


def _limpiar(base, quedan):
    """Quita las versiones viejas y los restos de preparaciones cortadas."""
    for p in base.iterdir() if base.is_dir() else []:
        if p.is_dir() and p.name not in quedan:
            shutil.rmtree(p, ignore_errors=True)


def actualizar(res, datos_dir, url=None, timeout=TIEMPO_S, probar=None, aviso=print):
    """Pone al día el código: ver la explicación de arriba. `probar(carpeta)` pasa el
    autodiagnóstico a una carpeta y devuelve (ok, texto). `aviso` recibe lo que va
    pasando, para la consola.

    Devuelve {'estado', 'version', 'detalle'}; estado es 'desactivado', 'sin_conexion',
    'error', 'al_dia', 'instalador' (se usa el código del instalador, que es el
    publicado), 'actualizado', 'necesita_instalador' o 'no_pasa_prueba'. Nunca lanza
    una excepción: la app arranca con el código que haya."""
    url = url_manifiesto() if url is None else url
    out = {'estado': 'desactivado', 'version': '', 'detalle': ''}
    if not url:
        return out
    try:
        lista, datos = leer_manifiesto(url, timeout)
    except ValueError as e:
        out.update(estado='error', detalle=str(e))
        return out
    except urllib.error.HTTPError as e:
        out.update(estado='error', detalle=f'no se encuentra la lista del código ({e.code})')
        return out
    except Exception as e:                              # noqa: BLE001  (red, DNS…)
        out.update(estado='sin_conexion', detalle=f'{type(e).__name__}: {e}')
        return out
    out['version'] = lista['version']
    try:
        if lista['instalador_minimo'] > INSTALADOR:
            out.update(estado='necesita_instalador',
                       detalle=f'la versión {lista["version"]} necesita el instalador nuevo: '
                               f'{DESCARGAS_RELEASE}')
            return out
        base = carpeta_codigo(datos_dir)
        act = version_activa(datos_dir, res)
        if act and completa(act['carpeta'], lista):
            out['estado'] = 'al_dia'
            return out
        if completa(res, lista) or mas_vieja(lista['version'], res):
            # El instalador ya trae este código, o uno más nuevo: se vuelve a él.
            if (base / ACTIVO).exists():
                (base / ACTIVO).unlink()
            _limpiar(base, set())
            out.update(estado='instalador', version=version_instalador(res) or out['version'])
            return out
        aviso(f'  Actualizando la app a la versión {lista["version"] or lista["commit"][:7]}…')
        nombre = f'{lista["commit"][:12]}'
        base.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix=f'.preparando-{nombre}-', dir=base))
        try:
            bajados = preparar(lista, {'url': url, 'datos': datos}, res, tmp,
                               anterior=act['carpeta'] if act else None)
            if probar is not None:
                aviso('  Comprobando la versión nueva (puede tardar un minuto)…')
                ok, texto = probar(tmp)
                if not ok:
                    out.update(estado='no_pasa_prueba', detalle=texto)
                    return out
            final = base / nombre
            if final.exists():
                shutil.rmtree(final, ignore_errors=True)
            os.replace(tmp, final)
        finally:
            if tmp.exists():
                shutil.rmtree(tmp, ignore_errors=True)
        _escribir_json(base / ACTIVO, {
            'carpeta': nombre, 'version': lista['version'], 'commit': lista['commit'],
            'fecha': lista['fecha'],
            'instalado': datetime.datetime.now().isoformat(timespec='seconds')})
        _limpiar(base, {nombre, act['carpeta'].name} if act else {nombre})
        out.update(estado='actualizado',
                   detalle=f'{bajados} fichero{"" if bajados == 1 else "s"} descargado'
                           f'{"" if bajados == 1 else "s"}')
        return out
    except Exception as e:                              # noqa: BLE001
        out.update(estado='error', detalle=f'{type(e).__name__}: {e}')
        return out


def texto_estado(r):
    """Una línea para la consola del lanzador."""
    e, v = r.get('estado'), r.get('version') or ''
    if e == 'actualizado':
        return f'  App actualizada a la versión {v} ({r.get("detalle")}).'
    if e == 'al_dia':
        return f'  App al día (versión {v}).'
    if e == 'instalador':
        return f'  App al día (versión {v}, la del instalador).'
    if e == 'sin_conexion':
        return '  Sin conexión: se usa el código que ya había.'
    if e == 'necesita_instalador':
        return (f'  Hay una versión nueva ({v}) que necesita el instalador nuevo. '
                f'Descárgalo de {DESCARGAS_RELEASE}')
    if e == 'no_pasa_prueba':
        return (f'  La versión nueva ({v}) no ha pasado la comprobación: se sigue con la '
                'de antes.')
    if e == 'error':
        return f'  No se ha podido buscar la versión nueva: {r.get("detalle")}'
    return ''


# ------------------------------------------- usar el código de una carpeta
class _BuscadorDeCodigo(importlib.abc.MetaPathFinder):
    """Sirve los módulos de la app desde la carpeta elegida, antes que nada más. Dentro
    del ejecutable, algunos módulos de la app quedan también compilados dentro de él
    (PyInstaller los ve al analizar el lanzador) y, sin esto, se usarían esos aunque la
    carpeta tenga una versión más nueva."""

    def __init__(self, carpeta, nombres):
        self.carpeta, self.nombres = Path(carpeta), set(nombres)

    def find_spec(self, fullname, path=None, target=None):
        if fullname in self.nombres:
            return importlib.util.spec_from_file_location(
                fullname, str(self.carpeta / f'{fullname}.py'))
        return None


def usar_codigo_de(carpeta):
    """Hace que `import borreguil_pipeline`, `import i18n`… carguen los de `carpeta`."""
    carpeta = Path(carpeta).resolve()
    nombres = {p.stem for p in carpeta.glob('*.py')} - set(PROPIOS)
    for n in list(nombres):
        m = sys.modules.get(n)
        origen = getattr(m, '__file__', None) if m else None
        if m is not None and (not origen or Path(origen).resolve().parent != carpeta):
            del sys.modules[n]
    sys.meta_path[:] = [b for b in sys.meta_path if not isinstance(b, _BuscadorDeCodigo)]
    sys.meta_path.insert(0, _BuscadorDeCodigo(carpeta, nombres))
    if str(carpeta) not in sys.path:
        sys.path.insert(0, str(carpeta))
    return nombres
