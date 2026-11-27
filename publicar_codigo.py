"""Publica el código de la app para que los programas instalados se actualicen solos.

Al abrirse, el programa instalado lee `codigo.json` en GitHub y, si el código publicado es
otro, lo descarga, lo comprueba y arranca con él (actualizador.py). Este script escribe esa
lista con el commit actual y la huella de cada fichero; después hay que subirla.

Antes de publicar:
  1. Pasa las pruebas (pytest y la de pantalla).
  2. Sube la versión en CITATION.cff (por ejemplo, version: 1.0.1): es la que verá la app.
  3. Haz commit y push del código.

    py publicar_codigo.py
    py publicar_codigo.py --instalador-minimo 2   # el código necesita el instalador nuevo
    py publicar_codigo.py --comprobar             # ¿la lista coincide con lo publicado?

Luego:
    git add codigo.json
    git commit -m "Publicar el código 1.0.1"
    git push

Para dar marcha atrás: vuelve a publicar desde el commit bueno (o con la versión del
instalador) y sube codigo.json otra vez.
"""
import argparse
import ast
import datetime
import json
import re
import subprocess
import sys
from pathlib import Path

import actualizador as act

REPO = Path(__file__).resolve().parent
LISTA = REPO / act.MANIFIESTO
NOTA = ('Código publicado que los programas instalados descargan solos al abrirse. No se '
        'edita a mano: usar publicar_codigo.py.')
PYTHON_INSTALADOR = (3, 12)          # el Python con el que se compila el instalador


def git(*args, binario=False):
    r = subprocess.run(['git', *args], cwd=REPO, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode('utf-8', 'replace').strip()
                           or f'git {" ".join(args)} ha fallado')
    return r.stdout if binario else r.stdout.decode('utf-8', 'replace')


def existe(commit, ruta):
    try:
        git('cat-file', '-e', f'{commit}:{ruta}')
        return True
    except RuntimeError:
        return False


def rutas_publicadas(commit):
    """Los ficheros que se publican, tal como están en el commit."""
    rutas = [r for r in (*act.MODULOS_APP, *act.DOCS_APP, *act.DATOS_APP)
             if existe(commit, r)]
    for sub in ('plantilla', 'imagenes'):
        rutas += [r for r in git('ls-tree', '-r', '--name-only', commit, f'{sub}/')
                  .splitlines() if r and not Path(r).name.startswith('.')]
    return rutas


def version_de(datos):
    m = re.search(r'^version:\s*"?([^"\s#]+)', datos.decode('utf-8', 'replace'), re.M)
    return m.group(1) if m else ''


def leer_lista():
    if not LISTA.exists():
        return None
    return json.loads(LISTA.read_text(encoding='utf-8'))


def construir_lista(commit, instalador_minimo=1):
    """La lista de `commit` (sin escribirla): {formato, version, commit, …, ficheros}.
    Lanza ValueError con los problemas si algo no se puede publicar."""
    commit = git('rev-parse', commit).strip()
    rutas = rutas_publicadas(commit)
    if 'app.py' not in rutas:
        raise ValueError('no se encuentra app.py en el commit: ¿estás en el repositorio?')
    ficheros, problemas = [], []
    for ruta in rutas:
        datos = git('cat-file', 'blob', f'{commit}:{ruta}', binario=True)
        if len(datos) > act.MAX_BYTES:
            problemas.append(f'{ruta}: pesa demasiado ({len(datos)} bytes)')
            continue
        if ruta.endswith('.py'):
            try:
                ast.parse(datos.decode('utf-8'), filename=ruta,
                          feature_version=PYTHON_INSTALADOR)
            except SyntaxError as e:
                problemas.append(f'{ruta}: no es Python {PYTHON_INSTALADOR[0]}.'
                                 f'{PYTHON_INSTALADOR[1]} válido (línea {e.lineno}: {e.msg})')
        sha, sha_lf = act.huellas_de(datos, ruta)
        d = {'ruta': ruta, 'sha256': sha, 'bytes': len(datos)}
        if sha_lf != sha:
            d['sha256_lf'] = sha_lf
        ficheros.append(d)
    if problemas:
        raise ValueError('No se publica:\n- ' + '\n- '.join(problemas))
    version = version_de(git('cat-file', 'blob', f'{commit}:CITATION.cff', binario=True))
    lista = {'formato': act.FORMATO, 'nota': NOTA, 'version': version, 'commit': commit,
             'fecha': datetime.date.today().isoformat(),
             'instalador_minimo': instalador_minimo, 'ficheros': ficheros}
    # Que la lista se entienda igual que la entenderá el programa instalado
    act.interpretar(lista, act.URL_POR_DEFECTO)
    return lista


def publicar(instalador_minimo=None):
    commit = git('rev-parse', 'HEAD').strip()
    rutas = rutas_publicadas(commit)
    sin_commit = git('status', '--porcelain', '--', *rutas).strip() if rutas else ''
    if sin_commit:
        sys.exit('Hay cambios sin commit en ficheros que se publican; haz commit (y push) '
                 'primero:\n' + sin_commit)
    if 'origin/main' not in git('branch', '-r', '--contains', commit):
        sys.exit('El commit actual no está en GitHub (origin/main): haz push primero. Los '
                 'programas instalados descargan el código de GitHub, no de aquí.')
    anterior = leer_lista() or {}
    if instalador_minimo is None:
        instalador_minimo = int(anterior.get('instalador_minimo') or 1)
    try:
        lista = construir_lista(commit, instalador_minimo)
    except ValueError as e:
        sys.exit(str(e))
    version = lista['version']
    if anterior.get('version') == version and anterior.get('commit') != commit:
        print(f'Aviso: la versión sigue siendo {version}. Los programas instalados se '
              'actualizarán igual, pero la app seguirá diciendo esa versión. Si quieres que '
              'se note, cámbiala en CITATION.cff, haz commit y push, y vuelve a publicar.')
    if (act.tupla_version(version) and act.tupla_version(anterior.get('version'))
            and act.tupla_version(version) < act.tupla_version(anterior.get('version'))):
        print(f'Aviso: {version} es anterior a la publicada ({anterior.get("version")}). '
              'Vale para dar marcha atrás, pero los instaladores con un código más nuevo '
              'que esta versión no la descargarán.')
    LISTA.write_text(json.dumps(lista, ensure_ascii=False, indent=2) + '\n',
                     encoding='utf-8', newline='\n')
    print(f'Escrita {LISTA.name}: versión {version}, commit {commit[:7]}, '
          f'{len(lista["ficheros"])} ficheros, instalador mínimo {instalador_minimo}.')
    print('Ahora súbela:\n'
          f'    git add {LISTA.name}\n'
          f'    git commit -m "Publicar el código {version}"\n'
          '    git push')


def comprobar():
    """Que cada fichero de la lista esté en su commit con la misma huella."""
    datos = leer_lista()
    if not datos:
        sys.exit(f'No hay {LISTA.name}.')
    lista = act.interpretar(datos, act.URL_POR_DEFECTO)
    mal = []
    for f in lista['ficheros']:
        try:
            blob = git('cat-file', 'blob', f'{lista["commit"]}:{f["ruta"]}', binario=True)
        except RuntimeError:
            mal.append(f'{f["ruta"]}: no está en el commit {lista["commit"][:7]}')
            continue
        if act.huella_bytes(blob) != f['sha256']:
            mal.append(f'{f["ruta"]}: la huella no coincide')
    if mal:
        sys.exit('La lista no coincide con lo publicado:\n- ' + '\n- '.join(mal))
    print(f'{LISTA.name} está bien: versión {lista["version"]}, commit '
          f'{lista["commit"][:7]}, {len(lista["ficheros"])} ficheros.')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--instalador-minimo', type=int, default=None,
                   help='nivel de instalador que necesita este código (actualizador.'
                        'INSTALADOR); por defecto, el de la lista anterior')
    p.add_argument('--comprobar', action='store_true',
                   help='solo comprobar la lista que hay')
    a = p.parse_args(argv)
    try:
        if a.comprobar:
            comprobar()
        else:
            publicar(a.instalador_minimo)
    except RuntimeError as e:
        sys.exit(f'Error de git: {e}')


if __name__ == '__main__':
    main()
