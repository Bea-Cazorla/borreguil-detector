"""El instalador de Mac (instalador/macos/instalar.command), como los de MonitorEO y eHabitat+.

27-11-2026: en Mac, en lugar del ejecutable de PyInstaller, un instalador que crea con
micromamba el entorno de environment.yml en ~/Borreguil y pone la app «Borreguil» en el
Escritorio. La instalación de verdad la prueba la compilación (trabajo «macos» de
build-standalone.yml); aquí, lo que se puede comprobar sin instalar nada.
"""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP))

INSTALADOR = APP / 'instalador' / 'macos' / 'instalar.command'
WORKFLOW = APP / '.github' / 'workflows' / 'build-standalone.yml'


def _nombre(n):
    return n.strip().lower().replace('_', '-')


def librerias_requirements():
    """{nombre: condición} de requirements.txt (lo que usa la compilación)."""
    libs = {}
    for linea in (APP / 'requirements.txt').read_text(encoding='utf-8').splitlines():
        linea = linea.split('#', 1)[0].strip()
        if not linea:
            continue
        m = re.match(r'^([A-Za-z0-9_.-]+)\s*(.*)$', linea)
        libs[_nombre(m.group(1))] = m.group(2).replace(' ', '')
    return libs


def librerias_environment():
    """{nombre: condición} de environment.yml (conda y pip), sin python ni pip."""
    libs, dentro = {}, False
    for linea in (APP / 'environment.yml').read_text(encoding='utf-8').splitlines():
        sin_nota = linea.split('#', 1)[0].rstrip()
        if not sin_nota.strip():
            continue
        if not sin_nota.startswith(' '):
            dentro = sin_nota.startswith('dependencies:')
            continue
        m = re.match(r'^\s*-\s+([A-Za-z0-9_.-]+)\s*([^:]*)$', sin_nota)
        if dentro and m and _nombre(m.group(1)) not in ('python', 'pip'):
            cond = m.group(2).replace(' ', '')
            # conda dice «=1.9.0» lo que pip dice «==1.9.0»
            libs[_nombre(m.group(1))] = '==' + cond[1:] if re.match(r'^=[^=]', cond) else cond
    return libs


def test_mismas_librerias_que_requirements():
    req, env = librerias_requirements(), librerias_environment()
    assert req, 'requirements.txt sin librerías'
    assert env == req, ('environment.yml y requirements.txt deben llevar las mismas librerías '
                        f'con las mismas versiones:\n  solo en requirements: '
                        f'{sorted(set(req) - set(env))}\n  solo en environment: '
                        f'{sorted(set(env) - set(req))}\n  distintas: '
                        f'{sorted(k for k in set(req) & set(env) if req[k] != env[k])}')
    assert env['scikit-learn'].startswith('=='), 'la versión de scikit-learn tiene que ser exacta'


def test_python_y_pywebview_en_el_entorno():
    texto = (APP / 'environment.yml').read_text(encoding='utf-8')
    assert re.search(r'^\s*-\s+python=3\.12\s*$', texto, re.M), 'Python 3.12, como la compilación'
    assert re.search(r'^\s*-\s+pip:\s*$', texto, re.M)
    # en conda-forge no está pywebview para Mac: va por pip
    pip = texto.split('- pip:', 1)[1]
    assert 'pywebview' in pip
    assert 'conda-forge' in texto and 'nodefaults' in texto


# Nombre del módulo de cada librería que no se importa con su propio nombre
MODULO = {'scikit-learn': 'sklearn', 'scikit-image': 'skimage', 'pillow': 'PIL',
          'earthengine-api': 'ee', 'python-docx': 'docx', 'pywebview': 'webview'}


def test_el_instalador_comprueba_todas_las_librerias():
    """El instalador decide si el entorno vale importando las librerías de environment.yml:
    si falta una en su lista, un entorno sin ella pasaría por bueno."""
    texto = INSTALADOR.read_text(encoding='utf-8')
    modulos = set()
    for m in re.finditer(r'^\s*MODULOS="(?:\$MODULOS,)?([^"]+)"', texto, re.M):
        modulos |= set(m.group(1).split(','))
    assert modulos
    faltan = [lib for lib in librerias_environment()
              if MODULO.get(lib, lib.replace('-', '_')) not in modulos]
    assert not faltan, f'el instalador no comprueba: {faltan}'
    # y se reinstala si la lista de librerías ha cambiado
    assert 'misma_lista' in texto and 'cp "$NEW/environment.yml" "$LISTA_ENV"' in texto


def test_instalador_ejecutable_y_sin_errores_de_sintaxis():
    assert os.access(INSTALADOR, os.X_OK), 'el instalador tiene que tener permiso de ejecución'
    assert INSTALADOR.read_bytes().startswith(b'#!/bin/bash\n')
    assert b'\r\n' not in INSTALADOR.read_bytes(), 'saltos de línea de Windows en el instalador'
    bash = shutil.which('bash')
    if bash is None:
        pytest.skip('sin bash')
    for shell in {bash, '/bin/bash'} if os.path.exists('/bin/bash') else {bash}:
        r = subprocess.run([shell, '-n', str(INSTALADOR)], capture_output=True, text=True)
        assert r.returncode == 0, (shell, r.stderr)
    texto = INSTALADOR.read_text(encoding='utf-8')
    # una descarga cortada no ejecuta nada: todo dentro de main, que se llama al final
    assert texto.rstrip().endswith('main "$@"')


def _linea_sed(clave):
    texto = INSTALADOR.read_text(encoding='utf-8')
    m = re.search(rf'^\s*{clave}=\$\(printf .*?\| (sed -n \'[^\']+\')', texto, re.M)
    assert m, f'no se encuentra cómo lee {clave} el instalador'
    return m.group(1)


@pytest.mark.skipif(shutil.which('bash') is None or shutil.which('sed') is None,
                    reason='sin bash o sed')
def test_lee_el_commit_y_la_version_de_codigo_json():
    """El instalador lee codigo.json con sed (en un Mac recién estrenado no hay Python):
    tiene que dar lo mismo que la lista publicada."""
    import json
    lista = json.loads((APP / 'codigo.json').read_text(encoding='utf-8'))
    for clave, campo in (('COMMIT', 'commit'), ('VERSION', 'version')):
        orden = f'cat {str(APP / "codigo.json")!r} | {_linea_sed(clave)} | head -n 1'
        r = subprocess.run(['bash', '-c', orden], capture_output=True, text=True)
        assert r.stdout.strip() == lista[campo], (clave, r.stdout, r.stderr)


def test_la_app_del_escritorio():
    texto = INSTALADOR.read_text(encoding='utf-8')
    # se abre con el Python del entorno, sin los paquetes de la cuenta del usuario
    assert '"$ENV/bin/python" -s "$CODE/launcher.py" "$@"' in texto
    assert 'PYTHONNOUSERSITE=1' in texto
    # los datos, en la carpeta de la instalación; lo que escribe, a salida.log
    assert 'BORREGUIL_DATA_DIR="$DIR"' in texto and 'BORREGUIL_SALIDA="$SALIDA"' in texto
    assert '<key>CFBundleExecutable</key><string>borreguil</string>' in texto
    # el código, el publicado (codigo.json), no el de main
    assert '/archive/$COMMIT.zip' in texto and '$RAW/codigo.json' in texto
    # micromamba comprobado con su huella
    assert 'shasum -a 256' in texto


def test_pagina_de_error_dice_donde_esta_el_motivo(monkeypatch):
    import launcher
    monkeypatch.delenv('BORREGUIL_SALIDA', raising=False)
    assert 'ventana de consola' in launcher.no_arranca()
    monkeypatch.setenv('BORREGUIL_SALIDA', '/Users/ana/Borreguil/salida<1>.log')
    pagina = launcher.no_arranca()
    assert '/Users/ana/Borreguil/salida&lt;1&gt;.log' in pagina
    assert 'ventana de consola' not in pagina


def test_la_release_lleva_el_instalador_de_mac():
    texto = WORKFLOW.read_text(encoding='utf-8')
    assert 'Borreguil-macos.zip' in texto
    assert '"Instalar Borreguil.command"' in texto
    assert 'needs: [build, macos]' in texto
    # el comando de la Release es el de este repositorio y este instalador
    assert ('raw.githubusercontent.com/monitoreo-obsnev/borreguil-detector/main/'
            'instalador/macos/instalar.command') in texto
    # ya no se compila Mac con PyInstaller
    assert 'Borreguil-macos-arm64' not in texto and 'pyproj/.libs' not in texto
