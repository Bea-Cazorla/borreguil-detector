"""Pruebas de los registros de la app y de «Enviar los registros» (registros.py y app.py),
hechos como en el asistente de MonitorEO: qué se apunta en registro.log (lo que sale por la
consola, lo del análisis, los errores en pantalla con su traza, los mensajes de
Streamlit), que no crece sin fin, qué se quita antes de enseñarlo (quién es: carpeta,
usuario, ordenador, correos, claves, el identificador de la instalación y el proyecto de
Earth Engine), el ruido que se quita, lo que se envía al formulario anónimo (simulado
aquí: el formulario de verdad solo se usa a mano) y los botones de la app.

Se ejecutan con `python tests/test_registros.py` (o con pytest); no necesitan red.
"""
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import urllib.error
import urllib.parse
from pathlib import Path

APP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP))
os.environ.setdefault('BORREGUIL_MODELOS_URL', '')

import registros as rg       # noqa: E402

CASA = os.path.expanduser('~')
TOKEN = 'ya29.' + 'a0Falso' * 6          # secretos de mentira, hechos aquí para que ningún
CLAVE = 'AIza' + 'Falsa' * 8             # buscador de secretos tome esto por una fuga


class _Cambiar:
    """Cambia atributos de un módulo y los devuelve al salir (como monkeypatch)."""

    def __init__(self, modulo, **cambios):
        self.modulo, self.cambios, self.antes = modulo, cambios, {}

    def __enter__(self):
        for k, v in self.cambios.items():
            self.antes[k] = getattr(self.modulo, k)
            setattr(self.modulo, k, v)
        return self

    def __exit__(self, *a):
        for k, v in self.antes.items():
            setattr(self.modulo, k, v)


# ── Lo que se quita: quién es ────────────────────────────────────────────────────

def test_se_quita_quien_es():
    with _Cambiar(rg, _usuario=lambda: 'jperez7', _ordenador=lambda: 'Portatil-de-Juana'):
        texto = '\n'.join([
            f'File "{CASA}/Borreguil/app.py", line 3',
            json.dumps(CASA + '/datos/puntos.gpkg'),
            'C:\\Users\\Otra\\Borreguil\\registro.log y /home/otro/x y /Users/otra/y',
            'Entró jperez7 en Portatil-de-Juana.local con juana.perez+ee@ejemplo.org',
            'monitoreo-public@ee-ugr-default-project.iam.gserviceaccount.com calcula',
            f'Authorization: Bearer {TOKEN}', f'clave {CLAVE}', 'PL_API_KEY=PLAKabcdef0123456789abcdef',
            '{"access_token": "abc123", "project": "ee-jperez7-tesis"}',
            'etiqueta mo-0123456789abcdef del identificador anónimo',
            'proyecto ee-jperez7-tesis en Earth Engine'])
        limpio = rg.ocultar_datos(texto, ocultar=['ee-jperez7-tesis'])
    assert CASA not in limpio and '~/Borreguil/app.py' in limpio
    assert 'Otra' not in limpio and 'otro' not in limpio and '/Users/<usuario>/y' in limpio
    assert 'jperez7' not in limpio and 'Portatil-de-Juana' not in limpio
    assert '<usuario>' in limpio and '<ordenador>' in limpio
    assert 'juana.perez' not in limpio and '<correo>' in limpio
    assert 'monitoreo-public@ee-ugr-default-project.iam.gserviceaccount.com' in limpio
    assert TOKEN not in limpio and CLAVE not in limpio and 'PLAKabcdef' not in limpio
    assert 'abc123' not in limpio
    assert 'mo-0123456789abcdef' not in limpio and 'mo-<instalación>' in limpio
    assert 'ee-jperez7-tesis' not in limpio and '<proyecto>' in limpio


def test_un_usuario_corto_o_comun_no_se_cambia():
    with _Cambiar(rg, _usuario=lambda: 'user', _ordenador=lambda: 'pc'):
        texto = 'the user said it on the pc'
        assert rg.ocultar_datos(texto) == texto


# ── El ruido que se quita ─────────────────────────────────────────────────────────

def test_compactar_quita_el_ruido_y_guarda_los_errores():
    aviso = ['C:\\py\\ee\\data.py:335: UserWarning: Your project has exceeded its quota',
             '  warnings.warn(']
    texto = '\n'.join(
        ['Leyendo Sentinel-2…'] * 5 + aviso + ['punto 1'] + aviso + ['punto 2'] +
        ['Traceback (most recent call last):', '  File "app.py", line 3, in <module>',
         '    x = 1 / 0', 'ZeroDivisionError: division by zero'] +
        ['ZeroDivisionError: division by zero'])
    c = rg.compactar(texto).split('\n')
    assert c[0] == 'Leyendo Sentinel-2… (repetida 5 veces)'
    assert sum('UserWarning' in x for x in c) == 1               # el aviso, una vez
    assert any('(2 veces; solo se ve la última)' in x for x in c)
    assert c.index('punto 1') < [i for i, x in enumerate(c) if 'UserWarning' in x][0]
    assert '    x = 1 / 0' in c and c.count('ZeroDivisionError: division by zero') == 2


def test_compactar_junta_los_reintentos_de_googleapiclient():
    lineas = []
    for i in range(1, 4):
        lineas += [f'[11/26/26 10:0{i}:00] WARNING  Sleeping 1.{i} seconds before retry {i} of 5 '
                   'for request: POST https://earthengine.googleapis.com/v1/projects/x/value:compute, '
                   'after 503']
    c = rg.compactar('\n'.join(lineas))
    assert c.count('\n') == 0 and 'reintentó peticiones 3 veces' in c and '3 tras 503' in c


# ── Recoger ─────────────────────────────────────────────────────────────────────────

def test_recoger_sin_registros():
    with tempfile.TemporaryDirectory() as d:
        t = rg.recoger('1.2.3', donde=d)
    assert t.startswith('Registros de Borreguil · Wet Meadow Finder 1.2.3\n')
    assert 'Sistema: ' in t and 'Todavía no hay registros' in t


def test_recoger_cabe_en_el_formulario_y_trae_lo_mas_nuevo():
    with tempfile.TemporaryDirectory() as d:
        lineas = [f'línea {i} en {CASA}\\Borreguil con acentos ñ' for i in range(20000)]
        Path(d, rg.FICHERO).write_text('\n'.join(lineas) + '\n', encoding='utf-8')
        t = rg.recoger('1.2.3', ocultar=['ee-proyecto-secreto'], donde=d)
    assert rg.tamano(t) <= rg.LOGS_BYTES
    assert 'línea 19999 ' in t and 'línea 0 ' not in t
    assert CASA not in t and '~\\Borreguil' in t
    assert '== registro.log (sus últimas líneas) ==' in t


# ── Enviar ──────────────────────────────────────────────────────────────────────────

class _Respuesta:
    def __init__(self, url):
        self.url = url

    def geturl(self):
        return self.url

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _enviar_y_ver(texto, descripcion='', parte='analisis', final=None, fallo=None):
    vistas = []

    def urlopen(peticion, timeout=None):
        vistas.append(peticion)
        if fallo:
            raise fallo
        return _Respuesta(final or rg.FORMULARIO_RESPUESTA)

    with _Cambiar(rg.urllib.request, urlopen=urlopen):
        rg.enviar(texto, '1.0.3', descripcion, parte)
    p = vistas[0]
    return p, dict(urllib.parse.parse_qsl(p.data.decode('utf-8')))


def test_enviar_rellena_el_formulario():
    p, datos = _enviar_y_ver('Registros…\nTraceback: algo', 'Analizaba 20 puntos')
    assert p.full_url == rg.FORMULARIO_RESPUESTA and p.full_url.endswith('/formResponse')
    c = rg.CAMPOS
    assert datos[c['que']] == 'Report an error / Informar de un error'
    assert datos[c['parte']] == 'Analysis of the points / Análisis de los puntos'
    assert datos[c['version']] == '1.0.3'
    assert datos[c['ordenador']] in ('Windows', 'macOS', 'Linux', "I don't know / No lo sé")
    assert datos[c['descripcion']] == 'Analizaba 20 puntos'
    assert datos[c['consola']] == 'Registros…\nTraceback: algo'


def test_sin_descripcion_y_partes():
    _, datos = _enviar_y_ver('algo', '', 'no-existe')
    assert datos[rg.CAMPOS['descripcion']] == rg.SIN_DESCRIPCION
    assert datos[rg.CAMPOS['parte']] == 'Other / Otra'
    # las opciones son exactamente las del formulario (FB_PUBLIC_LOAD_DATA_, 26-11-2026)
    assert [v[2] for v in rg.PARTES.values()] == [
        'Installation or start / Instalación o arranque',
        'Analysis of the points / Análisis de los puntos', 'Reviewing points / Revisar puntos',
        'Borreguil map (RF model map) / Mapa de borreguiles (Mapa Modelo RF)',
        'Documentation or manual / Documentación o manual', 'Other / Otra']


def test_lo_que_no_cabe_se_acorta():
    largo = 'Registros de Borreguil 1.0.3\nSistema: x\n\n' + '\n'.join(
        f'línea número {i} ñ' for i in range(30000))
    p, datos = _enviar_y_ver(largo, 'qué pasó ' * 3000)
    assert len(p.data) <= rg.BODY_BYTES
    consola = datos[rg.CAMPOS['consola']]
    assert consola.startswith('Registros de Borreguil 1.0.3\nSistema: x\n[…]\n')
    assert consola.rstrip().endswith('línea número 29999 ñ')
    assert datos[rg.CAMPOS['descripcion']].endswith(' […]')
    assert rg.tamano(datos[rg.CAMPOS['descripcion']]) >= rg.DESCRIPCION_BYTES - 10


def test_cuando_no_se_puede_enviar():
    for fallo, dice in ((urllib.error.HTTPError(rg.FORMULARIO_RESPUESTA, 413, 'grande', {}, None),
                         'HTTP 413'),
                        (urllib.error.URLError('sin red'), 'mira la conexión')):
        try:
            _enviar_y_ver('algo', fallo=fallo)
        except RuntimeError as e:
            assert dice in str(e) and 'formulario anónimo' in str(e) and 'issue' in str(e)
        else:
            raise AssertionError('tenía que fallar')
    try:
        _enviar_y_ver('algo', final='https://docs.google.com/forms/d/e/x/closedform')
    except RuntimeError as e:
        assert 'no admite respuestas' in str(e)
    else:
        raise AssertionError('tenía que fallar')
    try:
        rg.enviar('   ')
    except ValueError:
        pass
    else:
        raise AssertionError('tenía que fallar')


# ── Escribir el registro (en otro proceso: cambia sys.stdout y sys.stderr) ─────────

def _en_otro_proceso(codigo, carpeta, **env):
    entorno = dict(os.environ, BORREGUIL_DATA_DIR=str(carpeta), PYTHONIOENCODING='utf-8', **env)
    entorno.pop('BORREGUIL_SIN_REGISTRO', None)
    r = subprocess.run([sys.executable, '-c', textwrap.dedent(codigo)], cwd=str(APP),
                       env=entorno, capture_output=True, text=True, encoding='utf-8', timeout=120)
    assert r.returncode == 0, r.stderr
    return r


def test_lo_que_sale_por_la_consola_va_al_registro():
    with tempfile.TemporaryDirectory() as d:
        r = _en_otro_proceso('''
            import logging, sys, threading
            import registros
            registros.empezar('9.9.9')
            registros.empezar('9.9.9')                 # la segunda vez no hace nada
            print('hola desde print')
            print('aviso por stderr', file=sys.stderr)
            registros.anotar('lo del análisis\\n')
            try:
                1 / 0
            except ZeroDivisionError:
                registros.anotar_error('Error: no se pudo', sys.exc_info())
            h = threading.Thread(target=lambda: print('desde un hilo'))
            h.start(); h.join()
        ''', d)
        texto = Path(d, rg.FICHERO).read_text(encoding='utf-8')
    assert 'hola desde print' in r.stdout and 'aviso por stderr' in r.stderr   # la consola, igual
    assert texto.count('=== Borreguil 9.9.9 · ') == 1
    for x in ('hola desde print', 'aviso por stderr', 'lo del análisis', 'desde un hilo',
              'ERROR en pantalla: Error: no se pudo', 'ZeroDivisionError: division by zero'):
        assert x in texto, x


def test_lo_que_escribe_click_en_bytes_llega_sin_error():
    """click 8.5 escribe bytes cuando la salida no es una terminal (el «Stopping...» de
    Streamlit al pararse): no puede dar error, o el servidor no se cierra por las buenas."""
    with tempfile.TemporaryDirectory() as d:
        r = _en_otro_proceso('''
            import sys
            import registros
            registros.empezar('9.9.9')
            sys.stdout.write(b'en bytes \\xc3\\xb1\\n')
            import click
            click.secho('  Stopping...', fg='blue')
            print('después')
        ''', d)
        texto = Path(d, rg.FICHERO).read_text(encoding='utf-8')
    for x in ('en bytes ñ', 'Stopping...', 'después'):
        assert x in texto and x in r.stdout, x


def test_los_mensajes_de_streamlit_llegan_al_registro():
    with tempfile.TemporaryDirectory() as d:
        _en_otro_proceso('''
            from streamlit import logger as stlogger
            log = stlogger.get_logger('streamlit.prueba_registros')
            import registros
            registros.empezar('9.9.9')
            log.error('Uncaught app execution de prueba')
        ''', d)
        texto = Path(d, rg.FICHERO).read_text(encoding='utf-8')
    assert 'Uncaught app execution de prueba' in texto


def test_el_registro_no_crece_sin_fin():
    with tempfile.TemporaryDirectory() as d:
        _en_otro_proceso('''
            import registros
            registros.MAX_BYTES = 20000
            registros.empezar('9.9.9')
            for i in range(3000):
                print(f'línea {i:05d} ' + 'x' * 40)
        ''', d)
        texto = Path(d, rg.FICHERO).read_text(encoding='utf-8')
    assert len(texto.encode('utf-8')) <= 20000 + 200
    assert 'línea 02999' in texto and 'línea 00000' not in texto


def test_con_salida_por_hilos_la_copia_va_debajo():
    """En la app, sys.stdout reparte por hilos (_SalidaPorHilo): la copia al registro va
    en su salida de siempre, no encima."""
    with tempfile.TemporaryDirectory() as d:
        _en_otro_proceso('''
            import sys, threading, io
            class PorHilo(io.TextIOBase):
                def __init__(self, original):
                    self.original = original
                    self.hilos = threading.local()
                def write(self, s):
                    return self.original.write(s)
            sys.stdout = PorHilo(sys.stdout)
            import registros
            registros.empezar('9.9.9')
            assert isinstance(sys.stdout, PorHilo)
            assert sys.stdout.original._es_copia_al_registro
            print('por la salida de siempre')
        ''', d)
        texto = Path(d, rg.FICHERO).read_text(encoding='utf-8')
    assert 'por la salida de siempre' in texto


# ── En la app ──────────────────────────────────────────────────────────────────────

def test_la_app_tiene_enviar_los_registros():
    """El botón en «Sugerir un cambio o informar de un error» (panel y documentación):
    pulsarlo prepara los registros (sin la carpeta personal) y abre la ventana."""
    with tempfile.TemporaryDirectory() as d:
        Path(d, rg.FICHERO).write_text(f'algo pasó en {CASA}\\Borreguil\n', encoding='utf-8')
        antes = {k: os.environ.get(k) for k in ('BORREGUIL_DATA_DIR', 'BORREGUIL_SIN_REGISTRO')}
        os.environ['BORREGUIL_DATA_DIR'] = d
        os.environ['BORREGUIL_SIN_REGISTRO'] = '1'     # sin cambiar la consola de las pruebas
        try:
            import sklearn.ensemble  # noqa: F401  (antes que la app; ver test_cauces)
            from streamlit.testing.v1 import AppTest
            at = AppTest.from_file(str(APP / 'app.py'), default_timeout=180)
            at.run()
            assert not at.exception, at.exception
            botones = [b for b in at.button if b.key and b.key.startswith('registros_')]
            assert {b.key for b in botones} >= {'registros_lateral', 'registros_documentacion'}
            assert all(b.label.startswith('→ Enviar los registros de este ordenador') for b in botones)
            at.button(key='registros_lateral').click().run()
            assert not at.exception, at.exception
            texto = at.session_state['registros_texto']
            assert 'algo pasó en ~\\Borreguil' in texto and CASA not in texto
            assert texto.startswith('Registros de Borreguil · Wet Meadow Finder ')
            import streamlit as st    # cada st.error va al registro (y con su botón)
            assert getattr(st.error, '_registra', False)

            def con_un_error():       # cualquier error de la app, como en el asistente
                import streamlit as st
                try:
                    1 / 0
                except ZeroDivisionError:
                    st.error('Error: algo ha fallado')
                st.error('Falta el área de estudio')

            ae = AppTest.from_function(con_un_error, default_timeout=60)
            ae.run()
            assert [e.value for e in ae.error] == ['Error: algo ha fallado',
                                                   'Falta el área de estudio']
            debajo = [b for b in ae.button if b.label == 'Enviar los registros']
            assert len(debajo) == 2
            debajo[0].click().run()
            assert ae.session_state['_abrir_registros']
            assert ae.session_state['registros_texto'].startswith('Registros de Borreguil')
        finally:
            for k, v in antes.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


if __name__ == '__main__':
    pruebas = [(n, f) for n, f in sorted(globals().items())
               if n.startswith('test_') and callable(f)]
    fallos = 0
    for nombre, f in pruebas:
        try:
            f()
            print(f'  ok     {nombre}')
        except Exception as e:                      # noqa: BLE001
            fallos += 1
            print(f'  FALLO  {nombre}: {type(e).__name__}: {e}')
    print(f'{len(pruebas) - fallos}/{len(pruebas)} pruebas correctas')
    sys.exit(1 if fallos else 0)
