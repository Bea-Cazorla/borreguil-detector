"""La ventana del programa mientras arranca (launcher.py), con un pywebview de mentira.

27-11-2026: la ventana se abre enseguida y es ella la que busca las versiones nuevas de la
app, diciendo en la página de carga lo que va pasando; si de ahí sale algo que hay que
hacer (un programa nuevo, una versión que no pasa la comprobación), lo enseña en un
cuadro. En Mac, la app del Escritorio del instalador no tiene consola: sin esto, esos
avisos solo iban a salida.log.
"""
import sys
import threading
import time
import types
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP))

import actualizador  # noqa: E402
import launcher      # noqa: E402


class VentanaFalsa:
    def __init__(self, html):
        self.paginas = [html]
        self.url = None
        self.dialogos = []
        self.respuesta = True
        self.destruida = False

    def load_html(self, html):
        self.paginas.append(html)

    def load_url(self, url):
        self.url = url

    def evaluate_js(self, js):
        return 'Encuentra los borreguiles'

    def destroy(self):
        self.destruida = True

    def create_confirmation_dialog(self, titulo, texto):
        self.dialogos.append((titulo, texto))
        return self.respuesta


class Mundo:
    """pywebview, el servidor y el navegador de mentira; apunta qué pasa y en qué orden.

    `como`: 'normal' (pywebview llama a la función y la ventana sigue abierta hasta que
    acaba), 'cierra' (la persona cierra la ventana en cuanto empieza la búsqueda) o
    'falla' (pywebview falla después de haber empezado la función)."""

    def __init__(self, monkeypatch, como='normal'):
        self.orden, self.abierto, self.ventana = [], [], None
        self.como, self.hilo, self.opciones = como, None, {}
        self.buscando = threading.Event()        # la búsqueda ha empezado
        self.cuadro = threading.Event()          # el cuadro se ha enseñado
        self.hilo_del_cuadro = None
        falso = types.ModuleType('webview')
        falso.settings = {}
        falso.create_window = self.create_window
        falso.start = self.start
        monkeypatch.setitem(sys.modules, 'webview', falso)
        monkeypatch.setattr(launcher, 'lanzar_servidor', self.lanzar_servidor)
        monkeypatch.setattr(launcher, 'esperar_servidor', lambda port, hijo: True)
        monkeypatch.setattr(launcher, 'free_port', lambda preferido=8501: preferido)
        monkeypatch.setattr(launcher, 'parar_servidor',
                            lambda hijo: self.orden.append(('parar', hijo)))
        monkeypatch.setattr(launcher.webbrowser, 'open', self.abierto.append)
        original = launcher.mostrar_aviso

        def mostrar_aviso(ventana, aviso):
            self.hilo_del_cuadro = threading.current_thread()
            try:
                original(ventana, aviso)
            finally:
                self.cuadro.set()
        monkeypatch.setattr(launcher, 'mostrar_aviso', mostrar_aviso)

    def create_window(self, titulo, html, **kw):
        self.ventana = VentanaFalsa(html)
        return self.ventana

    def start(self, func, args, **kw):
        self.opciones = kw
        if self.como == 'normal':
            func(*args)
            return
        # como pywebview: la función, en un hilo que no es daemon
        self.hilo = threading.Thread(target=func, args=args)
        self.hilo.start()
        assert self.buscando.wait(5)
        if self.como == 'falla':
            raise RuntimeError('la ventana no se ha podido crear')

    def lanzar_servidor(self, port):
        self.orden.append(('servidor', port))
        return 'proceso'


def _buscar(mundo, r, pasos=('  Actualizando la app a la versión 9.9.9…',)):
    def buscar(mostrar):
        mundo.orden.append(('buscar',))
        mundo.buscando.set()
        for p in pasos:
            mostrar(p.strip())
        return r
    return buscar


def test_la_ventana_se_abre_antes_y_dice_lo_que_pasa(monkeypatch, tmp_path):
    m = Mundo(monkeypatch)
    r = {'estado': 'actualizado', 'version': '9.9.9', 'detalle': '3 ficheros descargados'}
    resultado = launcher.abrir_en_ventana('http://localhost:1', 1, tmp_path,
                                          buscar=_buscar(m, r))
    assert resultado == ('cerrada', None, True, 1)
    # primero se busca y después se arranca el servidor; al cerrar, se para
    assert m.orden == [('buscar',), ('servidor', 1), ('parar', 'proceso')]
    paginas = m.ventana.paginas
    assert 'Buscando una versión nueva de la app' in paginas[0]
    assert any('Actualizando la app a la versión 9.9.9' in p for p in paginas)
    assert 'App actualizada a la versión 9.9.9' in paginas[-1]
    assert m.ventana.url == 'http://localhost:1'
    assert not m.cuadro.wait(0.3)             # nada que hacer: sin cuadro
    # los botones de los cuadros, en español
    assert m.opciones['localization'] == {'global.ok': 'Aceptar', 'global.cancel': 'Cancelar'}


def test_sin_buscar_la_ventana_dice_abriendo(monkeypatch, tmp_path):
    m = Mundo(monkeypatch)
    launcher.abrir_en_ventana('http://localhost:1', 1, tmp_path)
    assert 'Abriendo la app' in m.ventana.paginas[0]
    assert m.orden[0] == ('servidor', 1)


def test_si_otro_coge_el_puerto_mientras_busca_se_usa_otro(monkeypatch, tmp_path):
    m = Mundo(monkeypatch)
    monkeypatch.setattr(launcher, 'free_port', lambda preferido=8501: 8600)
    resultado = launcher.abrir_en_ventana('http://localhost:8501', 8501, tmp_path,
                                          buscar=_buscar(m, None, ()))
    assert ('servidor', 8600) in m.orden and m.ventana.url == 'http://localhost:8600'
    assert resultado[3] == 8600


@pytest.mark.parametrize('plataforma, dice, enlace', [
    ('darwin', 'vuelve a ejecutar el instalador', launcher.INSTRUCCIONES_MAC),
    ('win32', 'descarga el programa nuevo', actualizador.DESCARGAS_RELEASE),
])
def test_si_hace_falta_un_programa_nuevo_se_dice_en_un_cuadro(monkeypatch, tmp_path, capsys,
                                                               plataforma, dice, enlace):
    monkeypatch.setattr(sys, 'platform', plataforma)
    monkeypatch.setattr(launcher, 'hay_motor_de_edge', lambda: True)
    m = Mundo(monkeypatch)
    r = {'estado': 'necesita_instalador', 'version': '2.0.0', 'detalle': ''}
    launcher.abrir_en_ventana('http://localhost:1', 1, tmp_path, buscar=_buscar(m, r, ()))
    assert m.ventana.url == 'http://localhost:1'          # la app abre igual
    assert m.cuadro.wait(5)
    assert m.hilo_del_cuadro.daemon          # el cuadro no retiene el programa al cerrarse
    [(titulo, texto)] = m.ventana.dialogos
    assert titulo == 'Hay una versión nueva de la app'
    assert '2.0.0' in texto and dice in texto and 'Tus modelos' in texto
    # en Mac, pywebview no enseña el título: va al principio del texto
    assert texto.startswith(titulo) == (plataforma == 'darwin')
    assert m.abierto == [enlace]                          # aceptar abre las instrucciones
    assert 'Aviso en la ventana: Hay una versión nueva' in capsys.readouterr().out


def test_si_no_se_acepta_no_se_abre_el_navegador(monkeypatch, tmp_path):
    m = Mundo(monkeypatch)
    r = {'estado': 'no_pasa_prueba', 'version': '2.0.0', 'detalle': 'x'}

    def crear(titulo, html, **kw):
        m.ventana = VentanaFalsa(html)
        m.ventana.respuesta = False
        return m.ventana
    sys.modules['webview'].create_window = crear
    launcher.abrir_en_ventana('http://localhost:1', 1, tmp_path, buscar=_buscar(m, r, ()))
    assert m.cuadro.wait(5)
    [(titulo, texto)] = m.ventana.dialogos
    assert titulo == 'La versión nueva no se ha podido usar' and '2.0.0' in texto
    assert m.abierto == []


def test_en_la_prueba_de_la_ventana_no_hay_cuadros(monkeypatch, tmp_path, capsys):
    m = Mundo(monkeypatch)
    r = {'estado': 'necesita_instalador', 'version': '2.0.0', 'detalle': ''}
    monkeypatch.setattr(launcher.time, 'sleep', lambda s: None)
    estado, _, _, _ = launcher.abrir_en_ventana('http://localhost:1', 1, tmp_path,
                                                probar=True, buscar=_buscar(m, r, ()))
    assert estado == 'ok' and m.ventana.destruida
    assert not m.cuadro.wait(0.3) and not m.ventana.dialogos
    assert 'Aviso en la ventana: Hay una versión nueva' in capsys.readouterr().out


def test_cerrada_mientras_busca_el_programa_termina_ya(monkeypatch, tmp_path):
    """pywebview espera a su hilo al cerrarse: si la búsqueda (hasta minutos) fuera en él,
    el programa seguiría vivo sin ventana. Va en otro hilo, daemon, y no deja servidor."""
    m = Mundo(monkeypatch, como='cierra')
    seguir, en_daemon = threading.Event(), []

    def buscar(mostrar):
        en_daemon.append(threading.current_thread().daemon)
        m.buscando.set()
        seguir.wait(10)
        return {'estado': 'al_dia', 'version': '1.0.0', 'detalle': ''}

    resultado = launcher.abrir_en_ventana('http://localhost:1', 1, tmp_path, buscar=buscar)
    assert resultado == ('cerrada', None, True, 1)
    m.hilo.join(3)                     # el hilo de pywebview acaba aunque se siga buscando
    assert not m.hilo.is_alive() and en_daemon == [True]
    seguir.set()
    assert ('servidor', 1) not in m.orden and m.orden == [('parar', None)]


def test_si_la_ventana_falla_a_medias_espera_a_la_busqueda(monkeypatch, tmp_path):
    """Si pywebview falla después de empezar la búsqueda, la app sigue en el navegador:
    antes de elegir el código, la búsqueda tiene que haber acabado."""
    m = Mundo(monkeypatch, como='falla')
    acabada = threading.Event()

    def buscar(mostrar):
        m.buscando.set()
        time.sleep(0.5)
        mostrar('Comprobando la versión nueva…')      # la ventana ya no está: no espera
        acabada.set()
        return {'estado': 'actualizado', 'version': '1.0.1', 'detalle': ''}

    resultado = launcher.abrir_en_ventana('http://localhost:1', 1, tmp_path, buscar=buscar)
    assert acabada.is_set()
    assert resultado == ('sin_ventana', None, True, 1)
    m.hilo.join(3)
    assert ('servidor', 1) not in m.orden
    assert not any('Comprobando' in p for p in m.ventana.paginas)


def test_un_fallo_al_buscar_no_deja_la_ventana_colgada(monkeypatch, tmp_path):
    m = Mundo(monkeypatch)

    def buscar(mostrar):
        raise RuntimeError('algo raro')

    launcher.abrir_en_ventana('http://localhost:1', 1, tmp_path, buscar=buscar)
    assert m.ventana.url == 'http://localhost:1'


@pytest.mark.parametrize('estado', ['al_dia', 'instalador', 'actualizado', 'sin_conexion',
                                    'error', 'desactivado'])
def test_lo_demas_no_necesita_cuadro(estado):
    assert launcher.aviso_de_la_actualizacion({'estado': estado, 'version': '1'}) is None
    assert launcher.aviso_de_la_actualizacion(None) is None


def test_la_pagina_de_carga_escapa_el_texto_y_no_repite_lo_que_tarda():
    assert '&lt;b&gt;' in launcher.cargando('<b>') and '<b>' not in launcher.cargando('<b>')
    assert 'La primera vez puede tardar' in launcher.cargando()
    comprobando = launcher.cargando('Comprobando la versión nueva (puede tardar un minuto)…')
    assert comprobando.count('minuto') == 1


@pytest.mark.parametrize('argv, url, busca', [
    (['launcher.py'], 'https://ejemplo.org/codigo.json', True),
    (['launcher.py', '--sin-actualizar'], 'https://ejemplo.org/codigo.json', False),
    (['launcher.py'], '', False),
])
def test_la_ventana_solo_dice_buscando_si_va_a_buscar(monkeypatch, tmp_path, argv, url, busca):
    pasado = {}
    monkeypatch.setattr(sys, 'argv', argv)
    monkeypatch.setenv(actualizador.VARIABLE_URL, url)
    monkeypatch.setenv('BORREGUIL_DATA_DIR', str(tmp_path))
    monkeypatch.delenv('BORREGUIL_NO_BROWSER', raising=False)
    for v in ('HEADLESS', 'PORT', 'ADDRESS', 'FILE_WATCHER_TYPE', 'MAX_UPLOAD_SIZE'):
        monkeypatch.setenv(f'STREAMLIT_SERVER_{v}', '')
        monkeypatch.delenv(f'STREAMLIT_SERVER_{v}')
    monkeypatch.setenv('STREAMLIT_BROWSER_GATHER_USAGE_STATS', '')
    monkeypatch.delenv('STREAMLIT_BROWSER_GATHER_USAGE_STATS')
    monkeypatch.chdir(tmp_path)

    def abrir(*a, buscar=None, **k):
        pasado['buscar'] = buscar
        return 'cerrada', None, buscar is not None, 8501
    monkeypatch.setattr(launcher, 'abrir_en_ventana', abrir)
    launcher.main()
    assert (pasado['buscar'] is not None) == busca


@pytest.mark.parametrize('buscado_por_la_ventana', [False, True])
def test_sin_ventana_se_busca_igual_antes_del_navegador(monkeypatch, tmp_path,
                                                        buscado_por_la_ventana):
    """Si la ventana no se puede abrir, la app va al navegador: la búsqueda de versiones
    nuevas se hace entonces (si la ventana no llegó a hacerla) y antes de elegir el código."""
    orden = []
    monkeypatch.setattr(sys, 'argv', ['launcher.py'])
    monkeypatch.setenv('BORREGUIL_DATA_DIR', str(tmp_path))
    # main() pone estas variables: que se quiten al acabar la prueba
    for v in ('HEADLESS', 'PORT', 'ADDRESS', 'FILE_WATCHER_TYPE', 'MAX_UPLOAD_SIZE'):
        monkeypatch.setenv(f'STREAMLIT_SERVER_{v}', '')
        monkeypatch.delenv(f'STREAMLIT_SERVER_{v}')
    monkeypatch.setenv('STREAMLIT_BROWSER_GATHER_USAGE_STATS', '')
    monkeypatch.delenv('STREAMLIT_BROWSER_GATHER_USAGE_STATS')
    monkeypatch.delenv('BORREGUIL_NO_BROWSER', raising=False)
    monkeypatch.setattr(launcher, 'abrir_en_ventana',
                        lambda *a, **k: ('sin_ventana', None, buscado_por_la_ventana, 8501))
    monkeypatch.setattr(launcher, 'buscar_version_nueva',
                        lambda res, out, mostrar=None: orden.append('buscar'))
    monkeypatch.setattr(launcher, 'codigo_de_la_app',
                        lambda res, out: orden.append('codigo') or APP / 'app.py')
    monkeypatch.setattr(launcher, 'open_browser_when_ready', lambda url, port: None)
    monkeypatch.setattr(launcher, 'arrancar_streamlit',
                        lambda app, port: orden.append('streamlit'))
    monkeypatch.chdir(tmp_path)
    launcher.main()
    esperado = ['codigo', 'streamlit'] if buscado_por_la_ventana else ['buscar', 'codigo',
                                                                        'streamlit']
    assert orden == esperado
