"""Pruebas de la cuenta compartida de MonitorEO en Earth Engine (gee_backend.py).

Sin proyecto, la app calcula con la cuenta compartida de MonitorEO, como sus
herramientas: pide a su servicio (el «broker») un permiso de una hora y etiqueta sus
cálculos con el identificador anónimo de la instalación. Con un proyecto, con la cuenta
de Google de quien usa la app. Aquí el servicio es uno de mentira en este ordenador y
Earth Engine está simulado, así que no hace falta red. Se ejecutan con
`python tests/test_cuenta_compartida.py` (o con pytest).
"""
import json
import os
import re
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

APP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP))

import gee_backend as geb   # noqa: E402


class _Servicio:
    """Un broker de mentira: /token y /usage, con la respuesta que se le diga."""

    def __init__(self):
        self.pedidos = []
        self.token = {'access_token': 'tok-1', 'expires_at': time.time() + 3600,
                      'project': 'proyecto-compartido'}
        self.error = None            # (código HTTP, {'error': …, 'code': …})
        self.uso = {'left_hours': 21.5, 'limit_hours': 25.0, 'used_hours': 3.5,
                    'resets': '2026-12-01', 'restricted': False}
        servicio = self

        class _H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                u = urlparse(self.path)
                servicio.pedidos.append((u.path, parse_qs(u.query)))
                if servicio.error and u.path == '/token':
                    codigo, cuerpo = servicio.error
                else:
                    codigo, cuerpo = 200, (servicio.token if u.path == '/token'
                                           else servicio.uso)
                datos = json.dumps(cuerpo).encode()
                self.send_response(codigo)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(datos)))
                self.end_headers()
                self.wfile.write(datos)

        self.srv = ThreadingHTTPServer(('127.0.0.1', 0), _H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f'http://127.0.0.1:{self.srv.server_address[1]}'

    def cerrar(self):
        self.srv.shutdown()


class _EE:
    """Lo que la app usa de earthengine-api, simulado."""

    def __init__(self):
        self.inicios, self.etiquetas, self.quitadas = [], [], 0
        ee = self

        class _Num:
            def __init__(self, v):
                self.v = v

            def getInfo(self):
                return self.v

        class _Data:
            def setDefaultWorkloadTag(self, t):
                ee.etiquetas.append(t)

            def resetWorkloadTag(self, *a):
                ee.quitadas += 1

        self.Number = _Num
        self.data = _Data()

    def Initialize(self, credentials=None, project=None, **kw):
        self.inicios.append({'credentials': credentials, 'project': project})


def _preparar():
    """Servicio y Earth Engine de mentira, y un fichero de identificador nuevo."""
    servicio = _Servicio()
    ee = _EE()
    carpeta = Path(tempfile.mkdtemp(prefix='usuario_'))
    antes = {'MONITOREO_BROKER': os.environ.get('MONITOREO_BROKER'),
             'EE_SERVICE_ACCOUNT_KEY': os.environ.get('EE_SERVICE_ACCOUNT_KEY')}
    os.environ['MONITOREO_BROKER'] = servicio.url
    os.environ.pop('EE_SERVICE_ACCOUNT_KEY', None)
    estado = (geb.FICHERO_USUARIO, geb._usuario, geb._compartida, sys.modules.get('ee'))
    geb.FICHERO_USUARIO = str(carpeta / 'monitoreo' / 'user_id')
    geb._usuario, geb._compartida = None, False
    sys.modules['ee'] = ee

    def deshacer():
        servicio.cerrar()
        geb.FICHERO_USUARIO, geb._usuario, geb._compartida = estado[:3]
        if estado[3] is None:
            sys.modules.pop('ee', None)
        else:
            sys.modules['ee'] = estado[3]
        for k, v in antes.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return servicio, ee, carpeta, deshacer


def test_sin_proyecto_usa_la_cuenta_compartida():
    servicio, ee, _c, deshacer = _preparar()
    try:
        ok, msg = geb.initialize(None)
        assert ok, msg
        assert 'cuenta compartida de MonitorEO' in msg and 'proyecto-compartido' in msg
        assert geb.cuenta_compartida()
        ini = ee.inicios[-1]
        assert ini['project'] == 'proyecto-compartido'
        assert ini['credentials'].token == 'tok-1'
        # los cálculos llevan el identificador de la instalación, el que se pidió
        uid = geb.id_usuario()
        assert ee.etiquetas == [uid]
        ruta, params = servicio.pedidos[0]
        assert ruta == '/token' and params['user'] == [uid]
        # un proyecto vacío o con espacios es lo mismo que ninguno
        ok, _ = geb.initialize('   ')
        assert ok and len(ee.inicios) == 2 and ee.inicios[-1]['project'] == 'proyecto-compartido'
    finally:
        deshacer()


def test_el_permiso_se_renueva_solo():
    servicio, ee, _c, deshacer = _preparar()
    try:
        assert geb.initialize(None)[0]
        cred = ee.inicios[-1]['credentials']
        servicio.token = {'access_token': 'tok-2', 'expires_at': time.time() + 3600,
                          'project': 'proyecto-compartido'}
        cred.refresh(None)
        assert cred.token == 'tok-2'
        assert sum(1 for r, _p in servicio.pedidos if r == '/token') == 2
    finally:
        deshacer()


def test_con_proyecto_usa_la_cuenta_propia_y_quita_la_etiqueta():
    servicio, ee, _c, deshacer = _preparar()
    try:
        assert geb.initialize(None)[0] and geb.cuenta_compartida()
        ok, msg = geb.initialize('ee-mi-proyecto')
        assert ok, msg
        assert ee.inicios[-1] == {'credentials': None, 'project': 'ee-mi-proyecto'}
        assert not geb.cuenta_compartida()
        assert ee.quitadas == 1           # lo propio no se cuenta como de la compartida
        assert [r for r, _p in servicio.pedidos].count('/token') == 1
    finally:
        deshacer()


def test_con_clave_de_cuenta_de_servicio_no_pide_permiso():
    """En un despliegue en la nube (EE_SERVICE_ACCOUNT_KEY) manda la clave, como antes."""
    servicio, ee, _c, deshacer = _preparar()
    try:
        ok, msg = geb.initialize(None, service_account_json='{"no es": "una clave"}')
        assert not ok and 'service account' in msg
        assert not servicio.pedidos and not geb.cuenta_compartida()
    finally:
        deshacer()


def test_horas_gastadas_dice_que_use_su_cuenta():
    servicio, ee, _c, deshacer = _preparar()
    try:
        servicio.error = (403, {'error': 'spent', 'code': 'spent'})
        ok, msg = geb.initialize(None)
        assert not ok and not geb.cuenta_compartida()
        assert 'Has gastado las horas de cálculo de este mes' in msg
        assert 'tu propia cuenta' in msg and 'Autenticar GEE' in msg
        servicio.error = (403, {'error': 'project', 'code': 'project'})
        ok, msg = geb.initialize(None)
        assert not ok and 'ha gastado las horas de cálculo de este mes' in msg
        servicio.error = (500, {'error': 'roto'})
        ok, msg = geb.initialize(None)
        assert not ok and 'Prueba más tarde' in msg and not ee.inicios
    finally:
        deshacer()


def test_sin_servicio_no_conecta_y_lo_dice():
    servicio, ee, _c, deshacer = _preparar()
    try:
        os.environ['MONITOREO_BROKER'] = 'http://127.0.0.1:9'     # nadie escucha
        ok, msg = geb.initialize(None)
        assert not ok and 'cuenta compartida de MonitorEO' in msg and 'tu propia cuenta' in msg
        assert geb.uso_compartida() is None
    finally:
        deshacer()


def test_identificador_anonimo_se_crea_una_vez_y_se_reutiliza():
    servicio, ee, carpeta, deshacer = _preparar()
    try:
        uid = geb.id_usuario()
        assert re.fullmatch(r'mo-[0-9a-f]{16}', uid)
        fichero = carpeta / 'monitoreo' / 'user_id'
        assert fichero.read_text(encoding='utf-8').strip() == uid
        geb._usuario = None
        assert geb.id_usuario() == uid           # el mismo, leído del fichero
        # el fichero de las herramientas de MonitorEO vale tal cual
        fichero.write_text('mo-0123456789abcdef\n', encoding='utf-8')
        geb._usuario = None
        assert geb.id_usuario() == 'mo-0123456789abcdef'
        # uno que no tiene esa forma no se usa
        fichero.write_text('otra cosa\n', encoding='utf-8')
        geb._usuario = None
        assert geb.id_usuario() != 'otra cosa'
    finally:
        deshacer()


def test_uso_de_la_cuenta_compartida():
    servicio, ee, _c, deshacer = _preparar()
    try:
        u = geb.uso_compartida()
        assert u['left_hours'] == 21.5 and u['resets'] == '2026-12-01'
        ruta, params = servicio.pedidos[-1]
        assert ruta == '/usage' and params['user'] == [geb.id_usuario()]
    finally:
        deshacer()


def test_la_app_pone_la_compartida_por_defecto_y_deja_elegir_la_propia():
    texto = (APP / 'app.py').read_text(encoding='utf-8')
    assert "'Proyecto de Google Cloud (vacío: cuenta compartida)'" in texto
    assert 'def _uso_cuenta_compartida' in texto
    # el mapa ya no exige un proyecto: sin él, la cuenta compartida
    assert '_puede = bool(_res[\'listo\'] and _predictores and _anios)' in texto
    assert 'Escribe el proyecto de Earth Engine en el panel lateral.' not in texto


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
