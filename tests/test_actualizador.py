"""El código de la app se actualiza solo al abrirla (actualizador.py, launcher.py).

25-11-2026, la autora: «haz el lanzador para que se actualice sola» (lo pidieron Javi y
Domingo para las aulas). Las pruebas usan un «GitHub» de mentira en una carpeta
(direcciones file://): main/codigo.json y, al lado, una carpeta por commit.
"""
import json
import shutil
import sys
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP))

import actualizador as act    # noqa: E402

COMMIT = 'a' * 40
COMMIT2 = 'b' * 40


def _escribir(ruta, texto, crlf=False):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    datos = texto.encode('utf-8')
    ruta.write_bytes(datos.replace(b'\n', b'\r\n') if crlf else datos)


def instalador(tmp, version='1.0.0', crlf=True):
    """Lo que trae el instalador: el código (con CRLF, como al compilar en Windows), un
    modelo y la plantilla."""
    res = tmp / 'instalador'
    _escribir(res / 'app.py', 'VERSION_APP = "vieja"\n', crlf)
    _escribir(res / 'borreguil_pipeline.py', 'X = 1\n', crlf)
    _escribir(res / 'CITATION.cff', f'version: {version}\n', crlf)
    _escribir(res / 'plantilla' / 'definiciones.md', '# Definiciones\n', crlf)
    (res / 'rf_prueba.joblib').write_bytes(b'\x00modelo' * 10)
    return res


def publicar(tmp, ficheros, version='1.0.1', commit=COMMIT, minimo=1, roto=None):
    """Publica en el GitHub de mentira: los ficheros en su commit y la lista en main.
    `roto`: ruta cuyo contenido en el servidor no coincide con la lista."""
    srv = tmp / 'github'
    lista = []
    for ruta, texto in ficheros.items():
        datos = texto.encode('utf-8')
        _escribir(srv / commit / ruta, texto)
        if ruta == roto:                        # mismo tamaño, otro contenido
            _escribir(srv / commit / ruta, texto.swapcase())
        sha, lf = act.huellas_de(datos, ruta)
        lista.append({'ruta': ruta, 'sha256': sha, 'sha256_lf': lf, 'bytes': len(datos)})
    datos = {'formato': 1, 'version': version, 'commit': commit, 'fecha': '2026-11-25',
             'instalador_minimo': minimo, 'ficheros': lista}
    _escribir(srv / 'main' / 'codigo.json', json.dumps(datos))
    return (srv / 'main' / 'codigo.json').as_uri()


def codigo(version='1.0.1', app='VERSION_APP = "nueva"\n'):
    return {'app.py': app, 'borreguil_pipeline.py': 'X = 1\n',
            'CITATION.cff': f'version: {version}\n',
            'plantilla/definiciones.md': '# Definiciones\n'}


def bien(carpeta):
    return True, 'RESULTADO: correcto'


def test_si_el_publicado_es_el_del_instalador_no_descarga_nada(tmp_path):
    res = instalador(tmp_path)
    url = publicar(tmp_path, codigo(version='1.0.0', app='VERSION_APP = "vieja"\n'),
                   version='1.0.0')
    r = act.actualizar(res, tmp_path / 'datos', url=url, probar=bien, aviso=lambda t: 0)
    # los saltos de línea de Windows (CRLF) del instalador no cuentan como cambio
    assert r['estado'] == 'instalador', r
    assert act.carpeta_a_usar(res, tmp_path / 'datos') == (res, None)


def test_descarga_solo_lo_cambiado_la_prueba_y_la_usa(tmp_path):
    res = instalador(tmp_path)
    datos = tmp_path / 'datos'
    probadas = []
    url = publicar(tmp_path, codigo())
    r = act.actualizar(res, datos, url=url, aviso=lambda t: 0,
                       probar=lambda c: (probadas.append(Path(c)), bien(c))[1])
    assert r['estado'] == 'actualizado', r
    # solo app.py y CITATION.cff han cambiado: lo demás se copia del instalador
    assert r['detalle'] == '2 ficheros descargados'
    assert len(probadas) == 1
    carpeta, version = act.carpeta_a_usar(res, datos)
    assert version['version'] == '1.0.1' and carpeta.parent == datos / 'codigo'
    assert (carpeta / 'app.py').read_text() == 'VERSION_APP = "nueva"\n'
    # la app busca a su lado los modelos y la plantilla: se copian del instalador
    assert (carpeta / 'rf_prueba.joblib').read_bytes() == b'\x00modelo' * 10
    assert (carpeta / 'plantilla' / 'definiciones.md').exists()
    assert sorted(p.name for p in (datos / 'codigo').iterdir()) == sorted([
        'activo.json', COMMIT[:12]])
    # La siguiente vez está al día: no se descarga nada (en el servidor ya no está)
    shutil.rmtree(tmp_path / 'github' / COMMIT)
    r = act.actualizar(res, datos, url=url, probar=bien, aviso=lambda t: 0)
    assert r['estado'] == 'al_dia', r


def test_una_version_que_no_pasa_la_prueba_no_se_usa(tmp_path):
    res = instalador(tmp_path)
    url = publicar(tmp_path, codigo())
    r = act.actualizar(res, tmp_path / 'datos', url=url, aviso=lambda t: 0,
                       probar=lambda c: (False, 'ModuleNotFoundError: libreria_nueva'))
    assert r['estado'] == 'no_pasa_prueba' and 'libreria_nueva' in r['detalle']
    assert act.carpeta_a_usar(res, tmp_path / 'datos') == (res, None)
    # y no quedan restos de la preparación
    assert not any((tmp_path / 'datos' / 'codigo').glob('.preparando-*'))


def test_un_fichero_cambiado_por_el_camino_no_se_usa(tmp_path):
    res = instalador(tmp_path)
    url = publicar(tmp_path, codigo(), roto='app.py')
    r = act.actualizar(res, tmp_path / 'datos', url=url, probar=bien, aviso=lambda t: 0)
    assert r['estado'] == 'error' and 'huella' in r['detalle'], r
    assert act.carpeta_a_usar(res, tmp_path / 'datos') == (res, None)


def test_si_necesita_un_instalador_nuevo_avisa_y_no_descarga(tmp_path):
    res = instalador(tmp_path)
    url = publicar(tmp_path, codigo(), minimo=act.INSTALADOR + 1)
    r = act.actualizar(res, tmp_path / 'datos', url=url, probar=bien, aviso=lambda t: 0)
    assert r['estado'] == 'necesita_instalador'
    assert 'releases/latest' in act.texto_estado(r)
    assert not (tmp_path / 'datos' / 'codigo').exists()


def test_sin_conexion_o_desactivado_sigue_con_lo_que_hay(tmp_path):
    res = instalador(tmp_path)
    datos = tmp_path / 'datos'
    act.actualizar(res, datos, url=publicar(tmp_path, codigo()), probar=bien,
                   aviso=lambda t: 0)
    no_hay = (tmp_path / 'no_existe' / 'main' / 'codigo.json').as_uri()
    assert act.actualizar(res, datos, url=no_hay, aviso=lambda t: 0)['estado'] == \
        'sin_conexion'
    assert act.actualizar(res, datos, url='', aviso=lambda t: 0)['estado'] == 'desactivado'
    # en los dos casos se sigue usando la versión descargada
    assert act.carpeta_a_usar(res, datos)[1]['version'] == '1.0.1'


def test_marcha_atras_publicando_el_codigo_del_instalador(tmp_path):
    res = instalador(tmp_path)
    datos = tmp_path / 'datos'
    act.actualizar(res, datos, url=publicar(tmp_path, codigo()), probar=bien,
                   aviso=lambda t: 0)
    url = publicar(tmp_path, codigo(version='1.0.0', app='VERSION_APP = "vieja"\n'),
                   version='1.0.0', commit=COMMIT2)
    r = act.actualizar(res, datos, url=url, probar=bien, aviso=lambda t: 0)
    assert r['estado'] == 'instalador'
    assert act.carpeta_a_usar(res, datos) == (res, None)
    assert not (datos / 'codigo' / 'activo.json').exists()


def test_un_instalador_mas_nuevo_manda_sobre_lo_descargado(tmp_path):
    res = instalador(tmp_path)
    datos = tmp_path / 'datos'
    url = publicar(tmp_path, codigo())
    act.actualizar(res, datos, url=url, probar=bien, aviso=lambda t: 0)
    # se instala el programa nuevo (1.0.2) antes de publicar su código
    _escribir(res / 'CITATION.cff', 'version: 1.0.2\n')
    _escribir(res / 'app.py', 'VERSION_APP = "del instalador nuevo"\n')
    assert act.carpeta_a_usar(res, datos) == (res, None)
    r = act.actualizar(res, datos, url=url, probar=bien, aviso=lambda t: 0)
    assert r['estado'] == 'instalador' and r['version'] == '1.0.2'


@pytest.mark.parametrize('cambio, error', [
    ({'ruta': '../fuera.py'}, 'nombre de fichero'),
    ({'ruta': 'plantilla/../../x.py'}, 'nombre de fichero'),
    ({'sha256': 'abc'}, 'huella'),
    ({'bytes': -1}, 'tamaño'),
])
def test_la_lista_se_comprueba_antes_de_usarla(cambio, error):
    datos = {'commit': COMMIT, 'ficheros': [dict(
        {'ruta': 'app.py', 'sha256': 'a' * 64, 'bytes': 3}, **cambio)]}
    with pytest.raises(ValueError, match=error):
        act.interpretar(datos, act.URL_POR_DEFECTO)


def test_la_lista_solo_vale_del_mismo_sitio_y_por_https(tmp_path):
    bien_hecha = {'commit': COMMIT, 'ficheros': [
        {'ruta': 'app.py', 'sha256': 'a' * 64, 'bytes': 3}]}
    lista = act.interpretar(bien_hecha, act.URL_POR_DEFECTO)
    # los ficheros se bajan del commit publicado, no de la rama
    assert lista['ficheros'][0]['url'] == act.REPO + COMMIT + '/app.py'
    with pytest.raises(ValueError, match='mismo sitio'):
        act.interpretar(dict(bien_hecha, base='https://otro.sitio/'), act.URL_POR_DEFECTO)
    with pytest.raises(ValueError, match='app.py'):
        act.interpretar({'commit': COMMIT, 'ficheros': [
            {'ruta': 'i18n.py', 'sha256': 'a' * 64, 'bytes': 3}]}, act.URL_POR_DEFECTO)
    with pytest.raises(ValueError, match='commit'):
        act.interpretar({'ficheros': bien_hecha['ficheros']}, act.URL_POR_DEFECTO)
    with pytest.raises(ValueError, match='HTTPS'):
        act.leer_manifiesto('http://raw.githubusercontent.com/x/main/codigo.json')


def test_los_modulos_de_la_carpeta_elegida_van_por_delante(tmp_path, monkeypatch):
    """Dentro del ejecutable, un módulo de la app puede estar también compilado; el de la
    carpeta del código tiene que ganar."""
    otra, elegida = tmp_path / 'otra', tmp_path / 'elegida'
    _escribir(otra / 'modulo_de_prueba_act.py', 'VALOR = "otra"\n')
    _escribir(elegida / 'modulo_de_prueba_act.py', 'VALOR = "elegida"\n')
    _escribir(elegida / 'launcher.py', 'raise RuntimeError("no se cambia el lanzador")\n')
    monkeypatch.setattr(sys, 'path', [str(otra)] + sys.path)
    monkeypatch.setattr(sys, 'meta_path', list(sys.meta_path))
    import modulo_de_prueba_act
    assert modulo_de_prueba_act.VALOR == 'otra'
    nombres = act.usar_codigo_de(elegida)
    assert 'launcher' not in nombres
    sys.modules.pop('modulo_de_prueba_act', None)
    import modulo_de_prueba_act as m2
    assert m2.VALOR == 'elegida'
    sys.modules.pop('modulo_de_prueba_act', None)


def test_el_spec_empaqueta_los_mismos_modulos_que_se_publican():
    spec = (APP / 'borreguil.spec').read_text(encoding='utf-8')
    assert 'from actualizador import MODULOS_APP' in spec
    assert all((APP / m).exists() for m in act.MODULOS_APP)


def test_la_prueba_en_otro_proceso_usa_el_codigo_de_la_carpeta(tmp_path):
    """Con el código de verdad pasa el autodiagnóstico; con un módulo roto, no: así se
    ve que se prueba el de la carpeta y no el del repositorio."""
    import launcher
    carpeta = tmp_path / 'version'
    for r in act.ficheros_de_la_app(APP):
        if r.startswith(('retrain_', 'publicar_')) or r in ('launcher.py',
                                                             'actualizador.py'):
            continue
        (carpeta / r).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(APP / r, carpeta / r)
    ok, texto = launcher.probar_en_otro_proceso(carpeta)
    assert ok, texto
    _escribir(carpeta / 'point_profile.py', 'import libreria_que_no_existe\n')
    ok, texto = launcher.probar_en_otro_proceso(carpeta)
    assert not ok and 'libreria_que_no_existe' in texto


def test_publicar_construye_una_lista_que_el_programa_entiende():
    import publicar_codigo as pc
    lista = pc.construir_lista('HEAD')
    rutas = [f['ruta'] for f in lista['ficheros']]
    assert set(act.MODULOS_APP) <= set(rutas) and 'CITATION.cff' in rutas
    assert any(r.startswith('plantilla/') for r in rutas)
    assert not any(r.endswith('.joblib') for r in rutas)    # los modelos van aparte
    assert lista['version'] and len(lista['commit']) == 40
    act.interpretar(lista, act.URL_POR_DEFECTO)
