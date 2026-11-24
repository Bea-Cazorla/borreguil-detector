"""Pruebas de quién revisa cada punto, su perfil y su certidumbre.

Cada vez que una persona dice si un punto es borreguil (en «Revisar puntos», al pinchar
en el mapa o en su fichero de campo) se guarda quién lo dice (`observer`), su perfil
(`expertise`: student · manager · researcher) y lo segura que está (`certainty`: high ·
medium · low), con las columnas de la plantilla de datos de campo. Con eso, el acierto
del modelo se puede separar según la certidumbre y el perfil. Se ejecutan con
`python tests/test_revisor.py` (o con pytest); no necesitan red y tardan unos segundos.
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                 # noqa: E402

import borreguil_pipeline as bp   # noqa: E402

bp._imports()

TMP = Path(tempfile.mkdtemp(prefix='revisor_'))


def test_perfil_y_certidumbre_en_ingles_y_en_espanol():
    assert [bp.perfil_de(v) for v in ('Investigadora', 'gestor/a', ' student ', 'Gestor')] \
        == ['researcher', 'manager', 'student', 'manager']
    assert [bp.certidumbre_de(v) for v in ('Alta', 'medium', 'BAJA', 'media')] \
        == ['high', 'medium', 'low', 'medium']
    assert bp.perfil_de('') == '' and bp.perfil_de(None) == '' and bp.perfil_de(float('nan')) == ''
    assert bp.perfil_de('astronauta') == '?' and bp.certidumbre_de('muy alta') == '?'


def test_anotar_revision_pone_y_quita():
    r = {'ID': 'p1', 'observer': 'Luis', 'expertise': 'student', 'certainty': 'low'}
    bp.anotar_revision(r, ' Ana ', 'investigadora', 'alta')
    assert (r['observer'], r['expertise'], r['certainty']) == ('Ana', 'researcher', 'high')
    # una revisión nueva sin datos quita los de la anterior: eran de otra persona
    bp.anotar_revision(r, '', '', '')
    assert not any(k in r for k in bp.CAMPOS_QUIEN)
    # un valor que no se entiende no se guarda
    bp.anotar_revision(r, 'Ana', 'astronauta', 'segurísima')
    assert r['observer'] == 'Ana' and 'expertise' not in r and 'certainty' not in r


def test_leer_un_fichero_con_columnas_en_espanol():
    rows = [{'ID': 'a', 'Borreguil': 'si', 'revisor': 'Ana', 'perfil': 'Gestora',
             'certidumbre': 'Alta'},
            {'ID': 'b', 'Borreguil': 'no', 'revisor': '', 'perfil': 'astronauta',
             'certidumbre': 'baja'},
            {'ID': 'c', 'Borreguil': 'si'}]
    info = bp.leer_etiquetas(rows)
    rev = info['revision']
    assert rows[0]['observer'] == 'Ana' and rows[0]['expertise'] == 'manager' \
        and rows[0]['certainty'] == 'high'
    assert 'expertise' not in rows[1] and rows[1]['certainty'] == 'low' \
        and 'observer' not in rows[1]
    assert not any(k in rows[2] for k in bp.CAMPOS_QUIEN)
    # las columnas en español se pasan a las de la plantilla
    assert not any(k in r for r in rows for k in ('revisor', 'perfil', 'certidumbre'))
    assert rev['expertise']['n'] == 1 and rev['expertise']['no_entendidos'] == {'astronauta': 1}
    assert rev['certainty']['n'] == 2
    textos = ' '.join(t for _, t in bp.resumen_etiquetas(info))
    assert 'certidumbre en 2 puntos' in textos and '«astronauta» (1)' in textos


def test_punto_del_mapa_lleva_quien_perfil_y_certidumbre():
    base = [{'ID': 'p1', 'lon': -3.30, 'lat': 37.05}]
    r = bp.punto_del_mapa(base, -3.35, 37.04, 'si', observador='Ana', perfil='manager',
                          certidumbre='medium', fecha='2026-11-27')
    assert (r['observer'], r['expertise'], r['certainty']) == ('Ana', 'manager', 'medium')
    # quien no sabe decir si hay borreguil no tiene certidumbre que dar
    d = bp.punto_del_mapa(base, -3.36, 37.04, 'duda', observador='Ana', perfil='manager',
                          certidumbre='high')
    assert d['Duda'] == 'si' and 'certainty' not in d and d['expertise'] == 'manager'


def test_un_punto_de_campo_pasa_quien_y_como_al_candidato():
    rows = [{'ID': 'c1', 'lon': -3.300000, 'lat': 37.050000}]
    campo = [{'ID': 'f1', 'lon': -3.300010, 'lat': 37.050010, 'presence': '1',
              'observer': 'BPC', 'expertise': 'researcher', 'certainty': 'high',
              'method': 'field', 'date': '2026-07-15'}]
    n_match, n_add, _, _ = bp.merge_truth_points(rows, campo)
    assert (n_match, n_add) == (1, 0)
    r = rows[0]
    assert (r['observer'], r['expertise'], r['certainty'], r['method'], r['date']) == \
        ('BPC', 'researcher', 'high', 'field', '2026-07-15')


def test_va_y_vuelve_en_el_geojson():
    rows = [{'ID': 'a', 'lon': -3.30, 'lat': 37.05, 'Borreguil': 'si', 'origin': 'verificado',
             'decision': 'BORREGUIL VERIFICADO'},
            {'ID': 'b', 'lon': -3.31, 'lat': 37.06, 'Borreguil': 'no', 'origin': 'verificado',
             'decision': 'NO BORREGUIL VERIFICADO'}]
    bp.anotar_revision(rows[0], 'Ana', 'researcher', 'high')
    bp.anotar_revision(rows[1], 'Luis', 'student', 'low')
    ruta = TMP / 'puntos.geojson'
    bp.save_points_geojson(rows, ruta)
    props = {f['properties']['ID']: f['properties']
             for f in json.loads(ruta.read_text(encoding='utf-8'))['features']}
    assert (props['a']['observer'], props['a']['expertise'], props['a']['certainty']) == \
        ('Ana', 'researcher', 'high')
    assert props['b']['certainty'] == 'low' and props['b']['presence'] == 0
    leidos = bp.read_points(str(ruta))
    bp.leer_etiquetas(leidos)
    otra = {r['ID']: r for r in leidos}
    assert (otra['b']['observer'], otra['b']['expertise'], otra['b']['certainty']) == \
        ('Luis', 'student', 'low')


def test_excel_con_quien_perfil_y_certidumbre_en_espanol():
    from openpyxl import load_workbook
    rows = [{'ID': 'a', 'lon': -3.30, 'lat': 37.05, 'Borreguil': 'si', 'rf_proba': 0.9,
             'decision': 'BORREGUIL VERIFICADO'}]
    bp.anotar_revision(rows[0], 'Ana', 'manager', 'medium')
    ruta = TMP / 'puntos.xlsx'
    bp.save_xlsx(rows, ruta)
    ws = load_workbook(ruta).active
    cab = [c.value for c in ws[1]]
    fila = dict(zip(cab, [c.value for c in ws[2]]))
    assert (fila['Revisado por'], fila['Perfil'], fila['Certidumbre']) == \
        ('Ana', 'gestor/a', 'media')


def test_acierto_separado_por_certidumbre_y_perfil():
    rows = []
    # 4 borreguiles de certidumbre alta (el modelo acierta los 4), 2 de baja (falla 1),
    # 3 ausencias sin certidumbre (acierta 2) y un punto sin etiqueta
    for i in range(4):
        rows.append(bp.anotar_revision({'Borreguil': 'si'}, 'Ana', 'researcher', 'high'))
    for i in range(2):
        rows.append(bp.anotar_revision({'Borreguil': 'si'}, 'Luis', 'student', 'low'))
    for i in range(3):
        rows.append({'Borreguil': 'no'})
    rows.append({})
    proba = np.array([0.9, 0.8, 0.95, 0.7, 0.9, 0.1, 0.1, 0.2, 0.9, 0.5])
    y = bp.etiquetas_de_campo(rows)
    ev = bp._evaluacion(y, rows, partida=proba)
    assert ev['certidumbre'] == ['high'] * 4 + ['low'] * 2 + [''] * 3
    rev = bp.aciertos_por_revision(ev, 0.5, 'partida')
    c = rev['certidumbre']
    assert list(c) == ['high', 'low', '']                 # en su orden, sin los vacíos
    assert (c['high']['ok_si'], c['high']['n_si']) == (4, 4)
    assert (c['low']['ok_si'], c['low']['n_si']) == (1, 2)
    assert (c['']['ok_no'], c['']['n_no']) == (2, 3)
    assert set(rev['perfil']) == {'researcher', 'student', ''}
    # sin certidumbre ni perfil en ningún punto, no hay nada que separar
    sin = bp._evaluacion(y, [{'Borreguil': r.get('Borreguil')} for r in rows], partida=proba)
    assert bp.aciertos_por_revision(sin, 0.5, 'partida') == {}
    # ni sin `rows` (evaluaciones guardadas antes)
    assert bp.aciertos_por_revision(bp._evaluacion(y, partida=proba), 0.5) == {}


if __name__ == '__main__':
    pruebas = [(n, f) for n, f in sorted(globals().items())
               if n.startswith('test_') and callable(f)]
    fallos = 0
    try:
        for nombre, f in pruebas:
            try:
                f()
                print(f'  ok     {nombre}')
            except Exception as e:                      # noqa: BLE001
                fallos += 1
                print(f'  FALLO  {nombre}: {type(e).__name__}: {e}')
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    print(f'{len(pruebas) - fallos}/{len(pruebas)} pruebas correctas')
    sys.exit(1 if fallos else 0)
