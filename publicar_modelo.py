"""Publica un modelo Random Forest para que la app lo descargue sola (modelos.json).

La app, al abrirse, lee modelos.json en GitHub y descarga los modelos que no tiene o que
han cambiado (modelos_remotos.py). Este script pone el modelo en la carpeta del
repositorio y escribe en la lista su tamaño y su huella; después hay que subir los dos
ficheros a GitHub.

    py publicar_modelo.py rf_mi_zona.joblib --nombre "Mi zona (Sentinel-2 + relieve)"
    py publicar_modelo.py rf_mi_zona.joblib --sin-radar rf_mi_zona_sin_radar.joblib
    py publicar_modelo.py                  # recalcula las huellas (tras reentrenar)
    py publicar_modelo.py --quitar rf_mi_zona.joblib
    py publicar_modelo.py --comprobar      # ¿coincide la lista con los ficheros?

Luego:
    git add modelos.json rf_mi_zona.joblib
    git commit -m "Modelo de mi zona"
    git push
"""
import argparse
import datetime
import json
import shutil
import sys
from pathlib import Path

import sklearn.ensemble  # noqa: F401  (primero, por Smart App Control)
import joblib

import modelos_remotos as mrm

REPO = Path(__file__).resolve().parent
LISTA = REPO / mrm.MANIFIESTO
NOTA = ('Modelos públicos que la app descarga sola al abrirse. No se edita a mano: '
        'usar publicar_modelo.py.')


def leer_lista():
    if not LISTA.exists():
        return {'formato': mrm.FORMATO, 'nota': NOTA, 'modelos': []}
    return json.loads(LISTA.read_text(encoding='utf-8'))


def guardar_lista(datos):
    # Sierra Nevada primero, como en el desplegable de la app
    datos['modelos'].sort(key=lambda d: (d['archivo'] != 'rf_sierra_nevada.joblib',
                                         d['archivo']))
    LISTA.write_text(json.dumps(datos, ensure_ascii=False, indent=2) + '\n',
                     encoding='utf-8', newline='\n')


def comprobar_modelo(ruta):
    """Que sea un modelo de la app: un diccionario con el Random Forest y sus variables."""
    b = joblib.load(ruta)
    if not isinstance(b, dict) or 'model' not in b or not b.get('features'):
        sys.exit(f'{ruta}: no parece un modelo de la app (le falta model o features)')
    return b


def al_repositorio(origen, sub=''):
    """Copia el fichero a la carpeta del repositorio (o a sin_radar/) si no está ya."""
    origen = Path(origen).resolve()
    destino = REPO / sub / origen.name
    if origen != destino.resolve():
        destino.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origen, destino)
        print(f'  copiado a {destino.relative_to(REPO)}')
    return destino


def _entrada(ruta, sub, nombre, fecha):
    url = f'{sub}/{ruta.name}' if sub else None
    return mrm.entrada(ruta, url=url, nombre=nombre, fecha=fecha)


def anadir(ruta, nombre='', sin_radar=None):
    ruta = Path(ruta)
    if not mrm._NOMBRE_OK.match(ruta.name):
        sys.exit(f'{ruta.name}: el nombre solo puede llevar letras sin tildes, números, '
                 '«_», «-» y «.», y acabar en .joblib')
    b = comprobar_modelo(ruta)
    hoy = datetime.date.today().isoformat()
    destino = al_repositorio(ruta)
    datos = leer_lista()
    viejo = next((d for d in datos['modelos'] if d['archivo'] == destino.name), {})
    nombre = nombre or viejo.get('nombre') or str(b.get('training_zone', '')).split(' ·')[0]
    d = _entrada(destino, '', nombre, hoy)
    if sin_radar:
        comprobar_modelo(sin_radar)
        sr = al_repositorio(sin_radar, mrm.CARPETA_SIN_RADAR)
        esperado = destino.name.replace('.joblib', '_sin_radar.joblib')
        if sr.name != esperado:
            sys.exit(f'la versión sin radar tiene que llamarse {esperado}')
        d['sin_radar'] = _entrada(sr, mrm.CARPETA_SIN_RADAR, '', '')
    elif viejo.get('sin_radar'):
        d['sin_radar'] = viejo['sin_radar']
    datos['modelos'] = [x for x in datos['modelos'] if x['archivo'] != destino.name] + [d]
    guardar_lista(datos)
    print(f"  {'puesto al día' if viejo else 'añadido'}: {destino.name}")


def recalcular():
    """Vuelve a calcular tamaño y huella de todo lo que está en la lista (y cambia la
    fecha de lo que ha cambiado)."""
    datos = leer_lista()
    hoy = datetime.date.today().isoformat()
    for d in datos['modelos']:
        for f, sub in ((d, ''), (d.get('sin_radar'), mrm.CARPETA_SIN_RADAR)):
            if not f:
                continue
            ruta = REPO / sub / f['archivo']
            if not ruta.exists():
                sys.exit(f'falta {ruta.relative_to(REPO)}: quítalo con --quitar o súbelo')
            nuevo = mrm.entrada(ruta)
            if (nuevo['sha256'], nuevo['bytes']) != (f.get('sha256'), f.get('bytes')):
                f.update(sha256=nuevo['sha256'], bytes=nuevo['bytes'])
                d['fecha'] = hoy
                print(f"  cambiado: {f['archivo']}")
    guardar_lista(datos)


def quitar(nombre):
    datos = leer_lista()
    antes = len(datos['modelos'])
    datos['modelos'] = [d for d in datos['modelos'] if d['archivo'] != nombre]
    if len(datos['modelos']) == antes:
        sys.exit(f'{nombre} no está en la lista')
    guardar_lista(datos)
    print(f'  quitado de la lista: {nombre} (el fichero sigue en la carpeta; quien ya lo '
          'descargó lo conserva)')


def diferencias():
    """Lo que no coincide entre la lista y los ficheros del repositorio ([] si nada)."""
    datos = leer_lista()
    base = LISTA.resolve().as_uri()
    malos = []
    for d in datos['modelos']:
        for f, sub in ((d, ''), (d.get('sin_radar'), mrm.CARPETA_SIN_RADAR)):
            if not f:
                continue
            try:
                mrm.fichero(f, base)
            except ValueError as e:
                malos.append(str(e))
                continue
            ruta = REPO / (f.get('url') or f['archivo'])
            if not ruta.exists():
                malos.append(f"falta {f.get('url') or f['archivo']}")
            elif mrm.entrada(ruta)['sha256'] != f['sha256']:
                malos.append(f"{f['archivo']}: la huella de la lista no es la del fichero "
                             '(py publicar_modelo.py)')
    return malos


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('modelo', nargs='?', help='fichero .joblib que se publica')
    ap.add_argument('--nombre', default='', help='nombre que se enseña (zona y fuentes)')
    ap.add_argument('--sin-radar', help='su versión sin radar (<nombre>_sin_radar.joblib)')
    ap.add_argument('--quitar', metavar='MODELO', help='quitar un modelo de la lista')
    ap.add_argument('--comprobar', action='store_true',
                    help='solo decir si la lista coincide con los ficheros')
    a = ap.parse_args()
    if a.comprobar:
        malos = diferencias()
        print('\n'.join(malos) if malos else 'La lista coincide con los ficheros.')
        sys.exit(1 if malos else 0)
    if a.quitar:
        quitar(a.quitar)
    elif a.modelo:
        anadir(a.modelo, a.nombre, a.sin_radar)
    else:
        recalcular()
    print(f'Listo. Sube {mrm.MANIFIESTO} y los modelos a GitHub (git add, commit y push).')


if __name__ == '__main__':
    main()
