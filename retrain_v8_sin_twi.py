"""
retrain_v8_sin_twi.py — Los modelos de Sierra Nevada dejan de usar el TWI (índice
topográfico de humedad).

Earth Engine no calcula el TWI (hay que seguir el agua cuesta abajo por todo el
relieve), así que con Earth Engine la app lo rellenaba siempre con el valor típico. Con
validación por cuencas y 5 semillas (20-11-2026), el modelo acierta lo mismo sin él:

    BASE  con TWI 0,846 · TWI rellenado (Earth Engine) 0,844 · sin TWI 0,848
    CIR   0,945 en los tres casos

Datos: classification_v7.csv (796 puntos, radar de Earth Engine). Misma receta que la
v7: una variable por grupo de las que dicen lo mismo (retrain_v6_grupos.una_por_grupo),
ahora sobre las variables sin el TWI (borreguil_pipeline.FEATURES_RF ya no lo tiene).

    py -X utf8 retrain_v8_sin_twi.py            # solo compara
    py -X utf8 retrain_v8_sin_twi.py --guardar  # además, reentrena y guarda los 4 modelos

Con --guardar, antes de sobrescribir copia los modelos con TWI en backups/ con el
sufijo `_con_twi`, y guarda los nuevos: BASE y CIR, y sus versiones sin radar
(sin_radar/).
"""
import os
import shutil
import sys

import numpy as np

import retrain_v5_win as rt
import retrain_v6_grupos as rg
import retrain_v7_s1gee as r7

ZONA = r7.ZONA.replace('v7:', 'v8:').replace('una variable por grupo',
                                              'sin TWI, una variable por grupo')
MODELOS = (('rf_sierra_nevada.joblib', rt.F50, ZONA),
           ('rf_sierra_nevada_cir.joblib', rt.F56, ZONA + ' + CIR 0,25 m'))
ZONA_SR = ZONA.replace('radar de Earth Engine', 'sin radar (para cuando no se lee)')
MODELOS_SIN_RADAR = (
    (os.path.join(r7.CARPETA_SIN_RADAR, 'rf_sierra_nevada_sin_radar.joblib'), rt.F50, ZONA_SR),
    (os.path.join(r7.CARPETA_SIN_RADAR, 'rf_sierra_nevada_cir_sin_radar.joblib'), rt.F56,
     ZONA_SR + ' + CIR 0,25 m'))


def con_twi(todas):
    """Las mismas variables con el TWI en su sitio de antes (detrás de la pendiente)."""
    i = todas.index('slope_deg') + 1
    return todas[:i] + ['twi'] + todas[i:]


def comparar(v7, train, y, groups):
    for nombre, todas in (('BASE', rt.F50), ('CIR', rt.F56)):
        f1, a1, s1, p1 = r7.evaluar(v7, con_twi(todas), train, y, groups)
        f0, a0, s0, p0 = r7.evaluar(v7, todas, train, y, groups)
        print(f'\n{nombre}')
        print(f'  con TWI: {len(f1)} variables · AUC {a1:.3f} ± {s1:.3f}')
        print(f'  sin TWI: {len(f0)} variables · AUC {a0:.3f} ± {s0:.3f}')
        print(f'  misma decisión (0,5): {np.mean((p1 >= 0.5) == (p0 >= 0.5)) * 100:.1f} %')
        print(f'  cambian las variables: fuera {sorted(set(f1) - set(f0))}, '
              f'entran {sorted(set(f0) - set(f1))}')


def copia_con_twi():
    for nombre, _, _ in MODELOS + MODELOS_SIN_RADAR:
        src = os.path.join(rt.BASE, nombre)
        dst = os.path.join(rt.BASE, 'backups', nombre.replace('.joblib', '_con_twi.joblib'))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            print(f'  copia del anterior -> {os.path.relpath(dst, rt.BASE)}')


def guardar(v7, train, y, groups):
    print('\nGuardando…')
    copia_con_twi()
    os.makedirs(os.path.join(rt.APP, r7.CARPETA_SIN_RADAR), exist_ok=True)
    ref = rt.compute_ref_stats(v7)
    ref['source'] = 'classification_v7.csv — Sierra Nevada v8 (796 pts, sin TWI)'
    for nombre, todas, zona in MODELOS + MODELOS_SIN_RADAR:
        if r7.CARPETA_SIN_RADAR in nombre:
            todas = [f for f in todas if f not in r7.S1]
        feats = [g[0] for g in rg.una_por_grupo(v7, train, y, list(todas))]
        X, _ = rt.build(v7, feats)
        auc = rt.cv_auc(X[train], y, groups)
        rt.save_variant(v7, feats, train, y, groups, nombre, zona, auc, ref)


def main():
    assert 'twi' not in rt.F50, 'borreguil_pipeline.FEATURES_RF todavía tiene el TWI'
    v7 = r7.leer(r7.CSV_V7)
    train, y, groups = r7.preparar(v7)
    comparar(v7, train, y, groups)
    if '--guardar' in sys.argv:
        guardar(v7, train, y, groups)


if __name__ == '__main__':
    main()
