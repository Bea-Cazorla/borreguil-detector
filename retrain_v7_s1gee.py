"""
retrain_v7_s1gee.py — El radar (Sentinel-1) de los modelos de Sierra Nevada pasa a
leerse de Earth Engine.

Hasta el 19-11-2026 las 5 variables de radar se leían de Planetary Computer
(sentinel-1-rtc): con solo 20 puntos, más de media hora sin terminar. Ahora se leen de
Earth Engine (gee_backend.fetch_s1_gee, lo mismo que hace la app), en segundos. Es otro
producto (ver gee_backend: COPERNICUS/S1_GRD con la corrección del relieve de Vollrath
et al. 2020), así que hay que recalcularlas en los 796 puntos y reentrenar.

    py -X utf8 retrain_v7_s1gee.py            # calcula la v7 (si falta) y compara
    py -X utf8 retrain_v7_s1gee.py --guardar  # además, reentrena y guarda los modelos
    py -X utf8 retrain_v7_s1gee.py --sin-radar  # solo las versiones sin radar (sin_radar/)

1. classification_v7.csv = classification_v6.csv con las 5 variables de radar de
   Earth Engine (la v6 queda intacta).
2. Compara, con validación por cuencas y 5 semillas, el radar antiguo (v6), el nuevo
   (v7) y sin radar, con una variable por grupo (retrain_v6_grupos.una_por_grupo).
3. Con --guardar: copia los modelos actuales en backups/ con el sufijo `_s1_mpc` y
   guarda los nuevos (BASE y CIR) entrenados con la v7.
"""
import csv
import os
import shutil
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict

import retrain_v5_win as rt
import retrain_v6_grupos as rg

CSV_V6 = os.path.join(rt.BASE, 'classification_v6.csv')
CSV_V7 = os.path.join(rt.BASE, 'classification_v7.csv')
S1 = ['s1_vv_mean', 's1_vh_mean', 's1_vv_sd', 's1_vh_sd', 's1_ratio_mean']
ZONA = ('Sierra Nevada (Andalucía) — v7: 796 pts, offset BOA corregido, EVI normal, '
        'radar de Earth Engine, una variable por grupo, presencia/ausencia')
SEMILLAS = (0, 1, 2, 3, 4)


def leer(ruta):
    with open(ruta, encoding='utf-8') as f:
        return list(csv.DictReader(f))


def crear_v7():
    import gee_backend as geb
    proyecto = os.environ.get('BORREGUIL_GEE_PROJECT', '').strip() or 'ee-ugr-default-project'
    ok, msg = geb.initialize(proyecto)
    print(msg)
    if not ok:
        sys.exit(1)
    rows = leer(CSV_V6)
    pts = [{'lon': float(r['lon']), 'lat': float(r['lat'])} for r in rows]
    lons = [p['lon'] for p in pts]; lats = [p['lat'] for p in pts]
    bbox = (min(lons) - 0.01, min(lats) - 0.01, max(lons) + 0.01, max(lats) + 0.01)
    fallidos = geb.fetch_s1_gee(pts, bbox)
    print(f'  fallidos: {fallidos}')
    for r, p in zip(rows, pts):
        for k in S1:
            v = p.get(k, float('nan'))
            r[k] = '' if v != v else f'{v:.4f}'
    sin = sum(1 for r in rows if r['s1_vv_mean'] == '')
    print(f'  sin radar: {sin}/{len(rows)}')
    with open(CSV_V7, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f'  -> {CSV_V7}')


def preparar(rows):
    y_all = np.array([1 if r.get('gt_borreguil') == 'si'
                      else 0 if r.get('gt_borreguil') == 'no' else -1 for r in rows])
    duda = np.array([str(r.get('gt_duda')).lower() == 'si' for r in rows])
    cuenca = np.array([r.get('cuenca_id', '') for r in rows])
    train = (y_all != -1) & (~duda)
    return train, y_all[train], cuenca[train]


def prob_cv(X, y, groups, semilla):
    clf = RandomForestClassifier(n_estimators=400, max_depth=12, min_samples_split=4,
                                 min_samples_leaf=2, class_weight='balanced',
                                 random_state=semilla, n_jobs=-1)
    return cross_val_predict(clf, X, y, cv=GroupKFold(n_splits=5), groups=groups,
                             method='predict_proba', n_jobs=-1)[:, 1]


def evaluar(rows, todas, train, y, groups):
    feats = [g[0] for g in rg.una_por_grupo(rows, train, y, list(todas))]
    X, _ = rt.build(rows, feats)
    probs = [prob_cv(X[train], y, groups, s) for s in SEMILLAS]
    aucs = [roc_auc_score(y, p) for p in probs]
    return feats, float(np.mean(aucs)), float(np.std(aucs)), np.mean(probs, axis=0)


def main():
    if not os.path.exists(CSV_V7):
        crear_v7()
    v6, v7 = leer(CSV_V6), leer(CSV_V7)
    train, y, groups = preparar(v7)
    for nombre, todas in (('BASE', rt.F50), ('CIR', rt.F56)):
        sin_s1 = [f for f in todas if f not in S1]
        f6, a6, s6, p6 = evaluar(v6, todas, train, y, groups)
        f7, a7, s7, p7 = evaluar(v7, todas, train, y, groups)
        f0, a0, s0, p0 = evaluar(v6, sin_s1, train, y, groups)
        print(f'\n{nombre}')
        print(f'  radar antiguo (Planetary Computer): {len(f6)} variables · AUC {a6:.3f} ± {s6:.3f}')
        print(f'  radar de Earth Engine             : {len(f7)} variables · AUC {a7:.3f} ± {s7:.3f}')
        print(f'  sin radar                         : {len(f0)} variables · AUC {a0:.3f} ± {s0:.3f}')
        print(f'  misma decisión (0,5) antiguo vs Earth Engine: '
              f'{np.mean((p6 >= 0.5) == (p7 >= 0.5)) * 100:.1f} %')
        print(f'  radar de Earth Engine que queda: {[f for f in f7 if f in S1]}')
        print(f'  cambian las variables: fuera {sorted(set(f6) - set(f7))}, '
              f'entran {sorted(set(f7) - set(f6))}')

    if '--guardar' in sys.argv:
        print('\nGuardando…')
        for nombre in ('rf_sierra_nevada.joblib', 'rf_sierra_nevada_cir.joblib'):
            src = os.path.join(rt.BASE, nombre)
            dst = os.path.join(rt.BASE, 'backups', nombre.replace('.joblib', '_s1_mpc.joblib'))
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)
                print(f'  copia del anterior -> backups/{os.path.basename(dst)}')
        ref = rt.compute_ref_stats(v7)
        ref['source'] = 'classification_v7.csv — Sierra Nevada v7 (796 pts, radar de Earth Engine)'
        for todas, nombre, zona in ((rt.F50, 'rf_sierra_nevada.joblib', ZONA),
                                    (rt.F56, 'rf_sierra_nevada_cir.joblib',
                                     ZONA + ' + CIR 0,25 m')):
            feats = [g[0] for g in rg.una_por_grupo(v7, train, y, list(todas))]
            X, _ = rt.build(v7, feats)
            auc = rt.cv_auc(X[train], y, groups)
            rt.save_variant(v7, feats, train, y, groups, nombre, zona, auc, ref)


# Versiones SIN radar, para cuando no se lee (Planetary Computer, o «Saltar Sentinel-2»).
# El modelo con radar, si le falta el radar, acierta bastante menos (AUC 0,796 frente a
# 0,846; 0,927 frente a 0,946 el _cir) que uno entrenado desde el principio sin él
# (0,829; 0,938). La app usa estas versiones sola cuando el radar no se ha leído. Van en
# la carpeta sin_radar/ para que no salgan en el desplegable de modelos.
CARPETA_SIN_RADAR = 'sin_radar'


def guardar_sin_radar():
    v7 = leer(CSV_V7)
    train, y, groups = preparar(v7)
    print('\nVersiones sin radar…')
    for d in (os.path.join(rt.BASE, CARPETA_SIN_RADAR),
              os.path.join(rt.APP, CARPETA_SIN_RADAR),
              os.path.join(rt.BASE, 'backups', CARPETA_SIN_RADAR)):
        os.makedirs(d, exist_ok=True)
    ref = rt.compute_ref_stats(v7)
    ref['source'] = 'classification_v7.csv — Sierra Nevada v7 (796 pts), sin radar'
    zona = ZONA.replace('radar de Earth Engine', 'sin radar (para cuando no se lee)')
    for todas, nombre, z in ((rt.F50, 'rf_sierra_nevada_sin_radar.joblib', zona),
                             (rt.F56, 'rf_sierra_nevada_cir_sin_radar.joblib',
                              zona + ' + CIR 0,25 m')):
        sin_s1 = [f for f in todas if f not in S1]
        feats = [g[0] for g in rg.una_por_grupo(v7, train, y, sin_s1)]
        X, _ = rt.build(v7, feats)
        auc = rt.cv_auc(X[train], y, groups)
        rt.save_variant(v7, feats, train, y, groups,
                        os.path.join(CARPETA_SIN_RADAR, nombre), z, auc, ref)


if __name__ == '__main__':
    if '--sin-radar' in sys.argv:
        guardar_sin_radar()
        sys.exit(0)
    main()
