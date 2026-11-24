"""
retrain_v6_grupos.py — Reentrena los modelos de Sierra Nevada (BASE y CIR) con UNA
variable por grupo de variables repetidas. Datos: classification_v6.csv (796 puntos de
All_borreguil_lagunas+500ptos, EVI normal; 769 sin duda).

Grupos: agrupamiento jerárquico por enlace COMPLETO con la correlación de Spearman, de
modo que dentro de un grupo CADA pareja de variables tiene |r| ≥ 0,8 (dicen lo mismo).
De cada grupo queda la que mejor separa borreguil de no borreguil por sí sola (AUC
univariante más lejos de 0,5). Las que no se parecen a ninguna, quedan todas.

Comprobado el 19-11-2026 (validación por cuencas, 5 semillas):
  BASE  50 → 28 variables: AUC 0,827 ± 0,002 → 0,838 ± 0,001; misma decisión (0,5) en
        el 93,7 % de los puntos; con las variables de Earth Engine, 0,834 → 0,843.
  CIR   56 → 33 variables: AUC 0,934 ± 0,001 → 0,940 ± 0,001; misma decisión en el
        97,2 %; con Earth Engine, 0,934 → 0,939.

Antes de sobrescribir, guarda los modelos anteriores (todas las variables) en backups/
con el sufijo `_v6_todas`.
"""
import csv
import os
import shutil

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

import retrain_v5_win as rt

CSV_IN = os.path.join(rt.BASE, 'classification_v6.csv')
UMBRAL = 0.8
ZONA = ('Sierra Nevada (Andalucía) — v6: 796 pts, offset BOA corregido, EVI normal, '
        'una variable por grupo, presencia/ausencia')


def una_por_grupo(rows, train, y, feats, umbral=UMBRAL):
    """[(representante, [variables del grupo], |r| mínimo dentro)], en el orden de feats."""
    X, _ = rt.build(rows, feats)
    X = X[train]
    rho = np.nan_to_num(spearmanr(X).correlation, nan=0.0)
    d = 1 - np.abs(rho)
    np.fill_diagonal(d, 0)
    etiq = fcluster(linkage(squareform(d, checks=False), method='complete'),
                    t=1 - umbral, criterion='distance')
    grupos = []
    for g in dict.fromkeys(etiq):                     # en el orden de feats
        idx = [i for i in range(len(feats)) if etiq[i] == g]
        sep = {feats[i]: abs(roc_auc_score(y, X[:, i]) - 0.5) for i in idx}
        rep = max(sep, key=sep.get)
        sub = np.abs(rho[np.ix_(idx, idx)])
        min_r = float(sub[np.triu_indices(len(idx), 1)].min()) if len(idx) > 1 else 1.0
        grupos.append((rep, [feats[i] for i in idx], min_r))
    return grupos


def main():
    with open(CSV_IN, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    y_all = np.array([1 if r.get('gt_borreguil') == 'si'
                      else 0 if r.get('gt_borreguil') == 'no' else -1 for r in rows])
    duda = np.array([str(r.get('gt_duda')).lower() == 'si' for r in rows])
    cuenca = np.array([r.get('cuenca_id', '') for r in rows])
    train = (y_all != -1) & (~duda)
    y = y_all[train]; groups = cuenca[train]

    os.makedirs(os.path.join(rt.BASE, 'backups'), exist_ok=True)
    for nombre in ('rf_sierra_nevada.joblib', 'rf_sierra_nevada_cir.joblib'):
        src = os.path.join(rt.BASE, nombre)
        dst = os.path.join(rt.BASE, 'backups', nombre.replace('.joblib', '_v6_todas.joblib'))
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            print(f'  copia del anterior -> backups/{os.path.basename(dst)}')

    ref = rt.compute_ref_stats(rows)
    ref['source'] = 'classification_v6.csv — Sierra Nevada v6 (796 pts, EVI normal)'
    for todas, nombre, zona in ((rt.F50, 'rf_sierra_nevada.joblib', ZONA),
                                (rt.F56, 'rf_sierra_nevada_cir.joblib', ZONA + ' + CIR 0,25 m')):
        grupos = una_por_grupo(rows, train, y, list(todas))
        feats = [g[0] for g in grupos]
        print(f'\n{nombre}: {len(todas)} → {len(feats)} variables')
        for rep, vs, min_r in grupos:
            if len(vs) > 1:
                print(f'  queda {rep} (|r| ≥ {min_r:.3f} en el grupo); quita: '
                      + ', '.join(v for v in vs if v != rep))
        X, _ = rt.build(rows, feats)
        auc = rt.cv_auc(X[train], y, groups)
        print(f'  AUC por cuencas: {auc:.4f}')
        rt.save_variant(rows, feats, train, y, groups, nombre, zona, auc, ref)


if __name__ == '__main__':
    main()
