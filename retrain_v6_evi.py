"""
retrain_v6_evi.py — Reentrena los modelos BASE (50) y CIR (56) de Sierra Nevada con
classification_v6.csv: la misma tabla que la v5 (796 puntos de
All_borreguil_lagunas+500ptos) con el EVI NORMAL.

Hasta el 19-11-2026, el EVI de la tabla se calculó con las bandas en números digitales
(0–10 000), donde el «+1» de la fórmula casi no pesa: salía unas 3 veces mayor que el
EVI de siempre y pasaba de 1. La v6 lleva el EVI de inicio y de fin de verano
recalculado con Earth Engine, igual que lo calcula ahora la app (gee_backend.
_pila_detector), con la reflectancia de 0 a 1. Todo lo demás es idéntico a la v5.

Antes de sobrescribir, guarda los modelos anteriores en backups/ con el sufijo
`_evi_dn`. Mismos parámetros y validación que retrain_v5_win.py (que se reutiliza).
Medido el 19-11-2026: AUC por cuencas 0,828 → 0,830 (50 variables) y 0,933 → 0,934
(56, con CIR).
"""
import csv
import os
import shutil

import numpy as np

import retrain_v5_win as rt

CSV_IN = os.path.join(rt.BASE, 'classification_v6.csv')
ZONA = 'Sierra Nevada (Andalucía) — v6: 796 pts, offset BOA corregido, EVI normal, presencia/ausencia'


def main():
    with open(CSV_IN, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    print(f'Filas: {len(rows)}')
    evi = np.array([rt.to_f(r.get('evi_early')) for r in rows])
    assert np.nanmax(evi) < 1, 'la tabla no lleva el EVI normal'
    y_all = np.array([1 if r.get('gt_borreguil') == 'si'
                      else 0 if r.get('gt_borreguil') == 'no' else -1 for r in rows])
    duda = np.array([str(r.get('gt_duda')).lower() == 'si' for r in rows])
    cuenca = np.array([r.get('cuenca_id', '') for r in rows])
    train = (y_all != -1) & (~duda)
    y = y_all[train]; groups = cuenca[train]

    os.makedirs(os.path.join(rt.BASE, 'backups'), exist_ok=True)
    for nombre in ('rf_sierra_nevada.joblib', 'rf_sierra_nevada_cir.joblib'):
        src = os.path.join(rt.BASE, nombre)
        dst = os.path.join(rt.BASE, 'backups', nombre.replace('.joblib', '_evi_dn.joblib'))
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            print(f'  copia del anterior -> backups/{os.path.basename(dst)}')

    ref = rt.compute_ref_stats(rows)
    ref['source'] = 'classification_v6.csv — Sierra Nevada v6 (796 pts, EVI normal)'
    Xb, _ = rt.build(rows, rt.F50); Xc, _ = rt.build(rows, rt.F56)
    auc50 = rt.cv_auc(Xb[train], y, groups)
    auc56 = rt.cv_auc(Xc[train], y, groups)
    print(f'  AUC por cuencas  BASE (50): {auc50:.4f} · CIR (56): {auc56:.4f}')
    rt.save_variant(rows, rt.F50, train, y, groups, 'rf_sierra_nevada.joblib', ZONA, auc50, ref)
    rt.save_variant(rows, rt.F56, train, y, groups, 'rf_sierra_nevada_cir.joblib',
                    ZONA + ' + CIR 0,25 m', auc56, ref)


if __name__ == '__main__':
    main()
