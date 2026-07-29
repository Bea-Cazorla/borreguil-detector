"""
retrain_v5_win.py — Reentrena los modelos BASE (50) y CIR (56) desde
classification_v5.csv con la scikit-learn instalada (1.9.0), para eliminar la
corrupción de versión del pickle antiguo (guardado con 1.5.1). Réplica fiel de
train_v5.py con rutas de Windows. Conserva TODAS las variables (incluida la
altitud y las de textura). Backup del modelo anterior y sincroniza a la app.
"""
import csv, os, shutil
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
import borreguil_pipeline as bp

# borreguil_app/ es este directorio; el repo está un nivel arriba
APP = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(APP)
CSV_IN = os.path.join(BASE, 'classification_v5.csv')

F50 = list(bp.FEATURES_RF)              # 50 base
F56 = list(bp.FEATURES_RF_CIR)          # 50 + 6 CIR


def to_f(v):
    if v is None or v == '':
        return np.nan
    try:
        return float(v)
    except Exception:
        return np.nan


CHART_FEATS = [
    'ndvi_early', 'ndvi_late', 'clre_early', 'clre_late', 'ndmi_early', 'ndmi_late',
    'evi_early', 'evi_late', 'gndvi_early', 'gndvi_late', 'ndvi_drop',
    'ndwi_late', 'nbr_late',
    'ndmi_mean', 'ndmi_min', 'ndmi_max', 'ndwi_mean', 'ndwi_min', 'ndwi_max',
    'wavi_mean', 'wavi_min', 'wavi_max', 'albedo_mean', 'albedo_min', 'albedo_max',
    'cir_ndvi_mean', 'cir_ndvi_p90',
]
_PCTS = [('p10', 10), ('p25', 25), ('p50', 50), ('p75', 75), ('p90', 90)]


def compute_ref_stats(rows):
    y = np.array([r.get('gt_borreguil') for r in rows])
    duda = np.array([str(r.get('gt_duda')).lower() == 'si' for r in rows])
    ref_feats = list(dict.fromkeys(list(CHART_FEATS) + list(bp.FEATURES_RF_CIR)))
    feats = {}
    for f in ref_feats:
        col = np.array([to_f(r.get(f)) for r in rows], dtype=float)
        d = {}
        for cls in ('si', 'no'):
            m = (y == cls) & (~duda) & ~np.isnan(col)
            if int(m.sum()) >= 10:
                vals = col[m]
                d[cls] = {p: float(np.nanpercentile(vals, q)) for p, q in _PCTS}
                d[cls]['n'] = int(m.sum())
        if d:
            feats[f] = d
    return {
        'features': feats,
        'n_si': int(((y == 'si') & (~duda)).sum()),
        'n_no': int(((y == 'no') & (~duda)).sum()),
        'source': 'classification_v5.csv — Sierra Nevada v5 (796 pts, offset BOA corregido)',
    }


def build(rows, feats):
    X = np.array([[to_f(r.get(f)) for f in feats] for r in rows], dtype=np.float32)
    med = np.nanmedian(X, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    for j in range(X.shape[1]):
        m = np.isnan(X[:, j]); X[m, j] = med[j]
    return X, med


def rf():
    return RandomForestClassifier(n_estimators=400, max_depth=12, min_samples_split=4,
                                  min_samples_leaf=2, class_weight='balanced',
                                  random_state=42, n_jobs=-1)


def cv_auc(X, y, groups):
    pred = cross_val_predict(rf(), X, y, cv=GroupKFold(n_splits=5), groups=groups,
                             method='predict_proba', n_jobs=-1)[:, 1]
    return roc_auc_score(y, pred)


def save_variant(rows, feats, train, y, groups, name, zone, auc, ref_stats=None):
    X, med = build(rows, feats)
    clf = rf(); clf.fit(X[train], y)
    y_all = np.array([1 if r.get('gt_borreguil') == 'si'
                      else 0 if r.get('gt_borreguil') == 'no' else -1 for r in rows])
    lab = y_all != -1
    proba_all = clf.predict_proba(X)[:, 1]
    auc_train = roc_auc_score(y_all[lab], proba_all[lab])
    bundle = {
        'model': clf, 'features': feats, 'medians': med.tolist(),
        'auc_groupkfold': round(float(auc), 4), 'auc_train_eval': round(float(auc_train), 4),
        'training_zone': zone, 'training_n': int(train.sum()),
        'n_positives': int((y == 1).sum()), 'n_negatives': int((y == 0).sum()),
        'ref_stats': ref_stats,
        'importances': {ff: float(w) for ff, w in zip(feats, clf.feature_importances_)},
        'sklearn_retrained': True,
    }
    path = os.path.join(BASE, name)
    os.makedirs(os.path.join(BASE, 'backups'), exist_ok=True)
    backup = os.path.join(BASE, 'backups', name.replace('.joblib', '_sklearn15.joblib'))
    if os.path.exists(path) and not os.path.exists(backup):
        shutil.copy2(path, backup)
        print(f'  backup del anterior -> backups/{os.path.basename(backup)}')
    bp.save_model_bundle(bundle, path)
    shutil.copy2(path, os.path.join(APP, name))
    print(f'  OK {name}: {len(feats)} feats · AUC honesto {auc:.4f} · guardado + sincronizado')


def main():
    with open(CSV_IN, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    print(f'Filas: {len(rows)}')
    import sklearn
    print(f'scikit-learn: {sklearn.__version__}')
    y_all = np.array([1 if r.get('gt_borreguil') == 'si'
                      else 0 if r.get('gt_borreguil') == 'no' else -1 for r in rows])
    duda = np.array([str(r.get('gt_duda')).lower() == 'si' for r in rows])
    cuenca = np.array([r.get('cuenca_id', '') for r in rows])
    train = (y_all != -1) & (~duda)
    y = y_all[train]; groups = cuenca[train]
    print(f'Entrenamiento (sin dudas): {train.sum()} · si={int((y==1).sum())} · '
          f'no={int((y==0).sum())} · cuencas={len(set(groups))}')
    ref = compute_ref_stats(rows)
    print(f'ref_stats: {len(ref["features"])} features · si={ref["n_si"]} no={ref["n_no"]}\n')

    Xb, _ = build(rows, F50); Xc, _ = build(rows, F56)
    auc50 = cv_auc(Xb[train], y, groups)
    auc56 = cv_auc(Xc[train], y, groups)
    print('=' * 60)
    print(f'  AUC GroupKFold-por-cuenca  BASE (50):  {auc50:.4f}')
    print(f'  AUC GroupKFold-por-cuenca  CIR  (56):  {auc56:.4f}')
    print('=' * 60 + '\n')

    zone = 'Sierra Nevada (Andalucía) — v5: 796 pts, offset BOA corregido, presencia/ausencia'
    save_variant(rows, F50, train, y, groups, 'rf_sierra_nevada.joblib', zone, auc50, ref)
    save_variant(rows, F56, train, y, groups, 'rf_sierra_nevada_cir.joblib',
                 zone + ' + CIR 0,25 m', auc56, ref)


if __name__ == '__main__':
    main()
