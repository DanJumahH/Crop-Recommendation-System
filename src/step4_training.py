"""
=============================================================
  MENDELEY CROP RECOMMENDATION SYSTEM
  Step 4: Model Training
  Group 18 | Kwara State University
=============================================================
  1. BorderlineSMOTE on training set only
  2. Reduced Grid Search for XGBoost hyperparameters
  3. Train final XGBoost with best params
  4. Train Random Forest
  5. Soft Ensemble (XGB + RF)
  6. Save all models
=============================================================
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import BorderlineSMOTE
from xgboost import XGBClassifier
from collections import Counter
import pickle
import os
import time
import warnings
warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────
PROC_PATH = 'data/processed'
MDL_PATH  = 'models'
os.makedirs(MDL_PATH, exist_ok=True)

# ── Load preprocessed data ────────────────────────────────
print("=" * 60)
print("         STEP 4: MODEL TRAINING")
print("=" * 60)

print(f"\n   Loading preprocessed data...")
X_train   = pd.read_csv(f'{PROC_PATH}/X_train.csv')
X_test    = pd.read_csv(f'{PROC_PATH}/X_test.csv')
y_train   = pd.read_csv(f'{PROC_PATH}/y_train.csv').squeeze()
y_test    = pd.read_csv(f'{PROC_PATH}/y_test.csv').squeeze()
le_label  = pickle.load(open(f'{MDL_PATH}/le_label.pkl', 'rb'))
feat_cols = pickle.load(open(f'{MDL_PATH}/feat_cols.pkl', 'rb'))

print(f"   X_train : {X_train.shape}")
print(f"   X_test  : {X_test.shape}")

# ════════════════════════════════════════════════════════
# 4A — BORDERLINE SMOTE
# Applied on training set only (no leakage to test set)
# ════════════════════════════════════════════════════════
print(f"\n--- 4A: BorderlineSMOTE on Training Set ---")
print(f"   Target : bring any class under 500 to 500 samples")
print(f"   Applied on train only — test set untouched\n")

counter = Counter(y_train)
print(f"   Before SMOTE:")
for k, v in sorted(counter.items()):
    print(f"   {le_label.classes_[k]:<14} → {v} rows")

TARGET = 500
sampling_strategy = {k: TARGET for k, v in counter.items()
                     if v < TARGET}

smote = BorderlineSMOTE(
    sampling_strategy=sampling_strategy,
    random_state=42,
    k_neighbors=5
)
X_res, y_res = smote.fit_resample(X_train, y_train)

print(f"\n   After SMOTE:")
counter2 = Counter(y_res)
for k, v in sorted(counter2.items()):
    print(f"   {le_label.classes_[k]:<14} → {v} rows")

print(f"\n   Training set: {X_train.shape[0]} → {X_res.shape[0]} rows")

sw = compute_sample_weight('balanced', y_res)

# ════════════════════════════════════════════════════════
# 4B — REDUCED GRID SEARCH FOR XGBOOST
# Reduced parameter grid to keep training time manageable
# ════════════════════════════════════════════════════════
print(f"\n--- 4B: Reduced Grid Search (XGBoost) ---")
print(f"   This will take 30–60 minutes. Please wait...\n")

param_grid = {
    'n_estimators':  [200, 500],
    'max_depth':     [5, 7],
    'learning_rate': [0.05, 0.1],
    'subsample':     [0.8],
    'colsample_bytree': [0.8],
}

total_combos = (len(param_grid['n_estimators']) *
                len(param_grid['max_depth']) *
                len(param_grid['learning_rate']) *
                len(param_grid['subsample']) *
                len(param_grid['colsample_bytree']))

print(f"   Parameter combinations : {total_combos}")
print(f"   Cross-validation folds : 3")
print(f"   Total fits             : {total_combos * 3}\n")

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

xgb_base = XGBClassifier(
    eval_metric='mlogloss',
    random_state=42,
    n_jobs=-1,
    use_label_encoder=False
)

grid_search = GridSearchCV(
    estimator=xgb_base,
    param_grid=param_grid,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

start = time.time()
grid_search.fit(X_res, y_res, sample_weight=sw)
elapsed = round((time.time() - start) / 60, 1)

print(f"\n   Grid search complete in {elapsed} minutes")
print(f"   Best params   : {grid_search.best_params_}")
print(f"   Best CV score : {grid_search.best_score_*100:.2f}%")

best_params = grid_search.best_params_

# ════════════════════════════════════════════════════════
# 4C — TRAIN FINAL XGBOOST WITH BEST PARAMS
# ════════════════════════════════════════════════════════
print(f"\n--- 4C: Training Final XGBoost (best params) ---")
print(f"   Please wait...\n")

xgb = XGBClassifier(
    **best_params,
    eval_metric='mlogloss',
    early_stopping_rounds=50,
    random_state=42,
    n_jobs=-1,
    use_label_encoder=False
)
xgb.fit(
    X_res, y_res,
    sample_weight=sw,
    eval_set=[(X_test, y_test)],
    verbose=False
)

y_pred_xgb = xgb.predict(X_test)
xgb_train  = accuracy_score(y_train, xgb.predict(X_train))
xgb_test   = accuracy_score(y_test,  y_pred_xgb)

print(f"   XGBoost Train Accuracy : {xgb_train*100:.2f}%")
print(f"   XGBoost Test  Accuracy : {xgb_test*100:.2f}%")
print(f"   Best iteration         : {xgb.best_iteration}")

# ════════════════════════════════════════════════════════
# 4D — TRAIN RANDOM FOREST
# ════════════════════════════════════════════════════════
print(f"\n--- 4D: Training Random Forest ---")
print(f"   Please wait...\n")

rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    class_weight='balanced_subsample',
    min_samples_leaf=1,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_res, y_res)

y_pred_rf = rf.predict(X_test)
rf_train  = accuracy_score(y_train, rf.predict(X_train))
rf_test   = accuracy_score(y_test,  y_pred_rf)

print(f"   RF Train Accuracy : {rf_train*100:.2f}%")
print(f"   RF Test  Accuracy : {rf_test*100:.2f}%")

# ════════════════════════════════════════════════════════
# 4E — SOFT ENSEMBLE (XGBoost + Random Forest)
# ════════════════════════════════════════════════════════
print(f"\n--- 4E: Soft Ensemble (XGB + RF) ---")

prob_xgb   = xgb.predict_proba(X_test)
prob_rf    = rf.predict_proba(X_test)
prob_ens   = (prob_xgb + prob_rf) / 2
y_pred_ens = np.argmax(prob_ens, axis=1)
ens_test   = accuracy_score(y_test, y_pred_ens)

print(f"\n   Ensemble Test Accuracy : {ens_test*100:.2f}%")

# ════════════════════════════════════════════════════════
# 4F — MODEL COMPARISON SUMMARY
# ════════════════════════════════════════════════════════
print(f"\n--- 4F: Model Comparison ---")
print(f"\n   {'Model':<22} {'Train Acc':>10} {'Test Acc':>10}")
print(f"   {'-'*44}")
print(f"   {'XGBoost':<22} {xgb_train*100:>9.2f}% {xgb_test*100:>9.2f}%")
print(f"   {'Random Forest':<22} {rf_train*100:>9.2f}% {rf_test*100:>9.2f}%")
print(f"   {'Ensemble (XGB+RF)':<22} {'—':>10} {ens_test*100:>9.2f}%")

accs  = [xgb_test, rf_test, ens_test]
names = ['XGBoost', 'Random Forest', 'Ensemble (XGB+RF)']
best  = names[int(np.argmax(accs))]
print(f"\n   🏆 Best Model : {best} ({max(accs)*100:.2f}%)")

# ════════════════════════════════════════════════════════
# 4G — SAVE MODELS
# ════════════════════════════════════════════════════════
print(f"\n--- 4G: Saving Models ---")

pickle.dump(xgb,         open(f'{MDL_PATH}/xgb_model.pkl',    'wb'))
pickle.dump(rf,          open(f'{MDL_PATH}/rf_model.pkl',     'wb'))
pickle.dump(best_params, open(f'{MDL_PATH}/best_params.pkl',  'wb'))

print(f"\n   ✅ models/xgb_model.pkl   saved")
print(f"   ✅ models/rf_model.pkl    saved")
print(f"   ✅ models/best_params.pkl saved")
print(f"   Best XGBoost params: {best_params}")

print(f"\n✅ Step 4 Complete.")
print(f"   Next → Run: python src/step5_evaluation.py")
