"""
=============================================================
  MENDELEY CROP RECOMMENDATION SYSTEM
  Step 5: Model Evaluation
  Group 18 | Kwara State University
=============================================================
  1. Final predictions (XGBoost, RF, Ensemble)
  2. Classification report
  3. Confusion matrix
  4. Feature importance (XGBoost)
  5. Per-class accuracy
  6. Save evaluation report
=============================================================
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix)
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────
PROC_PATH = 'data/processed'
MDL_PATH  = 'models'
OUT_PATH  = 'outputs'
os.makedirs(OUT_PATH, exist_ok=True)

# ── Load data and models ──────────────────────────────────
print("=" * 60)
print("          STEP 5: MODEL EVALUATION")
print("=" * 60)

print(f"\n   Loading data and models...")
X_train    = pd.read_csv(f'{PROC_PATH}/X_train.csv')
X_test     = pd.read_csv(f'{PROC_PATH}/X_test.csv')
y_train    = pd.read_csv(f'{PROC_PATH}/y_train.csv').squeeze()
y_test     = pd.read_csv(f'{PROC_PATH}/y_test.csv').squeeze()
xgb        = pickle.load(open(f'{MDL_PATH}/xgb_model.pkl',   'rb'))
rf         = pickle.load(open(f'{MDL_PATH}/rf_model.pkl',    'rb'))
le_label   = pickle.load(open(f'{MDL_PATH}/le_label.pkl',    'rb'))
feat_cols  = pickle.load(open(f'{MDL_PATH}/feat_cols.pkl',   'rb'))
best_params= pickle.load(open(f'{MDL_PATH}/best_params.pkl', 'rb'))

print(f"   ✅ All models and data loaded")

# ════════════════════════════════════════════════════════
# 5A — GENERATE PREDICTIONS
# ════════════════════════════════════════════════════════
print(f"\n--- 5A: Generating Predictions ---")

y_pred_xgb = xgb.predict(X_test)
y_pred_rf  = rf.predict(X_test)

prob_xgb   = xgb.predict_proba(X_test)
prob_rf    = rf.predict_proba(X_test)
prob_ens   = (prob_xgb + prob_rf) / 2
y_pred_ens = np.argmax(prob_ens, axis=1)

xgb_acc = accuracy_score(y_test, y_pred_xgb)
rf_acc  = accuracy_score(y_test, y_pred_rf)
ens_acc = accuracy_score(y_test, y_pred_ens)

print(f"\n   XGBoost  Test Accuracy : {xgb_acc*100:.2f}%")
print(f"   RF       Test Accuracy : {rf_acc*100:.2f}%")
print(f"   Ensemble Test Accuracy : {ens_acc*100:.2f}%")

accs      = [xgb_acc, rf_acc, ens_acc]
preds     = [y_pred_xgb, y_pred_rf, y_pred_ens]
names     = ['XGBoost', 'Random Forest', 'Ensemble']
best_i    = int(np.argmax(accs))
best_pred = preds[best_i]
best_name = names[best_i]
best_acc  = accs[best_i]

print(f"\n   🏆 Best : {best_name} ({best_acc*100:.2f}%)")

# ════════════════════════════════════════════════════════
# 5B — CLASSIFICATION REPORT
# ════════════════════════════════════════════════════════
print(f"\n--- 5B: Classification Report ({best_name}) ---\n")

report = classification_report(
    y_test, best_pred,
    target_names=le_label.classes_
)
print(report)

# ════════════════════════════════════════════════════════
# 5C — CONFUSION MATRIX
# ════════════════════════════════════════════════════════
print(f"\n--- 5C: Confusion Matrix ---\n")

cm    = confusion_matrix(y_test, best_pred)
cm_df = pd.DataFrame(
    cm,
    index=le_label.classes_,
    columns=le_label.classes_
)
print(cm_df.to_string())

print(f"\n   Most Confused Pairs:")
found = False
for i, crop_i in enumerate(le_label.classes_):
    for j, crop_j in enumerate(le_label.classes_):
        if i != j and cm[i, j] >= 5:
            print(f"   {crop_i:<14} predicted as "
                  f"{crop_j:<14} : {cm[i,j]} times")
            found = True
if not found:
    print(f"   No major confusion pairs (threshold: 5)")

# ════════════════════════════════════════════════════════
# 5D — FEATURE IMPORTANCE (XGBoost)
# ════════════════════════════════════════════════════════
print(f"\n--- 5D: Top 20 Feature Importances (XGBoost) ---\n")

importance = pd.Series(
    xgb.feature_importances_,
    index=feat_cols
).sort_values(ascending=False)

print(f"   {'Rank':<6} {'Feature':<28} {'Score':>8}")
print(f"   {'-'*44}")
for i, (feat, score) in enumerate(importance.head(20).items(), 1):
    bar = '█' * int(score * 300)
    print(f"   {i:<6} {feat:<28} {score:.4f}  {bar}")

# ════════════════════════════════════════════════════════
# 5E — PER CLASS ACCURACY
# ════════════════════════════════════════════════════════
print(f"\n--- 5E: Per-Class Accuracy ---\n")

print(f"   {'Crop':<14} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
print(f"   {'-'*44}")
for i, crop in enumerate(le_label.classes_):
    mask    = (y_test == i)
    total   = mask.sum()
    correct = (best_pred[mask] == i).sum()
    acc     = correct / total * 100 if total > 0 else 0
    bar     = '█' * int(acc // 10)
    print(f"   {crop:<14} {correct:>8} {total:>8} "
          f"{acc:>9.1f}%  {bar}")

# ════════════════════════════════════════════════════════
# 5F — PIPELINE SUMMARY
# ════════════════════════════════════════════════════════
print(f"\n--- 5F: Full Pipeline Summary ---")
print(f"""
   Dataset        : Mendeley Ethiopian Crop Dataset
   University     : Kwara State University, Malete
   Group          : 18

   Pipeline Steps :
   ├── Step 1  EDA           : 3,867 rows, 12 crops, 48.5x imbalance
   ├── Step 2A NASADEM       : Real elevation/slope/aspect from SRTM
   ├── Step 2B Augmentation  : Fallow removed, 15 features engineered,
   │                           Gaussian noise → minority classes to 300
   ├── Step 3  Preprocessing : One-Hot Soilcolor, Label crop,
   │                           80/20 split, StandardScaler
   ├── Step 4  Training      : BorderlineSMOTE → 500,
   │                           Grid search → best XGBoost params,
   │                           XGBoost + Random Forest + Ensemble
   └── Step 5  Evaluation    : {best_name} best at {best_acc*100:.2f}%

   Best Params (Grid Search) : {best_params}
   Final Crops (11)          : {list(le_label.classes_)}
""")

# ════════════════════════════════════════════════════════
# 5G — SAVE EVALUATION REPORT
# ════════════════════════════════════════════════════════
print(f"--- 5G: Saving Evaluation Report ---")

with open(f'{OUT_PATH}/evaluation_report.txt', 'w') as f:
    f.write("=" * 60 + "\n")
    f.write("  MENDELEY CROP RECOMMENDATION — EVALUATION REPORT\n")
    f.write("  Group 18 | Kwara State University\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Dataset        : Mendeley Ethiopian Crop Dataset\n")
    f.write(f"Original rows  : 3,867\n")
    f.write(f"After Fallow   : 3,841\n")
    f.write(f"Augmented rows : ~5,400\n")
    f.write(f"Total features : {len(feat_cols)}\n")
    f.write(f"Crop classes   : 11 (Fallow removed)\n\n")
    f.write(f"Best XGBoost Params (Grid Search):\n")
    for k, v in best_params.items():
        f.write(f"  {k}: {v}\n")
    f.write(f"\nModel Accuracies:\n")
    f.write(f"  XGBoost       : {xgb_acc*100:.2f}%\n")
    f.write(f"  Random Forest : {rf_acc*100:.2f}%\n")
    f.write(f"  Ensemble      : {ens_acc*100:.2f}%\n\n")
    f.write(f"Best Model : {best_name} ({best_acc*100:.2f}%)\n\n")
    f.write("Classification Report:\n")
    f.write(report)
    f.write("\n\nConfusion Matrix:\n")
    f.write(cm_df.to_string())
    f.write("\n\nTop 20 Feature Importances (XGBoost):\n")
    for i, (feat, score) in enumerate(importance.head(20).items(), 1):
        f.write(f"  {i:2}. {feat:<28} {score:.4f}\n")
    f.write("\n\nPer-Class Accuracy:\n")
    for i, crop in enumerate(le_label.classes_):
        mask    = (y_test == i)
        total   = mask.sum()
        correct = (best_pred[mask] == i).sum()
        acc     = correct / total * 100 if total > 0 else 0
        f.write(f"  {crop:<14} {correct}/{total}  {acc:.1f}%\n")

print(f"\n   ✅ outputs/evaluation_report.txt saved")
print(f"\n✅ Step 5 Complete.")
print(f"   Next → Build Flask App: python app/app.py")
