"""
=============================================================
  MENDELEY CROP RECOMMENDATION SYSTEM
  Step 3: Preprocessing
  Group 18 | Kwara State University
=============================================================
  1. Outlier capping (IQR method)
  2. One-Hot Encoding for Soilcolor
  3. Label Encoding for crop target
  4. Train/Test split (80/20 stratified)
  5. StandardScaler on numeric features
  6. Save all encoders, scalers and splits
=============================================================
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────
IN_PATH   = 'data/processed/crop_augmented.csv'
PROC_PATH = 'data/processed'
MDL_PATH  = 'models'
os.makedirs(PROC_PATH, exist_ok=True)
os.makedirs(MDL_PATH,  exist_ok=True)

# ── Load dataset ──────────────────────────────────────────
df = pd.read_csv(IN_PATH)

print("=" * 60)
print("        STEP 3: PREPROCESSING")
print("=" * 60)
print(f"\n   Input : {IN_PATH}")
print(f"   Shape : {df.shape}")

# ════════════════════════════════════════════════════════
# 3A — OUTLIER CAPPING (1st–99th percentile)
# ════════════════════════════════════════════════════════
print(f"\n--- 3A: Outlier Capping (1st–99th percentile) ---")

cols_to_cap = ['K', 'P', 'Zn', 'S']
for col in cols_to_cap:
    q_low  = df[col].quantile(0.01)
    q_high = df[col].quantile(0.99)
    before_max = df[col].max()
    df[col] = df[col].clip(lower=q_low, upper=q_high)
    after_max  = df[col].max()
    print(f"   {col:<6} → [{q_low:.3f}, {q_high:.3f}]  "
          f"(max was {before_max:.2f}, now {after_max:.2f})")

# ════════════════════════════════════════════════════════
# 3B — ONE-HOT ENCODING FOR SOILCOLOR
# (per proposal Section 3.2.4)
# ════════════════════════════════════════════════════════
print(f"\n--- 3B: One-Hot Encoding for Soilcolor ---")
print(f"   Soilcolor groups: {sorted(df['Soilcolor'].unique())}")

df = pd.get_dummies(df, columns=['Soilcolor'], prefix='soil')
soil_cols = [c for c in df.columns if c.startswith('soil_')]
print(f"\n   One-hot columns created ({len(soil_cols)}):")
for col in soil_cols:
    print(f"   ✅ {col}")

# ════════════════════════════════════════════════════════
# 3C — LABEL ENCODING FOR CROP TARGET
# ════════════════════════════════════════════════════════
print(f"\n--- 3C: Label Encoding for Crop Target ---")

le_label = LabelEncoder()
df['label_enc'] = le_label.fit_transform(df['label'])

print(f"\n   Crop label encoding:")
for i, cls in enumerate(le_label.classes_):
    count = (df['label_enc'] == i).sum()
    print(f"   {i:2} → {cls:<14} ({count} rows)")

# ════════════════════════════════════════════════════════
# 3D — DEFINE FEATURES AND TARGET
# ════════════════════════════════════════════════════════
print(f"\n--- 3D: Defining Features and Target ---")

feat_cols = [c for c in df.columns if c not in ['label', 'label_enc']]
X = df[feat_cols]
y = df['label_enc']

print(f"\n   Total features : {len(feat_cols)}")
print(f"   Total samples  : {len(X)}")
print(f"\n   Feature list:")
for i, col in enumerate(feat_cols, 1):
    print(f"   {i:2}. {col}")

# ════════════════════════════════════════════════════════
# 3E — TRAIN/TEST SPLIT (80/20 stratified)
# ════════════════════════════════════════════════════════
print(f"\n--- 3E: Train/Test Split (80/20, stratified) ---")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\n   Train set : {X_train.shape[0]} rows")
print(f"   Test  set : {X_test.shape[0]}  rows")
print(f"\n   Train class distribution:")
for i, cls in enumerate(le_label.classes_):
    count = (y_train == i).sum()
    print(f"   {cls:<14} → {count} rows")

# ════════════════════════════════════════════════════════
# 3F — FEATURE SCALING (StandardScaler)
# Fit on train only — transform both train and test
# (prevents data leakage)
# ════════════════════════════════════════════════════════
print(f"\n--- 3F: StandardScaler (fit on train only) ---")

# Scale only numeric columns — leave one-hot columns as 0/1
numeric_feat_cols = [c for c in feat_cols if not c.startswith('soil_')]
onehot_feat_cols  = [c for c in feat_cols if c.startswith('soil_')]

scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train[numeric_feat_cols])
X_test_num  = scaler.transform(X_test[numeric_feat_cols])

# Reconstruct full dataframes
X_train_scaled = pd.DataFrame(X_train_num, columns=numeric_feat_cols)
X_test_scaled  = pd.DataFrame(X_test_num,  columns=numeric_feat_cols)

# Add one-hot columns back (no scaling needed)
X_train_scaled[onehot_feat_cols] = X_train[onehot_feat_cols].values
X_test_scaled[onehot_feat_cols]  = X_test[onehot_feat_cols].values

# Reorder columns to match original feat_cols order
X_train_scaled = X_train_scaled[feat_cols]
X_test_scaled  = X_test_scaled[feat_cols]

print(f"\n   Numeric cols scaled  : {len(numeric_feat_cols)}")
print(f"   One-hot cols kept    : {len(onehot_feat_cols)}")
print(f"   Train mean (numeric) : {X_train_scaled[numeric_feat_cols].mean().mean():.6f} (~0)")
print(f"   Train std  (numeric) : {X_train_scaled[numeric_feat_cols].std().mean():.6f}  (~1)")

# ════════════════════════════════════════════════════════
# 3G — SAVE EVERYTHING
# ════════════════════════════════════════════════════════
print(f"\n--- 3G: Saving All Outputs ---")

X_train_scaled.to_csv(f'{PROC_PATH}/X_train.csv', index=False)
X_test_scaled.to_csv( f'{PROC_PATH}/X_test.csv',  index=False)
y_train.to_csv(        f'{PROC_PATH}/y_train.csv', index=False)
y_test.to_csv(         f'{PROC_PATH}/y_test.csv',  index=False)

pickle.dump(le_label,         open(f'{MDL_PATH}/le_label.pkl',  'wb'))
pickle.dump(scaler,           open(f'{MDL_PATH}/scaler.pkl',    'wb'))
pickle.dump(feat_cols,        open(f'{MDL_PATH}/feat_cols.pkl', 'wb'))
pickle.dump(numeric_feat_cols,open(f'{MDL_PATH}/numeric_feat_cols.pkl', 'wb'))
pickle.dump(onehot_feat_cols, open(f'{MDL_PATH}/onehot_feat_cols.pkl',  'wb'))
pickle.dump(soil_cols,        open(f'{MDL_PATH}/soil_cols.pkl', 'wb'))

print(f"\n   ✅ data/processed/X_train.csv  — {X_train_scaled.shape[0]} rows")
print(f"   ✅ data/processed/X_test.csv   — {X_test_scaled.shape[0]} rows")
print(f"   ✅ data/processed/y_train.csv")
print(f"   ✅ data/processed/y_test.csv")
print(f"   ✅ models/le_label.pkl         — crop label encoder")
print(f"   ✅ models/scaler.pkl           — StandardScaler")
print(f"   ✅ models/feat_cols.pkl        — full feature list")
print(f"   ✅ models/numeric_feat_cols.pkl")
print(f"   ✅ models/onehot_feat_cols.pkl")
print(f"   ✅ models/soil_cols.pkl        — one-hot soil columns")

print(f"\n✅ Step 3 Complete.")
print(f"   Next → Run: python src/step4_training.py")
