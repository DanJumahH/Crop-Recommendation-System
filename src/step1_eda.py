"""
=============================================================
  MENDELEY CROP RECOMMENDATION SYSTEM
  Step 1: Data Loading & Exploratory Data Analysis (EDA)
  Group 18 | Kwara State University
=============================================================
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ── Load dataset ──────────────────────────────────────────
RAW_PATH = 'data/raw/crop_dataset.csv'
df = pd.read_csv(RAW_PATH)

print("=" * 60)
print("   STEP 1: DATA LOADING & EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# ── 1.1 Basic Info ────────────────────────────────────────
print(f"\n📦 Dataset Shape   : {df.shape}")
print(f"   Rows            : {df.shape[0]}")
print(f"   Columns         : {df.shape[1]}")

print(f"\n📋 Column Names:")
for i, col in enumerate(df.columns, 1):
    print(f"   {i:2}. {col}")

print(f"\n🔢 Data Types:")
print(df.dtypes.to_string())

# ── 1.2 Missing Values ────────────────────────────────────
print(f"\n❓ Missing Values:")
missing = df.isnull().sum()
if missing.sum() == 0:
    print("   No missing values found ✅")
else:
    print(missing[missing > 0].to_string())

# ── 1.3 Duplicate Rows ───────────────────────────────────
dupes = df.duplicated().sum()
print(f"\n🔁 Duplicate Rows  : {dupes}")

# ── 1.4 Class Distribution ───────────────────────────────
print(f"\n🌾 Crop Class Distribution:")
counts = df['label'].value_counts()
for crop, count in counts.items():
    bar = '█' * (count // 50)
    print(f"   {crop:<14} {count:>5}  {bar}")

print(f"\n⚖️  Class Imbalance:")
print(f"   Most common  : {counts.idxmax()} ({counts.max()} rows)")
print(f"   Least common : {counts.idxmin()} ({counts.min()} rows)")
print(f"   Imbalance ratio : {counts.max()/counts.min():.1f}x")
print(f"   Total classes   : {df['label'].nunique()}")

# ── 1.5 Fallow Class Note ────────────────────────────────
print(f"\n⚠️  Fallow Class:")
fallow_count = counts.get('Fallow', 0)
print(f"   Fallow has {fallow_count} rows")
print(f"   → Will be REMOVED in Step 2 (per methodology)")

# ── 1.6 Soilcolor Analysis ───────────────────────────────
print(f"\n🎨 Soilcolor — Unique Values ({df['Soilcolor'].nunique()}):")
for val, cnt in df['Soilcolor'].value_counts().items():
    print(f"   '{val}' → {cnt} rows")
print(f"   → Will be cleaned and One-Hot Encoded in Step 3")

# ── 1.7 Numeric Feature Summary ──────────────────────────
print(f"\n📊 Numeric Feature Statistics:")
numeric_df = df.select_dtypes(include=[np.number])
print(numeric_df.describe().T[['mean','std','min','max']].round(3).to_string())

# ── 1.8 Outlier Check ────────────────────────────────────
print(f"\n🔍 Outlier Check (values beyond 3 std from mean):")
for col in ['K', 'P', 'Zn', 'S', 'Ph']:
    mean = df[col].mean()
    std  = df[col].std()
    out  = ((df[col] < mean - 3*std) | (df[col] > mean + 3*std)).sum()
    if out > 0:
        print(f"   {col:<8} : {out} outliers  "
              f"(max={df[col].max():.2f}, mean={mean:.2f})")

# ── 1.9 Feature Groups ───────────────────────────────────
print(f"\n📂 Feature Groups:")
print(f"   Soil nutrients : Ph, K, P, N, Zn, S")
print(f"   Soil color     : Soilcolor (categorical)")
print(f"   Climate W      : QV2M-W, T2M_MAX-W, T2M_MIN-W, PRECTOTCORR-W")
print(f"   Climate Sp     : QV2M-Sp, T2M_MAX-Sp, T2M_MIN-Sp, PRECTOTCORR-Sp")
print(f"   Climate Su     : QV2M-Su, T2M_MAX-Su, T2M_MIN-Su, PRECTOTCORR-Su")
print(f"   Climate Au     : QV2M-Au, T2M_MAX-Au, T2M_MIN-Au, PRECTOTCORR-Au")
print(f"   Other climate  : WD10M, GWETTOP, CLOUD_AMT, WS2M_RANGE, PS")
print(f"   Topographic    : elevation, slope, aspect (to be added in Step 2A)")

print(f"\n✅ Step 1 Complete.")
print(f"   Next → Run: python src/step2a_nasadem.py")
