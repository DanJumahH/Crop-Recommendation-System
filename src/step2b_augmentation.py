"""
=============================================================
  MENDELEY CROP RECOMMENDATION SYSTEM
  Step 2B: Data Augmentation & Feature Engineering
  Group 18 | Kwara State University
=============================================================
  1. Clean Soilcolor (45 variants → 8 clean groups)
  2. Engineer 15 new features
  3. Gaussian noise augmentation (minority → 300 samples)
=============================================================
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────
IN_PATH  = 'data/processed/crop_topo.csv'
OUT_PATH = 'data/processed/crop_augmented.csv'
os.makedirs('data/processed', exist_ok=True)

np.random.seed(42)

# ── Load dataset ──────────────────────────────────────────
df = pd.read_csv(IN_PATH)

print("=" * 60)
print("   STEP 2B: AUGMENTATION & FEATURE ENGINEERING")
print("=" * 60)
print(f"\n   Input : {IN_PATH}")
print(f"   Shape : {df.shape}")
print(f"   Crops : {sorted(df['label'].unique())}")

# ════════════════════════════════════════════════════════
# 2B-1: CLEAN SOILCOLOR
# ════════════════════════════════════════════════════════
print(f"\n--- 2B-1: Cleaning Soilcolor ---")
print(f"   Before: {df['Soilcolor'].nunique()} unique values")

def normalize_soil(s):
    s = str(s).lower().strip()
    s = s.split(';')[0].strip()
    s = (s.replace('broown',    'brown')
          .replace('reddis ',   'reddish ')
          .replace('redish',    'reddish')
          .replace('lihgtish',  'lightish')
          .replace('darkbrown', 'dark brown')
          .replace('perl ',     'pearl ')
          .replace('greyish',   'grayish')
          .strip())
    mapping = {
        'reddish':            'reddish brown',
        'red brown':          'reddish brown',
        'dark reddish brown': 'reddish brown',
        'reddishbrown':       'reddish brown',
        'reddish gray':       'gray',
        'grayish brown':      'gray',
        'grayish brown (gb)': 'gray',
        'dark grayish':       'dark gray',
        'dark grayish brown': 'dark gray',
        'dark greyish brown': 'dark gray',
        'very dark brown':    'dark brown',
        'very dark gray':     'dark gray',
        'dark red':           'red',
        'light red':          'red',
        'lightish brown':     'brown',
        'pearl brown':        'brown',
        'yellowish brown':    'yellow',
        'other':              'brown',
        'replacement of inaccessible target red': 'red',
    }
    return mapping.get(s, s)

df['Soilcolor'] = df['Soilcolor'].apply(normalize_soil)

print(f"   After : {df['Soilcolor'].nunique()} clean groups")
print(f"\n   Final Soilcolor groups:")
for val, cnt in df['Soilcolor'].value_counts().items():
    print(f"   {val:<20} → {cnt} rows")

# ════════════════════════════════════════════════════════
# 2B-2: FEATURE ENGINEERING
# ════════════════════════════════════════════════════════
print(f"\n--- 2B-2: Feature Engineering ---")

# Soil nutrient interactions
df['K_P_ratio']   = df['K']  / (df['P']  + 1e-6)
df['N_K_ratio']   = df['N']  / (df['K']  + 1e-6)
df['NPK_sum']     = df['N']  +  df['P']  + df['K']
df['Ph_N_inter']  = df['Ph'] *  df['N']

# Seasonal temperature ranges
df['T_range_W']   = df['T2M_MAX-W']  - df['T2M_MIN-W']
df['T_range_Sp']  = df['T2M_MAX-Sp'] - df['T2M_MIN-Sp']
df['T_range_Su']  = df['T2M_MAX-Su'] - df['T2M_MIN-Su']
df['T_range_Au']  = df['T2M_MAX-Au'] - df['T2M_MIN-Au']

# Annual aggregates
df['T_MAX_annual'] = df[['T2M_MAX-W','T2M_MAX-Sp',
                          'T2M_MAX-Su','T2M_MAX-Au']].mean(axis=1)
df['T_MIN_annual'] = df[['T2M_MIN-W','T2M_MIN-Sp',
                          'T2M_MIN-Su','T2M_MIN-Au']].mean(axis=1)
df['PREC_annual']  = df[['PRECTOTCORR-W','PRECTOTCORR-Sp',
                          'PRECTOTCORR-Su','PRECTOTCORR-Au']].sum(axis=1)
df['QV2M_annual']  = df[['QV2M-W','QV2M-Sp',
                          'QV2M-Su','QV2M-Au']].mean(axis=1)

# Wet season rainfall ratio
df['wet_ratio'] = (df['PRECTOTCORR-Su'] + df['PRECTOTCORR-Sp']) / \
                  (df['PREC_annual'] + 1e-6)

# Topographic interactions
df['elev_temp_inter']  = df['elevation'] * df['T_MAX_annual']
df['slope_rain_inter'] = df['slope']     * df['PREC_annual']

new_feats = [
    'K_P_ratio','N_K_ratio','NPK_sum','Ph_N_inter',
    'T_range_W','T_range_Sp','T_range_Su','T_range_Au',
    'T_MAX_annual','T_MIN_annual','PREC_annual','QV2M_annual',
    'wet_ratio','elev_temp_inter','slope_rain_inter'
]

print(f"\n   {len(new_feats)} new features added:")
for f in new_feats:
    print(f"   ✅ {f}")
print(f"\n   Total features (excl. label): {df.shape[1] - 1}")

# ════════════════════════════════════════════════════════
# 2B-3: GAUSSIAN NOISE AUGMENTATION
# ════════════════════════════════════════════════════════
print(f"\n--- 2B-3: Gaussian Noise Augmentation ---")
print(f"   Strategy : bring every class to minimum 300 samples")
print(f"   Method   : 3% Gaussian noise on numeric columns")
print(f"   Soilcolor: preserved as-is (categorical)\n")

TARGET    = 300
noise_std = 0.03
num_cols  = [c for c in df.columns if c not in ['label', 'Soilcolor']]

print(f"   Class distribution before augmentation:")
counts = df['label'].value_counts()
for crop, count in counts.items():
    bar = '█' * (count // 50)
    print(f"   {crop:<14} {count:>5}  {bar}")

augmented_frames = [df.copy()]

for crop, count in counts.items():
    if count < TARGET:
        needed   = TARGET - count
        crop_df  = df[df['label'] == crop].copy()
        reps     = (needed // len(crop_df)) + 1
        pool     = pd.concat([crop_df] * reps,
                              ignore_index=True).iloc[:needed].copy()

        noise          = np.random.normal(0, noise_std,
                                          pool[num_cols].shape)
        pool[num_cols] = pool[num_cols] * (1 + noise)
        pool[num_cols] = pool[num_cols].clip(lower=0)

        augmented_frames.append(pool)
        print(f"\n   {crop:<14} {count} → {count + needed}"
              f"  (+{needed} synthetic rows)")
    else:
        print(f"\n   {crop:<14} {count}  (no augmentation needed)")

df_aug = pd.concat(augmented_frames, ignore_index=True)
df_aug = df_aug.sample(frac=1, random_state=42).reset_index(drop=True)

# ── Summary ───────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"   AUGMENTATION SUMMARY")
print(f"{'='*60}")
print(f"\n   Original  rows : 3,841  (after Fallow removal)")
print(f"   Augmented rows : {len(df_aug)}")
print(f"   Total features : {df_aug.shape[1] - 1} (excl. label)")

print(f"\n   New class distribution:")
new_counts = df_aug['label'].value_counts()
for crop, cnt in new_counts.items():
    bar = '█' * (cnt // 50)
    print(f"   {crop:<14} {cnt:>5}  {bar}")

old_ratio = counts.max() / counts.min()
new_ratio = new_counts.max() / new_counts.min()
print(f"\n   Imbalance ratio : {old_ratio:.1f}x → {new_ratio:.1f}x ✅")

# ── Save ──────────────────────────────────────────────────
df_aug.to_csv(OUT_PATH, index=False)
print(f"\n💾 Saved → {OUT_PATH}")
print(f"\n✅ Step 2B Complete.")
print(f"   Next → Run: python src/step3_preprocessing.py")
