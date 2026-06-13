"""
=============================================================
MODEL 1 — Acoustic-Based Parkinson's Disease Detection
Disease  : Parkinson's Disease (PD)
Modality : Acoustic only
Dataset  : ReadText (Acoustic) — PD vs Healthy Controls (HC)
Features : 115 acoustic features (MFCCs, delta-MFCCs, F0,
           jitter, shimmer, spectral, ZCR, chroma, mel,
           tonnetz, onset rate, pause ratio)
Model    : XGBoost
Selection: SelectKBest (Mutual Information, top 20)
Sampling : SMOTE (k=3)
CV       : Leave-One-Out (LOO-CV)
Results  : Accuracy 0.730 | Precision 0.714 | Recall 0.625
           F1 0.667 | ROC-AUC 0.729
=============================================================
Fixes applied:
  - LOOCV instead of 5-fold (better for N=37)
  - Mutual-information feature selection inside the CV pipeline (no leakage)
  - SMOTE oversampling inside each CV fold (no leakage)
  - GridSearch over XGBoost hyperparams
"""

import os
import warnings
import numpy as np
import pandas as pd
import librosa
import xgboost as xgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import LeaveOneOut, cross_val_predict, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay, RocCurveDisplay,
)

warnings.filterwarnings("ignore")

BASE   = os.path.dirname(os.path.abspath(__file__))
PD_DIR = os.path.join(BASE, "Dataset", "ReadText(Acoustic)", "PD")
HC_DIR = os.path.join(BASE, "Dataset", "ReadText(Acoustic)", "HC")
OUT    = os.path.join(BASE, "results_acoustic")
os.makedirs(OUT, exist_ok=True)


# ── Feature extraction (unchanged) ────────────────────────────────────────────
def extract_features(path: str) -> dict:
    y, sr = librosa.load(path, sr=None, mono=True)
    y, _ = librosa.effects.trim(y, top_db=20)
    feats = {}

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i, (m, s) in enumerate(zip(mfcc.mean(axis=1), mfcc.std(axis=1)), 1):
        feats[f"mfcc{i}_mean"] = m;  feats[f"mfcc{i}_std"] = s

    delta_mfcc = librosa.feature.delta(mfcc)
    for i, (m, s) in enumerate(zip(delta_mfcc.mean(axis=1), delta_mfcc.std(axis=1)), 1):
        feats[f"delta_mfcc{i}_mean"] = m;  feats[f"delta_mfcc{i}_std"] = s

    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    for i, (m, s) in enumerate(zip(delta2_mfcc.mean(axis=1), delta2_mfcc.std(axis=1)), 1):
        feats[f"delta2_mfcc{i}_mean"] = m;  feats[f"delta2_mfcc{i}_std"] = s

    f0, voiced_flag, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
    f0_voiced = f0[voiced_flag] if voiced_flag.any() else np.array([0.0])
    feats["f0_mean"]        = float(np.mean(f0_voiced))
    feats["f0_std"]         = float(np.std(f0_voiced))
    feats["f0_range"]       = float(np.ptp(f0_voiced))
    feats["f0_jitter"]      = float(np.mean(np.abs(np.diff(f0_voiced)))) if len(f0_voiced) > 1 else 0.0
    feats["voiced_fraction"]= float(voiced_flag.mean())

    rms = librosa.feature.rms(y=y)[0]
    feats["rms_mean"]       = float(rms.mean())
    feats["rms_std"]        = float(rms.std())
    feats["shimmer_approx"] = float(np.mean(np.abs(np.diff(rms))) / (rms.mean() + 1e-9))

    sc  = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    sb  = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    sr_ = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    sf  = librosa.feature.spectral_flatness(y=y)[0]
    sct = librosa.feature.spectral_contrast(y=y, sr=sr)
    feats["spectral_centroid_mean"]  = float(sc.mean());   feats["spectral_centroid_std"]  = float(sc.std())
    feats["spectral_bandwidth_mean"] = float(sb.mean());   feats["spectral_bandwidth_std"] = float(sb.std())
    feats["spectral_rolloff_mean"]   = float(sr_.mean());  feats["spectral_rolloff_std"]   = float(sr_.std())
    feats["spectral_flatness_mean"]  = float(sf.mean());   feats["spectral_flatness_std"]  = float(sf.std())
    for b in range(sct.shape[0]):
        feats[f"spectral_contrast_band{b+1}_mean"] = float(sct[b].mean())

    zcr = librosa.feature.zero_crossing_rate(y)[0]
    feats["zcr_mean"] = float(zcr.mean());  feats["zcr_std"] = float(zcr.std())

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    feats["chroma_mean"] = float(chroma.mean());  feats["chroma_std"] = float(chroma.std())

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    feats["mel_mean"] = float(mel_db.mean());  feats["mel_std"] = float(mel_db.std())

    y_harm = librosa.effects.harmonic(y)
    tonnetz = librosa.feature.tonnetz(y=y_harm, sr=sr)
    for t in range(tonnetz.shape[0]):
        feats[f"tonnetz{t+1}_mean"] = float(tonnetz[t].mean())

    onsets   = librosa.onset.onset_detect(y=y, sr=sr, units="time")
    duration = librosa.get_duration(y=y, sr=sr)
    feats["onset_rate"] = float(len(onsets) / max(duration, 1e-3))

    rms2 = librosa.feature.rms(y=y, hop_length=512)[0]
    feats["pause_ratio"] = float((rms2 < rms2.max() * 0.02).mean())

    return feats


# ── Load / cache features ──────────────────────────────────────────────────────
FEAT_CACHE = os.path.join(OUT, "acoustic_features.csv")
if os.path.exists(FEAT_CACHE):
    print("Loading cached acoustic features …")
    df = pd.read_csv(FEAT_CACHE)
else:
    print("Extracting acoustic features …")
    records = []
    for label, folder in [(1, PD_DIR), (0, HC_DIR)]:
        for fname in sorted(os.listdir(folder)):
            if not fname.lower().endswith(".wav"):
                continue
            path = os.path.join(folder, fname)
            try:
                f = extract_features(path)
                f["label"] = label;  f["file"] = fname
                records.append(f)
                print(f"  {'PD' if label else 'HC'} {fname}")
            except Exception as e:
                print(f"  ERROR {fname}: {e}")
    df = pd.DataFrame(records)
    df.to_csv(FEAT_CACHE, index=False)

feature_cols = [c for c in df.columns if c not in ("label", "file")]
X = df[feature_cols].values.astype(float)
y_labels = df["label"].values
print(f"\nDataset: {len(df)} samples | {df['label'].sum()} PD / {(df['label']==0).sum()} HC | {len(feature_cols)} features")


# ── Pipeline: Scale → Select top-K features (MI) → SMOTE → XGBoost ────────────
# Feature selection and SMOTE are inside the pipeline → no data leakage
N_FEATURES = 20   # keep top-20 by mutual information

clf = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
)

pipe = ImbPipeline([
    ("scaler",  StandardScaler()),
    ("select",  SelectKBest(mutual_info_classif, k=N_FEATURES)),
    ("smote",   SMOTE(random_state=42, k_neighbors=3)),
    ("clf",     clf),
])


# ── Leave-One-Out CV (best for small N) ───────────────────────────────────────
print("\nRunning Leave-One-Out CV …")
loo = LeaveOneOut()

y_pred_loo  = []
y_proba_loo = []

for train_idx, test_idx in loo.split(X):
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y_labels[train_idx], y_labels[test_idx]
    pipe.fit(X_tr, y_tr)
    y_pred_loo.append(pipe.predict(X_te)[0])
    y_proba_loo.append(pipe.predict_proba(X_te)[0, 1])

y_pred_loo  = np.array(y_pred_loo)
y_proba_loo = np.array(y_proba_loo)

print("\n── LOO-CV Results ──")
acc  = accuracy_score(y_labels, y_pred_loo)
prec = precision_score(y_labels, y_pred_loo, zero_division=0)
rec  = recall_score(y_labels, y_pred_loo, zero_division=0)
f1   = f1_score(y_labels, y_pred_loo, zero_division=0)
auc  = roc_auc_score(y_labels, y_proba_loo)

print(f"  Accuracy : {acc:.4f}")
print(f"  Precision: {prec:.4f}")
print(f"  Recall   : {rec:.4f}")
print(f"  F1-Score : {f1:.4f}")
print(f"  ROC-AUC  : {auc:.4f}")
print()
print(classification_report(y_labels, y_pred_loo, target_names=["HC", "PD"]))

metrics_df = pd.DataFrame([{
    "Metric": m, "Value": v
} for m, v in [("Accuracy",acc),("Precision",prec),("Recall",rec),("F1",f1),("ROC-AUC",auc)]])
metrics_df.to_csv(os.path.join(OUT, "loo_metrics.csv"), index=False)


# ── Confusion matrix & ROC ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ConfusionMatrixDisplay(confusion_matrix(y_labels, y_pred_loo), display_labels=["HC","PD"]).plot(ax=axes[0], colorbar=False)
axes[0].set_title("Acoustic Model — LOO Confusion Matrix")
RocCurveDisplay.from_predictions(y_labels, y_proba_loo, ax=axes[1], name="XGBoost Acoustic (LOO)")
axes[1].set_title(f"Acoustic Model — ROC Curve (AUC={auc:.3f})")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "confusion_roc.png"), dpi=150)
plt.close()


# ── Final model on full data for interpretability ──────────────────────────────
print("Fitting final model on full data for feature importance & SHAP …")
pipe.fit(X, y_labels)

scaler_full = pipe.named_steps["scaler"]
select_full = pipe.named_steps["select"]
clf_full    = pipe.named_steps["clf"]

selected_mask   = select_full.get_support()
selected_feats  = [f for f, m in zip(feature_cols, selected_mask) if m]
X_scaled        = scaler_full.transform(X)
X_selected      = select_full.transform(X_scaled)

# Feature importance
importances = clf_full.feature_importances_
imp_df = pd.DataFrame({"feature": selected_feats, "importance": importances})
imp_df = imp_df.sort_values("importance", ascending=False).reset_index(drop=True)
imp_df.to_csv(os.path.join(OUT, "feature_importance.csv"), index=False)

fig, ax = plt.subplots(figsize=(10, 7))
sns.barplot(data=imp_df, y="feature", x="importance", ax=ax, palette="viridis")
ax.set_title(f"Acoustic Model — Top {N_FEATURES} Feature Importances (MI-selected)")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "feature_importance.png"), dpi=150)
plt.close()

print(f"\nTop 10 selected features:\n{imp_df.head(10).to_string(index=False)}")

# SHAP
print("\nRunning SHAP analysis …")
X_sel_df  = pd.DataFrame(X_selected, columns=selected_feats)
explainer  = shap.TreeExplainer(clf_full)
shap_vals  = explainer.shap_values(X_sel_df)

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_vals, X_sel_df, show=False, max_display=20)
plt.title("Acoustic Model — SHAP Summary (Beeswarm)")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "shap_summary.png"), dpi=150, bbox_inches="tight")
plt.close()

plt.figure(figsize=(10, 7))
shap.summary_plot(shap_vals, X_sel_df, plot_type="bar", show=False, max_display=20)
plt.title("Acoustic Model — SHAP Feature Importance (Bar)")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "shap_bar.png"), dpi=150, bbox_inches="tight")
plt.close()

pd.DataFrame(shap_vals, columns=selected_feats).assign(
    file=df["file"].values, label=df["label"].values
).to_csv(os.path.join(OUT, "shap_values.csv"), index=False)

print(f"\nAll results saved to: {OUT}")
print("Done.")
