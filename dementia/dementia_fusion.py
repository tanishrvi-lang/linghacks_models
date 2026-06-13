"""
=============================================================
MODEL 5 — Fusion v1 (Acoustic + Linguistic) Dementia Detection
Disease  : Dementia
Modality : Acoustic + Linguistic (fusion)
Dataset  : 80% of subjects sampled per class
           (67 dementia / 36 no-dementia → 103 subjects)
Transcript: Whisper base.en (CLI, word timestamps)
Features : 144 combined (115 acoustic + 29 linguistic)
           averaged per subject → MI selection top 25
Model    : XGBoost
Sampling : SMOTE (k=5)
CV       : 5-Fold Stratified CV
Results  : Accuracy 0.631 | Precision 0.699 | Recall 0.761
           F1 0.729 | ROC-AUC 0.559
=============================================================
- 80% of subjects sampled per class (subject-level to avoid leakage)
- Acoustic features: librosa (MFCCs, F0, spectral, ZCR, etc.)
- Linguistic features: Whisper base.en transcription + text analysis
- Features averaged per subject across all their clips
- Pipeline: StandardScaler → SelectKBest (MI) → SMOTE → XGBoost
- 5-fold Stratified CV (N~104 subjects, large enough for k-fold)
- SHAP analysis + full metrics
"""

import os, re, json, subprocess, tempfile, warnings, random
import numpy as np
import pandas as pd
import librosa
import xgboost as xgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay, RocCurveDisplay,
)

warnings.filterwarnings("ignore")
random.seed(42)
np.random.seed(42)

BASE      = os.path.dirname(os.path.abspath(__file__))
DEM_DIR   = os.path.join(BASE, "dementia")
NODEM_DIR = os.path.join(BASE, "nodementia")
OUT       = os.path.join(BASE, "results_fusion")
os.makedirs(OUT, exist_ok=True)

ACOU_CACHE  = os.path.join(OUT, "acoustic_features.csv")
TRANS_CACHE = os.path.join(OUT, "transcripts.csv")
SAMPLE_SEED = 42


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Sample 80% of subjects per class
# ══════════════════════════════════════════════════════════════════════════════
def sample_subjects(folder, label, pct=0.80):
    subjects = sorted([
        d for d in os.listdir(folder)
        if os.path.isdir(os.path.join(folder, d))
    ])
    n = max(1, int(len(subjects) * pct))
    random.seed(SAMPLE_SEED)
    chosen = random.sample(subjects, n)
    rows = []
    for subj in chosen:
        subj_path = os.path.join(folder, subj)
        for fname in sorted(os.listdir(subj_path)):
            if fname.lower().endswith(".wav"):
                rows.append({
                    "subject": subj,
                    "file":    fname,
                    "path":    os.path.join(subj_path, fname),
                    "label":   label,
                })
    return rows, len(subjects), n


print("=" * 60)
print("  DEMENTIA FUSION MODEL")
print("=" * 60)

dem_rows,   dem_total,   dem_chosen   = sample_subjects(DEM_DIR,   1)
nodem_rows, nodem_total, nodem_chosen = sample_subjects(NODEM_DIR, 0)
all_rows = dem_rows + nodem_rows
manifest = pd.DataFrame(all_rows)

print(f"\nDementia    : {dem_chosen}/{dem_total} subjects → {len(dem_rows)} clips")
print(f"No-Dementia : {nodem_chosen}/{nodem_total} subjects → {len(nodem_rows)} clips")
print(f"Total clips : {len(manifest)}")
print(f"Total subjects: {manifest['subject'].nunique()}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Acoustic feature extraction
# ══════════════════════════════════════════════════════════════════════════════
def extract_acoustic(path):
    y, sr = librosa.load(path, sr=None, mono=True)
    y, _  = librosa.effects.trim(y, top_db=20)
    f = {}

    # MFCCs 1-13 + delta + delta-delta
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i, (m, s) in enumerate(zip(mfcc.mean(axis=1), mfcc.std(axis=1)), 1):
        f[f"ac_mfcc{i}_mean"] = m;  f[f"ac_mfcc{i}_std"] = s
    d1 = librosa.feature.delta(mfcc)
    for i, (m, s) in enumerate(zip(d1.mean(axis=1), d1.std(axis=1)), 1):
        f[f"ac_dmfcc{i}_mean"] = m;  f[f"ac_dmfcc{i}_std"] = s
    d2 = librosa.feature.delta(mfcc, order=2)
    for i, (m, s) in enumerate(zip(d2.mean(axis=1), d2.std(axis=1)), 1):
        f[f"ac_d2mfcc{i}_mean"] = m;  f[f"ac_d2mfcc{i}_std"] = s

    # F0 / voicing
    f0, voiced, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
    f0v = f0[voiced] if voiced.any() else np.array([0.0])
    f["ac_f0_mean"]          = float(np.mean(f0v))
    f["ac_f0_std"]           = float(np.std(f0v))
    f["ac_f0_range"]         = float(np.ptp(f0v))
    f["ac_f0_jitter"]        = float(np.mean(np.abs(np.diff(f0v)))) if len(f0v) > 1 else 0.0
    f["ac_voiced_fraction"]  = float(voiced.mean())

    # RMS / shimmer
    rms = librosa.feature.rms(y=y)[0]
    f["ac_rms_mean"]         = float(rms.mean())
    f["ac_rms_std"]          = float(rms.std())
    f["ac_shimmer_approx"]   = float(np.mean(np.abs(np.diff(rms))) / (rms.mean() + 1e-9))

    # Spectral
    sc  = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    sb  = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    sro = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    sf  = librosa.feature.spectral_flatness(y=y)[0]
    sct = librosa.feature.spectral_contrast(y=y, sr=sr)
    f["ac_spec_centroid_mean"]  = float(sc.mean());  f["ac_spec_centroid_std"]  = float(sc.std())
    f["ac_spec_bandwidth_mean"] = float(sb.mean());  f["ac_spec_bandwidth_std"] = float(sb.std())
    f["ac_spec_rolloff_mean"]   = float(sro.mean()); f["ac_spec_rolloff_std"]   = float(sro.std())
    f["ac_spec_flatness_mean"]  = float(sf.mean());  f["ac_spec_flatness_std"]  = float(sf.std())
    for b in range(sct.shape[0]):
        f[f"ac_spec_contrast_b{b+1}"] = float(sct[b].mean())

    # ZCR
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    f["ac_zcr_mean"] = float(zcr.mean());  f["ac_zcr_std"] = float(zcr.std())

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    f["ac_chroma_mean"] = float(chroma.mean());  f["ac_chroma_std"] = float(chroma.std())

    # Mel
    mel_db = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40), ref=np.max)
    f["ac_mel_mean"] = float(mel_db.mean());  f["ac_mel_std"] = float(mel_db.std())

    # Tonnetz
    y_h = librosa.effects.harmonic(y)
    ton = librosa.feature.tonnetz(y=y_h, sr=sr)
    for t in range(ton.shape[0]):
        f[f"ac_tonnetz{t+1}"] = float(ton[t].mean())

    # Onset rate & pause ratio
    onsets   = librosa.onset.onset_detect(y=y, sr=sr, units="time")
    duration = librosa.get_duration(y=y, sr=sr)
    f["ac_onset_rate"]  = float(len(onsets) / max(duration, 1e-3))
    rms2 = librosa.feature.rms(y=y, hop_length=512)[0]
    f["ac_pause_ratio"] = float((rms2 < rms2.max() * 0.02).mean())

    return f


if os.path.exists(ACOU_CACHE):
    print("\nLoading cached acoustic features …")
    acou_df = pd.read_csv(ACOU_CACHE)
else:
    print(f"\nExtracting acoustic features from {len(manifest)} clips …")
    records = []
    for i, row in manifest.iterrows():
        try:
            feat = extract_acoustic(row["path"])
            feat["subject"] = row["subject"]
            feat["file"]    = row["file"]
            feat["label"]   = row["label"]
            records.append(feat)
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(manifest)} done …")
        except Exception as e:
            print(f"  SKIP {row['file']}: {e}")
    acou_df = pd.DataFrame(records)
    acou_df.to_csv(ACOU_CACHE, index=False)
    print(f"  Acoustic extraction complete: {len(acou_df)} clips")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Whisper transcription (base.en via CLI)
# ══════════════════════════════════════════════════════════════════════════════
WHISPER_WORKER = """
import sys, json, subprocess, tempfile, os
audio_path = sys.argv[1]
with tempfile.TemporaryDirectory() as tmp:
    cmd = ["whisper", audio_path, "--model", "base.en", "--language", "en",
           "--output_format", "json", "--word_timestamps", "True",
           "--output_dir", tmp, "--fp16", "False", "--verbose", "False"]
    subprocess.run(cmd, check=True, capture_output=True)
    jfiles = [f for f in os.listdir(tmp) if f.endswith(".json")]
    if not jfiles:
        print(json.dumps({"text": "", "words": []})); sys.exit(0)
    with open(os.path.join(tmp, jfiles[0])) as f:
        data = json.load(f)
text  = data.get("text","").strip()
words = []
for seg in data.get("segments",[]):
    for w in seg.get("words",[]):
        words.append([w["word"].strip(), w["start"], w["end"]])
print(json.dumps({"text": text, "words": words}))
"""

def transcribe(path):
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(WHISPER_WORKER);  script = f.name
    try:
        out = subprocess.check_output(
            ["python3", script, path], stderr=subprocess.DEVNULL, timeout=300
        )
        return json.loads(out.decode())
    finally:
        os.unlink(script)


if os.path.exists(TRANS_CACHE):
    print("\nLoading cached transcripts …")
    trans_df = pd.read_csv(TRANS_CACHE)
else:
    print(f"\nTranscribing {len(manifest)} clips with Whisper base.en …")
    rows = []
    for i, row in manifest.iterrows():
        print(f"  [{i+1}/{len(manifest)}] {row['subject']} / {row['file']}", flush=True)
        try:
            result = transcribe(row["path"])
            rows.append({
                "subject":    row["subject"],
                "file":       row["file"],
                "label":      row["label"],
                "text":       result["text"],
                "words_json": json.dumps(result["words"]),
            })
        except Exception as e:
            print(f"    ERROR: {e}")
            rows.append({
                "subject": row["subject"], "file": row["file"],
                "label": row["label"], "text": "", "words_json": "[]",
            })
    trans_df = pd.DataFrame(rows)
    trans_df.to_csv(TRANS_CACHE, index=False)
    print(f"  Transcription complete: {len(trans_df)} clips")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Linguistic feature extraction
# ══════════════════════════════════════════════════════════════════════════════
FUNCTION_WORDS = {
    "the","a","an","is","are","was","were","be","been","being","have","has","had",
    "do","does","did","will","would","shall","should","may","might","must","can","could",
    "and","but","or","nor","so","yet","for","at","by","in","on","to","up","as","of",
    "if","then","than","that","this","these","those","it","its","i","you","he","she",
    "we","they","me","him","her","us","them","my","your","his","our","their","which",
    "who","what","when","where","how","with","from","into","through","during","not",
    "no","nor","both","either","neither","each","every","all","any","some","such",
}
FILLERS = {"uh","um","er","ah","like","okay","well","right","so","yeah"}


def tokenize(text):
    return re.findall(r"\b[a-z']+\b", text.lower())


def extract_linguistic(text, words_json):
    f       = {}
    tokens  = tokenize(text)
    sents   = [s.strip() for s in re.split(r"[.!?]+", text.strip()) if s.strip()]
    n_tok   = len(tokens)
    n_ch    = len(text.replace(" ", ""))
    n_sent  = max(len(sents), 1)
    vocab   = set(tokens)

    f["li_ttr"]          = len(vocab) / max(n_tok, 1)
    f["li_vocab_size"]   = len(vocab)
    f["li_word_count"]   = n_tok

    wl = [len(t) for t in tokens]
    f["li_avg_word_len"] = np.mean(wl) if wl else 0.0
    f["li_std_word_len"] = np.std(wl)  if wl else 0.0

    sl = [len(tokenize(s)) for s in sents]
    f["li_avg_sent_len"] = np.mean(sl) if sl else 0.0
    f["li_std_sent_len"] = np.std(sl)  if sl else 0.0
    f["li_num_sentences"]= n_sent

    n_func = sum(1 for t in tokens if t in FUNCTION_WORDS)
    f["li_func_word_ratio"]    = n_func / max(n_tok, 1)
    f["li_content_word_ratio"] = 1.0 - f["li_func_word_ratio"]
    f["li_lexical_density"]    = f["li_content_word_ratio"]

    bigrams  = list(zip(tokens, tokens[1:]))
    trigrams = list(zip(tokens, tokens[1:], tokens[2:]))
    f["li_bigram_rep"]   = (len(bigrams)  - len(set(bigrams)))  / max(len(bigrams),  1)
    f["li_trigram_rep"]  = (len(trigrams) - len(set(trigrams))) / max(len(trigrams), 1)
    f["li_top_word_dom"] = (Counter(tokens).most_common(1)[0][1] / n_tok) if tokens else 0.0

    n_fill = sum(1 for t in tokens if t in FILLERS)
    f["li_filler_ratio"] = n_fill / max(n_tok, 1)
    f["li_filler_count"] = n_fill

    f["li_punct_density"]        = sum(1 for c in text if c in ".,;:!?") / max(len(text), 1)
    f["li_commas_per_sentence"]  = text.count(",") / max(n_sent, 1)

    try:
        wl_list = json.loads(words_json)
    except Exception:
        wl_list = []

    if len(wl_list) >= 2:
        durs       = [e - s for _, s, e in wl_list]
        gaps       = [wl_list[i+1][1] - wl_list[i][2] for i in range(len(wl_list)-1)]
        gaps_c     = [g for g in gaps if g >= 0]
        total_sp   = sum(durs)
        total_time = wl_list[-1][2] - wl_list[0][1]

        f["li_avg_word_dur"]     = np.mean(durs)
        f["li_std_word_dur"]     = np.std(durs)
        f["li_avg_gap"]          = np.mean(gaps_c) if gaps_c else 0.0
        f["li_std_gap"]          = np.std(gaps_c)  if gaps_c else 0.0
        f["li_max_pause"]        = max(gaps_c)      if gaps_c else 0.0
        f["li_long_pause_count"] = sum(1 for g in gaps_c if g > 0.5)
        f["li_long_pause_ratio"] = f["li_long_pause_count"] / max(len(gaps_c), 1)
        f["li_speech_rate"]      = len(wl_list) / max(total_time, 1e-3)
        f["li_articu_rate"]      = len(wl_list) / max(total_sp,   1e-3)
        f["li_speaking_ratio"]   = total_sp / max(total_time, 1e-3)
        f["li_chars_per_sec"]    = n_ch / max(total_time, 1e-3)
    else:
        for k in ["li_avg_word_dur","li_std_word_dur","li_avg_gap","li_std_gap",
                  "li_max_pause","li_long_pause_count","li_long_pause_ratio",
                  "li_speech_rate","li_articu_rate","li_speaking_ratio","li_chars_per_sec"]:
            f[k] = 0.0

    return f


print("\nExtracting linguistic features …")
ling_records = []
for _, row in trans_df.iterrows():
    feat = extract_linguistic(str(row["text"]), str(row["words_json"]))
    feat["subject"] = row["subject"]
    feat["file"]    = row["file"]
    ling_records.append(feat)
ling_df = pd.DataFrame(ling_records)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Merge & average per subject
# ══════════════════════════════════════════════════════════════════════════════
acou_cols = [c for c in acou_df.columns if c.startswith("ac_")]
ling_cols = [c for c in ling_df.columns  if c.startswith("li_")]

acou_sub = acou_df.groupby("subject")[acou_cols + ["label"]].mean().reset_index()
ling_sub = ling_df.groupby("subject")[ling_cols].mean().reset_index()

merged = acou_sub.merge(ling_sub, on="subject")
merged["label"] = merged["label"].round().astype(int)

all_feat_cols = acou_cols + ling_cols
feat_modality = {c: "Acoustic" for c in acou_cols}
feat_modality.update({c: "Linguistic" for c in ling_cols})

X        = merged[all_feat_cols].values.astype(float)
y_labels = merged["label"].values

print(f"\nSubject-level dataset: {len(merged)} subjects")
print(f"  Dementia    : {y_labels.sum()}")
print(f"  No-Dementia : {(y_labels==0).sum()}")
print(f"  Acoustic features  : {len(acou_cols)}")
print(f"  Linguistic features: {len(ling_cols)}")
print(f"  Total features     : {len(all_feat_cols)}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Pipeline & 5-Fold CV
# ══════════════════════════════════════════════════════════════════════════════
N_FEATURES = 25

clf = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
)

pipe = ImbPipeline([
    ("scaler", StandardScaler()),
    ("select", SelectKBest(mutual_info_classif, k=N_FEATURES)),
    ("smote",  SMOTE(random_state=42, k_neighbors=5)),
    ("clf",    clf),
])

print("\nRunning 5-Fold Stratified CV …")
cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ["accuracy","precision","recall","f1","roc_auc"]
cv_res  = cross_validate(pipe, X, y_labels, cv=cv, scoring=scoring, return_train_score=False)

print("\n── 5-Fold CV Results ──")
metrics = {
    "Accuracy":  "test_accuracy",
    "Precision": "test_precision",
    "Recall":    "test_recall",
    "F1-Score":  "test_f1",
    "ROC-AUC":   "test_roc_auc",
}
summary = []
for name, key in metrics.items():
    vals = cv_res[key]
    summary.append({"Metric": name, "Mean": vals.mean(), "Std": vals.std(),
                    **{f"Fold{i+1}": v for i, v in enumerate(vals)}})
    print(f"  {name:12s}: {vals.mean():.4f} ± {vals.std():.4f}  |  folds: {np.round(vals,3)}")

pd.DataFrame(summary).to_csv(os.path.join(OUT, "cv_metrics.csv"), index=False)

# Cross-val predictions for confusion matrix & ROC
y_pred_cv  = cross_val_predict(pipe, X, y_labels, cv=cv, method="predict")
y_proba_cv = cross_val_predict(pipe, X, y_labels, cv=cv, method="predict_proba")[:, 1]

print("\n── Classification Report (CV predictions) ──")
print(classification_report(y_labels, y_pred_cv, target_names=["No Dementia","Dementia"]))


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — Confusion matrix & ROC
# ══════════════════════════════════════════════════════════════════════════════
auc = roc_auc_score(y_labels, y_proba_cv)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ConfusionMatrixDisplay(
    confusion_matrix(y_labels, y_pred_cv), display_labels=["No Dementia","Dementia"]
).plot(ax=axes[0], colorbar=False, cmap="Blues")
axes[0].set_title("Dementia Fusion — Confusion Matrix (5-Fold CV)")
RocCurveDisplay.from_predictions(
    y_labels, y_proba_cv, ax=axes[1], name="XGBoost Fusion"
)
axes[1].set_title(f"Dementia Fusion — ROC Curve (AUC={auc:.3f})")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "confusion_roc.png"), dpi=150)
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — Final model for interpretability
# ══════════════════════════════════════════════════════════════════════════════
print("\nFitting final model on full data …")
pipe.fit(X, y_labels)

scaler_f = pipe.named_steps["scaler"]
select_f = pipe.named_steps["select"]
clf_f    = pipe.named_steps["clf"]
sel_mask = select_f.get_support()
sel_feat = [f for f, m in zip(all_feat_cols, sel_mask) if m]
X_sel    = select_f.transform(scaler_f.transform(X))

imp_df = pd.DataFrame({
    "feature":    sel_feat,
    "importance": clf_f.feature_importances_,
    "modality":   [feat_modality[f] for f in sel_feat],
}).sort_values("importance", ascending=False).reset_index(drop=True)
imp_df.to_csv(os.path.join(OUT, "feature_importance.csv"), index=False)

print(f"\nTop 15 features:\n{imp_df.head(15).to_string(index=False)}")

modal_imp = imp_df.groupby("modality")["importance"].sum()
print(f"\nImportance by modality:\n{modal_imp.to_string()}")

palette = {"Acoustic": "#1565C0", "Linguistic": "#E64A19"}
fig, ax = plt.subplots(figsize=(12, 9))
sns.barplot(data=imp_df, y="feature", x="importance",
            hue="modality", dodge=False, palette=palette, ax=ax)
ax.set_title("Dementia Fusion — Feature Importances (blue=Acoustic, orange=Linguistic)")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "feature_importance.png"), dpi=150)
plt.close()

# Modality pie
fig, ax = plt.subplots(figsize=(6, 6))
ax.pie(modal_imp.values, labels=modal_imp.index, autopct="%1.1f%%",
       colors=[palette[m] for m in modal_imp.index], startangle=90)
ax.set_title("Feature Importance by Modality")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "modality_pie.png"), dpi=150)
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 9 — SHAP
# ══════════════════════════════════════════════════════════════════════════════
print("\nRunning SHAP analysis …")
X_df      = pd.DataFrame(X_sel, columns=sel_feat)
explainer = shap.TreeExplainer(clf_f)
shap_vals = explainer.shap_values(X_df)

plt.figure(figsize=(12, 9))
shap.summary_plot(shap_vals, X_df, show=False, max_display=25)
plt.title("Dementia Fusion — SHAP Summary (Beeswarm)")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "shap_summary.png"), dpi=150, bbox_inches="tight")
plt.close()

plt.figure(figsize=(12, 8))
shap.summary_plot(shap_vals, X_df, plot_type="bar", show=False, max_display=25)
plt.title("Dementia Fusion — SHAP Feature Importance (Bar)")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "shap_bar.png"), dpi=150, bbox_inches="tight")
plt.close()

pd.DataFrame(shap_vals, columns=sel_feat).assign(
    subject=merged["subject"].values, label=merged["label"].values
).to_csv(os.path.join(OUT, "shap_values.csv"), index=False)


# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
acc  = accuracy_score(y_labels, y_pred_cv)
prec = precision_score(y_labels, y_pred_cv, zero_division=0)
rec  = recall_score(y_labels, y_pred_cv, zero_division=0)
f1   = f1_score(y_labels, y_pred_cv, zero_division=0)

print("\n" + "="*50)
print("  FINAL SUMMARY")
print("="*50)
print(f"  Subjects used      : {len(merged)}")
print(f"  Features (selected): {len(sel_feat)}")
print(f"  Accuracy           : {acc:.4f}")
print(f"  Precision          : {prec:.4f}")
print(f"  Recall             : {rec:.4f}")
print(f"  F1-Score           : {f1:.4f}")
print(f"  ROC-AUC            : {auc:.4f}")
print(f"\nAll results → {OUT}")
print("Done.")
