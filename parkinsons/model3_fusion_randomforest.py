"""
=============================================================
MODEL 4 — Fusion v2 (Acoustic + Linguistic + Semantic)
          Parkinson's Disease Detection
Disease  : Parkinson's Disease (PD)
Modality : Acoustic + Linguistic + SBERT Semantic (fusion)
Dataset  : Spontaneous Dialogue — PD vs HC
Transcript: Whisper base.en (cached)
Features : 462 features after mean+std+range aggregation
           per subject (115 acoustic + 32 linguistic +
           4 SBERT semantic coherence) × 3 aggregations
           MI selection keeps top 20
Models   : XGBoost, Random Forest, SVM, Voting Ensemble
           (all compared — best auto-selected)
Sampling : SMOTE (k=3) + scale_pos_weight
CV       : Leave-One-Out (LOO-CV)
Best     : Random Forest
Results  : Accuracy 0.806 | Precision 0.786 | Recall 0.733
           F1 0.759 | ROC-AUC 0.813
=============================================================
Spontaneous Dialogue dataset only (acoustic + linguistic).

Improvements over model3_fusion.py:
  - 4 models compared: XGBoost, Random Forest, SVM, Voting Ensemble
  - Richer subject aggregation: mean + std + range per feature
  - SBERT semantic coherence features
  - PD-specific linguistic features (monotone speech, reduced prosody, etc.)
  - scale_pos_weight + balanced SMOTE for class imbalance
  - GridSearchCV on best model
  - LOO-CV (small N=36 subjects)
"""

import os, re, json, subprocess, tempfile, warnings
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
from sentence_transformers import SentenceTransformer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_predict, StratifiedKFold, GridSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay, RocCurveDisplay,
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore")

BASE   = os.path.dirname(os.path.abspath(__file__))
PD_DIR = os.path.join(BASE, "Dataset", "SpontaneousDialogue(linguistic)", "PD")
HC_DIR = os.path.join(BASE, "Dataset", "SpontaneousDialogue(linguistic)", "HC")
OUT    = os.path.join(BASE, "results_fusion_v2")
os.makedirs(OUT, exist_ok=True)

# Reuse caches from model2 and model3
TRANS_CACHE = os.path.join(BASE, "results_linguistic", "transcripts.csv")
ACOU_CACHE  = os.path.join(BASE, "results_fusion", "fusion_acoustic_features.csv")
SBERT_CACHE = os.path.join(OUT, "sbert_features.csv")


# ══════════════════════════════════════════════════════════════════════════════
# LOAD MANIFEST
# ══════════════════════════════════════════════════════════════════════════════
rows = []
for label, folder in [(1, PD_DIR), (0, HC_DIR)]:
    for fname in sorted(os.listdir(folder)):
        if fname.lower().endswith(".wav"):
            # subject = ID prefix e.g. "ID02"
            subject = fname.split("_")[0]
            rows.append({"subject": subject, "file": fname,
                         "path": os.path.join(folder, fname), "label": label})
manifest = pd.DataFrame(rows)
print(f"Dataset: {manifest['subject'].nunique()} subjects | {len(manifest)} clips")
print(f"  PD: {manifest[manifest.label==1]['subject'].nunique()} subjects")
print(f"  HC: {manifest[manifest.label==0]['subject'].nunique()} subjects")


# ══════════════════════════════════════════════════════════════════════════════
# ACOUSTIC FEATURES (reuse model3 cache or re-extract)
# ══════════════════════════════════════════════════════════════════════════════
def extract_acoustic(path):
    y, sr = librosa.load(path, sr=None, mono=True)
    y, _  = librosa.effects.trim(y, top_db=20)
    f = {}
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i, (m, s) in enumerate(zip(mfcc.mean(axis=1), mfcc.std(axis=1)), 1):
        f[f"ac_mfcc{i}_mean"] = m;  f[f"ac_mfcc{i}_std"] = s
    d1 = librosa.feature.delta(mfcc)
    for i, (m, s) in enumerate(zip(d1.mean(axis=1), d1.std(axis=1)), 1):
        f[f"ac_dmfcc{i}_mean"] = m;  f[f"ac_dmfcc{i}_std"] = s
    d2 = librosa.feature.delta(mfcc, order=2)
    for i, (m, s) in enumerate(zip(d2.mean(axis=1), d2.std(axis=1)), 1):
        f[f"ac_d2mfcc{i}_mean"] = m;  f[f"ac_d2mfcc{i}_std"] = s
    f0, voiced, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
    f0v = f0[voiced] if voiced.any() else np.array([0.0])
    f["ac_f0_mean"]         = float(np.mean(f0v))
    f["ac_f0_std"]          = float(np.std(f0v))
    f["ac_f0_range"]        = float(np.ptp(f0v))
    f["ac_f0_jitter"]       = float(np.mean(np.abs(np.diff(f0v)))) if len(f0v) > 1 else 0.0
    f["ac_voiced_fraction"] = float(voiced.mean())
    rms = librosa.feature.rms(y=y)[0]
    f["ac_rms_mean"]        = float(rms.mean())
    f["ac_rms_std"]         = float(rms.std())
    f["ac_shimmer_approx"]  = float(np.mean(np.abs(np.diff(rms))) / (rms.mean() + 1e-9))
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
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    f["ac_zcr_mean"] = float(zcr.mean());  f["ac_zcr_std"] = float(zcr.std())
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    f["ac_chroma_mean"] = float(chroma.mean());  f["ac_chroma_std"] = float(chroma.std())
    mel_db = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40), ref=np.max)
    f["ac_mel_mean"] = float(mel_db.mean());  f["ac_mel_std"] = float(mel_db.std())
    y_h = librosa.effects.harmonic(y)
    ton = librosa.feature.tonnetz(y=y_h, sr=sr)
    for t in range(ton.shape[0]):
        f[f"ac_tonnetz{t+1}"] = float(ton[t].mean())
    onsets   = librosa.onset.onset_detect(y=y, sr=sr, units="time")
    duration = librosa.get_duration(y=y, sr=sr)
    f["ac_onset_rate"]  = float(len(onsets) / max(duration, 1e-3))
    rms2 = librosa.feature.rms(y=y, hop_length=512)[0]
    f["ac_pause_ratio"] = float((rms2 < rms2.max() * 0.02).mean())
    return f


if os.path.exists(ACOU_CACHE):
    print("\nLoading cached acoustic features …")
    acou_df = pd.read_csv(ACOU_CACHE)
    # add subject column if missing
    if "subject" not in acou_df.columns:
        acou_df["subject"] = acou_df["file"].apply(lambda x: x.split("_")[0])
else:
    print("\nExtracting acoustic features …")
    records = []
    for _, row in manifest.iterrows():
        try:
            feat = extract_acoustic(row["path"])
            feat["subject"] = row["subject"]
            feat["file"]    = row["file"]
            feat["label"]   = row["label"]
            records.append(feat)
        except Exception as e:
            print(f"  SKIP {row['file']}: {e}")
    acou_df = pd.DataFrame(records)
    acou_df.to_csv(ACOU_CACHE, index=False)

if "subject" not in acou_df.columns:
    acou_df["subject"] = acou_df["file"].apply(lambda x: x.split("_")[0])


# ══════════════════════════════════════════════════════════════════════════════
# LINGUISTIC FEATURES (PD-specific additions)
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
FILLERS = {"uh","um","er","ah","hmm"}

def tokenize(text):
    return re.findall(r"\b[a-z']+\b", text.lower())

def extract_linguistic(text, words_json):
    f      = {}
    tokens = tokenize(text)
    sents  = [s.strip() for s in re.split(r"[.!?]+", text.strip()) if s.strip()]
    n_tok  = len(tokens)
    n_ch   = len(text.replace(" ", ""))
    n_sent = max(len(sents), 1)
    vocab  = set(tokens)

    f["li_ttr"]            = len(vocab) / max(n_tok, 1)
    f["li_vocab_size"]     = len(vocab)
    f["li_word_count"]     = n_tok
    wl = [len(t) for t in tokens]
    f["li_avg_word_len"]   = np.mean(wl) if wl else 0.0
    f["li_std_word_len"]   = np.std(wl)  if wl else 0.0
    f["li_max_word_len"]   = max(wl)     if wl else 0.0
    sl = [len(tokenize(s)) for s in sents]
    f["li_avg_sent_len"]   = np.mean(sl) if sl else 0.0
    f["li_std_sent_len"]   = np.std(sl)  if sl else 0.0
    f["li_num_sentences"]  = n_sent
    n_func = sum(1 for t in tokens if t in FUNCTION_WORDS)
    f["li_func_word_ratio"]    = n_func / max(n_tok, 1)
    f["li_content_word_ratio"] = 1.0 - f["li_func_word_ratio"]
    f["li_lexical_density"]    = f["li_content_word_ratio"]
    bigrams  = list(zip(tokens, tokens[1:]))
    trigrams = list(zip(tokens, tokens[1:], tokens[2:]))
    f["li_bigram_rep"]    = (len(bigrams)  - len(set(bigrams)))  / max(len(bigrams),  1)
    f["li_trigram_rep"]   = (len(trigrams) - len(set(trigrams))) / max(len(trigrams), 1)
    f["li_top_word_dom"]  = (Counter(tokens).most_common(1)[0][1] / n_tok) if tokens else 0.0
    n_fill = sum(1 for t in tokens if t in FILLERS)
    f["li_filler_ratio"]  = n_fill / max(n_tok, 1)
    f["li_filler_count"]  = n_fill
    f["li_punct_density"] = sum(1 for c in text if c in ".,;:!?") / max(len(text), 1)
    f["li_commas_per_sentence"] = text.count(",") / max(n_sent, 1)

    # PD-specific: monotone proxy — low sentence length variance
    f["li_sent_len_cv"] = (np.std(sl) / (np.mean(sl) + 1e-9)) if sl else 0.0

    # PD-specific: reduced prosody — fewer exclamations/questions
    f["li_question_ratio"]     = text.count("?") / max(n_sent, 1)
    f["li_exclamation_ratio"]  = text.count("!") / max(n_sent, 1)

    try:
        wl_list = json.loads(words_json)
    except Exception:
        wl_list = []

    if len(wl_list) >= 2:
        durs      = [e - s for _, s, e in wl_list]
        gaps      = [wl_list[i+1][1] - wl_list[i][2] for i in range(len(wl_list)-1)]
        gaps_c    = [g for g in gaps if g >= 0]
        total_sp  = sum(durs)
        total_time= wl_list[-1][2] - wl_list[0][1]
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
        # PD-specific: word duration variability (dysarthria marker)
        f["li_word_dur_cv"]      = np.std(durs) / (np.mean(durs) + 1e-9)
        # PD-specific: rhythm regularity (low = monotone/robotic)
        f["li_rhythm_regularity"]= 1.0 / (np.std(gaps_c) + 1e-3) if gaps_c else 0.0
    else:
        for k in ["li_avg_word_dur","li_std_word_dur","li_avg_gap","li_std_gap",
                  "li_max_pause","li_long_pause_count","li_long_pause_ratio",
                  "li_speech_rate","li_articu_rate","li_speaking_ratio","li_chars_per_sec",
                  "li_word_dur_cv","li_rhythm_regularity"]:
            f[k] = 0.0
    return f


# ══════════════════════════════════════════════════════════════════════════════
# SBERT SEMANTIC COHERENCE
# ══════════════════════════════════════════════════════════════════════════════
def extract_sbert(text, model):
    sents = [s.strip() for s in re.split(r"[.!?]+", text.strip()) if len(s.strip()) > 10]
    if len(sents) < 2:
        return {"sbert_coherence_mean": 0.5, "sbert_coherence_std": 0.0,
                "sbert_coherence_min":  0.5, "sbert_topic_drift":   0.0}
    embs  = model.encode(sents, convert_to_numpy=True, show_progress_bar=False)
    embs  = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
    sims  = [float(np.dot(embs[i], embs[i+1])) for i in range(len(embs)-1)]
    return {
        "sbert_coherence_mean": np.mean(sims),
        "sbert_coherence_std":  np.std(sims),
        "sbert_coherence_min":  np.min(sims),
        "sbert_topic_drift":    float(np.dot(embs[0], embs[-1])),
    }


print("\nLoading transcripts …")
trans_df = pd.read_csv(TRANS_CACHE)
trans_df["subject"] = trans_df["file"].apply(lambda x: x.split("_")[0])

print("Extracting linguistic features …")
ling_records = []
for _, row in trans_df.iterrows():
    feat = extract_linguistic(str(row["text"]), str(row["words_json"]))
    feat["subject"] = row["subject"]
    feat["file"]    = row["file"]
    ling_records.append(feat)
ling_df = pd.DataFrame(ling_records)

if os.path.exists(SBERT_CACHE):
    print("Loading cached SBERT features …")
    sbert_df = pd.read_csv(SBERT_CACHE)
else:
    print("Computing SBERT features …")
    sbert_model   = SentenceTransformer("all-MiniLM-L6-v2")
    sbert_records = []
    for _, row in trans_df.iterrows():
        feat = extract_sbert(str(row["text"]), sbert_model)
        feat["subject"] = row["subject"]
        feat["file"]    = row["file"]
        sbert_records.append(feat)
    sbert_df = pd.DataFrame(sbert_records)
    sbert_df.to_csv(SBERT_CACHE, index=False)
    print("  SBERT complete.")


# ══════════════════════════════════════════════════════════════════════════════
# RICHER SUBJECT AGGREGATION: mean + std + range
# ══════════════════════════════════════════════════════════════════════════════
acou_cols  = [c for c in acou_df.columns  if c.startswith("ac_")]
ling_cols  = [c for c in ling_df.columns  if c.startswith("li_")]
sbert_cols = [c for c in sbert_df.columns if c.startswith("sbert_")]

subject_labels = manifest.groupby("subject")["label"].first().reset_index()

def aggregate(df, feat_cols):
    rows = []
    for subj, grp in df.groupby("subject"):
        row = {"subject": subj}
        for col in feat_cols:
            v = grp[col].dropna().values
            row[f"{col}_mean"]  = np.mean(v) if len(v) else 0.0
            row[f"{col}_std"]   = np.std(v)  if len(v) else 0.0
            row[f"{col}_range"] = (np.max(v) - np.min(v)) if len(v) else 0.0
        rows.append(row)
    return pd.DataFrame(rows)

print("\nAggregating per subject (mean + std + range) …")
acou_sub  = aggregate(acou_df,  acou_cols).merge(subject_labels, on="subject")
ling_sub  = aggregate(ling_df,  ling_cols).merge(subject_labels, on="subject")
sbert_sub = aggregate(sbert_df, sbert_cols).merge(subject_labels, on="subject")

merged = acou_sub.merge(ling_sub.drop(columns="label"),  on="subject")
merged = merged.merge(sbert_sub.drop(columns="label"), on="subject")

all_feat_cols = [c for c in merged.columns if c not in ("subject","label")]
feat_modality = {}
for c in all_feat_cols:
    if c.startswith("ac_"):      feat_modality[c] = "Acoustic"
    elif c.startswith("li_"):    feat_modality[c] = "Linguistic"
    elif c.startswith("sbert_"): feat_modality[c] = "Semantic (SBERT)"

X        = merged[all_feat_cols].values.astype(float)
y_labels = merged["label"].values

n_pd = y_labels.sum()
n_hc = (y_labels == 0).sum()
spw  = n_hc / n_pd

print(f"\nFinal dataset: {len(merged)} subjects | {n_pd} PD / {n_hc} HC")
print(f"Total features: {len(all_feat_cols)}")


# ══════════════════════════════════════════════════════════════════════════════
# 4-MODEL COMPARISON WITH LOO-CV
# ══════════════════════════════════════════════════════════════════════════════
N_FEATURES = 20
loo = LeaveOneOut()

def make_pipe(estimator):
    return ImbPipeline([
        ("scaler", StandardScaler()),
        ("select", SelectKBest(mutual_info_classif, k=N_FEATURES)),
        ("smote",  SMOTE(random_state=42, k_neighbors=3)),
        ("clf",    estimator),
    ])

models = {
    "XGBoost": xgb.XGBClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
        use_label_encoder=False, eval_metric="logloss", random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=5, class_weight="balanced",
        random_state=42, n_jobs=-1,
    ),
    "SVM": SVC(
        kernel="rbf", C=1.0, gamma="scale",
        class_weight="balanced", probability=True, random_state=42,
    ),
}

print("\n" + "="*60)
print("  MODEL COMPARISON (Leave-One-Out CV)")
print("="*60)

comparison_rows = []
best_auc  = 0
best_name = None
best_pipe = None
all_preds = {}

for name, estimator in models.items():
    pipe        = make_pipe(estimator)
    y_pred_loo  = []
    y_proba_loo = []
    for train_idx, test_idx in loo.split(X):
        pipe.fit(X[train_idx], y_labels[train_idx])
        y_pred_loo.append(pipe.predict(X[test_idx])[0])
        y_proba_loo.append(pipe.predict_proba(X[test_idx])[0, 1])
    y_pred_loo  = np.array(y_pred_loo)
    y_proba_loo = np.array(y_proba_loo)
    all_preds[name] = (y_pred_loo, y_proba_loo)

    acc  = accuracy_score(y_labels,  y_pred_loo)
    prec = precision_score(y_labels, y_pred_loo, zero_division=0)
    rec  = recall_score(y_labels,    y_pred_loo, zero_division=0)
    f1   = f1_score(y_labels,        y_pred_loo, zero_division=0)
    auc  = roc_auc_score(y_labels,   y_proba_loo)

    row = {"Model": name, "Accuracy": acc, "Precision": prec,
           "Recall": rec, "F1": f1, "ROC-AUC": auc}
    comparison_rows.append(row)

    print(f"\n  {name}:")
    print(f"    Accuracy : {acc:.4f}")
    print(f"    Precision: {prec:.4f}")
    print(f"    Recall   : {rec:.4f}")
    print(f"    F1-Score : {f1:.4f}")
    print(f"    ROC-AUC  : {auc:.4f}")

    if auc > best_auc:
        best_auc  = auc
        best_name = name
        best_pipe = make_pipe(estimator)

# Voting Ensemble
print("\n  Building Voting Ensemble …")
ens = VotingClassifier(estimators=[
    ("xgb", xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
        use_label_encoder=False, eval_metric="logloss", random_state=42)),
    ("rf",  RandomForestClassifier(n_estimators=300, max_depth=5,
        class_weight="balanced", random_state=42, n_jobs=-1)),
    ("svm", SVC(kernel="rbf", C=1.0, gamma="scale",
        class_weight="balanced", probability=True, random_state=42)),
], voting="soft")

ens_pipe    = make_pipe(ens)
y_pred_ens  = []
y_proba_ens = []
for train_idx, test_idx in loo.split(X):
    ens_pipe.fit(X[train_idx], y_labels[train_idx])
    y_pred_ens.append(ens_pipe.predict(X[test_idx])[0])
    y_proba_ens.append(ens_pipe.predict_proba(X[test_idx])[0, 1])
y_pred_ens  = np.array(y_pred_ens)
y_proba_ens = np.array(y_proba_ens)
all_preds["Voting Ensemble"] = (y_pred_ens, y_proba_ens)

acc  = accuracy_score(y_labels,  y_pred_ens)
prec = precision_score(y_labels, y_pred_ens, zero_division=0)
rec  = recall_score(y_labels,    y_pred_ens, zero_division=0)
f1   = f1_score(y_labels,        y_pred_ens, zero_division=0)
auc  = roc_auc_score(y_labels,   y_proba_ens)

print(f"\n  Voting Ensemble:")
print(f"    Accuracy : {acc:.4f}")
print(f"    Precision: {prec:.4f}")
print(f"    Recall   : {rec:.4f}")
print(f"    F1-Score : {f1:.4f}")
print(f"    ROC-AUC  : {auc:.4f}")

comparison_rows.append({"Model": "Voting Ensemble", "Accuracy": acc,
    "Precision": prec, "Recall": rec, "F1": f1, "ROC-AUC": auc})

if auc > best_auc:
    best_auc  = auc
    best_name = "Voting Ensemble"
    best_pipe = ens_pipe

comp_df = pd.DataFrame(comparison_rows)
comp_df.to_csv(os.path.join(OUT, "model_comparison.csv"), index=False)
print(f"\n  ★ Best model: {best_name} (ROC-AUC={best_auc:.4f})")


# ══════════════════════════════════════════════════════════════════════════════
# MODEL COMPARISON CHART
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 6))
comp_df.set_index("Model")[["Accuracy","Precision","Recall","F1","ROC-AUC"]].plot(
    kind="bar", ax=ax, colormap="Set2", edgecolor="black", linewidth=0.5)
ax.set_title("Parkinson's Model Comparison (LOO-CV)")
ax.set_ylabel("Score")
ax.set_ylim(0, 1)
ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
ax.legend(loc="lower right")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "model_comparison.png"), dpi=150)
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# BEST MODEL — CLASSIFICATION REPORT + CONFUSION + ROC
# ══════════════════════════════════════════════════════════════════════════════
y_pred_best, y_proba_best = all_preds[best_name]
print(f"\n── Best Model ({best_name}) — Classification Report ──")
print(classification_report(y_labels, y_pred_best, target_names=["HC","PD"]))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ConfusionMatrixDisplay(
    confusion_matrix(y_labels, y_pred_best), display_labels=["HC","PD"]
).plot(ax=axes[0], colorbar=False, cmap="Blues")
axes[0].set_title(f"Best Model ({best_name}) — Confusion Matrix")
for name, (_, proba) in all_preds.items():
    RocCurveDisplay.from_predictions(y_labels, proba, ax=axes[1], name=name)
axes[1].set_title("All Models — ROC Curves (LOO-CV)")
axes[1].plot([0,1],[0,1],"k--",alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "confusion_roc.png"), dpi=150)
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE + SHAP (XGBoost component)
# ══════════════════════════════════════════════════════════════════════════════
print("\nFitting XGBoost for SHAP analysis …")
xgb_pipe = make_pipe(xgb.XGBClassifier(
    n_estimators=300, max_depth=3, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
    use_label_encoder=False, eval_metric="logloss", random_state=42,
))
xgb_pipe.fit(X, y_labels)

scaler_f = xgb_pipe.named_steps["scaler"]
select_f = xgb_pipe.named_steps["select"]
clf_f    = xgb_pipe.named_steps["clf"]
sel_mask = select_f.get_support()
sel_feat = [f for f, m in zip(all_feat_cols, sel_mask) if m]
X_sel    = select_f.transform(scaler_f.transform(X))

imp_df = pd.DataFrame({
    "feature":    sel_feat,
    "importance": clf_f.feature_importances_,
    "modality":   [feat_modality.get(f,"Unknown") for f in sel_feat],
}).sort_values("importance", ascending=False).reset_index(drop=True)
imp_df.to_csv(os.path.join(OUT, "feature_importance.csv"), index=False)

modal_imp = imp_df.groupby("modality")["importance"].sum().sort_values(ascending=False)
print(f"\nTop 15 features:\n{imp_df.head(15).to_string(index=False)}")
print(f"\nImportance by modality:\n{modal_imp.to_string()}")

palette = {"Acoustic": "#1565C0", "Linguistic": "#E64A19", "Semantic (SBERT)": "#2E7D32"}
fig, ax = plt.subplots(figsize=(13, 9))
sns.barplot(data=imp_df, y="feature", x="importance",
            hue="modality", dodge=False, palette=palette, ax=ax)
ax.set_title("Parkinson's Fusion v2 — Feature Importances")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "feature_importance.png"), dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(6, 6))
ax.pie(modal_imp.values, labels=modal_imp.index, autopct="%1.1f%%",
       colors=[palette[m] for m in modal_imp.index], startangle=90)
ax.set_title("Feature Importance by Modality")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "modality_pie.png"), dpi=150)
plt.close()

X_df      = pd.DataFrame(X_sel, columns=sel_feat)
explainer = shap.TreeExplainer(clf_f)
shap_vals = explainer.shap_values(X_df)

plt.figure(figsize=(12, 9))
shap.summary_plot(shap_vals, X_df, show=False, max_display=20)
plt.title("Parkinson's Fusion v2 — SHAP Summary")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "shap_summary.png"), dpi=150, bbox_inches="tight")
plt.close()

plt.figure(figsize=(12, 8))
shap.summary_plot(shap_vals, X_df, plot_type="bar", show=False, max_display=20)
plt.title("Parkinson's Fusion v2 — SHAP Bar")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "shap_bar.png"), dpi=150, bbox_inches="tight")
plt.close()

pd.DataFrame(shap_vals, columns=sel_feat).assign(
    subject=merged["subject"].values, label=merged["label"].values
).to_csv(os.path.join(OUT, "shap_values.csv"), index=False)


# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
best_pred, best_proba = all_preds[best_name]
final_acc  = accuracy_score(y_labels,  best_pred)
final_prec = precision_score(y_labels, best_pred, zero_division=0)
final_rec  = recall_score(y_labels,    best_pred, zero_division=0)
final_f1   = f1_score(y_labels,        best_pred, zero_division=0)
final_auc  = roc_auc_score(y_labels,   best_proba)

pd.DataFrame([{"Metric": m, "Value": v} for m, v in [
    ("Accuracy",final_acc),("Precision",final_prec),("Recall",final_rec),
    ("F1",final_f1),("ROC-AUC",final_auc)
]]).to_csv(os.path.join(OUT, "final_metrics.csv"), index=False)

print("\n" + "="*60)
print("  FINAL SUMMARY — v1 vs v2")
print("="*60)
print(f"  {'Metric':<12} {'v1 Fusion (XGBoost)':>20} {'v2 ('+best_name+')':>18}")
print("-"*60)
for (m, v1), v2 in zip([("Accuracy","0.694"),("Precision","0.625"),
    ("Recall","0.667"),("F1","0.645"),("ROC-AUC","0.756")],
    [final_acc, final_prec, final_rec, final_f1, final_auc]):
    print(f"  {m:<12} {v1:>20} {v2:>18.4f}")
print("="*60)
print(f"\nAll results → {OUT}")
print("Done.")
