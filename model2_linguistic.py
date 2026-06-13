"""
Model 2: Linguistic-Based Parkinson's Disease Detection
Fixes applied:
  - Whisper CLI with base.en model (avoids PyTorch segfault, better quality)
  - LOOCV instead of 5-fold
  - Mutual-information feature selection inside the CV pipeline
  - SMOTE oversampling inside each CV fold
"""

import os
import re
import json
import subprocess
import tempfile
import warnings
import numpy as np
import pandas as pd
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
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay, RocCurveDisplay,
)

warnings.filterwarnings("ignore")

BASE   = os.path.dirname(os.path.abspath(__file__))
PD_DIR = os.path.join(BASE, "Dataset", "SpontaneousDialogue(linguistic)", "PD")
HC_DIR = os.path.join(BASE, "Dataset", "SpontaneousDialogue(linguistic)", "HC")
OUT    = os.path.join(BASE, "results_linguistic")
os.makedirs(OUT, exist_ok=True)

TRANSCRIPT_CACHE = os.path.join(OUT, "transcripts.csv")
WHISPER_MODEL    = "base.en"


# ── Whisper CLI transcription ──────────────────────────────────────────────────
def transcribe_file_cli(audio_path: str) -> dict:
    """
    Run `whisper` CLI in a temp dir to get JSON output with word timestamps.
    CLI runs in its own process — avoids the PyTorch segfault from the Python API.
    """
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [
            "whisper", audio_path,
            "--model", WHISPER_MODEL,
            "--language", "en",
            "--output_format", "json",
            "--word_timestamps", "True",
            "--output_dir", tmp,
            "--fp16", "False",
            "--verbose", "False",
        ]
        subprocess.run(cmd, check=True, capture_output=True)

        # find the JSON file whisper wrote
        json_files = [f for f in os.listdir(tmp) if f.endswith(".json")]
        if not json_files:
            return {"text": "", "words": []}

        with open(os.path.join(tmp, json_files[0])) as f:
            data = json.load(f)

    text  = data.get("text", "").strip()
    words = []
    for seg in data.get("segments", []):
        for w in seg.get("words", []):
            words.append((w["word"].strip(), w["start"], w["end"]))

    return {"text": text, "words": words}


def transcribe_all():
    if os.path.exists(TRANSCRIPT_CACHE):
        print("Loading cached transcripts …")
        return pd.read_csv(TRANSCRIPT_CACHE)

    print(f"Transcribing with Whisper {WHISPER_MODEL} (CLI) …")
    rows = []
    for label, folder in [(1, PD_DIR), (0, HC_DIR)]:
        for fname in sorted(os.listdir(folder)):
            if not fname.lower().endswith(".wav"):
                continue
            path = os.path.join(folder, fname)
            print(f"  {'PD' if label else 'HC'} {fname} …", flush=True)
            result = transcribe_file_cli(path)
            rows.append({
                "file":       fname,
                "label":      label,
                "text":       result["text"],
                "words_json": json.dumps(result["words"]),
            })
    df = pd.DataFrame(rows)
    df.to_csv(TRANSCRIPT_CACHE, index=False)
    return df


# ── Linguistic feature extraction ──────────────────────────────────────────────
FUNCTION_WORDS = {
    "the","a","an","is","are","was","were","be","been","being","have","has","had",
    "do","does","did","will","would","shall","should","may","might","must","can","could",
    "and","but","or","nor","so","yet","for","at","by","in","on","to","up","as","of",
    "if","then","than","that","this","these","those","it","its","i","you","he","she",
    "we","they","me","him","her","us","them","my","your","his","our","their","which",
    "who","what","when","where","how","with","from","into","through","during","not",
    "no","nor","both","either","neither","each","every","all","any","some","such",
}
FILLERS = {"uh","um","er","ah","like","okay","well","right","so"}


def tokenize(text: str) -> list:
    return re.findall(r"\b[a-z']+\b", text.lower())


def extract_linguistic_features(text: str, words_json: str) -> dict:
    feats  = {}
    tokens = tokenize(text)
    sentences = [s.strip() for s in re.split(r"[.!?]+", text.strip()) if s.strip()]

    n_tokens    = len(tokens)
    n_chars     = len(text.replace(" ", ""))
    n_sentences = max(len(sentences), 1)
    vocab       = set(tokens)

    # Lexical richness
    feats["type_token_ratio"]  = len(vocab) / max(n_tokens, 1)
    feats["vocab_size"]        = len(vocab)
    feats["total_word_count"]  = n_tokens

    # Word-level stats
    wl = [len(t) for t in tokens]
    feats["avg_word_length"] = np.mean(wl) if wl else 0.0
    feats["std_word_length"] = np.std(wl)  if wl else 0.0
    feats["max_word_length"] = max(wl)     if wl else 0.0

    # Sentence stats
    sl = [len(tokenize(s)) for s in sentences]
    feats["avg_sentence_length"] = np.mean(sl) if sl else 0.0
    feats["std_sentence_length"] = np.std(sl)  if sl else 0.0
    feats["num_sentences"]       = n_sentences

    # Function / content word ratio
    n_func = sum(1 for t in tokens if t in FUNCTION_WORDS)
    feats["function_word_ratio"] = n_func / max(n_tokens, 1)
    feats["content_word_ratio"]  = 1.0 - feats["function_word_ratio"]
    feats["lexical_density"]     = feats["content_word_ratio"]

    # Repetition / perseveration
    bigrams  = list(zip(tokens, tokens[1:]))
    trigrams = list(zip(tokens, tokens[1:], tokens[2:]))
    feats["bigram_repetition_ratio"]  = (len(bigrams)  - len(set(bigrams)))  / max(len(bigrams),  1)
    feats["trigram_repetition_ratio"] = (len(trigrams) - len(set(trigrams))) / max(len(trigrams), 1)
    feats["top_word_dominance"] = (Counter(tokens).most_common(1)[0][1] / n_tokens) if tokens else 0.0

    # Disfluencies
    n_fill = sum(1 for t in tokens if t in FILLERS)
    feats["filler_ratio"] = n_fill / max(n_tokens, 1)
    feats["filler_count"] = n_fill

    # Punctuation density
    feats["punctuation_density"] = sum(1 for c in text if c in ".,;:!?") / max(len(text), 1)
    feats["avg_commas_per_sentence"] = text.count(",") / max(n_sentences, 1)

    # Timing features from Whisper word timestamps
    try:
        word_list = json.loads(words_json)   # list of [word, start, end]
    except Exception:
        word_list = []

    if len(word_list) >= 2:
        durations   = [e - s for _, s, e in word_list]
        gaps        = [word_list[i+1][1] - word_list[i][2] for i in range(len(word_list)-1)]
        gaps_clean  = [g for g in gaps if g >= 0]
        total_speech = sum(durations)
        total_time   = word_list[-1][2] - word_list[0][1]

        feats["avg_word_duration"]     = np.mean(durations)
        feats["std_word_duration"]     = np.std(durations)
        feats["avg_gap_between_words"] = np.mean(gaps_clean) if gaps_clean else 0.0
        feats["std_gap_between_words"] = np.std(gaps_clean)  if gaps_clean else 0.0
        feats["max_pause"]             = max(gaps_clean)      if gaps_clean else 0.0
        feats["long_pause_count"]      = sum(1 for g in gaps_clean if g > 0.5)
        feats["long_pause_ratio"]      = feats["long_pause_count"] / max(len(gaps_clean), 1)
        feats["speech_rate_wps"]       = len(word_list) / max(total_time, 1e-3)
        feats["articulation_rate"]     = len(word_list) / max(total_speech, 1e-3)
        feats["speaking_time_ratio"]   = total_speech / max(total_time, 1e-3)
        feats["chars_per_second"]      = n_chars / max(total_time, 1e-3)
    else:
        for k in ["avg_word_duration","std_word_duration","avg_gap_between_words",
                  "std_gap_between_words","max_pause","long_pause_count","long_pause_ratio",
                  "speech_rate_wps","articulation_rate","speaking_time_ratio","chars_per_second"]:
            feats[k] = 0.0

    return feats


# ── Build dataset ──────────────────────────────────────────────────────────────
trans_df = transcribe_all()
print(f"\nTranscribed {len(trans_df)} files.")

# Delete old cache if words_json is in old str format (not JSON)
sample = str(trans_df["words_json"].iloc[0])
if sample.startswith("[('") or sample.startswith("[(\""):
    print("Old transcript format detected — re-transcribing with base.en …")
    os.remove(TRANSCRIPT_CACHE)
    trans_df = transcribe_all()

print("Extracting linguistic features …")
records = []
for _, row in trans_df.iterrows():
    f = extract_linguistic_features(row["text"], row["words_json"])
    f["label"] = row["label"];  f["file"] = row["file"]
    records.append(f)
    print(f"  {'PD' if row['label'] else 'HC'} {row['file']} — preview: \"{str(row['text'])[:60]}\"")

df = pd.DataFrame(records)
df.to_csv(os.path.join(OUT, "linguistic_features.csv"), index=False)

feature_cols = [c for c in df.columns if c not in ("label", "file")]
X        = df[feature_cols].values.astype(float)
y_labels = df["label"].values
print(f"\nDataset: {len(df)} samples | {df['label'].sum()} PD / {(df['label']==0).sum()} HC | {len(feature_cols)} features")


# ── Pipeline: Scale → MI feature selection → SMOTE → XGBoost ─────────────────
N_FEATURES = min(15, len(feature_cols))

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
    ("scaler", StandardScaler()),
    ("select", SelectKBest(mutual_info_classif, k=N_FEATURES)),
    ("smote",  SMOTE(random_state=42, k_neighbors=3)),
    ("clf",    clf),
])


# ── Leave-One-Out CV ───────────────────────────────────────────────────────────
print("\nRunning Leave-One-Out CV …")
loo = LeaveOneOut()

y_pred_loo  = []
y_proba_loo = []

for train_idx, test_idx in loo.split(X):
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr        = y_labels[train_idx]
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

pd.DataFrame([{"Metric": m, "Value": v} for m, v in [
    ("Accuracy",acc),("Precision",prec),("Recall",rec),("F1",f1),("ROC-AUC",auc)
]]).to_csv(os.path.join(OUT, "loo_metrics.csv"), index=False)


# ── Confusion matrix & ROC ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ConfusionMatrixDisplay(confusion_matrix(y_labels, y_pred_loo), display_labels=["HC","PD"]).plot(ax=axes[0], colorbar=False)
axes[0].set_title("Linguistic Model — LOO Confusion Matrix")
RocCurveDisplay.from_predictions(y_labels, y_proba_loo, ax=axes[1], name="XGBoost Linguistic (LOO)")
axes[1].set_title(f"Linguistic Model — ROC Curve (AUC={auc:.3f})")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "confusion_roc.png"), dpi=150)
plt.close()


# ── Final model on full data for interpretability ──────────────────────────────
print("Fitting final model on full data …")
pipe.fit(X, y_labels)

scaler_full  = pipe.named_steps["scaler"]
select_full  = pipe.named_steps["select"]
clf_full     = pipe.named_steps["clf"]
sel_mask     = select_full.get_support()
sel_feats    = [f for f, m in zip(feature_cols, sel_mask) if m]
X_sel        = select_full.transform(scaler_full.transform(X))

imp_df = pd.DataFrame({"feature": sel_feats, "importance": clf_full.feature_importances_})
imp_df = imp_df.sort_values("importance", ascending=False).reset_index(drop=True)
imp_df.to_csv(os.path.join(OUT, "feature_importance.csv"), index=False)

fig, ax = plt.subplots(figsize=(10, 7))
sns.barplot(data=imp_df, y="feature", x="importance", ax=ax, palette="magma")
ax.set_title("Linguistic Model — Feature Importances (MI-selected)")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "feature_importance.png"), dpi=150)
plt.close()

print(f"\nTop 10 features:\n{imp_df.head(10).to_string(index=False)}")

# SHAP
print("\nRunning SHAP analysis …")
X_sel_df   = pd.DataFrame(X_sel, columns=sel_feats)
explainer  = shap.TreeExplainer(clf_full)
shap_vals  = explainer.shap_values(X_sel_df)

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_vals, X_sel_df, show=False, max_display=15)
plt.title("Linguistic Model — SHAP Summary (Beeswarm)")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "shap_summary.png"), dpi=150, bbox_inches="tight")
plt.close()

plt.figure(figsize=(10, 7))
shap.summary_plot(shap_vals, X_sel_df, plot_type="bar", show=False, max_display=15)
plt.title("Linguistic Model — SHAP Feature Importance (Bar)")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "shap_bar.png"), dpi=150, bbox_inches="tight")
plt.close()

pd.DataFrame(shap_vals, columns=sel_feats).assign(
    file=df["file"].values, label=df["label"].values
).to_csv(os.path.join(OUT, "shap_values.csv"), index=False)

print(f"\nAll results saved to: {OUT}")
print("Done.")
