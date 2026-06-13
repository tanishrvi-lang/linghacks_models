"""
=============================================================
MODEL 6 — Fusion v2 (Acoustic + Linguistic + Semantic)
          Dementia Detection
Disease  : Dementia
Modality : Acoustic + Linguistic + SBERT Semantic (fusion)
Dataset  : 80% of subjects sampled per class
           (67 dementia / 36 no-dementia → 103 subjects)
Transcript: Whisper base.en (cached from Model 5)
Features : 465 features after mean+std+range aggregation
           per subject (115 acoustic + 34 linguistic +
           4 SBERT semantic coherence) × 3 aggregations
           MI selection keeps top 30
Models   : XGBoost, Random Forest, SVM, Voting Ensemble
           (all compared — best auto-selected)
Sampling : SMOTE + scale_pos_weight
CV       : 5-Fold Stratified CV
Best     : SVM (RBF kernel)
Results  : Accuracy 0.748 | Precision 0.789 | Recall 0.836
           F1 0.812 | ROC-AUC 0.692
=============================================================
Improvements over v1:
  1. Richer subject aggregation: mean + std + range per feature
  2. Dementia-specific linguistic features (semantic coherence, pronoun ratio,
     noun-finding pauses, vocabulary regression, sentence complexity)
  3. SBERT sentence embeddings for semantic coherence
  4. Class imbalance: scale_pos_weight + balanced SMOTE
  5. Model comparison: XGBoost vs SVM vs Random Forest vs Voting Ensemble
  6. GridSearchCV hyperparameter tuning on best model
"""

import os, re, json, warnings, random
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
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay, RocCurveDisplay,
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore")
random.seed(42)
np.random.seed(42)

BASE      = os.path.dirname(os.path.abspath(__file__))
DEM_DIR   = os.path.join(BASE, "dementia")
NODEM_DIR = os.path.join(BASE, "nodementia")
OUT       = os.path.join(BASE, "results_fusion_v2")
os.makedirs(OUT, exist_ok=True)

# Reuse v1 caches — no need to re-extract or re-transcribe
ACOU_CACHE  = os.path.join(BASE, "results_fusion", "acoustic_features.csv")
TRANS_CACHE = os.path.join(BASE, "results_fusion", "transcripts.csv")
SBERT_CACHE = os.path.join(OUT, "sbert_features.csv")


# ══════════════════════════════════════════════════════════════════════════════
# SAME 80% SUBJECT SAMPLE AS V1 (same seed → same subjects)
# ══════════════════════════════════════════════════════════════════════════════
def sample_subjects(folder, label, pct=0.80, seed=42):
    subjects = sorted([d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))])
    n = max(1, int(len(subjects) * pct))
    random.seed(seed)
    chosen = set(random.sample(subjects, n))
    rows = []
    for subj in chosen:
        for fname in sorted(os.listdir(os.path.join(folder, subj))):
            if fname.lower().endswith(".wav"):
                rows.append({"subject": subj, "file": fname, "label": label})
    return rows

dem_rows   = sample_subjects(DEM_DIR,   1)
nodem_rows = sample_subjects(NODEM_DIR, 0)
manifest   = pd.DataFrame(dem_rows + nodem_rows)
print(f"Subjects: {manifest['subject'].nunique()} | Clips: {len(manifest)}")
print(f"  Dementia: {manifest[manifest.label==1]['subject'].nunique()} subjects")
print(f"  No-Dementia: {manifest[manifest.label==0]['subject'].nunique()} subjects")


# ══════════════════════════════════════════════════════════════════════════════
# LOAD CACHED ACOUSTIC & TRANSCRIPT DATA
# ══════════════════════════════════════════════════════════════════════════════
print("\nLoading cached acoustic features …")
acou_df  = pd.read_csv(ACOU_CACHE)

print("Loading cached transcripts …")
trans_df = pd.read_csv(TRANS_CACHE)


# ══════════════════════════════════════════════════════════════════════════════
# IMPROVED LINGUISTIC FEATURES (dementia-specific)
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
FILLERS   = {"uh","um","er","ah","hmm"}
PRONOUNS  = {"he","she","it","they","him","her","them","this","that","these","those"}
# High-frequency "safe" words dementia patients retreat to
SAFE_WORDS= {"thing","stuff","know","like","get","got","go","came","said","went","make","put"}

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

    # ── Basic lexical ──
    f["li_ttr"]           = len(vocab) / max(n_tok, 1)
    f["li_vocab_size"]    = len(vocab)
    f["li_word_count"]    = n_tok
    wl = [len(t) for t in tokens]
    f["li_avg_word_len"]  = np.mean(wl) if wl else 0.0
    f["li_std_word_len"]  = np.std(wl)  if wl else 0.0
    f["li_max_word_len"]  = max(wl)     if wl else 0.0

    # ── Sentence stats ──
    sl = [len(tokenize(s)) for s in sents]
    f["li_avg_sent_len"]  = np.mean(sl) if sl else 0.0
    f["li_std_sent_len"]  = np.std(sl)  if sl else 0.0
    f["li_max_sent_len"]  = max(sl)     if sl else 0.0
    f["li_min_sent_len"]  = min(sl)     if sl else 0.0
    f["li_num_sentences"] = n_sent

    # ── Function / content word ──
    n_func = sum(1 for t in tokens if t in FUNCTION_WORDS)
    f["li_func_word_ratio"]    = n_func / max(n_tok, 1)
    f["li_content_word_ratio"] = 1.0 - f["li_func_word_ratio"]
    f["li_lexical_density"]    = f["li_content_word_ratio"]

    # ── Repetition / perseveration ──
    bigrams  = list(zip(tokens, tokens[1:]))
    trigrams = list(zip(tokens, tokens[1:], tokens[2:]))
    f["li_bigram_rep"]    = (len(bigrams)  - len(set(bigrams)))  / max(len(bigrams),  1)
    f["li_trigram_rep"]   = (len(trigrams) - len(set(trigrams))) / max(len(trigrams), 1)
    f["li_top_word_dom"]  = (Counter(tokens).most_common(1)[0][1] / n_tok) if tokens else 0.0

    # ── Fillers / disfluencies ──
    n_fill = sum(1 for t in tokens if t in FILLERS)
    f["li_filler_ratio"]  = n_fill / max(n_tok, 1)
    f["li_filler_count"]  = n_fill

    # ── Dementia-specific: pronoun overuse (vague reference) ──
    n_pron = sum(1 for t in tokens if t in PRONOUNS)
    f["li_pronoun_ratio"] = n_pron / max(n_tok, 1)

    # ── Dementia-specific: safe/vague word retreat ──
    n_safe = sum(1 for t in tokens if t in SAFE_WORDS)
    f["li_safe_word_ratio"] = n_safe / max(n_tok, 1)

    # ── Dementia-specific: vocabulary regression (% words in 500 most common) ──
    COMMON_500 = {
        "the","of","and","a","to","in","is","you","that","it","he","was","for","on",
        "are","as","with","his","they","i","at","be","this","have","from","or","one",
        "had","by","word","but","not","what","all","were","we","when","your","can",
        "said","there","use","an","each","which","she","do","how","their","if","will",
        "up","other","about","out","many","then","them","these","so","some","her","would",
        "make","like","him","into","time","has","look","two","more","write","go","see",
        "number","no","way","could","people","my","than","first","water","been","call",
        "who","oil","its","now","find","long","down","day","did","get","come","made",
        "may","part","over","new","sound","take","only","little","work","know","place",
        "years","live","me","back","give","most","very","after","thing","our","just",
        "name","good","sentence","man","think","say","great","where","help","through",
        "much","before","line","right","too","mean","old","any","same","tell","boy",
        "follow","came","want","show","also","around","form","small","set","put","end",
        "does","another","well","large","need","big","high","such","turn","here","why",
        "ask","went","men","read","land","different","home","us","move","try","kind",
        "hand","picture","again","change","off","play","spell","air","away","animal",
        "house","point","page","letter","mother","answer","found","study","still","learn",
        "plant","cover","food","sun","four","between","state","keep","eye","never","last",
        "let","thought","city","tree","cross","farm","hard","start","might","story","saw",
        "far","sea","draw","left","late","run","while","press","close","night","real",
        "life","few","north","open","seem","together","next","white","children","begin",
        "got","walk","example","ease","paper","group","always","music","those","both",
        "mark","book","often","until","mile","river","car","feet","care","second","enough",
        "plain","girl","usual","young","ready","above","ever","red","list","though","feel",
        "talk","bird","soon","body","dog","family","direct","pose","leave","song","measure",
        "door","product","black","short","numeral","class","wind","question","happen",
        "complete","ship","area","half","rock","order","fire","south","problem","piece",
        "told","knew","pass","since","top","whole","king","space","heard","best","hour",
        "better","true","during","hundred","five","remember","step","early","hold","west",
        "ground","interest","reach","fast","verb","sing","listen","six","table","travel",
        "less","morning","ten","simple","several","vowel","toward","war","lay","against",
        "pattern","slow","center","love","person","money","serve","appear","road","map",
        "rain","rule","govern","pull","cold","notice","voice","unit","power","town","fine",
        "drive","explain","color","face","wood","main","open","seem","together","next"
    }
    n_common = sum(1 for t in tokens if t in COMMON_500)
    f["li_vocab_regression"] = n_common / max(n_tok, 1)

    # ── Punctuation ──
    f["li_punct_density"]       = sum(1 for c in text if c in ".,;:!?") / max(len(text), 1)
    f["li_commas_per_sentence"] = text.count(",") / max(n_sent, 1)

    # ── Timing features from Whisper word timestamps ──
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

        # Noun-finding pause: pause BEFORE longer words (proxy for word retrieval difficulty)
        long_word_pre_gaps = []
        for i, (word, _, _) in enumerate(wl_list[1:], 1):
            if len(word) >= 6 and i > 0:
                g = wl_list[i][1] - wl_list[i-1][2]
                if g >= 0:
                    long_word_pre_gaps.append(g)
        f["li_noun_finding_pause"] = np.mean(long_word_pre_gaps) if long_word_pre_gaps else 0.0
    else:
        for k in ["li_avg_word_dur","li_std_word_dur","li_avg_gap","li_std_gap",
                  "li_max_pause","li_long_pause_count","li_long_pause_ratio",
                  "li_speech_rate","li_articu_rate","li_speaking_ratio","li_chars_per_sec",
                  "li_noun_finding_pause"]:
            f[k] = 0.0

    return f


# ══════════════════════════════════════════════════════════════════════════════
# SBERT SEMANTIC COHERENCE FEATURES
# ══════════════════════════════════════════════════════════════════════════════
def extract_sbert_features(text, model):
    """
    Encode each sentence with SBERT, then compute:
    - mean cosine similarity between consecutive sentences (coherence)
    - std of similarities (consistency)
    - min similarity (most incoherent moment)
    """
    sents = [s.strip() for s in re.split(r"[.!?]+", text.strip()) if len(s.strip()) > 10]
    if len(sents) < 2:
        return {"sbert_coherence_mean": 0.5, "sbert_coherence_std": 0.0,
                "sbert_coherence_min": 0.5,  "sbert_topic_drift": 0.0}

    embs = model.encode(sents, convert_to_numpy=True, show_progress_bar=False)
    # Normalize
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs  = embs / (norms + 1e-9)

    # Consecutive cosine similarities
    sims = [float(np.dot(embs[i], embs[i+1])) for i in range(len(embs)-1)]

    # Topic drift: similarity between first and last sentence
    topic_drift = float(np.dot(embs[0], embs[-1]))

    return {
        "sbert_coherence_mean": np.mean(sims),
        "sbert_coherence_std":  np.std(sims),
        "sbert_coherence_min":  np.min(sims),
        "sbert_topic_drift":    topic_drift,
    }


# ══════════════════════════════════════════════════════════════════════════════
# EXTRACT ALL LINGUISTIC + SBERT FEATURES
# ══════════════════════════════════════════════════════════════════════════════
print("\nExtracting improved linguistic features …")
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
    print("Computing SBERT semantic coherence features …")
    sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
    sbert_records = []
    for i, row in trans_df.iterrows():
        feat = extract_sbert_features(str(row["text"]), sbert_model)
        feat["subject"] = row["subject"]
        feat["file"]    = row["file"]
        sbert_records.append(feat)
        if (i + 1) % 30 == 0:
            print(f"  {i+1}/{len(trans_df)} done …")
    sbert_df = pd.DataFrame(sbert_records)
    sbert_df.to_csv(SBERT_CACHE, index=False)
    print(f"  SBERT complete.")


# ══════════════════════════════════════════════════════════════════════════════
# RICHER SUBJECT-LEVEL AGGREGATION: mean + std + range
# ══════════════════════════════════════════════════════════════════════════════
acou_cols  = [c for c in acou_df.columns  if c.startswith("ac_")]
ling_cols  = [c for c in ling_df.columns  if c.startswith("li_")]
sbert_cols = [c for c in sbert_df.columns if c.startswith("sbert_")]

def aggregate_subject(df, feat_cols, subject_labels):
    """Compute mean, std, and range (max-min) per subject per feature."""
    rows = []
    for subj, grp in df.groupby("subject"):
        row = {"subject": subj}
        for col in feat_cols:
            vals = grp[col].dropna().values
            row[f"{col}_mean"]  = np.mean(vals) if len(vals) else 0.0
            row[f"{col}_std"]   = np.std(vals)  if len(vals) else 0.0
            row[f"{col}_range"] = (np.max(vals) - np.min(vals)) if len(vals) else 0.0
        rows.append(row)
    agg = pd.DataFrame(rows)
    agg = agg.merge(subject_labels, on="subject")
    return agg

subject_labels = manifest.groupby("subject")["label"].first().reset_index()

print("\nAggregating features per subject (mean + std + range) …")
acou_sub  = aggregate_subject(acou_df,  acou_cols,  subject_labels)
ling_sub  = aggregate_subject(ling_df,  ling_cols,  subject_labels)
sbert_sub = aggregate_subject(sbert_df, sbert_cols, subject_labels)

# Merge all three
merged = acou_sub.merge(ling_sub.drop(columns="label"), on="subject")
merged = merged.merge(sbert_sub.drop(columns="label"), on="subject")

all_feat_cols = [c for c in merged.columns if c not in ("subject","label")]
feat_modality = {}
for c in all_feat_cols:
    if c.startswith("ac_"):       feat_modality[c] = "Acoustic"
    elif c.startswith("li_"):     feat_modality[c] = "Linguistic"
    elif c.startswith("sbert_"):  feat_modality[c] = "Semantic (SBERT)"

X        = merged[all_feat_cols].values.astype(float)
y_labels = merged["label"].values

n_dem   = y_labels.sum()
n_nodem = (y_labels == 0).sum()
spw     = n_nodem / n_dem   # scale_pos_weight to correct imbalance

print(f"\nFinal dataset: {len(merged)} subjects")
print(f"  Dementia    : {n_dem}")
print(f"  No-Dementia : {n_nodem}")
print(f"  scale_pos_weight: {spw:.2f}")
print(f"  Total features  : {len(all_feat_cols)}")


# ══════════════════════════════════════════════════════════════════════════════
# MODEL COMPARISON: XGBoost vs SVM vs Random Forest vs Voting Ensemble
# ══════════════════════════════════════════════════════════════════════════════
N_FEATURES = 30
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def make_pipeline(estimator):
    return ImbPipeline([
        ("scaler", StandardScaler()),
        ("select", SelectKBest(mutual_info_classif, k=N_FEATURES)),
        ("smote",  SMOTE(random_state=42, k_neighbors=5,
                         sampling_strategy=min(1.0, n_nodem/n_dem * 1.5))),
        ("clf",    estimator),
    ])

models = {
    "XGBoost": xgb.XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw,
        use_label_encoder=False, eval_metric="logloss", random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=400, max_depth=6, class_weight="balanced",
        random_state=42, n_jobs=-1,
    ),
    "SVM": SVC(
        kernel="rbf", C=1.0, gamma="scale",
        class_weight="balanced", probability=True, random_state=42,
    ),
}

print("\n" + "="*65)
print("  MODEL COMPARISON (5-Fold Stratified CV)")
print("="*65)

comparison_rows = []
best_auc   = 0
best_name  = None
best_pipe  = None

for name, estimator in models.items():
    pipe = make_pipeline(estimator)
    res  = cross_validate(pipe, X, y_labels, cv=cv,
                          scoring=["accuracy","precision","recall","f1","roc_auc"],
                          return_train_score=False)
    row = {
        "Model":     name,
        "Accuracy":  res["test_accuracy"].mean(),
        "Precision": res["test_precision"].mean(),
        "Recall":    res["test_recall"].mean(),
        "F1":        res["test_f1"].mean(),
        "ROC-AUC":   res["test_roc_auc"].mean(),
        "Acc_std":   res["test_accuracy"].std(),
        "AUC_std":   res["test_roc_auc"].std(),
    }
    comparison_rows.append(row)
    print(f"\n  {name}:")
    print(f"    Accuracy : {row['Accuracy']:.4f} ± {row['Acc_std']:.4f}")
    print(f"    Precision: {row['Precision']:.4f}")
    print(f"    Recall   : {row['Recall']:.4f}")
    print(f"    F1       : {row['F1']:.4f}")
    print(f"    ROC-AUC  : {row['ROC-AUC']:.4f} ± {row['AUC_std']:.4f}")

    if row["ROC-AUC"] > best_auc:
        best_auc  = row["ROC-AUC"]
        best_name = name
        best_pipe = make_pipeline(estimator)

# Voting Ensemble
print("\n  Building Voting Ensemble …")
xgb_clf = xgb.XGBClassifier(
    n_estimators=400, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
    use_label_encoder=False, eval_metric="logloss", random_state=42,
)
rf_clf  = RandomForestClassifier(
    n_estimators=400, max_depth=6, class_weight="balanced",
    random_state=42, n_jobs=-1,
)
svm_clf = SVC(
    kernel="rbf", C=1.0, gamma="scale",
    class_weight="balanced", probability=True, random_state=42,
)
ensemble = VotingClassifier(
    estimators=[("xgb", xgb_clf), ("rf", rf_clf), ("svm", svm_clf)],
    voting="soft",
)
ens_pipe = make_pipeline(ensemble)
res = cross_validate(ens_pipe, X, y_labels, cv=cv,
                     scoring=["accuracy","precision","recall","f1","roc_auc"],
                     return_train_score=False)
ens_row = {
    "Model":     "Voting Ensemble",
    "Accuracy":  res["test_accuracy"].mean(),
    "Precision": res["test_precision"].mean(),
    "Recall":    res["test_recall"].mean(),
    "F1":        res["test_f1"].mean(),
    "ROC-AUC":   res["test_roc_auc"].mean(),
    "Acc_std":   res["test_accuracy"].std(),
    "AUC_std":   res["test_roc_auc"].std(),
}
comparison_rows.append(ens_row)
print(f"\n  Voting Ensemble:")
print(f"    Accuracy : {ens_row['Accuracy']:.4f} ± {ens_row['Acc_std']:.4f}")
print(f"    Precision: {ens_row['Precision']:.4f}")
print(f"    Recall   : {ens_row['Recall']:.4f}")
print(f"    F1       : {ens_row['F1']:.4f}")
print(f"    ROC-AUC  : {ens_row['ROC-AUC']:.4f} ± {ens_row['AUC_std']:.4f}")

if ens_row["ROC-AUC"] > best_auc:
    best_auc  = ens_row["ROC-AUC"]
    best_name = "Voting Ensemble"
    best_pipe = ens_pipe

comp_df = pd.DataFrame(comparison_rows)
comp_df.to_csv(os.path.join(OUT, "model_comparison.csv"), index=False)
print(f"\n  ★ Best model: {best_name} (ROC-AUC={best_auc:.4f})")


# ══════════════════════════════════════════════════════════════════════════════
# HYPERPARAMETER TUNING ON BEST MODEL (if XGBoost or RF)
# ══════════════════════════════════════════════════════════════════════════════
if best_name == "XGBoost":
    print("\nHyperparameter tuning (XGBoost GridSearch) …")
    param_grid = {
        "clf__n_estimators":  [200, 400],
        "clf__max_depth":     [3, 4, 5],
        "clf__learning_rate": [0.03, 0.05, 0.1],
        "clf__subsample":     [0.7, 0.9],
    }
    gs = GridSearchCV(best_pipe, param_grid, cv=cv, scoring="roc_auc",
                      n_jobs=-1, verbose=0)
    gs.fit(X, y_labels)
    best_pipe = gs.best_estimator_
    print(f"  Best params: {gs.best_params_}")
    print(f"  Best CV AUC: {gs.best_score_:.4f}")

elif best_name == "Random Forest":
    print("\nHyperparameter tuning (RF GridSearch) …")
    param_grid = {
        "clf__n_estimators": [200, 400, 600],
        "clf__max_depth":    [4, 6, 8, None],
        "clf__min_samples_leaf": [1, 2, 4],
    }
    gs = GridSearchCV(best_pipe, param_grid, cv=cv, scoring="roc_auc",
                      n_jobs=-1, verbose=0)
    gs.fit(X, y_labels)
    best_pipe = gs.best_estimator_
    print(f"  Best params: {gs.best_params_}")
    print(f"  Best CV AUC: {gs.best_score_:.4f}")

else:
    print(f"\nSkipping GridSearch for {best_name} (ensemble — too slow).")
    best_pipe.fit(X, y_labels)


# ══════════════════════════════════════════════════════════════════════════════
# FINAL EVALUATION WITH BEST TUNED MODEL
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nFinal CV evaluation with tuned {best_name} …")
y_pred_cv  = cross_val_predict(best_pipe, X, y_labels, cv=cv, method="predict")
y_proba_cv = cross_val_predict(best_pipe, X, y_labels, cv=cv, method="predict_proba")[:, 1]

acc  = accuracy_score(y_labels,  y_pred_cv)
prec = precision_score(y_labels, y_pred_cv, zero_division=0)
rec  = recall_score(y_labels,    y_pred_cv, zero_division=0)
f1   = f1_score(y_labels,        y_pred_cv, zero_division=0)
auc  = roc_auc_score(y_labels,   y_proba_cv)

print("\n── Final Results ──")
print(f"  Accuracy : {acc:.4f}")
print(f"  Precision: {prec:.4f}")
print(f"  Recall   : {rec:.4f}")
print(f"  F1-Score : {f1:.4f}")
print(f"  ROC-AUC  : {auc:.4f}")
print()
print(classification_report(y_labels, y_pred_cv, target_names=["No Dementia","Dementia"]))

pd.DataFrame([{"Metric": m, "Value": v} for m, v in [
    ("Accuracy",acc),("Precision",prec),("Recall",rec),("F1",f1),("ROC-AUC",auc)
]]).to_csv(os.path.join(OUT, "final_metrics.csv"), index=False)


# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════
# Model comparison bar chart
fig, ax = plt.subplots(figsize=(12, 6))
comp_plot = comp_df.set_index("Model")[["Accuracy","Precision","Recall","F1","ROC-AUC"]]
comp_plot.plot(kind="bar", ax=ax, colormap="Set2", edgecolor="black", linewidth=0.5)
ax.set_title("Dementia Model Comparison (5-Fold CV)")
ax.set_ylabel("Score")
ax.set_ylim(0, 1)
ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
ax.legend(loc="lower right")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "model_comparison.png"), dpi=150)
plt.close()

# Confusion matrix + ROC
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ConfusionMatrixDisplay(
    confusion_matrix(y_labels, y_pred_cv), display_labels=["No Dementia","Dementia"]
).plot(ax=axes[0], colorbar=False, cmap="Blues")
axes[0].set_title(f"Best Model ({best_name}) — Confusion Matrix")
RocCurveDisplay.from_predictions(
    y_labels, y_proba_cv, ax=axes[1], name=best_name
)
axes[1].set_title(f"ROC Curve (AUC={auc:.3f})")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "confusion_roc.png"), dpi=150)
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE + SHAP (fit on full data)
# ══════════════════════════════════════════════════════════════════════════════
print("\nFitting final model on full data for SHAP …")
best_pipe.fit(X, y_labels)

# Only do SHAP for XGBoost-based models (TreeExplainer)
try:
    if best_name in ["XGBoost", "Voting Ensemble"]:
        raise ValueError("Use KernelExplainer or skip for ensemble")

    scaler_f = best_pipe.named_steps["scaler"]
    select_f = best_pipe.named_steps["select"]
    clf_f    = best_pipe.named_steps["clf"]
    sel_mask = select_f.get_support()
    sel_feat = [f for f, m in zip(all_feat_cols, sel_mask) if m]
    X_sel    = select_f.transform(scaler_f.transform(X))

    imp_df = pd.DataFrame({
        "feature":    sel_feat,
        "importance": clf_f.feature_importances_,
        "modality":   [feat_modality[f] for f in sel_feat],
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    imp_df.to_csv(os.path.join(OUT, "feature_importance.csv"), index=False)

    X_df      = pd.DataFrame(X_sel, columns=sel_feat)
    explainer = shap.TreeExplainer(clf_f)
    shap_vals = explainer.shap_values(X_df)

    plt.figure(figsize=(12, 9))
    shap.summary_plot(shap_vals, X_df, show=False, max_display=25)
    plt.title(f"SHAP Summary — {best_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "shap_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_vals, X_df, plot_type="bar", show=False, max_display=25)
    plt.title(f"SHAP Bar — {best_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "shap_bar.png"), dpi=150, bbox_inches="tight")
    plt.close()

except Exception:
    # For ensemble/SVM: use XGBoost sub-model for SHAP
    print("  Using XGBoost sub-estimator for SHAP …")
    xgb_pipe = make_pipeline(xgb.XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05,
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
        "modality":   [feat_modality.get(f, "Unknown") for f in sel_feat],
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    imp_df.to_csv(os.path.join(OUT, "feature_importance.csv"), index=False)

    X_df      = pd.DataFrame(X_sel, columns=sel_feat)
    explainer = shap.TreeExplainer(clf_f)
    shap_vals = explainer.shap_values(X_df)

    plt.figure(figsize=(12, 9))
    shap.summary_plot(shap_vals, X_df, show=False, max_display=25)
    plt.title("SHAP Summary (XGBoost component)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "shap_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_vals, X_df, plot_type="bar", show=False, max_display=25)
    plt.title("SHAP Bar (XGBoost component)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "shap_bar.png"), dpi=150, bbox_inches="tight")
    plt.close()

print(f"\nTop 15 features:\n{imp_df.head(15).to_string(index=False)}")

modal_imp = imp_df.groupby("modality")["importance"].sum().sort_values(ascending=False)
print(f"\nImportance by modality:\n{modal_imp.to_string()}")

palette = {"Acoustic": "#1565C0", "Linguistic": "#E64A19", "Semantic (SBERT)": "#2E7D32"}
fig, ax = plt.subplots(figsize=(13, 9))
sns.barplot(data=imp_df, y="feature", x="importance",
            hue="modality", dodge=False, palette=palette, ax=ax)
ax.set_title("Feature Importances by Modality")
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


# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  FINAL SUMMARY — v1 vs v2")
print("="*60)
print(f"  {'Metric':<12} {'v1 (XGBoost)':>14} {'v2 ('+best_name+')':>20}")
print("-"*60)
v1 = [("Accuracy","0.631"),("Precision","0.699"),
      ("Recall","0.761"),("F1","0.729"),("ROC-AUC","0.559")]
v2 = [acc, prec, rec, f1, auc]
for (name, v1val), v2val in zip(v1, v2):
    print(f"  {name:<12} {v1val:>14} {v2val:>20.4f}")
print("="*60)
print(f"\nAll results saved to: {OUT}")
print("Done.")
