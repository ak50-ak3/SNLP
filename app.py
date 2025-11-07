# resume_app.py
# Streamlit app: Skills search + summaries, Naive Bayes resume classifier, and n-gram analytics (χ² only, no PMI).

import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import spacy

from typing import List, Set
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import chi2
from sklearn.naive_bayes import MultinomialNB

st.set_page_config(page_title="Resume SNLP App", layout="wide")

# --------------------------
# Skill cleaning helpers
# --------------------------

SKILL_STOPWORDS = {
    "basic","basics","beginner","intermediate","advanced","fresher","experienced",
    "project","projects","resume","curriculum","vitae","cv","contact","phone","email",
    "male","female","married","single","hobbies","strengths","weaknesses",
    "objective","summary","declaration","work","experience","education","university","college",
    "year","years","month","months","role","responsibility","responsibilities"
    
}

_re_starts_with_letter = re.compile(r"^[A-Za-z]")
_re_has_digit           = re.compile(r"\d")
_re_forbidden_dash      = re.compile(r"-")
_re_letters_spaces_only = re.compile(r"^[A-Za-z &/\.#\+]+$")  # allow C#, C++, .NET, R&D, Excel

def normalize_skill(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"^\s*[•·\-–—>]+\s*", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def is_valid_skill(s: str) -> bool:
    if not s:
        return False
    if not _re_starts_with_letter.match(s):
        return False
    if _re_has_digit.search(s):
        return False
    if _re_forbidden_dash.search(s):
        return False
    if len(s) < 3:
        return False
    if s.lower() in SKILL_STOPWORDS:
        return False
    if not _re_letters_spaces_only.match(s):
        return False
    return True

def prettify_skill(s: str) -> str:
    if " " in s:
        return " ".join(w.capitalize() for w in s.split())
    return s if s.isupper() else s.capitalize()

# --------------------------
# Core app code
# --------------------------

@st.cache_resource
def load_spacy():
    return spacy.load("en_core_web_sm")

@st.cache_data
def load_data(path: str):
    df = pd.read_csv(path)
    df = df.dropna(subset=["Resume"]).reset_index(drop=True)
    if "Category" not in df.columns:
        df["Category"] = "Unknown"
    return df

@st.cache_data
def simple_preclean(text: str) -> str:
    t = re.sub(r"\s+", " ", str(text)).strip()
    return t

def extract_skills_spacy(doc) -> Set[str]:
    candidates = set()

    for ent in doc.ents:
        if ent.label_ in {"ORG","PRODUCT","WORK_OF_ART","FAC","GPE","NORP"}:
            candidates.add(ent.text)

    for nc in doc.noun_chunks:
        candidates.add(nc.text)

    for tok in doc:
        if tok.pos_ in {"PROPN","NOUN"} and tok.is_alpha and len(tok.text) > 2:
            candidates.add(tok.text)

    cleaned = set()
    for c in candidates:
        s = normalize_skill(c)
        if is_valid_skill(s):
            cleaned.add(prettify_skill(s))
    return cleaned

@st.cache_data
def compute_skills_column(df: pd.DataFrame) -> pd.DataFrame:
    nlp = load_spacy()
    skills_col = []
    for txt in df["Resume"].tolist():
        doc = nlp(simple_preclean(txt).lower())
        skills_col.append(extract_skills_spacy(doc))
    out = df.copy()
    out["skills"] = skills_col
    return out

@st.cache_resource
def train_nb(df: pd.DataFrame, ngram_max: int = 2):
    X = df["Resume"].apply(simple_preclean).tolist()
    y = df["Category"].astype(str).tolist()
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    vect = TfidfVectorizer(lowercase=True, stop_words="english", max_features=30000, ngram_range=(1, ngram_max))
    Xtr = vect.fit_transform(X_train)
    Xva = vect.transform(X_valid)

    nb = MultinomialNB()
    nb.fit(Xtr, y_train)

    pred = nb.predict(Xva)
    acc = accuracy_score(y_valid, pred)
    f1 = f1_score(y_valid, pred, average="weighted")
    cm = confusion_matrix(y_valid, pred, labels=sorted(list(set(y))))

    report = classification_report(y_valid, pred, zero_division=0, output_dict=True)
    labels_sorted = sorted(list(set(y)))
    return nb, vect, acc, f1, cm, labels_sorted, report

def predict_category(text: str, model, vect, top_k: int = 5):
    X = vect.transform([simple_preclean(text)])
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        classes = model.classes_
        idx = np.argsort(proba)[::-1][:top_k]
        ranked = [(classes[i], float(proba[i])) for i in idx]
        return ranked
    else:
        yhat = model.predict(X)[0]
        return [(yhat, 1.0)]

def summarize_text_tfidf(text: str, n_sent: int = 3):
    nlp = load_spacy()
    sents = [s.text.strip() for s in nlp(simple_preclean(text)).sents]
    if len(sents) <= n_sent:
        return " ".join(sents)
    vec = TfidfVectorizer(stop_words="english", max_features=2000)
    X = vec.fit_transform(sents)
    scores = X.sum(axis=1).A.ravel()
    idx = np.argsort(scores)[-n_sent:]
    return " ".join([sents[i] for i in sorted(idx)])

def top_chi2_ngrams(texts: List[str], labels: List[str], ngram_low=1, ngram_high=2, k=20):
    vec = CountVectorizer(ngram_range=(ngram_low, ngram_high), min_df=2, lowercase=True, stop_words="english", max_features=50000)
    X = vec.fit_transform(texts)
    y = np.array(labels)
    classes = sorted(list(set(y)))
    feats = np.array(vec.get_feature_names_out())
    result = {}
    for c in classes:
        y_bin = (y == c).astype(int)
        chi, _ = chi2(X, y_bin)
        top_idx = np.argsort(-chi)[:k]
        result[c] = pd.DataFrame({"ngram": feats[top_idx], "chi2": chi[top_idx]})
    return result

df = load_data("UpdatedResumeDataSet.csv")
df = compute_skills_column(df)

st.title("Resume SNLP App")
tabs = st.tabs(["Search by Skill", "NB Classifier", "N-gram Analytics"])

with tabs[0]:
    st.subheader("Search by Skill and Summarize Candidates")

    raw_skills = sorted(set().union(*df["skills"].tolist())) if len(df) else []
    all_skills = [s for s in raw_skills if is_valid_skill(s)]

    if not all_skills:
        st.info("No skills extracted. Try again after data loads.")
    else:
        skill = st.selectbox("Pick a skill", all_skills)
        subset = df[df["skills"].apply(lambda s: skill.lower() in set(map(str.lower, s)))]
        st.write(f"Found {len(subset)} resumes with skill: {skill}")
        for i, row in subset.head(50).iterrows():
            st.markdown("---")
            st.write(f"Category: {row['Category']}")
            st.write("Summary:")
            st.write(summarize_text_tfidf(row["Resume"], n_sent=3))

with tabs[1]:
    st.subheader("Naive Bayes Resume Classifier")

    # Fixed n-gram max at 2 (unigram + bigram)
    nmax = 2
    st.caption("TF-IDF n-gram max: **2 (fixed)**")

    nb, vect, acc, f1, cm, labels_sorted, report = train_nb(df, ngram_max=nmax)
    left, right = st.columns(2)
    with left:
        st.write(f"Validation Accuracy: {round(acc,4)}")
        st.write(f"Validation F1 (weighted): {round(f1,4)}")
        st.write("Classification Report:")
        st.dataframe(pd.DataFrame(report).T)
    with right:
        fig, ax = plt.subplots()
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels_sorted)))
        ax.set_yticks(range(len(labels_sorted)))
        ax.set_xticklabels(labels_sorted, rotation=45, ha="right")
        ax.set_yticklabels(labels_sorted)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center", fontsize=9)
        st.pyplot(fig)

    st.markdown("---")
    st.write("Try a resume snippet for prediction")
    sample_txt = st.text_area("Paste resume text here", height=200)
    if st.button("Predict Category"):
        if sample_txt.strip():
            ranked = predict_category(sample_txt, nb, vect, top_k=5)
            st.write("Top predictions:")
            st.table(pd.DataFrame(ranked, columns=["Category", "Probability"]))
        else:
            st.warning("Please paste some text.")

with tabs[2]:
    st.subheader("N-gram Analytics (χ² only)")
    n_low = st.selectbox("Min n", [1, 2], index=1)
    n_high = st.selectbox("Max n", [2, 3], index=0)
    topk = st.slider("Top n-grams", 5, 40, 20, 1)

    texts = df["Resume"].apply(simple_preclean).tolist()
    labels = df["Category"].astype(str).tolist()

    st.write("Top χ² n-grams per category")
    chi_by_cat = top_chi2_ngrams(texts, labels, ngram_low=n_low, ngram_high=n_high, k=topk)
    pick_cat = st.selectbox("Category", sorted(chi_by_cat.keys()))
    chi_df = chi_by_cat[pick_cat]
    st.dataframe(chi_df)

    fig1, ax1 = plt.subplots()
    ax1.barh(chi_df["ngram"][::-1], chi_df["chi2"][::-1])
    ax1.set_xlabel("chi2")
    ax1.set_ylabel("ngram")
    ax1.set_title(f"Top χ² n-grams for {pick_cat}")
    st.pyplot(fig1)
