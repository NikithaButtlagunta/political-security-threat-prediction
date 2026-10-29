"""Hybrid TF-IDF + lexicon-feature text classifier."""
from __future__ import annotations

import re
from typing import Any

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


POSITIVE_WORDS = {
    "peace", "stable", "support", "secure", "safe", "progress", "unity",
    "hope", "positive", "calm", "fair", "trust", "improve", "success",
    "cooperation", "democracy", "development", "reform", "resolve",
}
NEGATIVE_WORDS = {
    "attack", "violence", "violent", "threat", "threaten", "terror",
    "terrorism", "hate", "anger", "fear", "crisis", "conflict", "war",
    "unstable", "danger", "dangerous", "riot", "corruption", "abuse",
    "extremist", "disrupt", "weapon", "kill", "bomb", "chaos",
}
NEGATIONS = {"not", "no", "never", "neither", "without", "hardly"}


def lexicon_score(text: str) -> float:
    """Return a transparent normalized word-count score in [-1, 1]."""
    tokens = re.findall(r"[a-zA-Z']+", str(text).lower())
    if not tokens:
        return 0.0
    score = 0
    for i, token in enumerate(tokens):
        value = 1 if token in POSITIVE_WORDS else (-1 if token in NEGATIVE_WORDS else 0)
        if value and i > 0 and tokens[i - 1] in NEGATIONS:
            value *= -1
        score += value
    return float(score / max(1, len(tokens)))


def _lexicon_column(texts: Any) -> np.ndarray:
    values = np.asarray(texts, dtype=object).reshape(-1)
    return np.array([[lexicon_score(value)] for value in values], dtype=float)


def build_pipeline(model_name: str) -> Pipeline:
    classifiers = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=30, min_samples_leaf=2, class_weight="balanced", random_state=42
        ),
        "Naive Bayes": MultinomialNB(),
        "SVM": LinearSVC(class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
        ),
    }
    if model_name not in classifiers:
        raise ValueError("Unsupported classifier.")
    features = FeatureUnion([
        ("tfidf", TfidfVectorizer(
            lowercase=True, strip_accents="unicode", ngram_range=(1, 2),
            max_features=30000, sublinear_tf=True
        )),
        ("lexicon", Pipeline([
            ("score", FunctionTransformer(_lexicon_column, validate=False)),
        ])),
    ])
    return Pipeline([("features", features), ("classifier", classifiers[model_name])])
