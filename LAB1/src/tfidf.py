"""From-scratch TF-IDF vectorizer (word 1-2 grams + char 2/3/4-grams).

No sklearn, no scipy.sparse. Documents are converted to feature-count dicts,
a vocabulary + IDF table is learned from the training corpus, and vectors are
returned as our own CSRMatrix (see sparse.py), L2-normalized.
"""
import math
import re
from collections import Counter

import numpy as np

from sparse import CSRMatrix

_TOKEN_RE = re.compile(r'\S+')


def _words(text):
    return _TOKEN_RE.findall(text)


def _char_ngrams(word, n):
    padded = f" {word} "
    L = len(padded)
    if L < n:
        return []
    return [padded[i:i + n] for i in range(L - n + 1)]


def extract_features(text):
    """Return a Counter of raw feature strings for one sentence."""
    words = _words(text)
    feats = Counter()

    for w in words:
        feats[f"W1:{w}"] += 1
    for i in range(len(words) - 1):
        feats[f"W2:{words[i]}_{words[i + 1]}"] += 1

    for w in words:
        for n in (2, 3, 4):
            for g in _char_ngrams(w, n):
                feats[f"C{n}:{g}"] += 1

    return feats


class TfidfVectorizer:
    def __init__(self, min_df=2, max_features=150_000):
        self.min_df = min_df
        self.max_features = max_features
        self.vocab_ = None       # feature string -> column index
        self.idf_ = None         # np.ndarray, aligned with vocab_ indices
        self.n_features_ = 0

    def fit(self, texts):
        df = Counter()
        doc_feats = []
        for t in texts:
            f = extract_features(t)
            doc_feats.append(f)
            df.update(f.keys())

        n_docs = len(texts)
        candidates = [(feat, d) for feat, d in df.items() if d >= self.min_df]
        candidates.sort(key=lambda x: x[1], reverse=True)
        if self.max_features is not None and len(candidates) > self.max_features:
            candidates = candidates[:self.max_features]

        self.vocab_ = {feat: i for i, (feat, _) in enumerate(candidates)}
        self.n_features_ = len(self.vocab_)

        idf = np.zeros(self.n_features_, dtype=np.float64)
        for feat, d in candidates:
            j = self.vocab_[feat]
            idf[j] = math.log((1 + n_docs) / (1 + d)) + 1.0
        self.idf_ = idf

        return self._transform_docs(doc_feats)

    def transform(self, texts):
        doc_feats = [extract_features(t) for t in texts]
        return self._transform_docs(doc_feats)

    def _transform_docs(self, doc_feats):
        if self.vocab_ is None:
            raise RuntimeError("Vectorizer not fitted yet")
        rows = []
        for feats in doc_feats:
            row = {}
            for feat, tf in feats.items():
                j = self.vocab_.get(feat)
                if j is None:
                    continue
                row[j] = tf * self.idf_[j]
            norm = math.sqrt(sum(v * v for v in row.values()))
            if norm > 0:
                row = {j: v / norm for j, v in row.items()}
            rows.append(row)
        return CSRMatrix.from_rows(rows, self.n_features_)
