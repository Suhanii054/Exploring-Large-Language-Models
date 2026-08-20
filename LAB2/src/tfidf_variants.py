"""[Lab 2] Configurable TF-IDF vectorizer supporting multiple TF/IDF
normalization schemes, built the same way as Lab 1 (no sklearn, no
scipy.sparse) but parameterized so all 6 combinations can be produced.

Features are the same as Lab 1: word unigrams, word bigrams, and
char 2/3/4-grams (word-boundary padded).

TF modes
--------
  "raw"      : tf(t,d) = count(t,d)                         [Lab 1 baseline]
  "len"      : tf(t,d) = count(t,d) / total_words(d)         [part 1a]
  "maxfreq"  : tf(t,d) = count(t,d) / max_word_count(d)      [part 1b]

  total_words(d) and max_word_count(d) are computed from the sentence's
  WORD tokens only (per the assignment wording: "total number of words in
  the sentence" / "frequency of the most frequent word in the sentence"),
  then applied as a single per-document denominator to every feature in
  that document's vector (word unigrams, word bigrams, and char n-grams
  alike).

IDF modes
---------
  "unnorm"   : idf(t) = log((1 + N) / (1 + df(t))) + 1        [Lab 1 baseline]
  "maxdf"    : idf(t) = log(max_df / df(t))                   [part 2]

  max_df = the document frequency of the single most common feature in
  the (pruned) vocabulary. This mirrors the "maxfreq" TF idea but at the
  corpus level: instead of dividing by the total document count N, we
  divide by the highest df any feature reaches.

No post-hoc L2 row-normalization is applied. "len" and "maxfreq" TF each
divide every feature in a document's row by a single per-document
constant (total_words(d) or max_word_count(d)); L2-normalizing a row
afterward exactly cancels any positive per-row constant (v/c normalized
== v normalized, for any c > 0), so an earlier version of this file that
L2-normalized every row made all three TF modes produce bit-identical
vectors and therefore identical model scores. Leaving tf * idf as-is is
what lets the three TF modes actually differ.
"""
import math
import re
from collections import Counter

from sparse import CSRMatrix

_TOKEN_RE = re.compile(r'\S+')

TF_MODES = ("raw", "len", "maxfreq")
IDF_MODES = ("unnorm", "maxdf")


def _words(text):
    return _TOKEN_RE.findall(text)


def _char_ngrams(word, n):
    padded = f" {word} "
    L = len(padded)
    if L < n:
        return []
    return [padded[i:i + n] for i in range(L - n + 1)]


def extract_features(text):
    """Return (feature_counts, total_words, max_word_count) for one sentence."""
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

    word_counts = Counter(words)
    total_words = len(words) if words else 1
    max_word_count = max(word_counts.values()) if word_counts else 1
    return feats, total_words, max_word_count


class TfidfVectorizer:
    def __init__(self, tf_mode="raw", idf_mode="unnorm", min_df=2, max_features=150_000):
        if tf_mode not in TF_MODES:
            raise ValueError(f"tf_mode must be one of {TF_MODES}")
        if idf_mode not in IDF_MODES:
            raise ValueError(f"idf_mode must be one of {IDF_MODES}")
        self.tf_mode = tf_mode
        self.idf_mode = idf_mode
        self.min_df = min_df
        self.max_features = max_features

        self.vocab_ = None
        self.idf_ = None
        self.n_features_ = 0

    def _raw_tf(self, count, total_words, max_word_count):
        if self.tf_mode == "raw":
            return float(count)
        if self.tf_mode == "len":
            return count / total_words
        if self.tf_mode == "maxfreq":
            return count / max_word_count
        raise AssertionError

    def fit(self, texts):
        df = Counter()
        doc_info = []  # (feats, total_words, max_word_count)
        for t in texts:
            feats, total_words, max_word_count = extract_features(t)
            doc_info.append((feats, total_words, max_word_count))
            df.update(feats.keys())

        n_docs = len(texts)
        candidates = [(feat, d) for feat, d in df.items() if d >= self.min_df]
        candidates.sort(key=lambda x: x[1], reverse=True)
        if self.max_features is not None and len(candidates) > self.max_features:
            candidates = candidates[:self.max_features]

        self.vocab_ = {feat: i for i, (feat, _) in enumerate(candidates)}
        self.n_features_ = len(self.vocab_)

        idf = [0.0] * self.n_features_
        if self.idf_mode == "unnorm":
            for feat, d in candidates:
                j = self.vocab_[feat]
                idf[j] = math.log((1 + n_docs) / (1 + d)) + 1.0
        elif self.idf_mode == "maxdf":
            max_df = max(d for _, d in candidates) if candidates else 1
            for feat, d in candidates:
                j = self.vocab_[feat]
                idf[j] = math.log(max_df / d)
        self.idf_ = idf

        return self._transform_docs(doc_info)

    def transform(self, texts):
        doc_info = [extract_features(t) for t in texts]
        return self._transform_docs(doc_info)

    def _transform_docs(self, doc_info):
        if self.vocab_ is None:
            raise RuntimeError("Vectorizer not fitted yet")
        rows = []
        for feats, total_words, max_word_count in doc_info:
            row = {}
            for feat, count in feats.items():
                j = self.vocab_.get(feat)
                if j is None:
                    continue
                tf = self._raw_tf(count, total_words, max_word_count)
                row[j] = tf * self.idf_[j]
            rows.append(row)
        return CSRMatrix.from_rows(rows, self.n_features_)
