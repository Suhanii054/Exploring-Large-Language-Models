"""Build a language-identification dataset from ai4bharat/IndicCorpV2.

For each of the 23 language splits defined in the dataset's config, stream
documents (paragraphs), split them into sentences, and keep the first
SENTENCES_PER_LANG unique, well-formed sentences. Then do a stratified
80/10/10 train/val/test split (same count per language in each split).

Output: data/train.csv, data/val.csv, data/test.csv  (columns: text,label)
"""
import csv
import os
import random
import sys
import time

from datasets import load_dataset

sys.path.insert(0, os.path.dirname(__file__))
from sentence_split import split_paragraph

REPO_ID = "ai4bharat/IndicCorpV2"
CONFIG = "indiccorp_v2"

# The 23 language splits exactly as defined in the dataset repo's README config
# (label = split name, already encodes language + script where relevant).
LANGUAGES = [
    "asm_Beng", "ben_Beng", "brx_Deva", "doi_Deva", "gom_Deva", "guj_Gujr",
    "hin_Deva", "kan_Knda", "kas_Arab", "mai_Deva", "mal_Mlym", "mar_Deva",
    "mni_Mtei", "npi_Deva", "ory_Orya", "pan_Guru", "san_Deva", "snd_Deva",
    "tam_Taml", "tel_Telu", "urd_Arab", "khasi", "santhali",
]

SENTENCES_PER_LANG = 1000
TRAIN_N, VAL_N, TEST_N = 800, 100, 100
SEED = 42

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def collect_sentences(lang: str, target: int, max_docs: int = 200_000):
    out_path = os.path.join(RAW_DIR, f"{lang}.txt")
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            existing = [l.rstrip("\n") for l in f if l.strip()]
        if len(existing) >= target:
            return existing[:target]

    seen = set()
    sentences = []
    attempts = 0
    while attempts < 3:
        try:
            ds = load_dataset(REPO_ID, CONFIG, split=lang, streaming=True)
            break
        except Exception as e:
            attempts += 1
            print(f"  [{lang}] load error ({e}); retry {attempts}/3")
            time.sleep(5)
    else:
        raise RuntimeError(f"could not open stream for {lang}")

    n_docs = 0
    for ex in ds:
        n_docs += 1
        if n_docs > max_docs:
            break
        text = ex.get("text") or ""
        for s in split_paragraph(text):
            if s in seen:
                continue
            seen.add(s)
            sentences.append(s)
            if len(sentences) >= target:
                break
        if len(sentences) >= target:
            break

    os.makedirs(RAW_DIR, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for s in sentences:
            f.write(s + "\n")

    print(f"  [{lang}] collected {len(sentences)} sentences from {n_docs} docs")
    return sentences


def main():
    rng = random.Random(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    train_rows, val_rows, test_rows = [], [], []

    for lang in LANGUAGES:
        print(f"Processing {lang} ...")
        sentences = collect_sentences(lang, SENTENCES_PER_LANG)
        if len(sentences) < SENTENCES_PER_LANG:
            print(f"  WARNING: only got {len(sentences)} sentences for {lang} "
                  f"(need {SENTENCES_PER_LANG})")
        sentences = sentences[:SENTENCES_PER_LANG]
        rng.shuffle(sentences)

        train_rows += [(s, lang) for s in sentences[:TRAIN_N]]
        val_rows += [(s, lang) for s in sentences[TRAIN_N:TRAIN_N + VAL_N]]
        test_rows += [(s, lang) for s in sentences[TRAIN_N + VAL_N:TRAIN_N + VAL_N + TEST_N]]

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    rng.shuffle(test_rows)

    for name, rows in [("train", train_rows), ("val", val_rows), ("test", test_rows)]:
        path = os.path.join(OUT_DIR, f"{name}.csv")
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["text", "label"])
            w.writerows(rows)
        print(f"Wrote {path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
