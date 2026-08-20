"""[Lab 2] Train + evaluate logistic regression on all 6 TF/IDF combinations,
reusing the exact dataset built in Lab 1 (no re-splitting, no re-sampling).
"""
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from tfidf_variants import TfidfVectorizer, TF_MODES, IDF_MODES
from logreg import LogisticRegression
from metrics import macro_f1, accuracy

LAB1_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "LAB1", "LangID", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load_csv(name):
    path = os.path.join(LAB1_DATA_DIR, f"{name}.csv")
    texts, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            texts.append(row[0])
            labels.append(row[1])
    return texts, labels


def run_one(tf_mode, idf_mode, train_texts, y_train, val_texts, y_val,
            test_texts, y_test, n_classes):
    print(f"\n===== TF={tf_mode}  IDF={idf_mode} =====")
    t0 = time.time()
    vec = TfidfVectorizer(tf_mode=tf_mode, idf_mode=idf_mode, min_df=2, max_features=150_000)
    X_train = vec.fit(train_texts)
    X_val = vec.transform(val_texts)
    X_test = vec.transform(test_texts)
    fit_time = time.time() - t0
    print(f"  vocab={vec.n_features_}  vectorize_time={fit_time:.1f}s")

    clf = LogisticRegression(
        n_features=vec.n_features_, n_classes=n_classes,
        l2=1e-4, lr=0.05, batch_size=256, epochs=30, verbose=False,
    )
    t0 = time.time()
    clf.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    train_time = time.time() - t0

    val_pred = clf.predict(X_val)
    test_pred = clf.predict(X_test)

    result = {
        "tf_mode": tf_mode,
        "idf_mode": idf_mode,
        "vocab_size": vec.n_features_,
        "val_accuracy": accuracy(y_val, val_pred),
        "val_macro_f1": macro_f1(y_val, val_pred, n_classes),
        "test_accuracy": accuracy(y_test, test_pred),
        "test_macro_f1": macro_f1(y_test, test_pred, n_classes),
        "vectorize_time_s": fit_time,
        "train_time_s": train_time,
    }
    print(f"  val_macro_f1={result['val_macro_f1']:.4f}  test_macro_f1={result['test_macro_f1']:.4f}")
    return result


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading Lab 1 dataset...")
    train_texts, train_labels = load_csv("train")
    val_texts, val_labels = load_csv("val")
    test_texts, test_labels = load_csv("test")

    classes = sorted(set(train_labels))
    label2id = {c: i for i, c in enumerate(classes)}
    y_train = np.array([label2id[l] for l in train_labels])
    y_val = np.array([label2id[l] for l in val_labels])
    y_test = np.array([label2id[l] for l in test_labels])
    n_classes = len(classes)
    print(f"train={len(train_texts)} val={len(val_texts)} test={len(test_texts)} classes={n_classes}")

    results = []
    for tf_mode in TF_MODES:
        for idf_mode in IDF_MODES:
            r = run_one(tf_mode, idf_mode, train_texts, y_train, val_texts, y_val,
                        test_texts, y_test, n_classes)
            results.append(r)

    out_path = os.path.join(RESULTS_DIR, "comparison.csv")
    fieldnames = list(results[0].keys())
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)

    print("\n\n================ SUMMARY (sorted by test macro-F1) ================")
    header = f"{'TF':<10}{'IDF':<10}{'vocab':>8}{'val_acc':>10}{'val_F1':>10}{'test_acc':>10}{'test_F1':>10}"
    print(header)
    for r in sorted(results, key=lambda x: -x["test_macro_f1"]):
        print(f"{r['tf_mode']:<10}{r['idf_mode']:<10}{r['vocab_size']:>8}"
              f"{r['val_accuracy']:>10.4f}{r['val_macro_f1']:>10.4f}"
              f"{r['test_accuracy']:>10.4f}{r['test_macro_f1']:>10.4f}")
    print(f"\nSaved detailed results to {out_path}")


if __name__ == "__main__":
    main()
