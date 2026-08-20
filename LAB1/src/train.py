import csv
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from tfidf import TfidfVectorizer
from logreg import LogisticRegression
from metrics import macro_f1, accuracy, classification_report

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
ART_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")


def load_csv(name):
    path = os.path.join(DATA_DIR, f"{name}.csv")
    texts, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            texts.append(row[0])
            labels.append(row[1])
    return texts, labels


def main():
    os.makedirs(ART_DIR, exist_ok=True)

    print("Loading data...")
    train_texts, train_labels = load_csv("train")
    val_texts, val_labels = load_csv("val")
    test_texts, test_labels = load_csv("test")

    classes = sorted(set(train_labels))
    label2id = {c: i for i, c in enumerate(classes)}
    y_train = np.array([label2id[l] for l in train_labels])
    y_val = np.array([label2id[l] for l in val_labels])
    y_test = np.array([label2id[l] for l in test_labels])

    print(f"train={len(train_texts)} val={len(val_texts)} test={len(test_texts)} classes={len(classes)}")

    print("Fitting TF-IDF vectorizer...")
    t0 = time.time()
    vec = TfidfVectorizer(min_df=2, max_features=150_000)
    X_train = vec.fit(train_texts)
    print(f"  vocab size = {vec.n_features_}  ({time.time()-t0:.1f}s)")

    X_val = vec.transform(val_texts)
    X_test = vec.transform(test_texts)

    print("Training logistic regression...")
    clf = LogisticRegression(
        n_features=vec.n_features_, n_classes=len(classes),
        l2=1e-4, lr=0.05, batch_size=256, epochs=30, verbose=True,
    )
    clf.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    print("\n=== Validation ===")
    val_pred = clf.predict(X_val)
    print(f"accuracy={accuracy(y_val, val_pred):.4f}  macro_f1={macro_f1(y_val, val_pred, len(classes)):.4f}")

    print("\n=== Test ===")
    test_pred = clf.predict(X_test)
    print(f"accuracy={accuracy(y_test, test_pred):.4f}  macro_f1={macro_f1(y_test, test_pred, len(classes)):.4f}")
    print()
    print(classification_report(y_test, test_pred, classes))

    with open(os.path.join(ART_DIR, "vectorizer.pkl"), "wb") as f:
        pickle.dump(vec, f)
    with open(os.path.join(ART_DIR, "model.pkl"), "wb") as f:
        pickle.dump(clf, f)
    with open(os.path.join(ART_DIR, "classes.pkl"), "wb") as f:
        pickle.dump(classes, f)
    print("\nSaved artifacts to", ART_DIR)


if __name__ == "__main__":
    main()
