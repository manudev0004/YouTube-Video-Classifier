"""Train or run a title-based classifier with category-derived proxy labels."""

import argparse
import hashlib
import json
from pathlib import Path
import sys


def load_dataset(path):
    import pandas as pd

    frame = pd.read_csv(path)
    required = {"Title", "Category_ID"}
    if not required.issubset(frame.columns):
        raise ValueError("CSV must contain Title and Category_ID columns.")
    if frame["Title"].isna().any():
        raise ValueError("Every row needs a non-empty Title.")
    frame["Title"] = frame["Title"].astype(str).str.strip()
    if frame["Title"].eq("").any():
        raise ValueError("Every row needs a non-empty Title.")
    category = pd.to_numeric(frame["Category_ID"], errors="raise")
    if category.isna().any() or (category % 1 != 0).any():
        raise ValueError("Category_ID must contain integer YouTube category IDs.")
    frame["label"] = category.astype(int).isin([27, 28]).astype(int)
    frame["title_key"] = frame["Title"].str.casefold().str.split().str.join(" ")
    if frame.groupby("title_key")["label"].nunique().gt(1).any():
        raise ValueError("A duplicate title has conflicting proxy labels; resolve it first.")
    before = len(frame)
    frame = frame.drop_duplicates("title_key").reset_index(drop=True)
    counts = frame["label"].value_counts()
    if len(counts) != 2 or counts.min() < 2:
        raise ValueError("Need both proxy classes, with at least two unique titles each.")
    return frame, before - len(frame)


def make_pipeline(name, seed):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    models = {
        "random-forest": lambda: RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1),
        "naive-bayes": MultinomialNB,
        "svm": SVC,
        "knn": KNeighborsClassifier,
        "decision-tree": lambda: DecisionTreeClassifier(random_state=seed),
    }
    if name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ValueError("For XGBoost, install requirements-optional.txt first.") from exc
        model = XGBClassifier(random_state=seed, eval_metric="logloss", n_jobs=1)
    else:
        model = models[name]()
    return Pipeline([("tfidf", TfidfVectorizer()), ("classifier", model)])


def train(args):
    import joblib
    import sklearn
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.model_selection import train_test_split

    if not 0 < args.test_size < 1:
        raise ValueError("test-size must be between 0 and 1.")
    frame, duplicate_count = load_dataset(args.data)
    # Split raw titles first: the vectorizer must never learn the test vocabulary.
    x_train, x_test, y_train, y_test = train_test_split(
        frame["Title"], frame["label"], test_size=args.test_size,
        random_state=args.seed, stratify=frame["label"],
    )
    pipeline = make_pipeline(args.model, args.seed)
    pipeline.fit(x_train, y_train)
    predicted = pipeline.predict(x_test)
    baseline = DummyClassifier(strategy="most_frequent").fit(x_train.to_frame(), y_train)
    baseline_predictions = baseline.predict(x_test.to_frame())
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predicted, average="binary", zero_division=0,
    )
    synthetic = (
        "Data_Source" in frame and
        frame["Data_Source"].fillna("").astype(str).str.lower().eq("synthetic").all()
    )
    metrics = {
        "task": "binary category proxy from video titles",
        "positive_categories": [27, 28],
        "model": args.model,
        "seed": args.seed,
        "test_fraction": args.test_size,
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "train_label_counts": {str(k): int(v) for k, v in y_train.value_counts().items()},
        "test_label_counts": {str(k): int(v) for k, v in y_test.value_counts().items()},
        "duplicate_titles_removed": duplicate_count,
        "accuracy": float(accuracy_score(y_test, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predicted)),
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "confusion_matrix_labels": [0, 1],
        "confusion_matrix": confusion_matrix(y_test, predicted, labels=[0, 1]).tolist(),
        "majority_baseline_accuracy": float(accuracy_score(y_test, baseline_predictions)),
        "synthetic_demo": bool(synthetic),
        "dataset_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "scikit_learn_version": sklearn.__version__,
        "limitations": [
            "Labels come from YouTube categories, not human judgments of educational quality.",
            "Exact normalized title duplicates are removed; related videos/channels may still cross the split.",
            "This is one exploratory split, not independent production validation.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, args.output_dir / "pipeline.joblib")
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    if synthetic:
        print("Synthetic demonstration only: these scores are not project benchmark results.", file=sys.stderr)


def predict(args):
    import joblib

    # Only load a model artifact you created or trust; joblib deserializes Python objects.
    pipeline = joblib.load(args.artifact)
    if any(not title.strip() for title in args.title):
        raise ValueError("Prediction titles must be non-empty.")
    labels = pipeline.predict(args.title)
    rows = [
        {"title": title, "label": int(label), "category_proxy": "education-or-science" if label else "other"}
        for title, label in zip(args.title, labels)
    ]
    print(json.dumps(rows, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="Validate CSV, evaluate a split and save a pipeline")
    training.add_argument("--data", type=Path, required=True)
    training.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--test-size", type=float, default=0.2)
    training.add_argument("--model", choices=["random-forest", "naive-bayes", "svm", "knn", "decision-tree", "xgboost"], default="random-forest")
    training.set_defaults(run=train)
    prediction = commands.add_parser("predict", help="Predict one or more titles using a saved pipeline")
    prediction.add_argument("--artifact", type=Path, required=True)
    prediction.add_argument("--title", action="append", required=True)
    prediction.set_defaults(run=predict)
    args = parser.parse_args(argv)
    try:
        args.run(args)
    except (ValueError, OSError, ImportError, KeyError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
