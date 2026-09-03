"""Train, compare, evaluate, and serialize customer churn models."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from joblib import dump

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.preprocessing import TARGET, load_dataset, clean_dataset, make_pipeline


def evaluate(name, pipeline, X_test, y_test):
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    scores = {
        "model": name,
        "accuracy": round(accuracy_score(y_test, predictions), 4),
        "precision": round(precision_score(y_test, predictions, pos_label="Yes"), 4),
        "recall": round(recall_score(y_test, predictions, pos_label="Yes"), 4),
        "f1": round(f1_score(y_test, predictions, pos_label="Yes"), 4),
        "roc_auc": round(roc_auc_score((y_test == "Yes").astype(int), probabilities), 4),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=["No", "Yes"]).tolist(),
    }
    return scores


def main():
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "model").mkdir(exist_ok=True)
    raw = load_dataset(ROOT / "data")
    data, audit = clean_dataset(raw)
    X = data.drop(columns=[TARGET])
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    candidates = {
        "logistic_regression": GridSearchCV(
            make_pipeline(LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42)),
            {"model__C": [0.1, 1.0, 3.0]}, scoring="roc_auc", cv=3, n_jobs=-1,
        ),
        "random_forest": GridSearchCV(
            make_pipeline(RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)),
            {"model__n_estimators": [200], "model__max_depth": [None, 12], "model__min_samples_leaf": [1, 3]},
            scoring="roc_auc", cv=3, n_jobs=-1,
        ),
    }
    results = []
    fitted = {}
    for name, search in candidates.items():
        search.fit(X_train, y_train)
        fitted[name] = search.best_estimator_
        results.append({**evaluate(name, search.best_estimator_, X_test, y_test), "best_params": search.best_params_})

    final_result = max(results, key=lambda row: (row["roc_auc"], row["f1"]))
    final_pipeline = fitted[final_result["model"]]
    dump(final_pipeline, ROOT / "model" / "churn_pipeline.joblib")

    pd.DataFrame(results).drop(columns=["confusion_matrix"]).to_csv(ROOT / "reports" / "model_comparison.csv", index=False)
    (ROOT / "reports" / "metrics.json").write_text(json.dumps({
        "dataset": {"source": "IBM Telco Customer Churn dataset", **audit, "target_distribution": y.value_counts().to_dict()},
        "test_size": len(X_test), "random_state": 42, "models": results,
        "selected_model": final_result["model"], "selection_rule": "highest ROC-AUC, then F1-score",
    }, indent=2))
    for result in results:
        matrix = result["confusion_matrix"]
        plt.figure(figsize=(4, 3))
        sns.heatmap(matrix, annot=True, fmt="d", cmap="YlOrBr", cbar=False, xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
        plt.xlabel("Predicted churn"); plt.ylabel("Actual churn"); plt.title(result["model"])
        plt.tight_layout(); plt.savefig(ROOT / "reports" / f"{result['model']}_confusion_matrix.png", dpi=140); plt.close()
    print(json.dumps({"rows": len(data), "results": results, "selected_model": final_result["model"], "artifact": str(ROOT / "model" / "churn_pipeline.joblib")}, indent=2))


if __name__ == "__main__":
    main()