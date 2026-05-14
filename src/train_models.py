"""
train_models.py
Train, evaluate, and compare multiple supervised learning models.
Saves the best model via Joblib.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)

# ── Classifiers ──────────────────────────────────────────────────────────────
from sklearn.linear_model   import LogisticRegression
from sklearn.tree           import DecisionTreeClassifier
from sklearn.ensemble       import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors      import KNeighborsClassifier
from sklearn.svm            import SVC
from sklearn.naive_bayes    import GaussianNB

# ── Local import ─────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.preprocessing import load_data, preprocess, run_eda

# ── Plot style ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#0f1117",
    "axes.edgecolor":   "#444",
    "axes.labelcolor":  "#ccc",
    "xtick.color":      "#ccc",
    "ytick.color":      "#ccc",
    "text.color":       "#ccc",
    "grid.color":       "#333",
})

ACCENT    = "#7c3aed"
POSITIVE  = "#2ecc71"
NEGATIVE  = "#e74c3c"
HIGHLIGHT = "#f39c12"


# ─── 1. Define models ─────────────────────────────────────────────────────────
def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=500, C=1.0,
                                                   random_state=42),
        "Decision Tree":       DecisionTreeClassifier(max_depth=5,
                                                      random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=200,
                                                      max_depth=7,
                                                      random_state=42),
        "KNN":                 KNeighborsClassifier(n_neighbors=7),
        "SVM":                 SVC(probability=True, kernel="rbf",
                                   C=1.0, random_state=42),
        "Naive Bayes":         GaussianNB(),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200,
                                                          learning_rate=0.05,
                                                          max_depth=4,
                                                          random_state=42),
    }


# ─── 2. Train + evaluate ──────────────────────────────────────────────────────
def train_and_evaluate(X_train, X_test, y_train, y_test,
                       output_dir: str = "outputs"):
    os.makedirs(output_dir, exist_ok=True)
    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    models  = get_models()
    cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    print("\n" + "="*60)
    print("  MODEL TRAINING & EVALUATION")
    print("="*60)

    for name, clf in models.items():
        clf.fit(X_tr_sc, y_train)
        y_pred    = clf.predict(X_te_sc)
        y_proba   = clf.predict_proba(X_te_sc)[:, 1]
        acc       = accuracy_score(y_test, y_pred)
        roc_auc   = roc_auc_score(y_test, y_proba)
        cv_scores = cross_val_score(clf, X_tr_sc, y_train, cv=cv,
                                    scoring="accuracy")

        results[name] = {
            "model":     clf,
            "accuracy":  round(acc, 4),
            "roc_auc":   round(roc_auc, 4),
            "cv_mean":   round(cv_scores.mean(), 4),
            "cv_std":    round(cv_scores.std(), 4),
            "y_pred":    y_pred,
            "y_proba":   y_proba,
        }
        print(f"  {name:<25} Acc={acc:.4f}  ROC-AUC={roc_auc:.4f}"
              f"  CV={cv_scores.mean():.4f}±{cv_scores.std():.4f}")

    # ── Best model ────────────────────────────────────────────────────────────
    best_name = max(results, key=lambda k: results[k]["cv_mean"])
    print(f"\n  ★ Best model: {best_name}  "
          f"(CV={results[best_name]['cv_mean']:.4f})")

    # ── Save visuals ──────────────────────────────────────────────────────────
    _plot_comparison(results, output_dir)
    _plot_confusion_matrices(results, y_test, output_dir)
    _plot_roc_curves(results, y_test, output_dir)
    _plot_feature_importance(results, X_train.columns.tolist(), output_dir)

    return results, best_name, scaler


# ─── 3. Comparison bar chart ──────────────────────────────────────────────────
def _plot_comparison(results: dict, output_dir: str):
    names    = list(results.keys())
    accs     = [results[n]["accuracy"] for n in names]
    aucs     = [results[n]["roc_auc"]  for n in names]
    cv_means = [results[n]["cv_mean"]  for n in names]
    cv_stds  = [results[n]["cv_std"]   for n in names]

    x     = np.arange(len(names))
    width = 0.27

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    b1 = ax.bar(x - width, accs,     width, label="Test Accuracy",
                color=ACCENT,    edgecolor="white", linewidth=0.5)
    b2 = ax.bar(x,          aucs,     width, label="ROC-AUC",
                color=POSITIVE,  edgecolor="white", linewidth=0.5)
    b3 = ax.bar(x + width,  cv_means, width, label="CV Accuracy",
                color=HIGHLIGHT, edgecolor="white", linewidth=0.5,
                yerr=cv_stds, capsize=4, error_kw={"ecolor": "white",
                                                    "linewidth": 1.2})

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom",
                    fontsize=7.5, color="white")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=10)
    ax.set_ylim(0.5, 1.05)
    ax.set_ylabel("Score", color="#ccc")
    ax.set_title("Model Comparison — Accuracy, ROC-AUC & Cross-Validation",
                 color="white", fontsize=13, fontweight="bold", pad=12)
    ax.legend(facecolor="#222", edgecolor="#555", labelcolor="white",
              fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print(f"[Chart] Saved: {path}")


# ─── 4. Confusion matrices ────────────────────────────────────────────────────
def _plot_confusion_matrices(results: dict, y_test, output_dir: str):
    n  = len(results)
    nc = 4
    nr = (n + nc - 1) // nc
    fig, axes = plt.subplots(nr, nc, figsize=(nc*4, nr*3.8))
    axes_flat = axes.flatten()
    fig.patch.set_facecolor("#0f1117")
    fig.suptitle("Confusion Matrices", color="white", fontsize=15,
                 fontweight="bold", y=1.01)

    for i, (name, res) in enumerate(results.items()):
        ax = axes_flat[i]
        ax.set_facecolor("#0f1117")
        cm = confusion_matrix(y_test, res["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                    ax=ax, linewidths=1, linecolor="#333",
                    annot_kws={"size": 13, "weight": "bold"},
                    xticklabels=["Not Surv.", "Survived"],
                    yticklabels=["Not Surv.", "Survived"])
        ax.set_title(name, color="white", fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted", color="#aaa", fontsize=8)
        ax.set_ylabel("Actual",    color="#aaa", fontsize=8)

    # Hide unused axes
    for j in range(i+1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print(f"[Chart] Saved: {path}")


# ─── 5. ROC curves ────────────────────────────────────────────────────────────
def _plot_roc_curves(results: dict, y_test, output_dir: str):
    COLORS = ["#7c3aed", "#2ecc71", "#e74c3c", "#f39c12",
              "#3498db", "#e67e22", "#1abc9c"]
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    for (name, res), color in zip(results.items(), COLORS):
        fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
        ax.plot(fpr, tpr, color=color, linewidth=2.2,
                label=f"{name} (AUC={res['roc_auc']:.3f})")

    ax.plot([0, 1], [0, 1], "w--", linewidth=1, alpha=0.4,
            label="Random Classifier")
    ax.set_xlabel("False Positive Rate", color="#ccc")
    ax.set_ylabel("True Positive Rate",  color="#ccc")
    ax.set_title("ROC Curves – All Models",
                 color="white", fontsize=13, fontweight="bold", pad=10)
    ax.legend(loc="lower right", facecolor="#1a1a2e",
              edgecolor="#555", labelcolor="white", fontsize=9)
    ax.grid(linestyle="--", alpha=0.2)

    plt.tight_layout()
    path = os.path.join(output_dir, "roc_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print(f"[Chart] Saved: {path}")


# ─── 6. Feature importance ────────────────────────────────────────────────────
def _plot_feature_importance(results: dict, feature_names: list,
                              output_dir: str):
    # Use Random Forest importance
    rf = results.get("Random Forest", {}).get("model")
    if rf is None or not hasattr(rf, "feature_importances_"):
        return

    imps = pd.Series(rf.feature_importances_, index=feature_names).sort_values()
    colors_bar = [ACCENT if v > imps.median() else "#444" for v in imps]

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    bars = ax.barh(imps.index, imps.values, color=colors_bar,
                   edgecolor="white", linewidth=0.4)
    for bar, val in zip(bars, imps.values):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=9, color="white")

    ax.set_xlabel("Importance", color="#ccc")
    ax.set_title("Feature Importances (Random Forest)",
                 color="white", fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print(f"[Chart] Saved: {path}")


# ─── 7. Save best model ───────────────────────────────────────────────────────
def save_model(results: dict, best_name: str, scaler,
               feature_names: list, model_dir: str = "models"):
    os.makedirs(model_dir, exist_ok=True)
    best_clf = results[best_name]["model"]

    pipeline = Pipeline([
        ("scaler",     scaler),
        ("classifier", best_clf),
    ])

    model_path = os.path.join(model_dir, "best_model.pkl")
    joblib.dump(pipeline, model_path)
    print(f"\n[Model] Saved pipeline → {model_path}")

    # Save metadata
    meta = {
        "best_model":  best_name,
        "accuracy":    results[best_name]["accuracy"],
        "roc_auc":     results[best_name]["roc_auc"],
        "cv_mean":     results[best_name]["cv_mean"],
        "features":    feature_names,
        "all_results": {
            k: {"accuracy": v["accuracy"], "roc_auc": v["roc_auc"],
                "cv_mean": v["cv_mean"]}
            for k, v in results.items()
        },
    }
    meta_path = os.path.join(model_dir, "model_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[Model] Metadata  → {meta_path}")
    return model_path


# ─── 8. Generate predictions on test set ─────────────────────────────────────
def generate_predictions(test_df: pd.DataFrame, model_path: str,
                          feature_names: list, output_dir: str = "outputs"):
    pipeline = joblib.load(model_path)
    X_test, _, _ = preprocess(test_df, is_train=False)
    X_test = X_test[feature_names]
    preds = pipeline.predict(X_test)

    submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived":    preds.astype(int),
    })
    path = os.path.join(output_dir, "submission.csv")
    submission.to_csv(path, index=False)
    print(f"\n[Output] Submission CSV → {path}")
    print(f"         Predicted survivors: "
          f"{preds.sum()} / {len(preds)}")
    return submission


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA = os.path.join(BASE, "data")
    OUT  = os.path.join(BASE, "outputs")
    MDL  = os.path.join(BASE, "models")

    # Load
    train_df, test_df = load_data(
        os.path.join(DATA, "train.csv"),
        os.path.join(DATA, "test.csv"),
    )

    # EDA
    run_eda(train_df, output_dir=OUT)

    # Preprocess
    X, y, features = preprocess(train_df, is_train=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train
    results, best_name, scaler = train_and_evaluate(
        X_train, X_test, y_train, y_test, output_dir=OUT
    )

    # Save
    model_path = save_model(results, best_name, scaler, features, model_dir=MDL)

    # Predict test set
    generate_predictions(test_df, model_path, features, output_dir=OUT)

    print("\n" + "="*60)
    print("  ✅  Pipeline complete!")
    print("="*60)