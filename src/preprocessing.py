"""
preprocessing.py
Data loading, cleaning, EDA, and feature engineering for Titanic ML project.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
import os
warnings.filterwarnings("ignore")

# ─── Consistent visual style ─────────────────────────────────────────────────
PALETTE = {"Survived": "#2ecc71", "Not Survived": "#e74c3c"}
plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#0f1117",
    "axes.edgecolor":   "#444",
    "axes.labelcolor":  "#ccc",
    "xtick.color":      "#ccc",
    "ytick.color":      "#ccc",
    "text.color":       "#ccc",
    "grid.color":       "#333",
    "font.family":      "DejaVu Sans",
})


# ─── 1. Load data ─────────────────────────────────────────────────────────────
def load_data(train_path: str, test_path: str):
    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)
    return train, test


# ─── 2. Exploratory Data Analysis ────────────────────────────────────────────
def run_eda(df: pd.DataFrame, output_dir: str = "outputs"):
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "="*60)
    print("  EXPLORATORY DATA ANALYSIS")
    print("="*60)
    print(f"\nShape     : {df.shape}")
    print(f"Columns   : {list(df.columns)}")
    print("\n--- Missing values ---")
    missing = df.isnull().sum()
    print(missing[missing > 0])
    print("\n--- Basic statistics ---")
    print(df.describe())

    # ── Figure 1: Survival count ─────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.patch.set_facecolor("#0f1117")
    fig.suptitle("Titanic – Exploratory Data Analysis", fontsize=16,
                 color="white", fontweight="bold", y=1.01)

    labels = ["Not Survived", "Survived"]
    colors = ["#e74c3c", "#2ecc71"]

    # Survival count
    ax = axes[0, 0]
    counts = df["Survived"].value_counts()
    bars = ax.bar([labels[i] for i in counts.index], counts.values,
                  color=[colors[i] for i in counts.index], edgecolor="white",
                  linewidth=0.6, width=0.5)
    ax.set_title("Survival Count", color="white", fontweight="bold")
    ax.set_ylabel("Count", color="#ccc")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(val), ha="center", va="bottom", color="white", fontsize=11)

    # Survival by Sex
    ax = axes[0, 1]
    sex_surv = df.groupby(["Sex", "Survived"]).size().unstack()
    sex_surv.columns = labels
    sex_surv.plot(kind="bar", ax=ax, color=colors, edgecolor="white",
                  linewidth=0.6, width=0.6)
    ax.set_title("Survival by Sex", color="white", fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Count", color="#ccc")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(facecolor="#222", edgecolor="#555", labelcolor="white",
              fontsize=9)

    # Survival by Pclass
    ax = axes[0, 2]
    pc_surv = df.groupby(["Pclass", "Survived"]).size().unstack()
    pc_surv.columns = labels
    pc_surv.plot(kind="bar", ax=ax, color=colors, edgecolor="white",
                 linewidth=0.6, width=0.6)
    ax.set_title("Survival by Passenger Class", color="white", fontweight="bold")
    ax.set_xlabel("Pclass", color="#ccc")
    ax.set_ylabel("Count", color="#ccc")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(facecolor="#222", edgecolor="#555", labelcolor="white",
              fontsize=9)

    # Age distribution
    ax = axes[1, 0]
    for surv, label, color in zip([0, 1], labels, colors):
        ax.hist(df[df["Survived"] == surv]["Age"].dropna(),
                bins=25, alpha=0.7, color=color, label=label,
                edgecolor="white", linewidth=0.4)
    ax.set_title("Age Distribution by Survival", color="white",
                 fontweight="bold")
    ax.set_xlabel("Age", color="#ccc")
    ax.set_ylabel("Count", color="#ccc")
    ax.legend(facecolor="#222", edgecolor="#555", labelcolor="white",
              fontsize=9)

    # Fare distribution
    ax = axes[1, 1]
    for surv, label, color in zip([0, 1], labels, colors):
        ax.hist(df[df["Survived"] == surv]["Fare"].dropna(),
                bins=30, alpha=0.7, color=color, label=label,
                edgecolor="white", linewidth=0.4)
    ax.set_title("Fare Distribution by Survival", color="white",
                 fontweight="bold")
    ax.set_xlabel("Fare", color="#ccc")
    ax.set_ylabel("Count", color="#ccc")
    ax.legend(facecolor="#222", edgecolor="#555", labelcolor="white",
              fontsize=9)

    # Embarked survival
    ax = axes[1, 2]
    emb_surv = df.groupby(["Embarked", "Survived"]).size().unstack(fill_value=0)
    emb_surv.columns = labels
    emb_surv.plot(kind="bar", ax=ax, color=colors, edgecolor="white",
                  linewidth=0.6, width=0.6)
    ax.set_title("Survival by Embarkation Port", color="white",
                 fontweight="bold")
    ax.set_xlabel("Embarked", color="#ccc")
    ax.set_ylabel("Count", color="#ccc")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(facecolor="#222", edgecolor="#555", labelcolor="white",
              fontsize=9)

    plt.tight_layout()
    path1 = os.path.join(output_dir, "eda_overview.png")
    plt.savefig(path1, dpi=150, bbox_inches="tight",
                facecolor="#0f1117")
    plt.close()
    print(f"\n[EDA] Saved: {path1}")

    # ── Figure 2: Correlation heat map ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")
    num_cols = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]
    corr = df[num_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
                ax=ax, linewidths=0.5, linecolor="#333",
                annot_kws={"size": 11, "color": "white"},
                cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Heatmap", color="white",
                 fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    path2 = os.path.join(output_dir, "correlation_heatmap.png")
    plt.savefig(path2, dpi=150, bbox_inches="tight",
                facecolor="#0f1117")
    plt.close()
    print(f"[EDA] Saved: {path2}")

    return path1, path2


# ─── 3. Feature Engineering ───────────────────────────────────────────────────
def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Title from Name
    df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    rare_titles = df["Title"].value_counts()[df["Title"].value_counts() < 10].index
    df["Title"] = df["Title"].replace(rare_titles, "Rare")
    title_map = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    df["Title"] = df["Title"].map(title_map).fillna(5).astype(int)

    # Family features
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"]    = (df["FamilySize"] == 1).astype(int)

    # Fare per person
    df["FarePerPerson"] = df["Fare"] / df["FamilySize"]

    # Cabin known
    df["HasCabin"] = df["Cabin"].notna().astype(int)

    # Age × Class interaction
    df["Age*Class"] = df["Age"].fillna(df["Age"].median()) * df["Pclass"]

    return df


# ─── 4. Preprocessing ────────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame, is_train: bool = True):
    df = feature_engineer(df)

    # Fill missing Age with median per Title
    df["Age"] = df.groupby("Title")["Age"].transform(
        lambda x: x.fillna(x.median())
    )
    df["Age"].fillna(df["Age"].median(), inplace=True)

    # Fill Fare missing
    df["Fare"].fillna(df["Fare"].median(), inplace=True)
    df["FarePerPerson"].fillna(df["FarePerPerson"].median(), inplace=True)

    # Fill Embarked with mode
    df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

    # Encode Sex
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

    # Encode Embarked
    df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

    FEATURES = [
        "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare",
        "Embarked", "Title", "FamilySize", "IsAlone",
        "FarePerPerson", "HasCabin", "Age*Class",
    ]

    X = df[FEATURES].copy()
    # Final safety: fill any remaining NaN with column median
    X = X.fillna(X.median())
    if is_train and "Survived" in df.columns:
        y = df["Survived"]
        return X, y, FEATURES
    return X, None, FEATURES


if __name__ == "__main__":
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train, test = load_data(
        os.path.join(BASE, "data", "train.csv"),
        os.path.join(BASE, "data", "test.csv"),
    )
    run_eda(train, output_dir=os.path.join(BASE, "outputs"))
    X, y, feats = preprocess(train, is_train=True)
    print("\nFeature matrix shape:", X.shape)
    print("Features:", feats)