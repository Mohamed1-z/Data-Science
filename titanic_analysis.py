import importlib.util
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")


def ensure_runtime_dependencies():
    required_packages = {
        "numpy": "numpy",
        "pandas": "pandas",
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
        "sklearn": "scikit-learn",
    }

    missing = [
        package
        for module_name, package in required_packages.items()
        if importlib.util.find_spec(module_name) is None
    ]

    if not missing:
        return

    print(f"Installing missing Python packages: {', '.join(missing)}")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
        importlib.invalidate_caches()
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Automatic dependency installation failed. "
            "Please run: "
            f"{sys.executable} -m pip install {' '.join(missing)}"
        ) from exc


ensure_runtime_dependencies()

import os
import subprocess
import sys
import matplotlib

try:
    matplotlib.use("TkAgg")
except Exception:
    try:
        matplotlib.use("Qt5Agg")
    except Exception:
        matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

BASE_DIR = Path(__file__).resolve().parent
FIG_DIR = BASE_DIR / "figures"
OUT_DIR = BASE_DIR / "output"
UPLOAD_DIR = BASE_DIR  # place train.csv, test.csv, gender_submission.csv here

FIG_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

stats_summary = {}  # collects numbers we will quote in the written report

# =========================================================================
# TASK 1: DATA ACQUISITION
# =========================================================================
print("=" * 75)
print("TASK 1: DATA ACQUISITION")
print("=" * 75)

required_files = ["train.csv", "test.csv", "gender_submission.csv"]
missing = [
    f
    for f in required_files
    if not (UPLOAD_DIR / f).exists()
]
if missing:
    raise FileNotFoundError(
        f"Missing required file(s): {missing}. "
        f"Place train.csv, test.csv and gender_submission.csv in '{UPLOAD_DIR}' "
        f"(the Kaggle Titanic dataset) before running this script."
    )

train = pd.read_csv(UPLOAD_DIR / "train.csv")
test = pd.read_csv(UPLOAD_DIR / "test.csv")
gender_submission = pd.read_csv(UPLOAD_DIR / "gender_submission.csv")

print("\nTraining set dimensions:", train.shape)
print("Test set dimensions:", test.shape)
print("\nColumn names:", list(train.columns))
print("\nFirst five observations:\n", train.head())
print("\nData types:\n", train.dtypes)

stats_summary["train_shape"] = train.shape
stats_summary["test_shape"] = test.shape
stats_summary["columns"] = list(train.columns)

test_passenger_ids = test["PassengerId"].copy()

# =========================================================================
# TASK 2: DATA CLEANING
# =========================================================================
print("\n" + "=" * 75)
print("TASK 2: DATA CLEANING")
print("=" * 75)

# ---- 2.1 DETECT MISSING VALUES ----
print("\n--- 2.1 Detecting missing values ---")

missing_train = train.isnull().sum()
missing_train_pct = (missing_train / len(train) * 100).round(2)
missing_train_report = (
    pd.DataFrame({"missing_count": missing_train, "missing_pct": missing_train_pct})
    .query("missing_count > 0")
    .sort_values("missing_count", ascending=False)
)

missing_test = test.isnull().sum()
missing_test_pct = (missing_test / len(test) * 100).round(2)
missing_test_report = (
    pd.DataFrame({"missing_count": missing_test, "missing_pct": missing_test_pct})
    .query("missing_count > 0")
    .sort_values("missing_count", ascending=False)
)

print("\nMissing values in TRAIN set:\n", missing_train_report)
print("\nMissing values in TEST set:\n", missing_test_report)

stats_summary["missing_train"] = train.isnull().sum().to_dict()
stats_summary["missing_test"] = test.isnull().sum().to_dict()

# ---- 2.2 DETECT DUPLICATE OBSERVATIONS ----
print("\n--- 2.2 Detecting duplicate observations ---")

dupes_train = train.duplicated().sum()
dupes_test = test.duplicated().sum()
print(f"\nDuplicate rows found in TRAIN set: {dupes_train}")
print(f"Duplicate rows found in TEST set: {dupes_test}")

if dupes_train > 0:
    print("\nDuplicate rows in TRAIN (preview):\n", train[train.duplicated()])
if dupes_test > 0:
    print("\nDuplicate rows in TEST (preview):\n", test[test.duplicated()])

stats_summary["duplicates_train"] = int(dupes_train)
stats_summary["duplicates_test"] = int(dupes_test)

# ---- 2.3 HANDLE MISSING VALUES + REMOVE DUPLICATES ----
print("\n--- 2.3 Cleaning the data ---")

# Impute values are computed from the TRAINING set only. This avoids
# "data leakage": if we computed the median/mode from the combined
# train+test data, information from the test set would leak into how
# we treat the training set, which can make evaluation misleadingly
# optimistic.
age_median = train["Age"].median()
fare_median = train["Fare"].median()
embarked_mode = train["Embarked"].mode()[0]


def clean_data(df, age_median, fare_median, embarked_mode, label=""):
    """
    Cleaning decisions (explained):
    - Age (~20% missing in train, ~21% in test): imputed with the MEDIAN
      age computed from the TRAINING set only, to avoid leaking test
      information into the model. Median is used instead of mean because
      Age is right-skewed and less sensitive to outliers.
    - Embarked (2 missing in train): imputed with the MODE ('S'), since
      only two values are missing out of 891 and the port distribution is
      heavily dominated by Southampton.
    - Fare (1 missing in test): imputed with the median fare from training
      data, for the same right-skew reasoning as Age.
    - Cabin (~77% missing in train, ~78% in test): far too sparse to impute
      reliably, so instead of dropping the column outright we convert it
      into a binary indicator 'HasCabin' (1 = cabin recorded, 0 = missing).
      This preserves the signal that recording a cabin correlates with
      higher fare/class, while avoiding fabricated cabin values.
    - Duplicates: detected and removed if present (none are expected in
      this dataset since each row is a unique passenger, but the check
      and removal step is kept for correctness and reproducibility).
    - Dtypes: re-enforced after imputation, since filling NaNs can leave
      columns as generic 'object' dtype.
    """

    df = df.copy()
    n_before = len(df)

    df["Age"] = df["Age"].fillna(age_median)
    df["Embarked"] = df["Embarked"].fillna(embarked_mode)
    df["Fare"] = df["Fare"].fillna(fare_median)

    df["HasCabin"] = df["Cabin"].notnull().astype(int)
    df.drop(columns=["Cabin"], inplace=True)

    n_dupes = df.duplicated().sum()
    df.drop_duplicates(inplace=True)

    df["Pclass"] = df["Pclass"].astype(int)
    df["Age"] = df["Age"].astype(float)
    df["Fare"] = df["Fare"].astype(float)

    n_after = len(df)
    print(f"\n[{label}] Rows before cleaning: {n_before}")
    print(f"[{label}] Duplicate rows removed: {n_dupes}")
    print(f"[{label}] Rows after cleaning: {n_after}")
    remaining_na = df.isnull().sum()
    print(f"[{label}] Remaining missing values:\n{remaining_na[remaining_na > 0]}")

    return df


train_clean = clean_data(
    train,
    age_median,
    fare_median,
    embarked_mode,
    label="TRAIN",
)
test_clean = clean_data(
    test,
    age_median,
    fare_median,
    embarked_mode,
    label="TEST",
)

print("\nImputation values used (derived from TRAIN only):")
print(f"  Age median    = {age_median}")
print(f"  Fare median   = {fare_median:.4f}")
print(f"  Embarked mode = '{embarked_mode}'")
print(f"\nTrain shape after cleaning: {train_clean.shape}")
print(f"Test shape after cleaning: {test_clean.shape}")

stats_summary["age_median_used"] = float(age_median)
stats_summary["fare_median_used"] = float(fare_median)
stats_summary["embarked_mode_used"] = embarked_mode

# =========================================================================
# TASK 3: DATA VISUALISATION
# =========================================================================
print("\n" + "=" * 75)
print("TASK 3: DATA VISUALISATION")
print("=" * 75)

# Keep this section lightweight so the script reaches the text-based analysis
# and machine learning results promptly.
print("Generating a small set of summary figures...")

plt.figure(figsize=(8, 5))
plt.hist(train_clean["Age"], bins=30, color="#4F81BD", edgecolor="#2F4F4F")
plt.title("Distribution of Passenger Ages")
plt.xlabel("Age (years)")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.savefig(FIG_DIR / "1_age_histogram.png")
try:
    if sys.platform.startswith("win"):
        os.startfile(FIG_DIR / "1_age_histogram.png")
    else:
        subprocess.run(["xdg-open", str(FIG_DIR / "1_age_histogram.png")], check=False)
except Exception:
    pass
plt.close()

plt.figure(figsize=(7, 5))
class_counts = train_clean["Pclass"].value_counts().sort_index()
bars = plt.bar(class_counts.index.astype(str), class_counts.values, color=["#4F81BD", "#A6A6A6", "#2F5597"])
plt.title("Passenger Class Distribution")
plt.xlabel("Passenger Class")
plt.ylabel("Number of Passengers")
for bar in bars:
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 5,
        str(int(bar.get_height())),
        ha="center",
    )
plt.tight_layout()
plt.savefig(FIG_DIR / "2_pclass_bar.png")
try:
    if sys.platform.startswith("win"):
        os.startfile(FIG_DIR / "2_pclass_bar.png")
    else:
        subprocess.run(["xdg-open", str(FIG_DIR / "2_pclass_bar.png")], check=False)
except Exception:
    pass
plt.close()

numeric_cols = [
    "Survived",
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "HasCabin",
]
corr = train_clean[numeric_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", center=0, square=True)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(FIG_DIR / "3_correlation_heatmap.png")
try:
    if sys.platform.startswith("win"):
        os.startfile(FIG_DIR / "3_correlation_heatmap.png")
    else:
        subprocess.run(["xdg-open", str(FIG_DIR / "3_correlation_heatmap.png")], check=False)
except Exception:
    pass
plt.close()

plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=train_clean,
    x="Age",
    y="Fare",
    hue="Survived",
    style="Pclass",
    palette=["#4F81BD", "#C0504D"],
    alpha=0.75,
    s=60,
)
plt.title("Age vs. Fare by Survival and Passenger Class")
plt.xlabel("Age (years)")
plt.ylabel("Fare (£)")
plt.tight_layout()
plt.savefig(FIG_DIR / "4_age_vs_fare_scatter.png")
try:
    if sys.platform.startswith("win"):
        os.startfile(FIG_DIR / "4_age_vs_fare_scatter.png")
    else:
        subprocess.run(["xdg-open", str(FIG_DIR / "4_age_vs_fare_scatter.png")], check=False)
except Exception:
    pass
plt.close()

pairplot = sns.pairplot(
    train_clean[["Age", "Fare", "Pclass", "Survived"]],
    hue="Survived",
    vars=["Age", "Fare", "Pclass"],
    diag_kind="hist",
    palette=["#4F81BD", "#C0504D"],
    plot_kws={"alpha": 0.7},
)
pairplot.fig.suptitle("Pair Plot of Age, Fare, and Pclass by Survival", y=1.02)
pairplot.fig.tight_layout()
pairplot.fig.savefig(FIG_DIR / "5_pairplot_age_fare_pclass_survival.png")
try:
    if sys.platform.startswith("win"):
        os.startfile(FIG_DIR / "5_pairplot_age_fare_pclass_survival.png")
    else:
        subprocess.run(["xdg-open", str(FIG_DIR / "5_pairplot_age_fare_pclass_survival.png")], check=False)
except Exception:
    pass

print(f"\nBasic EDA figures saved to '{FIG_DIR}/'")

# =========================================================================
# TASK 4: STATISTICAL ANALYSIS
# =========================================================================
print("\n" + "=" * 75)
print("TASK 4: STATISTICAL ANALYSIS")
print("=" * 75)

# ---- 4.1 Descriptive statistics ----
desc_numeric = train_clean[["Age", "Fare", "SibSp", "Parch"]].describe()
print("\nDescriptive statistics (numerical variables):\n", desc_numeric)
desc_numeric.to_csv(OUT_DIR / "descriptive_statistics.csv")

# ---- 4.2 Frequency distribution (categorical variables) ----
freq_sex = train_clean["Sex"].value_counts()
freq_survived = train_clean["Survived"].value_counts().rename(
    {0: "Did not survive", 1: "Survived"}
)
freq_pclass = train_clean["Pclass"].value_counts().sort_index()
freq_embarked = train_clean["Embarked"].value_counts()

print("\nFrequency distribution - Sex:\n", freq_sex)
print("\nFrequency distribution - Survived:\n", freq_survived)
print("\nFrequency distribution - Pclass:\n", freq_pclass)
print("\nFrequency distribution - Embarked:\n", freq_embarked)

survival_rate = train_clean["Survived"].mean()
survival_by_sex = train_clean.groupby("Sex")["Survived"].mean()
survival_by_class = train_clean.groupby("Pclass")["Survived"].mean()

print(f"\nOverall survival rate: {survival_rate:.4f}")
print("\nSurvival rate by sex:\n", survival_by_sex)
print("\nSurvival rate by class:\n", survival_by_class)

# ---- 4.3 Correlation analysis ----
print("\nFull correlation matrix:\n", corr)

corr_with_survival = corr["Survived"].drop("Survived").sort_values(ascending=False)
strongest_positive = corr_with_survival.idxmax()
strongest_positive_val = corr_with_survival.max()
strongest_negative = corr_with_survival.idxmin()
strongest_negative_val = corr_with_survival.min()

print(
    f"\nStrongest POSITIVE correlation with Survived: {strongest_positive} "
    f"(r = {strongest_positive_val:.3f})"
)
print(
    f"Strongest NEGATIVE correlation with Survived: {strongest_negative} "
    f"(r = {strongest_negative_val:.3f})"
)

# ---- 4.4 Three important statistical findings (printed for the report) ----
print("\nTHREE KEY STATISTICAL FINDINGS:")
print(
    "1. Sex was the single strongest predictor of survival: women survived "
    f"at {survival_by_sex['female']*100:.1f}% versus {survival_by_sex['male']*100:.1f}% "
    f"for men, a gap of {abs(survival_by_sex['female']-survival_by_sex['male'])*100:.1f} "
    "percentage points."
)
print(
    "2. Passenger class was strongly associated with survival: 1st class "
    f"passengers survived at {survival_by_class[1]*100:.1f}% compared with "
    f"{survival_by_class[3]*100:.1f}% for 3rd class."
)
print(
    "3. Fare (r = "
    f"{corr.loc['Fare','Survived']:.2f}) and HasCabin "
    f"(r = {corr.loc['HasCabin','Survived']:.2f}) were both positively correlated "
    "with survival, reinforcing that socio-economic status shaped survival odds, "
    "while Age (r = "
    f"{corr.loc['Age','Survived']:.2f}) was only weakly related."
)

stats_summary["survival_rate_overall"] = float(survival_rate)
stats_summary["survival_by_sex"] = survival_by_sex.to_dict()
stats_summary["survival_by_class"] = survival_by_class.to_dict()
stats_summary["strongest_positive_corr"] = {
    "feature": strongest_positive,
    "r": float(strongest_positive_val),
}
stats_summary["strongest_negative_corr"] = {
    "feature": strongest_negative,
    "r": float(strongest_negative_val),
}
stats_summary["corr_matrix"] = corr.round(3).to_dict()

# =========================================================================
# TASK 5: MACHINE LEARNING - FEATURE ENGINEERING
# =========================================================================
print("\n" + "=" * 75)
print("TASK 5: FEATURE ENGINEERING")
print("=" * 75)

def engineer_features(df):
    df = df.copy()
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["Title"] = df["Name"].str.extract(r",\s*([^\.]*)\.")
    rare_titles = [
        "Lady",
        "Countess",
        "Capt",
        "Col",
        "Don",
        "Dr",
        "Major",
        "Rev",
        "Sir",
        "Jonkheer",
        "Dona",
    ]
    df["Title"] = df["Title"].replace(rare_titles, "Rare")
    df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    return df


train_fe = engineer_features(train_clean)
test_fe = engineer_features(test_clean)

cat_cols = ["Sex", "Embarked", "Title"]
for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([train_fe[col], test_fe[col]], axis=0).astype(str)
    le.fit(combined)
    train_fe[col] = le.transform(train_fe[col].astype(str))
    test_fe[col] = le.transform(test_fe[col].astype(str))

# Select suitable predictor variables
feature_cols = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked",
    "HasCabin",
    "FamilySize",
    "IsAlone",
    "Title",
]
print(f"\nPredictor variables selected: {feature_cols}")
print(
    "Rationale: Pclass, Sex, Fare and HasCabin proxy socio-economic status "
    "(shown in Task 4 to correlate with survival); Age, SibSp, Parch, "
    "FamilySize and IsAlone capture demographic/family context; Embarked "
    "and Title add small additional signal. PassengerId, Name and Ticket "
    "are excluded as identifiers with no predictive meaning."
)

X = train_fe[feature_cols]
y = train_fe["Survived"]
X_test_final = test_fe[feature_cols]

# Split the dataset into training and testing sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTraining set size: {X_train.shape[0]} passengers")
print(f"Validation (hold-out test) set size: {X_val.shape[0]} passengers")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_final_scaled = scaler.transform(X_test_final)

# =========================================================================
# TASK 6: MACHINE LEARNING - LOGISTIC REGRESSION (Model Evaluation)
# =========================================================================
print("\n" + "=" * 75)
print("TASK 6: MACHINE LEARNING (LOGISTIC REGRESSION MODEL EVALUATION)")
print("=" * 75)

# 3. Train a Logistic Regression classifier
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_scaled, y_train)

# 4. Predict the testing data
y_pred = log_reg.predict(X_val_scaled)

# 5. Compute accuracy, confusion matrix, classification report
acc = accuracy_score(y_val, y_pred)
cm = confusion_matrix(y_val, y_pred)
report = classification_report(
    y_val, y_pred, target_names=["Did not survive", "Survived"]
)

print(f"\nModel Accuracy: {acc:.4f}")
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", report)

stats_summary["logreg_accuracy"] = float(acc)
stats_summary["logreg_confusion_matrix"] = cm.tolist()

plt.figure(figsize=(6, 5))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Did not survive", "Survived"],
)
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix - Logistic Regression")
plt.tight_layout()
plt.savefig(FIG_DIR / "7_confusion_matrix.png")
try:
    if sys.platform.startswith("win"):
        os.startfile(FIG_DIR / "7_confusion_matrix.png")
    else:
        subprocess.run(["xdg-open", str(FIG_DIR / "7_confusion_matrix.png")], check=False)
except Exception:
    pass
plt.close()

# Coefficients (for discussion of which features drive predictions)
coef_series = pd.Series(log_reg.coef_[0], index=feature_cols).sort_values()
print("\nLogistic Regression coefficients (standardized features):\n", coef_series)

plt.figure(figsize=(8, 6))
coef_series.plot(kind="barh", color=["#4F81BD", "#A6A6A6", "#5B9BD5", "#7F7F7F", "#2F5597", "#6C8EBF", "#C9D9F2", "#B4B4B4", "#4A7FB8", "#4F6180", "#1F497D"])
plt.title("Logistic Regression Coefficients (Standardized Features)")
plt.xlabel("Coefficient value")
plt.ylabel("Feature")
plt.axvline(0, color="black", linewidth=0.8)
plt.tight_layout()
plt.savefig(FIG_DIR / "8_logreg_coefficients.png")
try:
    if sys.platform.startswith("win"):
        os.startfile(FIG_DIR / "8_logreg_coefficients.png")
    else:
        subprocess.run(["xdg-open", str(FIG_DIR / "8_logreg_coefficients.png")], check=False)
except Exception:
    pass
plt.close()

# 6. Predict on the official Kaggle test set + build submission file
test_predictions = log_reg.predict(X_test_final_scaled)
submission = pd.DataFrame({"PassengerId": test_passenger_ids, "Survived": test_predictions})
submission.to_csv(OUT_DIR / "submission.csv", index=False)

baseline_acc = accuracy_score(gender_submission["Survived"], test_predictions)
print(
    f"\nSubmission file saved. Agreement with gender_submission.csv baseline: "
    f"{baseline_acc:.4f}"
)
stats_summary["kaggle_test_agreement_with_baseline"] = float(baseline_acc)

# Save the stats summary for use when writing the report
with open(OUT_DIR / "stats_summary.json", "w") as f:
    json.dump(stats_summary, f, indent=2, default=str)

print("\n" + "=" * 75)
print("ALL TASKS COMPLETE")
print("=" * 75)
