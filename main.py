"""
Complete ML Pipeline v2
=======================
- Categorical & Numerical columns auto-detect
- Irrelevant columns auto-removal (constant, ID-like, high missing)
- Missing value handling
- Outlier handling (IQR capping)
- StandardScaler (numerical) + LabelEncoder/OneHotEncoder (categorical)
- Class Imbalance handling (SMOTE)
- Clean CSV export
- Auto-detect task: Classification / Regression
- Supervised: multiple models + accuracy/metrics
- Added: LightGBM, XGBoost, Bagging, ExtraTrees, AdaBoost, VotingClassifier/Regressor
- Added: Hyperparameter Tuning via GridSearchCV
- NEW: Smart model selection with TIE-BREAKING logic
- NEW: Best model selection REASON clearly explained
- Final: All models accuracy table (Test Accuracy only)
"""

import pandas as pd
import numpy as np
import warnings
import sys
import os
import pickle

warnings.filterwarnings("ignore")

try:
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import (
        accuracy_score, mean_squared_error, r2_score,
        precision_score, recall_score, f1_score
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import (
        RandomForestClassifier, GradientBoostingClassifier,
        BaggingClassifier, ExtraTreesClassifier, AdaBoostClassifier,
        VotingClassifier
    )
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import (
        RandomForestRegressor, GradientBoostingRegressor,
        BaggingRegressor, ExtraTreesRegressor, AdaBoostRegressor,
        VotingRegressor
    )
    from sklearn.svm import SVR
except ImportError:
    os.system("pip install scikit-learn pandas numpy --break-system-packages -q")
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import (
        accuracy_score, mean_squared_error, r2_score,
        precision_score, recall_score, f1_score
    )
    from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.ensemble import (
        RandomForestClassifier, GradientBoostingClassifier,
        BaggingClassifier, ExtraTreesClassifier, AdaBoostClassifier, VotingClassifier,
        RandomForestRegressor, GradientBoostingRegressor,
        BaggingRegressor, ExtraTreesRegressor, AdaBoostRegressor, VotingRegressor
    )
    from sklearn.svm import SVC, SVR
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB

# ── SMOTE ────────────────────────────────────────────────────────────────────
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    os.system("pip install imbalanced-learn --break-system-packages -q")
    try:
        from imblearn.over_sampling import SMOTE
        SMOTE_AVAILABLE = True
    except ImportError:
        SMOTE_AVAILABLE = False

# ── LightGBM ─────────────────────────────────────────────────────────────────
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    os.system("pip install lightgbm --break-system-packages -q")
    try:
        import lightgbm as lgb
        LIGHTGBM_AVAILABLE = True
    except ImportError:
        LIGHTGBM_AVAILABLE = False

# ── XGBoost ──────────────────────────────────────────────────────────────────
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    os.system("pip install xgboost --break-system-packages -q")
    try:
        import xgboost as xgb
        XGBOOST_AVAILABLE = True
    except ImportError:
        XGBOOST_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
CSV_PATH              = "titanic - titanic.csv"
TARGET_COLUMN         = "Survived"
CLEANED_CSV_PATH      = "cleaned_data.csv"
TEST_SIZE             = 0.2
RANDOM_STATE          = 42
MISSING_THRESHOLD     = 0.5
OUTLIER_IQR_FACTOR    = 1.5
IMBALANCE_THRESHOLD   = 0.2

# ── TIE-BREAK PRIORITY ORDER ──────────────────────────────────────────────────
# Jab 2+ models ki accuracy BILKUL same ho, toh yeh order decide karta hai winner.
# Index 0 = highest priority (prefer karo), last index = lowest priority.
# Simple models pehle — woh faster, interpretable, aur less prone to overfitting hote hain.
TIE_BREAK_PRIORITY = [
    "Logistic Regression",      # simplest, most interpretable
    "Linear Regression",
    "Ridge Regression",
    "Lasso Regression",
    "ElasticNet",
    "Bayesian Ridge",
    "Naive Bayes",
    "K-Nearest Neighbors",
    "Decision Tree",
    "Decision Tree Regressor",
    "SVM (SVC)",
    "SVR",
    "Bagging (DT base)",
    "AdaBoost",
    "AdaBoost Regressor",
    "Extra Trees",
    "Extra Trees Regressor",
    "Gradient Boosting",
    "Gradient Boosting Regressor",
    "Random Forest",
    "Random Forest Regressor",
    "LightGBM",
    "LightGBM Regressor",
    "XGBoost",
    "XGBoost Regressor",
    "Voting (LR+RF+GB)",        # ensemble of ensembles — most complex
]
# ══════════════════════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────────────────────
# MODEL COMPLEXITY MAP  (interpretability score: lower = simpler)
# Used in selection reasoning to describe the chosen model's nature.
# ─────────────────────────────────────────────────────────────────────────────
MODEL_COMPLEXITY = {
    "Logistic Regression"         : ("simple",   "linear, highly interpretable, fast"),
    "Linear Regression"           : ("simple",   "linear, highly interpretable, fast"),
    "Ridge Regression"            : ("simple",   "regularised linear, handles multicollinearity well"),
    "Lasso Regression"            : ("simple",   "regularised linear with built-in feature selection"),
    "ElasticNet"                  : ("simple",   "combines Ridge + Lasso penalties"),
    "Bayesian Ridge"              : ("simple",   "probabilistic linear model, robust to noise"),
    "Naive Bayes"                 : ("simple",   "probabilistic, extremely fast, few assumptions"),
    "K-Nearest Neighbors"         : ("simple",   "non-parametric, no training phase, instance-based"),
    "Decision Tree"               : ("moderate", "rule-based, fully interpretable via tree structure"),
    "Decision Tree Regressor"     : ("moderate", "rule-based, fully interpretable via tree structure"),
    "SVM (SVC)"                   : ("moderate", "margin-maximising, effective in high dimensions"),
    "SVR"                         : ("moderate", "margin-maximising regressor, robust to outliers"),
    "Bagging (DT base)"           : ("moderate", "reduces variance by averaging many DT models"),
    "AdaBoost"                    : ("moderate", "boosting ensemble, corrects weak learner mistakes"),
    "AdaBoost Regressor"          : ("moderate", "boosting ensemble, corrects weak learner mistakes"),
    "Extra Trees"                 : ("complex",  "fully randomised forest, very fast training"),
    "Extra Trees Regressor"       : ("complex",  "fully randomised forest, very fast training"),
    "Gradient Boosting"           : ("complex",  "sequential boosting, handles non-linear data well"),
    "Gradient Boosting Regressor" : ("complex",  "sequential boosting, handles non-linear data well"),
    "Random Forest"               : ("complex",  "bagged ensemble of trees, robust and stable"),
    "Random Forest Regressor"     : ("complex",  "bagged ensemble of trees, robust and stable"),
    "LightGBM"                    : ("complex",  "leaf-wise gradient boosting, very fast on large data"),
    "LightGBM Regressor"          : ("complex",  "leaf-wise gradient boosting, very fast on large data"),
    "XGBoost"                     : ("complex",  "regularised gradient boosting, competition favourite"),
    "XGBoost Regressor"           : ("complex",  "regularised gradient boosting, competition favourite"),
    "Voting (LR+RF+GB)"           : ("complex",  "soft-voting ensemble of 3 models, blended predictions"),
}


def print_header(text):
    print("\n" + "═" * 60)
    print(f"  {text}")
    print("═" * 60)


def print_section(text):
    print(f"\n{'─'*50}")
    print(f"  {text}")
    print(f"{'─'*50}")


# ─────────────────────────────────────────────────────────────────────────────
# SMART MODEL SELECTOR  ← NEW
# ─────────────────────────────────────────────────────────────────────────────
def select_best_model_with_reason(results: dict, task: str,
                                   trained_models: dict,
                                   X_train, y_train,
                                   X_test,  y_test):
    """
    Smart model selection with full reasoning explanation.

    Logic:
    ──────
    Classification:
      1. Find the best Test Accuracy score.
      2. Collect ALL models that share that exact top score (ties).
      3. If only one winner → that's the best. Done.
      4. If multiple tied winners → apply TIE_BREAK_PRIORITY order.
         (Prefer simpler/faster models when accuracy is identical.)
      5. Run 5-fold CV on the tied models to check generalisation.
      6. Print a clear REASON section explaining every decision made.

    Regression:
      Same logic but uses R² Score as the primary metric.

    Returns:
      best_name  (str)   — name of the chosen model
      best_model (object)— the trained model object
    """
    print_header("STEP 10b: Smart Model Selection + Reason")

    if task == "classification":
        metric_key  = "Test Accuracy"
        metric_label = "Test Accuracy"
        best_score  = max(v[metric_key] for v in results.values())
        tied_models = [name for name, v in results.items()
                       if v[metric_key] == best_score]
    else:
        metric_key  = "R² Score"
        metric_label = "R² Score"
        best_score  = max(v[metric_key] for v in results.values())
        tied_models = [name for name, v in results.items()
                       if v[metric_key] == best_score]

    print(f"\n  Best {metric_label}  : {best_score}")
    print(f"  Models achieving this score ({len(tied_models)}):")
    for m in tied_models:
        print(f"    • {m}")

    # ── Case 1: Only one best model ──────────────────────────────────────────
    if len(tied_models) == 1:
        best_name  = tied_models[0]
        best_model = trained_models[best_name]
        complexity, description = MODEL_COMPLEXITY.get(
            best_name, ("unknown", "no description available"))

        print_section("Selection Reason")
        print(f"  ✔  Winner       : {best_name}")
        print(f"  ✔  Why chosen   : This model achieved the HIGHEST {metric_label} of"
              f" {best_score} — no other model matched it.")
        print(f"  ✔  Model nature : {complexity.upper()} — {description}.")

        if complexity == "simple":
            print(f"  ✔  Bonus        : Being a simpler model means it's also faster,"
                  f" more interpretable, and less likely to overfit.")
        elif complexity == "moderate":
            print(f"  ✔  Note         : Moderate complexity — good balance of"
                  f" performance and interpretability.")
        else:
            print(f"  ✔  Note         : Complex ensemble model — strong performance"
                  f" but less interpretable than linear models.")

        return best_name, best_model

    # ── Case 2: TIE — multiple models share the top score ───────────────────
    print(f"\n  ⚠  TIE DETECTED — {len(tied_models)} models have identical {metric_label}.")
    print(f"  Applying tie-break logic...")

    # Step A: Priority order tie-break
    print(f"\n  Step A — Priority Order Tie-Break")
    print(f"  (Simpler models preferred when accuracy is equal)")

    priority_winner = None
    for preferred in TIE_BREAK_PRIORITY:
        if preferred in tied_models:
            priority_winner = preferred
            print(f"    → Priority winner : {preferred}")
            break

    if priority_winner is None:
        priority_winner = tied_models[0]
        print(f"    → Fallback to first tied model : {priority_winner}")

    # Step B: CV score check among tied models
    print(f"\n  Step B — Cross-Validation (5-fold) among tied models")
    print(f"  {'Model':<40} {'CV Mean':>10} {'CV Std':>8}")
    print("  " + "-" * 62)

    cv_scores = {}
    for name in tied_models:
        try:
            model  = trained_models[name]
            scoring = "accuracy" if task == "classification" else "r2"
            scores  = cross_val_score(model, X_train, y_train,
                                      cv=5, scoring=scoring, n_jobs=-1)
            cv_scores[name] = {"mean": scores.mean(), "std": scores.std()}
            print(f"  {name:<40} {scores.mean():>9.4f} {scores.std():>7.4f}")
        except Exception as e:
            cv_scores[name] = {"mean": 0.0, "std": 9999.0}
            print(f"  {name:<40}   [CV failed: {e}]")

    print("  " + "-" * 62)

    # Step C: Best CV model among tied
    best_cv_model = max(cv_scores, key=lambda k: cv_scores[k]["mean"])
    best_cv_mean  = cv_scores[best_cv_model]["mean"]

    # Step D: Final decision
    # If CV winner matches priority winner → clear choice
    # If they disagree → prefer CV winner (generalisation > priority)
    if best_cv_model == priority_winner:
        final_winner = priority_winner
        decision_basis = "Both priority order AND cross-validation agree on this model."
    else:
        # Check if CV scores are meaningfully different (>0.005 threshold)
        priority_cv   = cv_scores[priority_winner]["mean"]
        cv_winner_cv  = cv_scores[best_cv_model]["mean"]
        diff = abs(cv_winner_cv - priority_cv)

        if diff > 0.005:
            final_winner   = best_cv_model
            decision_basis = (
                f"CV score of {best_cv_model} ({cv_winner_cv:.4f}) is meaningfully "
                f"higher than priority model {priority_winner} ({priority_cv:.4f}). "
                f"Better generalisation preferred."
            )
        else:
            final_winner   = priority_winner
            decision_basis = (
                f"CV scores are nearly equal (diff={diff:.4f} < 0.005 threshold). "
                f"Simpler/faster model {priority_winner} preferred."
            )

    # ── Final Reason Print ───────────────────────────────────────────────────
    complexity, description = MODEL_COMPLEXITY.get(
        final_winner, ("unknown", "no description available"))

    print_section("Selection Reason (Tie-Break)")
    print(f"  ✔  Winner             : {final_winner}")
    print(f"  ✔  Why there was a tie: {len(tied_models)} models all scored"
          f" {metric_label} = {best_score}.")
    print(f"  ✔  Tie-break decision : {decision_basis}")
    print(f"  ✔  Model nature       : {complexity.upper()} — {description}.")
    print(f"  ✔  CV Score (5-fold)  : {cv_scores[final_winner]['mean']:.4f}"
          f" ± {cv_scores[final_winner]['std']:.4f}")

    if complexity == "simple":
        print(f"\n  ℹ  Why prefer simpler models in a tie?")
        print(f"     • Faster prediction at inference time.")
        print(f"     • Easier to explain to stakeholders.")
        print(f"     • Less risk of overfitting on unseen data.")
        print(f"     • Easier to debug if something goes wrong.")
    elif complexity == "complex":
        print(f"\n  ℹ  A complex model won — this suggests the data has non-linear")
        print(f"     patterns that simpler models couldn't capture as well.")

    return final_winner, trained_models[final_winner]


def print_results_table_classification(results: dict):
    print_header("FINAL RESULTS TABLE – Classification")
    header = f"  {'Model':<40} {'Test Acc %':>10}"
    print(header)
    print("  " + "-" * 52)
    sorted_results = sorted(results.items(),
                            key=lambda x: x[1]["Test Accuracy"], reverse=True)
    for i, (name, metrics) in enumerate(sorted_results):
        prefix = "🏆" if i == 0 else "  "
        print(f"{prefix} {name:<40} {metrics['Test Accuracy']:>9.2f}%")
    print("  " + "-" * 52)


def print_results_table_regression(results: dict):
    print_header("FINAL RESULTS TABLE – Regression")
    header = f"  {'Model':<40} {'R² Score':>10} {'RMSE':>12}"
    print(header)
    print("  " + "-" * 65)
    sorted_results = sorted(results.items(),
                            key=lambda x: x[1]["R² Score"], reverse=True)
    for i, (name, metrics) in enumerate(sorted_results):
        prefix = "🏆" if i == 0 else "  "
        print(f"{prefix} {name:<40} {metrics['R² Score']:>10.4f} {metrics['RMSE']:>12.4f}")
    print("  " + "-" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETER GRIDS
# ─────────────────────────────────────────────────────────────────────────────
CLASSIFICATION_PARAM_GRIDS = {
    "Logistic Regression" : {"C": [0.01, 0.1, 1, 10]},
    "Decision Tree"       : {"max_depth": [None, 5, 10, 20], "min_samples_split": [2, 5, 10]},
    "Random Forest"       : {"n_estimators": [100, 200], "max_depth": [None, 10, 20]},
    "Gradient Boosting"   : {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1, 0.2]},
    "SVM (SVC)"           : {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]},
    "K-Nearest Neighbors" : {"n_neighbors": [3, 5, 7, 11]},
    "Extra Trees"         : {"n_estimators": [100, 200], "max_depth": [None, 10, 20]},
    "AdaBoost"            : {"n_estimators": [50, 100, 200], "learning_rate": [0.5, 1.0]},
    "LightGBM"            : {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "num_leaves": [31, 63]},
    "XGBoost"             : {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [4, 6]},
}

REGRESSION_PARAM_GRIDS = {
    "Ridge Regression"            : {"alpha": [0.1, 1.0, 10.0, 100.0]},
    "Lasso Regression"            : {"alpha": [0.01, 0.1, 1.0, 10.0]},
    "ElasticNet"                  : {"alpha": [0.01, 0.1, 1.0], "l1_ratio": [0.2, 0.5, 0.8]},
    "Decision Tree Regressor"     : {"max_depth": [None, 5, 10, 20], "min_samples_split": [2, 5, 10]},
    "Random Forest Regressor"     : {"n_estimators": [100, 200], "max_depth": [None, 10, 20]},
    "Gradient Boosting Regressor" : {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1, 0.2]},
    "SVR"                         : {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]},
    "Extra Trees Regressor"       : {"n_estimators": [100, 200], "max_depth": [None, 10, 20]},
    "AdaBoost Regressor"          : {"n_estimators": [50, 100, 200], "learning_rate": [0.5, 1.0]},
    "LightGBM Regressor"          : {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "num_leaves": [31, 63]},
    "XGBoost Regressor"           : {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [4, 6]},
}


def tune_model(name, model, param_grid, X_train, y_train, scoring):
    if not param_grid:
        return model, {}
    try:
        gs = GridSearchCV(model, param_grid, cv=5, scoring=scoring, n_jobs=-1)
        gs.fit(X_train, y_train)
        print(f"    Best Params : {gs.best_params_}")
        return gs.best_estimator_, gs.best_params_
    except Exception as e:
        print(f"    [WARN] Tuning failed ({e}), using default params.")
        return model, {}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 – Load Dataset
# ─────────────────────────────────────────────────────────────────────────────
def load_data(path):
    print_header("STEP 1: Load Dataset")
    df = pd.read_csv(path)
    print(f"  Shape        : {df.shape}")
    print(f"  Columns      : {list(df.columns)}")
    print(f"\n  First 3 rows:\n{df.head(3).to_string()}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 – Remove Irrelevant Columns
# ─────────────────────────────────────────────────────────────────────────────
def remove_irrelevant_columns(df, target_col):
    print_header("STEP 2: Remove Irrelevant Columns")
    total_rows   = len(df)
    cols_to_drop = {}

    for col in df.columns:
        if col == target_col:
            continue
        missing_ratio = df[col].isnull().mean()
        n_unique      = df[col].nunique(dropna=True)

        if missing_ratio > MISSING_THRESHOLD:
            pct = round(missing_ratio * 100, 1)
            cols_to_drop[col] = f"High missing values ({pct}% missing > {int(MISSING_THRESHOLD*100)}% threshold)"
        elif n_unique <= 1:
            cols_to_drop[col] = f"Constant column (only {n_unique} unique value)"
        elif n_unique == total_rows:
            cols_to_drop[col] = f"ID-like column ({n_unique} unique values == {total_rows} rows)"

    if cols_to_drop:
        print(f"\n  {len(cols_to_drop)} irrelevant column(s) removed:\n")
        print(f"  {'Column':<30} {'Reason'}")
        print("  " + "-" * 75)
        for col, reason in cols_to_drop.items():
            print(f"  {col:<30} {reason}")
        print("  " + "-" * 75)
        df = df.drop(columns=list(cols_to_drop.keys()))
        print(f"\n  Shape after removal : {df.shape}")
    else:
        print("\n  No irrelevant columns detected.")
        print(f"  Shape unchanged     : {df.shape}")

    print(f"\n  Remaining columns : {list(df.columns)}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 – Identify Numerical & Categorical Columns
# ─────────────────────────────────────────────────────────────────────────────
def identify_columns(df, target_col):
    print_header("STEP 3: Column Identification")
    feature_cols = [c for c in df.columns if c != target_col]
    num_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df[feature_cols].select_dtypes(exclude=[np.number]).columns.tolist()

    for col in num_cols[:]:
        if df[col].nunique() <= 10:
            cat_cols.append(col)
            num_cols.remove(col)
            print(f"  [INFO] '{col}' treated as categorical (<=10 unique values).")

    print(f"\n  Numerical columns  ({len(num_cols)}) : {num_cols}")
    print(f"  Categorical columns({len(cat_cols)}) : {cat_cols}")
    if target_col:
        print(f"  Target column               : {target_col}")
    return num_cols, cat_cols


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 – Handle Missing Values
# ─────────────────────────────────────────────────────────────────────────────
def handle_missing(df, num_cols, cat_cols):
    print_header("STEP 4: Missing Value Handling")
    total_missing = df.isnull().sum().sum()
    print(f"  Total missing values before : {total_missing}")

    if total_missing == 0:
        print("  No missing values found. Data is already clean.")
        return df

    missing_info = df.isnull().sum()
    missing_info = missing_info[missing_info > 0]
    print(f"\n  Missing per column:\n{missing_info.to_string()}")

    if num_cols:
        num_imputer = SimpleImputer(strategy="median")
        df[num_cols] = num_imputer.fit_transform(df[num_cols])
        print(f"\n  Numerical columns filled with median.")

    if cat_cols:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
        print(f"  Categorical columns filled with mode.")

    print(f"\n  Total missing values after  : {df.isnull().sum().sum()}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 – Outlier Handling (IQR Capping)
# ─────────────────────────────────────────────────────────────────────────────
def handle_outliers(df, num_cols):
    print_header("STEP 5: Outlier Handling (IQR Capping)")

    if not num_cols:
        print("  No numerical columns found. Skipping.")
        return df

    total_outliers = 0
    print(f"\n  {'Column':<30} {'Lower Bound':>13} {'Upper Bound':>13} {'Outliers Capped':>16}")
    print("  " + "-" * 75)

    for col in num_cols:
        if col not in df.columns:
            continue
        Q1  = df[col].quantile(0.25)
        Q3  = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - OUTLIER_IQR_FACTOR * IQR
        upper = Q3 + OUTLIER_IQR_FACTOR * IQR

        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        total_outliers += n_outliers
        df[col] = df[col].clip(lower=lower, upper=upper)
        print(f"  {col:<30} {lower:>13.4f} {upper:>13.4f} {n_outliers:>16}")

    print("  " + "-" * 75)
    print(f"\n  Total outlier values capped : {total_outliers}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 – Encoding + Scaling
# ─────────────────────────────────────────────────────────────────────────────
def encode_and_scale(df, num_cols, cat_cols, target_col):
    print_header("STEP 6: Encoding & Scaling")
    df_processed = df.copy()
    encoders     = {}

    for col in cat_cols:
        unique_count = df_processed[col].nunique()
        if unique_count <= 10:
            dummies = pd.get_dummies(df_processed[col], prefix=col, drop_first=True)
            df_processed = pd.concat([df_processed.drop(columns=[col]), dummies], axis=1)
            print(f"  [OK] '{col}' → OneHotEncoding ({unique_count} unique values)")
        else:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            encoders[col] = le
            print(f"  [OK] '{col}' → LabelEncoding ({unique_count} unique values)")

    target_encoder = None
    if target_col and target_col in df_processed.columns:
        if df_processed[target_col].dtype == object or \
                df_processed[target_col].dtype.name == "category":
            le_target = LabelEncoder()
            df_processed[target_col] = le_target.fit_transform(
                df_processed[target_col].astype(str))
            target_encoder = le_target
            print(f"  [OK] Target '{target_col}' → LabelEncoding"
                  f" (classes: {list(le_target.classes_)})")

    scale_cols = [c for c in num_cols if c in df_processed.columns]
    if scale_cols:
        scaler = StandardScaler()
        df_processed[scale_cols] = scaler.fit_transform(df_processed[scale_cols])
        print(f"\n  [OK] StandardScaler applied on: {scale_cols}")

    return df_processed, target_encoder


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 – Export Cleaned CSV
# ─────────────────────────────────────────────────────────────────────────────
def export_clean_csv(df, path):
    print_header("STEP 7: Export Cleaned CSV")
    df.to_csv(path, index=False)
    print(f"  [OK] Cleaned data saved to: '{path}'")
    print(f"  Shape: {df.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 – Determine Task Type
# ─────────────────────────────────────────────────────────────────────────────
def determine_task(df, target_col):
    print_header("STEP 8: Determine Task Type")
    if target_col is None or target_col not in df.columns:
        print("  No target column found. Exiting.")
        sys.exit(1)

    target   = df[target_col]
    n_unique = target.nunique()
    dtype    = target.dtype

    if dtype == object or n_unique <= 20:
        task = "classification"
        print(f"  Target '{target_col}': {n_unique} unique values, dtype={dtype}")
        print(f"  → Task: CLASSIFICATION")
    else:
        task = "regression"
        print(f"  Target '{target_col}': {n_unique} unique values, dtype={dtype}")
        print(f"  → Task: REGRESSION")

    return task


# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 – Class Imbalance Handling (SMOTE)
# ─────────────────────────────────────────────────────────────────────────────
def handle_imbalance(X_train, y_train):
    print_header("STEP 9: Class Imbalance Handling (SMOTE)")

    classes, counts = np.unique(y_train, return_counts=True)
    total           = len(y_train)
    minority_ratio  = counts.min() / total

    print(f"\n  Class distribution in training data:")
    print(f"  {'Class':<15} {'Count':>8} {'Ratio':>8}")
    print("  " + "-" * 35)
    for cls, cnt in zip(classes, counts):
        ratio = cnt / total
        flag  = "  ← minority" if cnt == counts.min() and len(classes) > 1 else ""
        print(f"  {str(cls):<15} {cnt:>8} {ratio:>7.1%}{flag}")
    print("  " + "-" * 35)
    print(f"\n  Minority class ratio : {minority_ratio:.1%}")
    print(f"  Imbalance threshold  : {IMBALANCE_THRESHOLD:.0%}")

    if minority_ratio < IMBALANCE_THRESHOLD:
        if SMOTE_AVAILABLE:
            smote        = SMOTE(random_state=RANDOM_STATE)
            X_res, y_res = smote.fit_resample(X_train, y_train)
            new_classes, new_counts = np.unique(y_res, return_counts=True)
            print(f"\n  After SMOTE — Class distribution:")
            for cls, cnt in zip(new_classes, new_counts):
                print(f"  {str(cls):<15} {cnt:>8} {cnt/len(y_res):>7.1%}")
            return X_res, y_res
        else:
            print(f"\n  [WARN] SMOTE not available. Skipping.")
            return X_train, y_train
    else:
        print(f"\n  [BALANCED] SMOTE not needed.")
        return X_train, y_train


# ─────────────────────────────────────────────────────────────────────────────
# STEP 10a – Classification Models + Hypertuning
# ─────────────────────────────────────────────────────────────────────────────
def run_classification(X_train, X_test, y_train, y_test):
    print_header("STEP 10: Classification Models + Hyperparameter Tuning")

    models = {
        "Logistic Regression" : LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree"       : DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest"       : RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "Gradient Boosting"   : GradientBoostingClassifier(random_state=RANDOM_STATE),
        "SVM (SVC)"           : SVC(random_state=RANDOM_STATE),
        "K-Nearest Neighbors" : KNeighborsClassifier(),
        "Naive Bayes"         : GaussianNB(),
        "Bagging (DT base)"   : BaggingClassifier(
                                    estimator=DecisionTreeClassifier(),
                                    n_estimators=50, random_state=RANDOM_STATE),
        "AdaBoost"            : AdaBoostClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "Extra Trees"         : ExtraTreesClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "Voting (LR+RF+GB)"   : VotingClassifier(
                                    estimators=[
                                        ("lr", LogisticRegression(max_iter=1000)),
                                        ("rf", RandomForestClassifier(n_estimators=100,
                                                                      random_state=RANDOM_STATE)),
                                        ("gb", GradientBoostingClassifier(random_state=RANDOM_STATE))
                                    ], voting="soft"),
    }

    if LIGHTGBM_AVAILABLE:
        models["LightGBM"] = lgb.LGBMClassifier(n_estimators=200,
                                                  random_state=RANDOM_STATE, verbose=-1)
    if XGBOOST_AVAILABLE:
        models["XGBoost"]  = xgb.XGBClassifier(n_estimators=200, random_state=RANDOM_STATE,
                                                eval_metric="logloss", verbosity=0)

    results        = {}
    trained_models = {}

    for name, model in models.items():
        try:
            print(f"\n  [{name}]")
            param_grid = CLASSIFICATION_PARAM_GRIDS.get(name, {})
            if param_grid:
                print(f"    Running GridSearchCV (cv=5)...")
                model, _ = tune_model(name, model, param_grid, X_train, y_train, "accuracy")
            else:
                model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            acc    = accuracy_score(y_test, y_pred)
            results[name]        = {"Test Accuracy": round(acc * 100, 2)}
            trained_models[name] = model
            print(f"    Test Accuracy : {acc*100:.2f}%")

        except Exception as e:
            print(f"    [ERROR] {e}")

    print_results_table_classification(results)

    # ── Smart selection with reason ──────────────────────────────────────────
    best_name, best_model = select_best_model_with_reason(
        results, "classification", trained_models,
        X_train, y_train, X_test, y_test
    )

    return results, best_model, best_name


# ─────────────────────────────────────────────────────────────────────────────
# STEP 10b – Regression Models + Hypertuning
# ─────────────────────────────────────────────────────────────────────────────
def run_regression(X_train, X_test, y_train, y_test):
    print_header("STEP 10: Regression Models + Hyperparameter Tuning")

    models = {
        "Linear Regression"           : LinearRegression(),
        "Ridge Regression"            : Ridge(random_state=RANDOM_STATE),
        "Lasso Regression"            : Lasso(random_state=RANDOM_STATE),
        "Decision Tree Regressor"     : DecisionTreeRegressor(random_state=RANDOM_STATE),
        "Random Forest Regressor"     : RandomForestRegressor(n_estimators=100,
                                                               random_state=RANDOM_STATE),
        "Gradient Boosting Regressor" : GradientBoostingRegressor(random_state=RANDOM_STATE),
        "SVR"                         : SVR(),
        "ElasticNet"                  : ElasticNet(random_state=RANDOM_STATE),
        "Bayesian Ridge"              : BayesianRidge(),
        "Bagging (DT base)"           : BaggingRegressor(
                                            estimator=DecisionTreeRegressor(),
                                            n_estimators=50, random_state=RANDOM_STATE),
        "AdaBoost Regressor"          : AdaBoostRegressor(n_estimators=100,
                                                           random_state=RANDOM_STATE),
        "Extra Trees Regressor"       : ExtraTreesRegressor(n_estimators=100,
                                                             random_state=RANDOM_STATE),
        "Voting (LR+RF+GB)"           : VotingRegressor(
                                            estimators=[
                                                ("lr", LinearRegression()),
                                                ("rf", RandomForestRegressor(n_estimators=100,
                                                                             random_state=RANDOM_STATE)),
                                                ("gb", GradientBoostingRegressor(
                                                    random_state=RANDOM_STATE))
                                            ]),
    }

    if LIGHTGBM_AVAILABLE:
        models["LightGBM Regressor"] = lgb.LGBMRegressor(n_estimators=200,
                                                           random_state=RANDOM_STATE, verbose=-1)
    if XGBOOST_AVAILABLE:
        models["XGBoost Regressor"]  = xgb.XGBRegressor(n_estimators=200,
                                                          random_state=RANDOM_STATE, verbosity=0)

    results        = {}
    trained_models = {}

    for name, model in models.items():
        try:
            print(f"\n  [{name}]")
            param_grid = REGRESSION_PARAM_GRIDS.get(name, {})
            if param_grid:
                print(f"    Running GridSearchCV (cv=5)...")
                model, _ = tune_model(name, model, param_grid, X_train, y_train, "r2")
            else:
                model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
            r2     = r2_score(y_test, y_pred)
            results[name]        = {"R² Score": round(r2, 4), "RMSE": round(rmse, 4)}
            trained_models[name] = model
            print(f"    R² Score : {r2:.4f}")
            print(f"    RMSE     : {rmse:.4f}")

        except Exception as e:
            print(f"    [ERROR] {e}")

    print_results_table_regression(results)

    # ── Smart selection with reason ──────────────────────────────────────────
    best_name, best_model = select_best_model_with_reason(
        results, "regression", trained_models,
        X_train, y_train, X_test, y_test
    )

    return results, best_model, best_name


# ─────────────────────────────────────────────────────────────────────────────
# STEP 11 – Save Best Model as Pickle
# ─────────────────────────────────────────────────────────────────────────────
def save_best_model(model, model_name, task):
    safe_name   = model_name.replace(" ", "_").replace("²", "2").replace("/", "_")
    pickle_path = f"best_model_{safe_name}.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\n  [OK] Best model saved as pickle!")
    print(f"    Model : {model_name}")
    print(f"    File  : {pickle_path}")
    return pickle_path


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "█" * 60)
    print("  ML PIPELINE – Auto Preprocessing + Modeling + Hypertuning")
    print("█" * 60)

    if not os.path.exists(CSV_PATH):
        print(f"\n[ERROR] File not found: '{CSV_PATH}'")
        print("  → Update the CSV_PATH variable with the correct file path.")
        sys.exit(1)

    df = load_data(CSV_PATH)
    df = remove_irrelevant_columns(df, TARGET_COLUMN)
    num_cols, cat_cols = identify_columns(df, TARGET_COLUMN)
    df = handle_missing(df, num_cols, cat_cols)
    df = handle_outliers(df, num_cols)
    df_processed, target_encoder = encode_and_scale(df, num_cols, cat_cols, TARGET_COLUMN)
    export_clean_csv(df_processed, CLEANED_CSV_PATH)
    task = determine_task(df_processed, TARGET_COLUMN)

    X = df_processed.drop(columns=[TARGET_COLUMN]).values
    y = df_processed[TARGET_COLUMN].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"\n  Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

    if task == "classification":
        X_train, y_train = handle_imbalance(X_train, y_train)
        _, best_model, best_name = run_classification(X_train, X_test, y_train, y_test)
    else:
        _, best_model, best_name = run_regression(X_train, X_test, y_train, y_test)

    print_header("STEP 11: Save Best Model")
    save_best_model(best_model, best_name, task)

    safe = best_name.replace(" ", "_").replace("²", "2").replace("/", "_")
    print_header("PIPELINE COMPLETE – FINAL SUMMARY")
    print(f"  Cleaned CSV      : {CLEANED_CSV_PATH}")
    print(f"  Task Detected    : {task.upper()}")
    print(f"  Best Model       : {best_name}")
    print(f"  Pickle Saved     : best_model_{safe}.pkl")
    print("\n  Done!")


if __name__ == "__main__":
    main()