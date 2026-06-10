# auto-ml-pipeline 🤖

A complete, plug-and-play Machine Learning pipeline in Python that automatically handles everything — from raw CSV to a saved trained model — with zero manual preprocessing.

> Just change **2 lines** (`CSV_PATH` and `TARGET_COLUMN`) and run.

---

## What It Does

This pipeline automates the entire ML workflow in 11 steps:

| Step | What Happens |
|------|-------------|
| 1 | Load CSV dataset |
| 2 | Remove irrelevant columns (ID-like, constant, high missing) |
| 3 | Identify numerical vs categorical columns |
| 4 | Fill missing values (median / mode) |
| 5 | Cap outliers using IQR method |
| 6 | Encode (OneHot / Label) + Scale (StandardScaler) |
| 7 | Export cleaned data as CSV |
| 8 | Auto-detect task: Classification or Regression |
| 9 | Handle class imbalance using SMOTE |
| 10 | Train 13+ models + GridSearchCV hyperparameter tuning |
| 11 | Save best model as `.pkl` |

---

## Smart Model Selection

When multiple models score the **same accuracy**, the pipeline does not pick randomly.

It applies a 3-step tie-breaking logic:

1. **Priority Order** — Simpler models are preferred (Logistic Regression > Random Forest > XGBoost)
2. **5-fold Cross Validation** — Runs CV on all tied models to check generalisation
3. **Final Decision** — If CV difference > 0.005, better CV model wins. Otherwise, simpler model wins.

The **reason** for choosing the model is printed clearly in the output.

---

## Models Included

### Classification (13 models)
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- SVM (SVC)
- K-Nearest Neighbors
- Naive Bayes
- Bagging Classifier
- AdaBoost
- Extra Trees
- Voting Classifier (LR + RF + GB)
- LightGBM
- XGBoost

### Regression (15 models)
- Linear Regression
- Ridge, Lasso, ElasticNet, Bayesian Ridge
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- SVR
- Bagging, AdaBoost, Extra Trees Regressor
- Voting Regressor (LR + RF + GB)
- LightGBM Regressor
- XGBoost Regressor

---

## Installation

```bash
git clone https://github.com/your-username/auto-ml-pipeline.git
cd auto-ml-pipeline
pip install -r requirements.txt
```

### Requirements
```
pandas
numpy
scikit-learn
imbalanced-learn
lightgbm
xgboost
```

Or install manually:
```bash
pip install pandas numpy scikit-learn imbalanced-learn lightgbm xgboost
```

---

## Usage

### Step 1 — Open `ml_pipeline_v2.py` and update these 2 lines:

```python
CSV_PATH      = "your_data.csv"     # path to your CSV file
TARGET_COLUMN = "your_target"       # column you want to predict
```

### Step 2 — Run the pipeline:

```bash
python ml_pipeline_v2.py
```

### Output:
- `cleaned_data.csv` — preprocessed dataset
- `best_model_<ModelName>.pkl` — saved best model

---

## Configuration

All settings are at the top of the file and easy to change:

```python
TEST_SIZE             = 0.2    # 80% train, 20% test
RANDOM_STATE          = 42     # for reproducibility
MISSING_THRESHOLD     = 0.5    # drop column if >50% values missing
OUTLIER_IQR_FACTOR    = 1.5    # standard IQR multiplier
IMBALANCE_THRESHOLD   = 0.2    # apply SMOTE if minority class < 20%
```

---

## Sample Output

```
══════════════════════════════════════════════════════════
  FINAL RESULTS TABLE - Classification
══════════════════════════════════════════════════════════
  Model                                    Test Acc %
  ────────────────────────────────────────────────────
🏆 Random Forest                              88.27%
   XGBoost                                   87.71%
   Gradient Boosting                         86.59%
   Logistic Regression                       83.24%
   ...
  ────────────────────────────────────────────────────

──────────────────────────────────────────────────
  Selection Reason
──────────────────────────────────────────────────
  Winner       : Random Forest
  Why chosen   : Highest Test Accuracy of 88.27%
  Model nature : COMPLEX — bagged ensemble of trees, robust and stable.
```

---

## Project Structure

```
auto-ml-pipeline/
│
├── ml_pipeline_v2.py      # main pipeline script
├── cleaned_data.csv        # generated after running (gitignored)
├── best_model_*.pkl        # saved model (gitignored)
├── requirements.txt        # dependencies
└── README.md               # this file
```

---

## Important Notes

- **100% accuracy on all models** = data leakage, not a good sign. Check your columns with `df.corr()[TARGET_COLUMN]`
- **SMOTE** is applied only on training data — test data is never touched
- **LightGBM and XGBoost** are optional — pipeline works without them too
- Columns with **>50% missing values** are automatically dropped

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-latest-green)
![LightGBM](https://img.shields.io/badge/LightGBM-latest-yellow)
![pandas](https://img.shields.io/badge/pandas-latest-lightblue)

---

## License

MIT License — free to use, modify, and distribute.

---

## Author

Made with love for the ML community.
If this helped you, drop a star!
