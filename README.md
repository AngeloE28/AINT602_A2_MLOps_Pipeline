# AINT602 Assessment 2
## Angelo Estoque
## Student Performance Classification - MLOps Pipeline

## 1. Project Description

This repository implements a modular, reproducible MLOps pipeline that predicts `Student_Performance` (`Excellent`, `Good`, `Average`, `Poor`) from the [Student Performance & Teaching Dataset](https://www.kaggle.com/datasets/zara2099/student-performance-and-teaching-dataset/data).

---

## 2. What This Repository Contains

```
├── configs/
│   └── baseline.yaml        # Central YAML config for dataset, splits, and model hyperparameters
├── data/
│   └── raw/                 # DVC-tracked raw dataset (Student_Performance_Teaching_Effectiveness.csv)
├── models/                  # Exported reusable scikit-learn pipeline model (.joblib)
├── notebooks/
│   └── staging.ipynb        # Initial exploratory analysis and prototyping notebook
├── src/
│   ├── __init__.py          # Package marker
│   ├── validate.py          # Input schema, null, duplicate, category, and range checks
│   ├── features.py          # Engineered features and preprocessing pipeline builder
│   ├── train.py             # 10-fold CV training, MLflow logging, test evaluation, and model export
│   └── predict.py           # CLI batch inference script using the saved pipeline
├── tests/
│   └── test_pipeline.py     # Automated pytest suite for features, pipeline, and saved model
├── Dockerfile               # Reproducible container definition (Python 3.12-slim)
├── .dockerignore            # Excludes Git, DVC cache, and local artifacts from image builds
└── requirements.txt         # Python package dependencies

```

---

## 3. Data & Model Specification

### Required Input Schema

Both training and inference expect the following 7 raw columns (enforced by `src/validate.py`):

| Column | Type | Valid Values / Range |
| --- | --- | --- |
| `Age` | Numeric | `18` – `100` |
| `Gender` | Categorical | `Male`, `Female`, `Other` |
| `Attendance_Percentage` | Numeric | `0` – `100` |
| `Assignments_Completed` | Numeric | `0` – `10` |
| `Class_Participation` | Numeric | `0` – `10` |
| `Study_Hours_per_Week` | Numeric | `0` – `168` |
| `Projects_Submitted` | Numeric | `0` – `5` |

* **Target column (training only):** `Student_Performance` (`Excellent`, `Good`, `Average`, `Poor`).

### Feature Engineering & Preprocessing

Inside `src/features.py`, two domain features are created automatically inside the pipeline:

* `Coursework_Completion_Rate = ((Assignments_Completed / 10.0) + (Projects_Submitted / 5.0)) / 2.0`
* `Engagement_Score = (Attendance_Percentage / 100.0) * Class_Participation`

Numeric features are median-imputed and standardized (`StandardScaler`); categorical features are most-frequent-imputed and one-hot encoded (`OneHotEncoder(handle_unknown="ignore")`).

---

## 4. Setup

### Prerequisites

* Git
* Docker

### Step 1: Clone the Repository

```bash
git clone [https://github.com/AngeloE28/AINT602_A2_MLOps_Pipeline.git](https://github.com/AngeloE28/AINT602_A2_MLOps_Pipeline.git)
cd AINT602_A2_MLOps_Pipeline

```

### Step 2: Obtain and Place the Dataset

The raw dataset is tracked with DVC and excluded from Git. Before building the Docker image, place the CSV using one of the options below:

* **Option A - Manual download *(Currently Only Option Available)***: Download `Student_Performance_Teaching_Effectiveness.csv` from [Kaggle](https://www.kaggle.com/datasets/zara2099/student-performance-and-teaching-dataset/data) and save it to:
```text
data/raw/Student_Performance_Teaching_Effectiveness.csv

```


* **Option B - Pull via DVC** *(DVC remote was not setup for this project, only local DVC)*:
```bash
dvc pull

```



### Step 3: Build the Docker Image

```bash
docker build -t student-perf-mlops .

```

---

## 5. Running the Pipeline

All pipeline stages can be executed with single-line Docker commands from the project root.

### Step 1: Run Automated Quality Checks (`pytest`)

Verifies feature engineering output, end-to-end pipeline fitting, and saved model loading/prediction.

```bash
docker run --rm student-perf-mlops

```

### Step 2: Validate Data

Check the raw training dataset (includes target column validation):

```bash
docker run --rm student-perf-mlops python -m src.validate

```

Check a new dataset for inference (skips the `Student_Performance` target check):

```bash
docker run --rm student-perf-mlops python -m src.validate --input data/raw/Student_Performance_Teaching_Effectiveness.csv --inference

```

### Step 3: Train, Compare, and Export the Winning Model

Runs 10-fold CV across Dummy, Decision Tree, and Random Forest models, logs parameters/metrics/artifacts to MLflow, evaluates the winner on the 20% holdout test set, and saves the winning `.joblib` pipeline model.

```bash
docker run --rm student-perf-mlops python -m src.train

```

### Step 4: Generate Batch Predictions

Validates the input CSV (`is_training=False`), loads the saved `.joblib` pipeline, and prints predictions and class counts to the terminal:

```bash
docker run --rm student-perf-mlops python -m src.predict --input data/raw/Student_Performance_Teaching_Effectiveness.csv

```

*(Optional)* To also save predictions to a CSV file inside the container, pass `--output <path>`:

```bash
docker run --rm student-perf-mlops python -m src.predict --input data/raw/Student_Performance_Teaching_Effectiveness.csv --output data/predictions.csv

```

---

## 6. Modifying Experiments

All experiment parameters live in `configs/baseline.yaml`-no Python edits are required to tune models or splits:

1. Edit hyperparameters (e.g., `max_depth`, `n_estimators`, `min_samples_leaf`), `cv_folds`, or `output.model_path` in `configs/baseline.yaml`.
2. Rebuild the image:
```bash
docker build -t student-perf-mlops .

```


3. Re-run training and tests:
```bash
docker run --rm student-perf-mlops python -m src.train
docker run --rm student-perf-mlops

```