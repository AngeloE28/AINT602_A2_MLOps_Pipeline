from pathlib import Path
import joblib
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from src.features import add_student_work_engagement_features, build_model_pipeline

DATA_PATH = Path("data/raw/Student_Performance_Teaching_Effectiveness.csv")
MODEL_PATH = Path("models/student_performance_pipeline.joblib")
TARGET_COL = "Student_Performance"

@pytest.fixture
def sample_data():
    df = pd.read_csv(DATA_PATH).head(20)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y

def test_feature_engineering_adds_columns(sample_data):
    X, y = sample_data
    transformed = add_student_work_engagement_features(X)
    assert transformed.shape[0] == X.shape[0]
    assert transformed.shape[1] > X.shape[1], "Expected engineered features to be added."

def test_pipeline_fit_and_predict(sample_data):
    X, y = sample_data
    pipe = build_model_pipeline(DecisionTreeClassifier(max_depth=3, random_state=42))
    pipe.fit(X, y)
    preds = pipe.predict(X)

    assert len(preds) == len(X)
    assert set(preds).issubset(set(y.unique()))

def test_saved_model_loads_and_predicts(sample_data):
    assert MODEL_PATH.exists(), f"Saved model missing at {MODEL_PATH}"
    X, y = sample_data
    pipe = joblib.load(MODEL_PATH)
    preds = pipe.predict(X.head(5))

    assert len(preds) == 5
    assert pd.Series(preds).notna().all()