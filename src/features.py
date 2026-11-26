import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# store string in variables to prevent typos
Age = "Age"
Gender = "Gender"
Attendance_Percentage = "Attendance_Percentage"
Assignments_Completed = "Assignments_Completed"
Class_Participation = "Class_Participation"
Study_Hours_per_Week = "Study_Hours_per_Week"
Projects_Submitted = "Projects_Submitted"

# Two engineered features
Coursework_Completion_Rate="Coursework_Completion_Rate"
Engagement_Score="Engagement_Score"
NUMERIC_FEATURES = [
    Age, Attendance_Percentage, Assignments_Completed, 
    Class_Participation, Study_Hours_per_Week, Projects_Submitted,
    Coursework_Completion_Rate, Engagement_Score
]

CATEGORICAL_FEATURES = [
    Gender
]


def add_student_work_engagement_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create engineered coursework and engagement features."""
    df = df.copy()
    # Coursework completion based on an average of assignments and projects submitted
    df["Coursework_Completion_Rate"] = ((df[Assignments_Completed] / 10.0) + (df[Projects_Submitted] / 5.0)) / 2.0
    
    # Engagement Score
    df["Engagement_Score"] = ((df[Attendance_Percentage] / 100.0) * df[Class_Participation])
    return df

def build_model_pipeline(model) -> Pipeline:
    """Bundle feature engineering, preprocessing, and estimator into one reusable pipeline."""
    
    # Compatibility for older and newer scikit-learn versions.
    try:
        one_hot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        one_hot = OneHotEncoder(handle_unknown="ignore", sparse=False)
        
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", one_hot)
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop"
    )

    return Pipeline(
        steps=[
            ("feature_engineering", FunctionTransformer(add_student_work_engagement_features, validate=False)),
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )