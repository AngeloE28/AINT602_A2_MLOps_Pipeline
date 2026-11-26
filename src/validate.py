import argparse
import sys
from pathlib import Path
import pandas as pd

# Column constants (single source of truth)
AGE = "Age"
GENDER = "Gender"
ATTENDANCE_PERCENTAGE = "Attendance_Percentage"
ASSIGNMENTS_COMPLETED = "Assignments_Completed"
CLASS_PARTICIPATION = "Class_Participation"
STUDY_HOURS_PER_WEEK = "Study_Hours_per_Week"
PROJECTS_SUBMITTED = "Projects_Submitted"
TARGET_COL = "Student_Performance"

REQUIRED_COLUMNS = [
    AGE,
    GENDER,
    ATTENDANCE_PERCENTAGE,
    ASSIGNMENTS_COMPLETED,
    CLASS_PARTICIPATION,
    STUDY_HOURS_PER_WEEK,
    PROJECTS_SUBMITTED,
]

ALLOWED_GENDERS = {"Male", "Female", "Other"}
ALLOWED_STUDENT_PERFORMANCE = {"Excellent", "Good", "Average", "Poor"}

RANGE_CHECKS = {
    AGE: (18, 100),
    ATTENDANCE_PERCENTAGE: (0, 100),
    ASSIGNMENTS_COMPLETED: (0, 10),
    CLASS_PARTICIPATION: (0, 10),
    STUDY_HOURS_PER_WEEK: (0, 168),
    PROJECTS_SUBMITTED: (0, 5),
}


def validate_data(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    """Validate schema, nulls, duplicates, categories, and numeric ranges."""
    if df.empty:
        raise ValueError("Dataset is empty (0 rows).")

    expected_cols = REQUIRED_COLUMNS + ([TARGET_COL] if is_training else [])

    # Missing columns
    missing = set(expected_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Nulls & duplicates
    if df[expected_cols].isnull().any().any():
        raise ValueError("Null values detected in required columns.")
    if df.duplicated().any():
        df = df.drop_duplicates().copy()

    # Expected categories
    unexpected_genders = set(df[GENDER].unique()) - ALLOWED_GENDERS
    if unexpected_genders:
        raise ValueError(f"Unexpected Gender values: {unexpected_genders}")

    if is_training:
        unexpected_targets = set(df[TARGET_COL].unique()) - ALLOWED_STUDENT_PERFORMANCE
        if unexpected_targets:
            raise ValueError(f"Unexpected target categories detected: {unexpected_targets}")

    # Valid numeric ranges
    for col, (low, high) in RANGE_CHECKS.items():
        if not df[col].between(low, high).all():
            raise ValueError(f"Column '{col}' has values outside [{low}, {high}].")

    print(f"Validation passed ({len(df)} rows).")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate raw student performance CSV.")
    parser.add_argument(
        "--input",
        default="data/raw/Student_Performance_Teaching_Effectiveness.csv",
        help="Path to CSV file to validate.",
    )
    parser.add_argument(
        "--inference",
        action="store_true",
        help="Skip target column checks (use for new prediction data).",
    )
    args = parser.parse_args()

    file_path = Path(args.input)
    if not file_path.exists():
        print(f"Validation failed: file not found at {file_path}", file=sys.stderr)
        sys.exit(1)

    try:
        df = pd.read_csv(file_path)
        validate_data(df, is_training=not args.inference)
    except Exception as exc:
        print(f"Validation failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()