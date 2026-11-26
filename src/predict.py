from pathlib import Path
import argparse
import joblib
import pandas as pd

from src.validate import REQUIRED_COLUMNS, validate_data


def predict(
    input_path: str,
    output_path: str,
    model_path: str = "models/student_performance_model.joblib",
) -> pd.DataFrame:
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model not found at {model_file}. Run `python -m src.train` first.")

    student_performance_model = joblib.load(model_file)
    df = validate_data(pd.read_csv(input_path), is_training=False)
    X_new = df[REQUIRED_COLUMNS]

    result = df.copy()
    result["Predicted_Student_Performance"] = student_performance_model.predict(X_new)

    print(result[["Predicted_Student_Performance"]].head(10).to_string())
    print("\nPrediction counts:")
    print(result["Predicted_Student_Performance"].value_counts().to_string())

    if output_path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(out_file, index=False)
        print(f"\nSaved {len(result)} predictions -> {out_file}")

    return result


def main() -> None:
    # Argparser for CLI commands
    parser = argparse.ArgumentParser(description="Generate student performance predictions.")
    parser.add_argument("--input", required=True, help="Path to input CSV with raw features.")
    parser.add_argument("--output", default=None, help="Optional path to save predictions CSV.")
    parser.add_argument(
        "--model",
        default="models/student_performance_model.joblib",
        help="Path to saved joblib pipeline.",
    )
    args = parser.parse_args()
    predict(args.input, args.output, args.model)


if __name__ == "__main__":
    main()