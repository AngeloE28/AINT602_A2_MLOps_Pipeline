from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.tree import DecisionTreeClassifier

from src.features import build_model_pipeline


def load_config(path: str = "configs/baseline.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_model_candidates(config: dict) -> dict:
    RANDOM_STATE = config["split"]["random_state"]
    models = config["models"]

    return {
        models["dummy"]["name"]: build_model_pipeline(
            DummyClassifier(strategy=models["dummy"]["strategy"])
        ),
        models["decision_tree"]["name"]: build_model_pipeline(
            DecisionTreeClassifier(
                class_weight=models["decision_tree"]["class_weight"],
                random_state=RANDOM_STATE,
                max_depth=models["decision_tree"]["max_depth"],
                min_samples_leaf=models["decision_tree"]["min_samples_leaf"],
            )
        ),
        models["random_forest"]["name"]: build_model_pipeline(
            RandomForestClassifier(
                n_estimators=models["random_forest"]["n_estimators"],
                class_weight=models["random_forest"]["class_weight"],
                random_state=RANDOM_STATE,
                max_depth=models["random_forest"]["max_depth"],
                min_samples_leaf=models["random_forest"]["min_samples_leaf"],
                n_jobs=-1,
            )
        ),
    }


def main(config_path: str = "configs/baseline.yaml") -> None:
    config = load_config(config_path)

    # Load data & train split
    df = pd.read_csv(config["dataset"]["path"])
    target = config["dataset"]["target"]
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["split"]["test_size"],
        random_state=config["split"]["random_state"],
        stratify=y,
    )

    # Set up MLflow (SQLite in project root)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("student_performance_classification")

    cv = KFold(
        n_splits=config["split"]["cv_folds"],
        shuffle=True,
        random_state=config["split"]["random_state"],
    )

    candidates = build_model_candidates(config)
    best_name = None
    best_pipe = None
    best_f1 = -1.0

    # Run CV & log each candidate
    for name, pipe in candidates.items():
        with mlflow.start_run(run_name=name):
            mlflow.set_tags({
                "model_version": f"{name}_v1",
                "preprocessing_version": config["preprocessing"]["version"],
                "dataset_version": config["dataset"]["name"],
            })
            mlflow.log_artifact(config_path)

            estimator = pipe.named_steps["model"]
            params = {
                "model_name": name,
                "cv_folds": config["split"]["cv_folds"],
                "random_state": config["split"]["random_state"],
            }
            for param in ["strategy", "class_weight", "n_estimators", "max_depth", "min_samples_leaf"]:
                if hasattr(estimator, param):
                    params[param] = getattr(estimator, param)
            mlflow.log_params(params)

            scores = cross_validate(
                pipe, X_train, y_train, cv=cv, scoring=["f1_macro", "accuracy"], n_jobs=-1
            )

            f1_mean = scores["test_f1_macro"].mean()
            f1_std = scores["test_f1_macro"].std()
            acc_mean = scores["test_accuracy"].mean()
            acc_std = scores["test_accuracy"].std()

            mlflow.log_metrics({
                "cv_f1_macro_mean": f1_mean,
                "cv_f1_macro_std": f1_std,
                "cv_accuracy_mean": acc_mean,
                "cv_accuracy_std": acc_std,
            })

            pipe.fit(X_train, y_train)
            mlflow.sklearn.log_model(
                pipe,
                name="model",
                serialization_format="cloudpickle",
            )

            print(f"{name:25s} | CV Macro F1: {f1_mean:.4f} (+/- {f1_std:.4f}) | Acc: {acc_mean:.4f}")

            if f1_mean > best_f1:
                best_f1 = f1_mean
                best_name = name
                best_pipe = pipe

    # Evaluate winning model on untouched test set & save reusable pipeline
    test_preds = best_pipe.predict(X_test)
    test_f1 = f1_score(y_test, test_preds, average="macro")
    test_acc = accuracy_score(y_test, test_preds)

    output_path = Path(config["output"]["model_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipe, output_path)

    print("\n--- Final Winning Model ---")
    print(f"Winner:        {best_name}")
    print(f"Test Macro F1: {test_f1:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Saved to:      {output_path}")


if __name__ == "__main__":
    main()