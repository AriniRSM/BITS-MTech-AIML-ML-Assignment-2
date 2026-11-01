import pandas as pd
import joblib
from pathlib import Path

from src.preprocessing import (
    separate_features_target,
    split_dataset
)

from src.evaluation import evaluate_model

from src.logistic_regression_model import build_logistic_regression_model
from src.decision_tree_model import build_decision_tree_model
from src.knn_model import build_knn_model
from src.naive_bayes_model import build_naive_bayes_model
from src.svm_model import build_svc_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "default of credit card clients.xls"

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

def load_dataset():
    df = pd.read_excel(DATA_PATH, header=1)
    return df

def prepare_data(df):
    X, y = separate_features_target(df)

    X_train, X_test, y_train, y_test = split_dataset(
        X,
        y
    )

    return X_train, X_test, y_train, y_test

def get_models(feature_columns):
    models = {
        "Logistic Regression": build_logistic_regression_model(feature_columns),
        "Decision Tree": build_decision_tree_model(feature_columns),
        "KNN": build_knn_model(feature_columns),
        "Gaussian Naive Bayes": build_naive_bayes_model(feature_columns),
        "SVM": build_svc_model(feature_columns)
    }

    return models

def train_and_evaluate(models, X_train, X_test, y_train, y_test):
    results = {}

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")

        model.fit(X_train, y_train)

        metrics = evaluate_model(
            model,
            X_test,
            y_test
        )

        results[model_name] = metrics

        model_filename = model_name.lower().replace(" ", "_") + ".pkl"

        model_path = MODEL_DIR / model_filename

        joblib.dump(model, model_path)

        print(f"{model_name} completed.")
        print(metrics)

    return results

def main():
    print("Loading dataset...")
    df = load_dataset()

    print("Preparing training and test data...")
    X_train, X_test, y_train, y_test = prepare_data(df)

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    models = get_models(X_train.columns)

    results = train_and_evaluate(
        models,
        X_train,
        X_test,
        y_train,
        y_test
    )

    results_df = pd.DataFrame(results).T

    print("\nModel Comparison:")
    print(results_df)

    results_df.to_csv(
        PROJECT_ROOT / "model_comparison.csv",
        index=True
    )

    print("\nTraining and evaluation completed!")

if __name__ == "__main__":
    main()