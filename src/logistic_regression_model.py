from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.preprocessing import build_preprocessor

def build_logistic_regression_model(feature_columns):
    preprocessor = build_preprocessor(
        feature_columns,
        model_type='logistic_regression'
    )

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    return pipeline