from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

from src.preprocessing import build_preprocessor

def build_naive_bayes_model(feature_columns):
    preprocessor = build_preprocessor(
        feature_columns,
        model_type="naive_bayes",
    )

    model = GaussianNB()

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    return pipeline