from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from src.preprocessing import build_preprocessor

def build_knn_model(feature_columns):
    preprocessor = build_preprocessor(
        feature_columns,
        model_type="knn"
    )

    model = KNeighborsClassifier(n_neighbors=5)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    return pipeline