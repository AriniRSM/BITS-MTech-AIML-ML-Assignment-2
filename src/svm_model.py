from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from src.preprocessing import build_preprocessor


def build_svc_model(feature_columns):
    preprocessor = build_preprocessor(
        feature_columns,
        model_type="svm",
    )

    svm = SVC(
        kernel="linear",
        random_state=42,
    )

    model = CalibratedClassifierCV(
        svm,
        ensemble=False,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    return pipeline