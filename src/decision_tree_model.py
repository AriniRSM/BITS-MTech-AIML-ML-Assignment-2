from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

from src.preprocessing import build_preprocessor

def build_decision_tree_model(feature_columns):
    preprocessor = build_preprocessor(
        feature_columns,
        model_type='decision_tree'
    )

    model = DecisionTreeClassifier(
        random_state=42
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    return pipeline