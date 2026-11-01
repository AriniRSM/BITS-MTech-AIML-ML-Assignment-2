from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

# Target variable
TARGET_COLUMN = 'default payment next month'

# Identifier and EDA-only feature to exclude
DROP_COLUMNS = ['ID', 'AGE_GROUP']

# Categorical input features
CATEGORICAL_FEATURES = [
    'SEX',
    'EDUCATION',
    'MARRIAGE'
]

def separate_features_target(df):
    """
    Separate input features and target variable.
    Removes customer ID and AGE_GROUP from model inputs.
    """

    X = df.drop(
        columns=DROP_COLUMNS + [TARGET_COLUMN],
        errors='raise'
    )

    y = df[TARGET_COLUMN]

    return X, y

def build_preprocessor(feature_columns, model_type):
    """
    Build a preprocessing pipeline based on the model type.
    Categorical features are one-hot encoded.
    Numerical features are scaled for models sensitive
    to differences in feature magnitudes.
    """

    model_type = model_type.lower()

    numerical_features = [
        column for column in feature_columns
        if column not in CATEGORICAL_FEATURES
    ]

    if model_type in ['logistic_regression', 'knn', 'svm']:
        numerical_transformer = StandardScaler()
    else:
        numerical_transformer = 'passthrough'

    preprocessor = ColumnTransformer(
        transformers=[
            (
                'numerical',
                numerical_transformer,
                numerical_features
            ),
            (
                'categorical',
                OneHotEncoder(
                    handle_unknown='ignore',
                    sparse_output=False
                ),
                CATEGORICAL_FEATURES
            )
        ],
        remainder='drop'
    )

    return preprocessor

def split_dataset(X, y, test_size=0.20, random_state=42):
    """
    Split the dataset into training and testing sets.
    Stratification preserves the target class proportions.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test

def separate_features_target(df):
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found.")

    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])
    X = X.drop(columns=DROP_COLUMNS, errors='ignore')

    return X, y