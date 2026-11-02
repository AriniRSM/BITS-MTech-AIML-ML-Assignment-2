
import pandas as pd
from pathlib import Path

from src.preprocessing import (
    separate_features_target,
    split_dataset,
    TARGET_COLUMN
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "default of credit card clients.xls"

# Load the original dataset
df = pd.read_excel(DATA_PATH, header=1)

# Separate features and target
X, y = separate_features_target(df)

# Recreate the same stratified split used for model training
X_train, X_test, y_train, y_test = split_dataset(
    X, y, test_size=0.20, random_state=42
)

# Combine test features with actual target labels
test_data = X_test.copy()
test_data[TARGET_COLUMN] = y_test.values

# Save labelled test data
test_data.to_csv(PROJECT_ROOT / "test_data.csv", index=False)

print("Test CSV created successfully!")
print("Shape:", test_data.shape)
print("Columns:", test_data.columns.tolist())
print("Target distribution:")
print(test_data[TARGET_COLUMN].value_counts())
