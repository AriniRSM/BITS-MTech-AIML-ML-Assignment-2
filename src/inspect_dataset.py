
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "default of credit card clients.xls"

# Use the second Excel row as the actual column header
df = pd.read_excel(DATA_PATH, header=1, engine="calamine")

print("DATASET OVERVIEW")
print("-" * 50)

print("Shape:", df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nTarget distribution:")
target = "default payment next month"
print(df[target].value_counts())
print("\nTarget distribution (%):")
print(df[target].value_counts(normalize=True) * 100)

print("\nFirst five records:")
print(df.head())