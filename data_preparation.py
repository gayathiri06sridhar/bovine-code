import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold, StratifiedGroupKFold, StratifiedShuffleSplit
from scipy.stats import pointbiserialr

df = pd.read_csv("cow_milk_mastitis_dataset.csv")

cow_counts = df["Cow_ID"].value_counts()
n_unique_cows = df["Cow_ID"].nunique()
n_rows = len(df)
has_repeated_cows = n_unique_cows < n_rows

feature_cols = [
    "Day", "Milk_Temperature", "Milk_pH", "Milk_Conductivity",
    "Somatic_Cell_Count", "Milk_Yield", "Clotting"
]
target_col = "class1"

X = df[feature_cols]
y = df[target_col]
groups = df["Cow_ID"]

leakage_flags = []
for col in feature_cols:
    r, p = pointbiserialr(y, X[col])
    if abs(r) > 0.70:
        leakage_flags.append((col, abs(r)))

RANDOM_STATE = 42

if has_repeated_cows:
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    cv_strategy = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
else:
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE)
    train_idx, test_idx = next(sss.split(X, y))
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
groups_train = groups.iloc[train_idx]
groups_test = groups.iloc[test_idx]
