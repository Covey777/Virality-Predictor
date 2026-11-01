"""
preprocessing.py
----------------
Loads the dataset, selects ONLY pre-publish features (no leakage), splits into train/test,
and builds the preprocessing step that turns text columns into numbers.

Key ideas for your viva:
  * One-Hot Encoding: 'category' (9 values) and 'post_day' (7 values) are text. Models need
    numbers, so each value becomes its own 0/1 column (e.g. category_Comedy = 1).
  * Random Forests do NOT need feature scaling (trees split on thresholds), but Logistic /
    Linear Regression (our baselines) do, so we allow scale=True for them.
  * The preprocessor lives INSIDE a scikit-learn Pipeline together with the model. That way the
    exact same transformation is applied at training time and in the Streamlit app.
  * Stratified split: keeps Low/Medium/High in equal proportion in both train and test sets.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config as C


def load_data() -> pd.DataFrame:
    if not C.DATASET_PATH.exists():
        raise FileNotFoundError("Dataset not found. Run:  python -m src.generate_dataset")
    return pd.read_csv(C.DATASET_PATH)


def get_train_test(df: pd.DataFrame):
    """Returns X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test."""
    X = df[C.PRE_PUBLISH_FEATURES]            # <- leakage-safe inputs only
    y_class = df["performance_label"]
    y_reg = df["engagement_rate"]
    return train_test_split(
        X, y_class, y_reg,
        test_size=C.TEST_SIZE,
        stratify=y_class,
        random_state=C.RANDOM_SEED,
    )


def build_preprocessor(scale: bool = False) -> ColumnTransformer:
    numeric_step = StandardScaler() if scale else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), C.CATEGORICAL_FEATURES),
            ("num", numeric_step, C.NUMERIC_FEATURES),
        ]
    )