"""
Incarcarea datelor si separarea train/val/test.

Setul de test (CB_OUALD_test.csv) NU are etichete reale (sunt placeholder).
De aceea taiem 20% din train ca validare.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET_CLF = "final_result"
TARGET_REG = "final_coursework_score"

# Coloane cu rol special — nu sunt features predictive
ID_COLS = []  # nu apare id_student in CSV-ul real
TARGET_COLS = [TARGET_CLF, TARGET_REG]


def load_csv(path):
    """Citeste un CSV, tratand spatiile goale ca NaN."""
    return pd.read_csv(path, na_values=["", " ", "?", "NA", "N/A", "nan"])


def split_xy(df):
    """Separa features (X) de tinte (y_clf, y_reg)."""
    cols_drop = [c for c in TARGET_COLS + ID_COLS if c in df.columns]
    X = df.drop(columns=cols_drop)
    y_clf = df[TARGET_CLF] if TARGET_CLF in df.columns else None
    y_reg = df[TARGET_REG] if TARGET_REG in df.columns else None
    return X, y_clf, y_reg


def load_train_val_test(train_path, test_path, val_size=0.2, seed=42):
    """Incarca train si test; taie val_size din train pentru validare."""
    train_full = load_csv(train_path)
    test_df = load_csv(test_path) if test_path else None

    train_df, val_df = train_test_split(train_full, test_size=val_size, random_state=seed)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    return train_df, val_df, test_df


def get_feature_types(X):
    """Detecteaza coloane numerice vs. categoriale dupa dtype."""
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()
    return num_cols, cat_cols