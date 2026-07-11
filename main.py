"""
Tema 1 — Introducere in Machine Learning (Seria CB) - OUALD

Punct de intrare. Orchestreaza:
  3.1 EDA (eda.py)
  3.2 Preprocesare (preprocess.py)
  3.3 Clasificare + Regresie (classification.py, regression.py)

Rulare:
    python3 main.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # ca sa salvam graficele fara display

from data import (load_train_val_test, split_xy, get_feature_types,
                   TARGET_CLF, TARGET_REG)
from eda import run_eda
from preprocess import Preprocessor, report_missing, report_outliers
from classification import run_classification
from regression import run_regression
from models import add_bias

np.random.seed(42)

TRAIN_PATH = "CB_OUALD_train.csv"
TEST_PATH = "CB_private_test.csv"


def main():
    # ===== 1. INCARCARE =====
    print("=" * 70)
    print(" Tema 1 — OUALD")
    print("=" * 70)

    train_df, val_df, test_df = load_train_val_test(TRAIN_PATH, TEST_PATH, val_size=0.2, seed=42)
    print(f"\n[1] Date: train={train_df.shape}, val={val_df.shape}, "
          f"test={test_df.shape if test_df is not None else 'N/A'}")

    # ===== 2. EDA (3.1) =====
    run_eda(train_df)

    # ===== 3. PREPROCESARE (3.2) =====
    print("\n" + "=" * 70)
    print(" 3.2 — PREPROCESARE")
    print("=" * 70)

    X_train_raw, y_clf_train, y_reg_train = split_xy(train_df)
    X_val_raw, y_clf_val, y_reg_val = split_xy(val_df)
    X_test_raw = (test_df.drop(columns=[c for c in [TARGET_CLF, TARGET_REG]
                                          if c in test_df.columns])
                  if test_df is not None else None)

    num_cols, cat_cols = get_feature_types(X_train_raw)
    print(f"\n[2] Numerice ({len(num_cols)}): {num_cols}")
    print(f"    Categoriale ({len(cat_cols)}): {cat_cols}")

    miss = report_missing(X_train_raw)
    print(f"\n[3] Valori lipsa per coloana (afectate):")
    print(miss.to_string() if len(miss) else "  (niciuna)")

    print("\n[4] Outliere (IQR k=1.5) per coloana numerica:")
    print(report_outliers(X_train_raw, num_cols).to_string(index=False))

    print("\n[5] Aplicare pipeline (mediana + IQR + standardizare + one-hot)")
    preproc = Preprocessor(num_cols=num_cols, cat_cols=cat_cols, iqr_k=1.5,
                           num_impute="median", standardize=True, remove_outliers=True)
    X_train = preproc.fit_transform(X_train_raw)
    X_val = preproc.transform(X_val_raw)
    print(f"    X_train: {X_train.shape}, X_val: {X_val.shape}")
    print(f"    features finale: {len(preproc.feature_names_)}")

    # ===== 4. CLASIFICARE (3.3) =====
    # eliminam randurile fara tinta de clasificare
    mask = y_clf_train.notna()
    mask_v = y_clf_val.notna()
    sk_rf, le = run_classification(
        X_train_proc=X_train[mask.values],
        X_val_proc=X_val[mask_v.values],
        X_train_raw_clf=X_train_raw[mask.values].reset_index(drop=True),
        X_val_raw_clf=X_val_raw[mask_v.values].reset_index(drop=True),
        y_clf_train=y_clf_train[mask].reset_index(drop=True),
        y_clf_val=y_clf_val[mask_v].reset_index(drop=True),
        num_cols=num_cols,
        cat_cols=cat_cols,
    )

    # ===== 5. REGRESIE (3.3) =====
    mask = y_reg_train.notna()
    mask_v = y_reg_val.notna()
    best_ridge, best_alpha = run_regression(
        X_train_proc=X_train[mask.values],
        X_val_proc=X_val[mask_v.values],
        y_reg_train=y_reg_train[mask].values,
        y_reg_val=y_reg_val[mask_v].values,
        num_cols=num_cols,
    )

    # ===== 6. SUBMISIE =====
    print("\n" + "=" * 70)
    print(" SUBMISIE PE TEST")
    print("=" * 70)
    if test_df is not None:
        X_test = preproc.transform(X_test_raw)
 
        # id = indexul randului din test.csv (0..N-1)
        ids = np.arange(len(test_df))
 
        # ----- submisie clasificare -----
        pred_clf = le.inverse_transform(sk_rf.predict(X_test))
        sub_clf = pd.DataFrame({"id": ids, "prediction": pred_clf})
        sub_clf.to_csv("submission_classification.csv", index=False)
        print(f"  Salvat submission_classification.csv ({sub_clf.shape})")
        print(sub_clf.head().to_string(index=False))
 
        # ----- submisie regresie -----
        pred_reg = best_ridge.predict(add_bias(X_test))
        sub_reg = pd.DataFrame({"id": ids, "prediction": pred_reg})
        sub_reg.to_csv("submission_regression.csv", index=False)
        print(f"\n  Salvat submission_regression.csv ({sub_reg.shape})")
        print(sub_reg.head().to_string(index=False))
    else:
        print("  Test lipseste — sarim peste submisii.")

    print("\n" + "=" * 70)
    print(" GATA")
    print("=" * 70)


if __name__ == "__main__":
    main()