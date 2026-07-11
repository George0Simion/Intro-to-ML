"""
Cerinta 3.2 — Preprocesarea datelor.

Pipeline:
  1. detectie + inlocuire outliere (IQR) -> NaN
  2. imputare numerice (mediana) si categoriale (modul)
  3. standardizare numerice (StandardScaler)
  4. one-hot pentru categoriale

Toti parametrii sunt invatati pe TRAIN si aplicati pe val/test.
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def detect_outliers_iqr(s, k=1.5):
    """Masca booleana: True acolo unde valoarea e in afara intervalului IQR."""
    Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
    IQR = Q3 - Q1
    return (s < Q1 - k * IQR) | (s > Q3 + k * IQR)


class Preprocessor:
    """Aplica imputare + outliere -> NaN + standardizare + one-hot."""

    def __init__(self, num_cols, cat_cols, iqr_k=1.5, num_impute="median", cat_impute="most_frequent", standardize=True, remove_outliers=True):
        self.num_cols = list(num_cols)
        self.cat_cols = list(cat_cols)
        self.iqr_k = iqr_k
        self.num_impute = num_impute
        self.cat_impute = cat_impute
        self.standardize = standardize
        self.remove_outliers = remove_outliers
        self._iqr_bounds = {}

    def _mask_outliers(self, X_num, fit):
        """Inlocuieste outlierele cu NaN — vor fi imputate de SimpleImputer."""
        X_num = X_num.copy()
        if not self.remove_outliers:
            return X_num
        for c in X_num.columns:
            if fit:
                Q1, Q3 = X_num[c].quantile([0.25, 0.75])
                IQR = Q3 - Q1
                self._iqr_bounds[c] = (Q1 - self.iqr_k * IQR, Q3 + self.iqr_k * IQR)
            lo, hi = self._iqr_bounds[c]
            X_num.loc[(X_num[c] < lo) | (X_num[c] > hi), c] = np.nan
        return X_num

    def fit(self, X):
        # numeric
        X_num = self._mask_outliers(X[self.num_cols], fit=True)
        self.num_imputer = SimpleImputer(strategy=self.num_impute)
        X_num_imp = self.num_imputer.fit_transform(X_num)

        self.scaler = StandardScaler() if self.standardize else None
        if self.scaler is not None:
            self.scaler.fit(X_num_imp)

        # categorial
        X_cat = X[self.cat_cols].astype(object)
        self.cat_imputer = SimpleImputer(strategy=self.cat_impute)
        X_cat_imp = self.cat_imputer.fit_transform(X_cat)
        self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.ohe.fit(X_cat_imp)

        self.feature_names_ = (list(self.num_cols) +
                               list(self.ohe.get_feature_names_out(self.cat_cols)))
        return self

    def transform(self, X):
        X_num = self._mask_outliers(X[self.num_cols], fit=False)
        X_num_imp = self.num_imputer.transform(X_num)
        X_num_out = self.scaler.transform(X_num_imp) if self.scaler is not None else X_num_imp

        X_cat = X[self.cat_cols].astype(object)
        X_cat_imp = self.cat_imputer.transform(X_cat)
        X_cat_out = self.ohe.transform(X_cat_imp)

        return np.hstack([X_num_out, X_cat_out])

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


def report_missing(df):
    """Listeaza coloanele cu valori lipsa si numarul lor."""
    miss = df.isna().sum()
    return miss[miss > 0].sort_values(ascending=False)


def report_outliers(df, num_cols, k=1.5):
    """Numar de outliere per coloana numerica."""
    rows = []
    for c in num_cols:
        n = detect_outliers_iqr(df[c], k=k).sum()
        rows.append({"atribut": c, "n_outliere": n,
                     "procent": round(100 * n / len(df), 2)})
    return pd.DataFrame(rows).sort_values("n_outliere", ascending=False)