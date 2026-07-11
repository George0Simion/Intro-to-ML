"""
Implementari manuale, asa cum au fost facute la laborator.

  - DecisionTree (ID3) + RandomForest (lab Arbori de Decizie)
  - LinearRegression + RidgeRegression (lab Regresie Liniara)
  - extract_polynomial_features (lab Regresie Liniara)
"""

import numpy as np
from copy import deepcopy
from collections import Counter


# ============================================================================
# ARBORI DE DECIZIE + RANDOM FOREST
# ============================================================================
class DecisionTreeNode:
    def __init__(self, feature=None, children=None, label=None):
        self.split_feature = feature
        self.children = children if (children is not None and feature is not None) else {}
        self.label = label
        self.depth = 1
        self.num_samples = 0


class DecisionTree:
    """Arbore ID3 — selecteaza atributul cu cel mai mare information gain."""

    def __init__(self, max_depth=np.inf, min_samples_per_node=1):
        self._root = None
        self._max_depth = max_depth
        self._min_samples_per_node = min_samples_per_node

    @staticmethod
    def _entropy(y):
        probs = y.value_counts(normalize=True)
        return -np.sum(probs * np.log2(probs + 1e-12))

    @staticmethod
    def _info_gain(X, y, feature):
        H = DecisionTree._entropy(y)
        N = len(y)
        H_after = 0.0
        for v in X[feature].unique():
            mask = X[feature] == v
            if mask.sum() > 0:
                H_after += (mask.sum() / N) * DecisionTree._entropy(y[mask])
        return H - H_after

    def _build(self, X, y, features, depth):
        node = DecisionTreeNode(label=y.mode().iloc[0])
        node.depth = depth
        node.num_samples = len(y)

        if (not features
                or depth >= self._max_depth
                or len(y) < self._min_samples_per_node
                or y.nunique() == 1):
            return node

        # alege atributul cu cel mai mare info gain
        best = max(features, key=lambda f: DecisionTree._info_gain(X, y, f))
        node.split_feature = best
        remaining = [f for f in features if f != best]

        for v in X[best].unique():
            mask = X[best] == v
            if mask.sum() == 0:
                child = DecisionTreeNode(label=y.mode().iloc[0])
                child.depth = depth + 1
                node.children[v] = child
            else:
                node.children[v] = self._build(X[mask], y[mask], remaining, depth + 1)
        return node

    def fit(self, X, y):
        self._root = self._build(X, y, X.columns.tolist(), depth=0)

    def _predict_one(self, x):
        node = self._root
        while node.split_feature is not None:
            v = x.get(node.split_feature)
            if v in node.children:
                node = node.children[v]
            else:
                break
        return node.label

    def predict(self, X):
        return np.array([self._predict_one(row) for _, row in X.iterrows()])


class RandomForest:
    """Padure de arbori de decizie cu bagging (lab)."""

    def __init__(self, n_estimators=20, max_depth=5, min_samples_per_node=2,
                 sample_ratio=0.7, feature_ratio=0.75, random_state=42):
        self._trees = []
        self._tree_features = []
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_per_node = min_samples_per_node
        self.sample_ratio = sample_ratio
        self.feature_ratio = feature_ratio
        self.random_state = random_state

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        n_feat = max(1, int(self.feature_ratio * X.shape[1]))
        n_samp = max(1, int(self.sample_ratio * X.shape[0]))
        all_features = X.columns.tolist()

        for _ in range(self.n_estimators):
            idx = rng.choice(X.shape[0], size=n_samp, replace=False)
            feats = list(rng.choice(all_features, size=n_feat, replace=False))
            X_sub = X.iloc[idx][feats].reset_index(drop=True)
            y_sub = y.iloc[idx].reset_index(drop=True)
            t = DecisionTree(max_depth=self.max_depth,
                             min_samples_per_node=self.min_samples_per_node)
            t.fit(X_sub, y_sub)
            self._trees.append(t)
            self._tree_features.append(feats)

    def predict(self, X):
        preds = np.array([t.predict(X[feats])
                          for t, feats in zip(self._trees, self._tree_features)]).T
        return np.array([Counter(row).most_common(1)[0][0] for row in preds])


# ============================================================================
# REGRESIE LINIARA SI RIDGE
# ============================================================================
class LinearRegressionLab:
    """Regresie liniara prin pseudoinversa: w = pinv(X) @ t."""

    def fit(self, X, t):
        self.w = np.linalg.pinv(X) @ t

    def predict(self, X):
        return X @ self.w


class RidgeRegressionLab:
    """Regresie liniara cu regularizare L2: w = (X.T X + alpha I)^-1 X.T t."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, t):
        D = X.shape[1]
        self.w = np.linalg.solve(X.T @ X + self.alpha * np.eye(D), X.T @ t)

    def predict(self, X):
        return X @ self.w


def extract_polynomial_features(X, M):
    """Pentru fiecare coloana x: 1, x, x^2, ..., x^M (din lab)."""
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    parts = [np.ones((X.shape[0], 1))]
    for m in range(1, M + 1):
        parts.append(X ** m)
    return np.hstack(parts)


def add_bias(X):
    """Adauga o coloana de 1 la inceputul matricei (termen liber)."""
    return np.hstack([np.ones((X.shape[0], 1)), X])


# ============================================================================
# DISCRETIZARE PT. ID3 MANUAL (ID3 lucreaza doar pe atribute discrete)
# ============================================================================
def discretize(X_train_raw, X_other_raw, num_cols, cat_cols, n_bins=4):
    """
    Discretizeaza numericele in quantile (4 bin-uri) si imputeaza valorile lipsa.
    Bin-urile sunt invatate pe TRAIN si aplicate identic pe celelalte seturi.
    """
    def _impute(df, num_cols, cat_cols):
        df = df.copy()
        for c in num_cols:
            df[c] = df[c].fillna(df[c].median())
        for c in cat_cols:
            df[c] = df[c].fillna(df[c].mode().iloc[0])
        return df

    train = _impute(X_train_raw, num_cols, cat_cols)
    other = _impute(X_other_raw, num_cols, cat_cols)

    bins = {}
    for c in num_cols:
        try:
            _, edges = pd.qcut(train[c], q=n_bins, retbins=True, duplicates="drop")
            edges[0] -= 1e-9
            edges[-1] += 1e-9
            bins[c] = edges
        except Exception:
            bins[c] = None

    def _apply_bins(df):
        df = df.copy()
        for c in num_cols:
            if bins[c] is not None:
                df[c] = pd.cut(df[c], bins=bins[c],
                               labels=[f"q{i}" for i in range(len(bins[c]) - 1)],
                               include_lowest=True).astype(object).fillna("q0")
            else:
                df[c] = df[c].astype(str)
        for c in cat_cols:
            df[c] = df[c].astype(object)
        return df

    return _apply_bins(train), _apply_bins(other)


# importat aici pentru ca discretize() foloseste pd.qcut/pd.cut
import pandas as pd  # noqa: E402