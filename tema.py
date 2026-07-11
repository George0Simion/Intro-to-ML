from __future__ import annotations
from copy import deepcopy
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import chi2_contingency

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge as SkRidge, Lasso as SkLasso
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              classification_report, confusion_matrix,
                              mean_absolute_error, mean_squared_error, r2_score)

np.random.seed(42)


TRAIN_PATH = "CB_OUALD_train.csv"
TEST_PATH = "CB_OUALD_test.csv"
PRIVATE_CLF_PATH = "CB_private_test.csv"
PRIVATE_REG_PATH = "CB_private_test.csv"

TARGET_CLF = "final_result"
TARGET_REG = "final_coursework_score"

SAMPLE_FOR_MANUAL = 3000  # ID3 manual e lent, esantion pentru ablatie

SECTION = lambda title: print(f"\n{'='*70}\n {title}\n{'='*70}")
SUBSECTION = lambda title: print(f"\n--- {title} ---")


# implementari lab
class DecisionTreeNode:
    def __init__(self, feature=None, label=None):
        self.split_feature = feature
        self.children = {}
        self.label = label
        self.depth = 0


class DecisionTree:
    # ID3 — selecteaza atributul cu cel mai mare information gain
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
        H_after = sum((mask.sum() / N) * DecisionTree._entropy(y[mask])
                      for v in X[feature].unique()
                      for mask in [X[feature] == v] if mask.sum() > 0)
        return H - H_after

    def _build(self, X, y, features, depth):
        node = DecisionTreeNode(label=y.mode().iloc[0])
        node.depth = depth
        if (not features or depth >= self._max_depth
                or len(y) < self._min_samples_per_node or y.nunique() == 1):
            return node
        best = max(features, key=lambda f: DecisionTree._info_gain(X, y, f))
        node.split_feature = best
        for v in X[best].unique():
            mask = X[best] == v
            if mask.sum() == 0:
                child = DecisionTreeNode(label=y.mode().iloc[0])
                child.depth = depth + 1
                node.children[v] = child
            else:
                node.children[v] = self._build(X[mask], y[mask],
                                                [f for f in features if f != best], depth + 1)
        return node

    def fit(self, X, y):
        self._root = self._build(X, y, X.columns.tolist(), 0)

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
    # Random forest with bagging
    def __init__(self, n_estimators=20, max_depth=5, min_samples_per_node=2, sample_ratio=0.7, feature_ratio=0.75, random_state=42):
        self._trees, self._tree_features = [], []
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
        for _ in range(self.n_estimators):
            idx = rng.choice(X.shape[0], size=n_samp, replace=False)
            feats = list(rng.choice(X.columns.tolist(), size=n_feat, replace=False))
            t = DecisionTree(max_depth=self.max_depth,
                             min_samples_per_node=self.min_samples_per_node)
            t.fit(X.iloc[idx][feats].reset_index(drop=True),
                  y.iloc[idx].reset_index(drop=True))
            self._trees.append(t); self._tree_features.append(feats)

    def predict(self, X):
        preds = np.array([t.predict(X[feats])
                          for t, feats in zip(self._trees, self._tree_features)]).T
        return np.array([Counter(row).most_common(1)[0][0] for row in preds])


class LinearRegressionLab:
    # simple linear regression
    def fit(self, X, t):
        self.w = np.linalg.pinv(X) @ t

    def predict(self, X):
        return X @ self.w


class RidgeRegressionLab:
    # l2 regularization linear regression - ridge
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, t):
        D = X.shape[1]
        self.w = np.linalg.solve(X.T @ X + self.alpha * np.eye(D), X.T @ t)

    def predict(self, X):
        return X @ self.w


def add_bias(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])


# stuff for preprocessing phase

def detect_outliers_iqr(s, k=1.5):
    Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
    IQR = Q3 - Q1
    return (s < Q1 - k * IQR) | (s > Q3 + k * IQR)


class Preprocessor:
    # a pipeline class which applies: outlier masking (IQR) -> imputation (mediana pentru numere, mod pentru categorii) -> standardization (z-score) pentru numere, one-hot encoding pentru categorii
    def __init__(self, num_cols, cat_cols, iqr_k=1.5,
                 num_impute="median", remove_outliers=True, standardize=True):
        self.num_cols, self.cat_cols = list(num_cols), list(cat_cols)
        self.iqr_k = iqr_k
        self.num_impute = num_impute
        self.remove_outliers = remove_outliers
        self.standardize = standardize
        self._iqr_bounds = {}

    def _mask_outliers(self, X_num, fit):
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
        X_num = self._mask_outliers(X[self.num_cols], fit=True)
        self.num_imputer = SimpleImputer(strategy=self.num_impute)
        X_num_imp = self.num_imputer.fit_transform(X_num)
        self.scaler = StandardScaler() if self.standardize else None
        if self.scaler is not None:
            self.scaler.fit(X_num_imp)
        X_cat = X[self.cat_cols].astype(object)
        self.cat_imputer = SimpleImputer(strategy="most_frequent")
        X_cat_imp = self.cat_imputer.fit_transform(X_cat)
        self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.ohe.fit(X_cat_imp)
        return self

    def transform(self, X):
        X_num = self._mask_outliers(X[self.num_cols], fit=False)
        X_num_imp = self.num_imputer.transform(X_num)
        X_num_out = self.scaler.transform(X_num_imp) if self.scaler else X_num_imp
        X_cat = X[self.cat_cols].astype(object)
        X_cat_imp = self.cat_imputer.transform(X_cat)
        X_cat_out = self.ohe.transform(X_cat_imp)
        return np.hstack([X_num_out, X_cat_out])

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def discretize(X_train, X_other, num_cols, cat_cols, n_bins=4):
    # manual id3 needs discrete -> discretize numeric features in quantiles (4 bins) + impute missing values (median for numeric, mode for categorical), learn bins on TRAIN and apply identically on OTHER sets
    def _impute(df):
        df = df.copy()
        for c in num_cols:
            df[c] = df[c].fillna(df[c].median())
        for c in cat_cols:
            df[c] = df[c].fillna(df[c].mode().iloc[0])
        return df

    train, other = _impute(X_train), _impute(X_other)
    bins = {}
    for c in num_cols:
        try:
            _, edges = pd.qcut(train[c], q=n_bins, retbins=True, duplicates="drop")
            edges[0] -= 1e-9; edges[-1] += 1e-9
            bins[c] = edges
        except Exception:
            bins[c] = None

    def _apply(df):
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

    return _apply(train), _apply(other)


# metrici

def eval_clf(y_true, y_pred, classes):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0, labels=classes),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0, labels=classes),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0, labels=classes),
    }


def reg_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {"MAE": mean_absolute_error(y_true, y_pred),
            "MSE": mse, "RMSE": np.sqrt(mse),
            "R2": r2_score(y_true, y_pred)}


def plot_confusion(cm, classes, title, out):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=30)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    ax.set_xlabel("Predictie"); ax.set_ylabel("Real"); ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im); plt.tight_layout(); plt.savefig(out, dpi=120); plt.close()


# 3.1 — data exploration (EDA)

def run_eda(train_df, num_cols, cat_cols):
    SECTION("1. Explorarea Datelor (EDA)")

    # 1.1 Analiza tipului de atribute și a plajei de valori a acestora
    SUBSECTION("1.1 Analiza tipului de atribute si a plajei de valori a acestora")
    
    # Subsetul de atribute numerice cerut in documentatie
    num_subset = ['studied_credits', 'mean_score_early', 'weighted_mean_score_early', 
                  'total_clicks_early', 'num_of_prev_attempts']
    # Filtram doar atributele care exista efectiv in dataset
    num_subset = [c for c in num_subset if c in train_df.columns]
    
    if num_subset:
        print("Atribute numerice (extras):")
        stats_num = train_df[num_subset].describe().T[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
        print(stats_num.round(2).to_string())

    print("\nAtribute categoriale:")
    cat_stats = pd.DataFrame([{
        "Atribut": c,
        "count": train_df[c].notna().sum(),
        "n_unique": train_df[c].nunique(dropna=True),
    } for c in cat_cols])
    print(cat_stats.to_string(index=False))

    # Boxplot pentru studied_credits
    if 'studied_credits' in train_df.columns:
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=train_df['studied_credits'].dropna(), color='skyblue')
        plt.title('Boxplot pentru studied_credits')
        plt.xlabel('Numar credite')
        plt.tight_layout()
        plt.savefig("eda_boxplot_studied_credits.png", dpi=120)
        plt.close()
        print("  [Grafic salvat: eda_boxplot_studied_credits.png]")

    # Distributii pentru region si highest_education
    for col in ['region', 'highest_education']:
        if col in train_df.columns:
            plt.figure(figsize=(10, 5))
            # order pentru a sorta barele descrescator
            sns.countplot(data=train_df, y=col, order=train_df[col].value_counts().index, palette='viridis', hue=col, legend=False)
            plt.title(f'Distributia pentru {col}')
            plt.xlabel('Numar studenti')
            plt.ylabel(col)
            plt.tight_layout()
            plt.savefig(f"eda_distributie_{col}.png", dpi=120)
            plt.close()
            print(f"  [Grafic salvat: eda_distributie_{col}.png]")

    # 1.2 Analiza echilibrului de clase
    SUBSECTION("1.2 Analiza echilibrului de clase")
    
    counts = train_df[TARGET_CLF].value_counts()
    pcts = train_df[TARGET_CLF].value_counts(normalize=True) * 100
    df_clase = pd.DataFrame({'Numar exemple': counts, 'Procent': pcts.round(1).astype(str) + '%'})
    df_clase.index.name = 'Clasa'
    
    print(df_clase.to_string())
    print(f"\n  Raport max/min: {counts.max() / counts.min():.2f}")

    plt.figure(figsize=(8, 5))
    sns.countplot(data=train_df, x=TARGET_CLF, order=counts.index, palette='magma', hue=TARGET_CLF, legend=False)
    plt.title('Echilibrul de clase pentru final_result')
    plt.xlabel('Rezultat Final')
    plt.ylabel('Numar de exemple')
    plt.tight_layout()
    plt.savefig("eda_class_balance.png", dpi=120)
    plt.close()
    print("  [Grafic salvat: eda_class_balance.png]")

    # 1.3 Analiza corelatiilor
    SUBSECTION("1.3 Analiza corelatiilor")
    
    # a) Numeric vs. Clasificare
    print("a) Numeric vs. Clasificare (studied_credits vs final_result)")
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=train_df, x=TARGET_CLF, y='studied_credits', palette='Set2', hue=TARGET_CLF, legend=False)
    plt.title('Numar de credite corelat cu rezultatul final')
    plt.xlabel('Rezultat final')
    plt.ylabel('studied_credits')
    plt.tight_layout()
    plt.savefig("eda_corr_num_clf.png", dpi=120)
    plt.close()
    print("  [Grafic salvat: eda_corr_num_clf.png]")

    # b) Categorial vs Clasificare
    print("\nb) Categorial vs Clasificare (gender vs final_result)")
    tabel_gen = pd.crosstab(train_df['gender'], train_df[TARGET_CLF], normalize='index').round(3) * 100
    print(tabel_gen.to_string())
    
    plt.figure(figsize=(8, 5))
    sns.countplot(data=train_df, x='gender', hue=TARGET_CLF, palette='magma')
    plt.title('Relatia dintre gender si rezultatul final')
    plt.xlabel('Sex (M / F)')
    plt.ylabel('Numar de studenti')
    plt.legend(title='Rezultat Final')
    plt.tight_layout()
    plt.savefig("eda_corr_cat_clf.png", dpi=120)
    plt.close()
    print("  [Grafic salvat: eda_corr_cat_clf.png]")

    # c) Numeric vs Regresie
    print("\nc) Numeric vs. Regresie (mean_score_early vs scor final)")
    plt.figure(figsize=(8, 5))
    valid_data = train_df[['mean_score_early', TARGET_REG]].dropna()
    sns.scatterplot(data=valid_data, x='mean_score_early', y=TARGET_REG, alpha=0.3)
    m, b = np.polyfit(valid_data['mean_score_early'], valid_data[TARGET_REG], 1)
    plt.plot(valid_data['mean_score_early'], m * valid_data['mean_score_early'] + b, color='red')
    plt.title('Corelatie: mean_score_early vs scor final')
    plt.xlabel('Scor mediu timpuriu')
    plt.ylabel('Nota finala')
    plt.tight_layout()
    plt.savefig("eda_corr_num_reg.jpg", dpi=120)
    plt.close()
    print("  [Grafic salvat: eda_corr_num_reg.jpg]")

    # d) Categorial vs Regresie
    print("\nd) Categorial vs. Regresie (highest_education vs scor final)")
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=train_df, x='highest_education', y=TARGET_REG, palette='Blues_d', hue='highest_education', legend=False)
    plt.title('Distributia notelor finale in functie de nivelul de educatie')
    plt.xlabel('Nivel de educatie')
    plt.ylabel('Nota finala')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig("eda_corr_cat_reg.png", dpi=120)
    plt.close()
    print("  [Grafic salvat: eda_corr_cat_reg.png]")

    # 1.4 Analiza redundantei intre atribute
    SUBSECTION("1.4 Analiza redundantei intre atribute")
    
    # a) Corelatia intre atribute numerice continue (Pearson)
    print("a) Corelatia intre atribute numerice continue (Pearson)")
    corr = train_df[num_cols].corr(method='pearson')
    high_corr = []
    for i in range(len(corr.columns)):
        for j in range(i):
            if abs(corr.iloc[i, j]) > 0.8:
                high_corr.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))
                
    df_hc = pd.DataFrame(high_corr, columns=['Atribut 1', 'Atribut 2', 'Corelatie']).sort_values(by="Corelatie", ascending=False)
    print("Perechi cu |corr| > 0.8:")
    print(df_hc.round(3).to_string(index=False))

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, vmin=-1, vmax=1)
    plt.title("Matricea de Corelatie Pearson")
    plt.tight_layout()
    plt.savefig("eda_pearson_heatmap.png", dpi=120)
    plt.close()
    print("  [Grafic salvat: eda_pearson_heatmap.png]")

    # b) Corelatie atribute categoriale vs. final_result -> Chipatrat
    print("\nb) Corelatie atribute categoriale vs. final_result -> Chi-patrat")
    chi_rows = []
    for c in cat_cols:
        if c == TARGET_CLF: continue
        ct = pd.crosstab(train_df[c], train_df[TARGET_CLF])
        chi2, p, dof, _ = chi2_contingency(ct)
        chi_rows.append({"Atribut": c, "chi2": round(chi2, 2), "p-value": f"{p:.4f}"})
    df_chi = pd.DataFrame(chi_rows).sort_values("chi2", ascending=False)
    print(df_chi.to_string(index=False))

    # c) Corelatie features - tinta
    print("\nc) Corelatie features - tinta")
    
    print("Numeric vs. tinta de regresie (Pearson) - top 5:")
    pear = train_df[num_cols + [TARGET_REG]].corr()[TARGET_REG].drop(TARGET_REG)
    print(pear.abs().sort_values(ascending=False).head(5).round(3).to_frame("|Pearson|").to_string())

    print("\nNumeric vs. tinta de clasificare (corelatie cu eticheta encodata) - top 5:")
    y_num = pd.Categorical(train_df[TARGET_CLF]).codes
    corrs = {c: np.corrcoef(train_df[c].fillna(train_df[c].median()), y_num)[0, 1] for c in num_cols}
    s = pd.Series(corrs).abs().sort_values(ascending=False).head(5)
    print(s.round(3).to_frame("|corelatie|").to_string())

    print("\nCategorial vs. tinta de regresie (spread mediei pe grupuri) - top 5:")
    cat_reg = []
    for c in cat_cols:
        if c == TARGET_CLF: continue
        grp = train_df.groupby(c)[TARGET_REG].mean()
        cat_reg.append({"Atribut": c, "spread medie": round(grp.max() - grp.min(), 2), "n_grupuri": len(grp)})
    print(pd.DataFrame(cat_reg).sort_values("spread medie", ascending=False).head(5).to_string(index=False))

    return None


# 3.2 Preprocessing phase 

def run_preprocess(X_train_raw, X_val_raw, num_cols, cat_cols):
    SECTION("3.2 — PREPROCESARE")

    SUBSECTION("Valori lipsa per coloana")
    miss = X_train_raw.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    if len(miss):
        print(miss.to_frame("n_lipsa").to_string())
    else:
        print("  (niciuna)")

    SUBSECTION("Outliere (cu IQR k=1.5) per coloana numerica")
    out_rows = [{"atribut": c,
                 "n_outliere": detect_outliers_iqr(X_train_raw[c]).sum(),
                 "procent": round(100 * detect_outliers_iqr(X_train_raw[c]).sum()
                                  / len(X_train_raw), 2)}
                for c in num_cols]
    df_out = pd.DataFrame(out_rows).sort_values("n_outliere", ascending=False)
    print(df_out.to_string(index=False))

    SUBSECTION("Pipeline: mediana + IQR + standardizare + one-hot")
    preproc = Preprocessor(num_cols=num_cols, cat_cols=cat_cols, iqr_k=1.5,
                           num_impute="median", standardize=True, remove_outliers=True)
    X_train = preproc.fit_transform(X_train_raw)
    X_val = preproc.transform(X_val_raw)
    print(f"  X_train post-preproc: {X_train.shape}")
    print(f"  X_val   post-preproc: {X_val.shape}")
    return X_train, X_val, preproc


# 3.3 Classfication phase

def run_classification(X_train, X_val, X_train_raw, X_val_raw, y_train, y_val, num_cols, cat_cols):
    SECTION("3.3 — CLASIFICARE")

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)
    classes = le.classes_.tolist()

    print(f"\nClase: {classes}")
    dist_tr = dict(zip(classes, np.bincount(y_train_enc)))
    dist_va = dict(zip(classes, np.bincount(y_val_enc)))
    print(f"Distributie train: {dist_tr}")
    print(f"Distributie val:   {dist_va}")

    # discretizare pt. ID3 manual
    X_train_cat, X_val_cat = discretize(X_train_raw[num_cols + cat_cols], X_val_raw[num_cols + cat_cols], num_cols, cat_cols, n_bins=4)
    if len(X_train_cat) > SAMPLE_FOR_MANUAL:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_train_cat), size=SAMPLE_FOR_MANUAL, replace=False)
        X_tr_s = X_train_cat.iloc[idx].reset_index(drop=True)
        y_tr_s = y_train.iloc[idx].reset_index(drop=True)
    else:
        X_tr_s, y_tr_s = X_train_cat, y_train

    # Experimentele pt ID3 realizat manual
    # in general lente (fata de skleanr) deci folosim un esantion pentru ablatie, dar variem hiperparametrii pentru a vedea cum influenteaza performanta

    ablation = []  # (config_name, model_name, hyperparams_dict, metrics_dict)

    SUBSECTION("Experiment 1: Baseline ID3 manual cu max_depth=3")
    print("Configuratie: ID3 manual cu max_depth=3, min_samples=2")

    dt = DecisionTree(max_depth=3, min_samples_per_node=2)
    dt.fit(X_tr_s, y_tr_s)
    m = eval_clf(y_val, dt.predict(X_val_cat), classes)
    ablation.append(("Exp 1: ID3 baseline", "ID3 manual",
                     {"max_depth": 3, "min_samples": 2}, m))
    print(pd.DataFrame([m]).round(4).to_string(index=False))

    SUBSECTION("Experiment 2: ID3 manual cu variere max_depth (5, 7, 10)")
    print("Modificam doar max_depth, restul ramane identic cu baseline-ul")

    for d in [5, 7, 10]:
        dt = DecisionTree(max_depth=d, min_samples_per_node=2)
        dt.fit(X_tr_s, y_tr_s)
        m = eval_clf(y_val, dt.predict(X_val_cat), classes)
        ablation.append((f"Exp 2: ID3 depth={d}", "ID3 manual",
                         {"max_depth": d, "min_samples": 2}, m))
    df_e2 = pd.DataFrame([{"max_depth": a[2]["max_depth"], **a[3]}
                          for a in ablation if a[0].startswith("Exp 2")]).round(4)
    print(df_e2.to_string(index=False))
    best_depth = max([a for a in ablation if a[0].startswith("Exp 2")],
                     key=lambda a: a[3]["f1_macro"])[2]["max_depth"]
    print(f"  -> Cea mai buna adancime: {best_depth}")

    SUBSECTION(f"Experiment 3: ID3 manual cu variere min_samples (5, 20, 50) si cea mai buna adancime gasita: {best_depth}")
    print("Pastram cea mai buna adancime din experimentul 2 si variem min_samples_per_node")

    for ms in [5, 20, 50]:
        dt = DecisionTree(max_depth=best_depth, min_samples_per_node=ms)
        dt.fit(X_tr_s, y_tr_s)
        m = eval_clf(y_val, dt.predict(X_val_cat), classes)
        ablation.append((f"Exp 3: ID3 d={best_depth},ms={ms}", "ID3 manual",
                         {"max_depth": best_depth, "min_samples": ms}, m))
    df_e3 = pd.DataFrame([{"min_samples": a[2]["min_samples"], **a[3]}
                          for a in ablation if a[0].startswith("Exp 3")]).round(4)
    print(df_e3.to_string(index=False))

    SUBSECTION("Experiment 4: RandomForest manual cu variere de n_estimators")
    print(f"RandomForest cu max_depth={best_depth}, variem n_estimators ∈ {{10, 25, 50}}.")

    for n in [10, 25, 50]:
        rf = RandomForest(n_estimators=n, max_depth=best_depth, min_samples_per_node=2)
        rf.fit(X_tr_s, y_tr_s)
        m = eval_clf(y_val, rf.predict(X_val_cat), classes)
        ablation.append((f"Exp 4: RF manual n={n}", "RandomForest manual",
                         {"n_estimators": n, "max_depth": best_depth}, m))
    df_e4 = pd.DataFrame([{"n_estimators": a[2]["n_estimators"], **a[3]}
                          for a in ablation if a[0].startswith("Exp 4")]).round(4)
    print(df_e4.to_string(index=False))


    # Experimente sklearn
    # fiind mai rapid, putem varia hiperparametrii fara esantionare, folosind toate datele de antrenament

    SUBSECTION("Experiment 5: sklearn DecisionTreeClassifier cu variere max_depth")
    print("Folosim atributele numerice direct (fara binning) + class_weight='balanced'")

    for d in [5, 10, 20, None]:
        sk_dt = DecisionTreeClassifier(max_depth=d, min_samples_leaf=5,
                                        class_weight="balanced", random_state=42)
        sk_dt.fit(X_train, y_train_enc)
        y_pred_str = le.inverse_transform(sk_dt.predict(X_val))
        m = eval_clf(y_val, y_pred_str, classes)
        ablation.append((f"Exp 5: sk DT depth={d}", "sklearn DT",
                         {"max_depth": d, "min_samples_leaf": 5}, m))
    df_e5 = pd.DataFrame([{"max_depth": str(a[2]["max_depth"]), **a[3]}
                          for a in ablation if a[0].startswith("Exp 5")]).round(4)
    print(df_e5.to_string(index=False))

    SUBSECTION("Experiment 6: sklearn RandomForestClassifier cu variere de n_estimators")
    print("Pastram max_depth=15 si variem n_estimators ∈ {50, 100, 200}.")

    best_sk_rf = None
    best_sk_rf_metrics = None
    for n in [50, 100, 200]:
        sk_rf = RandomForestClassifier(n_estimators=n, max_depth=15, min_samples_leaf=5,
                                        class_weight="balanced", n_jobs=-1, random_state=42)
        sk_rf.fit(X_train, y_train_enc)
        y_pred_str = le.inverse_transform(sk_rf.predict(X_val))
        m = eval_clf(y_val, y_pred_str, classes)
        ablation.append((f"Exp 6: sk RF n={n}", "sklearn RF",
                         {"n_estimators": n, "max_depth": 15}, m))
        if best_sk_rf is None or m["f1_macro"] > best_sk_rf_metrics["f1_macro"]:
            best_sk_rf = sk_rf
            best_sk_rf_metrics = m
    df_e6 = pd.DataFrame([{"n_estimators": a[2]["n_estimators"], **a[3]}
                          for a in ablation if a[0].startswith("Exp 6")]).round(4)
    print(df_e6.to_string(index=False))

    # tabelul comparativ al tuturor experimentelor de ablatie

    SECTION("3.3 EVALUARE — Tabel comparativ al tuturor experimentelor de ablatie")

    df_all = pd.DataFrame([{"experiment": a[0], "model": a[1],
                             **a[3]} for a in ablation]).round(4)
    print(df_all.to_string(index=False))

    SUBSECTION("Cel mai bun model per familie")
    by_model = {}
    for cfg, model, hp, m in ablation:
        if model not in by_model or m["f1_macro"] > by_model[model][3]["f1_macro"]:
            by_model[model] = (cfg, model, hp, m)
    df_best = pd.DataFrame([{"experiment": v[0], "model": v[1],
                              "hiperparametri": str(v[2]), **v[3]}
                             for v in by_model.values()]).round(4)
    print(df_best.to_string(index=False))

    # matricea de confuzie + raport pe clase pentru cel mai bun model

    SUBSECTION("Matrice de confuzie — cel mai bun model: sklearn RandomForest")
    y_pred = best_sk_rf.predict(X_val)
    cm = confusion_matrix(y_val_enc, y_pred, labels=range(len(classes)))
    df_cm = pd.DataFrame(cm, index=[f"true_{c}" for c in classes],
                              columns=[f"pred_{c}" for c in classes])
    print(df_cm.to_string())
    plot_confusion(cm, classes, "Matrice de confuzie — sklearn RF",
                    "confusion_matrix.png")
    print("  [Grafic salvat: confusion_matrix.png]")

    SUBSECTION("Raport pe clase (precizie/recall/F1 per clasa)")
    print(classification_report(y_val, le.inverse_transform(y_pred),
                                 target_names=classes, zero_division=0))

    return best_sk_rf, le


# 3.3 Regression phase 

def run_regression(X_train, X_val, y_train, y_val, num_cols):
    SECTION("3.3 — REGRESIE")

    Xtr = add_bias(X_train); Xva = add_bias(X_val)
    print(f"\nshape: train={Xtr.shape}, val={Xva.shape}")
    print(f"tinta: mean={y_train.mean():.2f}, std={y_train.std():.2f}, "f"[{y_train.min():.1f}, {y_train.max():.1f}]")

    results = []  # (exp_name, model, hp, train_metrics, val_metrics)

    # Experiment 1: LinearRegression manual
    # stabilim un baseline cu regresie liniara simpla (fara regularizare) realizata manual prin pseudoinversa, pentru a avea un punct de referinta pentru experimentele ulterioare cu Ridge si varierea complexitatii polinomiale

    SUBSECTION("Experiment 1: Baseline LinearRegression")
    print("Regresie liniara prin pseudoinversa, fara regularizare")

    lin = LinearRegressionLab()
    lin.fit(Xtr, y_train)
    mt = reg_metrics(y_train, lin.predict(Xtr))
    mv = reg_metrics(y_val, lin.predict(Xva))
    results.append(("Exp 1: LinReg baseline", "LinearRegression manual", {}, mt, mv))
    print(pd.DataFrame([{"set": "train", **mt}, {"set": "val", **mv}]).round(4).to_string(index=False))

    # Experiment 2: RidgeRegression manual cu variere alpha
    SUBSECTION("Experiment 2: RidgeRegression manual cu variere alpha")
    print("Variem factorul de regularizare alpha pe 7 valori")

    alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    mse_tr, mse_va = [], []
    for a in alphas:
        r = RidgeRegressionLab(alpha=a)
        r.fit(Xtr, y_train)
        mt = reg_metrics(y_train, r.predict(Xtr))
        mv = reg_metrics(y_val, r.predict(Xva))
        results.append((f"Exp 2: Ridge alpha={a}", "RidgeRegression manual",
                        {"alpha": a}, mt, mv))
        mse_tr.append(mt["MSE"]); mse_va.append(mv["MSE"])

    df_alpha = pd.DataFrame([{"alpha": a, "train_MSE": mt, "val_MSE": mv}
                              for a, mt, mv in zip(alphas, mse_tr, mse_va)]).round(4)
    print(df_alpha.to_string(index=False))
    best_alpha = alphas[int(np.argmin(mse_va))]
    print(f"  -> Cel mai bun alpha (min val MSE): {best_alpha}")

    # plot Ridge alpha
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(alphas, mse_tr, "o-", label="MSE train", color="green")
    ax.semilogx(alphas, mse_va, "o-", label="MSE val", color="red")
    ax.axvline(best_alpha, ls="--", color="gray", alpha=0.5,
                label=f"best alpha={best_alpha}")
    ax.set_xlabel("alpha (log)"); ax.set_ylabel("MSE")
    ax.set_title("Ridge — train vs. val MSE in functie de alpha")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("ridge_alpha_curves.png", dpi=120); plt.close()
    print("  [Grafic salvat: ridge_alpha_curves.png]")

    # Experiment 3: variem complexitate (M = grad polinomial)
    SUBSECTION("Experiment 3: RidgeRegression manual cu variere complexitate polinomiala")
    print("Adaugam features x^2 si x^3 pe coloanele numerice si pastram alpha optim")

    n_num = len(num_cols)
    Xtr_n, Xva_n = X_train[:, :n_num], X_val[:, :n_num]
    Xtr_c, Xva_c = X_train[:, n_num:], X_val[:, n_num:]
    Ms = [1, 2, 3]
    mse_tr_M, mse_va_M = [], []
    for M in Ms:
        # x, x^2, ..., x^M + bias
        Xtr_p = np.hstack([np.ones((Xtr_n.shape[0], 1))] +
                          [Xtr_n ** m for m in range(1, M + 1)])
        Xva_p = np.hstack([np.ones((Xva_n.shape[0], 1))] +
                          [Xva_n ** m for m in range(1, M + 1)])
        Xtr_full = np.hstack([Xtr_p, Xtr_c])
        Xva_full = np.hstack([Xva_p, Xva_c])
        r = RidgeRegressionLab(alpha=best_alpha)
        r.fit(Xtr_full, y_train)
        mt = reg_metrics(y_train, r.predict(Xtr_full))
        mv = reg_metrics(y_val, r.predict(Xva_full))
        results.append((f"Exp 3: Ridge poly M={M}", "RidgeRegression poly",
                        {"alpha": best_alpha, "M": M}, mt, mv))
        mse_tr_M.append(mt["MSE"]); mse_va_M.append(mv["MSE"])

    df_M = pd.DataFrame([{"M": M, "train_MSE": tr, "val_MSE": va}
                          for M, tr, va in zip(Ms, mse_tr_M, mse_va_M)]).round(4)
    print(df_M.to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(Ms, mse_tr_M, "o-", label="MSE train", color="green")
    ax.plot(Ms, mse_va_M, "o-", label="MSE val", color="red")
    ax.set_xlabel("M (grad polinomial)"); ax.set_ylabel("MSE")
    ax.set_title("Complexitate (M) vs. eroare")
    ax.legend(); ax.grid(alpha=0.3); ax.set_xticks(Ms)
    plt.tight_layout(); plt.savefig("complexity_curves.png", dpi=120); plt.close()
    print("  [Grafic salvat: complexity_curves.png]")

    # Experiment 4: sklearn Ridge + Lasso
    SUBSECTION("Experiment 4: sklearn Ridge + Lasso")
    print("Comparatie intre Ridge si Lasso din sklearn, cu variere de alpha (fiecare ia cate 3 valori)")

    sk_rows = []
    for a in [best_alpha * 0.1, best_alpha, best_alpha * 10]:
        m = SkRidge(alpha=a, random_state=42)
        m.fit(X_train, y_train)
        mv = reg_metrics(y_val, m.predict(X_val))
        results.append((f"Exp 4: sk Ridge a={a}", "sklearn Ridge",
                        {"alpha": a}, None, mv))
        sk_rows.append({"model": "sk Ridge", "alpha": a, **mv})
    for a in [0.01, 0.1, 1.0]:
        m = SkLasso(alpha=a, random_state=42, max_iter=5000)
        m.fit(X_train, y_train)
        mv = reg_metrics(y_val, m.predict(X_val))
        nz = (np.abs(m.coef_) > 1e-8).sum()
        results.append((f"Exp 4: sk Lasso a={a}", "sklearn Lasso",
                        {"alpha": a, "nonzero": nz}, None, mv))
        sk_rows.append({"model": "sk Lasso", "alpha": a,
                        "nonzero": f"{nz}/{len(m.coef_)}", **mv})
    print(pd.DataFrame(sk_rows).round(4).to_string(index=False))

    # tabel de comparatie al tuturor experimentelor de regresie
    SECTION("Tavel comparativ regresii")
    df_final = pd.DataFrame([{"experiment": r[0], "model": r[1],
                                "hiperparametri": str(r[2]), **r[4]}
                               for r in results]).round(4)
    print(df_final.to_string(index=False))

    best_idx = int(df_final["MSE"].idxmin())
    best = df_final.iloc[best_idx]
    print(f"\n  -> Cel mai bun: {best['experiment']} "
          f"(MSE={best['MSE']:.4f}, R2={best['R2']:.4f})")

    # antrenam si returnam modelul cel mai bun (Ridge cu best_alpha) pentru submisie
    best_ridge = RidgeRegressionLab(alpha=best_alpha)
    best_ridge.fit(Xtr, y_train)
    return best_ridge, best_alpha


# pt submisii
def write_submission_classification(preproc, model, le, num_cols, cat_cols):
    SUBSECTION("Submisie clasificare")
    try:
        df_priv = pd.read_csv(PRIVATE_CLF_PATH)
    except FileNotFoundError:
        print(f"  [!] {PRIVATE_CLF_PATH} nu exista — sarim peste.")
        return
    ids = df_priv["id"].values
    X_priv = preproc.transform(df_priv)
    pred = le.inverse_transform(model.predict(X_priv))
    sub = pd.DataFrame({"id": ids, "prediction": pred})
    sub.to_csv("submisie_kaggle_clasificare.csv", index=False)
    print(f"  Salvat: submisie_kaggle_clasificare.csv ({sub.shape})")


def write_submission_regression(preproc, model, num_cols, cat_cols):
    SUBSECTION("Submisie regresie")
    try:
        df_priv = pd.read_csv(PRIVATE_REG_PATH)
    except FileNotFoundError:
        print(f"  [!] {PRIVATE_REG_PATH} nu exista — sarim peste.")
        return
    ids = df_priv["id"].values
    X_priv = preproc.transform(df_priv)
    pred = model.predict(add_bias(X_priv))
    sub = pd.DataFrame({"id": ids, "prediction": pred})
    sub.to_csv("submisie_kaggle_regresie.csv", index=False)
    print(f"  Salvat: submisie_kaggle_regresie.csv ({sub.shape})")



def main():
    train_full = pd.read_csv(TRAIN_PATH, na_values=["", " ", "?", "NA"])
    print(f"Set complet: {train_full.shape}")
    train_df, val_df = train_test_split(train_full, test_size=0.2, random_state=42)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    print(f"  train: {train_df.shape}, val: {val_df.shape}")

    # tipuri features (excludem tintele)
    feat_df = train_df.drop(columns=[TARGET_CLF, TARGET_REG])
    num_cols = feat_df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = feat_df.select_dtypes(exclude=["number"]).columns.tolist()
    print(f"  Numerice: {len(num_cols)}, Categoriale: {len(cat_cols)}")

    # EDA
    run_eda(train_df, num_cols, cat_cols)

    # preprocesare
    X_train_raw = train_df.drop(columns=[TARGET_CLF, TARGET_REG])
    X_val_raw = val_df.drop(columns=[TARGET_CLF, TARGET_REG])
    X_train, X_val, preproc = run_preprocess(X_train_raw, X_val_raw, num_cols, cat_cols)

    y_clf_train = train_df[TARGET_CLF]
    y_clf_val = val_df[TARGET_CLF]
    y_reg_train = train_df[TARGET_REG]
    y_reg_val = val_df[TARGET_REG]

    # clasificare
    m_clf = y_clf_train.notna()
    m_clf_v = y_clf_val.notna()
    best_clf, le = run_classification(
        X_train[m_clf.values], X_val[m_clf_v.values],
        X_train_raw[m_clf.values].reset_index(drop=True),
        X_val_raw[m_clf_v.values].reset_index(drop=True),
        y_clf_train[m_clf].reset_index(drop=True),
        y_clf_val[m_clf_v].reset_index(drop=True),
        num_cols, cat_cols)

    # regresii
    m_reg = y_reg_train.notna()
    m_reg_v = y_reg_val.notna()
    best_reg, best_alpha = run_regression(
        X_train[m_reg.values], X_val[m_reg_v.values],
        y_reg_train[m_reg].values, y_reg_val[m_reg_v].values, num_cols)

    # out
    write_submission_classification(preproc, best_clf, le, num_cols, cat_cols)
    write_submission_regression(preproc, best_reg, num_cols, cat_cols)


if __name__ == "__main__":
    main()