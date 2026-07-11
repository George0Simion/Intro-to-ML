"""
Cerinta 3.3 — Clasificare.

Modele:
  - DecisionTree manual (ID3 din lab) pe atribute discretizate
  - RandomForest manual (din lab)
  - sklearn DecisionTreeClassifier + RandomForestClassifier (extensie practica)

Variem hiperparametri (max_depth, min_samples, n_estimators) — ablatie.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              classification_report, confusion_matrix)

from models import DecisionTree, RandomForest, discretize


# ----- metrici -----
def eval_clf(y_true, y_pred, classes):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro",
                                           zero_division=0, labels=classes),
        "recall_macro": recall_score(y_true, y_pred, average="macro",
                                     zero_division=0, labels=classes),
        "f1_macro": f1_score(y_true, y_pred, average="macro",
                             zero_division=0, labels=classes),
    }


def plot_confusion(cm, class_names, out="cm.png", title="Matrice de confuzie"):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names, rotation=30)
    ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predictie"); ax.set_ylabel("Real"); ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im); plt.tight_layout()
    plt.savefig(out, dpi=120); plt.close()


def run_classification(X_train_proc, X_val_proc,
                       X_train_raw_clf, X_val_raw_clf,
                       y_clf_train, y_clf_val,
                       num_cols, cat_cols,
                       sample_for_manual=3000):
    """Antreneaza toate modelele de clasificare si raporteaza ablatia."""
    print("\n" + "=" * 70)
    print(" 3.3 — CLASIFICARE")
    print("=" * 70)

    # encodare tinta
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_clf_train)
    y_val_enc = le.transform(y_clf_val)
    classes = le.classes_.tolist()
    print(f"\nClase: {classes}")
    print(f"Distributie train: {dict(zip(classes, np.bincount(y_train_enc)))}")
    print(f"Distributie val:   {dict(zip(classes, np.bincount(y_val_enc)))}")

    # ===== DISCRETIZARE PT. ID3 MANUAL =====
    print("\n[CLF-1] Discretizare numericele -> categorial (ID3 manual)")
    X_train_cat, X_val_cat = discretize(X_train_raw_clf[num_cols + cat_cols],
                                         X_val_raw_clf[num_cols + cat_cols],
                                         num_cols, cat_cols, n_bins=4)
    print(f"  shape train: {X_train_cat.shape}")

    # esantion pentru ID3 (e lent)
    if len(X_train_cat) > sample_for_manual:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_train_cat), size=sample_for_manual, replace=False)
        X_tr_s = X_train_cat.iloc[idx].reset_index(drop=True)
        y_tr_s = y_clf_train.iloc[idx].reset_index(drop=True)
        print(f"  [!] Esantion {sample_for_manual} pentru ID3 manual")
    else:
        X_tr_s, y_tr_s = X_train_cat, y_clf_train

    # ===== ABLATIE ID3 MANUAL =====
    print("\n[CLF-2] Ablatie ID3 manual")
    ablation_id3 = []

    # baseline
    dt = DecisionTree(max_depth=3, min_samples_per_node=2)
    dt.fit(X_tr_s, y_tr_s)
    m = eval_clf(y_clf_val, dt.predict(X_val_cat), classes)
    ablation_id3.append(("ID3 baseline (depth=3)", 3, 2, m))
    print(f"  baseline depth=3: acc={m['accuracy']:.4f}, f1={m['f1_macro']:.4f}")

    # variere max_depth
    for d in [5, 7, 10]:
        dt = DecisionTree(max_depth=d, min_samples_per_node=2)
        dt.fit(X_tr_s, y_tr_s)
        m = eval_clf(y_clf_val, dt.predict(X_val_cat), classes)
        ablation_id3.append((f"ID3 depth={d}", d, 2, m))
        print(f"  depth={d}: acc={m['accuracy']:.4f}, f1={m['f1_macro']:.4f}")

    best_d = max(ablation_id3, key=lambda x: x[3]["f1_macro"])[1]
    print(f"  -> cel mai bun max_depth: {best_d}")

    # variere min_samples (cu best_depth fixat)
    for ms in [5, 20, 50]:
        dt = DecisionTree(max_depth=best_d, min_samples_per_node=ms)
        dt.fit(X_tr_s, y_tr_s)
        m = eval_clf(y_clf_val, dt.predict(X_val_cat), classes)
        ablation_id3.append((f"ID3 d={best_d}, ms={ms}", best_d, ms, m))
        print(f"  min_samples={ms}: acc={m['accuracy']:.4f}, f1={m['f1_macro']:.4f}")

    df_id3 = pd.DataFrame([{
        "config": cfg, "max_depth": d, "min_samples": ms,
        **{k: round(v, 4) for k, v in m.items()},
    } for cfg, d, ms, m in ablation_id3])
    print("\n  === Tabel ablatie ID3 ===")
    print(df_id3.to_string(index=False))

    # ===== RANDOM FOREST MANUAL =====
    print("\n[CLF-3] RandomForest manual (variere n_estimators)")
    ablation_rf = []
    for n in [10, 25, 50]:
        rf = RandomForest(n_estimators=n, max_depth=best_d, min_samples_per_node=2)
        rf.fit(X_tr_s, y_tr_s)
        m = eval_clf(y_clf_val, rf.predict(X_val_cat), classes)
        ablation_rf.append((f"RF manual n={n}", n, m))
        print(f"  n_estimators={n}: acc={m['accuracy']:.4f}, f1={m['f1_macro']:.4f}")

    # ===== SKLEARN =====
    print("\n[CLF-4] sklearn DecisionTree + RandomForest")
    ablation_sk = []
    for d in [5, 10, 20, None]:
        m = _eval_sklearn(DecisionTreeClassifier(max_depth=d, min_samples_leaf=5,
                                                  class_weight="balanced", random_state=42),
                          X_train_proc, y_train_enc, X_val_proc, y_clf_val, le, classes)
        ablation_sk.append((f"sk DT depth={d}", m))
        print(f"  sklearn DT depth={d}: acc={m['accuracy']:.4f}, f1={m['f1_macro']:.4f}")

    for n in [50, 100, 200]:
        m = _eval_sklearn(RandomForestClassifier(n_estimators=n, max_depth=15,
                                                  min_samples_leaf=5, class_weight="balanced",
                                                  n_jobs=-1, random_state=42),
                          X_train_proc, y_train_enc, X_val_proc, y_clf_val, le, classes)
        ablation_sk.append((f"sk RF n={n}", m))
        print(f"  sklearn RF n_estimators={n}: acc={m['accuracy']:.4f}, f1={m['f1_macro']:.4f}")

    # ===== CEL MAI BUN MODEL + MATRICE DE CONFUZIE =====
    print("\n[CLF-5] Tabel comparativ — cel mai bun din fiecare familie")
    best_id3 = max(ablation_id3, key=lambda x: x[3]["f1_macro"])
    best_rf = max(ablation_rf, key=lambda x: x[2]["f1_macro"])
    best_sk = max(ablation_sk, key=lambda x: x[1]["f1_macro"])
    df_cmp = pd.DataFrame([
        {"algoritm": best_id3[0], **best_id3[3]},
        {"algoritm": best_rf[0], **best_rf[2]},
        {"algoritm": best_sk[0], **best_sk[1]},
    ]).round(4)
    print(df_cmp.to_string(index=False))

    # reantrenam cel mai bun sklearn pt. matrice de confuzie + raport pe clase
    best_sk_model = RandomForestClassifier(n_estimators=200, max_depth=15,
                                            min_samples_leaf=5, class_weight="balanced",
                                            n_jobs=-1, random_state=42)
    best_sk_model.fit(X_train_proc, y_train_enc)
    y_pred_final = best_sk_model.predict(X_val_proc)
    cm = confusion_matrix(y_val_enc, y_pred_final, labels=range(len(classes)))
    df_cm = pd.DataFrame(cm, index=[f"true_{c}" for c in classes],
                              columns=[f"pred_{c}" for c in classes])
    print("\n  === Matrice de confuzie (sklearn RF, n=200) ===")
    print(df_cm.to_string())
    print("\n  === Raport pe clase ===")
    print(classification_report(y_clf_val, le.inverse_transform(y_pred_final),
                                target_names=classes, zero_division=0))
    plot_confusion(cm, classes, out="confusion_matrix.png",
                    title="Matrice de confuzie — sklearn RF")
    print("  Salvat: confusion_matrix.png")

    return best_sk_model, le


def _eval_sklearn(model, X_tr, y_tr, X_va, y_va_str, le, classes):
    model.fit(X_tr, y_tr)
    y_pred_str = le.inverse_transform(model.predict(X_va))
    return eval_clf(y_va_str, y_pred_str, classes)