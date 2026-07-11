"""
Cerinta 3.3 — Regresie.

Modele:
  - LinearRegression manual (lab) — baseline
  - RidgeRegression manual (lab) — variere alpha (cerinta explicita: regularizare)
  - sklearn Ridge + Lasso — extensie
  - features polinomiale (M=1,2,3) pentru observat overfitting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge as SkRidge, Lasso as SkLasso
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score)

from models import (LinearRegressionLab, RidgeRegressionLab,
                         extract_polynomial_features, add_bias)


def reg_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def run_regression(X_train_proc, X_val_proc,
                   y_reg_train, y_reg_val,
                   num_cols):
    """Antreneaza modelele de regresie si raporteaza."""
    print("\n" + "=" * 70)
    print(" 3.3 — REGRESIE")
    print("=" * 70)

    Xtr = add_bias(X_train_proc)
    Xva = add_bias(X_val_proc)
    print(f"\nshape train: {Xtr.shape}, val: {Xva.shape}")
    print(f"Tinta: mean={y_reg_train.mean():.2f}, std={y_reg_train.std():.2f}, "
          f"[{y_reg_train.min():.1f}, {y_reg_train.max():.1f}]")

    # ===== BASELINE LINREG =====
    print("\n[REG-1] Baseline LinearRegression (lab)")
    lin = LinearRegressionLab()
    lin.fit(Xtr, y_reg_train)
    m_train = reg_metrics(y_reg_train, lin.predict(Xtr))
    m_val = reg_metrics(y_reg_val, lin.predict(Xva))
    print(f"  train: MSE={m_train['MSE']:.4f}, R2={m_train['R2']:.4f}")
    print(f"  val:   MSE={m_val['MSE']:.4f}, R2={m_val['R2']:.4f}")

    # ===== ABLATIE ALPHA (RIDGE LAB) =====
    print("\n[REG-2] RidgeRegression (lab) — variere alpha")
    alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    mse_tr, mse_va, all_res = [], [], []
    for a in alphas:
        r = RidgeRegressionLab(alpha=a)
        r.fit(Xtr, y_reg_train)
        mt = reg_metrics(y_reg_train, r.predict(Xtr))
        mv = reg_metrics(y_reg_val, r.predict(Xva))
        mse_tr.append(mt["MSE"]); mse_va.append(mv["MSE"])
        all_res.append((a, mt, mv))
        print(f"  alpha={a:>7.3f}: train MSE={mt['MSE']:.4f}, val MSE={mv['MSE']:.4f}, R2={mv['R2']:.4f}")

    best_alpha = alphas[int(np.argmin(mse_va))]
    print(f"  -> cel mai bun alpha: {best_alpha}")

    # plot train vs val MSE in functie de alpha
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(alphas, mse_tr, "o-", label="MSE train", color="green")
    ax.semilogx(alphas, mse_va, "o-", label="MSE val", color="red")
    ax.axvline(best_alpha, ls="--", color="gray", alpha=0.5, label=f"best alpha={best_alpha}")
    ax.set_xlabel("alpha (log)"); ax.set_ylabel("MSE")
    ax.set_title("Ridge (lab) — MSE in functie de alpha")
    ax.legend(); ax.grid(alpha=0.3); plt.tight_layout()
    plt.savefig("ridge_alpha_curves.png", dpi=120); plt.close()
    print("  Salvat: ridge_alpha_curves.png")

    # ===== COMPLEXITATE — POLYNOMIAL FEATURES =====
    print("\n[REG-3] Variere complexitate (M = grad polinomial pe numericele)")
    n_num = len(num_cols)
    Xtr_num, Xtr_cat = X_train_proc[:, :n_num], X_train_proc[:, n_num:]
    Xva_num, Xva_cat = X_val_proc[:, :n_num], X_val_proc[:, n_num:]
    Ms = [1, 2, 3]
    mse_tr_M, mse_va_M = [], []
    for M in Ms:
        Xtr_full = np.hstack([extract_polynomial_features(Xtr_num, M), Xtr_cat])
        Xva_full = np.hstack([extract_polynomial_features(Xva_num, M), Xva_cat])
        r = RidgeRegressionLab(alpha=best_alpha)
        r.fit(Xtr_full, y_reg_train)
        mt = reg_metrics(y_reg_train, r.predict(Xtr_full))
        mv = reg_metrics(y_reg_val, r.predict(Xva_full))
        mse_tr_M.append(mt["MSE"]); mse_va_M.append(mv["MSE"])
        print(f"  M={M}: train MSE={mt['MSE']:.4f}, val MSE={mv['MSE']:.4f}, R2={mv['R2']:.4f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(Ms, mse_tr_M, "o-", label="MSE train", color="green")
    ax.plot(Ms, mse_va_M, "o-", label="MSE val", color="red")
    ax.set_xlabel("M (grad polinomial)"); ax.set_ylabel("MSE")
    ax.set_title("Complexitate (M) vs. eroare")
    ax.legend(); ax.grid(alpha=0.3); ax.set_xticks(Ms)
    plt.tight_layout(); plt.savefig("complexity_curves.png", dpi=120); plt.close()
    print("  Salvat: complexity_curves.png")

    # ===== SKLEARN RIDGE + LASSO =====
    print("\n[REG-4] sklearn Ridge + Lasso (extensie)")
    sk_results = []
    for a in [best_alpha * 0.1, best_alpha, best_alpha * 10]:
        m = SkRidge(alpha=a, random_state=42)
        m.fit(X_train_proc, y_reg_train)
        mv = reg_metrics(y_reg_val, m.predict(X_val_proc))
        sk_results.append((f"sk Ridge alpha={a}", mv))
        print(f"  Ridge alpha={a}: val MSE={mv['MSE']:.4f}, R2={mv['R2']:.4f}")
    for a in [0.01, 0.1, 1.0]:
        m = SkLasso(alpha=a, random_state=42, max_iter=5000)
        m.fit(X_train_proc, y_reg_train)
        mv = reg_metrics(y_reg_val, m.predict(X_val_proc))
        nz = (np.abs(m.coef_) > 1e-8).sum()
        sk_results.append((f"sk Lasso alpha={a}", mv))
        print(f"  Lasso alpha={a}: val MSE={mv['MSE']:.4f}, R2={mv['R2']:.4f}, nonzero={nz}/{len(m.coef_)}")

    # ===== TABEL COMPARATIV =====
    print("\n[REG-5] Tabel comparativ regresie (pe val)")
    df = pd.DataFrame([
        {"algoritm": "LinReg (lab)", **m_val},
        {"algoritm": f"Ridge (lab) alpha={best_alpha}",
            **all_res[int(np.argmin(mse_va))][2]},
    ] + [{"algoritm": n, **m} for n, m in sk_results]).round(4)
    print(df.to_string(index=False))
    best = df.iloc[int(df["MSE"].idxmin())]
    print(f"\n  -> Cel mai bun: {best['algoritm']} (MSE={best['MSE']:.4f}, R2={best['R2']:.4f})")

    # returnam cel mai bun model lab antrenat (pt. submisie)
    best_ridge = RidgeRegressionLab(alpha=best_alpha)
    best_ridge.fit(Xtr, y_reg_train)
    return best_ridge, best_alpha