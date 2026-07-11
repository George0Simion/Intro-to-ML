"""
Cerinta 3.1 — Explorarea datelor (EDA).

Genereaza:
  - tabel statistici pentru atribute numerice (count/mean/std/min/quartile/max)
  - tabel statistici pentru atribute categoriale (count/n_unique)
  - boxplot atribute numerice
  - histograme atribute categoriale
  - bar plot distributie clase
  - matrice de corelatie Pearson (numerice)
  - tabel chi-patrat (categoriale vs. clasa)
  - corelatie features-tinta (numerice si categoriale)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

from data import TARGET_CLF, TARGET_REG


def stats_numeric(df, num_cols):
    """Tabel cu count/mean/std/min/Q1/median/Q3/max pentru numerice."""
    return df[num_cols].describe().T  # transpus ca atributele sa fie pe linii


def stats_categorical(df, cat_cols):
    """Tabel cu count (non-NaN) si numar de valori unice pentru categoriale."""
    rows = []
    for c in cat_cols:
        rows.append({
            "atribut": c,
            "count": df[c].notna().sum(),
            "n_unique": df[c].nunique(dropna=True),
            "valori_unice": list(df[c].dropna().unique())[:5],  # primele 5
        })
    return pd.DataFrame(rows)


def plot_boxplots(df, num_cols, out="eda_boxplots.png", max_cols_per_fig=6):
    """Boxplot-uri pentru atribute numerice (pe figuri separate cand sunt multe)."""
    if len(num_cols) <= max_cols_per_fig:
        fig, ax = plt.subplots(figsize=(max(8, len(num_cols)), 5))
        df[num_cols].boxplot(ax=ax, rot=45)
        ax.set_title("Boxplot atribute numerice")
        plt.tight_layout()
        plt.savefig(out, dpi=120)
        plt.close()
        print(f"  Salvat: {out}")
    else:
        # Cand sunt multe, normalizam (z-score) ca sa incapa pe acelasi grafic
        df_z = (df[num_cols] - df[num_cols].mean()) / (df[num_cols].std() + 1e-9)
        fig, ax = plt.subplots(figsize=(max(10, len(num_cols) * 0.5), 6))
        df_z.boxplot(ax=ax, rot=90)
        ax.set_title("Boxplot atribute numerice (standardizate z-score)")
        plt.tight_layout()
        plt.savefig(out, dpi=120)
        plt.close()
        print(f"  Salvat: {out}")


def plot_categorical_histograms(df, cat_cols, out="eda_histograms.png", max_unique=15):
    """Histograme pentru categoriale (saritem peste cele cu prea multe valori)."""
    cols = [c for c in cat_cols if df[c].nunique() <= max_unique]
    if not cols:
        print("  [!] Nicio coloana categoriala potrivita pentru histograme.")
        return
    n = len(cols)
    cols_per_row = 3
    rows = (n + cols_per_row - 1) // cols_per_row
    fig, axes = plt.subplots(rows, cols_per_row, figsize=(cols_per_row * 4, rows * 3))
    axes = np.array(axes).reshape(-1)
    for i, c in enumerate(cols):
        df[c].value_counts(dropna=False).plot.bar(ax=axes[i])
        axes[i].set_title(c)
        axes[i].tick_params(axis="x", rotation=45)
    for j in range(len(cols), len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  Salvat: {out}")


def plot_class_balance(y_clf, out="eda_class_balance.png"):
    """Bar plot frecventa fiecarei clase."""
    counts = y_clf.value_counts()
    fig, ax = plt.subplots(figsize=(7, 4))
    counts.plot.bar(ax=ax, color="steelblue")
    ax.set_title(f"Distributia claselor in {TARGET_CLF}")
    ax.set_ylabel("Numar exemple")
    for i, v in enumerate(counts):
        ax.text(i, v, str(v), ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  Salvat: {out}")
    return counts


def correlation_numeric(df, num_cols, out="eda_corr_matrix.png"):
    """Matrice de corelatie Pearson + heatmap."""
    corr = df[num_cols].corr(method="pearson")
    fig, ax = plt.subplots(figsize=(max(6, len(num_cols) * 0.5), max(5, len(num_cols) * 0.5)))
    im = ax.matshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(num_cols))); ax.set_xticklabels(num_cols, rotation=90)
    ax.set_yticks(range(len(num_cols))); ax.set_yticklabels(num_cols)
    plt.colorbar(im)
    ax.set_title("Corelatie Pearson — atribute numerice")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  Salvat: {out}")
    return corr


def chi2_categorical_vs_target(df, cat_cols, target=TARGET_CLF):
    """Test Chi-patrat intre fiecare atribut categorial si tinta de clasificare."""
    rows = []
    for c in cat_cols:
        if c == target:
            continue
        ct = pd.crosstab(df[c], df[target])
        chi2, p, dof, _ = chi2_contingency(ct)
        rows.append({"atribut": c, "chi2": chi2, "p_value": p, "dof": dof})
    return pd.DataFrame(rows).sort_values("chi2", ascending=False)


def correlation_with_targets(df, num_cols, cat_cols):
    """
    Corelatie features-tinta:
      - numerice vs. tinta_regresie: Pearson
      - numerice vs. tinta_clasificare: ANOVA F (reprezentat ca eta^2 simplificat
        prin Pearson dintre feature si clasa numerica encodata; valori mai mari = mai informativ)
      - categoriale vs. tinta_regresie: medie/std a tintei pe valoarea atributului
    """
    out = {}

    # numerice vs. regresie
    if TARGET_REG in df.columns:
        out["num_vs_regresie"] = df[num_cols + [TARGET_REG]].corr(
            method="pearson")[TARGET_REG].drop(TARGET_REG).sort_values(key=abs, ascending=False)

    # numerice vs. clasificare (encodare numerica)
    if TARGET_CLF in df.columns:
        y_num = pd.Categorical(df[TARGET_CLF]).codes
        corrs = {}
        for c in num_cols:
            corrs[c] = np.corrcoef(df[c].fillna(df[c].median()), y_num)[0, 1]
        out["num_vs_clasificare"] = pd.Series(corrs).sort_values(key=abs, ascending=False)

    # categoriale vs. regresie (cat de mult variaza media tintei intre grupuri)
    if TARGET_REG in df.columns:
        rows = []
        for c in cat_cols:
            if c == TARGET_CLF:
                continue
            grp = df.groupby(c)[TARGET_REG].agg(["mean", "std", "count"])
            rows.append({"atribut": c,
                         "spread_mean": grp["mean"].max() - grp["mean"].min(),
                         "n_grupuri": len(grp)})
        out["cat_vs_regresie"] = pd.DataFrame(rows).sort_values("spread_mean", ascending=False)

    return out


def run_eda(train_df):
    """Punctul de intrare al EDA — apeleaza tot si printeaza rezultatele."""
    print("\n" + "=" * 70)
    print(" 3.1 — EXPLORAREA DATELOR (EDA)")
    print("=" * 70)

    num_cols = train_df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = train_df.select_dtypes(exclude=["number"]).columns.tolist()

    # Scoatem tintele din lista de features pt. statistici
    num_feat = [c for c in num_cols if c != TARGET_REG]
    cat_feat = [c for c in cat_cols if c != TARGET_CLF]

    print(f"\n[EDA-1] Tipuri atribute: {len(num_feat)} numerice, {len(cat_feat)} categoriale")
    print(f"  Numerice:    {num_feat}")
    print(f"  Categoriale: {cat_feat}")

    print("\n[EDA-2] Statistici atribute numerice")
    print(stats_numeric(train_df, num_feat).round(2).to_string())

    print("\n[EDA-3] Statistici atribute categoriale")
    print(stats_categorical(train_df, cat_feat).to_string(index=False))

    print("\n[EDA-4] Distributia claselor")
    counts = plot_class_balance(train_df[TARGET_CLF])
    print(counts.to_string())
    print(f"  Raport max/min: {counts.max() / counts.min():.2f}")

    print("\n[EDA-5] Boxplot-uri numerice")
    plot_boxplots(train_df, num_feat)

    print("\n[EDA-6] Histograme categoriale")
    plot_categorical_histograms(train_df, cat_feat)

    print("\n[EDA-7] Corelatie Pearson — atribute numerice")
    corr = correlation_numeric(train_df, num_feat)
    high_corr = []
    for i in range(len(corr)):
        for j in range(i + 1, len(corr)):
            if abs(corr.iloc[i, j]) > 0.8:
                high_corr.append((corr.index[i], corr.columns[j], corr.iloc[i, j]))
    print(f"  Perechi cu |corr| > 0.8 (potential redundante):")
    for a, b, v in sorted(high_corr, key=lambda x: -abs(x[2])):
        print(f"    {a} ~ {b}: {v:.3f}")

    print("\n[EDA-8] Chi-patrat: atribute categoriale vs. tinta de clasificare")
    print(chi2_categorical_vs_target(train_df, cat_feat).round(3).head(10).to_string(index=False))

    print("\n[EDA-9] Corelatie features-tinta")
    rels = correlation_with_targets(train_df, num_feat, cat_feat)
    print("  Numerice vs. tinta regresie (top 10 dupa |Pearson|):")
    print("   ", rels["num_vs_regresie"].head(10).round(3).to_string().replace("\n", "\n    "))
    print("\n  Numerice vs. tinta clasificare (top 10 dupa |corelatie cu eticheta encodata|):")
    print("   ", rels["num_vs_clasificare"].head(10).round(3).to_string().replace("\n", "\n    "))
    print("\n  Categoriale vs. tinta regresie (top 5 dupa spread mediei):")
    print(rels["cat_vs_regresie"].head(5).round(2).to_string(index=False))

    print("\n  Fisiere EDA generate: eda_class_balance.png, eda_boxplots.png,")
    print("                         eda_histograms.png, eda_corr_matrix.png")