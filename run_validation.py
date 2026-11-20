"""
COVID-19 Wave Analysis - extra validation, baselines and EDA
============================================================
Run AFTER run_pipeline.py (it needs the files in data/processed/):

    python run_validation.py

What it adds to the project
  A. Out-of-sample regression test with simple baselines
     (the main pipeline reports in-sample R2/RMSE only).
  B. Clustering checks: silhouette vs a random "null", stability under
     re-sampling, K-Means vs Ward agreement.
  C. Clustering on NORMALISED epidemic curves (each curve scaled to peak = 1),
     compared with the feature-based clusters.
  D. EDA figures (feature distributions, correlations).

Outputs
  data/processed/validation_holdout_metrics.csv
  data/processed/validation_holdout_summary.csv
  data/processed/validation_clustering_k_table.csv
  data/processed/validation_curve_clusters.csv
  data/processed/eda_feature_summary.csv
  data/processed/validation_summary.json
  figures/fig7_feature_distributions.png
  figures/fig8_feature_correlation.png
  figures/fig9_holdout_baselines.png
  figures/fig10_cluster_null_and_stability.png
  figures/fig11_normalised_curve_clusters.png
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

warnings.filterwarnings("ignore")

SEED = 42
BASE_DIR = Path(__file__).resolve().parent
PROC = BASE_DIR / "data" / "processed"
FIGS = BASE_DIR / "figures"
FIGS.mkdir(exist_ok=True)

WAVES = {
    "Wave 1": ("2020-05-01", "2020-12-31"),
    "Wave 2": ("2021-03-01", "2021-07-31"),
}
FEATURES = [
    "Wave1_Peak_Per_100k", "Wave2_Peak_Per_100k", "Wave1_Days_To_Peak",
    "Wave2_Days_To_Peak", "Delta_Surge_Ratio", "Total_Incidence_Per_100k",
    "Case_Fatality_Rate_Pct", "Dose1_Coverage_Pct", "Dose2_Coverage_Pct",
]
COLORS = ["#c2185b", "#e65100", "#1565c0", "#2e7d32", "#6a1b9a", "#455a64"]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def wave_series(panel, state, wave):
    """7-day average series of one state in one wave (same rules as the pipeline)."""
    start, end = pd.to_datetime(WAVES[wave][0]), pd.to_datetime(WAVES[wave][1])
    d = panel[(panel["State_Code"] == state) & (panel["Date"] >= start) & (panel["Date"] <= end)]
    d = d.sort_values("Date")
    if len(d) < 20:
        return None
    y = d["Confirmed_7D_Avg"].to_numpy(dtype=float)
    if y.max() <= 5:
        return None
    return y


def holdout_forecasts(y, frac=0.2):
    """Train on the first 80 % of a wave, predict the last 20 %. Returns dict model -> predictions."""
    n = len(y)
    n_test = max(5, int(round(n * frac)))
    n_train = n - n_test
    t = (np.arange(n) / float(n - 1)).reshape(-1, 1)
    t_tr, t_te = t[:n_train], t[n_train:]
    y_tr, y_te = y[:n_train], y[n_train:]
    cv = TimeSeriesSplit(n_splits=4)

    preds = {
        "Persistence (last value)": np.full(n_test, y_tr[-1]),
        "Training mean": np.full(n_test, y_tr.mean()),
    }

    # polynomial: degree 2-5 chosen by time-series cross-validation on the training part only
    best_deg, best_score = 2, np.inf
    for deg in (2, 3, 4, 5):
        model = make_pipeline(PolynomialFeatures(deg), LinearRegression())
        score = -cross_val_score(model, t_tr, y_tr, cv=cv, scoring="neg_root_mean_squared_error").mean()
        if score < best_score:
            best_deg, best_score = deg, score
    poly = make_pipeline(PolynomialFeatures(best_deg), LinearRegression()).fit(t_tr, y_tr)
    preds["Polynomial (CV degree)"] = np.clip(poly.predict(t_te), 0, None)

    # ridge: degree 4, alpha chosen by time-series cross-validation on the training part only
    grid = GridSearchCV(
        make_pipeline(PolynomialFeatures(4), Ridge()),
        {"ridge__alpha": np.logspace(-3, 3, 13)},
        cv=cv, scoring="neg_root_mean_squared_error",
    ).fit(t_tr, y_tr)
    preds["Ridge (degree 4, CV alpha)"] = np.clip(grid.predict(t_te), 0, None)
    return y_te, preds


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def norm_curve(y, m=40):
    """Resample a wave curve to m points and scale so its peak equals 1."""
    t = np.linspace(0, 1, len(y))
    tt = np.linspace(0, 1, m)
    c = np.interp(tt, t, y)
    return c / c.max()


def best_k_by_silhouette(X, ks=range(2, 7)):
    out = {}
    for k in ks:
        if k >= len(X):
            continue
        lab = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit_predict(X)
        out[k] = float(silhouette_score(X, lab))
    k_best = max(out, key=out.get)
    return k_best, out


def save_fig(fig, name):
    path = FIGS / name
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# --------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(SEED)
    panel = pd.read_csv(PROC / "cleaned_state_panel.csv", parse_dates=["Date"])
    feats = pd.read_csv(PROC / "state_wave_metrics.csv")
    results = {}

    # ===================================================================
    # A. Hold-out regression test with baselines
    # ===================================================================
    print("\n[A] Hold-out test: train on first 80% of each wave, predict last 20%")
    rows = []
    for wave in WAVES:
        for st in sorted(panel["State_Code"].unique()):
            if st in ("TT", "UN"):
                continue
            y = wave_series(panel, st, wave)
            if y is None:
                continue
            y_te, preds = holdout_forecasts(y)
            for name, p in preds.items():
                rows.append({"Wave": wave, "State_Code": st, "Model": name, "Test_Days": len(y_te),
                             "RMSE_test": rmse(y_te, p), "MAE_test": mae(y_te, p)})
    hold = pd.DataFrame(rows)
    hold.round(3).to_csv(PROC / "validation_holdout_metrics.csv", index=False)

    piv = hold.pivot_table(index=["Wave", "State_Code"], columns="Model", values="RMSE_test")
    persist = piv["Persistence (last value)"]
    summary_rows = []
    for wave in WAVES:
        for model in piv.columns:
            s = piv.loc[wave, model]
            p = persist.loc[wave]
            summary_rows.append({
                "Wave": wave, "Model": model, "States": int(s.notna().sum()),
                "Mean_RMSE": round(float(s.mean()), 2), "Median_RMSE": round(float(s.median()), 2),
                "States_better_than_persistence": int((s < p).sum()),
            })
    hold_sum = pd.DataFrame(summary_rows)
    hold_sum.to_csv(PROC / "validation_holdout_summary.csv", index=False)
    print(hold_sum.to_string(index=False))
    results["holdout_summary"] = hold_sum.to_dict(orient="records")

    # figure 9
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, wave in zip(axes, WAVES):
        sub = hold_sum[hold_sum["Wave"] == wave]
        ax.bar(range(len(sub)), sub["Median_RMSE"], color=COLORS[: len(sub)])
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels([m.replace(" (", "\n(") for m in sub["Model"]], fontsize=8)
        ax.set_ylabel("Median test RMSE (cases/day)")
        ax.set_title(f"{wave}: error on the held-out last 20% of days")
    fig.suptitle("Out-of-sample comparison with simple baselines (lower is better)", weight="bold")
    save_fig(fig, "fig9_holdout_baselines.png")

    # ===================================================================
    # B. Clustering checks
    # ===================================================================
    print("\n[B] Clustering checks")
    X = feats[FEATURES].to_numpy(dtype=float)
    Xs = StandardScaler().fit_transform(X)
    n = len(Xs)
    k_best, sil_by_k = best_k_by_silhouette(Xs)
    km_full = KMeans(n_clusters=k_best, random_state=SEED, n_init=20).fit(Xs)
    ward_full = AgglomerativeClustering(n_clusters=k_best, linkage="ward").fit(Xs)
    sil_obs = float(silhouette_score(Xs, km_full.labels_))
    sil_ward = float(silhouette_score(Xs, ward_full.labels_))
    ari_km_ward = float(adjusted_rand_score(km_full.labels_, ward_full.labels_))

    # null: shuffle every feature column independently (destroys the structure between features)
    B = 500
    null = np.empty(B)
    for b in range(B):
        Xp = np.column_stack([rng.permutation(X[:, j]) for j in range(X.shape[1])])
        Xp = StandardScaler().fit_transform(Xp)
        lab = KMeans(n_clusters=k_best, random_state=b, n_init=5).fit_predict(Xp)
        null[b] = silhouette_score(Xp, lab)
    p_null = float((np.sum(null >= sil_obs) + 1) / (B + 1))

    # stability: repeatedly cluster random 80 % subsets, compare with the full-data clusters
    stab = {}
    for k in range(2, 7):
        if k >= n:
            continue
        full = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit_predict(Xs)
        aris = []
        for r in range(100):
            idx = rng.choice(n, size=int(0.8 * n), replace=False)
            sub = KMeans(n_clusters=k, random_state=r, n_init=5).fit_predict(Xs[idx])
            aris.append(adjusted_rand_score(full[idx], sub))
        stab[k] = float(np.mean(aris))

    ktab = pd.DataFrame({
        "k": list(sil_by_k.keys()),
        "KMeans_silhouette": [round(v, 4) for v in sil_by_k.values()],
        "Stability_ARI_mean": [round(stab[k], 3) for k in sil_by_k.keys()],
    })
    ktab.to_csv(PROC / "validation_clustering_k_table.csv", index=False)
    print(ktab.to_string(index=False))
    print(f"  chosen k = {k_best}; K-Means silhouette = {sil_obs:.4f}; Ward silhouette = {sil_ward:.4f}")
    print(f"  random-null silhouette: mean {null.mean():.3f}, 95th pct {np.percentile(null, 95):.3f}; "
          f"p = {p_null:.3f}")
    print(f"  agreement K-Means vs Ward (ARI) = {ari_km_ward:.3f}")
    sizes = np.bincount(km_full.labels_).tolist()
    results["clustering_checks"] = {
        "k": int(k_best), "kmeans_silhouette": round(sil_obs, 4), "ward_silhouette": round(sil_ward, 4),
        "null_silhouette_mean": round(float(null.mean()), 4),
        "null_silhouette_p95": round(float(np.percentile(null, 95)), 4),
        "p_value_vs_null": round(p_null, 4), "ari_kmeans_vs_ward": round(ari_km_ward, 4),
        "stability_ari_by_k": {str(k): round(v, 3) for k, v in stab.items()},
        "cluster_sizes": sizes,
    }

    # figure 10
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(null, bins=30, color="#90a4ae")
    axes[0].axvline(sil_obs, color="#c2185b", lw=2.5, label=f"Real data = {sil_obs:.3f}")
    axes[0].set_xlabel("Silhouette score")
    axes[0].set_ylabel("Number of shuffled datasets")
    axes[0].set_title(f"Real silhouette vs {B} shuffled (null) datasets, k={k_best}")
    axes[0].legend()
    axes[1].bar([str(k) for k in stab], list(stab.values()), color="#1565c0")
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel("Number of clusters k")
    axes[1].set_ylabel("Mean stability (Adjusted Rand Index)")
    axes[1].set_title("Stability under 80% re-sampling (1 = identical groups)")
    save_fig(fig, "fig10_cluster_null_and_stability.png")

    # ===================================================================
    # C. Clustering on normalised curves
    # ===================================================================
    print("\n[C] Clustering on normalised epidemic curves")
    curves, kept = [], []
    for st in feats["State_Code"]:
        y1, y2 = wave_series(panel, st, "Wave 1"), wave_series(panel, st, "Wave 2")
        if y1 is None or y2 is None:
            continue
        curves.append(np.concatenate([norm_curve(y1), norm_curve(y2)]))
        kept.append(st)
    C = np.array(curves)
    kc, sil_c = best_k_by_silhouette(C)
    km_c = KMeans(n_clusters=kc, random_state=SEED, n_init=20).fit(C)
    ward_c = AgglomerativeClustering(n_clusters=kc, linkage="ward").fit(C)
    f_idx = feats.set_index("State_Code").loc[kept]
    feat_labels = KMeans(n_clusters=k_best, random_state=SEED, n_init=20).fit(
        StandardScaler().fit_transform(f_idx[FEATURES].to_numpy(dtype=float))).labels_
    ari_curve_feat = float(adjusted_rand_score(km_c.labels_, feat_labels))
    out_c = pd.DataFrame({
        "State_Code": kept, "Curve_KMeans": km_c.labels_, "Curve_Ward": ward_c.labels_,
        "Feature_KMeans": feat_labels,
        "Dose1_Coverage_Pct": f_idx["Dose1_Coverage_Pct"].values,
        "Dose2_Coverage_Pct": f_idx["Dose2_Coverage_Pct"].values,
    })
    out_c.to_csv(PROC / "validation_curve_clusters.csv", index=False)
    prof = out_c.groupby("Curve_KMeans").agg(
        States=("State_Code", "count"), Mean_Dose1=("Dose1_Coverage_Pct", "mean"),
        Mean_Dose2=("Dose2_Coverage_Pct", "mean")).round(2)
    print(f"  curve-based k = {kc}; silhouette = {sil_c[kc]:.4f}; "
          f"agreement with feature clusters (ARI) = {ari_curve_feat:.3f}")
    print(prof.to_string())
    results["curve_clustering"] = {
        "states": len(kept), "k": int(kc), "silhouette": round(sil_c[kc], 4),
        "ari_vs_feature_clusters": round(ari_curve_feat, 4),
        "profile": prof.reset_index().to_dict(orient="records"),
    }

    # figure 11
    fig, axes = plt.subplots(2, kc, figsize=(4 * kc, 6.5), sharey=True, squeeze=False)
    tt = np.linspace(0, 1, 40)
    for c in range(kc):
        members = np.where(km_c.labels_ == c)[0]
        for r, name in enumerate(["Wave 1", "Wave 2"]):
            ax = axes[r, c]
            block = C[members, r * 40:(r + 1) * 40]
            for row in block:
                ax.plot(tt, row, color="#90a4ae", lw=0.8)
            ax.plot(tt, block.mean(axis=0), color=COLORS[c % len(COLORS)], lw=3)
            ax.set_title(f"Curve cluster {c} ({len(members)} states) - {name}", fontsize=9)
            if c == 0:
                ax.set_ylabel("Cases / peak")
            if r == 1:
                ax.set_xlabel("Time within wave (0-1)")
    fig.suptitle("Clusters of normalised epidemic curves (each curve scaled to peak = 1)", weight="bold")
    save_fig(fig, "fig11_normalised_curve_clusters.png")

    # ===================================================================
    # D. EDA figures
    # ===================================================================
    print("\n[D] EDA figures")
    feats[FEATURES].describe().T.round(3).to_csv(PROC / "eda_feature_summary.csv")

    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    for ax, f in zip(axes.ravel(), FEATURES):
        ax.hist(feats[f].dropna(), bins=12, color="#1565c0", edgecolor="white")
        ax.set_title(f.replace("_", " "), fontsize=9)
    fig.suptitle("Distribution of the nine state-level features (35 states/UTs)", weight="bold")
    fig.tight_layout()
    save_fig(fig, "fig7_feature_distributions.png")

    corr = feats[FEATURES].corr()
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    labels = [f.replace("_Per_100k", "").replace("_Pct", "").replace("_", " ") for f in FEATURES]
    ax.set_xticks(range(len(FEATURES)))
    ax.set_yticks(range(len(FEATURES)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(len(FEATURES)):
        for j in range(len(FEATURES)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Pearson correlation between the nine features", weight="bold")
    save_fig(fig, "fig8_feature_correlation.png")

    with open(PROC / "validation_summary.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nDone. Numbers for the report are in {PROC / 'validation_summary.json'}")


if __name__ == "__main__":
    main()
