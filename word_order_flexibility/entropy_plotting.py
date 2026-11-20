"""The four Delta H plots used in Figures 6–8 of the paper.

Plotting only: estimates come from saved, paper-aligned UD experiments.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .languages import FUTRELL_LANGUAGE_NAMES
from .plotting import (
    DECAYS,
    GREEN,
    RED,
    TTR,
    _save,
    _set_style,
    _validation_panel,
    _blank_correlation_diagonal,
    bakker_figures,
    correlation,
)

METRIC = "entropy_difference"
LABEL = r"$\Delta H$ (bits/token)"
PAPER_ROOT = Path("results/entropy/ud-paper-aligned")
FIGURE_STEMS = (
    "entropy_futrell_validation_hde",
    "bakker_metric_correlations",
    "entropy_stability_value_changes",
    "entropy_stability_triangular",
)


def load_entropy_ud(ud_csv: str, entropy_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Join by language, rejecting duplicates and incompatible token counts."""
    ud = pd.read_csv(ud_csv)
    if ud.Language.duplicated().any():
        raise ValueError("UD input must have exactly one row per language")
    full, sizes, sources = [], [], []
    for path in sorted(Path(entropy_dir).glob("*.json")):
        result = json.loads(path.read_text())
        if "full" not in result or "language" not in result:
            continue
        if result.get("corpus") != "ud":
            raise ValueError(f"Expected UD entropy results: {path}")
        score = result["full"]
        full.append(
            {
                "Language": result["language"],
                METRIC: score["difference"]["mean"],
                "entropy_intact": score["intact"]["mean"],
                "entropy_shuffled": score["within_shuffled"]["mean"],
                "entropy_mc_se": score["difference"]["mc_se"],
                "entropy_tokens": score["tokens"],
                "entropy_sentences": score["sentences"],
                "entropy_half_full_pass": result["diagnostics"]["difference"]["passes_half_full"],
                "entropy_input_sha256": result["input_sha256"],
                "entropy_preprocessing_version": result.get("preprocessing", {}).get("version"),
            }
        )
        for point in result["size_convergence"]:
            sizes.append(
                {
                    "Language": result["language"],
                    "fraction": point["sentences"] / score["sentences"],
                    "sentences": point["sentences"],
                    "tokens": point["tokens"],
                    **{key: point[key]["mean"] for key in ("intact", "within_shuffled", "difference")},
                }
            )
        sources.append(str(path.resolve()))
    if not full:
        raise ValueError("No per-language UD entropy JSON files found")
    entropy = pd.DataFrame(full)
    if entropy.Language.duplicated().any():
        raise ValueError("Duplicate entropy languages")
    merged = ud.merge(entropy, on="Language", how="inner", validate="one_to_one")
    if len(merged) < 3:
        raise ValueError("Need at least three matching languages")
    bad = merged.loc[merged.word_tokens != merged.entropy_tokens, "Language"].tolist()
    if bad:
        raise ValueError(f"UD/entropy token counts differ: {bad}")
    if "input_sha256" in merged.columns:
        wrong = merged.loc[merged.input_sha256 != merged.entropy_input_sha256, "Language"].tolist()
        if wrong:
            raise ValueError(f"UD/entropy input hashes differ: {wrong}")
    if "preprocessing_version" in merged.columns:
        wrong = merged.loc[merged.preprocessing_version != merged.entropy_preprocessing_version, "Language"].tolist()
        if wrong:
            raise ValueError(f"UD/entropy preprocessing differs: {wrong}")
    numeric = [
        "HDE",
        "ROE",
        TTR,
        METRIC,
        *[f"F_lambda_{int(d * 100):03d}" for d in DECAYS],
    ]
    if not np.isfinite(merged[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Analysis columns must contain finite numeric values")
    metadata = {
        "ud_csv": str(Path(ud_csv).resolve()),
        "entropy_files": sources,
        "matched_languages": merged.Language.tolist(),
        "missing_entropy": sorted(set(ud.Language) - set(entropy.Language)),
        "entropy_only": sorted(set(entropy.Language) - set(ud.Language)),
        "sample_check": (
            "Language, token count, input hash, and preprocessing version"
            if "input_sha256" in ud
            else "Language and token counts only; legacy CSV has no hashes"
        ),
        "direction": "Delta H = H(intact) - H(within-sentence shuffled); paper definition.",
        "controlled_spearman": "Spearman correlation of linear TTR residuals, matching existing analyses.",
        "futrell_subset": "Futrell language names on broad UD samples, not dedicated Futrell treebanks.",
        "half_full_pass_count": int(merged.entropy_half_full_pass.eq(True).sum()),
    }
    sizes = pd.DataFrame(sizes)
    return merged, sizes[sizes.Language.isin(merged.Language)], metadata


def _load_stability(path: str | Path, languages) -> pd.DataFrame:
    wide = pd.read_csv(path, index_col="language")[["70", "80", "90", "100"]]
    if wide.index.duplicated().any() or not np.isfinite(wide.to_numpy()).all():
        raise ValueError("Need unique languages and complete finite scores at all four sizes")
    if set(wide.index) != set(languages):
        raise ValueError("Stability and validation must contain the same languages")
    return wide


def _plot_hde(frame, subset, output):
    fig, ax = plt.subplots(figsize=(8, 6.8))
    _validation_panel(ax, frame, subset, "HDE", metric=METRIC, metric_label=LABEL)
    # Preserve the typography of the entropy figure independently of main's
    # smaller figures for other metrics.
    ax.xaxis.label.set_fontsize(22)
    ax.yaxis.label.set_fontsize(22)
    ax.tick_params(labelsize=15)
    for annotation in ax.texts:
        annotation.set_fontsize(18)
    handles, labels = ax.get_legend_handles_labels()
    order = (0, 2, 1, 3)
    ax.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        borderaxespad=0,
        frameon=True,
        framealpha=1,
        fontsize=15,
        columnspacing=0.9,
        labelspacing=0.45,
        handlelength=1.6,
        handletextpad=0.45,
        borderpad=0.45,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    _save(fig, output / FIGURE_STEMS[0])


def _plot_stability_values(wide, output, use_tex=False):
    x = [70, 80, 90, 100]
    labels = [rf"{p}\%" if use_tex else f"{p}%" for p in x]
    change = wide.sub(wide["70"], axis=0).abs()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    for ax, values, ylabel, title in [
        (axes[0], wide, r"Entropy contrast $\Delta H$ (bits/token)", "Entropy scores"),
        (axes[1], change, r"$|\Delta H_p-\Delta H_{70\%}|$ (bits/token)", "Absolute change from 70 percent of data"),
    ]:
        for row in values.to_numpy():
            ax.plot(x, row, color=GREEN, linewidth=0.8, alpha=0.18)
        ax.plot(x, values.median().to_numpy(), color=GREEN, marker="o", linewidth=2.5, label="Across-language median")
        ax.plot(
            x,
            values.mean().to_numpy(),
            color=RED,
            marker="D",
            linestyle="--",
            linewidth=2.2,
            label="Across-language mean",
        )
        ax.set(xlabel="Available sentences used", ylabel=ylabel, title=title)
        ax.xaxis.label.set_fontsize(22)
        ax.yaxis.label.set_fontsize(22)
        ax.set_xticks(x, labels)
        if ax is axes[1]:
            ax.set_ylim(bottom=0)
        else:
            ax.margins(y=0.05)
        ax.tick_params(labelsize=14)
        ax.grid(alpha=0.25)
    axes[0].legend(loc="lower right", fontsize=12, frameon=True)
    axes[1].legend(loc="upper left", fontsize=12, frameon=True)
    fig.tight_layout()
    _save(fig, output / FIGURE_STEMS[2])
    wide.to_csv(output / "entropy_stability_scores.csv")
    change.to_csv(output / "entropy_absolute_changes_from_70.csv")
    pd.DataFrame(
        {
            "percent": x,
            "mean_score": wide.mean().to_numpy(),
            "median_score": wide.median().to_numpy(),
            "mean_absolute_change_from_70": change.mean().to_numpy(),
            "median_absolute_change_from_70": change.median().to_numpy(),
        }
    ).to_csv(output / "entropy_stability_value_summary.csv", index=False)


def generate_entropy_artifacts(
    ud_csv: str = str(PAPER_ROOT / "metrics.csv"),
    entropy_dir: str = str(PAPER_ROOT / "curated"),
    bakker_csv: str = "data/bakker-flexibility-metrics.csv",
    output_dir: str = str(PAPER_ROOT / "paper-figures"),
    *,
    use_tex: bool = False,
    stability_csv: str | None = None,
) -> None:
    """Render the paper's validation, Bakker, and two 70–100% stability figures."""
    frame, _, metadata = load_entropy_ud(ud_csv, entropy_dir)
    if metadata["missing_entropy"] or metadata["entropy_only"]:
        raise ValueError("UD metrics and entropy results must cover exactly the same languages")
    stability_path = (
        Path(stability_csv) if stability_csv else Path(entropy_dir).parent / "stability" / "difference_scores_wide.csv"
    )
    wide = _load_stability(stability_path, frame.Language)
    subset = frame[frame.Language.isin(FUTRELL_LANGUAGE_NAMES)]
    if len(subset) < 3:
        raise ValueError("Need at least three Futrell languages for the validation overlay")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "language_metrics.csv", index=False)
    correlations = []
    for group, data in [("all_ud", frame), ("futrell_language_subset", subset)]:
        for target in ["HDE", "ROE", *[f"F_lambda_{int(d * 100):03d}" for d in DECAYS]]:
            for controlled in [False, True]:
                correlations.append({"sample": group, **correlation(data, METRIC, target, controlled)})
    pd.DataFrame(correlations).to_csv(output / "entropy_correlations.csv", index=False)
    with plt.rc_context():
        _set_style()
        plt.rcParams["text.usetex"] = use_tex
        _plot_hde(frame, subset, output)
        bakker_figures(
            frame,
            pd.read_csv(bakker_csv),
            output,
            metric=METRIC,
            metric_label=LABEL,
            extra_metrics={r"$\Delta H$": METRIC},
            matrix_only=True,
        )
        _plot_stability_values(wide, output, use_tex)
        _plot_stability_correlations(wide, output, use_tex)
    metadata.update(
        futrell_languages=subset.Language.tolist(),
        stability_csv=str(stability_path.resolve()),
        figure_files=[f"{stem}.{extension}" for stem in FIGURE_STEMS for extension in ("pdf", "png")],
        stability_note="70/80/90/100% nested samples; 100% reuses 20 of the 100 saved trials, not their 100-trial mean.",
    )
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def _plot_stability_correlations(wide, output, use_tex=False):
    labels = [rf"{p}\%" if use_tex else f"{p}%" for p in (70, 80, 90, 100)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    for ax, method, label in zip(axes, ("pearson", "spearman"), (r"Pearson $r$", r"Spearman $\rho$")):
        matrix = wide.corr(method=method)
        matrix.to_csv(output / f"entropy_stability_{method}.csv")
        matrix.index = matrix.columns = labels
        sns.heatmap(
            matrix,
            mask=np.triu(np.ones_like(matrix, dtype=bool)),
            annot=False,
            vmin=-1,
            vmax=1,
            center=0,
            cmap=sns.diverging_palette(10, 130, s=65, l=55, as_cmap=True),
            linewidths=0.7,
            linecolor="white",
            square=True,
            ax=ax,
            cbar_kws={
                "label": label,
                "orientation": "horizontal",
                "location": "bottom",
                "shrink": 0.72,
                "pad": 0.09,
                "ticks": [-1, -0.5, 0, 0.5, 1],
            },
        )
        _blank_correlation_diagonal(ax, len(matrix))
        for row in range(4):
            for col in range(row):
                ax.text(
                    col + 0.5,
                    row + 0.53,
                    f"{matrix.iloc[row, col]:.3f}",
                    ha="center",
                    va="center",
                    fontsize=18,
                    fontweight="bold",
                    color="#1A1A1A",
                )
        ax.set(xlabel="", ylabel="", title=label)
        ax.tick_params(axis="both", labelsize=18)
        ax.grid(False)
        bar = ax.collections[0].colorbar
        bar.ax.tick_params(labelsize=13)
        bar.set_label(label, fontsize=18)
    fig.tight_layout()
    _save(fig, output / FIGURE_STEMS[3])
