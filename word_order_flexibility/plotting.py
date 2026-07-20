from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from .analysis import residualize

GREEN = "#1B7837"
RED = "#8E1B1B"
SLATE = "#4C566A"
MID_GRAY = "#9A9A9A"
LIGHT_GRAY = "#F2F2F2"
AXIS_LABEL_SIZE = 22
TICK_LABEL_SIZE = 15
ANNOTATION_SIZE = 18
LEGEND_SIZE = 18
DECAYS = (0.25, 0.5, 0.75, 1.0)
SIZES = (1000, 2000, 5000, 10000)
FLEX = "F_lambda_050"
TTR = "type_token_ratio"


FUTRELL_HDE_ORDER = (
    "Ancient Greek",
    "Latin",
    "Basque",
    "Croatian",
    "Slovak",
    "German",
    "Estonian",
    "Russian",
    "Slovenian",
    "Danish",
    "Hungarian",
    "Persian",
    "Czech",
    "Hebrew",
    "Finnish",
    "Dutch",
    "Spanish",
    "Modern Greek",
    "Catalan",
    "Bulgarian",
    "Romanian",
    "Swedish",
    "Portuguese",
    "Italian",
    "French",
    "Japanese",
    "Arabic",
    "Turkish",
    "English",
    "Hindi",
    "Irish",
    "Telugu",
    "Tamil",
)
FUTRELL_ROE_ORDER = (
    "Latin",
    "Basque",
    "Ancient Greek",
    "Estonian",
    "Telugu",
    "Russian",
    "Bulgarian",
    "Persian",
    "Romanian",
    "Turkish",
    "Hebrew",
    "Croatian",
    "German",
    "Japanese",
    "Finnish",
    "Spanish",
    "Slovak",
    "Czech",
    "Danish",
    "Modern Greek",
    "Arabic",
    "Catalan",
    "Tamil",
    "Portuguese",
    "Slovenian",
    "Hungarian",
    "Italian",
    "Dutch",
    "French",
    "Hindi",
    "Swedish",
    "Irish",
    "English",
)


def _set_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
            "font.size": 14,
            "axes.labelsize": AXIS_LABEL_SIZE,
            "axes.titlesize": 18,
            "axes.labelweight": "normal",
            "xtick.labelsize": TICK_LABEL_SIZE,
            "ytick.labelsize": TICK_LABEL_SIZE,
            "legend.fontsize": LEGEND_SIZE,
            "figure.titlesize": 20,
            "legend.framealpha": 1.0,
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{newtxtext}",
        }
    )


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def correlation(frame: pd.DataFrame, x: str, y: str, controlled: bool) -> dict:
    data = frame[[x, y, TTR]].replace([np.inf, -np.inf], np.nan).dropna()
    xv, yv = data[x].to_numpy(), data[y].to_numpy()
    if controlled:
        xv, yv = residualize(data[x], data[TTR]), residualize(data[y], data[TTR])
    pearson = stats.pearsonr(xv, yv)
    spearman = stats.spearmanr(xv, yv)
    return {
        "x": x,
        "y": y,
        "control": "TTR" if controlled else "none",
        "n": len(data),
        "pearson_r": pearson.statistic,
        "pearson_p": pearson.pvalue,
        "spearman_rho": spearman.statistic,
        "spearman_p": spearman.pvalue,
    }


def _format_p(value: float) -> str:
    if value < 0.001:
        return r"$p < 0.001$"
    if value < 0.01:
        return r"$p < 0.01$"
    if value < 0.05:
        return r"$p < 0.05$"
    return r"$p \geq 0.05$"


def _regression(ax, x, y, color, label) -> None:
    ax.scatter(x, y, color=color, alpha=0.75, s=60, edgecolor="white", linewidth=0.4)
    slope, intercept = np.polyfit(x, y, 1)
    grid = np.linspace(min(x), max(x), 100)
    ax.plot(grid, slope * grid + intercept, color=color, linewidth=2.3, label=label)


def _validation_panel(ax, all_results: pd.DataFrame, futrell_results: pd.DataFrame, target: str) -> None:
    data = all_results[["Language", FLEX, target, TTR]].dropna().copy()
    data["x"] = residualize(data[target], data[TTR])
    data["y"] = residualize(data[FLEX], data[TTR])
    futrell_languages = set(futrell_results["Language"])
    selected = all_results[all_results["Language"].isin(futrell_languages)][
        ["Language", FLEX, target, TTR]
    ].dropna().copy()
    selected["x"] = residualize(selected[target], selected[TTR])
    selected["y"] = residualize(selected[FLEX], selected[TTR])
    other = data[~data["Language"].isin(futrell_languages)]

    ax.scatter(
        other["x"],
        other["y"],
        s=58,
        color=MID_GRAY,
        alpha=0.42,
        edgecolor="white",
        linewidth=0.35,
        label=rf"Other UD languages ($n={len(other)}$)",
        zorder=1,
    )
    ax.scatter(
        selected["x"],
        selected["y"],
        s=72,
        color=RED,
        alpha=0.78,
        edgecolor="white",
        linewidth=0.45,
        label=rf"Futrell languages ($n={len(selected)}$)",
        zorder=2,
    )
    for subset, color, label in (
        (data, GREEN, rf"All UD fit ($n={len(data)}$)"),
        (selected, RED, rf"Futrell fit ($n={len(selected)}$)"),
    ):
        slope, intercept = np.polyfit(subset["x"], subset["y"], 1)
        grid = np.linspace(subset["x"].min(), subset["x"].max(), 100)
        linewidth = 3.4 if label.startswith("All UD") else 2.0
        zorder = 3 if label.startswith("All UD") else 4
        ax.plot(
            grid,
            slope * grid + intercept,
            color=color,
            linewidth=linewidth,
            label=label,
            zorder=zorder,
        )

    all_r = stats.pearsonr(data["x"], data["y"])
    futrell_r = stats.pearsonr(selected["x"], selected["y"])
    ax.text(
        0.96,
        0.05,
        f"All UD: $r={all_r.statistic:.3f}$, {_format_p(all_r.pvalue)}\n"
        f"Futrell: $r={futrell_r.statistic:.3f}$, {_format_p(futrell_r.pvalue)}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        multialignment="left",
        fontsize=ANNOTATION_SIZE,
        linespacing=1.25,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": LIGHT_GRAY,
            "edgecolor": "0.70",
            "alpha": 0.94,
        },
    )
    ax.set_xlabel(rf"$\mathrm{{{target}}} \mid \mathrm{{TTR}}$ (resid.)", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel(r"$F^{(.5)} \mid \mathrm{TTR}$ (resid.)", fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    ax.grid(True, alpha=0.3)


def plot_validation(all_results: pd.DataFrame, futrell_results: pd.DataFrame, output: Path) -> None:
    for target in ("HDE", "ROE"):
        fig, ax = plt.subplots(figsize=(8, 6.8))
        _validation_panel(ax, all_results, futrell_results, target)
        handles, labels = ax.get_legend_handles_labels()
        order = (0, 2, 1, 3)
        ax.legend(
            [handles[index] for index in order],
            [labels[index] for index in order],
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=2,
            borderaxespad=0,
            frameon=True,
            framealpha=1.0,
            fontsize=15,
            columnspacing=0.9,
            labelspacing=0.45,
            handlelength=1.6,
            handletextpad=0.45,
            borderpad=0.45,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.84))
        _save(fig, output / f"word_futrell_validation_decay_{target.lower()}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.8), sharey=True)
    for ax, target in zip(axes, ("HDE", "ROE"), strict=True):
        _validation_panel(ax, all_results, futrell_results, target)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.52, 1.02),
        ncol=4,
        frameon=True,
        framealpha=1.0,
        fontsize=LEGEND_SIZE,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, output / "word_futrell_validation_decay")


def lambda_analysis(results: pd.DataFrame, output: Path) -> pd.DataFrame:
    rows = []
    for decay in DECAYS:
        metric = f"F_lambda_{int(decay * 100):03d}"
        for target in ("HDE", "ROE"):
            for controlled in (False, True):
                rows.append({"lambda": decay, **correlation(results, metric, target, controlled)})
    table = pd.DataFrame(rows)
    table.to_csv(output / "soft_lambda_correlations.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 6))
    specs = (
        ("HDE", "none", RED, "o", "--", "white", "HDE"),
        ("HDE", "TTR", RED, "o", "-", None, r"$\mathrm{HDE} \mid \mathrm{TTR}$"),
        ("ROE", "none", GREEN, "D", "--", "white", "ROE"),
        ("ROE", "TTR", GREEN, "D", "-", None, r"$\mathrm{ROE} \mid \mathrm{TTR}$"),
    )
    for target, control, color, marker, linestyle, fill, label in specs:
        line = table[(table.y == target) & (table.control == control)].sort_values("lambda")
        ax.plot(
            line["lambda"],
            line.pearson_r,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=2.2,
            markersize=7,
            markerfacecolor=color if fill is None else fill,
            markeredgecolor=color,
            markeredgewidth=1.4,
            label=label,
        )
    ax.axhline(0, color="black", linewidth=1, alpha=0.45)
    ax.margins(x=0.08, y=0.12)
    ax.set_xlabel(r"Distance decay $\lambda$", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel(r"Pearson / partial $r$", fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, loc="lower left", fontsize=LEGEND_SIZE, framealpha=1.0)
    fig.tight_layout()
    _save(fig, output / "word_soft_lambda_profile")

    fig, ax = plt.subplots(figsize=(10, 6))
    columns = [f"F_lambda_{int(decay * 100):03d}" for decay in DECAYS]
    for _, row in results.sort_values("Language").iterrows():
        ax.plot(DECAYS, row[columns], color=MID_GRAY, alpha=0.22, linewidth=0.9, zorder=1)
    ax.plot(
        DECAYS,
        results[columns].median(),
        color=GREEN,
        marker="s",
        linewidth=2.8,
        markersize=7,
        label="Median",
        zorder=3,
    )
    ax.plot(
        DECAYS,
        results[columns].mean(),
        color=RED,
        marker="o",
        markerfacecolor="white",
        markeredgecolor=RED,
        markeredgewidth=1.5,
        linewidth=2.4,
        markersize=7,
        label="Mean",
        zorder=3,
    )
    ax.set_xlabel(r"Distance decay $\lambda$", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel(r"Flexibility $F^{(\lambda)}$", fontsize=AXIS_LABEL_SIZE)
    ax.set_xticks(DECAYS)
    ax.set_xticklabels([r"$0.25$", r"$0.5$", r"$0.75$", r"$1.0$"])
    ax.set_ylim(0, 1)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=LEGEND_SIZE, framealpha=1.0)
    fig.tight_layout()
    _save(fig, output / "word_language_soft_lambda_profiles")
    return table


def _bh(values) -> list[float]:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    ranked = values[order] * len(values) / np.arange(1, len(values) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1)
    result = np.empty_like(ranked)
    result[order] = ranked
    return result.tolist()


def _stars(value: float) -> str:
    if value < 0.001:
        return "***"
    if value < 0.01:
        return "**"
    if value < 0.05:
        return "*"
    return ""


def bakker_figures(results: pd.DataFrame, bakker: pd.DataFrame, output: Path) -> None:
    bakker = bakker.copy()
    bakker["Language"] = bakker["Language"].replace({"Greek": "Modern Greek"})
    merged = results.merge(bakker[["Language", "Flexibility"]], on="Language").rename(columns={"Flexibility": "Bakker"})
    metrics = {
        "Bakker": "Bakker",
        "HDE": "HDE",
        "ROE": "ROE",
        **{rf"$F^{{({decay:g})}}$": f"F_lambda_{int(decay * 100):03d}" for decay in DECAYS},
    }
    labels = list(metrics)
    matrix = merged[list(metrics.values())].corr(method="pearson")
    matrix.index = matrix.columns = labels

    tests = []
    for index, left in enumerate(labels):
        for right in labels[:index]:
            test = stats.pearsonr(merged[metrics[left]], merged[metrics[right]])
            tests.append(
                {
                    "left": left,
                    "right": right,
                    "pearson_r": test.statistic,
                    "p": test.pvalue,
                    "n": len(merged),
                }
            )
    for row, value in zip(tests, _bh([row["p"] for row in tests]), strict=True):
        row["q_bh"] = value
    pd.DataFrame(tests).to_csv(output / "bakker_metric_correlations.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 8))
    mask = np.triu(np.ones_like(matrix, dtype=bool), 1)
    sns.heatmap(
        matrix,
        mask=mask,
        annot=False,
        vmin=-1,
        vmax=1,
        center=0,
        cmap=sns.diverging_palette(10, 130, s=65, l=55, as_cmap=True),
        linewidths=0.7,
        linecolor="white",
        square=True,
        cbar_kws={
            "label": r"Pearson $r$",
            "orientation": "horizontal",
            "location": "bottom",
            "shrink": 0.72,
            "pad": 0.08,
        },
        ax=ax,
    )
    q_values = {(row["left"], row["right"]): row["q_bh"] for row in tests}
    for row_index, left in enumerate(labels):
        for column_index, right in enumerate(labels):
            if column_index > row_index:
                continue
            value = matrix.loc[left, right]
            ax.text(
                column_index + 0.5,
                row_index + 0.53,
                "--" if pd.isna(value) else f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=18,
                fontweight="bold",
                color="#1A1A1A",
            )
            q_value = q_values.get((left, right), q_values.get((right, left), float("nan")))
            star = "" if pd.isna(q_value) else _stars(q_value)
            if star:
                ax.text(
                    column_index + 0.78,
                    row_index + 0.23,
                    star,
                    ha="right",
                    va="top",
                    fontsize=8.5,
                    fontweight="bold",
                    color="#1A1A1A",
                )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="both", labelsize=18)
    ax.grid(False)
    colorbar = ax.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=15)
    colorbar.set_label(r"Pearson $r$", fontsize=18, fontweight="normal")
    fig.tight_layout()
    _save(fig, output / "bakker_metric_correlations")

    pearson = stats.pearsonr(merged["Bakker"], merged[FLEX])
    spearman = stats.spearmanr(merged["Bakker"], merged[FLEX])
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.regplot(
        data=merged,
        x="Bakker",
        y=FLEX,
        scatter_kws={
            "s": 60,
            "alpha": 0.75,
            "facecolor": GREEN,
            "edgecolor": "white",
            "linewidths": 0.4,
        },
        line_kws={"color": RED, "linewidth": 2.2},
        ci=95,
        color=RED,
        ax=ax,
    )
    ax.set_xlabel("Bakker flexibility rating", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel(r"Flexibility $F^{(.5)}$", fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    ax.grid(True, alpha=0.3)
    ax.text(
        0.03,
        0.97,
        f"Pearson $r={pearson.statistic:.2f}$, {_format_p(pearson.pvalue)}\n"
        f"Spearman $\\rho={spearman.statistic:.2f}$, {_format_p(spearman.pvalue)}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": LIGHT_GRAY, "edgecolor": "0.75"},
    )
    fig.tight_layout()
    _save(fig, output / "bakker_vs_decay_flexibility")


def _format_sentence_axis(ax) -> None:
    ax.set_xscale("log")
    ax.set_xticks(SIZES)
    ax.set_xticklabels([f"{size:,}" for size in SIZES])
    ax.set_xlabel("Number of sentences", fontsize=AXIS_LABEL_SIZE)


def sentence_size_figures(results: pd.DataFrame, output: Path) -> None:
    rows = []
    for size, subset in results.groupby("n_sentences"):
        for target in ("HDE", "ROE"):
            for controlled in (False, True):
                rows.append({"n_sentences": size, **correlation(subset, FLEX, target, controlled)})
        for decay in DECAYS:
            metric = f"F_lambda_{int(decay * 100):03d}"
            for target in ("HDE", "ROE"):
                rows.append(
                    {
                        "n_sentences": size,
                        "lambda": decay,
                        **correlation(subset, metric, target, True),
                    }
                )
    table = pd.DataFrame(rows)
    table.to_csv(output / "sentence_size_correlations.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    specs = (
        ("HDE", "none", RED, "o", "--", "white", "HDE"),
        ("HDE", "TTR", RED, "o", "-", None, r"$\mathrm{HDE} \mid \mathrm{TTR}$"),
        ("ROE", "none", GREEN, "D", "--", "white", "ROE"),
        ("ROE", "TTR", GREEN, "D", "-", None, r"$\mathrm{ROE} \mid \mathrm{TTR}$"),
    )
    base = table[table["lambda"].isna()]
    for target, control, color, marker, linestyle, fill, label in specs:
        line = base[(base.y == target) & (base.control == control)].sort_values("n_sentences")
        ax.plot(
            line.n_sentences,
            line.pearson_r,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=2.4,
            markersize=7,
            markerfacecolor=color if fill is None else fill,
            markeredgecolor=color,
            markeredgewidth=1.5,
            label=label,
        )
    ax.axhline(0, color="black", linewidth=1, alpha=0.5)
    _format_sentence_axis(ax)
    ax.set_ylabel(r"Pearson / partial $r$", fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=LEGEND_SIZE, framealpha=1.0)
    fig.tight_layout()
    _save(fig, output / "common_language_correlations_by_sentence_size")

    largest = results[results.n_sentences == max(SIZES)][["Language", FLEX]]
    stability = []
    for size, subset in results.groupby("n_sentences"):
        joined = subset.merge(largest, on="Language", suffixes=("", "_largest"))
        stability.append(
            {
                "n_sentences": size,
                "pearson_r": stats.pearsonr(joined[FLEX], joined[FLEX + "_largest"]).statistic,
                "spearman_rho": stats.spearmanr(joined[FLEX], joined[FLEX + "_largest"]).statistic,
                "n_languages": len(joined),
            }
        )
    stability = pd.DataFrame(stability)
    stability.to_csv(output / "size_stability.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        stability.n_sentences,
        stability.pearson_r,
        color=RED,
        marker="o",
        linewidth=2.4,
        markersize=7,
        label=r"Pearson $r$",
    )
    ax.plot(
        stability.n_sentences,
        stability.spearman_rho,
        color=GREEN,
        marker="s",
        linewidth=2.4,
        markersize=7,
        label=r"Spearman $\rho$",
    )
    _format_sentence_axis(ax)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Correlation with largest sample", fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=LEGEND_SIZE, framealpha=1.0)
    fig.tight_layout()
    _save(fig, output / "flexibility_stability_by_sentence_size")

    profile = table[table["lambda"].notna()]
    alphas = np.linspace(0.38, 1.0, len(SIZES))
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    for size, alpha in zip(SIZES, alphas, strict=True):
        for target, color, marker, linestyle, fill in (
            ("HDE", RED, "o", "-", None),
            ("ROE", GREEN, "D", "--", "white"),
        ):
            line = profile[(profile.n_sentences == size) & (profile.y == target)].sort_values("lambda")
            ax.plot(
                line["lambda"],
                line.pearson_r,
                color=color,
                alpha=float(alpha),
                linestyle=linestyle,
                marker=marker,
                linewidth=2.45,
                markersize=7,
                markerfacecolor=color if fill is None else fill,
                markeredgecolor=color,
                markeredgewidth=1.5,
                label="_nolegend_",
            )
    ax.axhline(0, color="black", linewidth=1, alpha=0.5)
    ax.set_xticks(DECAYS)
    ax.set_xticklabels([r"$0.25$", r"$0.5$", r"$0.75$", r"$1.0$"])
    ax.set_xlabel(r"Distance decay $\lambda$", fontsize=26)
    ax.set_ylabel(r"Partial $r$", fontsize=26)
    ax.tick_params(axis="both", labelsize=18)
    ax.grid(True, alpha=0.3)
    target_handles = [
        plt.Line2D(
            [0],
            [0],
            color=RED,
            marker="o",
            linestyle="-",
            linewidth=2.45,
            markersize=7,
            label=r"$\mathrm{HDE} \mid \mathrm{TTR}$",
        ),
        plt.Line2D(
            [0],
            [0],
            color=GREEN,
            marker="D",
            linestyle="--",
            linewidth=2.45,
            markersize=7,
            markerfacecolor="white",
            label=r"$\mathrm{ROE} \mid \mathrm{TTR}$",
        ),
    ]
    size_handles = [
        plt.Line2D([0], [0], color=SLATE, alpha=float(alpha), linewidth=3, label=f"{size:,}")
        for size, alpha in zip(SIZES, alphas, strict=True)
    ]
    blanks = [plt.Line2D([], [], linestyle="", label="") for _ in range(2)]
    ax.legend(
        handles=[*target_handles, *blanks, *size_handles],
        fontsize=LEGEND_SIZE,
        loc="lower left",
        ncol=4,
        frameon=True,
        framealpha=1.0,
        columnspacing=0.45,
        handlelength=1.45,
        handletextpad=0.35,
        borderpad=0.35,
    )
    fig.tight_layout()
    _save(fig, output / "decay_profile_by_sentence_size")


def plot_language_trajectories(results: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for _, group in results.groupby("Language"):
        line = group.sort_values("n_sentences")
        ax.plot(
            line["n_sentences"],
            line[FLEX],
            color=MID_GRAY,
            linewidth=0.8,
            alpha=0.22,
            zorder=1,
        )
    summary = results.groupby("n_sentences")[FLEX].agg(["mean", "median"]).reset_index()
    ax.plot(
        summary["n_sentences"],
        summary["median"],
        color=GREEN,
        marker="s",
        linewidth=2.6,
        markersize=7,
        label="Median",
        zorder=3,
    )
    ax.plot(
        summary["n_sentences"],
        summary["mean"],
        color=RED,
        marker="o",
        linewidth=2.2,
        markersize=7,
        markerfacecolor="white",
        markeredgecolor=RED,
        markeredgewidth=1.5,
        label="Mean",
        zorder=3,
    )
    _format_sentence_axis(ax)
    ax.set_ylabel(r"Flexibility $F^{(.5)}$", fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=LEGEND_SIZE, framealpha=1.0)
    fig.tight_layout()
    _save(fig, output / "language_trajectories_by_sentence_size")


def _rank_delta_cell(value: float) -> str:
    delta = int(round(value))
    if delta == 0:
        return r"\cellcolor{paperLightGray!45}0"
    color = "paperGreen" if delta < 0 else "paperRed"
    intensity = min(60, 10 + 2 * abs(delta))
    return rf"\cellcolor{{{color}!{intensity}}}{delta:+d}"


def futrell_ranks(results: pd.DataFrame, output: Path) -> None:
    rows, summaries = [], []
    for metric, order in (("HDE", FUTRELL_HDE_ORDER), ("ROE", FUTRELL_ROE_ORDER)):
        shared = [language for language in order if language in set(results.Language)]
        subset = results[results.Language.isin(shared)][["Language", metric]].copy()
        subset["published_rank"] = subset.Language.map({language: index for index, language in enumerate(shared, 1)})
        subset["current_rank"] = subset[metric].rank(ascending=False, method="average")
        subset["delta"] = subset["current_rank"] - subset["published_rank"]
        subset["metric"] = metric
        rows.append(subset)
        test = stats.spearmanr(subset.published_rank, subset.current_rank)
        summaries.append(
            {
                "metric": metric,
                "n_languages": len(subset),
                "spearman_rho": test.statistic,
                "p": test.pvalue,
            }
        )
    comparison = pd.concat(rows)
    summary = pd.DataFrame(summaries)
    comparison.to_csv(output / "futrell_rank_comparison.csv", index=False)
    summary.to_csv(output / "futrell_rank_correlations.csv", index=False)

    wide = comparison.pivot(index="Language", columns="metric")
    languages = [language for language in FUTRELL_HDE_ORDER if language in wide.index]
    row_end = " " + "\\" * 2
    lines = [
        r"% Requires: \usepackage{booktabs} and \usepackage[table]{xcolor}.",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\caption{Ranking comparison with Futrell et al. Lower ranks indicate larger HDE or ROE values; $\Delta=\mathrm{UD}-F$.}\label{tab:futrell-rank-comparison}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Language & HDE$_F$ & HDE$_{\mathrm{UD}}$ & $\Delta$HDE & ROE$_F$ & ROE$_{\mathrm{UD}}$ & $\Delta$ROE \\",
        r"\midrule",
    ]
    for language in languages:
        values = []
        for metric in ("HDE", "ROE"):
            published = wide.loc[language, ("published_rank", metric)]
            current = wide.loc[language, ("current_rank", metric)]
            delta = wide.loc[language, ("delta", metric)]
            values.extend([f"{published:.0f}", f"{current:.0f}", _rank_delta_cell(delta)])
        lines.append(" & ".join([_latex_escape(language), *values]) + row_end)
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    (output / "futrell-rank-comparison-table.tex").write_text("\n".join(lines), encoding="utf-8")


def analysis_language_sets(
    all_results: pd.DataFrame,
    futrell: pd.DataFrame,
    sizes: pd.DataFrame,
    bakker: pd.DataFrame,
    output: Path,
) -> None:
    bakker_names = set(bakker["Language"].replace({"Greek": "Modern Greek"}))
    largest = sizes[sizes.n_sentences == sizes.n_sentences.max()]
    sets = [
        ("UD 1k sample", sorted(set(all_results.Language))),
        ("UD 10k / sentence-size common sample", sorted(set(largest.Language))),
        ("Futrell overlap", sorted(set(futrell.Language))),
        ("Bakker overlap", sorted(set(all_results.Language) & bakker_names)),
    ]
    markdown = ["# Language sets by analysis", "", "| Analysis | n | Languages |", "| --- | ---: | --- |"]
    for label, languages in sets:
        markdown.append(f"| {label} | {len(languages)} | {', '.join(languages)} |")
    (output / "analysis-language-sets.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")

    row_end = " " + "\\" * 2
    lines = [
        r"% Requires: \usepackage{booktabs,tabularx,array}.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{Language sets used in each analysis.}\label{tab:analysis-language-sets}",
        r"\begin{tabularx}{\textwidth}{@{}lr>{\raggedright\arraybackslash}X@{}}",
        r"\toprule",
        "Analysis & $n$ & Languages " + row_end,
        r"\midrule",
    ]
    for label, languages in sets:
        lines.append(
            " & ".join([_latex_escape(label), str(len(languages)), ", ".join(map(_latex_escape, languages))]) + row_end
        )
    lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table*}", ""])
    (output / "analysis-language-sets.tex").write_text("\n".join(lines), encoding="utf-8")


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for source, replacement in replacements.items():
        value = value.replace(source, replacement)
    return value


def _gradient_cell(value, mean: float, low: float, high: float, digits: int = 2) -> str:
    if pd.isna(value):
        return "--"
    color = "paperGreen"
    intensity = 0
    if value >= mean and high > mean:
        intensity = round(100 * (value - mean) / (high - mean))
    elif value < mean and mean > low:
        color = "paperRed"
        intensity = round(100 * (mean - value) / (mean - low))
    intensity = max(0, min(60, intensity))
    rendered = f"{value:.{digits}f}"
    return rendered if intensity == 0 else rf"\cellcolor{{{color}!{intensity}}}{rendered}"


def appendix_table(results: pd.DataFrame, bakker: pd.DataFrame, output: Path) -> None:
    bakker = bakker.copy()
    bakker["Language"] = bakker["Language"].replace({"Greek": "Modern Greek"})
    bakker = bakker[["Language", "Flexibility"]].rename(columns={"Flexibility": "Bakker"})
    table = results.merge(bakker, on="Language", how="left").sort_values("Language")
    metrics = [
        ("Bakker", "Bak."),
        ("HDE", "HDE"),
        ("ROE", "ROE"),
        ("F_lambda_025", r"$F^{(.25)}$"),
        ("F_lambda_050", r"$F^{(.5)}$"),
        ("F_lambda_075", r"$F^{(.75)}$"),
        ("F_lambda_100", r"$F^{(1)}$"),
        (TTR, "TTR"),
    ]
    statistics = {
        column: (
            table[column].mean(skipna=True),
            table[column].min(skipna=True),
            table[column].max(skipna=True),
        )
        for column, _ in metrics
    }
    labels = ["Language", *[label for _, label in metrics], r"$|\mathcal{P}|$"]
    row_end = " " + "\\" * 2
    header = " & ".join(labels) + row_end
    lines = [
        r"% Requires: \usepackage{booktabs,longtable} and \usepackage[table]{xcolor}.",
        r"% Uses the paperGreen and paperRed color definitions.",
        r"\begingroup",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\setlength{\LTleft}{\fill}",
        r"\setlength{\LTright}{\fill}",
        r"\begin{longtable}{@{}lrrrrrrrrr@{}}",
        (
            r"\caption{Language-level metrics. Colors are scaled independently within each "
            r"column around its mean: red indicates lower values and green higher values. "
            r"Missing Bakker ratings are marked as unavailable.}"
            r"\label{tab:appendix-language-metrics}\\"
        ),
        r"\toprule",
        header,
        r"\midrule",
        r"\endfirsthead",
        r"\caption[]{Language-level metrics (continued).}\\",
        r"\toprule",
        header,
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{10}{r}{\emph{Continued on next page}}\\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for row in table.itertuples(index=False):
        cells = [_latex_escape(row.Language)]
        for column, _ in metrics:
            mean, low, high = statistics[column]
            cells.append(_gradient_cell(getattr(row, column), mean, low, high))
        cells.append(f"{int(row.qualifying_pairs_lambda_050):,}")
        lines.append(" & ".join(cells) + row_end)
    lines.extend([r"\end{longtable}", r"\endgroup", ""])
    (output / "appendix-language-metrics-table.tex").write_text("\n".join(lines), encoding="utf-8")


def generate_paper_artifacts(all_csv: str, futrell_csv: str, sizes_csv: str, bakker_csv: str, output_dir: str) -> None:
    _set_style()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    all_results = pd.read_csv(all_csv)
    futrell = pd.read_csv(futrell_csv)
    sizes = pd.read_csv(sizes_csv)
    bakker = pd.read_csv(bakker_csv)
    plot_validation(all_results, futrell, output)
    lambda_analysis(all_results, output)
    bakker_figures(all_results, bakker, output)
    sentence_size_figures(sizes, output)
    plot_language_trajectories(sizes, output)
    futrell_ranks(futrell, output)
    analysis_language_sets(all_results, futrell, sizes, bakker, output)
    appendix_table(all_results, bakker, output)
    metadata = {
        "all_languages": all_results.Language.nunique(),
        "futrell_languages": futrell.Language.nunique(),
        "sentence_size_languages": sizes.Language.nunique(),
        "sentence_sizes": sorted(sizes.n_sentences.unique().tolist()),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def validation_plot(rows: list[dict], output: str | Path, *, control_ttr: bool = True) -> None:
    """Plot flexibility against HDE and ROE for the public CSV interface."""
    _set_style()
    frame = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6), constrained_layout=True)
    for ax, target, color in zip(axes, ("HDE", "ROE"), (RED, GREEN), strict=True):
        data = frame[["flexibility", target, "ttr"]].astype(float).dropna()
        x, y = data[target].to_numpy(), data["flexibility"].to_numpy()
        if control_ttr:
            x = residualize(x, data["ttr"])
            y = residualize(y, data["ttr"])
        _regression(ax, x, y, color, "Linear fit")
        suffix = " residual" if control_ttr else ""
        ax.set_xlabel(target + suffix)
        ax.set_ylabel("$F(.5)$" + suffix)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def profile_plot(rows: list[dict], output: str | Path) -> None:
    """Plot per-language and mean flexibility profiles from long-form rows."""
    _set_style()
    frame = pd.DataFrame(rows)
    frame[["lambda", "flexibility"]] = frame[["lambda", "flexibility"]].astype(float)
    fig, ax = plt.subplots(figsize=(5.2, 3.6), constrained_layout=True)
    for _, group in frame.groupby("language"):
        line = group.sort_values("lambda")
        ax.plot(
            line["lambda"],
            line["flexibility"],
            color=MID_GRAY,
            alpha=0.18,
            linewidth=0.8,
        )
    mean = frame.groupby("lambda", as_index=False)["flexibility"].mean()
    ax.plot(
        mean["lambda"],
        mean["flexibility"],
        color=GREEN,
        marker="o",
        linewidth=2,
        label="Mean",
    )
    ax.set_xlabel("Distance decay $\\lambda$")
    ax.set_ylabel("Flexibility $F(\\lambda)$")
    ax.legend(framealpha=1)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
