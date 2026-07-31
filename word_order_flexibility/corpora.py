from __future__ import annotations

import csv
import json
import random
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .analysis import residualize
from .flexibility import word_order_flexibility
from .languages import UD_MAPPING
from .plot_utils import figure_size
from .plotting import _format_p, _save, _set_style
from .text import read_sentences


def calculate_corpus_file(
    code: str,
    input_path: str | Path,
    output_path: str | Path,
    *,
    input_format: str,
    splitter: str = "default",
    n_sentences: int = 1000,
    decay: float = 0.5,
    sample_output_path: str | Path | None = None,
    sample_seeds: tuple[int, ...] = (),
    sample_sentences: int = 1000,
    min_sentence_words: int = 1,
) -> None:
    """Calculate result and seeded runs after one segmentation pass."""
    if min_sentence_words < 1:
        raise ValueError("min_sentence_words must be at least 1")
    sentences = read_sentences(input_path, input_format=input_format, apply_splitter=splitter)
    sentences = [sentence for sentence in sentences if len(sentence) >= min_sentence_words]
    required = max(n_sentences, sample_sentences if sample_seeds else 0)
    if len(sentences) < required:
        exclusion = {
            "code": code,
            "excluded": True,
            "min_sentence_words": min_sentence_words,
            "available_sentences": len(sentences),
            "required_sentences": required,
            "reason": "insufficient_sentences",
        }
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(exclusion, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if sample_output_path is not None:
            sample_output = Path(sample_output_path)
            sample_output.parent.mkdir(parents=True, exist_ok=True)
            sample_output.write_text("[]\n", encoding="utf-8")
        return

    result = word_order_flexibility(sentences[:n_sentences], decay=decay)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "code": code,
                "excluded": False,
                "min_sentence_words": min_sentence_words,
                "available_sentences": len(sentences),
                "required_sentences": required,
                **result,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    if sample_output_path is not None:
        sample_rows = []
        for seed in sample_seeds:
            sample = random.Random(seed).sample(sentences, sample_sentences)
            sample_rows.append(
                {
                    "code": code,
                    "seed": seed,
                    "excluded": False,
                    "min_sentence_words": min_sentence_words,
                    "available_sentences": len(sentences),
                    **word_order_flexibility(sample, decay=decay),
                }
            )
        sample_output = Path(sample_output_path)
        sample_output.parent.mkdir(parents=True, exist_ok=True)
        sample_output.write_text(json.dumps(sample_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def combine_corpus_results(inputs: list[str], output_path: str | Path) -> None:
    rows = [json.loads(Path(path).read_text(encoding="utf-8")) for path in inputs]
    if not rows:
        raise ValueError("No corpus results to combine")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        columns = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def combine_corpus_samples(inputs: list[str], output_path: str | Path) -> None:
    rows = [row for path in inputs for row in json.loads(Path(path).read_text(encoding="utf-8"))]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        columns = (
            list(dict.fromkeys(key for row in rows for key in row))
            if rows
            else [
                "code",
                "seed",
                "excluded",
                "min_sentence_words",
                "available_sentences",
                "flexibility",
                "lambda",
                "min_pair_count",
                "qualifying_pairs",
                "sentences",
                "tokens",
                "types",
                "ttr",
            ]
        )
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def comparison_correlations(rows: list[dict]) -> list[dict]:
    """Report raw and TTR-controlled correlations against UD flexibility."""
    valid = [
        row
        for row in rows
        if all(row.get(key) not in (None, "") for key in ("flexibility", "ud_flexibility", "ttr", "ud_ttr"))
    ]
    if len(valid) < 3:
        return [
            {
                "control": "ttr" if controlled else "none",
                "n": len(valid),
                "pearson_r": None,
                "pearson_p": None,
                "spearman_rho": None,
                "spearman_p": None,
            }
            for controlled in (False, True)
        ]

    corpus_flexibility = np.asarray([float(row["flexibility"]) for row in valid])
    ud_flexibility = np.asarray([float(row["ud_flexibility"]) for row in valid])
    corpus_ttr = np.asarray([float(row["ttr"]) for row in valid])
    ud_ttr = np.asarray([float(row["ud_ttr"]) for row in valid])
    results = []
    for controlled in (False, True):
        x, y = corpus_flexibility, ud_flexibility
        if controlled:
            x = residualize(x, corpus_ttr)
            y = residualize(y, ud_ttr)
        pearson = stats.pearsonr(x, y)
        spearman = stats.spearmanr(x, y)
        results.append(
            {
                "control": "ttr" if controlled else "none",
                "n": len(valid),
                "pearson_r": float(pearson.statistic),
                "pearson_p": float(pearson.pvalue),
                "spearman_rho": float(spearman.statistic),
                "spearman_p": float(spearman.pvalue),
            }
        )
    return results


def _ud_results(path: str | Path) -> pd.DataFrame:
    ud = pd.read_csv(path)
    reverse_mapping = {language: code for code, language in UD_MAPPING.items()}
    ud["code"] = ud["Language"].map(reverse_mapping)
    return ud[["code", "F_lambda_050", "type_token_ratio"]].rename(
        columns={"F_lambda_050": "ud_flexibility", "type_token_ratio": "ud_ttr"}
    )


def analyze_corpora(
    corpus_results: list[tuple[str, str]],
    ud_results_path: str | Path,
    output_dir: str | Path,
) -> None:
    """Write overlap tables and raw/TTR-controlled correlations."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    ud = _ud_results(ud_results_path)

    summary = {}
    overlaps = []
    for name, path in corpus_results:
        corpus = pd.read_csv(path)
        overlap = corpus.merge(ud, on="code", how="inner").dropna(
            subset=["flexibility", "ttr", "ud_flexibility", "ud_ttr"]
        )
        overlap.to_csv(output / f"{name}_overlap.csv", index=False)
        summary[name] = comparison_correlations(overlap.to_dict("records"))
        overlaps.append((name, overlap))
    (output / "correlations.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    _set_style()
    figure, axes = plt.subplots(1, len(overlaps), figsize=figure_size(2, height_ratio=0.33), squeeze=False)
    for axis, (name, overlap) in zip(axes[0], overlaps, strict=True):
        x = residualize(overlap["ud_flexibility"], overlap["ud_ttr"])
        y = residualize(overlap["flexibility"], overlap["ttr"])
        axis.scatter(x, y, s=16, color="#8E1B1B", alpha=0.75, edgecolor="white", linewidth=0.4)
        if len(overlap) >= 2:
            slope, intercept = np.polyfit(x, y, 1)
            grid = np.linspace(x.min(), x.max(), 100)
            axis.plot(grid, slope * grid + intercept, color="#1B7837", linewidth=1.1)
        controlled = summary[name][1]
        rho = controlled["spearman_rho"]
        pvalue = controlled["spearman_p"]
        annotation = f"$n={len(overlap)}$"
        if rho is not None and pvalue is not None:
            annotation += "\n" + rf"$\rho={rho:.3f}$, {_format_p(pvalue)}"
        axis.text(
            0.04,
            0.96,
            annotation,
            transform=axis.transAxes,
            ha="left",
            va="top",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "0.75"},
        )
        axis.set_title(name.title())
        axis.set_xlabel(r"UD $F^{(.5)} \mid \mathrm{TTR}$ (resid.)")
        axis.set_ylabel(rf"{name.title()} $F^{{(.5)}} \mid \mathrm{{TTR}}$ (resid.)")
    figure.tight_layout()
    _save(figure, output / "ud_versus_corpora")


def analyze_corpus_stability(
    corpus_samples: list[tuple[str, str]],
    ud_results_path: str | Path,
    output_dir: str | Path,
) -> None:
    """Summarize language estimates and validation correlations across samples."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    ud = _ud_results(ud_results_path)

    correlation_rows = []
    summary_rows = []
    reliability_rows = []
    for corpus, path in corpus_samples:
        samples = pd.read_csv(path)
        summary = (
            samples.groupby("code")
            .agg(
                samples=("seed", "nunique"),
                flexibility_mean=("flexibility", "mean"),
                flexibility_sd=("flexibility", "std"),
                flexibility_min=("flexibility", "min"),
                flexibility_max=("flexibility", "max"),
                ttr_mean=("ttr", "mean"),
                qualifying_pairs_mean=("qualifying_pairs", "mean"),
            )
            .reset_index()
        )
        summary.insert(0, "corpus", corpus)
        summary_rows.extend(summary.to_dict("records"))

        overlap = samples.merge(ud, on="code", how="inner").dropna(
            subset=["flexibility", "ttr", "ud_flexibility", "ud_ttr"]
        )
        for seed, group in overlap.groupby("seed"):
            for result in comparison_correlations(group.to_dict("records")):
                correlation_rows.append({"corpus": corpus, "seed": int(seed), **result})

        pivot = samples.pivot(index="code", columns="seed", values="flexibility").dropna()
        for first, second in combinations(pivot.columns, 2):
            correlation = stats.spearmanr(pivot[first], pivot[second])
            reliability_rows.append(
                {
                    "corpus": corpus,
                    "seed_a": int(first),
                    "seed_b": int(second),
                    "n": len(pivot),
                    "spearman_rho": float(correlation.statistic),
                    "p": float(correlation.pvalue),
                }
            )

    correlations = pd.DataFrame(correlation_rows)
    summaries = pd.DataFrame(summary_rows)
    reliability = pd.DataFrame(reliability_rows)
    correlations.to_csv(output / "sample_validation_correlations.csv", index=False)
    summaries.to_csv(output / "sample_language_summary.csv", index=False)
    reliability.to_csv(output / "sample_rank_reliability.csv", index=False)

    if correlations.empty:
        return

    _set_style()
    for control, suffix in (("none", "raw"), ("ttr", "ttr")):
        figure, axis = plt.subplots(figsize=figure_size(1, height_ratio=0.75))
        subset = correlations[correlations["control"] == control]
        corpora = sorted(subset["corpus"].unique())
        for position, corpus in enumerate(corpora):
            group = subset[subset["corpus"] == corpus].sort_values("seed")
            x = np.full(len(group), position, dtype=float)
            if len(group) > 1:
                x += np.linspace(-0.08, 0.08, len(group))
            axis.scatter(x, group["spearman_rho"], s=16, color="#8E1B1B", alpha=0.75)
            axis.hlines(
                group["spearman_rho"].median(),
                position - 0.18,
                position + 0.18,
                color="#1B7837",
                linewidth=1.3,
            )
        axis.set_xticks(range(len(corpora)), [name.title() for name in corpora])
        ylabel = r"Spearman $\rho$ with UD $F^{(.5)}$"
        if control == "ttr":
            ylabel = r"Spearman $\rho$ with UD $F^{(.5)} \mid \mathrm{TTR}$"
        axis.set_ylabel(ylabel)
        figure.tight_layout()
        _save(figure, output / f"sample_validation_stability_{suffix}")

    figure, axis = plt.subplots(figsize=figure_size(1, height_ratio=0.71))
    corpus_groups = list(summaries.groupby("corpus"))
    grouped = [group["flexibility_sd"].dropna() for _, group in corpus_groups]
    labels = [name.title() for name, _ in corpus_groups]
    axis.boxplot(
        grouped,
        tick_labels=labels,
        patch_artist=True,
        boxprops={"facecolor": "#E8F3EA", "edgecolor": "#1B7837", "linewidth": 1.0},
        medianprops={"color": "#8E1B1B", "linewidth": 1.2},
        whiskerprops={"color": "#4C566A", "linewidth": 0.9},
        capprops={"color": "#4C566A", "linewidth": 0.9},
        flierprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "#8E1B1B",
            "markersize": 3.5,
        },
    )
    axis.set_ylabel(r"SD of $F^{(.5)}$ across samples")
    figure.tight_layout()
    _save(figure, output / "sample_language_variability")
