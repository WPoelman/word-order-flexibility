from __future__ import annotations

import csv
import json
import random
from itertools import combinations
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .flexibility import word_order_flexibility
from .plot_utils import figure_size
from .plotting import GREEN, LIGHT_GRAY, RED, SLATE, _bh, _save, _set_style, _stars
from .text import read_sentences
from .ud import calculate_ud_metrics, sample_conllu, surface_sentences

DISJOINT_CODES = (
    "ara_Arab",
    "ces_Latn",
    "deu_Latn",
    "eus_Latn",
    "heb_Hebr",
    "hin_Deva",
    "ind_Latn",
    "ita_Latn",
    "kor_Hang",
    "nld_Latn",
    "pcm_Latn",
    "pol_Latn",
    "por_Latn",
    "ron_Latn",
    "rus_Cyrl",
    "spa_Latn",
    "swe_Latn",
    "tur_Latn",
    "ukr_Cyrl",
    "urd_Arab",
)
SAMPLE_COUNT = 5
SAMPLE_SENTENCES = 1000
TOTAL_SENTENCES = SAMPLE_COUNT * SAMPLE_SENTENCES
DISJOINT_VERSION = 1


def valid_disjoint_result(path: str | Path, code: str) -> bool:
    """Return whether a cached language result contains all expected samples."""
    try:
        rows = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        isinstance(rows, list)
        and len(rows) == SAMPLE_COUNT
        and {row.get("sample") for row in rows if isinstance(row, dict)} == set(range(SAMPLE_COUNT))
        and all(
            isinstance(row, dict)
            and row.get("code") == code
            and row.get("sentences") == SAMPLE_SENTENCES
            and row.get("study_version") == DISJOINT_VERSION
            for row in rows
        )
    )


def _blocks(items: list, *, seed: int = 0) -> list[list]:
    """Return five deterministic, disjoint 1,000-item blocks."""
    if len(items) < TOTAL_SENTENCES:
        raise ValueError(f"Found {len(items)} sentences; need {TOTAL_SENTENCES}")
    selected = random.Random(seed).sample(items, TOTAL_SENTENCES)
    return [selected[start : start + SAMPLE_SENTENCES] for start in range(0, TOTAL_SENTENCES, SAMPLE_SENTENCES)]


def calculate_disjoint_text(
    code: str,
    input_path: str | Path,
    output_path: str | Path,
    *,
    input_format: str,
    splitter: str,
    min_sentence_words: int,
) -> None:
    """Calculate F(.5) on five disjoint raw-text samples."""
    sentences = read_sentences(input_path, input_format=input_format, apply_splitter=splitter)
    sentences = [sentence for sentence in sentences if len(sentence) >= min_sentence_words]
    rows = []
    for sample, block in enumerate(_blocks(sentences)):
        rows.append(
            {
                "code": code,
                "sample": sample,
                "study_version": DISJOINT_VERSION,
                "available_sentences": len(sentences),
                **word_order_flexibility(block, decay=0.5),
            }
        )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def calculate_disjoint_ud(
    code: str,
    paths: list[str],
    output_path: str | Path,
) -> None:
    """Calculate F(.5), HDE, and ROE on five disjoint UD samples."""
    selected, available = sample_conllu(paths, TOTAL_SENTENCES, seed=0, single_root_only=True)
    rows = []
    for sample, block in enumerate(_blocks(selected)):
        flexibility = word_order_flexibility(surface_sentences(block), decay=0.5)
        ud = calculate_ud_metrics(block)
        rows.append(
            {
                "code": code,
                "sample": sample,
                "study_version": DISJOINT_VERSION,
                "available_sentences": available,
                "HDE": ud["head_direction_entropy_weighted"],
                "ROE": ud["relation_order_entropy_weighted"],
                "accepted_ud_sentences": ud["ud_sentences"],
                **flexibility,
            }
        )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def combine_disjoint_results(inputs: list[str], output_path: str | Path) -> None:
    rows = [row for path in inputs for row in json.loads(Path(path).read_text(encoding="utf-8"))]
    if not rows:
        raise ValueError("No disjoint-sample results to combine")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)


def _pairwise_reliability(frame: pd.DataFrame, metric: str) -> list[dict]:
    pivot = frame.pivot(index="code", columns="sample", values=metric).dropna()
    rows = []
    for first, second in combinations(pivot.columns, 2):
        test = stats.spearmanr(pivot[first], pivot[second])
        rows.append(
            {
                "sample_a": int(first),
                "sample_b": int(second),
                "n": len(pivot),
                "spearman_rho": float(test.statistic),
            }
        )
    return rows


def analyze_disjoint_study(results_dir: str | Path, output_dir: str | Path | None = None) -> None:
    """Generate compact reliability and cross-domain artifacts."""
    root = Path(results_dir)
    output = Path(output_dir) if output_dir else root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    frames = {name: pd.read_csv(root / name / "results.csv") for name in ("ud", "bible", "wikipedia")}
    common_set = set.intersection(*(set(frame["code"]) for frame in frames.values()))
    expected = set(DISJOINT_CODES)
    if common_set != expected:
        missing = sorted(expected - common_set)
        unexpected = sorted(common_set - expected)
        raise ValueError(f"Wrong language set; missing={missing}, unexpected={unexpected}")
    common = sorted(common_set)
    for name in frames:
        frames[name] = frames[name][frames[name]["code"].isin(common)]
        if "study_version" not in frames[name] or not frames[name]["study_version"].eq(DISJOINT_VERSION).all():
            raise ValueError(f"{name}: stale or unknown disjoint-study version")
        counts = frames[name].groupby("code")["sample"].nunique()
        if not counts.eq(SAMPLE_COUNT).all():
            raise ValueError(f"{name}: expected {SAMPLE_COUNT} samples for every language")
        rows_per_language = frames[name].groupby("code").size()
        if not rows_per_language.eq(SAMPLE_COUNT).all():
            raise ValueError(f"{name}: duplicate or missing sample rows")
        if set(frames[name]["sample"]) != set(range(SAMPLE_COUNT)):
            raise ValueError(f"{name}: expected sample labels 0 through {SAMPLE_COUNT - 1}")
        if not frames[name]["sentences"].eq(SAMPLE_SENTENCES).all():
            raise ValueError(f"{name}: every sample must contain {SAMPLE_SENTENCES} sentences")
    if not frames["ud"]["accepted_ud_sentences"].eq(SAMPLE_SENTENCES).all():
        raise ValueError("UD metrics did not accept every sampled single-root sentence")
    pd.DataFrame({"code": common}).to_csv(output / "languages.csv", index=False)
    pools = pd.DataFrame({"code": common})
    for name, frame in frames.items():
        available = frame.groupby("code")["available_sentences"].first()
        pools[f"{name}_sentences"] = pools["code"].map(available)
    pools.to_csv(output / "language_sentence_pools.csv", index=False)

    reliability = []
    reliability_specs = (
        (r"UD $F$", "ud", "flexibility"),
        ("UD HDE", "ud", "HDE"),
        ("UD ROE", "ud", "ROE"),
        (r"Bible $F$", "bible", "flexibility"),
        (r"Wiki $F$", "wikipedia", "flexibility"),
    )
    for label, corpus, metric in reliability_specs:
        reliability.extend({"measure": label, **row} for row in _pairwise_reliability(frames[corpus], metric))
    reliability_frame = pd.DataFrame(reliability)
    reliability_frame.to_csv(output / "within_domain_reliability.csv", index=False)

    display_names = {"ud": "UD", "bible": "Bible", "wikipedia": "Wikipedia"}
    means = pd.DataFrame(
        {display_names[name]: frame.groupby("code")["flexibility"].mean().loc[common] for name, frame in frames.items()}
    )
    matrix = means.corr(method="spearman")
    labels = list(matrix)
    tests = []
    for row_index, left in enumerate(labels):
        for right in labels[:row_index]:
            test = stats.spearmanr(means[left], means[right])
            tests.append(
                {
                    "left": left,
                    "right": right,
                    "n": len(means),
                    "spearman_rho": float(test.statistic),
                    "p": float(test.pvalue),
                }
            )
    for row, q_value in zip(tests, _bh([row["p"] for row in tests]), strict=True):
        row["q_bh"] = q_value
    pd.DataFrame(tests).to_csv(output / "cross_domain_correlations.csv", index=False)

    _set_style()
    figure, axis = plt.subplots(figsize=figure_size(1, height_ratio=0.42))
    rng = np.random.default_rng(0)
    for position, (label, group) in enumerate(reliability_frame.groupby("measure", sort=False)):
        values = group["spearman_rho"].to_numpy()
        axis.scatter(
            np.full(len(values), position) + rng.uniform(-0.08, 0.08, len(values)),
            values,
            color=SLATE,
            alpha=0.32,
            s=9,
            linewidth=0,
        )
        axis.hlines(np.median(values), position - 0.20, position + 0.20, color=GREEN, linewidth=1.7)
    axis.set_xticks(range(len(reliability_specs)), [spec[0] for spec in reliability_specs], rotation=30, ha="right")
    axis.set_ylabel(r"Spearman $\rho$")
    minimum = reliability_frame["spearman_rho"].min()
    lower = 0.5 if minimum >= 0.5 else -1.0
    axis.set_ylim(lower, 1.02)
    if lower < 0:
        axis.axhline(0, color=SLATE, linewidth=0.6, alpha=0.55)
    axis.grid(axis="x", visible=False)
    figure.subplots_adjust(left=0.18, right=0.98, bottom=0.28, top=0.98)
    _save(figure, output / "disjoint_sample_reliability")

    figure, axis = plt.subplots(figsize=figure_size(1, height_ratio=0.55))
    color_map = mpl.colors.LinearSegmentedColormap.from_list("paper_red_green", [RED, LIGHT_GRAY, GREEN])
    axis.imshow(matrix, vmin=-1, vmax=1, cmap=color_map, aspect="equal")
    q_values = {(row["left"], row["right"]): row["q_bh"] for row in tests}
    for row_index, left in enumerate(labels):
        for column_index, right in enumerate(labels):
            if column_index > row_index:
                axis.add_patch(
                    plt.Rectangle((column_index - 0.5, row_index - 0.5), 1, 1, facecolor="white", edgecolor="white")
                )
            elif column_index == row_index:
                axis.add_patch(
                    plt.Rectangle(
                        (column_index - 0.5, row_index - 0.5),
                        1,
                        1,
                        facecolor=LIGHT_GRAY,
                        edgecolor="white",
                    )
                )
                axis.text(column_index, row_index, r"\textemdash", ha="center", va="center", color=SLATE)
            else:
                q_value = q_values.get((left, right), q_values.get((right, left)))
                text_color = "white" if abs(matrix.loc[left, right]) > 0.65 else "#1A1A1A"
                axis.text(
                    column_index,
                    row_index + 0.08,
                    rf"$\mathbf{{{matrix.loc[left, right]:.2f}}}$",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=text_color,
                )
                axis.text(
                    column_index,
                    row_index - 0.25,
                    _stars(q_value),
                    ha="center",
                    va="center",
                    fontsize=5,
                    color=text_color,
                )
    axis.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    axis.grid(False)
    figure.subplots_adjust(left=0.23, right=0.97, bottom=0.25, top=0.98)
    _save(figure, output / "disjoint_cross_domain_correlations")
