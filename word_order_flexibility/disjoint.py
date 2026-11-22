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

from .entropy import METRIC_DEFINITION, sentence_order_entropy
from .flexibility import word_order_flexibility
from .plot_utils import add_y_axis_break, figure_size
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
DISJOINT_VERSION = 2
ENTROPY_REPEATS = 100


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
            and row.get("entropy_definition") == METRIC_DEFINITION
            and row.get("entropy_repeats") == ENTROPY_REPEATS
            and row.get("entropy_seed") == row.get("sample")
            and all(
                isinstance(row.get(key), (int, float)) and np.isfinite(row[key])
                for key in ("entropy_difference", "entropy_mc_se")
            )
            for row in rows
        )
    )


def _blocks(items: list, *, seed: int = 0) -> list[list]:
    """Return five deterministic, disjoint 1,000-item blocks."""
    if len(items) < TOTAL_SENTENCES:
        raise ValueError(f"Found {len(items)} sentences; need {TOTAL_SENTENCES}")
    selected = random.Random(seed).sample(items, TOTAL_SENTENCES)
    return [selected[start : start + SAMPLE_SENTENCES] for start in range(0, TOTAL_SENTENCES, SAMPLE_SENTENCES)]


def _entropy_metrics(sentences: list[list[str]], sample: int) -> dict:
    """Estimate Delta H on the same block as WPA, using a reproducible trial seed."""
    result = sentence_order_entropy(sentences, repeats=ENTROPY_REPEATS, seed=sample)
    return {
        "entropy_difference": result["difference"]["mean"],
        "entropy_mc_se": result["difference"]["mc_se"],
        "entropy_repeats": result["repeats"],
        "entropy_seed": result["seed"],
        "entropy_definition": result["metric_definition"],
    }


def calculate_disjoint_text(
    code: str,
    input_path: str | Path,
    output_path: str | Path,
    *,
    input_format: str,
    splitter: str,
    min_sentence_words: int,
) -> None:
    """Calculate F(.5) and Delta H on five disjoint raw-text samples."""
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
                **_entropy_metrics(block, sample),
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
    """Calculate F(.5), HDE, ROE, and Delta H on five disjoint UD samples."""
    selected, available = sample_conllu(paths, TOTAL_SENTENCES, seed=0, single_root_only=True)
    rows = []
    for sample, block in enumerate(_blocks(selected)):
        sentences = surface_sentences(block)
        flexibility = word_order_flexibility(sentences, decay=0.5)
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
                **_entropy_metrics(sentences, sample),
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
        if "study_version" not in frames[name] or not frames[name]["study_version"].isin([1, DISJOINT_VERSION]).all():
            raise ValueError(f"{name}: stale or unknown disjoint-study version")
        counts = frames[name].groupby("code")["sample"].nunique()
        if not counts.eq(SAMPLE_COUNT).all():
            raise ValueError(f"{name}: expected {SAMPLE_COUNT} samples for every language")
        rows_per_language = frames[name].groupby("code").size()
        if not rows_per_language.eq(SAMPLE_COUNT).all():
            raise ValueError(f"{name}: duplicate or missing sample rows")
        if set(frames[name]["sample"]) != set(range(SAMPLE_COUNT)):
            raise ValueError(f"{name}: expected sample labels 0 through {SAMPLE_COUNT - 1}")
        if name != "ud" and not frames[name]["sentences"].eq(SAMPLE_SENTENCES).all():
            raise ValueError(f"{name}: every sample must contain {SAMPLE_SENTENCES} sentences")
    if not frames["ud"]["accepted_ud_sentences"].eq(SAMPLE_SENTENCES).all():
        raise ValueError("UD metrics did not accept every sampled single-root sentence")
    if not frames["ud"]["sentences"].between(1, SAMPLE_SENTENCES).all():
        raise ValueError("UD surface-sentence counts are invalid after punctuation removal")
    versions = set().union(*(set(frame.study_version) for frame in frames.values()))
    if len(versions) != 1:
        raise ValueError("Mixed disjoint-study versions; rerun all three domains")
    include_entropy = versions == {DISJOINT_VERSION}
    if include_entropy:
        for name, frame in frames.items():
            required = {"entropy_difference", "entropy_mc_se", "entropy_repeats", "entropy_seed", "entropy_definition"}
            if not required.issubset(frame.columns):
                raise ValueError(f"{name}: missing entropy results; rerun the disjoint study")
            if not np.isfinite(frame[["entropy_difference", "entropy_mc_se"]].to_numpy()).all():
                raise ValueError(f"{name}: entropy results must be finite for every sample")
            if not (
                frame.entropy_definition.eq(METRIC_DEFINITION).all()
                and frame.entropy_repeats.eq(ENTROPY_REPEATS).all()
                and frame.entropy_seed.eq(frame["sample"]).all()
            ):
                raise ValueError(f"{name}: incompatible entropy settings; rerun the disjoint study")
    pd.DataFrame({"code": common}).to_csv(output / "languages.csv", index=False)
    pools = pd.DataFrame({"code": common})
    for name, frame in frames.items():
        available = frame.groupby("code")["available_sentences"].first()
        pools[f"{name}_sentences"] = pools["code"].map(available)
    pools.to_csv(output / "language_sentence_pools.csv", index=False)

    reliability = []
    reliability_specs = [
        (r"UD $F$", "ud", "flexibility"),
        ("UD HDE", "ud", "HDE"),
        ("UD ROE", "ud", "ROE"),
    ]
    if include_entropy:
        reliability_specs.append((r"UD $\Delta H$", "ud", "entropy_difference"))
    for label, domain in (("Bible", "bible"), ("Wiki", "wikipedia")):
        reliability_specs.append((rf"{label} $F$", domain, "flexibility"))
        if include_entropy:
            reliability_specs.append((rf"{label} $\Delta H$", domain, "entropy_difference"))
    for label, corpus, metric in reliability_specs:
        reliability.extend({"measure": label, **row} for row in _pairwise_reliability(frames[corpus], metric))
    reliability_frame = pd.DataFrame(reliability)
    if not np.isfinite(reliability_frame.spearman_rho).all():
        raise ValueError("Undefined rank correlations: check for constant scores across languages")
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
    size = figure_size(2, height_ratio=0.20) if include_entropy else figure_size(1, height_ratio=0.34)
    figure, axis = plt.subplots(figsize=size)
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
            zorder=3,
        )
        axis.hlines(np.median(values), position - 0.20, position + 0.20, color=GREEN, linewidth=1.7, zorder=4)
    axis.set_ylabel(r"Spearman $\rho$")
    # some fiddling to make the corners nice while displaying what we need
    lowest = min(0.7, float(reliability_frame.spearman_rho.min()) - 0.02)
    tick_step = 0.1 if lowest >= 0.5 else 0.25
    lower = max(-1.0, round(np.floor(round(lowest / tick_step, 10)) * tick_step, 2))
    axis.set_ylim(lower, 1.01)
    axis.set_yticks(np.arange(lower, 1.001, tick_step))
    axis.spines["top"].set_position(("data", 1.0))
    for side in ("left", "right"):
        axis.spines[side].set_bounds(lower, 1.0)
    # apply label rotation (which gets reset by the spines)
    axis.set_xticks(range(len(reliability_specs)), [spec[0] for spec in reliability_specs], rotation=30, ha="right")
    if lower > 0:
        add_y_axis_break(axis)
    axis.grid(axis="x", visible=False)
    figure.subplots_adjust(left=0.18, right=0.98, bottom=0.34, top=0.97)
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
