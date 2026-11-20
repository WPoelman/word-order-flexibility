"""Sentence-conditioned Montemurro–Zanette (2011) entropy contrasts."""

import math
import random
import statistics
from collections.abc import Iterable, Sequence

METRIC_DEFINITION = "intact_minus_within_shuffled_v1"


def lz_entropy(tokens: Sequence[str]) -> float:
    """dit's increasing-window LZ match-length estimate, in bits/token.

    Uses dit.inference.lz_entropy_rate, including its finite-length conventions:
    skip the first position, allow overlapping earlier-starting matches, and
    count a terminal match with length remaining + 1 (right censoring).
    """
    from dit.inference import lz_entropy_rate

    if len(tokens) < 2:
        raise ValueError("Entropy estimation requires at least two tokens")
    return float(lz_entropy_rate(list(tokens)))


def _summary(values):
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else None
    se = sd / math.sqrt(len(values)) if sd is not None else None
    return {
        "mean": mean,
        "sd": sd,
        "mc_se": se,
        "mc_95_half_width": 1.96 * se if se is not None else None,
    }


def sentence_order_entropy(
    sentences: Iterable[Sequence[str]],
    *,
    repeats: int = 100,
    seed: int = 0,
    checkpoints: Sequence[int] | None = None,
) -> dict:
    """Paired H(intact) - H(within-sentence shuffled), randomizing sentence order.
    matches may span boundaries.
    """
    corpus = [list(s) for s in sentences if s]
    if repeats < 2:
        raise ValueError("repeats must be at least 2")
    if sum(map(len, corpus)) < 2:
        raise ValueError("Corpus must contain at least two tokens")
    points = sorted(
        set(checkpoints if checkpoints is not None else [2, 5, 10, 20, 50, repeats])
    )
    if checkpoints is not None and any(p < 2 or p > repeats for p in points):
        raise ValueError("checkpoints must lie between 2 and repeats")
    points = sorted({p for p in points if 2 <= p <= repeats} | {repeats})
    order_rng = random.Random(seed)
    word_rng = random.Random(f"words:{seed}")
    trials = []
    for trial in range(repeats):
        order = list(range(len(corpus)))
        order_rng.shuffle(order)
        intact, shuffled = [], []
        for index in order:
            sentence = corpus[index].copy()
            intact.extend(sentence)
            word_rng.shuffle(sentence)
            shuffled.extend(sentence)
        baseline = lz_entropy(intact)
        scrambled = lz_entropy(shuffled)
        trials.append(
            {
                "trial": trial + 1,
                "intact": baseline,
                "within_shuffled": scrambled,
                "difference": baseline - scrambled,
            }
        )
    curves = [
        {
            "repeats": n,
            **{
                key: _summary([t[key] for t in trials[:n]])
                for key in ("intact", "within_shuffled", "difference")
            },
        }
        for n in points
    ]
    return {
        "metric_definition": METRIC_DEFINITION,
        "estimator": "dit.inference.lz_entropy_rate",
        "units": "bits/token",
        "seed": seed,
        "sentences": len(corpus),
        "tokens": sum(map(len, corpus)),
        "repeats": repeats,
        **{key: curves[-1][key] for key in ("intact", "within_shuffled", "difference")},
        "shuffle_convergence": curves,
        "trials": trials,
    }


def entropy_convergence(
    sentences: Iterable[Sequence[str]],
    *,
    repeats: int = 100,
    seed: int = 0,
    sizes: Sequence[int] | None = None,
    tolerance: float = 0.1,
) -> dict:
    """
    Following M&Z's half sample convergence test. It generally doesn't pass, so I tested another way.
    """
    corpus = [list(s) for s in sentences if s]
    if len(corpus) < 4:
        raise ValueError("Needs at least four nonempty sentences")
    if not 0 < tolerance < 1:
        raise ValueError("tolerance must be between 0 and 1")
    random.Random(f"subsets:{seed}").shuffle(corpus)
    n = len(corpus)
    sizes = sorted(
        set(sizes if sizes is not None else [max(2, n // 8), max(2, n // 4), n // 2, n])
    )
    if any(size < 2 or size > n for size in sizes):
        raise ValueError("sizes must be between 2 and the corpus sentence count")
    sizes = sorted(set(sizes) | {n})
    curve = [
        sentence_order_entropy(corpus[:size], repeats=repeats, seed=seed)
        for size in sizes
    ]
    full = curve[-1]
    halves = [
        sentence_order_entropy(part, repeats=repeats, seed=seed)
        for part in (corpus[: n // 2], corpus[n // 2 :])
    ]
    diagnostics = {}
    for key in ("intact", "within_shuffled", "difference"):
        reference = full[key]["mean"]
        drift = [abs(half[key]["mean"] - reference) for half in halves]
        relative = [d / abs(reference) for d in drift] if reference != 0 else None
        diagnostics[key] = {
            "half_full_absolute_drift": drift,
            "half_full_relative_drift": relative,
            "passes_half_full": (
                max(relative) <= tolerance if relative is not None else None
            ),
        }
    return {
        "seed": seed,
        "tolerance": tolerance,
        "full": full,
        "size_convergence": curve,
        "halves": halves,
        "diagnostics": diagnostics,
    }


# Recorded input contract; implementation now comes directly from main.
UD_PREPROCESSING = {'version': 'paper-844d300-v1', 'casefolded': True, 'single_root_only': True, 'punctuation_removed': True}
