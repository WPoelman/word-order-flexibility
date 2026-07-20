from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence

Sentence = Sequence[str]


def _pair_and_direction(left: str, right: str) -> tuple[tuple[str, str], int] | None:
    if left == right:
        return None
    pair = (left, right) if left < right else (right, left)
    return pair, 0 if (left, right) == pair else 1


def word_order_flexibility(
    sentences: Iterable[Sentence],
    *,
    decay: float = 0.5,
    min_pair_count: int = 3,
    return_pairs: bool = False,
) -> dict:
    """
    For every unordered recurring word-type pair, directional observations at
    word distance ``d`` receive weight ``decay ** d``. Pair flexibility is
    ``1 - |forward - reverse| / (forward + reverse)``. The corpus score is the
    mean over pairs with at least ``min_pair_count`` undecayed observations.
    Identical-word pairs and punctuation are expected to have been removed during tokenization.
    """
    if not 0 < decay <= 1:
        raise ValueError("decay must be in (0, 1]")
    if min_pair_count < 1:
        raise ValueError("min_pair_count must be at least 1")

    decayed = defaultdict(lambda: [0.0, 0.0])
    observed = defaultdict(lambda: [0, 0])
    token_counts = Counter()
    n_sentences = 0

    for sentence in sentences:
        words = list(sentence)
        if not words:
            continue
        n_sentences += 1
        token_counts.update(words)
        for left_position, left in enumerate(words):
            for right_position in range(left_position + 1, len(words)):
                pair_direction = _pair_and_direction(left, words[right_position])
                if pair_direction is None:
                    continue
                pair, direction = pair_direction
                distance = right_position - left_position
                observed[pair][direction] += 1
                decayed[pair][direction] += decay**distance

    pair_rows = []
    values = []
    for pair in sorted(decayed):
        raw_count = sum(observed[pair])
        if raw_count < min_pair_count:
            continue
        forward, reverse = decayed[pair]
        total = forward + reverse
        if total <= 0:
            continue
        flexibility = 1.0 - abs(forward - reverse) / total
        values.append(flexibility)
        pair_rows.append(
            {
                "word_a": pair[0],
                "word_b": pair[1],
                "forward": forward,
                "reverse": reverse,
                "observations": raw_count,
                "flexibility": flexibility,
            }
        )

    n_tokens = sum(token_counts.values())
    result = {
        "flexibility": sum(values) / len(values) if values else 0.0,
        "lambda": decay,
        "min_pair_count": min_pair_count,
        "qualifying_pairs": len(values),
        "sentences": n_sentences,
        "tokens": n_tokens,
        "types": len(token_counts),
        "ttr": len(token_counts) / n_tokens if n_tokens else 0.0,
    }
    if return_pairs:
        result["pairs"] = pair_rows
    return result


def flexibility_profile(
    sentences: Iterable[Sentence],
    decays: Sequence[float] = (0.25, 0.5, 0.75, 1.0),
    *,
    min_pair_count: int = 3,
) -> list[dict]:
    """Calculate the metric at several decay values on the same corpus."""
    cached = [list(sentence) for sentence in sentences]
    return [word_order_flexibility(cached, decay=value, min_pair_count=min_pair_count) for value in decays]
