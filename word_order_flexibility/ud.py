import random
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

import conllu
import numpy as np


def chao_shen_entropy(values: Iterable) -> float:
    """Estimate Shannon entropy in bits with the Chao--Shen estimator."""
    counts = np.asarray(list(Counter(values).values()), dtype=float)
    n = counts.sum()
    if n == 0:
        return 0.0
    singletons = np.sum(counts == 1)
    if singletons == n:
        singletons = n - 1
    coverage = 1 - singletons / n
    covered_probabilities = coverage * counts / n
    inclusion = 1 - (1 - covered_probabilities) ** n
    valid = (covered_probabilities > 0) & (inclusion > 0)
    entropy = -np.sum(covered_probabilities[valid] * np.log2(covered_probabilities[valid]) / inclusion[valid])
    return max(0.0, float(entropy))


def load_conllu(
    paths: Iterable[str | Path],
    limit: int | None = None,
    *,
    single_root_only: bool = False,
) -> list:
    """Load CoNLL-U data in train, dev, test order."""
    sentences = []
    files = []
    for path_value in paths:
        path = Path(path_value)
        if path.is_dir():
            for split in ("train", "dev", "test"):
                files.extend(sorted(path.rglob(f"*{split}.conllu")))
        else:
            files.append(path)
    for path in files:
        with path.open(encoding="utf-8") as handle:
            for sentence in conllu.parse_incr(handle):
                if single_root_only and not is_single_root(sentence):
                    continue
                sentences.append(sentence)
                if limit is not None and len(sentences) >= limit:
                    return sentences
    return sentences


def is_single_root(sentence) -> bool:
    """Return whether a UD sentence has exactly one integer-token root."""
    return sum(1 for token in sentence if isinstance(token.get("id"), int) and token.get("head") == 0) == 1


def surface_sentences(ud_sentences: Iterable, *, lowercase: bool = True) -> list[list[str]]:
    """Extract integer-ID UD words while omitting punctuation.

    Forms are case-folded by default to match raw-text preprocessing.
    """
    return [
        [
            token["form"].casefold() if lowercase else token["form"]
            for token in sentence
            if isinstance(token.get("id"), int) and token.get("upos") != "PUNCT"
        ]
        for sentence in ud_sentences
    ]


def calculate_ud_metrics(ud_sentences: Iterable) -> dict[str, float | int]:
    """Calculate HDE and ROE following Futrell et al. (2015).

    HDE is ``H(direction | relation, head POS, dependent POS)``. ROE is
    ``H(linearization | unordered local subtree configuration)``. We omit
    ``punct`` dependencies, punctuation tokens, and dependencies headed by
    punctuation, and retain only single-root sentences.
    """
    hde_groups = defaultdict(list)
    roe_groups = defaultdict(list)
    accepted = 0

    for sentence in ud_sentences:
        if not is_single_root(sentence):
            continue
        tokens = {token["id"]: token for token in sentence if isinstance(token.get("id"), int)}
        accepted += 1
        dependents = defaultdict(list)

        for token in tokens.values():
            head_id = token.get("head")
            if token.get("upos") == "PUNCT" or token.get("deprel") == "punct":
                continue
            if not isinstance(head_id, int) or head_id <= 0 or head_id not in tokens:
                continue
            head = tokens[head_id]
            if head.get("upos") == "PUNCT" or head.get("deprel") == "punct":
                continue
            dependents[head_id].append(token)
            context = (token.get("deprel"), head.get("upos"), token.get("upos"))
            hde_groups[context].append("R" if token["id"] > head_id else "L")

        for head_id, deps in dependents.items():
            head = tokens[head_id]
            configuration = (
                head.get("upos"),
                tuple(sorted((dep.get("deprel"), dep.get("upos")) for dep in deps)),
            )
            elements = [(head_id, "HEAD", head.get("upos"))]
            elements.extend((dep["id"], dep.get("deprel"), dep.get("upos")) for dep in deps)
            linearization = tuple((relation, pos) for _, relation, pos in sorted(elements))
            roe_groups[configuration].append(linearization)

    def weighted_conditional_entropy(groups: dict) -> float:
        total = sum(len(values) for values in groups.values())
        if total == 0:
            return 0.0
        return sum(len(values) / total * chao_shen_entropy(values) for values in groups.values())

    def unweighted_conditional_entropy(groups: dict) -> float:
        if not groups:
            return 0.0
        return sum(chao_shen_entropy(values) for values in groups.values()) / len(groups)

    hde_weighted = weighted_conditional_entropy(hde_groups)
    roe_weighted = weighted_conditional_entropy(roe_groups)

    return {
        "head_direction_entropy_weighted": hde_weighted,
        "relation_order_entropy_weighted": roe_weighted,
        "head_direction_entropy_unweighted": unweighted_conditional_entropy(hde_groups),
        "relation_order_entropy_unweighted": unweighted_conditional_entropy(roe_groups),
        "ud_sentences": accepted,
        "hde_contexts": len(hde_groups),
        "roe_configurations": len(roe_groups),
    }


def sample_conllu(
    paths: Iterable[str | Path],
    size: int,
    *,
    seed: int = 0,
    single_root_only: bool = False,
) -> tuple[list, int]:
    """Reservoir-sample sentences without loading an entire treebank into memory."""
    rng = random.Random(seed)
    reservoir = []
    seen = 0
    files = []
    for path_value in paths:
        path = Path(path_value)
        if path.is_dir():
            for split in ("train", "dev", "test"):
                files.extend(sorted(path.rglob(f"*{split}.conllu")))
        else:
            files.append(path)
    for path in files:
        with path.open(encoding="utf-8") as handle:
            for sentence in conllu.parse_incr(handle):
                if single_root_only and not is_single_root(sentence):
                    continue
                seen += 1
                if len(reservoir) < size:
                    reservoir.append(sentence)
                    continue
                position = rng.randrange(seen)
                if position < size:
                    reservoir[position] = sentence
    if seen < size:
        raise ValueError(f"Found {seen} sentences; need {size}")
    rng.shuffle(reservoir)
    return reservoir, seen


def count_single_root_sentences(paths: Iterable[str | Path], stop_at: int) -> int:
    """Count single-root CoNLL-U sentences, stopping at the requested threshold."""
    count = 0
    files = []
    for path_value in paths:
        path = Path(path_value)
        if path.is_dir():
            for split in ("train", "dev", "test"):
                files.extend(sorted(path.rglob(f"*{split}.conllu")))
        else:
            files.append(path)
    for path in files:
        roots = 0
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    if roots == 1:
                        count += 1
                        if count >= stop_at:
                            return count
                    roots = 0
                    continue
                if stripped.startswith("#"):
                    continue
                fields = stripped.split("\t")
                if fields[0].isdigit() and len(fields) > 6 and fields[6] == "0":
                    roots += 1
        if roots == 1:
            count += 1
            if count >= stop_at:
                return count
    return count
