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


def load_conllu(paths: Iterable[str | Path], limit: int | None = None) -> list:
    """Load CoNLL-U data in the original train, dev, test experiment order."""
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
                sentences.append(sentence)
                if limit is not None and len(sentences) >= limit:
                    return sentences
    return sentences


def surface_sentences(ud_sentences: Iterable) -> list[list[str]]:
    """Extract case-sensitive words while omitting UD punctuation tokens."""
    return [
        [token["form"] for token in sentence if isinstance(token.get("id"), int) and token.get("upos") != "PUNCT"]
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
        tokens = {token["id"]: token for token in sentence if isinstance(token.get("id"), int)}
        roots = [token for token in tokens.values() if token.get("head") == 0]
        if len(roots) != 1:
            continue
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
