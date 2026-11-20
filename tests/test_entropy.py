import math
import random

import pytest

from word_order_flexibility.entropy import entropy_convergence, lz_entropy, sentence_order_entropy


def reference_entropy(tokens):
    # Independent exhaustive match search, including dit's boundary conventions.
    total = 0
    for i in range(1, len(tokens)):
        longest = 0
        for j in range(i):
            length = 0
            while i + length < len(tokens) and tokens[j + length] == tokens[i + length]:
                length += 1
            longest = max(longest, length)
        total += (longest + 1) / math.log2(i + 1)
    return (len(tokens) - 1) / total


@pytest.mark.parametrize('tokens', [list('abababab'), list('abcdef'), list('aaaa'), list('CDABCDEABCZ')])
def test_estimator_matches_exhaustive_reference(tokens):
    assert lz_entropy(tokens) == pytest.approx(reference_entropy(tokens))


def test_random_short_sequences_match_reference():
    rng = random.Random(73)
    for n in range(2, 70):
        tokens = [str(rng.randrange(4)) for _ in range(n)]
        assert lz_entropy(tokens) == pytest.approx(reference_entropy(tokens))


def test_token_identity_not_spelling():
    assert lz_entropy(['cat', 'dog', 'cat', 'dog']) == lz_entropy(['x', 'yyyy', 'x', 'yyyy'])


def test_paired_design_preserves_sentences(monkeypatch):
    import word_order_flexibility.entropy as module
    corpus = [['a', 'b', 'c'], ['d', 'e'], ['f', 'g', 'h', 'i']]
    calls = []
    def capture(tokens):
        calls.append(tokens)
        return 1.0
    monkeypatch.setattr(module, 'lz_entropy', capture)
    result = sentence_order_entropy(corpus, repeats=10)
    by_first = {s[0]: s for s in corpus}
    for intact, shuffled in zip(calls[::2], calls[1::2]):
        offset = 0
        while offset < len(intact):
            sentence = by_first[intact[offset]]
            end = offset + len(sentence)
            assert intact[offset:end] == sentence
            assert sorted(shuffled[offset:end]) == sorted(sentence)
            offset = end
    assert result['difference']['mean'] == 0
    assert corpus == [['a', 'b', 'c'], ['d', 'e'], ['f', 'g', 'h', 'i']]


def test_reproducibility_and_prefix_repeats():
    corpus = [list('abcd'), list('bcda'), list('dabc'), list('abcd')] * 3
    a = sentence_order_entropy(corpus, repeats=5, seed=42)
    b = sentence_order_entropy(corpus, repeats=10, seed=42)
    assert a == sentence_order_entropy(corpus, repeats=5, seed=42)
    assert a['trials'] == b['trials'][:5]
    assert b['difference']['mean'] == pytest.approx(b['intact']['mean'] - b['within_shuffled']['mean'])


def test_single_word_sentences_have_zero_contrast():
    result = entropy_convergence([['a'], ['b'], ['a'], ['c']], repeats=4)
    assert result['full']['difference']['mean'] == 0
    assert result['diagnostics']['difference']['passes_half_full'] is None


def test_rigid_order_has_positive_contrast():
    result = sentence_order_entropy([list('abcdefgh')] * 40, repeats=5)
    assert result['difference']['mean'] < 0


def test_convergence_sizes_and_halves():
    result = entropy_convergence([list('abcd')] * 9, repeats=3, sizes=[2, 4])
    assert [r['sentences'] for r in result['size_convergence']] == [2, 4, 9]
    assert [r['sentences'] for r in result['halves']] == [4, 5]


@pytest.mark.parametrize('corpus,repeats', [([], 2), ([['a']], 2), ([['a', 'b']], 1)])
def test_invalid_inputs(corpus, repeats):
    with pytest.raises(ValueError):
        sentence_order_entropy(corpus, repeats=repeats)


def test_metric_definition_is_saved():
    from word_order_flexibility.entropy import METRIC_DEFINITION, sentence_order_entropy
    score = sentence_order_entropy([["a", "b"], ["b", "a"]], repeats=2)
    assert score["metric_definition"] == METRIC_DEFINITION
    for trial in score["trials"]:
        assert trial["difference"] == trial["intact"] - trial["within_shuffled"]
