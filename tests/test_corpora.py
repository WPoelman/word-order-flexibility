import json

import pytest

from word_order_flexibility.corpora import calculate_corpus_file, comparison_correlations


def test_corpus_job_uses_exact_sentence_prefix(tmp_path):
    source = tmp_path / "corpus.txt"
    source.write_text("a b\na b\nb a\nb a\n", encoding="utf-8")
    output = tmp_path / "result.json"

    calculate_corpus_file("abc_Latn", source, output, input_format="tokenized", n_sentences=3)

    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["code"] == "abc_Latn"
    assert result["sentences"] == 3


def test_corpus_job_rejects_short_input(tmp_path):
    source = tmp_path / "corpus.txt"
    source.write_text("a b\nb a\n", encoding="utf-8")

    with pytest.raises(ValueError, match="found 2 sentences; need 3"):
        calculate_corpus_file("abc_Latn", source, tmp_path / "result.json", input_format="tokenized", n_sentences=3)


def test_corpus_comparison_reports_raw_and_separate_ttr_control():
    rows = []
    shared = [0.0, 1.0, -1.0, 1.0, -1.0]
    for index, value in enumerate(shared):
        rows.append(
            {
                "flexibility": 2 * index + value,
                "ud_flexibility": -3 * (4 - index) + 2 * value,
                "ttr": index,
                "ud_ttr": 4 - index,
            }
        )

    raw, controlled = comparison_correlations(rows)

    assert raw["control"] == "none"
    assert controlled["control"] == "ttr"
    assert controlled["n"] == 5
    assert controlled["pearson_r"] == pytest.approx(1.0)


def test_primary_and_random_samples_share_one_sentence_read(tmp_path, monkeypatch):
    calls = 0
    sentences = [
        ["a", "b"],
        ["a", "b"],
        ["b", "a"],
        ["b", "a"],
        ["a", "x", "b"],
    ]

    def fake_read(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return sentences

    monkeypatch.setattr("word_order_flexibility.corpora.read_sentences", fake_read)
    output = tmp_path / "result.json"
    samples = tmp_path / "samples.json"
    calculate_corpus_file(
        "abc_Latn",
        tmp_path / "unused.txt",
        output,
        input_format="tokenized",
        n_sentences=3,
        sample_output_path=samples,
        sample_seeds=(0, 1),
        sample_sentences=3,
    )

    rows = json.loads(samples.read_text(encoding="utf-8"))
    assert calls == 1
    assert [row["seed"] for row in rows] == [0, 1]
    assert all(row["sentences"] == 3 for row in rows)
