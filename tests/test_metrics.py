from word_order_flexibility.flexibility import word_order_flexibility
from word_order_flexibility.text import read_sentences, tokenize_text


def test_balanced_pair_is_flexible():
    sentences = [
        ["cats", "chase"],
        ["chase", "cats"],
        ["cats", "chase"],
        ["chase", "cats"],
    ]
    result = word_order_flexibility(sentences, decay=0.5)
    assert result["flexibility"] == 1.0
    assert result["qualifying_pairs"] == 1


def test_one_direction_is_rigid():
    sentences = [["small", "dogs"]] * 4
    assert word_order_flexibility(sentences, decay=0.5)["flexibility"] == 0.0


def test_distance_decay_changes_directional_balance():
    sentences = [["a", "b"], ["b", "x", "a"], ["a", "b"]]
    low_decay = word_order_flexibility(sentences, decay=0.25)["flexibility"]
    no_decay = word_order_flexibility(sentences, decay=1.0)["flexibility"]
    assert low_decay < no_decay


def test_tokenized_input_removes_attached_punctuation(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("Cats chase dogs.\nDogs chase cats!\n", encoding="utf-8")
    assert read_sentences(path, input_format="tokenized") == [
        ["cats", "chase", "dogs"],
        ["dogs", "chase", "cats"],
    ]


def test_raw_text_tokenization_removes_punctuation(monkeypatch):
    monkeypatch.setattr(
        "word_order_flexibility.text.segment_text",
        lambda _text: ["Cats chase dogs.", "Dogs chase cats!"],
    )
    assert tokenize_text("Cats chase dogs. Dogs chase cats!") == [
        ["cats", "chase", "dogs"],
        ["dogs", "chase", "cats"],
    ]


def test_raw_text_uses_sentence_segmenter(monkeypatch):
    segmenter = type(
        "Segmenter",
        (),
        {"split": lambda self, _text: ["Dr. Smith left. ", "他回家了。"]},
    )()
    monkeypatch.setattr("word_order_flexibility.text._sentence_segmenter", lambda: segmenter)
    assert len(tokenize_text("ignored")) == 2
