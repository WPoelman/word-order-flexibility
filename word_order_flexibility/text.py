import re
from functools import lru_cache
from pathlib import Path

from sacrebleu.tokenizers.tokenizer_intl import TokenizerV14International
from sacrebleu.tokenizers.tokenizer_ja_mecab import TokenizerJaMecab
from sacrebleu.tokenizers.tokenizer_ko_mecab import TokenizerKoMecab
from sacrebleu.tokenizers.tokenizer_spm import Flores200Tokenizer
from sacrebleu.tokenizers.tokenizer_zh import TokenizerZh

WORD_RE = re.compile(r"[^\W_]+(?:[’'\-][^\W_]+)*", re.UNICODE)
ASIAN_PUNCT = r"([\u3001\u3002\u3008-\u3011\u3014-\u301f\uff61-\uff65\u30fb])"
FULL_WIDTH_PUNCT = r"([\uff0e\uff0c\uff1f\uff1a\uff1b\uff01\uff02\uff08\uff09])"
SPLITTER_MAPPING = {
    "chinese": TokenizerZh(),
    "japanese": TokenizerJaMecab(),
    "korean": TokenizerKoMecab(),
    "default": TokenizerV14International(),
    "flores-spm": Flores200Tokenizer(),
}


# copied from sacrebleu.tokenizers.tokenizer_ter.py
def _remove_asian_punct(sent: str) -> str:
    sent = re.sub(ASIAN_PUNCT, r" \1 ", sent)
    sent = re.sub(FULL_WIDTH_PUNCT, r" \1 ", sent)
    return sent


def _remove_punct(sent: str) -> str:
    return re.sub(r"[\.,\?:;!\"\(\)]", "", sent)


@lru_cache(maxsize=1)
def _sentence_segmenter():
    import onnxruntime as ort
    from wtpsplit import SaT

    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    return SaT(
        "sat-3l-sm",
        ort_providers=["CPUExecutionProvider"],
        ort_kwargs={"sess_options": session_options},
    )


def segment_text(text: str) -> list[str]:
    """Segment raw text with the multilingual SaT sentence model."""
    return [segment.strip() for segment in _sentence_segmenter().split(text) if segment.strip()]


def tokenize_text(text: str, *, lowercase: bool = True, apply_splitter: str = "default") -> list[list[str]]:
    sentences = []
    tokenizer = SPLITTER_MAPPING[apply_splitter]
    for segment in segment_text(text):
        if apply_splitter == "flores-spm":
            segment = tokenizer(line=segment)
        else:
            segment = tokenizer(segment)
        if apply_splitter in {"chinese", "japanese", "korean"}:
            segment = _remove_asian_punct(segment)
        elif apply_splitter == "default":
            segment = _remove_punct(segment)
        words = segment.split()
        if lowercase:
            words = [word.casefold() for word in words]
        if words:
            sentences.append(words)
    return sentences


def tokenize_sentence(sentence: str, *, lowercase: bool = True, apply_splitter: str = "default") -> list[str]:
    tokenizer = SPLITTER_MAPPING[apply_splitter]
    if apply_splitter == "flores-spm":
        sentence = tokenizer(line=sentence)
    else:
        sentence = tokenizer(sentence)
    if apply_splitter in {"chinese", "japanese", "korean"}:
        sentence = _remove_asian_punct(sentence)
    elif apply_splitter == "default":
        sentence = _remove_punct(sentence)
    words = sentence.split()
    if lowercase:
        words = [word.casefold() for word in words]
    return words


def read_sentences(
    path: str | Path,
    *,
    input_format: str = "text",
    lowercase: bool = True,
    apply_splitter: str = "default",
) -> list[list[str]]:
    """Read raw text, one-sentence-per-line text, or tokenized lines."""
    text = Path(path).read_text(encoding="utf-8")
    if input_format == "text":
        return tokenize_text(text, lowercase=lowercase, apply_splitter=apply_splitter)
    if input_format not in {"lines", "tokenized"}:
        raise ValueError("input_format must be 'text', 'lines', or 'tokenized'")

    sentences = []
    for line in text.splitlines():
        if input_format == "tokenized":
            words = _remove_punct(line).split()
            if lowercase:
                words = [word.casefold() for word in words]
        else:
            words = tokenize_sentence(line, lowercase=lowercase, apply_splitter=apply_splitter)
        if words:
            sentences.append(words)
    return sentences
