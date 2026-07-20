import re
from pathlib import Path

WORD_RE = re.compile(r"[^\W_]+(?:[’'\-][^\W_]+)*", re.UNICODE)
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def tokenize_text(text: str, *, lowercase: bool = True) -> list[list[str]]:
    sentences = []
    for segment in SENTENCE_BOUNDARY_RE.split(text.strip()):
        words = WORD_RE.findall(segment)
        if lowercase:
            words = [word.casefold() for word in words]
        if words:
            sentences.append(words)
    return sentences


def read_sentences(
    path: str | Path,
    *,
    input_format: str = "text",
    lowercase: bool = True,
) -> list[list[str]]:
    """Read one-sentence-per-line text, or tokenized lines."""
    text = Path(path).read_text(encoding="utf-8")
    if input_format == "text":
        return tokenize_text(text, lowercase=lowercase)
    if input_format not in {"lines", "tokenized"}:
        raise ValueError("input_format must be 'text', 'lines', or 'tokenized'")

    sentences = []
    for line in text.splitlines():
        if input_format == "tokenized":
            words = [word for token in line.split() for word in WORD_RE.findall(token)]
            if lowercase:
                words = [word.casefold() for word in words]
        else:
            words = WORD_RE.findall(line)
            if lowercase:
                words = [word.casefold() for word in words]
        if words:
            sentences.append(words)
    return sentences
