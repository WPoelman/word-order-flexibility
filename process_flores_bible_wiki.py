from __future__ import annotations

import argparse
import filecmp
import os
import shutil
from collections import defaultdict
from pathlib import Path

import regex
from datasets import load_dataset
from qq import Database, Languoid
from tqdm import tqdm


def wikipedia_languages() -> dict[str, tuple[str, list[str]]]:
    """Map Wikipedia edition codes to ISO 639-3 and ISO 15924 codes."""
    database = Database.load()
    mappings = {}
    for language in database.query(Languoid):
        if language.wikipedia and language.wikipedia.code and language.iso_639_3 and language.script_codes:
            mappings[language.wikipedia.code] = (language.iso_639_3, language.script_codes)
    return mappings


def save_flores(output_dir: str | Path = "data/flores") -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    datasets = load_dataset("openlanguagedata/flores_plus", split=["dev", "devtest"])

    texts_by_language = defaultdict(list)
    for dataset in datasets:
        for language, script, text in zip(
            dataset["iso_639_3"],
            dataset["iso_15924"],
            dataset["text"],
            strict=True,
        ):
            texts_by_language[f"{language}_{script}"].append(text)

    for code, texts in tqdm(texts_by_language.items(), desc="Writing FLORES files"):
        (output / f"{code}.txt").write_text("\n".join(texts) + "\n", encoding="utf-8")


SCRIPT_ALIASES = {
    "Hanb": "Hani",
    "Hans": "Hani",
    "Hant": "Hani",
    "Kore": "Hang",
    "Cyrs": "Cyrl",
    "Syre": "Syrc",
    "Syrj": "Syrc",
    "Syrn": "Syrc",
}


def dominant_script(text: str, scripts: list[str]) -> str:
    """Select the listed script accounting for the most characters."""
    scores = []
    for position, script in enumerate(scripts):
        unicode_script = SCRIPT_ALIASES.get(script, script)
        try:
            count = len(regex.findall(rf"\p{{sc={unicode_script}}}", text))
        except regex.error:
            count = 0
        scores.append((count, -position, script))
    return max(scores)[2]


def normalize_wikipedias(input_dir: str | Path, output_dir: str | Path) -> None:
    """Create one canonical, dominant-script file per existing Wikipedia edition."""
    source = Path(input_dir)
    output = Path(output_dir)
    if source.resolve() == output.resolve():
        raise ValueError("Use a different output directory so normalization is non-destructive")
    output.mkdir(parents=True, exist_ok=True)

    script_order = {}
    for _wiki_code, (language, scripts) in wikipedia_languages().items():
        script_order.setdefault(language, scripts)

    groups = defaultdict(list)
    for path in sorted(source.glob("*.txt")):
        if "_" not in path.stem:
            raise ValueError(f"Expected language-script filename; got {path.name}")
        language, script = path.stem.split("_", 1)
        groups[language].append((script, path))

    if not groups:
        raise ValueError(f"No .txt files found in {source}")

    for language, variants in tqdm(groups.items(), desc="Normalizing Wikipedia files"):
        reference = variants[0][1]
        for _script, candidate in variants[1:]:
            if not filecmp.cmp(reference, candidate, shallow=False):
                raise ValueError(f"{language}: script variants are not identical; refusing to discard data")
        available = {script for script, _path in variants}
        candidates = [script for script in script_order.get(language, []) if script in available]
        if not candidates:
            candidates = [script for script, _path in variants]
        with reference.open(encoding="utf-8") as handle:
            sample = handle.read(2_000_000)
        selected_script = dominant_script(sample, candidates)
        selected = next(path for script, path in variants if script == selected_script)
        target = output / f"{language}_{selected_script}.txt"
        if target.exists():
            raise FileExistsError(f"Output already exists: {target}")
        try:
            os.link(selected, target)
        except OSError:
            shutil.copy2(selected, target)
    print(f"Wrote {len(groups)} canonical Wikipedia files to {output}")


def save_wikipedias(
    output_dir: str | Path = "data/wikipedia",
    *,
    documents: int = 2000,
    seed: int = 0,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for wiki_code, (language, scripts) in tqdm(
        wikipedia_languages().items(),
        desc="Processing Wikipedia datasets",
    ):
        existing = list(output.glob(f"{language}_*.txt"))
        if existing:
            continue
        try:
            dataset = load_dataset(
                "WikiQuality/pre_filtered",
                wiki_code,
                streaming=True,
                split="train",
            ).shuffle(seed=seed)
            texts = [row["text"] for row in dataset.take(documents)]
        except Exception as error:
            print(f"Failed to load {language} ({wiki_code}): {error}")
            continue
        rendered = "\n".join(texts) + "\n"
        script = dominant_script(rendered[:2_000_000], scripts)
        (output / f"{language}_{script}.txt").write_text(rendered, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", choices=("flores", "wikipedia", "normalize-wikipedia"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--documents", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.corpus == "flores":
        save_flores(args.output_dir or Path("data/flores"))
    elif args.corpus == "normalize-wikipedia":
        if args.input_dir is None or args.output_dir is None:
            parser.error("normalize-wikipedia requires --input-dir and --output-dir")
        normalize_wikipedias(args.input_dir, args.output_dir)
    else:
        save_wikipedias(
            args.output_dir or Path("data/wikipedia"),
            documents=args.documents,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
