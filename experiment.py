from __future__ import annotations

import argparse
import csv
import json
import logging
from functools import partial
from pathlib import Path

import submitit
from planit import Chain, MailType, Parallel, Plan, SlurmArgs, Step

from word_order_flexibility.corpora import (
    calculate_corpus_file,
    combine_corpus_results,
    combine_corpus_samples,
)
from word_order_flexibility.flexibility import flexibility_profile
from word_order_flexibility.languages import FUTRELL_LANGUAGES, LANGUAGES
from word_order_flexibility.ud import (
    calculate_ud_metrics,
    load_conllu,
    surface_sentences,
)

DECAYS = (0.25, 0.5, 0.75, 1.0)
SENTENCE_SIZES = (1000, 2000, 5000, 10000)


def lambda_label(value: float) -> str:
    return f"{int(round(value * 100)):03d}"


def treebank_paths(ud_root: str, treebanks: str | list[str]) -> list[str]:
    names = [treebanks] if isinstance(treebanks, str) else treebanks
    return [str(Path(ud_root) / name) for name in names]


def run_language(
    language: str,
    treebanks: str | list[str],
    ud_root: str,
    n_sentences: int,
    output: str,
) -> None:
    """Compute the word-level metrics used in the paper experiments."""
    sentences = load_conllu(treebank_paths(ud_root, treebanks), limit=n_sentences)
    if len(sentences) < n_sentences:
        raise ValueError(f"{language}: found {len(sentences)} sentences; need {n_sentences}")
    words = surface_sentences(sentences)
    ud = calculate_ud_metrics(sentences)
    profile = flexibility_profile(words, DECAYS)
    primary = next(row for row in profile if row["lambda"] == 0.5)
    row = {
        "Language": language,
        "n_sentences": n_sentences,
        "HDE": ud["head_direction_entropy_weighted"],
        "ROE": ud["relation_order_entropy_weighted"],
        "HDE_unweighted": ud["head_direction_entropy_unweighted"],
        "ROE_unweighted": ud["relation_order_entropy_unweighted"],
        "type_token_ratio": primary["ttr"],
        "word_types": primary["types"],
        "word_tokens": primary["tokens"],
    }
    for values in profile:
        label = lambda_label(values["lambda"])
        row[f"F_lambda_{label}"] = values["flexibility"]
        row[f"qualifying_pairs_lambda_{label}"] = values["qualifying_pairs"]

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def combine_results(inputs: list[str], output: str) -> None:
    rows = [json.loads(Path(path).read_text(encoding="utf-8")) for path in inputs]
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def count_sentences(paths: list[str], stop_at: int) -> int:
    """Count CoNLL-U sentence blocks without parsing them."""
    count = 0
    for configured in paths:
        path = Path(configured)
        files = []
        for split in ("train", "dev", "test"):
            files.extend(sorted(path.rglob(f"*{split}.conllu")))
        for file in files:
            previous_blank = True
            with file.open(encoding="utf-8") as handle:
                for line in handle:
                    blank = not line.strip()
                    if blank and not previous_blank:
                        count += 1
                        if count >= stop_at:
                            return count
                    previous_blank = blank
            if not previous_blank:
                count += 1
                if count >= stop_at:
                    return count
    return count


def available_languages(ud_root: Path, languages, n_sentences: int):
    return [
        (language, treebanks)
        for language, treebanks in languages
        if count_sentences(treebank_paths(str(ud_root), treebanks), n_sentences) >= n_sentences
    ]


def experiment_branch(name, languages, sizes, ud_root, output_root, metric_resources, small_resources):
    outputs = []
    steps = []
    for size in sizes:
        for language, treebanks in languages:
            output = output_root / name / "languages" / str(size) / f"{language.replace(' ', '_')}.json"
            outputs.append(str(output))
            steps.append(
                Step(
                    f"{name}-{size}-{language}",
                    metric_resources,
                    partial(
                        run_language,
                        language,
                        treebanks,
                        str(ud_root),
                        size,
                        str(output),
                    ),
                )
            )
    combined = output_root / name / "results.csv"
    return Chain(
        Parallel(*steps),
        Step(
            f"combine-{name}",
            small_resources,
            partial(combine_results, outputs, str(combined)),
        ),
    ), combined


def script_splitter(code: str) -> str:
    if "_" not in code:
        raise ValueError(f"Expected a language-script filename such as eng_Latn.txt; got {code}.txt")
    script = code.split("_", 1)[1].split("-", 1)[0]
    if script in {"Hans", "Hant", "Hani"}:
        return "chinese"
    if script in {"Jpan", "Hira", "Kana"}:
        return "japanese"
    if script == "Hang":
        return "korean"
    return "default"


def corpus_branch(
    name,
    input_dir,
    input_format,
    n_sentences,
    sample_count,
    sample_sentences,
    min_sentence_words,
    output_root,
    metric_resources,
    small_resources,
):
    files = sorted(Path(input_dir).glob("*.txt"))
    if name == "bible":
        files = [path for path in files if not path.stem.endswith("-sampled")]
    outputs = []
    sample_outputs = []
    steps = []
    for path in files:
        code = path.stem
        output = output_root / name / "languages" / f"{code}.json"
        sample_output = output_root / name / "samples" / f"{code}.json"
        outputs.append(str(output))
        sample_outputs.append(str(sample_output))
        steps.append(
            Step(
                f"{name}-{code}",
                metric_resources,
                partial(
                    calculate_corpus_file,
                    code,
                    str(path),
                    str(output),
                    input_format=input_format,
                    splitter=script_splitter(code),
                    n_sentences=n_sentences,
                    sample_output_path=str(sample_output),
                    sample_seeds=tuple(range(sample_count)),
                    sample_sentences=sample_sentences,
                    min_sentence_words=min_sentence_words,
                ),
            )
        )
    if not steps:
        raise ValueError(f"No .txt files found in {input_dir}")
    combined = output_root / name / "results.csv"
    combined_samples = output_root / name / "samples.csv"
    return (
        Chain(
            Parallel(*steps),
            Parallel(
                Step(f"combine-{name}", small_resources, partial(combine_corpus_results, outputs, str(combined))),
                Step(
                    f"combine-{name}-samples",
                    small_resources,
                    partial(combine_corpus_samples, sample_outputs, str(combined_samples)),
                ),
            ),
        ),
        combined,
        combined_samples,
    )


def build_plan(args) -> Plan:
    common = {
        "time": args.time,
        "partition": args.partition,
        "cluster": args.cluster,
        "cpus_per_task": args.cpus,
    }
    if args.account:
        common["account"] = args.account
    if args.email:
        common["mail_type"] = [MailType.FAIL]
        common["mail_user"] = args.email
    metric_resources = SlurmArgs(**common)
    small_resources = SlurmArgs(**{**common, "cpus_per_task": 1})

    parallel_branches = []
    if not args.corpora_only:
        broad = available_languages(args.ud_path, list(LANGUAGES.items()), 1000)
        futrell = available_languages(args.ud_path, FUTRELL_LANGUAGES, 1000)
        common_sizes = available_languages(args.ud_path, list(LANGUAGES.items()), max(SENTENCE_SIZES))
        all_branch, _all_results = experiment_branch(
            "all-1000", broad, (1000,), args.ud_path, args.output_dir, metric_resources, small_resources
        )
        futrell_branch, _futrell_results = experiment_branch(
            "futrell-1000", futrell, (1000,), args.ud_path, args.output_dir, metric_resources, small_resources
        )
        size_branch, _size_results = experiment_branch(
            "sentence-sizes",
            common_sizes,
            SENTENCE_SIZES,
            args.ud_path,
            args.output_dir,
            metric_resources,
            small_resources,
        )
        parallel_branches.extend((all_branch, futrell_branch, size_branch))
    corpus_settings = (
        ("flores", args.flores_dir, "lines", 1),
        ("bible", args.bible_dir, "tokenized", 1),
        ("wikipedia", args.wikipedia_dir, "text", args.wikipedia_min_words),
    )
    for name, input_dir, input_format, min_sentence_words in corpus_settings:
        if input_dir is None:
            continue
        branch, _combined, _combined_samples = corpus_branch(
            name,
            input_dir,
            input_format,
            args.corpus_sentences,
            args.corpus_samples,
            args.corpus_sample_sentences,
            min_sentence_words,
            args.output_dir,
            metric_resources,
            small_resources,
        )
        parallel_branches.append(branch)
    if not parallel_branches:
        raise ValueError("No experiment branches selected")
    return Plan("word-order-flexibility", Parallel(*parallel_branches))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ud-path", type=Path, default=Path("ud/ud-treebanks-v2.17"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--flores-dir", type=Path)
    parser.add_argument("--bible-dir", type=Path)
    parser.add_argument("--wikipedia-dir", type=Path)
    parser.add_argument(
        "--wikipedia-min-words",
        type=int,
        default=5,
        help="Minimum tokenized words per Wikipedia sentence",
    )
    parser.add_argument(
        "--corpus-sentences",
        type=int,
        default=1000,
        help="Exact deterministic sentence prefix used for every external corpus language",
    )
    parser.add_argument(
        "--corpus-samples",
        type=int,
        default=10,
        help="Number of deterministic random samples per external-corpus language",
    )
    parser.add_argument(
        "--corpus-sample-sentences",
        type=int,
        default=1000,
        help="Sentences per random stability sample",
    )
    parser.add_argument("--partition", default="batch")
    parser.add_argument("--account")
    parser.add_argument("--cluster", default="wice")
    parser.add_argument("--time", default="00:30:00")
    parser.add_argument("--cpus", type=int, default=1)
    parser.add_argument("--email", help="Email address for Slurm failure notifications")
    parser.add_argument(
        "--corpora-only",
        action="store_true",
        help="Run only external-corpus calculation and combination jobs",
    )
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if args.wikipedia_min_words < 1:
        parser.error("--wikipedia-min-words must be at least 1")
    if min(args.corpus_sentences, args.corpus_sample_sentences) < 1000:
        parser.error("external corpus analyses require at least 1000 sentences per sample")
    if args.corpus_samples < 2:
        parser.error("--corpus-samples must be at least 2")

    corpus_dirs = (args.flores_dir, args.bible_dir, args.wikipedia_dir)
    if args.corpora_only and not any(corpus_dirs):
        parser.error("--corpora-only requires at least one external corpus directory")
    if not args.corpora_only and not args.ud_path.is_dir():
        raise FileNotFoundError(f"UD directory not found: {args.ud_path}; run download_ud.py first")
    for corpus_dir in corpus_dirs:
        if corpus_dir is not None and not corpus_dir.is_dir():
            raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")
    plan = build_plan(args)
    plan.describe()
    if args.submit or args.debug:
        executor = submitit.AutoExecutor(
            folder=str(args.output_dir / "slurm_logs"),
            cluster="debug" if args.debug else None,
        )
        plan.submit(executor)
        if args.debug:
            plan.wait()


if __name__ == "__main__":
    main()
