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
    merge_csv_files,
    valid_corpus_outputs,
)
from word_order_flexibility.disjoint import (
    DISJOINT_CODES,
    TOTAL_SENTENCES,
    calculate_disjoint_text,
    calculate_disjoint_ud,
    combine_disjoint_results,
    valid_disjoint_result,
)
from word_order_flexibility.flexibility import flexibility_profile
from word_order_flexibility.languages import LANGUAGES, UD_MAPPING
from word_order_flexibility.ud import (
    calculate_ud_metrics,
    count_single_root_sentences,
    load_conllu,
    surface_sentences,
)

DECAYS = (0.25, 0.5, 0.75, 1.0)
SENTENCE_SIZES = (1000, 2000, 5000, 10000)
DEPENDENCY_BATCH_SIZE = 50
UD_EXPERIMENT_VERSION = 3


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
    lowercase: bool = True,
) -> None:
    """Compute the word-level metrics used in the paper experiments."""
    sentences = load_conllu(treebank_paths(ud_root, treebanks), limit=n_sentences, single_root_only=True)
    if len(sentences) < n_sentences:
        raise ValueError(f"{language}: found {len(sentences)} sentences; need {n_sentences}")
    words = surface_sentences(sentences, lowercase=lowercase)
    ud = calculate_ud_metrics(sentences)
    profile = flexibility_profile(words, DECAYS)
    primary = next(row for row in profile if row["lambda"] == 0.5)
    row = {
        "experiment_version": UD_EXPERIMENT_VERSION,
        "casefolded": lowercase,
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


def available_languages(ud_root: Path, languages, n_sentences: int):
    return [
        (language, treebanks)
        for language, treebanks in languages
        if count_single_root_sentences(treebank_paths(str(ud_root), treebanks), n_sentences) >= n_sentences
    ]


def valid_ud_result(path: Path, language: str, n_sentences: int, *, lowercase: bool) -> bool:
    """Return whether a cached UD result matches the current experiment."""
    if not path.is_file():
        return False
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        isinstance(row, dict)
        and row.get("experiment_version") == UD_EXPERIMENT_VERSION
        and row.get("Language") == language
        and row.get("n_sentences") == n_sentences
        and row.get("casefolded") is lowercase
    )


def experiment_branch(
    name,
    languages,
    sizes,
    ud_root,
    output_root,
    metric_resources,
    small_resources,
    *,
    lowercase=True,
):
    jobs = []
    for size in sizes:
        for language, treebanks in languages:
            output = output_root / name / "languages" / str(size) / f"{language.replace(' ', '_')}.json"
            jobs.append((language, treebanks, size, output))

    batch_branches = []
    batch_results = []
    for batch_index, start in enumerate(range(0, len(jobs), DEPENDENCY_BATCH_SIZE)):
        batch = jobs[start : start + DEPENDENCY_BATCH_SIZE]
        outputs = [str(output) for _language, _treebanks, _size, output in batch]
        steps = []
        for language, treebanks, size, output in batch:
            if valid_ud_result(output, language, size, lowercase=lowercase):
                continue
            steps.append(
                Step(
                    f"{name}-{size}-{language}",
                    metric_resources,
                    partial(run_language, language, treebanks, str(ud_root), size, str(output), lowercase),
                )
            )
        batch_result = output_root / name / "_batches" / f"{batch_index:03d}-results.csv"
        batch_results.append(str(batch_result))
        collector = Step(
            f"combine-{name}-{batch_index:03d}",
            small_resources,
            partial(combine_results, outputs, str(batch_result)),
        )
        batch_branches.append(Chain(Parallel(*steps), collector) if steps else collector)

    combined = output_root / name / "results.csv"
    final_collector = Step(
        f"combine-{name}",
        small_resources,
        partial(merge_csv_files, batch_results, str(combined)),
    )
    return Chain(Parallel(*batch_branches), final_collector), combined


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


def valid_json(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return True


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
    if not files:
        raise ValueError(f"No .txt files found in {input_dir}")

    batch_branches = []
    batch_results = []
    batch_samples = []
    for batch_index, start in enumerate(range(0, len(files), DEPENDENCY_BATCH_SIZE)):
        batch = files[start : start + DEPENDENCY_BATCH_SIZE]
        outputs = []
        sample_outputs = []
        language_steps = []
        for path in batch:
            code = path.stem
            output = output_root / name / "languages" / f"{code}.json"
            sample_output = output_root / name / "samples" / f"{code}.json"
            outputs.append(str(output))
            sample_outputs.append(str(sample_output))
            if valid_corpus_outputs(
                output,
                sample_output,
                code=code,
                n_sentences=n_sentences,
                sample_count=sample_count,
                sample_sentences=sample_sentences,
                min_sentence_words=min_sentence_words,
            ):
                continue
            language_steps.append(
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

        batch_result = output_root / name / "_batches" / f"{batch_index:03d}-results.csv"
        batch_sample = output_root / name / "_batches" / f"{batch_index:03d}-samples.csv"
        batch_results.append(str(batch_result))
        batch_samples.append(str(batch_sample))
        collectors = Parallel(
            Step(
                f"combine-{name}-{batch_index:03d}",
                small_resources,
                partial(combine_corpus_results, outputs, str(batch_result)),
            ),
            Step(
                f"combine-{name}-{batch_index:03d}-samples",
                small_resources,
                partial(combine_corpus_samples, sample_outputs, str(batch_sample)),
            ),
        )
        batch_branches.append(Chain(Parallel(*language_steps), collectors) if language_steps else collectors)

    combined = output_root / name / "results.csv"
    combined_samples = output_root / name / "samples.csv"
    final_collectors = Parallel(
        Step(f"combine-{name}", small_resources, partial(merge_csv_files, batch_results, str(combined))),
        Step(
            f"combine-{name}-samples",
            small_resources,
            partial(merge_csv_files, batch_samples, str(combined_samples)),
        ),
    )
    return Chain(Parallel(*batch_branches), final_collectors), combined, combined_samples


def disjoint_domain_branch(args, metric_resources, wikipedia_resources, small_resources):
    """Build three parallel branches for the matched disjoint-sample study."""
    root = args.output_dir / "disjoint-domain-study"
    branches = []

    ud_outputs = []
    ud_steps = []
    for code in DISJOINT_CODES:
        language = UD_MAPPING[code]
        paths = treebank_paths(str(args.ud_path), LANGUAGES[language])
        available = count_single_root_sentences(paths, TOTAL_SENTENCES)
        if available < TOTAL_SENTENCES:
            raise ValueError(f"{code}: found {available} single-root UD sentences; need {TOTAL_SENTENCES}")
        output = root / "ud" / "languages" / f"{code}.json"
        ud_outputs.append(str(output))
        if not valid_disjoint_result(output, code):
            ud_steps.append(
                Step(
                    f"disjoint-ud-{code}",
                    metric_resources,
                    partial(
                        calculate_disjoint_ud,
                        code,
                        paths,
                        str(output),
                    ),
                )
            )
    ud_combined = root / "ud" / "results.csv"
    ud_collector = Step(
        "combine-disjoint-ud",
        small_resources,
        partial(combine_disjoint_results, ud_outputs, str(ud_combined)),
    )
    branches.append(Chain(Parallel(*ud_steps), ud_collector) if ud_steps else ud_collector)

    corpus_settings = (
        ("bible", args.bible_dir, "tokenized", 1, metric_resources),
        ("wikipedia", args.wikipedia_dir, "text", args.wikipedia_min_words, wikipedia_resources),
    )
    for name, input_dir, input_format, min_words, resources in corpus_settings:
        files = {path.stem: path for path in Path(input_dir).glob("*.txt")}
        if name == "bible":
            files = {code: path for code, path in files.items() if not code.endswith("-sampled")}
        missing = sorted(set(DISJOINT_CODES) - set(files))
        if missing:
            raise FileNotFoundError(f"{name}: missing input files for {', '.join(missing)}")
        outputs = []
        steps = []
        for code in DISJOINT_CODES:
            output = root / name / "languages" / f"{code}.json"
            outputs.append(str(output))
            if valid_disjoint_result(output, code):
                continue
            steps.append(
                Step(
                    f"disjoint-{name}-{code}",
                    resources,
                    partial(
                        calculate_disjoint_text,
                        code,
                        str(files[code]),
                        str(output),
                        input_format=input_format,
                        splitter=script_splitter(code),
                        min_sentence_words=min_words,
                    ),
                )
            )
        combined = root / name / "results.csv"
        collector = Step(
            f"combine-disjoint-{name}",
            small_resources,
            partial(combine_disjoint_results, outputs, str(combined)),
        )
        branches.append(Chain(Parallel(*steps), collector) if steps else collector)
    return Parallel(*branches)


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
    wikipedia_resources = SlurmArgs(**{**common, "cpus_per_task": args.wikipedia_cpus})

    if args.disjoint_domain_study:
        return Plan(
            "word-order-flexibility-disjoint",
            disjoint_domain_branch(args, metric_resources, wikipedia_resources, small_resources),
        )

    parallel_branches = []
    if not args.corpora_only:
        broad = available_languages(args.ud_path, list(LANGUAGES.items()), 1000)
        common_sizes = available_languages(args.ud_path, list(LANGUAGES.items()), max(SENTENCE_SIZES))
        for suffix, lowercase in (("", True), ("-cased", False)):
            all_branch, _all_results = experiment_branch(
                f"all-1000{suffix}",
                broad,
                (1000,),
                args.ud_path,
                args.output_dir,
                metric_resources,
                small_resources,
                lowercase=lowercase,
            )
            size_branch, _size_results = experiment_branch(
                f"sentence-sizes{suffix}",
                common_sizes,
                SENTENCE_SIZES,
                args.ud_path,
                args.output_dir,
                metric_resources,
                small_resources,
                lowercase=lowercase,
            )
            parallel_branches.extend((all_branch, size_branch))
    corpus_settings = (
        ("bible", args.bible_dir, "tokenized", 1, metric_resources),
        ("wikipedia", args.wikipedia_dir, "text", args.wikipedia_min_words, wikipedia_resources),
    )
    for name, input_dir, input_format, min_sentence_words, resources in corpus_settings:
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
            resources,
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
    parser.add_argument("--bible-dir", type=Path)
    parser.add_argument("--wikipedia-dir", type=Path)
    parser.add_argument(
        "--wikipedia-cpus",
        type=int,
        default=8,
        help="CPUs per Wikipedia job; also controls memory on CPU-scaled clusters",
    )
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
        "--disjoint-domain-study",
        action="store_true",
        help="Run only the matched five-by-1,000 disjoint UD, Bible, and Wikipedia study",
    )
    parser.add_argument(
        "--corpora-only",
        action="store_true",
        help="Run only external-corpus calculation and combination jobs",
    )
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if args.wikipedia_cpus < 1:
        parser.error("--wikipedia-cpus must be at least 1")
    if args.wikipedia_min_words < 1:
        parser.error("--wikipedia-min-words must be at least 1")
    if min(args.corpus_sentences, args.corpus_sample_sentences) < 1000:
        parser.error("external corpus analyses require at least 1000 sentences per sample")
    if args.corpus_samples < 2:
        parser.error("--corpus-samples must be at least 2")

    if args.disjoint_domain_study and (args.bible_dir is None or args.wikipedia_dir is None):
        parser.error("--disjoint-domain-study requires --bible-dir and --wikipedia-dir")
    if args.disjoint_domain_study and args.corpora_only:
        parser.error("--disjoint-domain-study cannot be combined with --corpora-only")

    corpus_dirs = (args.bible_dir, args.wikipedia_dir)
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
