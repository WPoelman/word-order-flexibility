import argparse
import csv
import json
from pathlib import Path

from .analysis import correlations
from .corpora import analyze_corpora, analyze_corpus_stability
from .flexibility import flexibility_profile, word_order_flexibility
from .plotting import generate_paper_artifacts, profile_plot, validation_plot
from .text import read_sentences
from .ud import calculate_ud_metrics, load_conllu, surface_sentences


def _write_json(data, path: str | None) -> None:
    rendered = json.dumps(data, indent=2, ensure_ascii=False)
    if path:
        Path(path).write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


def _add_metric_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--lambda",
        dest="decays",
        type=float,
        action="append",
        help="Decay value; repeat for a profile",
    )
    parser.add_argument("--min-pair-count", type=int, default=3)


def _metric_results(sentences, args):
    decays = args.decays or [0.5]
    if len(decays) == 1:
        return word_order_flexibility(
            sentences,
            decay=decays[0],
            min_pair_count=args.min_pair_count,
            return_pairs=args.pairs,
        )
    if args.pairs:
        raise SystemExit("--pairs can only be used with one --lambda value")
    return flexibility_profile(sentences, decays, min_pair_count=args.min_pair_count)


def text_command(args) -> None:
    sentences = read_sentences(
        args.input, input_format=args.format, lowercase=not args.keep_case, apply_splitter=args.apply_splitter
    )
    _write_json(_metric_results(sentences, args), args.output)


def ud_command(args) -> None:
    ud_sentences = load_conllu(args.inputs, args.sentences)
    words = surface_sentences(ud_sentences)
    ud_metrics = calculate_ud_metrics(ud_sentences)
    flexibility = _metric_results(words, args)
    result = (
        {**ud_metrics, **flexibility}
        if isinstance(flexibility, dict)
        else {**ud_metrics, "flexibility_profile": flexibility}
    )
    _write_json(result, args.output)


def analyze_command(args) -> None:
    with Path(args.input).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row.setdefault("HDE", row.get("head_direction_entropy_weighted"))
        row.setdefault("ROE", row.get("relation_order_entropy_weighted"))
    results = [
        correlations(rows, target, control_ttr=control) for target in ("HDE", "ROE") for control in (False, True)
    ]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(results, str(output_dir / "correlations.json"))
    validation_plot(rows, output_dir / "validation.png", control_ttr=not args.raw_plot)
    if rows and {"language", "lambda"}.issubset(rows[0]):
        profile_plot(rows, output_dir / "decay_profile.png")


def corpus_analysis_command(args) -> None:
    root = Path(args.results_dir)
    output = Path(args.output_dir) if args.output_dir else root / "corpus-analysis"
    corpus_results = []
    corpus_samples = []
    for name in ("flores", "bible", "wikipedia"):
        results = root / name / "results.csv"
        samples = root / name / "samples.csv"
        if results.is_file():
            corpus_results.append((name, str(results)))
        if samples.is_file():
            corpus_samples.append((name, str(samples)))
    if not corpus_results:
        raise FileNotFoundError(f"No combined corpus results found under {root}")
    ud_results = root / "all-1000" / "results.csv"
    if not ud_results.is_file():
        raise FileNotFoundError(f"UD validation results not found: {ud_results}")
    analyze_corpora(corpus_results, ud_results, output)
    if corpus_samples:
        analyze_corpus_stability(corpus_samples, ud_results, output)


def paper_command(args) -> None:
    root = Path(args.results_dir)
    output = Path(args.output_dir) if args.output_dir else root / "paper"
    generate_paper_artifacts(
        str(root / "all-1000" / "results.csv"),
        str(root / "futrell-1000" / "results.csv"),
        str(root / "sentence-sizes" / "results.csv"),
        args.bakker,
        str(output),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="word-flexibility")
    commands = parser.add_subparsers(dest="command", required=True)

    text_parser = commands.add_parser("text", help="Calculate F(lambda) from raw text")
    text_parser.add_argument("input")
    text_parser.add_argument("--format", choices=("text", "lines", "tokenized"), default="text")
    text_parser.add_argument(
        "--apply-splitter",
        choices=("chinese", "japanese", "korean", "flores-spm", "default"),
        default="default",
        help="Use this flag to split Chinese, Japanese, and Korean characters into separate tokens. \
                             The flores-spm option uses the SentencePiece model trained on FLORES-200 data to tokenize all languages.",
    )
    text_parser.add_argument("--keep-case", action="store_true")
    text_parser.add_argument("--pairs", action="store_true", help="Include qualifying pair-level values")
    text_parser.add_argument("--output")
    _add_metric_arguments(text_parser)
    text_parser.set_defaults(function=text_command)

    ud_parser = commands.add_parser("ud", help="Calculate HDE, ROE, and F(lambda) from CoNLL-U")
    ud_parser.add_argument("inputs", nargs="+")
    ud_parser.add_argument("--sentences", type=int)
    ud_parser.add_argument("--pairs", action="store_true")
    ud_parser.add_argument("--output")
    _add_metric_arguments(ud_parser)
    ud_parser.set_defaults(function=ud_command)

    analysis_parser = commands.add_parser("analyze", help="Correlate flexibility with HDE and ROE")
    analysis_parser.add_argument(
        "input",
        help="CSV with language, flexibility, ttr, and the weighted entropy columns",
    )
    analysis_parser.add_argument("--output-dir", default="analysis")
    analysis_parser.add_argument(
        "--raw-plot",
        action="store_true",
        help="Plot raw instead of TTR-controlled values",
    )
    analysis_parser.set_defaults(function=analyze_command)

    corpus_parser = commands.add_parser("corpus-analysis", help="Generate external-corpus tables and plots")
    corpus_parser.add_argument("results_dir", help="Experiment output directory")
    corpus_parser.add_argument("--output-dir", help="Default: RESULTS_DIR/corpus-analysis")
    corpus_parser.set_defaults(function=corpus_analysis_command)

    paper_parser = commands.add_parser("paper", help="Generate all paper artifacts")
    paper_parser.add_argument("results_dir", help="Experiment output directory")
    paper_parser.add_argument(
        "--bakker",
        default="data/bakker-flexibility-metrics.csv",
        help="Bakker ratings CSV",
    )
    paper_parser.add_argument("--output-dir", help="Default: RESULTS_DIR/paper")
    paper_parser.set_defaults(function=paper_command)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
