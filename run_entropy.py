"""Reproducible entropy experiments: paper, baseline, ud, flores, stability, sizes."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
from scipy import stats

from word_order_flexibility.entropy import (
    METRIC_DEFINITION, UD_PREPROCESSING as PREPROCESSING, _summary,
    entropy_convergence, sentence_order_entropy,
)
from word_order_flexibility.flexibility import flexibility_profile
from word_order_flexibility.languages import LANGUAGES
from word_order_flexibility.text import tokenize_sentence
from word_order_flexibility.ud import calculate_ud_metrics, load_conllu, surface_sentences

ROOT = Path("results/entropy/ud-paper-aligned")
FRACTIONS = (70, 80, 90, 100)


def map_jobs(function, jobs, workers):
    if workers == 1:
        return list(map(function, jobs))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(function, jobs))


def run_entropy_language(job):
    name, paths, args = job
    if args.kind == 'ud':
        sentences = surface_sentences(load_conllu(paths, args.sentences, single_root_only=True))
    else:
        script = name.split('_')[1]
        splitter = {'Hans': 'chinese', 'Hant': 'chinese', 'Jpan': 'japanese', 'Hang': 'korean'}.get(script, 'default')
        texts = []
        for path in paths:
            if path.suffix == '.jsonl':
                texts.extend(json.loads(line)['text'] for line in path.read_text().splitlines() if line.strip())
            else:
                texts.extend(path.read_text().splitlines())
        sentences = [tokenize_sentence(t, apply_splitter=splitter) for t in texts[:args.sentences]]
    if sum(bool(s) for s in sentences) < 4:
        return {'language': name, 'status': 'insufficient_sentences'}
    digest = hashlib.sha256(json.dumps(sentences, ensure_ascii=False).encode()).hexdigest()
    path = Path(args.output_dir) / f'{name}.json'
    if path.exists():
        result = json.loads(path.read_text())
        if (result.get('input_sha256') != digest or result.get('seed') != args.seed
                or result['full'].get('metric_definition') != METRIC_DEFINITION
                or result['full']['repeats'] != args.repeats
                or (args.kind == 'ud' and result.get('preprocessing') != PREPROCESSING)):
            raise ValueError(f'Existing result settings differ: {path}; use a new output directory')
    else:
        result = entropy_convergence(sentences, repeats=args.repeats, seed=args.seed)
        result.update(language=name, corpus=args.kind, input_sha256=digest,
                      preprocessing=PREPROCESSING if args.kind == 'ud' else None,
                      source_files=[str(p.resolve()) for p in paths],
                      dit_revision='42bdafabb94402eb2be00ff2f531e0f930cc0ed7',
                      dataset_revision=('UD 2.17' if args.kind == 'ud' else
                                        (args.input_dir.parent / 'revision.txt').read_text().strip()
                                        if (args.input_dir.parent / 'revision.txt').exists() else 'unrecorded'))
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(result, indent=2) + '\n')
        temporary.replace(path)
    row = {'language': name, 'status': 'ok', 'sentences': result['full']['sentences'],
           'tokens': result['full']['tokens'], 'repeats': args.repeats}
    for key in ('intact', 'within_shuffled', 'difference'):
        row[key] = result['full'][key]['mean']
        row[key + '_mc_se'] = result['full'][key]['mc_se']
        row[key + '_half_full_pass'] = result['diagnostics'][key]['passes_half_full']
        row[key + '_max_relative_drift'] = max(result['diagnostics'][key]['half_full_relative_drift'] or [float('nan')])
    print(f'{args.kind} {name}: difference={row["difference"]:.4f}', flush=True)
    return row

def calculate_baseline(job):
    name, args = job
    destination = args.output_dir / 'baseline' / f'{name}.json'
    raw = load_conllu([args.ud_dir / p for p in LANGUAGES[name]], args.sentences, single_root_only=True)
    if len(raw) < args.sentences:
        return None
    words = surface_sentences(raw)
    digest = hashlib.sha256(json.dumps(words, ensure_ascii=False).encode()).hexdigest()
    if destination.exists():
        row = json.loads(destination.read_text())
        if row['input_sha256'] != digest or row['preprocessing_version'] != PREPROCESSING['version']:
            raise ValueError(f'Stale baseline: {destination}')
        return row
    metrics = calculate_ud_metrics(raw)
    profile = flexibility_profile(words)
    row = dict(Language=name, n_sentences=args.sentences, input_sha256=digest,
               preprocessing_version=PREPROCESSING['version'], casefolded=True, single_root_only=True,
               HDE=metrics['head_direction_entropy_weighted'], ROE=metrics['relation_order_entropy_weighted'],
               HDE_unweighted=metrics['head_direction_entropy_unweighted'], ROE_unweighted=metrics['relation_order_entropy_unweighted'],
               type_token_ratio=profile[1]['ttr'], word_tokens=profile[1]['tokens'], word_types=profile[1]['types'])
    for score in profile:
        label=f"{round(score['lambda']*100):03d}"
        row[f'F_lambda_{label}']=score['flexibility']
        row[f'qualifying_pairs_lambda_{label}']=score['qualifying_pairs']
    atomic_json(destination, row)
    print(f'BASELINE {name}', flush=True)
    return row

def retained_trials(full, repeats):
    """Summarize the same initial Monte Carlo trials, without rerunning the full sample."""
    if len(full['trials']) < repeats:
        raise ValueError('Requested repeats exceed the cached full-sample trials')
    trials = full['trials'][:repeats]
    return {'metric_definition': METRIC_DEFINITION, 'sentences': full['sentences'], 'tokens': full['tokens'], 'repeats': repeats,
            'seed': full['seed'], 'trials': trials,
            **{key: _summary([trial[key] for trial in trials])
               for key in ('intact', 'within_shuffled', 'difference')}}

def run_stability_language(job):
    language, args = job
    old = json.loads((args.entropy_dir / f'{language}.json').read_text())
    corpus = surface_sentences(load_conllu([args.ud_dir / p for p in LANGUAGES[language]], args.sentences, single_root_only=True))
    digest = hashlib.sha256(json.dumps(corpus, ensure_ascii=False).encode()).hexdigest()
    if old['full'].get('metric_definition') != METRIC_DEFINITION or digest != old['input_sha256'] or old.get('preprocessing') != PREPROCESSING:
        raise ValueError(f'{language}: source corpus differs from cached entropy run')
    if old['corpus'] != 'ud' or old['seed'] != args.seed or old['full']['seed'] != args.seed:
        raise ValueError(f'{language}: cached corpus/seed mismatch')
    corpus = [s for s in corpus if s]
    random.Random(f'subsets:{args.seed}').shuffle(corpus)
    path = args.output_dir / 'languages' / f'{language}.json'
    settings = {'fractions': list(FRACTIONS), 'repeats': args.repeats, 'seed': args.seed,
                'input_sha256': digest, 'design_version': 2, 'metric_definition': METRIC_DEFINITION, 'preprocessing': PREPROCESSING}
    if path.exists():
        result = json.loads(path.read_text())
        if result['settings'] != settings:
            raise ValueError(f'{language}: cached settings differ; choose a new output directory')
    else:
        result = {'language': language, 'settings': settings, 'sizes': {},
                  'dit_revision': old.get('dit_revision'), 'dataset_revision': old.get('dataset_revision')}
    for percentage in FRACTIONS:
        key = str(percentage)
        if key in result['sizes']:
            continue
        size = len(corpus) * percentage // 100
        if percentage == 100:
            score = retained_trials(old['full'], args.repeats)
        else:
            score = sentence_order_entropy(corpus[:size], repeats=args.repeats, seed=args.seed)
        result['sizes'][key] = score
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(result, indent=2) + '\n')
        temporary.replace(path)
    print(f'Completed {language}', flush=True)
    return result

def analyze(results, output):
    rows = []
    for result in results:
        for percentage, score in result['sizes'].items():
            rows.append({'language': result['language'], 'percentage': int(percentage),
                         'sentences': score['sentences'], 'tokens': score['tokens'],
                         'repeats': score['repeats'],
                         **{f'{key}_{stat}': score[key][stat]
                            for key in ('intact', 'within_shuffled', 'difference') for stat in ('mean', 'mc_se')}})
    frame = pd.DataFrame(rows).sort_values(['language', 'percentage'])
    frame.to_csv(output / 'scores.csv', index=False)
    pairs = []
    for metric in ('difference', 'intact', 'within_shuffled'):
        wide = frame.pivot(index='language', columns='percentage', values=f'{metric}_mean')[list(FRACTIONS)]
        if wide.isna().any().any():
            raise ValueError('All sizes must have exactly the same languages')
        wide.to_csv(output / f'{metric}_scores_wide.csv')
        for method in ('pearson', 'spearman'):
            wide.corr(method=method).to_csv(output / f'{metric}_{method}.csv')
        ranks = wide.rank(ascending=False)
        for i, left in enumerate(FRACTIONS):
            for right in FRACTIONS[i + 1:]:
                x, y = wide[left].to_numpy(), wide[right].to_numpy()
                pearson, spearman = stats.pearsonr(x, y), stats.spearmanr(x, y)
                shift = (ranks[right] - ranks[left]).abs()
                pairs.append({'metric': metric, 'from_percent': left, 'to_percent': right, 'n': len(wide),
                              'pearson_r': pearson.statistic, 'spearman_rho': spearman.statistic,
                              'mean_change_bits_per_token': np.mean(y - x),
                              'median_absolute_change': np.median(abs(y - x)),
                              'mean_absolute_change': np.mean(abs(y - x)),
                              'rmse': np.sqrt(np.mean((y - x) ** 2)),
                              'median_absolute_rank_change': shift.median(),
                              'max_absolute_rank_change': shift.max()})
    pd.DataFrame(pairs).to_csv(output / 'pairwise_comparisons.csv', index=False)
    wide = frame.pivot(index='language', columns='percentage', values='difference_mean')[list(FRACTIONS)]
    main = pd.DataFrame(pairs).query("metric == 'difference'")
    lines = ['# UD entropy stability at 70%, 80%, 90%, and 100%', '',
             f'{len(wide)} languages; {results[0]["settings"]["repeats"]} paired shuffles per size; '
             f'seed {results[0]["settings"]["seed"]}.', '',
             'Percentages refer to the nonempty sentences in each existing approximately 1,000-sentence UD sample, '
             'not to the full UD treebanks. Samples are nested prefixes of a seeded random permutation, '
             'matching the earlier subset design. Full-sample scores reuse the same initial saved trials. '
             'No larger corpora are used.', '',
             '| Comparison | Pearson r | Spearman rho | Mean change (bits/token) | Median absolute rank change |',
             '|---|---:|---:|---:|---:|']
    for row in main.itertuples():
        lines.append(f'| {row.from_percent}%–{row.to_percent}% | {row.pearson_r:.4f} | {row.spearman_rho:.4f} | '
                     f'{row.mean_change_bits_per_token:+.4f} | {row.median_absolute_rank_change:.1f} |')
    lines += ['', 'Pearson assesses linear correspondence of values; it does not establish numerical agreement. '
              'Spearman assesses ranking consistency. Mean/absolute changes and RMSE are therefore included in '
              '`pairwise_comparisons.csv`. Signed changes use the paper definition H(intact) minus H(within-sentence shuffled).', '',
              'These are descriptive comparisons across languages on overlapping nested samples. They use one '
              'subset trajectory per language, not repeated independent subset selections; high correlations '
              'do not establish asymptotic convergence or stability under other selections. '
              'Delta H = H(intact) minus H(within-sentence shuffled); more negative values indicate a larger entropy increase under shuffling.', '',
              'scores.csv includes actual sentence/token counts and Monte Carlo standard errors. '
              'Separate matrices for intact and within-sentence-shuffled entropies are also provided.']
    (output / 'README.md').write_text('\n'.join(lines) + '\n')

def available(paths, limit):
    from word_order_flexibility.ud import count_single_root_sentences
    return count_single_root_sentences(paths, limit) >= limit

def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)

def run_size(job):
    name, size, args = job
    paths = [args.ud_dir / p for p in LANGUAGES[name]]
    raw = load_conllu(paths, size, single_root_only=True)
    if len(raw) != size:
        raise ValueError(f'{name}: expected {size} sentences, got {len(raw)}')
    corpus = surface_sentences(raw)
    digest = hashlib.sha256(json.dumps(corpus, ensure_ascii=False).encode()).hexdigest()
    output = args.output_dir / str(size) / f'{name}.json'
    expected = dict(language=name, requested_sentences=size, input_sha256=digest,
                    seed=args.seed, repeats=args.repeats, preprocessing=PREPROCESSING, metric_definition=METRIC_DEFINITION)
    if output.exists():
        result = json.loads(output.read_text())
        if any(result.get(k) != v for k, v in expected.items()):
            raise ValueError(f'Cached settings differ: {output}')
        return result
    started = time.time()
    print(f'START {name} {size}', flush=True)
    cached = args.baseline_dir / f'{name}.json'
    if size == 1000 and cached.exists():
        prior = json.loads(cached.read_text())
        if (prior['full'].get('metric_definition') != METRIC_DEFINITION or prior.get('preprocessing') != PREPROCESSING or prior['input_sha256'] != digest or prior['seed'] != args.seed
                or prior['full']['repeats'] != args.repeats):
            raise ValueError(f'Baseline does not match: {cached}')
        full = prior['full']
    else:
        # Match the pre-randomization used by entropy_convergence at 1000.
        corpus = [s for s in corpus if s]
        random.Random(f'subsets:{args.seed}').shuffle(corpus)
        full = sentence_order_entropy(corpus, repeats=args.repeats, seed=args.seed)
    result = {**expected, 'full': full, 'elapsed_seconds': time.time()-started,
              'source_files': [str(p.resolve()) for p in paths],
              'dataset_revision': 'UD 2.17',
              'dit_revision': '42bdafabb94402eb2be00ff2f531e0f930cc0ed7'}
    atomic_json(output, result)
    print(f'DONE {name} {size}: delta_H={full["difference"]["mean"]:.6f}; {result["elapsed_seconds"]:.1f}s', flush=True)
    return result

def write_size_summary(output, results):
    rows = []
    for result in sorted(results, key=lambda r: (r['language'], r['requested_sentences'])):
        full = result['full']
        rows.append(dict(language=result['language'], requested_sentences=result['requested_sentences'],
                         sentences=full['sentences'], tokens=full['tokens'], repeats=full['repeats'],
                         delta_h=full['difference']['mean'], mc_se=full['difference']['mc_se'],
                         mc_95_half_width=full['difference']['mc_95_half_width'],
                         intact=full['intact']['mean'], shuffled=full['within_shuffled']['mean']))
    if rows:
        temporary = output / 'summary.tmp'
        with temporary.open('w') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(output / 'summary.csv')

def run_batch(args):
    with args.languages_csv.open() as handle:
        rows = list(csv.DictReader(handle))
    names = sorted({r['Language' if args.kind == 'ud' else 'code'] for r in rows})
    jobs, missing = [], []
    for name in names:
        if args.kind == 'ud':
            paths = [args.input_dir / treebank for treebank in LANGUAGES.get(name, [])]
            if not paths or not all(p.is_dir() for p in paths):
                paths = []
        else:
            paths = sorted(args.input_dir.glob(f'{name}_*.jsonl'))
            if (args.input_dir / f'{name}.jsonl').exists():
                paths.insert(0, args.input_dir / f'{name}.jsonl')
            if not paths and (args.input_dir / f'{name}.txt').exists():
                paths = [args.input_dir / f'{name}.txt']
        if paths:
            jobs.append((name, paths, args))
        else:
            missing.append(name)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    results = map_jobs(run_entropy_language, jobs, args.workers)
    results.extend({'language': name, 'status': 'missing_input'} for name in missing)
    fields = list(dict.fromkeys(k for row in results for k in row))
    with (Path(args.output_dir) / 'summary.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    print(f'Finished {len(jobs)} languages; {len(missing)} missing inputs', flush=True)

def run_baseline(args):
    (args.output_dir / 'baseline').mkdir(parents=True, exist_ok=True)
    rows = [row for row in map_jobs(calculate_baseline,
            [(name, args) for name in sorted(LANGUAGES)], args.workers) if row is not None]
    if not rows:
        raise ValueError('No languages have enough eligible sentences')
    with (args.output_dir / 'metrics.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'BASELINES COMPLETE: {len(rows)} languages', flush=True)


def run_stability(args):
    (args.output_dir / 'languages').mkdir(parents=True, exist_ok=True)
    names = sorted(pd.read_csv(args.languages_csv).Language.unique())
    results = map_jobs(run_stability_language, [(name, args) for name in names], args.workers)
    analyze(results, args.output_dir)
    print(f'Finished all {len(results)} languages', flush=True)

def run_sizes(args):
    names = sorted(n for n, paths in LANGUAGES.items()
                   if available([args.ud_dir / p for p in paths], 10000))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / 'manifest.json', dict(languages=names, sizes=[1000,2000,5000,10000],
                repeats=args.repeats, seed=args.seed, selection='LANGUAGES with >=10000 sentences, as in experiment.py',
                total_jobs=len(names)*4))
    print(f'{len(names)} languages; {len(names)*4} language/size results', flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_size, (name,size,args)) for size in (1000,2000,5000,10000) for name in names]
        for future in as_completed(futures):
            results.append(future.result())
            write_size_summary(args.output_dir, results)
            atomic_json(args.output_dir / 'progress.json', dict(completed=len(results), total=len(futures),
                        updated_unix=time.time(), status='complete' if len(results)==len(futures) else 'running'))
    print('COMPLETE', flush=True)

def run_paper(args):
    """Run the stages directly; each stage resumes its validated checkpoints."""
    from word_order_flexibility.entropy_plotting import generate_entropy_artifacts

    root = args.output_dir
    shared = dict(ud_dir=args.ud_dir, sentences=args.sentences, repeats=args.repeats,
                  seed=args.seed, workers=args.workers)
    stages = [
        ("baseline", lambda: run_baseline(argparse.Namespace(**shared, output_dir=root))),
        ("entropy", lambda: run_batch(argparse.Namespace(
            **shared, kind="ud", input_dir=args.ud_dir,
            languages_csv=root / "metrics.csv", output_dir=root / "curated"))),
        ("stability", lambda: run_stability(argparse.Namespace(
            **{**shared, "repeats": args.stability_repeats},
            entropy_dir=root / "curated", languages_csv=root / "metrics.csv",
            output_dir=root / "stability"))),
        ("paper_figures", lambda: generate_entropy_artifacts(
            ud_csv=str(root / "metrics.csv"), entropy_dir=str(root / "curated"),
            output_dir=str(root / "paper-figures"))),
    ]
    if args.include_sizes:
        stages.append(("sizes", lambda: run_sizes(argparse.Namespace(
            **shared, baseline_dir=root / "curated", output_dir=root / "sizes"))))
    for name, action in stages:
        status = dict(stage=name, status="running", updated_unix=time.time())
        atomic_json(root / "pipeline-status.json", status)
        print(f"STAGE {name}", flush=True)
        try:
            action()
        except Exception as error:
            atomic_json(root / "pipeline-status.json",
                        {**status, "status": "failed", "error": str(error),
                         "updated_unix": time.time()})
            raise
    atomic_json(root / "pipeline-status.json", dict(
        status="complete", include_sizes=args.include_sizes, updated_unix=time.time()))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="kind", required=True)
    descriptions = {
        "paper": "Run baseline, entropy, stability, and the four paper figures",
        "baseline": "Compute paper-matched UD baseline metrics",
        "ud": "Estimate entropy/convergence for selected UD languages",
        "flores": "Estimate entropy/convergence for local FLORES files",
        "stability": "Compare nested 70/80/90/100 percent UD samples",
        "sizes": "Run the optional 1k/2k/5k/10k UD experiment",
    }
    handlers = dict(paper=run_paper, baseline=run_baseline, ud=run_batch,
                    flores=run_batch, stability=run_stability, sizes=run_sizes)
    for name, description in descriptions.items():
        command = commands.add_parser(name, help=description, description=description)
        command.set_defaults(handler=handlers[name])
        command.add_argument("--workers", type=int, default=4)
        command.add_argument("--seed", type=int, default=20261113)
        command.add_argument("--repeats", type=int, default=20 if name == "stability" else 100)
        command.add_argument("--sentences", type=int, default=None if name == "flores" else 1000)
        if name in ("ud", "flores"):
            command.add_argument("input_dir", type=Path)
            command.add_argument("--languages-csv", type=Path, required=True)
            command.add_argument("--output-dir", type=Path, required=True)
        else:
            command.add_argument("--ud-dir", type=Path, default=Path("ud/ud-treebanks-v2.17"))
            command.add_argument("--output-dir", type=Path,
                                 default=ROOT / name if name in ("stability", "sizes") else ROOT)
        if name == "stability":
            command.add_argument("--entropy-dir", type=Path, default=ROOT / "curated")
            command.add_argument("--languages-csv", type=Path, default=ROOT / "metrics.csv")
        if name == "sizes":
            command.add_argument("--baseline-dir", type=Path, default=ROOT / "curated")
        if name == "paper":
            command.add_argument("--stability-repeats", type=int, default=20)
            command.add_argument("--include-sizes", action="store_true",
                                 help="Also run the separate 1k/2k/5k/10k experiment")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.workers < 1 or args.repeats < 2 or (args.sentences is not None and args.sentences < 4):
        parser.error("Need workers >= 1, repeats >= 2, sentences >= 4")
    if args.kind == "paper" and not 2 <= args.stability_repeats <= args.repeats:
        parser.error("Need 2 <= stability-repeats <= repeats")
    args.handler(args)


if __name__ == "__main__":
    main()

