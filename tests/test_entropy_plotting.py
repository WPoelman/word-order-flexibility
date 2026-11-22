import json

import pandas as pd
import pytest

from word_order_flexibility.entropy_plotting import load_entropy_ud
from word_order_flexibility.cli import build_parser


def fixture_inputs(tmp_path):
    rows = []
    folder = tmp_path / "entropy"
    folder.mkdir()
    for index, name in enumerate(["Alpha", "Beta", "Gamma"]):
        rows.append(
            dict(
                Language=name,
                word_tokens=20 + index,
                HDE=index,
                ROE=index / 2,
                type_token_ratio=0.1 + index / 10,
                F_lambda_025=0.2,
                F_lambda_050=0.3,
                F_lambda_075=0.4,
                F_lambda_100=0.5,
                qualifying_pairs_lambda_050=10 + index,
            )
        )
        score = {
            "sentences": 4,
            "tokens": 20 + index,
            "difference": {"mean": 0.2 + index, "mc_se": 0.01},
            "intact": {"mean": 3},
            "within_shuffled": {"mean": 3.2 + index},
        }
        result = {
            "language": name,
            "corpus": "ud",
            "full": score,
            "input_sha256": "test",
            "diagnostics": {"difference": {"passes_half_full": False}},
            "size_convergence": [score],
        }
        (folder / f"{name}.json").write_text(json.dumps(result))
    path = tmp_path / "ud.csv"
    pd.DataFrame(rows[::-1]).to_csv(path, index=False)
    return path, folder


def test_join_is_by_language_and_preserves_sign(tmp_path):
    path, folder = fixture_inputs(tmp_path)
    joined, sizes, meta = load_entropy_ud(path, folder)
    assert joined.set_index("Language").loc["Alpha", "entropy_difference"] == 0.2
    assert len(joined) == len(sizes) == 3
    assert meta["missing_entropy"] == []
    assert meta["half_full_pass_count"] == 0


def test_mismatched_samples_rejected(tmp_path):
    path, folder = fixture_inputs(tmp_path)
    data = pd.read_csv(path)
    data.loc[0, "word_tokens"] = 999
    data.to_csv(path, index=False)
    with pytest.raises(ValueError, match="token counts differ"):
        load_entropy_ud(path, folder)


def test_duplicate_languages_rejected(tmp_path):
    path, folder = fixture_inputs(tmp_path)
    data = pd.read_csv(path)
    pd.concat([data, data.iloc[:1]]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="one row per language"):
        load_entropy_ud(path, folder)


def test_entropy_paper_cli():
    args = build_parser().parse_args(["entropy-paper", "ud.csv", "entropy", "--tex"])
    assert args.ud_csv == "ud.csv"
    assert args.tex


def test_entropy_paper_defaults():
    args = build_parser().parse_args(["entropy-paper"])
    assert args.ud_csv.endswith("/ud-paper-aligned/metrics.csv")
    assert args.entropy_dir.endswith("/ud-paper-aligned/curated")
    assert args.output_dir.endswith("/paper-figures")
    assert args.stability_csv is None


def test_entropy_artifacts_include_full_table(tmp_path, monkeypatch):
    from word_order_flexibility import entropy_plotting as plotting

    path, folder = fixture_inputs(tmp_path)
    monkeypatch.setattr(plotting, "FUTRELL_LANGUAGE_NAMES", ["Alpha", "Beta", "Gamma"])
    for name in ("_plot_hde", "bakker_figures", "_plot_stability_values", "_plot_stability_correlations"):
        monkeypatch.setattr(plotting, name, lambda *args, **kwargs: None)
    stability = tmp_path / "stability.csv"
    pd.DataFrame({
        "language": ["Alpha", "Beta", "Gamma"],
        **{size: [-0.1, -0.2, -0.3] for size in ("70", "80", "90", "100")},
    }).to_csv(stability, index=False)
    bakker = tmp_path / "bakker.csv"
    pd.DataFrame({"Language": ["Alpha"], "Flexibility": [0.5]}).to_csv(bakker, index=False)
    output = tmp_path / "output"
    plotting.generate_entropy_artifacts(path, folder, bakker, output, stability_csv=stability)
    table = output / "appendix-language-metrics-table.tex"
    assert r"$\Delta H$" in table.read_text()
    assert all(name in table.read_text() for name in ("Alpha", "Beta", "Gamma"))
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["table_files"] == [table.name]


def test_stability_requires_same_languages(tmp_path):
    from word_order_flexibility.entropy_plotting import _load_stability

    path = tmp_path / "scores.csv"
    pd.DataFrame({"language": ["A"], "70": [-0.2], "80": [-0.3], "90": [-0.4], "100": [-0.5]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="same languages"):
        _load_stability(path, ["A", "B"])


def test_negative_stability_scores_remain_visible(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    from word_order_flexibility import entropy_plotting as plotting

    wide = pd.DataFrame(
        [[-0.2, -0.3, -0.4, -0.5], [-2.0, -2.3, -2.5, -3.0]], columns=["70", "80", "90", "100"], index=["A", "B"]
    )
    captured = []
    monkeypatch.setattr(plotting, "_save", lambda fig, path: captured.append(fig))
    with plt.rc_context({"text.usetex": False}):
        plotting._plot_stability_values(wide, tmp_path)
    left, right = captured[0].axes
    low, high = left.get_ylim()
    assert low <= wide.to_numpy().min() and high >= wide.to_numpy().max()
    assert right.get_ylim()[0] == 0
    change = pd.read_csv(tmp_path / "entropy_absolute_changes_from_70.csv", index_col=0)
    assert change["70"].eq(0).all()
    assert change.loc["B", "100"] == pytest.approx(1.0)
    plt.close(captured[0])
