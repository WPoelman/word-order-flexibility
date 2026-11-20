import json

import pytest

import run_entropy as runner


def test_paper_propagates_paths_and_settings(tmp_path, monkeypatch):
    import word_order_flexibility.entropy_plotting as plotting

    calls = []
    for name in ("baseline", "batch", "stability", "sizes"):
        monkeypatch.setattr(runner, f"run_{name}",
                            lambda args, name=name: calls.append((name, vars(args))))
    monkeypatch.setattr(plotting, "generate_entropy_artifacts",
                        lambda **kwargs: calls.append(("figures", kwargs)))
    runner.main(["paper", "--output-dir", str(tmp_path), "--ud-dir", "custom-ud",
                 "--repeats", "8", "--stability-repeats", "3", "--workers", "1"])
    assert [name for name, _ in calls] == ["baseline", "batch", "stability", "figures"]
    assert calls[1][1]["languages_csv"] == tmp_path / "metrics.csv"
    assert calls[1][1]["output_dir"] == tmp_path / "curated"
    assert calls[1][1]["repeats"] == 8
    assert calls[2][1]["entropy_dir"] == tmp_path / "curated"
    assert calls[2][1]["repeats"] == 3
    assert calls[3][1]["output_dir"] == str(tmp_path / "paper-figures")
    assert json.loads((tmp_path / "pipeline-status.json").read_text())["status"] == "complete"


def test_paper_records_failure_and_stops(tmp_path, monkeypatch):
    def fail(args):
        raise ValueError("baseline mismatch")

    monkeypatch.setattr(runner, "run_baseline", fail)
    with pytest.raises(ValueError, match="baseline mismatch"):
        runner.main(["paper", "--output-dir", str(tmp_path)])
    status = json.loads((tmp_path / "pipeline-status.json").read_text())
    assert status["stage"] == "baseline"
    assert status["status"] == "failed"


def test_paper_rejects_more_stability_trials_than_saved():
    with pytest.raises(SystemExit):
        runner.main(["paper", "--repeats", "5", "--stability-repeats", "6"])
