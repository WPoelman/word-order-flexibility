import numpy as np
import pandas as pd
import pytest

from run_entropy import analyze, retained_trials


def test_retained_trials_uses_requested_prefix():
    trials = [{'intact': float(i), 'within_shuffled': 2.0 * i, 'difference': float(i)} for i in range(10)]
    full = {'sentences': 4, 'tokens': 20, 'seed': 7, 'trials': trials}
    result = retained_trials(full, 3)
    assert result['difference']['mean'] == 1.0
    assert len(result['trials']) == 3
    assert len(full['trials']) == 10
    with pytest.raises(ValueError, match='exceed'):
        retained_trials(full, 11)


def test_analysis_separates_correlation_from_value_drift(tmp_path):
    results = []
    for i, name in enumerate(['A', 'B', 'C', 'D']):
        sizes = {}
        for percentage in [70, 80, 90, 100]:
            value = i + percentage / 100
            sizes[str(percentage)] = {'sentences': percentage, 'tokens': percentage * 10, 'repeats': 2,
                                      **{key: {'mean': value, 'mc_se': .001}
                                         for key in ['intact', 'within_shuffled', 'difference']}}
        results.append({'language': name, 'sizes': sizes, 'settings': {'seed': 7, 'repeats': 2}})
    analyze(results, tmp_path)
    pairs = pd.read_csv(tmp_path / 'pairwise_comparisons.csv')
    assert len(pairs) == 18
    assert np.allclose(pairs.pearson_r, 1)
    assert np.allclose(pairs.spearman_rho, 1)
    pair = pairs.query("metric == 'difference' and from_percent == 70 and to_percent == 100").iloc[0]
    assert pair.mean_change_bits_per_token == pytest.approx(.3)
    assert pair.mean_absolute_change == pytest.approx(.3)
    assert pair.median_absolute_rank_change == 0
