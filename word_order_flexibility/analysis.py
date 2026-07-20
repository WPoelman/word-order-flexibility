import numpy as np
from scipy import stats


def residualize(values, control) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    control = np.asarray(control, dtype=float)
    design = np.column_stack((np.ones(len(control)), control))
    return values - design @ np.linalg.lstsq(design, values, rcond=None)[0]


def correlations(rows: list[dict], target: str, *, control_ttr: bool = False) -> dict:
    """Return Pearson r and Spearman rho for flexibility and a target metric."""
    valid = [row for row in rows if all(row.get(key) not in (None, "") for key in ("flexibility", target, "ttr"))]
    x = np.asarray([float(row["flexibility"]) for row in valid])
    y = np.asarray([float(row[target]) for row in valid])
    if control_ttr:
        ttr = np.asarray([float(row["ttr"]) for row in valid])
        x, y = residualize(x, ttr), residualize(y, ttr)
    if len(valid) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return {
            "target": target,
            "control": "ttr" if control_ttr else "none",
            "n": len(valid),
            "pearson_r": None,
            "pearson_p": None,
            "spearman_rho": None,
            "spearman_p": None,
        }
    pearson = stats.pearsonr(x, y)
    spearman = stats.spearmanr(x, y)
    return {
        "target": target,
        "control": "ttr" if control_ttr else "none",
        "n": len(valid),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }
