"""Conjugate Bayesian linear regression and severity tiers (paper Sec. 2.3).

z-scored Lambda is regressed on z-scored GSS with a normal-inverse-gamma prior.
Everything is closed form; no sampling is involved.
"""
import numpy as np
from scipy import stats

PRIOR = {"beta0": [0.0, 0.0], "V0_diag": [4.0, 4.0], "a0": 2.0, "b0": 1.0}
N_TIERS = 4


class Scaler:
    """z-scoring with moments from the calibration set only."""

    def __init__(self, v):
        self.mu = float(np.mean(v))
        self.sd = float(np.std(v, ddof=1))

    def fwd(self, v):
        return (np.asarray(v, float) - self.mu) / self.sd

    def inv(self, z):
        return np.asarray(z, float) * self.sd + self.mu


def design(x):
    """Design matrix [1, x]."""
    return np.column_stack([np.ones(len(x)), x])


def nig_posterior(X, y, prior=PRIOR):
    """Posterior (beta, V, a, b) of beta | s2 ~ N(beta0, s2 V0), s2 ~ IG(a0, b0)."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    b0 = np.asarray(prior["beta0"], float)
    V0inv = np.diag(1.0 / np.asarray(prior["V0_diag"], float))
    Vn_inv = V0inv + X.T @ X
    Vn = np.linalg.inv(Vn_inv)
    bn = Vn @ (V0inv @ b0 + X.T @ y)
    an = prior["a0"] + len(y) / 2.0
    bn_scale = prior["b0"] + 0.5 * (y @ y + b0 @ V0inv @ b0 - bn @ Vn_inv @ bn)
    return {"beta": bn, "V": Vn, "a": an, "b": float(bn_scale), "n": len(y)}


def predictive(post, X):
    """Posterior predictive for new rows of X: Student-t (loc, scale, dof)."""
    X = np.atleast_2d(np.asarray(X, float))
    loc = X @ post["beta"]
    quad = np.einsum("ij,jk,ik->i", X, post["V"], X)
    return loc, np.sqrt((post["b"] / post["a"]) * (1.0 + quad)), 2.0 * post["a"]


def tier_edges(values, n_tiers=N_TIERS):
    """Interior quantile cutpoints of the calibration targets."""
    return np.percentile(np.asarray(values, float),
                         np.linspace(0.0, 100.0, n_tiers + 1)[1:-1])


def assign_tier(values, edges):
    """0-based tier; 0 is least severe."""
    return np.digitize(np.asarray(values, float), edges)


def tier_probabilities(loc, scale, df, edges_z):
    """(n, n_tiers) predictive tier probabilities from Student-t CDF differences."""
    F = stats.t.cdf(np.asarray(edges_z)[:, None], df=df, loc=loc, scale=scale)
    lo = np.vstack([np.zeros((1, len(loc))), F])
    hi = np.vstack([F, np.ones((1, len(loc)))])
    return (hi - lo).T


def median_tier(P):
    """Posterior median tier: the first tier whose cumulative probability reaches 1/2."""
    return (np.cumsum(P, axis=1) >= 0.5).argmax(axis=1)


def confusion(pred, true, n_tiers=N_TIERS):
    """Confusion matrix (rows predicted, columns true) and summary scores.

    Chance accuracy uses the matched marginals; kappa is quadratic-weighted.
    """
    pred, true = np.asarray(pred, int), np.asarray(true, int)
    n = len(pred)
    C = np.zeros((n_tiers, n_tiers), int)
    for p, t in zip(pred, true):
        C[p, t] += 1
    mp, mt = C.sum(axis=1) / n, C.sum(axis=0) / n
    idx = np.arange(n_tiers)
    d = np.abs(idx[:, None] - idx[None, :])
    W = (d / (n_tiers - 1)) ** 2
    expected = float((W * np.outer(mp, mt)).sum())
    chance = float(mp @ mt)
    return {
        "n": int(n),
        "matrix_pred_by_true": C.tolist(),
        "row_normalized": (C / np.maximum(C.sum(axis=1, keepdims=True), 1)).tolist(),
        "accuracy": float((pred == true).mean()),
        "chance_matched_marginal": chance,
        "adjacent_accuracy": float((np.abs(pred - true) <= 1).mean()),
        "kappa_quadratic": float(1.0 - (W * C).sum() / n / expected),
    }


def scores(pred, truth):
    """r2 (1 - SSE/SST), Pearson r2, RMSE and Spearman rho."""
    pred, truth = np.asarray(pred, float), np.asarray(truth, float)
    sse = float(((truth - pred) ** 2).sum())
    return {
        "r2": 1.0 - sse / float(((truth - truth.mean()) ** 2).sum()),
        "pearson_r2": float(stats.pearsonr(pred, truth)[0] ** 2),
        "rmse": float(np.sqrt(sse / len(truth))),
        "spearman": float(stats.spearmanr(pred, truth)[0]),
    }


def response_scale_posterior(post, xs, ys):
    """Joint posterior of (alpha, beta, sigma) on the unscaled Lambda-vs-GSS line.

    The standardized fit maps linearly onto y = alpha + beta x, so the
    multivariate-t posterior carries over exactly.
    """
    A = np.array([[ys.sd, -ys.sd * xs.mu / xs.sd], [0.0, ys.sd / xs.sd]])
    loc = A @ post["beta"] + np.array([ys.mu, 0.0])
    shape = (post["b"] / post["a"]) * (A @ post["V"] @ A.T)
    scale = np.sqrt(np.diag(shape))
    return {
        "loc": loc.tolist(),
        "shape": shape.tolist(),
        "df": float(2.0 * post["a"]),
        "a": float(post["a"]),
        "b_sigma": float(ys.sd ** 2 * post["b"]),
        "scale_alpha": float(scale[0]),
        "scale_beta": float(scale[1]),
        "corr_alpha_beta": float(shape[0, 1] / (scale[0] * scale[1])),
    }


def fit(x_cal, y_cal, x_val, y_val, n_tiers=N_TIERS):
    """Fit on calibration, predict calibration and validation, and tier both.

    Returns a JSON-serializable dict: posterior, scores, tier edges, tier
    probabilities, median-tier calls and confusion matrices per split, and a
    300-point predictive curve with its 95% band.
    """
    xs, ys = Scaler(x_cal), Scaler(y_cal)
    post = nig_posterior(design(xs.fwd(x_cal)), ys.fwd(y_cal))
    edges = tier_edges(y_cal, n_tiers)
    out = {"n_cal": len(x_cal), "n_val": len(x_val),
           "tier_edges": edges.tolist(),
           "posterior": response_scale_posterior(post, xs, ys),
           "standardized": {"beta": post["beta"].tolist(), "V": post["V"].tolist(),
                            "a": post["a"], "b": post["b"]},
           "scaler": {"x": [xs.mu, xs.sd], "y": [ys.mu, ys.sd]}}

    for name, x, y in (("calibration", x_cal, y_cal), ("validation", x_val, y_val)):
        loc, scale, df = predictive(post, design(xs.fwd(x)))
        P = tier_probabilities(loc, scale, df, ys.fwd(edges))
        true, pred = assign_tier(y, edges), median_tier(P)
        out[name] = {
            "scores": scores(ys.inv(loc), y),
            "prediction": ys.inv(loc).tolist(),
            "true_tier": true.tolist(),
            "predicted_tier": pred.tolist(),
            "tier_probabilities": P.tolist(),
            "confusion": confusion(pred, true, n_tiers),
        }

    x_all = np.concatenate([x_cal, x_val])
    grid = np.linspace(float(x_all.min()), float(x_all.max()), 300)
    loc, scale, df = predictive(post, design(xs.fwd(grid)))
    half = stats.t.ppf(0.975, df=df)
    out["curve"] = {"x": grid.tolist(), "mean": ys.inv(loc).tolist(),
                    "lo": ys.inv(loc - half * scale).tolist(),
                    "hi": ys.inv(loc + half * scale).tolist()}
    return out
