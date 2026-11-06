"""Data, features, models and metrics for claim frequency, severity and pure premium.

Frequency: claims per year of exposure, Poisson deviance, exposure as weight. Severity: mean claim per policy,
gamma deviance, number of claims as weight, claims capped at a large-claim threshold. Large claims: the excess over
the threshold follows a generalised Pareto distribution and is spread over all claims as a load. Pure premium:
frequency of claims with a recorded amount times (capped severity plus the observed mean excess per claim), compared with a Tweedie GLM fitted to the pure premium directly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import genpareto
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import GammaRegressor, PoissonRegressor, TweedieRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

FREQ_ID, SEV_ID = 41214, 41215          # OpenML: freMTPL2freq, freMTPL2sev
CATEGORICAL = ["Area", "VehBrand", "VehGas", "Region"]
NUMERIC = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]
DRIV_BINS = [17, 20, 25, 30, 40, 50, 60, 70, 120]
VEH_BINS = [-1, 0, 1, 5, 10, 15, 120]


def load(cache: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Policies with capped claim counts and exposure, and the individual claims of those policies."""
    freq = fetch_openml(data_id=FREQ_ID, as_frame=True, data_home=str(cache), parser="auto").frame
    sev = fetch_openml(data_id=SEV_ID, as_frame=True, data_home=str(cache), parser="auto").frame
    freq["IDpol"] = freq["IDpol"].astype(int)
    freq["VehGas"] = freq["VehGas"].astype(str).str.strip("'")
    for c in CATEGORICAL:
        freq[c] = freq[c].astype(str).str.strip("'").astype("category")
    freq["ClaimNb"] = freq["ClaimNb"].clip(upper=4)
    freq["Exposure"] = freq["Exposure"].clip(upper=1.0)
    sev = sev[sev["IDpol"].isin(freq["IDpol"])].reset_index(drop=True)
    agg = sev.groupby("IDpol")["ClaimAmount"].agg(ClaimTotal="sum", ClaimCount="size")
    freq = freq.merge(agg, left_on="IDpol", right_index=True, how="left")
    freq[["ClaimTotal", "ClaimCount"]] = freq[["ClaimTotal", "ClaimCount"]].fillna(0)
    return freq, sev


def bin_features(x: pd.DataFrame) -> pd.DataFrame:
    """Actuarial tariff cells: age bands, vehicle age bands, capped power and bonus-malus, log density."""
    out = pd.DataFrame(index=x.index)
    out["DrivAgeBand"] = pd.cut(x["DrivAge"], DRIV_BINS).astype(str)
    out["VehAgeBand"] = pd.cut(x["VehAge"], VEH_BINS).astype(str)
    out["VehPowerCap"] = x["VehPower"].clip(upper=9).astype(str)
    bm = x["BonusMalus"].clip(upper=150).astype(float)
    out["BonusMalus"] = bm
    out["LogBonusMalus"] = np.log(bm)
    out["LogDensity"] = np.log(x["Density"].astype(float))
    for c in CATEGORICAL:
        out[c] = x[c].astype(str)
    return out


def glm_design() -> ColumnTransformer:
    cats = ["DrivAgeBand", "VehAgeBand", "VehPowerCap"] + CATEGORICAL
    return ColumnTransformer([("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), cats),
                              ("num", StandardScaler(), ["BonusMalus", "LogBonusMalus", "LogDensity"])])


def glm(kind: str, alpha: float = 1e-4, power: float = 1.9):
    reg = {"poisson": PoissonRegressor(alpha=alpha, solver="newton-cholesky", max_iter=300),
           "gamma": GammaRegressor(alpha=alpha, solver="newton-cholesky", max_iter=300),
           "tweedie": TweedieRegressor(power=power, link="log", alpha=alpha, solver="newton-cholesky", max_iter=300)}[kind]
    return make_pipeline(FunctionTransformer(bin_features), glm_design(), reg)


def gbm(loss: str, seed: int = 0, max_iter: int = 500) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(loss=loss, learning_rate=0.05, max_iter=max_iter, max_leaf_nodes=31,
                                         min_samples_leaf=200, l2_regularization=1.0, early_stopping=True,
                                         validation_fraction=0.15, n_iter_no_change=30, categorical_features="from_dtype",
                                         random_state=seed)


def gbm_features(x: pd.DataFrame) -> pd.DataFrame:
    return x[NUMERIC + CATEGORICAL]


def gpd_load(claims: np.ndarray, threshold: float) -> dict:
    """GPD on excesses over the threshold and the expected excess per claim, the large-claim load."""
    exc = claims[claims > threshold] - threshold
    xi, _, sigma = genpareto.fit(exc, floc=0)
    p_u = len(exc) / len(claims)
    mean_excess = sigma / (1 - xi) if xi < 1 else np.inf
    return {"threshold": float(threshold), "n_above": int(len(exc)), "p_above": float(p_u), "xi": float(xi),
            "sigma": float(sigma), "mean_excess_gpd": float(mean_excess), "mean_excess_empirical": float(exc.mean()),
            "load_per_claim_gpd": float(p_u * mean_excess), "load_per_claim_empirical": float(exc.sum() / len(claims))}


def retained_excess_gpd(gpd: dict, retention: float) -> float:
    """Expected retained excess per large claim, E[min(Y, R - u)] = integral of the GPD survival from 0 to R - u."""
    top = retention - gpd["threshold"]
    if top <= 0:
        return 0.0
    y = np.linspace(0, top, 20_001)
    return float(np.trapezoid(genpareto.sf(y, gpd["xi"], scale=gpd["sigma"]), y))


def lorenz_gini(y_true: np.ndarray, y_pred: np.ndarray, weight: np.ndarray) -> float:
    """Gini of the concentration curve: policies ranked from low to high predicted risk, weighted by exposure."""
    order = np.argsort(y_pred, kind="stable")
    w, y = weight[order], (y_true * weight)[order]
    cum_w = np.concatenate([[0], np.cumsum(w) / w.sum()])
    cum_y = np.concatenate([[0], np.cumsum(y) / y.sum()])
    return float(1 - 2 * np.trapezoid(cum_y, cum_w))


def calibration_table(y_true: np.ndarray, y_pred: np.ndarray, weight: np.ndarray, groups: int = 10) -> pd.DataFrame:
    """Exposure-weighted deciles of the prediction: observed against predicted totals."""
    order = np.argsort(y_pred, kind="stable")
    cw = np.cumsum(weight[order]) / weight.sum()
    g = np.minimum((cw * groups).astype(int), groups - 1)
    d = pd.DataFrame({"g": g, "w": weight[order], "obs": (y_true * weight)[order], "pred": (y_pred * weight)[order]})
    t = d.groupby("g").sum()
    t["observed_rate"], t["predicted_rate"] = t["obs"] / t["w"], t["pred"] / t["w"]
    return t[["w", "observed_rate", "predicted_rate"]].rename(columns={"w": "exposure"})


def double_lift(y_true: np.ndarray, a: np.ndarray, b: np.ndarray, weight: np.ndarray, groups: int = 10) -> pd.DataFrame:
    """Policies sorted by the ratio of model a to model b; per decile, observed and predicted rates relative to their
    portfolio means, so that a difference in overall level does not count: which model follows the observed better."""
    ratio = a / b
    order = np.argsort(ratio, kind="stable")
    cw = np.cumsum(weight[order]) / weight.sum()
    g = np.minimum((cw * groups).astype(int), groups - 1)
    d = pd.DataFrame({"g": g, "w": weight[order], "obs": (y_true * weight)[order],
                      "a": (a * weight)[order], "b": (b * weight)[order]}).groupby("g").sum()
    for c in ("obs", "a", "b"):
        d[c] = d[c] / d["w"] / (d[c].sum() / d["w"].sum())    # relative to each column's own portfolio mean
    return d.rename(columns={"w": "exposure", "obs": "observed", "a": "model_a", "b": "model_b"})


def simulate_aggregate(expected_claims: float, body: np.ndarray, gpd: dict, years: int = 20_000, cv_systematic: float = 0.0,
                       seed: int = 0, chunk: int = 500, retention: float = np.inf) -> np.ndarray:
    """Annual aggregate retained claims of a portfolio: Poisson (or gamma-mixed Poisson) count, severities from the
    body below the threshold with probability 1 - p_above and threshold plus a GPD excess otherwise, each claim
    capped at the per-claim excess-of-loss retention."""
    rng = np.random.default_rng(seed)
    out = np.empty(years)
    for start in range(0, years, chunk):
        m = min(chunk, years - start)
        lam = np.full(m, expected_claims)
        if cv_systematic > 0:
            k = 1 / cv_systematic ** 2
            lam = lam * rng.gamma(k, 1 / k, m)
        n = rng.poisson(lam)
        tot = n.sum()
        large = rng.random(tot) < gpd["p_above"]
        sev = rng.choice(body, tot)
        sev[large] = gpd["threshold"] + genpareto.rvs(gpd["xi"], scale=gpd["sigma"], size=int(large.sum()), random_state=rng)
        sev = np.minimum(sev, retention)
        idx = np.repeat(np.arange(m), n)
        agg = np.zeros(m)
        np.add.at(agg, idx, sev)
        out[start:start + m] = agg
    return out
