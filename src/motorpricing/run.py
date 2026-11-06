"""Fit and compare the pricing models, write results/ and figures/.

    python -m motorpricing.run      # downloads freMTPL2 from OpenML on the first run (about 40 MB, cached in data/)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_gamma_deviance, mean_poisson_deviance, mean_tweedie_deviance
from sklearn.model_selection import train_test_split

from . import model as m

ROOT = Path(__file__).resolve().parents[2]
SEED = 7
THRESHOLDS = (0.90, 0.95, 0.975, 0.99)   # large-claim threshold candidates (quantiles of individual claims)
THRESHOLD_Q = 0.95
YEARS = 20_000
CV_SYSTEMATIC = (0.0, 0.05, 0.10)
RETENTIONS = (100_000.0, 250_000.0, 1_000_000.0, np.inf)   # per-claim excess-of-loss retention, illustrative


def d2(dev_model: float, dev_null: float) -> float:
    return 1 - dev_model / dev_null


def main() -> dict:
    out, figs = ROOT / "results", ROOT / "figures"
    out.mkdir(exist_ok=True)
    figs.mkdir(exist_ok=True)
    pol, claims = m.load(ROOT / "data")
    train, test = train_test_split(pol, test_size=0.2, random_state=SEED)
    w_tr, w_te = train["Exposure"].to_numpy(), test["Exposure"].to_numpy()
    f_tr, f_te = (train["ClaimNb"] / train["Exposure"]).to_numpy(), (test["ClaimNb"] / test["Exposure"]).to_numpy()

    # Frequency
    fglm = m.glm("poisson").fit(train, f_tr, poissonregressor__sample_weight=w_tr)
    fgbm = m.gbm("poisson", SEED).fit(m.gbm_features(train), f_tr, sample_weight=w_tr)
    null_f = np.full(len(test), np.average(f_tr, weights=w_tr))
    pf = {"mean": null_f, "glm": fglm.predict(test), "gbm": fgbm.predict(m.gbm_features(test))}
    freq_rows = []
    for k, p in pf.items():
        dev = mean_poisson_deviance(f_te, p, sample_weight=w_te)
        freq_rows.append({"model": k, "poisson_deviance": dev, "d2": d2(dev, mean_poisson_deviance(f_te, null_f, sample_weight=w_te)),
                          "gini": m.lorenz_gini(f_te, p, w_te), "predicted_claims": float((p * w_te).sum()),
                          "observed_claims": float(test["ClaimNb"].sum())})
    freq = pd.DataFrame(freq_rows)
    freq.to_csv(out / "frequency.csv", index=False, float_format="%.5f")

    # Large claims: GPD above a threshold of individual claims of the training policies
    cl_tr = claims[claims["IDpol"].isin(train["IDpol"])]["ClaimAmount"].to_numpy()
    stab = pd.DataFrame([{"quantile": q, **m.gpd_load(cl_tr, float(np.quantile(cl_tr, q)))} for q in THRESHOLDS])
    stab.to_csv(out / "large_claims_threshold.csv", index=False, float_format="%.4f")
    gpd = m.gpd_load(cl_tr, float(np.quantile(cl_tr, THRESHOLD_Q)))
    u = gpd["threshold"]

    # Severity: mean capped claim per policy, weighted by the number of claims
    capped = claims.assign(Capped=claims["ClaimAmount"].clip(upper=u)).groupby("IDpol")["Capped"].sum()
    def sev_frame(df):
        s = df[df["ClaimCount"] > 0].copy()
        s["MeanCapped"] = capped.reindex(s["IDpol"]).to_numpy() / s["ClaimCount"]
        return s
    s_tr, s_te = sev_frame(train), sev_frame(test)
    sw_tr, sw_te = s_tr["ClaimCount"].to_numpy(), s_te["ClaimCount"].to_numpy()
    sglm = m.glm("gamma").fit(s_tr, s_tr["MeanCapped"], gammaregressor__sample_weight=sw_tr)
    sgbm = m.gbm("gamma", SEED).fit(m.gbm_features(s_tr), s_tr["MeanCapped"], sample_weight=sw_tr)
    null_s = np.full(len(s_te), np.average(s_tr["MeanCapped"], weights=sw_tr))
    ps = {"mean": null_s, "glm": sglm.predict(s_te), "gbm": sgbm.predict(m.gbm_features(s_te))}
    sev_rows = []
    for k, p in ps.items():
        dev = mean_gamma_deviance(s_te["MeanCapped"], p, sample_weight=sw_te)
        sev_rows.append({"model": k, "gamma_deviance": dev,
                         "d2": d2(dev, mean_gamma_deviance(s_te["MeanCapped"], null_s, sample_weight=sw_te))})
    sev = pd.DataFrame(sev_rows)
    sev.to_csv(out / "severity.csv", index=False, float_format="%.5f")

    # Pure premium per year of exposure: frequency x (capped severity + large-claim load), and a Tweedie GLM
    load = gpd["load_per_claim_empirical"]
    paid_share = float(train["ClaimCount"].sum() / train["ClaimNb"].sum())   # claims with a recorded amount
    pp_te = (test["ClaimTotal"] / test["Exposure"]).to_numpy()
    pp_tr = (train["ClaimTotal"] / train["Exposure"]).to_numpy()
    tw = m.glm("tweedie").fit(train, pp_tr, tweedieregressor__sample_weight=w_tr)
    pp = {"mean": np.full(len(test), np.average(pp_tr, weights=w_tr)),
          "glm freq x sev": paid_share * pf["glm"] * (sglm.predict(test) + load),
          "gbm freq x sev": paid_share * pf["gbm"] * (sgbm.predict(m.gbm_features(test)) + load),
          "tweedie glm": tw.predict(test)}
    pp_rows = []
    for k, p in pp.items():
        pp_rows.append({"model": k, "gini": m.lorenz_gini(pp_te, p, w_te),
                        "tweedie_deviance_1.9": mean_tweedie_deviance(pp_te, p, sample_weight=w_te, power=1.9),
                        "predicted_total_eur": float((p * w_te).sum()), "observed_total_eur": float(test["ClaimTotal"].sum())})
    prem = pd.DataFrame(pp_rows)
    prem.to_csv(out / "pure_premium.csv", index=False, float_format="%.4f")
    m.calibration_table(f_te, pf["glm"], w_te).to_csv(out / "calibration_frequency_glm.csv", float_format="%.5f")
    m.calibration_table(f_te, pf["gbm"], w_te).to_csv(out / "calibration_frequency_gbm.csv", float_format="%.5f")
    dl = m.double_lift(pp_te, pp["gbm freq x sev"], pp["glm freq x sev"], w_te)
    dl.to_csv(out / "double_lift_gbm_vs_glm.csv", float_format="%.3f")

    # Tariff relativities of the frequency GLM (one-hot against the first level, numeric per standard deviation)
    names = fglm[1].get_feature_names_out()
    rel = pd.DataFrame({"feature": names, "relativity": np.exp(fglm[-1].coef_)})
    rel.to_csv(out / "frequency_relativities.csv", index=False, float_format="%.4f")

    # Premium risk of the test portfolio: 1-in-200 annual retained aggregate against its expectation
    body = cl_tr[cl_tr <= u]
    lam = paid_share * float(pf["glm"] @ w_te)
    cap_rows = []
    for r in RETENTIONS:
        for cv in CV_SYSTEMATIC:
            agg = m.simulate_aggregate(lam, body, gpd, YEARS, cv, seed=SEED, retention=r)
            mean, var995 = agg.mean(), np.quantile(agg, 0.995)
            cap_rows.append({"retention_eur": r, "cv_systematic": cv, "expected_claims": lam, "mean_aggregate_eur": mean,
                             "var995_eur": var995, "capital_over_mean": (var995 - mean) / mean, "cv_aggregate": agg.std() / mean,
                             "retained_large_excess_gpd": m.retained_excess_gpd(gpd, r) if np.isfinite(r) else np.nan,
                             "retained_large_excess_empirical": float(np.clip(np.minimum(cl_tr, r) - u, 0, None).sum()
                                                                      / (cl_tr > u).sum())})
    cap = pd.DataFrame(cap_rows)
    cap.to_csv(out / "premium_risk.csv", index=False, float_format="%.4f")

    # Figures: concentration curves and frequency calibration
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for k in ("glm freq x sev", "gbm freq x sev", "tweedie glm"):
        o = np.argsort(pp[k])
        ax[0].plot(np.cumsum(w_te[o]) / w_te.sum(), np.cumsum((pp_te * w_te)[o]) / (pp_te * w_te).sum(), label=k)
    ax[0].plot([0, 1], [0, 1], color="grey", lw=0.8)
    ax[0].set(title="Concentration of claim amounts (test)", xlabel="exposure, ranked by premium", ylabel="share of claims")
    ax[0].legend(frameon=False)
    for k, c in (("glm", "C0"), ("gbm", "C1")):
        t = m.calibration_table(f_te, pf[k], w_te)
        ax[1].plot(t["predicted_rate"], t["observed_rate"], "o-", color=c, label=k)
    lim = [0, float(t["predicted_rate"].max()) * 1.1]
    ax[1].plot(lim, lim, color="grey", lw=0.8)
    ax[1].set(title="Frequency by decile of prediction (test)", xlabel="predicted claims per year", ylabel="observed")
    ax[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figs / "lorenz_calibration.png", dpi=130)

    summary = {"policies": len(pol), "claims_in_severity_file": len(claims), "train": len(train), "test": len(test),
               "policies_with_count_but_no_amount": int(((pol["ClaimNb"] > 0) & (pol["ClaimCount"] == 0)).sum()),
               "gbm_iterations": {"frequency": int(fgbm.n_iter_), "severity": int(sgbm.n_iter_)},
               "large_claims": gpd, "paid_share_of_claim_counts": paid_share, "frequency": freq.to_dict("records"), "severity": sev.to_dict("records"),
               "pure_premium": prem.to_dict("records"), "premium_risk": cap.to_dict("records")}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for t in (freq, sev, prem, stab, cap):
        print(t.round(4).to_string(), "\n")
    print(dl.round(3).to_string())
    print(json.dumps({k: summary[k] for k in ("policies", "claims_in_severity_file", "policies_with_count_but_no_amount", "gbm_iterations")}))
    return summary


if __name__ == "__main__":
    main()
