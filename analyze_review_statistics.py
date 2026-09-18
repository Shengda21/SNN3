"""Derive manuscript revision contrasts from the retained R2.1 observations.

This is an offline reanalysis, not model execution or a new training experiment.
Intervals use independent adaptation seeds, conditional on the fixed starting
checkpoint, test examples, selected procedures and calibration subset.
"""
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
from scipy.stats import t

ROOT = Path(__file__).resolve().parent
R2 = ROOT / "revision_round2"
OUT = ROOT / "review_analysis"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def interval(values):
    x = np.asarray(values, dtype=float)
    n = len(x)
    mean = float(x.mean())
    sd = float(x.std(ddof=1))
    half = float(t.ppf(.975, n - 1) * sd / math.sqrt(n))
    return dict(n=n, mean=mean, sd=sd, low=mean-half, high=mean+half,
                positive=int((x > 0).sum()), negative=int((x < 0).sum()))


def main():
    OUT.mkdir(exist_ok=True)
    rows = []
    lock = read(R2 / "results/model_analysis_lock.json")
    for row in lock["scoring"]:
        folder = R2 / "results/scoring" / row["id"]
        meta = read(folder / "complete.json")
        with np.load(folder / "test.npz", allow_pickle=False) as z:
            counts = z["counts"]
            ce = float(z["loss_sum"].sum(dtype=np.float64) / counts.sum())
            acc = float(z["correct"].sum() / counts.sum())
            assert int(z["correct"].sum()) == int((z["pred"] == z["labels"]).sum())
            assert len(np.unique(z["block_id"])) == len(z["block_id"])
            assert np.isfinite(z["loss_sum"]).all()
        assert np.allclose([ce, acc], [meta["ce"], meta["accuracy"]], atol=1e-12, rtol=0)
        rows.append({**row, "ce": ce, "accuracy": acc})
    scores = pd.DataFrame(rows)
    scores.to_csv(OUT / "quality_scores.csv", index=False)
    raw = scores[scores.engine == "eager"]
    gains, gain_summary, accuracy, interactions = [], [], [], []
    for (task, cal), group in raw.groupby(["task", "calibration"]):
        zeros = group[group.state == "shared_initial"].set_index("infer_T")
        adapted = group[group.state == "adapted"]
        for (K, a, e), g in adapted.groupby(["updates", "adapt_T", "infer_T"]):
            values = float(zeros.loc[e, "ce"]) - g.sort_values("seed").ce
            gain_summary.append(dict(task=task, calibration=cal, updates=int(K),
                                     adapt_T=int(a), infer_T=int(e), **interval(values)))
            for (_, row), value in zip(g.sort_values("seed").iterrows(), values):
                gains.append(dict(task=task, calibration=cal, updates=int(K),
                                  seed=int(row.seed), adapt_T=int(a), infer_T=int(e),
                                  initial_ce=float(zeros.loc[e, "ce"]), ce=float(row.ce), gain=float(value)))
        for K, g in adapted[adapted.infer_T == 4].groupby("updates"):
            pivot = g.pivot(index="seed", columns="adapt_T", values="accuracy")
            d = 100 * (pivot[4] - pivot[2])
            for seed, value in d.items():
                accuracy.append(dict(task=task, calibration=cal, updates=int(K),
                                     seed=int(seed), accuracy_s4_minus_s2_pp=float(value)))
    for seed, g in raw[(raw.task == "vision") & (raw.state == "adapted") & (raw.infer_T == 4)].groupby("seed"):
        p = g.pivot(index=["calibration", "updates"], columns="adapt_T", values="ce")
        contrasts = p[2] - p[4]
        raw_change = contrasts.loc["raw", 2048] - contrasts.loc["raw", 512]
        bn_change = contrasts.loc["bn_only", 2048] - contrasts.loc["bn_only", 512]
        interactions.append(dict(seed=int(seed), raw_budget_change=float(raw_change),
                                 bn_budget_change=float(bn_change), J=float(bn_change-raw_change)))
    pd.DataFrame(gains).to_csv(OUT / "initialization_gains_by_seed.csv", index=False)
    gs = pd.DataFrame(gain_summary)
    gs.to_csv(OUT / "initialization_gain_intervals.csv", index=False)
    ac = pd.DataFrame(accuracy)
    ac.to_csv(OUT / "accuracy_contrasts_by_seed.csv", index=False)
    ac_summary = [dict(task=task, calibration=cal, updates=int(K),
                       **interval(g.accuracy_s4_minus_s2_pp))
                  for (task, cal, K), g in ac.groupby(["task", "calibration", "updates"])]
    pd.DataFrame(ac_summary).to_csv(OUT / "accuracy_contrast_intervals.csv", index=False)
    pd.DataFrame(interactions).to_csv(OUT / "normalization_budget_interaction_by_seed.csv", index=False)

    screen = []
    for path in sorted((R2 / "results/screen").glob("*/complete.json")):
        obj = read(path)
        cfg = obj["config"]
        ev = next(x["validation"] for x in obj["evaluations"] if x["step"] == 512)
        screen.append(dict(T=cfg["T"], lr=cfg["lr"], seed=cfg["seed"], updates=512,
                           validation_ce=ev["ce"], validation_accuracy=ev["accuracy"],
                           training_seconds=obj["training_seconds"],
                           peak_allocated_gib=obj["peak_allocated_gib"]))
    sf = pd.DataFrame(screen).sort_values(["T", "lr", "seed"])
    sf.to_csv(OUT / "vision_screen.csv", index=False)
    selection = read(R2 / "results/selection.json")
    for row in selection["candidates"]:
        g = sf[sf["T"] == row["T"]]
        g = g[g.lr == row["lr"]]
        assert np.isclose(g.validation_ce.mean(), row["mean_validation_ce"], atol=1e-12, rtol=0)
    formal = [read(x) for x in sorted((R2 / "results/formal").glob("*/complete.json"))]
    resource = {str(T): dict(
        seconds_per_update=interval([x["training_seconds"] / x["step"] for x in formal if x["config"]["T"] == T]),
        peak_allocated_gib=max(x["peak_allocated_gib"] for x in formal if x["config"]["T"] == T)) for T in [2, 4]}
    summary = dict(
        source="retained R2.1 per-item arrays and screening records",
        scoring_paths_recomputed=len(rows),
        interval_scope="pointwise Student intervals over adaptation seeds; fixed initialization, selected procedure, inputs and calibration subset",
        normalization_budget_interaction=interval([x["J"] for x in interactions]),
        vision_final_bn_accuracy_s4_minus_s2_pp=next(x for x in ac_summary if x["task"] == "vision" and x["calibration"] == "bn_only" and x["updates"] == 2048),
        formal_training_resources=resource,
        budget_accounting=dict(language_screen_updates=2*4*2*2048,
                               language_formal_updates=2*3*4096,
                               vision_screen_updates=2*3*2*512,
                               vision_formal_updates=2*5*2048,
                               vision_bn_calibrations=42,
                               vision_bn_image_presentations=42*2048,
                               vision_bn_forward_batches=42*32))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
