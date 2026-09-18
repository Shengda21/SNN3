"""Recompute the reported evidence offline; no GPU, corpus, or model weights needed."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd


def long_path(path):
    path = Path(path).resolve()
    if os.name == "nt" and not str(path).startswith("\\\\?\\"):
        path = Path("\\\\?\\" + str(path))
    return path


ROOT = long_path(Path(__file__).parent)
LANG = ROOT / "experiment/orthogonal_pilot_20260907"
RAW = LANG / "results/paper_experiments_20260908"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def close(a, b, label):
    if not np.allclose(a, b, rtol=1e-11, atol=1e-11, equal_nan=True):
        raise AssertionError(label)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "recomputed")
    parser.add_argument("--figures", action="store_true", help="Also redraw the seven numerical figures and SVG workflow")
    args = parser.parse_args()
    out = long_path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    data = out / "support/figure_data"
    data.mkdir(parents=True, exist_ok=True)
    manifest = read(RAW / "confirm_manifest.json")

    # Verify the two code identities that were recorded before confirmation.
    identity = {}
    for field, name in [("analysis_code_sha256", "paper_analyze.py"),
                        ("confirmation_code_sha256", "paper_confirm.py")]:
        identity[name] = hashlib.sha256((LANG / "code" / name).read_bytes()).hexdigest() == manifest[field]
        if not identity[name]:
            raise AssertionError(f"Locked code identity mismatch: {name}")
    spec = importlib.util.spec_from_file_location("frozen_analysis", LANG / "code/paper_analyze.py")
    analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analysis)

    arrays, scores, rows = {}, {}, []
    for model in manifest["models"]:
        folder = RAW / "confirmation" / model["id"]
        saved = read(folder / "complete.json")
        assert saved["model"] == model
        for split, expected in saved["results"].items():
            with np.load(folder / f"{split}.npz", allow_pickle=False) as archive:
                arr = {key: archive[key] for key in archive.files}
            arrays[model["id"], split] = arr
            n = int(arr["counts"].sum())
            ce = float(arr["loss_sum"].sum() / n)
            accuracy = float(arr["correct"].sum() / n)
            close([ce, accuracy], [expected["ce"], expected["accuracy"]], model["id"])
            assert n == expected["masked_tokens"] and len(arr["block_id"]) == expected["texts"]
            scores[model["id"], split] = ce
            rows.append({**model, "split": split, "ce": ce, "accuracy": accuracy,
                         "accuracy_pct": accuracy * 100, "masked_tokens": n})
    pd.DataFrame(rows).to_csv(out / "quality_recomputed.csv", index=False)

    # Use the locked bootstrap and paired-seed procedure on the preserved block arrays.
    differences = []
    for saved in read(RAW / "quality_differences.json"):
        split, stage = saved["split"], saved["stage"]
        if stage == "formal_equal_time":
            chosen = {(p["T"], p["seed"]): p["step"] for p in manifest["time_budget"]["points"]
                      if p["fraction"] == saved["fraction"] and "step" in p}
            seeds = [8621, 8622, 8623]
            ids = lambda T: [f"formal_T{T}_seed{s}_step{chosen[T,s]}" for s in seeds]
        else:
            seeds = [8621, 8622, 8623] if stage == "formal" else [7621, 7622, 7623]
            ids = lambda T: [f"{stage}_T{T}_seed{s}_step{saved['step']}" for s in seeds]
        a = [arrays[mid, split] for mid in ids(2)]
        b = [arrays[mid, split] for mid in ids(4)] if saved["comparison"] == "matched" else [arrays["frozen_T4", split]] * 3
        result = analysis.difference(a, b)
        for key, value in result.items():
            close(value, saved[key], f"difference {stage}/{split}/{key}")
        differences.append({**saved, **result})
    write(data / "quality_differences.json", differences)

    q = pd.DataFrame(rows)
    selected = q[(q["split"] == "test") & ((q.stage != "formal") | q.step.isin([512, 1024, 2048, 4096]))]
    summary = selected.groupby(["stage", "step", "T"], sort=False).agg(
        ce_mean=("ce", "mean"), ce_sd=("ce", "std"), accuracy_pct_mean=("accuracy_pct", "mean")).reset_index()
    summary.to_csv(data / "quality_summary.csv", index=False)

    units, calls, unavailable = [], [], []
    groups = {"historical", "formal", "legacy", "masked"}
    for path in sorted((RAW / "benchmark").rglob("T*_b*.json")):
        group = path.relative_to(RAW / "benchmark").parts[0]
        if group not in groups:
            continue  # E1 migration probes are retained separately.
        r = read(path)
        if r["status"] != "complete":
            unavailable.append(path.relative_to(RAW).as_posix())
            continue
        medians = {engine: float(np.median([c["wall_ms"] for c in r["measurements"] if c["engine"] == engine]))
                   for engine in r["medians"]}
        for engine, value in medians.items():
            close(value, r["medians"][engine], f"latency median {path.name}/{engine}")
        units.append({**r, "group": group, "medians": medians})
        calls.extend({"group": group, "tag": r["tag"], "T": r["T"], "batch": r["batch"],
                      "route": r["route"], "session": r["session"], "gpu_id": r["gpu_id"], **c}
                     for c in r["measurements"])
    pd.DataFrame(calls).to_csv(out / "timing_calls_recomputed.csv", index=False)
    paired = {}
    for r in units:
        tag = re.sub(r"_T[24]_", "_Tpaired_", r["tag"]) if r["group"] in {"formal", "legacy"} else r["tag"]
        key = (r["architecture"], tag, r["session"], r["batch"], r["route"], r["group"])
        paired.setdefault(key, {})[r["T"]] = r
    ratios = []
    for key, pair in sorted(paired.items()):
        assert set(pair) == {2, 4}
        a, b = pair[2]["medians"], pair[4]["medians"]
        row = dict(zip(["architecture", "tag", "session", "batch", "route", "group"], key))
        row.update(cross_T2_eager_over_T4_graph=a["eager"] / b["graph"],
                   same_T4_graph_over_T2_graph=b["graph"] / a["graph"])
        if "eager_with_copy" in a:
            row["matched_copy_T2_eager_over_T4_graph"] = a["eager_with_copy"] / b["graph_with_copy"]
        ratios.append(row)
    pd.DataFrame(ratios).to_csv(data / "execution_ratios.csv", index=False)

    frontier = []
    for r in units:
        if r["group"] != "formal":
            continue
        for engine in ["eager", "graph"]:
            frontier.append({"seed": r["session"], "batch": r["batch"], "id": r["tag"], "T": r["T"],
                             "engine": engine, "latency_ms": r["medians"][engine],
                             "ce": scores[r["tag"], "test"], "gpu_id": r["gpu_id"]})
    for row in frontier:
        peers = [r for r in frontier if (r["seed"], r["batch"]) == (row["seed"], row["batch"])]
        row["nondominated"] = not any(r["latency_ms"] <= row["latency_ms"] and r["ce"] <= row["ce"]
                                     and (r["latency_ms"] < row["latency_ms"] or r["ce"] < row["ce"])
                                     for r in peers)
    pd.DataFrame(frontier).to_csv(data / "frontier.csv", index=False)

    search, runtime = [], []
    for path in sorted((RAW / "screen").glob("*/evaluations.json")):
        cfg = read(path.parent / "config.json")
        search.extend({"T": cfg["T"], "seed": cfg["seed"], "lr": cfg["lr"], "step": e["step"],
                       "ce": e["tune"]["ce"]} for e in read(path))
    pd.DataFrame(search).to_csv(data / "learning_rate_search.csv", index=False)
    for path in sorted((RAW / "formal").glob("*/steps.jsonl")):
        cfg = read(path.parent / "config.json")
        steps = [json.loads(s) for s in path.read_text().splitlines()]
        for start in range(0, 4096, 256):
            chunk = [s["seconds"] for s in steps if start < s["step"] <= start + 256]
            assert len(chunk) == 256
            runtime.append({"seed": cfg["seed"], "T": cfg["T"], "midpoint_step": start + 128,
                            "median_seconds": float(np.median(chunk))})
    pd.DataFrame(runtime).to_csv(data / "training_runtime_chunks.csv", index=False)
    pd.DataFrame(manifest["time_budget"]["points"]).to_csv(data / "time_budget_points.csv", index=False)

    # Compare regenerated figure tables with the supplied paper-source data, allowing only row order.
    keys = {"quality_summary.csv": ["stage", "step", "T"],
            "execution_ratios.csv": ["group", "architecture", "tag", "session", "batch", "route"],
            "frontier.csv": ["seed", "batch", "id", "engine"],
            "learning_rate_search.csv": ["T", "seed", "lr", "step"],
            "training_runtime_chunks.csv": ["seed", "T", "midpoint_step"],
            "time_budget_points.csv": ["T", "seed", "fraction"]}
    for name, sort_keys in keys.items():
        a = pd.read_csv(data / name).sort_values(sort_keys).reset_index(drop=True)
        b = pd.read_csv(ROOT / "figure_source/published_data" / name).sort_values(sort_keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(a[b.columns], b, check_dtype=False, check_exact=False, rtol=1e-11, atol=1e-11)

    primary = [r for r in differences if r["split"] == "test" and r["stage"] == "formal" and r.get("step") == 4096]
    report = {"status": "passed", "quality_rows": len(rows), "locked_models": len(manifest["models"]),
              "paired_differences": len(differences), "timing_units": len(units), "timing_calls": len(calls),
              "unavailable_timing_units": unavailable, "figure_source_tables_match": True,
              "locked_analysis_code_identities": identity, "checkpoint_weights_included": False,
              "checkpoint_identity_source": "confirm_manifest.json; weights are external assets",
              "ABA_checks": sum(len(r["checks"]) for r in units),
              "all_ABA_bitwise_equal": all(c["bitwise_equal"] for r in units for c in r["checks"]),
              "primary_differences": {r["comparison"]: r["ce_difference_mean"] for r in primary},
              "figures_redrawn": args.figures}
    if args.figures:
        for name in ["make_figures.py", "make_workflow.py"]:
            script = out / "support" / name
            shutil.copy2(ROOT / "figure_source" / name, script)
            subprocess.run([sys.executable, str(script)], check=True)
    write(out / "verification.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
