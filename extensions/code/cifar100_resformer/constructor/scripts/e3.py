"""E3: validation-only matched-BN reselection of the archived A-conf MTT rate.

The original raw-selected policy is never overwritten. Plan and select are CPU
operations; evaluate is the only action that loads models. No formal training
is started by this module.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys

POLICY_ID = "A-conf-BN-selection"
SCREEN_SEEDS = (9611, 9612)
RATES = (5e-6, 2e-5, 8e-5, 2e-4, 5e-4)
FORMAL_SEEDS = (10021, 10022, 10023)
OLD_RATE = 8e-5
CALIBRATION_SEED = 9640


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+".partial")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def source_key(source):
    return int(source["seed"]), float(source["lr"])


def resolve(root, path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (Path(root)/path).resolve()


def validate_sources(sources, root=None):
    require(isinstance(sources, list), "Screen sources must be a list")
    keys = [source_key(source) for source in sources]
    expected = {(seed, rate) for seed in SCREEN_SEEDS for rate in RATES}
    require(len(keys) == 10 and len(set(keys)) == 10 and set(keys) == expected,
            "E3 requires all ten original A-conf MTT screen sources, five rates by two seeds")
    seen = {}
    for source in sources:
        digest = source["checkpoint_sha256"]
        require(isinstance(digest, str) and len(digest) == 64 and set(digest) <= set("0123456789abcdef"),
                f"Invalid checkpoint identity for {source_key(source)}")
        require(bool(source["checkpoint"]), "Missing screen checkpoint path")
        if root is not None:
            path = resolve(root, source["checkpoint"])
            require(path.is_file(), f"Original MTT screen checkpoint missing: {path}")
            if path not in seen:
                seen[path] = sha256(path)
            require(seen[path] == digest, f"Original MTT screen checkpoint hash mismatch: {path}")
    return {source_key(source): source for source in sources}


def select_policy(sources, evaluations, source_manifest_sha256=None):
    """Select the two-seed mean of matched-BN CE over both inference windows."""
    lookup = validate_sources(sources)
    require(len(evaluations) == len(lookup), "Missing or additional calibrated validation measurements")
    measured = {}
    for row in evaluations:
        key = source_key(row)
        require(key in lookup and key not in measured, "Unexpected or duplicate E3 validation record")
        require(row["policy_id"] == POLICY_ID and row["checkpoint_sha256"] == lookup[key]["checkpoint_sha256"],
                "Calibrated validation identity differs from its archived screen source")
        require(row.get("calibration_seed") == CALIBRATION_SEED and row.get("calibration_examples") == 2048
                and row.get("calibration_batch") == 64, "Calibration procedure differs from E3")
        require(row.get("selection_uses_test") is False, "Selection must never use test outcomes")
        windows = row["validation_by_window"]
        require(set(windows) == {"2", "4"}, "Both fixed inference windows must be measured")
        for window in (2, 4):
            value = windows[str(window)]
            require(value.get("split") == "validation" and value.get("samples") == 5000,
                    "E3 screening requires the complete 5000-image validation split")
            require(value.get("calibration") == "matched_BN" and value.get("cal_T") == window,
                    "Selection requires matched BN statistics for each inference window")
            require(isinstance(value["ce"], (float, int)) and math.isfinite(value["ce"]) and value["ce"] >= 0,
                    "Invalid matched-BN validation CE")
        measured[key] = row
    require(set(measured) == set(lookup), "Calibrated validation screen is incomplete")
    candidates = []
    for rate in RATES:
        seed_values = []
        for seed in SCREEN_SEEDS:
            row = measured[(seed, rate)]
            ce = sum(row["validation_by_window"][str(e)]["ce"] for e in (2, 4))/2
            seed_values.append({"seed": seed, "mean_window_ce": ce,
                                "checkpoint_sha256": row["checkpoint_sha256"],
                                "validation_by_window": row["validation_by_window"]})
        candidates.append({"lr": rate, "mean_ce": sum(row["mean_window_ce"] for row in seed_values)/2,
                           "seeds": seed_values})
    selected = min(candidates, key=lambda row: (row["mean_ce"], row["lr"]))["lr"]
    return {"policy_id": POLICY_ID, "cohort": "A-conf", "selected_lr": selected,
            "criterion": "mean matched-BN validation CE over inference windows 2/4 and screen seeds 9611/9612",
            "tie_rule": "exact tie selects smaller learning rate", "candidates": candidates,
            "selection_uses_test": False, "calibration_seed": CALIBRATION_SEED,
            "calibration_examples": 2048, "calibration_batch": 64,
            "old_policy_id": "A-conf-raw-selection", "old_selected_lr": OLD_RATE,
            "old_policy_preserved": True, "same_as_old": selected == OLD_RATE,
            "conditional_retrain_needed": selected != OLD_RATE,
            "source_manifest_sha256": source_manifest_sha256,
            "formal_training_started_by_this_action": False,
            "interpretation": "unchanged rate permits reuse of the original formal trajectories; a changed rate requires new formal trajectories and distinct policy labels"}


def formal_tasks(selection, retained_start=None):
    if not selection["conditional_retrain_needed"]:
        return []
    require(retained_start is not None and "checkpoint" in retained_start and "checkpoint_sha256" in retained_start,
            "A changed selection requires the retained A start identity, not a screen checkpoint")
    tasks = []
    for seed in FORMAL_SEEDS:
        tasks.append({"id": f"Aconf_BNpolicy_mtt_s{seed}_lr{selection['selected_lr']:g}",
                      "cohort": POLICY_ID, "policy_id": POLICY_ID, "method": "mtt",
                      "seed": seed, "lr": selection["selected_lr"], "steps": 2048,
                      "budgets": [512, 1024, 2048], "initial_checkpoint": retained_start["checkpoint"],
                      "initial_checkpoint_sha256": retained_start["checkpoint_sha256"],
                      "reason": "calibrated validation criterion changed the selected rate",
                      "overwrite_original_Aconf": False})
    return tasks


def load_sources(root):
    path = root/"protocol/a_mtt_screen_sources.json"
    sources = json.loads(path.read_text(encoding="utf-8"))
    validate_sources(sources, root)
    return sources, sha256(path)


def result_path(root, source):
    return root/"results/e3"/POLICY_ID/f"screen_s{source['seed']}_lr{source['lr']:g}.json"


def evaluate_sources(root, api=None):
    """Evaluate all ten sources using train-only calibration and validation CE.

    No learning-rate outcome is read from the test set, and this function never
    launches the three conditional formal trajectories.
    """
    root = Path(root).resolve()
    sources, source_manifest_digest = load_sources(root)
    protocol_file = root/"environment/protocol.json"
    require(protocol_file.is_file(), "Prepare the locked environment/protocol.json before E3 evaluation")
    protocol_digest = sha256(protocol_file)
    api = importlib.import_module("runtime") if api is None else api
    api.configure()
    api_file = Path(api.__file__).resolve()
    api_digest = sha256(api_file)
    for source in sources:
        cache = result_path(root, source)
        expected_identity = {"policy_id": POLICY_ID, "seed": source["seed"], "lr": source["lr"],
                             "checkpoint_sha256": source["checkpoint_sha256"],
                             "source_manifest_sha256": source_manifest_digest,
                             "protocol_sha256": protocol_digest, "runtime_sha256": api_digest}
        if cache.exists():
            old = json.loads(cache.read_text(encoding="utf-8"))
            require(all(old.get(key) == value for key, value in expected_identity.items()),
                    f"Cached E3 measurement identity differs; use a fresh run root: {cache}")
            require(old.get("status") == "complete", f"Incomplete E3 cache: {cache}")
            continue
        windows = {}
        reports = {}
        for e in (2, 4):
            model = api.load_model(root, "resformer", resolve(root, source["checkpoint"]), e)
            api.prepare_mode(model, False)
            bank, report = api.calibrate(root, model, seed=CALIBRATION_SEED)
            require(report.get("samples") == 2048 and report.get("batches") == 32,
                    "Runtime calibration did not process the required 2048 training images in 32 batches")
            require(report.get("protected_values_and_dtypes_unchanged") is True,
                    "Runtime must verify parameters and non-BN persistent state are unchanged")
            metric = api.evaluate_validation(root, model)
            windows[str(e)] = {**metric, "calibration": "matched_BN", "cal_T": e}
            reports[str(e)] = report
            del model, bank
            # CUDA memory cleanup is operational only and does not alter the
            # model state or any measurement. runtime already depends on torch.
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        measured = {**expected_identity, "checkpoint": source["checkpoint"], "status": "complete",
                    "calibration_seed": CALIBRATION_SEED, "calibration_examples": 2048,
                    "calibration_batch": 64, "selection_uses_test": False,
                    "validation_by_window": windows, "calibration_reports": reports}
        # Run the same value and split checks as final selection before caching.
        for e in (2, 4):
            value = windows[str(e)]
            require(value.get("split") == "validation" and value.get("samples") == 5000
                    and math.isfinite(value["ce"]) and value["ce"] >= 0,
                    "Runtime returned an invalid validation measurement")
        write_json(cache, measured)
    return finish_selection(root)


def finish_selection(root):
    sources, source_digest = load_sources(root)
    measurements = [json.loads(result_path(root, source).read_text(encoding="utf-8")) for source in sources]
    protocol_digest = sha256(root/"environment/protocol.json")
    runtime_digest = sha256(Path(__file__).with_name("runtime.py"))
    for row in measurements:
        require(row.get("status") == "complete" and row.get("source_manifest_sha256") == source_digest
                and row.get("protocol_sha256") == protocol_digest and row.get("runtime_sha256") == runtime_digest,
                "E3 cached measurements are incomplete or no longer match the locked protocol/runtime/source list")
    selection = select_policy(sources, measurements, source_digest)
    retained = None
    if selection["conditional_retrain_needed"]:
        retained_file = root/"protocol/a_mtt_retained_start.json"
        require(retained_file.is_file(), "Prepare protocol/a_mtt_retained_start.json before generating changed-rate formal jobs")
        retained = json.loads(retained_file.read_text(encoding="utf-8"))
        checkpoint = resolve(root, retained["checkpoint"])
        require(checkpoint.is_file() and sha256(checkpoint) == retained["checkpoint_sha256"],
                "Retained A start hash mismatch")
    tasks = formal_tasks(selection, retained)
    directory = root/"results/e3"/POLICY_ID
    write_json(directory/"selection.json", selection)
    write_json(directory/"conditional_formal_tasks.json", tasks)
    return selection


def self_check():
    sources = [{"seed": seed, "lr": rate, "checkpoint": f"screen_s{seed}_lr{rate:g}.pt",
                "checkpoint_sha256": hashlib.sha256(f"{seed}_{rate}".encode()).hexdigest()}
               for seed in SCREEN_SEEDS for rate in RATES]

    def measurements(winner):
        rows = []
        for source in sources:
            ce = 1. if source["lr"] == winner else 2.
            rows.append({**source, "policy_id": POLICY_ID, "calibration_seed": CALIBRATION_SEED,
                         "calibration_examples": 2048, "calibration_batch": 64, "selection_uses_test": False,
                         "validation_by_window": {str(e): {"ce": ce+e*.001, "split": "validation",
                            "samples": 5000, "calibration": "matched_BN", "cal_T": e} for e in (2, 4)}})
        return rows

    unchanged = select_policy(sources, measurements(OLD_RATE))
    require(unchanged["same_as_old"] and formal_tasks(unchanged) == [], "Unchanged selection must not retrain")
    changed = select_policy(sources, measurements(2e-5))
    tasks = formal_tasks(changed, {"checkpoint": "retained_A.pt", "checkpoint_sha256": "a"*64})
    require(len(tasks) == 3 and {task["seed"] for task in tasks} == set(FORMAL_SEEDS), "Wrong conditional formal job grid")
    ties = measurements(OLD_RATE)
    for row in ties:
        for value in row["validation_by_window"].values():
            value["ce"] = 1.
    require(select_policy(sources, ties)["selected_lr"] == min(RATES), "Tie must prefer smaller rate")
    corrupted = measurements(OLD_RATE)
    corrupted[0]["checkpoint_sha256"] = "0"*64
    try:
        select_policy(sources, corrupted)
    except ValueError:
        pass
    else:
        raise AssertionError("Source identity corruption accepted")
    try:
        select_policy(sources, measurements(OLD_RATE)[:-1])
    except ValueError:
        pass
    else:
        raise AssertionError("Incomplete screen accepted")
    return {"status": "passed", "scope": "pure software fixtures; no scientific results",
            "unchanged_rate_reuses_existing_trajectories": True, "changed_rate_creates_three_jobs": True,
            "tie_smaller_rate": True, "source_mismatch_rejected": True, "incomplete_grid_rejected": True,
            "GPU_experiments_executed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "evaluate", "select", "self-check"), nargs="?", default="plan")
    parser.add_argument("--root", type=Path, default=Path("runs/revision2"))
    args = parser.parse_args()
    if args.action == "self-check":
        print(json.dumps(self_check(), indent=2))
    elif args.action == "select":
        print(json.dumps(finish_selection(args.root.resolve()), indent=2))
    elif args.action == "evaluate":
        print(json.dumps(evaluate_sources(args.root.resolve()), indent=2))
    else:
        print(json.dumps({"policy_id": POLICY_ID, "sources_required": 10, "screen_seeds": SCREEN_SEEDS,
                          "rates": RATES, "calibration_seed": CALIBRATION_SEED,
                          "selection": "matched-BN validation CE, mean over two windows and two screen seeds",
                          "default_action_starts_GPU_work": False, "formal_training_is_conditional": True,
                          "calibrated_validation_evaluation": "called by the revision2 model driver"}, indent=2))
