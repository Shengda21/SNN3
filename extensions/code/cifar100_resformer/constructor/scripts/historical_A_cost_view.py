"""Recompute the historical A-conf measured training-time view from its ZIP.

Only recorded 512/1024/2048 endpoints are emitted. The retained original
raw-selected MTT policy is distinct from the prospective BN-selected policy.
No GPU operation, weight replay, quality interpolation, or new training occurs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

SEEDS = (10021, 10022, 10023)
BUDGETS = (512, 1024, 2048)
POLICY = "A-conf-raw-selection"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class RecordedArchive:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.zip = ZipFile(self.path)
        self.members = {}

    def read(self, name):
        data = self.zip.read(name)
        self.members[name] = {"member": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        return data

    def json(self, name):
        return json.loads(self.read(name))

    def close(self):
        self.zip.close()


def run(archive_path, output):
    archive = RecordedArchive(archive_path)
    try:
        selection = archive.json("extensions/selection.json")
        jobs = archive.json("extensions/formal_jobs.json")
        queue = archive.json("extensions/scoring_queue.json")
        quality_summary = pd.read_csv(io.BytesIO(archive.read("extensions/analysis/quality_scores.csv")))
        strategy_summary = pd.read_csv(io.BytesIO(archive.read("extensions/analysis/strategy_quality_costs_by_seed.csv")))
        require(selection["mtt_joint"] == 8e-5 and selection["uses_test"] is False,
                "Historical policy differs from the original raw-selected A-conf MTT policy")
        require(selection["strategy_common_rates"] == [5e-6, 2e-5, 8e-5, 2e-4, 5e-4],
                "Historical strategies do not share the expected five-rate candidate set")
        chosen = selection["dual_native_common"]
        selected_jobs = {}
        timing = {}
        hardware = set()
        initial_sources = set()
        quality_reference = None
        score_cache = {}

        def job_for(method, seed, T=None):
            role = "mtt_joint" if method == "mtt" else "dual_native_common"
            rows = [row for row in jobs if row["method"] == method and row["seed"] == seed
                    and row["T"] == T and role in row["roles"]]
            require(len(rows) == 1, f"Missing/ambiguous historical strategy trajectory: {(method, seed, T)}")
            job = rows[0]
            expected_rate = selection["mtt_joint"] if method == "mtt" else chosen[str(T)]
            require(job["lr"] == expected_rate, "Formal job does not match validation-selected policy")
            key = (method, seed, T)
            selected_jobs[key] = job
            return job

        def measured_times(job):
            key = (job["stage"], job["tag"])
            if key in timing:
                return timing[key]
            base = f"r3/results/{job['stage']}/{job['tag']}"
            complete = archive.json(base+"/complete.json")
            steps = [json.loads(line) for line in archive.read(base+"/steps.jsonl").decode().splitlines() if line]
            require(complete["status"] == "complete" and complete["step"] == 2048,
                    f"Incomplete historical trajectory: {base}")
            cfg = complete["config"]
            require(cfg["seed"] == job["seed"] and cfg["lr"] == job["lr"], "Training log identity mismatch")
            if job["method"] == "native":
                require(cfg["T"] == job["T"], "Native timestep identity mismatch")
            else:
                require(cfg["method"] == "MTT_stagewise", "Unexpected historical MTT implementation")
            hardware.add((cfg["gpu"], cfg["torch"], cfg["cuda"]))
            initial_sources.add(cfg["initial_checkpoint"])
            require([row["step"] for row in steps] == list(range(1, 2049)), "Historical step log is incomplete or duplicated")
            cumulative = 0.
            by_budget = {}
            for row in steps:
                require(math.isfinite(row["seconds"]) and row["seconds"] > 0, "Invalid measured step time")
                cumulative += row["seconds"]
                require(abs(cumulative-row["training_seconds"]) < 1e-6, "Step-time integral differs from recorded cumulative training time")
                if row["step"] in BUDGETS:
                    by_budget[row["step"]] = row["training_seconds"]
            evaluations = {row["step"]: row for row in complete["evaluations"]}
            require(set(evaluations) == set(BUDGETS), "Historical validation budget endpoints incomplete")
            for K in BUDGETS:
                require(abs(by_budget[K]-evaluations[K]["training_seconds"]) < 1e-6, "Endpoint and step training times disagree")
            require(abs(by_budget[2048]-complete["training_seconds"]) < 1e-6, "Final cumulative training time mismatch")
            timing[key] = {"times": by_budget, "evaluations": evaluations,
                           "complete_member": base+"/complete.json", "steps_member": base+"/steps.jsonl"}
            return timing[key]

        def score(job, K, e, bn):
            nonlocal quality_reference
            source_id = f"{job['stage']}_{job['tag']}_K{K}"
            rows = [row for row in queue if row["source_id"] == source_id and row["infer_T"] == e
                    and ((row["calibration"] == "bn_cross" and row.get("cal_T") == e and row.get("cal_seed") == 9640)
                         if bn else row["calibration"] == "raw")]
            require(len(rows) == 1, f"Missing or ambiguous historical quality state: {(source_id, e, bn)}")
            row = rows[0]
            if row["id"] in score_cache:
                return score_cache[row["id"]]
            base = f"r3/results/scoring/{row['id']}"
            meta = archive.json(base+"/complete.json")
            require(meta["status"] == "complete" and meta["row"] == row, "Score marker identity mismatch")
            arrays_bytes = archive.read(base+"/test.npz")
            with np.load(io.BytesIO(arrays_bytes), allow_pickle=False) as source:
                arrays = {name: source[name] for name in ("loss_sum", "counts", "correct", "pred", "labels", "block_id")}
            require(all(values.shape == (10000,) and np.isfinite(values).all() for values in arrays.values()), "Invalid per-example test arrays")
            require(np.array_equal(arrays["counts"], np.ones(10000)), "Wrong test target counts")
            require(np.array_equal(arrays["correct"], arrays["pred"] == arrays["labels"]), "Prediction/correctness mismatch")
            order = np.argsort(arrays["block_id"])
            require(np.array_equal(arrays["block_id"][order], np.arange(10000)), "Test example ids missing or duplicated")
            labels = arrays["labels"][order]
            if quality_reference is None:
                quality_reference = labels.copy()
            require(np.array_equal(quality_reference, labels), "Test labels differ across historical states")
            ce = float(arrays["loss_sum"].sum(dtype=np.float64)/10000)
            accuracy = float(arrays["correct"].sum(dtype=np.float64)/10000)
            require(abs(ce-meta["ce"]) < 1e-10 and abs(accuracy-meta["accuracy"]) < 1e-12, "Per-example recomputation differs from score marker")
            summary = quality_summary[quality_summary.id == row["id"]]
            require(len(summary) == 1 and abs(float(summary.ce.iloc[0])-ce) < 1e-10
                    and abs(float(summary.accuracy.iloc[0])-accuracy) < 1e-12, "Quality CSV differs from per-example recomputation")
            result = {"ce": ce, "accuracy": accuracy, "source_id": source_id,
                      "checkpoint_sha256": row["checkpoint_sha256"], "scoring_member": base+"/test.npz",
                      "scoring_member_sha256": hashlib.sha256(arrays_bytes).hexdigest()}
            score_cache[row["id"]] = result
            return result

        output_rows = []
        brackets = []
        for seed in SEEDS:
            native = {T: job_for("native", seed, T) for T in (2, 4)}
            mtt = job_for("mtt", seed)
            native_records = {T: measured_times(job) for T, job in native.items()}
            mtt_record = measured_times(mtt)
            dual_time = {K: sum(native_records[T]["times"][K] for T in (2, 4)) for K in BUDGETS}
            for K in BUDGETS:
                for strategy in ("independent_T2_T4", "MTT_single"):
                    is_mtt = strategy == "MTT_single"
                    time = mtt_record["times"][K] if is_mtt else dual_time[K]
                    for e in (2, 4):
                        selected_job = mtt if is_mtt else native[e]
                        measured = mtt_record if is_mtt else native_records[e]
                        raw_validation = measured["evaluations"][K]["validation_by_window"][str(e)] if is_mtt else measured["evaluations"][K]["validation"]
                        require(raw_validation["split"] == "validation" and raw_validation["samples"] == 5000, "Wrong validation provenance")
                        for bn in (False, True):
                            scored = score(selected_job, K, e, bn)
                            normalization = "matched_BN" if bn else "raw"
                            if K == 2048:
                                check = strategy_summary[(strategy_summary.strategy == strategy)
                                          & (strategy_summary.seed == seed) & (strategy_summary.inference_T == e)
                                          & (strategy_summary.normalization == normalization)]
                                require(len(check) == 1 and abs(float(check.ce.iloc[0])-scored["ce"]) < 1e-10
                                        and abs(float(check.training_seconds_this_formal_candidate.iloc[0])-time) < 1e-6,
                                        "Final quality/cost strategy table differs from recomputation")
                            cost_members = [mtt_record["steps_member"]] if is_mtt else [native_records[T]["steps_member"] for T in (2, 4)]
                            output_rows.append({"cohort": "A-conf", "policy_id": POLICY, "strategy": strategy,
                                "seed": seed, "updates_per_trajectory": K, "training_seconds_policy": time,
                                "training_time_scope": "MTT trajectory" if is_mtt else "sum of independent T2 and T4 trajectories",
                                "inference_T": e, "normalization": normalization, "split": "test",
                                "ce": scored["ce"], "accuracy": scored["accuracy"],
                                "raw_validation_ce": raw_validation["ce"] if not bn else None,
                                "raw_validation_accuracy": raw_validation["accuracy"] if not bn else None,
                                "quality_source_id": scored["source_id"], "checkpoint_sha256": scored["checkpoint_sha256"],
                                "quality_npz_member": scored["scoring_member"], "quality_npz_sha256": scored["scoring_member_sha256"],
                                "training_log_members": " | ".join(cost_members)})
            reference = dual_time[2048]
            lower = [K for K in BUDGETS if mtt_record["times"][K] <= reference]
            upper = [K for K in BUDGETS if mtt_record["times"][K] >= reference]
            require(bool(lower) and bool(upper), "The independent policy reference time is not bracketed by saved MTT times")
            lo, hi = max(lower), min(upper)
            brackets.append({"seed": seed, "dual_updates_each": 2048, "dual_total_training_seconds": reference,
                "MTT_measured_lower_updates": lo, "MTT_measured_lower_training_seconds": mtt_record["times"][lo],
                "MTT_measured_upper_updates": hi, "MTT_measured_upper_training_seconds": mtt_record["times"][hi],
                "meaning": "training-time bracket only; it does not bracket CE or accuracy and creates no interpolated model state"})
        require(len(hardware) == 1 and len(initial_sources) == 1, "Historical trajectories do not share the recorded hardware label and initial source")
        require(len(output_rows) == 72, "Measured endpoint grid incomplete")
        output = Path(output).resolve()
        output.mkdir(parents=True, exist_ok=True)
        destination = output/"A_conf_equal_time_measured_points.csv"
        pd.DataFrame(output_rows).to_csv(destination, index=False, float_format="%.15g")
        provenance = {"status": "passed", "cohort": "A-conf", "policy_id": POLICY,
            "original_raw_MTT_selected_lr": selection["mtt_joint"], "dual_common_selected_lr": chosen,
            "input_archive": str(archive.path), "input_archive_sha256": file_sha(archive.path),
            "output_file": destination.name, "output_sha256": file_sha(destination), "measured_rows": len(output_rows),
            "formal_seeds": list(SEEDS), "measured_budgets_only": list(BUDGETS),
            "test_metrics_recomputed_from_per_example_arrays": True, "test_examples_per_state": 10000,
            "training_time_recomputed_from_steps_and_checked_against_endpoint_records": True,
            "timing_scope": "cumulative measured optimization-loop seconds; excludes validation, checkpoint I/O, search, S0 construction, BN calibration, and billing elapsed time",
            "recorded_training_hardware": [{"gpu": gpu, "torch": torch, "cuda": cuda} for gpu, torch, cuda in sorted(hardware)],
            "same_hardware_label_not_a_repeated_device_identity_test": True,
            "retained_initial_checkpoint_recorded_path": next(iter(initial_sources)),
            "model_weight_bytes_replayed_or_rehashed_by_this_script": False,
            "checkpoint_identity_source": "locked historical scoring queue and exact matching score marker; per-example NPZ bytes rehashed now",
            "equal_time_reference_brackets": brackets, "quality_interpolation_performed": False,
            "new_model_state_at_equal_time_created": False, "new_BN_selected_policy_overwritten": False,
            "comparison_with_future_C_hardware_times": False, "GPU_training_or_inference_executed": False,
            "members": [archive.members[name] for name in sorted(archive.members)]}
        (output/"provenance.json").write_text(json.dumps(provenance, indent=2, allow_nan=False), encoding="utf-8")
        return {key: provenance[key] for key in ("status", "measured_rows", "output_file", "output_sha256", "equal_time_reference_brackets")}
    finally:
        archive.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("../unified_results_20261016/export/reports_without_full_weights.zip"))
    parser.add_argument("--output", type=Path, default=Path("historical_analysis"))
    args = parser.parse_args()
    print(json.dumps(run(args.archive, args.output), indent=2, allow_nan=False))
