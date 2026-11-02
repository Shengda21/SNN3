"""Recompute revision-2 scores with separate C-conf and F-arch test families.

The repeated unit is an adaptation seed conditional on one retained start.
Cohorts are never pooled into independent initialization repeats. J follows the
manuscript definition: change in the BN contrast minus change in raw contrast.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from scipy import stats

FAMILIES = {
    "C-conf": {"seeds": [11021, 11022, 11023],
               "contrasts": ["P2_raw_K2048", "G44_raw_K2048", "P4_BN_K2048",
                             "J_512_to_2048", "MTT_minus_dual_CE_e2", "MTT_minus_dual_CE_e4"]},
    "F-arch": {"seeds": [13021, 13022, 13023],
               "contrasts": ["P2_raw_K2048", "G44_raw_K2048", "P4_BN_K2048"]},
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def resolve(root, path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (root/path).resolve()


def holm(p_values):
    values = np.asarray(p_values, dtype=float)
    require(values.ndim == 1 and bool(np.isfinite(values).all()) and bool(((values >= 0) & (values <= 1)).all()),
            "Invalid p values")
    order = np.argsort(values, kind="stable")
    adjusted = np.maximum.accumulate(values[order]*np.arange(len(values), 0, -1))
    result = np.empty(len(values))
    result[order] = np.minimum(1., adjusted)
    return result


def paired_interval(values, family_size=1):
    values = np.asarray(values, dtype=float)
    require(values.ndim == 1 and len(values) >= 2 and bool(np.isfinite(values).all()),
            "At least two finite adaptation-seed contrasts required")
    n = len(values)
    mean = float(values.mean())
    se = float(values.std(ddof=1)/math.sqrt(n))
    half = float(stats.t.ppf(.975, n-1)*se)
    family_half = float(stats.t.ppf(1-.05/(2*family_size), n-1)*se)
    p = float(2*stats.t.sf(abs(mean/se), n-1)) if se > 0 else (1. if mean == 0 else 0.)
    return {"n_adaptation_seeds": n, "mean": mean, "standard_error": se, "degrees_of_freedom": n-1,
            "ci95_low": mean-half, "ci95_high": mean+half,
            "family_ci95_low": mean-family_half, "family_ci95_high": mean+family_half,
            "p_unadjusted": p}


class ScoreReader:
    def __init__(self, root):
        self.root = Path(root).resolve()
        locations = [self.root/"results/scoring", self.root/"r3/results/scoring"]
        found = [path for path in locations if path.is_dir()]
        require(len(found) == 1, "Expected exactly one revision-2 scoring directory (results/scoring or r3/results/scoring)")
        self.directory = found[0]
        self.checkpoints = {}
        self.labels = None

    def read(self, row):
        require(Path(row["id"]).name == row["id"] and row["id"] not in ("", ".", ".."), "Invalid scoring id")
        folder = self.directory/row["id"]
        require((folder/"complete.json").is_file() and (folder/"test.npz").is_file(), f"Incomplete scoring row: {row['id']}")
        meta = json.loads((folder/"complete.json").read_text(encoding="utf-8"))
        require(meta.get("status") == "complete" and meta.get("row") == row, f"Score identity mismatch: {row['id']}")
        checkpoint = resolve(self.root, row["checkpoint"])
        if checkpoint not in self.checkpoints:
            require(checkpoint.is_file(), f"Missing checkpoint: {checkpoint}")
            self.checkpoints[checkpoint] = digest(checkpoint)
        require(self.checkpoints[checkpoint] == row["checkpoint_sha256"], f"Source SHA256 mismatch: {row['id']}")
        with np.load(folder/"test.npz", allow_pickle=False) as source:
            keys = ["block_id", "loss_sum", "counts", "correct", "pred", "labels"]
            require(all(key in source.files for key in keys), f"Incomplete per-example arrays: {row['id']}")
            data = {key: source[key] for key in keys}
        for key, values in data.items():
            require(values.shape == (10000,) and bool(np.isfinite(values).all()), f"Invalid 10000-example {key}: {row['id']}")
        for key in ("block_id", "counts", "correct", "pred", "labels"):
            require(bool(np.equal(data[key], data[key].astype(np.int64)).all()), f"Noninteger {key}: {row['id']}")
        require(bool(np.equal(data["counts"], 1).all()), f"Wrong target counts: {row['id']}")
        require(bool((data["loss_sum"] >= -1e-7).all()), f"Negative CE: {row['id']}")
        require(bool(((data["labels"] >= 0) & (data["labels"] < 100)).all()
                     and ((data["pred"] >= 0) & (data["pred"] < 100)).all()), f"Class outside CIFAR-100 range: {row['id']}")
        require(bool(np.array_equal(data["correct"], data["pred"] == data["labels"])), f"Correctness/prediction mismatch: {row['id']}")
        order = np.argsort(data["block_id"])
        require(bool(np.array_equal(data["block_id"][order], np.arange(10000))), f"Duplicate/missing test ids: {row['id']}")
        labels = data["labels"][order]
        if self.labels is None:
            self.labels = labels.copy()
        require(bool(np.array_equal(labels, self.labels)), f"Test labels differ across rows: {row['id']}")
        ce = float(data["loss_sum"].sum(dtype=np.float64)/10000)
        accuracy = float(data["correct"].sum(dtype=np.float64)/10000)
        for key, value in (("ce", ce), ("accuracy", accuracy)):
            require(key in meta and math.isfinite(float(meta[key])) and abs(float(meta[key])-value) <= 1e-8,
                    f"Recomputed {key} disagrees with aggregate marker: {row['id']}")
        return {**row, "ce": ce, "accuracy": accuracy, "test_examples": 10000,
                "score_file": str((folder/"test.npz").relative_to(self.root)), "score_sha256": digest(folder/"test.npz")}


def role_matches(row, role):
    roles = row.get("roles")
    if roles is not None:
        return role in roles
    explicit = row.get("candidate_role", row.get("role"))
    if explicit in ("native_expanded", "dual_native_common", "mtt_joint", "retain"):
        return explicit == role
    return True


def cohort_contrasts(quality, cohort):
    require(cohort in FAMILIES, f"No primary family specified for {cohort}")
    spec = FAMILIES[cohort]
    rows = [row for row in quality if row["cohort"] == cohort]
    require(bool(rows), f"Missing planned cohort: {cohort}")
    native_seeds = {row["seed"] for row in rows if row["method"] == "native"}
    require(native_seeds == set(spec["seeds"]), f"Wrong/missing adaptation seeds in {cohort}")
    if cohort == "C-conf":
        require({row["seed"] for row in rows if row["method"] == "mtt"} == set(spec["seeds"]), "C-conf MTT seeds incomplete")

    def ce(method, seed, T, K, e, bn=False, role=None):
        eligible = [row for row in rows if row["method"] == method and row["seed"] == seed
                    and row.get("adapt_T") == T and row["updates"] == K and row["infer_T"] == e
                    and ((row["calibration"] == "bn_cross" and row.get("cal_T") == e) if bn else row["calibration"] == "raw")]
        if role is not None:
            eligible = [row for row in eligible if role_matches(row, role)]
        require(len(eligible) == 1,
                f"Expected one {cohort} source for {(method, seed, T, K, e, bn, role)}, got {len(eligible)}")
        return eligible[0]["ce"]

    def p4(seed, K, bn=False):
        return ce("native", seed, 2, K, 4, bn, "native_expanded")-ce("native", seed, 4, K, 4, bn, "native_expanded")

    # Endpoints used in descriptive figures are part of the planned run too.
    # Require their raw and matched-BN measurements rather than allowing a
    # shortened queue to look like completion of the full budget experiment.
    for e in (2, 4):
        for bn in (False, True):
            ce("initial", None, None, 0, e, bn)
    for seed in spec["seeds"]:
        for K in (512, 1024, 2048):
            for T in (2, 4):
                for e in (2, 4):
                    for bn in (False, True):
                        ce("native", seed, T, K, e, bn, "native_expanded")
                        if cohort == "C-conf":
                            ce("native", seed, T, K, e, bn, "dual_native_common")
            if cohort == "C-conf":
                for e in (2, 4):
                    for bn in (False, True):
                        ce("mtt", seed, None, K, e, bn, "mtt_joint")

    output = []
    for seed in spec["seeds"]:
        values = {
            "P2_raw_K2048": ce("native", seed, 4, 2048, 2, role="native_expanded")-ce("native", seed, 2, 2048, 2, role="native_expanded"),
            "G44_raw_K2048": ce("initial", None, None, 0, 4)-ce("native", seed, 4, 2048, 4, role="native_expanded"),
            "P4_BN_K2048": p4(seed, 2048, True),
        }
        if cohort == "C-conf":
            values["J_512_to_2048"] = (p4(seed, 2048, True)-p4(seed, 512, True))-(p4(seed, 2048)-p4(seed, 512))
            for e in (2, 4):
                values[f"MTT_minus_dual_CE_e{e}"] = ce("mtt", seed, None, 2048, e, role="mtt_joint")-ce("native", seed, e, 2048, e, role="dual_native_common")
        require(list(values) == spec["contrasts"], "Primary family order changed")
        output.extend({"cohort": cohort, "seed": seed, "contrast": name, "value": value} for name, value in values.items())
    return pd.DataFrame(output)


def primary_intervals(contrasts, cohort):
    spec = FAMILIES[cohort]
    require(set(contrasts.cohort) == {cohort}, "Cohort families must not be pooled")
    records = []
    for contrast in spec["contrasts"]:
        group = contrasts[contrasts.contrast == contrast]
        require(len(group) == 3 and set(group.seed) == set(spec["seeds"]), f"Missing/duplicate paired contrast: {cohort} {contrast}")
        records.append({"cohort": cohort, "contrast": contrast,
                        **paired_interval(group.sort_values("seed").value.to_numpy(), len(spec["contrasts"])),
                        "family_size": len(spec["contrasts"]), "independent_start_count": 1,
                        "inferential_unit": "paired adaptation seed conditional on one retained start"})
    result = pd.DataFrame(records)
    result["p_Holm"] = holm(result.p_unadjusted)
    result["holm_reject_05"] = result.p_Holm < .05
    return result


def run(root, cohorts=("C-conf", "F-arch")):
    root = Path(root).resolve()
    protocol_file = root/"environment/protocol.json"
    protocol = json.loads(protocol_file.read_text(encoding="utf-8"))
    require(protocol["budgets"] == [512, 1024, 2048] and protocol["updates"] == 2048,
            "Analysis endpoints differ from the pre-specified revision-2 plan")
    for cohort in cohorts:
        require(protocol["cohorts"][cohort]["formal_seeds"] == FAMILIES[cohort]["seeds"],
                f"Protocol seed family changed: {cohort}")
    queue_file = root/"protocol/scoring_queue.json"
    source_lock = json.loads((root/"protocol/source_lock.json").read_text(encoding="utf-8"))
    require(source_lock["queue_sha256"] == digest(queue_file), "Scoring queue changed after source lock")
    require(source_lock["execution_protocol_sha256"] == digest(protocol_file), "Execution protocol changed after source lock")
    queue = json.loads(queue_file.read_text(encoding="utf-8"))
    require(isinstance(queue, list) and queue and len({row["id"] for row in queue}) == len(queue), "Empty/duplicate scoring queue")
    for row in queue:
        if row["cohort"] in cohorts:
            require(row["architecture"] == protocol["cohorts"][row["cohort"]]["architecture"],
                    f"Source architecture differs from its cohort: {row['id']}")
            if row["calibration"] != "raw":
                require(row.get("cal_seed") == 9640, "Primary analysis requires the registered calibration subset")
    reader = ScoreReader(root)
    quality = [reader.read(row) for row in queue]
    by_seed = []
    intervals = []
    for cohort in cohorts:
        paired = cohort_contrasts(quality, cohort)
        by_seed.append(paired)
        intervals.append(primary_intervals(paired, cohort))
    contrasts = pd.concat(by_seed, ignore_index=True)
    tests = pd.concat(intervals, ignore_index=True)
    # This is a display of cohort estimates, not an omnibus test or a test of
    # the difference between checkpoint construction procedures.
    consistency = tests[["contrast", "cohort", "mean", "ci95_low", "ci95_high"]].copy()
    consistency["sign"] = np.sign(consistency["mean"]).astype(int)
    consistency["comparison_status"] = "descriptive across cohorts; no test of a start effect"
    availability = []
    for contrast in FAMILIES["C-conf"]["contrasts"]:
        for cohort in cohorts:
            eligible = consistency[(consistency.cohort == cohort) & (consistency.contrast == contrast)]
            availability.append({"contrast": contrast, "cohort": cohort,
                                 "available": len(eligible) == 1, "mean": None if eligible.empty else float(eligible["mean"].iloc[0]),
                                 "reason_if_na": "not in this cohort's registered contrast set" if eligible.empty else "",
                                 "counts_toward_sign_denominator": len(eligible) == 1})
    agreement = []
    for contrast in FAMILIES["C-conf"]["contrasts"]:
        group = consistency[consistency.contrast == contrast]
        positive = int((group.sign > 0).sum())
        negative = int((group.sign < 0).sum())
        zero = int((group.sign == 0).sum())
        agreement.append({"contrast": contrast, "available_cohorts": len(group),
                          "positive": positive, "negative": negative, "zero": zero,
                          "dominant_nonzero_sign_count": max(positive, negative),
                          "NA_excluded_from_denominator": True,
                          "interpretation": "descriptive counts, not independent-start evidence or a sign-frequency test"})
    output = root/"analysis/revision2"
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in (("quality_scores", pd.DataFrame(quality)), ("contrasts_by_seed", contrasts),
                        ("primary_intervals", tests), ("cross_cohort_estimates_descriptive", consistency),
                        ("contrast_availability", pd.DataFrame(availability)),
                        ("sign_agreement_descriptive", pd.DataFrame(agreement))):
        frame.to_csv(output/f"{name}.csv", index=False, float_format="%.15g")
    report = {"status": "passed", "scoring_rows_recomputed": len(quality), "checkpoint_hashes_verified": len(reader.checkpoints),
              "queue_sha256": digest(queue_file), "families": {cohort: FAMILIES[cohort] for cohort in cohorts},
              "protocol_sha256": digest(protocol_file),
              "multiplicity": "separate Holm families, six C-conf contrasts and three F-arch contrasts; no omnibus or cross-cohort test",
              "n_definition": "three paired adaptation seeds conditional on each single retained start",
              "J_definition": "[P4_BN(2048)-P4_BN(512)]-[P4_raw(2048)-P4_raw(512)]",
              "non_rejection": "does not establish equivalence or a construction-maturity effect",
              "A_cohorts": "A-expl and A-conf share an initialization; they are not independent starts and are not pooled here",
              "cross_cohort_signs": "descriptive, NA excluded; no sign-frequency significance test",
              "GPU_training_executed_by_analysis": False, "manuscript_modified": False}
    (output/"verification.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


def self_check():
    quality = []
    for cohort, spec in FAMILIES.items():
        sources = [("initial", None, None, 0)]
        for seed in spec["seeds"]:
            for K in (512, 1024, 2048):
                sources.extend(("native", seed, T, K) for T in (2, 4))
                if cohort == "C-conf":
                    sources.append(("mtt", seed, None, K))
        for method, seed, T, K in sources:
            for e in (2, 4):
                for bn in (False, True):
                    # Make deltaBN positive and deltaRaw zero, so J must be
                    # positive. This guards against the old opposite sign.
                    ce = 2. + (T or 0)*.03 + (seed or 0)%10*.001
                    if bn and T == 2 and e == 4:
                        ce += K*.0001
                    quality.append({"cohort": cohort, "method": method, "seed": seed, "adapt_T": T,
                                    "updates": K, "infer_T": e, "calibration": "bn_cross" if bn else "raw",
                                    "cal_T": e if bn else None, "ce": ce})
    c = cohort_contrasts(quality, "C-conf")
    f = cohort_contrasts(quality, "F-arch")
    require(bool(np.allclose(c[c.contrast == "J_512_to_2048"].value, .1536)), "J sign is not deltaBN minus deltaRaw")
    ci = primary_intervals(c, "C-conf")
    fi = primary_intervals(f, "F-arch")
    require(len(ci) == 6 and len(fi) == 3 and set(ci.n_adaptation_seeds) == {3}
            and set(fi.independent_start_count) == {1}, "Wrong cohort family or sampling unit")
    require(bool(np.allclose(holm([.01, .04, .03]), [.03, .06, .06])), "Holm adjustment failed")
    # Seven-rate native and five-rate dual candidates can have different LRs.
    # Give the dual candidates different losses and check that only the MTT
    # contrast uses them, rather than silently using the seven-rate candidate.
    split_roles = []
    for row in quality:
        if row["cohort"] == "C-conf" and row["method"] == "native":
            split_roles.append({**row, "roles": ["native_expanded"]})
            split_roles.append({**row, "roles": ["dual_native_common"], "ce": row["ce"]+.4})
        else:
            split_roles.append(row)
    switched = cohort_contrasts(split_roles, "C-conf")
    for name in FAMILIES["C-conf"]["contrasts"]:
        observed = switched[switched.contrast == name].value.to_numpy()-c[c.contrast == name].value.to_numpy()
        require(bool(np.allclose(observed, -.4 if name.startswith("MTT_minus") else 0.)),
                "Seven-rate native and common-grid dual roles were mixed")
    try:
        cohort_contrasts([row for row in quality if not (row["cohort"] == "C-conf" and row["seed"] == 11023)], "C-conf")
    except ValueError:
        pass
    else:
        raise AssertionError("Missing C-conf seed accepted")
    try:
        primary_intervals(pd.concat([c, f], ignore_index=True), "C-conf")
    except ValueError:
        pass
    else:
        raise AssertionError("Mixed statistical families accepted")
    software_root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="cohort_software_check_", dir=software_root) as temporary:
        root = Path(temporary).resolve()
        require(root.is_relative_to(software_root), "Software fixture escaped its workspace")
        checkpoint = root/"fixture.pt"
        checkpoint.write_bytes(b"SOFTWARE FIXTURE ONLY; NOT A TRAINED MODEL")
        row = {"id": "software_fixture", "cohort": "C-conf", "checkpoint": str(checkpoint),
               "checkpoint_sha256": digest(checkpoint)}
        folder = root/"results/scoring"/row["id"]
        folder.mkdir(parents=True)
        labels = np.arange(10000) % 100

        def save_arrays(label_values):
            np.savez_compressed(folder/"test.npz", block_id=np.arange(10000), loss_sum=np.full(10000, 1.25),
                                counts=np.ones(10000, dtype=np.int64), correct=np.ones(10000, dtype=np.int64),
                                pred=label_values, labels=label_values)

        save_arrays(labels)
        marker = {"status": "complete", "row": row, "ce": 1.25, "accuracy": 1.}
        (folder/"complete.json").write_text(json.dumps(marker))
        reader = ScoreReader(root)
        observed = reader.read(row)
        require(observed["ce"] == 1.25 and observed["accuracy"] == 1., "Per-example score recomputation failed")
        save_arrays((labels+1) % 100)
        try:
            reader.read(row)
        except ValueError:
            pass
        else:
            raise AssertionError("Cross-source label mismatch accepted")
        save_arrays(labels)
        marker["ce"] = 1.5
        (folder/"complete.json").write_text(json.dumps(marker))
        try:
            reader.read(row)
        except ValueError:
            pass
        else:
            raise AssertionError("Incorrect aggregate CE accepted")
    return {"status": "passed", "scope": "pure software fixtures; not research results",
            "C_family_size": 6, "F_family_size": 3, "n_adaptation_seeds_per_start": 3,
            "J_deltaBN_minus_deltaRaw": True, "missing_seed_rejected": True,
            "mixed_families_rejected": True, "native_and_dual_roles_separated": True,
            "per_example_scores_recomputed": True, "label_mismatch_rejected": True,
            "incorrect_aggregate_rejected": True, "GPU_experiments_executed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/revision2"))
    parser.add_argument("--cohort", choices=tuple(FAMILIES), action="append")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    result = self_check() if args.self_check else run(args.root, args.cohort or tuple(FAMILIES))
    print(json.dumps(result, indent=2, allow_nan=False))
