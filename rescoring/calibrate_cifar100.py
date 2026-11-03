"""Temperature-calibrated cross-entropy of the CIFAR-100 scoring routes from the re-exported logits.

Inputs
  --logits    directory with the family subdirectories of the released logits (legacy, ext_0, ext_1, ext_2,
              native42_0), each holding one <route>.npz per route and summary_<family>.jsonl
  --recorded  directory with the recorded score tables (rescoring/recorded in the repository)
  --out       output directory

Each exported route is matched to its recorded score: the A/B, A-conf, A-cal, C-conf and F-arch cohorts by
checkpoint SHA-256, inference window, calibration kind, calibration window and calibration seed; the six replication
starts by start, source, budget, window and BN policy. The BN policy is "raw" for stored statistics and "matched" for
a bank fitted at the inference window with calibration seed 9640; routes with any other bank are not part of the
per-start comparison and are skipped. One temperature per route is selected on the validation logits by grid search
over 301 log-spaced values in [0.2, 5] minimising validation CE; the logits are divided by it and the calibrated test
CE is reported next to the recorded CE, the CE recomputed from the exported logits and both accuracies.

Outputs
  cifar100_calibrated_routes.csv   one row per route: start_id, method (S0, S2, S4, MTT), seed, K (updates),
                                   window, BN, variant, route_id, family, ce_recorded, ce_rescored, ce_diff,
                                   acc_recorded, acc_rescored, acc_diff_pp (percentage points), temperature,
                                   val_ce_raw, val_ce_cal, ce_cal
  cifar100_rescore_report.json     route counts per family, agreement with the recorded scores, temperature range,
                                   routes listed twice in the release manifests

usage: python calibrate_cifar100.py --logits DIR --recorded DIR --out DIR
"""
import argparse, collections, csv, glob, json, os, sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--logits", required=True)
ap.add_argument("--recorded", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
GRID = np.exp(np.linspace(np.log(0.2), np.log(5.0), 301))


def rd(name):
    with open(os.path.join(a.recorded, name), encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def inum(x):
    try:
        return int(float(x))
    except Exception:
        return -1


def ce_of(logits, y, T=1.0):
    z = logits.astype(np.float64) / T
    m = z.max(1, keepdims=True); lse = m[:, 0] + np.log(np.exp(z - m).sum(1))
    return float((lse - z[np.arange(len(y)), y]).mean())


def fit_T(lv, yv):
    vals = [ce_of(lv, yv, T) for T in GRID]
    i = int(np.argmin(vals)); return float(GRID[i]), vals[i]


def kind_of(r):
    if r["calibration"] == "raw":
        return "raw"
    if r["calibration"] == "bn_cross" and inum(r["cal_T"]) == inum(r["infer_T"]) and inum(r["cal_seed"]) == 9640:
        return "matched"
    return "other"


def rkey(sha, infer_T, calibration, cal_T, cal_seed):
    return (sha, inum(infer_T), calibration, inum(cal_T) if calibration != "raw" else -1, inum(cal_seed) if calibration != "raw" else -1)


# recorded scores keyed by checkpoint identity -> (start_id, method, seed, K, variant, ce, BN kind, accuracy)
rec = {}


def put(key, val):
    if key in rec and rec[key][:5] != val[:5]:
        raise SystemExit(f"conflicting record for {key}: {rec[key]} vs {val}")
    rec[key] = val


for r in rd("ab_cohorts_quality_scores.csv"):
    if r["init_id"] not in ("A", "B") or r["role"] != "primary":
        continue
    m = "S0" if r["state"] == "shared_initial" else f"S{inum(r['adapt_T'])}"
    put(rkey(r["checkpoint_sha256"], r["infer_T"], r["calibration"], r["cal_T"], r["cal_seed"]),
        (r["init_id"], m, "" if m == "S0" else str(inum(r["seed"])), 0 if m == "S0" else inum(r["updates"]), "", float(r["ce"]), kind_of(r), float(r["accuracy"])))
for r in rd("a_conf_quality_scores.csv"):
    if r["method"] == "initial":
        m, seed, K, var = "S0", "", 0, ""
    else:
        m = "MTT" if r["method"] == "mtt" else f"S{inum(r['adapt_T'])}"; seed = str(inum(r["seed"])); K = inum(r["updates"]); var = "A-conf"
    put(rkey(r["checkpoint_sha256"], r["infer_T"], r["calibration"], r["cal_T"], r["cal_seed"]), ("A", m, seed, K, var, float(r["ce"]), kind_of(r), float(r["accuracy"])))
for r in rd("c_f_cohorts_quality_scores.csv"):
    kind = kind_of(r); coh = r["cohort"]
    if coh == "A-conf-BN-selection":
        val = ("A", "MTT", str(inum(r["seed"])), inum(r["updates"]), "A-cal", float(r["ce"]), kind, float(r["accuracy"]))
    else:
        sid = {"C-conf": "C", "F-arch": "F"}[coh]
        if r["method"] == "initial":
            val = (sid, "S0", "", 0, "", float(r["ce"]), kind, float(r["accuracy"]))
        elif r["method"] == "mtt":
            val = (sid, "MTT", str(inum(r["seed"])), inum(r["updates"]), "", float(r["ce"]), kind, float(r["accuracy"]))
        else:
            # the C start has two four-step sources: the seven-rate native selection (lr 1.25e-6, the P/G contrasts)
            # and the common-grid pair at lr 5e-6 used by the MTT contrast
            var = "S4common5e-6" if (coh == "C-conf" and inum(r["adapt_T"]) == 4 and "native_expanded" not in r["roles"]) else ""
            val = (sid, f"S{inum(r['adapt_T'])}", str(inum(r["seed"])), inum(r["updates"]), var, float(r["ce"]), kind, float(r["accuracy"]))
    put(rkey(r["checkpoint_sha256"], r["infer_T"], r["calibration"], r["cal_T"], r["cal_seed"]), val)
SIX = {"C-conf_s22651": "C'1", "C-conf_s22652": "C'2", "C-conf_s22653": "C'3", "F-arch_s22751": "F'1", "F-arch_s22752": "F'2", "F-arch_s22753": "F'3"}
six_rec = {}
for r in rd("replication_starts_test_routes.csv"):
    if r["split"] != "test":
        continue
    m = {"retained": "S0", "no_mix_T2": "S2", "no_mix_T4": "S4"}[r["role"]]
    six_rec[(SIX[r["start_id"]], m, 0 if m == "S0" else inum(r["budget"]), inum(r["infer_T"]), r["BN"])] = (float(r["ce"]), "" if m == "S0" else str(inum(r["adaptation_seed"])), float(r["accuracy"]))
print("recorded keys:", len(rec), "replication-start keys:", len(six_rec))

rows = []; report = collections.Counter(); maxdiff = 0.0; unmatched = []; seen = {}; dups = []; maxacc = 0.0
for fam_dir in sorted(glob.glob(os.path.join(a.logits, "*"))):
    for sj in glob.glob(os.path.join(fam_dir, "summary_*.jsonl")):
        fam = os.path.basename(sj)[len("summary_"):-len(".jsonl")]
        for line in open(sj, encoding="utf-8"):
            s = json.loads(line); npz = os.path.join(fam_dir, s["route_id"] + ".npz")
            if not os.path.exists(npz):
                continue
            if fam in ("legacy", "ext"):
                key = rkey(s["checkpoint_sha256"], s["infer_T"], s["calibration"], s["cal_T"], s.get("cal_seed"))
                if key not in rec:
                    unmatched.append(s["route_id"]); continue
                sid, m, seed, K, var, ce_rec, kind, acc_rec = rec[key]
                if kind == "other":
                    report["skipped_other"] += 1; continue
                BN = kind; window = int(s["infer_T"])
            else:
                rid = s["record_id"]; sid = SIX[rid.split("__")[0]]
                part = rid.split("__")[1]; m = "S0" if part == "retained" else f"S{part[-1]}"
                K = int(rid.split("__k")[1]); window = int(s["window"]); BN = s["BN"]; var = ""
                ce_rec, seed, acc_rec = six_rec[(sid, m, K, window, BN)]
            k5 = (sid, m, seed, K, window, BN, var)
            if k5 in seen:
                # the same route listed under two manifest tables: keep the first listing, record the agreement
                with np.load(npz) as d:
                    dup_ce = ce_of(d["test_logits"], d["test_labels"])
                dups.append((seen[k5][0], s["route_id"], abs(dup_ce - seen[k5][1]))); continue
            with np.load(npz) as d:
                lv, yv, lt, yt = d["validation_logits"], d["validation_labels"], d["test_logits"], d["test_labels"]
            ce_new = ce_of(lt, yt); T, vce = fit_T(lv, yv); ce_cal = ce_of(lt, yt, T); seen[k5] = (s["route_id"], ce_new)
            acc = float((lt.argmax(1) == yt).mean()) * 100
            acc_rec_pct = 100 * acc_rec if acc_rec <= 1.0 else acc_rec
            maxdiff = max(maxdiff, abs(ce_new - ce_rec)); maxacc = max(maxacc, abs(acc - acc_rec_pct)); report[fam] += 1
            rows.append(dict(start_id=sid, method=m, seed=seed, K=K, window=window, BN=BN, variant=var, route_id=s["route_id"], family=fam,
                             ce_recorded=ce_rec, ce_rescored=ce_new, ce_diff=ce_new - ce_rec, acc_recorded=acc_rec_pct, acc_rescored=acc, acc_diff_pp=acc - acc_rec_pct, temperature=T,
                             val_ce_raw=ce_of(lv, yv), val_ce_cal=vce, ce_cal=ce_cal))
if unmatched:
    raise SystemExit(f"{len(unmatched)} exported routes have no recorded score: {unmatched[:5]}")
print("routes:", dict(report), "| max |CE rescored - recorded| =", f"{maxdiff:.6f}", "| max |acc diff| pp =", f"{maxacc:.3f}")
print("duplicate listings:", len(dups), "| max |CE diff| between duplicate exports =", f"{max([d[2] for d in dups], default=0.0):.2e}")
with open(os.path.join(a.out, "cifar100_calibrated_routes.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
temps = np.array([r["temperature"] for r in rows]); diffs = np.array([r["ce_diff"] for r in rows])
json.dump({"routes": len(rows), "by_family": dict(report), "max_abs_ce_diff": float(maxdiff), "max_abs_acc_diff_pp": float(maxacc), "mean_abs_ce_diff": float(np.abs(diffs).mean()),
           "temperature_range": [float(temps.min()), float(temps.max())], "grid": "301 log-spaced values in [0.2, 5], selected on the validation logits",
           "duplicate_listings": [{"kept": d[0], "duplicate": d[1], "abs_ce_diff": d[2]} for d in dups]},
          open(os.path.join(a.out, "cifar100_rescore_report.json"), "w", encoding="utf-8"), indent=1)
print("written", a.out)
