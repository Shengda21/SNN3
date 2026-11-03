"""Export the validation and test logits of every CIFAR-100 scoring route of the released checkpoint collections.

The volumes of release model-assets-v2 carry their own loaders; this program drives them and writes one .npz per
route. Three loader families exist, and each runs in its own process because their module namespaces differ:

  legacy    the A and B cohorts (legacy112 tree): replay_quality.py and r2_vision.load_model
  ext       the C-conf, F-arch, A-conf and A-cal cohorts (extension352 tree): runtime.load_model and
            normalization.apply_bank
  native42  the six replication starts C'1-3 and F'1-3 (native42 volume): native_loader.load_deployment

Every route is evaluated in FP32 with TF32 disabled. Where the route uses a BN bank, the saved bank is applied; no
statistics are recomputed. The inputs are the 10,000 official CIFAR-100 test images and the 5,000 validation images
listed in CIFAR100_split.json. The .npz holds float32 logits, labels, example ids, per-example cross-entropy and
predictions for both splits. One line per route is appended to summary_<family>.jsonl with the test and validation CE
and accuracy and, where the manifest records them, the differences to the recorded test scores (ce_error, acc_error).
Routes whose BN bank was fitted at a window other than the inference window are skipped: they are not part of the
per-start comparison. A route whose .npz already exists is skipped, so an interrupted run resumes.

usage: python export_logits.py legacy   <legacy112_dir>    <out_dir> [--shard i/n] [--batch 64]
       python export_logits.py ext      <extension352_dir> <out_dir> [--shard i/n] [--batch 64]
       python export_logits.py native42 <native42_dir>     <out_dir> --cifar-root <dir> [--shard i/n] [--batch 64]

<legacy112_dir> and <extension352_dir> are the directories of those names after the four public_part_*.zip shards are
extracted into one tree, each with its manifest.json and loader modules and with the CIFAR-100 inputs staged under
inputs/ as verify_public.py --require-inputs expects. <native42_dir> is the directory into which the native42_part_*.zip
shards were extracted (it contains native42/source, native42/models and native42/banks). --cifar-root is a directory
with the extracted CIFAR-100 Python archive and CIFAR100_split.json; the first two families take their inputs from
their own runtime trees. --shard i/n assigns every n-th manifest row to this process so that several processes share
one GPU; the recorded run used three ext shards and batch 64 on one RTX 5090.
"""
import argparse, json, os, sys, time
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("family", choices=["legacy", "ext", "native42"])
ap.add_argument("release_root", type=Path)
ap.add_argument("out_dir", type=Path)
ap.add_argument("--shard", default="0/1")
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--cifar-root", type=Path, default=None)
a = ap.parse_args()
shard_i, shard_n = (int(x) for x in a.shard.split("/"))
release = a.release_root.resolve(); out = a.out_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
log = open(out / f"export_{a.family}_{shard_i}of{shard_n}.log", "a", encoding="utf-8")


def say(*x):
    print(time.strftime("%H:%M:%S"), *x, file=log, flush=True); print(*x, flush=True)


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


import numpy as np
import torch
from torch.nn import functional as F
from torchvision import datasets
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
assert torch.cuda.is_available()


def load_data(data_root, transform, split_json):
    test = datasets.CIFAR100(str(data_root), train=False, transform=transform)
    train = datasets.CIFAR100(str(data_root), train=True, transform=transform)
    val_ids = np.asarray(read(split_json)["validation"], dtype=np.int64)
    xt = torch.stack([test[i][0] for i in range(len(test))]); yt = torch.tensor(test.targets, dtype=torch.long)
    xv = torch.stack([train[int(i)][0] for i in val_ids]); yv = torch.tensor([train.targets[int(i)] for i in val_ids], dtype=torch.long)
    assert len(xt) == 10000 and len(xv) == 5000
    return {"test": (xt.cuda(), yt.cuda(), np.arange(10000)), "validation": (xv.cuda(), yv.cuda(), val_ids)}


@torch.no_grad()
def evaluate(forward, x, y, batch):
    outs = []
    for s in range(0, len(x), batch):
        z = forward(x[s:s + batch])
        if not torch.isfinite(z).all():
            raise FloatingPointError("nonfinite logits")
        outs.append(z.float())
    z = torch.cat(outs)
    ce = F.cross_entropy(z, y, reduction="none")
    return z.cpu().numpy().astype(np.float32), ce.double().cpu().numpy(), z.argmax(-1).cpu().numpy()


def run_route(route_id, forward, data, meta, batch):
    f = out / f"{route_id}.npz"
    if f.exists():
        return None
    rec = {"route_id": route_id, **meta}
    arrays = {}
    for split, (x, y, ids) in data.items():
        t0 = time.time()
        logits, ce, pred = evaluate(forward, x, y, batch)
        yn = y.cpu().numpy()
        arrays[f"{split}_logits"] = logits; arrays[f"{split}_labels"] = yn; arrays[f"{split}_ids"] = ids
        arrays[f"{split}_ce"] = ce; arrays[f"{split}_pred"] = pred
        rec[f"{split}_ce"] = float(ce.mean()); rec[f"{split}_acc"] = float((pred == yn).mean()); rec[f"{split}_seconds"] = round(time.time() - t0, 1)
    tmp = f.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays); os.replace(tmp, f)
    if meta.get("expected_ce") is not None:
        rec["ce_error"] = rec["test_ce"] - meta["expected_ce"]; rec["acc_error"] = rec["test_acc"] - meta["expected_accuracy"]
    with open(out / f"summary_{a.family}.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
    say(json.dumps({k: rec[k] for k in rec if k in ("route_id", "test_ce", "test_acc", "validation_ce", "validation_acc", "ce_error", "acc_error", "test_seconds")}))
    return rec


def my_rows(rows, prio=lambda r: 0):
    """Rows are assigned to shards by their position in the manifest, so the assignment is stable across restarts.
    Within a shard, the retained and final-budget routes come first."""
    mine = [r for i, r in enumerate(rows) if i % shard_n == shard_i]
    return sorted(mine, key=prio)


def crossed(calibration, cal_T, infer_T):
    return calibration != "raw" and cal_T is not None and int(cal_T) != int(infer_T)


if a.family == "legacy":
    sys.path.insert(0, str(release))
    import replay_quality as rq
    manifest = read(release / "manifest.json")
    work = out / "runtime_legacy"
    rq.materialize_runtime(release, work, manifest)
    os.environ.update(R2_ARCHIVE=str(work / "archive"), R2_WORK=str(work / "r3"), R3_LEGACY_WORK=str(work / "unused_legacy"),
                      OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    sys.path.insert(0, str(work / "r3/code"))
    import r2_vision as rv
    rv.configure()
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    vision = work / "archive/direction1_budget_rotation_20260907"
    data = load_data(vision / "data", rv.evaluate_transform(), vision / "data/CIFAR100_split.json")
    split = read(vision / "data/CIFAR100_split.json")
    assets = {r["path"]: r for r in manifest["assets"]}
    for row in my_rows(manifest["rows"], prio=lambda r: 0 if int(r["updates"] or 0) in (0, 2048) else 1):
        route_id = row["row_id"]
        if (out / f"{route_id}.npz").exists() or crossed(row["calibration"], row["cal_T"], row["infer_T"]):
            continue
        e = row["infer_T"]; c = row["cal_T"] if row["calibration"] != "raw" else e
        cp = rq.bounded(release, row["checkpoint"])
        m, sites, _, _ = rv.load_model(c, cp, False)
        if row["calibration"] != "raw":
            bank = rq.require_bank(row, release, assets)
            saved = torch.load(bank, map_location="cpu", weights_only=False)
            ids = np.random.default_rng(row["cal_seed"]).choice(np.asarray(split["train"]), 2048, replace=False)
            assert saved["checkpoint_sha256"] == row["checkpoint_sha256"] and saved["cal_T"] == row["cal_T"] and np.array_equal(saved["train_ids"], ids)
            current = m.state_dict()
            for name, tensor in saved["bn_delta"].items():
                assert name.endswith((".running_mean", ".running_var", ".num_batches_tracked")) and current[name].dtype == tensor.dtype
                current[name].copy_(tensor)
            del saved
        m.T = e; rv.v.functional.reset_net(m); rv.v.clear_costs(sites); rv.v.prepare_inference(m)
        m = m.cuda()

        def forward(x, m=m, sites=sites):
            rv.v.functional.reset_net(m); rv.v.clear_costs(sites)
            return m(x).mean(0).float()
        meta = {k: row.get(k) for k in ("cohort", "starting_state", "source_id", "adaptation_seed", "adapt_T", "updates", "infer_T",
                                        "calibration", "cal_T", "cal_seed", "method", "checkpoint_sha256", "expected_ce", "expected_accuracy", "table")}
        run_route(route_id, forward, data, meta, a.batch)
        del m; torch.cuda.empty_cache()

elif a.family == "ext":
    sys.path.insert(0, str(release))
    import replay_extension352 as rx
    from extension_common import bounded, read as xread
    manifest = xread(release / "manifest.json")
    work = out / "runtime_ext"
    if not (work / "runtime").exists():
        rx.materialize(release, work, manifest)
    runtime_root = work / "runtime"
    sys.path.insert(0, str(release / "original_bundle/scripts"))
    import runtime as rt
    import normalization
    rt.configure()
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    data_root = rt.vision_root(runtime_root) / "data"
    data = load_data(data_root, rt.evaluate_transform(), data_root / "CIFAR100_split.json")
    split = xread(data_root / "CIFAR100_split.json")
    train_ids = np.random.default_rng(9640).choice(np.asarray(split["train"]), 2048, replace=False)
    for row in my_rows(manifest["rows"], prio=lambda r: 0 if int(r["updates"] or 0) in (0, 2048) else 1):
        route_id = row["id"]
        if (out / f"{route_id}.npz").exists() or crossed(row["calibration"], row["cal_T"], row["infer_T"]):
            continue
        cal_window = row["cal_T"] if row["calibration"] != "raw" else row["infer_T"]
        model = rt.load_model(runtime_root, row["architecture"], bounded(release, row["checkpoint"]), cal_window)
        if row["calibration"] != "raw":
            saved = torch.load(bounded(release, row["bn_bank"]), map_location="cpu", weights_only=False)
            assert saved.get("checkpoint_sha256") == row["checkpoint_sha256"] and saved.get("cal_T") == row["cal_T"] and row["cal_seed"] == 9640
            assert np.array_equal(np.asarray(saved.get("train_ids")), train_ids)
            rep = normalization.apply_bank(model, saved["bn_delta"]); assert rep.get("loaded_saved_statistics") is True
            rt.set_window(model, row["infer_T"]); rt.prepare_mode(model, False)
            del saved
        model = model.cuda()

        def forward(x, model=model):
            return rt.forward(model, x)
        meta = {k: row.get(k) for k in ("cohort", "architecture", "method", "calibration", "checkpoint_sha256", "seed", "adapt_T", "updates",
                                        "infer_T", "cal_T", "cal_seed", "expected_ce", "expected_accuracy")}
        run_route(route_id, forward, data, meta, a.batch)
        del model; torch.cuda.empty_cache()

else:
    assert a.cifar_root is not None, "--cifar-root is required for the native42 family"
    src = release / "native42/source"
    sys.path.insert(0, str(src))
    from native_loader import load_deployment, runtime
    plan = read(src / "BRIDGE_PLAN.json")
    rt0 = runtime()
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    data = load_data(a.cifar_root, rt0.evaluate_transform(), a.cifar_root / "CIFAR100_split.json")
    routes = [(rec, v) for rec in plan["records"] for v in rec["views"]]
    for rec, view in my_rows(routes, prio=lambda rv: 0 if rv[0]["id"].endswith(("__k0", "__k2048")) else 1):
        route_id = f"{rec['id']}__e{view['window']}_{view['BN']}"
        if (out / f"{route_id}.npz").exists():
            continue
        cp = release / "native42/models" / f"{rec['portable_CP']['sha256']}.pt"
        bank = release / "native42/banks" / f"{view['portable_bank']['sha256']}.pt"
        model, rt = load_deployment(rec["architecture"], cp, view["window"], expected_sha256=rec["portable_CP"]["sha256"],
                                    bank_path=bank, bank_sha256=view["portable_bank"]["sha256"], bank_container=None)
        model = model.cuda()

        def forward(x, model=model, rt=rt):
            z = rt.forward(model, x); rt.reset(model); return z
        meta = {"record_id": rec["id"], "architecture": rec["architecture"], "window": view["window"], "BN": view["BN"],
                "checkpoint_sha256": rec["portable_CP"]["sha256"], "bank_sha256": view["portable_bank"]["sha256"],
                "original_checkpoint_path": rec["reference_CP"]["path"], "expected_ce": None, "expected_accuracy": None}
        run_route(route_id, forward, data, meta, a.batch)
        del model; torch.cuda.empty_cache()
say(f"=== DONE {a.family} shard {shard_i}/{shard_n}")
