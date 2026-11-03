# CIFAR-100 logit re-export and temperature calibration

The CIFAR-100 scoring archives of this project saved per-example losses, predictions and labels but not logits, so the temperature-calibrated cross-entropy of the manuscript could at first be computed only for the four additional datasets, whose logits were saved at scoring time. This directory holds the program that re-exported the validation and test logits of every CIFAR-100 scoring route from the released checkpoints and BN banks, the per-route temperatures and calibrated losses derived from them, and the agreement between the re-exported and the recorded scores. The logits themselves are distributed as release [`model-assets-v4`](https://github.com/Shengda21/SNN3/releases/tag/model-assets-v4).

## Files

| File | Contents |
|---|---|
| `export_logits.py` | Drives the loaders of release `model-assets-v2` (the legacy112, extension352 and native42 volumes) and writes one `.npz` per scoring route with the float32 validation and test logits, labels, example ids, per-example cross-entropy and predictions, plus a `summary_<family>.jsonl` line with the test and validation CE and accuracy and the difference to the recorded test score. One process per loader family; see the module docstring for the directory layout it expects. |
| `calibrate_cifar100.py` | Matches every exported route to its recorded score, recomputes the test CE from the logits, selects one temperature per route on the validation logits and writes the calibrated test CE. |
| `recorded/` | The columns of the recorded score tables that the matching needs: `ab_cohorts_quality_scores.csv` (A and B cohorts, 506 rows, from `revision_round3/analysis/quality_scores.csv`), `a_conf_quality_scores.csv` (A-conf cohort, 132 rows), `c_f_cohorts_quality_scores.csv` (C-conf, F-arch and A-cal cohorts, 352 rows) and `replication_starts_test_routes.csv` (the retained, two-step and four-step sources of the six replication starts, 168 test routes). Each row carries the checkpoint SHA-256 or route identity, the inference window, the BN policy, the recorded CE and accuracy. |
| `cifar100_calibrated_routes.csv` | One row per matched route (536 rows): `start_id` (A, B, C, F, C'1--3, F'1--3), `method` (S0 retained, S2 and S4 separately adapted at two and four steps, MTT), `seed`, `K` (updates), `window`, `BN` (`raw` or `matched`), `variant` (empty, `A-conf`, `A-cal` or `S4common5e-6`), `route_id`, `family`, `ce_recorded`, `ce_rescored`, `ce_diff`, `acc_recorded`, `acc_rescored`, `acc_diff_pp`, `temperature`, `val_ce_raw`, `val_ce_cal`, `ce_cal`. |
| `cifar100_rescore_report.json` | Route counts per loader family, the maximum and mean absolute difference between the re-exported and the recorded test CE, the maximum accuracy difference, the temperature range and the routes listed twice in the release manifests. |
| `cifar100_logits_index.json` | Index of the release: byte count and SHA-256 of every file in the two archives, and for every `.npz` its route identity and whether it enters the calibrated table. |
| `verify_cifar100_logits.py` | Checks the downloaded archives, or the extracted files, against the index. |

## Recorded run

The export ran on 2026-11-03 on one RTX 5090 (driver 580.76.05) with Python 3.12.3, PyTorch 2.7.1+cu128, torchvision 0.22.1, CuPy 13.6.0 and SpikingJelly 0.0.0.0.14, in FP32 with TF32 disabled for matrix multiplication and cuDNN, batch 64, with the saved BN banks applied where a route uses one. The Spikformer routes use the CuPy backend of SpikingJelly's multi-step LIF neuron; on this GPU that backend needs an NVRTC able to compile for compute capability 12.0 (the NVRTC of the CUDA 12.8 pip wheel), and its forward pass was checked to agree bit for bit with the PyTorch backend before any route was scored.

548 routes were exported: 112 of the A and B cohorts, 268 of the A-conf, A-cal, C-conf and F-arch cohorts and 168 of the six replication starts (42 checkpoints at both windows and under both BN policies). 536 of them enter the per-start comparison of the manuscript; the other twelve are eight routes whose BN bank was fitted at the other window and four routes that the release manifest lists under two tables and that duplicate a route already in the table.

Agreement with the recorded scores over the 536 routes: the re-exported test CE differs from the recorded value by at most 0.0025 nat (mean absolute difference 0.0003) and the accuracy by at most 0.21 percentage points; the recorded scoring used batch 4 and partly CUDA Graphs. In the manuscript the recorded values remain the raw columns, and the re-exported logits supply only the temperature and the calibrated loss.

Calibration follows the definition of the manuscript: one temperature per route is selected on the validation logits by grid search over 301 log-spaced values in [0.2, 5] minimising validation CE, the test logits are divided by it, and the CE of the result is the calibrated loss. Accuracy is unchanged by this operation. The selected temperatures lie between 0.53 and 1.03 for the routes at 0 and 2,048 updates, above one only on four Spikformer replication routes.

## Reproduce

Verify and extract the release, then recompute the calibrated table from the extracted logits (CPU, a few minutes):

```sh
python rescoring/verify_cifar100_logits.py --index rescoring/cifar100_logits_index.json --shards DOWNLOAD_DIR
unzip DOWNLOAD_DIR/cifar100_logits_part_000.zip -d EXTRACTED
unzip DOWNLOAD_DIR/cifar100_logits_part_001.zip -d EXTRACTED
python rescoring/verify_cifar100_logits.py --index rescoring/cifar100_logits_index.json --root EXTRACTED
python rescoring/calibrate_cifar100.py --logits EXTRACTED/cifar100_logits --recorded rescoring/recorded --out OUT
```

`OUT/cifar100_calibrated_routes.csv` and `OUT/cifar100_rescore_report.json` then equal the files in this directory.

To re-export the logits from the checkpoints, extract the four `public_part_*.zip` shards of `model-assets-v2` into one tree, stage the CIFAR-100 inputs under `legacy112/inputs/` and `extension352/inputs/` as `verify_public.py --require-inputs` expects, extract the two `native42_part_*.zip` shards into a second directory, and run one process per family:

```sh
python rescoring/export_logits.py legacy   TREE/legacy112    OUT/legacy
python rescoring/export_logits.py ext      TREE/extension352 OUT/ext_0 --shard 0/3
python rescoring/export_logits.py native42 NATIVE42_DIR      OUT/native42_0 --cifar-root CIFAR_DIR
```

A new GPU run reproduces the recorded logits only up to the run-to-run variation of the kernels; the `summary_<family>.jsonl` written by the export states the agreement of every route with the recorded test score.

## Release archives

| Archive | Bytes | SHA-256 | Contents |
|---|---|---|---|
| `cifar100_logits_part_000.zip` | 1,540,897,815 | `1704d13d7e1217ed86a790fa6780af7606d3b1d4887e17e81a204f49a54a1dd0` | legacy, ext_0, ext_1: 277 files |
| `cifar100_logits_part_001.zip` | 1,556,849,055 | `8f2c963a5df03b9d09cb1dac90cc3bfea939487354c53fe4e2fb6508bc455520` | ext_2, native42_0: 281 files |

Licence: the logits and the tables are released under CC BY-NC 4.0 (`LICENSE-DATA.md`), the programs under MIT (`LICENSE`).
