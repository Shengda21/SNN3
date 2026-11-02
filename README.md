# SNN3: what finite-budget adaptation buys across two inference windows

Code, recorded experimental outputs and released model checkpoints for the manuscript *One checkpoint, two inference windows: what finite-budget adaptation buys in spiking Transformers*. The experiments compare, at a fixed inference window, a retained four-step starting checkpoint, a checkpoint adapted at two timesteps, a checkpoint adapted at four timesteps and a stage-wise Mixed Time-step Training (MTT) checkpoint, with and without recomputed batch-normalization statistics. The study covers SpikingResformer and Spikformer on CIFAR-100 with 30- and 600-epoch starts, and 22 independently constructed starts on five visual datasets (CIFAR-100, CIFAR-10, BloodMNIST64, EuroSAT RGB, PlantWild).

## Model checkpoints

Checkpoints are distributed as GitHub Releases of this repository. Each release lists the archives with their byte counts and SHA-256 values; the verification programs in this repository check the archives member by member.

| Release | Contents | Verification |
|---|---|---|
| [`model-assets-v1`](https://github.com/Shengda21/SNN3/releases/tag/model-assets-v1) | `revision3_model_assets.zip`: the R3 companion (A/B-cohort scoring checkpoints, shared initialization, BN deltas, calibration subsets; 261 files). `vision_model_assets_r2_1.zip`: the vision subset of the earlier R2.1 companion (20 adapted checkpoints and the shared T4 initialization). | `python verify_revision3_assets.py --archive revision3_model_assets.zip`; `assets_vision_subset.json` inside the second archive lists its members. |
| `model-assets-v2` | The 464-route CIFAR-100 collection (A/B cohorts, C-conf, F-arch, A-cal: 92 checkpoints and 184 BN banks in `public_part_*.zip`, indexed by `public_index.json`) and the six independent replication starts C'1--3 and F'1--3 (42 checkpoints and 126 BN banks in `native42_part_*.zip`, indexed by `native42_index.json`). | `verify_public.py` (inside the shards) and `verify_native42.py` (release asset) check shard hashes and extracted files. |
| [`model-assets-v3`](https://github.com/Shengda21/SNN3/releases/tag/model-assets-v3) | The four additional datasets, one archive each (`fourds_<dataset>.zip`): three retained 600-epoch starts, 18 separately adapted and 9 MTT states, the saved validation and test logits of every retained, native and MTT scoring route, and the scoring and selection records; indexed by `fourds_index.json`. | `verify_fourds.py` (release asset) checks archive hashes and extracted files. |

The language-model weights of the earlier companion are derived from a third-party model and are not redistributed. No dataset images or corpora are included in any release; see [DATA.md](DATA.md).

## Recompute the results

Python 3.12 is recommended. From the repository root:

```sh
python -m pip install -r requirements-analysis.txt
python reproduce_revision3.py --figures
python reproduce_all.py --figures
python analyze_review_statistics.py
python extensions/recompute.py
```

These commands run on a CPU from the included per-item outputs. They do not download data, load model weights, or run new training. Generated files are written under `recomputed/`; the retained-cohort statistics are written under `review_analysis/`. `python extensions/validate_offline.py` performs the CPU interface checks of the extension programs, and `python check_revision3_records.py` checks the correspondence between the scoring queue, source identities and saved outputs.

The numerical figure commands export PDF, SVG and PNG without requiring LaTeX. Optional PGF export requires a working LaTeX installation and the environment variable `SNN3_EXPORT_PGF=1`. The standalone workflow diagram, `figure_source/make_workflow_revision3.py`, also requires LaTeX. On Windows, use a short checkout directory and enable Git long-path support if needed.

| Evidence | Directory | Entry point |
|---|---|---|
| CIFAR-100 A/B cohorts and fixed-checkpoint runtime replay (R3) | `revision_round3/` | `reproduce_revision3.py --figures` |
| Retained cross-window evaluation and timing (R2.1) | `revision_round2/` | `reproduce_all.py --figures` |
| Earlier language trajectories and execution measurements | `experiment/orthogonal_pilot_20260907/` | `reproduce_all.py --figures` |
| Additional datasets, replication starts and MTT (construction, adaptation, scoring, resource programs; per-start result records; saved logits are in release `model-assets-v3`) | `extensions/` | `extensions/recompute.py`; see [extensions/README.md](extensions/README.md) |
| Figure programs and reference numerical inputs | `figure_source/` | The commands above |

R3 contains 380 vision scoring paths and 126 runtime replays of earlier fixed checkpoints. Initializations A and B use five and three paired adaptation seeds. Calibration subsets are sensitivity conditions, not additional training seeds. The language diagnostic is a separate cohort.

## Additional datasets and replication starts

[extensions/README.md](extensions/README.md) documents the construction, adaptation, MTT, scoring and resource-measurement programs for CIFAR-10, BloodMNIST64, EuroSAT RGB and PlantWild v1, together with the small per-start result records. Each dataset has three independent 600-epoch SpikingResformer constructions; every retained, two-step, four-step and MTT source is scored at 512, 1,024 and 2,048 updates, at both inference windows and under both BN policies, with validation and test logits saved. The same directory contains the programs of a conditional-adaptation construction that was explored during the revision; that construction is not part of the manuscript, and its result records are retained only as a complete account of the executed runs.

## Files and protocols

Within each revision directory, `code/` contains execution and analysis programs, `protocol/` records the experimental configuration, and `results/scoring/` contains per-item arrays and scoring metadata. `analysis/` contains reference statistical tables. Training and timing records preserve the individual observations used in the analysis.

The original directory names and historical server paths in the experiment records identify their provenance. They are not download links or paths that must exist to run the CPU analyses. The replay preparation programs relocate the recorded paths when the released model assets are extracted.

See [DATA.md](DATA.md) for dataset sources and the boundary between the public results, the released checkpoints and model execution. See [THIRD_PARTY.md](THIRD_PARTY.md) for upstream software.

## Model execution

Fixed-checkpoint replay needs the released archives above and the original datasets. `replay_revision3.py`, `replay_recorded.py` and `prepare_revision_rerun.py` provide preparation and execution entry points; their command-line help describes the required directories. The asset inventories are `revision3_assets.json` and `revision_round2/protocol/`. The 464-route collection carries its own loaders (`legacy112/replay_quality.py`, `extension352/replay_extension352.py`) and bootstrap files inside the shards; the replication-start volume carries the frozen native loader under `native42/source/`.

R3, the extensions and the replication starts were run with PyTorch 2.7.1, CUDA 12.8 and CuPy 13.6.0 on RTX 5090 GPUs. The earlier cohort used PyTorch 2.6.0 and CUDA 12.4. Environment records accompany each cohort. New GPU runs can differ numerically across runtimes and keep their own result identity.

This repository contains experimental materials. Manuscript files, author biographies and photographs are not included.

## Licence

The program code is released under the MIT licence (`LICENSE`). The experimental outputs in this repository and the model checkpoints and BN statistic banks in the Releases are released under CC BY-NC 4.0 (`LICENSE-DATA.md`). Copyright (c) 2026 Shengda Liu and Tingting Su. Third-party components keep their own licences (`THIRD_PARTY.md`); datasets are not redistributed.
