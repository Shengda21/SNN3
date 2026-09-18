# SNN3: finite-budget temporal adaptation

Code and recorded experimental outputs for comparing a shared starting checkpoint, a checkpoint adapted at two timesteps, and a checkpoint adapted at four timesteps. Each candidate is evaluated at both inference windows. The vision experiments also examine recalibration of batch-normalization statistics while keeping model weights fixed.

## Recompute the results

Python 3.12 is recommended. From the repository root:

```sh
python -m pip install -r requirements-analysis.txt
python reproduce_revision3.py --figures
python reproduce_all.py --figures
python analyze_review_statistics.py
```

These commands run on a CPU from the included per-item outputs. They do not download data, load model weights, or run new training. Generated files are written under `recomputed/`; the additional retained-cohort statistics are written under `review_analysis/`.

The numerical figure commands export PDF, SVG and PNG without requiring LaTeX. Optional PGF export requires a working LaTeX installation and the environment variable `SNN3_EXPORT_PGF=1`. The standalone current workflow diagram, `figure_source/make_workflow_revision3.py`, also requires LaTeX. On Windows, use a short checkout directory and enable Git long-path support if needed.

The current vision analysis recomputes 506 scoring paths and checks all 11 statistical CSV files against the recorded summaries. Its figure command regenerates three figures. The earlier observations have a separate entry point and retain their original scoring and timing conditions.

| Evidence | Directory | Entry point |
|---|---|---|
| Current vision experiment and fixed-checkpoint runtime replay (R3) | `revision_round3/` | `reproduce_revision3.py --figures` |
| Retained cross-window evaluation and timing (R2.1) | `revision_round2/` | `reproduce_all.py --figures` |
| Earlier language trajectories and execution measurements | `experiment/orthogonal_pilot_20260907/` | `reproduce_all.py --figures` |
| Figure programs and reference numerical inputs | `figure_source/` | The commands above |

R3 contains 380 new vision scoring paths and 126 runtime replays of earlier fixed checkpoints. Initializations A and B use five and three paired adaptation seeds, respectively. Calibration subsets are sensitivity conditions, not additional training seeds. The retained language diagnostic remains a separate cohort. Comparing a replay with an earlier runtime does not constitute another independent test set.

`python check_revision3_records.py` additionally checks the correspondence between the scoring queue, source identities and saved outputs. It recomputes loss and accuracy from the arrays. These checks validate the saved evidence; they do not perform a second GPU replay.

## Files and protocols

Within each revision directory, `code/` contains execution and analysis programs, `protocol/` records the experimental configuration, and `results/scoring/` contains per-item arrays and scoring metadata. `analysis/` contains reference statistical tables. Training and timing records preserve the individual observations used in the analysis.

The original directory names and historical server paths in the experiment records identify their provenance. They are not download links or paths that must exist to run the CPU analyses. The supplied replay preparation programs relocate the recorded paths when the required model and input assets are available.

See [DATA.md](DATA.md) for dataset sources and the boundary between the public results and model execution. See [THIRD_PARTY.md](THIRD_PARTY.md) for upstream software.

## Model execution

The execution programs are included, but the large model/input companion archives are **not distributed in this repository**. The repository alone supports numerical and figure recomputation, not a complete checkpoint replay. No model weights or raw corpora need to be downloaded for the CPU commands above.

For readers who already have the companion materials, `replay_revision3.py`, `replay_recorded.py`, and `prepare_revision_rerun.py` provide preparation and execution entry points. Their command-line help describes the required directories. The asset inventories are `revision3_assets.json` and `revision_round2/protocol/`; the verification programs check supplied archives or extracted folders. The R3 companion name used by these tools is `revision3_model_assets.zip`; the earlier companion is `model_data_assets.zip`. Their presence in a script is not a claim that they are hosted here.

R3 was run with PyTorch 2.7.1, CUDA 12.8 and CuPy 13.6.0 on an RTX 5090. The earlier cohort used PyTorch 2.6.0 and CUDA 12.4. Environment records accompany each cohort. New GPU runs can differ numerically across runtimes and must retain their own result identity.

This repository contains experimental materials. Manuscript files, author biographies and photographs are not included.
