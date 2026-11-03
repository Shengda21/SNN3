# Additional datasets and replication starts

This directory holds the programs and the small result records of the four additional datasets of the manuscript (CIFAR-10, BloodMNIST64, EuroSAT RGB and PlantWild v1): data preparation with fixed splits, the construction of three independent 600-epoch SpikingResformer starts per dataset, the fixed two-step, four-step and stage-wise MTT adaptation trajectories saved at 512, 1,024 and 2,048 updates, the scoring of every source at both inference windows under stored and window-matched BN statistics, and per-board resource measurement. The retained scientific functions are paired with local filesystem and execution adapters, so the programs do not require the machines or the filesystem layout of the recorded experiments. The checkpoints and the saved validation and test logits of the recorded runs are distributed as release [`model-assets-v3`](https://github.com/Shengda21/SNN3/releases/tag/model-assets-v3).

The same programs carry a conditional-adaptation construction that was explored during the revision and is not part of the manuscript. It is kept so that every executed run of the recorded campaign remains reproducible, and it is identified as exploratory wherever it appears below.

## Recorded results and CPU checks

```sh
python extensions/validate_offline.py
python extensions/recompute.py
```

The first command checks the six profiles, their constructor and checkpoint interfaces, the original temporal-transfer tables and the names of the 720 recorded score packets. It does not run a model. `scientific_body_correspondence.json` records an AST comparison of 197 retained scientific objects, among them the optimization, averaging, scalar-fitting, bank-transport and CIFAR-100 construction functions, against their original sources.

The second command summarizes the per-route records under `results/<dataset>/`: `routes.csv` and `routes.json` list, for each of the three starts (`formal1` to `formal3`), the 52 validation and 52 test routes of the retained start, the two-step, four-step and MTT sources and the exploratory conditional source at every budget, window and BN policy, with sample count, accuracy and ordinary cross-entropy. The retained, two-step, four-step and MTT rows are the four-dataset entries of the manuscript's 22-start comparison. `gate.json` records the pre-registered validation criterion of the exploratory construction; it is false on all four datasets. This default is a scalar-record summary, not a raw-prediction audit.

To recompute cross-entropy and accuracy from prediction arrays that you generated:

```sh
python extensions/recompute.py --dataset eurosat --prediction-root /runs/eurosat/replication
```

The program reads `scores/<route>/scores.npz`. `--verify-recorded` additionally requires the recorded prediction hashes and metrics of every listed route, including the exploratory ones; the released scoring files of `model-assets-v3` cover the retained, two-step, four-step and MTT routes.

## GPU environment

The recorded GPU runs used Python 3.12, PyTorch 2.7.1 with CUDA 12.8, torchvision 0.22.1, CuPy 13.6.0, NumPy 2.1.3, Pillow 13.0.0, timm 0.9.16 and SpikingJelly 0.0.0.0.14 on RTX 5090 GPUs. Linux is required for the GPU measurement and process interface. Install the recorded environment, including CUDA-compatible wheels:

```sh
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r extensions/requirements.txt
```

The filesystem and execution adapters have CPU interface tests; they have not been exercised by a further end-to-end GPU run after packaging. A new run records its own hardware and forms its own result identity; bit equality across environments is not implied.

## Data

| Profile | Training / validation / test | Input | Construction |
|---|---|---|---|
| `cifar10` | 45000 / 5000 / 10000 | RGB32 | SpikingResformer, 600 epochs |
| `bloodmnist64` | 11959 / 1712 / 3421 | RGB64 | SpikingResformer, 600 epochs |
| `eurosat` | 18900 / 4050 / 4050 | native RGB64 | SpikingResformer, 600 epochs |
| `plantwild` | 12741 / 1787 / 3677 | full-frame RGB64 | SpikingResformer, 600 epochs |
| `cifar100_resformer` | 45000 / 5000 / 10000 | RGB32 | SpikingResformer, 600 epochs |
| `cifar100_spikformer` | 45000 / 5000 / 10000 | RGB32 | Spikformer, 30 epochs |

The two CIFAR-100 profiles serve the exploratory construction described at the end of this document; the CIFAR-100 cohorts of the manuscript were produced by the programs in `revision_round2/` and `revision_round3/`, and their states are in releases `model-assets-v1` and `model-assets-v2`.

Obtain the official CIFAR Python batches from [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html); the data directory must contain `cifar-10-batches-py` or `cifar-100-python`. CIFAR-10 uses the fixed 20261030 training/validation permutation; CIFAR-100 keeps the original 20260907 split rule.

Obtain the 64-pixel BloodMNIST archive from [MedMNIST](https://medmnist.com/), the original RGB archive from [EuroSAT](https://github.com/phelber/EuroSAT), and **v1** `plantwild.zip` from [PlantWild](https://tqwei05.github.io/PlantWild/) ([dataset repository](https://huggingface.co/datasets/uqtwei2/PlantWild)). The preparation programs check the archive identities. Raw and resized images are not redistributed; PlantWild's CC BY-NC-ND terms apply to its source data.

```sh
python extensions/prepare_data.py --dataset bloodmnist64 --archive bloodmnist_64.npz --destination /data/blood64
python extensions/prepare_data.py --dataset eurosat --archive EuroSAT.zip --destination /data/eurosat_rgb
python extensions/prepare_data.py --dataset plantwild --archive plantwild.zip --destination /data/plantwild_v1
```

EuroSAT uses a fixed class-stratified 70/15/15 assignment of exact and perceptual duplicate components; it is an in-distribution split, not a geographic generalization test. PlantWild keeps the official test set, drops the lower-priority training and validation members of fixed duplicate components and keeps the originals, so its training/validation split is derived rather than official. Its variable-size images receive EXIF correction, RGB conversion and a full-frame bilinear resize to 64 pixels; the aspect-ratio distortion is disclosed. Normalization uses training pixels only. No model result selected the grouping rule, the image size or the preprocessing.

## Replication on an additional dataset

EuroSAT, after data preparation:

```sh
python extensions/run.py init --dataset eurosat --data /data/eurosat_rgb/execution_data_fullframe64_v1 --work /runs/eurosat --gpu 0
for role in formal1 formal2 formal3; do
  python extensions/run.py construct --work /runs/eurosat --role "$role" --execute
  python extensions/run.py replicate --work /runs/eurosat --role "$role" --execute
done
python extensions/run.py freeze-test --work /runs/eurosat
for role in formal1 formal2 formal3; do
  python extensions/run.py test --work /runs/eurosat --role "$role" --execute
done
python extensions/run.py analyze --work /runs/eurosat
```

Substitute `cifar10`, `bloodmnist64` or `plantwild` and the corresponding prepared data directory; CIFAR uses the raw batch directory. These commands train the backbones; they do not read cached result tables. Each role constructs its own 600-epoch start, then runs the fixed two-step, four-step and MTT trajectories for 2,048 updates, together with the four-view trajectory of the exploratory construction, which the recorded campaign executed in the same protocol. Budgets, the selected learning rates and the seeds are in `code/<profile>/replication.json`. The three RGB64 datasets use a physical microbatch of 16 and an effective batch of 128; BN behaviour is not claimed to equal that of a physical batch of 128.

`freeze-test` requires all 39 checkpoints of a dataset and the complete route and analysis declaration; each of the 156 test routes then has a single attempt. The protocol does not reselect a method after validation or test results are seen. Failed and completed directories are kept; use a fresh work directory for a distinct run.

## Resource measurements

Every training, fitting and scoring stage records wall time, sampled board energy, peak board memory, sampling errors and gaps with the per-board measurement implementation that the recorded runs used. Invalid energy observations stay invalid; they are neither replaced by estimates nor used to rerun quality experiments. `resources/recorded_resource_summary.csv` and `resources/recorded_resource_screening_totals.csv` hold the recorded per-dataset summaries (process seconds, board energy and peak memory per phase, and the totals of the rate-screening trajectories). Board power is not whole-system power, and per-board measurements do not describe an isolated CPU host. New hardware measurements form their own records, separate from the recorded ones.

## Exploratory conditional-adaptation construction (not part of the manuscript)

During the revision a bank-covariant conditional adaptation was constructed and run under the same fixed protocol: a two-by-64 training calculation, the original construction optimizer and schedule of each architecture, a quarter-tail average, scalar fitting on 8,192 training examples and a bank-covariant deployment transform. Each conditional attempt calls four view-specific optimizers, so its committed clocks are recorded separately. The manuscript does not report this construction; its programs and records remain here so that the executed runs can be repeated.

The two CIFAR-100 profiles run it on SpikingResformer and on the documented 100-class transfer of the upstream CIFAR-10 Spikformer (not an official CIFAR-100 pretrained configuration):

```sh
python extensions/run.py init --dataset cifar100_spikformer --data /data/cifar100 --work /runs/c100-spikformer
for role in formal1 formal2 formal3; do
  python extensions/run.py construct --work /runs/c100-spikformer --role "$role" --execute
  python extensions/run.py conditional --work /runs/c100-spikformer --role "$role" --execute
done
python extensions/run.py freeze-test --work /runs/c100-spikformer
for role in formal1 formal2 formal3; do
  python extensions/run.py test --work /runs/c100-spikformer --role "$role" --execute
done
python extensions/run.py analyze --work /runs/c100-spikformer
```

Use `cifar100_resformer` for the other backbone. This entry scores the retained and the conditional checkpoints only, 12 frozen checkpoints and 48 test routes per architecture; it is not the manuscript's fixed-window comparison and carries no comparative gate. `resources/make_conditional_plan.py` writes the training and fitting packets of a conditional run for a separately measured execution, `resources/measure_plan.py` executes them with the recorded 0.5-second sampling interval, and `resources/summarize.py` aggregates the receipts:

```sh
python extensions/resources/make_conditional_plan.py --work /runs/c100-spikformer --role formal1 --output /runs/c100-spikformer/resource-plan.json
python extensions/resources/measure_plan.py --plan /runs/c100-spikformer/resource-plan.json --output /runs/c100-spikformer/resource-measurement --execute
python extensions/resources/summarize.py --receipts /runs/c100-spikformer/resource-measurement --output /runs/c100-spikformer/resources.csv
```

The measured plan is a train-and-fit-only execution, not an extra repeat after the full `conditional` command. The retained R2 model, runtime, calibration and inference-timing code is included under each CIFAR-100 `constructor/scripts/`.
