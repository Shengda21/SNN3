# Additional datasets and bank-covariant adaptation

This directory provides construction, training, fitting, deployment and evaluation code. The retained scientific functions are paired with local filesystem and execution-identity adapters. They do not require the machines or filesystem layout used for the recorded experiments.

## Recorded results and CPU checks

```sh
python extensions/validate_offline.py
python extensions/recompute.py
```

The first command checks six profiles, their constructor/checkpoint interfaces, the original temporal-transfer tables and 720 actual score-packet names. It does not run a model. `scientific_body_correspondence.json` records an AST comparison of 197 retained scientific objects, including optimization, averaging, scalar fitting, bank transport and the original CIFAR-100 construction functions.

The second command summarizes the complete, small per-route records supplied under `results/`: three starts, all methods and budgets, both windows and both BN policies. This default is **scalar-record recomputation**, not a raw-prediction audit. The four added-dataset experiments each have 156 validation and 156 test routes. Their original all-condition validation criteria remain false; all gains and losses are reported.

To independently recompute CE and accuracy from predictions you generated:

```sh
python extensions/recompute.py --dataset eurosat --prediction-root /runs/eurosat/replication
```

The program reads `scores/<route>/scores.npz`. `--verify-recorded` additionally requires the exact recorded prediction hashes and metrics; it applies only when the original prediction companion is available. Raw prediction archives and full checkpoints are not included in this code update.

## GPU environment

The recorded GPU implementation used Python 3.12, PyTorch 2.7.1 with CUDA 12.8, torchvision 0.22.1, CuPy 13.6.0, NumPy 2.1.3, Pillow 13.0.0, timm 0.9.16 and SpikingJelly 0.0.0.0.14 on RTX 5090 GPUs. Linux is required for the GPU measurement/process interface. Install the recorded environment, including CUDA-compatible wheels:

```sh
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r extensions/requirements.txt
```

The public path adapters have CPU interface tests. They have not undergone an additional end-to-end GPU rerun after packaging. New runs record their actual hardware and create a new result identity; cross-environment bit equality is not implied.

## Data

| Profile | Training / validation / test | Input | Construction |
|---|---|---|---|
| `cifar10` | 45000 / 5000 / 10000 | RGB32 | Resformer, 600 epochs |
| `bloodmnist64` | 11959 / 1712 / 3421 | RGB64 | Resformer, 600 epochs |
| `eurosat` | 18900 / 4050 / 4050 | native RGB64 | Resformer, 600 epochs |
| `plantwild` | 12741 / 1787 / 3677 | full-frame RGB64 | Resformer, 600 epochs |
| `cifar100_resformer` | 45000 / 5000 / 10000 | RGB32 | Resformer, 600 epochs |
| `cifar100_spikformer` | 45000 / 5000 / 10000 | RGB32 | Spikformer, 30 epochs |

Obtain official CIFAR Python batches from [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html). The supplied data directory must contain `cifar-10-batches-py` or `cifar-100-python`. CIFAR-10 uses the fixed 20261030 training/validation permutation; CIFAR-100 retains the original 20260907 split rule.

Obtain the 64-pixel BloodMNIST archive from [MedMNIST](https://medmnist.com/), the original RGB archive from [EuroSAT](https://github.com/phelber/EuroSAT), and **v1** `plantwild.zip` from [PlantWild](https://tqwei05.github.io/PlantWild/) ([dataset repository](https://huggingface.co/datasets/uqtwei2/PlantWild)). Archive identities are checked by the preparation programs. Raw images are not redistributed. PlantWild's CC BY-NC-ND terms apply to its source data; this repository does not distribute resized PlantWild images.

```sh
python extensions/prepare_data.py --dataset bloodmnist64 --archive bloodmnist_64.npz --destination /data/blood64
python extensions/prepare_data.py --dataset eurosat --archive EuroSAT.zip --destination /data/eurosat_rgb
python extensions/prepare_data.py --dataset plantwild --archive plantwild.zip --destination /data/plantwild_v1
```

EuroSAT uses fixed class-stratified 70/15/15 assignment of exact/perceptual connected components; it is an in-distribution split, not a geographic generalization test. PlantWild retains the official test set, excludes lower-priority training/validation members of fixed duplicate components, and retains the originals. Its resulting train/validation split is derived, not an unchanged official split. Its variable-size images undergo EXIF correction, RGB conversion and full-frame bilinear resize to 64, with aspect-ratio distortion disclosed. Normalization uses training pixels only. No model result selects the grouping rule, image size or preprocessing.

## Full added-dataset replication

For example, initialize EuroSAT after data preparation:

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

Substitute `cifar10`, `bloodmnist64` or `plantwild` and the corresponding prepared data directory. CIFAR uses the raw batch directory. These commands train actual backbones and adapters; they do not read cached result tables. Every role starts from its own constructed checkpoint, then runs the fixed native2/native4/MTT trajectories and the four-view conditional trajectory for 2048 logical attempts. Each conditional attempt calls the four view-specific optimizers; actual committed clocks are recorded separately. Budgets 512/1024/2048, selected learning rates and seeds are in `code/<profile>/replication.json`. The conditional route includes the original training-only 8192-example scalar fitting and bank-covariant deployment transform. The three RGB64 datasets use physical microbatch 16 and effective batch 128; BN behavior is not claimed equivalent to batch 128.

Testing requires all 39 checkpoints and the full route/analysis declaration to be frozen. Each of the 156 routes has a single exclusive attempt. The fixed protocol does not reselect a method after seeing validation or test results. Failed and completed directories are retained; use a fresh work directory for a distinct run.

## CIFAR-100 core method and both architectures

The two CIFAR-100 profiles expose the complete bank-covariant conditional method with the original two-by-64 training calculation, original construction optimizer/schedule for each architecture, quarter-tail average, augmented scalar fitting and fixed covariant transport. They share the retained scientific primitives rather than importing private experiment directories. The Spikformer is the documented 100-class transfer of the upstream CIFAR-10 architecture, not an official CIFAR-100 pretrained configuration.

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

Use `cifar100_resformer` for the other backbone. This core-method entry scores S0 and conditional checkpoints: 12 frozen checkpoints and 48 test routes per architecture. It does not label that subset as the original full baseline comparison or invent a comparative gate. Original baseline execution and scoring programs remain in `revision_round2/` and `revision_round3/`; exact replay of their recorded starts still requires the separately supplied original model/input assets. No full S0 or full adapted checkpoint download is hosted by this update. Existing small `bn_delta.pt` files are calibration increments, not complete models.

## Resource measurements

Every training/fitting/scoring stage uses the retained per-board measurement implementation. It records wall time, sampled board energy, peak board memory, sampling errors and gaps. Invalid energy observations remain invalid; they are not replaced by estimates or used to rerun quality experiments.

`resources/make_conditional_plan.py` creates the original train/fit packets for a separately measured run; `resources/measure_plan.py` executes them with the recorded 0.5-second sampling interval. Generate this plan before running the corresponding conditional command, so no completed trajectory is repeated:

```sh
python extensions/resources/make_conditional_plan.py --work /runs/c100-spikformer --role formal1 --output /runs/c100-spikformer/resource-plan.json
python extensions/resources/measure_plan.py --plan /runs/c100-spikformer/resource-plan.json --output /runs/c100-spikformer/resource-measurement --execute
python extensions/resources/summarize.py --receipts /runs/c100-spikformer/resource-measurement --output /runs/c100-spikformer/resources.csv
```

The measured plan is an alternative train/fit-only execution, not an extra repeat after the full `conditional` command. For the full validation/test workflow, use `conditional` and its existing per-stage receipts. Three budgets and four fitted views per role give 36 fits per architecture, or 72 for both CIFAR-100 architectures. Final deployment exports have three checkpoints per role. The retained R2 model/runtime, calibration and inference timing code is also included under each CIFAR-100 `constructor/scripts/`. Board power is not whole-system power, and per-board measurements do not imply an isolated CPU host. New hardware measurements must remain separate from the recorded experiments.
