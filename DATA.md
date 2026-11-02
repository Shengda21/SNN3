# Data and model access

## Included here

- Per-item loss sums, target counts, predictions and labels used to calculate the reported quality measures.
- Recorded training and timing observations, paired-seed contrasts, uncertainty intervals and aggregate tables.
- Split and sampling information, scoring configurations, source identities and analysis programs.
- Figure programs and the numerical inputs needed by the CPU reproduction commands.

The NPZ files contain experimental outputs and sampling records. They do not contain the full source image dataset or text corpus.

## Original datasets

**CIFAR-100:** obtain the Python distribution from the [official CIFAR page](https://www.cs.toronto.edu/~kriz/cifar.html). The recorded split and preprocessing configuration are supplied with the experiment. Images are not redistributed here.

**WikiText-2 raw:** obtain the `wikitext-2-raw-v1` configuration from the [WikiText dataset repository](https://huggingface.co/datasets/Salesforce/wikitext). The language experiment records its tokenization, holdout permutation and masking seeds. The recorded official-test source information is in `experiment/orthogonal_pilot_20260907/results/paper_experiments_20260908/E0/test_source.json`. Its repository-commit field is null; the recorded file identity is available. Original corpus files are not redistributed here.

Use each dataset under its original terms. Reconstructing inputs from a newer upstream file should not be assumed to reproduce the archived file exactly.

## Models and external code

The vision initialization was trained from random initialization with the included SpikingResformer-based construction code. Initializations A and B have separate seeds and runtime histories. Their construction and adaptation records are included; the full checkpoints are not.

The language model starts from the public [SmoothSpike model](https://modelscope.cn/models/kailai1104/SmoothSpike), converted using its upstream implementation. Fetch that implementation separately from [SmoothSpike](https://github.com/CayleyZ/SmoothSpike). The archived source-file identities are retained in `revision_round2/protocol/source_identities.json`; obtaining a current upstream revision does not by itself establish that these identities match.

Full initial and adapted checkpoints, fixed model inputs and the large companion archives are not hosted in this repository. Some small recorded normalization-state deltas are included with the earlier experiment; they are not complete checkpoints. Accordingly, the earlier fixed-checkpoint replay commands recompute the reported results from saved outputs, while their fixed-checkpoint GPU replay also requires the external code and model/input assets described above. The additional from-scratch construction and adaptation entrypoints are documented under `extensions/`.
# Additional image experiments

The additional data protocols and preparation commands are in [extensions/README.md](extensions/README.md). They use official CIFAR batches, official BloodMNIST64 splits, a fixed grouped EuroSAT RGB split and a declared derived PlantWild v1 clean-train/validation protocol with its official test set retained. No source image or resized image is distributed in this update. The small per-route tables are model-output statistics; optional raw-array recomputation requires locally supplied saved predictions.
