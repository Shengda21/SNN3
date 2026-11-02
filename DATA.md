# Data and model access

## Included in this repository

- Per-item loss sums, target counts, predictions and labels used to calculate the reported quality measures, for every scoring route of the CIFAR-100 cohorts, the six replication starts and the four additional datasets.
- Recorded training and timing observations, paired-seed contrasts, uncertainty intervals and aggregate tables.
- Split and sampling information, scoring configurations, source identities and analysis programs.
- Figure programs and the numerical inputs needed by the CPU reproduction commands.

The NPZ files contain experimental outputs and sampling records. They do not contain source images or text.

## Released model checkpoints

Checkpoints are published as GitHub Releases of this repository (see the table in [README.md](README.md)). The archives contain the authors' own checkpoints, BN statistic banks and the loader code needed to evaluate them. Each release note lists byte counts and SHA-256 values; the verification programs check every member. Releases published so far:

- `model-assets-v1`: the R3 companion (A/B cohorts) and the vision subset of the R2.1 companion.
- `model-assets-v2`: the 464-route CIFAR-100 collection (A/B, C-conf, F-arch, A-cal) and the six replication starts C'1--3 and F'1--3.
- `model-assets-v3`: the four additional datasets (CIFAR-10, BloodMNIST64, EuroSAT RGB, PlantWild): retained starts, adapted and MTT states, the saved validation and test logits of every scoring route (from which the manuscript's temperature-calibrated cross-entropy is computed) and the scoring records.

Not redistributed: the language-model weights derived from the third-party SmoothSpike model, and every dataset archive.

All released checkpoints, BN banks and the result records in this repository are licensed under CC BY-NC 4.0 (`LICENSE-DATA.md`); the code is MIT-licensed (`LICENSE`).

## Original datasets

**CIFAR-100 and CIFAR-10:** obtain the Python distributions from the [official CIFAR page](https://www.cs.toronto.edu/~kriz/cifar.html). The recorded split and preprocessing configurations are supplied with the experiments. Images are not redistributed here. The expected CIFAR-100 archive has 169,001,437 bytes and SHA-256 `85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7`.

**BloodMNIST64, EuroSAT RGB and PlantWild v1:** obtain the 64-pixel BloodMNIST archive from [MedMNIST](https://medmnist.com/), the original RGB archive from [EuroSAT](https://github.com/phelber/EuroSAT), and v1 `plantwild.zip` from [PlantWild](https://tqwei05.github.io/PlantWild/) ([dataset repository](https://huggingface.co/datasets/uqtwei2/PlantWild)). The preparation programs under `extensions/` check archive identities and rebuild the fixed splits (official BloodMNIST split; grouped, stratified 70/15/15 EuroSAT split; PlantWild official test set with de-duplicated training and validation sets). Raw or resized images are not redistributed. PlantWild's CC BY-NC-ND terms apply to its source data.

**WikiText-2 raw:** obtain the `wikitext-2-raw-v1` configuration from the [WikiText dataset repository](https://huggingface.co/datasets/Salesforce/wikitext). The language experiment records its tokenization, holdout permutation and masking seeds. The recorded official-test source information is in `experiment/orthogonal_pilot_20260907/results/paper_experiments_20260908/E0/test_source.json`. Original corpus files are not redistributed here.

Use each dataset under its original terms. Reconstructing inputs from a newer upstream file should not be assumed to reproduce the archived file exactly.

## Models and external code

The vision initializations were trained from random initialization with the included SpikingResformer-based and Spikformer construction code. Each start has its own seed and runtime history; construction and adaptation records are included, and the checkpoints are in the releases listed above.

The language model starts from the public [SmoothSpike model](https://modelscope.cn/models/kailai1104/SmoothSpike), converted using its upstream implementation. Fetch that implementation separately from [SmoothSpike](https://github.com/CayleyZ/SmoothSpike). The archived source-file identities are retained in `revision_round2/protocol/source_identities.json`; obtaining a current upstream revision does not by itself establish that these identities match.

Fixed-checkpoint replay requires the released archives, the original dataset files and a CUDA environment matching the recorded one (PyTorch 2.7.1, CUDA 12.8 for the current cohorts). The replay commands are documented in the README of each release volume; recomputation of the reported tables and figures from the saved outputs needs none of these.
