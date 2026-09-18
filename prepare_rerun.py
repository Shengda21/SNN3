"""Create a clean local execution tree; never download assets or launch experiments."""
from pathlib import Path
import argparse
import os
import shutil


def path(value):
    result = Path(value).resolve()
    return Path("\\\\?\\" + str(result)) if os.name == "nt" and not str(result).startswith("\\\\?\\") else result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args()
    source = path(Path(__file__).parent / "experiment")
    destination = path(args.destination)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "setup").mkdir()
    language = "orthogonal_pilot_20260907"
    vision = "direction1_budget_rotation_20260907"
    for project in [language, vision]:
        for folder in ["code", "vendor"]:
            shutil.copytree(source / project / folder, destination / project / folder,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        (destination / project / "data").mkdir()
        (destination / project / "results").mkdir()
    shutil.copytree(source / language / "protocol", destination / language / "protocol")
    shutil.copytree(source / language / "results/trained_state/baselines",
                    destination / language / "results/trained_state/baselines")
    shutil.copytree(source / vision / "results/screen_identity_p0.0",
                    destination / vision / "results/screen_identity_p0.0")
    shutil.copy2(source / vision / "data/CIFAR100_split.json", destination / vision / "data/CIFAR100_split.json")
    (destination / language / "results/paper_experiments_20260908").mkdir()
    print("Prepared code, protocol, and small input records. No external assets were downloaded and no experiments were run.")
    print(str(destination).removeprefix("\\\\?\\"))


if __name__ == "__main__":
    main()
