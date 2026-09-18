"""Check restored weights against the existing confirmation and archive identities."""
from pathlib import Path
import argparse
import hashlib
import json
import os


def absolute(value):
    result = Path(value).resolve()
    return Path("\\\\?\\" + str(result)) if os.name == "nt" and not str(result).startswith("\\\\?\\") else result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-root", required=True, type=Path,
                        help="Restored archive root containing the two original project directories")
    args = parser.parse_args()
    package = absolute(Path(__file__).parent)
    weights = absolute(args.weights_root)
    record = package / "experiment/orthogonal_pilot_20260907/results/paper_experiments_20260908"
    manifest = json.loads((record / "confirm_manifest.json").read_text())
    cache, rows = {}, []
    for model in manifest["models"]:
        relative = "orthogonal_pilot_20260907/data/fused_state.pt" if model["checkpoint"] is None else "orthogonal_pilot_20260907/" + model["checkpoint"].split("orthogonal_pilot_20260907/", 1)[1]
        path = weights / relative
        if relative not in cache:
            if path.is_file():
                with path.open("rb") as f:
                    cache[relative] = hashlib.file_digest(f, "sha256").hexdigest()
            else:
                cache[relative] = None
        rows.append({"id": model["id"], "status": "missing" if cache[relative] is None else
                     "match" if cache[relative] == model["checkpoint_sha256"] else "mismatch"})
    archive = json.loads((record / "archive_manifest.json").read_text())
    vision_path = "direction1_budget_rotation_20260907/results/reconstructed_identity_p0.0/best.pt"
    expected = next(r for r in archive["files"] if r["path"] == vision_path)
    path = weights / vision_path
    if path.is_file():
        with path.open("rb") as f:
            actual = hashlib.file_digest(f, "sha256").hexdigest()
        status = "match" if actual == expected["sha256"] else "mismatch"
    else:
        status = "missing"
    rows.append({"id": "reconstructed_vision_best", "status": status})
    result = {"matches": sum(r["status"] == "match" for r in rows),
              "missing": sum(r["status"] == "missing" for r in rows),
              "mismatches": sum(r["status"] == "mismatch" for r in rows), "models": rows}
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if all(r["status"] == "match" for r in rows) else 1)


if __name__ == "__main__":
    main()
