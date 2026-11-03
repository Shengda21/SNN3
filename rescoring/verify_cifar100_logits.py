"""Verify the cifar100_logits release volume: archive bytes and SHA-256 and the member set, or, after extraction,
every file's size and SHA-256 against cifar100_logits_index.json. Standard library only.

usage: python verify_cifar100_logits.py --index cifar100_logits_index.json --shards DIR_WITH_ZIPS
       python verify_cifar100_logits.py --index cifar100_logits_index.json --root EXTRACTED_DIR
"""
import argparse, hashlib, json, zipfile
from pathlib import Path


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", type=Path, required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--shards", type=Path); g.add_argument("--root", type=Path)
    a = ap.parse_args()
    idx = json.loads(a.index.read_text(encoding="utf-8"))
    files = {r["path"]: r for r in idx["files"]}
    if a.shards:
        seen = {}
        for s in idx["shards"]:
            p = a.shards / s["name"]
            assert p.stat().st_size == s["bytes"] and sha(p) == s["sha256"], f"archive identity {s['name']}"
            with zipfile.ZipFile(p) as z:
                for zi in z.infolist():
                    assert zi.filename not in seen, "duplicate member"; seen[zi.filename] = zi.file_size
        assert seen == {k: v["bytes"] for k, v in files.items()}, "member set differs from index"
        print(json.dumps({"status": "passed_archive_sha_and_member_metadata", "archives": len(idx["shards"]), "files": len(files)}))
    else:
        for k, r in files.items():
            p = a.root / k
            assert p.is_file() and p.stat().st_size == r["bytes"] and sha(p) == r["sha256"], f"file identity {k}"
        print(json.dumps({"status": "passed_extracted_file_identity", "files": len(files)}))


if __name__ == "__main__":
    main()
