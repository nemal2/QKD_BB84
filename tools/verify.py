"""Verify the distributed repository using only the Python standard library.

Run from any directory. The checksum manifest defines the portable release;
Git metadata, virtual environments and generated workspaces are outside it.
"""
from pathlib import Path
import ast
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]


def verify():
    """Check every declared file, validate text formats and inspect front matter."""
    manifest = json.loads((ROOT / "checksums.json").read_text())
    for name, expected in manifest.items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise RuntimeError(f"Missing or unsafe release path: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Changed release file: {name}")
        if path.suffix == ".py":
            ast.parse(path.read_text(), filename=name)
        elif path.suffix == ".json":
            json.loads(path.read_text())
    main = (ROOT / "overleaf/main.tex").read_text()
    if r"\documentclass[manuscript,screen,review,anonymous]{acmart}" not in main:
        raise RuntimeError("The repository manuscript must use anonymous front matter.")
    if re.search(r"\\(?:author|email|affiliation|orcid)\{", main):
        raise RuntimeError("Author metadata found in repository manuscript.")
    protocols = [f"supplementary/{name}/protocol.json" for name in
                 ["study", "extensions", "diagnostic_validation",
                  "observation_validation", "precision_validation"]]
    if not all(name in manifest for name in protocols):
        raise RuntimeError("A scientific protocol is missing from the release.")
    return {"status": "passed", "files_checked": len(manifest),
            "python_modules_parsed": sum(name.endswith(".py") for name in manifest),
            "scientific_protocols": len(protocols), "anonymous_front_matter": True}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
