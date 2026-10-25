"""Create a verified portable ZIP from the repository's declared release files."""
from pathlib import Path
import argparse
import hashlib
import json
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED
from verify import ROOT, verify


def main():
    """Package only declared inputs; omit local history, caches and run outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = verify()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new archive path: {output.name}")
    manifest = json.loads((ROOT / "checksums.json").read_text())
    output.parent.mkdir(parents=True, exist_ok=True)
    # Fixed archive timestamps and modes avoid host-specific metadata and make
    # the same set of input bytes produce the same ZIP in the same environment.
    with ZipFile(output, "x", compression=ZIP_DEFLATED, compresslevel=4) as archive:
        for name in sorted([*manifest, "checksums.json"]):
            item = ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            item.compress_type = ZIP_DEFLATED
            item.create_system = 3
            item.external_attr = 0o100644 << 16
            archive.writestr(item, (ROOT / name).read_bytes(), compresslevel=4)
    with ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Archive CRC verification failed.")
    report.update(archive=output.name,
                  archive_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
