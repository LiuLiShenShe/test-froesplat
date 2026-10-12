from __future__ import annotations

import argparse
import sys
from pathlib import Path


STAGE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STAGE / "scripts"))
from phase20d_common import ROOT, read_csv, relative, sha256, unique_index, write_csv


def backup(stage):
    completion = stage / "completion"
    inventory = []
    sources = read_csv(completion / "GT32_SOURCE_ARCHIVE.csv")
    frozen = read_csv(stage / "TEST32_INPUT_MANIFEST.csv")
    if len(sources) != 32 or len(unique_index(sources, "frame_id")) != 32 or {row["frame_id"] for row in sources} != {row["frame_id"] for row in frozen}:
        raise ValueError("Full 32-frame source archive coverage required")
    for row in sources:
        source = ROOT / row["source_json"]
        expected = row["source_sha256_before"]
        if sha256(source) != expected:
            raise ValueError("Source changed before private byte-exact backup")
        destination = completion / "local_gt_backup" / expected / f"{row['frame_id']}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as handle:
            handle.write(source.read_bytes())
        if sha256(source) != expected or sha256(destination) != expected:
            raise ValueError("Byte-exact backup identity mismatch")
        inventory.append({"frame_id": row["frame_id"], "source_json": row["source_json"],
                          "source_sha256": expected, "local_backup": relative(destination),
                          "backup_sha256": sha256(destination), "bytes": destination.stat().st_size,
                          "purpose": "UNCHANGED_HUMAN_EXPORT_BACKUP_NOT_APPROVED_GT"})
    write_csv(completion / "GT32_LOCAL_BACKUP_INDEX.csv", inventory)
    print(f"Preserved {len(inventory)} unchanged manual exports locally; no GT generated or approved")


def main():
    parser = argparse.ArgumentParser(description="Private byte-exact backup, never redraw/relabel/approve GT")
    parser.add_argument("--stage-dir", type=Path, default=STAGE)
    args = parser.parse_args()
    backup(args.stage_dir.resolve())


if __name__ == "__main__":
    main()
