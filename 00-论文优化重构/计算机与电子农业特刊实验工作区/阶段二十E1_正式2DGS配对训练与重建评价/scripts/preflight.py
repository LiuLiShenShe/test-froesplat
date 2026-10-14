from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess

from common import DATA, E0, GROUPS, MATRIX, ROOT, SAMPLES, STAGE, config_for, now, read_csv, sha256, write_json


def verify_managed():
    rows = read_csv(E0 / "reports/MANAGED_ARTIFACT_HASHES.csv")
    hashes = {}
    failures = []
    for row in rows:
        from pathlib import Path
        path = Path(row["path"])
        try:
            stat = path.stat()
            key = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
            if key not in hashes:
                hashes[key] = sha256(path)
            valid = stat.st_size == int(row["bytes"]) and hashes[key] == row["sha256"]
            valid = valid and path.is_symlink() == (row["symlink"] == "True")
            valid = valid and str(path.resolve()) == row["resolved_path"]
            if not valid:
                failures.append(str(path))
        except OSError as error:
            failures.append(f"{path}: {error}")
    return {"managed_rows": len(rows), "unique_inodes_hashed": len(hashes), "failures": failures}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    evidence = verify_managed()
    assert not evidence["failures"], evidence["failures"]
    hashes = json.loads((E0 / "reports/FROZEN_TRAIN_CONFIG_HASHES.json").read_text())
    for path, digest in hashes.items():
        assert sha256(ROOT / path) == digest, path
    code_rows = [row for row in read_csv(E0 / "reports/SOURCE_PROTECTION.csv")
                 if row["kind"] == "PRODUCTION_CODE"]
    for row in code_rows:
        assert sha256(ROOT / row["path"]) == row["sha256"], row["path"]
    hardware = json.loads((E0 / "reports/FROZEN_HARDWARE_IDENTITY.json").read_text())
    gpu = subprocess.check_output(["nvidia-smi", "-i", "1", "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    assert gpu == hardware["training_gpu_uuid"]
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "1"
    gate = json.loads((E0 / "reports/FINAL_ACCEPTANCE.json").read_text())
    split_rows = read_csv(E0 / "TRAIN_VALIDATION_SPLIT.csv")
    dataset = read_csv(E0 / "MATCHED_2DGS_DATASET_MANIFEST.csv")
    config_records = []
    for sample in SAMPLES:
        assert gate["scenes"][sample]["ready"]
        assert json.loads((E0 / f"reports/{sample}_PREFLIGHT.json").read_text())["status"] == "PASS"
        config_sets = []
        for group in GROUPS:
            config = config_for(sample, group)
            assert config["seed"] == 0 and config["CUDA_VISIBLE_DEVICES"] == "1"
            assert config["model"]["resolution"] == 2 and config["model"]["sh_degree"] == 3
            assert not config["model"]["white_background"] and config["optimization"]["iterations"] == 30000
            comparable = json.loads(json.dumps(config))
            for key in ("model", "all_parameters"):
                comparable[key].pop("source_path")
                comparable[key].pop("model_path")
            config_sets.append(comparable)
            from pathlib import Path
            source = Path(config["model"]["source_path"])
            assert source.is_dir()
            if (sample, group) in MATRIX and not args.final:
                assert not Path(config["model"]["model_path"]).exists(), "Output collision"
            selected = [row for row in dataset if row["sample"] == sample and row["group"] == group]
            assert len(selected) == gate["scenes"][sample]["registered"]
            config_records.append({"sample": sample, "group": group, "config_sha256":
                                   sha256(E0 / f"reports/{sample}_{group}_TRAIN_CONFIG.json")})
        assert all(value == config_sets[0] for value in config_sets)
        selected = [row for row in split_rows if row["sample"] == sample]
        names_train = {row["registered_name"] for row in selected if row["split"] == "train"}
        names_validation = {row["registered_name"] for row in selected if row["split"] == "validation"}
        assert not names_train & names_validation
    from common import import_paths
    import_paths()
    import torch
    import simple_knn._C
    import diff_surfel_rasterization._C
    extensions = [{"path": path, "sha256": sha256(path)} for path in
                  (simple_knn._C.__file__, diff_surfel_rasterization._C.__file__)]
    if args.final:
        initial = json.loads((STAGE / "reports/PRETRAINING_INPUT_RECHECK.json").read_text())
        assert extensions == initial["extensions"], "Compiled extension identity changed"
        assert torch.__version__ == initial["torch_version"] and torch.version.cuda == initial["cuda"]
    assert shutil.disk_usage(DATA).free > 50 * 1024 ** 3
    evidence.update({"status": "PASS", "timestamp": now(), "physical_gpu_uuid": gpu,
                     "source_files_hashed": len(code_rows), "config_records": config_records,
                     "torch_version": torch.__version__, "cuda": torch.version.cuda,
                     "extensions": extensions, "disk_free_bytes": shutil.disk_usage(DATA).free,
                     "repeated_forward_audit": False, "optimizer_started": False})
    target = "FINAL_INPUT_RECHECK.json" if args.final else "PRETRAINING_INPUT_RECHECK.json"
    write_json(STAGE / "reports" / target, evidence)
    print(json.dumps(evidence, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
