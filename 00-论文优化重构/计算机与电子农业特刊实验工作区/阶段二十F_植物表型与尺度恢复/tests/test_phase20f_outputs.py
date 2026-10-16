import csv
import json
import unittest
from pathlib import Path


PHASE = Path(__file__).resolve().parents[1]


def rows(name):
    with (PHASE / name).open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


class Phase20FOutputTests(unittest.TestCase):
    def test_real_mesh_output_has_three_scene_pairs_and_no_physical_values(self):
        reference = rows("PHENOTYPE_MODEL_REFERENCE.csv")
        self.assertEqual(
            {(row["sample_id"], row["mesh_stage"]) for row in reference},
            {
                (scene, stage)
                for scene in ("WanNianQing1", "CaoMei2", "HongZhang")
                for stage in ("raw", "post")
            },
        )
        physical = rows("PHENOTYPE_PHYSICAL_RESULTS.csv")
        self.assertEqual(len(physical), 24)
        self.assertTrue(all(row["value_cm"] == "" for row in physical))
        self.assertEqual(
            {row["status"] for row in physical},
            {"PHYSICAL_SCALE_NOT_AVAILABLE"},
        )

    def test_historical_audit_covers_21_plants_and_63_leaf_slots(self):
        matching = rows("HISTORICAL_PHENOTYPE_MATCHING.csv")
        self.assertEqual(len(matching), 84)
        self.assertEqual(sum(row["record_level"] == "PLANT" for row in matching), 21)
        self.assertEqual(sum(row["record_level"] == "LEAF" for row in matching), 63)
        self.assertEqual(
            {row["historical_leaf_id"] for row in matching if row["record_level"] == "LEAF"},
            {"LEAF_ID_NOT_AVAILABLE"},
        )
        self.assertTrue(all(row["virtual_values_are_new_mesh_results"] == "False" for row in matching))

    def test_component_audit_has_source_hash_summary_for_each_scene(self):
        audit = rows("REMOVED_STRUCTURE_AUDIT.csv")
        summaries = [row for row in audit if row["record_type"] == "SUMMARY"]
        self.assertEqual(
            {row["sample_id"] for row in summaries},
            {"WanNianQing1", "CaoMei2", "HongZhang"},
        )
        self.assertTrue(all(len(row["raw_mesh_sha256"]) == 64 for row in summaries))
        self.assertTrue(all(len(row["post_mesh_sha256"]) == 64 for row in summaries))
        self.assertTrue(
            all(
                row["record_type"] in {"REMOVED_COMPONENT", "SUMMARY"}
                for row in audit
            )
        )

    def test_execution_log_blocks_physical_claims(self):
        execution = json.loads((PHASE / "logs/EXECUTION.json").read_text(encoding="utf-8"))
        self.assertEqual(execution["physical_scale"], "PHYSICAL_SCALE_NOT_AVAILABLE")
        self.assertEqual(execution["vertical_reference"], "NOT_AVAILABLE")
        self.assertEqual(execution["valid_physical_trait_values"], 0)
        self.assertEqual(execution["valid_error_metric_rows"], 0)
        self.assertFalse(execution["mesh_files_modified"])
        self.assertFalse(execution["paper_tex_modified"])
