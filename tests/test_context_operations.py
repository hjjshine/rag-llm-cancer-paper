import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from scripts import delete_old_context_db
from scripts import update_context_db


OLD_VERSION = "2026-03-05"
NEW_VERSION = "2026-05-07"


class ContextOperationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.original_update_root = update_context_db.ROOT
        self.original_delete_root = delete_old_context_db.ROOT
        update_context_db.ROOT = self.root
        delete_old_context_db.ROOT = self.root

    def tearDown(self):
        update_context_db.ROOT = self.original_update_root
        delete_old_context_db.ROOT = self.original_delete_root
        self.temp_dir.cleanup()

    def write_version(self, version):
        (self.root / "db_version_cache.json").write_text(
            json.dumps({"version": version})
        )

    def create_generated_files(self, version):
        for path in update_context_db.expected_files(version):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"test data")

    def run_update_with(self, builder):
        fake_context_db = types.ModuleType("utils.context_db")
        fake_context_db.update_db_files = builder
        with (
            mock.patch.object(
                update_context_db, "get_latest_release", return_value=NEW_VERSION
            ),
            mock.patch.dict(sys.modules, {"utils.context_db": fake_context_db}),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            update_context_db.main()

    def test_update_advances_version_after_all_files_are_created(self):
        self.write_version(OLD_VERSION)

        def builder(version, organizations, force_rebuild):
            self.assertEqual(version, NEW_VERSION)
            self.assertEqual(organizations, ["fda", "ema"])
            self.assertTrue(force_rebuild)
            self.create_generated_files(version)

        self.run_update_with(builder)

        saved = json.loads((self.root / "db_version_cache.json").read_text())
        self.assertEqual(saved["version"], NEW_VERSION)

    def test_update_keeps_old_version_when_a_file_is_missing(self):
        self.write_version(OLD_VERSION)

        def incomplete_builder(version, organizations, force_rebuild):
            first_file = update_context_db.expected_files(version)[0]
            first_file.parent.mkdir(parents=True, exist_ok=True)
            first_file.write_bytes(b"incomplete update")

        with self.assertRaises(RuntimeError):
            self.run_update_with(incomplete_builder)

        saved = json.loads((self.root / "db_version_cache.json").read_text())
        self.assertEqual(saved["version"], OLD_VERSION)

    def test_cleanup_removes_only_the_old_version(self):
        self.write_version(NEW_VERSION)
        self.create_generated_files(OLD_VERSION)
        self.create_generated_files(NEW_VERSION)

        with contextlib.redirect_stdout(io.StringIO()):
            delete_old_context_db.main()

        for path in update_context_db.expected_files(NEW_VERSION):
            self.assertTrue(path.exists(), path)
        for path in update_context_db.expected_files(OLD_VERSION):
            self.assertFalse(path.exists(), path)


if __name__ == "__main__":
    unittest.main()
