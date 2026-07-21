import sqlite3
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from scripts._common import REPO_ROOT, extract_zip, remove_files, run_script, write_sqlite_table
from scripts.prepare import discover_datasets


class ExtractZipTests(unittest.TestCase):
    def _make_zip(self, zip_path: Path, entries: dict[str, bytes]) -> None:
        with zipfile.ZipFile(zip_path, "w") as zf:
            for name, content in entries.items():
                zf.writestr(name, content)

    def test_extract_zip_preserves_directory_structure_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            zip_path = tmp_path / "archive.zip"
            self._make_zip(zip_path, {"sub/a.csv": b"a", "b.csv": b"b"})
            out_dir = tmp_path / "out"

            extracted = extract_zip(zip_path, out_dir)

            self.assertEqual({p.relative_to(out_dir) for p in extracted}, {Path("sub/a.csv"), Path("b.csv")})
            self.assertEqual((out_dir / "sub" / "a.csv").read_bytes(), b"a")
            self.assertEqual((out_dir / "b.csv").read_bytes(), b"b")

    def test_extract_zip_junk_paths_flattens_into_out_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            zip_path = tmp_path / "archive.zip"
            self._make_zip(zip_path, {"sub/a.csv": b"a", "b.csv": b"b"})
            out_dir = tmp_path / "out"

            extracted = extract_zip(zip_path, out_dir, junk_paths=True)

            self.assertEqual({p.name for p in extracted}, {"a.csv", "b.csv"})
            self.assertEqual({p.parent for p in extracted}, {out_dir})

    def test_extract_zip_skips_directory_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            zip_path = tmp_path / "archive.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("sub/", "")
                zf.writestr("sub/a.csv", "a")
            out_dir = tmp_path / "out"

            extracted = extract_zip(zip_path, out_dir)

            self.assertEqual(len(extracted), 1)
            self.assertTrue(extracted[0].is_file())


class RemoveFilesTests(unittest.TestCase):
    def test_remove_files_deletes_all_given_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            paths = [tmp_path / "a.csv", tmp_path / "b.csv"]
            for path in paths:
                path.write_text("data")

            remove_files(paths)

            for path in paths:
                self.assertFalse(path.exists())

    def test_remove_files_does_not_raise_on_missing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "does-not-exist.csv"
            remove_files([missing])  # should not raise


class RunScriptTests(unittest.TestCase):
    @patch("scripts._common.subprocess.run")
    def test_run_script_invokes_current_interpreter_with_args(self, mock_run):
        script = Path("some_script.py")
        run_script(script, "--foo", "bar")

        mock_run.assert_called_once_with([sys.executable, str(script), "--foo", "bar"], check=True, cwd=REPO_ROOT)


class WriteSqliteTableTests(unittest.TestCase):
    def test_writes_indexed_table_readable_via_sqlite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "processed" / "tsa.db"
            df = pd.DataFrame({"state": [1, 2, 3], "CCT": [0.1, 0.2, 0.3]})

            write_sqlite_table(df, path, table="tsa")

            self.assertTrue(path.exists())
            with sqlite3.connect(path) as conn:
                result = pd.read_sql_query("SELECT * FROM tsa WHERE state IN (?, ?)", conn, params=["1", "3"])
                index_names = {row[1] for row in conn.execute("PRAGMA index_list(tsa)").fetchall()}
            self.assertEqual(sorted(result["state"]), ["1", "3"])
            self.assertTrue(any("state" in name for name in index_names))

    def test_casts_index_col_to_str(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "processed" / "fsa.db"
            df = pd.DataFrame({"state": [1, 2], "minF": [0.9, 0.8]})

            write_sqlite_table(df, path, table="fsa")

            with sqlite3.connect(path) as conn:
                result = pd.read_sql_query("SELECT * FROM fsa", conn)
            self.assertEqual(result["state"].tolist(), ["1", "2"])


class DiscoverDatasetsTests(unittest.TestCase):
    def test_discovers_all_known_datasets_by_directory_layout(self):
        datasets = discover_datasets()

        for expected_key in ("bus39", "eles/2026-01", "interscada/fr", "interscada/pl"):
            self.assertIn(expected_key, datasets)
            self.assertEqual(datasets[expected_key].name, "prepare.py")
            self.assertTrue(datasets[expected_key].is_file())

    def test_dataset_key_matches_path_relative_to_datasets_dir(self):
        datasets = discover_datasets()

        for key, script in datasets.items():
            relative = script.relative_to(REPO_ROOT / "datasets")
            self.assertEqual("/".join(relative.parts[:-1]), key)


if __name__ == "__main__":
    unittest.main()
