import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import pandas as pd

from predict import load_dataset, main


class ClassifierChecks(unittest.TestCase):
    def test_vectorizer_does_not_learn_unseen_test_words(self):
        train_titles = pd.Series(["learn algebra", "music concert", "learn chemistry", "music album"])
        test_titles = pd.Series(["unseenholdouttoken algebra", "differentholdouttoken concert"])
        train_labels, test_labels = pd.Series([1, 0, 1, 0]), pd.Series([1, 0])
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "input.csv"
            pd.DataFrame({"Title": [*train_titles, *test_titles], "Category_ID": [27, 10, 27, 10, 27, 10]}).to_csv(source, index=False)
            split = (train_titles, test_titles, train_labels, test_labels)
            with patch("sklearn.model_selection.train_test_split", return_value=split):
                with contextlib.redirect_stdout(io.StringIO()):
                    main(["train", "--data", str(source), "--output-dir", folder, "--model", "naive-bayes"])
            pipeline = joblib.load(Path(folder) / "pipeline.joblib")
            self.assertNotIn("unseenholdouttoken", pipeline.named_steps["tfidf"].vocabulary_)
            self.assertNotIn("differentholdouttoken", pipeline.named_steps["tfidf"].vocabulary_)

    def test_duplicate_titles_cannot_cross_split_with_conflicting_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "input.csv"
            pd.DataFrame({"Title": ["Same Title", " same   title "], "Category_ID": [27, 10]}).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "conflicting"):
                load_dataset(path)

    def test_missing_columns_are_explained(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "input.csv"
            path.write_text("Title\nA title\n")
            with self.assertRaisesRegex(ValueError, "Category_ID"):
                load_dataset(path)

    def test_synthetic_cli_train_save_reload_predict(self):
        sample = Path(__file__).resolve().parents[1] / "examples/demo.csv"
        with tempfile.TemporaryDirectory() as folder:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                main(["train", "--data", str(sample), "--output-dir", folder])
            report = json.loads((Path(folder)/"metrics.json").read_text())
            self.assertTrue(report["synthetic_demo"])
            self.assertEqual(report["train_rows"] + report["test_rows"], 40)
            model = Path(folder) / "pipeline.joblib"
            self.assertEqual(len(joblib.load(model).predict(["An algebra lesson"])), 1)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                main(["predict", "--artifact", str(model), "--title", "An algebra lesson"])
            self.assertEqual(json.loads(output.getvalue())[0]["title"], "An algebra lesson")


if __name__ == "__main__":
    unittest.main()
