import json
import tempfile
import unittest
from pathlib import Path

from kaptia_prepare_text_corpus import read_documents, normalize


class TextCorpusPreparationTests(unittest.TestCase):
    def test_normalize_line_endings_and_trailing_space(self):
        self.assertEqual(normalize("甲  \r\n乙\r"), "甲\n乙")

    def test_reads_txt_and_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.txt").write_text("這是足夠長的文字內容。\n", encoding="utf-8")
            (root / "b.jsonl").write_text(
                json.dumps({"text": "另一段足夠長的訓練文字。"}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            docs = list(read_documents(root))
            self.assertEqual(len(docs), 2)
            self.assertIn("這是足夠長的文字內容。", docs)
            self.assertIn("另一段足夠長的訓練文字。", docs)


if __name__ == "__main__":
    unittest.main()
