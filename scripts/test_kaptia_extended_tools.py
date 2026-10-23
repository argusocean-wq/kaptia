"""Regression tests for Kaptia's optional extended tools; no network/model required."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import kaptia_extended_tools as tools


class CalculatorTests(unittest.TestCase):
    def test_arithmetic_and_precedence(self):
        result = tools.calculate("2 + 3 * (4 ** 2)")
        self.assertTrue(result["ok"])
        self.assertEqual(result["value"], 50)

    def test_division_by_zero_fails_closed(self):
        result = tools.calculate("1 / 0")
        self.assertFalse(result["ok"])

    def test_names_calls_and_attributes_are_rejected(self):
        for expression in ("__import__('os')", "open('secret')", "x + 1", "(1).__class__"):
            with self.subTest(expression=expression):
                self.assertFalse(tools.calculate(expression)["ok"])

    def test_boolean_is_not_a_number(self):
        self.assertFalse(tools.calculate("True + 1")["ok"])

    def test_expression_and_exponent_limits(self):
        self.assertFalse(tools.calculate("1" * 301)["ok"])
        self.assertFalse(tools.calculate("2 ** 101")["ok"])

    def test_non_finite_or_excessive_results_rejected(self):
        self.assertFalse(tools.calculate("1e309")["ok"])
        self.assertFalse(tools.calculate("10 ** 100")["ok"])


class DispatchAndDomainTests(unittest.TestCase):
    def test_dispatch_rejects_unknown_tool(self):
        result = tools.dispatch_tool("execute_shell", {"command": "whoami"})
        self.assertFalse(result["ok"])
        self.assertIn("allowed_tools", result)

    def test_dispatch_requires_object_arguments(self):
        result = tools.dispatch_tool("calculate", ["1+1"])
        self.assertFalse(result["ok"])

    def test_dispatch_rejects_unknown_arguments(self):
        result = tools.dispatch_tool("calculate", {"expression": "1+1", "extra": True})
        self.assertFalse(result["ok"])

    def test_search_input_limits_are_checked_before_network(self):
        self.assertFalse(tools.web_search("", 5)["ok"])
        self.assertFalse(tools.web_search("x", True)["ok"])
        self.assertFalse(tools.web_search("x", 9)["ok"])

    def test_missing_domain_config_returns_safe_defaults(self):
        with patch.object(tools, "DOMAIN_CONFIG", Path("/path/that/does/not/exist/kaptia-domain.json")):
            config = tools.get_domain_config()
        self.assertFalse(config["enabled"])
        self.assertEqual(config["allowed_topics"], [])

    def test_malformed_domain_config_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "domain.json"
            path.write_text(json.dumps({"enabled": True, "allowed_topics": "not-a-list"}), encoding="utf-8")
            with patch.object(tools, "DOMAIN_CONFIG", path):
                with self.assertRaises(ValueError):
                    tools.get_domain_config()


class RagIndexPointerTests(unittest.TestCase):
    def test_invalid_pointer_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pointer = root / "CURRENT"
            indexes = root / "indexes"
            indexes.mkdir()
            pointer.write_text("../outside", encoding="ascii")
            with patch.object(tools, "VECTOR_DIR", root), \
                 patch.object(tools, "CURRENT_INDEX_FILE", pointer), \
                 patch.object(tools, "INDEXES_DIR", indexes):
                with self.assertRaises(ValueError):
                    tools._resolve_active_index_dir()

    def test_current_resolves_only_valid_versioned_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            indexes = root / "indexes"
            index_id = "index-" + "a" * 32
            index_dir = indexes / index_id
            index_dir.mkdir(parents=True)
            pointer = root / "CURRENT"
            pointer.write_text(index_id + "\\n", encoding="ascii")
            with patch.object(tools, "VECTOR_DIR", root), \
                 patch.object(tools, "CURRENT_INDEX_FILE", pointer), \
                 patch.object(tools, "INDEXES_DIR", indexes):
                self.assertEqual(tools._resolve_active_index_dir(), index_dir)

    def test_missing_pointer_allows_legacy_layout_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("vectors.npy", "documents.json", "manifest.json"):
                (root / name).write_text("legacy", encoding="utf-8")
            with patch.object(tools, "VECTOR_DIR", root), \
                 patch.object(tools, "CURRENT_INDEX_FILE", root / "CURRENT"), \
                 patch.object(tools, "INDEXES_DIR", root / "indexes"):
                self.assertEqual(tools._resolve_active_index_dir(), root)

    def test_atomic_pointer_switch_preserves_previous_pointer_on_replace_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            root.mkdir(exist_ok=True)
            pointer = root / "CURRENT"
            pointer.write_text("index-" + "b" * 32 + "\\n", encoding="ascii")
            with patch.object(tools, "VECTOR_DIR", root), \
                 patch.object(tools, "CURRENT_INDEX_FILE", pointer), \
                 patch("kaptia_extended_tools.os.replace", side_effect=OSError("simulated crash")):
                with self.assertRaises(OSError):
                    tools._publish_index_pointer("index-" + "c" * 32)
            self.assertEqual(pointer.read_text(encoding="ascii").strip(), "index-" + "b" * 32)


if __name__ == "__main__":
    unittest.main()
