import json
import unittest

from kaptia_agent_loop import run_tool_loop, _validate_request


class AgentLoopTests(unittest.TestCase):
    def test_unknown_tool_is_rejected(self):
        with self.assertRaises(ValueError):
            _validate_request({"action": "tool", "name": "run_shell", "arguments": {"command": "echo hi"}})

    def test_extra_arguments_are_rejected(self):
        with self.assertRaises(ValueError):
            _validate_request({"action": "tool", "name": "read_project_file",
                               "arguments": {"path": "README.md", "other": "x"}})

    def test_unapproved_file_is_rejected(self):
        with self.assertRaises(ValueError):
            _validate_request({"action": "tool", "name": "read_project_file",
                               "arguments": {"path": "../../secrets",}})

    def test_done_stops_without_dispatch(self):
        calls = []
        def generate(messages, **kwargs):
            return json.dumps({"action": "done", "summary": "不需要工具"})
        def dispatch(name, args):
            calls.append(name)
            return {"ok": True}
        result = run_tool_loop(generate, "hello", dispatch)
        self.assertEqual(calls, [])
        self.assertEqual(result["results"], [])

    def test_tool_result_is_returned_to_next_model_turn(self):
        calls = []
        def generate(messages, **kwargs):
            calls.append(messages[:])
            if len(calls) == 1:
                return json.dumps({"action": "tool", "name": "search_project_knowledge",
                                   "arguments": {"query": "Kaptia", "top_k": 2}})
            return json.dumps({"action": "done", "summary": "完成"})
        result = run_tool_loop(generate, "查專案", lambda name, args: {"ok": True, "results": [{"text": "sample"}]})
        self.assertEqual(len(result["results"]), 1)
        self.assertIn("sample", calls[1][-1]["content"])

    def test_invalid_json_fails_closed(self):
        called = []
        result = run_tool_loop(lambda messages, **kwargs: "not json",
                               "task", lambda name, args: called.append(name) or {"ok": True})
        self.assertEqual(called, [])
        self.assertFalse(result["results"])

    def test_loop_cap_is_enforced(self):
        count = []
        def generate(messages, **kwargs):
            return json.dumps({"action": "tool", "name": "search_project_knowledge",
                               "arguments": {"query": "Kaptia", "top_k": 1}})
        result = run_tool_loop(generate, "task",
                               lambda name, args: count.append(name) or {"ok": True},
                               max_steps=2)
        self.assertEqual(len(count), 2)
        self.assertTrue(any("上限" in event for event in result["events"]))


if __name__ == "__main__":
    unittest.main()
