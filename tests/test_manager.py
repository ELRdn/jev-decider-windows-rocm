import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

config = json.dumps({"python": "python.exe", "pythonw": "pythonw.exe", "node": "node.exe",
                     "gateway_launcher": "jev-codex.mjs", "cache": "test-cache", "decider_dir": ".",
                     "upstream": "https://chatgpt.com/backend-api/codex"})
spec = importlib.util.spec_from_file_location("candidate_manager", Path(__file__).resolve().parents[1] / "service" / "jev_local.py")
manager = importlib.util.module_from_spec(spec)
with patch.object(Path, "read_text", return_value=config):
    spec.loader.exec_module(manager)


class Child:
    def __init__(self):
        self.stopping = False
        self.polls = 0
    def poll(self):
        if not self.stopping:
            return None
        self.polls += 1
        return None if self.polls == 1 else 0


class Function:
    def __call__(self, *args):
        return 1


class ManagerTests(unittest.TestCase):
    def test_planned_recycles_survive_delayed_child_exit_without_crash_limit(self):
        recycle = {'service': 'jev-local-decider', 'phase': 'recycle_pending'}
        responses = [None, recycle, None, recycle, None, recycle, None]
        children = []
        def request(url, post=False):
            return responses.pop(0) if url == manager.BACKEND + '/health' else {}
        def spawn(*args, **kwargs):
            child = Child()
            children.append(child)
            return child
        def stop():
            if children:
                children[-1].stopping = True
        stop_marker = SimpleNamespace(exists=lambda: not responses)
        kernel = SimpleNamespace(CreateMutexW=Function(), CloseHandle=Function())
        with patch.object(manager, 'STOP', stop_marker), patch.object(manager, 'request', side_effect=request), \
             patch.object(manager, 'gateway_start'), patch.object(manager, 'log'), \
             patch.object(manager, 'stop_backend', side_effect=stop), patch.object(manager, 'occupied', return_value=False), \
             patch.object(manager.ctypes, 'WinDLL', return_value=kernel), patch.object(manager.ctypes, 'get_last_error', return_value=0), \
             patch.object(manager.time, 'sleep'), patch.object(manager.subprocess, 'Popen', side_effect=spawn), patch.object(Path, 'open', unittest.mock.mock_open()):
            manager.supervise()
        self.assertEqual(len(children), 4)


if __name__ == '__main__':
    unittest.main()
