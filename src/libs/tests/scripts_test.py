"""Exercise benchmark tasks without changing GPU settings."""

import json
import os
import signal
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TASKS = ROOT / ".config/mise/tasks"
ARTIFACTS = ROOT / ".ai/benchmark-harness"

SMI = """#!/usr/bin/env python3
import os, sys
with open(os.environ['MOCK_LOG'], 'a') as log:
    log.write('smi ' + ' '.join(sys.argv[1:]) + '\\n')
args = ' '.join(sys.argv[1:])
if '--query-supported-clocks' in args:
    print('\\n'.join(['900', '1200', '1000'] * 10000))
elif '--query-compute-apps' in args:
    if os.environ.get('MOCK_BUSY'): print('1234')
elif '--query-gpu=' in args:
    field = next(a.split('=', 1)[1] for a in sys.argv if a.startswith('--query-gpu='))
    values = {'uuid': 'GPU-1234-abcd', 'clocks.max.sm': '1400',
              'persistence_mode': 'Disabled', 'power.draw': '64.5',
              'temperature.gpu': os.environ.get('MOCK_TEMP', '32'),
              'name': 'Mock GPU', 'driver_version': 'test',
              'power.limit': '300', 'clocks.sm': '1000', 'clocks.mem': '1215'}
    print(values[field])
elif '-rgc' in sys.argv and os.environ.get('MOCK_RESTORE_FAIL'):
    sys.exit(1)
"""

APP = """#!/usr/bin/env python3
import json, os, signal, sys, time
with open(os.environ['MOCK_LOG'], 'a') as log:
    log.write('app ' + os.environ.get('CUDA_VISIBLE_DEVICES', '') + '\\n')
if os.environ.get('MOCK_WAIT'):
    def stop(*_):
        with open(os.environ['MOCK_LOG'], 'a') as log: log.write('app stopped\\n')
        sys.exit(143)
    signal.signal(signal.SIGTERM, stop)
    while True: time.sleep(0.01)
print(json.dumps({'args': sys.argv[1:]}))
sys.exit(int(os.environ.get('MOCK_EXIT', '0')))
"""

UV = """#!/usr/bin/env python3
import json, os, pathlib, sys
paths = [pathlib.Path(p) for p in sys.argv[-2:]]
assert paths[0] != paths[1]
for path in paths:
    value = json.loads(path.read_text())
    assert len(value['benchmarks']) == 1
    assert value['benchmarks'][0]['name'] == 'comparison'
print('comparison ran')
sys.exit(int(os.environ.get('MOCK_EXIT', '0')))
"""


class ScriptTests(unittest.TestCase):
    def setUp(self):
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=ARTIFACTS, prefix="scripts-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.mock = self.root / "mock"
        self.mock.mkdir()
        self.log = self.root / "events.log"
        self.env = dict(os.environ)
        for key in list(self.env):
            if key.startswith(("BENCH_", "MOCK_")):
                del self.env[key]
        self.env.update(
            PATH=f"{self.mock}:{os.environ['PATH']}",
            MISE_PROJECT_ROOT=str(self.root),
            MOCK_LOG=str(self.log),
            SSH_HOST="local",
        )
        self.executable(self.mock / "nvidia-smi", SMI)
        self.executable(self.mock / "sudo", "#!/bin/sh\nexit 1\n")
        self.executable(self.mock / "uv", UV)
        self.executable(self.root / "bin/test/app", APP)
        script = (
            self.root
            / ".cache/fetchcontent/nvbench-src/python/scripts/nvbench_compare_robust.py"
        )
        script.parent.mkdir(parents=True)
        script.touch()

    @staticmethod
    def executable(path, text):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        path.chmod(0o755)

    def run_task(self, task, *args, **env):
        return subprocess.run(
            ["bash", str(TASKS / task), *args],
            cwd=self.root,
            env=self.env | env,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )

    def events(self):
        return self.log.read_text() if self.log.exists() else ""

    def test_uuid_and_stdout(self):
        run = self.run_task("bench", "test/app", "--json", "stdout")
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout)["args"], ["--json", "stdout"])
        self.assertIn("app GPU-1234-abcd", self.events())
        self.assertNotIn("-lgc", self.events())
        self.assertNotIn("-rgc", self.events())

    def test_auto_clock_restore(self):
        run = self.run_task(
            "bench", "test/app", BENCH_CLOCK="auto", BENCH_RESTORE_CLOCK="off"
        )
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("-lgc 1000,1000", self.events())
        self.assertIn("-rgc", self.events())
        self.assertIn("-pm 0", self.events())

    def test_prior_clock_and_exit(self):
        run = self.run_task(
            "bench",
            "test/app",
            BENCH_CLOCK="1000",
            BENCH_RESTORE_CLOCK="900,1200",
            MOCK_EXIT="7",
        )
        self.assertEqual(run.returncode, 7, run.stderr)
        self.assertIn("-lgc 900,1200", self.events())
        self.assertNotIn("-rgc", self.events())

    def test_restore_failure(self):
        run = self.run_task(
            "bench",
            "test/app",
            BENCH_CLOCK="1000",
            BENCH_RESTORE_CLOCK="off",
            MOCK_RESTORE_FAIL="1",
        )
        self.assertEqual(run.returncode, 1, run.stderr)

    def test_invalid_settings(self):
        for env in (
            {"BENCH_CLOCK_FRACTION": "x"},
            {"BENCH_CLOCK_FRACTION": "101"},
            {"BENCH_CLOCK": "auto"},
            {"BENCH_IDLE_TIMEOUT": "-1"},
        ):
            with self.subTest(env=env):
                self.assertEqual(
                    self.run_task("bench", "test/app", **env).returncode, 2
                )

    def test_busy_and_hot(self):
        self.assertEqual(self.run_task("bench", "test/../../outside").returncode, 2)
        self.assertEqual(
            self.run_task("bench", "test/app", MOCK_BUSY="1").returncode, 1
        )
        self.assertEqual(
            self.run_task(
                "bench", "test/app", MOCK_TEMP="80", BENCH_IDLE_TIMEOUT="0"
            ).returncode,
            1,
        )
        self.assertNotIn("app GPU", self.events())

    def test_signal_and_lock(self):
        process = subprocess.Popen(
            ["bash", str(TASKS / "bench"), "test/app"],
            cwd=self.root,
            env=self.env
            | {"MOCK_WAIT": "1", "BENCH_CLOCK": "1000", "BENCH_RESTORE_CLOCK": "off"},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            deadline = time.monotonic() + 10
            while "app GPU" not in self.events() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertIn("app GPU", self.events())
            self.assertEqual(self.run_task("bench", "test/app").returncode, 1)
            process.send_signal(signal.SIGTERM)
            process.communicate(timeout=5)
            self.assertEqual(process.returncode, 143)
            self.assertLess(
                self.events().index("app stopped"), self.events().index("-rgc")
            )
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()

    def test_compare_cleanup(self):
        source = self.root / "run.json"
        source.write_text(
            json.dumps(
                {
                    "benchmarks": [
                        {"name": "base", "states": [{"is_skipped": False}]},
                        {"name": "candidate", "states": [{"is_skipped": False}]},
                    ]
                }
            )
        )
        sentinel = self.root / ".compare-ref.json"
        sentinel.write_text("user data")
        for code in ("0", "7"):
            run = self.run_task("bench-compare", str(source), MOCK_EXIT=code)
            self.assertEqual(run.returncode, int(code), run.stderr)
            self.assertEqual(list(self.root.glob(".compare-*.??????.json")), [])
            self.assertEqual(sentinel.read_text(), "user data")
        self.assertNotEqual(
            self.run_task(
                "bench-compare", str(source), "missing", "candidate"
            ).returncode,
            0,
        )
        source.write_text("invalid json")
        self.assertNotEqual(self.run_task("bench-compare", str(source)).returncode, 0)


if __name__ == "__main__":
    unittest.main()
