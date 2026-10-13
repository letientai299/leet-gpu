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
              'power.limit': '300', 'clocks.mem': '1215',
              'clocks.sm': os.environ.get('MOCK_SM_CLOCK', '1000')}
    for reason in ('hw_slowdown', 'hw_thermal_slowdown', 'sw_thermal_slowdown',
                   'hw_power_brake_slowdown'):
        values['clocks_event_reasons.' + reason] = (
            'Active' if os.environ.get('MOCK_UNHEALTHY') == reason else 'Not Active')
    print(', '.join(values[f] for f in field.split(',')))
elif '-rgc' in sys.argv and os.environ.get('MOCK_RESTORE_FAIL'):
    sys.exit(1)
elif '-lgc' in sys.argv and os.environ.get('MOCK_LOCK_FAIL'):
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
time.sleep(float(os.environ.get('MOCK_SLEEP', '0')))
print(json.dumps({'args': sys.argv[1:],
                  'locked': os.environ.get('LG_LOCKED_CLOCK_MHZ')}))
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
            if key.startswith(("BENCH_", "MOCK_", "GPU_", "CLOCK_SESSION", "LG_")):
                del self.env[key]
        self.env.update(
            PATH=f"{self.mock}:{os.environ['PATH']}",
            MISE_PROJECT_ROOT=str(self.root),
            LG_GPU_LOCK_DIR=str(self.root),
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
        run = self.run_task("bench", "test/app", "--json", "stdout", BENCH_CLOCK="off")
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout)["args"], ["--json", "stdout"])
        self.assertIn("app GPU-1234-abcd", self.events())
        self.assertNotIn("-lgc", self.events())
        self.assertNotIn("-rgc", self.events())

    def test_auto_clock_restore(self):
        run = self.run_task("bench", "test/app")
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("-lgc 1000,1000", self.events())
        self.assertIn("-rgc", self.events())
        self.assertIn("-pm 0", self.events())

    def test_lock_failure_runs_unlocked(self):
        run = self.run_task("bench", "test/app", MOCK_LOCK_FAIL="1")
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("running unlocked", run.stderr)
        self.assertIn("app GPU-1234-abcd", self.events())
        self.assertNotIn("-rgc", self.events())

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
            {"BENCH_RESTORE_CLOCK": "x"},
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

    def test_deploy_clock(self):
        run = self.run_task("deploy", "test/app", "--bench")
        self.assertEqual(run.returncode, 0, run.stderr)
        events = self.events()
        self.assertLess(events.index("-lgc 1000,1000"), events.index("app "))
        self.assertLess(events.index("app "), events.index("-rgc"))
        self.assertIn("app GPU-1234-abcd", events)
        args = json.loads(run.stdout)["args"]
        self.assertEqual(args[-2:], ["--throttle-threshold", "69"])

        for args, env in ((["x"], {}), (["--bench"], {"GPU_CLOCK": "off"})):
            with self.subTest(args=args, env=env):
                self.log.unlink()
                run = self.run_task("deploy", "test/app", *args, **env)
                self.assertEqual(run.returncode, 0, run.stderr)
                self.assertNotIn("-lgc", self.events())
                self.assertNotIn("-rgc", self.events())
                self.assertNotIn("--throttle-threshold", run.stdout)

        run = self.run_task("deploy", "test/app", "--bench", MOCK_LOCK_FAIL="1")
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("running unlocked", run.stderr)

    def test_throttle_lowers_ceiling(self):
        ceiling = self.root / "lg-gpu-GPU-1234-abcd.clock"
        sampled = {
            "LG_CLOCK_SAMPLE_SECONDS": "0.01",
            "MOCK_SLEEP": "1",
            "TMPDIR": str(self.root),
        }
        run = self.run_task("deploy", "test/app", "--bench", **sampled)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertNotIn("next runs lock", run.stderr)
        self.assertFalse(ceiling.exists())

        run = self.run_task(
            "deploy", "test/app", "--bench", MOCK_SM_CLOCK="985", **sampled
        )
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout)["locked"], "1000")
        self.assertIn("next runs lock to 900MHz", run.stderr)
        self.assertEqual(ceiling.read_text().strip(), "900")
        self.assertEqual(list(self.root.glob("lg-clock.*")), [])

        run = self.run_task("deploy", "test/app", "--bench")
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout)["locked"], "900")
        self.assertEqual(ceiling.read_text().strip(), "900")

        run = self.run_task("deploy", "test/app", "--bench", GPU_CLOCK="1200")
        self.assertEqual(json.loads(run.stdout)["locked"], "1200")

        run = self.run_task("bench", "test/app")
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout)["locked"], "900")

    def test_unhealthy_gpu_refused(self):
        for task, args in (("deploy", ["--bench"]), ("bench", [])):
            with self.subTest(task=task):
                self.log.unlink(missing_ok=True)
                run = self.run_task(
                    task, "test/app", *args, MOCK_UNHEALTHY="hw_thermal_slowdown"
                )
                self.assertNotEqual(run.returncode, 0)
                self.assertIn("hw_thermal_slowdown active", run.stderr)
                self.assertNotIn("app ", self.events())
        run = self.run_task("deploy", "test/app", "x", MOCK_UNHEALTHY="hw_slowdown")
        self.assertEqual(run.returncode, 0, run.stderr)

    def test_deploy_criterion(self):
        entropy = ["--stopping-criterion", "entropy"]
        cases = (
            (["x"], ["x"]),
            (["--bench"], ["--bench", *entropy]),
            (["--bench", "--quiet"], ["--bench", *entropy, "--quiet"]),
            (["--bench", "--max-noise", "1"], ["--bench", "--max-noise", "1"]),
            (["--bench", "--help"], ["--bench", "--help"]),
            (
                ["--bench", "--stopping-criterion", "stdrel"],
                ["--bench", "--stopping-criterion", "stdrel"],
            ),
        )
        for args, expected in cases:
            with self.subTest(args=args):
                run = self.run_task("deploy", "test/app", *args, GPU_CLOCK="off")
                self.assertEqual(run.returncode, 0, run.stderr)
                self.assertEqual(json.loads(run.stdout)["args"], expected)

    def test_deploy_bench_app(self):
        self.executable(self.root / "bin/test/app.bench", APP)
        run = self.run_task("deploy", "test/app.bench", "--quiet", "--bench")
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(
            json.loads(run.stdout)["args"],
            [
                "--stopping-criterion",
                "entropy",
                "--quiet",
                "--bench",
                "--throttle-threshold",
                "69",
            ],
        )
        self.assertIn("-lgc 1000,1000", self.events())

    def test_deploy_session(self):
        session = self.root / "session"
        session.touch()
        for env in ({}, {"GPU_CLOCK": "900"}, {"GPU_INDEX": "1"}):
            run = self.run_task(
                "deploy", "test/app", "--bench", CLOCK_SESSION=str(session), **env
            )
            self.assertEqual(run.returncode, 0, run.stderr)
        events = self.events()
        self.assertEqual(events.count("-lgc"), 3)
        self.assertEqual(events.count("-rgc"), 1)
        self.assertLess(events.index("-lgc 900,900"), events.index("-rgc"))
        self.assertLess(events.index("-rgc"), events.rindex("-lgc"))
        self.assertEqual(session.read_text().split(), ["local", "1"])

    def test_deploy_waits_for_gpu(self):
        holder = subprocess.Popen(
            ["flock", str(self.root / "lg-gpu-GPU-1234-abcd.lock"), "sleep", "2"]
        )
        try:
            time.sleep(0.3)
            start = time.monotonic()
            run = self.run_task("deploy", "test/app", "x")
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn("busy; waiting", run.stderr)
            self.assertGreater(time.monotonic() - start, 1.0)
        finally:
            holder.kill()
            holder.wait()

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

    def test_lock_holder_and_timeout(self):
        lock = self.root / "lg-gpu-GPU-1234-abcd.lock"
        process = subprocess.Popen(
            [
                "bash",
                str(ROOT / ".config/mise/lib/gpu.sh"),
                "run",
                "0",
                "off",
                "85",
                "0",
                "--",
                "bin/test/app",
                "x",
            ],
            cwd=self.root,
            env=self.env | {"MOCK_WAIT": "1", "LG_GPU_MAIN": "1"},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            deadline = time.monotonic() + 10
            while "app GPU" not in self.events() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertIn("app GPU", self.events())
            holder = lock.read_text().strip()
            self.assertIn("user=", holder)
            self.assertIn("app=bin/test/app", holder)
            self.assertRegex(holder, r"started=\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
            self.log.unlink()
            start = time.monotonic()
            run = self.run_task(
                "deploy", "test/app", "--bench", LG_GPU_LOCK_TIMEOUT="0.2"
            )
            self.assertEqual(run.returncode, 1, run.stderr)
            self.assertGreaterEqual(time.monotonic() - start, 0.2)
            self.assertIn(holder, run.stderr)
            self.assertIn("lock wait timed out", run.stderr)
            self.assertEqual(lock.read_text().strip(), holder)
            self.assertNotIn("app ", self.events())
            self.assertNotIn("-lgc", self.events())
            run = self.run_task("bench", "test/app")
            self.assertEqual(run.returncode, 1, run.stderr)
            self.assertIn(holder, run.stderr)
        finally:
            process.send_signal(signal.SIGTERM)
            process.communicate(timeout=5)
        self.assertEqual(process.returncode, 143)
        self.assertEqual(lock.read_text(), "")
        self.assertEqual(self.run_task("deploy", "test/app", "x").returncode, 0)

    def test_invalid_lock_timeout(self):
        for timeout in ("-1", "bad"):
            with self.subTest(timeout=timeout):
                run = self.run_task(
                    "deploy", "test/app", "--bench", LG_GPU_LOCK_TIMEOUT=timeout
                )
                self.assertEqual(run.returncode, 2, run.stderr)
                self.assertNotIn("app ", self.events())
                self.assertNotIn("-lgc", self.events())

    def test_remote_lock_timeout(self):
        self.executable(
            self.mock / "ssh",
            '#!/bin/bash\nbash -c "${!#}"\n',
        )
        run = subprocess.run(
            [
                "bash",
                "-c",
                'source "$1"; run_gpu remote run 0 off 85 0 -- bin/test/app x',
                "test",
                str(ROOT / ".config/mise/lib/target.sh"),
            ],
            cwd=self.root,
            env=self.env | {"LG_GPU_LOCK_TIMEOUT": "bad"},
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        self.assertEqual(run.returncode, 2, run.stderr)
        self.assertIn("invalid LG_GPU_LOCK_TIMEOUT: bad", run.stderr)
        self.assertNotIn("app ", self.events())


if __name__ == "__main__":
    unittest.main()
