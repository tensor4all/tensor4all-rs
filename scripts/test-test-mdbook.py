#!/usr/bin/env python3
"""Exercise standalone preparation and exact extern reuse without compiling Rust."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class MdbookPreparationTests(unittest.TestCase):
    def check_preparation(self, supplied_log):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            commands = directory / "bin"
            commands.mkdir()
            extern = "--crate-name book_tests --extern example=/exact/libexample.rlib"
            scripts = {
                "cargo": f'#!/bin/sh\necho called > "{directory}/cargo-called"\necho "Running {extern}"\n',
                "rustup": f'#!/bin/sh\necho "{commands}/real-rustdoc"\n',
                "pkg-config": '#!/bin/sh\nexit 1\n',
                "mdbook": '#!/bin/sh\nexec rustdoc --test chapter.rs\n',
                "real-rustdoc": f'#!/usr/bin/env python3\nimport json,sys\nopen("{directory}/args.json", "w").write(json.dumps(sys.argv[1:]))\n',
            }
            for name, text in scripts.items():
                path = commands / name
                path.write_text(text)
                path.chmod(0o755)
            env = {**os.environ, "PATH": str(commands) + os.pathsep + os.environ["PATH"]}
            env.pop("TENSOR4ALL_RUSTDOC_LOG", None)
            if supplied_log:
                log = directory / "rustdoc.log"
                log.write_text(f"[1.234s] Running {extern}\n")
                env["TENSOR4ALL_RUSTDOC_LOG"] = str(log)
            result = subprocess.run(["bash", str(Path(__file__).with_name("test-mdbook.sh"))],
                                    env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads((directory / "args.json").read_text()),
                             ["--extern", "example=/exact/libexample.rlib", "--test", "chapter.rs"])
            self.assertEqual((directory / "cargo-called").exists(), not supplied_log)
            self.assertIn("mdBook preparation:", result.stdout)

    def test_timestamped_log_reuses_exact_extern_without_cargo(self):
        self.check_preparation(supplied_log=True)

    def test_standalone_call_prepares_a_rustdoc_log(self):
        self.check_preparation(supplied_log=False)

    def test_explicit_missing_log_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(["bash", str(Path(__file__).with_name("test-mdbook.sh"))],
                                    env={**os.environ, "TENSOR4ALL_RUSTDOC_LOG": directory + "/missing"},
                                    text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("rustdoc log does not exist:", result.stderr)


if __name__ == "__main__":
    unittest.main()
