#!/usr/bin/env python3
"""Apply the diagnostic-only patch to the pinned, downloaded camera component."""
from pathlib import Path
import subprocess

repo = Path(__file__).resolve().parents[3]
patch = Path(__file__).with_name("camera-diagnostics.patch").resolve()
base = ["git", "apply", "--directory=firmware/esp32-s3-cam/managed_components/espressif__esp32-camera"]
def run(*args):
    return subprocess.run(base + list(args) + [str(patch)], cwd=repo,
                          capture_output=True, text=True)

if run("--reverse", "--check").returncode == 0:
    print("Camera diagnostic patch already applied")
else:
    check = run("--check")
    if check.returncode:
        raise SystemExit(check.stderr)
    result = run()
    if result.returncode:
        raise SystemExit(result.stderr)
    print("Applied camera diagnostic patch")
