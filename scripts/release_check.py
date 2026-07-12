from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    run([sys.executable, "-m", "ruff", "check", "."])
    run([sys.executable, "-m", "pytest"])
    with tempfile.TemporaryDirectory(prefix="cc-lite-dist-") as output_dir:
        run([sys.executable, "-m", "build", "--outdir", output_dir])
        artifacts = sorted(str(path) for path in Path(output_dir).iterdir())
        run([sys.executable, "-m", "twine", "check", *artifacts])
    print("release check ok")


if __name__ == "__main__":
    main()
