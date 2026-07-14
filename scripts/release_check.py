from __future__ import annotations

import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

from eval.openings import load_opening_suite

ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    if metadata["project"]["version"] != "0.2.0":
        raise SystemExit("pyproject.toml must declare version 0.2.0")
    suite = load_opening_suite(ROOT / "configs" / "openings" / "paired-v1.json")
    print(f"opening suite ok: {suite.suite_id} v{suite.version} {suite.sha256}")
    run([sys.executable, "-m", "cc_lite", "--version"])
    run([sys.executable, "-m", "ruff", "check", "."])
    run([sys.executable, "-m", "pytest"])
    for opponent in ("random", "material"):
        evidence_dir = ROOT / "evidence" / "debug-v1" / opponent
        if evidence_dir.is_dir():
            run(
                [
                    sys.executable,
                    "-m",
                    "eval.evidence",
                    "validate",
                    "--run-dir",
                    str(evidence_dir),
                ]
            )
    with tempfile.TemporaryDirectory(prefix="cc-lite-dist-") as output_dir:
        run([sys.executable, "-m", "build", "--outdir", output_dir])
        artifacts = sorted(str(path) for path in Path(output_dir).iterdir())
        run([sys.executable, "-m", "twine", "check", *artifacts])
    print("release check ok")


if __name__ == "__main__":
    main()
