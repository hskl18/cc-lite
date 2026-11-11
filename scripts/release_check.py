from __future__ import annotations

import re
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
    version = metadata["project"]["version"]
    # Whole token, so v0.3.0rc1, v0.3.0+local or v0.3.0.1 in the README does not pass for 0.3.0.
    readme_version = rf"(?<![\w.])v{re.escape(version)}(?![\w+-]|\.\w)"
    if not re.search(readme_version, (ROOT / "README.md").read_text(encoding="utf-8")):
        raise SystemExit(f"README.md must name the release version v{version}")
    suite = load_opening_suite(ROOT / "configs" / "openings" / "paired-v1.json")
    print(f"opening suite ok: {suite.suite_id} v{suite.version} {suite.sha256}")
    run([sys.executable, "-m", "cc_lite", "--version"])
    run([sys.executable, "-m", "ruff", "check", "."])
    run([sys.executable, "-m", "pytest"])
    runs = (
        "random",
        "material",
        "vs-debug",
        "vs-v1",
        "vs-half-data",
        "vs-v3",
        "vs-v4",
        "pikafish-depth-1",
        "pikafish-depth-2",
        "pikafish-depth-4",
    )
    bundles = (
        "debug-v1",
        "baseline-v1",
        "distill-v1",
        "distill-v2",
        "distill-v1-long",
        "distill-v3",
        "distill-v3-half-data",
        "distill-v3-half-epochs",
        "distill-v4",
        "distill-v4-half-epochs",
        "distill-v4-large",
    )
    for bundle in bundles:
        for opponent in runs:
            evidence_dir = ROOT / "evidence" / bundle / opponent
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
