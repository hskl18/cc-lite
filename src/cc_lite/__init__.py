import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

try:
    __version__ = version("cc-lite")
except PackageNotFoundError:
    metadata_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    __version__ = tomllib.loads(metadata_path.read_text(encoding="utf-8"))["project"]["version"]

__all__ = ["__version__"]
