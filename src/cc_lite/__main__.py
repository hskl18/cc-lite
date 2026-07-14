from __future__ import annotations

import argparse

from cc_lite import __version__


def main() -> None:
    parser = argparse.ArgumentParser(description="cc-lite Xiangqi research workbench")
    parser.add_argument("--version", action="version", version=f"cc-lite {__version__}")
    parser.parse_args()
    parser.print_help()


if __name__ == "__main__":
    main()
