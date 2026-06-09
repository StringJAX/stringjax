"""Command-line interface for the StringJAX ecosystem.

Provides ``stringjax doctor`` (environment diagnostics) and
``stringjax versions`` (member-package versions).
"""

from __future__ import annotations

import argparse
from typing import Sequence

from ._version import __version__
from .doctor import MEMBERS, hints, report


def _cmd_doctor() -> int:
    info = report()
    width = max(len(k) for k in info) + 2
    print("StringJAX environment report")
    print("=" * 28)
    for key, value in info.items():
        print(f"{key:<{width}} {value}")
    suggestions = hints(info)
    if suggestions:
        print("\nSuggestions")
        print("-" * 11)
        for line in suggestions:
            print(f"  - {line}")
    return 0


def _cmd_versions() -> int:
    info = report()
    print(f"stringjax {__version__}")
    for pkg in MEMBERS:
        print(f"{pkg} {info.get(pkg)}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``stringjax`` console script."""
    parser = argparse.ArgumentParser(
        prog="stringjax",
        description="Helper utilities for the StringJAX ecosystem.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("doctor", help="Diagnose the Python/JAX environment.")
    sub.add_parser("versions", help="Show installed member-package versions.")

    args = parser.parse_args(argv)
    if args.command in (None, "doctor"):
        return _cmd_doctor()
    if args.command == "versions":
        return _cmd_versions()
    parser.print_help()
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
