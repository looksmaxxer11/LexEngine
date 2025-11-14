"""
Compatibility CLI wrapper for src.orchestrator
"""

from typing import Optional, List

from src.orchestrator import parse_args as _parse_args, main as _main


def parse_args(argv: Optional[List[str]] = None):
    return _parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    return _main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
