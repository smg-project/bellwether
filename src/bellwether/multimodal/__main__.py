"""``python -m bellwether.multimodal``: the command group on its own, for an environment that has none of bellwether's
other dependencies, such as the engine's image where the recorder runs (``multimodal-record.yml``): the modules here
import the standard library, ``bellwether.jsonl`` and ``bellwether.storage``, which needs ``zstandard`` only to write
or read a compressed set."""

from __future__ import annotations

import argparse
import sys

from bellwether.multimodal import add_parser


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m bellwether.multimodal")
    add_parser(parser.add_subparsers(dest="command", metavar="command"))
    args = parser.parse_args(["multimodal", *(sys.argv[1:] if argv is None else argv)])
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
