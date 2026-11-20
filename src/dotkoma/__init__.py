"""Dotkoma: 16x16 pixel-art loop dataset pipeline."""

__version__ = "0.1.0"


def main() -> None:
    from .cli import main as _main

    _main()
