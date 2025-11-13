"""
Lean A-share simulation toolkit.
"""

from importlib import metadata

__all__ = ["__version__"]


def __getattr__(name: str):
    if name == "__version__":
        try:
            return metadata.version("lianghua")
        except metadata.PackageNotFoundError:
            return "0.0.0"
    raise AttributeError(name)
