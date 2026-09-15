"""betedge - find and track positive expected value bets by de-vigging a
sharp book's prices and comparing them to softer books."""

__version__ = "0.1.0"

from . import pricing  # noqa: F401

__all__ = ["pricing", "__version__"]
