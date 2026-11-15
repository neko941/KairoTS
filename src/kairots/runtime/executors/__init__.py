"""Lazy execution backend discovery."""

from kairots.registry import surface

get, names, __getattr__, __dir__ = surface(package=__name__)

__all__ = ["get", "names"]
