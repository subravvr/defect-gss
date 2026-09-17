"""Geometric severity score (GSS) for defects in laser powder bed fusion.

    from gss import load, gss
    gss(load("ellipsoids", "g05")).value
"""
from .curvature import band_limit
from .mesh import Mesh, load
from .score import gss

__all__ = ["Mesh", "load", "gss", "band_limit"]
