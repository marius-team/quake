try:
    import torch

    """
    Quake Python API
    
    This module provides the Python bindings for the Quake dynamic index.
    """
    from ._bindings import *

except ModuleNotFoundError as e:
    print(e)
    print("Bindings not installed")