try:
    import torch
    from ._bindings import *
except ModuleNotFoundError as e:
    print(e)
    print("Bindings not installed")