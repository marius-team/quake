try:
    import torch
    from _bindings import *
except ModuleNotFoundError:
    print("Bindings not installed")