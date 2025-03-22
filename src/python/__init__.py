try:
    import torch  # noqa: F401

    from ._bindings import *  # noqa: F401 F403

except ModuleNotFoundError as e:
    print(e)
    print("Bindings not installed")
