import sys

if sys.version_info >= (3, 10):
    from .waymo_loader import waymo_loader
    __all__ = ["waymo_loader"]
elif sys.version_info >= (3, 8):
    from .lyft_loader import lyft_loader
    __all__ = ["lyft_loader"]