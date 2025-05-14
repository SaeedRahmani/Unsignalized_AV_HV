import sys

if sys.version_info >= (3, 10):
    from src.intersection.waymo.core import filter_all_unsignalized_intersections
elif sys.version_info >= (3, 8):
    from src.intersection.lyft.core import filter_all_unsignalized_intersections

__all__ = ["filter_all_unsignalized_intersections"]