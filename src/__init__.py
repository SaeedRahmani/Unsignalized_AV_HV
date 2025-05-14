import sys
from .conflict.conflict import Conflict
from .conflict.trajectory import Trajectory

__all__ = [
    'Conflict',
    'Trajectory',
]

if sys.version_info == (3, 10):
    from .intersection.waymo import StopSign
    __all__.append('StopSign')

