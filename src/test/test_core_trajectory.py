import numpy as np
from ..core.trajectory import Trajectory

def test_core_trajectory():
    length = 100
    trajectory = Trajectory(coord_x=np.arange(length), coord_y=np.arange(length))
    assert trajectory.coords.shape == (length, 2)