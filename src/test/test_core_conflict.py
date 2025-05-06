import numpy as np

from ..core.conflict import Conflict
from ..core.trajectory import Trajectory
from src.core.road_user import RoadUser

def test_core_conflict() -> None:
    leader_length = 100
    follower_length = 200

    conflict = Conflict(
        pet=5,
        leader_traj=Trajectory(coord_x=np.arange(leader_length), coord_y=np.arange(leader_length)),
        leader_role=RoadUser.HumanDrivenVehicle,
        follower_traj=Trajectory(coord_x=np.arange(follower_length), coord_y=np.arange(follower_length)),
        follower_role=RoadUser.AutomatedVehicle,
    )