import numpy as np

from ..conflict.conflict import Conflict
from ..conflict.trajectory import Trajectory
from src.conflict.road_user import RoadUser

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