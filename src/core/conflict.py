import numpy as np
from enum import Enum
from .trajectory import Trajectory
from .road_user import RoadUser

class ConflictCategory(Enum):
    Cross = "Cross"
    Merge = "Merge"


class Conflict(object):
    """
    A conflict between two trajectories (leader and follower)
    """
    def __init__(
            self,
            pet: float,
            # leader
            leader_traj: Trajectory,
            leader_role: RoadUser,
            # follower
            follower_traj: Trajectory,
            follower_role: RoadUser,
            category: ConflictCategory,
            dataset_properties: dict = None,
    ):
        # Post Encroachment Time - a surrogate safety measure
        assert pet > 0, f"PET must be greater than 0, but got {pet}."
        self.pet = pet

        # Trajectories
        self.leader_traj = leader_traj
        self.follower_traj = follower_traj

        # Road user roles
        self.leader_role = leader_role
        self.follower_role = follower_role

        # Conflict category
        self.category = category

        # Dataset info
        self.dataset_properties = dataset_properties

    @property
    def ttc(self) -> np.ndarray:
        """ Time To Collision (TTC) """
        return np.zeros(2)

    @property
    def min_ttc(self):
        """ Minimum Time To Collision (minTTC) """
        return np.min(self.ttc)

    def __str__(self) -> str:
        return f"Conflict([{self.leader_role}-{self.follower_role}], {self.pet}s)"

    def __repr__(self) -> str:
        return self.__str__()

if __name__ == "__main__":
    leader_length = 500
    follower_length = 200

    conflict = Conflict(
        pet=5,
        leader_traj=Trajectory(coord_x=np.arange(leader_length), coord_y=np.arange(leader_length)),
        leader_role=RoadUser.HumanDrivenVehicle,
        follower_traj=Trajectory(coord_x=np.arange(follower_length), coord_y=np.arange(follower_length)),
        follower_role=RoadUser.AutomatedVehicle,
    )

    print(conflict)