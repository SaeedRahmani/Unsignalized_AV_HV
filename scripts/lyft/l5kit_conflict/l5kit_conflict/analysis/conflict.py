import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString

from .trajectory import Trajectory
from .utils import three_sigma_smoothing, compute_position_based_velocities

class Conflict:
    def __init__(
            self,
            first_trajectory,
            second_trajectory,
            is_first_AV,
            is_second_AV,
            first_time_at_conflict,
            second_time_at_conflict,
            first_id,
            second_id,
            PET,
            category,
            direction,
            dataset,
            scene_indices,
    ):
        self.first_trajectory = Trajectory(xy=first_trajectory.trajectory_xy, t=first_trajectory.trajectory_t)
        self.second_trajectory = Trajectory(xy=second_trajectory.trajectory_xy, t=second_trajectory.trajectory_t)

        self.is_first_AV = is_first_AV
        self.is_second_AV = is_second_AV

        self.PET = PET / 10  # seconds

        self.first_time_at_conflict = first_time_at_conflict
        self.second_time_at_conflict = second_time_at_conflict
        self.first_id = first_id
        self.second_id = second_id

        if category not in ["cross", "merge"]:
            raise ValueError()
        self.category = category  # cross or merge
        self.direction = direction
        self.dataset = dataset
        self.scene_indices = scene_indices


    def __repr__(self):
        return f"""Conflict(
            {"AV" if self.is_first_AV else "HV"}: {self.first_trajectory},
            {"AV" if self.is_second_AV else "HV"}: {self.second_trajectory}
        )"""

    def plot(self, savefig_path=None, show_full_trajectory=False, show_figure: bool = True):
        plt.figure(figsize=(3, 3))
        if show_full_trajectory:
            plt.xlim([-850, -550])
            plt.ylim([-1150, -850])
            # first trajectory
            plt.scatter(
                x=self.first_trajectory.xy[1:, 0], y=self.first_trajectory.xy[1:, 1], c=self.first_trajectory.v,
                cmap="Reds", s=1, vmin=0, vmax=15, alpha=0.95
            )
            plt.scatter(
                x=self.second_trajectory.xy[1:, 0], y=self.second_trajectory.xy[1:, 1], c=self.second_trajectory.v,
                cmap="Greens", s=1, vmin=0, vmax=15, alpha=0.95
            )
            plt.title(f"Full trajectory {self.category} {self.direction}", fontdict={"fontsize": 6})
        else:
            plt.xlim([Trajectory.LEFT_TOP[0], Trajectory.RIGHT_TOP[0]])
            plt.ylim([Trajectory.LEFT_BOTTOM[1], Trajectory.LEFT_TOP[1]])
            plt.scatter(
                x=self.first_trajectory.xy_Tjunction[1:, 0], y=self.first_trajectory.xy_Tjunction[1:, 1],
                c=self.first_trajectory.v_Tjunction,
                cmap="Reds", s=1, vmin=0, vmax=15, alpha=0.95
            )
            plt.scatter(
                x=self.second_trajectory.xy_Tjunction[1:, 0], y=self.second_trajectory.xy_Tjunction[1:, 1],
                c=self.second_trajectory.v_Tjunction,
                cmap="Greens", s=1, vmin=0, vmax=15, alpha=0.95
            )
            plt.title(f"Trajectory in T-junction {self.scene_indices} {self.PET:.1f}", fontdict={"fontsize": 6})
        # plt.colorbar(label="speed")
        if savefig_path is not None and isinstance(savefig_path, str):
            plt.savefig(savefig_path, dpi=300, bbox_inches='tight')
        if show_figure:
            plt.show()
        plt.close()
        
    @property
    def first_veh_speed_at_conflict(self):
        index = np.where(self.first_trajectory.t == self.first_time_at_conflict)[0]
        speed = self.first_trajectory.v[index-2]
        if speed.shape == (2,):
            speed = speed[0]
        return float(speed)

    @property
    def second_veh_speed_at_conflict(self):
        index = np.where(self.second_trajectory.t == self.second_time_at_conflict)[0]
        return float(self.second_trajectory.v[index-2])

    @property
    def first_veh_average_speed(self):
        return self.first_trajectory.L / self.first_trajectory.T 

    @property
    def second_veh_average_speed(self):
        return self.second_trajectory.L / self.second_trajectory.T 
    
    @property
    def minimum_TTC(self):
        TTCs = self.TTCs
        return min(TTCs) if len(TTCs) > 0 else None 

    @property
    def TTCs(self):
        """ calculate TTC """      
        # get leader vehicle's trajectory 
        # between follower vehicle first appear and leader vehicle arrives at conflict point
        firstTimeInArea = max(self.first_trajectory.firstTimeInArea, self.second_trajectory.firstTimeInArea)
        
        first_trajectory_before_collision = self.first_trajectory.xy[
            np.where((firstTimeInArea < self.first_trajectory.t) &
                    (self.first_trajectory.t < self.first_time_at_conflict))]
        
        # get follower vehicle's trajectory 
        # between follower vehicle first appear and leader vehicle arrives at conflict point
        second_trajectory_before_collision = self.second_trajectory.xy[
            np.where((firstTimeInArea < self.second_trajectory.t) &
                    (self.second_trajectory.t < self.second_time_at_conflict))]
        # if first_trajectory_before_collision.shape != second_trajectory_before_collision.shape:
        #         print(f"{first_trajectory_before_collision.shape}, {second_trajectory_before_collision.shape}")
        #         shorter = min(first_trajectory_before_collision.shape[0], second_trajectory_before_collision.shape[0])
        #         first_trajectory_before_collision = first_trajectory_before_collision[:shorter,:]
        #         second_trajectory_before_collision = second_trajectory_before_collision[:shorter,:]
        
        TTCs = []
        len_traj = first_trajectory_before_collision.shape[0]
        first_speeds_before_collision = np.sqrt(np.sum(np.diff(first_trajectory_before_collision, axis=0) ** 2, axis=1)) / 0.1
        second_speeds_before_collision = np.sqrt(np.sum(np.diff(second_trajectory_before_collision, axis=0) ** 2, axis=1)) / 0.1
        
        for i in range(len_traj-2):
            length_trajectory = LineString(second_trajectory_before_collision[i:,:]).length
            if self.category == "merge" and (second_speeds_before_collision[i] - first_speeds_before_collision[i]) > 0:
                TTCs.append(length_trajectory / (second_speeds_before_collision[i] - first_speeds_before_collision[i]))
            elif self.category == "cross":
                TTCs.append(length_trajectory / second_speeds_before_collision[i])
        return TTCs
 
    def req_dec(self, truncate: bool = False, n_frames_truncated: int = 10):
        """ Return the req decs of the second vehicle """
        first_xyt = np.hstack([self.first_trajectory.xy, self.first_trajectory.t.reshape((-1,1))])
        second_xyt = np.hstack([self.second_trajectory.xy, self.second_trajectory.t.reshape((-1,1))])
        
        # extract the trajectory within the study rectangle
        first_xyt = first_xyt[
            (first_xyt[:, 0] <= Trajectory.X_MAX) &
            (first_xyt[:, 0] >= Trajectory.X_MIN) &
            (first_xyt[:, 1] <= Trajectory.Y_MAX) &
            (first_xyt[:, 1] >= Trajectory.Y_MIN)
        ]
        second_xyt = second_xyt[
            (second_xyt[:, 0] <= Trajectory.X_MAX) &
            (second_xyt[:, 0] >= Trajectory.X_MIN) &
            (second_xyt[:, 1] <= Trajectory.Y_MAX) &
            (second_xyt[:, 1] >= Trajectory.Y_MIN)
        ]
        
        # take the first timestamp that two vehicles co-exist
        coexist_start_time = max(first_xyt[0, 2], second_xyt[0, 2])
        first_xyt = first_xyt[
            (self.first_time_at_conflict > first_xyt[:, 2]) &
            (first_xyt[:, 2] >= coexist_start_time)
        ]    
        second_xyt = second_xyt[
            (self.first_time_at_conflict > second_xyt[:, 2]) &
            (second_xyt[:, 2] >= coexist_start_time)
        ]
        
        if truncate:
            if first_xyt.shape[0] > n_frames_truncated:
                first_xyt = first_xyt[n_frames_truncated:,]
            else: 
                return False, None
            if second_xyt.shape[0] > n_frames_truncated:
                second_xyt = second_xyt[n_frames_truncated:,]
            else:
                return False, None
        
        if first_xyt.shape[0] !=  second_xyt.shape[0]:
            # print("Non-equal two trajectories", first_xyt.shape, second_xyt.shape)
            len_traj = min(first_xyt.shape[0], second_xyt.shape[0]) - 2
        else:
            len_traj = first_xyt.shape[0] - 2      

        if len_traj == 0:
            return False, None
        
        # collect the required decelerations of the second vehicle 
        req_decs = []
        for jdx in range(len_traj):
            # calculate the RD and Max RD for the second vehicle
            len_second_traj = LineString(second_xyt[jdx:, :2]).length
            second_speed = three_sigma_smoothing(compute_position_based_velocities(second_xyt[jdx:jdx+3, :2]))[0]
            req_dec = 0.5 * second_speed**2 / len_second_traj
            req_decs.append(req_dec)
        req_decs = np.array(req_decs)
        req_decs = req_decs[req_decs <= 8]
        if len(req_decs) == 0:
            return False, None
        else:
            return True, req_decs
    