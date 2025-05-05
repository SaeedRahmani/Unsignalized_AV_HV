import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Polygon, Point
from typing import List, Tuple
import scipy.signal
from scipy.signal import butter, filtfilt


class ConflictLyft():
    """
    Define and analyse the conflict found in the lyft level 5 dataset
    """
    def __init__(
        self,
        leader_id: int,
        follower_id: int,
        leader_time_at_conflict: float,
        follower_time_at_conflict: float,
        leader_is_av: int,
        follower_is_av: int,
        PET: float,
        conflict_type,
        # states:
        leader_traj: np.array,
        follower_traj: np.array,
        leader_traj_timestamp: np.array,
        follower_traj_timestamp: np.array,
        leader_traj_speed: np.array,
        follower_traj_speed: np.array,
        junction_id: int = 1,
    ):      
        self.leader_id, self.follower_id = leader_id, follower_id
        self.leader_time_at_conflict, self.follower_time_at_conflict = leader_time_at_conflict * 0.1, follower_time_at_conflict * 0.1  
        self.leader_is_av, self.follower_is_av = leader_is_av, follower_is_av
        self.leader_type = "AV" if self.leader_is_av else "HV"
        self.follower_type = "AV" if self.follower_is_av else "HV"
        self.vehicle_order = self.leader_type + '-' + self.follower_type

        self.PET = PET
        self.conflict_type = conflict_type

        # trajectory as np.array
        self.leader_traj = leader_traj
        self.follower_traj = follower_traj
        assert self.leader_traj.shape[1] == 2
        assert self.follower_traj.shape[1] == 2

        # trajectory linestring
        self.leader_traj_lineString = LineString(coordinates=self.leader_traj)
        self.follower_traj_lineString = LineString(coordinates=self.follower_traj)

        # trajectory length
        self.leader_traj_length = self.leader_traj_lineString.length
        self.follower_traj_length = self.follower_traj_lineString.length

        # trajectory time
        self.leader_T = self.leader_traj.shape[0] * 0.1
        self.follower_T = self.follower_traj.shape[0] * 0.1

        # trajectory timestamps
        self.leader_traj_timestamp = leader_traj_timestamp * 0.1
        self.follower_traj_timestamp = follower_traj_timestamp * 0.1   
        assert self.leader_traj_timestamp.shape[0] == self.leader_traj.shape[0]
        assert self.follower_traj_timestamp.shape[0] == self.follower_traj.shape[0]

        # trajectory speeds
        self.leader_traj_speed = leader_traj_speed
        self.follower_traj_speed = follower_traj_speed
        assert self.leader_traj_speed.shape[0] == self.leader_traj.shape[0]
        assert self.follower_traj_speed.shape[0] == self.follower_traj.shape[0]

        def remove_outliers(data, jump_threshold=2, offset:int=10):
            smoothed_data = data.copy()
        
            # remove the jumps in the beginning of speed profile
            for index in [4,3,2,1,0]:
                if abs(smoothed_data[index] - smoothed_data[index: index + offset].mean()) > jump_threshold:
                    smoothed_data[index] = smoothed_data[index: index + offset].mean()
        
            # remove the jumps in the middle of speed profile
            for i in range(1, len(smoothed_data) - 2):
                if abs(smoothed_data[i] - smoothed_data[i-1]) > jump_threshold and \
                   abs(smoothed_data[i] - smoothed_data[i+2]) > jump_threshold:
                    smoothed_data[i] = (smoothed_data[i-1] + smoothed_data[i+2]) / 2
            return smoothed_data
        
        def robust_low_pass_filter(data, cutoff=0.5, fs=10.0, order=4, jump_threshold=1):
            # Remove outliers
            data_no_outliers = remove_outliers(data, jump_threshold)
            # Apply low-pass filter
            nyquist = 0.5 * fs
            normal_cutoff = cutoff / nyquist
            b, a = butter(order, normal_cutoff, btype='low', analog=False)
            y = filtfilt(b, a, data_no_outliers)
            return y

        # smooth the speeds
        self.leader_traj_speed = robust_low_pass_filter(self.leader_traj_speed, cutoff=0.5, fs=10.0, order=4, jump_threshold=1)
        self.follower_traj_speed = robust_low_pass_filter(self.follower_traj_speed, cutoff=0.5, fs=10.0, order=4, jump_threshold=1)

        assert 1 <= junction_id <= 2 
        self.junction_id = junction_id

        if self.junction_id == 1:
            LEFT_TOP = (-850, -875)
            LEFT_BOTTOM = (-850, -925)
            RIGHT_TOP = (-700, -875)
            RIGHT_BOTTOM = (-700, -925)
        else:
            LEFT_TOP 	 = (-200, -865)
            LEFT_BOTTOM  = (-200, -915)
            RIGHT_TOP 	 = ( -50, -865)
            RIGHT_BOTTOM = ( -50, -915)
            
        intersection_coords = (LEFT_TOP, LEFT_BOTTOM, RIGHT_BOTTOM, RIGHT_TOP, LEFT_TOP)
        intersection_area = Polygon(intersection_coords)        
        leader_index_in_intersection = list()
        for index, t in enumerate(self.leader_traj_timestamp):
            coord_t = self.leader_traj[index,:]
            if np.isnan(coord_t[0]) or np.isnan(coord_t[1]): 
                pass
            else:
                if Point((coord_t[0], coord_t[1])).within(intersection_area):
                    leader_index_in_intersection.append(index)  
        self.leader_traj_inIntersection = self.leader_traj[leader_index_in_intersection,:]
        self.leader_traj_speed_inIntersection = self.leader_traj_speed[leader_index_in_intersection]
        self.leader_traj_timestamp_inIntersection = self.leader_traj_timestamp[leader_index_in_intersection]

        follower_index_in_intersection = list()
        for index, t in enumerate(self.follower_traj_timestamp):
            coord_t = self.follower_traj[index,:]
            if np.isnan(coord_t[0]) or np.isnan(coord_t[1]): 
                pass
            else:
                if Point((coord_t[0], coord_t[1])).within(intersection_area):
                    follower_index_in_intersection.append(index)  
        self.follower_traj_inIntersection = self.follower_traj[follower_index_in_intersection,:]
        self.follower_traj_speed_inIntersection = self.follower_traj_speed[follower_index_in_intersection]
        self.follower_traj_timestamp_inIntersection = self.follower_traj_timestamp[follower_index_in_intersection]
        
        """ process the states before conflict point """
        self.process_before_conflict()
    
    def process_before_conflict(self,):
        """ process the states before conflict point """
        # get the start_index and end_index
        leader_appear_time = self.leader_traj_timestamp[0]
        follower_appear_time = self.follower_traj_timestamp[0]
        co_exist_time = max(leader_appear_time, follower_appear_time)        

        if follower_appear_time > self.leader_time_at_conflict:
            # Special case: when the leader reached conflict point, the follower was not detected 
            # TTC, required deceleration cannot be calculated
            self.two_vehicles_co_exist = False

        else:
            self.two_vehicles_co_exist = True
            self.leader_beforeConflict_timestamps = self.leader_traj_timestamp[np.where(
                (co_exist_time <= self.leader_traj_timestamp) & (self.leader_traj_timestamp <= self.leader_time_at_conflict)
            )]
            self.leader_beforeConflict_coordinates = self.leader_traj[np.where(
                (co_exist_time <= self.leader_traj_timestamp) & (self.leader_traj_timestamp <= self.leader_time_at_conflict)
            )]
            self.leader_beforeConflict_speeds = self.leader_traj_speed[np.where(
                (co_exist_time <= self.leader_traj_timestamp) & (self.leader_traj_timestamp <= self.leader_time_at_conflict)
            )]
            
            self.follower_beforeConflict_timestamps = self.follower_traj_timestamp[np.where(
                (co_exist_time <= self.follower_traj_timestamp) & (self.follower_traj_timestamp <= self.leader_time_at_conflict)
            )]
            self.follower_beforeConflict_coordinates = self.follower_traj[np.where(
                (co_exist_time <= self.follower_traj_timestamp) & (self.follower_traj_timestamp <= self.follower_time_at_conflict)
            )]
            self.follower_beforeConflict_speeds = self.follower_traj_speed[np.where(
                (co_exist_time <= self.follower_traj_timestamp) & (self.follower_traj_timestamp <= self.leader_time_at_conflict)
            )]    
            
            self.follower_beforeConflict_speeds_followerReach = self.follower_traj_speed_inIntersection[np.where(
                (co_exist_time <= self.follower_traj_timestamp_inIntersection) & (self.follower_traj_timestamp_inIntersection <= self.follower_time_at_conflict)
            )]  
            
            # buffers to store metrics at each timestamp
            TTCs, req_deccelerations, TAs = list(), list(), list()
            leader_trajectory_lengths, follower_trajectory_lengths = list(), list()
            leader_speeds, follower_speeds = list(), list()
            
            # use the shorter one as the timestamps for simpilicity                
            if len(self.leader_beforeConflict_timestamps) >= len(self.follower_beforeConflict_timestamps): 
                self.ttc_timestamps = self.follower_beforeConflict_timestamps
            else:
                self.ttc_timestamps = self.leader_beforeConflict_timestamps

            # calculate metrics at each timestamp
            for t in self.ttc_timestamps:
                # get indices
                leader_index = np.where(t == self.leader_beforeConflict_timestamps)[0]
                if leader_index.shape[0] > 1:
                    leader_index = int(leader_index[0])
                else:
                    leader_index = int(leader_index)
                follower_index = int(np.where(t == self.follower_beforeConflict_timestamps)[0])
                
                # get length
                leader_sub_trajectory = self.leader_beforeConflict_coordinates[leader_index:,:]
                follower_sub_trajectory = self.follower_beforeConflict_coordinates[follower_index:,:]
                
                # check trajectory validity
                if follower_sub_trajectory.shape[0] == 1 or leader_sub_trajectory.shape[0] == 1:
                    # in case that the sub-trajectory is just a Point, thus cannot construct a LineString
                    req_deccelerations.append(np.NaN)
                    TTCs.append(np.NaN)
                    TAs.append(np.NaN)
                    leader_trajectory_lengths.append(np.NaN)
                    follower_trajectory_lengths.append(np.NaN)
                    leader_speeds.append(np.NaN)
                    follower_speeds.append(np.NaN)
                    continue
                
                # calculate the lengths of the trajectories
                leader_trajectory_length = LineString(leader_sub_trajectory).length
                leader_trajectory_lengths.append(leader_trajectory_length)
                follower_trajectory_length = LineString(follower_sub_trajectory).length
                follower_trajectory_lengths.append(follower_trajectory_length)

                # get speeds
                leader_speeds.append(self.leader_beforeConflict_speeds[leader_index])
                follower_speeds.append(self.follower_beforeConflict_speeds[follower_index])
                
                # calculate required deceleration
                req_deccelerations.append(0.5 * self.follower_beforeConflict_speeds[follower_index]**2 / follower_trajectory_length)
                
                if follower_trajectory_length > 0.001:
                    # calculate time-to-collision
                    if self.conflict_type == "CROSS":
                        TTCs.append(follower_trajectory_length / np.abs(self.follower_beforeConflict_speeds[follower_index]))
                    elif self.conflict_type == "MERGE":
                        if (self.follower_beforeConflict_speeds[follower_index] - self.leader_beforeConflict_speeds[leader_index]) > 0.01:
                            # only when the difference between follower and leader is positive, calculate TTC
                            # TTCs.append(length / (follower_ttc_speeds[follower_index]))
                            TTCs.append(
                                follower_trajectory_length / (
                                    self.follower_beforeConflict_speeds[follower_index] - self.leader_beforeConflict_speeds[leader_index]
                                )
                            )
                        else:
                            TTCs.append(np.NaN) 
                else:
                    TTCs.append(np.NaN)
                        
                # calculate time-advantage
                TAs.append(
                    follower_trajectory_length / self.follower_beforeConflict_speeds[follower_index] - \
                    leader_trajectory_length / self.leader_beforeConflict_speeds[leader_index]
                )
            
            self.states_before_conflictPoint = np.stack(
                [
                    self.ttc_timestamps, 
                    np.array(TTCs),
                    np.array(req_deccelerations),
                    np.array(TAs),
                    np.array(leader_trajectory_lengths), 
                    np.array(follower_trajectory_lengths),
                    np.array(leader_speeds), 
                    np.array(follower_speeds)
                ], axis=1)

    
    def plot_trajectory(
        self, 
        leader_before_conflict: bool, follower_before_conflict: bool
    ):
        """
        visualize the trajectories involved in this conflict.
        """
        # constants
        FIGURE_SIZE = 4
        LEADER_COLOR = "r"
        FOLLOWER_COLOR = "g"
        SCATTER_SIZE = 1

        # trajectory plot
        fig = plt.figure(figsize=(FIGURE_SIZE, FIGURE_SIZE))
        
        leader_end_index = int(np.where(self.leader_time_at_conflict == self.leader_traj_timestamp)[0]) if leader_before_conflict else self.leader_traj.shape[0]
        follower_end_index = int(np.where(self.follower_time_at_conflict == self.follower_traj_timestamp)[0]) if follower_before_conflict else self.follower_traj.shape[0]
        
        plt.scatter(x=self.leader_traj[:leader_end_index,0], y=self.leader_traj[:leader_end_index,1], 
                    c=LEADER_COLOR, s=SCATTER_SIZE, label="Leader")
        plt.scatter(x=self.follower_traj[:follower_end_index,0], y=self.follower_traj[:follower_end_index,1], 
                    c=FOLLOWER_COLOR, s=SCATTER_SIZE, label="Follower")
        plt.legend()
        plt.xlim([-800, -700])
        plt.ylim([-975, -875])
        plt.title(f"{self.leader_type+self.follower_type} {self.conflict_type}: PET={self.PET:.2f}s")
        plt.show()

    def plot_speed_profile(self):
        FIGURE_WIDTH, FIGURE_HEIGHT = 5, 2
        LEADER_COLOR = "r"
        FOLLOWER_COLOR = "g"
        MARKER_SIZE = 0.5
        VERTICAL_LINE_STYLE = "--"
        fig, axs = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
        
        # speed profiles:
        axs.plot(self.leader_traj_timestamp, self.leader_traj_speed, c=LEADER_COLOR, linewidth= MARKER_SIZE, label="Leader")
        axs.plot(self.follower_traj_timestamp, self.follower_traj_speed, c=FOLLOWER_COLOR, linewidth= MARKER_SIZE, label="Follower")
        axs.axvline(x=self.leader_time_at_conflict, c=LEADER_COLOR, ls=VERTICAL_LINE_STYLE)
        axs.axvline(x=self.follower_time_at_conflict, c=FOLLOWER_COLOR, ls=VERTICAL_LINE_STYLE)
        axs.set_xlim([0, 50])
        axs.set_ylabel("Speed")
        axs.legend()    
    
    def plot_profiles(self):
        FIGURE_WIDTH, FIGURE_HEIGHT = 4, 6
        LEADER_COLOR = "r"
        FOLLOWER_COLOR = "g"
        MARKER_SIZE = 2
        VERTICAL_LINE_STYLE = "--"
        TTC_COLOR = "#7C00FE"
        REQ_DEC_COLOR = "#FFAF00"
        TA_COLOR = "#F5004F"
        
        fig, axs = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
        
        # speed profiles:
        axs[0].scatter(self.leader_traj_timestamp, self.leader_traj_speed, c=LEADER_COLOR, s= MARKER_SIZE, label="Leader")
        axs[0].scatter(self.follower_traj_timestamp, self.follower_traj_speed, c=FOLLOWER_COLOR, s= MARKER_SIZE, label="Follower")
        axs[0].axvline(x=self.leader_time_at_conflict, c=LEADER_COLOR, ls=VERTICAL_LINE_STYLE)
        axs[0].axvline(x=self.follower_time_at_conflict, c=FOLLOWER_COLOR, ls=VERTICAL_LINE_STYLE)
        axs[0].set_ylabel("Speed")
        axs[0].set_xlim([0, 50])
        axs[0].legend()
        # TTC profiles:
        axs[1].set_ylabel("TTC")
        axs[1].axvline(x=self.leader_time_at_conflict, c=LEADER_COLOR, ls=VERTICAL_LINE_STYLE)
        axs[1].axvline(x=self.follower_time_at_conflict, c=FOLLOWER_COLOR, ls=VERTICAL_LINE_STYLE)

        axs[1].scatter(x=self.states_before_conflictPoint[:,0], y=self.states_before_conflictPoint[:,1],
                       c=TTC_COLOR, s=MARKER_SIZE)
        
        axs[1].set_ylim([0,10])

        # Required deceleration 
        axs[2].set_ylabel("Req deceleration")
        axs[2].axvline(x=self.leader_time_at_conflict, c=LEADER_COLOR, ls=VERTICAL_LINE_STYLE)
        axs[2].axvline(x=self.follower_time_at_conflict, c=FOLLOWER_COLOR, ls=VERTICAL_LINE_STYLE)

        axs[2].scatter(x=self.states_before_conflictPoint[:,0], y=self.states_before_conflictPoint[:,2],
                       c=REQ_DEC_COLOR, s=MARKER_SIZE)
        
        axs[2].set_ylim([-1,2])   

        # Time advantage 
        axs[3].set_ylabel("Time advantage")
        axs[3].set_xlabel("Time")
        axs[3].axvline(x=self.leader_time_at_conflict, c=LEADER_COLOR, ls=VERTICAL_LINE_STYLE)
        axs[3].axvline(x=self.follower_time_at_conflict, c=FOLLOWER_COLOR, ls=VERTICAL_LINE_STYLE)

        axs[3].scatter(x=self.states_before_conflictPoint[:,0], y=self.states_before_conflictPoint[:,3],
                       c=TA_COLOR, s=MARKER_SIZE)
        
        axs[3].set_ylim([-15,20]) 
        
        fig.suptitle(f"{self.leader_type+self.follower_type} {self.conflict_type}")
        plt.show()
    
    @property
    def leader_average_speed(self) -> float:
        """ Return the average speed of the leader vehicle """
        return self.leader_traj_length / self.leader_T

    @property
    def follower_average_speed(self) -> float:
        """ Return the average speed of the follower vehicle """
        return self.follower_traj_length / self.follower_T

    @property
    def leader_conflict_speed(self) -> float:
        """ Return the speed of the leader vehicle at the conflict point """
        index = np.where(self.leader_traj_timestamp == self.leader_time_at_conflict)[0]
        if index.shape[0] > 1:
            index = int(index[0])
        elif index.shape[0] == 1:
            index = int(index)
        else:
            return None
        return float(self.leader_traj_speed[index])
    
    @property
    def follower_conflict_speed(self) -> float:
        """ Return the speed of the follower vehicle at the conflict point """
        index = np.where(self.follower_traj_timestamp == self.follower_time_at_conflict)[0]
        if index.shape[0] > 1:
            index = int(index[0])
        elif index.shape[0] == 1:
            index = int(index)
        else:
            return None
            
        return float(self.follower_traj_speed[index])

    @property
    def minimum_TTC(self):
        TTCs = self.TTCs
        if len(TTCs) == 0:
            return np.NaN
        else:
            return min(TTCs)
    
    @property
    def TTCs(self) -> List[float]:
        if self.two_vehicles_co_exist:
            TTCs = self.states_before_conflictPoint[:,1]
            if np.all(np.isnan(TTCs)):
                return []
            else: 
                return TTCs[~np.isnan(TTCs)].tolist()
        else:
            return []

    @property
    def max_req_deceleration(self):
        req_decelerations = self.req_decelerations
        if len(req_decelerations) == 0:
            return np.NaN
        else:
            return max(req_decelerations)
    
    @property
    def req_decelerations(self) -> List[float]:
        if self.two_vehicles_co_exist:
            req_decelerations = self.states_before_conflictPoint[:,2]
            if np.all(np.isnan(req_decelerations)):
                return []
            else: 
                return req_decelerations[~np.isnan(req_decelerations)].tolist()
        else:
            return []

    @property
    def time_advantages(self) -> List[float]:
        if self.two_vehicles_co_exist: 
            time_advantages = self.states_before_conflictPoint[:,[3, 5]] # TA and follower's distance to conflict point
            time_advantages = time_advantages[np.where(
                time_advantages[:,1] <= 40
            )][:,0]
            return time_advantages[~np.isnan(time_advantages)].tolist()
        else:
            return [] 

    @property
    def minTTC_speeds(self):
        if self.two_vehicles_co_exist:
            # extract the 3 columns (TTC, leader speed and follower speed) from state
            TTC_speeds = self.states_before_conflictPoint[:,[1,6,7]]   
            # remove NaN values
            TTC_speeds = TTC_speeds[~np.isnan(TTC_speeds).any(axis=1)]
            if TTC_speeds.shape == (0,3):
                return []
            else:
                # return the min TTC and corresponding speeds of leader and follower
                return TTC_speeds[np.argmin(TTC_speeds[:, 0])].tolist()
        else:
            return []  
