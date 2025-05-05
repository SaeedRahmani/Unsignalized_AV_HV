import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, MultiLineString, Polygon
from ..filter.helper import multiline_to_single_line
from .utils import compute_position_based_velocities, three_sigma_smoothing

class Trajectory:

    LEFT_TOP 	 = (-200, -865)
    LEFT_BOTTOM  = (-200, -915)
    RIGHT_TOP 	 = ( -50, -865)
    RIGHT_BOTTOM = ( -50, -915)
    X_MIN = -200
    X_MAX = -50
    Y_MIN = -915
    Y_MAX = -865
        
    # LEFT_TOP = (-800, -875)
    # LEFT_BOTTOM = (-800, -950)
    # RIGHT_TOP = (-700, -875)
    # RIGHT_BOTTOM = (-700, -950)

    # X_MIN = -800
    # X_MAX = -700
    # Y_MIN = -950
    # Y_MAX = -875
    
    def __init__(self, xy, t):
        assert xy.shape[0] == t.shape[0]
        assert xy.shape[0] > 2, f"The trajectory is not long enough to compute the position-based velocities."
        
        # positions: (x, y) with shape (N, 2)
        self.xy = xy
        
        # time indices: t with shape (N,)
        self.t = t
        
        # conbime positions and time indices (x, y, t) -> shape (N, 3) 
        self.xyt = np.hstack([self.xy, self.t.reshape((-1,1))])
        
        # get the length of the trajectory in meters
        self.lineString = LineString(coordinates=self.xy)
        self.L = self.lineString.length
        
        # get the duration of the trajectory in seconds
        self.T = self.t.shape[0] / 10 

        # get the position-based velocites of the trajectories in meter/second
        self.v = compute_position_based_velocities(xy=self.xy)
        assert self.v.shape[0] + 2 == self.xy.shape[0], f"({self.v.shape})"
        
        # smoothe the velocities
        self.smoothed_v = three_sigma_smoothing(self.v)

        # self.a = np.diff(self.v) / 0.1
        # assert self.a.shape[0] + 2 == self.xy.shape[0], f"({self.a.shape})"

        polygon_Tjunction = Polygon([self.LEFT_TOP, self.LEFT_BOTTOM, self.RIGHT_BOTTOM, self.RIGHT_TOP, self.LEFT_TOP])
        self.lineString_Tjunction = self.lineString.intersection(polygon_Tjunction)

        # TODO: convert MultiLineString to LineString
        if isinstance(self.lineString_Tjunction, MultiLineString):
            self.lineString_Tjunction = multiline_to_single_line(self.lineString_Tjunction)

        self.xy_Tjunction = np.array(self.lineString_Tjunction.coords)
        self.v_Tjunction = np.sqrt(np.sum(np.diff(self.xy_Tjunction, axis=0) ** 2, axis=1)) / 0.1

        self.firstTimeInArea = self.t[
            np.argmin(np.sqrt(np.sum(np.square(self.xy - self.xy_Tjunction[0, :]), axis=1)) < 0.35)]

    def __str__(self, ):
        return f"Trajectory({self.L:.1f} meters, {self.T:.1f} seconds)"

    def __repr__(self, ):
        return f"Trajectory({self.L:.1f} meters, {self.T:.1f} seconds)"

    def plot(self, show_full_trajectory=False):
        plt.figure(figsize=(3, 2.5))
        if show_full_trajectory:
            plt.xlim([-850, -550])
            plt.ylim([-1150, -850])
            plt.scatter(
                x=self.xy[1:, 0], y=self.xy[1:, 1], c=self.v,
                cmap="jet", s=1, vmin=0, vmax=15, alpha=0.95
            )
            plt.title("Full trajectory", fontdict={"fontsize": 10})
        else:
            plt.xlim([self.LEFT_TOP[0], self.RIGHT_TOP[0]])
            plt.ylim([self.LEFT_BOTTOM[1], self.LEFT_TOP[1]])
            plt.scatter(
                x=self.xy_Tjunction[1:, 0], y=self.xy_Tjunction[1:, 1], c=self.v_Tjunction,
                cmap="jet", s=1, vmin=0, vmax=15, alpha=0.95
            )
            plt.title("Trajectory in T-junction", fontdict={"fontsize": 10})
        plt.colorbar(label="speed")
        plt.show()
