import numpy as np
from shapely import Polygon

from src import Trajectory
from typing import List, Tuple
from shapely.geometry import LineString, Polygon
from waymo_open_dataset.protos.scenario_pb2 import Scenario


def get_all_indices_pairs(file_path: str) -> List[Tuple[str, int]]:
    """
    Get all pairs of (tfrecord_id: str, scenario_id: int)
    from the saved csv files (i.e., scenario_3_stop_signs.csv),
    written with list comprehension for simplicity.

    Args:
        file_path: str, the path to the csv file

    Returns:
        indices_pairs: List, a list of (tfrecord_id: str, scenario_id: int) pairs
    """
    with open(file_path, "r") as f:
        return [(str(tfrecord_id), int(scenario_id)) for tfrecord_id, scenario_id in
                (line.strip().split(",") for line in f)]


def get_ego_trajectory_from_scenario(scenario: Scenario) -> Trajectory:
    """
    Construct the AV's ego trajectory from the Scenario proto object.

    Args:
        scenario: Scenario, a scenario proto object from the Waymo Open Motion Dataset

    Returns:
        ego_trajectory: Trajectory, the AV ego-vehicle's trajectory
    """

    # retrieve AV trajectory from EgoState proto.
    ego_id = scenario.tracks[scenario.sdc_track_index].id
    ego_sdc_track_index = scenario.sdc_track_index
    ego_coord_x = np.array([ego_state.center_x for ego_state in scenario.tracks[scenario.sdc_track_index].states])
    ego_coord_y = np.array([ego_state.center_y for ego_state in scenario.tracks[scenario.sdc_track_index].states])
    ego_heading = np.array([ego_state.heading for ego_state in scenario.tracks[scenario.sdc_track_index].states])
    ego_velocity_x = np.array([ego_state.velocity_x for ego_state in scenario.tracks[scenario.sdc_track_index].states])
    ego_velocity_y = np.array([ego_state.velocity_y for ego_state in scenario.tracks[scenario.sdc_track_index].states])

    ego_trajectory = Trajectory(
        coord_x=ego_coord_x, coord_y=ego_coord_y,
    )
    return ego_trajectory


def get_driver_trajectories_from_scenario(scenario: Scenario, intersection_area: Polygon) -> List[Trajectory]:
    """
    Construct a list of the driver's trajectories from the Scenario proto object.

    Args:
        scenario: Scenario, a scenario proto object from the Waymo Open Motion Dataset
        intersection_area: Polygon, a polygon object representing the intersection area
    Returns:
        driver_trajectories: List[Trajectory], a list of the driver's trajectories
    """
    driver_trajectories = []
    for track_index, track in enumerate(scenario.tracks):
        # the object tracked is not a self-driving car AND
        # the object is a vehicle.
        if track_index != scenario.sdc_track_index and track.object_type == 1:
            # retrieve the driver's trajectory from State proto.
            driver_coord_x = np.array([ego_state.center_x for ego_state in track.states])
            driver_coord_y = np.array([ego_state.center_y for ego_state in track.states])
            driver_heading = np.array([ego_state.heading for ego_state in track.states])
            driver_velocity_x = np.array([ego_state.velocity_x for ego_state in track.states])
            driver_velocity_y = np.array([ego_state.velocity_y for ego_state in track.states])

            # only consider the drivers' trajectories overlapped with the intersection area
            driver_trajectory = Trajectory(coord_x=driver_coord_x, coord_y=driver_coord_y)
            if LineString(driver_trajectory.coords).intersects(intersection_area):
                driver_trajectories.append(driver_trajectory)

    return driver_trajectories


def build_lane_segments(lanes: List, buffer_size: float = 1) -> List[Polygon]:
    """
    Build a list of LineString objects, representing the lane segments with fixed lane width.

    Args:
        lanes:
        buffer_size: float, buffer is equal to half of the lane width (2 meters in this case).

    Returns:
        lane_segments: List[Polygon], a list of LineString objects representing the lane segments.
    """
    return [LineString(lane[2]).buffer(buffer_size) for lane in lanes]


def build_intersection_circle_area(circle_center: Tuple[float, float], radius: float, n_vertices: int = 100) -> Polygon:
    """
    Build a polygon object, representing the intersection as a circle area.

    Args:
        circle_center: Tuple[float, float], center coordinate of the circle
        radius: float, radius of the circle
        n_vertices: int, number of vertices to construct the circle area polygon

    Returns:
        intersection_polygon: Polygon, the polygon of the intersection area
    """
    center_coordx, center_coordy = circle_center[0], circle_center[1]
    theta = np.linspace(0, 2 * np.pi, n_vertices)
    xs, ys = center_coordx + radius * np.cos(theta), center_coordy + radius * np.sin(theta)
    vertices_coords = np.vstack([xs, ys]).T
    assert vertices_coords.shape == (n_vertices, 2)
    return Polygon(shell=vertices_coords, holes=None)