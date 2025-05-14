import numpy as np
from src import Trajectory
from src.intersection.waymo import StopSign
from typing import List, Tuple
from collections import defaultdict
from shapely import LineString, Polygon, Point
from scipy.spatial.ckdtree import cKDTree
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


def get_all_stop_signs_from_scenario(scenario: Scenario) -> List[StopSign]:
    """
    Construct a list of the stop signs from the Scenario proto object.

    Args:
        scenario: Scenario, a scenario proto object from the Waymo Open Motion Dataset

    Returns:
        all_stop_signs: List[Tuple], a list of stop signs including the id and coordinate
    """
    return [StopSign(id=map_feature.id, coord_x=map_feature.stop_sign.position.x, coord_y=map_feature.stop_sign.position.y)
                for map_feature in scenario.map_features if map_feature.WhichOneof("feature_data") == "stop_sign"]


def get_intersection_lanes_from_scenario(scenario: Scenario, intersection_area: Polygon) -> Tuple[List, List]:
    """
    Get all the inbound lanes and outbound lanes of the unsignalized intersection

    Args:
        scenario: Scenario, a scenario proto object from the Waymo Open Motion Dataset
        intersection_area: Polygon, a polygon object representing the intersection area

    Returns:
        inbound_lanes_list: List, a list of inbound lanes including the id and coordinate
        outbound_lanes_list: List, a list of outbound lanes including the id and coordinate
    """
    lane_centers = get_lane_centers_from_scenario(scenario=scenario)

    lanes = []
    # remove lanes that fully inside the intersection polygons
    for lane in lane_centers:
        if lane[2].shape[0] == 1:
            continue
        lane_lineString = LineString(lane[2])
        if not lane_lineString.within(intersection_area):
            lanes.append(lane)

    # divide remaining lanes into inbound and outbound lanes
    inbound_lanes, outbound_lanes = list(), list()
    for lane in lanes:
        # inbound lane := end point in the polygon
        if Point(lane[2][-1, 0], lane[2][-1, 1]).within(intersection_area):
            inbound_lanes.append(lane)
        # outbound lane := start point in the polygon
        elif Point(lane[2][0, 0], lane[2][0, 1]).within(intersection_area):
            outbound_lanes.append(lane)

    return inbound_lanes, outbound_lanes


def get_intersection_stop_signs_from_scenario(scenario: Scenario, distance_threshold: float = 45) -> List[StopSign]:
    """
    Return a list of stop sign (id, coordinate) pairs,
        each stop sign is close to all the others within `distance_threshold` meters,
    assuming these stop signs (at least n_legs) are located
    in the same unsignalized intersection.

    @param: scenario
    @param: distance_threshold: float, distance threshold between two stop signs,
                                       default to 45 meters
    @return: intersection_stopSigns: List, a list of at least 4 stop signs
                                           within the same unsignalized intersection
    """
    all_stop_signs = get_all_stop_signs_from_scenario(scenario)

    if len(all_stop_signs) == 0:
        return []

    stop_sign_coords = [np.array(ss.coords) for ss in all_stop_signs]


    # Apply the cKDTree to find pairs of 2 stop signs close enough
    ckdt_tree = cKDTree(stop_sign_coords)
    pairs = ckdt_tree.query_pairs(distance_threshold)

    def _build_graph(pair_list: List) -> defaultdict:
        graph = defaultdict(set)
        for a, b in pair_list:
            graph[a].add(b)
            graph[b].add(a)
        return graph

    def _find_cliques(graph, potential_clique=[], remaining_nodes=None, skip_nodes=set(), cliques=[]):
        if remaining_nodes is None:
            remaining_nodes = set(graph.keys())

        if not remaining_nodes and not skip_nodes:
            cliques.append(potential_clique)
            return

        for node in list(remaining_nodes):
            new_potential_clique = potential_clique + [node]
            new_remaining_nodes = remaining_nodes.intersection(graph[node])
            new_skip_nodes = skip_nodes.intersection(graph[node])
            _find_cliques(graph, new_potential_clique, new_remaining_nodes, new_skip_nodes, cliques)
            remaining_nodes.remove(node)
            skip_nodes.add(node)

    def _find_max_clique(pairs) -> List:
        # Find a full-connected graph that all stop signs are close to each other
        graph = _build_graph(pairs)
        cliques: List[List] = []
        _find_cliques(graph, cliques=cliques)
        max_clique = max(cliques, key=len)
        return max_clique

    return [all_stop_signs[index] for index in _find_max_clique(pairs)]


def get_lane_centers_from_scenario(scenario: Scenario) -> List[Tuple]:
    """
    Return a list of lane centers (id, type, lane coordinates) pairs.

    @param: scenario
    @return: lane_centers: List, a list of all the lane centers in this scenario's static HD map
    """
    lane_centers = list()

    for mapFeature in scenario.map_features:
        if mapFeature.WhichOneof("feature_data") == "lane":
            lane_xs, lane_ys = zip(*[(p.x, p.y) for p in mapFeature.lane.polyline])
            lane_coordinate = np.array([lane_xs, lane_ys]).T
            lane_centers.append((
                mapFeature.id,
                mapFeature.lane.type,
                lane_coordinate,
            ))

    return lane_centers


def get_intersection_circle(
        intersection_stopSigns: List,
        aggregation: str = "max",
        buffer: float = 5
) -> Tuple:
    """
    Return the center coordinate and radius of the unsignalised intersection
    @param: intersection_stopSigns: List, a list of at least 4 stop signs
    @param: aggregation: str, mean or max operation on
                              the distances between center and each stop signs
    @param: buffer: float, radius = average_distance + buffer
    @return: coordinate of intersection (x_center, y_center)
    @return: radius of intersection
    """
    intersection_stopSignCoordinates = list()
    for stopSign in intersection_stopSigns:
        intersection_stopSignCoordinates.append(stopSign[1])
    intersection_stopSignCoordinates = np.array(intersection_stopSignCoordinates)
    assert intersection_stopSignCoordinates.shape[
               1] == 2, f"Got {intersection_stopSignCoordinates.shape[1]}, expected 2"
    radius = 0
    intersection_center_coordinate = np.mean(intersection_stopSignCoordinates, axis=0)
    if aggregation == "max":
        radius = np.max(
            np.linalg.norm(intersection_center_coordinate - intersection_stopSignCoordinates, axis=1)) + buffer
    elif aggregation == "mean":
        radius = np.mean(
            np.linalg.norm(intersection_center_coordinate - intersection_stopSignCoordinates, axis=1)) + buffer
    else:
        assert False, "Specify the metric."
    return intersection_center_coordinate.tolist(), radius
