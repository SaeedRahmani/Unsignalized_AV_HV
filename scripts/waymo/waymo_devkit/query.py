import numpy as np
from typing import Dict, List, Tuple
from scipy.spatial import cKDTree
from collections import defaultdict
from shapely import Polygon, LineString, Point
from waymo_open_dataset.protos.scenario_pb2 import Scenario

def get_scenario_metadata(txt_path: str):
    metadatas = []
    with open(txt_path, "r") as f:
        for line in f:
            tfrecord_id, scenario_id = str(line).split(",")
            metadatas.append((tfrecord_id, int(scenario_id)))
    return metadatas

def get_egoTrajectory(scenario: Scenario) -> Tuple:
    """ 
    Return the ego trajectory as a tuple (id, index, xyt)
    """
    ego_id = scenario.tracks[scenario.sdc_track_index].id   
    ego_index = scenario.sdc_track_index
    ego_trajectory = []
    for time_index, ego_state in enumerate(scenario.tracks[scenario.sdc_track_index].states):
        ego_trajectory.append((
            ego_state.center_x, 
            ego_state.center_y,
            time_index * 0.1,
            ego_state.heading,
            ego_state.velocity_x,
            ego_state.velocity_y,
        ))
    ego_trajectory = np.array(ego_trajectory)
    assert ego_trajectory.shape[1] == 6
    egoTrajectory = (ego_id, ego_index, ego_trajectory)
    return egoTrajectory 

def get_vehicleTrajectories(scenario: Scenario, intersection_polygon: Polygon) -> List[Tuple]:
    """ 
    Return a list of the agent trajectories
    """    
    vehicleTrajectories = list()
    
    for veh_index, track in enumerate(scenario.tracks):
        if veh_index != scenario.sdc_track_index and track.object_type == 1:
            vehicle_trajectory = []
            for time_index, veh_state in enumerate(track.states):
                if veh_state.valid:
                    vehicle_trajectory.append((
                        veh_state.center_x, 
                        veh_state.center_y,
                        time_index * 0.1,
                        veh_state.heading,
                        veh_state.velocity_x,
                        veh_state.velocity_y,
                    ))
            vehicle_trajectory = np.array(vehicle_trajectory)
            assert vehicle_trajectory.shape[1] == 6
            # print(LineString(vehicle_trajectory[:,:2]).intersects(intersection_polygon))
            if LineString(vehicle_trajectory[:,:2]).intersects(intersection_polygon):
                vehicleTrajectories.append((track.id, veh_index, vehicle_trajectory))
            # else:
            #     vehicleTrajectories.append((track.id, veh_index, vehicle_trajectory))
                
    return vehicleTrajectories

def get_intersection_lanes(
    scenario: Scenario,
    intersection_polygon: Polygon
):
    """ 
    Return the inbound and outbound lanes of the unsignalised intersection
    @param: scenario
    @param: intersection_polygon: Polygon, the polygon of the intersection area
    @return: inbound_lanes: List
    @return: outbound_lanes: List
    """
    laneCenters = get_laneCenters(scenario)
    
    lanes = []
    # remove lanes that fully inside the intersection polygons
    for lane in laneCenters:
        if lane[2].shape[0] == 1:
            continue
        lane_lineString = LineString(lane[2])
        if not lane_lineString.within(intersection_polygon):
            lanes.append(lane)
    
    # divide remaining lanes into inbound and outbound lanes             
    inbound_lanes, outbound_lanes = list(), list()
    for lane in lanes:
        # inbound lane := end point in the polygon
        if Point(lane[2][-1,0], lane[2][-1,1]).within(intersection_polygon):
            inbound_lanes.append(lane)
        # outbound lane := start point in the polygon
        elif Point(lane[2][0,0], lane[2][0,1]).within(intersection_polygon):
            outbound_lanes.append(lane)

    return inbound_lanes, outbound_lanes

def get_laneCenters(scenario: Scenario) -> List[Tuple]:
    """
    Return a list of lane centers (id, type, lane coordinates) pairs.
    
    @param: scenario
    @return: laneCenters: List, a list of all the lane centers in this scenario's static HD map
    """
    laneCenters = list()
    
    for mapFeature in scenario.map_features:
        if mapFeature.WhichOneof("feature_data") == "lane":
            lane_xs, lane_ys = zip(*[(p.x, p.y) for p in mapFeature.lane.polyline]) 
            lane_coordinate = np.array([lane_xs, lane_ys]).T
            laneCenters.append((
                mapFeature.id,
                mapFeature.lane.type,
                lane_coordinate,
            ))
            
    return laneCenters

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
    assert intersection_stopSignCoordinates.shape[1] == 2, f"Got {intersection_stopSignCoordinates.shape[1]}, expected 2"
    radius = 0
    intersection_centerCoordinate = np.mean(intersection_stopSignCoordinates, axis=0)
    if aggregation == "max":
        radius = np.max(np.linalg.norm(intersection_centerCoordinate - intersection_stopSignCoordinates, axis=1)) + buffer
    elif aggregation == "mean":
        radius = np.mean(np.linalg.norm(intersection_centerCoordinate - intersection_stopSignCoordinates, axis=1)) + buffer
    else:
        assert False, "Specify the metric."
    return (intersection_centerCoordinate.tolist(), radius)
    
def get_intersection_stopSigns(
    scenario: Scenario, 
    distance_threshold: float = 45,
    n_legs: int = 4,
) -> List[Tuple]:
    """ 
    Return a list of stop sign (id, coordinate) pairs,
    each stop sign is close to all the others within 30 meters,
    assuming these stop signs (at least n_legs) are located 
    in the same unsignalised intersection. 
    
    @param: scenario
    @param: distance_threshold: float, distance threshold between two stop signs, 
                                       default to 20 meters
    @return: intersection_stopSigns: List, a list of at least 4 stop signs 
                                           within the same unsignalised intersection
    """
    stopSigns = get_stopSigns(scenario)
    
    if len(stopSigns) == 0:
        return []
    
    stopSignCoordinates = list()
    for stopSign in stopSigns:
        stopSignCoordinates.append(stopSign[1])
    
    # Apply the cKDTree to find pairs of 2 stop signs close enough
    tree = cKDTree(np.array(stopSignCoordinates))
    pairs = tree.query_pairs(distance_threshold)

    def _build_graph(pairs):
        graph = defaultdict(set)
        for a, b in pairs:
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
    
    def _find_max_clique(pairs):
        graph = _build_graph(pairs)
        cliques = []
        _find_cliques(graph, cliques=cliques)
        max_clique = max(cliques, key=len)
        return max_clique
    
    # Find a full-connected graph that all stop signs are close to each other
    intersection_stopSign_indices = _find_max_clique(pairs)

    # Returned list
    intersection_stopSigns = list()
    for index in intersection_stopSign_indices:
        intersection_stopSigns.append(stopSigns[index])
    
    return intersection_stopSigns

def get_stopSigns(scenario: Scenario) -> List[Tuple]:
    """
    Return a list of stop sign (id, coordinate) pairs.
    
    @param: scenario
    @return: stopSigns: List, a list of all the stop signs in this scenario's static HD map
    """
    stopSigns = list()
    
    for mapFeature in scenario.map_features:
        if mapFeature.WhichOneof("feature_data") == "stop_sign":
            stopSigns.append((
                mapFeature.id,
                (mapFeature.stop_sign.position.x, 
                 mapFeature.stop_sign.position.y)
            ))
            
    return stopSigns