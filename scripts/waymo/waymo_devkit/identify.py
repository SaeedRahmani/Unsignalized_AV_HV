import numpy as np
from typing import List, Tuple
from shapely import LineString, Polygon, MultiPoint, Point, MultiPolygon
from waymo_devkit.shape import (
    construct_intersection_polygon, construct_LaneLineString
)
# from waymo_devkit.conflict import Conflict, ConflictType
    

def identify_conflict(
    traj_a, 
    traj_b,
    inbound_lanes: List,
    outbound_lanes: List,
    a_is_av: bool,
    b_is_av: bool,
    tfrecord_index, scenario_index, 
    intersection_circle: Polygon,
    center,
    radius: float,
    PET: float = 10,
    buffer: float = 1,
) -> Tuple[str, dict]:   
    inbound_lane_lineStrings = construct_LaneLineString(inbound_lanes)
    outboundbound_lane_lineStrings = construct_LaneLineString(outbound_lanes)
    assert len(inbound_lanes) == len(inbound_lane_lineStrings)
    assert len(outbound_lanes) == len(outboundbound_lane_lineStrings)
        
    # traj a
    traj_a_lineString = LineString(traj_a[2][:,:2])
    traj_a_in_id = None
    for index, lane in enumerate(inbound_lanes): 
        if inbound_lane_lineStrings[index].intersects(traj_a_lineString):
            traj_a_in_id = lane[0]
    traj_a_out_id = None
    for index, lane in enumerate(outbound_lanes): 
        if outboundbound_lane_lineStrings[index].intersects(traj_a_lineString):
            traj_a_out_id = lane[0]        
    
    # traj b
    traj_b_lineString = LineString(traj_b[2][:,:2])
    traj_b_in_id = None
    for index, lane in enumerate(inbound_lanes): 
        if inbound_lane_lineStrings[index].intersects(traj_b_lineString):
            traj_b_in_id = lane[0]
    traj_b_out_id = None
    for index, lane in enumerate(outbound_lanes): 
        if outboundbound_lane_lineStrings[index].intersects(traj_b_lineString):
            traj_b_out_id = lane[0]  
            
    if (traj_a_in_id is None or traj_a_out_id is None) or \
        (traj_b_in_id is None or traj_b_out_id is None):
        # at least one trajectory is not sufficient.
        return ("UNKNOWN", None)
    else:    
        # cross criteria
        if traj_a_in_id != traj_b_in_id and traj_a_out_id != traj_b_out_id: # different inbound lanes but the same outbound lanes
            conflict_point = traj_a_lineString.intersection(traj_b_lineString)
            if conflict_point.is_empty:
                # two trajectories do not intersect
                return ("NO_CONFLICT", None)
            elif conflict_point.geom_type == "Point":
                # return as a Point
                conflict_point_coord = np.array([conflict_point.x, conflict_point.y])
            elif conflict_point.geom_type == "MultiPoint":
                # return as a MultiPoint, then select the first Point
                conflict_point_coord = np.array([conflict_point.geoms[0].x, conflict_point.geoms[0].y])
                conflict_point = Point(conflict_point_coord)
            else:
                assert False, f"Non-considered type in cross conflict identification: {type(conflict_point)}"   

            # check if the conflict point is inside the intersection circle
            if conflict_point.within(intersection_circle):
                time_a, time_b, pet = calculate_PET(traj_a, traj_b, conflict_point_coord)
                if abs(pet) < PET:
                    if pet < 0:
                        return ("CROSS", dict(
                            leader_id=traj_a[0],
                            leader_index=traj_a[1],
                            leader_states=traj_a[2],
                            leader_time_at_conflict=time_a,
                            follower_id=traj_b[0],
                            follower_index=traj_b[1],
                            follower_states=traj_b[2],
                            follower_time_at_conflict=time_b,
                            PET=abs(pet),
                            tfrecord_index=tfrecord_index,
                            scenario_index=scenario_index,
                            conflict_type="CROSS",
                            leader_is_av=a_is_av, follower_is_av=b_is_av, 
                            center=center, radius=radius,
                        ))
                    else:
                        return ("CROSS", dict(
                            leader_id=traj_b[0],
                            leader_index=traj_b[1],
                            leader_states=traj_b[2],
                            leader_time_at_conflict=time_b,
                            follower_id=traj_a[0],
                            follower_index=traj_a[1],
                            follower_states=traj_a[2],
                            follower_time_at_conflict=time_a,
                            PET=abs(pet),
                            tfrecord_index=tfrecord_index,
                            scenario_index=scenario_index,
                            conflict_type="CROSS",
                            leader_is_av=b_is_av, follower_is_av=a_is_av, 
                            center=center, radius=radius,
                        ))
                else:
                    return ("NO_CONFLICT", None)
            else:
                return ("NO_CONFLICT", None)        
            
        # merge criteria
        elif traj_a_in_id != traj_b_in_id and traj_a_out_id == traj_b_out_id:
            # retrieve the conflict point and if it is valid
            isIntersected, conflict_point_coord = _get_merge_conflict_point_coordinate(traj_a, traj_b, buffer)

            # check if the conflict point is inside the intersection circle
            if isIntersected and Point(conflict_point_coord).within(intersection_circle):
                time_a, time_b, pet = calculate_PET(traj_a, traj_b, conflict_point_coord)
                if pet == None:
                    return ("NO_CONFLICT", None)                
                if abs(pet) < PET:
                    if pet < 0:
                        return ("MERGE", dict(
                            leader_id=traj_a[0],
                            leader_index=traj_a[1],
                            leader_states=traj_a[2],
                            leader_time_at_conflict=time_a,
                            follower_id=traj_b[0],
                            follower_index=traj_b[1],
                            follower_states=traj_b[2],
                            follower_time_at_conflict=time_b,
                            PET=abs(pet),
                            tfrecord_index=tfrecord_index,
                            scenario_index=scenario_index,
                            conflict_type="MERGE",
                            leader_is_av=a_is_av, follower_is_av=b_is_av, 
                            center=center, radius=radius,
                        ))
                    else:
                        return ("MERGE", dict(
                            leader_id=traj_b[0],
                            leader_index=traj_b[1],
                            leader_states=traj_b[2],
                            leader_time_at_conflict=time_b,
                            follower_id=traj_a[0],
                            follower_index=traj_a[1],
                            follower_states=traj_a[2],
                            follower_time_at_conflict=time_a,
                            PET=abs(pet),
                            tfrecord_index=tfrecord_index,
                            scenario_index=scenario_index,
                            conflict_type="MERGE",
                            leader_is_av=b_is_av, follower_is_av=a_is_av,
                            center=center, radius=radius,
                        ))
                else:
                    return ("NO_CONFLICT", None)
            else:
                return ("NO_CONFLICT", None)

        else:
            return ("UNKNOWN", None)
        
def calculate_PET(
    traj_a: np.array, 
    traj_b: np.array,
    conflict_point_coordinate,
) -> Tuple[float, float, float]:
    """
    Calculate the PET of one conflict.
    @return: time_a_at_conflict: float, the timestamp of trajectory a reaching the conflict point.
    @return: time_b_at_conflict: float, the timestamp of trajectory b reaching the conflict point.
    @return: signed_pet: float, the difference between two trajectories' timestamp at conflict point,
                                negative if a is the leader vehicle,
                                positive if a is the follower vehicle.
    """
    assert conflict_point_coordinate.shape == (2,)    

    # get trajectory coordinates and timestamps
    traj_a_coordinates, traj_a_timestamps = traj_a[2][:,:2], traj_a[2][:,2] 
    traj_b_coordinates, traj_b_timestamps = traj_b[2][:,:2], traj_b[2][:,2]
    
    # calculate the timestamp at the conflict point
    time_a_at_conflict = traj_a_timestamps[np.argmin(np.linalg.norm(traj_a_coordinates - conflict_point_coordinate, axis=1))]
    time_b_at_conflict = traj_b_timestamps[np.argmin(np.linalg.norm(traj_b_coordinates - conflict_point_coordinate, axis=1))]

    # calculate signed pet
    signed_pet = time_a_at_conflict - time_b_at_conflict

    return time_a_at_conflict, time_b_at_conflict, signed_pet

def _get_merge_conflict_point_coordinate(traj_a, traj_b, buffer: float):
    traj_a_lineString = LineString(traj_a[2][:,:2])
    traj_b_lineString = LineString(traj_b[2][:,:2])

    left_a = traj_a_lineString.parallel_offset(1, "left")
    right_a = traj_a_lineString.parallel_offset(1, "right")
    left_b = traj_b_lineString.parallel_offset(1, "left")
    right_b = traj_b_lineString.parallel_offset(1, "right")
    
    if left_a.intersects(right_b) and not right_a.intersects(left_b):
        intersection = left_a.intersection(right_b)
        if isinstance(intersection, Point):
            intersection_coord = np.array([intersection.x, intersection.y])
        elif isinstance(intersection, MultiPoint):
            intersection = intersection.geoms[0]
            intersection_coord = np.array([intersection.x, intersection.y])
    
    elif not left_a.intersects(right_b) and right_a.intersects(left_b):
        intersection = right_a.intersection(left_b)
        if isinstance(intersection, Point):
            intersection_coord = np.array([intersection.x, intersection.y])
        elif isinstance(intersection, MultiPoint):
            intersection = intersection.geoms[0]
            intersection_coord = np.array([intersection.x, intersection.y])        
    else:
        intersection1 = left_a.intersection(right_b)
        intersection2 = right_a.intersection(left_b)

        is_a_valid, is_b_valid = True, True
        
        # traj a
        if isinstance(intersection1, Point):
            intersection_coord1 = np.array([intersection1.x, intersection1.y])
        elif isinstance(intersection1, LineString):
            if np.array(intersection1.coords).shape[0] != 0:
                intersection_coord1 = np.array(intersection1.coords)[0]
            else:
                is_a_valid = False
        elif isinstance(intersection1, MultiPoint):
            intersection1 = intersection1.geoms[0]
            intersection_coord1 = np.array([intersection1.x, intersection1.y])
        else:
            assert False, f"Non-considered type: {type(intersection1)}"
            
        # traj b
        if isinstance(intersection2, Point):
            intersection_coord2 = np.array([intersection2.x, intersection2.y])
        elif isinstance(intersection2, LineString):
            if np.array(intersection2.coords).shape[0] != 0:
                intersection_coord2 = np.array(intersection2.coords)[0]
            else:
                is_b_valid = False
        elif isinstance(intersection2, MultiPoint):
            intersection2 = intersection2.geoms[0]
            intersection_coord2 = np.array([intersection2.x, intersection2.y])
        else:
            assert False, f"Non-considered type: {type(intersection2)}"
            
        if is_a_valid and is_b_valid:
            if np.argmin(np.linalg.norm(traj_a[2][:,:2] - intersection_coord1, axis=1)) <= np.argmin(np.linalg.norm(traj_a[2][:,:2] - intersection_coord2, axis=1)):
                intersection_coord = intersection_coord1
            else:
                intersection_coord = intersection_coord2
        elif not is_a_valid and is_b_valid:
            intersection_coord = intersection_coord2
        elif is_a_valid and not is_b_valid:
            intersection_coord = intersection_coord1 
        else:           
            return False, None

    return True, intersection_coord 

def identify_complex_conflicts(list_potential_conflicts: List[dict]) -> List[dict]:
    # retrieve a list of pairs of (leader id, follower id)
    list_2pair = [(conflict["leader_id"], conflict["follower_id"]) for conflict in list_potential_conflicts]
    # retrieve a list of tuples of (leader id, follower id, time of leader reaching conflict point)
    list_3tuple = [(conflict["leader_id"], conflict["follower_id"], conflict["leader_time_at_conflict"]) for conflict in list_potential_conflicts]
    # sort this list of tuples based on the time of leader vehicle reaching conflict time
    list_3tuple = sorted(list_3tuple, key=lambda x: x[2])

    # construct the table
    table = []
    for element in list_3tuple:
        table.append(list(element[:2]))
    table = np.array(table).T
    assert table.shape[0] == 2
    table = table.reshape((-1))
    # vehicle sequences
    sequences = []
    for element in table:
        if element not in sequences:
            sequences.append(element)
        else:
            pass

    list_conflicts = []
    list_ids = []
    for element in range(len(sequences)-1):
        id_pair = (sequences[element], sequences[element+1])
        if id_pair in list_2pair:
            list_ids.append(id_pair)
            position = list_2pair.index(id_pair)
            list_conflicts.append(list_potential_conflicts[position])

    return list_conflicts