import numpy as np
from typing import List, Tuple, Any
from shapely import LineString, Point, MultiPoint, Polygon, MultiPolygon
from src import Trajectory, Conflict
from .conflict import ConflictCategory
from .road_user import RoadUser
from ..datasets.waymo.load_from_proto import build_lane_segments

def detect_conflict_between_two_trajectories(
    traj_a: Trajectory, traj_b: Trajectory,
    inbound_lanes: List, outbound_lanes: List,
    a_is_av: bool, b_is_av: bool,
    tfrecord_index, scenario_index, intersection_area: Polygon,
    # center, radius,
    pet_threshold: float = 10, buffer: float = 1
) -> Any:
    """
    Detect whether there is a conflict between two trajectories.

    Args:
        traj_a: Trajectory,
        traj_b: Trajectory,
        inbound_lanes:
        outbound_lanes:
        a_is_av:
        b_is_av:
        tfrecord_index:
        scenario_index:
        intersection_area:
        # center:
        # radius:
        pet_threshold:
        buffer:

    Returns:
        conflict: Conflict, the conflict between two trajectories,
                            if not, return None.
    """

    conflict = None
    dataset_properties = {
        "dataset": "waymo",
        "tfrecord_index": tfrecord_index,
        "scenario_index": scenario_index,
    }

    inbound_lane_linestring_list: List[Polygon] = build_lane_segments(inbound_lanes)
    outbound_lane_linestring_list: List[Polygon] = build_lane_segments(outbound_lanes)

    traj_a_linestring: LineString = LineString(traj_a.coords)
    traj_b_linestring: LineString = LineString(traj_b.coords)

    traj_a_inbound_id, traj_a_outbound_id, traj_b_inbound_id, traj_b_outbound_id = None, None, None, None
    for index, lane in enumerate(inbound_lanes):
        if inbound_lane_linestring_list[index].intersects(traj_a_linestring):
            traj_a_inbound_id = lane[0]
        if inbound_lane_linestring_list[index].intersects(traj_b_linestring):
            traj_b_inbound_id = lane[0]
    for index, lane in enumerate(outbound_lanes):
        if outbound_lane_linestring_list[index].intersects(traj_a_linestring):
            traj_a_outbound_id = lane[0]
        if outbound_lane_linestring_list[index].intersects(traj_b_linestring):
            traj_b_outbound_id = lane[0]

    # keep detecting conflict only if all in/outbound lane ids are valid
    if  traj_a_inbound_id and traj_a_outbound_id and traj_b_inbound_id and traj_b_outbound_id:
        # ---------------------
        # detect cross conflict
        if (traj_a_inbound_id != traj_b_inbound_id) and (traj_a_outbound_id != traj_b_outbound_id):
            conflict_point: Any = traj_a_linestring.intersection(traj_b_linestring)
            # check the type of `conflict_point`
            if "Point" in conflict_point.geom_type:
                conflict_point_coord= None
                if conflict_point.geom_type == "Point":
                    conflict_point: Point = conflict_point
                    conflict_point_coord = np.array([conflict_point.x, conflict_point.y])
                elif conflict_point.geom_type == "MultiPoint":
                    conflict_point: MultiPoint = conflict_point
                    conflict_point_coord = np.array([conflict_point.geoms[0].x, conflict_point.geoms[0].y])

                if conflict_point.within(intersection_area):
                    # determine the conflict based on the PET threshold
                    signed_pet, timestamp_a, timestamp_b = calculate_pet(traj_a, traj_b, conflict_point_coord)
                    if abs(signed_pet) < pet_threshold:
                        if signed_pet < 0:
                            conflict = Conflict(pet=abs(signed_pet),
                                leader_traj=traj_a, follower_traj=traj_b,
                                leader_role=RoadUser.AutomatedVehicle if a_is_av else RoadUser.HumanDrivenVehicle,
                                follower_role=RoadUser.AutomatedVehicle if b_is_av else RoadUser.HumanDrivenVehicle,
                                category=ConflictCategory.Cross, dataset_properties=dataset_properties)
                        else:
                            conflict = Conflict(pet=abs(signed_pet),
                                leader_traj=traj_b, follower_traj=traj_a,
                                leader_role=RoadUser.AutomatedVehicle if b_is_av else RoadUser.HumanDrivenVehicle,
                                follower_role=RoadUser.AutomatedVehicle if a_is_av else RoadUser.HumanDrivenVehicle,
                                category=ConflictCategory.Cross, dataset_properties=dataset_properties)
                    else:
                        conflict = None
                else:
                    conflict = None
            else:
                conflict = None
        # ---------------------
        # detect merge conflict
        elif traj_a_inbound_id != traj_b_inbound_id and traj_a_outbound_id == traj_b_outbound_id:
            # FIXME
            conflict = None
            conflict_point_coord = calculate_merge_conflict_point_coord(traj_a, traj_b, buffer)
            if isinstance(conflict_point_coord, np.ndarray) and \
                Point(conflict_point_coord).within(intersection_area):
                signed_pet, timestamp_a, timestamp_b = calculate_pet(traj_a, traj_b, conflict_point_coord)
                if abs(signed_pet) < pet_threshold:
                    if signed_pet < 0:
                        conflict = Conflict(pet=abs(signed_pet),
                                            leader_traj=traj_a, follower_traj=traj_b,
                                            leader_role=RoadUser.AutomatedVehicle if a_is_av else RoadUser.HumanDrivenVehicle,
                                            follower_role=RoadUser.AutomatedVehicle if b_is_av else RoadUser.HumanDrivenVehicle,
                                            category=ConflictCategory.Merge, dataset_properties=dataset_properties)
                    else:
                        conflict = Conflict(pet=abs(signed_pet),
                                            leader_traj=traj_b, follower_traj=traj_a,
                                            leader_role=RoadUser.AutomatedVehicle if b_is_av else RoadUser.HumanDrivenVehicle,
                                            follower_role=RoadUser.AutomatedVehicle if a_is_av else RoadUser.HumanDrivenVehicle,
                                            category=ConflictCategory.Merge, dataset_properties=dataset_properties)
                else:
                    conflict = None
            else:
                conflict = None
        else:
            conflict = None
    else:
        conflict = None
    return conflict


def calculate_pet(traj_a: Trajectory, traj_b: Trajectory, conflict_point_coords: np.array) -> Tuple[float, float, float]:
    """
    Calculate the PET between two trajectories at the conflict point.

    Args:
        traj_a: Trajectory
        traj_b: Trajectory
        conflict_point_coords:

    Returns:
        signed_pet: float, the difference between two trajectories' timestamp at the conflict point,
                                     negative if trajectory_a is the leader vehicle,
                                     positive if trajectory_a is the follower vehicle.
        timestamp_a_at_conflict: float, the timestamp of trajectory a reaching the conflict point.
        timestamp_b_at_conflict: float, the timestamp of trajectory b reaching the conflict point.
    """
    assert conflict_point_coords.shape == (2,)

    traj_a_coords, traj_b_coords = traj_a.coords, traj_b.coords
    traj_a_timestamps, traj_b_timestamps = traj_a.timestamps, traj_b.timestamps

    # timestamps of two trajectories at the conflict point, respectively
    timestamp_a_at_conflict = traj_a_timestamps[
        np.argmin(np.linalg.norm(traj_a_coords - conflict_point_coords, axis=1))]
    timestamp_b_at_conflict = traj_b_timestamps[
        np.argmin(np.linalg.norm(traj_b_coords - conflict_point_coords, axis=1))]

    # PET with a sign, indicating which is the leader vehicle
    signed_pet = timestamp_a_at_conflict - timestamp_b_at_conflict
    return signed_pet, timestamp_a_at_conflict, timestamp_b_at_conflict


def calculate_merge_conflict_point_coord(
        traj_a: Trajectory, traj_b: Trajectory, buffer: float) -> Any[None, np.ndarray]:
    """
    Calculate the merge conflict point coordinate between two trajectories.

    Args:
        traj_a: Trajectory
        traj_b: Trajectory
        buffer:

    Returns:
        coord: the coordinate of the merge conflict point, if exists, else None.
    """
    traj_a_linestring = LineString(traj_a.coords)
    traj_b_linestring = LineString(traj_b.coords)

    left_a = traj_a_linestring.parallel_offset(distance=buffer, side="left")
    right_a = traj_a_linestring.parallel_offset(distance=buffer, side="right")
    left_b = traj_b_linestring.parallel_offset(distance=buffer, side="left")
    right_b = traj_b_linestring.parallel_offset(distance=buffer, side="right")

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
            if np.argmin(np.linalg.norm(traj_a[2][:, :2] - intersection_coord1, axis=1)) <= np.argmin(
                    np.linalg.norm(traj_a[2][:, :2] - intersection_coord2, axis=1)):
                intersection_coord = intersection_coord1
            else:
                intersection_coord = intersection_coord2
        elif not is_a_valid and is_b_valid:
            intersection_coord = intersection_coord2
        elif is_a_valid and not is_b_valid:
            intersection_coord = intersection_coord1
        else:
            return None

    return intersection_coord


def filter_complex_conflicts(list_potential_conflicts: List[dict]) -> List[Conflict]:
    # retrieve a list of pairs of (leader id, follower id)
    list_2pair = [(conflict["leader_id"], conflict["follower_id"]) for conflict in list_potential_conflicts]
    # retrieve a list of tuples of (leader id, follower id, time of leader reaching objects point)
    list_3tuple = [(conflict["leader_id"], conflict["follower_id"], conflict["leader_time_at_conflict"]) for conflict in
                   list_potential_conflicts]
    # sort this list of tuples based on the time of leader vehicle reaching objects time
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
    for element in range(len(sequences) - 1):
        id_pair = (sequences[element], sequences[element + 1])
        if id_pair in list_2pair:
            list_ids.append(id_pair)
            position = list_2pair.index(id_pair)
            list_conflicts.append(list_potential_conflicts[position])

    return list_conflicts