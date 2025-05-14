import tensorflow as tf
from typing import List
from shapely import LineString, Polygon
from waymo_open_dataset.protos.scenario_pb2 import Scenario
from src.core.detect_conflict import detect_conflict_between_two_trajectories
from src import Trajectory, Conflict
from .load_from_proto import *
from . import StopSign



def detect_all_conflicts(
    version: str ="v1.2.1",
    distance_threshold: float = 45,
    buffer: float = 10,
    pet_threshold: float = 10,
    num_stop_signs: int = 4,
):
    """
    Detect all the conflicts in the Waymo Motion Dataset.

    Args:
        version: str, the version of the Waymo Motion dataset to load, e.g., "v1.2.1" or "v1.3.0"
        distance_threshold:
        buffer:
        pet_threshold: float, Post-encroachment time threshold
        num_stop_signs: float

    Returns:

    """

    all_conflicts_list, all_merge_conflicts_list, all_cross_conflicts_list = list(), list(), list()

    # Detect all conflicts from the intersections with 4 stop signs nearby
    csv_path=f"./processed/waymo/scenario_{num_stop_signs}_stop_signs.csv"
    for scene_idx, (tfrecord_id, scenario_id) in enumerate(get_all_indices_pairs(csv_path)):
        all_conflicts_this_scene_list = list()
        print(scene_idx, (tfrecord_id, int(scenario_id)))

        # load the dataset
        tfrecord_path = f"./raw_data/waymo/{version}_training_20s/training_20s.tfrecord-{tfrecord_id}-of-01000"
        dataset = tf.data.TFRecordDataset(tfrecord_path)

        # find the scenario
        scenario: Scenario = None
        for index, scenario_proto in enumerate(dataset):
            if index == int(scenario_id):
                scenario = Scenario()
                scenario.ParseFromString(scenario_proto.numpy())
                break

        # load trajectory and map elements from the scenario
        stop_signs: List[StopSign] = get_intersection_stop_signs_from_scenario(
            scenario=scenario, distance_threshold=distance_threshold)
        intersection_coords, intersection_radius = get_intersection_circle(
            intersection_stop_signs=stop_signs, buffer=buffer, aggregation="max")
        intersection_area: Polygon = build_intersection_circle_area(
            center_coords=intersection_coords, radius=intersection_radius)
        ego_trajectory: Trajectory = get_ego_trajectory_from_scenario(scenario=scenario)
        driver_trajectories: List[Trajectory] = get_driver_trajectories_from_scenario(
            scenario=scenario, intersection_area=intersection_area)
        (inbound_lanes, outbound_lanes) = get_intersection_lanes_from_scenario(
            scenario=scenario, intersection_area=intersection_area)

        # detect the conflicts
        # # AV-HV

        for driver_trajectory in driver_trajectories:
            detect_conflict_between_two_trajectories()

        #     conflict_type, c = identify_conflict(
        #         ego_trajectory, veh_trajectory,
        #         inbound_lanes, outbound_lanes, True, False,
        #         intersection_circle=intersection_polygon,
        #         center=intersection_centerCoordinate, radius=intersection_radius,
        #         PET=15,
        #         tfrecord_index=tfrecord_index, scenario_index=scenario_index,
        #     )
        #     # @AV-HV: merge
        #     if conflict_type == "MERGE":
        #         scene_conflicts.append(c)
        #     # @AV-HV: cross
        #     elif conflict_type == "CROSS":
        #         scene_conflicts.append(c)
        #     all_conflicts.append(c)
        #
        # # HV-HV
        # for i, veh_trajectory1 in enumerate(vehicleTrajectories):
        #     for j, veh_trajectory2 in enumerate(vehicleTrajectories):
        #         if i != j:
        #             conflict_type, c = identify_conflict(
        #                 veh_trajectory1, veh_trajectory2,
        #                 inbound_lanes, outbound_lanes, False, False,
        #                 intersection_circle=intersection_polygon,
        #                 center=intersection_centerCoordinate, radius=intersection_radius,
        #                 PET=15,
        #                 tfrecord_index=tfrecord_index, scenario_index=scenario_index,
        #             )
        #             # @HV-HV: merge
        #             if conflict_type == "MERGE":
        #                 # print(tfrecord_index, int(scenario_index), conflict_type, ego_trajectory[0], veh_trajectory[0])
        #                 # NUM_MERGE += 1
        #                 scene_conflicts.append(c)
        #             # @HV-HV: cross
        #             elif conflict_type == "CROSS":
        #                 # print(tfrecord_index, int(scenario_index), conflict_type, ego_trajectory[0], veh_trajectory[0])
        #                 # NUM_CROSS += 1
        #                 scene_conflicts.append(c)
        #
        # # if exists complex objects:
        # if len(scene_conflicts) == 1:
        #     all_conflicts.append(scene_conflicts[0])
        #     scene_conflicts[0]["scenario_index"] = scenario_index
        #     scene_conflicts[0]["tfrecord_index"] = tfrecord_index
        #     scene_conflicts[0]["scene_index"] = scene_index
        #     # visualize_gif()
        #     print(tfrecord_index, scenario_index)
        #     if scene_conflicts[0]["conflict_type"] == "MERGE":
        #         all_merge_conflicts.append(scene_conflicts[0])
        #     elif scene_conflicts[0]["conflict_type"] == "CROSS":
        #         all_cross_conflicts.append(scene_conflicts[0])
        #
        # elif len(scene_conflicts) > 1:
        #     scene_conflicts = identify_complex_conflicts(scene_conflicts)
        #     # visualize_gif()
        #     for scene_index, c in enumerate(scene_conflicts):
        #         c["scenario_index"] = scenario_index
        #         c["tfrecord_index"] = tfrecord_index
        #         c["scene_index"] = scene_index
        #         print(tfrecord_index, scenario_index)
        #         all_conflicts.append(c)
        #         if c["conflict_type"] == "MERGE":
        #             all_merge_conflicts.append(c)
        #         elif c["conflict_type"] == "CROSS":
        #             all_cross_conflicts.append(c)

        print(f"#Total conflicts: {len(all_conflicts_list)}")