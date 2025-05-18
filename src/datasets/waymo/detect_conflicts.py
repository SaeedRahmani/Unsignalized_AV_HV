import tensorflow as tf
import pickle
from waymo_open_dataset.protos.scenario_pb2 import Scenario
from src.core.detect_conflict import detect_conflict_between_two_trajectories, filter_complex_conflicts
# from src import Trajectory, Conflict
from .load_from_proto import *
from . import StopSign



def detect_all_conflicts(
    version: str ="v1.2.1",
    distance_threshold: float = 45,
    buffer: float = 10,
    pet_threshold: float = 10,
    num_stop_signs: int = 4,
) -> None:
    """
    Detect all the conflicts in the Waymo Motion Dataset.

    Args:
        version: str, the version of the Waymo Motion dataset to load, e.g., "v1.2.1" or "v1.3.0"
        distance_threshold:
        buffer:
        pet_threshold: float, Post-encroachment time threshold
        num_stop_signs: float
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

        # ----------------------
        # detect AV-HV conflicts
        for driver_trajectory in driver_trajectories:
            conflict = detect_conflict_between_two_trajectories(
                traj_a=ego_trajectory, traj_b=driver_trajectory,
                inbound_lanes=inbound_lanes, outbound_lanes=outbound_lanes,
                a_is_av=True, b_is_av=False,
                tfrecord_index=tfrecord_id, scenario_index=scenario_id,
                intersection_area=intersection_area,
                pet_threshold=pet_threshold,
            )
            if conflict is not None:
                all_conflicts_this_scene_list.append(conflict)
        # ----------------------
        # detect HV-HV conflicts
        for i, driver_trajectory_i in enumerate(driver_trajectories):
            for j, driver_trajectory_j in enumerate(driver_trajectories):
                if i != j:
                    conflict = detect_conflict_between_two_trajectories(
                        traj_a=driver_trajectory_i, traj_b=driver_trajectory_j,
                        inbound_lanes=inbound_lanes, outbound_lanes=outbound_lanes,
                        a_is_av=False, b_is_av=False,
                        tfrecord_index=tfrecord_id, scenario_index=scenario_id,
                        intersection_area=intersection_area,
                        pet_threshold=pet_threshold,
                    )
                    if conflict is not None:
                        all_conflicts_this_scene_list.append(conflict)
        # ----------------------
        # save if there is only one single conflict
        if len(all_conflicts_this_scene_list) == 1:
            conflict = all_conflicts_this_scene_list[0]
            all_conflicts_list.append(conflict)
            if conflict.category == "Merge":
                all_merge_conflicts_list.append(conflict)
            elif conflict.category == "Cross":
                all_cross_conflicts_list.append(conflict)
            else:
                raise ValueError(f"Unknown conflict type: {conflict.category}")
        # ----------------------
        # double-check the complex conflict orders
        elif len(all_conflicts_this_scene_list) > 1:
            checked_all_conflicts_this_scene_list = filter_complex_conflicts(all_conflicts_this_scene_list)
            for conflict in checked_all_conflicts_this_scene_list:
                all_conflicts_list.append(conflict)
                if conflict.category == "Merge":
                    all_merge_conflicts_list.append(conflict)
                elif conflict.category == "Cross":
                    all_cross_conflicts_list.append(conflict)
                else:
                    raise ValueError(f"Unknown conflict type: {conflict.category}")
    # -------
    # summary
    print(f"#Total conflicts      : {len(all_conflicts_list)}")
    print(f"#Total merge conflicts: {len(all_merge_conflicts_list)}")
    print(f"#Total cross conflicts: {len(all_cross_conflicts_list)}")
    # save to pickle
    with open("./processed/waymo/conflicts_all.pkl", "wb") as f:
        pickle.dump(all_conflicts_list, f)
    with open("./processed/waymo/conflicts_merge.pkl", "wb") as f:
        pickle.dump(all_merge_conflicts_list, f)
    with open("./processed/waymo/conflicts_cross.pkl", "wb") as f:
        pickle.dump(all_cross_conflicts_list, f)
