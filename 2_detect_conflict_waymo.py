import tensorflow as tf
from waymo_open_dataset.protos import scenario_pb2, map_pb2
from src.datasets.waymo.utils import *
from src.core.conflict import identify_conflict

txt_path="./processed/waymo/scenario_4_stop_signs.csv"
version = "v1.2.1"
distance_threshold = 45
buffer = 10
pet = 10

def get_index_pair(txt_path: str):
    metadatas = []
    with open(txt_path, "r") as f:
        for line in f:
            tfrecord_id, scenario_id = str(line).split(",")
            metadatas.append((tfrecord_id, int(scenario_id)))
    return metadatas

metadatas = get_index_pair(txt_path)

all_conflicts = []
all_merge_conflicts = []
all_cross_conflicts = []

for scene_id, (tfrecord_index, scenario_index) in enumerate(metadatas):
    print(scene_id, (tfrecord_index, int(scenario_index)))
    scene_conflicts = []

    tfrecord_path = f"./raw_data/waymo/{version}_training_20s/training_20s.tfrecord-{tfrecord_index}-of-01000"
    # get scenario object
    dataset = tf.data.TFRecordDataset(tfrecord_path)

    scenario = None
    for index, scenario_proto in enumerate(dataset):
        if index == int(scenario_index):
            scenario_proto = scenario_proto.numpy()
            scenario = scenario_pb2.Scenario()
            scenario.ParseFromString(scenario_proto)
            break

    stop_signs = get_intersection_stop_signs(scenario=scenario, distance_threshold=distance_threshold)
    intersection_centerCoordinate, intersection_radius = get_intersection_circle(stop_signs, buffer=buffer)
    intersection_polygon = construct_intersection_polygon(intersection_centerCoordinate, intersection_radius)

    inbound_lanes, outbound_lanes = get_intersection_lanes(scenario, intersection_polygon)

    # get trajectories
    egoTrajectory = get_ego_trajectory(scenario)
    vehicleTrajectories = get_vehicle_trajectories(scenario, intersection_polygon)

    # """ identify objects """
    # # AV-HV
    for veh_trajectory in vehicleTrajectories:
        conflict_type, c = identify_conflict(
            egoTrajectory, veh_trajectory,
            inbound_lanes, outbound_lanes, True, False,
            intersection_circle=intersection_polygon,
            center=intersection_centerCoordinate, radius=intersection_radius,
            PET=15,
            tfrecord_index=tfrecord_index, scenario_index=scenario_index,
        )
        # @AV-HV: merge
        if conflict_type == "MERGE":
            scene_conflicts.append(c)
        # @AV-HV: cross
        elif conflict_type == "CROSS":
            scene_conflicts.append(c)
        all_conflicts.append(c)
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
    #                 # print(tfrecord_index, int(scenario_index), conflict_type, egoTrajectory[0], veh_trajectory[0])
    #                 # NUM_MERGE += 1
    #                 scene_conflicts.append(c)
    #             # @HV-HV: cross
    #             elif conflict_type == "CROSS":
    #                 # print(tfrecord_index, int(scenario_index), conflict_type, egoTrajectory[0], veh_trajectory[0])
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

print(f"#Total conflicts: {len(all_conflicts)}")