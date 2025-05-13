import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # disable tensorflow warnings, not related.
import tensorflow as tf
from waymo_open_dataset.protos import scenario_pb2, map_pb2
from src.intersection_filter.waymo.utils import *
from src.detection.detect_conflict import identify_conflict

txt_path="./processed/waymo/scenario_4_stop_signs.csv"
version = "v1.2.1"
distance_threshold = 45
buffer = 10
pet_threshold = 10
metadatas = get_all_indices_pairs(txt_path)

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

    ego_trajectory: Trajectory = get_ego_trajectory_from_scenario(scenario=scenario)
    break