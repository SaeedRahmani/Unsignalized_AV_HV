import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import tensorflow as tf

from typing import List
from shapely import LineString, Polygon
from waymo_open_dataset.protos.scenario_pb2 import Scenario
from src.datasets.waymo import StopSign
from src.datasets.waymo.load_from_proto import (
    get_intersection_stop_signs_from_scenario,
    get_ego_trajectory_from_scenario
)


def filter_all_unsignalized_intersections(version: str ="v1.2.1", distance_threshold: float = 45):
    """
    Filter out the all unsignalized intersections within Waymo Motion Dataset.

    Args:
        version: str, the version of the Waymo Motion dataset to load
        distance_threshold: float, the distance threshold to determine
                                    the connectivity between two stop signs
    """
    assert version in ["v1.2.1", "v1.3.0"], f"Got unsupported version {version} for Waymo Motion Dataset"
    assert distance_threshold <= 50, f"Got too larger distance threshold {distance_threshold}"

    df_3stop_signs, df_4stop_signs = [], []
    waymo_raw_dataset_path: str = f"./raw_data/waymo/{version}_training_20s/"
    tfrecord_filenames: List[str] = sorted(os.listdir(waymo_raw_dataset_path))
    tfrecord_paths = [waymo_raw_dataset_path + str(filename) for filename in tfrecord_filenames]

    for tfrecord_path in tfrecord_paths:
        # load a tfrecord file
        dataset = tf.data.TFRecordDataset(filenames=tfrecord_path)
        tfrecord_id: str = str(tfrecord_path.split("-")[-3])
        assert len(tfrecord_id) == 5
        print(tfrecord_id)

        # iterate each scenario
        for scenario_index, scenario_proto in enumerate(dataset):
            # load a scenario
            scenario = Scenario()
            scenario.ParseFromString(scenario_proto.numpy())

            # search for the collection of stop signs that are close enough
            stop_signs: List[StopSign] = get_intersection_stop_signs_from_scenario(
                scenario=scenario, distance_threshold=distance_threshold)

            # filter expected unsignalized intersection,
            # with the criteria:
            # 1. at least 3 stop signs nearby
            # 2. the ego trajectory passes the intersection circle area
            if len(stop_signs) >= 3:
                # build shapely objects for intersection conflict
                ego_trajectory_linestring: LineString = LineString(
                    get_ego_trajectory_from_scenario(scenario).coords)
                intersection_area: Polygon = Polygon([ss.coords for ss in stop_signs])

                # criteria 2: intersection overlap
                if intersection_area.intersects(ego_trajectory_linestring):
                    # criteria 1: #stop signs >= 3
                    if len(stop_signs) == 3:
                        df_3stop_signs.append([str(tfrecord_id), scenario_index])
                    elif len(stop_signs) >= 4:
                        df_4stop_signs.append([str(tfrecord_id), scenario_index])

    # aggregate all scenarios found
    df_scenario_w3_stop_signs: pd.DataFrame = pd.DataFrame(df_3stop_signs, columns=["TFRecord_ID", "Scene_ID"])
    df_scenario_w4_stop_signs: pd.DataFrame = pd.DataFrame(df_4stop_signs, columns=["TFRecord_ID", "Scene_ID"])
    df_scenario_w3_stop_signs.to_csv("./processed/waymo/scenario_3_stop_signs.csv", index=False, header=False)
    df_scenario_w4_stop_signs.to_csv("./processed/waymo/scenario_4_stop_signs.csv", index=False, header=False)
    print(f"#Unsignalised intersections with 3 stop signs: {df_scenario_w3_stop_signs.shape[0]}")
    print(f"#Unsignalised intersections with 4 stop signs: {df_scenario_w4_stop_signs.shape[0]}")
