import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
import tensorflow as tf
from shapely import LineString, Polygon
from waymo_open_dataset.protos import scenario_pb2
from .waymo.utils import get_intersection_stop_signs, get_ego_trajectory

def waymo_loader(
        distance_threshold: float = 45
):
    """
    Filter the unsignalized intersections with >= 3 stop signs nearby.

    @param: distance_threshold: float, distance threshold between two stop signs, default is 45 meters.
    """
    dataset_directory = f"./raw_data/waymo/training_20s/"  # 20s training dataset
    all_tfrecord_names = os.listdir(dataset_directory)
    all_tfrecord_paths = [dataset_directory + str(name) for name in all_tfrecord_names]

    df_3stopSigns, df_4stopSigns = [], []

    for tfrecord_path in all_tfrecord_paths[:]:

        # create the dataset object
        dataset = tf.data.TFRecordDataset(tfrecord_path)
        tfrecord_id = tfrecord_path.split("-")[-3]

        print(tfrecord_id)

        # iterate every scenario
        for scenario_index, scenario_proto in enumerate(dataset):
            # convert proto to python object
            scenario = scenario_pb2.Scenario()
            scenario.ParseFromString(scenario_proto.numpy())

            stopSigns = get_intersection_stop_signs(scenario, distance_threshold=distance_threshold)
            if len(stopSigns) >= 3:
                # construct shapely objects
                intersectionPolygon = Polygon([ss[1] for ss in stopSigns])
                ego_trajectory = get_ego_trajectory(scenario)
                ego_trajectoryLineString = LineString(ego_trajectory[2][:, :2])

                if len(stopSigns) == 3 and intersectionPolygon.intersects(ego_trajectoryLineString):
                    # visualize_map(scenario, tfrecord_id, scenario_index, 3, distance_threshold)
                    df_3stopSigns.append([str(tfrecord_id), scenario_index])
                elif len(stopSigns) >= 4 and intersectionPolygon.intersects(ego_trajectoryLineString):
                    # visualize_map(scenario, tfrecord_id, scenario_index, 4, distance_threshold)
                    df_4stopSigns.append([str(tfrecord_id), scenario_index])

    df_3stopSigns = pd.DataFrame(df_3stopSigns)  # , columns=["TFRecord_ID", "Scene_ID"]
    df_4stopSigns = pd.DataFrame(df_4stopSigns)  # , columns=["TFRecord_ID", "Scene_ID"]
    # df_3stopSigns.to_csv("./outputs/scenario_metadata/3stopSigns.csv", index=False)
    # df_4stopSigns.to_csv("./outputs/scenario_metadata/4stopSigns.csv", index=False)

    print(f"#Unsignalised intersections with 3 stop signs: {df_3stopSigns.shape[0]}")
    print(f"#Unsignalised intersections with 4 stop signs: {df_4stopSigns.shape[0]}")

    return df_3stopSigns, df_4stopSigns


if __name__ == "__main__":
    waymo_loader()
