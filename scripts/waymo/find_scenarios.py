import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tqdm import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
from shapely import LineString, Polygon 

from waymo_open_dataset.protos import scenario_pb2 
from waymo_devkit.query import get_egoTrajectory, get_intersection_stopSigns
from waymo_devkit.visualize import visualize_map


# prepare the tfrecord file paths
dataset_directory = f"/mnt/u/wsl/Waymo/waymo_v1.2/training_20s/"  # 20s training dataset
all_tfrecord_names = os.listdir(dataset_directory)
all_tfrecord_paths = [dataset_directory + str(name) for name in all_tfrecord_names]

NUM_3stopSigns, df_3stopSigns = 0, []
NUM_4stopSigns, df_4stopSigns  = 0, []

distance_threshold = 45

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

        stopSigns = get_intersection_stopSigns(scenario, distance_threshold=distance_threshold)
        if len(stopSigns) >= 3:
            # construct shapely objects
            intersectionPolygon = Polygon([ss[1] for ss in stopSigns])
            ego_trajectory = get_egoTrajectory(scenario)
            ego_trajectoryLineString = LineString(ego_trajectory[2][:,:2])
            
            if len(stopSigns) == 3 and intersectionPolygon.intersects(ego_trajectoryLineString):
                NUM_3stopSigns += 1
                visualize_map(scenario, tfrecord_id, scenario_index, 3, distance_threshold)
                df_3stopSigns.append([str(tfrecord_id), scenario_index])
            elif len(stopSigns) >= 4 and intersectionPolygon.intersects(ego_trajectoryLineString):
                NUM_4stopSigns += 1
                visualize_map(scenario, tfrecord_id, scenario_index, 4, distance_threshold)
                df_4stopSigns.append([str(tfrecord_id), scenario_index])

df_3stopSigns = pd.DataFrame(df_3stopSigns) # , columns=["TFRecord_ID", "Scene_ID"]
df_4stopSigns = pd.DataFrame(df_4stopSigns) # , columns=["TFRecord_ID", "Scene_ID"]
df_3stopSigns.to_csv("./outputs/scenario_metadata/3stopSigns.csv", index=False)
df_4stopSigns.to_csv("./outputs/scenario_metadata/4stopSigns.csv", index=False)

print(f"3 stopSigns: {NUM_3stopSigns}")
print(f"4 stopSigns: {NUM_4stopSigns}")