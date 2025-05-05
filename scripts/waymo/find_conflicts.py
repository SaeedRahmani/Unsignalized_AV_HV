import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
np.set_printoptions(suppress=True)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from shapely import Polygon, LineString

from waymo_open_dataset.protos import scenario_pb2, map_pb2
from waymo_devkit.query import get_intersection_stopSigns, get_intersection_circle, get_intersection_lanes
from waymo_devkit.query import get_egoTrajectory, get_vehicleTrajectories, get_scenario_metadata
from waymo_devkit.shape import construct_intersection_polygon
from waymo_devkit.identify import identify_conflict, identify_complex_conflicts
# from waymo_devkit.visualize import visualize_gif, visualize_map, visualize_traj

all_conflicts = []
all_merge_conflicts = []
all_cross_conflicts = []
scenario_set = set()
# NUM_CROSS, NUM_MERGE = 0, 0

txt_path="./outputs/scenario_metadata/3stopSigns.csv"
metadatas = get_scenario_metadata(txt_path)
distance_threshold = 45
buffer = 10
pet = 10

for scene_id, (tfrecord_index, scenario_index) in enumerate(metadatas):
    # print(scene_id, (tfrecord_index, int(scenario_index)))
    scene_conflicts = []

    tfrecord_path = f"./training_20s/training_20s.tfrecord-{tfrecord_index}-of-01000"
    # get scenario object
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    for index, scenario_proto in enumerate(dataset):
        if index == int(scenario_index): 
            scenario_proto = scenario_proto.numpy()
            scenario = scenario_pb2.Scenario()
            scenario.ParseFromString(scenario_proto)
            break

    """ extract info in scenario """
    # get map elements
    stopSigns = get_intersection_stopSigns(scenario)
    intersection_centerCoordinate, intersection_radius = get_intersection_circle(stopSigns, buffer=buffer)
    intersection_polygon = construct_intersection_polygon(intersection_centerCoordinate, intersection_radius)

    inbound_lanes, outbound_lanes = get_intersection_lanes(scenario, intersection_polygon)

    # get trajectories
    egoTrajectory = get_egoTrajectory(scenario)
    vehicleTrajectories = get_vehicleTrajectories(scenario, intersection_polygon)
    # n_legs=3 if len(stopSigns)==3 else 4
    # visualize_map(scenario, tfrecord_index, int(scenario_index), n_legs, distance_threshold=distance_threshold, buffer=buffer)
    
    """ identify conflict """
    # AV-HV
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
                        
    # HV-HV
    for i, veh_trajectory1 in enumerate(vehicleTrajectories):
        for j, veh_trajectory2 in enumerate(vehicleTrajectories):
            if i != j:
                conflict_type, c = identify_conflict(
                    veh_trajectory1, veh_trajectory2,
                    inbound_lanes, outbound_lanes, False, False, 
                    intersection_circle=intersection_polygon, 
                    center=intersection_centerCoordinate, radius=intersection_radius,
                    PET=15,
                    tfrecord_index=tfrecord_index, scenario_index=scenario_index,
                )
                # @HV-HV: merge
                if conflict_type == "MERGE":
                    # print(tfrecord_index, int(scenario_index), conflict_type, egoTrajectory[0], veh_trajectory[0])
                    # NUM_MERGE += 1
                    scene_conflicts.append(c)
                # @HV-HV: cross
                elif conflict_type == "CROSS":
                    # print(tfrecord_index, int(scenario_index), conflict_type, egoTrajectory[0], veh_trajectory[0])
                    # NUM_CROSS += 1
                    scene_conflicts.append(c)

    # if exists complex conflict:
    if len(scene_conflicts) == 1:
        all_conflicts.append(scene_conflicts[0])
        scene_conflicts[0]["scenario_index"] = scenario_index
        scene_conflicts[0]["tfrecord_index"] = tfrecord_index
        scene_conflicts[0]["scene_index"] =  scene_index
        # visualize_gif()
        print(tfrecord_index, scenario_index)
        if scene_conflicts[0]["conflict_type"] == "MERGE":
            all_merge_conflicts.append(scene_conflicts[0])
        elif scene_conflicts[0]["conflict_type"] == "CROSS":
            all_cross_conflicts.append(scene_conflicts[0])
            
    elif len(scene_conflicts) > 1:
        scene_conflicts = identify_complex_conflicts(scene_conflicts)
        # visualize_gif()
        for scene_index, c in enumerate(scene_conflicts):
            c["scenario_index"] = scenario_index
            c["tfrecord_index"] = tfrecord_index
            c["scene_index"] =  scene_index
            print(tfrecord_index, scenario_index)
            all_conflicts.append(c)
            if c["conflict_type"] == "MERGE":
                all_merge_conflicts.append(c)
            elif c["conflict_type"] == "CROSS":
                all_cross_conflicts.append(c)

txt_path="./outputs/scenario_metadata/4stopSigns.csv"
metadatas = get_scenario_metadata(txt_path)
distance_threshold = 45
buffer = 5

for scene_id, (tfrecord_index, scenario_index) in enumerate(metadatas):
    # print(scene_id, (tfrecord_index, scenario_index))
    scene_conflicts = []

    tfrecord_path = f"./training_20s/training_20s.tfrecord-{tfrecord_index}-of-01000"
    # get scenario object
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    for index, scenario_proto in enumerate(dataset):
        if index == int(scenario_index): 
            scenario_proto = scenario_proto.numpy()
            scenario = scenario_pb2.Scenario()
            scenario.ParseFromString(scenario_proto)
            break

    """ extract info in scenario """
    # get map elements
    stopSigns = get_intersection_stopSigns(scenario)
    intersection_centerCoordinate, intersection_radius = get_intersection_circle(stopSigns, buffer=buffer)
    intersection_polygon = construct_intersection_polygon(intersection_centerCoordinate, intersection_radius)

    inbound_lanes, outbound_lanes = get_intersection_lanes(scenario, intersection_polygon)

    # get trajectories
    egoTrajectory = get_egoTrajectory(scenario)
    vehicleTrajectories = get_vehicleTrajectories(scenario, intersection_polygon)
    # n_legs=3 if len(stopSigns)==3 else 4
    # visualize_traj(scenario, tfrecord_index, int(scenario_index), distance_threshold=distance_threshold)
    
    """ identify conflict """
    # AV-HV
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
                        
    # HV-HV
    for i, veh_trajectory1 in enumerate(vehicleTrajectories):
        for j, veh_trajectory2 in enumerate(vehicleTrajectories):
            if i > j:
                conflict_type, c = identify_conflict(
                    veh_trajectory1, veh_trajectory2,
                    inbound_lanes, outbound_lanes,
                    False, False, 
                    intersection_circle=intersection_polygon, 
                    center=intersection_centerCoordinate, radius=intersection_radius,
                    PET=15,          
                    tfrecord_index=tfrecord_index, scenario_index=scenario_index,      
                )
                # @HV-HV: merge
                if conflict_type == "MERGE":
                    # NUM_MERGE += 1
                    scene_conflicts.append(c)
                # @HV-HV: cross
                elif conflict_type == "CROSS":
                    # NUM_CROSS += 1
                    scene_conflicts.append(c)

    # if exists complex conflict:
    if len(scene_conflicts) == 1:
        scene_conflicts[0]["scenario_index"] = scenario_index
        scene_conflicts[0]["tfrecord_index"] = tfrecord_index
        scene_conflicts[0]["scene_index"] =  scene_index
        all_conflicts.append(scene_conflicts[0])
        # visualize_gif()
        print(tfrecord_index, scenario_index)
        if scene_conflicts[0]["conflict_type"] == "MERGE":
            all_merge_conflicts.append(scene_conflicts[0])
        elif scene_conflicts[0]["conflict_type"] == "CROSS":
            all_cross_conflicts.append(scene_conflicts[0])
            
    elif len(scene_conflicts) > 1:
        scene_conflicts = identify_complex_conflicts(scene_conflicts)
        # visualize_gif()
        for scene_index, c in enumerate(scene_conflicts):
            c["scenario_index"] = scenario_index
            c["tfrecord_index"] = tfrecord_index
            c["scene_index"] =  scene_index
            print(tfrecord_index, scenario_index)
            all_conflicts.append(c)
            if c["conflict_type"] == "MERGE":
                all_merge_conflicts.append(c)
            elif c["conflict_type"] == "CROSS":
                all_cross_conflicts.append(c)


import pickle
with open('cross_conflict_pet15s.pkl', 'wb') as file:
    pickle.dump(all_cross_conflicts, file)
with open('merge_conflict_pet15s.pkl', 'wb') as file:
    pickle.dump(all_merge_conflicts, file)
with open('conflict_pet15s.pkl', 'wb') as file:
    pickle.dump(all_conflicts, file)

print(f"#Cross conflicts: {len(all_cross_conflicts)}")
print(f"#Merge conflicts: {len(all_merge_conflicts)}")
