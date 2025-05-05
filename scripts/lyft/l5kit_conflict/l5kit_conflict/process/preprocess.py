import os
import pickle
import numpy as np
import pandas as pd

from shapely.geometry import LineString, MultiLineString, Point, MultiPoint, Polygon
from typing import List, Tuple, Dict, Any, Union

from l5kit.configs.config import load_metadata, load_config_data
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import EgoDataset
from l5kit.rasterization import build_rasterizer

from ..objects.trajectory import Trajectory

# specify the environmental variable:
os.environ["L5KIT_DATA_FOLDER"] = "./"

LEFT_TOP = (-850, -875)
LEFT_BOTTOM = (-850, -925)
RIGHT_TOP = (-700, -875)
RIGHT_BOTTOM = (-700, -925)
intersection_coords = (LEFT_TOP, LEFT_BOTTOM, RIGHT_BOTTOM, RIGHT_TOP, LEFT_TOP)
intersection_area = Polygon(intersection_coords)

LEFT_TOP = (-850, -875)
LEFT_BOTTOM = (-850, -925)
RIGHT_TOP = (-700, -875)
RIGHT_BOTTOM = (-700, -925)
intersection_coords = (LEFT_TOP, LEFT_BOTTOM, RIGHT_BOTTOM, RIGHT_TOP, LEFT_TOP)
intersection_area = Polygon(intersection_coords)


# FIXME:
def extract_intersection_scenarios(
		scene_indices: list
) -> List[Tuple[int, int]]:
	intersection_scenarios1 = []

	# step1
	scene_indices.sort()
	diff = np.array(scene_indices[1:]) - np.array(scene_indices[:-1])

	for idx, val in enumerate(diff):
		if val != 1:
			pass
		elif val == 1:
			intersection_scenarios1.append((scene_indices[idx], scene_indices[idx + 1]))

	# step 2
	intersection_scenarios2 = []
	for idx, val in enumerate(intersection_scenarios1[:-1]):
		if val[1] == intersection_scenarios1[idx + 1][0]:
			intersection_scenarios2.append((val[0], val[1], intersection_scenarios1[idx + 1][1]))
		else:
			intersection_scenarios2.append(val)

	# step 3
	intersection_scenarios3 = []
	for idx, val in enumerate(intersection_scenarios2[:-1]):
		if (len(val) == 2 and len(intersection_scenarios2[idx-1]) == 2) or (len(val) == 2 and idx-1 == -1):
			intersection_scenarios3.append(val)
		elif len(val) == 3 and len(intersection_scenarios2[idx-1]) == 2:
			intersection_scenarios3.append(val)
		elif len(val) == 2 and len(intersection_scenarios2[idx-1]) == 3:
			pass

	return intersection_scenarios3


def get_rotate_matrix(angle_for_rotation: float) -> np.ndarray:
	"""
	Get a 2 by 2 rotation matrix for coordinate transformation, given the angle for rotation in degree .
	"""
	rotate_matrix = np.array([
		[np.cos(np.deg2rad(angle_for_rotation)), np.sin(np.deg2rad(angle_for_rotation))],
		[-1 * np.sin(np.deg2rad(angle_for_rotation)), np.cos(np.deg2rad(angle_for_rotation))]
	])
	assert rotate_matrix.shape == (2, 2), \
		f"Check the shape of the rotate matrix, not (2,2), but {rotate_matrix.shape}"
	return rotate_matrix


def get_scene_ego_trajectory(
		dataset: EgoDataset,
		dataset_name: str,
		scene_index: int,
) -> Trajectory:
	""" Return the ego trajectory for a given scene index. """
	assert isinstance(dataset, EgoDataset)
	ego_trajectory_xy = np.zeros((1, 2))
	frame_indices = dataset.get_scene_indices(scene_idx=scene_index)

	# iterate to retrieve the centroids of ego vehicle across all the frames within a scene
	for frame_index in frame_indices:
		frame = dataset[frame_index]
		centroid = frame["centroid"].reshape((1, 2))
		ego_trajectory_xy = np.append(arr=ego_trajectory_xy, values=centroid, axis=0)

	ego_trajectory_xy = ego_trajectory_xy[1:, :]  # shape (N_frame, 2)
	ego_trajectory_t = np.arange(ego_trajectory_xy.shape[0])

	assert ego_trajectory_xy.shape[1] == 2
	assert ego_trajectory_t.shape[0] == ego_trajectory_xy.shape[0]

	# create a ego trajectory object
	ego_trajectory = Trajectory(
		scene_indices=scene_index,
		agent_index=-1,
		trajectory_xy=ego_trajectory_xy,
		trajectory_t=ego_trajectory_t,
		dataset=dataset_name
	)
	return ego_trajectory


def transform_scene_ego_trajectory(
		ego_trajectory: Trajectory,
		rotate_matrix: np.ndarray
) -> Trajectory:
	"""
	Transform the WORLD coordinates into the ROTATED coordinates, using a rotate matrix.
	"""
	assert isinstance(ego_trajectory, Trajectory)
	assert rotate_matrix.shape == (2, 2)

	rotated_ego_trajectory_xy = np.matmul(rotate_matrix, ego_trajectory.trajectory_xy.T).T  # shape (N_frame, 2)
	assert rotated_ego_trajectory_xy.shape == ego_trajectory.trajectory_xy.shape
	ego_trajectory.trajectory_xy = rotated_ego_trajectory_xy
	return ego_trajectory


def get_scene_agent_trajectories(
		dataset: EgoDataset,
		dataset_name: str,
		scene_index: int,
		scene_filter_threshold: float
) -> Dict[int, Trajectory]:
	""" Return the agent trajectories for a given scene index. """
	assert isinstance(dataset, EgoDataset)

	agents_trajectories = {}
	frame_indices = dataset.get_scene_indices(scene_idx=scene_index)

	# iterate to retrieve the centroids of ego vehicle across all the frames within a scene
	for time_index, frame_index in enumerate(frame_indices):
		frame = dataset[frame_index]
		agent_centroids = frame["agents_centroid"]

		for agent_id, centroid in agent_centroids.items():
			if agent_id not in agents_trajectories.keys():
				agents_trajectories[agent_id] = {
					"traj": centroid.reshape((1, 2)),
					"time": [time_index]
				}
			else:
				agents_trajectories[agent_id]["traj"] = np.append(
					arr=agents_trajectories[agent_id]["traj"],
					values=centroid.reshape((1, 2)), axis=0)
				agents_trajectories[agent_id]["time"].append(time_index)

	if scene_filter_threshold > 0:
		filtered_agents_trajectories = {}
		for agent_id, agent_attr in agents_trajectories.items():
			if agent_attr["traj"].shape[0] >= scene_filter_threshold:
				filtered_agents_trajectories[agent_id] = agent_attr
	else:
		filtered_agents_trajectories = agents_trajectories

	# create agents' trajectory objects
	agents_trajectories = {}
	for agent_id, agent_attr in filtered_agents_trajectories.items():
		agent_trajectory_xy = agent_attr["traj"]
		agent_trajectory_t = np.array(agent_attr["time"])
		assert agent_trajectory_xy.shape[0] == agent_trajectory_t.shape[0]
		agents_trajectories[agent_id] = Trajectory(
			scene_indices=scene_index,
			agent_index=None,
			trajectory_xy=agent_trajectory_xy,
			trajectory_t=agent_trajectory_t,
			dataset=dataset_name
		)

	return agents_trajectories


def transform_scene_agent_trajectories(
		agents_trajectories: Dict[int, Trajectory],
		rotate_matrix: np.ndarray
) -> Dict[int, Trajectory]:
	for agent_id, agent_trajectory in agents_trajectories.items():
		rotated_agent_trajectory_xy = np.matmul(rotate_matrix, agent_trajectory.trajectory_xy.T).T  # shape (N_frame, 2)
		assert rotated_agent_trajectory_xy.shape == agent_trajectory.trajectory_xy.shape
		agents_trajectories[agent_id].trajectory_xy = rotated_agent_trajectory_xy
	return agents_trajectories


def filter_scene_agent_trajectory(
		scene_agent_trajectories: Dict[int, Trajectory],
		length_filter_threshold: float = 20
) -> Dict[int, Trajectory]:
	removedKeys = []
	for agent_id, agent_traj in scene_agent_trajectories.items():
		# create a LineString object
		agent_trajectory_lineString = agent_traj.get_lineString()
		# filter by comparing the length threshold
		if agent_trajectory_lineString.length < length_filter_threshold:
			removedKeys.append(agent_id)
	# remove keys
	for key in removedKeys:
		scene_agent_trajectories.pop(key)

	return scene_agent_trajectories


def is_same_trajectory_in_two_frame(
		traj_a: Trajectory,
		traj_b: Trajectory,
		distance_filter_threshold: float = 2.5):
	# step 1: check if time is continuous
	if traj_a.trajectory_t[-1] + 1 == traj_a.trajectory_xy.shape[0] and traj_b.trajectory_t[0] == 0:
		# step 2: check if space is continuous
		distance = np.sqrt(np.sum(np.square(traj_a.trajectory_xy[-1, :] - traj_b.trajectory_xy[-1, :])))
		return True if distance < distance_filter_threshold else False
	else:
		return False


def get_process_trajectories(
		dataset: EgoDataset,
		dataset_name: str,
		scene_indices: Union[Tuple[int, int], Tuple[int, int, int]],
		angle_for_rotation: float = 56.5):
	"""
	Concatenate the trajectories between 2 or 3 consecutive scenes,
	and return the ego trajectory and agent trajectories.
	"""
	# determine how many consecutive scenes are there to be concatenated.
	n_scenes = len(scene_indices)
	assert n_scenes == 2 or n_scenes == 3

	# get the rotate matrix for transformation.
	rotate_matrix = get_rotate_matrix(angle_for_rotation=angle_for_rotation)

	""" extract the ego and agent trajectories in the first scene """
	first_scene_index = scene_indices[0]

	# ego trajectory in the first scene
	first_scene_ego_trajectory = get_scene_ego_trajectory(
		dataset=dataset, dataset_name=dataset_name, scene_index=first_scene_index)
	first_scene_ego_trajectory = transform_scene_ego_trajectory(
		ego_trajectory=first_scene_ego_trajectory, rotate_matrix=rotate_matrix)

	# agent trajectories in the first scene
	first_scene_agent_trajectories = get_scene_agent_trajectories(
		dataset=dataset, dataset_name=dataset_name, scene_index=first_scene_index, scene_filter_threshold=90)
	first_scene_agent_trajectories = transform_scene_agent_trajectories(
		agents_trajectories=first_scene_agent_trajectories, rotate_matrix=rotate_matrix)
	# first_scene_agent_trajectories = filter_scene_agent_trajectory(
	# 	scene_agent_trajectories=first_scene_agent_trajectories)

	""" extract the ego and agent trajectories in the second scene """
	second_scene_index = scene_indices[1]

	# ego trajectory in the second scene
	second_scene_ego_trajectory = get_scene_ego_trajectory(
		dataset=dataset, dataset_name=dataset_name, scene_index=second_scene_index)
	second_scene_ego_trajectory = transform_scene_ego_trajectory(
		ego_trajectory=second_scene_ego_trajectory, rotate_matrix=rotate_matrix)

	# agent trajectories in the second scene
	second_scene_agent_trajectories = get_scene_agent_trajectories(
		dataset=dataset, dataset_name=dataset_name, scene_index=second_scene_index, scene_filter_threshold=90)
	second_scene_agent_trajectories = transform_scene_agent_trajectories(
		agents_trajectories=second_scene_agent_trajectories, rotate_matrix=rotate_matrix)
	# second_scene_agent_trajectories = filter_scene_agent_trajectory(
	# 	scene_agent_trajectories=second_scene_agent_trajectories)

	""" concatenate two scenes """
	# concatenate ego trajectory_xy between the first and second scenes
	concatenated_ego_trajectory_xy = np.concatenate([
		first_scene_ego_trajectory.trajectory_xy,
		second_scene_ego_trajectory.trajectory_xy])
	# concatenate ego trajectory_t between the first and second scenes
	concatenated_ego_trajectory_t = np.arange(concatenated_ego_trajectory_xy.shape[0])
	# create a concatenated ego trajectory object
	concatenated_ego_trajectory = Trajectory(
		scene_indices=scene_indices,
		agent_index=-1,
		trajectory_xy=concatenated_ego_trajectory_xy,
		trajectory_t=concatenated_ego_trajectory_t,
		dataset=dataset_name,
	)

	# record the number of frames in the first scene
	len_first_scene = first_scene_ego_trajectory.trajectory_xy.shape[0]

	# concatenate agent trajectories between the first and second scenes
	concatenated_agent_trajectories = {}
	same_agent_id_pairs = []

	# determine the continuous trajectories across two scenes
	for agent_id_1st, agent_traj_1st in first_scene_agent_trajectories.items():
		for agent_id_2nd, agent_traj_2nd in second_scene_agent_trajectories.items():
			if is_same_trajectory_in_two_frame(traj_a=agent_traj_1st, traj_b=agent_traj_2nd):
				same_agent_id_pairs.append((agent_id_1st, agent_id_2nd))

	# concatenate the continuous trajectories across two scenes
	for same_agent_id_pair in same_agent_id_pairs:
		# retrieve agent trajectory a and b:
		agent_a_trajectory: Trajectory = first_scene_agent_trajectories[same_agent_id_pair[0]]
		agent_b_trajectory: Trajectory = second_scene_agent_trajectories[same_agent_id_pair[1]]

		# remove from the original dicts
		first_scene_agent_trajectories.pop(same_agent_id_pair[0])
		second_scene_agent_trajectories.pop(same_agent_id_pair[1])

		# concate the trajectory_xy
		agent_a_trajectory_xy: np.ndarray = agent_a_trajectory.trajectory_xy
		agent_b_trajectory_xy: np.ndarray = agent_b_trajectory.trajectory_xy
		concatenated_agent_trajectory_xy = np.concatenate([agent_a_trajectory_xy, agent_b_trajectory_xy])
		assert concatenated_agent_trajectory_xy.shape[0] == agent_a_trajectory_xy.shape[0] + agent_b_trajectory_xy.shape[0]

		# concate the trajectory_t
		agent_a_trajectory_t: np.ndarray = agent_a_trajectory.trajectory_t
		agent_b_trajectory_t: np.ndarray = agent_b_trajectory.trajectory_t + agent_a_trajectory_t[-1]
		concatenated_agent_trajectory_t = np.concatenate([agent_a_trajectory_t, agent_b_trajectory_t])
		assert concatenated_agent_trajectory_t.shape[0] == agent_a_trajectory_t.shape[0] + agent_b_trajectory_t.shape[0]

		# create a concatenated agent trajectory object
		concatenated_agent_trajectories[len(concatenated_agent_trajectories)] = Trajectory(
			scene_indices=scene_indices,
			agent_index=len(concatenated_agent_trajectories),
			trajectory_xy=concatenated_agent_trajectory_xy,
			trajectory_t=concatenated_agent_trajectory_t,
			dataset=dataset_name,
		)

	# process agent trajectories that only appear in the first scene
	for agent_id, agent_traj in first_scene_agent_trajectories.items():
		agent_index = len(concatenated_agent_trajectories)
		concatenated_agent_trajectories[agent_index] = agent_traj
		concatenated_agent_trajectories[agent_index].agent_index = agent_index

	# process agent trajectories that only appear in the second scene
	for agent_id, agent_traj in second_scene_agent_trajectories.items():
		agent_traj.trajectory_t = agent_traj.trajectory_t + len_first_scene
		agent_index = len(concatenated_agent_trajectories)
		concatenated_agent_trajectories[agent_index] = agent_traj
		concatenated_agent_trajectories[agent_index].agent_index = agent_index

	""" extract the ego and agent trajectories in the third scene (optional) """
	if n_scenes == 3:
		third_scene_index = scene_indices[2]

		# ego trajectory in the third scene
		third_scene_ego_trajectory = get_scene_ego_trajectory(
			dataset=dataset, dataset_name=dataset_name, scene_index=third_scene_index)
		third_scene_ego_trajectory = transform_scene_ego_trajectory(
			ego_trajectory=third_scene_ego_trajectory, rotate_matrix=rotate_matrix)

		# agent trajectories in the third scene
		third_scene_agent_trajectories = get_scene_agent_trajectories(
			dataset=dataset, dataset_name=dataset_name, scene_index=third_scene_index, scene_filter_threshold=90)
		third_scene_agent_trajectories = transform_scene_agent_trajectories(
			agents_trajectories=third_scene_agent_trajectories, rotate_matrix=rotate_matrix)
		# third_scene_agent_trajectories = filter_scene_agent_trajectory(third_scene_agent_trajectories)

		""" concatenate two scenes """
		# concatenate ego trajectory_xy between the first two and the third scenes
		concatenated_ego_trajectory_xy = np.concatenate([
			concatenated_ego_trajectory.trajectory_xy,
			third_scene_ego_trajectory.trajectory_xy])
		# concatenate ego trajectory_t between the first and second scenes
		concatenated_ego_trajectory_t = np.arange(concatenated_ego_trajectory_xy.shape[0])
		# create a concatenated ego trajectory object
		concatenated_ego_trajectory = Trajectory(
			scene_indices=scene_indices,
			agent_index=-1,
			trajectory_xy=concatenated_ego_trajectory_xy,
			trajectory_t=concatenated_ego_trajectory_t,
			dataset=dataset_name,
		)

		# record the number of frames in the first scene
		len_first2_scene = first_scene_ego_trajectory.trajectory_xy.shape[0]

		# concatenate agent trajectories between the first and second scenes
		concatenated_agent_trajectories_optional = {}
		same_agent_id_pairs = []

		# determine the continuous trajectories across two scenes
		for agent_id_1st, agent_traj_1st in concatenated_agent_trajectories.items():
			for agent_id_2nd, agent_traj_2nd in third_scene_agent_trajectories.items():
				if is_same_trajectory_in_two_frame(traj_a=agent_traj_1st, traj_b=agent_traj_2nd):
					same_agent_id_pairs.append((agent_id_1st, agent_id_2nd))

		# concatenate the continuous trajectories across two scenes
		for same_agent_id_pair in same_agent_id_pairs:
			# retrieve agent trajectory a and b:
			agent_a_trajectory: Trajectory = concatenated_agent_trajectories[same_agent_id_pair[0]]
			agent_b_trajectory: Trajectory = third_scene_agent_trajectories[same_agent_id_pair[1]]

			# remove from the original dicts
			concatenated_agent_trajectories.pop(same_agent_id_pair[0])
			third_scene_agent_trajectories.pop(same_agent_id_pair[1])

			# concate the trajectory_xy
			agent_a_trajectory_xy: np.ndarray = agent_a_trajectory.trajectory_xy
			agent_b_trajectory_xy: np.ndarray = agent_b_trajectory.trajectory_xy
			concatenated_agent_trajectory_xy = np.concatenate([agent_a_trajectory_xy, agent_b_trajectory_xy])
			assert concatenated_agent_trajectory_xy.shape[0] == agent_a_trajectory_xy.shape[0] + agent_b_trajectory_xy.shape[0]

			# concate the trajectory_t
			agent_a_trajectory_t: np.ndarray = agent_a_trajectory.trajectory_t
			agent_b_trajectory_t: np.ndarray = agent_b_trajectory.trajectory_t + agent_a_trajectory_t[-1]
			concatenated_agent_trajectory_t = np.concatenate([agent_a_trajectory_t, agent_b_trajectory_t])
			assert concatenated_agent_trajectory_t.shape[0] == agent_a_trajectory_t.shape[0] + agent_b_trajectory_t.shape[0]

			# create a concatenated agent trajectory object
			concatenated_agent_trajectories_optional[len(concatenated_agent_trajectories_optional)] = Trajectory(
				scene_indices=scene_indices,
				agent_index=len(concatenated_agent_trajectories_optional),
				trajectory_xy=concatenated_agent_trajectory_xy,
				trajectory_t=concatenated_agent_trajectory_t,
				dataset=dataset_name,
			)

		# process agent trajectories that only appear in the first scene
		for agent_id, agent_traj in concatenated_agent_trajectories.items():
			agent_index = len(concatenated_agent_trajectories_optional)
			concatenated_agent_trajectories_optional[agent_index] = agent_traj
			concatenated_agent_trajectories_optional[agent_index].agent_index = agent_index

		# process agent trajectories that only appear in the second scene
		for agent_id, agent_traj in third_scene_agent_trajectories.items():
			agent_index = len(concatenated_agent_trajectories_optional)
			agent_traj.trajectory_t = agent_traj.trajectory_t + len_first2_scene
			concatenated_agent_trajectories_optional[agent_index] = agent_traj
			concatenated_agent_trajectories_optional[agent_index].agent_index = agent_index

		concatenated_agent_trajectories = concatenated_agent_trajectories_optional

	return concatenated_ego_trajectory, concatenated_agent_trajectories
