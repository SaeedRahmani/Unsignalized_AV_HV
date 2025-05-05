import numpy as np
import sys
# np.set_printoptions(suppress=True)
import warnings

warnings.filterwarnings("ignore")

from tqdm import tqdm
from typing import Tuple

from l5kit_conflict.pickle.io import *
from l5kit_conflict.process.preprocess import *
from l5kit_conflict.filter.trajectory_filter import *
from l5kit_conflict.filter.conflict_filter import *

# %%
if len(sys.argv) != 4 and sys.argv[1] not in ["train1", "train2", "validate", "sample"]:
	print("Usage: python conflict_identification.py [train1 | train2 | validate | sample]")
	sys.exit(1)

# %%
""" Initialize the dataset/config and load the extracted #scenes of interest  """
dataset_name = sys.argv[1]
dataset = init_dataset(dataset_name)
scene_indices, frame_indices = load_pickle_v2(dataset_name)

scene_indices.sort()
# get all the scene tuples (2 or 3-element tuple)
scene_tuples = extract_intersection_scenarios(scene_indices=scene_indices)
print(f"#tuples in {dataset_name} dataset: {scene_tuples.__len__()}")

delta_time = float(sys.argv[2])
margin = float(sys.argv[3])
# %%

# pbar = tqdm(
# 	iterable=enumerate(scene_pairs),
# 	desc=f"Identify conflicts in {dataset_name}",
# 	total=len(scene_pairs),
# 	unit="items",
# 	leave=True,
# )

# for scene_pair_index, _ in enumerate(pbar):

for scene_tuple_index, scene_tuple in enumerate(scene_tuples):

	if scene_tuple_index % 50 == 0:
		print(f"---- {scene_tuple_index} ----")

	# concate the trajectory in two scene
	ego_trajectory, agent_trajectories = get_process_trajectories(
		dataset=dataset,
		dataset_name=dataset_name,
		scene_indices=scene_tuple)

	""" Ego trajectory filter """
	# only focus on the study area
	ego_trajectory_lineString = ego_trajectory.get_lineString()
	ego_trajectory_intersection_lineString = ego_trajectory_lineString.intersection(Trajectory.intersection_area)
	ego_trajectory_intersection_lineString = multiline_to_single_line(ego_trajectory_intersection_lineString)

	# identify ego vehicle's behaviour (2 options: AV going left or AV going right):
	bIs_AV_go_left = is_AV_go_left(ego_trajectory_intersection_lineString) 
  
	""" Agent trajectory filter """
	HV_straight2right, HV_straight2left = [], []
	HV_turnleft_from_left = []
	HV_turnleft_from_top, HV_turnright_from_top = [], []

	# only focus on the study area
	agent_trajectories = filter_scene_agent_trajectory(agent_trajectories, length_filter_threshold=20)

	# identify agent vehicle's behaviour (6 behaviours)
	agent_trajectories_intersection = {}
	for agent_id, agent_traj in agent_trajectories.items():
		agent_trajectory_intersection_lineString = agent_traj.get_lineString().intersection(Trajectory.intersection_area)
		agent_trajectory_intersection_lineString = multiline_to_single_line(agent_trajectory_intersection_lineString)
		if agent_trajectory_intersection_lineString:
			agent_trajectory_intersection_lineString = multiline_to_single_line(
				agent_trajectory_intersection_lineString)
			# whether HV goes straight towards right
			if is_HV_straight2right(agent_trajectory_intersection_lineString):
				HV_straight2right.append(agent_id)
			# whether HV goes straight towards left
			elif is_HV_straight2left(agent_trajectory_intersection_lineString):
				HV_straight2left.append(agent_id)
			# whether HV turns left from top-left arm of T-junction
			elif is_HV_turnleft_from_left(agent_trajectory_intersection_lineString):
				HV_turnleft_from_left.append(agent_id)
			# whether HV turns right from bottom-center arm of T-junction
			elif is_HV_turnleft_from_top(agent_trajectory_intersection_lineString):
				HV_turnleft_from_top.append(agent_id)
			# whether HV turns left from top-right arm of T-junction
			elif is_HV_turnright_from_top(agent_trajectory_intersection_lineString):
				HV_turnright_from_top.append(agent_id)

	# %%
	""" Identify the AV-HV conflicts """
	if bIs_AV_go_left:
		""" There are 3 cases in the AV going left scenarios:
		1) HV goes straight -> AV-HV crossing
		2) HV turns right   -> AV-HV merging
		3) HV turns left    -> AV-HV crossing """
		if HV_turnleft_from_left:
			# Case #1: AV turns left + HV goes straight => crossing
			for agent_id in HV_turnleft_from_left:
				is_conflict, conflict = whether_line_conflict( # cross, use line.
					traj_a=ego_trajectory, traj_b=agent_trajectories[agent_id], delta_time=delta_time)
				if is_conflict:
					print(scene_tuples[scene_tuple_index], (-1, agent_id), "AV-goLeft+HV-topleft -> Crossing")
					AVHV_conflict_v2["cross"]["goLeft&turnLeftFromLeft"].append({scene_tuple: conflict})

		if HV_turnright_from_top:
			# Case #2: AV goes left + HV turns right from top => merging
			for agent_id in HV_turnright_from_top:
				is_conflict, conflict = whether_buffer_conflict( # merge, use buffer.
					traj_a=ego_trajectory, traj_b=agent_trajectories[agent_id],
					traj_a_side="right", traj_b_side="left", margin=margin, delta_time=delta_time)
				if is_conflict:
					print(scene_tuples[scene_tuple_index], (-1, agent_id), "AV-goLeft+HV-right -> Merging")
					AVHV_conflict_v2["merge"]["goLeft&turnRightFromTop"].append({scene_tuple: conflict})

		if HV_turnleft_from_top:
			# Case #3: AV turns left + HV turns left => crossing
			for agent_id in HV_turnleft_from_top:
				is_conflict, conflict = whether_line_conflict( # cross, use line.
					traj_a=ego_trajectory, traj_b=agent_trajectories[agent_id], delta_time=delta_time
				)
				if is_conflict:
					print(scene_tuples[scene_tuple_index], (-1, agent_id), "AV-goLeft+HV-leftleft -> Crossing")
					AVHV_conflict_v2["cross"]["goLeft&turnLeftFromTop"].append({scene_tuple: conflict})

	elif not bIs_AV_go_left:
		""" There are 1 case in the AV going right scenarios:
		Case #4: HV turn left from top -> AV-HV merging """
		if HV_turnleft_from_top:
			for agent_id in HV_straight2right:
				is_conflict, conflict = whether_buffer_conflict( # merging, use buffer
					traj_a=ego_trajectory, traj_b=agent_trajectories[agent_id],
					traj_a_side="left", traj_b_side="right", margin=margin, delta_time=delta_time
				)
				if is_conflict:
					print(scene_tuples[scene_tuple_index], (-1, agent_id), "AV-goRight+HV-turnleft -> Merging")
					AVHV_conflict_v2["merge"]["goRight&turnLeftFromTop"].append({scene_tuple: conflict})

	# %%
	""" Identify the HV-HV conflicts 
	There are 6 cases of HV-HV conflicts """

	# Case #1: 
 	# HV turns left from left arm of T-junction 2 +
	# HV turns left from bottom-center arm of T-junction 2 => crossing
	if HV_turnleft_from_left and HV_turnleft_from_top:
		for agent1_id in HV_turnleft_from_left:
			for agent2_id in HV_turnleft_from_top:
				is_conflict, conflict = whether_line_conflict(
					traj_a=agent_trajectories[agent1_id], traj_b=agent_trajectories[agent2_id], delta_time=delta_time)
				if is_conflict:
					print(scene_tuples[scene_tuple_index], (agent1_id, agent2_id), "HV-leftR+HV-leftB -> Crossing")
					HVHV_conflict_v2["cross"]["turnLeftFromLeft&turnLeftFromTop"].append({scene_tuple: conflict})

	# Case #2: HV turns left from top-right arm of T-junction + HV goes straight => crossing
	if HV_turnleft_from_left and HV_straight2left:
		for agent1_id in HV_turnleft_from_left:
			for agent2_id in HV_straight2left:
				is_conflict, conflict = whether_line_conflict(
					traj_a=agent_trajectories[agent1_id], traj_b=agent_trajectories[agent2_id], delta_time=delta_time)
				if is_conflict:
					print(scene_tuples[scene_tuple_index], (agent1_id, agent2_id), "HV-leftR+HV-straight -> Crossing")
					HVHV_conflict_v2["cross"]["goLeft&turnLeftFromLeft"].append({scene_tuple: conflict})

	# Case #3: HV turns left from bottom-center arm of T-junction + HV goes straight => crossing
	if HV_turnleft_from_top and HV_straight2left:
		for agent1_id in HV_turnleft_from_top:
			for agent2_id in HV_straight2left:
				is_conflict, conflict = whether_line_conflict(
					traj_a=agent_trajectories[agent1_id], traj_b=agent_trajectories[agent2_id], delta_time=delta_time)
				if is_conflict:
					print(scene_tuples[scene_tuple_index], (agent1_id, agent2_id), "HV-leftB+HV-straight -> Crossing")
					HVHV_conflict["cross"]["goRight&turnLeftFromTop"].append({scene_tuple: conflict})

	# Case #4: HV turns right from top-left arm of T-junction +
	# HV turns left from top-right arm of T-junction => merging
	if HV_turnright_from_top and HV_straight2left:
		for agent1_id in HV_straight2left:
			for agent2_id in HV_turnright_from_top:
				is_conflict, conflict = whether_buffer_conflict(
					traj_a=agent_trajectories[agent1_id], traj_b=agent_trajectories[agent2_id],
					traj_a_side="right", traj_b_side="left", margin=margin, delta_time=delta_time)
				if is_conflict:
					print(scene_tuples[scene_tuple_index], (agent1_id, agent2_id), "HV-leftR+HV-rightL -> Merging")
					HVHV_conflict["merge"]["goLeft&turnRightFromTop"].append({scene_tuple: conflict})

	# Case #5: HV turns right from bottom-center arm of T-junction + HV goes straight to right => merging
	if HV_turnleft_from_top and HV_straight2right:
		for agent1_id in HV_straight2right:
			for agent2_id in HV_turnleft_from_top:
				is_conflict, conflict = whether_buffer_conflict(
					traj_a=agent_trajectories[agent1_id], traj_b=agent_trajectories[agent2_id],
					traj_a_side="left", traj_b_side="right", margin=margin, delta_time=delta_time)
				if is_conflict:
					print(scene_tuples[scene_tuple_index], (agent1_id, agent2_id), "HV-rightB+HV-straight -> Merging")
					HVHV_conflict["merge"]["goRight&turnLeftFromTop"].append({scene_tuple: conflict})

""" Update the progress bar """
# pbar.update(1)


# %% save as pickle
save_pickle(obj=AVHV_conflict, name=f"AVHV_conflict_{dataset_name}_{int(delta_time//10)}_{margin}_v2")
save_pickle(obj=HVHV_conflict, name=f"HVHV_conflict_{dataset_name}_{int(delta_time//10)}_{margin}_v2")

# %% report the result of identification
report_AVHV_conflicts(collection=AVHV_conflict)
report_HVHV_conflicts(collection=HVHV_conflict)
