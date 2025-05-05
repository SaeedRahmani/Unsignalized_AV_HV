import numpy as np
import sys
# np.set_printoptions(suppress=True)
import warnings

warnings.filterwarnings("ignore")

from ordered_set import OrderedSet
from typing import Tuple

from l5kit_conflict.pickle.io import *
from l5kit_conflict.process.preprocess import *
from l5kit_conflict.filter.trajectory_filter import *
from l5kit_conflict.filter.conflict_filter import *

if len(sys.argv) != 2 and sys.argv[1] not in ["train2", "validate", "sample"]:
	print("Usage: python conflict_identification.py [train2 | validate]")
	sys.exit(1)

dataset_name = sys.argv[1]
dataset = init_dataset(dataset_name)
scene_tuples, _ = load_pickle(dataset_name)
scene_tuples.sort()
scene_tuples = extract_intersection_scenarios(scene_tuples)

# normal case:
# AV (speed/acceleration * left/right) = 2 * 2 = 8
AV_turnleft_trajectory_normal = OrderedSet()
AV_turnright_trajectory_normal = OrderedSet()
# HV (speed/acceleration * 6 trajectories) = 2 * 6 = 12
HV_turnLeftFromRight_trajectory_normal = OrderedSet()
HV_goStraightRight_trajectory_normal = OrderedSet()
HV_turnRightFromLeft_trajectory_normal = OrderedSet()
HV_goStraightLeft_trajectory_normal = OrderedSet()
HV_turnLeftFromBottom_trajectory_normal = OrderedSet()
HV_turnRightFromBottom_trajectory_normal = OrderedSet()

# 5 seconds
# AV (speed/acceleration * left/right) = 2 * 2 = 8
AV_turnleft_trajectory_5s = OrderedSet()
AV_turnright_trajectory_5s = OrderedSet()
# HV (speed/acceleration * 6 trajectories) = 2 * 6 = 12
HV_turnLeftFromRight_trajectory_5s = OrderedSet()
HV_goStraightRight_trajectory_5s = OrderedSet()
HV_turnRightFromLeft_trajectory_5s = OrderedSet()
HV_goStraightLeft_trajectory_5s = OrderedSet()
HV_turnLeftFromBottom_trajectory_5s = OrderedSet()
HV_turnRightFromBottom_trajectory_5s = OrderedSet()

# 10 seconds
# AV (speed/acceleration * left/right) = 2 * 2 = 8
AV_turnleft_trajectory_10s = OrderedSet()
AV_turnright_trajectory_10s = OrderedSet()
# HV (speed/acceleration * 6 trajectories) = 2 * 6 = 12
HV_turnLeftFromRight_trajectory_10s = OrderedSet()
HV_goStraightRight_trajectory_10s = OrderedSet()
HV_turnRightFromLeft_trajectory_10s = OrderedSet()
HV_goStraightLeft_trajectory_10s = OrderedSet()
HV_turnLeftFromBottom_trajectory_10s = OrderedSet()
HV_turnRightFromBottom_trajectory_10s = OrderedSet()

for scene_tuple_index, scene_tuple in enumerate(scene_tuples):

	if scene_tuple_index % 100 == 0:
		print(scene_tuple_index)

	# concate the trajectory in two scene
	ego_trajectory, agent_trajectories = get_process_trajectories(
		dataset=dataset,
		dataset_name=dataset_name,
		scene_indices=scene_tuple)

	""" Ego trajectory filter """
	# only focus on the study area
	ego_trajectory_lineString = ego_trajectory.get_lineString()
	ego_trajectory_intersection_lineString = ego_trajectory_lineString.intersection(intersection_area)
	ego_trajectory_intersection_lineString = multiline_to_single_line(ego_trajectory_intersection_lineString)

	# identify ego vehicle's behaviour (2 options: AV_left turning or AV_right turning):
	is_AV_left_turn = is_AV_turnleft_or_right(ego_trajectory_intersection_lineString)

	""" Agent trajectory filter """
	HV_straight2right, HV_straight2left = [], []
	HV_turnright_from_left, HV_turnright_from_bottom = [], []
	HV_turnleft_from_right, HV_turnleft_from_bottom = [], []

	# only focus on the study area
	agent_trajectories = filter_scene_agent_trajectory(agent_trajectories, length_filter_threshold=20)

	# identify agent vehicle's behaviour (6 behaviours)
	agent_trajectories_intersection = {}
	for agent_id, agent_traj in agent_trajectories.items():
		agent_trajectory_intersection_lineString = agent_traj.get_lineString().intersection(intersection_area)
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
			# whether HV turns right from top-left arm of T-junction
			elif is_HV_turnright_from_left(agent_trajectory_intersection_lineString):
				HV_turnright_from_left.append(agent_id)
			# whether HV turns right from bottom-center arm of T-junction
			elif is_HV_turnright_from_bottom(agent_trajectory_intersection_lineString):
				HV_turnright_from_bottom.append(agent_id)
			# whether HV turns left from top-right arm of T-junction
			elif is_HV_turnleft_from_right(agent_trajectory_intersection_lineString):
				HV_turnleft_from_right.append(agent_id)
			# whether HV turns left from bottom-center arm of T-junction
			elif is_HV_turnleft_from_bottom(agent_trajectory_intersection_lineString):
				HV_turnleft_from_bottom.append(agent_id)

	# print(HV_straight2right, HV_straight2left)
	# print(HV_turnright_from_left, HV_turnright_from_bottom)
	# print(HV_turnleft_from_bottom, HV_turnleft_from_right)

	# %%
	""" Identify the AV-HV conflicts """
	if is_AV_left_turn:
		""" There are 3 cases in the AV left turning scenarios:
		1) HV goes straight -> AV-HV crossing
		2) HV turns right   -> AV-HV merging
		3) HV turns left    -> AV-HV crossing """
		if HV_straight2right:
			# Case #1: AV turns left + HV goes straight => crossing
			for agent_id in HV_straight2right:
				is_conflict, conflict = whether_line_conflict(
					traj_a=ego_trajectory, traj_b=agent_trajectories[agent_id], delta_time=50)
				if is_conflict:
					print(scene_tuples[scene_tuple_index], (-1, agent_id), "AV-left+HV-straight -> Crossing")
					AVHV_conflict["cross"]["straight&turnleft"].append({scene_tuple: conflict})
					# save
					AV_turnleft_trajectory_5s.add(ego_trajectory)
					AV_turnleft_trajectory_10s.add(ego_trajectory)
					HV_goStraightRight_trajectory_5s.add(agent_trajectories[agent_id])
					HV_goStraightRight_trajectory_10s.add(agent_trajectories[agent_id])
				else:
					is_conflict, conflict = whether_line_conflict(
						traj_a=ego_trajectory, traj_b=agent_trajectories[agent_id], delta_time=100)
					if is_conflict:
						print(scene_tuples[scene_tuple_index], (-1, agent_id), "AV-left+HV-straight -> Crossing")
						AVHV_conflict["cross"]["straight&turnleft"].append({scene_tuple: conflict})
						# save
						AV_turnleft_trajectory_10s.add(ego_trajectory)
						HV_goStraightRight_trajectory_10s.add(agent_trajectories[agent_id])
					else:
						AV_turnleft_trajectory_normal.add(ego_trajectory)
						HV_goStraightRight_trajectory_normal.add(agent_trajectories[agent_id])

		if HV_turnright_from_left:
			# Case #2: AV turns left + HV turns right => merging
			for agent_id in HV_turnright_from_left:
				is_conflict, conflict = whether_buffer_conflict(
					traj_a=ego_trajectory, traj_b=agent_trajectories[agent_id],
					traj_a_side="right", traj_b_side="left", margin=1, delta_time=50)
				if is_conflict:
					print(scene_tuples[scene_tuple_index], (-1, agent_id), "AV-left+HV-right -> Merging")
					AVHV_conflict["merge"]["turnleft&turnright"].append({scene_tuple: conflict})
					# save
					AV_turnleft_trajectory_5s.add(ego_trajectory)
					AV_turnleft_trajectory_10s.add(ego_trajectory)
					HV_turnRightFromLeft_trajectory_5s.add(agent_trajectories[agent_id])
					HV_turnRightFromLeft_trajectory_10s.add(agent_trajectories[agent_id])
				else:
					is_conflict, conflict = whether_buffer_conflict(
						traj_a=ego_trajectory, traj_b=agent_trajectories[agent_id],
						traj_a_side="right", traj_b_side="left", margin=1, delta_time=100)
					if is_conflict:
						print(scene_tuples[scene_tuple_index], (-1, agent_id), "AV-left+HV-right -> Merging")
						AVHV_conflict["merge"]["turnleft&turnright"].append({scene_tuple: conflict})
						# save
						AV_turnleft_trajectory_10s.add(ego_trajectory)
						HV_turnRightFromLeft_trajectory_10s.add(agent_trajectories[agent_id])
					else:
						AV_turnleft_trajectory_normal.add(ego_trajectory)
						HV_turnRightFromLeft_trajectory_normal.add(agent_trajectories[agent_id])

		if HV_turnleft_from_bottom:
			# Case #3: AV turns left + HV turns left => crossing
			for agent_id in HV_turnleft_from_bottom:
				is_conflict, conflict = whether_line_conflict(
					traj_a=ego_trajectory, traj_b=agent_trajectories[agent_id], delta_time=50
				)
				if is_conflict:
					print(scene_tuples[scene_tuple_index], (-1, agent_id), "AV-left+HV-left -> Crossing")
					AVHV_conflict["cross"]["turnleft&turnleft"].append({scene_tuple: conflict})
					# save
					AV_turnleft_trajectory_5s.add(ego_trajectory)
					AV_turnleft_trajectory_10s.add(ego_trajectory)
					HV_turnLeftFromBottom_trajectory_5s.add(agent_trajectories[agent_id])
					HV_turnLeftFromBottom_trajectory_10s.add(agent_trajectories[agent_id])
				else:
					is_conflict, conflict = whether_line_conflict(
						traj_a=ego_trajectory, traj_b=agent_trajectories[agent_id], delta_time=100
					)
					if is_conflict:
						print(scene_tuples[scene_tuple_index], (-1, agent_id), "AV-left+HV-left -> Crossing")
						AVHV_conflict["cross"]["turnleft&turnleft"].append({scene_tuple: conflict})
						# save
						AV_turnleft_trajectory_10s.add(ego_trajectory)
						HV_turnLeftFromBottom_trajectory_10s.add(agent_trajectories[agent_id])
					else:
						AV_turnleft_trajectory_normal.add(ego_trajectory)
						HV_turnLeftFromBottom_trajectory_normal.add(agent_trajectories[agent_id])

	elif not is_AV_left_turn:
		""" There are 1 case in the AV right turning scenarios:
		Case #4: HV goes straight -> AV-HV merging """
		# Case #4: AV turns right + HV goes straight => merging
		if HV_straight2right:
			for agent_id in HV_straight2right:
				is_conflict, conflict = whether_buffer_conflict(
					traj_a=ego_trajectory, traj_b=agent_trajectories[agent_id],
					traj_a_side="left", traj_b_side="right", margin=1, delta_time=50
				)
				if is_conflict:
					print(scene_tuples[scene_tuple_index], (-1, agent_id), "AV-right+HV-straight -> Merging")
					AVHV_conflict["merge"]["straight&turnright"].append({scene_tuple: conflict})
					# save
					AV_turnright_trajectory_5s.add(ego_trajectory)
					AV_turnright_trajectory_10s.add(ego_trajectory)
					HV_goStraightRight_trajectory_5s.add(agent_trajectories[agent_id])
					HV_goStraightRight_trajectory_10s.add(agent_trajectories[agent_id])
				else:
					is_conflict, conflict = whether_buffer_conflict(
						traj_a=ego_trajectory, traj_b=agent_trajectories[agent_id],
						traj_a_side="left", traj_b_side="right", margin=1, delta_time=100
					)
					if is_conflict:
						print(scene_tuples[scene_tuple_index], (-1, agent_id), "AV-right+HV-straight -> Merging")
						AVHV_conflict["merge"]["straight&turnright"].append({scene_tuple: conflict})
						# save
						AV_turnright_trajectory_10s.add(ego_trajectory)
						HV_goStraightRight_trajectory_10s.add(agent_trajectories[agent_id])
					else:
						AV_turnright_trajectory_normal.add(ego_trajectory)
						HV_goStraightRight_trajectory_normal.add(agent_trajectories[agent_id])

	# %%
	""" Identify the HV-HV conflicts 
	There are 6 cases of HV-HV conflicts """

	# Case #1: HV turns left from top-right arm of T-junction +
	# HV turns left from bottom-center arm of T-junction => crossing
	if HV_turnleft_from_right and HV_turnleft_from_bottom:
		for agent1_id in HV_turnleft_from_right:
			for agent2_id in HV_turnleft_from_bottom:
				is_conflict, conflict = whether_line_conflict(
					traj_a=agent_trajectories[agent1_id], traj_b=agent_trajectories[agent2_id], delta_time=50)
				if is_conflict:
					print(scene_tuples[scene_tuple_index], (agent1_id, agent2_id), "HV-leftR+HV-leftB -> Crossing")
					HVHV_conflict["cross"]["turnleft&turnleft"].append({scene_tuple: conflict})
					# save
					HV_turnLeftFromRight_trajectory_5s.add(agent_trajectories[agent1_id])
					HV_turnLeftFromRight_trajectory_10s.add(agent_trajectories[agent1_id])
					HV_turnLeftFromBottom_trajectory_5s.add(agent_trajectories[agent2_id])
					HV_turnLeftFromBottom_trajectory_10s.add(agent_trajectories[agent2_id])
				else:
					is_conflict, conflict = whether_line_conflict(
						traj_a=agent_trajectories[agent1_id], traj_b=agent_trajectories[agent2_id], delta_time=100)
					if is_conflict:
						print(scene_tuples[scene_tuple_index], (agent1_id, agent2_id), "HV-leftR+HV-leftB -> Crossing")
						HVHV_conflict["cross"]["turnleft&turnleft"].append({scene_tuple: conflict})
						# save
						HV_turnLeftFromRight_trajectory_10s.add(agent_trajectories[agent1_id])
						HV_turnLeftFromBottom_trajectory_10s.add(agent_trajectories[agent2_id])
					else:
						HV_turnLeftFromRight_trajectory_normal.add(agent_trajectories[agent1_id])
						HV_turnLeftFromBottom_trajectory_normal.add(agent_trajectories[agent2_id])

	# Case #2: HV turns left from top-right arm of T-junction + HV goes straight => crossing
	if HV_turnleft_from_right and HV_straight2right:
		for agent1_id in HV_turnleft_from_right:
			for agent2_id in HV_straight2right:
				is_conflict, conflict = whether_line_conflict(
					traj_a=agent_trajectories[agent1_id], traj_b=agent_trajectories[agent2_id], delta_time=50)
				if is_conflict:
					print(scene_tuples[scene_tuple_index], (agent1_id, agent2_id), "HV-leftR+HV-straight -> Crossing")
					HVHV_conflict["cross"]["straight&turnleftRight"].append({scene_tuple: conflict})
					# save
					HV_turnLeftFromRight_trajectory_5s.add(agent_trajectories[agent1_id])
					HV_turnLeftFromRight_trajectory_10s.add(agent_trajectories[agent1_id])
					HV_goStraightRight_trajectory_5s.add(agent_trajectories[agent2_id])
					HV_goStraightRight_trajectory_10s.add(agent_trajectories[agent2_id])
				else:
					is_conflict, conflict = whether_line_conflict(
						traj_a=agent_trajectories[agent1_id], traj_b=agent_trajectories[agent2_id], delta_time=100)
					if is_conflict:
						print(scene_tuples[scene_tuple_index], (agent1_id, agent2_id), "HV-leftR+HV-straight -> Crossing")
						HVHV_conflict["cross"]["straight&turnleftRight"].append({scene_tuple: conflict})
						# save
						HV_turnLeftFromRight_trajectory_10s.add(agent_trajectories[agent1_id])
						HV_goStraightRight_trajectory_10s.add(agent_trajectories[agent2_id])
					else:
						HV_turnLeftFromRight_trajectory_normal.add(agent_trajectories[agent1_id])
						HV_goStraightRight_trajectory_normal.add(agent_trajectories[agent2_id])

	# Case #3: HV turns left from bottom-center arm of T-junction + HV goes straight => crossing
	if HV_turnleft_from_bottom and HV_straight2right:
		for agent1_id in HV_turnleft_from_bottom:
			for agent2_id in HV_straight2right:
				is_conflict, conflict = whether_line_conflict(
					traj_a=agent_trajectories[agent1_id], traj_b=agent_trajectories[agent2_id], delta_time=50)
				if is_conflict:
					print(scene_tuples[scene_tuple_index], (agent1_id, agent2_id), "HV-leftB+HV-straight -> Crossing")
					HVHV_conflict["cross"]["straight&turnleftBottom"].append({scene_tuple: conflict})
					# save
					HV_turnLeftFromBottom_trajectory_5s.add(agent_trajectories[agent1_id])
					HV_turnLeftFromBottom_trajectory_10s.add(agent_trajectories[agent1_id])
					HV_goStraightRight_trajectory_5s.add(agent_trajectories[agent2_id])
					HV_goStraightRight_trajectory_10s.add(agent_trajectories[agent2_id])
				else:
					is_conflict, conflict = whether_line_conflict(
						traj_a=agent_trajectories[agent1_id], traj_b=agent_trajectories[agent2_id], delta_time=100)
					if is_conflict:
						print(scene_tuples[scene_tuple_index], (agent1_id, agent2_id), "HV-leftB+HV-straight -> Crossing")
						HVHV_conflict["cross"]["straight&turnleftBottom"].append({scene_tuple: conflict})
						# save
						HV_turnLeftFromBottom_trajectory_10s.add(agent_trajectories[agent1_id])
						HV_goStraightRight_trajectory_10s.add(agent_trajectories[agent2_id])
					else:
						HV_turnLeftFromBottom_trajectory_normal.add(agent_trajectories[agent1_id])
						HV_goStraightRight_trajectory_normal.add(agent_trajectories[agent2_id])

	# Case #4: HV turns right from top-left arm of T-junction +
	# HV turns left from top-right arm of T-junction => merging
	if HV_turnleft_from_right and HV_turnright_from_left:
		for agent1_id in HV_turnleft_from_right:
			for agent2_id in HV_turnright_from_left:
				is_conflict, conflict = whether_buffer_conflict(
					traj_a=agent_trajectories[agent1_id], traj_b=agent_trajectories[agent2_id],
					traj_a_side="right", traj_b_side="left", margin=1, delta_time=50)
				if is_conflict:
					print(scene_tuples[scene_tuple_index], (agent1_id, agent2_id), "HV-leftR+HV-rightL -> Merging")
					HVHV_conflict["merge"]["turnleft&turnright"].append({scene_tuple: conflict})
					# save
					HV_turnLeftFromRight_trajectory_5s.add(agent_trajectories[agent1_id])
					HV_turnLeftFromRight_trajectory_10s.add(agent_trajectories[agent1_id])
					HV_turnRightFromLeft_trajectory_5s.add(agent_trajectories[agent2_id])
					HV_turnRightFromLeft_trajectory_10s.add(agent_trajectories[agent2_id])
				else:
					is_conflict, conflict = whether_buffer_conflict(
						traj_a=agent_trajectories[agent1_id], traj_b=agent_trajectories[agent2_id],
						traj_a_side="right", traj_b_side="left", margin=1, delta_time=100)
					if is_conflict:
						print(scene_tuples[scene_tuple_index], (agent1_id, agent2_id), "HV-leftR+HV-rightL -> Merging")
						HVHV_conflict["merge"]["turnleft&turnright"].append({scene_tuple: conflict})
						# save
						HV_turnLeftFromRight_trajectory_10s.add(agent_trajectories[agent1_id])
						HV_turnRightFromLeft_trajectory_10s.add(agent_trajectories[agent2_id])
					else:
						HV_turnLeftFromRight_trajectory_normal.add(agent_trajectories[agent1_id])
						HV_turnRightFromLeft_trajectory_normal.add(agent_trajectories[agent2_id])

	# Case #5: HV turns right from bottom-center arm of T-junction + HV goes straight to right => merging
	if HV_turnright_from_bottom and HV_straight2right:
		for agent1_id in HV_turnright_from_bottom:
			for agent2_id in HV_straight2right:
				is_conflict, conflict = whether_buffer_conflict(
					traj_a=agent_trajectories[agent1_id], traj_b=agent_trajectories[agent2_id],
					traj_a_side="left", traj_b_side="right", margin=1, delta_time=50)
				if is_conflict:
					print(scene_tuples[scene_tuple_index], (agent1_id, agent2_id), "HV-rightB+HV-straight -> Merging")
					HVHV_conflict["merge"]["straight&turnright"].append({scene_tuple: conflict})
					# save
					HV_turnRightFromBottom_trajectory_5s.add(agent_trajectories[agent1_id])
					HV_turnRightFromBottom_trajectory_10s.add(agent_trajectories[agent1_id])
					HV_goStraightRight_trajectory_5s.add(agent_trajectories[agent2_id])
					HV_goStraightRight_trajectory_10s.add(agent_trajectories[agent2_id])
				else:
					is_conflict, conflict = whether_buffer_conflict(
						traj_a=agent_trajectories[agent1_id], traj_b=agent_trajectories[agent2_id],
						traj_a_side="left", traj_b_side="right", margin=1, delta_time=100)
					if is_conflict:
						print(scene_tuples[scene_tuple_index], (agent1_id, agent2_id), "HV-rightB+HV-straight -> Merging")
						HVHV_conflict["merge"]["straight&turnright"].append({scene_tuple: conflict})
						# save
						HV_turnRightFromBottom_trajectory_10s.add(agent_trajectories[agent1_id])
						HV_goStraightRight_trajectory_10s.add(agent_trajectories[agent2_id])
					else:
						HV_turnRightFromBottom_trajectory_normal.add(agent_trajectories[agent1_id])
						HV_goStraightRight_trajectory_normal.add(agent_trajectories[agent2_id])

	# Case #6: HV turns left from bottom-center arm of T-junction + HV goes straight to left => merging
	if HV_turnleft_from_bottom and HV_straight2left:
		for agent1_id in HV_turnleft_from_bottom:
			for agent2_id in HV_straight2left:
				is_conflict, conflict = whether_buffer_conflict(
					traj_a=agent_trajectories[agent1_id], traj_b=agent_trajectories[agent2_id],
					traj_a_side="right", traj_b_side="left", margin=1, delta_time=50)
				if is_conflict:
					print(scene_tuples[scene_tuple_index], (agent1_id, agent2_id), "HV-leftB+HV-straight -> Merging")
					HVHV_conflict["merge"]["straight&turnleft"].append({scene_tuple: conflict})
					# save
					HV_turnLeftFromBottom_trajectory_5s.add(agent_trajectories[agent1_id])
					HV_turnLeftFromBottom_trajectory_10s.add(agent_trajectories[agent1_id])
					HV_goStraightLeft_trajectory_5s.add(agent_trajectories[agent2_id])
					HV_goStraightLeft_trajectory_10s.add(agent_trajectories[agent2_id])
				else:
					is_conflict, conflict = whether_buffer_conflict(
						traj_a=agent_trajectories[agent1_id], traj_b=agent_trajectories[agent2_id],
						traj_a_side="right", traj_b_side="left", margin=1, delta_time=10)
					if is_conflict:
						print(scene_tuples[scene_tuple_index], (agent1_id, agent2_id), "HV-leftB+HV-straight -> Merging")
						HVHV_conflict["merge"]["straight&turnleft"].append({scene_tuple: conflict})
						# save
						HV_turnLeftFromBottom_trajectory_10s.add(agent_trajectories[agent1_id])
						HV_goStraightLeft_trajectory_10s.add(agent_trajectories[agent2_id])
					else:
						HV_turnLeftFromBottom_trajectory_normal.add(agent_trajectories[agent1_id])
						HV_goStraightLeft_trajectory_normal.add(agent_trajectories[agent2_id])

# normal case:
save_pickle(AV_turnleft_trajectory_normal, name="AV_turnleft_trajectory_normal"+f"_{dataset_name}")
save_pickle(AV_turnright_trajectory_normal, name="AV_turnright_trajectory_normal"+f"_{dataset_name}")
save_pickle(HV_turnLeftFromRight_trajectory_normal, name="HV_turnLeftFromRight_trajectory_normal"+f"_{dataset_name}")
save_pickle(HV_goStraightRight_trajectory_normal, name="HV_goStraightRight_trajectory_normal"+f"_{dataset_name}")
save_pickle(HV_turnRightFromLeft_trajectory_normal, name="HV_turnRightFromLeft_trajectory_normal"+f"_{dataset_name}")
save_pickle(HV_goStraightLeft_trajectory_normal, name="HV_goStraightLeft_trajectory_normal"+f"_{dataset_name}")
save_pickle(HV_turnLeftFromBottom_trajectory_normal, name="HV_turnLeftFromBottom_trajectory_normal"+f"_{dataset_name}")
save_pickle(HV_turnRightFromBottom_trajectory_normal, name="HV_turnRightFromBottom_trajectory_normal"+f"_{dataset_name}")

# 5 seconds
save_pickle(AV_turnleft_trajectory_5s, name="AV_turnleft_trajectory_5s"+f"_{dataset_name}")
save_pickle(AV_turnright_trajectory_5s, name="AV_turnright_trajectory_5s"+f"_{dataset_name}")
save_pickle(HV_turnLeftFromRight_trajectory_5s, name="HV_turnLeftFromRight_trajectory_5s"+f"_{dataset_name}")
save_pickle(HV_goStraightRight_trajectory_5s, name="HV_goStraightRight_trajectory_5s"+f"_{dataset_name}")
save_pickle(HV_turnRightFromLeft_trajectory_5s, name="HV_turnRightFromLeft_trajectory_5s"+f"_{dataset_name}")
save_pickle(HV_goStraightLeft_trajectory_5s, name="HV_goStraightLeft_trajectory_5s"+f"_{dataset_name}")
save_pickle(HV_turnLeftFromBottom_trajectory_5s, name="HV_turnLeftFromBottom_trajectory_5s"+f"_{dataset_name}")
save_pickle(HV_turnRightFromBottom_trajectory_5s, name="HV_turnRightFromBottom_trajectory_5s"+f"_{dataset_name}")

# 10 seconds
save_pickle(AV_turnleft_trajectory_10s, name="AV_turnleft_trajectory_10s"+f"_{dataset_name}")
save_pickle(AV_turnright_trajectory_10s, name="AV_turnright_trajectory_10s"+f"_{dataset_name}")
save_pickle(HV_turnLeftFromRight_trajectory_10s, name="HV_turnLeftFromRight_trajectory_10s"+f"_{dataset_name}")
save_pickle(HV_goStraightRight_trajectory_10s, name="HV_goStraightRight_trajectory_10s"+f"_{dataset_name}")
save_pickle(HV_turnRightFromLeft_trajectory_10s, name="HV_turnRightFromLeft_trajectory_10s"+f"_{dataset_name}")
save_pickle(HV_goStraightLeft_trajectory_10s, name="HV_goStraightLeft_trajectory_10s"+f"_{dataset_name}")
save_pickle(HV_turnLeftFromBottom_trajectory_10s, name="HV_turnLeftFromBottom_trajectory_10s"+f"_{dataset_name}")
save_pickle(HV_turnRightFromBottom_trajectory_10s, name="HV_turnRightFromBottom_trajectory_10s"+f"_{dataset_name}")