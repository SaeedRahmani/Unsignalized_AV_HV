import numpy as np
from shapely.geometry import MultiPoint
from typing import Tuple, Union
from ..objects.trajectory import Trajectory
from ..objects.conflict import Conflict
from .helper import *


def whether_buffer_conflict(
		traj_a: Trajectory,
		traj_b: Trajectory,
		traj_a_side: str,
		traj_b_side: str,
		margin: float = 1,
		delta_time: float = 50
) -> Tuple[bool, Union[Conflict, None]]:
	"""Check whether there is a merging conflict or not."""
	assert isinstance(traj_a, Trajectory)
	assert isinstance(traj_b, Trajectory)

	traj_a_lineString = traj_a.get_lineString()
	traj_b_lineString = traj_b.get_lineString()
	traj_a_lineStringBuffer = traj_a_lineString.buffer(margin)
	traj_b_lineStringBuffer = traj_b_lineString.buffer(margin)
	traj_a_offsetCurve = traj_a_lineString.parallel_offset(distance=margin, side=traj_a_side)
	traj_b_offsetCurve = traj_b_lineString.parallel_offset(distance=margin, side=traj_b_side)

	if traj_a_lineStringBuffer.intersects(traj_b_lineStringBuffer):
		intersection_points = traj_a_offsetCurve.intersection(traj_b_offsetCurve)
		if isinstance(intersection_points, Point):
			# single intersection point
			pass
		elif isinstance(intersection_points, MultiPoint):
			# multiple intersection points
			intersection_points = intersection_points[0]
		else:
			return False, None

		point_in_traj = nearest_point_in_trajectory(
			trajectory=traj_a,
			point=np.array(intersection_points.xy))
		time_a, _ = nearest_time_index(
			intersect_point_xy=point_in_traj,
			traj_a=traj_a,
			traj_b=None,
		)
		point_in_traj = nearest_point_in_trajectory(
			trajectory=traj_b,
			point=np.array(intersection_points.xy))
		time_b, _ = nearest_time_index(
			intersect_point_xy=point_in_traj,
			traj_a=traj_b,
			traj_b=None,
		)

		# print(time_a, time_b)

		if abs(time_b - time_a) < delta_time:
			# create a conflict object:
			if time_a < time_b:
				first_agent_trajectory = traj_a
				second_agent_trajectory = traj_b
				first_agent = traj_a.agent_index
				second_agent = traj_b.agent_index
				first_agent_conflict_time = time_a
				second_agent_conflict_time = time_b
			else:
				first_agent_trajectory = traj_b
				second_agent_trajectory = traj_a
				first_agent = traj_b.agent_index
				second_agent = traj_a.agent_index
				first_agent_conflict_time = time_b
				second_agent_conflict_time = time_a

			conflict = Conflict(
				first_agent_trajectory=first_agent_trajectory,
				second_agent_trajectory=second_agent_trajectory,
				first_agent_trajectory_id=first_agent,
				second_agent_trajectory_id=second_agent,
				first_agent_conflict_time=first_agent_conflict_time,
				second_agent_conflict_time=second_agent_conflict_time,
				delta_time=second_agent_conflict_time - first_agent_conflict_time
			)
			return True, conflict
		else:
			return False, None
	else:
		return False, None


def whether_line_conflict(
		traj_a: Trajectory,
		traj_b: Trajectory,
		delta_time: float = 50
) -> Tuple[bool, Union[Conflict, None]]:
	"""Check whether there is a crossing conflict or not."""
	assert isinstance(traj_a, Trajectory)
	assert isinstance(traj_b, Trajectory)

	traj_a_lineString = traj_a.get_lineString()
	traj_b_lineString = traj_b.get_lineString()

	if traj_a_lineString.intersects(traj_b_lineString):
		intersection_points = traj_a_lineString.intersection(traj_b_lineString)
		if isinstance(intersection_points, Point):
			# single intersection point
			pass
		elif isinstance(intersection_points, MultiPoint):
			# multiple intersection points
			intersection_points = intersection_points[0]
		time_a, time_b = nearest_time_index(
			intersect_point_xy=np.array(intersection_points.xy),
			traj_a=traj_a,
			traj_b=traj_b,
		)

		if abs(time_b - time_a) < delta_time:
			# create a conflict object:
			if time_a < time_b:
				first_agent_trajectory = traj_a
				second_agent_trajectory = traj_b
				first_agent = traj_a.agent_index
				second_agent = traj_b.agent_index
				first_agent_conflict_time = time_a
				second_agent_conflict_time = time_b
			else:
				first_agent_trajectory = traj_b
				second_agent_trajectory = traj_a
				first_agent = traj_b.agent_index
				second_agent = traj_a.agent_index
				first_agent_conflict_time = time_b
				second_agent_conflict_time = time_a

			conflict = Conflict(
				first_agent_trajectory=first_agent_trajectory,
				second_agent_trajectory=second_agent_trajectory,
				first_agent_trajectory_id=first_agent,
				second_agent_trajectory_id=second_agent,
				first_agent_conflict_time=first_agent_conflict_time,
				second_agent_conflict_time=second_agent_conflict_time,
				delta_time=second_agent_conflict_time - first_agent_conflict_time
			)
			return True, conflict
		else:
			return False, None
	else:
		return False, None
