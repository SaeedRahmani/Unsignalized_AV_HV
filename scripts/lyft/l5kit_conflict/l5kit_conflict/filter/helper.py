import numpy as np
from typing import Union
from shapely.geometry import Point, LineString, MultiLineString
from ..objects.trajectory import Trajectory


def multiline_to_single_line(geometry: Union[LineString, MultiLineString]) -> LineString:
	if isinstance(geometry, LineString):
		return geometry
	coords = list(map(lambda part: list(part.coords), geometry.geoms))
	flat_coords = [Point(*point) for segment in coords for point in segment]

	# TODO：
	return LineString(flat_coords)


def check_order(line_segment_collection):
	num_line_segments = len(line_segment_collection)
	# enumerate all the possible pair of two line segments
	pairs_to_check = []
	for i in range(num_line_segments):
		for j in range(num_line_segments):
			if i != j:
				pairs_to_check.append((i, j))
	assert len(pairs_to_check) == num_line_segments * (num_line_segments - 1)
	# check the valid pairs
	threshold = 0.8
	while True:
		pairs_in_order = []

		for (line1, line2) in pairs_to_check:
			endpoint_line1 = np.array(line_segment_collection[line1].xy)[:, -1]
			startpoint_line2 = np.array(line_segment_collection[line2].xy)[:, 0]
			if np.sqrt(np.sum(np.square(endpoint_line1 - startpoint_line2))) < threshold:
				pairs_in_order.append((line1, line2))
		if len(pairs_in_order) > num_line_segments - 1:
			threshold -= 0.01
		elif len(pairs_in_order) == num_line_segments - 1:
			break
		else:
			raise ArithmeticError(f"{len(pairs_in_order)}, {num_line_segments - 1}")
	pairs_in_order.sort(reverse=True)
	return pairs_in_order


def multi2singleLineString(line: Union[LineString, MultiLineString]):
	if isinstance(line, LineString):
		return line
	elif isinstance(line, MultiLineString):
		# concatenate the line segments into a new lineString
		line_segment_collection = []
		for line_segment in line:
			line_segment_collection.append(line_segment)
		pairs_in_order = check_order(line_segment_collection)
		sequence = [pairs_in_order[0][0], ]
		for idx, pair in enumerate(pairs_in_order):
			sequence.append(pair[1])
		assert len(sequence) == len(line_segment_collection), f"{len(sequence), len(line_segment_collection)}"

		points = []
		for line_index in sequence:
			line_segment = line_segment_collection[line_index]
			for point in np.array(line_segment.coords.xy).T:
				points.append(Point(point))

		return LineString(points)
	else:
		raise TypeError("Check the type of argument: line")


def nearest_time_index(intersect_point_xy, traj_a: Trajectory, traj_b: Union[None, Trajectory]=None):
	"""Return the nearest time index to the conflict point."""
	intersect_point_xy = intersect_point_xy.reshape((1, 2))
	assert intersect_point_xy.shape == (1, 2)

	# ego trajectory if it is a ndarray, otherwise it is agent trajectory if it is a dict
	if isinstance(traj_a, Trajectory):
		diff = traj_a.trajectory_xy - intersect_point_xy
		diff = np.sqrt(np.sum(np.square(diff), axis=1))
		assert diff.shape[0] == traj_a.trajectory_xy.shape[0]
		time_a = traj_a.trajectory_t[np.argmin(diff)]
	else:
		raise TypeError("Invalid type of argument: `traj_a` in function `nearest_time_index`")

	# agent b
	if isinstance(traj_b, Trajectory):
		diff = traj_b.trajectory_xy - intersect_point_xy
		diff = np.sqrt(np.sum(np.square(diff), axis=1))
		assert diff.shape[0] == traj_b.trajectory_xy.shape[0]
		time_b = traj_b.trajectory_t[np.argmin(diff)]
	elif traj_b is None:
		time_b = None
	else:
		raise TypeError("Invalid type of argument: `traj_b` in function `nearest_time_index`")

	return time_a, time_b


def nearest_point_in_trajectory(trajectory: Trajectory, point: np.ndarray) -> np.ndarray:
	assert isinstance(point, np.ndarray)
	assert isinstance(trajectory, Trajectory)
	point = point.T
	assert point.shape == (1, 2), f"{point.shape}"

	diff = trajectory.trajectory_xy - point
	diff = np.sqrt(np.sum(np.square(diff), axis=1))
	point_index = np.argmin(diff)
	return trajectory.trajectory_xy[point_index]
