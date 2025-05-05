import numpy as np
from shapely.geometry import LineString, MultiLineString, Polygon, Point
from typing import Union, Tuple, List


# def check_order(line_segment_collection):
# 	num_line_segments = len(line_segment_collection)
# 	# enumerate all the possible pair of two line segments
# 	pairs_to_check = []
# 	for i in range(num_line_segments):
# 		for j in range(num_line_segments):
# 			if i != j:
# 				pairs_to_check.append((i, j))
# 	assert len(pairs_to_check) == num_line_segments * (num_line_segments - 1)
# 	# check the valid pairs
# 	threshold = 0.2
# 	while True:
# 		pairs_in_order = []
#
# 		for (line1, line2) in pairs_to_check:
# 			endpoint_line1 = np.array(line_segment_collection[line1].xy)[:, -1]
# 			startpoint_line2 = np.array(line_segment_collection[line2].xy)[:, 0]
# 			if np.sqrt(np.sum(np.square(endpoint_line1 - startpoint_line2))) < threshold:
# 				pairs_in_order.append((line1, line2))
# 		if len(pairs_in_order) > num_line_segments - 1:
# 			threshold -= 0.01
# 		elif len(pairs_in_order) == num_line_segments - 1:
# 			break
# 		else:
# 			raise ArithmeticError(f"{len(pairs_in_order)}, {num_line_segments - 1}")
# 	pairs_in_order.sort(reverse=True)
# 	return pairs_in_order
#
#
# def multi2singleLineString(line: Union[LineString, MultiLineString]):
# 	if isinstance(line, LineString):
# 		return line
# 	elif isinstance(line, MultiLineString):
# 		# concatenate the line segments into a new lineString
# 		line_segment_collection = []
# 		for line_segment in line:
# 			line_segment_collection.append(line_segment)
# 		pairs_in_order = check_order(line_segment_collection)
# 		sequence = [pairs_in_order[0][0], ]
# 		for idx, pair in enumerate(pairs_in_order):
# 			sequence.append(pair[1])
# 		assert len(sequence) == len(line_segment_collection), f"{len(sequence), len(line_segment_collection)}"
#
# 		points = []
# 		for line_index in sequence:
# 			line_segment = line_segment_collection[line_index]
# 			for point in np.array(line_segment.coords.xy).T:
# 				points.append(Point(point))
#
# 		return LineString(points)
# 	else:
# 		raise TypeError("Check the type of argument: line")


def multiline_to_single_line(geometry: Union[LineString, MultiLineString]) -> LineString:
	if isinstance(geometry, LineString):
		return geometry
	coords = list(map(lambda part: list(part.coords), geometry.geoms))
	flat_coords = [Point(*point) for segment in coords for point in segment]

	# TODO：
	return LineString(flat_coords)


class Trajectory:
	"""
	A class to represent a trajectory within the T-junction.
	"""
	# LEFT_TOP = (-850, -875)
	# LEFT_BOTTOM = (-850, -925)
	# RIGHT_TOP = (-700, -875)
	# RIGHT_BOTTOM = (-700, -925)
	LEFT_TOP 	 = (-200, -865)
	LEFT_BOTTOM  = (-200, -915)
	RIGHT_TOP 	 = ( -50, -865)
	RIGHT_BOTTOM = ( -50, -915)
 
	intersection_coords = (LEFT_TOP, LEFT_BOTTOM, RIGHT_BOTTOM, RIGHT_TOP, LEFT_TOP)
	intersection_area = Polygon(intersection_coords)

	def __init__(
			self,
			scene_indices: Union[int, tuple],
			agent_index: Union[int, None],
			trajectory_xy: np.ndarray,
			trajectory_t: np.ndarray,
			dataset: str,
	):
		assert trajectory_t.shape[0] == trajectory_xy.shape[0]

		self.scene_indices = scene_indices
		# self.is_trajectory_concatenated = False if isinstance(self.scene_indices, int) else True
		self.agent_index = agent_index
		assert dataset in ["sample", "train2", "validate"]
		self.dataset = dataset
		self.trajectory_index = (dataset, scene_indices, agent_index)

		# set trajectory as a sequence of (x, y, t)
		self.trajectory_xy = trajectory_xy
		self.trajectory_t = trajectory_t

		self.lineString = LineString(coordinates=self.trajectory_xy)

	def __hash__(self) -> int:
		return hash(self.trajectory_index)

	def __eq__(self, other) -> bool:
		return self.trajectory_index == other.trajectory_index

	def get_lineString(self, ) -> LineString:
		return LineString(coordinates=self.trajectory_xy)

	def __derive_intersection_lineString(self, ) -> Union[LineString, None]:
		""" Derive the lineString object within the intersection (study area) """
		self.lineString = LineString(coordinates=self.trajectory_xy)
		if self.lineString.intersects(Trajectory.intersection_area):
			intersection_lineString = self.lineString.intersection(Trajectory.intersection_area)
			if isinstance(intersection_lineString, MultiLineString):
				intersection_lineString = multiline_to_single_line(intersection_lineString)
			return intersection_lineString
		else:
			return None

	@property
	def average_speed_intersection(self, ) -> float:
		# focus on the intersection area
		intersection_lineString = self.__derive_intersection_lineString()
		# get trajectory inside the intersection
		intersection_trajectory_xy = np.array(intersection_lineString.coords)
		# derive the average speed (2 layer of average)
		delta_distances = np.sqrt(np.sum(np.diff(intersection_trajectory_xy, axis=0) ** 2, axis=1))
		assert delta_distances.shape[0] == intersection_trajectory_xy.shape[0] - 1, \
			f"{intersection_trajectory_xy.shape[0], delta_distances.shape[0]}"
		speeds = delta_distances / 0.1

		average_speed = np.mean(speeds)
		return average_speed

	@property
	def average_acceleration_intersection(self, ) -> float:
		# focus on the intersection area
		intersection_lineString = self.__derive_intersection_lineString()
		# get trajectory inside the intersection
		intersection_trajectory_xy = np.array(intersection_lineString.coords)
		# derive the average acceleration (2 layer of average)
		delta_distances = np.sqrt(np.sum(np.diff(intersection_trajectory_xy, axis=0) ** 2, axis=1))
		assert delta_distances.shape[0] == intersection_trajectory_xy.shape[0] - 1
		delta_speeds = np.diff(delta_distances / 0.1)
		assert delta_speeds.shape[0] == intersection_trajectory_xy.shape[0] - 2
		accelerations = delta_speeds / 0.1

		average_acceleration = np.mean(accelerations)
		return average_acceleration
