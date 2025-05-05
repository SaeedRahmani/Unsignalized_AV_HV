from .trajectory import Trajectory


class Conflict:
	"""
	A class to represent a conflict within the T-junction.
	"""
	def __init__(
		self,
		first_agent_trajectory: Trajectory,
		second_agent_trajectory: Trajectory,
		first_agent_trajectory_id: int,
		second_agent_trajectory_id: int,
		first_agent_conflict_time: float,
		second_agent_conflict_time: float,
		delta_time: float,
	):
		self.first_agent_trajectory = first_agent_trajectory
		self.second_agent_trajectory = second_agent_trajectory
		self.first_agent_trajectory_id = first_agent_trajectory_id
		self.second_agent_trajectory_id = second_agent_trajectory_id
		self.first_agent_conflict_time = first_agent_conflict_time
		self.second_agent_conflict_time = second_agent_conflict_time
		self.delta_time = delta_time
