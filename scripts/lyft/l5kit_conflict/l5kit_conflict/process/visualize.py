import numpy as np
import matplotlib.pyplot as plt
from typing import Union
from ..objects.trajectory import Trajectory


X_FIGURE_SIZE = 5
Y_FIGURE_SIZE = 5

X_LEFT_BORDER = -850
X_RIGHT_BORDER = -550
Y_TOP_BORDER = -850
Y_BOTTOM_BORDER = -1150


def plot_scene_ego_trajectory(scene_ego_trajectory: Trajectory):
	ego_coordinate_x = scene_ego_trajectory.trajectory_xy[:, 0]
	ego_coordinate_y = scene_ego_trajectory.trajectory_xy[:, 1]
	# plt.xlim([X_LEFT_BORDER, X_RIGHT_BORDER])
	# plt.ylim([Y_BOTTOM_BORDER, Y_TOP_BORDER])
	plt.plot(ego_coordinate_x, ego_coordinate_y)


def plot_scene_agent_trajectories(scene_agent_trajectories: Trajectory):
	# plt.xlim([X_LEFT_BORDER, X_RIGHT_BORDER])
	# plt.ylim([Y_BOTTOM_BORDER, Y_TOP_BORDER])

	for agent_id, agent_traj in scene_agent_trajectories.items():
		agent_coordinate_x = agent_traj.trajectory_xy[:, 0]
		agent_coordinate_y = agent_traj.trajectory_xy[:, 1]

		plt.plot(agent_coordinate_x, agent_coordinate_y)


def plot_scene_trajectories(
		scene_ego_trajectory: Union[Trajectory, None] = None,
		scene_agent_trajectories: Union[Trajectory, None] = None,
):
	# plt.figure(figsize=(X_FIGURE_SIZE, Y_FIGURE_SIZE))
	if scene_ego_trajectory is not None:
		plot_scene_ego_trajectory(scene_ego_trajectory)
	if scene_agent_trajectories is not None:
		plot_scene_agent_trajectories(scene_agent_trajectories)