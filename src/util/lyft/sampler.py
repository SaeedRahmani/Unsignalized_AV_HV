import numpy as np
from typing import Optional
from l5kit.data import filter_agents_by_labels
from l5kit.data.filter import filter_agents_by_track_id
from l5kit.geometry import compute_agent_pose, rotation33_as_yaw
from l5kit.kinematic import Perturbation
from l5kit.rasterization import EGO_EXTENT_HEIGHT, EGO_EXTENT_LENGTH, EGO_EXTENT_WIDTH
from l5kit.rasterization import Rasterizer, RenderContext
from l5kit.sampling.agent_sampling import (
    get_agent_context, get_relative_poses, compute_agent_velocity)

from .rasterizer import IntersectionRasterizer


def get_agent_centroid(
    agents: np.ndarray,
    filter_agents_threshold,
):
    agents_centroid = {}
    agents = filter_agents_by_labels(agents[0], filter_agents_threshold)
    for agent in agents:
        if agent["label_probabilities"][3]:
            agents_centroid[agent["track_id"]] = agent["centroid"]
    return agents_centroid


def generate_agent_sample(
        state_index: int,
        frames: np.ndarray,
        agents: np.ndarray,
        tl_faces: np.ndarray,
        selected_track_id: Optional[int],
        render_context: RenderContext,
        history_num_frames: int,
        future_num_frames: int,
        step_time: float,
        filter_agents_threshold: float,
        rasterizer: Rasterizer,
        perturbation: Optional[Perturbation] = None,
) -> dict:
    """Generates the inputs and targets to train a deep prediction model. A deep prediction model takes as input
    the state of the world (here: an image we will call the "raster"), and outputs where that agent will be some
    seconds into the future.

    This function has a lot of arguments and is intended for internal use, you should try to use higher level classes
    and partials that use this function.

    Args:
        state_index (int): The anchor frame index, i.e. the "current" timestep in the scene
        frames (np.ndarray): The scene frames array, can be numpy array or a zarr array
        agents (np.ndarray): The full agents array, can be numpy array or a zarr array
        tl_faces (np.ndarray): The full traffic light faces array, can be numpy array or a zarr array
        selected_track_id (Optional[int]): Either None for AV, or the ID of an agent that you want to
        predict the future of. This agent is centered in the raster and the returned targets are derived from
        their future states.
        render_context (RenderContext): The context for rasterisation
        history_num_frames (int): Amount of history frames to draw into the rasters
        future_num_frames (int): Amount of history frames to draw into the rasters
        step_time (float): seconds between consecutive steps
        filter_agents_threshold (float): Value between 0 and 1 to use as cutoff value for agent filtering
        based on their probability of being a relevant agent
        rasterizer: Rasterizer of some sort that draws a map image
        perturbation (Optional[Perturbation]): Object that perturbs the input and targets, used
        to train models that can recover from slight divergence from training set data

    Raises:
        IndexError: An IndexError is returned if the specified ``selected_track_id`` is not present in the scene
        or was filtered by applying the ``filter_agent_threshold`` probability filtering.

    Returns:
        dict: a dict object with the raster array, the future offset coordinates (meters),
        the future yaw angular offset, the future_availability as a binary mask
    """
    (
        history_frames,
        future_frames,
        history_agents,
        future_agents,
        history_tl_faces,
        future_tl_faces,
    ) = get_agent_context(state_index, frames, agents, tl_faces, history_num_frames, future_num_frames, )

    if perturbation is not None and len(future_frames) == future_num_frames:
        history_frames, future_frames = perturbation.perturb(
            history_frames=history_frames, future_frames=future_frames
        )

    # State you want to predict the future of.
    cur_frame = history_frames[0]
    cur_agents = history_agents[0]

    if selected_track_id is None:
        agent_centroid_m = cur_frame["ego_translation"][:2]
        agent_yaw_rad = rotation33_as_yaw(cur_frame["ego_rotation"])
        agent_extent_m = np.asarray((EGO_EXTENT_LENGTH, EGO_EXTENT_WIDTH, EGO_EXTENT_HEIGHT))
        selected_agent = None
    else:
        # this will raise IndexError if the agent is not in the frame or under agent-threshold
        # this is a strict error, we cannot recover from this situation
        filtered_agents = filter_agents_by_labels(cur_agents, filter_agents_threshold)
        agent = filter_agents_by_track_id(filtered_agents, selected_track_id)[0]
        agent_centroid_m = agent["centroid"]
        agent_yaw_rad = float(agent["yaw"])
        agent_extent_m = agent["extent"]
        selected_agent = agent

    # ADDED by XU, see comments below.
    is_intersection_included = None

    if isinstance(rasterizer, IntersectionRasterizer):
        # our extension for the intersection checking:
        input_im, is_intersection_included = rasterizer.rasterize(
            history_frames, history_agents, history_tl_faces, selected_agent)
    else:
        input_im = rasterizer.rasterize(history_frames, history_agents, history_tl_faces, selected_agent)

    agents_centroid = get_agent_centroid(history_agents, filter_agents_threshold)

    world_from_agent = compute_agent_pose(agent_centroid_m, agent_yaw_rad)
    agent_from_world = np.linalg.inv(world_from_agent)
    raster_from_world = render_context.raster_from_world(agent_centroid_m, agent_yaw_rad)

    future_positions_m, future_yaws_rad, future_extents, future_availabilities = get_relative_poses(
        future_num_frames, future_frames, selected_track_id, future_agents, agent_from_world, agent_yaw_rad,
    )
    # history_num_frames + 1 because it also includes the current frame
    history_positions_m, history_yaws_rad, history_extents, history_availabilities = get_relative_poses(
        history_num_frames + 1, history_frames, selected_track_id, history_agents, agent_from_world, agent_yaw_rad,
    )

    history_vels_mps, future_vels_mps = compute_agent_velocity(history_positions_m, future_positions_m, step_time)

    result = {
        "frame_index": state_index,
        "image": input_im,
        "target_positions": future_positions_m,
        "target_yaws": future_yaws_rad,
        "target_velocities": future_vels_mps,
        "target_availabilities": future_availabilities,
        "history_positions": history_positions_m,
        "history_yaws": history_yaws_rad,
        "history_velocities": history_vels_mps,
        "history_availabilities": history_availabilities,
        "world_to_image": raster_from_world,  # TODO deprecate
        "raster_from_agent": raster_from_world @ world_from_agent,
        "raster_from_world": raster_from_world,
        "agent_from_world": agent_from_world,
        "world_from_agent": world_from_agent,
        "centroid": agent_centroid_m,
        "yaw": agent_yaw_rad,
        "extent": agent_extent_m,
        "history_extents": history_extents,
        "future_extents": future_extents,
        "agents_centroid": agents_centroid,
        # ADDED by XU:
        # To know whether this scene includes the expected intersection or not.
        "is_intersection_included": is_intersection_included,
    }
    if len(history_vels_mps) > 0:
        # estimated current speed based on displacement between current frame at T and past frame at T-1
        result["curr_speed"] = np.linalg.norm(history_vels_mps[0])
    return result