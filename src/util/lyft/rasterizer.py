import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
from .mapAPI import IntersectionMapAPI
from l5kit.data import DataManager
from l5kit.configs.config import load_metadata
from l5kit.rasterization import Rasterizer, RenderContext
from l5kit.geometry import transform_point, rotation33_as_yaw
from l5kit.rasterization.semantic_rasterizer import indices_in_bounds


def build_intersection_rasterizer(
    cfg: dict,
    data_manager: DataManager,
    intersection_id: str = "WTgZ",
) -> Rasterizer:
    raster_cfg = cfg["raster_params"]
    dataset_meta_key = raster_cfg["dataset_meta_key"]

    render_context = RenderContext(
        raster_size_px=np.array(raster_cfg["raster_size"]),
        pixel_size_m=np.array(raster_cfg["pixel_size"]),
        center_in_raster_ratio=np.array(raster_cfg["ego_center"]),
        set_origin_to_bottom=raster_cfg["set_origin_to_bottom"],
    )

    filter_agents_threshold = raster_cfg["filter_agents_threshold"]
    history_num_frames = cfg["model_params"]["history_num_frames"]
    render_ego_history = cfg["model_params"]["render_ego_history"]

    semantic_map_filepath = data_manager.require(raster_cfg["semantic_map_key"])
    dataset_meta = load_metadata(data_manager.require(dataset_meta_key))
    world_to_ecef = np.array(dataset_meta["world_to_ecef"], dtype=np.float64)

    return IntersectionRasterizer(  # ADDED by XU: return this new rasterizer class.
        render_context=render_context,
        semantic_map_path=semantic_map_filepath,
        world_to_ecef=world_to_ecef,
        intersection_id=intersection_id,
    )


class IntersectionRasterizer(Rasterizer):
    INTERPOLATION_POINTS = 20
    def __init__(
        self,
        render_context: RenderContext,
        semantic_map_path: str,
        world_to_ecef: np.ndarray,
        intersection_id: str,
    ):
        self.render_context = render_context
        self.raster_size = render_context.raster_size_px
        self.pixel_size = render_context.pixel_size_m
        self.ego_center = render_context.center_in_raster_ratio

        self.world_to_ecef = world_to_ecef

        # ADDED by XU =================
        # Check whether it is the expected intersection ID
        assert intersection_id in ["WTgZ", "sGK1"]
        self.intersection_id = intersection_id
        # Use the customized MapAPI class
        self.mapAPI = IntersectionMapAPI(semantic_map_path, world_to_ecef)
        # END ==========================

    def rasterize(
            self,
            history_frames: np.ndarray,
            history_agents: List[np.ndarray],
            history_tl_faces: List[np.ndarray],
            agent: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, bool]:
        if agent is None:
            ego_translation_m = history_frames[0]["ego_translation"]
            ego_yaw_rad = rotation33_as_yaw(history_frames[0]["ego_rotation"])
        else:
            ego_translation_m = np.append(agent["centroid"], history_frames[0]["ego_translation"][-1])
            ego_yaw_rad = agent["yaw"]

        raster_from_world = self.render_context.raster_from_world(ego_translation_m, ego_yaw_rad)
        world_from_raster = np.linalg.inv(raster_from_world)

        # get XY of center pixel in world coordinates
        center_in_raster_px = np.asarray(self.raster_size) * (0.5, 0.5)
        center_in_world_m = transform_point(center_in_raster_px, world_from_raster)

        sem_img, is_intersection_included = self.render_semantic_map(
            center_in_world_m, raster_from_world, history_tl_faces[0])
        return sem_img, is_intersection_included

    def render_semantic_map(
            self, center_in_world: np.ndarray, raster_from_world: np.ndarray, tl_faces: np.ndarray
    ) -> Tuple[np.ndarray, bool]:
        """Renders the semantic map at given x,y coordinates.

        Args:
            center_in_world (np.ndarray): XY of the image center in world ref system
            raster_from_world (np.ndarray):
        Returns:
            np.ndarray: RGB raster
        """
        img = 255 * np.ones(shape=(self.raster_size[1], self.raster_size[0], 3), dtype=np.uint8)
        is_intersection_included = False
        # filter using half a radius from the center
        raster_radius = float(np.linalg.norm(self.raster_size * self.pixel_size)) / 2

        # get active traffic light faces
        # active_tl_ids = set(filter_tl_faces_by_status(tl_faces, "ACTIVE")["face_id"].tolist())

        # get all lanes as interpolation so that we can transform them all together

        lane_indices = indices_in_bounds(center_in_world, self.mapAPI.bounds_info["lanes"]["bounds"], raster_radius)
        # lanes_mask: Dict[str, np.ndarray] = defaultdict(lambda: np.zeros(len(lane_indices) * 2, dtype=np.bool))
        # lanes_area = np.zeros((len(lane_indices) * 2, IntersectionRasterizer.INTERPOLATION_POINTS, 2))

        # ADDED by XU =================================
        # To Find the intersection id, i.e., "sGK1"
        for junction_idx, junction_id in enumerate(self.mapAPI.bounds_info["junctions"]["ids"]):
            if self.intersection_id == junction_id:
                # check whether its lanes are in the scene
                lane_ids_in_junction = self.mapAPI.bounds_info["junctions"]["lane_ids"][junction_idx]
                for idx, lane_idx in enumerate(lane_indices):
                    lane_id = self.mapAPI.bounds_info["lanes"]["ids"][lane_idx]
                    if lane_id in lane_ids_in_junction:
                        # Mark as True if including it.
                        is_intersection_included = True
                        break
                break
        # END ===========================================

        # for idx, lane_idx in enumerate(lane_indices):
        #     lane_idx = self.mapAPI.bounds_info["lanes"]["ids"][lane_idx]
        #
        #     # interpolate over polyline to always have the same number of points
        #     lane_coords = self.mapAPI.get_lane_as_interpolation(
        #         lane_idx, INTERPOLATION_POINTS, InterpolationMethod.INTER_ENSURE_LEN
        #     )
        #     lanes_area[idx * 2] = lane_coords["xyz_left"][:, :2]
        #     lanes_area[idx * 2 + 1] = lane_coords["xyz_right"][::-1, :2]
        #
        #     lane_type = RasterEls.LANE_NOTL.name
        #     lane_tl_ids = set(self.mapAPI.get_lane_traffic_control_ids(lane_idx))
        #     for tl_id in lane_tl_ids.intersection(active_tl_ids):
        #         lane_type = self.mapAPI.get_color_for_face(tl_id)
        #
        #     lanes_mask[lane_type][idx * 2: idx * 2 + 2] = True
        #
        # if len(lanes_area):
        #     lanes_area = cv2_subpixel(transform_points(lanes_area.reshape((-1, 2)), raster_from_world))
        #
        #     for lane_area in lanes_area.reshape((-1, INTERPOLATION_POINTS * 2, 2)):
        #         # need to for-loop otherwise some of them are empty
        #         cv2.fillPoly(img, [lane_area], COLORS[RasterEls.ROAD.name], **CV2_SUB_VALUES)
        #
        #     lanes_area = lanes_area.reshape((-1, INTERPOLATION_POINTS, 2))
        #     for name, mask in lanes_mask.items():  # draw each type of lane with its own color
        #         cv2.polylines(img, lanes_area[mask], False, COLORS[name], **CV2_SUB_VALUES)
        #
        # # plot crosswalks
        # crosswalks = []
        # for idx in indices_in_bounds(center_in_world, self.mapAPI.bounds_info["crosswalks"]["bounds"], raster_radius):
        #     crosswalk = self.mapAPI.get_crosswalk_coords(self.mapAPI.bounds_info["crosswalks"]["ids"][idx])
        #     xy_cross = cv2_subpixel(transform_points(crosswalk["xyz"][:, :2], raster_from_world))
        #     crosswalks.append(xy_cross)
        #
        # cv2.polylines(img, crosswalks, True, COLORS[RasterEls.CROSSWALK.name], **CV2_SUB_VALUES)

        return img, is_intersection_included

    def to_rgb(self, in_im: np.ndarray, **kwargs: dict) -> np.ndarray:
        return (in_im * 255).astype(np.uint8)

    def num_channels(self) -> int:
        return 3
