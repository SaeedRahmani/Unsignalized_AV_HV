import numpy as np
from typing import  no_type_check
from l5kit.data.map_api import MapAPI, MapElement


class IntersectionMapAPI(MapAPI):
    @staticmethod
    @no_type_check
    def is_junction(element: MapElement) -> bool:
        """
        @ Added by ourselves. @
        @ For determining whether an element is junction or not. @

        Check whether an element is a valid junction

        Args:
            element (MapElement): a proto element

        Returns:
            bool: True if the element is a valid junction
        """
        return bool(element.element.HasField("junction"))

    def get_bounds(self) -> dict:
        """
        For each elements of interest returns bounds [[min_x, min_y],[max_x, max_y]] and proto ids
        Coords are computed by the MapAPI and, as such, are in the world ref system.

        Returns:
            dict: keys are classes of elements, values are dict with `bounds` and `ids` keys
        """
        lanes_ids = []
        crosswalks_ids = []
        junctions_ids = []  # intersection
        lane_ids_in_junction = []
        node_ids_in_junction = []

        lanes_bounds = np.empty((0, 2, 2), dtype=float)  # [(X_MIN, Y_MIN), (X_MAX, Y_MAX)]
        crosswalks_bounds = np.empty((0, 2, 2), dtype=float)  # [(X_MIN, Y_MIN), (X_MAX, Y_MAX)]

        for element in self.elements:
            element_id = MapAPI.id_as_str(element.id)

            if self.is_lane(element):
                lane = self.get_lane_coords(element_id)
                x_min = min(np.min(lane["xyz_left"][:, 0]), np.min(lane["xyz_right"][:, 0]))
                y_min = min(np.min(lane["xyz_left"][:, 1]), np.min(lane["xyz_right"][:, 1]))
                x_max = max(np.max(lane["xyz_left"][:, 0]), np.max(lane["xyz_right"][:, 0]))
                y_max = max(np.max(lane["xyz_left"][:, 1]), np.max(lane["xyz_right"][:, 1]))

                lanes_bounds = np.append(lanes_bounds, np.asarray([[[x_min, y_min], [x_max, y_max]]]), axis=0)
                lanes_ids.append(element_id)

            if self.is_crosswalk(element):
                crosswalk = self.get_crosswalk_coords(element_id)
                x_min, y_min = np.min(crosswalk["xyz"], axis=0)[:2]
                x_max, y_max = np.max(crosswalk["xyz"], axis=0)[:2]

                crosswalks_bounds = np.append(
                    crosswalks_bounds, np.asarray([[[x_min, y_min], [x_max, y_max]]]), axis=0,
                )
                crosswalks_ids.append(element_id)

            # @ Add by ourselves. @
            # @ To get the IDs related to junctions, nodes and lanes. @
            # @ Use these IDs to identify the interested intersections. @
            if self.is_junction(element):
                junction = element.element.junction
                lanes_in_junction = junction.lanes
                nodes_in_junction = junction.road_network_nodes
                junctions_ids.append(element_id)
                lane_ids_in_junction.append([str(lane_id)[5:9] for lane_id in list(lanes_in_junction)])
                node_ids_in_junction.append([str(lane_id)[5:9] for lane_id in list(nodes_in_junction)])

        return {
            "lanes": {"bounds": lanes_bounds, "ids": lanes_ids},
            "crosswalks": {"bounds": crosswalks_bounds, "ids": crosswalks_ids},
            # @ Add by ourselves. @
            "junctions": {
                "ids": junctions_ids,
                "lane_ids": lane_ids_in_junction,
                "node_ids": node_ids_in_junction,
            },
        }