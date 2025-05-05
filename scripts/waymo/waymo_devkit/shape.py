import numpy as np
from shapely import Point, LineString, Polygon

def construct_LaneLineString(
    lanes: list,
    buffer: float = 1,
) -> list[LineString]:
    """ 
    Return a shapely.polygon of the intersection.
    @param: center_coordinate: 
    @param: radius:
    @param: num_points:
    @return: a intersection polygon 
    """
    lane_lineStrings = list()
    for lane in lanes:
        lane_lineStrings.append(
            LineString(lane[2]).buffer(buffer)
        )
    
    return lane_lineStrings
    
def construct_intersection_polygon(
    center_coordinate: list,
    radius: float,
    num_points: int = 100,    
) -> Polygon:
    """ 
    Return a shapely.polygon of the intersection.
    @param: center_coordinate: 
    @param: radius:
    @param: num_points:
    @return: a intersection polygon 
    """
    theta = np.linspace(0, 2 * np.pi, num_points) 
    xs = center_coordinate[0] + radius * np.cos(theta)
    ys = center_coordinate[1] + radius * np.sin(theta)
    coordinates = np.vstack([xs, ys]).T
    assert coordinates.shape[1] == 2
    return Polygon(coordinates)