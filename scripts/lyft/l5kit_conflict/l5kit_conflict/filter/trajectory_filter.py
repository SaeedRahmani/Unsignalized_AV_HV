import numpy as np
from shapely.geometry import LineString

"""
These seven functions are used to identify potential conflicts 
in the first intersection with Junction_ID of "sGK1". 
"""
def is_AV_turnleft_or_right(trajectoryLineString: LineString) -> bool:
	""" Check if the AV turns left (True) or right (False) """
	assert isinstance(trajectoryLineString, LineString)
	positions = np.array(trajectoryLineString.coords)
	delta_position = positions[0] - positions[-1]
	delta_position_x, delta_position_y = delta_position[0], delta_position[1]
	return True if delta_position_x > 0 and delta_position_y > 0 else False


def is_HV_straight2right(trajectoryLineString: LineString) -> bool:
	""" Check if the HV go straight from left to right"""
	assert isinstance(trajectoryLineString, LineString)
	positions = np.array(trajectoryLineString.coords)
	delta_position = positions[0] - positions[-1]
	delta_position_x, delta_position_y = delta_position[0], delta_position[1]
	return True if delta_position_x < -30 and abs(delta_position_y) < 2 else False


def is_HV_straight2left(trajectoryLineString: LineString) -> bool:
	""" Check if the HV go straight from right to left"""
	assert isinstance(trajectoryLineString, LineString)
	positions = np.array(trajectoryLineString.coords)
	delta_position = positions[0] - positions[-1]
	delta_position_x, delta_position_y = delta_position[0], delta_position[1]
	return True if delta_position_x > 30 and abs(delta_position_y) < 2 else False


def is_HV_turnright_from_left(trajectoryLineString: LineString) -> bool:
	""" Check if the HV turns right from the top-left arm of T-junction """
	assert isinstance(trajectoryLineString, LineString)
	positions = np.array(trajectoryLineString.coords)
	delta_position = positions[0] - positions[-1]
	delta_position_x, delta_position_y = delta_position[0], delta_position[1]
	return True if delta_position_x < -5 and delta_position_y > 5 else False


def is_HV_turnright_from_bottom(trajectoryLineString: LineString) -> bool:
	""" Check if the HV turns right from the bottom-center arm of T-junction """
	assert isinstance(trajectoryLineString, LineString)
	positions = np.array(trajectoryLineString.coords)
	delta_position = positions[0] - positions[-1]
	delta_position_x, delta_position_y = delta_position[0], delta_position[1]
	return True if delta_position_x < -5 and delta_position_y < -5 else False


def is_HV_turnleft_from_right(trajectoryLineString: LineString) -> bool:
	""" Check if the HV turns left from the top-right arm of T-junction """
	assert isinstance(trajectoryLineString, LineString)
	positions = np.array(trajectoryLineString.coords)
	delta_position = positions[0] - positions[-1]
	delta_position_x, delta_position_y = delta_position[0], delta_position[1]
	return True if delta_position_x > 15 and delta_position_y > 5 else False


def is_HV_turnleft_from_bottom(trajectoryLineString: LineString) -> bool:
	""" Check if the HV turns left from the bottom-center arm of T-junction """
	assert isinstance(trajectoryLineString, LineString)
	positions = np.array(trajectoryLineString.coords)
	delta_position = positions[0] - positions[-1]
	delta_position_x, delta_position_y = delta_position[0], delta_position[1]
	return True if delta_position_x > 5 and delta_position_y < -15 else False


"""
These N functions are used to identify potential conflicts 
in the second intersection with Junction_ID of "WTgZ". 
"""

def is_AV_go_left(trajectoryLineString: LineString) -> bool:
    """
    Check if the autonomous vehicle (AV) is moving from right to left.

    Args:
        trajectoryLineString (LineString): The trajectory of the AV.

    Returns:
        bool: True if moving right to left, False if left to right.
    """
    assert isinstance(trajectoryLineString, LineString)
    positions = np.array(trajectoryLineString.coords)
    delta_position = positions[0] - positions[-1]
    delta_position_x = delta_position[0]
    
    return True if delta_position_x > 0 else False


def is_HV_turnleft_from_left(trajectoryLineString: LineString) -> bool:
	""" Check if the HV turns left the left-arm of T-junction 2. """
	assert isinstance(trajectoryLineString, LineString)
	positions = np.array(trajectoryLineString.coords)
	delta_position = positions[0] - positions[-1]
	delta_position_x, delta_position_y = delta_position[0], delta_position[1]
	return True if delta_position_x < -5 and delta_position_y < -5 else False


def is_HV_turnleft_from_top(trajectoryLineString: LineString) -> bool:
    """ Check if the HV turns left the top-arm of T-junction 2. """
    assert isinstance(trajectoryLineString, LineString)
    positions = np.array(trajectoryLineString.coords)
    delta_position = positions[0] - positions[-1]
    delta_position_x, delta_position_y = delta_position[0], delta_position[1]
    return True if delta_position_x < -5  and delta_position_y > 5 else False


def is_HV_turnright_from_top(trajectoryLineString: LineString) -> bool:
    """ Check if the HV turns right the top-arm of T-junction 2. """
    assert isinstance(trajectoryLineString, LineString)
    positions = np.array(trajectoryLineString.coords)
    delta_position = positions[0] - positions[-1]
    delta_position_x, delta_position_y = delta_position[0], delta_position[1]
    return True if delta_position_x > 5 and delta_position_y > 5 else False
