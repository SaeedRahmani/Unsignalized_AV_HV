import numpy as np

def build_trajectory_coordinates(coord_x: np.ndarray, coord_y: np.ndarray):
    assert coord_x.shape == coord_y.shape, \
        f"""The shape of coord_x and coord_y must be the same, but received {coord_x.shape} and {coord_y.shape}."""
    coords = np.column_stack((coord_x, coord_y))
    assert coords.shape[1] == 2
    return coords

class Trajectory(object):
    """
    A trajectory of the road users (mainly vehicles,
    both human-driven and automated vehicles).
    """
    def __init__(
            self,
            coord_x: np.ndarray,
            coord_y: np.ndarray,
    ):
        # Coordinates
        self.coord_x: np.ndarray = coord_x
        self.coord_y: np.ndarray = coord_y
        self.coords: np.ndarray = build_trajectory_coordinates(coord_x, coord_y)

        # Velocity

        # Acceleration

    def __str__(self) -> str:
        return f"Trajectory{self.coords.shape}"

    def __repr__(self) -> str:
        return f"Trajectory{self.coords})"

if __name__ == "__main__":
    length = 3
    trajectory = Trajectory(coord_x=np.arange(length), coord_y=np.arange(length))
    assert trajectory.coords.shape == (length, 2)
    print(trajectory)
