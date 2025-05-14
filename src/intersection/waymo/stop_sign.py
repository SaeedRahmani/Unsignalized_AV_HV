class StopSign:
    def __init__(self, id: int, coord_x: float, coord_y: float):
        self.id = id
        self.coord_x = coord_x
        self.coord_y = coord_y
        self.coords = [self.coord_x, self.coord_y]

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f'StopSign({self.id}, {self.coord_x}, {self.coord_y})'