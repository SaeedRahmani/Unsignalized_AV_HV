from enum import Enum

class RoadUser(Enum):
    HumanDrivenVehicle = "HV"
    AutomatedVehicle = "AV"
    Pedestrian = "Pedestrian"
    Cyclist = "Cyclist"
    Unknown = "Unknown"

    def __str__(self):
        return self.value
