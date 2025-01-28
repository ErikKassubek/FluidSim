'''
File: boundary.py
Project: High-Performance Computing: Fluid Mechanics with Python 2023
Author: Erik Kassubek
-----
Copyright (c) 2023 Erik Kassubek
----------------------------------------------------------------------
'''

from enum import Enum


class Direction(Enum):
    """
    Enum for the direction of the boundary.
    """
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3


class Boundary:
    """
    Base class for boundary conditions.
    """
    def __init__(self, direction: Direction) -> None:
        """
        Constructor
        Args:
            direction (str): Direction of the boundary.
        """
        self.direction = direction
