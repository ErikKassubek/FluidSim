'''
File: bounce_back.py
Project: High-Performance Computing: Fluid Mechanics with Python 2023
Author: Erik Kassubek
-----
Copyright (c) 2023 Erik Kassubek
'''

import numpy as np

from simulation.boundaries.boundary import Boundary, Direction
from simulation.lattice import Lattice


class BounceBack(Boundary):
    """
    Class to implement bounce back boundary conditions.
    """
    def __init__(self, direction: Direction) -> None:
        """
        Constructor
        Args:
            direction (str): Direction of the boundary.
        """
        super().__init__(direction)

    def apply(self, lattice: Lattice):
        """
        Apply the bounce back boundary condition to the given distribution
        Args:
            lattice (Lattice): Lattice to apply the boundary condition to.
        Raises:
            ValueError: If the direction is not valid.
        """
        if self.direction == Direction.SOUTH:
            lattice.f[2, :, 1] = lattice.f[4, :, 0].copy()
            lattice.f[5, :, 1] = np.roll(lattice.f[7, :, 0], 1)
            lattice.f[6, :, 1] = np.roll(lattice.f[8, :, 0], -1)
        elif self.direction == Direction.NORTH:
            lattice.f[4, :, -2] = lattice.f[2, :, -1]
            lattice.f[7, :, -2] = np.roll(lattice.f[5, :, -1], -1)
            lattice.f[8, :, -2] = np.roll(lattice.f[6, :, -1], 1)
        elif self.direction == Direction.EAST:
            lattice.f[3, -2, :] = lattice.f[1, -1, :]
            lattice.f[6, -2, :] = np.roll(lattice.f[8, -1, :], -1)
            lattice.f[7, -2, :] = np.roll(lattice.f[5, -1, :], 1)
        elif self.direction == Direction.WEST:
            lattice.f[1, 1, :] = lattice.f[3, 0, :]
            lattice.f[8, 1, :] = np.roll(lattice.f[6, 0, :], 1)
            lattice.f[5, 1, :] = np.roll(lattice.f[7, 0, :], -1)
        else:
            raise ValueError("Direction not valid.")
