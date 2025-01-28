'''
File: moving_wall.py
Project: High-Performance Computing: Fluid Mechanics with Python 2023
Author: Erik Kassubek
-----
Copyright (c) 2023 Erik Kassubek
'''

import numpy as np

from simulation.boundaries.boundary import Boundary, Direction
from simulation.lattice import Lattice


class MovingWall(Boundary):
    """
    Class to implement moving wall boundary conditions.
    """
    def __init__(self, direction: Direction, velocity: float) -> None:
        """
        Constructor
        Args:
            direction (str): Direction of the boundary.
        """
        super().__init__(direction)
        self.v = velocity

    def apply(self, lattice: Lattice):
        """
        Apply the moving wall boundary condition to the given distribution
        Args:
            lattice (Lattice): Lattice to apply the boundary condition to.
        Raises:
            NotImplementedError: If the boundary condition is not implemented
        """
        if self.direction == Direction.NORTH:
            lattice.f[5, :, -1] = np.roll(lattice.f[5, :, -1], 1)
            lattice.f[6, :, -1] = np.roll(lattice.f[6, :, -1], -1)

            rho = (lattice.f[0, :, -1] + lattice.f[1, :, -1] +
                   lattice.f[3, :, -1] + 2 * (lattice.f[2, :, -1] +
                                              lattice.f[5, :, -1] +
                                              lattice.f[6, :, -1]))
            lattice.f[4, :, -1] = lattice.f[2, :, -1]
            lattice.f[7, :, -1] = lattice.f[5, :, -1] + 1/2 * \
                (lattice.f[1, :, -1] - lattice.f[3, :, -1]) - \
                1/2 * rho * self.v
            lattice.f[8, :, -1] = lattice.f[6, :, -1] - 1/2 * \
                (lattice.f[1, :, -1] - lattice.f[3, :, -1]) + \
                1/2 * rho * self.v

        else:
            raise NotImplementedError(
                "Moving wall not implemented for this direction")
