'''
File: periodic_pressure.py
Project: High-Performance Computing: Fluid Mechanics with Python 2023
Author: Erik Kassubek
-----
Copyright (c) 2023 Erik Kassubek
----------	---	---------------------------------------------------------
'''

from simulation.boundaries.boundary import Boundary, Direction
from simulation.lattice import Lattice
from simulation.constants import cs2


class PeriodicPressure(Boundary):
    """
    Class to implement periodic boundary conditions with pressure variations.
    """
    def __init__(self, direction: Direction, pressure: float) -> None:
        """
        Constructor
        Args:
            direction (Direction): Direction of the boundary.
            pressure (float): Pressure.
        """
        super().__init__(direction)
        self.rho = pressure / (cs2)

    def apply(self, lattice: Lattice):
        """
        Apply the periodic boundary condition with perasure variation to the
        given distribution.
        Args:
            lattice (Lattice): Lattice to apply the boundary condition to.
        """
        if self.direction == Direction.WEST:
            pos1 = 0
            pos2 = -2
        elif self.direction == Direction.EAST:
            pos1 = -1
            pos2 = 1
        else:
            raise NotImplementedError("Periodic boundary condition not " +
                                      "implemented for this direction.")

        feq1 = lattice.calculate_equilibrium(self.rho)[:, pos1, :]
        feq2 = lattice.calculate_equilibrium_total()[:, pos2, :]

        lattice.f[:, pos1, :] = feq1 + (lattice.f[:, pos2, :] - feq2)
