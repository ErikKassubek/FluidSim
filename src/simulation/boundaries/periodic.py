'''
File: periodic.py
Project: High-Performance Computing: Fluid Mechanics with Python 2023
Created Date: 22 Jul 2023
Author: Erik Kassubek
-----
Copyright (c) 2023 Erik Kassubek
----------------------------------------------------------------------
'''

from simulation.boundaries.boundary import Boundary
from simulation.lattice import Lattice
from simulation.boundaries.boundary import Direction


class Periodic(Boundary):
    """
    Class to implement periodic boundary conditions.
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
        Apply the periodic boundary condition to the given distribution.
        The periodic boundary condition is implemented automatically by
        streaming. Therefore there is nothing to do here.
        Args:
            lattice (Lattice): Lattice to apply the boundary condition to.
        Raises:
            NotImplementedError: If the boundary condition is not implemented
        """
        pass
