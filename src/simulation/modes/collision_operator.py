'''
File: milestone_2.py
Project: High-Performance Computing: Fluid Mechanics with Python 2023
Author: Erik Kassubek
-----
Copyright (c) 2023 Erik Kassubek
'''

import numpy as np

from simulation.lbm import LBM
from simulation.config import ConfigVis, ConfigData2
from simulation.boundaries.periodic import Periodic
from simulation.boundaries.boundary import Direction


class CollisionOperator(LBM):
    """
    Class for mode 2.
    """

    def __init__(self, Nx: int, Ny: int, omega: float, n: int,
                 vis_conf: ConfigVis, verbose: bool, time: bool) -> None:
        """
        Constructor
        Args:
            Nx (int): Number of horizontal pixels.
            Ny (int): Number of vertical pixels.
            omega (float): Relaxation parameter.
            n (int): Number of iterations.
            vis_conf (ConfigVis): Visualization configuration.
            verbose (bool): Whether to print the progress of the simulation.
            time (bool): Whether to time the simulation.
        """
        boundaries = [
            Periodic(Direction.NORTH),
            Periodic(Direction.EAST),
            Periodic(Direction.SOUTH),
            Periodic(Direction.WEST)
        ]
        super().__init__(Nx, Ny, omega, n, boundaries, False, "milestone_2",
                         vis_conf, ConfigData2(), verbose, time)

    def create_f(self) -> None:
        """
        Create the initial density probability function f.
        """
        self.lattice.f = np.zeros((self.lattice.Nv, self.lattice.Nx,
                                   self.lattice.Ny))
        for i in range(self.lattice.Nv):
            for j in range(self.lattice.Nx):
                for k in range(self.lattice.Ny):
                    self.lattice.f[i][j][k] = 1
        for i in range(self.lattice.Nv):
            self.lattice.f[i][self.lattice.Nx//2][self.lattice.Ny//2] = 100
