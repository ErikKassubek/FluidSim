'''
File: couette_flow.py
Project: High-Performance Computing: Fluid Mechanics with Python 2023
Author: Erik Kassubek
-----
Copyright (c) 2023 Erik Kassubek
'''

from simulation.lbm import LBM
from simulation.boundaries.bounce_back import BounceBack
from simulation.boundaries.moving_wall import MovingWall
from simulation.boundaries.periodic import Periodic
from simulation.boundaries.boundary import Direction
from simulation.config import ConfigVis, ConfigData, ConfigData2


class CouetteFlow(LBM):
    """
    Class to run the couette flow simulation
    """
    def __init__(self, Nx: int, Ny: int, omega: float, n: int,
                 vis_conf: ConfigVis, data_conf: ConfigData,
                 verbose: bool, time: bool, vel: float) -> None:
        """
        Constructor
        Args:
            Nx (int): Number of horizontal pixels.
            Ny (int): Number of vertical pixels.
            omega (float): Relaxation parameter.
            n (int): Number of iterations.
            vis_conf (ConfigVis): Visualization configuration.
            data_conf (ConfigData): Data configuration.
            verbose (bool): Whether to print the progress of the simulation.
            time (bool): Whether to time the simulation.
            vel (float): Velocity of the moving wall.
        """
        boundaries = [
            MovingWall(Direction.NORTH, vel),
            BounceBack(Direction.SOUTH),
            Periodic(Direction.EAST),
            Periodic(Direction.WEST)
        ]

        conf_data = ConfigData2(
            amplitude_vel=data_conf.amplitude,
            vertical_vel=data_conf.cut
        )

        super().__init__(Nx, Ny, omega, n, boundaries, True, "couette",
                         vis_conf, conf_data, verbose, time)
