'''
File: sliding_lid_serial.py
Project: High-Performance Computing: Fluid Mechanics with Python 2023
Author: Erik Kassubek
-----
Copyright (c) 2023 Erik Kassubek
'''

from simulation.lbm import LBM
from simulation.config import ConfigVis, ConfigData, ConfigData2
from simulation.boundaries.bounce_back import BounceBack
from simulation.boundaries.moving_wall import MovingWall
from simulation.boundaries.boundary import Direction


class SlidingLidSerial(LBM):
    """
    Class to run the serial sliding lid simulation
    """
    def __init__(self, Nx: int, Ny: int, omega: float, n: int,
                 vis_conf: ConfigVis, data_conf: ConfigData,
                 verbose: bool, time: bool, vel: float):
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
            vel (float): Velocity of the sliding lid.
        """
        boundaries = [
            MovingWall(Direction.NORTH, vel),
            BounceBack(Direction.SOUTH),
            BounceBack(Direction.WEST),
            BounceBack(Direction.EAST)
        ]

        conf_data = ConfigData2(
            amplitude_vel=data_conf.amplitude,
            vertical_vel=data_conf.cut
        )

        super().__init__(Nx, Ny, omega, n, boundaries, True,
                         "sliding_lid_serial", vis_conf, conf_data, verbose,
                         time)
