'''
File: poiseuille_flow.py
Project: High-Performance Computing: Fluid Mechanics with Python 2023
Author: Erik Kassubek
-----
Copyright (c) 2023 Erik Kassubek
'''

from simulation.lbm import LBM
from simulation.config import ConfigVis, ConfigData, ConfigData2
from simulation.boundaries.bounce_back import BounceBack
from simulation.boundaries.periodic_pressure import PeriodicPressure
from simulation.boundaries.boundary import Direction
from simulation.constants import cs2


class PoiseuilleFlow(LBM):
    """
    Class for poiseuille flow.
    """
    def __init__(self, Nx: int, Ny: int, omega: float, n: int,
                 vis_conf: ConfigVis, data_conf: ConfigData,
                 verbose: bool, time: bool, delta_p: float) -> None:
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
            delta_p (float): Pressure difference.
        """
        p_in = cs2 + delta_p / 2  # p = cs2 -> rho = 1
        p_out = cs2 - delta_p / 2

        boundaries = [
            BounceBack(Direction.NORTH),
            BounceBack(Direction.SOUTH),
            PeriodicPressure(Direction.WEST, p_in),  # p in
            PeriodicPressure(Direction.EAST, p_out)  # p out
        ]

        conf_data = ConfigData2(
            amplitude_vel=data_conf.amplitude,
            vertical_vel=data_conf.cut
        )

        vis_conf.plot_size = (9, 3)

        super().__init__(Nx, Ny, omega, n, boundaries, True, "poiseuille",
                         vis_conf, conf_data, verbose, time)
