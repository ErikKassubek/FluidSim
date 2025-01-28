'''
File: shear_wave_decay.py
Project: High-Performance Computing: Fluid Mechanics with Python 2023
Author: Erik Kassubek
-----
Copyright (c) 2023 Erik Kassubek
'''

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from simulation.lbm import LBM
from simulation.config import ConfigData, ConfigVis, ConfigData2
from simulation.boundaries.periodic import Periodic
from simulation.boundaries.boundary import Direction


class ShearWaveDecay(LBM):
    """
    Class to run the shear wave decay simulation.
    """
    def __init__(self, Nx: int, Ny: int, omega: float, n: int, rho: float,
                 epsilon: float, vis_conf: ConfigVis, data_conf: ConfigData,
                 verbose: bool, time: bool
                 ) -> None:
        """
        Constructor
        Args:
            Nx (int): Number of horizontal pixels.
            Ny (int): Number of vertical pixels.
            omega (float): Relaxation parameter.
            n (int): Number of iterations.
            rho_dens (float): Shift. If None: simulate swd in velocity,
                otherwise in density.
            epsilon_amplitude (float): Density amplitude.
            epsilon_vel (float): Velocity amplitude.
            vis_conf (ConfigVis): Visualization configuration.
            data_conf (ConfigData): Data configuration.
            verbose (bool): If True, print additional information.
            time (bool): If True, time the simulation.
        """
        self.rho = rho
        self.epsilon = epsilon

        self.density = True if rho is not None else False

        folder = "swd_des" if self.density else "swd_vel"

        if self.rho is not None:
            assert self.rho > self.epsilon  # otherwise the density is negative

        # set data config
        conf_data = ConfigData2(
            amplitude_dens=self.density and data_conf.amplitude,
            point_dens=(Nx//4, Ny//2),
            horizontal_dens=self.density and data_conf.cut,
            vertical_dens=self.density and data_conf.cut,
            amplitude_vel=not self.density and data_conf.amplitude,
            point_vel=(Nx//2, Ny//4),
            horizontal_vel=not self.density and data_conf.cut,
            vertical_vel=not self.density and data_conf.cut,
        )

        # set boundary conditions
        boundaries = [
            Periodic(Direction.EAST),
            Periodic(Direction.WEST),
            Periodic(Direction.NORTH),
            Periodic(Direction.SOUTH)
        ]

        super().__init__(Nx, Ny, omega, n, boundaries, False, folder, vis_conf,
                         conf_data, verbose, time)

    def create_f_density(self) -> np.ndarray:
        """
        Get the density probability function f for the initial state of the
        shear wave decay with rho(r, 0) = rho0 + epsilon * sin(2 * pi * x / Lx)
        and u(r, 0) = 0.
        Args:
            rho0 (float): shift
            epsilon (float): amplitude
        >>> s = ShearWaveDecay(9, 2)
        >>> s.create_f_density(1, 0.1)
        >>> u = s.get_velocity()
        >>> not np.any(u)  # all velocities are zero
        True
        """
        self.lattice.f = np.zeros((self.lattice.Nv, self.lattice.Nx,
                                   self.lattice.Ny))
        for i in range(self.lattice.Nx):
            for j in range(self.lattice.Ny):
                self.lattice.f[0][i][j] = self.rho + self.epsilon * \
                    np.sin(2 * np.pi * i / self.lattice.Nx)

    def create_f_velocity(self) -> np.ndarray:
        """
        Get the density probability function f for the initial state of the
        shear wave decay with rho(r, 0) = 1 and
        u(r, 0) = epsilon * sin(2 * pi * y / Ly).
        Args:
            epsilon (float): amplitude
        >>> s = ShearWaveDecay(9, 2)
        >>> s.create_f_velocity(0.1)
        >>> rho = s.get_density()
        ... # Test that all values of rho are 1 except for floating point
        ... # inaccuracies
        >>> np.all(rho <= 1.0 + 10**(-12), axis=None)
        True
        >>> np.all(rho >= 1.0 - 10**(-12), axis=None)
        True
        """
        self.lattice.f = np.zeros((self.lattice.Nv, self.lattice.Nx,
                                   self.lattice.Ny))
        for i in range(self.lattice.Nx):
            for j in range(self.lattice.Ny):
                vel = self.epsilon * np.sin(2 * np.pi * j / self.lattice.Ny)
                if vel >= 0:
                    self.lattice.f[1][i][j] = vel
                else:
                    vel *= -1
                    self.lattice.f[3][i][j] = vel

                self.lattice.f[0][i][j] = 1 - vel

    def run(self) -> None:
        """
        Perform the shear wave decay simulation.
        Args:
            density (bool): If true, use sin in density (see
                create_f_density), otherwise use sin in velocity (see
                create_f_velocity).
        """
        if self.density:
            self.create_f_density()
            self.simulate()
        else:
            self.create_f_velocity()
            self.simulate()


def calculate_viscosity(file_name, steps, analytic: str) -> float:
    """
    Calculate the viscosity of the data.
    Args:
        file_name (str): file name
        steps (int): number of steps. Defaults to 5000.
        analytic (str): vel for velocity, dens for density.
    Returns:
        str: viscosity
    """
    data = pd.read_csv(file_name, sep=',', header=None).values[:, 0]
    if analytic == "dens":
        x, data = get_peaks(data)

        def func(x, t):
            return 1 + 0.01 * np.exp(-x * (2 * np.pi / 100) ** 2 * t)
    elif analytic == "vel":
        x = np.arange(0, steps + 1)

        def func(x, t):
            return 0.05 * np.exp(-x * (2 * np.pi / 100) ** 2 * t)
    else:
        return 0.0

    popt, pcov = curve_fit(func, x, list(data))
    return popt[0]


def get_peaks(data: np.ndarray):
    """
    Get the peaks of the data.
    Args:
        data (np.ndarray): data
    Returns:
        tuple[np.ndarray, np.ndarray]: peaks
    """
    res_ind = []
    res_val = []
    for i in range(1, len(data) - 1):
        if data[i - 1] < data[i] and data[i] > data[i + 1]:
            res_ind.append(i)
            res_val.append(data[i])
    return np.array(res_ind), np.array(res_val)


if __name__ == "__main__":
    print(calculate_viscosity(
        "../shear_wave_decay_density/data/density.csv", analytic="dens"))
    print(calculate_viscosity(
        "../shear_wave_decay_velocity/data/velocity.csv", analytic="vel"))
