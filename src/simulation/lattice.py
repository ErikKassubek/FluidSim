'''
File: lattice.py
Project: High-Performance Computing: Fluid Mechanics with Python 2023
Author: Erik Kassubek
-----
Copyright (c) 2023 Erik Kassubek
'''

import numpy as np
from simulation.constants import c, w


class Lattice:
    """
    Class to implement the lattice
    """
    def __init__(self, Nx: int, Ny: int, dry: bool):
        """
        Constructor
        Args:
            Nx (int): Number of horizontal pixels.
            Ny (int): Number of vertical pixels.
            dry (bool): Whether to simulation contains dry nodes.
        """
        assert Nx > 0, "Nx must be positive"
        assert Ny > 0, "Ny must be positive"

        self.Nv = 9

        self.Nx = Nx + 2 if dry else Nx
        self.Ny = Ny + 2 if dry else Ny

        self.f = np.zeros((self.Nv, self.Nx, self.Ny))
        self.dry = dry

    def set_f(self, f: np.ndarray) -> None:
        """
        Set the density function.
        Args:
            f (np.ndarray): Density function.
        """
        assert f.shape == self.f.shape, \
            "f must have shape (Nv, Nx, Ny)"
        self.f = f

    def get_density(self) -> np.ndarray:
        """
        Calculate the dencity distribution based on the density function.
        Returns:
            np.ndarray: density distribution
        >>> l = Lattice(100, 100, False)
        >>> l.f = np.zeros((l.Nv, l.Nx, l.Ny))
        >>> rho0 = l.get_density()
        >>> exp0 = np.zeros((l.Nx, l.Ny))
        >>> (rho0 == exp0).all()
        True
        >>> l.f = np.ones((l.Nv, l.Nx, l.Ny))
        >>> rho1 = l.get_density()
        >>> exp1 = np.ones((l.Nx, l.Ny)) * l.Nv
        >>> (rho1 == exp1).all()
        True
        """
        return np.sum(self.f, axis=0)

    def get_velocity(self) -> np.ndarray:
        """
        Calculate the velocity distribution from the density function.
        Returns:
            np.array: velocity distribution
        """
        rho = self.get_density()
        u = np.zeros((2, self.Nx, self.Ny))
        for i in range(self.Nv):
            u[0] += self.f[i] * c[i, 0]
            u[1] += self.f[i] * c[i, 1]
        return np.divide(u, rho)

    def calculate_equilibrium_total(self) -> np.ndarray:
        """
        Calculate the equilibrium distribution for the whole lattice.
        Returns:
            np.ndarray: equilibrium distribution
        """
        rho = self.get_density()
        return self.calculate_equilibrium(rho)

    def calculate_equilibrium(self, rho) -> np.ndarray:
        """
        Calculate the equilibrium distribution for given density and velocity.
        Args:
            rho (np.ndarray): density
            u (np.ndarray): velocity
        Returns:
            np.ndarray: equilibrium distribution
        """
        u = self.get_velocity()
        feq = np.zeros((self.Nv, self.Nx, self.Ny))
        cu = np.tensordot(c, u, axes=([1], [0]))
        for i in range(self.Nv):
            feq[i] = w[i] * rho * (1 + 3 * cu[i] + 9/2 * cu[i] ** 2
                                   - 3/2 * np.linalg.norm(u, axis=0)**2)
        return feq
