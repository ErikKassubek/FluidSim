'''
File: lbm.py
Project: High-Performance Computing: Fluid Mechanics with Python 2023
Author: Erik Kassubek
-----
Copyright (c) 2023 Erik Kassubek
'''

import numpy as np
import os
import shutil
import time

from simulation.results.visualization import Visualization
from simulation.results.data import Data
from simulation.config import ConfigVis, ConfigData2
from simulation.lattice import Lattice
from simulation.constants import c


class LBM:
    """
    Parent class to run the Lattice Boltzmann method (LBM).
    """
    def __init__(self, Nx: int, Ny: int, omega: float, n: int,
                 boundaries, dry: bool,
                 folder: str, vis_conf: ConfigVis,
                 data_conf: ConfigData2,
                 verbose: bool, time: bool,
                 create_output: bool = True) -> None:
        """
        Constructor
        Args:
            Nx (int): Number of horizontal pixels.
            Ny (int): Number of vertical pixels.
            omega (float): Inverse relaxation time.
            boundaries (List[Boundary]): List of boundaries.
            dry (bool): Whether to simulation contains dry nodes.
            folder (str): Name of the folder to save the plots and data in.
            vis_conf (ConfigVis): Visualization configuration.
            data_conf (ConfigData2): Data configuration.
            verbose (bool): Whether to print the progress of the simulation.
            time (bool): Whether to time the simulation.
            create_output (bool): Whether to create the output folder.
        """
        self.omega = omega  # inverse relaxation time
        self.n = n  # number of simulation steps
        self.boundaries = boundaries  # list of boundaries
        self.vis_conf = vis_conf  # visualization configuration
        self.data_conf = data_conf  # data configuration
        self.plot = vis_conf.density or vis_conf.flow  # whether to plot
        self.time = time  # whether to time the simulation
        self.time_lattice = 0  # time spent in the lattice step
        self.time_total = 0  # time spent total
        self.verbose = verbose  # whether to print the progress of the sim
        self.folder = folder

        # create folder
        if create_output:
            if os.path.exists("../out/" + folder):
                shutil.rmtree("../out/" + folder)
            os.mkdir("../out/" + folder)

        # lattice
        self.lattice = Lattice(Nx, Ny, dry)
        # objects for visualization and data extraction
        self.visualization = Visualization(Nx, Ny, folder, dry, vis_conf,
                                           create_output)
        self.data = Data(folder, self.data_conf)

    def streaming(self):
        """
        Perform the streaming step. np.ndarrays parameters are call by
        reference and the function will therefore transform the value of the
        parameter of f in the function call
        >>> l = LBM(100, 100, 1, 100, [], False, "test", ConfigVis())
        >>> l.lattice.f = np.random.rand(l.lattice.Nv, l.lattice.Nx,
        ...                              l.lattice.Ny)
        >>> f0 = l.lattice.f.copy()  # copy f to f0 because f is changed
        >>> mass_pre = np.sum(l.lattice.get_density())
        >>> l.streaming()
        >>> mass_post = np.sum(l.lattice.get_density())
        >>> # check that f was changed by streaming
        >>> (l.lattice.f == f0).all()
        False
        >>> # check that the total mass has not changed
        >>> abs(mass_post - mass_pre) < 10**(-11)
        True
        """
        for i in range(self.lattice.Nv):
            self.lattice.f[i, :, :] = np.roll(self.lattice.f[i, :, :],
                                              c[i], axis=(0, 1))

    def create_f(self) -> None:
        """
        Create the initial density probability function f for a resting fluid
        with density 1.
        """
        # create f including dry nodes
        self.lattice.f = np.zeros((self.lattice.Nv, self.lattice.Nx,
                                   self.lattice.Ny))
        for j in range(self.lattice.Nx):
            for k in range(self.lattice.Ny):
                self.lattice.f[0][j][k] = 1

    def collision(self):
        """
        Perform the collision step.
        """
        feq = self.lattice.calculate_equilibrium_total()
        # omega is the inverse relaxation time
        self.lattice.f += self.omega * (feq - self.lattice.f)

    def update(self) -> None:
        """
        Perform one update step of the siumulation.
        """
        self.streaming()
        self.handle_boundaries()
        self.collision()

    def handle_boundaries(self):
        """
        Handle the boundaries.
        """
        for b in self.boundaries:
            b.apply(self.lattice)

    def simulate(self) -> None:
        """
        Perform n steps of the simulation.
        """
        if self.time:
            self.time_lattice = 0  # time for only the lattic update operations
            time_total_start = time.time()  # start time of the simulation

        # number of digits of n needed for file names
        length = len(str(self.n))
        self.visualization.visualize(self.lattice, "0" * length)

        self.data.save(self.lattice)

        for i in range(self.n):
            save_name = f"{i + 1:0{length}d}"
            if self.verbose:
                print(f"\r{self.folder}: {save_name}/{self.n}", end="")

            if self.time:
                time_lattice_start = time.time()
            self.update()
            if self.time:
                self.time_lattice += time.time() - time_lattice_start

            self.visualization.visualize(self.lattice, save_name=save_name)

            self.data.save(self.lattice)

        self.visualization.visualize(self.lattice, save_name="final")

        if self.verbose:
            print()

        if self.vis_conf.gif and self.plot:
            if self.verbose:
                print(f"{self.folder}: Creating gif...")
            self.visualization.create_gif()

        if self.time:
            self.time_total = time.time() - time_total_start

    def run(self) -> None:
        """
        Run the simulation.
        """
        self.create_f()
        self.simulate()
        if self.time:
            print(f"Total time: {self.time_total:.2f} s")
            print(f"Total lattice update time: {self.time_lattice:.2f} s")
            tpla = self.time_lattice / self.n * 1000
            print(f"Avg. time per lattice update: {tpla:.4f} ms")
            mlups = (self.lattice.Nx * self.lattice.Ny * self.n /
                     self.time_lattice / 10**6)
            print(f"MLUPS: {mlups:.2f}")
