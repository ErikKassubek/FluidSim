'''
File: data.py
Project: High-Performance Computing: Fluid Mechanics with Python 2023
Author: Erik Kassubek
-----
Copyright (c) 2023 Erik Kassubek
'''

import numpy as np
import os

from simulation.config import ConfigData2
from simulation.lattice import Lattice


class Data:
    def __init__(self, folder_name: str, conf_data: ConfigData2) -> None:
        """
        Create a Data object
        Args:
            folder_name (str): name of the folder to save the data in.
            conf_data (ConfigData2): data configuration
        """
        self.folder_name = "../out/" + folder_name

        self.conf = conf_data

        # create folder
        if conf_data.any:
            os.mkdir(self.folder_name + "/data")

    def save(self, lattice: Lattice) -> None:
        """
        Save the data
        Args:
            lattice (Lattice): lattice
        """
        # make sure input is coherent
        if self.conf.amplitude_dens:
            if self.point_dens is None:
                raise ValueError(
                    "point_dens must be specified if amplitude_dens is True")
            if len(self.point_dens) != 2:
                raise ValueError("point_dens must be a tuple of length 2")
        if self.conf.horizontal_dens:
            if self.vertical_dens:
                raise ValueError(
                    "only one of horizontal_dens and vertical_dens can " +
                    "be True")
            if len(self.point_dens) != 2:
                raise ValueError("point_dens must be a tuple of length 2")

        if self.conf.amplitude_dens:
            self.save_density_point(lattice, self.conf.point_dens[0],
                                    self.conf.point_dens[1])
        if self.conf.horizontal_dens:
            self.save_density_horizontal_cut(lattice)
        if self.conf.vertical_dens:
            self.save_density_vertical_cut(lattice)

        if self.conf.amplitude_vel:
            self.save_velocity_point(lattice, self.conf.point_vel[0],
                                     self.conf.points_vel[1])
        if self.conf.horizontal_vel:
            self.save_velocity_horizontal_cut(lattice)
        if self.conf.vertical_vel:
            self.save_velocity_vertical_cut(lattice)

    def save_density_horizontal_cut(self, lattice: Lattice) -> None:
        """
        Save the density distribution for y == self.Ny//2 of the last step in
            a txt file
        Args:
            lattice (Lattice): lattice
        """
        with open(self.folder_name + "/data/density_h.csv", "a") as file:
            density = lattice.get_density()
            for i in range(0, lattice.Nx):
                if i != 0:
                    print(",", file=file, end="")
                print(density[i, lattice.Ny//2], file=file, end="")
            print("", file=file)

    def save_density_vertical_cut(self, lattice: Lattice) -> None:
        """
        Save the density distribution for x == self.Nx//2 of the last step in
            a txt file
        Args:
            lattice (Lattice): lattice
        """
        with open(self.folder_name + "/data/density_v.csv", "a") as file:
            density = lattice.get_density()
            for i in range(0, lattice.Ny, 2):
                if i != 0:
                    print(",", file=file, end="")
                print(density[lattice.Nx//2, i], file=file, end="")
            print("", file=file)

    def save_density_point(self, lattice: Lattice, x: int, y: int) -> None:
        """
        Save the density at a given point
        Args:
            lattice (Lattice): lattice
            x (int): x coordinate of the point
            y (int): y coordinate of the point
        """
        with open(self.folder_name + "/data/density.csv", "a") as file:
            density = lattice.get_density()
            print(density[x][y], file=file)

    def save_velocity_horizontal_cut(self, lattice: Lattice) -> None:
        """
        Save the density distribution for y == self.Ny//2 of the last step in
            a txt file
        Args:
            lattice (Lattice): lattice
        """
        with open(self.folder_name + "/data/velocity_h.csv", "a") as file:
            velocity = lattice.get_velocity()
            for i in range(0, lattice.Nx):
                if i != 0:
                    print(",", file=file, end="")
                vel = np.sqrt(velocity[0, i, lattice.Ny//2]**2 +
                              velocity[1, i, lattice.Ny//2]**2)
                if velocity[1, i, lattice.Ny//2] < 0:
                    vel *= -1
                print(vel, file=file, end="")
            print("", file=file)

    def save_velocity_vertical_cut(self, lattice: Lattice) -> None:
        """
        Save the density distribution for x == self.Nx//2 of the last step in
            a txt file
        Args:
            lattice (Lattice): lattice
        """
        with open(self.folder_name + "/data/velocity_v.csv", "a") as file:
            velocity = lattice.get_velocity()
            for i in range(0, lattice.Ny):
                if i != 0:
                    print(",", file=file, end="")
                vel = np.sqrt(velocity[0, lattice.Nx//2, i]**2 +
                              velocity[1, lattice.Nx//2, i]**2)
                if velocity[0, lattice.Nx//2, i] < 0:
                    vel *= -1
                print(vel, file=file, end="")
            print("", file=file)

    def save_velocity_point(self, lattice: Lattice, x: int, y: int) -> None:
        """
        Save the maximum velocity
        Args:
            lattice (Lattice): lattice
            x (int): x coordinate of the point
            y (int): y coordinate of the point
        """
        with open(self.folder_name + "/data/velocity.csv", "a") as file:
            velocity = lattice.get_velocity()
            velocity_abs = np.sqrt(velocity[0]**2 + velocity[1]**2)
            print(velocity_abs[x][y], file=file)
