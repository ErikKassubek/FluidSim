'''
File: visualization.py
Project: High-Performance Computing: Fluid Mechanics with Python 2023
Author: Erik Kassubek
-----
Copyright (c) 2023 Erik Kassubek
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib
import imageio
import os
import glob

from simulation.config import ConfigVis
from simulation.lattice import Lattice

matplotlib.use('agg')  # needed to prevent memory leak in visualization


class Visualization:
    def __init__(self, Nx: int, Ny: int, folder: str, dry: bool,
                 vis_conf: ConfigVis, create_output: bool) -> None:
        """
        Constructor
        Args:
            folder (str): name of the folder to save the plots in
            dry (bool): true if f contains dry nodes
            vis_conf (ConfigVis): visualization configuration
            create_output (bool): whether to create the output folder
        """
        self.folder_name = "../out/" + folder
        self.gif_name = self.folder_name + "/" + folder + ".gif"
        self.dry = dry
        self.conf = vis_conf

        # create folder
        if vis_conf.any and create_output:
            os.mkdir("../out/" + folder + "/img")

    def visualize(self, lattice: Lattice, save_name: str, size=(0, 0),
                  dpi: int = 100) -> None:
        """
        Visualize the density and velocity distributions of f
        Args:
            lattice (Lattice): lattice
            save_name (str): name of the file to save
            size (tuple): size of the plot
            dpi (int): dpi of the plot
        """
        # do not save anything if nothing is plotted
        if not (self.conf.density or self.conf.flow):
            return

        sizeX = lattice.Nx - 2 if lattice.dry else lattice.Nx
        sizeY = lattice.Ny - 2 if lattice.dry else lattice.Ny

        if self.conf.plot_size == (0, 0):
            fig, ax = plt.subplots(dpi=100)
        else:
            fig, ax = plt.subplots(figsize=self.conf.plot_size, dpi=dpi)
        rho = lattice.get_density()
        v = lattice.get_velocity()
        x, y = np.meshgrid(np.arange(sizeX), np.arange(sizeY))

        if self.dry:
            dx, dy = v[0, 1:-1, 1:-1], v[1, 1:-1, 1:-1]
            rho = rho[1:-1, 1:-1]
            v = v[:, 1:-1, 1:-1]
        dx, dy = v[0, :, :], v[1, :, :]

        ax.set_xlim(-0.5, sizeX - 0.5)
        ax.set_ylim(-0.5, sizeY - 0.5)
        if self.conf.density:
            plt.imshow(rho.transpose(), cmap='plasma', origin="lower")
        if self.conf.flow:
            v_abs = np.sqrt(v[0] ** 2 + v[1] ** 2)
            norm = colors.Normalize(vmin=v_abs.min(), vmax=v_abs.max())
            color = norm(v_abs.transpose())
            try:  # not sure why the first 1 or 2 images fail sometime
                plt.streamplot(x, y, dx.transpose(), dy.transpose(), density=2.5,
                               linewidth=1, color=color)
            except:
                pass

        name = self.folder_name + "/img/" + save_name + ".png"
        plt.savefig(name, bbox_inches='tight', transparent=False)
        plt.close()

    def create_gif(self) -> None:
        """
        Create a gif from the png files
        """
        images = []
        filenames = sorted(glob.glob(self.folder_name + "/img/*.png"))
        for filename in filenames:
            images.append(imageio.imread(filename))
        imageio.mimsave(self.gif_name, images)
