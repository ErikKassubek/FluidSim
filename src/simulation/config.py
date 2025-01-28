'''
File: config.py
Project: High-Performance Computing: Fluid Mechanics with Python 2023
Author: Erik Kassubek
-----
Copyright (c) 2023 Erik Kassubek
'''


class ConfigVis:
    def __init__(self, density: bool = False, flow: bool = False,
                 gif: bool = False, plot_size=(0, 0)) -> None:
        """
        Configuration for the visualization.
        Args:
            density (bool, optional): Plot the density. Defaults to False.
            flow (bool, optional): Plot the flow. Defaults to False.
            gif (bool, optional): Create a gif. Defaults to False.
            plot_size (Tuple[int, int], optional): Size of the plot.
                Defaults to (0,0).
        """
        self.density = density
        self.flow = flow
        self.gif = gif
        self.any = density or flow
        self.plot_size = plot_size


class ConfigData:
    def __init__(self, amplitude: bool = False, cut: bool = False) -> None:
        """
        Configuration for the data.
        Args:
            amplitude (bool, optional): Save the amplitude. Defaults to False.
            cut (bool, optional): Save a cut. Defaults to False.
        """
        self.amplitude = amplitude
        self.cut = cut
        self.point = (0, 0)


class ConfigData2:
    """
    Configuration for the data.
    Args:
        amplitude_dens (bool, optional): Save the density at a point.
            Defaults to False.
        point_dens (Tuple[int, int], optional): Point to save the density at.
            Defaults to None.
        horizontal_dens (bool, optional): Save the horizontal density cut.
            Defaults to False.
        vertical_dens (bool, optional): Save the vertical density cut.
            Defaults to False.
        amplitude_vel (bool, optional): Save the velocity at a point.
            Defaults to False.
        point_vel (Tuple[int, int], optional): Point to save the velocity at.
            Defaults to None.
        horizontal_vel (bool, optional): Save the horizontal velocity cut.
            Defaults to False.
        vertical_vel (bool, optional): Save the vertical velocity cut.
            Defaults to False.
    """

    def __init__(self, amplitude_dens: bool = False,
                 point_dens=None,
                 horizontal_dens: bool = False,
                 vertical_dens: bool = False,
                 amplitude_vel: bool = False,
                 point_vel=None,
                 horizontal_vel: bool = False,
                 vertical_vel: bool = False) -> None:
        self.amplitude_dens = amplitude_dens
        self.point_dens = point_dens
        self.horizontal_dens = horizontal_dens
        self.vertical_dens = vertical_dens
        self.amplitude_vel = amplitude_vel
        self.point_vel = point_vel
        self.horizontal_vel = horizontal_vel
        self.vertical_vel = vertical_vel
        self.any = amplitude_dens or horizontal_dens or vertical_dens or \
            amplitude_vel or horizontal_vel or vertical_vel
