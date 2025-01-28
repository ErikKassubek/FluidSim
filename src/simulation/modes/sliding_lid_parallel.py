'''
File: sliding_lid_parallel.py
Project: High-Performance Computing: Fluid Mechanics with Python 2023
Author: Erik Kassubek
-----
Copyright (c) 2023 Erik Kassubek
'''

from mpi4py import MPI
import numpy as np
import time

from simulation.lbm import LBM
from simulation.config import ConfigVis, ConfigData, ConfigData2
from simulation.boundaries.bounce_back import BounceBack
from simulation.boundaries.moving_wall import MovingWall
from simulation.boundaries.periodic import Periodic
from simulation.boundaries.boundary import Direction
from simulation.lattice import Lattice


class SlidingLidParallel(LBM):
    """
    Class to run the parallel sliding lid simulation
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
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()

        # get the best grid for the decomposition of the lattice
        # (number of sub grids in the x and y direction)
        self.grid = self.find_grid(self.size)

        self.cartcomm = self.comm.Create_cart(self.grid, periods=(True, True),
                                              reorder=False)

        boundaries = [
            Periodic(Direction.NORTH),
            Periodic(Direction.EAST),
            Periodic(Direction.SOUTH),
            Periodic(Direction.WEST)
        ]

        conf_data = ConfigData2(
            amplitude_vel=data_conf.amplitude,
            vertical_vel=data_conf.cut
        )

        # get the size of the current sub lattice
        self.Nx_local = int(np.ceil(Nx / self.grid[0]))
        self.Ny_local = int(np.ceil(Ny / self.grid[1]))

        self.Nx_total = Nx
        self.Ny_total = Ny

        # correct the size of the sub lattice for the first row and column
        # if the number of sub lattices in that row or col is not a multiple of
        # Nx or Ny
        if self.rank >= self.grid[0] * (self.grid[1] - 1):
            self.Ny_local = Ny - (self.grid[1] - 1) * self.Ny_local
        if (self.rank + 1) % self.grid[0] == 0:
            self.Nx_local = Nx - (self.grid[0] - 1) * self.Nx_local

        # correct the boundary if the sub lattice is at the edge of the lattice
        if self.rank < self.grid[0]:
            boundaries[0] = MovingWall(Direction.NORTH, vel)
        if self.rank % self.grid[0] == 0:
            boundaries[3] = BounceBack(Direction.WEST)
        if self.rank >= self.grid[0] * (self.grid[1] - 1):
            boundaries[2] = BounceBack(Direction.SOUTH)
        if self.rank % self.grid[0] == self.grid[0] - 1:
            boundaries[1] = BounceBack(Direction.EAST)

        # call the constructor of the parent class
        super().__init__(self.Nx_local, self.Ny_local, omega, n, boundaries,
                         True, "sliding_lid_parallel", vis_conf, conf_data,
                         verbose, time, create_output=(self.rank == 0))

    def run(self) -> None:
        """
        Run the simulation.
        """
        self.create_f()
        self.simulate()

    def find_grid(self, n: int):
        """
        Find the best grid for the decomposition of the grid.
        Args:
            n (int): Number of processes.
        Returns:
            tuple: Best grid. (number of rows, number of columns)
        """
        def is_prime(n: int) -> bool:
            """
            Check if n is prime.
            Args:
                n (int): Number to check.
            Returns:
                bool: True if n is prime, False otherwise.
            """
            for i in range(2, int(np.sqrt(n)) + 1):
                if (n % i) == 0:
                    return False
            return True

        solutions = []

        for b in range(2, int(np.sqrt(n)) + 1):
            if n % b == 0:
                solutions.append((n // b, b))

        best_solution = (n, 1)
        best_diff = n - 1
        for solution in solutions:
            diff = abs(solution[0] - solution[1])
            if diff < best_diff:
                best_diff = diff
                best_solution = solution
        return best_solution

    def simulate(self) -> None:
        """
        Simulate the system.
        """
        time_start = time.time()
        length = len(str(self.n))
        for i in range(self.n):
            self.update()
            if self.verbose and self.rank == 0:
                print(f"\r{self.folder}: {i + 1:0{length}d}/{self.n}", end="")

            # if self.rank == 0 and i % 200 == 0:
            #     print("Saving...", i)
            #     self.visualize(str(i).zfill(length))
        if self.verbose and self.rank == 0:
            print()

        if self.rank == 0:
            if self.time:
                time_lattice = time.time() - time_start
                print(f"Total lattice update time: {time_lattice:.2f} s")
                tpla = time_lattice / self.n * 1000
                print(f"Avg. time per lattice update: {tpla:.4f} ms")
                mlups = (self.lattice.Nx * self.lattice.Ny * self.n
                         / time_lattice / 10**6)
                print(f"MLUPS: {mlups:.2f}")

            # self.visualize()

    def update(self) -> None:
        """
        Perform one update step.
        """
        self.streaming()
        self.communication()
        self.handle_boundaries()
        self.collision()
        # if self.rank == 0:
        #     print(self.lattice.f.max())

    def communication(self) -> None:
        """
        Communicate the boundary values.
        """
        shift_left, shift_right = self.cartcomm.Shift(1, 0)
        shift_up, shift_down = self.cartcomm.Shift(0, 1)

        # send to the right, receive from the left
        sendbuf = np.ascontiguousarray(self.lattice.f[:, :, -2])
        recvbuf = np.empty_like(sendbuf)
        self.comm.Sendrecv(sendbuf, dest=shift_right, recvbuf=recvbuf,
                           source=shift_left)
        self.lattice.f[:, :, -1] = recvbuf

        # send to the left, receive from the right
        sendbuf = np.ascontiguousarray(self.lattice.f[:, :, 1])
        recvbuf = np.empty_like(sendbuf)
        self.comm.Sendrecv(sendbuf, dest=shift_left, recvbuf=recvbuf,
                           source=shift_right)
        self.lattice.f[:, :, 0] = recvbuf

        # send up, receive from down
        sendbuf = np.ascontiguousarray(self.lattice.f[:, 1, :])
        recvbuf = np.empty_like(sendbuf)
        self.comm.Sendrecv(sendbuf, dest=shift_up, recvbuf=recvbuf,
                           source=shift_down)
        self.lattice.f[:, 0, :] = recvbuf

        # send down, receive from up
        sendbuf = np.ascontiguousarray(self.lattice.f[:, -2, :])
        recvbuf = np.empty_like(sendbuf)
        self.comm.Sendrecv(sendbuf, dest=shift_down, recvbuf=recvbuf,
                           source=shift_up)
        self.lattice.f[:, -1, :] = recvbuf

    def visualize(self, name: str = "final") -> None:
        """
        Visualize the system.
        Args:
            name (str, optional): Name of the file. Defaults to "final".
        """
        if self.vis_conf.flow:
            if self.rank == 0:
                all_lattices = np.zeros((self.size, 9, self.Nx_local,
                                         self.Ny_local))
                sizes = np.zeros((self.size, 2), int)
            else:
                all_lattices = None
                sizes = None
            f = np.ascontiguousarray(self.lattice.f[:, 1:-1, 1:-1])
            self.comm.Gather(f, all_lattices, root=0)
            self.comm.Gather(np.array([self.Nx_local, self.Ny_local]), sizes,
                             root=0)
            if self.rank == 0:
                grid = self.assemble(all_lattices, sizes)
                lattice = Lattice(self.Nx_total, self.Ny_total, False)
                lattice.set_f(grid)
                self.visualization.dry = False
                self.visualization.visualize(lattice, name, size=(8, 8),
                                             dpi=100)

    def assemble(self, all_lattices: np.ndarray,
                 sizes: np.ndarray) -> np.ndarray:
        """
        Assamble the sub lattices into one lattice.
        Args:
            all_lattices (np.ndarray): All sub lattices.
        """
        lattice = np.zeros((9, self.Nx_total, self.Ny_total))
        for i, lattice_local in enumerate(all_lattices):
            grid = self.cartcomm.Get_coords(i)
            x_start = sizes[:grid[0], 0].sum()
            x_end = x_start + sizes[i, 0]
            y_start = sizes[:grid[1], 1].sum()
            y_end = y_start + sizes[i, 1]
            lattice[:, x_start:x_end, y_start:y_end] = lattice_local
        return lattice
