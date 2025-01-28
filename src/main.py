'''
File: main.py
Project: High-Performance Computing: Fluid Mechanics with Python 2023
Author: Erik Kassubek
-----
Copyright (c) 2023 Erik Kassubek
'''

import argparse
import sys
import os

from simulation.modes.collision_operator import CollisionOperator
from simulation.modes.shear_wave_decay import ShearWaveDecay
from simulation.modes.couette_flow import CouetteFlow
from simulation.modes.poiseuille_flow import PoiseuilleFlow
from simulation.modes.sliding_lid_serial import SlidingLidSerial
from simulation.modes.sliding_lid_parallel import SlidingLidParallel
from simulation.config import ConfigVis, ConfigData


def command_line_arguments() -> argparse.Namespace:
    """
    Get command line arguments
    """
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-m2", "--mode2", action="store_true",
                            help="Run the collision operator simulation", dest="collision")
    arg_parser.add_argument("-m3d", "--mode3density", action="store_true",
                            help="Run the shear wave decay simulation with " +
                            "sin in density (mode 3)",
                            dest="shear_wave_density")
    arg_parser.add_argument("-m3v", "--mode3velocity",
                            action="store_true", help="Run the shear wave " +
                            "decay simulation with sin in velocity " +
                            "(mode 3)", dest="shear_wave_velocity")
    arg_parser.add_argument("-m4", "--mode4", action="store_true",
                            help="Run the couette flow simulation " +
                            "(mode 4)", dest="couette")
    arg_parser.add_argument("-m5", "--mode5", action="store_true",
                            help="Run the poiseuille flow simulation " +
                            "(mode 5)", dest="poiseuille")
    arg_parser.add_argument("-m6s", "--mode6serial", action="store_true",
                            help="Run the sliding lid simulation " +
                            "(mode 6)", dest="sliding_s")
    arg_parser.add_argument("-m6p", "--mode6parallel",
                            action="store_true",
                            help="Run the sliding lid simulation " +
                            "(mode 6)", dest="sliding_p")
    arg_parser.add_argument("-x", "--len_x", action="store", type=int,
                            help="Length of the x dimension (default: 100)",
                            default=100, dest="len_x")
    arg_parser.add_argument("-y", "--len_y", action="store", type=int,
                            help="Length of the y dimension (default: 100)",
                            default=100, dest="len_y")
    arg_parser.add_argument("-s", "--steps", help="Number of steps to run"
                            + " (default: 1000)", action="store", type=int,
                            default=1000, dest="steps")
    arg_parser.add_argument("-o", "--omega", help="Relaxation parameter " +
                            "(default: 1)",
                            action="store", type=float, default=1,
                            dest="omega")
    arg_parser.add_argument("-d", "--density", help="Show the " +
                            "density in the plot", action="store_true",
                            dest="density")
    arg_parser.add_argument("-f", "--flow", help="Show the " +
                            "flow in the plot", action="store_true",
                            dest="flow")
    arg_parser.add_argument("-g", "--gif",
                            help="Create a gif animation (only possible if " +
                            "-f or -d is set)",
                            action="store_true", dest="gif")
    arg_parser.add_argument("-a", "--amplitude", action="store_true",
                            dest="amplitude", help="Save the amplitude of " +
                            "the flow for each timestep in a file")
    arg_parser.add_argument("-c", "--cut", action="store_true", dest="cut",
                            help="Save a cut throw the middle of the " +
                            "simulation area for each timestep in a file. " +
                            "The orientation depends on the simulation.")
    arg_parser.add_argument("-t", "--time", action="store_true", dest="time",
                            help="Measure the calculation time")
    arg_parser.add_argument("-v", "--verbose", action="store_true",
                            help="Print the current time step", dest="verbose")

    return arg_parser.parse_args()


def main():
    """
    Main method
    """
    # get command line arguments
    args = command_line_arguments()

    # if no simulation was selected
    if not (args.collision or args.shear_wave_density
            or args.shear_wave_velocity or args.couette or args.poiseuille
            or args.sliding_s or args.sliding_p):
        print("Please specify which simulations to run. Run with -h for help.")
        sys.exit(1)

    # cut is not implemented for parallel sliding lid
    if args.sliding_p and (args.cut or args.amplitude or args.density):
        print("For parallel only the flow can be shown.")
        sys.exit(1)

    # if an invalid number was specified for x, y, or steps
    if args.steps <= 0 or args.len_x < 10 or args.len_y < 10:
        print("Please specify a value greater than 0 for steps, " +
              "and a value greater or equal 10 for x and y.")
        sys.exit(1)

    # gif can only be created if flow or density is shown
    if args.gif and not (args.density or args.flow):
        print("Gif can only be created if flow or density is shown. " +
              "Run with -h for help.")
        sys.exit(1)

    # omega must be between 0 and 2 (not including)
    if args.omega <= 0 or args.omega >= 2:
        print("Omega must be between 0 and 2 (not including). " +
              "Run with -h for help.")
        sys.exit(1)

    # configuration for drawing and data extraction
    conf_vis = ConfigVis(
        density=args.density,
        flow=args.flow,
        gif=args.gif
    )

    conf_data = ConfigData(
        amplitude=args.amplitude,
        cut=args.cut
    )

    # create output folder
    if not os.path.exists("../out"):
        os.mkdir("../out")

    if args.collision:  # run mode 2
        M2 = CollisionOperator(args.len_x, args.len_y, args.omega, args.steps,
                               conf_vis, args.verbose, args.time)
        M2.run()

    if args.shear_wave_density:  # run mode 3 with sin in density
        M3_dens = ShearWaveDecay(args.len_x, args.len_y, args.omega,
                                 args.steps, 1, 0.01, conf_vis, conf_data,
                                 args.verbose, args.time)
        M3_dens.run()

    if args.shear_wave_velocity:  # run mode 3 with sin in velocity
        M3_vel = ShearWaveDecay(args.len_x, args.len_y, args.omega,
                                args.steps, None, 0.05, conf_vis, conf_data,
                                args.verbose, args.time)
        M3_vel.run()

    if args.couette:  # run mode 4 (couette flow)
        M4 = CouetteFlow(args.len_x, args.len_y, args.omega, args.steps,
                         conf_vis, conf_data,
                         args.verbose, args.time, 0.2)
        M4.run()

    if args.poiseuille:  # run mode 5 (poisseuille flow)
        M5 = PoiseuilleFlow(args.len_x, args.len_y, args.omega, args.steps,
                            conf_vis, conf_data, args.verbose, args.time, 0.01)
        M5.run()

    if args.sliding_s:  # run mode 6 serial (serial sliding lid)
        M6_s = SlidingLidSerial(args.len_x, args.len_y, args.omega, args.steps,
                                conf_vis, conf_data, args.verbose, args.time,
                                0.2)
        M6_s.run()

    if args.sliding_p:  # run mode 6 parallel (parallel sliding lid)
        M6_p = SlidingLidParallel(args.len_x, args.len_y, args.omega,
                                  args.steps, conf_vis, conf_data,
                                  args.verbose, args.time, 0.05)
        M6_p.run()


if __name__ == "__main__":
    main()
