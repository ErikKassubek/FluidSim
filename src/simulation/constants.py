'''
File: constants.py
Project: High-Performance Computing: Fluid Mechanics with Python 2023
Author: Erik Kassubek
-----
Copyright (c) 2023 Erik Kassubek
'''

import numpy as np

c = np.transpose(np.array([
            [0, 1, 0, -1, 0, 1, -1, -1, 1],
            [0, 0, 1, 0, -1, 1, 1, -1, -1]]))
w = np.array([4/9, 1/9, 1/9, 1/9, 1/9,
              1/36, 1/36, 1/36, 1/36])
cs2 = 1/3  # speed of sound
