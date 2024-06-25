# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 14:25:35 2024

@author: mrkondratyev
"""

import numpy as np
from grid_setup import *


class AdvState:
    def __init__(self, grid):

        # Fluid state at each grid cell including ghost cells
        self.adv = np.zeros(grid.grid_shape, dtype=np.double)

        #boundary marker 
        self.boundMark = np.zeros(4, dtype=np.int32)
        self.boundMark[:] = 300
        
        self.vel1 = 1.0/np.sqrt(2.0)
        self.vel2 = 1.0/np.sqrt(2.0)