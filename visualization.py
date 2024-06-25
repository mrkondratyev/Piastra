# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 13:21:17 2024

@author: mrkondratyev
"""

import matplotlib.pyplot as plt
import numpy as np


def visual(grid, var):
    Ngc = grid.Ngc
    rhomin = np.min(var[Ngc:-Ngc, Ngc:-Ngc])
    rhomax = np.max(var[Ngc:-Ngc, Ngc:-Ngc])
    plt.cla()
    
    if grid.Nx2 == 1:
        # 1D plot along x1 axis
        plt.plot(grid.cx1[Ngc:-Ngc, Ngc], var[Ngc:-Ngc, Ngc])
        plt.xlabel('x1')
        plt.ylabel('var')
    elif grid.Nx1 == 1:
        # 1D plot along x2 axis
        plt.plot(grid.cx2[Ngc, Ngc:-Ngc], var[Ngc, Ngc:-Ngc])
        plt.xlabel('x2')
        plt.ylabel('var')
    else:
        # 2D plot
        plt.imshow(var[Ngc:-Ngc, Ngc:-Ngc], cmap='jet')
        plt.clim(rhomin, rhomax)
        ax = plt.gca()
        ax.invert_yaxis()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_aspect('equal')
    
        plt.clim(rhomin, rhomax)
        
    plt.pause(0.03)