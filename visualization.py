# -*- coding: utf-8 -*-
"""
Visualization utilities for simulation output.

This module provides simple routines for visualizing physical quantities
from numerical simulations. It supports both 1D and 2D plots:

- ``plot_setup`` creates and configures the figure for the chosen grid dimension.
- ``plotting`` updates the figure during the simulation runtime.

Both 1D and 2D visualizations are supported, with automatic axis scaling
and color normalization in the 2D case.

Author
------
mrkondratyev
Date: Tue Jun 25 13:21:17 2024
"""

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display, clear_output 


def plot_setup(grid, var, time):
    """
    Initialize a matplotlib figure for visualization.

    Depending on the grid dimensionality (1D in x, 1D in y, or 2D),
    this function sets up either a line plot or a 2D colormap.

    Notes
    -----
    - In 1D cases (Nx2 == 1 or Nx1 == 1), a line plot is produced.
    - In 2D cases, an ``imshow`` plot is created with color limits
      adjusted to the data range.

    Parameters
    ----------
    grid : object
        Grid object containing domain information, cell centers,
        boundaries, and ghost cells.
    var : ndarray
        Variable to be visualized (2D array including ghost cells).
    time : float
        Current physical time for the plot title.

    Returns
    -------
    line : matplotlib Line2D or int
        Line handle for 1D plots. If 2D, returns 0 (unused).
    ax : matplotlib Axes
        Axes object containing the plot.
    fig : matplotlib Figure
        Figure object for the plot.
    im : matplotlib image or int
        Image handle for 2D plots. If 1D, returns 0 (unused).
    """
    Ng = grid.Ngc
    im = 0
    line = 0
    
    if (grid.Nx2 == 1):  # 1D in x
        fig, ax = plt.subplots()
        line, = ax.plot(grid.cx1[Ng:-Ng, Ng], var[Ng:-Ng, Ng])
        ax.set_title('sol at time = ' + str(np.round(time, 4)))
        ax.set_xlabel('x')
        ax.set_ylabel('solution')
        plt.close()  
    
    elif (grid.Nx1 == 1):  # 1D in y
        fig, ax = plt.subplots()
        line, = ax.plot(grid.cx2[Ng, Ng:-Ng], var[Ng, Ng:-Ng])
        ax.set_title('sol at time = ' + str(np.round(time, 4)))
        ax.set_xlabel('y')
        ax.set_ylabel('solution')
        plt.close()  
        
    else:  # 2D case
        fig, ax = plt.subplots()
        varmin = np.min(var[Ng:-Ng, Ng:-Ng])
        varmax = np.max(var[Ng:-Ng, Ng:-Ng])
        
        im = ax.imshow(var[Ng:-Ng, Ng:-Ng].T, 
                       origin='lower',
                       extent=[grid.x1ini, grid.x1fin, grid.x2ini, grid.x2fin],
                       cmap='plasma',
                       vmin=varmin, vmax=varmax)
            
        ax.set_title('solution at time = ' + str(np.round(time, 2)))
        ax.set_xlabel('x1') 
        ax.set_ylabel('x2')
        cbar = plt.colorbar(im, ax=ax)
        plt.ion()
        plt.show()

    return line, ax, fig, im



def plotting(grid, var, time, line, ax, fig, im):
    """
    Update the visualization during the simulation.

    This function updates either a line plot (for 1D grids) or an image
    plot (for 2D grids) with the current variable values. Titles and color
    ranges are refreshed at each call.

    Notes
    -----
    - In 1D mode, ``ax.relim()`` and ``ax.autoscale_view()`` are used
      to rescale the axes automatically.
    - In 2D mode, color limits are updated to the current data range.
    - Russian comment preserved: 
      *"plot the figure"* → function updates the figure in runtime.

    Parameters
    ----------
    grid : object
        Grid object with ghost cells and cell centers.
    var : ndarray
        Variable to plot (2D array including ghost cells).
    time : float
        Current physical time for the plot title.
    line : matplotlib Line2D or int
        Line object returned by ``plot_setup`` for 1D plots.
    ax : matplotlib Axes
        Axes object returned by ``plot_setup``.
    fig : matplotlib Figure
        Figure object returned by ``plot_setup``.
    im : matplotlib image or int
        Image object returned by ``plot_setup`` for 2D plots.

    Returns
    -------
    None
    """
    Ng = grid.Ngc
    
    if (grid.Nx2 == 1):  # 1D in x
        line.set_data(grid.cx1[Ng:-Ng, Ng], var[Ng:-Ng, Ng])
        ax.set_title('solution at time = '+ str(np.round(time, 4)))
        ax.relim()
        ax.autoscale_view()        
                
    elif (grid.Nx1 == 1):  # 1D in y
        line.set_data(grid.cx2[Ng, Ng:-Ng], var[Ng, Ng:-Ng])
        ax.set_title('solution at time = '+ str(np.round(time, 4)))
        ax.relim()
        ax.autoscale_view()
            
    else:  # 2D case
        varmin = np.min(var[Ng:-Ng, Ng:-Ng])
        varmax = np.max(var[Ng:-Ng, Ng:-Ng])
        
        im.set_data(var[Ng:-Ng, Ng:-Ng].T)  
        im.set_clim(vmin=varmin, vmax=varmax)
        ax.set_title('solution at time = '+ str(np.round(time, 4)))
        
    clear_output(wait=True)
    display(fig)
