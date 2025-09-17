"""
===============================================================================
reconstruction.py
===============================================================================

High-order reconstruction routines for finite-volume fluid solvers in 2D.

This module provides a collection of spatial reconstruction methods for
finite-volume schemes used in computational astrophysics and fluid dynamics.
These routines compute the left and right states of a fluid variable at cell
faces, ready for flux evaluation by a Riemann solver. The module supports
uniform Cartesian grids and includes the following reconstruction schemes:

    1. PCM   : Piecewise-constant (1st order, no slope limiting)
    2. PLM   : Piecewise-linear (2nd order, monotonicity-limited)
    3. WENO  : Weighted Essentially Non-Oscillatory (3rd or 5th order)
    4. PPMorig: Standard Piecewise Parabolic Method (3rd order)
    5. PPM   : Fifth-order improved PPM (Mignone 2014)

Key routines
-------------
- rec_PLM   : Limited piecewise-linear reconstruction
- limiter   : Slope limiter for PLM and PPM schemes
- rec_WENO  : WENO/CWENO reconstruction for high-order accuracy
- rec_PPMorig: Standard PPM reconstruction (3rd order)
- rec_PPM5  : Fifth-order PPM reconstruction
- VarReconstruct : Unified interface for selecting reconstruction type

References
----------
- Collela, P., & Woodward, P.R. (1984). The Piecewise Parabolic Method (PPM)
  for Gas-Dynamical Simulations.
- Mignone, A. (2014). High-order conservative reconstruction schemes 
  for finite volume methods in cylindrical and spherical coordinates,, Journal of Computational Physics.
- D.S. Balsara, "Higher-order accurate space-time schemes for computational
  astrophysics—Part I: finite volume methods", Living Rev Comput Astrophys (2017) 3:2.

Author: mrkondratyev 

Notes
-----
- All routines assume the presence of ghost cells.
- Designed for modular integration into finite-volume fluid solvers.
===============================================================================
"""


import numpy as np


def VarReconstruct(var, grid, rec_type, dim):
    """
    #!!!DESCRIPTION OF VarReconstruct!!!
    
    High-order reconstruction of a fluid variable for use in finite-volume schemes.
    This routine reconstructs the state variable at the **faces of each cell** in a desired dimension
    for input to a Riemann solver.
    
    Parameters
    ----------
    var : ndarray
        2D array of the fluid variable including ghost cells.
    grid : object
        Grid class object containing cell centers and face coordinates (e.g., grid.cx1, grid.cx2, grid.fx1, grid.fx2).
    rec_type : str
        Type of reconstruction. Supported options:
        - 'PCM'   : Piecewise-constant (1st order in space)
        - 'PLM'   : Piecewise-linear (2nd order in space)
        - 'WENO'  : Weighted ENO (3rd order for CWENO or 5th order for WENO5)
        - 'PPMorig': Standard PPM (3rd order)
        - 'PPM'   : Fifth-order PPM (Mignone 2014)
    dim : int
        Dimension along which to perform the reconstruction (1 or 2).
    
    Returns
    -------
    var_rec_L : ndarray
        Reconstructed variable at the **left side** of each cell face.
    var_rec_R : ndarray
        Reconstructed variable at the **right side** of each cell face.
    
    Description
    -----------
    This routine provides a unified interface to several reconstruction schemes:
    
    - **PCM**: Copies cell averages directly to the faces (1st order, no slope limiting).
    - **PLM**: Limited linear reconstruction (2nd order), uses `rec_PLM`.
    - **WENO**: High-order ENO/WENO reconstruction (3rd or 5th order), uses `rec_WENO`.
    - **PPMorig**: Standard third-order PPM reconstruction, uses `rec_PPMorig`.
    - **PPM**: Fifth-order improved PPM (Mignone 2014), uses `rec_PPM5`.
    
    The output arrays `var_rec_L` and `var_rec_R` are ready for flux computation in 
    the Riemann solver. The function automatically selects the correct stencil 
    and reconstruction procedure based on `rec_type` and `dim`.
    
    Notes
    -----
    - Works for **2D grids with ghost cells**.
    - Reconstruction type must match supported options; otherwise, an error will occur.
    - Provides a consistent interface for first-, second-, third-, and fifth-order schemes.
    
    References
    ----------
    - Collela, P., & Woodward, P.R. (1984). The Piecewise Parabolic Method (PPM) for Gas-Dynamical Simulations.
    - Mignone, A. (2014). High-order conservative reconstruction schemes for 
      finite volume methods in cylindrical and spherical coordinates, JCP.
    - D.S. Balsara, "Higher-order accurate space-time schemes for 
      computational astrophysics—Part I: finite volume methods", Living Rev Comput Astrophys (2017) 3:2.
    """
    #very simple first order piecewise-constant reconstruction
    #here we just rewrite fluid variables from the cells onto the faces
    if (rec_type == 'PCM'):
        
        if (dim == 1):
            
            var_rec_L = var[grid.Ngc-1 : grid.Nx1r, grid.Ngc : -grid.Ngc]
            var_rec_R = var[grid.Ngc : grid.Nx1r+1, grid.Ngc : -grid.Ngc]
            
        elif (dim == 2):
            
            var_rec_L = var[grid.Ngc : -grid.Ngc, grid.Ngc-1 : grid.Nx2r]
            var_rec_R = var[grid.Ngc : -grid.Ngc, grid.Ngc : grid.Nx2r+1]
    
    
    #second-order piecewise-linear reconstruction, see rec_PLM for more details
    elif (rec_type == 'PLM'):    
        
        
        if (dim == 1):
            
            #piecewise linear limited reconstructed states in 1-dimension 
            var_rec_L, var_rec_R = rec_PLM(grid, var, 1)
        
        elif (dim == 2):
            
            #piecewise linear limited reconstructed states in 2-dimension 
            var_rec_L, var_rec_R = rec_PLM(grid, var, 2)
        
        
    #WENO reconstruction, see rec_WENO function for more details
    elif (rec_type == 'WENO'):
        
        if (dim == 1):
            
            #WENO reconstructed states in 1-dimension 
            var_rec_L, var_rec_R = rec_WENO(grid.Ngc, grid.Nx1r, var, 1)
        
        elif (dim == 2):
            
            #WENO reconstructed states in 2-dimension 
            var_rec_L, var_rec_R = rec_WENO(grid.Ngc, grid.Nx2r, var, 2)
    
    #standard PPM reconstruction, introduced by Collela and Woorward (1984)
    elif (rec_type == 'PPMorig'):
        
        if (dim == 1):
            
            #PPM reconstructed states in 1-dimension 
            var_rec_L, var_rec_R = rec_PPMorig(grid.Ngc, grid.Nx1r, var, 1)
        
        elif (dim == 2):
            
            #PPM reconstructed states in 2-dimension 
            var_rec_L, var_rec_R = rec_PPMorig(grid.Ngc, grid.Nx2r, var, 2)
        
    #fifth-order PPM reconstruction, following Mignone (2014)
    elif (rec_type == 'PPM'):
        
        if (dim == 1):
            
            #PPM reconstructed states in 1-dimension 
            var_rec_L, var_rec_R = rec_PPM5(grid.Ngc, grid.Nx1r, var, 1)
        
        elif (dim == 2):
            
            #PPM reconstructed states in 2-dimension 
            var_rec_L, var_rec_R = rec_PPM5(grid.Ngc, grid.Nx2r, var, 2)    
    
    #in case of erroneous limiter type input throw message and stop the program
    else:
        raise ValueError(f"Unknown rec_type: {rec_type}. Expected one of ['PCM', 'PLM', 'WENO', 'PPMorig', 'PPM'].")
    
    
    #return reconstructed values of some fluid variable on each side of the face
    return var_rec_L, var_rec_R




def rec_PLM(grid, var, dim):
    """
   #!!!DESCRIPTION OF rec_PLM!!!

   Python realization of a limited second-order piecewise linear method (PLM) for finite volume solvers.
   This function works for any system of equations.

   Parameters
   ----------
   grid : GRID class object
       Grid object containing cell-centered coordinates, face coordinates, and ghost cell information.
   var : ndarray
       2D array of the state variable to reconstruct (including ghost cells).
   dim : int
       Dimension along which to perform the reconstruction (1 or 2).

   Returns
   -------
   var_rec_L : ndarray
       Reconstructed variable at the left side of the cell faces.
   var_rec_R : ndarray
       Reconstructed variable at the right side of the cell faces.

   Description
   -----------
   The PLM method reconstructs the solution at cell faces up to the second order in space using linear extensions of cell-averaged values.
   Slopes (gradients) are computed using neighboring cells, and extrapolation is done via a Taylor expansion:
   
       var_rec(face) = var(cell) + (x(face) - x(cell)) * gradient(cell)

   Without limiting, these linear profiles can produce oscillations near discontinuities (Godunov theorem).
   To maintain monotonicity and the TVD property, the reconstruction uses slope limiters based on a smoothness
   indicator R = (var(i+1) - var(i)) / (var(i) - var(i-1)) (van Leer 1974, Sweby 1984).
   (when R >> 1 or R << 1 ==> then we are near the discontinuity, and the slopes should be limited 
   when R < 0 ==> then we are at extremum, and the slopes should be turned off to achieve non-increasing of extremum values)
   The function supports uniform and non-uniform grids and uses the `limiter` function to select among
   various slope-limiting strategies.

   References
   ----------
   - D.S. Balsara, "Higher-order accurate space-time schemes for computational astrophysics—Part I: finite volume methods",
     Living Rev Comput Astrophys (2017) 3:2.
   - van Leer, B. (1974)
   - Sweby, P. K. (1984)
   """
   
    #initializing local variables from the GRID class object to simplify the notation
    Ngc = grid.Ngc 
    Nx1r = grid.Nx1r
    Nx2r = grid.Nx2r
    
    #limiter type, see "limiter" function in this file below
    limtype = 'VL'
    
    #reconstruction along 1-dimension
    if (dim == 1):
        
        # here we calculate left and right gradients 
        #left face 
        grad_L = (var[Ngc-1:Nx1r+1, Ngc:-Ngc] - var[Ngc-2:Nx1r, Ngc:-Ngc]) / \
            (grid.cx1[Ngc-1:Nx1r+1, Ngc:-Ngc] - grid.cx1[Ngc-2:Nx1r, Ngc:-Ngc])  
            
        #right face
        grad_R = (var[Ngc:Nx1r+2, Ngc:-Ngc] - var[Ngc-1:Nx1r+1, Ngc:-Ngc]) / \
            (grid.cx1[Ngc:Nx1r+2, Ngc:-Ngc] - grid.cx1[Ngc-1:Nx1r+1, Ngc:-Ngc])
           
        #limited gradient for all real cells + 1 row from each side from the boundary 
        #limiter function make the PLM profile monotonic 
        lim_grad = limiter(grad_L, grad_R, limtype)
            
        # Left linearly reconstructed state at the face (interpolation from the left cell)
        var_rec_L = var[Ngc-1:Nx1r, Ngc:-Ngc] + \
            (grid.fx1[Ngc:-Ngc, Ngc:-Ngc] - grid.cx1[Ngc-1:-Ngc, Ngc:-Ngc]) * \
            lim_grad[:-1,:]
        
        # Right linearly reconstructed state at the face (interpolation from the right cell)
        var_rec_R = var[Ngc:Nx1r+1, Ngc:-Ngc] + \
            (grid.fx1[Ngc:-Ngc, Ngc:-Ngc] - grid.cx1[Ngc:-Ngc+1, Ngc:-Ngc]) * \
            lim_grad[1:,:]
        
    #reconstruction along 2-dimension
    elif (dim == 2):
            
        # here we calculate left and right gradients 
        #left face for the cell
        grad_L = (var[Ngc:-Ngc, Ngc-1:Nx2r+1] - var[Ngc:-Ngc, Ngc-2:Nx2r]) / \
            (grid.cx2[Ngc:-Ngc, Ngc-1:Nx2r+1] - grid.cx2[Ngc:-Ngc, Ngc-2:Nx2r])
            
        #right face for the cell
        grad_R = (var[Ngc:-Ngc, Ngc:Nx2r+2] - var[Ngc:-Ngc, Ngc-1:Nx2r+1]) / \
            (grid.cx2[Ngc:-Ngc, Ngc:Nx2r+2] - grid.cx2[Ngc:-Ngc, Ngc-1:Nx2r+1])
         
        #limited gradient for all real cells + 1 row from each side from the boundary
        #limiter function make the PLM profile monotonic 
        lim_grad = limiter(grad_L, grad_R, limtype)
            
        # Left linearly reconstructed state at the face (interpolation from the left cell)
        var_rec_L = var[Ngc:-Ngc, Ngc-1:Nx2r] + \
            (grid.fx2[Ngc:-Ngc, Ngc:-Ngc] - grid.cx2[Ngc:-Ngc, Ngc-1:-Ngc]) * \
            lim_grad[:,:-1]
        
        # Right linearly reconstructed state at the face (interpolation from the right cell)
        var_rec_R = var[Ngc:-Ngc, Ngc:Nx2r+1] + \
            (grid.fx2[Ngc:-Ngc, Ngc:-Ngc] - grid.cx2[Ngc:-Ngc, Ngc:-Ngc+1]) * \
            lim_grad[:,1:]

    #return the linearly reconstructed values 
    return var_rec_L, var_rec_R



def limiter(x, y, limiter_type):
    """
    Slope limiter for second-order monotonic piecewise linear reconstruction.
    
    Parameters
    ----------
    x : ndarray
        Left gradient.
    y : ndarray
        Right gradient.
    limiter_type : str
        Limiter type. Options:
        - 'VL'   : van Leer limiter
        - 'MM'   : minmod limiter (most diffusive but robust)
        - 'MC'   : monotonized central limiter (MC)
        - 'KOR'  : Koren third-order limiter (3rd order only for uniform grids)
        - 'PCM'  : first-order piecewise constant scheme
        - 'NO'   : unlimited reconstruction (may produce oscillations)
    
    Returns
    -------
    df : ndarray
        Limited slope to ensure monotonicity.
    
    Description
    -----------
    The limiter function enforces monotonicity on the linear reconstruction in PLM schemes.
    It computes a smoothness ratio R = y / x and adjusts the gradient according to the chosen limiter type:
    
        - R >> 1 or R << 1: near a discontinuity, limit the slope.
        - R < 0: at a local extremum, turn off the slope to prevent new extrema.
    
    Each limiter represents a popular strategy in computational fluid dynamics to control
    oscillations while preserving accuracy.
    
    Notes
    -----
    - VL : van Leer, smooth and monotonic
    - MM : minmod, very diffusive but robust
    - MC : monotonized central, good balance
    - KOR: Koren third-order, accurate on uniform grids
    - PCM: piecewise constant, first-order
    - NO : unlimited, may be unstable (Lax-Wendroff-like)
    """
    
    # Smoothness analyzer
    r = (y + 1e-14) / (x + 1e-14)

    if limiter_type == 'VL':
        #vanLeer limiter
        df = x * (np.abs(r) + r) / (1.0 + np.abs(r))

    elif limiter_type == 'MM':
        # minmod limiter -- the most diffusive one (but the most stable)
        df = 0.5 * x * (1.0 + np.sign(r)) * np.minimum(1.0, np.abs(r))

    elif limiter_type == 'MC':
        #monotonized-central (MC) limiter
        beta = 2.0
        #in other cases beta should be in range [1.0..2.0]
        df = x * np.maximum(0.0, np.minimum( (1.0 + r) / 2.0,  np.minimum(r * beta, beta)))

    elif limiter_type == 'KOR':
        #third-order Koren limiter (third order approximation stands only for unifrom cartesian grids)
        df = x * np.maximum(0.0, np.minimum(2.0 * r, np.minimum(1.0 / 3.0 + 2.0 * r / 3.0, 2.0)))

    elif limiter_type == 'PCM':
        #first order scheme (for tests)
        df = 0.0
        
    elif limiter_type == 'NO':
        #unlimited second-order reconstruction (it can crash)
        df = x #Lax-Wendroff-like scheme
        #df = (x + y) / 2.0 #central difference
        
    else:
        #in case of erroneous limiter type input throw message and stop the program
        raise ValueError(f"Unknown limiter_type: {limiter_type}. Expected one of ['VL', 'MC', 'KOR', 'PCM', 'NO'].")
        
    #return the limited piece-wise linear addition to the piece-wise constant volume-averaged value
    return df




def rec_WENO(Ngc, Nr, var, dim):
    """
    #!!!DESCRIPTION OF rec_WENO!!!

    Python implementation of a method-of-lines WENO (Weighted Essentially Non-Oscillatory) scheme 
    for finite volume solvers with high-order Runge-Kutta time integration.

    Parameters
    ----------
    Ngc : int
        Number of ghost cells in each dimension.
    Nr : int
        Number of real cells in the desired dimension.
    var : ndarray
        2D array of the state variable to reconstruct (including ghost cells).
    dim : int
        Dimension along which to perform the reconstruction (1 or 2).

    Returns
    -------
    var_rec_L : ndarray
        Reconstructed variable at the left side of the cell faces.
    var_rec_R : ndarray
        Reconstructed variable at the right side of the cell faces.

    Description
    -----------
    WENO schemes extend the idea of ENO (Essentially Non-Oscillatory) methods. Unlike PLM,
    which uses piecewise linear reconstructions, WENO uses higher-order polynomial extensions 
    based on several candidate stencils. The smoothness of each stencil is measured via smoothness 
    indicators (IS), which guide the reconstruction:
    
        - If a stencil contains a discontinuity, its IS is large, reducing its weight.
        - Smooth stencils contribute more to the final convex combination of reconstructed values.

    In 1D along the x-direction, for instance, three stencils may be used for cell i:
        - Left:   cells [i-2, i-1, i]
        - Central: cells [i-1, i, i+1]
        - Right:  cells [i, i+1, i+2]

    The final reconstructed value is obtained by a convex combination of the stencils, weighted
    according to their smoothness (WENO) or by choosing the smoothest stencil (ENO). 

    The reconstruction adopts Legendre-polynomial-based expansions:
    
        var_rec = var(cell) + var_x(cell) * x + var_xx(cell) * (x^2 - 1/12)

    where `var_x` and `var_xx` are approximations of the first and second derivatives,
    ensuring that the cell-average is preserved.

    This implementation supports both WENO5 finite-difference (fifth-order in space) and 
    CWENO (central WENO, third-order) reconstructions. Currently, the routine is restricted 
    to **uniform Cartesian grids**.

    References
    ----------
    - D.S. Balsara, "Higher-order accurate space-time schemes for computational astrophysics—Part I: finite volume methods",
      Living Rev Comput Astrophys (2017) 3:2.
    - Shu, C.-W. (1998)
    """
    #choose the dimension of reconsturction
    if (dim == 1):
        
        #first and second derivatives for left stencil reconstructed state
        uxl = -2.0 * var[Ngc-2:Nr, Ngc:-Ngc] + 1.5 * var[Ngc-1:Nr+1, Ngc:-Ngc] + 0.5 * var[Ngc-3:Nr-1, Ngc:-Ngc]
        uxxl = 0.5 * (var[Ngc-3:Nr-1, Ngc:-Ngc] - 2.0 * var[Ngc-2:Nr, Ngc:-Ngc] + var[Ngc-1:Nr+1, Ngc:-Ngc])
        
        #first and second derivatives for central stencil reconstructed state
        uxc = 0.5 * var[Ngc:Nr+2, Ngc:-Ngc] - 0.5 * var[Ngc-2:Nr, Ngc:-Ngc]
        uxxc = 0.5 * (var[Ngc-2:Nr, Ngc:-Ngc] - 2.0 * var[Ngc-1:Nr+1, Ngc:-Ngc] + var[Ngc:Nr+2, Ngc:-Ngc])
        
        #first and second derivatives for right stencil reconstructed state
        uxr = 2.0 * var[Ngc:Nr+2, Ngc:-Ngc] - 1.5 * var[Ngc-1:Nr+1, Ngc:-Ngc] - 0.5 * var[Ngc+1:Nr+3, Ngc:-Ngc]
        uxxr = 0.5 * (var[Ngc-1:Nr+1, Ngc:-Ngc] - 2.0 * var[Ngc:Nr+2, Ngc:-Ngc] + var[Ngc+1:Nr+3, Ngc:-Ngc])
        
    elif (dim == 2):
        
        #first and second derivatives for left stencil reconstructed state
        uxl = -2.0 * var[Ngc:-Ngc, Ngc-2:Nr] + 1.5 * var[Ngc:-Ngc, Ngc-1:Nr+1] + 0.5 * var[Ngc:-Ngc, Ngc-3:Nr-1]
        uxxl = 0.5 * (var[Ngc:-Ngc, Ngc-3:Nr-1] - 2.0 * var[Ngc:-Ngc, Ngc-2:Nr] + var[Ngc:-Ngc, Ngc-1:Nr+1])
        
        #first and second derivatives for central stencil reconstructed state
        uxc = 0.5 * var[Ngc:-Ngc, Ngc:Nr+2] - 0.5 * var[Ngc:-Ngc, Ngc-2:Nr]
        uxxc = 0.5 * (var[Ngc:-Ngc, Ngc-2:Nr] - 2.0 * var[Ngc:-Ngc, Ngc-1:Nr+1] + var[Ngc:-Ngc, Ngc:Nr+2])
        
        #first and second derivatives for right stencil reconstructed state
        uxr = 2.0 * var[Ngc:-Ngc, Ngc:Nr+2] - 1.5 * var[Ngc:-Ngc, Ngc-1:Nr+1] - 0.5 * var[Ngc:-Ngc, Ngc+1:Nr+3]
        uxxr = 0.5 * (var[Ngc:-Ngc, Ngc-1:Nr+1] - 2.0 * var[Ngc:-Ngc, Ngc:Nr+2] + var[Ngc:-Ngc, Ngc+1:Nr+3])
        
        
    #smoothness indicators for left, central and right reconstruction stencils
    ISl = uxl ** 2 + 13.0 / 3.0 * uxxl ** 2
    ISc = uxc ** 2 + 13.0 / 3.0 * uxxc ** 2
    ISr = uxr ** 2 + 13.0 / 3.0 * uxxr ** 2
        
        
    #linear weights and IS degree in the denominator for WENO5 reconstruction
    gammal = 0.1
    gammac = 0.6
    gammar = 0.3
    IS_deg = 2  


    #linear weights and IS degree in the denominator for CWENO reconstruction
    #gammal = 1.0
    #gammac = 200.0
    #gammar = 1.0
    #IS_deg = 4     
        
    #weights without normalization
    wl = gammal / (ISl + 1e-12) ** IS_deg
    wc = gammac / (ISc + 1e-12) ** IS_deg
    wr = gammar / (ISr + 1e-12) ** IS_deg
        
    #normalized weights for WENO reconstruction
    wwl = wl / (wl + wc + wr)
    wwc = wc / (wl + wc + wr)
    wwr = wr / (wl + wc + wr)
        
    #limited first derivative
    ux = wwl * uxl + wwc * uxc + wwr * uxr
    #limited second derivative 
    uxx = wwl * uxxl + wwc * uxxc + wwr * uxxr
    
    
    #final reconstructed states in desired direction
    if (dim == 1):        
        
        #left reconstructed state
        var_rec_L = var[Ngc-1:Nr, Ngc:-Ngc] + ux[:-1,:] * 0.5 + uxx[:-1, :] * ((0.5) ** 2 - 1.0 / 12.0)
        
        #right reconstructed state
        var_rec_R = var[Ngc:Nr+1, Ngc:-Ngc] - ux[1:,:] * 0.5 + uxx[1:, :] * ((-0.5) ** 2 - 1.0 / 12.0)  
    
    
    elif (dim == 2):
 
        #left reconstructed state 
        var_rec_L = var[Ngc:-Ngc, Ngc-1:Nr] + ux[:, :-1] * 0.5 + uxx[:, :-1] * (0.5 ** 2 - 1.0 / 12.0)
        
        #right reconstructed state
        var_rec_R = var[Ngc:-Ngc, Ngc:Nr+1] - ux[:, 1:] * 0.5 + uxx[:, 1:] * ((-0.5) ** 2 - 1.0 / 12.0) 
         
            
    #return final arrays of reconstructed values        
    return var_rec_L, var_rec_R




def rec_PPMorig(Ngc, Nr, var, dim):
    """
   #!!!DESCRIPTION OF rec_PPM!!!

   Python implementation of a standard third-order Piecewise Parabolic Method (PPM) 
   for finite volume solvers, following Collela and Woodward (1984).

   Parameters
   ----------
   Ngc : int
       Number of ghost cells in each dimension.
   Nr : int
       Number of real cells in the desired dimension.
   var : ndarray
       2D array of the state variable to reconstruct (including ghost cells).
   dim : int
       Dimension along which to perform the reconstruction (1 or 2).

   Returns
   -------
   var_rec_L : ndarray
       Reconstructed variable at the left side of the cell faces.
   var_rec_R : ndarray
       Reconstructed variable at the right side of the cell faces.

   Description
   -----------
   PPM extends PLM by constructing a **parabolic profile** within each cell 
   instead of a linear one. The reconstruction steps are:

   1. Compute limited differences between neighboring cells to estimate slopes.
   2. Construct preliminary face values using the parabolic profile.
   3. Enforce monotonicity by checking if face values lie within physically allowed limits.
   4. Regulate the curvature of the parabola to avoid introducing new extrema inside the cell.

   The final reconstructed values use Legendre-polynomial expansions:
   
       var_rec = var(cell) + var_x(cell)*x + var_xx(cell)*(x^2 - 1/12)

   where `var_x` and `var_xx` approximate first and second derivatives, 
   ensuring that the cell-average is preserved.

   Notes
   -----
   - Works only for **uniform Cartesian grids**.
   - Uses standard limiters (e.g., van Leer or MC) to maintain monotonicity.

   References
   ----------
   - Collela, P., & Woodward, P.R. (1984). The Piecewise Parabolic Method (PPM) for Gas-Dynamical Simulations.
   - D.S. Balsara, "Higher-order accurate space-time schemes for computational astrophysics—Part I: finite volume methods", 
     Living Rev Comput Astrophys (2017) 3:2.
   """
    
    #choose the dimension of reconsturction
    if (dim == 1):
        
        #limited differences
        deltaU = limiter(var[Ngc-1:Nr+3, Ngc:-Ngc] - var[Ngc-2:Nr+2, Ngc:-Ngc], var[Ngc-2:Nr+2, Ngc:-Ngc] - var[Ngc-3:Nr+1, Ngc:-Ngc], 'VL')
        
        #reconstruction with PPM method: step 1, here we derive the reconstructed profile for all real faces + 1 ghost face on each side 
        fvar0 = var[Ngc-2:Nr+1, Ngc:-Ngc] + 0.5 * (var[Ngc-1:Nr+2, Ngc:-Ngc] - var[Ngc-2:Nr+1, Ngc:-Ngc]) - (deltaU[1:,:] - deltaU[:-1,:]) / 6.0
          
        #reconstruction with PPM method: step 2, check if we have face value outside the allowed interval
        fvar0_L = np.where( (fvar0[1:,:] - var[Ngc-1:Nr+1, Ngc:-Ngc]) * (var[Ngc-1:Nr+1, Ngc:-Ngc] - fvar0[:-1,:]) < 0.0, \
            var[Ngc-1:Nr+1, Ngc:-Ngc], fvar0[:-1,:])
        
        fvar0_R = np.where( (fvar0[1:,:] - var[Ngc-1:Nr+1, Ngc:-Ngc]) * (var[Ngc-1:Nr+1, Ngc:-Ngc] - fvar0[:-1,:]) < 0.0, \
            var[Ngc-1:Nr+1, Ngc:-Ngc], fvar0[1:,:])
            
        #reconstruction with PPM method: step 3, regulate the curvature to exclude extrema inside the cell
        var_rec_L = np.where( (fvar0_R - fvar0_L) * (var[Ngc-1:Nr+1, Ngc:-Ngc] - 0.5 * (fvar0_R + fvar0_L)) > (fvar0_R - fvar0_L) ** 2 / 6.0, \
            3.0 * var[Ngc-1:Nr+1, Ngc:-Ngc] - 2.0 * fvar0_R, fvar0_L)
        
        var_rec_R = np.where( (fvar0_R - fvar0_L) * (var[Ngc-1:Nr+1, Ngc:-Ngc] - 0.5 * (fvar0_R + fvar0_L)) < -(fvar0_R - fvar0_L) ** 2 / 6.0, \
            3.0 * var[Ngc-1:Nr+1, Ngc:-Ngc] - 2.0 * fvar0_L, fvar0_R)
           
        #final coefficients for Legendre polynomial
        ux = var_rec_R - var_rec_L
        uxx = 3.0 * var_rec_R - 6.0 * var[Ngc-1:Nr+1, Ngc:-Ngc] + 3.0 * var_rec_L
        
        #left reconstructed state
        var_rec_L = var[Ngc-1:Nr, Ngc:-Ngc] + ux[:-1,:] * 0.5 + uxx[:-1, :] * ((0.5) ** 2 - 1.0 / 12.0)
        
        #right reconstructed state
        var_rec_R = var[Ngc:Nr+1, Ngc:-Ngc] - ux[1:,:] * 0.5 + uxx[1:, :] * ((-0.5) ** 2 - 1.0 / 12.0)
            
    elif (dim == 2):
        
        #limited differences
        deltaU = limiter(var[Ngc:-Ngc, Ngc-1:Nr+3] - var[Ngc:-Ngc, Ngc-2:Nr+2], var[Ngc:-Ngc, Ngc-2:Nr+2] - var[Ngc:-Ngc, Ngc-3:Nr+1], 'MC')
        
        #reconstruction with PPM method: step 1, here we derive the reconstructed profile for all real faces + 1 ghost face on each side       
        fvar0 = var[Ngc:-Ngc, Ngc-2:Nr+1] + 0.5 * (var[Ngc:-Ngc, Ngc-1:Nr+2] - var[Ngc:-Ngc, Ngc-2:Nr+1]) - (deltaU[:,1:] - deltaU[:,:-1]) / 6.0
        
        #reconstruction with PPM method: step 2, check if we have face value outside the allowed interval
        fvar0_L = np.where( (fvar0[:,1:] - var[Ngc:-Ngc, Ngc-1:Nr+1]) * (var[Ngc:-Ngc, Ngc-1:Nr+1] - fvar0[:,:-1]) < 0.0, \
            var[Ngc:-Ngc, Ngc-1:Nr+1], fvar0[:,:-1])
            
        fvar0_R = np.where( (fvar0[:,1:] - var[Ngc:-Ngc, Ngc-1:Nr+1]) * (var[Ngc:-Ngc, Ngc-1:Nr+1] - fvar0[:,:-1]) < 0.0, \
            var[Ngc:-Ngc, Ngc-1:Nr+1], fvar0[:,1:])

        #reconstruction with PPM method: step 3, regulate the curvature to exclude extrema inside the cell
        var_rec_L = np.where( (fvar0_R - fvar0_L) * (var[Ngc:-Ngc, Ngc-1:Nr+1] - 0.5 * (fvar0_R + fvar0_L)) > (fvar0_R - fvar0_L) ** 2 / 6.0, \
            3.0 * var[Ngc:-Ngc, Ngc-1:Nr+1] - 2.0 * fvar0_R, fvar0_L)
            
        var_rec_R = np.where( (fvar0_R - fvar0_L) * (var[Ngc:-Ngc, Ngc-1:Nr+1] - 0.5 * (fvar0_R + fvar0_L)) < -(fvar0_R - fvar0_L) ** 2 / 6.0, \
            3.0 * var[Ngc:-Ngc, Ngc-1:Nr+1] - 2.0 * fvar0_L, fvar0_R)
        
        #final coefficients for Legendre polynomial
        ux = var_rec_R - var_rec_L
        uxx = 3.0 * var_rec_R - 6.0 * var[Ngc:-Ngc, Ngc-1:Nr+1] + 3.0 * var_rec_L
            
        #left reconstructed state 
        var_rec_L = var[Ngc:-Ngc, Ngc-1:Nr] + ux[:, :-1] * 0.5 + uxx[:, :-1] * (0.5 ** 2 - 1.0 / 12.0)
        
        #right reconstructed state
        var_rec_R = var[Ngc:-Ngc, Ngc:Nr+1] - ux[:, 1:] * 0.5 + uxx[:, 1:] * ((-0.5) ** 2 - 1.0 / 12.0) 
        
    #return final arrays of reconstructed values        
    return var_rec_L, var_rec_R




def rec_PPM5(Ngc, Nr, var, dim):
    """
    #!!!DESCRIPTION OF rec_PPM5!!!

    Python implementation of an improved PPM reconstruction achieving **fifth-order spatial accuracy**, 
    following A. Mignone (JCP, 2014).

    Parameters
    ----------
    Ngc : int
        Number of ghost cells in each dimension.
    Nr : int
        Number of real cells in the desired dimension.
    var : ndarray
        2D array of the state variable to reconstruct (including ghost cells).
    dim : int
        Dimension along which to perform the reconstruction (1 or 2).

    Returns
    -------
    var_rec_L : ndarray
        Reconstructed variable at the left side of the cell faces.
    var_rec_R : ndarray
        Reconstructed variable at the right side of the cell faces.

    Description
    -----------
    This PPM variant improves on the standard third-order method by using a **five-point stencil** 
    to achieve fifth-order accuracy. The reconstruction procedure is:

    1. Compute preliminary left and right face values using weighted combinations of five neighboring cells.
    2. Apply **monotonicity constraints** to ensure that face values remain within the range of neighboring cell averages.
    3. Adjust the reconstructed states depending on local extrema and slope ratios.
    4. Compute final face values and preserve the cell average.

    Notes
    -----
    - Works only for **uniform Cartesian grids**.
    - Implements a more accurate parabolic reconstruction for smooth regions while avoiding spurious oscillations near discontinuities.

    References
    ----------
    - Mignone, A. (2014). "High-order conservative finite difference schemes for astrophysical flows," 
      Journal of Computational Physics.
    - Collela, P., & Woodward, P.R. (1984). The Piecewise Parabolic Method (PPM) for Gas-Dynamical Simulations.
    """
    PPM5c = np.zeros(5)
    PPM5c[0] = 2.0 / 60.0
    PPM5c[1] = - 13.0 / 60.0
    PPM5c[2] = 47.0 / 60.0
    PPM5c[3] = 27.0 / 60.0
    PPM5c[4] = - 3.0 / 60.0
    
    #choose the dimension of reconsturction
    if (dim == 1):
        
        var_L = var[Ngc-3:Nr-1, Ngc:-Ngc] * PPM5c[4] + var[Ngc-2:Nr, Ngc:-Ngc] * PPM5c[3] + \
            var[Ngc-1:Nr+1, Ngc:-Ngc] * PPM5c[2] + var[Ngc:Nr+2, Ngc:-Ngc] * PPM5c[1] + \
            var[Ngc+1:Nr+3, Ngc:-Ngc] * PPM5c[0]
        var_R = var[Ngc-3:Nr-1, Ngc:-Ngc] * PPM5c[0] + var[Ngc-2:Nr, Ngc:-Ngc] * PPM5c[1] + \
            var[Ngc-1:Nr+1, Ngc:-Ngc] * PPM5c[2] + var[Ngc:Nr+2, Ngc:-Ngc] * PPM5c[3] + \
            var[Ngc+1:Nr+3, Ngc:-Ngc] * PPM5c[4]
        
        var_L = np.minimum(var_L, np.maximum(var[Ngc-2:Nr, Ngc:-Ngc], var[Ngc-1:Nr+1, Ngc:-Ngc]))
        var_R = np.minimum(var_R, np.maximum(var[Ngc-1:Nr+1, Ngc:-Ngc], var[Ngc:Nr+2, Ngc:-Ngc]))
    
        var_L = np.maximum(var_L, np.minimum(var[Ngc-2:Nr, Ngc:-Ngc], var[Ngc-1:Nr+1, Ngc:-Ngc]))
        var_R = np.maximum(var_R, np.minimum(var[Ngc-1:Nr+1, Ngc:-Ngc], var[Ngc:Nr+2, Ngc:-Ngc]))
    
        dvar_R = var_R - var[Ngc-1:Nr+1, Ngc:-Ngc]
        dvar_L = var_L - var[Ngc-1:Nr+1, Ngc:-Ngc]
        
        
        var_rec_L = np.where((dvar_R * dvar_L >= 0.0), var[Ngc-1:Nr+1, Ngc:-Ngc], \
            np.where((np.abs(dvar_L) >= 2.0 * np.abs(dvar_R)) & (dvar_R * dvar_L < 0.0), var[Ngc-1:Nr+1, Ngc:-Ngc] - 2.0 * dvar_R, \
            var[Ngc-1:Nr+1, Ngc:-Ngc] + dvar_L))            
        
        var_rec_R = np.where((dvar_R * dvar_L >= 0.0), var[Ngc-1:Nr+1, Ngc:-Ngc], \
            np.where((np.abs(dvar_R) >= 2.0 * np.abs(dvar_L)) & (dvar_R * dvar_L < 0.0), var[Ngc-1:Nr+1, Ngc:-Ngc] - 2.0 * dvar_L, \
            var[Ngc-1:Nr+1, Ngc:-Ngc] + dvar_R))  
            
        #dQ = var_rec_R - var_rec_L
        #Q6 = 6.0 * var[Ngc-1:Nr+1, Ngc:-Ngc] - 3.0 * (var_rec_R + var_rec_L)
        
        var_rec_L, var_rec_R = var_rec_R[:-1,:], var_rec_L[1:,:]
            
    elif (dim == 2):
        
        var_L = var[Ngc:-Ngc, Ngc-3:Nr-1] * PPM5c[4] + var[Ngc:-Ngc, Ngc-2:Nr] * PPM5c[3] + \
            var[Ngc:-Ngc, Ngc-1:Nr+1] * PPM5c[2] + var[Ngc:-Ngc, Ngc:Nr+2] * PPM5c[1] + \
            var[Ngc:-Ngc, Ngc+1:Nr+3] * PPM5c[0]
        var_R = var[Ngc:-Ngc, Ngc-3:Nr-1] * PPM5c[0] + var[Ngc:-Ngc, Ngc-2:Nr] * PPM5c[1] + \
            var[Ngc:-Ngc, Ngc-1:Nr+1] * PPM5c[2] + var[Ngc:-Ngc, Ngc:Nr+2] * PPM5c[3] + \
            var[Ngc:-Ngc, Ngc+1:Nr+3] * PPM5c[4]
        
        
        var_L = np.minimum(var_L, np.maximum(var[Ngc:-Ngc, Ngc-2:Nr], var[Ngc:-Ngc, Ngc-1:Nr+1]))
        var_R = np.minimum(var_R, np.maximum(var[Ngc:-Ngc, Ngc-1:Nr+1], var[Ngc:-Ngc, Ngc:Nr+2]))
    
        var_L = np.maximum(var_L, np.minimum(var[Ngc:-Ngc, Ngc-2:Nr], var[Ngc:-Ngc, Ngc-1:Nr+1]))
        var_R = np.maximum(var_R, np.minimum(var[Ngc:-Ngc, Ngc-1:Nr+1], var[Ngc:-Ngc, Ngc:Nr+2]))
    
        dvar_R = var_R - var[Ngc:-Ngc, Ngc-1:Nr+1]
        dvar_L = var_L - var[Ngc:-Ngc, Ngc-1:Nr+1]
        
        
        var_rec_L = np.where((dvar_R * dvar_L >= 0.0), var[Ngc:-Ngc, Ngc-1:Nr+1], \
            np.where((np.abs(dvar_L) >= 2.0 * np.abs(dvar_R)) & (dvar_R * dvar_L < 0.0), var[Ngc:-Ngc, Ngc-1:Nr+1] - 2.0 * dvar_R, \
            var[Ngc:-Ngc, Ngc-1:Nr+1] + dvar_L))            
        
        var_rec_R = np.where((dvar_R * dvar_L >= 0.0), var[Ngc:-Ngc, Ngc-1:Nr+1], \
            np.where((np.abs(dvar_R) >= 2.0 * np.abs(dvar_L)) & (dvar_R * dvar_L < 0.0), var[Ngc:-Ngc, Ngc-1:Nr+1] - 2.0 * dvar_L, \
            var[Ngc:-Ngc, Ngc-1:Nr+1] + dvar_R))  
        
        #dQ = var_rec_R - var_rec_L
        #Q6 = 6.0 * var[Ngc:-Ngc, Ngc-1:Nr+1] - 3.0 * (var_rec_R + var_rec_L)
            
        var_rec_L, var_rec_R = var_rec_R[:, :-1], var_rec_L[:, 1:]
        
    #return final arrays of reconstructed values        
    return var_rec_L, var_rec_R


