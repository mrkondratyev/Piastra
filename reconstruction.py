import numpy as np
import sys
import numba


"""
high order reconstruction for fluid variables for the usage in finite-volume schemes
procedure is done in a desired dimension for a fluid state variable
input: 
    1) 'var' -- a fluid variable (2D array including ghost cells) 
    2) 'grid' -- grid class object (multiple 2d arrays of cell centers and face coordinates)
    3) 'rec_type' ('PCM', 'PLM' or 'WENO' currently) -- reconstruction type --
        PCM -- piece-wise constant reconstruction (1st order in space) 
        PLM -- piece-wise linear reconstrcution (2nd order in space)
        WENO -- WENO reconstruction (3rd (CWENO) or 5th (WENO5) order in space)
    4) 'dim' -- dimension (1 or 2) 
output:
    var_rec_L, var_rec_R -- reconstructed state variable on the faces of the cells,
        further it should be sended to the Riemann solver among the other fluid state variables
"""
def VarReconstruct(var, grid, rec_type, dim):
    
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
    if (rec_type == 'PLM'):    
        
        
        if (dim == 1):
            
            #piecewise linear limited reconstructed states in 1-dimension 
            var_rec_L, var_rec_R = rec_PLM(grid, var, 1)
        
        elif (dim == 2):
            
            #piecewise linear limited reconstructed states in 2-dimension 
            var_rec_L, var_rec_R = rec_PLM(grid, var, 2)
        
        
    #WENO reconstruction, see rec_WENO function for more details
    if (rec_type == 'WENO'):
        
        if (dim == 1):
            
            #WENO reconstructed states in 1-dimension 
            var_rec_L, var_rec_R = rec_WENO(grid.Ngc, grid.Nx1r, var, 1)
        
        elif (dim == 2):
            
            #WENO reconstructed states in 2-dimension 
            var_rec_L, var_rec_R = rec_WENO(grid.Ngc, grid.Nx2r, var, 2)
    
    
    #return reconstructed values of some fluid variable on each side of the face
    return var_rec_L, var_rec_R



"""
#!!!DESCRIPTION OF rec_PLM!!! 
the function "rec_PLM" below is a python realization of limited second-order piecewise linear method (PLM) for finite volume solvers
it can be done for any system of equations, the input are GRID class object, 2D array of state variable "var" and 
integer "dim" = 1 or 2, which switches between the dimension of reconsturction.
The output is a 2D array with reconstructed state variable at left and right sides of the cell faces 
on the faces of the 2D grid in 1- or 2- dimensions, depending on parameter dim. It applies the limiter function "def limiter"
with several popular slope limiters, that can be adjusted inside the rec_PLM function

The idea of PLM is pretty simple -- instead of using piecewise constant values (just the cell averages)
one can you the piecewise linear extensions in each dimension, using the information from neighbouring cells
to construct the slopes (or the gradients). Further we use the linear function to extrapolate the solution in the cell face by using Taylor expansion.
Unfortunately, the reconstructed profiles can suffer from oscillations (Godunov (1959) theorem)
near the discontinuities of the flow, because the solution's total variation can rise (for linear problems, 
the non-increasing property of total variation of some function corresponds to monotonicity).
To make the scheme monotonic or TVD ("total variation diminishing", as is satisfied for the exact solution of, for instance, 
gas dynamics equations), one has to bound the reconstructed profiles in order to make them monotonic again.
This limiting procedure is done by applying some sort of "limiter" function, which depends on the solution itself.
It is done with monotonicity/smoothness analyzer (van Leer (1974)), which has the form of R = (var(i+1)-var(i))/(var(i)-var(i-1)).
when R >> 1 or R << 1 ==> then we are near the discontinuity, and the slopes should be limited 
when R < 0 ==> then we are at extremum, and the slopes should be turned off to achieve non-increasing of extremum values.
(see P Sweby (1984))

everywhere below the reconstructed value looks like extrapolation in the face from the center of the cell
var_rec(face) = var(cell) + (x(face) - x(cell))*gradient(cell), where for the gradient the limiting is applied.


########################################
further reading -- see e.g. -- D.S. Balsara "Higher-order accurate space-time schemes
for computational astrophysics—Part I: finite volume
methods", Living Rev Comput Astrophys (2017) 3:2
########################################

PLM routine implemented here works for uniform and non-uniform grids
limiter type for PLM scheme can also be adjusted in this file (see function 'limiter' below in this file)
"""
def rec_PLM(grid, var, dim):
    
    #initializing local variables from the GRID class object to simplify the notation
    Ngc = grid.Ngc 
    Nx1r = grid.Nx1r
    Nx2r = grid.Nx2r
    
    #limiter type, see "limiter" function in this file below
    limtype = 3
    
    #reconstruction along 1-dimension
    if (dim == 1):
            
        # here we calculate three gradients (slopes)
            
        #left face 
        grad_L = (var[Ngc-1:Nx1r, Ngc:-Ngc] - var[Ngc-2:Nx1r-1, Ngc:-Ngc]) / \
            (grid.cx1[Ngc-1:Nx1r, Ngc:-Ngc] - grid.cx1[Ngc-2:Nx1r-1, Ngc:-Ngc])  
            
        #face where we look for the reconstructed states
        grad_C = (var[Ngc:Nx1r+1, Ngc:-Ngc] - var[Ngc-1:Nx1r,Ngc:-Ngc]) / \
            (grid.cx1[Ngc:Nx1r+1, Ngc:-Ngc] - grid.cx1[Ngc-1:Nx1r,Ngc:-Ngc])
            
        #right face
        grad_R = (var[Ngc+1:Nx1r+2, Ngc:-Ngc] - var[Ngc:Nx1r+1, Ngc:-Ngc]) / \
            (grid.cx1[Ngc+1:Nx1r+2, Ngc:-Ngc] - grid.cx1[Ngc:Nx1r+1, Ngc:-Ngc])
            
            
        # Left linearly reconstructed state + apply the limiter to achieve monotonicity
        var_rec_L = var[Ngc-1:Nx1r, Ngc:-Ngc] + \
            (grid.fx1[Ngc:-Ngc, Ngc:-Ngc] - grid.cx1[Ngc-1:-Ngc, Ngc:-Ngc]) * \
            limiter(grad_L, grad_C, limtype)
        
        # Right linearly reconstructed state + apply the limiter to achieve monotonicity
        var_rec_R = var[Ngc:Nx1r+1, Ngc:-Ngc] + \
            (grid.fx1[Ngc:-Ngc, Ngc:-Ngc] - grid.cx1[Ngc:-Ngc+1, Ngc:-Ngc]) * \
            limiter(grad_C, grad_R, limtype)
        
    #reconstruction along 2-dimension
    elif (dim == 2):
            
        # here we calculate three gradients (slopes)
            
        #left face 
        grad_L = (var[Ngc:-Ngc, Ngc-1:Nx2r] - var[Ngc:-Ngc, Ngc-2:Nx2r-1]) / \
            (grid.cx2[Ngc:-Ngc, Ngc-1:Nx2r] - grid.cx2[Ngc:-Ngc, Ngc-2:Nx2r-1])
            
        #face where we look for the reconstructed states
        grad_C = (var[Ngc:-Ngc, Ngc:Nx2r+1] - var[Ngc:-Ngc, Ngc-1:Nx2r]) / \
            (grid.cx2[Ngc:-Ngc, Ngc:Nx2r+1] - grid.cx2[Ngc:-Ngc, Ngc-1:Nx2r])
            
        #right face 
        grad_R = (var[Ngc:-Ngc, Ngc+1:Nx2r+2] - var[Ngc:-Ngc,Ngc:Nx2r+1]) / \
            (grid.cx2[Ngc:-Ngc, Ngc+1:Nx2r+2] - grid.cx2[Ngc:-Ngc,Ngc:Nx2r+1])
            
            
        # Left linearly reconstructed state + apply the limiter to achieve monotonicity
        var_rec_L = var[Ngc:-Ngc, Ngc-1:Nx2r] + \
            (grid.fx2[Ngc:-Ngc, Ngc:-Ngc] - grid.cx2[Ngc:-Ngc, Ngc-1:-Ngc]) * \
            limiter(grad_L, grad_C, limtype)
        
        # Right linearly reconstructed state + apply the limiter to achieve monotonicity
        var_rec_R = var[Ngc:-Ngc, Ngc:Nx2r+1] + \
            (grid.fx2[Ngc:-Ngc, Ngc:-Ngc] - grid.cx2[Ngc:-Ngc, Ngc:-Ngc+1]) * \
            limiter(grad_C, grad_R, limtype)


    #return the linearly reconstructed values 
    return var_rec_L, var_rec_R





"""
limiter function for second-order monotonic piecewise linear reconstruction
it offers 5 possible options -- 
1 - van Leer limiter
2 - minmod limiter - most diffusive one, but no oscillations observed for it on any problem, including MHD
3 - MC limiter - also a good option, a bit better, than van Leer
4 - Koren third-order limiter (third order of approximation only on uniform grid)
0 - no limiter - simply first order piecewise-constant scheme
"""
def limiter(x, y, limiter_type):
    
    # Smoothness analyzer
    r = (y + 1e-14) / (x + 1e-14)

    if limiter_type == 1:
        #vanLeer limiter
        df = x * (np.abs(r) + r) / (1.0 + np.abs(r))

    elif limiter_type == 2:
        # minmod limiter -- the most diffusive one (but the most stable)
        df = 0.5 * x * (1.0 + np.sign(r)) * np.minimum(1.0, np.abs(r))

    elif limiter_type == 3:
        #monotonized-central (MC) limiter
        beta = 2.0
        #in other cases beta should be in range [1.0..2.0]
        
        df = x * np.maximum(0.0, np.minimum( (1.0 + r) / 2.0,  np.minimum(r * beta, beta)))

    elif limiter_type == 4:
        #third-order Koren limiter (third order approximation stands only for unifrom cartesian grids)
        df = x * np.maximum(0.0, np.minimum(2.0 * r, np.minimum(1.0 / 3.0 + 2.0 * r / 3.0, 2.0)))

    elif limiter_type == 0:
        #first order scheme (for tests)
        df = 0.0
        
    else:
        #in case of erroneous limiter type input throw message and stop the program
        sys.exit("error, the slope limiter is undefined, see def 'limiter' in 'reconstruction.py'")
        
    #return the limited piece-wise linear addition to the piece-wise constant volume-averaged value
    return df





"""
#!!!DESCRIPTION OF rec_WENO!!! 
the function "rec_WENO" below is a python realization of method-of-lines WENO scheme for 
finite volume solvers with Runge-Kutta high-order timestepping
it can be done for any system of equations, the input are the number of ghost zones NGC, number of cells in desired dimension Nr + NGC, 
2D array of state variable "var" and integer "dim" = 1 or 2, which switches between the dimension of reconsturction.
The output is a 2D array with reconstructed state variable at left and right sides of the cell faces 
on the faces of the 2D grid in 1- or 2- dimensions, depending on parameter dim. 

The idea of ENO (essentually non-oscillating) and WENO (weighted ENO) is a bit different from PLM -- instead of using piecewise constant values (just the cell averages)
we again try to find higher order polynomial extensions in each dimension, using the information from neighbouring cells
to construct the polynomial coefficients. All of the considerations about TVD/monotonicity (see "rec_PLM") stands the same here. 
But here we construct our high order polynomials on different stencils (e.g. in 1D along X-coordinate we can use 3 stencils - left(i-2,i-1,i), central(i-1,i,i+1) 
and right(i,i+1,i+2) for the cell with index i). Further we can find the so called indicator of smootheness (IS below) for each stencil, which show, how strongly the function varies near the cell.
They depend on the coefficients of original reconsturction. For example, if in the cell i+1 we have some sort of discontinuity,
in original ENO we should not use the stencils C and R, but inside the left stencil the solution will still be smooth, so that we can use it for high order reconstruction.  

By chosing the smoothest stencil (ENO) with the smallest IS, or by using the weighted convex combination of different stencils (WENO), we can obtain the reconstructed state on the face.

everywhere below the reconstructed value looks adopts the Legendre polynomial inside the cell along the 1- or 2-dimension
var_rec = = var(cell) + var_x(cell)*x + var_xx(cell)* (x^2 - 1/12), the integral over the cell volume will be var(cell)*Volume.
the var_x and var_xx are just the approximations for the derivatives.
We look for 3 stencils and further use either finite-difference WENO5 scheme for the weights (fifth order in space) of CWENO (C for central) weights, 
which has third order in space.

########################################
further reading -- see e.g. -- D.S. Balsara "Higher-order accurate space-time schemes
for computational astrophysics—Part I: finite volume
methods", Living Rev Comput Astrophys (2017) 3:2
########################################

WENO routine implemented here works only for uniform Cartesian grids
a switch between WENO5 finite-diffence reconstrcution and CWENO rec can also be adjusted in this file 
"""
def rec_WENO(Ngc, Nr, var, dim):
    
    
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



