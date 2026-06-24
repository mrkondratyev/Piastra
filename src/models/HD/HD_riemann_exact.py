# -*- coding: utf-8 -*-
"""
HD_riemann_exact.py

Exact Riemann solver for the Euler equations of ideal-gas dynamics
=================================================================

This module provides the exact solution to the Riemann problem for
compressible gas dynamics with an ideal-gas equation of state (gamma-law).
The algorithm follows Toro's textbook:

  Toro, E. F., "Riemann Solvers and Numerical Methods for Fluid Dynamics",
  3rd edition, Springer (2009)

The module contains two main interfaces:

1. ``exact_riemann_solution`` -- computes the exact solution profile at
   arbitrary spatial positions and a given time. Intended for generating
   reference solutions to validate approximate Riemann solvers and
   numerical schemes.

2. ``exact_riemann_godunov_state`` -- samples the exact solution at the
   interface (x/t = 0) for use as a Godunov numerical flux. Works with
   arrays for integration into the finite-volume framework
   (flux_type = 'Exact' in Riemann_nr_hydro).

References
----------
- Toro, E. F., "Riemann Solvers and Numerical Methods for Fluid Dynamics",
  3rd edition, Springer (2009)
- Godunov, S. K. (1959), "A difference method for the numerical computation
  of discontinuous solutions of the equations of hydrodynamics", Mat. Sbornik (in Russian)
  
 Notes
----------
- works only for ideal gas EOS! 
  by now, this function will silently corrupt with non-ideal EOS

Author
------
mrkondratyev
"""

import numpy as np


# =========================================================================
#  Internal helper functions
# =========================================================================

def _pressure_function(p, rho, pres, cs, gamma):
    """
    Evaluate the pressure function f_K(p) for one side of the Riemann fan.

    For a shock wave (p > p_K):
        f_K = (p - p_K) * sqrt(A_K / (p + B_K))

    For a rarefaction wave (p <= p_K):
        f_K = (2 * c_K / (gamma - 1)) * ((p / p_K)^((gamma-1)/(2*gamma)) - 1)

    Parameters
    ----------
    p : float or ndarray
        Pressure in the star region (current iterate).
    rho : float or ndarray
        Density on side K.
    pres : float or ndarray
        Pressure on side K.
    cs : float or ndarray
        Sound speed on side K.
    gamma : float
        Adiabatic index.

    Returns
    -------
    fK : float or ndarray
        Value of the pressure function.
    """

    A = 2.0 / ((gamma + 1.0) * rho)
    B = (gamma - 1.0) / (gamma + 1.0) * pres

    #shock branch: Rankine-Hugoniot jump relation
    f_shock = (p - pres) * np.sqrt(A / (p + B))

    #rarefaction branch: isentropic relation
    f_raref = 2.0 * cs / (gamma - 1.0) * ((p / pres) ** ((gamma - 1.0) / (2.0 * gamma)) - 1.0)

    return np.where(p > pres, f_shock, f_raref)


def _pressure_function_deriv(p, rho, pres, cs, gamma):
    """
    Evaluate the derivative f'_K(p) of the pressure function for one side.

    For a shock wave (p > p_K):
        f'_K = sqrt(A_K / (p + B_K)) * (1 - (p - p_K) / (2 * (p + B_K)))

    For a rarefaction wave (p <= p_K):
        f'_K = (1 / (rho_K * c_K)) * (p / p_K)^(-(gamma+1) / (2*gamma))

    Parameters
    ----------
    p : float or ndarray
        Pressure in the star region (current iterate).
    rho : float or ndarray
        Density on side K.
    pres : float or ndarray
        Pressure on side K.
    cs : float or ndarray
        Sound speed on side K.
    gamma : float
        Adiabatic index.

    Returns
    -------
    dfK : float or ndarray
        Derivative of the pressure function.
    """

    A = 2.0 / ((gamma + 1.0) * rho)
    B = (gamma - 1.0) / (gamma + 1.0) * pres

    #shock branch
    qrt = np.sqrt(A / (p + B))
    df_shock = qrt * (1.0 - (p - pres) / (2.0 * (p + B)))

    #rarefaction branch
    df_raref = 1.0 / (rho * cs) * (p / pres) ** (-(gamma + 1.0) / (2.0 * gamma))

    return np.where(p > pres, df_shock, df_raref)


def _initial_pressure_guess(rhol, vxl, pl, csl, rhor, vxr, pr, csr, gamma):
    """
    Compute an initial guess for the star-region pressure p* using
    Toro's adaptive approach (Toro, Section 9.3).

    Uses PVRS (Primitive Variable Riemann Solver) as baseline, then
    switches to two-rarefaction (TRRS) or two-shock (TSRS) estimates
    depending on the wave pattern.

    Parameters
    ----------
    rhol, rhor : float or ndarray
        Left and right densities.
    vxl, vxr : float or ndarray
        Left and right velocities.
    pl, pr : float or ndarray
        Left and right pressures.
    csl, csr : float or ndarray
        Left and right sound speeds.
    gamma : float
        Adiabatic index.

    Returns
    -------
    p0 : float or ndarray
        Initial guess for star-region pressure.
    """
    #PVRS (linearized) estimate (Toro, Eq. 9.28)
    Cup = 0.25 * (rhol + rhor) * (csl + csr)
    p_pvrs = 0.5 * (pl + pr) + 0.5 * (vxl - vxr) * Cup
    p_pvrs = np.maximum(p_pvrs, 1e-14)

    pmin = np.minimum(pl, pr)
    pmax = np.maximum(pl, pr)

    #two-rarefaction estimate TRRS (Toro, Eq. 9.31)
    z = (gamma - 1.0) / (2.0 * gamma)
    p_tr = ((csl + csr - (gamma - 1.0) / 2.0 * (vxr - vxl)) /
            (csl / pl ** z + csr / pr ** z)) ** (1.0 / z)
    p_tr = np.maximum(p_tr, 1e-14)

    #two-shock estimate TSRS using PVRS as starting point (Toro, Eq. 9.29)
    gl = np.sqrt(2.0 / ((gamma + 1.0) * rhol) / (p_pvrs + (gamma - 1.0) / (gamma + 1.0) * pl))
    gr = np.sqrt(2.0 / ((gamma + 1.0) * rhor) / (p_pvrs + (gamma - 1.0) / (gamma + 1.0) * pr))
    p_ts = (gl * pl + gr * pr - (vxr - vxl)) / (gl + gr)
    p_ts = np.maximum(p_ts, 1e-14)

    #adaptive selection: TRRS for strong rarefactions, TSRS for strong shocks
    p0 = np.where(p_pvrs < pmin, p_tr, np.where(p_pvrs > pmax, p_ts, p_pvrs))
    #guarantee pressure positivity
    p0 = np.maximum(p0, 1e-14)

    return p0


# =========================================================================
#  Newton method for pressure
# =========================================================================
def _solve_star_pressure(rhol, vxl, pl, csl, rhor, vxr, pr, csr, gamma,
                          eps=1e-8, max_iter=100):
    """
    Find the star-region pressure p* by Newton-Raphson iteration.

    Solves the nonlinear equation:

        f(p) = f_L(p) + f_R(p) + (u_R - u_L) = 0

    Parameters
    ----------
    rhol, rhor : float or ndarray
        Left and right densities.
    vxl, vxr : float or ndarray
        Left and right velocities.
    pl, pr : float or ndarray
        Left and right pressures.
    csl, csr : float or ndarray
        Left and right sound speeds.
    gamma : float
        Adiabatic index.
    eps : float, optional
        Relative tolerance for convergence. Default is 1e-8.
    max_iter : int, optional
        Maximum number of iterations. Default is 50.

    Returns
    -------
    pstar : float or ndarray
        Converged star-region pressure.
    """
    #initial guess using adaptive approach
    pstar = _initial_pressure_guess(rhol, vxl, pl, csl, rhor, vxr, pr, csr, gamma)

    #Newton-Raphson iteration
    for _ in range(max_iter):
        #evaluate pressure function and its derivative at both sides from the discontinuity
        fL = _pressure_function(pstar, rhol, pl, csl, gamma)
        fR = _pressure_function(pstar, rhor, pr, csr, gamma)

        dfL = _pressure_function_deriv(pstar, rhol, pl, csl, gamma)
        dfR = _pressure_function_deriv(pstar, rhor, pr, csr, gamma)

        #total function and derivative
        f = fL + fR + (vxr - vxl)
        df = dfL + dfR

        #Newton update
        dp = -f / df
        pstar_new = pstar + dp

        #guarantee positivity
        pstar_new = np.maximum(pstar_new, 1e-14)

        #relative residual (Toro, Eq. 4.44)
        res = np.abs(pstar_new - pstar) / (0.5 * (pstar_new + pstar) + 1e-30)

        pstar = pstar_new
        
        if np.all(res < eps):
            break

    return pstar


def _compute_star_velocity(pstar, rhol, vxl, pl, csl, rhor, vxr, pr, csr, gamma):
    """
    Compute the star-region velocity u* from the converged star pressure.

        u* = 0.5 * (u_L + u_R) + 0.5 * (f_R(p*) - f_L(p*))

    Parameters
    ----------
    pstar : float or ndarray
        Converged star-region pressure.
    rhol, rhor : float or ndarray
        Left and right densities.
    vxl, vxr : float or ndarray
        Left and right velocities.
    pl, pr : float or ndarray
        Left and right pressures.
    csl, csr : float or ndarray
        Left and right sound speeds.
    gamma : float
        Adiabatic index.

    Returns
    -------
    ustar : float or ndarray
        Star-region velocity.
    """
    fL = _pressure_function(pstar, rhol, pl, csl, gamma)
    fR = _pressure_function(pstar, rhor, pr, csr, gamma)

    ustar = 0.5 * (vxl + vxr) + 0.5 * (fR - fL)

    return ustar


# this function chooses the correct wave structure inside the Riemann fan
def _sample_solution(S, rhol, vxl, pl, csl, rhor, vxr, pr, csr,
                      pstar, ustar, gamma):
    """
    Sample the exact Riemann solution at a given similarity variable S = x/t
    (Toro, Section 4.5, Figure 4.14).

    Given the star-region values (p*, u*), determines which wave region
    the sampling point falls in and computes the corresponding state.

    Parameters
    ----------
    S : float or ndarray
        Sampling speed S = (x - x0) / t.
    rhol, rhor : float or ndarray
        Left and right densities.
    vxl, vxr : float or ndarray
        Left and right velocities.
    pl, pr : float or ndarray
        Left and right pressures.
    csl, csr : float or ndarray
        Left and right sound speeds.
    pstar : float or ndarray
        Star-region pressure.
    ustar : float or ndarray
        Star-region velocity.
    gamma : float
        Adiabatic index.

    Returns
    -------
    dens : float or ndarray
        Sampled density.
    vel : float or ndarray
        Sampled velocity.
    pres : float or ndarray
        Sampled pressure.
    """
    gm1 = gamma - 1.0
    gp1 = gamma + 1.0

    #precompute pressure ratios
    pratio_L = pstar / (pl + 1e-30)
    pratio_R = pstar / (pr + 1e-30)

    # =================================================================
    #  Left wave 
    # =================================================================

    #--- left rarefaction (p* <= p_L) ---

    #sound speed behind left rarefaction (Toro, Eq. 4.54)
    cstarL_rar = csl * pratio_L ** (gm1 / (2.0 * gamma))
    #head and tail speeds of the left rarefaction fan 
    SHL = vxl - csl          #head (leading edge)
    STL = ustar - cstarL_rar #tail (trailing edge)

    #density in star region behind left rarefaction 
    rhostarL_rar = rhol * pratio_L ** (1.0 / gamma)

    #solution inside left rarefaction fan 
    #guard the base against negative values outside the fan
    baseL = np.maximum(2.0 / gp1 + gm1 / (gp1 * csl) * (vxl - S), 1e-30)
    rho_fanL = rhol * baseL ** (2.0 / gm1)
    vel_fanL = 2.0 / gp1 * (csl + gm1 / 2.0 * vxl + S)
    prs_fanL = pl * baseL ** (2.0 * gamma / gm1)

    #--- left shock (p* > p_L) ---

    #left shock speed
    SL = vxl - csl * np.sqrt(gp1 / (2.0 * gamma) * pratio_L + gm1 / (2.0 * gamma))

    #density behind left shock
    rhostarL_shk = rhol * (pratio_L + gm1 / gp1) / (gm1 / gp1 * pratio_L + 1.0)

    #assemble left-of-contact state: rarefaction case
    dens_left_rar = np.where(S <= SHL, rhol,
                   np.where(S <= STL, rho_fanL, rhostarL_rar))
    vel_left_rar  = np.where(S <= SHL, vxl,
                   np.where(S <= STL, vel_fanL, ustar))
    prs_left_rar  = np.where(S <= SHL, pl,
                   np.where(S <= STL, prs_fanL, pstar))

    #assemble left-of-contact state: shock case
    dens_left_shk = np.where(S <= SL, rhol, rhostarL_shk)
    vel_left_shk  = np.where(S <= SL, vxl,  ustar)
    prs_left_shk  = np.where(S <= SL, pl,   pstar)

    #select rarefaction or shock for the left wave
    dens_left = np.where(pstar <= pl, dens_left_rar, dens_left_shk)
    vel_left  = np.where(pstar <= pl, vel_left_rar,  vel_left_shk)
    prs_left  = np.where(pstar <= pl, prs_left_rar,  prs_left_shk)

    # =================================================================
    #  Right wave
    # =================================================================

    #--- right rarefaction (p* <= p_R) ---

    #sound speed behind right rarefaction 
    cstarR_rar = csr * pratio_R ** (gm1 / (2.0 * gamma))
    #head and tail speeds of the right rarefaction fan 
    SHR = vxr + csr          #head (leading edge)
    STR = ustar + cstarR_rar #tail (trailing edge)

    #density in star region behind right rarefaction 
    rhostarR_rar = rhor * pratio_R ** (1.0 / gamma)

    #solution inside right rarefaction fan
    baseR = np.maximum(2.0 / gp1 - gm1 / (gp1 * csr) * (vxr - S), 1e-30)
    rho_fanR = rhor * baseR ** (2.0 / gm1)
    vel_fanR = 2.0 / gp1 * (-csr + gm1 / 2.0 * vxr + S)
    prs_fanR = pr * baseR ** (2.0 * gamma / gm1)

    #--- right shock (p* > p_R) ---

    #right shock speed 
    SR = vxr + csr * np.sqrt(gp1 / (2.0 * gamma) * pratio_R + gm1 / (2.0 * gamma))

    #density behind right shock (Toro, Eq. 4.57)
    rhostarR_shk = rhor * (pratio_R + gm1 / gp1) / (gm1 / gp1 * pratio_R + 1.0)

    #assemble right-of-contact state: rarefaction case
    dens_right_rar = np.where(S >= SHR, rhor,
                    np.where(S >= STR, rho_fanR, rhostarR_rar))
    vel_right_rar  = np.where(S >= SHR, vxr,
                    np.where(S >= STR, vel_fanR, ustar))
    prs_right_rar  = np.where(S >= SHR, pr,
                    np.where(S >= STR, prs_fanR, pstar))

    #assemble right-of-contact state: shock case
    dens_right_shk = np.where(S >= SR, rhor, rhostarR_shk)
    vel_right_shk  = np.where(S >= SR, vxr,  ustar)
    prs_right_shk  = np.where(S >= SR, pr,   pstar)

    #select rarefaction or shock for the right wave
    dens_right = np.where(pstar <= pr, dens_right_rar, dens_right_shk)
    vel_right  = np.where(pstar <= pr, vel_right_rar,  vel_right_shk)
    prs_right  = np.where(pstar <= pr, prs_right_rar,  prs_right_shk)

    # =================================================================
    #  Final assembly: left or right of contact 
    # =================================================================
    dens = np.where(S < ustar, dens_left, dens_right)
    vel  = np.where(S < ustar, vel_left,  vel_right)
    pres = np.where(S < ustar, prs_left,  prs_right)

    return dens, vel, pres


# =========================================================================
#  Public interfaces
# =========================================================================

def exact_riemann_godunov_state(rhol, rhor, vxl, vxr, pl, pr, gamma):
    """
    Sample the exact Riemann solution at the cell interface (S = x/t = 0)
    to obtain the Godunov state for numerical flux computation.

    This function works with arrays and is used internally by
    ``Riemann_HD`` when ``flux_type = 'Exact'``.

    Parameters
    ----------
    rhol, rhor : ndarray
        Left and right densities at each interface.
    vxl, vxr : ndarray
        Left and right normal velocities at each interface.
    pl, pr : ndarray
        Left and right pressures at each interface.
    gamma : float
        Adiabatic index.

    Returns
    -------
    dens : ndarray
        Density at the interface (Godunov state).
    vel : ndarray
        Normal velocity at the interface (Godunov state).
    pres : ndarray
        Pressure at the interface (Godunov state).
    ustar : ndarray
        Star-region velocity (needed to assign tangential velocities).
    """
    #sound speeds
    csl = np.sqrt(gamma * pl / rhol)
    csr = np.sqrt(gamma * pr / rhor)

    #solve for star-region pressure via vectorized Newton-Raphson
    pstar = _solve_star_pressure(rhol, vxl, pl, csl, rhor, vxr, pr, csr, gamma)

    #compute star-region velocity
    ustar = _compute_star_velocity(pstar, rhol, vxl, pl, csl,
                                    rhor, vxr, pr, csr, gamma)

    #sample at S = 0 (the cell interface)
    S = np.zeros_like(rhol)
    dens, vel, pres = _sample_solution(S, rhol, vxl, pl, csl,
                                        rhor, vxr, pr, csr,
                                        pstar, ustar, gamma)

    return dens, vel, pres, ustar


def exact_riemann_solution(rhol, vxl, pl, rhor, vxr, pr, gamma, x, t, x0=0.5):
    """
    Compute the exact solution of the Riemann problem for ideal gas dynamics.

    Solves the Riemann problem with initial data (rhol, vxl, pl) for x < x0
    and (rhor, vxr, pr) for x > x0, then evaluates the self-similar solution
    at positions *x* at time *t*.

    The algorithm follows Chapter 4 of Toro (2009): Newton-Raphson iteration
    for the star-region pressure, followed by exact sampling of all wave
    patterns (shocks, rarefactions, and contact discontinuity).

    Parameters
    ----------
    rhol : float
        Left-state density.
    vxl : float
        Left-state velocity.
    pl : float
        Left-state pressure.
    rhor : float
        Right-state density.
    vxr : float
        Right-state velocity.
    pr : float
        Right-state pressure.
    gamma : float
        Adiabatic index (ratio of specific heats).
    x : ndarray
        Array of spatial positions where the solution is evaluated.
    t : float
        Time at which to evaluate the solution (must be > 0).
    x0 : float, optional
        Initial position of the discontinuity. Default is 0.5.

    Returns
    -------
    dens : ndarray
        Density profile at time t.
    vel : ndarray
        Velocity profile at time t.
    pres : ndarray
        Pressure profile at time t.

    Raises
    ------
    ValueError
        If the initial data generates a vacuum (pressure positivity violated)
        or if t <= 0.

    Notes
    -----
    - Assumes ideal gas equation of state: p = (gamma - 1) * rho * e.
    - Uses Toro's adaptive initial guess (PVRS / TRRS / TSRS).
    - Newton-Raphson converges to machine precision in 5-10 iterations
      for typical problems.
    - The solution includes all wave patterns: left/right shocks or
      rarefactions, and a contact discontinuity.

    Examples
    --------
    Sod shock tube:

    >>> import numpy as np
    >>> x = np.linspace(0, 1, 1000)
    >>> dens, vel, pres = exact_riemann_solution(
    ...     1.0, 0.0, 1.0, 0.125, 0.0, 0.1, 1.4, x, 0.2)
    """
    #validate time
    if t <= 0.0:
        raise ValueError(f"Time must be positive, got t = {t}.")

    #ensure x is a numpy array
    x = np.asarray(x, dtype=np.float64)

    #sound speeds
    csl = np.sqrt(gamma * pl / rhol)
    csr = np.sqrt(gamma * pr / rhor)

    #check for vacuum generation (Toro, Eq. 4.40)
    gm1 = gamma - 1.0
    if 2.0 * csl / gm1 + 2.0 * csr / gm1 <= (vxr - vxl):
        raise ValueError(
            "Initial data generates a vacuum. "
            "The exact Riemann solver does not handle vacuum states. "
            f"2*(c_L + c_R)/(gamma-1) = {2.0*(csl + csr)/gm1:.6f}, "
            f"u_R - u_L = {vxr - vxl:.6f}")

    #solve for star-region pressure via Newton-Raphson
    pstar = _solve_star_pressure(rhol, vxl, pl, csl, rhor, vxr, pr, csr, gamma)

    #compute star-region velocity
    ustar = _compute_star_velocity(pstar, rhol, vxl, pl, csl,
                                    rhor, vxr, pr, csr, gamma)

    #similarity variable S = (x - x0) / t
    S = (x - x0) / t

    #sample the solution at each position
    dens, vel, pres = _sample_solution(S, rhol, vxl, pl, csl,
                                        rhor, vxr, pr, csr,
                                        pstar, ustar, gamma)

    return dens, vel, pres
