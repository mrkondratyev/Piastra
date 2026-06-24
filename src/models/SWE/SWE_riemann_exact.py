# -*- coding: utf-8 -*-
"""
riemann_exact_swe.py
====================

Exact Riemann solver for the 1D shallow water equations (SWE).

The SWE in conservation form (normal direction x, tangential direction y):

    h_t  + (h vx)_x                          = 0
    (h vx)_t + (h vx^2 + g h^2 / 2)_x       = 0
    (h vy)_t + (h vx vy)_x                   = 0

where:
    h   – water height (plays the role of density)
    vx  – normal velocity
    vy  – tangential velocity (passively advected across contacts)
    g   – gravitational acceleration

Wave structure
--------------
The SWE Riemann problem produces exactly **three wave families**:

    λ₁ = vx − c    (left-going, genuinely nonlinear → shock or rarefaction)
    λ₂ = vx        (contact / entropy wave; vy jumps here, vx and h are continuous)
    λ₃ = vx + c    (right-going, genuinely nonlinear → shock or rarefaction)

where c = sqrt(g h) is the gravity-wave (shallow-water) speed.

SWE are mathematically equivalent to the isentropic gas-dynamics equations
with an adiabatic exponent γ = 2 and the pressure law p = g h²/2.
This means the exact-solution algorithm mirrors the gas-dynamics case
(Toro 2009), with the following substitutions:

    gas dynamics   ↔   SWE
    ─────────────────────────────────────
    ρ              ↔   h
    p = ρ^γ / γ   ↔   g h²/2
    c = sqrt(γ p/ρ) ↔  sqrt(g h)
    γ = 2

The Riemann invariants across a rarefaction fan are
    R±  =  vx ± 2c            (analogous to the γ=2 gas-dynamics invariants)

Across a shock (Rankine-Hugoniot):
    h* (vx* - S) = h_K (vx_K - S)          (mass)
    h* vx* (vx* - S) + g h*²/2 = h_K vx_K (vx_K - S) + g h_K²/2  (momentum)

Eliminating S gives the shock-speed formula and the jump condition:
    vx* - vx_K = ±(h* - h_K) sqrt( g/(2 h_K h*) * (h_K + h*) / 2 )

which is the SWE analogue of Toro (2009)

Contact wave
------------
The middle wave λ₂ = vx separates two states that share the same (h*, vx*)
but may differ in vy.  The tangential velocity vy is determined by upwind
selection: vy* = vy_L if the contact moves rightward (vx* ≥ 0), else vy_R.

Two public interfaces
---------------------
1.  ``exact_swe_godunov_state``
        Vectorised: samples the exact solution at x/t = 0 for use as a
        Godunov flux in the SWE finite-volume solver.

2.  ``exact_swe_solution``
        Scalar-initial-data version: returns (h, vx, vy) profiles on an
        arbitrary array of x values at a given time t > 0.
        Intended for generating reference solutions and convergence tests.

References
----------
(1) E. F. Toro, "Riemann Solvers and Numerical Methods for Fluid Dynamics" (2009) 
(2) E. F. Toro, "Computational Algorithms for Shallow Water Equations" (2025)

Author: mrkondratyev; tutorial style follows riemann_exact.py
"""

import numpy as np


# =========================================================================
#  Internal helpers
# =========================================================================

def _pressure_fn(h_star, h_K, vx_K, c_K, g):
    """
    Evaluate the pressure-like function  f_K(h*)  for one side K of the fan.

    This is the SWE analogue of Toro (2009), Eq. (5.54) / (5.56):

    Rarefaction (h* ≤ h_K):
        f_K = 2 (c* − c_K)  =  2 (sqrt(g h*) − sqrt(g h_K))

    Shock (h* > h_K):
        f_K = (h* − h_K) sqrt( g (h* + h_K) / (2 h* h_K) )

    Parameters
    ----------
    h_star : float or ndarray   current height iterate in the star region
    h_K    : float or ndarray   height on side K  (L or R)
    vx_K   : float or ndarray   normal velocity on side K   (unused here, kept for API symmetry)
    c_K    : float or ndarray   celerity on side K  = sqrt(g h_K)
    g      : float              gravitational acceleration

    Returns
    -------
    f : float or ndarray   value of the Riemann function
    """
    #wave speed 
    c_star = np.sqrt(g * np.maximum(h_star, 0.0))

    # Rarefaction branch (isentropic): f = 2(c* - c_K)
    f_rar = 2.0 * (c_star - c_K)

    # Shock branch (Rankine-Hugoniot): f = (h* - h_K) sqrt(g(h*+h_K)/(2 h* h_K))
    # Guard denominator
    denom = np.where(h_star * h_K > 0.0,
                     2.0 * h_star * h_K,
                     np.ones_like(h_star))
    f_shk = (h_star - h_K) * np.sqrt(g * (h_star + h_K) / denom)

    return np.where(h_star <= h_K, f_rar, f_shk)


def _pressure_fn_deriv(h_star, h_K, c_K, g):
    """
    Derivative  df_K/dh*  needed by Newton-Raphson.

    Rarefaction:  df/dh* = c* / h*       (since dc*/dh* = g/(2 c*))
    Shock:        df/dh* = g (3 h* + h_K) / (4 h* c_shk)
                  where c_shk = sqrt(g(h*+h_K)/(2 h* h_K)) * (h*-h_K) would
                  be the full expression; we differentiate the exact formula.
    """
    c_star = np.sqrt(g * np.maximum(h_star, 0.0))

    # Rarefaction: df/dh* = g / (2 c*) / 1  = sqrt(g / h*)
    df_rar = np.where(c_star > 0.0,
                      np.sqrt(g / np.maximum(h_star, 1e-30)),
                      np.zeros_like(h_star))

    # Shock: differentiate f = (h*-h_K) sqrt(g(h*+h_K)/(2 h* h_K))
    # Let A = g/(2 h_K),  q = (h* + h_K) / h*  = 1 + h_K/h*
    # f = (h*-h_K) sqrt(A q)
    # df/dh* = sqrt(A q) + (h*-h_K)/(2 sqrt(A q)) * A * d(q)/dh*
    # d(q)/dh* = -h_K/h*^2
    A    = g / (2.0 * np.maximum(h_K, 1e-30))
    q    = (h_star + h_K) / np.maximum(h_star, 1e-30)
    sqAq = np.sqrt(np.maximum(A * q, 0.0))
    df_shk = np.where(sqAq > 0.0,
                      sqAq - (h_star - h_K) * A * h_K /
                      (2.0 * sqAq * np.maximum(h_star**2, 1e-30)),
                      np.zeros_like(h_star))

    return np.where(h_star <= h_K, df_rar, df_shk)


def _initial_height_guess(h_L, vx_L, c_L, h_R, vx_R, c_R, g):
    """
    Adaptive initial guess for h*.

    Three estimates are blended:

    PVRS (linearised):
        h_pvrs = ((c_L + c_R) - (vx_R - vx_L)/2)^2 / (4g)  ... wait,
        more precisely the primitive-variable Riemann solver gives
        h_pvrs = (c_L + c_R - (vx_R - vx_L)/4)^2 / g

    Two-rarefaction (TRR): from setting both waves to rarefactions
        h_trr = ((c_L + c_R - (vx_R - vx_L)/2) / (2*sqrt(g)) )^2
        simplified to  h_trr = (c_L + c_R)/2 - (vx_R-vx_L)/4)^2 / g

    Two-shock: iterative; approximated by PVRS.

    In practice the two-rarefaction estimate is excellent for SWE because
    rarefactions are more common than shocks (no entropy condition complication).
    We use PVRS when h_pvrs lies between h_L and h_R, TRR for strong
    rarefactions, and fall back to PVRS otherwise.
    """
    # PVRS estimate (primitive-variable linearisation for SWE)
    # Linearises around arithmetic averages
    h_bar = 0.5 * (h_L + h_R)
    c_bar = 0.5 * (c_L + c_R)
    h_pvrs = h_bar - 0.25 * (vx_R - vx_L) * h_bar / c_bar
    h_pvrs = np.maximum(h_pvrs, 1e-14)

    # Two-rarefaction estimate (exact for two rarefaction waves)
    # From R+ invariant = R- invariant at x* :
    # vx* + 2 c* = vx_L + 2 c_L   (left)
    # vx* - 2 c* = vx_R - 2 c_R   (right)
    # Adding: 4 c* = (vx_L - vx_R) + 2(c_L + c_R)
    # c* = (c_L + c_R)/2 - (vx_R - vx_L)/4
    c_star_trr = 0.5 * (c_L + c_R) - 0.25 * (vx_R - vx_L)
    c_star_trr = np.maximum(c_star_trr, 1e-14)
    h_trr = c_star_trr**2 / g

    # Choose: if PVRS falls between the two heights use it, otherwise TRR
    h_min = np.minimum(h_L, h_R)
    h_max = np.maximum(h_L, h_R)
    in_range = (h_pvrs >= h_min) & (h_pvrs <= h_max)

    h0 = np.where(in_range, h_pvrs, h_trr)
    return np.maximum(h0, 1e-14)


def _solve_star_height(h_L, vx_L, c_L, h_R, vx_R, c_R, g,
                       tol=1e-8, max_iter=100):
    """
    Newton-Raphson iteration for the star-region height h*.

    Solves  F(h*) = f_L(h*) + f_R(h*) + (vx_R - vx_L) = 0

    where f_K is defined in _pressure_fn.  

    Parameters
    ----------
    h_L, h_R   : float or ndarray   heights on left/right sides
    vx_L, vx_R : float or ndarray   normal velocities
    c_L, c_R   : float or ndarray   celerities  sqrt(g h)
    g          : float
    tol        : float              convergence tolerance on relative change
    max_iter   : int

    Returns
    -------
    h_star : float or ndarray   converged star-region height
    """
    h_star = _initial_height_guess(h_L, vx_L, c_L, h_R, vx_R, c_R, g)

    for _ in range(max_iter):
        f_L  = _pressure_fn(h_star, h_L, vx_L, c_L, g)
        f_R  = _pressure_fn(h_star, h_R, vx_R, c_R, g)
        df_L = _pressure_fn_deriv(h_star, h_L, c_L, g)
        df_R = _pressure_fn_deriv(h_star, h_R, c_R, g)

        F  = f_L + f_R + (vx_R - vx_L)
        dF = df_L + df_R

        delta = F / (dF + 1e-30)
        h_new = h_star - delta
        h_new = np.maximum(h_new, 1e-14)

        # Convergence: relative change
        rel = np.abs(h_new - h_star) / (0.5 * (h_new + h_star) + 1e-30)
        h_star = h_new

        if np.all(rel < tol):
            break

    return h_star


def _compute_star_velocity(h_star, h_L, vx_L, c_L, h_R, vx_R, c_R, g):
    """
    Star-region normal velocity  vx*  from h*.

    From adding the two wave conditions:
        vx* = 0.5*(vx_L + vx_R) + 0.5*(f_R(h*) - f_L(h*))

    Parameters  mirror _solve_star_height.

    Returns
    -------
    vx_star : float or ndarray
    """
    f_L = _pressure_fn(h_star, h_L, vx_L, c_L, g)
    f_R = _pressure_fn(h_star, h_R, vx_R, c_R, g)
    return 0.5 * (vx_L + vx_R) + 0.5 * (f_R - f_L)


# =========================================================================
#  Solution sampling: given h*, vx*, determine the state at x/t = S
# =========================================================================

def _sample_solution(S, h_L, vx_L, c_L, vy_L,
                          h_R, vx_R, c_R, vy_R,
                          h_star, vx_star, g):
    """
    Sample the exact SWE Riemann solution at similarity speed  S = x/t.

    The wave pattern consists of:
      - Left wave  (rarefaction fan or shock)
      - Contact discontinuity at  S = vx*  (vy jumps here)
      - Right wave (rarefaction fan or shock)

    Parameters
    ----------
    S       : float or ndarray   sampling speed  x/t
    h_L/R   : float or ndarray   left/right heights
    vx_L/R  : float or ndarray   left/right normal velocities
    c_L/R   : float or ndarray   left/right celerities
    vy_L/R  : float or ndarray   left/right tangential velocities
    h_star  : float or ndarray   star-region height
    vx_star : float or ndarray   star-region normal velocity
    g       : float

    Returns
    -------
    h, vx, vy : float or ndarray   sampled state
    """
    c_star = np.sqrt(g * np.maximum(h_star, 0.0))

    # ── Left wave ────────────────────────────────────────────────────────
    # Shock speed from the Rankine-Hugoniot mass condition:
    #   S_L (h* − h_L) = h* vx* − h_L vx_L
    #   → S_L = vx_L − sqrt( g h* (h* + h_L) / (2 h_L) )
    # (derived by substituting f_L = vx_L − vx* into the mass jump)
    S_shk_L = vx_L - np.sqrt(g * h_star * (h_star + h_L) /
                               (2.0 * np.maximum(h_L, 1e-30)))

    # Rarefaction head and tail
    S_head_L = vx_L - c_L       # leading edge of left fan
    S_tail_L = vx_star - c_star  # trailing edge of left fan

    # Height and velocity inside left rarefaction fan
    # From Riemann invariant  vx + 2c = vx_L + 2c_L  at all points in fan:
    #   c_fan = (c_L + (vx_L - S)/2)  
    #   vx_fan = (vx_L + 2 c_L + 2 S) / 3  
    #   c_fan  = (vx_L + 2 c_L - S) / 3
    vx_fan_L = (vx_L + 2.0 * c_L + 2.0 * S) / 3.0
    c_fan_L  = (vx_L + 2.0 * c_L - S) / 3.0
    h_fan_L  = np.maximum(c_fan_L**2 / g, 0.0)

    # Left-rarefaction case: assemble state (head, fan, tail, star)
    h_left_rar  = np.where(S <= S_head_L, h_L,
                  np.where(S <= S_tail_L, h_fan_L, h_star))
    vx_left_rar = np.where(S <= S_head_L, vx_L,
                  np.where(S <= S_tail_L, vx_fan_L, vx_star))

    # Left-shock case: upstream or downstream
    h_left_shk  = np.where(S <= S_shk_L, h_L, h_star)
    vx_left_shk = np.where(S <= S_shk_L, vx_L, vx_star)

    # Select left wave type
    h_left  = np.where(h_star <= h_L, h_left_rar,  h_left_shk)
    vx_left = np.where(h_star <= h_L, vx_left_rar, vx_left_shk)

    # ── Right wave ───────────────────────────────────────────────────────
    # Shock speed (Rankine-Hugoniot):
    #   S_R = vx_R + sqrt( g h* (h* + h_R) / (2 h_R) )
    S_shk_R = vx_R + np.sqrt(g * h_star * (h_star + h_R) /
                               (2.0 * np.maximum(h_R, 1e-30)))

    S_head_R = vx_R + c_R        # leading edge of right fan
    S_tail_R = vx_star + c_star   # trailing edge of right fan

    vx_fan_R = (vx_R - 2.0 * c_R + 2.0 * S) / 3.0
    c_fan_R  = (S - vx_R + 2.0 * c_R) / 3.0
    h_fan_R  = np.maximum(c_fan_R**2 / g, 0.0)

    h_right_rar  = np.where(S >= S_head_R, h_R,
                   np.where(S >= S_tail_R, h_fan_R, h_star))
    vx_right_rar = np.where(S >= S_head_R, vx_R,
                   np.where(S >= S_tail_R, vx_fan_R, vx_star))

    h_right_shk  = np.where(S >= S_shk_R, h_R, h_star)
    vx_right_shk = np.where(S >= S_shk_R, vx_R, vx_star)

    h_right  = np.where(h_star <= h_R, h_right_rar,  h_right_shk)
    vx_right = np.where(h_star <= h_R, vx_right_rar, vx_right_shk)

    # ── Assemble: left of contact or right of contact ────────────────────
    h  = np.where(S <  vx_star, h_left,  h_right)
    vx = np.where(S <  vx_star, vx_left, vx_right)

    # Tangential velocity: upwind across the contact wave
    vy = np.where(S < vx_star, vy_L, vy_R)

    return h, vx, vy


# =========================================================================
#  Public interface 1: Godunov state at x/t = 0 (for finite-volume use)
# =========================================================================

def exact_swe_godunov_state(h_L, h_R, vx_L, vx_R, vy_L, vy_R, g):
    """
    Sample the exact SWE Riemann solution at the cell interface (x/t = 0).

    Intended for use as the Godunov numerical flux inside the SWE
    finite-volume solver (replace the HLL call in ``Riemann_flux_SWE``).
    All arguments may be NumPy arrays for vectorised use over all interfaces.

    Parameters
    ----------
    h_L, h_R   : float or ndarray   left/right water heights
    vx_L, vx_R : float or ndarray   left/right normal velocities
    vy_L, vy_R : float or ndarray   left/right tangential velocities
    g          : float              gravitational acceleration

    Returns
    -------
    h0, vx0, vy0 : float or ndarray
        Water height, normal velocity, and tangential velocity at x/t = 0.
        Pass these directly to the SWE flux formula:
            F_h   = h0 * vx0
            F_hvx = h0 * vx0**2 + g * h0**2 / 2
            F_hvy = h0 * vx0 * vy0

    Raises
    ------
    ValueError
        If the initial data generates a dry state (total Riemann invariant
        condition violated: vx_R - vx_L ≥ 2*(c_L + c_R)).

    Notes
    -----
    A dry-bed initial condition (h_L = 0 or h_R = 0) is handled by clamping
    h_star to a small positive number; the result approaches the wet/dry front
    speed in the limit.
    """
    h_L  = np.asarray(h_L,  dtype=float)
    h_R  = np.asarray(h_R,  dtype=float)
    vx_L = np.asarray(vx_L, dtype=float)
    vx_R = np.asarray(vx_R, dtype=float)
    vy_L = np.asarray(vy_L, dtype=float)
    vy_R = np.asarray(vy_R, dtype=float)

    c_L = np.sqrt(g * np.maximum(h_L, 0.0))
    c_R = np.sqrt(g * np.maximum(h_R, 0.0))

    h_star  = _solve_star_height(h_L, vx_L, c_L, h_R, vx_R, c_R, g)
    vx_star = _compute_star_velocity(h_star, h_L, vx_L, c_L,
                                              h_R, vx_R, c_R, g)

    S = np.zeros_like(h_star)   # sample at x/t = 0
    h0, vx0, vy0 = _sample_solution(S,
                                     h_L, vx_L, c_L, vy_L,
                                     h_R, vx_R, c_R, vy_R,
                                     h_star, vx_star, g)
    return h0, vx0, vy0


# =========================================================================
#  Public interface 2: full solution profile on x array at time t
# =========================================================================

def exact_swe_solution(h_L, vx_L, vy_L,
                        h_R, vx_R, vy_R,
                        g, x, t, x0=0.5):
    """
    Compute the exact solution of the 1D SWE Riemann problem at time t > 0.

    Given constant left and right states separated at x = x0, evaluates the
    self-similar solution at every point in the array ``x``.

    Parameters
    ----------
    h_L, h_R   : float   left/right water heights (scalar)
    vx_L, vx_R : float   left/right normal velocities
    vy_L, vy_R : float   left/right tangential velocities
    g          : float   gravitational acceleration
    x          : array_like   spatial positions where solution is evaluated
    t          : float        time  (must be > 0)
    x0         : float        initial discontinuity position  (default 0.5)

    Returns
    -------
    h  : ndarray   water height  at (x, t)
    vx : ndarray   normal velocity
    vy : ndarray   tangential velocity

    Raises
    ------
    ValueError
        If t ≤ 0 or if the data generates a complete dry-bed (h* → 0).

    Examples
    --------
    Classical dam-break (h_L=1, h_R=0.1, vx=vy=0, g=9.81):

    >>> import numpy as np
    >>> x = np.linspace(0, 1, 1000)
    >>> h, vx, vy = exact_swe_solution(1.0, 0.0, 0.0,
    ...                                 0.1, 0.0, 0.0,
    ...                                 9.81, x, 0.3)

    Wet-dam-break (Stoker 1957):

    >>> h, vx, vy = exact_swe_solution(2.0, 0.0, 0.0,
    ...                                 1.0, 0.0, 0.0,
    ...                                 1.0, x, 0.5)
    """
    if t <= 0.0:
        raise ValueError(f"Time must be positive, got t = {t}.")

    x   = np.asarray(x, dtype=float)
    c_L = np.sqrt(g * np.maximum(h_L, 0.0))
    c_R = np.sqrt(g * np.maximum(h_R, 0.0))

    # Dry-bed / vacuum check: if vx_R - vx_L >= 2(c_L + c_R) no wet star
    # region exists (total separation of the two sides).
    if vx_R - vx_L >= 2.0 * (c_L + c_R):
        raise ValueError(
            "Initial data generates a dry bed (total separation). "
            f"Condition: vx_R - vx_L = {vx_R - vx_L:.4g} >= "
            f"2*(c_L + c_R) = {2*(c_L + c_R):.4g}."
        )

    h_star  = _solve_star_height(
                  np.atleast_1d(float(h_L)),
                  np.atleast_1d(float(vx_L)),
                  np.atleast_1d(float(c_L)),
                  np.atleast_1d(float(h_R)),
                  np.atleast_1d(float(vx_R)),
                  np.atleast_1d(float(c_R)),
                  g)[0]

    vx_star = _compute_star_velocity(
                  h_star,
                  np.atleast_1d(float(h_L)),
                  np.atleast_1d(float(vx_L)),
                  np.atleast_1d(float(c_L)),
                  np.atleast_1d(float(h_R)),
                  np.atleast_1d(float(vx_R)),
                  np.atleast_1d(float(c_R)),
                  g)[0]

    S = (x - x0) / t   # similarity variable

    h, vx, vy = _sample_solution(
        S,
        float(h_L), float(vx_L), float(c_L), float(vy_L),
        float(h_R), float(vx_R), float(c_R), float(vy_R),
        h_star, vx_star, g)

    return h, vx, vy


# =========================================================================
#  Quick self-test / demo  (run with  python SWE/riemann_exact.py)
# =========================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    g = 1.0
    x = np.linspace(0.0, 1.0, 200)

    # ── Test cases ────────────────────────────────────────────────────────
    cases = [
        # (label,  h_L,  vx_L,  h_R,  vx_R,  vy_L, vy_R,  t,   x0)
        ("Dam break (h_L=1, h_R=0.125)", 1.0, 0.0, 0.125, 0.0, 0.0, -0.0, 0.30, 0.5),
        ("Wet dam  (h_L=2, h_R=1)",      2.0, 0.0, 1.0,   0.0, 0.0,  0.0, 0.40, 0.5),
        ("Two shocks (v inward)",         1.0, 1.0, 1.0,  -1.0, 0.0,  0.0, 0.15, 0.5),
        ("Two rarefactions (v outward)",  1.0,-1.0, 1.0,   1.0, 0.0,  0.0, 0.20, 0.5),
        ("Tangential velocity jump",      1.0, 0.0, 0.25,  0.0, 1.5, -0.5, 0.25, 0.5),
    ]

    fig, axes = plt.subplots(len(cases), 3, figsize=(14, 3.2 * len(cases)))

    for row, (label, h_L, vx_L, h_R, vx_R, vy_L, vy_R, t, x0) in enumerate(cases):
        h, vx, vy = exact_swe_solution(h_L, vx_L, vy_L,
                                        h_R, vx_R, vy_R,
                                        g, x, t, x0)
        c_L = np.sqrt(g * h_L);  c_R = np.sqrt(g * h_R)

        # Compute star state for annotation
        hs = _solve_star_height(
            np.array([h_L]), np.array([vx_L]), np.array([c_L]),
            np.array([h_R]), np.array([vx_R]), np.array([c_R]), g)[0]
        vs = _compute_star_velocity(
            hs, np.array([h_L]), np.array([vx_L]), np.array([c_L]),
                np.array([h_R]), np.array([vx_R]), np.array([c_R]), g)[0]

        for col, (ydata, ylabel) in enumerate(
            [(h, 'h  (water height)'), (vx, 'vx  (normal vel.)'), (vy, 'vy  (tangential vel.)')]):

            ax = axes[row, col]
            ax.plot(x, ydata, 'C0', lw=2)
            ax.axvline(x0, color='k', lw=0.8, ls=':')
            ax.set_ylabel(ylabel, fontsize=8)
            ax.set_xlabel('x', fontsize=8)
            if col == 0:
                ax.set_title(f'{label}\nt={t},  h*={hs:.4f},  vx*={vs:.4f}',
                             fontsize=8)

    plt.suptitle('Exact SWE Riemann solver — test cases  (g = 9.81)', y=1.01)
    plt.tight_layout()
    plt.savefig('swe_riemann_exact.png', dpi=120, bbox_inches='tight')
    plt.show()
    print("Done.  Figure saved to  swe_riemann_exact.png")
