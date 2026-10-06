# -*- coding: utf-8 -*-
"""
SWE_riemann_exact.py
====================

Exact Riemann solver for the 1D shallow water equations (SWE).

Equations (x = normal direction, y = tangential direction):

    h_t      + (h vx)_x                = 0
    (h vx)_t + (h vx^2 + g h^2 / 2)_x  = 0
    (h vy)_t + (h vx vy)_x             = 0

h is the water height, vx, vy the normal and tangential velocities, g the
gravitational acceleration, c = sqrt(g h) the gravity-wave speed.

Wave structure
--------------
    lambda_1 = vx - c   left wave:  shock or rarefaction
    lambda_2 = vx       shear wave: only vy jumps (h, vx continuous)
    lambda_3 = vx + c   right wave: shock or rarefaction

The SWE are the isentropic gas-dynamics equations with rho -> h,
p -> g h^2 / 2 (gamma = 2), so the algorithm follows Toro (2009), Ch. 4.

Star state
----------
h* is the root of

    F(h*) = f_L(h*) + f_R(h*) + (vx_R - vx_L) = 0,

    f_K = 2 (c* - c_K)                                    rarefaction, h* <= h_K
    f_K = (h* - h_K) sqrt( g (h* + h_K) / (2 h* h_K) )    shock,       h* >  h_K

with c* = sqrt(g h*), and then  vx* = (vx_L + vx_R)/2 + (f_R - f_L)/2.
Shocks move with  S_L = vx_L - sqrt(g h* (h* + h_L) / (2 h_L))  and
S_R = vx_R + sqrt(g h* (h* + h_R) / (2 h_R)); inside a rarefaction the
Riemann invariant vx +- 2c is constant. Across the shear wave vy is
upwinded: vy_L where x/t < vx*, vy_R otherwise.

Public interfaces
-----------------
exact_swe_godunov_state : vectorised state at x/t = 0 (Godunov flux,
                          solver_type = 'Exact' in Riemann_SWE).
exact_swe_solution      : (h, vx, vy) profiles at time t for scalar
                          initial data (reference solutions, tests).

References
----------
E. F. Toro, "Riemann Solvers and Numerical Methods for Fluid Dynamics",
    3rd ed., Springer (2009)
E. F. Toro, "Shock-Capturing Methods for Free-Surface Shallow Flows",
    Wiley (2001)
E. F. Toro, "Computational Algorithms for Shallow Water Equations",
    Springer (2024)

Author: mrkondratyev
"""

import numpy as np


# =========================================================================
#  Internal helpers
# =========================================================================

def _pressure_fn(h_star, h_K, vx_K, c_K, g):
    """
    Wave function f_K(h*) for side K (L or R):

        rarefaction (h* <= h_K):  f_K = 2 (c* - c_K)
        shock       (h* >  h_K):  f_K = (h* - h_K) sqrt( g (h* + h_K) / (2 h* h_K) )

    vx_K is unused (kept for a uniform call signature).
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
    Derivative df_K/dh* for the Newton iteration:

        rarefaction:  df/dh* = g / c* = sqrt(g / h*)
        shock:        df/dh* = g (2 h*^2 + h* h_K + h_K^2) / (4 h*^2 h_K Q),
                      Q = sqrt( g (h* + h_K) / (2 h* h_K) )

    The shock branch is coded as  Q - (h* - h_K) g / (4 h*^2 Q),  which is
    the same expression.
    """
    c_star = np.sqrt(g * np.maximum(h_star, 0.0))

    # Rarefaction: df/dh* = g / c* = sqrt(g / h*)
    df_rar = np.where(c_star > 0.0,
                      np.sqrt(g / np.maximum(h_star, 1e-30)),
                      np.zeros_like(h_star))

    # Shock: differentiate f = (h*-h_K) sqrt(A q),
    # A = g/(2 h_K),  q = (h* + h_K) / h*,  dq/dh* = -h_K / h*^2
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
    Initial guess for h* (Toro 2001, Ch. 5):

        PVRS (linearised):  h_pvrs = h_bar - (vx_R - vx_L) h_bar / (4 c_bar),
                            h_bar = (h_L + h_R)/2,  c_bar = (c_L + c_R)/2
        two rarefactions:   h_trr = c*^2 / g,
                            c* = (c_L + c_R)/2 - (vx_R - vx_L)/4   (exact for two fans)

    h_pvrs is used when it lies between h_L and h_R, h_trr otherwise.
    """
    # PVRS estimate (linearised around arithmetic averages)
    h_bar = 0.5 * (h_L + h_R)
    c_bar = 0.5 * (c_L + c_R)
    h_pvrs = h_bar - 0.25 * (vx_R - vx_L) * h_bar / c_bar
    h_pvrs = np.maximum(h_pvrs, 1e-14)

    # Two-rarefaction estimate, from the Riemann invariants
    #   vx* + 2 c* = vx_L + 2 c_L,   vx* - 2 c* = vx_R - 2 c_R
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
    Newton iteration for  F(h*) = f_L(h*) + f_R(h*) + (vx_R - vx_L) = 0.

    Stops when the relative change of h* is below tol in every cell (or
    after max_iter iterations); h* is kept >= 1e-14.

    Parameters
    ----------
    h_L, h_R, vx_L, vx_R, c_L, c_R : float or ndarray   left/right states, c = sqrt(g h)
    g        : float
    tol      : float   tolerance on the relative change of h*
    max_iter : int

    Returns
    -------
    h_star : float or ndarray
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
    Star-region normal velocity:  vx* = (vx_L + vx_R)/2 + (f_R(h*) - f_L(h*))/2.
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
    Exact solution at the similarity speed S = x/t.

    Left wave  (S < vx*):
        shock (h* > h_L):        h_L for S <= S_L, h* otherwise,
                                 S_L = vx_L - sqrt(g h* (h* + h_L) / (2 h_L))
        rarefaction (h* <= h_L): head vx_L - c_L, tail vx* - c*, inside
                                 vx = (vx_L + 2 c_L + 2 S)/3,  c = (vx_L + 2 c_L - S)/3
    Right wave (S >= vx*), mirror image:
                                 S_R = vx_R + sqrt(g h* (h* + h_R) / (2 h_R)),
                                 head vx_R + c_R, tail vx* + c*, inside
                                 vx = (vx_R - 2 c_R + 2 S)/3,  c = (S - vx_R + 2 c_R)/3
    Inside a fan h = c^2 / g. Tangential velocity: vy_L for S < vx*, else vy_R.

    Returns
    -------
    h, vx, vy : float or ndarray
    """
    c_star = np.sqrt(g * np.maximum(h_star, 0.0))

    # ── Left wave ────────────────────────────────────────────────────────
    # Shock speed (mass jump with vx_L - vx* = f_L):
    #   S_L = vx_L - sqrt( g h* (h* + h_L) / (2 h_L) )
    S_shk_L = vx_L - np.sqrt(g * h_star * (h_star + h_L) /
                               (2.0 * np.maximum(h_L, 1e-30)))

    # Rarefaction head and tail
    S_head_L = vx_L - c_L       # leading edge of left fan
    S_tail_L = vx_star - c_star  # trailing edge of left fan

    # Inside the left fan: vx + 2c = vx_L + 2c_L and vx - c = S
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
    # Shock speed:  S_R = vx_R + sqrt( g h* (h* + h_R) / (2 h_R) )
    S_shk_R = vx_R + np.sqrt(g * h_star * (h_star + h_R) /
                               (2.0 * np.maximum(h_R, 1e-30)))

    S_head_R = vx_R + c_R        # leading edge of right fan
    S_tail_R = vx_star + c_star   # trailing edge of right fan

    # Inside the right fan: vx - 2c = vx_R - 2c_R and vx + c = S
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
    Exact SWE Riemann solution at the interface, x/t = 0 (vectorised).

    Used as the Godunov flux by Riemann_SWE with solver_type = 'Exact':
        F_h = h0 vx0,   F_hvx = h0 vx0^2 + g h0^2 / 2,   F_hvy = h0 vx0 vy0.

    Parameters
    ----------
    h_L, h_R, vx_L, vx_R, vy_L, vy_R : float or ndarray   left/right states
    g : float

    Returns
    -------
    h0, vx0, vy0 : float or ndarray   state at x/t = 0

    Notes
    -----
    No error is raised for dry or vacuum-generating data: h* is clamped to
    1e-14, which approximates the wet/dry front.
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
    Exact solution of the 1D SWE Riemann problem at time t > 0.

    Left and right constant states are separated at x = x0; the
    self-similar solution is evaluated at the points x.

    Parameters
    ----------
    h_L, vx_L, vy_L, h_R, vx_R, vy_R : float   left/right states
    g  : float
    x  : array_like   positions
    t  : float        time (> 0)
    x0 : float        initial discontinuity position (default 0.5)

    Returns
    -------
    h, vx, vy : ndarray

    Raises
    ------
    ValueError
        If t <= 0, or if the data separate completely (dry bed between
        the two waves): vx_R - vx_L >= 2 (c_L + c_R).

    Examples
    --------
    Stoker dam break (wet bed), g = 1:

    >>> x = np.linspace(0, 1, 1000)
    >>> h, vx, vy = exact_swe_solution(2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, x, 0.5)
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
#  Quick self-test / demo
#  (from the repository root:  python -m src.models.SWE.SWE_riemann_exact)
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

    plt.suptitle(f'Exact SWE Riemann solver — test cases  (g = {g})', y=1.01)
    plt.tight_layout()
    plt.savefig('swe_riemann_exact.png', dpi=120, bbox_inches='tight')
    plt.show()
    print("Done.  Figure saved to  swe_riemann_exact.png")