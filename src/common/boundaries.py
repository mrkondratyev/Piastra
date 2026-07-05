# -*- coding: utf-8 -*-
"""
boundaries.py
=============

Boundary condition module for 2D hydrodynamics, diffusion, and MHD simulations.

This module provides functions to fill ghost cells for scalar and vector fields.
Supported boundary types:
    'free'  - non-reflective (zero-gradient) boundary
    'wall'  - reflective (normal component flips) boundary
    'peri'  - periodic boundary
    'axis'  - axis boundary (normal and azimuthal components flip)

Functions
---------
- apply_bc_scalar(var, Ngc, BC_type, axis=1, side='inner'):
      Fill ghost cells for a scalar field along a given axis.
- apply_bc_vector(V1, V2, V3, Ngc, BC_type, axis=1, side='inner'):
      Fill ghost cells for a 3-component vector field along a given axis.
- apply_bc_scalar_Ngc1(var, Ngc, BC_type, axis=1, side='inner', bc_value=0.0):
      Fill the single ghost layer a second-order, 3-point stencil needs
      (e.g. the finite-volume Poisson solver in poisson_solver.py),
      including a true Dirichlet ('dirichlet') boundary value.

Ghost Cell Implementation Note
------------------------------
The approach separates scalar and vector fields for clarity and correctness:

1. ``apply_bc_scalar(var, ...)``
   - For scalar quantities (e.g., density, pressure, temperature).
   - Fills ghost cells along the specified axis according to BC_type.

2. ``apply_bc_vector(V1, V2, V3, ...)``
   - For 3-component vector quantities (e.g., velocity, cell-centered magnetic field).
   - Treats the normal component differently for reflective (wall) boundaries
     while leaving tangential components unchanged.

3. ``apply_bc_fixed(V1, V2, V3, ...)``
   - Pin ghost cells to prescribed (Dirichlet) values on one face

4. ``apply_bc_scalar_Ngc1(var, ...)``
   - Single-ghost-layer scalar filler for second-order, 3-point stencils
     (e.g. the Poisson solver in poisson_solver.py), with true Dirichlet
     ('dirichlet') support via ghost mirroring about a fixed face value.

The face-centered z-electric field (Efld3 along x1/x2) needed for CT MHD is
implemented as a separate function ``boundCond_electric_field`` in the
corresponding MHD solver modules (MHD_step_CT.py, rMHD_step.py).

Author: mrkondratyev
"""

def apply_bc_scalar(var, Ngc, BC_type, axis=1, side='inner'):
    """
    Fill ghost cells for a scalar field along a given axis.

    Parameters
    ----------
    var : np.ndarray
        Scalar field array including ghost cells.
    Ngc : int
        Number of ghost cells.
    BC_type : str
        Boundary type: 'free', 'wall', 'peri', 'axis'.
    axis : int
        Axis along which to apply BC (1 for x1, 2 for x2).
    side : str
        'inner' or 'outer' boundary.

    Returns
    -------
    var : np.ndarray
        Field with ghost cells updated.
    """
    shape = var.shape
    N1, N2 = shape[0], shape[1]

    for i in range(Ngc):
        if axis == 1:  # x1-direction
            if side == 'inner':
                if BC_type == 'free':
                    var[i, :] = var[2*Ngc - 1 - i, :]
                elif BC_type == 'wall':
                    var[i, :] = var[2*Ngc - 1 - i, :]
                elif BC_type == 'peri':
                    var[i, :] = var[N1 - 2*Ngc + i, :]
                elif BC_type == 'axis':
                    var[i, :] = var[2*Ngc - 1 - i, :]
            elif side == 'outer':
                if BC_type == 'free':
                    var[N1 - Ngc + i, :] = var[N1 - Ngc - 1 - i, :]
                elif BC_type == 'wall':
                    var[N1 - Ngc + i, :] = var[N1 - Ngc - 1 - i, :]
                elif BC_type == 'peri':
                    var[N1 - Ngc + i, :] = var[Ngc + i, :]
        elif axis == 2:  # x2-direction
            if side == 'inner':
                if BC_type == 'free':
                    var[:, i] = var[:, 2*Ngc - 1 - i]
                elif BC_type == 'wall':
                    var[:, i] = var[:, 2*Ngc - 1 - i]
                elif BC_type == 'peri':
                    var[:, i] = var[:, N2 - 2*Ngc + i]
                elif BC_type == 'axis':
                    var[:, i] = var[:, 2*Ngc - 1 - i] #spherical axis
            elif side == 'outer':
                if BC_type == 'free':
                    var[:, N2 - Ngc + i] = var[:, N2 - Ngc - 1 - i]
                elif BC_type == 'wall':
                    var[:, N2 - Ngc + i] = var[:, N2 - Ngc - 1 - i]
                elif BC_type == 'peri':
                    var[:, N2 - Ngc + i] = var[:, Ngc + i]
                elif BC_type == 'axis':
                    var[:, N2 - Ngc + i] = var[:, N2 - Ngc - 1 - i] #spherical axis

    return var



def apply_bc_vector(V1, V2, V3, Ngc, BC_type, axis=1, side='inner'):
    """
    Fill ghost cells for a 3-component vector field along a given axis.

    Parameters
    ----------
    V1, V2, V3 : np.ndarray
        Vector field components including ghost cells.
    Ngc : int
        Number of ghost cells.
    BC_type : str
        Boundary type: 'free', 'wall', 'peri', 'axis'.
    axis : int
        Axis along which to apply BC (1 for x1, 2 for x2).
    side : str
        'inner' or 'outer' boundary.

    Returns
    -------
    V1, V2, V3 : np.ndarray
        Vector field components with ghost cells updated.
    """
    shape = V1.shape
    N1, N2 = shape[0], shape[1]
    
    for i in range(Ngc):
        if axis == 1:  # x1-direction
            if side == 'inner':
                if BC_type == 'free':
                    V1[i, :] = V1[2*Ngc - 1 - i, :]
                    V2[i, :] = V2[2*Ngc - 1 - i, :]
                    V3[i, :] = V3[2*Ngc - 1 - i, :]
                elif BC_type == 'wall':
                    V1[i, :] = -V1[2*Ngc - 1 - i, :]  # normal component flips
                    V2[i, :] = V2[2*Ngc - 1 - i, :]
                    V3[i, :] = V3[2*Ngc - 1 - i, :]
                elif BC_type == 'peri':
                    V1[i, :] = V1[N1 - 2*Ngc + i, :]
                    V2[i, :] = V2[N1 - 2*Ngc + i, :]
                    V3[i, :] = V3[N1 - 2*Ngc + i, :]
                elif BC_type == 'axis':
                    V1[i, :] = -V1[2*Ngc - 1 - i, :]  # normal component flips
                    V2[i, :] = V2[2*Ngc - 1 - i, :]
                    V3[i, :] = -V3[2*Ngc - 1 - i, :] # azimuthal component flips
            elif side == 'outer':
                if BC_type == 'free':
                    V1[N1 - Ngc + i, :] = V1[N1 - Ngc - 1 - i, :]
                    V2[N1 - Ngc + i, :] = V2[N1 - Ngc - 1 - i, :]
                    V3[N1 - Ngc + i, :] = V3[N1 - Ngc - 1 - i, :]
                elif BC_type == 'wall':
                    V1[N1 - Ngc + i, :] = -V1[N1 - Ngc - 1 - i, :]
                    V2[N1 - Ngc + i, :] = V2[N1 - Ngc - 1 - i, :]
                    V3[N1 - Ngc + i, :] = V3[N1 - Ngc - 1 - i, :]
                elif BC_type == 'peri':
                    V1[N1 - Ngc + i, :] = V1[Ngc + i, :]
                    V2[N1 - Ngc + i, :] = V2[Ngc + i, :]
                    V3[N1 - Ngc + i, :] = V3[Ngc + i, :]
        elif axis == 2:  # x2-direction
            if side == 'inner':
                if BC_type == 'free':
                    V1[:, i] = V1[:, 2*Ngc - 1 - i]
                    V2[:, i] = V2[:, 2*Ngc - 1 - i]
                    V3[:, i] = V3[:, 2*Ngc - 1 - i]
                elif BC_type == 'wall':
                    V1[:, i] = V1[:, 2*Ngc - 1 - i]
                    V2[:, i] = -V2[:, 2*Ngc - 1 - i]  # normal component flips
                    V3[:, i] = V3[:, 2*Ngc - 1 - i]
                elif BC_type == 'peri':
                    V1[:, i] = V1[:, N2 - 2*Ngc + i]
                    V2[:, i] = V2[:, N2 - 2*Ngc + i]
                    V3[:, i] = V3[:, N2 - 2*Ngc + i]
                elif BC_type == 'axis':
                    V1[:, i] = V1[:, 2*Ngc - 1 - i]
                    V2[:, i] = -V2[:, 2*Ngc - 1 - i]  # normal component flips
                    V3[:, i] = -V3[:, 2*Ngc - 1 - i]  # azimuthal component flips
            elif side == 'outer':
                if BC_type == 'free':
                    V1[:, N2 - Ngc + i] = V1[:, N2 - Ngc - 1 - i]
                    V2[:, N2 - Ngc + i] = V2[:, N2 - Ngc - 1 - i]
                    V3[:, N2 - Ngc + i] = V3[:, N2 - Ngc - 1 - i]
                elif BC_type == 'wall':
                    V1[:, N2 - Ngc + i] = V1[:, N2 - Ngc - 1 - i]
                    V2[:, N2 - Ngc + i] = -V2[:, N2 - Ngc - 1 - i]  # normal component flips
                    V3[:, N2 - Ngc + i] = V3[:, N2 - Ngc - 1 - i]
                elif BC_type == 'peri':
                    V1[:, N2 - Ngc + i] = V1[:, Ngc + i]
                    V2[:, N2 - Ngc + i] = V2[:, Ngc + i]
                    V3[:, N2 - Ngc + i] = V3[:, Ngc + i]
                elif BC_type == 'axis':
                    V1[:, N2 - Ngc + i] = V1[:, N2 - Ngc - 1 - i]
                    V2[:, N2 - Ngc + i] = -V2[:, N2 - Ngc - 1 - i]  # normal component flips
                    V3[:, N2 - Ngc + i] = -V3[:, N2 - Ngc - 1 - i]  # azimuthal component flips

    return V1, V2, V3



def apply_bc_fixed(state_fields, Ngc, N1, N2, face, patches):
    """
    Pin ghost cells to prescribed (Dirichlet) values on one face.

    Applied AFTER the standard apply_bc_* fill, so the fixed region overwrites
    the zero-gradient/reflecting ghost values there. Because the prescribed
    state lives in the ghost cells, the boundary-face Riemann solve produces
    the correct inlet flux with NO change to the flux routine.

    Parameters
    ----------
    state_fields : dict[str, ndarray]
        Maps field name -> full array (with ghosts), e.g.
        {'dens': fluid.dens, 'vel1': fluid.vel1, ...}. Only fields named in a
        patch's dict are touched.
    Ngc : int                  number of ghost cells.
    N1, N2 : int               full array sizes (incl. ghosts) along axis 0, 1.
    face : int                 0=x1_inner, 1=x2_inner, 2=x1_outer, 3=x2_outer.
    patches : list of (start, end, dict)
        Each tuple gives an INTERIOR index range [start, end) along the
        boundary (0-based from the first interior cell) and a {field: value}
        dict of prescribed values.
    """
    for (start, end, sdict) in patches:
        t0 = Ngc + start          # interior index -> full-array index (tangential)
        t1 = Ngc + end

        for name, value in sdict.items():
            if name not in state_fields:
                continue
            arr = state_fields[name]

            if face == 0:            # x1 inner: ghost rows [0:Ngc], tangential = x2
                arr[0:Ngc, t0:t1] = value
            elif face == 2:          # x1 outer: ghost rows [N1-Ngc:N1]
                arr[N1 - Ngc:N1, t0:t1] = value
            elif face == 1:          # x2 inner: ghost cols [0:Ngc], tangential = x1
                arr[t0:t1, 0:Ngc] = value
            elif face == 3:          # x2 outer: ghost cols [N2-Ngc:N2]
                arr[t0:t1, N2 - Ngc:N2] = value
    return state_fields



def apply_bc_scalar_Ngc1(var, Ngc, BC_type, axis=1, side='inner', bc_value=0.0):
    """
    Fill the single ghost-cell layer bordering the domain, as needed by a
    second-order, 3-point finite-volume stencil (e.g. the Laplacian
    operator in poisson_solver.py).

    Unlike ``apply_bc_scalar`` (which fills all ``Ngc`` ghost layers, as
    required by high-order hyperbolic reconstruction stencils), this only
    touches the ONE layer immediately next to the real domain -- all a
    centered 3-point stencil ever reads. That makes it equally usable on
    a dedicated Ngc=1 grid and on a deeper-ghosted grid borrowed from a
    hyperbolic solver (e.g. an Ngc=2/3 HD/MHD grid reused for
    self-gravity or magnetic divergence cleaning): pass that grid's own
    Ngc and only the layer bordering the domain is updated, leaving any
    deeper ghost layers untouched.

    Parameters
    ----------
    var : np.ndarray
        Scalar field, shape (N1, N2), with at least one ghost cell on
        every side.
    Ngc : int
        Number of ghost cells `var` carries on each side. Only the layer
        bordering the real domain (index Ngc-1 / N-Ngc) is filled.
    BC_type : str
        'peri'      - periodic.
        'free'      - zero-gradient (homogeneous Neumann): ghost set
                      equal to the nearest interior cell.
        'dirichlet' - fixed boundary VALUE `bc_value`: the ghost cell is
                      mirrored about it, ``ghost = 2*bc_value - interior``,
                      so the face-averaged value is exactly `bc_value` to
                      second order.
    axis : int
        Axis along which to apply the BC (1 for x1, 2 for x2).
    side : str
        'inner' or 'outer'.
    bc_value : float or ndarray
        Prescribed boundary value for BC_type='dirichlet'. Broadcastable
        against the tangential dimension of the face. Ignored otherwise.

    Returns
    -------
    var : np.ndarray
        Field with the boundary-adjacent ghost layer updated.
    """
    N1, N2 = var.shape

    if axis == 1:  # x1-direction
        if side == 'inner':
            if BC_type == 'free':
                var[Ngc - 1, :] = var[Ngc, :]
            elif BC_type == 'peri':
                var[Ngc - 1, :] = var[N1 - Ngc - 1, :]
            elif BC_type == 'dirichlet':
                var[Ngc - 1, :] = 2.0 * bc_value - var[Ngc, :]
            else:
                raise ValueError(
                    f"Invalid BC_type: '{BC_type}'. "
                    f"Expected 'free', 'peri', or 'dirichlet'.")
        elif side == 'outer':
            if BC_type == 'free':
                var[N1 - Ngc, :] = var[N1 - Ngc - 1, :]
            elif BC_type == 'peri':
                var[N1 - Ngc, :] = var[Ngc, :]
            elif BC_type == 'dirichlet':
                var[N1 - Ngc, :] = 2.0 * bc_value - var[N1 - Ngc - 1, :]
            else:
                raise ValueError(
                    f"Invalid BC_type: '{BC_type}'. "
                    f"Expected 'free', 'peri', or 'dirichlet'.")
    elif axis == 2:  # x2-direction
        if side == 'inner':
            if BC_type == 'free':
                var[:, Ngc - 1] = var[:, Ngc]
            elif BC_type == 'peri':
                var[:, Ngc - 1] = var[:, N2 - Ngc - 1]
            elif BC_type == 'dirichlet':
                var[:, Ngc - 1] = 2.0 * bc_value - var[:, Ngc]
            else:
                raise ValueError(
                    f"Invalid BC_type: '{BC_type}'. "
                    f"Expected 'free', 'peri', or 'dirichlet'.")
        elif side == 'outer':
            if BC_type == 'free':
                var[:, N2 - Ngc] = var[:, N2 - Ngc - 1]
            elif BC_type == 'peri':
                var[:, N2 - Ngc] = var[:, Ngc]
            elif BC_type == 'dirichlet':
                var[:, N2 - Ngc] = 2.0 * bc_value - var[:, N2 - Ngc - 1]
            else:
                raise ValueError(
                    f"Invalid BC_type: '{BC_type}'. "
                    f"Expected 'free', 'peri', or 'dirichlet'.")
    else:
        raise ValueError(f"Invalid axis: {axis}. Expected 1 or 2.")

    return var
