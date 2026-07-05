"""
===============================================================================
grid_setup.py
===============================================================================

Grid module for structured 2D meshes with ghost cells.

This module provides the `Grid` class, which handles the construction of
2D computational grids including ghost zones. It supports multiple
geometries (Cartesian, cylindrical, polar, and spherical) and provides methods to
compute cell-centered coordinates, face-centered coordinates, face
areas, and cell volumes.

Once a grid is constructed, numerical solvers can operate on the
finite-volume data (volumes, areas, resolutions) without needing
explicit knowledge of the underlying geometry.

Additionally, the module provides "reconstruct_grid" function, which can be used to 
reload the grid data after the restart.
Author: mrkondratyev
"""

import numpy as np


class Grid:
    """
    Class for representing a 2D computational grid with ghost cells.

    Attributes
    ----------
    Nx1 : int
        Number of real (non-ghost) cells in the first dimension.
    Nx2 : int
        Number of real (non-ghost) cells in the second dimension.
    Ngc : int
        Number of ghost cells on each boundary.
    Nx1r : int
        Final index (exclusive) for real cells in the first dimension.
    Nx2r : int
        Final index (exclusive) for real cells in the second dimension.
    grid_shape : tuple of int
        Shape of arrays including ghost cells.
    fS1 : ndarray
        Face areas perpendicular to the first dimension.
    fS2 : ndarray
        Face areas perpendicular to the second dimension.
    fS3 : ndarray
        Face areas perpendicular to the third dimension (needed for CT MHD).
    cVol : ndarray
        Cell volumes.
    fx1 : ndarray
        Face coordinates in the first dimension.
    fx2 : ndarray
        Face coordinates in the second dimension.
    cx1 : ndarray
        Cell center coordinates in the first dimension.
    cx2 : ndarray
        Cell center coordinates in the second dimension.
    dx1 : ndarray
        Local grid spacing in the first dimension.
    dx2 : ndarray
        Local grid spacing in the second dimension.
    ax1 : ndarray
        Volumetric centroid coordinate in the first dimension.
    ax2 : ndarray
        Volumetric centroid coordinate in the second dimension.
    edg1 : ndarray
        Grid edges along the first dimension.
    edg2 : ndarray
        Grid edges along the second dimension.
    edg3 : ndarray
        Grid edges along the third dimension (needed for CT MHD).
    hx2 : ndarray, shape grid_shape
        Lamé coefficient (metric scale factor) for the x2 coordinate
        direction.  Equal to 1 for Cartesian and cylindrical grids where
        the x2 coordinate already measures physical arc length; equal to
        the radial coordinate `cx1` (R for polar, r for spherical-polar)
        where the physical arc length is h₂ · dx2.  Using this array
        allows geometry-aware operators to be written without branching:
        ``∂f/∂(arc) = ∂f/∂x2 / hx2``.
    geom : str
        Geometry marker: `'cart'`, `'cyl'`, `'pol'`, or `'sph'`.
    """


    def __init__(self, Nx1, Nx2, Ngc):
        """
        Initialize grid object with dimensions and ghost cells.

        Parameters
        ----------
        Nx1 : int
            Number of real cells in the first dimension.
        Nx2 : int
            Number of real cells in the second dimension.
        Ngc : int
            Number of ghost cells on each boundary.
        """
        
        # Number of cells in each direction and ghost zones number (32-bit integers)
        self.Nx1, self.Nx2, self.Ngc = np.int32(Nx1), np.int32(Nx2), np.int32(Ngc)

        # Indices for looping over *real* cells
        self.Nx1r = Nx1 + Ngc
        self.Nx2r = Nx2 + Ngc

        # Full grid shape including ghost zones
        self.grid_shape = (Nx1 + Ngc * 2, Nx2 + Ngc * 2)

        # Allocate arrays for geometry
        self.fS1 = np.zeros((Nx1 + 1, Nx2), dtype=np.double)   # face areas ⟂ x1
        self.fS2 = np.zeros((Nx1, Nx2 + 1), dtype=np.double)   # face areas ⟂ x2
        self.fS3 = np.zeros((Nx1, Nx2), dtype=np.double)       # face areas ⟂ x3 
        self.cVol = np.zeros((Nx1, Nx2), dtype=np.double)      # cell volumes

        # Face coordinates
        self.fx1 = np.zeros((Nx1 + Ngc * 2 + 1, Nx2 + Ngc * 2), dtype=np.double)
        self.fx2 = np.zeros((Nx1 + Ngc * 2, Nx2 + Ngc * 2 + 1), dtype=np.double)

        # Cell center coordinates
        self.cx1 = np.zeros((Nx1 + Ngc * 2, Nx2 + Ngc * 2), dtype=np.double)
        self.cx2 = np.zeros((Nx1 + Ngc * 2, Nx2 + Ngc * 2), dtype=np.double)

        # Grid resolution
        self.dx1 = np.zeros((Nx1 + Ngc * 2, Nx2 + Ngc * 2), dtype=np.double)
        self.dx2 = np.zeros((Nx1 + Ngc * 2, Nx2 + Ngc * 2), dtype=np.double)

        # Volumetric centroids
        self.ax1 = np.zeros((Nx1 + Ngc * 2, Nx2 + Ngc * 2), dtype=np.double)
        self.ax2 = np.zeros((Nx1 + Ngc * 2, Nx2 + Ngc * 2), dtype=np.double)
        
        #grid edges (needed for CT MHD)
        self.edg1 = np.zeros((Nx1, Nx2 + 1), dtype=np.double)
        self.edg2 = np.zeros((Nx1 + 1, Nx2), dtype=np.double)
        self.edg3 = np.zeros((Nx1 + 1, Nx2 + 1), dtype=np.double)

        # Lamé coefficient for the x2 direction (= 1 for cart/cyl; set to
        # cx1 by PolarGrid / SphericalPolarGrid after cx1 is constructed)
        self.hx2 = np.ones((Nx1 + Ngc * 2, Nx2 + Ngc * 2), dtype=np.double)



    def CartesianGrid(self, x1ini, x1fin, x2ini, x2fin):
        """
        Construct a uniform Cartesian grid.

        Parameters
        ----------
        x1ini : float
            Start of domain in the first dimension.
        x1fin : float
            End of domain in the first dimension.
        x2ini : float
            Start of domain in the second dimension.
        x2fin : float
            End of domain in the second dimension.

        Notes
        -----
        - Grid spacing is uniform in both directions.
        - Computes face coordinates, cell centers, face areas, edges, and cell volumes.

        Examples
        --------
        >>> g = Grid(64, 32, 2)
        >>> g.CartesianGrid(0.0, 1.0, -1.0, 1.0)
        >>> g.cVol.shape
        (64, 32)
        """
        # Geometry marker
        self.geom = 'cart'

        # Store domain bounds as floats
        self.x1ini, self.x1fin = np.double(x1ini), np.double(x1fin)
        self.x2ini, self.x2fin = np.double(x2ini), np.double(x2fin)

        Nx1, Nx2, Ngc = self.Nx1, self.Nx2, self.Ngc

        # Uniform grid resolution
        dx1uc = (x1fin - x1ini) / Nx1
        dx2uc = (x2fin - x2ini) / Nx2
        self.dx1uc, self.dx2uc = dx1uc, dx2uc

        # Fill grid spacing arrays
        dx1 = np.full(Nx1 + Ngc * 2, dx1uc, dtype=np.double)
        dx2 = np.full(Nx2 + Ngc * 2, dx2uc, dtype=np.double)
        self.dx1 = np.tile(dx1, (Nx2 + Ngc * 2, 1)).T
        self.dx2 = np.tile(dx2, (Nx1 + Ngc * 2, 1))

        # Face coordinates
        fx1 = np.linspace(x1ini - Ngc * dx1uc, x1fin + Ngc * dx1uc, Nx1 + Ngc * 2 + 1)
        fx2 = np.linspace(x2ini - Ngc * dx2uc, x2fin + Ngc * dx2uc, Nx2 + Ngc * 2 + 1)
        self.fx1 = np.tile(fx1, (Nx2 + Ngc * 2, 1)).T
        self.fx2 = np.tile(fx2, (Nx1 + Ngc * 2, 1))

        # Cell center coordinates
        cx1 = np.linspace(x1ini - (Ngc - 0.5) * dx1uc, x1fin + (Ngc - 0.5) * dx1uc, Nx1 + Ngc * 2)
        cx2 = np.linspace(x2ini - (Ngc - 0.5) * dx2uc, x2fin + (Ngc - 0.5) * dx2uc, Nx2 + Ngc * 2)
        self.cx1 = np.tile(cx1, (Nx2 + Ngc * 2, 1)).T
        self.cx2 = np.tile(cx2, (Nx1 + Ngc * 2, 1))

        # Volumetric centroids (same as centers for Cartesian grid)
        self.ax1 = self.cx1
        self.ax2 = self.cx2

        # Face areas
        self.fS1[:, :] = (self.fx2[Ngc:Nx1+Ngc+1, Ngc+1:-Ngc] - self.fx2[Ngc:Nx1+Ngc+1, Ngc:-Ngc-1])
        self.fS2[:, :] = (self.fx1[Ngc+1:-Ngc, Ngc:Nx2+Ngc+1] - self.fx1[Ngc:-Ngc-1, Ngc:Nx2+Ngc+1])
        self.fS3[:, :] = self.dx1[Ngc:-Ngc, Ngc:-Ngc] * self.dx2[Ngc:-Ngc, Ngc:-Ngc]
        # Cell volumes
        self.cVol[:, :] = self.dx1[Ngc:-Ngc, Ngc:-Ngc] * self.dx2[Ngc:-Ngc, Ngc:-Ngc]
        
        #grid edges
        self.edg1[:, :] = self.dx1[Ngc:-Ngc, Ngc:Nx2+Ngc+1]
        self.edg2[:, :] = self.dx2[Ngc:Nx1+Ngc+1, Ngc:-Ngc]
        self.edg3[:, :] = 1.0



    def CylindricalGrid(self, x1ini, x1fin, x2ini, x2fin):
        """
        Construct a uniform cylindrical (R, Z) grid.

        Parameters
        ----------
        x1ini : float
            Start of domain in radial direction (R).
        x1fin : float
            End of domain in radial direction (R).
        x2ini : float
            Start of domain in axial direction (Z).
        x2fin : float
            End of domain in axial direction (Z).

        Notes
        -----
        - Radial integrals and surfaces account for 2π azimuthal symmetry.
        - Face areas, edges and volumes are computed analytically.

        Examples
        --------
        >>> g = Grid(32, 64, 2)
        >>> g.CylindricalGrid(0.0, 1.0, -2.0, 2.0)
        >>> g.cVol[0, 0] > 0
        True
        """
        # Geometry marker
        self.geom = 'cyl'
        self.x1ini, self.x1fin = np.double(x1ini), np.double(x1fin)
        self.x2ini, self.x2fin = np.double(x2ini), np.double(x2fin)

        Nx1, Nx2, Ngc = self.Nx1, self.Nx2, self.Ngc

        # Uniform grid resolution
        dx1uc = (x1fin - x1ini) / Nx1
        dx2uc = (x2fin - x2ini) / Nx2
        dx1 = np.full(Nx1 + Ngc * 2, dx1uc, dtype=np.double)
        dx2 = np.full(Nx2 + Ngc * 2, dx2uc, dtype=np.double)
        self.dx1 = np.tile(dx1, (Nx2 + Ngc * 2, 1)).T
        self.dx2 = np.tile(dx2, (Nx1 + Ngc * 2, 1))

        # Face coordinates
        fx1 = np.linspace(x1ini - Ngc * dx1uc, x1fin + Ngc * dx1uc, Nx1 + Ngc * 2 + 1)
        fx2 = np.linspace(x2ini - Ngc * dx2uc, x2fin + Ngc * dx2uc, Nx2 + Ngc * 2 + 1)
        self.fx1 = np.tile(fx1, (Nx2 + Ngc * 2, 1)).T
        self.fx2 = np.tile(fx2, (Nx1 + Ngc * 2, 1))

        # Cell centers
        cx1 = np.linspace(x1ini - (Ngc - 0.5) * dx1uc, x1fin + (Ngc - 0.5) * dx1uc, Nx1 + Ngc * 2)
        cx2 = np.linspace(x2ini - (Ngc - 0.5) * dx2uc, x2fin + (Ngc - 0.5) * dx2uc, Nx2 + Ngc * 2)
        self.cx1 = np.tile(cx1, (Nx2 + Ngc * 2, 1)).T
        self.cx2 = np.tile(cx2, (Nx1 + Ngc * 2, 1))

        # Volumetric centroids
        self.ax1 = 2.0 * (self.fx1[1:, :]**3 - self.fx1[:-1, :]**3) / (self.fx1[1:, :]**2 - self.fx1[:-1, :]**2) / 3.0
        self.ax2 = self.cx2
        
        # Face areas and volumes
        r_f   = fx1[Ngc:Ngc+Nx1+1]                      # radial faces      (Nx1+1,)
        dz    = dx2[Ngc:Ngc+Nx2]                        # axial widths      (Nx2,)
        r2dif = fx1[Ngc+1:Ngc+Nx1+1]**2 - fx1[Ngc:Ngc+Nx1]**2   # r²₊ − r²₋ (Nx1,)
        self.fS1[:, :]  = r_f[:, None] * dz[None, :] * 2.0*np.pi
        self.fS2[:, :]  = (r2dif * np.pi)[:, None]      # independent of j
        self.fS3[:, :]  = self.dx1[Ngc:-Ngc, Ngc:-Ngc] * self.dx2[Ngc:-Ngc, Ngc:-Ngc]
        self.cVol[:, :] = r2dif[:, None] * dz[None, :] * np.pi
        
        #grid edges
        self.edg1[:, :] = self.dx1[Ngc:-Ngc, Ngc:Nx2+Ngc+1]
        self.edg2[:, :] = self.dx2[Ngc:Nx1+Ngc+1, Ngc:-Ngc]
        self.edg3[:, :] = self.fx1[Ngc:Nx1+Ngc+1,Ngc:Nx2+Ngc+1]*2.0*np.pi
        
        

    def PolarGrid(self, x1ini, x1fin, x2ini, x2fin):
        """
        Construct a uniform polar (R, φ) grid.

        Parameters
        ----------
        x1ini : float
            Start of domain in radial direction (R).
        x1fin : float
            End of domain in radial direction (R).
        x2ini : float
            Start of domain in angular direction (φ).
        x2fin : float
            End of domain in angular direction (φ).

        Notes
        -----
        - Radial integrals use analytic formulas for volumetric centroids.
        - Face areas, edges, and volumes account for polar geometry.

        Examples
        --------
        >>> g = Grid(16, 32, 2)
        >>> g.PolarGrid(0.0, 1.0, 0.0, np.pi)
        >>> g.fS1[0, 0] > 0
        True
        """
        # Geometry marker
        self.geom = 'pol'
        self.x1ini, self.x1fin = np.double(x1ini), np.double(x1fin)
        self.x2ini, self.x2fin = np.double(x2ini), np.double(x2fin)

        Nx1, Nx2, Ngc = self.Nx1, self.Nx2, self.Ngc

        # Uniform resolution
        dx1uc = (x1fin - x1ini) / Nx1
        dx2uc = (x2fin - x2ini) / Nx2
        self.dx1uc, self.dx2uc = dx1uc, dx2uc
        dx1 = np.full(Nx1 + Ngc * 2, dx1uc, dtype=np.double)
        dx2 = np.full(Nx2 + Ngc * 2, dx2uc, dtype=np.double)
        self.dx1 = np.tile(dx1, (Nx2 + Ngc * 2, 1)).T
        self.dx2 = np.tile(dx2, (Nx1 + Ngc * 2, 1))

        # Face coordinates
        fx1 = np.linspace(x1ini - Ngc * dx1uc, x1fin + Ngc * dx1uc, Nx1 + Ngc * 2 + 1)
        fx2 = np.linspace(x2ini - Ngc * dx2uc, x2fin + Ngc * dx2uc, Nx2 + Ngc * 2 + 1)
        self.fx1 = np.tile(fx1, (Nx2 + Ngc * 2, 1)).T
        self.fx2 = np.tile(fx2, (Nx1 + Ngc * 2, 1))

        # Cell centers
        cx1 = np.linspace(x1ini - (Ngc - 0.5) * dx1uc, x1fin + (Ngc - 0.5) * dx1uc, Nx1 + Ngc * 2)
        cx2 = np.linspace(x2ini - (Ngc - 0.5) * dx2uc, x2fin + (Ngc - 0.5) * dx2uc, Nx2 + Ngc * 2)
        self.cx1 = np.tile(cx1, (Nx2 + Ngc * 2, 1)).T
        self.cx2 = np.tile(cx2, (Nx1 + Ngc * 2, 1))

        # Volumetric centroids
        self.ax1 = 2.0 * (self.fx1[1:, :]**3 - self.fx1[:-1, :]**3) / (self.fx1[1:, :]**2 - self.fx1[:-1, :]**2) / 3.0
        self.ax2 = self.cx2

        # Face areas and volumes
        r_f   = fx1[Ngc:Ngc+Nx1+1]                      # radial faces      (Nx1+1,)
        dphi  = dx2[Ngc:Ngc+Nx2]                        # angular widths    (Nx2,)
        dr    = fx1[Ngc+1:Ngc+Nx1+1] - fx1[Ngc:Ngc+Nx1]         # r₊ − r₋   (Nx1,)
        r2dif = fx1[Ngc+1:Ngc+Nx1+1]**2 - fx1[Ngc:Ngc+Nx1]**2   # r²₊ − r²₋ (Nx1,)
        self.fS1[:, :]  = r_f[:, None] * dphi[None, :] * 1.0 # dz = 1
        self.fS2[:, :]  = dr[:, None]                   # independent of j
        self.cVol[:, :] = (r2dif / 2.0)[:, None] * dphi[None, :] * 1.0 # dz = 1
        self.fS3[:, :]  = (r2dif / 2.0)[:, None] * dphi[None, :]
        
        #grid edges
        self.edg1[:, :] = self.dx1[Ngc:-Ngc, Ngc:Nx2+Ngc+1]
        self.edg2[:, :] = self.dx2[Ngc:Nx1+Ngc+1, Ngc:-Ngc]*self.fx1[Ngc:Nx1+Ngc+1,Ngc:-Ngc]
        self.edg3[:, :] = 1.0

        # Lamé coefficient: physical arc length per unit φ-coordinate is R
        self.hx2 = self.cx1.copy()



    def SphericalPolarGrid(self, x1ini, x1fin, x2ini, x2fin):
        """
        Construct a uniform spherical-polar (r, θ) grid.

        Parameters
        ----------
        x1ini : float
            Start of domain in radial direction (r).
        x1fin : float
            End of domain in radial direction (r).
        x2ini : float
            Start of domain in lateral angle direction (θ, from north pole).
        x2fin : float
            End of domain in lateral angle direction (θ).

        Notes
        -----
        - 2π azimuthal symmetry is assumed; face areas and volumes are
          integrated over the full φ ∈ [0, 2π) ring.
        - Face areas and cell volumes use exact analytic integrals:
          fS1[i,j] = 2π r_i² (cos θ_j − cos θ_{j+1})   (face ⟂ r)
          fS2[i,j] = π sin(θ_j) (r_{i+1}² − r_i²)       (face ⟂ θ)
          cVol[i,j] = 2π/3 (r_{i+1}³ − r_i³)(cos θ_j − cos θ_{j+1})
        - Volumetric centroids in r: ax1 = 3(r_{i+1}⁴−r_i⁴) / (4(r_{i+1}³−r_i³))
        - Volumetric centroids in θ: ax2 from ∫θ sin θ dθ / ∫sin θ dθ.

        Examples
        --------
        >>> g = Grid(32, 64, 1)
        >>> g.SphericalPolarGrid(1.0, 2.0, 0.0, np.pi)
        >>> g.cVol[0, 0] > 0
        True
        """
        # Geometry marker
        self.geom = 'sph'
        self.x1ini, self.x1fin = np.double(x1ini), np.double(x1fin)
        self.x2ini, self.x2fin = np.double(x2ini), np.double(x2fin)

        Nx1, Nx2, Ngc = self.Nx1, self.Nx2, self.Ngc

        # Uniform resolution in (r, θ)
        dx1uc = (x1fin - x1ini) / Nx1
        dx2uc = (x2fin - x2ini) / Nx2
        self.dx1uc, self.dx2uc = dx1uc, dx2uc
        dx1 = np.full(Nx1 + Ngc * 2, dx1uc, dtype=np.double)
        dx2 = np.full(Nx2 + Ngc * 2, dx2uc, dtype=np.double)
        self.dx1 = np.tile(dx1, (Nx2 + Ngc * 2, 1)).T
        self.dx2 = np.tile(dx2, (Nx1 + Ngc * 2, 1))

        # Face coordinates
        fx1 = np.linspace(x1ini - Ngc * dx1uc, x1fin + Ngc * dx1uc, Nx1 + Ngc * 2 + 1)
        fx2 = np.linspace(x2ini - Ngc * dx2uc, x2fin + Ngc * dx2uc, Nx2 + Ngc * 2 + 1)
        self.fx1 = np.tile(fx1, (Nx2 + Ngc * 2, 1)).T
        self.fx2 = np.tile(fx2, (Nx1 + Ngc * 2, 1))

        #helpers for radial face coordinates
        r_lo = self.fx1[:-1, :]
        r_hi = self.fx1[1:,  :]
        # Cell center in r
        self.cx1 = 2.0 * (r_hi**3 - r_lo**3) / (3.0 * (r_hi**2 - r_lo**2))
        # Volumetric centroid in r: ∫r·r²dr / ∫r²dr = 3(r⁴₊−r⁴₋) / (4(r³₊−r³₋))
        self.ax1 = 3.0 * (r_hi**4 - r_lo**4) / (4.0 * (r_hi**3 - r_lo**3))

        #helpers for θ face coordinates
        th_lo = self.fx2[:, :-1]
        th_hi = self.fx2[:, 1:]
        # Cell center in θ
        self.cx2 = (th_hi + th_lo) / 2.0
        # Volumetric centroid in θ: ∫θ·sinθ dθ / ∫sinθ dθ
        # Antiderivative of θ sinθ is sinθ − θ cosθ.
        num_ax2 = (np.sin(th_hi) - th_hi * np.cos(th_hi)) - \
                  (np.sin(th_lo) - th_lo * np.cos(th_lo))
        den_ax2 = np.cos(th_lo) - np.cos(th_hi)
        den_ax2 = np.where(np.abs(den_ax2) > 1e-14, den_ax2, 1e-14)
        self.ax2 = num_ax2 / den_ax2

        # Face areas and cell volumes
        r_f    = fx1[Ngc:Ngc+Nx1+1]                     # radial faces      (Nx1+1,)
        th_f   = fx2[Ngc:Ngc+Nx2+1]                     # polar-angle faces (Nx2+1,)
        r2dif  = fx1[Ngc+1:Ngc+Nx1+1]**2 - fx1[Ngc:Ngc+Nx1]**2   # (Nx1,)
        r3dif  = fx1[Ngc+1:Ngc+Nx1+1]**3 - fx1[Ngc:Ngc+Nx1]**3   # (Nx1,)
        cosdif = np.cos(th_f[:-1]) - np.cos(th_f[1:])   # cosθ_j − cosθ_{j+1} (Nx2,)
        sin_lo = np.sin(th_f)                           # sinθ at lower face  (Nx2+1,)
        # radial face area: 2π r² (cosθ_j − cosθ_{j+1})
        self.fS1[:, :]  = 2.0*np.pi * (r_f**2)[:, None] * cosdif[None, :]
        # polar-angle face area: π sinθ (r²₊ − r²₋)
        self.fS2[:, :]  = np.pi * sin_lo[None, :] * r2dif[:, None]
        # cell volume: 2π/3 (r³₊ − r³₋)(cosθ_j − cosθ_{j+1})
        self.cVol[:, :] = 2.0*np.pi/3.0 * r3dif[:, None] * cosdif[None, :]
        # azimuthal (r-θ) face area placeholder for CT MHD
        self.fS3[:, :]  = 0.5 * r2dif[:, None] * dx2uc

        # Grid edges (for CT MHD)
        # edg1: radial edge length (dr) at each (real cell, θ-face) point
        self.edg1[:, :] = self.dx1[Ngc:-Ngc, Ngc:Nx2 + Ngc + 1]
        # edg2: polar-arc edge length (r dθ) at each (r-face, real cell) point
        self.edg2[:, :] = (self.dx2[Ngc:Nx1 + Ngc + 1, Ngc:-Ngc] *
                           self.fx1[Ngc:Nx1 + Ngc + 1, Ngc:-Ngc])
        # edg3: azimuthal circumference (2π r sinθ) at each grid node
        self.edg3[:, :] = (2.0 * np.pi *
                           self.fx1[Ngc:Nx1 + Ngc + 1, Ngc:Nx2 + Ngc + 1] *
                           np.sin(self.fx2[Ngc:Nx1 + Ngc + 1, Ngc:Nx2 + Ngc + 1]))

        # Lamé coefficient: physical arc length per unit θ-coordinate is r
        self.hx2 = self.cx1.copy()



def reconstruct_grid(Nx1, Nx2, Ngc, geom, x1ini, x1fin, x2ini, x2fin):
    """
    Rebuild a Grid from its construction metadata by re-running the matching
    geometry constructor.

    Parameters
    ----------
    Nx1, Nx2, Ngc : int
    geom : {'cart', 'cyl', 'pol', 'sph'}
    x1ini, x1fin, x2ini, x2fin : float
        Domain bounds passed to the constructor.

    Returns
    -------
    Grid
    """
    g = Grid(Nx1, Nx2, Ngc)
    builders = {
        'cart': g.CartesianGrid,
        'cyl':  g.CylindricalGrid,
        'pol':  g.PolarGrid,
        'sph':  g.SphericalPolarGrid,
    }
    if geom not in builders:
        raise ValueError(f"reconstruct_grid: unknown geometry '{geom}'. "
                         f"Expected one of {sorted(builders)}.")
    builders[geom](x1ini, x1fin, x2ini, x2fin)
    return g
    
