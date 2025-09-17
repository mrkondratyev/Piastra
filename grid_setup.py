"""
===============================================================================
grid_setup.py
===============================================================================

Grid module for structured 2D meshes with ghost cells.

This module provides the `Grid` class, which handles the construction of
2D computational grids including ghost zones. It supports multiple
geometries (Cartesian, cylindrical, and polar) and provides methods to
compute cell-centered coordinates, face-centered coordinates, face
areas, and cell volumes.

Once a grid is constructed, numerical solvers can operate on the
finite-volume data (volumes, areas, resolutions) without needing
explicit knowledge of the underlying geometry.

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
    i0r : int
        Starting index for real (non-ghost) cell loops.
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
    geom : str
        Geometry marker: `'cart'`, `'cyl'`, or `'pol'`.
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
        self.fx2 = np.zeros((Nx2 + Ngc * 2 + 1, Nx2 + Ngc * 2), dtype=np.double)

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
        - Computes face coordinates, cell centers, face areas, egdes, and cell volumes.

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
        self.fS1[:, :] = (self.fx2[Ngc:-Ngc+1, Ngc+1:-Ngc] - self.fx2[Ngc:-Ngc+1, Ngc:-Ngc-1])
        self.fS2[:, :] = (self.fx1[Ngc+1:-Ngc, Ngc:-Ngc+1] - self.fx1[Ngc:-Ngc-1, Ngc:-Ngc+1])
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
        for i in range(Nx1 + 1):
            for j in range(Nx2):
                self.fS1[i, j] = fx1[i + Ngc] * dx2[j + Ngc] * 2.0 * np.pi
        for i in range(Nx1):
            for j in range(Nx2 + 1):
                self.fS2[i, j] = (fx1[i+1+Ngc]**2 - fx1[i+Ngc]**2) * np.pi
        self.fS3[:, :] = self.dx1[Ngc:-Ngc, Ngc:-Ngc] * self.dx2[Ngc:-Ngc, Ngc:-Ngc]
        for i in range(Nx1):
            for j in range(Nx2):
                self.cVol[i, j] = (fx1[i+1+Ngc]**2 - fx1[i+Ngc]**2) * dx2[j+Ngc] * np.pi
        
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
        - Face areas, edges, and volumes account for cylindrical geometry.

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
        for i in range(Nx1 + 1):
            for j in range(Nx2):
                self.fS1[i, j] = fx1[i+Ngc] * dx2[j+Ngc]
        for i in range(Nx1):
            for j in range(Nx2 + 1):
                self.fS2[i, j] = (fx1[i+1+Ngc] - fx1[i+Ngc])
        for i in range(Nx1):
            for j in range(Nx2):
                self.cVol[i, j] = (fx1[i+1+Ngc]**2 - fx1[i+Ngc]**2) / 2.0 * dx2[j+Ngc]
        self.fS3[:, :] = self.cVol[:,:]
        
        #grid edges
        self.edg1[:, :] = self.dx1[Ngc:-Ngc, Ngc:Nx2+Ngc+1]
        self.edg2[:, :] = self.dx2[Ngc:Nx1+Ngc+1, Ngc:-Ngc]*self.fx1[Ngc:Nx1+Ngc+1,Ngc:-Ngc]
        self.edg3[:, :] = 1.0
        
