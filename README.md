# Piastra2D

**Piastra2D** is a teaching-oriented code for solving:
- Linear advection  
- Inviscid compressible hydrodynamics (HD)  
- Magnetohydrodynamics (MHD)  

within a **finite-volume Godunov-type framework** with **TVD Runge–Kutta time integration**.  
The code is written in **Python** with extensive use of **NumPy**, and includes tools for visualization with **matplotlib**.  

---

## Features

- **Dimensionality**: 1D and 2D Cartesian uniform grids  
- **Finite-volume solver** with approximate Riemann solvers for fluxes  
- **Hydrodynamics solvers**:
  - LLF (Rusanov, 1961)  
  - HLL (Harten–Lax–van Leer, 1983)  
  - HLLC (Toro et al., 1994)  
  - Roe (Roe, 1981)  
- **MHD solvers**:
  - LLF, HLL, HLLD  
  - Divergence control:  
    - Powell 8-wave method (Powell 1994, 1999; Tóth 2000)  
    - Constrained Transport (Flux-CT; Balsara & Spicer 1999)  
- **Advection**: high-order RK with Riemann solver or Lax–Wendroff scheme  
- **Reconstruction methods**:  
  - PCM (piecewise constant)  
  - PLM (piecewise linear with slope limiter, 2nd order)  
  - PPM (Colella & Woodward 1984, Mignone 2014)  
  - PPMorig (original Colella & Woodward version)  
  - WENO5 (5th order, Jiang & Shu 1996)  
- **Time integration**:  
  - RK1 (Euler)  
  - RK2, RK3 (TVD Runge–Kutta; Shu & Osher 1988)  
- **Ghost cells**: automatically adjusted (2 for PLM/PCM, 3 for PPM/WENO)  
- **Simulation control** through the `Parameters` class with defaults and validation  
- **Modular design**:
  - `parameters.py`: central parameter container  
  - `grid_setup.py`: structured grid definition  
  - `sim_state.py`: storage for fluid variables  
  - `helpers.py`: initial condition dispatch and simulation loop  
  - `advection_one_step.py`, `hydro_one_step.py`, `MHD_one_step_CT.py`, `MHD_one_step_8wave.py`: solver backends  
  - `visualization.py`: plotting utilities  

---

## Usage

**Requirements:**
- Python 3.9+  
- NumPy  
- matplotlib  
- IPython (recommended for notebooks)  

**Example (main.py)**

```python
from parameters import Parameters
from grid_setup import Grid
from sim_state import SimState
from helpers import initial_model, run_simulation
from hydro_one_step import Hydro2D

# Setup parameters
par = Parameters(mode="HD", problem="sod2Dcart", Nx1=64, Nx2=64)

# Setup grid and state
grid = Grid(par.Nx1, par.Nx2, par.Ngc)
simstate = SimState(grid, par)
grid, simstate, par, eos = initial_model(grid, simstate, par)

# Run solver
solver = Hydro2D(grid, simstate, eos, par)
simstate, par.timenow = run_simulation(grid, simstate, par, solver, simstate.dens, nsteps=200)
