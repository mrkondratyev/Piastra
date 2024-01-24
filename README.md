# Piastra2D
Piastra2D is a simple code, which solves inviscid compressible hydrodynamics equations within a finite volume framework, using high order Godunov-type methods with TVD Runge-Kutta timestepping. It is written for teaching purposes in Python with an extensive usage of NumPy library.

By now, Piastra2D can solve 1D and 2D compressible hydro equations on a structured uniform Cartesian grid. The solver utilizes a finite volume approach with approximate Riemann solver solution to obtain fluxes of conservative state variables on cell faces. Users can choose between Local Lax-Friedrichs (Rusanov (1961)), Harten-Lax-van Leer (HLL; Harten, Lax and van Leer (1983)) and HLLC (Toro et al (1994)) flux functions. To increase the order of approximation in space, the code uses piecewise-linear method with slope limiter (second order of accuracy in space) or Weighted Essentially Non-Oscillating method WENO5 (fifth order in space), as well as Piecewise Parabolic Method PPM (third order in space). To achieve a better accuracy in time, the code utilizes TVD multistage Runge-Kutta timestepping (methods RK2 and RK3 by Shu and Osher (1988), as well as RK1 with the first order of accuracy in time).   

To use the code, you should have Python 3 with NumPy installed. You can adjust the parameters of the simulation by varying the variables "aux.rec_type", "aux.flux_type" and "aux.RK_order" in the main file "piastra2D_main.py".
By default, I set [aux.rec_type = 'PPM', aux.flux_type = 'HLLC', aux.RK_order = 'RK3']. 

To set the initial conditions for the simulation, you can choose between different examples/test in the file "init_cond.py" and add their names into the main file, or write them by yourself. Have fun!

To learn more about finite volume approach in fluid dynamics simulations, one can read a book by E.F. Toro "Riemann Solvers and Numerical Methods for Fluid Dynamics: A practical introduction" (2009)
To read about some modern solvers -- I recommend a review article by D.S. Balsara "Higher-order accurate space-time schemes for computational astrophysics—Part I: finite volume methods", Living Rev Comput Astrophys 3:2 (2017)
