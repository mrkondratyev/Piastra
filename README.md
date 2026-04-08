# #ENG
# Piastra

**Piastra** is a teaching-oriented framework for solving:
- Linear advection
- Inviscid compressible hydrodynamics (HD)
- Special-relativistic hydrodynamics (rHD)
- Ideal Magnetohydrodynamics (MHD)
- Special-relativistic Magnetohydrodynamics (rMHD)
- Thermal diffusion (explicit and RKL2 super time-stepping)
  
within a **finite-volume framework**. The code is written in **Python** with extensive use of **NumPy**, and includes tools for visualization with **matplotlib**.

---

## Features

- **Dimensionality**: 1D and 2D Cartesian (X,Y) and Cylindrical (R,Z) structured grids  
- **Finite-volume solver** with approximate Riemann solvers for fluxes  
- **Hydrodynamics solvers**:
  - LLF (Rusanov, 1961)  
  - HLL (Harten–Lax–van Leer, 1983)  
  - HLLC (Toro et al., 1994)  
  - Roe (Roe, 1981)  
- **Special-relativistic HD solvers** (rHD):
  - LLF, HLL, HLLC (Mignone & Bodo, 2005)
  - 4-velocity reconstruction (guarantees |v| < 1 at cell faces)
  - Newton-Raphson conservative-to-primitive inversion
- **MHD solvers**:
  - LLF, HLL, HLLD (Miyoshi and Kusano, 2005)
  - Divergence control:
    - Powell 8-wave method (Powell 1994, 1999; Tóth 2000)
    - Constrained Transport (Flux-CT; Balsara & Spicer 1999)
- **Special-relativistic MHD solvers** (rMHD):
  - LLF, HLL
  - 4-velocity reconstruction for guaranteed sub-luminal face states
  - Newton-Raphson conservative-to-primitive inversion (Mignone & McKinney 2007)
  - Constrained Transport (Flux-CT) for divergence-free magnetic evolution
- **Thermal diffusion**:
  - Explicit forward Euler
  - RKL2 super time-stepping (Meyer, Balsara & Aslam 2014) with configurable stages
  - Geometry-aware FV Laplacian (Cartesian, cylindrical, polar)
- **Advection**: high-order RK with exact Riemann solver or Lax–Wendroff scheme
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
  - `grid_misc.py`: grid utility functions (divergence, gradient, norms)
  - `sim_state.py`: storage for fluid variables
  - `boundaries.py`: boundary conditions handling for scalar and vector variables
  - `high_order_rec.py`: spatial reconstruction routines (PCM/PLM/PPM/WENO)
  - `helpers.py`: initial condition dispatch and simulation loop  
  - `advection_one_step.py`, `hydro_one_step.py`, `rHD_one_step.py`, `MHD_one_step_CT.py`, `MHD_one_step_8wave.py`, `rMHD_one_step.py`, `diffusion_one_step.py`: solver backends
  - `hydro_phys.py`, `rHD_phys.py`, `MHD_phys.py`, `rMHD_phys.py`: supplementary physics modules
  - `io_visual.py`: plotting utilities
  - `advection_init_cond.py`, `hydro_init_cond.py`, `rHD_init_cond.py`, `MHD_init_cond.py`, `rMHD_init_cond.py`, `diffusion_init_cond.py`: initial conditions
  - `main.py`: launcher (alternatively can be run via Jupyter notebook **main.ipynb**)
---

## Usage

**Requirements:**
- Python 3.9+  
- NumPy  
- matplotlib  
- IPython (recommended for notebooks)  

**Example (main.py)**

```python
from src.parameters import Parameters
from src.grid.grid_setup import Grid
from src.sim_state import SimState
from src.misc.helpers import initial_model, run_simulation
from src.models.HD.hydro_one_step import Hydro2D

# Setup parameters
par = Parameters(mode="HD", problem="sod2Dcart", Nx1=64, Nx2=64)

# Setup grid and state
grid = Grid(par.Nx1, par.Nx2, par.Ngc)
simstate = SimState(grid, par)
grid, simstate, par, eos = initial_model(grid, simstate, par)

# Run solver
solver = Hydro2D(grid, simstate, eos, par)
simstate, par.timenow = run_simulation(grid, simstate, par, solver, simstate.dens, nsteps=200)
```


# Parameters

All parameters are stored in the Parameters class. Defaults are applied automatically.

## Required:

- mode: `'adv'`, `'HD'`, `'rHD'`, `'MHD'`, `'rMHD'`, or `'diff'`

- problem: name of the test problem

- Nx1, Nx2: grid resolution

## Optional:

- flux_type: depends on solver (defaults assigned automatically)

- rec_type: `'PCM'`, `'PLM'`, `'PPM'`, `'PPMorig'`, `'WENO'`

- RK_order: `'RK1'`, `'RK2'`, `'RK3'`

- CFL: Courant number (default 0.7)

- divb_tr: `'CT'` or `'8wave'` (MHD only); rMHD uses CT exclusively

## Available Problems

Advection: smooth/discontinuous 1D/2D tests

Hydrodynamics: Sod shock tubes, Noh, strong shocks, Shu–Osher, Einfeldt, Kelvin–Helmholtz, Rayleigh–Taylor, Sedov blast waves, double Mach reflection, implosion, Gresho vortex, shock–cloud, gap-opening disk, cylindrical jet

Relativistic HD: Mignone & Bodo (2005) Riemann problems (RP1–RP5), 2D Riemann problem, relativistic Rayleigh–Taylor instability, perturbed shock, shock heating, relativistic jet

MHD: Brio–Wu shock, Tóth problem, Ryu–Jones, Alfvén wave, blast wave, Orszag–Tang vortex, rotor, current sheet, field loop, disk, shock–cloud

Relativistic MHD: Brio–Wu 1D (rMHD), Mignone & Bodo (2006) Riemann problems (RP2–RP4), 2D blast wave, 2D relativistic rotor

Diffusion: 2D Gaussian pulse, crossed Gaussian ridges, annular ring, 1D Gaussian, 1D step, 1D sine, cylindrical 2D

## References

- E. F. Toro, Riemann Solvers and Numerical Methods for Fluid Dynamics (2009)

- D. S. Balsara, Higher-order accurate space-time schemes for computational astrophysics—Part I: finite volume methods, Living Rev Comput Astrophys 3:2 (2017)

- G. Tóth, The ∇·B constraint in shock-capturing MHD codes, JCP 161, 605 (2000)

- A. Mignone & G. Bodo, An HLLC Riemann solver for relativistic flows, MNRAS 364, 126 (2005)

- A. Mignone & G. Bodo, An HLLC solver for relativistic flows — II. Magnetohydrodynamics, MNRAS 368, 1040 (2006)

- L. Del Zanna, N. Bucciantini & P. Londrillo, An efficient shock-capturing central-type scheme for multidimensional relativistic flows — II. MHD, A&A 400, 397 (2003)

- A. Mignone & J. C. McKinney, Equation of state in relativistic MHD: variable versus constant adiabatic index, MNRAS 378, 1118 (2007)

- C. D. Meyer, D. S. Balsara & T. D. Aslam, A stabilized Runge–Kutta–Legendre method for explicit super-time-stepping of parabolic and mixed equations, JCP 257, 594 (2014)

- M. Zingale, Introduction to Computational Astrophysical Hydrodynamics (2015+)


## Additional notes

The folder **notebooks** contains lightweight solvers in separate independent ipynb-files (with comments in Russian): 
2D shallow water simulator, 1D hydrodynamics, 1D advection solver, and 1D diffusion equation. My lecture notes are provided as well (they are also in Russian, however. I plan to rewrite them soon).

---
---
---

# #RUS

# Piastra

**Piastra** — учебный код для моделирования:
- Линейного уравнения переноса (адвекции)
- Сжимаемой невязкой гидродинамики (HD)
- Специально-релятивистской гидродинамики (rHD)
- Идеальной магнитной гидродинамики (MHD)
- Специально-релятивистской магнитной гидродинамики (rMHD)
- Теплопроводности (явная схема и RKL2 super time-stepping)
в рамках **метода конечных объемов Годуновского типа** с **TVD методами Рунге–Кутты** для интегрирования по времени и **ограниченной кусочно-полиномиальной реконструкцией высокого порядка по пространству**. Код написан на **Python** с активным использованием **NumPy** и включает инструменты визуализации через **matplotlib**.

---

## Возможности

- **Измерения**: 1D и 2D структурированные сетки -- декартовы (X,Y) и цилиндрические (R,Z)  
- **Решатель методом конечных объемов** с приближенными решениями задачи Римана для вычисления потоков  
- **Гидродинамические решатели**:
  - LLF (Русанов, 1961)  
  - HLL (Хартен–Лакс–ван Леер, 1983)  
  - HLLC (Toro et al., 1994)  
  - Roe (Roe, 1981)  
- **Решатели специально-релятивистской гидродинамики** (rHD):
  - LLF, HLL, HLLC (Mignone & Bodo, 2005)
  - Реконструкция 4-скорости (гарантирует |v| < 1 на гранях ячеек)
  - Обращение консервативных переменных методом Ньютона–Рафсона
- **Решатели МГД**:
  - LLF, HLL, HLLD (Miyoshi & Kusano, 2005)
  - Контроль дивергенции:  
    - Метод 8 волн Пауэлла (Powell 1994, 1999; Tóth 2000)  
    - Constrained Transport (Flux-CT; Balsara & Spicer 1999)  
- **Решатели специально-релятивистской МГД** (rMHD):
  - LLF, HLL
  - Реконструкция 4-скорости для гарантированно досветовых состояний на гранях ячеек
  - Обращение консервативных переменных методом Ньютона–Рафсона (Mignone & McKinney 2007)
  - Constrained Transport (Flux-CT) для безывергентной эволюции магнитного поля
- **Теплопроводность**:
  - Явная схема Эйлера
  - RKL2 super time-stepping (Meyer, Balsara & Aslam 2014) с настраиваемым числом стадий
  - Геометрически корректный лапласиан МКО (декартова, цилиндрическая, полярная геометрии)
- **Перенос**: Методы Рунге-Кутты с точным решением Римана или схема Лакса–Вендроффа  
- **Методы реконструкции**:  
  - PCM (кусочно-постоянная)  
  - PLM (кусочно-линейная с ограничителем наклона, 2-й порядок)  
  - PPM (Colella & Woodward 1984, Mignone 2014)  
  - PPMorig (оригинальная версия Colella & Woodward)  
  - WENO5 (5-й порядок, Jiang & Shu 1996)  
- **Интегрирование по времени**:  
  - RK1 (Эйлер)  
  - RK2, RK3 (TVD Runge–Kutta; Shu & Osher 1988)  
- **Призрачные ячейки (ghost cells)**: подбираются автоматически (2 для PLM/PCM, 3 для PPM/WENO)  
- **Управление симуляцией** через класс `Parameters` с установкой значений по умолчанию и проверкой корректности  
- **Модульная структура**:
  - `parameters.py`: центральный контейнер параметров  
  - `grid_setup.py`: определение структуры сетки
  - `grid_misc.py`: вспомогательные функции сетки (дивергенция, градиент, нормы)
  - `sim_state.py`: хранение переменных жидкости  
  - `boundaries.py`: учет граничных условий на сетке для скалярных и векторных функций
  - `high_order_rec.py`: алгоритмы реконструкции (PCM/PLM/PPM/WENO)
  - `helpers.py`: диспетчер начальных условий и основной цикл симуляции
  - `advection_one_step.py`, `hydro_one_step.py`, `rHD_one_step.py`, `MHD_one_step_CT.py`, `MHD_one_step_8wave.py`, `rMHD_one_step.py`, `diffusion_one_step.py`: бэкенды решателей
  - `hydro_phys.py`, `rHD_phys.py`, `MHD_phys.py`, `rMHD_phys.py`: вспомогательные физические модули
  - `io_visual.py`: функции визуализации
  - `advection_init_cond.py`, `hydro_init_cond.py`, `rHD_init_cond.py`, `MHD_init_cond.py`, `rMHD_init_cond.py`, `diffusion_init_cond.py`: начальные условия
  - `main.py`: запуск симуляции (можно запускать также через Jupyter notebook **main.ipynb**)  

---

## Использование

**Требования:**
- Python 3.9+  
- NumPy  
- matplotlib  
- IPython (рекомендуется для работы с ноутбуками)  

**Пример (main.py)**

```python
from src.parameters import Parameters
from src.grid.grid_setup import Grid
from src.sim_state import SimState
from src.misc.helpers import initial_model, run_simulation
from src.models.HD.hydro_one_step import Hydro2D

# Настройка параметров
par = Parameters(mode="HD", problem="sod2Dcart", Nx1=64, Nx2=64)

# Инициализация сетки и состояния
grid = Grid(par.Nx1, par.Nx2, par.Ngc)
simstate = SimState(grid, par)
grid, simstate, par, eos = initial_model(grid, simstate, par)

# Запуск решателя
solver = Hydro2D(grid, simstate, eos, par)
simstate, par.timenow = run_simulation(grid, simstate, par, solver, simstate.dens, nsteps=200)
```


# Параметры

Все параметры хранятся в классе Parameters. Значения по умолчанию присваиваются автоматически.

## Необходимые:

- mode: `'adv'`, `'HD'`, `'rHD'`, `'MHD'`, `'rMHD'`, или `'diff'`

- problem: название тестовой задачи

- Nx1, Nx2: разрешение сетки

## Опциональные:

- flux_type: зависит от решателя (default присваивается автоматически)

- rec_type: `'PCM'`, `'PLM'`, `'PPM'`, `'PPMorig'`, `'WENO'`

- RK_order: `'RK1'`, `'RK2'`, `'RK3'`

- CFL: Число Куранта (default 0.7)

- divb_tr: `'CT'` или `'8wave'` (только для МГД); rMHD использует только CT

## Доступные проблемы

Адвекция: smooth/discontinuous 1D/2D тесты

Газовая динамика: задача Зода, Но, сильный удар, Шу–Ошер, Эйнфельдт, Кельвин–Гельмгольц, Рэлей–Тейлор, взрывные волны Седова, двойное отражение Маха, сжатие, вихрь Грешо, удар–облако, диск с пробелом, цилиндрическая струя

Релятивистская гидродинамика: задачи Римана Mignone & Bodo (2005) (RP1–RP5), 2D задача Римана, релятивистская неустойчивость Рэлея–Тейлора, возмущённый удар, нагрев ударной волной, релятивистская струя

МГД: Брио–Ву, задача Тота, Рю–Джонс, волна Альвена, взрывная волна, вихрь Оршага–Танга, ротор, токовый лист, петля поля, диск, удар–облако

Релятивистская МГД: Брио–Ву 1D (rMHD), задачи Римана Mignone & Bodo (2006) (RP2–RP4), 2D взрывная волна, 2D релятивистский ротор

Диффузия: 2D гауссов импульс, перекрёстные гауссовы гребни, кольцевой импульс, 1D гауссов импульс, 1D ступенька, 1D синус, цилиндрическая 2D

## Ссылки

- Д. В. Бисикало, А. Г. Жилкин, А. А. Боярчук, Газодинамика Тесных Двойных Звезд, Физматлит (2013)

- E. F. Toro, Riemann Solvers and Numerical Methods for Fluid Dynamics (2009)

- D. S. Balsara, Higher-order accurate space-time schemes for computational astrophysics—Part I: finite volume methods, Living Rev Comput Astrophys 3:2 (2017)

- G. Tóth, The ∇·B constraint in shock-capturing MHD codes, JCP 161, 605 (2000)

- A. Mignone & G. Bodo, An HLLC Riemann solver for relativistic flows, MNRAS 364, 126 (2005)

- A. Mignone & G. Bodo, An HLLC solver for relativistic flows — II. Magnetohydrodynamics, MNRAS 368, 1040 (2006)

- L. Del Zanna, N. Bucciantini & P. Londrillo, A&A 400, 397 (2003)

- A. Mignone & J. C. McKinney, MNRAS 378, 1118 (2007)

- C. D. Meyer, D. S. Balsara & T. D. Aslam, JCP 257, 594 (2014)

- M. Zingale, Introduction to Computational Astrophysical Hydrodynamics (2015+)

## Дополнения

В папке **notebooks** представлены облегченнные решатели в отдельных ipynb-файлах с комментариями на русском: 
2D симулятора уравнений мелкой воды, 1D газовая динамика, 1D решатель адвекции, и 1D модель диффузии. 
Также в корневой папке лежат мои лекционные записи лекции для участников Третьей Школы НЦФМ "Экспериментальная и Лабораторная Астрофизика и Геофизика".
