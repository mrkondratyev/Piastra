# #ENG
# Piastra2D

**Piastra2D** is a teaching-oriented code for solving:
- Linear advection 
- Inviscid compressible hydrodynamics (HD)  
- Ideal Magnetohydrodynamics (MHD)  
within a **finite-volume Godunov-type framework** via **TVD Runge–Kutta time integration**.  
The code is written in **Python** with extensive use of **NumPy**, and includes tools for visualization with **matplotlib**.  

Additionally, the folder **notebooks** contains lightweight solvers in separate independent ipynb-files with comments in Russian: 
2D shallow water simulator, 1D hydrodynamics, 1D advection solver, and 1D diffusion equation

---

## Features

- **Dimensionality**: 1D and 2D Cartesian (X,Y) and Cylindrical (R,Z) structured grids  
- **Finite-volume solver** with approximate Riemann solvers for fluxes  
- **Hydrodynamics solvers**:
  - LLF (Rusanov, 1961)  
  - HLL (Harten–Lax–van Leer, 1983)  
  - HLLC (Toro et al., 1994)  
  - Roe (Roe, 1981)  
- **MHD solvers**:
  - LLF, HLL, HLLD (Miyoshi and Kusano, 2005)
  - Divergence control:  
    - Powell 8-wave method (Powell 1994, 1999; Tóth 2000)  
    - Constrained Transport (Flux-CT; Balsara & Spicer 1999)  
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
  - `sim_state.py`: storage for fluid variables  
  - `helpers.py`: initial condition dispatch and simulation loop  
  - `advection_one_step.py`, `hydro_one_step.py`, `MHD_one_step_CT.py`, `MHD_one_step_8wave.py`: solver backends
  - `hydro_phys.py`, `MHD_phys.py`: supplementary modules  
  - `visualization.py`: plotting utilities  
  - `advection_init_cond.py`, `hydro_init_cond.py`, `MHD_init_cond.py`: initial conditions
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
```


# Parameters

All parameters are stored in the Parameters class. Defaults are applied automatically.

## Required:

- mode: 'adv', 'HD', or 'MHD'

- problem: name of the test problem

- Nx1, Nx2: grid resolution

## Optional:

- flux_type: depends on solver (defaults assigned automatically)

- rec_type: 'PCM', 'PLM', 'PPM', 'PPMorig', 'WENO'

- RK_order: 'RK1', 'RK2', 'RK3'

- CFL: Courant number (default 0.7)

- divb_tr: 'CT' or '8wave' (MHD only)

## Available Problems

Advection: smooth/discontinuous 1D/2D tests

Hydrodynamics: Sod shock tubes, strong shocks, Kelvin–Helmholtz, Rayleigh–Taylor, Sedov blast waves

MHD: Brio–Wu shock, Tóth problem, blast wave, Orszag–Tang vortex

## References

- E. F. Toro, Riemann Solvers and Numerical Methods for Fluid Dynamics (2009)

- D. S. Balsara, Higher-order accurate space-time schemes for computational astrophysics—Part I: finite volume methods, Living Rev Comput Astrophys 3:2 (2017)

- G. Tóth, The ∇·B constraint in shock-capturing MHD codes, JCP 161, 605 (2000)

- M. Zingale, Introduction to Computational Astrophysical Hydrodynamics (2015+)


# #RUS

# Piastra2D

**Piastra2D** — учебный фреймворк для решения:
- Линейного уравнения переноса (адвекции)  
- Сжимаемой невязкой гидродинамики (HD)  
- Идеальной магнитной гидродинамики (MHD)  
в рамках **метода конечных объемов Годуновского типа** с **TVD методами Рунге–Кутты** для интегрирования по времени.  
Код написан на **Python** с активным использованием **NumPy** и включает инструменты визуализации через **matplotlib**.

Дополнительно, в папке **notebooks** представлены облегченнные решатели в отдельных ipynb-файлах с комментариями на русском: 
2D симулятора уравнений мелкой воды, 1D газовая динамика, 1D решатель адвекции, и 1D модель диффузии

---

## Возможности

- **Измерения**: 1D и 2D структурированные сетки -- декартовые (X,Y) и цилиндрические (R,Z)  
- **Решатель методом конечных объемов** с приближенными решениями задачи Римана для вычисления потоков  
- **Гидродинамические решатели**:
  - LLF (Русанов, 1961)  
  - HLL (Хартен–Лакс–ван Леер, 1983)  
  - HLLC (Toro et al., 1994)  
  - Roe (Roe, 1981)  
- **Решатели МГД**:
  - LLF, HLL, HLLD (Miyoshi & Kusano, 2005)
  - Контроль дивергенции:  
    - Метод 8 волн Пауэлла (Powell 1994, 1999; Tóth 2000)  
    - Constrained Transport (Flux-CT; Balsara & Spicer 1999)  
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
  - `sim_state.py`: хранение переменных жидкости  
  - `helpers.py`: диспетчер начальных условий и основной цикл симуляции  
  - `advection_one_step.py`, `hydro_one_step.py`, `MHD_one_step_CT.py`, `MHD_one_step_8wave.py`: бэкенды решателей  
  - `hydro_phys.py`, `MHD_phys.py`: вспомогательные модули  
  - `visualization.py`: функции визуализации  
  - `advection_init_cond.py`, `hydro_init_cond.py`, `MHD_init_cond.py`: начальные условия  
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
from parameters import Parameters
from grid_setup import Grid
from sim_state import SimState
from helpers import initial_model, run_simulation
from hydro_one_step import Hydro2D

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


# Parameters

Все параметры хранятся в классе Parameters. Значения по умолчанию присваиваются автоматически.

## Необходимые:

- mode: 'adv', 'HD', или 'MHD'

- problem: название тестовой задачи

- Nx1, Nx2: разрешение сетки

## Опциональные:

- flux_type: зависит от решателя (default присваивается автоматически)

- rec_type: 'PCM', 'PLM', 'PPM', 'PPMorig', 'WENO'

- RK_order: 'RK1', 'RK2', 'RK3'

- CFL: Число Куранта (default 0.7)

- divb_tr: 'CT' или '8wave' (только для МГД)

## Доступные проблемы

Адвекция: smooth/discontinuous 1D/2D tests

Газовая динамика: Sod shock tubes, strong shocks, Kelvin–Helmholtz, Rayleigh–Taylor, Sedov blast waves

МГД: Brio–Wu shock, Tóth problem, blast wave, Orszag–Tang vortex

## Ссылки

- Д. В. Бисикало, А. Г. Жилкин, А. А. Боярчук, Газодинамика Тесных Двойных Звезд, Физматлит (2013)

- E. F. Toro, Riemann Solvers and Numerical Methods for Fluid Dynamics (2009)

- D. S. Balsara, Higher-order accurate space-time schemes for computational astrophysics—Part I: finite volume methods, Living Rev Comput Astrophys 3:2 (2017)

- G. Tóth, The ∇·B constraint in shock-capturing MHD codes, JCP 161, 605 (2000)

- M. Zingale, Introduction to Computational Astrophysical Hydrodynamics (2015+)


