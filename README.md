
# Genetic Algorithm (GA) Toolbox

## Overview

This GA toolbox is a free and open-source optimization tool originally developed during Colherinhas' master's dissertation (available in `refs/2016_Master_FERRAMENTA_DE_OTIMIZAÇÃO VIA ALGORITMOS GENÉTICOS COM APLICAÇÕES EM ENGENHARIA.pdf` - Portuguese version).  
The goal of this toolbox is to search for the minimum or maximum of a specific fitness function using Genetic Algorithms (GAs).

Implementations are currently available in MATLAB, Julia, and Python.

---

## What’s New (Python Version)

- 🧠 **NSGA-II extended for 2 or 3 objectives**
  - Generalized non-dominated sorting and crowding distance
  - Supports both 2D and 3D Pareto front visualization

- 🆕 **SBX crossover** added (Deb & Agrawal, 1995)
  - Ideal for continuous multi-objective optimization via NSGA-II
  - Configurable η parameter for offspring distribution

- 🛠️ **Unified crossover logic**
  - Shared interface for traditional and NSGA-II modes
  - Dynamic selection via `config['crossover']`

- 📊 **Enhanced Pareto front analysis**
  - Uses only non-dominated solutions for plotting
  - Annotated 2D plots and interactive 3D Plotly visualization
  - Representative solutions: `min_f1`, `min_f2`, `min_f3`, `balanced`

- 📌 **New test case**: `real_multiobjective_2`
  - 3-objective benchmark with distance-based targets

---

## How to Cite this Toolbox

If you use this toolbox in your work, please cite it as follows:

**In Word documents (e.g., academic papers):**
> Colherinhas, Gino Bertollucci. *Genetic Algorithm (GA) Toolbox for Optimization*. 2016. Available at: [https://github.com/Ginobc/Genetic-Algorithm-GBC-toolbox](https://github.com/Ginobc/Genetic-Algorithm-GBC-toolbox). Accessed: April 28, 2025.

**In LaTeX documents:**
```latex
@misc{colherinhas2016ga_toolbox,
  author       = {Gino Bertollucci Colherinhas},
  title        = {Genetic Algorithm (GA) Toolbox for Optimization},
  year         = {2016},
  howpublished = {\url{https://github.com/Ginobc/Genetic-Algorithm-GBC-toolbox}},
  note         = {Accessed: April 28, 2025}
}
```

---

## Project Structure

```
Genetic-Algorithm-GBC-toolbox/
├── julia/          # Julia implementation
│   ├── evolution_strategies.jl
│   ├── fitness.jl
│   ├── main.jl
│   └── newpop.jl
│
├── matlab/         # MATLAB implementation
│   ├── examples/
│   │   ├── eason_function.m
│   │   ├── hadel_function.m
│   │   ├── simple_function.m
│   │   └── sphere_function.m
│   ├── fix/
│   │   ├── evolution_strategies.m
│   │   ├── newpop.m
│   │   └── fitness.m
│   └── main.m
│
├── python/         # Python implementation
│   ├── example.py
│   ├── ga_core.py
│   ├── main.py
│   └── requirements.txt
│
├── refs/           # Reference material
│   └── 2016_Master_FERRAMENTA_DE_OTIMIZAÇÃO.pdf
│
├── z_backup/       # Personal backup folder (not versioned)
│
├── LICENSE
├── README.md
└── .gitignore
```

---

# How to Run the GA Toolbox

## Python

- Navigate to the `python/` folder.
- Install the requirements:
  ```bash
  pip install -r requirements.txt
  ```
- Run `main.py` to start the optimization.

### Settings inside `main.py`

- **Mode**:
  - `'continuous'` – for floating-point variables.
  - `'discrete'` – for integer-indexed or grouped variables (e.g., materials or standard thicknesses).

- **Example Function**:
  - `'sphere'`: unimodal, convex benchmark function.
  - `'easom'`: multimodal function with a sharp global minimum.
  - `'hadel'`: nonlinear function with polynomial and trigonometric terms.
  - `'simple'`: parabolic surface with a cross-product term.
  - `'real_multi_2v'`: two-objective continuous function for NSGA-II testing.
  - `'real_multi_3v'`: **three-objective** continuous function (benchmarking distances to three targets).
  - `'discrete_alloy'`: discrete material selection problem optimizing cost, strength, and density.

- **Optimization Strategy**:
  - `'traditional'`: single-objective GA with fitness transformation.
  - `'nsga2'`: multi-objective NSGA-II (**supports 2 or 3 objectives**).

### Python Modules

- `example.py`: Contains test functions for single- and multi-objective problems.
- `ga_core.py`: Unified core for both traditional GA and NSGA-II, including:
  - Continuous and discrete variable handling.
  - Crossover, mutation, and elitism logic.
- `main.py`: Main execution script:
  - Configures the problem, runs the GA, plots results.
  - Exports Pareto-optimal solutions to Excel.
  - Highlights representative solutions (`min_f1`, `min_f2`, `balanced`, `min_f3` when applicable).
- `requirements.txt`: Python dependencies.

### Evolutionary Strategies in Python

Unified architecture for continuous and discrete problems using both traditional GA and NSGA-II.

**Selection**:
- Roulette-wheel based selection.

**Crossover Operators**:
- `BLX-α`: for continuous and discretized variables.
- `SBX`: Simulated Binary Crossover (ideal for NSGA-II, with configurable η).
- One-point and two-point crossover: for discrete/indexed problems.
- Linear convex: experimental, combines parents via convex interpolation.

**Mutation**:
- Gaussian mutation (continuous).
- Uniform random mutation or re-sampling (discrete).

**Elitism and Diversity**:
- Best individuals are preserved across generations.
- **Decimation**: periodically regenerates part of the population to avoid premature convergence.

**NSGA-II (Multi-objective Optimization)**:
- Non-dominated sorting for `n ≥ 2` objectives.
- Crowding distance calculation to maintain Pareto front diversity.
- Dynamic Pareto front generation.
- Representative solutions extracted and annotated.
- **Visualization**:
  - 2D plots: using only non-dominated solutions (`f1 vs f2`, `f1 vs f3`).
  - 3D plot (Plotly): when optimizing 3 objectives (`f1 × f2 × f3`), with uniform axis scaling.

---

## MATLAB

- Navigate to the `matlab/` folder.
- Open and run the `main.m` file in MATLAB.
- When running, select one of the example functions inside `examples/`:
  - `eason_function.m`
  - `hadel_function.m`
  - `simple_function.m`
  - `sphere_function.m`
- Configure the number of generations, chromosomes, and probabilities of decimation, elitism, mutation, and crossover.

**Evolutionary strategies used:**
- Roulette-Wheel selection
- BLX-α crossover
- Deterministic elitism and decimation.

---

## Julia

- Navigate to the `julia/` folder.
- Install the necessary packages by typing:
  ```julia
  ] add JLD, Statistics, LinearAlgebra, Printf, Plots
  ```
- Run the `main.jl` file.
- Configure optimization parameters and function bounds within the script.

**Evolutionary strategies implemented:**
- Roulette-Wheel selection
- BLX-α crossover
- Mutation
- Elitism and decimation.

---

## Results and Post-Processing

Upon completion of optimization:
- Elapsed execution time is displayed.
- Fittest inputs and the optimal solution found is printed.
- A plot is generated showing:
  - Best fitness over generations.
  - Mean fitness evolution over generations.

---

## License

This project is licensed under the [GNU General Public License v3.0 (GPL-3.0)](https://www.gnu.org/licenses/gpl-3.0.html).

---

## References

- Gino Bertolucci Colherinhas,  
  "**Ferramenta de Otimização via Algoritmos Genéticos com Aplicações em Engenharia**" (2016).  
  Master's dissertation available in the `refs/` folder.
