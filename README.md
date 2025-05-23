
# Genetic Algorithm (GA) Toolbox

## Overview

This GA toolbox is a free and open-source optimization tool originally developed during Colherinhas' master's dissertation (available in `refs/2016_Master_FERRAMENTA_DE_OTIMIZAÇÃO VIA ALGORITMOS GENÉTICOS COM APLICAÇÕES EM ENGENHARIA.pdf` - Portuguese version).  
The goal of this toolbox is to search for the minimum or maximum of a specific fitness function using Genetic Algorithms (GAs).

Implementations are currently available in MATLAB, Julia, and Python.

---

### How to Cite this Toolbox

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

## How to Run the GA Toolbox

### MATLAB

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

### Julia

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

### Python

- Navigate to the `python/` folder.
- Install the requirements:
  ```bash
  pip install -r requirements.txt
  ```
- Run `main.py` to start the optimization.

**Settings inside `main.py`:**
- Select the **mode**:
  - `'continuous'` for floating-point variables.
  - `'discrete'` for grouped/discrete variable optimization.
- Select the **example function**:
  - `'sphere'`: standard continuous benchmark function (unimodal, convex).
  - `'easom'`: multimodal function with a sharp global minimum.
  - `'hadel'`: nonlinear function combining polynomial and trigonometric terms.
  - `'simple'`: a parabolic function with cross-product terms.
  - `'real_multi'`: two-objective continuous function used for NSGA-II testing.
  - `'discrete_alloy'`: discrete optimization example using integer-indexed variables. Each variable represents a standardized material thickness or dimension. The input vector consists of integer indices, and the evaluation function computes a weighted cost, material strength, and density. The goal is to minimize a cost-strength objective while penalizing high-density solutions, simulating a constrained materials engineering design scenario.

- Choose the **optimization strategy**:
  - `'traditional'`: single-objective optimization using fitness transformation.
  - `'nsga2'`: multi-objective optimization using NSGA-II (**implemented only in Python**).

### Python Modules
- `example.py`: Contains benchmark functions for testing the GA, including:
  - Single-objective: `sphere`, `eason`, `hadel`, `simple`
  - Multi-objective: `real_multiobjective`
- `ga_core.py`: Unified module that implements:
  - Traditional GA (single-objective)
  - NSGA-II (multi-objective)
  - Continuous and discrete variable handling
  - All crossover and mutation logic
- `main.py`: Entry point of the Python implementation.
  - Configures optimization parameters via `config` dictionary
  - Selects the mode (`'continuous'` or `'discrete'`)
  - Handles plotting and result display
- `requirements.txt`: Lists dependencies.

---

### Evolutionary Strategies in Python
- **Selection**: Roulette-Wheel selection  
- **Crossover Operators**:
  - BLX-α crossover
  - One-point crossover
  - Two-point crossover
- **Mutation**: Uniform random mutation  
- **Elitism**: Best individuals are preserved between generations  
- **Decimation**: Part of the population is periodically regenerated to maintain diversity  
- **NSGA-II Features**:
  - Multi-objective evaluation
  - Non-dominated sorting into Pareto fronts
  - Crowding distance to maintain solution diversity
  - Pareto front plotted after final generation
  - Objective values and corresponding input vector printed after optimization

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
