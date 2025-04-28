
# Genetic Algorithm (GA) Toolbox

## Overview

This GA toolbox is a free and open-source optimization tool originally developed during Colherinhas' master's dissertation (available in `refs/2016_Master_FERRAMENTA_DE_OTIMIZAÇÃO VIA ALGORITMOS GENÉTICOS COM APLICAÇÕES EM ENGENHARIA.pdf` - Portuguese version).  
The goal of this toolbox is to search for the minimum or maximum of a specific fitness function using Genetic Algorithms (GAs).

Currently, implementations are available in **MATLAB**, **Julia**, and under development in **Python**.

---

## Project Structure

```
Genetic-Algorithm-GBC-toolbox/
├── julia/
│   ├── evolution_strategies.jl
│   ├── fitness.jl
│   ├── main.jl
│   └── newpop.jl
│
├── matlab/
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
├── python/  # Implementation in progress
│
├── refs/
│   └── 2016_Master_FERRAMENTA_DE_OTIMIZAÇÃO.pdf
│
├── LICENSE
└── README.md
```

---

## How to Run the GA Toolbox

### MATLAB

- Navigate to the `matlab/` folder.
- Open and run the `main.m` file.

The `main.m` script is responsible for running the GA optimization by defining:
- Number of generations
- Population size (chromosomes)
- Decimation step
- Probabilities for decimation, elitism, mutation, and crossover.

Upon execution, the user will be prompted to insert a `.m` file defining the fitness function to be minimized.  
Example problems are provided in `matlab/examples/` (`eason_function`, `hadel_function`, `simple_function`, `sphere_function`).

**Details:**
- By default, the GA toolbox minimizes the fitness function `y`.
- To perform maximization, set the objective as `1/y`.
- The toolbox automatically identifies the dimension of the boundary vector `bounds = [L_1 U_1; L_2 U_2; ...]`.

The folder `matlab/fix/` contains definitions of evolutionary strategies:
- Roulette-Wheel selection
- BLX-α crossover
- Elitism
- Decimation
- Random generation of new chromosomes.

---

### Julia

- Navigate to the `julia/` folder.
- Install required Julia packages:
  ```julia
  ] add JLD, Statistics, LinearAlgebra, Printf, Plots
  ```
- Run the `main.jl` file.

**Details:**
- Fitness function is defined in `fitness.jl`.
- Initial population generation is handled by `newpop.jl`.
- Evolutionary strategies (decimation, crossover, mutation) are defined in `evolution_strategies.jl`.
- The upper and lower bounds of variables are defined by the `CromLim` matrix in `main.jl`.

---

### Python (Coming Soon)

The Python version will follow the same overall structure:
- `main.py`: main driver file for optimization setup and execution.
- `fitness.py`: fitness function definition.
- `evolution_strategies.py`: crossover, mutation, and selection operators.
- `newpop.py`: initial population generation.

The Python version will adopt popular scientific libraries such as:
- `numpy`
- `matplotlib`
- `random`
- (Possibly `pandas` or `scipy` for additional utilities).

Stay tuned for future updates!

---

## Results and Post-Processing

Upon completion of the optimization:
- The elapsed execution time (in seconds) is displayed.
- The optimum result found is printed.
- An optimization curve is generated showing:
  - Evolution of the best fitness value per generation.
  - Evolution of the mean fitness of the population.

---

## License

This project is licensed under the [GNU General Public License v3.0 (GPL-3.0)](https://www.gnu.org/licenses/gpl-3.0.html).

---

## References

- Gino Bertolucci Colherinhas,  
  "**Ferramenta de Otimização via Algoritmos Genéticos com Aplicações em Engenharia**" (2016).  
  Master's dissertation available in the `refs/` folder.
