
# Genetic Algorithm (GA) Toolbox

## Overview

This GA toolbox is a free and open-source optimization tool originally developed during Colherinhas' master's dissertation (available in `refs/2016_Master_FERRAMENTA_DE_OTIMIZAÇÃO VIA ALGORITMOS GENÉTICOS COM APLICAÇÕES EM ENGENHARIA.pdf` - Portuguese version).  
The goal of this toolbox is to search for the minimum or maximum of a specific fitness function using Genetic Algorithms (GAs).

Implementations are currently available in **MATLAB**, **Julia**, and **Python**.

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
│   ├── ga_continuous.py
│   ├── ga_discrete.py
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
  - `'continuous'` for floating-point variables (based on MATLAB logic).
  - `'discrete'` for grouped/discrete variable optimization.
- Select the **example function**:
  - `'sphere'`, `'eason'`, `'hadel'`, `'simple'`.

### Python Modules
- `example.py`: Contains example functions translated from MATLAB.
- `ga_continuous.py`: Genetic Algorithm for continuous (floating point) variables.
- `ga_discrete.py`: Genetic Algorithm for discrete variable problems, supporting multiple crossover types.
- `main.py`: Main script to configure and run the GA optimization.
- `requirements.txt`: List of required Python packages.

---

### Evolutionary Strategies in Python
- **Selection**: Roulette-Wheel selection
- **Crossover Operators**:
  - BLX-α crossover (continuous and discrete modes)
  - One-point crossover (discrete)
  - Two-point crossover (discrete)
- **Mutation**: Random generation of new individuals
- **Elitism**: Preservation of the best-performing individuals
- **Decimation**: Periodic replacement of a portion of the population

---

## Results and Post-Processing

Upon completion of optimization:
- Elapsed execution time is displayed.
- The optimal solution found is printed.
- A plot is generated showing:
  - Best fitness over generations.
  - Mean fitness evolution over generations.

---

## Requirements (Python)

The project uses the following Python libraries:
- `numpy`
- `matplotlib`

Install them using:
```bash
pip install -r requirements.txt
```

---

## License

This project is licensed under the [GNU General Public License v3.0 (GPL-3.0)](https://www.gnu.org/licenses/gpl-3.0.html).

---

## References

- Gino Bertolucci Colherinhas,  
  "**Ferramenta de Otimização via Algoritmos Genéticos com Aplicações em Engenharia**" (2016).  
  Master's dissertation available in the `refs/` folder.
