# Genetic Algorithm GBC Toolbox (Python)

This repository implements a modular and didactic genetic algorithm framework for solving both single-objective and multi-objective optimization problems.

## Structure

- `main.py`: Main script to configure and run the optimization.
- `ga_continuous.py`: Core logic for continuous-variable optimization, includes NSGA-II support.
- `ga_discrete.py`: Handles discrete-variable optimization.
- `example.py`: Contains example benchmark functions (single and multi-objective).
- `requirements.txt`: Python dependencies.

## Features

### General

- Modular structure for easy adaptation to new problems.
- Continuous and discrete modes.
- Evolutionary operators: selection, crossover, mutation, elitism.
- Optional population decimation to preserve diversity.

### Supported Modes

- `'traditional'`: Single-objective optimization (e.g., sphere, eason, hadel, simple).
- `'nsga2'`: Multi-objective optimization (e.g., real_multi) using NSGA-II.

### NSGA-II

- Non-dominated sorting of individuals into Pareto fronts.
- Crowding distance to preserve diversity.
- Final Pareto front plotting (objective 1 vs objective 2).
- Best individual shown with corresponding real objective values (not surrogate fitness).

## Usage

### Setup

```bash
cd python/
pip install -r requirements.txt
```

### Configuration

In `main.py`, set the parameters at the top:

```python
modo = 'continuous'         # 'continuous' or 'discrete'
modo_otimizacao = 'nsga2'   # 'traditional' or 'nsga2'
func_name = 'real_multi'    # 'sphere', 'eason', 'hadel', 'simple', 'real_multi'
```

### Output

- Displays best objective value(s) and input vector.
- Plots best and mean fitness/objective evolution.
- Plots Pareto front if in `nsga2` mode.
- Reports total optimization time.

## Example

To run NSGA-II with the `real_multiobjective` function:

```bash
python main.py
```

This will generate two plots:
1. Objective 1 evolution over generations.
2. Final Pareto front.

## License

MIT License.