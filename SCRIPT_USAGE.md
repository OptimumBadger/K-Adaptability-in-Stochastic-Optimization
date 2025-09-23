# Facility Location Script Usage Guide

This guide explains how to run your facility location problem without manually executing each cell.

## Option 1: Run in Julia Notebook/REPL (Recommended)

### Step 1: Load the run_all_cells.jl script
In your Julia notebook or REPL, run:
```julia
include("run_all_cells.jl")
```

This will execute all the computations automatically and display progress and results.

### Step 2: Alternative - Copy and paste
You can also copy the contents of `run_all_cells.jl` and paste it into a single cell in your notebook, then run that cell.

## Option 2: Run as Standalone Julia Script

### Prerequisites
1. Install Julia from https://julialang.org/downloads/
2. Install required packages:
```bash
julia -e 'using Pkg; Pkg.add(["JuMP", "Gurobi", "Random", "Distributions", "DataFrames", "LinearAlgebra"])'
```

### Run the script
```bash
julia facility_location_script.jl
```

## What the Scripts Do

Both scripts automatically:

1. **Load Data**: Read facility location data from ORLIB format files
2. **Generate Scenarios**: Create stochastic demand scenarios
3. **Solve Optimization**: Run the facility location optimization for each scenario
4. **Reformulated Extensive Form**: Select K optimal solutions from candidates
5. **L-Augmentation Heuristic**: Iteratively improve solution quality
6. **Evaluation**: Compare performance against offline optimal solutions
7. **Report Results**: Display comprehensive performance metrics

## Key Features

- **Progress Indicators**: Shows progress during long computations
- **Automatic Parameter Setup**: Uses sensible defaults for all parameters
- **Comprehensive Output**: Displays all key metrics and results
- **Error Handling**: Gracefully handles infeasible solutions
- **Reproducible**: Uses fixed random seed for consistent results

## Customization

You can modify these parameters in the script:

```julia
# Problem parameters
unmet_pen = 50                    # Penalty for unmet demand
scaling_factor = 0.000015         # Scaling factor for transportation costs
bernoulli_case = true             # Use Bernoulli demand model

# Sampling parameters
nscen = 1000                      # Number of scenarios for sampling
nsamples = 100                    # Number of online samples
test_samples = 300                # Number of test samples
K = 5                            # Number of solutions to select
max_iter = 10                     # Maximum iterations for heuristic
```

## Output Interpretation

The script provides:
- **Offline optimal cost**: Best possible cost if you knew demand in advance
- **Model cost**: Cost using the selected K solutions
- **Heuristic cost**: Cost after L-augmentation improvement
- **Performance gaps**: Percentage difference from optimal
- **Improvement**: How much the heuristic improved over the basic model

## Troubleshooting

1. **File not found**: Update the `file_path` variable to point to your data file
2. **Gurobi license**: Ensure you have a valid Gurobi license
3. **Memory issues**: Reduce `nscen`, `nsamples`, or `test_samples` for smaller problems
4. **Timeout**: Increase the time limit in the optimization models if needed

## Performance Notes

- The script may take several minutes to run depending on problem size
- Larger values of `nscen` and `nsamples` give better results but take longer
- The L-augmentation heuristic typically provides significant improvements
