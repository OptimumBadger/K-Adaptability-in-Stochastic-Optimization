# Candidate-selection-via-Stochastic-IP

## Quick Start

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd Candidate-selection-via-Stochastic-IP
   ```

2. **Install Julia** (if not already installed):
   ```bash
   # On Mac with Homebrew
   brew install julia
   ```

3. **Install required packages**:
   ```bash
   julia -e 'using Pkg; Pkg.add(["JuMP", "Gurobi", "Random", "Distributions", "DataFrames", "LinearAlgebra"])'
   ```

4. **Run the scripts**:
   ```bash
   # Core 3 phases only (faster)
   julia complete_facility_location.jl
   
   # Full pipeline with L-Augmentation heuristic
   julia l_augmentation_heuristic.jl
   ```

## Files

### Main Scripts
- `complete_facility_location.jl` - **Core 3 phases (recommended for basic analysis)**
  - Phase 1: Scenario Generation
  - Phase 2: Reformulated Extensive Form  
  - Phase 3: Evaluation
  - Faster execution, good for initial analysis

- `l_augmentation_heuristic.jl` - **Full pipeline with L-Augmentation heuristic (recommended for advanced analysis)**
  - All 3 core phases PLUS L-Augmentation heuristic
  - Shows before/after performance comparison
  - Demonstrates improvement from heuristic optimization
  - Longer execution time but better solutions

### Additional Files
- `facility_location.ipynb` - Original notebook implementation
- `debug_facility_location.jl` - Debug version with detailed progress
- `facility_location_script.jl` - Alternative implementation
- `run_all_cells.jl` - For notebook use

### Power Flow Models
- PowerFlow Model.ipynb  
  Base model for power flow analysis with demand uncertainty

- PowerFlow with Generator Uncertainty.ipynb: 
  Extends the model with generator failure scenarios(Randomness in generators)

- Powerflow with additional cost uncertainty.ipynb:  
  Powerflow Model with uncertainty in costs 

## Input Data

- `instances/`: Contains instances for facility location problem.
- `powerflow instances/`: Input files specific to power flow simulations.

## Requirements

- Julia 1.6+
- Gurobi solver (academic license recommended)
- Required Julia packages: JuMP, Gurobi, Random, Distributions, DataFrames, LinearAlgebra
