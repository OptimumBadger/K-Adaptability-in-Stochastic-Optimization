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

4. **Run the script**:
   ```bash
   julia complete_facility_location.jl
   ```

## Files

- `complete_facility_location.jl` - **Main script (recommended)**
- `facility_location.ipynb` - Original notebook implementation

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
