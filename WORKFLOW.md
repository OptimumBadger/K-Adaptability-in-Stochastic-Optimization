# Complete Workflow for Facility Location Column Generation

This workflow guides you through the entire process from scenario generation to column generation evaluation.

## Prerequisites
- Instance file: `cap91.txt` (or any other facility location instance)
- Parameters:
  - `S`: Number of scenarios to generate in master file (e.g., 300)
  - `L`: Number of candidate policies to use (e.g., 50)
  - `S_slice`: Number of scenarios to use for SAA (e.g., 30)
  - `K`: Number of policies to select (e.g., 2)
  - `experiment_type`: Experiment type (e.g., "bernoulli")
  - `unmet_pen`: Unmet demand penalty (auto-detected from instance file, can be overridden)
    - `cap91.txt` (ORLIB): default 15.0
    - `capa11.txt` (Beasley): default 100000.0
  - `scaling_factor`: Scaling factor for transportation cost (auto-detected from instance file, can be overridden)
    - `cap91.txt` (ORLIB): default 0.2
    - `capa11.txt` (Beasley): default 1.0
  - `capacity_factor`: Capacity factor (auto-detected from instance file, can be overridden)
    - `cap91.txt` (ORLIB): default 0.3
    - `capa11.txt` (Beasley): default 0.05
  - `pricing_type`: Pricing type for CG (1 or 2, default: 2)

---

## Step 1: Generate Scenarios (One-Time)

**Location**: `Samples/FL/`

**Command**:
```bash
cd Samples/FL
julia Samples.jl <instance_file> <S> [experiment_type]
```

**Example**:
```bash
cd Samples/FL
julia Samples.jl cap91.txt 25 bernoulli
```

**Output**: 
- `Samples/FL/Scenarios/S<S>_<experiment_type>.txt` (e.g., `S300_bernoulli.txt`)
  - **Note**: This is the master scenario file. Different slices (S_slice) will be used in later steps.

---

## Step 2: Calculate Wait-and-See Costs (One-Time)

**Location**: `Samples/FL/`

**Command**:
```bash
julia WS_Samples.jl <instance_file> <scenarios_file> <experiment_type>
```

**Example**:
```bash
julia WS_Samples.jl cap91.txt Scenarios/S25_bernoulli.txt bernoulli
```

**Output**:
- `Samples/FL/Scenarios/WS<S>_<experiment_type>.txt` (e.g., `WS300_bernoulli.txt`) - wait-and-see costs
- `Samples/FL/Binary_vectors/B<S>_<experiment_type>.txt` (e.g., `B300_bernoulli.txt`) - binary vectors from solving scenarios
  - **Note**: These binary vectors are used directly as candidate policies in Step 3

---

## Step 3: Assignment Phase (Select K Candidates)

**Location**: `Facility_Location/Heuristic/`

**Command**:
```bash
julia Assignment_Phase.jl <instance_file> <L> <S> <K> <binary_vectors_file> <samples_file> [T] [unmet_pen] [scaling_factor] [capacity_factor]
```

**Key arguments**:
- `L`: Number of candidate policies to use (first `L` rows of `<binary_vectors_file>`)
- `S`: Number of scenarios to use (first `S` rows of `<samples_file>`)
- `K`: Number of policies to select
- `binary_vectors_file`: Path to binary vectors file (e.g., `B300_bernoulli.txt` from Step 2)
- `samples_file`: Path to scenarios file (e.g., `S300_bernoulli.txt` from Step 1)
- `T`: Number of feasible solutions to collect from solution pool (optional, default: 10)

**Example**:
```bash
# Use first L=50 candidates and first S=30 scenarios (K=2 selected policies, T=10 feasible solutions)
cd Facility_Location/Heuristic
julia Assignment_Phase.jl cap91.txt 50 30 2 ../../Samples/FL/Binary_vectors/B300_bernoulli.txt ../../Samples/FL/Scenarios/S300_bernoulli.txt
```

**Output**:
- `Samples/FL/Selected_Candidates/selected_candidates_L<L>_S<S>_K<K>_<experiment>.txt` (e.g., `selected_candidates_L50_S30_K2_bernoulli.txt`) - optimal solution
- `Samples/FL/Feasible_Solutions/feasible_solutions_L<L>_S<S>_K<K>_T<T>_<experiment>.txt` (e.g., `feasible_solutions_L50_S30_K2_T10_bernoulli.txt`) - up to T feasible solutions from Gurobi's solution pool
- `Results/Facility_Location/assignment_results_<instance>_L<L>_S<S>_K<K>_<experiment>.json` (contains `work_units_continuous` and `assignment_work_units`)

**Important Notes**:
- Binary vectors come directly from Step 2 (wait-and-see solutions)
- This phase **slices** the first `L` candidates from the binary vectors file
- This phase **slices** the first `S` scenarios from the scenarios file
- The `samples_file` argument is **required** (scenarios are not generated inside this script)
- The feasible solutions file contains T solutions (or fewer if duplicates are filtered), each with K binary vectors
- Duplicate solutions are automatically filtered out

---

## Step 4: Evaluation Phase (Compare Heuristic vs Wait-and-See)

**Location**: `Facility_Location/Heuristic/`

**Command**:
```bash
julia Evaluation_Phase.jl <instance_file> <S> <K> <selected_policies_file> <samples_file> <wait_and_see_costs_file> [experiment_type] [unmet_pen] [scaling_factor] [capacity_factor]
```

**Examples**:
```bash
# Using auto-detected parameters (recommended)
julia Evaluation_Phase.jl cap91.txt 25 2 ../../Samples/FL/Selected_Candidates/selected_candidates_L50_S25_K2_bernoulli.txt ../../Samples/FL/Scenarios/S25_bernoulli.txt ../../Samples/FL/Scenarios/WS25_bernoulli.txt bernoulli
julia Evaluation_Phase.jl capa11.txt 25 2 ../../Samples/FL/Selected_Candidates/selected_candidates_L50_S25_K2_bernoulli.txt ../../Samples/FL/Scenarios/S25_bernoulli.txt ../../Samples/FL/Scenarios/WS25_bernoulli.txt bernoulli  # Uses FL2 defaults automatically

# Overriding parameters explicitly (if needed)
julia Evaluation_Phase.jl cap91.txt 25 2 ../../Samples/FL/Selected_Candidates/selected_candidates_L50_S25_K2_bernoulli.txt ../../Samples/FL/Scenarios/S25_bernoulli.txt ../../Samples/FL/Scenarios/WS25_bernoulli.txt bernoulli 20.0 0.3 0.4
```

**Output**:
- `Facility_Location/Heuristic/Results/evaluation_results_<instance>_L<L>_S<S>_K<K>_<experiment>.txt` (comparison results with model gap and work units)
- `Results/Facility_Location/evaluation_results_<instance>_L<L>_S<S>_K<K>_<experiment>.json` (aggregated metrics incl. WS/model costs + work units)

---

## Step 5: Create Partitions for SF

**Location**: `Samples/FL/`

**Command**:
```bash
julia Create_Partitions.jl <instance_file> <scenarios_file> <S> <feasible_solutions_file> [unmet_pen] [scaling_factor] [capacity_factor]
```

**Key arguments**:
- `instance_file`: Instance file path (e.g., `Facility_Location/instances/cap91.txt`)
- `scenarios_file`: Master scenarios file (e.g., `S300_bernoulli.txt`)
- `S`: Number of scenarios to use (first `S` scenarios from `<scenarios_file>`)
- `feasible_solutions_file`: File containing feasible solutions from Step 3 (e.g., `Feasible_Solutions/feasible_solutions_L50_S30_K2_T10_bernoulli.txt`)

**Example**:
```bash
cd ../../Samples/FL
julia Create_Partitions.jl Facility_Location/instances/cap91.txt Scenarios/S300_bernoulli.txt 30 Feasible_Solutions/feasible_solutions_L50_S30_K2_T10_bernoulli.txt
```

**Output**:
- `Samples/FL/InitialSubsets_L<L>_S<S>_K<K>_T<T_actual>_FL.txt` (e.g., `InitialSubsets_L50_S30_K2_T3_FL.txt`)
  - Contains T_actual × K subsets (one per line), where T_actual is the number of unique solutions found
  - Each line contains space-separated scenario indices (1-based) assigned to that subset

**Important Notes**:
- This phase **slices** the first `S` scenarios from the master scenarios file
- The feasible solutions file should come from Step 3 (Assignment Phase)
- Each solution in the feasible solutions file has K binary vectors
- For each solution, scenarios are assigned to the vector that minimizes cost
- This creates K subsets per solution, resulting in T_actual × K total subsets

---

## Step 6: Column Generation - Assignment Formulation (AF)

**Location**: `CCG/`

**Command**:
```bash
cd ../../CCG
julia CG_AF_FL.jl <instance_file> <scenarios_file> <binary_vectors_file> <L> <S> <K> [pricing_type] [unmet_pen] [scaling_factor] [capacity_factor]
```

**Key arguments**:
- `L`: Number of candidate policies to use (first `L` from `<binary_vectors_file>`)
- `S`: Number of scenarios to use (first `S` from `<scenarios_file>`)
- `K`: Number of solutions to select
- `binary_vectors_file`: Path to binary vectors file (e.g., `B300_bernoulli.txt` from Step 2)
- `scenarios_file`: Path to scenarios file (e.g., `S300_bernoulli.txt` from Step 1)

**Example**:
```bash
cd ../../CCG
julia CG_AF_FL.jl Facility_Location/instances/cap91.txt Samples/FL/Scenarios/S300_bernoulli.txt Samples/FL/Binary_vectors/B300_bernoulli.txt 50 30 2
```

**Output**:
- Column generation results (console output and log files)
- Logs in `CCG/logs/` (named `CG_AF_FL_<instance>_L<L>_S<S>_K<K>_master_<timestamp>.log`)

**Important Notes**:
- Scenario generation happens **only once** in Step 1
- This phase **slices** the first `S` scenarios from the master scenario file
- This phase **slices** the first `L` candidate policies from the binary vectors file
- Binary vectors come from wait-and-see solutions (Step 2), not from selected candidates

---

## Step 7: Column Generation - Subset Formulation (SF)

**Location**: `CCG/`

**Command**:
```bash
julia CG_SF_FL.jl <instance_file> <scenarios_file> <initial_subsets_file> <K> <S> [pricing_type] [unmet_pen] [scaling_factor] [capacity_factor]
```

**Key arguments**:
- `initial_subsets_file`: File containing initial subsets from Step 5 (e.g., `InitialSubsets_L50_S30_K2_T3_FL.txt`)
- `S`: Number of scenarios to use (should match the S used in Step 5)

**Example**:
```bash
julia CG_SF_FL.jl Facility_Location/instances/cap91.txt Samples/FL/Scenarios/S300_bernoulli.txt Samples/FL/InitialSubsets_L50_S30_K2_T3_FL.txt 2 30 2 15.0 0.2 0.3
```

**Output**:
- Column generation results (console output and log files)
- Logs in `CCG/logs/`

**Important Notes**:
- The initial subsets file contains T_actual × K subsets (one per line)
- Each subset is a list of scenario indices (1-based) assigned to a particular vector from a feasible solution
- The `S` parameter should match the S used when creating the partitions in Step 5

---

## Complete Workflow Example (All Steps)

Here's a complete example running all steps with S=300 (master), L=50, S_slice=30, K=2, experiment_type=bernoulli:

```bash
# Step 1: Generate scenarios (master file with S=300 scenarios)
cd Samples/FL
julia Samples.jl cap91.txt 300 bernoulli

# Step 2: Solve scenarios from scratch (wait-and-see)
julia WS_Samples.jl cap91.txt Scenarios/S300_bernoulli.txt bernoulli

# Step 3: Assignment phase (use first L=50 candidates, first S=30 scenarios, select K=2)
cd ../../Facility_Location/Heuristic
julia Assignment_Phase.jl cap91.txt 50 30 2 ../../Samples/FL/Binary_vectors/B300_bernoulli.txt ../../Samples/FL/Scenarios/S300_bernoulli.txt

# Step 4: Evaluation phase
julia Evaluation_Phase.jl cap91.txt 30 2 ../../Samples/FL/Selected_Candidates/selected_candidates_L50_S30_K2_bernoulli.txt ../../Samples/FL/Scenarios/S300_bernoulli.txt ../../Samples/FL/Scenarios/WS300_bernoulli.txt bernoulli

# Step 5: Create partitions (use first S=30 scenarios, from feasible solutions file)
cd ../../Samples/FL
julia Create_Partitions.jl Facility_Location/instances/cap91.txt Scenarios/S300_bernoulli.txt 30 Feasible_Solutions/feasible_solutions_L50_S30_K2_T10_bernoulli.txt

# Step 6: Column Generation - AF (use first L=50 candidates, first S=30 scenarios, select K=2)
cd ../../CCG
julia CG_AF_FL.jl ../Facility_Location/instances/cap91.txt ../Samples/FL/Scenarios/S300_bernoulli.txt ../Samples/FL/Binary_vectors/B300_bernoulli.txt 50 30 2

# Step 7: Column Generation - SF (use initial subsets from Step 5)
julia CG_SF_FL.jl Facility_Location/instances/cap91.txt Samples/FL/Scenarios/S300_bernoulli.txt Samples/FL/InitialSubsets_L50_S30_K2_T3_FL.txt 2 30 2 15.0 0.2 0.3
```

---

## File Naming Conventions

### Facility Location (FL):
- **Scenarios**: `S<S>_<experiment_type>.txt` (e.g., `S300_bernoulli.txt`)
- **Wait-and-see costs**: `WS<S>_<experiment_type>.txt` (e.g., `WS300_bernoulli.txt`)
- **Binary vectors (from WS)**: `B<S>_<experiment_type>.txt` (e.g., `B300_bernoulli.txt`)
  - Used directly as candidate policies (no separate candidate generation phase)
- **Selected candidates**: `Selected_Candidates/selected_candidates_L<L>_S<S>_K<K>_<experiment>.txt` (e.g., `selected_candidates_L50_S30_K2_bernoulli.txt`)
- **Feasible solutions**: `Feasible_Solutions/feasible_solutions_L<L>_S<S>_K<K>_T<T>_<experiment>.txt` (e.g., `feasible_solutions_L50_S30_K2_T10_bernoulli.txt`)
- **Initial partitions**: `InitialSubsets_L<L>_S<S>_K<K>_T<T_actual>_FL.txt` (e.g., `InitialSubsets_L50_S30_K2_T3_FL.txt`)
  - T_actual is the number of unique solutions found (may be less than T parameter)

### Transmission Switching (TS):
- **Scenarios**: `S<S>_<case_name>_<experiment_type>.txt` (e.g., `S100_case57_bernoulli.txt`)
  - Includes case name to distinguish between different MATPOWER instances
- **Wait-and-see costs**: `WS<S>_<experiment_type>.txt` (e.g., `WS100_bernoulli.txt`)
  - Format: 3-column table (`scenario_index`, `wait_and_see_cost`, `work_units`)
- **Binary vectors (from WS)**: `B<S>_<experiment_type>.txt` (e.g., `B100_bernoulli.txt`)
  - Only successful solves are saved
- **Selected candidates**: `selected_candidates_S<S>_K<K>.txt` (e.g., `selected_candidates_S30_K2.txt`)

---

## Output Locations

### Facility Location (FL):
- **Scenarios**: `Samples/FL/Scenarios/`
- **Binary vectors**: `Samples/FL/Binary_vectors/`
- **Selected candidates**: `Samples/FL/`
- **Feasible solutions**: `Samples/FL/`
- **Initial partitions**: `Samples/FL/`
- **Evaluation results**: `Facility_Location/Heuristic/results.txt`
- **Column generation logs**: `CCG/logs/`

### Transmission Switching (TS):
- **Scenarios**: `Samples/TS/Scenarios/`
- **Binary vectors**: `Samples/TS/Binary_vectors/`
- **Selected candidates**: `Samples/TS/`
- **Evaluation results**: `Transmission_Switching/Heuristic/results.txt`

---

## Notes

1. All paths in commands are relative to the current working directory.
2. **Parameter Auto-Detection**: The parameters `unmet_pen`, `scaling_factor`, and `capacity_factor` are automatically detected based on the instance file:
   - For `cap91.txt` (ORLIB format): Uses FL1 defaults (15.0, 0.2, 0.3)
   - For `capa11.txt` (Beasley format): Uses FL2 defaults (100000.0, 1.0, 0.05)
   - You can still override them explicitly if needed
3. **Scenario Generation**: Happens **only once** in Step 1. The master scenario file is then reused with different slices (S) in later steps.
4. **Binary Vectors**: Come directly from wait-and-see solutions (Step 2). No separate candidate generation phase is needed.
5. **Slicing**: Both Assignment Phase (Step 3) and Column Generation (Step 6) use `L` and `S` to slice the first `L` candidates and first `S` scenarios from the master files.
6. **Feasible Solutions**: The Assignment Phase (Step 3) now outputs both the optimal solution and a file containing up to T feasible solutions from Gurobi's solution pool. Duplicate solutions are automatically filtered out.
7. **Create Partitions**: Step 5 now takes the feasible solutions file (not the selected candidates file) and creates partitions by assigning scenarios to vectors based on minimum cost. Each solution creates K subsets, resulting in T_actual × K total subsets.
8. Make sure to use consistent parameters across all steps (or let auto-detection handle it).
9. The experiment_type should match across all steps that use it.
10. For SF column generation, initial subsets are always loaded from the file specified in the `initial_subsets_file` parameter.
11. The tolerance for convergence in column generation is now:
    - AF: `reduced_cost < 1`
    - SF: `reduced_cost >= -1`

---

# Complete Workflow for Transmission Switching

This workflow guides you through the entire process for Transmission Switching problems, from scenario generation to evaluation.

## Prerequisites (TS)
- Instance file: MATPOWER case file (e.g., `case57.m`, `case118Blumsack.m`)
- Parameters:
  - `S`: Number of scenarios to generate (e.g., 100)
  - `L`: Number of candidate policies to use (e.g., 50)
  - `K`: Number of policies to select (e.g., 2)
  - `experiment_type`: Experiment type (e.g., "bernoulli" or "gaussian")
  - `sigma_low`, `sigma_high`: Parameters for scenario generation (optional, defaults: 0.04, 0.06)

---

## TS Step 1: Generate Scenarios

**Location**: `Samples/TS/`

**Command**:
```bash
cd Samples/TS
julia Samples.jl <instance_file> <S> [sigma_low] [sigma_high] [bernoulli]
```

**Example**:
```bash
cd Samples/TS
julia Samples.jl case57.m 100 0.04 0.06 true
```

**Output**: 
- `Samples/TS/Scenarios/S<S>_<case_name>_<experiment_type>.txt` (e.g., `S100_case57_bernoulli.txt`)
  - **Note**: The filename includes the case name to distinguish between different instances

---

## TS Step 2: Solve Scenarios from Scratch (Wait-and-See)

**Location**: `Samples/TS/`

**Command**:
```bash
julia WS_Samples.jl <instance_file> <scenarios_file> <experiment_type>
```

**Example**:
```bash
julia WS_Samples.jl case57.m Scenarios/S100_case57_bernoulli.txt bernoulli
```

**Output**:
- `Samples/TS/Scenarios/WS<S>_<experiment_type>.txt` (wait-and-see costs file)
  - **Format**: 3-column table with `scenario_index`, `wait_and_see_cost`, and `work_units`
- `Samples/TS/Binary_vectors/B<S>_<experiment_type>.txt` (binary vectors from solving scenarios)
  - **Note**: Only binary vectors from successful solves are saved

---

## TS Step 3: Assignment Phase (Select K Candidates)

**Location**: `Transmission_Switching/Heuristic/`

**Command**:
```bash
julia Assignment_Phase.jl <instance_file> <L> <S> <K> <binary_vectors_file> <samples_file>
```

**Key arguments**:
- `L`: Number of candidate policies to use (first `L` rows of `<binary_vectors_file>`)
- `S`: Number of scenarios to use (first `S` rows of `<samples_file>`)
- `K`: Number of policies to select
- `binary_vectors_file`: Path to binary vectors file (e.g., from Step 2 or candidate generation)
- `samples_file`: Path to scenarios file (from Step 1)

**Example**:
```bash
# Use first L=50 candidates and first S=30 scenarios (K=2 selected policies)
cd Transmission_Switching/Heuristic
julia Assignment_Phase.jl case57.m 50 30 2 ../Samples/TS/Binary_vectors/B100_bernoulli.txt ../Samples/TS/Scenarios/S100_case57_bernoulli.txt
```

**Output**:
- `Samples/TS/selected_candidates_S<S>_K<K>.txt` (K selected binary vectors)

**Important Notes**:
- Scenario generation happens **only once** in Step 1
- This phase **slices** the first `S` scenarios from the master scenario file
- This phase **slices** the first `L` candidate policies from the binary vectors file
- The `samples_file` argument is **required** (scenarios are not generated inside this script)

---

## TS Step 4: Evaluation Phase (Compare Heuristic vs Wait-and-See)

**Location**: `Transmission_Switching/Heuristic/`

**Command**:
```bash
julia Evaluation_Phase.jl <instance_file> <S> <K> <selected_policies_file> <samples_file> <wait_and_see_costs_file>
```

**Example**:
```bash
julia Evaluation_Phase.jl case57.m 30 2 ../../Samples/TS/selected_candidates_S30_K2.txt ../../Samples/TS/Scenarios/S100_case57_bernoulli.txt ../../Samples/TS/Scenarios/WS100_bernoulli.txt
```

**Output**:
- `Transmission_Switching/Heuristic/results.txt` (comparison results with model gap)

**Note**: The wait-and-see costs file format is automatically parsed (3-column format with `scenario_index`, `wait_and_see_cost`, `work_units`)

---

## Complete TS Workflow Example

Here's a complete example running all TS steps with S=100, L=50, S_slice=30, K=2, experiment_type=bernoulli:

```bash
# Step 1: Generate scenarios (master file with S=100 scenarios)
cd Samples/TS
julia Samples.jl case57.m 100 0.04 0.06 true

# Step 2: Solve scenarios from scratch (wait-and-see)
julia WS_Samples.jl case57.m Scenarios/S100_case57_bernoulli.txt bernoulli

# Step 3: Assignment phase (use first L=50 candidates, first S=30 scenarios, select K=2)
cd ../../Transmission_Switching/Heuristic
julia Assignment_Phase.jl case57.m 50 30 2 ../Samples/TS/Binary_vectors/B100_bernoulli.txt ../Samples/TS/Scenarios/S100_case57_bernoulli.txt

# Step 4: Evaluation phase
julia Evaluation_Phase.jl case57.m 30 2 ../../Samples/TS/selected_candidates_S30_K2.txt ../../Samples/TS/Scenarios/S100_case57_bernoulli.txt ../../Samples/TS/Scenarios/WS100_bernoulli.txt
```

---

## TS File Naming Conventions

- **Scenarios**: `S<S>_<case_name>_<experiment_type>.txt` (e.g., `S100_case57_bernoulli.txt`)
  - Includes case name to distinguish instances
- **Wait-and-see costs**: `WS<S>_<experiment_type>.txt` (e.g., `WS100_bernoulli.txt`)
  - Format: 3-column table (`scenario_index`, `wait_and_see_cost`, `work_units`)
- **Binary vectors (from WS)**: `B<S>_<experiment_type>.txt` (e.g., `B100_bernoulli.txt`)
  - Only successful solves are saved
- **Selected candidates**: `selected_candidates_S<S>_K<K>.txt` (e.g., `selected_candidates_S30_K2.txt`)

---

## TS Output Locations

- **Scenarios**: `Samples/TS/Scenarios/`
- **Binary vectors**: `Samples/TS/Binary_vectors/`
- **Selected candidates**: `Samples/TS/`
- **Evaluation results**: `Transmission_Switching/Heuristic/results.txt`

---

## TS Workflow Notes

1. **Scenario Generation**: Happens **only once** in Step 1. The master scenario file is then reused with different slices.
2. **Slicing**: The Assignment Phase uses `L` and `S` to slice the first `L` candidates and first `S` scenarios from the master files.
3. **File Naming**: Scenario files include the case name (e.g., `case57`) to distinguish between different MATPOWER instances.
4. **Wait-and-See Format**: The WS costs file uses a 3-column format with `scenario_index`, `wait_and_see_cost`, and `work_units`.
5. **Binary Vectors**: Only binary vectors from successful wait-and-see solves are saved.
6. **No Candidate Generation Phase**: Unlike FL, TS does not have a separate candidate list generation phase. Candidates come from wait-and-see solutions or can be generated separately if needed.

