# Complete Workflow for Facility Location Column Generation

This workflow guides you through the entire process from scenario generation to column generation evaluation.

## Prerequisites
- Instance file: `cap91.txt` (or any other facility location instance)
- Parameters:
  - `S`: Number of scenarios (e.g., 25)
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

## Step 1: Generate Scenarios

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
- `Samples/FL/Scenarios/S25_bernoulli.txt` (scenario file)

---

## Step 2: Solve Scenarios from Scratch (Wait-and-See)

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
- `Samples/FL/Scenarios/WS25_bernoulli.txt` (wait-and-see costs)
- `Samples/FL/Binary_vectors/B25_bernoulli.txt` (binary vectors from solving scenarios)

---

## Step 3: Generate Candidate List (Phase 1)

**Location**: `Facility_Location/Heuristic/`

**Command**:
```bash
cd ../../Facility_Location/Heuristic
julia Candidate_List_Generation_Phase.jl <instance_file> <L> [experiment_type] [unmet_pen] [scaling_factor] [capacity_factor]
```

**Examples**:
```bash
# Using auto-detected parameters (recommended)
cd ../../Facility_Location/Heuristic
julia Candidate_List_Generation_Phase.jl cap91.txt 100 bernoulli
julia Candidate_List_Generation_Phase.jl capa11.txt 100 bernoulli  # Uses FL2 defaults automatically

# Overriding parameters explicitly (if needed)
julia Candidate_List_Generation_Phase.jl cap91.txt 100 bernoulli 20.0 0.3 0.4
```

**Output**:
- `Samples/FL/Binary_vectors/L100_bernoulli.txt` (candidate binary vectors)

---

## Step 4: Assignment Phase (Select K Candidates)

**Location**: `Facility_Location/Heuristic/`

**Command**:
```bash
julia Assignment_Phase.jl <instance_file> <S> <K> <binary_vectors_file> [samples_file] [experiment_type] [unmet_pen] [scaling_factor] [capacity_factor]
```

**Examples**:
```bash
# Using auto-detected parameters (recommended)
julia Assignment_Phase.jl cap91.txt 25 2 ../../Samples/FL/Binary_vectors/L100_bernoulli.txt ../../Samples/FL/Scenarios/S25_bernoulli.txt bernoulli
julia Assignment_Phase.jl capa11.txt 25 2 ../../Samples/FL/Binary_vectors/L100_bernoulli.txt ../../Samples/FL/Scenarios/S25_bernoulli.txt bernoulli  # Uses FL2 defaults automatically

# Overriding parameters explicitly (if needed)
julia Assignment_Phase.jl cap91.txt 25 2 ../../Samples/FL/Binary_vectors/L100_bernoulli.txt ../../Samples/FL/Scenarios/S25_bernoulli.txt bernoulli 20.0 0.3 0.4
```

**Output**:
- `Samples/FL/selected_candidates_S25_K2.txt` (K selected binary vectors)

---

## Step 5: Evaluation Phase (Compare Heuristic vs Wait-and-See)

**Location**: `Facility_Location/Heuristic/`

**Command**:
```bash
julia Evaluation_Phase.jl <instance_file> <S> <K> <selected_policies_file> <samples_file> <wait_and_see_costs_file> [experiment_type] [unmet_pen] [scaling_factor] [capacity_factor]
```

**Examples**:
```bash
# Using auto-detected parameters (recommended)
julia Evaluation_Phase.jl cap91.txt 25 2 ../../Samples/FL/selected_candidates_S25_K2.txt ../../Samples/FL/Scenarios/S25_bernoulli.txt ../../Samples/FL/Scenarios/WS25_bernoulli.txt bernoulli
julia Evaluation_Phase.jl capa11.txt 25 2 ../../Samples/FL/selected_candidates_S25_K2.txt ../../Samples/FL/Scenarios/S25_bernoulli.txt ../../Samples/FL/Scenarios/WS25_bernoulli.txt bernoulli  # Uses FL2 defaults automatically

# Overriding parameters explicitly (if needed)
julia Evaluation_Phase.jl cap91.txt 25 2 ../../Samples/FL/selected_candidates_S25_K2.txt ../../Samples/FL/Scenarios/S25_bernoulli.txt ../../Samples/FL/Scenarios/WS25_bernoulli.txt bernoulli 20.0 0.3 0.4
```

**Output**:
- `Facility_Location/Heuristic/results.txt` (comparison results with model gap)

---

## Step 6: Create Partitions for SF

**Location**: `Samples/FL/`

**Command**:
```bash
cd ../../Samples/FL
julia Create_Partitions.jl <instance_file> <scenarios_file> <candidates_file> [unmet_pen] [scaling_factor] [capacity_factor]
```

**Example**:
```bash
cd ../../Samples/FL
julia Create_Partitions.jl Facility_Location/instances/cap91.txt Scenarios/S25_bernoulli.txt selected_candidates_S25_K2.txt
```

**Output**:
- `Samples/FL/SubsetS25_K2_FL.txt` (partitions file for SF initialization)

---

## Step 7: Column Generation - Assignment Formulation (AF)

**Location**: `CCG/`

**Command**:
```bash
cd ../../CCG
julia CG_AF_FL.jl <instance_file> <scenarios_file> <binary_vectors_file> <K> <S> [pricing_type] [unmet_pen] [scaling_factor] [capacity_factor]
```

**Example**:
```bash
cd ../../CCG
julia CG_AF_FL.jl Facility_Location/instances/cap91.txt Samples/FL/Scenarios/S25_bernoulli.txt Samples/FL/selected_candidates_S25_K2.txt 2 25
```

**Output**:
- Column generation results (console output and log files)
- Logs in `CCG/logs/`

---

## Step 8: Column Generation - Subset Formulation (SF)

**Location**: `CCG/`

**Command**:
```bash
julia CG_SF_FL.jl <instance_file> <scenarios_file> <initial_subsets_file> <K> <S> [pricing_type] [unmet_pen] [scaling_factor] [capacity_factor] [init_type]
```

**Example**:
```bash
julia CG_SF_FL.jl Facility_Location/instances/cap91.txt Samples/FL/Scenarios/S25_bernoulli.txt Samples/FL/SubsetS25_K2_FL.txt 2 25 2 15.0 0.2 0.3 3
```

**Note**: `init_type=3` means loading initial subsets from file.

**Output**:
- Column generation results (console output and log files)
- Logs in `CCG/logs/`

---

## Complete Workflow Example (All Steps)

Here's a complete example running all steps with S=25, K=2, experiment_type=bernoulli:

```bash
# Step 1: Generate scenarios
cd Samples/FL
julia Samples.jl cap91.txt 25 bernoulli

# Step 2: Solve scenarios from scratch
julia WS_Samples.jl cap91.txt Scenarios/S25_bernoulli.txt bernoulli

# Step 3: Generate candidate list (L=100)
cd ../../Facility_Location/Heuristic
julia Candidate_List_Generation_Phase.jl cap91.txt 100 bernoulli

# Step 4: Assignment phase (select K=2 candidates)
julia Assignment_Phase.jl cap91.txt 25 2 ../../Samples/FL/Binary_vectors/L100_bernoulli.txt ../../Samples/FL/Scenarios/S25_bernoulli.txt bernoulli

# Step 5: Evaluation phase
julia Evaluation_Phase.jl cap91.txt 25 2 ../../Samples/FL/selected_candidates_S25_K2.txt ../../Samples/FL/Scenarios/S25_bernoulli.txt ../../Samples/FL/Scenarios/WS25_bernoulli.txt bernoulli

# Step 6: Create partitions
cd ../../Samples/FL
julia Create_Partitions.jl Facility_Location/instances/cap91.txt Scenarios/S25_bernoulli.txt selected_candidates_S25_K2.txt

# Step 7: Column Generation - AF
cd ../../CCG
julia CG_AF_FL.jl Facility_Location/instances/cap91.txt Samples/FL/Scenarios/S25_bernoulli.txt Samples/FL/selected_candidates_S25_K2.txt 2 25

# Step 8: Column Generation - SF 
julia CG_SF_FL.jl Facility_Location/instances/cap91.txt Samples/FL/Scenarios/S25_bernoulli.txt Samples/FL/SubsetS25_K2_FL.txt 2 25 2 15.0 0.2 0.3 3
```

---

## File Naming Conventions

- **Scenarios**: `S<S>_<experiment_type>.txt` (e.g., `S25_bernoulli.txt`)
- **Wait-and-see costs**: `WS<S>_<experiment_type>.txt` (e.g., `WS25_bernoulli.txt`)
- **Binary vectors (from WS)**: `B<S>_<experiment_type>.txt` (e.g., `B25_bernoulli.txt`)
- **Binary vectors (candidate list)**: `L<L>_<experiment_type>.txt` (e.g., `L100_bernoulli.txt`)
- **Selected candidates**: `selected_candidates_S<S>_K<K>.txt` (e.g., `selected_candidates_S25_K2.txt`)
- **Partitions**: `SubsetS<S>_K<K>_FL.txt` (e.g., `SubsetS25_K2_FL.txt`)

---

## Output Locations

- **Scenarios**: `Samples/FL/Scenarios/`
- **Binary vectors**: `Samples/FL/Binary_vectors/`
- **Selected candidates**: `Samples/FL/`
- **Partitions**: `Samples/FL/`
- **Evaluation results**: `Facility_Location/Heuristic/results.txt`
- **Column generation logs**: `CCG/logs/`

---

## Notes

1. All paths in commands are relative to the current working directory.
2. **Parameter Auto-Detection**: The parameters `unmet_pen`, `scaling_factor`, and `capacity_factor` are automatically detected based on the instance file:
   - For `cap91.txt` (ORLIB format): Uses FL1 defaults (15.0, 0.2, 0.3)
   - For `capa11.txt` (Beasley format): Uses FL2 defaults (100000.0, 1.0, 0.05)
   - You can still override them explicitly if needed
3. Make sure to use consistent parameters across all steps (or let auto-detection handle it).
4. The experiment_type should match across all steps that use it.
5. For SF column generation, `init_type=3` is required to load initial subsets from file (this parameter was removed from CLI).
6. The tolerance for convergence in column generation is now:
   - AF: `reduced_cost < 1`
   - SF: `reduced_cost >= -1`

