# Run All Cells - Facility Location Problem
# This script can be executed in a Julia notebook or REPL to run all computations at once

println("="^60)
println("STOCHASTIC FACILITY LOCATION PROBLEM - AUTOMATED RUN")
println("="^60)

# Load all required packages
using JuMP, Gurobi, Random, Distributions, DataFrames, LinearAlgebra

# Set random seed for reproducibility
Random.seed!(1234)

# =============================================================================
# CONFIGURATION
# =============================================================================
println("Setting up parameters...")

# File path - update this to your actual file path
file_path = "/Users/Patron/code/Candidate-selection-via-Stochastic-IP/instances/cap91.txt"

# Problem parameters
unmet_pen = 50
scaling_factor = 0.000015
bernoulli_case = true

# Sampling parameters
nscen = 1000  # Number of scenarios for sampling
nsamples = 100  # Number of online samples
test_samples = 300  # Number of test samples
K = 5  # Number of solutions to select
max_iter = 10  # Maximum iterations for heuristic

# =============================================================================
# PHASE 1: DATA LOADING AND SETUP
# =============================================================================
println("\n" * "="^40)
println("PHASE 1: DATA LOADING AND SETUP")
println("="^40)

# Read data
println("Loading data from: $file_path")
capacity, fixedcost, cost, demandmean = read_orlib_cap(file_path)
nloc = length(fixedcost)
ncust = length(demandmean)
println("Problem size: $nloc facilities, $ncust customers")

# Generate demand standard deviations
theta = rand(Uniform(0, 0.5), ncust)
demandstdev = theta .* demandmean
println("Generated demand uncertainty parameters")

# =============================================================================
# PHASE 2: SCENARIO GENERATION
# =============================================================================
println("\n" * "="^40)
println("PHASE 2: SCENARIO GENERATION")
println("="^40)

println("Generating $nscen scenarios...")
xsolutions = sampling(nscen, nloc, demandmean, demandstdev, capacity, cost, fixedcost, unmet_pen, scaling_factor, bernoulli_case)

# Count unique solutions
_, u1 = n_unique_rows(xsolutions)
println("Generated $nscen scenarios with $u1 unique solutions")

# Generate online samples
println("Generating $nsamples online samples...")
online_samples = zeros(nsamples, ncust)
for s in 1:nsamples
    online_samples[s, :] = generate_demand_scenario(ncust, demandmean, demandstdev, bernoulli_case)
end

# Generate test samples
println("Generating $test_samples test samples...")
test_samples_data = zeros(test_samples, ncust)
for s in 1:test_samples
    test_samples_data[s, :] = generate_demand_scenario(ncust, demandmean, demandstdev, bernoulli_case)
end

# =============================================================================
# PHASE 3: REFORMULATED EXTENSIVE FORM
# =============================================================================
println("\n" * "="^40)
println("PHASE 3: REFORMULATED EXTENSIVE FORM")
println("="^40)

# Solve reformulated extensive form
obj_matrix, r = Calculations_For_Extensive_Form(nloc, ncust, capacity, cost, online_samples, xsolutions, fixedcost, unmet_pen, scaling_factor)
n_neg1 = count(x -> x == -1, obj_matrix)
println("Infeasible scenario-solution pairs: $n_neg1")

Solution, z = Reformulated_Extensive_Form(nsamples, nscen, K, obj_matrix)
println("Reformulated extensive form objective: $Solution")

# =============================================================================
# PHASE 4: EVALUATION
# =============================================================================
println("\n" * "="^40)
println("PHASE 4: EVALUATION")
println("="^40)

# Calculate offline costs
offline_cost = OfflineTest(nloc, ncust, capacity, cost, test_samples_data, fixedcost, unmet_pen, scaling_factor)
println("Average offline cost: $offline_cost")

# Evaluate model performance
modelcost, pcos, mcost = Evaluation(nloc, ncust, cost, capacity, test_samples_data, xsolutions, z, fixedcost, unmet_pen, scaling_factor)
println("Average model cost: $modelcost")

# Calculate gap
gap = (modelcost - offline_cost) / offline_cost * 100
println("Performance gap: $(round(gap, digits=2))%")

# Analyze penalties
penalty_pos = findall(x -> x > 0.1, pcos)
penalty_val = pcos[penalty_pos]
println("Scenarios with penalties: $(length(penalty_pos))")

# =============================================================================
# PHASE 5: L-AUGMENTATION HEURISTIC
# =============================================================================
println("\n" * "="^40)
println("PHASE 5: L-AUGMENTATION HEURISTIC")
println("="^40)

# Run L-Augmentation Heuristic
candidates, znew, cnt = Augmentation_Heuristic(nloc, ncust, capacity, cost, online_samples, xsolutions, max_iter, K, fixedcost, unmet_pen, scaling_factor)
println("Heuristic completed in $cnt iterations")

# Evaluate heuristic performance
heuristiccost, p1, m1 = Evaluation(nloc, ncust, cost, capacity, test_samples_data, candidates, znew, fixedcost, unmet_pen, scaling_factor)
println("Average heuristic cost: $heuristiccost")

# Calculate heuristic gap
heuristic_gap = (heuristiccost - offline_cost) / offline_cost * 100
println("Heuristic performance gap: $(round(heuristic_gap, digits=2))%")

# =============================================================================
# FINAL RESULTS
# =============================================================================
println("\n" * "="^60)
println("FINAL RESULTS SUMMARY")
println("="^60)
println("Offline optimal cost:     $(round(offline_cost, digits=2))")
println("Model cost:               $(round(modelcost, digits=2))")
println("Heuristic cost:           $(round(heuristiccost, digits=2))")
println("Model gap:                $(round(gap, digits=2))%")
println("Heuristic gap:            $(round(heuristic_gap, digits=2))%")
println("Improvement:              $(round(gap - heuristic_gap, digits=2))%")
println("="^60)

println("\n✅ All computations completed successfully!")
