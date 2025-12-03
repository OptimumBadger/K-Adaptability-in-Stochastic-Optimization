# Evaluation_Phase.jl
# Phase 3: Evaluation Phase
# This file handles:
#   - Loading samples from SAA_samples.txt
#   - Loading selected K policies from file
#   - Loading wait-and-see costs from file (computed by WS_SAA_Samples.jl)
#   - Computing model cost (select best policy from K for each scenario)
#   - Computing performance metrics

using PowerModels
using JuMP, Gurobi
using MathOptInterface
const MOI = MathOptInterface
using DataFrames, XLSX, Random, Distributions
using Printf
using Statistics
using Dates

# Include utility functions from Candidate_List_Generation_Phase.jl
include("Candidate_List_Generation_Phase.jl")

# Include functions from Assignment_Phase.jl
include("Assignment_Phase.jl")

# Include wait-and-see cost computation
include("WS_SAA_Samples.jl")

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

function load_selected_policies_from_file(file_path::String)
    """Load selected binary vectors from a file (one vector per line, space-separated)
    
    Same format as load_binary_vectors_from_file
    """
    return load_binary_vectors_from_file(file_path)
end

# =============================================================================
# SECOND STAGE COST FUNCTION
# =============================================================================

function Second_Stage_Cost_direct(data, sample, x_chosen, z, log_file=nothing)
    """Calculate second stage cost for evaluation - Direct implementation
    
    Args:
        data: Network data
        sample: Demand scenario
        x_chosen: List of K binary vectors (selected policies)
        z: Binary vector indicating available policies (should be all 1s for K policies)
        log_file: Optional log file path
    
    Returns:
        objective_value: Objective value
        work_units: Work units used
        penalty_cost: Penalty cost
    """
    # Ensure x_chosen contains exact 0/1 values
    x_chosen = [round.(Int, vec) for vec in x_chosen]
    
    # Initialize the model
    model = Model(Gurobi.Optimizer)
    MOI.set(model, MOI.TimeLimitSec(), 3600.0)
    
    # Set logging
    if log_file !== nothing
        if isfile(log_file)
            rm(log_file)
        end
        set_optimizer_attribute(model, "LogFile", log_file)
        set_optimizer_attribute(model, "LogToConsole", 0)
    else
        set_optimizer_attribute(model, "OutputFlag", 0)
    end
    
    # Sets
    B = data["bus"]
    L = data["branch"]
    G = data["gen"]
    
    # Decision Variables
    @variable(model, p_g[1:length(B)])           # Power generation at each bus
    @variable(model, f[1:length(L)])             # Power flow on each line
    @variable(model, theta[1:length(B)])         # Voltage angle at each bus
    @variable(model, x[1:length(L)], Bin)        # Line status variables (BINARY!)
    @variable(model, unmet[1:length(B)] >= 0)    # Unmet demand
    @variable(model, w[1:length(z)], Bin)        # Weight variables for policy selection (BINARY!)
    
    # Parameters
    B_ij, f_ij, p_max, p_min, A, c_i = parameters(data; line_capacity_factor=1.0)
    Pd = sample .* 0.9
    Penalty = 10000
    
    # Policy selection constraints
    @constraint(model, sum(w[i] for i in 1:length(z)) == 1)  # Exactly one policy selected
    for i in 1:length(z)
        @constraint(model, w[i] <= z[i])  # Can only select from available policies
    end
    
    # Line status based on selected policies
    for (branch_id, branch) in L
        branch_index = parse(Int, branch_id)
        # x_chosen[i][branch_index] where i is policy index, branch_index is branch index
        @constraint(model, x[branch_index] == sum(w[i] * x_chosen[i][branch_index] for i in 1:length(z)))
    end
    
    # Objective Function - Direct implementation
    @objective(model, Min, sum(c_i[i] * p_g[i] for i in 1:length(B)) + 
                          Penalty * sum(unmet[i] for i in 1:length(B)))
    
    # Power Balance Constraint (18b) - Direct implementation
    for i in 1:length(B)
        outgoing_flows = 0.0
        incoming_flows = 0.0
        for (branch_id, branch) in L
            branch_index = parse(Int, branch_id)
            if branch["f_bus"] == i
                outgoing_flows += f[branch_index]
            end
            if branch["t_bus"] == i
                incoming_flows += f[branch_index]
            end
        end
        @constraint(model, p_g[i] - Pd[i] == outgoing_flows - incoming_flows - unmet[i])
    end
    
    # Generator Limits (18c) - Direct implementation
    for i in 1:length(B)
        @constraint(model, p_g[i] >= p_min[i])
        @constraint(model, p_g[i] <= p_max[i])
    end
    
    # Line Flow Limits (18d) - With weighted x values
    for (branch_id, branch) in L
        branch_index = parse(Int, branch_id)
        # Use weighted line status: x[branch_index] is the weighted combination
        @constraint(model, -f_ij[branch_index] * x[branch_index] <= f[branch_index])
        @constraint(model, f[branch_index] <= f_ij[branch_index] * x[branch_index])
    end
    
    # DC Power Flow (18e) - With weighted x values
    for (branch_id, branch) in L
        branch_index = parse(Int, branch_id)
        f_bus = branch["f_bus"]
        t_bus = branch["t_bus"]
        M_ij = -12.6 * B_ij[branch_index]  # Big-M value
        
        # Use Big-M formulation for weighted line status
        @constraint(model, B_ij[branch_index] * (theta[f_bus] - theta[t_bus]) - 
                            M_ij * (1 - x[branch_index]) <= f[branch_index])
        @constraint(model, f[branch_index] <= B_ij[branch_index] * (theta[f_bus] - theta[t_bus]) + 
                                            M_ij * (1 - x[branch_index]))
    end
    
    # Reference angle
    @constraint(model, theta[1] == 0)
    
    # Solve
    optimize!(model)
    
    if termination_status(model) == MOI.OPTIMAL
        penalty_cost = Penalty * sum(value.(unmet))
        work_units = solve_time(model)
        return objective_value(model), work_units, penalty_cost
    else
        status = termination_status(model)
        return -1, solve_time(model), -1
    end
end

# =============================================================================
# MAIN FUNCTION: EVALUATION PHASE
# =============================================================================

function evaluation_phase(instance_file::String, S::Int, K::Int, 
                         selected_policies_file::String, samples_file::String,
                         wait_and_see_costs_file::String;
                         log_dir::String="logs")
    """
    Evaluation Phase: Compute model cost and compare with wait-and-see cost
    
    Args:
        instance_file: Name of the instance file (e.g., "case118Blumsack.m")
        S: Number of samples (should match samples in file)
        K: Number of selected policies
        selected_policies_file: Path to file containing K selected binary vectors
        samples_file: Path to file containing S samples 
        wait_and_see_costs_file: Path to file containing wait-and-see costs (from WS_SAA_Samples.jl)
        log_dir: Directory for log files
    
    Returns:
        results: Dictionary containing:
            - wait_and_see_cost: Average cost from wait-and-see file
            - model_cost: Average cost from using selected policies
            - performance_ratio: model_cost / wait_and_see_cost
            - wait_and_see_costs: List of costs for each scenario
            - model_costs: List of costs for each scenario
    """
    
    println("="^80)
    println("EVALUATION PHASE")
    println("="^80)
    println("Instance: $instance_file")
    println("Number of samples: $S")
    println("Number of selected policies: $K")
    println("="^80)
    
    # Get the directory where this script is located
    script_dir = @__DIR__
    project_root = dirname(dirname(script_dir))  # Go up two levels to project root
    
    # Construct file paths
    instance_path = joinpath(project_root, "Transmission_Switching", "instances", instance_file)
    
    # Create log directory if it doesn't exist
    log_path = joinpath(script_dir, log_dir)
    if !isdir(log_path)
        mkpath(log_path)
        println("Created log directory: $log_path")
    end
    
    # Check if files exist
    if !isfile(instance_path)
        error("❌ Error: Instance file not found! Expected: $instance_path")
    end
    
    if !isfile(selected_policies_file)
        error("❌ Error: Selected policies file not found! Expected: $selected_policies_file")
    end
    
    if !isfile(samples_file)
        error("❌ Error: Samples file not found! Expected: $samples_file")
    end
    
    if !isfile(wait_and_see_costs_file)
        error("❌ Error: Wait-and-see costs file not found! Expected: $wait_and_see_costs_file")
    end
    
    println("\n" * "="^40)
    println("STEP 1: LOADING DATA")
    println("="^40)
    
    # Extract case name from instance file
    case_name = extract_case_name(instance_file)
    println("Extracted case name: $case_name")
    
    # Load network data
    println("Loading network from: $instance_path")
    network = PowerModels.parse_file(instance_path)
    println("✅ Network loaded: $(length(network["bus"])) buses, $(length(network["branch"])) branches, $(length(network["gen"])) generators")
    
    # Load selected policies
    println("\nLoading selected policies from: $selected_policies_file")
    selected_policies = load_selected_policies_from_file(selected_policies_file)
    println("✅ Loaded $(length(selected_policies)) selected policies")
    
    if length(selected_policies) != K
        println("⚠️  Warning: Expected $K policies, but loaded $(length(selected_policies))")
    end
    
    # Load samples
    println("\nLoading samples from: $samples_file")
    samples = load_samples_from_file(samples_file)
    println("✅ Loaded $(length(samples)) samples")
    
    if length(samples) != S
        println("⚠️  Warning: Expected $S samples, but loaded $(length(samples))")
        S = length(samples)  # Update S to match loaded samples
    end
    
    println("\n" * "="^40)
    println("STEP 2: LOADING WAIT-AND-SEE COSTS")
    println("="^40)
    
    # Load wait-and-see costs from file (computed by WS_SAA_Samples.jl)
    wait_and_see_cost, wait_and_see_costs, wait_and_see_successful, wait_and_see_failed, wait_and_see_work_units = load_wait_and_see_costs_from_file(
        wait_and_see_costs_file
    )
    
    println("✅ Wait-and-see costs loaded from file")
    println("   File: $wait_and_see_costs_file")
    println("   Successful solves: $wait_and_see_successful")
    println("   Failed solves: $wait_and_see_failed")
    println("   Average wait-and-see cost: $(round(wait_and_see_cost, digits=2))")
    println("   Total work units: $(round(wait_and_see_work_units, digits=2))")
    
    # Filter out samples with Inf wait-and-see costs
    valid_indices = findall(c -> isfinite(c), wait_and_see_costs)
    valid_samples = [samples[i] for i in valid_indices]
    valid_wait_and_see_costs = [wait_and_see_costs[i] for i in valid_indices]
    num_valid_samples = length(valid_samples)
    
    println("\n📊 Filtering samples:")
    println("   Total samples: $(length(samples))")
    println("   Valid samples (finite wait-and-see cost): $num_valid_samples")
    println("   Invalid samples (Inf wait-and-see cost): $(length(samples) - num_valid_samples)")
    
    if num_valid_samples == 0
        error("❌ Error: No valid samples with finite wait-and-see costs!")
    end
    
    println("\n" * "="^40)
    println("STEP 3: COMPUTING MODEL COST")
    println("="^40)
    println("Selecting best policy from K candidates for each valid scenario...")
    
    # Generate timestamp for this run
    timestamp = Dates.format(now(), "YYYYmmdd_HHMMSS")
    
    # Compute model cost: use Second_Stage_Cost_direct for each valid scenario
    model_costs = Float64[]
    model_work_units = 0.0
    model_successful = 0
    model_failed = 0
    
    # z should be all 1s (all K policies are available)
    z = ones(Int, K)
    
    for (idx, sample) in enumerate(valid_samples)
        original_idx = valid_indices[idx]
        log_file = joinpath(log_path, "model_cost_scenario$(original_idx)_$(timestamp).log")
        
        obj_value, work_units, penalty_cost = Second_Stage_Cost_direct(
            network, sample, selected_policies, z, log_file
        )
        
        if obj_value != -1
            push!(model_costs, obj_value)
            model_work_units += work_units
            model_successful += 1
        else
            push!(model_costs, Inf)  # Mark as failed
            model_failed += 1
        end
        
        if idx % 10 == 0
            println("  Processed $idx valid scenarios")
        end
    end
    
    # Compute average model cost (only from successful solves, only for valid samples)
    successful_model = [c for c in model_costs if isfinite(c)]
    model_cost = length(successful_model) > 0 ? mean(successful_model) : Inf
    
    println("✅ Model cost computed")
    println("   Successful solves: $model_successful")
    println("   Failed solves: $model_failed")
    println("   Average model cost: $(round(model_cost, digits=2))")
    println("   Total work units: $(round(model_work_units, digits=2))")
    
    println("\n" * "="^40)
    println("STEP 4: COMPUTING PERFORMANCE METRICS")
    println("="^40)
    
    # Compute model gap
    if isfinite(wait_and_see_cost) && isfinite(model_cost) && wait_and_see_cost > 0
        model_gap = (model_cost - wait_and_see_cost) / wait_and_see_cost
        performance_ratio = model_cost / wait_and_see_cost  # Keep for results dict
        println("Model gap: $(round(model_gap, digits=4))")
    else
        model_gap = Inf
        performance_ratio = Inf
        println("⚠️  Warning: Could not compute model gap (wait-and-see cost: $wait_and_see_cost, model cost: $model_cost)")
    end
    
    println("\n" * "="^40)
    println("STEP 5: SUMMARY")
    println("="^40)
    println("Total samples: $S")
    println("Valid samples (evaluated): $num_valid_samples")
    println("Selected policies: $K")
    println("Wait-and-see cost (average over $num_valid_samples valid samples): $(round(wait_and_see_cost, digits=2))")
    println("Model cost (average over $num_valid_samples valid samples): $(round(model_cost, digits=2))")
    if isfinite(model_gap)
        println("Model gap: $(round(model_gap, digits=4))")
    end
    println("Log files stored in: $log_path")
    
    println("\n✅ Evaluation phase completed!")
    
    # Save results to file
    results_file = joinpath(script_dir, "results.txt")
    open(results_file, "w") do file
        println(file, "="^80)
        println(file, "EVALUATION PHASE RESULTS")
        println(file, "="^80)
        println(file, "")
        println(file, "Instance: $instance_file")
        println(file, "Total samples: $S")
        println(file, "Valid samples (evaluated): $num_valid_samples")
        println(file, "Selected policies (K): $K")
        println(file, "")
        println(file, "-"^80)
        println(file, "WAIT-AND-SEE COST")
        println(file, "-"^80)
        println(file, "Average wait-and-see cost: $(round(wait_and_see_cost, digits=2))")
        println(file, "Successful solves: $wait_and_see_successful")
        println(file, "Failed solves: $wait_and_see_failed")
        println(file, "Total work units: $(round(wait_and_see_work_units, digits=2))")
        println(file, "")
        println(file, "-"^80)
        println(file, "MODEL COST")
        println(file, "-"^80)
        println(file, "Average model cost: $(round(model_cost, digits=2))")
        println(file, "Successful solves: $model_successful")
        println(file, "Failed solves: $model_failed")
        println(file, "Total work units: $(round(model_work_units, digits=2))")
        println(file, "")
        println(file, "-"^80)
        println(file, "PERFORMANCE METRICS")
        println(file, "-"^80)
        if isfinite(wait_and_see_cost) && isfinite(model_cost) && wait_and_see_cost > 0
            model_gap = (model_cost - wait_and_see_cost) / wait_and_see_cost
            println(file, "Model gap: $(round(model_gap, digits=4))")
        else
            println(file, "Model gap: N/A (could not compute)")
        end
        println(file, "")
        println(file, "-"^80)
        println(file, "DETAILED COSTS")
        println(file, "-"^80)
        println(file, "Wait-and-see costs (one per line):")
        for (i, cost) in enumerate(wait_and_see_costs)
            if isfinite(cost)
                println(file, "  Scenario $i: $(round(cost, digits=2))")
            else
                println(file, "  Scenario $i: Inf (failed)")
            end
        end
        println(file, "")
        println(file, "Model costs (one per line, only for valid samples):")
        for (idx, cost) in enumerate(model_costs)
            original_scenario_idx = valid_indices[idx]
            if isfinite(cost)
                println(file, "  Valid sample $original_scenario_idx: $(round(cost, digits=2))")
            else
                println(file, "  Valid sample $original_scenario_idx: Inf (failed)")
            end
        end
        println(file, "")
        println(file, "="^80)
        println(file, "Results saved on: $(Dates.format(now(), "YYYY-mm-dd HH:MM:SS"))")
        println(file, "="^80)
    end
    
    println("📁 Results saved to: $results_file")
    
    # Return results
    results = Dict(
        "wait_and_see_cost" => wait_and_see_cost,
        "model_cost" => model_cost,
        "performance_ratio" => performance_ratio,
        "model_gap" => model_gap,
        "wait_and_see_costs" => wait_and_see_costs,
        "model_costs" => model_costs,
        "wait_and_see_work_units" => wait_and_see_work_units,
        "model_work_units" => model_work_units,
        "wait_and_see_successful" => wait_and_see_successful,
        "wait_and_see_failed" => wait_and_see_failed,
        "model_successful" => model_successful,
        "model_failed" => model_failed,
        "total_samples" => S,
        "valid_samples" => num_valid_samples
    )
    
    return results
end

# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 6
        println("❌ Error: Incorrect number of arguments!")
        println("Usage: julia Evaluation_Phase.jl <instance_file> <S> <K> <selected_policies_file> <samples_file> <wait_and_see_costs_file>")
        println("  instance_file: Instance file name (e.g., case118Blumsack.m)")
        println("  S: Number of samples (should match samples in file)")
        println("  K: Number of selected policies")
        println("  selected_policies_file: Path to file containing K selected binary vectors")
        println("  samples_file: Path to file containing S samples")
        println("  wait_and_see_costs_file: Path to file containing wait-and-see costs (from WS_SAA_Samples.jl)")
        println("\nExample:")
        println("  julia Evaluation_Phase.jl case118Blumsack.m 25 2 selected_policies.txt SAA_samples.txt wait_and_see_costs.txt")
        exit(1)
    end
    
    instance_file = ARGS[1]
    
    # Parse S (number of samples)
    S = try
        parsed = parse(Int, ARGS[2])
        if parsed <= 0
            throw(ArgumentError("S must be positive"))
        end
        parsed
    catch
        println("❌ Error: S must be a positive integer!")
        println("You provided: '$(ARGS[2])'")
        exit(1)
    end
    
    # Parse K (number of selected policies)
    K = try
        parsed = parse(Int, ARGS[3])
        if parsed <= 0
            throw(ArgumentError("K must be positive"))
        end
        parsed
    catch
        println("❌ Error: K must be a positive integer!")
        println("You provided: '$(ARGS[3])'")
        exit(1)
    end
    
    selected_policies_file = ARGS[4]
    samples_file = ARGS[5]
    wait_and_see_costs_file = ARGS[6]
    
    # Run evaluation phase
    results = evaluation_phase(
        instance_file, S, K, selected_policies_file, samples_file, wait_and_see_costs_file
    )
    
    println("\n" * "="^80)
    println("EVALUATION PHASE COMPLETED")
    println("="^80)
    println("Final Results:")
    println("  Wait-and-see cost: $(round(results["wait_and_see_cost"], digits=2))")
    println("  Model cost: $(round(results["model_cost"], digits=2))")
    if isfinite(results["model_gap"])
        println("  Model gap: $(round(results["model_gap"], digits=4))")
    end
end

