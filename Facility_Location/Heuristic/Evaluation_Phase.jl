# Evaluation_Phase.jl
# Phase 3: Evaluation Phase for Facility Location
# This file handles:
#   - Loading samples from SAA_samples.txt
#   - Loading selected K policies from file
#   - Loading wait-and-see costs from file (computed by Samples/FL/WS_Samples.jl)
#   - Computing model cost (select best policy from K for each scenario using Second_Stage_Cost)
#   - Computing performance metrics

using JuMP, Gurobi
using MathOptInterface
const MOI = MathOptInterface
using Random, Distributions
using Printf
using Statistics
using Dates

# Include utility functions from Candidate_List_Generation_Phase.jl
include("Candidate_List_Generation_Phase.jl")

function get_default_fl_parameters(instance_file::String)
    """Return default (unmet_pen, scaling_factor, capacity_factor) based on instance name.
    
    By default we use FL1-style parameters for ORLIB instances like cap91.txt and
    FL2-style parameters for Beasley instances like capa11.txt.
    """
    inst = lowercase(basename(instance_file))
    if inst == "capa11.txt"
        # FL2-style parameters
        return 100000.0, 1.0, 0.05
    else
        # FL1-style parameters (cap91, etc.)
        return 15.0, 0.2, 0.3
    end
end

# Include functions from Assignment_Phase.jl
include("Assignment_Phase.jl")

# Include wait-and-see cost computation (for load_wait_and_see_costs_from_file function)
# Note: The actual WS computation is now in Samples/FL/WS_Samples.jl
include("WS_SAA_Samples.jl")  # Keep for load_wait_and_see_costs_from_file function

# =============================================================================
# SECOND STAGE COST FUNCTION
# =============================================================================

function Second_Stage_Cost_direct(nloc, ncust, cost, capacity, demand, xsolutions, z, 
                                  fixedcost, unmet_pen, scaling_factor, log_file=nothing)
    """Calculate second stage cost for evaluation - Direct implementation from FL1.jl
    
    Args:
        nloc: Number of facilities
        ncust: Number of customers
        cost: Cost matrix (nloc x ncust)
        capacity: Capacity of each facility
        demand: Demand scenario
        xsolutions: Matrix of K solutions (K x nloc), each row is a binary vector
        z: Binary vector indicating available policies (should be all 1s for K policies)
        fixedcost: Fixed cost for opening each facility
        unmet_pen: Penalty for unmet demand
        scaling_factor: Scaling factor for transportation cost
        log_file: Optional log file path
    
    Returns:
        objective_value: Objective value
        penalty_cost: Penalty cost
        work_units: Work units used
    """
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
    
    # Decision Variables
    @variable(model, x[1:nloc], Bin)
    @variable(model, y[1:nloc, 1:ncust] >= 0)
    @variable(model, shortfall[1:ncust] >= 0)
    @variable(model, w[1:length(z)], Bin)

    # Constraints
    @constraint(model, [j in 1:ncust], sum(y[i, j] for i in 1:nloc) + shortfall[j] == demand[j])
    @constraint(model, [i in 1:nloc], sum(y[i, j] for j in 1:ncust) - capacity[i]*x[i] <= 0)
    @constraint(model, [i in 1:length(z)], w[i] <= z[i])
    @constraint(model, sum(w[i] for i in 1:length(z)) == 1)
    @constraint(model, x .== sum(w[i] * xsolutions[i, :] for i in 1:length(z)))

    # Objective function
    @objective(model, Min, 
        sum(fixedcost[i]*x[i] for i in 1:nloc) + 
        scaling_factor * sum(cost[i, j]*y[i, j] for i in 1:nloc for j in 1:ncust) + 
        sum(unmet_pen*shortfall[j] for j in 1:ncust))

    optimize!(model)
    
    # Get work units
    work_units = solve_time(model)
    
    if termination_status(model) == MOI.OPTIMAL
        penalty_cost = sum(unmet_pen * value(shortfall[j]) for j in 1:ncust)
        return objective_value(model), penalty_cost, work_units
    else
        return -1, -1, work_units
    end
end

# =============================================================================
# MAIN FUNCTION: EVALUATION PHASE
# =============================================================================

function evaluation_phase(instance_file::String, S::Int, K::Int, 
                         selected_policies_file::String, samples_file::String,
                         wait_and_see_costs_file::String;
                         experiment_type::String="low",
                         unmet_pen::Float64=15.0,
                         scaling_factor::Float64=0.2,
                         capacity_factor::Float64=0.3,
                         log_dir::String="logs")
    """
    Evaluation Phase: Compute model cost and compare with wait-and-see cost
    
    Args:
        instance_file: Name of the instance file (e.g., "cap91.txt")
        S: Number of samples (should match samples in file)
        K: Number of selected policies
        selected_policies_file: Path to file containing K selected binary vectors
        samples_file: Path to file containing S samples
        wait_and_see_costs_file: Path to file containing wait-and-see costs (from WS_SAA_Samples.jl)
        experiment_type: "low", "high", or "bernoulli"
        unmet_pen: Penalty for unmet demand
        scaling_factor: Scaling factor for transportation cost
        capacity_factor: Factor to scale facility capacities
        log_dir: Directory for log files
    
    Returns:
        results: Dictionary containing:
            - wait_and_see_cost: Average cost from wait-and-see file
            - model_cost: Average cost from using selected policies
            - model_gap: (model_cost - wait_and_see_cost) / wait_and_see_cost
            - wait_and_see_costs: List of costs for each scenario
            - model_costs: List of costs for each scenario
    """
    
    println("="^80)
    println("EVALUATION PHASE - FACILITY LOCATION")
    println("="^80)
    println("Instance: $instance_file")
    println("Number of samples: $S")
    println("Number of selected policies: $K")
    println("="^80)
    
    # Get the directory where this script is located
    script_dir = @__DIR__
    project_root = dirname(dirname(script_dir))  # Go up two levels to project root
    
    # Construct file paths
    instance_path = joinpath(project_root, "Facility_Location", "instances", instance_file)
    
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
    
    # Load instance data
    println("Loading instance from: $instance_path")
    capacity, fixedcost, cost, demandmean = read_orlib_cap(instance_path)
    
    # Apply capacity factor
    capacity = capacity .* capacity_factor
    println("✅ Instance loaded: $(length(capacity)) facilities, $(length(demandmean)) customers")
    
    nloc = length(capacity)
    ncust = length(demandmean)
    
    # Load selected policies
    println("\nLoading selected policies from: $selected_policies_file")
    selected_policies = load_binary_vectors_from_file(selected_policies_file)
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
    
    # Convert selected policies to matrix format (K x nloc)
    xsolutions = zeros(Int, K, nloc)
    for (i, vec) in enumerate(selected_policies)
        if length(vec) != nloc
            error("❌ Error: Selected policy $i has length $(length(vec)), expected $nloc")
        end
        xsolutions[i, :] = vec
    end
    
    # z should be all 1s (all K policies are available)
    z = ones(Int, K)
    
    # Compute model cost: use Second_Stage_Cost_direct for each valid scenario
    model_costs = Float64[]
    model_work_units = 0.0
    model_successful = 0
    model_failed = 0
    
    for (idx, sample) in enumerate(valid_samples)
        original_idx = valid_indices[idx]
        log_file = joinpath(log_path, "model_cost_scenario$(original_idx)_$(timestamp).log")
        
        obj_value, penalty_cost, work_units = Second_Stage_Cost_direct(
            nloc, ncust, cost, capacity, sample, xsolutions, z,
            fixedcost, unmet_pen, scaling_factor, log_file
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
        println("Model gap: $(round(model_gap, digits=4))")
    else
        model_gap = Inf
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
        println(file, "EVALUATION PHASE RESULTS - FACILITY LOCATION")
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
        println("Usage: julia Evaluation_Phase.jl <instance_file> <S> <K> <selected_policies_file> <samples_file> <wait_and_see_costs_file> [experiment_type] [unmet_pen] [scaling_factor] [capacity_factor]")
        println("  instance_file: Instance file name (e.g., cap91.txt)")
        println("  S: Number of samples (should match samples in file)")
        println("  K: Number of selected policies")
        println("  selected_policies_file: Path to file containing K selected binary vectors")
        println("  samples_file: Path to file containing S samples")
        println("  wait_and_see_costs_file: Path to file containing wait-and-see costs (from Samples/FL/WS_Samples.jl)")
        println("  experiment_type: 'low', 'high', or 'bernoulli' (default: 'low')")
        println("  unmet_pen: Penalty for unmet demand (default: 15.0)")
        println("  scaling_factor: Scaling factor for transportation cost (default: 0.2)")
        println("  capacity_factor: Factor to scale facility capacities (default: 0.3)")
        println("\nExample:")
        println("  julia Evaluation_Phase.jl cap91.txt 25 2 Samples/FL/selected_candidates_S25_K2.txt Samples/FL/Scenarios/S25_bernoulli.txt Samples/FL/Scenarios/wait_and_see_costs_S25_bernoulli.txt")
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
    experiment_type = length(ARGS) > 6 ? ARGS[7] : "low"
    # Choose defaults based on instance file (FL1 vs FL2 style)
    default_unmet, default_scaling, default_capacity = get_default_fl_parameters(instance_file)
    unmet_pen = length(ARGS) > 7 ? parse(Float64, ARGS[8]) : default_unmet
    scaling_factor = length(ARGS) > 8 ? parse(Float64, ARGS[9]) : default_scaling
    capacity_factor = length(ARGS) > 9 ? parse(Float64, ARGS[10]) : default_capacity
    
    # Run evaluation phase
    results = evaluation_phase(
        instance_file, S, K, selected_policies_file, samples_file, wait_and_see_costs_file;
        experiment_type=experiment_type,
        unmet_pen=unmet_pen,
        scaling_factor=scaling_factor,
        capacity_factor=capacity_factor
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

