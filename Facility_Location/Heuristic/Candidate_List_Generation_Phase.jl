# Candidate_List_Generation_Phase.jl
# Phase 1: Scenario Generation and Candidate List Generation for Facility Location
# This file handles:
#   - Loading facility location instance data (ORLIB format)
#   - Generating demand scenarios (with reproducibility)
#   - Solving each scenario to get binary vectors (facility opening decisions)
#   - Storing log files for each scenario

using JuMP, Gurobi
using MathOptInterface
const MOI = MathOptInterface
using Random, Distributions
using Printf
using Statistics
using Dates

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

function read_orlib_cap(filepath::String)
    """Read facility location data from ORLIB format files"""
    open(filepath) do io
        # 1) read n (#facilities) and m (#cities)
        n, m = parse.(Int, split(readline(io)))

        # 2) read facility data: capacity and opening cost
        capacity  = Vector{Float64}(undef, n)
        fixedcost = Vector{Float64}(undef, n)
        for i in 1:n
            # each line: "<capacity> <opening cost>"
            c1, c2 = parse.(Float64, split(readline(io)))
            capacity[i], fixedcost[i] = c1, c2
        end

        # 3) read city‐by‐city: demand + cost to each facility
        demand = Vector{Float64}(undef, m)
        cost = Array{Float64}(undef, n, m)
        for j in 1:m
            # read demand line
            demand[j] = parse(Float64, strip(readline(io)))
            # now read cost entries until we have n of them
            vals = Float64[]
            while length(vals) < n
                append!(vals, parse.(Float64, split(readline(io))))
            end
            cost[:,j] = vals[1:n]
        end

        # Transform cost matrix: divide each column by customer demand
        # This converts from total cost to cost per unit
        for j in 1:m
            if demand[j] > 0  # Avoid division by zero
                cost[:, j] ./= demand[j]
            end
        end

        return capacity, fixedcost, cost, demand
    end
end

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

function generate_demand_scenario(ncust, demandmean, experiment_type="low")
    """Generate a single demand scenario using PowerFlow-style formula
    
    Args:
        ncust: Number of customers
        demandmean: Mean demand for each customer
        experiment_type: "low", "high", or "bernoulli"
    
    Returns:
        scenario_demand: Vector of demand values for each customer
    """
    sigma_range = Dict(
        "low" => (0.1, 0.2),
        "high" => (0.2, 0.3),
        "bernoulli" => (0.1, 0.2)
    )

    if !haskey(sigma_range, lowercase(experiment_type))
        error("Invalid experiment_type: $experiment_type. Must be 'low', 'high', or 'bernoulli'.")
    end

    min_sigma, max_sigma = sigma_range[lowercase(experiment_type)]
    sigma = rand(Uniform(min_sigma, max_sigma))

    scenario_demand = Vector{Float64}(undef, ncust)
    global_noise = randn() # Global random variable

    for j in 1:ncust
        individual_noise = randn() # Individual random variable
        
        # PowerFlow-style demand formula
        demand_val = demandmean[j] + sigma * individual_noise * demandmean[j] + global_noise * sigma * demandmean[j]
        
        # Apply Bernoulli multiplier if experiment_type is "bernoulli"
        if lowercase(experiment_type) == "bernoulli"
            if rand(Bernoulli(0.5)) == 0
                demand_val = 0.0 # Set to zero if Bernoulli is 0
            end
        end
        
        scenario_demand[j] = max(0.0, demand_val) # Ensure non-negativity
    end
    return scenario_demand
end

function generate_facility_location_scenarios(nscen::Int, demandmean::Vector{Float64}, 
                                             experiment_type::String="low")
    """Generate demand scenarios for facility location
    
    Args:
        nscen: Number of scenarios to generate
        demandmean: Mean demand for each customer
        experiment_type: "low", "high", or "bernoulli"
    
    Returns:
        scenarios_list: List of demand scenarios
    """
    ncust = length(demandmean)
    bernoulli_str = experiment_type == "bernoulli" ? " (Bernoulli mode)" : ""
    println("Generating $nscen Facility Location scenarios with experiment type '$experiment_type'$bernoulli_str...")
    
    scenarios_list = Vector{Vector{Float64}}()
    
    for scenario in 1:nscen
        scenario_demand = generate_demand_scenario(ncust, demandmean, experiment_type)
        push!(scenarios_list, scenario_demand)
        
        if scenario <= 3  # Debug first 3 scenarios
            println("   Scenario $scenario: total demand = $(round(sum(scenario_demand), digits=2))")
        end
    end
    
    println("✅ Generated $nscen scenarios successfully!")
    return scenarios_list
end

function offline_calc_direct(nloc, ncust, capacity, cost, demand, fixedcost, 
                            unmet_pen, scaling_factor, log_file=nothing)
    """Solve facility location problem from scratch for a given demand scenario
    
    Args:
        nloc: Number of facilities
        ncust: Number of customers
        capacity: Capacity of each facility
        cost: Cost matrix (nloc x ncust)
        demand: Demand vector for customers
        fixedcost: Fixed cost for opening each facility
        unmet_pen: Penalty for unmet demand
        scaling_factor: Scaling factor for transportation cost
        log_file: Optional log file path
    
    Returns:
        objective_value: Objective value
        binary_vector: Binary vector (facility opening decisions)
        penalty_flag: 1 if penalty > 0, 0 otherwise
        penalty_cost: Total penalty cost
        work_units: Work units used
    """
    model = Model(Gurobi.Optimizer)
    MOI.set(model, MOI.TimeLimitSec(), 120.0)  # 120 seconds
    
    # Set logging
    if log_file !== nothing
        # Delete existing log file if it exists to ensure fresh start
        if isfile(log_file)
            rm(log_file)
        end
        set_optimizer_attribute(model, "LogFile", log_file)
        set_optimizer_attribute(model, "LogToConsole", 1)
    end
    # Always enable console output to see gap and solver progress
    set_optimizer_attribute(model, "OutputFlag", 1)
    
    # Decision Variables
    @variable(model, x[1:nloc], Bin)  # Facility opening decisions
    @variable(model, y[1:nloc, 1:ncust] >= 0)  # Allocation variables
    @variable(model, shortfall[1:ncust] >= 0)  # Unmet demand

    # Constraints
    @constraint(model, [j in 1:ncust], sum(y[i,j] for i in 1:nloc) + shortfall[j] == demand[j])
    @constraint(model, [i in 1:nloc], sum(y[i,j] for j in 1:ncust) - capacity[i]*x[i] <= 0)

    # Objective function
    @objective(model, Min, 
        sum(fixedcost[i]*x[i] for i in 1:nloc) + 
        scaling_factor * sum(cost[i,j]*y[i,j] for i in 1:nloc for j in 1:ncust) + 
        sum(unmet_pen*shortfall[j] for j in 1:ncust))

    optimize!(model)
    
    if termination_status(model) == MOI.OPTIMAL
        # Get binary solution
        x_solution = round.(Int, collect(value.(x)))
        shortfall_solution = value.(shortfall)
        
        work_units = solve_time(model)
        penalty_cost = sum(unmet_pen * shortfall_solution[j] for j in 1:ncust)
        penalty_flag = penalty_cost > 1e-6 ? 1 : 0
        
        return objective_value(model), collect(x_solution), penalty_flag, penalty_cost, work_units
    else
        status = termination_status(model)
        println("❌ Scenario solve failed with status: $status")
        return -1, [], -1, -1, -1
    end
end

# =============================================================================
# UTILITY FUNCTIONS FOR SAVING BINARY VECTORS
# =============================================================================

function save_binary_vectors_to_file(binary_vectors::Vector{Vector{Int}}, file_path::String)
    """Save binary vectors to a file (one vector per line, space-separated)
    
    Format matches what load_binary_vectors_from_file expects in Assignment_Phase.jl
    """
    if length(binary_vectors) == 0
        println("⚠️  Warning: No binary vectors to save!")
        return
    end
    
    # Create directory if it doesn't exist
    dir = dirname(file_path)
    if !isempty(dir) && !isdir(dir)
        mkpath(dir)
    end
    
    open(file_path, "w") do file
        for vec in binary_vectors
            # Write vector as space-separated integers
            println(file, join(vec, " "))
        end
    end
    
    println("✅ Binary vectors saved to: $file_path")
    println("   Total vectors: $(length(binary_vectors))")
    if length(binary_vectors) > 0
        println("   Vector length: $(length(binary_vectors[1]))")
    end
end

# =============================================================================
# MAIN FUNCTION: CANDIDATE LIST GENERATION
# =============================================================================

function generate_candidate_list(instance_file::String, S::Int; 
                                 experiment_type::String="low",
                                 unmet_pen::Float64=15.0,
                                 scaling_factor::Float64=0.2,
                                 capacity_factor::Float64=0.3,
                                 log_dir::String="logs",
                                 output_file::Union{String, Nothing}=nothing)
    """
    Generate candidate list by:
    1. Loading facility location instance data
    2. Generating S scenarios (with reproducibility via random seed)
    3. Solving each scenario to get binary vectors (facility opening decisions)
    4. Storing log files for each scenario
    
    Args:
        instance_file: Name of instance file (e.g., "cap91.txt")
        S: Number of scenarios to generate
        experiment_type: "low", "high", or "bernoulli"
        unmet_pen: Penalty for unmet demand
        scaling_factor: Scaling factor for transportation cost
        capacity_factor: Factor to scale facility capacities
        log_dir: Directory for log files
        output_file: Optional output file path for binary vectors
    
    Returns:
        binary_vectors: List of binary vectors (facility opening decisions)
        objective_values: List of objective values
        penalty_flags: List of penalty flags
        penalty_costs: List of penalty costs
        work_units_list: List of work units
    """
    
    println("="^80)
    println("CANDIDATE LIST GENERATION PHASE - FACILITY LOCATION")
    println("="^80)
    println("Instance: $instance_file")
    println("Number of scenarios: $S")
    println("Experiment type: $experiment_type")
    println("="^80)
    
    # Set random seed for reproducibility (always 1234)
    Random.seed!(1234)
    println("✅ Random seed set to 1234 (ensuring reproducibility)")
    
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
    
    # Check if file exists
    if !isfile(instance_path)
        error("❌ Error: Instance file not found! Expected: $instance_path")
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
    println("   Applied capacity factor: $capacity_factor")
    println("   Capacity range: min=$(round(minimum(capacity), digits=0)), max=$(round(maximum(capacity), digits=0))")
    
    nloc = length(capacity)
    ncust = length(demandmean)
    
    println("\n" * "="^40)
    println("STEP 2: GENERATING SCENARIOS")
    println("="^40)
    
    # Generate scenarios (with seed already set above for reproducibility)
    scenarios = generate_facility_location_scenarios(S, demandmean, experiment_type)
    
    println("\n" * "="^40)
    println("STEP 3: SOLVING SCENARIOS AND STORING BINARY VECTORS")
    println("="^40)
    
    # Storage for binary vectors and results
    binary_vectors = Vector{Vector{Int}}()
    objective_values = Float64[]
    penalty_flags = Int[]
    penalty_costs = Float64[]
    work_units_list = Float64[]
    
    # Solve each scenario
    successful_solves = 0
    failed_solves = 0
    
    # Generate timestamp for this run (format: YYYYMMDD_HHMMSS)
    # This ensures each run creates unique log files instead of appending to existing ones
    timestamp = Dates.format(now(), "YYYYmmdd_HHMMSS")
    
    for (scenario_idx, scenario_demand) in enumerate(scenarios)
        println("Solving scenario $scenario_idx/$S...")
        
        # Create log file for this scenario with timestamp
        log_file = joinpath(log_path, "scenario$(scenario_idx)_$(timestamp).log")
        
        obj_value, binary_vector, penalty_flag, penalty_cost, work_units = offline_calc_direct(
            nloc, ncust, capacity, cost, scenario_demand, fixedcost, 
            unmet_pen, scaling_factor, log_file
        )
        
        if obj_value != -1
            push!(binary_vectors, binary_vector)
            push!(objective_values, obj_value)
            push!(penalty_flags, penalty_flag)
            push!(penalty_costs, penalty_cost)
            push!(work_units_list, work_units)
            successful_solves += 1
            println("  ✅ Solved successfully: Objective = $(round(obj_value, digits=2)), Work Units = $(round(work_units, digits=2))")
            println("  📝 Log saved to: $log_file")
        else
            failed_solves += 1
            println("  ❌ Failed to solve scenario $scenario_idx")
            println("  📝 Log saved to: $log_file")
        end
    end
    
    println("\n" * "="^40)
    println("STEP 4: SUMMARY")
    println("="^40)
    println("Total scenarios: $S")
    println("Successfully solved: $successful_solves")
    println("Failed: $failed_solves")
    println("Binary vectors stored: $(length(binary_vectors))")
    println("Log files stored in: $log_path")
    
    if successful_solves > 0
        println("\nStatistics:")
        println("  Average objective value: $(round(mean(objective_values), digits=2))")
        println("  Average work units: $(round(mean(work_units_list), digits=2))")
        println("  Scenarios with penalties: $(sum(penalty_flags))")
        println("  Total penalty cost: $(round(sum(penalty_costs), digits=2))")
        
        # Check uniqueness of binary vectors
        unique_count = 0
        unique_vectors = Vector{Vector{Int}}()
        tolerance = 1e-6
        
        for (i, vec) in enumerate(binary_vectors)
            is_unique = true
            for j in 1:(i-1)
                if all(abs(binary_vectors[i][k] - binary_vectors[j][k]) < tolerance for k in 1:length(binary_vectors[i]))
                    is_unique = false
                    break
                end
            end
            if is_unique
                unique_count += 1
                push!(unique_vectors, vec)
            end
        end
        
        println("  Unique binary vectors: $unique_count out of $(length(binary_vectors))")
    end
    
    println("\n✅ Candidate list generation completed!")
    println("Binary vectors are stored in the 'binary_vectors' list")
    
    # Save binary vectors to file
    if length(binary_vectors) > 0
        if output_file === nothing
            # Save to Samples/FL/Binary_vectors/ with naming: L<S>_<experiment_type>.txt
            project_root = dirname(dirname(script_dir))  # Go up two levels to project root
            binary_vectors_dir = joinpath(project_root, "Samples", "FL", "Binary_vectors")
            if !isdir(binary_vectors_dir)
                mkpath(binary_vectors_dir)
            end
            output_file = joinpath(binary_vectors_dir, "L$(S)_$(experiment_type).txt")
        end
        
        save_binary_vectors_to_file(binary_vectors, output_file)
        println("✅ Binary vectors saved to: $output_file")
        println("   Total vectors: $(length(binary_vectors))")
        println("   You can use this file in Assignment_Phase.jl")
    else
        println("⚠️  Warning: No binary vectors to save!")
    end
    
    return binary_vectors, objective_values, penalty_flags, penalty_costs, work_units_list
end

# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 2
        println("❌ Error: Incorrect number of arguments!")
        println("Usage: julia Candidate_List_Generation_Phase.jl <instance_file> <S> [experiment_type] [unmet_pen] [scaling_factor] [capacity_factor]")
        println("  instance_file: Instance file name (e.g., cap91.txt)")
        println("  S: Number of scenarios to generate")
        println("  experiment_type: 'low', 'high', or 'bernoulli' (default: 'low')")
        println("  unmet_pen: Penalty for unmet demand (default: 15.0)")
        println("  scaling_factor: Scaling factor for transportation cost (default: 0.2)")
        println("  capacity_factor: Factor to scale facility capacities (default: 0.3)")
        println("\nExample:")
        println("  julia Candidate_List_Generation_Phase.jl cap91.txt 25")
        println("  julia Candidate_List_Generation_Phase.jl cap91.txt 25 bernoulli")
        println("  julia Candidate_List_Generation_Phase.jl cap91.txt 25 low 15.0 0.2 0.3")
        exit(1)
    end
    
    instance_file = ARGS[1]
    
    # Parse S (number of scenarios)
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
    
    # Parse optional arguments
    experiment_type = length(ARGS) > 2 ? ARGS[3] : "low"
    if !(experiment_type in ["low", "high", "bernoulli"])
        println("❌ Error: experiment_type must be 'low', 'high', or 'bernoulli'!")
        println("You provided: '$experiment_type'")
        exit(1)
    end
    
    # Choose defaults based on instance file (FL1 vs FL2 style)
    default_unmet, default_scaling, default_capacity = get_default_fl_parameters(instance_file)
    unmet_pen = length(ARGS) > 3 ? parse(Float64, ARGS[4]) : default_unmet
    scaling_factor = length(ARGS) > 4 ? parse(Float64, ARGS[5]) : default_scaling
    capacity_factor = length(ARGS) > 5 ? parse(Float64, ARGS[6]) : default_capacity
    
    # Run main function (random seed is always 1234, hardcoded)
    binary_vectors, objective_values, penalty_flags, penalty_costs, work_units_list = generate_candidate_list(
        instance_file, S; 
        experiment_type=experiment_type,
        unmet_pen=unmet_pen,
        scaling_factor=scaling_factor,
        capacity_factor=capacity_factor
    )
    
    println("\n📊 Results stored in variables:")
    println("  - binary_vectors: $(length(binary_vectors)) binary vectors")
    println("  - objective_values: $(length(objective_values)) objective values")
    println("  - penalty_flags: $(length(penalty_flags)) penalty flags")
    println("  - penalty_costs: $(length(penalty_costs)) penalty costs")
    println("  - work_units_list: $(length(work_units_list)) work units")
end

