# Assignment_Phase.jl
# Phase 2: Assignment Phase for Facility Location
# This file handles:
#   - Loading binary vectors from Candidate_List_Generation_Phase.jl
#   - Loading samples from Samples_For_SAA.jl
#   - Computing extensive form matrix (optimized: build model once, update constraints)
#   - Solving reformulated extensive form to select K policies
#   - Saving selected policies to file 

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

# =============================================================================
# MODEL STRUCTURE FOR OPTIMIZED SOLVING
# =============================================================================

struct ModelStructure
    model::Model
    x::Vector{VariableRef}  # Facility opening variables
    y::Matrix{VariableRef}  # Allocation variables
    shortfall::Vector{VariableRef}  # Unmet demand variables
    demand_constraints::Vector{ConstraintRef}  # Demand balance constraints (RHS will be updated)
    capacity_constraints::Vector{ConstraintRef}  # Capacity constraints (fixed)
    x_fixing_constraints::Vector{ConstraintRef}  # x fixing constraints (will be deleted/recreated)
end

function build_model_structure(nloc, ncust, capacity, cost, fixedcost, unmet_pen, scaling_factor, log_file=nothing)
    """Build the model structure once (variables, objective, fixed constraints)
    
    Args:
        nloc: Number of facilities
        ncust: Number of customers
        capacity: Capacity of each facility
        cost: Cost matrix (nloc x ncust)
        fixedcost: Fixed cost for opening each facility
        unmet_pen: Penalty for unmet demand
        scaling_factor: Scaling factor for transportation cost
        log_file: Optional log file path
    
    Returns:
        model_struct: ModelStructure containing model and constraint references
    """
    model = Model(Gurobi.Optimizer)
    MOI.set(model, MOI.TimeLimitSec(), 120.0)
    
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
    
    # Objective function (fixed)
    @objective(model, Min, 
        sum(fixedcost[i]*x[i] for i in 1:nloc) + 
        scaling_factor * sum(cost[i,j]*y[i,j] for i in 1:nloc for j in 1:ncust) + 
        sum(unmet_pen*shortfall[j] for j in 1:ncust))
    
    # Capacity constraints (fixed, don't change)
    capacity_constraints = Vector{ConstraintRef}()
    for i in 1:nloc
        push!(capacity_constraints, @constraint(model, sum(y[i,j] for j in 1:ncust) - capacity[i]*x[i] <= 0))
    end
    
    # Demand constraints (RHS will be updated, initialize with dummy values)
    demand_constraints = Vector{ConstraintRef}()
    for j in 1:ncust
        push!(demand_constraints, @constraint(model, sum(y[i,j] for i in 1:nloc) + shortfall[j] == 0.0))
    end
    
    # x fixing constraints (will be deleted and recreated, initialize empty)
    x_fixing_constraints = Vector{ConstraintRef}()
    
    return ModelStructure(model, x, y, shortfall, demand_constraints, capacity_constraints, x_fixing_constraints)
end

function update_model_for_solve(model_struct::ModelStructure, demand::Vector{Float64}, solved_x::Vector{Int})
    """Update model for a specific scenario-solution pair
    
    Args:
        model_struct: ModelStructure containing model and constraints
        demand: Demand vector for this scenario (updates RHS of demand constraints)
        solved_x: Binary vector for this solution (creates x fixing constraints)
    """
    # 1. Update RHS of demand constraints
    for j in 1:length(demand)
        set_normalized_rhs(model_struct.demand_constraints[j], demand[j])
    end
    
    # 2. Delete existing x fixing constraints
    for constraint in model_struct.x_fixing_constraints
        delete(model_struct.model, constraint)
    end
    empty!(model_struct.x_fixing_constraints)
    
    # 3. Create new x fixing constraints for this solution
    for i in 1:length(solved_x)
        push!(model_struct.x_fixing_constraints, 
              @constraint(model_struct.model, model_struct.x[i] == solved_x[i]))
    end
end

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

function load_samples_from_file(file_path::String)
    """Load samples from a file (one sample per line, space-separated demand values)
    
    Each line should contain space-separated floating point numbers (demand values)
    """
    if !isfile(file_path)
        error("❌ Error: Samples file not found! Expected: $file_path")
    end
    
    println("Loading samples from: $file_path")
    samples = Vector{Vector{Float64}}()
    
    open(file_path, "r") do file
        for (line_num, line) in enumerate(eachline(file))
            line = strip(line)
            if length(line) > 0  # Skip empty lines
                try
                    # Parse space-separated floating point numbers
                    values = [parse(Float64, x) for x in split(line)]
                    push!(samples, values)
                catch e
                    error("❌ Error parsing line $line_num in samples file: $e")
                end
            end
        end
    end
    
    println("✅ Loaded $(length(samples)) samples")
    if length(samples) > 0
        println("   Sample length: $(length(samples[1]))")
    end
    
    return samples
end

function load_binary_vectors_from_file(file_path::String)
    """Load binary vectors from a file (one vector per line)
    
    Each line should contain space-separated integers (0s and 1s)
    """
    if !isfile(file_path)
        error("❌ Error: Binary vectors file not found! Expected: $file_path")
    end
    
    println("Loading binary vectors from: $file_path")
    binary_vectors = Vector{Vector{Int}}()
    
    open(file_path, "r") do file
        for (line_num, line) in enumerate(eachline(file))
            line = strip(line)
            if length(line) > 0  # Skip empty lines
                try
                    # Parse space-separated integers
                    values = [parse(Int, x) for x in split(line)]
                    push!(binary_vectors, values)
                catch e
                    error("❌ Error parsing line $line_num in binary vectors file: $e")
                end
            end
        end
    end
    
    println("✅ Loaded $(length(binary_vectors)) binary vectors")
    if length(binary_vectors) > 0
        println("   Vector length: $(length(binary_vectors[1]))")
    end
    
    return binary_vectors
end

function save_selected_binary_vectors(selected_vectors::Vector{Vector{Int}}, file_path::String)
    """Save selected binary vectors to a file (one vector per line, space-separated)
    
    Format matches what load_binary_vectors_from_file expects
    """
    if length(selected_vectors) == 0
        println("⚠️  Warning: No binary vectors to save!")
        return
    end
    
    # Create directory if it doesn't exist
    dir = dirname(file_path)
    if !isempty(dir) && !isdir(dir)
        mkpath(dir)
    end
    
    open(file_path, "w") do file
        for vec in selected_vectors
            # Write vector as space-separated integers
            println(file, join(vec, " "))
        end
    end
    
    println("✅ Selected binary vectors saved to: $file_path")
    println("   Total vectors: $(length(selected_vectors))")
    if length(selected_vectors) > 0
        println("   Vector length: $(length(selected_vectors[1]))")
    end
end

# =============================================================================
# EXTENSIVE FORM FUNCTIONS
# =============================================================================

function Calculations_For_Extensive_Form(nloc, ncust, capacity, cost, online_samples, xsolutions, 
                                         fixedcost, unmet_pen, scaling_factor, log_file=nothing)
    """Calculate objective values for all scenario-solution pairs (OPTIMIZED)
    
    Args:
        nloc: Number of facilities
        ncust: Number of customers
        capacity: Capacity of each facility
        cost: Cost matrix (nloc x ncust)
        online_samples: List of S online samples (each is a vector of demands)
        xsolutions: Matrix of l solutions (l x nloc), each row is a binary vector
        fixedcost: Fixed cost for opening each facility
        unmet_pen: Penalty for unmet demand
        scaling_factor: Scaling factor for transportation cost
        log_file: Optional log file path
    
    Returns:
        obj_value: S x l matrix of objective values
        ratio: S x l matrix of fixed cost ratios
        total_work_units: Total work units across all solves
    """
    total_scenarios = length(online_samples)
    total_solutions = size(xsolutions, 1)
    obj_value = zeros(total_scenarios, total_solutions)
    ratio = zeros(total_scenarios, total_solutions)
    total_work_units = 0.0
    
    println("Computing extensive form matrix (optimized: building model once, updating constraints per solve)...")
    
    # Build model structure ONCE
    model_struct = build_model_structure(nloc, ncust, capacity, cost, fixedcost, unmet_pen, scaling_factor, log_file)
    
    # For each scenario-solution pair, update constraints and solve
    for scenario in 1:total_scenarios
        if scenario % 20 == 0
            println("  Processed $scenario scenarios")
        end
        demand = online_samples[scenario]
        for solution in 1:total_solutions
            # Update model for this scenario-solution pair:
            # 1. Update RHS of demand constraints (demand-dependent)
            # 2. Delete and recreate x fixing constraints (solution-dependent)
            update_model_for_solve(model_struct, demand, xsolutions[solution, :])
            
            # Solve
            optimize!(model_struct.model)
            
            if termination_status(model_struct.model) == MOI.OPTIMAL
                obj_value[scenario, solution] = objective_value(model_struct.model)
                fixed_cost = sum(fixedcost[i] * value(model_struct.x[i]) for i in 1:nloc)
                ratio[scenario, solution] = fixed_cost / obj_value[scenario, solution]
                work_units = solve_time(model_struct.model)
                total_work_units += work_units
            else
                obj_value[scenario, solution] = -1
                ratio[scenario, solution] = -1
                total_work_units += solve_time(model_struct.model)
            end
        end
    end
    
    return obj_value, ratio, total_work_units
end

function Reformulated_Extensive_Form(S, l, K, obj_value, log_file=nothing)
    """Solve the reformulated extensive form to select K optimal solutions
    
    Args:
        S: Number of scenarios
        l: Number of candidate solutions
        K: Number of policies to select
        obj_value: S x l matrix of objective values
        log_file: Optional log file path
    
    Returns:
        objective_value: Optimal objective value
        z: Binary vector indicating selected policies (length l)
    """
    model = Model(Gurobi.Optimizer)
    MOI.set(model, MOI.TimeLimitSec(), 3600.0)
    
    # Set logging based on whether we want to save logs
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
    @variable(model, z[1:l], Bin)
    @variable(model, w[1:l, 1:S], Bin)

    # Constraints for infeasible solutions
    for scenario in 1:S
        for solution in 1:l
            if obj_value[scenario, solution] == -1
                @constraint(model, w[solution, scenario] == 0)
            end
        end
    end

    # Assignment constraints
    for j in 1:S
        @constraint(model, [i in 1:l], w[i, j] <= z[i])
        @constraint(model, sum(w[i, j] for i in 1:l) == 1)
    end
    @constraint(model, sum(z[i] for i in 1:l) == K)

    # Objective
    objective = sum(w[solution, scenario] * obj_value[scenario, solution] 
                   for scenario in 1:S for solution in 1:l)
    @objective(model, Min, objective / S)

    optimize!(model)

    if termination_status(model) == MOI.OPTIMAL
        return objective_value(model), value.(z)
    else
        return -1, []
    end
end

# =============================================================================
# MAIN FUNCTION: ASSIGNMENT PHASE
# =============================================================================

function assignment_phase(instance_file::String, S::Int, binary_vectors_input::Union{Vector{Vector{Int}}, String}, K::Int;
                          experiment_type::String="low",
                          unmet_pen::Float64=15.0,
                          scaling_factor::Float64=0.2,
                          capacity_factor::Float64=0.3,
                          log_dir::String="logs",
                          samples_file::Union{String, Nothing}=nothing)
    """
    Assignment Phase: Evaluate candidate solutions on online samples and select K policies
    
    Args:
        instance_file: Name of the instance file (e.g., "cap91.txt")
        S: Number of online samples (used if samples_file is not provided)
        binary_vectors_input: Either:
            - List of binary vectors from Phase 1 (Vector{Vector{Int}})
            - Path to file containing binary vectors (String), one vector per line
        K: Number of policies to select
        experiment_type: "low", "high", or "bernoulli"
        unmet_pen: Penalty for unmet demand
        scaling_factor: Scaling factor for transportation cost
        capacity_factor: Factor to scale facility capacities
        log_dir: Directory for log files
        samples_file: Optional path to file containing samples (if provided, loads from file instead of generating)
    
    Returns:
        results: Dictionary containing:
            - obj_value: S x l cost matrix
            - z: Selected policies (binary vector)
            - reformulated_obj: Objective value from reformulated extensive form
            - total_work_units: Total work units
    """
    
    # Load binary vectors if filename is provided, otherwise use directly
    binary_vectors = if isa(binary_vectors_input, String)
        load_binary_vectors_from_file(binary_vectors_input)
    else
        binary_vectors_input
    end
    
    if length(binary_vectors) == 0
        error("❌ Error: No binary vectors provided or loaded!")
    end
    
    println("="^80)
    println("ASSIGNMENT PHASE - FACILITY LOCATION")
    println("="^80)
    println("Instance: $instance_file")
    println("Number of online samples: $S")
    println("Number of candidate solutions: $(length(binary_vectors))")
    println("Number of policies to select: $K")
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
    
    # Check if instance file exists
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
    
    nloc = length(capacity)
    ncust = length(demandmean)
    
    # Load or generate samples
    if samples_file !== nothing
        println("\nLoading samples from file: $samples_file")
        online_samples = load_samples_from_file(samples_file)
        if length(online_samples) != S
            println("⚠️  Warning: Expected $S samples, but loaded $(length(online_samples))")
            S = length(online_samples)  # Update S to match loaded samples
        end
    else
        println("\nGenerating $S samples...")
        Random.seed!(1234)  # Same seed as Phase 1
        online_samples = generate_facility_location_scenarios(S, demandmean, experiment_type)
        println("✅ Generated $S samples")
    end
    
    println("\n" * "="^40)
    println("STEP 2: COMPUTING EXTENSIVE FORM")
    println("="^40)
    
    # Convert binary vectors to matrix format (l x nloc)
    l = length(binary_vectors)
    xsolutions = zeros(Int, l, nloc)
    for (i, vec) in enumerate(binary_vectors)
        if length(vec) != nloc
            error("❌ Error: Binary vector $i has length $(length(vec)), expected $nloc")
        end
        xsolutions[i, :] = vec
    end
    
    # Generate timestamp for log files
    timestamp = Dates.format(now(), "YYYYmmdd_HHMMSS")
    extensive_form_log = joinpath(log_path, "extensive_form_$(timestamp).log")
    
    # Compute extensive form matrix
    obj_value, ratio, total_work_units = Calculations_For_Extensive_Form(
        nloc, ncust, capacity, cost, online_samples, xsolutions,
        fixedcost, unmet_pen, scaling_factor, extensive_form_log
    )
    
    println("✅ Extensive form matrix computed")
    println("   Matrix size: $(size(obj_value))")
    println("   Total work units: $(round(total_work_units, digits=2))")
    
    println("\n" * "="^40)
    println("STEP 3: SOLVING REFORMULATED EXTENSIVE FORM")
    println("="^40)
    
    reformulated_log = joinpath(log_path, "reformulated_extensive_form_$(timestamp).log")
    reformulated_obj, z = Reformulated_Extensive_Form(S, l, K, obj_value, reformulated_log)
    
    if reformulated_obj == -1
        error("❌ Error: Reformulated extensive form did not solve to optimality!")
    end
    
    println("✅ Reformulated extensive form solved")
    println("   Objective value: $(round(reformulated_obj, digits=2))")
    println("   Selected policies: $(sum(z .> 0.5))")
    
    # Save selected K binary vectors to file
    selected_indices = findall(x -> x > 0.5, z)
    selected_vectors = [binary_vectors[idx] for idx in selected_indices]
    
    if length(selected_vectors) > 0
        # Generate filename for selected vectors: selected_candidates_S<S>_K<K>.txt
        # Save to Samples/FL/ folder
        samples_fl_dir = joinpath(project_root, "Samples", "FL")
        if !isdir(samples_fl_dir)
            mkpath(samples_fl_dir)
        end
        selected_vectors_file = joinpath(samples_fl_dir, "selected_candidates_S$(S)_K$(K).txt")
        save_selected_binary_vectors(selected_vectors, selected_vectors_file)
    else
        println("⚠️  Warning: No policies were selected!")
    end
    
    println("\n✅ Assignment phase completed!")
    
    # Return results
    results = Dict(
        "obj_value" => obj_value,
        "ratio" => ratio,
        "z" => z,
        "reformulated_obj" => reformulated_obj,
        "total_work_units" => total_work_units
    )
    
    return results
end

# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 4
        println("❌ Error: Incorrect number of arguments!")
        println("Usage: julia Assignment_Phase.jl <instance_file> <S> <K> <binary_vectors_file> [samples_file] [experiment_type] [unmet_pen] [scaling_factor] [capacity_factor]")
        println("  instance_file: Instance file name (e.g., cap91.txt)")
        println("  S: Number of online samples")
        println("  K: Number of policies to select")
        println("  binary_vectors_file: Path to file containing binary vectors")
        println("  samples_file: Optional path to file containing samples (if not provided, will generate)")
        println("  experiment_type: 'low', 'high', or 'bernoulli' (default: 'low', used if samples_file not provided)")
        println("  unmet_pen: Penalty for unmet demand (default: 15.0)")
        println("  scaling_factor: Scaling factor for transportation cost (default: 0.2)")
        println("  capacity_factor: Factor to scale facility capacities (default: 0.3)")
        println("\nExample:")
        println("  julia Assignment_Phase.jl cap91.txt 25 2 Samples/FL/Binary_vectors/B50_bernoulli.txt")
        println("  julia Assignment_Phase.jl cap91.txt 25 2 Samples/FL/Binary_vectors/B50_bernoulli.txt Samples/FL/Scenarios/S25_bernoulli.txt")
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
    
    # Parse K (number of policies to select)
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
    
    binary_vectors_file = ARGS[4]
    samples_file = length(ARGS) > 4 ? ARGS[5] : nothing
    experiment_type = length(ARGS) > 5 ? ARGS[6] : "low"
    # Choose defaults based on instance file (FL1 vs FL2 style)
    default_unmet, default_scaling, default_capacity = get_default_fl_parameters(instance_file)
    unmet_pen = length(ARGS) > 6 ? parse(Float64, ARGS[7]) : default_unmet
    scaling_factor = length(ARGS) > 7 ? parse(Float64, ARGS[8]) : default_scaling
    capacity_factor = length(ARGS) > 8 ? parse(Float64, ARGS[9]) : default_capacity
    
    # Run assignment phase
    results = assignment_phase(
        instance_file, S, binary_vectors_file, K;
        experiment_type=experiment_type,
        unmet_pen=unmet_pen,
        scaling_factor=scaling_factor,
        capacity_factor=capacity_factor,
        samples_file=samples_file
    )
    
    println("\n" * "="^80)
    println("ASSIGNMENT PHASE COMPLETED")
    println("="^80)
    println("Reformulated objective: $(round(results["reformulated_obj"], digits=2))")
    println("Total work units: $(round(results["total_work_units"], digits=2))")
end

