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

function extract_experiment_type_from_filename(file_path::String)
    """Extract experiment type (low/high/bernoulli) from a filename."""
    filename = lowercase(basename(file_path))
    m = match(r"_(low|high|bernoulli)\.txt$", filename)
    if m === nothing
        return "unknown"
    end
    return m.captures[1]
end

function json_escape(value::String)
    escaped = replace(value, "\\" => "\\\\", "\"" => "\\\"", "\n" => "\\n", "\r" => "\\r", "\t" => "\\t")
    return "\"" * escaped * "\""
end

function json_value(value)
    if value === nothing
        return "null"
    elseif value isa String
        return json_escape(value)
    elseif value isa Bool
        return value ? "true" : "false"
    elseif value isa Number
        return string(value)
    elseif value isa Vector
        return "[" * join([json_value(v) for v in value], ", ") * "]"
    elseif value isa Dict
        keys_sorted = sort(collect(keys(value)))
        entries = [json_escape(string(k)) * ": " * json_value(value[k]) for k in keys_sorted]
        return "{" * join(entries, ", ") * "}"
    else
        return json_escape(string(value))
    end
end

function write_json_file(file_path::String, data::Dict)
    dir = dirname(file_path)
    if !isempty(dir) && !isdir(dir)
        mkpath(dir)
    end
    open(file_path, "w") do file
        println(file, json_value(data))
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
    set_optimizer_attribute(model, "WorkLimit", 3600.0)
    
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

function filter_unique_solutions(feasible_solutions::Vector{Vector{Float64}}, tolerance::Float64=1e-6)
    """Filter out duplicate solutions from the feasible solutions list
    
    Args:
        feasible_solutions: List of z vectors (one per solution)
        tolerance: Tolerance for comparing floating point values (default: 1e-6)
    
    Returns:
        unique_solutions: List of unique z vectors
    """
    unique_solutions = Vector{Vector{Float64}}()
    
    for z_vector in feasible_solutions
        is_duplicate = false
        for existing_z in unique_solutions
            # Check if z_vector is the same as existing_z (within tolerance)
            if length(z_vector) == length(existing_z)
                if all(abs(z_vector[i] - existing_z[i]) < tolerance for i in 1:length(z_vector))
                    is_duplicate = true
                    break
                end
            end
        end
        if !is_duplicate
            push!(unique_solutions, z_vector)
        end
    end
    
    return unique_solutions
end

function save_feasible_solutions(feasible_solutions::Vector{Vector{Float64}}, 
                                all_binary_vectors::Vector{Vector{Int}}, 
                                K::Int, file_path::String)
    """Save feasible solutions to a file
    
    Format:
    Solution 1
    <binary vector for policy 1>
    <binary vector for policy 2>
    ...
    <binary vector for policy K>
    Solution 2
    ...
    
    Args:
        feasible_solutions: List of z vectors (one per solution)
        all_binary_vectors: List of all candidate binary vectors (to map z indices to actual vectors)
        K: Number of policies per solution
        file_path: Path to output file
    """
    if isempty(feasible_solutions)
        println("⚠️  Warning: No feasible solutions to save!")
        return
    end
    
    # Filter out duplicate solutions
    unique_solutions = filter_unique_solutions(feasible_solutions)
    
    if length(unique_solutions) < length(feasible_solutions)
        println("ℹ️  Filtered out $(length(feasible_solutions) - length(unique_solutions)) duplicate solutions")
        println("   Keeping $(length(unique_solutions)) unique solutions")
    end
    
    dir = dirname(file_path)
    if !isempty(dir) && !isdir(dir)
        mkpath(dir)
    end
    
    open(file_path, "w") do file
        for (sol_idx, z_vector) in enumerate(unique_solutions)
            println(file, "Solution $sol_idx")
            selected_indices = findall(x -> x > 0.5, z_vector)
            if length(selected_indices) != K
                println("⚠️  Warning: Solution $sol_idx has $(length(selected_indices)) policies, expected $K. Saving anyway.")
            end
            for idx in selected_indices
                println(file, join(all_binary_vectors[idx], " "))
            end
        end
    end
    println("📁 Feasible solutions saved to: $file_path")
    println("   Total unique feasible solutions saved: $(length(unique_solutions))")
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
            
            work_units = 0.0
            try
                work_units = MOI.get(backend(model_struct.model), Gurobi.ModelAttribute("Work"))
            catch
            end
            if termination_status(model_struct.model) == MOI.OPTIMAL
                obj_value[scenario, solution] = objective_value(model_struct.model)
                fixed_cost = sum(fixedcost[i] * value(model_struct.x[i]) for i in 1:nloc)
                ratio[scenario, solution] = fixed_cost / obj_value[scenario, solution]
                total_work_units += work_units
            else
                obj_value[scenario, solution] = -1
                ratio[scenario, solution] = -1
                total_work_units += work_units
            end
        end
    end
    
    return obj_value, ratio, total_work_units
end

function Reformulated_Extensive_Form(S, l, K, obj_value, log_file=nothing; T::Int=10)
    """Solve the reformulated extensive form to select K optimal solutions
    
    Args:
        S: Number of scenarios
        l: Number of candidate solutions
        K: Number of policies to select
        obj_value: S x l matrix of objective values
        log_file: Optional log file path
        T: Number of feasible solutions to collect from solution pool (default: 10)
    
    Returns:
        objective_value: Optimal objective value
        z: Binary vector indicating selected policies (length l) - optimal solution
        feasible_solutions: List of z vectors (one per feasible solution, up to T solutions)
        work_units: Work units used by Gurobi for this solve
    """
    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "WorkLimit", 3600.0)
    
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

    # Enable solution pool to collect multiple feasible solutions
    set_optimizer_attribute(model, "PoolSearchMode", 2)  # Find all solutions
    set_optimizer_attribute(model, "PoolSolutions", T)   # Keep top T solutions

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
        work_units = 0.0
        try
            work_units = MOI.get(backend(model), Gurobi.ModelAttribute("Work"))
        catch
        end
        optimal_z = value.(z)
        
        # Collect feasible solutions from solution pool (filtering duplicates as we go)
        feasible_solutions = Vector{Vector{Float64}}()
        opt = backend(model)
        tolerance = 1e-6
        
        try
            solcount = MOI.get(opt, Gurobi.ModelAttribute("SolCount"))
            println("ℹ️  Gurobi found $solcount feasible solution(s) in the solution pool")
            num_solutions = min(T, solcount)
            
            for k in 0:num_solutions-1
                MOI.set(opt, MOI.RawOptimizerAttribute("SolutionNumber"), k)
                z_k = [MOI.get(opt, Gurobi.VariableAttribute("Xn"), JuMP.index(z[i])) for i in 1:l]
                
                # Check if this solution is a duplicate
                is_duplicate = false
                for existing_z in feasible_solutions
                    if length(z_k) == length(existing_z)
                        if all(abs(z_k[i] - existing_z[i]) < tolerance for i in 1:length(z_k))
                            is_duplicate = true
                            break
                        end
                    end
                end
                
                if !is_duplicate
                    push!(feasible_solutions, z_k)
                end
            end
            
            # Note: Solution #0 from Gurobi's pool is always the optimal solution,
            # so optimal_z should already be in feasible_solutions (as the first entry)
            
            # Print summary of solutions collected
            if length(feasible_solutions) < solcount
                println("ℹ️  After filtering duplicates: $(length(feasible_solutions)) unique solution(s) kept")
            end
        catch e
            println("⚠️  Warning: Could not access solution pool: $e")
            # Fallback: only return optimal solution
            push!(feasible_solutions, optimal_z)
        end
        
        return objective_value(model), optimal_z, feasible_solutions, work_units
    else
        return -1, [], Vector{Vector{Float64}}(), -1.0
    end
end

# =============================================================================
# MAIN FUNCTION: ASSIGNMENT PHASE
# =============================================================================

function assignment_phase(instance_file::String, L::Int, S::Int, binary_vectors_input::Union{Vector{Vector{Int}}, String}, K::Int;
                          unmet_pen::Float64=15.0,
                          scaling_factor::Float64=0.2,
                          capacity_factor::Float64=0.3,
                          log_dir::String="logs",
                          samples_file::String,
                          T::Int=10)
    """
    Assignment Phase: Evaluate candidate solutions on online samples and select K policies
    
    Args:
        instance_file: Name of the instance file (e.g., "cap91.txt")
        L: Number of candidate policies to use (first L from binary_vectors_file)
        S: Number of scenarios to use (first S from samples_file)
        binary_vectors_input: Either:
            - List of binary vectors from Phase 1 (Vector{Vector{Int}})
            - Path to file containing binary vectors (String), one vector per line
        K: Number of policies to select
        unmet_pen: Penalty for unmet demand
        scaling_factor: Scaling factor for transportation cost
        capacity_factor: Factor to scale facility capacities
        log_dir: Directory for log files
        samples_file: Path to file containing samples (required)
        T: Number of feasible solutions to collect from solution pool (default: 10)
    
    Returns:
        results: Dictionary containing:
            - obj_value: S x l cost matrix
            - z: Selected policies (binary vector)
            - reformulated_obj: Objective value from reformulated extensive form
            - total_work_units: Total work units
            - feasible_solutions: List of feasible z vectors (up to T solutions)
    """
    
    # Load binary vectors if filename is provided, otherwise use directly
    all_binary_vectors = if isa(binary_vectors_input, String)
        load_binary_vectors_from_file(binary_vectors_input)
    else
        binary_vectors_input
    end
    
    if length(all_binary_vectors) == 0
        error("❌ Error: No binary vectors provided or loaded!")
    end
    
    # Validate L and slice first L candidates
    if L > length(all_binary_vectors)
        error("❌ Error: Requested L = $L candidate policies, but binary_vectors_file contains only $(length(all_binary_vectors)) vectors.")
    end
    binary_vectors = all_binary_vectors[1:L]
    l = L  # Update l to reflect the sliced L
    
    println("="^80)
    println("ASSIGNMENT PHASE - FACILITY LOCATION")
    println("="^80)
    println("Instance: $instance_file")
    println("Number of candidate solutions available: $(length(all_binary_vectors))")
    println("Using first L = $L candidates and first S = $S scenarios")
    println("Number of policies to select: $K")
    println("="^80)
    
    # Get the directory where this script is located
    script_dir = @__DIR__
    project_root = dirname(dirname(script_dir))  # Go up two levels to project root
    
    # Construct file paths
    instance_path = joinpath(project_root, "Facility_Location", "instances", instance_file)
    
    # Create log directories if they don't exist
    log_path = joinpath(script_dir, log_dir)
    extensive_log_dir = joinpath(log_path, "extensive_form")
    assignment_log_dir = joinpath(log_path, "assignment_phase")
    for dir in (log_path, extensive_log_dir, assignment_log_dir)
        if !isdir(dir)
            mkpath(dir)
            println("Created log directory: $dir")
        end
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
    
    # Load samples from file (required)
    println("\nLoading samples from file: $samples_file")
    all_samples = load_samples_from_file(samples_file)
    
    # Validate S and slice first S scenarios
    if S > length(all_samples)
        error("❌ Error: Requested S = $S scenarios, but samples_file contains only $(length(all_samples)) scenarios.")
    end
    online_samples = all_samples[1:S]
    println("✅ Loaded $(length(online_samples)) samples (first S from file)")
    
    println("\n" * "="^40)
    println("STEP 2: COMPUTING EXTENSIVE FORM")
    println("="^40)
    
    # Convert binary vectors to matrix format (l x nloc)
    # Note: l is already set to L from slicing above
    xsolutions = zeros(Int, l, nloc)
    for (i, vec) in enumerate(binary_vectors)
        if length(vec) != nloc
            error("❌ Error: Binary vector $i has length $(length(vec)), expected $nloc")
        end
        xsolutions[i, :] = vec
    end
    
    # Generate timestamp and experiment tag for log files
    timestamp = Dates.format(now(), "YYYYmmdd_HHMMSS")
    exp_tag = extract_experiment_type_from_filename(samples_file)
    extensive_form_log = joinpath(extensive_log_dir, "extensive_form_L$(L)_S$(S)_$(exp_tag)_$(timestamp).log")
    
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
    
    reformulated_log = joinpath(assignment_log_dir, "reformulated_extensive_form_L$(L)_S$(S)_K$(K)_$(exp_tag)_$(timestamp).log")
    reformulated_obj, z, feasible_solutions, work_units_reformulated = Reformulated_Extensive_Form(S, l, K, obj_value, reformulated_log; T=T)
    
    if reformulated_obj == -1
        error("❌ Error: Reformulated extensive form did not solve to optimality!")
    end
    
    println("✅ Reformulated extensive form solved")
    println("   Objective value: $(round(reformulated_obj, digits=2))")
    println("   Selected policies: $(sum(z .> 0.5))")
    println("   Feasible solutions found: $(length(feasible_solutions))")
    println("   Work units (reformulated): $(round(work_units_reformulated, digits=2))")
    
    # Prepare output directories under Samples/FL
    samples_fl_dir = joinpath(project_root, "Samples", "FL")
    if !isdir(samples_fl_dir)
        mkpath(samples_fl_dir)
    end
    selected_candidates_dir = joinpath(samples_fl_dir, "Selected_Candidates")
    feasible_solutions_dir = joinpath(samples_fl_dir, "Feasible_Solutions")
    if !isdir(selected_candidates_dir)
        mkpath(selected_candidates_dir)
    end
    if !isdir(feasible_solutions_dir)
        mkpath(feasible_solutions_dir)
    end
    
    # Save selected K binary vectors to file
    selected_indices = findall(x -> x > 0.5, z)
    selected_vectors = [binary_vectors[idx] for idx in selected_indices]
    
    if length(selected_vectors) > 0
        # Generate filename for selected vectors: selected_candidates_L<L>_S<S>_K<K>_<experiment>.txt
        selected_vectors_file = joinpath(selected_candidates_dir, "selected_candidates_L$(L)_S$(S)_K$(K)_$(exp_tag).txt")
        save_selected_binary_vectors(selected_vectors, selected_vectors_file)
    else
        println("⚠️  Warning: No policies were selected!")
    end
    
    # Save feasible solutions to file
    if length(feasible_solutions) > 0
        feasible_solutions_file = joinpath(feasible_solutions_dir, "feasible_solutions_L$(L)_S$(S)_K$(K)_T$(T)_$(exp_tag).txt")
        save_feasible_solutions(feasible_solutions, all_binary_vectors, K, feasible_solutions_file)
        println("📁 Feasible solutions file saved: $feasible_solutions_file")
    end
    
    # Write JSON summary for Assignment Phase
    instance_tag = splitext(basename(instance_file))[1]
    results_dir = joinpath(project_root, "Results", "Facility_Location")
    assignment_json_file = joinpath(results_dir, "assignment_results_$(instance_tag)_L$(L)_S$(S)_K$(K)_$(exp_tag).json")
    assignment_json = Dict(
        "instance" => instance_file,
        "L" => L,
        "S" => S,
        "K" => K,
        "experiment" => exp_tag,
        "work_units_continuous" => total_work_units,
        "assignment_work_units" => work_units_reformulated
    )
    write_json_file(assignment_json_file, assignment_json)
    println("📁 Assignment JSON saved to: $assignment_json_file")
    
    println("\n✅ Assignment phase completed!")
    
    # Return results
    results = Dict(
        "obj_value" => obj_value,
        "ratio" => ratio,
        "z" => z,
        "reformulated_obj" => reformulated_obj,
        "total_work_units" => total_work_units,
        "feasible_solutions" => feasible_solutions,
        "work_units_reformulated" => work_units_reformulated
    )
    
    return results
end

# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 6
        println("❌ Error: Incorrect number of arguments!")
        println("Usage: julia Assignment_Phase.jl <instance_file> <L> <S> <K> <binary_vectors_file> <samples_file> [T] [unmet_pen] [scaling_factor] [capacity_factor]")
        println("  instance_file: Instance file name (e.g., cap91.txt)")
        println("  L: Number of candidate policies to use (first L from binary_vectors_file)")
        println("  S: Number of scenarios to use (first S from samples_file)")
        println("  K: Number of policies to select")
        println("  binary_vectors_file: Path to file containing binary vectors (e.g., B300_bernoulli.txt)")
        println("  samples_file: Path to file containing scenarios (e.g., S300_bernoulli.txt)")
        println("  T: Number of feasible solutions to collect from solution pool (default: 10)")
        println("  unmet_pen: Penalty for unmet demand (default: auto-detected from instance)")
        println("  scaling_factor: Scaling factor for transportation cost (default: auto-detected from instance)")
        println("  capacity_factor: Factor to scale facility capacities (default: auto-detected from instance)")
        println("\nExample:")
        println("  julia Assignment_Phase.jl cap91.txt 50 30 2 Samples/FL/Binary_vectors/B300_bernoulli.txt Samples/FL/Scenarios/S300_bernoulli.txt")
        exit(1)
    end
    
    instance_file = ARGS[1]
    
    # Parse L (number of candidate policies to use)
    L = try
        parsed = parse(Int, ARGS[2])
        if parsed <= 0
            throw(ArgumentError("L must be positive"))
        end
        parsed
    catch
        println("❌ Error: L must be a positive integer!")
        println("You provided: '$(ARGS[2])'")
        exit(1)
    end
    
    # Parse S (number of scenarios to use)
    S = try
        parsed = parse(Int, ARGS[3])
        if parsed <= 0
            throw(ArgumentError("S must be positive"))
        end
        parsed
    catch
        println("❌ Error: S must be a positive integer!")
        println("You provided: '$(ARGS[3])'")
        exit(1)
    end
    
    # Parse K (number of policies to select)
    K = try
        parsed = parse(Int, ARGS[4])
        if parsed <= 0
            throw(ArgumentError("K must be positive"))
        end
        parsed
    catch
        println("❌ Error: K must be a positive integer!")
        println("You provided: '$(ARGS[4])'")
        exit(1)
    end
    
    binary_vectors_file = ARGS[5]
    samples_file = ARGS[6]
    
    # Parse T (number of feasible solutions to collect)
    T = length(ARGS) > 6 ? try
        parsed = parse(Int, ARGS[7])
        if parsed <= 0
            throw(ArgumentError("T must be positive"))
        end
        parsed
    catch
        println("❌ Error: T must be a positive integer!")
        println("You provided: '$(ARGS[7])'")
        exit(1)
    end : 10
    
    # Choose defaults based on instance file (FL1 vs FL2 style)
    default_unmet, default_scaling, default_capacity = get_default_fl_parameters(instance_file)
    unmet_pen = length(ARGS) > 7 ? parse(Float64, ARGS[8]) : default_unmet
    scaling_factor = length(ARGS) > 8 ? parse(Float64, ARGS[9]) : default_scaling
    capacity_factor = length(ARGS) > 9 ? parse(Float64, ARGS[10]) : default_capacity
    
    # Run assignment phase
    results = assignment_phase(
        instance_file, L, S, binary_vectors_file, K;
        unmet_pen=unmet_pen,
        scaling_factor=scaling_factor,
        capacity_factor=capacity_factor,
        samples_file=samples_file,
        T=T
    )
    
    println("\n" * "="^80)
    println("ASSIGNMENT PHASE COMPLETED")
    println("="^80)
    println("Reformulated objective: $(round(results["reformulated_obj"], digits=2))")
    println("Total work units: $(round(results["total_work_units"], digits=2))")
end

