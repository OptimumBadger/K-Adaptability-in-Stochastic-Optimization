# Assignment_Phase.jl
# Phase 2: Assignment Phase
# This file handles:
#   - Loading binary vectors from Candidate_List_Generation_Phase
#   - Generating S online samples (with reproducibility via random seed)
#   - Computing extensive form cost matrix
#   - Solving reformulated extensive form for policy selection

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

# =============================================================================
# PHASE 2: SOLVE FOR CONTINUOUS VARIABLES
# =============================================================================

# Structure to hold model and constraint references for efficient updates
mutable struct ModelStructure
    model::Model
    p_g::Vector{VariableRef}
    f::Vector{VariableRef}
    theta::Vector{VariableRef}
    unmet::Vector{VariableRef}
    power_balance::Vector{ConstraintRef}  # References to power balance constraints
    line_flow_lower::Dict{Int, ConstraintRef}  # Lower bounds on flow (depends on solved_x)
    line_flow_upper::Dict{Int, ConstraintRef}  # Upper bounds on flow (depends on solved_x)
    line_flow_zero::Dict{Int, ConstraintRef}  # Flow = 0 when line is off (depends on solved_x)
    dc_power_flow::Dict{Int, ConstraintRef}  # DC power flow constraints (depends on solved_x)
    B_ij::Vector{Float64}
    f_ij::Vector{Float64}
    p_max::Vector{Float64}
    p_min::Vector{Float64}
    c_i::Vector{Float64}
    L::Dict  # Branch dictionary
    B::Dict  # Bus dictionary
end

function build_model_structure(data, log_file=nothing)
    """Build the model structure once (without solved_x-dependent constraints)"""
    # Initialize the model
    model = Model(Gurobi.Optimizer)
    MOI.set(model, MOI.TimeLimitSec(), 1800.0)
    
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
    @variable(model, unmet[1:length(B)] >= 0)    # Unmet demand
    
    # Parameters
    B_ij, f_ij, p_max, p_min, A, c_i = parameters(data; line_capacity_factor=1.0)
    Penalty = 10000
    
    # Objective Function
    @objective(model, Min, sum(c_i[i] * p_g[i] for i in 1:length(B)) + 
                          Penalty * sum(unmet[i] for i in 1:length(B)))
    
    # Power Balance Constraints - Create with placeholder RHS (will be updated)
    power_balance = Vector{ConstraintRef}()
    for i in 1:length(B)
        # Calculate outgoing and incoming flows
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
        # Create constraint with placeholder RHS (0 initially, will be updated)
        push!(power_balance, @constraint(model, p_g[i] - outgoing_flows + incoming_flows + unmet[i] == 0))
    end
    
    # Generator Limits (don't depend on solved_x or sample)
    for i in 1:length(B)
        @constraint(model, p_g[i] >= p_min[i])
        @constraint(model, p_g[i] <= p_max[i])
    end
    
    # Reference angle (doesn't depend on solved_x or sample)
    @constraint(model, theta[1] == 0)
    
    # Initialize dictionaries for solved_x-dependent constraints (will be created/updated later)
    line_flow_lower = Dict{Int, ConstraintRef}()
    line_flow_upper = Dict{Int, ConstraintRef}()
    line_flow_zero = Dict{Int, ConstraintRef}()
    dc_power_flow = Dict{Int, ConstraintRef}()
    
    return ModelStructure(model, p_g, f, theta, unmet, power_balance,
                         line_flow_lower, line_flow_upper, line_flow_zero, dc_power_flow,
                         B_ij, f_ij, p_max, p_min, c_i, L, B)
end

function update_model_for_solve(model_struct::ModelStructure, sample::Vector{Float64}, solved_x::Vector{Int})
    """Update model for a new solve: update RHS of power balance and recreate solved_x-dependent constraints"""
    model = model_struct.model
    L = model_struct.L
    B_ij = model_struct.B_ij
    f_ij = model_struct.f_ij
    f = model_struct.f
    theta = model_struct.theta
    
    # Update RHS of power balance constraints
    Pd = sample .* 0.9
    for i in 1:length(model_struct.power_balance)
        set_normalized_rhs(model_struct.power_balance[i], Pd[i])
    end
    
    # Delete old solved_x-dependent constraints
    for branch_index in keys(model_struct.line_flow_lower)
        delete(model, model_struct.line_flow_lower[branch_index])
    end
    for branch_index in keys(model_struct.line_flow_upper)
        delete(model, model_struct.line_flow_upper[branch_index])
    end
    for branch_index in keys(model_struct.line_flow_zero)
        delete(model, model_struct.line_flow_zero[branch_index])
    end
    for branch_index in keys(model_struct.dc_power_flow)
        delete(model, model_struct.dc_power_flow[branch_index])
    end
    
    # Clear dictionaries
    empty!(model_struct.line_flow_lower)
    empty!(model_struct.line_flow_upper)
    empty!(model_struct.line_flow_zero)
    empty!(model_struct.dc_power_flow)
    
    # Recreate solved_x-dependent constraints
    for (branch_id, branch) in L
        branch_index = parse(Int, branch_id)
        f_bus = branch["f_bus"]
        t_bus = branch["t_bus"]
        
        if solved_x[branch_index] == 1  # Line is in service
            # Line flow limits
            model_struct.line_flow_lower[branch_index] = @constraint(model, -f_ij[branch_index] <= f[branch_index])
            model_struct.line_flow_upper[branch_index] = @constraint(model, f[branch_index] <= f_ij[branch_index])
            # DC Power Flow
            model_struct.dc_power_flow[branch_index] = @constraint(model, f[branch_index] == B_ij[branch_index] * (theta[f_bus] - theta[t_bus]))
        else  # Line is out of service
            model_struct.line_flow_zero[branch_index] = @constraint(model, f[branch_index] == 0)
        end
    end
end

function solve_for_continuous_var_direct(data, sample, solved_x, log_file=nothing)
    """Solve for continuous variables given fixed binary decisions - Direct implementation"""
    # Initialize the model
    model = Model(Gurobi.Optimizer)
    MOI.set(model, MOI.TimeLimitSec(), 1800.0)
    
    # Set logging
    if log_file !== nothing
        # Delete existing log file if it exists to ensure fresh start (no appending)
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
    
    # Decision Variables (explicit p_g, f, theta)
    @variable(model, p_g[1:length(B)])           # Power generation at each bus
    @variable(model, f[1:length(L)])             # Power flow on each line
    @variable(model, theta[1:length(B)])         # Voltage angle at each bus
    @variable(model, unmet[1:length(B)] >= 0)    # Unmet demand
    
    # Parameters
    B_ij, f_ij, p_max, p_min, A, c_i = parameters(data; line_capacity_factor=1.0)
    Pd = sample .* 0.9
    Penalty = 10000
    
    # Objective Function - Direct implementation
    @objective(model, Min, sum(c_i[i] * p_g[i] for i in 1:length(B)) + 
                          Penalty * sum(unmet[i] for i in 1:length(B)))
    
    # Power Balance Constraint (18b) - Direct implementation
    for i in 1:length(B)
        # Calculate outgoing and incoming flows
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
    
    # Line Flow Limits (18d) - With fixed x values
    for (branch_id, branch) in L
        branch_index = parse(Int, branch_id)
        if solved_x[branch_index] == 1  # Line is in service
            @constraint(model, -f_ij[branch_index] <= f[branch_index])
            @constraint(model, f[branch_index] <= f_ij[branch_index])
        else  # Line is out of service
            @constraint(model, f[branch_index] == 0)
        end
    end
    
    # DC Power Flow (18e) - With fixed x values
    for (branch_id, branch) in L
        branch_index = parse(Int, branch_id)
        f_bus = branch["f_bus"]
        t_bus = branch["t_bus"]
        
        if solved_x[branch_index] == 1  # Line is in service
            @constraint(model, f[branch_index] == B_ij[branch_index] * (theta[f_bus] - theta[t_bus]))
        end
        # If line is out of service, f[branch_index] is already fixed to 0 above
    end
    
    # Reference angle
    @constraint(model, theta[1] == 0)
    
    # Solve
    optimize!(model)
    
    if termination_status(model) == MOI.OPTIMAL
        penalty_cost = Penalty * sum(value.(unmet))
        penalty_flag = sum(value.(unmet)) >= 0.000001 ? 1 : 0
        work_units = solve_time(model)
        
        return objective_value(model), collect(value.(f)), penalty_flag, penalty_cost, work_units
    else
        return -1, [], -1, -1, -1
    end
end

# =============================================================================
# UTILITY FUNCTIONS FOR BINARY VECTORS
# =============================================================================

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
                    # Validate that all values are 0 or 1
                    if !all(v -> v == 0 || v == 1, values)
                        println("⚠️  Warning: Line $line_num contains non-binary values, converting to binary")
                        values = [v > 0.5 ? 1 : 0 for v in values]
                    end
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

# =============================================================================
# EXTENSIVE FORM FUNCTIONS
# =============================================================================

function Calculations_For_Extensive_Form(data, online_samples, S, l, x_chosen, log_file=nothing)
    """Calculate cost matrix for extensive form (OPTIMIZED: builds model once per solution)
    
    Args:
        data: Network data
        online_samples: List of S online samples (each is a vector of demands)
        S: Number of online samples
        l: Number of candidate solutions (binary vectors)
        x_chosen: List of l binary vectors (candidate solutions)
        log_file: Optional log file path
    
    Returns:
        obj_value: S x l matrix of objective values
        solved_x: List of S lists, each containing l flow vectors
        total_work_units: Total work units across all solves
    """
    L = data["branch"]
    solved_x = [Vector{Float64}[] for _ in 1:S]
    obj_value = zeros(S, l)
    
    total_work_units = 0.0
    
    println("Computing extensive form matrix (optimized: building model once, updating constraints per solve)...")
    
    # Build model structure ONCE (base structure with variables, objective, generator limits, etc.)
    model_struct = build_model_structure(data, log_file)
    
    # Initialize solved_x storage
    for i in 1:S
        solved_x[i] = [zeros(length(L)) for _ in 1:l]
    end
    
    # For each scenario-solution pair, update constraints and solve
    for (i, sample) in zip(1:S, online_samples)
        for solution in 1:l
            # Update model for this scenario-solution pair:
            # 1. Update RHS of power balance constraints (sample-dependent)
            # 2. Delete and recreate solved_x-dependent constraints (solution-dependent)
            update_model_for_solve(model_struct, sample, x_chosen[solution])
            
            # Solve
            optimize!(model_struct.model)
            
            if termination_status(model_struct.model) == MOI.OPTIMAL
                obj_value[i, solution] = objective_value(model_struct.model)
                solved_x[i][solution] = collect(value.(model_struct.f))
                penalty_cost = 10000 * sum(value.(model_struct.unmet))
                penalty_flag = sum(value.(model_struct.unmet)) >= 0.000001 ? 1 : 0
                work_units = solve_time(model_struct.model)
                total_work_units += work_units
            else
                obj_value[i, solution] = -1
                solved_x[i][solution] = zeros(length(L))
                total_work_units += solve_time(model_struct.model)
            end
        end
        
        if i % 20 == 0
            println("  Processed $i scenarios")
        end
    end
    
    return obj_value, solved_x, total_work_units
end

function Reformulated_Extensive_Form(S, l, K, obj_value, log_file=nothing)
    """Solve reformulated extensive form for policy selection
    
    Args:
        S: Number of scenarios
        l: Number of candidate solutions
        K: Number of policies to select
        obj_value: S x l matrix of objective values
        log_file: Optional log file path
    
    Returns:
        objective_value: Optimal objective value
        z: Binary vector indicating selected policies (length l)
        solve_time: Time taken to solve
        work_units: Work units used
    """
    # Initialize model
    model = Model(Gurobi.Optimizer)
    MOI.set(model, MOI.TimeLimitSec(), 3600.0)  # 1 hour

    # Set logging based on whether we want to save logs
    if log_file !== nothing
        # Delete existing log file if it exists to ensure fresh start (no appending)
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

    # Constraints
    for scenario in 1:S
        for solution in 1:l
            if obj_value[scenario, solution] == -1
                @constraint(model, w[solution, scenario] == 0)   # weights of infeasible scenario/first stage decision pair is 0
            end
        end
    end
        
    for j in 1:S
        @constraint(model, [i in 1:l], w[i, j] <= z[i])
        @constraint(model, sum(w[i, j] for i in 1:l) == 1)
    end
    @constraint(model, sum(z[i] for i in 1:l) <= K)

    # Objective
    objective = 0.0
    for scenario in 1:S
        for solution in 1:l
            objective += w[solution, scenario]*obj_value[scenario, solution]
        end
    end

    # Define the objective function
    @objective(model, Min, objective/S)

    # Solve the model
    optimize!(model)

    # Check solver status
    if termination_status(model) == MOI.OPTIMAL
        work_units = solve_time(model)  # Get work units
        return objective_value(model), value.(z), solve_time(model), work_units
    else
        return -1, [], -1, -1
    end    
end

# =============================================================================
# MAIN FUNCTION: ASSIGNMENT PHASE
# =============================================================================

function assignment_phase(instance_file::String, S::Int, binary_vectors_input::Union{Vector{Vector{Int}}, String}, K::Int;
                          sigma_low::Float64=0.04, sigma_high::Float64=0.06,
                          use_bernoulli::Bool=false, 
                          log_dir::String="logs",
                          samples_file::Union{String, Nothing}=nothing)
    """
    Assignment Phase: Evaluate candidate solutions on online samples and select K policies
    
    Args:
        instance_file: Name of the instance file (e.g., "case118Blumsack.m")
        S: Number of online samples (used if samples_file is not provided)
        binary_vectors_input: Either:
            - List of binary vectors from Phase 1 (Vector{Vector{Int}})
            - Path to file containing binary vectors (String), one vector per line
        K: Number of policies to select
        sigma_low, sigma_high: Parameters for scenario generation (used if samples_file is not provided)
        use_bernoulli: Whether to use Bernoulli mode for scenario generation (used if samples_file is not provided)
        log_dir: Directory for log files
        samples_file: Optional path to file containing samples (if provided, loads from file instead of generating)
    
    Returns:
        results: Dictionary containing:
            - obj_value: S x l cost matrix
            - solved_x: List of flow vectors
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
    println("ASSIGNMENT PHASE")
    println("="^80)
    println("Instance: $instance_file")
    println("Number of online samples: $S")
    println("Number of candidate solutions: $(length(binary_vectors))")
    println("Number of policies to select: $K")
    println("="^80)
    
    # Set random seed for reproducibility (always 1234)
    Random.seed!(1234)
    println("✅ Random seed set to 1234 (ensuring reproducibility)")
    
    # Get the directory where this script is located
    script_dir = @__DIR__
    project_root = dirname(dirname(script_dir))  # Go up two levels to project root
    
    # Construct file paths
    instance_path = joinpath(project_root, "Transmission_Switching", "instances", instance_file)
    excel_path = joinpath(project_root, "Transmission_Switching", "instances", "Random_Demands.xlsx")
    
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
    
    if !isfile(excel_path)
        error("❌ Error: Excel file not found! Expected: $excel_path")
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
    
    # Load or generate online samples
    if samples_file !== nothing
        println("\n" * "="^40)
        println("STEP 2: LOADING ONLINE SAMPLES FROM FILE")
        println("="^40)
        
        # Load samples from file
        online_samples = load_samples_from_file(samples_file)
        S = length(online_samples)  # Update S to match loaded samples
        println("✅ Loaded $S samples from file")
    else
        # Load demand data from Excel (needed for generation)
        demand_mean_data = load_demand_data(excel_path, case_name)
        
        println("\n" * "="^40)
        println("STEP 2: GENERATING ONLINE SAMPLES")
        println("="^40)
        
        # Generate online samples (with seed already set above for reproducibility)
        online_samples = generate_powerflow_scenarios(S, demand_mean_data, sigma_low, sigma_high, use_bernoulli)
    end
    
    println("\n" * "="^40)
    println("STEP 3: COMPUTING EXTENSIVE FORM")
    println("="^40)
    
    # Generate timestamp for this run (format: YYYYMMDD_HHMMSS)
    timestamp = Dates.format(now(), "YYYYmmdd_HHMMSS")
    log_file_extensive = joinpath(log_path, "extensive_form_$(timestamp).log")
    
    # Convert binary_vectors to the format expected by Calculations_For_Extensive_Form
    # x_chosen should be a list of l binary vectors
    l = length(binary_vectors)
    x_chosen = binary_vectors  # Already in the correct format
    
    # Compute extensive form cost matrix
    obj_value, solved_x, total_work_units = Calculations_For_Extensive_Form(
        network, online_samples, S, l, x_chosen, log_file_extensive
    )
    
    println("✅ Extensive form computed: $(S) scenarios × $(l) solutions")
    println("   Total work units: $(round(total_work_units, digits=2))")
    
    println("\n" * "="^40)
    println("STEP 4: SOLVING REFORMULATED EXTENSIVE FORM")
    println("="^40)
    
    log_file_reformulated = joinpath(log_path, "reformulated_extensive_form_$(timestamp).log")
    
    # Solve reformulated extensive form to select K policies
    reformulated_obj, z, solve_time_reform, work_units_reform = Reformulated_Extensive_Form(
        S, l, K, obj_value, log_file_reformulated
    )
    
    if reformulated_obj != -1
        println("✅ Reformulated extensive form solved successfully!")
        println("   Objective value: $(round(reformulated_obj, digits=2))")
        println("   Selected policies: $(findall(x -> x > 0.5, z))")
        println("   Solve time: $(round(solve_time_reform, digits=2)) seconds")
        println("   Work units: $(round(work_units_reform, digits=2))")
    else
        println("❌ Failed to solve reformulated extensive form")
    end
    
    println("\n" * "="^40)
    println("STEP 5: SUMMARY")
    println("="^40)
    println("Total candidate solutions: $l")
    println("Total online samples: $S")
    println("Policies to select: $K")
    println("Log files stored in: $log_path")
    
    # Calculate statistics
    successful_evals = 0
    failed_evals = 0
    
    for scenario in 1:S
        for solution in 1:l
            if obj_value[scenario, solution] != -1
                successful_evals += 1
            else
                failed_evals += 1
            end
        end
    end
    
    println("\nStatistics:")
    println("  Successful evaluations: $successful_evals")
    println("  Failed evaluations: $failed_evals")
    println("  Total work units (extensive form): $(round(total_work_units, digits=2))")
    if reformulated_obj != -1
        println("  Reformulated objective: $(round(reformulated_obj, digits=2))")
        println("  Work units (reformulated): $(round(work_units_reform, digits=2))")
    end
    
    println("\n✅ Assignment phase completed!")
    
    # Save selected K binary vectors to file
    if reformulated_obj != -1 && length(z) > 0
        # Find indices of selected policies (z[i] > 0.5 means selected)
        selected_indices = findall(x -> x > 0.5, z)
        selected_vectors = [binary_vectors[idx] for idx in selected_indices]
        
        if length(selected_vectors) > 0
            # Save into Samples/TS with name: selected_candidates_S<S>_K<K>.txt
            selected_dir = joinpath(project_root, "Samples", "TS")
            if !isdir(selected_dir)
                mkpath(selected_dir)
            end
            selected_vectors_file = joinpath(selected_dir, "selected_candidates_S$(S)_K$(K).txt")
            save_selected_binary_vectors(selected_vectors, selected_vectors_file)
            println("📁 Selected policies file saved: $selected_vectors_file")
        else
            println("⚠️  Warning: No policies were selected!")
        end
    end
    
    # Return results
    results = Dict(
        "obj_value" => obj_value,
        "solved_x" => solved_x,
        "z" => z,
        "reformulated_obj" => reformulated_obj,
        "total_work_units" => total_work_units,
        "work_units_reform" => work_units_reform
    )
    
    return results
end

# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 4
        println("❌ Error: Incorrect number of arguments!")
        println("Usage: julia Assignment_Phase.jl <instance_file> <S> <K> <binary_vectors_file> [samples_file] [sigma_low] [sigma_high] [bernoulli]")
        println("  instance_file: Instance file name (e.g., case118Blumsack.m)")
        println("  S: Number of online samples (used if samples_file not provided)")
        println("  K: Number of policies to select")
        println("  binary_vectors_file: Path to file containing binary vectors (one vector per line)")
        println("  samples_file: Optional path to file containing samples (if provided, loads from file)")
        println("  sigma_low: Lower bound for sigma (default: 0.04, used if samples_file not provided)")
        println("  sigma_high: Upper bound for sigma (default: 0.06, used if samples_file not provided)")
        println("  bernoulli: Use Bernoulli mode (default: false, used if samples_file not provided)")
        println("\nExample:")
        println("  julia Assignment_Phase.jl case118Blumsack.m 25 2 binary_vectors.txt")
        println("  julia Assignment_Phase.jl case118Blumsack.m 25 2 binary_vectors.txt SAA_samples.txt")
        println("  julia Assignment_Phase.jl case118Blumsack.m 25 2 binary_vectors.txt SAA_samples.txt 0.04 0.06 false")
        exit(1)
    end
    
    instance_file = ARGS[1]
    
    # Parse S (number of online samples for Phase 2)
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
    
    # Binary vectors file path
    binary_vectors_file = ARGS[4]
    
    # Check if samples_file is provided (5th argument)
    # If it's a file path (contains .txt), treat it as samples_file
    # Otherwise, treat it as sigma_low
    samples_file = nothing
    arg_offset = 0
    
    if length(ARGS) > 4
        if endswith(ARGS[5], ".txt") || isfile(ARGS[5])
            samples_file = ARGS[5]
            arg_offset = 1
        end
    end
    
    # Parse optional arguments (accounting for samples_file if provided)
    sigma_low = length(ARGS) > (4 + arg_offset) ? parse(Float64, ARGS[5 + arg_offset]) : 0.04
    sigma_high = length(ARGS) > (5 + arg_offset) ? parse(Float64, ARGS[6 + arg_offset]) : 0.06
    use_bernoulli = length(ARGS) > (6 + arg_offset) ? parse(Bool, ARGS[7 + arg_offset]) : false
    
    # Run Phase 2 with binary vectors loaded from file
    results = assignment_phase(
        instance_file, S, binary_vectors_file, K;
        sigma_low=sigma_low, sigma_high=sigma_high,
        use_bernoulli=use_bernoulli,
        samples_file=samples_file
    )
    
    println("\n" * "="^80)
    println("ASSIGNMENT PHASE COMPLETED")
    println("="^80)
    if results["reformulated_obj"] != -1
        println("Final objective value: $(round(results["reformulated_obj"], digits=2))")
    end
end
