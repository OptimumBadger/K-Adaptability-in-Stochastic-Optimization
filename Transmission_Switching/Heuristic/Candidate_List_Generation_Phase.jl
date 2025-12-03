# Candidate_List_Generation_Phase.jl
# Phase 1: Scenario Generation and Candidate List Generation
# This file handles:
#   - Loading demand data from Excel
#   - Generating PowerFlow scenarios (with reproducibility)
#   - Solving each scenario to get binary vectors (candidate solutions)
#   - Storing log files for each scenario

using PowerModels
using JuMP, Gurobi
using MathOptInterface
const MOI = MathOptInterface
using DataFrames, XLSX, Random, Distributions
using Printf
using Statistics
using Dates

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

function extract_case_name(instance_file::String)
    """Extract case name from instance file (e.g., 'case118' from 'case118Blumsack.m')"""
    # Remove extension
    base_name = replace(instance_file, r"\.m$" => "")
    # Extract case number (e.g., 'case118' from 'case118Blumsack')
    match_result = match(r"^(case\d+)", base_name)
    if match_result !== nothing
        return String(match_result.captures[1])  # Convert SubString to String
    else
        error("Could not extract case name from instance file: $instance_file")
    end
end

function load_demand_data(file_path_excel::String, case_name::String; instance_file::Union{String,Nothing}=nothing)
    """Load demand mean values from Excel file.
    
    If the worksheet is missing, fall back to extracting demand from MATPOWER file
    (aggregating load pd per bus).
    """
    # Try Excel first
    try
        if !isfile(file_path_excel)
            error("Excel file not found")
        end
        
        println("Loading demand data from: $file_path_excel (worksheet: $case_name)")
        param_data = XLSX.readtable(file_path_excel, case_name) |> DataFrame
        demand_mean_data = Vector{Float64}(param_data[1, :])
        
        println("✅ Demand data loaded successfully!")
        println("Number of buses: $(length(demand_mean_data))")
        return demand_mean_data
    catch e
        # Fallback to MATPOWER file
        if instance_file === nothing || !isfile(instance_file)
            error("❌ Error: Could not load demand data from Excel (worksheet '$case_name' not found) and no valid MATPOWER file provided for fallback.\nExcel error: $e")
        end
        
        println("⚠️  Excel worksheet '$case_name' not found. Falling back to MATPOWER file: $instance_file")
        data = PowerModels.parse_file(instance_file)
        B = data["bus"]
        L = data["load"]
        nb = length(B)
        
        demand_mean_data = zeros(Float64, nb)
        for (_, load) in L
            if haskey(load, "load_bus") && haskey(load, "pd")
                bus_idx = load["load_bus"]
                if 1 <= bus_idx <= nb
                    demand_mean_data[bus_idx] += load["pd"]
                end
            end
        end
        
        println("✅ Demand data extracted from MATPOWER file!")
        println("Number of buses: $(length(demand_mean_data))")
        println("Total demand: $(round(sum(demand_mean_data), digits=2))")
        return demand_mean_data
    end
end

function generate_powerflow_scenarios(nscen::Int, demand_mean_data::Vector{Float64}, 
                                      sigma_low::Float64=0.04, sigma_high::Float64=0.06, 
                                      use_bernoulli::Bool=false)
    """Generate demand scenarios using the PowerFlow-style formula:
    demand[node] = demand_mean[node] + sigma * N(0,1)[node] * demand_mean[node] + global_noise * sigma * demand_mean[node]
    
    For Bernoulli mode:
    demand[node] = Bernoulli(0.8) * (demand_mean[node] + sigma * N(0,1)[node] * demand_mean[node] + global_noise * sigma * demand_mean[node])
    
    Where:
    - sigma ~ Uniform(sigma_low, sigma_high) (common for all nodes)
    - N(0,1)[node] = independent normal random variable for each node
    - global_noise ~ N(0,1) (common for all nodes)
    - use_bernoulli: if true, apply Bernoulli(0.8) multiplication
    """
    bernoulli_str = use_bernoulli ? " (Bernoulli mode)" : ""
    println("Generating $nscen PowerFlow scenarios with sigma range [$sigma_low, $sigma_high]$bernoulli_str...")
    
    scenarios_list = Vector{Vector{Float64}}()
    
    for scenario in 1:nscen
        # Generate sigma (common for all nodes in this scenario)
        sigma = rand(Uniform(sigma_low, sigma_high))
        
        # Generate global noise (common for all nodes in this scenario)
        global_noise = rand(Normal(0, 1))
        
        if scenario <= 3  # Debug first 3 scenarios
            println("   Scenario $scenario: sigma=$(round(sigma, digits=4)), global_noise=$(round(global_noise, digits=4))")
        end
        
        # Generate scenario demand for each node
        scenario_demand = Float64[]
        
        for node in 1:length(demand_mean_data)
            demand_mean_node = demand_mean_data[node]
            
            # Generate node-specific normal random variable
            node_noise = rand(Normal(0, 1))
            
            # Calculate base demand
            if use_bernoulli
                # For Bernoulli mode: apply formula
                base_demand = demand_mean_node + 
                             sigma * node_noise * demand_mean_node + 
                             global_noise * sigma * demand_mean_node
                
                # Apply Bernoulli(0.8): 20% chance of 0, 80% chance of base_demand
                demand_node = rand(Bernoulli(0.8)) * base_demand
            else
                # Standard formula
                demand_node = demand_mean_node + 
                             sigma * node_noise * demand_mean_node + 
                             global_noise * sigma * demand_mean_node
            end
            
            push!(scenario_demand, max(0.0, demand_node))  # Ensure non-negativity
        end
        
        push!(scenarios_list, scenario_demand)
    end
    
    println("✅ Generated $nscen scenarios successfully!")
    return scenarios_list
end

function parameters(network; line_capacity_factor=1.0)
    """Extract network parameters from PowerModels data"""
    # Sets
    B = network["bus"]
    L = network["branch"]
    G = network["gen"]

    # Parameters
    B_ij = zeros(length(L)) # Susceptance values
    fbar_ij = zeros(length(L)) # Upper bound on the load
    p_max = zeros(length(B)) # Max power output
    p_min = zeros(length(B)) # Min power output
    c = zeros(length(B)) # Linear Cost coefficient with zero intercept

    for (branch_id, branch) in L
        B_ij[parse(Int, branch_id)] = -branch["br_x"]/(branch["br_r"]^2 + branch["br_x"]^2)
        # Handle missing rate_a key - use a default value if not present
        rate_a = haskey(branch, "rate_a") ? branch["rate_a"] : 2.2  # Default thermal limit
        fbar_ij[parse(Int, branch_id)] = rate_a * line_capacity_factor
    end

    for (gen_id, generator) in G
        c[generator["gen_bus"]] = generator["cost"][1]
        p_max[generator["gen_bus"]] = generator["pmax"]
        p_min[generator["gen_bus"]] = generator["pmin"]
    end
    
    n = 2*length(network["bus"])        # Number of rows
    m = length(network["branch"])       # Number of columns

    # Initialize matrix
    A = zeros(n, m)

    # Create A
    for (branch_id, branch) in L
        f_bus = branch["f_bus"]  # From bus
        t_bus = branch["t_bus"]  # To bus

        # Update incidence matrix
        branch_index = parse(Int, branch_id)
        A[2*f_bus-1, branch_index] = 1   # From bus
        A[2*f_bus, branch_index] = -1    # From bus
        A[2*t_bus-1, branch_index] = -1  # To bus
        A[2*t_bus, branch_index] = 1     # To bus
    end

    return B_ij, fbar_ij, p_max, p_min, A, c
end

function offline_calc_direct(data, center_data, log_file=nothing)
    """Direct implementation of the MILP formulation for offline calculations"""
    # Initialize the model
    model = Model(Gurobi.Optimizer)
    MOI.set(model, MOI.TimeLimitSec(), 120.0)  # 120 seconds
    
    # Set logging - always show console output to see gap and progress
    if log_file !== nothing
        # Delete existing log file if it exists to ensure fresh start (no appending)
        if isfile(log_file)
            rm(log_file)
        end
        set_optimizer_attribute(model, "LogFile", log_file)
        set_optimizer_attribute(model, "LogToConsole", 1)
    end
    # Always enable console output to see gap and solver progress
    set_optimizer_attribute(model, "OutputFlag", 1)
    
    # Sets
    B = data["bus"]
    L = data["branch"] 
    G = data["gen"]
    
    # Decision Variables
    @variable(model, p_g[1:length(B)])           # Power generation at each bus
    @variable(model, f[1:length(L)])             # Power flow on each line
    @variable(model, theta[1:length(B)])         # Voltage angle at each bus
    @variable(model, x[1:length(L)], Bin)        # Binary line status
    @variable(model, unmet[1:length(B)] >= 0)    # Unmet demand
    
    # Parameters
    B_ij, f_ij, p_max, p_min, A, c_i = parameters(data; line_capacity_factor=1.0)
    Pd = center_data .* 0.9
    Penalty = 5000
    
    # Calculate M value for all lines using the new method
    # Step 1: Calculate fbar_ij / |B_ij| (handle division by zero, use absolute value since B_ij can be negative)
    # n_nodes = length(B)
    # n_branches = length(L)
    # fbar_over_B = Float64[]
    # for i in 1:n_branches
    #     if abs(B_ij[i]) > 1e-10  # Avoid division by zero
    #         push!(fbar_over_B, f_ij[i] / abs(B_ij[i]))
    #     else
    #         push!(fbar_over_B, 0.0)  # If B_ij is zero, set to 0
    #     end
    # end
    # 
    # # Step 2: Sort in decreasing order and take first n-1 elements
    # sorted_fbar_over_B = sort(fbar_over_B, rev=true)
    # n_minus_1 = n_nodes - 1
    # top_n_minus_1 = sorted_fbar_over_B[1:n_minus_1]
    # 
    # # Step 3: Sum them up
    # sum_top = sum(top_n_minus_1)
    # 
    # # Step 4: Multiply by largest value of B_ij
    # max_B_ij = maximum(abs.(B_ij))
    # 
    # # Step 5: Use this as M for all lines
    # M_ij = sum_top * max_B_ij
    
    # Calculate M_ij for each line: M_ij = -12.6 * B_ij
    M_ij = -12.6 .* B_ij
    
    # Objective Function
    @objective(model, Min, sum(c_i[parse(Int, bus_id)] * p_g[parse(Int, bus_id)] for bus_id in keys(B)) + 
                          Penalty * sum(unmet[i] for i in 1:length(B)))
    
    # Power Balance Constraint
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
    
    # Generator Limits
    for i in 1:length(B)
        @constraint(model, p_g[i] >= p_min[i])
        @constraint(model, p_g[i] <= p_max[i])
    end
    
    # Line Flow Limits
    for (branch_id, branch) in L
        branch_index = parse(Int, branch_id)
        @constraint(model, -f_ij[branch_index] * x[branch_index] <= f[branch_index])
        @constraint(model, f[branch_index] <= f_ij[branch_index] * x[branch_index])
    end
    
    # DC Power Flow
    for (branch_id, branch) in L
        branch_index = parse(Int, branch_id)
        f_bus = branch["f_bus"]
        t_bus = branch["t_bus"]
        # Use M_ij[branch_index] for each line
        
        @constraint(model, B_ij[branch_index] * (theta[f_bus] - theta[t_bus]) - 
                            M_ij[branch_index] * (1 - x[branch_index]) <= f[branch_index])
        @constraint(model, f[branch_index] <= B_ij[branch_index] * (theta[f_bus] - theta[t_bus]) + 
                                            M_ij[branch_index] * (1 - x[branch_index]))
    end
    
    # Reference angle
    @constraint(model, theta[1] == 0)
    
    # Solve
    optimize!(model)
    
    if termination_status(model) == MOI.OPTIMAL
        result = value.(x)
        # Round binary variables to exact 0/1
        rounded_x = round.(Int, result)
        work_units = solve_time(model)
        penalty_cost = Penalty * sum(value(unmet[i]) for i in 1:length(B))
        penalty_flag = penalty_cost > 1e-6 ? 1 : 0
        theta_values = value.(theta)  # Get theta values
        return objective_value(model), collect(rounded_x), penalty_flag, penalty_cost, work_units, theta_values
    else
        status = termination_status(model)
        println("❌ Scenario solve failed with status: $status")
        return -1, [], -1, -1, -1, []
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
                                 sigma_low::Float64=0.04, sigma_high::Float64=0.06,
                                 use_bernoulli::Bool=false, 
                                 log_dir::String="logs",
                                 output_file::Union{String, Nothing}=nothing)
    """
    Generate candidate list by:
    1. Loading demand data from Excel
    2. Generating S scenarios (with reproducibility via random seed)
    3. Solving each scenario to get binary vectors
    4. Storing log files for each scenario
    
    Returns:
    - binary_vectors: List of binary vectors (candidate solutions)
    - objective_values: List of objective values
    - penalty_flags: List of penalty flags
    - penalty_costs: List of penalty costs
    - work_units_list: List of work units
    """
    
    println("="^80)
    println("CANDIDATE LIST GENERATION PHASE")
    println("="^80)
    println("Instance: $instance_file")
    println("Number of scenarios: $S")
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
    
    # Load demand data (Excel if available, else MATPOWER fallback)
    demand_mean_data = load_demand_data(excel_path, case_name; instance_file=instance_path)
    
    println("\n" * "="^40)
    println("STEP 2: GENERATING SCENARIOS")
    println("="^40)
    
    # Generate scenarios (with seed already set above for reproducibility)
    scenarios = generate_powerflow_scenarios(S, demand_mean_data, sigma_low, sigma_high, use_bernoulli)
    
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
        
        obj_value, binary_vector, penalty_flag, penalty_cost, work_units, theta_values = offline_calc_direct(network, scenario_demand, log_file)
        
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
            # Determine experiment type
            experiment_type = use_bernoulli ? "bernoulli" : "normal"
            
            # Save to Samples/TS/Binary_vectors/L<S>_<experiment_type>.txt
            binary_vectors_dir = joinpath(project_root, "Samples", "TS", "Binary_vectors")
            if !isdir(binary_vectors_dir)
                mkpath(binary_vectors_dir)
            end
            output_file = joinpath(binary_vectors_dir, "L$(S)_$(experiment_type).txt")
        end
        
        save_binary_vectors_to_file(binary_vectors, output_file)
        println("✅ Binary vectors saved to: $output_file")
        println("   Total vectors: $(length(binary_vectors))")
        println("   Vector length: $(length(binary_vectors[1]))")
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
        println("Usage: julia Candidate_List_Generation_Phase.jl <instance_file> <S> [sigma_low] [sigma_high] [bernoulli]")
        println("Example: julia Candidate_List_Generation_Phase.jl case118Blumsack.m 25")
        println("Example: julia Candidate_List_Generation_Phase.jl case118Blumsack.m 25 0.04 0.06 false")
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
    sigma_low = length(ARGS) > 2 ? parse(Float64, ARGS[3]) : 0.04
    sigma_high = length(ARGS) > 3 ? parse(Float64, ARGS[4]) : 0.06
    use_bernoulli = length(ARGS) > 4 ? parse(Bool, ARGS[5]) : false
    
    # Run main function (random seed is always 1234, hardcoded)
    binary_vectors, objective_values, penalty_flags, penalty_costs, work_units_list = generate_candidate_list(
        instance_file, S; 
        sigma_low=sigma_low, sigma_high=sigma_high,
        use_bernoulli=use_bernoulli
    )
    
    println("\n📊 Results stored in variables:")
    println("  - binary_vectors: $(length(binary_vectors)) binary vectors")
    println("  - objective_values: $(length(objective_values)) objective values")
    println("  - penalty_flags: $(length(penalty_flags)) penalty flags")
    println("  - penalty_costs: $(length(penalty_costs)) penalty costs")
    println("  - work_units_list: $(length(work_units_list)) work units")
end

