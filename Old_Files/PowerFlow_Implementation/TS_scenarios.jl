# TS_scenarios.jl - Scenario Generation and Binary Vector Storage
# Usage: julia TS_scenarios.jl <instance_file> <S>
# Example: julia TS_scenarios.jl case118Blumsack.m 25

using PowerModels
using JuMP, Gurobi
using MathOptInterface
const MOI = MathOptInterface
using DataFrames, XLSX, Random, Distributions
using Printf
using Base.MathConstants

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

function load_demand_data(file_path_excel::String, case_name::String)
    """Load demand mean values from Excel file"""
    if !isfile(file_path_excel)
        error("❌ Error: Excel file not found! Expected: $file_path_excel")
    end
    
    println("Loading demand data from: $file_path_excel (worksheet: $case_name)")
    
    # Read the Excel data containing demand parameters
    param_data = XLSX.readtable(file_path_excel, case_name) |> DataFrame
    
    # Extract demand mean from Excel file (first row contains the demand values)
    demand_mean_data = Vector{Float64}(param_data[1, :])
    
    println("✅ Demand data loaded successfully!")
    println("Number of buses: $(length(demand_mean_data))")
    
    return demand_mean_data
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
    @variable(model, theta[1:length(B)])  # Voltage angle at each bus (restricted to [0, π/4])
    @variable(model, x[1:length(L)], Bin)        # Binary line status
    @variable(model, unmet[1:length(B)] >= 0)    # Unmet demand
    
    # Parameters
    B_ij, f_ij, p_max, p_min, A, c_i = parameters(data; line_capacity_factor=1.0)
    Pd = center_data .* 0.9
    Penalty = 5000
    
    # Calculate M value for all lines using the new method
    # Step 1: Calculate fbar_ij / |B_ij| (handle division by zero, use absolute value since B_ij can be negative)
    n_nodes = length(B)
    n_branches = length(L)
    fbar_over_B = Float64[]
    for i in 1:n_branches
        if abs(B_ij[i]) > 1e-10  # Avoid division by zero
            push!(fbar_over_B, f_ij[i] / abs(B_ij[i]))
        else
            push!(fbar_over_B, 0.0)  # If B_ij is zero, set to 0
        end
    end
    
    # Step 2: Sort in decreasing order and take first n-1 elements
    sorted_fbar_over_B = sort(fbar_over_B, rev=true)
    n_minus_1 = n_nodes - 1
    top_n_minus_1 = sorted_fbar_over_B[1:n_minus_1]
    
    # Step 3: Sum them up
    sum_top = sum(top_n_minus_1)
    
    # Step 4: Multiply by largest value of B_ij
    max_B_ij = maximum(abs.(B_ij))
    
    # Step 5: Use this as M for all lines (make it negative as in original formula)
    M_ij = sum_top
    
    println("  Calculated M_ij = $M_ij (using sum of top $(n_minus_1) fbar/B values: $sum_top, max_B_ij: $max_B_ij)")
    
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
        #@constraint(model, theta[i] <= π/4)
        #@constraint(model, theta[i] >= -π/4)
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
        #M_ij = -12.6/8 * B_ij[branch_index]  # Big-M value
        
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
# MAIN FUNCTION
# =============================================================================

function main(instance_file::String, S::Int)
    println("="^80)
    println("TS_SCENARIOS - Scenario Generation and Binary Vector Storage")
    println("="^80)
    println("Instance: $instance_file")
    println("Number of scenarios: $S")
    println("="^80)
    
    # Set random seed for reproducibility
    Random.seed!(1234)
    println("✅ Random seed set to 1234")
    
    # Get the directory where this script is located
    script_dir = @__DIR__
    
    # Construct file paths (instance and Excel file in powerflow_instances subdirectory)
    instance_path = joinpath(script_dir, "powerflow_instances", instance_file)
    excel_path = joinpath(script_dir, "powerflow_instances", "Random_Demands.xlsx")
    
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
    
    # Load demand data from Excel
    demand_mean_data = load_demand_data(excel_path, case_name)
    
    println("\n" * "="^40)
    println("STEP 2: GENERATING SCENARIOS")
    println("="^40)
    
    # Generate scenarios (using default sigma range and no Bernoulli)
    scenarios = generate_powerflow_scenarios(S, demand_mean_data, 0.04, 0.06, false)
    
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
    
    for (scenario_idx, scenario_demand) in enumerate(scenarios)
        println("Solving scenario $scenario_idx/$S...")
        
        obj_value, binary_vector, penalty_flag, penalty_cost, work_units, theta_values = offline_calc_direct(network, scenario_demand)
        
        if obj_value != -1
            push!(binary_vectors, binary_vector)
            push!(objective_values, obj_value)
            push!(penalty_flags, penalty_flag)
            push!(penalty_costs, penalty_cost)
            push!(work_units_list, work_units)
            successful_solves += 1
            println("  ✅ Solved successfully: Objective = $(round(obj_value, digits=2)), Work Units = $(round(work_units, digits=2))")
            
            # Print theta_i - theta_j table for this scenario
            println("\n  THETA DIFFERENCES TABLE (Scenario $scenario_idx):")
            println("  " * "-"^76)
            @printf("  %-8s | %-8s | %-8s | %-20s\n", "Branch", "From", "To", "theta_i - theta_j")
            println("  " * "-"^76)
            
            L = network["branch"]
            for (branch_id, branch) in L
                branch_index = parse(Int, branch_id)
                f_bus = branch["f_bus"]
                t_bus = branch["t_bus"]
                theta_diff = theta_values[f_bus] - theta_values[t_bus]
                @printf("  %-8d | %-8d | %-8d | %-20.6f\n", branch_index, f_bus, t_bus, theta_diff)
            end
            println("  " * "-"^76)
            println()
        else
            failed_solves += 1
            println("  ❌ Failed to solve scenario $scenario_idx")
        end
    end
    
    println("\n" * "="^40)
    println("STEP 4: SUMMARY")
    println("="^40)
    println("Total scenarios: $S")
    println("Successfully solved: $successful_solves")
    println("Failed: $failed_solves")
    println("Binary vectors stored: $(length(binary_vectors))")
    
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
    
    println("\n✅ Scenario generation and solving completed!")
    println("Binary vectors are stored in the 'binary_vectors' list")
    
    return binary_vectors, objective_values, penalty_flags, penalty_costs, work_units_list
end

# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) != 2
        println("❌ Error: Incorrect number of arguments!")
        println("Usage: julia TS_scenarios.jl <instance_file> <S>")
        println("Example: julia TS_scenarios.jl case118Blumsack.m 25")
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
    
    # Run main function
    binary_vectors, objective_values, penalty_flags, penalty_costs, work_units_list = main(instance_file, S)
    
    println("\n📊 Results stored in variables:")
    println("  - binary_vectors: $(length(binary_vectors)) binary vectors")
    println("  - objective_values: $(length(objective_values)) objective values")
    println("  - penalty_flags: $(length(penalty_flags)) penalty flags")
    println("  - penalty_costs: $(length(penalty_costs)) penalty costs")
    println("  - work_units_list: $(length(work_units_list)) work units")
end

