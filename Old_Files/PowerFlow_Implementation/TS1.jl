# Complete PowerFlow Model - Direct Implementation (case118Blumsack.m)
# Core 3 Phases: Scenario Generation, Policy Selection, Evaluation
# Usage: julia complete_powerflow_direct.jl <S> <experiment_type>
# Example: julia complete_powerflow_direct.jl 100 low

using PowerModels
using JuMP, Gurobi, DataFrames, XLSX, Random, Distributions
using Graphs, GraphPlot, Plots

# Include shared scenarios
# =============================================================================
# SCENARIO GENERATION FUNCTIONS (Embedded from powerflow_scenarios.jl)
# =============================================================================

# Global variables to store shared data and scenarios
global demand_mean = Float64[]
global scenarios_dict = Dict{Int, Vector{Vector{Float64}}}()
global scenario_params_dict = Dict{Int, Dict{String, Any}}()
global scenario_counter = 0  # Counter for generating unique scenarios

function load_demand_data(file_path_excel = "powerflow_instances/Random_Demands.xlsx", instance_file_path = "powerflow_instances/case118Blumsack.m")
    """Load demand mean values from Excel file"""
    if !isfile(file_path_excel)
        error("❌ Error: Excel file not found! Expected: $file_path_excel")
    end
    
    # For TS1.jl, use case118 worksheet
    case_name = "case118"
    
    println("Loading demand data from: $file_path_excel (worksheet: $case_name)")
    
    # Read the Excel data containing demand parameters
    param_data = XLSX.readtable(file_path_excel, case_name) |> DataFrame
    
    # Extract demand mean from Excel file (first row contains the demand values)
    demand_mean_data = Vector{Float64}(param_data[1, :])
    
    println("✅ Demand data loaded successfully!")
    println("Number of buses: $(length(demand_mean_data))")
    
    return demand_mean_data
end

function generate_powerflow_scenarios(nscen::Int, demand_mean_data::Vector{Float64}, sigma_low::Float64=0.04, sigma_high::Float64=0.06, use_bernoulli::Bool=false)
    """Generate demand scenarios using the new formula:
    demand[node] = demand_mean[node] + sigma * N(0,1)[node] * demand_mean[node] + global_noise * sigma * demand_mean[node]
    
    For Bernoulli mode:
    demand[node] = Bernoulli(0.5) * (demand_mean[node] * 2 + sigma * N(0,1)[node] * demand_mean[node] * 2 + global_noise * sigma * demand_mean[node] * 2)
    
    Where:
    - sigma ~ Uniform(sigma_low, sigma_high) (common for all nodes)
    - N(0,1)[node] = independent normal random variable for each node
    - global_noise ~ N(0,1) (common for all nodes)
    - use_bernoulli: if true, apply Bernoulli(0.5) multiplication and double demand_mean
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
            println("   Initial scenario $scenario: sigma=$(round(sigma, digits=4)), global_noise=$(round(global_noise, digits=4))")
        end
        
        # Generate scenario demand for each node
        scenario_demand = Float64[]
        
        for node in 1:length(demand_mean_data)
            demand_mean_node = demand_mean_data[node]
            
            # Generate node-specific normal random variable
            node_noise = rand(Normal(0, 1))
            
            # Calculate base demand
            if use_bernoulli
                # For Bernoulli mode: apply formula without doubling
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
            
            push!(scenario_demand, demand_node)
        end
        
        push!(scenarios_list, scenario_demand)
    end
    
    println("✅ Generated $nscen scenarios successfully!")
    return scenarios_list
end

function generate_scenarios_for_count(nscen::Int, file_path_excel = "powerflow_instances/Random_Demands.xlsx", sigma_low::Float64=0.04, sigma_high::Float64=0.06, use_bernoulli::Bool=false, instance_file_path = "powerflow_instances/case118Blumsack.m")
    """Generate scenarios for a specific count and store them globally"""
    global demand_mean, scenarios_dict, scenario_params_dict
    
    # Load demand data if not already loaded
    if isempty(demand_mean)
        demand_mean = load_demand_data(file_path_excel, instance_file_path)
    end
    
    # Generate scenarios for this count
    scenarios = generate_powerflow_scenarios(nscen, demand_mean, sigma_low, sigma_high, use_bernoulli)
    
    # Store scenarios and parameters for this count
    scenarios_dict[nscen] = scenarios
    scenario_params_dict[nscen] = Dict(
        "nscen" => nscen,
        "file_path" => file_path_excel,
        "random_seed" => 1234,
        "sigma_range" => (0.04, 0.06),
        "nbuses" => length(demand_mean)
    )
    
    println("✅ Generated and stored $nscen scenarios!")
    return scenarios
end

function get_shared_powerflow_scenarios(nscen::Int)
    """Get the globally stored scenarios for a specific count"""
    global scenarios_dict
    if !haskey(scenarios_dict, nscen)
        error("❌ No scenarios found for nscen=$nscen. Run generate_scenarios_for_count($nscen) first.")
    end
    return scenarios_dict[nscen]
end

function add_scenario_to_existing_set(nscen::Int, file_path_excel = "powerflow_instances/Random_Demands.xlsx", sigma_low::Float64=0.04, sigma_high::Float64=0.06, use_bernoulli::Bool=false)
    """Add one more scenario to an existing scenario set"""
    global demand_mean, scenarios_dict, scenario_params_dict
    
    # Check if the scenario set exists
    if !haskey(scenarios_dict, nscen)
        error("❌ No existing scenario set found for nscen=$nscen. Run generate_scenarios_for_count($nscen) first.")
    end
    
    # Load demand data if not already loaded
    if isempty(demand_mean)
        demand_mean = load_demand_data(file_path_excel)
    end
    
    # Generate one new scenario (now truly random since no seeds are set)
    global scenario_counter
    scenario_counter += 1
    
    println("🔄 Generating new scenario #$scenario_counter...")
    
    # Test if rand() is actually producing different values
    test_rand1 = rand()
    test_rand2 = rand()
    println("   Random test: $test_rand1, $test_rand2")
    
    new_scenario = generate_powerflow_scenarios(1, demand_mean, sigma_low, sigma_high, use_bernoulli)[1]
    println("   New scenario first 3 values: [$(round(new_scenario[1], digits=3)), $(round(new_scenario[2], digits=3)), $(round(new_scenario[3], digits=3))]")
    
    # Add the new scenario to the existing set
    push!(scenarios_dict[nscen], new_scenario)
    
    # Update the count in parameters
    scenario_params_dict[nscen]["nscen"] = length(scenarios_dict[nscen])
    
    println("✅ Added 1 new scenario to set $nscen (now has $(length(scenarios_dict[nscen])) scenarios)")
    return new_scenario
end

# =============================================================================
# COMMAND-LINE ARGUMENT PARSING
# =============================================================================

function parse_arguments()
    """Parse command-line arguments and return S and experiment_type"""
    if length(ARGS) != 2
        println("❌ Error: Incorrect number of arguments!")
        println("Usage: julia complete_powerflow_direct.jl <S> <experiment_type>")
        println("Example: julia complete_powerflow_direct.jl 100 low")
        println("")
        println("Arguments:")
        println("  S: Number of scenarios (positive integer)")
        println("  experiment_type: One of 'low', 'high', or 'bernoulli'")
        exit(1)
    end
    
    # Parse S (number of scenarios)
    num_scenarios = try
        parsed = parse(Int, ARGS[1])
        if parsed <= 0
            throw(ArgumentError("S must be positive"))
        end
        parsed
    catch
        println("❌ Error: S must be a positive integer!")
        println("You provided: '$(ARGS[1])'")
        exit(1)
    end
    
    # Parse experiment_type
    experiment_type = lowercase(ARGS[2])
    if !(experiment_type in ["low", "high", "bernoulli"])
        println("❌ Error: experiment_type must be one of 'low', 'high', or 'bernoulli'!")
        println("You provided: '$(ARGS[2])'")
        exit(1)
    end
    
    return num_scenarios, experiment_type
end

# Set random seed for reproducibility
Random.seed!(1234)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

function append_summary_to_log(log_file, phase_name, parameters, results)
    """Append summary statistics to log file"""
    open(log_file, "a") do io
        println(io, "\n" * "="^60)
        println(io, "SUMMARY STATISTICS - $phase_name")
        println(io, "="^60)
        println(io, "Parameters:")
        for (key, value) in parameters
            println(io, "  $key: $value")
        end
        println(io, "\nResults:")
        for (key, value) in results
            println(io, "  $key: $value")
        end
        println(io, "="^60)
    end
end

# =============================================================================
# PHASE 1: SCENARIO GENERATION & OFFLINE CALCULATIONS
# =============================================================================

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
    """Direct implementation of the MILP formulation from the image"""
    # Initialize the model
    model = Model(Gurobi.Optimizer)
    MOI.set(model, MOI.TimeLimitSec(), 120.0)  # 60 seconds
    #set_optimizer_attribute(model, "MIPGap", 0.0001)  # Stop when MIP gap < 0.1%
    
    # Set logging based on whether we want to save logs
    if log_file !== nothing
        set_optimizer_attribute(model, "LogFile", log_file)
        set_optimizer_attribute(model, "LogToConsole", 0)
    else
        set_optimizer_attribute(model, "OutputFlag", 0)  # Disable console output for offline calculations
    end
    
    # Sets
    B = data["bus"]
    L = data["branch"] 
    G = data["gen"]
    
    # Decision Variables (exactly as in the image)
    @variable(model, p_g[1:length(B)])           # Power generation at each bus
    @variable(model, f[1:length(L)])             # Power flow on each line
    @variable(model, theta[1:length(B)])         # Voltage angle at each bus
    @variable(model, x[1:length(L)], Bin)        # Binary line status
    @variable(model, unmet[1:length(B)] >= 0)    # Unmet demand (your extension)
    
    # Parameters
    B_ij, f_ij, p_max, p_min, A, c_i = parameters(data; line_capacity_factor=1.0)
    Pd = center_data .* 0.9
    Penalty = 5000
    
    # Objective Function (18a) - Direct implementation (with penalty)
    @objective(model, Min, sum(c_i[parse(Int, bus_id)] * p_g[parse(Int, bus_id)] for bus_id in keys(B)) + Penalty * sum(unmet[i] for i in 1:length(B)))
    
    # Power Balance Constraint (18b) - Direct implementation
    for i in 1:length(B)
        # Calculate outgoing flows
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
    
    # Line Flow Limits (18d) - Direct implementation
    for (branch_id, branch) in L
        branch_index = parse(Int, branch_id)
        @constraint(model, -f_ij[branch_index] * x[branch_index] <= f[branch_index])
        @constraint(model, f[branch_index] <= f_ij[branch_index] * x[branch_index])
    end
    
    # DC Power Flow (18e) - Direct implementation
    for (branch_id, branch) in L
        branch_index = parse(Int, branch_id)
        f_bus = branch["f_bus"]
        t_bus = branch["t_bus"]
        M_ij = -12.6 * B_ij[branch_index]  # Big-M value
        
        @constraint(model, B_ij[branch_index] * (theta[f_bus] - theta[t_bus]) - 
                            M_ij * (1 - x[branch_index]) <= f[branch_index])
        @constraint(model, f[branch_index] <= B_ij[branch_index] * (theta[f_bus] - theta[t_bus]) + 
                                            M_ij * (1 - x[branch_index]))
    end
    
    # Reference angle (standard practice)
    @constraint(model, theta[1] == 0)  # Set reference bus angle to 0
    
    # Solve
    optimize!(model)
    
    if termination_status(model) == MOI.OPTIMAL
        result = value.(x)
        # Round binary variables to exact 0/1 to avoid numerical precision issues
        rounded_x = round.(Int, result)
        # Get Gurobi work units
        work_units = solve_time(model)  # This gives work units in Gurobi
        penalty_cost = Penalty * sum(value(unmet[i]) for i in 1:length(B))
        penalty_flag = penalty_cost > 1e-6 ? 1 : 0
        return objective_value(model), collect(rounded_x), penalty_flag, penalty_cost, work_units
    else
        # Debug infeasibility
        status = termination_status(model)
        println("❌ offline_calc_direct failed with status: $status")
        println("   Model has $(num_variables(model)) variables and $(num_constraints(model; count_variable_in_set_constraints=true)) constraints")
        
        if status == MOI.INFEASIBLE || status == MOI.INFEASIBLE_OR_UNBOUNDED
            status_name = status == MOI.INFEASIBLE ? "INFEASIBLE" : "INFEASIBLE_OR_UNBOUNDED"
            println("   Model is $status_name - attempting to compute conflicting constraints...")
            try
                # Try to compute an Irreducible Inconsistent Subsystem (IIS)
                compute_conflict!(model)
                println("   IIS computed - displaying conflicting constraints:")
                
                # Print conflicting constraints
                println("   Identified infeasible constraints:")
                for (F, S) in list_of_constraint_types(model)
                    for (i, constr) in enumerate(all_constraints(model, F, S))
                        conflict_status = MOI.get(model, MOI.ConstraintConflictStatus(), constr)
                        if conflict_status == MOI.IN_CONFLICT
                            println("   🔴 Constraint $i is in the IIS: ", constr)
                        end
                    end
                end
                
            catch e
                println("   Could not compute IIS: $e")
            end
            
            # Print model summary information
            println("   Model Summary:")
            println("     Variables: $(num_variables(model))")
            println("     Constraints: $(num_constraints(model; count_variable_in_set_constraints=true))")
            println("     Binary variables: $(length(x))")
            println("     Continuous variables: $(num_variables(model) - length(x))")
        elseif status == MOI.TIME_LIMIT
            println("   Model hit TIME_LIMIT")
        else
            println("   Other termination status: $status")
        end
        
        return -1, [], -1, -1, -1
    end
end


function sampling_for_offline_calc(network, nscen; capacity_factor=1.20, line_capacity_factor=1.0, log_file=nothing, sigma_low=0.04, sigma_high=0.06, use_bernoulli=false)
    """Generate feasible scenarios for offline calculations using shared scenarios with dynamic generation"""
    # Get the initial scenario set
    shared_scenarios = get_shared_powerflow_scenarios(nscen)
    
    # 1) unpack network parameters
    B_ij, fbar_ij, p_max_base, p_min_base, A, c = parameters(network; line_capacity_factor=line_capacity_factor)

    # index sets
    buses = sort(parse.(Int, collect(keys(network["bus"]))))
    branches = sort(parse.(Int, collect(keys(network["branch"]))))
    nb = length(buses)
    nl = length(branches)

    # 2) pre-allocate storage
    demand_samples = zeros(Float64, nscen, nb)
    x_solutions = zeros(nscen, nl)
    offline_costs = zeros(nscen)  # Store optimal costs
    penalty_costs = zeros(nscen)  # Store penalty costs
    penalty_scenarios = Int[]  # Store scenario indices that had penalties
    solve_times = Float64[]  # Store solve times for successfully solved scenarios
    work_units_offline = Float64[]  # Store work units for successfully solved scenarios

    # 3) sampling loop with dynamic scenario generation
    found = 0
    scenario_index = 1
    max_attempts = nscen * 10  # Safety limit to prevent infinite loops
    attempts = 0
    
    while found < nscen && attempts < max_attempts
        attempts += 1
        
        # Get current scenario (may need to refresh if we added new ones)
        current_scenarios = get_shared_powerflow_scenarios(nscen)
        
        # Check if we need more scenarios
        if scenario_index > length(current_scenarios)
            println("🔄 Scenario $scenario_index not available, generating new scenario...")
            add_scenario_to_existing_set(nscen, "powerflow_instances/Random_Demands.xlsx", sigma_low, sigma_high, use_bernoulli)
            current_scenarios = get_shared_powerflow_scenarios(nscen)  # Refresh
        end
        
        # Use current scenario
        ds = current_scenarios[scenario_index]

        # make *local* scaled copies of your capacities
        p_max = p_max_base .* capacity_factor
        p_min = p_min_base .* capacity_factor

        # try to solve
        solve_time = @elapsed begin
            obj, x, p, penalty_cost, work_units = offline_calc_direct(network, ds, log_file)
        end

        # if we get here, it solved to optimality
        if obj != -1
            found += 1
            demand_samples[found, :] = ds
            x_solutions[found, :] = x
            offline_costs[found] = obj  # Store the optimal cost
            penalty_costs[found] = penalty_cost  # Store the penalty cost
            if p == 1
                push!(penalty_scenarios, found)  # Track scenarios with penalties
            end
            push!(solve_times, solve_time)  # Store the solve time
            push!(work_units_offline, work_units)  # Store the work units
            println("✅ Scenario $scenario_index solved successfully (found: $found/$nscen) in $(round(solve_time, digits=2))s")
        else
            println("❌ Scenario $scenario_index failed (time limit or infeasible), generating replacement...")
            # Generate a new scenario to replace this failed one
            add_scenario_to_existing_set(nscen, "powerflow_instances/Random_Demands.xlsx", sigma_low, sigma_high, use_bernoulli)
            # Refresh scenarios and try the new scenario at the same position
            current_scenarios = get_shared_powerflow_scenarios(nscen)
            ds = current_scenarios[scenario_index]  # This will be the new scenario we just added
            
            # Try to solve the new scenario
            #replacement_solve_time = @elapsed begin
            #    obj, x, p, penalty_cost, work_units = offline_calc_direct(network, ds, log_file)
            #end
            
            if obj != -1
                found += 1
                demand_samples[found, :] = ds
                x_solutions[found, :] = x
                offline_costs[found] = obj
                penalty_costs[found] = penalty_cost  # Store the penalty cost
                if p == 1
                    push!(penalty_scenarios, found)  # Track scenarios with penalties
                end
                push!(solve_times, replacement_solve_time)  # Store the solve time
                push!(work_units_offline, work_units)  # Store the work units
                println("✅ Replacement scenario $scenario_index solved successfully (found: $found/$nscen) in $(round(replacement_solve_time, digits=2))s")
            else
                println(" Replacement scenario $scenario_index will be tried now!")
        end
    end

        # Move to next scenario only if successful
        scenario_index += 1
    end
    
    if found < nscen
        println("⚠️  Warning: Only found $found feasible scenarios out of $nscen requested after $attempts attempts")
        println("   This may be due to time limits or infeasible scenarios")
    else
        println("🎉 Successfully found all $nscen feasible scenarios!")
    end

    return demand_samples, x_solutions, offline_costs, solve_times, penalty_costs, penalty_scenarios, work_units_offline
end

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

function check_binary_vector_uniqueness(x_solutions, nscen)
    """Check how many unique binary vectors exist in the solution set"""
    tolerance = 1e-6
    unique_vectors = 0
    
    for i in 1:nscen
        is_unique = true
        for j in 1:(i-1)
            # Check if vectors are approximately equal
            if all(abs(x_solutions[i][k] - x_solutions[j][k]) < tolerance for k in 1:length(x_solutions[i]))
                is_unique = false
                break
            end
        end
        if is_unique
            unique_vectors += 1
        end
    end
    
    return unique_vectors
end

function print_binary_vectors(x_solutions, nscen)
    """Print all binary vectors for debugging"""
    println("\n" * "="^60)
    println("BINARY VECTORS FOR NSCEN = $nscen")
    println("="^60)
    
    for i in 1:nscen
        println("Scenario $i (length $(length(x_solutions[i]))): $(x_solutions[i])")
    end
    
    # Also print a summary of unique vectors
    tolerance = 1e-6
    unique_vectors = 0
    unique_indices = Int[]
    
    for i in 1:nscen
        is_unique = true
        for j in 1:(i-1)
            if all(abs(x_solutions[i][k] - x_solutions[j][k]) < tolerance for k in 1:length(x_solutions[i]))
                is_unique = false
                break
            end
        end
        if is_unique
            unique_vectors += 1
            push!(unique_indices, i)
        end
    end
    
    println("\nUnique vector indices: $unique_indices")
    println("Total unique vectors: $unique_vectors out of $nscen")
    println("="^60)
end

# =============================================================================
# PHASE 2: POLICY SELECTION & EXTENSIVE FORM
# =============================================================================

function solve_for_continuous_var_direct(data, sample, solved_x, log_file=nothing)
    """Solve for continuous variables given fixed binary decisions - Direct implementation"""
    # Initialize the model
    model = Model(Gurobi.Optimizer)
    MOI.set(model, MOI.TimeLimitSec(), 1800.0)
    
    # Set logging
    if log_file !== nothing
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

function Calculations_For_Extensive_Form(data, online_samples, S, l, x_chosen, log_file=nothing)
    """Calculate cost matrix for extensive form"""
    L = data["branch"]
    solved_x = [Vector{Float64}[] for _ in 1:S]
    obj_value = zeros(S, l)
    
    total_work_units = 0.0
    
    for (i, sample) in zip(1:S, eachrow(online_samples))
        solved_x[i] = [zeros(length(L)) for _ in 1:l]
        for solution in 1:l
            obj_value[i, solution], solved_x[i][solution], penalty_flag, penalty_cost, work_units = solve_for_continuous_var_direct(data, sample, x_chosen[solution, :], log_file)
            total_work_units += work_units
        end
    end
    return obj_value, solved_x, total_work_units
end

function Reformulated_Extensive_Form(S, l, K, obj_value, log_file=nothing)
    """Solve reformulated extensive form for policy selection"""
    # Initialize model
    model = Model(Gurobi.Optimizer)
    MOI.set(model, MOI.TimeLimitSec(), 3600.0)  # 1 minute

    # Set logging based on whether we want to save logs
    if log_file !== nothing
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
# PHASE 3: EVALUATION
# =============================================================================

function OfflineTest(data, test_sample, log_file=nothing)
    """Offline test calculations for test scenarios"""
    num_rows = size(test_sample, 1)
    offline_cost = zeros(num_rows)
    flag_list = zeros(num_rows)
    
    # Offline calculations for the test data
    total_cost = 0.0
    count = 0
    for (i, sample) in zip(1:num_rows, eachrow(test_sample))
        obj_value, x_found, flag = offline_calc_direct(data, sample, log_file)
        offline_cost[i] = obj_value
        flag_list[i] = flag
        if obj_value != -1
            total_cost += obj_value
            count += 1
        end
    end
    total_offline_cost = total_cost/count
    return total_offline_cost, offline_cost, flag_list
end

function Second_Stage_Cost_direct(data, sample, x_chosen, z, log_file=nothing)
    """Calculate second stage cost for evaluation - Direct implementation"""
    # Ensure x_chosen contains exact 0/1 values
    x_chosen = [round.(Int, x_chosen[i]) for i in 1:length(x_chosen)]
    
    # Initialize the model
    model = Model(Gurobi.Optimizer)
    MOI.set(model, MOI.TimeLimitSec(), 3600.0)
    
    # Set logging
    if log_file !== nothing
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
        # Debug infeasibility
        status = termination_status(model)
        println("❌ Second_Stage_Cost_direct failed with status: $status")
        
        if status == MOI.INFEASIBLE || status == MOI.INFEASIBLE_OR_UNBOUNDED
            status_name = status == MOI.INFEASIBLE ? "INFEASIBLE" : "INFEASIBLE_OR_UNBOUNDED"
            println("   Model is $status_name - attempting to compute conflicting constraints...")
            try
                compute_conflict!(model)
                println("   IIS computed - displaying conflicting constraints:")
                
                for (F, S) in list_of_constraint_types(model)
                    for (i, constr) in enumerate(all_constraints(model, F, S))
                        conflict_status = MOI.get(model, MOI.ConstraintConflictStatus(), constr)
                        if conflict_status == MOI.IN_CONFLICT
                            println("   🔴 Constraint $i is in the IIS: ", constr)
                        end
                    end
                end
            catch e
                println("   Could not compute IIS: $e")
            end
        end
        
        return -1, -1, -1
    end
end

function Evaluation(data, test_sample, offline_cost_list, x_chosen, z, log_file=nothing)
    """Evaluate model performance using stored offline costs"""
    num_rows = size(test_sample, 1)
    cost = zeros(num_rows)
    penalty = zeros(num_rows)

    online_list = zeros(num_rows)

    # Second stage cost
    model_cost = 0.0
    count = 0
    work_units_second_stage_total = 0.0  # Track total work units for second stage
    for (index, sample) in zip(1:num_rows, eachrow(test_sample))
        # Use stored offline cost (no need to check if != -1 since we stored valid costs)
            model_obj_value, work_units_second_stage, penalty_cost = Second_Stage_Cost_direct(data, sample, x_chosen, z, log_file)
            online_list[index] = model_obj_value
            penalty[index] = penalty_cost
            if model_obj_value != -1
                model_cost += model_obj_value
                count += 1
                work_units_second_stage_total += work_units_second_stage  # Accumulate work units
        end
    end
    total_model_cost = model_cost / count

    return total_model_cost, online_list, penalty, work_units_second_stage_total
end

# =============================================================================
# MAIN FUNCTION
# =============================================================================

function run_single_experiment(network, S, K_values, capacity_factor, line_capacity_factor, sigma_low, sigma_high, use_bernoulli, file_path_excel, file_path)
    """Run a single experiment with given sigma parameters"""
    # Store results for all scenario counts and K combinations
    all_results = Dict{Int, Dict{String, Any}}()
    
    # Loop through each scenario count
    for (s_idx, nscen) in enumerate(S)
        println("\n" * "="^60)
        println("PROCESSING NSCEN = $nscen ($(s_idx)/$(length(S)))")
        println("="^60)
    
        # Generate initial scenarios (we'll add more dynamically if needed)
        println("Generating initial $nscen scenarios...")
        generate_scenarios_for_count(nscen, file_path_excel, sigma_low, sigma_high, use_bernoulli, file_path)
    
        # =============================================================================
        # PHASE 1: SCENARIO GENERATION & OFFLINE CALCULATIONS
        # =============================================================================
    
        println("Starting Phase 1: Scenario Generation & Offline Calculations for nscen=$nscen...")
        phase1_time = @elapsed begin
            demand_samples, x_solutions_matrix, offline_costs, solve_times, penalty_costs, penalty_scenarios, work_units_offline = sampling_for_offline_calc(network, nscen; capacity_factor=capacity_factor, line_capacity_factor=line_capacity_factor, sigma_low=sigma_low, sigma_high=sigma_high, use_bernoulli=use_bernoulli)
        
        # Convert matrix to vector of vectors for proper binary vector handling
        x_solutions = [x_solutions_matrix[i, :] for i in 1:nscen]
        end
        
        println("Generated $nscen scenarios with $(length(x_solutions)) solutions")
        println("Average offline cost: $(round(mean(offline_costs), digits=2))")
        println("Average offline solve time: $(round(mean(solve_times), digits=2)) seconds")
        
        
        # =============================================================================
        # PHASE 2: POLICY SELECTION (EXTENSIVE FORM)
        # =============================================================================
        
        println("Starting Phase 2: Policy Selection for nscen=$nscen...")
        phase2_calc_time = @elapsed begin
            # Convert x_solutions back to matrix for Calculations_For_Extensive_Form
            x_solutions_matrix = hcat(x_solutions...)'  # Convert vector of vectors to matrix
            obj_value, solved_x, phase2_work_units = Calculations_For_Extensive_Form(network, demand_samples, nscen, nscen, x_solutions_matrix)
        end
        
        n_neg1 = count(x -> x == -1, obj_value)
        println("Infeasible scenario-solution pairs: $n_neg1")
        
        # Check uniqueness of binary vectors
        unique_binary_vectors = check_binary_vector_uniqueness(x_solutions, nscen)
        println("Unique binary vectors: $unique_binary_vectors out of $nscen scenarios")
        
        # Store results for this nscen
        all_results[nscen] = Dict(
            "calculations_time" => phase2_calc_time,
            "calculations_work_units" => phase2_work_units,
            "unique_binary_vectors" => unique_binary_vectors,
            "total_scenarios" => nscen,
            "average_offline_cost" => mean(offline_costs),
            "average_offline_solve_time" => mean(solve_times),
            "penalty_costs" => penalty_costs,
            "penalty_scenarios" => penalty_scenarios,
            "offline_costs" => offline_costs,  # Store the full offline costs array
            "binary_vectors" => x_solutions,  # Store the actual binary vectors
            "total_work_units_offline" => sum(work_units_offline),  # Total work units for offline phase
            "average_work_units_offline" => mean(work_units_offline)  # Average work units per scenario
        )
        
        # =============================================================================
        # K-SPECIFIC PHASES (LOOP THROUGH K VALUES)
        # =============================================================================
        
        for (k_idx, K) in enumerate(K_values)
            println("\n" * "="^40)
            println("PROCESSING K = $K for nscen=$nscen ($(k_idx)/$(length(K_values)))")
            println("="^40)
            
            # =============================================================================
            # PHASE 2B: REFORMULATED EXTENSIVE FORM (K-SPECIFIC)
            # =============================================================================
            
            println("Starting Phase 2B: Reformulated Extensive Form for K=$K...")
            phase2_reform_time = @elapsed begin
                Solution, z, time_taken, work_units_reform = Reformulated_Extensive_Form(nscen, nscen, K, obj_value)
            end
            
            println("Reformulated extensive form objective for K=$K: $Solution")
            
            # =============================================================================
            # PHASE 3: EVALUATION (K-SPECIFIC)
            # =============================================================================
            
            println("Starting Phase 3: Evaluation for K=$K...")
            phase3_eval_time = @elapsed begin
                model_cost, model_cost_list, penlist, work_units_second_stage_total = Evaluation(network, demand_samples, offline_costs, x_solutions, z)
            end
            
            # Calculate gap
            avg_offline_cost = mean(offline_costs)
            gap = (model_cost - avg_offline_cost) / avg_offline_cost * 100
            
            println("Results for nscen=$nscen, K=$K:")
            println("  Average offline cost: $(round(avg_offline_cost, digits=2))")
            println("  Model cost: $(round(model_cost, digits=2))")
            println("  Performance gap: $(round(gap, digits=2))%")
            println("  Phase 2B time: $(round(phase2_reform_time, digits=2)) seconds")
            println("  Phase 3B time: $(round(phase3_eval_time, digits=2)) seconds")
            
            # Store results for this K
            all_results[nscen][string(K)] = Dict(
                "solution" => Solution,
                "avg_offline_cost" => avg_offline_cost,
                "model_cost" => model_cost,
                "gap" => gap,
                "reformulated_extensive_time" => phase2_reform_time,
                "work_units_reformulated_extensive" => work_units_reform,
                "work_units_second_stage" => work_units_second_stage_total,
                "phase3b_time" => phase3_eval_time,
                "penalty_data" => penlist,  # Store penalty data for later analysis
                "model_cost_per_scenario" => model_cost_list  # Store individual scenario costs
            )
            
            println("✅ nscen=$nscen, K=$K completed successfully!")
        end
        
        println("✅ nscen=$nscen completed successfully!")
    end
    
    return all_results
end

function main(num_scenarios, experiment_type)
    println("="^80)
    println("POWERFLOW MODEL - DIRECT IMPLEMENTATION")
    println("="^80)
    println("Running experiment: $experiment_type with S=$num_scenarios scenarios")
    println("="^80)
    
    # =============================================================================
    # DATA LOADING
    # =============================================================================
    
    println("\n" * "="^40)
    println("DATA LOADING")
    println("="^40)
    
    # File paths (using relative paths for portability)
    file_path = "PowerFlow_Implementation/powerflow_instances/case118Blumsack.m"
    file_path_excel = "PowerFlow_Implementation/powerflow_instances/Random_Demands.xlsx"
    
    # Check if files exist
    if !isfile(file_path)
        println("❌ Error: MATPOWER file not found!")
        println("Expected file: $file_path")
        println("Please make sure you're running from the project root directory")
        println("and that the powerflow_instances/ folder contains the data files.")
        return
    end
    
    if !isfile(file_path_excel)
        println("❌ Error: Excel file not found!")
        println("Expected file: $file_path_excel")
        println("Please make sure you're running from the project root directory")
        println("and that the powerflow_instances/ folder contains the data files.")
        return
    end
    
    println("Loading data from: $file_path")
    println("Loading parameters from: $file_path_excel")
    
    # Parse the MATPOWER case file
    network = PowerModels.parse_file(file_path)
    
    println("✅ Network data loaded successfully!")
    println("Network: $(length(network["bus"])) buses, $(length(network["branch"])) branches, $(length(network["gen"])) generators")
    
    # =============================================================================
    # PARAMETERS
    # =============================================================================
    
    # Model parameters
    K_values = [1, 2, 3, 4, 5, 10]  # Number of policies to select
    
    println("\nModel Parameters:")
    println("  Scenario count: $num_scenarios")
    println("  K values (policies to select): $K_values")
    println("  Capacity factor: 1.0")
    
    # =============================================================================
    # EXPERIMENT CONFIGURATION
    # =============================================================================
    
    # Map experiment_type to parameters
    if experiment_type == "low"
        exp_name = "Gaussian Low"
        sigma_low = 0.04
        sigma_high = 0.06
        use_bernoulli = false
    elseif experiment_type == "high"
        exp_name = "Gaussian High"
        sigma_low = 0.09
        sigma_high = 0.11
        use_bernoulli = false
    elseif experiment_type == "bernoulli"
        exp_name = "Gaussian Bernoulli"
        sigma_low = 0.04
        sigma_high = 0.06
        use_bernoulli = true
    end
    
    println("\n" * "="^80)
    println("STARTING EXPERIMENT: $exp_name")
    println("Sigma range: [$sigma_low, $sigma_high]")
    if use_bernoulli
        println("Mode: Bernoulli (doubled demand_mean, 50% chance of zero demand)")
    end
    println("="^80)
    
    # Set capacity factors
    capacity_factor = 1
    line_capacity_factor = 1
    println("Generator capacity factor set to: $capacity_factor")
    println("Line capacity factor set to: $line_capacity_factor")
    
    # Reset random seed
    Random.seed!(1234)
    
    # =============================================================================
    # RUN SINGLE EXPERIMENT
    # =============================================================================
    
    # Run the entire algorithm for this experiment
    experiment_results = run_single_experiment(network, [num_scenarios], K_values, capacity_factor, line_capacity_factor, sigma_low, sigma_high, use_bernoulli, file_path_excel, file_path)
    
    println("\n✅ Experiment '$exp_name' completed successfully!")
    
    # =============================================================================
    # RESULTS SUMMARY
    # =============================================================================
    
    println("\n" * "="^80)
    println("FINAL RESULTS SUMMARY")
    println("="^80)
    
    # Print results for the single experiment
    println("\n" * "="^80)
    println("EXPERIMENT: $exp_name")
    println("="^80)
    
    all_results = experiment_results
    
    # Since we're running a single S value, get the results for that S
    scenario_info = all_results[num_scenarios]
    println("\n=== SCENARIO SET S=$num_scenarios ===")
    println("Average Offline Cost: $(round(scenario_info["average_offline_cost"], digits=2))")
    println("Average Offline Solve Time: $(round(scenario_info["average_offline_solve_time"], digits=2)) seconds")
    println("Total Offline Work Units: $(round(scenario_info["total_work_units_offline"], digits=2))")
    println("Average Offline Work Units: $(round(scenario_info["average_work_units_offline"], digits=2))")
    println("Calculations_For_Extensive_Form Time: $(round(scenario_info["calculations_time"], digits=2)) seconds")
    println("Calculations_For_Extensive_Form Work Units: $(round(scenario_info["calculations_work_units"], digits=2))")
    println("Unique Binary Vectors: $(scenario_info["unique_binary_vectors"]) out of $(scenario_info["total_scenarios"]) scenarios")
    
    println("\n┌─────┬─────────────┬─────────────┬─────────────────────────────┬─────────────────────┬─────────────────────┐")
    println("│  K  │ Model Cost  │ Model Gap   │ Reformulated Extensive Time │ Reform Work Units   │ Second Stage WU     │")
    println("├─────┼─────────────┼─────────────┼─────────────────────────────┼─────────────────────┼─────────────────────┤")
    
    for K in K_values
        result = scenario_info[string(K)]
        k_str = string(K)
        
        # Handle NaN values
        if isnan(result["model_cost"])
            model_cost_str = "NaN"
            gap_str = "NaN"
        else
            model_cost_str = string(round(result["model_cost"], digits=2))
            gap_str = string(round(result["gap"], digits=2)) * "%"
        end
        
        time_str = string(round(result["reformulated_extensive_time"], digits=2)) * "s"
        reform_work_units_str = string(round(result["work_units_reformulated_extensive"], digits=2))
        second_stage_work_units_str = string(round(result["work_units_second_stage"], digits=2))
        
        println("│ $(lpad(k_str, 3)) │ $(lpad(model_cost_str, 11)) │ $(lpad(gap_str, 11)) │ $(lpad(time_str, 27)) │ $(lpad(reform_work_units_str, 19)) │ $(lpad(second_stage_work_units_str, 19)) │")
    end
    println("└─────┴─────────────┴─────────────┴─────────────────────────────┴─────────────────────┴─────────────────────┘")
        
    # =============================================================================
    # OFFLINE PHASE PENALTY ANALYSIS
    # =============================================================================
    
    println("\n" * "="^60)
    println("OFFLINE PHASE PENALTY ANALYSIS - $exp_name")
    println("="^60)
    
    # Get offline penalty data
    offline_penalty_costs = scenario_info["penalty_costs"]
    offline_penalty_scenarios = scenario_info["penalty_scenarios"]
    offline_costs = scenario_info["offline_costs"]
    
    # Summary statistics
    total_offline_penalty = sum(offline_penalty_costs)
    scenarios_with_penalties = length(offline_penalty_scenarios)
    total_scenarios = length(offline_costs)
    
    println("Scenarios with penalties: $scenarios_with_penalties out of $total_scenarios")
    println("Total offline penalty cost: $(round(total_offline_penalty, digits=2))")
    println("Average offline penalty cost: $(round(mean(offline_penalty_costs), digits=2))")
    
    # Individual scenario breakdown
    if scenarios_with_penalties > 0
        println("\nIndividual scenario breakdown:")
        for scenario_idx in offline_penalty_scenarios
            model_cost = offline_costs[scenario_idx]
            penalty_cost = offline_penalty_costs[scenario_idx]
            println("  Scenario $scenario_idx: Model Cost = $(round(model_cost, digits=2)), Penalty Cost = $(round(penalty_cost, digits=2))")
        end
    else
        println("No scenarios had penalties in offline phase")
    end
        
    # =============================================================================
    # ONLINE PHASE PENALTY ANALYSIS SUMMARY FOR THIS EXPERIMENT
    # =============================================================================
    
    println("\n" * "="^60)
    println("PENALTY ANALYSIS SUMMARY - $exp_name")
    println("="^60)
    
    # Since we're running a single S value
    if haskey(all_results, num_scenarios)
        println("\n=== SCENARIO SET S=$num_scenarios ===")
        
        for K in K_values
            k_str = string(K)
            if haskey(all_results[num_scenarios], k_str) && haskey(all_results[num_scenarios][k_str], "penalty_data")
                penalty_data = all_results[num_scenarios][k_str]["penalty_data"]
                model_cost_per_scenario = all_results[num_scenarios][k_str]["model_cost_per_scenario"]
                penalty_pos = findall(x -> x > 0.1, penalty_data)
                
                if length(penalty_pos) > 0
                    println("K=$K: $(length(penalty_pos)) scenarios with penalties")
                    for (idx, pos) in enumerate(penalty_pos)
                        individual_model_cost = model_cost_per_scenario[pos]
                        penalty_cost = penalty_data[pos]
                        println("  Scenario $pos: Model Cost = $(round(individual_model_cost, digits=2)), Penalty Cost = $(round(penalty_cost, digits=1))")
                    end
                else
                    println("K=$K: No scenarios with penalties")
                end
            end
        end
    end
    
    println("\n" * "="^60)
    
    println("\n" * "="^80)
    
    # Binary vector debugging removed - issue has been resolved
    
    # =============================================================================
    # GENERATE 4 SEPARATE TEXT FILES
    # =============================================================================
    
    # Get instance metadata
    n_buses = length(network["bus"])
    n_lines = length(network["branch"])
    n_generators = length(network["gen"])
    instance_name = basename(file_path)
    
    println("\n" * "="^80)
    println("GENERATING 4 SEPARATE TEXT FILES")
    println("="^80)
    
    # 1. SUMMARY FILE
    summary_filename = "TS1_S$(num_scenarios)_$(experiment_type)_summary.txt"
    open(summary_filename, "w") do io
        println(io, "$instance_name\t$n_buses\t$n_lines\t$n_generators\t$experiment_type\t$(round(scenario_info["average_offline_cost"], digits=2))\t$(round(scenario_info["average_work_units_offline"], digits=2))\t$(round(scenario_info["calculations_work_units"], digits=2))\t$(scenario_info["unique_binary_vectors"])")
    end
    println("✅ Summary file: $summary_filename")
    
    # 2. K RESULTS FILE
    kresults_filename = "TS1_S$(num_scenarios)_$(experiment_type)_kresults.txt"
    open(kresults_filename, "w") do io
        for K in K_values
            result = scenario_info[string(K)]
            k_str = string(K)
            
            # Handle NaN values
            if isnan(result["model_cost"])
                model_cost = -1.0
                gap = -100.0
            else
                model_cost = round(result["model_cost"], digits=2)
                gap = round(result["gap"], digits=2)
            end
            
            println(io, "$instance_name\t$n_buses\t$n_lines\t$n_generators\t$experiment_type\t$K\t$model_cost\t$gap\t$(round(result["work_units_reformulated_extensive"], digits=2))\t$(round(result["work_units_second_stage"], digits=2))")
        end
    end
    println("✅ K Results file: $kresults_filename")
    
    # 3. OFFLINE COST FILE
    offline_cost_filename = "TS1_S$(num_scenarios)_$(experiment_type)_offline_cost.txt"
    open(offline_cost_filename, "w") do io
        offline_penalty_costs = scenario_info["penalty_costs"]
        offline_costs = scenario_info["offline_costs"]
        
        for scenario_idx in 1:length(offline_costs)
            model_cost = offline_costs[scenario_idx]
            penalty_cost = offline_penalty_costs[scenario_idx]
            println(io, "$instance_name\t$n_buses\t$n_lines\t$n_generators\t$experiment_type\t$scenario_idx\t$(round(model_cost, digits=2))\t$(round(penalty_cost, digits=2))")
        end
    end
    println("✅ Offline Cost file: $offline_cost_filename")
    
    # 4. ONLINE PENALTY FILE
    online_penalty_filename = "TS1_S$(num_scenarios)_$(experiment_type)_online_penalty.txt"
    open(online_penalty_filename, "w") do io
        for K in K_values
            k_str = string(K)
            if haskey(scenario_info, k_str) && haskey(scenario_info[k_str], "penalty_data")
                penalty_data = scenario_info[k_str]["penalty_data"]
                model_cost_per_scenario = scenario_info[k_str]["model_cost_per_scenario"]
                penalty_pos = findall(x -> x > 0.1, penalty_data)
                
                for pos in penalty_pos
                    individual_model_cost = model_cost_per_scenario[pos]
                    penalty_cost = penalty_data[pos]
                    println(io, "$instance_name\t$n_buses\t$n_lines\t$n_generators\t$experiment_type\t$K\t$pos\t$(round(individual_model_cost, digits=2))\t$(round(penalty_cost, digits=2))")
                end
            end
        end
    end
    println("✅ Online Penalty file: $online_penalty_filename")
    
    println("\n📊 Generated 4 separate files with instance metadata in each")
    println("📝 Format: Tab-separated values (TSV) for easy processing")
    
    println("\n🎉 PowerFlow model execution completed successfully!")
    
    return nothing
end

# Run the main function
if abspath(PROGRAM_FILE) == @__FILE__
    # Parse command-line arguments
    num_scenarios, experiment_type = parse_arguments()
    
    # Run the main function with parsed arguments
    main(num_scenarios, experiment_type)
end
