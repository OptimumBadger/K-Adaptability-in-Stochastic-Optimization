# Complete PowerFlow Model - Standalone Julia Script
# Core 3 Phases: Scenario Generation, Policy Selection, Evaluation
# Usage: julia complete_powerflow.jl <S> <experiment_type>
# Example: julia complete_powerflow.jl 100 low

using PowerModels
using JuMP, Gurobi, DataFrames, XLSX, Random, Distributions
using Graphs, GraphPlot, Plots

# Include shared scenarios
include("powerflow_scenarios.jl")

# =============================================================================
# COMMAND-LINE ARGUMENT PARSING
# =============================================================================

function parse_arguments()
    """Parse command-line arguments and return S and experiment_type"""
    if length(ARGS) != 2
        println("❌ Error: Incorrect number of arguments!")
        println("Usage: julia complete_powerflow.jl <S> <experiment_type>")
        println("Example: julia complete_powerflow.jl 100 low")
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
        fbar_ij[parse(Int, branch_id)] = branch["rate_a"] * line_capacity_factor
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

function offline_calc(data, center_data, log_file=nothing)
    """Offline calculations for binary switching optimization"""
    # Initialize the model
    model = Model(Gurobi.Optimizer)
    MOI.set(model, MOI.TimeLimitSec(), 120.0)  # 2 minutes

    # Set logging based on whether we want to save logs
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
    @variable(model, theta[1:length(B)])  # Voltage angle decision variable
    @variable(model, f[1:length(L)])      # Current flow decision variable 
    @variable(model, x[1:length(L)], Bin) # Switching decision variable
    @variable(model, unmet[1:2*length(B)] >= 0)
    Penalty = 5000
    penalty_flag = 0

    n = 2*length(data["bus"])        # Number of rows
    b = zeros(n)
    
    B_ij, f_ij, p_max, p_min, A, c_i = parameters(data; line_capacity_factor=1.0)
    Pd = center_data

    # Create b
    for (bus_id, bus) in B
        bus_index = parse(Int, bus_id)
        b[2*bus_index-1] = p_min[bus_index] - Pd[bus_index]
        b[2*bus_index] = -p_max[bus_index] + Pd[bus_index]
    end

    # Objective Expression
    objective = 0.0
    for (gen_id, generator) in G
        # Generator index
        gen_index = generator["gen_bus"]
        sum_fij = 0.0
        sum_fji = 0.0
        for (branch_id, branch) in L
            branch_index = parse(Int, branch_id)
            if branch["f_bus"] == gen_index
                sum_fij += f[branch_index]
            end
            if branch["t_bus"] == gen_index
                sum_fji += f[branch_index]
            end
        end
        objective += c_i[gen_index]*(Pd[gen_index] + sum_fij - sum_fji)
    end
    objective += Penalty*sum(unmet[i] for i in 1:2*length(B))
    
    # Constraints
    @constraint(model, A * f + unmet .>= b)
    
    for (branch_id, branch) in L
        branch_index = parse(Int, branch_id)
        f_bus = branch["f_bus"]
        t_bus = branch["t_bus"]
        M_ij = -12.6*B_ij[branch_index]
        @constraint(model, -f[branch_index] - f_ij[branch_index]*x[branch_index] <= 0)
        @constraint(model, f[branch_index] - f_ij[branch_index]*x[branch_index] <= 0)
    
        # Linearizing them
        @constraint(model, B_ij[branch_index]*(theta[f_bus] - theta[t_bus]) - M_ij*(1-x[branch_index])-f[branch_index] <= 0)
        @constraint(model, -B_ij[branch_index]*(theta[f_bus] - theta[t_bus]) - M_ij*(1-x[branch_index])+f[branch_index] <= 0)
    end

    # Define the objective function
    @objective(model, Min, objective)
    
    optimize!(model)

    if termination_status(model) == MOI.OPTIMAL
        penalty_cost = Penalty * sum(value.(unmet))
        if sum(value.(unmet)) >= 0.000001
            penalty_flag = 1
        end
        result = value.(x)
        # Round binary variables to exact 0/1 to avoid numerical precision issues
        rounded_x = round.(Int, result)
        # Get Gurobi work units
        work_units = solve_time(model)  # This gives work units in Gurobi
        return objective_value(model), collect(rounded_x), penalty_flag, penalty_cost, work_units
    else
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
            obj, x, p, penalty_cost, work_units = offline_calc(network, ds, log_file)
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
            replacement_solve_time = @elapsed begin
                obj, x, p, penalty_cost, work_units = offline_calc(network, ds, log_file)
            end
            
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
                println("❌ Replacement scenario $scenario_index also failed, will try next scenario")
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

function solve_for_continuous_var(data, sample, solved_x, log_file=nothing)
    """Solve for continuous variables given fixed binary decisions"""
    # Initialize the model
    model = Model(Gurobi.Optimizer)
    MOI.set(model, MOI.TimeLimitSec(), 300.0)

    # Set logging based on whether we want to save logs
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
    @variable(model, theta[1:length(B)])  # Voltage angle decision variable
    @variable(model, f[1:length(L)])      # Current flow decision variable 
    @variable(model, unmet[1:2*length(B)] >= 0)

    # Parameters
    Pd = zeros(length(B)) # Demand values
    B_ij = zeros(length(L)) # Susceptance values
    f_ij = zeros(length(L)) # Upper bound on the load
    c_i = zeros(length(B)) # Linear Cost coefficient with zero intercept
    p_max = zeros(length(B)) # Max power output
    p_min = zeros(length(B)) # Min power output
    Penalty = 5000

    # Select demand parameter
    Pd = sample

    for (branch_id, branch) in L
        B_ij[parse(Int, branch_id)] = -branch["br_x"]/(branch["br_r"]^2 + branch["br_x"]^2)
        f_ij[parse(Int, branch_id)] = branch["rate_a"]
    end

    for (gen_id, generator) in G
        c_i[generator["gen_bus"]] = generator["cost"][1]
        p_max[generator["gen_bus"]] = generator["pmax"]
        p_min[generator["gen_bus"]] = generator["pmin"]
    end

    n = 2*length(data["bus"])        # Number of rows
    m = length(data["branch"])       # Number of columns

    # Initialize matrix
    A = zeros(n, m)
    b = zeros(n)
    
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

    # Create b
    for (bus_id, bus) in B
        bus_index = parse(Int, bus_id)
        b[2*bus_index-1] = p_min[bus_index] - Pd[bus_index] 
        b[2*bus_index] = -p_max[bus_index] + Pd[bus_index]
    end

    # Objective Expression
    objective = 0.0
    for (gen_id, generator) in G
        # Generator index
        gen_index = generator["gen_bus"]
        sum_fij = 0.0
        sum_fji = 0.0
        for (branch_id, branch) in L
            branch_index = parse(Int, branch_id)
            if branch["f_bus"] == gen_index
                sum_fij += f[branch_index]
            end
            if branch["t_bus"] == gen_index
                sum_fji += f[branch_index]
            end
        end
        objective += c_i[gen_index]*(Pd[gen_index] + sum_fij - sum_fji)
    end
    
    objective += Penalty*sum(unmet[i] for i in 1:2*length(B))

    # Constraints
    @constraint(model, A * f + unmet .>= b)

    for (branch_id, branch) in L
        branch_index = parse(Int, branch_id)
        f_bus = branch["f_bus"]
        t_bus = branch["t_bus"]
        M_ij = -12.6*B_ij[branch_index]
        @constraint(model, -f[branch_index] - f_ij[branch_index]*solved_x[branch_index] <= 0)
        @constraint(model, f[branch_index] - f_ij[branch_index]*solved_x[branch_index] <= 0)
    
        # Linearizing them
        @constraint(model, B_ij[branch_index]*(theta[f_bus] - theta[t_bus]) - M_ij*(1-solved_x[branch_index])-f[branch_index] <= 0)
        @constraint(model, -B_ij[branch_index]*(theta[f_bus] - theta[t_bus]) - M_ij*(1-solved_x[branch_index])+f[branch_index] <= 0)
    end

    # Define the objective function
    @objective(model, Min, objective)
    
    # Solve the model
    optimize!(model)

    # Check solver status
    if termination_status(model) == MOI.OPTIMAL
        work_units = solve_time(model)  # Get work units
        return objective_value(model), collect(value.(f)), work_units
    else
        return -1, [], -1
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
            obj_value[i, solution], solved_x[i][solution], work_units = solve_for_continuous_var(data, sample, x_chosen[solution, :], log_file)
            total_work_units += work_units
        end
    end
    return obj_value, solved_x, total_work_units
end

function Reformulated_Extensive_Form(S, l, K, obj_value, log_file=nothing)
    """Solve reformulated extensive form for policy selection"""
    # Initialize model
    model = Model(Gurobi.Optimizer)
    MOI.set(model, MOI.TimeLimitSec(), 60.0)  # 1 minute

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
        obj_value, x_found, flag = offline_calc(data, sample, log_file)
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

function Second_Stage_Cost(data, sample, x_chosen, z, log_file=nothing)
    """Calculate second stage cost for evaluation"""
    # Ensure x_chosen contains exact 0/1 values (round any numerical precision issues)
    x_chosen = [round.(Int, x_chosen[i]) for i in 1:length(x_chosen)]
    
    # Initialize the model
    model = Model(Gurobi.Optimizer)
    MOI.set(model, MOI.TimeLimitSec(), 300.0)

    # Set logging based on whether we want to save logs
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

    n = 2*length(data["bus"])        # Number of rows
    m = length(data["branch"])       # Number of columns

    # Penalty
    Penalty = 5000

    # Decision Variables
    @variable(model, theta[1:length(B)])  # Voltage angle decision variable
    @variable(model, f[1:length(L)])      # Current flow decision variable 
    @variable(model, x[1:length(L)], Bin) # Switching decision variable
    @variable(model, w[1:length(z)], Bin)
    @variable(model, unmet[1:2*length(B)] >= 0)

    # Parameters
    Pd = zeros(length(B)) # Demand values
    B_ij = zeros(length(L)) # Susceptance values
    f_ij = zeros(length(L)) # Upper bound on the load
    c_i = zeros(length(B)) # Linear Cost coefficient with zero intercept
    p_max = zeros(length(B)) # Max power output
    p_min = zeros(length(B)) # Min power output

    # Select demand parameter
    Pd = sample

    for (branch_id, branch) in L
        B_ij[parse(Int, branch_id)] = -branch["br_x"]/(branch["br_r"]^2 + branch["br_x"]^2)
        f_ij[parse(Int, branch_id)] = branch["rate_a"]
    end

    for (gen_id, generator) in G
        c_i[generator["gen_bus"]] = generator["cost"][1]
        p_max[generator["gen_bus"]] = generator["pmax"]
        p_min[generator["gen_bus"]] = generator["pmin"]
    end

    # Initialize matrix
    A = zeros(n, m)
    b = zeros(n)
    
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

    # Create b
    for (bus_id, bus) in B
        bus_index = parse(Int, bus_id)
        b[2*bus_index-1] = p_min[bus_index] - Pd[bus_index]
        b[2*bus_index] = -p_max[bus_index] + Pd[bus_index]
    end

    # Objective Expression
    objective = 0.0
    for (gen_id, generator) in G
        # Generator index
        gen_index = generator["gen_bus"]
        sum_fij = 0.0
        sum_fji = 0.0
        for (branch_id, branch) in L
            branch_index = parse(Int, branch_id)
            if branch["f_bus"] == gen_index
                sum_fij += f[branch_index]
            end
            if branch["t_bus"] == gen_index
                sum_fji += f[branch_index]
            end
        end
        objective += c_i[gen_index]*(Pd[gen_index] + sum_fij - sum_fji)
    end

    objective += Penalty*sum(unmet[i] for i in 1:2*length(B))

    # Constraints
    @constraint(model, A * f + unmet .>= b)

    for (branch_id, branch) in L
        branch_index = parse(Int, branch_id)
        f_bus = branch["f_bus"]
        t_bus = branch["t_bus"]
        M_ij = -12.6*B_ij[branch_index]
        @constraint(model, -f[branch_index] - f_ij[branch_index]*x[branch_index] <= 0)
        @constraint(model, f[branch_index] - f_ij[branch_index]*x[branch_index] <= 0)
    
        # Linearizing them
        @constraint(model, B_ij[branch_index]*(theta[f_bus] - theta[t_bus]) - M_ij*(1-x[branch_index])-f[branch_index] <= 0)
        @constraint(model, -B_ij[branch_index]*(theta[f_bus] - theta[t_bus]) - M_ij*(1-x[branch_index])+f[branch_index] <= 0)
    end

    @constraint(model, [i in 1:length(z)], w[i] <= z[i])
    @constraint(model, sum(w[i] for i in 1:length(z)) == 1)
    @constraint(model, [j in 1:length(x)], x[j] == sum(w[i] * x_chosen[i][j] for i in 1:length(z)))

    # Define the objective function
    @objective(model, Min, objective)

    # Solve the model
    optimize!(model)

    # Check solver status
    status = termination_status(model)
    if status == MOI.OPTIMAL
        work_units = solve_time(model)  # Get work units
        return objective_value(model), value.(w), Penalty*(sum(value.(unmet))), work_units
    else
        # Debug infeasibility
        println("❌ Second_Stage_Cost failed with status: $status")
        println("   Model has $(num_variables(model)) variables and $(num_constraints(model; count_variable_in_set_constraints=true)) constraints")
        
        if status == MOI.INFEASIBLE || status == MOI.INFEASIBLE_OR_UNBOUNDED
            status_name = status == MOI.INFEASIBLE ? "INFEASIBLE" : "INFEASIBLE_OR_UNBOUNDED"
            println("   Model is $status_name - attempting to compute conflicting constraints...")
            try
                # Try to compute an Irreducible Inconsistent Subsystem (IIS)
                compute_conflict!(model)
                println("   IIS computed - displaying conflicting constraints:")
                
                # Print conflicting constraints using the working approach from PowerFlow Model.ipynb
                println("   Identified infeasible constraints:")
                for (F, S) in list_of_constraint_types(model)
                    for (i, constr) in enumerate(all_constraints(model, F, S))
                        status = MOI.get(model, MOI.ConstraintConflictStatus(), constr)
                        if status == MOI.IN_CONFLICT
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
            println("     Binary variables: 189")
            println("     Continuous variables: $(num_variables(model) - 189)")
            println("     Constraint breakdown:")
            println("       - Equality constraints: 1")
            println("       - Greater-than constraints: 236") 
            println("       - Less-than constraints: 747")
            println("       - Variable bounds: 236")
        elseif status == MOI.TIME_LIMIT
            println("   Model hit TIME_LIMIT")
        else
            println("   Other termination status: $status")
        end
        
        return -1, [], -1, -1
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
            model_obj_value, candidate_index, penalty_cost, work_units_second_stage = Second_Stage_Cost(data, sample, x_chosen, z, log_file)
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

function run_single_experiment(network, S, K_values, capacity_factor, line_capacity_factor, sigma_low, sigma_high, use_bernoulli, file_path_excel)
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
        generate_scenarios_for_count(nscen, file_path_excel, sigma_low, sigma_high, use_bernoulli)
    
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
    println("POWERFLOW MODEL - CORE 3 PHASES")
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
    file_path = "powerflow_instances/case118Blumsack.m"
    file_path_excel = "powerflow_instances/Random_Demands.xlsx"
    
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
    capacity_factor = 2.0
    line_capacity_factor = 10.0
    println("Generator capacity factor set to: $capacity_factor")
    println("Line capacity factor set to: $line_capacity_factor")
    
    # Reset random seed
    Random.seed!(1234)
    
    # =============================================================================
    # RUN SINGLE EXPERIMENT
    # =============================================================================
    
    # Run the entire algorithm for this experiment
    experiment_results = run_single_experiment(network, [num_scenarios], K_values, capacity_factor, line_capacity_factor, sigma_low, sigma_high, use_bernoulli, file_path_excel)
    
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
