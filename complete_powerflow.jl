# Complete PowerFlow Model - Standalone Julia Script
# Core 3 Phases: Scenario Generation, Policy Selection, Evaluation

using PowerModels
using JuMP, Gurobi, DataFrames, XLSX, Random, Distributions
using Graphs, GraphPlot, Plots

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

function parameters(network)
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
        fbar_ij[parse(Int, branch_id)] = branch["rate_a"]
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
    MOI.set(model, MOI.TimeLimitSec(), 60.0)

    # Set logging based on whether we want to save logs
    if log_file !== nothing
        set_optimizer_attribute(model, "LogFile", log_file)
        set_optimizer_attribute(model, "LogToConsole", 0)
    else
        set_optimizer_attribute(model, "OutputFlag", 1)
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
    Penalty = 10000
    penalty_flag = 0

    n = 2*length(data["bus"])        # Number of rows
    b = zeros(n)
    
    B_ij, f_ij, p_max, p_min, A, c_i = parameters(data)
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
        if sum(value.(unmet)) >= 0.000001
            penalty_flag = 1
        end
        result = value.(x)
        return objective_value(model), collect(value.(x)), penalty_flag
    else
        return -1, [], -1
    end
end

function generate_scenarios(demand_data, numscenarios)
    """Generate demand scenarios"""
    samples = zeros(numscenarios, length(demand_data))
    
    for scenario in 1:numscenarios
        global_factor = rand(Uniform(0.8, 1.2)) # Global factor
        for i in 1:length(demand_data)
            center = demand_data[i]
            if center == 0
                local_factor_i = rand(Uniform(0, 1))
            else
                local_factor_i = 1 + rand(Uniform(-0.1, 0.1))  # ±10% local
            end
            bus_demand_i = demand_data[i] * 1 * local_factor_i
            samples[scenario, i] = bus_demand_i
        end
    end
    return samples
end

function sampling_for_offline_calc(network, demand_data, total_scenarios; capacity_factor=1.20, log_file=nothing)
    """Generate feasible scenarios for offline calculations"""
    # 1) unpack network parameters
    B_ij, fbar_ij, p_max_base, p_min_base, A, c = parameters(network)

    # index sets
    buses = sort(parse.(Int, collect(keys(network["bus"]))))
    branches = sort(parse.(Int, collect(keys(network["branch"]))))
    nb = length(buses)
    nl = length(branches)

    # 2) pre-allocate storage
    demand_samples = zeros(Float64, total_scenarios, nb)
    x_solutions = zeros(total_scenarios, nl)

    # 3) sampling loop
    found = 0
    while found < total_scenarios
        # generate one draw of demand & cost
        ds = generate_scenarios(demand_data, 1)

        # make *local* scaled copies of your capacities
        p_max = p_max_base .* capacity_factor
        p_min = p_min_base .* capacity_factor

        # try to solve
        obj, x, p = offline_calc(network, generate_scenarios(demand_data, 1), log_file)

        # if we get here, it solved to optimality
        if obj != -1
            found += 1
            demand_samples[found, :] = ds
            x_solutions[found, :] = x
        else
            println("Scenario is infeasible, trying again")
            # just loop and sample again
        end
    end

    return demand_samples, x_solutions
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
        set_optimizer_attribute(model, "OutputFlag", 1)
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
    Penalty = 10000

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
        return objective_value(model), collect(value.(f))
    else
        return -1, []
    end
end

function Calculations_For_Extensive_Form(data, online_samples, S, l, x_chosen, log_file=nothing)
    """Calculate cost matrix for extensive form"""
    L = data["branch"]
    solved_x = [Vector{Float64}[] for _ in 1:S]
    obj_value = zeros(S, l)
    
    for (i, sample) in zip(1:S, eachrow(online_samples))
        solved_x[i] = [zeros(length(L)) for _ in 1:l]
        for solution in 1:l
            obj_value[i, solution], solved_x[i][solution] = solve_for_continuous_var(data, sample, x_chosen[solution, :], log_file)
        end
    end
    return obj_value, solved_x
end

function Reformulated_Extensive_Form(S, l, K, obj_value, log_file=nothing)
    """Solve reformulated extensive form for policy selection"""
    # Initialize model
    model = Model(Gurobi.Optimizer)
    MOI.set(model, MOI.TimeLimitSec(), 3600.0)

    # Set logging based on whether we want to save logs
    if log_file !== nothing
        set_optimizer_attribute(model, "LogFile", log_file)
        set_optimizer_attribute(model, "LogToConsole", 0)
    else
        set_optimizer_attribute(model, "OutputFlag", 1)
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
        return objective_value(model), value.(z), solve_time(model)
    else
        return -1, [], -1
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
    # Initialize the model
    model = Model(Gurobi.Optimizer)
    MOI.set(model, MOI.TimeLimitSec(), 300.0)

    # Set logging based on whether we want to save logs
    if log_file !== nothing
        set_optimizer_attribute(model, "LogFile", log_file)
        set_optimizer_attribute(model, "LogToConsole", 0)
    else
        set_optimizer_attribute(model, "OutputFlag", 1)
    end

    # Sets
    B = data["bus"]
    L = data["branch"]
    G = data["gen"]

    n = 2*length(data["bus"])        # Number of rows
    m = length(data["branch"])       # Number of columns

    # Penalty
    Penalty = 10000

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
    @constraint(model, x == sum(w[i] * x_chosen[i, :] for i in 1:length(z)))

    # Define the objective function
    @objective(model, Min, objective)

    # Solve the model
    optimize!(model)

    # Check solver status
    if termination_status(model) == MOI.OPTIMAL
        return objective_value(model), value.(w), Penalty*(sum(value.(unmet)))
    else
        return -1, [], -1
    end
end

function Evaluation(data, test_sample, offline_cost_list, x_chosen, z, log_file=nothing)
    """Evaluate model performance"""
    num_rows = size(test_sample, 1)
    cost = zeros(num_rows)
    penalty = zeros(num_rows)

    online_list = zeros(num_rows)

    # Second stage cost
    model_cost = 0.0
    count = 0
    for (index, sample) in zip(1:num_rows, eachrow(test_sample))
        if offline_cost_list[index] != -1
            model_obj_value, candidate_index, penalty_cost = Second_Stage_Cost(data, sample, x_chosen, z, log_file)
            online_list[index] = model_obj_value
            penalty[index] = penalty_cost
            if model_obj_value != -1
                model_cost += model_obj_value
                count += 1
            end
        else
            online_list[index] = -1
        end
    end
    total_model_cost = model_cost / count

    return total_model_cost, online_list, penalty
end

# =============================================================================
# MAIN FUNCTION
# =============================================================================

function main()
    println("="^80)
    println("POWERFLOW MODEL - CORE 3 PHASES")
    println("="^80)
    
    # =============================================================================
    # DATA LOADING
    # =============================================================================
    
    println("\n" * "="^40)
    println("DATA LOADING")
    println("="^40)
    
    # File paths (using relative paths for portability)
    file_path = "powerflow instances/case118Blumsack.m"
    file_path_excel = "powerflow instances/Random Demands.xlsx"
    
    # Check if files exist
    if !isfile(file_path)
        println("❌ Error: MATPOWER file not found!")
        println("Expected file: $file_path")
        println("Please make sure you're running from the project root directory")
        println("and that the powerflow instances/ folder contains the data files.")
        return
    end
    
    if !isfile(file_path_excel)
        println("❌ Error: Excel file not found!")
        println("Expected file: $file_path_excel")
        println("Please make sure you're running from the project root directory")
        println("and that the powerflow instances/ folder contains the data files.")
        return
    end
    
    println("Loading data from: $file_path")
    println("Loading parameters from: $file_path_excel")
    
    # Parse the MATPOWER case file
    network = PowerModels.parse_file(file_path)
    
    # Read the Excel data containing random parameters 
    param_data = XLSX.readtable(file_path_excel, "Sheet1") |> DataFrame
    
    # Extract center data from Excel file
    center_data = Vector{Float64}(param_data[1, :])  # First row contains the demand values
    
    println("✅ Data loaded successfully!")
    println("Network: $(length(network["bus"])) buses, $(length(network["branch"])) branches, $(length(network["gen"])) generators")
    
    # =============================================================================
    # PARAMETERS
    # =============================================================================
    
    # Model parameters
    nscen_list = [20, 30, 50]  # Number of scenarios for sampling (will loop through these)
    nsamples = 100             # Number of online samples
    test_samples = 50          # Number of test samples
    K_values = [1, 2, 3, 4, 5, 6, 7]  # Number of policies to select (will loop through these)
    capacity_factor = 1.20     # Capacity scaling factor
    
    println("\nModel Parameters:")
    println("  nscen values: $nscen_list")
    println("  Online samples: $nsamples")
    println("  Test samples: $test_samples")
    println("  K values (policies to select): $K_values")
    println("  Capacity factor: $capacity_factor")
    
    # =============================================================================
    # LOGGING SETUP
    # =============================================================================
    
    # Create log directory structure
    log_base_dir = "Logs/PowerFlow Logs"
    log_common_dir = "$log_base_dir/common"
    
    # Create common directory
    mkpath(log_common_dir)
    
    # Define common log file paths (used once across all nscen)
    log_offline_test = "$log_common_dir/offline_test.log"
    timing_common_file = "$log_common_dir/timing_summary_common.txt"
    
    println("\n📁 Log files will be saved to: $log_base_dir")
    println("📁 Common logs: $log_common_dir")
    
    # =============================================================================
    # ULTRA-COMMON PHASES (RUN ONCE ACROSS ALL NSCEN)
    # =============================================================================
    
    println("\n" * "="^60)
    println("ULTRA-COMMON PHASES: ONLINE SAMPLES, TEST SAMPLES & OFFLINE TEST")
    println("="^60)
    
    # Generate online samples (common across all nscen)
    println("Generating $nsamples online samples...")
    online_samples = generate_scenarios(center_data, nsamples)
    
    # Generate test samples (common across all nscen)
    println("Generating $test_samples test samples...")
    test_sample = generate_scenarios(center_data, test_samples)
    
    # Run offline test (common across all nscen)
    println("Starting Offline Test...")
    offline_test_time = @elapsed begin
        original_cost, offline_cost_list, penalty_blist = OfflineTest(network, test_sample, log_offline_test)
    end
    println("Average offline cost: $original_cost")
    println("✅ Ultra-common phases completed in $(round(offline_test_time, digits=2)) seconds")
    
    # =============================================================================
    # NSCEN-SPECIFIC AND K-SPECIFIC PHASES (LOOP THROUGH NSCEN VALUES)
    # =============================================================================
    
    println("\n" * "="^60)
    println("NSCEN-SPECIFIC AND K-SPECIFIC PHASES")
    println("="^60)
    
    # Store results for all nscen and K combinations
    all_results = Dict()
    
    for (nscen_idx, nscen) in enumerate(nscen_list)
        println("\n" * "="^50)
        println("PROCESSING NSCEN = $nscen ($(nscen_idx)/$(length(nscen_list)))")
        println("="^50)
        
        # Create nscen-specific log directory structure
        log_nscen_dir = "$log_base_dir/L$(nscen)"
        mkpath("$log_nscen_dir/phase1")
        mkpath("$log_nscen_dir/phase2")
        mkpath("$log_nscen_dir/phase3")
        
        # Define nscen-specific log file paths
        log_phase1 = "$log_nscen_dir/phase1/scenario_generation.log"
        log_phase2a = "$log_nscen_dir/phase2/decoupling_calculations.log"
        timing_nscen_file = "$log_nscen_dir/timing_summary_L$(nscen).txt"
        
        # =============================================================================
        # PHASE 1: SCENARIO GENERATION (NSCEN-SPECIFIC)
        # =============================================================================
        
        println("Starting Phase 1: Scenario Generation for nscen=$nscen...")
        phase1_time = @elapsed begin
            demand_samples, x_solutions = sampling_for_offline_calc(network, center_data, nscen; capacity_factor=capacity_factor, log_file=log_phase1)
        end
        
        println("Generated $nscen scenarios with $(size(x_solutions, 1)) solutions")
        
        # =============================================================================
        # PHASE 2A: DECOUPLING CALCULATIONS (NSCEN-SPECIFIC)
        # =============================================================================
        
        println("Starting Phase 2A: Decoupling Calculations for nscen=$nscen...")
        phase2_calc_time = @elapsed begin
            obj_value, solved_x = Calculations_For_Extensive_Form(network, online_samples, nsamples, nscen, x_solutions, log_phase2a)
        end
        
        n_neg1 = count(x -> x == -1, obj_value)
        println("Infeasible scenario-solution pairs: $n_neg1")
        
        # Store results for this nscen
        all_results[nscen] = Dict()
        
        # =============================================================================
        # K-SPECIFIC PHASES (LOOP THROUGH K VALUES FOR THIS NSCEN)
        # =============================================================================
        
        for (k_idx, K) in enumerate(K_values)
            println("\n" * "="^40)
            println("PROCESSING K = $K for nscen=$nscen ($(k_idx)/$(length(K_values)))")
            println("="^40)
            
            # Define K-specific log file paths
            log_phase2b = "$log_nscen_dir/phase2/reformulated_extensive_form_K$(K).log"
            log_phase3b = "$log_nscen_dir/phase3/evaluation_K$(K).log"
            timing_k_file = "$log_nscen_dir/timing_summary_K$(K).txt"
            
            # =============================================================================
            # PHASE 2B: REFORMULATED EXTENSIVE FORM (K-SPECIFIC)
            # =============================================================================
            
            println("Starting Phase 2B: Reformulated Extensive Form for K=$K...")
            phase2_reform_time = @elapsed begin
                Solution, z, time_taken = Reformulated_Extensive_Form(nsamples, nscen, K, obj_value, log_phase2b)
            end
            
            println("Reformulated extensive form objective for K=$K: $Solution")
            
            # =============================================================================
            # PHASE 3B: EVALUATION (K-SPECIFIC)
            # =============================================================================
            
            println("Starting Phase 3B: Evaluation for K=$K...")
            phase3_eval_time = @elapsed begin
                model_cost, model_cost_list, penlist = Evaluation(network, test_sample, offline_cost_list, x_solutions, z, log_phase3b)
            end
            
            # Calculate gap
            gap = (model_cost - original_cost) / original_cost * 100
            
            # Analyze penalties
            penalty_pos = findall(x -> x > 0.1, penlist)
            
            println("Results for nscen=$nscen, K=$K:")
            println("  Model cost: $(round(model_cost, digits=2))")
            println("  Performance gap: $(round(gap, digits=2))%")
            println("  Scenarios with penalties: $(length(penalty_pos))")
            println("  Phase 2B time: $(round(phase2_reform_time, digits=2)) seconds")
            println("  Phase 3B time: $(round(phase3_eval_time, digits=2)) seconds")
            
            # Store results for this K
            all_results[nscen][K] = Dict(
                "solution" => Solution,
                "model_cost" => model_cost,
                "gap" => gap,
                "penalties" => length(penalty_pos),
                "phase2b_time" => phase2_reform_time,
                "phase3b_time" => phase3_eval_time
            )
            
            # Write K-specific timing summary
            open(timing_k_file, "w") do io
                println(io, "POWERFLOW MODEL TIMING SUMMARY - K=$K")
                println(io, "="^50)
                println(io, "Parameters:")
                println(io, "  nscen: $nscen")
                println(io, "  K: $K")
                println(io, "  nsamples: $nsamples")
                println(io, "  test_samples: $test_samples")
                println(io, "  capacity_factor: $capacity_factor")
                println(io, "")
                println(io, "Execution Times:")
                println(io, "  Phase 2B (Reformulated Ext Form):  $(round(phase2_reform_time, digits=2)) seconds")
                println(io, "  Phase 3B (Evaluation):             $(round(phase3_eval_time, digits=2)) seconds")
                println(io, "  Total K-specific Time:             $(round(phase2_reform_time + phase3_eval_time, digits=2)) seconds")
                println(io, "")
                println(io, "Results:")
                println(io, "  Offline optimal cost:     $(round(original_cost, digits=2))")
                println(io, "  Model cost:               $(round(model_cost, digits=2))")
                println(io, "  Model gap:                $(round(gap, digits=2))%")
                println(io, "  Scenarios with penalties: $(length(penalty_pos))")
            end
            
            println("✅ nscen=$nscen, K=$K completed successfully!")
        end
        
        # Write nscen-specific timing summary
        open(timing_nscen_file, "w") do io
            println(io, "POWERFLOW MODEL TIMING SUMMARY - NSCEN=$nscen")
            println(io, "="^50)
            println(io, "Parameters:")
            println(io, "  nscen: $nscen")
            println(io, "  K_values: $K_values")
            println(io, "  nsamples: $nsamples")
            println(io, "  test_samples: $test_samples")
            println(io, "  capacity_factor: $capacity_factor")
            println(io, "")
            println(io, "Execution Times:")
            println(io, "  Phase 1 (Scenario Generation):     $(round(phase1_time, digits=2)) seconds")
            println(io, "  Phase 2A (Decoupling Calc):        $(round(phase2_calc_time, digits=2)) seconds")
            println(io, "")
            println(io, "Results by K:")
            for K in K_values
                result = all_results[nscen][K]
                println(io, "  K=$K:")
                println(io, "    Model cost:               $(round(result["model_cost"], digits=2))")
                println(io, "    Model gap:                $(round(result["gap"], digits=2))%")
                println(io, "    Scenarios with penalties: $(result["penalties"])")
                println(io, "    Phase 2B time:            $(round(result["phase2b_time"], digits=2)) seconds")
                println(io, "    Phase 3B time:            $(round(result["phase3b_time"], digits=2)) seconds")
            end
        end
        
        println("✅ nscen=$nscen completed successfully!")
    end
    
    
    # =============================================================================
    # RESULTS SUMMARY
    # =============================================================================
    
    println("\n" * "="^60)
    println("FINAL RESULTS SUMMARY")
    println("="^60)
    println("Offline optimal cost:     $(round(original_cost, digits=2))")
    println("")
    println("Results by nscen and K:")
    for nscen in nscen_list
        println("  nscen=$nscen:")
        for K in K_values
            result = all_results[nscen][K]
            println("    K=$K:")
            println("      Model cost:               $(round(result["model_cost"], digits=2))")
            println("      Model gap:                $(round(result["gap"], digits=2))%")
            println("      Scenarios with penalties: $(result["penalties"])")
        end
    end
    println("="^60)
    
    # Print timing summary to console
    println("\n" * "="^60)
    println("TIMING SUMMARY")
    println("="^60)
    println("Ultra-common phases (Offline Test): $(round(offline_test_time, digits=2)) seconds")
    println("")
    println("Breakdown by nscen:")
    for nscen in nscen_list
        # Calculate total time for this nscen (we need to get these from the loop)
        # For now, just show the structure
        println("  nscen=$nscen: [Phase 1 + Phase 2A + all K-specific phases]")
    end
    println("="^60)
    
    # =============================================================================
    # WRITE COMMON TIMING SUMMARY TO FILE
    # =============================================================================
    
    open(timing_common_file, "w") do io
        println(io, "POWERFLOW MODEL TIMING SUMMARY - COMMON PHASES")
        println(io, "="^50)
        println(io, "Parameters:")
        println(io, "  nscen_list: $nscen_list")
        println(io, "  nsamples: $nsamples")
        println(io, "  test_samples: $test_samples")
        println(io, "  K_values: $K_values")
        println(io, "  capacity_factor: $capacity_factor")
        println(io, "")
        println(io, "Ultra-Common Execution Times:")
        println(io, "  Offline Test:               $(round(offline_test_time, digits=2)) seconds")
        println(io, "")
        println(io, "Results:")
        println(io, "  Offline optimal cost:       $(round(original_cost, digits=2))")
        println(io, "")
        println(io, "Note: This file contains timing for phases that run once")
        println(io, "across all nscen values. See individual L{nscen} directories")
        println(io, "for nscen-specific and K-specific timing details.")
    end
    
    # =============================================================================
    # APPEND SUMMARY STATISTICS TO LOG FILES
    # =============================================================================
    
    # Common offline test summary
    offline_test_params = Dict("test_samples" => test_samples)
    offline_test_results = Dict("average_offline_cost" => round(original_cost, digits=2), "execution_time" => "$(round(offline_test_time, digits=2)) seconds")
    append_summary_to_log(log_offline_test, "Ultra-Common - Offline Test", offline_test_params, offline_test_results)
    
    # nscen-specific and K-specific summaries
    for nscen in nscen_list
        log_nscen_dir = "$log_base_dir/L$(nscen)"
        
        # Phase 1 summary for this nscen
        log_phase1 = "$log_nscen_dir/phase1/scenario_generation.log"
        phase1_params = Dict("nscen" => nscen, "capacity_factor" => capacity_factor)
        phase1_results = Dict("scenarios_generated" => nscen, "execution_time" => "See timing files for details")
        append_summary_to_log(log_phase1, "Phase 1 - Scenario Generation", phase1_params, phase1_results)
        
        # Phase 2A summary for this nscen
        log_phase2a = "$log_nscen_dir/phase2/decoupling_calculations.log"
        phase2a_params = Dict("nsamples" => nsamples, "nscen" => nscen)
        phase2a_results = Dict("execution_time" => "See timing files for details")
        append_summary_to_log(log_phase2a, "Phase 2A - Decoupling Calculations", phase2a_params, phase2a_results)
        
        # K-specific summaries for this nscen
        for K in K_values
            result = all_results[nscen][K]
            
            # Phase 2B summary for this K
            log_phase2b = "$log_nscen_dir/phase2/reformulated_extensive_form_K$(K).log"
            phase2b_params = Dict("nsamples" => nsamples, "nscen" => nscen, "K" => K)
            phase2b_results = Dict("objective_value" => result["solution"], "execution_time" => "$(round(result["phase2b_time"], digits=2)) seconds")
            append_summary_to_log(log_phase2b, "Phase 2B - Reformulated Extensive Form", phase2b_params, phase2b_results)
            
            # Phase 3B summary for this K
            log_phase3b = "$log_nscen_dir/phase3/evaluation_K$(K).log"
            phase3b_params = Dict("test_samples" => test_samples, "K" => K)
            phase3b_results = Dict("average_model_cost" => round(result["model_cost"], digits=2), "performance_gap" => "$(round(result["gap"], digits=2))%", "scenarios_with_penalties" => result["penalties"], "execution_time" => "$(round(result["phase3b_time"], digits=2)) seconds")
            append_summary_to_log(log_phase3b, "Phase 3B - Evaluation", phase3b_params, phase3b_results)
        end
    end
    
    println("\n📝 Log files and timing summary saved successfully!")
    println("\n🎉 PowerFlow model execution completed successfully!")
    
    return nothing
end

# Run the main function
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
