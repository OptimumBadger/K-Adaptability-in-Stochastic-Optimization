#!/usr/bin/env julia

"""
Complete Facility Location Problem - Standalone Script (FL1)
This script runs the entire facility location optimization pipeline automatically.
Configured for ORLIB format files (e.g., cap91.txt)
"""

using JuMP, Gurobi, Random, Distributions, DataFrames, LinearAlgebra, XLSX, Dates

# Set random seed for reproducibility
Random.seed!(1234)

# =============================================================================
# DATA READING AND PREPROCESSING
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

# =============================================================================
# CORE OPTIMIZATION FUNCTIONS
# =============================================================================

function facility_location(nloc, ncust, capacity, cost, demand, fixedcost, unmet_pen, scaling_factor, log_file=nothing)
    """Solve the basic facility location problem"""
    model = Model(Gurobi.Optimizer)
    
    # Set logging based on whether we want to save logs
    if log_file !== nothing
        set_optimizer_attribute(model, "LogFile", log_file)
        set_optimizer_attribute(model, "LogToConsole", 0)  # Don't show in console
    else
        # Enable console logging for offline phase debugging
        set_optimizer_attribute(model, "OutputFlag", 0)  # Show in console
    end
    
    # Decision Variables
    @variable(model, x[1:nloc], Bin)
    @variable(model, y[1:nloc, 1:ncust] >= 0)
    @variable(model, shortfall[1:ncust] >= 0)

    # Constraints
    @constraint(model, [j in 1:ncust], sum(y[i,j] for i in 1:nloc) + shortfall[j] == demand[j])
    @constraint(model, [i in 1:nloc], sum(y[i,j] for j in 1:ncust) - capacity[i]*x[i] <= 0)

    # Objective function
    @objective(model, Min, 
        sum(fixedcost[i]*x[i] for i in 1:nloc) + 
        scaling_factor * sum(cost[i,j]*y[i,j] for i in 1:nloc for j in 1:ncust) + 
        sum(unmet_pen*shortfall[j] for j in 1:ncust))

    optimize!(model)
    
    # Get work units
    work_units = MOI.get(backend(model), Gurobi.ModelAttribute("Work"))
    
    # Calculate cost breakdown
    x_solution = round.(Int, collect(value.(x)))
    y_solution = value.(y)
    shortfall_solution = value.(shortfall)
    
    fixed_cost = sum(fixedcost[i] * x_solution[i] for i in 1:nloc)
    transportation_cost = scaling_factor * sum(cost[i,j] * y_solution[i,j] for i in 1:nloc for j in 1:ncust)
    penalty_cost = sum(unmet_pen * shortfall_solution[j] for j in 1:ncust)
    total_cost = fixed_cost + transportation_cost + penalty_cost
    
    return total_cost, x_solution, (fixed_cost, transportation_cost, penalty_cost), work_units
end

function solve_for_continuous_var(nloc, ncust, capacity, cost, demand, solved_x, fixedcost, unmet_pen, scaling_factor, log_file=nothing)
    """Solve for continuous variables with fixed binary decisions"""
    model = Model(Gurobi.Optimizer)
    
    # Set logging based on whether we want to save logs
    if log_file !== nothing
        set_optimizer_attribute(model, "LogFile", log_file)
        set_optimizer_attribute(model, "LogToConsole", 0)  # Don't show in console
    else
        set_optimizer_attribute(model, "OutputFlag", 0)  # Don't show in console
    end
    
    # Decision Variables
    @variable(model, x[1:nloc], Bin)
    @variable(model, y[1:nloc, 1:ncust] >= 0)
    @variable(model, shortfall[1:ncust] >= 0)

    # Constraints
    @constraint(model, [j in 1:ncust], sum(y[i,j] for i in 1:nloc) + shortfall[j] == demand[j])
    @constraint(model, [i in 1:nloc], sum(y[i,j] for j in 1:ncust) - capacity[i]*x[i] <= 0)
    @constraint(model, [i in 1:nloc], x[i] == solved_x[i])  # fixing the binary decision variable

    # Objective function
    @objective(model, Min, 
        sum(fixedcost[i]*x[i] for i in 1:nloc) + 
        scaling_factor * sum(cost[i,j]*y[i,j] for i in 1:nloc for j in 1:ncust) + 
        sum(unmet_pen*shortfall[j] for j in 1:ncust))

    optimize!(model)
    
    # Get work units
    work_units = MOI.get(backend(model), Gurobi.ModelAttribute("Work"))
    
    if termination_status(model) == MOI.OPTIMAL
        return objective_value(model), sum(fixedcost[i]*value.(x[i]) for i in 1:nloc)/objective_value(model), work_units
    else
        return -1, -1, work_units
    end
end

# =============================================================================
# SAMPLING AND SCENARIO GENERATION
# =============================================================================

function generate_demand_scenario(ncust, demandmean, demandstdev=nothing, bernoulli_case=false, method="new", experiment_type="low")
    """Generate a single demand scenario using PowerFlow-style formula"""
    
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

function sampling(nscen, nloc, demandmean, capacity, cost, fixedcost, unmet_pen, scaling_factor, demandstdev=nothing, bernoulli_case=false, method="new", experiment_type="low", log_file=nothing)
    """Generate multiple scenarios and solve for each"""
    ncust = length(demandmean)
    xsolutions = zeros(Int, nscen, nloc)
    cost_breakdowns = []
    total_work_units = 0.0
    
    println("Generating $nscen scenarios using $method method...")
    for scenario in 1:nscen
        if scenario % 100 == 0
            println("  Processed $scenario scenarios")
        end
        demand = generate_demand_scenario(ncust, demandmean, demandstdev, bernoulli_case, method, experiment_type)
        _, xsolutions[scenario, :], breakdown, work_units = facility_location(nloc, ncust, capacity, cost, demand, fixedcost, unmet_pen, scaling_factor, log_file)
        push!(cost_breakdowns, breakdown)
        total_work_units += work_units
    end
    return xsolutions, cost_breakdowns, total_work_units
end

# =============================================================================
# REFORMULATED EXTENSIVE FORM
# =============================================================================

function Calculations_For_Extensive_Form(nloc, ncust, capacity, cost, online_samples, xsolutions, fixedcost, unmet_pen, scaling_factor, log_file=nothing)
    """Calculate objective values for all scenario-solution pairs"""
    total_scenarios = length(online_samples)
    total_solutions = size(xsolutions, 1)
    obj_value = zeros(total_scenarios, total_solutions)
    ratio = zeros(total_scenarios, total_solutions)
    total_work_units = 0.0
    
    println("Computing extensive form matrix...")
    for scenario in 1:total_scenarios
        if scenario % 20 == 0
            println("  Processed $scenario scenarios")
        end
        demand = online_samples[scenario]
        for solution in 1:total_solutions
            obj_value[scenario, solution], ratio[scenario, solution], work_units = solve_for_continuous_var(
                nloc, ncust, capacity, cost, demand, xsolutions[solution, :], 
                fixedcost, unmet_pen, scaling_factor, log_file)
            total_work_units += work_units
        end
    end
    return obj_value, ratio, total_work_units
end

function Reformulated_Extensive_Form(S, l, K, obj_value, log_file=nothing)
    """Solve the reformulated extensive form to select K optimal solutions"""
    model = Model(Gurobi.Optimizer)
    MOI.set(model, MOI.TimeLimitSec(), 3600.0)
    
    # Set logging based on whether we want to save logs
    if log_file !== nothing
        set_optimizer_attribute(model, "LogFile", log_file)
        set_optimizer_attribute(model, "LogToConsole", 0)  # Don't show in console
    else
        set_optimizer_attribute(model, "OutputFlag", 0)  # Don't show in console
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
# L-AUGMENTATION HEURISTIC
# =============================================================================

function n_unique_rows(A::AbstractMatrix)
    """Count unique rows in a matrix"""
    rows_as_tuples = (Tuple(view(A, i, :)) for i in axes(A, 1))
    uniq = unique(rows_as_tuples)
    return uniq, length(uniq)
end


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

function Second_Stage_Cost(nloc, ncust, cost, capacity, demand, xsolutions, z, fixedcost, unmet_pen, scaling_factor, log_file=nothing)
    """Calculate second stage cost for given solution"""
    model = Model(Gurobi.Optimizer)
    
    # Set logging based on whether we want to save logs
    if log_file !== nothing
        set_optimizer_attribute(model, "LogFile", log_file)
        set_optimizer_attribute(model, "LogToConsole", 0)  # Don't show in console
    else
        set_optimizer_attribute(model, "OutputFlag", 0)  # Don't show in console
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
    work_units = MOI.get(backend(model), Gurobi.ModelAttribute("Work"))
    
    if termination_status(model) == MOI.OPTIMAL
        return objective_value(model), sum(unmet_pen*value.(shortfall[j]) for j in 1:ncust), work_units
    else
        return -1, -1, work_units
    end
end

function Evaluation(nloc, ncust, cost, capacity, test_sample, x_solutions, z, fixedcost, unmet_pen, scaling_factor, log_file=nothing)
    """Evaluate solution performance on test data"""
    num_rows = size(test_sample, 1)
    p_cost = zeros(num_rows)
    m_cost = zeros(num_rows)
    
    model_cost = 0.0
    total_work_units = 0.0
    for index in 1:num_rows
        model_obj_value, p, work_units = Second_Stage_Cost(nloc, ncust, cost, capacity, test_sample[index], x_solutions, z, fixedcost, unmet_pen, scaling_factor, log_file)
        model_cost += model_obj_value
        p_cost[index] = p
        m_cost[index] = model_obj_value
        total_work_units += work_units
    end
    
    total_model_cost = model_cost / num_rows
    return total_model_cost, p_cost, m_cost, total_work_units
end

function OfflineTest(nloc, ncust, capacity, cost, test_samples, fixedcost, unmet_pen, scaling_factor, log_file=nothing)
    """Calculate offline optimal costs for test data"""
    num_rows = size(test_samples, 1)
    offline_cost = zeros(num_rows)
    
    println("Computing offline optimal costs...")
    total_cost = 0.0
    for i in 1:num_rows
        if i % 50 == 0
            println("  Processed $i test samples")
        end
        obj_value, _, _, _ = facility_location(nloc, ncust, capacity, cost, test_samples[i, :], fixedcost, unmet_pen, scaling_factor, log_file)
        offline_cost[i] = obj_value
        total_cost += obj_value
    end
    total_offline_cost = total_cost / num_rows
    return total_offline_cost
end

# =============================================================================
# LOGGING UTILITIES
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
# MAIN EXECUTION
# =============================================================================

function parse_arguments()
    """Parse command line arguments for scenario count and experiment type"""
    if length(ARGS) != 2
        println("Usage: julia complete_facility_location.jl <num_scenarios> <experiment_type>")
        println("  num_scenarios: Number of scenarios to generate (e.g., 10, 20, 50)")
        println("  experiment_type: low, high, or bernoulli")
        println("")
        println("Examples:")
        println("  julia complete_facility_location.jl 10 low")
        println("  julia complete_facility_location.jl 20 high")
        println("  julia complete_facility_location.jl 50 bernoulli")
        exit(1)
    end
    
    # Parse number of scenarios
    num_scenarios = try
        parse(Int, ARGS[1])
    catch
        println("Error: First argument must be an integer (number of scenarios)")
        exit(1)
    end
    
    if num_scenarios <= 0
        println("Error: Number of scenarios must be positive")
        exit(1)
    end
    
    # Parse experiment type
    experiment_type = lowercase(ARGS[2])
    if !(experiment_type in ["low", "high", "bernoulli"])
        println("Error: Experiment type must be 'low', 'high', or 'bernoulli'")
        exit(1)
    end
    
    return num_scenarios, experiment_type
end

function main(num_scenarios, experiment_type)
    println("="^60)
    println("STOCHASTIC FACILITY LOCATION PROBLEM")
    println("="^60)
    
    # Parameters
    # Use relative path - works for anyone who clones the repository
    file_path = "cap91.txt"
    
    # Check if file exists, provide helpful error message
    if !isfile(file_path)
        println("❌ Error: Data file not found!")
        println("Expected file: $file_path")
        println("Please make sure you're running from the project root directory")
        println("and that the facility_location_instances/ folder contains the data files.")
        return
    end
    unmet_pen = 15
    scaling_factor = 0.2
    capacity_factor = 0.3  # Scale all facility capacities by this factor
    S = [num_scenarios]  # Use command line parameter
    K_values = [2, 3, 4]  # K values for each scenario set
    
    # Set experiment parameters based on experiment type
    if experiment_type == "low"
        method = "new"
        bernoulli_case = false
        println("🔬 Running Gaussian Low experiment with $num_scenarios scenarios")
    elseif experiment_type == "high"
        method = "new"
    bernoulli_case = false
        println("🔬 Running Gaussian High experiment with $num_scenarios scenarios")
    elseif experiment_type == "bernoulli"
        method = "new"
        bernoulli_case = true
        println("🔬 Running Gaussian Bernoulli experiment with $num_scenarios scenarios")
    end
    
    println("Loading data from: $file_path")
    capacity, fixedcost, cost, demandmean = read_orlib_cap(file_path)
    
    # Apply capacity factor to scale all facility capacities
    capacity = capacity .* capacity_factor
    println("Applied capacity factor of $capacity_factor to all facilities")
    println("Capacity range after scaling: min=$(round(minimum(capacity), digits=0)), max=$(round(maximum(capacity), digits=0))")
    
    # Generate demand standard deviations for old method
    if method == "old"
        theta = rand(Uniform(0, 0.5), length(demandmean))
    demandstdev = theta .* demandmean
        println("Using OLD demand generation method with customer-specific demandstdev")
    else
        demandstdev = nothing
        if bernoulli_case
            println("Using NEW demand generation method with Gaussian Bernoulli case")
        else
            println("Using NEW demand generation method (PowerFlow-inspired)")
        end
    end
    
    # Perturb fixed costs by ±20%
    # println("Perturbing fixed costs by ±20%...")
    # for i in 1:length(fixedcost)
    #     perturbation = rand(Uniform(0.8, 1.2))  # 20% up and down
    #     fixedcost[i] = fixedcost[i] * perturbation
    # end
    
    nloc = length(fixedcost)
    ncust = length(demandmean)
    
    println("Problem size: $nloc facilities, $ncust customers")
    println("Fixed costs after perturbation: min=$(round(minimum(fixedcost), digits=2)), max=$(round(maximum(fixedcost), digits=2)), mean=$(round(mean(fixedcost), digits=2))")
    
    # Store all experiment results
    all_experiment_results = Dict()
    
    println("\n" * "="^80)
    println("RUNNING EXPERIMENTS")
    println("="^80)
    
    # Run experiments for each scenario set size S
    for (s_idx, nscen) in enumerate(S)
        println("\n" * "="^80)
        println("🎯 EVALUATING SCENARIO SET S=$nscen / $(length(S))")
        println("="^80)
        
        # Store results for this scenario set
        all_experiment_results[nscen] = Dict()
        
        # Phase 1: Generate scenarios and solve them from scratch
        println("\n" * "="^40)
        println("PHASE 1: SCENARIO GENERATION & SOLUTION (S=$nscen)")
        println("="^40)
        
        println("Generating $nscen scenarios and solving from scratch...")
        phase1_time = @elapsed xsolutions, cost_breakdowns, phase1_work_units = sampling(nscen, nloc, demandmean, capacity, cost, fixedcost, unmet_pen, scaling_factor, demandstdev, bernoulli_case, method, experiment_type)
        
        # Count unique solutions
        unique_solutions, unique_count = n_unique_rows(xsolutions)
        println("Generated $nscen scenarios with $unique_count unique solutions")
        
        # Calculate average facilities opened per scenario for final results
        total_opened = sum(sum(sol) for sol in unique_solutions)
        avg_facilities_per_solution = total_opened / length(unique_solutions)
        
        # Calculate average fixed cost ratio from cost breakdowns
        fixed_cost_ratios = [breakdown[1] / (breakdown[1] + breakdown[2] + breakdown[3]) for breakdown in cost_breakdowns]
        avg_fixed_cost_ratio = sum(fixed_cost_ratios) / length(fixed_cost_ratios)
        # end
        
        
        # Phase 2: Calculate extensive form matrix
        println("\n" * "="^40)
        println("PHASE 2: EXTENSIVE FORM CALCULATIONS (S=$nscen)")
        println("="^40)
        
        println("Calculating extensive form matrix...")
        # Generate the same S scenarios for extensive form calculations
        scenarios = []
        for s in 1:nscen
            push!(scenarios, generate_demand_scenario(ncust, demandmean, demandstdev, false, method, experiment_type))
        end
        
        # Calculate average demand per scenario and total capacity
        total_demand_per_scenario = [sum(scenario) for scenario in scenarios]
        avg_demand_per_scenario = sum(total_demand_per_scenario) / length(total_demand_per_scenario)
        total_capacity = sum(capacity)
        
        phase2_time = @elapsed obj_value, ratio, phase2_work_units = Calculations_For_Extensive_Form(nloc, ncust, capacity, cost, scenarios, xsolutions, fixedcost, unmet_pen, scaling_factor, nothing)
        println("Phase 2 completed. Objective value matrix computed, Ratio matrix computed")
        
        # Calculate offline costs for the same S scenarios
        println("Calculating offline costs for the same S scenarios...")
        offline_costs = []
        offline_breakdowns = []
        for scenario in scenarios
            offline_cost, _, breakdown, _ = facility_location(nloc, ncust, capacity, cost, scenario, fixedcost, unmet_pen, scaling_factor, nothing)
            push!(offline_costs, offline_cost)
            push!(offline_breakdowns, breakdown)
        end
        avg_offline_cost = sum(offline_costs) / length(offline_costs)
        println("Average offline cost: $(round(avg_offline_cost, digits=2))")
        
        
        # Store common results for this scenario set
        all_experiment_results[nscen]["phase1_time"] = phase1_time
        all_experiment_results[nscen]["phase2_time"] = phase2_time
        all_experiment_results[nscen]["phase1_work_units"] = phase1_work_units
        all_experiment_results[nscen]["phase2_work_units"] = phase2_work_units
        all_experiment_results[nscen]["unique_solutions"] = unique_count
        all_experiment_results[nscen]["avg_facilities_opened"] = avg_facilities_per_solution
        all_experiment_results[nscen]["avg_fixed_cost_ratio"] = avg_fixed_cost_ratio
        all_experiment_results[nscen]["avg_demand_per_scenario"] = avg_demand_per_scenario
        all_experiment_results[nscen]["total_capacity"] = total_capacity
        all_experiment_results[nscen]["obj_value"] = obj_value
        all_experiment_results[nscen]["ratio"] = ratio
        all_experiment_results[nscen]["xsolutions"] = xsolutions
        all_experiment_results[nscen]["scenarios"] = scenarios
        all_experiment_results[nscen]["avg_offline_cost"] = avg_offline_cost
        all_experiment_results[nscen]["cost_breakdowns"] = cost_breakdowns
        
        # Phase 3: For each K value, solve reformulated extensive form and evaluate
            println("\n" * "="^40)
        println("PHASE 3: REFORMULATED EXTENSIVE FORM & EVALUATION (S=$nscen)")
            println("="^40)
            
        # For each K value
        for (k_idx, K) in enumerate(K_values)
            println("\n" * "="^60)
            println("K = $K / $(length(K_values)) (S=$nscen)")
            println("="^60)
            
            # Solve reformulated extensive form
            println("Solving reformulated extensive form for K=$K...")
            reformulated_time = @elapsed Solution, z = Reformulated_Extensive_Form(nscen, nscen, K, obj_value, nothing)
            println("Reformulated extensive form objective: $Solution")
            
            # Evaluate on the same S scenarios
            println("Evaluating selected K solutions on the same S scenarios...")
            evaluation_time = @elapsed model_cost, p_cost, m_cost, second_stage_work_units = Evaluation(nloc, ncust, cost, capacity, scenarios, xsolutions, z, fixedcost, unmet_pen, scaling_factor, nothing)
            
            # Calculate gap
            gap = (model_cost - avg_offline_cost) / avg_offline_cost * 100
            println("Model cost: $(round(model_cost, digits=2)), Gap: $(round(gap, digits=2))%")
            
            # Store K-specific results
            all_experiment_results[nscen][string(K)] = Dict(
                "reformulated_time" => reformulated_time,
                "evaluation_time" => evaluation_time,
                "model_cost" => model_cost,
                "gap" => gap,
                "second_stage_work_units" => second_stage_work_units,
                "penalty_costs" => p_cost,
                "model_costs" => m_cost,
                "z" => z,
                "avg_offline_cost" => avg_offline_cost
            )
            
            println("Completed K = $K (S=$nscen)")
            println("")
        end
        
        println("Completed scenario set S=$nscen")
        println("")
    end
    
    # Print final results summary
    println("\n" * "="^80)
    println("FINAL RESULTS SUMMARY")
    println("="^80)
    
    for nscen in S
        println("\n" * "="^60)
        println("SCENARIO SET S=$nscen")
        println("="^60)
        println("Average Offline Cost: $(round(all_experiment_results[nscen]["avg_offline_cost"], digits=2))")
        println("Solve Time (Work Units) - Offline: $(round(all_experiment_results[nscen]["phase1_work_units"], digits=2))")
        println("Calculations For Extensive Form Time (Work Units): $(round(all_experiment_results[nscen]["phase2_work_units"], digits=2))")
        println("Unique Binary Vectors: $(all_experiment_results[nscen]["unique_solutions"])")
        println("Average Facilities Opened per Scenario (Offline): $(round(all_experiment_results[nscen]["avg_facilities_opened"], digits=2))")
        println("Average Fixed Cost Ratio (Offline): $(round(all_experiment_results[nscen]["avg_fixed_cost_ratio"] * 100, digits=2))%")
        println("Average Demand per Scenario: $(round(all_experiment_results[nscen]["avg_demand_per_scenario"], digits=0))")
        println("Total Capacity (All Facilities): $(round(all_experiment_results[nscen]["total_capacity"], digits=0))")
        
        # Create comprehensive table
        println("\nK | Model Cost | Model Gap (%) | REF Time (WU) | Second Stage (WU)")
        println("-"^70)
        
        for K in K_values
            k_str = string(K)
            if haskey(all_experiment_results[nscen], k_str)
                k_data = all_experiment_results[nscen][k_str]
                println("$K | $(round(k_data["model_cost"], digits=2)) | $(round(k_data["gap"], digits=2)) | $(round(k_data["reformulated_time"], digits=2)) | $(round(k_data["second_stage_work_units"], digits=2))")
            end
        end
    end
    
    # =============================================================================
    # PENALTY ANALYSIS SUMMARY
    # =============================================================================
    
    println("\n" * "="^60)
    println("PENALTY ANALYSIS SUMMARY")
    println("="^60)
    
    for nscen in S
        println("\n=== SCENARIO SET S=$nscen ===")
        
        for K in K_values
            k_str = string(K)
            if haskey(all_experiment_results[nscen], k_str)
                k_data = all_experiment_results[nscen][k_str]
                
                # Get penalty data if available
                if haskey(k_data, "penalty_costs") && haskey(k_data, "model_costs")
                    penalty_data = k_data["penalty_costs"]
                    model_cost_per_scenario = k_data["model_costs"]
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
                else
                    println("K=$K: Penalty data not available")
                end
            end
        end
    end
    
    # =============================================================================
    # OFFLINE COST BREAKDOWN SUMMARY
    # =============================================================================
    
    println("\n" * "="^80)
    println("OFFLINE COST BREAKDOWN SUMMARY")
    println("="^80)
    
    for nscen in S
        println("\n=== SCENARIO SET S=$nscen ===")
        println("Scenario | Fixed Cost | Transportation Cost | Penalty Cost | Total Cost")
        println("-"^80)
        
        cost_breakdowns = all_experiment_results[nscen]["cost_breakdowns"]
        for scenario in 1:nscen
            breakdown = cost_breakdowns[scenario]
            fixed_cost = breakdown[1]
            transportation_cost = breakdown[2]
            penalty_cost = breakdown[3]
            total_cost = fixed_cost + transportation_cost + penalty_cost
            
            println("$scenario | $(round(fixed_cost, digits=2)) | $(round(transportation_cost, digits=2)) | $(round(penalty_cost, digits=2)) | $(round(total_cost, digits=2))")
        end
    end
    
    # =============================================================================
    # GENERATE 4 SEPARATE TEXT FILES
    # =============================================================================
    
    # Get instance metadata
    n_facilities = length(capacity)
    n_customers = length(demandmean)
    instance_name = basename(file_path)
    
    println("\n" * "="^80)
    println("GENERATING 4 SEPARATE TEXT FILES")
    println("="^80)
    
    # 1. SUMMARY FILE
    summary_filename = "FL1_S$(num_scenarios)_$(experiment_type)_summary.txt"
    open(summary_filename, "w") do io
        for nscen in S
            println(io, "$instance_name\t$n_facilities\t$n_customers\t$nscen\t$experiment_type\t$(round(all_experiment_results[nscen]["avg_offline_cost"], digits=2))\t$(round(all_experiment_results[nscen]["phase1_work_units"], digits=2))\t$(round(all_experiment_results[nscen]["phase2_work_units"], digits=2))\t$(all_experiment_results[nscen]["unique_solutions"])\t$(round(all_experiment_results[nscen]["avg_facilities_opened"], digits=2))\t$(round(all_experiment_results[nscen]["avg_demand_per_scenario"], digits=0))\t$(round(all_experiment_results[nscen]["total_capacity"], digits=0))")
        end
    end
    println("✅ Summary file: $summary_filename")
    
    # 2. K RESULTS FILE
    kresults_filename = "FL1_S$(num_scenarios)_$(experiment_type)_kresults.txt"
    open(kresults_filename, "w") do io
        for nscen in S
            for K in K_values
                k_str = string(K)
                if haskey(all_experiment_results[nscen], k_str)
                    k_data = all_experiment_results[nscen][k_str]
                    println(io, "$instance_name\t$n_facilities\t$n_customers\t$nscen\t$experiment_type\t$K\t$(round(k_data["model_cost"], digits=2))\t$(round(k_data["gap"], digits=2))\t$(round(k_data["reformulated_time"], digits=2))\t$(round(k_data["second_stage_work_units"], digits=2))")
                end
            end
        end
    end
    println("✅ K Results file: $kresults_filename")
    
    # 3. OFFLINE COST BREAKDOWN FILE
    offline_filename = "FL1_S$(num_scenarios)_$(experiment_type)_offline_cost_breakdown.txt"
    open(offline_filename, "w") do io
        for nscen in S
            cost_breakdowns = all_experiment_results[nscen]["cost_breakdowns"]
            for scenario in 1:nscen
                breakdown = cost_breakdowns[scenario]
                fixed_cost = breakdown[1]
                transportation_cost = breakdown[2]
                penalty_cost = breakdown[3]
                total_cost = fixed_cost + transportation_cost + penalty_cost
                
                println(io, "$instance_name\t$n_facilities\t$n_customers\t$nscen\t$experiment_type\t$scenario\t$(round(fixed_cost, digits=2))\t$(round(transportation_cost, digits=2))\t$(round(penalty_cost, digits=2))\t$(round(total_cost, digits=2))")
            end
        end
    end
    println("✅ Offline Cost Breakdown file: $offline_filename")
    
    # 4. PENALTY ANALYSIS FILE
    penalty_filename = "FL1_S$(num_scenarios)_$(experiment_type)_penalty_analysis.txt"
    open(penalty_filename, "w") do io
        for nscen in S
            for K in K_values
                k_str = string(K)
                if haskey(all_experiment_results[nscen], k_str)
                    k_data = all_experiment_results[nscen][k_str]
                    if haskey(k_data, "penalty_costs") && haskey(k_data, "model_costs")
                        penalty_data = k_data["penalty_costs"]
                        model_cost_per_scenario = k_data["model_costs"]
                        penalty_pos = findall(x -> x > 0.1, penalty_data)
                        
                        for pos in penalty_pos
                            println(io, "$instance_name\t$n_facilities\t$n_customers\t$nscen\t$experiment_type\t$K\t$pos\t$(round(model_cost_per_scenario[pos], digits=2))\t$(round(penalty_data[pos], digits=2))")
                        end
                    end
                end
            end
        end
    end
    println("✅ Penalty Analysis file: $penalty_filename")
    
    println("\n📊 Generated 4 separate files with instance metadata in each")
    println("📝 Format: Tab-separated values (TSV) for easy processing")
    
    # =============================================================================
    # GENERATE XLSX OUTPUT (COMMENTED OUT FOR DEBUGGING)
    # =============================================================================
    
    # # Generate filename based on S and experiment_type parameters
    # xlsx_filename = "FL_results_S$(num_scenarios)_$(experiment_type).xlsx"
    # 
    # println("\n" * "="^80)
    # println("GENERATING XLSX OUTPUT: $xlsx_filename")
    # println("="^80)
    # 
    # # Prepare data for XLSX using DataFrames
    # summary_data = DataFrame()
    # k_results_data = DataFrame()
    # offline_breakdown_data = DataFrame()
    # penalty_analysis_data = DataFrame()
    # 
    # # Initialize summary data
    # summary_data.Scenario_Set = Int[]
    # summary_data.Experiment_Type = String[]
    # summary_data.Avg_Offline_Cost = Float64[]
    # summary_data.Offline_Solve_Time_Work_Units = Float64[]
    # summary_data.Continuous_Variable_Calc_Work_Units = Float64[]
    # summary_data.Unique_Solutions = Int[]
    # summary_data.Avg_Facilities_Opened = Float64[]
    # summary_data.Avg_Demand = Float64[]
    # summary_data.Total_Capacity = Float64[]
    # 
    # # Initialize K results data
    # k_results_data.K = Int[]
    # k_results_data.Model_Cost = Float64[]
    # k_results_data.Model_Gap_Percent = Float64[]
    # k_results_data.Assignment_Formulation_Work_Units = Float64[]
    # k_results_data.Second_Stage_Work_Units = Float64[]
    # 
    # # Initialize offline breakdown data
    # offline_breakdown_data.Scenario_Number = Int[]
    # offline_breakdown_data.Fixed_Cost = Float64[]
    # offline_breakdown_data.Transportation_Cost = Float64[]
    # offline_breakdown_data.Penalty_Cost = Float64[]
    # offline_breakdown_data.Total_Cost = Float64[]
    # 
    # # Initialize penalty analysis data
    # penalty_analysis_data.K = Int[]
    # penalty_analysis_data.Scenario_Number = Int[]
    # penalty_analysis_data.Model_Cost = Float64[]
    # penalty_analysis_data.Penalty_Cost = Float64[]
    # 
    # for nscen in S
    #     # Summary data
    #     push!(summary_data, (
    #         nscen,
    #         experiment_type,
    #         round(all_experiment_results[nscen]["avg_offline_cost"], digits=2),
    #         round(all_experiment_results[nscen]["phase1_work_units"], digits=2),
    #         round(all_experiment_results[nscen]["phase2_work_units"], digits=2),
    #         all_experiment_results[nscen]["unique_solutions"],
    #         round(all_experiment_results[nscen]["avg_facilities_opened"], digits=2),
    #         round(all_experiment_results[nscen]["avg_demand_per_scenario"], digits=0),
    #         round(all_experiment_results[nscen]["total_capacity"], digits=0)
    #     ))
    #     
    #     # K-specific results
    #     for K in K_values
    #         k_str = string(K)
    #         if haskey(all_experiment_results[nscen], k_str)
    #             k_data = all_experiment_results[nscen][k_str]
    #             push!(k_results_data, (
    #                 K,
    #                 round(k_data["model_cost"], digits=2),
    #                 round(k_data["gap"], digits=2),
    #                 round(k_data["reformulated_time"], digits=2),
    #                 round(k_data["second_stage_work_units"], digits=2)
    #             ))
    #         end
    #     end
    #     
    #     # Offline breakdown data
    #     cost_breakdowns = all_experiment_results[nscen]["cost_breakdowns"]
    #     for scenario in 1:nscen
    #         breakdown = cost_breakdowns[scenario]
    #         fixed_cost = breakdown[1]
    #         transportation_cost = breakdown[2]
    #         penalty_cost = breakdown[3]
    #         total_cost = fixed_cost + transportation_cost + penalty_cost
    #         
    #         push!(offline_breakdown_data, (
    #             scenario,
    #             round(fixed_cost, digits=2),
    #             round(transportation_cost, digits=2),
    #             round(penalty_cost, digits=2),
    #             round(total_cost, digits=2)
    #         ))
    #     end
    #     
    #     # Penalty analysis data
    #     for K in K_values
    #         k_str = string(K)
    #         if haskey(all_experiment_results[nscen], k_str)
    #             k_data = all_experiment_results[nscen][k_str]
    #             if haskey(k_data, "penalty_costs") && haskey(k_data, "model_costs")
    #                 penalty_data = k_data["penalty_costs"]
    #                 model_cost_per_scenario = k_data["model_costs"]
    #                 penalty_pos = findall(x -> x > 0.1, penalty_data)
    #                 
    #                 for pos in penalty_pos
    #                     push!(penalty_analysis_data, (
    #                         K,
    #                         pos,
    #                         round(model_cost_per_scenario[pos], digits=2),
    #                         round(penalty_data[pos], digits=2)
    #                     ))
    #                 end
    #             end
    #         end
    #     end
    # end
    # 
    # # Create XLSX file
    # XLSX.openxlsx(xlsx_filename, mode="w") do xf
    #     # Create worksheets (this will replace the default Sheet1)
    #     XLSX.addsheet!(xf, "Summary")
    #     XLSX.addsheet!(xf, "K_Results")
    #     XLSX.addsheet!(xf, "Offline_Breakdown")
    #     XLSX.addsheet!(xf, "Penalty_Analysis")
    #     
    #     # Summary worksheet
    #     XLSX.writetable!(xf["Summary"], summary_data)
    #     
    #     # K Results worksheet
    #     XLSX.writetable!(xf["K_Results"], k_results_data)
    #     
    #     # Offline Breakdown worksheet
    #     XLSX.writetable!(xf["Offline_Breakdown"], offline_breakdown_data)
    #     
    #     # Penalty Analysis worksheet
    #     XLSX.writetable!(xf["Penalty_Analysis"], penalty_analysis_data)
    # end
    # 
    # println("✅ XLSX file generated successfully: $xlsx_filename")
    # println("📊 Contains 4 worksheets: Summary, K_Results, Offline_Breakdown, Penalty_Analysis")
    
    println("\n" * "="^80)
    println("ALL EXPERIMENTS COMPLETED!")
    println("="^80)
    
    # Show all solutions for each scenario set
    # println("\n" * "="^80)
    # println("ALL SOLUTIONS FOR EACH SCENARIO SET")
    # println("="^80)
    # 
    # for nscen in S
    #     println("\n" * "="^60)
    #     println("SCENARIO SET S=$nscen - ALL SOLUTIONS")
    #     println("="^60)
    #     
    #     xsolutions = all_experiment_results[nscen]["xsolutions"]
    #     println("All $nscen solutions (including duplicates):")
    #     for i in 1:nscen
    #         solution_vector = xsolutions[i, :]
    #         println("  Scenario $i: $solution_vector")
    #     end
    # end
    
    return nothing
end
# Run the main function
if abspath(PROGRAM_FILE) == @__FILE__
    num_scenarios, experiment_type = parse_arguments()
    main(num_scenarios, experiment_type)
end

