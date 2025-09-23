#!/usr/bin/env julia

"""
Complete Facility Location Problem - Standalone Script
This script runs the entire facility location optimization pipeline automatically.
"""

using JuMP, Gurobi, Random, Distributions, DataFrames, LinearAlgebra

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
        set_optimizer_attribute(model, "OutputFlag", 1)  # Show in console
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
    return objective_value(model), collect(value.(x))
end

function solve_for_continuous_var(nloc, ncust, capacity, cost, demand, solved_x, fixedcost, unmet_pen, scaling_factor, log_file=nothing)
    """Solve for continuous variables with fixed binary decisions"""
    model = Model(Gurobi.Optimizer)
    
    # Set logging based on whether we want to save logs
    if log_file !== nothing
        set_optimizer_attribute(model, "LogFile", log_file)
        set_optimizer_attribute(model, "LogToConsole", 0)  # Don't show in console
    else
        set_optimizer_attribute(model, "OutputFlag", 1)  # Show in console
    end
    
    # Decision Variables
    @variable(model, x[1:nloc], Bin)
    @variable(model, y[1:nloc, 1:ncust] >= 0)
    @variable(model, shortfall[1:ncust] >= 0)

    # Constraints
    @constraint(model, [j in 1:ncust], sum(y[i,j] for i in 1:nloc) + shortfall[j] == demand[j])
    @constraint(model, [i in 1:nloc], sum(y[i,j] for j in 1:ncust) - capacity[i]*x[i] <= 0)
    @constraint(model, x .== solved_x)  # fixing the binary decision variable

    # Objective function
    @objective(model, Min, 
        sum(fixedcost[i]*x[i] for i in 1:nloc) + 
        scaling_factor * sum(cost[i,j]*y[i,j] for i in 1:nloc for j in 1:ncust) + 
        sum(unmet_pen*shortfall[j] for j in 1:ncust))

    optimize!(model)
    if termination_status(model) == MOI.OPTIMAL
        return objective_value(model), sum(fixedcost[i]*value.(x[i]) for i in 1:nloc)/objective_value(model)
    else
        return -1, -1
    end
end

# =============================================================================
# SAMPLING AND SCENARIO GENERATION
# =============================================================================

function generate_demand_scenario(ncust, demandmean, demandstdev, bernoulli_case)
    """Generate a single demand scenario"""
    globaldem = round(max(rand(Normal(1000, 300)), 0), digits=0)
    demand = zeros(ncust)
    
    for i in 1:ncust
        if bernoulli_case
            Bernoulli_demand = rand(Bernoulli(0.5))
        else
            Bernoulli_demand = 1
        end
        demand[i] = Bernoulli_demand * (globaldem + round(max(rand(Normal(demandmean[i], demandstdev[i])), 0), digits=0))
    end
    return demand
end

function sampling(nscen, nloc, demandmean, demandstdev, capacity, cost, fixedcost, unmet_pen, scaling_factor, bernoulli_case, log_file=nothing)
    """Generate multiple scenarios and solve for each"""
    ncust = length(demandmean)
    xsolutions = zeros(nscen, nloc)
    
    println("Generating $nscen scenarios...")
    for scenario in 1:nscen
        if scenario % 100 == 0
            println("  Processed $scenario scenarios")
        end
        demand = generate_demand_scenario(ncust, demandmean, demandstdev, bernoulli_case)
        _, xsolutions[scenario, :] = facility_location(nloc, ncust, capacity, cost, demand, fixedcost, unmet_pen, scaling_factor, log_file)
    end
    return xsolutions
end

# =============================================================================
# REFORMULATED EXTENSIVE FORM
# =============================================================================

function Calculations_For_Extensive_Form(nloc, ncust, capacity, cost, online_samples, xsolutions, fixedcost, unmet_pen, scaling_factor, log_file=nothing)
    """Calculate objective values for all scenario-solution pairs"""
    total_scenarios = size(online_samples, 1)
    total_solutions = size(xsolutions, 1)
    obj_value = zeros(total_scenarios, total_solutions)
    ratio = zeros(total_scenarios, total_solutions)
    
    println("Computing extensive form matrix...")
    for scenario in 1:total_scenarios
        if scenario % 20 == 0
            println("  Processed $scenario scenarios")
        end
        demand = online_samples[scenario, :]
        for solution in 1:total_solutions
            obj_value[scenario, solution], ratio[scenario, solution] = solve_for_continuous_var(
                nloc, ncust, capacity, cost, demand, xsolutions[solution, :], 
                fixedcost, unmet_pen, scaling_factor, log_file)
        end
    end
    return obj_value, ratio
end

function Reformulated_Extensive_Form(S, l, K, obj_value, log_file=nothing)
    """Solve the reformulated extensive form to select K optimal solutions"""
    model = Model(Gurobi.Optimizer)
    MOI.set(model, MOI.TimeLimitSec(), 30.0)
    
    # Set logging based on whether we want to save logs
    if log_file !== nothing
        set_optimizer_attribute(model, "LogFile", log_file)
        set_optimizer_attribute(model, "LogToConsole", 0)  # Don't show in console
    else
        set_optimizer_attribute(model, "OutputFlag", 1)  # Show in console
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
        set_optimizer_attribute(model, "OutputFlag", 1)  # Show in console
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
    return objective_value(model), sum(unmet_pen*value.(shortfall[j]) for j in 1:ncust)
end

function Evaluation(nloc, ncust, cost, capacity, test_sample, x_solutions, z, fixedcost, unmet_pen, scaling_factor, log_file=nothing)
    """Evaluate solution performance on test data"""
    num_rows = size(test_sample, 1)
    p_cost = zeros(num_rows)
    m_cost = zeros(num_rows)
    
    model_cost = 0.0
    for index in 1:num_rows
        model_obj_value, p = Second_Stage_Cost(nloc, ncust, cost, capacity, test_sample[index, :], x_solutions, z, fixedcost, unmet_pen, scaling_factor, log_file)
        model_cost += model_obj_value
        p_cost[index] = p
        m_cost[index] = model_obj_value
    end
    
    total_model_cost = model_cost / num_rows
    return total_model_cost, p_cost, m_cost
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
        obj_value, _ = facility_location(nloc, ncust, capacity, cost, test_samples[i, :], fixedcost, unmet_pen, scaling_factor, log_file)
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

function main()
    println("="^60)
    println("STOCHASTIC FACILITY LOCATION PROBLEM")
    println("="^60)
    
    # Parameters
    # Use relative path - works for anyone who clones the repository
    file_path = "facility location instances/cap91.txt"
    
    # Check if file exists, provide helpful error message
    if !isfile(file_path)
        println("❌ Error: Data file not found!")
        println("Expected file: $file_path")
        println("Please make sure you're running from the project root directory")
        println("and that the facility location instances/ folder contains the data files.")
        return
    end
    unmet_pen = 50
    scaling_factor = 0.000015
    bernoulli_case = false
    nscen = 30  # Number of scenarios for sampling
    nsamples = 100  # Number of online samples
    test_samples = 100  # Number of test samples
    K = 5  # Number of solutions to select
    
    println("Loading data from: $file_path")
    capacity, fixedcost, cost, demandmean = read_orlib_cap(file_path)
    nloc = length(fixedcost)
    ncust = length(demandmean)
    
    println("Problem size: $nloc facilities, $ncust customers")
    
    # Generate demand standard deviations
    theta = rand(Uniform(0, 0.5), ncust)
    demandstdev = theta .* demandmean
    
    println("\n" * "="^40)
    println("PHASE 1: SCENARIO GENERATION")
    println("="^40)
    
    # Create parameter-based directory structure
    param_folder = "nscen$(nscen)_K$(K)"
    base_log_path = "Logs/Facility Location Logs/$(param_folder)"
    
    # Create all necessary directories
    for phase in ["phase1", "phase2", "phase3"]
        phase_dir = "$(base_log_path)/$(phase)"
        if !isdir(phase_dir)
            mkpath(phase_dir)
        end
    end
    
    # Set up all 5 log files
    phase1_log = "$(base_log_path)/phase1/scenario_generation.log"
    phase2_calc_log = "$(base_log_path)/phase2/extensive_form_calculations.log"
    phase2_reform_log = "$(base_log_path)/phase2/reformulated_extensive_form.log"
    phase3_offline_log = "$(base_log_path)/phase3/offline_test.log"
    phase3_eval_log = "$(base_log_path)/phase3/evaluation.log"
    
    println("Saving logs to: $base_log_path")
    println("  Phase 1: scenario_generation.log")
    println("  Phase 2: extensive_form_calculations.log, reformulated_extensive_form.log")
    println("  Phase 3: offline_test.log, evaluation.log")
    
    # Generate scenarios and solutions with logging
    xsolutions = sampling(nscen, nloc, demandmean, demandstdev, capacity, cost, fixedcost, unmet_pen, scaling_factor, bernoulli_case, phase1_log)
    
    # Count unique solutions
    _, u1 = n_unique_rows(xsolutions)
    println("Generated $nscen scenarios with $u1 unique solutions")
    
    # Add summary to Phase 1 log
    phase1_params = Dict("nscen" => nscen, "nloc" => nloc, "ncust" => ncust, "bernoulli_case" => bernoulli_case)
    phase1_results = Dict("unique_solutions" => u1, "total_scenarios" => nscen)
    append_summary_to_log(phase1_log, "PHASE 1: SCENARIO GENERATION", phase1_params, phase1_results)
    
    # Generate online samples
    println("Generating $nsamples online samples...")
    online_samples = zeros(nsamples, ncust)
    for s in 1:nsamples
        online_samples[s, :] = generate_demand_scenario(ncust, demandmean, demandstdev, bernoulli_case)
    end
    
    # Generate test samples
    println("Generating $test_samples test samples...")
    test_samples_data = zeros(test_samples, ncust)
    for s in 1:test_samples
        test_samples_data[s, :] = generate_demand_scenario(ncust, demandmean, demandstdev, bernoulli_case)
    end
    
    println("\n" * "="^40)
    println("PHASE 2: REFORMULATED EXTENSIVE FORM")
    println("="^40)
    
    # Solve reformulated extensive form with logging
    obj_matrix, r = Calculations_For_Extensive_Form(nloc, ncust, capacity, cost, online_samples, xsolutions, fixedcost, unmet_pen, scaling_factor, phase2_calc_log)
    n_neg1 = count(x -> x == -1, obj_matrix)
    println("Infeasible scenario-solution pairs: $n_neg1")
    
    Solution, z = Reformulated_Extensive_Form(nsamples, nscen, K, obj_matrix, phase2_reform_log)
    println("Reformulated extensive form objective: $Solution")
    
    # Add summary to Phase 2 logs
    phase2_params = Dict("nsamples" => nsamples, "nscen" => nscen, "K" => K, "infeasible_pairs" => n_neg1)
    phase2_calc_results = Dict("total_scenarios" => nsamples, "total_solutions" => nscen, "infeasible_pairs" => n_neg1)
    phase2_reform_results = Dict("objective_value" => Solution, "selected_solutions" => K)
    append_summary_to_log(phase2_calc_log, "PHASE 2A: EXTENSIVE FORM CALCULATIONS", phase2_params, phase2_calc_results)
    append_summary_to_log(phase2_reform_log, "PHASE 2B: REFORMULATED EXTENSIVE FORM", phase2_params, phase2_reform_results)
    
    println("\n" * "="^40)
    println("PHASE 3: EVALUATION")
    println("="^40)
    
    # Calculate offline costs with logging
    offline_cost = OfflineTest(nloc, ncust, capacity, cost, test_samples_data, fixedcost, unmet_pen, scaling_factor, phase3_offline_log)
    println("Average offline cost: $offline_cost")
    
    # Evaluate model performance with logging
    modelcost, pcos, mcost = Evaluation(nloc, ncust, cost, capacity, test_samples_data, xsolutions, z, fixedcost, unmet_pen, scaling_factor, phase3_eval_log)
    println("Average model cost: $modelcost")
    
    # Calculate gap
    gap = (modelcost - offline_cost) / offline_cost * 100
    println("Performance gap: $(round(gap, digits=2))%")
    
    # Analyze penalties
    penalty_pos = findall(x -> x > 0.1, pcos)
    penalty_val = pcos[penalty_pos]
    println("Scenarios with penalties: $(length(penalty_pos))")
    
    # Add summary to Phase 3 logs
    phase3_params = Dict("test_samples" => test_samples, "nloc" => nloc, "ncust" => ncust)
    phase3_offline_results = Dict("average_offline_cost" => offline_cost, "total_test_samples" => test_samples)
    phase3_eval_results = Dict("average_model_cost" => modelcost, "performance_gap_percent" => round(gap, digits=2), "scenarios_with_penalties" => length(penalty_pos))
    append_summary_to_log(phase3_offline_log, "PHASE 3A: OFFLINE TEST", phase3_params, phase3_offline_results)
    append_summary_to_log(phase3_eval_log, "PHASE 3B: EVALUATION", phase3_params, phase3_eval_results)
    
    println("\n" * "="^60)
    println("FINAL RESULTS SUMMARY (First 3 Phases)")
    println("="^60)
    println("Offline optimal cost:     $(round(offline_cost, digits=2))")
    println("Model cost:               $(round(modelcost, digits=2))")
    println("Model gap:                $(round(gap, digits=2))%")
    println("Scenarios with penalties: $(length(penalty_pos))")
    println("="^60)
    
    return offline_cost, modelcost, gap
end

# Run the main function
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
