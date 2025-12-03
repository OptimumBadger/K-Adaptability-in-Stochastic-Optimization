#!/usr/bin/env julia

"""
L-Augmentation Heuristic - Standalone Implementation
This script runs the L-Augmentation heuristic for facility location optimization.
It includes all necessary infrastructure to run independently.
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

function facility_location(nloc, ncust, capacity, cost, demand, fixedcost, unmet_pen, scaling_factor)
    """Solve the basic facility location problem"""
    model = Model(Gurobi.Optimizer)
    
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

function solve_for_continuous_var(nloc, ncust, capacity, cost, demand, solved_x, fixedcost, unmet_pen, scaling_factor)
    """Solve for continuous variables with fixed binary decisions"""
    model = Model(Gurobi.Optimizer)
    
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

function sampling(nscen, nloc, demandmean, demandstdev, capacity, cost, fixedcost, unmet_pen, scaling_factor, bernoulli_case)
    """Generate multiple scenarios and solve for each"""
    ncust = length(demandmean)
    xsolutions = zeros(nscen, nloc)
    
    println("🔄 Generating $nscen scenarios...")
    for scenario in 1:nscen
        if scenario % 100 == 0
            println("  Processed $scenario scenarios")
        end
        demand = generate_demand_scenario(ncust, demandmean, demandstdev, bernoulli_case)
        _, xsolutions[scenario, :] = facility_location(nloc, ncust, capacity, cost, demand, fixedcost, unmet_pen, scaling_factor)
    end
    return xsolutions
end

# =============================================================================
# REFORMULATED EXTENSIVE FORM
# =============================================================================

function Calculations_For_Extensive_Form(nloc, ncust, capacity, cost, online_samples, xsolutions, fixedcost, unmet_pen, scaling_factor)
    """Calculate objective values for all scenario-solution pairs"""
    total_scenarios = size(online_samples, 1)
    total_solutions = size(xsolutions, 1)
    obj_value = zeros(total_scenarios, total_solutions)
    ratio = zeros(total_scenarios, total_solutions)
    
    println("🔄 Computing extensive form matrix...")
    for scenario in 1:total_scenarios
        if scenario % 20 == 0
            println("  Processed $scenario scenarios")
        end
        demand = online_samples[scenario, :]
        for solution in 1:total_solutions
            obj_value[scenario, solution], ratio[scenario, solution] = solve_for_continuous_var(
                nloc, ncust, capacity, cost, demand, xsolutions[solution, :], 
                fixedcost, unmet_pen, scaling_factor)
        end
    end
    return obj_value, ratio
end

function Reformulated_Extensive_Form(S, l, K, obj_value)
    """Solve the reformulated extensive form to select K optimal solutions"""
    println("🔄 Solving reformulated extensive form...")
    model = Model(Gurobi.Optimizer)
    MOI.set(model, MOI.TimeLimitSec(), 30.0)

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
        println("✅ Reformulated extensive form solved successfully")
        return objective_value(model), value.(z)
    else
        println("❌ Reformulated extensive form failed to solve")
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

function obtain_policy(nloc, ncust, capacity, cost, online_samples, xsolutions, K, fixedcost, unmet_pen, scaling_factor)
    """Obtain K optimal solutions using reformulated extensive form"""
    S = size(online_samples, 1)
    l = size(xsolutions, 1)
    obj_matrix, _ = Calculations_For_Extensive_Form(nloc, ncust, capacity, cost, online_samples, xsolutions, fixedcost, unmet_pen, scaling_factor)
    _, z = Reformulated_Extensive_Form(S, l, K, obj_matrix)
    return z
end

function assign_partitions(nloc, ncust, capacity, cost, z, xsolutions, online_samples, fixedcost, unmet_pen, scaling_factor)
    """Assign scenarios to optimal solutions"""
    selected_indices = findall(v -> v ≥ 0.5, z)
    K = length(selected_indices)
    S = size(online_samples, 1)
    partitions = [Int[] for _ in 1:K]

    println("🔄 Assigning scenarios to partitions...")
    for j in 1:S
        if j % 20 == 0
            println("  Processing scenario $j/$S")
        end
        sample = online_samples[j, :]
        best_val = Inf
        best_k = 0
        for (k_idx, sol_idx) in enumerate(selected_indices)
            val, _ = solve_for_continuous_var(nloc, ncust, capacity, cost, sample, xsolutions[sol_idx, :], fixedcost, unmet_pen, scaling_factor)
            if val < best_val
                best_val = val
                best_k = k_idx
            end
        end
        push!(partitions[best_k], j)
    end
    println("✅ Completed partition assignment")
    return partitions
end

function extensive_form(nloc, ncust, capacity, cost, partition, online_samples, fixedcost, unmet_pen, scaling_factor)
    """Solve extensive form for a given partition"""
    model = Model(Gurobi.Optimizer)
    nscen = length(partition)

    # Decision Variables
    @variable(model, x[1:nloc], Bin)
    @variable(model, y[1:nloc, 1:ncust, 1:nscen] >= 0)
    @variable(model, shortfall[1:ncust, 1:nscen] >= 0)

    # Constraints
    for k in 1:nscen
        scenario_index = partition[k]
        @constraint(model, [j in 1:ncust], sum(y[i, j, k] for i in 1:nloc) + shortfall[j, k] == online_samples[scenario_index, j])
        @constraint(model, [i in 1:nloc], sum(y[i, j, k] for j in 1:ncust) - capacity[i]*x[i] <= 0)
    end

    # Objective function
    @objective(model, Min, 
        sum(fixedcost[i]*x[i] for i in 1:nloc) + 
        scaling_factor/nscen * sum(cost[i, j]*y[i, j, k] for i in 1:nloc for j in 1:ncust for k in 1:nscen) + 
        1/nscen * sum(unmet_pen*shortfall[j, k] for j in 1:ncust for k in 1:nscen))

    optimize!(model)
    return collect(value.(x))
end

function Augmentation_Heuristic(nloc, ncust, capacity, cost, online_samples, xsolutions, max_iter, K, fixedcost, unmet_pen, scaling_factor)
    """L-Augmentation Heuristic"""
    println("🔄 Starting L-Augmentation Heuristic...")
    println("📊 Maximum iterations: $max_iter")
    L = copy(xsolutions)
    count = 0
    
    for i in 1:max_iter
        count += 1
        println("  🔄 Iteration $count/$max_iter")
        L_prev = copy(L)
        
        # Obtain K candidates
        println("    📊 Obtaining policy...")
        z = obtain_policy(nloc, ncust, capacity, cost, online_samples, L, K, fixedcost, unmet_pen, scaling_factor)
        
        # Generate partitions
        println("    📊 Assigning partitions...")
        partitions = assign_partitions(nloc, ncust, capacity, cost, z, L, online_samples, fixedcost, unmet_pen, scaling_factor)
        
        # Solve extensive form for each partition
        println("    📊 Solving extensive forms...")
        Y_new = zeros(K, nloc)
        for (i, part) in enumerate(partitions)
            if !isempty(part)
                Y_new[i, :] = extensive_form(nloc, ncust, capacity, cost, part, online_samples, fixedcost, unmet_pen, scaling_factor)
            end
        end
        
        L = unique(vcat(L_prev, Y_new), dims=1)
        
        if size(L, 1) == size(L_prev, 1)
            println("    ✅ No new solutions found. Stopping.")
            return L, z, count
        end
        println("    ✅ Added $(size(L, 1) - size(L_prev, 1)) new solutions")
    end
    
    znew = obtain_policy(nloc, ncust, capacity, cost, online_samples, L, K, fixedcost, unmet_pen, scaling_factor)
    return L, znew, count
end

# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

function Second_Stage_Cost(nloc, ncust, cost, capacity, demand, xsolutions, z, fixedcost, unmet_pen, scaling_factor)
    """Calculate second stage cost for given solution"""
    model = Model(Gurobi.Optimizer)
    
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

function Evaluation(nloc, ncust, cost, capacity, test_sample, x_solutions, z, fixedcost, unmet_pen, scaling_factor)
    """Evaluate solution performance on test data"""
    num_rows = size(test_sample, 1)
    p_cost = zeros(num_rows)
    m_cost = zeros(num_rows)
    
    println("🔄 Evaluating solution performance...")
    model_cost = 0.0
    for index in 1:num_rows
        if index % 50 == 0
            println("  📈 Processing test sample $index/$num_rows")
        end
        model_obj_value, p = Second_Stage_Cost(nloc, ncust, cost, capacity, test_sample[index, :], x_solutions, z, fixedcost, unmet_pen, scaling_factor)
        model_cost += model_obj_value
        p_cost[index] = p
        m_cost[index] = model_obj_value
    end
    
    total_model_cost = model_cost / num_rows
    return total_model_cost, p_cost, m_cost
end

function OfflineTest(nloc, ncust, capacity, cost, test_samples, fixedcost, unmet_pen, scaling_factor)
    """Calculate offline optimal costs for test data"""
    num_rows = size(test_samples, 1)
    offline_cost = zeros(num_rows)
    
    println("🔄 Computing offline optimal costs...")
    total_cost = 0.0
    for i in 1:num_rows
        if i % 50 == 0
            println("  📈 Processing test sample $i/$num_rows")
        end
        obj_value, _ = facility_location(nloc, ncust, capacity, cost, test_samples[i, :], fixedcost, unmet_pen, scaling_factor)
        offline_cost[i] = obj_value
        total_cost += obj_value
    end
    total_offline_cost = total_cost / num_rows
    return total_offline_cost
end

# =============================================================================
# MAIN EXECUTION
# =============================================================================

function main()
    println("="^60)
    println("L-AUGMENTATION HEURISTIC FOR FACILITY LOCATION")
    println("="^60)
    
    # Parameters
    file_path = "instances/cap91.txt"
    unmet_pen = 50
    scaling_factor = 0.000015
    bernoulli_case = false
    nscen = 30  # Number of scenarios for sampling
    nsamples = 100  # Number of online samples
    test_samples = 100  # Number of test samples
    K = 5  # Number of solutions to select
    max_iter = 10  # Maximum iterations for heuristic
    
    # Check if file exists
    if !isfile(file_path)
        println("❌ Error: Data file not found!")
        println("Expected file: $file_path")
        println("Please make sure you're running from the project root directory")
        println("and that the instances/ folder contains the data files.")
        return
    end
    
    println("📁 Loading data from: $file_path")
    capacity, fixedcost, cost, demandmean = read_orlib_cap(file_path)
    nloc = length(fixedcost)
    ncust = length(demandmean)
    
    println("✅ Problem size: $nloc facilities, $ncust customers")
    
    # Generate demand standard deviations
    theta = rand(Uniform(0, 0.5), ncust)
    demandstdev = theta .* demandmean
    
    println("\n" * "="^40)
    println("PHASE 1: SCENARIO GENERATION")
    println("="^40)
    
    # Generate scenarios and solutions
    xsolutions = sampling(nscen, nloc, demandmean, demandstdev, capacity, cost, fixedcost, unmet_pen, scaling_factor, bernoulli_case)
    
    # Count unique solutions
    _, u1 = n_unique_rows(xsolutions)
    println("✅ Generated $nscen scenarios with $u1 unique solutions")
    
    # Generate online samples
    println("🔄 Generating $nsamples online samples...")
    online_samples = zeros(nsamples, ncust)
    for s in 1:nsamples
        online_samples[s, :] = generate_demand_scenario(ncust, demandmean, demandstdev, bernoulli_case)
    end
    
    # Generate test samples
    println("🔄 Generating $test_samples test samples...")
    test_samples_data = zeros(test_samples, ncust)
    for s in 1:test_samples
        test_samples_data[s, :] = generate_demand_scenario(ncust, demandmean, demandstdev, bernoulli_case)
    end
    
    println("\n" * "="^40)
    println("PHASE 2: REFORMULATED EXTENSIVE FORM")
    println("="^40)
    
    # Solve reformulated extensive form
    obj_matrix, r = Calculations_For_Extensive_Form(nloc, ncust, capacity, cost, online_samples, xsolutions, fixedcost, unmet_pen, scaling_factor)
    n_neg1 = count(x -> x == -1, obj_matrix)
    println("✅ Infeasible scenario-solution pairs: $n_neg1")
    
    Solution, z = Reformulated_Extensive_Form(nsamples, nscen, K, obj_matrix)
    println("✅ Reformulated extensive form objective: $Solution")
    
    println("\n" * "="^40)
    println("PHASE 3: EVALUATION (BEFORE HEURISTIC)")
    println("="^40)
    
    # Calculate offline costs
    offline_cost = OfflineTest(nloc, ncust, capacity, cost, test_samples_data, fixedcost, unmet_pen, scaling_factor)
    println("✅ Average offline cost: $offline_cost")
    
    # Evaluate model performance
    modelcost, pcos, mcost = Evaluation(nloc, ncust, cost, capacity, test_samples_data, xsolutions, z, fixedcost, unmet_pen, scaling_factor)
    println("✅ Average model cost: $modelcost")
    
    # Calculate gap
    gap = (modelcost - offline_cost) / offline_cost * 100
    println("✅ Performance gap: $(round(gap, digits=2))%")
    
    # Analyze penalties
    penalty_pos = findall(x -> x > 0.1, pcos)
    penalty_val = pcos[penalty_pos]
    println("✅ Scenarios with penalties: $(length(penalty_pos))")
    
    println("\n" * "="^40)
    println("PHASE 4: L-AUGMENTATION HEURISTIC")
    println("="^40)
    
    # Run L-Augmentation Heuristic
    candidates, znew, cnt = Augmentation_Heuristic(nloc, ncust, capacity, cost, online_samples, xsolutions, max_iter, K, fixedcost, unmet_pen, scaling_factor)
    println("✅ Heuristic completed in $cnt iterations")
    
    # Evaluate heuristic performance
    heuristiccost, p1, m1 = Evaluation(nloc, ncust, cost, capacity, test_samples_data, candidates, znew, fixedcost, unmet_pen, scaling_factor)
    println("✅ Average heuristic cost: $heuristiccost")
    
    # Calculate heuristic gap
    heuristic_gap = (heuristiccost - offline_cost) / offline_cost * 100
    println("✅ Heuristic performance gap: $(round(heuristic_gap, digits=2))%")
    
    println("\n" * "="^60)
    println("FINAL RESULTS SUMMARY")
    println("="^60)
    println("Offline optimal cost:     $(round(offline_cost, digits=2))")
    println("Model cost (before):      $(round(modelcost, digits=2))")
    println("Heuristic cost (after):   $(round(heuristiccost, digits=2))")
    println("Model gap:                $(round(gap, digits=2))%")
    println("Heuristic gap:            $(round(heuristic_gap, digits=2))%")
    println("Improvement:              $(round(gap - heuristic_gap, digits=2))%")
    println("Heuristic iterations:     $cnt")
    println("="^60)
    
    return offline_cost, modelcost, heuristiccost, gap, heuristic_gap, cnt
end

# Run the main function
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end



