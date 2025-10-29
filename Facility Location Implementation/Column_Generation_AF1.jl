# Column Generation Implementation - AF Formulation (Refactored)
# Self-contained Assignment Formulation with PowerFlow-style demand generation

using JuMP, Gurobi, Random, Distributions, DataFrames, LinearAlgebra

# Set random seed for reproducibility
Random.seed!(1234)

# Command line argument parsing
if length(ARGS) != 3
    println("Usage: julia CG_AF1.jl <K> <S> <variance>")
    println("Example: julia CG_AF1.jl 2 10 low")
    println("Variance options: low, high, bernoulli")
    exit(1)
end

K_input = parse(Int, ARGS[1])
S_input = parse(Int, ARGS[2])
variance_input = ARGS[3]

# Validate variance parameter
if !(variance_input in ["low", "high", "bernoulli"])
    println("❌ Error: Invalid variance parameter: $variance_input")
    println("Valid options: low, high, bernoulli")
    exit(1)
end

println("="^80)
println("COLUMN GENERATION AF - Assignment Formulation (Refactored)")
println("="^80)
println("Parameters: K=$K_input, S=$S_input, Variance=$variance_input")
println("Instance: cap91.txt")
println("Max Iterations: 100")
println("Pricing Type: 2 (Alternative Formulation)")
println("Solution Pool Limit: 10")
println("="^80)

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
# POWERFLOW-STYLE DEMAND GENERATION
# =============================================================================

function generate_demand_scenario_powerflow(ncust, demandmean, variance_type)
    """Generate demand scenario using PowerFlow-style formula with variance types"""
    
    # Set sigma range based on variance type
    if variance_type == "low"
        sigma = rand(Uniform(0.1, 0.2))
    elseif variance_type == "high"
        sigma = rand(Uniform(0.2, 0.3))
    elseif variance_type == "bernoulli"
        sigma = rand(Uniform(0.1, 0.2))
    else
        error("Invalid variance type: $variance_type")
    end
    
    # Generate global noise component
    global_noise = randn()
    
    # Generate individual noise components and apply PowerFlow formula
    scenario_demand = zeros(ncust)
    for j in 1:ncust
        individual_noise = randn()
        
        # PowerFlow-style formula: demand[i] = demandmean[i] + sigma * individual_noise * demandmean[i] + global_noise * sigma * demandmean[i]
        raw_demand = demandmean[j] + sigma * individual_noise * demandmean[j] + global_noise * sigma * demandmean[j]
        
        # Apply Bernoulli multiplier if bernoulli case
        if variance_type == "bernoulli"
            bernoulli_multiplier = rand(Bernoulli(0.5)) == 1 ? 1.0 : 0.0
            raw_demand *= bernoulli_multiplier
        end
        
        # Ensure non-negativity
        scenario_demand[j] = max(0.0, raw_demand)
    end
    
    return scenario_demand
end

# =============================================================================
# FACILITY LOCATION SOLVING FUNCTIONS
# =============================================================================

function solve_facility_location_for_binary(nloc, ncust, capacity, cost, demand, fixedcost, unmet_pen, scaling_factor)
    """Solve facility location problem and return binary solution vector x"""
    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 0)
    set_optimizer_attribute(model, "WorkLimit", 60)
    
    # Decision Variables
    @variable(model, x[1:nloc], Bin)  # Binary: facility open/closed
    @variable(model, y[1:nloc, 1:ncust] >= 0)  # Continuous: flow
    @variable(model, shortfall[1:ncust] >= 0)  # Continuous: unmet demand
    
    # Constraints
    @constraint(model, [j in 1:ncust], sum(y[i,j] for i in 1:nloc) + shortfall[j] == demand[j])
    @constraint(model, [i in 1:nloc], sum(y[i,j] for j in 1:ncust) - capacity[i]*x[i] <= 0)
    
    # Objective function
    @objective(model, Min, 
        sum(fixedcost[i]*x[i] for i in 1:nloc) + 
        scaling_factor * sum(cost[i,j]*y[i,j] for i in 1:nloc for j in 1:ncust) + 
        sum(unmet_pen*shortfall[j] for j in 1:ncust))
    
    optimize!(model)
    
    if termination_status(model) == MOI.OPTIMAL
        return value.(x), objective_value(model)
    else
        return zeros(nloc), Inf
    end
end

function calculate_scenario_solution_cost(nloc, ncust, capacity, cost, scenario_demand, binary_vector, fixedcost, unmet_pen, scaling_factor)
    """Calculate the cost r_s(x) of using binary vector x under scenario s"""
    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 0)
    
    # Decision Variables (x is fixed, only y and shortfall are variables)
    @variable(model, x[1:nloc], Bin)
    @variable(model, y[1:nloc, 1:ncust] >= 0)
    @variable(model, shortfall[1:ncust] >= 0)
    
    # Fix x to the given binary vector
    for i in 1:nloc
        fix(x[i], binary_vector[i])
    end
    
    # Constraints
    @constraint(model, [j in 1:ncust], sum(y[i,j] for i in 1:nloc) + shortfall[j] == scenario_demand[j])
    @constraint(model, [i in 1:nloc], sum(y[i,j] for j in 1:ncust) - capacity[i]*x[i] <= 0)
    
    # Objective function
    @objective(model, Min, 
        sum(fixedcost[i]*x[i] for i in 1:nloc) + 
        scaling_factor * sum(cost[i,j]*y[i,j] for i in 1:nloc for j in 1:ncust) + 
        sum(unmet_pen*shortfall[j] for j in 1:ncust))
    
    optimize!(model)
    
    if termination_status(model) == MOI.OPTIMAL
        return objective_value(model)
    else
        return Inf  # Return infinite cost if infeasible
    end
end

function build_cost_matrix(scenarios, binary_vectors, nloc, ncust, capacity, cost, fixedcost, unmet_pen, scaling_factor)
    """Build the cost matrix r_s(x) for Assignment Formulation"""
    nscen = length(scenarios)
    nsol = length(binary_vectors)
    
    println("Building cost matrix r_s(x) for $nscen scenarios and $nsol solutions...")
    
    cost_matrix = zeros(nscen, nsol)
    
    for s in 1:nscen
        for x_idx in 1:nsol
            cost_matrix[s, x_idx] = calculate_scenario_solution_cost(
                nloc, ncust, capacity, cost, scenarios[s], binary_vectors[x_idx], 
                fixedcost, unmet_pen, scaling_factor)
        end
        
        if s % 10 == 0
            println("  Processed $s scenarios")
        end
    end
    
    println("Cost matrix built successfully")
    return cost_matrix
end

# =============================================================================
# ASSIGNMENT FORMULATION FUNCTIONS
# =============================================================================

function solve_assignment_formulation(cost_matrix, K; integer_vars=false)
    """Solve the Assignment Formulation (AF) Master Problem from Image 1 - LINEAR RELAXATION"""
    nscen, nsol = size(cost_matrix)
    
    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 0)
    
    # Decision Variables - LINEAR RELAXATION (continuous, not binary) or INTEGER
    # Use vectors that can be dynamically extended
    if integer_vars
        sigma = [@variable(model, binary = true) for x in 1:nsol]  # σ_x: weight of solution x (binary)
        rho = [[@variable(model, binary = true) for x in 1:nsol] for s in 1:nscen]  # ρ_{s,x}: assignment of scenario s to solution x (binary)
    else
        sigma = [@variable(model, lower_bound = 0) for x in 1:nsol]  # σ_x: weight of solution x (continuous)
        rho = [[@variable(model, lower_bound = 0) for x in 1:nsol] for s in 1:nscen]  # ρ_{s,x}: assignment of scenario s to solution x (continuous)
    end
    
    # Objective Function (5a)
    @objective(model, Min, sum(cost_matrix[s, x] * rho[s][x] for s in 1:nscen for x in 1:nsol))
    
    # Constraints - Store references for dual extraction
    # Assignment constraints (5b): Each scenario must be assigned to exactly one solution
    assignment_constraints = @constraint(model, [s in 1:nscen], sum(rho[s][x] for x in 1:nsol) == 1)
    
    # Selection constraints (5c): ρ_{s,x} <= σ_x for all s, x
    @constraint(model, [s in 1:nscen, x in 1:nsol], rho[s][x] <= sigma[x])
    
    # K constraint (5d): Sum of σ_x <= K
    k_constraint = @constraint(model, sum(sigma[x] for x in 1:nsol) <= K)
    
    # Solve
    optimize!(model)
    
    if termination_status(model) == MOI.OPTIMAL
        return model, sigma, rho, objective_value(model), assignment_constraints, k_constraint
    else
        return model, sigma, rho, Inf, assignment_constraints, k_constraint
    end
end

# =============================================================================
# PRICING SUBPROBLEM FUNCTIONS
# =============================================================================

function big_m_calculator(nloc, ncust, capacity, cost, scenario_demand, fixedcost, unmet_pen, scaling_factor, dual_lambda)
    """Calculate m^s by solving Formulation 2 for a single scenario - Based on Image Formulation"""
    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 0)  # Disable Gurobi output
    set_optimizer_attribute(model, "LogToConsole", 0)  # Disable console logging
    set_optimizer_attribute(model, "MIPGap", 0.01)  # Set gap tolerance to 1%
    set_optimizer_attribute(model, "WorkLimit", 1800)  # Set 1800 work limit for pricing subproblem
    
    # Decision Variables 
    @variable(model, x[1:nloc], Bin)  # Binary: facility open/closed
    @variable(model, y[1:nloc, 1:ncust] >= 0)  # Continuous: flow
    @variable(model, s_j[1:ncust] >= 0)  # Continuous: unmet demand (shortfall)
    
    # Constraints 
    
    # Constraint 1: Demand satisfaction constraint
    @constraint(model, [j in 1:ncust], sum(y[i,j] for i in 1:nloc) + s_j[j] == scenario_demand[j])
    
    # Constraint 2: Facility capacity constraint
    @constraint(model, [i in 1:nloc], sum(y[i,j] for j in 1:ncust) - capacity[i]*x[i] <= 0)
    
    # Objective function (as per image: m^s = min { (λ^s)* - Σ f_i x_i - ΣΣ c_ij y_ij^s - Σ α s_j^s })
    @objective(model, Min, 
        dual_lambda - sum(fixedcost[i]*x[i] for i in 1:nloc) - 
        scaling_factor * sum(cost[i,j]*y[i,j] for i in 1:nloc for j in 1:ncust) - 
        sum(unmet_pen*s_j[j] for j in 1:ncust))
    
    optimize!(model)
    
    if termination_status(model) == MOI.OPTIMAL
        return objective_value(model)
    else
        return Inf  # Return infinite if infeasible
    end
end

function af_pricing_type1(nloc, ncust, capacity, cost, scenarios, m_values, dual_lambda, dual_mu, unmet_pen, scaling_factor; use_solution_pool::Bool=true, return_nodes::Bool=false, work_limit::Float64=Inf)
    """Solve AF Pricing Type 1 using m^s as parameters - Based on Updated Image Formulation"""
    nscen = length(scenarios)
    
    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 1)  # Disable Gurobi output
    set_optimizer_attribute(model, "LogToConsole", 1)  # Disable console logging
    set_optimizer_attribute(model, "MIPGap", 0.01)  # Set gap tolerance to 1%
    set_optimizer_attribute(model, "WorkLimit", 100)  # Set 1800 work limit for pricing subproblem
    if use_solution_pool
        set_optimizer_attribute(model, "PoolSearchMode", 2)
        set_optimizer_attribute(model, "PoolSolutions", 10)
    end
    
    # Decision Variables 
    @variable(model, v[1:nloc, 1:nscen], Bin)  # Binary: facility i open in scenario s
    @variable(model, pi[1:nscen], Bin)  # Binary: scenario selection
    @variable(model, x[1:nloc], Bin)  # Binary: facility open/closed (the solution vector we want)
    @variable(model, u[1:nloc, 1:ncust, 1:nscen] >= 0)  # Continuous: flow component
    @variable(model, y[1:nloc, 1:ncust, 1:nscen] >= 0)  # Continuous: total flow
    @variable(model, w[1:ncust, 1:nscen] >= 0)  # Continuous: auxiliary shortfall variable
    @variable(model, b[1:ncust, 1:nscen] >= 0)  # Continuous: shortfall decision variable
    
    # Pre-calculate N[i,j,s] = min(capacity[i], scenarios[s][j])
    N = Array{Float64}(undef, nloc, ncust, nscen)
    for i in 1:nloc, j in 1:ncust, s in 1:nscen
        N[i,j,s] = min(capacity[i], scenarios[s][j])
    end
    
    # Objective Function 
    @objective(model, Max, 
        sum(dual_lambda[s] * pi[s] for s in 1:nscen) - 
        sum(fixedcost[i] * v[i,s] for i in 1:nloc for s in 1:nscen) - 
        scaling_factor * sum(cost[i,j] * u[i,j,s] for i in 1:nloc for j in 1:ncust for s in 1:nscen) - 
        sum(unmet_pen * w[j,s] for j in 1:ncust for s in 1:nscen))
    
    # Constraints 
    # Constraint 1: Flow balance for each scenario (updated to use b_j^s)
    @constraint(model, [j in 1:ncust, s in 1:nscen], 
        sum(y[i,j,s] for i in 1:nloc) + b[j,s] == scenarios[s][j])
    
    # Constraint 2: Capacity constraints for each scenario
    @constraint(model, [i in 1:nloc, s in 1:nscen], 
        sum(y[i,j,s] for j in 1:ncust) - capacity[i] * x[i] <= 0)
    
    # Constraint 3: Key pricing constraint (updated to use w_j^s)
    @constraint(model, [s in 1:nscen], 
        dual_lambda[s] - sum(fixedcost[i] * x[i] for i in 1:nloc) - 
        scaling_factor * sum(cost[i,j] * y[i,j,s] for i in 1:nloc for j in 1:ncust) - 
        sum(unmet_pen * b[j,s] for j in 1:ncust) >= m_values[s] * (1 - pi[s]))
    
    # Constraint 4: v_i^s ≤ x_i
    @constraint(model, [i in 1:nloc, s in 1:nscen], v[i,s] <= x[i])
    
    # Constraint 5: v_i^s ≤ π^s
    @constraint(model, [i in 1:nloc, s in 1:nscen], v[i,s] <= pi[s])
    
    # Constraint 6: v_i^s ≥ x_i + π^s - 1
    @constraint(model, [i in 1:nloc, s in 1:nscen], v[i,s] >= x[i] + pi[s] - 1)
    
    # Constraint 7: u_ij^s ≤ y_ij^s
    @constraint(model, [i in 1:nloc, j in 1:ncust, s in 1:nscen], u[i,j,s] <= y[i,j,s])
    
    # Constraint 8: u_ij^s ≤ min{ū_i, d_j^s} π^s
    @constraint(model, [i in 1:nloc, j in 1:ncust, s in 1:nscen], u[i,j,s] <= N[i,j,s] * pi[s])
    
    # Constraint 9: u_ij^s ≥ y_ij^s + min{ū_i, d_j^s}(π^s - 1)
    @constraint(model, [i in 1:nloc, j in 1:ncust, s in 1:nscen], 
        u[i,j,s] >= y[i,j,s] + N[i,j,s] * (pi[s] - 1))
    
    # Constraint 10: w_j^s ≤ b_j^s
    @constraint(model, [j in 1:ncust, s in 1:nscen], w[j,s] <= b[j,s])
    
    # Constraint 11: w_j^s ≤ d_j^s π^s
    @constraint(model, [j in 1:ncust, s in 1:nscen], w[j,s] <= scenarios[s][j] * pi[s])
    
    # Constraint 12: w_j^s ≥ b_j^s + d_j^s (π^s - 1)
    @constraint(model, [j in 1:ncust, s in 1:nscen], 
        w[j,s] >= b[j,s] + scenarios[s][j] * (pi[s] - 1))
    
    # Solve
    optimize!(model)

    println("status = ", termination_status(model))
    println("obj    = ", objective_value(model))
    println("sum(pi)= ", sum(value.(pi)))
    println("any x? = ", sum(value.(x)) > 0)
    
    # Get work units
    pricing_work_units = 0.0
    try
        pricing_work_units = MOI.get(backend(model), Gurobi.ModelAttribute("Work"))
    catch e
        println("Error getting pricing work units: $e")
    end
    
    ts = termination_status(model)
    if !(ts == MOI.OPTIMAL || ts == MOI.TIME_LIMIT || ts == MOI.OTHER_ERROR)
        if return_nodes
            return [zeros(nloc)], [Inf], pricing_work_units, 0
        else
            return [zeros(nloc)], [Inf], pricing_work_units
        end
    end

    if !use_solution_pool
        if return_nodes
            return [value.(x)], [objective_value(model)], pricing_work_units, 0
        else
            return [value.(x)], [objective_value(model)], pricing_work_units
        end
    end

    # Retrieve solution pool and filter by reduced cost criterion rc = obj + dual_mu > 0
    opt = backend(model)
    kept_solutions = Vector{Vector{Float64}}()
    kept_objs = Float64[]
    try
        solcount = MOI.get(opt, Gurobi.ModelAttribute("SolCount"))
        println("  Pricing pool: found $(solcount) solutions")
        
        # Limit to best 10 solutions as requested
        max_solutions = min(10, solcount)
        
        for k in 0:max_solutions-1
            MOI.set(opt, MOI.RawOptimizerAttribute("SolutionNumber"), k)
            pool_obj = MOI.get(opt, Gurobi.ModelAttribute("PoolObjVal"))
            x_k = [MOI.get(opt, Gurobi.VariableAttribute("Xn"), JuMP.index(x[i])) for i in 1:nloc]
            rc_k = pool_obj + dual_mu
            if rc_k > 0
                push!(kept_solutions, x_k)
                push!(kept_objs, pool_obj)
            end
        end
        println("  Pricing pool: kept $(length(kept_solutions)) with rc>0 (limited to best 10)")
    catch e
        println("  Warning: solution pool access failed: $e")
    end

    # Get B&B nodes if requested
    nodes = 0
    if return_nodes
        try
            nodes = MOI.get(backend(model), Gurobi.ModelAttribute("NodeCount"))
        catch e
            println("Warning: Could not get B&B nodes: $e")
        end
    end
    
    if isempty(kept_solutions)
        if return_nodes
            return [value.(x)], [objective_value(model)], pricing_work_units, nodes
        else
            return [value.(x)], [objective_value(model)], pricing_work_units
        end
    end
    
    if return_nodes
        return kept_solutions, kept_objs, pricing_work_units, nodes
    else
        return kept_solutions, kept_objs, pricing_work_units
    end
end

function af_pricing_type2(nloc, ncust, capacity, cost, scenarios, m_values, dual_lambda, dual_mu, unmet_pen, scaling_factor; use_solution_pool::Bool=true, return_nodes::Bool=false, work_limit::Float64=Inf)
    """Solve AF Pricing Type 2 using alternative linearization technique - Based on New Image Formulation"""
    nscen = length(scenarios)
    
    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 1)  # Disable Gurobi output
    set_optimizer_attribute(model, "LogToConsole", 1)  # Disable console logging
    #set_optimizer_attribute(model, "MIPGap", 0.01)  # Set gap tolerance to 1%
    if isfinite(work_limit) && work_limit > 0
        set_optimizer_attribute(model, "WorkLimit", work_limit)
    end
    if use_solution_pool
        set_optimizer_attribute(model, "PoolSearchMode", 2)
        set_optimizer_attribute(model, "PoolSolutions", 10)  # Find up to 50 solutions
    end
    
    # Decision Variables
    @variable(model, x[1:nloc], Bin)  # Binary: first-stage facility decisions
    @variable(model, x_s[1:nloc, 1:nscen], Bin)  # Binary: second-stage facility decisions
    @variable(model, y[1:nloc, 1:ncust, 1:nscen] >= 0)  # Continuous: flow
    @variable(model, b[1:ncust, 1:nscen] >= 0)  # Continuous: shortfall
    @variable(model, pi[1:nscen], Bin)  # Binary: scenario selection
    
    # Objective Function (as per new image: MAXIMIZATION)
    @objective(model, Max, 
        sum(dual_lambda[s] * pi[s] for s in 1:nscen) - 
        sum(fixedcost[i] * x_s[i,s] for i in 1:nloc for s in 1:nscen) - 
        scaling_factor * sum(cost[i,j] * y[i,j,s] for i in 1:nloc for j in 1:ncust for s in 1:nscen) - 
        sum(unmet_pen * b[j,s] for j in 1:ncust for s in 1:nscen))
    
    # Constraint 1: x^s ≤ e π^s for all s ∈ S (e is vector of ones)
    @constraint(model, [i in 1:nloc, s in 1:nscen], x_s[i,s] <= pi[s])
    
    # Constraint 2: x - x^s ≤ e(1 - π^s) for all s ∈ S (e is vector of ones)
    @constraint(model, [i in 1:nloc, s in 1:nscen], x[i] - x_s[i,s] <= (1 - pi[s]))
    
    # Constraint 3: x - x^s ≥ -e(1 - π^s) for all s ∈ S (e is vector of ones)
    @constraint(model, [i in 1:nloc, s in 1:nscen], x[i] - x_s[i,s] >= -(1 - pi[s]))
    
    # Constraint 4: (λ*)^s π^s - Σ f_i x_i^s - ΣΣ c_ij y_ij^s - Σ α b_j^s ≥ 0 for all s ∈ S
    @constraint(model, [s in 1:nscen], 
        dual_lambda[s] * pi[s] - sum(fixedcost[i] * x_s[i,s] for i in 1:nloc) - 
        scaling_factor * sum(cost[i,j] * y[i,j,s] for i in 1:nloc for j in 1:ncust) - 
        sum(unmet_pen * b[j,s] for j in 1:ncust) >= 0)
    
    # Constraint 5: Σ y_ij^s - u_i x_i^s ≤ 0 for all i ∈ I, s ∈ S
    @constraint(model, [i in 1:nloc, s in 1:nscen], 
        sum(y[i,j,s] for j in 1:ncust) - capacity[i] * x_s[i,s] <= 0)
    
    # Constraint 6: Σ y_ij^s + b_j^s = d_j^s π^s for all j ∈ J, s ∈ S
    @constraint(model, [j in 1:ncust, s in 1:nscen], 
        sum(y[i,j,s] for i in 1:nloc) + b[j,s] == scenarios[s][j] * pi[s])
    
    # Solve
    optimize!(model)

    println("status = ", termination_status(model))
    println("obj    = ", objective_value(model))
    
    # Get work units
    pricing_work_units = 0.0
    try
        pricing_work_units = MOI.get(backend(model), Gurobi.ModelAttribute("Work"))
    catch e
        println("Error getting pricing work units: $e")
    end
    
    ts = termination_status(model)
    if !(ts == MOI.OPTIMAL || ts == MOI.TIME_LIMIT || ts == MOI.OTHER_ERROR)
        if return_nodes
            return [zeros(nloc)], [Inf], pricing_work_units, 0
        else
            return [zeros(nloc)], [Inf], pricing_work_units
        end
    end

    # Get B&B nodes if requested
    nodes = 0
    if return_nodes
        try
            nodes = MOI.get(backend(model), Gurobi.ModelAttribute("NodeCount"))
        catch e
            println("Warning: Could not get B&B nodes: $e")
        end
    end

    if !use_solution_pool
        if return_nodes
            return [value.(x)], [objective_value(model)], pricing_work_units, nodes
        else
            return [value.(x)], [objective_value(model)], pricing_work_units
        end
    end

    opt = backend(model)
    kept_solutions = Vector{Vector{Float64}}()
    kept_objs = Float64[]
    try
        solcount = MOI.get(opt, Gurobi.ModelAttribute("SolCount"))
        println("  Pricing pool: found $(solcount) solutions")
        
        # Limit to best 10 solutions as requested
        max_solutions = min(10, solcount)
        
        for k in 0:max_solutions-1
            MOI.set(opt, MOI.RawOptimizerAttribute("SolutionNumber"), k)
            pool_obj = MOI.get(opt, Gurobi.ModelAttribute("PoolObjVal"))
            x_k = [MOI.get(opt, Gurobi.VariableAttribute("Xn"), JuMP.index(x[i])) for i in 1:nloc]
            rc_k = pool_obj + dual_mu
            if rc_k > 0
                push!(kept_solutions, x_k)
                push!(kept_objs, pool_obj)
            end
        end
        println("  Pricing pool: kept $(length(kept_solutions)) with rc>0 (limited to best 10)")
    catch e
        println("  Warning: solution pool access failed: $e")
    end

    if isempty(kept_solutions)
        if return_nodes
            return [value.(x)], [objective_value(model)], pricing_work_units, nodes
        else
            return [value.(x)], [objective_value(model)], pricing_work_units
        end
    end
    
    if return_nodes
        return kept_solutions, kept_objs, pricing_work_units, nodes
    else
        return kept_solutions, kept_objs, pricing_work_units
    end
end

function solve_pricing_subproblem(nloc, ncust, capacity, cost, scenarios, dual_lambda, dual_mu, unmet_pen, scaling_factor, pricing_type=2; use_solution_pool::Bool=true, work_limit::Float64=Inf)
    """Solve the complete pricing subproblem with choice of formulation type"""
    nscen = length(scenarios)
    
    if pricing_type == 1
        # Step 1: Calculate m^s for all scenarios using Formulation 2
        println("  Calculating m^s values for all scenarios...")
        m_values = zeros(nscen)
        
        for s in 1:nscen
            m_values[s] = big_m_calculator(nloc, ncust, capacity, cost, scenarios[s], fixedcost, unmet_pen, scaling_factor, dual_lambda[s])
        end

        # Debug: Print m values range
        println("  DEBUG: m^s values calculated:")
        println("    Min m^s: $(round(minimum(m_values), digits=6))")
        println("    Max m^s: $(round(maximum(m_values), digits=6))")
        println("    Mean m^s: $(round(mean(m_values), digits=6))")
        println("    All m^s values: $(round.(m_values, digits=6))")
        
        # Step 2: Solve AF Pricing Type 1 using m^s as parameters
        println("  Solving AF Pricing Type 1...")
        sols, objs, pricing_work_units = af_pricing_type1(nloc, ncust, capacity, cost, scenarios, m_values, dual_lambda, dual_mu, unmet_pen, scaling_factor; use_solution_pool=use_solution_pool, work_limit=work_limit)
        return sols, objs, m_values, pricing_work_units
        
    elseif pricing_type == 2
        # Solve AF Pricing Type 2 (alternative formulation - no m^s calculation needed)
        println("  Solving AF Pricing Type 2 (Alternative Formulation)...")
        m_values = zeros(nscen)  # Not used in Type 2, but kept for compatibility
        sols, objs, pricing_work_units = af_pricing_type2(nloc, ncust, capacity, cost, scenarios, m_values, dual_lambda, dual_mu, unmet_pen, scaling_factor; use_solution_pool=use_solution_pool, work_limit=work_limit)
        return sols, objs, m_values, pricing_work_units
        
    else
        error("Invalid pricing_type. Use 1 for original formulation or 2 for alternative formulation.")
    end
end

# =============================================================================
# MAIN COLUMN GENERATION ALGORITHM
# =============================================================================

function column_generation_algorithm(scenarios, nloc, ncust, capacity, cost, fixedcost, unmet_pen, scaling_factor, K, max_iterations, pricing_type=2, init_type=1; use_solution_pool::Bool=true)
    """Main Column Generation Algorithm with 5 stages"""
    nscen = length(scenarios)
    
    println(" Starting Column Generation Algorithm")
    println("Parameters: nscen=$nscen, nloc=$nloc, ncust=$ncust, K=$K, max_iter=$max_iterations")

    # Global work budget tracker
    total_work_budget = 10000.0
    total_master_work_units = 0.0
    total_pricing_work_units = 0.0
    total_update_work_units = 0.0
    println()
    
    # Initialize pricing + dual_mu tracking
    iteration_pricing_data = []
    iteration_master_data = []  # Track master problem objectives
    iteration_columns_data = [] # Track number of columns added each iteration
    actual_iterations = 0  # Track actual number of iterations completed
    
    # Initialize subproblem comparison tracking (for first 10 iterations)
    type1_metrics = []  # Store (work_units, nodes) for Type 1 subproblem
    type2_metrics = []  # Store (work_units, nodes) for Type 2 subproblem
    
    # Initialize work units tracking
    total_master_work_units = 0.0
    total_pricing_work_units = 0.0
    total_update_work_units = 0.0
    total_solutions_added = 0
    
    # Stage 1: Initialization
    println(" Stage 1: Initialization")
    
    if init_type == 1
        # Option 1: Solve all scenarios from scratch
        println("  Initialization Type 1: Solving all scenarios from scratch...")
        initial_binary_vectors = []
        initial_objective_values = []
        
        for s in 1:nscen
            println("    Solving scenario $s/$nscen...")
            x_optimal, obj_optimal = solve_facility_location_for_binary(nloc, ncust, capacity, cost, scenarios[s], fixedcost, unmet_pen, scaling_factor)
            push!(initial_binary_vectors, x_optimal)
            push!(initial_objective_values, obj_optimal)
        end
        
        nsol = length(initial_binary_vectors)
        println("  Generated $nsol initial binary vectors from scenario solutions")
    else
        # Option 2: Start with zero vector (default behavior)
        println("  Initialization Type 2: Starting with zero vector...")
        initial_binary_vectors = [zeros(nloc)]
        initial_objective_values = [0.0]  # Dummy objective value
        
        nsol = length(initial_binary_vectors)
        println("  Generated $nsol initial binary vector (all zeros)")
    end
    
    # Build initial cost matrix
    cost_matrix = build_cost_matrix(scenarios, initial_binary_vectors, nloc, ncust, capacity, cost, fixedcost, unmet_pen, scaling_factor)
    
    # Initialize master problem
    model, sigma, rho, master_obj, assignment_constraints, k_constraint = solve_assignment_formulation(cost_matrix, K)
    
    if master_obj == Inf
        println(" Error: Initial master problem is infeasible!")
        return [], Inf, 0, "FAILED", [], [], [], [], Inf, [], [], 0.0, 0.0, 0.0, 0
    end
    
    println("  Initial master problem objective: $master_obj")
    println()
    
    # Column Generation Loop - Following JuMP pattern
    for iteration in 1:max_iterations
        println("\033[34m================ ITERATION ", iteration, " ===================\033[0m")
        
        # Print available work units at start of iteration
        used_work = total_master_work_units + total_pricing_work_units + total_update_work_units
        available_work = max(0.0, total_work_budget - used_work)
        println(" Available work units: $(round(available_work, digits=2)) / $(total_work_budget)")
        
        # Stage 2: Master Problem Solution
        println(" Stage 2: Solving Master Problem")
        
        # Track work units for master problem
        master_work_units = 0.0
        try
            optimize!(model)
            master_work_units = MOI.get(backend(model), Gurobi.ModelAttribute("Work"))
            total_master_work_units += master_work_units
        catch e
            println("Error getting master work units: $e")
        end
        
        if termination_status(model) != MOI.OPTIMAL
            println("Master problem failed to solve")
            break
        end
        
        master_obj = objective_value(model)
        println("  Master objective: $master_obj")
        println("  Master work units: $master_work_units")
        
        # Store master problem objective for analysis
        push!(iteration_master_data, (iteration = iteration, master_obj = master_obj))
        
        # Stage 3: Dual Extraction - Following JuMP pattern
        println("Stage 3: Extracting Dual Solutions")
        
        # Extract dual values using JuMP pattern
        dual_lambda = dual.(assignment_constraints)  # Extract duals from assignment constraints
        dual_mu = dual(k_constraint)  # Extract dual from K constraint
        
        println("  Dual μ value: $dual_mu")
        
        # Stage 4: Pricing Subproblem
        println(" Stage 4: Solving Pricing Subproblem")
        # Compute remaining work budget before pricing
        used_work = total_master_work_units + total_pricing_work_units + total_update_work_units
        remaining_work = max(0.0, total_work_budget - used_work)
        println("  Remaining work budget before pricing: $(round(remaining_work, digits=2))")
        if remaining_work <= 0.0
            println("  Global work budget exhausted. Terminating.")
            break
        end
        
        # Declare variables before try block
        new_solutions = []
        pool_objs = []
        m_values = []
        pricing_work_units = 0.0
        
        # Special handling for first 10 iterations: solve both subproblems for comparison
        if iteration <= 10
            println("  Solving both Type 1 and Type 2 subproblems for comparison (iteration $iteration/10)")
            
            try
                # Calculate m^s values for Type 1 (same as if using Type 1 as main algorithm)
                println("  Calculating m^s values for all scenarios...")
                m_values = zeros(nscen)
                for s in 1:nscen
                    m_values[s] = big_m_calculator(nloc, ncust, capacity, cost, scenarios[s], fixedcost, unmet_pen, scaling_factor, dual_lambda[s])
                end
                
                
                # Solve Type 1 subproblem (for comparison only)
                type1_solutions, type1_objs, type1_work_units, type1_nodes = af_pricing_type1(nloc, ncust, capacity, cost, scenarios, m_values, dual_lambda, dual_mu, unmet_pen, scaling_factor; use_solution_pool=use_solution_pool, return_nodes=true, work_limit=remaining_work)
                
                # Solve Type 2 subproblem (for algorithm)
                type2_solutions, type2_objs, type2_work_units, type2_nodes = af_pricing_type2(nloc, ncust, capacity, cost, scenarios, zeros(nscen), dual_lambda, dual_mu, unmet_pen, scaling_factor; use_solution_pool=use_solution_pool, return_nodes=true, work_limit=remaining_work)
                
                # Use Type 2 solutions for the algorithm (filter rc and cap 10)
                new_solutions = type2_solutions
                pool_objs = type2_objs
                rc_ok_indices = [idx for idx in 1:length(pool_objs) if pool_objs[idx] + dual_mu < -1e-6]
                if !isempty(rc_ok_indices)
                    keep = rc_ok_indices[1:min(10, length(rc_ok_indices))]
                    new_solutions = new_solutions[keep]
                    pool_objs = pool_objs[keep]
                else
                    keep_n = min(10, length(pool_objs))
                    new_solutions = new_solutions[1:keep_n]
                    pool_objs = pool_objs[1:keep_n]
                end
                m_values = zeros(nscen)  # Not used in Type 2
                pricing_work_units = type2_work_units
                
                # Track metrics for comparison
                push!(type1_metrics, (work_units=type1_work_units, nodes=type1_nodes))
                push!(type2_metrics, (work_units=type2_work_units, nodes=type2_nodes))
                
                total_pricing_work_units += pricing_work_units
                println("  Type 1 work units: $type1_work_units, B&B nodes: $type1_nodes")
                println("  Type 2 work units: $type2_work_units, B&B nodes: $type2_nodes")
                println("  Using Type 2 solutions for algorithm")
                
            catch e
                println("Error in subproblem comparison: $e")
                break
            end
        else
            # After 10 iterations, use normal approach (always Type 2)
            try
                new_solutions, pool_objs, m_values, pricing_work_units = solve_pricing_subproblem(nloc, ncust, capacity, cost, scenarios, dual_lambda, dual_mu, unmet_pen, scaling_factor, 2; use_solution_pool=use_solution_pool, work_limit=remaining_work)
                # Filter by reduced-cost condition and cap to 10
                rc_ok_indices = [idx for idx in 1:length(pool_objs) if pool_objs[idx] + dual_mu > -1e-6]
                if !isempty(rc_ok_indices)
                    keep = rc_ok_indices[1:min(10, length(rc_ok_indices))]
                    new_solutions = new_solutions[keep]
                    pool_objs = pool_objs[keep]
                else
                    # If none satisfy reduced cost, keep whatever we found (at most 10) per your guidance
                    keep_n = min(10, length(pool_objs))
                    new_solutions = new_solutions[1:keep_n]
                    pool_objs = pool_objs[1:keep_n]
                end
                total_pricing_work_units += pricing_work_units
                println("  Pricing work units: $pricing_work_units")
            catch e
                println("Error in pricing subproblem: $e")
                break
            end
        end
        
        # Normalize best pricing objective for logging and convergence
        obj_pricing = minimum(pool_objs)
        if isempty(pool_objs) || obj_pricing == Inf
            println(" Pricing subproblem failed to solve")
            break
        end
        
        println("  Pricing objective: $obj_pricing")
        # If any pricing solve hit limit, print its gap if available
        try
            gap_t2 = MOI.get(backend(model_pricing_type2), Gurobi.ModelAttribute("MIPGap"))
            ts_t2 = termination_status(model_pricing_type2)
            if ts_t2 == MOI.TIME_LIMIT
                println("  NOTE Type 2: Hit Work/Time limit. Reported MIP gap: $(round(gap_t2*100, digits=2))%")
            end
        catch
        end
        
        # Stage 5: Convergence Check & Master Problem Update
        println(" Stage 5: Convergence Check & Master Problem Update")
        
        # Check convergence: if obj_pricing + dual_mu < 0, we're done
        reduced_cost = obj_pricing + dual_mu
        println("  Reduced cost (obj_pricing + μ): $reduced_cost")
        
        # Store pricing + dual_mu data for table (before potential break)
        push!(iteration_pricing_data, (iteration = iteration, obj_pricing = obj_pricing, dual_mu = dual_mu, obj_pricing_plus_dual_mu = obj_pricing + dual_mu))
        
        if reduced_cost < 0.001  # Small tolerance for numerical issues
            println(" Algorithm converged! Reduced cost < 0")
            # No columns added on this iteration
            push!(iteration_columns_data, (iteration = iteration, num_added = 0))
            break
        else
            if use_solution_pool
                println(" Adding $(length(new_solutions)) column(s) from pricing pool")
                local_added = 0
                update_start_time = time()
                for x_new in new_solutions
                    push!(initial_binary_vectors, x_new)
                    nsol += 1
                    new_column_costs = zeros(nscen)
                    for s in 1:nscen
                        new_column_costs[s] = calculate_scenario_solution_cost(nloc, ncust, capacity, cost, scenarios[s], x_new, fixedcost, unmet_pen, scaling_factor)
                    end
                    new_sigma = @variable(model, lower_bound = 0)
                    new_rho = [@variable(model, lower_bound = 0) for s in 1:nscen]
                    push!(sigma, new_sigma)
                    for s in 1:nscen
                        push!(rho[s], new_rho[s])
                    end
                    for s in 1:nscen
                        set_objective_coefficient(model, new_rho[s], new_column_costs[s])
                        set_normalized_coefficient(assignment_constraints[s], new_rho[s], 1.0)
                        @constraint(model, new_rho[s] <= new_sigma)
                    end
                    set_normalized_coefficient(k_constraint, new_sigma, 1.0)
                    cost_matrix = hcat(cost_matrix, new_column_costs)
                    local_added += 1
                end
                update_time = time() - update_start_time
                total_update_work_units += update_time  # Using time as proxy for update work units
                total_solutions_added += local_added
                println(" Total solutions: $nsol")
                println(" Update time: $update_time seconds")
                push!(iteration_columns_data, (iteration = iteration, num_added = local_added))
            else
                println(" Adding new column to master problem")
                x_new = new_solutions[1]
                push!(initial_binary_vectors, x_new)
                nsol += 1
                new_column_costs = zeros(nscen)
                for s in 1:nscen
                    new_column_costs[s] = calculate_scenario_solution_cost(nloc, ncust, capacity, cost, scenarios[s], x_new, fixedcost, unmet_pen, scaling_factor)
                end
                new_sigma = @variable(model, lower_bound = 0)
                new_rho = [@variable(model, lower_bound = 0) for s in 1:nscen]
                push!(sigma, new_sigma)
                for s in 1:nscen
                    push!(rho[s], new_rho[s])
                end
                for s in 1:nscen
                    set_objective_coefficient(model, new_rho[s], new_column_costs[s])
                    set_normalized_coefficient(assignment_constraints[s], new_rho[s], 1.0)
                    @constraint(model, new_rho[s] <= new_sigma)
                end
                set_normalized_coefficient(k_constraint, new_sigma, 1.0)
                cost_matrix = hcat(cost_matrix, new_column_costs)
                println("New column added. Total solutions: $nsol")
                push!(iteration_columns_data, (iteration = iteration, num_added = 1))
            end
        end
        
        # Increment actual iterations counter
        actual_iterations = iteration
        
        println()
    end
    
    # Final solution
    optimize!(model)
    if termination_status(model) == MOI.OPTIMAL
        final_sigma_values = value.(sigma)
        final_rho_values = [value.(rho[s]) for s in 1:nscen]
        final_objective = objective_value(model)
        
        # Find selected solutions
        selected_solutions = findall(x -> x > 1e-6, final_sigma_values)
        
        println("  Final Results:")
        println("  Iterations: $actual_iterations")
        println("  Final objective: $final_objective")
        println("  Selected solutions: $selected_solutions")
        
        # Solve integer version of the master problem
        println("\n=== Solving Integer Master Problem ===")
        integer_model, integer_sigma, integer_rho, integer_obj, integer_assignment_constraints, integer_k_constraint = 
            solve_assignment_formulation(cost_matrix, K; integer_vars=true)
        
        if termination_status(integer_model) == MOI.OPTIMAL
            integer_sigma_values = value.(integer_sigma)
            integer_rho_values = [value.(integer_rho[s]) for s in 1:nscen]
            integer_selected_solutions = findall(x -> x > 1e-6, integer_sigma_values)
            
            println("  Integer Master Problem Results:")
            println("  Integer objective: $integer_obj")
            println("  Integer selected solutions: $integer_selected_solutions")
            println("  Gap (Integer - Linear): $(integer_obj - final_objective)")
            println("  Gap percentage: $((integer_obj - final_objective) / final_objective * 100)%")
            
            # Return integer selected solutions instead of linear relaxation ones
            return integer_selected_solutions, final_objective, actual_iterations, "OPTIMAL", iteration_pricing_data, initial_binary_vectors, cost_matrix, sigma, integer_obj, iteration_master_data, iteration_columns_data, total_master_work_units, total_pricing_work_units, total_update_work_units, total_solutions_added, type1_metrics, type2_metrics
        else
            println("  Integer master problem failed to solve")
            integer_obj = Inf
            # If integer solve fails, return linear relaxation solutions
            return selected_solutions, final_objective, actual_iterations, "OPTIMAL", iteration_pricing_data, initial_binary_vectors, cost_matrix, sigma, integer_obj, iteration_master_data, iteration_columns_data, total_master_work_units, total_pricing_work_units, total_update_work_units, total_solutions_added, type1_metrics, type2_metrics
        end
    else
        println(" Final master problem failed to solve")
        return [], Inf, actual_iterations, "FAILED", iteration_pricing_data, [], [], [], Inf, iteration_master_data, iteration_columns_data, total_master_work_units, total_pricing_work_units, total_update_work_units, total_solutions_added, type1_metrics, type2_metrics
    end
end

# =============================================================================
# MAIN EXECUTION
# =============================================================================

# Load data and generate scenarios
println(" Loading data and generating scenarios...")

# Load instance data
file_path = "cap91.txt"
if !isfile(file_path)
    println("❌ Error: Data file not found!")
    println("Expected file: $file_path")
    exit(1)
end

# Read instance data
capacity, fixedcost, cost, demandmean = read_orlib_cap(file_path)

# Get dimensions
nloc = length(capacity)
ncust = length(demandmean)

# Parameters
unmet_pen = 15.0
scaling_factor = 0.2
capacity_factor = 0.3

# Apply capacity factor to scale all facility capacities
capacity = capacity .* capacity_factor

println("  Instance loaded: $nloc facilities, $ncust customers")
println("  Applied capacity factor of $capacity_factor to all facilities")
println("  Capacity range after scaling: min=$(round(minimum(capacity), digits=0)), max=$(round(maximum(capacity), digits=0))")
println("  Generating $S_input scenarios with variance type: $variance_input")

# Generate scenarios using PowerFlow-style formula
scenarios = []
for i in 1:S_input
    scenario_demand = generate_demand_scenario_powerflow(ncust, demandmean, variance_input)
    push!(scenarios, scenario_demand)
end

println("  Generated $(length(scenarios)) scenarios successfully")

# Parameters from command line arguments
K_test = K_input      # Number of solutions to select
max_iter_test = 100  # Maximum iterations for testing
pricing_type_test = 1  # Pricing formulation type: 1 = original, 2 = alternative
init_type_test = 1  # Initialization type: 1 = solve all scenarios, 2 = start with zero vector
use_solution_pool_flag = 1  # Toggle multi-column generation: 1 = ON (true), 0 = OFF (false)
use_solution_pool_test = (use_solution_pool_flag == 1)

println(" Testing Column Generation Algorithm")
println("Test Parameters:")
println("  nscen: $S_input")
println("  K: $K_test")
println("  max_iterations: $max_iter_test")
println("  pricing_type: $pricing_type_test")
println("  init_type: $init_type_test")
println("  use_solution_pool (flag=$use_solution_pool_flag): $use_solution_pool_test")
println()

# Run Column Generation Algorithm
selected_solutions, final_objective, iterations, status, iteration_pricing_data, binary_vectors, cost_matrix, sigma, integer_obj, iteration_master_data, iteration_columns_data, total_master_work_units, total_pricing_work_units, total_update_work_units, total_solutions_added, type1_metrics, type2_metrics = column_generation_algorithm(
    scenarios, nloc, ncust, capacity, cost, fixedcost, 
    unmet_pen, scaling_factor, K_test, max_iter_test, pricing_type_test, init_type_test; use_solution_pool=use_solution_pool_test)

# Extract master objectives for iteration tracking
master_objectives = [data.master_obj for data in iteration_master_data]

println()
println("  Column Generation Results:")
println("  Iterations: $iterations")
println("  Final objective (Linear): $final_objective")
if integer_obj != Inf
    println("  Final objective (Integer): $integer_obj")
    println("  Gap: $(integer_obj - final_objective)")
    println("  Gap percentage: $((integer_obj - final_objective) / final_objective * 100)%")
    println("  Selected solutions (Integer): $selected_solutions")
else
    println("  Selected solutions (Linear): $selected_solutions")
end

# Calculate total work units
total_work_units = total_master_work_units + total_pricing_work_units + total_update_work_units

# Calculate gap if algorithm failed to converge
gap_percentage = 0.0
if status != "OPTIMAL" && integer_obj != Inf && final_objective != Inf
    gap_percentage = (integer_obj - final_objective) / final_objective * 100
end

# Generate output file with enhanced format
output_filename = "CG_AF1_cap91_K$(K_test)_S$(S_input)_$(variance_input)_results.txt"
open(output_filename, "w") do io
    println(io, "cap91.txt\t$nloc\t$ncust\t$K_test\t$S_input\t$iterations\t$total_work_units\t$total_solutions_added\t$status\t$(round(gap_percentage, digits=2))\t$(round(total_master_work_units, digits=2))\t$(round(total_pricing_work_units, digits=2))\t$(round(total_update_work_units, digits=2))\t$(round(final_objective, digits=2))\t$(round(integer_obj, digits=2))")
end

# Generate iteration tracking file
iteration_filename = "CG_AF1_cap91_K$(K_test)_S$(S_input)_$(variance_input)_iterations.txt"
open(iteration_filename, "w") do io
    for i in 1:iterations
        println(io, "$i\t$(round(master_objectives[i], digits=2))")
    end
end

# Generate subproblem comparison file
comparison_filename = "CG_AF1_cap91_K$(K_test)_S$(S_input)_$(variance_input)_subproblem_comparison.txt"
open(comparison_filename, "w") do io
    if !isempty(type1_metrics) && !isempty(type2_metrics)
        # Calculate averages manually
        avg_type1_work = sum([m.work_units for m in type1_metrics]) / length(type1_metrics)
        avg_type1_nodes = sum([m.nodes for m in type1_metrics]) / length(type1_metrics)
        avg_type2_work = sum([m.work_units for m in type2_metrics]) / length(type2_metrics)
        avg_type2_nodes = sum([m.nodes for m in type2_metrics]) / length(type2_metrics)
        
        println(io, "cap91.txt\t$S_input\t$K_test\t$(round(avg_type1_work, digits=3))\t$(round(avg_type1_nodes, digits=3))\t$(round(avg_type2_work, digits=3))\t$(round(avg_type2_nodes, digits=3))")
    else
        println(io, "cap91.txt\t$S_input\t$K_test\t0.000\t0.000\t0.000\t0.000")
    end
end

println()
println("="^80)
println("COLUMN GENERATION SUMMARY")
println("="^80)
println("Instance: cap91.txt")
println("Facilities: $nloc")
println("Customers: $ncust")
println("K: $K_test")
println("S: $S_input")
println("Variance: $variance_input")
println("Iterations: $iterations")
println("Status: $status")
println("Total Work Units: $total_work_units")
println("  - Master Problem: $total_master_work_units")
println("  - Pricing Subproblem: $total_pricing_work_units")
println("  - Master Update: $total_update_work_units")
println("Total Solutions Added: $total_solutions_added")
if gap_percentage > 0
    println("Gap (when failed to converge): $(round(gap_percentage, digits=2))%")
end
println("Output file: $output_filename")
println("="^80)
