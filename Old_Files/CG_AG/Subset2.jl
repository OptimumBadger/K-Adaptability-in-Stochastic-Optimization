using JuMP, Gurobi, Random, Distributions, LinearAlgebra

# Include required dependencies for candidate-based initialization
include("FL1.jl")

# Set random seed for reproducibility
Random.seed!(1234)

# Two-phase pricing parameters
const DEFAULT_OPT_WORKLIMIT = 300.0

# Sanitize helper: replace non-finite costs with a large finite penalty
sanitize_cost(c::Float64) = isfinite(c) ? c : 1.0e12

# Lightweight local replacements for mean/std to avoid using Statistics
mean_simple(v) = sum(v) / length(v)
function std_simple(v)
    n = length(v)
    if n <= 1
        return 0.0
    end
    m = mean_simple(v)
    s = 0.0
    for x in v
        s += (x - m) ^ 2
    end
    return sqrt(s / (n - 1))
end

# Command line argument parsing
if length(ARGS) != 3
    println("Usage: julia Subset.jl <K> <S> <variance>")
    println("Example: julia Subset.jl 3 5 bernoulli")
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
println("COLUMN GENERATION SF - Subset Formulation")
println("="^80)
println("Parameters: K=$K_input, S=$S_input, Variance=$variance_input")
println("Instance: capa11.txt")
println("Max Iterations: 100")
println("Pricing Type: 2 (Alternative Formulation)")
println("Solution Pool Limit: 10")
println("="^80)

 # Additional parameters for column generation (hardcoded for capa11.txt)
unmet_pen = 100000.0
scaling_factor = 1.0
capacity_factor = 0.05

# =============================================================================
# DATA READING AND PREPROCESSING
# =============================================================================


function read_beasley_cap(filepath::String)
    """Read facility location data from Beasley format files (like cap61.txt)"""
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

        # 3) read all customer demands on a single line
        demand_line = readline(io)
        demand_values = parse.(Float64, split(demand_line))
        demand = demand_values[1:m]  # Take only the first m values

        # 4) read cost matrix (n x m)
        cost = Array{Float64}(undef, n, m)
        for i in 1:n
            cost_line = readline(io)
            cost_values = parse.(Float64, split(cost_line))
            cost[i, :] = cost_values[1:m]  # Take only the first m values
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
# HELPER FUNCTIONS FOR TWO-PHASE PRICING
# =============================================================================

function collect_pool_rc_cap10(model, pi, dual_mu, use_solution_pool)
    """Collect up to 10 solutions from pool that satisfy reduced cost condition (for π vectors)
    
    Note: Pricing objective already includes dual terms, so pool_obj IS the reduced cost.
    We want negative reduced cost (rc < 0) to add columns.
    """
    if !use_solution_pool
        return Vector{Vector{Float64}}(), Float64[]
    end
    
    opt = backend(model)
    kept_solutions = Vector{Vector{Float64}}()
    kept_objs = Float64[]
    
    try
        solcount = MOI.get(opt, Gurobi.ModelAttribute("SolCount"))
        max_solutions = min(10, solcount)
        
        for k in 0:max_solutions-1
            MOI.set(opt, MOI.RawOptimizerAttribute("SolutionNumber"), k)
            pool_obj = MOI.get(opt, Gurobi.ModelAttribute("PoolObjVal"))
            pi_k = [MOI.get(opt, Gurobi.VariableAttribute("Xn"), JuMP.index(pi[i])) for i in 1:length(pi)]
            rc_k = pool_obj  # Pricing objective IS the reduced cost (already includes -sum(λ*_s π^s) - μ*)
            if rc_k < -1e-6  # Negative reduced cost (improving) - want to add column
                push!(kept_solutions, pi_k)
                push!(kept_objs, pool_obj)
            end
        end
    catch e
        println("  Warning: solution pool access failed: $e")
    end
    
    # If none satisfy RC < 0, return empty and let caller decide next steps
    if isempty(kept_solutions)
        return Vector{Vector{Float64}}(), Float64[]
    end
    
    return kept_solutions, kept_objs
end

function return_pricing_payload(kept_solutions, kept_objs, pricing_work_units, return_nodes, model)
    """Return pricing results in the correct format"""
    if return_nodes
        nodes = 0
        try
            nodes = MOI.get(backend(model), Gurobi.ModelAttribute("NodeCount"))
        catch
        end
        return kept_solutions, kept_objs, pricing_work_units, nodes
    else
        return kept_solutions, kept_objs, pricing_work_units
    end
end

# =============================================================================
# HELPER FUNCTIONS FOR CANDIDATE-BASED INITIALIZATION
# =============================================================================

function evaluate_scenario_cost(scenario, candidate_locations, nloc, ncust, capacity, cost, fixedcost, unmet_pen, scaling_factor)
    """Evaluate the cost of serving a scenario with fixed candidate locations"""
    # Solve facility location for this scenario with fixed candidate locations
    # Use the solve_for_continuous_var function from complete_facility_location.jl
    obj_value, _ = solve_for_continuous_var(nloc, ncust, capacity, cost, scenario, candidate_locations, fixedcost, unmet_pen, scaling_factor, nothing)
    return obj_value
end

function get_candidate_based_partition(scenarios, nloc, ncust, capacity, cost, fixedcost, unmet_pen, scaling_factor, K)
    """Get initial partition based on K candidate locations from complete facility location problem"""
    # Generate scenarios and solve facility location for each (like in complete_facility_location.jl)
    println("Generating scenarios and solving facility location for K=$K candidates...")
    
    # Convert scenarios to the format expected by sampling function
    nscen = length(scenarios)
    ncust = length(scenarios[1])
    demandmean = [mean_simple([scenarios[s][j] for s in 1:nscen]) for j in 1:ncust]  # Average demand
    demandstdev = [std_simple([scenarios[s][j] for s in 1:nscen]) for j in 1:ncust]  # Standard deviation
    
    # Generate solutions using the sampling function (correct signature: demandmean, capacity, cost, fixedcost, unmet_pen, scaling_factor, demandstdev, bernoulli_case, method, experiment_type, log_file)
    xsolutions, _, _ = sampling(nscen, nloc, demandmean, capacity, cost, fixedcost, unmet_pen, scaling_factor, demandstdev, false, "new", "low", nothing)
    
    # Now solve the reformulated extensive form to get K optimal solutions
    println("Solving reformulated extensive form to select K=$K optimal solutions...")
    
    # Calculations_For_Extensive_Form expects scenarios as a Vector, not matrix
    # Convert scenarios to vector format for Calculations_For_Extensive_Form
    scenario_vector = scenarios  # Already a vector of vectors
    
    # Calculate objective matrix (scenario-solution costs)
    obj_matrix, _ = Calculations_For_Extensive_Form(nloc, ncust, capacity, cost, scenario_vector, xsolutions, fixedcost, unmet_pen, scaling_factor, nothing)
    
    # Solve to get K optimal solutions
    _, z = Reformulated_Extensive_Form(nscen, nscen, K, obj_matrix, nothing)
    
    # Extract the K selected solutions
    # Note: Reformulated_Extensive_Form may return [] if optimization doesn't terminate optimally
    selected_solutions = []
    if !isempty(z)
        # z has length nscen, and we need to map indices to xsolutions rows
        # Since obj_matrix is nscen × nscen (scenario × solution), z[i] corresponds to xsolutions[i, :]
        num_solutions = size(xsolutions, 1)
        for k in 1:min(length(z), num_solutions)
            if z[k] == 1
                push!(selected_solutions, xsolutions[k, :])
            end
        end
    end
    
    # Check if we have enough solutions
    if isempty(selected_solutions)
        println("Warning: No solutions selected by Reformulated_Extensive_Form (may not have terminated optimally). Using first K solutions from xsolutions.")
        # Use first K solutions as fallback
        for k in 1:min(K, size(xsolutions, 1))
            push!(selected_solutions, xsolutions[k, :])
        end
    elseif length(selected_solutions) < K
        println("Warning: Only found $(length(selected_solutions)) solutions, but need $K. Padding with first solution.")
        # If we don't have enough, pad with the first solution
        while length(selected_solutions) < K
            push!(selected_solutions, selected_solutions[1])  # Duplicate the first solution
        end
    end
    
    # For each scenario, find which candidate gives lowest cost
    partition = [zeros(Int, length(scenarios)) for _ in 1:K]  # K empty subsets
    
    for (s, scenario) in enumerate(scenarios)
        best_cost = Inf
        best_candidate = 1
        
        # Evaluate cost of serving scenario s with each candidate
        for k in 1:K
            candidate_cost = evaluate_scenario_cost(scenario, selected_solutions[k], nloc, ncust, capacity, cost, fixedcost, unmet_pen, scaling_factor)
            if candidate_cost < best_cost
                best_cost = candidate_cost
                best_candidate = k
            end
        end
        
        # Assign scenario s to best candidate
        partition[best_candidate][s] = 1
    end
    
    return partition
end

println(" Problem Parameters:")
println("  Unmet penalty: $unmet_pen")
println("  Scaling factor: $scaling_factor")
println("  Capacity factor: $capacity_factor")

# Function to calculate associated cost for a subset of scenarios
function associated_cost_function(A, scenarios, nloc, ncust, capacity, cost, fixedcost, unmet_pen, scaling_factor)
    """
    Parameter A description:
    - A: 0-1 vector of length nscen, where A[s] = 1 if scenario s is included, 0 otherwise
    """
    nscen = length(scenarios)
    
    # Check if A is valid
    if length(A) != nscen
        error("A must have length equal to number of scenarios ($nscen)")
    end
    
    # Check if subset is empty
    if sum(A) == 0
        return 0.0  # Cost of empty subset is 0
    end
    
    # Create model
    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 0)  # Suppress output
    set_optimizer_attribute(model, "TimeLimit", 60)  # 1 minute time limit
    
    # Decision Variables
    @variable(model, x[1:nloc], Bin)  # Binary: facility open/closed
    @variable(model, y[1:nloc, 1:ncust, 1:nscen] >= 0)  # Flow: y[i,j,s]
    @variable(model, b[1:ncust, 1:nscen] >= 0)  # Unmet demand: b[j,s]
    
    # Constraints
    # 1. Demand satisfaction constraint: sum_i y[i,j,s] + b[j,s] = d[j,s] for s∈A, j∈J
    for s in 1:nscen
        if A[s] == 1  # Only add constraint for scenarios in subset A
            for j in 1:ncust
                @constraint(model, sum(y[i,j,s] for i in 1:nloc) + b[j,s] == scenarios[s][j])
            end
        end
    end
    
    # 2. Capacity constraint: sum_j y[i,j,s] - capacity[i]*x[i] <= 0 for i∈I, s∈A
    for s in 1:nscen
        if A[s] == 1  # Only add constraint for scenarios in subset A
            for i in 1:nloc
                @constraint(model, sum(y[i,j,s] for j in 1:ncust) - capacity[i]*x[i] <= 0)
            end
        end
    end
    
    # 3. Set y[i,j,s] = 0 and b[j,s] = 0 for scenarios NOT in subset A
    for s in 1:nscen
        if A[s] == 0  # For scenarios not in subset A
            for i in 1:nloc, j in 1:ncust
                @constraint(model, y[i,j,s] == 0)
            end
            for j in 1:ncust
                @constraint(model, b[j,s] == 0)
            end
        end
    end
    
    # Objective Function with scaling by |A|/|S| and 1/|S|
    # fixed costs term weighted by |A|/|S|; flow and penalty terms weighted by 1/|S|
    subset_size = sum(A)
    weight_fixed = 1
    weight_rest = 1.0 / subset_size
    @objective(model, Min,
        weight_fixed * sum(fixedcost[i] * x[i] for i in 1:nloc) +
        weight_rest * (
            scaling_factor * sum(cost[i,j] * y[i,j,s] for s in 1:nscen for i in 1:nloc for j in 1:ncust if A[s] == 1) +
            sum(unmet_pen * b[j,s] for s in 1:nscen for j in 1:ncust if A[s] == 1)
        )
    )
    
    # Solve
    optimize!(model)
    
    if termination_status(model) == MOI.OPTIMAL
        return objective_value(model)
    else
        return Inf  # Return infinite cost if infeasible
    end
end

# Function to solve SF Master Problem (Partition Formulation)
function solve_sf_master_problem(cost_vector, z_matrix, K)
    """
    Solve the SF Master Problem (Partition Formulation)
    
    Parameters:
    - cost_vector: f(A) values for each subset A
    - z_matrix: |S| × |A| matrix where z_matrix[s, A] = 1 if scenario s is in subset A
    - K: Cardinality constraint limit
    
    Returns:
    - model, v_variables, objective_value, partition_constraints, k_constraint
    """
    nscen, nsubsets = size(z_matrix)
    
    # Create model
    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 0)
    set_optimizer_attribute(model, "LogFile", "CG_SF_master_K$(K_input)_S$(S_input).log")
    
    # Decision Variables - Dynamic array of v_A variables
    v_variables = [@variable(model, lower_bound = 0, upper_bound = 1) for A in 1:nsubsets]
    
    # Objective Function (9a): min ∑_{A∈F} f(A)v_A
    @objective(model, Min, sum(cost_vector[A] * v_variables[A] for A in 1:nsubsets))
    
    # Constraints - Store references for dual extraction
    # Partition constraints (9b): ∑_{A∈F} z_sA v_A = 1 ∀s ∈ S
    partition_constraints = @constraint(model, [s in 1:nscen], 
        sum(z_matrix[s, A] * v_variables[A] for A in 1:nsubsets) == 1)
    
    # Cardinality constraint (9c): ∑_{A∈F} v_A ≤ K
    k_constraint = @constraint(model, sum(v_variables[A] for A in 1:nsubsets) <= K)
    
    # Solve
    optimize!(model)
    
    if termination_status(model) == MOI.OPTIMAL
        return model, v_variables, objective_value(model), partition_constraints, k_constraint
    else
        return model, v_variables, Inf, partition_constraints, k_constraint
    end
end

# Function to solve SF Pricing Type 1
function sf_pricing_type1(nloc, ncust, capacity, cost, scenarios, dual_lambda, dual_mu, unmet_pen, scaling_factor; use_solution_pool::Bool=true, return_nodes::Bool=false, work_limit::Float64=Inf, opt_worklimit::Float64=DEFAULT_OPT_WORKLIMIT)
    """
    Solve SF Pricing Type 1 - Generate new subset of scenarios (solve to optimality with work limit)
    
    Parameters:
    - nloc, ncust: number of locations and customers
    - capacity, cost: facility location data
    - scenarios: list of all scenarios
    - dual_lambda: dual values from partition constraints (vector of length nscen)
    - dual_mu: dual value from cardinality constraint (scalar)
    - unmet_pen, scaling_factor: penalty and scaling parameters
    - use_solution_pool: Enable solution pool collection
    - return_nodes: Return B&B node count
    - work_limit: Total work budget for pricing (solve to optimality with this limit)
    - opt_worklimit: Not used for Type 1 (kept for compatibility)
    
    Returns:
    - solutions: Vector of π vectors (binary scenario selection vectors)
    - objectives: Vector of objective values
    - work_units: Total work units used
    - (optionally nodes if return_nodes=true)
    """
    nscen = length(scenarios)
    
    # Create model
    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 1)
    set_optimizer_attribute(model, "LogFile", "CG_SF_pricing_K$(K_input)_S$(S_input).log")
    if use_solution_pool
        set_optimizer_attribute(model, "PoolSearchMode", 2)
        set_optimizer_attribute(model, "PoolSolutions", 10)
    end
    
    # Decision Variables
    @variable(model, x[1:nloc], Bin)  # Binary: facility open/closed
    @variable(model, y[1:nloc, 1:ncust, 1:nscen] >= 0)  # Flow: y[i,j,s]
    @variable(model, b[1:ncust, 1:nscen] >= 0)  # Unmet demand: b[j,s]
    @variable(model, v[1:nloc, 1:nscen] >= 0)  # Auxiliary variable: v[i,s] = x[i] * pi[s]
    @variable(model, u[1:nloc, 1:ncust, 1:nscen] >= 0)  # Auxiliary flow: u[i,j,s] = y[i,j,s] * pi[s]
    @variable(model, w[1:ncust, 1:nscen] >= 0)  # Auxiliary unmet demand: w[j,s] = b[j,s] * pi[s]
    @variable(model, pi[1:nscen], Bin)  # Binary: scenario selection π^s
    
    # Objective Function: min ∑_i f_i v_i^s + ∑_s ∑_i ∑_j c_ij u_ij^s + ∑_s ∑_j α w_j^s - ∑_s (λ^s)*π^s - μ*
    @objective(model, Min, 
        1.0/nscen * (sum(fixedcost[i]*v[i,s] for i in 1:nloc for s in 1:nscen) + 
        scaling_factor * sum(cost[i,j]*u[i,j,s] for i in 1:nloc for j in 1:ncust for s in 1:nscen) + 
        sum(unmet_pen*w[j,s] for j in 1:ncust for s in 1:nscen)) - 
        sum(dual_lambda[s]*pi[s] for s in 1:nscen) - 
        dual_mu)
    
    # Constraints
    # Constraint 1: ∑_i y_ij^s + b_j^s - d_j^s ≤ d_j^s (1 - π^s)
    @constraint(model, [j in 1:ncust, s in 1:nscen], 
        sum(y[i,j,s] for i in 1:nloc) + b[j,s] - scenarios[s][j] <= scenarios[s][j] * (1 - pi[s]))
    
    # Constraint 2: v_i^s ≤ x_i
    @constraint(model, [i in 1:nloc, s in 1:nscen], 
        v[i,s] <= x[i])
    
    # Constraint 3: v_i^s ≤ π^s
    @constraint(model, [i in 1:nloc, s in 1:nscen], 
        v[i,s] <= pi[s])
    
    # Constraint 4: v_i^s ≥ x_i + π^s - 1
    @constraint(model, [i in 1:nloc, s in 1:nscen], 
        v[i,s] >= x[i] + pi[s] - 1)
    
    # Constraint 5: ∑_i y_ij^s + b_j^s - d_j^s ≥ -d_j^s (1 - π^s)
    @constraint(model, [j in 1:ncust, s in 1:nscen], 
        sum(y[i,j,s] for i in 1:nloc) + b[j,s] - scenarios[s][j] >= -scenarios[s][j] * (1 - pi[s]))
    
    # Constraint 6: ∑_j y_ij^s - ū_i x_i ≤ ū_i (1 - π^s)
    @constraint(model, [i in 1:nloc, s in 1:nscen], 
        sum(y[i,j,s] for j in 1:ncust) - capacity[i]*x[i] <= capacity[i] * (1 - pi[s]))
    
    # Constraint 7: u_ij^s ≤ y_ij^s
    @constraint(model, [i in 1:nloc, j in 1:ncust, s in 1:nscen], 
        u[i,j,s] <= y[i,j,s])
    
    # Constraint 8: u_ij^s ≤ d_j^s π^s
    @constraint(model, [i in 1:nloc, j in 1:ncust, s in 1:nscen], 
        u[i,j,s] <= scenarios[s][j] * pi[s])
    
    # Constraint 9: u_ij^s ≥ y_ij^s + d_j^s (π^s - 1)
    @constraint(model, [i in 1:nloc, j in 1:ncust, s in 1:nscen], 
        u[i,j,s] >= y[i,j,s] + scenarios[s][j] * (pi[s] - 1))
    
    # Constraint 10: w_j^s ≤ b_j^s
    @constraint(model, [j in 1:ncust, s in 1:nscen], 
        w[j,s] <= b[j,s])
    
    # Constraint 11: w_j^s ≤ d_j^s π^s
    @constraint(model, [j in 1:ncust, s in 1:nscen], 
        w[j,s] <= scenarios[s][j] * pi[s])
    
    # Constraint 12: w_j^s ≥ b_j^s + d_j^s (π^s - 1)
    @constraint(model, [j in 1:ncust, s in 1:nscen], 
        w[j,s] >= b[j,s] + scenarios[s][j] * (pi[s] - 1))
    
    # Solve to optimality with given work limit
    pricing_work_units = 0.0
    
    println("  Solving Type 1 pricing to optimality with work limit: $work_limit")
    if isfinite(work_limit) && work_limit > 0.0
        set_optimizer_attribute(model, "MIPFocus", 0)  # Default optimality focus
        set_optimizer_attribute(model, "WorkLimit", work_limit)
    end
    
    optimize!(model)
    
    try
        pricing_work_units = MOI.get(backend(model), Gurobi.ModelAttribute("Work"))
    catch
    end
    
    # Check termination status and collect solutions
    if termination_status(model) == MOI.OPTIMAL
        println("  Optimal solution found")
        kept_solutions, kept_objs = collect_pool_rc_cap10(model, pi, dual_mu, use_solution_pool)
        if !isempty(kept_solutions)
            println("  Found $(length(kept_solutions)) RC-improving solutions")
            for (idx, objv) in enumerate(kept_objs)
                rc = objv  # Pricing objective IS the reduced cost (already includes dual terms)
                println("    -> pool obj[$idx] = $(round(objv, digits=6)), rc = $(round(rc, digits=6))")
            end
        end
        return return_pricing_payload(kept_solutions, kept_objs, pricing_work_units, return_nodes, model)
    else
        println("  Work units exhausted or not optimal. Checking for RC-improving incumbents...")
        try
            lb = MOI.get(backend(model), Gurobi.ModelAttribute("ObjBound"))
            ub = MOI.get(backend(model), Gurobi.ModelAttribute("ObjVal"))
            println("  Lower bound: $(round(lb, digits=6)), Upper bound: $(round(ub, digits=6))")
        catch
        end
        kept_solutions, kept_objs = collect_pool_rc_cap10(model, pi, dual_mu, use_solution_pool)
        if !isempty(kept_solutions)
            println("  Found $(length(kept_solutions)) RC-improving solutions")
            for (idx, objv) in enumerate(kept_objs)
                rc = objv
                println("    -> pool obj[$idx] = $(round(objv, digits=6)), rc = $(round(rc, digits=6))")
            end
        end
        return return_pricing_payload(kept_solutions, kept_objs, pricing_work_units, return_nodes, model)
    end
end

# Function to solve SF Pricing Type 2
function sf_pricing_type2(nloc, ncust, capacity, cost, scenarios, dual_lambda, dual_mu, unmet_pen, scaling_factor; use_solution_pool::Bool=true, return_nodes::Bool=false, work_limit::Float64=Inf, opt_worklimit::Float64=DEFAULT_OPT_WORKLIMIT, analysis::Dict{Symbol,Any}=Dict{Symbol,Any}())
    """
    Solve SF Pricing Type 2 - Alternative formulation (two-phase with solution pool)
    
    Parameters:
    - nloc, ncust: number of locations and customers
    - capacity, cost: facility location data
    - scenarios: list of all scenarios
    - dual_lambda: dual values from partition constraints (vector of length nscen)
    - dual_mu: dual value from cardinality constraint (scalar)
    - unmet_pen, scaling_factor: penalty and scaling parameters
    - use_solution_pool: Enable solution pool collection
    - return_nodes: Return B&B node count
    - work_limit: Total work budget for pricing
    - opt_worklimit: Phase 1 work limit
    - analysis: Dictionary to store phase analysis
    
    Returns:
    - solutions: Vector of π vectors (binary scenario selection vectors)
    - objectives: Vector of objective values
    - work_units: Total work units used
    - (optionally nodes if return_nodes=true)
    """
    nscen = length(scenarios)
    
    # Create model
    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 1)
    set_optimizer_attribute(model, "LogFile", "CG_SF_pricing_K$(K_input)_S$(S_input).log")
    if use_solution_pool
        set_optimizer_attribute(model, "PoolSearchMode", 2)
        set_optimizer_attribute(model, "PoolSolutions", 10)
    end
    
    # Decision Variables
    @variable(model, x[1:nloc], Bin)  # Binary: facility open/closed (first-stage)
    @variable(model, x_s[1:nloc, 1:nscen], Bin)  # Binary copies: facility open/closed for scenario s
    @variable(model, y[1:nloc, 1:ncust, 1:nscen] >= 0)  # Flow: y[i,j,s]
    @variable(model, b[1:ncust, 1:nscen] >= 0)  # Unmet demand: b[j,s]
    @variable(model, pi[1:nscen], Bin)  # Binary: scenario selection π^s
    
    # Vector of ones (e)
    e = ones(nloc)
    
    # Objective Function: min ∑_s ( (∑_i f_i x_i^s) + (∑_i ∑_j c_ij y_ij^s) + (∑_j α b_j^s) - (λ^s)*π^s ) - μ*
    @objective(model, Min, 
        1/nscen * (sum(fixedcost[i]*x_s[i,s] for i in 1:nloc for s in 1:nscen) + 
            scaling_factor * sum(cost[i,j]*y[i,j,s] for i in 1:nloc for j in 1:ncust for s in 1:nscen) + 
            sum(unmet_pen*b[j,s] for j in 1:ncust for s in 1:nscen))- 
            sum(dual_lambda[s]*pi[s] for s in 1:nscen) - 
        dual_mu)
    
    # Constraints
    # Constraint 1: ∑_i y_ij^s + b_j^s = d_j^s for all s∈S, j∈J
    @constraint(model, [j in 1:ncust, s in 1:nscen], 
        sum(y[i,j,s] for i in 1:nloc) + b[j,s] == scenarios[s][j]*pi[s])
    
    # Constraint 2: ∑_j y_ij^s - u_i x_i^s ≤ 0 for all i∈I, s∈S
    @constraint(model, [i in 1:nloc, s in 1:nscen], 
        sum(y[i,j,s] for j in 1:ncust) - capacity[i]*x_s[i,s] <= 0)
    
    # Constraint 3: x^s ≤ e π^s for all s∈S
    @constraint(model, [i in 1:nloc, s in 1:nscen], 
        x_s[i,s] <= pi[s])
    
    # Constraint 4: x - x^s ≤ e(1 - π^s) for all s∈S
    @constraint(model, [i in 1:nloc, s in 1:nscen], 
        x[i] - x_s[i,s] <= (1 - pi[s]))
    
    # Constraint 5: x - x^s ≥ -e(1 - π^s) for all s∈S
    @constraint(model, [i in 1:nloc, s in 1:nscen], 
        x[i] - x_s[i,s] >= -(1 - pi[s]))
    
    # Two-phase solve structure with analysis tracking
    pricing_work_units = 0.0
    if !isempty(analysis)
        analysis[:phase1_work] = 0.0
        analysis[:phase2_work] = 0.0
        analysis[:went_phase2] = false
        analysis[:lb] = NaN
        analysis[:ub] = NaN
    end

    # Phase 1: Quick optimal try
    if isfinite(opt_worklimit) && opt_worklimit > 0.0
        println("  Phase 1: Quick optimal try with $(opt_worklimit) work units")
        set_optimizer_attribute(model, "MIPFocus", 1)
        set_optimizer_attribute(model, "WorkLimit", opt_worklimit)
    optimize!(model)
    
        try
            local w = MOI.get(backend(model), Gurobi.ModelAttribute("Work"))
            pricing_work_units += w
            if !isempty(analysis); analysis[:phase1_work] = w; end
        catch
        end
        
        # Check for RC-improving solutions
    if termination_status(model) == MOI.OPTIMAL
            println("  Phase 1: Optimal solution found")
            kept_solutions, kept_objs = collect_pool_rc_cap10(model, pi, dual_mu, use_solution_pool)
            if !isempty(kept_solutions)
                println("  Phase 1: Found $(length(kept_solutions)) RC-improving solutions")
                for (idx, objv) in enumerate(kept_objs)
                    rc = objv  # Pricing objective IS the reduced cost (already includes dual terms)
                    println("    -> pool obj[$idx] = $(round(objv, digits=6)), rc = $(round(rc, digits=6))")
                end
                if !isempty(analysis); analysis[:went_phase2] = false; end
                return return_pricing_payload(kept_solutions, kept_objs, pricing_work_units, return_nodes, model)
            else
                println("  Phase 1: Optimal but no RC-improving solutions — terminating pricing")
                return Vector{Vector{Float64}}(), Float64[], pricing_work_units
            end
        else
            println("  Phase 1: Not optimal - checking pool for RC-improving incumbents")
            kept_solutions, kept_objs = collect_pool_rc_cap10(model, pi, dual_mu, use_solution_pool)
            if !isempty(kept_solutions)
                println("  Phase 1: Found $(length(kept_solutions)) RC-improving solutions (non-optimal status)")
                for (idx, objv) in enumerate(kept_objs)
                    rc = objv  # Pricing objective IS the reduced cost (already includes dual terms)
                    println("    -> pool obj[$idx] = $(round(objv, digits=6)), rc = $(round(rc, digits=6))")
                end
                if !isempty(analysis); analysis[:went_phase2] = false; end
                return return_pricing_payload(kept_solutions, kept_objs, pricing_work_units, return_nodes, model)
            else
                println("  Phase 1: No RC-improving solutions found, proceeding to Phase 2")
            end
        end
    end

    # Phase 2: Optimality push
    heur_worklimit = work_limit - opt_worklimit
    if isfinite(heur_worklimit) && heur_worklimit > 0.0
        println("  Phase 2: Optimality push with $(heur_worklimit) work units")
        if !isempty(analysis); analysis[:went_phase2] = true; end
        set_optimizer_attribute(model, "MIPFocus", 0)
        set_optimizer_attribute(model, "WorkLimit", heur_worklimit)
        optimize!(model)
        
        try
            local w2 = MOI.get(backend(model), Gurobi.ModelAttribute("Work"))
            pricing_work_units += w2
            if !isempty(analysis); analysis[:phase2_work] = w2; end
        catch
        end
        
        if termination_status(model) == MOI.OPTIMAL
            println("  Phase 2: Optimal solution found")
            try
                local lb = MOI.get(backend(model), Gurobi.ModelAttribute("ObjBound"))
                local ub = MOI.get(backend(model), Gurobi.ModelAttribute("ObjVal"))
                if !isempty(analysis)
                    analysis[:lb] = lb
                    analysis[:ub] = ub
                end
            catch
            end
            kept_solutions, kept_objs = collect_pool_rc_cap10(model, pi, dual_mu, use_solution_pool)
            return return_pricing_payload(kept_solutions, kept_objs, pricing_work_units, return_nodes, model)
        else
            println("  Phase 2: Work units exhausted. Reporting bounds...")
            try
                local lb = MOI.get(backend(model), Gurobi.ModelAttribute("ObjBound"))
                local ub = MOI.get(backend(model), Gurobi.ModelAttribute("ObjVal"))
                println("  Lower bound: $lb, Upper bound: $ub")
                if !isempty(analysis)
                    analysis[:lb] = lb
                    analysis[:ub] = ub
                end
            catch
                println("  Could not retrieve bounds")
            end
            kept_solutions, kept_objs = collect_pool_rc_cap10(model, pi, dual_mu, use_solution_pool)
            return return_pricing_payload(kept_solutions, kept_objs, pricing_work_units, return_nodes, model)
        end
    end
    
    # Fallback if no work limits set
    return return_pricing_payload(Vector{Vector{Float64}}(), Float64[], pricing_work_units, return_nodes, model)
end

# Function to solve SF pricing subproblem with choice of formulation type
function solve_sf_pricing_subproblem(nloc, ncust, capacity, cost, scenarios, dual_lambda, dual_mu, unmet_pen, scaling_factor, pricing_type=2; use_solution_pool::Bool=true, work_limit::Float64=Inf, opt_worklimit::Float64=DEFAULT_OPT_WORKLIMIT)
    """
    Solve the SF pricing subproblem with choice of formulation type
    Returns: (solutions_vector, objs_vector, work_units)
    """
    if pricing_type == 1
        println("  Solving SF Pricing Type 1...")
        sols, objs, pwu = sf_pricing_type1(nloc, ncust, capacity, cost, scenarios, dual_lambda, dual_mu, unmet_pen, scaling_factor; use_solution_pool=use_solution_pool, work_limit=work_limit, opt_worklimit=opt_worklimit)
        return sols, objs, pwu
        
    elseif pricing_type == 2
        println("  Solving SF Pricing Type 2...")
        sols, objs, pwu = sf_pricing_type2(nloc, ncust, capacity, cost, scenarios, dual_lambda, dual_mu, unmet_pen, scaling_factor; use_solution_pool=use_solution_pool, work_limit=work_limit, opt_worklimit=opt_worklimit)
        return sols, objs, pwu
        
    else
        error("Invalid pricing_type. Use 1 for sf_pricing_type1 or 2 for sf_pricing_type2.")
    end
end

# Main SF Column Generation Algorithm
function sf_column_generation_algorithm(scenarios, nloc, ncust, capacity, cost, fixedcost, unmet_pen, scaling_factor, K, max_iterations, pricing_type=2, init_type=1; use_solution_pool::Bool=true, custom_partition=nothing)
    """Main SF Column Generation Algorithm with 5 stages"""
    nscen = length(scenarios)
    
    println(" Starting SF Column Generation Algorithm")
    println("Parameters: nscen=$nscen, nloc=$nloc, ncust=$ncust, K=$K, max_iter=$max_iterations")
    println()
    
    # Global work budget tracking
    total_work_budget = 30000.0
    total_master_work_units = 0.0
    total_pricing_work_units = 0.0
    total_update_work_units = 0.0
    last_master_work_cumulative = 0.0
    
    # Initialize pricing + dual_mu tracking
    iteration_pricing_data = []
    iteration_master_data = []
    iteration_columns_data = []
    pricing_iteration_analysis = []
    actual_iterations = 0  # Track actual number of iterations completed
    
    # Initialize subproblem comparison tracking (for first 10 iterations)
    type1_metrics = []
    type2_metrics = []
    
    # Stage 1: Initialization
    println(" Stage 1: Initialization")
    
    if init_type == 1
        # Option 1: Each scenario as its own subset (one scenario per subset)
        println("  Initialization Type 1: Each scenario as its own subset...")
        
        # Create one subset for each scenario (up to K subsets)
        num_init_subsets = min(nscen, K)
        z_matrix = zeros(nscen, num_init_subsets)
        
        for s in 1:num_init_subsets
            z_matrix[s, s] = 1  # Scenario s goes to subset s
        end
        
        # Calculate associated cost for each subset
        cost_vector = []
        for k in 1:num_init_subsets
            A_k = z_matrix[:, k]
            cost_k = associated_cost_function(A_k, scenarios, nloc, ncust, capacity, cost, fixedcost, unmet_pen, scaling_factor)
            push!(cost_vector, sanitize_cost(cost_k))
        end
        
        nsubsets = length(cost_vector)
        println("  Generated $nsubsets initial subsets (one scenario per subset)")
        
    elseif init_type == 2
        # Option 2: Start with single subset (first scenario only)
        println("  Initialization Type 2: Starting with single subset (first scenario)...")
        A_1 = zeros(nscen)
        A_1[1] = 1
        
        # Calculate associated cost
        cost_1 = associated_cost_function(A_1, scenarios, nloc, ncust, capacity, cost, fixedcost, unmet_pen, scaling_factor)
        cost_vector = [sanitize_cost(cost_1)]
        z_matrix = reshape(A_1, nscen, 1)  # Single column matrix
        
        nsubsets = length(cost_vector)
        println("  Generated $nsubsets initial subset (first scenario only)")
        
    elseif init_type == 3
        # Option 3: Candidate-based partition
        println("  Initialization Type 3: Candidate-based partition...")
        partition = get_candidate_based_partition(scenarios, nloc, ncust, capacity, cost, fixedcost, unmet_pen, scaling_factor, K)
        
        # Build cost_vector and z_matrix from this partition
        cost_vector = Float64[]
        z_matrix = zeros(nscen, K)
        
        for k in 1:K
            subset_cost = associated_cost_function(partition[k], scenarios, nloc, ncust, capacity, cost, fixedcost, unmet_pen, scaling_factor)
            push!(cost_vector, sanitize_cost(subset_cost))
            
            # Update z_matrix
            for s in 1:nscen
                if partition[k][s] == 1
                    z_matrix[s, k] = 1
                end
            end
        end
        
        nsubsets = length(cost_vector)
        println("  Generated $nsubsets initial subsets (candidate-based partition)")
    elseif init_type == 99
        # Option 99: Custom partition (for comparison script)
        println("  Initialization Type 99: Using custom partition...")
        if custom_partition === nothing
            error("init_type=99 requires custom_partition keyword argument")
        end
        
        # Build cost_vector and z_matrix from custom partition
        cost_vector = Float64[]
        ncustom_subsets = length(custom_partition)
        z_matrix = zeros(nscen, ncustom_subsets)
        
        for k in 1:ncustom_subsets
            subset_cost = associated_cost_function(custom_partition[k], scenarios, nloc, ncust, capacity, cost, fixedcost, unmet_pen, scaling_factor)
            push!(cost_vector, sanitize_cost(subset_cost))
            
            # Update z_matrix
            for s in 1:nscen
                if custom_partition[k][s] == 1
                    z_matrix[s, k] = 1
                end
            end
        end
        
        nsubsets = length(cost_vector)
        println("  Using $nsubsets custom initial subsets")
    end
    
    # Initialize master problem
    model, v_variables, master_obj, partition_constraints, k_constraint = solve_sf_master_problem(cost_vector, z_matrix, K)
    
    if master_obj == Inf
        println(" Error: Initial master problem is infeasible!")
        return [], Inf, 0, "FAILED"
    end
    
    println("  Initial master problem objective: $master_obj")
    println()
    
    # Column Generation Loop
    for iteration in 1:max_iterations
        println("\033[34m================ ITERATION ", iteration, " ===================\033[0m")
        
        # Work budget check
        used_work = total_master_work_units + total_pricing_work_units + total_update_work_units
        available_work = max(0.0, total_work_budget - used_work)
        println(" Available work units: $(round(available_work, digits=2)) / $(total_work_budget)")
        
        # Stage 2: Master Problem Solution
        println(" Stage 2: Solving Master Problem")
        master_work_units = 0.0
        try
        optimize!(model)
            local cum = MOI.get(backend(model), Gurobi.ModelAttribute("Work"))
            master_work_units = max(0.0, cum - last_master_work_cumulative)
            last_master_work_cumulative = cum
            total_master_work_units += master_work_units
        catch e
            println("  Warning: unable to get master work units: $e")
        end
        
        if termination_status(model) != MOI.OPTIMAL
            println("Master problem failed to solve")
            break
        end
        
        master_obj = objective_value(model)
        println("  Master objective: $(round(master_obj, digits=6))  (+$(round(master_work_units, digits=2)) work)")
        push!(iteration_master_data, (iteration=iteration, master_obj=master_obj))
        
        # Stage 3: Dual Extraction
        println("Stage 3: Extracting Dual Solutions")
        
        # Extract dual values using JuMP pattern
        dual_lambda = dual.(partition_constraints)  # Extract duals from partition constraints (9b)
        dual_mu = dual(k_constraint)  # Extract dual from cardinality constraint (9c)
        
        println("  Dual μ value: $dual_mu")
        
        # Stage 4: Pricing Subproblem
        println(" Stage 4: Solving Pricing Subproblem")
        used_work = total_master_work_units + total_pricing_work_units + total_update_work_units
        remaining_work = max(0.0, total_work_budget - used_work)
        if remaining_work <= 0.0
            println("  Work budget exhausted. Terminating.")
            break
        end
        
        # Always solve Pricing Type 2 only (no Type 1 comparisons)
        new_solutions = Vector{Vector{Float64}}()
        pool_objs = Float64[]
        pricing_work_units = 0.0
        try
            local analysis = Dict{Symbol,Any}()
            new_solutions, pool_objs, pricing_work_units = solve_sf_pricing_subproblem(nloc, ncust, capacity, cost, scenarios, dual_lambda, dual_mu, unmet_pen, scaling_factor, 2; use_solution_pool=use_solution_pool, work_limit=remaining_work, opt_worklimit=DEFAULT_OPT_WORKLIMIT)
            if isempty(new_solutions)
                println("  No improving solutions from pricing. Terminating algorithm.")
                break
            end
            total_pricing_work_units += pricing_work_units
            push!(pricing_iteration_analysis, (
                iteration = iteration,
                phase1_work = get(analysis, :phase1_work, NaN),
                went_phase2 = get(analysis, :went_phase2, false),
                phase2_work = get(analysis, :phase2_work, NaN),
                lb = get(analysis, :lb, NaN),
                ub = get(analysis, :ub, NaN),
                total_pricing_work = pricing_work_units,
            ))
            println("  Pricing work units: $(round(pricing_work_units, digits=2))")
        catch e
            println("Error in pricing subproblem: $e")
            break
        end
        
        # Filter by reduced cost and cap to 10
        # Pricing objective IS the reduced cost (already includes -sum(λ*_s π^s) - μ*)
        # We want negative reduced cost (rc < 0) to add columns
        rc_ok_indices = [idx for idx in 1:length(pool_objs) if pool_objs[idx] < -1e-6]
        if isempty(rc_ok_indices)
            println("  No RC-improving solutions (reduced cost >= 0). Terminating.")
            obj_pricing_term = isempty(pool_objs) ? 0.0 : sanitize_cost(minimum(pool_objs))
            push!(iteration_pricing_data, (iteration = iteration, obj_pricing = obj_pricing_term, dual_mu = dual_mu, reduced_cost = obj_pricing_term))
            push!(iteration_columns_data, (iteration = iteration, num_added = 0))
            break
        end
        keep = rc_ok_indices[1:min(10, length(rc_ok_indices))]
        println("  Will add the following columns (rank, obj, rc):")
        for (rank, idx) in enumerate(keep)
            local objv = pool_objs[idx]
            local rcv = objv  # Pricing objective IS the reduced cost
            println("    $(rank)) obj=$(round(objv, digits=6))\trc=$(round(rcv, digits=6))")
        end
        new_solutions = new_solutions[keep]
        pool_objs = pool_objs[keep]
        
        obj_pricing = minimum(pool_objs)
        obj_pricing = sanitize_cost(obj_pricing)  # Sanitize in case of Inf/NaN
        reduced_cost = obj_pricing  # Pricing objective IS the reduced cost
        
        # Stage 5: Convergence Check & Master Problem Update
        println(" Stage 5: Convergence Check & Master Problem Update")
        println("  Pricing objective (reduced cost): $obj_pricing")
        println("  Reduced cost < 0 means we should add column")
        
        # Store pricing data for table
        push!(iteration_pricing_data, (iteration = iteration, obj_pricing = obj_pricing, dual_mu = dual_mu, reduced_cost = reduced_cost))
        
        # Check convergence: if reduced cost >= 0, terminate; if < 0, add columns
        if reduced_cost >= -1e-6
            println(" Algorithm converged! Reduced cost >= 0")
            push!(iteration_columns_data, (iteration = iteration, num_added = 0))
            break
        else
            println(" Adding $(length(new_solutions)) column(s) to master problem")
            local_added = 0
            local_update_work = 0.0
            
            for pi_new in new_solutions
            # Calculate associated cost for new subset
            new_cost = associated_cost_function(pi_new, scenarios, nloc, ncust, capacity, cost, fixedcost, unmet_pen, scaling_factor)
            push!(cost_vector, sanitize_cost(new_cost))
            
            # Update z_matrix with new column
            z_matrix = hcat(z_matrix, pi_new)
            nsubsets += 1
            
            # Add new v_A variable to master problem
            new_v = @variable(model, lower_bound = 0, upper_bound = 1)
            push!(v_variables, new_v)
            
            # Update objective: set coefficient for new v_A variable
            set_objective_coefficient(model, new_v, sanitize_cost(new_cost))
            
            # Update the constraint matrix entries
            # Partition constraints (9b): each scenario sums to 1
            for s in 1:nscen
                set_normalized_coefficient(partition_constraints[s], new_v, pi_new[s])
            end
            
            # Cardinality constraint (9c): sum v_A <= K
            set_normalized_coefficient(k_constraint, new_v, 1.0)
            
                local_added += 1
            end
            total_update_work_units += local_update_work
            println("  Update work units: $(round(local_update_work, digits=2))")
            println("New columns added. Total subsets: $nsubsets")
            push!(iteration_columns_data, (iteration = iteration, num_added = local_added))
        end
        
        # Increment actual iterations counter
        actual_iterations = iteration
        
        println()
    end
    
    # Final solution - Linear Relaxation
    optimize!(model)
    status = termination_status(model) == MOI.OPTIMAL ? "OPTIMAL" : "FAILED"
    final_objective_lp = objective_value(model)
    total_added = isempty(iteration_columns_data) ? 0 : sum(t.num_added for t in iteration_columns_data; init=0)
    
    # Solve master problem with integrality constraints (binary v_variables)
    println("\n Solving final master problem with integrality constraints...")
    model_ip = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model_ip, "OutputFlag", 0)
    set_optimizer_attribute(model_ip, "LogFile", "CG_SF_master_IP_K$(K_input)_S$(S_input).log")
    
    # Decision Variables - Binary v_A variables
    v_variables_ip = [@variable(model_ip, binary=true) for A in 1:nsubsets]
    
    # Objective Function (same as LP)
    @objective(model_ip, Min, sum(cost_vector[A] * v_variables_ip[A] for A in 1:nsubsets))
    
    # Constraints (same as LP)
    @constraint(model_ip, [s in 1:nscen], 
        sum(z_matrix[s, A] * v_variables_ip[A] for A in 1:nsubsets) == 1)
    @constraint(model_ip, sum(v_variables_ip[A] for A in 1:nsubsets) <= K)
    
    # Solve IP
    optimize!(model_ip)
    status_ip = termination_status(model_ip) == MOI.OPTIMAL ? "OPTIMAL" : string(termination_status(model_ip))
    
    if status == "OPTIMAL"
        final_v_values = value.(v_variables)
        
        # Find selected subsets from LP (where v_A > 1e-6)
        selected_subsets = findall(x -> x > 1e-6, final_v_values)
        
        println("  Final Results (Linear Relaxation):")
        println("  Iterations: $actual_iterations")
        println("  Final objective (LP): $final_objective_lp")
        println("  Selected subsets (LP): $selected_subsets")
    else
        println(" Final master problem (LP) failed to solve")
        selected_subsets = []
    end
    
    # IP results
    if termination_status(model_ip) == MOI.OPTIMAL
        final_objective_ip = objective_value(model_ip)
        final_v_values_ip = value.(v_variables_ip)
        selected_subsets_ip = findall(x -> x > 0.5, final_v_values_ip)
        println("  Final Results (Integer Solution):")
        println("  Final objective (IP): $final_objective_ip")
        println("  Selected subsets (IP): $selected_subsets_ip")
    else
        final_objective_ip = Inf
        selected_subsets_ip = []
        println("  Final master problem (IP) failed to solve: $status_ip")
    end
    
    # Use IP objective as final if available, otherwise LP
    final_objective = isfinite(final_objective_ip) ? final_objective_ip : final_objective_lp
    
    # Print analysis tables
    println("\nMaster Objective by Iteration:")
    println("Iter\tMasterObj")
    for entry in iteration_master_data
        println("$(entry.iteration)\t$(round(entry.master_obj, digits=4))")
    end

    println("\nSubproblem Analysis by Iteration (Type 2 used by algorithm):")
    if isempty(pricing_iteration_analysis)
        println("<no pricing analysis recorded>")
    else
        println("Iter\tP1 Work\tP2?\tP2 Work\tLB\tUB\tTotal Pricing Work\tSolutions Added")
        let added_per_iter = Dict{Int,Int}()
            for c in iteration_columns_data
                added_per_iter[c.iteration] = get(added_per_iter, c.iteration, 0) + c.num_added
            end
            for pa in pricing_iteration_analysis
                local iter = pa.iteration
                local num_added = get(added_per_iter, iter, 0)
                println("$(iter)\t$(round(pa.phase1_work, digits=2))\t$(pa.went_phase2)\t$(round(pa.phase2_work, digits=2))\t$(pa.lb)\t$(pa.ub)\t$(round(pa.total_pricing_work, digits=2))\t$(num_added)")
            end
        end
    end
    
    # Calculate total work units
    total_work_units = total_master_work_units + total_pricing_work_units + total_update_work_units
    
    return (; selected_subsets = selected_subsets, selected_subsets_ip = selected_subsets_ip, 
            final_objective = final_objective, final_objective_lp = final_objective_lp, final_objective_ip = final_objective_ip,
            iterations = actual_iterations, status = status, status_ip = status_ip,
            total_master_work_units, total_pricing_work_units, total_update_work_units,
            total_solutions_added = total_added, iteration_pricing_data = iteration_pricing_data,
            cost_vector = cost_vector, z_matrix = z_matrix, v_variables = v_variables,
            iteration_master_data = iteration_master_data, iteration_columns_data = iteration_columns_data,
            type1_metrics = type1_metrics, type2_metrics = type2_metrics, pricing_iteration_analysis = pricing_iteration_analysis)
end

# =============================================================================
# MAIN EXECUTION
# =============================================================================

# Load data and generate scenarios
println(" Loading data and generating scenarios...")

# Load instance data (hardcoded for capa11.txt)
file_path = joinpath(@__DIR__, "capa11.txt")
if !isfile(file_path)
    println("❌ Error: Data file not found!")
    println("Expected file: $file_path")
    exit(1)
end

# Read instance data (Beasley format for capa11.txt)
capacity, fixedcost, cost, demandmean = read_beasley_cap(file_path)

# Apply capacity factor and scaling factor
capacity = capacity .* capacity_factor
cost = cost .* scaling_factor

# Get dimensions
nloc = length(capacity)
ncust = length(demandmean)

println("  Instance loaded: $nloc facilities, $ncust customers")
println("  Applied capacity factor of $capacity_factor to all facilities")
println("  Applied scaling factor of $scaling_factor to transportation costs")
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
 max_iter_test = 10  # Maximum iterations for testing
pricing_type_test = 2  # Pricing formulation type: 1 = original, 2 = alternative
init_type_test = 3  # Initialization type: 1 = each scenario as own subset, 2 = single subset (first scenario), 3 = candidate-based partition (default), 4 = all scenarios in one subset, 5 = equal-sized partition
use_solution_pool_flag = 1  # Toggle multi-column generation: 1 = ON (true), 0 = OFF (false)
use_solution_pool_test = (use_solution_pool_flag == 1)

println(" Testing SF Column Generation Algorithm")
println("Test Parameters:")
println("  nscen: $S_input")
println("  K: $K_test")
println("  max_iterations: $max_iter_test")
println("  pricing_type: $pricing_type_test")
println("  init_type: $init_type_test")
println("  use_solution_pool (flag=$use_solution_pool_flag): $use_solution_pool_test")
println()

# Run SF Column Generation Algorithm
res = sf_column_generation_algorithm(
    scenarios, nloc, ncust, capacity, cost, fixedcost, 
    unmet_pen, scaling_factor, K_test, max_iter_test, pricing_type_test, init_type_test; use_solution_pool=true)

println()
println("  SF Column Generation Results:")
println("  Iterations: $(res.iterations)")
println("  Final objective (LP): $(res.final_objective_lp)")
if isfinite(res.final_objective_ip)
    println("  Final objective (IP): $(res.final_objective_ip)")
    println("  Gap: $(round(100 * (res.final_objective_ip - res.final_objective_lp) / res.final_objective_lp, digits=2))%")
else
    println("  Final objective (IP): Not solved")
end
println("  Selected subsets (LP): $(res.selected_subsets)")
if !isempty(res.selected_subsets_ip)
    println("  Selected subsets (IP): $(res.selected_subsets_ip)")
end
println("  Status (LP): $(res.status)")
println("  Status (IP): $(res.status_ip)")
println("  Total Work Units: $(res.total_master_work_units + res.total_pricing_work_units + res.total_update_work_units)")
println("    - Master Problem: $(res.total_master_work_units)")
println("    - Pricing Subproblem: $(res.total_pricing_work_units)")
println("    - Master Update: $(res.total_update_work_units)")
println("  Total Solutions Added: $(res.total_solutions_added)")

# Display pricing objective table
println()
println("📊 SF PRICING SUBPROBLEM ANALYSIS:")
println("┌─────────────┬─────────────────────┐")
println("│ Iteration   │ obj_pricing         │")
println("├─────────────┼─────────────────────┤")

for data in res.iteration_pricing_data
    obj_pricing_str = string(round(data.obj_pricing, digits=6))
    println("│ $(lpad(data.iteration, 11)) │ $(lpad(obj_pricing_str, 19)) │")
end

println("└─────────────┴─────────────────────┘")

# Debug: Print all subsets
println()
println("All Generated Subsets:")
println("Total subsets: $(length(res.cost_vector))")
for (idx, subset_cost) in enumerate(res.cost_vector)
    subset_vector = res.z_matrix[:, idx]
    selected_scenarios = findall(x -> x > 0.5, subset_vector)
    println("  Subset $idx: scenarios $selected_scenarios (cost: $(round(subset_cost, digits=2)))")
end

# Generate output files (similar to CG_AF1.jl)
total_work_units = res.total_master_work_units + res.total_pricing_work_units + res.total_update_work_units
instance_name = "capa11"

# Generate results file (include both LP and IP objectives)
output_filename = "CG_SF_$(instance_name)_K$(K_test)_S$(S_input)_$(variance_input)_results.txt"
open(output_filename, "w") do io
    ip_obj_str = isfinite(res.final_objective_ip) ? string(round(res.final_objective_ip, digits=2)) : "NaN"
    gap_val = isfinite(res.final_objective_ip) ? round(100 * (res.final_objective_ip - res.final_objective_lp) / res.final_objective_lp, digits=2) : NaN
    println(io, "$(instance_name).txt\t$nloc\t$ncust\t$K_test\t$S_input\t$(res.iterations)\t$total_work_units\t$(res.total_solutions_added)\t$(res.status)\t0.00\t$(round(res.total_master_work_units, digits=2))\t$(round(res.total_pricing_work_units, digits=2))\t$(round(res.total_update_work_units, digits=2))\t$(round(res.final_objective_lp, digits=2))\t$ip_obj_str\t$gap_val")
end

# Generate iteration tracking file
iteration_filename = "CG_SF_$(instance_name)_K$(K_test)_S$(S_input)_$(variance_input)_iterations.txt"
open(iteration_filename, "w") do io
    for entry in res.iteration_master_data
        println(io, "$(entry.iteration)\t$(round(entry.master_obj, digits=2))")
    end
end

# Generate subproblem comparison file
comparison_filename = "CG_SF_$(instance_name)_K$(K_test)_S$(S_input)_$(variance_input)_subproblem_comparison.txt"
open(comparison_filename, "w") do io
    if !isempty(res.type1_metrics) && !isempty(res.type2_metrics)
        # Calculate averages manually
        avg_type1_work = sum([m.work_units for m in res.type1_metrics]) / length(res.type1_metrics)
        avg_type1_nodes = sum([m.nodes for m in res.type1_metrics]) / length(res.type1_metrics)
        avg_type2_work = sum([m.work_units for m in res.type2_metrics]) / length(res.type2_metrics)
        avg_type2_nodes = sum([m.nodes for m in res.type2_metrics]) / length(res.type2_metrics)
        
        println(io, "$(instance_name).txt\t$S_input\t$K_test\t$(round(avg_type1_work, digits=3))\t$(round(avg_type1_nodes, digits=3))\t$(round(avg_type2_work, digits=3))\t$(round(avg_type2_nodes, digits=3))")
    else
        println(io, "$(instance_name).txt\t$S_input\t$K_test\t0.000\t0.000\t0.000\t0.000")
    end
end

println()
println("="^80)
gap_str = isfinite(res.final_objective_ip) ? string(round(100 * (res.final_objective_ip - res.final_objective_lp) / res.final_objective_lp, digits=2)) : "N/A"
ip_obj_str = isfinite(res.final_objective_ip) ? string(round(res.final_objective_ip, digits=2)) : "N/A"
println("COLUMN GENERATION SUMMARY")
println("="^80)
println("Instance: $(instance_name).txt")
println("Facilities: $nloc")
println("Customers: $ncust")
println("K: $K_test")
println("S: $S_input")
println("Variance: $variance_input")
println("Iterations: $(res.iterations)")
println("Status (LP): $(res.status)")
println("Status (IP): $(res.status_ip)")
println("Final Objective (LP): $(round(res.final_objective_lp, digits=2))")
println("Final Objective (IP): $ip_obj_str")
println("Gap: $gap_str%")
println("Total Work Units: $total_work_units")
println("  - Master Problem: $(res.total_master_work_units)")
println("  - Pricing Subproblem: $(res.total_pricing_work_units)")
println("  - Master Update: $(res.total_update_work_units)")
println("Total Solutions Added: $(res.total_solutions_added)")
println("Output file: $output_filename")
println("="^80)

