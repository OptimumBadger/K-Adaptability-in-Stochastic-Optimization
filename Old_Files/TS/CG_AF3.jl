# Column Generation - Transmission Switching (Skeleton)

using JuMP, Gurobi, Random, Distributions, LinearAlgebra
using XLSX, DataFrames
using PowerModels



Random.seed!(1234) 

# ----------------------------- CLI parsing -----------------------------
if length(ARGS) != 3
    println("Usage: julia CG_AF3.jl <K> <S> <variance>")
    println("Example: julia CG_AF3.jl 3 50 bernoulli")
    println("Variance options: low, high, bernoulli")
    exit(1)
end

const K_input = parse(Int, ARGS[1])
const S_input = parse(Int, ARGS[2])
const variance_input = ARGS[3]

if !(variance_input in ["low", "high", "bernoulli"]) 
    println("❌ Invalid variance parameter: $variance_input")
    println("Valid options: low, high, bernoulli")
    exit(1)
end

println("="^80)
println("COLUMN GENERATION - Transmission Switching (Skeleton)")
println("Parameters: K=$K_input, S=$S_input, Variance=$variance_input")
println("Instance: <MATPOWER case path>")
println("Pricing Types: 1 (with m^s), 2 (compact)")
println("="^80)


# ----------------------- Data loading and helpers ----------------------

function parameters_TSpm(data; line_capacity_factor::Float64=1.0)
    # Sets
    B = data["bus"]
    L = data["branch"]
    G = data["gen"]

    # Parameters
    B_ij   = zeros(length(L))      # susceptance per line
    fbar_ij = zeros(length(L))     # thermal limit per line
    p_max  = zeros(length(B))      # max generation per bus
    p_min  = zeros(length(B))      # min generation per bus
    c      = zeros(length(B))      # linear gen cost per bus

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

function generate_demand_scenario_powerflow(nbus::Int, demandmean::Vector{Float64}, variance_type::String)
    if variance_type == "low"
        sigma = rand(Uniform(0.1, 0.2))
    elseif variance_type == "high"
        sigma = rand(Uniform(0.2, 0.3))
    elseif variance_type == "bernoulli"
        sigma = rand(Uniform(0.1, 0.2))
    else
        error("Invalid variance type: $variance_type")
    end
    global_noise = randn()
    d = zeros(nbus)
    for i in 1:nbus
        z = randn()
        val = demandmean[i] + sigma*z*demandmean[i] + sigma*global_noise*demandmean[i]
        if variance_type == "bernoulli"
            val *= (rand(Bernoulli(0.8)) == 1 ? 1.0 : 0.0)
        end
        d[i] = max(0.0, val)
    end
    return d
end

# --------------------- Cost evaluation r_s(x) (TS) ---------------------

"""
evaluate_ts_operational_cost(x::Vector{Float64}, scenario_demand::Vector{Float64}, data)
  For a fixed switch plan x (binary per candidate line), solve DC-OPF with unmet.
  Uses parameters_TSpm(data) to obtain (B_ij, fbar_ij, p_max, p_min, A, c).
  Returns (cost::Float64, work_units::Float64).
"""
function evaluate_ts_operational_cost(x::Vector{Float64}, scenario_demand::Vector{Float64}, data)
    B = data["bus"]
    L = data["branch"]
    nb = length(B)
    nl = length(L)
    B_ij, fbar_ij, p_max, p_min, A, c_g = parameters_TSpm(data)
    ref = 1
    penalty = 5000.0

    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 0)

    # Variables
    @variable(model, p_g[1:nb])
    @variable(model, f[1:nl])
    @variable(model, theta[1:nb])
    @variable(model, unmet[1:nb] >= 0)

    # Generator bounds per bus
    for i in 1:nb
        @constraint(model, p_g[i] >= p_min[i])
        @constraint(model, p_g[i] <= p_max[i])
    end

    # Line capacity + DC flow with fixed x (match TS1 approach)
    # Treat x as binary 0/1
    for (branch_id, branch) in L
        l = parse(Int, branch_id)
        xl = round(Int, x[l])
        if xl == 1
            @constraint(model, -fbar_ij[l] <= f[l])
            @constraint(model,  f[l] <= fbar_ij[l])
            i = branch["f_bus"]
            j = branch["t_bus"]
            @constraint(model, f[l] == B_ij[l] * (theta[i] - theta[j]))
        else
            @constraint(model, f[l] == 0)
        end
    end

    # Power balance (outgoing - incoming)
    for i in 1:nb
        outgoing = 0.0
        incoming = 0.0
        for (branch_id, branch) in L
            l = parse(Int, branch_id)
            if branch["f_bus"] == i
                outgoing += f[l]
            end
            if branch["t_bus"] == i
                incoming += f[l]
            end
        end
        @constraint(model, p_g[i] - scenario_demand[i] + unmet[i] == outgoing - incoming)
    end

    # Reference angle
    @constraint(model, theta[ref] == 0)

    # Objective
    @objective(model, Min, sum(c_g[i] * p_g[i] for i in 1:nb) + penalty * sum(unmet[i] for i in 1:nb))

    optimize!(model)

    work_units = 0.0
    try
        work_units = JuMP.MOI.get(backend(model), Gurobi.ModelAttribute("Work"))
    catch
    end

    if termination_status(model) == MOI.OPTIMAL
        return objective_value(model), work_units
    else
        return Inf, work_units
    end
end

"""
solve_powerflow_for_binary(data, demand)
  Solve a TS MILP from scratch for a single scenario demand, returning a binary switch plan.
  Structure inspired by TS1.offline_calc_direct.
  Returns (x_binary::Vector{Float64}, obj::Float64).
"""
function solve_powerflow_for_binary(data, demand::Vector{Float64}; alpha::Float64=5000.0, log_file=nothing, work_limit::Float64=120.0)
    B = data["bus"]
    L = data["branch"]
    nb = length(B)
    nl = length(L)
    B_ij, fbar_ij, p_max, p_min, A, c = parameters_TSpm(data)

    model = Model(Gurobi.Optimizer)
    if log_file !== nothing
        set_optimizer_attribute(model, "LogFile", log_file)
        set_optimizer_attribute(model, "LogToConsole", 0)
    else
        set_optimizer_attribute(model, "OutputFlag", 0)
    end
    if isfinite(work_limit) && work_limit > 0.0
        set_optimizer_attribute(model, "WorkLimit", work_limit)
    end

    @variable(model, p_g[1:nb])
    @variable(model, f[1:nl])
    @variable(model, theta[1:nb])
    @variable(model, x[1:nl], Bin)
    @variable(model, unmet[1:nb] >= 0)

    # Generator limits
    for i in 1:nb
        @constraint(model, p_g[i] >= p_min[i])
        @constraint(model, p_g[i] <= p_max[i])
    end

    # Line capacities
    for (branch_id, branch) in L
        l = parse(Int, branch_id)
        @constraint(model, -fbar_ij[l] * x[l] <= f[l])
        @constraint(model,  f[l] <= fbar_ij[l] * x[l])
    end

    # DC power flow with big-M
    for (branch_id, branch) in L
        l = parse(Int, branch_id)
        i = branch["f_bus"]
        j = branch["t_bus"]
        Mij = -12.6 * B_ij[l]
        @constraint(model, B_ij[l] * (theta[i] - theta[j]) - Mij * (1 - x[l]) <= f[l])
        @constraint(model, f[l] <= B_ij[l] * (theta[i] - theta[j]) + Mij * (1 - x[l]))
    end

    # Power balance
    for i in 1:nb
        outgoing = 0.0
        incoming = 0.0
        for (branch_id, branch) in L
            l = parse(Int, branch_id)
            if branch["f_bus"] == i
                outgoing += f[l]
            end
            if branch["t_bus"] == i
                incoming += f[l]
            end
        end
        @constraint(model, p_g[i] - demand[i] + unmet[i] == outgoing - incoming)
    end

    @constraint(model, theta[1] == 0)

    @objective(model, Min, sum(c[i] * p_g[i] for i in 1:nb) + alpha * sum(unmet[i] for i in 1:nb))

    optimize!(model)

    if termination_status(model) == MOI.OPTIMAL
        x_val = value.(x)
        x_bin = Float64.(round.(Int, x_val))
        return x_bin, objective_value(model)
    else
        return Float64[], Inf
    end
end

function build_cost_matrix_TS(scenarios::Vector{Vector{Float64}}, switch_plans::Vector{Vector{Float64}}, data)
    nscen = length(scenarios)
    nsol  = length(switch_plans)
    println("Building TS cost matrix for $nscen scenarios and $nsol switch plans…")
    C = zeros(Float64, nscen, nsol)
    total_work = 0.0
    for s in 1:nscen
        demand_s = scenarios[s]
        for k in 1:nsol
            x = switch_plans[k]
            cost, wu = evaluate_ts_operational_cost(x, demand_s, data)
            C[s, k] = cost
            total_work += wu
        end
        if s % 10 == 0
            println("  Processed $s scenarios")
        end
    end
    println("TS cost matrix built.")
    return C, total_work
end

# =============================================================================
# ASSIGNMENT FORMULATION FUNCTIONS
# =============================================================================
function solve_assignment_formulation_TS(cost_matrix, K; integer_vars::Bool=false)
    nscen, nsol = size(cost_matrix)

    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 1)
    set_optimizer_attribute(model, "LogFile", "CG_AF3_master_K$(K_input)_S$(S_input).log")

    if integer_vars
        sigma = [@variable(model, binary = true) for x in 1:nsol]
        rho = [[@variable(model, binary = true) for x in 1:nsol] for s in 1:nscen]
    else
        sigma = [@variable(model, lower_bound = 0) for x in 1:nsol]
        rho = [[@variable(model, lower_bound = 0) for x in 1:nsol] for s in 1:nscen]
    end

    @objective(model, Min, sum(cost_matrix[s, x] * rho[s][x] for s in 1:nscen for x in 1:nsol))

    assignment_constraints = @constraint(model, [s in 1:nscen], sum(rho[s][x] for x in 1:nsol) == 1)
    k_constraint = @constraint(model, sum(sigma[x] for x in 1:nsol) <= K)
    @constraint(model, [s in 1:nscen, x in 1:nsol], rho[s][x] <= sigma[x])

    optimize!(model)

    work_units = 0.0
    try
        work_units = JuMP.MOI.get(backend(model), Gurobi.ModelAttribute("Work"))
    catch
    end

    if termination_status(model) == MOI.OPTIMAL
        return model, sigma, rho, objective_value(model), assignment_constraints, k_constraint, work_units
    else
        return model, sigma, rho, Inf, assignment_constraints, k_constraint, work_units
    end
end

# ---------------------------- Pricing stubs ----------------------------

const DEFAULT_OPT_WORKLIMIT = 20.0

"""
Collect up to 10 pool solutions with positive reduced cost and return their x-vectors and pool objective values.
"""
function collect_pool_rc_cap10(model, x, dual_mu, use_solution_pool)
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
            x_k = [MOI.get(opt, Gurobi.VariableAttribute("Xn"), JuMP.index(x[i])) for i in 1:length(x)]
            rc_k = pool_obj + dual_mu
            if rc_k > 1e-6
                push!(kept_solutions, x_k)
                push!(kept_objs, pool_obj)
            end
        end
    catch e
        println("  Warning: solution pool access failed: $e")
    end
    if isempty(kept_solutions)
        return Vector{Vector{Float64}}(), Float64[]
    end
    return kept_solutions, kept_objs
end

"""
Return pricing results with optional NodeCount.
"""
function return_pricing_payload(kept_solutions, kept_objs, pricing_work_units, return_nodes, model)
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

function af_pricing_type1_TS(data, scenarios, m_values, dual_lambda, dual_mu; use_solution_pool::Bool=true, work_limit::Float64=Inf, opt_worklimit::Float64=DEFAULT_OPT_WORKLIMIT, alpha::Float64=5000.0, return_nodes::Bool=false)
    # Parameters from PowerModels data (TS1 style)
    B_ij, fbar_ij, p_max, p_min, A, c = parameters_TSpm(data)
    L = data["branch"]
    nb = length(data["bus"])       # buses
    nl = length(L)                  # lines
    nscen = length(scenarios)

    # Big-M for DC equality relaxation when x_l = 0
    M_ij = [12.6 * abs(B_ij[l]) for l in 1:nl]

    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 1)
    set_optimizer_attribute(model, "LogFile", "CG_AF3_pricing_K$(K_input)_S$(S_input).log")
    if isfinite(opt_worklimit)
        set_optimizer_attribute(model, "WorkLimit", 10)
    end
    if use_solution_pool
        set_optimizer_attribute(model, "PoolSearchMode", 2)
        set_optimizer_attribute(model, "PoolSolutions", 10)
    end

    # Decision variables
    @variable(model, x[1:nl], Bin)                       # switch plan (columns)
    @variable(model, pi[1:nscen], Bin)                   # scenario selection
    #@variable(model, v[1:nl, 1:nscen], Bin)              # linking x and pi

    @variable(model, p_g[1:nb, 1:nscen])                 # generation
    @variable(model, r[1:nb, 1:nscen] >= 0)              # unmet demand
    @variable(model, f[1:nl, 1:nscen])                   # line flows
    @variable(model, theta[1:nb, 1:nscen])               # bus angles

    # Aux variables to linearize products with pi in objective
    @variable(model, pbar[1:nb, 1:nscen])
    @variable(model, rbar[1:nb, 1:nscen] >= 0)

    # Objective: sum_s [ - sum_i c_i pbar_i^s - alpha sum_i rbar_i^s + (lambda*)^s pi^s ]
    @objective(model, Max, sum(dual_lambda[s] * pi[s] for s in 1:nscen)
                        - sum(c[i] * pbar[i,s] for i in 1:nb for s in 1:nscen)
                        - alpha * sum(rbar[i,s] for i in 1:nb for s in 1:nscen))

    @constraint(model, [i in 1:nb, s in 1:nscen], pbar[i, s] >= min(p_min[i],0))

    # Flow balance per bus and scenario
    for s in 1:nscen
        d_s = scenarios[s]
        for i in 1:nb
            outgoing = zero(A[1,1])
            incoming = zero(A[1,1])
            for (branch_id, branch) in L
                l = parse(Int, branch_id)
                if branch["f_bus"] == i
                    outgoing += f[l, s]
                end
                if branch["t_bus"] == i
                    incoming += f[l, s]
                end
            end
            @constraint(model, p_g[i, s] + r[i, s] - d_s[i] == outgoing - incoming)
        end
    end

    # Generator bounds
    for i in 1:nb, s in 1:nscen
        @constraint(model, p_g[i, s] >= p_min[i])
        @constraint(model, p_g[i, s] <= p_max[i])
    end

    # Line capacity and DC constraints with x
    for (branch_id, branch) in L
        l = parse(Int, branch_id)
        i = branch["f_bus"]
        j = branch["t_bus"]
        for s in 1:nscen
            @constraint(model, -fbar_ij[l] * x[l] <= f[l, s])
            @constraint(model,  f[l, s] <= fbar_ij[l] * x[l])
            @constraint(model, B_ij[l] * (theta[i, s] - theta[j, s]) - M_ij[l] * (1 - x[l]) <= f[l, s])
            @constraint(model, f[l, s] <= B_ij[l] * (theta[i, s] - theta[j, s]) + M_ij[l] * (1 - x[l]))
        end
    end

    # Reference angle per scenario (fix bus 1)
    for s in 1:nscen
        @constraint(model, theta[1, s] == 0)
    end

    # Pricing key inequality per scenario: lambda* - sum c p_g - alpha sum r >= m^s(1 - pi^s)
    for s in 1:nscen
        @constraint(model, dual_lambda[s]
            - sum(c[i] * p_g[i, s] for i in 1:nb)
            - alpha * sum(r[i, s] for i in 1:nb)
            >= m_values[s] * (1 - pi[s]))
    end

    # v linking constraints (like AF1): v_l^s is 1 when both x_l and pi_s are 1
    #@constraint(model, [l in 1:nl, s in 1:nscen], v[l, s] <= x[l])
    #@constraint(model, [l in 1:nl, s in 1:nscen], v[l, s] <= pi[s])
    #@constraint(model, [l in 1:nl, s in 1:nscen], v[l, s] >= x[l] + pi[s] - 1)

    # Linearization for pbar and rbar with pi
    for i in 1:nb, s in 1:nscen
        # pbar ≤ p_g ; pbar ≤ p_max π ; pbar ≥ p_g − p_max (1 − π)
        @constraint(model, pbar[i, s] <= p_g[i, s])
        @constraint(model, pbar[i, s] <= p_max[i] * pi[s])
        @constraint(model, pbar[i, s] >= p_g[i, s] - p_max[i] * (1 - pi[s]))
        # rbar ≤ r ; rbar ≤ d_i^s π ; rbar ≥ r − d_i^s (1 − π)
        d_is = scenarios[s][i]
        @constraint(model, rbar[i, s] <= r[i, s])
        @constraint(model, rbar[i, s] <= d_is * pi[s])
        @constraint(model, rbar[i, s] >= r[i, s] - d_is * (1 - pi[s]))
    end

    # Solve Phase 1 (opt work limit)
    optimize!(model)

    pricing_work_units = 0.0
    try
        pricing_work_units = MOI.get(backend(model), Gurobi.ModelAttribute("Work"))
    catch
    end

    # Collect Phase 1 pool solutions
    kept_solutions = Vector{Vector{Float64}}()
    kept_objs = Float64[]

    if use_solution_pool
        opt = backend(model)
        try
            solcount = MOI.get(opt, Gurobi.ModelAttribute("SolCount"))
            max_solutions = min(10, solcount)
            for k in 0:max_solutions-1
                MOI.set(opt, MOI.RawOptimizerAttribute("SolutionNumber"), k)
                pool_obj = MOI.get(opt, Gurobi.ModelAttribute("PoolObjVal"))
                x_k = [MOI.get(opt, Gurobi.VariableAttribute("Xn"), JuMP.index(x[l])) for l in 1:nl]
                rc_k = pool_obj + dual_mu
                if rc_k > 0
                    push!(kept_solutions, x_k)
                    push!(kept_objs, pool_obj)
                end
            end
        catch
        end
    else
        # Single incumbent
        push!(kept_solutions, value.(x))
        push!(kept_objs, objective_value(model))
    end

    nodes = 0
    if return_nodes
        try
            nodes = MOI.get(backend(model), Gurobi.ModelAttribute("NodeCount"))
        catch
        end
        return kept_solutions, kept_objs, pricing_work_units, nodes
    else
        return kept_solutions, kept_objs, pricing_work_units
    end
end


# ------------------------ Scenario Big-M calculator ------------------------
function big_m_calculator_TS(dual_lambda::Float64, c::Vector{Float64}, p_min::Vector{Float64}, p_max::Vector{Float64}, demand::Vector{Float64}, alpha::Float64; log_file=nothing)
    nb = length(demand)
    @assert length(c) == nb "c length must equal number of buses"
    @assert length(p_min) == nb && length(p_max) == nb "p_min/p_max must have length nb"

    model = Model(Gurobi.Optimizer)
    if log_file !== nothing
        set_optimizer_attribute(model, "LogFile", log_file)
        set_optimizer_attribute(model, "LogToConsole", 0)
    else
        set_optimizer_attribute(model, "OutputFlag", 0)
    end

    @variable(model, p_g[1:nb])
    @variable(model, r[1:nb] >= 0)

    # Bounds
    for i in 1:nb
        @constraint(model, p_g[i] >= p_min[i])
        @constraint(model, p_g[i] <= p_max[i])
        @constraint(model, r[i] <= demand[i])
    end

    # Minimize the LHS directly
    @objective(model, Min, dual_lambda - sum(c[i] * p_g[i] for i in 1:nb) - alpha * sum(r[i] for i in 1:nb))

    optimize!(model)

    if termination_status(model) == MOI.OPTIMAL
        return objective_value(model)
    else
        return -Inf
    end
end

"""
af_pricing_type2_TS(...)
  Compact pricing (Image 2). Returns (solutions::Vector{Vector{Float64}}, pool_objs::Vector{Float64}, work_units::Float64)
"""
function af_pricing_type2_TS(data, scenarios, dual_lambda, dual_mu; use_solution_pool::Bool=true, work_limit::Float64=Inf, opt_worklimit::Float64=DEFAULT_OPT_WORKLIMIT, alpha::Float64=5000.0, return_nodes::Bool=false, analysis::Dict{Symbol,Any}=Dict{Symbol,Any}())
    # TS parameters
    B_ij, fbar_ij, p_max, p_min, A, c = parameters_TSpm(data)
    L = data["branch"]
    nb = length(data["bus"])        # buses
    nl = length(L)                   # lines
    nscen = length(scenarios)

    # Big-M following instruction: M_ij = -12.6 * B_ij
    M_ij = [-12.6 * B_ij[l] for l in 1:nl]

    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 1)
    set_optimizer_attribute(model, "LogFile", "CG_AF3_pricing_K$(K_input)_S$(S_input).log")
    if use_solution_pool
        set_optimizer_attribute(model, "PoolSearchMode", 2)
        set_optimizer_attribute(model, "PoolSolutions", 10)
    end

    # Decision variables (Type 2: x and per-scenario x_s linked by pi)
    @variable(model, x[1:nl], Bin)
    @variable(model, x_s[1:nl, 1:nscen], Bin)
    @variable(model, pi[1:nscen], Bin)

    # TS operational variables per scenario
    @variable(model, p_g[1:nb, 1:nscen])
    @variable(model, r[1:nb, 1:nscen] >= 0)
    @variable(model, f[1:nl, 1:nscen])
    @variable(model, theta[1:nb, 1:nscen])

    # Objective: maximize sum_s lambda_s pi_s - sum_i c_i p_g - alpha sum_i r_i
    @objective(model, Max, sum(dual_lambda[s] * pi[s] for s in 1:nscen)
                        - sum(c[i] * p_g[i,s] for i in 1:nb for s in 1:nscen)
                        - alpha * sum(r[i,s] for i in 1:nb for s in 1:nscen))

    # Linking constraints: x_s ≤ pi ; x - x_s ≤ (1- pi) ; x - x_s ≥ -(1 - pi)
    @constraint(model, [l in 1:nl, s in 1:nscen], x_s[l, s] <= pi[s])
    @constraint(model, [l in 1:nl, s in 1:nscen], x[l] - x_s[l, s] <= (1 - pi[s]))
    @constraint(model, [l in 1:nl, s in 1:nscen], x[l] - x_s[l, s] >= -(1 - pi[s]))

    # Generator bounds per scenario
    for i in 1:nb, s in 1:nscen
        @constraint(model, p_g[i, s] >= p_min[i]*pi[s])
        @constraint(model, p_g[i, s] <= p_max[i]*pi[s])
    end

    # Line capacity and DC constraints with x_s
    for (branch_id, branch) in L
        l = parse(Int, branch_id)
        i = branch["f_bus"]
        j = branch["t_bus"]
        for s in 1:nscen
            @constraint(model, -fbar_ij[l] * x_s[l, s] <= f[l, s])
            @constraint(model,  f[l, s] <= fbar_ij[l] * x_s[l, s])
            @constraint(model, B_ij[l] * (theta[i, s] - theta[j, s]) - M_ij[l] * (1 - x_s[l, s]) <= f[l, s])
            @constraint(model, f[l, s] <= B_ij[l] * (theta[i, s] - theta[j, s]) + M_ij[l] * (1 - x_s[l, s]))
        end
    end

    # Power balance per bus and scenario
    for s in 1:nscen
        d_s = scenarios[s]
        for i in 1:nb
            outgoing = 0.0
            incoming = 0.0
            for (branch_id, branch) in L
                l = parse(Int, branch_id)
                if branch["f_bus"] == i
                    outgoing += f[l, s]
                end
                if branch["t_bus"] == i
                    incoming += f[l, s]
                end
            end
            @constraint(model, p_g[i, s] + r[i, s] - d_s[i]*pi[s] == outgoing - incoming)
        end
        @constraint(model, theta[1, s] == 0)
    end

    # Key inequality: dual_lambda[s] * pi[s] - cost terms >= 0 (Type 2 form)
    @constraint(model, [s in 1:nscen],
        dual_lambda[s] * pi[s]
        - sum(c[i] * p_g[i, s] for i in 1:nb)
        - alpha * sum(r[i, s] for i in 1:nb) >= 0)

    # Two-phase solve like AF1
    pricing_work_units = 0.0
    if !isempty(analysis)
        analysis[:phase1_work] = 0.0
        analysis[:phase2_work] = 0.0
        analysis[:went_phase2] = false
        analysis[:lb] = NaN
        analysis[:ub] = NaN
    end

    # Phase 1 (AF1-style quick optimal try)
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

        # Collect pool and check RC-improving incumbents
        opt = backend(model)
        kept_solutions = Vector{Vector{Float64}}()
        kept_objs = Float64[]
        if use_solution_pool
            try
                solcount = MOI.get(opt, Gurobi.ModelAttribute("SolCount"))
                max_solutions = min(10, solcount)
                for k in 0:max_solutions-1
                    MOI.set(opt, MOI.RawOptimizerAttribute("SolutionNumber"), k)
                    pool_obj = MOI.get(opt, Gurobi.ModelAttribute("PoolObjVal"))
                    x_k = [MOI.get(opt, Gurobi.VariableAttribute("Xn"), JuMP.index(x[l])) for l in 1:nl]
                    rc_k = pool_obj + dual_mu
                    if rc_k > 0
                        push!(kept_solutions, x_k)
                        push!(kept_objs, pool_obj)
                    end
                end
            catch
            end
        else
            push!(kept_solutions, value.(x))
            push!(kept_objs, objective_value(model))
        end

        if termination_status(model) == MOI.OPTIMAL
            println("  Phase 1: Optimal status")
        else
            println("  Phase 1: Non-optimal status, checking incumbents")
        end

        if !isempty(kept_solutions)
            println("  Phase 1: Found $(length(kept_solutions)) RC-improving solutions")
            for (idx, objv) in enumerate(kept_objs)
                rc = objv + dual_mu
                println("    -> pool obj[$idx] = $(round(objv, digits=6)), rc = $(round(rc, digits=6))")
            end
            if !isempty(analysis); analysis[:went_phase2] = false; end
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
        else
            println("  Phase 1: No RC-improving solutions, proceeding to Phase 2")
        end
    end

    # Phase 2 (Optimality push, AF1-style)
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
            kept_solutions, kept_objs = collect_pool_rc_cap10(model, x, dual_mu, use_solution_pool)
            return return_pricing_payload(kept_solutions, kept_objs, pricing_work_units, return_nodes, model)
        else
            println("  Phase 2: Work units exhausted. Reporting bounds…")
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
            return return_pricing_payload(Vector{Vector{Float64}}(), Float64[], pricing_work_units, return_nodes, model)
        end
    end

    return return_pricing_payload(Vector{Vector{Float64}}(), Float64[], pricing_work_units, return_nodes, model)
end

function solve_pricing_subproblem_TS(data, scenarios, dual_lambda, dual_mu, pricing_type::Int; use_solution_pool::Bool=true, work_limit::Float64=Inf, opt_worklimit::Float64=DEFAULT_OPT_WORKLIMIT, alpha::Float64=5000.0)
    nscen = length(scenarios)
    if pricing_type == 1
        B_ij, fbar_ij, p_max, p_min, A, c = parameters_TSpm(data)
        m_values = zeros(nscen)
        for s in 1:nscen
            m_values[s] = big_m_calculator_TS(dual_lambda[s], c, p_min, p_max, scenarios[s], alpha)
        end
        sols, objs, pwu = af_pricing_type1_TS(data, scenarios, m_values, dual_lambda, dual_mu; use_solution_pool=use_solution_pool, work_limit=work_limit, opt_worklimit=opt_worklimit, alpha=alpha)
        return sols, objs, m_values, pwu
    elseif pricing_type == 2
        m_values = zeros(nscen)
        sols, objs, pwu = af_pricing_type2_TS(data, scenarios, dual_lambda, dual_mu; use_solution_pool=use_solution_pool, work_limit=work_limit, opt_worklimit=opt_worklimit, alpha=alpha)
        return sols, objs, m_values, pwu
    else
        error("Invalid pricing_type")
    end
end

# ----------------------- Column generation loop -----------------------



function column_generation_TS(scenarios, data; K::Int=K_input, max_iterations::Int=100, pricing_type::Int=2, use_solution_pool::Bool=true, init_type::Int=1)
    println(" Starting Column Generation for TS (AF master)")
    nscen = length(scenarios)

    # Global work budget bookkeeping
    total_work_budget = 100.0
    total_master_work_units = 0.0
    total_pricing_work_units = 0.0
    total_update_work_units = 0.0
    last_master_work_cumulative = 0.0

    # Tracking tables
    iteration_master_data = []
    iteration_columns_data = []
    type1_metrics = []
    type2_metrics = []
    pricing_iteration_analysis = []

    # Stage 1: Initialization (Type 1 solves each scenario; Type 2 single vector)
    println(" Stage 1: Initialization")
    nl = length(data["branch"])
    if init_type == 1
        println("  Initialization Type 1: Solving scenarios with 120 work units until we collect $nscen solutions…")
        switch_plans = Vector{Vector{Float64}}()
        init_objs = Float64[]
        s_idx = 1
        attempts = 0
        while length(switch_plans) < nscen && attempts < 10*nscen
            println("    Attempt $(attempts+1): solving scenario $(s_idx)/$(nscen)…")
            x_bin, obj = solve_powerflow_for_binary(data, scenarios[s_idx]; work_limit=120.0)
            if !isempty(x_bin)
                push!(switch_plans, x_bin)
                push!(init_objs, obj)
                println("      ✓ accepted solution $(length(switch_plans))/$(nscen)")
            else
                println("      ✗ not optimal within work limit; trying next scenario")
            end
            s_idx = s_idx % nscen + 1
            attempts += 1
        end
        if length(switch_plans) < nscen
            println("  Could not collect $nscen optimal solutions within attempts; filling remaining with all-lines ON")
            nl_local = length(data["branch"])
            while length(switch_plans) < nscen
                push!(switch_plans, ones(nl_local))
                push!(init_objs, Inf)
            end
        end
        println("  Generated $(length(switch_plans)) initial binary vectors")
    else
        println("  Initialization Type 2: Starting with all-lines ON…")
        switch_plans = [ones(nl)]
        println("  Generated 1 initial binary vector (all ones)")
    end

    # Build initial cost matrix
    cost_matrix, init_work = build_cost_matrix_TS(scenarios, switch_plans, data)
    total_update_work_units += init_work

    # Build master
    master_model, sigma, rho, master_obj, assignment_constraints, k_constraint, master_work =
        solve_assignment_formulation_TS(cost_matrix, K; integer_vars=false)
    total_master_work_units += master_work
    println("  Initial master objective: $master_obj")

    for iteration in 1:max_iterations
        println("\033[34m================ ITERATION ", iteration, " ===================\033[0m")
        used_work = total_master_work_units + total_pricing_work_units + total_update_work_units
        available_work = max(0.0, total_work_budget - used_work)
        println(" Available work units: $(round(available_work, digits=2)) / $(total_work_budget)")

        # Stage 1: solve master
        println(" Stage 1: Solving master…")
        master_work_units = 0.0
        try
            optimize!(master_model)
            local cum = MOI.get(backend(master_model), Gurobi.ModelAttribute("Work"))
            master_work_units = max(0.0, cum - last_master_work_cumulative)
            last_master_work_cumulative = cum
            total_master_work_units += master_work_units
        catch e
            println("  Warning: unable to get master work units: $e")
        end
        if termination_status(master_model) != MOI.OPTIMAL
            println("  Master failed. Terminating.")
            break
        end
        master_obj = objective_value(master_model)
        println("  Master objective: $(round(master_obj, digits=6))  (+$(round(master_work_units, digits=2)) work)")
        push!(iteration_master_data, (iteration=iteration, master_obj=master_obj))

        # Stage 2: dual extraction
        println(" Stage 2: Extracting duals…")
        dual_lambda = dual.(assignment_constraints)
        dual_mu = dual(k_constraint)
        println("  μ = $(round(dual_mu, digits=6))")

        # Stage 3: pricing
        println(" Stage 3: Pricing…")
        used_work = total_master_work_units + total_pricing_work_units + total_update_work_units
        remaining_work = max(0.0, total_work_budget - used_work)
        if remaining_work <= 0.0
            println("  Work budget exhausted. Terminating.")
            break
        end
        sols = Vector{Vector{Float64}}()
        objs = Float64[]
        if iteration <= 10
            println("  Solving both Type 1 and Type 2 subproblems for comparison (iteration $(iteration)/10)")
            try
                # Type 1: compute m^s and solve (comparison only)
                nscen = length(scenarios)
                B_ij, fbar_ij, p_max, p_min, A, c = parameters_TSpm(data)
                m_values = zeros(nscen)
                for s in 1:nscen
                    m_values[s] = big_m_calculator_TS(dual_lambda[s], c, p_min, p_max, scenarios[s], 5000.0)
                end
                type1_solutions, type1_objs, type1_work_units, type1_nodes = af_pricing_type1_TS(data, scenarios, m_values, dual_lambda, dual_mu; use_solution_pool=use_solution_pool, return_nodes=true, work_limit=remaining_work, opt_worklimit=DEFAULT_OPT_WORKLIMIT)

                # Type 2: solve (algorithm uses these solutions)
                local analysis = Dict{Symbol,Any}()
                type2_solutions, type2_objs, type2_work_units, type2_nodes = af_pricing_type2_TS(data, scenarios, dual_lambda, dual_mu; use_solution_pool=use_solution_pool, return_nodes=true, work_limit=remaining_work, opt_worklimit=DEFAULT_OPT_WORKLIMIT, analysis=analysis)

                # Only count Type 2 work units
                total_pricing_work_units += type2_work_units

                # Use Type 2 solutions for algorithm
                sols = type2_solutions
                objs = type2_objs

                # Record/print metrics
                push!(type1_metrics, (work_units=type1_work_units, nodes=type1_nodes))
                push!(type2_metrics, (work_units=type2_work_units, nodes=type2_nodes))
                push!(pricing_iteration_analysis, (
                    iteration = iteration,
                    phase1_work = get(analysis, :phase1_work, NaN),
                    went_phase2 = get(analysis, :went_phase2, false),
                    phase2_work = get(analysis, :phase2_work, NaN),
                    lb = get(analysis, :lb, NaN),
                    ub = get(analysis, :ub, NaN),
                    total_pricing_work = type2_work_units,
                ))
                println("  Type 1 work units: $(round(type1_work_units, digits=2)), B&B nodes: $type1_nodes")
                println("  Type 2 work units: $(round(type2_work_units, digits=2)), B&B nodes: $type2_nodes")
                println("  Using Type 2 solutions for algorithm")
            catch e
                println("Error in subproblem comparison: $e")
                break
            end
        else
            sols_, objs_, _, pwu_ = solve_pricing_subproblem_TS(data, scenarios, dual_lambda, dual_mu, pricing_type; use_solution_pool=use_solution_pool, work_limit=remaining_work, opt_worklimit=DEFAULT_OPT_WORKLIMIT)
            if isempty(sols_)
                println("  No solutions returned by pricing. Terminating.")
                break
            end
            total_pricing_work_units += pwu_
            sols = sols_
            objs = objs_
            println("  Pricing work units: $(round(pwu_, digits=2))")
            push!(pricing_iteration_analysis, (
                iteration = iteration,
                phase1_work = NaN,
                went_phase2 = NaN,
                phase2_work = NaN,
                lb = NaN,
                ub = NaN,
                total_pricing_work = pwu_,
            ))
        end

        # Filter by reduced cost and cap to 10
        rc_ok = [idx for idx in 1:length(objs) if objs[idx] + dual_mu > 1e-6]
        if isempty(rc_ok)
            println("  No RC-improving solutions. Terminating.")
            break
        end
        keep = rc_ok[1:min(10, length(rc_ok))]
        println("  Will add the following columns (rank, obj, rc):")
        for (rank, idx) in enumerate(keep)
            local objv = objs[idx]
            local rcv = objv + dual_mu
            println("    $(rank)) obj=$(round(objv, digits=6))\trc=$(round(rcv, digits=6))")
        end
        sols = sols[keep]
        objs = objs[keep]

        # Stage 4: add columns
        println(" Stage 4: Adding $(length(sols)) columns…")
        local_added = 0
        local_update_work = 0.0
        for x_new in sols
            C_new, w_new = build_cost_matrix_TS(scenarios, [x_new], data)
            col_costs = vec(C_new[:, 1])
            local_update_work += w_new

            new_sigma = @variable(master_model, lower_bound = 0)
            new_rho = [@variable(master_model, lower_bound = 0) for s in 1:nscen]
            push!(sigma, new_sigma)
            for s in 1:nscen
                push!(rho[s], new_rho[s])
                set_objective_coefficient(master_model, new_rho[s], col_costs[s])
                set_normalized_coefficient(assignment_constraints[s], new_rho[s], 1.0)
                @constraint(master_model, new_rho[s] <= new_sigma)
            end
            set_normalized_coefficient(k_constraint, new_sigma, 1.0)
            cost_matrix = hcat(cost_matrix, col_costs)
            push!(switch_plans, x_new)
            local_added += 1
        end
        total_update_work_units += local_update_work
        println("  Update work units: $(round(local_update_work, digits=2))")
        push!(iteration_columns_data, (iteration=iteration, num_added=local_added))

        # Stage 5: convergence indicator
        obj_pricing = minimum(objs)
        reduced_cost = obj_pricing + dual_mu
        println(" Stage 5: pricing obj=$(round(obj_pricing, digits=6)), rc=$(round(reduced_cost, digits=6))")
        if reduced_cost < 1e-3
            println(" Converged (rc < 0)")
            break
        end
    end

    # Finalize
    optimize!(master_model)
    status = termination_status(master_model) == MOI.OPTIMAL ? "OPTIMAL" : "FAILED"
    final_obj = objective_value(master_model)
    total_added = isempty(iteration_columns_data) ? 0 : sum(t.num_added for t in iteration_columns_data)
    return (; selected_solutions = Int[], final_objective = final_obj, iterations = length(iteration_master_data), status,
            total_master_work_units, total_pricing_work_units, total_update_work_units,
            total_solutions_added = total_added)
end

# ------------------------------ Main block -----------------------------

println(" Loading TS data & scenarios…")

# Paths (mirror TS1.jl)
file_path = "case118Blumsack.m"
file_path_excel = "Random_Demands.xlsx"  # reserved for future use

# Existence checks (as in TS1)
if !isfile(file_path)
    println("❌ Error: MATPOWER file not found!")
    println("Expected file: $file_path")
    println("Please run from project root and ensure powerflow_instances/ is present.")
    exit(1)
end

println("Loading data from: $file_path")
network = PowerModels.parse_file(file_path)
println("✅ Network data loaded successfully!")
println("Network: $(length(network["bus"])) buses, $(length(network["branch"])) branches, $(length(network["gen"])) generators")

# Demand mean from Excel (same method as TS1)
if !isfile(file_path_excel)
    error("❌ Error: Excel file not found! Expected: $file_path_excel")
end
case_name = "case118"
println("Loading demand data from: $file_path_excel (worksheet: $case_name)")
param_data = XLSX.readtable(file_path_excel, case_name) |> DataFrame
demandmean = Vector{Float64}(param_data[1, :])
nb = length(demandmean)
println("✅ Demand data loaded successfully! Number of buses: $nb")

println("  Generating $S_input scenarios (variance=$variance_input)…")
scenarios = Vector{Vector{Float64}}()
for s in 1:S_input
    push!(scenarios, generate_demand_scenario_powerflow(nb, demandmean, variance_input))
end
println("  Generated $(length(scenarios)) scenarios")

# Initial switch plan: all lines ON
nl = length(network["branch"])
switch_plans = [ones(nl)]

# Build initial cost matrix (single plan)
cost_matrix, total_work = build_cost_matrix_TS(scenarios, switch_plans, network)
println(" Initial evaluation work units: ", total_work)

# =============================================================================
# MAIN EXECUTION (mirrors AF1 structure and parameters)
# =============================================================================

println(" Testing Column Generation Algorithm (TS)")
K_test = K_input
max_iter_test = 100
pricing_type_test = 2
use_solution_pool_flag = 1
use_solution_pool_test = (use_solution_pool_flag == 1)

println("Test Parameters:")
println("  S: $S_input")
println("  K: $K_test")
println("  max_iterations: $max_iter_test")
println("  pricing_type: $pricing_type_test")
println("  use_solution_pool (flag=$use_solution_pool_flag): $use_solution_pool_test")
println()

# Run Column Generation (TS)
res = column_generation_TS(scenarios, network; K=K_test, max_iterations=max_iter_test, pricing_type=pricing_type_test, use_solution_pool=use_solution_pool_test, init_type=1)

# Calculate total work units
total_work_units = res.total_master_work_units + res.total_pricing_work_units + res.total_update_work_units

# Write compact results file
output_filename = "CG_AF3_case118_K$(K_test)_S$(S_input)_$(variance_input)_results.txt"
open(output_filename, "w") do io
    println(io, "case118Blumsack.m\t$S_input\t$K_test\t$(res.iterations)\t$total_work_units\t$(res.total_solutions_added)\t$(res.status)\t$(round(res.total_master_work_units, digits=2))\t$(round(res.total_pricing_work_units, digits=2))\t$(round(res.total_update_work_units, digits=2))\t$(round(res.final_objective, digits=2))")
end

println()
println("="^80)
println("TS COLUMN GENERATION SUMMARY")
println("Instance: case118Blumsack.m")
println("K: $K_test")
println("S: $S_input")
println("Variance: $variance_input")
println("Iterations: ", res.iterations)
println("Status: ", res.status)
println("Total Work Units: ", total_work_units)
println("  - Master Problem: ", res.total_master_work_units)
println("  - Pricing Subproblem: ", res.total_pricing_work_units)
println("  - Master Update: ", res.total_update_work_units)
println("Final Objective: ", res.final_objective)
println("Output file: $output_filename")
println("="^80)

try
    redirect_stdout(_orig_stdout)
    redirect_stderr(_orig_stderr)
catch
end


