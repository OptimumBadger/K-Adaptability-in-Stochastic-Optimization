# TransmissionSwitchingCG.jl
# Problem-specific implementations for Transmission Switching Column Generation
# This file provides the interface functions required by ColumnGenerationCore.jl

using JuMP, Gurobi
using MathOptInterface
const MOI = MathOptInterface
using Printf
using Statistics
using Random, Distributions
using PowerModels
using XLSX, DataFrames

# Include the core column generation algorithm
include("ColumnGenerationCore.jl")

# =============================================================================
# CONSTANTS
# =============================================================================

const DEFAULT_OPT_WORKLIMIT = 20.0
const DEFAULT_ALPHA = 5000.0

# =============================================================================
# DATA LOADING
# =============================================================================

function parameters_TSpm(data; line_capacity_factor::Float64=1.0)
    """Extract parameters from PowerModels data structure"""
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
        rate_a = haskey(branch, "rate_a") ? branch["rate_a"] : 2.2
        fbar_ij[parse(Int, branch_id)] = rate_a * line_capacity_factor
    end

    for (gen_id, generator) in G
        c[generator["gen_bus"]] = generator["cost"][1]
        p_max[generator["gen_bus"]] = generator["pmax"]
        p_min[generator["gen_bus"]] = generator["pmin"]
    end

    n = 2*length(B)
    m = length(L)
    A = zeros(n, m)

    # Create incidence matrix A
    for (branch_id, branch) in L
        f_bus = branch["f_bus"]
        t_bus = branch["t_bus"]
        branch_index = parse(Int, branch_id)
        A[2*f_bus-1, branch_index] = 1
        A[2*f_bus, branch_index] = -1
        A[2*t_bus-1, branch_index] = -1
        A[2*t_bus, branch_index] = 1
    end

    return B_ij, fbar_ij, p_max, p_min, A, c
end

function load_TS_data(instance_file::String; line_capacity_factor::Float64=1.0)
    """Load Transmission Switching instance data from MATPOWER file
    
    Returns:
        Dict with PowerModels data structure
    """
    if !isfile(instance_file)
        error("❌ Error: Instance file not found! Expected: $instance_file")
    end
    
    data = PowerModels.parse_file(instance_file)
    
    # Store capacity factor for later use
    data["line_capacity_factor"] = line_capacity_factor
    
    return data
end

# =============================================================================
# COST EVALUATION r_s(x)
# =============================================================================

function evaluate_cost_TS(binary_vector::Vector{Int}, scenario::Vector{Float64}, problem_data::Dict)
    """Evaluate cost r_s(x) for Transmission Switching
    
    Args:
        binary_vector: Binary solution vector (0/1 for each line)
        scenario: Demand scenario vector
        problem_data: PowerModels data structure
    
    Returns:
        (cost::Float64, work_units::Float64)
    """
    B = problem_data["bus"]
    L = problem_data["branch"]
    nb = length(B)
    nl = length(L)
    B_ij, fbar_ij, p_max, p_min, A, c_g = parameters_TSpm(problem_data)
    ref = 1
    penalty = DEFAULT_ALPHA

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

    # Line capacity + DC flow with fixed x
    for (branch_id, branch) in L
        l = parse(Int, branch_id)
        xl = binary_vector[l]
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
        @constraint(model, p_g[i] - scenario[i] + unmet[i] == outgoing - incoming)
    end

    # Reference angle
    @constraint(model, theta[ref] == 0)

    # Objective
    @objective(model, Min, sum(c_g[i] * p_g[i] for i in 1:nb) + penalty * sum(unmet[i] for i in 1:nb))

    optimize!(model)

    work_units = 0.0
    try
        work_units = MOI.get(backend(model), Gurobi.ModelAttribute("Work"))
    catch
    end

    if termination_status(model) == MOI.OPTIMAL
        return objective_value(model), work_units
    else
        return Inf, work_units
    end
end

# =============================================================================
# M VALUES CALCULATION (for Pricing Type 1)
# =============================================================================

function calculate_m_values_TS(scenarios::Vector{Vector{Float64}}, dual_lambda::Vector{Float64}, problem_data::Dict)
    """Calculate m^s values for all scenarios (for Pricing Type 1)"""
    B_ij, fbar_ij, p_max, p_min, A, c = parameters_TSpm(problem_data)
    nb = length(problem_data["bus"])
    nscen = length(scenarios)
    alpha = DEFAULT_ALPHA
    
    m_values = zeros(nscen)
    for s in 1:nscen
        m_values[s] = big_m_calculator_TS(dual_lambda[s], c, p_min, p_max, scenarios[s], alpha)
    end
    
    return m_values
end

function big_m_calculator_TS(dual_lambda, c, p_min, p_max, demand, alpha)
    """Calculate m^s for a single scenario"""
    nb = length(demand)
    
    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 0)
    set_optimizer_attribute(model, "MIPGap", 0.01)
    set_optimizer_attribute(model, "WorkLimit", 1800)
    
    @variable(model, p_g[1:nb])
    @variable(model, r[1:nb] >= 0)
    
    for i in 1:nb
        @constraint(model, p_g[i] >= p_min[i])
        @constraint(model, p_g[i] <= p_max[i])
    end
    
    @constraint(model, [i in 1:nb], p_g[i] + r[i] >= demand[i])
    
    @objective(model, Min, dual_lambda - sum(c[i] * p_g[i] for i in 1:nb) - alpha * sum(r[i] for i in 1:nb))
    
    optimize!(model)
    
    if termination_status(model) == MOI.OPTIMAL
        return objective_value(model)
    else
        return Inf
    end
end

# =============================================================================
# PRICING SUBPROBLEM (Type 1)
# =============================================================================

function af_pricing_type1_TS(data, scenarios, m_values, dual_lambda, dual_mu; use_solution_pool::Bool=true, work_limit::Float64=Inf, opt_worklimit::Float64=DEFAULT_OPT_WORKLIMIT, alpha::Float64=DEFAULT_ALPHA, return_nodes::Bool=false)
    """Solve AF Pricing Type 1 for Transmission Switching"""
    B_ij, fbar_ij, p_max, p_min, A, c = parameters_TSpm(data)
    L = data["branch"]
    nb = length(data["bus"])
    nl = length(L)
    nscen = length(scenarios)

    # Big-M for DC equality relaxation when x_l = 0
    M_ij = [12.6 * abs(B_ij[l]) for l in 1:nl]

    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 1)
    if isfinite(opt_worklimit)
        set_optimizer_attribute(model, "WorkLimit", opt_worklimit)
    end
    if use_solution_pool
        set_optimizer_attribute(model, "PoolSearchMode", 2)
        set_optimizer_attribute(model, "PoolSolutions", 10)
    end

    # Decision variables
    @variable(model, x[1:nl], Bin)
    @variable(model, pi[1:nscen], Bin)
    @variable(model, p_g[1:nb, 1:nscen])
    @variable(model, r[1:nb, 1:nscen] >= 0)
    @variable(model, f[1:nl, 1:nscen])
    @variable(model, theta[1:nb, 1:nscen])
    @variable(model, pbar[1:nb, 1:nscen])
    @variable(model, rbar[1:nb, 1:nscen] >= 0)

    # Objective
    @objective(model, Max, sum(dual_lambda[s] * pi[s] for s in 1:nscen)
                        - sum(c[i] * pbar[i, s] for i in 1:nb for s in 1:nscen)
                        - alpha * sum(rbar[i, s] for i in 1:nb for s in 1:nscen))

    # Generator bounds per scenario
    for i in 1:nb, s in 1:nscen
        @constraint(model, p_g[i, s] >= p_min[i]*pi[s])
        @constraint(model, p_g[i, s] <= p_max[i]*pi[s])
    end

    # Line capacity and DC constraints
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

    # Pricing key inequality per scenario
    for s in 1:nscen
        @constraint(model, dual_lambda[s]
            - sum(c[i] * p_g[i, s] for i in 1:nb)
            - alpha * sum(r[i, s] for i in 1:nb)
            >= m_values[s] * (1 - pi[s]))
    end

    # Linearization for pbar and rbar with pi
    for i in 1:nb, s in 1:nscen
        @constraint(model, pbar[i, s] <= p_g[i, s])
        @constraint(model, pbar[i, s] <= p_max[i] * pi[s])
        @constraint(model, pbar[i, s] >= p_g[i, s] - p_max[i] * (1 - pi[s]))
        d_is = scenarios[s][i]
        @constraint(model, rbar[i, s] <= r[i, s])
        @constraint(model, rbar[i, s] <= d_is * pi[s])
        @constraint(model, rbar[i, s] >= r[i, s] - d_is * (1 - pi[s]))
    end

    # Solve Phase 1
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
                x_k = [MOI.get(opt, Gurobi.VariableAttribute("Xn"), JuMP.index(x[i])) for i in 1:nl]
                rc_k = pool_obj + dual_mu
                if rc_k > 1e-6
                    push!(kept_solutions, x_k)
                    push!(kept_objs, pool_obj)
                end
            end
        catch e
            println("  Warning: solution pool access failed: $e")
        end
    end

    if isempty(kept_solutions)
        if return_nodes
            nodes = 0
            try
                nodes = MOI.get(backend(model), Gurobi.ModelAttribute("NodeCount"))
            catch
            end
            return [value.(x)], [objective_value(model)], pricing_work_units, nodes
        else
            return [value.(x)], [objective_value(model)], pricing_work_units
        end
    end

    return return_pricing_payload(kept_solutions, kept_objs, pricing_work_units, return_nodes, model)
end

# =============================================================================
# PRICING SUBPROBLEM (Type 2)
# =============================================================================

function af_pricing_type2_TS(data, scenarios, dual_lambda, dual_mu; use_solution_pool::Bool=true, work_limit::Float64=Inf, opt_worklimit::Float64=DEFAULT_OPT_WORKLIMIT, alpha::Float64=DEFAULT_ALPHA, return_nodes::Bool=false, analysis::Dict{Symbol,Any}=Dict{Symbol,Any}())
    """Solve AF Pricing Type 2 for Transmission Switching using two-phase approach"""
    B_ij, fbar_ij, p_max, p_min, A, c = parameters_TSpm(data)
    L = data["branch"]
    nb = length(data["bus"])
    nl = length(L)
    nscen = length(scenarios)

    # Big-M following instruction: M_ij = -12.6 * B_ij
    M_ij = [-12.6 * B_ij[l] for l in 1:nl]

    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 1)
    if use_solution_pool
        set_optimizer_attribute(model, "PoolSearchMode", 2)
        set_optimizer_attribute(model, "PoolSolutions", 10)
    end

    # Decision variables (Type 2: x and per-scenario x_s linked by pi)
    @variable(model, x[1:nl], Bin)
    @variable(model, x_s[1:nl, 1:nscen], Bin)
    @variable(model, pi[1:nscen], Bin)
    @variable(model, p_g[1:nb, 1:nscen])
    @variable(model, r[1:nb, 1:nscen] >= 0)
    @variable(model, f[1:nl, 1:nscen])
    @variable(model, theta[1:nb, 1:nscen])

    # Objective
    @objective(model, Max, sum(dual_lambda[s] * pi[s] for s in 1:nscen)
                        - sum(c[i] * p_g[i,s] for i in 1:nb for s in 1:nscen)
                        - alpha * sum(r[i,s] for i in 1:nb for s in 1:nscen))

    # Linking constraints
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

    # Key inequality
    @constraint(model, [s in 1:nscen],
        dual_lambda[s] * pi[s]
        - sum(c[i] * p_g[i, s] for i in 1:nb)
        - alpha * sum(r[i, s] for i in 1:nb) >= 0)

    # Two-phase solve
    pricing_work_units = 0.0
    if !isempty(analysis)
        analysis[:phase1_work] = 0.0
        analysis[:phase2_work] = 0.0
        analysis[:went_phase2] = false
        analysis[:lb] = NaN
        analysis[:ub] = NaN
    end

    # Phase 1
    if isfinite(opt_worklimit) && opt_worklimit > 0.0
        println("  Phase 1: Quick optimal try with $(opt_worklimit) work units")
        set_optimizer_attribute(model, "MIPFocus", 1)
        set_optimizer_attribute(model, "WorkLimit", opt_worklimit)
        optimize!(model)
        
        try
            w = MOI.get(backend(model), Gurobi.ModelAttribute("Work"))
            pricing_work_units += w
            if !isempty(analysis)
                analysis[:phase1_work] = w
            end
        catch
        end
        
        if termination_status(model) == MOI.OPTIMAL
            println("  Phase 1: Optimal solution found")
            kept_solutions, kept_objs = collect_pool_rc_cap10(model, x, dual_mu, use_solution_pool)
            if !isempty(kept_solutions)
                println("  Phase 1: Found $(length(kept_solutions)) RC-improving solutions")
                return return_pricing_payload(kept_solutions, kept_objs, pricing_work_units, return_nodes, model)
            else
                println("  Phase 1: Optimal but no RC-improving solutions — terminating pricing")
                return Vector{Vector{Float64}}(), Float64[], pricing_work_units
            end
        else
            println("  Phase 1: Not optimal - checking pool for RC-improving incumbents")
            kept_solutions, kept_objs = collect_pool_rc_cap10(model, x, dual_mu, use_solution_pool)
            if !isempty(kept_solutions)
                println("  Phase 1: Found $(length(kept_solutions)) RC-improving solutions (non-optimal status)")
                return return_pricing_payload(kept_solutions, kept_objs, pricing_work_units, return_nodes, model)
            else
                println("  Phase 1: No RC-improving solutions found, proceeding to Phase 2")
            end
        end
    end

    # Phase 2
    heur_worklimit = work_limit - opt_worklimit
    if isfinite(heur_worklimit) && heur_worklimit > 0.0
        println("  Phase 2: Optimality push with $(heur_worklimit) work units")
        if !isempty(analysis)
            analysis[:went_phase2] = true
        end
        set_optimizer_attribute(model, "MIPFocus", 0)
        set_optimizer_attribute(model, "WorkLimit", heur_worklimit)
        optimize!(model)
        
        try
            w2 = MOI.get(backend(model), Gurobi.ModelAttribute("Work"))
            pricing_work_units += w2
            if !isempty(analysis)
                analysis[:phase2_work] = w2
            end
        catch
        end
        
        if termination_status(model) == MOI.OPTIMAL
            println("  Phase 2: Optimal solution found")
            try
                lb = MOI.get(backend(model), Gurobi.ModelAttribute("ObjBound"))
                ub = MOI.get(backend(model), Gurobi.ModelAttribute("ObjVal"))
                if !isempty(analysis)
                    analysis[:lb] = lb
                    analysis[:ub] = ub
                end
            catch
            end
            kept_solutions, kept_objs = collect_pool_rc_cap10(model, x, dual_mu, use_solution_pool)
            return return_pricing_payload(kept_solutions, kept_objs, pricing_work_units, return_nodes, model)
        else
            println("  Phase 2: Work units exhausted. Reporting bounds...")
            try
                lb = MOI.get(backend(model), Gurobi.ModelAttribute("ObjBound"))
                ub = MOI.get(backend(model), Gurobi.ModelAttribute("ObjVal"))
                println("  Lower bound: $lb, Upper bound: $ub")
                if !isempty(analysis)
                    analysis[:lb] = lb
                    analysis[:ub] = ub
                end
            catch
            end
            return return_pricing_payload(Vector{Vector{Float64}}(), Float64[], pricing_work_units, return_nodes, model)
        end
    end
    
    # Fallback
    return return_pricing_payload(Vector{Vector{Float64}}(), Float64[], pricing_work_units, return_nodes, model)
end

# =============================================================================
# PRICING SUBPROBLEM WRAPPER (Interface for ColumnGenerationCore)
# =============================================================================

function build_pricing_model_TS(scenarios::Vector{Vector{Float64}}, dual_lambda::Vector{Float64}, dual_mu::Float64, problem_data::Dict, pricing_type::Int, work_limit::Float64, opt_worklimit::Float64, use_solution_pool::Bool, return_nodes::Bool, log_file::Union{String, Nothing})
    """Build and solve pricing subproblem for Transmission Switching
    
    This is the interface function called by ColumnGenerationCore.jl
    """
    nscen = length(scenarios)
    alpha = DEFAULT_ALPHA
    
    if pricing_type == 1
        # Calculate m^s values
        m_values = calculate_m_values_TS(scenarios, dual_lambda, problem_data)
        
        # Solve Type 1
        return af_pricing_type1_TS(problem_data, scenarios, m_values, dual_lambda, dual_mu; 
                                   use_solution_pool=use_solution_pool, return_nodes=return_nodes, 
                                   work_limit=work_limit, opt_worklimit=opt_worklimit, alpha=alpha)
    elseif pricing_type == 2
        # Solve Type 2
        m_values = zeros(nscen)  # Not used but kept for compatibility
        analysis = Dict{Symbol,Any}()
        return af_pricing_type2_TS(problem_data, scenarios, dual_lambda, dual_mu; 
                                   use_solution_pool=use_solution_pool, return_nodes=return_nodes, 
                                   work_limit=work_limit, opt_worklimit=opt_worklimit, alpha=alpha, analysis=analysis)
    else
        error("Invalid pricing_type. Use 1 or 2.")
    end
end

