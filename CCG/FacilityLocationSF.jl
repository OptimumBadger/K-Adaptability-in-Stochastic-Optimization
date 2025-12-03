# FacilityLocationSF.jl
# Problem-specific implementations for Facility Location - Subset Formulation (SF)
# This file provides the interface functions required by ColumnGenerationCore.jl for SF

using JuMP, Gurobi
using MathOptInterface
const MOI = MathOptInterface
using Printf
using Statistics
using Random, Distributions
using Dates

# Include shared utilities
include("ColumnGenerationCore.jl")
include("SubsetFormulation.jl")

# =============================================================================
# CONSTANTS
# =============================================================================

const DEFAULT_OPT_WORKLIMIT = 300.0

# =============================================================================
# DATA LOADING (shared with AF)
# =============================================================================

# Import read_orlib_cap from Samples/FL/FLUtilities.jl (data generation utilities)
# This keeps CCG dependent on Samples, which is correct since we generate data first
include("../Samples/FL/FLUtilities.jl")

# =============================================================================
# SUBSET COST EVALUATION f(A)
# =============================================================================

function evaluate_subset_cost_FL(subset_vector::Vector{Int}, scenarios::Vector{Vector{Float64}}, problem_data::Dict)
    """Evaluate cost f(A) for a subset of scenarios (Facility Location)
    
    Args:
        subset_vector: Binary vector of length S, where subset_vector[s] = 1 if scenario s is in subset A
        scenarios: List of S scenarios
        problem_data: Dict containing capacity, fixedcost, cost, unmet_pen, scaling_factor, nloc, ncust
    
    Returns:
        (cost::Float64, work_units::Float64)
    """
    nloc = problem_data["nloc"]
    ncust = problem_data["ncust"]
    capacity = problem_data["capacity"]
    cost = problem_data["cost"]
    fixedcost = problem_data["fixedcost"]
    unmet_pen = problem_data["unmet_pen"]
    scaling_factor = problem_data["scaling_factor"]
    
    nscen = length(scenarios)
    
    # Check if subset is empty
    if sum(subset_vector) == 0
        return 0.0, 0.0  # Cost of empty subset is 0
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
        if subset_vector[s] == 1  # Only add constraint for scenarios in subset A
            for j in 1:ncust
                @constraint(model, sum(y[i,j,s] for i in 1:nloc) + b[j,s] == scenarios[s][j])
            end
        end
    end
    
    # 2. Capacity constraint: sum_j y[i,j,s] - capacity[i]*x[i] <= 0 for i∈I, s∈A
    for s in 1:nscen
        if subset_vector[s] == 1  # Only add constraint for scenarios in subset A
            for i in 1:nloc
                @constraint(model, sum(y[i,j,s] for j in 1:ncust) - capacity[i]*x[i] <= 0)
            end
        end
    end
    
    # 3. Set y[i,j,s] = 0 and b[j,s] = 0 for scenarios NOT in subset A
    for s in 1:nscen
        if subset_vector[s] == 0  # For scenarios not in subset A
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
    subset_size = sum(subset_vector)
    weight_fixed = 1.0
    weight_rest = 1.0 
    @objective(model, Min,
        subset_size * sum(fixedcost[i] * x[i] for i in 1:nloc) +scaling_factor * sum(cost[i,j] * y[i,j,s] for s in 1:nscen for i in 1:nloc for j in 1:ncust if subset_vector[s] == 1) +
            sum(unmet_pen * b[j,s] for s in 1:nscen for j in 1:ncust if subset_vector[s] == 1)
        )
    
    
    # Solve
    optimize!(model)
    
    # Get work units
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
# PRICING SUBPROBLEM (Type 1) - SF
# =============================================================================

function sf_pricing_type1_FL(nloc, ncust, capacity, cost, scenarios, dual_lambda, dual_mu, unmet_pen, scaling_factor, fixedcost; use_solution_pool::Bool=true, return_nodes::Bool=false, work_limit::Float64=Inf, opt_worklimit::Float64=DEFAULT_OPT_WORKLIMIT, log_file=nothing, iteration=nothing)
    """Solve SF Pricing Type 1 - Generate new subset of scenarios (solve to optimality with work limit)
    
    Returns:
        (solutions, objectives, work_units) or (solutions, objectives, work_units, nodes) if return_nodes=true
    """
    nscen = length(scenarios)
    
    # Create model
    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 1)
    if use_solution_pool
        set_optimizer_attribute(model, "PoolSearchMode", 2)
        set_optimizer_attribute(model, "PoolSolutions", 10)
    end
    
    # Set up log file with iteration marker
    if log_file !== nothing
        if iteration !== nothing
            open(log_file, "a") do file
                println(file, "\n" * "="^80)
                println(file, "PRICING SUBPROBLEM TYPE 1 (SF) - ITERATION $iteration")
                println(file, "="^80)
                println(file, "Timestamp: $(now())")
                println(file, "Problem size: $(length(scenarios)) scenarios, $nloc facilities, $ncust customers")
                println(file, "="^80)
            end
        end
        set_optimizer_attribute(model, "OutputFlag", 1)  # Ensure output is enabled for log file
        set_optimizer_attribute(model, "LogFile", log_file)
        set_optimizer_attribute(model, "LogToConsole", 0)
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
    
    # Constraints (same as in Subset.jl)
    @constraint(model, [j in 1:ncust, s in 1:nscen], 
        sum(y[i,j,s] for i in 1:nloc) + b[j,s] - scenarios[s][j] <= scenarios[s][j] * (1 - pi[s]))
    @constraint(model, [i in 1:nloc, s in 1:nscen], v[i,s] <= x[i])
    @constraint(model, [i in 1:nloc, s in 1:nscen], v[i,s] <= pi[s])
    @constraint(model, [i in 1:nloc, s in 1:nscen], v[i,s] >= x[i] + pi[s] - 1)
    @constraint(model, [j in 1:ncust, s in 1:nscen], 
        sum(y[i,j,s] for i in 1:nloc) + b[j,s] - scenarios[s][j] >= -scenarios[s][j] * (1 - pi[s]))
    @constraint(model, [i in 1:nloc, s in 1:nscen], 
        sum(y[i,j,s] for j in 1:ncust) - capacity[i]*x[i] <= capacity[i] * (1 - pi[s]))
    @constraint(model, [i in 1:nloc, j in 1:ncust, s in 1:nscen], u[i,j,s] <= y[i,j,s])
    @constraint(model, [i in 1:nloc, j in 1:ncust, s in 1:nscen], u[i,j,s] <= scenarios[s][j] * pi[s])
    @constraint(model, [i in 1:nloc, j in 1:ncust, s in 1:nscen], 
        u[i,j,s] >= y[i,j,s] + scenarios[s][j] * (pi[s] - 1))
    @constraint(model, [j in 1:ncust, s in 1:nscen], w[j,s] <= b[j,s])
    @constraint(model, [j in 1:ncust, s in 1:nscen], w[j,s] <= scenarios[s][j] * pi[s])
    @constraint(model, [j in 1:ncust, s in 1:nscen], 
        w[j,s] >= b[j,s] + scenarios[s][j] * (pi[s] - 1))
    
    # Solve to optimality with given work limit
    pricing_work_units = 0.0
    
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
        kept_solutions, kept_objs = collect_pool_rc_cap10(model, pi, dual_mu, use_solution_pool, "SF")
        return return_pricing_payload(kept_solutions, kept_objs, pricing_work_units, return_nodes, model)
    else
        kept_solutions, kept_objs = collect_pool_rc_cap10(model, pi, dual_mu, use_solution_pool, "SF")
        return return_pricing_payload(kept_solutions, kept_objs, pricing_work_units, return_nodes, model)
    end
end

# =============================================================================
# PRICING SUBPROBLEM (Type 2) - SF
# =============================================================================

function sf_pricing_type2_FL(nloc, ncust, capacity, cost, scenarios, dual_lambda, dual_mu, unmet_pen, scaling_factor, fixedcost; use_solution_pool::Bool=true, return_nodes::Bool=false, work_limit::Float64=Inf, opt_worklimit::Float64=DEFAULT_OPT_WORKLIMIT, analysis::Dict{Symbol,Any}=Dict{Symbol,Any}(), log_file=nothing, iteration=nothing)
    """Solve SF Pricing Type 2 - Alternative formulation (two-phase with solution pool)
    
    Returns:
        (solutions, objectives, work_units) or (solutions, objectives, work_units, nodes) if return_nodes=true
    """
    nscen = length(scenarios)
    
    # Create model
    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 1)
    if use_solution_pool
        set_optimizer_attribute(model, "PoolSearchMode", 2)
        set_optimizer_attribute(model, "PoolSolutions", 10)
    end
    
    # Set up log file with iteration marker
    if log_file !== nothing
        if iteration !== nothing
            open(log_file, "a") do file
                println(file, "\n" * "="^80)
                println(file, "PRICING SUBPROBLEM TYPE 2 (SF) - ITERATION $iteration")
                println(file, "="^80)
                println(file, "Timestamp: $(now())")
                println(file, "Problem size: $(length(scenarios)) scenarios, $nloc facilities, $ncust customers")
                println(file, "="^80)
            end
        end
        set_optimizer_attribute(model, "OutputFlag", 1)  # Ensure output is enabled for log file
        set_optimizer_attribute(model, "LogFile", log_file)
        set_optimizer_attribute(model, "LogToConsole", 1)
    end
    
    # Decision Variables
    @variable(model, x[1:nloc], Bin)  # Binary: facility open/closed (first-stage)
    @variable(model, x_s[1:nloc, 1:nscen], Bin)  # Binary copies: facility open/closed for scenario s
    @variable(model, y[1:nloc, 1:ncust, 1:nscen] >= 0)  # Flow: y[i,j,s]
    @variable(model, b[1:ncust, 1:nscen] >= 0)  # Unmet demand: b[j,s]
    @variable(model, pi[1:nscen], Bin)  # Binary: scenario selection π^s
    
    # Objective Function: min ∑_s ( (∑_i f_i x_i^s) + (∑_i ∑_j c_ij y_ij^s) + (∑_j α b_j^s) - (λ^s)*π^s ) - μ*
    @objective(model, Min, 
          sum(fixedcost[i]*x_s[i,s] for i in 1:nloc for s in 1:nscen) + 
            scaling_factor * sum(cost[i,j]*y[i,j,s] for i in 1:nloc for j in 1:ncust for s in 1:nscen) + 
            sum(unmet_pen*b[j,s] for j in 1:ncust for s in 1:nscen)- 
            sum(dual_lambda[s]*pi[s] for s in 1:nscen) - 
        dual_mu)
    
    # Constraints
    @constraint(model, [j in 1:ncust, s in 1:nscen], 
        sum(y[i,j,s] for i in 1:nloc) + b[j,s] == scenarios[s][j]*pi[s])
    @constraint(model, [i in 1:nloc, s in 1:nscen], 
        sum(y[i,j,s] for j in 1:ncust) - capacity[i]*x_s[i,s] <= 0)
    @constraint(model, [i in 1:nloc, s in 1:nscen], x_s[i,s] <= pi[s])
    @constraint(model, [i in 1:nloc, s in 1:nscen], x[i] - x_s[i,s] <= (1 - pi[s]))
    @constraint(model, [i in 1:nloc, s in 1:nscen], x[i] - x_s[i,s] >= -(1 - pi[s]))
    
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
        set_optimizer_attribute(model, "MIPFocus", 1)
        set_optimizer_attribute(model, "WorkLimit", opt_worklimit)
        optimize!(model)
        
        try
            w = MOI.get(backend(model), Gurobi.ModelAttribute("Work"))
            pricing_work_units += w
            if !isempty(analysis); analysis[:phase1_work] = w; end
        catch
        end
        
        # Check for RC-improving solutions
        if termination_status(model) == MOI.OPTIMAL
            kept_solutions, kept_objs = collect_pool_rc_cap10(model, pi, dual_mu, use_solution_pool, "SF")
            if !isempty(kept_solutions)
                if !isempty(analysis); analysis[:went_phase2] = false; end
                return return_pricing_payload(kept_solutions, kept_objs, pricing_work_units, return_nodes, model)
            else
                # No RC-improving solutions, proceed to Phase 2
            end
        else
            kept_solutions, kept_objs = collect_pool_rc_cap10(model, pi, dual_mu, use_solution_pool, "SF")
            if !isempty(kept_solutions)
                if !isempty(analysis); analysis[:went_phase2] = false; end
                return return_pricing_payload(kept_solutions, kept_objs, pricing_work_units, return_nodes, model)
            end
        end
    end

    # Phase 2: Optimality push
    heur_worklimit = work_limit - opt_worklimit
    if isfinite(heur_worklimit) && heur_worklimit > 0.0
        if !isempty(analysis); analysis[:went_phase2] = true; end
        set_optimizer_attribute(model, "MIPFocus", 0)
        set_optimizer_attribute(model, "WorkLimit", heur_worklimit)
        optimize!(model)
        
        try
            w2 = MOI.get(backend(model), Gurobi.ModelAttribute("Work"))
            pricing_work_units += w2
            if !isempty(analysis); analysis[:phase2_work] = w2; end
        catch
        end
        
        kept_solutions, kept_objs = collect_pool_rc_cap10(model, pi, dual_mu, use_solution_pool, "SF")
        return return_pricing_payload(kept_solutions, kept_objs, pricing_work_units, return_nodes, model)
    end
    
    # Fallback if no work limits set
    return return_pricing_payload(Vector{Vector{Float64}}(), Float64[], pricing_work_units, return_nodes, model)
end

# =============================================================================
# PRICING SUBPROBLEM WRAPPER (Interface for ColumnGenerationCore)
# =============================================================================

function build_pricing_model_SF_FL(scenarios::Vector{Vector{Float64}}, dual_lambda::Vector{Float64}, dual_mu::Float64, problem_data::Dict, pricing_type::Int, work_limit::Float64, opt_worklimit::Float64, use_solution_pool::Bool, return_nodes::Bool, log_file::Union{String, Nothing}, iteration::Union{Int, Nothing}=nothing)
    """Build and solve pricing subproblem for Facility Location - Subset Formulation
    
    This is the interface function called by ColumnGenerationCore.jl
    
    Returns:
        (solutions, objectives, work_units) or (solutions, objectives, work_units, nodes) if return_nodes=true
    """
    nloc = problem_data["nloc"]
    ncust = problem_data["ncust"]
    capacity = problem_data["capacity"]
    cost = problem_data["cost"]
    fixedcost = problem_data["fixedcost"]
    unmet_pen = problem_data["unmet_pen"]
    scaling_factor = problem_data["scaling_factor"]
    
    if pricing_type == 1
        return sf_pricing_type1_FL(nloc, ncust, capacity, cost, scenarios, dual_lambda, dual_mu, unmet_pen, scaling_factor, fixedcost; 
                                   use_solution_pool=use_solution_pool, return_nodes=return_nodes, work_limit=work_limit, opt_worklimit=opt_worklimit, log_file=log_file, iteration=iteration)
    elseif pricing_type == 2
        analysis = Dict{Symbol,Any}()
        return sf_pricing_type2_FL(nloc, ncust, capacity, cost, scenarios, dual_lambda, dual_mu, unmet_pen, scaling_factor, fixedcost; 
                                   use_solution_pool=use_solution_pool, return_nodes=return_nodes, work_limit=work_limit, opt_worklimit=opt_worklimit, analysis=analysis, log_file=log_file, iteration=iteration)
    else
        error("Invalid pricing_type. Use 1 or 2.")
    end
end

