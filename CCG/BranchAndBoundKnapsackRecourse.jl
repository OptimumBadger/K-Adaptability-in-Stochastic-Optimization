# BranchAndBoundKnapsackRecourse.jl
# Knapsack (KS, with recourse) — problem-specific functions for the B&B pricing
# framework.
#
# Capacity constraint is soft: violations penalised at PENALTY_R per unit excess.
#
# Subproblem at node (F0, F1) for scenario s:
#   g_s(F0,F1) = min_{x binary, v >= 0}  -profit_s^T x + PENALTY_R * v
#                s.t.  weight_s^T x - v <= capacity_s
#                      x_j = 0  for j in F0,  x_j = 1  for j in F1
#
# This is always feasible (v absorbs any capacity violation).
#
# Scenario encoding (from KnapsackRecourseAF.jl):
#   sc = [profit_1..profit_n, weight_1..weight_n, capacity]
#   decode_scenario_KS_r(sc, n)  ->  (profits, weights, capacity)
#
# Requires BranchAndBoundCore.jl to be included first.
# `_resolve_penalty_r` is defined in KnapsackRecourseAF.jl.

# =============================================================================
# PERSISTENT SCENARIO MODEL — KS with recourse
# =============================================================================

struct PersistentKSRecourseModel
    model::Model
    x::Vector{VariableRef}
    v::VariableRef                   # excess capacity slack
    work_supported::Bool
    grb::Gurobi.Optimizer
    vi::Vector{MOI.VariableIndex}
    x_buf::Vector{Float64}
end

function build_persistent_models_KS_r(
    problem_data::Dict,
    scenarios::Vector{Vector{Float64}}
)
    n       = problem_data["n"]
    penalty = _resolve_penalty_r(problem_data)
    nscen   = length(scenarios)
    pmodels = Vector{PersistentKSRecourseModel}(undef, nscen)

    # Per-scenario Gurobi.Env so concurrent solves from Threads.@threads are safe.
    # Gurobi.jl's README: Envs are NOT thread-safe to share. Models sharing an
    # env cannot be solved simultaneously on different threads. So we give each
    # scenario its own env. (Academic license — env tokens are unrestricted.)
    for s in 1:nscen
        profits, weights, cap = decode_scenario_KS_r(scenarios[s], n)

        env   = Gurobi.Env()
        model = direct_model(Gurobi.Optimizer(env))
        set_optimizer_attribute(model, "OutputFlag",   0)
        set_optimizer_attribute(model, "LogToConsole", 0)
        # Single-thread each Gurobi solve; parallelism is at the scenario level.
        set_optimizer_attribute(model, "Threads",      1)

        @variable(model, x[1:n], Bin)
        @variable(model, v >= 0)

        # soft capacity: weight^T x - v <= cap  =>  v >= weight^T x - cap
        @constraint(model, sum(weights[j] * x[j] for j in 1:n) - v <= cap)

        @objective(model, Min, -sum(profits[j] * x[j] for j in 1:n) + penalty * v)

        grb = backend(model)
        work_supported = try; MOI.get(grb, _GRB_WORK); true; catch; false; end
        vi = MOI.VariableIndex[JuMP.index(xj) for xj in x]
        pmodels[s] = PersistentKSRecourseModel(model, x, v, work_supported, grb, vi, zeros(n))
    end

    return pmodels
end

function solve_subproblem_persistent_KS_r(
    pm::PersistentKSRecourseModel,
    F0::UInt64,
    F1::UInt64
)
    grb = pm.grb
    vi  = pm.vi
    n   = length(vi)

    _apply_mask_bounds!(grb, _GRB_UB, vi, F0, 0.0)
    _apply_mask_bounds!(grb, _GRB_LB, vi, F1, 1.0)

    MOI.optimize!(grb)

    work_units = pm.work_supported ? MOI.get(grb, _GRB_WORK) : 0.0
    runtime    = MOI.get(grb, _GRB_RUNTIME)
    status     = MOI.get(grb, _GRB_STATUS)

    # Recourse model is always feasible
    if status == _GRB_OPTIMAL
        obj = MOI.get(grb, _GRB_OBJVAL)
        _read_x_into_buf!(grb, vi, pm.x_buf)
        result = (obj, pm.x_buf, work_units, runtime)
    else
        @inbounds for j in 1:n; pm.x_buf[j] = 0.0; end
        result = (Inf, pm.x_buf, work_units, runtime)
    end

    _apply_mask_bounds!(grb, _GRB_UB, vi, F0, 1.0)
    _apply_mask_bounds!(grb, _GRB_LB, vi, F1, 0.0)

    return result
end

function solve_subproblem_scenario_KS_r(
    problem_data::Dict,
    scenario::Vector{Float64},
    F0::UInt64,
    F1::UInt64
)
    n       = problem_data["n"]
    penalty = _resolve_penalty_r(problem_data)
    profits, weights, cap = decode_scenario_KS_r(scenario, n)

    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag",   0)
    set_optimizer_attribute(model, "LogToConsole", 0)

    @variable(model, x[1:n], Bin)
    @variable(model, v >= 0)

    _m = F0
    while _m != UInt64(0)
        j = trailing_zeros(_m) + 1
        fix(x[j], 0; force=true)
        _m &= _m - UInt64(1)
    end
    _m = F1
    while _m != UInt64(0)
        j = trailing_zeros(_m) + 1
        fix(x[j], 1; force=true)
        _m &= _m - UInt64(1)
    end

    @constraint(model, sum(weights[j] * x[j] for j in 1:n) - v <= cap)
    @objective(model, Min, -sum(profits[j] * x[j] for j in 1:n) + penalty * v)

    optimize!(model)

    work_units = 0.0
    try; work_units = MOI.get(backend(model), Gurobi.ModelAttribute("Work")); catch; end
    runtime = solve_time(model)

    if termination_status(model) == MOI.OPTIMAL
        return objective_value(model), value.(x), work_units, runtime
    else
        return Inf, zeros(n), work_units, runtime
    end
end

# =============================================================================
# PERSISTENT SCENARIO MODEL — MK with recourse
# =============================================================================

struct PersistentMKRecourseModel
    model::Model
    x::Vector{VariableRef}
    v::Vector{VariableRef}           # one excess slack per dimension
    work_supported::Bool
    grb::Gurobi.Optimizer
    vi::Vector{MOI.VariableIndex}
    x_buf::Vector{Float64}
end

function build_persistent_models_MK_r(
    problem_data::Dict,
    scenarios::Vector{Vector{Float64}}
)
    n       = problem_data["n"]
    dims    = get(problem_data, "dims", MK_DIMS)
    penalty = _resolve_penalty_r(problem_data)
    nscen   = length(scenarios)
    pmodels = Vector{PersistentMKRecourseModel}(undef, nscen)

    # Per-scenario Gurobi.Env (see KS_r builder for rationale).
    for s in 1:nscen
        profits, weights, caps = decode_scenario_MK_r(scenarios[s], n, dims)

        env   = Gurobi.Env()
        model = direct_model(Gurobi.Optimizer(env))
        set_optimizer_attribute(model, "OutputFlag",   0)
        set_optimizer_attribute(model, "LogToConsole", 0)
        set_optimizer_attribute(model, "Threads",      1)

        @variable(model, x[1:n], Bin)
        @variable(model, v[1:dims] >= 0)

        for d in 1:dims
            @constraint(model, sum(weights[d, j] * x[j] for j in 1:n) - v[d] <= caps[d])
        end
        @objective(model, Min, -sum(profits[j] * x[j] for j in 1:n) + penalty * sum(v))

        grb = backend(model)
        work_supported = try; MOI.get(grb, _GRB_WORK); true; catch; false; end
        vi = MOI.VariableIndex[JuMP.index(xj) for xj in x]
        pmodels[s] = PersistentMKRecourseModel(model, x, collect(v), work_supported, grb, vi, zeros(n))
    end

    return pmodels
end

function solve_subproblem_persistent_MK_r(
    pm::PersistentMKRecourseModel,
    F0::UInt64,
    F1::UInt64
)
    grb = pm.grb
    vi  = pm.vi
    n   = length(vi)

    _apply_mask_bounds!(grb, _GRB_UB, vi, F0, 0.0)
    _apply_mask_bounds!(grb, _GRB_LB, vi, F1, 1.0)

    MOI.optimize!(grb)

    work_units = pm.work_supported ? MOI.get(grb, _GRB_WORK) : 0.0
    runtime    = MOI.get(grb, _GRB_RUNTIME)
    status     = MOI.get(grb, _GRB_STATUS)

    # Recourse model is always feasible
    if status == _GRB_OPTIMAL
        obj = MOI.get(grb, _GRB_OBJVAL)
        _read_x_into_buf!(grb, vi, pm.x_buf)
        result = (obj, pm.x_buf, work_units, runtime)
    else
        @inbounds for j in 1:n; pm.x_buf[j] = 0.0; end
        result = (Inf, pm.x_buf, work_units, runtime)
    end

    _apply_mask_bounds!(grb, _GRB_UB, vi, F0, 1.0)
    _apply_mask_bounds!(grb, _GRB_LB, vi, F1, 0.0)

    return result
end

function solve_subproblem_scenario_MK_r(
    problem_data::Dict,
    scenario::Vector{Float64},
    F0::UInt64,
    F1::UInt64
)
    n       = problem_data["n"]
    dims    = get(problem_data, "dims", MK_DIMS)
    penalty = _resolve_penalty_r(problem_data)
    profits, weights, caps = decode_scenario_MK_r(scenario, n, dims)

    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag",   0)
    set_optimizer_attribute(model, "LogToConsole", 0)

    @variable(model, x[1:n], Bin)
    @variable(model, v[1:dims] >= 0)

    _m = F0
    while _m != UInt64(0)
        j = trailing_zeros(_m) + 1
        fix(x[j], 0; force=true)
        _m &= _m - UInt64(1)
    end
    _m = F1
    while _m != UInt64(0)
        j = trailing_zeros(_m) + 1
        fix(x[j], 1; force=true)
        _m &= _m - UInt64(1)
    end

    for d in 1:dims
        @constraint(model, sum(weights[d, j] * x[j] for j in 1:n) - v[d] <= caps[d])
    end
    @objective(model, Min, -sum(profits[j] * x[j] for j in 1:n) + penalty * sum(v))

    optimize!(model)

    work_units = 0.0
    try; work_units = MOI.get(backend(model), Gurobi.ModelAttribute("Work")); catch; end
    runtime = solve_time(model)

    if termination_status(model) == MOI.OPTIMAL
        return objective_value(model), value.(x), work_units, runtime
    else
        return Inf, zeros(n), work_units, runtime
    end
end

# =============================================================================
# CONVENIENCE WRAPPER — MK with recourse
# =============================================================================

function branch_and_bound_MK_r(
    problem_data::Dict,
    scenarios::Vector{Vector{Float64}},
    lambda_star::Vector{Float64};
    work_limit::Float64             = 2000.0,
    time_limit::Float64             = Inf,
    cache::BBCache                  = BBCache(),
    log_file::Union{String,Nothing} = nothing,
    branching_strategy::String      = "F",
    verbose::Bool                   = false
)
    n_vars = problem_data["n"]
    return branch_and_bound(
        problem_data, scenarios, lambda_star, n_vars,
        build_persistent_models_MK_r,
        solve_subproblem_persistent_MK_r,
        solve_subproblem_scenario_MK_r;
        work_limit         = work_limit,
        time_limit         = time_limit,
        cache              = cache,
        log_file           = log_file,
        branching_strategy = branching_strategy,
        verbose            = verbose
    )
end

# =============================================================================
# CONVENIENCE WRAPPER — KS with recourse
# =============================================================================

function branch_and_bound_KS_r(
    problem_data::Dict,
    scenarios::Vector{Vector{Float64}},
    lambda_star::Vector{Float64};
    work_limit::Float64             = 2000.0,
    time_limit::Float64             = Inf,
    cache::BBCache                  = BBCache(),
    log_file::Union{String,Nothing} = nothing,
    branching_strategy::String      = "F",
    verbose::Bool                   = false
)
    n_vars = problem_data["n"]
    return branch_and_bound(
        problem_data, scenarios, lambda_star, n_vars,
        build_persistent_models_KS_r,
        solve_subproblem_persistent_KS_r,
        solve_subproblem_scenario_KS_r;
        work_limit         = work_limit,
        time_limit         = time_limit,
        cache              = cache,
        log_file           = log_file,
        branching_strategy = branching_strategy,
        verbose            = verbose
    )
end
