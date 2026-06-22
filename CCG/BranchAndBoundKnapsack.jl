# BranchAndBoundKnapsack.jl
# Knapsack (KS, without recourse) — problem-specific functions for the B&B
# pricing framework.
#
# Subproblem at node (F0, F1) for scenario s:
#   g_s(F0,F1) = min_{x binary}  -profit_s^T x
#                s.t.  weight_s^T x <= capacity_s
#                      x_j = 0  for j in F0,  x_j = 1  for j in F1
#
# Scenario encoding (from KnapsackAF.jl):
#   sc = [profit_1..profit_n, weight_1..weight_n, capacity]
#   decode_scenario_KS(sc, n)  ->  (profits, weights, capacity)
#
# Requires BranchAndBoundCore.jl to be included first.

# =============================================================================
# PERSISTENT SCENARIO MODEL — KS (without recourse)
# =============================================================================

struct PersistentKSModel
    model::Model
    x::Vector{VariableRef}
    work_supported::Bool
    grb::Gurobi.Optimizer            # = backend(model) in direct mode
    vi::Vector{MOI.VariableIndex}    # opt-side variable indices for x
    x_buf::Vector{Float64}           # preallocated solution buffer
end

function build_persistent_models_KS(
    problem_data::Dict,
    scenarios::Vector{Vector{Float64}}
)
    n     = problem_data["n"]
    nscen = length(scenarios)
    pmodels = Vector{PersistentKSModel}(undef, nscen)

    # Per-scenario Gurobi.Env so concurrent solves from Threads.@threads are safe.
    # Gurobi.jl's README: Envs are NOT thread-safe to share. Models sharing an
    # env cannot be solved simultaneously on different threads. So we give each
    # scenario its own env. (Academic license — env tokens are unrestricted.)
    for s in 1:nscen
        profits, weights, cap = decode_scenario_KS(scenarios[s], n)

        env   = Gurobi.Env()
        model = direct_model(Gurobi.Optimizer(env))
        set_optimizer_attribute(model, "OutputFlag",   0)
        set_optimizer_attribute(model, "LogToConsole", 0)
        # Single-thread each Gurobi solve; parallelism is at the scenario level.
        set_optimizer_attribute(model, "Threads",      1)

        @variable(model, x[1:n], Bin)
        @constraint(model, sum(weights[j] * x[j] for j in 1:n) <= cap)
        @objective(model, Min, -sum(profits[j] * x[j] for j in 1:n))

        grb = backend(model)
        work_supported = try; MOI.get(grb, _GRB_WORK); true; catch; false; end
        vi = MOI.VariableIndex[JuMP.index(xj) for xj in x]
        pmodels[s] = PersistentKSModel(model, x, work_supported, grb, vi, zeros(n))
    end

    return pmodels
end

function solve_subproblem_persistent_KS(
    pm::PersistentKSModel,
    F0::UInt64,
    F1::UInt64
)
    grb = pm.grb
    vi  = pm.vi
    n   = length(vi)

    _t = time_ns()
    _apply_mask_bounds!(grb, _GRB_UB, vi, F0, 0.0)
    _apply_mask_bounds!(grb, _GRB_LB, vi, F1, 1.0)
    Threads.atomic_add!(_T_SOLVE_SET_F0F1, time_ns() - _t)

    _t = time_ns()
    MOI.optimize!(grb)
    Threads.atomic_add!(_T_SOLVE_OPTIMIZE, time_ns() - _t)

    _t = time_ns()
    work_units = pm.work_supported ? MOI.get(grb, _GRB_WORK) : 0.0
    status     = MOI.get(grb, _GRB_STATUS)
    Threads.atomic_add!(_T_SOLVE_READ_META, time_ns() - _t)

    if status == _GRB_OPTIMAL
        _t = time_ns()
        obj = MOI.get(grb, _GRB_OBJVAL)
        Threads.atomic_add!(_T_SOLVE_READ_META, time_ns() - _t)

        _t = time_ns()
        _read_x_into_buf!(grb, vi, pm.x_buf)
        Threads.atomic_add!(_T_SOLVE_READ_X, time_ns() - _t)

        result = (obj, pm.x_buf, work_units)
    else
        # Infeasible: F1 items already exceed capacity — this scenario contributes 0
        @inbounds for j in 1:n; pm.x_buf[j] = 0.0; end
        result = (Inf, pm.x_buf, work_units)
    end

    _t = time_ns()
    _apply_mask_bounds!(grb, _GRB_UB, vi, F0, 1.0)
    _apply_mask_bounds!(grb, _GRB_LB, vi, F1, 0.0)
    Threads.atomic_add!(_T_SOLVE_RESET_F0F1, time_ns() - _t)

    return result
end

function solve_subproblem_scenario_KS(
    problem_data::Dict,
    scenario::Vector{Float64},
    F0::UInt64,
    F1::UInt64
)
    n = problem_data["n"]
    profits, weights, cap = decode_scenario_KS(scenario, n)

    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag",   0)
    set_optimizer_attribute(model, "LogToConsole", 0)

    @variable(model, x[1:n], Bin)

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

    @constraint(model, sum(weights[j] * x[j] for j in 1:n) <= cap)
    @objective(model, Min, -sum(profits[j] * x[j] for j in 1:n))

    optimize!(model)

    work_units = 0.0
    try; work_units = MOI.get(backend(model), Gurobi.ModelAttribute("Work")); catch; end

    if termination_status(model) == MOI.OPTIMAL
        return objective_value(model), value.(x), work_units
    else
        return Inf, zeros(n), work_units
    end
end

# =============================================================================
# PERSISTENT SCENARIO MODEL — MK (without recourse)
# =============================================================================

struct PersistentMKModel
    model::Model
    x::Vector{VariableRef}
    work_supported::Bool
    grb::Gurobi.Optimizer
    vi::Vector{MOI.VariableIndex}
    x_buf::Vector{Float64}
end

function build_persistent_models_MK(
    problem_data::Dict,
    scenarios::Vector{Vector{Float64}}
)
    n     = problem_data["n"]
    dims  = get(problem_data, "dims", MK_DIMS)
    nscen = length(scenarios)
    pmodels = Vector{PersistentMKModel}(undef, nscen)

    for s in 1:nscen
        profits, weights, caps = decode_scenario_MK(scenarios[s], n, dims)

        env   = Gurobi.Env()
        model = direct_model(Gurobi.Optimizer(env))
        set_optimizer_attribute(model, "OutputFlag",   0)
        set_optimizer_attribute(model, "LogToConsole", 0)
        set_optimizer_attribute(model, "Threads",      1)

        @variable(model, x[1:n], Bin)
        for d in 1:dims
            @constraint(model, sum(weights[d, j] * x[j] for j in 1:n) <= caps[d])
        end
        @objective(model, Min, -sum(profits[j] * x[j] for j in 1:n))

        grb = backend(model)
        work_supported = try; MOI.get(grb, _GRB_WORK); true; catch; false; end
        vi = MOI.VariableIndex[JuMP.index(xj) for xj in x]
        pmodels[s] = PersistentMKModel(model, x, work_supported, grb, vi, zeros(n))
    end

    return pmodels
end

function solve_subproblem_persistent_MK(
    pm::PersistentMKModel,
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
    status     = MOI.get(grb, _GRB_STATUS)

    if status == _GRB_OPTIMAL
        obj = MOI.get(grb, _GRB_OBJVAL)
        _read_x_into_buf!(grb, vi, pm.x_buf)
        result = (obj, pm.x_buf, work_units)
    else
        @inbounds for j in 1:n; pm.x_buf[j] = 0.0; end
        result = (Inf, pm.x_buf, work_units)
    end

    _apply_mask_bounds!(grb, _GRB_UB, vi, F0, 1.0)
    _apply_mask_bounds!(grb, _GRB_LB, vi, F1, 0.0)

    return result
end

function solve_subproblem_scenario_MK(
    problem_data::Dict,
    scenario::Vector{Float64},
    F0::UInt64,
    F1::UInt64
)
    n    = problem_data["n"]
    dims = get(problem_data, "dims", MK_DIMS)
    profits, weights, caps = decode_scenario_MK(scenario, n, dims)

    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag",   0)
    set_optimizer_attribute(model, "LogToConsole", 0)

    @variable(model, x[1:n], Bin)

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
        @constraint(model, sum(weights[d, j] * x[j] for j in 1:n) <= caps[d])
    end
    @objective(model, Min, -sum(profits[j] * x[j] for j in 1:n))

    optimize!(model)

    work_units = 0.0
    try; work_units = MOI.get(backend(model), Gurobi.ModelAttribute("Work")); catch; end

    if termination_status(model) == MOI.OPTIMAL
        return objective_value(model), value.(x), work_units
    else
        return Inf, zeros(n), work_units
    end
end

# =============================================================================
# CONVENIENCE WRAPPER
# =============================================================================

function branch_and_bound_MK(
    problem_data::Dict,
    scenarios::Vector{Vector{Float64}},
    lambda_star::Vector{Float64};
    work_limit::Float64             = 2000.0,
    cache::BBCache                  = BBCache(),
    log_file::Union{String,Nothing} = nothing,
    branching_strategy::String      = "F",
    verbose::Bool                   = false
)
    n_vars = problem_data["n"]
    return branch_and_bound(
        problem_data, scenarios, lambda_star, n_vars,
        build_persistent_models_MK,
        solve_subproblem_persistent_MK,
        solve_subproblem_scenario_MK;
        work_limit         = work_limit,
        cache              = cache,
        log_file           = log_file,
        branching_strategy = branching_strategy,
        verbose            = verbose
    )
end

# =============================================================================
# CONVENIENCE WRAPPER — KS
# =============================================================================

function branch_and_bound_KS(
    problem_data::Dict,
    scenarios::Vector{Vector{Float64}},
    lambda_star::Vector{Float64};
    work_limit::Float64             = 2000.0,
    cache::BBCache                  = BBCache(),
    log_file::Union{String,Nothing} = nothing,
    branching_strategy::String      = "F",
    verbose::Bool                   = false
)
    n_vars = problem_data["n"]
    return branch_and_bound(
        problem_data, scenarios, lambda_star, n_vars,
        build_persistent_models_KS,
        solve_subproblem_persistent_KS,
        solve_subproblem_scenario_KS;
        work_limit         = work_limit,
        cache              = cache,
        log_file           = log_file,
        branching_strategy = branching_strategy,
        verbose            = verbose
    )
end
