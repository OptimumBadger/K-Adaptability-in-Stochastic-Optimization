# BranchAndBoundFL.jl
# Facility Location — problem-specific functions for the B&B pricing framework.
#
# Subproblem at node (F0, F1):
#   g_s(F0,F1) = min_{x,y,sf}  fixedcost^T x + scaling * cost:y + unmet_pen * sf
#                s.t.  demand satisfaction, capacity constraints
#                      x_j = 0  for j in F0,  x_j = 1  for j in F1
#
# Requires BranchAndBoundCore.jl to be included first (for BBCache, BBNode, etc.)

# =============================================================================
# PERSISTENT SCENARIO MODEL — FL
# =============================================================================

struct PersistentScenarioModel
    model::Model
    x::Vector{VariableRef}
    y::Matrix{VariableRef}
    shortfall::Vector{VariableRef}
end

function build_persistent_models_FL(
    problem_data::Dict,
    scenarios::Vector{Vector{Float64}}
)
    nloc           = problem_data["nloc"]
    ncust          = problem_data["ncust"]
    capacity       = problem_data["capacity"]
    cost           = problem_data["cost"]
    fixedcost      = problem_data["fixedcost"]
    unmet_pen      = problem_data["unmet_pen"]
    scaling_factor = problem_data["scaling_factor"]
    nscen          = length(scenarios)

    pmodels = Vector{PersistentScenarioModel}(undef, nscen)

    for s in 1:nscen
        scenario = scenarios[s]
        model    = Model(Gurobi.Optimizer)
        set_optimizer_attribute(model, "OutputFlag",   0)
        set_optimizer_attribute(model, "LogToConsole", 0)

        @variable(model, x_s[1:nloc], Bin)
        @variable(model, y_s[1:nloc, 1:ncust] >= 0)
        @variable(model, sf_s[1:ncust] >= 0)

        @constraint(model, [j in 1:ncust],
            sum(y_s[i,j] for i in 1:nloc) + sf_s[j] == scenario[j])
        @constraint(model, [i in 1:nloc],
            sum(y_s[i,j] for j in 1:ncust) - capacity[i] * x_s[i] <= 0)

        @objective(model, Min,
            sum(fixedcost[i] * x_s[i] for i in 1:nloc) +
            scaling_factor * sum(cost[i,j] * y_s[i,j] for i in 1:nloc, j in 1:ncust) +
            sum(unmet_pen * sf_s[j] for j in 1:ncust))

        pmodels[s] = PersistentScenarioModel(model, x_s, y_s, sf_s)
    end

    return pmodels
end

function solve_subproblem_persistent_FL(
    pm::PersistentScenarioModel,
    F0::UInt64,
    F1::UInt64
)
    n_vars = length(pm.x)

    _m = F0
    while _m != UInt64(0)
        j = trailing_zeros(_m) + 1
        fix(pm.x[j], 0; force=true)
        _m &= _m - UInt64(1)
    end
    _m = F1
    while _m != UInt64(0)
        j = trailing_zeros(_m) + 1
        fix(pm.x[j], 1; force=true)
        _m &= _m - UInt64(1)
    end

    optimize!(pm.model)

    work_units = 0.0
    try; work_units = MOI.get(backend(pm.model), Gurobi.ModelAttribute("Work")); catch; end
    runtime = solve_time(pm.model)

    if termination_status(pm.model) == MOI.OPTIMAL
        result = (objective_value(pm.model), value.(pm.x), work_units, runtime)
    else
        result = (Inf, zeros(n_vars), work_units, runtime)
    end

    _m = F0
    while _m != UInt64(0)
        j = trailing_zeros(_m) + 1
        unfix(pm.x[j])
        _m &= _m - UInt64(1)
    end
    _m = F1
    while _m != UInt64(0)
        j = trailing_zeros(_m) + 1
        unfix(pm.x[j])
        _m &= _m - UInt64(1)
    end

    return result
end

function solve_subproblem_scenario_FL(
    problem_data::Dict,
    scenario::Vector{Float64},
    F0::UInt64,
    F1::UInt64
)
    nloc           = problem_data["nloc"]
    ncust          = problem_data["ncust"]
    capacity       = problem_data["capacity"]
    cost           = problem_data["cost"]
    fixedcost      = problem_data["fixedcost"]
    unmet_pen      = problem_data["unmet_pen"]
    scaling_factor = problem_data["scaling_factor"]

    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag",   0)
    set_optimizer_attribute(model, "LogToConsole", 0)

    @variable(model, x[1:nloc], Bin)
    @variable(model, y[1:nloc, 1:ncust] >= 0)
    @variable(model, shortfall[1:ncust]  >= 0)

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

    @constraint(model, [j in 1:ncust],
        sum(y[i,j] for i in 1:nloc) + shortfall[j] == scenario[j])
    @constraint(model, [i in 1:nloc],
        sum(y[i,j] for j in 1:ncust) - capacity[i] * x[i] <= 0)

    @objective(model, Min,
        sum(fixedcost[i] * x[i] for i in 1:nloc) +
        scaling_factor * sum(cost[i,j] * y[i,j] for i in 1:nloc, j in 1:ncust) +
        sum(unmet_pen * shortfall[j] for j in 1:ncust))

    optimize!(model)

    work_units = 0.0
    try; work_units = MOI.get(backend(model), Gurobi.ModelAttribute("Work")); catch; end
    runtime = solve_time(model)

    if termination_status(model) == MOI.OPTIMAL
        return objective_value(model), value.(x), work_units, runtime
    else
        return Inf, zeros(nloc), work_units, runtime
    end
end

# =============================================================================
# CONVENIENCE WRAPPER  (same external signature as the old branch_and_bound)
# =============================================================================

function branch_and_bound_FL(
    problem_data::Dict,
    scenarios::Vector{Vector{Float64}},
    lambda_star::Vector{Float64};
    work_limit::Float64            = 2000.0,
    time_limit::Float64            = Inf,
    cache::BBCache                 = BBCache(),
    log_file::Union{String,Nothing} = nothing,
    branching_strategy::String     = "F"
)
    n_vars = problem_data["nloc"]
    return branch_and_bound(
        problem_data, scenarios, lambda_star, n_vars,
        build_persistent_models_FL,
        solve_subproblem_persistent_FL,
        solve_subproblem_scenario_FL;
        work_limit         = work_limit,
        time_limit         = time_limit,
        cache              = cache,
        log_file           = log_file,
        branching_strategy = branching_strategy
    )
end
