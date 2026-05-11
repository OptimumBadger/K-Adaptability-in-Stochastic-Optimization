# lagrangian_pricing.jl
# Standalone Frank-Wolfe Progressive Hedging Lagrangian relaxation of the AF
# pricing problem (41).  Computes a Lagrangian upper bound at the root node for
# a given dual vector lambda_star (read from a CG_dual_values_*.txt file).
#
# Usage:
#   julia CCG/lagrangian_pricing.jl <instance_file> <scenarios_file> \
#         <binary_vectors_file> <dual_values_file> <L> <S> <cg_iter> \
#         [rho=1.0] [epsilon=0.001] [max_iter=200]

using JuMP, Gurobi
using MathOptInterface
using Printf, LinearAlgebra, Dates
const MOI_LR = MathOptInterface

include("FacilityLocationAF.jl")  # transitively includes FLUtilities (load_*, evaluate_cost_FL)

# =============================================================================
# DUAL VALUES FILE READER
# =============================================================================

function read_dual_values(path::String, target_iter::Int)
    isfile(path) || error("Dual values file not found: $path")
    open(path, "r") do io
        cur_iter = 0
        capture_next = false
        for line in eachline(io)
            line = strip(line)
            isempty(line) && continue
            if startswith(line, "iteration=")
                cur_iter = parse(Int, split(line, "=")[2])
                capture_next = (cur_iter == target_iter)
            elseif capture_next
                return parse.(Float64, split(line))
            end
        end
        error("Iteration $target_iter not found in $path")
    end
end

# =============================================================================
# SUBPROBLEM (44):  L_s(mu^s) = max_{x,y,pi} (lambda_s* - c^s x - d^s y) pi + mu^s x
# Solved via case-split: take the better of pi=0 and pi=1.
# =============================================================================

function solve_subproblem_pi0(mu_s::Vector{Float64})
    # max{ mu^s . x : x in {0,1}^p }   (FL has no Ax >= b constraints)
    nloc = length(mu_s)
    x = [mu_s[j] > 0.0 ? 1.0 : 0.0 for j in 1:nloc]
    obj = sum(max(0.0, mu_s[j]) for j in 1:nloc)
    return obj, x
end

function solve_subproblem_pi1(
    fl_data::Dict,
    scenario::Vector{Float64},
    lambda_s::Float64,
    mu_s::Vector{Float64}
)
    nloc           = fl_data["nloc"]
    ncust          = fl_data["ncust"]
    capacity       = fl_data["capacity"]
    cost           = fl_data["cost"]
    fixedcost      = fl_data["fixedcost"]
    unmet_pen      = fl_data["unmet_pen"]
    scaling_factor = fl_data["scaling_factor"]

    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag",   0)
    set_optimizer_attribute(model, "LogToConsole", 0)

    @variable(model, x[1:nloc], Bin)
    @variable(model, y[1:nloc, 1:ncust] >= 0)
    @variable(model, shortfall[1:ncust]  >= 0)

    @constraint(model, [j in 1:ncust],
        sum(y[i,j] for i in 1:nloc) + shortfall[j] == scenario[j])
    @constraint(model, [i in 1:nloc],
        sum(y[i,j] for j in 1:ncust) - capacity[i] * x[i] <= 0)

    # max lambda_s - fixed.x - scaling*cost.y - unmet.shortfall + mu.x
    #   = lambda_s + (mu - fixed).x - scaling*cost.y - unmet.shortfall
    @objective(model, Max,
        lambda_s
        + sum((mu_s[i] - fixedcost[i]) * x[i] for i in 1:nloc)
        - scaling_factor * sum(cost[i,j] * y[i,j] for i in 1:nloc for j in 1:ncust)
        - sum(unmet_pen * shortfall[j] for j in 1:ncust))

    optimize!(model)

    wu = MOI_LR.get(backend(model), Gurobi.ModelAttribute("Work"))
    if termination_status(model) == MOI_LR.OPTIMAL
        return objective_value(model), value.(x), wu
    else
        return -Inf, zeros(nloc), wu
    end
end

function solve_subproblem_44(
    fl_data::Dict,
    scenario::Vector{Float64},
    lambda_s::Float64,
    mu_s::Vector{Float64}
)
    L0, x0 = solve_subproblem_pi0(mu_s)
    L1, x1, wu = solve_subproblem_pi1(fl_data, scenario, lambda_s, mu_s)
    if L1 >= L0
        return L1, x1, 1, wu
    else
        return L0, x0, 0, wu
    end
end

# =============================================================================
# QP SUBPROBLEM (46):  per-scenario bundle update with proximal term
#
#   max  sum_j q_{sj} alpha_j + mu^s . x - (rho/2) ||x - x_bar_prev||^2
#   s.t. x = sum_j v^{sj} alpha_j
#        sum_j alpha_j = 1
#        alpha_j >= 0
# =============================================================================

function solve_qp_46(
    V_x::Vector{Vector{Float64}},        # bundle vectors
    V_q::Vector{Float64},                # bundle q-values
    mu_s::Vector{Float64},
    x_bar_prev::Vector{Float64},
    rho::Float64
)
    nbundle = length(V_x)
    nloc    = length(V_x[1])

    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag",   0)
    set_optimizer_attribute(model, "LogToConsole", 0)

    @variable(model, alpha[1:nbundle] >= 0)
    @variable(model, x[1:nloc])

    @constraint(model, sum(alpha) == 1.0)
    @constraint(model, [i in 1:nloc],
        x[i] == sum(V_x[j][i] * alpha[j] for j in 1:nbundle))

    @objective(model, Max,
        sum(V_q[j] * alpha[j] for j in 1:nbundle)
        + sum(mu_s[i] * x[i] for i in 1:nloc)
        - (rho / 2.0) * sum((x[i] - x_bar_prev[i])^2 for i in 1:nloc))

    optimize!(model)

    wu = MOI_LR.get(backend(model), Gurobi.ModelAttribute("Work"))
    if termination_status(model) == MOI_LR.OPTIMAL
        return value.(x), objective_value(model), wu
    else
        @warn "QP (46) failed: $(termination_status(model))"
        return zeros(nloc), -Inf, wu
    end
end

# =============================================================================
# INITIAL BUNDLE: for each scenario s, compute q_s(v) = max(0, lambda_s - f(v, w_s))
# for every v in the initial set X_hat (first L binary vectors).
# =============================================================================

function build_initial_bundles(
    fl_data::Dict,
    scenarios::Vector{Vector{Float64}},
    bundle_vectors::Vector{Vector{Float64}},
    lambda_star::Vector{Float64},
    log_io::IO
)
    S = length(scenarios)
    L = length(bundle_vectors)

    V_x = [copy(bundle_vectors) for _ in 1:S]
    V_q = [zeros(L) for _ in 1:S]

    println("  Building initial bundles: $S scenarios x $L vectors = $(S*L) recourse evaluations")
    println(log_io, "Initial Bundle")
    for s in 1:S
        println(log_io, "Scenario $s:")
        for j in 1:L
            x_int = round.(Int, bundle_vectors[j])
            f_val, _ = evaluate_cost_FL(x_int, scenarios[s], fl_data)
            q_val = max(0.0, lambda_star[s] - f_val)
            V_q[s][j] = q_val
            println(log_io, "  Solution $j: max(lambda_s* - f(x,w)) = $(round(q_val, digits=6))")
        end
    end
    flush(log_io)
    return V_x, V_q
end

# =============================================================================
# MAIN
# =============================================================================

function lagrangian_root(;
    instance_file::String,
    scenarios_file::String,
    binary_vectors_file::String,
    dual_values_file::String,
    L::Int,
    S::Int,
    cg_iter::Int,
    rho::Float64        = 5000.0,
    epsilon::Float64    = 0.1,
    max_iter::Int       = 50,
    capacity_factor::Float64 = 0.3,
    unmet_pen::Float64       = 15.0,
    scaling_factor::Float64  = 0.2,
    scenario_indices_file::Union{String,Nothing} = nothing
)
    inst_name = splitext(basename(instance_file))[1]
    log_path  = joinpath(@__DIR__, "lagrangian_log_$(inst_name)_S$(S)_L$(L)_iter$(max_iter)_rho$(rho)_$(Dates.format(now(), "yyyymmdd_HHMMSS")).txt")
    log_io   = open(log_path, "w")

    println("="^80)
    println("LAGRANGIAN ROOT-NODE BOUND  (Frank-Wolfe Progressive Hedging)")
    println("="^80)
    println("  Instance        : $instance_file")
    println("  Scenarios       : $scenarios_file")
    println("  Indices file    : $(scenario_indices_file !== nothing ? scenario_indices_file : "none (first S)")")
    println("  Binary vectors  : $binary_vectors_file")
    println("  Dual values     : $dual_values_file  (CG iter $cg_iter)")
    println("  L (bundle size) : $L")
    println("  S (scenarios)   : $S")
    println("  rho             : $rho")
    println("  epsilon         : $epsilon")
    println("  max_iter        : $max_iter")
    println("="^80)
    println("  Log file        : $log_path")
    println()

    # ---- Load data ----------------------------------------------------------
    fl_data = load_FL_data(instance_file; capacity_factor=capacity_factor)
    fl_data["unmet_pen"]      = unmet_pen
    fl_data["scaling_factor"] = scaling_factor
    nloc = fl_data["nloc"]

    all_scenarios = load_scenarios_from_file(scenarios_file)
    S > length(all_scenarios) && error(
        "Requested S=$S but scenarios file has $(length(all_scenarios))")
    if scenario_indices_file !== nothing
        selected_indices = parse.(Int, split(strip(read(scenario_indices_file, String))))
        length(selected_indices) != S && error(
            "Indices file has $(length(selected_indices)) indices but S=$S")
        any(i -> i < 1 || i > length(all_scenarios), selected_indices) && error(
            "Indices file contains out-of-range index (max=$(length(all_scenarios)))")
        scenarios = all_scenarios[selected_indices]
        println("  Selected scenario indices: $selected_indices")
    else
        scenarios = all_scenarios[1:S]
    end

    all_bv = load_binary_vectors_from_file(binary_vectors_file)
    L > length(all_bv) && error(
        "Requested L=$L but binary vectors file has $(length(all_bv))")
    S > length(all_bv) && error(
        "Need at least S=$S binary vectors (one wait-and-see solution per scenario), file has $(length(all_bv))")

    bundle_vectors = [Float64.(v) for v in all_bv[1:L]]   # X_hat
    ws_indices     = scenario_indices_file !== nothing ? selected_indices : collect(1:S)
    ws_vectors     = [Float64.(v) for v in all_bv[ws_indices]]   # v^s per selected scenario

    all_lambdas = read_dual_values(dual_values_file, cg_iter)
    lambda_star = scenario_indices_file !== nothing ? all_lambdas[selected_indices] : all_lambdas[1:S]
    println("  Loaded lambda_star: $(length(lambda_star)) values, sum=$(round(sum(lambda_star), digits=2))")
    println()

    # ---- Initialize bundles V_0^s ------------------------------------------
    V_x, V_q = build_initial_bundles(fl_data, scenarios, bundle_vectors, lambda_star, log_io)

    # ---- Initialize mu and x_bar -------------------------------------------
    mu    = [zeros(nloc) for _ in 1:S]                 # mu_0^s = 0
    x_bar = sum(ws_vectors) ./ S                       # x_bar_0
    # mu_1^s = mu_0^s - rho * (v^s - x_bar_0)
    for s in 1:S
        mu[s] = -rho .* (ws_vectors[s] .- x_bar)
    end

    println(log_io)
    println(log_io, "Starting FW-PH iterations...")
    println(log_io, "  iter |     UB(L=sum L_s)  |   best UB         | step ||x_hat - x_bar_prev|| |   iter GWU")
    println(log_io, "  ----- + ------------------- + ------------------- + ------------------------------- + ------------")

    best_ub        = Inf
    best_iter      = 0
    best_x_bar_44  = zeros(nloc)
    best_x_bar_46  = zeros(nloc)
    history        = Tuple{Int, Float64, Float64}[]
    wu_subproblems = 0.0
    wu_qps         = 0.0

    for k in 1:max_iter
        # ----- Step 1: solve subproblems (44) per scenario -------------------
        UB_k    = 0.0
        wu_iter = 0.0
        new_v = [zeros(nloc) for _ in 1:S]
        for s in 1:S
            L_s, x_s, _, wu = solve_subproblem_44(fl_data, scenarios[s], lambda_star[s], mu[s])
            UB_k           += L_s
            new_v[s]        = x_s
            wu_subproblems += wu
            wu_iter        += wu
            x_int = round.(Int, x_s)
            f_val, _ = evaluate_cost_FL(x_int, scenarios[s], fl_data)
            q_ks = max(0.0, lambda_star[s] - f_val)
            push!(V_x[s], x_s)
            push!(V_q[s], q_ks)
        end

        if UB_k < best_ub
            best_ub       = UB_k
            best_iter     = k
            best_x_bar_44 = sum(new_v) ./ S
        end

        # ----- Step 2: solve QP (46) per scenario ----------------------------
        x_hat = [zeros(nloc) for _ in 1:S]
        for s in 1:S
            x_hat[s], _, wu = solve_qp_46(V_x[s], V_q[s], mu[s], x_bar, rho)
            wu_qps  += wu
            wu_iter += wu
        end

        if k == best_iter
            best_x_bar_46 = sum(x_hat) ./ S
        end

        # ----- Step 3: stopping criterion (uses x_bar = x_bar_{k-1}) ---------
        sq_sum = 0.0
        for s in 1:S
            d = x_hat[s] .- x_bar
            sq_sum += dot(d, d)
        end
        step_norm = sqrt(sq_sum / S)

        @printf(log_io, "  %5d |  %18.4f |  %18.4f |  %30.6f |  %10.4f\n",
                k, UB_k, best_ub, step_norm, wu_iter)
        flush(log_io)
        push!(history, (k, UB_k, step_norm))

        if step_norm < epsilon
            println(log_io, "\n  Stopping criterion satisfied: step_norm=$(round(step_norm, digits=8)) < epsilon=$epsilon")
            break
        end

        # ----- Step 4: update mu and x_bar -----------------------------------
        x_bar_new = sum(x_hat) ./ S
        for s in 1:S
            mu[s] = mu[s] .- rho .* (x_hat[s] .- x_bar_new)
        end
        x_bar = x_bar_new
    end

    wu_total = wu_subproblems + wu_qps
    rms_44_46 = sqrt(sum((best_x_bar_44[j] - best_x_bar_46[j])^2 for j in 1:nloc) / nloc)
    for io in (stdout, log_io)
        println(io)
        println(io, "="^80)
        println(io, "FINAL RESULT")
        println(io, "="^80)
        println(io, "  Best Lagrangian UB : $(best_ub)   (at iter $best_iter)")
        println(io, "  Total iterations    : $(length(history))")
        println(io, "="^80)
        println(io, "CONVERGENCE METRIC (at best iter $best_iter)")
        println(io, "  x_bar_44 (avg of SP-44 solutions) : $(round.(best_x_bar_44, digits=4))")
        println(io, "  x_bar_46 (avg of QP-46 solutions) : $(round.(best_x_bar_46, digits=4))")
        println(io, "  RMS(x_bar_44, x_bar_46)           : $(round(rms_44_46, digits=6))")
        println(io, "="^80)
        println(io, "GUROBI WORK UNITS")
        println(io, "  FW-PH subproblems (MILPs) : $(round(wu_subproblems, digits=4))")
        println(io, "  FW-PH QPs                 : $(round(wu_qps, digits=4))")
        println(io, "  FW-PH total               : $(round(wu_total, digits=4))")
        println(io, "="^80)
        println(io, "FINAL MU VECTORS (at termination)")
        for s in 1:S
            println(io, "  mu[s=$s]: $(round.(mu[s], digits=6))  |  component-sum = $(round(sum(mu[s]), digits=6))")
        end
        mu_cross_sum = sum(mu)   # component-wise sum across all scenarios — should be ~0
        println(io, "  Component-wise sum across all s (should be ~0): $(round.(mu_cross_sum, digits=8))")
        println(io, "  Max abs deviation from 0: $(round(maximum(abs.(mu_cross_sum)), digits=8))")
        println(io, "="^80)
    end

    close(log_io)

    return Dict(
        "best_ub"   => best_ub,
        "best_iter" => best_iter,
        "history"   => history,
    )
end

# =============================================================================
# CLI
# =============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 7
        println("Usage: julia CCG/lagrangian_pricing.jl <instance_file> <scenarios_file> <binary_vectors_file> <dual_values_file> <L> <S> <cg_iter> [rho=5000] [epsilon=0.001] [max_iter=50] [scenario_indices_file=none]")
        exit(1)
    end

    instance_file         = ARGS[1]
    scenarios_file        = ARGS[2]
    binary_vectors_file   = ARGS[3]
    dual_values_file      = ARGS[4]
    L                     = parse(Int, ARGS[5])
    S                     = parse(Int, ARGS[6])
    cg_iter               = parse(Int, ARGS[7])
    rho                   = length(ARGS) >= 8  ? parse(Float64, ARGS[8])  : 5000.0
    epsilon               = length(ARGS) >= 9  ? parse(Float64, ARGS[9])  : 0.001
    max_iter              = length(ARGS) >= 10 ? parse(Int,     ARGS[10]) : 50
    scenario_indices_file = length(ARGS) >= 11 ? ARGS[11] : nothing

    lagrangian_root(
        instance_file         = instance_file,
        scenarios_file        = scenarios_file,
        binary_vectors_file   = binary_vectors_file,
        dual_values_file      = dual_values_file,
        L                     = L,
        S                     = S,
        cg_iter               = cg_iter,
        rho                   = rho,
        epsilon               = epsilon,
        max_iter              = max_iter,
        scenario_indices_file = scenario_indices_file,
    )
end
