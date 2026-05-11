"""
lagrangian_pricing_grouping.jl

Lagrangian relaxation (FW-PH) for the partition-based pricing problem.

Analogous to lagrangian_pricing.jl but indexed by PARTITIONS instead of scenarios:
  - Subproblem (4)  ↔  Subproblem (44): partition type-2 MIP + mu^p bonus, one per partition
  - QP (6)          ↔  QP (46):         convex bundle QP, one per partition
  - mu^p per partition p,  sum_p mu^p = 0 maintained throughout
  - UB_k = sum_p L_p(mu^p) — valid upper bound, tighter than the plain partition bound

Usage:
  julia CCG/lagrangian_pricing_grouping.jl <instance_file> <scenarios_file>
        <binary_vectors_file> <dual_values_file> <L> <S> <cg_iter> <partition_file>
        [rho=5000] [epsilon=0.001] [max_iter=50] [scenario_indices_file=none]

Arguments:
  instance_file         : e.g. Facility_Location/instances/cap91.txt
  scenarios_file        : e.g. Samples/FL/Scenarios/instance_1/S300_bernoulli.txt
  binary_vectors_file   : e.g. Samples/FL/Binary_vectors/instance_1/B300_bernoulli.txt
  dual_values_file      : e.g. CCG/dual_lambda/CG_dual_lambda_S25_K1_cap91.txt
  L                     : bundle size (number of binary vectors)
  S                     : number of scenarios
  cg_iter               : which CG iteration's dual values to use (1-indexed)
  partition_file        : each line = space-separated 1-indexed scenario indices
  rho                   : FW-PH penalty parameter (default: 5000)
  epsilon               : stopping criterion on step norm (default: 0.001)
  max_iter              : max FW-PH iterations (default: 50)
  scenario_indices_file : optional file with S indices into the full scenario file
  unmet_pen             : unmet demand penalty (default: 15.0)
  scaling_factor        : transport cost scaling (default: 0.2)
  capacity_factor       : capacity scaling (default: 0.3)
"""

using JuMP, Gurobi, Dates, LinearAlgebra, Printf
import MathOptInterface as MOI
const MOI_LPG = MOI

const PROJECT_ROOT = dirname(@__DIR__)
push!(LOAD_PATH, joinpath(PROJECT_ROOT, "CCG"))
include(joinpath(PROJECT_ROOT, "CCG", "FacilityLocationAF.jl"))

# =============================================================================
# Helpers
# =============================================================================

function read_dual_values_lpg(path::String, target_iter::Int)
    isfile(path) || error("Dual values file not found: $path")
    open(path, "r") do io
        cur_iter      = 0
        capture_next  = false
        for line in eachline(io)
            line = strip(line)
            isempty(line) && continue
            if startswith(line, "iteration=")
                cur_iter     = parse(Int, split(line, "=")[2])
                capture_next = (cur_iter == target_iter)
            elseif capture_next
                return parse.(Float64, split(line))
            end
        end
        error("Iteration $target_iter not found in $path")
    end
end

function read_partition_file_lpg(filepath::String)
    partitions = Vector{Int}[]
    open(filepath, "r") do f
        for line in eachline(f)
            line = strip(line)
            isempty(line) && continue
            push!(partitions, parse.(Int, split(line)))
        end
    end
    isempty(partitions) && error("No partitions found in $filepath")
    return partitions
end

# =============================================================================
# Subproblem (44): partition MIP + (mu^p)^T x^p Lagrangian bonus
#
#   L_p(mu^p) = max_{x^p, x^s, y^s, pi^s}
#       sum_{s in S_p} [ lambda_s pi^s - (c^s)^T x^s - (d^s)^T y^s ]
#       + (mu^p)^T x^p
#
#   s.t.  A x^p >= b                                 (x^p in {0,1}^nloc for FL)
#         A x^s >= b pi^s          for all s in S_p  (absorbed into [0,1] bounds)
#         x^s <= e pi^s            for all s in S_p
#         -e(1-pi^s) <= x^p - x^s <= e(1-pi^s)   for all s in S_p
#         T^s x^s + W^s y^s >= h^s pi^s           for all s in S_p
#         y^s >= 0                 for all s in S_p
#
#   In the FL instantiation:
#     (c^s)^T x^s  = fixedcost^T x^s  (per scenario, on the copy)
#     (d^s)^T y^s  = scaling * transport(y^s) + unmet_pen * shortfall(b^s)
#     T^s x^s + W^s y^s >= h^s pi^s  = capacity + demand balance constraints
#
# Returns (L_p, work_units, x_sol_binary)   where x_sol_binary is x^p at optimum
# =============================================================================

function solve_partition_subproblem_4(
        partition::Vector{Int},
        nloc::Int, ncust::Int,
        capacity::Vector{Float64},
        cost::Matrix{Float64},
        fixedcost::Vector{Float64},
        scenarios::Vector{Vector{Float64}},
        dual_lambda::Vector{Float64},
        unmet_pen::Float64,
        scaling_factor::Float64,
        mu_p::Vector{Float64};
        log_file::Union{String,Nothing}=nothing,
        part_index::Union{Int,Nothing}=nothing)

    n_i = length(partition)

    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag",   1)
    set_optimizer_attribute(model, "LogToConsole", 0)
    if log_file !== nothing
        set_optimizer_attribute(model, "LogFile", log_file)
    end

    @variable(model, x[1:nloc],                      Bin)
    @variable(model, 0 <= x_s[1:nloc, 1:n_i] <= 1)
    @variable(model, y[1:nloc, 1:ncust, 1:n_i]    >= 0)
    @variable(model, b[1:ncust, 1:n_i]            >= 0)
    @variable(model, pi_v[1:n_i],                    Bin)

    @objective(model, Max,
        sum(dual_lambda[partition[k]] * pi_v[k] for k in 1:n_i) -
        sum(fixedcost[i] * x_s[i,k] for i in 1:nloc, k in 1:n_i) -
        scaling_factor * sum(cost[i,j] * y[i,j,k] for i in 1:nloc, j in 1:ncust, k in 1:n_i) -
        sum(unmet_pen * b[j,k] for j in 1:ncust, k in 1:n_i) +
        dot(mu_p, x))          # ← Lagrangian bonus term

    # Linking: x_s[:,k] = x when pi_v[k]=1, else x_s[:,k]=0
    @constraint(model, [i in 1:nloc, k in 1:n_i], x_s[i,k] <= pi_v[k])
    @constraint(model, [i in 1:nloc, k in 1:n_i], x[i] - x_s[i,k] <=  (1 - pi_v[k]))
    @constraint(model, [i in 1:nloc, k in 1:n_i], x[i] - x_s[i,k] >= -(1 - pi_v[k]))

    # Capacity
    @constraint(model, [i in 1:nloc, k in 1:n_i],
        sum(y[i,j,k] for j in 1:ncust) <= capacity[i] * x_s[i,k])

    # Demand
    @constraint(model, [j in 1:ncust, k in 1:n_i],
        sum(y[i,j,k] for i in 1:nloc) + b[j,k] == scenarios[partition[k]][j] * pi_v[k])

    optimize!(model)

    work_units = 0.0
    try work_units = MOI_LPG.get(backend(model), Gurobi.ModelAttribute("Work")) catch end

    status = termination_status(model)
    if status == MOI_LPG.OPTIMAL || status == MOI_LPG.FEASIBLE_POINT
        L_p   = objective_value(model)
        x_sol = round.(Int, value.(x))
        return L_p, work_units, x_sol
    else
        @warn "Partition subproblem $(part_index !== nothing ? "#$part_index" : "") failed: $status"
        return 0.0, work_units, zeros(Int, nloc)
    end
end

# =============================================================================
# QP (6): per-partition bundle QP — identical structure to QP (46)
#
#   max  sum_j q_{pj} alpha_j + (mu^p)^T x - (rho/2) ||x - x_bar||^2
#   s.t. x = sum_j v^{pj} alpha_j
#        sum_j alpha_j = 1,  alpha_j >= 0
#
# Returns (x_hat_p, obj_val, work_units)
# =============================================================================

function solve_qp_partition_6(
        V_x::Vector{Vector{Float64}},
        V_q::Vector{Float64},
        mu_p::Vector{Float64},
        x_bar::Vector{Float64},
        rho::Float64)

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
        sum(V_q[j] * alpha[j] for j in 1:nbundle) +
        dot(mu_p, x) -
        (rho / 2.0) * sum((x[i] - x_bar[i])^2 for i in 1:nloc))

    optimize!(model)

    work_units = 0.0
    try work_units = MOI_LPG.get(backend(model), Gurobi.ModelAttribute("Work")) catch end

    if termination_status(model) == MOI_LPG.OPTIMAL
        return value.(x), objective_value(model), work_units
    else
        @warn "QP (6) failed: $(termination_status(model))"
        return zeros(nloc), -Inf, work_units
    end
end

# =============================================================================
# Cost of solution x^{p'} in partition p  (= q_{pk} bundle value, no mu bonus)
#
#   q_p(x^{p'}) = max_{x^s, y^s, pi^s}
#       sum_{s in S_p} [ lambda_s pi^s - (c^s)^T x^s - (d^s)^T y^s ]
#
#   s.t.  x^s <= e pi^s                              for all s in S_p
#         -e(1-pi^s) <= x^{p'} - x^s <= e(1-pi^s)   for all s in S_p
#         sum_j y[i,j,s] <= cap[i] * x^s[i]          for all i, s
#         sum_i y[i,j,s] + b[j,s] = d^s[j] pi^s      for all j, s
#         x^s in [0,1]^nloc,  y^s >= 0,  pi^s in {0,1}
#
#   x^{p'} is FIXED (not a variable).  Returns the MIP objective value.
# =============================================================================

function compute_partition_cost_mip(
        x_prime::Vector{<:Real},
        partition::Vector{Int},
        nloc::Int, ncust::Int,
        capacity::Vector{Float64},
        cost::Matrix{Float64},
        fixedcost::Vector{Float64},
        scenarios::Vector{Vector{Float64}},
        dual_lambda::Vector{Float64},
        unmet_pen::Float64,
        scaling_factor::Float64)

    n_p = length(partition)

    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag",   0)
    set_optimizer_attribute(model, "LogToConsole", 0)

    @variable(model, 0 <= x_s[1:nloc, 1:n_p] <= 1)
    @variable(model, y[1:nloc, 1:ncust, 1:n_p] >= 0)
    @variable(model, b[1:ncust, 1:n_p]         >= 0)
    @variable(model, pi_v[1:n_p],               Bin)

    @objective(model, Max,
        sum(dual_lambda[partition[k]] * pi_v[k] for k in 1:n_p) -
        sum(fixedcost[i] * x_s[i,k] for i in 1:nloc, k in 1:n_p) -
        scaling_factor * sum(cost[i,j] * y[i,j,k] for i in 1:nloc, j in 1:ncust, k in 1:n_p) -
        sum(unmet_pen * b[j,k] for j in 1:ncust, k in 1:n_p))

    # x^s <= e pi^s
    @constraint(model, [i in 1:nloc, k in 1:n_p], x_s[i,k] <= pi_v[k])
    # -e(1-pi^s) <= x^{p'} - x^s <= e(1-pi^s)
    @constraint(model, [i in 1:nloc, k in 1:n_p], x_prime[i] - x_s[i,k] <=  (1 - pi_v[k]))
    @constraint(model, [i in 1:nloc, k in 1:n_p], x_prime[i] - x_s[i,k] >= -(1 - pi_v[k]))
    # Capacity
    @constraint(model, [i in 1:nloc, k in 1:n_p],
        sum(y[i,j,k] for j in 1:ncust) <= capacity[i] * x_s[i,k])
    # Demand
    @constraint(model, [j in 1:ncust, k in 1:n_p],
        sum(y[i,j,k] for i in 1:nloc) + b[j,k] == scenarios[partition[k]][j] * pi_v[k])

    optimize!(model)

    status = termination_status(model)
    if status == MOI_LPG.OPTIMAL || status == MOI_LPG.FEASIBLE_POINT
        return objective_value(model)
    else
        @warn "Partition cost MIP failed: $status — returning 0.0"
        return 0.0
    end
end

# =============================================================================
# Initial bundle: for each partition p and each bundle vector v,
#   q_p(v) = cost of v in partition p  (solved via compute_partition_cost_mip)
# =============================================================================

function build_initial_partition_bundles(
        fl_data::Dict,
        scenarios::Vector{Vector{Float64}},
        partitions::Vector{Vector{Int}},
        bundle_vectors::Vector{Vector{Float64}},
        lambda_star::Vector{Float64},
        log_io::IO)

    npart          = length(partitions)
    L              = length(bundle_vectors)
    nloc           = fl_data["nloc"]
    ncust          = fl_data["ncust"]
    capacity       = fl_data["capacity"]
    cost_mat       = fl_data["cost"]
    fixedcost      = fl_data["fixedcost"]
    unmet_pen      = get(fl_data, "unmet_pen",      15.0)
    scaling_factor = get(fl_data, "scaling_factor",  0.2)

    V_x = [copy(bundle_vectors) for _ in 1:npart]
    V_q = [zeros(L) for _ in 1:npart]

    println("  Building initial bundles: $npart partitions x $L vectors = $(npart*L) MIP solves")
    println(log_io, "Initial Bundle")

    for p in 1:npart
        println(log_io, "Partition $p (scenarios $(partitions[p])):")
        for j in 1:L
            q_val = compute_partition_cost_mip(
                bundle_vectors[j], partitions[p],
                nloc, ncust, capacity, cost_mat, fixedcost,
                scenarios, lambda_star, unmet_pen, scaling_factor)
            V_q[p][j] = q_val
            println(log_io, "  v[$j]: partition cost = $(round(q_val, digits=6))")
        end
    end
    flush(log_io)
    return V_x, V_q
end

# =============================================================================
# Main FW-PH loop
# =============================================================================

function lagrangian_partition_root(;
        instance_file::String,
        scenarios_file::String,
        binary_vectors_file::String,
        dual_values_file::String,
        L::Int,
        S::Int,
        cg_iter::Int,
        partition_file::String,
        rho::Float64     = 5000.0,
        epsilon::Float64 = 0.001,
        max_iter::Int    = 50,
        scenario_indices_file::Union{String,Nothing} = nothing,
        unmet_pen::Float64      = 15.0,
        scaling_factor::Float64 =  0.2,
        capacity_factor::Float64=  0.3)

    timestamp  = Dates.format(now(), "yyyymmdd_HHMMSS")
    inst_name  = splitext(basename(instance_file))[1]
    log_path   = joinpath(@__DIR__, "logs",
                     "lagrangian_grouping_$(inst_name)_S$(S)_$(timestamp).log")
    mkpath(dirname(log_path))
    log_io = open(log_path, "w")

    println("="^80)
    println("LAGRANGIAN PARTITION PRICING  (FW-PH over partitions)")
    println("="^80)
    println("  Instance        : $instance_file")
    println("  Scenarios       : $scenarios_file")
    println("  Indices file    : $(scenario_indices_file !== nothing ? scenario_indices_file : "none (first S)")")
    println("  Binary vectors  : $binary_vectors_file")
    println("  Dual values     : $dual_values_file  (CG iter $cg_iter)")
    println("  Partition file  : $partition_file")
    println("  L (bundle size) : $L")
    println("  S (scenarios)   : $S")
    println("  rho             : $rho")
    println("  epsilon         : $epsilon")
    println("  max_iter        : $max_iter")
    println("  unmet_pen       : $unmet_pen")
    println("  scaling_factor  : $scaling_factor")
    println("  capacity_factor : $capacity_factor")
    println("="^80)
    println("  Log file        : $log_path")
    println()

    # ── Load instance ─────────────────────────────────────────────────────────
    fl_data         = load_FL_data(instance_file; capacity_factor=capacity_factor)
    nloc            = fl_data["nloc"]
    ncust           = fl_data["ncust"]
    capacity        = fl_data["capacity"]
    cost_mat        = fl_data["cost"]
    fixedcost       = fl_data["fixedcost"]
    fl_data["unmet_pen"]      = unmet_pen
    fl_data["scaling_factor"] = scaling_factor

    # ── Load scenarios ────────────────────────────────────────────────────────
    all_scenarios = load_scenarios_from_file(scenarios_file)
    S > length(all_scenarios) && error("Requested S=$S but file has $(length(all_scenarios))")

    selected_indices = nothing
    if scenario_indices_file !== nothing
        selected_indices = parse.(Int, split(strip(read(scenario_indices_file, String))))
        length(selected_indices) != S && error(
            "Indices file has $(length(selected_indices)) indices but S=$S")
        scenarios = all_scenarios[selected_indices]
        println("  Selected scenario indices: $selected_indices")
    else
        scenarios = all_scenarios[1:S]
    end

    # ── Load binary vectors ───────────────────────────────────────────────────
    all_bv = load_binary_vectors_from_file(binary_vectors_file)
    L > length(all_bv) && error("Requested L=$L but file has $(length(all_bv))")
    bundle_vectors = [Float64.(v) for v in all_bv[1:L]]

    # ── Load dual lambda ──────────────────────────────────────────────────────
    all_lambdas = read_dual_values_lpg(dual_values_file, cg_iter)
    lambda_star = selected_indices !== nothing ? all_lambdas[selected_indices] : all_lambdas[1:S]
    println("  lambda_star: $(length(lambda_star)) values, sum=$(round(sum(lambda_star), digits=2))")

    # ── Load and validate partition ───────────────────────────────────────────
    partitions = read_partition_file_lpg(partition_file)
    npart      = length(partitions)
    println("  Partitions: $npart")
    for (p, part) in enumerate(partitions)
        println("    Partition $p: $(length(part)) scenarios — $part")
        for idx in part
            (idx < 1 || idx > S) && error("Partition $p has out-of-range index $idx (S=$S)")
        end
    end
    println()

    # ── Build initial bundles ─────────────────────────────────────────────────
    V_x, V_q = build_initial_partition_bundles(
        fl_data, scenarios, partitions, bundle_vectors, lambda_star, log_io)

    # ── Initialize mu^p and x_bar ─────────────────────────────────────────────
    # v^p = first scenario's binary vector in each partition (warm-start proxy)
    ws_vectors = [Float64.(all_bv[partitions[p][1]]) for p in 1:npart]
    mu    = [zeros(nloc) for _ in 1:npart]           # mu^p_0 = 0
    x_bar = sum(ws_vectors) ./ npart                 # x_bar_0
    # mu^p_1 = mu^p_0 - rho*(v^p - x_bar_0)
    for p in 1:npart
        mu[p] = -rho .* (ws_vectors[p] .- x_bar)
    end

    # ── FW-PH loop ────────────────────────────────────────────────────────────
    println(log_io)
    println(log_io, "Starting FW-PH iterations (over partitions)...")
    println(log_io, "  iter |    UB (sum L_p)    |     best UB        |    step_norm     |   iter GWU")
    println(log_io, "  ----- + ------------------- + ------------------- + ---------------- + -----------")

    best_ub        = Inf
    best_iter      = 0
    history        = Tuple{Int, Float64, Float64}[]
    wu_subproblems = 0.0
    wu_qps         = 0.0
    last_new_v     = [zeros(nloc) for _ in 1:npart]
    last_x_hat     = [zeros(nloc) for _ in 1:npart]

    for k in 1:max_iter
        # ── Step 1: solve subproblem (44) per partition ──────────────────────
        UB_k    = 0.0
        wu_iter = 0.0
        new_v   = [zeros(nloc) for _ in 1:npart]

        for p in 1:npart
            L_p, wu, x_sol = solve_partition_subproblem_4(
                partitions[p], nloc, ncust, capacity, cost_mat, fixedcost,
                scenarios, lambda_star, unmet_pen, scaling_factor, mu[p])
            UB_k           += L_p
            new_v[p]        = Float64.(x_sol)
            wu_subproblems += wu
            wu_iter        += wu

            # partition cost with x^{p'} = x_sol fixed  (q_{pk})
            q_pk = compute_partition_cost_mip(
                Float64.(x_sol), partitions[p],
                nloc, ncust, capacity, cost_mat, fixedcost,
                scenarios, lambda_star, unmet_pen, scaling_factor)
            push!(V_x[p], Float64.(x_sol))
            push!(V_q[p], q_pk)
        end

        if UB_k < best_ub
            best_ub   = UB_k
            best_iter = k
        end

        # ── Step 2: solve QP (6) per partition ───────────────────────────────
        x_hat = [zeros(nloc) for _ in 1:npart]
        for p in 1:npart
            x_hat[p], _, wu = solve_qp_partition_6(V_x[p], V_q[p], mu[p], x_bar, rho)
            wu_qps  += wu
            wu_iter += wu
        end

        # ── Step 3: stopping criterion ────────────────────────────────────────
        sq_sum = 0.0
        for p in 1:npart
            d = x_hat[p] .- x_bar
            sq_sum += dot(d, d)
        end
        step_norm = sqrt(sq_sum / npart)

        last_new_v  = new_v
        last_x_hat  = x_hat
        @printf(log_io, "  %5d |  %18.4f |  %18.4f |  %16.6f |  %10.4f\n",
                k, UB_k, best_ub, step_norm, wu_iter)
        flush(log_io)
        push!(history, (k, UB_k, step_norm))

        if step_norm < epsilon
            println(log_io, "\n  Stopping: step_norm=$(round(step_norm, digits=8)) < epsilon=$epsilon")
            break
        end

        # ── Step 4: update mu^p and x_bar ────────────────────────────────────
        x_bar_new = sum(x_hat) ./ npart
        for p in 1:npart
            mu[p] = mu[p] .- rho .* (x_hat[p] .- x_bar_new)
        end
        x_bar = x_bar_new
    end

    # ── x44_bar vs x46_bar summary ────────────────────────────────────────────
    x44_bar = sum(last_new_v) ./ npart
    x46_bar = sum(last_x_hat) ./ npart
    rms_diff = sqrt(sum((x44_bar .- x46_bar).^2) / nloc)
    for io in (stdout, log_io)
        println(io)
        println(io, "x44_bar (mean of subproblem-44 solutions): $(round.(x44_bar, digits=6))")
        println(io, "x46_bar (mean of QP-46 solutions)        : $(round.(x46_bar, digits=6))")
        @printf(io, "RMS(x44_bar - x46_bar)                   : %.8f\n", rms_diff)
    end

    wu_total = wu_subproblems + wu_qps

    for io in (stdout, log_io)
        println(io)
        println(io, "="^80)
        println(io, "FINAL RESULT")
        println(io, "="^80)
        println(io, "  Best Lagrangian UB : $best_ub   (at iter $best_iter)")
        println(io, "  Total iterations   : $(length(history))")
        println(io, "="^80)
        println(io, "GUROBI WORK UNITS")
        println(io, "  Subproblems (44) MILPs : $(round(wu_subproblems, digits=4))")
        println(io, "  QPs (6)               : $(round(wu_qps, digits=4))")
        println(io, "  Total                 : $(round(wu_total, digits=4))")
        println(io, "="^80)
        println(io, "FINAL MU VECTORS (at termination)")
        for p in 1:npart
            println(io, "  mu[p=$p]: $(round.(mu[p], digits=6))  |  component-sum = $(round(sum(mu[p]), digits=6))")
        end
        mu_cross_sum = sum(mu)
        println(io, "  Component-wise sum across all p (should be ~0): $(round.(mu_cross_sum, digits=8))")
        println(io, "  Max abs deviation from 0: $(round(maximum(abs.(mu_cross_sum)), digits=8))")
        println(io, "="^80)
    end

    close(log_io)
    return Dict("best_ub" => best_ub, "best_iter" => best_iter, "history" => history)
end

# =============================================================================
# CLI
# =============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 8
        println("""
Usage: julia CCG/lagrangian_pricing_grouping.jl <instance_file> <scenarios_file>
       <binary_vectors_file> <dual_values_file> <L> <S> <cg_iter> <partition_file>
       [rho=5000] [epsilon=0.001] [max_iter=50] [scenario_indices_file=none]
       [unmet_pen=15.0] [scaling_factor=0.2] [capacity_factor=0.3]
        """)
        exit(1)
    end

    instance_file         = ARGS[1]
    scenarios_file        = ARGS[2]
    binary_vectors_file   = ARGS[3]
    dual_values_file      = ARGS[4]
    L                     = parse(Int, ARGS[5])
    S                     = parse(Int, ARGS[6])
    cg_iter               = parse(Int, ARGS[7])
    partition_file        = ARGS[8]
    rho                   = length(ARGS) >= 9  ? parse(Float64, ARGS[9])  : 5000.0
    epsilon               = length(ARGS) >= 10 ? parse(Float64, ARGS[10]) : 0.05
    max_iter              = length(ARGS) >= 11 ? parse(Int,     ARGS[11]) : 30
    scenario_indices_file = (length(ARGS) >= 12 && ARGS[12] != "none") ? ARGS[12] : nothing
    unmet_pen             = length(ARGS) >= 13 ? parse(Float64, ARGS[13]) : 15.0
    scaling_factor        = length(ARGS) >= 14 ? parse(Float64, ARGS[14]) : 0.2
    capacity_factor       = length(ARGS) >= 15 ? parse(Float64, ARGS[15]) : 0.3

    lagrangian_partition_root(;
        instance_file         = instance_file,
        scenarios_file        = scenarios_file,
        binary_vectors_file   = binary_vectors_file,
        dual_values_file      = dual_values_file,
        L                     = L,
        S                     = S,
        cg_iter               = cg_iter,
        partition_file        = partition_file,
        rho                   = rho,
        epsilon               = epsilon,
        max_iter              = max_iter,
        scenario_indices_file = scenario_indices_file,
        unmet_pen             = unmet_pen,
        scaling_factor        = scaling_factor,
        capacity_factor       = capacity_factor)
end
