# GroupBranchAndBound.jl
# Grouped Branch and Bound for solving the AF pricing subproblem (29a):
#
#   max_{x in X}  sum_{s in [N]} max{0, lambda_s* - f(x, omega_s)}
#
# Instead of solving each scenario independently (wait-and-see), scenarios are
# grouped into partitions of size ~5.  For each group S_i at node (F0, F1) we solve:
#
#   P_i*(F0,F1) = max_{x in X}  sum_{s in S_i} max{0, lambda_s* - f(x, omega_s)}
#                 s.t.  x_j = 0  for j in F0
#                       x_j = 1  for j in F1
#
# using shared x and pi[k] binary variables to linearise the max(0,…) clipping,
# exactly as in the partition Type-2 pricing formulation.  Summing over groups
# gives a valid (and tighter than wait-and-see) upper bound at each node:
#
#   U_k = sum_i P_i*(F0^k, F1^k)
#
# NOTE: this file assumes BranchAndBound.jl has already been included so that
# the BBNode struct is available.

using JuMP, Gurobi
using MathOptInterface
using Printf
const MOI_GBB = MathOptInterface   # avoid re-const if already defined

# =============================================================================
# SCENARIO GROUPING
# =============================================================================

"""
    make_groups(nscen, group_size=5) -> Vector{Vector{Int}}

Partition 1:nscen into groups of `group_size`.
- Remainder 1 or 2  → merge into the last full group  (last group has 6 or 7)
- Remainder 3 or 4  → keep as a separate tail group
"""
function make_groups(nscen::Int, group_size::Int=8)
    n_full    = div(nscen, group_size)
    remainder = mod(nscen, group_size)

    groups = Vector{Int}[]

    if remainder == 0
        for g in 1:n_full
            push!(groups, collect((g-1)*group_size+1 : g*group_size))
        end
    elseif remainder <= 2
        # Merge remainder into last full group
        for g in 1:n_full-1
            push!(groups, collect((g-1)*group_size+1 : g*group_size))
        end
        last_start = (n_full - 1) * group_size + 1
        push!(groups, collect(last_start:nscen))
    else
        # remainder is 3 or 4 — keep as its own group
        for g in 1:n_full
            push!(groups, collect((g-1)*group_size+1 : g*group_size))
        end
        push!(groups, collect(n_full*group_size+1 : nscen))
    end

    return groups
end

# =============================================================================
# GROUP SUBPROBLEM SOLVER
# Solves the Type-2 pricing MIP for one group with variable fixings (F0, F1).
# =============================================================================

"""
    solve_group_node(partition, problem_data, scenarios, lambda_star, F0, F1)
        -> (P_i*, work_units, x_sol)

For the given group (1-indexed global scenario indices), solves:

    P_i* = max_{x in X}  sum_{k in group} max{0, lambda_{s_k}* - f(x, omega_{s_k})}
           s.t.  x_j = 0  for j in F0
                 x_j = 1  for j in F1
"""
function solve_group_node(
    partition::Vector{Int},
    problem_data::Dict,
    scenarios::Vector{Vector{Float64}},
    lambda_star::Vector{Float64},
    F0::Set{Int},
    F1::Set{Int}
)
    nloc           = problem_data["nloc"]
    ncust          = problem_data["ncust"]
    capacity       = problem_data["capacity"]
    cost           = problem_data["cost"]
    fixedcost      = problem_data["fixedcost"]
    unmet_pen      = problem_data["unmet_pen"]
    scaling_factor = problem_data["scaling_factor"]

    n_i = length(partition)

    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag",   0)
    set_optimizer_attribute(model, "LogToConsole", 0)

    # ── Variables ─────────────────────────────────────────────────────────────
    @variable(model, x[1:nloc],               Bin)          # shared facility vector
    @variable(model, 0 <= x_s[1:nloc, 1:n_i] <= 1)         # x copy per scenario
    @variable(model, y[1:nloc, 1:ncust, 1:n_i] >= 0)        # transport
    @variable(model, b[1:ncust, 1:n_i]        >= 0)          # shortfall
    @variable(model, pi_var[1:n_i],            Bin)          # 1 = scenario active

    # ── Apply B&B node fixings ─────────────────────────────────────────────────
    for j in F0
        fix(x[j], 0; force=true)
    end
    for j in F1
        fix(x[j], 1; force=true)
    end

    # ── Objective ─────────────────────────────────────────────────────────────
    @objective(model, Max,
        sum(lambda_star[partition[k]] * pi_var[k] for k in 1:n_i) -
        sum(fixedcost[i] * x_s[i,k] for i in 1:nloc, k in 1:n_i) -
        scaling_factor * sum(cost[i,j] * y[i,j,k] for i in 1:nloc, j in 1:ncust, k in 1:n_i) -
        sum(unmet_pen * b[j,k] for j in 1:ncust, k in 1:n_i))

    # ── Linking: x_s[i,k] = x[i] when pi_var[k]=1, else 0 ───────────────────
    @constraint(model, [i in 1:nloc, k in 1:n_i], x_s[i,k] <= pi_var[k])
    @constraint(model, [i in 1:nloc, k in 1:n_i], x[i] - x_s[i,k] <=  (1 - pi_var[k]))
    @constraint(model, [i in 1:nloc, k in 1:n_i], x[i] - x_s[i,k] >= -(1 - pi_var[k]))

    # ── Capacity ──────────────────────────────────────────────────────────────
    @constraint(model, [i in 1:nloc, k in 1:n_i],
        sum(y[i,j,k] for j in 1:ncust) <= capacity[i] * x_s[i,k])

    # ── Demand ────────────────────────────────────────────────────────────────
    @constraint(model, [j in 1:ncust, k in 1:n_i],
        sum(y[i,j,k] for i in 1:nloc) + b[j,k] == scenarios[partition[k]][j] * pi_var[k])

    optimize!(model)

    work_units = 0.0
    try
        work_units = MOI_GBB.get(backend(model), Gurobi.ModelAttribute("Work"))
    catch
    end

    status = termination_status(model)
    if status == MOI_GBB.OPTIMAL || status == MOI_GBB.FEASIBLE_POINT
        return objective_value(model), work_units, value.(x)
    else
        return 0.0, work_units, zeros(nloc)
    end
end

# =============================================================================
# NODE EVALUATION: solve all groups, compute U_k and x_bar
# =============================================================================

"""
    evaluate_node_grouped(groups, problem_data, scenarios, lambda_star, F0, F1)
        -> (U_k, x_bar, total_work_units)

Evaluates a B&B node by solving one group subproblem per partition.
U_k = sum_i P_i*  (valid upper bound on the pricing objective at this node).
x_bar is the weighted mean of per-group x solutions (weight = group size).
"""
function evaluate_node_grouped(
    groups::Vector{Vector{Int}},
    problem_data::Dict,
    scenarios::Vector{Vector{Float64}},
    lambda_star::Vector{Float64},
    F0::Set{Int},
    F1::Set{Int}
)
    nloc       = problem_data["nloc"]
    nscen      = sum(length(g) for g in groups)
    U_k        = 0.0
    total_work = 0.0
    x_bar      = zeros(nloc)

    for group in groups
        P_i, wu, x_sol = solve_group_node(group, problem_data, scenarios, lambda_star, F0, F1)
        U_k       += P_i
        total_work += wu
        x_bar     .+= length(group) .* x_sol
    end

    x_bar ./= nscen
    return U_k, x_bar, total_work
end

# =============================================================================
# MAIN GROUPED BRANCH AND BOUND
# =============================================================================

function branch_and_bound_grouped(
    problem_data::Dict,
    scenarios::Vector{Vector{Float64}},
    lambda_star::Vector{Float64};
    work_limit::Float64 = 2000.0,
    log_file::Union{String,Nothing} = nothing
)
    nloc  = problem_data["nloc"]
    nscen = length(scenarios)

    log_io = log_file !== nothing ? open(log_file, "w") : stdout

    try
        println(log_io, "\n" * "="^60)
        println(log_io, "GROUPED BRANCH AND BOUND — PRICING SUBPROBLEM")
        println(log_io, "  Facilities : $nloc")
        println(log_io, "  Scenarios  : $nscen")
        println(log_io, "  Branching  : MOST-FRACTIONAL")

        # ------------------------------------------------------------------
        # Build groups once (shared across all nodes)
        # ------------------------------------------------------------------
        groups    = make_groups(nscen)
        ngroups   = length(groups)
        group_sizes = length.(groups)
        println(log_io, "  Groups     : $ngroups  sizes=$(group_sizes)")
        println(log_io, "="^60)

        total_work_units = 0.0

        # ------------------------------------------------------------------
        # Step 1: Evaluate root node
        # ------------------------------------------------------------------
        U_root, x_bar_root, wu_root =
            evaluate_node_grouped(groups, problem_data, scenarios, lambda_star,
                                  Set{Int}(), Set{Int}())
        total_work_units += wu_root

        println(log_io, "  Root UB = $(round(U_root, digits=6))")

        # ------------------------------------------------------------------
        # Step 2: Warm-start incumbent — round x_bar at root and evaluate
        # ------------------------------------------------------------------
        x_ws      = Int[floor(Int, x_bar_root[j] + 0.5) for j in 1:nloc]
        F0_ws     = Set{Int}([j for j in 1:nloc if x_ws[j] == 0])
        F1_ws     = Set{Int}([j for j in 1:nloc if x_ws[j] == 1])
        L_ws, _, wu_ws =
            evaluate_node_grouped(groups, problem_data, scenarios, lambda_star, F0_ws, F1_ws)
        total_work_units += wu_ws

        L_hat  = L_ws
        x_star = x_ws
        println(log_io, "  Warm-start incumbent (rounded root x_bar): L_hat = $(round(L_hat, digits=6))")

        nodes_processed = 0
        open_nodes = BBNode[BBNode(Set{Int}(), Set{Int}(), U_root)]

        # ------------------------------------------------------------------
        # Step 3: B&B loop
        # ------------------------------------------------------------------
        work_limit_hit = false

        while !isempty(open_nodes)

            if total_work_units >= work_limit
                work_limit_hit = true
                break
            end

            # Best-first search
            idx  = argmax(n.upper_bound for n in open_nodes)
            node = open_nodes[idx]
            deleteat!(open_nodes, idx)

            if node.upper_bound <= L_hat
                continue
            end

            nodes_processed += 1

            U_k, x_bar, wu_node =
                evaluate_node_grouped(groups, problem_data, scenarios, lambda_star,
                                      node.F0, node.F1)
            total_work_units += wu_node

            UB  = isempty(open_nodes) ? U_k : max(U_k, maximum(n.upper_bound for n in open_nodes))
            LB  = L_hat
            gap = UB > 1e-10 ? 100.0 * (UB - LB) / abs(UB) : 0.0

            if U_k <= L_hat
                @printf(log_io, "  Node %4d  |F0|=%2d  |F1|=%2d  U_k=%.4f  LB=%.4f  UB=%.4f  gap=%.2f%%  work=%.4f  outcome=FATHOMED  branch_var=∅  open_nodes=%d\n",
                        nodes_processed, length(node.F0), length(node.F1),
                        U_k, LB, UB, gap, total_work_units, length(open_nodes))
                continue
            end

            # Check if x_bar is integer (feasible leaf)
            if all(v -> isapprox(v, 0.0; atol=1e-6) || isapprox(v, 1.0; atol=1e-6), x_bar)
                if U_k > L_hat
                    L_hat  = U_k
                    x_star = round.(Int, x_bar)
                    println(log_io, "  *** New incumbent: L_hat = $(round(L_hat, digits=6))")
                end
                UB  = isempty(open_nodes) ? L_hat : maximum(n.upper_bound for n in open_nodes)
                gap = UB > 1e-10 ? 100.0 * (UB - L_hat) / abs(UB) : 0.0
                @printf(log_io, "  Node %4d  |F0|=%2d  |F1|=%2d  U_k=%.4f  LB=%.4f  UB=%.4f  gap=%.2f%%  work=%.4f  outcome=FEASIBLE  branch_var=∅  open_nodes=%d\n",
                        nodes_processed, length(node.F0), length(node.F1),
                        U_k, L_hat, UB, gap, total_work_units, length(open_nodes))
                continue
            end

            # Fractional variables (not already fixed)
            frac_vars = [j for j in 1:nloc
                         if !(j in node.F0) && !(j in node.F1) &&
                            !isapprox(x_bar[j], 0.0; atol=1e-6) &&
                            !isapprox(x_bar[j], 1.0; atol=1e-6)]

            # ------------------------------------------------------------------
            # Rounding heuristic: evaluate rounded x_bar as candidate incumbent
            # ------------------------------------------------------------------
            x_rounded  = Int[floor(Int, x_bar[j] + 0.5) for j in 1:nloc]
            F0_rounded = Set{Int}([j for j in 1:nloc if x_rounded[j] == 0])
            F1_rounded = Set{Int}([j for j in 1:nloc if x_rounded[j] == 1])
            LB_rnd, _, wu_rnd =
                evaluate_node_grouped(groups, problem_data, scenarios, lambda_star,
                                      F0_rounded, F1_rounded)
            total_work_units += wu_rnd
            if LB_rnd > L_hat
                L_hat  = LB_rnd
                x_star = x_rounded
                println(log_io, "  *** New incumbent (rounding heuristic): L_hat = $(round(L_hat, digits=6))")
            end

            # ------------------------------------------------------------------
            # Most-fractional branching
            # ------------------------------------------------------------------
            j_star, frac_score = select_branching_variable(frac_vars, x_bar)

            F0_down = union(node.F0, Set([j_star]))
            F1_up   = union(node.F1, Set([j_star]))

            push!(open_nodes, BBNode(F0_down, node.F1, U_k))
            push!(open_nodes, BBNode(node.F0, F1_up,   U_k))

            UB  = maximum(n.upper_bound for n in open_nodes)
            gap = UB > 1e-10 ? 100.0 * (UB - L_hat) / abs(UB) : 0.0
            @printf(log_io, "  Node %4d  |F0|=%2d  |F1|=%2d  U_k=%.4f  LB=%.4f  UB=%.4f  gap=%.2f%%  work=%.4f  outcome=BRANCHED  branch_var=%d  frac_score=%.4f  nfrac=%d  open_nodes=%d\n",
                    nodes_processed, length(node.F0), length(node.F1),
                    U_k, L_hat, UB, gap, total_work_units,
                    j_star, frac_score, length(frac_vars), length(open_nodes))
        end

        # Final summary
        UB_final  = isempty(open_nodes) ? L_hat : maximum(n.upper_bound for n in open_nodes)
        gap_final = UB_final > 1e-10 ? 100.0 * (UB_final - L_hat) / abs(UB_final) : 0.0

        println(log_io, "="^60)
        if work_limit_hit
            println(log_io, "B&B terminated: work limit reached ($(round(total_work_units, digits=4)) work units)")
        else
            println(log_io, "B&B complete: $nodes_processed nodes processed")
        end
        println(log_io, "LB (best feasible) : $(round(L_hat, digits=6))")
        println(log_io, "UB (best node)     : $(round(UB_final, digits=6))")
        println(log_io, "Gap                : $(round(gap_final, digits=4))%")
        println(log_io, "Optimal x*         : $x_star")
        println(log_io, "Total work units   : $(round(total_work_units, digits=4))")
        println(log_io, "="^60)

        return x_star, L_hat, UB_final, gap_final, total_work_units, nodes_processed

    finally
        if log_file !== nothing
            close(log_io)
        end
    end
end
