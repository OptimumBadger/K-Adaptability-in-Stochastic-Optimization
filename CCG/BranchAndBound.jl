# BranchAndBound.jl
# Custom Branch and Bound for solving the AF pricing subproblem (29a):
#
#   max_{x in X}  sum_{s in [N]} max{0, lambda_s* - f(x, omega_s)}
#
# The key decomposition: for fixed (F0, F1), each scenario independently solves
#
#   g_s(F0, F1) = min  (c_s)^T x + (d_s)^T y          (36a)
#                 s.t. T^s x + W^s y >= h^s
#                      x in X
#                      x_j = 0  for j in F0
#                      x_j = 1  for j in F1
#
# giving the node upper bound U_k = sum_s max{0, lambda_s* - g_s(F0^k, F1^k)}.

using JuMP, Gurobi
using MathOptInterface
using Printf
const MOI = MathOptInterface

# =============================================================================
# NODE STRUCTURE AND CACHE
# =============================================================================

struct BBNode
    F0::Set{Int}             # Variables fixed to 0
    F1::Set{Int}             # Variables fixed to 1
    upper_bound::Float64     # UB estimate (parent's U_k) used for best-first ordering
end

# Cache key: (F0, F1) as sorted tuples for hashing
# Cache value: (g_s_vec, x_sk) — both independent of lambda_star, fully reusable
const BBCacheKey   = Tuple{Vector{Int}, Vector{Int}}   # (sorted F0, sorted F1)
const BBCacheValue = Tuple{Vector{Float64}, Matrix{Float64}}  # (g_s_vec, x_sk)
const BBCache      = Dict{BBCacheKey, BBCacheValue}

cache_key(F0::Set{Int}, F1::Set{Int}) = (sort(collect(F0)), sort(collect(F1)))

# =============================================================================
# PERSISTENT SCENARIO MODELS
# Build S models once; reuse across nodes by adjusting bounds (no rebuild).
# =============================================================================

struct PersistentScenarioModel
    model::Model
    x::Vector{VariableRef}
    y::Matrix{VariableRef}
    shortfall::Vector{VariableRef}
end

function build_persistent_models(
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
        model = Model(Gurobi.Optimizer)
        set_optimizer_attribute(model, "OutputFlag",   0)
        set_optimizer_attribute(model, "LogToConsole", 0)

        @variable(model, x_s[1:nloc], Bin)
        @variable(model, y_s[1:nloc, 1:ncust] >= 0)
        @variable(model, sf_s[1:ncust] >= 0)
        x         = x_s
        y         = y_s
        shortfall = sf_s

        @constraint(model, [j in 1:ncust],
            sum(y[i,j] for i in 1:nloc) + shortfall[j] == scenario[j])
        @constraint(model, [i in 1:nloc],
            sum(y[i,j] for j in 1:ncust) - capacity[i] * x[i] <= 0)

        @objective(model, Min,
            sum(fixedcost[i] * x[i] for i in 1:nloc) +
            scaling_factor * sum(cost[i,j] * y[i,j] for i in 1:nloc for j in 1:ncust) +
            sum(unmet_pen * shortfall[j] for j in 1:ncust))

        pmodels[s] = PersistentScenarioModel(model, x, y, shortfall)
    end

    return pmodels
end

function solve_subproblem_persistent(
    pm::PersistentScenarioModel,
    F0::Set{Int},
    F1::Set{Int}
)
    nloc = length(pm.x)

    # Fix variables (removes integrality, same as fresh model approach)
    for j in F0
        fix(pm.x[j], 0; force=true)
    end
    for j in F1
        fix(pm.x[j], 1; force=true)
    end

    optimize!(pm.model)

    work_units = 0.0
    try
        work_units = MOI.get(backend(pm.model), Gurobi.ModelAttribute("Work"))
    catch
    end

    if termination_status(pm.model) == MOI.OPTIMAL
        result = (objective_value(pm.model), value.(pm.x), work_units)
    else
        result = (Inf, zeros(nloc), work_units)
    end

    # Unfix variables for next node
    for j in F0
        unfix(pm.x[j])
    end
    for j in F1
        unfix(pm.x[j])
    end

    return result
end

# =============================================================================
# SUBPROBLEM SOLVER: g_s(F0, F1) FOR ONE SCENARIO  (36a)
# =============================================================================

function solve_subproblem_scenario(
    problem_data::Dict,
    scenario::Vector{Float64},
    F0::Set{Int},
    F1::Set{Int}
)
    nloc          = problem_data["nloc"]
    ncust         = problem_data["ncust"]
    capacity      = problem_data["capacity"]
    cost          = problem_data["cost"]
    fixedcost     = problem_data["fixedcost"]
    unmet_pen     = problem_data["unmet_pen"]
    scaling_factor = problem_data["scaling_factor"]

    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag",    0)
    set_optimizer_attribute(model, "LogToConsole",  0)

    @variable(model, x[1:nloc], Bin)
    @variable(model, y[1:nloc, 1:ncust] >= 0)
    @variable(model, shortfall[1:ncust]  >= 0)

    # Apply variable fixings
    for j in F0
        fix(x[j], 0; force=true)
    end
    for j in F1
        fix(x[j], 1; force=true)
    end

    # Demand satisfaction
    @constraint(model, [j in 1:ncust],
        sum(y[i,j] for i in 1:nloc) + shortfall[j] == scenario[j])

    # Capacity
    @constraint(model, [i in 1:nloc],
        sum(y[i,j] for j in 1:ncust) - capacity[i] * x[i] <= 0)

    # Minimise cost (c_s^T x + d_s^T y in the paper)
    @objective(model, Min,
        sum(fixedcost[i] * x[i] for i in 1:nloc) +
        scaling_factor * sum(cost[i,j] * y[i,j] for i in 1:nloc for j in 1:ncust) +
        sum(unmet_pen * shortfall[j] for j in 1:ncust))

    optimize!(model)

    work_units = 0.0
    try
        work_units = MOI.get(backend(model), Gurobi.ModelAttribute("Work"))
    catch
    end

    if termination_status(model) == MOI.OPTIMAL
        return objective_value(model), value.(x), work_units
    else
        return Inf, zeros(nloc), work_units
    end
end

# =============================================================================
# NODE EVALUATION: solve all N scenario subproblems, compute U_k
# =============================================================================

function evaluate_node(
    problem_data::Dict,
    scenarios::Vector{Vector{Float64}},
    F0::Set{Int},
    F1::Set{Int},
    lambda_star::Vector{Float64}
)
    nloc  = problem_data["nloc"]
    nscen = length(scenarios)

    x_sk       = zeros(nloc, nscen)
    g_s_vec    = zeros(nscen)
    work_units = 0.0

    for s in 1:nscen
        g_s, x_s, wu   = solve_subproblem_scenario(problem_data, scenarios[s], F0, F1)
        g_s_vec[s]     = g_s
        x_sk[:, s]     = x_s
        work_units    += wu
    end

    U_k = sum(max(0.0, lambda_star[s] - g_s_vec[s]) for s in 1:nscen)

    return U_k, x_sk, g_s_vec, work_units
end

function evaluate_node_persistent(
    pmodels::Vector{PersistentScenarioModel},
    F0::Set{Int},
    F1::Set{Int},
    lambda_star::Vector{Float64}
)
    nloc  = length(pmodels[1].x)
    nscen = length(pmodels)

    x_sk       = zeros(nloc, nscen)
    g_s_vec    = zeros(nscen)
    work_units = 0.0

    for s in 1:nscen
        g_s, x_s, wu = solve_subproblem_persistent(pmodels[s], F0, F1)
        g_s_vec[s]   = g_s
        x_sk[:, s]   = x_s
        work_units  += wu
    end

    U_k = sum(max(0.0, lambda_star[s] - g_s_vec[s]) for s in 1:nscen)
    return U_k, x_sk, g_s_vec, work_units
end

# =============================================================================
# SENSITIVITY PRE-COMPUTATION
# Computes lambda_{sj}^0 and lambda_{sj}^1 at the root (F0 = F1 = empty).
#
# Let x^s = optimal solution of 36a at root for scenario s.
# If x_j^s = 0:
#   lambda0[s,j] = 0          (no cost to stay at 0)
#   lambda1[s,j] = g_s(∅,{j}) - g_root[s]   (cost of forcing x_j = 1)
# If x_j^s = 1:
#   lambda0[s,j] = g_s({j},∅) - g_root[s]   (cost of forcing x_j = 0)
#   lambda1[s,j] = 0          (no cost to stay at 1)
# =============================================================================

function precompute_sensitivities(
    problem_data::Dict,
    scenarios::Vector{Vector{Float64}},
    log_io::IO
)
    nloc  = problem_data["nloc"]
    nscen = length(scenarios)
    empty_set = Set{Int}()

    println(log_io, "  [BB] Precomputing sensitivities...")
    println(log_io, "  [BB] Step 1/2 — root solves ($nscen scenarios)...")

    # Step 1: Solve root for every scenario
    x_root  = zeros(nloc, nscen)
    g_root  = zeros(nscen)

    work_units = 0.0
    for s in 1:nscen
        g_s, x_s, wu = solve_subproblem_scenario(problem_data, scenarios[s], empty_set, empty_set)
        g_root[s]    = g_s
        x_root[:, s] = x_s
        work_units  += wu
        println(log_io, "    scenario $s: g_s(root) = $g_s")
    end

    # Step 2: For each variable j, one extra solve per scenario
    println(log_io, "  [BB] Step 2/2 — sensitivity solves ($nloc variables × $nscen scenarios)...")

    lambda0 = zeros(nscen, nloc)
    lambda1 = zeros(nscen, nloc)

    for j in 1:nloc
        F0_j = Set{Int}([j])
        F1_j = Set{Int}([j])

        for s in 1:nscen
            x_js = x_root[j, s] >= 0.9 ? 1 : 0

            if x_js == 0
                # x_j already 0 in optimal; measure cost of forcing to 1
                g_F1j, _, wu = solve_subproblem_scenario(problem_data, scenarios[s], empty_set, F1_j)
                work_units  += wu
                diff1 = g_F1j - g_root[s]
                if diff1 < -0.001
                    println(log_io, "  [BB] WARNING: lambda1[s=$s, j=$j] is negative ($diff1) — solver numerical issue")
                end
                lambda0[s, j] = 0.0
                lambda1[s, j] = max(0.0, diff1)
            else
                # x_j already 1 in optimal; measure cost of forcing to 0
                g_F0j, _, wu = solve_subproblem_scenario(problem_data, scenarios[s], F0_j, empty_set)
                work_units  += wu
                diff0 = g_F0j - g_root[s]
                if diff0 < -0.001
                    println(log_io, "  [BB] WARNING: lambda0[s=$s, j=$j] is negative ($diff0) — solver numerical issue")
                end
                lambda0[s, j] = max(0.0, diff0)
                lambda1[s, j] = 0.0
            end
        end

        if j % 5 == 0 || j == nloc
            println(log_io, "    variable $j / $nloc done")
        end
    end

    println(log_io, "  [BB] Precomputation complete.")
    return lambda0, lambda1, x_root, g_root, work_units
end

# =============================================================================
# BRANCHING VARIABLE SELECTION
#
# Most-fractional rule: pick j* = argmax_j |x̄_j - floor(x̄_j + 0.5)|
# i.e., the variable whose average value across scenarios is closest to 0.5.
# =============================================================================

function select_branching_variable(
    frac_vars::Vector{Int},
    x_bar::Vector{Float64}
)
    best_j     = frac_vars[1]
    best_score = -Inf
    for j in frac_vars
        score = abs(x_bar[j] - floor(x_bar[j] + 0.5))
        if score > best_score
            best_score = score
            best_j     = j
        end
    end
    return best_j, best_score
end

function select_branching_variable_product(
    frac_vars::Vector{Int},
    x_sk::Matrix{Float64},
    lambda0::Matrix{Float64},
    lambda1::Matrix{Float64},
    nscen::Int
)
    best_j     = frac_vars[1]
    best_score = -Inf
    for j in frac_vars
        e0 = sum(lambda0[s, j] * x_sk[j, s] for s in 1:nscen)
        e1 = sum(lambda1[s, j] * (1.0 - x_sk[j, s]) for s in 1:nscen)
        score = e0 * e1
        if score > best_score
            best_score = score
            best_j     = j
        end
    end
    return best_j, best_score
end

# =============================================================================
# MAIN BRANCH AND BOUND
# =============================================================================

function branch_and_bound(
    problem_data::Dict,
    scenarios::Vector{Vector{Float64}},
    lambda_star::Vector{Float64};
    work_limit::Float64 = 2000.0,
    cache::BBCache = BBCache(),
    log_file::Union{String,Nothing} = nothing,
    branching_strategy::String = "F"
)
    nloc  = problem_data["nloc"]
    nscen = length(scenarios)

    # Open log file (or fall back to stdout if not provided)
    log_io = log_file !== nothing ? open(log_file, "w") : stdout

    try
        println(log_io, "\n" * "="^60)
        println(log_io, "CUSTOM BRANCH AND BOUND — PRICING SUBPROBLEM")
        println(log_io, "  Facilities : $nloc")
        println(log_io, "  Scenarios  : $nscen")
        println(log_io, "="^60)

        # ------------------------------------------------------------------
        # Step 1: Evaluate root node
        # ------------------------------------------------------------------
        total_work_units = 0.0

        root_key = cache_key(Set{Int}(), Set{Int}())
        if haskey(cache, root_key)
            g_root, x_sk_root = cache[root_key]
            U_root = sum(max(0.0, lambda_star[s] - g_root[s]) for s in 1:nscen)
        else
            U_root, x_sk_root, g_root, wu_root = evaluate_node(
                problem_data, scenarios, Set{Int}(), Set{Int}(), lambda_star)
            total_work_units += wu_root
            cache[root_key] = (g_root, x_sk_root)
        end

        println(log_io, "  Root UB = $(round(U_root, digits=6))")

        # ------------------------------------------------------------------
        # Step 2: Warm-start incumbent from scenario 1's root solution
        # ------------------------------------------------------------------
        x_s1  = round.(Int, x_sk_root[:, 1])
        F0_s1 = Set{Int}([j for j in 1:nloc if x_s1[j] == 0])
        F1_s1 = Set{Int}([j for j in 1:nloc if x_s1[j] == 1])
        ws_key = cache_key(F0_s1, F1_s1)
        if haskey(cache, ws_key)
            g_s1_vec, _ = cache[ws_key]
            L_s1 = sum(max(0.0, lambda_star[s] - g_s1_vec[s]) for s in 1:nscen)
        else
            L_s1, _, g_s1_vec, wu_s1 = evaluate_node(
                problem_data, scenarios, F0_s1, F1_s1, lambda_star)
            total_work_units += wu_s1
            cache[ws_key] = (g_s1_vec, x_sk_root)
        end

        L_hat  = L_s1
        x_star = x_s1
        println(log_io, "  Warm-start incumbent (x from scenario 1): L_hat = $(round(L_hat, digits=6))")

        # ------------------------------------------------------------------
        # Precompute sensitivities if using product branching strategy
        # ------------------------------------------------------------------
        lambda0 = zeros(nscen, nloc)
        lambda1 = zeros(nscen, nloc)
        if branching_strategy == "P"
            println(log_io, "  Branching strategy: PRODUCT (sensitivity-based)")
            lambda0, lambda1, _, _, wu_sens = precompute_sensitivities(problem_data, scenarios, log_io)
            total_work_units += wu_sens
        else
            println(log_io, "  Branching strategy: MOST-FRACTIONAL")
        end

        nodes_processed = 0
        cache_hits      = 0
        cache_misses    = 0

        open_nodes = BBNode[BBNode(Set{Int}(), Set{Int}(), U_root)]

        # ------------------------------------------------------------------
        # Step 3: B&B loop
        # ------------------------------------------------------------------
        work_limit_hit = false

        while !isempty(open_nodes)

            # Check work limit
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

            key      = cache_key(node.F0, node.F1)
            is_hit   = haskey(cache, key)
            hit_str  = is_hit ? "HIT" : "MISS"
            if is_hit
                g_s_vec, x_sk = cache[key]
                U_k = sum(max(0.0, lambda_star[s] - g_s_vec[s]) for s in 1:nscen)
                cache_hits += 1
            else
                U_k, x_sk, g_s_vec, wu_node =
                    evaluate_node(problem_data, scenarios, node.F0, node.F1, lambda_star)
                total_work_units += wu_node
                cache[key] = (g_s_vec, x_sk)
                cache_misses += 1
            end

            UB  = isempty(open_nodes) ? U_k : max(U_k, maximum(n.upper_bound for n in open_nodes))
            LB  = L_hat
            gap = UB > 1e-10 ? 100.0 * (UB - LB) / abs(UB) : 0.0

            if U_k <= L_hat
                @printf(log_io, "  Node %4d  |F0|=%2d  |F1|=%2d  cache=%-4s  U_k=%.4f  LB=%.4f  UB=%.4f  gap=%.2f%%  work=%.4f  outcome=FATHOMED  branch_var=∅  open_nodes=%d\n",
                        nodes_processed, length(node.F0), length(node.F1), hit_str, U_k, LB, UB, gap, total_work_units, length(open_nodes))
                continue
            end

            x_bar = vec(sum(x_sk, dims=2)) / nscen

            if all(v -> isapprox(v, 0.0; atol=1e-6) || isapprox(v, 1.0; atol=1e-6), x_bar)
                if U_k > L_hat
                    L_hat  = U_k
                    x_star = round.(Int, x_bar)
                    println(log_io, "  *** New incumbent: L_hat = $(round(L_hat, digits=6))")
                end
                UB  = isempty(open_nodes) ? L_hat : maximum(n.upper_bound for n in open_nodes)
                gap = UB > 1e-10 ? 100.0 * (UB - L_hat) / abs(UB) : 0.0
                @printf(log_io, "  Node %4d  |F0|=%2d  |F1|=%2d  cache=%-4s  U_k=%.4f  LB=%.4f  UB=%.4f  gap=%.2f%%  work=%.4f  outcome=FEASIBLE  branch_var=∅  open_nodes=%d\n",
                        nodes_processed, length(node.F0), length(node.F1), hit_str, U_k, L_hat, UB, gap, total_work_units, length(open_nodes))
                continue
            end

            frac_vars = [j for j in 1:nloc
                         if !(j in node.F0) && !(j in node.F1) &&
                            !isapprox(x_bar[j], 0.0; atol=1e-6) &&
                            !isapprox(x_bar[j], 1.0; atol=1e-6)]

            # if isempty(frac_vars)
            #     # Numerically all variables are 0/1 — treat as feasible
            #     if U_k > L_hat
            #         L_hat  = U_k
            #         x_star = round.(Int, x_bar)
            #         println(log_io, "  *** New incumbent (numeric): L_hat = $(round(L_hat, digits=6))")
            #     end
            #     continue
            # end

            # ------------------------------------------------------------------
            # Rounding heuristic: round x_bar to nearest integer and evaluate
            # as a candidate lower bound
            # ------------------------------------------------------------------
            x_rounded  = Int[floor(Int, x_bar[j] + 0.5) for j in 1:nloc]
            F0_rounded = Set{Int}([j for j in 1:nloc if x_rounded[j] == 0])
            F1_rounded = Set{Int}([j for j in 1:nloc if x_rounded[j] == 1])
            rnd_key    = cache_key(F0_rounded, F1_rounded)
            if haskey(cache, rnd_key)
                g_rnd, _ = cache[rnd_key]
                LB_rnd   = sum(max(0.0, lambda_star[s] - g_rnd[s]) for s in 1:nscen)
            else
                LB_rnd, _, g_rnd, wu_rnd =
                    evaluate_node(problem_data, scenarios, F0_rounded, F1_rounded, lambda_star)
                total_work_units += wu_rnd
                cache[rnd_key] = (g_rnd, zeros(nloc, nscen))
            end
            if LB_rnd > L_hat
                L_hat  = LB_rnd
                x_star = x_rounded
                println(log_io, "  *** New incumbent (rounding heuristic): L_hat = $(round(L_hat, digits=6))")
            end

            if branching_strategy == "P"
                j_star, branch_score = select_branching_variable_product(frac_vars, x_sk, lambda0, lambda1, nscen)
                score_label = "product_score"
            else
                j_star, branch_score = select_branching_variable(frac_vars, x_bar)
                score_label = "frac_score"
            end

            F0_down = union(node.F0, Set([j_star]))
            F1_up   = union(node.F1, Set([j_star]))

            push!(open_nodes, BBNode(F0_down, node.F1, U_k))
            push!(open_nodes, BBNode(node.F0, F1_up,   U_k))

            UB  = maximum(n.upper_bound for n in open_nodes)
            gap = UB > 1e-10 ? 100.0 * (UB - L_hat) / abs(UB) : 0.0
            @printf(log_io, "  Node %4d  |F0|=%2d  |F1|=%2d  cache=%-4s  U_k=%.4f  LB=%.4f  UB=%.4f  gap=%.2f%%  work=%.4f  outcome=BRANCHED  branch_var=%d  %s=%.4f  nfrac=%d  open_nodes=%d\n",
                    nodes_processed, length(node.F0), length(node.F1), hit_str, U_k, L_hat, UB, gap, total_work_units, j_star, score_label, branch_score, length(frac_vars), length(open_nodes))
        end

        # Final UB and gap
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
        println(log_io, "Cached nodes       : $(length(cache))")
        println(log_io, "Cache hits         : $cache_hits")
        println(log_io, "Cache misses       : $cache_misses")
        hit_rate = (cache_hits + cache_misses) > 0 ? 100.0 * cache_hits / (cache_hits + cache_misses) : 0.0
        println(log_io, "Cache hit rate     : $(round(hit_rate, digits=1))%")
        println(log_io, "="^60)

        hit_rate_final = (cache_hits + cache_misses) > 0 ? 100.0 * cache_hits / (cache_hits + cache_misses) : 0.0
        return x_star, L_hat, UB_final, gap_final, cache, total_work_units, nodes_processed, hit_rate_final

    finally
        if log_file !== nothing
            close(log_io)
        end
    end
end
