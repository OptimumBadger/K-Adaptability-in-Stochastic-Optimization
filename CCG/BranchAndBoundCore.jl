# BranchAndBoundCore.jl
# Generic B&B framework for solving the AF pricing subproblem:
#
#   max_{x in X}  sum_{s in [S]} max{0, lambda_s* - f(x, omega_s)}
#
# Problem-specific logic (subproblem models, objective) is injected via
# function arguments: build_persistent_models_fn, solve_subproblem_persistent_fn,
# solve_subproblem_scenario_fn.
#
# Included by BranchAndBoundFL.jl, BranchAndBoundKnapsack.jl, etc.

using JuMP, Gurobi
using MathOptInterface
using Printf
const MOI = MathOptInterface

# =============================================================================
# DIRECT-GUROBI HOT-PATH CONSTANTS  (shared by all BranchAndBound* files)
# =============================================================================
# Persistent models use direct_model(Gurobi.Optimizer()) so backend(model) IS
# the Gurobi.Optimizer and JuMP.index(x) returns its native variable index.
# Hot-path solve functions bypass JuMP wrappers and call MOI.set/get on the
# optimizer with Gurobi-native attributes.

const _GRB_LB      = Gurobi.VariableAttribute("LB")
const _GRB_UB      = Gurobi.VariableAttribute("UB")
const _GRB_STATUS  = Gurobi.ModelAttribute("Status")
const _GRB_OBJVAL  = Gurobi.ModelAttribute("ObjVal")
const _GRB_WORK    = Gurobi.ModelAttribute("Work")
const _GRB_RUNTIME = Gurobi.ModelAttribute("Runtime")
const _MOI_VARPRIM = MOI.VariablePrimal()
const _GRB_OPTIMAL = 2   # GRB_OPTIMAL

# Slack on the g_s < lambda_s* test that defines the active scenario set under
# branching_strategy == "FA". A scenario with g_s == lambda_s* contributes
# exactly 0 to U_k, so it is excluded; the tolerance keeps float noise from
# pulling such a scenario back into the average.
const LAMBDA_ACTIVE_TOL = 1e-9

# =============================================================================
# PER-NODE TRACE LOGGING  (BB_LOG)
# =============================================================================
# Set BB_LOG=1 to make the pricing B&B write its per-node trace — one line per
# node, including `nactive` (scenarios entering the x_bar average) and `ntied`
# (fractional variables sharing the winning branch score). Off by default.
#
#   BB_LOG=1 julia -t 8 CCG/CG_AF_Knapsack.jl KS <inst> 3 300 <pool> 12000 BB
#
# One file per CG iteration:  <BB_LOG_DIR>/bb_<tag>_iter<k>.log
# BB_LOG_DIR defaults to CCG/logs/bb; the tag comes from problem_data["bb_log_tag"]
# (set by the CG_AF_* drivers) or BB_LOG_TAG.
#
# Intended for work-unit mode. Work units come from Gurobi's Work attribute, so
# this host-side I/O never touches the budget, and work mode sets no wall-clock
# limit — the trace can only make a run slower, never shorter. In TIME mode the
# I/O would eat the budget and silently shrink the tree, so don't combine them.
# Expect ~200 bytes/node: a 10^6-node tree writes a ~200 MB file.
_bb_log_on() = get(ENV, "BB_LOG", "0") == "1"

function _bb_log_file(problem_data::Dict, iteration)
    _bb_log_on() || return nothing
    dir = get(ENV, "BB_LOG_DIR", joinpath(@__DIR__, "logs", "bb"))
    mkpath(dir)
    tag = get(ENV, "BB_LOG_TAG", get(problem_data, "bb_log_tag", "run"))
    return joinpath(dir, "bb_$(tag)_iter$(iteration === nothing ? 0 : iteration).log")
end

@inline function _read_x_into_buf!(grb, vi::Vector{MOI.VariableIndex}, buf::Vector{Float64})
    @inbounds for j in eachindex(vi)
        buf[j] = MOI.get(grb, _MOI_VARPRIM, vi[j])
    end
    return nothing
end

# Apply a UInt64 bitmask to Gurobi variable bounds: for each set bit j,
# call MOI.set(grb, attr, vi[j], val). Used to fix/reset F0/F1 bounds
# without materialising a Vector{Int}.
@inline function _apply_mask_bounds!(grb, attr, vi::Vector{MOI.VariableIndex},
                                     mask::UInt64, val::Float64)
    m = mask
    @inbounds while m != UInt64(0)
        j = trailing_zeros(m) + 1
        MOI.set(grb, attr, vi[j], val)
        m &= m - UInt64(1)
    end
    return nothing
end

# =============================================================================
# PROFILING TIMERS  (nanoseconds; minimal 9-bucket pass)
# =============================================================================
# Commented out — re-enable by removing the #= ... =# block-comment wrapper
# (and the corresponding `# PROF:` line-comments scattered throughout this
# file and BranchAndBoundKnapsack.jl) when you want per-bucket profiling again.
#
#=
const _T_SOLVE_SET_F0F1   = Threads.Atomic{UInt64}(0)
const _T_SOLVE_OPTIMIZE   = Threads.Atomic{UInt64}(0)
const _T_SOLVE_READ_META  = Threads.Atomic{UInt64}(0)
const _T_SOLVE_READ_X     = Threads.Atomic{UInt64}(0)
const _T_SOLVE_RESET_F0F1 = Threads.Atomic{UInt64}(0)
const _T_EVAL_XSK_ALLOC   = Ref{UInt64}(0)
const _T_EVAL_MAIN        = Ref{UInt64}(0)
const _T_EVAL_ROUNDING    = Ref{UInt64}(0)
# Tree-management buckets (main loop)
const _T_HEAP_OPS         = Ref{UInt64}(0)
const _T_CACHE_LOOKUP     = Ref{UInt64}(0)
const _T_CACHE_STORE      = Ref{UInt64}(0)
const _T_XBAR_INTCHECK    = Ref{UInt64}(0)
const _T_FRAC_BUILD       = Ref{UInt64}(0)
const _T_BRANCH_SELECT    = Ref{UInt64}(0)
# One-shot setup buckets
const _T_ROOT             = Ref{UInt64}(0)
const _T_WARMSTART        = Ref{UInt64}(0)
const _T_TOTAL            = Ref{UInt64}(0)

# Cumulative accumulators across ALL branch_and_bound calls in this Julia
# session. Never reset (except on module reload). Read the value printed in
# the LAST timing block to get the total across all CG iterations.
const _T_EVAL_MAIN_CUM    = Ref{UInt64}(0)
const _T_TOTAL_CUM        = Ref{UInt64}(0)
const _BB_CALLS_CUM       = Ref{Int}(0)

function _reset_bb_timers!()
    _T_SOLVE_SET_F0F1[]   = 0
    _T_SOLVE_OPTIMIZE[]   = 0
    _T_SOLVE_READ_META[]  = 0
    _T_SOLVE_READ_X[]     = 0
    _T_SOLVE_RESET_F0F1[] = 0
    _T_EVAL_XSK_ALLOC[]   = 0
    _T_EVAL_MAIN[]        = 0
    _T_EVAL_ROUNDING[]    = 0
    _T_HEAP_OPS[]         = 0
    _T_CACHE_LOOKUP[]     = 0
    _T_CACHE_STORE[]      = 0
    _T_XBAR_INTCHECK[]    = 0
    _T_FRAC_BUILD[]       = 0
    _T_BRANCH_SELECT[]    = 0
    _T_ROOT[]             = 0
    _T_WARMSTART[]        = 0
    _T_TOTAL[]            = 0
    return nothing
end

function _print_bb_timers(io::IO)
    total_ns = _T_TOTAL[]
    total_ms = total_ns / 1_000_000
    pct(x)   = total_ns > 0 ? 100.0 * x / total_ns : 0.0

    # Section 1: top-level setup (one-shots) — disjoint from main-loop buckets
    setup_rows = [
        ("root"        , _T_ROOT[]),
        ("warmstart"   , _T_WARMSTART[]),
    ]
    # Section 2: main-loop per-node buckets (disjoint, sum ≈ main-loop wall time)
    loop_rows = [
        ("heap_ops"      , _T_HEAP_OPS[]),
        ("cache_lookup"  , _T_CACHE_LOOKUP[]),
        ("eval_main"     , _T_EVAL_MAIN[]),
        ("cache_store"   , _T_CACHE_STORE[]),
        ("xbar_intcheck" , _T_XBAR_INTCHECK[]),
        ("frac_build"    , _T_FRAC_BUILD[]),
        ("eval_rounding" , _T_EVAL_ROUNDING[]),
        ("branch_select" , _T_BRANCH_SELECT[]),
    ]
    # Section 3: solve-internal buckets — NESTED inside any eval_* / root / warmstart
    nested_rows = [
        ("solve_optimize"   , _T_SOLVE_OPTIMIZE[]),
        ("solve_set_F0F1"   , _T_SOLVE_SET_F0F1[]),
        ("solve_reset_F0F1" , _T_SOLVE_RESET_F0F1[]),
        ("solve_read_x"     , _T_SOLVE_READ_X[]),
        ("solve_read_meta"  , _T_SOLVE_READ_META[]),
        ("eval_xsk_alloc"   , _T_EVAL_XSK_ALLOC[]),
    ]

    println(io, "=== BB TIMING BREAKDOWN  (total = $(round(total_ms, digits=1)) ms) ===")
    @printf(io, "  %-20s %10s  %7s\n", "bucket", "ms", "%total")

    println(io, "  -- setup (disjoint) ----------------------------------")
    sort!(setup_rows; by = r -> r[2], rev = true)
    for (name, ns) in setup_rows
        @printf(io, "  %-20s %10.2f  %6.2f%%\n", name, ns / 1_000_000, pct(ns))
    end

    println(io, "  -- main loop (disjoint) ------------------------------")
    sort!(loop_rows; by = r -> r[2], rev = true)
    for (name, ns) in loop_rows
        @printf(io, "  %-20s %10.2f  %6.2f%%\n", name, ns / 1_000_000, pct(ns))
    end

    accounted_disjoint = sum(r[2] for r in setup_rows) + sum(r[2] for r in loop_rows)
    @printf(io, "  %-20s %10.2f  %6.2f%%  (setup + loop)\n",
            "accounted", accounted_disjoint / 1_000_000, pct(accounted_disjoint))
    @printf(io, "  %-20s %10.2f  %6.2f%%  (residual)\n",
            "unaccounted", (total_ns - accounted_disjoint) / 1_000_000,
            pct(total_ns - accounted_disjoint))

    println(io, "  -- nested inside eval/root/warmstart -----------------")
    sort!(nested_rows; by = r -> r[2], rev = true)
    for (name, ns) in nested_rows
        @printf(io, "  %-20s %10.2f  %6.2f%%\n", name, ns / 1_000_000, pct(ns))
    end

    # Cumulative across all BB calls in this session — read the LAST block
    # printed to get the totals across all CG iterations.
    eval_main_cum_ms = _T_EVAL_MAIN_CUM[] / 1_000_000
    total_cum_ms     = _T_TOTAL_CUM[]     / 1_000_000
    eval_share = total_cum_ms > 0 ? 100.0 * eval_main_cum_ms / total_cum_ms : 0.0
    println(io, "  -- cumulative across $(_BB_CALLS_CUM[]) BB calls (session-wide) ---")
    @printf(io, "  %-20s %10.2f ms\n", "eval_main_total", eval_main_cum_ms)
    @printf(io, "  %-20s %10.2f ms  (eval_main = %.1f%% of BB total)\n",
            "BB_wall_total", total_cum_ms, eval_share)
    println(io, "="^60)
    return nothing
end
=#

# =============================================================================
# NODE STRUCTURE AND CACHE
# =============================================================================

struct BBNode
    F0::UInt64             # bitmask: bit j (= 1<<(j-1)) set ⇒ variable j fixed to 0
    F1::UInt64             # bitmask: bit j (= 1<<(j-1)) set ⇒ variable j fixed to 1
    upper_bound::Float64   # parent U_k used for best-first ordering
end

const BBCacheKey   = Tuple{UInt64, UInt64}
const BBCacheValue = Tuple{Vector{Float64}, Matrix{Float64}}   # (g_s_vec, x_sk)
const BBCache      = Dict{BBCacheKey, BBCacheValue}

@inline cache_key(F0::UInt64, F1::UInt64) = (F0, F1)

# Bitmask helpers — variable j ↔ bit (j-1). Caller must ensure 1 ≤ j ≤ 64.
@inline _bit(j::Int)             = UInt64(1) << (j - 1)
@inline _has_bit(m::UInt64, j::Int) = (m & _bit(j)) != UInt64(0)
@inline _set_bit(m::UInt64, j::Int) = m | _bit(j)
@inline _popcount(m::UInt64)        = count_ones(m)

# =============================================================================
# MAX-HEAP HELPERS  (BBNode ordered by upper_bound, max on top)
# =============================================================================

@inline function heap_push!(heap::Vector{BBNode}, node::BBNode)
    push!(heap, node)
    i = length(heap)
    @inbounds while i > 1
        p = i >> 1
        if heap[p].upper_bound < heap[i].upper_bound
            heap[p], heap[i] = heap[i], heap[p]
            i = p
        else
            break
        end
    end
end

@inline function heap_pop!(heap::Vector{BBNode})
    top = heap[1]
    last_node = pop!(heap)
    n = length(heap)
    if n > 0
        @inbounds heap[1] = last_node
        i = 1
        @inbounds while true
            l = 2i
            r = 2i + 1
            best = i
            if l <= n && heap[l].upper_bound > heap[best].upper_bound
                best = l
            end
            if r <= n && heap[r].upper_bound > heap[best].upper_bound
                best = r
            end
            best == i && break
            heap[i], heap[best] = heap[best], heap[i]
            i = best
        end
    end
    return top
end

@inline heap_peek(heap::Vector{BBNode}) = heap[1]

# =============================================================================
# BRANCHING VARIABLE SELECTION
# =============================================================================
# frac_vars is a UInt64 bitmask; we iterate set bits via the "clear lowest set
# bit" idiom: trailing_zeros(m) gives the lowest set bit position (0-indexed);
# m & (m-1) clears it.

function select_branching_variable(
    frac_vars::UInt64,
    x_bar::Vector{Float64}
)
    @assert frac_vars != UInt64(0) "select_branching_variable called with no fractional vars"
    best_j::Int = trailing_zeros(frac_vars) + 1
    best_score  = abs(x_bar[best_j] - floor(x_bar[best_j] + 0.5))
    m = frac_vars & (frac_vars - UInt64(1))
    @inbounds while m != UInt64(0)
        j = trailing_zeros(m) + 1
        score = abs(x_bar[j] - floor(x_bar[j] + 0.5))
        if score > best_score
            best_score = score
            best_j     = j
        end
        m &= m - UInt64(1)
    end
    return best_j, best_score
end

function select_branching_variable_product(
    frac_vars::UInt64,
    x_sk::Matrix{Float64},
    lambda0::Matrix{Float64},
    lambda1::Matrix{Float64},
    nscen::Int
)
    @assert frac_vars != UInt64(0) "select_branching_variable_product called with no fractional vars"
    best_j::Int = trailing_zeros(frac_vars) + 1
    e0 = sum(lambda0[s, best_j] * x_sk[best_j, s] for s in 1:nscen)
    e1 = sum(lambda1[s, best_j] * (1.0 - x_sk[best_j, s]) for s in 1:nscen)
    best_score = e0 * e1
    m = frac_vars & (frac_vars - UInt64(1))
    @inbounds while m != UInt64(0)
        j = trailing_zeros(m) + 1
        e0 = sum(lambda0[s, j] * x_sk[j, s] for s in 1:nscen)
        e1 = sum(lambda1[s, j] * (1.0 - x_sk[j, s]) for s in 1:nscen)
        score = e0 * e1
        if score > best_score
            best_score = score
            best_j     = j
        end
        m &= m - UInt64(1)
    end
    return best_j, best_score
end

# =============================================================================
# NODE EVALUATION (generic — injected solve function)
# =============================================================================

function evaluate_node(
    problem_data::Dict,
    scenarios::Vector{Vector{Float64}},
    F0::UInt64,
    F1::UInt64,
    lambda_star::Vector{Float64},
    n_vars::Int,
    solve_subproblem_scenario_fn::Function
)
    nscen      = length(scenarios)
    x_sk       = zeros(n_vars, nscen)
    g_s_vec    = zeros(nscen)
    work_units = 0.0
    runtime    = 0.0

    for s in 1:nscen
        g_s, x_s, wu, rt = solve_subproblem_scenario_fn(problem_data, scenarios[s], F0, F1)
        g_s_vec[s]   = g_s
        x_sk[:, s]   = x_s
        work_units  += wu
        runtime     += rt
    end

    U_k = sum(max(0.0, lambda_star[s] - g_s_vec[s]) for s in 1:nscen)
    return U_k, x_sk, g_s_vec, work_units, runtime
end

function evaluate_node_persistent(
    pmodels,
    F0::UInt64,
    F1::UInt64,
    lambda_star::Vector{Float64},
    n_vars::Int,
    solve_subproblem_persistent_fn::Function
)
    nscen      = length(pmodels)
    # PROF: _t_alloc = time_ns()
    x_sk       = zeros(n_vars, nscen)
    g_s_vec    = zeros(nscen)
    wu_vec     = zeros(nscen)
    rt_vec     = zeros(nscen)
    # PROF: _T_EVAL_XSK_ALLOC[] += time_ns() - _t_alloc

    # Parallel over scenarios. Safe because each pmodels[s] has its own Gurobi.Env
    # (built in build_persistent_models_*) and its own x_buf. Each thread touches
    # exactly one pm. Falls back to serial if julia was launched with -t 1.
    # :dynamic — threads pull scenarios from a shared queue rather than getting
    # an equal-sized chunk upfront. Wins when scenario solve times are uneven
    # (which they are: Gurobi B&B time on small knapsacks varies wildly per scen).
    Threads.@threads :dynamic for s in 1:nscen
        g_s, x_s, wu, rt = solve_subproblem_persistent_fn(pmodels[s], F0, F1)
        g_s_vec[s]   = g_s
        @inbounds for j in 1:n_vars
            x_sk[j, s] = x_s[j]
        end
        wu_vec[s]    = wu
        rt_vec[s]    = rt
    end

    work_units = sum(wu_vec)
    runtime    = sum(rt_vec)
    U_k = sum(max(0.0, lambda_star[s] - g_s_vec[s]) for s in 1:nscen)
    return U_k, x_sk, g_s_vec, work_units, runtime
end

# =============================================================================
# SENSITIVITY PRE-COMPUTATION  (product branching strategy)
# =============================================================================

function precompute_sensitivities(
    problem_data::Dict,
    scenarios::Vector{Vector{Float64}},
    n_vars::Int,
    solve_subproblem_scenario_fn::Function,
    log_io::IO,
    verbose::Bool = false
)
    nscen      = length(scenarios)
    empty_mask = UInt64(0)

    verbose && println(log_io, "  [BB] Precomputing sensitivities...")
    verbose && println(log_io, "  [BB] Step 1/2 — root solves ($nscen scenarios)...")

    x_root     = zeros(n_vars, nscen)
    g_root     = zeros(nscen)
    work_units = 0.0
    runtime    = 0.0

    for s in 1:nscen
        g_s, x_s, wu, rt = solve_subproblem_scenario_fn(problem_data, scenarios[s], empty_mask, empty_mask)
        g_root[s]    = g_s
        x_root[:, s] = x_s
        work_units  += wu
        runtime     += rt
        verbose && println(log_io, "    scenario $s: g_s(root) = $g_s")
    end

    verbose && println(log_io, "  [BB] Step 2/2 — sensitivity solves ($n_vars variables × $nscen scenarios)...")

    lambda0 = zeros(nscen, n_vars)
    lambda1 = zeros(nscen, n_vars)

    for j in 1:n_vars
        bit_j = _bit(j)

        for s in 1:nscen
            x_js = x_root[j, s] >= 0.9 ? 1 : 0

            if x_js == 0
                g_F1j, _, wu, rt = solve_subproblem_scenario_fn(problem_data, scenarios[s], empty_mask, bit_j)
                work_units  += wu
                runtime     += rt
                diff1 = g_F1j - g_root[s]
                if diff1 < -0.001
                    println(log_io, "  [BB] WARNING: lambda1[s=$s, j=$j] negative ($diff1)")
                end
                lambda0[s, j] = 0.0
                lambda1[s, j] = max(0.0, diff1)
            else
                g_F0j, _, wu, rt = solve_subproblem_scenario_fn(problem_data, scenarios[s], bit_j, empty_mask)
                work_units  += wu
                runtime     += rt
                diff0 = g_F0j - g_root[s]
                if diff0 < -0.001
                    println(log_io, "  [BB] WARNING: lambda0[s=$s, j=$j] negative ($diff0)")
                end
                lambda0[s, j] = max(0.0, diff0)
                lambda1[s, j] = 0.0
            end
        end

        verbose && (j % 5 == 0 || j == n_vars) && println(log_io, "    variable $j / $n_vars done")
    end

    verbose && println(log_io, "  [BB] Precomputation complete.")
    return lambda0, lambda1, x_root, g_root, work_units, runtime
end

# =============================================================================
# MAIN BRANCH AND BOUND  (problem-agnostic)
# =============================================================================

function branch_and_bound(
    problem_data::Dict,
    scenarios::Vector{Vector{Float64}},
    lambda_star::Vector{Float64},
    n_vars::Int,
    build_persistent_models_fn::Function,
    solve_subproblem_persistent_fn::Function,
    solve_subproblem_scenario_fn::Function;
    work_limit::Float64       = 2000.0,
    time_limit::Float64       = Inf,
    cache::BBCache            = BBCache(),
    log_file::Union{String,Nothing} = nothing,
    branching_strategy::String = "F",
    verbose::Bool              = false
)
    # PROF: _reset_bb_timers!()
    # PROF: _t_total0 = time_ns()
    @assert n_vars <= 64 "Bitmask BB supports n_vars ≤ 64. Got n_vars=$n_vars. Migrate BBNode F0/F1 to a multi-word bitset (NTuple{K, UInt64}) for larger n."
    @assert branching_strategy in ("F", "FA", "P") "Unknown branching_strategy \"$branching_strategy\". Expected \"F\" (most-fractional, feasible-scenario average), \"FA\" (most-fractional, lambda*-active average) or \"P\" (product/sensitivity)."
    nscen  = length(scenarios)
    log_io = log_file !== nothing ? open(log_file, "w") : stdout

    try
        println(log_io, "\n" * "="^60)
        println(log_io, "CUSTOM BRANCH AND BOUND — PRICING SUBPROBLEM")
        println(log_io, "  Variables  : $n_vars")
        println(log_io, "  Scenarios  : $nscen")
        println(log_io, "="^60)

        total_work_units = 0.0
        total_runtime    = 0.0
        pmodels = get!(problem_data, "bb_pmodels") do
            build_persistent_models_fn(problem_data, scenarios)
        end

        # ── Root node ────────────────────────────────────────────────────────
        # PROF: _t_root0 = time_ns()
        empty_mask = UInt64(0)
        root_key  = cache_key(empty_mask, empty_mask)
        local x_sk_root::Matrix{Float64}
        if haskey(cache, root_key)
            g_root, x_sk_root = cache[root_key]
            U_root = sum(max(0.0, lambda_star[s] - g_root[s]) for s in 1:nscen)
        else
            U_root, x_sk_root, g_root, wu_root, rt_root = evaluate_node_persistent(
                pmodels, empty_mask, empty_mask, lambda_star, n_vars,
                solve_subproblem_persistent_fn)
            total_work_units += wu_root
            total_runtime    += rt_root
            cache[root_key] = (g_root, x_sk_root)
        end
        # PROF: _T_ROOT[] += time_ns() - _t_root0
        println(log_io, "  Root UB = $(round(U_root, digits=6))")

        # ── Warm-start incumbent from scenario 1's root solution ──────────────
        # PROF: _t_ws0 = time_ns()
        x_s1 = round.(Int, x_sk_root[:, 1])
        F0_s1 = UInt64(0); F1_s1 = UInt64(0)
        for j in 1:n_vars
            if x_s1[j] == 0
                F0_s1 = _set_bit(F0_s1, j)
            else
                F1_s1 = _set_bit(F1_s1, j)
            end
        end
        ws_key = cache_key(F0_s1, F1_s1)
        if haskey(cache, ws_key)
            g_s1_vec, _ = cache[ws_key]
            L_s1 = sum(max(0.0, lambda_star[s] - g_s1_vec[s]) for s in 1:nscen)
        else
            L_s1, x_sk_ws, g_s1_vec, wu_s1, rt_s1 = evaluate_node_persistent(
                pmodels, F0_s1, F1_s1, lambda_star, n_vars,
                solve_subproblem_persistent_fn)
            total_work_units += wu_s1
            total_runtime    += rt_s1
            cache[ws_key] = (g_s1_vec, x_sk_ws)
        end
        L_hat  = L_s1
        x_star = x_s1
        # PROF: _T_WARMSTART[] += time_ns() - _t_ws0
        println(log_io, "  Warm-start incumbent (x from scenario 1): L_hat = $(round(L_hat, digits=6))")

        # ── Sensitivity pre-computation (product branching only) ──────────────
        lambda0 = zeros(nscen, n_vars)
        lambda1 = zeros(nscen, n_vars)
        if branching_strategy == "P"
            println(log_io, "  Branching strategy: PRODUCT (sensitivity-based)")
            lambda0, lambda1, _, _, wu_sens, rt_sens = precompute_sensitivities(
                problem_data, scenarios, n_vars, solve_subproblem_scenario_fn, log_io, verbose)
            total_work_units += wu_sens
            total_runtime    += rt_sens
        elseif branching_strategy == "FA"
            println(log_io, "  Branching strategy: MOST-FRACTIONAL (active set = scenarios with g_s < lambda_s*)")
        else
            println(log_io, "  Branching strategy: MOST-FRACTIONAL (active set = feasible scenarios)")
        end

        nodes_processed = 0
        cache_hits      = 0
        cache_misses    = 0
        open_nodes      = BBNode[]
        heap_push!(open_nodes, BBNode(UInt64(0), UInt64(0), U_root))
        work_limit_hit  = false
        bb_start_time   = time()
        x_bar           = Vector{Float64}(undef, n_vars)
        x_rounded       = Vector{Int}(undef, n_vars)

        # ── Branching diagnostics ────────────────────────────────────────────
        # Accumulated over nodes that actually BRANCH (fathomed and integral
        # nodes make no branching choice, so including them would dilute the
        # rates). Raw counters rather than percentages, so the caller can sum
        # across CG iterations and divide once for an exact run-level figure.
        #   branch_decisions : nodes that chose a branching variable
        #   tied_decisions   : of those, where >=2 variables shared the winning
        #                      score — the rule did not pick, lowest index did
        #   active_sum       : sum of n_active, for the mean
        #   min_active       : smallest active set seen at a branch decision
        # Cost is a <=64-iteration bit loop against a node that runs S Gurobi
        # solves, so this is always-on rather than gated behind logging.
        branch_decisions = 0
        tied_decisions   = 0
        active_sum       = 0
        min_active       = typemax(Int)
        # Which scenarios enter the x_bar average — see the comment at its use site.
        use_lambda_active = (branching_strategy == "FA")

        # ── B&B loop ──────────────────────────────────────────────────────────
        while !isempty(open_nodes)

            if total_work_units >= work_limit
                work_limit_hit = true
                break
            end
            # Time-based termination (only active when caller passes a finite time_limit).
            if isfinite(time_limit) && (time() - bb_start_time) >= time_limit
                work_limit_hit = true   # same downstream path — CG core will label as TIME_LIMIT via its own flag
                break
            end

            # Best-first (O(log N) heap pop)
            # PROF: _t_h = time_ns()
            node = heap_pop!(open_nodes)
            # PROF: _T_HEAP_OPS[] += time_ns() - _t_h

            if node.upper_bound <= L_hat
                continue
            end

            nodes_processed += 1
            # PROF: _t_cl = time_ns()
            key    = cache_key(node.F0, node.F1)
            is_hit = haskey(cache, key)
            # PROF: _T_CACHE_LOOKUP[] += time_ns() - _t_cl
            hit_str = is_hit ? "HIT" : "MISS"
            local g_s_vec::Vector{Float64}
            x_sk_or_nothing::Union{Matrix{Float64}, Nothing} = nothing

            if is_hit
                # PROF: _t_cl2 = time_ns()
                g_s_vec, xsk_cached = cache[key]
                # PROF: _T_CACHE_LOOKUP[] += time_ns() - _t_cl2
                U_k = sum(max(0.0, lambda_star[s] - g_s_vec[s]) for s in 1:nscen)
                x_sk_or_nothing = xsk_cached
                cache_hits += 1
            else
                # PROF: _t_em = time_ns()
                U_k, xsk_new, g_s_vec, wu_node, rt_node = evaluate_node_persistent(
                    pmodels, node.F0, node.F1, lambda_star, n_vars,
                    solve_subproblem_persistent_fn)
                # PROF: _T_EVAL_MAIN[] += time_ns() - _t_em
                total_work_units += wu_node
                total_runtime    += rt_node
                # PROF: _t_cs = time_ns()
                cache[key] = (g_s_vec, xsk_new)
                # PROF: _T_CACHE_STORE[] += time_ns() - _t_cs
                x_sk_or_nothing = xsk_new
                cache_misses += 1
            end

            UB  = isempty(open_nodes) ? U_k : max(U_k, heap_peek(open_nodes).upper_bound)
            gap = UB > 1e-10 ? 100.0 * (UB - L_hat) / abs(UB) : 0.0

            if U_k <= L_hat
                verbose && @printf(log_io, "  Node %4d  |F0|=%2d  |F1|=%2d  cache=%-4s  U_k=%.4f  LB=%.4f  UB=%.4f  gap=%.2f%%  work=%.4f  outcome=FATHOMED  branch_var=∅  open_nodes=%d\n",
                        nodes_processed, _popcount(node.F0), _popcount(node.F1), hit_str, U_k, L_hat, UB, gap, total_work_units, length(open_nodes))
                continue
            end

            # ── Compute x_bar (Float64). Fast path for full assignments. ─────
            # PROF: _t_xb = time_ns()
            n_fixed = _popcount(node.F0) + _popcount(node.F1)
            is_integer::Bool = false
            # Scenarios entering the x_bar average at this node. Declared here so
            # it is still in scope for the trace line at the branch point below.
            # x_bar[j] is a multiple of 1/n_active, so the most-fractional score
            # takes only floor(n_active/2) distinct non-zero values — this is the
            # quantity that governs how often branch scores tie.
            n_active::Int = 0
            if n_fixed == n_vars
                # All variables determined by F0/F1 — no x_sk needed
                fill!(x_bar, 0.0)
                _m_f1 = node.F1
                @inbounds while _m_f1 != UInt64(0)
                    j = trailing_zeros(_m_f1) + 1
                    x_bar[j] = 1.0
                    _m_f1 &= _m_f1 - UInt64(1)
                end
                is_integer = true
            else
                # Need x_sk — always available from evaluate or cache
                x_sk = x_sk_or_nothing::Matrix{Float64}
                # Average over ACTIVE scenarios (manual loop, no allocations).
                #
                # Two definitions of "active", selected by branching_strategy:
                #
                #   "F"/"P" : g_s finite, i.e. scenario s is feasible at this node.
                #
                #   "FA"    : g_s < lambda_star[s], i.e. scenario s contributes a
                #             STRICTLY POSITIVE term to U_k = sum_s max(0, lam_s - g_s).
                #             Scenarios with g_s >= lam_s contribute exactly 0 to the
                #             node bound, so their optimal x carries no information
                #             about what makes this node good — averaging them in only
                #             blurs x_bar. Infeasible scenarios (g_s = Inf) are still
                #             excluded automatically, so "FA" refines "F".
                #
                #             The integrality shortcut below stays EXACT under "FA":
                #             if x_bar is integral then every contributing scenario
                #             attains its optimum at x_star, so c_s(x_star) = g_s for
                #             those; and for every excluded scenario c_s(x_star) >= g_s
                #             >= lam_s (g_s is a MIN over the node region), so its term
                #             is 0. Hence the true value of x_star equals U_k exactly.
                fill!(x_bar, 0.0)
                @inbounds for s in 1:nscen
                    if use_lambda_active
                        g_s_vec[s] >= lambda_star[s] - LAMBDA_ACTIVE_TOL && continue
                    else
                        isinf(g_s_vec[s]) && continue
                    end
                    n_active += 1
                    for j in 1:n_vars
                        x_bar[j] += x_sk[j, s]
                    end
                end
                if n_active == 0
                    # Every scenario contributes (essentially) nothing: U_k ~ 0, so
                    # nothing in this subtree can beat L_hat >= 0. Normally the
                    # U_k <= L_hat fathom above already caught this; this guard covers
                    # the LAMBDA_ACTIVE_TOL boundary, where U_k can be a hair above 0.
                    verbose && @printf(log_io, "  Node %4d  |F0|=%2d  |F1|=%2d  cache=%-4s  U_k=%.4f  LB=%.4f  UB=%.4f  gap=%.2f%%  work=%.4f  outcome=FATHOMED_EMPTY_ACTIVE  branch_var=∅  open_nodes=%d\n",
                            nodes_processed, _popcount(node.F0), _popcount(node.F1), hit_str, U_k, L_hat, UB, gap, total_work_units, length(open_nodes))
                    continue
                end
                inv_n = 1.0 / n_active
                @inbounds for j in 1:n_vars
                    x_bar[j] *= inv_n
                end
                # Integer check — manual, no isapprox overhead
                is_integer = true
                @inbounds for j in 1:n_vars
                    v = x_bar[j]
                    if v > 1e-6 && v < 1.0 - 1e-6
                        is_integer = false
                        break
                    end
                end
            end
            # PROF: _T_XBAR_INTCHECK[] += time_ns() - _t_xb

            # Integer node — update incumbent
            if is_integer
                if U_k > L_hat
                    L_hat  = U_k
                    x_star = round.(Int, x_bar)
                    println(log_io, "  *** New incumbent: L_hat = $(round(L_hat, digits=6))")
                end
                UB  = isempty(open_nodes) ? L_hat : heap_peek(open_nodes).upper_bound
                gap = UB > 1e-10 ? 100.0 * (UB - L_hat) / abs(UB) : 0.0
                # nactive is logged here too: this is where the active scenarios
                # agreed on a single x, so it measures the "fewer scenarios agree
                # sooner" effect. nactive=0 marks the all-fixed fast path, where
                # x_bar comes straight off F1 and no averaging happened.
                verbose && @printf(log_io, "  Node %4d  |F0|=%2d  |F1|=%2d  cache=%-4s  U_k=%.4f  LB=%.4f  UB=%.4f  gap=%.2f%%  work=%.4f  outcome=FEASIBLE  branch_var=∅  nactive=%d  open_nodes=%d\n",
                        nodes_processed, _popcount(node.F0), _popcount(node.F1), hit_str, U_k, L_hat, UB, gap, total_work_units, n_active, length(open_nodes))
                continue
            end

            x_sk = x_sk_or_nothing::Matrix{Float64}

            # Build fractional candidates and rounding key in one pass.
            # All three are UInt64 bitmasks (bit j ↔ variable j).
            # PROF: _t_fb = time_ns()
            frac_vars::UInt64  = UInt64(0)
            F0_rounded::UInt64 = UInt64(0)
            F1_rounded::UInt64 = UInt64(0)
            fixed_mask = node.F0 | node.F1
            @inbounds for j in 1:n_vars
                bit_j = _bit(j)
                v = x_bar[j]
                if v < 0.5
                    x_rounded[j] = 0
                    F0_rounded |= bit_j
                else
                    x_rounded[j] = 1
                    F1_rounded |= bit_j
                end
                # frac_vars: not fixed AND not integer-valued
                if v > 1e-6 && v < 1.0 - 1e-6 && (fixed_mask & bit_j) == UInt64(0)
                    frac_vars |= bit_j
                end
            end
            # PROF: _T_FRAC_BUILD[] += time_ns() - _t_fb

            # Rounding heuristic — candidate incumbent
            # PROF: _t_rcl = time_ns()
            rnd_key = cache_key(F0_rounded, F1_rounded)
            rnd_hit = haskey(cache, rnd_key)
            # PROF: _T_CACHE_LOOKUP[] += time_ns() - _t_rcl
            local LB_rnd::Float64
            if rnd_hit
                # PROF: _t_rcl2 = time_ns()
                g_rnd, _ = cache[rnd_key]
                # PROF: _T_CACHE_LOOKUP[] += time_ns() - _t_rcl2
                LB_rnd = sum(max(0.0, lambda_star[s] - g_rnd[s]) for s in 1:nscen)
            else
                # PROF: _t_er = time_ns()
                LB_rnd, x_sk_rnd, g_rnd, wu_rnd, rt_rnd = evaluate_node_persistent(
                    pmodels, F0_rounded, F1_rounded, lambda_star, n_vars,
                    solve_subproblem_persistent_fn)
                # PROF: _T_EVAL_ROUNDING[] += time_ns() - _t_er
                total_work_units += wu_rnd
                total_runtime    += rt_rnd
                # PROF: _t_rcs = time_ns()
                cache[rnd_key] = (g_rnd, x_sk_rnd)
                # PROF: _T_CACHE_STORE[] += time_ns() - _t_rcs
            end
            if LB_rnd > L_hat
                L_hat  = LB_rnd
                x_star = copy(x_rounded)
                println(log_io, "  *** New incumbent (rounding heuristic): L_hat = $(round(L_hat, digits=6))")
            end

            # Branch
            # PROF: _t_bs = time_ns()
            if branching_strategy == "P"
                j_star, branch_score = select_branching_variable_product(
                    frac_vars, x_sk, lambda0, lambda1, nscen)
                score_label = "product_score"
            else
                j_star, branch_score = select_branching_variable(frac_vars, x_bar)
                score_label = "frac_score"
            end
            # PROF: _T_BRANCH_SELECT[] += time_ns() - _t_bs

            # Branching diagnostics: how many candidates tied for the win.
            n_tied = 0
            @inbounds begin
                _m_t = frac_vars
                while _m_t != UInt64(0)
                    j = trailing_zeros(_m_t) + 1
                    if abs(abs(x_bar[j] - floor(x_bar[j] + 0.5)) - branch_score) <= 1e-12
                        n_tied += 1
                    end
                    _m_t &= _m_t - UInt64(1)
                end
            end
            branch_decisions += 1
            n_tied >= 2 && (tied_decisions += 1)
            active_sum += n_active
            n_active < min_active && (min_active = n_active)

            # PROF: _t_hp = time_ns()
            bit_jstar = _bit(j_star)
            F0_down = node.F0 | bit_jstar
            F1_up   = node.F1 | bit_jstar
            heap_push!(open_nodes, BBNode(F0_down, node.F1, U_k))
            heap_push!(open_nodes, BBNode(node.F0, F1_up,   U_k))
            # PROF: _T_HEAP_OPS[] += time_ns() - _t_hp

            UB  = heap_peek(open_nodes).upper_bound
            gap = UB > 1e-10 ? 100.0 * (UB - L_hat) / abs(UB) : 0.0
            if verbose
                # n_tied was computed above with the branching diagnostics.
                @printf(log_io, "  Node %4d  |F0|=%2d  |F1|=%2d  cache=%-4s  U_k=%.4f  LB=%.4f  UB=%.4f  gap=%.2f%%  work=%.4f  outcome=BRANCHED  branch_var=%d  %s=%.4f  nfrac=%d  nactive=%d  ntied=%d  open_nodes=%d\n",
                        nodes_processed, _popcount(node.F0), _popcount(node.F1), hit_str, U_k, L_hat, UB, gap, total_work_units, j_star, score_label, branch_score, _popcount(frac_vars), n_active, n_tied, length(open_nodes))
            end
        end

        UB_final  = isempty(open_nodes) ? L_hat : heap_peek(open_nodes).upper_bound
        gap_final = UB_final > 1e-10 ? 100.0 * (UB_final - L_hat) / abs(UB_final) : 0.0

        println(log_io, "="^60)
        if work_limit_hit
            println(log_io, "B&B terminated: work limit ($(round(total_work_units, digits=4)) wu)")
        else
            println(log_io, "B&B complete: $nodes_processed nodes processed")
        end
        println(log_io, "LB (best feasible) : $(round(L_hat, digits=6))")
        println(log_io, "UB (best node)     : $(round(UB_final, digits=6))")
        println(log_io, "Gap                : $(round(gap_final, digits=4))%")
        println(log_io, "Optimal x*         : $x_star")
        println(log_io, "Total work units   : $(round(total_work_units, digits=4))")
        println(log_io, "Total Gurobi runtime: $(round(total_runtime, digits=4)) s")
        println(log_io, "Cached nodes       : $(length(cache))")
        hit_rate = (cache_hits + cache_misses) > 0 ?
            100.0 * cache_hits / (cache_hits + cache_misses) : 0.0
        println(log_io, "Cache hit rate     : $(round(hit_rate, digits=1))%")
        println(log_io, "="^60)

        # PROF: _T_TOTAL[] = time_ns() - _t_total0
        # PROF: _T_EVAL_MAIN_CUM[] += _T_EVAL_MAIN[]
        # PROF: _T_TOTAL_CUM[]     += _T_TOTAL[]
        # PROF: _BB_CALLS_CUM[]    += 1
        # PROF: _print_bb_timers(log_io)

        # 10th element: branching diagnostics for this BB call (one CG iteration).
        branch_stats = (
            branch_decisions = branch_decisions,
            tied_decisions   = tied_decisions,
            active_sum       = active_sum,
            min_active       = branch_decisions > 0 ? min_active : 0,
            tie_pct          = branch_decisions > 0 ? 100.0 * tied_decisions / branch_decisions : 0.0,
            mean_active      = branch_decisions > 0 ? active_sum / branch_decisions : 0.0,
        )

        return x_star, L_hat, UB_final, gap_final, cache,
               total_work_units, nodes_processed,
               (cache_hits + cache_misses) > 0 ? 100.0 * cache_hits / (cache_hits + cache_misses) : 0.0,
               total_runtime, branch_stats

    finally
        log_file !== nothing && close(log_io)
    end
end
