# CG_AF_Knapsack.jl
# Entry point for Knapsack (KS/MK) Column Generation — Assignment Formulation
#
# Usage:
#   julia CG_AF_Knapsack.jl <problem_type> <instance_file> <K> <L> <pool_file> [work_budget]
#
#   problem_type : KS | MK
#   instance_file: path to instance txt (e.g. Knapsack/instances/KS/KS_n10_S10_1.txt)
#   K            : number of policies to select
#   L            : number of initial vectors to use (first L from pool_file)
#   pool_file    : path to binary vectors file to use as initial column pool
#   work_budget  : total Gurobi work unit cap (default 30000.0)

using Printf
using Dates
using Statistics

include("KnapsackAF.jl")
include("BranchAndBoundCore.jl")
include("BranchAndBoundKnapsack.jl")

# =============================================================================
# JSON HELPERS
# =============================================================================

function _json_escape_kaf(v::String)
    return "\"" * replace(v, "\\" => "\\\\", "\"" => "\\\"", "\n" => "\\n", "\t" => "\\t") * "\""
end

function _json_val_kaf(x)
    if x === nothing;         return "null"
    elseif x isa Bool;        return x ? "true" : "false"
    elseif x isa Integer;     return string(x)
    elseif x isa AbstractFloat
        (isnan(x) || !isfinite(x)) && return "null"
        return string(x)
    elseif x isa String;      return _json_escape_kaf(x)
    elseif x isa AbstractVector
        return "[" * join([_json_val_kaf(v) for v in x], ", ") * "]"
    elseif x isa Dict
        items = [_json_escape_kaf(string(k)) * ": " * _json_val_kaf(v) for (k, v) in x]
        return "{" * join(items, ", ") * "}"
    else
        return _json_escape_kaf(string(x))
    end
end

function _write_json_kaf(path::String, data)
    open(path, "w") do io
        write(io, _json_val_kaf(data))
        write(io, "\n")
    end
end

# =============================================================================
# CLI
# =============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 5
        println("Usage: julia CG_AF_Knapsack.jl <problem_type> <instance_file> <K> <L> <pool_file> [work_budget] [mode]")
        println("  problem_type : KS | MK")
        println("  instance_file: path to instance txt")
        println("  K            : number of policies to select")
        println("  L            : number of initial vectors (first L from pool_file)")
        println("  pool_file    : path to binary vectors file for initial column pool")
        println("  work_budget  : total work unit cap (default 30000.0)")
        println("  mode         : BB (most-fractional B&B) | BB_P (product-branching B&B) | omit for MIP pricing")
        println("\nExamples:")
        println("  julia CCG/CG_AF_Knapsack.jl KS Knapsack/instances/KS/KS_n10_S10_1.txt 2 20 \"Samples/Knapsack/Binary_vectors/Heuristic_solution_pool/KS_n10_S100.txt\"")
        println("  julia CCG/CG_AF_Knapsack.jl KS Knapsack/instances/KS/KS_n10_S10_1.txt 2 20 \"Samples/Knapsack/Binary_vectors/Heuristic_solution_pool/KS_n10_S100.txt\" 30000 BB")
        println("  julia CCG/CG_AF_Knapsack.jl MK Knapsack/instances/MK/MK_n10_S10_1.txt 3 50 \"Samples/Knapsack/Binary_vectors/Heuristic_solution_pool/MK_n10_S100.txt\" 50000")
        exit(1)
    end

    problem_type_str  = uppercase(strip(ARGS[1]))
    problem_type      = problem_type_str == "KS" ? :KS : :MK
    instance_file     = ARGS[2]
    K                 = parse(Int, ARGS[3])
    L                 = parse(Int, ARGS[4])
    pool_file         = ARGS[5]
    total_work_budget = length(ARGS) >= 6 ? parse(Float64, ARGS[6]) : 30000.0

    arg7                = length(ARGS) >= 7 ? uppercase(strip(ARGS[7])) : ""
    use_bb_pricing      = arg7 in ("BB", "BB_F", "BB_P")
    bb_branching_strategy = arg7 == "BB_P" ? "P" : "F"
    mode_tag            = use_bb_pricing ? (bb_branching_strategy == "P" ? "BB_P" : "BB") : "std"

    println("="^70)
    println("COLUMN GENERATION — KNAPSACK  ($problem_type_str)")
    println("="^70)
    println("Instance     : $(basename(instance_file))")
    println("K            : $K")
    println("L            : $L")
    println("Pool file    : $(basename(pool_file))")
    println("Work budget  : $total_work_budget")
    println("Pricing mode : $(use_bb_pricing ? "branch-and-bound (branching: $(bb_branching_strategy == "P" ? "product/sensitivity" : "most-fractional"))" : "MIP")")
    if use_bb_pricing
        nthr = Threads.nthreads()
        println("Julia threads: $nthr  (CPU threads available: $(Sys.CPU_THREADS))")
        if nthr == 1
            println("  ⚠  WARNING: BB mode parallelises the scenario loop with Threads.@threads.")
            println("     With 1 Julia thread it falls back to serial — no speedup.")
            println("     Re-run with:  julia -t $(min(8, Sys.CPU_THREADS)) CCG/CG_AF_Knapsack.jl ...")
        end
    end
    println("="^70)
    println()

    start_time = time()

    # ── Load instance ─────────────────────────────────────────────────────────
    local n, S, profits, weights, capacities
    if problem_type == :KS
        n, S, profits, weights, capacities = read_ks_instance(instance_file)
    else
        n, S, profits, weights, capacities = read_mk_instance(instance_file)
    end
    println("n=$n  S=$S")

    inst_tag     = splitext(basename(instance_file))[1]
    project_root = dirname(@__DIR__)

    # ── Load WS binary vectors (for WS profit computation only) ───────────────
    ws_subdir = occursin("KS_Large", instance_file) ? "KS_Large" : problem_type_str
    ws_file = joinpath(project_root, "Samples", "Knapsack", "Binary_vectors",
                       ws_subdir, "$(inst_tag).txt")
    if !isfile(ws_file)
        error("WS binary vectors file not found: $ws_file")
    end
    ws_binary_vectors = load_binary_vectors(ws_file)
    println("Loaded $(length(ws_binary_vectors)) WS binary vectors (for profit evaluation)")

    # ── Load solution pool (initial column pool, first L vectors) ─────────────
    if !isfile(pool_file)
        error("Solution pool file not found: $pool_file")
    end
    all_pool_vectors = load_binary_vectors(pool_file)
    if L > length(all_pool_vectors)
        error("L=$L exceeds number of available vectors ($(length(all_pool_vectors))) in $pool_file")
    end
    initial_binary_vectors = all_pool_vectors[1:L]
    println("Loaded $(length(initial_binary_vectors)) initial vectors from pool (first L=$L of $(basename(pool_file)))")

    # ── WS profits (for gap reporting) ────────────────────────────────────────
    ws_profits_vec = if problem_type == :KS
        compute_ws_profits_KS(n, S, profits, weights, capacities, ws_binary_vectors)
    else
        compute_ws_profits_MK(n, S, profits, weights, capacities, ws_binary_vectors)
    end
    avg_ws_profit = mean(ws_profits_vec)
    @printf("WS avg profit : %.4f\n\n", avg_ws_profit)

    # ── Encode scenarios ──────────────────────────────────────────────────────
    scenarios = if problem_type == :KS
        encode_scenarios_KS(profits, weights, capacities, n, S)
    else
        encode_scenarios_MK(profits, weights, capacities, n, S)
    end

    # ── Problem data ──────────────────────────────────────────────────────────
    problem_data = Dict{String, Any}(
        "n"                    => n,
        "problem_type"         => problem_type,
        "dims"                 => MK_DIMS,
        "S"                    => S,
        "K"                    => K,
        "instance"             => instance_file,
        "f_s"                  => [-p for p in ws_profits_vec],
        "use_decomp"           => false,
        "convergence_tol"      => 0.01,
        "use_bb_pricing"       => use_bb_pricing,
        "bb_branching_strategy" => bb_branching_strategy
    )

    # ── Log files ─────────────────────────────────────────────────────────────
    logs_dir  = joinpath(@__DIR__, "logs")
    isdir(logs_dir) || mkpath(logs_dir)
    ts        = Dates.format(now(), "YYYYmmdd_HHMMSS")
    master_log_file  = joinpath(logs_dir, "CG_KS_$(problem_type_str)_$(inst_tag)_K$(K)_L$(L)_$(mode_tag)_master_$(ts).log")
    pricing_log_file = joinpath(logs_dir, "CG_KS_$(problem_type_str)_$(inst_tag)_K$(K)_L$(L)_$(mode_tag)_pricing_$(ts).log")

    for lf in [master_log_file, pricing_log_file]
        open(lf, "w") do file
            println(file, "="^70)
            println(file, "CG KNAPSACK — $problem_type_str   $(basename(instance_file))   K=$K")
            println(file, "Started: $(now())")
            println(file, "="^70)
        end
    end
    println("Logs:")
    println("  Master  : $master_log_file")
    println("  Pricing : $pricing_log_file")
    println()

    # ── Run CG ────────────────────────────────────────────────────────────────
    eval_cost_fn  = problem_type == :KS ? evaluate_cost_KS  : evaluate_cost_MK
    eval_batch_fn = problem_type == :KS ? evaluate_cost_batch_KS : evaluate_cost_batch_MK

    results = column_generation_algorithm(
        scenarios,
        initial_binary_vectors,
        problem_data,
        K,
        typemax(Int);
        evaluate_cost       = eval_cost_fn,
        build_pricing_model = build_pricing_model_AF_KS,
        calculate_m_values  = calculate_m_values_KS,
        evaluate_cost_batch = eval_batch_fn,
        pricing_type        = 2,
        use_solution_pool   = true,
        total_work_budget   = total_work_budget,
        opt_worklimit       = 300.0,
        master_log_file     = master_log_file,
        pricing_log_file    = pricing_log_file,
        convergence_tol     = 0.01
    )

    # ── Convert objectives  (master minimises -profit, so negate to get profit) ──
    obj_lp_raw           = results["final_objective_lp"]   # already divided by S by core
    obj_ip_raw           = results["final_objective_ip"]
    avg_profit_lp        = -obj_lp_raw
    avg_profit_ip        = -obj_ip_raw
    gap_pct              = avg_ws_profit > 0 ?
                           (avg_ws_profit - avg_profit_ip) / avg_ws_profit * 100 : 0.0

    println("\n" * "="^70)
    println("COLUMN GENERATION RESULTS")
    println("="^70)
    println("Status              : $(results["status"])")
    println("Iterations          : $(results["iterations"])")
    @printf("Avg profit  (LP)    : %.4f\n", avg_profit_lp)
    @printf("Avg profit  (IP)    : %.4f\n", avg_profit_ip)
    @printf("WS avg profit       : %.4f\n", avg_ws_profit)
    @printf("Gap (WS−IP)/WS      : %.2f%%\n", gap_pct)
    println("Solutions added     : $(results["total_solutions_added"])")
    println("Master work units   : $(round(results["total_master_work_units"], digits=2))")
    println("Pricing work units  : $(round(results["total_pricing_work_units"], digits=2))")
    println("="^70)

    # ── Save results ──────────────────────────────────────────────────────────
    results_dir = joinpath(project_root, "Results", "Knapsack", "CG", problem_type_str)
    mkpath(results_dir)

    elapsed = time() - start_time

    summary = Dict(
        "instance"                 => basename(instance_file),
        "problem_type"             => problem_type_str,
        "n"                        => n,
        "S"                        => S,
        "K"                        => K,
        "L"                        => L,
        "pool_file"                => basename(pool_file),
        "status"                   => results["status"],
        "iterations"               => results["iterations"],
        "total_solutions_added"    => results["total_solutions_added"],
        "avg_profit_lp"            => avg_profit_lp,
        "avg_profit_ip"            => avg_profit_ip,
        "avg_ws_profit"            => avg_ws_profit,
        "gap_pct"                  => gap_pct,
        "total_work_units"         => get(results, "total_work_units", nothing),
        "total_master_work_units"  => results["total_master_work_units"],
        "total_pricing_work_units" => results["total_pricing_work_units"],
        "total_update_work_units"  => results["total_update_work_units"],
        "termination_iteration"    => get(results, "termination_iteration", results["iterations"]),
        "elapsed_seconds"          => elapsed
    )

    out_file  = joinpath(results_dir, "cg_kaf_$(inst_tag)_K$(K)_L$(L)_$(mode_tag).json")
    iter_file = joinpath(results_dir, "cg_kaf_iter_$(inst_tag)_K$(K)_L$(L)_$(mode_tag).json")

    _write_json_kaf(out_file,  summary)
    _write_json_kaf(iter_file, get(results, "iteration_details", []))

    println("\nResults saved to : $out_file")
    println("Iterations saved : $iter_file")

    @printf("\nElapsed time     : %.2f seconds (%.2f minutes)\n", elapsed, elapsed / 60)
end
