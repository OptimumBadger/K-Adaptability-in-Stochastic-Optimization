# CG_SF_TS.jl
# Entry point for Transmission Switching Column Generation - Subset Formulation (SF)
# Usage: julia CG_SF_TS.jl <instance_file> <scenarios_file> <initial_subsets_file> <K> <S> [pricing_type] [line_capacity_factor]

using Printf
using Dates

# =============================================================================
# JSON OUTPUT HELPERS
# =============================================================================

function json_escape(value::String)
    escaped = replace(value, "\\" => "\\\\", "\"" => "\\\"", "\n" => "\\n", "\r" => "\\r", "\t" => "\\t")
    return "\"" * escaped * "\""
end

function json_value(x)
    if x === nothing
        return "null"
    elseif x isa Bool
        return x ? "true" : "false"
    elseif x isa Integer
        return string(x)
    elseif x isa AbstractFloat
        if isnan(x) || !isfinite(x)
            return "null"
        end
        return string(x)
    elseif x isa String
        return json_escape(x)
    elseif x isa AbstractVector
        return "[" * join([json_value(v) for v in x], ", ") * "]"
    elseif x isa Dict
        items = String[]
        for (k, v) in x
            push!(items, json_escape(string(k)) * ": " * json_value(v))
        end
        return "{" * join(items, ", ") * "}"
    else
        return json_escape(string(x))
    end
end

function write_json(path::String, data)
    open(path, "w") do io
        write(io, json_value(data))
        write(io, "\n")
    end
end

function extract_experiment_type(file_path::String)
    filename = lowercase(basename(file_path))
    if occursin("bernoulli", filename)
        return "bernoulli"
    elseif occursin("low", filename)
        return "low"
    elseif occursin("high", filename)
        return "high"
    end
    return "unknown"
end

# Include required files
include("CGUtilities.jl")
include("SubsetFormulation.jl")
include("TransmissionSwitchingSF.jl")  # Includes PowerModels
include("../Samples/TS/TSUtilities.jl")  # For extract_case_name

# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 5
        println("❌ Error: Incorrect number of arguments!")
        println("Usage: julia CG_SF_TS.jl <instance_file> <scenarios_file> <initial_subsets_file> <K> <S> [pricing_type] [line_capacity_factor]")
        println("\nArguments:")
        println("  instance_file: Instance file path (e.g., Transmission_Switching/instances/case118.m)")
        println("  scenarios_file: File containing scenarios (e.g., Samples/TS/Scenarios/instance_1/S300_bernoulli.txt)")
        println("  initial_subsets_file: File containing initial subset vectors (π vectors) - required")
        println("  K: Number of subsets to select")
        println("  S: Number of scenarios (should match scenarios_file)")
        println("  pricing_type: 1 or 2 (default: 2)")
        println("  line_capacity_factor: Line capacity factor (default: 1.0)")
        println("\nExample:")
        println("  julia CG_SF_TS.jl Transmission_Switching/instances/case118.m Samples/TS/Scenarios/instance_1/S300_bernoulli.txt Samples/TS/Subsets/initial_subsets.txt 2 300 2 1.0")
        exit(1)
    end
    
    # Parse required arguments
    instance_file = ARGS[1]
    scenarios_file = ARGS[2]
    initial_subsets_file = ARGS[3]
    K = parse(Int, ARGS[4])
    S = parse(Int, ARGS[5])
    
    # Parse optional arguments
    pricing_type = length(ARGS) > 5 ? parse(Int, ARGS[6]) : 2
    line_capacity_factor = length(ARGS) > 6 ? parse(Float64, ARGS[7]) : 1.0
    
    # Validate pricing_type
    if !(pricing_type in [1, 2])
        println("❌ Error: pricing_type must be 1 or 2!")
        exit(1)
    end
    
    println("="^80)
    println("COLUMN GENERATION SF - TRANSMISSION SWITCHING")
    println("="^80)
    println("Instance: $instance_file")
    println("Scenarios file: $scenarios_file")
    println("Initial subsets file: $initial_subsets_file")
    println("Parameters: K=$K, S=$S, pricing_type=$pricing_type")
    println("Problem parameters: line_capacity_factor=$line_capacity_factor")
    println("="^80)
    println()
    
    # Load instance data
    println("Loading instance data...")
    if !isfile(instance_file)
        error("❌ Error: Instance file not found! Expected: $instance_file")
    end
    ts_data = parse_file_silent(instance_file)
    
    # Store instance path for downstream use (e.g., fixed lines reading)
    ts_data["instance_file"] = instance_file
    ts_data["line_capacity_factor"] = line_capacity_factor
    
    # Extract case name and set num_lines_fixed for case118.m (always 110 lines fixed)
    case_name = extract_case_name(instance_file)
    if case_name == "case118"
        ts_data["num_lines_fixed"] = 110
        println("✅ Detected case118.m: Setting 110 lines fixed ON")
    end
    
    # Apply line capacity factor if needed
    if line_capacity_factor != 1.0
        for (branch_id, branch) in ts_data["branch"]
            if haskey(branch, "rate_a")
                branch["rate_a"] = branch["rate_a"] * line_capacity_factor
            end
        end
    end
    
    nb = length(ts_data["bus"])
    nl = length(ts_data["branch"])
    println("✅ Instance loaded: $nb buses, $nl lines")
    println()
    
    # Load scenarios from file
    println("Loading scenarios from file...")
    all_scenarios = load_scenarios_from_file(scenarios_file)
    if length(all_scenarios) < S
        error("❌ Error: Requested S = $S scenarios, but scenarios_file contains only $(length(all_scenarios)) scenarios.")
    end
    # Slice first S scenarios
    scenarios = all_scenarios[1:S]
    println("✅ Loaded $(length(all_scenarios)) scenarios from file, using first S = $S scenarios")
    
    # Apply demand multiplier for consistency with heuristic
    case_name = extract_case_name(instance_file)
    demand_mult = get_demand_multiplier(instance_file=instance_file, case_name=case_name)
    println("Applying demand multiplier: $demand_mult (for consistency with heuristic)")
    scenarios = [scenario .* demand_mult for scenario in scenarios]
    println()
    
    # Load initial subsets from file (required)
    println("Loading initial subsets from file...")
    initial_subsets = Vector{Vector{Int}}()
    if isfile(initial_subsets_file)
        # Load subset vectors from file (one line per subset, space-separated scenario indices)
        # Convert scenario indices (1-based) to binary vectors of length S
        line_num_ref = Ref(0)
        skipped_indices_ref = Ref(0)
        open(initial_subsets_file, "r") do io
            for line in eachline(io)
                line_num_ref[] += 1
                if !isempty(strip(line))
                    scenario_indices = [parse(Int, x) for x in split(strip(line))]
                    # Convert to binary vector: 1 if scenario is in subset, 0 otherwise
                    binary_vec = zeros(Int, S)
                    for idx in scenario_indices
                        if 1 <= idx <= S
                            binary_vec[idx] = 1
                        else
                            skipped_indices_ref[] += 1
                            println("⚠️  Warning: Line $(line_num_ref[]), scenario index $idx is out of range [1, $S], ignoring")
                        end
                    end
                    push!(initial_subsets, binary_vec)
                end
            end
        end
        println("✅ Loaded $(length(initial_subsets)) initial subsets")
        if skipped_indices_ref[] > 0
            println("⚠️  Warning: Skipped $(skipped_indices_ref[]) scenario indices that were out of range")
        end
        
        # Diagnostic: Check if all scenarios are covered
        if length(initial_subsets) > 0
            coverage = zeros(Int, S)
            for subset in initial_subsets
                for s in 1:S
                    if subset[s] == 1
                        coverage[s] += 1
                    end
                end
            end
            uncovered = [s for s in 1:S if coverage[s] == 0]
            if !isempty(uncovered)
                println("⚠️  WARNING: The following scenarios are NOT covered by any initial subset:")
                println("   Uncovered scenarios: $uncovered")
                println("   This will cause the master problem to be INFEASIBLE!")
            else
                println("✅ All $S scenarios are covered by at least one initial subset")
            end
        end
    else
        println("❌ Error: Initial subsets file not found: $initial_subsets_file")
        exit(1)
    end
    println()
    
    # Create logs directory
    script_dir = @__DIR__
    logs_dir = joinpath(script_dir, "logs")
    if !isdir(logs_dir)
        mkpath(logs_dir)
        println("Created logs directory: $logs_dir")
    end
    
    # Generate log file names with timestamp
    timestamp = Dates.format(now(), "YYYYmmdd_HHMMSS")
    instance_name = replace(splitext(basename(instance_file))[1], "." => "_")
    master_log_file = joinpath(logs_dir, "CG_SF_TS_$(instance_name)_K$(K)_S$(S)_master_$(timestamp).log")
    pricing_log_file = joinpath(logs_dir, "CG_SF_TS_$(instance_name)_K$(K)_S$(S)_pricing_$(timestamp).log")
    
    # Initialize log files with headers
    open(master_log_file, "w") do file
        println(file, "="^80)
        println(file, "COLUMN GENERATION SF - MASTER PROBLEM LOGS")
        println(file, "="^80)
        println(file, "Instance: $instance_file")
        println(file, "K: $K, S: $S")
        println(file, "Pricing Type: $pricing_type")
        println(file, "Started: $(now())")
        println(file, "="^80)
    end
    
    open(pricing_log_file, "w") do file
        println(file, "="^80)
        println(file, "COLUMN GENERATION SF - PRICING SUBPROBLEM LOGS")
        println(file, "="^80)
        println(file, "Instance: $instance_file")
        println(file, "K: $K, S: $S")
        println(file, "Pricing Type: $pricing_type")
        println(file, "Started: $(now())")
        println(file, "="^80)
    end
    
    println("Log files:")
    println("  Master: $master_log_file")
    println("  Pricing: $pricing_log_file")
    println()
    
    # Run column generation
    println("Starting SF column generation algorithm...")
    println()
    
    results = sf_column_generation_algorithm(
        scenarios,
        initial_subsets,
        ts_data,
        K,
        500;  # max_iterations
        evaluate_subset_cost = evaluate_subset_cost_TS,
        evaluate_subset_cost_batch = evaluate_subset_cost_batch_template_TS,
        build_pricing_model = build_pricing_model_SF_TS,
        pricing_type = pricing_type,
        use_solution_pool = true,
        total_work_budget = 30000.0,
        opt_worklimit = DEFAULT_OPT_WORKLIMIT,
        master_log_file = master_log_file,
        pricing_log_file = pricing_log_file
    )
    
    # Print results
    println("\n" * "="^80)
    println("COLUMN GENERATION SF RESULTS")
    println("="^80)
    
    # Write JSON outputs (summary + iterations)
    project_root = dirname(@__DIR__)
    results_dir = joinpath(project_root, "Results", "Transmission_Switching", "CG")
    if !isdir(results_dir)
        mkpath(results_dir)
    end
    
    experiment = extract_experiment_type(scenarios_file)
    instance_tag = replace(splitext(basename(instance_file))[1], "." => "_")
    
    # Extract T from initial_subsets_file if present (similar to FL)
    t_match = match(r"_T(\d+)_", initial_subsets_file)
    T = t_match === nothing ? nothing : parse(Int, t_match.captures[1])
    t_str = T === nothing ? "Tunknown" : "T$(T)"
    
    summary = Dict(
        "instance" => basename(instance_file),
        "experiment" => experiment,
        "S" => S,
        "K" => K,
        "T" => T,
        "pricing_type" => pricing_type,
        "iterations" => results["iterations"],
        "total_subsets_added" => results["total_subsets_added"],
        "final_objective_lp" => results["final_objective_lp"],
        "final_objective_ip" => get(results, "final_objective_ip", nothing),
        "total_master_work_units" => results["total_master_work_units"],
        "total_pricing_work_units" => results["total_pricing_work_units"],
        "total_update_work_units" => results["total_update_work_units"]
    )
    
    summary_file = joinpath(
        results_dir,
        "cg_sf_results_$(instance_tag)_S$(S)_K$(K)_$(t_str)_$(experiment).json"
    )
    write_json(summary_file, summary)
    
    # Build iteration details table from results data
    iteration_details = Vector{Dict{String, Any}}()
    master_map = Dict{Int, Any}()
    for row in get(results, "iteration_master_data", [])
        master_map[row.iteration] = row
    end
    pricing_map = Dict{Int, Any}()
    for row in get(results, "iteration_pricing_data", [])
        pricing_map[row.iteration] = row
    end
    columns_map = Dict{Int, Any}()
    for row in get(results, "iteration_columns_data", [])
        columns_map[row.iteration] = row
    end
    
    iterations = sort(collect(union(keys(master_map), keys(pricing_map), keys(columns_map))))
    for it in iterations
        m = get(master_map, it, nothing)
        p = get(pricing_map, it, nothing)
        c = get(columns_map, it, nothing)
        push!(iteration_details, Dict(
            "iteration" => it,
            "master_obj" => m === nothing ? nothing : m.master_obj,
            "pricing_objective" => p === nothing ? nothing : p.obj_pricing,
            "reduced_cost" => p === nothing ? nothing : p.reduced_cost,
            "subsets_added" => c === nothing ? nothing : c.num_added
        ))
    end
    
    iteration_file = joinpath(
        results_dir,
        "cg_sf_iterations_$(instance_tag)_S$(S)_K$(K)_$(t_str)_$(experiment).json"
    )
    write_json(iteration_file, iteration_details)
    
    println("📁 CG SF results JSON saved to: $summary_file")
    println("📁 CG SF iteration JSON saved to: $iteration_file")
    println("Status: $(results["status"])")
    println("Iterations: $(results["iterations"])")
    println("Final objective (LP): $(round(results["final_objective_lp"], digits=2))")
    if haskey(results, "final_objective_ip")
        println("Final objective (IP): $(round(results["final_objective_ip"], digits=2))")
        if results["final_objective_lp"] != Inf && results["final_objective_ip"] != Inf
            gap = results["final_objective_ip"] - results["final_objective_lp"]
            gap_percent = (gap / results["final_objective_lp"]) * 100
            println("Gap (IP - LP): $(round(gap, digits=2)) ($(round(gap_percent, digits=2))%)")
        end
    end
    println("Selected subsets: $(results["selected_subsets"])")
    println("Total subsets added: $(results["total_subsets_added"])")
    println("Total master work units: $(round(results["total_master_work_units"], digits=2))")
    println("Total pricing work units: $(round(results["total_pricing_work_units"], digits=2))")
    println("Total update work units: $(round(results["total_update_work_units"], digits=2))")
    println("="^80)
end
