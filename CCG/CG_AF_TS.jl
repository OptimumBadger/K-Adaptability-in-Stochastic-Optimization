# CG_AF_TS.jl
# Entry point for Transmission Switching Column Generation
# Usage: julia CG_AF_TS.jl <instance_file> <scenarios_file> <binary_vectors_file> <K> <S> [pricing_type] [line_capacity_factor]

using Printf
using Dates

# Include required files
include("CGUtilities.jl")
include("AssignmentFormulation.jl")
include("TransmissionSwitchingAF.jl")

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
    elseif occursin("new", filename)
        return "new"
    else
        return "unknown"
    end
end

function parse_bool_flag(raw::String)
    v = lowercase(strip(raw))
    return v in ("1", "true", "yes", "y", "on")
end

# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 5
        println("❌ Error: Incorrect number of arguments!")
        println("Usage: julia CG_AF_TS.jl <instance_file> <scenarios_file> <binary_vectors_file> <K> <S> [pricing_type] [line_capacity_factor] [no_fixed_lines] [use_decomp]")
        println("\nArguments:")
        println("  instance_file: Instance file path (e.g., Transmission_Switching/instances/case118Blumsack.m)")
        println("  scenarios_file: File containing SAA samples (e.g., Transmission_Switching/Heuristic/SAA_samples_S25.txt)")
        println("  binary_vectors_file: File containing initial binary vectors (e.g., Transmission_Switching/Heuristic/binary_vectors_L25.txt)")
        println("  K: Number of solutions to select")
        println("  S: Number of scenarios (should match scenarios_file)")
        println("  pricing_type: 1 or 2 (default: 2)")
        println("  line_capacity_factor: Line capacity factor (default: 1.0)")
        println("  no_fixed_lines: Set to 'true', '1', 'yes', or 'switchable' to disable fixed lines (all lines switchable, default: false)")
        println("  use_decomp: Enable decomposition for Type-2 pricing (default: false)")
        println("\nExample:")
        println("  julia CG_AF_TS.jl Transmission_Switching/instances/case118Blumsack.m Transmission_Switching/Heuristic/SAA_samples_S25.txt Transmission_Switching/Heuristic/binary_vectors_L25.txt 2 25")
        exit(1)
    end
    
    # Parse required arguments
    instance_file = ARGS[1]
    scenarios_file = ARGS[2]
    binary_vectors_file = ARGS[3]
    K = parse(Int, ARGS[4])
    S = parse(Int, ARGS[5])
    
    # Parse optional arguments
    pricing_type = length(ARGS) > 5 ? parse(Int, ARGS[6]) : 2
    line_capacity_factor = length(ARGS) > 6 ? parse(Float64, ARGS[7]) : 1.0
    no_fixed_lines = length(ARGS) > 7 ? (lowercase(ARGS[8]) in ["true", "1", "yes", "switchable"]) : false
    use_decomp = length(ARGS) > 8 ? parse_bool_flag(ARGS[9]) : false
    
    # Validate pricing_type
    if !(pricing_type in [1, 2])
        println("❌ Error: pricing_type must be 1 or 2!")
        exit(1)
    end
    
    println("="^80)
    println("COLUMN GENERATION - TRANSMISSION SWITCHING")
    println("="^80)
    println("Instance: $instance_file")
    println("Scenarios file: $scenarios_file")
    println("Binary vectors file: $binary_vectors_file")
    println("Parameters: K=$K, S=$S, pricing_type=$pricing_type")
    println("Problem parameters: line_capacity_factor=$line_capacity_factor")
    println("Pricing mode: $(use_decomp ? "decomposition" : "standard")")
    println("="^80)
    println()
    
    # Load instance data
    println("Loading instance data...")
    ts_data = load_TS_data(instance_file; line_capacity_factor=line_capacity_factor)
    nb = length(ts_data["bus"])
    nl = length(ts_data["branch"])
    println("✅ Instance loaded: $nb buses, $nl lines")
    println()
    
    # Load scenarios from file
    println("Loading scenarios from file...")
    all_scenarios = load_scenarios_from_file(scenarios_file)
    
    # Slice first S scenarios
    if S > length(all_scenarios)
        error("❌ Error: Requested S = $S scenarios, but scenarios_file contains only $(length(all_scenarios)) scenarios.")
    end
    scenarios = all_scenarios[1:S]
    println("✅ Using first $S scenarios from file (file contains $(length(all_scenarios)) scenarios)")
    
    # Store experiment metadata for pricing cache
    ts_data["experiment_type"] = extract_experiment_type(scenarios_file)
    ts_data["scenarios_file"] = scenarios_file
    
    # Apply demand multiplier (0.9) for consistency with heuristic
    case_name = extract_case_name(instance_file)
    demand_mult = get_demand_multiplier(instance_file=instance_file, case_name=case_name)
    println("Applying demand multiplier: $demand_mult (for consistency with heuristic)")
    scenarios = [scenario .* demand_mult for scenario in scenarios]
    println()
    
    # Load initial binary vectors from file
    println("Loading initial binary vectors from file...")
    initial_binary_vectors = load_binary_vectors_from_file(binary_vectors_file)
    L = length(initial_binary_vectors)  # Extract L from loaded vectors
    
    # Extract number of lines to fix ON from binary vectors filename
    # e.g., "B_case118_30_bernoulli.txt" -> 30
    # OR use no_fixed_lines flag to disable all fixed lines
    num_lines_fixed = nothing
    if no_fixed_lines
        println("🔓 All lines switchable mode enabled (no fixed lines)")
        # Explicitly set to empty array by storing a special marker
        ts_data["no_fixed_lines"] = true
    else
        binary_vectors_basename = basename(binary_vectors_file)
        # Pattern matches: _(\d+)_ or _(\d+).txt (handles both B_case118_30_bernoulli.txt and B25_bernoulli_90.txt)
        m = match(r"_(\d+)(?:_|\.)", binary_vectors_basename)
        if m !== nothing
            num_lines_fixed = parse(Int, m.captures[1])
            println("Detected $num_lines_fixed lines to fix ON from filename")
            ts_data["num_lines_fixed"] = num_lines_fixed
        else
            println("Warning: Could not extract number of lines from filename, all lines will be switchable")
            ts_data["no_fixed_lines"] = true
        end
    end
    println()
    
    # Extract f_s values from WS file (if decomposition is enabled)
    if use_decomp && pricing_type == 2
        println("Extracting f_s values (wait-and-see costs) from WS file...")
        # Determine WS file path from scenarios_file
        # scenarios_file format: Samples/TS/Scenarios/instance_1/S300_bernoulli.txt
        # WS file format: Samples/TS/Scenarios/instance_1/WS300_bernoulli.txt or WS25_bernoulli_110.txt
        scenarios_dir = dirname(scenarios_file)
        experiment = extract_experiment_type(scenarios_file)
        
        # Try to extract instance number from scenarios_file path
        instance_num = "1"  # default
        if occursin("instance_1", scenarios_dir) || occursin("instance1", scenarios_dir)
            instance_num = "1"
        elseif occursin("instance_2", scenarios_dir) || occursin("instance2", scenarios_dir)
            instance_num = "2"
        end
        
        # Try multiple possible WS file paths
        # Pattern 1: WS300_bernoulli.txt (same directory as scenarios)
        # Pattern 2: WS25_bernoulli_110.txt (with S and N in filename)
        # Pattern 3: instance_X/WS300_bernoulli.txt (subdirectory)
        ws_file_candidates = [
            joinpath(scenarios_dir, "WS300_$(experiment).txt"),  # same directory as scenarios
            joinpath(scenarios_dir, "WS$(S)_$(experiment).txt"),  # with S
            joinpath(scenarios_dir, "instance_$(instance_num)", "WS300_$(experiment).txt"),  # instance_X subdirectory
            joinpath(dirname(scenarios_dir), "instance_$(instance_num)", "WS300_$(experiment).txt"),  # parent/instance_X
        ]
        
        # If num_lines_fixed is known, also try patterns with N
        if num_lines_fixed !== nothing
            prepend!(ws_file_candidates, [
                joinpath(scenarios_dir, "WS$(S)_$(experiment)_$(num_lines_fixed).txt"),
                joinpath(scenarios_dir, "WS300_$(experiment)_$(num_lines_fixed).txt"),
            ])
        end
        
        local ws_file = nothing
        for candidate in ws_file_candidates
            if isfile(candidate)
                ws_file = candidate
                break
            end
        end
        
        if ws_file === nothing
            error("❌ Error: WS file not found! Tried:\n" * join(["  - $c" for c in ws_file_candidates], "\n") * "\nPlease ensure WS file exists.")
        end
        
        println("  WS file: $ws_file")
        f_s = extract_f_s_from_ws_file(ws_file, S)
        println("✅ Extracted $(length(f_s)) f_s values (wait-and-see costs) from WS file")
        println()
        
        # Store f_s in problem_data
        ts_data["f_s"] = f_s
        ts_data["use_decomp"] = use_decomp
    elseif use_decomp && pricing_type != 2
        println("⚠️  Warning: use_decomp is only supported for pricing_type=2. Ignoring use_decomp flag.")
        ts_data["use_decomp"] = false
    else
        ts_data["use_decomp"] = false
    end
    
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
    master_log_file = joinpath(logs_dir, "CG_AF_TS_$(instance_name)_L$(L)_S$(S)_K$(K)_master_$(timestamp).log")
    pricing_log_file = joinpath(logs_dir, "CG_AF_TS_$(instance_name)_L$(L)_S$(S)_K$(K)_pricing_$(timestamp).log")
    
    # Initialize log files with headers
    pricing_type_str = if pricing_type == 1
        "Type 1"
    elseif pricing_type == 2
        use_decomp ? "Type 2 (Decomposition)" : "Type 2 (Standard)"
    else
        "Type $pricing_type"
    end
    
    # Determine fixed lines description (reuse case_name from earlier)
    if no_fixed_lines || get(ts_data, "no_fixed_lines", false)
        fixed_lines_desc = "all lines switchable"
    elseif num_lines_fixed !== nothing
        fixed_lines_desc = "$num_lines_fixed lines fixed ON"
    else
        fixed_lines_desc = "all lines switchable"
    end
    instance_desc = "$case_name.m with $fixed_lines_desc"
    
    open(master_log_file, "w") do file
        println(file, "="^80)
        println(file, "COLUMN GENERATION - MASTER PROBLEM LOGS")
        println(file, "="^80)
        println(file, "Instance: $instance_desc")
        println(file, "Instance file: $instance_file")
        println(file, "L: $L, S: $S, K: $K")
        println(file, "Pricing Type: $pricing_type_str")
        println(file, "Started: $(now())")
        println(file, "="^80)
    end
    
    open(pricing_log_file, "w") do file
        println(file, "="^80)
        println(file, "COLUMN GENERATION - PRICING SUBPROBLEM LOGS")
        println(file, "="^80)
        println(file, "Instance: $instance_desc")
        println(file, "Instance file: $instance_file")
        println(file, "L: $L, S: $S, K: $K")
        println(file, "Pricing Type: $pricing_type_str")
        println(file, "Started: $(now())")
        println(file, "="^80)
    end
    
    println("Log files:")
    println("  Master: $master_log_file")
    println("  Pricing: $pricing_log_file")
    println()
    
    # Run column generation
    println("Starting column generation algorithm...")
    println()
    
    results = column_generation_algorithm(
        scenarios,
        initial_binary_vectors,
        ts_data,
        K,
        100;  # max_iterations
        evaluate_cost = evaluate_cost_TS,
        build_pricing_model = build_pricing_model_TS,
        calculate_m_values = calculate_m_values_TS,
        evaluate_cost_batch = nothing,  # Using original evaluate_cost_TS (no batch template for now)
        pricing_type = pricing_type,
        use_solution_pool = true,
        total_work_budget = 30000.0,
        opt_worklimit = DEFAULT_OPT_WORKLIMIT,
        master_log_file = master_log_file,
        pricing_log_file = pricing_log_file
    )
    
    # Print results
    println("\n" * "="^80)
    println("COLUMN GENERATION RESULTS")
    println("="^80)
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
    println("Selected solutions: $(results["selected_solutions"])")
    println("Total solutions added: $(results["total_solutions_added"])")
    println("Total master work units: $(round(results["total_master_work_units"], digits=2))")
    println("Total pricing work units: $(round(results["total_pricing_work_units"], digits=2))")
    println("Total update work units: $(round(results["total_update_work_units"], digits=2))")
    println("="^80)
    
    # Write JSON outputs
    project_root = dirname(@__DIR__)
    results_dir = joinpath(project_root, "Results", "Transmission_Switching", "CG", "AF")
    if !isdir(results_dir)
        mkpath(results_dir)
    end
    
    experiment = extract_experiment_type(scenarios_file)
    instance_tag = replace(case_name, "." => "_")  # Reuse case_name from earlier
    
    summary = Dict(
        "instance" => instance_file,
        "experiment" => experiment,
        "L" => L,
        "S" => S,
        "K" => K,
        "pricing_type" => pricing_type,
        "iterations" => results["iterations"],
        "termination_iteration" => get(results, "termination_iteration", results["iterations"]),
        "phase2_iterations" => get(results, "phase2_iterations", 0),
        "total_solutions_added" => results["total_solutions_added"],
        "final_objective_lp" => results["final_objective_lp"],
        "final_objective_ip" => get(results, "final_objective_ip", nothing),
        "total_work_units" => get(results, "total_work_units", nothing),
        "total_master_work_units" => results["total_master_work_units"],
        "total_pricing_work_units" => results["total_pricing_work_units"],
        "total_update_work_units" => results["total_update_work_units"]
    )
    
    summary_file = joinpath(
        results_dir,
        "cg_af_results_$(instance_tag)_L$(L)_S$(S)_K$(K)_$(experiment).json"
    )
    write_json(summary_file, summary)
    
    iteration_file = joinpath(
        results_dir,
        "cg_af_iterations_$(instance_tag)_L$(L)_S$(S)_K$(K)_$(experiment).json"
    )
    write_json(iteration_file, get(results, "iteration_details", []))
    
    println("📁 CG results JSON saved to: $summary_file")
    println("📁 CG iteration JSON saved to: $iteration_file")
end

