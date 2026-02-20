# CG_AF_FL.jl
# Entry point for Facility Location Column Generation
# Usage: julia CG_AF_FL.jl <instance_file> <scenarios_file> <binary_vectors_file> <L> <S> <K> [pricing_type] [unmet_pen] [scaling_factor] [capacity_factor] [total_work_budget] [decomp_mode]

using Printf
using Dates

# Include required files
include("CGUtilities.jl")
include("AssignmentFormulation.jl")
include("FacilityLocationAF.jl")

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

function parse_bool_flag(raw::String)
    v = lowercase(strip(raw))
    return v in ("1", "true", "yes", "y", "on")
end

# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 6
        println("❌ Error: Incorrect number of arguments!")
        println("Usage: julia CG_AF_FL.jl <instance_file> <scenarios_file> <binary_vectors_file> <L> <S> <K> [pricing_type] [unmet_pen] [scaling_factor] [capacity_factor] [total_work_budget] [decomp_mode]")
        println("\nArguments:")
        println("  instance_file: Instance file path (e.g., Facility_Location/instances/cap91.txt)")
        println("  scenarios_file: File containing SAA samples (e.g., Samples/FL/Scenarios/S300_bernoulli.txt)")
        println("  binary_vectors_file: File containing binary vectors (e.g., Samples/FL/Binary_vectors/B300_bernoulli.txt)")
        println("  L: Number of candidate policies to use (first L from binary_vectors_file)")
        println("  S: Number of scenarios to use (first S from scenarios_file)")
        println("  K: Number of solutions to select")
        println("  pricing_type: 1 or 2 (default: 2)")
        println("  unmet_pen: Unmet demand penalty (default: 15.0)")
        println("  scaling_factor: Scaling factor for transportation cost (default: 0.2)")
        println("  capacity_factor: Capacity factor (default: 0.3)")
        println("  total_work_budget: Total work unit cap for CG (default: 30000.0)")
        println("  decomp_mode: Decomposition mode for Type-2 pricing (default: 0)")
        println("               0 = No decomposition (standard pricing)")
        println("               1 = Simple LP-based decomposition")
        println("               2 = Dual-based unfixing decomposition")
        println("               3 = Partial MIP with F1=open facilities, F0=empty")
        println("               4 = Partial MIP with random subsets of F0 and F1")
        println("               5 = Gap-based dual unfixing (unfix until sum|dual| < gap)")
        println("\nExample:")
        println("  julia CG_AF_FL.jl Facility_Location/instances/cap91.txt Samples/FL/Scenarios/S300_bernoulli.txt Samples/FL/Binary_vectors/B300_bernoulli.txt 50 30 2")
        exit(1)
    end
    
    # Parse required arguments
    instance_file = ARGS[1]
    scenarios_file = ARGS[2]
    binary_vectors_file = ARGS[3]
    
    # Parse L (number of candidate policies to use)
    L = try
        parsed = parse(Int, ARGS[4])
        if parsed <= 0
            throw(ArgumentError("L must be positive"))
        end
        parsed
    catch
        println("❌ Error: L must be a positive integer!")
        println("You provided: '$(ARGS[4])'")
        exit(1)
    end
    
    # Parse S (number of scenarios to use)
    S = try
        parsed = parse(Int, ARGS[5])
        if parsed <= 0
            throw(ArgumentError("S must be positive"))
        end
        parsed
    catch
        println("❌ Error: S must be a positive integer!")
        println("You provided: '$(ARGS[5])'")
        exit(1)
    end
    
    # Parse K (number of solutions to select)
    K = try
        parsed = parse(Int, ARGS[6])
        if parsed <= 0
            throw(ArgumentError("K must be positive"))
        end
        parsed
    catch
        println("❌ Error: K must be a positive integer!")
        println("You provided: '$(ARGS[6])'")
        exit(1)
    end
    
    # Parse optional arguments
    pricing_type = length(ARGS) > 6 ? parse(Int, ARGS[7]) : 2
    unmet_pen = length(ARGS) > 7 ? parse(Float64, ARGS[8]) : 15.0
    scaling_factor = length(ARGS) > 8 ? parse(Float64, ARGS[9]) : 0.2
    capacity_factor = length(ARGS) > 9 ? parse(Float64, ARGS[10]) : 0.3
    total_work_budget = length(ARGS) > 10 ? parse(Float64, ARGS[11]) : 30000.0
    decomp_mode = length(ARGS) > 11 ? parse(Int, ARGS[12]) : 0
    
    # Validate decomp_mode
    if !(decomp_mode in [0, 1, 2, 3, 4, 5])
        println("❌ Error: decomp_mode must be 0, 1, 2, 3, 4, or 5!")
        println("  0 = No decomposition (standard pricing)")
        println("  1 = Simple LP-based decomposition")
        println("  2 = Dual-based unfixing decomposition")
        println("  3 = Partial MIP with F1=open facilities, F0=empty")
        println("  4 = Partial MIP with random subsets of F0 and F1")
        println("  5 = Gap-based dual unfixing (unfix until sum|dual| < gap)")
        exit(1)
    end
    
    # Set use_decomp and use_simple_decomp based on decomp_mode
    use_decomp = decomp_mode in [1, 2, 3, 4, 5]
    use_simple_decomp = decomp_mode == 1
    
    # Validate pricing_type
    if !(pricing_type in [1, 2])
        println("❌ Error: pricing_type must be 1 or 2!")
        exit(1)
    end
    
    println("="^80)
    println("COLUMN GENERATION - FACILITY LOCATION")
    println("="^80)
    println("Instance: $instance_file")
    println("Scenarios file: $scenarios_file")
    println("Binary vectors file: $binary_vectors_file")
    println("Parameters: L=$L, S=$S, K=$K, pricing_type=$pricing_type")
    println("Problem parameters: unmet_pen=$unmet_pen, scaling_factor=$scaling_factor, capacity_factor=$capacity_factor")
    decomp_mode_str = if decomp_mode == 0
        "standard (no decomposition)"
    elseif decomp_mode == 1
        "simple LP-based decomposition"
    elseif decomp_mode == 2
        "dual-based unfixing decomposition"
    elseif decomp_mode == 3
        "partial MIP with F1=open facilities, F0=empty"
    elseif decomp_mode == 4
        "partial MIP with random subsets of F0 and F1"
    else  # decomp_mode == 5
        "gap-based dual unfixing (unfix until sum|dual| < gap)"
    end
    println("Pricing mode: $decomp_mode_str")
    println("="^80)
    println()
    
    # Load instance data
    println("Loading instance data...")
    fl_data = load_FL_data(instance_file; capacity_factor=capacity_factor)
    fl_data["unmet_pen"] = unmet_pen
    fl_data["scaling_factor"] = scaling_factor
    println("✅ Instance loaded: $(fl_data["nloc"]) facilities, $(fl_data["ncust"]) customers")
    println()
    
    # Load scenarios from file and slice first S
    println("Loading scenarios from file...")
    all_scenarios = load_scenarios_from_file(scenarios_file)
    
    # Validate S and slice first S scenarios
    if S > length(all_scenarios)
        error("❌ Error: Requested S = $S scenarios, but scenarios_file contains only $(length(all_scenarios)) scenarios.")
    end
    scenarios = all_scenarios[1:S]
    println("✅ Loaded $(length(scenarios)) scenarios (first S from file)")
    println()
    
    # Load binary vectors from file and slice first L
    println("Loading binary vectors from file...")
    all_binary_vectors = load_binary_vectors_from_file(binary_vectors_file)
    
    # Validate L and slice first L vectors
    if L > length(all_binary_vectors)
        error("❌ Error: Requested L = $L candidate policies, but binary_vectors_file contains only $(length(all_binary_vectors)) vectors.")
    end
    initial_binary_vectors = all_binary_vectors[1:L]
    println("✅ Loaded $(length(initial_binary_vectors)) binary vectors (first L from file)")
    println()
    
    # For decomposition: Load all 300 binary vectors (default L=300)
    if use_decomp
        # Load all vectors from file (up to 300)
        all_initial_vectors = load_binary_vectors_from_file(binary_vectors_file)
        # Take first 300 (or all if less than 300)
        num_vectors_to_use = min(300, length(all_initial_vectors))
        initial_vectors_for_decomp = all_initial_vectors[1:num_vectors_to_use]
        fl_data["initial_binary_vectors"] = initial_vectors_for_decomp
        println("✅ Loaded $(length(initial_vectors_for_decomp)) initial binary vectors for decomposition cuts")
        println()
    end
    
    # Extract f_s values from WS file
    println("Extracting f_s values (wait-and-see costs) from WS file...")
    # Determine WS file path from scenarios_file
    # scenarios_file format: Samples/FL/Scenarios/S300_bernoulli.txt
    # WS file format: Samples/FL/Scenarios/instance_X/WS300_bernoulli.txt
    scenarios_dir = dirname(scenarios_file)
    experiment = extract_experiment_type(scenarios_file)
    
    # Try to extract instance number from instance_file
    instance_name = lowercase(replace(splitext(basename(instance_file))[1], "." => "_"))
    instance_num = "1"  # default
    if occursin("cap91", instance_name) || occursin("instance_1", instance_name) || occursin("instance1", instance_name)
        instance_num = "1"
    elseif occursin("cap101", instance_name) || occursin("instance_2", instance_name) || occursin("instance2", instance_name)
        instance_num = "2"
    end
    
    # Try multiple possible WS file paths
    ws_file_candidates = [
        joinpath(scenarios_dir, "WS300_$(experiment).txt"),  # same directory as scenarios (most common)
        joinpath(scenarios_dir, "instance_$(instance_num)", "WS300_$(experiment).txt"),  # instance_X subdirectory (if scenarios not in instance_X)
        joinpath(dirname(scenarios_dir), "instance_$(instance_num)", "WS300_$(experiment).txt"),  # parent/instance_X
    ]
    
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
    fl_data["f_s"] = f_s
    fl_data["use_decomp"] = use_decomp
    fl_data["use_simple_decomp"] = use_simple_decomp
    fl_data["decomp_mode"] = decomp_mode
    
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
    master_log_file = joinpath(logs_dir, "CG_AF_FL_$(instance_name)_L$(L)_S$(S)_K$(K)_M$(decomp_mode)_master_$(timestamp).log")
    pricing_log_file = joinpath(logs_dir, "CG_AF_FL_$(instance_name)_L$(L)_S$(S)_K$(K)_M$(decomp_mode)_pricing_$(timestamp).log")
    
    # Initialize log files with headers
    pricing_type_str = if pricing_type == 1
        "Type 1"
    elseif pricing_type == 2
        if decomp_mode == 0
            "Type 2 (Standard)"
        elseif decomp_mode == 1
            "Type 2 (Decomposition - Simple LP-based)"
        elseif decomp_mode == 2
            "Type 2 (Decomposition - Dual-based Unfixing)"
        elseif decomp_mode == 3
            "Type 2 (Decomposition - Partial MIP F1=open, F0=empty)"
        elseif decomp_mode == 4
            "Type 2 (Decomposition - Partial MIP Random F0/F1)"
        else  # decomp_mode == 5
            "Type 2 (Decomposition - Gap-based Dual Unfixing)"
        end
    else
        "Type $pricing_type"
    end
    
    open(master_log_file, "w") do file
        println(file, "="^80)
        println(file, "COLUMN GENERATION - MASTER PROBLEM LOGS")
        println(file, "="^80)
        println(file, "Instance: $instance_file")
        println(file, "L: $L, S: $S, K: $K")
        println(file, "Pricing Type: $pricing_type_str")
        println(file, "Started: $(now())")
        println(file, "="^80)
    end
    
    open(pricing_log_file, "w") do file
        println(file, "="^80)
        println(file, "COLUMN GENERATION - PRICING SUBPROBLEM LOGS")
        println(file, "="^80)
        println(file, "Instance: $instance_file")
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
        fl_data,
        K,
        500;  # max_iterations
        evaluate_cost = evaluate_cost_FL,
        build_pricing_model = build_pricing_model_AF_FL,
        calculate_m_values = calculate_m_values_FL,
        evaluate_cost_batch = evaluate_cost_batch_template_FL,  # Use optimized batch evaluation
        pricing_type = pricing_type,
        use_solution_pool = true,
        total_work_budget = total_work_budget,
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
    results_dir = joinpath(project_root, "Results", "Facility_Location")
    if !isdir(results_dir)
        mkpath(results_dir)
    end
    
    experiment = extract_experiment_type(scenarios_file)
    instance_tag = replace(splitext(basename(instance_file))[1], "." => "_")
    
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
        "final_objective_ip" => results["final_objective_ip"],
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

