# CG_AF_FL.jl
# Entry point for Facility Location Column Generation
# Usage: julia CG_AF_FL.jl <instance_file> <scenarios_file> <binary_vectors_file> <K> <S> [pricing_type] [unmet_pen] [scaling_factor] [capacity_factor]

using Printf
using Dates

# Include required files
include("CGUtilities.jl")
include("AssignmentFormulation.jl")
include("FacilityLocationAF.jl")

# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 5
        println("❌ Error: Incorrect number of arguments!")
        println("Usage: julia CG_AF_FL.jl <instance_file> <scenarios_file> <binary_vectors_file> <K> <S> [pricing_type] [unmet_pen] [scaling_factor] [capacity_factor]")
        println("\nArguments:")
        println("  instance_file: Instance file path (e.g., Facility_Location/instances/cap91.txt)")
        println("  scenarios_file: File containing SAA samples (e.g., Samples/FL/Scenarios/S25_bernoulli.txt)")
        println("  binary_vectors_file: File containing initial binary vectors (e.g., Samples/FL/Binary_vectors/B25_bernoulli.txt)")
        println("  K: Number of solutions to select")
        println("  S: Number of scenarios (should match scenarios_file)")
        println("  pricing_type: 1 or 2 (default: 2)")
        println("  unmet_pen: Unmet demand penalty (default: 15.0)")
        println("  scaling_factor: Scaling factor for transportation cost (default: 0.2)")
        println("  capacity_factor: Capacity factor (default: 0.3)")
        println("\nExample:")
        println("  julia CG_AF_FL.jl Facility_Location/instances/cap91.txt Samples/FL/Scenarios/S25_bernoulli.txt Samples/FL/Binary_vectors/B25_bernoulli.txt 2 25")
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
    unmet_pen = length(ARGS) > 6 ? parse(Float64, ARGS[7]) : 15.0
    scaling_factor = length(ARGS) > 7 ? parse(Float64, ARGS[8]) : 0.2
    capacity_factor = length(ARGS) > 8 ? parse(Float64, ARGS[9]) : 0.3
    
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
    println("Parameters: K=$K, S=$S, pricing_type=$pricing_type")
    println("Problem parameters: unmet_pen=$unmet_pen, scaling_factor=$scaling_factor, capacity_factor=$capacity_factor")
    println("="^80)
    println()
    
    # Load instance data
    println("Loading instance data...")
    fl_data = load_FL_data(instance_file; capacity_factor=capacity_factor)
    fl_data["unmet_pen"] = unmet_pen
    fl_data["scaling_factor"] = scaling_factor
    println("✅ Instance loaded: $(fl_data["nloc"]) facilities, $(fl_data["ncust"]) customers")
    println()
    
    # Load scenarios from file
    println("Loading scenarios from file...")
    scenarios = load_scenarios_from_file(scenarios_file)
    if length(scenarios) != S
        println("⚠️  Warning: Expected $S scenarios, but loaded $(length(scenarios))")
        S = length(scenarios)  # Update S to match loaded scenarios
    end
    println()
    
    # Load initial binary vectors from file
    println("Loading initial binary vectors from file...")
    initial_binary_vectors = load_binary_vectors_from_file(binary_vectors_file)
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
    master_log_file = joinpath(logs_dir, "CG_AF_FL_$(instance_name)_K$(K)_S$(S)_master_$(timestamp).log")
    pricing_log_file = joinpath(logs_dir, "CG_AF_FL_$(instance_name)_K$(K)_S$(S)_pricing_$(timestamp).log")
    
    # Initialize log files with headers
    open(master_log_file, "w") do file
        println(file, "="^80)
        println(file, "COLUMN GENERATION - MASTER PROBLEM LOGS")
        println(file, "="^80)
        println(file, "Instance: $instance_file")
        println(file, "K: $K, S: $S")
        println(file, "Pricing Type: $pricing_type")
        println(file, "Started: $(now())")
        println(file, "="^80)
    end
    
    open(pricing_log_file, "w") do file
        println(file, "="^80)
        println(file, "COLUMN GENERATION - PRICING SUBPROBLEM LOGS")
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
end

