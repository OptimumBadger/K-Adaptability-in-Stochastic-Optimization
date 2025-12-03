# CG_AF_TS.jl
# Entry point for Transmission Switching Column Generation
# Usage: julia CG_AF_TS.jl <instance_file> <scenarios_file> <binary_vectors_file> <K> <S> [pricing_type] [line_capacity_factor]

using Printf

# Include required files
include("CGUtilities.jl")
include("TransmissionSwitchingCG.jl")

# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 5
        println("❌ Error: Incorrect number of arguments!")
        println("Usage: julia CG_AF_TS.jl <instance_file> <scenarios_file> <binary_vectors_file> <K> <S> [pricing_type] [line_capacity_factor]")
        println("\nArguments:")
        println("  instance_file: Instance file path (e.g., Transmission_Switching/instances/case118Blumsack.m)")
        println("  scenarios_file: File containing SAA samples (e.g., Transmission_Switching/Heuristic/SAA_samples_S25.txt)")
        println("  binary_vectors_file: File containing initial binary vectors (e.g., Transmission_Switching/Heuristic/binary_vectors_L25.txt)")
        println("  K: Number of solutions to select")
        println("  S: Number of scenarios (should match scenarios_file)")
        println("  pricing_type: 1 or 2 (default: 2)")
        println("  line_capacity_factor: Line capacity factor (default: 1.0)")
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
        pricing_type = pricing_type,
        use_solution_pool = true,
        total_work_budget = 30000.0,
        opt_worklimit = DEFAULT_OPT_WORKLIMIT,
        master_log_file = "CG_AF_TS_master.log",
        pricing_log_file = "CG_AF_TS_pricing.log"
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

