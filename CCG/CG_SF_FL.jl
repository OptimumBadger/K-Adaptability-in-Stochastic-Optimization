# CG_SF_FL.jl
# Entry point for Facility Location Column Generation - Subset Formulation (SF)
# Usage: julia CG_SF_FL.jl <instance_file> <scenarios_file> <initial_subsets_file> <K> <S> [pricing_type] [unmet_pen] [scaling_factor] [capacity_factor]

using Printf
using Dates

# Include required files
include("CGUtilities.jl")
include("SubsetFormulation.jl")
include("FacilityLocationSF.jl")

# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 5
        println("❌ Error: Incorrect number of arguments!")
        println("Usage: julia CG_SF_FL.jl <instance_file> <scenarios_file> <initial_subsets_file> <K> <S> [pricing_type] [unmet_pen] [scaling_factor] [capacity_factor]")
        println("\nArguments:")
        println("  instance_file: Instance file path (e.g., Facility_Location/instances/cap91.txt)")
        println("  scenarios_file: File containing SAA samples (e.g., Samples/FL/Scenarios/S25_bernoulli.txt)")
        println("  initial_subsets_file: File containing initial subset vectors (π vectors) - required")
        println("  K: Number of subsets to select")
        println("  S: Number of scenarios (should match scenarios_file)")
        println("  pricing_type: 1 or 2 (default: 2)")
        println("  unmet_pen: Unmet demand penalty (default: 15.0)")
        println("  scaling_factor: Scaling factor for transportation cost (default: 0.2)")
        println("  capacity_factor: Capacity factor (default: 0.3)")
        println("\nExample:")
        println("  julia CG_SF_FL.jl Facility_Location/instances/cap91.txt Samples/FL/Scenarios/S25_bernoulli.txt Samples/FL/SubsetS25_K2_FL.txt 2 25 2 15.0 0.2 0.3")
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
    unmet_pen = length(ARGS) > 6 ? parse(Float64, ARGS[7]) : 15.0
    scaling_factor = length(ARGS) > 7 ? parse(Float64, ARGS[8]) : 0.2
    capacity_factor = length(ARGS) > 8 ? parse(Float64, ARGS[9]) : 0.3
    
    # Validate pricing_type
    if !(pricing_type in [1, 2])
        println("❌ Error: pricing_type must be 1 or 2!")
        exit(1)
    end
    
    println("="^80)
    println("COLUMN GENERATION SF - FACILITY LOCATION")
    println("="^80)
    println("Instance: $instance_file")
    println("Scenarios file: $scenarios_file")
    println("Initial subsets file: $initial_subsets_file")
    println("Parameters: K=$K, S=$S, pricing_type=$pricing_type")
    println("Problem parameters: unmet_pen=$unmet_pen, scaling_factor=$scaling_factor, capacity_factor=$capacity_factor")
    println("="^80)
    println()
    
    # Load instance data
    println("Loading instance data...")
    capacity, fixedcost, cost, demandmean = read_orlib_cap(instance_file)
    capacity = capacity .* capacity_factor
    
    fl_data = Dict(
        "capacity" => capacity,
        "fixedcost" => fixedcost,
        "cost" => cost,
        "demandmean" => demandmean,
        "nloc" => length(capacity),
        "ncust" => length(demandmean),
        "unmet_pen" => unmet_pen,
        "scaling_factor" => scaling_factor
    )
    println("✅ Instance loaded: $(fl_data["nloc"]) facilities, $(fl_data["ncust"]) customers")
    println()
    
    # Load scenarios from file
    println("Loading scenarios from file...")
    scenarios = load_scenarios_from_file(scenarios_file)
    if length(scenarios) != S
        println("⚠️  Warning: Expected $S scenarios, but loaded $(length(scenarios))")
        S = length(scenarios)  # Update S to match loaded scenarios
    end
    println("✅ Loaded $(length(scenarios)) scenarios")
    println()
    
    # Load initial subsets from file (required)
    println("Loading initial subsets from file...")
    initial_subsets = Vector{Vector{Int}}()
    if isfile(initial_subsets_file)
        # Load subset vectors from file (one line per subset, space-separated scenario indices)
        # Convert scenario indices (1-based) to binary vectors of length S
        open(initial_subsets_file, "r") do io
            for line in eachline(io)
                if !isempty(strip(line))
                    scenario_indices = [parse(Int, x) for x in split(strip(line))]
                    # Convert to binary vector: 1 if scenario is in subset, 0 otherwise
                    binary_vec = zeros(Int, S)
                    for idx in scenario_indices
                        if 1 <= idx <= S
                            binary_vec[idx] = 1
                        else
                            println("⚠️  Warning: Scenario index $idx is out of range [1, $S], ignoring")
                        end
                    end
                    push!(initial_subsets, binary_vec)
                end
            end
        end
        println("✅ Loaded $(length(initial_subsets)) initial subsets")
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
    master_log_file = joinpath(logs_dir, "CG_SF_FL_$(instance_name)_K$(K)_S$(S)_master_$(timestamp).log")
    pricing_log_file = joinpath(logs_dir, "CG_SF_FL_$(instance_name)_K$(K)_S$(S)_pricing_$(timestamp).log")
    
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
        fl_data,
        K,
        500;  # max_iterations
        evaluate_subset_cost = evaluate_subset_cost_FL,
        build_pricing_model = build_pricing_model_SF_FL,
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

