# Create_Partitions.jl
# Creates initial partitions for SF column generation by assigning scenarios to K candidate vectors
# Usage: julia Create_Partitions.jl <instance_file> <scenarios_file> <candidates_file> [unmet_pen] [scaling_factor] [capacity_factor]

using JuMP, Gurobi
using MathOptInterface
const MOI = MathOptInterface

# Include required files
include("FLUtilities.jl")  # For read_orlib_cap, load_scenarios_from_file, load_binary_vectors_from_file, and evaluate_cost_FL

# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 3
        println("❌ Error: Incorrect number of arguments!")
        println("Usage: julia Create_Partitions.jl <instance_file> <scenarios_file> <candidates_file> [unmet_pen] [scaling_factor] [capacity_factor]")
        println("\nArguments:")
        println("  instance_file: Instance file path (e.g., Facility_Location/instances/cap91.txt)")
        println("  scenarios_file: File containing scenarios (e.g., Samples/FL/Scenarios/S25_bernoulli.txt)")
        println("  candidates_file: File containing K candidate binary vectors (e.g., Facility_Location/Heuristic/selected_candidates_S50_K2.txt)")
        println("  unmet_pen: Unmet demand penalty (default: 15.0)")
        println("  scaling_factor: Scaling factor for transportation cost (default: 0.2)")
        println("  capacity_factor: Capacity factor (default: 0.3)")
        println("\nOutput:")
        println("  Creates SubsetS#_K#_FL.txt in Samples/FL/ directory")
        println("\nExample:")
        println("  julia Create_Partitions.jl Facility_Location/instances/cap91.txt Samples/FL/Scenarios/S25_bernoulli.txt Facility_Location/Heuristic/selected_candidates_S50_K2.txt")
        exit(1)
    end
    
    # Parse arguments
    instance_file = ARGS[1]
    scenarios_file = ARGS[2]
    candidates_file = ARGS[3]
    unmet_pen = length(ARGS) > 3 ? parse(Float64, ARGS[4]) : 15.0
    scaling_factor = length(ARGS) > 4 ? parse(Float64, ARGS[5]) : 0.2
    capacity_factor = length(ARGS) > 5 ? parse(Float64, ARGS[6]) : 0.3
    
    println("="^80)
    println("CREATE PARTITIONS FOR SF COLUMN GENERATION")
    println("="^80)
    println("Instance: $instance_file")
    println("Scenarios file: $scenarios_file")
    println("Candidates file: $candidates_file")
    println("Parameters: unmet_pen=$unmet_pen, scaling_factor=$scaling_factor, capacity_factor=$capacity_factor")
    println("="^80)
    println()
    
    # Handle instance file path (make absolute if relative)
    script_dir = @__DIR__
    project_root = dirname(dirname(script_dir))  # Go up two levels: Samples/FL -> Samples -> project root
    if !isabspath(instance_file)
        # If relative, assume it's relative to Facility_Location/instances/
        if !isfile(instance_file)
            instance_file = joinpath(project_root, "Facility_Location", "instances", basename(instance_file))
        end
    end
    
    # Handle scenarios file path
    if !isabspath(scenarios_file)
        if !isfile(scenarios_file)
            scenarios_file = joinpath(script_dir, "Scenarios", basename(scenarios_file))
        end
    end
    
    # Handle candidates file path
    if !isabspath(candidates_file)
        if !isfile(candidates_file)
            # Try in Facility_Location/Heuristic/
            candidates_file = joinpath(project_root, "Facility_Location", "Heuristic", basename(candidates_file))
        end
    end
    
    # Load instance data
    println("Loading instance data...")
    capacity, fixedcost, cost, demandmean = read_orlib_cap(instance_file)
    capacity = capacity .* capacity_factor
    
    nloc = length(capacity)
    ncust = length(demandmean)
    
    problem_data = Dict(
        "capacity" => capacity,
        "fixedcost" => fixedcost,
        "cost" => cost,
        "demandmean" => demandmean,
        "nloc" => nloc,
        "ncust" => ncust,
        "unmet_pen" => unmet_pen,
        "scaling_factor" => scaling_factor
    )
    println("✅ Instance loaded: $nloc facilities, $ncust customers")
    println()
    
    # Load scenarios
    println("Loading scenarios from file...")
    scenarios = load_scenarios_from_file(scenarios_file)
    nscen = length(scenarios)
    println("✅ Loaded $nscen scenarios")
    println()
    
    # Load candidate binary vectors
    println("Loading candidate binary vectors from file...")
    candidates = load_binary_vectors_from_file(candidates_file)
    K = length(candidates)
    println("✅ Loaded $K candidate vectors")
    
    # Validate candidate vectors
    for (idx, candidate) in enumerate(candidates)
        if length(candidate) != nloc
            error("❌ Error: Candidate vector $idx has length $(length(candidate)) but expected $nloc")
        end
    end
    println()
    
    # Assign each scenario to the candidate that gives lowest cost
    println("Assigning scenarios to candidates based on lowest cost...")
    partitions = [Int[] for _ in 1:K]  # K partitions, each contains list of scenario indices
    
    for s in 1:nscen
        best_cost = Inf
        best_candidate = 1
        
        # Evaluate cost of serving scenario s with each candidate
        for k in 1:K
            cost_val, _ = evaluate_cost_FL(candidates[k], scenarios[s], problem_data)
            if cost_val < best_cost
                best_cost = cost_val
                best_candidate = k
            end
        end
        
        # Assign scenario s to best candidate (scenario indices are 1-based)
        push!(partitions[best_candidate], s)
        
        if s % 10 == 0
            println("  Processed $s scenarios...")
        end
    end
    
    println("✅ Completed assignment of all scenarios")
    println()
    
    # Print summary
    println("Partition Summary:")
    for k in 1:K
        println("  Partition $k: $(length(partitions[k])) scenarios")
    end
    println()
    
    # Determine output file name based on scenarios file and K
    # Extract S and experiment type from scenarios file name (e.g., S25_bernoulli.txt -> S=25, type=bernoulli)
    scenarios_basename = basename(scenarios_file)
    scenarios_name = splitext(scenarios_basename)[1]  # e.g., "S25_bernoulli"
    
    # Extract S and experiment type
    if startswith(scenarios_name, "S")
        parts = split(scenarios_name, "_")
        S_str = parts[1][2:end]  # Remove "S" prefix
        experiment_type = length(parts) > 1 ? parts[2] : "unknown"
    else
        S_str = string(nscen)
        experiment_type = "unknown"
    end
    
    # Create output file name: SubsetS#K#FL.txt in the same directory as this script
    script_dir = @__DIR__
    output_filename = "SubsetS$(S_str)_K$(K)_FL.txt"
    output_file_path = joinpath(script_dir, output_filename)
    
    # Write output file (scenario indices per partition)
    # Format: Each line contains space-separated scenario indices (1-based) assigned to that partition
    println("Writing partitions to output file: $output_file_path")
    open(output_file_path, "w") do io
        for k in 1:K
            # Write scenario indices for partition k (space-separated)
            if length(partitions[k]) > 0
                println(io, join(partitions[k], " "))
            else
                println(io, "")  # Empty line if partition is empty
            end
        end
    end
    println("✅ Partitions written to $output_file_path")
    println()
    
    println("="^80)
    println("PARTITION CREATION COMPLETE")
    println("="^80)
    println("Output file: $output_file_path")
    println("Format: Scenario indices per partition (one partition per line)")
    println("  Each line contains space-separated scenario indices (1-based)")
    println("  Line k contains the scenario indices assigned to partition k")
    println("="^80)
end

