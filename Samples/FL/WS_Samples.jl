# WS_Samples.jl
# Wait-and-See: Solve SAA samples from scratch for Facility Location
# This file solves each scenario from scratch and computes the average objective value

using JuMP, Gurobi
using MathOptInterface
const MOI = MathOptInterface
using Random, Distributions
using Printf
using Statistics
using Dates

# Get project root for includes
script_dir = @__DIR__
project_root = dirname(dirname(script_dir))  # Go up two levels to project root

# Include utility functions from Candidate_List_Generation_Phase.jl
include(joinpath(project_root, "Facility_Location", "Heuristic", "Candidate_List_Generation_Phase.jl"))

# Include functions from Assignment_Phase.jl for load_samples_from_file
include(joinpath(project_root, "Facility_Location", "Heuristic", "Assignment_Phase.jl"))

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

function load_wait_and_see_costs_from_file(file_path::String)
    """Load wait-and-see costs from a file
    
    Args:
        file_path: Path to the wait-and-see costs file
    
    Returns:
        average_cost: Average cost (from header)
        costs: List of costs (one per line, Inf for failed solves)
        successful_solves: Number of successful solves (from header)
        failed_solves: Number of failed solves (from header)
        total_work_units: Total work units (from header)
    """
    if !isfile(file_path)
        error("❌ Error: Wait-and-see costs file not found! Expected: $file_path")
    end
    
    costs = Float64[]
    average_cost = Inf
    successful_solves = 0
    failed_solves = 0
    total_work_units = 0.0
    
    open(file_path, "r") do file
        for line in eachline(file)
            # Skip comments and empty lines
            if startswith(strip(line), "#") || strip(line) == ""
                # Try to extract metadata from comments
                if occursin("Average cost:", line)
                    match_result = match(r"Average cost: ([\d.]+|Inf)", line)
                    if match_result !== nothing
                        avg_str = match_result.captures[1]
                        if avg_str != "Inf"
                            average_cost = parse(Float64, avg_str)
                        end
                    end
                elseif occursin("Successful solves:", line)
                    match_result = match(r"Successful solves: (\d+)", line)
                    if match_result !== nothing
                        successful_solves = parse(Int, match_result.captures[1])
                    end
                elseif occursin("Failed solves:", line)
                    match_result = match(r"Failed solves: (\d+)", line)
                    if match_result !== nothing
                        failed_solves = parse(Int, match_result.captures[1])
                    end
                elseif occursin("Total work units:", line)
                    match_result = match(r"Total work units: ([\d.]+)", line)
                    if match_result !== nothing
                        total_work_units = parse(Float64, match_result.captures[1])
                    end
                end
                continue
            end
            
            # Parse cost value
            cost_str = strip(line)
            if cost_str == "Inf"
                push!(costs, Inf)
            else
                push!(costs, parse(Float64, cost_str))
            end
        end
    end
    
    return average_cost, costs, successful_solves, failed_solves, total_work_units
end

# =============================================================================
# MAIN FUNCTION: WAIT-AND-SEE COST COMPUTATION
# =============================================================================

function wait_and_see_cost(instance_file::String, samples_file::String;
                           experiment_type::String="low",
                           unmet_pen::Float64=15.0,
                           scaling_factor::Float64=0.2,
                           capacity_factor::Float64=0.3,
                           log_dir::String="logs")
    """
    Compute wait-and-see cost by solving each scenario from scratch
    
    Args:
        instance_file: Name of the instance file (e.g., "cap91.txt")
        samples_file: Path to file containing S samples
        experiment_type: "low", "high", or "bernoulli"
        unmet_pen: Penalty for unmet demand
        scaling_factor: Scaling factor for transportation cost
        capacity_factor: Factor to scale facility capacities
        log_dir: Directory for log files
    
    Returns:
        average_cost: Average objective value across all scenarios
        costs: List of costs for each scenario
        total_work_units: Total work units used
        successful_solves: Number of successful solves
        failed_solves: Number of failed solves
        costs_file: Path to the saved costs file
    """
    
    println("="^80)
    println("WAIT-AND-SEE COST COMPUTATION - FACILITY LOCATION")
    println("="^80)
    println("Instance: $instance_file")
    println("Samples file: $samples_file")
    println("="^80)
    
    # Get the directory where this script is located
    script_dir = @__DIR__
    project_root = dirname(dirname(script_dir))  # Go up two levels to project root
    
    # Construct file paths
    instance_path = joinpath(project_root, "Facility_Location", "instances", instance_file)
    
    # Ensure samples_file is absolute or relative to project root
    if !isabspath(samples_file)
        # If relative, assume it's relative to Samples/FL/Scenarios/
        if !isfile(samples_file)
            samples_file = joinpath(script_dir, "Scenarios", basename(samples_file))
        end
    end
    
    # Create log directory if it doesn't exist
    log_path = joinpath(script_dir, log_dir)
    if !isdir(log_path)
        mkpath(log_path)
        println("Created log directory: $log_path")
    end
    
    # Check if files exist
    if !isfile(instance_path)
        error("❌ Error: Instance file not found! Expected: $instance_path")
    end
    
    if !isfile(samples_file)
        error("❌ Error: Samples file not found! Expected: $samples_file")
    end
    
    println("\n" * "="^40)
    println("STEP 1: LOADING DATA")
    println("="^40)
    
    # Load instance data
    println("Loading instance from: $instance_path")
    capacity, fixedcost, cost, demandmean = read_orlib_cap(instance_path)
    
    # Apply capacity factor
    capacity = capacity .* capacity_factor
    println("✅ Instance loaded: $(length(capacity)) facilities, $(length(demandmean)) customers")
    
    nloc = length(capacity)
    ncust = length(demandmean)
    
    # Load samples
    println("\nLoading samples from: $samples_file")
    samples = load_samples_from_file(samples_file)
    S = length(samples)
    println("✅ Loaded $S samples")
    
    println("\n" * "="^40)
    println("STEP 2: SOLVING SCENARIOS FROM SCRATCH")
    println("="^40)
    println("Solving each scenario from scratch...")
    
    # Generate timestamp for this run
    timestamp = Dates.format(now(), "YYYYmmdd_HHMMSS")
    
    # Compute wait-and-see cost: solve each scenario from scratch
    costs = Float64[]
    binary_vectors = Vector{Vector{Int}}()  # Store binary vectors from each scenario
    total_work_units = 0.0
    successful_solves = 0
    failed_solves = 0
    
    for (i, sample) in enumerate(samples)
        log_file = joinpath(log_path, "wait_and_see_scenario$(i)_$(timestamp).log")
        
        obj_value, binary_vector, penalty_flag, penalty_cost, work_units = offline_calc_direct(
            nloc, ncust, capacity, cost, sample, fixedcost, 
            unmet_pen, scaling_factor, log_file
        )
        
        if obj_value != -1
            push!(costs, obj_value)
            # Store binary vector (convert to Int if needed)
            push!(binary_vectors, round.(Int, binary_vector))
            total_work_units += work_units
            successful_solves += 1
        else
            push!(costs, Inf)  # Mark as failed
            # For failed solves, push empty vector or zeros (we'll skip these when saving)
            push!(binary_vectors, Int[])  # Empty vector for failed solves
            failed_solves += 1
        end
        
        if i % 10 == 0
            println("  Processed $i scenarios")
        end
    end
    
    # Compute average wait-and-see cost (only from successful solves)
    successful_costs = [c for c in costs if isfinite(c)]
    average_cost = length(successful_costs) > 0 ? mean(successful_costs) : Inf
    
    println("✅ Wait-and-see cost computation completed")
    println("   Total scenarios: $S")
    println("   Successful solves: $successful_solves")
    println("   Failed solves: $failed_solves")
    println("   Average wait-and-see cost: $(round(average_cost, digits=2))")
    println("   Total work units: $(round(total_work_units, digits=2))")
    println("   Log files stored in: $log_path")
    
    # Save wait-and-see costs to file: WS<S>_<experiment_type>.txt in Scenarios folder
    costs_file = joinpath(script_dir, "Scenarios", "WS$(S)_$(experiment_type).txt")
    if !isdir(dirname(costs_file))
        mkpath(dirname(costs_file))
    end
    
    open(costs_file, "w") do file
        # Write header
        println(file, "# Wait-and-See Costs")
        println(file, "# Average cost: $average_cost")
        println(file, "# Successful solves: $successful_solves")
        println(file, "# Failed solves: $failed_solves")
        println(file, "# Total work units: $total_work_units")
        println(file, "#")
        # Write costs (one per line, use "Inf" for failed solves)
        for cost in costs
            if isfinite(cost)
                println(file, cost)
            else
                println(file, "Inf")
            end
        end
    end
    
    println("📁 Wait-and-see costs saved to: $costs_file")
    
    # Save binary vectors to file: B<S>_<experiment_type>.txt in Binary_vectors folder
    binary_vectors_dir = joinpath(script_dir, "Binary_vectors")
    if !isdir(binary_vectors_dir)
        mkpath(binary_vectors_dir)
    end
    binary_vectors_file = joinpath(binary_vectors_dir, "B$(S)_$(experiment_type).txt")
    
    open(binary_vectors_file, "w") do file
        # Only save binary vectors from successful solves
        for (i, vec) in enumerate(binary_vectors)
            if !isempty(vec) && isfinite(costs[i])  # Only save if solve was successful
                # Write vector as space-separated integers
                println(file, join([string(v) for v in vec], " "))
            end
        end
    end
    
    println("📁 Binary vectors saved to: $binary_vectors_file")
    println("   Total vectors saved: $successful_solves (only successful solves)")
    
    println("\n✅ Wait-and-see computation completed!")
    
    return average_cost, costs, total_work_units, successful_solves, failed_solves, costs_file, binary_vectors_file
end

# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 2
        println("❌ Error: Incorrect number of arguments!")
        println("Usage: julia WS_SAA_Samples.jl <instance_file> <samples_file> [experiment_type] [unmet_pen] [scaling_factor] [capacity_factor]")
        println("  instance_file: Instance file name (e.g., cap91.txt)")
        println("  samples_file: Path to file containing S samples")
        println("  experiment_type: 'low', 'high', or 'bernoulli' (default: 'low')")
        println("  unmet_pen: Penalty for unmet demand (default: 15.0)")
        println("  scaling_factor: Scaling factor for transportation cost (default: 0.2)")
        println("  capacity_factor: Factor to scale facility capacities (default: 0.3)")
        println("\nExample:")
        println("  julia WS_SAA_Samples.jl cap91.txt SAA_samples_S25.txt")
        exit(1)
    end
    
    instance_file = ARGS[1]
    samples_file = ARGS[2]
    experiment_type = length(ARGS) > 2 ? ARGS[3] : "low"
    unmet_pen = length(ARGS) > 3 ? parse(Float64, ARGS[4]) : 15.0
    scaling_factor = length(ARGS) > 4 ? parse(Float64, ARGS[5]) : 0.2
    capacity_factor = length(ARGS) > 5 ? parse(Float64, ARGS[6]) : 0.3
    
    # Run wait-and-see cost computation
    average_cost, costs, total_work_units, successful_solves, failed_solves, costs_file, binary_vectors_file = wait_and_see_cost(
        instance_file, samples_file;
        experiment_type=experiment_type,
        unmet_pen=unmet_pen,
        scaling_factor=scaling_factor,
        capacity_factor=capacity_factor
    )
    
    println("\n" * "="^80)
    println("WAIT-AND-SEE COST COMPUTATION COMPLETED")
    println("="^80)
    println("Average wait-and-see cost: $(round(average_cost, digits=2))")
    println("Successful solves: $successful_solves")
    println("Failed solves: $failed_solves")
    println("Costs saved to: $costs_file")
    println("Binary vectors saved to: $binary_vectors_file")
end

