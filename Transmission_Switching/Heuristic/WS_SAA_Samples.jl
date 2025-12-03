# WS_SAA_Samples.jl
# Wait-and-See: Solve SAA samples from scratch
# This file solves each scenario from scratch and computes the average objective value

using PowerModels
using JuMP, Gurobi
using MathOptInterface
const MOI = MathOptInterface
using DataFrames, XLSX, Random, Distributions
using Printf
using Statistics
using Dates

# Include utility functions from Candidate_List_Generation_Phase.jl
include("Candidate_List_Generation_Phase.jl")

# Include functions from Assignment_Phase.jl for load_samples_from_file
include("Assignment_Phase.jl")

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
                           log_dir::String="logs")
    """
    Compute wait-and-see cost by solving each scenario from scratch
    
    Args:
        instance_file: Name of the instance file (e.g., "case118Blumsack.m")
        samples_file: Path to file containing S samples
        log_dir: Directory for log files
    
    Returns:
        average_cost: Average objective value across all scenarios
        costs: List of costs for each scenario
        total_work_units: Total work units used
        successful_solves: Number of successful solves
        failed_solves: Number of failed solves
    """
    
    println("="^80)
    println("WAIT-AND-SEE COST COMPUTATION")
    println("="^80)
    println("Instance: $instance_file")
    println("Samples file: $samples_file")
    println("="^80)
    
    # Get the directory where this script is located
    script_dir = @__DIR__
    project_root = dirname(dirname(script_dir))  # Go up two levels to project root
    
    # Construct file paths
    instance_path = joinpath(project_root, "Transmission_Switching", "instances", instance_file)
    
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
    
    # Extract case name from instance file
    case_name = extract_case_name(instance_file)
    println("Extracted case name: $case_name")
    
    # Load network data
    println("Loading network from: $instance_path")
    network = PowerModels.parse_file(instance_path)
    println("✅ Network loaded: $(length(network["bus"])) buses, $(length(network["branch"])) branches, $(length(network["gen"])) generators")
    
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
    total_work_units = 0.0
    successful_solves = 0
    failed_solves = 0
    
    for (i, sample) in enumerate(samples)
        log_file = joinpath(log_path, "wait_and_see_scenario$(i)_$(timestamp).log")
        
        obj_value, binary_vector, penalty_flag, penalty_cost, work_units, theta_values = offline_calc_direct(network, sample, log_file)
        
        if obj_value != -1
            push!(costs, obj_value)
            total_work_units += work_units
            successful_solves += 1
        else
            push!(costs, Inf)  # Mark as failed
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
    
    # Save wait-and-see costs to file: wait_and_see_costs_S<S>.txt
    costs_file = joinpath(script_dir, "wait_and_see_costs_S$(S).txt")
    
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
    
    println("\n✅ Wait-and-see computation completed!")
    
    return average_cost, costs, total_work_units, successful_solves, failed_solves, costs_file
end

# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 2
        println("❌ Error: Incorrect number of arguments!")
        println("Usage: julia WS_SAA_Samples.jl <instance_file> <samples_file>")
        println("  instance_file: Instance file name (e.g., case118Blumsack.m)")
        println("  samples_file: Path to file containing S samples")
        println("\nExample:")
        println("  julia WS_SAA_Samples.jl case118Blumsack.m SAA_samples.txt")
        exit(1)
    end
    
    instance_file = ARGS[1]
    samples_file = ARGS[2]
    
    # Run wait-and-see cost computation
    average_cost, costs, total_work_units, successful_solves, failed_solves, costs_file = wait_and_see_cost(
        instance_file, samples_file
    )
    
    println("\n" * "="^80)
    println("WAIT-AND-SEE COST COMPUTATION COMPLETED")
    println("="^80)
    println("Average wait-and-see cost: $(round(average_cost, digits=2))")
    println("Successful solves: $successful_solves")
    println("Failed solves: $failed_solves")
    println("Results saved to: $costs_file")
end

