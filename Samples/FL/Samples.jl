# Samples_For_SAA.jl
# Generate SAA samples for Facility Location
# This file generates demand scenarios and saves them to a file

using Random, Distributions
using Dates

# Include local FLUtilities.jl to use read_orlib_cap
include("FLUtilities.jl")

# =============================================================================
# SCENARIO GENERATION FUNCTIONS
# =============================================================================

function generate_demand_scenario(ncust, demandmean, experiment_type="low")
    """Generate a single demand scenario using PowerFlow-style formula
    
    Args:
        ncust: Number of customers
        demandmean: Mean demand for each customer
        experiment_type: "low", "high", or "bernoulli"
    
    Returns:
        scenario_demand: Vector of demand values for each customer
    """
    sigma_range = Dict(
        "low" => (0.1, 0.2),
        "high" => (0.2, 0.3),
        "bernoulli" => (0.1, 0.2)
    )

    if !haskey(sigma_range, lowercase(experiment_type))
        error("Invalid experiment_type: $experiment_type. Must be 'low', 'high', or 'bernoulli'.")
    end

    min_sigma, max_sigma = sigma_range[lowercase(experiment_type)]
    sigma = rand(Uniform(min_sigma, max_sigma))

    scenario_demand = Vector{Float64}(undef, ncust)
    global_noise = randn() # Global random variable

    for j in 1:ncust
        individual_noise = randn() # Individual random variable
        
        # PowerFlow-style demand formula
        demand_val = demandmean[j] + sigma * individual_noise * demandmean[j] + global_noise * sigma * demandmean[j]
        
        # Apply Bernoulli multiplier if experiment_type is "bernoulli"
        if lowercase(experiment_type) == "bernoulli"
            if rand(Bernoulli(0.5)) == 0
                demand_val = 0.0 # Set to zero if Bernoulli is 0
            end
        end
        
        scenario_demand[j] = max(0.0, demand_val) # Ensure non-negativity
    end
    return scenario_demand
end

function generate_facility_location_scenarios(nscen::Int, demandmean::Vector{Float64}, 
                                             experiment_type::String="low")
    """Generate demand scenarios for facility location
    
    Args:
        nscen: Number of scenarios to generate
        demandmean: Mean demand for each customer
        experiment_type: "low", "high", or "bernoulli"
    
    Returns:
        scenarios_list: List of demand scenarios
    """
    ncust = length(demandmean)
    bernoulli_str = experiment_type == "bernoulli" ? " (Bernoulli mode)" : ""
    println("Generating $nscen Facility Location scenarios with experiment type '$experiment_type'$bernoulli_str...")
    
    scenarios_list = Vector{Vector{Float64}}()
    
    for scenario in 1:nscen
        scenario_demand = generate_demand_scenario(ncust, demandmean, experiment_type)
        push!(scenarios_list, scenario_demand)
        
        if scenario <= 3  # Debug first 3 scenarios
            println("   Scenario $scenario: total demand = $(round(sum(scenario_demand), digits=2))")
        end
    end
    
    println("✅ Generated $nscen scenarios successfully!")
    return scenarios_list
end

# =============================================================================
# MAIN FUNCTION: GENERATE SAA SAMPLES
# =============================================================================

function generate_saa_samples(instance_file::String, S::Int;
                             experiment_type::String="low",
                             output_file::Union{String, Nothing}=nothing)
    """
    Generate SAA samples and save to file
    
    Args:
        instance_file: Instance file name (e.g., "cap91.txt")
        S: Number of samples to generate
        experiment_type: "low", "high", or "bernoulli"
        output_file: Output file path (default: FL/Scenarios/S<S>_<expt_type>.txt)
    
    Returns:
        samples: List of generated samples
        output_file_path: Path to the saved file
    """
    
    println("="^80)
    println("GENERATING SAA SAMPLES - FACILITY LOCATION")
    println("="^80)
    println("Instance: $instance_file")
    println("Number of samples: $S")
    println("Experiment type: $experiment_type")
    println("="^80)
    
    # Set random seed for reproducibility (always 1234)
    Random.seed!(1234)
    println("✅ Random seed set to 1234 (ensuring reproducibility)")
    
    # Get the directory where this script is located
    script_dir = @__DIR__
    project_root = dirname(dirname(script_dir))  # Go up two levels: Samples/FL -> Samples -> project root
    
    # Construct file paths
    instance_path = joinpath(project_root, "Facility_Location", "instances", instance_file)
    
    # Check if file exists
    if !isfile(instance_path)
        error("❌ Error: Instance file not found! Expected: $instance_path")
    end
    
    println("\n" * "="^40)
    println("STEP 1: LOADING DATA")
    println("="^40)
    
    # Load instance data to get demandmean
    println("Loading instance from: $instance_path")
    capacity, fixedcost, cost, demandmean = read_orlib_cap(instance_path)
    println("✅ Instance loaded: $(length(capacity)) facilities, $(length(demandmean)) customers")
    
    println("\n" * "="^40)
    println("STEP 2: GENERATING SAMPLES")
    println("="^40)
    
    # Generate scenarios
    samples = generate_facility_location_scenarios(S, demandmean, experiment_type)
    
    println("\n" * "="^40)
    println("STEP 3: SAVING SAMPLES TO FILE")
    println("="^40)
    
    # Generate output filename if not provided (new naming: S<S>_<expt_type>.txt)
    if output_file === nothing
        output_file = joinpath(script_dir, "Scenarios", "S$(S)_$(experiment_type).txt")
    end
    
    # Create directory if it doesn't exist
    dir = dirname(output_file)
    if !isempty(dir) && !isdir(dir)
        mkpath(dir)
    end
    
    # Save samples to file (one scenario per line, space-separated)
    open(output_file, "w") do file
        for sample in samples
            # Write sample as space-separated floating point numbers
            println(file, join([string(val) for val in sample], " "))
        end
    end
    
    println("✅ Samples saved to: $output_file")
    println("   Total samples: $(length(samples))")
    if length(samples) > 0
        println("   Sample length: $(length(samples[1]))")
    end
    
    return samples, output_file
end

# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 2
        println("❌ Error: Incorrect number of arguments!")
        println("Usage: julia Samples_For_SAA.jl <instance_file> <S> [experiment_type]")
        println("  instance_file: Instance file name (e.g., cap91.txt)")
        println("  S: Number of samples to generate")
        println("  experiment_type: 'low', 'high', or 'bernoulli' (default: 'low')")
        println("\nExample:")
        println("  julia Samples_For_SAA.jl cap91.txt 25")
        println("  julia Samples_For_SAA.jl cap91.txt 25 bernoulli")
        exit(1)
    end
    
    instance_file = ARGS[1]
    
    # Parse S (number of samples)
    S = try
        parsed = parse(Int, ARGS[2])
        if parsed <= 0
            throw(ArgumentError("S must be positive"))
        end
        parsed
    catch
        println("❌ Error: S must be a positive integer!")
        println("You provided: '$(ARGS[2])'")
        exit(1)
    end
    
    # Parse optional experiment_type
    experiment_type = length(ARGS) > 2 ? ARGS[3] : "low"
    if !(experiment_type in ["low", "high", "bernoulli"])
        println("❌ Error: experiment_type must be 'low', 'high', or 'bernoulli'!")
        println("You provided: '$experiment_type'")
        exit(1)
    end
    
    # Generate samples
    samples, output_file = generate_saa_samples(instance_file, S; experiment_type=experiment_type)
    
    println("\n" * "="^80)
    println("SAA SAMPLES GENERATION COMPLETED")
    println("="^80)
    println("Samples saved to: $output_file")
end

