# Samples_For_SAA.jl
# Generate SAA samples for Facility Location
# This file generates demand scenarios and saves them to a file

using Random, Distributions
using Dates

# Include utility functions from Candidate_List_Generation_Phase.jl
include("Candidate_List_Generation_Phase.jl")

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
        output_file: Output file path (default: SAA_samples_S<S>.txt)
    
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
    project_root = dirname(dirname(script_dir))  # Go up two levels to project root
    
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
    
    # Generate output filename if not provided
    if output_file === nothing
        output_file = joinpath(script_dir, "SAA_samples_S$(S).txt")
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

