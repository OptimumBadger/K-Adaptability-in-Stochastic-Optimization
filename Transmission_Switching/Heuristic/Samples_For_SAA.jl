# Samples_For_SAA.jl
# Generate SAA (Sample Average Approximation) samples and save to file
# This file generates scenarios that will be used consistently across Assignment and Evaluation phases

using PowerModels
using DataFrames, XLSX, Random, Distributions
using Dates

# Include utility functions from Candidate_List_Generation_Phase.jl
include("Candidate_List_Generation_Phase.jl")

# =============================================================================
# MAIN FUNCTION: GENERATE AND SAVE SAMPLES
# =============================================================================

function generate_and_save_samples(instance_file::String, S::Int;
                                    sigma_low::Float64=0.04, sigma_high::Float64=0.06,
                                    use_bernoulli::Bool=false,
                                    output_file::Union{String, Nothing}=nothing)
    """
    Generate SAA samples and save to file
    
    Args:
        instance_file: Instance file name (e.g., "case118Blumsack.m")
        S: Number of samples to generate
        sigma_low, sigma_high: Parameters for scenario generation
        use_bernoulli: Whether to use Bernoulli mode
        output_file: Output file path (default: SAA_samples_S<S>.txt)
    
    Returns:
        samples: List of generated samples
        output_file_path: Path to the saved file
    """
    
    println("="^80)
    println("GENERATING SAA SAMPLES")
    println("="^80)
    println("Instance: $instance_file")
    println("Number of samples: $S")
    println("="^80)
    
    # Set random seed for reproducibility (always 1234)
    Random.seed!(1234)
    println("✅ Random seed set to 1234 (ensuring reproducibility)")
    
    # Get the directory where this script is located
    script_dir = @__DIR__
    project_root = dirname(dirname(script_dir))  # Go up two levels to project root
    
    # Construct file paths
    instance_path = joinpath(project_root, "Transmission_Switching", "instances", instance_file)
    excel_path = joinpath(project_root, "Transmission_Switching", "instances", "Random_Demands.xlsx")
    
    # Check if files exist
    if !isfile(instance_path)
        error("❌ Error: Instance file not found! Expected: $instance_path")
    end
    
    if !isfile(excel_path)
        error("❌ Error: Excel file not found! Expected: $excel_path")
    end
    
    println("\n" * "="^40)
    println("STEP 1: LOADING DATA")
    println("="^40)
    
    # Extract case name from instance file
    case_name = extract_case_name(instance_file)
    println("Extracted case name: $case_name")
    
    # Load demand data from Excel
    demand_mean_data = load_demand_data(excel_path, case_name)
    
    println("\n" * "="^40)
    println("STEP 2: GENERATING SAMPLES")
    println("="^40)
    
    # Generate samples (with seed already set above for reproducibility)
    samples = generate_powerflow_scenarios(S, demand_mean_data, sigma_low, sigma_high, use_bernoulli)
    
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
            println(file, join(sample, " "))
        end
    end
    
    println("✅ Samples saved to: $output_file")
    println("   Total samples: $(length(samples))")
    if length(samples) > 0
        println("   Sample length: $(length(samples[1]))")
    end
    
    println("\n✅ SAA sample generation completed!")
    
    return samples, output_file
end

# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 1
        println("❌ Error: Incorrect number of arguments!")
        println("Usage: julia Samples_For_SAA.jl <S> [instance_file] [sigma_low] [sigma_high] [bernoulli]")
        println("  S: Number of samples to generate")
        println("  instance_file: Instance file name (default: case118Blumsack.m)")
        println("  sigma_low: Lower bound for sigma (default: 0.04)")
        println("  sigma_high: Upper bound for sigma (default: 0.06)")
        println("  bernoulli: Use Bernoulli mode (default: false)")
        println("\nExample:")
        println("  julia Samples_For_SAA.jl 25")
        println("  julia Samples_For_SAA.jl 25 case118Blumsack.m 0.04 0.06 false")
        exit(1)
    end
    
    # Parse S (number of samples)
    S = try
        parsed = parse(Int, ARGS[1])
        if parsed <= 0
            throw(ArgumentError("S must be positive"))
        end
        parsed
    catch
        println("❌ Error: S must be a positive integer!")
        println("You provided: '$(ARGS[1])'")
        exit(1)
    end
    
    # Parse optional arguments
    instance_file = length(ARGS) > 1 ? ARGS[2] : "case118Blumsack.m"
    sigma_low = length(ARGS) > 2 ? parse(Float64, ARGS[3]) : 0.04
    sigma_high = length(ARGS) > 3 ? parse(Float64, ARGS[4]) : 0.06
    use_bernoulli = length(ARGS) > 4 ? parse(Bool, ARGS[5]) : false
    
    # Generate and save samples
    samples, output_file = generate_and_save_samples(
        instance_file, S;
        sigma_low=sigma_low, sigma_high=sigma_high,
        use_bernoulli=use_bernoulli
    )
    
    println("\n📊 Results:")
    println("  - Samples generated: $(length(samples))")
    println("  - Output file: $output_file")
end

