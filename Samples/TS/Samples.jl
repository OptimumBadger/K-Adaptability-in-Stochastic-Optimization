# Samples.jl (TS)
# Generate SAA samples for Transmission Switching
#
# This is analogous to Samples/FL/Samples.jl but for the TS problem.
#
# Usage:
#   julia Samples.jl <instance_file> <S> [sigma_low] [sigma_high] [bernoulli]
#     instance_file: MATPOWER case file (e.g., case118Blumsack.m)
#     S:             number of scenarios
#     sigma_low:     lower bound for sigma (default: 0.04)
#     sigma_high:    upper bound for sigma (default: 0.06)
#     bernoulli:     use Bernoulli mode (true/false, default: false)

using Random, Distributions
using Dates

include("TSUtilities.jl")

# =============================================================================
# MAIN FUNCTION: GENERATE TS SAMPLES
# =============================================================================

function generate_ts_samples(instance_file::String, S::Int;
                             sigma_low::Float64=0.04, sigma_high::Float64=0.06,
                             use_bernoulli::Bool=false,
                             output_file::Union{String, Nothing}=nothing)
    """
    Generate TS samples and save to file.
    
    Args:
        instance_file: MATPOWER case file (e.g., "case118Blumsack.m")
        S: Number of samples to generate
        sigma_low, sigma_high: Parameters for scenario generation
        use_bernoulli: Whether to use Bernoulli mode
        output_file: Output file path (default: TS/Scenarios/S<S>.txt)
    """
    println("="^80)
    println("GENERATING SAA SAMPLES - TRANSMISSION SWITCHING")
    println("="^80)
    println("Instance: $instance_file")
    println("Number of samples: $S")
    println("Sigma range: [$sigma_low, $sigma_high]")
    println("Bernoulli mode: $(use_bernoulli ? "true" : "false")")
    println("="^80)
    
    # Set random seed for reproducibility
    Random.seed!(1234)
    println("✅ Random seed set to 1234 (ensuring reproducibility)")
    
    # Locate project root and instance / Excel paths
    script_dir = @__DIR__
    project_root = dirname(dirname(script_dir))  # Samples/TS -> Samples -> project root
    
    instance_path = joinpath(project_root, "Transmission_Switching", "instances", instance_file)
    excel_path    = joinpath(project_root, "Transmission_Switching", "instances", "Random_Demands.xlsx")
    
    if !isfile(instance_path)
        error("❌ Error: Instance file not found! Expected: $instance_path")
    end
    if !isfile(excel_path)
        error("❌ Error: Excel file not found! Expected: $excel_path")
    end
    
    println("\n" * "="^40)
    println("STEP 1: LOADING DEMAND DATA")
    println("="^40)
    
    case_name = extract_case_name(instance_file)
    println("Extracted case name: $case_name")
    demand_mean_data = load_demand_data(excel_path, case_name; instance_file=instance_path)
    
    println("\n" * "="^40)
    println("STEP 2: GENERATING SAMPLES")
    println("="^40)
    
    samples = generate_powerflow_scenarios(S, demand_mean_data, sigma_low, sigma_high, use_bernoulli)
    
    println("\n" * "="^40)
    println("STEP 3: SAVING SAMPLES TO FILE")
    println("="^40)
    
    # Choose experiment type label (for filename) similar to FL pattern
    expt_type = use_bernoulli ? "bernoulli" : "gaussian"
    
    # Default output file if not provided
    if output_file === nothing
        output_file = joinpath(script_dir, "Scenarios", "S$(S)_$(expt_type).txt")
    end
    
    # Ensure directory exists
    dir = dirname(output_file)
    if !isempty(dir) && !isdir(dir)
        mkpath(dir)
    end
    
    # Save samples (one scenario per line)
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
    
    return samples, output_file
end

# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 2
        println("❌ Error: Incorrect number of arguments!")
        println("Usage: julia Samples.jl <instance_file> <S> [sigma_low] [sigma_high] [bernoulli]")
        println("  instance_file: MATPOWER case file (e.g., case118Blumsack.m)")
        println("  S: Number of samples to generate")
        println("  sigma_low: Lower bound for sigma (default: 0.04)")
        println("  sigma_high: Upper bound for sigma (default: 0.06)")
        println("  bernoulli: Use Bernoulli mode (default: false)")
        println("\nExamples:")
        println("  julia Samples.jl case118Blumsack.m 25")
        println("  julia Samples.jl case118Blumsack.m 25 0.04 0.06 true")
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
    
    sigma_low    = length(ARGS) > 2 ? parse(Float64, ARGS[3]) : 0.04
    sigma_high   = length(ARGS) > 3 ? parse(Float64, ARGS[4]) : 0.06
    use_bernoulli = length(ARGS) > 4 ? parse(Bool, ARGS[5]) : false
    
    samples, output_file = generate_ts_samples(
        instance_file, S;
        sigma_low=sigma_low,
        sigma_high=sigma_high,
        use_bernoulli=use_bernoulli,
        output_file=nothing
    )
    
    println("\n" * "="^80)
    println("TS SAMPLES GENERATION COMPLETED")
    println("="^80)
    println("Samples saved to: $output_file")
end


