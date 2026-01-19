# Create_Partitions.jl
# Creates initial partitions for SF column generation by assigning scenarios to K candidate vectors
# from multiple feasible solutions
# Usage: julia Create_Partitions.jl <instance_file> <scenarios_file> <feasible_solutions_file> [unmet_pen] [scaling_factor] [capacity_factor]

using JuMP, Gurobi
using MathOptInterface
const MOI = MathOptInterface

# Include required files
include("FLUtilities.jl")  # For read_orlib_cap, load_scenarios_from_file, and evaluate_cost_FL

# =============================================================================
# LOAD FEASIBLE SOLUTIONS FROM FILE
# =============================================================================

function load_feasible_solutions_from_file(file_path::String)
    """Load feasible solutions from file
    
    Format:
    Solution 1
    <binary vector 1>
    <binary vector 2>
    ...
    <binary vector K>
    Solution 2
    ...
    
    Args:
        file_path: Path to feasible solutions file
    
    Returns:
        solutions: Vector of solutions, where each solution is a Vector of K binary vectors
    """
    if !isfile(file_path)
        error("❌ Error: Feasible solutions file not found! Expected: $file_path")
    end
    
    println("Loading feasible solutions from: $file_path")
    solutions = Vector{Vector{Vector{Int}}}()
    current_solution = Vector{Vector{Int}}()
    
    open(file_path, "r") do file
        for (line_num, line) in enumerate(eachline(file))
            line = strip(line)
            
            if startswith(line, "Solution")
                # New solution header - save previous solution if it exists
                if !isempty(current_solution)
                    push!(solutions, current_solution)
                    current_solution = Vector{Vector{Int}}()
                end
            elseif length(line) > 0
                # Parse binary vector (space-separated integers)
                try
                    values = [parse(Int, x) for x in split(line)]
                    push!(current_solution, values)
                catch e
                    error("❌ Error parsing line $line_num in feasible solutions file: $e")
                end
            end
        end
        
        # Don't forget the last solution
        if !isempty(current_solution)
            push!(solutions, current_solution)
        end
    end
    
    println("✅ Loaded $(length(solutions)) feasible solutions")
    if length(solutions) > 0
        K = length(solutions[1])
        println("   Each solution has $K binary vectors")
    end
    
    return solutions
end

# =============================================================================
# EXTRACT PARAMETERS FROM FILENAME
# =============================================================================

function extract_parameters_from_filename(filename::String)
    """Extract L, S, K, T from filename like feasible_solutions_L10_S10_K2_T10_bernoulli.txt
    
    Returns:
        (L, S, K, T) or (nothing, nothing, nothing, nothing) if parsing fails
    """
    # Remove extension
    name = splitext(basename(filename))[1]
    
    # Pattern: feasible_solutions_L<L>_S<S>_K<K>_T<T>_<experiment>
    parts = split(name, "_")
    
    L_val = nothing
    S_val = nothing
    K_val = nothing
    T_val = nothing
    
    for part in parts
        if startswith(part, "L")
            L_val = try parse(Int, part[2:end]) catch; nothing end
        elseif startswith(part, "S") && !startswith(part, "Solution")
            S_val = try parse(Int, part[2:end]) catch; nothing end
        elseif startswith(part, "K")
            K_val = try parse(Int, part[2:end]) catch; nothing end
        elseif startswith(part, "T")
            T_val = try parse(Int, part[2:end]) catch; nothing end
        end
    end
    
    return (L_val, S_val, K_val, T_val)
end

# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 4
        println("❌ Error: Incorrect number of arguments!")
        println("Usage: julia Create_Partitions.jl <instance_file> <scenarios_file> <S> <feasible_solutions_file> [unmet_pen] [scaling_factor] [capacity_factor]")
        println("\nArguments:")
        println("  instance_file: Instance file path (e.g., Facility_Location/instances/cap91.txt)")
        println("  scenarios_file: File containing scenarios (e.g., Samples/FL/Scenarios/S300_bernoulli.txt)")
        println("  S: Number of scenarios to use (first S from scenarios_file)")
        println("  feasible_solutions_file: File containing feasible solutions (e.g., Samples/FL/Feasible_Solutions/feasible_solutions_L10_S10_K2_T10.txt)")
        println("  unmet_pen: Unmet demand penalty (default: 15.0)")
        println("  scaling_factor: Scaling factor for transportation cost (default: 0.2)")
        println("  capacity_factor: Capacity factor (default: 0.3)")
        println("\nOutput:")
        println("  Creates InitialSubsets_L#_S#_K#_T#_FL.txt in Samples/FL/ directory")
        println("  where T# is the number of unique solutions found (not the original T parameter)")
        println("\nExample:")
        println("  julia Create_Partitions.jl Facility_Location/instances/cap91.txt Samples/FL/Scenarios/S300_bernoulli.txt 10 Samples/FL/Feasible_Solutions/feasible_solutions_L10_S10_K2_T10.txt")
        exit(1)
    end
    
    # Parse arguments
    instance_file = ARGS[1]
    scenarios_file = ARGS[2]
    S = try
        parsed = parse(Int, ARGS[3])
        if parsed <= 0
            throw(ArgumentError("S must be positive"))
        end
        parsed
    catch
        println("❌ Error: S must be a positive integer!")
        println("You provided: '$(ARGS[3])'")
        exit(1)
    end
    feasible_solutions_file = ARGS[4]
    unmet_pen = length(ARGS) > 4 ? parse(Float64, ARGS[5]) : 15.0
    scaling_factor = length(ARGS) > 5 ? parse(Float64, ARGS[6]) : 0.2
    capacity_factor = length(ARGS) > 6 ? parse(Float64, ARGS[7]) : 0.3
    
    println("="^80)
    println("CREATE PARTITIONS FOR SF COLUMN GENERATION")
    println("="^80)
    println("Instance: $instance_file")
    println("Scenarios file: $scenarios_file")
    println("Using first S = $S scenarios")
    println("Feasible solutions file: $feasible_solutions_file")
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
    
    # Handle feasible solutions file path
    if !isabspath(feasible_solutions_file)
        if !isfile(feasible_solutions_file)
            # Try in Samples/FL/
            feasible_solutions_file = joinpath(script_dir, basename(feasible_solutions_file))
        end
        if !isfile(feasible_solutions_file)
            # Try in Samples/FL/Feasible_Solutions/
            feasible_solutions_file = joinpath(script_dir, "Feasible_Solutions", basename(feasible_solutions_file))
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
    all_scenarios = load_scenarios_from_file(scenarios_file)
    
    # Validate S and slice first S scenarios
    if S > length(all_scenarios)
        error("❌ Error: Requested S = $S scenarios, but scenarios_file contains only $(length(all_scenarios)) scenarios.")
    end
    scenarios = all_scenarios[1:S]
    nscen = length(scenarios)
    println("✅ Loaded $(length(all_scenarios)) scenarios from file, using first S = $S scenarios")
    println()
    
    # Load feasible solutions
    println("Loading feasible solutions from file...")
    all_solutions = load_feasible_solutions_from_file(feasible_solutions_file)
    T_actual = length(all_solutions)  # Actual number of unique solutions found
    
    if T_actual == 0
        error("❌ Error: No feasible solutions found in file!")
    end
    
    # Validate that all solutions have the same K
    K = length(all_solutions[1])
    for (idx, sol) in enumerate(all_solutions)
        if length(sol) != K
            error("❌ Error: Solution $idx has $(length(sol)) vectors, but expected $K")
        end
        # Validate vector lengths
        for (vec_idx, vec) in enumerate(sol)
            if length(vec) != nloc
                error("❌ Error: Solution $idx, vector $vec_idx has length $(length(vec)), but expected $nloc")
            end
        end
    end
    
    println("✅ Loaded $T_actual feasible solutions, each with $K binary vectors")
    println()
    
    # Extract L, S, K from filename for output naming
    L_file, S_file, K_file, T_file = extract_parameters_from_filename(feasible_solutions_file)
    
    # Create all partitions: T solutions × K vectors = T*K subsets
    println("Creating partitions from $T_actual solutions...")
    all_subsets = Vector{Vector{Int}}()  # Will contain T*K subsets
    
    for (sol_idx, solution) in enumerate(all_solutions)
        # For this solution, create K partitions (one per vector)
        partitions = [Int[] for _ in 1:K]  # K partitions for this solution
        
        # Assign each scenario to the vector that gives lowest cost
        for s in 1:nscen
            best_cost = Inf
            best_vector = 1
            
            # Evaluate cost of serving scenario s with each vector in this solution
            for k in 1:K
                cost_val, _ = evaluate_cost_FL(solution[k], scenarios[s], problem_data)
                if cost_val < best_cost
                    best_cost = cost_val
                    best_vector = k
                end
            end
            
            # Assign scenario s to best vector (scenario indices are 1-based)
            push!(partitions[best_vector], s)
        end
        
        # Add all K partitions from this solution to the overall list
        for k in 1:K
            push!(all_subsets, partitions[k])
        end
        
        if sol_idx % 10 == 0
            println("  Processed $sol_idx solutions...")
        end
    end
    
    println("✅ Completed partition creation")
    println("   Total subsets created: $(length(all_subsets)) (=$T_actual solutions × $K vectors)")
    println()
    
    # Create output file name
    # Use L, K from filename, S from command-line argument, and T_actual (number of unique solutions found)
    L_str = L_file !== nothing ? string(L_file) : "?"
    S_str = string(S)  # Use S from command-line argument
    K_str = K_file !== nothing ? string(K_file) : string(K)
    T_str = string(T_actual)
    
    output_filename = "InitialSubsets_L$(L_str)_S$(S_str)_K$(K_str)_T$(T_str)_FL.txt"
    output_file_path = joinpath(script_dir, output_filename)
    
    # Write output file: T*K subsets, one per line
    # Each line contains space-separated scenario indices (1-based) for that subset
    println("Writing $(length(all_subsets)) subsets to output file: $output_file_path")
    open(output_file_path, "w") do io
        for subset in all_subsets
            if length(subset) > 0
                println(io, join(subset, " "))
            else
                println(io, "")  # Empty line if subset is empty
            end
        end
    end
    println("✅ Partitions written to $output_file_path")
    println()
    
    println("="^80)
    println("PARTITION CREATION COMPLETE")
    println("="^80)
    println("Output file: $output_file_path")
    println("Format: $(length(all_subsets)) subsets (one per line)")
    println("  Each line contains space-separated scenario indices (1-based)")
    println("  Total: $T_actual solutions × $K vectors = $(length(all_subsets)) subsets")
    println("="^80)
end
