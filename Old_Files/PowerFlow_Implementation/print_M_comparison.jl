# print_M_comparison.jl - Print comparison table for M_ij values
# Usage: julia print_M_comparison.jl <instance_file>
# Example: julia print_M_comparison.jl case118Blumsack.m

using PowerModels
using Printf
using Statistics

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

function parameters(network; line_capacity_factor=1.0)
    """Extract network parameters from PowerModels data"""
    # Sets
    B = network["bus"]
    L = network["branch"]
    G = network["gen"]

    # Parameters
    B_ij = zeros(length(L)) # Susceptance values
    fbar_ij = zeros(length(L)) # Upper bound on the load
    p_max = zeros(length(B)) # Max power output
    p_min = zeros(length(B)) # Min power output
    c = zeros(length(B)) # Linear Cost coefficient with zero intercept

    for (branch_id, branch) in L
        B_ij[parse(Int, branch_id)] = -branch["br_x"]/(branch["br_r"]^2 + branch["br_x"]^2)
        # Handle missing rate_a key - use a default value if not present
        rate_a = haskey(branch, "rate_a") ? branch["rate_a"] : 2.2  # Default thermal limit
        fbar_ij[parse(Int, branch_id)] = rate_a * line_capacity_factor
    end

    for (gen_id, generator) in G
        c[generator["gen_bus"]] = generator["cost"][1]
        p_max[generator["gen_bus"]] = generator["pmax"]
        p_min[generator["gen_bus"]] = generator["pmin"]
    end
    
    n = 2*length(network["bus"])        # Number of rows
    m = length(network["branch"])       # Number of columns

    # Initialize matrix
    A = zeros(n, m)

    # Create A
    for (branch_id, branch) in L
        f_bus = branch["f_bus"]  # From bus
        t_bus = branch["t_bus"]  # To bus

        # Update incidence matrix
        branch_index = parse(Int, branch_id)
        A[2*f_bus-1, branch_index] = 1   # From bus
        A[2*f_bus, branch_index] = -1    # From bus
        A[2*t_bus-1, branch_index] = -1  # To bus
        A[2*t_bus, branch_index] = 1     # To bus
    end

    return B_ij, fbar_ij, p_max, p_min, A, c
end

# =============================================================================
# MAIN FUNCTION
# =============================================================================

function main(instance_file::String)
    println("="^80)
    println("M_IJ COMPARISON TABLE")
    println("="^80)
    println("Instance: $instance_file")
    println("="^80)
    
    # Get the directory where this script is located
    script_dir = @__DIR__
    
    # Construct file path (instance file in powerflow_instances subdirectory)
    instance_path = joinpath(script_dir, "powerflow_instances", instance_file)
    
    # Check if file exists
    if !isfile(instance_path)
        error("❌ Error: Instance file not found! Expected: $instance_path")
    end
    
    println("\nLoading network from: $instance_path")
    network = PowerModels.parse_file(instance_path)
    println("✅ Network loaded: $(length(network["bus"])) buses, $(length(network["branch"])) branches")
    
    # Get parameters
    B_ij, fbar_ij, p_max, p_min, A, c = parameters(network; line_capacity_factor=1.0)
    
    L = network["branch"]
    n_branches = length(L)
    
    # Calculate values for each branch
    M_old = Float64[]  # -12.6 * B_ij
    fbar_over_B = Float64[]  # fbar_ij / |B_ij|
    
    for i in 1:n_branches
        push!(M_old, -12.6 * B_ij[i])
        
        if abs(B_ij[i]) > 1e-10  # Avoid division by zero
            push!(fbar_over_B, fbar_ij[i] / abs(B_ij[i]))
        else
            push!(fbar_over_B, 0.0)
        end
    end
    
    # Print table header
    println("\n" * "="^80)
    println("BRANCH COMPARISON TABLE")
    println("="^80)
    @printf("%-8s | %-20s | %-20s\n", "Branch", "-12.6 * B_ij", "fbar_ij / |B_ij|")
    println("-"^80)
    
    # Print table rows
    for i in 1:n_branches
        @printf("%-8d | %-20.6f | %-20.6f\n", i, M_old[i], fbar_over_B[i])
    end
    
    println("="^80)
    
    # Calculate new M_ij using the method from TS_scenarios.jl
    n_nodes = length(network["bus"])
    sorted_fbar_over_B = sort(fbar_over_B, rev=true)
    n_minus_1 = n_nodes - 1
    top_n_minus_1 = sorted_fbar_over_B[1:n_minus_1]
    sum_top = sum(top_n_minus_1)
    max_B_ij = maximum(abs.(B_ij))
    M_new = sum_top * max_B_ij
    
    # Print summary statistics
    println("\nSUMMARY STATISTICS:")
    println("-"^80)
    @printf("Number of branches: %d\n", n_branches)
    @printf("Number of nodes: %d\n", n_nodes)
    @printf("Min(-12.6 * B_ij): %.6f\n", minimum(M_old))
    @printf("Max(-12.6 * B_ij): %.6f\n", maximum(M_old))
    @printf("Mean(-12.6 * B_ij): %.6f\n", mean(M_old))
    @printf("\n")
    @printf("Min(fbar_ij / |B_ij|): %.6f\n", minimum(fbar_over_B))
    @printf("Max(fbar_ij / |B_ij|): %.6f\n", maximum(fbar_over_B))
    @printf("Mean(fbar_ij / |B_ij|): %.6f\n", mean(fbar_over_B))
    println("\n" * "-"^80)
    println("NEW M_IJ CALCULATION:")
    println("-"^80)
    @printf("Top %d values from sorted fbar_ij / |B_ij|:\n", n_minus_1)
    @printf("  Sum of top %d values: %.6f\n", n_minus_1, sum_top)
    @printf("  Max(|B_ij|): %.6f\n", max_B_ij)
    @printf("  New M_ij = sum_top * max_B_ij = %.6f\n", M_new)
    println("="^80)
    
    return nothing
end

# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) != 1
        println("❌ Error: Incorrect number of arguments!")
        println("Usage: julia print_M_comparison.jl <instance_file>")
        println("Example: julia print_M_comparison.jl case118Blumsack.m")
        exit(1)
    end
    
    instance_file = ARGS[1]
    
    # Run main function
    main(instance_file)
end

