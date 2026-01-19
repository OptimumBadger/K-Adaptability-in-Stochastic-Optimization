# Diagnostic script to check why initial master problem is infeasible
# Usage: julia diagnose_infeasibility.jl <initial_subsets_file> <S> <K>

if length(ARGS) < 3
    println("Usage: julia diagnose_infeasibility.jl <initial_subsets_file> <S> <K>")
    exit(1)
end

initial_subsets_file = ARGS[1]
S = parse(Int, ARGS[2])
K = parse(Int, ARGS[3])

println("="^80)
println("DIAGNOSTIC: Checking Initial Subsets for Infeasibility")
println("="^80)
println("Initial subsets file: $initial_subsets_file")
println("S (number of scenarios): $S")
println("K (cardinality limit): $K")
println()

# Load initial subsets
initial_subsets = Vector{Vector{Int}}()
if isfile(initial_subsets_file)
    open(initial_subsets_file, "r") do io
        for line in eachline(io)
            if !isempty(strip(line))
                scenario_indices = [parse(Int, x) for x in split(strip(line))]
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
else
    println("❌ Error: File not found: $initial_subsets_file")
    exit(1)
end

nsubsets = length(initial_subsets)
println("Loaded $nsubsets initial subsets")
println()

# Build z_matrix
z_matrix = zeros(Int, S, nsubsets)
for A_idx in 1:nsubsets
    subset_vector = initial_subsets[A_idx]
    for s in 1:S
        if subset_vector[s] == 1
            z_matrix[s, A_idx] = 1
        end
    end
end

println("="^80)
println("CHECK 1: Scenario Coverage")
println("="^80)
println("Checking if each scenario appears in at least one subset...")
println()

uncovered_scenarios = Int[]
for s in 1:S
    if sum(z_matrix[s, :]) == 0
        push!(uncovered_scenarios, s)
    end
end

if isempty(uncovered_scenarios)
    println("✅ All scenarios are covered by at least one subset")
else
    println("❌ PROBLEM FOUND: The following scenarios are NOT covered by any subset:")
    println("   Uncovered scenarios: $uncovered_scenarios")
    println("   This makes the partition constraint for these scenarios infeasible!")
    println("   Partition constraint: ∑_{A∈F} z_matrix[s, A] * v_A = 1")
    println("   If z_matrix[s, :] is all zeros, this constraint cannot be satisfied.")
end
println()

println("="^80)
println("CHECK 2: z_matrix Summary")
println("="^80)
println("z_matrix dimensions: $(size(z_matrix)) (scenarios × subsets)")
println()

# Show coverage per scenario
println("Coverage per scenario:")
for s in 1:S
    num_subsets_containing_s = sum(z_matrix[s, :])
    println("  Scenario $s: appears in $num_subsets_containing_s subset(s)")
end
println()

# Show scenarios per subset
println("Scenarios per subset:")
for A in 1:nsubsets
    num_scenarios_in_A = sum(z_matrix[:, A])
    println("  Subset $A: contains $num_scenarios_in_A scenario(s)")
end
println()

println("="^80)
println("CHECK 3: Cardinality Constraint Feasibility")
println("="^80)
println("Cardinality constraint: ∑_{A∈F} v_A ≤ K")
println("Number of subsets available: $nsubsets")
println("K (maximum subsets to select): $K")
if nsubsets < K
    println("⚠️  Warning: Only $nsubsets subsets available, but K=$K")
    println("   This is not necessarily a problem, but may limit solutions")
else
    println("✅ Sufficient subsets available for K=$K")
end
println()

println("="^80)
println("CHECK 4: z_matrix Visualization (first 10 scenarios, all subsets)")
println("="^80)
max_scenarios_to_show = min(10, S)
println("z_matrix[1:$max_scenarios_to_show, :] =")
for s in 1:max_scenarios_to_show
    print("  Scenario $s: [")
    for A in 1:nsubsets
        print("$(z_matrix[s, A])")
        if A < nsubsets
            print(" ")
        end
    end
    println("]")
end
if S > max_scenarios_to_show
    println("  ... (showing first $max_scenarios_to_show of $S scenarios)")
end
println()

println("="^80)
println("SUMMARY")
println("="^80)
if !isempty(uncovered_scenarios)
    println("❌ INFEASIBILITY CAUSE IDENTIFIED:")
    println("   Scenarios $uncovered_scenarios are not covered by any subset.")
    println("   The partition constraints for these scenarios cannot be satisfied.")
    println()
    println("SOLUTION:")
    println("   Ensure that every scenario 1..$S appears in at least one initial subset.")
else
    println("✅ All scenarios are covered.")
    println("   If the problem is still infeasible, it may be due to:")
    println("   1. The combination of partition constraints and K being too restrictive")
    println("   2. An issue with how the master problem is being solved")
    println("   3. Numerical issues in the solver")
end
println("="^80)
