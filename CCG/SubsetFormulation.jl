# SubsetFormulation.jl
# Subset Formulation (SF) Master Problem - Problem Independent
# This file contains SF-specific master problem logic

using JuMP, Gurobi
using MathOptInterface
const MOI = MathOptInterface
using Dates

# =============================================================================
# MASTER PROBLEM (SUBSET FORMULATION)
# =============================================================================

function solve_sf_master_problem(cost_vector, z_matrix, K; integer_vars=false, log_file=nothing, iteration=nothing)
    """Solve the SF Master Problem (Partition Formulation)
    
    Parameters:
        cost_vector: f(A) values for each subset A
        z_matrix: |S| × |A| matrix where z_matrix[s, A] = 1 if scenario s is in subset A
        K: Cardinality constraint limit
        integer_vars: If true, solve as integer program; if false, linear relaxation
        log_file: Optional log file path (will append with iteration markers)
        iteration: Optional iteration number for logging
    
    Returns:
        model: JuMP model
        v_variables: Vector of v_A variables
        objective_value: Objective value
        partition_constraints: Constraint references for dual extraction
        k_constraint: K constraint reference for dual extraction
        work_units: Work units used
    """
    nscen, nsubsets = size(z_matrix)
    
    model = Model(Gurobi.Optimizer)
    
    if log_file !== nothing
        # Append iteration marker to log file
        if iteration !== nothing
            open(log_file, "a") do file
                println(file, "\n" * "="^80)
                println(file, "MASTER PROBLEM - ITERATION $iteration")
                println(file, "="^80)
                println(file, "Timestamp: $(now())")
                println(file, "Problem size: $nscen scenarios, $nsubsets subsets, K=$K")
                println(file, "Type: $(integer_vars ? "INTEGER" : "LINEAR RELAXATION")")
                println(file, "="^80)
            end
        end
        # Enable file logging but disable console output
        set_optimizer_attribute(model, "OutputFlag", 1)  # Enable output (will go to file, not console due to LogToConsole=0)
        set_optimizer_attribute(model, "LogFile", log_file)
        set_optimizer_attribute(model, "LogToConsole", 1)
    else
        # No logging - disable all output
        set_optimizer_attribute(model, "OutputFlag", 0)
    end
    
    # Decision Variables - LINEAR RELAXATION (continuous) or INTEGER
    if integer_vars
        v_variables = [@variable(model, binary = true) for A in 1:nsubsets]  # v_A: binary
    else
        v_variables = [@variable(model, lower_bound = 0, upper_bound = 1) for A in 1:nsubsets]  # v_A: continuous [0,1]
    end
    
    # Objective Function (9a): min ∑_{A∈F} f(A)v_A
    @objective(model, Min, sum(cost_vector[A] * v_variables[A] for A in 1:nsubsets))
    
    # Constraints - Store references for dual extraction
    # Partition constraints (9b): ∑_{A∈F} z_sA v_A = 1 ∀s ∈ S
    partition_constraints = @constraint(model, [s in 1:nscen], 
        sum(z_matrix[s, A] * v_variables[A] for A in 1:nsubsets) == 1)
    
    # Cardinality constraint (9c): ∑_{A∈F} v_A ≤ K
    k_constraint = @constraint(model, sum(v_variables[A] for A in 1:nsubsets) <= K)
    
    # Solve
    optimize!(model)
    
    work_units = 0.0
    try
        work_units = solve_time(model)
    catch
    end
    
    if termination_status(model) == MOI.OPTIMAL
        return model, v_variables, objective_value(model), partition_constraints, k_constraint, work_units
    else
        # Diagnostic output for infeasibility
        term_status = termination_status(model)
        println("❌ Master problem termination status: $term_status")
        
        if term_status == MOI.INFEASIBLE || term_status == MOI.INFEASIBLE_OR_UNBOUNDED
            println("   Problem is INFEASIBLE")
            println("   Checking constraint violations...")
            
            # Check if any scenario is not covered
            uncovered_scenarios = Int[]
            for s in 1:nscen
                if sum(z_matrix[s, :]) == 0
                    push!(uncovered_scenarios, s)
                end
            end
            
            if !isempty(uncovered_scenarios)
                println("   ❌ CAUSE IDENTIFIED: Scenarios $uncovered_scenarios are not covered by any subset")
                println("      Partition constraint for these scenarios: ∑_{A∈F} z_matrix[s, A] * v_A = 1")
                println("      Since z_matrix[s, :] is all zeros, this constraint cannot be satisfied.")
            else
                println("   All scenarios are covered, but problem is still infeasible.")
                println("   This may be due to:")
                println("      - The combination of partition constraints and K=$K being too restrictive")
                println("      - Numerical issues in the solver")
                println("   z_matrix dimensions: $(size(z_matrix))")
                println("   Number of subsets: $(size(z_matrix, 2))")
                println("   K (cardinality limit): $K")
            end
        end
        
        return model, v_variables, Inf, partition_constraints, k_constraint, work_units
    end
end

# =============================================================================
# COST VECTOR BUILDING 
# =============================================================================

function build_cost_vector(scenarios::Vector{Vector{Float64}},
                          initial_subsets::Vector{Vector{Int}},
                          evaluate_subset_cost::Function,
                          problem_data)
    """Build the cost vector f(A) for Subset Formulation
    
    This is problem-independent - it just calls the problem-specific evaluate_subset_cost function.
    
    Args:
        scenarios: List of S scenarios
        initial_subsets: List of initial subset vectors (each is a binary vector of length S)
        evaluate_subset_cost: Problem-specific function: (subset_vector, scenarios, problem_data) -> (cost, work_units)
        problem_data: Problem-specific data structure
    
    Returns:
        cost_vector: Vector of f(A) values for each subset
        z_matrix: S × nsubsets matrix where z_matrix[s, A] = 1 if scenario s is in subset A
        total_work_units: Total work units used
    """
    nscen = length(scenarios)
    nsubsets = length(initial_subsets)
    
    println("Building cost vector f(A) for $nsubsets initial subsets...")
    
    cost_vector = Float64[]
    z_matrix = zeros(nscen, nsubsets)
    total_work_units = 0.0
    
    for A_idx in 1:nsubsets
        subset_vector = initial_subsets[A_idx]
        
        # Validate subset vector
        if length(subset_vector) != nscen
            error("Subset vector $A_idx has length $(length(subset_vector)) but expected $nscen")
        end
        
        # Calculate cost for this subset
        cost_val, work_units = evaluate_subset_cost(subset_vector, scenarios, problem_data)
        push!(cost_vector, cost_val)
        total_work_units += work_units
        
        # Update z_matrix
        for s in 1:nscen
            if subset_vector[s] == 1
                z_matrix[s, A_idx] = 1
            end
        end
        
        if A_idx % 10 == 0
            println("  Processed $A_idx subsets")
        end
    end
    
    println("Cost vector built successfully")
    
    # Diagnostic: Check scenario coverage
    uncovered_scenarios = Int[]
    for s in 1:nscen
        if sum(z_matrix[s, :]) == 0
            push!(uncovered_scenarios, s)
        end
    end
    
    if !isempty(uncovered_scenarios)
        println("⚠️  WARNING: The following scenarios are NOT covered by any subset:")
        println("   Uncovered scenarios: $uncovered_scenarios")
        println("   This will cause the master problem to be INFEASIBLE!")
        println("   Partition constraint for scenario s requires: ∑_{A∈F} z_matrix[s, A] * v_A = 1")
        println("   If z_matrix[s, :] is all zeros, this constraint cannot be satisfied.")
    else
        println("✅ All $nscen scenarios are covered by at least one subset")
    end
    
    return cost_vector, z_matrix, total_work_units
end

