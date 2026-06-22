# AssignmentFormulation.jl
# Assignment Formulation (AF) Master Problem - Problem Independent
# This file contains AF-specific master problem logic

using JuMP, Gurobi
using MathOptInterface
const MOI = MathOptInterface
using Dates

# =============================================================================
# MASTER PROBLEM (ASSIGNMENT FORMULATION)
# =============================================================================

function solve_assignment_formulation(cost_matrix, X_s, K; integer_vars=false, log_file=nothing, iteration=nothing)
    nscen, nsol = size(cost_matrix)

    model = Model(Gurobi.Optimizer)

    if log_file !== nothing
        if iteration !== nothing
            open(log_file, "a") do file
                println(file, "\n" * "="^80)
                println(file, "MASTER PROBLEM - ITERATION $iteration")
                println(file, "="^80)
                println(file, "Timestamp: $(now())")
                println(file, "Problem size: $nscen scenarios, $nsol solutions, K=$K")
                println(file, "Type: $(integer_vars ? "INTEGER" : "LINEAR RELAXATION")")
                println(file, "="^80)
            end
        end
        set_optimizer_attribute(model, "LogFile", log_file)
        set_optimizer_attribute(model, "LogToConsole", 0)
        set_optimizer_attribute(model, "OutputFlag", 1)
    else
        set_optimizer_attribute(model, "OutputFlag", 0)
    end

    if integer_vars
        sigma = [@variable(model, binary = true) for _ in 1:nsol]
        rho = [Dict(v => @variable(model, binary = true) for v in X_s[s]) for s in 1:nscen]
    else
        sigma = [@variable(model, lower_bound = 0) for _ in 1:nsol]
        rho = [Dict(v => @variable(model, lower_bound = 0) for v in X_s[s]) for s in 1:nscen]
    end

    # Objective (6a): sum only over feasible (s,v) pairs
    @objective(model, Min,
        sum(cost_matrix[s, v] * rho[s][v] for s in 1:nscen for v in X_s[s]))

    # Assignment constraints (6b): each scenario assigned to exactly one feasible solution
    assignment_constraints = [@constraint(model,
        sum(rho[s][v] for v in X_s[s]) == 1) for s in 1:nscen]

    # Selection constraints (6c): ρ_{s,v} ≤ σ_v
    for s in 1:nscen, v in X_s[s]
        @constraint(model, rho[s][v] <= sigma[v])
    end

    # K constraint (6d): Σ σ_v ≤ K
    k_constraint = @constraint(model, sum(sigma[v] for v in 1:nsol) <= K)

    optimize!(model)

    work_units = 0.0
    try
        work_units = solve_time(model)
    catch
    end

    if termination_status(model) == MOI.OPTIMAL
        return model, sigma, rho, objective_value(model), assignment_constraints, k_constraint, work_units
    else
        return model, sigma, rho, Inf, assignment_constraints, k_constraint, work_units
    end
end

# =============================================================================
# COST MATRIX BUILDING
# =============================================================================

function build_cost_matrix(
    scenarios::Vector{Vector{Float64}},
    binary_vectors::Vector{Vector{Int}},
    evaluate_cost::Function,
    problem_data;
    evaluate_cost_batch::Union{Function, Nothing}=nothing,
)
    """Build the cost matrix r_s(x) for Assignment Formulation
    
    This is problem-independent - it just calls the problem-specific evaluate_cost function.
    
    Args:
        scenarios: List of S scenarios
        binary_vectors: List of l binary solution vectors
        evaluate_cost: Problem-specific function: (solution, scenario, problem_data) -> (cost, work_units)
        problem_data: Problem-specific data structure
        evaluate_cost_batch: Optional batch evaluation function:
            (solutions, scenarios, problem_data) -> (cost_matrix, total_work_units)
    
    Returns:
        cost_matrix: S x l matrix where cost_matrix[s, x] = r_s(x)
        total_work_units: Total work units used
    """
    nscen = length(scenarios)
    nsol = length(binary_vectors)
    
    println("Building cost matrix r_s(x) for $nscen scenarios and $nsol solutions...")
    
    # If a batch evaluation function is provided, use it (more efficient)
    if evaluate_cost_batch !== nothing
        println("  Using batch cost evaluation (template model optimization)")
        cost_matrix, total_work_units = evaluate_cost_batch(binary_vectors, scenarios, problem_data)
        X_s = [findall(isfinite, cost_matrix[s, :]) for s in 1:nscen]
        println("Cost matrix built successfully (batch)")
        return cost_matrix, X_s, total_work_units
    end

    # Fallback: original per-(solution, scenario) evaluation
    cost_matrix = fill(Inf, nscen, nsol)
    total_work_units = 0.0

    for s in 1:nscen
        for x_idx in 1:nsol
            cost_val, work_units = evaluate_cost(binary_vectors[x_idx], scenarios[s], problem_data)
            cost_matrix[s, x_idx] = cost_val
            total_work_units += work_units
        end

        if s % 10 == 0
            println("  Processed $s scenarios")
        end
    end

    X_s = [findall(isfinite, cost_matrix[s, :]) for s in 1:nscen]
    println("Cost matrix built successfully")
    return cost_matrix, X_s, total_work_units
end

