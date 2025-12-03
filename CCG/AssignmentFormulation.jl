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

function solve_assignment_formulation(cost_matrix, K; integer_vars=false, log_file=nothing, iteration=nothing)
    """Solve the Assignment Formulation (AF) Master Problem - LINEAR RELAXATION or INTEGER
    
    This is IDENTICAL for all problem classes.
    
    Args:
        cost_matrix: S x l matrix where cost_matrix[s, x] = r_s(x)
        K: Number of solutions to select
        integer_vars: If true, solve as integer program; if false, linear relaxation
        log_file: Optional log file path (will append with iteration markers)
        iteration: Optional iteration number for logging
    
    Returns:
        model: JuMP model
        sigma: Vector of σ variables (solution weights)
        rho: Vector of vectors of ρ variables (scenario-solution assignments)
        objective_value: Objective value
        assignment_constraints: Constraint references for dual extraction
        k_constraint: K constraint reference for dual extraction
        work_units: Work units used
    """
    nscen, nsol = size(cost_matrix)
    
    model = Model(Gurobi.Optimizer)
    
    if log_file !== nothing
        # Append iteration marker to log file
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
        # Enable file logging but disable console output
        set_optimizer_attribute(model, "LogFile", log_file)
        set_optimizer_attribute(model, "LogToConsole", 0)
        set_optimizer_attribute(model, "OutputFlag", 1)  # Enable output (will go to file, not console due to LogToConsole=0)
    else
        # No logging - disable all output
        set_optimizer_attribute(model, "OutputFlag", 0)
    end
    
    # Decision Variables - LINEAR RELAXATION (continuous) or INTEGER
    if integer_vars
        sigma = [@variable(model, binary = true) for x in 1:nsol]  # σ_x: weight of solution x (binary)
        rho = [[@variable(model, binary = true) for x in 1:nsol] for s in 1:nscen]  # ρ_{s,x}: assignment of scenario s to solution x (binary)
    else
        sigma = [@variable(model, lower_bound = 0) for x in 1:nsol]  # σ_x: weight of solution x (continuous)
        rho = [[@variable(model, lower_bound = 0) for x in 1:nsol] for s in 1:nscen]  # ρ_{s,x}: assignment of scenario s to solution x (continuous)
    end
    
    # Objective Function (5a)
    @objective(model, Min, sum(cost_matrix[s, x] * rho[s][x] for s in 1:nscen for x in 1:nsol))
    
    # Constraints - Store references for dual extraction
    # Assignment constraints (5b): Each scenario must be assigned to exactly one solution
    assignment_constraints = @constraint(model, [s in 1:nscen], sum(rho[s][x] for x in 1:nsol) == 1)
    
    # Selection constraints (5c): ρ_{s,x} <= σ_x for all s, x
    @constraint(model, [s in 1:nscen, x in 1:nsol], rho[s][x] <= sigma[x])
    
    # K constraint (5d): Sum of σ_x <= K
    k_constraint = @constraint(model, sum(sigma[x] for x in 1:nsol) <= K)
    
    # Solve
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

function build_cost_matrix(scenarios::Vector{Vector{Float64}}, 
                          binary_vectors::Vector{Vector{Int}},
                          evaluate_cost::Function,
                          problem_data)
    """Build the cost matrix r_s(x) for Assignment Formulation
    
    This is problem-independent - it just calls the problem-specific evaluate_cost function.
    
    Args:
        scenarios: List of S scenarios
        binary_vectors: List of l binary solution vectors
        evaluate_cost: Problem-specific function: (solution, scenario, problem_data) -> (cost, work_units)
        problem_data: Problem-specific data structure
    
    Returns:
        cost_matrix: S x l matrix where cost_matrix[s, x] = r_s(x)
        total_work_units: Total work units used
    """
    nscen = length(scenarios)
    nsol = length(binary_vectors)
    
    println("Building cost matrix r_s(x) for $nscen scenarios and $nsol solutions...")
    
    cost_matrix = zeros(nscen, nsol)
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
    
    println("Cost matrix built successfully")
    return cost_matrix, total_work_units
end

