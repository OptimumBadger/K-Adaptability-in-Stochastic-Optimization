# ColumnGenerationCore.jl
# Core Column Generation Algorithm - Problem Independent
# This file contains the general column generation framework that works for any problem class and formulation

using JuMP, Gurobi
using MathOptInterface
const MOI = MathOptInterface
using Printf
using Statistics
using Dates

# =============================================================================
# CONSTANTS
# =============================================================================

const DEFAULT_OPT_WORKLIMIT = 300.0

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

function sanitize_cost(c::Float64)
    """Replace non-finite costs with a large finite penalty"""
    return isfinite(c) ? c : 1.0e12
end

# =============================================================================
# UTILITY FUNCTIONS FOR SOLUTION POOL
# =============================================================================

function collect_pool_rc_cap10(model, x, dual_mu, use_solution_pool, formulation_type="AF")
    """Collect up to 10 solutions from pool that satisfy reduced cost condition
    
    Args:
        model: JuMP model
        x: Variable reference (for AF) or pi (for SF)
        dual_mu: Dual value from K constraint
        use_solution_pool: Whether to use solution pool
        formulation_type: "AF" or "SF" - determines reduced cost calculation
    """
    if !use_solution_pool
        return Vector{Vector{Float64}}(), Float64[]
    end
    
    opt = backend(model)
    kept_solutions = Vector{Vector{Float64}}()
    kept_objs = Float64[]
    
    try
        solcount = MOI.get(opt, Gurobi.ModelAttribute("SolCount"))
        max_solutions = min(10, solcount)
        
        for k in 0:max_solutions-1
            MOI.set(opt, MOI.RawOptimizerAttribute("SolutionNumber"), k)
            pool_obj = MOI.get(opt, Gurobi.ModelAttribute("PoolObjVal"))
            x_k = [MOI.get(opt, Gurobi.VariableAttribute("Xn"), JuMP.index(x[i])) for i in 1:length(x)]
            
            # Reduced cost calculation depends on formulation
            if formulation_type == "AF"
                rc_k = pool_obj + dual_mu  # For AF: rc = obj_pricing + μ
                if rc_k > 1  # Positive reduced cost (improving)
                    push!(kept_solutions, x_k)
                    push!(kept_objs, pool_obj)
                end
            else  # SF
                rc_k = pool_obj  # For SF: pricing objective IS the reduced cost
                if rc_k < -1  # Negative reduced cost (improving)
                    push!(kept_solutions, x_k)
                    push!(kept_objs, pool_obj)
                end
            end
        end
    catch e
        println("  Warning: solution pool access failed: $e")
    end
    
    if isempty(kept_solutions)
        return Vector{Vector{Float64}}(), Float64[]
    end
    
    return kept_solutions, kept_objs
end

function return_pricing_payload(kept_solutions, kept_objs, pricing_work_units, return_nodes, model)
    """Return pricing results in the correct format"""
    if return_nodes
        nodes = 0
        try
            nodes = MOI.get(backend(model), Gurobi.ModelAttribute("NodeCount"))
        catch
        end
        return kept_solutions, kept_objs, pricing_work_units, nodes
    else
        return kept_solutions, kept_objs, pricing_work_units
    end
end

# Note: solve_assignment_formulation and build_cost_matrix have been moved to AssignmentFormulation.jl
# Include AssignmentFormulation.jl when using AF formulation

# =============================================================================
# COLUMN GENERATION ALGORITHM - ASSIGNMENT FORMULATION
# =============================================================================

# =============================================================================
# MAIN COLUMN GENERATION ALGORITHM
# =============================================================================

function column_generation_algorithm(
    scenarios::Vector{Vector{Float64}},
    initial_binary_vectors::Vector{Vector{Int}},
    problem_data,
    K::Int,
    max_iterations::Int;
    # Problem-specific functions:
    evaluate_cost::Function,              # (solution, scenario, problem_data) -> (cost, work_units)
    build_pricing_model::Function,         # (scenarios, dual_lambda, dual_mu, problem_data, pricing_type, work_limit) -> (solutions, objectives, work_units)
    calculate_m_values::Function,          # (scenarios, dual_lambda, problem_data) -> m_values (for Type 1 pricing)
    # Algorithm parameters:
    pricing_type::Int=2,
    use_solution_pool::Bool=true,
    total_work_budget::Float64=30000.0,
    opt_worklimit::Float64=DEFAULT_OPT_WORKLIMIT,
    master_log_file::Union{String, Nothing}=nothing,
    pricing_log_file::Union{String, Nothing}=nothing
)
    """
    Main Column Generation Algorithm - Problem Independent
    
    Args:
        scenarios: List of S scenarios (loaded from file)
        initial_binary_vectors: List of l initial binary vectors (loaded from file)
        problem_data: Problem-specific data structure
        K: Number of solutions to select
        max_iterations: Maximum number of iterations
        evaluate_cost: Function to compute r_s(x)
        build_pricing_model: Function to build and solve pricing subproblem
        calculate_m_values: Function to calculate m^s values (for Type 1 pricing)
        pricing_type: 1 or 2
        use_solution_pool: Whether to use solution pool
        total_work_budget: Global work budget
        opt_worklimit: Work limit for Phase 1 of pricing
        master_log_file: Optional log file for master problem
        pricing_log_file: Optional log file for pricing subproblem
    
    Returns:
        results: Dictionary with all results
    """
    nscen = length(scenarios)
    
    println("="^80)
    println("COLUMN GENERATION ALGORITHM")
    println("="^80)
    println("Parameters: nscen=$nscen, nsol=$(length(initial_binary_vectors)), K=$K, max_iter=$max_iterations")
    println("Pricing Type: $pricing_type")
    println("Use Solution Pool: $use_solution_pool")
    println()
    
    # Global work budget tracker
    total_master_work_units = 0.0
    total_pricing_work_units = 0.0
    total_update_work_units = 0.0
    last_master_work_cumulative = 0.0
    
    # Tracking data
    iteration_master_data = []
    iteration_pricing_data = []
    iteration_columns_data = []
    pricing_iteration_analysis = []
    actual_iterations = 0
    total_solutions_added = 0
    
    # Stage 1: Initialization (NO solving from scratch - vectors already loaded!)
    println("Stage 1: Initialization")
    println("  Loaded $(length(initial_binary_vectors)) initial binary vectors from file")
    println("  Loaded $nscen scenarios from file")
    
    binary_vectors = copy(initial_binary_vectors)  # Will be extended with new solutions
    nsol = length(binary_vectors)
    
    # Build initial cost matrix
    cost_matrix, initial_matrix_work_units = build_cost_matrix(
        scenarios, binary_vectors, evaluate_cost, problem_data
    )
    total_update_work_units += initial_matrix_work_units
    
    # Initialize master problem
    model, sigma, rho, master_obj, assignment_constraints, k_constraint, master_work = 
        solve_assignment_formulation(cost_matrix, K; log_file=master_log_file, iteration=0)
    
    if master_obj == Inf
        println("❌ Error: Initial master problem is infeasible!")
        return Dict(
            "status" => "FAILED",
            "final_objective" => Inf,
            "iterations" => 0,
            "total_master_work_units" => 0.0,
            "total_pricing_work_units" => 0.0,
            "total_update_work_units" => 0.0,
            "total_solutions_added" => 0
        )
    end
    
    total_master_work_units += master_work
    last_master_work_cumulative = master_work
    println("  Initial master problem objective: $(round(master_obj, digits=2))")
    println()
    
    # Column Generation Loop
    for iteration in 1:max_iterations
        println("\033[34m================ ITERATION $iteration ===================\033[0m")
        
        # Print available work units
        used_work = total_master_work_units + total_pricing_work_units + total_update_work_units
        available_work = max(0.0, total_work_budget - used_work)
        println("Available work units: $(round(available_work, digits=2)) / $total_work_budget")
        
        # Stage 2: Master Problem Solution
        println("Stage 2: Solving Master Problem")
        
        # Add iteration marker to master log file before solving
        if master_log_file !== nothing
            open(master_log_file, "a") do file
                println(file, "\n" * "="^80)
                println(file, "MASTER PROBLEM - ITERATION $iteration")
                println(file, "="^80)
                println(file, "Timestamp: $(now())")
                println(file, "Problem size: $nscen scenarios, $nsol solutions, K=$K")
                println(file, "Type: LINEAR RELAXATION")
                println(file, "="^80)
            end
        end
        
        master_work_units = 0.0
        try
            optimize!(model)
            current_master_work = solve_time(model)
            master_work_units = max(0.0, current_master_work - last_master_work_cumulative)
            total_master_work_units += master_work_units
            last_master_work_cumulative = current_master_work
        catch e
            println("Error getting master work units: $e")
        end
        
        if termination_status(model) != MOI.OPTIMAL
            println("❌ Master problem failed to solve")
            # Add failure note to log file
            if master_log_file !== nothing
                open(master_log_file, "a") do file
                    println(file, "\nMaster Problem Results:")
                    println(file, "  Status: FAILED")
                    println(file, "="^80)
                end
            end
            break
        end
        
        master_obj = objective_value(model)
        println("  Master objective: $(round(master_obj, digits=2))")
        println("  Master work units: $(round(master_work_units, digits=2))")
        
        # Add summary to master log file
        if master_log_file !== nothing
            open(master_log_file, "a") do file
                println(file, "\nMaster Problem Results:")
                println(file, "  Objective: $(round(master_obj, digits=2))")
                println(file, "  Work units: $(round(master_work_units, digits=2))")
                println(file, "="^80)
            end
        end
        
        push!(iteration_master_data, (iteration=iteration, master_obj=master_obj))
        
        # Stage 3: Dual Extraction
        println("Stage 3: Extracting Dual Solutions")
        
        dual_lambda = dual.(assignment_constraints)  # Extract duals from assignment constraints
        dual_mu = dual(k_constraint)  # Extract dual from K constraint
        
        println("  Dual μ value: $(round(dual_mu, digits=6))")
        
        # Stage 4: Pricing Subproblem
        println("Stage 4: Solving Pricing Subproblem")
        
        # Compute remaining work budget
        used_work = total_master_work_units + total_pricing_work_units + total_update_work_units
        remaining_work = max(0.0, total_work_budget - used_work)
        println("  Remaining work budget: $(round(remaining_work, digits=2))")
        
        if remaining_work <= 0.0
            println("  Global work budget exhausted. Terminating.")
            break
        end
        
        # Solve pricing subproblem using specified pricing type
        new_solutions = Vector{Vector{Float64}}()
        pool_objs = Float64[]
        pricing_work_units = 0.0
        
        try
            new_solutions, pool_objs, pricing_work_units = 
                build_pricing_model(scenarios, dual_lambda, dual_mu, problem_data, pricing_type, 
                                 remaining_work, opt_worklimit, use_solution_pool, false, pricing_log_file, iteration)
            
            total_pricing_work_units += pricing_work_units
            println("  Pricing work units: $(round(pricing_work_units, digits=2))")
            
        catch e
            println("❌ Error in pricing subproblem: $e")
            break
        end
        
        # Terminate if pricing found no improving solutions
        if isempty(new_solutions)
            println("  No improving solutions from pricing. Terminating algorithm.")
            break
        end
        
        # Filter by reduced-cost condition and cap to 10
        rc_ok_indices = [idx for idx in 1:length(pool_objs) if pool_objs[idx] + dual_mu > 1]
        if isempty(rc_ok_indices)
            println("  No RC-improving solutions from pricing. Terminating algorithm.")
            break
        end
        
        keep = rc_ok_indices[1:min(10, length(rc_ok_indices))]
        new_solutions = new_solutions[keep]
        pool_objs = pool_objs[keep]
        
        println("  Will add $(length(new_solutions)) columns (rank, obj, rc):")
        for (rank, idx) in enumerate(keep)
            objv = pool_objs[rank]
            rcv = objv + dual_mu
            println("    $rank) obj=$(round(objv, digits=6))\trc=$(round(rcv, digits=6))")
        end
        
        # Stage 5: Convergence Check & Master Problem Update
        println("Stage 5: Convergence Check & Master Problem Update")
        
        obj_pricing = minimum(pool_objs)
        reduced_cost = obj_pricing + dual_mu
        println("  Pricing objective: $(round(obj_pricing, digits=6))")
        println("  Reduced cost (obj_pricing + μ): $(round(reduced_cost, digits=6))")
        
        push!(iteration_pricing_data, (
            iteration=iteration, 
            obj_pricing=obj_pricing, 
            dual_mu=dual_mu, 
            obj_pricing_plus_dual_mu=reduced_cost
        ))
        
        if reduced_cost < 1  # Convergence tolerance
            println("✅ Algorithm converged! Reduced cost < 0")
            push!(iteration_columns_data, (iteration=iteration, num_added=0))
            break
        else
            # Add new columns to master problem
            println("  Adding $(length(new_solutions)) column(s) to master problem")
            local_added = 0
            local_update_work_units = 0.0
            
            for x_new in new_solutions
                # Convert solution to Int vector (pricing returns Float64, but evaluate_cost expects Int)
                x_new_int = round.(Int, x_new)
                
                # Calculate costs for all scenarios
                new_column_costs = zeros(nscen)
                for s in 1:nscen
                    cost_val, work_units = evaluate_cost(x_new_int, scenarios[s], problem_data)
                    new_column_costs[s] = cost_val
                    local_update_work_units += work_units
                end
                
                # Add new variables
                new_sigma = @variable(model, lower_bound = 0)
                new_rho = [@variable(model, lower_bound = 0) for s in 1:nscen]
                push!(sigma, new_sigma)
                for s in 1:nscen
                    push!(rho[s], new_rho[s])
                end
                
                # Update objective and constraints
                for s in 1:nscen
                    set_objective_coefficient(model, new_rho[s], new_column_costs[s])
                    set_normalized_coefficient(assignment_constraints[s], new_rho[s], 1.0)
                    @constraint(model, new_rho[s] <= new_sigma)
                end
                set_normalized_coefficient(k_constraint, new_sigma, 1.0)
                
                # Update cost matrix and binary vectors
                cost_matrix = hcat(cost_matrix, new_column_costs)
                push!(binary_vectors, x_new_int)
                nsol += 1
                local_added += 1
            end
            
            total_update_work_units += local_update_work_units
            total_solutions_added += local_added
            println("  Total solutions: $nsol")
            println("  Update work units: $(round(local_update_work_units, digits=2))")
            push!(iteration_columns_data, (iteration=iteration, num_added=local_added))
        end
        
        actual_iterations = iteration
        println()
    end
    
    # Final solution
    println("="^80)
    println("FINALIZING COLUMN GENERATION")
    println("="^80)
    
    optimize!(model)
    
    if termination_status(model) == MOI.OPTIMAL
        final_sigma_values = value.(sigma)
        final_rho_values = [value.(rho[s]) for s in 1:nscen]
        final_objective_lp_total = objective_value(model)
        # Divide by nscen to get average cost per scenario
        final_objective_lp = final_objective_lp_total / nscen
        
        # Find selected solutions (LP relaxation)
        selected_solutions_lp = findall(x -> x > 1e-6, final_sigma_values)
        
        println("  Final Results (Linear Relaxation):")
        println("  Iterations: $actual_iterations")
        println("  Final objective (LP, total): $(round(final_objective_lp_total, digits=2))")
        println("  Final objective (LP, average): $(round(final_objective_lp, digits=2))")
        println("  Selected solutions (LP): $selected_solutions_lp")
        
        # Solve integer version of the master problem
        println("\n=== Solving Integer Master Problem ===")
        integer_model, integer_sigma, integer_rho, integer_obj_total, integer_assignment_constraints, integer_k_constraint, integer_work = 
            solve_assignment_formulation(cost_matrix, K; integer_vars=true, log_file=master_log_file, iteration="FINAL_IP")
        
        if termination_status(integer_model) == MOI.OPTIMAL
            integer_sigma_values = value.(integer_sigma)
            integer_selected_solutions = findall(x -> x > 1e-6, integer_sigma_values)
            # Divide by nscen to get average cost per scenario
            integer_obj = integer_obj_total / nscen
            
            println("  Integer Master Problem Results:")
            println("  Integer objective (IP, total): $(round(integer_obj_total, digits=2))")
            println("  Integer objective (IP, average): $(round(integer_obj, digits=2))")
            println("  Selected solutions (IP): $integer_selected_solutions")
            if final_objective_lp != Inf && integer_obj != Inf
                gap = integer_obj - final_objective_lp
                gap_percent = (gap / final_objective_lp) * 100
                println("  Gap (IP - LP): $(round(gap, digits=2))")
                println("  Gap percentage: $(round(gap_percent, digits=2))%")
            end
            
            # Return integer selected solutions
            return Dict(
                "status" => "OPTIMAL",
                "final_objective_lp" => final_objective_lp,
                "final_objective_ip" => integer_obj,
                "iterations" => actual_iterations,
                "selected_solutions" => integer_selected_solutions,
                "binary_vectors" => binary_vectors,
                "cost_matrix" => cost_matrix,
                "total_master_work_units" => total_master_work_units,
                "total_pricing_work_units" => total_pricing_work_units,
                "total_update_work_units" => total_update_work_units,
                "total_solutions_added" => total_solutions_added,
                "iteration_master_data" => iteration_master_data,
                "iteration_pricing_data" => iteration_pricing_data,
                "iteration_columns_data" => iteration_columns_data
            )
        else
            println("  Integer master problem failed to solve")
            integer_obj = Inf
            return Dict(
                "status" => "OPTIMAL",
                "final_objective_lp" => final_objective_lp,
                "final_objective_ip" => integer_obj,
                "iterations" => actual_iterations,
                "selected_solutions" => selected_solutions_lp,
                "binary_vectors" => binary_vectors,
                "cost_matrix" => cost_matrix,
                "total_master_work_units" => total_master_work_units,
                "total_pricing_work_units" => total_pricing_work_units,
                "total_update_work_units" => total_update_work_units,
                "total_solutions_added" => total_solutions_added,
                "iteration_master_data" => iteration_master_data,
                "iteration_pricing_data" => iteration_pricing_data,
                "iteration_columns_data" => iteration_columns_data
            )
        end
    else
        println("❌ Final master problem failed to solve")
        return Dict(
            "status" => "FAILED",
            "final_objective_lp" => Inf,
            "final_objective_ip" => Inf,
            "iterations" => actual_iterations,
            "selected_solutions" => Int[],
            "binary_vectors" => binary_vectors,
            "cost_matrix" => cost_matrix,
            "total_master_work_units" => total_master_work_units,
            "total_pricing_work_units" => total_pricing_work_units,
            "total_update_work_units" => total_update_work_units,
            "total_solutions_added" => total_solutions_added,
            "iteration_master_data" => iteration_master_data,
            "iteration_pricing_data" => iteration_pricing_data,
            "iteration_columns_data" => iteration_columns_data,
            "type1_metrics" => type1_metrics,
            "type2_metrics" => type2_metrics
        )
    end
end

# =============================================================================
# SF COLUMN GENERATION ALGORITHM
# =============================================================================

function sf_column_generation_algorithm(
    scenarios::Vector{Vector{Float64}},
    initial_subsets::Vector{Vector{Int}},  # Initial π vectors (scenario selection vectors)
    problem_data,
    K::Int,
    max_iterations::Int;
    # Problem-specific functions:
    evaluate_subset_cost::Function,        # (subset_vector, scenarios, problem_data) -> (cost, work_units)
    build_pricing_model::Function,         # (scenarios, dual_lambda, dual_mu, problem_data, pricing_type, work_limit, opt_worklimit, use_solution_pool, return_nodes, log_file, iteration) -> (solutions, objectives, work_units)
    # Algorithm parameters:
    pricing_type::Int=2,
    use_solution_pool::Bool=true,
    total_work_budget::Float64=30000.0,
    opt_worklimit::Float64=DEFAULT_OPT_WORKLIMIT,
    master_log_file::Union{String, Nothing}=nothing,
    pricing_log_file::Union{String, Nothing}=nothing
)
    """Main SF Column Generation Algorithm - Problem Independent
    
    Args:
        scenarios: List of S scenarios (loaded from file)
        initial_subsets: List of initial subset vectors (π vectors) - can be empty if init_type is used
        problem_data: Problem-specific data structure
        K: Number of subsets to select
        max_iterations: Maximum number of iterations
        evaluate_subset_cost: Function to compute f(A)
        build_pricing_model: Function to build and solve pricing subproblem
        pricing_type: 1 or 2
        use_solution_pool: Whether to use solution pool
        total_work_budget: Global work budget
        opt_worklimit: Work limit for Phase 1 of pricing
        master_log_file: Optional log file for master problem
        pricing_log_file: Optional log file for pricing subproblem
    
    Returns:
        results: Dictionary with all results
    """
    nscen = length(scenarios)
    
    println("="^80)
    println("SF COLUMN GENERATION ALGORITHM")
    println("="^80)
    println("Parameters: nscen=$nscen, nsubsets=$(length(initial_subsets)), K=$K, max_iter=$max_iterations")
    println("Pricing Type: $pricing_type")
    println("Use Solution Pool: $use_solution_pool")
    println()
    
    # Global work budget tracker
    total_master_work_units = 0.0
    total_pricing_work_units = 0.0
    total_update_work_units = 0.0
    last_master_work_cumulative = 0.0
    
    # Tracking data
    iteration_master_data = []
    iteration_pricing_data = []
    iteration_columns_data = []
    pricing_iteration_analysis = []
    actual_iterations = 0
    total_subsets_added = 0
    
    # Stage 1: Initialization
    println("Stage 1: Initialization")
    
    # Use provided initial_subsets (must be loaded from file)
    if isempty(initial_subsets)
        error("Initial subsets must be provided (loaded from file). The initial_subsets parameter cannot be empty.")
    end
    
    println("  Using provided initial subsets from file...")
    println("  Loaded $(length(initial_subsets)) initial subsets")
    println("  Loaded $nscen scenarios from file")
    
    # Build initial cost vector and z_matrix
    cost_vector, z_matrix, initial_matrix_work_units = build_cost_vector(
        scenarios, initial_subsets, evaluate_subset_cost, problem_data
    )
    total_update_work_units += initial_matrix_work_units
    
    nsubsets = length(cost_vector)
    
    # Initialize master problem
    model, v_variables, master_obj, partition_constraints, k_constraint, master_work = 
        solve_sf_master_problem(cost_vector, z_matrix, K; log_file=master_log_file, iteration=0)
    
    if master_obj == Inf
        println("❌ Error: Initial master problem is infeasible!")
        return Dict(
            "status" => "FAILED",
            "final_objective_lp" => Inf,
            "final_objective_ip" => Inf,
            "iterations" => 0,
            "total_master_work_units" => 0.0,
            "total_pricing_work_units" => 0.0,
            "total_update_work_units" => 0.0,
            "total_subsets_added" => 0
        )
    end
    
    total_master_work_units += master_work
    last_master_work_cumulative = master_work
    println("  Initial master problem objective: $(round(master_obj, digits=2))")
    println()
    
    # Column Generation Loop
    for iteration in 1:max_iterations
        println("\033[34m================ ITERATION $iteration ===================\033[0m")
        
        # Print available work units
        used_work = total_master_work_units + total_pricing_work_units + total_update_work_units
        available_work = max(0.0, total_work_budget - used_work)
        println("Available work units: $(round(available_work, digits=2)) / $total_work_budget")
        
        # Stage 2: Master Problem Solution
        println("Stage 2: Solving Master Problem")
        
        # Add iteration marker to master log file
        if master_log_file !== nothing
            open(master_log_file, "a") do file
                println(file, "\n" * "="^80)
                println(file, "MASTER PROBLEM - ITERATION $iteration")
                println(file, "="^80)
                println(file, "Timestamp: $(now())")
                println(file, "Problem size: $nscen scenarios, $nsubsets subsets, K=$K")
                println(file, "Type: LINEAR RELAXATION")
                println(file, "="^80)
            end
        end
        
        master_work_units = 0.0
        try
            optimize!(model)
            current_master_work = solve_time(model)
            master_work_units = max(0.0, current_master_work - last_master_work_cumulative)
            total_master_work_units += master_work_units
            last_master_work_cumulative = current_master_work
        catch e
            println("Error getting master work units: $e")
        end
        
        if termination_status(model) != MOI.OPTIMAL
            println("❌ Master problem failed to solve")
            break
        end
        
        master_obj = objective_value(model)
        println("  Master objective: $(round(master_obj, digits=2))")
        println("  Master work units: $(round(master_work_units, digits=2))")
        push!(iteration_master_data, (iteration=iteration, master_obj=master_obj))
        
        # Stage 3: Dual Extraction
        println("Stage 3: Extracting Dual Solutions")
        
        dual_lambda = dual.(partition_constraints)  # Extract duals from partition constraints
        dual_mu = dual(k_constraint)  # Extract dual from K constraint
        
        println("  Dual μ value: $(round(dual_mu, digits=6))")
        
        # Stage 4: Pricing Subproblem
        println("Stage 4: Solving Pricing Subproblem")
        
        # Compute remaining work budget
        used_work = total_master_work_units + total_pricing_work_units + total_update_work_units
        remaining_work = max(0.0, total_work_budget - used_work)
        println("  Remaining work budget: $(round(remaining_work, digits=2))")
        
        if remaining_work <= 0.0
            println("  Global work budget exhausted. Terminating.")
            break
        end
        
        # Solve pricing subproblem
        new_subsets = Vector{Vector{Float64}}()
        pool_objs = Float64[]
        pricing_work_units = 0.0
        
        try
            new_subsets, pool_objs, pricing_work_units = 
                build_pricing_model(scenarios, dual_lambda, dual_mu, problem_data, pricing_type, 
                                 remaining_work, opt_worklimit, use_solution_pool, false, pricing_log_file, iteration)
            
            total_pricing_work_units += pricing_work_units
            println("  Pricing work units: $(round(pricing_work_units, digits=2))")
            
        catch e
            println("❌ Error in pricing subproblem: $e")
            break
        end
        
        # Terminate if pricing found no improving solutions
        if isempty(new_subsets)
            println("  No improving solutions from pricing. Terminating algorithm.")
            break
        end
        
        # Filter by reduced-cost condition and cap to 10
        # For SF: pricing objective IS the reduced cost (negative is improving)
        rc_ok_indices = [idx for idx in 1:length(pool_objs) if pool_objs[idx] < -1]
        if isempty(rc_ok_indices)
            println("  No RC-improving solutions from pricing. Terminating algorithm.")
            break
        end
        
        keep = rc_ok_indices[1:min(10, length(rc_ok_indices))]
        new_subsets = new_subsets[keep]
        pool_objs = pool_objs[keep]
        
        println("  Will add $(length(new_subsets)) columns (rank, obj, rc):")
        for (rank, idx) in enumerate(keep)
            objv = pool_objs[rank]
            rcv = objv  # For SF: pricing objective IS the reduced cost
            println("    $rank) obj=$(round(objv, digits=6))\trc=$(round(rcv, digits=6))")
        end
        
        # Stage 5: Convergence Check & Master Problem Update
        println("Stage 5: Convergence Check & Master Problem Update")
        
        obj_pricing = minimum(pool_objs)
        reduced_cost = obj_pricing  # For SF: pricing objective IS the reduced cost
        println("  Pricing objective (reduced cost): $(round(obj_pricing, digits=6))")
        
        push!(iteration_pricing_data, (
            iteration=iteration, 
            obj_pricing=obj_pricing, 
            dual_mu=dual_mu, 
            reduced_cost=reduced_cost
        ))
        
        if reduced_cost >= -1  # Convergence tolerance
            println("✅ Algorithm converged! Reduced cost >= 0")
            push!(iteration_columns_data, (iteration=iteration, num_added=0))
            break
        else
            # Add new columns to master problem
            println("  Adding $(length(new_subsets)) column(s) to master problem")
            local_added = 0
            local_update_work_units = 0.0
            
            for pi_new in new_subsets
                # Convert to Int vector
                pi_new_int = round.(Int, pi_new)
                
                # Calculate cost for this subset
                new_cost, work_units = evaluate_subset_cost(pi_new_int, scenarios, problem_data)
                local_update_work_units += work_units
                
                # Sanitize cost
                new_cost = sanitize_cost(new_cost)
                push!(cost_vector, new_cost)
                
                # Update z_matrix with new column
                z_matrix = hcat(z_matrix, pi_new_int)
                nsubsets += 1
                
                # Add new v_A variable to master problem
                new_v = @variable(model, lower_bound = 0, upper_bound = 1)
                push!(v_variables, new_v)
                
                # Update objective: set coefficient for new v_A variable
                set_objective_coefficient(model, new_v, new_cost)
                
                # Update the constraint matrix entries
                # Partition constraints: each scenario sums to 1
                for s in 1:nscen
                    set_normalized_coefficient(partition_constraints[s], new_v, pi_new_int[s])
                end
                
                # Cardinality constraint: sum v_A <= K
                set_normalized_coefficient(k_constraint, new_v, 1.0)
                
                local_added += 1
            end
            
            total_update_work_units += local_update_work_units
            total_subsets_added += local_added
            println("  Total subsets: $nsubsets")
            println("  Update work units: $(round(local_update_work_units, digits=2))")
            push!(iteration_columns_data, (iteration=iteration, num_added=local_added))
        end
        
        actual_iterations = iteration
        println()
    end
    
    # Final solution
    println("="^80)
    println("FINALIZING COLUMN GENERATION")
    println("="^80)
    
    optimize!(model)
    
    if termination_status(model) == MOI.OPTIMAL
        final_v_values = value.(v_variables)
        final_objective_lp = objective_value(model)
        
        # Find selected subsets (LP relaxation)
        selected_subsets_lp = findall(x -> x > 1e-6, final_v_values)
        
        println("  Final Results (Linear Relaxation):")
        println("  Iterations: $actual_iterations")
        println("  Final objective (LP): $(round(final_objective_lp, digits=2))")
        println("  Selected subsets (LP): $selected_subsets_lp")
        
        # Solve integer version of the master problem
        println("\n=== Solving Integer Master Problem ===")
        integer_model, integer_v_variables, integer_obj, integer_partition_constraints, integer_k_constraint, integer_work = 
            solve_sf_master_problem(cost_vector, z_matrix, K; integer_vars=true, log_file=master_log_file, iteration="FINAL_IP")
        
        if termination_status(integer_model) == MOI.OPTIMAL
            integer_v_values = value.(integer_v_variables)
            integer_selected_subsets = findall(x -> x > 0.5, integer_v_values)
            
            println("  Integer Master Problem Results:")
            println("  Integer objective (IP): $(round(integer_obj, digits=2))")
            println("  Selected subsets (IP): $integer_selected_subsets")
            if final_objective_lp != Inf && integer_obj != Inf
                gap = integer_obj - final_objective_lp
                gap_percent = (gap / final_objective_lp) * 100
                println("  Gap (IP - LP): $(round(gap, digits=2))")
                println("  Gap percentage: $(round(gap_percent, digits=2))%")
            end
            
            return Dict(
                "status" => "OPTIMAL",
                "final_objective_lp" => final_objective_lp,
                "final_objective_ip" => integer_obj,
                "iterations" => actual_iterations,
                "selected_subsets" => integer_selected_subsets,
                "cost_vector" => cost_vector,
                "z_matrix" => z_matrix,
                "total_master_work_units" => total_master_work_units,
                "total_pricing_work_units" => total_pricing_work_units,
                "total_update_work_units" => total_update_work_units,
                "total_subsets_added" => total_subsets_added,
                "iteration_master_data" => iteration_master_data,
                "iteration_pricing_data" => iteration_pricing_data,
                "iteration_columns_data" => iteration_columns_data
            )
        else
            println("  Integer master problem failed to solve")
            integer_obj = Inf
            return Dict(
                "status" => "OPTIMAL",
                "final_objective_lp" => final_objective_lp,
                "final_objective_ip" => integer_obj,
                "iterations" => actual_iterations,
                "selected_subsets" => selected_subsets_lp,
                "cost_vector" => cost_vector,
                "z_matrix" => z_matrix,
                "total_master_work_units" => total_master_work_units,
                "total_pricing_work_units" => total_pricing_work_units,
                "total_update_work_units" => total_update_work_units,
                "total_subsets_added" => total_subsets_added,
                "iteration_master_data" => iteration_master_data,
                "iteration_pricing_data" => iteration_pricing_data,
                "iteration_columns_data" => iteration_columns_data
            )
        end
    else
        println("❌ Final master problem failed to solve")
        return Dict(
            "status" => "FAILED",
            "final_objective_lp" => Inf,
            "final_objective_ip" => Inf,
            "iterations" => actual_iterations,
            "selected_subsets" => Int[],
            "cost_vector" => cost_vector,
            "z_matrix" => z_matrix,
            "total_master_work_units" => total_master_work_units,
            "total_pricing_work_units" => total_pricing_work_units,
            "total_update_work_units" => total_update_work_units,
            "total_subsets_added" => total_subsets_added,
            "iteration_master_data" => iteration_master_data,
            "iteration_pricing_data" => iteration_pricing_data,
            "iteration_columns_data" => iteration_columns_data
        )
    end
end

