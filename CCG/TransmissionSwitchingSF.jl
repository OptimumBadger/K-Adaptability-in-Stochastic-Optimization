# TransmissionSwitchingSF.jl
# Problem-specific implementations for Transmission Switching Column Generation - Subset Formulation (SF)
# This file provides the interface functions required by ColumnGenerationCore.jl for SF
# This file is used to evaluate the cost of a subset of scenarios for the Transmission Switching problem with appropriate fixed lines.

using JuMP, Gurobi
using MathOptInterface
const MOI = MathOptInterface
using Printf
using Statistics
using Random, Distributions
using Dates
using PowerModels

# Include shared utilities
include("ColumnGenerationCore.jl")
include("SubsetFormulation.jl")

# Include TS utilities (parameters_TSpm, extract_case_name)
include("../Samples/TS/TSUtilities.jl")

# Include heuristic functions for consistency (demand multiplier, penalty)
script_dir = @__DIR__
project_root = dirname(script_dir)
include(joinpath(project_root, "Transmission_Switching", "Heuristic", "Candidate_List_Generation_Phase.jl"))

# =============================================================================
# CONSTANTS
# =============================================================================

const DEFAULT_OPT_WORKLIMIT = 300.0

# =============================================================================
# SUBSET COST EVALUATION f(A)
# =============================================================================

function evaluate_subset_cost_TS(subset_vector::Vector{Int}, scenarios::Vector{Vector{Float64}}, problem_data::Dict)
    """Evaluate cost f(A) for a subset of scenarios (Transmission Switching)
    
    Args:
        subset_vector: Binary vector of length S, where subset_vector[s] = 1 if scenario s is in subset A
        scenarios: List of S scenarios (each is a demand vector)
        problem_data: Dict containing PowerModels data structure
    
    Returns:
        (cost::Float64, work_units::Float64)
    """
    B = problem_data["bus"]
    L = problem_data["branch"]
    nb = length(B)
    nl = length(L)
    B_ij, fbar_ij, p_max, p_min, A, c_g = parameters_TSpm(problem_data)
    ref = 1
    nscen = length(scenarios)
    
    # Use instance-specific penalty (consistent with heuristic)
    instance_file = get(problem_data, "instance_file", nothing)
    case_name = instance_file !== nothing ? extract_case_name(instance_file) : nothing
    penalty = get_penalty(instance_file=instance_file, case_name=case_name)
    
    # Big-M for DC equality relaxation when x_l = 0
    M_ij = [12.6 * abs(B_ij[l]) for l in 1:nl]
    
    # Check if subset is empty
    if sum(subset_vector) == 0
        return 0.0, 0.0  # Cost of empty subset is 0
    end
    
    # Create model
    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 0)  # Suppress output
    set_optimizer_attribute(model, "MIPGap", 0.001)  # Consistent with heuristic (0.1%)
    set_optimizer_attribute(model, "WorkLimit", 300.0)  # 300 work units limit
    
    # Decision Variables
    @variable(model, x[1:nl], Bin)  # Binary: line on/off
    @variable(model, p_g[1:nb, 1:nscen])  # Generation: p_g[i,s]
    @variable(model, f[1:nl, 1:nscen])  # Flow: f[l,s]
    @variable(model, theta[1:nb, 1:nscen])  # Phase angle: theta[i,s]
    @variable(model, unmet[1:nb, 1:nscen] >= 0)  # Unmet demand: unmet[i,s]
    
    # Fix lines ON based on capacity utilization file
    num_lines = haskey(problem_data, "num_lines_fixed") ? problem_data["num_lines_fixed"] : nothing
    fixed_lines = get_fixed_line_indices(problem_data; num_lines=num_lines)
    for l in fixed_lines
        if 1 <= l <= nl
            @constraint(model, x[l] == 1)
        else
            println("⚠️  Warning: Fixed line index $l is out of range (1-$nl)")
        end
    end
    
    # Constraints
    # 1. Generator bounds per bus and scenario (only for scenarios in subset A)
    for s in 1:nscen
        if subset_vector[s] == 1  # Only add constraint for scenarios in subset A
            for i in 1:nb
                @constraint(model, p_g[i, s] >= p_min[i])
                @constraint(model, p_g[i, s] <= p_max[i])
            end
        end
    end
    
    # 2. Line capacity + DC flow (only for scenarios in subset A)
    for s in 1:nscen
        if subset_vector[s] == 1  # Only add constraint for scenarios in subset A
            for (branch_id, branch) in L
                l = parse(Int, branch_id)
                i = branch["f_bus"]
                j = branch["t_bus"]
                @constraint(model, -fbar_ij[l] * x[l] <= f[l, s])
                @constraint(model,  f[l, s] <= fbar_ij[l] * x[l])
                # DC flow constraint with Big-M linearization
                @constraint(model, B_ij[l] * (theta[i, s] - theta[j, s]) - M_ij[l] * (1 - x[l]) <= f[l, s])
                @constraint(model, f[l, s] <= B_ij[l] * (theta[i, s] - theta[j, s]) + M_ij[l] * (1 - x[l]))
            end
        end
    end
    
    # 3. Power balance per bus and scenario (only for scenarios in subset A)
    for s in 1:nscen
        if subset_vector[s] == 1  # Only add constraint for scenarios in subset A
            d_s = scenarios[s]
            for i in 1:nb
                outgoing = 0.0
                incoming = 0.0
                for (branch_id, branch) in L
                    l = parse(Int, branch_id)
                    if branch["f_bus"] == i
                        outgoing += f[l, s]
                    end
                    if branch["t_bus"] == i
                        incoming += f[l, s]
                    end
                end
                @constraint(model, p_g[i, s] - d_s[i] + unmet[i, s] == outgoing - incoming)
            end
            @constraint(model, theta[ref, s] == 0)  # Reference angle
        end
    end
    
    # 4. Set variables to 0 for scenarios NOT in subset A
    for s in 1:nscen
        if subset_vector[s] == 0  # For scenarios not in subset A
            for i in 1:nb
                @constraint(model, p_g[i, s] == 0)
                @constraint(model, unmet[i, s] == 0)
            end
            for l in 1:nl
                @constraint(model, f[l, s] == 0)
            end
            for i in 1:nb
                @constraint(model, theta[i, s] == 0)
            end
        end
    end
    
    # Objective Function: min ∑_i c_i p_g[i,s] + α ∑_i unmet[i,s] for s∈A
    # Note: No subset_size multiplier needed (unlike Facility Location) since there's no first-stage cost
    @objective(model, Min,
        sum(c_g[i] * p_g[i, s] for i in 1:nb for s in 1:nscen if subset_vector[s] == 1) +
        sum(penalty * unmet[i, s] for i in 1:nb for s in 1:nscen if subset_vector[s] == 1)
    )
    
    # Solve
    optimize!(model)
    
    # Get work units
    work_units = 0.0
    try
        work_units = MOI.get(backend(model), Gurobi.ModelAttribute("Work"))
    catch
    end
    
    # Return best objective value found (even if not optimal)
    if termination_status(model) == MOI.OPTIMAL
        return objective_value(model), work_units
    else
        # Use best bound/objective found so far if available
        best_obj = Inf
        try
            if has_values(model)
                best_obj = objective_value(model)
            end
        catch
        end
        return best_obj, work_units
    end
end

# =============================================================================
# SUBSET COST EVALUATION f(A) - TEMPLATE MODEL (BATCH)
# =============================================================================

function build_subset_cost_template_TS(scenarios::Vector{Vector{Float64}}, problem_data::Dict)
    """Build a template model for batch subset cost evaluation (Transmission Switching)
    
    Returns:
        model, x, p_g, f, theta, unmet, power_balance_constraints, fixed_lines
    """
    B = problem_data["bus"]
    L = problem_data["branch"]
    nb = length(B)
    nl = length(L)
    B_ij, fbar_ij, p_max, p_min, A, c_g = parameters_TSpm(problem_data)
    ref = 1
    nscen = length(scenarios)
    
    # Use instance-specific penalty
    instance_file = get(problem_data, "instance_file", nothing)
    case_name = instance_file !== nothing ? extract_case_name(instance_file) : nothing
    penalty = get_penalty(instance_file=instance_file, case_name=case_name)
    
    # Big-M for DC equality relaxation when x_l = 0
    M_ij = [12.6 * abs(B_ij[l]) for l in 1:nl]
    
    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 0)
    set_optimizer_attribute(model, "MIPGap", 0.001)
    set_optimizer_attribute(model, "WorkLimit", 300.0)  # 300 work units limit
    
    # Decision Variables
    @variable(model, x[1:nl], Bin)
    @variable(model, p_g[1:nb, 1:nscen])
    @variable(model, f[1:nl, 1:nscen])
    @variable(model, theta[1:nb, 1:nscen])
    @variable(model, unmet[1:nb, 1:nscen] >= 0)
    
    # Fix lines ON based on capacity utilization file
    num_lines = haskey(problem_data, "num_lines_fixed") ? problem_data["num_lines_fixed"] : nothing
    fixed_lines = get_fixed_line_indices(problem_data; num_lines=num_lines)
    for l in fixed_lines
        if 1 <= l <= nl
            @constraint(model, x[l] == 1)
        else
            println("⚠️  Warning: Fixed line index $l is out of range (1-$nl)")
        end
    end
    
    # Power balance constraints (will update RHS based on subset membership)
    power_balance_constraints = Array{ConstraintRef}(undef, nscen, nb)
    for s in 1:nscen
        for i in 1:nb
            outgoing = 0.0
            incoming = 0.0
            for (branch_id, branch) in L
                l = parse(Int, branch_id)
                if branch["f_bus"] == i
                    outgoing += f[l, s]
                end
                if branch["t_bus"] == i
                    incoming += f[l, s]
                end
            end
            power_balance_constraints[s, i] = @constraint(model, p_g[i, s] - 0.0 + unmet[i, s] == outgoing - incoming)
        end
        @constraint(model, theta[ref, s] == 0)  # Reference angle
    end
    
    # Generator bounds (always active)
    for i in 1:nb, s in 1:nscen
        @constraint(model, p_g[i, s] >= p_min[i])
        @constraint(model, p_g[i, s] <= p_max[i])
    end
    
    # Line capacity + DC flow constraints (always active)
    for s in 1:nscen
        for (branch_id, branch) in L
            l = parse(Int, branch_id)
            i = branch["f_bus"]
            j = branch["t_bus"]
            @constraint(model, -fbar_ij[l] * x[l] <= f[l, s])
            @constraint(model,  f[l, s] <= fbar_ij[l] * x[l])
            @constraint(model, B_ij[l] * (theta[i, s] - theta[j, s]) - M_ij[l] * (1 - x[l]) <= f[l, s])
            @constraint(model, f[l, s] <= B_ij[l] * (theta[i, s] - theta[j, s]) + M_ij[l] * (1 - x[l]))
        end
    end
    
    # Objective (will update coefficients based on subset size)
    @objective(model, Min,
        sum(c_g[i] * p_g[i, s] for i in 1:nb for s in 1:nscen) +
        sum(penalty * unmet[i, s] for i in 1:nb for s in 1:nscen)
    )
    
    return model, x, p_g, f, theta, unmet, power_balance_constraints, fixed_lines, scenarios, penalty, c_g
end

function evaluate_subset_cost_batch_template_TS(subset_vectors::Vector{Vector{Int}},
                                                scenarios::Vector{Vector{Float64}},
                                                problem_data::Dict)
    """Evaluate costs for multiple subsets using a template model (batch evaluation)
    
    This is more efficient than calling evaluate_subset_cost_TS individually for each subset
    because it reuses the same model structure and only updates RHS values and bounds.
    
    Args:
        subset_vectors: List of subset vectors (each is a binary vector of length S)
        scenarios: List of S scenarios
        problem_data: Dict containing PowerModels data structure
    
    Returns:
        (costs::Vector{Float64}, total_work_units::Float64)
    """
    if isempty(subset_vectors)
        return Float64[], 0.0
    end
    
    nscen = length(scenarios)
    nb = length(problem_data["bus"])
    nl = length(problem_data["branch"])
    
    # Build template model once
    model, x, p_g, f, theta, unmet, power_balance_constraints, fixed_lines, scenarios_ref, penalty, c_g = 
        build_subset_cost_template_TS(scenarios, problem_data)
    
    costs = Float64[]
    total_work_units = 0.0
    
    for subset_vector in subset_vectors
        if sum(subset_vector) == 0
            push!(costs, 0.0)
            continue
        end
        
        # Update objective coefficients based on subset membership
        # Note: No subset_size multiplier needed (unlike Facility Location) since there's no first-stage cost
        # The objective has: sum(c_g[i] * p_g[i,s]) + sum(penalty * unmet[i,s]) for s in subset
        for i in 1:nb, s in 1:nscen
            if subset_vector[s] == 1
                set_objective_coefficient(model, p_g[i, s], c_g[i])
            else
                set_objective_coefficient(model, p_g[i, s], 0.0)
            end
        end
        
        # Update power balance RHS and bounds based on subset membership
        for s in 1:nscen
            if subset_vector[s] == 1
                # Scenario is in subset: update RHS to actual demand
                for i in 1:nb
                    set_normalized_rhs(power_balance_constraints[s, i], scenarios[s][i])
                end
            else
                # Scenario is NOT in subset: set RHS to 0 and add constraints to force variables to 0
                # We'll use bounds to force variables to 0 (more efficient than adding constraints)
                for i in 1:nb
                    set_normalized_rhs(power_balance_constraints[s, i], 0.0)
                    # Force p_g and unmet to 0 via bounds (override generator bounds temporarily)
                    set_upper_bound(p_g[i, s], 0.0)
                    set_lower_bound(p_g[i, s], 0.0)
                    set_upper_bound(unmet[i, s], 0.0)
                end
                for l in 1:nl
                    set_upper_bound(f[l, s], 0.0)
                    set_lower_bound(f[l, s], 0.0)
                end
                for i in 1:nb
                    set_upper_bound(theta[i, s], 0.0)
                    set_lower_bound(theta[i, s], 0.0)
                end
            end
        end
        
        optimize!(model)
        
        work_units = 0.0
        try
            work_units = MOI.get(backend(model), Gurobi.ModelAttribute("Work"))
        catch
        end
        total_work_units += work_units
        
        # Return best objective value found (even if not optimal)
        if termination_status(model) == MOI.OPTIMAL
            push!(costs, objective_value(model))
        else
            # Use best objective value found so far if available
            best_obj = Inf
            try
                if has_values(model)
                    best_obj = objective_value(model)
                end
            catch
            end
            push!(costs, best_obj)
        end
        
        # Reset bounds for next iteration (restore original bounds)
        # Note: Generator bounds are constraints in the model, so removing variable bounds
        # will allow the constraint bounds to take effect again
        for s in 1:nscen
            if subset_vector[s] == 0
                for i in 1:nb
                    # Remove the 0 bounds we set - generator constraint bounds will be active again
                    if has_upper_bound(p_g[i, s]) && upper_bound(p_g[i, s]) == 0.0
                        delete_upper_bound(p_g[i, s])
                    end
                    if has_lower_bound(p_g[i, s]) && lower_bound(p_g[i, s]) == 0.0
                        delete_lower_bound(p_g[i, s])
                    end
                    if has_upper_bound(unmet[i, s])
                        delete_upper_bound(unmet[i, s])
                    end
                end
                # Restore flow bounds (capacity constraints in model will be active again)
                for l in 1:nl
                    if has_upper_bound(f[l, s]) && upper_bound(f[l, s]) == 0.0
                        delete_upper_bound(f[l, s])
                    end
                    if has_lower_bound(f[l, s]) && lower_bound(f[l, s]) == 0.0
                        delete_lower_bound(f[l, s])
                    end
                end
                # Restore theta bounds (unbounded by default)
                for i in 1:nb
                    if has_upper_bound(theta[i, s])
                        delete_upper_bound(theta[i, s])
                    end
                    if has_lower_bound(theta[i, s])
                        delete_lower_bound(theta[i, s])
                    end
                end
            end
        end
    end
    
    return costs, total_work_units
end

# =============================================================================
# PRICING SUBPROBLEM (Type 1) - SF
# =============================================================================

function sf_pricing_type1_TS(data, scenarios, dual_lambda, dual_mu; use_solution_pool::Bool=true, return_nodes::Bool=false, work_limit::Float64=Inf, opt_worklimit::Float64=DEFAULT_OPT_WORKLIMIT, log_file=nothing, iteration=nothing)
    """Solve SF Pricing Type 1 - Generate new subset of scenarios (solve to optimality with work limit)
    
    Returns:
        (solutions, objectives, work_units) or (solutions, objectives, work_units, nodes) if return_nodes=true
    """
    B_ij, fbar_ij, p_max, p_min, A, c = parameters_TSpm(data)
    L = data["branch"]
    nb = length(data["bus"])
    nl = length(L)
    nscen = length(scenarios)
    
    # Use instance-specific penalty (consistent with heuristic)
    instance_file = get(data, "instance_file", nothing)
    case_name = instance_file !== nothing ? extract_case_name(instance_file) : nothing
    alpha = get_penalty(instance_file=instance_file, case_name=case_name)
    
    # Big-M for DC equality relaxation when x_l = 0
    M_ij = [12.6 * abs(B_ij[l]) for l in 1:nl]
    
    # Create model
    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 1)
    if use_solution_pool
        set_optimizer_attribute(model, "PoolSearchMode", 2)
        set_optimizer_attribute(model, "PoolSolutions", 10)
    end
    
    # Set up log file with iteration marker
    if log_file !== nothing
        if iteration !== nothing
            open(log_file, "a") do file
                println(file, "\n" * "="^80)
                println(file, "PRICING SUBPROBLEM TYPE 1 (SF) - ITERATION $iteration")
                println(file, "="^80)
                println(file, "Timestamp: $(now())")
                println(file, "Problem size: $(length(scenarios)) scenarios, $nb buses, $nl lines")
                println(file, "="^80)
            end
        end
        set_optimizer_attribute(model, "OutputFlag", 1)  # Ensure output is enabled for log file
        set_optimizer_attribute(model, "LogFile", log_file)
        set_optimizer_attribute(model, "LogToConsole", 0)
    end
    
    # Decision Variables
    @variable(model, x[1:nl], Bin)  # Binary: line on/off
    @variable(model, p_g[1:nb, 1:nscen])  # Generation: p_g[i,s]
    @variable(model, r[1:nb, 1:nscen] >= 0)  # Unmet demand: r[i,s]
    @variable(model, f[1:nl, 1:nscen])  # Flow: f[l,s]
    @variable(model, theta[1:nb, 1:nscen])  # Phase angle: theta[i,s]
    @variable(model, u[1:nb, 1:nscen] >= 0)  # Auxiliary generation: u[i,s] = p_g[i,s] * pi[s]
    @variable(model, w[1:nb, 1:nscen] >= 0)  # Auxiliary unmet: w[i,s] = r[i,s] * pi[s]
    @variable(model, pi[1:nscen], Bin)  # Binary: scenario selection π^s
    
    # Fix lines ON based on capacity utilization file
    fixed_lines = get_fixed_line_indices(data)
    for l in fixed_lines
        if 1 <= l <= nl
            @constraint(model, x[l] == 1)
        else
            println("⚠️  Warning: Fixed line index $l is out of range (1-$nl)")
        end
    end
    
    # Objective Function: min (1/|S|) * (∑_i ∑_s c_i u_i^s + ∑_i ∑_s α w_i^s) - ∑_s (λ^s)*π^s - μ*
    @objective(model, Min, 
        1.0/nscen * (sum(c[i]*u[i,s] for i in 1:nb for s in 1:nscen) + 
        sum(alpha*w[i,s] for i in 1:nb for s in 1:nscen)) - 
        sum(dual_lambda[s]*pi[s] for s in 1:nscen) - 
        dual_mu)
    
    # Constraints
    # 1. Demand satisfaction: sum_l (outgoing - incoming) + r[i,s] - d[i,s] <= d[i,s] * (1 - pi[s])
    for s in 1:nscen
        d_s = scenarios[s]
        for i in 1:nb
            outgoing = 0.0
            incoming = 0.0
            for (branch_id, branch) in L
                l = parse(Int, branch_id)
                if branch["f_bus"] == i
                    outgoing += f[l, s]
                end
                if branch["t_bus"] == i
                    incoming += f[l, s]
                end
            end
            @constraint(model, p_g[i, s] + r[i, s] - d_s[i] <= d_s[i] * (1 - pi[s]))
        end
    end
    
    # 2. Demand satisfaction (lower bound): sum_l (outgoing - incoming) + r[i,s] - d[i,s] >= -d[i,s] * (1 - pi[s])
    for s in 1:nscen
        d_s = scenarios[s]
        for i in 1:nb
            outgoing = 0.0
            incoming = 0.0
            for (branch_id, branch) in L
                l = parse(Int, branch_id)
                if branch["f_bus"] == i
                    outgoing += f[l, s]
                end
                if branch["t_bus"] == i
                    incoming += f[l, s]
                end
            end
            @constraint(model, p_g[i, s] + r[i, s] - d_s[i] >= -d_s[i] * (1 - pi[s]))
        end
    end
    
    # 3. Capacity constraint: -fbar[l] * x[l] <= f[l,s] <= fbar[l] * x[l] with DC flow and M_ij
    for s in 1:nscen
        for (branch_id, branch) in L
            l = parse(Int, branch_id)
            i = branch["f_bus"]
            j = branch["t_bus"]
            @constraint(model, -fbar_ij[l] * x[l] <= f[l, s])
            @constraint(model,  f[l, s] <= fbar_ij[l] * x[l])
            # DC flow constraint with M_ij
            @constraint(model, B_ij[l] * (theta[i, s] - theta[j, s]) - M_ij[l] * (1 - x[l]) <= f[l, s])
            @constraint(model, f[l, s] <= B_ij[l] * (theta[i, s] - theta[j, s]) + M_ij[l] * (1 - x[l]))
        end
    end
    
    # 4. u[i,s] <= p_g[i,s]
    @constraint(model, [i in 1:nb, s in 1:nscen], u[i, s] <= p_g[i, s])
    
    # 5. u[i,s] <= p_max[i] * pi[s]
    @constraint(model, [i in 1:nb, s in 1:nscen], u[i, s] <= p_max[i] * pi[s])
    
    # 6. u[i,s] >= p_g[i,s] + p_max[i] * (pi[s] - 1)
    @constraint(model, [i in 1:nb, s in 1:nscen], u[i, s] >= p_g[i, s] + p_max[i] * (pi[s] - 1))
    
    # 7. w[i,s] <= r[i,s]
    @constraint(model, [i in 1:nb, s in 1:nscen], w[i, s] <= r[i, s])
    
    # 8. w[i,s] <= d[i,s] * pi[s]
    @constraint(model, [i in 1:nb, s in 1:nscen], w[i, s] <= scenarios[s][i] * pi[s])
    
    # 9. w[i,s] >= r[i,s] + d[i,s] * (pi[s] - 1)
    @constraint(model, [i in 1:nb, s in 1:nscen], w[i, s] >= r[i, s] + scenarios[s][i] * (pi[s] - 1))
    
    # 10. Generator bounds
    for i in 1:nb, s in 1:nscen
        @constraint(model, p_g[i, s] >= p_min[i])
        @constraint(model, p_g[i, s] <= p_max[i])
    end
    
    # 11. Reference angle
    for s in 1:nscen
        @constraint(model, theta[1, s] == 0)
    end
    
    # Solve to optimality with given work limit
    pricing_work_units = 0.0
    
    if isfinite(work_limit) && work_limit > 0.0
        set_optimizer_attribute(model, "MIPFocus", 0)  # Default optimality focus
        set_optimizer_attribute(model, "WorkLimit", work_limit)
    end
    
    optimize!(model)
    
    try
        pricing_work_units = MOI.get(backend(model), Gurobi.ModelAttribute("Work"))
    catch
    end
    
    # Check termination status and collect solutions
    if termination_status(model) == MOI.OPTIMAL
        kept_solutions, kept_objs = collect_pool_rc_cap10(model, pi, dual_mu, use_solution_pool, "SF")
        return return_pricing_payload(kept_solutions, kept_objs, pricing_work_units, return_nodes, model)
    else
        kept_solutions, kept_objs = collect_pool_rc_cap10(model, pi, dual_mu, use_solution_pool, "SF")
        return return_pricing_payload(kept_solutions, kept_objs, pricing_work_units, return_nodes, model)
    end
end

# =============================================================================
# PRICING SUBPROBLEM (Type 2) - SF
# =============================================================================

function sf_pricing_type2_TS(data, scenarios, dual_lambda, dual_mu; use_solution_pool::Bool=true, return_nodes::Bool=false, work_limit::Float64=Inf, opt_worklimit::Float64=DEFAULT_OPT_WORKLIMIT, analysis::Dict{Symbol,Any}=Dict{Symbol,Any}(), log_file=nothing, iteration=nothing)
    """Solve SF Pricing Type 2 - Alternative formulation (two-phase with solution pool)
    
    Returns:
        (solutions, objectives, work_units) or (solutions, objectives, work_units, nodes) if return_nodes=true
    """
    B_ij, fbar_ij, p_max, p_min, A, c = parameters_TSpm(data)
    L = data["branch"]
    nb = length(data["bus"])
    nl = length(L)
    nscen = length(scenarios)
    
    # Use instance-specific penalty (consistent with heuristic)
    instance_file = get(data, "instance_file", nothing)
    case_name = instance_file !== nothing ? extract_case_name(instance_file) : nothing
    alpha = get_penalty(instance_file=instance_file, case_name=case_name)
    
    # Big-M for DC equality relaxation when x_l = 0
    M_ij = [12.6 * abs(B_ij[l]) for l in 1:nl]
    
    # Create model
    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 1)
    if use_solution_pool
        set_optimizer_attribute(model, "PoolSearchMode", 2)
        set_optimizer_attribute(model, "PoolSolutions", 10)
    end
    
    # Set up log file with iteration marker
    if log_file !== nothing
        if iteration !== nothing
            open(log_file, "a") do file
                println(file, "\n" * "="^80)
                println(file, "PRICING SUBPROBLEM TYPE 2 (SF) - ITERATION $iteration")
                println(file, "="^80)
                println(file, "Timestamp: $(now())")
                println(file, "Problem size: $(length(scenarios)) scenarios, $nb buses, $nl lines")
                println(file, "="^80)
            end
        end
        set_optimizer_attribute(model, "OutputFlag", 1)  # Ensure output is enabled for log file
        set_optimizer_attribute(model, "LogFile", log_file)
        set_optimizer_attribute(model, "LogToConsole", 1)
    end
    
    # Decision Variables
    @variable(model, x[1:nl], Bin)  # Binary: line on/off (first-stage)
    @variable(model, x_s[1:nl, 1:nscen], Bin)  # Binary copies: line on/off for scenario s
    @variable(model, p_g[1:nb, 1:nscen])  # Generation: p_g[i,s]
    @variable(model, r[1:nb, 1:nscen] >= 0)  # Unmet demand: r[i,s]
    @variable(model, f[1:nl, 1:nscen])  # Flow: f[l,s]
    @variable(model, theta[1:nb, 1:nscen])  # Phase angle: theta[i,s]
    @variable(model, pi[1:nscen], Bin)  # Binary: scenario selection π^s
    
    # Fix lines ON based on capacity utilization file
    fixed_lines = get_fixed_line_indices(data)
    for l in fixed_lines
        if 1 <= l <= nl
            @constraint(model, x[l] == 1)
            @constraint(model, [s in 1:nscen], x_s[l, s] == pi[s])
        else
            println("⚠️  Warning: Fixed line index $l is out of range (1-$nl)")
        end
    end
    
    # Objective Function: min ∑_s ( (∑_i c_i p_g[i,s]) + (∑_i α r[i,s]) - (λ^s)*π^s ) - μ*
    @objective(model, Min, 
        sum(c[i]*p_g[i,s] for i in 1:nb for s in 1:nscen) + 
        sum(alpha*r[i,s] for i in 1:nb for s in 1:nscen) - 
        sum(dual_lambda[s]*pi[s] for s in 1:nscen) - 
        dual_mu)
    
    # Constraints
    # 1. Demand satisfaction: p_g[i,s] + r[i,s] == d[i,s] * pi[s]
    for s in 1:nscen
        d_s = scenarios[s]
        for i in 1:nb
            outgoing = 0.0
            incoming = 0.0
            for (branch_id, branch) in L
                l = parse(Int, branch_id)
                if branch["f_bus"] == i
                    outgoing += f[l, s]
                end
                if branch["t_bus"] == i
                    incoming += f[l, s]
                end
            end
            @constraint(model, p_g[i, s] + r[i, s] - d_s[i]*pi[s] == outgoing - incoming)
        end
        @constraint(model, theta[1, s] == 0)  # Reference angle
    end
    
    # 2. Capacity constraint: -fbar[l] * x_s[l,s] <= f[l,s] <= fbar[l] * x_s[l,s]
    for s in 1:nscen
        for (branch_id, branch) in L
            l = parse(Int, branch_id)
            i = branch["f_bus"]
            j = branch["t_bus"]
            @constraint(model, -fbar_ij[l] * x_s[l, s] <= f[l, s])
            @constraint(model,  f[l, s] <= fbar_ij[l] * x_s[l, s])
            # DC flow constraint with M_ij
            @constraint(model, B_ij[l] * (theta[i, s] - theta[j, s]) - M_ij[l] * (1 - x_s[l, s]) <= f[l, s])
            @constraint(model, f[l, s] <= B_ij[l] * (theta[i, s] - theta[j, s]) + M_ij[l] * (1 - x_s[l, s]))
        end
    end
    
    # 3. x_s[l,s] <= pi[s]
    @constraint(model, [l in 1:nl, s in 1:nscen], x_s[l, s] <= pi[s])
    
    # 4. x[l] - x_s[l,s] <= (1 - pi[s])
    @constraint(model, [l in 1:nl, s in 1:nscen], x[l] - x_s[l, s] <= (1 - pi[s]))
    
    # 5. x[l] - x_s[l,s] >= -(1 - pi[s])
    @constraint(model, [l in 1:nl, s in 1:nscen], x[l] - x_s[l, s] >= -(1 - pi[s]))
    
    # 6. Generator bounds
    for i in 1:nb, s in 1:nscen
        @constraint(model, p_g[i, s] >= p_min[i]*pi[s])
        @constraint(model, p_g[i, s] <= p_max[i]*pi[s])
    end
    
    # Two-phase solve structure with analysis tracking
    pricing_work_units = 0.0
    if !isempty(analysis)
        analysis[:phase1_work] = 0.0
        analysis[:phase2_work] = 0.0
        analysis[:went_phase2] = false
        analysis[:lb] = NaN
        analysis[:ub] = NaN
    end

    # Phase 1: Quick optimal try
    if isfinite(opt_worklimit) && opt_worklimit > 0.0
        set_optimizer_attribute(model, "MIPFocus", 1)
        set_optimizer_attribute(model, "WorkLimit", opt_worklimit)
        optimize!(model)
        
        try
            w = MOI.get(backend(model), Gurobi.ModelAttribute("Work"))
            pricing_work_units += w
            if !isempty(analysis); analysis[:phase1_work] = w; end
        catch
        end
        
        # Check for RC-improving solutions
        if termination_status(model) == MOI.OPTIMAL
            kept_solutions, kept_objs = collect_pool_rc_cap10(model, pi, dual_mu, use_solution_pool, "SF")
            if !isempty(kept_solutions)
                if !isempty(analysis); analysis[:went_phase2] = false; end
                return return_pricing_payload(kept_solutions, kept_objs, pricing_work_units, return_nodes, model)
            else
                # No RC-improving solutions, proceed to Phase 2
            end
        else
            kept_solutions, kept_objs = collect_pool_rc_cap10(model, pi, dual_mu, use_solution_pool, "SF")
            if !isempty(kept_solutions)
                if !isempty(analysis); analysis[:went_phase2] = false; end
                return return_pricing_payload(kept_solutions, kept_objs, pricing_work_units, return_nodes, model)
            end
        end
    end

    # Phase 2: Optimality push
    heur_worklimit = work_limit - opt_worklimit
    if isfinite(heur_worklimit) && heur_worklimit > 0.0
        if !isempty(analysis); analysis[:went_phase2] = true; end
        set_optimizer_attribute(model, "MIPFocus", 0)
        set_optimizer_attribute(model, "WorkLimit", heur_worklimit)
        optimize!(model)
        
        try
            w2 = MOI.get(backend(model), Gurobi.ModelAttribute("Work"))
            pricing_work_units += w2
            if !isempty(analysis); analysis[:phase2_work] = w2; end
        catch
        end
        
        kept_solutions, kept_objs = collect_pool_rc_cap10(model, pi, dual_mu, use_solution_pool, "SF")
        return return_pricing_payload(kept_solutions, kept_objs, pricing_work_units, return_nodes, model)
    end
    
    # Fallback if no work limits set
    return return_pricing_payload(Vector{Vector{Float64}}(), Float64[], pricing_work_units, return_nodes, model)
end

# =============================================================================
# PRICING SUBPROBLEM WRAPPER (Interface for ColumnGenerationCore)
# =============================================================================

function build_pricing_model_SF_TS(scenarios::Vector{Vector{Float64}}, dual_lambda::Vector{Float64}, dual_mu::Float64, problem_data::Dict, pricing_type::Int, work_limit::Float64, opt_worklimit::Float64, use_solution_pool::Bool, return_nodes::Bool, log_file::Union{String, Nothing}, iteration::Union{Int, Nothing}=nothing)
    """Build and solve pricing subproblem for Transmission Switching - Subset Formulation
    
    This is the interface function called by ColumnGenerationCore.jl
    
    Returns:
        (solutions, objectives, work_units) or (solutions, objectives, work_units, nodes) if return_nodes=true
    """
    if pricing_type == 1
        return sf_pricing_type1_TS(problem_data, scenarios, dual_lambda, dual_mu; 
                                   use_solution_pool=use_solution_pool, return_nodes=return_nodes, work_limit=work_limit, opt_worklimit=opt_worklimit, log_file=log_file, iteration=iteration)
    elseif pricing_type == 2
        analysis = Dict{Symbol,Any}()
        return sf_pricing_type2_TS(problem_data, scenarios, dual_lambda, dual_mu; 
                                   use_solution_pool=use_solution_pool, return_nodes=return_nodes, work_limit=work_limit, opt_worklimit=opt_worklimit, analysis=analysis, log_file=log_file, iteration=iteration)
    else
        error("Invalid pricing_type. Use 1 or 2.")
    end
end
