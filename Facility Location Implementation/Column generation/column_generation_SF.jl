# Column Generation Implementation - SF Formulation
# New formulation with different master problem and pricing subproblem

include("column_generation_common.jl")

# Additional parameters for column generation
unmet_pen = 50.0
scaling_factor = 1  # Scaling factor as requested

println(" Problem Parameters:")
println("  Unmet penalty: $unmet_pen")
println("  Scaling factor: $scaling_factor")

# Function to calculate associated cost for a subset of scenarios
function associated_cost_function(A, scenarios, nloc, ncust, capacity, cost, fixedcost, unmet_pen, scaling_factor)
    """
    Parameter A description:
    - A: 0-1 vector of length nscen, where A[s] = 1 if scenario s is included, 0 otherwise
    """
    nscen = length(scenarios)
    
    # Check if A is valid
    if length(A) != nscen
        error("A must have length equal to number of scenarios ($nscen)")
    end
    
    # Check if subset is empty
    if sum(A) == 0
        return 0.0  # Cost of empty subset is 0
    end
    
    # Create model
    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 0)  # Suppress output
    set_optimizer_attribute(model, "TimeLimit", 60)  # 1 minute time limit
    
    # Decision Variables
    @variable(model, x[1:nloc], Bin)  # Binary: facility open/closed
    @variable(model, y[1:nloc, 1:ncust, 1:nscen] >= 0)  # Flow: y[i,j,s]
    @variable(model, b[1:ncust, 1:nscen] >= 0)  # Unmet demand: b[j,s]
    
    # Constraints
    # 1. Demand satisfaction constraint: sum_i y[i,j,s] + b[j,s] = d[j,s] for s∈A, j∈J
    for s in 1:nscen
        if A[s] == 1  # Only add constraint for scenarios in subset A
            for j in 1:ncust
                @constraint(model, sum(y[i,j,s] for i in 1:nloc) + b[j,s] == scenarios[s][j])
            end
        end
    end
    
    # 2. Capacity constraint: sum_j y[i,j,s] - capacity[i]*x[i] <= 0 for i∈I, s∈A
    for s in 1:nscen
        if A[s] == 1  # Only add constraint for scenarios in subset A
            for i in 1:nloc
                @constraint(model, sum(y[i,j,s] for j in 1:ncust) - capacity[i]*x[i] <= 0)
            end
        end
    end
    
    # 3. Set y[i,j,s] = 0 and b[j,s] = 0 for scenarios NOT in subset A
    for s in 1:nscen
        if A[s] == 0  # For scenarios not in subset A
            for i in 1:nloc, j in 1:ncust
                @constraint(model, y[i,j,s] == 0)
            end
            for j in 1:ncust
                @constraint(model, b[j,s] == 0)
            end
        end
    end
    
    # Objective Function with scaling by |A|/|S| and 1/|S|
    # fixed costs term weighted by |A|/|S|; flow and penalty terms weighted by 1/|S|
    subset_size = sum(A)
    weight_fixed = subset_size / nscen
    weight_rest = 1.0 / nscen
    @objective(model, Min,
        weight_fixed * sum(fixedcost[i] * x[i] for i in 1:nloc) +
        weight_rest * (
            scaling_factor * sum(cost[i,j] * y[i,j,s] for s in 1:nscen for i in 1:nloc for j in 1:ncust if A[s] == 1) +
            sum(unmet_pen * b[j,s] for s in 1:nscen for j in 1:ncust if A[s] == 1)
        )
    )
    
    # Solve
    optimize!(model)
    
    if termination_status(model) == MOI.OPTIMAL
        return objective_value(model)
    else
        return Inf  # Return infinite cost if infeasible
    end
end

# Function to solve SF Master Problem (Partition Formulation)
function solve_sf_master_problem(cost_vector, z_matrix, K)
    """
    Solve the SF Master Problem (Partition Formulation)
    
    Parameters:
    - cost_vector: f(A) values for each subset A
    - z_matrix: |S| × |A| matrix where z_matrix[s, A] = 1 if scenario s is in subset A
    - K: Cardinality constraint limit
    
    Returns:
    - model, v_variables, objective_value, partition_constraints, k_constraint
    """
    nscen, nsubsets = size(z_matrix)
    
    # Create model
    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 0)
    
    # Decision Variables - Dynamic array of v_A variables
    v_variables = [@variable(model, lower_bound = 0, upper_bound = 1) for A in 1:nsubsets]
    
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
    
    if termination_status(model) == MOI.OPTIMAL
        return model, v_variables, objective_value(model), partition_constraints, k_constraint
    else
        return model, v_variables, Inf, partition_constraints, k_constraint
    end
end

# Function to solve SF Pricing Type 1
function sf_pricing_type1(nloc, ncust, capacity, cost, scenarios, dual_lambda, dual_mu, unmet_pen, scaling_factor)
    """
    Solve SF Pricing Type 1 - Generate new subset of scenarios
    
    Parameters:
    - nloc, ncust: number of locations and customers
    - capacity, cost: facility location data
    - scenarios: list of all scenarios
    - dual_lambda: dual values from partition constraints (vector of length nscen)
    - dual_mu: dual value from cardinality constraint (scalar)
    - unmet_pen, scaling_factor: penalty and scaling parameters
    
    Returns:
    - pi_optimal: binary scenario selection vector (π^s values)
    - objective_value: optimal objective value
    """
    nscen = length(scenarios)
    
    # Create model
    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 1)  # Show Gurobi output
    set_optimizer_attribute(model, "TimeLimit", 300)
    
    # Decision Variables
    @variable(model, x[1:nloc], Bin)  # Binary: facility open/closed
    @variable(model, y[1:nloc, 1:ncust, 1:nscen] >= 0)  # Flow: y[i,j,s]
    @variable(model, b[1:ncust, 1:nscen] >= 0)  # Unmet demand: b[j,s]
    @variable(model, v[1:nloc, 1:nscen] >= 0)  # Auxiliary variable: v[i,s] = x[i] * pi[s]
    @variable(model, u[1:nloc, 1:ncust, 1:nscen] >= 0)  # Auxiliary flow: u[i,j,s] = y[i,j,s] * pi[s]
    @variable(model, w[1:ncust, 1:nscen] >= 0)  # Auxiliary unmet demand: w[j,s] = b[j,s] * pi[s]
    @variable(model, pi[1:nscen], Bin)  # Binary: scenario selection π^s
    
    # Objective Function: min ∑_i f_i v_i^s + ∑_s ∑_i ∑_j c_ij u_ij^s + ∑_s ∑_j α w_j^s - ∑_s (λ^s)*π^s - μ*
    @objective(model, Min, 
        1.0/nscen * (sum(fixedcost[i]*v[i,s] for i in 1:nloc for s in 1:nscen) + 
        scaling_factor * sum(cost[i,j]*u[i,j,s] for i in 1:nloc for j in 1:ncust for s in 1:nscen) + 
        sum(unmet_pen*w[j,s] for j in 1:ncust for s in 1:nscen)) - 
        sum(dual_lambda[s]*pi[s] for s in 1:nscen) - 
        dual_mu)
    
    # Constraints
    # Constraint 1: ∑_i y_ij^s + b_j^s - d_j^s ≤ d_j^s (1 - π^s)
    @constraint(model, [j in 1:ncust, s in 1:nscen], 
        sum(y[i,j,s] for i in 1:nloc) + b[j,s] - scenarios[s][j] <= scenarios[s][j] * (1 - pi[s]))
    
    # Constraint 2: v_i^s ≤ x_i
    @constraint(model, [i in 1:nloc, s in 1:nscen], 
        v[i,s] <= x[i])
    
    # Constraint 3: v_i^s ≤ π^s
    @constraint(model, [i in 1:nloc, s in 1:nscen], 
        v[i,s] <= pi[s])
    
    # Constraint 4: v_i^s ≥ x_i + π^s - 1
    @constraint(model, [i in 1:nloc, s in 1:nscen], 
        v[i,s] >= x[i] + pi[s] - 1)
    
    # Constraint 5: ∑_i y_ij^s + b_j^s - d_j^s ≥ -d_j^s (1 - π^s)
    @constraint(model, [j in 1:ncust, s in 1:nscen], 
        sum(y[i,j,s] for i in 1:nloc) + b[j,s] - scenarios[s][j] >= -scenarios[s][j] * (1 - pi[s]))
    
    # Constraint 6: ∑_j y_ij^s - ū_i x_i ≤ ū_i (1 - π^s)
    @constraint(model, [i in 1:nloc, s in 1:nscen], 
        sum(y[i,j,s] for j in 1:ncust) - capacity[i]*x[i] <= capacity[i] * (1 - pi[s]))
    
    # Constraint 7: u_ij^s ≤ y_ij^s
    @constraint(model, [i in 1:nloc, j in 1:ncust, s in 1:nscen], 
        u[i,j,s] <= y[i,j,s])
    
    # Constraint 8: u_ij^s ≤ d_j^s π^s
    @constraint(model, [i in 1:nloc, j in 1:ncust, s in 1:nscen], 
        u[i,j,s] <= scenarios[s][j] * pi[s])
    
    # Constraint 9: u_ij^s ≥ y_ij^s + d_j^s (π^s - 1)
    @constraint(model, [i in 1:nloc, j in 1:ncust, s in 1:nscen], 
        u[i,j,s] >= y[i,j,s] + scenarios[s][j] * (pi[s] - 1))
    
    # Constraint 10: w_j^s ≤ b_j^s
    @constraint(model, [j in 1:ncust, s in 1:nscen], 
        w[j,s] <= b[j,s])
    
    # Constraint 11: w_j^s ≤ d_j^s π^s
    @constraint(model, [j in 1:ncust, s in 1:nscen], 
        w[j,s] <= scenarios[s][j] * pi[s])
    
    # Constraint 12: w_j^s ≥ b_j^s + d_j^s (π^s - 1)
    @constraint(model, [j in 1:ncust, s in 1:nscen], 
        w[j,s] >= b[j,s] + scenarios[s][j] * (pi[s] - 1))
    
    # Solve
    optimize!(model)
    
    if termination_status(model) == MOI.OPTIMAL
        return value.(pi), objective_value(model)
    else
        return zeros(nscen), Inf
    end
end

# Function to solve SF Pricing Type 2
function sf_pricing_type2(nloc, ncust, capacity, cost, scenarios, dual_lambda, dual_mu, unmet_pen, scaling_factor)
    """
    Solve SF Pricing Type 2 - Alternative formulation for generating new subset of scenarios
    
    Parameters:
    - nloc, ncust: number of locations and customers
    - capacity, cost: facility location data
    - scenarios: list of all scenarios
    - dual_lambda: dual values from partition constraints (vector of length nscen)
    - dual_mu: dual value from cardinality constraint (scalar)
    - unmet_pen, scaling_factor: penalty and scaling parameters
    
    Returns:
    - pi_optimal: binary scenario selection vector (π^s values)
    - objective_value: optimal objective value
    """
    nscen = length(scenarios)
    
    # Create model
    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 0)
    set_optimizer_attribute(model, "TimeLimit", 60)
    
    # Decision Variables
    @variable(model, x[1:nloc], Bin)  # Binary: facility open/closed (first-stage)
    @variable(model, x_s[1:nloc, 1:nscen], Bin)  # Binary copies: facility open/closed for scenario s
    @variable(model, y[1:nloc, 1:ncust, 1:nscen] >= 0)  # Flow: y[i,j,s]
    @variable(model, b[1:ncust, 1:nscen] >= 0)  # Unmet demand: b[j,s]
    @variable(model, pi[1:nscen], Bin)  # Binary: scenario selection π^s
    
    # Vector of ones (e)
    e = ones(nloc)
    
    # Objective Function: min ∑_s ( (∑_i f_i x_i^s) + (∑_i ∑_j c_ij y_ij^s) + (∑_j α b_j^s) - (λ^s)*π^s ) - μ*
    @objective(model, Min, 
        1/nscen * (sum(fixedcost[i]*x_s[i,s] for i in 1:nloc for s in 1:nscen) + 
            scaling_factor * sum(cost[i,j]*y[i,j,s] for i in 1:nloc for j in 1:ncust for s in 1:nscen) + 
            sum(unmet_pen*b[j,s] for j in 1:ncust for s in 1:nscen))- 
            sum(dual_lambda[s]*pi[s] for s in 1:nscen) - 
        dual_mu)
    
    # Constraints
    # Constraint 1: ∑_i y_ij^s + b_j^s = d_j^s for all s∈S, j∈J
    @constraint(model, [j in 1:ncust, s in 1:nscen], 
        sum(y[i,j,s] for i in 1:nloc) + b[j,s] == scenarios[s][j]*pi[s])
    
    # Constraint 2: ∑_j y_ij^s - u_i x_i^s ≤ 0 for all i∈I, s∈S
    @constraint(model, [i in 1:nloc, s in 1:nscen], 
        sum(y[i,j,s] for j in 1:ncust) - capacity[i]*x_s[i,s] <= 0)
    
    # Constraint 3: x^s ≤ e π^s for all s∈S
    @constraint(model, [i in 1:nloc, s in 1:nscen], 
        x_s[i,s] <= pi[s])
    
    # Constraint 4: x - x^s ≤ e(1 - π^s) for all s∈S
    @constraint(model, [i in 1:nloc, s in 1:nscen], 
        x[i] - x_s[i,s] <= (1 - pi[s]))
    
    # Constraint 5: x - x^s ≥ -e(1 - π^s) for all s∈S
    @constraint(model, [i in 1:nloc, s in 1:nscen], 
        x[i] - x_s[i,s] >= -(1 - pi[s]))
    
    # Solve
    optimize!(model)
    
    if termination_status(model) == MOI.OPTIMAL
        return value.(pi), objective_value(model)
    else
        return zeros(nscen), Inf
    end
end

# Function to solve SF pricing subproblem with choice of formulation type
function solve_sf_pricing_subproblem(nloc, ncust, capacity, cost, scenarios, dual_lambda, dual_mu, unmet_pen, scaling_factor, pricing_type=1)
    """
    Solve the SF pricing subproblem with choice of formulation type
    """
    if pricing_type == 1
        # Solve SF Pricing Type 1
        println("  Solving SF Pricing Type 1...")
        pi_optimal, obj_value = sf_pricing_type1(nloc, ncust, capacity, cost, scenarios, dual_lambda, dual_mu, unmet_pen, scaling_factor)
        return pi_optimal, obj_value
        
    elseif pricing_type == 2
        # Solve SF Pricing Type 2
        println("  Solving SF Pricing Type 2...")
        pi_optimal, obj_value = sf_pricing_type2(nloc, ncust, capacity, cost, scenarios, dual_lambda, dual_mu, unmet_pen, scaling_factor)
        return pi_optimal, obj_value
        
    else
        error("Invalid pricing_type. Use 1 for sf_pricing_type1 or 2 for sf_pricing_type2.")
    end
end

# Main SF Column Generation Algorithm
function sf_column_generation_algorithm(scenarios, nloc, ncust, capacity, cost, fixedcost, unmet_pen, scaling_factor, K, max_iterations, pricing_type=1, init_type=1)
    """Main SF Column Generation Algorithm with 5 stages"""
    nscen = length(scenarios)
    
    println(" Starting SF Column Generation Algorithm")
    println("Parameters: nscen=$nscen, nloc=$nloc, ncust=$ncust, K=$K, max_iter=$max_iterations")
    println()
    
    # Initialize pricing + dual_mu tracking
    iteration_pricing_data = []
    actual_iterations = 0  # Track actual number of iterations completed
    
    # Stage 1: Initialization
    println(" Stage 1: Initialization")
    
    if init_type == 1
        # Option 1: Create random partition of scenarios into K subsets
        println("  Initialization Type 1: Creating random partition of scenarios into K subsets...")
        
        # Set seed for reproducibility
        Random.seed!(42)
        
        # Initialize K empty column vectors (all zeros)
        z_matrix = zeros(nscen, K)
        
        # Randomly assign each scenario to one of the K subsets
        for s in 1:nscen
            # Randomly choose which subset (1 to K) to assign scenario s to
            subset_idx = rand(1:K)
            z_matrix[s, subset_idx] = 1
        end
        
        # Calculate associated cost for each of the K subsets
        cost_vector = []
        for k in 1:K
            # Extract subset k as a vector
            A_k = z_matrix[:, k]
            
            # Calculate associated cost
            cost_k = associated_cost_function(A_k, scenarios, nloc, ncust, capacity, cost, fixedcost, unmet_pen, scaling_factor)
            push!(cost_vector, cost_k)
        end
        
        nsubsets = length(cost_vector)
        println("  Generated $nsubsets initial subsets (random partition)")
        
    else
        # Option 2: Start with single subset (first scenario only)
        println("  Initialization Type 2: Starting with single subset (first scenario)...")
        A_1 = zeros(nscen)
        A_1[1] = 1
        
        # Calculate associated cost
        cost_1 = associated_cost_function(A_1, scenarios, nloc, ncust, capacity, cost, fixedcost, unmet_pen, scaling_factor)
        cost_vector = [cost_1]
        z_matrix = reshape(A_1, nscen, 1)  # Single column matrix
        
        nsubsets = length(cost_vector)
        println("  Generated $nsubsets initial subset (first scenario only)")
    end
    
    # Initialize master problem
    model, v_variables, master_obj, partition_constraints, k_constraint = solve_sf_master_problem(cost_vector, z_matrix, K)
    
    if master_obj == Inf
        println(" Error: Initial master problem is infeasible!")
        return [], Inf, 0, "FAILED"
    end
    
    println("  Initial master problem objective: $master_obj")
    println()
    
    # Column Generation Loop
    for iteration in 1:max_iterations
        println("\033[34m================ ITERATION ", iteration, " ===================\033[0m")
        
        # Stage 2: Master Problem Solution
        println(" Stage 2: Solving Master Problem")
        optimize!(model)
        
        if termination_status(model) != MOI.OPTIMAL
            println("Master problem failed to solve")
            break
        end
        
        master_obj = objective_value(model)
        println("  Master objective: $master_obj")
        
        # Stage 3: Dual Extraction
        println("Stage 3: Extracting Dual Solutions")
        
        # Extract dual values using JuMP pattern
        dual_lambda = dual.(partition_constraints)  # Extract duals from partition constraints (9b)
        dual_mu = dual(k_constraint)  # Extract dual from cardinality constraint (9c)
        
        println("  Dual μ value: $dual_mu")
        
        # Stage 4: Pricing Subproblem
        println(" Stage 4: Solving Pricing Subproblem")
        pi_new, obj_pricing = solve_sf_pricing_subproblem(nloc, ncust, capacity, cost, scenarios, dual_lambda, dual_mu, unmet_pen, scaling_factor, pricing_type)
        
        if obj_pricing == Inf
            println(" Pricing subproblem failed to solve")
            break
        end
        
        println("  Pricing objective: $obj_pricing")
        
        # Stage 5: Convergence Check & Master Problem Update
        println(" Stage 5: Convergence Check & Master Problem Update")
        
        # Check convergence: if obj_pricing < 0.001, we're done
        reduced_cost = obj_pricing
        println("  Reduced cost (obj_pricing): $reduced_cost")
        
        # Store pricing data for table (before potential break)
        push!(iteration_pricing_data, (iteration = iteration, obj_pricing = obj_pricing, dual_mu = dual_mu, reduced_cost = reduced_cost))
        
        if reduced_cost > -1  # Small tolerance for numerical issues
            println(" Algorithm converged! Reduced cost < 0.001")
            break
        else
            println(" Adding new column to master problem")
            
            # Calculate associated cost for new subset
            new_cost = associated_cost_function(pi_new, scenarios, nloc, ncust, capacity, cost, fixedcost, unmet_pen, scaling_factor)
            push!(cost_vector, new_cost)
            
            # Update z_matrix with new column
            z_matrix = hcat(z_matrix, pi_new)
            nsubsets += 1
            
            # Add new v_A variable to master problem
            new_v = @variable(model, lower_bound = 0, upper_bound = 1)
            push!(v_variables, new_v)
            
            # Update objective: set coefficient for new v_A variable
            set_objective_coefficient(model, new_v, new_cost)
            
            # Update the constraint matrix entries
            # Partition constraints (9b): each scenario sums to 1
            for s in 1:nscen
                set_normalized_coefficient(partition_constraints[s], new_v, pi_new[s])
            end
            
            # Cardinality constraint (9c): sum v_A <= K
            set_normalized_coefficient(k_constraint, new_v, 1.0)
            
            println("New column added. Total subsets: $nsubsets")
        end
        
        # Increment actual iterations counter
        actual_iterations = iteration
        
        println()
        
        # Detailed output for analysis
        println(" DETAILED ITERATION ANALYSIS:")
        
        # Optimize model first to get values
        optimize!(model)
        
        println("  Master Problem Objective Value: $(objective_value(model))")
        
        v_values = value.(v_variables)
        
        # Get the last pricing objective
        if iteration <= max_iterations
            println("  Last Pricing Objective: $(obj_pricing)")
        end
    end
    
    # Final solution
    optimize!(model)
    if termination_status(model) == MOI.OPTIMAL
        final_v_values = value.(v_variables)
        final_objective = objective_value(model)
        
        # Find selected subsets (where v_A > 1e-6)
        selected_subsets = findall(x -> x > 1e-6, final_v_values)
        
        println("  Final Results:")
        println("  Iterations: $actual_iterations")
        println("  Final objective: $final_objective")
        println("  Selected subsets: $selected_subsets")
        println("  Final v_A values: $final_v_values")
        
        return selected_subsets, final_objective, actual_iterations, "OPTIMAL", iteration_pricing_data, cost_vector, z_matrix, v_variables
    else
        println(" Final master problem failed to solve")
        return [], Inf, actual_iterations, "FAILED", iteration_pricing_data, [], [], []
    end
end

# SF Formulation - Import data and scenarios from common file
println("="^60)
println("COLUMN GENERATION SF - New Formulation")
println("="^60)

# Import data and scenarios from common file
println(" Importing data and scenarios from common file...")

# Get shared scenarios (should be generated by common file first)
global scenarios = []
try
    global scenarios = get_shared_scenarios()
    scenario_params = get_shared_scenario_params()
    println("  Successfully imported $(length(scenarios)) shared scenarios")
    println("  Scenario parameters: nscen=$(scenario_params["nscen"]), bernoulli=$(scenario_params["bernoulli_case"])")
catch
    println("  ERROR: No shared scenarios available!")
    println("  Please run column_generation_common.jl first to generate scenarios.")
    exit(1)
end

println("  Data and scenarios imported successfully")

# Parameters for testing
K_test = 5      # Number of subsets to select
max_iter_test = 500  # Maximum iterations for testing
pricing_type_test = 1  # Pricing formulation type: 1 = sf_pricing_type1, 2 = sf_pricing_type2
init_type_test = 1  # Initialization type: 1 = random partition, 2 = single subset

println(" Testing SF Column Generation Algorithm")
println("Test Parameters:")
println("  nscen: $(length(scenarios))")
println("  K: $K_test")
println("  max_iterations: $max_iter_test")
println("  pricing_type: $pricing_type_test")
println("  init_type: $init_type_test")
println()

# Run SF Column Generation Algorithm
selected_subsets, final_objective, iterations, status, iteration_pricing_data, cost_vector, z_matrix, v_variables = sf_column_generation_algorithm(
    scenarios, nloc, ncust, capacity, cost, fixedcost, 
    unmet_pen, scaling_factor, K_test, max_iter_test, pricing_type_test, init_type_test)

println()
println("  SF Column Generation Results:")
println("  Iterations: $iterations")
println("  Final objective: $final_objective")
println("  Selected subsets: $selected_subsets")

# Display pricing objective table
println()
println("📊 SF PRICING SUBPROBLEM ANALYSIS:")
println("┌─────────────┬─────────────────────┐")
println("│ Iteration   │ obj_pricing         │")
println("├─────────────┼─────────────────────┤")

for data in iteration_pricing_data
    obj_pricing_str = string(round(data.obj_pricing, digits=6))
    println("│ $(lpad(data.iteration, 11)) │ $(lpad(obj_pricing_str, 19)) │")
end

println("└─────────────┴─────────────────────┘")

# Debug: Print all subsets
println()
println("All Generated Subsets:")
println("Total subsets: $(length(cost_vector))")
for (idx, subset_cost) in enumerate(cost_vector)
    subset_vector = z_matrix[:, idx]
    selected_scenarios = findall(x -> x > 0.5, subset_vector)
    println("  Subset $idx: scenarios $selected_scenarios (cost: $(round(subset_cost, digits=2)))")
end
