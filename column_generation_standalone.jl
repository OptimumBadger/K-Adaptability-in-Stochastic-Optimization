# Column Generation Implementation - Standalone Julia Script
# Based on the empty_notebook.ipynb implementation

using JuMP, Gurobi, Random, Distributions

# Set random seed for reproducibility
Random.seed!(1234)

# Data loading function (from facility location)
function read_orlib_cap(filepath::String)
    """Read facility location data from ORLIB format files"""
    open(filepath) do io
        # 1) read n (#facilities) and m (#cities)
        n, m = parse.(Int, split(readline(io)))

        # 2) read facility data: capacity and opening cost
        capacity  = Vector{Float64}(undef, n)
        fixedcost = Vector{Float64}(undef, n)
        for i in 1:n
            # each line: "<capacity> <opening cost>"
            c1, c2 = parse.(Float64, split(readline(io)))
            capacity[i], fixedcost[i] = c1, c2
        end

        # 3) read city‐by‐city: demand + cost to each facility
        demand = Vector{Float64}(undef, m)
        cost = Array{Float64}(undef, n, m)
        for j in 1:m
            # read demand line
            demand[j] = parse(Float64, strip(readline(io)))
            # now read cost entries until we have n of them
            vals = Float64[]
            while length(vals) < n
                append!(vals, parse.(Float64, split(readline(io))))
            end
            cost[:,j] = vals[1:n]
        end

        return capacity, fixedcost, cost, demand
    end
end

# Load original facility location data
println(" Loading facility location data...")

# Load data from file
file_path = "facility location instances/cap91.txt"
if !isfile(file_path)
    println("Error: File $file_path not found!")
    println("Please ensure the file exists in the correct location.")
    exit(1)
else
    println("Found data file: $file_path")
    capacity, fixedcost, cost, demand = read_orlib_cap(file_path)
    nloc = length(capacity)
    ncust = length(demand)
    
    # Transform cost matrix: divide each column by customer demand
    # This converts from total cost to cost per unit
    for j in 1:ncust
        if demand[j] > 0  # Avoid division by zero
            cost[:, j] ./= demand[j]
        end
    end
    
    println("  Data loaded successfully!")
    println("  Locations: $nloc")
    println("  Customers: $ncust")
    println("  Total demand: $(sum(demand))")
    println("  Total capacity: $(sum(capacity))")
    println("  Cost matrix transformed to cost per unit")
end

# Additional parameters for column generation
unmet_pen = 50.0
scaling_factor = 1  # Scaling factor as requested

println(" Problem Parameters:")
println("  Unmet penalty: $unmet_pen")
println("  Scaling factor: $scaling_factor")

# Function to generate demand scenarios (similar to facility location)
function generate_demand_scenario(ncust, demandmean, demandstdev, bernoulli_case)
    """Generate a single demand scenario"""
    if bernoulli_case
        # Bernoulli case: demand is either 0 or 2*demandmean
        scenario_demand = [rand() < 0.5 ? 0.0 : 2.0 * demandmean[j] for j in 1:ncust]
    else
        # Normal case: demand follows normal distribution
        scenario_demand = [max(0.0, demandmean[j] + demandstdev[j] * randn()) for j in 1:ncust]
    end
    return scenario_demand
end

# Function to solve facility location problem and get binary solution
function solve_facility_location_for_binary(nloc, ncust, capacity, cost, demand, fixedcost, unmet_pen, scaling_factor)
    """Solve facility location problem and return binary solution vector x"""
    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 0)  # Suppress output
    set_optimizer_attribute(model, "TimeLimit", 60)  # 1 minute time limit
    
    # Decision Variables
    @variable(model, x[1:nloc], Bin)  # Binary: facility open/closed
    @variable(model, y[1:nloc, 1:ncust] >= 0)  # Continuous: flow
    @variable(model, shortfall[1:ncust] >= 0)  # Continuous: unmet demand
    
    # Constraints
    @constraint(model, [j in 1:ncust], sum(y[i,j] for i in 1:nloc) + shortfall[j] == demand[j])
    @constraint(model, [i in 1:nloc], sum(y[i,j] for j in 1:ncust) - capacity[i]*x[i] <= 0)
    
    # Objective function
    @objective(model, Min, 
        sum(fixedcost[i]*x[i] for i in 1:nloc) + 
        scaling_factor * sum(cost[i,j]*y[i,j] for i in 1:nloc for j in 1:ncust) + 
        sum(unmet_pen*shortfall[j] for j in 1:ncust))
    
    optimize!(model)
    
    if termination_status(model) == MOI.OPTIMAL
        return value.(x), objective_value(model)
    else
        return zeros(nloc), Inf  # Return infeasible solution if failed
    end
end

# Function to generate scenarios and collect binary vectors
function generate_scenarios_and_binary_vectors(nscen, nloc, ncust, capacity, cost, demandmean, demandstdev, fixedcost, unmet_pen, scaling_factor, bernoulli_case)
    """Generate scenarios and solve to get binary solution vectors"""
    println("Generating $nscen scenarios and solving for binary vectors...")
    
    # Storage for scenarios and binary vectors
    scenarios = []
    binary_vectors = []
    objective_values = []
    
    for i in 1:nscen
        # Generate demand scenario
        scenario_demand = generate_demand_scenario(ncust, demandmean, demandstdev, bernoulli_case)
        push!(scenarios, scenario_demand)
        
        # Solve facility location problem
        x_binary, obj_val = solve_facility_location_for_binary(nloc, ncust, capacity, cost, scenario_demand, fixedcost, unmet_pen, scaling_factor)
        
        if obj_val < Inf  # Only store feasible solutions
            push!(binary_vectors, x_binary)
            push!(objective_values, obj_val)
        else
            println("Scenario $i failed to solve")
        end
        
        if i % 10 == 0
            println(" Processed $i scenarios")
        end
    end
    
    println("Generated $(length(binary_vectors)) feasible binary vectors from $nscen scenarios")
    return scenarios, binary_vectors, objective_values
end

# Function to calculate r_s(x) - cost of using binary vector x under scenario s
function calculate_scenario_solution_cost(nloc, ncust, capacity, cost, scenario_demand, binary_vector, fixedcost, unmet_pen, scaling_factor)
    """Calculate the cost r_s(x) of using binary vector x under scenario s"""
    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 0)
    
    # Decision Variables (x is fixed, only y and shortfall are variables)
    @variable(model, x[1:nloc], Bin)
    @variable(model, y[1:nloc, 1:ncust] >= 0)
    @variable(model, shortfall[1:ncust] >= 0)
    
    # Fix x to the given binary vector
    for i in 1:nloc
        fix(x[i], binary_vector[i])
    end
    
    # Constraints
    @constraint(model, [j in 1:ncust], sum(y[i,j] for i in 1:nloc) + shortfall[j] == scenario_demand[j])
    @constraint(model, [i in 1:nloc], sum(y[i,j] for j in 1:ncust) - capacity[i]*x[i] <= 0)
    
    # Objective function
    @objective(model, Min, 
        sum(fixedcost[i]*x[i] for i in 1:nloc) + 
        scaling_factor * sum(cost[i,j]*y[i,j] for i in 1:nloc for j in 1:ncust) + 
        sum(unmet_pen*shortfall[j] for j in 1:ncust))
    
    optimize!(model)
    
    if termination_status(model) == MOI.OPTIMAL
        return objective_value(model)
    else
        return Inf  # Return infinite cost if infeasible
    end
end

# Function to build cost matrix r_s(x) for all scenario-solution pairs
function build_cost_matrix(scenarios, binary_vectors, nloc, ncust, capacity, cost, fixedcost, unmet_pen, scaling_factor)
    """Build the cost matrix r_s(x) for Assignment Formulation"""
    nscen = length(scenarios)
    nsol = length(binary_vectors)
    
    println("Building cost matrix r_s(x) for $nscen scenarios and $nsol solutions...")
    
    cost_matrix = zeros(nscen, nsol)
    
    for s in 1:nscen
        for x_idx in 1:nsol
            cost_matrix[s, x_idx] = calculate_scenario_solution_cost(
                nloc, ncust, capacity, cost, scenarios[s], binary_vectors[x_idx], 
                fixedcost, unmet_pen, scaling_factor)
        end
        
        if s % 10 == 0
            println("  Processed $s scenarios")
        end
    end
    
    println("Cost matrix built successfully")
    return cost_matrix
end

# Function to solve Assignment Formulation (AF) Master Problem with LINEAR RELAXATION
# Following JuMP column generation pattern
function solve_assignment_formulation(cost_matrix, K)
    """Solve the Assignment Formulation (AF) Master Problem from Image 1 - LINEAR RELAXATION"""
    nscen, nsol = size(cost_matrix)
    
    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 0)
    
    # Decision Variables - LINEAR RELAXATION (continuous, not binary)
    # Use vectors that can be dynamically extended
    sigma = [@variable(model, lower_bound = 0) for x in 1:nsol]  # σ_x: weight of solution x (continuous)
    rho = [[@variable(model, lower_bound = 0) for x in 1:nsol] for s in 1:nscen]  # ρ_{s,x}: assignment of scenario s to solution x (continuous)
    
    # Objective Function (5a)
    @objective(model, Min, sum(cost_matrix[s, x] * rho[s][x] for s in 1:nscen for x in 1:nsol))
    
    # Constraints - Store references for dual extraction
    # Assignment constraints (5b): Each scenario must be assigned to exactly one solution
    assignment_constraints = @constraint(model, [s in 1:nscen], sum(rho[s][x] for x in 1:nsol) == 1)
    
    # Selection constraints (5c): ρ_{s,x} <= σ_x for all s, x
    @constraint(model, [s in 1:nscen, x in 1:nsol], rho[s][x] <= sigma[x])
    
    # K constraint (5d): Sum of σ_x <= K
    k_constraint = @constraint(model, sum(sigma[x] for x in 1:nsol) <= K)
    
    # Register early-termination callback: stop when incumbent objective ≥ -dual_mu
    function column_generation_callback(cb_data, cb_where)
        if cb_where == 4
            try
                current_obj = callback_value(cb_data, objective_function(model))
                if current_obj >= -dual_mu + 1e-9
                    return 1
                end
            catch
            end
        end
        return 0
    end
    MOI.set(model, Gurobi.CallbackFunction(), column_generation_callback)

    # Solve
    optimize!(model)
    
    if termination_status(model) == MOI.OPTIMAL
        return model, sigma, rho, objective_value(model), assignment_constraints, k_constraint
    else
        return model, sigma, rho, Inf, assignment_constraints, k_constraint
    end
end

# Function to analyze AF solution
function analyze_af_solution(sigma, rho, cost_matrix, K)
    """Analyze the solution of Assignment Formulation"""
    nscen, nsol = size(cost_matrix)
    
    # Get values from variable references
    sigma_values = value.(sigma)
    rho_values = [value.(rho[s]) for s in 1:nscen]
    
    println(" Assignment Formulation Solution Analysis:")
    println("  Objective value: $(sum(cost_matrix[s, x] * rho_values[s][x] for s in 1:nscen for x in 1:nsol))")
    println("  K constraint: $(sum(sigma_values)) <= $K")
    
    # Find selected solutions (σ_x > 0)
    selected_solutions = findall(x -> x > 1e-6, sigma_values)
    println("  Selected solutions: $selected_solutions")
    
    # Show scenario assignments
    println("  Scenario assignments:")
    for s in 1:min(5, nscen)  # Show first 5 scenarios
        assigned_solution = findfirst(x -> x > 1e-6, rho_values[s])
        if assigned_solution !== nothing
            println("    Scenario $s -> Solution $assigned_solution (weight: $(rho_values[s][assigned_solution]))")
        end
    end
    
    return selected_solutions
end

# Function to calculate m^s for a single scenario (Formulation 2) 
function big_m_calculator(nloc, ncust, capacity, cost, scenario_demand, fixedcost, unmet_pen, scaling_factor, dual_lambda)
    """Calculate m^s by solving Formulation 2 for a single scenario - Based on Image Formulation"""
    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 1)  # Enable Gurobi output
    set_optimizer_attribute(model, "LogToConsole", 1)  # Show progress in console
    set_optimizer_attribute(model, "MIPGap", 0.01)  # Set gap tolerance to 1%
    set_optimizer_attribute(model, "TimeLimit", 300)  # Set 300 second time limit
    
    # Decision Variables 
    @variable(model, x[1:nloc], Bin)  # Binary: facility open/closed
    @variable(model, y[1:nloc, 1:ncust] >= 0)  # Continuous: flow
    @variable(model, s_j[1:ncust] >= 0)  # Continuous: unmet demand (shortfall)
    
    # Constraints 
    
    # Constraint 1: Demand satisfaction constraint
    @constraint(model, [j in 1:ncust], sum(y[i,j] for i in 1:nloc) + s_j[j] == scenario_demand[j])
    
    # Constraint 2: Facility capacity constraint
    @constraint(model, [i in 1:nloc], sum(y[i,j] for j in 1:ncust) - capacity[i]*x[i] <= 0)
    
    # Objective function (as per image: m^s = min { (λ^s)* - Σ f_i x_i - ΣΣ c_ij y_ij^s - Σ α s_j^s })
    @objective(model, Min, 
        dual_lambda - sum(fixedcost[i]*x[i] for i in 1:nloc) - 
        scaling_factor * sum(cost[i,j]*y[i,j] for i in 1:nloc for j in 1:ncust) - 
        sum(unmet_pen*s_j[j] for j in 1:ncust))
    
    optimize!(model)
    
    if termination_status(model) == MOI.OPTIMAL
        return objective_value(model)
    else
        return Inf  # Return infinite if infeasible
    end
end

# Function to solve AF Pricing Type 1 
function af_pricing_type1(nloc, ncust, capacity, cost, scenarios, m_values, dual_lambda, dual_mu, unmet_pen, scaling_factor)
    """Solve AF Pricing Type 1 using m^s as parameters - Based on Updated Image Formulation"""
    nscen = length(scenarios)
    
    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 1)  # Enable Gurobi output
    set_optimizer_attribute(model, "LogToConsole", 1)  # Show progress in console
    set_optimizer_attribute(model, "MIPGap", 0.01)  # Set gap tolerance to 1%
    set_optimizer_attribute(model, "TimeLimit", 300)  # Set 30 second time limit
    
    # Decision Variables 
    @variable(model, v[1:nloc, 1:nscen], Bin)  # Binary: facility i open in scenario s
    @variable(model, pi[1:nscen], Bin)  # Binary: scenario selection
    @variable(model, x[1:nloc], Bin)  # Binary: facility open/closed (the solution vector we want)
    @variable(model, u[1:nloc, 1:ncust, 1:nscen] >= 0)  # Continuous: flow component
    @variable(model, y[1:nloc, 1:ncust, 1:nscen] >= 0)  # Continuous: total flow
    @variable(model, w[1:ncust, 1:nscen] >= 0)  # Continuous: auxiliary shortfall variable
    @variable(model, b[1:ncust, 1:nscen] >= 0)  # Continuous: shortfall decision variable
    
    # Pre-calculate N[i,j,s] = min(capacity[i], scenarios[s][j])
    N = Array{Float64}(undef, nloc, ncust, nscen)
    for i in 1:nloc, j in 1:ncust, s in 1:nscen
        N[i,j,s] = min(capacity[i], scenarios[s][j])
    end
    
    # Objective Function 
    @objective(model, Max, 
        sum(dual_lambda[s] * pi[s] for s in 1:nscen) - 
        sum(fixedcost[i] * v[i,s] for i in 1:nloc for s in 1:nscen) - 
        scaling_factor * sum(cost[i,j] * u[i,j,s] for i in 1:nloc for j in 1:ncust for s in 1:nscen) - 
        sum(unmet_pen * w[j,s] for j in 1:ncust for s in 1:nscen))
    
    # Constraints 
    # Constraint 1: Flow balance for each scenario (updated to use b_j^s)
    @constraint(model, [j in 1:ncust, s in 1:nscen], 
        sum(y[i,j,s] for i in 1:nloc) + b[j,s] == scenarios[s][j])
    
    # Constraint 2: Capacity constraints for each scenario
    @constraint(model, [i in 1:nloc, s in 1:nscen], 
        sum(y[i,j,s] for j in 1:ncust) - capacity[i] * x[i] <= 0)
    
    # Constraint 3: Key pricing constraint (updated to use w_j^s)
    @constraint(model, [s in 1:nscen], 
        dual_lambda[s] - sum(fixedcost[i] * x[i] for i in 1:nloc) - 
        scaling_factor * sum(cost[i,j] * y[i,j,s] for i in 1:nloc for j in 1:ncust) - 
        sum(unmet_pen * b[j,s] for j in 1:ncust) >= m_values[s] * (1 - pi[s]))
    
    # Constraint 4: v_i^s ≤ x_i
    @constraint(model, [i in 1:nloc, s in 1:nscen], v[i,s] <= x[i])
    
    # Constraint 5: v_i^s ≤ π^s
    @constraint(model, [i in 1:nloc, s in 1:nscen], v[i,s] <= pi[s])
    
    # Constraint 6: v_i^s ≥ x_i + π^s - 1
    @constraint(model, [i in 1:nloc, s in 1:nscen], v[i,s] >= x[i] + pi[s] - 1)
    
    # Constraint 7: u_ij^s ≤ y_ij^s
    @constraint(model, [i in 1:nloc, j in 1:ncust, s in 1:nscen], u[i,j,s] <= y[i,j,s])
    
    # Constraint 8: u_ij^s ≤ min{ū_i, d_j^s} π^s
    @constraint(model, [i in 1:nloc, j in 1:ncust, s in 1:nscen], u[i,j,s] <= N[i,j,s] * pi[s])
    
    # Constraint 9: u_ij^s ≥ y_ij^s + min{ū_i, d_j^s}(π^s - 1)
    @constraint(model, [i in 1:nloc, j in 1:ncust, s in 1:nscen], 
        u[i,j,s] >= y[i,j,s] + N[i,j,s] * (pi[s] - 1))
    
    # Constraint 10: w_j^s ≤ b_j^s
    @constraint(model, [j in 1:ncust, s in 1:nscen], w[j,s] <= b[j,s])
    
    # Constraint 11: w_j^s ≤ d_j^s π^s
    @constraint(model, [j in 1:ncust, s in 1:nscen], w[j,s] <= scenarios[s][j] * pi[s])
    
    # Constraint 12: w_j^s ≥ b_j^s + d_j^s (π^s - 1)
    @constraint(model, [j in 1:ncust, s in 1:nscen], 
        w[j,s] >= b[j,s] + scenarios[s][j] * (pi[s] - 1))
    
    # Register early-termination callback: stop when incumbent objective ≥ -dual_mu
    # function column_generation_callback(cb_data, cb_where)
    #     if cb_where == 4
    #         try
    #             current_obj = callback_value(cb_data, objective_function(model))
    #             if current_obj >= -dual_mu + 1e-9
    #                 return 1
    #             end
    #         catch
    #         end
    #     end
    #     return 0
    # end
    # MOI.set(model, Gurobi.CallbackFunction(), column_generation_callback)

    # Solve
    optimize!(model)
    
    ts = termination_status(model)
    if ts == MOI.OPTIMAL || ts == MOI.TIME_LIMIT || ts == MOI.OTHER_ERROR || ts == MOI.SUBOPTIMAL
        
        return value.(x), objective_value(model)
    else
        return zeros(nloc), Inf
    end
end

# Function to solve AF Pricing Type 2 (Alternative Formulation) - NEW IMPLEMENTATION
function af_pricing_type2(nloc, ncust, capacity, cost, scenarios, m_values, dual_lambda, dual_mu, unmet_pen, scaling_factor)
    """Solve AF Pricing Type 2 using alternative linearization technique - Based on New Image Formulation"""
    nscen = length(scenarios)
    
    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 1)  # Enable Gurobi output
    set_optimizer_attribute(model, "LogToConsole", 1)  # Show progress in console
    set_optimizer_attribute(model, "MIPGap", 0.01)  # Set gap tolerance to 1%
    set_optimizer_attribute(model, "TimeLimit", 300)  # Set 300 second time limit
    
    # Decision Variables
    @variable(model, x[1:nloc], Bin)  # Binary: first-stage facility decisions
    @variable(model, x_s[1:nloc, 1:nscen], Bin)  # Binary: second-stage facility decisions
    @variable(model, y[1:nloc, 1:ncust, 1:nscen] >= 0)  # Continuous: flow
    @variable(model, b[1:ncust, 1:nscen] >= 0)  # Continuous: shortfall
    @variable(model, pi[1:nscen], Bin)  # Binary: scenario selection
    
    
    # Objective Function (as per new image: MAXIMIZATION)
    @objective(model, Max, 
        sum(dual_lambda[s] * pi[s] for s in 1:nscen) - 
        sum(fixedcost[i] * x_s[i,s] for i in 1:nloc for s in 1:nscen) - 
        scaling_factor * sum(cost[i,j] * y[i,j,s] for i in 1:nloc for j in 1:ncust for s in 1:nscen) - 
        sum(unmet_pen * b[j,s] for j in 1:ncust for s in 1:nscen))
    
    
    # Constraint 1: x^s ≤ e π^s for all s ∈ S (e is vector of ones)
    @constraint(model, [i in 1:nloc, s in 1:nscen], x_s[i,s] <= pi[s])
    
    # Constraint 2: x - x^s ≤ e(1 - π^s) for all s ∈ S (e is vector of ones)
    @constraint(model, [i in 1:nloc, s in 1:nscen], x[i] - x_s[i,s] <= (1 - pi[s]))
    
    # Constraint 3: x - x^s ≥ -e(1 - π^s) for all s ∈ S (e is vector of ones)
    @constraint(model, [i in 1:nloc, s in 1:nscen], x[i] - x_s[i,s] >= -(1 - pi[s]))
    
    # Constraint 4: (λ*)^s π^s - Σ f_i x_i^s - ΣΣ c_ij y_ij^s - Σ α b_j^s ≥ 0 for all s ∈ S
    @constraint(model, [s in 1:nscen], 
        dual_lambda[s] * pi[s] - sum(fixedcost[i] * x_s[i,s] for i in 1:nloc) - 
        scaling_factor * sum(cost[i,j] * y[i,j,s] for i in 1:nloc for j in 1:ncust) - 
        sum(unmet_pen * b[j,s] for j in 1:ncust) >= 0)
    
    # Constraint 5: Σ y_ij^s - u_i x_i^s ≤ 0 for all i ∈ I, s ∈ S
    @constraint(model, [i in 1:nloc, s in 1:nscen], 
        sum(y[i,j,s] for j in 1:ncust) - capacity[i] * x_s[i,s] <= 0)
    
    # Constraint 6: Σ y_ij^s + b_j^s = d_j^s π^s for all j ∈ J, s ∈ S
    @constraint(model, [j in 1:ncust, s in 1:nscen], 
        sum(y[i,j,s] for i in 1:nloc) + b[j,s] == scenarios[s][j] * pi[s])
    
    # Register early-termination callback: stop when incumbent objective ≥ -dual_mu
    # function column_generation_callback(cb_data, cb_where)
    #     if cb_where == 4
    #         try
    #             current_obj = callback_value(cb_data, objective_function(model))
    #             if current_obj >= -dual_mu + 0.1
    #                 println("Callback triggered at objective: $current_obj")
    #                 return 1
    #             end
    #         catch
    #         end
    #     end
    #     return 0
    # end
    # MOI.set(model, Gurobi.CallbackFunction(), column_generation_callback)
    
    # Solve
    optimize!(model)
    
    ts = termination_status(model)
    if ts == MOI.OPTIMAL || ts == MOI.TIME_LIMIT || ts == MOI.OTHER_ERROR || ts == MOI.SUBOPTIMAL
        # If we hit BestObjStop or time limit with an incumbent, objective_value is defined
        return value.(x), objective_value(model)
    else
        return zeros(nloc), Inf
    end
end

# Function to solve pricing subproblem
function solve_pricing_subproblem(nloc, ncust, capacity, cost, scenarios, dual_lambda, dual_mu, unmet_pen, scaling_factor, pricing_type=1)
    """Solve the complete pricing subproblem with choice of formulation type"""
    nscen = length(scenarios)
    
    if pricing_type == 1
        # Step 1: Calculate m^s for all scenarios using Formulation 2
        println("  Calculating m^s values for all scenarios...")
        m_values = zeros(nscen)
        
        for s in 1:nscen
            m_values[s] = big_m_calculator(nloc, ncust, capacity, cost, scenarios[s], fixedcost, unmet_pen, scaling_factor, dual_lambda[s])
        end
        
        # Step 2: Solve AF Pricing Type 1 using m^s as parameters
        println("  Solving AF Pricing Type 1...")
        x_optimal, obj_value = af_pricing_type1(nloc, ncust, capacity, cost, scenarios, m_values, dual_lambda, dual_mu, unmet_pen, scaling_factor)
        
        return x_optimal, obj_value, m_values
        
    elseif pricing_type == 2
        # Solve AF Pricing Type 2 (alternative formulation - no m^s calculation needed)
        println("  Solving AF Pricing Type 2 (Alternative Formulation)...")
        m_values = zeros(nscen)  # Not used in Type 2, but kept for compatibility
        x_optimal, obj_value = af_pricing_type2(nloc, ncust, capacity, cost, scenarios, m_values, dual_lambda, dual_mu, unmet_pen, scaling_factor)
        
        return x_optimal, obj_value, m_values
        
    else
        error("Invalid pricing_type. Use 1 for original formulation or 2 for alternative formulation.")
    end
end



# Main Column Generation Algorithm
function column_generation_algorithm(scenarios, nloc, ncust, capacity, cost, fixedcost, unmet_pen, scaling_factor, K, max_iterations, pricing_type=1, init_type=2)
    """Main Column Generation Algorithm with 5 stages"""
    nscen = length(scenarios)
    
    println(" Starting Column Generation Algorithm")
    println("Parameters: nscen=$nscen, nloc=$nloc, ncust=$ncust, K=$K, max_iter=$max_iterations")
    println()
    
    # Initialize pricing + dual_mu tracking
    iteration_pricing_data = []
    actual_iterations = 0  # Track actual number of iterations completed
    
    
    # Stage 1: Initialization
    println(" Stage 1: Initialization")
    
    if init_type == 1
        # Option 1: Solve all scenarios from scratch
        println("  Initialization Type 1: Solving all scenarios from scratch...")
        initial_binary_vectors = []
        initial_objective_values = []
        
        for s in 1:nscen
            println("    Solving scenario $s/$nscen...")
            x_optimal, obj_optimal = solve_facility_location_for_binary(nloc, ncust, capacity, cost, scenarios[s], fixedcost, unmet_pen, scaling_factor)
            push!(initial_binary_vectors, x_optimal)
            push!(initial_objective_values, obj_optimal)
        end
        
        nsol = length(initial_binary_vectors)
        println("  Generated $nsol initial binary vectors from scenario solutions")
    else
        # Option 2: Start with zero vector (default behavior)
        println("  Initialization Type 2: Starting with zero vector...")
        initial_binary_vectors = [zeros(nloc)]
        initial_objective_values = [0.0]  # Dummy objective value
        
        nsol = length(initial_binary_vectors)
        println("  Generated $nsol initial binary vector (all zeros)")
    end
    
    # Build initial cost matrix
    cost_matrix = build_cost_matrix(scenarios, initial_binary_vectors, nloc, ncust, capacity, cost, fixedcost, unmet_pen, scaling_factor)
    
    # Initialize master problem
    model, sigma, rho, master_obj, assignment_constraints, k_constraint = solve_assignment_formulation(cost_matrix, K)
    
    if master_obj == Inf
        println(" Error: Initial master problem is infeasible!")
        return [], Inf, 0, "FAILED"
    end
    
    println("  Initial master problem objective: $master_obj")
    println()
    
    # Column Generation Loop - Following JuMP pattern
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
        
        # Stage 3: Dual Extraction - Following JuMP pattern
        println("Stage 3: Extracting Dual Solutions")
        
        # Extract dual values using JuMP pattern
        dual_lambda = dual.(assignment_constraints)  # Extract duals from assignment constraints
        dual_mu = dual(k_constraint)  # Extract dual from K constraint
        
        #println("  Dual λ values: $(dual_lambda[1:min(5, nscen)])...")
        println("  Dual μ value: $dual_mu")
        
        # Stage 4: Pricing Subproblem
        println(" Stage 4: Solving Pricing Subproblem")
        x_new, obj_pricing, m_values = solve_pricing_subproblem(nloc, ncust, capacity, cost, scenarios, dual_lambda, dual_mu, unmet_pen, scaling_factor, pricing_type)
        
        if obj_pricing == Inf
            println(" Pricing subproblem failed to solve")
            break
        end
        
        println("  Pricing objective: $obj_pricing")
        
        # Stage 5: Convergence Check & Master Problem Update
        println(" Stage 5: Convergence Check & Master Problem Update")
        
        # Check convergence: if obj_pricing + dual_mu < 0, we're done
        reduced_cost = obj_pricing + dual_mu
        println("  Reduced cost (obj_pricing + μ): $reduced_cost")
        
        # Store pricing + dual_mu data for table (before potential break)
        push!(iteration_pricing_data, (iteration = iteration, obj_pricing = obj_pricing, dual_mu = dual_mu, obj_pricing_plus_dual_mu = obj_pricing + dual_mu))
        
        if reduced_cost < 0.001  # Small tolerance for numerical issues
            println(" Algorithm converged! Reduced cost < 0")
            break
        else
            println(" Adding new column to master problem")
            
            # Add new binary vector to collection
            push!(initial_binary_vectors, x_new)
            nsol += 1
            
            # Calculate new column costs
            new_column_costs = zeros(nscen)
            for s in 1:nscen
                new_column_costs[s] = calculate_scenario_solution_cost(nloc, ncust, capacity, cost, scenarios[s], x_new, fixedcost, unmet_pen, scaling_factor)
            end
            
        # Add new variables to master problem - Following JuMP pattern
        new_sigma = @variable(model, lower_bound = 0)
        new_rho = [@variable(model, lower_bound = 0) for s in 1:nscen]
        push!(sigma, new_sigma)
        for s in 1:nscen
            push!(rho[s], new_rho[s])
        end
        
            
            # Update objective: set coefficients for new rho column
            for s in 1:nscen
                set_objective_coefficient(model, new_rho[s], new_column_costs[s])
            end
            
            # Update the constraint matrix entries
            # Assignment constraints (5b): each scenario sums to 1
            for s in 1:nscen
                set_normalized_coefficient(assignment_constraints[s], new_rho[s], 1.0)
            end
            
            # Selection constraints (5c): rho[s,x] <= sigma[x]
            for s in 1:nscen
                @constraint(model, new_rho[s] <= new_sigma)
            end
            
            # K constraint (5d): sum sigma <= K
            set_normalized_coefficient(k_constraint, new_sigma, 1.0)
            
            # Update cost matrix bookkeeping (optional for analysis)
            cost_matrix = hcat(cost_matrix, new_column_costs)
            
            println("New column added. Total solutions: $nsol")
            
        end
        
        # Increment actual iterations counter
        actual_iterations = iteration
        
        println()
        
        # Detailed output for analysis
        println(" DETAILED ITERATION ANALYSIS:")
        
        # Optimize model first to get values
        optimize!(model)
        
        
        println("  Master Problem Objective Value: $(objective_value(model))")
        
      
        sigma_values = value.(sigma)
        
        rho_values = [value.(rho[s]) for s in 1:nscen]
        
        # Get the last pricing objective and dual mu
        if iteration <= max_iterations
            println("  Last Pricing Objective: $(obj_pricing)")
            println("  Last Dual μ: $(dual_mu)")
            println("  Last obj_pricing + dual_mu: $(obj_pricing + dual_mu)")
        end
    end
    
    # Final solution
    optimize!(model)
    if termination_status(model) == MOI.OPTIMAL
        final_sigma_values = value.(sigma)
        final_rho_values = [value.(rho[s]) for s in 1:nscen]
        final_objective = objective_value(model)
        
        # Find selected solutions
        selected_solutions = findall(x -> x > 1e-6, final_sigma_values)
        
        println("  Final Results:")
        println("  Iterations: $actual_iterations")
        println("  Final objective: $final_objective")
        println("  Selected solutions: $selected_solutions")
        
        return selected_solutions, final_objective, actual_iterations, "OPTIMAL", iteration_pricing_data, initial_binary_vectors, cost_matrix, sigma
    else
        println(" Final master problem failed to solve")
        return [], Inf, actual_iterations, "FAILED", iteration_pricing_data, [], [], []
    end
end


# First, let's generate some test scenarios
println(" Generating test scenarios...")

# Generate demand scenarios for testing
nscen_test = 100
demandmean = demand  # Use the original demand as mean
demandstdev = 0.1 * demand  # 10% standard deviation
bernoulli_case = false

# Generate scenarios
scenarios = []
for i in 1:nscen_test
    scenario_demand = generate_demand_scenario(ncust, demandmean, demandstdev, bernoulli_case)
    push!(scenarios, scenario_demand)
end

println(" Generated $nscen_test test scenarios")

# Parameters for testing
K_test = 10      # Number of solutions to select
max_iter_test = 30  # Maximum iterations for testing
pricing_type_test = 1  # Pricing formulation type: 1 = original, 2 = alternative
init_type_test = 2  # Initialization type: 1 = solve all scenarios, 2 = start with zero vector

println(" Testing Column Generation Algorithm")
println("Test Parameters:")
println("  nscen: $nscen_test")
println("  K: $K_test")
println("  max_iterations: $max_iter_test")
println("  pricing_type: $pricing_type_test")
println("  init_type: $init_type_test")
println()

# Run Column Generation Algorithm
selected_solutions, final_objective, iterations, status, iteration_pricing_data, binary_vectors, cost_matrix, sigma = column_generation_algorithm(
    scenarios, nloc, ncust, capacity, cost, fixedcost, 
    unmet_pen, scaling_factor, K_test, max_iter_test, pricing_type_test, init_type_test)

println()
println("  Column Generation Results:")
println("  Iterations: $iterations")
println("  Final objective: $final_objective")
println("  Selected solutions: $selected_solutions")

# Display pricing + dual_mu table
println()
println("📊 PRICING SUBPROBLEM ANALYSIS:")
println("┌─────────────┬─────────────────────┬─────────────────────┬─────────────────────┐")
println("│ Iteration   │ obj_pricing         │ dual_mu             │ obj_pricing + dual_mu│")
println("├─────────────┼─────────────────────┼─────────────────────┼─────────────────────┤")

for data in iteration_pricing_data
    obj_pricing_str = string(round(data.obj_pricing, digits=6))
    dual_mu_str = string(round(data.dual_mu, digits=6))
    sum_str = string(round(data.obj_pricing_plus_dual_mu, digits=6))
    println("│ $(lpad(data.iteration, 11)) │ $(lpad(obj_pricing_str, 19)) │ $(lpad(dual_mu_str, 19)) │ $(lpad(sum_str, 19)) │")
end

println("└─────────────┴─────────────────────┴─────────────────────┴─────────────────────┘")


# Debug: Print all binary solutions
println()
println("All Binary Solutions:")
println("Total solutions: $(length(binary_vectors))")
for (idx, sol) in enumerate(binary_vectors)
    println("  Solution $idx: $sol")
end

# # Function to evaluate solution quality
# function evaluate_solution_quality(scenarios, selected_solutions, nloc, ncust, capacity, cost, fixedcost, unmet_pen, scaling_factor)
#     """Evaluate quality of selected solutions by comparing with from-scratch solutions"""
#     nscen = length(scenarios)
#     nselected = length(selected_solutions)
#     
#     println("\n" * "="^80)
#     println("SOLUTION QUALITY EVALUATION")
#     println("="^80)
#     println("Evaluating $nselected selected solutions against $nscen scenarios...")
#     
#     # Initialize results matrix: rows = scenarios, cols = [from_scratch, sol1, sol2, ...]
#     results = zeros(nscen, nselected + 1)
#     
#     # Solve each scenario from scratch
#     println("\nSolving scenarios from scratch...")
#     for s in 1:nscen
#         println("  Scenario $s/$nscen...")
#         x_optimal, obj_from_scratch = solve_facility_location_for_binary(
#             nloc, ncust, capacity, cost, scenarios[s], fixedcost, unmet_pen, scaling_factor)
#         results[s, 1] = obj_from_scratch
#     end
#     
#     # Evaluate each selected solution on each scenario
#     println("\nEvaluating selected solutions...")
#     for sol_idx in 1:nselected
#         println("  Selected Solution $sol_idx/$nselected...")
#         for s in 1:nscen
#             obj_cost = calculate_scenario_solution_cost(
#                 nloc, ncust, capacity, cost, scenarios[s], selected_solutions[sol_idx], 
#                 fixedcost, unmet_pen, scaling_factor)
#             results[s, sol_idx + 1] = obj_cost
#         end
#     end
#     
#     # Display results table
#     println("\n" * "="^80)
#     println("SOLUTION QUALITY COMPARISON TABLE")
#     println("="^80)
#     println("Rows: Scenarios | Col 1: From Scratch | Col 2+: Selected Solutions")
#     println("-"^80)
#     
#     # Header
#     header = lpad("Scenario", 10)
#     header *= lpad("From Scratch", 15)
#     for sol_idx in 1:nselected
#         header *= lpad("Sol $sol_idx", 12)
#     end
#     println(header)
#     println("-"^80)
#     
#     # Data rows
#     for s in 1:nscen
#         row = lpad("$s", 10)
#         for col in 1:(nselected + 1)
#             row *= lpad(round(results[s, col], digits=2), col == 1 ? 15 : 12)
#         end
#         println(row)
#     end
#     
#     # Summary statistics
#     println("-"^80)
#     println("SUMMARY STATISTICS:")
#     println("Average From Scratch Cost: $(round(mean(results[:, 1]), digits=2))")
#     for sol_idx in 1:nselected
#         avg_cost = round(mean(results[:, sol_idx + 1]), digits=2)
#         gap = round((avg_cost - mean(results[:, 1])) / mean(results[:, 1]) * 100, digits=1)
#         println("Average Selected Solution $sol_idx Cost: $avg_cost (Gap: $gap%)")
#     end
#     
#     return results
# end

# # Evaluate solution quality after column generation
# println("\n" * "="^80)
# println("EVALUATING SOLUTION QUALITY")
# println("="^80)

# # Get selected solutions (where sigma = 1)
# selected_indices = findall(x -> abs(x - 1.0) < 1e-6, value.(sigma))
# selected_solutions = [binary_vectors[idx] for idx in selected_indices]

# println("Selected solutions (sigma = 1): $selected_indices")
# println("Number of selected solutions: $(length(selected_solutions))")

# if length(selected_solutions) > 0
#     # Evaluate solution quality
#     evaluation_results = evaluate_solution_quality(
#         scenarios, selected_solutions, nloc, ncust, capacity, cost, 
#         fixedcost, unmet_pen, scaling_factor)
# else
#     println("No solutions selected (sigma = 1). Cannot evaluate solution quality.")
# end