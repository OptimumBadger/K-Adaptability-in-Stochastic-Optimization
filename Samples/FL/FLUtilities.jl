# FLUtilities.jl
# Utility functions for Facility Location - used in data generation phase
# This file contains basic utilities that can be imported by CCG when needed

using JuMP, Gurobi
using MathOptInterface
const MOI = MathOptInterface

# =============================================================================
# READ ORLIB CAP FORMAT
# =============================================================================

function read_orlib_cap(filepath::String)
    """Read facility location data from ORLIB format files
    
    Args:
        filepath: Path to the ORLIB format instance file
    
    Returns:
        capacity: Vector of facility capacities
        fixedcost: Vector of facility opening costs
        cost: Matrix of transportation costs (nloc x ncust)
        demand: Vector of customer demands (demandmean)
    """
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

        # Transform cost matrix: divide each column by customer demand
        # This converts from total cost to cost per unit
        for j in 1:m
            if demand[j] > 0  # Avoid division by zero
                cost[:, j] ./= demand[j]
            end
        end

        return capacity, fixedcost, cost, demand
    end
end

# =============================================================================
# LOAD SCENARIOS FROM FILE
# =============================================================================

function load_scenarios_from_file(file_path::String)
    """Load scenarios from a file (one scenario per line, space-separated values)
    
    Each line should contain space-separated floating point numbers (demand values)
    """
    if !isfile(file_path)
        error("❌ Error: Scenarios file not found! Expected: $file_path")
    end
    
    println("Loading scenarios from: $file_path")
    scenarios = Vector{Vector{Float64}}()
    
    open(file_path, "r") do file
        for (line_num, line) in enumerate(eachline(file))
            line = strip(line)
            if length(line) > 0  # Skip empty lines
                try
                    # Parse space-separated floating point numbers
                    values = [parse(Float64, x) for x in split(line)]
                    push!(scenarios, values)
                catch e
                    error("❌ Error parsing line $line_num in scenarios file: $e")
                end
            end
        end
    end
    
    println("✅ Loaded $(length(scenarios)) scenarios")
    if length(scenarios) > 0
        println("   Scenario length: $(length(scenarios[1]))")
    end
    
    return scenarios
end

# =============================================================================
# LOAD BINARY VECTORS FROM FILE
# =============================================================================

function load_binary_vectors_from_file(file_path::String)
    """Load binary vectors from a file (one vector per line)
    
    Each line should contain space-separated integers (0s and 1s)
    """
    if !isfile(file_path)
        error("❌ Error: Binary vectors file not found! Expected: $file_path")
    end
    
    println("Loading binary vectors from: $file_path")
    binary_vectors = Vector{Vector{Int}}()
    
    open(file_path, "r") do file
        for (line_num, line) in enumerate(eachline(file))
            line = strip(line)
            if length(line) > 0  # Skip empty lines
                try
                    # Parse space-separated integers
                    values = [parse(Int, x) for x in split(line)]
                    push!(binary_vectors, values)
                catch e
                    error("❌ Error parsing line $line_num in binary vectors file: $e")
                end
            end
        end
    end
    
    println("✅ Loaded $(length(binary_vectors)) binary vectors")
    if length(binary_vectors) > 0
        println("   Vector length: $(length(binary_vectors[1]))")
    end
    
    return binary_vectors
end

# =============================================================================
# EVALUATE COST FOR FACILITY LOCATION
# =============================================================================

function evaluate_cost_FL(binary_vector::Vector{Int}, scenario::Vector{Float64}, problem_data::Dict)
    """Evaluate cost r_s(x) for Facility Location
    
    Args:
        binary_vector: Binary solution vector (0/1 for each facility)
        scenario: Demand scenario vector
        problem_data: Dict containing capacity, fixedcost, cost, unmet_pen, scaling_factor, nloc, ncust
    
    Returns:
        (cost::Float64, work_units::Float64)
    """
    nloc = problem_data["nloc"]
    ncust = problem_data["ncust"]
    capacity = problem_data["capacity"]
    cost = problem_data["cost"]
    fixedcost = problem_data["fixedcost"]
    unmet_pen = problem_data["unmet_pen"]
    scaling_factor = problem_data["scaling_factor"]
    
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
    @constraint(model, [j in 1:ncust], sum(y[i,j] for i in 1:nloc) + shortfall[j] == scenario[j])
    @constraint(model, [i in 1:nloc], sum(y[i,j] for j in 1:ncust) - capacity[i]*x[i] <= 0)
    
    # Objective function
    @objective(model, Min, 
        sum(fixedcost[i]*x[i] for i in 1:nloc) + 
        scaling_factor * sum(cost[i,j]*y[i,j] for i in 1:nloc for j in 1:ncust) + 
        sum(unmet_pen*shortfall[j] for j in 1:ncust))
    
    optimize!(model)
    
    # Get work units
    work_units = 0.0
    try
        work_units = MOI.get(backend(model), Gurobi.ModelAttribute("Work"))
    catch
    end
    
    if termination_status(model) == MOI.OPTIMAL
        return objective_value(model), work_units
    else
        return Inf, work_units
    end
end

# =============================================================================
# BATCH COST EVALUATION USING TEMPLATE MODEL (OPTIMIZED)
# =============================================================================

function build_template_model_FL(problem_data::Dict)
    """Build a reusable template model for Facility Location cost evaluation
    
    Args:
        problem_data: Dict containing capacity, fixedcost, cost, unmet_pen, scaling_factor, nloc, ncust
    
    Returns:
        model: JuMP model with structure (variables, constraints, objective)
        x: Reference to x variables (will be fixed/unfixed)
        demand_constraints: References to demand constraints (RHS will be updated)
    """
    nloc = problem_data["nloc"]
    ncust = problem_data["ncust"]
    capacity = problem_data["capacity"]
    cost = problem_data["cost"]
    fixedcost = problem_data["fixedcost"]
    unmet_pen = problem_data["unmet_pen"]
    scaling_factor = problem_data["scaling_factor"]
    
    # Create model
    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 0)
    
    # ============================================================
    # VARIABLES (Template - no values fixed yet)
    # ============================================================
    @variable(model, x[1:nloc], Bin)  # Binary, NOT fixed initially
    @variable(model, y[1:nloc, 1:ncust] >= 0)
    @variable(model, shortfall[1:ncust] >= 0)
    
    # ============================================================
    # CONSTRAINTS (Template - RHS will be updated)
    # ============================================================
    # Demand constraints: RHS will be updated for each scenario
    demand_constraints = @constraint(model, [j in 1:ncust], 
        sum(y[i,j] for i in 1:nloc) + shortfall[j] == 0.0)  # Placeholder RHS
    
    # Capacity constraints: RHS depends on x[i], which will be fixed
    @constraint(model, [i in 1:nloc], 
        sum(y[i,j] for j in 1:ncust) - capacity[i]*x[i] <= 0)
    
    # ============================================================
    # OBJECTIVE (Template - structure only, coefficients are fixed)
    # ============================================================
    @objective(model, Min, 
        sum(fixedcost[i]*x[i] for i in 1:nloc) + 
        scaling_factor * sum(cost[i,j]*y[i,j] for i in 1:nloc for j in 1:ncust) + 
        sum(unmet_pen*shortfall[j] for j in 1:ncust))
    
    return model, x, demand_constraints
end

function evaluate_cost_batch_template_FL(
    new_solutions::Vector{Vector{Int}}, 
    scenarios::Vector{Vector{Float64}}, 
    problem_data::Dict
)
    """Evaluate costs for multiple solutions across multiple scenarios using template model
    
    This function builds the model structure once and reuses it by:
    - Fixing/unfixing x variables for each solution
    - Updating demand constraint RHS for each scenario
    
    Args:
        new_solutions: List of binary solution vectors (NEW solutions from pricing subproblem)
        scenarios: List of demand scenarios
        problem_data: Problem data dictionary
    
    Returns:
        cost_matrix: Matrix of costs [scenario_idx, solution_idx]
        total_work_units: Total work units used
    """
    nloc = problem_data["nloc"]
    ncust = problem_data["ncust"]
    nscen = length(scenarios)
    nsol = length(new_solutions)
    
    # ============================================================
    # STEP 1: Build template model ONCE
    # ============================================================
    model, x, demand_constraints = build_template_model_FL(problem_data)
    
    # Initialize results
    cost_matrix = zeros(nscen, nsol)
    total_work_units = 0.0
    
    # ============================================================
    # STEP 2: For each solution, fix x, then iterate scenarios
    # ============================================================
    for sol_idx in 1:nsol
        binary_vector = new_solutions[sol_idx]
        
        # ============================================================
        # FIX x variables to current solution values
        # ============================================================
        for i in 1:nloc
            fix(x[i], binary_vector[i])  # Fix x[i] to 0 or 1
        end
        
        # ============================================================
        # For each scenario, update RHS and solve
        # ============================================================
        for scen_idx in 1:nscen
            scenario = scenarios[scen_idx]
            
            # ============================================================
            # UPDATE demand constraint RHS to current scenario values
            # ============================================================
            for j in 1:ncust
                set_normalized_rhs(demand_constraints[j], scenario[j])
            end
            
            # ============================================================
            # SOLVE the model
            # ============================================================
            optimize!(model)
            
            # Extract results
            if termination_status(model) == MOI.OPTIMAL
                cost_matrix[scen_idx, sol_idx] = objective_value(model)
            else
                cost_matrix[scen_idx, sol_idx] = Inf
            end
            
            # Track work units
            try
                work_units = MOI.get(backend(model), Gurobi.ModelAttribute("Work"))
                total_work_units += work_units
            catch
            end
        end
        
        # ============================================================
        # UNFIX x variables to prepare for next solution
        # ============================================================
        for i in 1:nloc
            unfix(x[i])  # Unfix to prepare for next solution
        end
    end
    
    return cost_matrix, total_work_units
end
