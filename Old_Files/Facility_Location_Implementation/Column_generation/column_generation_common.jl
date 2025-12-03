# Facility Location Common Utilities
# Shared functions for data loading and scenario generation across different formulations

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

# Function to generate demand scenarios
function generate_demand_scenario(ncust, demandmean, demandstdev, bernoulli_case)
    """Generate a single demand scenario"""
    if bernoulli_case
        # Bernoulli case: demand_j = B(0.5) * N(mean=demandmean[j], sd=demandstdev[j])
        scenario_demand = [
            rand(Bernoulli(0.5)) == 1 ? max(0.0, demandmean[j] + demandstdev[j] * randn()) : 0.0
            for j in 1:ncust
        ]
    else
        # Normal case: demand follows normal distribution
        scenario_demand = [max(0.0, demandmean[j] + demandstdev[j] * randn()) for j in 1:ncust]
    end
    return scenario_demand
end

# Function to generate multiple scenarios
function generate_scenarios(nscen, ncust, demandmean, demandstdev, bernoulli_case)
    """Generate multiple demand scenarios"""
    scenarios = []
    for i in 1:nscen
        scenario_demand = generate_demand_scenario(ncust, demandmean, demandstdev, bernoulli_case)
        push!(scenarios, scenario_demand)
    end
    return scenarios
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


# Function to load and prepare facility location data
function load_facility_location_data(file_path::String, scaling_factor::Float64=1.0)
    """Load facility location data and transform cost matrix"""
    if !isfile(file_path)
        println("Error: File $file_path not found!")
        println("Please ensure the file exists in the correct location.")
        return nothing
    end
    
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
    
    return capacity, fixedcost, cost, demand, nloc, ncust
end

# Global variables for shared data and scenarios
global shared_scenarios = []
global shared_scenario_params = Dict()
global capacity = []
global fixedcost = []
global cost = []
global demand = []
global nloc = 0
global ncust = 0
global unmet_pen = 50.0
global scaling_factor = 1.0

# Function to generate shared scenarios for both AF and SF formulations
function generate_shared_scenarios(nscen, ncust, demandmean, demandstdev, bernoulli_case)
    """Generate scenarios once and store them globally for both AF and SF to use"""
    global shared_scenarios, shared_scenario_params
    
    println("Generating $nscen shared scenarios for both AF and SF formulations...")
    
    # Store parameters for reference
    shared_scenario_params = Dict(
        "nscen" => nscen,
        "ncust" => ncust, 
        "demandmean" => demandmean,
        "demandstdev" => demandstdev,
        "bernoulli_case" => bernoulli_case
    )
    
    # Generate scenarios
    shared_scenarios = generate_scenarios(nscen, ncust, demandmean, demandstdev, bernoulli_case)
    
    println("  Generated $(length(shared_scenarios)) shared scenarios")
    return shared_scenarios
end

# Function to get shared scenarios (for AF and SF files to import)
function get_shared_scenarios()
    """Get the pre-generated shared scenarios"""
    global shared_scenarios
    if isempty(shared_scenarios)
        error("No shared scenarios available. Call generate_shared_scenarios() first.")
    end
    return shared_scenarios
end

# Function to get shared scenario parameters
function get_shared_scenario_params()
    """Get the parameters used to generate shared scenarios"""
    global shared_scenario_params
    return shared_scenario_params
end

# =============================================================================
# MAIN EXECUTION: Load data and generate scenarios when common file is run
# =============================================================================

# Load facility location data
println("="^60)
println("FACILITY LOCATION COMMON - Loading Data and Generating Scenarios")
println("="^60)

# Parameters
unmet_pen = 50.0
scaling_factor = 1.0

# Load data and store globally
file_path = "../facility_location_instances/cap92.txt"
capacity, fixedcost, cost, demand, nloc, ncust = load_facility_location_data(file_path, scaling_factor)
global capacity = capacity
global fixedcost = fixedcost
global cost = cost
global demand = demand
global nloc = nloc
global ncust = ncust

println(" Problem Parameters:")
println("  Unmet penalty: $unmet_pen")
println("  Scaling factor: $scaling_factor")

# Generate shared scenarios
nscen_test = 10  # 10 scenarios for testing
demandmean = demand  # Use the original demand as mean
demandstdev = 0.1 * demand  # 10% standard deviation
bernoulli_case = true

generate_shared_scenarios(nscen_test, ncust, demandmean, demandstdev, bernoulli_case)

println("="^60)
println("Common file execution complete. Scenarios ready for AF and SF files.")
println("="^60)
