# Compact Facility Location Formulation (Stochastic, Scenario-Average)

using JuMP
using Gurobi
using Random
using Distributions

# Set random seed for reproducibility
Random.seed!(1234)

# =============================================================================
# DATA READING FUNCTIONS
# =============================================================================

function read_orlib_cap(filepath::String)
    """Read facility location data from ORLIB format files (compatible with FL1)."""
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
            if demand[j] > 0
                cost[:, j] ./= demand[j]
            end
        end

        return capacity, fixedcost, cost, demand
    end
end

function read_beasley_cap(filepath::String)
    """Read facility location data from Beasley format files (like cap61.txt)"""
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

        # 3) read all customer demands on a single line
        demand_line = readline(io)
        demand_values = parse.(Float64, split(demand_line))
        demand = demand_values[1:m]  # Take only the first m values

        # 4) read cost matrix (n x m)
        cost = Array{Float64}(undef, n, m)
        for i in 1:n
            cost_line = readline(io)
            cost_values = parse.(Float64, split(cost_line))
            cost[i, :] = cost_values[1:m]  # Take only the first m values
        end
       
        return capacity, fixedcost, cost, demand
    end
end

# =============================================================================
# DEMAND GENERATION FUNCTIONS
# =============================================================================

function generate_demand_scenario(ncust, demandmean, variance_type::String)
    """Generate a single demand scenario using PowerFlow-style formula"""
    
    sigma_range = Dict(
        "low" => (0.1, 0.2),
        "high" => (0.2, 0.3),
        "bernoulli" => (0.1, 0.2)
    )

    if !haskey(sigma_range, lowercase(variance_type))
        error("Invalid variance_type: $variance_type. Must be 'low', 'high', or 'bernoulli'.")
    end

    min_sigma, max_sigma = sigma_range[lowercase(variance_type)]
    sigma = rand(Uniform(min_sigma, max_sigma))

    scenario_demand = Vector{Float64}(undef, ncust)
    global_noise = randn() # Global random variable

    for j in 1:ncust
        individual_noise = randn() # Individual random variable
        
        # PowerFlow-style demand formula
        demand_val = demandmean[j] + sigma * individual_noise * demandmean[j] + global_noise * sigma * demandmean[j]
        
        # Apply Bernoulli multiplier if variance_type is "bernoulli"
        if lowercase(variance_type) == "bernoulli"
            if rand(Bernoulli(0.5)) == 0
                demand_val = 0.0 # Set to zero if Bernoulli is 0
            end
        end
        
        scenario_demand[j] = max(0.0, demand_val) # Ensure non-negativity
    end
    return scenario_demand
end

# =============================================================================
# COMPACT FACILITY LOCATION FUNCTION
# =============================================================================

function facility_location_compact_2(K::Int, S::Int, variance_type::String; 
                                  unmet_pen::Float64=100000.0, 
                                  scaling_factor::Float64=1.0, capacity_factor::Float64=0.05,
                                  file_path::String=joinpath(@__DIR__, "capa11.txt"))
   
    
    println("="^80)
    println("COMPACT FACILITY LOCATION PROBLEM: V2 (first in paper)")
    println("="^80)
    println("Instance: $(basename(file_path))")
    println("K (candidate solutions): $K")
    println("S (scenarios): $S")
    println("Variance type: $variance_type")
    println("Lambda penalty: $unmet_pen")
    println("Scaling factor: $scaling_factor")
    println("Capacity factor: $capacity_factor")
    
    # Read instance data (Beasley format for capa11.txt)
    #capacity, fixedcost, cost, demandmean = read_orlib_cap(file_path)
    capacity, fixedcost, cost, demandmean = read_beasley_cap(file_path)
    
    # Apply scaling factors
    capacity = capacity .* capacity_factor
    cost = cost .* scaling_factor
    
    nfacilities = length(capacity)
    ncustomers = length(demandmean)
    
    println("  Instance loaded: $nfacilities facilities, $ncustomers customers")
    println("  Applied capacity factor of $capacity_factor to all facilities")
    println("  Applied scaling factor of $scaling_factor to transportation costs")
    
    # Generate scenarios
    println("  Generating $S scenarios with variance type: $variance_type")
    scenarios = []
    for s in 1:S
        scenario_demand = generate_demand_scenario(ncustomers, demandmean, variance_type)
        push!(scenarios, scenario_demand)
    end
    println("  Generated $S scenarios successfully")
    
    # Build and solve compact model (inline)
    println("\nBuilding and solving compact formulation...")
    num_facilities = length(capacity)
    num_customers = size(cost, 2)
    num_scenarios = length(scenarios)

    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 1)
    set_optimizer_attribute(model, "LogToConsole", 1)
    set_optimizer_attribute(model, "MIPGap", 0.0001)
    
    # Enable Gurobi log file - format: instanceKS.log
    instance_name = splitext(basename(file_path))[1]  # Remove .txt extension
    log_filename = "$(instance_name)K$(K)S$(S)new.log"
    set_optimizer_attribute(model, "LogFile", log_filename)

    @variable(model, x[1:K, 1:num_facilities], Bin)
    @variable(model, q[1:K, 1:num_scenarios, 1:num_facilities], Bin)
    @variable(model, w[1:num_scenarios, 1:K], Bin)
    @variable(model, y[1:K, 1:num_scenarios, 1:num_facilities, 1:num_customers] >= 0)
    @variable(model, ylin[1:K, 1:num_scenarios, 1:num_facilities, 1:num_customers] >= 0)
	 
    @variable(model,unmet[1:K, 1:num_scenarios, 1:num_customers] >= 0)
    @variable(model,unmetlin[1:K, 1:num_scenarios, 1:num_customers] >= 0)

    @constraint(model, [k in 1:K, s in 1:num_scenarios, j in 1:num_customers],
        sum(y[k,s, i, j] for i in 1:num_facilities) == scenarios[s][j] - unmet[k,s, j]
    )

    @constraint(model, [k in 1:K, s in 1:num_scenarios, i in 1:num_facilities],
        sum(y[k, s, i, j] for j in 1:num_customers) <= capacity[i] * x[k,  i]
    )

    @constraint(model, [s in 1:num_scenarios],
        sum(w[s, k] for k in 1:K) == 1
    )

    @constraint(model, [s in 1:num_scenarios, k in 1:K, i in 1:num_facilities],
        x[k, i] - q[k, s, i] <= 1 - w[s, k]
    )
    @constraint(model, [s in 1:num_scenarios, k in 1:K, i in 1:num_facilities],
        x[k, i] - q[k, s, i] >= w[s, k] - 1
    )

	 # y linearization
	 @constraint(model, [s in 1:num_scenarios, k in 1:K, i in 1:num_facilities, j in 1:num_customers],
        ylin[k,s,i,j] <= y[k,s,i,j]
    )
	 @constraint(model, [s in 1:num_scenarios, k in 1:K, i in 1:num_facilities, j in 1:num_customers],
        ylin[k,s,i,j] <= min(capacity[i],scenarios[s][j])w[s,k]
    )
	 @constraint(model, [s in 1:num_scenarios, k in 1:K, i in 1:num_facilities, j in 1:num_customers],
        ylin[k,s,i,j] >= y[k,s,i,j] - min(capacity[i],scenarios[s][j])*(1-w[s,k])
    )

	  # unmet linearization
	 @constraint(model, [s in 1:num_scenarios, k in 1:K, j in 1:num_customers],
        unmetlin[k,s,j] <= unmet[k,s,j]
    )
	 @constraint(model, [s in 1:num_scenarios, k in 1:K, j in 1:num_customers],
        unmetlin[k,s,j] <= scenarios[s][j]w[s,k]
    )
	 @constraint(model, [s in 1:num_scenarios, k in 1:K,  j in 1:num_customers],
        unmetlin[k,s,j] >= unmet[k,s,j] - scenarios[s][j]*(1-w[s,k])
    )


    @objective(model, Min,
        (1.0 / num_scenarios) * (
            sum( fixedcost[i] * q[k, s, i] for s in 1:num_scenarios, i in 1:num_facilities, k in 1:K ) +
            sum( cost[i, j] * ylin[k,s, i, j] for s in 1:num_scenarios, i in 1:num_facilities, j in 1:num_customers, k in 1:K )
            + sum( unmet_pen * unmetlin[k,s, j] for s in 1:num_scenarios, j in 1:num_customers, k in 1:K )
    ))

    # Set WorkLimit here and solve
    set_optimizer_attribute(model, "WorkLimit", 10000)
    optimize!(model)

    obj_val = objective_value(model)
    status = termination_status(model)
    
    # Simple console output - all details are in Gurobi log file
    println("Results: S=$S, K=$K, variance=$variance_type, facilities=$nfacilities, customers=$ncustomers")
    println("Gap: See Gurobi log above")
    println("Work units: See Gurobi log above") 
    println("Nodes: See Gurobi log above")
    println("Detailed log saved to: $log_filename")

    # Return without extra tables/prints
    if primal_status(model) == MOI.FEASIBLE_POINT
        x_sol = value.(model[:x])
        q_sol = value.(model[:q])
        w_sol = value.(model[:w])
        y_sol = value.(model[:y])
        return model, Dict{String,Any}(), x_sol, q_sol, w_sol, y_sol, nothing
    else
        return model, Dict{String,Any}(), nothing, nothing, nothing, nothing, nothing
    end
end



# build_and_solve_compact removed per simplification

# End of Compact.jl

# =============================================================================
# CLI ENTRYPOINT
# =============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) != 3
        println("Usage: julia CompactFL2_Jim.jl <K> <S> <variance>")
        println("Example: julia CompactFL2_Jim.jl 4 25 bernoulli")
        println("Note: This file is hardcoded for capa11.txt instance")
        exit(1)
    end
    K = parse(Int, ARGS[1])
    S = parse(Int, ARGS[2])
    variance = ARGS[3]
    # Always use facility_location_compact_2 (advisor's new formulation)
    facility_location_compact_2(K, S, variance)
end


