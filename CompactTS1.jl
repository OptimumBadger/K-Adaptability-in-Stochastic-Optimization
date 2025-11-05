#!/usr/bin/env julia

# =============================================================================
# COMPACT POWERFLOW FORMULATION
# =============================================================================
# This file implements the compact PowerFlow formulation for candidate selection
# Instance: case118Blumsack.m
# Usage: julia CompactTS1.jl <K> <S> <variance_type>
# Example: julia CompactTS1.jl 2 20 low

using JuMP
using Gurobi
using XLSX
using Random
using Distributions
using DataFrames
using PowerModels

# =============================================================================
# DATA READING FUNCTIONS
# =============================================================================

function load_network(file_path::String)
    """Load MATPOWER case using PowerModels (same approach as TS1.jl)"""
    return PowerModels.parse_file(file_path)
end

function parameters(network; line_capacity_factor::Float64=1.0)
    """Extract network parameters (mirrors TS1.jl)"""
    B = network["bus"]
    L = network["branch"]
    G = network["gen"]

    B_ij = zeros(length(L))
    fbar_ij = zeros(length(L))
    p_max = zeros(length(B))
    p_min = zeros(length(B))
    c = zeros(length(B))

    for (branch_id, branch) in L
        B_ij[parse(Int, branch_id)] = -branch["br_x"]/(branch["br_r"]^2 + branch["br_x"]^2)
        rate_a = haskey(branch, "rate_a") ? branch["rate_a"] : 2.2
        fbar_ij[parse(Int, branch_id)] = rate_a * line_capacity_factor
    end

    for (gen_id, generator) in G
        c[generator["gen_bus"]] = generator["cost"][1]
        p_max[generator["gen_bus"]] = generator["pmax"]
        p_min[generator["gen_bus"]] = generator["pmin"]
    end

    n = 2*length(network["bus"])
    m = length(network["branch"])
    A = zeros(n, m)

    for (branch_id, branch) in L
        f_bus = branch["f_bus"]
        t_bus = branch["t_bus"]
        idx = parse(Int, branch_id)
        A[2*f_bus-1, idx] = 1
        A[2*f_bus, idx] = -1
        A[2*t_bus-1, idx] = -1
        A[2*t_bus, idx] = 1
    end

    return B_ij, fbar_ij, p_max, p_min, A, c
end

function load_demand_data(file_path_excel::String, worksheet_name::String)
    """Load demand mean values from Excel file (same as TS1.jl)"""
    param_data = XLSX.readtable(file_path_excel, worksheet_name) |> DataFrame
    demand_mean_data = Vector{Float64}(param_data[1, :])
    return demand_mean_data
end

# =============================================================================
# DEMAND GENERATION FUNCTIONS
# =============================================================================

function generate_demand_scenario(nbus, demand_mean, variance_type)
    """Generate demand scenario using PowerFlow-style formula"""
    
    # Set sigma range based on variance type
    if variance_type == "low"
        sigma = rand(Uniform(0.1, 0.2))
    elseif variance_type == "high"
        sigma = rand(Uniform(0.2, 0.3))
    elseif variance_type == "bernoulli"
        sigma = rand(Uniform(0.1, 0.2))
    else
        error("Invalid variance type. Use 'low', 'high', or 'bernoulli'")
    end
    
    # Generate global noise
    global_noise = randn()
    
    # Generate individual noise components and apply PowerFlow formula
    scenario_demand = zeros(nbus)
    for i in 1:nbus
        individual_noise = randn()
        
        # PowerFlow-style formula
        raw_demand = demand_mean[i] + sigma * individual_noise * demand_mean[i] + global_noise * sigma * demand_mean[i]
        
        # Apply Bernoulli multiplier if bernoulli case
        if variance_type == "bernoulli"
            bernoulli_multiplier = rand(Bernoulli(0.8)) == 1 ? 1.0 : 0.0
            raw_demand *= bernoulli_multiplier
        end
        
        # Ensure non-negativity
        scenario_demand[i] = max(0.0, raw_demand)
    end
    
    return scenario_demand
end

# =============================================================================
# COMPACT POWERFLOW FORMULATION
# =============================================================================

function powerflow_compact(K::Int, S::Int, variance_type::String; 
                          file_path::String="case118Blumsack.m",
                          demand_file::String="Random_Demands.xlsx",
                          worksheet_name::String="case118")
    
    println("="^80)
    println("COMPACT POWERFLOW FORMULATION")
    println("="^80)
    println("Instance: $(basename(file_path))")
    println("K (candidate solutions): $K")
    println("S (scenarios): $S")
    println("Variance type: $variance_type")
    
    # Load network using PowerModels (TS1 style)
    println("  Loading MATPOWER case via PowerModels...")
    network = load_network(file_path)
    B_ij, fbar_ij, p_max, p_min, A, c_i = parameters(network)
    
    nbus = length(network["bus"])
    nbranch = length(network["branch"])
    ngen = length(network["gen"])
    
    # Build fbus/tbus arrays for constraints
    fbus = Vector{Int}(undef, nbranch)
    tbus = Vector{Int}(undef, nbranch)
    for (branch_id, branch) in network["branch"]
        idx = parse(Int, branch_id)
        fbus[idx] = branch["f_bus"]
        tbus[idx] = branch["t_bus"]
    end
    
    # Big-M and penalties
    M_ij = -12.6 .* B_ij
    unmet = 5000
    rate_a = fbar_ij
    
    println("  Instance loaded: $nbus buses, $ngen generators, $nbranch branches")
    
    # Read demand data
    println("  Reading demand data from Excel...")
    demand_mean = load_demand_data(demand_file, worksheet_name)
    println("  Demand data loaded: $(length(demand_mean)) values")
    
    # Generate scenarios
    println("  Generating $S scenarios with variance type: $variance_type")
    scenarios = []
    for s in 1:S
        scenario_demand = generate_demand_scenario(nbus, demand_mean, variance_type)
        push!(scenarios, scenario_demand)
    end
    println("  Generated $(length(scenarios)) scenarios successfully")
    
    # Create model
    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 1)
    set_optimizer_attribute(model, "LogToConsole", 1)
    set_optimizer_attribute(model, "WorkLimit", 10000)
    
    # Set log file
    log_filename = "case118K$(K)S$(S)_$(variance_type).log"
    set_optimizer_attribute(model, "LogFile", log_filename)
    
    # Decision Variables
    @variable(model, p_g[1:nbus, 1:S] >= 0)  # Power generation
    @variable(model, f_ij[1:nbranch, 1:S])   # Power flow (free)
    @variable(model, theta[1:nbus, 1:S])     # Voltage angles (free)
    @variable(model, s_i[1:nbus, 1:S] >= 0)  # Load shedding
    @variable(model, x_ij[1:nbranch, 1:K], Bin)  # Line status in candidate solutions
    @variable(model, q_ij[1:nbranch, 1:S], Bin)  # Line status in scenarios
    @variable(model, w_sk[1:S, 1:K], Bin)    # Scenario assignment
    
    # Objective function
    @objective(model, Min, 
        (1/S) * sum(c_i[i] * p_g[i, s] for i in 1:nbus, s in 1:S) +
        (1/S) * 5000 * sum(s_i[i, s] for i in 1:nbus, s in 1:S)
    )
    
    # Constraints
    
    # 1. Power balance constraint (TS1 style: outgoing - incoming)
    for s in 1:S
        for i in 1:nbus
            @constraint(model,
                p_g[i, s] - scenarios[s][i] ==
                (sum(f_ij[j, s] for j in 1:nbranch if fbus[j] == i) -
                 sum(f_ij[j, s] for j in 1:nbranch if tbus[j] == i)) - s_i[i, s]
            )
        end
    end
    
    # 2. Generator limits
    @constraint(model, [i in 1:nbus, s in 1:S], p_g[i, s] >= p_min[i])
    @constraint(model, [i in 1:nbus, s in 1:S], p_g[i, s] <= p_max[i])
    
    # 3. Line flow limits
    @constraint(model, [j in 1:nbranch, s in 1:S],
        f_ij[j, s] >= -fbar_ij[j] * q_ij[j, s]
    )
    @constraint(model, [j in 1:nbranch, s in 1:S],
        f_ij[j, s] <= fbar_ij[j] * q_ij[j, s]
    )
    
    # 4. DC power flow approximation with big-M
    @constraint(model, [j in 1:nbranch, s in 1:S],
        f_ij[j, s] >= B_ij[j] * (theta[fbus[j], s] - theta[tbus[j], s]) - M_ij[j] * (1 - q_ij[j, s])
    )
    @constraint(model, [j in 1:nbranch, s in 1:S],
        f_ij[j, s] <= B_ij[j] * (theta[fbus[j], s] - theta[tbus[j], s]) + M_ij[j] * (1 - q_ij[j, s])
    )
    
    # 5. Scenario assignment constraint
    @constraint(model, [s in 1:S],
        sum(w_sk[s, k] for k in 1:K) == 1
    )
    
    # 6. Line status linking constraints
    @constraint(model, [j in 1:nbranch, k in 1:K, s in 1:S],
        x_ij[j, k] - q_ij[j, s] <= 1 - w_sk[s, k]
    )
    @constraint(model, [j in 1:nbranch, k in 1:K, s in 1:S],
        x_ij[j, k] - q_ij[j, s] >= w_sk[s, k] - 1
    )
    
    # 7. Reference bus constraint
    @constraint(model, [s in 1:S],
        theta[1, s] == 0
    )
    
    # Solve
    println("\nBuilding and solving compact PowerFlow formulation...")
    optimize!(model)
    
    # Results
    status = termination_status(model)
    
    if status == JuMP.MOI.INFEASIBLE || status == JuMP.MOI.INFEASIBLE_OR_UNBOUNDED
        println("Model is infeasible!")
        return model, Dict{String,Any}(), nothing, nothing, nothing, nothing, nothing
    end
    
    obj_val = objective_value(model)
    
    # Simple console output
    println("Results: S=$S, K=$K, variance=$variance_type, buses=$nbus, generators=$ngen, branches=$nbranch")
    println("Gap: See Gurobi log above")
    println("Work units: See Gurobi log above") 
    println("Nodes: See Gurobi log above")
    println("Detailed log saved to: $log_filename")
    
    # Return solution variables
    if primal_status(model) == JuMP.MOI.FEASIBLE_POINT
        p_g_sol = value.(model[:p_g])
        f_ij_sol = value.(model[:f_ij])
        theta_sol = value.(model[:theta])
        s_i_sol = value.(model[:s_i])
        x_ij_sol = value.(model[:x_ij])
        q_ij_sol = value.(model[:q_ij])
        w_sk_sol = value.(model[:w_sk])
        return model, Dict{String,Any}(), p_g_sol, f_ij_sol, theta_sol, s_i_sol, x_ij_sol, q_ij_sol, w_sk_sol
    else
        return model, Dict{String,Any}(), nothing, nothing, nothing, nothing, nothing, nothing, nothing
    end
end

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) != 3
        println("Usage: julia CompactTS1.jl <K> <S> <variance_type>")
        println("Example: julia CompactTS1.jl 2 20 low")
        exit(1)
    end
    
    K = parse(Int, ARGS[1])
    S = parse(Int, ARGS[2])
    variance_type = ARGS[3]
    
    # Run compact PowerFlow formulation
    powerflow_compact(K, S, variance_type)
end
