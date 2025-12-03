# TSUtilities.jl
# Utility functions for Transmission Switching - used in data generation phase
#
# This mirrors the role of FLUtilities.jl for Facility Location.
#
# Contents:
#   - extract_case_name: parse MATPOWER case name from instance file
#   - load_demand_data: load demand means from Random_Demands.xlsx
#   - generate_powerflow_scenarios: generate demand scenarios
#   - load_scenarios_from_file: generic text loader
#   - load_binary_vectors_from_file: generic text loader for 0/1 vectors
#   - parameters_TSpm: extract TS parameters from PowerModels data
#   - evaluate_ts_operational_cost: evaluate r_s(x) for TS (analogous to evaluate_cost_FL)

using PowerModels
using JuMP, Gurobi
using MathOptInterface
const MOI = MathOptInterface
using DataFrames, XLSX
using Random, Distributions

# =============================================================================
# CASE / DEMAND UTILITIES
# =============================================================================

function extract_case_name(instance_file::String)
    """Extract case name from instance file (e.g., 'case118' from 'case118Blumsack.m').
    
    This function is robust to receiving either a bare filename (\"case57.m\")
    or a full/relative path (\"../../Transmission_Switching/instances/case57.m\").
    """
    # Work with just the filename component if a path is provided
    file_name = splitpath(instance_file)[end]
    # Remove extension
    base_name = replace(file_name, r"\.m$" => "")
    # Extract case number (e.g., 'case118' from 'case118Blumsack')
    match_result = match(r"^(case\d+)", base_name)
    if match_result !== nothing
        return String(match_result.captures[1])  # Convert SubString to String
    else
        error("Could not extract case name from instance file: $instance_file")
    end
end

function load_demand_data(file_path_excel::String, case_name::String; instance_file::Union{String, Nothing}=nothing)
    """Load demand mean values from Excel file (Random_Demands.xlsx).
    
    If the Excel worksheet doesn't exist, falls back to extracting demand from MATPOWER file.
    
    Args:
        file_path_excel: Path to Random_Demands.xlsx
        case_name: Worksheet name (e.g., "case57")
        instance_file: Optional path to MATPOWER file for fallback
    """
    # Try to load from Excel first
    try
        if !isfile(file_path_excel)
            error("Excel file not found")
        end
        
        println("Loading demand data from: $file_path_excel (worksheet: $case_name)")
        
        # Read the Excel data containing demand parameters
        param_data = XLSX.readtable(file_path_excel, case_name) |> DataFrame
        
        # Extract demand mean from Excel file (first row contains the demand values)
        demand_mean_data = Vector{Float64}(param_data[1, :])
        
        println("✅ Demand data loaded successfully from Excel!")
        println("Number of buses: $(length(demand_mean_data))")
        
        return demand_mean_data
    catch e
        # Fallback: Extract from MATPOWER file
        if instance_file === nothing || !isfile(instance_file)
            error("❌ Error: Could not load demand data from Excel (worksheet '$case_name' not found) and no valid MATPOWER file provided for fallback.\nExcel error: $e")
        end
        
        println("⚠️  Excel worksheet '$case_name' not found. Falling back to MATPOWER file: $instance_file")
        
        # Load MATPOWER file
        data = PowerModels.parse_file(instance_file)
        B = data["bus"]
        L = data["load"]
        nb = length(B)
        
        # Extract pd (real power demand) from loads and aggregate by bus
        # PowerModels stores loads separately, each load has a "load_bus" field
        demand_mean_data = zeros(Float64, nb)
        for (load_id, load) in L
            if haskey(load, "load_bus") && haskey(load, "pd")
                bus_idx = load["load_bus"]
                if 1 <= bus_idx <= nb
                    demand_mean_data[bus_idx] += load["pd"]
                end
            end
        end
        
        println("✅ Demand data extracted from MATPOWER file!")
        println("Number of buses: $(length(demand_mean_data))")
        println("Total demand: $(round(sum(demand_mean_data), digits=2))")
        println("   (Aggregated from $(length(L)) loads)")
        
        return demand_mean_data
    end
end

function generate_powerflow_scenarios(nscen::Int, demand_mean_data::Vector{Float64}, 
                                      sigma_low::Float64=0.04, sigma_high::Float64=0.06, 
                                      use_bernoulli::Bool=false)
    """Generate demand scenarios using the PowerFlow-style formula.
    
    demand[node] = demand_mean[node] + sigma * N(0,1)[node] * demand_mean[node] + global_noise * sigma * demand_mean[node]
    
    For Bernoulli mode:
    demand[node] = Bernoulli(0.8) * (same expression as above)
    """
    bernoulli_str = use_bernoulli ? " (Bernoulli mode)" : ""
    println("Generating $nscen PowerFlow scenarios with sigma range [$sigma_low, $sigma_high]$bernoulli_str...")
    
    scenarios_list = Vector{Vector{Float64}}()
    
    for scenario in 1:nscen
        # Generate sigma (common for all nodes in this scenario)
        sigma = rand(Uniform(sigma_low, sigma_high))
        
        # Generate global noise (common for all nodes in this scenario)
        global_noise = rand(Normal(0, 1))
        
        if scenario <= 3  # Debug first 3 scenarios
            println("   Scenario $scenario: sigma=$(round(sigma, digits=4)), global_noise=$(round(global_noise, digits=4))")
        end
        
        # Generate scenario demand for each node
        scenario_demand = Float64[]
        
        for node in 1:length(demand_mean_data)
            demand_mean_node = demand_mean_data[node]
            
            # Generate node-specific normal random variable
            node_noise = rand(Normal(0, 1))
            
            # Calculate base demand
            if use_bernoulli
                base_demand = demand_mean_node + 
                              sigma * node_noise * demand_mean_node + 
                              global_noise * sigma * demand_mean_node
                # Apply Bernoulli(0.8): 20% chance of 0, 80% chance of base_demand
                demand_node = rand(Bernoulli(0.8)) * base_demand
            else
                demand_node = demand_mean_node + 
                              sigma * node_noise * demand_mean_node + 
                              global_noise * sigma * demand_mean_node
            end
            
            push!(scenario_demand, max(0.0, demand_node))  # Ensure non-negativity
        end
        
        push!(scenarios_list, scenario_demand)
    end
    
    println("✅ Generated $nscen scenarios successfully!")
    return scenarios_list
end

# =============================================================================
# GENERIC LOADERS (SCENARIOS / BINARY VECTORS)
# =============================================================================

function load_scenarios_from_file(file_path::String)
    """Load scenarios from a file (one scenario per line, space-separated values)."""
    if !isfile(file_path)
        error("❌ Error: Scenarios file not found! Expected: $file_path")
    end
    
    println("Loading scenarios from: $file_path")
    scenarios = Vector{Vector{Float64}}()
    
    open(file_path, "r") do file
        for (line_num, line) in enumerate(eachline(file))
            line = strip(line)
            if length(line) > 0
                try
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

function load_binary_vectors_from_file(file_path::String)
    """Load binary vectors from a file (one vector per line, space-separated 0/1)."""
    if !isfile(file_path)
        error("❌ Error: Binary vectors file not found! Expected: $file_path")
    end
    
    println("Loading binary vectors from: $file_path")
    binary_vectors = Vector{Vector{Int}}()
    
    open(file_path, "r") do file
        for (line_num, line) in enumerate(eachline(file))
            line = strip(line)
            if length(line) > 0
                try
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
# TS PARAMETERS AND COST EVALUATION (ANALOGOUS TO evaluate_cost_FL)
# =============================================================================

function parameters_TSpm(data; line_capacity_factor::Float64=1.0)
    """Extract parameters from PowerModels data structure (TS).
    
    Returns:
        B_ij, fbar_ij, p_max, p_min, A, c
    """
    B = data["bus"]
    L = data["branch"]
    G = data["gen"]

    # Parameters
    B_ij   = zeros(length(L))      # susceptance per line
    fbar_ij = zeros(length(L))     # thermal limit per line
    p_max  = zeros(length(B))      # max generation per bus
    p_min  = zeros(length(B))      # min generation per bus
    c      = zeros(length(B))      # linear gen cost per bus

    for (branch_id, branch) in L
        B_ij[parse(Int, branch_id)] = -branch["br_x"]/(branch["br_r"]^2 + branch["br_x"]^2)
        rate_a = haskey(branch, "rate_a") ? branch["rate_a"] : 2.2
        fbar_ij[parse(Int, branch_id)] = rate_a * line_capacity_factor
    end

    for (_gen_id, generator) in G
        c[generator["gen_bus"]] = generator["cost"][1]
        p_max[generator["gen_bus"]] = generator["pmax"]
        p_min[generator["gen_bus"]] = generator["pmin"]
    end

    n = 2*length(B)
    m = length(L)
    A = zeros(n, m)

    # Create incidence matrix A
    for (branch_id, branch) in L
        f_bus = branch["f_bus"]
        t_bus = branch["t_bus"]
        branch_index = parse(Int, branch_id)
        A[2*f_bus-1, branch_index] = 1
        A[2*f_bus,   branch_index] = -1
        A[2*t_bus-1, branch_index] = -1
        A[2*t_bus,   branch_index] = 1
    end

    return B_ij, fbar_ij, p_max, p_min, A, c
end

function evaluate_ts_operational_cost(binary_vector::Vector{Int}, scenario::Vector{Float64}, problem_data::Dict)
    """Evaluate TS operational cost r_s(x) for a fixed switch plan.
    
    This is analogous to `evaluate_cost_FL` for Facility Location.
    
    Args:
        binary_vector: Binary line status vector (0/1 for each line)
        scenario: Demand vector for this scenario
        problem_data: PowerModels data dictionary (network data)
    
    Returns:
        (cost::Float64, work_units::Float64)
    """
    B = problem_data["bus"]
    L = problem_data["branch"]
    nb = length(B)
    nl = length(L)
    B_ij, fbar_ij, p_max, p_min, A, c_g = parameters_TSpm(problem_data)
    ref = 1
    penalty = 5000.0

    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 0)

    # Variables
    @variable(model, p_g[1:nb])
    @variable(model, f[1:nl])
    @variable(model, theta[1:nb])
    @variable(model, unmet[1:nb] >= 0)

    # Generator bounds per bus
    for i in 1:nb
        @constraint(model, p_g[i] >= p_min[i])
        @constraint(model, p_g[i] <= p_max[i])
    end

    # Line capacity + DC flow with fixed x (binary_vector)
    for (branch_id, branch) in L
        l = parse(Int, branch_id)
        xl = binary_vector[l]
        if xl == 1
            @constraint(model, -fbar_ij[l] <= f[l])
            @constraint(model,  f[l] <= fbar_ij[l])
            i = branch["f_bus"]
            j = branch["t_bus"]
            @constraint(model, f[l] == B_ij[l] * (theta[i] - theta[j]))
        else
            @constraint(model, f[l] == 0)
        end
    end

    # Power balance at each bus
    for i in 1:nb
        outgoing = 0.0
        incoming = 0.0
        for (branch_id, branch) in L
            l = parse(Int, branch_id)
            if branch["f_bus"] == i
                outgoing += f[l]
            end
            if branch["t_bus"] == i
                incoming += f[l]
            end
        end
        @constraint(model, p_g[i] - scenario[i] + unmet[i] == outgoing - incoming)
    end

    # Reference angle
    @constraint(model, theta[ref] == 0)

    # Objective
    @objective(model, Min, sum(c_g[i] * p_g[i] for i in 1:nb) + penalty * sum(unmet[i] for i in 1:nb))

    optimize!(model)

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

