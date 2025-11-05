# PowerFlow Scenarios - Shared Scenario Generation
# This file generates demand scenarios for PowerFlow optimization problems

using XLSX, DataFrames, Random, Distributions

# Set random seed for reproducibility
Random.seed!(1234)

# =============================================================================
# GLOBAL VARIABLES FOR SHARED SCENARIOS
# =============================================================================

# Global variables to store shared data and scenarios
global demand_mean = Float64[]
global scenarios_dict = Dict{Int, Vector{Vector{Float64}}}()
global scenario_params_dict = Dict{Int, Dict{String, Any}}()
global scenario_counter = 0  # Counter for generating unique scenarios

# =============================================================================
# SCENARIO GENERATION FUNCTIONS
# =============================================================================

function load_demand_data(file_path_excel = "powerflow_instances/Random_Demands.xlsx", instance_file_path = "powerflow_instances/case300Blumsack.m")
    """Load demand mean values from Excel file"""
    if !isfile(file_path_excel)
        error("❌ Error: Excel file not found! Expected: $file_path_excel")
    end
    
    # Extract case name from instance file path (e.g., "case300Blumsack.m" -> "case300")
    case_name = basename(instance_file_path)
    case_name = replace(case_name, r"Blumsack\.m$" => "")
    case_name = replace(case_name, r"\.m$" => "")
    
    println("Loading demand data from: $file_path_excel (worksheet: $case_name)")
    
    # Read the Excel data containing demand parameters
    param_data = XLSX.readtable(file_path_excel, case_name) |> DataFrame
    
    # Extract demand mean from Excel file (first row contains the demand values)
    demand_mean_data = Vector{Float64}(param_data[1, :])
    
    println("✅ Demand data loaded successfully!")
    println("Number of buses: $(length(demand_mean_data))")
    
    return demand_mean_data
end

function generate_powerflow_scenarios(nscen::Int, demand_mean_data::Vector{Float64}, sigma_low::Float64=0.04, sigma_high::Float64=0.06, use_bernoulli::Bool=false)
    """Generate demand scenarios using the new formula:
    demand[node] = demand_mean[node] + sigma * N(0,1)[node] * demand_mean[node] + global_noise * sigma * demand_mean[node]
    
    For Bernoulli mode:
    demand[node] = Bernoulli(0.5) * (demand_mean[node] * 2 + sigma * N(0,1)[node] * demand_mean[node] * 2 + global_noise * sigma * demand_mean[node] * 2)
    
    Where:
    - sigma ~ Uniform(sigma_low, sigma_high) (common for all nodes)
    - N(0,1)[node] = independent normal random variable for each node
    - global_noise ~ N(0,1) (common for all nodes)
    - use_bernoulli: if true, apply Bernoulli(0.5) multiplication and double demand_mean
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
            println("   Initial scenario $scenario: sigma=$(round(sigma, digits=4)), global_noise=$(round(global_noise, digits=4))")
        end
        
        # Generate scenario demand for each node
        scenario_demand = Float64[]
        
        for node in 1:length(demand_mean_data)
            demand_mean_node = demand_mean_data[node]
            
            # Generate node-specific normal random variable
            node_noise = rand(Normal(0, 1))
            
            # Calculate base demand
            if use_bernoulli
                # For Bernoulli mode: apply formula without doubling
                base_demand = demand_mean_node + 
                             sigma * node_noise * demand_mean_node + 
                             global_noise * sigma * demand_mean_node
                
                # Apply Bernoulli(0.8): 20% chance of 0, 80% chance of base_demand
                demand_node = rand(Bernoulli(0.8)) * base_demand
            else
                # Standard formula
                demand_node = demand_mean_node + 
                             sigma * node_noise * demand_mean_node + 
                             global_noise * sigma * demand_mean_node
            end
            
            push!(scenario_demand, demand_node)
        end
        
        push!(scenarios_list, scenario_demand)
    end
    
    println("✅ Generated $nscen scenarios successfully!")
    return scenarios_list
end

function generate_scenarios_for_count(nscen::Int, file_path_excel = "powerflow_instances/Random_Demands.xlsx", sigma_low::Float64=0.04, sigma_high::Float64=0.06, use_bernoulli::Bool=false, instance_file_path = "powerflow_instances/case300Blumsack.m")
    """Generate scenarios for a specific count and store them globally"""
    global demand_mean, scenarios_dict, scenario_params_dict
    
    # Load demand data if not already loaded
    if isempty(demand_mean)
        demand_mean = load_demand_data(file_path_excel, instance_file_path)
    end
    
    # Generate scenarios for this count
    scenarios = generate_powerflow_scenarios(nscen, demand_mean, sigma_low, sigma_high, use_bernoulli)
    
    # Store scenarios and parameters for this count
    scenarios_dict[nscen] = scenarios
    scenario_params_dict[nscen] = Dict(
        "nscen" => nscen,
        "file_path" => file_path_excel,
        "random_seed" => 1234,
        "sigma_range" => (0.04, 0.06),
        "nbuses" => length(demand_mean)
    )
    
    println("✅ Generated and stored $nscen scenarios!")
    return scenarios
end

function generate_shared_powerflow_scenarios(nscen::Int, file_path_excel = "powerflow_instances/Random_Demands.xlsx")
    """Generate and store scenarios globally for sharing across files (legacy function)"""
    return generate_scenarios_for_count(nscen, file_path_excel)
end

function get_shared_powerflow_scenarios(nscen::Int)
    """Get the globally stored scenarios for a specific count"""
    global scenarios_dict
    if !haskey(scenarios_dict, nscen)
        error("❌ No scenarios found for nscen=$nscen. Run generate_scenarios_for_count($nscen) first.")
    end
    return scenarios_dict[nscen]
end

function get_shared_powerflow_scenario_params(nscen::Int)
    """Get the globally stored scenario parameters for a specific count"""
    global scenario_params_dict
    if !haskey(scenario_params_dict, nscen)
        error("❌ No scenario parameters found for nscen=$nscen. Run generate_scenarios_for_count($nscen) first.")
    end
    return scenario_params_dict[nscen]
end

function get_available_scenario_counts()
    """Get list of available scenario counts"""
    global scenarios_dict
    return collect(keys(scenarios_dict))
end

function add_scenario_to_existing_set(nscen::Int, file_path_excel = "powerflow_instances/Random_Demands.xlsx", sigma_low::Float64=0.04, sigma_high::Float64=0.06, use_bernoulli::Bool=false)
    """Add one more scenario to an existing scenario set"""
    global demand_mean, scenarios_dict, scenario_params_dict
    
    # Check if the scenario set exists
    if !haskey(scenarios_dict, nscen)
        error("❌ No existing scenario set found for nscen=$nscen. Run generate_scenarios_for_count($nscen) first.")
    end
    
    # Load demand data if not already loaded
    if isempty(demand_mean)
        demand_mean = load_demand_data(file_path_excel)
    end
    
    # Generate one new scenario (now truly random since no seeds are set)
    global scenario_counter
    scenario_counter += 1
    
    println("🔄 Generating new scenario #$scenario_counter...")
    
    # Test if rand() is actually producing different values
    test_rand1 = rand()
    test_rand2 = rand()
    println("   Random test: $test_rand1, $test_rand2")
    
    new_scenario = generate_powerflow_scenarios(1, demand_mean, sigma_low, sigma_high, use_bernoulli)[1]
    println("   New scenario first 3 values: [$(round(new_scenario[1], digits=3)), $(round(new_scenario[2], digits=3)), $(round(new_scenario[3], digits=3))]")
    
    # Add the new scenario to the existing set
    push!(scenarios_dict[nscen], new_scenario)
    
    # Update the count in parameters
    scenario_params_dict[nscen]["nscen"] = length(scenarios_dict[nscen])
    
    println("✅ Added 1 new scenario to set $nscen (now has $(length(scenarios_dict[nscen])) scenarios)")
    return new_scenario
end

function get_shared_demand_mean()
    """Get the globally stored demand mean values"""
    global demand_mean
    if isempty(demand_mean)
        error("❌ No demand mean found! Run generate_shared_powerflow_scenarios() first.")
    end
    return demand_mean
end

# =============================================================================
# MAIN EXECUTION BLOCK
# =============================================================================

# Generate scenarios when this file is run directly
if abspath(PROGRAM_FILE) == @__FILE__
    println("="^60)
    println("POWERFLOW SCENARIOS GENERATION")
    println("="^60)
    
    # Parameters
    nscen_test = 10  # Number of scenarios to generate
    
    # Generate shared scenarios
    scenarios = generate_scenarios_for_count(nscen_test, "powerflow_instances/Random_Demands.xlsx")
    
    # Display summary
    scenario_params = get_shared_powerflow_scenario_params(nscen_test)
    println("\n" * "="^40)
    println("SCENARIO GENERATION SUMMARY")
    println("="^40)
    for (key, value) in scenario_params
        println("$key: $value")
    end
    
    
    println("\n✅ PowerFlow scenarios ready for use!")
end
