# Simple debug script for offline phase infeasibility
using PowerModels
using JuMP, Gurobi, DataFrames, XLSX, Random, Distributions
using Graphs, GraphPlot, Plots

# Include shared scenarios
include("powerflow_scenarios.jl")

# Include the direct implementation functions
include("complete_powerflow_direct.jl")

println("================================================================================
SIMPLE DEBUGGING OFFLINE PHASE INFEASIBILITY
================================================================================")

# Load network data
network = PowerModels.parse_file("powerflow_instances/case118Blumsack.m")
println("✅ Network data loaded successfully!")

# Generate a few scenarios to test
nscen = 3
println("Generating $nscen scenarios for debugging...")

# First generate scenarios
generate_scenarios_for_count(nscen, "powerflow_instances/Random_Demands.xlsx", 0.04, 0.06, false)

# Get shared scenarios
shared_scenarios = get_shared_powerflow_scenarios(nscen)
println("✅ Generated $nscen scenarios!")

# Test each scenario individually
for i in 1:nscen
    println("\n========================================
TESTING SCENARIO $i
========================================")
    
    scenario = shared_scenarios[i]  # shared_scenarios is a Vector{Vector{Float64}}, not a matrix
    println("Scenario $i demand range: [$(minimum(scenario)), $(maximum(scenario))]")
    println("Scenario $i total demand: $(sum(scenario))")
    
    # Try to solve this scenario
    try
        obj_val, binary_sol, penalty_flag, penalty_cost, work_units = offline_calc_direct(network, scenario)
        
        if obj_val == -1
            println("❌ Scenario $i is INFEASIBLE!")
        else
            println("✅ Scenario $i solved successfully!")
            println("   Objective value: $obj_val")
            println("   Penalty cost: $penalty_cost")
            println("   Work units: $work_units")
        end
    catch e
        println("❌ Scenario $i failed with error: $e")
    end
end

println("\n================================================================================
DEBUGGING COMPLETE
================================================================================")
