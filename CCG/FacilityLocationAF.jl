# FacilityLocationAF.jl
# Problem-specific implementations for Facility Location - Assignment Formulation (AF)
# This file provides the interface functions required by ColumnGenerationCore.jl for AF

using JuMP, Gurobi
using MathOptInterface
const MOI = MathOptInterface
using Printf
using Statistics
using Random, Distributions
using Dates

# Include shared utilities
include("ColumnGenerationCore.jl")
include("AssignmentFormulation.jl")

# =============================================================================
# CONSTANTS
# =============================================================================

const DEFAULT_OPT_WORKLIMIT = 300.0

# =============================================================================
# F_S EXTRACTION FROM WS FILES
# =============================================================================

function extract_f_s_from_ws_file(ws_file::String, S::Int)
    """Extract f_s values (wait-and-see costs) from WS file
    
    Args:
        ws_file: Path to WS file (e.g., Samples/FL/Scenarios/instance_1/WS300_bernoulli.txt)
        S: Number of scenarios to extract (first S scenarios)
    
    Returns:
        f_s: Vector of f_s values (wait-and-see costs) for each scenario
    """
    if !isfile(ws_file)
        error("❌ Error: WS file not found! Expected: $ws_file")
    end
    
    f_s = Float64[]
    
    open(ws_file, "r") do file
        for line in eachline(file)
            line = strip(line)
            # Skip comments and empty lines
            if startswith(line, "#") || isempty(line)
                continue
            end
            
            # Parse the wait-and-see cost (first number on the line)
            try
                cost = parse(Float64, line)
                push!(f_s, cost)
            catch
                # If parsing fails, try to extract number from line (handle formats like "cost(gap%)")
                # For FL, the format is simpler - just a number per line
                # But let's be robust and extract first number
                match_result = match(r"([\d.]+)", line)
                if match_result !== nothing
                    cost = parse(Float64, match_result.captures[1])
                    push!(f_s, cost)
                else
                    error("❌ Error: Could not parse wait-and-see cost from line: $line")
                end
            end
            end
        end

    # Validate we have at least S values
    if length(f_s) < S
        error("❌ Error: WS file contains only $(length(f_s)) scenarios, but S = $S was requested.")
    end
    
    # Return first S values
    return f_s[1:S]
end

# =============================================================================
# DATA LOADING
# =============================================================================

# Import read_orlib_cap from Samples/FL/FLUtilities.jl (data generation utilities)
# This keeps CCG dependent on Samples, which is correct since we generate data first
include("../Samples/FL/FLUtilities.jl")

function load_FL_data(instance_file::String; capacity_factor::Float64=1.0)
    """Load Facility Location instance data
    
    Returns:
        Dict with keys: capacity, fixedcost, cost, demandmean, nloc, ncust
    """
    if !isfile(instance_file)
        error("❌ Error: Instance file not found! Expected: $instance_file")
    end
    
    capacity, fixedcost, cost, demandmean = read_orlib_cap(instance_file)
    
    # Apply capacity factor
    capacity = capacity .* capacity_factor
    
    return Dict(
        "capacity" => capacity,
        "fixedcost" => fixedcost,
        "cost" => cost,
        "demandmean" => demandmean,
        "nloc" => length(capacity),
        "ncust" => length(demandmean)
    )
end

# =============================================================================
# COST EVALUATION r_s(x)
# =============================================================================

# evaluate_cost_FL is now defined in Samples/FL/FLUtilities.jl
# It is imported via the include statement above

# =============================================================================
# M VALUES CALCULATION (for Pricing Type 1)
# =============================================================================

function calculate_m_values_FL(scenarios::Vector{Vector{Float64}}, dual_lambda::Vector{Float64}, problem_data::Dict)
    """Calculate m^s values for all scenarios (for Pricing Type 1)
    
    Args:
        scenarios: List of S scenarios
        dual_lambda: Dual values from assignment constraints
        problem_data: Dict containing problem data
    
    Returns:
        m_values: Vector of m^s values (one per scenario)
    """
    nloc = problem_data["nloc"]
    ncust = problem_data["ncust"]
    capacity = problem_data["capacity"]
    cost = problem_data["cost"]
    fixedcost = problem_data["fixedcost"]
    unmet_pen = problem_data["unmet_pen"]
    scaling_factor = problem_data["scaling_factor"]
    
    nscen = length(scenarios)
    m_values = zeros(nscen)
    
    for s in 1:nscen
        m_values[s] = big_m_calculator_FL(nloc, ncust, capacity, cost, scenarios[s], fixedcost, unmet_pen, scaling_factor, dual_lambda[s])
    end
    
    return m_values
end

function big_m_calculator_FL(nloc, ncust, capacity, cost, scenario_demand, fixedcost, unmet_pen, scaling_factor, dual_lambda)
    """Calculate m^s by solving Formulation 2 for a single scenario"""
    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 0)
    set_optimizer_attribute(model, "LogToConsole", 0)
    set_optimizer_attribute(model, "MIPGap", 0.01)
    set_optimizer_attribute(model, "WorkLimit", 1800)
    
    # Decision Variables 
    @variable(model, x[1:nloc], Bin)
    @variable(model, y[1:nloc, 1:ncust] >= 0)
    @variable(model, s_j[1:ncust] >= 0)
    
    # Constraints 
    @constraint(model, [j in 1:ncust], sum(y[i,j] for i in 1:nloc) + s_j[j] == scenario_demand[j])
    @constraint(model, [i in 1:nloc], sum(y[i,j] for j in 1:ncust) - capacity[i]*x[i] <= 0)
    
    # Objective function
    @objective(model, Min, 
        dual_lambda - sum(fixedcost[i]*x[i] for i in 1:nloc) - 
        scaling_factor * sum(cost[i,j]*y[i,j] for i in 1:nloc for j in 1:ncust) - 
        sum(unmet_pen*s_j[j] for j in 1:ncust))
    
    optimize!(model)
    
    if termination_status(model) == MOI.OPTIMAL
        return objective_value(model)
    else
        return Inf
    end
end

# =============================================================================
# PRICING SUBPROBLEM (Type 1)
# =============================================================================

function af_pricing_type1_FL(nloc, ncust, capacity, cost, scenarios, m_values, dual_lambda, dual_mu, unmet_pen, scaling_factor, fixedcost; use_solution_pool::Bool=true, return_nodes::Bool=false, work_limit::Float64=Inf, analysis::Dict{Symbol,Any}=Dict{Symbol,Any}(), log_file=nothing, iteration=nothing)
    """Solve AF Pricing Type 1 for Facility Location"""
    nscen = length(scenarios)
    
    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 1)
    set_optimizer_attribute(model, "LogToConsole", 1)
    set_optimizer_attribute(model, "MIPGap", 0.01)
    set_optimizer_attribute(model, "WorkLimit", 300)
    if use_solution_pool
        set_optimizer_attribute(model, "PoolSearchMode", 2)
        set_optimizer_attribute(model, "PoolSolutions", 10)
    end
    
    # Set up log file with iteration marker
    if log_file !== nothing
        if iteration !== nothing
            open(log_file, "a") do file
                println(file, "\n" * "="^80)
                println(file, "PRICING SUBPROBLEM TYPE 1 - ITERATION $iteration")
                println(file, "="^80)
                println(file, "Timestamp: $(now())")
                println(file, "Problem size: $(length(scenarios)) scenarios, $nloc facilities, $ncust customers")
                println(file, "="^80)
            end
        end
        set_optimizer_attribute(model, "LogFile", log_file)
    end
    
    # Decision Variables 
    @variable(model, v[1:nloc, 1:nscen], Bin)
    @variable(model, pi[1:nscen], Bin)
    @variable(model, x[1:nloc], Bin)
    @variable(model, u[1:nloc, 1:ncust, 1:nscen] >= 0)
    @variable(model, y[1:nloc, 1:ncust, 1:nscen] >= 0)
    @variable(model, w[1:ncust, 1:nscen] >= 0)
    @variable(model, b[1:ncust, 1:nscen] >= 0)
    
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
    @constraint(model, [j in 1:ncust, s in 1:nscen], 
        sum(y[i,j,s] for i in 1:nloc) + b[j,s] == scenarios[s][j])
    @constraint(model, [i in 1:nloc, s in 1:nscen], 
        sum(y[i,j,s] for j in 1:ncust) - capacity[i] * x[i] <= 0)
    @constraint(model, [s in 1:nscen], 
        dual_lambda[s] - sum(fixedcost[i] * x[i] for i in 1:nloc) - 
        scaling_factor * sum(cost[i,j] * y[i,j,s] for i in 1:nloc for j in 1:ncust) - 
        sum(unmet_pen * b[j,s] for j in 1:ncust) >= m_values[s] * (1 - pi[s]))
    @constraint(model, [i in 1:nloc, s in 1:nscen], v[i,s] <= x[i])
    @constraint(model, [i in 1:nloc, s in 1:nscen], v[i,s] <= pi[s])
    @constraint(model, [i in 1:nloc, s in 1:nscen], v[i,s] >= x[i] + pi[s] - 1)
    @constraint(model, [i in 1:nloc, j in 1:ncust, s in 1:nscen], u[i,j,s] <= y[i,j,s])
    @constraint(model, [i in 1:nloc, j in 1:ncust, s in 1:nscen], u[i,j,s] <= N[i,j,s] * pi[s])
    @constraint(model, [i in 1:nloc, j in 1:ncust, s in 1:nscen], 
        u[i,j,s] >= y[i,j,s] + N[i,j,s] * (pi[s] - 1))
    @constraint(model, [j in 1:ncust, s in 1:nscen], w[j,s] <= b[j,s])
    @constraint(model, [j in 1:ncust, s in 1:nscen], w[j,s] <= scenarios[s][j] * pi[s])
    @constraint(model, [j in 1:ncust, s in 1:nscen], 
        w[j,s] >= b[j,s] + scenarios[s][j] * (pi[s] - 1))
    
    # Solve
    optimize!(model)

    # Get work units
    pricing_work_units = 0.0
    try
        pricing_work_units = MOI.get(backend(model), Gurobi.ModelAttribute("Work"))
    catch
    end
    
    ts = termination_status(model)
    if analysis !== nothing
        analysis[:phase1_optimal] = ts == MOI.OPTIMAL
        analysis[:went_phase2] = false
    end
    if !(ts == MOI.OPTIMAL || ts == MOI.TIME_LIMIT || ts == MOI.OTHER_ERROR)
        return return_pricing_payload([zeros(nloc)], [Inf], pricing_work_units, return_nodes, model; analysis=analysis)
    end

    if !use_solution_pool
        # Add summary to log file
        if log_file !== nothing && iteration !== nothing
            open(log_file, "a") do file
                println(file, "\nPricing Results (no solution pool):")
                println(file, "  Objective: $(round(objective_value(model), digits=2))")
                println(file, "  Work units: $(round(pricing_work_units, digits=2))")
                println(file, "="^80)
            end
        end
        return return_pricing_payload([value.(x)], [objective_value(model)], pricing_work_units, return_nodes, model; analysis=analysis)
    end

    # Retrieve solution pool
    opt = backend(model)
    kept_solutions = Vector{Vector{Float64}}()
    kept_objs = Float64[]
    try
        solcount = MOI.get(opt, Gurobi.ModelAttribute("SolCount"))
        max_solutions = min(10, solcount)
        
        for k in 0:max_solutions-1
            MOI.set(opt, MOI.RawOptimizerAttribute("SolutionNumber"), k)
            pool_obj = MOI.get(opt, Gurobi.ModelAttribute("PoolObjVal"))
            x_k = [MOI.get(opt, Gurobi.VariableAttribute("Xn"), JuMP.index(x[i])) for i in 1:nloc]
            rc_k = pool_obj + dual_mu
            if rc_k > 0
                push!(kept_solutions, x_k)
                push!(kept_objs, pool_obj)
            end
        end
    catch e
        println("  Warning: solution pool access failed: $e")
    end

    nodes = 0
    if return_nodes
        try
            nodes = MOI.get(backend(model), Gurobi.ModelAttribute("NodeCount"))
        catch
        end
    end
    
    if isempty(kept_solutions)
        # Add summary to log file
        if log_file !== nothing && iteration !== nothing
            open(log_file, "a") do file
                println(file, "\nPricing Results:")
                println(file, "  Objective: $(round(objective_value(model), digits=2))")
                println(file, "  RC-improving solutions: 0 (using best solution)")
                println(file, "  Work units: $(round(pricing_work_units, digits=2))")
                println(file, "="^80)
            end
        end
        return return_pricing_payload([value.(x)], [objective_value(model)], pricing_work_units, return_nodes, model; analysis=analysis)
    end
    
    # Add summary to log file
    if log_file !== nothing && iteration !== nothing
        open(log_file, "a") do file
            println(file, "\nPricing Results:")
            println(file, "  RC-improving solutions found: $(length(kept_solutions))")
            println(file, "  Best objective: $(round(kept_objs[1], digits=2))")
            println(file, "  Work units: $(round(pricing_work_units, digits=2))")
            println(file, "="^80)
        end
    end
    
    return return_pricing_payload(kept_solutions, kept_objs, pricing_work_units, return_nodes, model; analysis=analysis)
end

# =============================================================================
# PRICING SUBPROBLEM (Type 2)
# =============================================================================

function af_pricing_type2_FL(nloc, ncust, capacity, cost, scenarios, m_values, dual_lambda, dual_mu, unmet_pen, scaling_factor, fixedcost; use_solution_pool::Bool=true, return_nodes::Bool=false, work_limit::Float64=Inf, opt_worklimit::Float64=DEFAULT_OPT_WORKLIMIT, analysis::Dict{Symbol,Any}=Dict{Symbol,Any}(), log_file=nothing, iteration=nothing, pricing_cache::Union{Dict,Nothing}=nothing)
    """Solve AF Pricing Type 2 for Facility Location using two-phase approach"""
    nscen = length(scenarios)

    # -------------------------------------------------------------------------
    # Model reuse: build once, update objective coefficients on subsequent calls
    # -------------------------------------------------------------------------
    model_cached = pricing_cache !== nothing && haskey(pricing_cache, "type2_model")

    if model_cached
        model   = pricing_cache["type2_model"]
        pi_vars = pricing_cache["type2_pi_vars"]
        x_vars  = pricing_cache["type2_x_vars"]
        # Update only the dual_lambda coefficients on pi[s] — everything else is unchanged
        for s in 1:nscen
            set_objective_coefficient(model, pi_vars[s], dual_lambda[s])
        end
        println("  [model reuse] Updated objective coefficients for $(nscen) scenarios")
    else
        model = Model(Gurobi.Optimizer)
        set_optimizer_attribute(model, "OutputFlag", 1)
        set_optimizer_attribute(model, "LogToConsole", 1)
        if use_solution_pool
            set_optimizer_attribute(model, "PoolSearchMode", 2)
            set_optimizer_attribute(model, "PoolSolutions", 10)
        end

        # Decision Variables
        @variable(model, x[1:nloc], Bin)
        @variable(model, 0 <= x_s[1:nloc, 1:nscen] <= 1)
        @variable(model, y[1:nloc, 1:ncust, 1:nscen] >= 0)
        @variable(model, b[1:ncust, 1:nscen] >= 0)
        @variable(model, pi[1:nscen], Bin)

        # Objective Function (MAXIMIZATION)
        @objective(model, Max,
            sum(dual_lambda[s] * pi[s] for s in 1:nscen) -
            sum(fixedcost[i] * x_s[i,s] for i in 1:nloc for s in 1:nscen) -
            scaling_factor * sum(cost[i,j] * y[i,j,s] for i in 1:nloc for j in 1:ncust for s in 1:nscen) -
            sum(unmet_pen * b[j,s] for j in 1:ncust for s in 1:nscen))

        # Constraints
        @constraint(model, [i in 1:nloc, s in 1:nscen], x_s[i,s] <= pi[s])
        @constraint(model, [i in 1:nloc, s in 1:nscen], x[i] - x_s[i,s] <= (1 - pi[s]))
        @constraint(model, [i in 1:nloc, s in 1:nscen], x[i] - x_s[i,s] >= -(1 - pi[s]))
        @constraint(model, [i in 1:nloc, s in 1:nscen],
            sum(y[i,j,s] for j in 1:ncust) - capacity[i] * x_s[i,s] <= 0)
        @constraint(model, [j in 1:ncust, s in 1:nscen],
            sum(y[i,j,s] for i in 1:nloc) + b[j,s] == scenarios[s][j] * pi[s])

        pi_vars = pi
        x_vars  = x

        # Cache model and variable references for reuse
        if pricing_cache !== nothing
            pricing_cache["type2_model"]   = model
            pricing_cache["type2_pi_vars"] = pi_vars
            pricing_cache["type2_x_vars"]  = x_vars
            println("  [model reuse] Model built and cached (first call)")
        end
    end

    # Set up log file with iteration marker (always, reused or new)
    if log_file !== nothing
        if iteration !== nothing
            open(log_file, "a") do file
                println(file, "\n" * "="^80)
                println(file, "PRICING SUBPROBLEM TYPE 2 (STANDARD) - ITERATION $iteration")
                println(file, "="^80)
                println(file, "Timestamp: $(now())")
                println(file, "Problem size: $(length(scenarios)) scenarios, $nloc facilities, $ncust customers")
                println(file, model_cached ? "Model reused (objective updated)" : "Model built fresh")
                println(file, "="^80)
            end
        end
        set_optimizer_attribute(model, "LogFile", log_file)
    end

    # Two-phase solve
    pricing_work_units = 0.0
    if analysis !== nothing
        analysis[:phase1_work] = 0.0
        analysis[:phase2_work] = 0.0
        analysis[:went_phase2] = false
        analysis[:lb] = NaN
        analysis[:ub] = NaN
    end

    # Baseline Work reading before any solve this call — handles model reuse across CG iterations
    last_work = 0.0
    try
        last_work = MOI.get(backend(model), Gurobi.ModelAttribute("Work"))
    catch
    end

    # Phase 1: Quick optimal try
    if isfinite(opt_worklimit) && opt_worklimit > 0.0
        println("  Phase 1: Quick optimal try with $(opt_worklimit) work units")
        set_optimizer_attribute(model, "Heuristics", 1)
        set_optimizer_attribute(model, "WorkLimit", opt_worklimit)
        optimize!(model)

        try
            w = MOI.get(backend(model), Gurobi.ModelAttribute("Work"))
            phase1_work = max(0.0, w - last_work)
            pricing_work_units += phase1_work
            last_work = w
            if analysis !== nothing
                analysis[:phase1_work] = phase1_work
            end
        catch
        end
        
        if analysis !== nothing
            analysis[:phase1_optimal] = termination_status(model) == MOI.OPTIMAL
        end
        if analysis !== nothing
            analysis[:phase2_optimal] = termination_status(model) == MOI.OPTIMAL
        end
        if termination_status(model) == MOI.OPTIMAL
            println("  Phase 1: Optimal solution found")
            # Extract only the optimal solution (not from pool)
            opt_obj = objective_value(model)
            opt_solution = value.(x_vars)
            rc_opt = opt_obj + dual_mu
            
            if rc_opt > 1.0  # Positive reduced cost (improving)
                println("  Phase 1: Optimal solution has positive reduced cost (rc = $(round(rc_opt, digits=4)))")
                kept_solutions = [opt_solution]
                kept_objs = [opt_obj]
                # Add summary to log file
                if log_file !== nothing && iteration !== nothing
                    open(log_file, "a") do file
                        println(file, "\nPhase 1 Results:")
                        println(file, "  Status: Optimal")
                        println(file, "  Returning only optimal solution (rc = $(round(rc_opt, digits=4)))")
                        println(file, "  Work units: $(round(pricing_work_units, digits=2))")
                        println(file, "="^80)
                    end
                end
                return return_pricing_payload(kept_solutions, kept_objs, pricing_work_units, return_nodes, model; analysis=analysis)
            else
                println("  Phase 1: Optimal solution has non-positive reduced cost (rc = $(round(rc_opt, digits=4))) — terminating pricing")
                # Add summary to log file
                if log_file !== nothing && iteration !== nothing
                    open(log_file, "a") do file
                        println(file, "\nPhase 1 Results:")
                        println(file, "  Status: Optimal but no RC-improving solution")
                        println(file, "  Reduced cost: $(round(rc_opt, digits=4))")
                        println(file, "  Work units: $(round(pricing_work_units, digits=2))")
                        println(file, "="^80)
                    end
                end
                return return_pricing_payload(Vector{Vector{Float64}}(), Float64[], pricing_work_units, return_nodes, model; analysis=analysis)
            end
        else
            println("  Phase 1: Not optimal - checking pool for RC-improving incumbents")
            kept_solutions, kept_objs = collect_pool_rc_cap10(model, x_vars, dual_mu, use_solution_pool)
            if !isempty(kept_solutions)
                println("  Phase 1: Found $(length(kept_solutions)) RC-improving solutions (non-optimal status)")
                return return_pricing_payload(kept_solutions, kept_objs, pricing_work_units, return_nodes, model; analysis=analysis)
            else
                println("  Phase 1: No RC-improving solutions found, proceeding to Phase 2")
            end
        end
    end

    # Phase 2: Optimality push
    heur_worklimit = work_limit - opt_worklimit
    if isfinite(heur_worklimit) && heur_worklimit > 0.0
        println("  Phase 2: Optimality push with $(heur_worklimit) work units")
        if analysis !== nothing
            analysis[:went_phase2] = true
        end
        set_optimizer_attribute(model, "MIPFocus", 0)
        set_optimizer_attribute(model, "WorkLimit", heur_worklimit)
        optimize!(model)
        
        try
            w2 = MOI.get(backend(model), Gurobi.ModelAttribute("Work"))
            phase2_work = max(0.0, w2 - last_work)
            pricing_work_units += phase2_work
            last_work = w2
            if analysis !== nothing
                analysis[:phase2_work] = phase2_work
            end
        catch
        end
        
        if termination_status(model) == MOI.OPTIMAL
            println("  Phase 2: Optimal solution found")
            try
                lb = MOI.get(backend(model), Gurobi.ModelAttribute("ObjBound"))
                ub = MOI.get(backend(model), Gurobi.ModelAttribute("ObjVal"))
                if analysis !== nothing
                    analysis[:lb] = lb
                    analysis[:ub] = ub
                end
            catch
            end
            kept_solutions, kept_objs = collect_pool_rc_cap10(model, x_vars, dual_mu, use_solution_pool)
            # Add summary to log file
            if log_file !== nothing && iteration !== nothing
                open(log_file, "a") do file
                    println(file, "\nPhase 2 Results:")
                    println(file, "  Status: Optimal")
                    println(file, "  RC-improving solutions found: $(length(kept_solutions))")
                    println(file, "  Total work units: $(round(pricing_work_units, digits=2))")
                    println(file, "="^80)
                end
            end
            return return_pricing_payload(kept_solutions, kept_objs, pricing_work_units, return_nodes, model; analysis=analysis)
        else
            println("  Phase 2: Work units exhausted. Reporting bounds...")
            try
                lb = MOI.get(backend(model), Gurobi.ModelAttribute("ObjBound"))
                ub = MOI.get(backend(model), Gurobi.ModelAttribute("ObjVal"))
                println("  Lower bound: $lb, Upper bound: $ub")
                if analysis !== nothing
                    analysis[:lb] = lb
                    analysis[:ub] = ub
                end
            catch
            end
            # Add summary to log file
            if log_file !== nothing && iteration !== nothing
                open(log_file, "a") do file
                    println(file, "\nPhase 2 Results:")
                    println(file, "  Status: Work units exhausted")
                    try
                        lb = MOI.get(backend(model), Gurobi.ModelAttribute("ObjBound"))
                        ub = MOI.get(backend(model), Gurobi.ModelAttribute("ObjVal"))
                        println(file, "  Lower bound: $lb, Upper bound: $ub")
                    catch
                    end
                    println(file, "  Total work units: $(round(pricing_work_units, digits=2))")
                    println(file, "="^80)
                end
            end
            return return_pricing_payload(Vector{Vector{Float64}}(), Float64[], pricing_work_units, return_nodes, model; analysis=analysis)
        end
    end
    
    # Fallback
    # Add summary to log file
    if log_file !== nothing && iteration !== nothing
        open(log_file, "a") do file
            println(file, "\nPricing Results:")
            println(file, "  Status: Fallback (no solutions found)")
            println(file, "  Work units: $(round(pricing_work_units, digits=2))")
            println(file, "="^80)
        end
    end
    return return_pricing_payload(Vector{Vector{Float64}}(), Float64[], pricing_work_units, return_nodes, model; analysis=analysis)
end

