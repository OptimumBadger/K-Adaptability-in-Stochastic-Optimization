# TransmissionSwitchingAF.jl
# Problem-specific implementations for Transmission Switching Column Generation
# This file provides the interface functions required by ColumnGenerationCore.jl

using JuMP, Gurobi
using MathOptInterface
const MOI = MathOptInterface
using Printf
using Statistics
using Random, Distributions
using PowerModels
using XLSX, DataFrames

# Include the core column generation algorithm
include("ColumnGenerationCore.jl")

# Include TS utilities (parameters_TSpm, extract_case_name, get_fixed_line_indices)
include("../Samples/TS/TSUtilities.jl")

# Include heuristic functions for consistency (demand multiplier, penalty)
script_dir = @__DIR__
project_root = dirname(script_dir)
include(joinpath(project_root, "Transmission_Switching", "Heuristic", "Candidate_List_Generation_Phase.jl"))

# =============================================================================
# CONSTANTS
# =============================================================================

const DEFAULT_OPT_WORKLIMIT = 300.0
# DEFAULT_ALPHA is now instance-specific (use get_penalty() instead)
# Keeping for backward compatibility, but should use get_penalty() for consistency

# =============================================================================
# WS FILE PARSING
# =============================================================================

function extract_f_s_from_ws_file(ws_file::String, S::Int)
    """Extract f_s values (wait-and-see costs) from WS file for TS
    
    Args:
        ws_file: Path to WS file (e.g., Samples/TS/Scenarios/instance_1/WS300_bernoulli.txt)
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
            
            # TS WS file format: scenario_index, wait_and_see_cost, work_units, penalty_pct, total_demand, unmet_value
            # We need the second column (wait_and_see_cost)
            parts = split(line, ',')
            if length(parts) >= 2
                try
                    cost = parse(Float64, strip(parts[2]))
                    push!(f_s, cost)
                catch
                    # If parsing fails, try to extract number from line
                    # Try to find first number in the line
                    numbers = [m.match for m in eachmatch(r"-?\d+\.?\d*", line)]
                    if !isempty(numbers)
                        try
                            cost = parse(Float64, numbers[1])
                            push!(f_s, cost)
                        catch
                            # Skip this line if we can't parse it
                            continue
                        end
                    end
                end
            end
        end
    end
    
    if length(f_s) < S
        error("❌ Error: WS file contains only $(length(f_s)) scenarios, but S = $S was requested.")
    end
    
    # Return first S values
    return f_s[1:S]
end

# =============================================================================
# DATA LOADING
# =============================================================================

function load_TS_data(instance_file::String; line_capacity_factor::Float64=1.0)
    """Load Transmission Switching instance data from MATPOWER file
    
    Returns:
        Dict with PowerModels data structure
    """
    if !isfile(instance_file)
        error("❌ Error: Instance file not found! Expected: $instance_file")
    end
    
    data = parse_file_silent(instance_file)
    
    # Store instance path for downstream pricing constraints
    data["instance_file"] = instance_file
    
    # Store capacity factor for later use
    data["line_capacity_factor"] = line_capacity_factor
    
    return data
end

# =============================================================================
# COST EVALUATION r_s(x)
# =============================================================================

function evaluate_cost_TS(binary_vector::Vector{Int}, scenario::Vector{Float64}, problem_data::Dict)
    """Evaluate cost r_s(x) for Transmission Switching
    
    Args:
        binary_vector: Binary solution vector (0/1 for each line)
        scenario: Demand scenario vector
        problem_data: PowerModels data structure
    
    Returns:
        (cost::Float64, work_units::Float64)
    """
    B = problem_data["bus"]
    L = problem_data["branch"]
    nb = length(B)
    nl = length(L)
    B_ij, fbar_ij, p_max, p_min, A, c_g = parameters_TSpm(problem_data)
    ref = 1
    
    # Use instance-specific penalty (consistent with heuristic)
    instance_file = get(problem_data, "instance_file", nothing)
    case_name = instance_file !== nothing ? extract_case_name(instance_file) : nothing
    penalty = get_penalty(instance_file=instance_file, case_name=case_name)

    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 0)
    set_optimizer_attribute(model, "MIPGap", 0.001)  # Consistent with heuristic (0.1%)

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

    # Line capacity + DC flow with fixed x (using binary_vector)
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

    # Power balance
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

# =============================================================================
# RECOURSE LP SOLVER WITH DUALS (for Decomposition)
# =============================================================================

function solve_ts_recourse_lp_with_duals(
    x_hat::Vector{Int},
    scenario::Vector{Float64},
    problem_data::Dict
)
    """Solve TS recourse LP for fixed x_hat and return q_s, dual variables, and work units.
    
    This solves the LP relaxation of the TS problem (Image 1) for a fixed binary vector x_hat.
    Returns q_s (recourse cost only, no fixed costs) and all dual variables.
    
    Args:
        x_hat: Fixed binary vector (0/1 for each line)
        scenario: Demand scenario vector
        problem_data: PowerModels data structure
    
    Returns:
        (q_s::Float64, duals::Dict, work_units::Float64)
        duals contains:
            - alpha: Vector{Float64} (power balance duals, free)
            - tau: Vector{Float64} (generator upper bound duals, >= 0)
            - sigma: Vector{Float64} (generator lower bound duals, >= 0)
            - phi: Dict{Int, Float64} (flow upper bound duals, >= 0)
            - psi: Dict{Int, Float64} (flow lower bound duals, >= 0)
            - eta: Dict{Int, Float64} (DC flow upper bound duals, >= 0)
            - zeta: Dict{Int, Float64} (DC flow lower bound duals, >= 0)
            - delta: Vector{Float64} (unmet upper bound duals, >= 0)
            - gamma: Vector{Float64} (unmet lower bound duals, >= 0)
    """
    B = problem_data["bus"]
    L = problem_data["branch"]
    nb = length(B)
    nl = length(L)
    B_ij, fbar_ij, p_max, p_min, A, c_g = parameters_TSpm(problem_data)
    ref = 1
    
    # Use instance-specific penalty
    instance_file = get(problem_data, "instance_file", nothing)
    case_name = instance_file !== nothing ? extract_case_name(instance_file) : nothing
    penalty = get_penalty(instance_file=instance_file, case_name=case_name)
    
    # Apply demand multiplier
    demand_mult = get_demand_multiplier(instance_file=instance_file, case_name=case_name)
    Pd = scenario .* demand_mult
    
    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 0)
    set_optimizer_attribute(model, "LogToConsole", 0)
    
    # Variables (LP, not MIP)
    @variable(model, p_g[1:nb])
    @variable(model, f[1:nl])
    @variable(model, theta[1:nb])
    @variable(model, unmet[1:nb] >= 0)
    
    # Constraint references for dual extraction
    power_balance_con = Vector{ConstraintRef}()
    gen_upper_con = Vector{ConstraintRef}()
    gen_lower_con = Vector{ConstraintRef}()
    flow_upper_con = Dict{Int, ConstraintRef}()
    flow_lower_con = Dict{Int, ConstraintRef}()
    dc_flow_upper_con = Dict{Int, ConstraintRef}()
    dc_flow_lower_con = Dict{Int, ConstraintRef}()
    unmet_upper_con = Vector{ConstraintRef}()
    unmet_lower_con = Vector{ConstraintRef}()
    
    # Constraint 1: Power balance (p_g[i] + u[i] - Σ_outgoing + Σ_incoming = p_d[i])
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
        push!(power_balance_con, @constraint(model, p_g[i] + unmet[i] - outgoing + incoming == Pd[i]))
    end
    
    # Constraint 2: Generator upper bounds (p_g[i] <= p_max[i])
    for i in 1:nb
        push!(gen_upper_con, @constraint(model, p_g[i] <= p_max[i]))
    end
    
    # Constraint 3: Generator lower bounds (p_g[i] >= p_min[i])
    for i in 1:nb
        push!(gen_lower_con, @constraint(model, p_g[i] >= p_min[i]))
    end
    
    # Constraints 4-5: Flow bounds (only for lines with x_hat[l] == 1)
    # Constraint 6-7: DC power flow (only for lines with x_hat[l] == 1)
    M_ij = Dict{Int, Float64}()
    for (branch_id, branch) in L
        l = parse(Int, branch_id)
        M_ij[l] = -12.6 * B_ij[l]  # Big-M value
        f_bus = branch["f_bus"]
        t_bus = branch["t_bus"]
        
        if x_hat[l] == 1  # Line is ON
            # Constraint 4: f[l] <= fbar_ij[l]
            flow_upper_con[l] = @constraint(model, f[l] <= fbar_ij[l])
            # Constraint 5: -f[l] <= fbar_ij[l] (i.e., f[l] >= -fbar_ij[l])
            flow_lower_con[l] = @constraint(model, -f[l] <= fbar_ij[l])
            # Constraint 6: f[l] - B_ij[l] * (theta[f_bus] - theta[t_bus]) <= M_ij[l]
            dc_flow_upper_con[l] = @constraint(model, f[l] - B_ij[l] * (theta[f_bus] - theta[t_bus]) <= M_ij[l])
            # Constraint 7: -f[l] + B_ij[l] * (theta[f_bus] - theta[t_bus]) <= M_ij[l]
            dc_flow_lower_con[l] = @constraint(model, -f[l] + B_ij[l] * (theta[f_bus] - theta[t_bus]) <= M_ij[l])
        else  # Line is OFF
            # f[l] == 0 (handled separately if needed, but typically not needed for LP)
        end
    end
    
    # Constraint 8: Unmet upper bounds (u[i] <= p_d[i])
    for i in 1:nb
        push!(unmet_upper_con, @constraint(model, unmet[i] <= Pd[i]))
    end
    
    # Constraint 9: Unmet lower bounds (u[i] >= 0) - already enforced by variable bounds
    
    # Reference angle (theta[1] == 0) - no dual needed for this
    
    # Objective: min Σ c_i * p_g[i] + K * Σ u[i]
    @objective(model, Min, sum(c_g[i] * p_g[i] for i in 1:nb) + penalty * sum(unmet[i] for i in 1:nb))
    
    optimize!(model)
    
    work_units = 0.0
    try
        work_units = MOI.get(backend(model), Gurobi.ModelAttribute("Work"))
    catch
    end
    
    ts = termination_status(model)
    if ts != MOI.OPTIMAL
        return Inf, Dict{String, Any}(), work_units
    end
    
    # Extract dual variables
    alpha = [dual(power_balance_con[i]) for i in 1:nb]
    tau = [dual(gen_upper_con[i]) for i in 1:nb]
    sigma = [dual(gen_lower_con[i]) for i in 1:nb]
    
    phi = Dict{Int, Float64}()
    psi = Dict{Int, Float64}()
    eta = Dict{Int, Float64}()
    zeta = Dict{Int, Float64}()
    for l in 1:nl
        if x_hat[l] == 1
            phi[l] = haskey(flow_upper_con, l) ? dual(flow_upper_con[l]) : 0.0
            psi[l] = haskey(flow_lower_con, l) ? dual(flow_lower_con[l]) : 0.0
            eta[l] = haskey(dc_flow_upper_con, l) ? dual(dc_flow_upper_con[l]) : 0.0
            zeta[l] = haskey(dc_flow_lower_con, l) ? dual(dc_flow_lower_con[l]) : 0.0
        else
            phi[l] = 0.0
            psi[l] = 0.0
            eta[l] = 0.0
            zeta[l] = 0.0
        end
    end
    
    delta = [dual(unmet_upper_con[i]) for i in 1:nb]
    gamma = zeros(Float64, nb)  # Unmet lower bounds are variable bounds, dual is 0
    
    # q_s = recourse cost only (no fixed costs for TS)
    q_s = objective_value(model)
    
    duals = Dict{String, Any}(
        "alpha" => alpha,
        "tau" => tau,
        "sigma" => sigma,
        "phi" => phi,
        "psi" => psi,
        "eta" => eta,
        "zeta" => zeta,
        "delta" => delta,
        "gamma" => gamma,
        "M_ij" => M_ij,
        "fbar_ij" => fbar_ij
    )
    
    return q_s, duals, work_units
end

# =============================================================================
# M_S CALCULATION (for Cut 34)
# =============================================================================

function compute_M_value_TS(
    duals::Dict{String, Any},
    scenario::Vector{Float64},
    problem_data::Dict,
    dual_lambda_s::Float64
)
    """Compute M_s(μ̂) for TS by solving the maximization problem:
    
    M_s = max_{x ∈ X} [Σ_{i∈B} p_i^d α_i 
          + Σ_{i∈G} (p_i^max τ_i - p_i^min σ_i)
          + Σ_{(i,j)∈L} f_{ij} x_{ij} (φ_ij + ψ_ij)
          + Σ_{(i,j)∈L} M_{ij} (1 - x_{ij}) (η_ij + ζ_ij)
          + Σ_{i∈B} p_i^d δ_i
          - λ_s^*]
    
    Where x is a binary vector (x_{ij} ∈ {0,1} for each line).
    
    This simplifies to:
    M_s = max_{x ∈ X} [constant_term + Σ_{(i,j)∈L} x_{ij} * (f_{ij}(φ_ij + ψ_ij) - M_{ij}(η_ij + ζ_ij))]
    
    Where constant_term = Σ p_i^d α_i + Σ (p_i^max τ_i - p_i^min σ_i) + Σ M_{ij}(η_ij + ζ_ij) + Σ p_i^d δ_i - λ_s^*
    
    For each line l:
    - If coefficient (f_{ij}(φ_ij + ψ_ij) - M_{ij}(η_ij + ζ_ij)) > 0, set x_l = 1
    - Otherwise, set x_l = 0
    
    Args:
        duals: Dictionary of dual variables from solve_ts_recourse_lp_with_duals
        scenario: Demand scenario vector
        problem_data: PowerModels data structure
        dual_lambda_s: dual_lambda[s] value
    
    Returns:
        M_s::Float64
    """
    B = problem_data["bus"]
    L = problem_data["branch"]
    nb = length(B)
    nl = length(L)
    
    # Apply demand multiplier
    demand_mult = get_demand_multiplier(instance_file=get(problem_data, "instance_file", nothing))
    Pd = scenario .* demand_mult
    
    B_ij, fbar_ij, p_max, p_min, A, c_g = parameters_TSpm(problem_data)
    
    alpha = duals["alpha"]
    tau = duals["tau"]
    sigma = duals["sigma"]
    phi = duals["phi"]
    psi = duals["psi"]
    eta = duals["eta"]
    zeta = duals["zeta"]
    delta = duals["delta"]
    M_ij = duals["M_ij"]
    
    # Constant term (independent of x)
    term1 = sum(Pd[i] * alpha[i] for i in 1:nb)
    term2 = sum(p_max[i] * tau[i] - p_min[i] * sigma[i] for i in 1:nb)
    term5 = sum(Pd[i] * delta[i] for i in 1:nb)
    
    # Term 4 constant part: Σ_{(i,j)∈L} M_{ij} (η_ij + ζ_ij) (when x_{ij} = 0)
    term4_constant = 0.0
    for (branch_id, branch) in L
        l = parse(Int, branch_id)
        eta_val = haskey(eta, l) ? eta[l] : 0.0
        zeta_val = haskey(zeta, l) ? zeta[l] : 0.0
        term4_constant += M_ij[l] * (eta_val + zeta_val)
    end
    
    constant_term = term1 + term2 + term4_constant + term5 - dual_lambda_s
    
    # Variable term: maximize over x
    # For each line l, coefficient is: f_{ij}(φ_ij + ψ_ij) - M_{ij}(η_ij + ζ_ij)
    # If coefficient > 0, set x_l = 1; otherwise x_l = 0
    M_s = constant_term
    for (branch_id, branch) in L
        l = parse(Int, branch_id)
        phi_val = haskey(phi, l) ? phi[l] : 0.0
        psi_val = haskey(psi, l) ? psi[l] : 0.0
        eta_val = haskey(eta, l) ? eta[l] : 0.0
        zeta_val = haskey(zeta, l) ? zeta[l] : 0.0
        
        # Coefficient for x_l: f_{ij}(φ_ij + ψ_ij) - M_{ij}(η_ij + ζ_ij)
        coefficient = fbar_ij[l] * (phi_val + psi_val) - M_ij[l] * (eta_val + zeta_val)
        
        # If coefficient > 0, set x_l = 1 to maximize
        if coefficient > 0.0
            M_s += coefficient
        end
        # If coefficient <= 0, x_l = 0 (no contribution)
    end
    
    return M_s
end

# =============================================================================
# HELPER FUNCTIONS FOR DECOMPOSITION
# =============================================================================

function x_signature_TS(x_hat::Vector{Int})
    """Create a string signature for x_hat for caching purposes."""
    return join(x_hat, ",")
end

# =============================================================================
# DECOMPOSITION PRICING (Type 2 with Cutting-Plane Method)
# =============================================================================

function af_pricing_type2_TS_decomp(
    problem_data::Dict,
    scenarios::Vector{Vector{Float64}},
    dual_lambda::Vector{Float64},
    dual_mu::Float64,
    f_s_vector::Vector{Float64},
    decomp_cache::Dict{String,Any};
    return_nodes::Bool=false,
    analysis::Dict{Symbol,Any}=Dict{Symbol,Any}(),
    log_file=nothing,
    iteration=nothing,
    max_cut_iters::Int=500,
    tol::Float64=1e-4
)
    """Solve AF Pricing Type 2 for TS with decomposition cutting-plane procedure."""
    B = problem_data["bus"]
    L = problem_data["branch"]
    nb = length(B)
    nl = length(L)
    nscen = length(scenarios)
    
    if length(f_s_vector) != nscen
        error("❌ Error: f_s vector length ($(length(f_s_vector))) must equal number of scenarios ($nscen).")
    end
    
    B_ij, fbar_ij, p_max, p_min, A, c_g = parameters_TSpm(problem_data)
    
    # Get fixed lines (if any)
    fixed_lines = get_fixed_line_indices(problem_data)
    
    println("\n" * "="^80)
    println("DECOMPOSITION ALGORITHM - PRICING TYPE 2 (TS)")
    println("="^80)
    println("  Using f_s vector (wait-and-see costs) from WS file")
    println("  Number of scenarios: $nscen")
    println("  Number of lines: $nl")
    println("  Fixed lines: $(length(fixed_lines))")
    println("  Max cutting-plane iterations: $max_cut_iters")
    println("  Tolerance: $tol")
    println("="^80)
    
    # Write to log file if provided
    if log_file !== nothing && iteration !== nothing
        open(log_file, "a") do file
            println(file, "\n" * "="^80)
            println(file, "PRICING SUBPROBLEM TYPE 2 (DECOMPOSITION) - ITERATION $iteration")
            println(file, "="^80)
            println(file, "Timestamp: $(now())")
            println(file, "Problem size: $nscen scenarios, $nl lines, $nb buses")
            println(file, "Fixed lines: $(length(fixed_lines))")
            println(file, "Max cutting-plane iterations: $max_cut_iters")
            println(file, "Tolerance: $tol")
            println(file, "="^80)
        end
    end
    
    println("\nBuilding initial master problem...")
    master = Model(Gurobi.Optimizer)
    set_optimizer_attribute(master, "OutputFlag", 0)
    set_optimizer_attribute(master, "LogToConsole", 0)
    
    @variable(master, x[1:nl], Bin)
    @variable(master, pi[1:nscen], Bin)
    @variable(master, theta[1:nscen] >= 0)
    
    # Objective: Max Σ theta[s]
    @objective(master, Max, sum(theta[s] for s in 1:nscen))
    
    # Base constraint: theta[s] <= (dual_lambda[s] - f_s) * pi[s]
    @constraint(master, [s in 1:nscen], theta[s] <= (dual_lambda[s] - f_s_vector[s]) * pi[s])
    
    # Fixed lines constraints: x[fixed_lines] = 1
    for l in fixed_lines
        if 1 <= l <= nl
            @constraint(master, x[l] == 1)
        end
    end
    
    println("  Master problem built: max Σ θ_s, with base constraints θ_s ≤ (λ_s^* - f_s)π_s")
    println("  Fixed $(length(fixed_lines)) lines to ON")
    
    scenario_x_cache = get(decomp_cache, "scenario_x_cache", Dict{Tuple{Int,String}, Dict{String,Any}}())
    # Preload all previously generated cuts with CURRENT dual_lambda.
    cut_records = get(decomp_cache, "records", Vector{Dict{String,Any}}())
    num_preloaded = length(cut_records)
    cut32_preloaded = 0
    cut33_preloaded = 0
    cut34_preloaded = 0
    
    if num_preloaded > 0
        println("\nPreloading $num_preloaded cached cut records with current dual_lambda values...")
        if log_file !== nothing && iteration !== nothing
            open(log_file, "a") do file
                println(file, "\nPreloading $num_preloaded cached cut records with current dual_lambda values...")
            end
        end
        
        for rec in cut_records
            s = rec["s"]::Int
            x_hat = rec["x_hat"]::Vector{Int}
            q_s = rec["q_s"]::Float64
            duals = rec["duals"]::Dict{String,Any}
            
            # Recompute M_s with current dual_lambda[s]
            M_s = compute_M_value_TS(duals, scenarios[s], problem_data, dual_lambda[s])
            
            # Compute F0 and F1 sets for delta_expr
            F0 = [l for l in 1:nl if x_hat[l] == 0]
            F1 = [l for l in 1:nl if x_hat[l] == 1]
            delta_expr = sum(1 - x[l] for l in F1; init=0.0) + sum(x[l] for l in F0; init=0.0)
            
            # Cut 32 / Cut 33: Check sign with CURRENT dual_lambda
            sign_check = dual_lambda[s] - q_s
            if sign_check <= 0.0
                @constraint(master, delta_expr >= pi[s])
                cut32_preloaded += 1
                if log_file !== nothing && iteration !== nothing
                    open(log_file, "a") do file
                        println(file, "  Scenario $s: Preloaded Cut 32 (Conditional No-Good) [λ_s^* - q_s = $(round(sign_check, digits=4)) ≤ 0]")
                    end
                end
            else
                @constraint(master, theta[s] <= (dual_lambda[s] - q_s) + (q_s - f_s_vector[s]) * delta_expr)
                cut33_preloaded += 1
                if log_file !== nothing && iteration !== nothing
                    open(log_file, "a") do file
                        println(file, "  Scenario $s: Preloaded Cut 33 (Integer L-Shaped) [λ_s^* - q_s = $(round(sign_check, digits=4)) > 0]")
                    end
                end
            end
            
            # Cut 34: Preload using formula from image 3
            alpha = duals["alpha"]
            tau = duals["tau"]
            sigma = duals["sigma"]
            phi = duals["phi"]
            psi = duals["psi"]
            eta = duals["eta"]
            zeta = duals["zeta"]
            delta = duals["delta"]
            M_ij = duals["M_ij"]
            fbar_ij = duals["fbar_ij"]
            
            # Apply demand multiplier
            demand_mult = get_demand_multiplier(instance_file=get(problem_data, "instance_file", nothing))
            Pd = scenarios[s] .* demand_mult
            
            # Cut 34: θ_s ≤ -Σ p_i^d α_i - Σ (p_i^max τ_i - p_i^min σ_i) - Σ F_ij (φ_ij + ψ_ij) 
            #         - Σ M̄_ij (η_ij + ζ_ij) - Σ p_i^d δ_i + λ_s^* + M_s(1 - π_s)
            # Compute terms as constants (not JuMP expressions)
            term1_cut34 = sum(Pd[i] * alpha[i] for i in 1:nb)
            term2_cut34 = sum(p_max[i] * tau[i] - p_min[i] * sigma[i] for i in 1:nb)
            term3_cut34 = 0.0
            for (branch_id, branch) in L
                l = parse(Int, branch_id)
                phi_val = haskey(phi, l) ? phi[l] : 0.0
                psi_val = haskey(psi, l) ? psi[l] : 0.0
                term3_cut34 += fbar_ij[l] * (phi_val + psi_val)
            end
            term4_cut34 = 0.0
            for (branch_id, branch) in L
                l = parse(Int, branch_id)
                eta_val = haskey(eta, l) ? eta[l] : 0.0
                zeta_val = haskey(zeta, l) ? zeta[l] : 0.0
                term4_cut34 += M_ij[l] * (eta_val + zeta_val)
            end
            term5_cut34 = sum(Pd[i] * delta[i] for i in 1:nb)
            
            # Constraint with precomputed constant terms
            @constraint(master,
                theta[s] <= 
                -term1_cut34 - term2_cut34 - term3_cut34 - term4_cut34 - term5_cut34 +
                dual_lambda[s] +
                M_s * (1 - pi[s])
            )
            cut34_preloaded += 1
            if log_file !== nothing && iteration !== nothing
                open(log_file, "a") do file
                    println(file, "  Scenario $s: Preloaded Cut 34 (Conditional Benders)")
                end
            end
        end
        
        println("  Preloaded: $cut32_preloaded Cut 32, $cut33_preloaded Cut 33, $cut34_preloaded Cut 34")
        if log_file !== nothing && iteration !== nothing
            open(log_file, "a") do file
                println(file, "  Preloaded: $cut32_preloaded Cut 32, $cut33_preloaded Cut 33, $cut34_preloaded Cut 34")
            end
        end
    else
        println("  No previous cuts to preload (first pricing solve)")
    end
    
    pricing_work_units = 0.0
    pi_fixed = falses(nscen)
    last_x_hat = zeros(Int, nl)
    last_obj = -Inf
    
    println("\nStarting cutting-plane iterations...")
    if log_file !== nothing && iteration !== nothing
        open(log_file, "a") do file
            println(file, "\nStarting cutting-plane iterations...")
        end
    end
    
    for it in 1:max_cut_iters
        iter_header = "DECOMPOSITION ITERATION $it"
        # Print in red to console
        println("\n\033[91m" * iter_header * "\033[0m")
        if log_file !== nothing && iteration !== nothing
            open(log_file, "a") do file
                println(file, "\n" * iter_header)
            end
        end
        
        println("  Solving master problem...")
        if log_file !== nothing && iteration !== nothing
            open(log_file, "a") do file
                println(file, "  Solving master problem...")
            end
        end
        optimize!(master)
        if termination_status(master) != MOI.OPTIMAL
            println("  Master problem did not solve to optimality. Status: $(termination_status(master))")
            return return_pricing_payload(Vector{Vector{Float64}}(), Float64[], pricing_work_units, return_nodes, master; analysis=analysis)
        end
        
        try
            pricing_work_units += MOI.get(backend(master), Gurobi.ModelAttribute("Work"))
        catch
        end
        
        x_hat = [round(Int, value(x[l])) for l in 1:nl]
        pi_hat = [round(Int, value(pi[s])) for s in 1:nscen]
        theta_hat = [value(theta[s]) for s in 1:nscen]
        obj_hat = objective_value(master)
        
        println("  Master solution: obj = $(round(obj_hat, digits=4))")
        println("  x_hat: $(sum(x_hat)) lines ON, $(nl - sum(x_hat)) lines OFF")
        println("  pi_hat: $(sum(pi_hat)) scenarios active")
        if log_file !== nothing && iteration !== nothing
            open(log_file, "a") do file
                println(file, "  Master solution: obj = $(round(obj_hat, digits=4))")
                println(file, "  x_hat: $(sum(x_hat)) lines ON")
                println(file, "  pi_hat: $(sum(pi_hat)) scenarios active")
            end
        end
        
        # Check for convergence (same solution as last iteration)
        if x_hat == last_x_hat && abs(obj_hat - last_obj) < 1e-6
            println("  Solution unchanged from previous iteration → terminating")
            if log_file !== nothing && iteration !== nothing
                open(log_file, "a") do file
                    println(file, "  Solution unchanged from previous iteration → terminating")
                end
            end
            break
        end
        last_x_hat = copy(x_hat)
        last_obj = obj_hat
        
        # Trivial condition: if f_s >= dual_lambda[s], force pi[s] = 0.
        added_trivial_fix = false
        trivial_count = 0
        for s in 1:nscen
            if !pi_fixed[s] && f_s_vector[s] >= dual_lambda[s]
                @constraint(master, pi[s] == 0)
                pi_fixed[s] = true
                added_trivial_fix = true
                trivial_count += 1
            end
        end
        if added_trivial_fix
            println("  Trivial condition: Fixed π[s] = 0 for $trivial_count scenarios (f_s >= λ_s^*)")
            println("  Re-solving master with trivial fixes...")
            if log_file !== nothing && iteration !== nothing
                open(log_file, "a") do file
                    println(file, "  Trivial condition: Fixed π[s] = 0 for $trivial_count scenarios (f_s >= λ_s^*)")
                    println(file, "  Re-solving master with trivial fixes...")
                end
            end
            continue
        end
        
        F1 = [l for l in 1:nl if x_hat[l] == 1]
        F0 = [l for l in 1:nl if x_hat[l] == 0]
        delta_expr = sum(1 - x[l] for l in F1; init=0) + sum(x[l] for l in F0; init=0)
        println("  Computed F0 (OFF): $(length(F0)) lines, F1 (ON): $(length(F1)) lines")
        if log_file !== nothing && iteration !== nothing
            open(log_file, "a") do file
                println(file, "  Computed F0 (OFF): $(length(F0)) lines, F1 (ON): $(length(F1)) lines")
                println(file, "  Processing scenarios...")
            end
        end
        
        added_cut34 = false
        cut32_new = 0
        cut33_new = 0
        cut34_new = 0
        cache_hits = 0
        cache_misses = 0
        
        sig = x_signature_TS(x_hat)
        println("  Solution signature: $sig")
        println("  Processing scenarios...")
        
        for s in 1:nscen
            key = (s, sig)
            q_s = Inf
            duals = Dict{String, Any}()
            M_s = 0.0
            from_cache = false
            
            if haskey(scenario_x_cache, key)
                cache_rec = scenario_x_cache[key]
                q_s = cache_rec["q_s"]::Float64
                duals = cache_rec["duals"]::Dict{String,Any}
                M_s = cache_rec["M_s"]::Float64
                from_cache = true
                cache_hits += 1
                println("    Scenario $s: Using cached values [q_s = $(round(q_s, digits=2)), M_s = $(round(M_s, digits=2))]")
            else
                print("    Scenario $s: Solving recourse LP... ")
                q_s, duals, sub_work = solve_ts_recourse_lp_with_duals(
                    x_hat, scenarios[s], problem_data
                )
                pricing_work_units += sub_work
                cache_misses += 1
                if isfinite(q_s)
                    M_s = compute_M_value_TS(duals, scenarios[s], problem_data, dual_lambda[s])
                    println("q_s = $(round(q_s, digits=2)), M_s = $(round(M_s, digits=2))")
                    cache_rec = Dict{String,Any}(
                        "s" => s,
                        "x_sig" => sig,
                        "x_hat" => copy(x_hat),
                        "q_s" => q_s,
                        "duals" => duals,
                        "M_s" => M_s
                    )
                    scenario_x_cache[key] = cache_rec
                    push!(cut_records, cache_rec)
                else
                    println("Infeasible/unbounded")
                end
            end
            
            if !isfinite(q_s)
                continue
            end
            
            # Cut 32 / Cut 33
            sign_check = dual_lambda[s] - q_s
            if sign_check <= 0.0
                @constraint(master, delta_expr >= pi[s])
                cut_msg = "    Scenario $s: Added Cut 32 (Conditional No-Good) [λ_s^* - q_s = $(round(sign_check, digits=4)) ≤ 0]" * (from_cache ? " [from cache]" : "")
                println(cut_msg)
                if log_file !== nothing && iteration !== nothing
                    open(log_file, "a") do file
                        println(file, cut_msg)
                    end
                end
                cut32_new += 1
            else
                @constraint(master, theta[s] <= (dual_lambda[s] - q_s) + (q_s - f_s_vector[s]) * delta_expr)
                cut_msg = "    Scenario $s: Added Cut 33 (Integer L-Shaped) [λ_s^* - q_s = $(round(sign_check, digits=4)) > 0]" * (from_cache ? " [from cache]" : "")
                println(cut_msg)
                if log_file !== nothing && iteration !== nothing
                    open(log_file, "a") do file
                        println(file, cut_msg)
                    end
                end
                cut33_new += 1
            end
            
            # Cut 34 trigger
            lhs_check = theta_hat[s]
            rhs_check = (dual_lambda[s] - q_s) * pi_hat[s] + tol
            if lhs_check >= rhs_check
                alpha = duals["alpha"]
                tau = duals["tau"]
                sigma = duals["sigma"]
                phi = duals["phi"]
                psi = duals["psi"]
                eta = duals["eta"]
                zeta = duals["zeta"]
                delta = duals["delta"]
                M_ij = duals["M_ij"]
                fbar_ij = duals["fbar_ij"]
                
                # Apply demand multiplier
                demand_mult = get_demand_multiplier(instance_file=get(problem_data, "instance_file", nothing))
                Pd = scenarios[s] .* demand_mult
                
                # Cut 34: θ_s ≤ -Σ p_i^d α_i - Σ (p_i^max τ_i - p_i^min σ_i) - Σ F_ij (φ_ij + ψ_ij) 
                #         - Σ M̄_ij (η_ij + ζ_ij) - Σ p_i^d δ_i + λ_s^* + M_s(1 - π_s)
                @constraint(master,
                    theta[s] <= 
                    -sum(Pd[i] * alpha[i] for i in 1:nb) -
                    sum(p_max[i] * tau[i] - p_min[i] * sigma[i] for i in 1:nb) -
                    sum((haskey(phi, l) ? phi[l] : 0.0) + (haskey(psi, l) ? psi[l] : 0.0)) * fbar_ij[l] for (branch_id, branch) in L for l in [parse(Int, branch_id)]) -
                    sum((haskey(eta, l) ? eta[l] : 0.0) + (haskey(zeta, l) ? zeta[l] : 0.0)) * M_ij[l] for (branch_id, branch) in L for l in [parse(Int, branch_id)]) -
                    sum(Pd[i] * delta[i] for i in 1:nb) +
                    dual_lambda[s] +
                    M_s * (1 - pi[s])
                )
                cut_msg = "    Scenario $s: Added Cut 34 (Conditional Benders) [θ_hat = $(round(lhs_check, digits=4)) >= $(round(rhs_check, digits=4))]" * (from_cache ? " [from cache]" : "")
                println(cut_msg)
                if log_file !== nothing && iteration !== nothing
                    open(log_file, "a") do file
                        println(file, cut_msg)
                    end
                end
                added_cut34 = true
                cut34_new += 1
            end
        end
        
        println("  Iteration $it summary: Cut 32: $cut32_new, Cut 33: $cut33_new, Cut 34: $cut34_new")
        println("  Cache: $cache_hits hits, $cache_misses misses")
        if log_file !== nothing && iteration !== nothing
            open(log_file, "a") do file
                println(file, "  Iteration $it summary: Cut 32: $cut32_new, Cut 33: $cut33_new, Cut 34: $cut34_new")
                println(file, "  Cache: $cache_hits hits, $cache_misses misses")
            end
        end
        
        if !added_cut34
            term_msg = "  No Cut 34 added → Termination condition satisfied"
            println(term_msg)
            if log_file !== nothing && iteration !== nothing
                open(log_file, "a") do file
                    println(file, term_msg)
                end
            end
            break
        end
    end
    
    decomp_cache["scenario_x_cache"] = scenario_x_cache
    decomp_cache["records"] = cut_records
    
    println("\n" * "="^80)
    println("DECOMPOSITION ALGORITHM COMPLETE")
    println("="^80)
    println("  Total cache entries: $(length(scenario_x_cache))")
    println("  Total cut records stored: $(length(cut_records))")
    println("  Total work units: $(round(pricing_work_units, digits=2))")
    if log_file !== nothing && iteration !== nothing
        open(log_file, "a") do file
            println(file, "\n" * "="^80)
            println(file, "DECOMPOSITION ALGORITHM COMPLETE")
            println(file, "="^80)
            println(file, "  Total cache entries: $(length(scenario_x_cache))")
            println(file, "  Total cut records stored: $(length(cut_records))")
            println(file, "  Total work units: $(round(pricing_work_units, digits=2))")
        end
    end
    
    # Extract final solution
    if termination_status(master) == MOI.OPTIMAL
        x_opt = value.(x)
        obj_opt = objective_value(master)
        rc_opt = obj_opt + dual_mu
        
        println("\nFinal solution:")
        println("  Objective: $(round(obj_opt, digits=4))")
        println("  Reduced cost: $(round(rc_opt, digits=4))")
        println("  Lines ON: $(sum(x_opt))")
        
        if rc_opt > 1.0
            println("  Solution has positive reduced cost → adding to column generation")
            if log_file !== nothing && iteration !== nothing
                open(log_file, "a") do file
                    println(file, "\nFinal solution:")
                    println(file, "  Objective: $(round(obj_opt, digits=4))")
                    println(file, "  Reduced cost: $(round(rc_opt, digits=4))")
                    println(file, "  Solution added to column generation")
                    println(file, "="^80)
                end
            end
            return return_pricing_payload([x_opt], [obj_opt], pricing_work_units, return_nodes, master; analysis=analysis)
        else
            println("  Solution has non-positive reduced cost → not adding")
            if log_file !== nothing && iteration !== nothing
                open(log_file, "a") do file
                    println(file, "\nFinal solution:")
                    println(file, "  Objective: $(round(obj_opt, digits=4))")
                    println(file, "  Reduced cost: $(round(rc_opt, digits=4))")
                    println(file, "  Solution NOT added (non-positive RC)")
                    println(file, "="^80)
                end
            end
            return return_pricing_payload(Vector{Vector{Float64}}(), Float64[], pricing_work_units, return_nodes, master; analysis=analysis)
        end
    else
        println("  Master problem did not solve to optimality")
        return return_pricing_payload(Vector{Vector{Float64}}(), Float64[], pricing_work_units, return_nodes, master; analysis=analysis)
    end
end

# =============================================================================
# BATCH COST EVALUATION USING TEMPLATE MODEL (OPTIMIZED)
# =============================================================================

# =============================================================================
# M VALUES CALCULATION (for Pricing Type 1)
# =============================================================================

function calculate_m_values_TS(scenarios::Vector{Vector{Float64}}, dual_lambda::Vector{Float64}, problem_data::Dict)
    """Calculate m^s values for all scenarios (for Pricing Type 1)"""
    B_ij, fbar_ij, p_max, p_min, A, c = parameters_TSpm(problem_data)
    nb = length(problem_data["bus"])
    nscen = length(scenarios)
    
    # Use instance-specific penalty (consistent with heuristic)
    instance_file = get(problem_data, "instance_file", nothing)
    case_name = instance_file !== nothing ? extract_case_name(instance_file) : nothing
    alpha = get_penalty(instance_file=instance_file, case_name=case_name)
    
    m_values = zeros(nscen)
    for s in 1:nscen
        m_values[s] = big_m_calculator_TS(dual_lambda[s], c, p_min, p_max, scenarios[s], alpha)
    end
    
    return m_values
end

function big_m_calculator_TS(dual_lambda, c, p_min, p_max, demand, alpha)
    """Calculate m^s for a single scenario"""
    nb = length(demand)
    
    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 0)
    set_optimizer_attribute(model, "MIPGap", 0.001)  # Consistent with heuristic (0.1%)
    set_optimizer_attribute(model, "WorkLimit", 1800)
    
    @variable(model, p_g[1:nb])
    @variable(model, r[1:nb] >= 0)
    
    for i in 1:nb
        @constraint(model, p_g[i] >= p_min[i])
        @constraint(model, p_g[i] <= p_max[i])
    end
    
    @constraint(model, [i in 1:nb], p_g[i] + r[i] >= demand[i])
    
    @objective(model, Min, dual_lambda - sum(c[i] * p_g[i] for i in 1:nb) - alpha * sum(r[i] for i in 1:nb))
    
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

function af_pricing_type1_TS(data, scenarios, m_values, dual_lambda, dual_mu; use_solution_pool::Bool=true, work_limit::Float64=Inf, opt_worklimit::Float64=DEFAULT_OPT_WORKLIMIT, alpha::Float64=DEFAULT_ALPHA, return_nodes::Bool=false)
    """Solve AF Pricing Type 1 for Transmission Switching"""
    B_ij, fbar_ij, p_max, p_min, A, c = parameters_TSpm(data)
    L = data["branch"]
    nb = length(data["bus"])
    nl = length(L)
    nscen = length(scenarios)

    # Big-M for DC equality relaxation when x_l = 0
    M_ij = [12.6 * abs(B_ij[l]) for l in 1:nl]

    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 1)
    if isfinite(opt_worklimit)
        set_optimizer_attribute(model, "WorkLimit", opt_worklimit)
    end
    if use_solution_pool
        set_optimizer_attribute(model, "PoolSearchMode", 2)
        set_optimizer_attribute(model, "PoolSolutions", 10)
    end

    # Decision variables
    @variable(model, x[1:nl], Bin)
    @variable(model, pi[1:nscen], Bin)
    @variable(model, p_g[1:nb, 1:nscen])
    @variable(model, r[1:nb, 1:nscen] >= 0)
    @variable(model, f[1:nl, 1:nscen])
    @variable(model, theta[1:nb, 1:nscen])
    @variable(model, pbar[1:nb, 1:nscen])
    @variable(model, rbar[1:nb, 1:nscen] >= 0)

    # Fix lines ON based on capacity utilization file or CSV (top K lines)
    # Check if no_fixed_lines flag is set (all lines switchable)
    if get(data, "no_fixed_lines", false)
        fixed_lines = Int[]  # No lines fixed, all switchable
    else
        num_lines = get(data, "num_lines_fixed", nothing)
        fixed_lines = get_fixed_line_indices(data; num_lines=num_lines)
    end
    for l in fixed_lines
        if 1 <= l <= nl
            @constraint(model, x[l] == 1)
        else
            println("⚠️  Warning: Fixed line index $l is out of range (1-$nl)")
        end
    end

    # Objective
    @objective(model, Max, sum(dual_lambda[s] * pi[s] for s in 1:nscen)
                        - sum(c[i] * pbar[i, s] for i in 1:nb for s in 1:nscen)
                        - alpha * sum(rbar[i, s] for i in 1:nb for s in 1:nscen))

    # Generator bounds per scenario
    for i in 1:nb, s in 1:nscen
        @constraint(model, p_g[i, s] >= p_min[i]*pi[s])
        @constraint(model, p_g[i, s] <= p_max[i]*pi[s])
    end

    # Line capacity and DC constraints
    for (branch_id, branch) in L
        l = parse(Int, branch_id)
        i = branch["f_bus"]
        j = branch["t_bus"]
        for s in 1:nscen
            @constraint(model, -fbar_ij[l] * x[l] <= f[l, s])
            @constraint(model,  f[l, s] <= fbar_ij[l] * x[l])
            @constraint(model, B_ij[l] * (theta[i, s] - theta[j, s]) - M_ij[l] * (1 - x[l]) <= f[l, s])
            @constraint(model, f[l, s] <= B_ij[l] * (theta[i, s] - theta[j, s]) + M_ij[l] * (1 - x[l]))
        end
    end

    # Power balance per bus and scenario
    for s in 1:nscen
        d_s = scenarios[s]
        for i in 1:nb
            outgoing = 0.0
            incoming = 0.0
            for (branch_id, branch) in L
                l = parse(Int, branch_id)
                if branch["f_bus"] == i
                    outgoing += f[l, s]
                end
                if branch["t_bus"] == i
                    incoming += f[l, s]
                end
            end
            @constraint(model, p_g[i, s] + r[i, s] - d_s[i]*pi[s] == outgoing - incoming)
        end
        @constraint(model, theta[1, s] == 0)
    end

    # Pricing key inequality per scenario
    for s in 1:nscen
        @constraint(model, dual_lambda[s]
            - sum(c[i] * p_g[i, s] for i in 1:nb)
            - alpha * sum(r[i, s] for i in 1:nb)
            >= m_values[s] * (1 - pi[s]))
    end

    # Linearization for pbar and rbar with pi
    for i in 1:nb, s in 1:nscen
        @constraint(model, pbar[i, s] <= p_g[i, s])
        @constraint(model, pbar[i, s] <= p_max[i] * pi[s])
        @constraint(model, pbar[i, s] >= p_g[i, s] - p_max[i] * (1 - pi[s]))
        d_is = scenarios[s][i]
        @constraint(model, rbar[i, s] <= r[i, s])
        @constraint(model, rbar[i, s] <= d_is * pi[s])
        @constraint(model, rbar[i, s] >= r[i, s] - d_is * (1 - pi[s]))
    end

    # Solve Phase 1
    optimize!(model)

    pricing_work_units = 0.0
    try
        pricing_work_units = MOI.get(backend(model), Gurobi.ModelAttribute("Work"))
    catch
    end

    # Collect Phase 1 pool solutions
    kept_solutions = Vector{Vector{Float64}}()
    kept_objs = Float64[]

    if use_solution_pool
        opt = backend(model)
        try
            solcount = MOI.get(opt, Gurobi.ModelAttribute("SolCount"))
            max_solutions = min(10, solcount)
            
            for k in 0:max_solutions-1
                MOI.set(opt, MOI.RawOptimizerAttribute("SolutionNumber"), k)
                pool_obj = MOI.get(opt, Gurobi.ModelAttribute("PoolObjVal"))
                x_k = [MOI.get(opt, Gurobi.VariableAttribute("Xn"), JuMP.index(x[i])) for i in 1:nl]
                rc_k = pool_obj + dual_mu
                if rc_k > 1e-6
                    push!(kept_solutions, x_k)
                    push!(kept_objs, pool_obj)
                end
            end
        catch e
            println("  Warning: solution pool access failed: $e")
        end
    end

    if isempty(kept_solutions)
        if return_nodes
            nodes = 0
            try
                nodes = MOI.get(backend(model), Gurobi.ModelAttribute("NodeCount"))
            catch
            end
            return [value.(x)], [objective_value(model)], pricing_work_units, nodes
        else
            return [value.(x)], [objective_value(model)], pricing_work_units
        end
    end

    return return_pricing_payload(kept_solutions, kept_objs, pricing_work_units, return_nodes, model)
end

# =============================================================================
# PRICING SUBPROBLEM (Type 2)
# =============================================================================

function af_pricing_type2_TS(data, scenarios, dual_lambda, dual_mu; use_solution_pool::Bool=true, work_limit::Float64=Inf, opt_worklimit::Float64=DEFAULT_OPT_WORKLIMIT, alpha::Float64=DEFAULT_ALPHA, return_nodes::Bool=false, analysis::Dict{Symbol,Any}=Dict{Symbol,Any}())
    """Solve AF Pricing Type 2 for Transmission Switching using two-phase approach"""
    B_ij, fbar_ij, p_max, p_min, A, c = parameters_TSpm(data)
    L = data["branch"]
    nb = length(data["bus"])
    nl = length(L)
    nscen = length(scenarios)

    # Big-M following instruction: M_ij = -12.6 * B_ij
    M_ij = [-12.6 * B_ij[l] for l in 1:nl]

    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 1)
    if use_solution_pool
        set_optimizer_attribute(model, "PoolSearchMode", 2)
        set_optimizer_attribute(model, "PoolSolutions", 10)
    end

    # Decision variables (Type 2: x and per-scenario x_s linked by pi)
    @variable(model, x[1:nl], Bin)
    @variable(model, x_s[1:nl, 1:nscen], Bin)
    @variable(model, pi[1:nscen], Bin)
    @variable(model, p_g[1:nb, 1:nscen])
    @variable(model, r[1:nb, 1:nscen] >= 0)
    @variable(model, f[1:nl, 1:nscen])
    @variable(model, theta[1:nb, 1:nscen])

    # Objective
    @objective(model, Max, sum(dual_lambda[s] * pi[s] for s in 1:nscen)
                        - sum(c[i] * p_g[i,s] for i in 1:nb for s in 1:nscen)
                        - alpha * sum(r[i,s] for i in 1:nb for s in 1:nscen))


    # Fix lines ON based on capacity utilization file or CSV (top K lines)
    # Check if no_fixed_lines flag is set (all lines switchable)
    if get(data, "no_fixed_lines", false)
        fixed_lines = Int[]  # No lines fixed, all switchable
    else
        num_lines = get(data, "num_lines_fixed", nothing)
        fixed_lines = get_fixed_line_indices(data; num_lines=num_lines)
    end
    for l in fixed_lines
        if 1 <= l <= nl
            @constraint(model, x[l] == 1)
            @constraint(model, [s in 1:nscen], x_s[l, s] == pi[s])
        else
            println("  Warning: Fixed line index $l is out of range (1-$nl)")
        end
    end

    # Linking constraints
    @constraint(model, [l in 1:nl, s in 1:nscen], x_s[l, s] <= pi[s])
    @constraint(model, [l in 1:nl, s in 1:nscen], x[l] - x_s[l, s] <= (1 - pi[s]))
    @constraint(model, [l in 1:nl, s in 1:nscen], x[l] - x_s[l, s] >= -(1 - pi[s]))

    # Generator bounds per scenario
    for i in 1:nb, s in 1:nscen
        @constraint(model, p_g[i, s] >= p_min[i]*pi[s])
        @constraint(model, p_g[i, s] <= p_max[i]*pi[s])
    end

    # Line capacity and DC constraints with x_s
    for (branch_id, branch) in L
        l = parse(Int, branch_id)
        i = branch["f_bus"]
        j = branch["t_bus"]
        for s in 1:nscen
            @constraint(model, -fbar_ij[l] * x_s[l, s] <= f[l, s])
            @constraint(model,  f[l, s] <= fbar_ij[l] * x_s[l, s])
            @constraint(model, B_ij[l] * (theta[i, s] - theta[j, s]) - M_ij[l] * (1 - x_s[l, s]) <= f[l, s])
            @constraint(model, f[l, s] <= B_ij[l] * (theta[i, s] - theta[j, s]) + M_ij[l] * (1 - x_s[l, s]))
        end
    end

    # Power balance per bus and scenario
    for s in 1:nscen
        d_s = scenarios[s]
        for i in 1:nb
            outgoing = 0.0
            incoming = 0.0
            for (branch_id, branch) in L
                l = parse(Int, branch_id)
                if branch["f_bus"] == i
                    outgoing += f[l, s]
                end
                if branch["t_bus"] == i
                    incoming += f[l, s]
                end
            end
            @constraint(model, p_g[i, s] + r[i, s] - d_s[i]*pi[s] == outgoing - incoming)
        end
        @constraint(model, theta[1, s] == 0)
    end

    # Key inequality
    @constraint(model, [s in 1:nscen],
        dual_lambda[s] * pi[s]
        - sum(c[i] * p_g[i, s] for i in 1:nb)
        - alpha * sum(r[i, s] for i in 1:nb) >= 0)

    # Two-phase solve
    pricing_work_units = 0.0
    if !isempty(analysis)
        analysis[:phase1_work] = 0.0
        analysis[:phase2_work] = 0.0
        analysis[:went_phase2] = false
        analysis[:lb] = NaN
        analysis[:ub] = NaN
    end

    # Phase 1
    if isfinite(opt_worklimit) && opt_worklimit > 0.0
        println("  Phase 1: Quick optimal try with $(opt_worklimit) work units")
        set_optimizer_attribute(model, "MIPFocus", 1)
        set_optimizer_attribute(model, "WorkLimit", opt_worklimit)
        optimize!(model)
        
        try
            w = MOI.get(backend(model), Gurobi.ModelAttribute("Work"))
            pricing_work_units += w
            if !isempty(analysis)
                analysis[:phase1_work] = w
            end
        catch
        end
        
        if termination_status(model) == MOI.OPTIMAL
            println("  Phase 1: Optimal solution found")
            kept_solutions, kept_objs = collect_pool_rc_cap10(model, x, dual_mu, use_solution_pool)
            if !isempty(kept_solutions)
                println("  Phase 1: Found $(length(kept_solutions)) RC-improving solutions")
                return return_pricing_payload(kept_solutions, kept_objs, pricing_work_units, return_nodes, model)
            else
                println("  Phase 1: Optimal but no RC-improving solutions — terminating pricing")
                return Vector{Vector{Float64}}(), Float64[], pricing_work_units
            end
        else
            println("  Phase 1: Not optimal - checking pool for RC-improving incumbents")
            kept_solutions, kept_objs = collect_pool_rc_cap10(model, x, dual_mu, use_solution_pool)
            if !isempty(kept_solutions)
                println("  Phase 1: Found $(length(kept_solutions)) RC-improving solutions (non-optimal status)")
                return return_pricing_payload(kept_solutions, kept_objs, pricing_work_units, return_nodes, model)
            else
                println("  Phase 1: No RC-improving solutions found, proceeding to Phase 2")
            end
        end
    end

    # Phase 2
    heur_worklimit = work_limit - opt_worklimit
    if isfinite(heur_worklimit) && heur_worklimit > 0.0
        println("  Phase 2: Optimality push with $(heur_worklimit) work units")
        if !isempty(analysis)
            analysis[:went_phase2] = true
        end
        set_optimizer_attribute(model, "MIPFocus", 0)
        set_optimizer_attribute(model, "WorkLimit", heur_worklimit)
        optimize!(model)
        
        try
            w2 = MOI.get(backend(model), Gurobi.ModelAttribute("Work"))
            pricing_work_units += w2
            if !isempty(analysis)
                analysis[:phase2_work] = w2
            end
        catch
        end
        
        if termination_status(model) == MOI.OPTIMAL
            println("  Phase 2: Optimal solution found")
            try
                lb = MOI.get(backend(model), Gurobi.ModelAttribute("ObjBound"))
                ub = MOI.get(backend(model), Gurobi.ModelAttribute("ObjVal"))
                if !isempty(analysis)
                    analysis[:lb] = lb
                    analysis[:ub] = ub
                end
            catch
            end
            kept_solutions, kept_objs = collect_pool_rc_cap10(model, x, dual_mu, use_solution_pool)
            return return_pricing_payload(kept_solutions, kept_objs, pricing_work_units, return_nodes, model)
        else
            println("  Phase 2: Work units exhausted. Reporting bounds...")
            try
                lb = MOI.get(backend(model), Gurobi.ModelAttribute("ObjBound"))
                ub = MOI.get(backend(model), Gurobi.ModelAttribute("ObjVal"))
                println("  Lower bound: $lb, Upper bound: $ub")
                if !isempty(analysis)
                    analysis[:lb] = lb
                    analysis[:ub] = ub
                end
            catch
            end
            return return_pricing_payload(Vector{Vector{Float64}}(), Float64[], pricing_work_units, return_nodes, model)
        end
    end
    
    # Fallback
    return return_pricing_payload(Vector{Vector{Float64}}(), Float64[], pricing_work_units, return_nodes, model)
end

# =============================================================================
# PRICING SUBPROBLEM WRAPPER (Interface for ColumnGenerationCore)
# =============================================================================

function build_pricing_model_TS(scenarios::Vector{Vector{Float64}}, dual_lambda::Vector{Float64}, dual_mu::Float64, problem_data::Dict, pricing_type::Int, work_limit::Float64, opt_worklimit::Float64, use_solution_pool::Bool, return_nodes::Bool, log_file::Union{String, Nothing}, iteration::Union{Int, Nothing}=nothing)
    """Build and solve pricing subproblem for Transmission Switching
    
    This is the interface function called by ColumnGenerationCore.jl
    """
    nscen = length(scenarios)
    
    # Use instance-specific penalty (consistent with heuristic)
    instance_file = get(problem_data, "instance_file", nothing)
    case_name = instance_file !== nothing ? extract_case_name(instance_file) : nothing
    alpha = get_penalty(instance_file=instance_file, case_name=case_name)
    
    if pricing_type == 1
        # Calculate m^s values
        m_values = calculate_m_values_TS(scenarios, dual_lambda, problem_data)
        
        # Solve Type 1
        return af_pricing_type1_TS(problem_data, scenarios, m_values, dual_lambda, dual_mu; 
                                   use_solution_pool=use_solution_pool, return_nodes=return_nodes, 
                                   work_limit=work_limit, opt_worklimit=opt_worklimit, alpha=alpha)
    elseif pricing_type == 2
        # Check if decomposition is enabled
        use_decomp = get(problem_data, "use_decomp", false)
        
        if use_decomp
            # Solve Type 2 with decomposition
            f_s_vector = get(problem_data, "f_s", nothing)
            if f_s_vector === nothing || length(f_s_vector) != nscen
                error("❌ Error: f_s vector not found or length mismatch. Expected length $nscen, got $(f_s_vector === nothing ? "nothing" : length(f_s_vector))")
            end
            
            # Initialize decomp_cache if it doesn't exist
            if !haskey(problem_data, "ts_decomp_cache")
                problem_data["ts_decomp_cache"] = Dict{String,Any}(
                    "scenario_x_cache" => Dict{Tuple{Int,String}, Dict{String,Any}}(),
                    "records" => Vector{Dict{String,Any}}()
                )
            end
            decomp_cache = problem_data["ts_decomp_cache"]
            
            analysis = Dict{Symbol,Any}()
            return af_pricing_type2_TS_decomp(
                problem_data, scenarios, dual_lambda, dual_mu, f_s_vector, decomp_cache;
                return_nodes=return_nodes, analysis=analysis, log_file=log_file, iteration=iteration
            )
        else
            # Solve Type 2 (standard pricing)
            m_values = zeros(nscen)  # Not used but kept for compatibility
            analysis = Dict{Symbol,Any}()
            return af_pricing_type2_TS(problem_data, scenarios, dual_lambda, dual_mu; 
                                       use_solution_pool=use_solution_pool, return_nodes=return_nodes, 
                                       work_limit=work_limit, opt_worklimit=opt_worklimit, alpha=alpha, analysis=analysis)
        end
    else
        error("Invalid pricing_type. Use 1 or 2.")
    end
end

