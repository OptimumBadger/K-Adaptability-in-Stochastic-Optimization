# FacilityLocationAFDecomposition.jl
# Decomposition-based pricing for Facility Location - Assignment Formulation
# Includes all decomposition helpers and the main af_pricing_type2_FL_decomp function.

include("FacilityLocationAF.jl")

# =============================================================================
# DECOMPOSITION SETTINGS
# =============================================================================

const USE_BENDERS_CUTS = true   # set to false to disable Cut 34 (Conditional Benders)

# =============================================================================
# DECOMPOSITION HELPERS (Type 2)
# =============================================================================

function solve_fl_recourse_lp_with_duals(
    x_hat::Vector{Int},
    scenario::AbstractVector{<:Real},
    capacity::AbstractVector{<:Real},
    cost::AbstractMatrix{<:Real},
    fixedcost::AbstractVector{<:Real},
    unmet_pen::Float64,
    scaling_factor::Float64;
    subproblem_cuts::Bool=true
)
    """Solve FL recourse LP for a fixed x and return objective + duals.

    Returns q_s which includes: fixed costs (for open facilities) + transportation costs + unmet demand penalty.
    subproblem_cuts: if true, add valid inequalities y[i,j] <= min(u_i, d_j) * x_hat[i] to the LP.
    """
    nloc = length(capacity)
    ncust = length(scenario)

    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 0)
    set_optimizer_attribute(model, "LogToConsole", 0)

    @variable(model, y[1:nloc, 1:ncust] >= 0)
    @variable(model, shortfall[1:ncust] >= 0)

    demand_con = @constraint(model, [j in 1:ncust], sum(y[i, j] for i in 1:nloc) + shortfall[j] == scenario[j])
    cap_con = @constraint(model, [i in 1:nloc], sum(y[i, j] for j in 1:ncust) <= capacity[i] * x_hat[i])
    if subproblem_cuts
        valid_con = @constraint(model, [i in 1:nloc, j in 1:ncust], y[i, j] <= min(capacity[i], scenario[j]) * x_hat[i])
    end

    @objective(model, Min,
        scaling_factor * sum(cost[i, j] * y[i, j] for i in 1:nloc for j in 1:ncust) +
        unmet_pen * sum(shortfall[j] for j in 1:ncust)
    )

    optimize!(model)

    ts = termination_status(model)
    if ts != MOI.OPTIMAL
        return Inf, zeros(ncust), zeros(nloc), zeros(nloc, ncust), 0.0
    end

    # Vectorized dual extraction is faster than element-wise extraction.
    mu_d = collect(dual.(demand_con))
    mu_c = collect(dual.(cap_con))
    lambda_v = subproblem_cuts ? collect(dual.(valid_con)) : zeros(nloc, ncust)

    work_units = 0.0
    try
        work_units = MOI.get(backend(model), Gurobi.ModelAttribute("Work"))
    catch
    end

    # q_s includes fixed costs for open facilities + recourse costs
    recourse_obj = objective_value(model)
    fixed_cost_term = sum(fixedcost[i] * x_hat[i] for i in 1:nloc)
    q_s = fixed_cost_term + recourse_obj

    return q_s, mu_d, mu_c, lambda_v, work_units
end

function solve_fl_recourse_lp_with_x_variables(
    x_hat::Vector{Int},
    scenario::AbstractVector{<:Real},
    capacity::AbstractVector{<:Real},
    cost::AbstractMatrix{<:Real},
    fixedcost::AbstractVector{<:Real},
    unmet_pen::Float64,
    scaling_factor::Float64;
    subproblem_cuts::Bool=true
)
    """Solve FL recourse LP with x as continuous variables [0,1] and constraints x[j] = x_hat[j].
    
    This allows us to extract dual values of the fixing constraints to identify which facilities
    can be unfixed (dual = 0).
    
    Returns q_s which includes: fixed costs + transportation costs + unmet demand penalty.
    Also returns dual values of x[j] = x_hat[j] constraints.
    """
    nloc = length(capacity)
    ncust = length(scenario)

    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 0)
    set_optimizer_attribute(model, "LogToConsole", 0)

    # Variables: x as continuous [0,1] instead of fixed constants
    @variable(model, 0 <= x[1:nloc] <= 1)
    @variable(model, y[1:nloc, 1:ncust] >= 0)
    @variable(model, shortfall[1:ncust] >= 0)

    # Constraints: x[j] = x_hat[j] for all j (to fix x values)
    x_fix_con = @constraint(model, [j in 1:nloc], x[j] == x_hat[j])

    demand_con = @constraint(model, [j in 1:ncust], sum(y[i, j] for i in 1:nloc) + shortfall[j] == scenario[j])
    cap_con = @constraint(model, [i in 1:nloc], sum(y[i, j] for j in 1:ncust) <= capacity[i] * x[i])
    if subproblem_cuts
        @constraint(model, [i in 1:nloc, j in 1:ncust], y[i, j] <= min(capacity[i], scenario[j]) * x[i])
    end

    @objective(model, Min,
        sum(fixedcost[i] * x[i] for i in 1:nloc) +
        scaling_factor * sum(cost[i, j] * y[i, j] for i in 1:nloc for j in 1:ncust) +
        unmet_pen * sum(shortfall[j] for j in 1:ncust)
    )

    optimize!(model)

    ts = termination_status(model)
    if ts != MOI.OPTIMAL
        return Inf, zeros(nloc), zeros(ncust), zeros(nloc), 0.0
    end

    # Vectorized dual extraction is faster than element-wise extraction.
    x_duals = collect(dual.(x_fix_con))
    mu_d = collect(dual.(demand_con))
    mu_c = collect(dual.(cap_con))

    work_units = 0.0
    try
        work_units = MOI.get(backend(model), Gurobi.ModelAttribute("Work"))
    catch
    end

    # q_s includes fixed costs + recourse costs
    q_s = objective_value(model)

    return q_s, x_duals, mu_d, mu_c, work_units
end

function solve_fl_partial_recourse_mip(
    F0::Vector{Int},
    F1::Vector{Int},
    scenario::AbstractVector{<:Real},
    capacity::AbstractVector{<:Real},
    cost::AbstractMatrix{<:Real},
    fixedcost::AbstractVector{<:Real},
    unmet_pen::Float64,
    scaling_factor::Float64;
    subproblem_cuts::Bool=true
)
    """Solve FL recourse MIP with partial fixing (formulation 36).
    
    Solves: g_s(F0, F1) = min (c_s)^T x + (d_s)^T y
    subject to:
    - x_j = 1 for j ∈ F1 (forced open)
    - x_j = 0 for j ∈ F0 (forced closed)
    - x_j ∈ {0,1} for j ∉ (F1 ∪ F0) (free binary)
    - Capacity and demand constraints
    
    Args:
        F0: Indices where x_j = 0 (forced closed)
        F1: Indices where x_j = 1 (forced open)
        scenario: Demand scenario vector
        capacity: Facility capacity vector
        cost: Transportation cost matrix
        fixedcost: Fixed cost vector
        unmet_pen: Unmet demand penalty
        scaling_factor: Scaling factor for transportation cost
    
    Returns:
        (g_s::Float64, work_units::Float64)
        g_s includes fixed costs + transportation costs + unmet demand penalty
    """
    nloc = length(capacity)
    ncust = length(scenario)
    
    # Validate F0 and F1 are disjoint
    if !isempty(intersect(Set(F0), Set(F1)))
        error("F0 and F1 must be disjoint")
    end
    
    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 0)
    set_optimizer_attribute(model, "LogToConsole", 0)
    set_optimizer_attribute(model, "MIPGap", 0.001)
    # No WorkLimit as requested
    
    # Variables
    @variable(model, x[1:nloc], Bin)  # Binary variables
    @variable(model, y[1:nloc, 1:ncust] >= 0)
    @variable(model, shortfall[1:ncust] >= 0)
    
    # Partial fixing constraints
    for j in F1
        @constraint(model, x[j] == 1)  # Force open
    end
    for j in F0
        @constraint(model, x[j] == 0)  # Force closed
    end
    
    # Capacity constraints
    for i in 1:nloc
        @constraint(model, sum(y[i, j] for j in 1:ncust) <= capacity[i] * x[i])
    end
    
    # Demand constraints
    for j in 1:ncust
        @constraint(model, sum(y[i, j] for i in 1:nloc) + shortfall[j] == scenario[j])
    end
    # Valid inequalities: y_ij <= min(u_i, d_j) * x_i
    if subproblem_cuts
        for i in 1:nloc, j in 1:ncust
            @constraint(model, y[i, j] <= min(capacity[i], scenario[j]) * x[i])
        end
    end
    
    # Objective: includes fixed costs
    @objective(model, Min,
        sum(fixedcost[i] * x[i] for i in 1:nloc) +
        scaling_factor * sum(cost[i, j] * y[i, j] for i in 1:nloc for j in 1:ncust) +
        unmet_pen * sum(shortfall[j] for j in 1:ncust)
    )
    
    optimize!(model)
    
    work_units = 0.0
    try
        work_units = MOI.get(backend(model), Gurobi.ModelAttribute("Work"))
    catch
    end
    
    ts = termination_status(model)
    if ts != MOI.OPTIMAL
        return Inf, work_units
    end
    
    g_s = objective_value(model)
    return g_s, work_units
end

function solve_fl_partial_recourse_lp_with_x_duals(
    F0::Vector{Int},
    F1::Vector{Int},
    scenario::AbstractVector{<:Real},
    capacity::AbstractVector{<:Real},
    cost::AbstractMatrix{<:Real},
    fixedcost::AbstractVector{<:Real},
    unmet_pen::Float64,
    scaling_factor::Float64;
    subproblem_cuts::Bool=true
)
    """Solve FL recourse LP with only F0/F1 variables fixed via equality constraints.

    x[i] ∈ [0,1] for all i. Equality constraints x[i]=0 (for i∈F0) and x[i]=1 (for i∈F1)
    are added so their duals can be retrieved. Variables not in F0∪F1 are free in [0,1].

    Returns: (q_s, x_duals::Dict{Int,Float64}, mu_d, mu_c, work_units)
        x_duals: dual values of fixing constraints, keyed by facility index
    """
    nloc = length(capacity)
    ncust = length(scenario)

    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 0)
    set_optimizer_attribute(model, "LogToConsole", 0)

    @variable(model, 0 <= x[1:nloc] <= 1)
    @variable(model, y[1:nloc, 1:ncust] >= 0)
    @variable(model, shortfall[1:ncust] >= 0)

    x_con = Dict{Int, ConstraintRef}()
    for i in F0
        x_con[i] = @constraint(model, x[i] == 0)
    end
    for i in F1
        x_con[i] = @constraint(model, x[i] == 1)
    end

    demand_con = @constraint(model, [j in 1:ncust], sum(y[i, j] for i in 1:nloc) + shortfall[j] == scenario[j])
    cap_con    = @constraint(model, [i in 1:nloc],  sum(y[i, j] for j in 1:ncust) <= capacity[i] * x[i])
    if subproblem_cuts
        @constraint(model, [i in 1:nloc, j in 1:ncust], y[i, j] <= min(capacity[i], scenario[j]) * x[i])
    end

    @objective(model, Min,
        sum(fixedcost[i] * x[i] for i in 1:nloc) +
        scaling_factor * sum(cost[i, j] * y[i, j] for i in 1:nloc for j in 1:ncust) +
        unmet_pen * sum(shortfall[j] for j in 1:ncust)
    )

    optimize!(model)

    work_units = 0.0
    try
        work_units = MOI.get(backend(model), Gurobi.ModelAttribute("Work"))
    catch
    end

    if termination_status(model) != MOI.OPTIMAL
        return Inf, Dict{Int,Float64}(), zeros(ncust), zeros(nloc), work_units
    end

    x_duals = Dict{Int,Float64}(i => dual(x_con[i]) for i in keys(x_con))
    q_s  = objective_value(model)
    mu_d = collect(dual.(demand_con))
    mu_c = collect(dual.(cap_con))

    return q_s, x_duals, mu_d, mu_c, work_units
end

function select_random_F0_F1(x_hat::Vector{Int})
    """Randomly select F1 (subset of indices where x_hat[i] == 1) and F0 (subset of indices where x_hat[i] == 0).
    
    Args:
        x_hat: Binary vector
    
    Returns:
        (F0::Vector{Int}, F1::Vector{Int})
    """
    nloc = length(x_hat)
    
    # Find indices where x_hat[i] == 1 (candidates for F1)
    indices_ones = [i for i in 1:nloc if x_hat[i] == 1]
    # Find indices where x_hat[i] == 0 (candidates for F0)
    indices_zeros = [i for i in 1:nloc if x_hat[i] == 0]
    
    # Randomly select subsets (random sizes, no minimum)
    F1 = Int[]
    if !isempty(indices_ones)
        # Random size between 1 and length(indices_ones)
        size_F1 = rand(1:length(indices_ones))
        F1 = sort(shuffle(indices_ones)[1:size_F1])
    end
    
    F0 = Int[]
    if !isempty(indices_zeros)
        # Random size between 1 and length(indices_zeros)
        size_F0 = rand(1:length(indices_zeros))
        F0 = sort(shuffle(indices_zeros)[1:size_F0])
    end
    
    return F0, F1
end

function compute_M_value_FL(
    mu_d::Vector{Float64},
    mu_c::Vector{Float64},
    lambda_v::AbstractMatrix{<:Real},
    demand_s::AbstractVector{<:Real},
    fixedcost::AbstractVector{<:Real},
    capacity::AbstractVector{<:Real},
    dual_lambda_s::Float64
)
    """Compute M_s(mu) for FL by solving the optimization problem:
    M(μ̂) = max_{x_i ∈ {0,1}} [Σ_{i∈I} f_i x_i + Σ_{j∈J} d_j^s μ̂_{d,j} + Σ_{i∈I} (u_i x_i) μ̂_{c,i}
                                + Σ_{i∈I} Σ_{j∈J} min(u_i, d_j^s) x_i λ̂_{v,i,j} - λ_s^*]

    This simplifies to (separable in x_i):
    M(μ̂) = Σ_{j∈J} d_j^s μ̂_{d,j} - λ_s^* + Σ_{i∈I} max(0, f_i + u_i μ̂_{c,i} + Σ_{j∈J} min(u_i, d_j^s) λ̂_{v,i,j})
    """
    nloc = length(fixedcost)
    ncust = length(demand_s)

    # Constant term (independent of x_i)
    constant_term = sum(demand_s[j] * mu_d[j] for j in 1:ncust) - dual_lambda_s

    # For each facility i, compute coefficient:
    # f_i + u_i * μ̂_{c,i} + Σ_j min(u_i, d_j^s) * λ̂_{v,i,j}
    # If coefficient > 0, set x_i = 1; otherwise x_i = 0
    M_value = constant_term
    for i in 1:nloc
        valid_coeff = sum(min(capacity[i], demand_s[j]) * lambda_v[i, j] for j in 1:ncust)
        coefficient = fixedcost[i] + capacity[i] * mu_c[i] + valid_coeff
        if coefficient > 0.0
            M_value += coefficient
        end
    end

    return M_value
end

function compute_M_value_FL_opt(
    mu_d::Vector{Float64},
    mu_c::Vector{Float64},
    lambda_v::AbstractMatrix{<:Real},
    demand_s::AbstractVector{<:Real},
    fixedcost::AbstractVector{<:Real},
    capacity::AbstractVector{<:Real},
    dual_lambda_s::Float64,
    a_dmin::Vector{Float64},
    const_dmin::Float64,
    max_q::Float64
)
    """Compute M_s via MIP with the d_min domination constraint.
    
    """
    nloc  = length(fixedcost)
    ncust = length(demand_s)

    # Objective: use scenario-s LP duals 
    a_current     = [fixedcost[i] + capacity[i] * mu_c[i] +
                      sum(min(capacity[i], demand_s[j]) * lambda_v[i, j] for j in 1:ncust)
                     for i in 1:nloc]
    const_current = sum(demand_s[j] * mu_d[j] for j in 1:ncust)

    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 0)
    set_optimizer_attribute(model, "LogToConsole", 0)

    @variable(model, x[1:nloc], Bin)
    @objective(model, Max, sum(a_current[i] * x[i] for i in 1:nloc) + const_current - dual_lambda_s)

    # d_min domination constraint
    if isfinite(max_q)
        @constraint(model, sum(a_dmin[i] * x[i] for i in 1:nloc) <= max_q - const_dmin)
    end

    optimize!(model)

    mip_work = 0.0
    try
        mip_work = MOI.get(backend(model), Gurobi.ModelAttribute("Work"))
    catch
    end

    if termination_status(model) == MOI.OPTIMAL
        return objective_value(model), mip_work
    end

    # Fallback: analytical formula (unconstrained max over binary x)
    M_value = const_current - dual_lambda_s
    for i in 1:nloc
        if a_current[i] > 0.0
            M_value += a_current[i]
        end
    end
    return M_value, mip_work
end

function x_signature_FL(x_hat::Vector{Int})
    return join(x_hat, ",")
end

function af_pricing_type2_FL_decomp(
    nloc,
    ncust,
    capacity,
    cost,
    scenarios,
    dual_lambda,
    dual_mu,
    unmet_pen,
    scaling_factor,
    fixedcost,
    f_s_vector,
    decomp_cache::Dict{String,Any},
    problem_data::Dict;
    return_nodes::Bool=false,
    analysis::Dict{Symbol,Any}=Dict{Symbol,Any}(),
    log_file=nothing,
    iteration=nothing,
    max_cut_iters::Int=500,
    tol::Float64=1e-4,
    use_simple_decomp::Bool=false,
    decomp_mode::Int=2,
    subproblem_cuts::Bool=true,
    use_mip_bigM::Bool=true,
    use_benders_cuts::Bool=true
)
    """Solve AF Pricing Type 2 with decomposition cutting-plane procedure."""
    nscen = length(scenarios)
    dual_zero_tol = 1e-6
    if length(f_s_vector) != nscen
        error("❌ Error: f_s vector length ($(length(f_s_vector))) must equal number of scenarios ($nscen).")
    end

    # Set random seed for reproducibility
    Random.seed!(42)

    println("\n" * "="^80)
    println("DECOMPOSITION ALGORITHM - PRICING TYPE 2")
    println("="^80)
    println("  Using f_s vector (wait-and-see costs) from WS file")
    println("  Number of scenarios: $nscen")
    println("  Number of facilities: $nloc")
    println("  Max cutting-plane iterations: $max_cut_iters")
    println("  Tolerance: $tol")
    mode_str = if decomp_mode == 1
        "SIMPLE (LP-based cuts)"
    elseif decomp_mode == 2
        "DUAL-BASED UNFIXING (partial MIP)"
    elseif decomp_mode == 3
        "PARTIAL MIP (F1=open facilities, F0=empty)"
    elseif decomp_mode == 4
        "PARTIAL MIP (Random subsets of F0 and F1)"
    elseif decomp_mode == 5
        "GAP-BASED DUAL UNFIXING (unfix until sum|dual| < gap)"
    else  # decomp_mode == 6
        "ITERATIVE SINGLE-VARIABLE UNFIXING (LP duals + partial MIP cuts)"
    end
    println("  Mode: $mode_str")
    println("="^80)

    # Write to log file if provided
    if log_file !== nothing && iteration !== nothing
        open(log_file, "a") do file
            println(file, "\n" * "="^80)
            println(file, "PRICING SUBPROBLEM TYPE 2 (DECOMPOSITION) - ITERATION $iteration")
            println(file, "="^80)
            println(file, "Timestamp: $(now())")
            println(file, "Problem size: $nscen scenarios, $nloc facilities")
            println(file, "Max cutting-plane iterations: $max_cut_iters")
            println(file, "Tolerance: $tol")
            println(file, "="^80)
        end
    end

    println("\nBuilding initial master problem...")
    master = Model(Gurobi.Optimizer)
    set_optimizer_attribute(master, "OutputFlag", 1)  # Enable Gurobi output for master problem
    set_optimizer_attribute(master, "LogToConsole", 1)  # Enable console logging for master problem

    @variable(master, x[1:nloc], Bin)
    @variable(master, pi[1:nscen], Bin)
    @variable(master, theta[1:nscen] >= 0)

    @objective(master, Max, sum(theta[s] for s in 1:nscen))
    @constraint(master, [s in 1:nscen], theta[s] <= (dual_lambda[s] - f_s_vector[s]) * pi[s])
    println("  Master problem built: max Σ θ_s, with base constraints θ_s ≤ (λ_s^* - f_s)π_s")

    # Initialize pricing_work_units for tracking
    pricing_work_units = 0.0

    # Add cuts from initial binary vectors (for every CG iteration)(Right now this portion is disabled to avoid adding cuts from initial binary vectors)
    initial_vectors = get(problem_data, "initial_binary_vectors", nothing)
    if initial_vectors !== nothing && !isempty(initial_vectors)
        println("\nAdding cuts from $(length(initial_vectors)) initial binary vectors...")
        if log_file !== nothing && iteration !== nothing
            open(log_file, "a") do file
                println(file, "\nAdding cuts from $(length(initial_vectors)) initial binary vectors...")
            end
        end
        
        initial_cut32_count = 0
        initial_cut33_count = 0
        initial_cut34_count = 0
        initial_pricing_work_units = 0.0
        
        for (vec_idx, v) in enumerate(initial_vectors)
            # Initial deterministic selection: F1 = all open facilities, F0 = all closed facilities
            F1_initial = [i for i in 1:nloc if v[i] == 1]  # All indices where v[i] == 1
            F0_initial = [i for i in 1:nloc if v[i] == 0]  # All indices where v[i] == 0
            
            if vec_idx == 1 || vec_idx % 50 == 0
                println("  Processing initial vector $vec_idx: Initial F1 = $(length(F1_initial)), F0 = $(length(F0_initial))")
            end
            
            # Process each scenario: solve LP, update F0/F1 based on duals, then solve MIP
            for s in 1:nscen
                # Step 1: Solve LP with x as variables to get dual information
                q_s, x_duals, mu_d, mu_c, lp_work = solve_fl_recourse_lp_with_x_variables(
                    v, scenarios[s], capacity, cost, fixedcost, unmet_pen, scaling_factor;
                    subproblem_cuts=subproblem_cuts
                )
                initial_pricing_work_units += lp_work
                
                if !isfinite(q_s)
                    continue  # Skip if infeasible/unbounded
                end
                
                # Step 2: Find near-zero dual indices using tolerance (avoid exact equality).
                zero_dual_indices = [i for i in 1:nloc if abs(x_duals[i]) <= dual_zero_tol]
                
                # Step 3: Update F1 and F0 by removing indices with zero duals
                F1_partial = [i for i in F1_initial if !(i in zero_dual_indices)]
                F0_partial = [i for i in F0_initial if !(i in zero_dual_indices)]
                
                # Step 4: Compute delta_expr for updated F0/F1
                delta_expr_partial = sum(1 - x[i] for i in F1_partial; init=0.0) + sum(x[i] for i in F0_partial; init=0.0)
                
                # Step 5: Solve partial MIP recourse with updated F0/F1
                g_s, mip_work = solve_fl_partial_recourse_mip(
                    F0_partial, F1_partial, scenarios[s], capacity, cost, fixedcost, unmet_pen, scaling_factor;
                    subproblem_cuts=subproblem_cuts
                )
                initial_pricing_work_units += mip_work
                
                if !isfinite(g_s)
                    continue  # Skip if infeasible/unbounded
                end
                
                # Print q_s, g_s, and dual_lambda[s] for this scenario
                println("        Vector $vec_idx, Scenario $s: q_s = $(round(q_s, digits=2)), g_s(F0,F1) = $(round(g_s, digits=2)), λ_s^* = $(round(dual_lambda[s], digits=2))")
                
                # Step 6: Generate cuts based on g_s vs dual_lambda[s]
                if g_s >= dual_lambda[s]
                    # Conditional no-good cut
                    @constraint(master, delta_expr_partial >= pi[s])
                    initial_cut32_count += 1
                else
                    # Integer L-shaped cut
                    @constraint(master, theta[s] <= dual_lambda[s] - g_s + (g_s - f_s_vector[s]) * delta_expr_partial)
                    initial_cut33_count += 1
                end
            end
            
            # Still generate Cut 34 using fully fixed v (current approach)
            for s in 1:nscen
                # Solve recourse LP to get q_s and duals for Cut 34
                q_s, mu_d, mu_c, lambda_v, sub_work = solve_fl_recourse_lp_with_duals(
                    v, scenarios[s], capacity, cost, fixedcost, unmet_pen, scaling_factor;
                    subproblem_cuts=subproblem_cuts
                )
                initial_pricing_work_units += sub_work

                if !isfinite(q_s)
                    continue  # Skip if infeasible/unbounded
                end

                # Add Cut 34: Compute M_s and add cut
                M_s = compute_M_value_FL(mu_d, mu_c, lambda_v, scenarios[s], fixedcost, capacity, dual_lambda[s])
                @constraint(master,
                    theta[s] +
                    sum(fixedcost[i] * x[i] for i in 1:nloc) +
                    sum(scenarios[s][j] * mu_d[j] for j in 1:ncust) +
                    sum((capacity[i] * x[i]) * mu_c[i] for i in 1:nloc) +
                    sum(min(capacity[i], scenarios[s][j]) * x[i] * lambda_v[i, j] for i in 1:nloc, j in 1:ncust)
                    <= dual_lambda[s] + M_s * (1 - pi[s])
                )
                initial_cut34_count += 1
            end
        end
        
        println("  Added from initial vectors: $initial_cut32_count Cut 32, $initial_cut33_count Cut 33, $initial_cut34_count Cut 34")
        println("  Work units for initial cuts: $(round(initial_pricing_work_units, digits=2))")
        pricing_work_units += initial_pricing_work_units
        
        if log_file !== nothing && iteration !== nothing
            open(log_file, "a") do file
                println(file, "  Added from initial vectors: $initial_cut32_count Cut 32, $initial_cut33_count Cut 33, $initial_cut34_count Cut 34")
                println(file, "  Work units for initial cuts: $(round(initial_pricing_work_units, digits=2))")
            end
        end
    else
        println("  No initial binary vectors provided for decomposition cuts")
    end

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
            mu_d = rec["mu_d"]::Vector{Float64}
            mu_c = rec["mu_c"]::Vector{Float64}
            lambda_v = rec["lambda_v"]::Matrix{Float64}

            # Recompute M_s with current dual_lambda[s] (since M_s depends on it)
            M_s = compute_M_value_FL(mu_d, mu_c, lambda_v, scenarios[s], fixedcost, capacity, dual_lambda[s])

            # Compute F0 and F1 sets for delta_expr
            F0 = [i for i in 1:nloc if x_hat[i] == 0]
            F1 = [i for i in 1:nloc if x_hat[i] == 1]
            delta_expr = sum(1 - x[i] for i in F1; init=0.0) + sum(x[i] for i in F0; init=0.0)

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

            # Cut 34: Always preload (it's valid regardless of termination condition)
            @constraint(master,
                theta[s] +
                sum(fixedcost[i] * x[i] for i in 1:nloc) +
                sum(scenarios[s][j] * mu_d[j] for j in 1:ncust) +
                sum((capacity[i] * x[i]) * mu_c[i] for i in 1:nloc) +
                sum(min(capacity[i], scenarios[s][j]) * x[i] * lambda_v[i, j] for i in 1:nloc, j in 1:ncust)
                <= dual_lambda[s] + M_s * (1 - pi[s])
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

    pi_fixed = falses(nscen)
    last_x_hat = zeros(Int, nloc)
    last_obj = -Inf

    # Track unique x_hats seen across decomposition iterations (reset each pricing call)
    unique_x_hats = Set{String}()

    # Track decomposition metrics
    decomp_iterations = 0
    decomp_work_units = 0.0
    total_cut32 = 0  # Total Conditional No-Good Cuts across all iterations
    total_cut33 = 0  # Total Integer L-Shaped Cuts across all iterations
    total_cut34 = 0  # Total Conditional Benders Cuts across all iterations

    println("\nStarting cutting-plane iterations...")
    if log_file !== nothing && iteration !== nothing
        open(log_file, "a") do file
            println(file, "\nStarting cutting-plane iterations...")
        end
    end

    # ── Decomposition table file (Mode 6 only) — one file per pricing call ───
    decomp_table_file = nothing
    if decomp_mode == 6
        decomp_table_dir = joinpath(@__DIR__, "logs", "Decomposition")
        if !isdir(decomp_table_dir)
            mkpath(decomp_table_dir)
        end
        run_tag = if log_file !== nothing
            splitext(basename(log_file))[1]
        else
            "run_$(Dates.format(now(), "YYYYmmdd_HHMMSS"))"
        end
        cg_iter_tag = iteration !== nothing ? "_CG$(iteration)" : ""
        decomp_table_file = joinpath(decomp_table_dir, "$(run_tag)$(cg_iter_tag)_tables.txt")
        open(decomp_table_file, "w") do tfile
            println(tfile, "="^100)
            println(tfile, "MODE 6 DECOMPOSITION TABLES  |  CG Iteration: $(iteration !== nothing ? iteration : "N/A")")
            println(tfile, "="^100)
        end
    end

    for it in 1:max_cut_iters
        decomp_iterations = it  # Track number of iterations completed
        decomp_iterations = it  # Track number of iterations completed
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

        x_hat = [round(Int, value(x[i])) for i in 1:nloc]
        pi_hat = [round(Int, value(pi[s])) for s in 1:nscen]
        theta_hat = value.(theta)
        last_x_hat = x_hat
        last_obj = objective_value(master)
        
        num_open = sum(x_hat)
        num_active_scenarios = sum(pi_hat)
        push!(unique_x_hats, x_signature_FL(x_hat))

        println("  Master solved: obj = $(round(last_obj, digits=4))")
        println("  Current solution: $num_open facilities open, $num_active_scenarios scenarios active (π=1)")
        println("  x_hat: $x_hat")
        println("  pi:    $pi_hat")
        println("  Unique x_hats so far: $(length(unique_x_hats))")
        if log_file !== nothing && iteration !== nothing
            open(log_file, "a") do file
                println(file, "  Master solved: obj = $(round(last_obj, digits=4))")
                println(file, "  Current solution: $num_open facilities open, $num_active_scenarios scenarios active (π=1)")
                println(file, "  x_hat: $x_hat")
                println(file, "  pi:    $pi_hat")
                println(file, "  Unique x_hats so far: $(length(unique_x_hats))")
            end
        end

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

        added_cut34 = false
        cut32_new = 0  # Conditional no-good cuts
        cut33_new = 0  # Integer L-shaped cuts
        cut34_new = 0
        lp_work_units = 0.0

        if use_simple_decomp
            # SIMPLE DECOMPOSITION: Solve LP with fully fixed x_hat, generate cuts based on q_s
            println("  Processing scenarios with SIMPLE decomposition (LP-based cuts)...")
            if log_file !== nothing && iteration !== nothing
                open(log_file, "a") do file
                    println(file, "  Processing scenarios with SIMPLE decomposition (LP-based cuts)...")
                end
            end
            
            for s in 1:nscen
                # Solve LP recourse with x_hat fully fixed
                print("    Scenario $s: Solving LP recourse (fully fixed x_hat)... ")
                q_s, mu_d, mu_c, _lambda_v, lp_work = solve_fl_recourse_lp_with_duals(
                    x_hat, scenarios[s], capacity, cost, fixedcost, unmet_pen, scaling_factor;
                    subproblem_cuts=subproblem_cuts
                )
                lp_work_units += lp_work
                pricing_work_units += lp_work
                
                if !isfinite(q_s)
                    println("LP infeasible/unbounded")
                    continue
                end
                
                println("q_s = $(round(q_s, digits=2))")
                
                # Print q_s and dual_lambda[s] for this scenario
                println("      Values: q_s = $(round(q_s, digits=2)), λ_s^* = $(round(dual_lambda[s], digits=2)), f_s = $(round(f_s_vector[s], digits=2))")
                if log_file !== nothing && iteration !== nothing
                    open(log_file, "a") do file
                        println(file, "      Values: q_s = $(round(q_s, digits=2)), λ_s^* = $(round(dual_lambda[s], digits=2)), f_s = $(round(f_s_vector[s], digits=2))")
                    end
                end
                
                # Generate cuts based on q_s vs dual_lambda[s]
                if dual_lambda[s] - q_s <= 0
                    # Conditional no-good cut: Σ_{j: x_hat[j]=1} (1-x_j) + Σ_{j: x_hat[j]=0} x_j >= π_s
                    delta_expr = sum(1 - x[j] for j in 1:nloc if x_hat[j] == 1; init=0.0) + 
                                  sum(x[j] for j in 1:nloc if x_hat[j] == 0; init=0.0)
                    @constraint(master, delta_expr >= pi[s])
                    cut_msg = "    Scenario $s: Added Conditional No-Good Cut [q_s = $(round(q_s, digits=2)) >= λ_s^* = $(round(dual_lambda[s], digits=2))]"
                    println(cut_msg)
                    if log_file !== nothing && iteration !== nothing
                        open(log_file, "a") do file
                            println(file, cut_msg)
                        end
                    end
                    cut32_new += 1
                else
                    # Integer L-shaped cut: θ_s <= λ_s^* - q_s + (Σ_{j: x_hat[j]=1} (1-x_j) + Σ_{j: x_hat[j]=0} x_j)(q_s - f_s)
                    delta_expr = sum(1 - x[j] for j in 1:nloc if x_hat[j] == 1; init=0.0) + 
                                  sum(x[j] for j in 1:nloc if x_hat[j] == 0; init=0.0)
                    @constraint(master, theta[s] <= dual_lambda[s] - q_s + (q_s - f_s_vector[s]) * delta_expr)
                    cut_msg = "    Scenario $s: Added Integer L-Shaped Cut [q_s = $(round(q_s, digits=2)) < λ_s^* = $(round(dual_lambda[s], digits=2))]"
                    println(cut_msg)
                    if log_file !== nothing && iteration !== nothing
                        open(log_file, "a") do file
                            println(file, cut_msg)
                        end
                    end
                    cut33_new += 1
                end
            end
            
            println("  LP work units: $(round(lp_work_units, digits=2))")
        elseif decomp_mode == 3
            # MODE 3: F1 = open facilities, F0 = empty set, solve partial MIP
            F1_initial = [j for j in 1:nloc if x_hat[j] == 1]  # All indices where x_hat[j] == 1
            F0_initial = Int[]  # Empty set
            println("  Selection: F1 (forced open) = $(length(F1_initial)) facilities, F0 (forced closed) = 0 facilities (empty)")
            if log_file !== nothing && iteration !== nothing
                open(log_file, "a") do file
                    println(file, "  Selection: F1 = $F1_initial, F0 = [] (empty)")
                    println(file, "  Processing scenarios with partial MIP (F1=open, F0=empty)...")
                end
            end

            partial_mip_work_units = 0.0
            println("  Processing scenarios with partial MIP (F1=open, F0=empty)...")
            
            # Process each scenario: solve partial MIP with F1=open facilities, F0=empty
            for s in 1:nscen
                # Compute delta_expr for F0/F1
                delta_expr_partial = sum(1 - x[j] for j in F1_initial; init=0.0)  # F0 is empty, so no x[j] terms
                
                # Solve partial MIP recourse with F1=open facilities, F0=empty
                print("    Scenario $s: Solving partial MIP (F1=$F1_initial, F0=[])... ")
                g_s, mip_work = solve_fl_partial_recourse_mip(
                    F0_initial, F1_initial, scenarios[s], capacity, cost, fixedcost, unmet_pen, scaling_factor;
                    subproblem_cuts=subproblem_cuts
                )
                partial_mip_work_units += mip_work
                pricing_work_units += mip_work
                
                if !isfinite(g_s)
                    println("MIP infeasible/unbounded")
                    continue
                end
                
                println("g_s = $(round(g_s, digits=2))")
                
                # Print g_s and dual_lambda[s] for this scenario
                println("      Values: g_s(F0,F1) = $(round(g_s, digits=2)), λ_s^* = $(round(dual_lambda[s], digits=2)), f_s = $(round(f_s_vector[s], digits=2))")
                if log_file !== nothing && iteration !== nothing
                    open(log_file, "a") do file
                        println(file, "      Values: g_s(F0,F1) = $(round(g_s, digits=2)), λ_s^* = $(round(dual_lambda[s], digits=2)), f_s = $(round(f_s_vector[s], digits=2))")
                    end
                end
                
                # Generate cuts based on g_s vs dual_lambda[s]
                if g_s >= dual_lambda[s]
                    # Conditional no-good cut: Σ_{j∈F_1} (1-x_j) >= π_s
                    @constraint(master, delta_expr_partial >= pi[s])
                    cut_msg = "    Scenario $s: Added Partial No-Good Cut [g_s(F0,F1) = $(round(g_s, digits=2)) >= λ_s^* = $(round(dual_lambda[s], digits=2))]"
                    println(cut_msg)
                    if log_file !== nothing && iteration !== nothing
                        open(log_file, "a") do file
                            println(file, cut_msg)
                        end
                    end
                    cut32_new += 1
                else
                    # Integer L-shaped cut: θ_s <= λ_s^* - g_s(F0,F1) + (Σ_{j∈F_1} (1-x_j))(g_s(F0,F1) - f_s)
                    @constraint(master, theta[s] <= dual_lambda[s] - g_s + (g_s - f_s_vector[s]) * delta_expr_partial)
                    cut_msg = "    Scenario $s: Added Partial Integer L-Shaped Cut [g_s(F0,F1) = $(round(g_s, digits=2)) < λ_s^* = $(round(dual_lambda[s], digits=2))]"
                    println(cut_msg)
                    if log_file !== nothing && iteration !== nothing
                        open(log_file, "a") do file
                            println(file, cut_msg)
                        end
                    end
                    cut33_new += 1
                end
            end
            
            println("  Partial MIP work units: $(round(partial_mip_work_units, digits=2))")
        elseif decomp_mode == 4
            # MODE 4: Random subsets of F0 and F1, solve partial MIP
            # Get all indices where x_hat[j] == 1 and x_hat[j] == 0
            all_F1_indices = [j for j in 1:nloc if x_hat[j] == 1]  # All indices where x_hat[j] == 1
            all_F0_indices = [j for j in 1:nloc if x_hat[j] == 0]  # All indices where x_hat[j] == 0
            
            # Randomly select subsets (for now, use 50% of each, but at least 1 if available)
            size_F1 = max(1, div(length(all_F1_indices), 2))
            size_F0 = max(1, div(length(all_F0_indices), 2))
            
            # Shuffle and take random subsets
            F1_random = if !isempty(all_F1_indices)
                sort(shuffle(all_F1_indices)[1:min(size_F1, length(all_F1_indices))])
            else
                Int[]
            end
            F0_random = if !isempty(all_F0_indices)
                sort(shuffle(all_F0_indices)[1:min(size_F0, length(all_F0_indices))])
            else
                Int[]
            end
            
            println("  Random selection: F1 (forced open) = $(length(F1_random)) facilities, F0 (forced closed) = $(length(F0_random)) facilities")
            if log_file !== nothing && iteration !== nothing
                open(log_file, "a") do file
                    println(file, "  Random selection: F1 = $F1_random, F0 = $F0_random")
                    println(file, "  Processing scenarios with partial MIP (random F0/F1)...")
                end
            end

            partial_mip_work_units = 0.0
            println("  Processing scenarios with partial MIP (random F0/F1)...")
            
            # Process each scenario: solve partial MIP with random F0/F1
            for s in 1:nscen
                # Compute delta_expr for F0/F1
                delta_expr_partial = sum(1 - x[j] for j in F1_random; init=0.0) + sum(x[j] for j in F0_random; init=0.0)
                
                # Solve partial MIP recourse with random F0/F1
                print("    Scenario $s: Solving partial MIP (F0=$F0_random, F1=$F1_random)... ")
                g_s, mip_work = solve_fl_partial_recourse_mip(
                    F0_random, F1_random, scenarios[s], capacity, cost, fixedcost, unmet_pen, scaling_factor;
                    subproblem_cuts=subproblem_cuts
                )
                partial_mip_work_units += mip_work
                pricing_work_units += mip_work
                
                if !isfinite(g_s)
                    println("MIP infeasible/unbounded")
                    continue
                end
                
                println("g_s = $(round(g_s, digits=2))")
                
                # Print g_s and dual_lambda[s] for this scenario
                println("      Values: g_s(F0,F1) = $(round(g_s, digits=2)), λ_s^* = $(round(dual_lambda[s], digits=2)), f_s = $(round(f_s_vector[s], digits=2))")
                if log_file !== nothing && iteration !== nothing
                    open(log_file, "a") do file
                        println(file, "      Values: g_s(F0,F1) = $(round(g_s, digits=2)), λ_s^* = $(round(dual_lambda[s], digits=2)), f_s = $(round(f_s_vector[s], digits=2))")
                    end
                end
                
                # Generate cuts based on g_s vs dual_lambda[s]
                if g_s >= dual_lambda[s]
                    # Conditional no-good cut: Σ_{j∈F_1} (1-x_j) + Σ_{j∈F_0} x_j >= π_s
                    @constraint(master, delta_expr_partial >= pi[s])
                    cut_msg = "    Scenario $s: Added Partial No-Good Cut [g_s(F0,F1) = $(round(g_s, digits=2)) >= λ_s^* = $(round(dual_lambda[s], digits=2))]"
                    println(cut_msg)
                    if log_file !== nothing && iteration !== nothing
                        open(log_file, "a") do file
                            println(file, cut_msg)
                        end
                    end
                    cut32_new += 1
                else
                    # Integer L-shaped cut: θ_s <= λ_s^* - g_s(F0,F1) + (Σ_{j∈F_1} (1-x_j) + Σ_{j∈F_0} x_j)(g_s(F0,F1) - f_s)
                    @constraint(master, theta[s] <= dual_lambda[s] - g_s + (g_s - f_s_vector[s]) * delta_expr_partial)
                    cut_msg = "    Scenario $s: Added Partial Integer L-Shaped Cut [g_s(F0,F1) = $(round(g_s, digits=2)) < λ_s^* = $(round(dual_lambda[s], digits=2))]"
                    println(cut_msg)
                    if log_file !== nothing && iteration !== nothing
                        open(log_file, "a") do file
                            println(file, cut_msg)
                        end
                    end
                    cut33_new += 1
                end
            end
            
            println("  Partial MIP work units: $(round(partial_mip_work_units, digits=2))")
        elseif decomp_mode == 5
            # MODE 5: Gap-based dual unfixing - unfix until sum|dual| < gap
            println("  Processing scenarios with gap-based dual unfixing...")
            if log_file !== nothing && iteration !== nothing
                open(log_file, "a") do file
                    println(file, "  Processing scenarios with gap-based dual unfixing...")
                end
            end

            lp_work_units = 0.0
            
            # Process each scenario
            for s in 1:nscen
                # Step 1: Solve LP with x as variables to get q_s and duals
                print("    Scenario $s: Solving LP with x variables... ")
                q_s, x_duals, mu_d, mu_c, lp_work = solve_fl_recourse_lp_with_x_variables(
                    x_hat, scenarios[s], capacity, cost, fixedcost, unmet_pen, scaling_factor;
                    subproblem_cuts=subproblem_cuts
                )
                lp_work_units += lp_work
                pricing_work_units += lp_work
                
                if !isfinite(q_s)
                    println("LP infeasible/unbounded")
                    continue
                end
                
                println("q_s = $(round(q_s, digits=2))")
                
                # Print initial values
                println("      q_s = $(round(q_s, digits=2)), λ_s^* = $(round(dual_lambda[s], digits=2)), f_s = $(round(f_s_vector[s], digits=2))")
                if log_file !== nothing && iteration !== nothing
                    open(log_file, "a") do file
                        println(file, "      q_s = $(round(q_s, digits=2)), λ_s^* = $(round(dual_lambda[s], digits=2)), f_s = $(round(f_s_vector[s], digits=2))")
                    end
                end
                
                # Step 2: Compare q_s with dual_lambda[s]
                if dual_lambda[s] >= q_s
                    println("      Region: L-shaped cut region (λ_s^* >= q_s)")
                    if log_file !== nothing && iteration !== nothing
                        open(log_file, "a") do file
                            println(file, "      Region: L-shaped cut region (λ_s^* >= q_s)")
                        end
                    end
                    # L-shaped cut region: Unfix indices with dual == 0, then add partial L-shaped cut
                    # Initialize F1 and F0
                    F1_initial = [j for j in 1:nloc if x_hat[j] == 1]
                    F0_initial = [j for j in 1:nloc if x_hat[j] == 0]
                    
                    # Find near-zero dual indices using tolerance (avoid exact equality).
                    zero_dual_indices = [j for j in 1:nloc if abs(x_duals[j]) <= dual_zero_tol]
                    
                    # Update F1 and F0 by removing indices with zero duals
                    F1_partial = [j for j in F1_initial if !(j in zero_dual_indices)]
                    F0_partial = [j for j in F0_initial if !(j in zero_dual_indices)]
                    
                    if !isempty(zero_dual_indices)
                        println("      Unfixed $(length(zero_dual_indices)) indices with dual == 0: $zero_dual_indices")
                        if log_file !== nothing && iteration !== nothing
                            open(log_file, "a") do file
                                println(file, "      Unfixed indices: $zero_dual_indices (dual = 0)")
                                println(file, "      Updated F1 = $F1_partial, F0 = $F0_partial")
                            end
                        end
                    else
                        println("      No indices unfixed (no zero duals)")
                        if log_file !== nothing && iteration !== nothing
                            open(log_file, "a") do file
                                println(file, "      No indices unfixed (no zero duals)")
                                println(file, "      F1 = $F1_partial, F0 = $F0_partial")
                            end
                        end
                    end
                    
                    # Compute delta_expr for updated F0/F1
                    delta_expr_partial = sum(1 - x[j] for j in F1_partial; init=0.0) + sum(x[j] for j in F0_partial; init=0.0)
                    
                    # Add partial L-shaped cut based on updated F0/F1
                    @constraint(master, theta[s] <= dual_lambda[s] - q_s + (q_s - f_s_vector[s]) * delta_expr_partial)
                    cut_msg = "    Scenario $s: Added Partial Integer L-Shaped Cut [q_s = $(round(q_s, digits=2)) <= λ_s^* = $(round(dual_lambda[s], digits=2)), zero-dual unfixing]"
                    println(cut_msg)
                    if log_file !== nothing && iteration !== nothing
                        open(log_file, "a") do file
                            println(file, cut_msg)
                        end
                    end
                    cut33_new += 1
                else
                    # No-good cut region: Unfix based on gap
                    println("      Region: No-good cut region (λ_s^* < q_s)")
                    if log_file !== nothing && iteration !== nothing
                        open(log_file, "a") do file
                            println(file, "      Region: No-good cut region (λ_s^* < q_s)")
                        end
                    end
                    
                    gap = q_s - dual_lambda[s]
                    println("      Initial gap = q_s - λ_s^* = $(round(gap, digits=2))")
                    
                    # Initialize F1 and F0
                    F1_initial = [j for j in 1:nloc if x_hat[j] == 1]
                    F0_initial = [j for j in 1:nloc if x_hat[j] == 0]
                    
                    # Create list of all indices with their absolute dual values
                    indices_with_duals = [(j, abs(x_duals[j])) for j in 1:nloc]
                    # Sort by absolute dual value (smallest first)
                    sort!(indices_with_duals, by=x -> x[2])
                    
                    # Unfix indices until sum of |dual| >= gap
                    unfixed_indices = Int[]
                    sum_duals = 0.0
                    for (j, abs_dual) in indices_with_duals
                        if sum_duals + abs_dual < gap
                            push!(unfixed_indices, j)
                            sum_duals += abs_dual
                        else
                            break  # Stop before exceeding gap
                        end
                    end
                    
                    # Update F1 and F0 by removing unfixed indices
                    F1_partial = [j for j in F1_initial if !(j in unfixed_indices)]
                    F0_partial = [j for j in F0_initial if !(j in unfixed_indices)]
                    
                    println("      Unfixed $(length(unfixed_indices)) indices: $unfixed_indices (sum|dual| = $(round(sum_duals, digits=2)) < gap = $(round(gap, digits=2)))")
                    if log_file !== nothing && iteration !== nothing
                        open(log_file, "a") do file
                            println(file, "      Initial gap = $(round(gap, digits=2))")
                            println(file, "      Unfixed indices: $unfixed_indices (sum|dual| = $(round(sum_duals, digits=2)))")
                            println(file, "      Updated F1 = $F1_partial, F0 = $F0_partial")
                        end
                    end
                    
                    # Solve MIP with updated F0/F1 to get updated objective value
                    print("      Solving MIP after unfixing... ")
                    g_s_updated, mip_work = solve_fl_partial_recourse_mip(
                        F0_partial, F1_partial, scenarios[s], capacity, cost, fixedcost, unmet_pen, scaling_factor;
                        subproblem_cuts=subproblem_cuts
                    )
                    pricing_work_units += mip_work
                    
                    if isfinite(g_s_updated)
                        updated_gap = dual_lambda[s] - g_s_updated
                        println("g_s = $(round(g_s_updated, digits=2))")
                        println("      Updated q_s (MIP objective) = $(round(g_s_updated, digits=2))")
                        println("      Updated gap = λ_s^* - g_s = $(round(updated_gap, digits=2))")
                        if log_file !== nothing && iteration !== nothing
                            open(log_file, "a") do file
                                println(file, "      Updated q_s (MIP objective) = $(round(g_s_updated, digits=2))")
                                println(file, "      Updated gap = λ_s^* - g_s = $(round(updated_gap, digits=2))")
                            end
                        end
                    else
                        println("MIP infeasible/unbounded")
                        if log_file !== nothing && iteration !== nothing
                            open(log_file, "a") do file
                                println(file, "      MIP after unfixing: infeasible/unbounded")
                            end
                        end
                    end
                    
                    # Compute delta_expr for updated F0/F1
                    delta_expr_partial = sum(1 - x[j] for j in F1_partial; init=0.0) + sum(x[j] for j in F0_partial; init=0.0)
                    
                    # Add partial no-good cut (no MIP solve needed)
                    @constraint(master, delta_expr_partial >= pi[s])
                    cut_msg = "    Scenario $s: Added Partial No-Good Cut [q_s = $(round(q_s, digits=2)) > λ_s^* = $(round(dual_lambda[s], digits=2)), gap-based unfixing]"
                    println(cut_msg)
                    if log_file !== nothing && iteration !== nothing
                        open(log_file, "a") do file
                            println(file, cut_msg)
                        end
                    end
                    cut32_new += 1
                end
            end
            
            println("  LP work units: $(round(lp_work_units, digits=2))")
        elseif decomp_mode == 6
            # MODE 6: ITERATIVE SINGLE-VARIABLE UNFIXING
            # Per scenario: start with all facilities fixed per x_hat.
            # Round 1: solve LP (all fixed) → unfix ALL zero-dual facilities.
            # Round 2+: solve LP (partial fix) → unfix ONE facility (smallest |dual|).
            # Each round: simultaneously solve partial MIP → add Cut 32 or 33.
            # Stop when only 1 fixed variable remains.
            println("  Processing scenarios with iterative single-variable unfixing (Mode 6)...")
            if log_file !== nothing && iteration !== nothing
                open(log_file, "a") do file
                    println(file, "  Processing scenarios with iterative single-variable unfixing (Mode 6)...")
                end
            end

            mode6_mip_work = 0.0

            # Write decomp iteration header to table file
            open(decomp_table_file, "a") do tfile
                println(tfile, "")
                println(tfile, "="^100)
                println(tfile, "DECOMPOSITION ITERATION $it  (master obj = $(round(last_obj, digits=4)))")
                println(tfile, "="^100)
            end

            for s in 1:nscen
                F1_cur = [j for j in 1:nloc if x_hat[j] == 1]
                F0_cur = [j for j in 1:nloc if x_hat[j] == 0]
                first_round = true
                inner_iter = 0
                table_rows = Vector{Tuple{Vector{Int},Vector{Int},Float64}}()  # (F1, F0, g_s) per inner iter

                println("    Scenario $s: starting with F1=$(length(F1_cur)), F0=$(length(F0_cur))")
                println("      F1: $F1_cur")
                println("      F0: $F0_cur")

                while true
                    inner_iter += 1
                    fixed_count = length(F0_cur) + length(F1_cur)

                    # ── Step 1: solve LP for dual information ──────────────────────────────
                    if first_round
                        # Full fix: use existing function (all nloc variables fixed to x_hat)
                        _, x_duals_vec, _, _, lp_work = solve_fl_recourse_lp_with_x_variables(
                            x_hat, scenarios[s], capacity, cost, fixedcost, unmet_pen, scaling_factor;
                            subproblem_cuts=subproblem_cuts
                        )
                        x_duals = Dict{Int,Float64}(i => x_duals_vec[i] for i in 1:nloc)
                    else
                        # Partial fix: only current F0/F1 variables are fixed
                        _, x_duals, _, _, lp_work = solve_fl_partial_recourse_lp_with_x_duals(
                            F0_cur, F1_cur, scenarios[s], capacity, cost, fixedcost, unmet_pen, scaling_factor;
                            subproblem_cuts=subproblem_cuts
                        )
                    end
                    lp_work_units += lp_work
                    pricing_work_units += lp_work

                    # ── Step 2: solve partial MIP for g_s(F0, F1) ─────────────────────────
                    g_s, mip_work = solve_fl_partial_recourse_mip(
                        F0_cur, F1_cur, scenarios[s], capacity, cost, fixedcost, unmet_pen, scaling_factor;
                        subproblem_cuts=subproblem_cuts
                    )
                    mode6_mip_work += mip_work
                    pricing_work_units += mip_work

                    if !isfinite(g_s)
                        println("      Inner iter $inner_iter: MIP infeasible/unbounded, stopping")
                        break
                    end

                    # Collect row for table: (F1_cur, F0_cur, g_s)
                    push!(table_rows, (copy(F1_cur), copy(F0_cur), g_s))

                    # ── Step 3: add cut based on λ_s* − g_s(F0, F1) ──────────────────────
                    delta_expr_partial = sum(1 - x[j] for j in F1_cur; init=0.0) +
                                         sum(x[j]     for j in F0_cur; init=0.0)

                    if g_s >= dual_lambda[s]
                        @constraint(master, delta_expr_partial >= pi[s])
                        cut32_new += 1
                        cut_type = "Cut 32 (No-Good)"
                    else
                        @constraint(master, theta[s] <= dual_lambda[s] - g_s + (g_s - f_s_vector[s]) * delta_expr_partial)
                        cut33_new += 1
                        cut_type = "Cut 33 (L-Shaped)"
                    end

                    cut_msg = "      Scenario $s, inner iter $inner_iter: |fixed|=$fixed_count, g_s=$(round(g_s, digits=2)), λ_s*=$(round(dual_lambda[s], digits=2)), f_s=$(round(f_s_vector[s], digits=2)), $cut_type"
                    println(cut_msg)
                    if log_file !== nothing && iteration !== nothing
                        open(log_file, "a") do file
                            println(file, cut_msg)
                        end
                    end

                    # ── Stopping condition ─────────────────────────────────────────────────
                    if fixed_count <= 1
                        break
                    end

                    # ── Step 4: decide which variable(s) to unfix ─────────────────────────
                    fixed_all = vcat(F0_cur, F1_cur)
                    if first_round
                        first_round = false
                        # Unfix ALL with |dual| ≤ dual_zero_tol
                        to_unfix = Set([i for i in fixed_all if abs(get(x_duals, i, 0.0)) <= dual_zero_tol])
                        if isempty(to_unfix)
                            # No zero-dual variables — fall back to single smallest
                            min_i = fixed_all[argmin([abs(get(x_duals, i, 0.0)) for i in fixed_all])]
                            to_unfix = Set([min_i])
                        end
                    else
                        # Unfix ONE with smallest |dual|
                        min_i = fixed_all[argmin([abs(get(x_duals, i, 0.0)) for i in fixed_all])]
                        to_unfix = Set([min_i])
                    end

                    println("      Unfixing $(length(to_unfix)): $(sort(collect(to_unfix)))")

                    F1_cur = [j for j in F1_cur if !(j in to_unfix)]
                    F0_cur = [j for j in F0_cur if !(j in to_unfix)]
                    println("      Updated F1 ($(length(F1_cur))): $F1_cur")
                    println("      Updated F0 ($(length(F0_cur))): $F0_cur")
                end

                println("    Scenario $s: $inner_iter inner iterations completed")
                println("\033[96m    " * "─"^70 * "\033[0m")  # cyan separator after each scenario
                if log_file !== nothing && iteration !== nothing
                    open(log_file, "a") do file
                        println(file, "    Scenario $s: $inner_iter inner iterations completed")
                        println(file, "    " * "─"^70)
                    end
                end

                # ── Write scenario table to decomp table file ─────────────────────
                open(decomp_table_file, "a") do tfile
                    println(tfile, "")
                    println(tfile, "  SCENARIO $s  (f_s = $(round(f_s_vector[s], digits=4)),  λ_s* = $(round(dual_lambda[s], digits=4)))")
                    println(tfile, "  " * "-"^96)
                    @printf(tfile, "  %-10s  %-55s  %-14s  %-14s\n", "Inner Iter", "Fixed Sets (F1 / F0)", "g_s", "f_s")
                    println(tfile, "  " * "-"^96)
                    for (row_idx, (f1, f0, gs)) in enumerate(table_rows)
                        fixed_str = "F1=$(f1)  F0=$(f0)"
                        # Wrap long fixed_str across multiple lines if needed
                        if length(fixed_str) <= 55
                            @printf(tfile, "  %-10d  %-55s  %-14.4f  %-14.4f\n", row_idx, fixed_str, gs, f_s_vector[s])
                        else
                            f1_str = "F1=$(f1)"
                            f0_str = "F0=$(f0)"
                            @printf(tfile, "  %-10d  %-55s  %-14.4f  %-14.4f\n", row_idx, f1_str, gs, f_s_vector[s])
                            @printf(tfile, "  %-10s  %-55s\n", "", f0_str)
                        end
                    end
                    println(tfile, "  " * "-"^96)
                end
            end

            println("  Mode 6 — LP work: $(round(lp_work_units, digits=2)), MIP work: $(round(mode6_mip_work, digits=2))")
        else
            # DUAL-BASED UNFIXING (MODE 2): Solve LP with x variables, unfix based on duals, solve partial MIP
            F1_initial = [j for j in 1:nloc if x_hat[j] == 1]  # All indices where x_hat[j] == 1
            F0_initial = [j for j in 1:nloc if x_hat[j] == 0]  # All indices where x_hat[j] == 0
            println("  Initial selection: F1 (forced open) = $(length(F1_initial)) facilities, F0 (forced closed) = $(length(F0_initial)) facilities")
            if log_file !== nothing && iteration !== nothing
                open(log_file, "a") do file
                    println(file, "  Initial selection: F1 = $F1_initial, F0 = $F0_initial")
                    println(file, "  Processing scenarios with dual-based unfixing...")
                end
            end

            partial_mip_work_units = 0.0
            println("  Processing scenarios with dual-based unfixing...")
            
            # Process each scenario: solve LP, update F0/F1 based on duals, then solve MIP
            for s in 1:nscen
                # Step 1: Solve LP with x as variables to get dual information
                print("    Scenario $s: Solving LP with x variables... ")
                q_s, x_duals, mu_d, mu_c, lp_work = solve_fl_recourse_lp_with_x_variables(
                    x_hat, scenarios[s], capacity, cost, fixedcost, unmet_pen, scaling_factor;
                    subproblem_cuts=subproblem_cuts
                )
                lp_work_units += lp_work
                pricing_work_units += lp_work
                
                if !isfinite(q_s)
                    println("LP infeasible/unbounded")
                    continue
                end
                
                # Step 2: Find near-zero dual indices using tolerance (avoid exact equality).
                zero_dual_indices = [j for j in 1:nloc if abs(x_duals[j]) <= dual_zero_tol]
                
                # Step 3: Update F1 and F0 by removing indices with zero duals
                F1_partial = [j for j in F1_initial if !(j in zero_dual_indices)]
                F0_partial = [j for j in F0_initial if !(j in zero_dual_indices)]
                
                # If F1 becomes empty, proceed with empty F1 (all facilities free)
                removed_from_F1 = [j for j in F1_initial if j in zero_dual_indices]
                removed_from_F0 = [j for j in F0_initial if j in zero_dual_indices]
                
                if !isempty(removed_from_F1) || !isempty(removed_from_F0)
                    println("LP solved, removed $(length(removed_from_F1)) from F1, $(length(removed_from_F0)) from F0")
                    if log_file !== nothing && iteration !== nothing
                        open(log_file, "a") do file
                            println(file, "      Removed from F1: $removed_from_F1 (dual = 0)")
                            println(file, "      Removed from F0: $removed_from_F0 (dual = 0)")
                            println(file, "      Updated F1 = $F1_partial, F0 = $F0_partial")
                        end
                    end
                else
                    println("LP solved, no facilities unfixed")
                end
                
                # Step 4: Compute delta_expr for updated F0/F1
                delta_expr_partial = sum(1 - x[j] for j in F1_partial; init=0.0) + sum(x[j] for j in F0_partial; init=0.0)
                
                # Step 5: Solve partial MIP recourse with updated F0/F1
                print("      Solving partial MIP (F0=$F0_partial, F1=$F1_partial)... ")
                g_s, mip_work = solve_fl_partial_recourse_mip(
                    F0_partial, F1_partial, scenarios[s], capacity, cost, fixedcost, unmet_pen, scaling_factor;
                    subproblem_cuts=subproblem_cuts
                )
                partial_mip_work_units += mip_work
                pricing_work_units += mip_work
                
                if !isfinite(g_s)
                    println("MIP infeasible/unbounded")
                    continue
                end
                
                println("g_s = $(round(g_s, digits=2))")
                
                # Print q_s, g_s, and dual_lambda[s] for this scenario
                println("      Values: q_s = $(round(q_s, digits=2)), g_s(F0,F1) = $(round(g_s, digits=2)), λ_s^* = $(round(dual_lambda[s], digits=2)), f_s = $(round(f_s_vector[s], digits=2))")
                if log_file !== nothing && iteration !== nothing
                    open(log_file, "a") do file
                        println(file, "      Values: q_s = $(round(q_s, digits=2)), g_s(F0,F1) = $(round(g_s, digits=2)), λ_s^* = $(round(dual_lambda[s], digits=2)), f_s = $(round(f_s_vector[s], digits=2))")
                    end
                end
                
                # Step 6: Generate cuts based on g_s vs dual_lambda[s]
                if g_s >= dual_lambda[s]
                    # Conditional no-good cut: Σ_{j∈F_1} (1-x_j) + Σ_{j∈F_0} x_j >= π_s
                    @constraint(master, delta_expr_partial >= pi[s])
                    cut_msg = "    Scenario $s: Added Partial No-Good Cut [g_s(F0,F1) = $(round(g_s, digits=2)) >= λ_s^* = $(round(dual_lambda[s], digits=2))]"
                    println(cut_msg)
                    if log_file !== nothing && iteration !== nothing
                        open(log_file, "a") do file
                            println(file, cut_msg)
                        end
                    end
                    cut32_new += 1
                else
                    # Integer L-shaped cut: θ_s <= λ_s^* - g_s(F0,F1) + (Σ_{j∈F_1} (1-x_j) + Σ_{j∈F_0} x_j)(g_s(F0,F1) - f_s)
                    @constraint(master, theta[s] <= dual_lambda[s] - g_s + (g_s - f_s_vector[s]) * delta_expr_partial)
                    cut_msg = "    Scenario $s: Added Partial Integer L-Shaped Cut [g_s(F0,F1) = $(round(g_s, digits=2)) < λ_s^* = $(round(dual_lambda[s], digits=2))]"
                    println(cut_msg)
                    if log_file !== nothing && iteration !== nothing
                        open(log_file, "a") do file
                            println(file, cut_msg)
                        end
                    end
                    cut33_new += 1
                end
            end
            
            println("  LP work units: $(round(lp_work_units, digits=2)), Partial MIP work units: $(round(partial_mip_work_units, digits=2))")
        end
        
        # Generate Benders cuts (Cut 34) using fully fixed x_hat
        sig = x_signature_FL(x_hat)
        if use_benders_cuts
        println("  Generating Cut 34 using fully fixed x_hat (signature: $sig)...")

        # ── MIP big-M pre-computation (once per x_hat, before the per-scenario loop) ──────────
        # We need two things for the MIP constraint:
        #   max_q      = max_s { c^T x_hat + f(x_hat, d_s) }   [RHS of domination constraint]
        #   a_dmin, const_dmin                                   [LHS coefficients from LP(x_hat, d_min)]
        # Computing these here (not inside the MIP function) avoids solving nscen redundant LPs.
        max_q_mip  = -Inf
        a_dmin     = zeros(nloc)
        const_dmin = 0.0
        if use_mip_bigM
            # Step 1: collect max_q by scanning all scenario LPs; populates cache for the loop below
            for s_pre in 1:nscen
                key_pre = (s_pre, sig)
                if haskey(scenario_x_cache, key_pre)
                    q_pre = scenario_x_cache[key_pre]["q_s"]::Float64
                else
                    q_pre, mu_d_pre, mu_c_pre, lv_pre, pre_work = solve_fl_recourse_lp_with_duals(
                        x_hat, scenarios[s_pre], capacity, cost, fixedcost, unmet_pen, scaling_factor;
                        subproblem_cuts=subproblem_cuts
                    )
                    pricing_work_units += pre_work
                    if isfinite(q_pre)
                        cache_rec_pre = Dict{String,Any}(
                            "s"        => s_pre,
                            "x_sig"    => sig,
                            "x_hat"    => copy(x_hat),
                            "q_s"      => q_pre,
                            "mu_d"     => copy(mu_d_pre),
                            "mu_c"     => copy(mu_c_pre),
                            "lambda_v" => copy(lv_pre),
                        )
                        scenario_x_cache[key_pre] = cache_rec_pre
                        push!(cut_records, cache_rec_pre)
                    end
                end
                isfinite(q_pre) && (max_q_mip = max(max_q_mip, q_pre))
            end
            println("  [MIP big-M] max_q = $(round(max_q_mip, digits=2))")

            # Step 2: compute d_min = component-wise minimum demand across all scenarios
            d_min = [minimum(scenarios[s][j] for s in 1:nscen) for j in 1:ncust]

            # Step 3: solve LP(x_hat, d_min) once to get duals for the domination constraint
            _, mu_d_dmin, mu_c_dmin, lv_dmin, dmin_work = solve_fl_recourse_lp_with_duals(
                x_hat, d_min, capacity, cost, fixedcost, unmet_pen, scaling_factor;
                subproblem_cuts=subproblem_cuts
            )
            pricing_work_units += dmin_work

            # Step 4: build constraint coefficients
            # a_dmin_i   = fixedcost_i + cap_i·μ_c_dmin_i + Σ_j min(cap_i, d_min_j)·λ_v_dmin_ij
            # const_dmin = Σ_j d_min_j·μ_d_dmin_j
            # Constraint in MIP: Σ_i a_dmin_i·x_i <= max_q - const_dmin
            const_dmin = sum(d_min[j] * mu_d_dmin[j] for j in 1:ncust)
            a_dmin     = [fixedcost[i] + capacity[i] * mu_c_dmin[i] +
                           sum(min(capacity[i], d_min[j]) * lv_dmin[i, j] for j in 1:ncust)
                          for i in 1:nloc]
            println("  [MIP big-M] d_min constraint built (const_dmin = $(round(const_dmin, digits=2)))")
        end
        # ─────────────────────────────────────────────────────────────────────────────────────────

        for s in 1:nscen
            key = (s, sig)
            if haskey(scenario_x_cache, key)
                cache_rec = scenario_x_cache[key]
                q_s      = cache_rec["q_s"]::Float64
                mu_d     = cache_rec["mu_d"]::Vector{Float64}
                mu_c     = cache_rec["mu_c"]::Vector{Float64}
                lambda_v = cache_rec["lambda_v"]::Matrix{Float64}
                println("    Scenario $s: Using cached LP values [q_s = $(round(q_s, digits=2))]")
            else
                print("    Scenario $s: Solving recourse LP for Cut 34... ")
                q_s, mu_d, mu_c, lambda_v, sub_work = solve_fl_recourse_lp_with_duals(
                    x_hat, scenarios[s], capacity, cost, fixedcost, unmet_pen, scaling_factor;
                    subproblem_cuts=subproblem_cuts
                )
                pricing_work_units += sub_work
                if isfinite(q_s)
                    println("q_s = $(round(q_s, digits=2))")
                    cache_rec = Dict{String,Any}(
                        "s"        => s,
                        "x_sig"   => sig,
                        "x_hat"   => copy(x_hat),
                        "q_s"     => q_s,
                        "mu_d"    => copy(mu_d),
                        "mu_c"    => copy(mu_c),
                        "lambda_v" => copy(lambda_v),
                    )
                    scenario_x_cache[key] = cache_rec
                    push!(cut_records, cache_rec)
                else
                    println("Infeasible/unbounded")
                    mu_d     = zeros(ncust)
                    mu_c     = zeros(nloc)
                    lambda_v = zeros(nloc, ncust)
                end
            end

            if !isfinite(q_s)
                continue
            end

            if use_mip_bigM
                M_s, m_work = compute_M_value_FL_opt(
                    mu_d, mu_c, lambda_v, scenarios[s], fixedcost, capacity, dual_lambda[s],
                    a_dmin, const_dmin, max_q_mip
                )
                pricing_work_units += m_work
            else
                M_s = compute_M_value_FL(mu_d, mu_c, lambda_v, scenarios[s], fixedcost, capacity, dual_lambda[s])
            end

            lhs_check = theta_hat[s]
            rhs_check = (dual_lambda[s] - q_s) * pi_hat[s] + tol
            if lhs_check >= rhs_check
                @constraint(master,
                    theta[s] +
                    sum(fixedcost[i] * x[i] for i in 1:nloc) +
                    sum(scenarios[s][j] * mu_d[j] for j in 1:ncust) +
                    sum((capacity[i] * x[i]) * mu_c[i] for i in 1:nloc) +
                    sum(min(capacity[i], scenarios[s][j]) * x[i] * lambda_v[i, j] for i in 1:nloc, j in 1:ncust)
                    <= dual_lambda[s] + M_s * (1 - pi[s])
                )
                bigM_mode_str = use_mip_bigM ? "MIP" : "analytical"
                cut_msg = "    Scenario $s: Added Cut 34 [M_s = $(round(M_s, digits=4)), bigM=$bigM_mode_str, θ_hat = $(round(lhs_check, digits=4)) >= $(round(rhs_check, digits=4))]"
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

        # Print summary of cuts added in this decomposition iteration
        println("\n  Summary of cuts added in this iteration:")
        println("    Partial No-Good Cuts (Cut 32): $cut32_new")
        println("    Partial L-Shaped Cuts (Cut 33): $cut33_new")
        println("    Conditional Benders Cuts (Cut 34): $cut34_new")
        if log_file !== nothing && iteration !== nothing
            open(log_file, "a") do file
                println(file, "\n  Summary of cuts added in this iteration:")
                println(file, "    Partial No-Good Cuts (Cut 32): $cut32_new")
                println(file, "    Partial L-Shaped Cuts (Cut 33): $cut33_new")
                println(file, "    Conditional Benders Cuts (Cut 34): $cut34_new")
            end
        end

        # Accumulate totals across all iterations
        total_cut32 += cut32_new
        total_cut33 += cut33_new
        total_cut34 += cut34_new

        end  # end if use_benders_cuts

        if use_benders_cuts && !added_cut34
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
    println("  Total Partial No-Good Cuts (Cut 32): $total_cut32")
    println("  Total Partial L-Shaped Cuts (Cut 33): $total_cut33")
    println("  Total Conditional Benders Cuts (Cut 34): $total_cut34")
    if log_file !== nothing && iteration !== nothing
        open(log_file, "a") do file
            println(file, "\n" * "="^80)
            println(file, "DECOMPOSITION ALGORITHM COMPLETE")
            println(file, "="^80)
            println(file, "  Total cache entries: $(length(scenario_x_cache))")
            println(file, "  Total cut records stored: $(length(cut_records))")
            println(file, "  Total work units: $(round(pricing_work_units, digits=2))")
            println(file, "  Total Partial No-Good Cuts (Cut 32): $total_cut32")
            println(file, "  Total Partial L-Shaped Cuts (Cut 33): $total_cut33")
            println(file, "  Total Conditional Benders Cuts (Cut 34): $total_cut34")
        end
    end

    if !isfinite(last_obj)
        println("  No finite solution found")
        return return_pricing_payload(Vector{Vector{Float64}}(), Float64[], pricing_work_units, return_nodes, master; analysis=analysis)
    end

    num_open = sum(last_x_hat)
    println("  Final master objective: $(round(last_obj, digits=4))")
    println("  Solution: $num_open facilities open")
    println("  Returning solution to column generation (reduced cost filtering will be done by ColumnGenerationCore)")
    println("="^80 * "\n")
    if log_file !== nothing && iteration !== nothing
        open(log_file, "a") do file
            println(file, "  Final master objective: $(round(last_obj, digits=4))")
            println(file, "  Solution: $num_open facilities open")
            println(file, "  Returning solution to column generation")
            println(file, "="^80)
        end
    end
    return return_pricing_payload([Float64.(last_x_hat)], [last_obj], pricing_work_units, return_nodes, master; analysis=analysis)
end
# =============================================================================
# PRICING SUBPROBLEM WRAPPER (Interface for ColumnGenerationCore)
# =============================================================================

function build_pricing_model_AF_FL(scenarios::Vector{Vector{Float64}}, dual_lambda::Vector{Float64}, dual_mu::Float64, problem_data::Dict, pricing_type::Int, work_limit::Float64, opt_worklimit::Float64, use_solution_pool::Bool, return_nodes::Bool, log_file::Union{String, Nothing}, iteration::Union{Int, Nothing}=nothing)
    """Build and solve pricing subproblem for Facility Location - Assignment Formulation
    
    This is the interface function called by ColumnGenerationCore.jl
    
    Args:
        scenarios: List of S scenarios
        dual_lambda: Dual values from assignment constraints
        dual_mu: Dual value from K constraint
        problem_data: Dict containing problem data
        pricing_type: 1 or 2
        work_limit: Total work limit for pricing
        opt_worklimit: Work limit for Phase 1 (Type 2 only)
        use_solution_pool: Whether to use solution pool
        return_nodes: Whether to return B&B node count
        log_file: Optional log file path
    
    Returns:
        (solutions, objectives, work_units) or (solutions, objectives, work_units, nodes) if return_nodes=true
    """
    nloc = problem_data["nloc"]
    ncust = problem_data["ncust"]
    capacity = problem_data["capacity"]
    cost = problem_data["cost"]
    fixedcost = problem_data["fixedcost"]
    unmet_pen = problem_data["unmet_pen"]
    scaling_factor = problem_data["scaling_factor"]
    
    nscen = length(scenarios)
    
    if pricing_type == 1
        # Calculate m^s values
        m_values = calculate_m_values_FL(scenarios, dual_lambda, problem_data)
        
        # Solve Type 1
        analysis = Dict{Symbol,Any}()
        return af_pricing_type1_FL(nloc, ncust, capacity, cost, scenarios, m_values, dual_lambda, dual_mu, unmet_pen, scaling_factor, fixedcost; 
                                   use_solution_pool=use_solution_pool, return_nodes=return_nodes, work_limit=work_limit, analysis=analysis, log_file=log_file, iteration=iteration)
    elseif pricing_type == 2
        # Solve Type 2 (standard or decomposition path)
        m_values = zeros(nscen)
        analysis = Dict{Symbol,Any}()
        use_decomp = get(problem_data, "use_decomp", false) == true ||
                     lowercase(get(ENV, "FL_PRICING_DECOMP", "0")) in ("1", "true", "yes")
        if use_decomp
            if iteration !== nothing
                println("\nPRICING TYPE 2 - DECOMPOSITION MODE (CG Iteration $iteration)")
            else
                println("\nPRICING TYPE 2 - DECOMPOSITION MODE")
            end
            f_s_vector = get(problem_data, "f_s", nothing)
            if f_s_vector === nothing
                error("❌ Error: Decomposition requires problem_data[\"f_s\"].")
            end
            println("  f_s vector loaded: $(length(f_s_vector)) wait-and-see costs")
            if !haskey(problem_data, "fl_decomp_cache")
                problem_data["fl_decomp_cache"] = Dict{String,Any}(
                    "scenario_x_cache" => Dict{Tuple{Int,String}, Dict{String,Any}}(),
                    "records" => Vector{Dict{String,Any}}()
                )
                println("  📝 Initializing decomposition cache (first use)")
            else
                cache_size = length(get(problem_data["fl_decomp_cache"], "scenario_x_cache", Dict()))
                records_size = length(get(problem_data["fl_decomp_cache"], "records", []))
                println("  📦 Using existing cache: $cache_size scenario-x pairs, $records_size cut records")
            end
            decomp_cache = problem_data["fl_decomp_cache"]::Dict{String,Any}
            use_simple_decomp = get(problem_data, "use_simple_decomp", false) == true ||
                                 lowercase(get(ENV, "FL_PRICING_SIMPLE_DECOMP", "0")) in ("1", "true", "yes")
            decomp_mode = get(problem_data, "decomp_mode", use_simple_decomp ? 1 : 2)  # Default: 1 if simple, 2 otherwise
            subproblem_cuts = get(problem_data, "subproblem_cuts", true) == true
            use_mip_bigM = get(problem_data, "use_mip_bigM", true) == true
            use_benders_cuts = USE_BENDERS_CUTS
            return af_pricing_type2_FL_decomp(
                nloc, ncust, capacity, cost, scenarios, dual_lambda, dual_mu, unmet_pen, scaling_factor, fixedcost, f_s_vector, decomp_cache, problem_data;
                return_nodes=return_nodes, analysis=analysis, log_file=log_file, iteration=iteration, max_cut_iters=100, tol=1e-4, use_simple_decomp=use_simple_decomp, decomp_mode=decomp_mode, subproblem_cuts=subproblem_cuts, use_mip_bigM=use_mip_bigM, use_benders_cuts=use_benders_cuts
            )
        end
        return af_pricing_type2_FL(nloc, ncust, capacity, cost, scenarios, m_values, dual_lambda, dual_mu, unmet_pen, scaling_factor, fixedcost; 
                                   use_solution_pool=use_solution_pool, return_nodes=return_nodes, work_limit=work_limit, opt_worklimit=opt_worklimit, analysis=analysis, log_file=log_file, iteration=iteration)
    else
        error("Invalid pricing_type. Use 1 or 2.")
    end
end

