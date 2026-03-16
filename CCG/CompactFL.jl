using JuMP, Gurobi
using MathOptInterface
const MOI = MathOptInterface
using Printf
using Statistics
using Dates

# ---------------------------------------------------------------------------
# Infrastructure includes
# ---------------------------------------------------------------------------
script_dir = @__DIR__
project_root = dirname(script_dir)

include(joinpath(project_root, "Facility_Location", "Heuristic", "Candidate_List_Generation_Phase.jl"))
include(joinpath(project_root, "Facility_Location", "Heuristic", "Assignment_Phase.jl"))

# ---------------------------------------------------------------------------
# Helper: extract work units from a solved Gurobi model
# ---------------------------------------------------------------------------
function get_work_units(model)
    try
        return MOI.get(backend(model), Gurobi.ModelAttribute("Work"))
    catch
        return 0.0
    end
end

# =============================================================================
# FORMULATION 1: COMPACT FORMULATION
# =============================================================================

function solve_compact_FL(
    instance_file::String,
    samples_file::String;
    K::Int=2,
    S::Int=25,
    unmet_pen::Float64=15.0,
    scaling_factor::Float64=0.2,
    capacity_factor::Float64=0.3,
    work_limit::Float64=30000.0,
    experiment::String="bernoulli",
    log_dir::String=joinpath(project_root, "Results", "Facility_Location", "Compact", "logs")
)
    println("="^70)
    println("COMPACT FORMULATION - Facility Location")
    println("="^70)
    println("Instance : $instance_file")
    println("S=$S  K=$K  unmet_pen=$unmet_pen  scaling=$scaling_factor  cap_factor=$capacity_factor")
    println("Work limit : $work_limit")
    println("="^70)

    # --- Load instance ---
    instance_path = joinpath(project_root, "Facility_Location", "instances", instance_file)
    capacity, fixedcost, cost, demandmean = read_orlib_cap(instance_path)
    capacity = capacity .* capacity_factor
    nloc  = length(capacity)
    ncust = length(demandmean)
    println("Facilities: $nloc  |  Customers: $ncust")

    # --- Load samples ---
    samples = load_samples_from_file(samples_file)
    if length(samples) < S
        error("Not enough samples in file: need $S, got $(length(samples))")
    end
    samples = samples[1:S]
    nscen = S

    # --- Build model ---
    inst_name = replace(instance_file, ".txt" => "")
    mkpath(log_dir)
    timestamp = Dates.format(now(), "YYYYmmdd_HHMMSS")
    log_file = joinpath(log_dir, "compact_$(inst_name)_S$(S)_K$(K)_$(experiment)_$(timestamp).log")

    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "NonConvex", 2)   # required for bilinear objective
    set_optimizer_attribute(model, "WorkLimit", work_limit)
    set_optimizer_attribute(model, "OutputFlag", 1)
    set_optimizer_attribute(model, "LogFile", log_file)
    println("Log file   : $log_file")

    # Variables
    @variable(model, x[1:nloc, 1:K], Bin)                         # facility open in cluster k
    @variable(model, y[1:nloc, 1:ncust, 1:nscen, 1:K] >= 0)       # flow: facility i -> customer j, scenario s, cluster k
    @variable(model, p[1:ncust, 1:nscen, 1:K] >= 0)               # unmet demand penalty
    @variable(model, w[1:K, 1:nscen], Bin)                        # scenario-to-cluster assignment

    # Bilinear objective: w_k^s * (fixed + transport + penalty costs)
    @objective(model, Min, (1.0 / nscen) * (
        sum(fixedcost[i] * x[i,k] * w[k,s]
            for i in 1:nloc, k in 1:K, s in 1:nscen) +
        sum(scaling_factor * cost[i,j] * y[i,j,s,k] * w[k,s]
            for i in 1:nloc, j in 1:ncust, k in 1:K, s in 1:nscen) +
        unmet_pen * sum(p[j,s,k] * w[k,s]
            for j in 1:ncust, k in 1:K, s in 1:nscen)
    ))

    # (C1) Each scenario assigned to exactly one cluster
    for s in 1:nscen
        @constraint(model, sum(w[k,s] for k in 1:K) == 1)
    end

    # (C2) Demand satisfaction: sum_i y_{ijs}^k + p_{js}^k = d_{js}
    for j in 1:ncust, s in 1:nscen, k in 1:K
        @constraint(model, sum(y[i,j,s,k] for i in 1:nloc) + p[j,s,k] == samples[s][j])
    end

    # (C3) Capacity: sum_j y_{ijs}^k <= u_i * x_i^k
    for i in 1:nloc, s in 1:nscen, k in 1:K
        @constraint(model, sum(y[i,j,s,k] for j in 1:ncust) <= capacity[i] * x[i,k])
    end

    println("\nVariables  : $(num_variables(model))")
    println("Solving...")

    t_start = time()
    optimize!(model)
    t_solve = time() - t_start
    work_units = get_work_units(model)

    status = termination_status(model)
    println("\nStatus      : $status")
    @printf("Solve time  : %.2f s\n", t_solve)
    @printf("Work units  : %.4f\n", work_units)

    if has_values(model)
        obj = objective_value(model)
        @printf("Objective   : %.4f\n", obj)

        println("\nCluster summary:")
        for k in 1:K
            open_facs      = [i for i in 1:nloc if value(x[i,k]) > 0.5]
            assigned_scens = [s for s in 1:nscen if value(w[k,s]) > 0.5]
            println("  Cluster $k : facilities=$open_facs  |  $(length(assigned_scens)) scenarios assigned")
        end

        # Save results
        out_dir = joinpath(project_root, "Results", "Facility_Location", "Compact")
        mkpath(out_dir)
        out_file = joinpath(out_dir, "compact_results_$(inst_name)_S$(S)_K$(K)_$(experiment).json")
        open(out_file, "w") do f
            println(f, "{")
            println(f, "  \"formulation\": \"compact\",")
            println(f, "  \"instance\": \"$instance_file\",")
            println(f, "  \"S\": $S,")
            println(f, "  \"K\": $K,")
            println(f, "  \"experiment\": \"$experiment\",")
            println(f, "  \"objective\": $obj,")
            println(f, "  \"solve_time_s\": $t_solve,")
            println(f, "  \"total_work_units\": $work_units,")
            println(f, "  \"termination\": \"$(string(status))\",")
            println(f, "  \"unmet_pen\": $unmet_pen,")
            println(f, "  \"scaling_factor\": $scaling_factor,")
            println(f, "  \"capacity_factor\": $capacity_factor,")
            println(f, "  \"work_limit\": $work_limit")
            println(f, "}")
        end
        println("\nResults saved to: $out_file")

        return obj, work_units, value.(x), value.(w)
    else
        println("No feasible solution found.")
        return Inf, work_units, nothing, nothing
    end
end

# =============================================================================
# FORMULATION 2: NEW COMPACT FORMULATION
# =============================================================================

function solve_new_compact_FL(
    instance_file::String,
    samples_file::String;
    K::Int=2,
    S::Int=25,
    unmet_pen::Float64=15.0,
    scaling_factor::Float64=0.2,
    capacity_factor::Float64=0.3,
    work_limit::Float64=30000.0,
    experiment::String="bernoulli",
    log_dir::String=joinpath(project_root, "Results", "Facility_Location", "Compact", "logs")
)
    println("="^70)
    println("NEW COMPACT FORMULATION - Facility Location")
    println("="^70)
    println("Instance : $instance_file")
    println("S=$S  K=$K  unmet_pen=$unmet_pen  scaling=$scaling_factor  cap_factor=$capacity_factor")
    println("Work limit : $work_limit")
    println("="^70)

    # --- Load instance ---
    instance_path = joinpath(project_root, "Facility_Location", "instances", instance_file)
    capacity, fixedcost, cost, demandmean = read_orlib_cap(instance_path)
    capacity = capacity .* capacity_factor
    nloc  = length(capacity)
    ncust = length(demandmean)
    println("Facilities: $nloc  |  Customers: $ncust")

    # --- Load samples ---
    samples = load_samples_from_file(samples_file)
    if length(samples) < S
        error("Not enough samples in file: need $S, got $(length(samples))")
    end
    samples = samples[1:S]
    nscen = S

    # --- Build model ---
    inst_name = replace(instance_file, ".txt" => "")
    mkpath(log_dir)
    timestamp = Dates.format(now(), "YYYYmmdd_HHMMSS")
    log_file = joinpath(log_dir, "new_compact_$(inst_name)_S$(S)_K$(K)_$(experiment)_$(timestamp).log")

    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "WorkLimit", work_limit)
    set_optimizer_attribute(model, "OutputFlag", 1)
    set_optimizer_attribute(model, "LogFile", log_file)
    println("Log file   : $log_file")

    # Variables
    @variable(model, q[1:nloc, 1:nscen], Bin)              # facility open in scenario s
    @variable(model, y[1:nloc, 1:ncust, 1:nscen] >= 0)    # flow: facility i -> customer j, scenario s
    @variable(model, p[1:ncust, 1:nscen] >= 0)             # unmet demand penalty
    @variable(model, w[1:K, 1:nscen], Bin)                 # scenario-to-cluster assignment
    @variable(model, x[1:nloc, 1:K], Bin)                  # cluster representative (facility opening)

    # Linear objective
    @objective(model, Min, (1.0 / nscen) * sum(
        sum(fixedcost[i] * q[i,s] for i in 1:nloc) +
        sum(scaling_factor * cost[i,j] * y[i,j,s] for i in 1:nloc, j in 1:ncust) +
        unmet_pen * sum(p[j,s] for j in 1:ncust)
        for s in 1:nscen
    ))

    # (C1) Demand satisfaction: sum_i y_{ij}^s + p_j^s = d_j^s
    for j in 1:ncust, s in 1:nscen
        @constraint(model, sum(y[i,j,s] for i in 1:nloc) + p[j,s] == samples[s][j])
    end

    # (C2) Capacity: sum_j y_{ij}^s <= u_i * q_i^s
    for i in 1:nloc, s in 1:nscen
        @constraint(model, sum(y[i,j,s] for j in 1:ncust) <= capacity[i] * q[i,s])
    end

    # (C3) Cluster consistency: (w_k^s - 1) <= x_i^k - q_i^s <= (1 - w_k^s)
    #      When w_k^s = 1: x_i^k = q_i^s  (scenario s uses cluster k's facility plan)
    #      When w_k^s = 0: -1 <= x_i^k - q_i^s <= 1  (trivially satisfied for binary vars)
    for k in 1:K, s in 1:nscen, i in 1:nloc
        @constraint(model, x[i,k] - q[i,s] >= w[k,s] - 1)
        @constraint(model, x[i,k] - q[i,s] <= 1 - w[k,s])
    end

    # (C4) Each scenario assigned to exactly one cluster
    for s in 1:nscen
        @constraint(model, sum(w[k,s] for k in 1:K) == 1)
    end

    println("\nVariables  : $(num_variables(model))")
    println("Solving...")

    t_start = time()
    optimize!(model)
    t_solve = time() - t_start
    work_units = get_work_units(model)

    status = termination_status(model)
    println("\nStatus      : $status")
    @printf("Solve time  : %.2f s\n", t_solve)
    @printf("Work units  : %.4f\n", work_units)

    if has_values(model)
        obj = objective_value(model)
        @printf("Objective   : %.4f\n", obj)

        println("\nCluster summary:")
        for k in 1:K
            open_facs      = [i for i in 1:nloc if value(x[i,k]) > 0.5]
            assigned_scens = [s for s in 1:nscen if value(w[k,s]) > 0.5]
            println("  Cluster $k : facilities=$open_facs  |  $(length(assigned_scens)) scenarios assigned")
        end

        # Save results
        out_dir = joinpath(project_root, "Results", "Facility_Location", "Compact")
        mkpath(out_dir)
        out_file = joinpath(out_dir, "new_compact_results_$(inst_name)_S$(S)_K$(K)_$(experiment).json")
        open(out_file, "w") do f
            println(f, "{")
            println(f, "  \"formulation\": \"new_compact\",")
            println(f, "  \"instance\": \"$instance_file\",")
            println(f, "  \"S\": $S,")
            println(f, "  \"K\": $K,")
            println(f, "  \"experiment\": \"$experiment\",")
            println(f, "  \"objective\": $obj,")
            println(f, "  \"solve_time_s\": $t_solve,")
            println(f, "  \"total_work_units\": $work_units,")
            println(f, "  \"termination\": \"$(string(status))\",")
            println(f, "  \"unmet_pen\": $unmet_pen,")
            println(f, "  \"scaling_factor\": $scaling_factor,")
            println(f, "  \"capacity_factor\": $capacity_factor,")
            println(f, "  \"work_limit\": $work_limit")
            println(f, "}")
        end
        println("\nResults saved to: $out_file")

        return obj, work_units, value.(q), value.(w), value.(x)
    else
        println("No feasible solution found.")
        return Inf, work_units, nothing, nothing, nothing
    end
end

# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================
#
# Usage:
#   julia CCG/CompactFL.jl <instance_file> <samples_file> <S> <K> [formulation] [unmet_pen] [scaling_factor] [capacity_factor] [work_limit] [experiment]
#
# Arguments:
#   instance_file   (required) Instance filename, e.g. cap91.txt
#   samples_file    (required) Path to scenarios file
#   S               (required) Number of scenarios to use
#   K               (required) Number of clusters
#   formulation     (optional, default 1)      1=Compact, 2=New Compact
#   unmet_pen       (optional, default 15.0)
#   scaling_factor  (optional, default 0.2)
#   capacity_factor (optional, default 0.3)
#   work_limit      (optional, default 30000.0) Gurobi work unit limit
#   experiment      (optional, default "bernoulli") Label for output files
#
# Examples:
#   julia CCG/CompactFL.jl cap91.txt Samples/FL/Scenarios/S300_bernoulli.txt 25 2
#   julia CCG/CompactFL.jl cap91.txt Samples/FL/Scenarios/S300_bernoulli.txt 25 2 2
#   julia CCG/CompactFL.jl cap91.txt Samples/FL/Scenarios/S300_bernoulli.txt 50 4 1 15.0 0.2 0.3 30000 bernoulli

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 4
        println("Usage: julia CCG/CompactFL.jl <instance_file> <samples_file> <S> <K> [formulation] [unmet_pen] [scaling_factor] [capacity_factor] [work_limit] [experiment]")
        println("  formulation: 1=Compact (default), 2=New Compact")
        exit(1)
    end

    instance_file   = ARGS[1]
    samples_file    = ARGS[2]
    S               = parse(Int,     ARGS[3])
    K               = parse(Int,     ARGS[4])
    formulation     = length(ARGS) > 4 ? parse(Int,     ARGS[5]) : 1
    unmet_pen       = length(ARGS) > 5 ? parse(Float64, ARGS[6]) : 15.0
    scaling_factor  = length(ARGS) > 6 ? parse(Float64, ARGS[7]) : 0.2
    capacity_factor = length(ARGS) > 7 ? parse(Float64, ARGS[8]) : 0.3
    work_limit      = length(ARGS) > 8 ? parse(Float64, ARGS[9]) : 30000.0
    experiment      = length(ARGS) > 9 ? ARGS[10] : "bernoulli"

    # Resolve samples_file relative to project root if not absolute
    if !isabspath(samples_file) && !isfile(samples_file)
        samples_file = joinpath(project_root, samples_file)
    end

    if formulation == 1
        solve_compact_FL(instance_file, samples_file;
            K=K, S=S, unmet_pen=unmet_pen, scaling_factor=scaling_factor,
            capacity_factor=capacity_factor, work_limit=work_limit, experiment=experiment)
    elseif formulation == 2
        solve_new_compact_FL(instance_file, samples_file;
            K=K, S=S, unmet_pen=unmet_pen, scaling_factor=scaling_factor,
            capacity_factor=capacity_factor, work_limit=work_limit, experiment=experiment)
    else
        println("Invalid formulation: $formulation. Use 1 (Compact) or 2 (New Compact).")
        exit(1)
    end
end
