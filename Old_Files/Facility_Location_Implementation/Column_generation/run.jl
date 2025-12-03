using JuMP, Gurobi, MathOptInterface
const MOI = MathOptInterface

function create_test_problem()
    model = Model(Gurobi.Optimizer)

    # logging/limits
    set_optimizer_attribute(model, "OutputFlag", 1)
    set_optimizer_attribute(model, "MIPGap", 0.01)
    set_optimizer_attribute(model, "TimeLimit", 60)

    # --- keep & search multiple solutions ---
    set_optimizer_attribute(model, "PoolSearchMode", 2)   # n-best systematic search
    set_optimizer_attribute(model, "PoolSolutions", 50)   # keep up to 50
    # optional: set_optimizer_attribute(model, "PoolGap", 0.10)

    @variable(model, x[1:5], Bin)
    @variable(model, y[1:5] >= 0)
    @objective(model, Max, sum(x) + sum(y))

    @constraint(model, x[1] + x[2] + y[1] <= 1.5)
    @constraint(model, x[2] + x[3] + y[2] <= 1.5)
    @constraint(model, x[3] + x[4] + y[3] <= 1.5)
    @constraint(model, x[4] + x[5] + y[4] <= 1.5)
    @constraint(model, x[1] + x[3] + x[5] + y[5] <= 2.5)

    return model
end

function print_solution_pool(model::Model)
    opt = backend(model)
    solcount = MOI.get(opt, Gurobi.ModelAttribute("SolCount"))
    println("\nSolutions kept in pool: $solcount")
    solcount == 0 && return

    x = model[:x]; y = model[:y]

    for k in 0:solcount-1            # Gurobi uses 0-based indices
        # IMPORTANT: SolutionNumber is a PARAMETER on the optimizer
        MOI.set(opt, MOI.RawOptimizerAttribute("SolutionNumber"), k)

        objk = MOI.get(opt, Gurobi.ModelAttribute("PoolObjVal"))
        if objk <= 7
            continue
        end
        println("\n=== Solution #$(k+1) (objective = $objk) ===")

        println("Binary variables:")
        for i in 1:5
            println("  x[$i] = ", MOI.get(opt, Gurobi.VariableAttribute("Xn"), JuMP.index(x[i])))
        end

        println("Continuous variables:")
        for i in 1:5
            println("  y[$i] = ", MOI.get(opt, Gurobi.VariableAttribute("Xn"), JuMP.index(y[i])))
        end
    end
end

function main()
    model = create_test_problem()
    println("\nSolving the model...")
    optimize!(model)

    ts = termination_status(model)
    println("\nTermination status: $ts")
    if ts in (MOI.OPTIMAL, MOI.TIME_LIMIT)
        println("Best objective: ", objective_value(model))
        println("\n--- Incumbent (best) ---")
        for i in 1:5 println("x[$i] = ", value(model[:x][i])) end
        for i in 1:5 println("y[$i] = ", value(model[:y][i])) end

        # Print every stored solution
        print_solution_pool(model)

        # Optional: write all stored solutions to a .sol file
        #write_to_file(model, "solution_pool.sol")
        #println("\nWrote solution pool to solution_pool.sol")
    end
end

main()
