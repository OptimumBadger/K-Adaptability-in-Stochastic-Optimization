

using LinearAlgebra

function read_beasley_cap(filename::String)
    """Read Beasley format facility location instance"""
    open(filename, "r") do io
        lines = readlines(io)
        
        # First line: nfacilities ncustomers
        first_line = split(lines[1])
        nfacilities = parse(Int, first_line[1])
        ncustomers = parse(Int, first_line[2])
        
        # Read facility data: capacity fixedcost
        capacity = Float64[]
        fixedcost = Float64[]
        
        for i in 1:nfacilities
            line = split(lines[1 + i])
            push!(capacity, parse(Float64, line[1]))
            push!(fixedcost, parse(Float64, line[2]))
        end
        
        # Read customer data: demand
        demand = Float64[]
        for j in 1:ncustomers
            line = split(lines[1 + nfacilities + j])
            push!(demand, parse(Float64, line[1]))
        end
        
        # Read cost matrix: cost[i][j] = cost from facility i to customer j
        cost = zeros(Float64, nfacilities, ncustomers)
        for i in 1:nfacilities
            line = split(lines[1 + nfacilities + ncustomers + i])
            for j in 1:ncustomers
                cost[i, j] = parse(Float64, line[j])
            end
        end
        
        return capacity, fixedcost, cost, demand, nfacilities, ncustomers
    end
end

function extract_and_save_instance(input_file::String, output_file::String, 
                                   nfacilities_extract::Int, ncustomers_extract::Int)
   
    
    
    # Check if input file exists
    if !isfile(input_file)
        error("Input file not found: $input_file")
    end
    
    # Read original instance
    println("Reading original instance...")
    capacity, fixedcost, cost, demand, nfacilities_orig, ncustomers_orig = read_beasley_cap(input_file)
    
    println("  Original size: $nfacilities_orig facilities, $ncustomers_orig customers")
    
    # Validate extraction parameters
    if nfacilities_extract > nfacilities_orig
        error("Cannot extract $nfacilities_extract facilities from instance with only $nfacilities_orig facilities")
    end
    if ncustomers_extract > ncustomers_orig
        error("Cannot extract $ncustomers_extract customers from instance with only $ncustomers_orig customers")
    end
    
    # Extract subset
    println("Extracting subset...")
    capacity_extract = capacity[1:nfacilities_extract]
    fixedcost_extract = fixedcost[1:nfacilities_extract]
    demand_extract = demand[1:ncustomers_extract]
    cost_extract = cost[1:nfacilities_extract, 1:ncustomers_extract]
    
    println("  Extracted size: $nfacilities_extract facilities, $ncustomers_extract customers")
    
    # Write to output file in Beasley format
    println("Writing to output file...")
    open(output_file, "w") do io
        # First line: nfacilities ncustomers
        println(io, "$nfacilities_extract $ncustomers_extract")
        
        # Facility data: capacity fixedcost (one per line)
        for i in 1:nfacilities_extract
            println(io, "$(capacity_extract[i]) $(fixedcost_extract[i])")
        end
        
        # Customer demands (one per line)
        for j in 1:ncustomers_extract
            println(io, "$(demand_extract[j])")
        end
        
        # Cost matrix: one row per facility (space-separated)
        for i in 1:nfacilities_extract
            cost_row = join([string(cost_extract[i, j]) for j in 1:ncustomers_extract], " ")
            println(io, cost_row)
        end
    end
    
    println("✅ Successfully created instance file: $output_file")
    println("  Facilities: $nfacilities_extract")
    println("  Customers: $ncustomers_extract")
    println("  Total capacity: $(sum(capacity_extract))")
    println("  Total demand: $(sum(demand_extract))")
    println("="^80)
end

# =============================================================================
# CLI ENTRYPOINT
# =============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) != 4
        println("Usage: julia scale_instance.jl <input_file> <output_file> <nfacilities> <ncustomers>")
        println("Example: julia scale_instance.jl capa1 capa11.txt 70 150")
        println()
        println("This script extracts the first N facilities and M customers")
        println("from an input instance and saves to a new file in Beasley format.")
        exit(1)
    end
    
    input_file = ARGS[1]
    output_file = ARGS[2]
    nfacilities = parse(Int, ARGS[3])
    ncustomers = parse(Int, ARGS[4])
    
    extract_and_save_instance(input_file, output_file, nfacilities, ncustomers)
end

