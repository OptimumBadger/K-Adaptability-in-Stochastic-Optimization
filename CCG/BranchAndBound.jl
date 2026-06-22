# BranchAndBound.jl
# Backward-compatibility shim — includes the split Core + FL files so that
# existing code (CG_AF_FL.jl, GroupBranchAndBound.jl, etc.) which does
# include("BranchAndBound.jl") continues to work without changes.
#
# New problem-specific files:
#   BranchAndBoundKnapsack.jl         — KS without recourse
#   BranchAndBoundKnapsackRecourse.jl — KS with recourse

include("BranchAndBoundCore.jl")
include("BranchAndBoundFL.jl")

# Keep the old function name as an alias so FacilityLocationAFDecomposition.jl
# calls continue to work without modification.
branch_and_bound(problem_data, scenarios, lambda_star; kwargs...) =
    branch_and_bound_FL(problem_data, scenarios, lambda_star; kwargs...)
