# CGUtilities.jl
# Utility functions for loading scenarios and binary vectors from files
# These functions are defined in both Samples/FL/FLUtilities.jl and Samples/TS/TSUtilities.jl
# We include both to support both Facility Location and Transmission Switching column generation
# Since the function signatures are identical, including both is safe

include("../Samples/FL/FLUtilities.jl")
include("../Samples/TS/TSUtilities.jl")

# =============================================================================
# POWERMODELS SILENT PARSING (for TS)
# =============================================================================

using PowerModels

function parse_file_silent(instance_file::String)
    """Parse PowerModels file while suppressing output messages.
    
    This is a wrapper around PowerModels.parse_file() that redirects stdout
    to suppress verbose parsing messages.
    """
    return redirect_stdout(devnull) do
        PowerModels.parse_file(instance_file)
    end
end

