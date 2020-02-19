export gmm_probability

using mlpack.util.cli

import mlpack_jll
const gmm_probabilityLibrary = mlpack_jll.libmlpack_gmm_probability

# Call the C binding of the mlpack gmm_probability binding.
function gmm_probability_mlpackMain()
  success = ccall((:gmm_probability, gmm_probabilityLibrary), Bool, ())
  if !success
    # Throw an exception---false means there was a C++ exception.
    throw(ErrorException("mlpack binding error; see output"))
  end
end

" Internal module to hold utility functions. "
module gmm_probability_internal
  import ..gmm_probabilityLibrary

" Get the value of a model pointer parameter of type GMM."
function CLIGetParamGMMPtr(paramName::String)
  return ccall((:CLI_GetParamGMMPtr, gmm_probabilityLibrary), Ptr{Nothing}, (Cstring,), paramName)
end

" Set the value of a model pointer parameter of type GMM."
function CLISetParamGMMPtr(paramName::String, ptr::Ptr{Nothing})
  ccall((:CLI_SetParamGMMPtr, gmm_probabilityLibrary), Nothing, (Cstring, Ptr{Nothing}), paramName, ptr)
end

end # module

"""
    gmm_probability(input, input_model; [verbose])

This program calculates the probability that given points came from a given GMM
(that is, P(X | gmm)).  The GMM is specified with the `input_model` parameter,
and the points are specified with the `input` parameter.  The output
probabilities may be saved via the `output` output parameter.

So, for example, to calculate the probabilities of each point in `points` coming
from the pre-trained GMM `gmm`, while storing those probabilities in `probs`,
the following command could be used:

julia> using CSV
julia> points = CSV.read("points.csv")
julia> probs = gmm_probability(points, gmm)

# Arguments

 - `input::Array{Float64, 2}`: Input matrix to calculate probabilities
      of.
 - `input_model::unknown_`: Input GMM to use as model.
 - `verbose::Bool`: Display informational messages and the full list of
      parameters and timers at the end of execution.  Default value `false`.
      

# Return values

 - `output::Array{Float64, 2}`: Matrix to store calculated probabilities
      in.

"""
function gmm_probability(input,
                         input_model::Ptr{Nothing};
                         verbose::Union{Bool, Missing} = missing,
                         points_are_rows::Bool = true)
  # Force the symbols to load.
  ccall((:loadSymbols, gmm_probabilityLibrary), Nothing, ());

  CLIRestoreSettings("GMM Probability Calculator")

  # Process each input argument before calling mlpackMain().
  CLISetParamMat("input", input, points_are_rows)
  gmm_probability_internal.CLISetParamGMMPtr("input_model", convert(Ptr{Nothing}, input_model))
  if verbose !== nothing && verbose === true
    CLIEnableVerbose()
  else
    CLIDisableVerbose()
  end

  CLISetPassed("output")
  # Call the program.
  gmm_probability_mlpackMain()

  return CLIGetParamMat("output", points_are_rows)
end
