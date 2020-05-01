export gmm_probability

import ..GMM

using mlpack._Internal.cli

import mlpack_jll
const gmm_probabilityLibrary = mlpack_jll.libmlpack_julia_gmm_probability

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

import ...GMM

# Get the value of a model pointer parameter of type GMM.
function CLIGetParamGMM(paramName::String)::GMM
  GMM(ccall((:CLI_GetParamGMMPtr, gmm_probabilityLibrary), Ptr{Nothing}, (Cstring,), paramName))
end

# Set the value of a model pointer parameter of type GMM.
function CLISetParamGMM(paramName::String, model::GMM)
  ccall((:CLI_SetParamGMMPtr, gmm_probabilityLibrary), Nothing, (Cstring, Ptr{Nothing}), paramName, model.ptr)
end

# Serialize a model to the given stream.
function serializeGMM(stream::IO, model::GMM)
  buf_len = UInt[0]
  buf_ptr = ccall((:SerializeGMMPtr, gmm_probabilityLibrary), Ptr{UInt8}, (Ptr{Nothing}, Ptr{UInt}), model.ptr, Base.pointer(buf_len))
  buf = Base.unsafe_wrap(Vector{UInt8}, buf_ptr, buf_len[1]; own=true)
  write(stream, buf)
end
# Deserialize a model from the given stream.
function deserializeGMM(stream::IO)::GMM
  buffer = read(stream)
  GMM(ccall((:DeserializeGMMPtr, gmm_probabilityLibrary), Ptr{Nothing}, (Ptr{UInt8}, UInt), Base.pointer(buffer), length(buffer)))
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

```julia
julia> using CSV
julia> points = CSV.read("points.csv")
julia> probs = gmm_probability(points, gmm)
```

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
                         input_model::GMM;
                         verbose::Union{Bool, Missing} = missing,
                         points_are_rows::Bool = true)
  # Force the symbols to load.
  ccall((:loadSymbols, gmm_probabilityLibrary), Nothing, ());

  CLIRestoreSettings("GMM Probability Calculator")

  # Process each input argument before calling mlpackMain().
  CLISetParamMat("input", input, points_are_rows)
  gmm_probability_internal.CLISetParamGMM("input_model", convert(GMM, input_model))
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
