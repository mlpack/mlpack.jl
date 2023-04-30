export gmm_probability

import ..GMM

using mlpack._Internal.params

import mlpack_jll
const gmm_probabilityLibrary = mlpack_jll.libmlpack_julia_gmm_probability

# Call the C binding of the mlpack gmm_probability binding.
function call_gmm_probability(p, t)
  success = ccall((:mlpack_gmm_probability, gmm_probabilityLibrary), Bool, (Ptr{Nothing}, Ptr{Nothing}), p, t)
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
function GetParamGMM(params::Ptr{Nothing}, paramName::String, modelPtrs::Set{Ptr{Nothing}})::GMM
  ptr = ccall((:GetParamGMMPtr, gmm_probabilityLibrary), Ptr{Nothing}, (Ptr{Nothing}, Cstring,), params, paramName)
  return GMM(ptr; finalize=!(ptr in modelPtrs))
end

# Set the value of a model pointer parameter of type GMM.
function SetParamGMM(params::Ptr{Nothing}, paramName::String, model::GMM)
  ccall((:SetParamGMMPtr, gmm_probabilityLibrary), Nothing, (Ptr{Nothing}, Cstring, Ptr{Nothing}), params, paramName, model.ptr)
end

# Delete an instantiated model pointer.
function DeleteGMM(ptr::Ptr{Nothing})
  ccall((:DeleteGMMPtr, gmm_probabilityLibrary), Nothing, (Ptr{Nothing},), ptr)
end

# Serialize a model to the given stream.
function serializeGMM(stream::IO, model::GMM)
  buf_len = UInt[0]
  buf_ptr = ccall((:SerializeGMMPtr, gmm_probabilityLibrary), Ptr{UInt8}, (Ptr{Nothing}, Ptr{UInt}), model.ptr, pointer(buf_len))
  buf = Base.unsafe_wrap(Vector{UInt8}, buf_ptr, buf_len[1]; own=true)
  write(stream, buf_len[1])
  write(stream, buf)
end
# Deserialize a model from the given stream.
function deserializeGMM(stream::IO)::GMM
  buf_len = read(stream, UInt)
  buffer = read(stream, buf_len)
  GC.@preserve buffer GMM(ccall((:DeserializeGMMPtr, gmm_probabilityLibrary), Ptr{Nothing}, (Ptr{UInt8}, UInt), pointer(buffer), length(buffer)))
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
 - `input_model::GMM`: Input GMM to use as model.
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

  # Create the set of model pointers to avoid setting multiple finalizers.
  modelPtrs = Set{Ptr{Nothing}}()

  p = GetParameters("gmm_probability")
  t = Timers()

  juliaOwnedMemory = Set{Ptr{Nothing}}()
  # Process each input argument before calling mlpackMain().
  SetParamMat(p, "input", input, points_are_rows, false, juliaOwnedMemory)
  push!(modelPtrs, convert(GMM, input_model).ptr)
  gmm_probability_internal.SetParamGMM(p, "input_model", convert(GMM, input_model))
  if verbose !== nothing && verbose === true
    EnableVerbose()
  else
    DisableVerbose()
  end

  SetPassed(p, "output")
  # Call the program.
  call_gmm_probability(p, t)

  results = (GetParamMat(p, "output", points_are_rows, juliaOwnedMemory))

  # We are responsible for cleaning up the `p` and `t` objects.
  DeleteParameters(p)
  DeleteTimers(t)

  return results
end
