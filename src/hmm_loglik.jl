export hmm_loglik

import ..HMMModel

using mlpack._Internal.params

import mlpack_jll
const hmm_loglikLibrary = mlpack_jll.libmlpack_julia_hmm_loglik

# Call the C binding of the mlpack hmm_loglik binding.
function call_hmm_loglik(p, t)
  success = ccall((:mlpack_hmm_loglik, hmm_loglikLibrary), Bool, (Ptr{Nothing}, Ptr{Nothing}), p, t)
  if !success
    # Throw an exception---false means there was a C++ exception.
    throw(ErrorException("mlpack binding error; see output"))
  end
end

" Internal module to hold utility functions. "
module hmm_loglik_internal
  import ..hmm_loglikLibrary

import ...HMMModel

# Get the value of a model pointer parameter of type HMMModel.
function GetParamHMMModel(params::Ptr{Nothing}, paramName::String, modelPtrs::Set{Ptr{Nothing}})::HMMModel
  ptr = ccall((:GetParamHMMModelPtr, hmm_loglikLibrary), Ptr{Nothing}, (Ptr{Nothing}, Cstring,), params, paramName)
  return HMMModel(ptr; finalize=!(ptr in modelPtrs))
end

# Set the value of a model pointer parameter of type HMMModel.
function SetParamHMMModel(params::Ptr{Nothing}, paramName::String, model::HMMModel)
  ccall((:SetParamHMMModelPtr, hmm_loglikLibrary), Nothing, (Ptr{Nothing}, Cstring, Ptr{Nothing}), params, paramName, model.ptr)
end

# Delete an instantiated model pointer.
function DeleteHMMModel(ptr::Ptr{Nothing})
  ccall((:DeleteHMMModelPtr, hmm_loglikLibrary), Nothing, (Ptr{Nothing},), ptr)
end

# Serialize a model to the given stream.
function serializeHMMModel(stream::IO, model::HMMModel)
  buf_len = UInt[0]
  buf_ptr = ccall((:SerializeHMMModelPtr, hmm_loglikLibrary), Ptr{UInt8}, (Ptr{Nothing}, Ptr{UInt}), model.ptr, pointer(buf_len))
  buf = Base.unsafe_wrap(Vector{UInt8}, buf_ptr, buf_len[1]; own=true)
  write(stream, buf_len[1])
  write(stream, buf)
end
# Deserialize a model from the given stream.
function deserializeHMMModel(stream::IO)::HMMModel
  buf_len = read(stream, UInt)
  buffer = read(stream, buf_len)
  GC.@preserve buffer HMMModel(ccall((:DeserializeHMMModelPtr, hmm_loglikLibrary), Ptr{Nothing}, (Ptr{UInt8}, UInt), pointer(buffer), length(buffer)))
end
end # module

"""
    hmm_loglik(input, input_model; [verbose])

This utility takes an already-trained HMM, specified with the `input_model`
parameter, and evaluates the log-likelihood of a sequence of observations, given
with the `input` parameter.  The computed log-likelihood is given as output.

For example, to compute the log-likelihood of the sequence `seq` with the
pre-trained HMM `hmm`, the following command may be used: 

```julia
julia> using CSV
julia> seq = CSV.read("seq.csv")
julia> _ = hmm_loglik(seq, hmm)
```

# Arguments

 - `input::Array{Float64, 2}`: File containing observations,
 - `input_model::HMMModel`: File containing HMM.
 - `verbose::Bool`: Display informational messages and the full list of
      parameters and timers at the end of execution.  Default value `false`.
      

# Return values

 - `log_likelihood::Float64`: Log-likelihood of the sequence.  Default
      value `0`.
      

"""
function hmm_loglik(input,
                    input_model::HMMModel;
                    verbose::Union{Bool, Missing} = missing,
                    points_are_rows::Bool = true)
  # Force the symbols to load.
  ccall((:loadSymbols, hmm_loglikLibrary), Nothing, ());

  # Create the set of model pointers to avoid setting multiple finalizers.
  modelPtrs = Set{Ptr{Nothing}}()

  p = GetParameters("hmm_loglik")
  t = Timers()

  juliaOwnedMemory = Set{Ptr{Nothing}}()
  # Process each input argument before calling mlpackMain().
  SetParamMat(p, "input", input, points_are_rows, false, juliaOwnedMemory)
  push!(modelPtrs, convert(HMMModel, input_model).ptr)
  hmm_loglik_internal.SetParamHMMModel(p, "input_model", convert(HMMModel, input_model))
  if verbose !== nothing && verbose === true
    EnableVerbose()
  else
    DisableVerbose()
  end

  SetPassed(p, "log_likelihood")
  # Call the program.
  call_hmm_loglik(p, t)

  results = (GetParamDouble(p, "log_likelihood"))

  # We are responsible for cleaning up the `p` and `t` objects.
  DeleteParameters(p)
  DeleteTimers(t)

  return results
end
