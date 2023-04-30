export hmm_viterbi

import ..HMMModel

using mlpack._Internal.params

import mlpack_jll
const hmm_viterbiLibrary = mlpack_jll.libmlpack_julia_hmm_viterbi

# Call the C binding of the mlpack hmm_viterbi binding.
function call_hmm_viterbi(p, t)
  success = ccall((:mlpack_hmm_viterbi, hmm_viterbiLibrary), Bool, (Ptr{Nothing}, Ptr{Nothing}), p, t)
  if !success
    # Throw an exception---false means there was a C++ exception.
    throw(ErrorException("mlpack binding error; see output"))
  end
end

" Internal module to hold utility functions. "
module hmm_viterbi_internal
  import ..hmm_viterbiLibrary

import ...HMMModel

# Get the value of a model pointer parameter of type HMMModel.
function GetParamHMMModel(params::Ptr{Nothing}, paramName::String, modelPtrs::Set{Ptr{Nothing}})::HMMModel
  ptr = ccall((:GetParamHMMModelPtr, hmm_viterbiLibrary), Ptr{Nothing}, (Ptr{Nothing}, Cstring,), params, paramName)
  return HMMModel(ptr; finalize=!(ptr in modelPtrs))
end

# Set the value of a model pointer parameter of type HMMModel.
function SetParamHMMModel(params::Ptr{Nothing}, paramName::String, model::HMMModel)
  ccall((:SetParamHMMModelPtr, hmm_viterbiLibrary), Nothing, (Ptr{Nothing}, Cstring, Ptr{Nothing}), params, paramName, model.ptr)
end

# Delete an instantiated model pointer.
function DeleteHMMModel(ptr::Ptr{Nothing})
  ccall((:DeleteHMMModelPtr, hmm_viterbiLibrary), Nothing, (Ptr{Nothing},), ptr)
end

# Serialize a model to the given stream.
function serializeHMMModel(stream::IO, model::HMMModel)
  buf_len = UInt[0]
  buf_ptr = ccall((:SerializeHMMModelPtr, hmm_viterbiLibrary), Ptr{UInt8}, (Ptr{Nothing}, Ptr{UInt}), model.ptr, pointer(buf_len))
  buf = Base.unsafe_wrap(Vector{UInt8}, buf_ptr, buf_len[1]; own=true)
  write(stream, buf_len[1])
  write(stream, buf)
end
# Deserialize a model from the given stream.
function deserializeHMMModel(stream::IO)::HMMModel
  buf_len = read(stream, UInt)
  buffer = read(stream, buf_len)
  GC.@preserve buffer HMMModel(ccall((:DeserializeHMMModelPtr, hmm_viterbiLibrary), Ptr{Nothing}, (Ptr{UInt8}, UInt), pointer(buffer), length(buffer)))
end
end # module

"""
    hmm_viterbi(input, input_model; [verbose])

This utility takes an already-trained HMM, specified as `input_model`, and
evaluates the most probable hidden state sequence of a given sequence of
observations (specified as '`input`, using the Viterbi algorithm.  The computed
state sequence may be saved using the `output` output parameter.

For example, to predict the state sequence of the observations `obs` using the
HMM `hmm`, storing the predicted state sequence to `states`, the following
command could be used:

```julia
julia> using CSV
julia> obs = CSV.read("obs.csv")
julia> states = hmm_viterbi(obs, hmm)
```

# Arguments

 - `input::Array{Float64, 2}`: Matrix containing observations,
 - `input_model::HMMModel`: Trained HMM to use.
 - `verbose::Bool`: Display informational messages and the full list of
      parameters and timers at the end of execution.  Default value `false`.
      

# Return values

 - `output::Array{Int, 2}`: File to save predicted state sequence to.

"""
function hmm_viterbi(input,
                     input_model::HMMModel;
                     verbose::Union{Bool, Missing} = missing,
                     points_are_rows::Bool = true)
  # Force the symbols to load.
  ccall((:loadSymbols, hmm_viterbiLibrary), Nothing, ());

  # Create the set of model pointers to avoid setting multiple finalizers.
  modelPtrs = Set{Ptr{Nothing}}()

  p = GetParameters("hmm_viterbi")
  t = Timers()

  juliaOwnedMemory = Set{Ptr{Nothing}}()
  # Process each input argument before calling mlpackMain().
  SetParamMat(p, "input", input, points_are_rows, false, juliaOwnedMemory)
  push!(modelPtrs, convert(HMMModel, input_model).ptr)
  hmm_viterbi_internal.SetParamHMMModel(p, "input_model", convert(HMMModel, input_model))
  if verbose !== nothing && verbose === true
    EnableVerbose()
  else
    DisableVerbose()
  end

  SetPassed(p, "output")
  # Call the program.
  call_hmm_viterbi(p, t)

  results = (GetParamUMat(p, "output", points_are_rows, juliaOwnedMemory))

  # We are responsible for cleaning up the `p` and `t` objects.
  DeleteParameters(p)
  DeleteTimers(t)

  return results
end
