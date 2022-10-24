export hmm_generate

import ..HMMModel

using mlpack._Internal.params

import mlpack_jll
const hmm_generateLibrary = mlpack_jll.libmlpack_julia_hmm_generate

# Call the C binding of the mlpack hmm_generate binding.
function call_hmm_generate(p, t)
  success = ccall((:mlpack_hmm_generate, hmm_generateLibrary), Bool, (Ptr{Nothing}, Ptr{Nothing}), p, t)
  if !success
    # Throw an exception---false means there was a C++ exception.
    throw(ErrorException("mlpack binding error; see output"))
  end
end

" Internal module to hold utility functions. "
module hmm_generate_internal
  import ..hmm_generateLibrary

import ...HMMModel

# Get the value of a model pointer parameter of type HMMModel.
function GetParamHMMModel(params::Ptr{Nothing}, paramName::String, modelPtrs::Set{Ptr{Nothing}})::HMMModel
  ptr = ccall((:GetParamHMMModelPtr, hmm_generateLibrary), Ptr{Nothing}, (Ptr{Nothing}, Cstring,), params, paramName)
  return HMMModel(ptr; finalize=!(ptr in modelPtrs))
end

# Set the value of a model pointer parameter of type HMMModel.
function SetParamHMMModel(params::Ptr{Nothing}, paramName::String, model::HMMModel)
  ccall((:SetParamHMMModelPtr, hmm_generateLibrary), Nothing, (Ptr{Nothing}, Cstring, Ptr{Nothing}), params, paramName, model.ptr)
end

# Delete an instantiated model pointer.
function DeleteHMMModel(ptr::Ptr{Nothing})
  ccall((:DeleteHMMModelPtr, hmm_generateLibrary), Nothing, (Ptr{Nothing},), ptr)
end

# Serialize a model to the given stream.
function serializeHMMModel(stream::IO, model::HMMModel)
  buf_len = UInt[0]
  buf_ptr = ccall((:SerializeHMMModelPtr, hmm_generateLibrary), Ptr{UInt8}, (Ptr{Nothing}, Ptr{UInt}), model.ptr, pointer(buf_len))
  buf = Base.unsafe_wrap(Vector{UInt8}, buf_ptr, buf_len[1]; own=true)
  write(stream, buf_len[1])
  write(stream, buf)
end
# Deserialize a model from the given stream.
function deserializeHMMModel(stream::IO)::HMMModel
  buf_len = read(stream, UInt)
  buffer = read(stream, buf_len)
  GC.@preserve buffer HMMModel(ccall((:DeserializeHMMModelPtr, hmm_generateLibrary), Ptr{Nothing}, (Ptr{UInt8}, UInt), pointer(buffer), length(buffer)))
end
end # module

"""
    hmm_generate(length, model; [seed, start_state, verbose])

This utility takes an already-trained HMM, specified as the `model` parameter,
and generates a random observation sequence and hidden state sequence based on
its parameters. The observation sequence may be saved with the `output` output
parameter, and the internal state  sequence may be saved with the `state` output
parameter.

The state to start the sequence in may be specified with the `start_state`
parameter.

For example, to generate a sequence of length 150 from the HMM `hmm` and save
the observation sequence to `observations` and the hidden state sequence to
`states`, the following command may be used: 

```julia
julia> observations, states = hmm_generate(150, hmm)
```

# Arguments

 - `length::Int`: Length of sequence to generate.
 - `model::HMMModel`: Trained HMM to generate sequences with.
 - `seed::Int`: Random seed.  If 0, 'std::time(NULL)' is used.  Default
      value `0`.
      
 - `start_state::Int`: Starting state of sequence.  Default value `0`.

 - `verbose::Bool`: Display informational messages and the full list of
      parameters and timers at the end of execution.  Default value `false`.
      

# Return values

 - `output::Array{Float64, 2}`: Matrix to save observation sequence to.
 - `state::Array{Int, 2}`: Matrix to save hidden state sequence to.

"""
function hmm_generate(length::Int,
                      model::HMMModel;
                      seed::Union{Int, Missing} = missing,
                      start_state::Union{Int, Missing} = missing,
                      verbose::Union{Bool, Missing} = missing,
                      points_are_rows::Bool = true)
  # Force the symbols to load.
  ccall((:loadSymbols, hmm_generateLibrary), Nothing, ());

  # Create the set of model pointers to avoid setting multiple finalizers.
  modelPtrs = Set{Ptr{Nothing}}()

  p = GetParameters("hmm_generate")
  t = Timers()

  juliaOwnedMemory = Set{Ptr{Nothing}}()
  # Process each input argument before calling mlpackMain().
  SetParam(p, "length", length)
  push!(modelPtrs, convert(HMMModel, model).ptr)
  hmm_generate_internal.SetParamHMMModel(p, "model", convert(HMMModel, model))
  if !ismissing(seed)
    SetParam(p, "seed", convert(Int, seed))
  end
  if !ismissing(start_state)
    SetParam(p, "start_state", convert(Int, start_state))
  end
  if verbose !== nothing && verbose === true
    EnableVerbose()
  else
    DisableVerbose()
  end

  SetPassed(p, "output")
  SetPassed(p, "state")
  # Call the program.
  call_hmm_generate(p, t)

  results = (GetParamMat(p, "output", points_are_rows, juliaOwnedMemory),
             GetParamUMat(p, "state", points_are_rows, juliaOwnedMemory))

  # We are responsible for cleaning up the `p` and `t` objects.
  DeleteParameters(p)
  DeleteTimers(t)

  return results
end
