export hmm_viterbi

import ..HMMModel

using mlpack._Internal.io

import mlpack_jll
const hmm_viterbiLibrary = mlpack_jll.libmlpack_julia_hmm_viterbi

# Call the C binding of the mlpack hmm_viterbi binding.
function hmm_viterbi_mlpackMain()
  success = ccall((:hmm_viterbi, hmm_viterbiLibrary), Bool, ())
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
function IOGetParamHMMModel(paramName::String)::HMMModel
  HMMModel(ccall((:IO_GetParamHMMModelPtr, hmm_viterbiLibrary), Ptr{Nothing}, (Cstring,), paramName))
end

# Set the value of a model pointer parameter of type HMMModel.
function IOSetParamHMMModel(paramName::String, model::HMMModel)
  ccall((:IO_SetParamHMMModelPtr, hmm_viterbiLibrary), Nothing, (Cstring, Ptr{Nothing}), paramName, model.ptr)
end

# Serialize a model to the given stream.
function serializeHMMModel(stream::IO, model::HMMModel)
  buf_len = UInt[0]
  buf_ptr = ccall((:SerializeHMMModelPtr, hmm_viterbiLibrary), Ptr{UInt8}, (Ptr{Nothing}, Ptr{UInt}), model.ptr, Base.pointer(buf_len))
  buf = Base.unsafe_wrap(Vector{UInt8}, buf_ptr, buf_len[1]; own=true)
  write(stream, buf)
end
# Deserialize a model from the given stream.
function deserializeHMMModel(stream::IO)::HMMModel
  buffer = read(stream)
  HMMModel(ccall((:DeserializeHMMModelPtr, hmm_viterbiLibrary), Ptr{Nothing}, (Ptr{UInt8}, UInt), Base.pointer(buffer), length(buffer)))
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

  IORestoreSettings("Hidden Markov Model (HMM) Viterbi State Prediction")

  # Process each input argument before calling mlpackMain().
  IOSetParamMat("input", input, points_are_rows)
  hmm_viterbi_internal.IOSetParamHMMModel("input_model", convert(HMMModel, input_model))
  if verbose !== nothing && verbose === true
    IOEnableVerbose()
  else
    IODisableVerbose()
  end

  IOSetPassed("output")
  # Call the program.
  hmm_viterbi_mlpackMain()

  return IOGetParamUMat("output", points_are_rows)
end
