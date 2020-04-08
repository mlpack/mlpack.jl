export hmm_viterbi

using mlpack._Internal.cli

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

" Get the value of a model pointer parameter of type HMMModel."
function CLIGetParamHMMModelPtr(paramName::String)
  return ccall((:CLI_GetParamHMMModelPtr, hmm_viterbiLibrary), Ptr{Nothing}, (Cstring,), paramName)
end

" Set the value of a model pointer parameter of type HMMModel."
function CLISetParamHMMModelPtr(paramName::String, ptr::Ptr{Nothing})
  ccall((:CLI_SetParamHMMModelPtr, hmm_viterbiLibrary), Nothing, (Cstring, Ptr{Nothing}), paramName, ptr)
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
 - `input_model::unknown_`: Trained HMM to use.
 - `verbose::Bool`: Display informational messages and the full list of
      parameters and timers at the end of execution.  Default value `false`.
      

# Return values

 - `output::Array{Int, 2}`: File to save predicted state sequence to.

"""
function hmm_viterbi(input,
                     input_model::Ptr{Nothing};
                     verbose::Union{Bool, Missing} = missing,
                     points_are_rows::Bool = true)
  # Force the symbols to load.
  ccall((:loadSymbols, hmm_viterbiLibrary), Nothing, ());

  CLIRestoreSettings("Hidden Markov Model (HMM) Viterbi State Prediction")

  # Process each input argument before calling mlpackMain().
  CLISetParamMat("input", input, points_are_rows)
  hmm_viterbi_internal.CLISetParamHMMModelPtr("input_model", convert(Ptr{Nothing}, input_model))
  if verbose !== nothing && verbose === true
    CLIEnableVerbose()
  else
    CLIDisableVerbose()
  end

  CLISetPassed("output")
  # Call the program.
  hmm_viterbi_mlpackMain()

  return CLIGetParamUMat("output", points_are_rows)
end
