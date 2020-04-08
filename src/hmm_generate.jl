export hmm_generate

using mlpack._Internal.cli

import mlpack_jll
const hmm_generateLibrary = mlpack_jll.libmlpack_julia_hmm_generate

# Call the C binding of the mlpack hmm_generate binding.
function hmm_generate_mlpackMain()
  success = ccall((:hmm_generate, hmm_generateLibrary), Bool, ())
  if !success
    # Throw an exception---false means there was a C++ exception.
    throw(ErrorException("mlpack binding error; see output"))
  end
end

" Internal module to hold utility functions. "
module hmm_generate_internal
  import ..hmm_generateLibrary

" Get the value of a model pointer parameter of type HMMModel."
function CLIGetParamHMMModelPtr(paramName::String)
  return ccall((:CLI_GetParamHMMModelPtr, hmm_generateLibrary), Ptr{Nothing}, (Cstring,), paramName)
end

" Set the value of a model pointer parameter of type HMMModel."
function CLISetParamHMMModelPtr(paramName::String, ptr::Ptr{Nothing})
  ccall((:CLI_SetParamHMMModelPtr, hmm_generateLibrary), Nothing, (Cstring, Ptr{Nothing}), paramName, ptr)
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
 - `model::unknown_`: Trained HMM to generate sequences with.
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
                      model::Ptr{Nothing};
                      seed::Union{Int, Missing} = missing,
                      start_state::Union{Int, Missing} = missing,
                      verbose::Union{Bool, Missing} = missing,
                      points_are_rows::Bool = true)
  # Force the symbols to load.
  ccall((:loadSymbols, hmm_generateLibrary), Nothing, ());

  CLIRestoreSettings("Hidden Markov Model (HMM) Sequence Generator")

  # Process each input argument before calling mlpackMain().
  CLISetParam("length", length)
  hmm_generate_internal.CLISetParamHMMModelPtr("model", convert(Ptr{Nothing}, model))
  if !ismissing(seed)
    CLISetParam("seed", convert(Int, seed))
  end
  if !ismissing(start_state)
    CLISetParam("start_state", convert(Int, start_state))
  end
  if verbose !== nothing && verbose === true
    CLIEnableVerbose()
  else
    CLIDisableVerbose()
  end

  CLISetPassed("output")
  CLISetPassed("state")
  # Call the program.
  hmm_generate_mlpackMain()

  return CLIGetParamMat("output", points_are_rows),
         CLIGetParamUMat("state", points_are_rows)
end
