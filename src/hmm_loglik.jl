export hmm_loglik

using mlpack.util.cli

import mlpack_jll
const hmm_loglikLibrary = mlpack_jll.libmlpack_julia_hmm_loglik

# Call the C binding of the mlpack hmm_loglik binding.
function hmm_loglik_mlpackMain()
  success = ccall((:hmm_loglik, hmm_loglikLibrary), Bool, ())
  if !success
    # Throw an exception---false means there was a C++ exception.
    throw(ErrorException("mlpack binding error; see output"))
  end
end

" Internal module to hold utility functions. "
module hmm_loglik_internal
  import ..hmm_loglikLibrary

" Get the value of a model pointer parameter of type HMMModel."
function CLIGetParamHMMModelPtr(paramName::String)
  return ccall((:CLI_GetParamHMMModelPtr, hmm_loglikLibrary), Ptr{Nothing}, (Cstring,), paramName)
end

" Set the value of a model pointer parameter of type HMMModel."
function CLISetParamHMMModelPtr(paramName::String, ptr::Ptr{Nothing})
  ccall((:CLI_SetParamHMMModelPtr, hmm_loglikLibrary), Nothing, (Cstring, Ptr{Nothing}), paramName, ptr)
end

end # module

"""
    hmm_loglik(input, input_model; [verbose])

This utility takes an already-trained HMM, specified with the `input_model`
parameter, and evaluates the log-likelihood of a sequence of observations, given
with the `input` parameter.  The computed log-likelihood is given as output.

For example, to compute the log-likelihood of the sequence `seq` with the
pre-trained HMM `hmm`, the following command may be used: 

julia> using CSV
julia> seq = CSV.read("seq.csv")
julia> _ = hmm_loglik(seq, hmm)

# Arguments

 - `input::Array{Float64, 2}`: File containing observations,
 - `input_model::unknown_`: File containing HMM.
 - `verbose::Bool`: Display informational messages and the full list of
      parameters and timers at the end of execution.  Default value `false`.
      

# Return values

 - `log_likelihood::Float64`: Log-likelihood of the sequence.  Default
      value `0`.
      

"""
function hmm_loglik(input,
                    input_model::Ptr{Nothing};
                    verbose::Union{Bool, Missing} = missing,
                    points_are_rows::Bool = true)
  # Force the symbols to load.
  ccall((:loadSymbols, hmm_loglikLibrary), Nothing, ());

  CLIRestoreSettings("Hidden Markov Model (HMM) Sequence Log-Likelihood")

  # Process each input argument before calling mlpackMain().
  CLISetParamMat("input", input, points_are_rows)
  hmm_loglik_internal.CLISetParamHMMModelPtr("input_model", convert(Ptr{Nothing}, input_model))
  if verbose !== nothing && verbose === true
    CLIEnableVerbose()
  else
    CLIDisableVerbose()
  end

  CLISetPassed("log_likelihood")
  # Call the program.
  hmm_loglik_mlpackMain()

  return CLIGetParamDouble("log_likelihood")
end
