export gmm_generate

import ..GMM

using mlpack._Internal.cli

import mlpack_jll
const gmm_generateLibrary = mlpack_jll.libmlpack_julia_gmm_generate

# Call the C binding of the mlpack gmm_generate binding.
function gmm_generate_mlpackMain()
  success = ccall((:gmm_generate, gmm_generateLibrary), Bool, ())
  if !success
    # Throw an exception---false means there was a C++ exception.
    throw(ErrorException("mlpack binding error; see output"))
  end
end

" Internal module to hold utility functions. "
module gmm_generate_internal
  import ..gmm_generateLibrary

import ...GMM

# Get the value of a model pointer parameter of type GMM.
function CLIGetParamGMM(paramName::String)::GMM
  GMM(ccall((:CLI_GetParamGMMPtr, gmm_generateLibrary), Ptr{Nothing}, (Cstring,), paramName))
end

# Set the value of a model pointer parameter of type GMM.
function CLISetParamGMM(paramName::String, model::GMM)
  ccall((:CLI_SetParamGMMPtr, gmm_generateLibrary), Nothing, (Cstring, Ptr{Nothing}), paramName, model.ptr)
end

# Serialize a model to the given stream.
function serializeGMM(stream::IO, model::GMM)
  buf_len = UInt[0]
  buf_ptr = ccall((:SerializeGMMPtr, gmm_generateLibrary), Ptr{UInt8}, (Ptr{Nothing}, Ptr{UInt}), model.ptr, Base.pointer(buf_len))
  buf = Base.unsafe_wrap(Vector{UInt8}, buf_ptr, buf_len[1]; own=true)
  write(stream, buf)
end
# Deserialize a model from the given stream.
function deserializeGMM(stream::IO)::GMM
  buffer = read(stream)
  GMM(ccall((:DeserializeGMMPtr, gmm_generateLibrary), Ptr{Nothing}, (Ptr{UInt8}, UInt), Base.pointer(buffer), length(buffer)))
end
end # module

"""
    gmm_generate(input_model, samples; [seed, verbose])

This program is able to generate samples from a pre-trained GMM (use gmm_train
to train a GMM).  The pre-trained GMM must be specified with the `input_model`
parameter.  The number of samples to generate is specified by the `samples`
parameter.  Output samples may be saved with the `output` output parameter.

The following command can be used to generate 100 samples from the pre-trained
GMM `gmm` and store those generated samples in `samples`:

```julia
julia> samples = gmm_generate(gmm, 100)
```

# Arguments

 - `input_model::unknown_`: Input GMM model to generate samples from.
 - `samples::Int`: Number of samples to generate.
 - `seed::Int`: Random seed.  If 0, 'std::time(NULL)' is used.  Default
      value `0`.
      
 - `verbose::Bool`: Display informational messages and the full list of
      parameters and timers at the end of execution.  Default value `false`.
      

# Return values

 - `output::Array{Float64, 2}`: Matrix to save output samples in.

"""
function gmm_generate(input_model::GMM,
                      samples::Int;
                      seed::Union{Int, Missing} = missing,
                      verbose::Union{Bool, Missing} = missing,
                      points_are_rows::Bool = true)
  # Force the symbols to load.
  ccall((:loadSymbols, gmm_generateLibrary), Nothing, ());

  CLIRestoreSettings("GMM Sample Generator")

  # Process each input argument before calling mlpackMain().
  gmm_generate_internal.CLISetParamGMM("input_model", convert(GMM, input_model))
  CLISetParam("samples", samples)
  if !ismissing(seed)
    CLISetParam("seed", convert(Int, seed))
  end
  if verbose !== nothing && verbose === true
    CLIEnableVerbose()
  else
    CLIDisableVerbose()
  end

  CLISetPassed("output")
  # Call the program.
  gmm_generate_mlpackMain()

  return CLIGetParamMat("output", points_are_rows)
end
