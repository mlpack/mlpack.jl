export gmm_generate

import ..GMM

using mlpack._Internal.params

import mlpack_jll
const gmm_generateLibrary = mlpack_jll.libmlpack_julia_gmm_generate

# Call the C binding of the mlpack gmm_generate binding.
function call_gmm_generate(p, t)
  success = ccall((:mlpack_gmm_generate, gmm_generateLibrary), Bool, (Ptr{Nothing}, Ptr{Nothing}), p, t)
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
function GetParamGMM(params::Ptr{Nothing}, paramName::String, modelPtrs::Set{Ptr{Nothing}})::GMM
  ptr = ccall((:GetParamGMMPtr, gmm_generateLibrary), Ptr{Nothing}, (Ptr{Nothing}, Cstring,), params, paramName)
  return GMM(ptr; finalize=!(ptr in modelPtrs))
end

# Set the value of a model pointer parameter of type GMM.
function SetParamGMM(params::Ptr{Nothing}, paramName::String, model::GMM)
  ccall((:SetParamGMMPtr, gmm_generateLibrary), Nothing, (Ptr{Nothing}, Cstring, Ptr{Nothing}), params, paramName, model.ptr)
end

# Delete an instantiated model pointer.
function DeleteGMM(ptr::Ptr{Nothing})
  ccall((:DeleteGMMPtr, gmm_generateLibrary), Nothing, (Ptr{Nothing},), ptr)
end

# Serialize a model to the given stream.
function serializeGMM(stream::IO, model::GMM)
  buf_len = UInt[0]
  buf_ptr = ccall((:SerializeGMMPtr, gmm_generateLibrary), Ptr{UInt8}, (Ptr{Nothing}, Ptr{UInt}), model.ptr, pointer(buf_len))
  buf = Base.unsafe_wrap(Vector{UInt8}, buf_ptr, buf_len[1]; own=true)
  write(stream, buf_len[1])
  write(stream, buf)
end
# Deserialize a model from the given stream.
function deserializeGMM(stream::IO)::GMM
  buf_len = read(stream, UInt)
  buffer = read(stream, buf_len)
  GC.@preserve buffer GMM(ccall((:DeserializeGMMPtr, gmm_generateLibrary), Ptr{Nothing}, (Ptr{UInt8}, UInt), pointer(buffer), length(buffer)))
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

 - `input_model::GMM`: Input GMM model to generate samples from.
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

  # Create the set of model pointers to avoid setting multiple finalizers.
  modelPtrs = Set{Ptr{Nothing}}()

  p = GetParameters("gmm_generate")
  t = Timers()

  juliaOwnedMemory = Set{Ptr{Nothing}}()
  # Process each input argument before calling mlpackMain().
  push!(modelPtrs, convert(GMM, input_model).ptr)
  gmm_generate_internal.SetParamGMM(p, "input_model", convert(GMM, input_model))
  SetParam(p, "samples", samples)
  if !ismissing(seed)
    SetParam(p, "seed", convert(Int, seed))
  end
  if verbose !== nothing && verbose === true
    EnableVerbose()
  else
    DisableVerbose()
  end

  SetPassed(p, "output")
  # Call the program.
  call_gmm_generate(p, t)

  results = (GetParamMat(p, "output", points_are_rows, juliaOwnedMemory))

  # We are responsible for cleaning up the `p` and `t` objects.
  DeleteParameters(p)
  DeleteTimers(t)

  return results
end
