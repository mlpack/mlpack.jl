export preprocess_one_hot_encoding


using mlpack._Internal.params

import mlpack_jll
const preprocess_one_hot_encodingLibrary = mlpack_jll.libmlpack_julia_preprocess_one_hot_encoding

# Call the C binding of the mlpack preprocess_one_hot_encoding binding.
function call_preprocess_one_hot_encoding(p, t)
  success = ccall((:mlpack_preprocess_one_hot_encoding, preprocess_one_hot_encodingLibrary), Bool, (Ptr{Nothing}, Ptr{Nothing}), p, t)
  if !success
    # Throw an exception---false means there was a C++ exception.
    throw(ErrorException("mlpack binding error; see output"))
  end
end

" Internal module to hold utility functions. "
module preprocess_one_hot_encoding_internal
  import ..preprocess_one_hot_encodingLibrary

end # module

"""
    preprocess_one_hot_encoding(input; [dimensions, verbose])

This utility takes a dataset and a vector of indices and does one-hot encoding
of the respective features at those indices. Indices represent the IDs of the
dimensions to be one-hot encoded.

If no dimensions are specified with `dimensions`, then all categorical-type
dimensions will be one-hot encoded. Otherwise, only the dimensions given in
`dimensions` will be one-hot encoded.

The output matrix with encoded features may be saved with the `output`
parameters.

So, a simple example where we want to encode 1st and 3rd feature from dataset
`X` into `X_output` would be

```julia
julia> using CSV
julia> X = CSV.read("X.csv")
julia> X_ouput = preprocess_one_hot_encoding(X; dimensions=1)
```

# Arguments

 - `input::Tuple{Array{Bool, 1}, Array{Float64, 2}}`: Matrix containing
      data.
 - `dimensions::Vector{Int}`: Index of dimensions that need to be one-hot
      encoded (if unspecified, all categorical dimensions are one-hot encoded).
 - `verbose::Bool`: Display informational messages and the full list of
      parameters and timers at the end of execution.  Default value `false`.
      

# Return values

 - `output::Array{Float64, 2}`: Matrix to save one-hot encoded features
      data to.

"""
function preprocess_one_hot_encoding(input::Tuple{Array{Bool, 1}, Array{Float64, 2}};
                                     dimensions::Union{Vector{Int}, Missing} = missing,
                                     verbose::Union{Bool, Missing} = missing,
                                     points_are_rows::Bool = true)
  # Force the symbols to load.
  ccall((:loadSymbols, preprocess_one_hot_encodingLibrary), Nothing, ());

  # Create the set of model pointers to avoid setting multiple finalizers.
  modelPtrs = Set{Ptr{Nothing}}()

  p = GetParameters("preprocess_one_hot_encoding")
  t = Timers()

  juliaOwnedMemory = Set{Ptr{Nothing}}()
  # Process each input argument before calling mlpackMain().
  SetParam(p, "input", convert(Tuple{Array{Bool, 1}, Array{Float64, 2}}, input), points_are_rows, juliaOwnedMemory)
  if !ismissing(dimensions)
    SetParam(p, "dimensions", convert(Vector{Int}, dimensions))
  end
  if verbose !== nothing && verbose === true
    EnableVerbose()
  else
    DisableVerbose()
  end

  SetPassed(p, "output")
  # Call the program.
  call_preprocess_one_hot_encoding(p, t)

  results = (GetParamMat(p, "output", points_are_rows, juliaOwnedMemory))

  # We are responsible for cleaning up the `p` and `t` objects.
  DeleteParameters(p)
  DeleteTimers(t)

  return results
end
