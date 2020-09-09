export preprocess_one_hot_encoding


using mlpack._Internal.io

import mlpack_jll
const preprocess_one_hot_encodingLibrary = mlpack_jll.libmlpack_julia_preprocess_one_hot_encoding

# Call the C binding of the mlpack preprocess_one_hot_encoding binding.
function preprocess_one_hot_encoding_mlpackMain()
  success = ccall((:preprocess_one_hot_encoding, preprocess_one_hot_encodingLibrary), Bool, ())
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
    preprocess_one_hot_encoding(dimensions, input; [verbose])

This utility takes a dataset and a vector of indices and does one-hot encoding
of the respective features at those indices. Indices represent the IDs of the
dimensions to be one-hot encoded.

The output matrix with encoded features may be saved with the `output`
parameters.

So, a simple example where we want to encode 1st and 3rd feature from dataset
`X` into `X_output` would be

```julia
julia> using CSV
julia> X = CSV.read("X.csv")
julia> X_ouput = preprocess_one_hot_encoding(1, X)
```

# Arguments

 - `dimensions::Vector{Int}`: Index of dimensions thatneed to be one-hot
      encoded.
 - `input::Array{Float64, 2}`: Matrix containing data.
 - `verbose::Bool`: Display informational messages and the full list of
      parameters and timers at the end of execution.  Default value `false`.
      

# Return values

 - `output::Array{Float64, 2}`: Matrix to save one-hot encoded features
      data to.

"""
function preprocess_one_hot_encoding(dimensions::Vector{Int},
                                     input;
                                     verbose::Union{Bool, Missing} = missing,
                                     points_are_rows::Bool = true)
  # Force the symbols to load.
  ccall((:loadSymbols, preprocess_one_hot_encodingLibrary), Nothing, ());

  IORestoreSettings("One Hot Encoding")

  # Process each input argument before calling mlpackMain().
  IOSetParam("dimensions", dimensions)
  IOSetParamMat("input", input, points_are_rows)
  if verbose !== nothing && verbose === true
    IOEnableVerbose()
  else
    IODisableVerbose()
  end

  IOSetPassed("output")
  # Call the program.
  preprocess_one_hot_encoding_mlpackMain()

  return IOGetParamMat("output", points_are_rows)
end
