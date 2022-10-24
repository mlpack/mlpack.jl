export preprocess_binarize


using mlpack._Internal.params

import mlpack_jll
const preprocess_binarizeLibrary = mlpack_jll.libmlpack_julia_preprocess_binarize

# Call the C binding of the mlpack preprocess_binarize binding.
function call_preprocess_binarize(p, t)
  success = ccall((:mlpack_preprocess_binarize, preprocess_binarizeLibrary), Bool, (Ptr{Nothing}, Ptr{Nothing}), p, t)
  if !success
    # Throw an exception---false means there was a C++ exception.
    throw(ErrorException("mlpack binding error; see output"))
  end
end

" Internal module to hold utility functions. "
module preprocess_binarize_internal
  import ..preprocess_binarizeLibrary

end # module

"""
    preprocess_binarize(input; [dimension, threshold, verbose])

This utility takes a dataset and binarizes the variables into either 0 or 1
given threshold. User can apply binarization on a dimension or the whole
dataset.  The dimension to apply binarization to can be specified using the
`dimension` parameter; if left unspecified, every dimension will be binarized. 
The threshold for binarization can also be specified with the `threshold`
parameter; the default threshold is 0.0.

The binarized matrix may be saved with the `output` output parameter.

For example, if we want to set all variables greater than 5 in the dataset `X`
to 1 and variables less than or equal to 5.0 to 0, and save the result to `Y`,
we could run

```julia
julia> using CSV
julia> X = CSV.read("X.csv")
julia> Y = preprocess_binarize(X; threshold=5)
```

But if we want to apply this to only the first (0th) dimension of `X`,  we could
instead run

```julia
julia> using CSV
julia> X = CSV.read("X.csv")
julia> Y = preprocess_binarize(X; dimension=0, threshold=5)
```

# Arguments

 - `input::Array{Float64, 2}`: Input data matrix.
 - `dimension::Int`: Dimension to apply the binarization. If not set, the
      program will binarize every dimension by default.  Default value `0`.
      
 - `threshold::Float64`: Threshold to be applied for binarization. If not
      set, the threshold defaults to 0.0.  Default value `0`.
      
 - `verbose::Bool`: Display informational messages and the full list of
      parameters and timers at the end of execution.  Default value `false`.
      

# Return values

 - `output::Array{Float64, 2}`: Matrix in which to save the output.

"""
function preprocess_binarize(input;
                             dimension::Union{Int, Missing} = missing,
                             threshold::Union{Float64, Missing} = missing,
                             verbose::Union{Bool, Missing} = missing,
                             points_are_rows::Bool = true)
  # Force the symbols to load.
  ccall((:loadSymbols, preprocess_binarizeLibrary), Nothing, ());

  # Create the set of model pointers to avoid setting multiple finalizers.
  modelPtrs = Set{Ptr{Nothing}}()

  p = GetParameters("preprocess_binarize")
  t = Timers()

  juliaOwnedMemory = Set{Ptr{Nothing}}()
  # Process each input argument before calling mlpackMain().
  SetParamMat(p, "input", input, points_are_rows, juliaOwnedMemory)
  if !ismissing(dimension)
    SetParam(p, "dimension", convert(Int, dimension))
  end
  if !ismissing(threshold)
    SetParam(p, "threshold", convert(Float64, threshold))
  end
  if verbose !== nothing && verbose === true
    EnableVerbose()
  else
    DisableVerbose()
  end

  SetPassed(p, "output")
  # Call the program.
  call_preprocess_binarize(p, t)

  results = (GetParamMat(p, "output", points_are_rows, juliaOwnedMemory))

  # We are responsible for cleaning up the `p` and `t` objects.
  DeleteParameters(p)
  DeleteTimers(t)

  return results
end
