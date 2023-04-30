export pca


using mlpack._Internal.params

import mlpack_jll
const pcaLibrary = mlpack_jll.libmlpack_julia_pca

# Call the C binding of the mlpack pca binding.
function call_pca(p, t)
  success = ccall((:mlpack_pca, pcaLibrary), Bool, (Ptr{Nothing}, Ptr{Nothing}), p, t)
  if !success
    # Throw an exception---false means there was a C++ exception.
    throw(ErrorException("mlpack binding error; see output"))
  end
end

" Internal module to hold utility functions. "
module pca_internal
  import ..pcaLibrary

end # module

"""
    pca(input; [decomposition_method, new_dimensionality, scale, var_to_retain, verbose])

This program performs principal components analysis on the given dataset using
the exact, randomized, randomized block Krylov, or QUIC SVD method. It will
transform the data onto its principal components, optionally performing
dimensionality reduction by ignoring the principal components with the smallest
eigenvalues.

Use the `input` parameter to specify the dataset to perform PCA on.  A desired
new dimensionality can be specified with the `new_dimensionality` parameter, or
the desired variance to retain can be specified with the `var_to_retain`
parameter.  If desired, the dataset can be scaled before running PCA with the
`scale` parameter.

Multiple different decomposition techniques can be used.  The method to use can
be specified with the `decomposition_method` parameter, and it may take the
values 'exact', 'randomized', or 'quic'.

For example, to reduce the dimensionality of the matrix `data` to 5 dimensions
using randomized SVD for the decomposition, storing the output matrix to
`data_mod`, the following command can be used:

```julia
julia> using CSV
julia> data = CSV.read("data.csv")
julia> data_mod = pca(data; decomposition_method="randomized",
            new_dimensionality=5)
```

# Arguments

 - `input::Array{Float64, 2}`: Input dataset to perform PCA on.
 - `decomposition_method::String`: Method used for the principal
      components analysis: 'exact', 'randomized', 'randomized-block-krylov',
      'quic'.  Default value `exact`.
      
 - `new_dimensionality::Int`: Desired dimensionality of output dataset. If
      0, no dimensionality reduction is performed.  Default value `0`.
      
 - `scale::Bool`: If set, the data will be scaled before running PCA, such
      that the variance of each feature is 1.  Default value `false`.
      
 - `var_to_retain::Float64`: Amount of variance to retain; should be
      between 0 and 1.  If 1, all variance is retained.  Overrides -d.  Default
      value `0`.
      
 - `verbose::Bool`: Display informational messages and the full list of
      parameters and timers at the end of execution.  Default value `false`.
      

# Return values

 - `output::Array{Float64, 2}`: Matrix to save modified dataset to.

"""
function pca(input;
             decomposition_method::Union{String, Missing} = missing,
             new_dimensionality::Union{Int, Missing} = missing,
             scale::Union{Bool, Missing} = missing,
             var_to_retain::Union{Float64, Missing} = missing,
             verbose::Union{Bool, Missing} = missing,
             points_are_rows::Bool = true)
  # Force the symbols to load.
  ccall((:loadSymbols, pcaLibrary), Nothing, ());

  # Create the set of model pointers to avoid setting multiple finalizers.
  modelPtrs = Set{Ptr{Nothing}}()

  p = GetParameters("pca")
  t = Timers()

  juliaOwnedMemory = Set{Ptr{Nothing}}()
  # Process each input argument before calling mlpackMain().
  SetParamMat(p, "input", input, points_are_rows, false, juliaOwnedMemory)
  if !ismissing(decomposition_method)
    SetParam(p, "decomposition_method", convert(String, decomposition_method))
  end
  if !ismissing(new_dimensionality)
    SetParam(p, "new_dimensionality", convert(Int, new_dimensionality))
  end
  if !ismissing(scale)
    SetParam(p, "scale", convert(Bool, scale))
  end
  if !ismissing(var_to_retain)
    SetParam(p, "var_to_retain", convert(Float64, var_to_retain))
  end
  if verbose !== nothing && verbose === true
    EnableVerbose()
  else
    DisableVerbose()
  end

  SetPassed(p, "output")
  # Call the program.
  call_pca(p, t)

  results = (GetParamMat(p, "output", points_are_rows, juliaOwnedMemory))

  # We are responsible for cleaning up the `p` and `t` objects.
  DeleteParameters(p)
  DeleteTimers(t)

  return results
end
