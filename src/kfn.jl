export kfn

using mlpack._Internal.cli

import mlpack_jll
const kfnLibrary = mlpack_jll.libmlpack_julia_kfn

# Call the C binding of the mlpack kfn binding.
function kfn_mlpackMain()
  success = ccall((:kfn, kfnLibrary), Bool, ())
  if !success
    # Throw an exception---false means there was a C++ exception.
    throw(ErrorException("mlpack binding error; see output"))
  end
end

" Internal module to hold utility functions. "
module kfn_internal
  import ..kfnLibrary

" Get the value of a model pointer parameter of type KFNModel."
function CLIGetParamKFNModelPtr(paramName::String)
  return ccall((:CLI_GetParamKFNModelPtr, kfnLibrary), Ptr{Nothing}, (Cstring,), paramName)
end

" Set the value of a model pointer parameter of type KFNModel."
function CLISetParamKFNModelPtr(paramName::String, ptr::Ptr{Nothing})
  ccall((:CLI_SetParamKFNModelPtr, kfnLibrary), Nothing, (Cstring, Ptr{Nothing}), paramName, ptr)
end

end # module

"""
    kfn(; [algorithm, epsilon, input_model, k, leaf_size, percentage, query, random_basis, reference, seed, tree_type, true_distances, true_neighbors, verbose])

This program will calculate the k-furthest-neighbors of a set of points. You may
specify a separate set of reference points and query points, or just a reference
set which will be used as both the reference and query set.

For example, the following will calculate the 5 furthest neighbors of eachpoint
in `input` and store the distances in `distances` and the neighbors in
`neighbors`: 

```julia
julia> using CSV
julia> input = CSV.read("input.csv")
julia> distances, neighbors, _ = kfn(k=5, reference=input)
```

The output files are organized such that row i and column j in the neighbors
output matrix corresponds to the index of the point in the reference set which
is the j'th furthest neighbor from the point in the query set with index i.  Row
i and column j in the distances output file corresponds to the distance between
those two points.

# Arguments

 - `algorithm::String`: Type of neighbor search: 'naive', 'single_tree',
      'dual_tree', 'greedy'.  Default value `dual_tree`.
      
 - `epsilon::Float64`: If specified, will do approximate furthest neighbor
      search with given relative error. Must be in the range [0,1).  Default
      value `0`.
      
 - `input_model::unknown_`: Pre-trained kFN model.
 - `k::Int`: Number of furthest neighbors to find.  Default value `0`.

 - `leaf_size::Int`: Leaf size for tree building (used for kd-trees, vp
      trees, random projection trees, UB trees, R trees, R* trees, X trees,
      Hilbert R trees, R+ trees, R++ trees, and octrees).  Default value `20`.
      
 - `percentage::Float64`: If specified, will do approximate furthest
      neighbor search. Must be in the range (0,1] (decimal form). Resultant
      neighbors will be at least (p*100) % of the distance as the true furthest
      neighbor.  Default value `1`.
      
 - `query::Array{Float64, 2}`: Matrix containing query points (optional).
 - `random_basis::Bool`: Before tree-building, project the data onto a
      random orthogonal basis.  Default value `false`.
      
 - `reference::Array{Float64, 2}`: Matrix containing the reference
      dataset.
 - `seed::Int`: Random seed (if 0, std::time(NULL) is used).  Default
      value `0`.
      
 - `tree_type::String`: Type of tree to use: 'kd', 'vp', 'rp', 'max-rp',
      'ub', 'cover', 'r', 'r-star', 'x', 'ball', 'hilbert-r', 'r-plus',
      'r-plus-plus', 'oct'.  Default value `kd`.
      
 - `true_distances::Array{Float64, 2}`: Matrix of true distances to
      compute the effective error (average relative error) (it is printed when
      -v is specified).
 - `true_neighbors::Array{Int, 2}`: Matrix of true neighbors to compute
      the recall (it is printed when -v is specified).
 - `verbose::Bool`: Display informational messages and the full list of
      parameters and timers at the end of execution.  Default value `false`.
      

# Return values

 - `distances::Array{Float64, 2}`: Matrix to output distances into.
 - `neighbors::Array{Int, 2}`: Matrix to output neighbors into.
 - `output_model::unknown_`: If specified, the kFN model will be output
      here.

"""
function kfn(;
             algorithm::Union{String, Missing} = missing,
             epsilon::Union{Float64, Missing} = missing,
             input_model::Union{Ptr{Nothing}, Missing} = missing,
             k::Union{Int, Missing} = missing,
             leaf_size::Union{Int, Missing} = missing,
             percentage::Union{Float64, Missing} = missing,
             query = missing,
             random_basis::Union{Bool, Missing} = missing,
             reference = missing,
             seed::Union{Int, Missing} = missing,
             tree_type::Union{String, Missing} = missing,
             true_distances = missing,
             true_neighbors = missing,
             verbose::Union{Bool, Missing} = missing,
             points_are_rows::Bool = true)
  # Force the symbols to load.
  ccall((:loadSymbols, kfnLibrary), Nothing, ());

  CLIRestoreSettings("k-Furthest-Neighbors Search")

  # Process each input argument before calling mlpackMain().
  if !ismissing(algorithm)
    CLISetParam("algorithm", convert(String, algorithm))
  end
  if !ismissing(epsilon)
    CLISetParam("epsilon", convert(Float64, epsilon))
  end
  if !ismissing(input_model)
    kfn_internal.CLISetParamKFNModelPtr("input_model", convert(Ptr{Nothing}, input_model))
  end
  if !ismissing(k)
    CLISetParam("k", convert(Int, k))
  end
  if !ismissing(leaf_size)
    CLISetParam("leaf_size", convert(Int, leaf_size))
  end
  if !ismissing(percentage)
    CLISetParam("percentage", convert(Float64, percentage))
  end
  if !ismissing(query)
    CLISetParamMat("query", query, points_are_rows)
  end
  if !ismissing(random_basis)
    CLISetParam("random_basis", convert(Bool, random_basis))
  end
  if !ismissing(reference)
    CLISetParamMat("reference", reference, points_are_rows)
  end
  if !ismissing(seed)
    CLISetParam("seed", convert(Int, seed))
  end
  if !ismissing(tree_type)
    CLISetParam("tree_type", convert(String, tree_type))
  end
  if !ismissing(true_distances)
    CLISetParamMat("true_distances", true_distances, points_are_rows)
  end
  if !ismissing(true_neighbors)
    CLISetParamUMat("true_neighbors", true_neighbors, points_are_rows)
  end
  if verbose !== nothing && verbose === true
    CLIEnableVerbose()
  else
    CLIDisableVerbose()
  end

  CLISetPassed("distances")
  CLISetPassed("neighbors")
  CLISetPassed("output_model")
  # Call the program.
  kfn_mlpackMain()

  return CLIGetParamMat("distances", points_are_rows),
         CLIGetParamUMat("neighbors", points_are_rows),
         kfn_internal.CLIGetParamKFNModelPtr("output_model")
end
