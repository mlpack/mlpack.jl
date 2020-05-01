export dbscan


using mlpack._Internal.cli

import mlpack_jll
const dbscanLibrary = mlpack_jll.libmlpack_julia_dbscan

# Call the C binding of the mlpack dbscan binding.
function dbscan_mlpackMain()
  success = ccall((:dbscan, dbscanLibrary), Bool, ())
  if !success
    # Throw an exception---false means there was a C++ exception.
    throw(ErrorException("mlpack binding error; see output"))
  end
end

" Internal module to hold utility functions. "
module dbscan_internal
  import ..dbscanLibrary

end # module

"""
    dbscan(input; [epsilon, min_size, naive, selection_type, single_mode, tree_type, verbose])

This program implements the DBSCAN algorithm for clustering using accelerated
tree-based range search.  The type of tree that is used may be parameterized, or
brute-force range search may also be used.

The input dataset to be clustered may be specified with the `input` parameter;
the radius of each range search may be specified with the `epsilon` parameters,
and the minimum number of points in a cluster may be specified with the
`min_size` parameter.

The `assignments` and `centroids` output parameters may be used to save the
output of the clustering. `assignments` contains the cluster assignments of each
point, and `centroids` contains the centroids of each cluster.

The range search may be controlled with the `tree_type`, `single_mode`, and
`naive` parameters.  `tree_type` can control the type of tree used for range
search; this can take a variety of values: 'kd', 'r', 'r-star', 'x',
'hilbert-r', 'r-plus', 'r-plus-plus', 'cover', 'ball'. The `single_mode`
parameter will force single-tree search (as opposed to the default dual-tree
search), and '`naive` will force brute-force range search.

An example usage to run DBSCAN on the dataset in `input` with a radius of 0.5
and a minimum cluster size of 5 is given below:

```julia
julia> using CSV
julia> input = CSV.read("input.csv")
julia> _, _ = dbscan(input; epsilon=0.5, min_size=5)
```

# Arguments

 - `input::Array{Float64, 2}`: Input dataset to cluster.
 - `epsilon::Float64`: Radius of each range search.  Default value `1`.

 - `min_size::Int`: Minimum number of points for a cluster.  Default value
      `5`.
      
 - `naive::Bool`: If set, brute-force range search (not tree-based) will
      be used.  Default value `false`.
      
 - `selection_type::String`: If using point selection policy, the type of
      selection to use ('ordered', 'random').  Default value `ordered`.
      
 - `single_mode::Bool`: If set, single-tree range search (not dual-tree)
      will be used.  Default value `false`.
      
 - `tree_type::String`: If using single-tree or dual-tree search, the type
      of tree to use ('kd', 'r', 'r-star', 'x', 'hilbert-r', 'r-plus',
      'r-plus-plus', 'cover', 'ball').  Default value `kd`.
      
 - `verbose::Bool`: Display informational messages and the full list of
      parameters and timers at the end of execution.  Default value `false`.
      

# Return values

 - `assignments::Array{Int, 1}`: Output matrix for assignments of each
      point.
 - `centroids::Array{Float64, 2}`: Matrix to save output centroids to.

"""
function dbscan(input;
                epsilon::Union{Float64, Missing} = missing,
                min_size::Union{Int, Missing} = missing,
                naive::Union{Bool, Missing} = missing,
                selection_type::Union{String, Missing} = missing,
                single_mode::Union{Bool, Missing} = missing,
                tree_type::Union{String, Missing} = missing,
                verbose::Union{Bool, Missing} = missing,
                points_are_rows::Bool = true)
  # Force the symbols to load.
  ccall((:loadSymbols, dbscanLibrary), Nothing, ());

  CLIRestoreSettings("DBSCAN clustering")

  # Process each input argument before calling mlpackMain().
  CLISetParamMat("input", input, points_are_rows)
  if !ismissing(epsilon)
    CLISetParam("epsilon", convert(Float64, epsilon))
  end
  if !ismissing(min_size)
    CLISetParam("min_size", convert(Int, min_size))
  end
  if !ismissing(naive)
    CLISetParam("naive", convert(Bool, naive))
  end
  if !ismissing(selection_type)
    CLISetParam("selection_type", convert(String, selection_type))
  end
  if !ismissing(single_mode)
    CLISetParam("single_mode", convert(Bool, single_mode))
  end
  if !ismissing(tree_type)
    CLISetParam("tree_type", convert(String, tree_type))
  end
  if verbose !== nothing && verbose === true
    CLIEnableVerbose()
  else
    CLIDisableVerbose()
  end

  CLISetPassed("assignments")
  CLISetPassed("centroids")
  # Call the program.
  dbscan_mlpackMain()

  return CLIGetParamURow("assignments"),
         CLIGetParamMat("centroids", points_are_rows)
end
