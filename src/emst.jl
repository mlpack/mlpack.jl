export emst


using mlpack._Internal.io

import mlpack_jll
const emstLibrary = mlpack_jll.libmlpack_julia_emst

# Call the C binding of the mlpack emst binding.
function emst_mlpackMain()
  success = ccall((:emst, emstLibrary), Bool, ())
  if !success
    # Throw an exception---false means there was a C++ exception.
    throw(ErrorException("mlpack binding error; see output"))
  end
end

" Internal module to hold utility functions. "
module emst_internal
  import ..emstLibrary

end # module

"""
    emst(input; [leaf_size, naive, verbose])

This program can compute the Euclidean minimum spanning tree of a set of input
points using the dual-tree Boruvka algorithm.

The set to calculate the minimum spanning tree of is specified with the `input`
parameter, and the output may be saved with the `output` output parameter.

The `leaf_size` parameter controls the leaf size of the kd-tree that is used to
calculate the minimum spanning tree, and if the `naive` option is given, then
brute-force search is used (this is typically much slower in low dimensions). 
The leaf size does not affect the results, but it may have some effect on the
runtime of the algorithm.

For example, the minimum spanning tree of the input dataset `data` can be
calculated with a leaf size of 20 and stored as `spanning_tree` using the
following command:

```julia
julia> using CSV
julia> data = CSV.read("data.csv")
julia> spanning_tree = emst(data; leaf_size=20)
```

The output matrix is a three-dimensional matrix, where each row indicates an
edge.  The first dimension corresponds to the lesser index of the edge; the
second dimension corresponds to the greater index of the edge; and the third
column corresponds to the distance between the two points.

# Arguments

 - `input::Array{Float64, 2}`: Input data matrix.
 - `leaf_size::Int`: Leaf size in the kd-tree.  One-element leaves give
      the empirically best performance, but at the cost of greater memory
      requirements.  Default value `1`.
      
 - `naive::Bool`: Compute the MST using O(n^2) naive algorithm.  Default
      value `false`.
      
 - `verbose::Bool`: Display informational messages and the full list of
      parameters and timers at the end of execution.  Default value `false`.
      

# Return values

 - `output::Array{Float64, 2}`: Output data.  Stored as an edge list.

"""
function emst(input;
              leaf_size::Union{Int, Missing} = missing,
              naive::Union{Bool, Missing} = missing,
              verbose::Union{Bool, Missing} = missing,
              points_are_rows::Bool = true)
  # Force the symbols to load.
  ccall((:loadSymbols, emstLibrary), Nothing, ());

  IORestoreSettings("Fast Euclidean Minimum Spanning Tree")

  # Process each input argument before calling mlpackMain().
  IOSetParamMat("input", input, points_are_rows)
  if !ismissing(leaf_size)
    IOSetParam("leaf_size", convert(Int, leaf_size))
  end
  if !ismissing(naive)
    IOSetParam("naive", convert(Bool, naive))
  end
  if verbose !== nothing && verbose === true
    IOEnableVerbose()
  else
    IODisableVerbose()
  end

  IOSetPassed("output")
  # Call the program.
  emst_mlpackMain()

  return IOGetParamMat("output", points_are_rows)
end
