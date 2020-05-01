export knn

import ..KNNModel

using mlpack._Internal.cli

import mlpack_jll
const knnLibrary = mlpack_jll.libmlpack_julia_knn

# Call the C binding of the mlpack knn binding.
function knn_mlpackMain()
  success = ccall((:knn, knnLibrary), Bool, ())
  if !success
    # Throw an exception---false means there was a C++ exception.
    throw(ErrorException("mlpack binding error; see output"))
  end
end

" Internal module to hold utility functions. "
module knn_internal
  import ..knnLibrary

import ...KNNModel

# Get the value of a model pointer parameter of type KNNModel.
function CLIGetParamKNNModel(paramName::String)::KNNModel
  KNNModel(ccall((:CLI_GetParamKNNModelPtr, knnLibrary), Ptr{Nothing}, (Cstring,), paramName))
end

# Set the value of a model pointer parameter of type KNNModel.
function CLISetParamKNNModel(paramName::String, model::KNNModel)
  ccall((:CLI_SetParamKNNModelPtr, knnLibrary), Nothing, (Cstring, Ptr{Nothing}), paramName, model.ptr)
end

# Serialize a model to the given stream.
function serializeKNNModel(stream::IO, model::KNNModel)
  buf_len = UInt[0]
  buf_ptr = ccall((:SerializeKNNModelPtr, knnLibrary), Ptr{UInt8}, (Ptr{Nothing}, Ptr{UInt}), model.ptr, Base.pointer(buf_len))
  buf = Base.unsafe_wrap(Vector{UInt8}, buf_ptr, buf_len[1]; own=true)
  write(stream, buf)
end
# Deserialize a model from the given stream.
function deserializeKNNModel(stream::IO)::KNNModel
  buffer = read(stream)
  KNNModel(ccall((:DeserializeKNNModelPtr, knnLibrary), Ptr{Nothing}, (Ptr{UInt8}, UInt), Base.pointer(buffer), length(buffer)))
end
end # module

"""
    knn(; [algorithm, epsilon, input_model, k, leaf_size, query, random_basis, reference, rho, seed, tau, tree_type, true_distances, true_neighbors, verbose])

This program will calculate the k-nearest-neighbors of a set of points using
kd-trees or cover trees (cover tree support is experimental and may be slow).
You may specify a separate set of reference points and query points, or just a
reference set which will be used as both the reference and query set.

For example, the following command will calculate the 5 nearest neighbors of
each point in `input` and store the distances in `distances` and the neighbors
in `neighbors`: 

```julia
julia> using CSV
julia> input = CSV.read("input.csv")
julia> distances, neighbors, _ = knn(k=5, reference=input)
```

The output is organized such that row i and column j in the neighbors output
matrix corresponds to the index of the point in the reference set which is the
j'th nearest neighbor from the point in the query set with index i.  Row j and
column i in the distances output matrix corresponds to the distance between
those two points.

# Arguments

 - `algorithm::String`: Type of neighbor search: 'naive', 'single_tree',
      'dual_tree', 'greedy'.  Default value `dual_tree`.
      
 - `epsilon::Float64`: If specified, will do approximate nearest neighbor
      search with given relative error.  Default value `0`.
      
 - `input_model::unknown_`: Pre-trained kNN model.
 - `k::Int`: Number of nearest neighbors to find.  Default value `0`.

 - `leaf_size::Int`: Leaf size for tree building (used for kd-trees, vp
      trees, random projection trees, UB trees, R trees, R* trees, X trees,
      Hilbert R trees, R+ trees, R++ trees, spill trees, and octrees).  Default
      value `20`.
      
 - `query::Array{Float64, 2}`: Matrix containing query points (optional).
 - `random_basis::Bool`: Before tree-building, project the data onto a
      random orthogonal basis.  Default value `false`.
      
 - `reference::Array{Float64, 2}`: Matrix containing the reference
      dataset.
 - `rho::Float64`: Balance threshold (only valid for spill trees). 
      Default value `0.7`.
      
 - `seed::Int`: Random seed (if 0, std::time(NULL) is used).  Default
      value `0`.
      
 - `tau::Float64`: Overlapping size (only valid for spill trees).  Default
      value `0`.
      
 - `tree_type::String`: Type of tree to use: 'kd', 'vp', 'rp', 'max-rp',
      'ub', 'cover', 'r', 'r-star', 'x', 'ball', 'hilbert-r', 'r-plus',
      'r-plus-plus', 'spill', 'oct'.  Default value `kd`.
      
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
 - `output_model::unknown_`: If specified, the kNN model will be output
      here.

"""
function knn(;
             algorithm::Union{String, Missing} = missing,
             epsilon::Union{Float64, Missing} = missing,
             input_model::Union{KNNModel, Missing} = missing,
             k::Union{Int, Missing} = missing,
             leaf_size::Union{Int, Missing} = missing,
             query = missing,
             random_basis::Union{Bool, Missing} = missing,
             reference = missing,
             rho::Union{Float64, Missing} = missing,
             seed::Union{Int, Missing} = missing,
             tau::Union{Float64, Missing} = missing,
             tree_type::Union{String, Missing} = missing,
             true_distances = missing,
             true_neighbors = missing,
             verbose::Union{Bool, Missing} = missing,
             points_are_rows::Bool = true)
  # Force the symbols to load.
  ccall((:loadSymbols, knnLibrary), Nothing, ());

  CLIRestoreSettings("k-Nearest-Neighbors Search")

  # Process each input argument before calling mlpackMain().
  if !ismissing(algorithm)
    CLISetParam("algorithm", convert(String, algorithm))
  end
  if !ismissing(epsilon)
    CLISetParam("epsilon", convert(Float64, epsilon))
  end
  if !ismissing(input_model)
    knn_internal.CLISetParamKNNModel("input_model", convert(KNNModel, input_model))
  end
  if !ismissing(k)
    CLISetParam("k", convert(Int, k))
  end
  if !ismissing(leaf_size)
    CLISetParam("leaf_size", convert(Int, leaf_size))
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
  if !ismissing(rho)
    CLISetParam("rho", convert(Float64, rho))
  end
  if !ismissing(seed)
    CLISetParam("seed", convert(Int, seed))
  end
  if !ismissing(tau)
    CLISetParam("tau", convert(Float64, tau))
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
  knn_mlpackMain()

  return CLIGetParamMat("distances", points_are_rows),
         CLIGetParamUMat("neighbors", points_are_rows),
         knn_internal.CLIGetParamKNNModel("output_model")
end
