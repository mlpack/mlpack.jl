export knn

import ..KNNModel

using mlpack._Internal.params

import mlpack_jll
const knnLibrary = mlpack_jll.libmlpack_julia_knn

# Call the C binding of the mlpack knn binding.
function call_knn(p, t)
  success = ccall((:mlpack_knn, knnLibrary), Bool, (Ptr{Nothing}, Ptr{Nothing}), p, t)
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
function GetParamKNNModel(params::Ptr{Nothing}, paramName::String, modelPtrs::Set{Ptr{Nothing}})::KNNModel
  ptr = ccall((:GetParamKNNModelPtr, knnLibrary), Ptr{Nothing}, (Ptr{Nothing}, Cstring,), params, paramName)
  return KNNModel(ptr; finalize=!(ptr in modelPtrs))
end

# Set the value of a model pointer parameter of type KNNModel.
function SetParamKNNModel(params::Ptr{Nothing}, paramName::String, model::KNNModel)
  ccall((:SetParamKNNModelPtr, knnLibrary), Nothing, (Ptr{Nothing}, Cstring, Ptr{Nothing}), params, paramName, model.ptr)
end

# Delete an instantiated model pointer.
function DeleteKNNModel(ptr::Ptr{Nothing})
  ccall((:DeleteKNNModelPtr, knnLibrary), Nothing, (Ptr{Nothing},), ptr)
end

# Serialize a model to the given stream.
function serializeKNNModel(stream::IO, model::KNNModel)
  buf_len = UInt[0]
  buf_ptr = ccall((:SerializeKNNModelPtr, knnLibrary), Ptr{UInt8}, (Ptr{Nothing}, Ptr{UInt}), model.ptr, pointer(buf_len))
  buf = Base.unsafe_wrap(Vector{UInt8}, buf_ptr, buf_len[1]; own=true)
  write(stream, buf_len[1])
  write(stream, buf)
end
# Deserialize a model from the given stream.
function deserializeKNNModel(stream::IO)::KNNModel
  buf_len = read(stream, UInt)
  buffer = read(stream, buf_len)
  GC.@preserve buffer KNNModel(ccall((:DeserializeKNNModelPtr, knnLibrary), Ptr{Nothing}, (Ptr{UInt8}, UInt), pointer(buffer), length(buffer)))
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
      
 - `input_model::KNNModel`: Pre-trained kNN model.
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
 - `output_model::KNNModel`: If specified, the kNN model will be output
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

  # Create the set of model pointers to avoid setting multiple finalizers.
  modelPtrs = Set{Ptr{Nothing}}()

  p = GetParameters("knn")
  t = Timers()

  juliaOwnedMemory = Set{Ptr{Nothing}}()
  # Process each input argument before calling mlpackMain().
  if !ismissing(algorithm)
    SetParam(p, "algorithm", convert(String, algorithm))
  end
  if !ismissing(epsilon)
    SetParam(p, "epsilon", convert(Float64, epsilon))
  end
  if !ismissing(input_model)
    push!(modelPtrs, convert(KNNModel, input_model).ptr)
    knn_internal.SetParamKNNModel(p, "input_model", convert(KNNModel, input_model))
  end
  if !ismissing(k)
    SetParam(p, "k", convert(Int, k))
  end
  if !ismissing(leaf_size)
    SetParam(p, "leaf_size", convert(Int, leaf_size))
  end
  if !ismissing(query)
    SetParamMat(p, "query", query, points_are_rows, juliaOwnedMemory)
  end
  if !ismissing(random_basis)
    SetParam(p, "random_basis", convert(Bool, random_basis))
  end
  if !ismissing(reference)
    SetParamMat(p, "reference", reference, points_are_rows, juliaOwnedMemory)
  end
  if !ismissing(rho)
    SetParam(p, "rho", convert(Float64, rho))
  end
  if !ismissing(seed)
    SetParam(p, "seed", convert(Int, seed))
  end
  if !ismissing(tau)
    SetParam(p, "tau", convert(Float64, tau))
  end
  if !ismissing(tree_type)
    SetParam(p, "tree_type", convert(String, tree_type))
  end
  if !ismissing(true_distances)
    SetParamMat(p, "true_distances", true_distances, points_are_rows, juliaOwnedMemory)
  end
  if !ismissing(true_neighbors)
    SetParamUMat(p, "true_neighbors", true_neighbors, points_are_rows, juliaOwnedMemory)
  end
  if verbose !== nothing && verbose === true
    EnableVerbose()
  else
    DisableVerbose()
  end

  SetPassed(p, "distances")
  SetPassed(p, "neighbors")
  SetPassed(p, "output_model")
  # Call the program.
  call_knn(p, t)

  results = (GetParamMat(p, "distances", points_are_rows, juliaOwnedMemory),
             GetParamUMat(p, "neighbors", points_are_rows, juliaOwnedMemory),
             knn_internal.GetParamKNNModel(p, "output_model", modelPtrs))

  # We are responsible for cleaning up the `p` and `t` objects.
  DeleteParameters(p)
  DeleteTimers(t)

  return results
end
