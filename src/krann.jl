export krann

import ..RANNModel

using mlpack._Internal.io

import mlpack_jll
const krannLibrary = mlpack_jll.libmlpack_julia_krann

# Call the C binding of the mlpack krann binding.
function krann_mlpackMain()
  success = ccall((:krann, krannLibrary), Bool, ())
  if !success
    # Throw an exception---false means there was a C++ exception.
    throw(ErrorException("mlpack binding error; see output"))
  end
end

" Internal module to hold utility functions. "
module krann_internal
  import ..krannLibrary

import ...RANNModel

# Get the value of a model pointer parameter of type RANNModel.
function IOGetParamRANNModel(paramName::String)::RANNModel
  RANNModel(ccall((:IO_GetParamRANNModelPtr, krannLibrary), Ptr{Nothing}, (Cstring,), paramName))
end

# Set the value of a model pointer parameter of type RANNModel.
function IOSetParamRANNModel(paramName::String, model::RANNModel)
  ccall((:IO_SetParamRANNModelPtr, krannLibrary), Nothing, (Cstring, Ptr{Nothing}), paramName, model.ptr)
end

# Serialize a model to the given stream.
function serializeRANNModel(stream::IO, model::RANNModel)
  buf_len = UInt[0]
  buf_ptr = ccall((:SerializeRANNModelPtr, krannLibrary), Ptr{UInt8}, (Ptr{Nothing}, Ptr{UInt}), model.ptr, pointer(buf_len))
  buf = Base.unsafe_wrap(Vector{UInt8}, buf_ptr, buf_len[1]; own=true)
  write(stream, buf)
end
# Deserialize a model from the given stream.
function deserializeRANNModel(stream::IO)::RANNModel
  buffer = read(stream)
  GC.@preserve buffer RANNModel(ccall((:DeserializeRANNModelPtr, krannLibrary), Ptr{Nothing}, (Ptr{UInt8}, UInt), pointer(buffer), length(buffer)))
end
end # module

"""
    krann(; [alpha, first_leaf_exact, input_model, k, leaf_size, naive, query, random_basis, reference, sample_at_leaves, seed, single_mode, single_sample_limit, tau, tree_type, verbose])

This program will calculate the k rank-approximate-nearest-neighbors of a set of
points. You may specify a separate set of reference points and query points, or
just a reference set which will be used as both the reference and query set. You
must specify the rank approximation (in %) (and optionally the success
probability).

For example, the following will return 5 neighbors from the top 0.1% of the data
(with probability 0.95) for each point in `input` and store the distances in
`distances` and the neighbors in `neighbors.csv`:

```julia
julia> using CSV
julia> input = CSV.read("input.csv")
julia> distances, neighbors, _ = krann(k=5, reference=input,
            tau=0.1)
```

Note that tau must be set such that the number of points in the corresponding
percentile of the data is greater than k.  Thus, if we choose tau = 0.1 with a
dataset of 1000 points and k = 5, then we are attempting to choose 5 nearest
neighbors out of the closest 1 point -- this is invalid and the program will
terminate with an error message.

The output matrices are organized such that row i and column j in the neighbors
output file corresponds to the index of the point in the reference set which is
the i'th nearest neighbor from the point in the query set with index j.  Row i
and column j in the distances output file corresponds to the distance between
those two points.

# Arguments

 - `alpha::Float64`: The desired success probability.  Default value
      `0.95`.
      
 - `first_leaf_exact::Bool`: The flag to trigger sampling only after
      exactly exploring the first leaf.  Default value `false`.
      
 - `input_model::RANNModel`: Pre-trained kNN model.
 - `k::Int`: Number of nearest neighbors to find.  Default value `0`.

 - `leaf_size::Int`: Leaf size for tree building (used for kd-trees, UB
      trees, R trees, R* trees, X trees, Hilbert R trees, R+ trees, R++ trees,
      and octrees).  Default value `20`.
      
 - `naive::Bool`: If true, sampling will be done without using a tree. 
      Default value `false`.
      
 - `query::Array{Float64, 2}`: Matrix containing query points (optional).
 - `random_basis::Bool`: Before tree-building, project the data onto a
      random orthogonal basis.  Default value `false`.
      
 - `reference::Array{Float64, 2}`: Matrix containing the reference
      dataset.
 - `sample_at_leaves::Bool`: The flag to trigger sampling at leaves. 
      Default value `false`.
      
 - `seed::Int`: Random seed (if 0, std::time(NULL) is used).  Default
      value `0`.
      
 - `single_mode::Bool`: If true, single-tree search is used (as opposed to
      dual-tree search.  Default value `false`.
      
 - `single_sample_limit::Int`: The limit on the maximum number of samples
      (and hence the largest node you can approximate).  Default value `20`.
      
 - `tau::Float64`: The allowed rank-error in terms of the percentile of
      the data.  Default value `5`.
      
 - `tree_type::String`: Type of tree to use: 'kd', 'ub', 'cover', 'r',
      'x', 'r-star', 'hilbert-r', 'r-plus', 'r-plus-plus', 'oct'.  Default value
      `kd`.
      
 - `verbose::Bool`: Display informational messages and the full list of
      parameters and timers at the end of execution.  Default value `false`.
      

# Return values

 - `distances::Array{Float64, 2}`: Matrix to output distances into.
 - `neighbors::Array{Int, 2}`: Matrix to output neighbors into.
 - `output_model::RANNModel`: If specified, the kNN model will be output
      here.

"""
function krann(;
               alpha::Union{Float64, Missing} = missing,
               first_leaf_exact::Union{Bool, Missing} = missing,
               input_model::Union{RANNModel, Missing} = missing,
               k::Union{Int, Missing} = missing,
               leaf_size::Union{Int, Missing} = missing,
               naive::Union{Bool, Missing} = missing,
               query = missing,
               random_basis::Union{Bool, Missing} = missing,
               reference = missing,
               sample_at_leaves::Union{Bool, Missing} = missing,
               seed::Union{Int, Missing} = missing,
               single_mode::Union{Bool, Missing} = missing,
               single_sample_limit::Union{Int, Missing} = missing,
               tau::Union{Float64, Missing} = missing,
               tree_type::Union{String, Missing} = missing,
               verbose::Union{Bool, Missing} = missing,
               points_are_rows::Bool = true)
  # Force the symbols to load.
  ccall((:loadSymbols, krannLibrary), Nothing, ());

  IORestoreSettings("K-Rank-Approximate-Nearest-Neighbors (kRANN)")

  # Process each input argument before calling mlpackMain().
  if !ismissing(alpha)
    IOSetParam("alpha", convert(Float64, alpha))
  end
  if !ismissing(first_leaf_exact)
    IOSetParam("first_leaf_exact", convert(Bool, first_leaf_exact))
  end
  if !ismissing(input_model)
    krann_internal.IOSetParamRANNModel("input_model", convert(RANNModel, input_model))
  end
  if !ismissing(k)
    IOSetParam("k", convert(Int, k))
  end
  if !ismissing(leaf_size)
    IOSetParam("leaf_size", convert(Int, leaf_size))
  end
  if !ismissing(naive)
    IOSetParam("naive", convert(Bool, naive))
  end
  if !ismissing(query)
    IOSetParamMat("query", query, points_are_rows)
  end
  if !ismissing(random_basis)
    IOSetParam("random_basis", convert(Bool, random_basis))
  end
  if !ismissing(reference)
    IOSetParamMat("reference", reference, points_are_rows)
  end
  if !ismissing(sample_at_leaves)
    IOSetParam("sample_at_leaves", convert(Bool, sample_at_leaves))
  end
  if !ismissing(seed)
    IOSetParam("seed", convert(Int, seed))
  end
  if !ismissing(single_mode)
    IOSetParam("single_mode", convert(Bool, single_mode))
  end
  if !ismissing(single_sample_limit)
    IOSetParam("single_sample_limit", convert(Int, single_sample_limit))
  end
  if !ismissing(tau)
    IOSetParam("tau", convert(Float64, tau))
  end
  if !ismissing(tree_type)
    IOSetParam("tree_type", convert(String, tree_type))
  end
  if verbose !== nothing && verbose === true
    IOEnableVerbose()
  else
    IODisableVerbose()
  end

  IOSetPassed("distances")
  IOSetPassed("neighbors")
  IOSetPassed("output_model")
  # Call the program.
  krann_mlpackMain()

  return IOGetParamMat("distances", points_are_rows),
         IOGetParamUMat("neighbors", points_are_rows),
         krann_internal.IOGetParamRANNModel("output_model")
end
