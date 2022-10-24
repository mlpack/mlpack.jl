export krann

import ..RAModel

using mlpack._Internal.params

import mlpack_jll
const krannLibrary = mlpack_jll.libmlpack_julia_krann

# Call the C binding of the mlpack krann binding.
function call_krann(p, t)
  success = ccall((:mlpack_krann, krannLibrary), Bool, (Ptr{Nothing}, Ptr{Nothing}), p, t)
  if !success
    # Throw an exception---false means there was a C++ exception.
    throw(ErrorException("mlpack binding error; see output"))
  end
end

" Internal module to hold utility functions. "
module krann_internal
  import ..krannLibrary

import ...RAModel

# Get the value of a model pointer parameter of type RAModel.
function GetParamRAModel(params::Ptr{Nothing}, paramName::String, modelPtrs::Set{Ptr{Nothing}})::RAModel
  ptr = ccall((:GetParamRAModelPtr, krannLibrary), Ptr{Nothing}, (Ptr{Nothing}, Cstring,), params, paramName)
  return RAModel(ptr; finalize=!(ptr in modelPtrs))
end

# Set the value of a model pointer parameter of type RAModel.
function SetParamRAModel(params::Ptr{Nothing}, paramName::String, model::RAModel)
  ccall((:SetParamRAModelPtr, krannLibrary), Nothing, (Ptr{Nothing}, Cstring, Ptr{Nothing}), params, paramName, model.ptr)
end

# Delete an instantiated model pointer.
function DeleteRAModel(ptr::Ptr{Nothing})
  ccall((:DeleteRAModelPtr, krannLibrary), Nothing, (Ptr{Nothing},), ptr)
end

# Serialize a model to the given stream.
function serializeRAModel(stream::IO, model::RAModel)
  buf_len = UInt[0]
  buf_ptr = ccall((:SerializeRAModelPtr, krannLibrary), Ptr{UInt8}, (Ptr{Nothing}, Ptr{UInt}), model.ptr, pointer(buf_len))
  buf = Base.unsafe_wrap(Vector{UInt8}, buf_ptr, buf_len[1]; own=true)
  write(stream, buf_len[1])
  write(stream, buf)
end
# Deserialize a model from the given stream.
function deserializeRAModel(stream::IO)::RAModel
  buf_len = read(stream, UInt)
  buffer = read(stream, buf_len)
  GC.@preserve buffer RAModel(ccall((:DeserializeRAModelPtr, krannLibrary), Ptr{Nothing}, (Ptr{UInt8}, UInt), pointer(buffer), length(buffer)))
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
      
 - `input_model::RAModel`: Pre-trained kNN model.
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
 - `output_model::RAModel`: If specified, the kNN model will be output
      here.

"""
function krann(;
               alpha::Union{Float64, Missing} = missing,
               first_leaf_exact::Union{Bool, Missing} = missing,
               input_model::Union{RAModel, Missing} = missing,
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

  # Create the set of model pointers to avoid setting multiple finalizers.
  modelPtrs = Set{Ptr{Nothing}}()

  p = GetParameters("krann")
  t = Timers()

  juliaOwnedMemory = Set{Ptr{Nothing}}()
  # Process each input argument before calling mlpackMain().
  if !ismissing(alpha)
    SetParam(p, "alpha", convert(Float64, alpha))
  end
  if !ismissing(first_leaf_exact)
    SetParam(p, "first_leaf_exact", convert(Bool, first_leaf_exact))
  end
  if !ismissing(input_model)
    push!(modelPtrs, convert(RAModel, input_model).ptr)
    krann_internal.SetParamRAModel(p, "input_model", convert(RAModel, input_model))
  end
  if !ismissing(k)
    SetParam(p, "k", convert(Int, k))
  end
  if !ismissing(leaf_size)
    SetParam(p, "leaf_size", convert(Int, leaf_size))
  end
  if !ismissing(naive)
    SetParam(p, "naive", convert(Bool, naive))
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
  if !ismissing(sample_at_leaves)
    SetParam(p, "sample_at_leaves", convert(Bool, sample_at_leaves))
  end
  if !ismissing(seed)
    SetParam(p, "seed", convert(Int, seed))
  end
  if !ismissing(single_mode)
    SetParam(p, "single_mode", convert(Bool, single_mode))
  end
  if !ismissing(single_sample_limit)
    SetParam(p, "single_sample_limit", convert(Int, single_sample_limit))
  end
  if !ismissing(tau)
    SetParam(p, "tau", convert(Float64, tau))
  end
  if !ismissing(tree_type)
    SetParam(p, "tree_type", convert(String, tree_type))
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
  call_krann(p, t)

  results = (GetParamMat(p, "distances", points_are_rows, juliaOwnedMemory),
             GetParamUMat(p, "neighbors", points_are_rows, juliaOwnedMemory),
             krann_internal.GetParamRAModel(p, "output_model", modelPtrs))

  # We are responsible for cleaning up the `p` and `t` objects.
  DeleteParameters(p)
  DeleteTimers(t)

  return results
end
