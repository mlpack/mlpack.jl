export lsh

import ..LSHSearch

using mlpack._Internal.params

import mlpack_jll
const lshLibrary = mlpack_jll.libmlpack_julia_lsh

# Call the C binding of the mlpack lsh binding.
function call_lsh(p, t)
  success = ccall((:mlpack_lsh, lshLibrary), Bool, (Ptr{Nothing}, Ptr{Nothing}), p, t)
  if !success
    # Throw an exception---false means there was a C++ exception.
    throw(ErrorException("mlpack binding error; see output"))
  end
end

" Internal module to hold utility functions. "
module lsh_internal
  import ..lshLibrary

import ...LSHSearch

# Get the value of a model pointer parameter of type LSHSearch.
function GetParamLSHSearch(params::Ptr{Nothing}, paramName::String, modelPtrs::Set{Ptr{Nothing}})::LSHSearch
  ptr = ccall((:GetParamLSHSearchPtr, lshLibrary), Ptr{Nothing}, (Ptr{Nothing}, Cstring,), params, paramName)
  return LSHSearch(ptr; finalize=!(ptr in modelPtrs))
end

# Set the value of a model pointer parameter of type LSHSearch.
function SetParamLSHSearch(params::Ptr{Nothing}, paramName::String, model::LSHSearch)
  ccall((:SetParamLSHSearchPtr, lshLibrary), Nothing, (Ptr{Nothing}, Cstring, Ptr{Nothing}), params, paramName, model.ptr)
end

# Delete an instantiated model pointer.
function DeleteLSHSearch(ptr::Ptr{Nothing})
  ccall((:DeleteLSHSearchPtr, lshLibrary), Nothing, (Ptr{Nothing},), ptr)
end

# Serialize a model to the given stream.
function serializeLSHSearch(stream::IO, model::LSHSearch)
  buf_len = UInt[0]
  buf_ptr = ccall((:SerializeLSHSearchPtr, lshLibrary), Ptr{UInt8}, (Ptr{Nothing}, Ptr{UInt}), model.ptr, pointer(buf_len))
  buf = Base.unsafe_wrap(Vector{UInt8}, buf_ptr, buf_len[1]; own=true)
  write(stream, buf_len[1])
  write(stream, buf)
end
# Deserialize a model from the given stream.
function deserializeLSHSearch(stream::IO)::LSHSearch
  buf_len = read(stream, UInt)
  buffer = read(stream, buf_len)
  GC.@preserve buffer LSHSearch(ccall((:DeserializeLSHSearchPtr, lshLibrary), Ptr{Nothing}, (Ptr{UInt8}, UInt), pointer(buffer), length(buffer)))
end
end # module

"""
    lsh(; [bucket_size, hash_width, input_model, k, num_probes, projections, query, reference, second_hash_size, seed, tables, true_neighbors, verbose])

This program will calculate the k approximate-nearest-neighbors of a set of
points using locality-sensitive hashing. You may specify a separate set of
reference points and query points, or just a reference set which will be used as
both the reference and query set. 

For example, the following will return 5 neighbors from the data for each point
in `input` and store the distances in `distances` and the neighbors in
`neighbors`:

```julia
julia> using CSV
julia> input = CSV.read("input.csv")
julia> distances, neighbors, _ = lsh(k=5, reference=input)
```

The output is organized such that row i and column j in the neighbors output
corresponds to the index of the point in the reference set which is the j'th
nearest neighbor from the point in the query set with index i.  Row j and column
i in the distances output file corresponds to the distance between those two
points.

Because this is approximate-nearest-neighbors search, results may be different
from run to run.  Thus, the `seed` parameter can be specified to set the random
seed.

This program also has many other parameters to control its functionality; see
the parameter-specific documentation for more information.

# Arguments

 - `bucket_size::Int`: The size of a bucket in the second level hash. 
      Default value `500`.
      
 - `hash_width::Float64`: The hash width for the first-level hashing in
      the LSH preprocessing. By default, the LSH class automatically estimates a
      hash width for its use.  Default value `0`.
      
 - `input_model::LSHSearch`: Input LSH model.
 - `k::Int`: Number of nearest neighbors to find.  Default value `0`.

 - `num_probes::Int`: Number of additional probes for multiprobe LSH; if
      0, traditional LSH is used.  Default value `0`.
      
 - `projections::Int`: The number of hash functions for each table 
      Default value `10`.
      
 - `query::Array{Float64, 2}`: Matrix containing query points (optional).
 - `reference::Array{Float64, 2}`: Matrix containing the reference
      dataset.
 - `second_hash_size::Int`: The size of the second level hash table. 
      Default value `99901`.
      
 - `seed::Int`: Random seed.  If 0, 'std::time(NULL)' is used.  Default
      value `0`.
      
 - `tables::Int`: The number of hash tables to be used.  Default value
      `30`.
      
 - `true_neighbors::Array{Int, 2}`: Matrix of true neighbors to compute
      recall with (the recall is printed when -v is specified).
 - `verbose::Bool`: Display informational messages and the full list of
      parameters and timers at the end of execution.  Default value `false`.
      

# Return values

 - `distances::Array{Float64, 2}`: Matrix to output distances into.
 - `neighbors::Array{Int, 2}`: Matrix to output neighbors into.
 - `output_model::LSHSearch`: Output for trained LSH model.

"""
function lsh(;
             bucket_size::Union{Int, Missing} = missing,
             hash_width::Union{Float64, Missing} = missing,
             input_model::Union{LSHSearch, Missing} = missing,
             k::Union{Int, Missing} = missing,
             num_probes::Union{Int, Missing} = missing,
             projections::Union{Int, Missing} = missing,
             query = missing,
             reference = missing,
             second_hash_size::Union{Int, Missing} = missing,
             seed::Union{Int, Missing} = missing,
             tables::Union{Int, Missing} = missing,
             true_neighbors = missing,
             verbose::Union{Bool, Missing} = missing,
             points_are_rows::Bool = true)
  # Force the symbols to load.
  ccall((:loadSymbols, lshLibrary), Nothing, ());

  # Create the set of model pointers to avoid setting multiple finalizers.
  modelPtrs = Set{Ptr{Nothing}}()

  p = GetParameters("lsh")
  t = Timers()

  juliaOwnedMemory = Set{Ptr{Nothing}}()
  # Process each input argument before calling mlpackMain().
  if !ismissing(bucket_size)
    SetParam(p, "bucket_size", convert(Int, bucket_size))
  end
  if !ismissing(hash_width)
    SetParam(p, "hash_width", convert(Float64, hash_width))
  end
  if !ismissing(input_model)
    push!(modelPtrs, convert(LSHSearch, input_model).ptr)
    lsh_internal.SetParamLSHSearch(p, "input_model", convert(LSHSearch, input_model))
  end
  if !ismissing(k)
    SetParam(p, "k", convert(Int, k))
  end
  if !ismissing(num_probes)
    SetParam(p, "num_probes", convert(Int, num_probes))
  end
  if !ismissing(projections)
    SetParam(p, "projections", convert(Int, projections))
  end
  if !ismissing(query)
    SetParamMat(p, "query", query, points_are_rows, false, juliaOwnedMemory)
  end
  if !ismissing(reference)
    SetParamMat(p, "reference", reference, points_are_rows, false, juliaOwnedMemory)
  end
  if !ismissing(second_hash_size)
    SetParam(p, "second_hash_size", convert(Int, second_hash_size))
  end
  if !ismissing(seed)
    SetParam(p, "seed", convert(Int, seed))
  end
  if !ismissing(tables)
    SetParam(p, "tables", convert(Int, tables))
  end
  if !ismissing(true_neighbors)
    SetParamUMat(p, "true_neighbors", true_neighbors, points_are_rows, false, juliaOwnedMemory)
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
  call_lsh(p, t)

  results = (GetParamMat(p, "distances", points_are_rows, juliaOwnedMemory),
             GetParamUMat(p, "neighbors", points_are_rows, juliaOwnedMemory),
             lsh_internal.GetParamLSHSearch(p, "output_model", modelPtrs))

  # We are responsible for cleaning up the `p` and `t` objects.
  DeleteParameters(p)
  DeleteTimers(t)

  return results
end
