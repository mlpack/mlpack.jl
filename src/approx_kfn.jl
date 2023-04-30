export approx_kfn

import ..ApproxKFNModel

using mlpack._Internal.params

import mlpack_jll
const approx_kfnLibrary = mlpack_jll.libmlpack_julia_approx_kfn

# Call the C binding of the mlpack approx_kfn binding.
function call_approx_kfn(p, t)
  success = ccall((:mlpack_approx_kfn, approx_kfnLibrary), Bool, (Ptr{Nothing}, Ptr{Nothing}), p, t)
  if !success
    # Throw an exception---false means there was a C++ exception.
    throw(ErrorException("mlpack binding error; see output"))
  end
end

" Internal module to hold utility functions. "
module approx_kfn_internal
  import ..approx_kfnLibrary

import ...ApproxKFNModel

# Get the value of a model pointer parameter of type ApproxKFNModel.
function GetParamApproxKFNModel(params::Ptr{Nothing}, paramName::String, modelPtrs::Set{Ptr{Nothing}})::ApproxKFNModel
  ptr = ccall((:GetParamApproxKFNModelPtr, approx_kfnLibrary), Ptr{Nothing}, (Ptr{Nothing}, Cstring,), params, paramName)
  return ApproxKFNModel(ptr; finalize=!(ptr in modelPtrs))
end

# Set the value of a model pointer parameter of type ApproxKFNModel.
function SetParamApproxKFNModel(params::Ptr{Nothing}, paramName::String, model::ApproxKFNModel)
  ccall((:SetParamApproxKFNModelPtr, approx_kfnLibrary), Nothing, (Ptr{Nothing}, Cstring, Ptr{Nothing}), params, paramName, model.ptr)
end

# Delete an instantiated model pointer.
function DeleteApproxKFNModel(ptr::Ptr{Nothing})
  ccall((:DeleteApproxKFNModelPtr, approx_kfnLibrary), Nothing, (Ptr{Nothing},), ptr)
end

# Serialize a model to the given stream.
function serializeApproxKFNModel(stream::IO, model::ApproxKFNModel)
  buf_len = UInt[0]
  buf_ptr = ccall((:SerializeApproxKFNModelPtr, approx_kfnLibrary), Ptr{UInt8}, (Ptr{Nothing}, Ptr{UInt}), model.ptr, pointer(buf_len))
  buf = Base.unsafe_wrap(Vector{UInt8}, buf_ptr, buf_len[1]; own=true)
  write(stream, buf_len[1])
  write(stream, buf)
end
# Deserialize a model from the given stream.
function deserializeApproxKFNModel(stream::IO)::ApproxKFNModel
  buf_len = read(stream, UInt)
  buffer = read(stream, buf_len)
  GC.@preserve buffer ApproxKFNModel(ccall((:DeserializeApproxKFNModelPtr, approx_kfnLibrary), Ptr{Nothing}, (Ptr{UInt8}, UInt), pointer(buffer), length(buffer)))
end
end # module

"""
    approx_kfn(; [algorithm, calculate_error, exact_distances, input_model, k, num_projections, num_tables, query, reference, verbose])

This program implements two strategies for furthest neighbor search. These
strategies are:

 - The 'qdafn' algorithm from "Approximate Furthest Neighbor in High Dimensions"
by R. Pagh, F. Silvestri, J. Sivertsen, and M. Skala, in Similarity Search and
Applications 2015 (SISAP).
 - The 'DrusillaSelect' algorithm from "Fast approximate furthest neighbors with
data-dependent candidate selection", by R.R. Curtin and A.B. Gardner, in
Similarity Search and Applications 2016 (SISAP).

These two strategies give approximate results for the furthest neighbor search
problem and can be used as fast replacements for other furthest neighbor
techniques such as those found in the mlpack_kfn program.  Note that typically,
the 'ds' algorithm requires far fewer tables and projections than the 'qdafn'
algorithm.

Specify a reference set (set to search in) with `reference`, specify a query set
with `query`, and specify algorithm parameters with `num_tables` and
`num_projections` (or don't and defaults will be used).  The algorithm to be
used (either 'ds'---the default---or 'qdafn')  may be specified with
`algorithm`.  Also specify the number of neighbors to search for with `k`.

Note that for 'qdafn' in lower dimensions, `num_projections` may need to be set
to a high value in order to return results for each query point.

If no query set is specified, the reference set will be used as the query set. 
The `output_model` output parameter may be used to store the built model, and an
input model may be loaded instead of specifying a reference set with the
`input_model` option.

Results for each query point can be stored with the `neighbors` and `distances`
output parameters.  Each row of these output matrices holds the k distances or
neighbor indices for each query point.

For example, to find the 5 approximate furthest neighbors with `reference_set`
as the reference set and `query_set` as the query set using DrusillaSelect,
storing the furthest neighbor indices to `neighbors` and the furthest neighbor
distances to `distances`, one could call

```julia
julia> using CSV
julia> query_set = CSV.read("query_set.csv")
julia> reference_set = CSV.read("reference_set.csv")
julia> distances, neighbors, _ = approx_kfn(algorithm="ds", k=5,
            query=query_set, reference=reference_set)
```

and to perform approximate all-furthest-neighbors search with k=1 on the set
`data` storing only the furthest neighbor distances to `distances`, one could
call

```julia
julia> using CSV
julia> reference_set = CSV.read("reference_set.csv")
julia> distances, _, _ = approx_kfn(k=1, reference=reference_set)
```

A trained model can be re-used.  If a model has been previously saved to
`model`, then we may find 3 approximate furthest neighbors on a query set
`new_query_set` using that model and store the furthest neighbor indices into
`neighbors` by calling

```julia
julia> using CSV
julia> new_query_set = CSV.read("new_query_set.csv")
julia> _, neighbors, _ = approx_kfn(input_model=model, k=3,
            query=new_query_set)
```

# Arguments

 - `algorithm::String`: Algorithm to use: 'ds' or 'qdafn'.  Default value
      `ds`.
      
 - `calculate_error::Bool`: If set, calculate the average distance error
      for the first furthest neighbor only.  Default value `false`.
      
 - `exact_distances::Array{Float64, 2}`: Matrix containing exact distances
      to furthest neighbors; this can be used to avoid explicit calculation when
      --calculate_error is set.
 - `input_model::ApproxKFNModel`: File containing input model.
 - `k::Int`: Number of furthest neighbors to search for.  Default value
      `0`.
      
 - `num_projections::Int`: Number of projections to use in each hash
      table.  Default value `5`.
      
 - `num_tables::Int`: Number of hash tables to use.  Default value `5`.

 - `query::Array{Float64, 2}`: Matrix containing query points.
 - `reference::Array{Float64, 2}`: Matrix containing the reference
      dataset.
 - `verbose::Bool`: Display informational messages and the full list of
      parameters and timers at the end of execution.  Default value `false`.
      

# Return values

 - `distances::Array{Float64, 2}`: Matrix to save furthest neighbor
      distances to.
 - `neighbors::Array{Int, 2}`: Matrix to save neighbor indices to.
 - `output_model::ApproxKFNModel`: File to save output model to.

"""
function approx_kfn(;
                    algorithm::Union{String, Missing} = missing,
                    calculate_error::Union{Bool, Missing} = missing,
                    exact_distances = missing,
                    input_model::Union{ApproxKFNModel, Missing} = missing,
                    k::Union{Int, Missing} = missing,
                    num_projections::Union{Int, Missing} = missing,
                    num_tables::Union{Int, Missing} = missing,
                    query = missing,
                    reference = missing,
                    verbose::Union{Bool, Missing} = missing,
                    points_are_rows::Bool = true)
  # Force the symbols to load.
  ccall((:loadSymbols, approx_kfnLibrary), Nothing, ());

  # Create the set of model pointers to avoid setting multiple finalizers.
  modelPtrs = Set{Ptr{Nothing}}()

  p = GetParameters("approx_kfn")
  t = Timers()

  juliaOwnedMemory = Set{Ptr{Nothing}}()
  # Process each input argument before calling mlpackMain().
  if !ismissing(algorithm)
    SetParam(p, "algorithm", convert(String, algorithm))
  end
  if !ismissing(calculate_error)
    SetParam(p, "calculate_error", convert(Bool, calculate_error))
  end
  if !ismissing(exact_distances)
    SetParamMat(p, "exact_distances", exact_distances, points_are_rows, false, juliaOwnedMemory)
  end
  if !ismissing(input_model)
    push!(modelPtrs, convert(ApproxKFNModel, input_model).ptr)
    approx_kfn_internal.SetParamApproxKFNModel(p, "input_model", convert(ApproxKFNModel, input_model))
  end
  if !ismissing(k)
    SetParam(p, "k", convert(Int, k))
  end
  if !ismissing(num_projections)
    SetParam(p, "num_projections", convert(Int, num_projections))
  end
  if !ismissing(num_tables)
    SetParam(p, "num_tables", convert(Int, num_tables))
  end
  if !ismissing(query)
    SetParamMat(p, "query", query, points_are_rows, false, juliaOwnedMemory)
  end
  if !ismissing(reference)
    SetParamMat(p, "reference", reference, points_are_rows, false, juliaOwnedMemory)
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
  call_approx_kfn(p, t)

  results = (GetParamMat(p, "distances", points_are_rows, juliaOwnedMemory),
             GetParamUMat(p, "neighbors", points_are_rows, juliaOwnedMemory),
             approx_kfn_internal.GetParamApproxKFNModel(p, "output_model", modelPtrs))

  # We are responsible for cleaning up the `p` and `t` objects.
  DeleteParameters(p)
  DeleteTimers(t)

  return results
end
