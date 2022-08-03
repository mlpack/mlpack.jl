export fastmks

import ..FastMKSModel

using mlpack._Internal.io

import mlpack_jll
const fastmksLibrary = mlpack_jll.libmlpack_julia_fastmks

# Call the C binding of the mlpack fastmks binding.
function fastmks_mlpackMain()
  success = ccall((:fastmks, fastmksLibrary), Bool, ())
  if !success
    # Throw an exception---false means there was a C++ exception.
    throw(ErrorException("mlpack binding error; see output"))
  end
end

" Internal module to hold utility functions. "
module fastmks_internal
  import ..fastmksLibrary

import ...FastMKSModel

# Get the value of a model pointer parameter of type FastMKSModel.
function IOGetParamFastMKSModel(paramName::String)::FastMKSModel
  FastMKSModel(ccall((:IO_GetParamFastMKSModelPtr, fastmksLibrary), Ptr{Nothing}, (Cstring,), paramName))
end

# Set the value of a model pointer parameter of type FastMKSModel.
function IOSetParamFastMKSModel(paramName::String, model::FastMKSModel)
  ccall((:IO_SetParamFastMKSModelPtr, fastmksLibrary), Nothing, (Cstring, Ptr{Nothing}), paramName, model.ptr)
end

# Serialize a model to the given stream.
function serializeFastMKSModel(stream::IO, model::FastMKSModel)
  buf_len = UInt[0]
  buf_ptr = ccall((:SerializeFastMKSModelPtr, fastmksLibrary), Ptr{UInt8}, (Ptr{Nothing}, Ptr{UInt}), model.ptr, pointer(buf_len))
  buf = Base.unsafe_wrap(Vector{UInt8}, buf_ptr, buf_len[1]; own=true)
  write(stream, buf)
end
# Deserialize a model from the given stream.
function deserializeFastMKSModel(stream::IO)::FastMKSModel
  buffer = read(stream)
  GC.@preserve buffer FastMKSModel(ccall((:DeserializeFastMKSModelPtr, fastmksLibrary), Ptr{Nothing}, (Ptr{UInt8}, UInt), pointer(buffer), length(buffer)))
end
end # module

"""
    fastmks(; [bandwidth, base, degree, input_model, k, kernel, naive, offset, query, reference, scale, single, verbose])

This program will find the k maximum kernels of a set of points, using a query
set and a reference set (which can optionally be the same set). More
specifically, for each point in the query set, the k points in the reference set
with maximum kernel evaluations are found.  The kernel function used is
specified with the `kernel` parameter.

For example, the following command will calculate, for each point in the query
set `query`, the five points in the reference set `reference` with maximum
kernel evaluation using the linear kernel.  The kernel evaluations may be saved
with the  `kernels` output parameter and the indices may be saved with the
`indices` output parameter.

```julia
julia> using CSV
julia> reference = CSV.read("reference.csv")
julia> query = CSV.read("query.csv")
julia> indices, kernels, _ = fastmks(k=5, kernel="linear",
            query=query, reference=reference)
```

The output matrices are organized such that row i and column j in the indices
matrix corresponds to the index of the point in the reference set that has j'th
largest kernel evaluation with the point in the query set with index i.  Row i
and column j in the kernels matrix corresponds to the kernel evaluation between
those two points.

This program performs FastMKS using a cover tree.  The base used to build the
cover tree can be specified with the `base` parameter.

# Arguments

 - `bandwidth::Float64`: Bandwidth (for Gaussian, Epanechnikov, and
      triangular kernels).  Default value `1`.
      
 - `base::Float64`: Base to use during cover tree construction.  Default
      value `2`.
      
 - `degree::Float64`: Degree of polynomial kernel.  Default value `2`.

 - `input_model::FastMKSModel`: Input FastMKS model to use.
 - `k::Int`: Number of maximum kernels to find.  Default value `0`.

 - `kernel::String`: Kernel type to use: 'linear', 'polynomial', 'cosine',
      'gaussian', 'epanechnikov', 'triangular', 'hyptan'.  Default value
      `linear`.
      
 - `naive::Bool`: If true, O(n^2) naive mode is used for computation. 
      Default value `false`.
      
 - `offset::Float64`: Offset of kernel (for polynomial and hyptan
      kernels).  Default value `0`.
      
 - `query::Array{Float64, 2}`: The query dataset.
 - `reference::Array{Float64, 2}`: The reference dataset.
 - `scale::Float64`: Scale of kernel (for hyptan kernel).  Default value
      `1`.
      
 - `single::Bool`: If true, single-tree search is used (as opposed to
      dual-tree search.  Default value `false`.
      
 - `verbose::Bool`: Display informational messages and the full list of
      parameters and timers at the end of execution.  Default value `false`.
      

# Return values

 - `indices::Array{Int, 2}`: Output matrix of indices.
 - `kernels::Array{Float64, 2}`: Output matrix of kernels.
 - `output_model::FastMKSModel`: Output for FastMKS model.

"""
function fastmks(;
                 bandwidth::Union{Float64, Missing} = missing,
                 base::Union{Float64, Missing} = missing,
                 degree::Union{Float64, Missing} = missing,
                 input_model::Union{FastMKSModel, Missing} = missing,
                 k::Union{Int, Missing} = missing,
                 kernel::Union{String, Missing} = missing,
                 naive::Union{Bool, Missing} = missing,
                 offset::Union{Float64, Missing} = missing,
                 query = missing,
                 reference = missing,
                 scale::Union{Float64, Missing} = missing,
                 single::Union{Bool, Missing} = missing,
                 verbose::Union{Bool, Missing} = missing,
                 points_are_rows::Bool = true)
  # Force the symbols to load.
  ccall((:loadSymbols, fastmksLibrary), Nothing, ());

  IORestoreSettings("FastMKS (Fast Max-Kernel Search)")

  # Process each input argument before calling mlpackMain().
  if !ismissing(bandwidth)
    IOSetParam("bandwidth", convert(Float64, bandwidth))
  end
  if !ismissing(base)
    IOSetParam("base", convert(Float64, base))
  end
  if !ismissing(degree)
    IOSetParam("degree", convert(Float64, degree))
  end
  if !ismissing(input_model)
    fastmks_internal.IOSetParamFastMKSModel("input_model", convert(FastMKSModel, input_model))
  end
  if !ismissing(k)
    IOSetParam("k", convert(Int, k))
  end
  if !ismissing(kernel)
    IOSetParam("kernel", convert(String, kernel))
  end
  if !ismissing(naive)
    IOSetParam("naive", convert(Bool, naive))
  end
  if !ismissing(offset)
    IOSetParam("offset", convert(Float64, offset))
  end
  if !ismissing(query)
    IOSetParamMat("query", query, points_are_rows)
  end
  if !ismissing(reference)
    IOSetParamMat("reference", reference, points_are_rows)
  end
  if !ismissing(scale)
    IOSetParam("scale", convert(Float64, scale))
  end
  if !ismissing(single)
    IOSetParam("single", convert(Bool, single))
  end
  if verbose !== nothing && verbose === true
    IOEnableVerbose()
  else
    IODisableVerbose()
  end

  IOSetPassed("indices")
  IOSetPassed("kernels")
  IOSetPassed("output_model")
  # Call the program.
  fastmks_mlpackMain()

  return IOGetParamUMat("indices", points_are_rows),
         IOGetParamMat("kernels", points_are_rows),
         fastmks_internal.IOGetParamFastMKSModel("output_model")
end
