export fastmks

using mlpack.util.cli

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

" Get the value of a model pointer parameter of type FastMKSModel."
function CLIGetParamFastMKSModelPtr(paramName::String)
  return ccall((:CLI_GetParamFastMKSModelPtr, fastmksLibrary), Ptr{Nothing}, (Cstring,), paramName)
end

" Set the value of a model pointer parameter of type FastMKSModel."
function CLISetParamFastMKSModelPtr(paramName::String, ptr::Ptr{Nothing})
  ccall((:CLI_SetParamFastMKSModelPtr, fastmksLibrary), Nothing, (Cstring, Ptr{Nothing}), paramName, ptr)
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

julia> using CSV
julia> reference = CSV.read("reference.csv")
julia> query = CSV.read("query.csv")
julia> indices, kernels, _ = fastmks(k=5, kernel="linear",
            query=query, reference=reference)

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

 - `input_model::unknown_`: Input FastMKS model to use.
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

 - `indices::Array{Int64, 2}`: Output matrix of indices.
 - `kernels::Array{Float64, 2}`: Output matrix of kernels.
 - `output_model::unknown_`: Output for FastMKS model.

"""
function fastmks(;
                 bandwidth::Union{Float64, Missing} = missing,
                 base::Union{Float64, Missing} = missing,
                 degree::Union{Float64, Missing} = missing,
                 input_model::Union{Ptr{Nothing}, Missing} = missing,
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

  CLIRestoreSettings("FastMKS (Fast Max-Kernel Search)")

  # Process each input argument before calling mlpackMain().
  if !ismissing(bandwidth)
    CLISetParam("bandwidth", convert(Float64, bandwidth))
  end
  if !ismissing(base)
    CLISetParam("base", convert(Float64, base))
  end
  if !ismissing(degree)
    CLISetParam("degree", convert(Float64, degree))
  end
  if !ismissing(input_model)
    fastmks_internal.CLISetParamFastMKSModelPtr("input_model", convert(Ptr{Nothing}, input_model))
  end
  if !ismissing(k)
    CLISetParam("k", convert(Int, k))
  end
  if !ismissing(kernel)
    CLISetParam("kernel", convert(String, kernel))
  end
  if !ismissing(naive)
    CLISetParam("naive", convert(Bool, naive))
  end
  if !ismissing(offset)
    CLISetParam("offset", convert(Float64, offset))
  end
  if !ismissing(query)
    CLISetParamMat("query", query, points_are_rows)
  end
  if !ismissing(reference)
    CLISetParamMat("reference", reference, points_are_rows)
  end
  if !ismissing(scale)
    CLISetParam("scale", convert(Float64, scale))
  end
  if !ismissing(single)
    CLISetParam("single", convert(Bool, single))
  end
  if verbose !== nothing && verbose === true
    CLIEnableVerbose()
  else
    CLIDisableVerbose()
  end

  CLISetPassed("indices")
  CLISetPassed("kernels")
  CLISetPassed("output_model")
  # Call the program.
  fastmks_mlpackMain()

  return CLIGetParamUMat("indices", points_are_rows),
         CLIGetParamMat("kernels", points_are_rows),
         fastmks_internal.CLIGetParamFastMKSModelPtr("output_model")
end
