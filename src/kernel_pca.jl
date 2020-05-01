export kernel_pca


using mlpack._Internal.cli

import mlpack_jll
const kernel_pcaLibrary = mlpack_jll.libmlpack_julia_kernel_pca

# Call the C binding of the mlpack kernel_pca binding.
function kernel_pca_mlpackMain()
  success = ccall((:kernel_pca, kernel_pcaLibrary), Bool, ())
  if !success
    # Throw an exception---false means there was a C++ exception.
    throw(ErrorException("mlpack binding error; see output"))
  end
end

" Internal module to hold utility functions. "
module kernel_pca_internal
  import ..kernel_pcaLibrary

end # module

"""
    kernel_pca(input, kernel; [bandwidth, center, degree, kernel_scale, new_dimensionality, nystroem_method, offset, sampling, verbose])

This program performs Kernel Principal Components Analysis (KPCA) on the
specified dataset with the specified kernel.  This will transform the data onto
the kernel principal components, and optionally reduce the dimensionality by
ignoring the kernel principal components with the smallest eigenvalues.

For the case where a linear kernel is used, this reduces to regular PCA.

For example, the following command will perform KPCA on the dataset `input`
using the Gaussian kernel, and saving the transformed data to `transformed`: 

```julia
julia> using CSV
julia> input = CSV.read("input.csv")
julia> transformed = kernel_pca(input, "gaussian")
```

The kernels that are supported are listed below:

 * 'linear': the standard linear dot product (same as normal PCA):
    K(x, y) = x^T y

 * 'gaussian': a Gaussian kernel; requires bandwidth:
    K(x, y) = exp(-(|| x - y || ^ 2) / (2 * (bandwidth ^ 2)))

 * 'polynomial': polynomial kernel; requires offset and degree:
    K(x, y) = (x^T y + offset) ^ degree

 * 'hyptan': hyperbolic tangent kernel; requires scale and offset:
    K(x, y) = tanh(scale * (x^T y) + offset)

 * 'laplacian': Laplacian kernel; requires bandwidth:
    K(x, y) = exp(-(|| x - y ||) / bandwidth)

 * 'epanechnikov': Epanechnikov kernel; requires bandwidth:
    K(x, y) = max(0, 1 - || x - y ||^2 / bandwidth^2)

 * 'cosine': cosine distance:
    K(x, y) = 1 - (x^T y) / (|| x || * || y ||)

The parameters for each of the kernels should be specified with the options
`bandwidth`, `kernel_scale`, `offset`, or `degree` (or a combination of those
parameters).

Optionally, the Nystroem method ("Using the Nystroem method to speed up kernel
machines", 2001) can be used to calculate the kernel matrix by specifying the
`nystroem_method` parameter. This approach works by using a subset of the data
as basis to reconstruct the kernel matrix; to specify the sampling scheme, the
`sampling` parameter is used.  The sampling scheme for the Nystroem method can
be chosen from the following list: 'kmeans', 'random', 'ordered'.

# Arguments

 - `input::Array{Float64, 2}`: Input dataset to perform KPCA on.
 - `kernel::String`: The kernel to use; see the above documentation for
      the list of usable kernels.
 - `bandwidth::Float64`: Bandwidth, for 'gaussian' and 'laplacian'
      kernels.  Default value `1`.
      
 - `center::Bool`: If set, the transformed data will be centered about the
      origin.  Default value `false`.
      
 - `degree::Float64`: Degree of polynomial, for 'polynomial' kernel. 
      Default value `1`.
      
 - `kernel_scale::Float64`: Scale, for 'hyptan' kernel.  Default value
      `1`.
      
 - `new_dimensionality::Int`: If not 0, reduce the dimensionality of the
      output dataset by ignoring the dimensions with the smallest eigenvalues. 
      Default value `0`.
      
 - `nystroem_method::Bool`: If set, the Nystroem method will be used. 
      Default value `false`.
      
 - `offset::Float64`: Offset, for 'hyptan' and 'polynomial' kernels. 
      Default value `0`.
      
 - `sampling::String`: Sampling scheme to use for the Nystroem method:
      'kmeans', 'random', 'ordered'  Default value `kmeans`.
      
 - `verbose::Bool`: Display informational messages and the full list of
      parameters and timers at the end of execution.  Default value `false`.
      

# Return values

 - `output::Array{Float64, 2}`: Matrix to save modified dataset to.

"""
function kernel_pca(input,
                    kernel::String;
                    bandwidth::Union{Float64, Missing} = missing,
                    center::Union{Bool, Missing} = missing,
                    degree::Union{Float64, Missing} = missing,
                    kernel_scale::Union{Float64, Missing} = missing,
                    new_dimensionality::Union{Int, Missing} = missing,
                    nystroem_method::Union{Bool, Missing} = missing,
                    offset::Union{Float64, Missing} = missing,
                    sampling::Union{String, Missing} = missing,
                    verbose::Union{Bool, Missing} = missing,
                    points_are_rows::Bool = true)
  # Force the symbols to load.
  ccall((:loadSymbols, kernel_pcaLibrary), Nothing, ());

  CLIRestoreSettings("Kernel Principal Components Analysis")

  # Process each input argument before calling mlpackMain().
  CLISetParamMat("input", input, points_are_rows)
  CLISetParam("kernel", kernel)
  if !ismissing(bandwidth)
    CLISetParam("bandwidth", convert(Float64, bandwidth))
  end
  if !ismissing(center)
    CLISetParam("center", convert(Bool, center))
  end
  if !ismissing(degree)
    CLISetParam("degree", convert(Float64, degree))
  end
  if !ismissing(kernel_scale)
    CLISetParam("kernel_scale", convert(Float64, kernel_scale))
  end
  if !ismissing(new_dimensionality)
    CLISetParam("new_dimensionality", convert(Int, new_dimensionality))
  end
  if !ismissing(nystroem_method)
    CLISetParam("nystroem_method", convert(Bool, nystroem_method))
  end
  if !ismissing(offset)
    CLISetParam("offset", convert(Float64, offset))
  end
  if !ismissing(sampling)
    CLISetParam("sampling", convert(String, sampling))
  end
  if verbose !== nothing && verbose === true
    CLIEnableVerbose()
  else
    CLIDisableVerbose()
  end

  CLISetPassed("output")
  # Call the program.
  kernel_pca_mlpackMain()

  return CLIGetParamMat("output", points_are_rows)
end
