export mean_shift

using mlpack._Internal.cli

import mlpack_jll
const mean_shiftLibrary = mlpack_jll.libmlpack_julia_mean_shift

# Call the C binding of the mlpack mean_shift binding.
function mean_shift_mlpackMain()
  success = ccall((:mean_shift, mean_shiftLibrary), Bool, ())
  if !success
    # Throw an exception---false means there was a C++ exception.
    throw(ErrorException("mlpack binding error; see output"))
  end
end

" Internal module to hold utility functions. "
module mean_shift_internal
  import ..mean_shiftLibrary

end # module

"""
    mean_shift(input; [force_convergence, in_place, labels_only, max_iterations, radius, verbose])

This program performs mean shift clustering on the given dataset, storing the
learned cluster assignments either as a column of labels in the input dataset or
separately.

The input dataset should be specified with the `input` parameter, and the radius
used for search can be specified with the `radius` parameter.  The maximum
number of iterations before algorithm termination is controlled with the
`max_iterations` parameter.

The output labels may be saved with the `output` output parameter and the
centroids of each cluster may be saved with the `centroid` output parameter.

For example, to run mean shift clustering on the dataset `data` and store the
centroids to `centroids`, the following command may be used: 

```julia
julia> using CSV
julia> data = CSV.read("data.csv")
julia> centroids, _ = mean_shift(data)
```

# Arguments

 - `input::Array{Float64, 2}`: Input dataset to perform clustering on.
 - `force_convergence::Bool`: If specified, the mean shift algorithm will
      continue running regardless of max_iterations until the clusters converge.
       Default value `false`.
      
 - `in_place::Bool`: If specified, a column containing the learned cluster
      assignments will be added to the input dataset file.  In this case,
      --output_file is overridden.  (Do not use with Python.)  Default value
      `false`.
      
 - `labels_only::Bool`: If specified, only the output labels will be
      written to the file specified by --output_file.  Default value `false`.
      
 - `max_iterations::Int`: Maximum number of iterations before mean shift
      terminates.  Default value `1000`.
      
 - `radius::Float64`: If the distance between two centroids is less than
      the given radius, one will be removed.  A radius of 0 or less means an
      estimate will be calculated and used for the radius.  Default value `0`.
      
 - `verbose::Bool`: Display informational messages and the full list of
      parameters and timers at the end of execution.  Default value `false`.
      

# Return values

 - `centroid::Array{Float64, 2}`: If specified, the centroids of each
      cluster will be written to the given matrix.
 - `output::Array{Float64, 2}`: Matrix to write output labels or labeled
      data to.

"""
function mean_shift(input;
                    force_convergence::Union{Bool, Missing} = missing,
                    in_place::Union{Bool, Missing} = missing,
                    labels_only::Union{Bool, Missing} = missing,
                    max_iterations::Union{Int, Missing} = missing,
                    radius::Union{Float64, Missing} = missing,
                    verbose::Union{Bool, Missing} = missing,
                    points_are_rows::Bool = true)
  # Force the symbols to load.
  ccall((:loadSymbols, mean_shiftLibrary), Nothing, ());

  CLIRestoreSettings("Mean Shift Clustering")

  # Process each input argument before calling mlpackMain().
  CLISetParamMat("input", input, points_are_rows)
  if !ismissing(force_convergence)
    CLISetParam("force_convergence", convert(Bool, force_convergence))
  end
  if !ismissing(in_place)
    CLISetParam("in_place", convert(Bool, in_place))
  end
  if !ismissing(labels_only)
    CLISetParam("labels_only", convert(Bool, labels_only))
  end
  if !ismissing(max_iterations)
    CLISetParam("max_iterations", convert(Int, max_iterations))
  end
  if !ismissing(radius)
    CLISetParam("radius", convert(Float64, radius))
  end
  if verbose !== nothing && verbose === true
    CLIEnableVerbose()
  else
    CLIDisableVerbose()
  end

  CLISetPassed("centroid")
  CLISetPassed("output")
  # Call the program.
  mean_shift_mlpackMain()

  return CLIGetParamMat("centroid", points_are_rows),
         CLIGetParamMat("output", points_are_rows)
end
