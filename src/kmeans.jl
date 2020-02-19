export kmeans

using mlpack.util.cli

const kmeansLibrary = joinpath(@__DIR__, "libmlpack_julia_kmeans.so")

# Call the C binding of the mlpack kmeans binding.
function kmeans_mlpackMain()
  success = ccall((:kmeans, kmeansLibrary), Bool, ())
  if !success
    # Throw an exception---false means there was a C++ exception.
    throw(ErrorException("mlpack binding error; see output"))
  end
end

" Internal module to hold utility functions. "
module kmeans_internal
  import ..kmeansLibrary

end # module

"""
    kmeans(clusters, input; [algorithm, allow_empty_clusters, in_place, initial_centroids, kill_empty_clusters, labels_only, max_iterations, percentage, refined_start, samplings, seed, verbose])

This program performs K-Means clustering on the given dataset.  It can return
the learned cluster assignments, and the centroids of the clusters.  Empty
clusters are not allowed by default; when a cluster becomes empty, the point
furthest from the centroid of the cluster with maximum variance is taken to fill
that cluster.

Optionally, the Bradley and Fayyad approach ("Refining initial points for
k-means clustering", 1998) can be used to select initial points by specifying
the `refined_start` parameter.  This approach works by taking random samplings
of the dataset; to specify the number of samplings, the `samplings` parameter is
used, and to specify the percentage of the dataset to be used in each sample,
the `percentage` parameter is used (it should be a value between 0.0 and 1.0).

There are several options available for the algorithm used for each Lloyd
iteration, specified with the `algorithm`  option.  The standard O(kN) approach
can be used ('naive').  Other options include the Pelleg-Moore tree-based
algorithm ('pelleg-moore'), Elkan's triangle-inequality based algorithm
('elkan'), Hamerly's modification to Elkan's algorithm ('hamerly'), the
dual-tree k-means algorithm ('dualtree'), and the dual-tree k-means algorithm
using the cover tree ('dualtree-covertree').

The behavior for when an empty cluster is encountered can be modified with the
`allow_empty_clusters` option.  When this option is specified and there is a
cluster owning no points at the end of an iteration, that cluster's centroid
will simply remain in its position from the previous iteration. If the
`kill_empty_clusters` option is specified, then when a cluster owns no points at
the end of an iteration, the cluster centroid is simply filled with DBL_MAX,
killing it and effectively reducing k for the rest of the computation.  Note
that the default option when neither empty cluster option is specified can be
time-consuming to calculate; therefore, specifying either of these parameters
will often accelerate runtime.

Initial clustering assignments may be specified using the `initial_centroids`
parameter, and the maximum number of iterations may be specified with the
`max_iterations` parameter.

As an example, to use Hamerly's algorithm to perform k-means clustering with
k=10 on the dataset `data`, saving the centroids to `centroids` and the
assignments for each point to `assignments`, the following command could be
used:

julia> using CSV
julia> data = CSV.read("data.csv")
julia> centroids, assignments = kmeans(10, data)

To run k-means on that same dataset with initial centroids specified in
`initial` with a maximum of 500 iterations, storing the output centroids in
`final` the following command may be used:

julia> using CSV
julia> data = CSV.read("data.csv")
julia> initial = CSV.read("initial.csv")
julia> final, _ = kmeans(10, data; initial_centroids=initial,
            max_iterations=500)

# Arguments

 - `clusters::Int`: Number of clusters to find (0 autodetects from initial
      centroids).
 - `input::Array{Float64, 2}`: Input dataset to perform clustering on.
 - `algorithm::String`: Algorithm to use for the Lloyd iteration ('naive',
      'pelleg-moore', 'elkan', 'hamerly', 'dualtree', or 'dualtree-covertree'). 
      Default value `naive`.
      
 - `allow_empty_clusters::Bool`: Allow empty clusters to be persist. 
      Default value `false`.
      
 - `in_place::Bool`: If specified, a column containing the learned cluster
      assignments will be added to the input dataset file.  In this case,
      --output_file is overridden. (Do not use in Python.)  Default value
      `false`.
      
 - `initial_centroids::Array{Float64, 2}`: Start with the specified
      initial centroids.
 - `kill_empty_clusters::Bool`: Remove empty clusters when they occur. 
      Default value `false`.
      
 - `labels_only::Bool`: Only output labels into output file.  Default
      value `false`.
      
 - `max_iterations::Int`: Maximum number of iterations before k-means
      terminates.  Default value `1000`.
      
 - `percentage::Float64`: Percentage of dataset to use for each refined
      start sampling (use when --refined_start is specified).  Default value
      `0.02`.
      
 - `refined_start::Bool`: Use the refined initial point strategy by
      Bradley and Fayyad to choose initial points.  Default value `false`.
      
 - `samplings::Int`: Number of samplings to perform for refined start (use
      when --refined_start is specified).  Default value `100`.
      
 - `seed::Int`: Random seed.  If 0, 'std::time(NULL)' is used.  Default
      value `0`.
      
 - `verbose::Bool`: Display informational messages and the full list of
      parameters and timers at the end of execution.  Default value `false`.
      

# Return values

 - `centroid::Array{Float64, 2}`: If specified, the centroids of each
      cluster will  be written to the given file.
 - `output::Array{Float64, 2}`: Matrix to store output labels or labeled
      data to.

"""
function kmeans(clusters::Int,
                input;
                algorithm::Union{String, Missing} = missing,
                allow_empty_clusters::Union{Bool, Missing} = missing,
                in_place::Union{Bool, Missing} = missing,
                initial_centroids = missing,
                kill_empty_clusters::Union{Bool, Missing} = missing,
                labels_only::Union{Bool, Missing} = missing,
                max_iterations::Union{Int, Missing} = missing,
                percentage::Union{Float64, Missing} = missing,
                refined_start::Union{Bool, Missing} = missing,
                samplings::Union{Int, Missing} = missing,
                seed::Union{Int, Missing} = missing,
                verbose::Union{Bool, Missing} = missing,
                points_are_rows::Bool = true)
  # Force the symbols to load.
  ccall((:loadSymbols, kmeansLibrary), Nothing, ());

  CLIRestoreSettings("K-Means Clustering")

  # Process each input argument before calling mlpackMain().
  CLISetParam("clusters", clusters)
  CLISetParamMat("input", input, points_are_rows)
  if !ismissing(algorithm)
    CLISetParam("algorithm", convert(String, algorithm))
  end
  if !ismissing(allow_empty_clusters)
    CLISetParam("allow_empty_clusters", convert(Bool, allow_empty_clusters))
  end
  if !ismissing(in_place)
    CLISetParam("in_place", convert(Bool, in_place))
  end
  if !ismissing(initial_centroids)
    CLISetParamMat("initial_centroids", initial_centroids, points_are_rows)
  end
  if !ismissing(kill_empty_clusters)
    CLISetParam("kill_empty_clusters", convert(Bool, kill_empty_clusters))
  end
  if !ismissing(labels_only)
    CLISetParam("labels_only", convert(Bool, labels_only))
  end
  if !ismissing(max_iterations)
    CLISetParam("max_iterations", convert(Int, max_iterations))
  end
  if !ismissing(percentage)
    CLISetParam("percentage", convert(Float64, percentage))
  end
  if !ismissing(refined_start)
    CLISetParam("refined_start", convert(Bool, refined_start))
  end
  if !ismissing(samplings)
    CLISetParam("samplings", convert(Int, samplings))
  end
  if !ismissing(seed)
    CLISetParam("seed", convert(Int, seed))
  end
  if verbose !== nothing && verbose === true
    CLIEnableVerbose()
  else
    CLIDisableVerbose()
  end

  CLISetPassed("centroid")
  CLISetPassed("output")
  # Call the program.
  kmeans_mlpackMain()

  return CLIGetParamMat("centroid", points_are_rows),
         CLIGetParamMat("output", points_are_rows)
end
