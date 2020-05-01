export lmnn


using mlpack._Internal.cli

import mlpack_jll
const lmnnLibrary = mlpack_jll.libmlpack_julia_lmnn

# Call the C binding of the mlpack lmnn binding.
function lmnn_mlpackMain()
  success = ccall((:lmnn, lmnnLibrary), Bool, ())
  if !success
    # Throw an exception---false means there was a C++ exception.
    throw(ErrorException("mlpack binding error; see output"))
  end
end

" Internal module to hold utility functions. "
module lmnn_internal
  import ..lmnnLibrary

end # module

"""
    lmnn(input; [batch_size, center, distance, k, labels, linear_scan, max_iterations, normalize, optimizer, passes, print_accuracy, range, rank, regularization, seed, step_size, tolerance, verbose])

This program implements Large Margin Nearest Neighbors, a distance learning
technique.  The method seeks to improve k-nearest-neighbor classification on a
dataset.  The method employes the strategy of reducing distance between similar
labeled data points (a.k.a target neighbors) and increasing distance between
differently labeled points (a.k.a impostors) using standard optimization
techniques over the gradient of the distance between data points.

To work, this algorithm needs labeled data.  It can be given as the last row of
the input dataset (specified with `input`), or alternatively as a separate
matrix (specified with `labels`).  Additionally, a starting point for
optimization (specified with `distance`can be given, having (r x d)
dimensionality.  Here r should satisfy 1 <= r <= d, Consequently a Low-Rank
matrix will be optimized. Alternatively, Low-Rank distance can be learned by
specifying the `rank`parameter (A Low-Rank matrix with uniformly distributed
values will be used as initial learning point). 

The program also requires number of targets neighbors to work with ( specified
with `k`), A regularization parameter can also be passed, It acts as a trade of
between the pulling and pushing terms (specified with `regularization`), In
addition, this implementation of LMNN includes a parameter to decide the
interval after which impostors must be re-calculated (specified with `range`).

Output can either be the learned distance matrix (specified with `output`), or
the transformed dataset  (specified with `transformed_data`), or both.
Additionally mean-centered dataset (specified with `centered_data`) can be
accessed given mean-centering (specified with `center`) is performed on the
dataset. Accuracy on initial dataset and final transformed dataset can be
printed by specifying the `print_accuracy`parameter. 

This implementation of LMNN uses AdaGrad, BigBatch_SGD, stochastic gradient
descent, mini-batch stochastic gradient descent, or the L_BFGS optimizer. 

AdaGrad, specified by the value 'adagrad' for the parameter `optimizer`, uses
maximum of past squared gradients. It primarily on six parameters: the step size
(specified with `step_size`), the batch size (specified with `batch_size`), the
maximum number of passes (specified with `passes`). Inaddition, a normalized
starting point can be used by specifying the `normalize` parameter. 

BigBatch_SGD, specified by the value 'bbsgd' for the parameter `optimizer`,
depends primarily on four parameters: the step size (specified with
`step_size`), the batch size (specified with `batch_size`), the maximum number
of passes (specified with `passes`).  In addition, a normalized starting point
can be used by specifying the `normalize` parameter. 

Stochastic gradient descent, specified by the value 'sgd' for the parameter
`optimizer`, depends primarily on three parameters: the step size (specified
with `step_size`), the batch size (specified with `batch_size`), and the maximum
number of passes (specified with `passes`).  In addition, a normalized starting
point can be used by specifying the `normalize` parameter. Furthermore,
mean-centering can be performed on the dataset by specifying the
`center`parameter. 

The L-BFGS optimizer, specified by the value 'lbfgs' for the parameter
`optimizer`, uses a back-tracking line search algorithm to minimize a function. 
The following parameters are used by L-BFGS: `max_iterations`, `tolerance`(the
optimization is terminated when the gradient norm is below this value).  For
more details on the L-BFGS optimizer, consult either the mlpack L-BFGS
documentation (in lbfgs.hpp) or the vast set of published literature on L-BFGS. 
In addition, a normalized starting point can be used by specifying the
`normalize` parameter.

By default, the AMSGrad optimizer is used.

Example - Let's say we want to learn distance on iris dataset with number of
targets as 3 using BigBatch_SGD optimizer. A simple call for the same will look
like: 

```julia
julia> using CSV
julia> iris = CSV.read("iris.csv")
julia> iris_labels = CSV.read("iris_labels.csv"; type=Int)
julia> _, output, _ = mlpack_lmnn(iris; k=3, labels=iris_labels,
            optimizer="bbsgd")
```

An another program call making use of range & regularization parameter with
dataset having labels as last column can be made as: 

```julia
julia> using CSV
julia> letter_recognition = CSV.read("letter_recognition.csv")
julia> _, output, _ = mlpack_lmnn(letter_recognition; k=5, range=10,
            regularization=0.4)
```

# Arguments

 - `input::Array{Float64, 2}`: Input dataset to run LMNN on.
 - `batch_size::Int`: Batch size for mini-batch SGD.  Default value `50`.
      
 - `center::Bool`: Perform mean-centering on the dataset. It is useful
      when the centroid of the data is far from the origin.  Default value
      `false`.
      
 - `distance::Array{Float64, 2}`: Initial distance matrix to be used as
      starting point
 - `k::Int`: Number of target neighbors to use for each datapoint. 
      Default value `1`.
      
 - `labels::Array{Int, 1}`: Labels for input dataset.
 - `linear_scan::Bool`: Don't shuffle the order in which data points are
      visited for SGD or mini-batch SGD.  Default value `false`.
      
 - `max_iterations::Int`: Maximum number of iterations for L-BFGS (0
      indicates no limit).  Default value `100000`.
      
 - `normalize::Bool`: Use a normalized starting point for optimization.
      Itis useful for when points are far apart, or when SGD is returning NaN. 
      Default value `false`.
      
 - `optimizer::String`: Optimizer to use; 'amsgrad', 'bbsgd', 'sgd', or
      'lbfgs'.  Default value `amsgrad`.
      
 - `passes::Int`: Maximum number of full passes over dataset for AMSGrad,
      BB_SGD and SGD.  Default value `50`.
      
 - `print_accuracy::Bool`: Print accuracies on initial and transformed
      dataset  Default value `false`.
      
 - `range::Int`: Number of iterations after which impostors needs to be
      recalculated  Default value `1`.
      
 - `rank::Int`: Rank of distance matrix to be optimized.   Default value
      `0`.
      
 - `regularization::Float64`: Regularization for LMNN objective function  
      Default value `0.5`.
      
 - `seed::Int`: Random seed.  If 0, 'std::time(NULL)' is used.  Default
      value `0`.
      
 - `step_size::Float64`: Step size for AMSGrad, BB_SGD and SGD (alpha). 
      Default value `0.01`.
      
 - `tolerance::Float64`: Maximum tolerance for termination of AMSGrad,
      BB_SGD, SGD or L-BFGS.  Default value `1e-07`.
      
 - `verbose::Bool`: Display informational messages and the full list of
      parameters and timers at the end of execution.  Default value `false`.
      

# Return values

 - `centered_data::Array{Float64, 2}`: Output matrix for mean-centered
      dataset.
 - `output::Array{Float64, 2}`: Output matrix for learned distance
      matrix.
 - `transformed_data::Array{Float64, 2}`: Output matrix for transformed
      dataset.

"""
function lmnn(input;
              batch_size::Union{Int, Missing} = missing,
              center::Union{Bool, Missing} = missing,
              distance = missing,
              k::Union{Int, Missing} = missing,
              labels = missing,
              linear_scan::Union{Bool, Missing} = missing,
              max_iterations::Union{Int, Missing} = missing,
              normalize::Union{Bool, Missing} = missing,
              optimizer::Union{String, Missing} = missing,
              passes::Union{Int, Missing} = missing,
              print_accuracy::Union{Bool, Missing} = missing,
              range::Union{Int, Missing} = missing,
              rank::Union{Int, Missing} = missing,
              regularization::Union{Float64, Missing} = missing,
              seed::Union{Int, Missing} = missing,
              step_size::Union{Float64, Missing} = missing,
              tolerance::Union{Float64, Missing} = missing,
              verbose::Union{Bool, Missing} = missing,
              points_are_rows::Bool = true)
  # Force the symbols to load.
  ccall((:loadSymbols, lmnnLibrary), Nothing, ());

  CLIRestoreSettings("Large Margin Nearest Neighbors (LMNN)")

  # Process each input argument before calling mlpackMain().
  CLISetParamMat("input", input, points_are_rows)
  if !ismissing(batch_size)
    CLISetParam("batch_size", convert(Int, batch_size))
  end
  if !ismissing(center)
    CLISetParam("center", convert(Bool, center))
  end
  if !ismissing(distance)
    CLISetParamMat("distance", distance, points_are_rows)
  end
  if !ismissing(k)
    CLISetParam("k", convert(Int, k))
  end
  if !ismissing(labels)
    CLISetParamURow("labels", labels)
  end
  if !ismissing(linear_scan)
    CLISetParam("linear_scan", convert(Bool, linear_scan))
  end
  if !ismissing(max_iterations)
    CLISetParam("max_iterations", convert(Int, max_iterations))
  end
  if !ismissing(normalize)
    CLISetParam("normalize", convert(Bool, normalize))
  end
  if !ismissing(optimizer)
    CLISetParam("optimizer", convert(String, optimizer))
  end
  if !ismissing(passes)
    CLISetParam("passes", convert(Int, passes))
  end
  if !ismissing(print_accuracy)
    CLISetParam("print_accuracy", convert(Bool, print_accuracy))
  end
  if !ismissing(range)
    CLISetParam("range", convert(Int, range))
  end
  if !ismissing(rank)
    CLISetParam("rank", convert(Int, rank))
  end
  if !ismissing(regularization)
    CLISetParam("regularization", convert(Float64, regularization))
  end
  if !ismissing(seed)
    CLISetParam("seed", convert(Int, seed))
  end
  if !ismissing(step_size)
    CLISetParam("step_size", convert(Float64, step_size))
  end
  if !ismissing(tolerance)
    CLISetParam("tolerance", convert(Float64, tolerance))
  end
  if verbose !== nothing && verbose === true
    CLIEnableVerbose()
  else
    CLIDisableVerbose()
  end

  CLISetPassed("centered_data")
  CLISetPassed("output")
  CLISetPassed("transformed_data")
  # Call the program.
  lmnn_mlpackMain()

  return CLIGetParamMat("centered_data", points_are_rows),
         CLIGetParamMat("output", points_are_rows),
         CLIGetParamMat("transformed_data", points_are_rows)
end
