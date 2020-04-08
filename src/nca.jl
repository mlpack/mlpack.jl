export nca

using mlpack._Internal.cli

import mlpack_jll
const ncaLibrary = mlpack_jll.libmlpack_julia_nca

# Call the C binding of the mlpack nca binding.
function nca_mlpackMain()
  success = ccall((:nca, ncaLibrary), Bool, ())
  if !success
    # Throw an exception---false means there was a C++ exception.
    throw(ErrorException("mlpack binding error; see output"))
  end
end

" Internal module to hold utility functions. "
module nca_internal
  import ..ncaLibrary

end # module

"""
    nca(input; [armijo_constant, batch_size, labels, linear_scan, max_iterations, max_line_search_trials, max_step, min_step, normalize, num_basis, optimizer, seed, step_size, tolerance, verbose, wolfe])

This program implements Neighborhood Components Analysis, both a linear
dimensionality reduction technique and a distance learning technique.  The
method seeks to improve k-nearest-neighbor classification on a dataset by
scaling the dimensions.  The method is nonparametric, and does not require a
value of k.  It works by using stochastic ("soft") neighbor assignments and
using optimization techniques over the gradient of the accuracy of the neighbor
assignments.

To work, this algorithm needs labeled data.  It can be given as the last row of
the input dataset (specified with `input`), or alternatively as a separate
matrix (specified with `labels`).

This implementation of NCA uses stochastic gradient descent, mini-batch
stochastic gradient descent, or the L_BFGS optimizer.  These optimizers do not
guarantee global convergence for a nonconvex objective function (NCA's objective
function is nonconvex), so the final results could depend on the random seed or
other optimizer parameters.

Stochastic gradient descent, specified by the value 'sgd' for the parameter
`optimizer`, depends primarily on three parameters: the step size (specified
with `step_size`), the batch size (specified with `batch_size`), and the maximum
number of iterations (specified with `max_iterations`).  In addition, a
normalized starting point can be used by specifying the `normalize` parameter,
which is necessary if many warnings of the form 'Denominator of p_i is 0!' are
given.  Tuning the step size can be a tedious affair.  In general, the step size
is too large if the objective is not mostly uniformly decreasing, or if
zero-valued denominator warnings are being issued.  The step size is too small
if the objective is changing very slowly.  Setting the termination condition can
be done easily once a good step size parameter is found; either increase the
maximum iterations to a large number and allow SGD to find a minimum, or set the
maximum iterations to 0 (allowing infinite iterations) and set the tolerance
(specified by `tolerance`) to define the maximum allowed difference between
objectives for SGD to terminate.  Be careful---setting the tolerance instead of
the maximum iterations can take a very long time and may actually never converge
due to the properties of the SGD optimizer. Note that a single iteration of SGD
refers to a single point, so to take a single pass over the dataset, set the
value of the `max_iterations` parameter equal to the number of points in the
dataset.

The L-BFGS optimizer, specified by the value 'lbfgs' for the parameter
`optimizer`, uses a back-tracking line search algorithm to minimize a function. 
The following parameters are used by L-BFGS: `num_basis` (specifies the number
of memory points used by L-BFGS), `max_iterations`, `armijo_constant`, `wolfe`,
`tolerance` (the optimization is terminated when the gradient norm is below this
value), `max_line_search_trials`, `min_step`, and `max_step` (which both refer
to the line search routine).  For more details on the L-BFGS optimizer, consult
either the mlpack L-BFGS documentation (in lbfgs.hpp) or the vast set of
published literature on L-BFGS.

By default, the SGD optimizer is used.

# Arguments

 - `input::Array{Float64, 2}`: Input dataset to run NCA on.
 - `armijo_constant::Float64`: Armijo constant for L-BFGS.  Default value
      `0.0001`.
      
 - `batch_size::Int`: Batch size for mini-batch SGD.  Default value `50`.
      
 - `labels::Array{Int, 1}`: Labels for input dataset.
 - `linear_scan::Bool`: Don't shuffle the order in which data points are
      visited for SGD or mini-batch SGD.  Default value `false`.
      
 - `max_iterations::Int`: Maximum number of iterations for SGD or L-BFGS
      (0 indicates no limit).  Default value `500000`.
      
 - `max_line_search_trials::Int`: Maximum number of line search trials for
      L-BFGS.  Default value `50`.
      
 - `max_step::Float64`: Maximum step of line search for L-BFGS.  Default
      value `1e+20`.
      
 - `min_step::Float64`: Minimum step of line search for L-BFGS.  Default
      value `1e-20`.
      
 - `normalize::Bool`: Use a normalized starting point for optimization.
      This is useful for when points are far apart, or when SGD is returning
      NaN.  Default value `false`.
      
 - `num_basis::Int`: Number of memory points to be stored for L-BFGS. 
      Default value `5`.
      
 - `optimizer::String`: Optimizer to use; 'sgd' or 'lbfgs'.  Default value
      `sgd`.
      
 - `seed::Int`: Random seed.  If 0, 'std::time(NULL)' is used.  Default
      value `0`.
      
 - `step_size::Float64`: Step size for stochastic gradient descent
      (alpha).  Default value `0.01`.
      
 - `tolerance::Float64`: Maximum tolerance for termination of SGD or
      L-BFGS.  Default value `1e-07`.
      
 - `verbose::Bool`: Display informational messages and the full list of
      parameters and timers at the end of execution.  Default value `false`.
      
 - `wolfe::Float64`: Wolfe condition parameter for L-BFGS.  Default value
      `0.9`.
      

# Return values

 - `output::Array{Float64, 2}`: Output matrix for learned distance
      matrix.

"""
function nca(input;
             armijo_constant::Union{Float64, Missing} = missing,
             batch_size::Union{Int, Missing} = missing,
             labels = missing,
             linear_scan::Union{Bool, Missing} = missing,
             max_iterations::Union{Int, Missing} = missing,
             max_line_search_trials::Union{Int, Missing} = missing,
             max_step::Union{Float64, Missing} = missing,
             min_step::Union{Float64, Missing} = missing,
             normalize::Union{Bool, Missing} = missing,
             num_basis::Union{Int, Missing} = missing,
             optimizer::Union{String, Missing} = missing,
             seed::Union{Int, Missing} = missing,
             step_size::Union{Float64, Missing} = missing,
             tolerance::Union{Float64, Missing} = missing,
             verbose::Union{Bool, Missing} = missing,
             wolfe::Union{Float64, Missing} = missing,
             points_are_rows::Bool = true)
  # Force the symbols to load.
  ccall((:loadSymbols, ncaLibrary), Nothing, ());

  CLIRestoreSettings("Neighborhood Components Analysis (NCA)")

  # Process each input argument before calling mlpackMain().
  CLISetParamMat("input", input, points_are_rows)
  if !ismissing(armijo_constant)
    CLISetParam("armijo_constant", convert(Float64, armijo_constant))
  end
  if !ismissing(batch_size)
    CLISetParam("batch_size", convert(Int, batch_size))
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
  if !ismissing(max_line_search_trials)
    CLISetParam("max_line_search_trials", convert(Int, max_line_search_trials))
  end
  if !ismissing(max_step)
    CLISetParam("max_step", convert(Float64, max_step))
  end
  if !ismissing(min_step)
    CLISetParam("min_step", convert(Float64, min_step))
  end
  if !ismissing(normalize)
    CLISetParam("normalize", convert(Bool, normalize))
  end
  if !ismissing(num_basis)
    CLISetParam("num_basis", convert(Int, num_basis))
  end
  if !ismissing(optimizer)
    CLISetParam("optimizer", convert(String, optimizer))
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
  if !ismissing(wolfe)
    CLISetParam("wolfe", convert(Float64, wolfe))
  end
  if verbose !== nothing && verbose === true
    CLIEnableVerbose()
  else
    CLIDisableVerbose()
  end

  CLISetPassed("output")
  # Call the program.
  nca_mlpackMain()

  return CLIGetParamMat("output", points_are_rows)
end
