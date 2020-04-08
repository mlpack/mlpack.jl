export gmm_train

using mlpack._Internal.cli

import mlpack_jll
const gmm_trainLibrary = mlpack_jll.libmlpack_julia_gmm_train

# Call the C binding of the mlpack gmm_train binding.
function gmm_train_mlpackMain()
  success = ccall((:gmm_train, gmm_trainLibrary), Bool, ())
  if !success
    # Throw an exception---false means there was a C++ exception.
    throw(ErrorException("mlpack binding error; see output"))
  end
end

" Internal module to hold utility functions. "
module gmm_train_internal
  import ..gmm_trainLibrary

" Get the value of a model pointer parameter of type GMM."
function CLIGetParamGMMPtr(paramName::String)
  return ccall((:CLI_GetParamGMMPtr, gmm_trainLibrary), Ptr{Nothing}, (Cstring,), paramName)
end

" Set the value of a model pointer parameter of type GMM."
function CLISetParamGMMPtr(paramName::String, ptr::Ptr{Nothing})
  ccall((:CLI_SetParamGMMPtr, gmm_trainLibrary), Nothing, (Cstring, Ptr{Nothing}), paramName, ptr)
end

end # module

"""
    gmm_train(gaussians, input; [diagonal_covariance, input_model, kmeans_max_iterations, max_iterations, no_force_positive, noise, percentage, refined_start, samplings, seed, tolerance, trials, verbose])

This program takes a parametric estimate of a Gaussian mixture model (GMM) using
the EM algorithm to find the maximum likelihood estimate.  The model may be
saved and reused by other mlpack GMM tools.

The input data to train on must be specified with the `input` parameter, and the
number of Gaussians in the model must be specified with the `gaussians`
parameter.  Optionally, many trials with different random initializations may be
run, and the result with highest log-likelihood on the training data will be
taken.  The number of trials to run is specified with the `trials` parameter. 
By default, only one trial is run.

The tolerance for convergence and maximum number of iterations of the EM
algorithm are specified with the `tolerance` and `max_iterations` parameters,
respectively.  The GMM may be initialized for training with another model,
specified with the `input_model` parameter. Otherwise, the model is initialized
by running k-means on the data.  The k-means clustering initialization can be
controlled with the `kmeans_max_iterations`, `refined_start`, `samplings`, and
`percentage` parameters.  If `refined_start` is specified, then the
Bradley-Fayyad refined start initialization will be used.  This can often lead
to better clustering results.

The 'diagonal_covariance' flag will cause the learned covariances to be diagonal
matrices.  This significantly simplifies the model itself and causes training to
be faster, but restricts the ability to fit more complex GMMs.

If GMM training fails with an error indicating that a covariance matrix could
not be inverted, make sure that the `no_force_positive` parameter is not
specified.  Alternately, adding a small amount of Gaussian noise (using the
`noise` parameter) to the entire dataset may help prevent Gaussians with zero
variance in a particular dimension, which is usually the cause of non-invertible
covariance matrices.

The `no_force_positive` parameter, if set, will avoid the checks after each
iteration of the EM algorithm which ensure that the covariance matrices are
positive definite.  Specifying the flag can cause faster runtime, but may also
cause non-positive definite covariance matrices, which will cause the program to
crash.

As an example, to train a 6-Gaussian GMM on the data in `data` with a maximum of
100 iterations of EM and 3 trials, saving the trained GMM to `gmm`, the
following command can be used:

```julia
julia> using CSV
julia> data = CSV.read("data.csv")
julia> gmm = gmm_train(6, data; trials=3)
```

To re-train that GMM on another set of data `data2`, the following command may
be used: 

```julia
julia> using CSV
julia> data2 = CSV.read("data2.csv")
julia> new_gmm = gmm_train(6, data2; input_model=gmm)
```

# Arguments

 - `gaussians::Int`: Number of Gaussians in the GMM.
 - `input::Array{Float64, 2}`: The training data on which the model will
      be fit.
 - `diagonal_covariance::Bool`: Force the covariance of the Gaussians to
      be diagonal.  This can accelerate training time significantly.  Default
      value `false`.
      
 - `input_model::unknown_`: Initial input GMM model to start training
      with.
 - `kmeans_max_iterations::Int`: Maximum number of iterations for the
      k-means algorithm (used to initialize EM).  Default value `1000`.
      
 - `max_iterations::Int`: Maximum number of iterations of EM algorithm
      (passing 0 will run until convergence).  Default value `250`.
      
 - `no_force_positive::Bool`: Do not force the covariance matrices to be
      positive definite.  Default value `false`.
      
 - `noise::Float64`: Variance of zero-mean Gaussian noise to add to data. 
      Default value `0`.
      
 - `percentage::Float64`: If using --refined_start, specify the percentage
      of the dataset used for each sampling (should be between 0.0 and 1.0). 
      Default value `0.02`.
      
 - `refined_start::Bool`: During the initialization, use refined initial
      positions for k-means clustering (Bradley and Fayyad, 1998).  Default
      value `false`.
      
 - `samplings::Int`: If using --refined_start, specify the number of
      samplings used for initial points.  Default value `100`.
      
 - `seed::Int`: Random seed.  If 0, 'std::time(NULL)' is used.  Default
      value `0`.
      
 - `tolerance::Float64`: Tolerance for convergence of EM.  Default value
      `1e-10`.
      
 - `trials::Int`: Number of trials to perform in training GMM.  Default
      value `1`.
      
 - `verbose::Bool`: Display informational messages and the full list of
      parameters and timers at the end of execution.  Default value `false`.
      

# Return values

 - `output_model::unknown_`: Output for trained GMM model.

"""
function gmm_train(gaussians::Int,
                   input;
                   diagonal_covariance::Union{Bool, Missing} = missing,
                   input_model::Union{Ptr{Nothing}, Missing} = missing,
                   kmeans_max_iterations::Union{Int, Missing} = missing,
                   max_iterations::Union{Int, Missing} = missing,
                   no_force_positive::Union{Bool, Missing} = missing,
                   noise::Union{Float64, Missing} = missing,
                   percentage::Union{Float64, Missing} = missing,
                   refined_start::Union{Bool, Missing} = missing,
                   samplings::Union{Int, Missing} = missing,
                   seed::Union{Int, Missing} = missing,
                   tolerance::Union{Float64, Missing} = missing,
                   trials::Union{Int, Missing} = missing,
                   verbose::Union{Bool, Missing} = missing,
                   points_are_rows::Bool = true)
  # Force the symbols to load.
  ccall((:loadSymbols, gmm_trainLibrary), Nothing, ());

  CLIRestoreSettings("Gaussian Mixture Model (GMM) Training")

  # Process each input argument before calling mlpackMain().
  CLISetParam("gaussians", gaussians)
  CLISetParamMat("input", input, points_are_rows)
  if !ismissing(diagonal_covariance)
    CLISetParam("diagonal_covariance", convert(Bool, diagonal_covariance))
  end
  if !ismissing(input_model)
    gmm_train_internal.CLISetParamGMMPtr("input_model", convert(Ptr{Nothing}, input_model))
  end
  if !ismissing(kmeans_max_iterations)
    CLISetParam("kmeans_max_iterations", convert(Int, kmeans_max_iterations))
  end
  if !ismissing(max_iterations)
    CLISetParam("max_iterations", convert(Int, max_iterations))
  end
  if !ismissing(no_force_positive)
    CLISetParam("no_force_positive", convert(Bool, no_force_positive))
  end
  if !ismissing(noise)
    CLISetParam("noise", convert(Float64, noise))
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
  if !ismissing(tolerance)
    CLISetParam("tolerance", convert(Float64, tolerance))
  end
  if !ismissing(trials)
    CLISetParam("trials", convert(Int, trials))
  end
  if verbose !== nothing && verbose === true
    CLIEnableVerbose()
  else
    CLIDisableVerbose()
  end

  CLISetPassed("output_model")
  # Call the program.
  gmm_train_mlpackMain()

  return gmm_train_internal.CLIGetParamGMMPtr("output_model")
end
