export adaboost

using mlpack._Internal.cli

import mlpack_jll
const adaboostLibrary = mlpack_jll.libmlpack_julia_adaboost

# Call the C binding of the mlpack adaboost binding.
function adaboost_mlpackMain()
  success = ccall((:adaboost, adaboostLibrary), Bool, ())
  if !success
    # Throw an exception---false means there was a C++ exception.
    throw(ErrorException("mlpack binding error; see output"))
  end
end

" Internal module to hold utility functions. "
module adaboost_internal
  import ..adaboostLibrary

" Get the value of a model pointer parameter of type AdaBoostModel."
function CLIGetParamAdaBoostModelPtr(paramName::String)
  return ccall((:CLI_GetParamAdaBoostModelPtr, adaboostLibrary), Ptr{Nothing}, (Cstring,), paramName)
end

" Set the value of a model pointer parameter of type AdaBoostModel."
function CLISetParamAdaBoostModelPtr(paramName::String, ptr::Ptr{Nothing})
  ccall((:CLI_SetParamAdaBoostModelPtr, adaboostLibrary), Nothing, (Cstring, Ptr{Nothing}), paramName, ptr)
end

end # module

"""
    adaboost(; [input_model, iterations, labels, test, tolerance, training, verbose, weak_learner])

This program implements the AdaBoost (or Adaptive Boosting) algorithm. The
variant of AdaBoost implemented here is AdaBoost.MH. It uses a weak learner,
either decision stumps or perceptrons, and over many iterations, creates a
strong learner that is a weighted ensemble of weak learners. It runs these
iterations until a tolerance value is crossed for change in the value of the
weighted training error.

For more information about the algorithm, see the paper "Improved Boosting
Algorithms Using Confidence-Rated Predictions", by R.E. Schapire and Y. Singer.

This program allows training of an AdaBoost model, and then application of that
model to a test dataset.  To train a model, a dataset must be passed with the
`training` option.  Labels can be given with the `labels` option; if no labels
are specified, the labels will be assumed to be the last column of the input
dataset.  Alternately, an AdaBoost model may be loaded with the `input_model`
option.

Once a model is trained or loaded, it may be used to provide class predictions
for a given test dataset.  A test dataset may be specified with the `test`
parameter.  The predicted classes for each point in the test dataset are output
to the `predictions` output parameter.  The AdaBoost model itself is output to
the `output_model` output parameter.

Note: the following parameter is deprecated and will be removed in mlpack 4.0.0:
`output`.
Use `predictions` instead of `output`.

For example, to run AdaBoost on an input dataset `data` with perceptrons as the
weak learner type, storing the trained model in `model`, one could use the
following command: 

```julia
julia> using CSV
julia> data = CSV.read("data.csv")
julia> _, model, _, _ = adaboost(training=data,
            weak_learner="perceptron")
```

Similarly, an already-trained model in `model` can be used to provide class
predictions from test data `test_data` and store the output in `predictions`
with the following command: 

```julia
julia> using CSV
julia> test_data = CSV.read("test_data.csv")
julia> _, _, predictions, _ = adaboost(input_model=model,
            test=test_data)
```

# Arguments

 - `input_model::unknown_`: Input AdaBoost model.
 - `iterations::Int`: The maximum number of boosting iterations to be run
      (0 will run until convergence.)  Default value `1000`.
      
 - `labels::Array{Int, 1}`: Labels for the training set.
 - `test::Array{Float64, 2}`: Test dataset.
 - `tolerance::Float64`: The tolerance for change in values of the
      weighted error during training.  Default value `1e-10`.
      
 - `training::Array{Float64, 2}`: Dataset for training AdaBoost.
 - `verbose::Bool`: Display informational messages and the full list of
      parameters and timers at the end of execution.  Default value `false`.
      
 - `weak_learner::String`: The type of weak learner to use:
      'decision_stump', or 'perceptron'.  Default value `decision_stump`.
      

# Return values

 - `output::Array{Int, 1}`: Predicted labels for the test set.
 - `output_model::unknown_`: Output trained AdaBoost model.
 - `predictions::Array{Int, 1}`: Predicted labels for the test set.
 - `probabilities::Array{Float64, 2}`: Predicted class probabilities for
      each point in the test set.

"""
function adaboost(;
                  input_model::Union{Ptr{Nothing}, Missing} = missing,
                  iterations::Union{Int, Missing} = missing,
                  labels = missing,
                  test = missing,
                  tolerance::Union{Float64, Missing} = missing,
                  training = missing,
                  verbose::Union{Bool, Missing} = missing,
                  weak_learner::Union{String, Missing} = missing,
                  points_are_rows::Bool = true)
  # Force the symbols to load.
  ccall((:loadSymbols, adaboostLibrary), Nothing, ());

  CLIRestoreSettings("AdaBoost")

  # Process each input argument before calling mlpackMain().
  if !ismissing(input_model)
    adaboost_internal.CLISetParamAdaBoostModelPtr("input_model", convert(Ptr{Nothing}, input_model))
  end
  if !ismissing(iterations)
    CLISetParam("iterations", convert(Int, iterations))
  end
  if !ismissing(labels)
    CLISetParamURow("labels", labels)
  end
  if !ismissing(test)
    CLISetParamMat("test", test, points_are_rows)
  end
  if !ismissing(tolerance)
    CLISetParam("tolerance", convert(Float64, tolerance))
  end
  if !ismissing(training)
    CLISetParamMat("training", training, points_are_rows)
  end
  if !ismissing(weak_learner)
    CLISetParam("weak_learner", convert(String, weak_learner))
  end
  if verbose !== nothing && verbose === true
    CLIEnableVerbose()
  else
    CLIDisableVerbose()
  end

  CLISetPassed("output")
  CLISetPassed("output_model")
  CLISetPassed("predictions")
  CLISetPassed("probabilities")
  # Call the program.
  adaboost_mlpackMain()

  return CLIGetParamURow("output"),
         adaboost_internal.CLIGetParamAdaBoostModelPtr("output_model"),
         CLIGetParamURow("predictions"),
         CLIGetParamMat("probabilities", points_are_rows)
end
