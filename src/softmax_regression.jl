export softmax_regression

import ..SoftmaxRegression

using mlpack._Internal.io

import mlpack_jll
const softmax_regressionLibrary = mlpack_jll.libmlpack_julia_softmax_regression

# Call the C binding of the mlpack softmax_regression binding.
function softmax_regression_mlpackMain()
  success = ccall((:softmax_regression, softmax_regressionLibrary), Bool, ())
  if !success
    # Throw an exception---false means there was a C++ exception.
    throw(ErrorException("mlpack binding error; see output"))
  end
end

" Internal module to hold utility functions. "
module softmax_regression_internal
  import ..softmax_regressionLibrary

import ...SoftmaxRegression

# Get the value of a model pointer parameter of type SoftmaxRegression.
function IOGetParamSoftmaxRegression(paramName::String)::SoftmaxRegression
  SoftmaxRegression(ccall((:IO_GetParamSoftmaxRegressionPtr, softmax_regressionLibrary), Ptr{Nothing}, (Cstring,), paramName))
end

# Set the value of a model pointer parameter of type SoftmaxRegression.
function IOSetParamSoftmaxRegression(paramName::String, model::SoftmaxRegression)
  ccall((:IO_SetParamSoftmaxRegressionPtr, softmax_regressionLibrary), Nothing, (Cstring, Ptr{Nothing}), paramName, model.ptr)
end

# Serialize a model to the given stream.
function serializeSoftmaxRegression(stream::IO, model::SoftmaxRegression)
  buf_len = UInt[0]
  buf_ptr = ccall((:SerializeSoftmaxRegressionPtr, softmax_regressionLibrary), Ptr{UInt8}, (Ptr{Nothing}, Ptr{UInt}), model.ptr, pointer(buf_len))
  buf = Base.unsafe_wrap(Vector{UInt8}, buf_ptr, buf_len[1]; own=true)
  write(stream, buf)
end
# Deserialize a model from the given stream.
function deserializeSoftmaxRegression(stream::IO)::SoftmaxRegression
  buffer = read(stream)
  GC.@preserve buffer SoftmaxRegression(ccall((:DeserializeSoftmaxRegressionPtr, softmax_regressionLibrary), Ptr{Nothing}, (Ptr{UInt8}, UInt), pointer(buffer), length(buffer)))
end
end # module

"""
    softmax_regression(; [input_model, labels, lambda, max_iterations, no_intercept, number_of_classes, test, test_labels, training, verbose])

This program performs softmax regression, a generalization of logistic
regression to the multiclass case, and has support for L2 regularization.  The
program is able to train a model, load  an existing model, and give predictions
(and optionally their accuracy) for test data.

Training a softmax regression model is done by giving a file of training points
with the `training` parameter and their corresponding labels with the `labels`
parameter. The number of classes can be manually specified with the
`number_of_classes` parameter, and the maximum number of iterations of the
L-BFGS optimizer can be specified with the `max_iterations` parameter.  The L2
regularization constant can be specified with the `lambda` parameter and if an
intercept term is not desired in the model, the `no_intercept` parameter can be
specified.

The trained model can be saved with the `output_model` output parameter. If
training is not desired, but only testing is, a model can be loaded with the
`input_model` parameter.  At the current time, a loaded model cannot be trained
further, so specifying both `input_model` and `training` is not allowed.

The program is also able to evaluate a model on test data.  A test dataset can
be specified with the `test` parameter. Class predictions can be saved with the
`predictions` output parameter.  If labels are specified for the test data with
the `test_labels` parameter, then the program will print the accuracy of the
predictions on the given test set and its corresponding labels.

For example, to train a softmax regression model on the data `dataset` with
labels `labels` with a maximum of 1000 iterations for training, saving the
trained model to `sr_model`, the following command can be used: 

```julia
julia> using CSV
julia> dataset = CSV.read("dataset.csv")
julia> labels = CSV.read("labels.csv"; type=Int)
julia> sr_model, _ = softmax_regression(labels=labels,
            training=dataset)
```

Then, to use `sr_model` to classify the test points in `test_points`, saving the
output predictions to `predictions`, the following command can be used:

```julia
julia> using CSV
julia> test_points = CSV.read("test_points.csv")
julia> _, predictions = softmax_regression(input_model=sr_model,
            test=test_points)
```

# Arguments

 - `input_model::SoftmaxRegression`: File containing existing model
      (parameters).
 - `labels::Array{Int, 1}`: A matrix containing labels (0 or 1) for the
      points in the training set (y). The labels must order as a row.
 - `lambda::Float64`: L2-regularization constant  Default value `0.0001`.
      
 - `max_iterations::Int`: Maximum number of iterations before termination.
       Default value `400`.
      
 - `no_intercept::Bool`: Do not add the intercept term to the model. 
      Default value `false`.
      
 - `number_of_classes::Int`: Number of classes for classification; if
      unspecified (or 0), the number of classes found in the labels will be
      used.  Default value `0`.
      
 - `test::Array{Float64, 2}`: Matrix containing test dataset.
 - `test_labels::Array{Int, 1}`: Matrix containing test labels.
 - `training::Array{Float64, 2}`: A matrix containing the training set
      (the matrix of predictors, X).
 - `verbose::Bool`: Display informational messages and the full list of
      parameters and timers at the end of execution.  Default value `false`.
      

# Return values

 - `output_model::SoftmaxRegression`: File to save trained softmax
      regression model to.
 - `predictions::Array{Int, 1}`: Matrix to save predictions for test
      dataset into.

"""
function softmax_regression(;
                            input_model::Union{SoftmaxRegression, Missing} = missing,
                            labels = missing,
                            lambda::Union{Float64, Missing} = missing,
                            max_iterations::Union{Int, Missing} = missing,
                            no_intercept::Union{Bool, Missing} = missing,
                            number_of_classes::Union{Int, Missing} = missing,
                            test = missing,
                            test_labels = missing,
                            training = missing,
                            verbose::Union{Bool, Missing} = missing,
                            points_are_rows::Bool = true)
  # Force the symbols to load.
  ccall((:loadSymbols, softmax_regressionLibrary), Nothing, ());

  IORestoreSettings("Softmax Regression")

  # Process each input argument before calling mlpackMain().
  if !ismissing(input_model)
    softmax_regression_internal.IOSetParamSoftmaxRegression("input_model", convert(SoftmaxRegression, input_model))
  end
  if !ismissing(labels)
    IOSetParamURow("labels", labels)
  end
  if !ismissing(lambda)
    IOSetParam("lambda", convert(Float64, lambda))
  end
  if !ismissing(max_iterations)
    IOSetParam("max_iterations", convert(Int, max_iterations))
  end
  if !ismissing(no_intercept)
    IOSetParam("no_intercept", convert(Bool, no_intercept))
  end
  if !ismissing(number_of_classes)
    IOSetParam("number_of_classes", convert(Int, number_of_classes))
  end
  if !ismissing(test)
    IOSetParamMat("test", test, points_are_rows)
  end
  if !ismissing(test_labels)
    IOSetParamURow("test_labels", test_labels)
  end
  if !ismissing(training)
    IOSetParamMat("training", training, points_are_rows)
  end
  if verbose !== nothing && verbose === true
    IOEnableVerbose()
  else
    IODisableVerbose()
  end

  IOSetPassed("output_model")
  IOSetPassed("predictions")
  # Call the program.
  softmax_regression_mlpackMain()

  return softmax_regression_internal.IOGetParamSoftmaxRegression("output_model"),
         IOGetParamURow("predictions")
end
