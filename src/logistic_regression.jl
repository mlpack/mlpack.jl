export logistic_regression

import ..LogisticRegression

using mlpack._Internal.params

import mlpack_jll
const logistic_regressionLibrary = mlpack_jll.libmlpack_julia_logistic_regression

# Call the C binding of the mlpack logistic_regression binding.
function call_logistic_regression(p, t)
  success = ccall((:mlpack_logistic_regression, logistic_regressionLibrary), Bool, (Ptr{Nothing}, Ptr{Nothing}), p, t)
  if !success
    # Throw an exception---false means there was a C++ exception.
    throw(ErrorException("mlpack binding error; see output"))
  end
end

" Internal module to hold utility functions. "
module logistic_regression_internal
  import ..logistic_regressionLibrary

import ...LogisticRegression

# Get the value of a model pointer parameter of type LogisticRegression.
function GetParamLogisticRegression(params::Ptr{Nothing}, paramName::String, modelPtrs::Set{Ptr{Nothing}})::LogisticRegression
  ptr = ccall((:GetParamLogisticRegressionPtr, logistic_regressionLibrary), Ptr{Nothing}, (Ptr{Nothing}, Cstring,), params, paramName)
  return LogisticRegression(ptr; finalize=!(ptr in modelPtrs))
end

# Set the value of a model pointer parameter of type LogisticRegression.
function SetParamLogisticRegression(params::Ptr{Nothing}, paramName::String, model::LogisticRegression)
  ccall((:SetParamLogisticRegressionPtr, logistic_regressionLibrary), Nothing, (Ptr{Nothing}, Cstring, Ptr{Nothing}), params, paramName, model.ptr)
end

# Delete an instantiated model pointer.
function DeleteLogisticRegression(ptr::Ptr{Nothing})
  ccall((:DeleteLogisticRegressionPtr, logistic_regressionLibrary), Nothing, (Ptr{Nothing},), ptr)
end

# Serialize a model to the given stream.
function serializeLogisticRegression(stream::IO, model::LogisticRegression)
  buf_len = UInt[0]
  buf_ptr = ccall((:SerializeLogisticRegressionPtr, logistic_regressionLibrary), Ptr{UInt8}, (Ptr{Nothing}, Ptr{UInt}), model.ptr, pointer(buf_len))
  buf = Base.unsafe_wrap(Vector{UInt8}, buf_ptr, buf_len[1]; own=true)
  write(stream, buf_len[1])
  write(stream, buf)
end
# Deserialize a model from the given stream.
function deserializeLogisticRegression(stream::IO)::LogisticRegression
  buf_len = read(stream, UInt)
  buffer = read(stream, buf_len)
  GC.@preserve buffer LogisticRegression(ccall((:DeserializeLogisticRegressionPtr, logistic_regressionLibrary), Ptr{Nothing}, (Ptr{UInt8}, UInt), pointer(buffer), length(buffer)))
end
end # module

"""
    logistic_regression(; [batch_size, decision_boundary, input_model, labels, lambda, max_iterations, optimizer, print_training_accuracy, step_size, test, tolerance, training, verbose])

An implementation of L2-regularized logistic regression using either the L-BFGS
optimizer or SGD (stochastic gradient descent).  This solves the regression
problem

  y = (1 / 1 + e^-(X * b)).

In this setting, y corresponds to class labels and X corresponds to data.

This program allows loading a logistic regression model (via the `input_model`
parameter) or training a logistic regression model given training data
(specified with the `training` parameter), or both those things at once.  In
addition, this program allows classification on a test dataset (specified with
the `test` parameter) and the classification results may be saved with the
`predictions` output parameter. The trained logistic regression model may be
saved using the `output_model` output parameter.

The training data, if specified, may have class labels as its last dimension. 
Alternately, the `labels` parameter may be used to specify a separate matrix of
labels.

When a model is being trained, there are many options.  L2 regularization (to
prevent overfitting) can be specified with the `lambda` option, and the
optimizer used to train the model can be specified with the `optimizer`
parameter.  Available options are 'sgd' (stochastic gradient descent) and
'lbfgs' (the L-BFGS optimizer).  There are also various parameters for the
optimizer; the `max_iterations` parameter specifies the maximum number of
allowed iterations, and the `tolerance` parameter specifies the tolerance for
convergence.  For the SGD optimizer, the `step_size` parameter controls the step
size taken at each iteration by the optimizer.  The batch size for SGD is
controlled with the `batch_size` parameter. If the objective function for your
data is oscillating between Inf and 0, the step size is probably too large. 
There are more parameters for the optimizers, but the C++ interface must be used
to access these.

For SGD, an iteration refers to a single point. So to take a single pass over
the dataset with SGD, `max_iterations` should be set to the number of points in
the dataset.

Optionally, the model can be used to predict the responses for another matrix of
data points, if `test` is specified.  The `test` parameter can be specified
without the `training` parameter, so long as an existing logistic regression
model is given with the `input_model` parameter.  The output predictions from
the logistic regression model may be saved with the `predictions` parameter.

This implementation of logistic regression does not support the general
multi-class case but instead only the two-class case.  Any labels must be either
1 or 2.  For more classes, see the softmax regression implementation.

As an example, to train a logistic regression model on the data '`data`' with
labels '`labels`' with L2 regularization of 0.1, saving the model to
'`lr_model`', the following command may be used:

```julia
julia> using CSV
julia> data = CSV.read("data.csv")
julia> labels = CSV.read("labels.csv"; type=Int)
julia> lr_model, _, _ = logistic_regression(labels=labels,
            lambda=0.1, print_training_accuracy=1, training=data)
```

Then, to use that model to predict classes for the dataset '`test`', storing the
output predictions in '`predictions`', the following command may be used: 

```julia
julia> using CSV
julia> test = CSV.read("test.csv")
julia> _, predictions, _ = logistic_regression(input_model=lr_model,
            test=test)
```

# Arguments

 - `batch_size::Int`: Batch size for SGD.  Default value `64`.

 - `decision_boundary::Float64`: Decision boundary for prediction; if the
      logistic function for a point is less than the boundary, the class is
      taken to be 0; otherwise, the class is 1.  Default value `0.5`.
      
 - `input_model::LogisticRegression`: Existing model (parameters).
 - `labels::Array{Int, 1}`: A matrix containing labels (0 or 1) for the
      points in the training set (y).
 - `lambda::Float64`: L2-regularization parameter for training.  Default
      value `0`.
      
 - `max_iterations::Int`: Maximum iterations for optimizer (0 indicates no
      limit).  Default value `10000`.
      
 - `optimizer::String`: Optimizer to use for training ('lbfgs' or 'sgd'). 
      Default value `lbfgs`.
      
 - `print_training_accuracy::Bool`: If set, then the accuracy of the model
      on the training set will be printed (verbose must also be specified). 
      Default value `false`.
      
 - `step_size::Float64`: Step size for SGD optimizer.  Default value
      `0.01`.
      
 - `test::Array{Float64, 2}`: Matrix containing test dataset.
 - `tolerance::Float64`: Convergence tolerance for optimizer.  Default
      value `1e-10`.
      
 - `training::Array{Float64, 2}`: A matrix containing the training set
      (the matrix of predictors, X).
 - `verbose::Bool`: Display informational messages and the full list of
      parameters and timers at the end of execution.  Default value `false`.
      

# Return values

 - `output_model::LogisticRegression`: Output for trained logistic
      regression model.
 - `predictions::Array{Int, 1}`: If test data is specified, this matrix is
      where the predictions for the test set will be saved.
 - `probabilities::Array{Float64, 2}`: If test data is specified, this
      matrix is where the class probabilities for the test set will be saved.

"""
function logistic_regression(;
                             batch_size::Union{Int, Missing} = missing,
                             decision_boundary::Union{Float64, Missing} = missing,
                             input_model::Union{LogisticRegression, Missing} = missing,
                             labels = missing,
                             lambda::Union{Float64, Missing} = missing,
                             max_iterations::Union{Int, Missing} = missing,
                             optimizer::Union{String, Missing} = missing,
                             print_training_accuracy::Union{Bool, Missing} = missing,
                             step_size::Union{Float64, Missing} = missing,
                             test = missing,
                             tolerance::Union{Float64, Missing} = missing,
                             training = missing,
                             verbose::Union{Bool, Missing} = missing,
                             points_are_rows::Bool = true)
  # Force the symbols to load.
  ccall((:loadSymbols, logistic_regressionLibrary), Nothing, ());

  # Create the set of model pointers to avoid setting multiple finalizers.
  modelPtrs = Set{Ptr{Nothing}}()

  p = GetParameters("logistic_regression")
  t = Timers()

  juliaOwnedMemory = Set{Ptr{Nothing}}()
  # Process each input argument before calling mlpackMain().
  if !ismissing(batch_size)
    SetParam(p, "batch_size", convert(Int, batch_size))
  end
  if !ismissing(decision_boundary)
    SetParam(p, "decision_boundary", convert(Float64, decision_boundary))
  end
  if !ismissing(input_model)
    push!(modelPtrs, convert(LogisticRegression, input_model).ptr)
    logistic_regression_internal.SetParamLogisticRegression(p, "input_model", convert(LogisticRegression, input_model))
  end
  if !ismissing(labels)
    SetParamURow(p, "labels", labels, juliaOwnedMemory)
  end
  if !ismissing(lambda)
    SetParam(p, "lambda", convert(Float64, lambda))
  end
  if !ismissing(max_iterations)
    SetParam(p, "max_iterations", convert(Int, max_iterations))
  end
  if !ismissing(optimizer)
    SetParam(p, "optimizer", convert(String, optimizer))
  end
  if !ismissing(print_training_accuracy)
    SetParam(p, "print_training_accuracy", convert(Bool, print_training_accuracy))
  end
  if !ismissing(step_size)
    SetParam(p, "step_size", convert(Float64, step_size))
  end
  if !ismissing(test)
    SetParamMat(p, "test", test, points_are_rows, false, juliaOwnedMemory)
  end
  if !ismissing(tolerance)
    SetParam(p, "tolerance", convert(Float64, tolerance))
  end
  if !ismissing(training)
    SetParamMat(p, "training", training, points_are_rows, false, juliaOwnedMemory)
  end
  if verbose !== nothing && verbose === true
    EnableVerbose()
  else
    DisableVerbose()
  end

  SetPassed(p, "output_model")
  SetPassed(p, "predictions")
  SetPassed(p, "probabilities")
  # Call the program.
  call_logistic_regression(p, t)

  results = (logistic_regression_internal.GetParamLogisticRegression(p, "output_model", modelPtrs),
             GetParamURow(p, "predictions", juliaOwnedMemory),
             GetParamMat(p, "probabilities", points_are_rows, juliaOwnedMemory))

  # We are responsible for cleaning up the `p` and `t` objects.
  DeleteParameters(p)
  DeleteTimers(t)

  return results
end
