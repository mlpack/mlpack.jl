export linear_svm

import ..LinearSVMModel

using mlpack._Internal.io

import mlpack_jll
const linear_svmLibrary = mlpack_jll.libmlpack_julia_linear_svm

# Call the C binding of the mlpack linear_svm binding.
function linear_svm_mlpackMain()
  success = ccall((:linear_svm, linear_svmLibrary), Bool, ())
  if !success
    # Throw an exception---false means there was a C++ exception.
    throw(ErrorException("mlpack binding error; see output"))
  end
end

" Internal module to hold utility functions. "
module linear_svm_internal
  import ..linear_svmLibrary

import ...LinearSVMModel

# Get the value of a model pointer parameter of type LinearSVMModel.
function IOGetParamLinearSVMModel(paramName::String)::LinearSVMModel
  LinearSVMModel(ccall((:IO_GetParamLinearSVMModelPtr, linear_svmLibrary), Ptr{Nothing}, (Cstring,), paramName))
end

# Set the value of a model pointer parameter of type LinearSVMModel.
function IOSetParamLinearSVMModel(paramName::String, model::LinearSVMModel)
  ccall((:IO_SetParamLinearSVMModelPtr, linear_svmLibrary), Nothing, (Cstring, Ptr{Nothing}), paramName, model.ptr)
end

# Serialize a model to the given stream.
function serializeLinearSVMModel(stream::IO, model::LinearSVMModel)
  buf_len = UInt[0]
  buf_ptr = ccall((:SerializeLinearSVMModelPtr, linear_svmLibrary), Ptr{UInt8}, (Ptr{Nothing}, Ptr{UInt}), model.ptr, pointer(buf_len))
  buf = Base.unsafe_wrap(Vector{UInt8}, buf_ptr, buf_len[1]; own=true)
  write(stream, buf)
end
# Deserialize a model from the given stream.
function deserializeLinearSVMModel(stream::IO)::LinearSVMModel
  buffer = read(stream)
  GC.@preserve buffer LinearSVMModel(ccall((:DeserializeLinearSVMModelPtr, linear_svmLibrary), Ptr{Nothing}, (Ptr{UInt8}, UInt), pointer(buffer), length(buffer)))
end
end # module

"""
    linear_svm(; [delta, epochs, input_model, labels, lambda, max_iterations, no_intercept, num_classes, optimizer, seed, shuffle, step_size, test, test_labels, tolerance, training, verbose])

An implementation of linear SVMs that uses either L-BFGS or parallel SGD
(stochastic gradient descent) to train the model.

This program allows loading a linear SVM model (via the `input_model` parameter)
or training a linear SVM model given training data (specified with the
`training` parameter), or both those things at once.  In addition, this program
allows classification on a test dataset (specified with the `test` parameter)
and the classification results may be saved with the `predictions` output
parameter. The trained linear SVM model may be saved using the `output_model`
output parameter.

The training data, if specified, may have class labels as its last dimension. 
Alternately, the `labels` parameter may be used to specify a separate vector of
labels.

When a model is being trained, there are many options.  L2 regularization (to
prevent overfitting) can be specified with the `lambda` option, and the number
of classes can be manually specified with the `num_classes`and if an intercept
term is not desired in the model, the `no_intercept` parameter can be
specified.Margin of difference between correct class and other classes can be
specified with the `delta` option.The optimizer used to train the model can be
specified with the `optimizer` parameter.  Available options are 'psgd'
(parallel stochastic gradient descent) and 'lbfgs' (the L-BFGS optimizer). 
There are also various parameters for the optimizer; the `max_iterations`
parameter specifies the maximum number of allowed iterations, and the
`tolerance` parameter specifies the tolerance for convergence.  For the parallel
SGD optimizer, the `step_size` parameter controls the step size taken at each
iteration by the optimizer and the maximum number of epochs (specified with
`epochs`). If the objective function for your data is oscillating between Inf
and 0, the step size is probably too large.  There are more parameters for the
optimizers, but the C++ interface must be used to access these.

Optionally, the model can be used to predict the labels for another matrix of
data points, if `test` is specified.  The `test` parameter can be specified
without the `training` parameter, so long as an existing linear SVM model is
given with the `input_model` parameter.  The output predictions from the linear
SVM model may be saved with the `predictions` parameter.

As an example, to train a LinaerSVM on the data '`data`' with labels '`labels`'
with L2 regularization of 0.1, saving the model to '`lsvm_model`', the following
command may be used:

```julia
julia> using CSV
julia> data = CSV.read("data.csv")
julia> labels = CSV.read("labels.csv"; type=Int)
julia> lsvm_model, _, _ = linear_svm(delta=1, labels=labels,
            lambda=0.1, num_classes=0, training=data)
```

Then, to use that model to predict classes for the dataset '`test`', storing the
output predictions in '`predictions`', the following command may be used: 

```julia
julia> using CSV
julia> test = CSV.read("test.csv")
julia> _, predictions, _ = linear_svm(input_model=lsvm_model,
            test=test)
```

# Arguments

 - `delta::Float64`: Margin of difference between correct class and other
      classes.  Default value `1`.
      
 - `epochs::Int`: Maximum number of full epochs over dataset for psgd 
      Default value `50`.
      
 - `input_model::LinearSVMModel`: Existing model (parameters).
 - `labels::Array{Int, 1}`: A matrix containing labels (0 or 1) for the
      points in the training set (y).
 - `lambda::Float64`: L2-regularization parameter for training.  Default
      value `0.0001`.
      
 - `max_iterations::Int`: Maximum iterations for optimizer (0 indicates no
      limit).  Default value `10000`.
      
 - `no_intercept::Bool`: Do not add the intercept term to the model. 
      Default value `false`.
      
 - `num_classes::Int`: Number of classes for classification; if
      unspecified (or 0), the number of classes found in the labels will be
      used.  Default value `0`.
      
 - `optimizer::String`: Optimizer to use for training ('lbfgs' or 'psgd').
       Default value `lbfgs`.
      
 - `seed::Int`: Random seed.  If 0, 'std::time(NULL)' is used.  Default
      value `0`.
      
 - `shuffle::Bool`: Don't shuffle the order in which data points are
      visited for parallel SGD.  Default value `false`.
      
 - `step_size::Float64`: Step size for parallel SGD optimizer.  Default
      value `0.01`.
      
 - `test::Array{Float64, 2}`: Matrix containing test dataset.
 - `test_labels::Array{Int, 1}`: Matrix containing test labels.
 - `tolerance::Float64`: Convergence tolerance for optimizer.  Default
      value `1e-10`.
      
 - `training::Array{Float64, 2}`: A matrix containing the training set
      (the matrix of predictors, X).
 - `verbose::Bool`: Display informational messages and the full list of
      parameters and timers at the end of execution.  Default value `false`.
      

# Return values

 - `output_model::LinearSVMModel`: Output for trained linear svm model.
 - `predictions::Array{Int, 1}`: If test data is specified, this matrix is
      where the predictions for the test set will be saved.
 - `probabilities::Array{Float64, 2}`: If test data is specified, this
      matrix is where the class probabilities for the test set will be saved.

"""
function linear_svm(;
                    delta::Union{Float64, Missing} = missing,
                    epochs::Union{Int, Missing} = missing,
                    input_model::Union{LinearSVMModel, Missing} = missing,
                    labels = missing,
                    lambda::Union{Float64, Missing} = missing,
                    max_iterations::Union{Int, Missing} = missing,
                    no_intercept::Union{Bool, Missing} = missing,
                    num_classes::Union{Int, Missing} = missing,
                    optimizer::Union{String, Missing} = missing,
                    seed::Union{Int, Missing} = missing,
                    shuffle::Union{Bool, Missing} = missing,
                    step_size::Union{Float64, Missing} = missing,
                    test = missing,
                    test_labels = missing,
                    tolerance::Union{Float64, Missing} = missing,
                    training = missing,
                    verbose::Union{Bool, Missing} = missing,
                    points_are_rows::Bool = true)
  # Force the symbols to load.
  ccall((:loadSymbols, linear_svmLibrary), Nothing, ());

  IORestoreSettings("Linear SVM is an L2-regularized support vector machine.")

  # Process each input argument before calling mlpackMain().
  if !ismissing(delta)
    IOSetParam("delta", convert(Float64, delta))
  end
  if !ismissing(epochs)
    IOSetParam("epochs", convert(Int, epochs))
  end
  if !ismissing(input_model)
    linear_svm_internal.IOSetParamLinearSVMModel("input_model", convert(LinearSVMModel, input_model))
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
  if !ismissing(num_classes)
    IOSetParam("num_classes", convert(Int, num_classes))
  end
  if !ismissing(optimizer)
    IOSetParam("optimizer", convert(String, optimizer))
  end
  if !ismissing(seed)
    IOSetParam("seed", convert(Int, seed))
  end
  if !ismissing(shuffle)
    IOSetParam("shuffle", convert(Bool, shuffle))
  end
  if !ismissing(step_size)
    IOSetParam("step_size", convert(Float64, step_size))
  end
  if !ismissing(test)
    IOSetParamMat("test", test, points_are_rows)
  end
  if !ismissing(test_labels)
    IOSetParamURow("test_labels", test_labels)
  end
  if !ismissing(tolerance)
    IOSetParam("tolerance", convert(Float64, tolerance))
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
  IOSetPassed("probabilities")
  # Call the program.
  linear_svm_mlpackMain()

  return linear_svm_internal.IOGetParamLinearSVMModel("output_model"),
         IOGetParamURow("predictions"),
         IOGetParamMat("probabilities", points_are_rows)
end
