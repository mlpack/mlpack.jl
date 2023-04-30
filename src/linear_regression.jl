export linear_regression

import ..LinearRegression

using mlpack._Internal.params

import mlpack_jll
const linear_regressionLibrary = mlpack_jll.libmlpack_julia_linear_regression

# Call the C binding of the mlpack linear_regression binding.
function call_linear_regression(p, t)
  success = ccall((:mlpack_linear_regression, linear_regressionLibrary), Bool, (Ptr{Nothing}, Ptr{Nothing}), p, t)
  if !success
    # Throw an exception---false means there was a C++ exception.
    throw(ErrorException("mlpack binding error; see output"))
  end
end

" Internal module to hold utility functions. "
module linear_regression_internal
  import ..linear_regressionLibrary

import ...LinearRegression

# Get the value of a model pointer parameter of type LinearRegression.
function GetParamLinearRegression(params::Ptr{Nothing}, paramName::String, modelPtrs::Set{Ptr{Nothing}})::LinearRegression
  ptr = ccall((:GetParamLinearRegressionPtr, linear_regressionLibrary), Ptr{Nothing}, (Ptr{Nothing}, Cstring,), params, paramName)
  return LinearRegression(ptr; finalize=!(ptr in modelPtrs))
end

# Set the value of a model pointer parameter of type LinearRegression.
function SetParamLinearRegression(params::Ptr{Nothing}, paramName::String, model::LinearRegression)
  ccall((:SetParamLinearRegressionPtr, linear_regressionLibrary), Nothing, (Ptr{Nothing}, Cstring, Ptr{Nothing}), params, paramName, model.ptr)
end

# Delete an instantiated model pointer.
function DeleteLinearRegression(ptr::Ptr{Nothing})
  ccall((:DeleteLinearRegressionPtr, linear_regressionLibrary), Nothing, (Ptr{Nothing},), ptr)
end

# Serialize a model to the given stream.
function serializeLinearRegression(stream::IO, model::LinearRegression)
  buf_len = UInt[0]
  buf_ptr = ccall((:SerializeLinearRegressionPtr, linear_regressionLibrary), Ptr{UInt8}, (Ptr{Nothing}, Ptr{UInt}), model.ptr, pointer(buf_len))
  buf = Base.unsafe_wrap(Vector{UInt8}, buf_ptr, buf_len[1]; own=true)
  write(stream, buf_len[1])
  write(stream, buf)
end
# Deserialize a model from the given stream.
function deserializeLinearRegression(stream::IO)::LinearRegression
  buf_len = read(stream, UInt)
  buffer = read(stream, buf_len)
  GC.@preserve buffer LinearRegression(ccall((:DeserializeLinearRegressionPtr, linear_regressionLibrary), Ptr{Nothing}, (Ptr{UInt8}, UInt), pointer(buffer), length(buffer)))
end
end # module

"""
    linear_regression(; [input_model, lambda, test, training, training_responses, verbose])

An implementation of simple linear regression and simple ridge regression using
ordinary least squares. This solves the problem

  y = X * b + e

where X (specified by `training`) and y (specified either as the last column of
the input matrix `training` or via the `training_responses` parameter) are known
and b is the desired variable.  If the covariance matrix (X'X) is not
invertible, or if the solution is overdetermined, then specify a Tikhonov
regularization constant (with `lambda`) greater than 0, which will regularize
the covariance matrix to make it invertible.  The calculated b may be saved with
the `output_predictions` output parameter.

Optionally, the calculated value of b is used to predict the responses for
another matrix X' (specified by the `test` parameter):

   y' = X' * b

and the predicted responses y' may be saved with the `output_predictions` output
parameter.  This type of regression is related to least-angle regression, which
mlpack implements as the 'lars' program.

For example, to run a linear regression on the dataset `X` with responses `y`,
saving the trained model to `lr_model`, the following command could be used:

```julia
julia> using CSV
julia> X = CSV.read("X.csv")
julia> y = CSV.read("y.csv")
julia> lr_model, _ = linear_regression(training=X,
            training_responses=y)
```

Then, to use `lr_model` to predict responses for a test set `X_test`, saving the
predictions to `X_test_responses`, the following command could be used:

```julia
julia> using CSV
julia> X_test = CSV.read("X_test.csv")
julia> _, X_test_responses = linear_regression(input_model=lr_model,
            test=X_test)
```

# Arguments

 - `input_model::LinearRegression`: Existing LinearRegression model to
      use.
 - `lambda::Float64`: Tikhonov regularization for ridge regression.  If 0,
      the method reduces to linear regression.  Default value `0`.
      
 - `test::Array{Float64, 2}`: Matrix containing X' (test regressors).
 - `training::Array{Float64, 2}`: Matrix containing training set X
      (regressors).
 - `training_responses::Array{Float64, 1}`: Optional vector containing y
      (responses). If not given, the responses are assumed to be the last row of
      the input file.
 - `verbose::Bool`: Display informational messages and the full list of
      parameters and timers at the end of execution.  Default value `false`.
      

# Return values

 - `output_model::LinearRegression`: Output LinearRegression model.
 - `output_predictions::Array{Float64, 1}`: If --test_file is specified,
      this matrix is where the predicted responses will be saved.

"""
function linear_regression(;
                           input_model::Union{LinearRegression, Missing} = missing,
                           lambda::Union{Float64, Missing} = missing,
                           test = missing,
                           training = missing,
                           training_responses = missing,
                           verbose::Union{Bool, Missing} = missing,
                           points_are_rows::Bool = true)
  # Force the symbols to load.
  ccall((:loadSymbols, linear_regressionLibrary), Nothing, ());

  # Create the set of model pointers to avoid setting multiple finalizers.
  modelPtrs = Set{Ptr{Nothing}}()

  p = GetParameters("linear_regression")
  t = Timers()

  juliaOwnedMemory = Set{Ptr{Nothing}}()
  # Process each input argument before calling mlpackMain().
  if !ismissing(input_model)
    push!(modelPtrs, convert(LinearRegression, input_model).ptr)
    linear_regression_internal.SetParamLinearRegression(p, "input_model", convert(LinearRegression, input_model))
  end
  if !ismissing(lambda)
    SetParam(p, "lambda", convert(Float64, lambda))
  end
  if !ismissing(test)
    SetParamMat(p, "test", test, points_are_rows, false, juliaOwnedMemory)
  end
  if !ismissing(training)
    SetParamMat(p, "training", training, points_are_rows, false, juliaOwnedMemory)
  end
  if !ismissing(training_responses)
    SetParamRow(p, "training_responses", training_responses, juliaOwnedMemory)
  end
  if verbose !== nothing && verbose === true
    EnableVerbose()
  else
    DisableVerbose()
  end

  SetPassed(p, "output_model")
  SetPassed(p, "output_predictions")
  # Call the program.
  call_linear_regression(p, t)

  results = (linear_regression_internal.GetParamLinearRegression(p, "output_model", modelPtrs),
             GetParamRow(p, "output_predictions", juliaOwnedMemory))

  # We are responsible for cleaning up the `p` and `t` objects.
  DeleteParameters(p)
  DeleteTimers(t)

  return results
end
