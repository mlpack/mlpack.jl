export bayesian_linear_regression

import ..BayesianLinearRegression

using mlpack._Internal.params

import mlpack_jll
const bayesian_linear_regressionLibrary = mlpack_jll.libmlpack_julia_bayesian_linear_regression

# Call the C binding of the mlpack bayesian_linear_regression binding.
function call_bayesian_linear_regression(p, t)
  success = ccall((:mlpack_bayesian_linear_regression, bayesian_linear_regressionLibrary), Bool, (Ptr{Nothing}, Ptr{Nothing}), p, t)
  if !success
    # Throw an exception---false means there was a C++ exception.
    throw(ErrorException("mlpack binding error; see output"))
  end
end

" Internal module to hold utility functions. "
module bayesian_linear_regression_internal
  import ..bayesian_linear_regressionLibrary

import ...BayesianLinearRegression

# Get the value of a model pointer parameter of type BayesianLinearRegression.
function GetParamBayesianLinearRegression(params::Ptr{Nothing}, paramName::String, modelPtrs::Set{Ptr{Nothing}})::BayesianLinearRegression
  ptr = ccall((:GetParamBayesianLinearRegressionPtr, bayesian_linear_regressionLibrary), Ptr{Nothing}, (Ptr{Nothing}, Cstring,), params, paramName)
  return BayesianLinearRegression(ptr; finalize=!(ptr in modelPtrs))
end

# Set the value of a model pointer parameter of type BayesianLinearRegression.
function SetParamBayesianLinearRegression(params::Ptr{Nothing}, paramName::String, model::BayesianLinearRegression)
  ccall((:SetParamBayesianLinearRegressionPtr, bayesian_linear_regressionLibrary), Nothing, (Ptr{Nothing}, Cstring, Ptr{Nothing}), params, paramName, model.ptr)
end

# Delete an instantiated model pointer.
function DeleteBayesianLinearRegression(ptr::Ptr{Nothing})
  ccall((:DeleteBayesianLinearRegressionPtr, bayesian_linear_regressionLibrary), Nothing, (Ptr{Nothing},), ptr)
end

# Serialize a model to the given stream.
function serializeBayesianLinearRegression(stream::IO, model::BayesianLinearRegression)
  buf_len = UInt[0]
  buf_ptr = ccall((:SerializeBayesianLinearRegressionPtr, bayesian_linear_regressionLibrary), Ptr{UInt8}, (Ptr{Nothing}, Ptr{UInt}), model.ptr, pointer(buf_len))
  buf = Base.unsafe_wrap(Vector{UInt8}, buf_ptr, buf_len[1]; own=true)
  write(stream, buf_len[1])
  write(stream, buf)
end
# Deserialize a model from the given stream.
function deserializeBayesianLinearRegression(stream::IO)::BayesianLinearRegression
  buf_len = read(stream, UInt)
  buffer = read(stream, buf_len)
  GC.@preserve buffer BayesianLinearRegression(ccall((:DeserializeBayesianLinearRegressionPtr, bayesian_linear_regressionLibrary), Ptr{Nothing}, (Ptr{UInt8}, UInt), pointer(buffer), length(buffer)))
end
end # module

"""
    bayesian_linear_regression(; [center, input, input_model, responses, scale, test, verbose])

An implementation of the bayesian linear regression.
This model is a probabilistic view and implementation of the linear regression.
The final solution is obtained by computing a posterior distribution from
gaussian likelihood and a zero mean gaussian isotropic  prior distribution on
the solution. 
Optimization is AUTOMATIC and does not require cross validation. The
optimization is performed by maximization of the evidence function. Parameters
are tuned during the maximization of the marginal likelihood. This procedure
includes the Ockham's razor that penalizes over complex solutions. 

This program is able to train a Bayesian linear regression model or load a model
from file, output regression predictions for a test set, and save the trained
model to a file.

To train a BayesianLinearRegression model, the `input` and `responses`parameters
must be given. The `center`and `scale` parameters control the centering and the
normalizing options. A trained model can be saved with the `output_model`. If no
training is desired at all, a model can be passed via the `input_model`
parameter.

The program can also provide predictions for test data using either the trained
model or the given input model.  Test points can be specified with the `test`
parameter.  Predicted responses to the test points can be saved with the
`predictions` output parameter. The corresponding standard deviation can be save
by precising the `stds` parameter.

For example, the following command trains a model on the data `data` and
responses `responses`with center set to true and scale set to false (so,
Bayesian linear regression is being solved, and then the model is saved to
`blr_model`:

```julia
julia> using CSV
julia> data = CSV.read("data.csv")
julia> responses = CSV.read("responses.csv")
julia> blr_model, _, _ = bayesian_linear_regression(center=1,
            input=data, responses=responses, scale=0)
```

The following command uses the `blr_model` to provide predicted  responses for
the data `test` and save those  responses to `test_predictions`: 

```julia
julia> using CSV
julia> test = CSV.read("test.csv")
julia> _, test_predictions, _ =
            bayesian_linear_regression(input_model=blr_model, test=test)
```

Because the estimator computes a predictive distribution instead of a simple
point estimate, the `stds` parameter allows one to save the prediction
uncertainties: 

```julia
julia> using CSV
julia> test = CSV.read("test.csv")
julia> _, test_predictions, stds =
            bayesian_linear_regression(input_model=blr_model, test=test)
```

# Arguments

 - `center::Bool`: Center the data and fit the intercept if enabled. 
      Default value `false`.
      
 - `input::Array{Float64, 2}`: Matrix of covariates (X).
 - `input_model::BayesianLinearRegression`: Trained
      BayesianLinearRegression model to use.
 - `responses::Array{Float64, 1}`: Matrix of responses/observations (y).
 - `scale::Bool`: Scale each feature by their standard deviations if
      enabled.  Default value `false`.
      
 - `test::Array{Float64, 2}`: Matrix containing points to regress on (test
      points).
 - `verbose::Bool`: Display informational messages and the full list of
      parameters and timers at the end of execution.  Default value `false`.
      

# Return values

 - `output_model::BayesianLinearRegression`: Output
      BayesianLinearRegression model.
 - `predictions::Array{Float64, 2}`: If --test_file is specified, this
      file is where the predicted responses will be saved.
 - `stds::Array{Float64, 2}`: If specified, this is where the standard
      deviations of the predictive distribution will be saved.

"""
function bayesian_linear_regression(;
                                    center::Union{Bool, Missing} = missing,
                                    input = missing,
                                    input_model::Union{BayesianLinearRegression, Missing} = missing,
                                    responses = missing,
                                    scale::Union{Bool, Missing} = missing,
                                    test = missing,
                                    verbose::Union{Bool, Missing} = missing,
                                    points_are_rows::Bool = true)
  # Force the symbols to load.
  ccall((:loadSymbols, bayesian_linear_regressionLibrary), Nothing, ());

  # Create the set of model pointers to avoid setting multiple finalizers.
  modelPtrs = Set{Ptr{Nothing}}()

  p = GetParameters("bayesian_linear_regression")
  t = Timers()

  juliaOwnedMemory = Set{Ptr{Nothing}}()
  # Process each input argument before calling mlpackMain().
  if !ismissing(center)
    SetParam(p, "center", convert(Bool, center))
  end
  if !ismissing(input)
    SetParamMat(p, "input", input, points_are_rows, false, juliaOwnedMemory)
  end
  if !ismissing(input_model)
    push!(modelPtrs, convert(BayesianLinearRegression, input_model).ptr)
    bayesian_linear_regression_internal.SetParamBayesianLinearRegression(p, "input_model", convert(BayesianLinearRegression, input_model))
  end
  if !ismissing(responses)
    SetParamRow(p, "responses", responses, juliaOwnedMemory)
  end
  if !ismissing(scale)
    SetParam(p, "scale", convert(Bool, scale))
  end
  if !ismissing(test)
    SetParamMat(p, "test", test, points_are_rows, false, juliaOwnedMemory)
  end
  if verbose !== nothing && verbose === true
    EnableVerbose()
  else
    DisableVerbose()
  end

  SetPassed(p, "output_model")
  SetPassed(p, "predictions")
  SetPassed(p, "stds")
  # Call the program.
  call_bayesian_linear_regression(p, t)

  results = (bayesian_linear_regression_internal.GetParamBayesianLinearRegression(p, "output_model", modelPtrs),
             GetParamMat(p, "predictions", points_are_rows, juliaOwnedMemory),
             GetParamMat(p, "stds", points_are_rows, juliaOwnedMemory))

  # We are responsible for cleaning up the `p` and `t` objects.
  DeleteParameters(p)
  DeleteTimers(t)

  return results
end
