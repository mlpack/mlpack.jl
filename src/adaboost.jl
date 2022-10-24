export adaboost

import ..AdaBoostModel

using mlpack._Internal.params

import mlpack_jll
const adaboostLibrary = mlpack_jll.libmlpack_julia_adaboost

# Call the C binding of the mlpack adaboost binding.
function call_adaboost(p, t)
  success = ccall((:mlpack_adaboost, adaboostLibrary), Bool, (Ptr{Nothing}, Ptr{Nothing}), p, t)
  if !success
    # Throw an exception---false means there was a C++ exception.
    throw(ErrorException("mlpack binding error; see output"))
  end
end

" Internal module to hold utility functions. "
module adaboost_internal
  import ..adaboostLibrary

import ...AdaBoostModel

# Get the value of a model pointer parameter of type AdaBoostModel.
function GetParamAdaBoostModel(params::Ptr{Nothing}, paramName::String, modelPtrs::Set{Ptr{Nothing}})::AdaBoostModel
  ptr = ccall((:GetParamAdaBoostModelPtr, adaboostLibrary), Ptr{Nothing}, (Ptr{Nothing}, Cstring,), params, paramName)
  return AdaBoostModel(ptr; finalize=!(ptr in modelPtrs))
end

# Set the value of a model pointer parameter of type AdaBoostModel.
function SetParamAdaBoostModel(params::Ptr{Nothing}, paramName::String, model::AdaBoostModel)
  ccall((:SetParamAdaBoostModelPtr, adaboostLibrary), Nothing, (Ptr{Nothing}, Cstring, Ptr{Nothing}), params, paramName, model.ptr)
end

# Delete an instantiated model pointer.
function DeleteAdaBoostModel(ptr::Ptr{Nothing})
  ccall((:DeleteAdaBoostModelPtr, adaboostLibrary), Nothing, (Ptr{Nothing},), ptr)
end

# Serialize a model to the given stream.
function serializeAdaBoostModel(stream::IO, model::AdaBoostModel)
  buf_len = UInt[0]
  buf_ptr = ccall((:SerializeAdaBoostModelPtr, adaboostLibrary), Ptr{UInt8}, (Ptr{Nothing}, Ptr{UInt}), model.ptr, pointer(buf_len))
  buf = Base.unsafe_wrap(Vector{UInt8}, buf_ptr, buf_len[1]; own=true)
  write(stream, buf_len[1])
  write(stream, buf)
end
# Deserialize a model from the given stream.
function deserializeAdaBoostModel(stream::IO)::AdaBoostModel
  buf_len = read(stream, UInt)
  buffer = read(stream, buf_len)
  GC.@preserve buffer AdaBoostModel(ccall((:DeserializeAdaBoostModelPtr, adaboostLibrary), Ptr{Nothing}, (Ptr{UInt8}, UInt), pointer(buffer), length(buffer)))
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

For example, to run AdaBoost on an input dataset `data` with labels `labels`and
perceptrons as the weak learner type, storing the trained model in `model`, one
could use the following command: 

```julia
julia> using CSV
julia> data = CSV.read("data.csv")
julia> labels = CSV.read("labels.csv"; type=Int)
julia> _, model, _, _ = adaboost(labels=labels, training=data,
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

 - `input_model::AdaBoostModel`: Input AdaBoost model.
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
 - `output_model::AdaBoostModel`: Output trained AdaBoost model.
 - `predictions::Array{Int, 1}`: Predicted labels for the test set.
 - `probabilities::Array{Float64, 2}`: Predicted class probabilities for
      each point in the test set.

"""
function adaboost(;
                  input_model::Union{AdaBoostModel, Missing} = missing,
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

  # Create the set of model pointers to avoid setting multiple finalizers.
  modelPtrs = Set{Ptr{Nothing}}()

  p = GetParameters("adaboost")
  t = Timers()

  juliaOwnedMemory = Set{Ptr{Nothing}}()
  # Process each input argument before calling mlpackMain().
  if !ismissing(input_model)
    push!(modelPtrs, convert(AdaBoostModel, input_model).ptr)
    adaboost_internal.SetParamAdaBoostModel(p, "input_model", convert(AdaBoostModel, input_model))
  end
  if !ismissing(iterations)
    SetParam(p, "iterations", convert(Int, iterations))
  end
  if !ismissing(labels)
    SetParamURow(p, "labels", labels, juliaOwnedMemory)
  end
  if !ismissing(test)
    SetParamMat(p, "test", test, points_are_rows, juliaOwnedMemory)
  end
  if !ismissing(tolerance)
    SetParam(p, "tolerance", convert(Float64, tolerance))
  end
  if !ismissing(training)
    SetParamMat(p, "training", training, points_are_rows, juliaOwnedMemory)
  end
  if !ismissing(weak_learner)
    SetParam(p, "weak_learner", convert(String, weak_learner))
  end
  if verbose !== nothing && verbose === true
    EnableVerbose()
  else
    DisableVerbose()
  end

  SetPassed(p, "output")
  SetPassed(p, "output_model")
  SetPassed(p, "predictions")
  SetPassed(p, "probabilities")
  # Call the program.
  call_adaboost(p, t)

  results = (GetParamURow(p, "output", juliaOwnedMemory),
             adaboost_internal.GetParamAdaBoostModel(p, "output_model", modelPtrs),
             GetParamURow(p, "predictions", juliaOwnedMemory),
             GetParamMat(p, "probabilities", points_are_rows, juliaOwnedMemory))

  # We are responsible for cleaning up the `p` and `t` objects.
  DeleteParameters(p)
  DeleteTimers(t)

  return results
end
