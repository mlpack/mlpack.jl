export nbc

import ..NBCModel

using mlpack._Internal.io

import mlpack_jll
const nbcLibrary = mlpack_jll.libmlpack_julia_nbc

# Call the C binding of the mlpack nbc binding.
function nbc_mlpackMain()
  success = ccall((:nbc, nbcLibrary), Bool, ())
  if !success
    # Throw an exception---false means there was a C++ exception.
    throw(ErrorException("mlpack binding error; see output"))
  end
end

" Internal module to hold utility functions. "
module nbc_internal
  import ..nbcLibrary

import ...NBCModel

# Get the value of a model pointer parameter of type NBCModel.
function IOGetParamNBCModel(paramName::String)::NBCModel
  NBCModel(ccall((:IO_GetParamNBCModelPtr, nbcLibrary), Ptr{Nothing}, (Cstring,), paramName))
end

# Set the value of a model pointer parameter of type NBCModel.
function IOSetParamNBCModel(paramName::String, model::NBCModel)
  ccall((:IO_SetParamNBCModelPtr, nbcLibrary), Nothing, (Cstring, Ptr{Nothing}), paramName, model.ptr)
end

# Serialize a model to the given stream.
function serializeNBCModel(stream::IO, model::NBCModel)
  buf_len = UInt[0]
  buf_ptr = ccall((:SerializeNBCModelPtr, nbcLibrary), Ptr{UInt8}, (Ptr{Nothing}, Ptr{UInt}), model.ptr, Base.pointer(buf_len))
  buf = Base.unsafe_wrap(Vector{UInt8}, buf_ptr, buf_len[1]; own=true)
  write(stream, buf)
end
# Deserialize a model from the given stream.
function deserializeNBCModel(stream::IO)::NBCModel
  buffer = read(stream)
  NBCModel(ccall((:DeserializeNBCModelPtr, nbcLibrary), Ptr{Nothing}, (Ptr{UInt8}, UInt), Base.pointer(buffer), length(buffer)))
end
end # module

"""
    nbc(; [incremental_variance, input_model, labels, test, training, verbose])

This program trains the Naive Bayes classifier on the given labeled training
set, or loads a model from the given model file, and then may use that trained
model to classify the points in a given test set.

The training set is specified with the `training` parameter.  Labels may be
either the last row of the training set, or alternately the `labels` parameter
may be specified to pass a separate matrix of labels.

If training is not desired, a pre-existing model may be loaded with the
`input_model` parameter.



The `incremental_variance` parameter can be used to force the training to use an
incremental algorithm for calculating variance.  This is slower, but can help
avoid loss of precision in some cases.

If classifying a test set is desired, the test set may be specified with the
`test` parameter, and the classifications may be saved with the
`predictions`predictions  parameter.  If saving the trained model is desired,
this may be done with the `output_model` output parameter.

Note: the `output` and `output_probs` parameters are deprecated and will be
removed in mlpack 4.0.0.  Use `predictions` and `probabilities` instead.

For example, to train a Naive Bayes classifier on the dataset `data` with labels
`labels` and save the model to `nbc_model`, the following command may be used:

```julia
julia> using CSV
julia> data = CSV.read("data.csv")
julia> labels = CSV.read("labels.csv"; type=Int)
julia> _, nbc_model, _, _, _ = nbc(labels=labels, training=data)
```

Then, to use `nbc_model` to predict the classes of the dataset `test_set` and
save the predicted classes to `predictions`, the following command may be used:

```julia
julia> using CSV
julia> test_set = CSV.read("test_set.csv")
julia> predictions, _, _, _, _ = nbc(input_model=nbc_model,
            test=test_set)
```

# Arguments

 - `incremental_variance::Bool`: The variance of each class will be
      calculated incrementally.  Default value `false`.
      
 - `input_model::NBCModel`: Input Naive Bayes model.
 - `labels::Array{Int, 1}`: A file containing labels for the training
      set.
 - `test::Array{Float64, 2}`: A matrix containing the test set.
 - `training::Array{Float64, 2}`: A matrix containing the training set.
 - `verbose::Bool`: Display informational messages and the full list of
      parameters and timers at the end of execution.  Default value `false`.
      

# Return values

 - `output::Array{Int, 1}`: The matrix in which the predicted labels for
      the test set will be written (deprecated).
 - `output_model::NBCModel`: File to save trained Naive Bayes model to.
 - `output_probs::Array{Float64, 2}`: The matrix in which the predicted
      probability of labels for the test set will be written (deprecated).
 - `predictions::Array{Int, 1}`: The matrix in which the predicted labels
      for the test set will be written.
 - `probabilities::Array{Float64, 2}`: The matrix in which the predicted
      probability of labels for the test set will be written.

"""
function nbc(;
             incremental_variance::Union{Bool, Missing} = missing,
             input_model::Union{NBCModel, Missing} = missing,
             labels = missing,
             test = missing,
             training = missing,
             verbose::Union{Bool, Missing} = missing,
             points_are_rows::Bool = true)
  # Force the symbols to load.
  ccall((:loadSymbols, nbcLibrary), Nothing, ());

  IORestoreSettings("Parametric Naive Bayes Classifier")

  # Process each input argument before calling mlpackMain().
  if !ismissing(incremental_variance)
    IOSetParam("incremental_variance", convert(Bool, incremental_variance))
  end
  if !ismissing(input_model)
    nbc_internal.IOSetParamNBCModel("input_model", convert(NBCModel, input_model))
  end
  if !ismissing(labels)
    IOSetParamURow("labels", labels)
  end
  if !ismissing(test)
    IOSetParamMat("test", test, points_are_rows)
  end
  if !ismissing(training)
    IOSetParamMat("training", training, points_are_rows)
  end
  if verbose !== nothing && verbose === true
    IOEnableVerbose()
  else
    IODisableVerbose()
  end

  IOSetPassed("output")
  IOSetPassed("output_model")
  IOSetPassed("output_probs")
  IOSetPassed("predictions")
  IOSetPassed("probabilities")
  # Call the program.
  nbc_mlpackMain()

  return IOGetParamURow("output"),
         nbc_internal.IOGetParamNBCModel("output_model"),
         IOGetParamMat("output_probs", points_are_rows),
         IOGetParamURow("predictions"),
         IOGetParamMat("probabilities", points_are_rows)
end
