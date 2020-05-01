export decision_stump

import ..DSModel

using mlpack._Internal.cli

import mlpack_jll
const decision_stumpLibrary = mlpack_jll.libmlpack_julia_decision_stump

# Call the C binding of the mlpack decision_stump binding.
function decision_stump_mlpackMain()
  success = ccall((:decision_stump, decision_stumpLibrary), Bool, ())
  if !success
    # Throw an exception---false means there was a C++ exception.
    throw(ErrorException("mlpack binding error; see output"))
  end
end

" Internal module to hold utility functions. "
module decision_stump_internal
  import ..decision_stumpLibrary

import ...DSModel

# Get the value of a model pointer parameter of type DSModel.
function CLIGetParamDSModel(paramName::String)::DSModel
  DSModel(ccall((:CLI_GetParamDSModelPtr, decision_stumpLibrary), Ptr{Nothing}, (Cstring,), paramName))
end

# Set the value of a model pointer parameter of type DSModel.
function CLISetParamDSModel(paramName::String, model::DSModel)
  ccall((:CLI_SetParamDSModelPtr, decision_stumpLibrary), Nothing, (Cstring, Ptr{Nothing}), paramName, model.ptr)
end

# Serialize a model to the given stream.
function serializeDSModel(stream::IO, model::DSModel)
  buf_len = UInt[0]
  buf_ptr = ccall((:SerializeDSModelPtr, decision_stumpLibrary), Ptr{UInt8}, (Ptr{Nothing}, Ptr{UInt}), model.ptr, Base.pointer(buf_len))
  buf = Base.unsafe_wrap(Vector{UInt8}, buf_ptr, buf_len[1]; own=true)
  write(stream, buf)
end
# Deserialize a model from the given stream.
function deserializeDSModel(stream::IO)::DSModel
  buffer = read(stream)
  DSModel(ccall((:DeserializeDSModelPtr, decision_stumpLibrary), Ptr{Nothing}, (Ptr{UInt8}, UInt), Base.pointer(buffer), length(buffer)))
end
end # module

"""
    decision_stump(; [bucket_size, input_model, labels, test, training, verbose])

This program implements a decision stump, which is a single-level decision tree.
 The decision stump will split on one dimension of the input data, and will
split into multiple buckets.  The dimension and bins are selected by maximizing
the information gain of the split.  Optionally, the minimum number of training
points in each bin can be specified with the `bucket_size` parameter.

The decision stump is parameterized by a splitting dimension and a vector of
values that denote the splitting values of each bin.

This program enables several applications: a decision tree may be trained or
loaded, and then that decision tree may be used to classify a given set of test
points.  The decision tree may also be saved to a file for later usage.

To train a decision stump, training data should be passed with the `training`
parameter, and their corresponding labels should be passed with the `labels`
option.  Optionally, if `labels` is not specified, the labels are assumed to be
the last dimension of the training dataset.  The `bucket_size` parameter
controls the minimum number of training points in each decision stump bucket.

For classifying a test set, a decision stump may be loaded with the
`input_model` parameter (useful for the situation where a stump has already been
trained), and a test set may be specified with the `test` parameter.  The
predicted labels can be saved with the `predictions` output parameter.

Because decision stumps are trained in batch, retraining does not make sense and
thus it is not possible to pass both `training` and `input_model`; instead,
simply build a new decision stump with the training data.

After training, a decision stump can be saved with the `output_model` output
parameter.  That stump may later be re-used in subsequent calls to this program
(or others).

# Arguments

 - `bucket_size::Int`: The minimum number of training points in each
      decision stump bucket.  Default value `6`.
      
 - `input_model::unknown_`: Decision stump model to load.
 - `labels::Array{Int, 1}`: Labels for the training set. If not specified,
      the labels are assumed to be the last row of the training data.
 - `test::Array{Float64, 2}`: A dataset to calculate predictions for.
 - `training::Array{Float64, 2}`: The dataset to train on.
 - `verbose::Bool`: Display informational messages and the full list of
      parameters and timers at the end of execution.  Default value `false`.
      

# Return values

 - `output_model::unknown_`: Output decision stump model to save.
 - `predictions::Array{Int, 1}`: The output matrix that will hold the
      predicted labels for the test set.

"""
function decision_stump(;
                        bucket_size::Union{Int, Missing} = missing,
                        input_model::Union{DSModel, Missing} = missing,
                        labels = missing,
                        test = missing,
                        training = missing,
                        verbose::Union{Bool, Missing} = missing,
                        points_are_rows::Bool = true)
  # Force the symbols to load.
  ccall((:loadSymbols, decision_stumpLibrary), Nothing, ());

  CLIRestoreSettings("Decision Stump")

  # Process each input argument before calling mlpackMain().
  if !ismissing(bucket_size)
    CLISetParam("bucket_size", convert(Int, bucket_size))
  end
  if !ismissing(input_model)
    decision_stump_internal.CLISetParamDSModel("input_model", convert(DSModel, input_model))
  end
  if !ismissing(labels)
    CLISetParamURow("labels", labels)
  end
  if !ismissing(test)
    CLISetParamMat("test", test, points_are_rows)
  end
  if !ismissing(training)
    CLISetParamMat("training", training, points_are_rows)
  end
  if verbose !== nothing && verbose === true
    CLIEnableVerbose()
  else
    CLIDisableVerbose()
  end

  CLISetPassed("output_model")
  CLISetPassed("predictions")
  # Call the program.
  decision_stump_mlpackMain()

  return decision_stump_internal.CLIGetParamDSModel("output_model"),
         CLIGetParamURow("predictions")
end
