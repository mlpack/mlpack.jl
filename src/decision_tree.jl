export decision_tree

import ..DecisionTreeModel

using mlpack._Internal.io

import mlpack_jll
const decision_treeLibrary = mlpack_jll.libmlpack_julia_decision_tree

# Call the C binding of the mlpack decision_tree binding.
function decision_tree_mlpackMain()
  success = ccall((:decision_tree, decision_treeLibrary), Bool, ())
  if !success
    # Throw an exception---false means there was a C++ exception.
    throw(ErrorException("mlpack binding error; see output"))
  end
end

" Internal module to hold utility functions. "
module decision_tree_internal
  import ..decision_treeLibrary

import ...DecisionTreeModel

# Get the value of a model pointer parameter of type DecisionTreeModel.
function IOGetParamDecisionTreeModel(paramName::String)::DecisionTreeModel
  DecisionTreeModel(ccall((:IO_GetParamDecisionTreeModelPtr, decision_treeLibrary), Ptr{Nothing}, (Cstring,), paramName))
end

# Set the value of a model pointer parameter of type DecisionTreeModel.
function IOSetParamDecisionTreeModel(paramName::String, model::DecisionTreeModel)
  ccall((:IO_SetParamDecisionTreeModelPtr, decision_treeLibrary), Nothing, (Cstring, Ptr{Nothing}), paramName, model.ptr)
end

# Serialize a model to the given stream.
function serializeDecisionTreeModel(stream::IO, model::DecisionTreeModel)
  buf_len = UInt[0]
  buf_ptr = ccall((:SerializeDecisionTreeModelPtr, decision_treeLibrary), Ptr{UInt8}, (Ptr{Nothing}, Ptr{UInt}), model.ptr, Base.pointer(buf_len))
  buf = Base.unsafe_wrap(Vector{UInt8}, buf_ptr, buf_len[1]; own=true)
  write(stream, buf)
end
# Deserialize a model from the given stream.
function deserializeDecisionTreeModel(stream::IO)::DecisionTreeModel
  buffer = read(stream)
  DecisionTreeModel(ccall((:DeserializeDecisionTreeModelPtr, decision_treeLibrary), Ptr{Nothing}, (Ptr{UInt8}, UInt), Base.pointer(buffer), length(buffer)))
end
end # module

"""
    decision_tree(; [input_model, labels, maximum_depth, minimum_gain_split, minimum_leaf_size, print_training_accuracy, print_training_error, test, test_labels, training, verbose, weights])

Train and evaluate using a decision tree.  Given a dataset containing numeric or
categorical features, and associated labels for each point in the dataset, this
program can train a decision tree on that data.

The training set and associated labels are specified with the `training` and
`labels` parameters, respectively.  The labels should be in the range [0,
num_classes - 1]. Optionally, if `labels` is not specified, the labels are
assumed to be the last dimension of the training dataset.

When a model is trained, the `output_model` output parameter may be used to save
the trained model.  A model may be loaded for predictions with the `input_model`
parameter.  The `input_model` parameter may not be specified when the `training`
parameter is specified.  The `minimum_leaf_size` parameter specifies the minimum
number of training points that must fall into each leaf for it to be split.  The
`minimum_gain_split` parameter specifies the minimum gain that is needed for the
node to split.  The `maximum_depth` parameter specifies the maximum depth of the
tree.  If `print_training_error` is specified, the training error will be
printed.

Test data may be specified with the `test` parameter, and if performance numbers
are desired for that test set, labels may be specified with the `test_labels`
parameter.  Predictions for each test point may be saved via the `predictions`
output parameter.  Class probabilities for each prediction may be saved with the
`probabilities` output parameter.

For example, to train a decision tree with a minimum leaf size of 20 on the
dataset contained in `data` with labels `labels`, saving the output model to
`tree` and printing the training error, one could call

```julia
julia> using CSV
julia> labels = CSV.read("labels.csv"; type=Int)
julia> tree, _, _ = decision_tree(labels=labels,
            minimum_gain_split=0.001, minimum_leaf_size=20,
            print_training_accuracy=1, training=data)
```

Then, to use that model to classify points in `test_set` and print the test
error given the labels `test_labels` using that model, while saving the
predictions for each point to `predictions`, one could call 

```julia
julia> using CSV
julia> test_labels = CSV.read("test_labels.csv"; type=Int)
julia> _, predictions, _ = decision_tree(input_model=tree,
            test=test_set, test_labels=test_labels)
```

# Arguments

 - `input_model::DecisionTreeModel`: Pre-trained decision tree, to be used
      with test points.
 - `labels::Array{Int, 1}`: Training labels.
 - `maximum_depth::Int`: Maximum depth of the tree (0 means no limit). 
      Default value `0`.
      
 - `minimum_gain_split::Float64`: Minimum gain for node splitting. 
      Default value `1e-07`.
      
 - `minimum_leaf_size::Int`: Minimum number of points in a leaf.  Default
      value `20`.
      
 - `print_training_accuracy::Bool`: Print the training accuracy.  Default
      value `false`.
      
 - `print_training_error::Bool`: Print the training error (deprecated;
      will be removed in mlpack 4.0.0).  Default value `false`.
      
 - `test::Tuple{Array{Bool, 1}, Array{Float64, 2}}`: Testing dataset (may
      be categorical).
 - `test_labels::Array{Int, 1}`: Test point labels, if accuracy
      calculation is desired.
 - `training::Tuple{Array{Bool, 1}, Array{Float64, 2}}`: Training dataset
      (may be categorical).
 - `verbose::Bool`: Display informational messages and the full list of
      parameters and timers at the end of execution.  Default value `false`.
      
 - `weights::Array{Float64, 2}`: The weight of labels

# Return values

 - `output_model::DecisionTreeModel`: Output for trained decision tree.
 - `predictions::Array{Int, 1}`: Class predictions for each test point.
 - `probabilities::Array{Float64, 2}`: Class probabilities for each test
      point.

"""
function decision_tree(;
                       input_model::Union{DecisionTreeModel, Missing} = missing,
                       labels = missing,
                       maximum_depth::Union{Int, Missing} = missing,
                       minimum_gain_split::Union{Float64, Missing} = missing,
                       minimum_leaf_size::Union{Int, Missing} = missing,
                       print_training_accuracy::Union{Bool, Missing} = missing,
                       print_training_error::Union{Bool, Missing} = missing,
                       test::Union{Tuple{Array{Bool, 1}, Array{Float64, 2}}, Missing} = missing,
                       test_labels = missing,
                       training::Union{Tuple{Array{Bool, 1}, Array{Float64, 2}}, Missing} = missing,
                       verbose::Union{Bool, Missing} = missing,
                       weights = missing,
                       points_are_rows::Bool = true)
  # Force the symbols to load.
  ccall((:loadSymbols, decision_treeLibrary), Nothing, ());

  IORestoreSettings("Decision tree")

  # Process each input argument before calling mlpackMain().
  if !ismissing(input_model)
    decision_tree_internal.IOSetParamDecisionTreeModel("input_model", convert(DecisionTreeModel, input_model))
  end
  if !ismissing(labels)
    IOSetParamURow("labels", labels)
  end
  if !ismissing(maximum_depth)
    IOSetParam("maximum_depth", convert(Int, maximum_depth))
  end
  if !ismissing(minimum_gain_split)
    IOSetParam("minimum_gain_split", convert(Float64, minimum_gain_split))
  end
  if !ismissing(minimum_leaf_size)
    IOSetParam("minimum_leaf_size", convert(Int, minimum_leaf_size))
  end
  if !ismissing(print_training_accuracy)
    IOSetParam("print_training_accuracy", convert(Bool, print_training_accuracy))
  end
  if !ismissing(print_training_error)
    IOSetParam("print_training_error", convert(Bool, print_training_error))
  end
  if !ismissing(test)
    IOSetParam("test", convert(Tuple{Array{Bool, 1}, Array{Float64, 2}}, test), points_are_rows)
  end
  if !ismissing(test_labels)
    IOSetParamURow("test_labels", test_labels)
  end
  if !ismissing(training)
    IOSetParam("training", convert(Tuple{Array{Bool, 1}, Array{Float64, 2}}, training), points_are_rows)
  end
  if !ismissing(weights)
    IOSetParamMat("weights", weights, points_are_rows)
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
  decision_tree_mlpackMain()

  return decision_tree_internal.IOGetParamDecisionTreeModel("output_model"),
         IOGetParamURow("predictions"),
         IOGetParamMat("probabilities", points_are_rows)
end
