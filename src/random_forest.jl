export random_forest

import ..RandomForestModel

using mlpack._Internal.params

import mlpack_jll
const random_forestLibrary = mlpack_jll.libmlpack_julia_random_forest

# Call the C binding of the mlpack random_forest binding.
function call_random_forest(p, t)
  success = ccall((:mlpack_random_forest, random_forestLibrary), Bool, (Ptr{Nothing}, Ptr{Nothing}), p, t)
  if !success
    # Throw an exception---false means there was a C++ exception.
    throw(ErrorException("mlpack binding error; see output"))
  end
end

" Internal module to hold utility functions. "
module random_forest_internal
  import ..random_forestLibrary

import ...RandomForestModel

# Get the value of a model pointer parameter of type RandomForestModel.
function GetParamRandomForestModel(params::Ptr{Nothing}, paramName::String, modelPtrs::Set{Ptr{Nothing}})::RandomForestModel
  ptr = ccall((:GetParamRandomForestModelPtr, random_forestLibrary), Ptr{Nothing}, (Ptr{Nothing}, Cstring,), params, paramName)
  return RandomForestModel(ptr; finalize=!(ptr in modelPtrs))
end

# Set the value of a model pointer parameter of type RandomForestModel.
function SetParamRandomForestModel(params::Ptr{Nothing}, paramName::String, model::RandomForestModel)
  ccall((:SetParamRandomForestModelPtr, random_forestLibrary), Nothing, (Ptr{Nothing}, Cstring, Ptr{Nothing}), params, paramName, model.ptr)
end

# Delete an instantiated model pointer.
function DeleteRandomForestModel(ptr::Ptr{Nothing})
  ccall((:DeleteRandomForestModelPtr, random_forestLibrary), Nothing, (Ptr{Nothing},), ptr)
end

# Serialize a model to the given stream.
function serializeRandomForestModel(stream::IO, model::RandomForestModel)
  buf_len = UInt[0]
  buf_ptr = ccall((:SerializeRandomForestModelPtr, random_forestLibrary), Ptr{UInt8}, (Ptr{Nothing}, Ptr{UInt}), model.ptr, pointer(buf_len))
  buf = Base.unsafe_wrap(Vector{UInt8}, buf_ptr, buf_len[1]; own=true)
  write(stream, buf_len[1])
  write(stream, buf)
end
# Deserialize a model from the given stream.
function deserializeRandomForestModel(stream::IO)::RandomForestModel
  buf_len = read(stream, UInt)
  buffer = read(stream, buf_len)
  GC.@preserve buffer RandomForestModel(ccall((:DeserializeRandomForestModelPtr, random_forestLibrary), Ptr{Nothing}, (Ptr{UInt8}, UInt), pointer(buffer), length(buffer)))
end
end # module

"""
    random_forest(; [input_model, labels, maximum_depth, minimum_gain_split, minimum_leaf_size, num_trees, print_training_accuracy, seed, subspace_dim, test, test_labels, training, verbose, warm_start])

This program is an implementation of the standard random forest classification
algorithm by Leo Breiman.  A random forest can be trained and saved for later
use, or a random forest may be loaded and predictions or class probabilities for
points may be generated.

The training set and associated labels are specified with the `training` and
`labels` parameters, respectively.  The labels should be in the range `[0,
num_classes - 1]`. Optionally, if `labels` is not specified, the labels are
assumed to be the last dimension of the training dataset.

When a model is trained, the `output_model` output parameter may be used to save
the trained model.  A model may be loaded for predictions with the
`input_model`parameter. The `input_model` parameter may not be specified when
the `training` parameter is specified.  The `minimum_leaf_size` parameter
specifies the minimum number of training points that must fall into each leaf
for it to be split.  The `num_trees` controls the number of trees in the random
forest.  The `minimum_gain_split` parameter controls the minimum required gain
for a decision tree node to split.  Larger values will force higher-confidence
splits.  The `maximum_depth` parameter specifies the maximum depth of the tree. 
The `subspace_dim` parameter is used to control the number of random dimensions
chosen for an individual node's split.  If `print_training_accuracy` is
specified, the calculated accuracy on the training set will be printed.

Test data may be specified with the `test` parameter, and if performance
measures are desired for that test set, labels for the test points may be
specified with the `test_labels` parameter.  Predictions for each test point may
be saved via the `predictions`output parameter.  Class probabilities for each
prediction may be saved with the `probabilities` output parameter.

For example, to train a random forest with a minimum leaf size of 20 using 10
trees on the dataset contained in `data`with labels `labels`, saving the output
random forest to `rf_model` and printing the training error, one could call

```julia
julia> using CSV
julia> data = CSV.read("data.csv")
julia> labels = CSV.read("labels.csv"; type=Int)
julia> rf_model, _, _ = random_forest(labels=labels,
            minimum_leaf_size=20, num_trees=10, print_training_accuracy=1,
            training=data)
```

Then, to use that model to classify points in `test_set` and print the test
error given the labels `test_labels` using that model, while saving the
predictions for each point to `predictions`, one could call 

```julia
julia> using CSV
julia> test_set = CSV.read("test_set.csv")
julia> test_labels = CSV.read("test_labels.csv"; type=Int)
julia> _, predictions, _ = random_forest(input_model=rf_model,
            test=test_set, test_labels=test_labels)
```

# Arguments

 - `input_model::RandomForestModel`: Pre-trained random forest to use for
      classification.
 - `labels::Array{Int, 1}`: Labels for training dataset.
 - `maximum_depth::Int`: Maximum depth of the tree (0 means no limit). 
      Default value `0`.
      
 - `minimum_gain_split::Float64`: Minimum gain needed to make a split when
      building a tree.  Default value `0`.
      
 - `minimum_leaf_size::Int`: Minimum number of points in each leaf node. 
      Default value `1`.
      
 - `num_trees::Int`: Number of trees in the random forest.  Default value
      `10`.
      
 - `print_training_accuracy::Bool`: If set, then the accuracy of the model
      on the training set will be predicted (verbose must also be specified). 
      Default value `false`.
      
 - `seed::Int`: Random seed.  If 0, 'std::time(NULL)' is used.  Default
      value `0`.
      
 - `subspace_dim::Int`: Dimensionality of random subspace to use for each
      split.  '0' will autoselect the square root of data dimensionality. 
      Default value `0`.
      
 - `test::Array{Float64, 2}`: Test dataset to produce predictions for.
 - `test_labels::Array{Int, 1}`: Test dataset labels, if accuracy
      calculation is desired.
 - `training::Array{Float64, 2}`: Training dataset.
 - `verbose::Bool`: Display informational messages and the full list of
      parameters and timers at the end of execution.  Default value `false`.
      
 - `warm_start::Bool`: If true and passed along with `training` and
      `input_model` then trains more trees on top of existing model.  Default
      value `false`.
      

# Return values

 - `output_model::RandomForestModel`: Model to save trained random forest
      to.
 - `predictions::Array{Int, 1}`: Predicted classes for each point in the
      test set.
 - `probabilities::Array{Float64, 2}`: Predicted class probabilities for
      each point in the test set.

"""
function random_forest(;
                       input_model::Union{RandomForestModel, Missing} = missing,
                       labels = missing,
                       maximum_depth::Union{Int, Missing} = missing,
                       minimum_gain_split::Union{Float64, Missing} = missing,
                       minimum_leaf_size::Union{Int, Missing} = missing,
                       num_trees::Union{Int, Missing} = missing,
                       print_training_accuracy::Union{Bool, Missing} = missing,
                       seed::Union{Int, Missing} = missing,
                       subspace_dim::Union{Int, Missing} = missing,
                       test = missing,
                       test_labels = missing,
                       training = missing,
                       verbose::Union{Bool, Missing} = missing,
                       warm_start::Union{Bool, Missing} = missing,
                       points_are_rows::Bool = true)
  # Force the symbols to load.
  ccall((:loadSymbols, random_forestLibrary), Nothing, ());

  # Create the set of model pointers to avoid setting multiple finalizers.
  modelPtrs = Set{Ptr{Nothing}}()

  p = GetParameters("random_forest")
  t = Timers()

  juliaOwnedMemory = Set{Ptr{Nothing}}()
  # Process each input argument before calling mlpackMain().
  if !ismissing(input_model)
    push!(modelPtrs, convert(RandomForestModel, input_model).ptr)
    random_forest_internal.SetParamRandomForestModel(p, "input_model", convert(RandomForestModel, input_model))
  end
  if !ismissing(labels)
    SetParamURow(p, "labels", labels, juliaOwnedMemory)
  end
  if !ismissing(maximum_depth)
    SetParam(p, "maximum_depth", convert(Int, maximum_depth))
  end
  if !ismissing(minimum_gain_split)
    SetParam(p, "minimum_gain_split", convert(Float64, minimum_gain_split))
  end
  if !ismissing(minimum_leaf_size)
    SetParam(p, "minimum_leaf_size", convert(Int, minimum_leaf_size))
  end
  if !ismissing(num_trees)
    SetParam(p, "num_trees", convert(Int, num_trees))
  end
  if !ismissing(print_training_accuracy)
    SetParam(p, "print_training_accuracy", convert(Bool, print_training_accuracy))
  end
  if !ismissing(seed)
    SetParam(p, "seed", convert(Int, seed))
  end
  if !ismissing(subspace_dim)
    SetParam(p, "subspace_dim", convert(Int, subspace_dim))
  end
  if !ismissing(test)
    SetParamMat(p, "test", test, points_are_rows, false, juliaOwnedMemory)
  end
  if !ismissing(test_labels)
    SetParamURow(p, "test_labels", test_labels, juliaOwnedMemory)
  end
  if !ismissing(training)
    SetParamMat(p, "training", training, points_are_rows, false, juliaOwnedMemory)
  end
  if !ismissing(warm_start)
    SetParam(p, "warm_start", convert(Bool, warm_start))
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
  call_random_forest(p, t)

  results = (random_forest_internal.GetParamRandomForestModel(p, "output_model", modelPtrs),
             GetParamURow(p, "predictions", juliaOwnedMemory),
             GetParamMat(p, "probabilities", points_are_rows, juliaOwnedMemory))

  # We are responsible for cleaning up the `p` and `t` objects.
  DeleteParameters(p)
  DeleteTimers(t)

  return results
end
