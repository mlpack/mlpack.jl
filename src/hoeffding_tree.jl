export hoeffding_tree

import ..HoeffdingTreeModel

using mlpack._Internal.params

import mlpack_jll
const hoeffding_treeLibrary = mlpack_jll.libmlpack_julia_hoeffding_tree

# Call the C binding of the mlpack hoeffding_tree binding.
function call_hoeffding_tree(p, t)
  success = ccall((:mlpack_hoeffding_tree, hoeffding_treeLibrary), Bool, (Ptr{Nothing}, Ptr{Nothing}), p, t)
  if !success
    # Throw an exception---false means there was a C++ exception.
    throw(ErrorException("mlpack binding error; see output"))
  end
end

" Internal module to hold utility functions. "
module hoeffding_tree_internal
  import ..hoeffding_treeLibrary

import ...HoeffdingTreeModel

# Get the value of a model pointer parameter of type HoeffdingTreeModel.
function GetParamHoeffdingTreeModel(params::Ptr{Nothing}, paramName::String, modelPtrs::Set{Ptr{Nothing}})::HoeffdingTreeModel
  ptr = ccall((:GetParamHoeffdingTreeModelPtr, hoeffding_treeLibrary), Ptr{Nothing}, (Ptr{Nothing}, Cstring,), params, paramName)
  return HoeffdingTreeModel(ptr; finalize=!(ptr in modelPtrs))
end

# Set the value of a model pointer parameter of type HoeffdingTreeModel.
function SetParamHoeffdingTreeModel(params::Ptr{Nothing}, paramName::String, model::HoeffdingTreeModel)
  ccall((:SetParamHoeffdingTreeModelPtr, hoeffding_treeLibrary), Nothing, (Ptr{Nothing}, Cstring, Ptr{Nothing}), params, paramName, model.ptr)
end

# Delete an instantiated model pointer.
function DeleteHoeffdingTreeModel(ptr::Ptr{Nothing})
  ccall((:DeleteHoeffdingTreeModelPtr, hoeffding_treeLibrary), Nothing, (Ptr{Nothing},), ptr)
end

# Serialize a model to the given stream.
function serializeHoeffdingTreeModel(stream::IO, model::HoeffdingTreeModel)
  buf_len = UInt[0]
  buf_ptr = ccall((:SerializeHoeffdingTreeModelPtr, hoeffding_treeLibrary), Ptr{UInt8}, (Ptr{Nothing}, Ptr{UInt}), model.ptr, pointer(buf_len))
  buf = Base.unsafe_wrap(Vector{UInt8}, buf_ptr, buf_len[1]; own=true)
  write(stream, buf_len[1])
  write(stream, buf)
end
# Deserialize a model from the given stream.
function deserializeHoeffdingTreeModel(stream::IO)::HoeffdingTreeModel
  buf_len = read(stream, UInt)
  buffer = read(stream, buf_len)
  GC.@preserve buffer HoeffdingTreeModel(ccall((:DeserializeHoeffdingTreeModelPtr, hoeffding_treeLibrary), Ptr{Nothing}, (Ptr{UInt8}, UInt), pointer(buffer), length(buffer)))
end
end # module

"""
    hoeffding_tree(; [batch_mode, bins, confidence, info_gain, input_model, labels, max_samples, min_samples, numeric_split_strategy, observations_before_binning, passes, test, test_labels, training, verbose])

This program implements Hoeffding trees, a form of streaming decision tree
suited best for large (or streaming) datasets.  This program supports both
categorical and numeric data.  Given an input dataset, this program is able to
train the tree with numerous training options, and save the model to a file. 
The program is also able to use a trained model or a model from file in order to
predict classes for a given test set.

The training file and associated labels are specified with the `training` and
`labels` parameters, respectively. Optionally, if `labels` is not specified, the
labels are assumed to be the last dimension of the training dataset.

The training may be performed in batch mode (like a typical decision tree
algorithm) by specifying the `batch_mode` option, but this may not be the best
option for large datasets.

When a model is trained, it may be saved via the `output_model` output
parameter.  A model may be loaded from file for further training or testing with
the `input_model` parameter.

Test data may be specified with the `test` parameter, and if performance
statistics are desired for that test set, labels may be specified with the
`test_labels` parameter.  Predictions for each test point may be saved with the
`predictions` output parameter, and class probabilities for each prediction may
be saved with the `probabilities` output parameter.

For example, to train a Hoeffding tree with confidence 0.99 with data `dataset`,
saving the trained tree to `tree`, the following command may be used:

```julia
julia> using CSV
julia> dataset = CSV.read("dataset.csv")
julia> tree, _, _ = hoeffding_tree(confidence=0.99,
            training=dataset)
```

Then, this tree may be used to make predictions on the test set `test_set`,
saving the predictions into `predictions` and the class probabilities into
`class_probs` with the following command: 

```julia
julia> using CSV
julia> test_set = CSV.read("test_set.csv")
julia> _, predictions, class_probs =
            hoeffding_tree(input_model=tree, test=test_set)
```

# Arguments

 - `batch_mode::Bool`: If true, samples will be considered in batch
      instead of as a stream.  This generally results in better trees but at the
      cost of memory usage and runtime.  Default value `false`.
      
 - `bins::Int`: If the 'domingos' split strategy is used, this specifies
      the number of bins for each numeric split.  Default value `10`.
      
 - `confidence::Float64`: Confidence before splitting (between 0 and 1). 
      Default value `0.95`.
      
 - `info_gain::Bool`: If set, information gain is used instead of Gini
      impurity for calculating Hoeffding bounds.  Default value `false`.
      
 - `input_model::HoeffdingTreeModel`: Input trained Hoeffding tree model.
 - `labels::Array{Int, 1}`: Labels for training dataset.
 - `max_samples::Int`: Maximum number of samples before splitting. 
      Default value `5000`.
      
 - `min_samples::Int`: Minimum number of samples before splitting. 
      Default value `100`.
      
 - `numeric_split_strategy::String`: The splitting strategy to use for
      numeric features: 'domingos' or 'binary'.  Default value `binary`.
      
 - `observations_before_binning::Int`: If the 'domingos' split strategy is
      used, this specifies the number of samples observed before binning is
      performed.  Default value `100`.
      
 - `passes::Int`: Number of passes to take over the dataset.  Default
      value `1`.
      
 - `test::Tuple{Array{Bool, 1}, Array{Float64, 2}}`: Testing dataset (may
      be categorical).
 - `test_labels::Array{Int, 1}`: Labels of test data.
 - `training::Tuple{Array{Bool, 1}, Array{Float64, 2}}`: Training dataset
      (may be categorical).
 - `verbose::Bool`: Display informational messages and the full list of
      parameters and timers at the end of execution.  Default value `false`.
      

# Return values

 - `output_model::HoeffdingTreeModel`: Output for trained Hoeffding tree
      model.
 - `predictions::Array{Int, 1}`: Matrix to output label predictions for
      test data into.
 - `probabilities::Array{Float64, 2}`: In addition to predicting labels,
      provide rediction probabilities in this matrix.

"""
function hoeffding_tree(;
                        batch_mode::Union{Bool, Missing} = missing,
                        bins::Union{Int, Missing} = missing,
                        confidence::Union{Float64, Missing} = missing,
                        info_gain::Union{Bool, Missing} = missing,
                        input_model::Union{HoeffdingTreeModel, Missing} = missing,
                        labels = missing,
                        max_samples::Union{Int, Missing} = missing,
                        min_samples::Union{Int, Missing} = missing,
                        numeric_split_strategy::Union{String, Missing} = missing,
                        observations_before_binning::Union{Int, Missing} = missing,
                        passes::Union{Int, Missing} = missing,
                        test::Union{Tuple{Array{Bool, 1}, Array{Float64, 2}}, Missing} = missing,
                        test_labels = missing,
                        training::Union{Tuple{Array{Bool, 1}, Array{Float64, 2}}, Missing} = missing,
                        verbose::Union{Bool, Missing} = missing,
                        points_are_rows::Bool = true)
  # Force the symbols to load.
  ccall((:loadSymbols, hoeffding_treeLibrary), Nothing, ());

  # Create the set of model pointers to avoid setting multiple finalizers.
  modelPtrs = Set{Ptr{Nothing}}()

  p = GetParameters("hoeffding_tree")
  t = Timers()

  juliaOwnedMemory = Set{Ptr{Nothing}}()
  # Process each input argument before calling mlpackMain().
  if !ismissing(batch_mode)
    SetParam(p, "batch_mode", convert(Bool, batch_mode))
  end
  if !ismissing(bins)
    SetParam(p, "bins", convert(Int, bins))
  end
  if !ismissing(confidence)
    SetParam(p, "confidence", convert(Float64, confidence))
  end
  if !ismissing(info_gain)
    SetParam(p, "info_gain", convert(Bool, info_gain))
  end
  if !ismissing(input_model)
    push!(modelPtrs, convert(HoeffdingTreeModel, input_model).ptr)
    hoeffding_tree_internal.SetParamHoeffdingTreeModel(p, "input_model", convert(HoeffdingTreeModel, input_model))
  end
  if !ismissing(labels)
    SetParamURow(p, "labels", labels, juliaOwnedMemory)
  end
  if !ismissing(max_samples)
    SetParam(p, "max_samples", convert(Int, max_samples))
  end
  if !ismissing(min_samples)
    SetParam(p, "min_samples", convert(Int, min_samples))
  end
  if !ismissing(numeric_split_strategy)
    SetParam(p, "numeric_split_strategy", convert(String, numeric_split_strategy))
  end
  if !ismissing(observations_before_binning)
    SetParam(p, "observations_before_binning", convert(Int, observations_before_binning))
  end
  if !ismissing(passes)
    SetParam(p, "passes", convert(Int, passes))
  end
  if !ismissing(test)
    SetParam(p, "test", convert(Tuple{Array{Bool, 1}, Array{Float64, 2}}, test), points_are_rows, juliaOwnedMemory)
  end
  if !ismissing(test_labels)
    SetParamURow(p, "test_labels", test_labels, juliaOwnedMemory)
  end
  if !ismissing(training)
    SetParam(p, "training", convert(Tuple{Array{Bool, 1}, Array{Float64, 2}}, training), points_are_rows, juliaOwnedMemory)
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
  call_hoeffding_tree(p, t)

  results = (hoeffding_tree_internal.GetParamHoeffdingTreeModel(p, "output_model", modelPtrs),
             GetParamURow(p, "predictions", juliaOwnedMemory),
             GetParamMat(p, "probabilities", points_are_rows, juliaOwnedMemory))

  # We are responsible for cleaning up the `p` and `t` objects.
  DeleteParameters(p)
  DeleteTimers(t)

  return results
end
