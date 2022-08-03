export det

import ..DTree

using mlpack._Internal.io

import mlpack_jll
const detLibrary = mlpack_jll.libmlpack_julia_det

# Call the C binding of the mlpack det binding.
function det_mlpackMain()
  success = ccall((:det, detLibrary), Bool, ())
  if !success
    # Throw an exception---false means there was a C++ exception.
    throw(ErrorException("mlpack binding error; see output"))
  end
end

" Internal module to hold utility functions. "
module det_internal
  import ..detLibrary

import ...DTree

# Get the value of a model pointer parameter of type DTree.
function IOGetParamDTree(paramName::String)::DTree
  DTree(ccall((:IO_GetParamDTreePtr, detLibrary), Ptr{Nothing}, (Cstring,), paramName))
end

# Set the value of a model pointer parameter of type DTree.
function IOSetParamDTree(paramName::String, model::DTree)
  ccall((:IO_SetParamDTreePtr, detLibrary), Nothing, (Cstring, Ptr{Nothing}), paramName, model.ptr)
end

# Serialize a model to the given stream.
function serializeDTree(stream::IO, model::DTree)
  buf_len = UInt[0]
  buf_ptr = ccall((:SerializeDTreePtr, detLibrary), Ptr{UInt8}, (Ptr{Nothing}, Ptr{UInt}), model.ptr, pointer(buf_len))
  buf = Base.unsafe_wrap(Vector{UInt8}, buf_ptr, buf_len[1]; own=true)
  write(stream, buf)
end
# Deserialize a model from the given stream.
function deserializeDTree(stream::IO)::DTree
  buffer = read(stream)
  GC.@preserve buffer DTree(ccall((:DeserializeDTreePtr, detLibrary), Ptr{Nothing}, (Ptr{UInt8}, UInt), pointer(buffer), length(buffer)))
end
end # module

"""
    det(; [folds, input_model, max_leaf_size, min_leaf_size, path_format, skip_pruning, test, training, verbose])

This program performs a number of functions related to Density Estimation Trees.
 The optimal Density Estimation Tree (DET) can be trained on a set of data
(specified by `training`) using cross-validation (with number of folds specified
with the `folds` parameter).  This trained density estimation tree may then be
saved with the `output_model` output parameter.

The variable importances (that is, the feature importance values for each
dimension) may be saved with the `vi` output parameter, and the density
estimates for each training point may be saved with the `training_set_estimates`
output parameter.

Enabling path printing for each node outputs the path from the root node to a
leaf for each entry in the test set, or training set (if a test set is not
provided).  Strings like 'LRLRLR' (indicating that traversal went to the left
child, then the right child, then the left child, and so forth) will be output.
If 'lr-id' or 'id-lr' are given as the `path_format` parameter, then the ID
(tag) of every node along the path will be printed after or before the L or R
character indicating the direction of traversal, respectively.

This program also can provide density estimates for a set of test points,
specified in the `test` parameter.  The density estimation tree used for this
task will be the tree that was trained on the given training points, or a tree
given as the parameter `input_model`.  The density estimates for the test points
may be saved using the `test_set_estimates` output parameter.

# Arguments

 - `folds::Int`: The number of folds of cross-validation to perform for
      the estimation (0 is LOOCV)  Default value `10`.
      
 - `input_model::DTree`: Trained density estimation tree to load.
 - `max_leaf_size::Int`: The maximum size of a leaf in the unpruned, fully
      grown DET.  Default value `10`.
      
 - `min_leaf_size::Int`: The minimum size of a leaf in the unpruned, fully
      grown DET.  Default value `5`.
      
 - `path_format::String`: The format of path printing: 'lr', 'id-lr', or
      'lr-id'.  Default value `lr`.
      
 - `skip_pruning::Bool`: Whether to bypass the pruning process and output
      the unpruned tree only.  Default value `false`.
      
 - `test::Array{Float64, 2}`: A set of test points to estimate the density
      of.
 - `training::Array{Float64, 2}`: The data set on which to build a density
      estimation tree.
 - `verbose::Bool`: Display informational messages and the full list of
      parameters and timers at the end of execution.  Default value `false`.
      

# Return values

 - `output_model::DTree`: Output to save trained density estimation tree
      to.
 - `tag_counters_file::String`: The file to output the number of points
      that went to each leaf.  Default value ``.
      
 - `tag_file::String`: The file to output the tags (and possibly paths)
      for each sample in the test set.  Default value ``.
      
 - `test_set_estimates::Array{Float64, 2}`: The output estimates on the
      test set from the final optimally pruned tree.
 - `training_set_estimates::Array{Float64, 2}`: The output density
      estimates on the training set from the final optimally pruned tree.
 - `vi::Array{Float64, 2}`: The output variable importance values for each
      feature.

"""
function det(;
             folds::Union{Int, Missing} = missing,
             input_model::Union{DTree, Missing} = missing,
             max_leaf_size::Union{Int, Missing} = missing,
             min_leaf_size::Union{Int, Missing} = missing,
             path_format::Union{String, Missing} = missing,
             skip_pruning::Union{Bool, Missing} = missing,
             test = missing,
             training = missing,
             verbose::Union{Bool, Missing} = missing,
             points_are_rows::Bool = true)
  # Force the symbols to load.
  ccall((:loadSymbols, detLibrary), Nothing, ());

  IORestoreSettings("Density Estimation With Density Estimation Trees")

  # Process each input argument before calling mlpackMain().
  if !ismissing(folds)
    IOSetParam("folds", convert(Int, folds))
  end
  if !ismissing(input_model)
    det_internal.IOSetParamDTree("input_model", convert(DTree, input_model))
  end
  if !ismissing(max_leaf_size)
    IOSetParam("max_leaf_size", convert(Int, max_leaf_size))
  end
  if !ismissing(min_leaf_size)
    IOSetParam("min_leaf_size", convert(Int, min_leaf_size))
  end
  if !ismissing(path_format)
    IOSetParam("path_format", convert(String, path_format))
  end
  if !ismissing(skip_pruning)
    IOSetParam("skip_pruning", convert(Bool, skip_pruning))
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

  IOSetPassed("output_model")
  IOSetPassed("tag_counters_file")
  IOSetPassed("tag_file")
  IOSetPassed("test_set_estimates")
  IOSetPassed("training_set_estimates")
  IOSetPassed("vi")
  # Call the program.
  det_mlpackMain()

  return det_internal.IOGetParamDTree("output_model"),
         Base.unsafe_string(IOGetParamString("tag_counters_file")),
         Base.unsafe_string(IOGetParamString("tag_file")),
         IOGetParamMat("test_set_estimates", points_are_rows),
         IOGetParamMat("training_set_estimates", points_are_rows),
         IOGetParamMat("vi", points_are_rows)
end
