export perceptron

import ..PerceptronModel

using mlpack._Internal.params

import mlpack_jll
const perceptronLibrary = mlpack_jll.libmlpack_julia_perceptron

# Call the C binding of the mlpack perceptron binding.
function call_perceptron(p, t)
  success = ccall((:mlpack_perceptron, perceptronLibrary), Bool, (Ptr{Nothing}, Ptr{Nothing}), p, t)
  if !success
    # Throw an exception---false means there was a C++ exception.
    throw(ErrorException("mlpack binding error; see output"))
  end
end

" Internal module to hold utility functions. "
module perceptron_internal
  import ..perceptronLibrary

import ...PerceptronModel

# Get the value of a model pointer parameter of type PerceptronModel.
function GetParamPerceptronModel(params::Ptr{Nothing}, paramName::String, modelPtrs::Set{Ptr{Nothing}})::PerceptronModel
  ptr = ccall((:GetParamPerceptronModelPtr, perceptronLibrary), Ptr{Nothing}, (Ptr{Nothing}, Cstring,), params, paramName)
  return PerceptronModel(ptr; finalize=!(ptr in modelPtrs))
end

# Set the value of a model pointer parameter of type PerceptronModel.
function SetParamPerceptronModel(params::Ptr{Nothing}, paramName::String, model::PerceptronModel)
  ccall((:SetParamPerceptronModelPtr, perceptronLibrary), Nothing, (Ptr{Nothing}, Cstring, Ptr{Nothing}), params, paramName, model.ptr)
end

# Delete an instantiated model pointer.
function DeletePerceptronModel(ptr::Ptr{Nothing})
  ccall((:DeletePerceptronModelPtr, perceptronLibrary), Nothing, (Ptr{Nothing},), ptr)
end

# Serialize a model to the given stream.
function serializePerceptronModel(stream::IO, model::PerceptronModel)
  buf_len = UInt[0]
  buf_ptr = ccall((:SerializePerceptronModelPtr, perceptronLibrary), Ptr{UInt8}, (Ptr{Nothing}, Ptr{UInt}), model.ptr, pointer(buf_len))
  buf = Base.unsafe_wrap(Vector{UInt8}, buf_ptr, buf_len[1]; own=true)
  write(stream, buf_len[1])
  write(stream, buf)
end
# Deserialize a model from the given stream.
function deserializePerceptronModel(stream::IO)::PerceptronModel
  buf_len = read(stream, UInt)
  buffer = read(stream, buf_len)
  GC.@preserve buffer PerceptronModel(ccall((:DeserializePerceptronModelPtr, perceptronLibrary), Ptr{Nothing}, (Ptr{UInt8}, UInt), pointer(buffer), length(buffer)))
end
end # module

"""
    perceptron(; [input_model, labels, max_iterations, test, training, verbose])

This program implements a perceptron, which is a single level neural network.
The perceptron makes its predictions based on a linear predictor function
combining a set of weights with the feature vector.  The perceptron learning
rule is able to converge, given enough iterations (specified using the
`max_iterations` parameter), if the data supplied is linearly separable.  The
perceptron is parameterized by a matrix of weight vectors that denote the
numerical weights of the neural network.

This program allows loading a perceptron from a model (via the `input_model`
parameter) or training a perceptron given training data (via the `training`
parameter), or both those things at once.  In addition, this program allows
classification on a test dataset (via the `test` parameter) and the
classification results on the test set may be saved with the `predictions`
output parameter.  The perceptron model may be saved with the `output_model`
output parameter.

Note: the following parameter is deprecated and will be removed in mlpack 4.0.0:
`output`.
Use `predictions` instead of `output`.

The training data given with the `training` option may have class labels as its
last dimension (so, if the training data is in CSV format, labels should be the
last column).  Alternately, the `labels` parameter may be used to specify a
separate matrix of labels.

All these options make it easy to train a perceptron, and then re-use that
perceptron for later classification.  The invocation below trains a perceptron
on `training_data` with labels `training_labels`, and saves the model to
`perceptron_model`.

```julia
julia> using CSV
julia> training_data = CSV.read("training_data.csv")
julia> training_labels = CSV.read("training_labels.csv"; type=Int)
julia> _, perceptron_model, _ = perceptron(labels=training_labels,
            training=training_data)
```

Then, this model can be re-used for classification on the test data `test_data`.
 The example below does precisely that, saving the predicted classes to
`predictions`.

```julia
julia> using CSV
julia> test_data = CSV.read("test_data.csv")
julia> _, _, predictions = perceptron(input_model=perceptron_model,
            test=test_data)
```

Note that all of the options may be specified at once: predictions may be
calculated right after training a model, and model training can occur even if an
existing perceptron model is passed with the `input_model` parameter.  However,
note that the number of classes and the dimensionality of all data must match. 
So you cannot pass a perceptron model trained on 2 classes and then re-train
with a 4-class dataset.  Similarly, attempting classification on a 3-dimensional
dataset with a perceptron that has been trained on 8 dimensions will cause an
error.

# Arguments

 - `input_model::PerceptronModel`: Input perceptron model.
 - `labels::Array{Int, 1}`: A matrix containing labels for the training
      set.
 - `max_iterations::Int`: The maximum number of iterations the perceptron
      is to be run  Default value `1000`.
      
 - `test::Array{Float64, 2}`: A matrix containing the test set.
 - `training::Array{Float64, 2}`: A matrix containing the training set.
 - `verbose::Bool`: Display informational messages and the full list of
      parameters and timers at the end of execution.  Default value `false`.
      

# Return values

 - `output::Array{Int, 1}`: The matrix in which the predicted labels for
      the test set will be written.
 - `output_model::PerceptronModel`: Output for trained perceptron model.
 - `predictions::Array{Int, 1}`: The matrix in which the predicted labels
      for the test set will be written.

"""
function perceptron(;
                    input_model::Union{PerceptronModel, Missing} = missing,
                    labels = missing,
                    max_iterations::Union{Int, Missing} = missing,
                    test = missing,
                    training = missing,
                    verbose::Union{Bool, Missing} = missing,
                    points_are_rows::Bool = true)
  # Force the symbols to load.
  ccall((:loadSymbols, perceptronLibrary), Nothing, ());

  # Create the set of model pointers to avoid setting multiple finalizers.
  modelPtrs = Set{Ptr{Nothing}}()

  p = GetParameters("perceptron")
  t = Timers()

  juliaOwnedMemory = Set{Ptr{Nothing}}()
  # Process each input argument before calling mlpackMain().
  if !ismissing(input_model)
    push!(modelPtrs, convert(PerceptronModel, input_model).ptr)
    perceptron_internal.SetParamPerceptronModel(p, "input_model", convert(PerceptronModel, input_model))
  end
  if !ismissing(labels)
    SetParamURow(p, "labels", labels, juliaOwnedMemory)
  end
  if !ismissing(max_iterations)
    SetParam(p, "max_iterations", convert(Int, max_iterations))
  end
  if !ismissing(test)
    SetParamMat(p, "test", test, points_are_rows, false, juliaOwnedMemory)
  end
  if !ismissing(training)
    SetParamMat(p, "training", training, points_are_rows, false, juliaOwnedMemory)
  end
  if verbose !== nothing && verbose === true
    EnableVerbose()
  else
    DisableVerbose()
  end

  SetPassed(p, "output")
  SetPassed(p, "output_model")
  SetPassed(p, "predictions")
  # Call the program.
  call_perceptron(p, t)

  results = (GetParamURow(p, "output", juliaOwnedMemory),
             perceptron_internal.GetParamPerceptronModel(p, "output_model", modelPtrs),
             GetParamURow(p, "predictions", juliaOwnedMemory))

  # We are responsible for cleaning up the `p` and `t` objects.
  DeleteParameters(p)
  DeleteTimers(t)

  return results
end
