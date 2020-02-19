export perceptron

using mlpack.util.cli

const perceptronLibrary = joinpath(@__DIR__, "libmlpack_julia_perceptron.so")

# Call the C binding of the mlpack perceptron binding.
function perceptron_mlpackMain()
  success = ccall((:perceptron, perceptronLibrary), Bool, ())
  if !success
    # Throw an exception---false means there was a C++ exception.
    throw(ErrorException("mlpack binding error; see output"))
  end
end

" Internal module to hold utility functions. "
module perceptron_internal
  import ..perceptronLibrary

" Get the value of a model pointer parameter of type PerceptronModel."
function CLIGetParamPerceptronModelPtr(paramName::String)
  return ccall((:CLI_GetParamPerceptronModelPtr, perceptronLibrary), Ptr{Nothing}, (Cstring,), paramName)
end

" Set the value of a model pointer parameter of type PerceptronModel."
function CLISetParamPerceptronModelPtr(paramName::String, ptr::Ptr{Nothing})
  ccall((:CLI_SetParamPerceptronModelPtr, perceptronLibrary), Nothing, (Cstring, Ptr{Nothing}), paramName, ptr)
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

julia> using CSV
julia> training_data = CSV.read("training_data.csv")
julia> training_labels = CSV.read("training_labels.csv"; type=Int64)
julia> _, perceptron_model, _ = perceptron(labels=training_labels,
            training=training_data)

Then, this model can be re-used for classification on the test data `test_data`.
 The example below does precisely that, saving the predicted classes to
`predictions`.

julia> using CSV
julia> test_data = CSV.read("test_data.csv")
julia> _, _, predictions = perceptron(input_model=perceptron_model,
            test=test_data)

Note that all of the options may be specified at once: predictions may be
calculated right after training a model, and model training can occur even if an
existing perceptron model is passed with the `input_model` parameter.  However,
note that the number of classes and the dimensionality of all data must match. 
So you cannot pass a perceptron model trained on 2 classes and then re-train
with a 4-class dataset.  Similarly, attempting classification on a 3-dimensional
dataset with a perceptron that has been trained on 8 dimensions will cause an
error.

# Arguments

 - `input_model::unknown_`: Input perceptron model.
 - `labels::Array{Int64, 1}`: A matrix containing labels for the training
      set.
 - `max_iterations::Int`: The maximum number of iterations the perceptron
      is to be run  Default value `1000`.
      
 - `test::Array{Float64, 2}`: A matrix containing the test set.
 - `training::Array{Float64, 2}`: A matrix containing the training set.
 - `verbose::Bool`: Display informational messages and the full list of
      parameters and timers at the end of execution.  Default value `false`.
      

# Return values

 - `output::Array{Int64, 1}`: The matrix in which the predicted labels for
      the test set will be written.
 - `output_model::unknown_`: Output for trained perceptron model.
 - `predictions::Array{Int64, 1}`: The matrix in which the predicted
      labels for the test set will be written.

"""
function perceptron(;
                    input_model::Union{Ptr{Nothing}, Missing} = missing,
                    labels = missing,
                    max_iterations::Union{Int, Missing} = missing,
                    test = missing,
                    training = missing,
                    verbose::Union{Bool, Missing} = missing,
                    points_are_rows::Bool = true)
  # Force the symbols to load.
  ccall((:loadSymbols, perceptronLibrary), Nothing, ());

  CLIRestoreSettings("Perceptron")

  # Process each input argument before calling mlpackMain().
  if !ismissing(input_model)
    perceptron_internal.CLISetParamPerceptronModelPtr("input_model", convert(Ptr{Nothing}, input_model))
  end
  if !ismissing(labels)
    CLISetParamURow("labels", labels)
  end
  if !ismissing(max_iterations)
    CLISetParam("max_iterations", convert(Int, max_iterations))
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

  CLISetPassed("output")
  CLISetPassed("output_model")
  CLISetPassed("predictions")
  # Call the program.
  perceptron_mlpackMain()

  return CLIGetParamURow("output"),
         perceptron_internal.CLIGetParamPerceptronModelPtr("output_model"),
         CLIGetParamURow("predictions")
end
