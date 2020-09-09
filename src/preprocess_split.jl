export preprocess_split


using mlpack._Internal.io

import mlpack_jll
const preprocess_splitLibrary = mlpack_jll.libmlpack_julia_preprocess_split

# Call the C binding of the mlpack preprocess_split binding.
function preprocess_split_mlpackMain()
  success = ccall((:preprocess_split, preprocess_splitLibrary), Bool, ())
  if !success
    # Throw an exception---false means there was a C++ exception.
    throw(ErrorException("mlpack binding error; see output"))
  end
end

" Internal module to hold utility functions. "
module preprocess_split_internal
  import ..preprocess_splitLibrary

end # module

"""
    preprocess_split(input; [input_labels, no_shuffle, seed, test_ratio, verbose])

This utility takes a dataset and optionally labels and splits them into a
training set and a test set. Before the split, the points in the dataset are
randomly reordered. The percentage of the dataset to be used as the test set can
be specified with the `test_ratio` parameter; the default is 0.2 (20%).

The output training and test matrices may be saved with the `training` and
`test` output parameters.

Optionally, labels can be also be split along with the data by specifying the
`input_labels` parameter.  Splitting labels works the same way as splitting the
data. The output training and test labels may be saved with the
`training_labels` and `test_labels` output parameters, respectively.

So, a simple example where we want to split the dataset `X` into `X_train` and
`X_test` with 60% of the data in the training set and 40% of the dataset in the
test set, we could run 

```julia
julia> using CSV
julia> X = CSV.read("X.csv")
julia> X_test, _, X_train, _ = preprocess_split(X; test_ratio=0.4)
```

Also by default the dataset is shuffled and split; you can provide the
`no_shuffle` option to avoid shuffling the data; an example to avoid shuffling
of data is:

```julia
julia> using CSV
julia> X = CSV.read("X.csv")
julia> X_test, _, X_train, _ = preprocess_split(X; no_shuffle=1,
            test_ratio=0.4)
```

If we had a dataset `X` and associated labels `y`, and we wanted to split these
into `X_train`, `y_train`, `X_test`, and `y_test`, with 30% of the data in the
test set, we could run

```julia
julia> using CSV
julia> X = CSV.read("X.csv")
julia> y = CSV.read("y.csv"; type=Int)
julia> X_test, y_test, X_train, y_train = preprocess_split(X;
            input_labels=y, test_ratio=0.3)
```

# Arguments

 - `input::Array{Float64, 2}`: Matrix containing data.
 - `input_labels::Array{Int, 2}`: Matrix containing labels.
 - `no_shuffle::Bool`: Avoid shuffling and splitting the data.  Default
      value `false`.
      
 - `seed::Int`: Random seed (0 for std::time(NULL)).  Default value `0`.

 - `test_ratio::Float64`: Ratio of test set; if not set,the ratio defaults
      to 0.2  Default value `0.2`.
      
 - `verbose::Bool`: Display informational messages and the full list of
      parameters and timers at the end of execution.  Default value `false`.
      

# Return values

 - `test::Array{Float64, 2}`: Matrix to save test data to.
 - `test_labels::Array{Int, 2}`: Matrix to save test labels to.
 - `training::Array{Float64, 2}`: Matrix to save training data to.
 - `training_labels::Array{Int, 2}`: Matrix to save train labels to.

"""
function preprocess_split(input;
                          input_labels = missing,
                          no_shuffle::Union{Bool, Missing} = missing,
                          seed::Union{Int, Missing} = missing,
                          test_ratio::Union{Float64, Missing} = missing,
                          verbose::Union{Bool, Missing} = missing,
                          points_are_rows::Bool = true)
  # Force the symbols to load.
  ccall((:loadSymbols, preprocess_splitLibrary), Nothing, ());

  IORestoreSettings("Split Data")

  # Process each input argument before calling mlpackMain().
  IOSetParamMat("input", input, points_are_rows)
  if !ismissing(input_labels)
    IOSetParamUMat("input_labels", input_labels, points_are_rows)
  end
  if !ismissing(no_shuffle)
    IOSetParam("no_shuffle", convert(Bool, no_shuffle))
  end
  if !ismissing(seed)
    IOSetParam("seed", convert(Int, seed))
  end
  if !ismissing(test_ratio)
    IOSetParam("test_ratio", convert(Float64, test_ratio))
  end
  if verbose !== nothing && verbose === true
    IOEnableVerbose()
  else
    IODisableVerbose()
  end

  IOSetPassed("test")
  IOSetPassed("test_labels")
  IOSetPassed("training")
  IOSetPassed("training_labels")
  # Call the program.
  preprocess_split_mlpackMain()

  return IOGetParamMat("test", points_are_rows),
         IOGetParamUMat("test_labels", points_are_rows),
         IOGetParamMat("training", points_are_rows),
         IOGetParamUMat("training_labels", points_are_rows)
end
