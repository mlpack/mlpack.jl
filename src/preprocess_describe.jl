export preprocess_describe


using mlpack._Internal.io

import mlpack_jll
const preprocess_describeLibrary = mlpack_jll.libmlpack_julia_preprocess_describe

# Call the C binding of the mlpack preprocess_describe binding.
function preprocess_describe_mlpackMain()
  success = ccall((:preprocess_describe, preprocess_describeLibrary), Bool, ())
  if !success
    # Throw an exception---false means there was a C++ exception.
    throw(ErrorException("mlpack binding error; see output"))
  end
end

" Internal module to hold utility functions. "
module preprocess_describe_internal
  import ..preprocess_describeLibrary

end # module

"""
    preprocess_describe(input; [dimension, population, precision, row_major, verbose, width])

This utility takes a dataset and prints out the descriptive statistics of the
data. Descriptive statistics is the discipline of quantitatively describing the
main features of a collection of information, or the quantitative description
itself. The program does not modify the original file, but instead prints out
the statistics to the console. The printed result will look like a table.

Optionally, width and precision of the output can be adjusted by a user using
the `width` and `precision` parameters. A user can also select a specific
dimension to analyze if there are too many dimensions. The `population`
parameter can be specified when the dataset should be considered as a
population.  Otherwise, the dataset will be considered as a sample.

So, a simple example where we want to print out statistical facts about the
dataset `X` using the default settings, we could run 

```julia
julia> using CSV
julia> X = CSV.read("X.csv")
julia> preprocess_describe(X; verbose=1)
```

If we want to customize the width to 10 and precision to 5 and consider the
dataset as a population, we could run

```julia
julia> using CSV
julia> X = CSV.read("X.csv")
julia> preprocess_describe(X; precision=5, verbose=1, width=10)
```

# Arguments

 - `input::Array{Float64, 2}`: Matrix containing data,
 - `dimension::Int`: Dimension of the data. Use this to specify a
      dimension  Default value `0`.
      
 - `population::Bool`: If specified, the program will calculate statistics
      assuming the dataset is the population. By default, the program will
      assume the dataset as a sample.  Default value `false`.
      
 - `precision::Int`: Precision of the output statistics.  Default value
      `4`.
      
 - `row_major::Bool`: If specified, the program will calculate statistics
      across rows, not across columns.  (Remember that in mlpack, a column
      represents a point, so this option is generally not necessary.)  Default
      value `false`.
      
 - `verbose::Bool`: Display informational messages and the full list of
      parameters and timers at the end of execution.  Default value `false`.
      
 - `width::Int`: Width of the output table.  Default value `8`.


# Return values


"""
function preprocess_describe(input;
                             dimension::Union{Int, Missing} = missing,
                             population::Union{Bool, Missing} = missing,
                             precision::Union{Int, Missing} = missing,
                             row_major::Union{Bool, Missing} = missing,
                             verbose::Union{Bool, Missing} = missing,
                             width::Union{Int, Missing} = missing,
                             points_are_rows::Bool = true)
  # Force the symbols to load.
  ccall((:loadSymbols, preprocess_describeLibrary), Nothing, ());

  IORestoreSettings("Descriptive Statistics")

  # Process each input argument before calling mlpackMain().
  IOSetParamMat("input", input, points_are_rows)
  if !ismissing(dimension)
    IOSetParam("dimension", convert(Int, dimension))
  end
  if !ismissing(population)
    IOSetParam("population", convert(Bool, population))
  end
  if !ismissing(precision)
    IOSetParam("precision", convert(Int, precision))
  end
  if !ismissing(row_major)
    IOSetParam("row_major", convert(Bool, row_major))
  end
  if !ismissing(width)
    IOSetParam("width", convert(Int, width))
  end
  if verbose !== nothing && verbose === true
    IOEnableVerbose()
  else
    IODisableVerbose()
  end

  # Call the program.
  preprocess_describe_mlpackMain()

  return 
end
