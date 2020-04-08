export preprocess_scale

using mlpack._Internal.cli

import mlpack_jll
const preprocess_scaleLibrary = mlpack_jll.libmlpack_julia_preprocess_scale

# Call the C binding of the mlpack preprocess_scale binding.
function preprocess_scale_mlpackMain()
  success = ccall((:preprocess_scale, preprocess_scaleLibrary), Bool, ())
  if !success
    # Throw an exception---false means there was a C++ exception.
    throw(ErrorException("mlpack binding error; see output"))
  end
end

" Internal module to hold utility functions. "
module preprocess_scale_internal
  import ..preprocess_scaleLibrary

" Get the value of a model pointer parameter of type ScalingModel."
function CLIGetParamScalingModelPtr(paramName::String)
  return ccall((:CLI_GetParamScalingModelPtr, preprocess_scaleLibrary), Ptr{Nothing}, (Cstring,), paramName)
end

" Set the value of a model pointer parameter of type ScalingModel."
function CLISetParamScalingModelPtr(paramName::String, ptr::Ptr{Nothing})
  ccall((:CLI_SetParamScalingModelPtr, preprocess_scaleLibrary), Nothing, (Cstring, Ptr{Nothing}), paramName, ptr)
end

end # module

"""
    preprocess_scale(input; [epsilon, input_model, inverse_scaling, max_value, min_value, scaler_method, seed, verbose])

This utility takes a dataset and performs feature scaling using one of the six
scaler methods namely: 'max_abs_scaler', 'mean_normalization', 'min_max_scaler'
,'standard_scaler', 'pca_whitening' and 'zca_whitening'. The function takes a
matrix as `input` and a scaling method type which you can specify using
`scaler_method` parameter; the default is standard scaler, and outputs a matrix
with scaled feature.

The output scaled feature matrix may be saved with the `output` output
parameters.

The model to scale features can be saved using `output_model` and later can be
loaded back using`input_model`.

So, a simple example where we want to scale the dataset `X` into `X_scaled` with
 standard_scaler as scaler_method, we could run 

```julia
julia> using CSV
julia> X = CSV.read("X.csv")
julia> X_scaled, _ = preprocess_scale(X;
            scaler_method="standard_scaler")
```

A simple example where we want to whiten the dataset `X` into `X_whitened` with 
PCA as whitening_method and use 0.01 as regularization parameter, we could run 

```julia
julia> using CSV
julia> X = CSV.read("X.csv")
julia> X_scaled, _ = preprocess_scale(X; epsilon=0.01,
            scaler_method="pca_whitening")
```

You can also retransform the scaled dataset back using`inverse_scaling`. An
example to rescale : `X_scaled` into `X`using the saved model `input_model` is:

```julia
julia> using CSV
julia> X_scaled = CSV.read("X_scaled.csv")
julia> X, _ = preprocess_scale(X_scaled; input_model=saved,
            inverse_scaling=1)
```

Another simple example where we want to scale the dataset `X` into `X_scaled`
with  min_max_scaler as scaler method, where scaling range is 1 to 3 instead of
default 0 to 1. We could run 

```julia
julia> using CSV
julia> X = CSV.read("X.csv")
julia> X_scaled, _ = preprocess_scale(X; max_value=3, min_value=1,
            scaler_method="min_max_scaler")
```

# Arguments

 - `input::Array{Float64, 2}`: Matrix containing data.
 - `epsilon::Float64`: regularization Parameter for pcawhitening, or
      zcawhitening, should be between -1 to 1.  Default value `1e-06`.
      
 - `input_model::unknown_`: Input Scaling model.
 - `inverse_scaling::Bool`: Inverse Scaling to get original dataset 
      Default value `false`.
      
 - `max_value::Int`: Ending value of range for min_max_scaler.  Default
      value `1`.
      
 - `min_value::Int`: Starting value of range for min_max_scaler.  Default
      value `0`.
      
 - `scaler_method::String`: method to use for scaling, the default is
      standard_scaler.  Default value `standard_scaler`.
      
 - `seed::Int`: Random seed (0 for std::time(NULL)).  Default value `0`.

 - `verbose::Bool`: Display informational messages and the full list of
      parameters and timers at the end of execution.  Default value `false`.
      

# Return values

 - `output::Array{Float64, 2}`: Matrix to save scaled data to.
 - `output_model::unknown_`: Output scaling model.

"""
function preprocess_scale(input;
                          epsilon::Union{Float64, Missing} = missing,
                          input_model::Union{Ptr{Nothing}, Missing} = missing,
                          inverse_scaling::Union{Bool, Missing} = missing,
                          max_value::Union{Int, Missing} = missing,
                          min_value::Union{Int, Missing} = missing,
                          scaler_method::Union{String, Missing} = missing,
                          seed::Union{Int, Missing} = missing,
                          verbose::Union{Bool, Missing} = missing,
                          points_are_rows::Bool = true)
  # Force the symbols to load.
  ccall((:loadSymbols, preprocess_scaleLibrary), Nothing, ());

  CLIRestoreSettings("Scale Data")

  # Process each input argument before calling mlpackMain().
  CLISetParamMat("input", input, points_are_rows)
  if !ismissing(epsilon)
    CLISetParam("epsilon", convert(Float64, epsilon))
  end
  if !ismissing(input_model)
    preprocess_scale_internal.CLISetParamScalingModelPtr("input_model", convert(Ptr{Nothing}, input_model))
  end
  if !ismissing(inverse_scaling)
    CLISetParam("inverse_scaling", convert(Bool, inverse_scaling))
  end
  if !ismissing(max_value)
    CLISetParam("max_value", convert(Int, max_value))
  end
  if !ismissing(min_value)
    CLISetParam("min_value", convert(Int, min_value))
  end
  if !ismissing(scaler_method)
    CLISetParam("scaler_method", convert(String, scaler_method))
  end
  if !ismissing(seed)
    CLISetParam("seed", convert(Int, seed))
  end
  if verbose !== nothing && verbose === true
    CLIEnableVerbose()
  else
    CLIDisableVerbose()
  end

  CLISetPassed("output")
  CLISetPassed("output_model")
  # Call the program.
  preprocess_scale_mlpackMain()

  return CLIGetParamMat("output", points_are_rows),
         preprocess_scale_internal.CLIGetParamScalingModelPtr("output_model")
end