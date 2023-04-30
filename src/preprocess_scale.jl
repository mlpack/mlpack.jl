export preprocess_scale

import ..ScalingModel

using mlpack._Internal.params

import mlpack_jll
const preprocess_scaleLibrary = mlpack_jll.libmlpack_julia_preprocess_scale

# Call the C binding of the mlpack preprocess_scale binding.
function call_preprocess_scale(p, t)
  success = ccall((:mlpack_preprocess_scale, preprocess_scaleLibrary), Bool, (Ptr{Nothing}, Ptr{Nothing}), p, t)
  if !success
    # Throw an exception---false means there was a C++ exception.
    throw(ErrorException("mlpack binding error; see output"))
  end
end

" Internal module to hold utility functions. "
module preprocess_scale_internal
  import ..preprocess_scaleLibrary

import ...ScalingModel

# Get the value of a model pointer parameter of type ScalingModel.
function GetParamScalingModel(params::Ptr{Nothing}, paramName::String, modelPtrs::Set{Ptr{Nothing}})::ScalingModel
  ptr = ccall((:GetParamScalingModelPtr, preprocess_scaleLibrary), Ptr{Nothing}, (Ptr{Nothing}, Cstring,), params, paramName)
  return ScalingModel(ptr; finalize=!(ptr in modelPtrs))
end

# Set the value of a model pointer parameter of type ScalingModel.
function SetParamScalingModel(params::Ptr{Nothing}, paramName::String, model::ScalingModel)
  ccall((:SetParamScalingModelPtr, preprocess_scaleLibrary), Nothing, (Ptr{Nothing}, Cstring, Ptr{Nothing}), params, paramName, model.ptr)
end

# Delete an instantiated model pointer.
function DeleteScalingModel(ptr::Ptr{Nothing})
  ccall((:DeleteScalingModelPtr, preprocess_scaleLibrary), Nothing, (Ptr{Nothing},), ptr)
end

# Serialize a model to the given stream.
function serializeScalingModel(stream::IO, model::ScalingModel)
  buf_len = UInt[0]
  buf_ptr = ccall((:SerializeScalingModelPtr, preprocess_scaleLibrary), Ptr{UInt8}, (Ptr{Nothing}, Ptr{UInt}), model.ptr, pointer(buf_len))
  buf = Base.unsafe_wrap(Vector{UInt8}, buf_ptr, buf_len[1]; own=true)
  write(stream, buf_len[1])
  write(stream, buf)
end
# Deserialize a model from the given stream.
function deserializeScalingModel(stream::IO)::ScalingModel
  buf_len = read(stream, UInt)
  buffer = read(stream, buf_len)
  GC.@preserve buffer ScalingModel(ccall((:DeserializeScalingModelPtr, preprocess_scaleLibrary), Ptr{Nothing}, (Ptr{UInt8}, UInt), pointer(buffer), length(buffer)))
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
      
 - `input_model::ScalingModel`: Input Scaling model.
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
 - `output_model::ScalingModel`: Output scaling model.

"""
function preprocess_scale(input;
                          epsilon::Union{Float64, Missing} = missing,
                          input_model::Union{ScalingModel, Missing} = missing,
                          inverse_scaling::Union{Bool, Missing} = missing,
                          max_value::Union{Int, Missing} = missing,
                          min_value::Union{Int, Missing} = missing,
                          scaler_method::Union{String, Missing} = missing,
                          seed::Union{Int, Missing} = missing,
                          verbose::Union{Bool, Missing} = missing,
                          points_are_rows::Bool = true)
  # Force the symbols to load.
  ccall((:loadSymbols, preprocess_scaleLibrary), Nothing, ());

  # Create the set of model pointers to avoid setting multiple finalizers.
  modelPtrs = Set{Ptr{Nothing}}()

  p = GetParameters("preprocess_scale")
  t = Timers()

  juliaOwnedMemory = Set{Ptr{Nothing}}()
  # Process each input argument before calling mlpackMain().
  SetParamMat(p, "input", input, points_are_rows, false, juliaOwnedMemory)
  if !ismissing(epsilon)
    SetParam(p, "epsilon", convert(Float64, epsilon))
  end
  if !ismissing(input_model)
    push!(modelPtrs, convert(ScalingModel, input_model).ptr)
    preprocess_scale_internal.SetParamScalingModel(p, "input_model", convert(ScalingModel, input_model))
  end
  if !ismissing(inverse_scaling)
    SetParam(p, "inverse_scaling", convert(Bool, inverse_scaling))
  end
  if !ismissing(max_value)
    SetParam(p, "max_value", convert(Int, max_value))
  end
  if !ismissing(min_value)
    SetParam(p, "min_value", convert(Int, min_value))
  end
  if !ismissing(scaler_method)
    SetParam(p, "scaler_method", convert(String, scaler_method))
  end
  if !ismissing(seed)
    SetParam(p, "seed", convert(Int, seed))
  end
  if verbose !== nothing && verbose === true
    EnableVerbose()
  else
    DisableVerbose()
  end

  SetPassed(p, "output")
  SetPassed(p, "output_model")
  # Call the program.
  call_preprocess_scale(p, t)

  results = (GetParamMat(p, "output", points_are_rows, juliaOwnedMemory),
             preprocess_scale_internal.GetParamScalingModel(p, "output_model", modelPtrs))

  # We are responsible for cleaning up the `p` and `t` objects.
  DeleteParameters(p)
  DeleteTimers(t)

  return results
end
