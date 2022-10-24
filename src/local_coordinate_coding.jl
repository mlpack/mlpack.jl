export local_coordinate_coding

import ..LocalCoordinateCoding

using mlpack._Internal.params

import mlpack_jll
const local_coordinate_codingLibrary = mlpack_jll.libmlpack_julia_local_coordinate_coding

# Call the C binding of the mlpack local_coordinate_coding binding.
function call_local_coordinate_coding(p, t)
  success = ccall((:mlpack_local_coordinate_coding, local_coordinate_codingLibrary), Bool, (Ptr{Nothing}, Ptr{Nothing}), p, t)
  if !success
    # Throw an exception---false means there was a C++ exception.
    throw(ErrorException("mlpack binding error; see output"))
  end
end

" Internal module to hold utility functions. "
module local_coordinate_coding_internal
  import ..local_coordinate_codingLibrary

import ...LocalCoordinateCoding

# Get the value of a model pointer parameter of type LocalCoordinateCoding.
function GetParamLocalCoordinateCoding(params::Ptr{Nothing}, paramName::String, modelPtrs::Set{Ptr{Nothing}})::LocalCoordinateCoding
  ptr = ccall((:GetParamLocalCoordinateCodingPtr, local_coordinate_codingLibrary), Ptr{Nothing}, (Ptr{Nothing}, Cstring,), params, paramName)
  return LocalCoordinateCoding(ptr; finalize=!(ptr in modelPtrs))
end

# Set the value of a model pointer parameter of type LocalCoordinateCoding.
function SetParamLocalCoordinateCoding(params::Ptr{Nothing}, paramName::String, model::LocalCoordinateCoding)
  ccall((:SetParamLocalCoordinateCodingPtr, local_coordinate_codingLibrary), Nothing, (Ptr{Nothing}, Cstring, Ptr{Nothing}), params, paramName, model.ptr)
end

# Delete an instantiated model pointer.
function DeleteLocalCoordinateCoding(ptr::Ptr{Nothing})
  ccall((:DeleteLocalCoordinateCodingPtr, local_coordinate_codingLibrary), Nothing, (Ptr{Nothing},), ptr)
end

# Serialize a model to the given stream.
function serializeLocalCoordinateCoding(stream::IO, model::LocalCoordinateCoding)
  buf_len = UInt[0]
  buf_ptr = ccall((:SerializeLocalCoordinateCodingPtr, local_coordinate_codingLibrary), Ptr{UInt8}, (Ptr{Nothing}, Ptr{UInt}), model.ptr, pointer(buf_len))
  buf = Base.unsafe_wrap(Vector{UInt8}, buf_ptr, buf_len[1]; own=true)
  write(stream, buf_len[1])
  write(stream, buf)
end
# Deserialize a model from the given stream.
function deserializeLocalCoordinateCoding(stream::IO)::LocalCoordinateCoding
  buf_len = read(stream, UInt)
  buffer = read(stream, buf_len)
  GC.@preserve buffer LocalCoordinateCoding(ccall((:DeserializeLocalCoordinateCodingPtr, local_coordinate_codingLibrary), Ptr{Nothing}, (Ptr{UInt8}, UInt), pointer(buffer), length(buffer)))
end
end # module

"""
    local_coordinate_coding(; [atoms, initial_dictionary, input_model, lambda, max_iterations, normalize, seed, test, tolerance, training, verbose])

An implementation of Local Coordinate Coding (LCC), which codes data that
approximately lives on a manifold using a variation of l1-norm regularized
sparse coding.  Given a dense data matrix X with n points and d dimensions, LCC
seeks to find a dense dictionary matrix D with k atoms in d dimensions, and a
coding matrix Z with n points in k dimensions.  Because of the regularization
method used, the atoms in D should lie close to the manifold on which the data
points lie.

The original data matrix X can then be reconstructed as D * Z.  Therefore, this
program finds a representation of each point in X as a sparse linear combination
of atoms in the dictionary D.

The coding is found with an algorithm which alternates between a dictionary
step, which updates the dictionary D, and a coding step, which updates the
coding matrix Z.

To run this program, the input matrix X must be specified (with -i), along with
the number of atoms in the dictionary (-k).  An initial dictionary may also be
specified with the `initial_dictionary` parameter.  The l1-norm regularization
parameter is specified with the `lambda` parameter.

For example, to run LCC on the dataset `data` using 200 atoms and an
l1-regularization parameter of 0.1, saving the dictionary `dictionary` and the
codes into `codes`, use

```julia
julia> using CSV
julia> data = CSV.read("data.csv")
julia> codes, dict, _ = local_coordinate_coding(atoms=200,
            lambda=0.1, training=data)
```

The maximum number of iterations may be specified with the `max_iterations`
parameter. Optionally, the input data matrix X can be normalized before coding
with the `normalize` parameter.

An LCC model may be saved using the `output_model` output parameter.  Then, to
encode new points from the dataset `points` with the previously saved model
`lcc_model`, saving the new codes to `new_codes`, the following command can be
used:

```julia
julia> using CSV
julia> points = CSV.read("points.csv")
julia> new_codes, _, _ =
            local_coordinate_coding(input_model=lcc_model, test=points)
```

# Arguments

 - `atoms::Int`: Number of atoms in the dictionary.  Default value `0`.

 - `initial_dictionary::Array{Float64, 2}`: Optional initial dictionary.
 - `input_model::LocalCoordinateCoding`: Input LCC model.
 - `lambda::Float64`: Weighted l1-norm regularization parameter.  Default
      value `0`.
      
 - `max_iterations::Int`: Maximum number of iterations for LCC (0
      indicates no limit).  Default value `0`.
      
 - `normalize::Bool`: If set, the input data matrix will be normalized
      before coding.  Default value `false`.
      
 - `seed::Int`: Random seed.  If 0, 'std::time(NULL)' is used.  Default
      value `0`.
      
 - `test::Array{Float64, 2}`: Test points to encode.
 - `tolerance::Float64`: Tolerance for objective function.  Default value
      `0.01`.
      
 - `training::Array{Float64, 2}`: Matrix of training data (X).
 - `verbose::Bool`: Display informational messages and the full list of
      parameters and timers at the end of execution.  Default value `false`.
      

# Return values

 - `codes::Array{Float64, 2}`: Output codes matrix.
 - `dictionary::Array{Float64, 2}`: Output dictionary matrix.
 - `output_model::LocalCoordinateCoding`: Output for trained LCC model.

"""
function local_coordinate_coding(;
                                 atoms::Union{Int, Missing} = missing,
                                 initial_dictionary = missing,
                                 input_model::Union{LocalCoordinateCoding, Missing} = missing,
                                 lambda::Union{Float64, Missing} = missing,
                                 max_iterations::Union{Int, Missing} = missing,
                                 normalize::Union{Bool, Missing} = missing,
                                 seed::Union{Int, Missing} = missing,
                                 test = missing,
                                 tolerance::Union{Float64, Missing} = missing,
                                 training = missing,
                                 verbose::Union{Bool, Missing} = missing,
                                 points_are_rows::Bool = true)
  # Force the symbols to load.
  ccall((:loadSymbols, local_coordinate_codingLibrary), Nothing, ());

  # Create the set of model pointers to avoid setting multiple finalizers.
  modelPtrs = Set{Ptr{Nothing}}()

  p = GetParameters("local_coordinate_coding")
  t = Timers()

  juliaOwnedMemory = Set{Ptr{Nothing}}()
  # Process each input argument before calling mlpackMain().
  if !ismissing(atoms)
    SetParam(p, "atoms", convert(Int, atoms))
  end
  if !ismissing(initial_dictionary)
    SetParamMat(p, "initial_dictionary", initial_dictionary, points_are_rows, juliaOwnedMemory)
  end
  if !ismissing(input_model)
    push!(modelPtrs, convert(LocalCoordinateCoding, input_model).ptr)
    local_coordinate_coding_internal.SetParamLocalCoordinateCoding(p, "input_model", convert(LocalCoordinateCoding, input_model))
  end
  if !ismissing(lambda)
    SetParam(p, "lambda", convert(Float64, lambda))
  end
  if !ismissing(max_iterations)
    SetParam(p, "max_iterations", convert(Int, max_iterations))
  end
  if !ismissing(normalize)
    SetParam(p, "normalize", convert(Bool, normalize))
  end
  if !ismissing(seed)
    SetParam(p, "seed", convert(Int, seed))
  end
  if !ismissing(test)
    SetParamMat(p, "test", test, points_are_rows, juliaOwnedMemory)
  end
  if !ismissing(tolerance)
    SetParam(p, "tolerance", convert(Float64, tolerance))
  end
  if !ismissing(training)
    SetParamMat(p, "training", training, points_are_rows, juliaOwnedMemory)
  end
  if verbose !== nothing && verbose === true
    EnableVerbose()
  else
    DisableVerbose()
  end

  SetPassed(p, "codes")
  SetPassed(p, "dictionary")
  SetPassed(p, "output_model")
  # Call the program.
  call_local_coordinate_coding(p, t)

  results = (GetParamMat(p, "codes", points_are_rows, juliaOwnedMemory),
             GetParamMat(p, "dictionary", points_are_rows, juliaOwnedMemory),
             local_coordinate_coding_internal.GetParamLocalCoordinateCoding(p, "output_model", modelPtrs))

  # We are responsible for cleaning up the `p` and `t` objects.
  DeleteParameters(p)
  DeleteTimers(t)

  return results
end
