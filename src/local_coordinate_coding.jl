export local_coordinate_coding

using mlpack._Internal.cli

import mlpack_jll
const local_coordinate_codingLibrary = mlpack_jll.libmlpack_julia_local_coordinate_coding

# Call the C binding of the mlpack local_coordinate_coding binding.
function local_coordinate_coding_mlpackMain()
  success = ccall((:local_coordinate_coding, local_coordinate_codingLibrary), Bool, ())
  if !success
    # Throw an exception---false means there was a C++ exception.
    throw(ErrorException("mlpack binding error; see output"))
  end
end

" Internal module to hold utility functions. "
module local_coordinate_coding_internal
  import ..local_coordinate_codingLibrary

" Get the value of a model pointer parameter of type LocalCoordinateCoding."
function CLIGetParamLocalCoordinateCodingPtr(paramName::String)
  return ccall((:CLI_GetParamLocalCoordinateCodingPtr, local_coordinate_codingLibrary), Ptr{Nothing}, (Cstring,), paramName)
end

" Set the value of a model pointer parameter of type LocalCoordinateCoding."
function CLISetParamLocalCoordinateCodingPtr(paramName::String, ptr::Ptr{Nothing})
  ccall((:CLI_SetParamLocalCoordinateCodingPtr, local_coordinate_codingLibrary), Nothing, (Cstring, Ptr{Nothing}), paramName, ptr)
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
parameter is specified with the `lambda` parameter.  For example, to run LCC on
the dataset `data` using 200 atoms and an l1-regularization parameter of 0.1,
saving the dictionary `dictionary` and the codes into `codes`, use

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
 - `input_model::unknown_`: Input LCC model.
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
 - `output_model::unknown_`: Output for trained LCC model.

"""
function local_coordinate_coding(;
                                 atoms::Union{Int, Missing} = missing,
                                 initial_dictionary = missing,
                                 input_model::Union{Ptr{Nothing}, Missing} = missing,
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

  CLIRestoreSettings("Local Coordinate Coding")

  # Process each input argument before calling mlpackMain().
  if !ismissing(atoms)
    CLISetParam("atoms", convert(Int, atoms))
  end
  if !ismissing(initial_dictionary)
    CLISetParamMat("initial_dictionary", initial_dictionary, points_are_rows)
  end
  if !ismissing(input_model)
    local_coordinate_coding_internal.CLISetParamLocalCoordinateCodingPtr("input_model", convert(Ptr{Nothing}, input_model))
  end
  if !ismissing(lambda)
    CLISetParam("lambda", convert(Float64, lambda))
  end
  if !ismissing(max_iterations)
    CLISetParam("max_iterations", convert(Int, max_iterations))
  end
  if !ismissing(normalize)
    CLISetParam("normalize", convert(Bool, normalize))
  end
  if !ismissing(seed)
    CLISetParam("seed", convert(Int, seed))
  end
  if !ismissing(test)
    CLISetParamMat("test", test, points_are_rows)
  end
  if !ismissing(tolerance)
    CLISetParam("tolerance", convert(Float64, tolerance))
  end
  if !ismissing(training)
    CLISetParamMat("training", training, points_are_rows)
  end
  if verbose !== nothing && verbose === true
    CLIEnableVerbose()
  else
    CLIDisableVerbose()
  end

  CLISetPassed("codes")
  CLISetPassed("dictionary")
  CLISetPassed("output_model")
  # Call the program.
  local_coordinate_coding_mlpackMain()

  return CLIGetParamMat("codes", points_are_rows),
         CLIGetParamMat("dictionary", points_are_rows),
         local_coordinate_coding_internal.CLIGetParamLocalCoordinateCodingPtr("output_model")
end
