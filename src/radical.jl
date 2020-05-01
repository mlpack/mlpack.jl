export radical


using mlpack._Internal.cli

import mlpack_jll
const radicalLibrary = mlpack_jll.libmlpack_julia_radical

# Call the C binding of the mlpack radical binding.
function radical_mlpackMain()
  success = ccall((:radical, radicalLibrary), Bool, ())
  if !success
    # Throw an exception---false means there was a C++ exception.
    throw(ErrorException("mlpack binding error; see output"))
  end
end

" Internal module to hold utility functions. "
module radical_internal
  import ..radicalLibrary

end # module

"""
    radical(input; [angles, noise_std_dev, objective, replicates, seed, sweeps, verbose])

An implementation of RADICAL, a method for independent component analysis (ICA).
 Assuming that we have an input matrix X, the goal is to find a square unmixing
matrix W such that Y = W * X and the dimensions of Y are independent components.
 If the algorithm is running particularly slowly, try reducing the number of
replicates.

The input matrix to perform ICA on should be specified with the `input`
parameter.  The output matrix Y may be saved with the `output_ic` output
parameter, and the output unmixing matrix W may be saved with the
`output_unmixing` output parameter.

For example, to perform ICA on the matrix `X` with 40 replicates, saving the
independent components to `ic`, the following command may be used: 

```julia
julia> using CSV
julia> X = CSV.read("X.csv")
julia> ic, _ = radical(X; replicates=40)
```

# Arguments

 - `input::Array{Float64, 2}`: Input dataset for ICA.
 - `angles::Int`: Number of angles to consider in brute-force search
      during Radical2D.  Default value `150`.
      
 - `noise_std_dev::Float64`: Standard deviation of Gaussian noise. 
      Default value `0.175`.
      
 - `objective::Bool`: If set, an estimate of the final objective function
      is printed.  Default value `false`.
      
 - `replicates::Int`: Number of Gaussian-perturbed replicates to use (per
      point) in Radical2D.  Default value `30`.
      
 - `seed::Int`: Random seed.  If 0, 'std::time(NULL)' is used.  Default
      value `0`.
      
 - `sweeps::Int`: Number of sweeps; each sweep calls Radical2D once for
      each pair of dimensions.  Default value `0`.
      
 - `verbose::Bool`: Display informational messages and the full list of
      parameters and timers at the end of execution.  Default value `false`.
      

# Return values

 - `output_ic::Array{Float64, 2}`: Matrix to save independent components
      to.
 - `output_unmixing::Array{Float64, 2}`: Matrix to save unmixing matrix
      to.

"""
function radical(input;
                 angles::Union{Int, Missing} = missing,
                 noise_std_dev::Union{Float64, Missing} = missing,
                 objective::Union{Bool, Missing} = missing,
                 replicates::Union{Int, Missing} = missing,
                 seed::Union{Int, Missing} = missing,
                 sweeps::Union{Int, Missing} = missing,
                 verbose::Union{Bool, Missing} = missing,
                 points_are_rows::Bool = true)
  # Force the symbols to load.
  ccall((:loadSymbols, radicalLibrary), Nothing, ());

  CLIRestoreSettings("RADICAL")

  # Process each input argument before calling mlpackMain().
  CLISetParamMat("input", input, points_are_rows)
  if !ismissing(angles)
    CLISetParam("angles", convert(Int, angles))
  end
  if !ismissing(noise_std_dev)
    CLISetParam("noise_std_dev", convert(Float64, noise_std_dev))
  end
  if !ismissing(objective)
    CLISetParam("objective", convert(Bool, objective))
  end
  if !ismissing(replicates)
    CLISetParam("replicates", convert(Int, replicates))
  end
  if !ismissing(seed)
    CLISetParam("seed", convert(Int, seed))
  end
  if !ismissing(sweeps)
    CLISetParam("sweeps", convert(Int, sweeps))
  end
  if verbose !== nothing && verbose === true
    CLIEnableVerbose()
  else
    CLIDisableVerbose()
  end

  CLISetPassed("output_ic")
  CLISetPassed("output_unmixing")
  # Call the program.
  radical_mlpackMain()

  return CLIGetParamMat("output_ic", points_are_rows),
         CLIGetParamMat("output_unmixing", points_are_rows)
end
