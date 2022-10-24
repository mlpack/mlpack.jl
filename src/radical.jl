export radical


using mlpack._Internal.params

import mlpack_jll
const radicalLibrary = mlpack_jll.libmlpack_julia_radical

# Call the C binding of the mlpack radical binding.
function call_radical(p, t)
  success = ccall((:mlpack_radical, radicalLibrary), Bool, (Ptr{Nothing}, Ptr{Nothing}), p, t)
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

  # Create the set of model pointers to avoid setting multiple finalizers.
  modelPtrs = Set{Ptr{Nothing}}()

  p = GetParameters("radical")
  t = Timers()

  juliaOwnedMemory = Set{Ptr{Nothing}}()
  # Process each input argument before calling mlpackMain().
  SetParamMat(p, "input", input, points_are_rows, juliaOwnedMemory)
  if !ismissing(angles)
    SetParam(p, "angles", convert(Int, angles))
  end
  if !ismissing(noise_std_dev)
    SetParam(p, "noise_std_dev", convert(Float64, noise_std_dev))
  end
  if !ismissing(objective)
    SetParam(p, "objective", convert(Bool, objective))
  end
  if !ismissing(replicates)
    SetParam(p, "replicates", convert(Int, replicates))
  end
  if !ismissing(seed)
    SetParam(p, "seed", convert(Int, seed))
  end
  if !ismissing(sweeps)
    SetParam(p, "sweeps", convert(Int, sweeps))
  end
  if verbose !== nothing && verbose === true
    EnableVerbose()
  else
    DisableVerbose()
  end

  SetPassed(p, "output_ic")
  SetPassed(p, "output_unmixing")
  # Call the program.
  call_radical(p, t)

  results = (GetParamMat(p, "output_ic", points_are_rows, juliaOwnedMemory),
             GetParamMat(p, "output_unmixing", points_are_rows, juliaOwnedMemory))

  # We are responsible for cleaning up the `p` and `t` objects.
  DeleteParameters(p)
  DeleteTimers(t)

  return results
end
