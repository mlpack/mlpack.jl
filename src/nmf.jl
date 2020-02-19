export nmf

using mlpack.util.cli

import mlpack_jll
const nmfLibrary = mlpack_jll.libmlpack_julia_nmf

# Call the C binding of the mlpack nmf binding.
function nmf_mlpackMain()
  success = ccall((:nmf, nmfLibrary), Bool, ())
  if !success
    # Throw an exception---false means there was a C++ exception.
    throw(ErrorException("mlpack binding error; see output"))
  end
end

" Internal module to hold utility functions. "
module nmf_internal
  import ..nmfLibrary

end # module

"""
    nmf(input, rank; [initial_h, initial_w, max_iterations, min_residue, seed, update_rules, verbose])

This program performs non-negative matrix factorization on the given dataset,
storing the resulting decomposed matrices in the specified files.  For an input
dataset V, NMF decomposes V into two matrices W and H such that 

V = W * H

where all elements in W and H are non-negative.  If V is of size (n x m), then W
will be of size (n x r) and H will be of size (r x m), where r is the rank of
the factorization (specified by the `rank` parameter).

Optionally, the desired update rules for each NMF iteration can be chosen from
the following list:

 - multdist: multiplicative distance-based update rules (Lee and Seung 1999)
 - multdiv: multiplicative divergence-based update rules (Lee and Seung 1999)
 - als: alternating least squares update rules (Paatero and Tapper 1994)

The maximum number of iterations is specified with `max_iterations`, and the
minimum residue required for algorithm termination is specified with the
`min_residue` parameter.

For example, to run NMF on the input matrix `V` using the 'multdist' update
rules with a rank-10 decomposition and storing the decomposed matrices into `W`
and `H`, the following command could be used: 

julia> using CSV
julia> V = CSV.read("V.csv")
julia> H, W = nmf(V, 10; update_rules="multdist")

# Arguments

 - `input::Array{Float64, 2}`: Input dataset to perform NMF on.
 - `rank::Int`: Rank of the factorization.
 - `initial_h::Array{Float64, 2}`: Initial H matrix.
 - `initial_w::Array{Float64, 2}`: Initial W matrix.
 - `max_iterations::Int`: Number of iterations before NMF terminates (0
      runs until convergence.  Default value `10000`.
      
 - `min_residue::Float64`: The minimum root mean square residue allowed
      for each iteration, below which the program terminates.  Default value
      `1e-05`.
      
 - `seed::Int`: Random seed.  If 0, 'std::time(NULL)' is used.  Default
      value `0`.
      
 - `update_rules::String`: Update rules for each iteration; ( multdist |
      multdiv | als ).  Default value `multdist`.
      
 - `verbose::Bool`: Display informational messages and the full list of
      parameters and timers at the end of execution.  Default value `false`.
      

# Return values

 - `h::Array{Float64, 2}`: Matrix to save the calculated H to.
 - `w::Array{Float64, 2}`: Matrix to save the calculated W to.

"""
function nmf(input,
             rank::Int;
             initial_h = missing,
             initial_w = missing,
             max_iterations::Union{Int, Missing} = missing,
             min_residue::Union{Float64, Missing} = missing,
             seed::Union{Int, Missing} = missing,
             update_rules::Union{String, Missing} = missing,
             verbose::Union{Bool, Missing} = missing,
             points_are_rows::Bool = true)
  # Force the symbols to load.
  ccall((:loadSymbols, nmfLibrary), Nothing, ());

  CLIRestoreSettings("Non-negative Matrix Factorization")

  # Process each input argument before calling mlpackMain().
  CLISetParamMat("input", input, points_are_rows)
  CLISetParam("rank", rank)
  if !ismissing(initial_h)
    CLISetParamMat("initial_h", initial_h, points_are_rows)
  end
  if !ismissing(initial_w)
    CLISetParamMat("initial_w", initial_w, points_are_rows)
  end
  if !ismissing(max_iterations)
    CLISetParam("max_iterations", convert(Int, max_iterations))
  end
  if !ismissing(min_residue)
    CLISetParam("min_residue", convert(Float64, min_residue))
  end
  if !ismissing(seed)
    CLISetParam("seed", convert(Int, seed))
  end
  if !ismissing(update_rules)
    CLISetParam("update_rules", convert(String, update_rules))
  end
  if verbose !== nothing && verbose === true
    CLIEnableVerbose()
  else
    CLIDisableVerbose()
  end

  CLISetPassed("h")
  CLISetPassed("w")
  # Call the program.
  nmf_mlpackMain()

  return CLIGetParamMat("h", points_are_rows),
         CLIGetParamMat("w", points_are_rows)
end
