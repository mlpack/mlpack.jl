export sparse_coding

import ..SparseCoding

using mlpack._Internal.params

import mlpack_jll
const sparse_codingLibrary = mlpack_jll.libmlpack_julia_sparse_coding

# Call the C binding of the mlpack sparse_coding binding.
function call_sparse_coding(p, t)
  success = ccall((:mlpack_sparse_coding, sparse_codingLibrary), Bool, (Ptr{Nothing}, Ptr{Nothing}), p, t)
  if !success
    # Throw an exception---false means there was a C++ exception.
    throw(ErrorException("mlpack binding error; see output"))
  end
end

" Internal module to hold utility functions. "
module sparse_coding_internal
  import ..sparse_codingLibrary

import ...SparseCoding

# Get the value of a model pointer parameter of type SparseCoding.
function GetParamSparseCoding(params::Ptr{Nothing}, paramName::String, modelPtrs::Set{Ptr{Nothing}})::SparseCoding
  ptr = ccall((:GetParamSparseCodingPtr, sparse_codingLibrary), Ptr{Nothing}, (Ptr{Nothing}, Cstring,), params, paramName)
  return SparseCoding(ptr; finalize=!(ptr in modelPtrs))
end

# Set the value of a model pointer parameter of type SparseCoding.
function SetParamSparseCoding(params::Ptr{Nothing}, paramName::String, model::SparseCoding)
  ccall((:SetParamSparseCodingPtr, sparse_codingLibrary), Nothing, (Ptr{Nothing}, Cstring, Ptr{Nothing}), params, paramName, model.ptr)
end

# Delete an instantiated model pointer.
function DeleteSparseCoding(ptr::Ptr{Nothing})
  ccall((:DeleteSparseCodingPtr, sparse_codingLibrary), Nothing, (Ptr{Nothing},), ptr)
end

# Serialize a model to the given stream.
function serializeSparseCoding(stream::IO, model::SparseCoding)
  buf_len = UInt[0]
  buf_ptr = ccall((:SerializeSparseCodingPtr, sparse_codingLibrary), Ptr{UInt8}, (Ptr{Nothing}, Ptr{UInt}), model.ptr, pointer(buf_len))
  buf = Base.unsafe_wrap(Vector{UInt8}, buf_ptr, buf_len[1]; own=true)
  write(stream, buf_len[1])
  write(stream, buf)
end
# Deserialize a model from the given stream.
function deserializeSparseCoding(stream::IO)::SparseCoding
  buf_len = read(stream, UInt)
  buffer = read(stream, buf_len)
  GC.@preserve buffer SparseCoding(ccall((:DeserializeSparseCodingPtr, sparse_codingLibrary), Ptr{Nothing}, (Ptr{UInt8}, UInt), pointer(buffer), length(buffer)))
end
end # module

"""
    sparse_coding(; [atoms, initial_dictionary, input_model, lambda1, lambda2, max_iterations, newton_tolerance, normalize, objective_tolerance, seed, test, training, verbose])

An implementation of Sparse Coding with Dictionary Learning, which achieves
sparsity via an l1-norm regularizer on the codes (LASSO) or an (l1+l2)-norm
regularizer on the codes (the Elastic Net).  Given a dense data matrix X with d
dimensions and n points, sparse coding seeks to find a dense dictionary matrix D
with k atoms in d dimensions, and a sparse coding matrix Z with n points in k
dimensions.

The original data matrix X can then be reconstructed as Z * D.  Therefore, this
program finds a representation of each point in X as a sparse linear combination
of atoms in the dictionary D.

The sparse coding is found with an algorithm which alternates between a
dictionary step, which updates the dictionary D, and a sparse coding step, which
updates the sparse coding matrix.

Once a dictionary D is found, the sparse coding model may be used to encode
other matrices, and saved for future usage.

To run this program, either an input matrix or an already-saved sparse coding
model must be specified.  An input matrix may be specified with the `training`
option, along with the number of atoms in the dictionary (specified with the
`atoms` parameter).  It is also possible to specify an initial dictionary for
the optimization, with the `initial_dictionary` parameter.  An input model may
be specified with the `input_model` parameter.

As an example, to build a sparse coding model on the dataset `data` using 200
atoms and an l1-regularization parameter of 0.1, saving the model into `model`,
use 

```julia
julia> using CSV
julia> data = CSV.read("data.csv")
julia> _, _, model = sparse_coding(atoms=200, lambda1=0.1,
            training=data)
```

Then, this model could be used to encode a new matrix, `otherdata`, and save the
output codes to `codes`: 

```julia
julia> using CSV
julia> otherdata = CSV.read("otherdata.csv")
julia> codes, _, _ = sparse_coding(input_model=model,
            test=otherdata)
```

# Arguments

 - `atoms::Int`: Number of atoms in the dictionary.  Default value `15`.

 - `initial_dictionary::Array{Float64, 2}`: Optional initial dictionary
      matrix.
 - `input_model::SparseCoding`: File containing input sparse coding
      model.
 - `lambda1::Float64`: Sparse coding l1-norm regularization parameter. 
      Default value `0`.
      
 - `lambda2::Float64`: Sparse coding l2-norm regularization parameter. 
      Default value `0`.
      
 - `max_iterations::Int`: Maximum number of iterations for sparse coding
      (0 indicates no limit).  Default value `0`.
      
 - `newton_tolerance::Float64`: Tolerance for convergence of Newton
      method.  Default value `1e-06`.
      
 - `normalize::Bool`: If set, the input data matrix will be normalized
      before coding.  Default value `false`.
      
 - `objective_tolerance::Float64`: Tolerance for convergence of the
      objective function.  Default value `0.01`.
      
 - `seed::Int`: Random seed.  If 0, 'std::time(NULL)' is used.  Default
      value `0`.
      
 - `test::Array{Float64, 2}`: Optional matrix to be encoded by trained
      model.
 - `training::Array{Float64, 2}`: Matrix of training data (X).
 - `verbose::Bool`: Display informational messages and the full list of
      parameters and timers at the end of execution.  Default value `false`.
      

# Return values

 - `codes::Array{Float64, 2}`: Matrix to save the output sparse codes of
      the test matrix (--test_file) to.
 - `dictionary::Array{Float64, 2}`: Matrix to save the output dictionary
      to.
 - `output_model::SparseCoding`: File to save trained sparse coding model
      to.

"""
function sparse_coding(;
                       atoms::Union{Int, Missing} = missing,
                       initial_dictionary = missing,
                       input_model::Union{SparseCoding, Missing} = missing,
                       lambda1::Union{Float64, Missing} = missing,
                       lambda2::Union{Float64, Missing} = missing,
                       max_iterations::Union{Int, Missing} = missing,
                       newton_tolerance::Union{Float64, Missing} = missing,
                       normalize::Union{Bool, Missing} = missing,
                       objective_tolerance::Union{Float64, Missing} = missing,
                       seed::Union{Int, Missing} = missing,
                       test = missing,
                       training = missing,
                       verbose::Union{Bool, Missing} = missing,
                       points_are_rows::Bool = true)
  # Force the symbols to load.
  ccall((:loadSymbols, sparse_codingLibrary), Nothing, ());

  # Create the set of model pointers to avoid setting multiple finalizers.
  modelPtrs = Set{Ptr{Nothing}}()

  p = GetParameters("sparse_coding")
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
    push!(modelPtrs, convert(SparseCoding, input_model).ptr)
    sparse_coding_internal.SetParamSparseCoding(p, "input_model", convert(SparseCoding, input_model))
  end
  if !ismissing(lambda1)
    SetParam(p, "lambda1", convert(Float64, lambda1))
  end
  if !ismissing(lambda2)
    SetParam(p, "lambda2", convert(Float64, lambda2))
  end
  if !ismissing(max_iterations)
    SetParam(p, "max_iterations", convert(Int, max_iterations))
  end
  if !ismissing(newton_tolerance)
    SetParam(p, "newton_tolerance", convert(Float64, newton_tolerance))
  end
  if !ismissing(normalize)
    SetParam(p, "normalize", convert(Bool, normalize))
  end
  if !ismissing(objective_tolerance)
    SetParam(p, "objective_tolerance", convert(Float64, objective_tolerance))
  end
  if !ismissing(seed)
    SetParam(p, "seed", convert(Int, seed))
  end
  if !ismissing(test)
    SetParamMat(p, "test", test, points_are_rows, juliaOwnedMemory)
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
  call_sparse_coding(p, t)

  results = (GetParamMat(p, "codes", points_are_rows, juliaOwnedMemory),
             GetParamMat(p, "dictionary", points_are_rows, juliaOwnedMemory),
             sparse_coding_internal.GetParamSparseCoding(p, "output_model", modelPtrs))

  # We are responsible for cleaning up the `p` and `t` objects.
  DeleteParameters(p)
  DeleteTimers(t)

  return results
end
