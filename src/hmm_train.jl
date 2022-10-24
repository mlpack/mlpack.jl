export hmm_train

import ..HMMModel

using mlpack._Internal.params

import mlpack_jll
const hmm_trainLibrary = mlpack_jll.libmlpack_julia_hmm_train

# Call the C binding of the mlpack hmm_train binding.
function call_hmm_train(p, t)
  success = ccall((:mlpack_hmm_train, hmm_trainLibrary), Bool, (Ptr{Nothing}, Ptr{Nothing}), p, t)
  if !success
    # Throw an exception---false means there was a C++ exception.
    throw(ErrorException("mlpack binding error; see output"))
  end
end

" Internal module to hold utility functions. "
module hmm_train_internal
  import ..hmm_trainLibrary

import ...HMMModel

# Get the value of a model pointer parameter of type HMMModel.
function GetParamHMMModel(params::Ptr{Nothing}, paramName::String, modelPtrs::Set{Ptr{Nothing}})::HMMModel
  ptr = ccall((:GetParamHMMModelPtr, hmm_trainLibrary), Ptr{Nothing}, (Ptr{Nothing}, Cstring,), params, paramName)
  return HMMModel(ptr; finalize=!(ptr in modelPtrs))
end

# Set the value of a model pointer parameter of type HMMModel.
function SetParamHMMModel(params::Ptr{Nothing}, paramName::String, model::HMMModel)
  ccall((:SetParamHMMModelPtr, hmm_trainLibrary), Nothing, (Ptr{Nothing}, Cstring, Ptr{Nothing}), params, paramName, model.ptr)
end

# Delete an instantiated model pointer.
function DeleteHMMModel(ptr::Ptr{Nothing})
  ccall((:DeleteHMMModelPtr, hmm_trainLibrary), Nothing, (Ptr{Nothing},), ptr)
end

# Serialize a model to the given stream.
function serializeHMMModel(stream::IO, model::HMMModel)
  buf_len = UInt[0]
  buf_ptr = ccall((:SerializeHMMModelPtr, hmm_trainLibrary), Ptr{UInt8}, (Ptr{Nothing}, Ptr{UInt}), model.ptr, pointer(buf_len))
  buf = Base.unsafe_wrap(Vector{UInt8}, buf_ptr, buf_len[1]; own=true)
  write(stream, buf_len[1])
  write(stream, buf)
end
# Deserialize a model from the given stream.
function deserializeHMMModel(stream::IO)::HMMModel
  buf_len = read(stream, UInt)
  buffer = read(stream, buf_len)
  GC.@preserve buffer HMMModel(ccall((:DeserializeHMMModelPtr, hmm_trainLibrary), Ptr{Nothing}, (Ptr{UInt8}, UInt), pointer(buffer), length(buffer)))
end
end # module

"""
    hmm_train(input_file; [batch, gaussians, input_model, labels_file, seed, states, tolerance, type, verbose])

This program allows a Hidden Markov Model to be trained on labeled or unlabeled
data.  It supports four types of HMMs: Discrete HMMs, Gaussian HMMs, GMM HMMs,
or Diagonal GMM HMMs

Either one input sequence can be specified (with `input_file`), or, a file
containing files in which input sequences can be found (when
`input_file`and`batch` are used together).  In addition, labels can be provided
in the file specified by `labels_file`, and if `batch` is used, the file given
to `labels_file` should contain a list of files of labels corresponding to the
sequences in the file given to `input_file`.

The HMM is trained with the Baum-Welch algorithm if no labels are provided.  The
tolerance of the Baum-Welch algorithm can be set with the `tolerance`option.  By
default, the transition matrix is randomly initialized and the emission
distributions are initialized to fit the extent of the data.

Optionally, a pre-created HMM model can be used as a guess for the transition
matrix and emission probabilities; this is specifiable with `output_model`.

# Arguments

 - `input_file::String`: File containing input observations.
 - `batch::Bool`: If true, input_file (and if passed, labels_file) are
      expected to contain a list of files to use as input observation sequences
      (and label sequences).  Default value `false`.
      
 - `gaussians::Int`: Number of gaussians in each GMM (necessary when type
      is 'gmm').  Default value `0`.
      
 - `input_model::HMMModel`: Pre-existing HMM model to initialize training
      with.
 - `labels_file::String`: Optional file of hidden states, used for labeled
      training.  Default value ``.
      
 - `seed::Int`: Random seed.  If 0, 'std::time(NULL)' is used.  Default
      value `0`.
      
 - `states::Int`: Number of hidden states in HMM (necessary, unless
      model_file is specified).  Default value `0`.
      
 - `tolerance::Float64`: Tolerance of the Baum-Welch algorithm.  Default
      value `1e-05`.
      
 - `type_::String`: Type of HMM: discrete | gaussian | diag_gmm | gmm. 
      Default value `gaussian`.
      
 - `verbose::Bool`: Display informational messages and the full list of
      parameters and timers at the end of execution.  Default value `false`.
      

# Return values

 - `output_model::HMMModel`: Output for trained HMM.

"""
function hmm_train(input_file::String;
                   batch::Union{Bool, Missing} = missing,
                   gaussians::Union{Int, Missing} = missing,
                   input_model::Union{HMMModel, Missing} = missing,
                   labels_file::Union{String, Missing} = missing,
                   seed::Union{Int, Missing} = missing,
                   states::Union{Int, Missing} = missing,
                   tolerance::Union{Float64, Missing} = missing,
                   type_::Union{String, Missing} = missing,
                   verbose::Union{Bool, Missing} = missing,
                   points_are_rows::Bool = true)
  # Force the symbols to load.
  ccall((:loadSymbols, hmm_trainLibrary), Nothing, ());

  # Create the set of model pointers to avoid setting multiple finalizers.
  modelPtrs = Set{Ptr{Nothing}}()

  p = GetParameters("hmm_train")
  t = Timers()

  juliaOwnedMemory = Set{Ptr{Nothing}}()
  # Process each input argument before calling mlpackMain().
  SetParam(p, "input_file", input_file)
  if !ismissing(batch)
    SetParam(p, "batch", convert(Bool, batch))
  end
  if !ismissing(gaussians)
    SetParam(p, "gaussians", convert(Int, gaussians))
  end
  if !ismissing(input_model)
    push!(modelPtrs, convert(HMMModel, input_model).ptr)
    hmm_train_internal.SetParamHMMModel(p, "input_model", convert(HMMModel, input_model))
  end
  if !ismissing(labels_file)
    SetParam(p, "labels_file", convert(String, labels_file))
  end
  if !ismissing(seed)
    SetParam(p, "seed", convert(Int, seed))
  end
  if !ismissing(states)
    SetParam(p, "states", convert(Int, states))
  end
  if !ismissing(tolerance)
    SetParam(p, "tolerance", convert(Float64, tolerance))
  end
  if !ismissing(type_)
    SetParam(p, "type", convert(String, type_))
  end
  if verbose !== nothing && verbose === true
    EnableVerbose()
  else
    DisableVerbose()
  end

  SetPassed(p, "output_model")
  # Call the program.
  call_hmm_train(p, t)

  results = (hmm_train_internal.GetParamHMMModel(p, "output_model", modelPtrs))

  # We are responsible for cleaning up the `p` and `t` objects.
  DeleteParameters(p)
  DeleteTimers(t)

  return results
end
