export hmm_train

import ..HMMModel

using mlpack._Internal.io

import mlpack_jll
const hmm_trainLibrary = mlpack_jll.libmlpack_julia_hmm_train

# Call the C binding of the mlpack hmm_train binding.
function hmm_train_mlpackMain()
  success = ccall((:hmm_train, hmm_trainLibrary), Bool, ())
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
function IOGetParamHMMModel(paramName::String)::HMMModel
  HMMModel(ccall((:IO_GetParamHMMModelPtr, hmm_trainLibrary), Ptr{Nothing}, (Cstring,), paramName))
end

# Set the value of a model pointer parameter of type HMMModel.
function IOSetParamHMMModel(paramName::String, model::HMMModel)
  ccall((:IO_SetParamHMMModelPtr, hmm_trainLibrary), Nothing, (Cstring, Ptr{Nothing}), paramName, model.ptr)
end

# Serialize a model to the given stream.
function serializeHMMModel(stream::IO, model::HMMModel)
  buf_len = UInt[0]
  buf_ptr = ccall((:SerializeHMMModelPtr, hmm_trainLibrary), Ptr{UInt8}, (Ptr{Nothing}, Ptr{UInt}), model.ptr, pointer(buf_len))
  buf = Base.unsafe_wrap(Vector{UInt8}, buf_ptr, buf_len[1]; own=true)
  write(stream, buf)
end
# Deserialize a model from the given stream.
function deserializeHMMModel(stream::IO)::HMMModel
  buffer = read(stream)
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

  IORestoreSettings("Hidden Markov Model (HMM) Training")

  # Process each input argument before calling mlpackMain().
  IOSetParam("input_file", input_file)
  if !ismissing(batch)
    IOSetParam("batch", convert(Bool, batch))
  end
  if !ismissing(gaussians)
    IOSetParam("gaussians", convert(Int, gaussians))
  end
  if !ismissing(input_model)
    hmm_train_internal.IOSetParamHMMModel("input_model", convert(HMMModel, input_model))
  end
  if !ismissing(labels_file)
    IOSetParam("labels_file", convert(String, labels_file))
  end
  if !ismissing(seed)
    IOSetParam("seed", convert(Int, seed))
  end
  if !ismissing(states)
    IOSetParam("states", convert(Int, states))
  end
  if !ismissing(tolerance)
    IOSetParam("tolerance", convert(Float64, tolerance))
  end
  if !ismissing(type_)
    IOSetParam("type", convert(String, type_))
  end
  if verbose !== nothing && verbose === true
    IOEnableVerbose()
  else
    IODisableVerbose()
  end

  IOSetPassed("output_model")
  # Call the program.
  hmm_train_mlpackMain()

  return hmm_train_internal.IOGetParamHMMModel("output_model")
end
