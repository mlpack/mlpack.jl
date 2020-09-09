export cf

import ..CFModel

using mlpack._Internal.io

import mlpack_jll
const cfLibrary = mlpack_jll.libmlpack_julia_cf

# Call the C binding of the mlpack cf binding.
function cf_mlpackMain()
  success = ccall((:cf, cfLibrary), Bool, ())
  if !success
    # Throw an exception---false means there was a C++ exception.
    throw(ErrorException("mlpack binding error; see output"))
  end
end

" Internal module to hold utility functions. "
module cf_internal
  import ..cfLibrary

import ...CFModel

# Get the value of a model pointer parameter of type CFModel.
function IOGetParamCFModel(paramName::String)::CFModel
  CFModel(ccall((:IO_GetParamCFModelPtr, cfLibrary), Ptr{Nothing}, (Cstring,), paramName))
end

# Set the value of a model pointer parameter of type CFModel.
function IOSetParamCFModel(paramName::String, model::CFModel)
  ccall((:IO_SetParamCFModelPtr, cfLibrary), Nothing, (Cstring, Ptr{Nothing}), paramName, model.ptr)
end

# Serialize a model to the given stream.
function serializeCFModel(stream::IO, model::CFModel)
  buf_len = UInt[0]
  buf_ptr = ccall((:SerializeCFModelPtr, cfLibrary), Ptr{UInt8}, (Ptr{Nothing}, Ptr{UInt}), model.ptr, Base.pointer(buf_len))
  buf = Base.unsafe_wrap(Vector{UInt8}, buf_ptr, buf_len[1]; own=true)
  write(stream, buf)
end
# Deserialize a model from the given stream.
function deserializeCFModel(stream::IO)::CFModel
  buffer = read(stream)
  CFModel(ccall((:DeserializeCFModelPtr, cfLibrary), Ptr{Nothing}, (Ptr{UInt8}, UInt), Base.pointer(buffer), length(buffer)))
end
end # module

"""
    cf(; [algorithm, all_user_recommendations, input_model, interpolation, iteration_only_termination, max_iterations, min_residue, neighbor_search, neighborhood, normalization, query, rank, recommendations, seed, test, training, verbose])

This program performs collaborative filtering (CF) on the given dataset. Given a
list of user, item and preferences (the `training` parameter), the program will
perform a matrix decomposition and then can perform a series of actions related
to collaborative filtering.  Alternately, the program can load an existing saved
CF model with the `input_model` parameter and then use that model to provide
recommendations or predict values.

The input matrix should be a 3-dimensional matrix of ratings, where the first
dimension is the user, the second dimension is the item, and the third dimension
is that user's rating of that item.  Both the users and items should be numeric
indices, not names. The indices are assumed to start from 0.

A set of query users for which recommendations can be generated may be specified
with the `query` parameter; alternately, recommendations may be generated for
every user in the dataset by specifying the `all_user_recommendations`
parameter.  In addition, the number of recommendations per user to generate can
be specified with the `recommendations` parameter, and the number of similar
users (the size of the neighborhood) to be considered when generating
recommendations can be specified with the `neighborhood` parameter.

For performing the matrix decomposition, the following optimization algorithms
can be specified via the `algorithm` parameter: 
 - 'RegSVD' -- Regularized SVD using a SGD optimizer
 - 'NMF' -- Non-negative matrix factorization with alternating least squares
update rules
 - 'BatchSVD' -- SVD batch learning
 - 'SVDIncompleteIncremental' -- SVD incomplete incremental learning
 - 'SVDCompleteIncremental' -- SVD complete incremental learning
 - 'BiasSVD' -- Bias SVD using a SGD optimizer
 - 'SVDPP' -- SVD++ using a SGD optimizer


The following neighbor search algorithms can be specified via the
`neighbor_search` parameter:
 - 'cosine'  -- Cosine Search Algorithm
 - 'euclidean'  -- Euclidean Search Algorithm
 - 'pearson'  -- Pearson Search Algorithm


The following weight interpolation algorithms can be specified via the
`interpolation` parameter:
 - 'average'  -- Average Interpolation Algorithm
 - 'regression'  -- Regression Interpolation Algorithm
 - 'similarity'  -- Similarity Interpolation Algorithm


The following ranking normalization algorithms can be specified via the
`normalization` parameter:
 - 'none'  -- No Normalization
 - 'item_mean'  -- Item Mean Normalization
 - 'overall_mean'  -- Overall Mean Normalization
 - 'user_mean'  -- User Mean Normalization
 - 'z_score'  -- Z-Score Normalization

A trained model may be saved to with the `output_model` output parameter.

To train a CF model on a dataset `training_set` using NMF for decomposition and
saving the trained model to `model`, one could call: 

```julia
julia> using CSV
julia> training_set = CSV.read("training_set.csv")
julia> _, model = cf(algorithm="NMF", training=training_set)
```

Then, to use this model to generate recommendations for the list of users in the
query set `users`, storing 5 recommendations in `recommendations`, one could
call 

```julia
julia> using CSV
julia> users = CSV.read("users.csv"; type=Int)
julia> recommendations, _ = cf(input_model=model, query=users,
            recommendations=5)
```

# Arguments

 - `algorithm::String`: Algorithm used for matrix factorization.  Default
      value `NMF`.
      
 - `all_user_recommendations::Bool`: Generate recommendations for all
      users.  Default value `false`.
      
 - `input_model::CFModel`: Trained CF model to load.
 - `interpolation::String`: Algorithm used for weight interpolation. 
      Default value `average`.
      
 - `iteration_only_termination::Bool`: Terminate only when the maximum
      number of iterations is reached.  Default value `false`.
      
 - `max_iterations::Int`: Maximum number of iterations. If set to zero,
      there is no limit on the number of iterations.  Default value `1000`.
      
 - `min_residue::Float64`: Residue required to terminate the factorization
      (lower values generally mean better fits).  Default value `1e-05`.
      
 - `neighbor_search::String`: Algorithm used for neighbor search.  Default
      value `euclidean`.
      
 - `neighborhood::Int`: Size of the neighborhood of similar users to
      consider for each query user.  Default value `5`.
      
 - `normalization::String`: Normalization performed on the ratings. 
      Default value `none`.
      
 - `query::Array{Int, 2}`: List of query users for which recommendations
      should be generated.
 - `rank::Int`: Rank of decomposed matrices (if 0, a heuristic is used to
      estimate the rank).  Default value `0`.
      
 - `recommendations::Int`: Number of recommendations to generate for each
      query user.  Default value `5`.
      
 - `seed::Int`: Set the random seed (0 uses std::time(NULL)).  Default
      value `0`.
      
 - `test::Array{Float64, 2}`: Test set to calculate RMSE on.
 - `training::Array{Float64, 2}`: Input dataset to perform CF on.
 - `verbose::Bool`: Display informational messages and the full list of
      parameters and timers at the end of execution.  Default value `false`.
      

# Return values

 - `output::Array{Int, 2}`: Matrix that will store output
      recommendations.
 - `output_model::CFModel`: Output for trained CF model.

"""
function cf(;
            algorithm::Union{String, Missing} = missing,
            all_user_recommendations::Union{Bool, Missing} = missing,
            input_model::Union{CFModel, Missing} = missing,
            interpolation::Union{String, Missing} = missing,
            iteration_only_termination::Union{Bool, Missing} = missing,
            max_iterations::Union{Int, Missing} = missing,
            min_residue::Union{Float64, Missing} = missing,
            neighbor_search::Union{String, Missing} = missing,
            neighborhood::Union{Int, Missing} = missing,
            normalization::Union{String, Missing} = missing,
            query = missing,
            rank::Union{Int, Missing} = missing,
            recommendations::Union{Int, Missing} = missing,
            seed::Union{Int, Missing} = missing,
            test = missing,
            training = missing,
            verbose::Union{Bool, Missing} = missing,
            points_are_rows::Bool = true)
  # Force the symbols to load.
  ccall((:loadSymbols, cfLibrary), Nothing, ());

  IORestoreSettings("Collaborative Filtering")

  # Process each input argument before calling mlpackMain().
  if !ismissing(algorithm)
    IOSetParam("algorithm", convert(String, algorithm))
  end
  if !ismissing(all_user_recommendations)
    IOSetParam("all_user_recommendations", convert(Bool, all_user_recommendations))
  end
  if !ismissing(input_model)
    cf_internal.IOSetParamCFModel("input_model", convert(CFModel, input_model))
  end
  if !ismissing(interpolation)
    IOSetParam("interpolation", convert(String, interpolation))
  end
  if !ismissing(iteration_only_termination)
    IOSetParam("iteration_only_termination", convert(Bool, iteration_only_termination))
  end
  if !ismissing(max_iterations)
    IOSetParam("max_iterations", convert(Int, max_iterations))
  end
  if !ismissing(min_residue)
    IOSetParam("min_residue", convert(Float64, min_residue))
  end
  if !ismissing(neighbor_search)
    IOSetParam("neighbor_search", convert(String, neighbor_search))
  end
  if !ismissing(neighborhood)
    IOSetParam("neighborhood", convert(Int, neighborhood))
  end
  if !ismissing(normalization)
    IOSetParam("normalization", convert(String, normalization))
  end
  if !ismissing(query)
    IOSetParamUMat("query", query, points_are_rows)
  end
  if !ismissing(rank)
    IOSetParam("rank", convert(Int, rank))
  end
  if !ismissing(recommendations)
    IOSetParam("recommendations", convert(Int, recommendations))
  end
  if !ismissing(seed)
    IOSetParam("seed", convert(Int, seed))
  end
  if !ismissing(test)
    IOSetParamMat("test", test, points_are_rows)
  end
  if !ismissing(training)
    IOSetParamMat("training", training, points_are_rows)
  end
  if verbose !== nothing && verbose === true
    IOEnableVerbose()
  else
    IODisableVerbose()
  end

  IOSetPassed("output")
  IOSetPassed("output_model")
  # Call the program.
  cf_mlpackMain()

  return IOGetParamUMat("output", points_are_rows),
         cf_internal.IOGetParamCFModel("output_model")
end
