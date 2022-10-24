export kde

import ..KDEModel

using mlpack._Internal.params

import mlpack_jll
const kdeLibrary = mlpack_jll.libmlpack_julia_kde

# Call the C binding of the mlpack kde binding.
function call_kde(p, t)
  success = ccall((:mlpack_kde, kdeLibrary), Bool, (Ptr{Nothing}, Ptr{Nothing}), p, t)
  if !success
    # Throw an exception---false means there was a C++ exception.
    throw(ErrorException("mlpack binding error; see output"))
  end
end

" Internal module to hold utility functions. "
module kde_internal
  import ..kdeLibrary

import ...KDEModel

# Get the value of a model pointer parameter of type KDEModel.
function GetParamKDEModel(params::Ptr{Nothing}, paramName::String, modelPtrs::Set{Ptr{Nothing}})::KDEModel
  ptr = ccall((:GetParamKDEModelPtr, kdeLibrary), Ptr{Nothing}, (Ptr{Nothing}, Cstring,), params, paramName)
  return KDEModel(ptr; finalize=!(ptr in modelPtrs))
end

# Set the value of a model pointer parameter of type KDEModel.
function SetParamKDEModel(params::Ptr{Nothing}, paramName::String, model::KDEModel)
  ccall((:SetParamKDEModelPtr, kdeLibrary), Nothing, (Ptr{Nothing}, Cstring, Ptr{Nothing}), params, paramName, model.ptr)
end

# Delete an instantiated model pointer.
function DeleteKDEModel(ptr::Ptr{Nothing})
  ccall((:DeleteKDEModelPtr, kdeLibrary), Nothing, (Ptr{Nothing},), ptr)
end

# Serialize a model to the given stream.
function serializeKDEModel(stream::IO, model::KDEModel)
  buf_len = UInt[0]
  buf_ptr = ccall((:SerializeKDEModelPtr, kdeLibrary), Ptr{UInt8}, (Ptr{Nothing}, Ptr{UInt}), model.ptr, pointer(buf_len))
  buf = Base.unsafe_wrap(Vector{UInt8}, buf_ptr, buf_len[1]; own=true)
  write(stream, buf_len[1])
  write(stream, buf)
end
# Deserialize a model from the given stream.
function deserializeKDEModel(stream::IO)::KDEModel
  buf_len = read(stream, UInt)
  buffer = read(stream, buf_len)
  GC.@preserve buffer KDEModel(ccall((:DeserializeKDEModelPtr, kdeLibrary), Ptr{Nothing}, (Ptr{UInt8}, UInt), pointer(buffer), length(buffer)))
end
end # module

"""
    kde(; [abs_error, algorithm, bandwidth, initial_sample_size, input_model, kernel, mc_break_coef, mc_entry_coef, mc_probability, monte_carlo, query, reference, rel_error, tree, verbose])

This program performs a Kernel Density Estimation. KDE is a non-parametric way
of estimating probability density function. For each query point the program
will estimate its probability density by applying a kernel function to each
reference point. The computational complexity of this is O(N^2) where there are
N query points and N reference points, but this implementation will typically
see better performance as it uses an approximate dual or single tree algorithm
for acceleration.

Dual or single tree optimization avoids many barely relevant calculations (as
kernel function values decrease with distance), so it is an approximate
computation. You can specify the maximum relative error tolerance for each query
value with `rel_error` as well as the maximum absolute error tolerance with the
parameter `abs_error`. This program runs using an Euclidean metric. Kernel
function can be selected using the `kernel` option. You can also choose what
which type of tree to use for the dual-tree algorithm with `tree`. It is also
possible to select whether to use dual-tree algorithm or single-tree algorithm
using the `algorithm` option.

Monte Carlo estimations can be used to accelerate the KDE estimate when the
Gaussian Kernel is used. This provides a probabilistic guarantee on the the
error of the resulting KDE instead of an absolute guarantee.To enable Monte
Carlo estimations, the `monte_carlo` flag can be used, and success probability
can be set with the `mc_probability` option. It is possible to set the initial
sample size for the Monte Carlo estimation using `initial_sample_size`. This
implementation will only consider a node, as a candidate for the Monte Carlo
estimation, if its number of descendant nodes is bigger than the initial sample
size. This can be controlled using a coefficient that will multiply the initial
sample size and can be set using `mc_entry_coef`. To avoid using the same amount
of computations an exact approach would take, this program recurses the tree
whenever a fraction of the amount of the node's descendant points have already
been computed. This fraction is set using `mc_break_coef`.

For example, the following will run KDE using the data in `ref_data` for
training and the data in `qu_data` as query data. It will apply an Epanechnikov
kernel with a 0.2 bandwidth to each reference point and use a KD-Tree for the
dual-tree optimization. The returned predictions will be within 5% of the real
KDE value for each query point.

```julia
julia> using CSV
julia> ref_data = CSV.read("ref_data.csv")
julia> qu_data = CSV.read("qu_data.csv")
julia> _, out_data = kde(bandwidth=0.2, kernel="epanechnikov",
            query=qu_data, reference=ref_data, rel_error=0.05, tree="kd-tree")
```

the predicted density estimations will be stored in `out_data`.
If no `query` is provided, then KDE will be computed on the `reference` dataset.
It is possible to select either a reference dataset or an input model but not
both at the same time. If an input model is selected and parameter values are
not set (e.g. `bandwidth`) then default parameter values will be used.

In addition to the last program call, it is also possible to activate Monte
Carlo estimations if a Gaussian kernel is used. This can provide faster results,
but the KDE will only have a probabilistic guarantee of meeting the desired
error bound (instead of an absolute guarantee). The following example will run
KDE using a Monte Carlo estimation when possible. The results will be within a
5% of the real KDE value with a 95% probability. Initial sample size for the
Monte Carlo estimation will be 200 points and a node will be a candidate for the
estimation only when it contains 700 (i.e. 3.5*200) points. If a node contains
700 points and 420 (i.e. 0.6*700) have already been sampled, then the algorithm
will recurse instead of keep sampling.

```julia
julia> using CSV
julia> ref_data = CSV.read("ref_data.csv")
julia> qu_data = CSV.read("qu_data.csv")
julia> _, out_data = kde(bandwidth=0.2, initial_sample_size=200,
            kernel="gaussian", mc_break_coef=0.6, mc_entry_coef=3.5,
            mc_probability=0.95, monte_carlo=, query=qu_data,
            reference=ref_data, rel_error=0.05, tree="kd-tree")
```

# Arguments

 - `abs_error::Float64`: Relative error tolerance for the prediction. 
      Default value `0`.
      
 - `algorithm::String`: Algorithm to use for the prediction.('dual-tree',
      'single-tree').  Default value `dual-tree`.
      
 - `bandwidth::Float64`: Bandwidth of the kernel.  Default value `1`.

 - `initial_sample_size::Int`: Initial sample size for Monte Carlo
      estimations.  Default value `100`.
      
 - `input_model::KDEModel`: Contains pre-trained KDE model.
 - `kernel::String`: Kernel to use for the prediction.('gaussian',
      'epanechnikov', 'laplacian', 'spherical', 'triangular').  Default value
      `gaussian`.
      
 - `mc_break_coef::Float64`: Controls what fraction of the amount of
      node's descendants is the limit for the sample size before it recurses. 
      Default value `0.4`.
      
 - `mc_entry_coef::Float64`: Controls how much larger does the amount of
      node descendants has to be compared to the initial sample size in order to
      be a candidate for Monte Carlo estimations.  Default value `3`.
      
 - `mc_probability::Float64`: Probability of the estimation being bounded
      by relative error when using Monte Carlo estimations.  Default value
      `0.95`.
      
 - `monte_carlo::Bool`: Whether to use Monte Carlo estimations when
      possible.  Default value `false`.
      
 - `query::Array{Float64, 2}`: Query dataset to KDE on.
 - `reference::Array{Float64, 2}`: Input reference dataset use for KDE.
 - `rel_error::Float64`: Relative error tolerance for the prediction. 
      Default value `0.05`.
      
 - `tree::String`: Tree to use for the prediction.('kd-tree', 'ball-tree',
      'cover-tree', 'octree', 'r-tree').  Default value `kd-tree`.
      
 - `verbose::Bool`: Display informational messages and the full list of
      parameters and timers at the end of execution.  Default value `false`.
      

# Return values

 - `output_model::KDEModel`: If specified, the KDE model will be saved
      here.
 - `predictions::Array{Float64, 1}`: Vector to store density predictions.

"""
function kde(;
             abs_error::Union{Float64, Missing} = missing,
             algorithm::Union{String, Missing} = missing,
             bandwidth::Union{Float64, Missing} = missing,
             initial_sample_size::Union{Int, Missing} = missing,
             input_model::Union{KDEModel, Missing} = missing,
             kernel::Union{String, Missing} = missing,
             mc_break_coef::Union{Float64, Missing} = missing,
             mc_entry_coef::Union{Float64, Missing} = missing,
             mc_probability::Union{Float64, Missing} = missing,
             monte_carlo::Union{Bool, Missing} = missing,
             query = missing,
             reference = missing,
             rel_error::Union{Float64, Missing} = missing,
             tree::Union{String, Missing} = missing,
             verbose::Union{Bool, Missing} = missing,
             points_are_rows::Bool = true)
  # Force the symbols to load.
  ccall((:loadSymbols, kdeLibrary), Nothing, ());

  # Create the set of model pointers to avoid setting multiple finalizers.
  modelPtrs = Set{Ptr{Nothing}}()

  p = GetParameters("kde")
  t = Timers()

  juliaOwnedMemory = Set{Ptr{Nothing}}()
  # Process each input argument before calling mlpackMain().
  if !ismissing(abs_error)
    SetParam(p, "abs_error", convert(Float64, abs_error))
  end
  if !ismissing(algorithm)
    SetParam(p, "algorithm", convert(String, algorithm))
  end
  if !ismissing(bandwidth)
    SetParam(p, "bandwidth", convert(Float64, bandwidth))
  end
  if !ismissing(initial_sample_size)
    SetParam(p, "initial_sample_size", convert(Int, initial_sample_size))
  end
  if !ismissing(input_model)
    push!(modelPtrs, convert(KDEModel, input_model).ptr)
    kde_internal.SetParamKDEModel(p, "input_model", convert(KDEModel, input_model))
  end
  if !ismissing(kernel)
    SetParam(p, "kernel", convert(String, kernel))
  end
  if !ismissing(mc_break_coef)
    SetParam(p, "mc_break_coef", convert(Float64, mc_break_coef))
  end
  if !ismissing(mc_entry_coef)
    SetParam(p, "mc_entry_coef", convert(Float64, mc_entry_coef))
  end
  if !ismissing(mc_probability)
    SetParam(p, "mc_probability", convert(Float64, mc_probability))
  end
  if !ismissing(monte_carlo)
    SetParam(p, "monte_carlo", convert(Bool, monte_carlo))
  end
  if !ismissing(query)
    SetParamMat(p, "query", query, points_are_rows, juliaOwnedMemory)
  end
  if !ismissing(reference)
    SetParamMat(p, "reference", reference, points_are_rows, juliaOwnedMemory)
  end
  if !ismissing(rel_error)
    SetParam(p, "rel_error", convert(Float64, rel_error))
  end
  if !ismissing(tree)
    SetParam(p, "tree", convert(String, tree))
  end
  if verbose !== nothing && verbose === true
    EnableVerbose()
  else
    DisableVerbose()
  end

  SetPassed(p, "output_model")
  SetPassed(p, "predictions")
  # Call the program.
  call_kde(p, t)

  results = (kde_internal.GetParamKDEModel(p, "output_model", modelPtrs),
             GetParamCol(p, "predictions", juliaOwnedMemory))

  # We are responsible for cleaning up the `p` and `t` objects.
  DeleteParameters(p)
  DeleteTimers(t)

  return results
end
