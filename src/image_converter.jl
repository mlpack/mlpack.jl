export image_converter


using mlpack._Internal.params

import mlpack_jll
const image_converterLibrary = mlpack_jll.libmlpack_julia_image_converter

# Call the C binding of the mlpack image_converter binding.
function call_image_converter(p, t)
  success = ccall((:mlpack_image_converter, image_converterLibrary), Bool, (Ptr{Nothing}, Ptr{Nothing}), p, t)
  if !success
    # Throw an exception---false means there was a C++ exception.
    throw(ErrorException("mlpack binding error; see output"))
  end
end

" Internal module to hold utility functions. "
module image_converter_internal
  import ..image_converterLibrary

end # module

"""
    image_converter(input; [channels, dataset, height, quality, save, verbose, width])

This utility takes an image or an array of images and loads them to a matrix.
You can optionally specify the height `height` width `width` and channel
`channels` of the images that needs to be loaded; otherwise, these parameters
will be automatically detected from the image.
There are other options too, that can be specified such as `quality`.

You can also provide a dataset and save them as images using `dataset` and
`save` as an parameter.

 An example to load an image : 

```julia
julia> Y = image_converter(X; channels=3, height=256, width=256)
```

 An example to save an image is :

```julia
julia> using CSV
julia> Y = CSV.read("Y.csv")
julia> _ = image_converter(X; channels=3, dataset=Y, height=256,
            save=1, width=256)
```

# Arguments

 - `input::Vector{String}`: Image filenames which have to be
      loaded/saved.
 - `channels::Int`: Number of channels in the image.  Default value `0`.

 - `dataset::Array{Float64, 2}`: Input matrix to save as images.
 - `height::Int`: Height of the images.  Default value `0`.

 - `quality::Int`: Compression of the image if saved as jpg (0-100). 
      Default value `90`.
      
 - `save::Bool`: Save a dataset as images.  Default value `false`.

 - `verbose::Bool`: Display informational messages and the full list of
      parameters and timers at the end of execution.  Default value `false`.
      
 - `width::Int`: Width of the image.  Default value `0`.


# Return values

 - `output::Array{Float64, 2}`: Matrix to save images data to, Onlyneeded
      if you are specifying 'save' option.

"""
function image_converter(input::Vector{String};
                         channels::Union{Int, Missing} = missing,
                         dataset = missing,
                         height::Union{Int, Missing} = missing,
                         quality::Union{Int, Missing} = missing,
                         save::Union{Bool, Missing} = missing,
                         verbose::Union{Bool, Missing} = missing,
                         width::Union{Int, Missing} = missing,
                         points_are_rows::Bool = true)
  # Force the symbols to load.
  ccall((:loadSymbols, image_converterLibrary), Nothing, ());

  # Create the set of model pointers to avoid setting multiple finalizers.
  modelPtrs = Set{Ptr{Nothing}}()

  p = GetParameters("image_converter")
  t = Timers()

  juliaOwnedMemory = Set{Ptr{Nothing}}()
  # Process each input argument before calling mlpackMain().
  SetParam(p, "input", input)
  if !ismissing(channels)
    SetParam(p, "channels", convert(Int, channels))
  end
  if !ismissing(dataset)
    SetParamMat(p, "dataset", dataset, points_are_rows, juliaOwnedMemory)
  end
  if !ismissing(height)
    SetParam(p, "height", convert(Int, height))
  end
  if !ismissing(quality)
    SetParam(p, "quality", convert(Int, quality))
  end
  if !ismissing(save)
    SetParam(p, "save", convert(Bool, save))
  end
  if !ismissing(width)
    SetParam(p, "width", convert(Int, width))
  end
  if verbose !== nothing && verbose === true
    EnableVerbose()
  else
    DisableVerbose()
  end

  SetPassed(p, "output")
  # Call the program.
  call_image_converter(p, t)

  results = (GetParamMat(p, "output", points_are_rows, juliaOwnedMemory))

  # We are responsible for cleaning up the `p` and `t` objects.
  DeleteParameters(p)
  DeleteTimers(t)

  return results
end
