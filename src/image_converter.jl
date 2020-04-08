export image_converter

using mlpack._Internal.cli

import mlpack_jll
const image_converterLibrary = mlpack_jll.libmlpack_julia_image_converter

# Call the C binding of the mlpack image_converter binding.
function image_converter_mlpackMain()
  success = ccall((:image_converter, image_converterLibrary), Bool, ())
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
`save` as an parameter. An example to load an image : 

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

  CLIRestoreSettings("Image Converter")

  # Process each input argument before calling mlpackMain().
  CLISetParam("input", input)
  if !ismissing(channels)
    CLISetParam("channels", convert(Int, channels))
  end
  if !ismissing(dataset)
    CLISetParamMat("dataset", dataset, points_are_rows)
  end
  if !ismissing(height)
    CLISetParam("height", convert(Int, height))
  end
  if !ismissing(quality)
    CLISetParam("quality", convert(Int, quality))
  end
  if !ismissing(save)
    CLISetParam("save", convert(Bool, save))
  end
  if !ismissing(width)
    CLISetParam("width", convert(Int, width))
  end
  if verbose !== nothing && verbose === true
    CLIEnableVerbose()
  else
    CLIDisableVerbose()
  end

  CLISetPassed("output")
  # Call the program.
  image_converter_mlpackMain()

  return CLIGetParamMat("output", points_are_rows)
end
