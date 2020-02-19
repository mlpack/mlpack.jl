export test_julia_binding

using mlpack.util.cli

const test_julia_bindingLibrary = joinpath(@__DIR__, "libmlpack_julia_test_julia_binding.so")

# Call the C binding of the mlpack test_julia_binding binding.
function test_julia_binding_mlpackMain()
  success = ccall((:test_julia_binding, test_julia_bindingLibrary), Bool, ())
  if !success
    # Throw an exception---false means there was a C++ exception.
    throw(ErrorException("mlpack binding error; see output"))
  end
end

" Internal module to hold utility functions. "
module test_julia_binding_internal
  import ..test_julia_bindingLibrary

" Get the value of a model pointer parameter of type GaussianKernel."
function CLIGetParamGaussianKernelPtr(paramName::String)
  return ccall((:CLI_GetParamGaussianKernelPtr, test_julia_bindingLibrary), Ptr{Nothing}, (Cstring,), paramName)
end

" Set the value of a model pointer parameter of type GaussianKernel."
function CLISetParamGaussianKernelPtr(paramName::String, ptr::Ptr{Nothing})
  ccall((:CLI_SetParamGaussianKernelPtr, test_julia_bindingLibrary), Nothing, (Cstring, Ptr{Nothing}), paramName, ptr)
end

end # module

"""
    test_julia_binding(double_in, int_in, string_in; [build_model, col_in, flag1, flag2, matrix_and_info_in, matrix_in, model_in, row_in, str_vector_in, ucol_in, umatrix_in, urow_in, vector_in, verbose])

A simple program to test Julia binding functionality.  You can build mlpack with
the BUILD_TESTS option set to off, and this binding will no longer be built.

# Arguments

 - `double_in::Float64`: Input double, must be 4.0.
 - `int_in::Int`: Input int, must be 12.
 - `string_in::String`: Input string, must be 'hello'.
 - `build_model::Bool`: If true, a model will be returned.  Default value
      `false`.
      
 - `col_in::Array{Float64, 1}`: Input column.
 - `flag1::Bool`: Input flag, must be specified.  Default value `false`.

 - `flag2::Bool`: Input flag, must not be specified.  Default value
      `false`.
      
 - `matrix_and_info_in::Tuple{Array{Bool, 1}, Array{Float64, 2}}`: Input
      matrix and info.
 - `matrix_in::Array{Float64, 2}`: Input matrix.
 - `model_in::unknown_`: Input model.
 - `row_in::Array{Float64, 1}`: Input row.
 - `str_vector_in::Vector{String}`: Input vector of strings.
 - `ucol_in::Array{Int64, 1}`: Input unsigned column.
 - `umatrix_in::Array{Int64, 2}`: Input unsigned matrix.
 - `urow_in::Array{Int64, 1}`: Input unsigned row.
 - `vector_in::Vector{Int}`: Input vector of numbers.
 - `verbose::Bool`: Display informational messages and the full list of
      parameters and timers at the end of execution.  Default value `false`.
      

# Return values

 - `col_out::Array{Float64, 1}`: Output column. 2x input column
 - `double_out::Float64`: Output double, will be 5.0.  Default value `0`.
      
 - `int_out::Int`: Output int, will be 13.  Default value `0`.

 - `matrix_and_info_out::Array{Float64, 2}`: Output matrix and info; all
      numeric elements multiplied by 3.
 - `matrix_out::Array{Float64, 2}`: Output matrix.
 - `model_bw_out::Float64`: The bandwidth of the model.  Default value
      `0`.
      
 - `model_out::unknown_`: Output model, with twice the bandwidth.
 - `row_out::Array{Float64, 1}`: Output row.  2x input row.
 - `str_vector_out::Vector{String}`: Output string vector.
 - `string_out::String`: Output string, will be 'hello2'.  Default value
      ``.
      
 - `ucol_out::Array{Int64, 1}`: Output unsigned column. 2x input column.
 - `umatrix_out::Array{Int64, 2}`: Output unsigned matrix.
 - `urow_out::Array{Int64, 1}`: Output unsigned row.  2x input row.
 - `vector_out::Vector{Int}`: Output vector.

"""
function test_julia_binding(double_in::Float64,
                            int_in::Int,
                            string_in::String;
                            build_model::Union{Bool, Missing} = missing,
                            col_in = missing,
                            flag1::Union{Bool, Missing} = missing,
                            flag2::Union{Bool, Missing} = missing,
                            matrix_and_info_in::Union{Tuple{Array{Bool, 1}, Array{Float64, 2}}, Missing} = missing,
                            matrix_in = missing,
                            model_in::Union{Ptr{Nothing}, Missing} = missing,
                            row_in = missing,
                            str_vector_in::Union{Vector{String}, Missing} = missing,
                            ucol_in = missing,
                            umatrix_in = missing,
                            urow_in = missing,
                            vector_in::Union{Vector{Int}, Missing} = missing,
                            verbose::Union{Bool, Missing} = missing,
                            points_are_rows::Bool = true)
  # Force the symbols to load.
  ccall((:loadSymbols, test_julia_bindingLibrary), Nothing, ());

  CLIRestoreSettings("Julia binding test")

  # Process each input argument before calling mlpackMain().
  CLISetParam("double_in", double_in)
  CLISetParam("int_in", int_in)
  CLISetParam("string_in", string_in)
  if !ismissing(build_model)
    CLISetParam("build_model", convert(Bool, build_model))
  end
  if !ismissing(col_in)
    CLISetParamCol("col_in", col_in)
  end
  if !ismissing(flag1)
    CLISetParam("flag1", convert(Bool, flag1))
  end
  if !ismissing(flag2)
    CLISetParam("flag2", convert(Bool, flag2))
  end
  if !ismissing(matrix_and_info_in)
    CLISetParam("matrix_and_info_in", convert(Tuple{Array{Bool, 1}, Array{Float64, 2}}, matrix_and_info_in), points_are_rows)
  end
  if !ismissing(matrix_in)
    CLISetParamMat("matrix_in", matrix_in, points_are_rows)
  end
  if !ismissing(model_in)
    test_julia_binding_internal.CLISetParamGaussianKernelPtr("model_in", convert(Ptr{Nothing}, model_in))
  end
  if !ismissing(row_in)
    CLISetParamRow("row_in", row_in)
  end
  if !ismissing(str_vector_in)
    CLISetParam("str_vector_in", convert(Vector{String}, str_vector_in))
  end
  if !ismissing(ucol_in)
    CLISetParamUCol("ucol_in", ucol_in)
  end
  if !ismissing(umatrix_in)
    CLISetParamUMat("umatrix_in", umatrix_in, points_are_rows)
  end
  if !ismissing(urow_in)
    CLISetParamURow("urow_in", urow_in)
  end
  if !ismissing(vector_in)
    CLISetParam("vector_in", convert(Vector{Int}, vector_in))
  end
  if verbose !== nothing && verbose === true
    CLIEnableVerbose()
  else
    CLIDisableVerbose()
  end

  CLISetPassed("col_out")
  CLISetPassed("double_out")
  CLISetPassed("int_out")
  CLISetPassed("matrix_and_info_out")
  CLISetPassed("matrix_out")
  CLISetPassed("model_bw_out")
  CLISetPassed("model_out")
  CLISetPassed("row_out")
  CLISetPassed("str_vector_out")
  CLISetPassed("string_out")
  CLISetPassed("ucol_out")
  CLISetPassed("umatrix_out")
  CLISetPassed("urow_out")
  CLISetPassed("vector_out")
  # Call the program.
  test_julia_binding_mlpackMain()

  return CLIGetParamCol("col_out"),
         CLIGetParamDouble("double_out"),
         CLIGetParamInt("int_out"),
         CLIGetParamMat("matrix_and_info_out", points_are_rows),
         CLIGetParamMat("matrix_out", points_are_rows),
         CLIGetParamDouble("model_bw_out"),
         test_julia_binding_internal.CLIGetParamGaussianKernelPtr("model_out"),
         CLIGetParamRow("row_out"),
         CLIGetParamVectorStr("str_vector_out"),
         Base.unsafe_string(CLIGetParamString("string_out")),
         CLIGetParamUCol("ucol_out"),
         CLIGetParamUMat("umatrix_out", points_are_rows),
         CLIGetParamURow("urow_out"),
         CLIGetParamVectorInt("vector_out")
end
