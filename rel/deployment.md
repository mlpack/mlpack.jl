# How to deploy a new mlpack version to mlpack.jl.

This process isn't automatic, and the steps below can be refined as time goes on
into an automated script:

 1. Check out mlpack code.
 2. Configure and build the `julia` target: `make julia`
 3. Copy the Julia bindings (`build/src/mlpack/bindings/julia/mlpack/src/*.jl`)
    to the `src/` directory in this repository.
 4. Copy `src/mlpack/bindings/julia/mlpack/Project.toml` to this root of this
    repository.
 5. Start a Julia session in the root of the repository with, e.g.,
    `JULIA_PROJECT=$PWD julia` and add `mlpack_jll` as a dependency:

```
julia> import Pkg
julia> Pkg.add("mlpack_jll")
```

 6. See that `mlpack_jll` has been added to the `[deps]` section of the
    `Project.toml` file.  Ensure that the right version is specified in the
    `[compat]` section.
 7. Modify all the bindings to use the packaged `mlpack_jll` wrappers instead of
    the handbuilt ones:

```
for i in src/*.jl;
  do sed -i -e 's/const \(.*\) = joinpath(@__DIR__, "\(.*\)\.so")/import mlpack_jll\nconst \1 = mlpack_jll.\2/' $i;
done
```

 8. Remove the test binding:

```
rm -f src/test_julia_binding.jl
grep -v 'include("test_julia_binding.jl")' src/mlpack.jl > src/mlpack-tmp.jl
mv src/mlpack-tmp.jl src/mlpack.jl
grep -v 'test_julia_binding = _Internal.test_julia_binding' src/functions.jl > src/functions-tmp.jl
mv src/functions-tmp.jl src/functions.jl
```

 9. Manually remove `GaussianKernel` from `src/types.jl`. (TODO: automate!)

 10. Manually remove the `GaussianKernel` serialization from
     `src/serialization.jl` (TODO: automate!)

 11. Add the Julia dependency information to `Project.toml`:

```
[compat]
julia = "1.3"
mlpack_jll = "x.y.z"
```

 12. Commit any changed files and any added files in `src/`.

 13. Update the package in the Julia registry by using the web interface on
     https://juliahub.com.  (Find the "Register packages" drop-down after
     logging in, and follow the directions there to open a PR to the registry.)
