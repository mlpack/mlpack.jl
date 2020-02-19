# How to deploy a new mlpack version to mlpack.jl.

This process isn't automatic, and the steps below can be refined as time goes on
into an automated script:

 1. Check out mlpack code.
 2. Configure and build the `julia` target: `make julia`
 3. Copy the contents of the `src/mlpack/bindings/julia/mlpack/src/` directory
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
    `Project.toml` file.
 7. Commit any changed files and any added files in `src/`.
