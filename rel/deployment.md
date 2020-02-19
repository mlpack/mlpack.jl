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
    `Project.toml` file.
 7. Modify all the bindings to use the packaged `mlpack_jll` wrappers instead of
    the handbuilt ones:

```
for i in src/*.jl;
  do sed -i -e 's/const \(.*\) = joinpath(@__DIR__, "\(.*\)\.so")/import mlpack_jll\nconst \1 = mlpack_jll.\2/' $i;
done
```

 8. Work around the fact that `mlpack_jll` has misnamed GMM probability bindings:

```
for i in src/*.jl;
  do sed -i -e 's/libmlpack_julia_gmm_probability/libmlpack_gmm_probability/' $i;
done
```

 9. Remove the test binding:

```
rm -f src/test_julia_binding.jl
grep -v 'include("test_julia_binding.jl")' src/mlpack.jl > src/mlpack-tmp.jl
mv src/mlpack-tmp.jl src/mlpack.jl
grep -v 'test_julia_binding = util.test_julia_binding' src/functions.jl > src/functions-tmp.jl
mv src/functions-tmp.jl src/functions.jl
```

 10. Remove logistic regression since it isn't currently in `mlpack_jll` (this
     should be fixed very soon!):

```
rm -f src/logistic_regression.jl
grep -v 'include("logistic_regression.jl")' src/mlpack.jl > src/mlpack-tmp.jl
mv src/mlpack-tmp.jl src/mlpack.jl
grep -v 'logistic_regression = util.logistic_regression' src/functions.jl > src/functions-tmp.jl
mv src/functions-tmp.jl src/functions.jl
```

 11. Commit any changed files and any added files in `src/`.
