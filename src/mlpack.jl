"""
    mlpack

mlpack is a fast, flexible machine learning library, written in C++, that aims
to provide fast, extensible implementations of cutting-edge machine learning
algorithms.  This module provides those implementations as Julia functions.

Each function inside the module performs a specific machine learning task.

For complete documentation of these functions, including example usage, see the
mlpack website's documentation for the Julia bindings:

https://www.mlpack.org/doc/user/bindings/julia.html

Each function also contains an equivalent docstring; the Julia REPL's help
functionality can be used to access the documentation that way.
"""
module mlpack

# Include any types of models.
include("types.jl")

"""
    mlpack._Internal

This module contains internal implementations details of mlpack.  There
shouldn't be any need to go digging around in here if you're just using mlpack.
(But don't let this comment discourage you if you're just curious and poking
around!)
"""
module _Internal

include("params.jl")
include("approx_kfn.jl")
include("bayesian_linear_regression.jl")
include("cf.jl")
include("dbscan.jl")
include("decision_tree.jl")
include("det.jl")
include("emst.jl")
include("fastmks.jl")
include("gmm_train.jl")
include("gmm_generate.jl")
include("gmm_probability.jl")
include("hmm_train.jl")
include("hmm_generate.jl")
include("hmm_loglik.jl")
include("hmm_viterbi.jl")
include("hoeffding_tree.jl")
include("image_converter.jl")
include("kde.jl")
include("kernel_pca.jl")
include("kmeans.jl")
include("lars.jl")
include("linear_svm.jl")
include("lmnn.jl")
include("local_coordinate_coding.jl")
include("logistic_regression.jl")
include("lsh.jl")
include("mean_shift.jl")
include("nbc.jl")
include("nca.jl")
include("knn.jl")
include("kfn.jl")
include("nmf.jl")
include("pca.jl")
include("perceptron.jl")
include("preprocess_split.jl")
include("preprocess_binarize.jl")
include("preprocess_describe.jl")
include("preprocess_scale.jl")
include("preprocess_one_hot_encoding.jl")
include("radical.jl")
include("random_forest.jl")
include("krann.jl")
include("softmax_regression.jl")
include("sparse_coding.jl")
include("adaboost.jl")
include("linear_regression.jl")

end
include("functions.jl")
include("serialization.jl")
end
