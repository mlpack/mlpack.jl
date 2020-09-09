# This file imports all serialization and deserialization functions
# from internal modules.
import Serialization

"""
    serialize_bin(stream::IO, model)

Serialize an mlpack model `model` in the binary boost::serialization
format to the given `stream`.  Example:

```julia
_, model, _, _ = mlpack.logistic_regression(training=x, labels=y)
mlpack.serialize_bin(open("model.bin", "w"), model)
```

The model can later be loaded with `mlpack.deserialize_bin()`, or even
from mlpack bindings in other languages.  However, the format used
here will *not* work with Julia's `Serialization` package!  Use
`Serialization.serialize()` instead.
"""
function serialize_bin(stream::IO, model) end

"""
    deserialize_bin(stream::IO, model_type::Type)
Deserialize an mlpack model type from an input stream.  Specify the 
type of the model manually with the `t` argument.  Example usage:

```julia
lr_model = mlpack.deserialize_bin(stream, LogisticRegression)
```

Only use this if you have saved the model in the boost::serialization
binary format using `serialize_bin()` or an mlpack binding in another
language!  If you used `Serialization.serialize()` to serialize your
model, then use `Serialization.deserialize()` to deserialize it.

Then, the returned model can be passed to appropriate mlpack functions
for machine learning tasks.
"""
function deserialize_bin(stream::IO, t::Type) end

serialize_bin(stream::IO, model::AdaBoostModel) =
    _Internal.adaboost_internal.serializeAdaBoostModel(stream, model)
deserialize_bin(stream::IO, ::Type{AdaBoostModel}) =
    _Internal.adaboost_internal.deserializeAdaBoostModel(stream)

function Serialization.serialize(s::Serialization.AbstractSerializer,
                                 model::AdaBoostModel)
  Serialization.writetag(s.io, Serialization.OBJECT_TAG)
  Serialization.serialize(s, AdaBoostModel)
  serialize_bin(s.io, model)
end

function Serialization.deserialize(s::Serialization.AbstractSerializer,
                                   ::Type{AdaBoostModel})
  deserialize_bin(s.io, AdaBoostModel)
end

serialize_bin(stream::IO, model::ApproxKFNModel) =
    _Internal.approx_kfn_internal.serializeApproxKFNModel(stream, model)
deserialize_bin(stream::IO, ::Type{ApproxKFNModel}) =
    _Internal.approx_kfn_internal.deserializeApproxKFNModel(stream)

function Serialization.serialize(s::Serialization.AbstractSerializer,
                                 model::ApproxKFNModel)
  Serialization.writetag(s.io, Serialization.OBJECT_TAG)
  Serialization.serialize(s, ApproxKFNModel)
  serialize_bin(s.io, model)
end

function Serialization.deserialize(s::Serialization.AbstractSerializer,
                                   ::Type{ApproxKFNModel})
  deserialize_bin(s.io, ApproxKFNModel)
end

serialize_bin(stream::IO, model::BayesianLinearRegression) =
    _Internal.bayesian_linear_regression_internal.serializeBayesianLinearRegression(stream, model)
deserialize_bin(stream::IO, ::Type{BayesianLinearRegression}) =
    _Internal.bayesian_linear_regression_internal.deserializeBayesianLinearRegression(stream)

function Serialization.serialize(s::Serialization.AbstractSerializer,
                                 model::BayesianLinearRegression)
  Serialization.writetag(s.io, Serialization.OBJECT_TAG)
  Serialization.serialize(s, BayesianLinearRegression)
  serialize_bin(s.io, model)
end

function Serialization.deserialize(s::Serialization.AbstractSerializer,
                                   ::Type{BayesianLinearRegression})
  deserialize_bin(s.io, BayesianLinearRegression)
end

serialize_bin(stream::IO, model::CFModel) =
    _Internal.cf_internal.serializeCFModel(stream, model)
deserialize_bin(stream::IO, ::Type{CFModel}) =
    _Internal.cf_internal.deserializeCFModel(stream)

function Serialization.serialize(s::Serialization.AbstractSerializer,
                                 model::CFModel)
  Serialization.writetag(s.io, Serialization.OBJECT_TAG)
  Serialization.serialize(s, CFModel)
  serialize_bin(s.io, model)
end

function Serialization.deserialize(s::Serialization.AbstractSerializer,
                                   ::Type{CFModel})
  deserialize_bin(s.io, CFModel)
end

serialize_bin(stream::IO, model::DSModel) =
    _Internal.decision_stump_internal.serializeDSModel(stream, model)
deserialize_bin(stream::IO, ::Type{DSModel}) =
    _Internal.decision_stump_internal.deserializeDSModel(stream)

function Serialization.serialize(s::Serialization.AbstractSerializer,
                                 model::DSModel)
  Serialization.writetag(s.io, Serialization.OBJECT_TAG)
  Serialization.serialize(s, DSModel)
  serialize_bin(s.io, model)
end

function Serialization.deserialize(s::Serialization.AbstractSerializer,
                                   ::Type{DSModel})
  deserialize_bin(s.io, DSModel)
end

serialize_bin(stream::IO, model::DecisionTreeModel) =
    _Internal.decision_tree_internal.serializeDecisionTreeModel(stream, model)
deserialize_bin(stream::IO, ::Type{DecisionTreeModel}) =
    _Internal.decision_tree_internal.deserializeDecisionTreeModel(stream)

function Serialization.serialize(s::Serialization.AbstractSerializer,
                                 model::DecisionTreeModel)
  Serialization.writetag(s.io, Serialization.OBJECT_TAG)
  Serialization.serialize(s, DecisionTreeModel)
  serialize_bin(s.io, model)
end

function Serialization.deserialize(s::Serialization.AbstractSerializer,
                                   ::Type{DecisionTreeModel})
  deserialize_bin(s.io, DecisionTreeModel)
end

serialize_bin(stream::IO, model::DTree) =
    _Internal.det_internal.serializeDTree(stream, model)
deserialize_bin(stream::IO, ::Type{DTree}) =
    _Internal.det_internal.deserializeDTree(stream)

function Serialization.serialize(s::Serialization.AbstractSerializer,
                                 model::DTree)
  Serialization.writetag(s.io, Serialization.OBJECT_TAG)
  Serialization.serialize(s, DTree)
  serialize_bin(s.io, model)
end

function Serialization.deserialize(s::Serialization.AbstractSerializer,
                                   ::Type{DTree})
  deserialize_bin(s.io, DTree)
end

serialize_bin(stream::IO, model::FastMKSModel) =
    _Internal.fastmks_internal.serializeFastMKSModel(stream, model)
deserialize_bin(stream::IO, ::Type{FastMKSModel}) =
    _Internal.fastmks_internal.deserializeFastMKSModel(stream)

function Serialization.serialize(s::Serialization.AbstractSerializer,
                                 model::FastMKSModel)
  Serialization.writetag(s.io, Serialization.OBJECT_TAG)
  Serialization.serialize(s, FastMKSModel)
  serialize_bin(s.io, model)
end

function Serialization.deserialize(s::Serialization.AbstractSerializer,
                                   ::Type{FastMKSModel})
  deserialize_bin(s.io, FastMKSModel)
end

serialize_bin(stream::IO, model::GMM) =
    _Internal.gmm_train_internal.serializeGMM(stream, model)
deserialize_bin(stream::IO, ::Type{GMM}) =
    _Internal.gmm_train_internal.deserializeGMM(stream)

function Serialization.serialize(s::Serialization.AbstractSerializer,
                                 model::GMM)
  Serialization.writetag(s.io, Serialization.OBJECT_TAG)
  Serialization.serialize(s, GMM)
  serialize_bin(s.io, model)
end

function Serialization.deserialize(s::Serialization.AbstractSerializer,
                                   ::Type{GMM})
  deserialize_bin(s.io, GMM)
end

serialize_bin(stream::IO, model::HMMModel) =
    _Internal.hmm_train_internal.serializeHMMModel(stream, model)
deserialize_bin(stream::IO, ::Type{HMMModel}) =
    _Internal.hmm_train_internal.deserializeHMMModel(stream)

function Serialization.serialize(s::Serialization.AbstractSerializer,
                                 model::HMMModel)
  Serialization.writetag(s.io, Serialization.OBJECT_TAG)
  Serialization.serialize(s, HMMModel)
  serialize_bin(s.io, model)
end

function Serialization.deserialize(s::Serialization.AbstractSerializer,
                                   ::Type{HMMModel})
  deserialize_bin(s.io, HMMModel)
end

serialize_bin(stream::IO, model::HoeffdingTreeModel) =
    _Internal.hoeffding_tree_internal.serializeHoeffdingTreeModel(stream, model)
deserialize_bin(stream::IO, ::Type{HoeffdingTreeModel}) =
    _Internal.hoeffding_tree_internal.deserializeHoeffdingTreeModel(stream)

function Serialization.serialize(s::Serialization.AbstractSerializer,
                                 model::HoeffdingTreeModel)
  Serialization.writetag(s.io, Serialization.OBJECT_TAG)
  Serialization.serialize(s, HoeffdingTreeModel)
  serialize_bin(s.io, model)
end

function Serialization.deserialize(s::Serialization.AbstractSerializer,
                                   ::Type{HoeffdingTreeModel})
  deserialize_bin(s.io, HoeffdingTreeModel)
end

serialize_bin(stream::IO, model::KDEModel) =
    _Internal.kde_internal.serializeKDEModel(stream, model)
deserialize_bin(stream::IO, ::Type{KDEModel}) =
    _Internal.kde_internal.deserializeKDEModel(stream)

function Serialization.serialize(s::Serialization.AbstractSerializer,
                                 model::KDEModel)
  Serialization.writetag(s.io, Serialization.OBJECT_TAG)
  Serialization.serialize(s, KDEModel)
  serialize_bin(s.io, model)
end

function Serialization.deserialize(s::Serialization.AbstractSerializer,
                                   ::Type{KDEModel})
  deserialize_bin(s.io, KDEModel)
end

serialize_bin(stream::IO, model::LARS) =
    _Internal.lars_internal.serializeLARS(stream, model)
deserialize_bin(stream::IO, ::Type{LARS}) =
    _Internal.lars_internal.deserializeLARS(stream)

function Serialization.serialize(s::Serialization.AbstractSerializer,
                                 model::LARS)
  Serialization.writetag(s.io, Serialization.OBJECT_TAG)
  Serialization.serialize(s, LARS)
  serialize_bin(s.io, model)
end

function Serialization.deserialize(s::Serialization.AbstractSerializer,
                                   ::Type{LARS})
  deserialize_bin(s.io, LARS)
end

serialize_bin(stream::IO, model::LinearRegression) =
    _Internal.linear_regression_internal.serializeLinearRegression(stream, model)
deserialize_bin(stream::IO, ::Type{LinearRegression}) =
    _Internal.linear_regression_internal.deserializeLinearRegression(stream)

function Serialization.serialize(s::Serialization.AbstractSerializer,
                                 model::LinearRegression)
  Serialization.writetag(s.io, Serialization.OBJECT_TAG)
  Serialization.serialize(s, LinearRegression)
  serialize_bin(s.io, model)
end

function Serialization.deserialize(s::Serialization.AbstractSerializer,
                                   ::Type{LinearRegression})
  deserialize_bin(s.io, LinearRegression)
end

serialize_bin(stream::IO, model::LinearSVMModel) =
    _Internal.linear_svm_internal.serializeLinearSVMModel(stream, model)
deserialize_bin(stream::IO, ::Type{LinearSVMModel}) =
    _Internal.linear_svm_internal.deserializeLinearSVMModel(stream)

function Serialization.serialize(s::Serialization.AbstractSerializer,
                                 model::LinearSVMModel)
  Serialization.writetag(s.io, Serialization.OBJECT_TAG)
  Serialization.serialize(s, LinearSVMModel)
  serialize_bin(s.io, model)
end

function Serialization.deserialize(s::Serialization.AbstractSerializer,
                                   ::Type{LinearSVMModel})
  deserialize_bin(s.io, LinearSVMModel)
end

serialize_bin(stream::IO, model::LocalCoordinateCoding) =
    _Internal.local_coordinate_coding_internal.serializeLocalCoordinateCoding(stream, model)
deserialize_bin(stream::IO, ::Type{LocalCoordinateCoding}) =
    _Internal.local_coordinate_coding_internal.deserializeLocalCoordinateCoding(stream)

function Serialization.serialize(s::Serialization.AbstractSerializer,
                                 model::LocalCoordinateCoding)
  Serialization.writetag(s.io, Serialization.OBJECT_TAG)
  Serialization.serialize(s, LocalCoordinateCoding)
  serialize_bin(s.io, model)
end

function Serialization.deserialize(s::Serialization.AbstractSerializer,
                                   ::Type{LocalCoordinateCoding})
  deserialize_bin(s.io, LocalCoordinateCoding)
end

serialize_bin(stream::IO, model::LogisticRegression) =
    _Internal.logistic_regression_internal.serializeLogisticRegression(stream, model)
deserialize_bin(stream::IO, ::Type{LogisticRegression}) =
    _Internal.logistic_regression_internal.deserializeLogisticRegression(stream)

function Serialization.serialize(s::Serialization.AbstractSerializer,
                                 model::LogisticRegression)
  Serialization.writetag(s.io, Serialization.OBJECT_TAG)
  Serialization.serialize(s, LogisticRegression)
  serialize_bin(s.io, model)
end

function Serialization.deserialize(s::Serialization.AbstractSerializer,
                                   ::Type{LogisticRegression})
  deserialize_bin(s.io, LogisticRegression)
end

serialize_bin(stream::IO, model::LSHSearch) =
    _Internal.lsh_internal.serializeLSHSearch(stream, model)
deserialize_bin(stream::IO, ::Type{LSHSearch}) =
    _Internal.lsh_internal.deserializeLSHSearch(stream)

function Serialization.serialize(s::Serialization.AbstractSerializer,
                                 model::LSHSearch)
  Serialization.writetag(s.io, Serialization.OBJECT_TAG)
  Serialization.serialize(s, LSHSearch)
  serialize_bin(s.io, model)
end

function Serialization.deserialize(s::Serialization.AbstractSerializer,
                                   ::Type{LSHSearch})
  deserialize_bin(s.io, LSHSearch)
end

serialize_bin(stream::IO, model::NBCModel) =
    _Internal.nbc_internal.serializeNBCModel(stream, model)
deserialize_bin(stream::IO, ::Type{NBCModel}) =
    _Internal.nbc_internal.deserializeNBCModel(stream)

function Serialization.serialize(s::Serialization.AbstractSerializer,
                                 model::NBCModel)
  Serialization.writetag(s.io, Serialization.OBJECT_TAG)
  Serialization.serialize(s, NBCModel)
  serialize_bin(s.io, model)
end

function Serialization.deserialize(s::Serialization.AbstractSerializer,
                                   ::Type{NBCModel})
  deserialize_bin(s.io, NBCModel)
end

serialize_bin(stream::IO, model::KNNModel) =
    _Internal.knn_internal.serializeKNNModel(stream, model)
deserialize_bin(stream::IO, ::Type{KNNModel}) =
    _Internal.knn_internal.deserializeKNNModel(stream)

function Serialization.serialize(s::Serialization.AbstractSerializer,
                                 model::KNNModel)
  Serialization.writetag(s.io, Serialization.OBJECT_TAG)
  Serialization.serialize(s, KNNModel)
  serialize_bin(s.io, model)
end

function Serialization.deserialize(s::Serialization.AbstractSerializer,
                                   ::Type{KNNModel})
  deserialize_bin(s.io, KNNModel)
end

serialize_bin(stream::IO, model::KFNModel) =
    _Internal.kfn_internal.serializeKFNModel(stream, model)
deserialize_bin(stream::IO, ::Type{KFNModel}) =
    _Internal.kfn_internal.deserializeKFNModel(stream)

function Serialization.serialize(s::Serialization.AbstractSerializer,
                                 model::KFNModel)
  Serialization.writetag(s.io, Serialization.OBJECT_TAG)
  Serialization.serialize(s, KFNModel)
  serialize_bin(s.io, model)
end

function Serialization.deserialize(s::Serialization.AbstractSerializer,
                                   ::Type{KFNModel})
  deserialize_bin(s.io, KFNModel)
end

serialize_bin(stream::IO, model::PerceptronModel) =
    _Internal.perceptron_internal.serializePerceptronModel(stream, model)
deserialize_bin(stream::IO, ::Type{PerceptronModel}) =
    _Internal.perceptron_internal.deserializePerceptronModel(stream)

function Serialization.serialize(s::Serialization.AbstractSerializer,
                                 model::PerceptronModel)
  Serialization.writetag(s.io, Serialization.OBJECT_TAG)
  Serialization.serialize(s, PerceptronModel)
  serialize_bin(s.io, model)
end

function Serialization.deserialize(s::Serialization.AbstractSerializer,
                                   ::Type{PerceptronModel})
  deserialize_bin(s.io, PerceptronModel)
end

serialize_bin(stream::IO, model::ScalingModel) =
    _Internal.preprocess_scale_internal.serializeScalingModel(stream, model)
deserialize_bin(stream::IO, ::Type{ScalingModel}) =
    _Internal.preprocess_scale_internal.deserializeScalingModel(stream)

function Serialization.serialize(s::Serialization.AbstractSerializer,
                                 model::ScalingModel)
  Serialization.writetag(s.io, Serialization.OBJECT_TAG)
  Serialization.serialize(s, ScalingModel)
  serialize_bin(s.io, model)
end

function Serialization.deserialize(s::Serialization.AbstractSerializer,
                                   ::Type{ScalingModel})
  deserialize_bin(s.io, ScalingModel)
end

serialize_bin(stream::IO, model::RandomForestModel) =
    _Internal.random_forest_internal.serializeRandomForestModel(stream, model)
deserialize_bin(stream::IO, ::Type{RandomForestModel}) =
    _Internal.random_forest_internal.deserializeRandomForestModel(stream)

function Serialization.serialize(s::Serialization.AbstractSerializer,
                                 model::RandomForestModel)
  Serialization.writetag(s.io, Serialization.OBJECT_TAG)
  Serialization.serialize(s, RandomForestModel)
  serialize_bin(s.io, model)
end

function Serialization.deserialize(s::Serialization.AbstractSerializer,
                                   ::Type{RandomForestModel})
  deserialize_bin(s.io, RandomForestModel)
end

serialize_bin(stream::IO, model::RANNModel) =
    _Internal.krann_internal.serializeRANNModel(stream, model)
deserialize_bin(stream::IO, ::Type{RANNModel}) =
    _Internal.krann_internal.deserializeRANNModel(stream)

function Serialization.serialize(s::Serialization.AbstractSerializer,
                                 model::RANNModel)
  Serialization.writetag(s.io, Serialization.OBJECT_TAG)
  Serialization.serialize(s, RANNModel)
  serialize_bin(s.io, model)
end

function Serialization.deserialize(s::Serialization.AbstractSerializer,
                                   ::Type{RANNModel})
  deserialize_bin(s.io, RANNModel)
end

serialize_bin(stream::IO, model::SoftmaxRegression) =
    _Internal.softmax_regression_internal.serializeSoftmaxRegression(stream, model)
deserialize_bin(stream::IO, ::Type{SoftmaxRegression}) =
    _Internal.softmax_regression_internal.deserializeSoftmaxRegression(stream)

function Serialization.serialize(s::Serialization.AbstractSerializer,
                                 model::SoftmaxRegression)
  Serialization.writetag(s.io, Serialization.OBJECT_TAG)
  Serialization.serialize(s, SoftmaxRegression)
  serialize_bin(s.io, model)
end

function Serialization.deserialize(s::Serialization.AbstractSerializer,
                                   ::Type{SoftmaxRegression})
  deserialize_bin(s.io, SoftmaxRegression)
end

serialize_bin(stream::IO, model::SparseCoding) =
    _Internal.sparse_coding_internal.serializeSparseCoding(stream, model)
deserialize_bin(stream::IO, ::Type{SparseCoding}) =
    _Internal.sparse_coding_internal.deserializeSparseCoding(stream)

function Serialization.serialize(s::Serialization.AbstractSerializer,
                                 model::SparseCoding)
  Serialization.writetag(s.io, Serialization.OBJECT_TAG)
  Serialization.serialize(s, SparseCoding)
  serialize_bin(s.io, model)
end

function Serialization.deserialize(s::Serialization.AbstractSerializer,
                                   ::Type{SparseCoding})
  deserialize_bin(s.io, SparseCoding)
end

