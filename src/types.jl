# This file defines all of the mlpack types.

mutable struct ApproxKFNModel
  ptr::Ptr{Nothing}

  # Construct object and set finalizer to free memory if `finalize` is true.
  function ApproxKFNModel(ptr::Ptr{Nothing}; finalize::Bool = false)::ApproxKFNModel
    result = new(ptr)
    if finalize
      finalizer(
          x -> _Internal.approx_kfn_internal.DeleteApproxKFNModel(x.ptr),
          result)
    end
    return result
  end
end

mutable struct BayesianLinearRegression
  ptr::Ptr{Nothing}

  # Construct object and set finalizer to free memory if `finalize` is true.
  function BayesianLinearRegression(ptr::Ptr{Nothing}; finalize::Bool = false)::BayesianLinearRegression
    result = new(ptr)
    if finalize
      finalizer(
          x -> _Internal.bayesian_linear_regression_internal.DeleteBayesianLinearRegression(x.ptr),
          result)
    end
    return result
  end
end

mutable struct CFModel
  ptr::Ptr{Nothing}

  # Construct object and set finalizer to free memory if `finalize` is true.
  function CFModel(ptr::Ptr{Nothing}; finalize::Bool = false)::CFModel
    result = new(ptr)
    if finalize
      finalizer(
          x -> _Internal.cf_internal.DeleteCFModel(x.ptr),
          result)
    end
    return result
  end
end

mutable struct DecisionTreeModel
  ptr::Ptr{Nothing}

  # Construct object and set finalizer to free memory if `finalize` is true.
  function DecisionTreeModel(ptr::Ptr{Nothing}; finalize::Bool = false)::DecisionTreeModel
    result = new(ptr)
    if finalize
      finalizer(
          x -> _Internal.decision_tree_internal.DeleteDecisionTreeModel(x.ptr),
          result)
    end
    return result
  end
end

mutable struct DTree
  ptr::Ptr{Nothing}

  # Construct object and set finalizer to free memory if `finalize` is true.
  function DTree(ptr::Ptr{Nothing}; finalize::Bool = false)::DTree
    result = new(ptr)
    if finalize
      finalizer(
          x -> _Internal.det_internal.DeleteDTree(x.ptr),
          result)
    end
    return result
  end
end

mutable struct FastMKSModel
  ptr::Ptr{Nothing}

  # Construct object and set finalizer to free memory if `finalize` is true.
  function FastMKSModel(ptr::Ptr{Nothing}; finalize::Bool = false)::FastMKSModel
    result = new(ptr)
    if finalize
      finalizer(
          x -> _Internal.fastmks_internal.DeleteFastMKSModel(x.ptr),
          result)
    end
    return result
  end
end

mutable struct GMM
  ptr::Ptr{Nothing}

  # Construct object and set finalizer to free memory if `finalize` is true.
  function GMM(ptr::Ptr{Nothing}; finalize::Bool = false)::GMM
    result = new(ptr)
    if finalize
      finalizer(
          x -> _Internal.gmm_train_internal.DeleteGMM(x.ptr),
          result)
    end
    return result
  end
end

mutable struct HMMModel
  ptr::Ptr{Nothing}

  # Construct object and set finalizer to free memory if `finalize` is true.
  function HMMModel(ptr::Ptr{Nothing}; finalize::Bool = false)::HMMModel
    result = new(ptr)
    if finalize
      finalizer(
          x -> _Internal.hmm_train_internal.DeleteHMMModel(x.ptr),
          result)
    end
    return result
  end
end

mutable struct HoeffdingTreeModel
  ptr::Ptr{Nothing}

  # Construct object and set finalizer to free memory if `finalize` is true.
  function HoeffdingTreeModel(ptr::Ptr{Nothing}; finalize::Bool = false)::HoeffdingTreeModel
    result = new(ptr)
    if finalize
      finalizer(
          x -> _Internal.hoeffding_tree_internal.DeleteHoeffdingTreeModel(x.ptr),
          result)
    end
    return result
  end
end

mutable struct KDEModel
  ptr::Ptr{Nothing}

  # Construct object and set finalizer to free memory if `finalize` is true.
  function KDEModel(ptr::Ptr{Nothing}; finalize::Bool = false)::KDEModel
    result = new(ptr)
    if finalize
      finalizer(
          x -> _Internal.kde_internal.DeleteKDEModel(x.ptr),
          result)
    end
    return result
  end
end

mutable struct LARS
  ptr::Ptr{Nothing}

  # Construct object and set finalizer to free memory if `finalize` is true.
  function LARS(ptr::Ptr{Nothing}; finalize::Bool = false)::LARS
    result = new(ptr)
    if finalize
      finalizer(
          x -> _Internal.lars_internal.DeleteLARS(x.ptr),
          result)
    end
    return result
  end
end

mutable struct LinearSVMModel
  ptr::Ptr{Nothing}

  # Construct object and set finalizer to free memory if `finalize` is true.
  function LinearSVMModel(ptr::Ptr{Nothing}; finalize::Bool = false)::LinearSVMModel
    result = new(ptr)
    if finalize
      finalizer(
          x -> _Internal.linear_svm_internal.DeleteLinearSVMModel(x.ptr),
          result)
    end
    return result
  end
end

mutable struct LocalCoordinateCoding
  ptr::Ptr{Nothing}

  # Construct object and set finalizer to free memory if `finalize` is true.
  function LocalCoordinateCoding(ptr::Ptr{Nothing}; finalize::Bool = false)::LocalCoordinateCoding
    result = new(ptr)
    if finalize
      finalizer(
          x -> _Internal.local_coordinate_coding_internal.DeleteLocalCoordinateCoding(x.ptr),
          result)
    end
    return result
  end
end

mutable struct LogisticRegression
  ptr::Ptr{Nothing}

  # Construct object and set finalizer to free memory if `finalize` is true.
  function LogisticRegression(ptr::Ptr{Nothing}; finalize::Bool = false)::LogisticRegression
    result = new(ptr)
    if finalize
      finalizer(
          x -> _Internal.logistic_regression_internal.DeleteLogisticRegression(x.ptr),
          result)
    end
    return result
  end
end

mutable struct LSHSearch
  ptr::Ptr{Nothing}

  # Construct object and set finalizer to free memory if `finalize` is true.
  function LSHSearch(ptr::Ptr{Nothing}; finalize::Bool = false)::LSHSearch
    result = new(ptr)
    if finalize
      finalizer(
          x -> _Internal.lsh_internal.DeleteLSHSearch(x.ptr),
          result)
    end
    return result
  end
end

mutable struct NBCModel
  ptr::Ptr{Nothing}

  # Construct object and set finalizer to free memory if `finalize` is true.
  function NBCModel(ptr::Ptr{Nothing}; finalize::Bool = false)::NBCModel
    result = new(ptr)
    if finalize
      finalizer(
          x -> _Internal.nbc_internal.DeleteNBCModel(x.ptr),
          result)
    end
    return result
  end
end

mutable struct KNNModel
  ptr::Ptr{Nothing}

  # Construct object and set finalizer to free memory if `finalize` is true.
  function KNNModel(ptr::Ptr{Nothing}; finalize::Bool = false)::KNNModel
    result = new(ptr)
    if finalize
      finalizer(
          x -> _Internal.knn_internal.DeleteKNNModel(x.ptr),
          result)
    end
    return result
  end
end

mutable struct KFNModel
  ptr::Ptr{Nothing}

  # Construct object and set finalizer to free memory if `finalize` is true.
  function KFNModel(ptr::Ptr{Nothing}; finalize::Bool = false)::KFNModel
    result = new(ptr)
    if finalize
      finalizer(
          x -> _Internal.kfn_internal.DeleteKFNModel(x.ptr),
          result)
    end
    return result
  end
end

mutable struct PerceptronModel
  ptr::Ptr{Nothing}

  # Construct object and set finalizer to free memory if `finalize` is true.
  function PerceptronModel(ptr::Ptr{Nothing}; finalize::Bool = false)::PerceptronModel
    result = new(ptr)
    if finalize
      finalizer(
          x -> _Internal.perceptron_internal.DeletePerceptronModel(x.ptr),
          result)
    end
    return result
  end
end

mutable struct ScalingModel
  ptr::Ptr{Nothing}

  # Construct object and set finalizer to free memory if `finalize` is true.
  function ScalingModel(ptr::Ptr{Nothing}; finalize::Bool = false)::ScalingModel
    result = new(ptr)
    if finalize
      finalizer(
          x -> _Internal.preprocess_scale_internal.DeleteScalingModel(x.ptr),
          result)
    end
    return result
  end
end

mutable struct RandomForestModel
  ptr::Ptr{Nothing}

  # Construct object and set finalizer to free memory if `finalize` is true.
  function RandomForestModel(ptr::Ptr{Nothing}; finalize::Bool = false)::RandomForestModel
    result = new(ptr)
    if finalize
      finalizer(
          x -> _Internal.random_forest_internal.DeleteRandomForestModel(x.ptr),
          result)
    end
    return result
  end
end

mutable struct RAModel
  ptr::Ptr{Nothing}

  # Construct object and set finalizer to free memory if `finalize` is true.
  function RAModel(ptr::Ptr{Nothing}; finalize::Bool = false)::RAModel
    result = new(ptr)
    if finalize
      finalizer(
          x -> _Internal.krann_internal.DeleteRAModel(x.ptr),
          result)
    end
    return result
  end
end

mutable struct SoftmaxRegression
  ptr::Ptr{Nothing}

  # Construct object and set finalizer to free memory if `finalize` is true.
  function SoftmaxRegression(ptr::Ptr{Nothing}; finalize::Bool = false)::SoftmaxRegression
    result = new(ptr)
    if finalize
      finalizer(
          x -> _Internal.softmax_regression_internal.DeleteSoftmaxRegression(x.ptr),
          result)
    end
    return result
  end
end

mutable struct SparseCoding
  ptr::Ptr{Nothing}

  # Construct object and set finalizer to free memory if `finalize` is true.
  function SparseCoding(ptr::Ptr{Nothing}; finalize::Bool = false)::SparseCoding
    result = new(ptr)
    if finalize
      finalizer(
          x -> _Internal.sparse_coding_internal.DeleteSparseCoding(x.ptr),
          result)
    end
    return result
  end
end

mutable struct AdaBoostModel
  ptr::Ptr{Nothing}

  # Construct object and set finalizer to free memory if `finalize` is true.
  function AdaBoostModel(ptr::Ptr{Nothing}; finalize::Bool = false)::AdaBoostModel
    result = new(ptr)
    if finalize
      finalizer(
          x -> _Internal.adaboost_internal.DeleteAdaBoostModel(x.ptr),
          result)
    end
    return result
  end
end

mutable struct LinearRegression
  ptr::Ptr{Nothing}

  # Construct object and set finalizer to free memory if `finalize` is true.
  function LinearRegression(ptr::Ptr{Nothing}; finalize::Bool = false)::LinearRegression
    result = new(ptr)
    if finalize
      finalizer(
          x -> _Internal.linear_regression_internal.DeleteLinearRegression(x.ptr),
          result)
    end
    return result
  end
end

