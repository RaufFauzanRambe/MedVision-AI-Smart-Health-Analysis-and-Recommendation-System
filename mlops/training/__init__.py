"""MedVision-AI training module.

Public API:
    from mlops.training import (
        TrainingConfig,
        Trainer,
        CheckpointManager,
        ModelEvaluator,
        DataValidator,
        HyperparameterTuner,
        seed_everything,
        DistributedContext,
    )
"""

from .training_config import (
    CheckpointSpec,
    DatasetSpec,
    DistributedSpec,
    ModelSpec,
    OptimizerSpec,
    ReproducibilitySpec,
    TrainingConfig,
)
from .training_utils import AverageMeter, format_eta, humanize_bytes
from .reproducibility import ReproducibilityState, seed_everything, seed_worker
from .data_validation import (
    DataValidationError,
    DataValidator,
    ValidationReport,
    image_shape_rule,
    label_in_range_rule,
    non_null_rule,
)
from .model_evaluation import EvaluationReport, ModelEvaluator
from .checkpoint_manager import CheckpointManager, CheckpointState
from .train_model import Trainer, TrainingHistory, default_model_factory
from .distributed_training import DistributedContext, distributed
from .hyperparameter_tuning import (
    HyperparameterTuner,
    TrialResult,
    TuningReport,
    suggest_batch_size,
    suggest_dropout,
    suggest_learning_rate,
    suggest_optimizer,
    suggest_weight_decay,
)

__all__ = [
    # training_config
    "TrainingConfig",
    "ModelSpec",
    "DatasetSpec",
    "OptimizerSpec",
    "DistributedSpec",
    "CheckpointSpec",
    "ReproducibilitySpec",
    # training_utils
    "AverageMeter",
    "format_eta",
    "humanize_bytes",
    # reproducibility
    "ReproducibilityState",
    "seed_everything",
    "seed_worker",
    # data_validation
    "DataValidator",
    "ValidationReport",
    "DataValidationError",
    "image_shape_rule",
    "label_in_range_rule",
    "non_null_rule",
    # model_evaluation
    "ModelEvaluator",
    "EvaluationReport",
    # checkpoint_manager
    "CheckpointManager",
    "CheckpointState",
    # train_model
    "Trainer",
    "TrainingHistory",
    "default_model_factory",
    # distributed_training
    "DistributedContext",
    "distributed",
    # hyperparameter_tuning
    "HyperparameterTuner",
    "TrialResult",
    "TuningReport",
    "suggest_learning_rate",
    "suggest_batch_size",
    "suggest_optimizer",
    "suggest_dropout",
    "suggest_weight_decay",
]
