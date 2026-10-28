"""
train_model.py
===============
Top-level training loop orchestrator for MedVision-AI.

The :class:`Trainer` glues together everything else in the module:

1. Loads a :class:`TrainingConfig` (YAML or programmatic).
2. Calls :func:`reproducibility.seed_everything` with the seed.
3. Instantiates the model, optimizer, scheduler, and (when applicable)
   the distributed process group.
4. Wires up :class:`data_validation.DataValidator` for a pre-training
   sanity check.
5. Runs the train/eval loop, saving checkpoints via
   :class:`checkpoint_manager.CheckpointManager`.
6. Optionally logs metrics & artifacts to MLflow through the
   :mod:`mlops.tracking` stack.

The trainer is intentionally model-agnostic: callers provide a *factory*
callable that returns a ``(model, optimizer, scheduler)`` triple given
the config. This keeps the trainer decoupled from specific architectures
(ResNet, ViT, etc.) and makes it equally usable for vision, signal, and
tabular pipelines.

Example
-------
    from mlops.training.training_config import TrainingConfig
    from mlops.training.train_model import Trainer, default_model_factory

    cfg = TrainingConfig.from_yaml("configs/baseline.yaml")
    trainer = Trainer(cfg, model_factory=default_model_factory)
    history = trainer.fit(train_loader, val_loader)
    print(history.best_metrics)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .checkpoint_manager import CheckpointManager, CheckpointState
from .data_validation import DataValidator
from .model_evaluation import ModelEvaluator
from .reproducibility import seed_everything
from .training_config import TrainingConfig
from .training_utils import AverageMeter

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------
# History
# --------------------------------------------------------------------


@dataclass
class TrainingHistory:
    """Aggregated metrics across epochs."""

    train: List[Dict[str, float]] = field(default_factory=list)
    val: List[Dict[str, float]] = field(default_factory=list)
    best_metrics: Dict[str, float] = field(default_factory=dict)
    best_epoch: int = 0
    final_epoch: int = 0
    total_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "train": list(self.train),
            "val": list(self.val),
            "best_metrics": dict(self.best_metrics),
            "best_epoch": self.best_epoch,
            "final_epoch": self.final_epoch,
            "total_seconds": self.total_seconds,
        }


# --------------------------------------------------------------------
# Model factory protocol
# --------------------------------------------------------------------


ModelFactory = Callable[[TrainingConfig], Tuple[Any, Any, Any]]


def default_model_factory(cfg: TrainingConfig) -> Tuple[Any, Any, Any]:
    """A minimal placeholder factory.

    Builds a small torch model so the trainer can run end-to-end in
    smoke tests without importing heavy vision backbones. Replace this
    with your real factory in production code.
    """
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("default_model_factory requires torch") from exc

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, cfg.model.num_classes),
    )

    opt_spec = cfg.optimizer
    if opt_spec.name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=opt_spec.lr, weight_decay=opt_spec.weight_decay)
    elif opt_spec.name == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=opt_spec.lr,
            momentum=opt_spec.momentum,
            weight_decay=opt_spec.weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {opt_spec.name!r}")

    if opt_spec.scheduler == "none":
        scheduler = None
    elif opt_spec.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs, **opt_spec.scheduler_params
        )
    elif opt_spec.scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=opt_spec.scheduler_params.get("step_size", 10),
            gamma=opt_spec.scheduler_params.get("gamma", 0.1),
        )
    elif opt_spec.scheduler == "exponential":
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=opt_spec.scheduler_params.get("gamma", 0.95)
        )
    else:
        raise ValueError(f"Unsupported scheduler: {opt_spec.scheduler!r}")
    return model, optimizer, scheduler


# --------------------------------------------------------------------
# Trainer
# --------------------------------------------------------------------


@dataclass
class Trainer:
    """Orchestrates a full training run from a :class:`TrainingConfig`.

    Attributes
    ----------
    config:
        The training configuration (loaded from YAML or built in code).
    model_factory:
        Callable returning ``(model, optimizer, scheduler)`` from the
        config. See :data:`default_model_factory` for an example.
    evaluator:
        Optional custom :class:`ModelEvaluator` (defaults to one that
        reports accuracy / F1 / AUC).
    validator:
        Optional :class:`DataValidator` run on the first batch before
        training begins.
    checkpoint_manager:
        Optional custom checkpoint manager; constructed from the
        config when not provided.
    loss_fn:
        Loss function used during training & evaluation. When not
        provided, falls back to ``nn.CrossEntropyLoss``.
    device:
        Torch device override (defaults to ``cuda`` if available).
    """

    config: TrainingConfig
    model_factory: ModelFactory
    evaluator: Optional[ModelEvaluator] = None
    validator: Optional[DataValidator] = None
    checkpoint_manager: Optional[CheckpointManager] = None
    loss_fn: Optional[Callable[[Any, Any], float]] = None
    device: Optional[str] = None

    _started: bool = field(default=False, init=False, repr=False)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        # Seed
        spec = self.config.reproducibility
        self.repro_state = seed_everything(
            seed=spec.seed,
            deterministic=spec.deterministic,
            cudnn_benchmark=spec.cudnn_benchmark,
            python_hash_seed=spec.python_hash_seed,
        )

    def fit(
        self,
        train_loader: Any,
        val_loader: Optional[Any] = None,
    ) -> TrainingHistory:
        """Run the full training loop and return a :class:`TrainingHistory`."""
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Trainer requires torch") from exc

        device = torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model, optimizer, scheduler = self.model_factory(self.config)
        model = model.to(device)

        loss_fn = self.loss_fn or torch.nn.CrossEntropyLoss()
        evaluator = self.evaluator or ModelEvaluator(device=str(device))
        ckpt_mgr = self.checkpoint_manager or CheckpointManager(
            checkpoint_dir=self.config.checkpoint.dir,
            monitor_metric=self.config.checkpoint.monitor_metric,
            monitor_mode=self.config.checkpoint.monitor_mode,
            save_top_k=self.config.checkpoint.save_top_k,
            save_every_n_epochs=self.config.checkpoint.save_every_n_epochs,
        )
        # Optionally resume from a checkpoint
        start_epoch = 0
        global_step = 0
        resume_path = self.config.checkpoint.resume_from
        if resume_path:
            state = CheckpointManager.load(
                resume_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                map_location=device,
            )
            start_epoch = state.epoch + 1
            global_step = state.global_step
            logger.info("Resumed from %s at epoch %d", resume_path, start_epoch)

        # Pre-training data validation
        if self.validator is not None:
            self._validate_loader(train_loader)

        history = TrainingHistory()
        best_score: Optional[float] = None
        best_metrics: Dict[str, float] = {}
        no_improve_epochs = 0
        t_run_start = time.time()

        for epoch in range(start_epoch, self.config.epochs):
            train_metrics = self._train_one_epoch(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                loss_fn=loss_fn,
                loader=train_loader,
                device=device,
                global_step_ref=[global_step],
            )
            history.train.append({**train_metrics, "epoch": float(epoch)})
            if scheduler is not None:
                scheduler.step()

            val_metrics: Dict[str, float] = {}
            if val_loader is not None and (
                epoch + 1
            ) % self.config.eval_every_n_epochs == 0:
                report = evaluator.evaluate(model, val_loader)
                val_metrics = dict(report.metrics)
                history.val.append({**val_metrics, "epoch": float(epoch)})

                # Checkpoint
                ckpt_mgr.save(
                    epoch=epoch,
                    global_step=global_step,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    metrics={**train_metrics, **val_metrics},
                    config=self.config.to_dict(),
                    extras={"reproducibility": self.repro_state.to_dict()},
                )

                # Best-so-far & early stopping
                monitor = self.config.checkpoint.monitor_metric
                if monitor in val_metrics:
                    score = val_metrics[monitor]
                    improved = (
                        best_score is None
                        or (self.config.checkpoint.monitor_mode == "min" and score < best_score)
                        or (self.config.checkpoint.monitor_mode == "max" and score > best_score)
                    )
                    if improved:
                        best_score = score
                        best_metrics = dict(val_metrics)
                        history.best_epoch = epoch
                        no_improve_epochs = 0
                    else:
                        no_improve_epochs += 1
                        if (
                            self.config.early_stopping_patience > 0
                            and no_improve_epochs >= self.config.early_stopping_patience
                        ):
                            logger.info(
                                "Early stopping triggered at epoch %d (no improvement for %d epochs)",
                                epoch,
                                no_improve_epochs,
                            )
                            break

            if self.config.max_steps is not None and global_step >= self.config.max_steps:
                logger.info("Reached max_steps=%d, stopping.", self.config.max_steps)
                break

        history.best_metrics = best_metrics
        history.final_epoch = epoch
        history.total_seconds = time.time() - t_run_start
        return history

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _train_one_epoch(
        self,
        epoch: int,
        model: Any,
        optimizer: Any,
        loss_fn: Any,
        loader: Any,
        device: Any,
        global_step_ref: List[int],
    ) -> Dict[str, float]:
        import torch

        model.train()
        loss_meter = AverageMeter()
        for step, batch in enumerate(loader):
            inputs, targets, _ = _unpack_batch(batch)
            inputs = _move(inputs, device)
            targets = _move(targets, device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            if self.config.optimizer.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.config.optimizer.grad_clip
                )
            optimizer.step()
            loss_meter.update(float(loss.item()), n=_batch_size(targets))
            global_step_ref[0] += 1
            if step % self.config.log_every_n_steps == 0:
                logger.info(
                    "epoch %d | step %d | loss %.4f",
                    epoch, global_step_ref[0], loss_meter.avg,
                )
        return {"loss": loss_meter.avg}

    def _validate_loader(self, loader: Any) -> None:
        samples = []
        try:
            for batch in loader:
                inputs, targets, _ = _unpack_batch(batch)
                samples.append({"image": inputs, "label": targets})
                if len(samples) >= 2:
                    break
        except Exception as exc:  # noqa: BLE001
            logger.warning("Pre-training data validation skipped: %s", exc)
            return
        report = self.validator.validate(samples)
        if not report.ok:
            report.raise_for_status()


# --------------------------------------------------------------------
# Free helpers (kept outside the dataclass for testability)
# --------------------------------------------------------------------


def _unpack_batch(batch: Any):
    if isinstance(batch, dict):
        return batch.get("image"), batch.get("label"), None
    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        return batch[0], batch[1], None
    return batch, None, None


def _move(obj: Any, device: Any) -> Any:
    from .training_utils import move_to_device

    return move_to_device(obj, device)


def _batch_size(targets: Any) -> int:
    try:
        return int(targets.shape[0])
    except Exception:  # noqa: BLE001
        return 1


__all__ = [
    "Trainer",
    "TrainingHistory",
    "ModelFactory",
    "default_model_factory",
]
