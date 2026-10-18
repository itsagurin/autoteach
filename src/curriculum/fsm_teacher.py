from enum import Enum
from typing import Dict, Any, Tuple, Optional
import torch


class StudentPhase(str, Enum):
    UNDERFITTING = "UNDERFITTING"
    MEMORIZATION_PLATEAU = "MEMORIZATION_PLATEAU"
    GROKKING_TRANSITION = "GROKKING_TRANSITION"
    GENERALIZED = "GENERALIZED"
    DIVERGED = "DIVERGED"


class TeacherAction(str, Enum):
    CONTINUE = "CONTINUE"
    WAIT_AND_OBSERVE = "WAIT_AND_OBSERVE"
    BOOST_REGULARIZATION = "BOOST_REGULARIZATION"
    CONFIRMING_STABILITY = "CONFIRMING_STABILITY"
    PROMOTE_NEXT_LEVEL = "PROMOTE_NEXT_LEVEL"
    ROLLBACK_LR = "ROLLBACK_LR"


class FSMTeacher:

    def __init__(self, config: Dict[str, Any], initial_level: int = 1, max_levels: int = 4):
        self.cfg = config["curriculum"]
        self.current_level = initial_level
        self.max_levels = max_levels

        self.phase = StudentPhase.UNDERFITTING
        self.consecutive_generalized = 0
        self.steps_in_current_plateau = 0

        self.norm_history: list[float] = []
        self.val_history: list[float] = []

    def evaluate_step(
            self,
            step: int,
            train_loss: float,
            train_em: float,
            val_em: float,
            weight_norm: float,
            optimizer: torch.optim.Optimizer
    ) -> Tuple[StudentPhase, TeacherAction, Dict[str, Any]]:

        if torch.isnan(torch.tensor(train_loss)) or train_loss > 10.0:
            self.phase = StudentPhase.DIVERGED
            return self.phase, TeacherAction.ROLLBACK_LR, {"reason": "loss_explosion"}

        self.norm_history.append(weight_norm)
        self.val_history.append(val_em)
        metadata: Dict[str, Any] = {}

        if train_em < self.cfg["train_memorized_threshold"]:
            self.phase = StudentPhase.UNDERFITTING
            self.consecutive_generalized = 0
            self.steps_in_current_plateau = 0
            action = TeacherAction.CONTINUE

        elif val_em < self.cfg["val_transition_threshold"]:
            self.phase = StudentPhase.MEMORIZATION_PLATEAU
            self.consecutive_generalized = 0
            self.steps_in_current_plateau += 1
            action, metadata = self._handle_plateau(optimizer)

        elif self.cfg["val_transition_threshold"] <= val_em < self.cfg["val_generalized_threshold"]:
            self.phase = StudentPhase.GROKKING_TRANSITION
            self.consecutive_generalized = 0
            self.steps_in_current_plateau = 0
            action = TeacherAction.WAIT_AND_OBSERVE
            metadata = {"info": "Phase transition active. Do not interfere."}

        else:
            self.phase = StudentPhase.GENERALIZED
            self.consecutive_generalized += 1
            self.steps_in_current_plateau = 0
            action, metadata = self._handle_generalization()

        return self.phase, action, metadata

    def _handle_plateau(self, optimizer: torch.optim.Optimizer) -> Tuple[TeacherAction, Dict[str, Any]]:
        window = self.cfg["plateau"]["stagnation_window"]
        min_drop = self.cfg["plateau"]["min_norm_drop_rate"]

        if len(self.norm_history) < window:
            return TeacherAction.WAIT_AND_OBSERVE, {"status": "accumulating_window"}

        recent_norms = self.norm_history[-window:]
        norm_delta = recent_norms[0] - recent_norms[-1]

        if norm_delta >= min_drop:
            return TeacherAction.WAIT_AND_OBSERVE, {
                "norm_drop": round(norm_delta, 3),
                "status": "healthy_compression"
            }

        current_wd = optimizer.param_groups[0]["weight_decay"]
        max_wd = self.cfg["interventions"]["max_weight_decay"]

        if current_wd < max_wd:
            multiplier = self.cfg["interventions"]["weight_decay_multiplier"]
            new_wd = min(current_wd * multiplier, max_wd)

            for param_group in optimizer.param_groups:
                param_group["weight_decay"] = new_wd

            return TeacherAction.BOOST_REGULARIZATION, {
                "old_wd": current_wd,
                "new_wd": new_wd,
                "reason": f"Norm delta {norm_delta:.3f} < threshold {min_drop}"
            }

        return TeacherAction.WAIT_AND_OBSERVE, {"status": "max_regularization_reached"}

    def _handle_generalization(self) -> Tuple[TeacherAction, Dict[str, Any]]:
        required_patience = self.cfg["patience"]

        if self.consecutive_generalized < required_patience:
            return TeacherAction.CONFIRMING_STABILITY, {
                "confirmed_streak": self.consecutive_generalized,
                "target_streak": required_patience
            }

        if self.current_level < self.max_levels:
            self.current_level += 1
            self.consecutive_generalized = 0
            self.norm_history.clear()
            self.val_history.clear()
            return TeacherAction.PROMOTE_NEXT_LEVEL, {
                "promoted_to_level": self.current_level
            }

        return TeacherAction.CONFIRMING_STABILITY, {"status": "curriculum_mastered"}