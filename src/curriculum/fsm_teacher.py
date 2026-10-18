from enum import Enum
from typing import Dict, Any, Tuple, Optional
import torch


class StudentPhase(str, Enum):
    UNDERFITTING = "UNDERFITTING"
    EQUILIBRIUM_TRAP = "EQUILIBRIUM_TRAP"
    MEMORIZATION_PLATEAU = "MEMORIZATION_PLATEAU"
    GROKKING_TRANSITION = "GROKKING_TRANSITION"
    GENERALIZED = "GENERALIZED"
    DIVERGED = "DIVERGED"


class TeacherAction(str, Enum):
    CONTINUE = "CONTINUE"
    WAIT_AND_OBSERVE = "WAIT_AND_OBSERVE"
    BOOST_REGULARIZATION = "BOOST_REGULARIZATION"
    REDUCE_REGULARIZATION = "REDUCE_REGULARIZATION"
    ESCAPE_EQUILIBRIUM = "ESCAPE_EQUILIBRIUM"
    CONFIRMING_STABILITY = "CONFIRMING_STABILITY"
    PROMOTE_NEXT_LEVEL = "PROMOTE_NEXT_LEVEL"
    ROLLBACK_LR = "ROLLBACK_LR"


class FSMTeacher:
    """
    Finite State Machine (FSM) Teacher that monitors student training dynamics,
    detects memorization, grokking transitions, and prevents/resolves
    stochastic mini-batch equilibrium traps.
    """

    def __init__(self, config: Dict[str, Any], initial_level: int = 1, max_levels: int = 4):
        self.cfg = config["curriculum"]
        self.current_level = initial_level
        self.max_levels = max_levels

        self.phase = StudentPhase.UNDERFITTING
        self.consecutive_generalized = 0
        self.steps_in_current_plateau = 0
        self.steps_in_current_level = 0
        self.intervention_cooldown = 0

        self.base_weight_decay: Optional[float] = None
        self.had_memorized = False

        self.norm_history: list[float] = []
        self.val_history: list[float] = []
        self.train_loss_history: list[float] = []
        self.train_em_history: list[float] = []

    def get_level_param(self, param_name: str, default: Any = None) -> Any:
        lvl_cfg = self.cfg.get("level_configs", {}).get(self.current_level, {})
        return lvl_cfg.get(param_name, default)

    def evaluate_step(
            self,
            step: int,
            train_loss: float,
            train_em: float,
            val_em: float,
            weight_norm: float,
            optimizer: torch.optim.Optimizer
    ) -> Tuple[StudentPhase, TeacherAction, Dict[str, Any]]:

        current_wd = optimizer.param_groups[0]["weight_decay"]
        if self.base_weight_decay is None:
            self.base_weight_decay = current_wd

        if torch.isnan(torch.tensor(train_loss)) or train_loss > 10.0:
            self.phase = StudentPhase.DIVERGED
            return self.phase, TeacherAction.ROLLBACK_LR, {"reason": "loss_explosion"}

        self.norm_history.append(weight_norm)
        self.val_history.append(val_em)
        self.train_loss_history.append(train_loss)
        self.train_em_history.append(train_em)

        self.steps_in_current_level += 1
        self.steps_in_current_plateau += 1

        if self.intervention_cooldown > 0:
            self.intervention_cooldown -= 1

        mem_thresh = float(self.get_level_param("train_memorized_threshold", self.cfg.get("train_memorized_threshold", 0.90)))
        val_trans_thresh = float(self.get_level_param("val_transition_threshold", self.cfg.get("val_transition_threshold", 0.15)))
        val_gen_thresh = float(self.get_level_param("val_generalized_threshold", self.cfg.get("val_generalized_threshold", 0.90)))

        is_memorized = train_em >= mem_thresh
        if is_memorized:
            self.had_memorized = True

        # Check for excessive regularization crushing train performance below 82% EM
        if self.had_memorized and train_em < 0.82 and current_wd > self.base_weight_decay:
            # Step down weight decay smoothly instead of hard collapsing to base,
            # preventing wild norm expansion / ping-pong cycles.
            multiplier = float(self.cfg["interventions"].get("weight_decay_multiplier", 1.3))
            stepped_down = current_wd / multiplier
            new_wd = max(self.base_weight_decay, round(stepped_down, 2))
            for g in optimizer.param_groups:
                g["weight_decay"] = new_wd

            self.phase = StudentPhase.EQUILIBRIUM_TRAP
            if train_em < 0.50:
                self.had_memorized = False
            self.intervention_cooldown = self.cfg["plateau"].get("stagnation_window", 10)
            return self.phase, TeacherAction.REDUCE_REGULARIZATION, {
                "status": "regularization_eased_step_down",
                "old_wd": current_wd,
                "new_wd": new_wd,
                "reason": f"Train EM dropped to {train_em:.2%} after boost; stepped down weight decay."
            }

        # Main Phase Logic
        if not is_memorized and not self.had_memorized and val_em < val_trans_thresh:
            self.consecutive_generalized = 0
            is_stagnant, trap_meta = self._check_equilibrium_trap(optimizer)
            if is_stagnant:
                self.phase = StudentPhase.EQUILIBRIUM_TRAP
                action, metadata = self._handle_equilibrium_trap(optimizer, trap_meta)
            else:
                self.phase = StudentPhase.UNDERFITTING
                action = TeacherAction.CONTINUE
                metadata = {}

        elif (is_memorized or self.had_memorized) and val_em < val_trans_thresh:
            self.phase = StudentPhase.MEMORIZATION_PLATEAU
            self.consecutive_generalized = 0
            action, metadata = self._handle_plateau(optimizer, train_em)

        elif val_em >= val_gen_thresh:
            self.phase = StudentPhase.GENERALIZED
            self.consecutive_generalized += 1
            action, metadata = self._handle_generalization()

        elif self.consecutive_generalized > 0 and val_em >= (val_gen_thresh - 0.03):
            # Soft tolerance: slight fluctuation does not break an established generalization streak
            self.phase = StudentPhase.GENERALIZED
            self.consecutive_generalized += 1
            action, metadata = self._handle_generalization()

        elif val_trans_thresh <= val_em < val_gen_thresh:
            self.phase = StudentPhase.GROKKING_TRANSITION
            self.consecutive_generalized = 0
            self.steps_in_current_plateau = 0
            action = TeacherAction.WAIT_AND_OBSERVE
            metadata = {"info": "Phase transition active. Generalization in progress."}

        return self.phase, action, metadata

    def _check_equilibrium_trap(self, optimizer: torch.optim.Optimizer) -> Tuple[bool, Dict[str, Any]]:
        """
        Detects if training is trapped in an equilibrium where stochastic mini-batch
        gradients balance regularization or noise, halting progress below the memorization threshold.
        """
        trap_cfg = self.cfg.get("equilibrium_trap", {})
        window = trap_cfg.get("stagnation_window", self.cfg["plateau"].get("stagnation_window", 10))
        loss_eps = trap_cfg.get("loss_epsilon", 0.008)
        em_eps = trap_cfg.get("em_epsilon", 0.04)

        if len(self.train_loss_history) < window:
            return False, {}

        recent_losses = self.train_loss_history[-window:]
        recent_ems = self.train_em_history[-window:]

        loss_drop = recent_losses[0] - recent_losses[-1]
        loss_spread = max(recent_losses) - min(recent_losses)
        em_spread = max(recent_ems) - min(recent_ems)

        mean_em = sum(recent_ems) / len(recent_ems)
        # If the student is performing with high accuracy (>= 80%), this is normal
        # memorization compression, NOT an equilibrium trap!
        if mean_em >= 0.80:
            return False, {}

        # Equilibrium trap is characterized by:
        # 1. No significant decrease in train loss over the stagnation window
        # 2. Train EM hovering in a low/suboptimal band without approaching memorization
        is_trapped = (loss_drop < loss_eps) and (loss_spread < 3 * loss_eps) and (em_spread < em_eps)

        max_steps = self.cfg["plateau"].get("max_steps_in_plateau", 10000)
        if self.steps_in_current_plateau > max_steps:
            is_trapped = True

        return is_trapped, {
            "loss_drop": round(loss_drop, 4),
            "loss_spread": round(loss_spread, 4),
            "em_spread": round(em_spread, 4),
            "mean_em": round(sum(recent_ems) / len(recent_ems), 4),
            "current_wd": optimizer.param_groups[0]["weight_decay"]
        }

    def _handle_equilibrium_trap(
            self,
            optimizer: torch.optim.Optimizer,
            trap_meta: Dict[str, Any]
    ) -> Tuple[TeacherAction, Dict[str, Any]]:
        """
        Intervention strategy when trapped in a stochastic equilibrium:
        1. If weight decay was boosted above base, reduce it back to allow fitting.
        2. If already at base weight decay and still stuck, signal to escape equilibrium
           (increase effective batch size / gradient accumulation / reduce WD further).
        """
        if self.intervention_cooldown > 0:
            return TeacherAction.WAIT_AND_OBSERVE, {"status": "cooldown_active_in_trap"}

        current_wd = optimizer.param_groups[0]["weight_decay"]
        cooldown = self.cfg["plateau"].get("stagnation_window", 10)

        if current_wd > self.base_weight_decay:
            # Regularization was excessive for the stochastic mini-batch size
            new_wd = self.base_weight_decay
            for g in optimizer.param_groups:
                g["weight_decay"] = new_wd
            self.intervention_cooldown = cooldown
            return TeacherAction.REDUCE_REGULARIZATION, {
                "action": "reset_to_base_wd",
                "old_wd": current_wd,
                "new_wd": new_wd,
                "reason": "Equilibrium trap: excessive weight decay preventing full fitting."
            }
        else:
            # Stuck at base weight decay: stochastic mini-batch noise is preventing convergence
            self.intervention_cooldown = cooldown
            return TeacherAction.ESCAPE_EQUILIBRIUM, {
                "action": "escape_stochastic_equilibrium",
                "current_wd": current_wd,
                "suggested_fix": "increase_batch_size_or_gradient_accumulation",
                "reason": "Equilibrium trap: mini-batch stochastic noise prevents reaching memorization threshold."
            }

    def _handle_plateau(self, optimizer: torch.optim.Optimizer, train_em: float) -> Tuple[TeacherAction, Dict[str, Any]]:
        is_grokking = bool(self.get_level_param("is_grokking_task", self.current_level in [1, 2]))
        if not is_grokking:
            # For non-grokking (standard algorithmic seq2seq) tasks, do NOT boost weight decay!
            return TeacherAction.WAIT_AND_OBSERVE, {"status": "algorithmic_task_fitting"}

        window = self.cfg["plateau"]["stagnation_window"]
        min_drop = self.cfg["plateau"]["min_norm_drop_rate"]

        if self.intervention_cooldown > 0:
            return TeacherAction.WAIT_AND_OBSERVE, {"status": "cooldown_after_intervention"}

        if len(self.norm_history) < window:
            return TeacherAction.WAIT_AND_OBSERVE, {"status": "accumulating_window"}

        recent_norms = self.norm_history[-window:]
        norm_delta = recent_norms[0] - recent_norms[-1]

        if norm_delta >= min_drop:
            return TeacherAction.WAIT_AND_OBSERVE, {
                "norm_drop": round(norm_delta, 3),
                "status": "healthy_compression"
            }

        # Check if weight norm has already reached the minimal complexity sweet spot
        min_norm_floor = float(self.cfg["plateau"].get("min_norm_floor", 26.0))
        if recent_norms[-1] <= min_norm_floor:
            return TeacherAction.WAIT_AND_OBSERVE, {
                "status": "in_grokking_norm_zone",
                "current_norm": round(recent_norms[-1], 2),
                "norm_floor": min_norm_floor,
                "info": "Weight norm is in optimal grokking basin; waiting for representation rotation."
            }

        # If train EM is not securely above solid fitting, do not boost weight decay
        if train_em < self.cfg.get("train_memorized_threshold", 0.98) - 0.05:
            return TeacherAction.CONTINUE, {"status": "waiting_for_solid_fitting"}

        current_wd = optimizer.param_groups[0]["weight_decay"]
        max_wd = float(self.get_level_param("max_weight_decay", self.cfg["interventions"]["max_weight_decay"]))

        if current_wd < max_wd:
            multiplier = self.cfg["interventions"]["weight_decay_multiplier"]
            new_wd = min(current_wd * multiplier, max_wd)

            for param_group in optimizer.param_groups:
                param_group["weight_decay"] = new_wd

            cooldown = self.cfg["interventions"].get("cooldown_steps", window)
            self.intervention_cooldown = cooldown

            return TeacherAction.BOOST_REGULARIZATION, {
                "old_wd": current_wd,
                "new_wd": new_wd,
                "cooldown": cooldown,
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
            self.steps_in_current_plateau = 0
            self.steps_in_current_level = 0
            self.intervention_cooldown = 0
            self.had_memorized = False
            self.norm_history.clear()
            self.val_history.clear()
            self.train_loss_history.clear()
            self.train_em_history.clear()

            return TeacherAction.PROMOTE_NEXT_LEVEL, {
                "promoted_to_level": self.current_level
            }

        return TeacherAction.CONFIRMING_STABILITY, {"status": "curriculum_mastered"}