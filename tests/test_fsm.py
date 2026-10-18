import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import yaml
from src.curriculum.fsm_teacher import FSMTeacher, StudentPhase, TeacherAction


def test_fsm_transitions():
    dummy_cfg = {
        "curriculum": {
            "train_memorized_threshold": 0.98,
            "val_transition_threshold": 0.15,
            "val_generalized_threshold": 0.98,
            "patience": 3,
            "plateau": {
                "stagnation_window": 3,
                "min_norm_drop_rate": 0.1,
                "max_steps_in_plateau": 5000
            },
            "interventions": {
                "weight_decay_multiplier": 1.5,
                "max_weight_decay": 5.0,
                "lr_decay_on_stagnation": 0.8,
                "cooldown_steps": 2
            },
            "equilibrium_trap": {
                "stagnation_window": 3,
                "loss_epsilon": 0.01,
                "em_epsilon": 0.05
            }
        }
    }

    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0)
    teacher = FSMTeacher(dummy_cfg, initial_level=1, max_levels=3)

    # 1. Underfitting
    phase, action, _ = teacher.evaluate_step(100, 2.5, 0.40, 0.01, 50.0, optimizer)
    assert phase == StudentPhase.UNDERFITTING
    assert action == TeacherAction.CONTINUE

    # 2. Reaching Memorization with Healthy Norm Drop
    teacher.evaluate_step(200, 0.05, 1.0, 0.02, 45.0, optimizer)
    teacher.evaluate_step(300, 0.05, 1.0, 0.02, 40.0, optimizer)
    phase, action, meta = teacher.evaluate_step(400, 0.05, 1.0, 0.02, 36.0, optimizer)
    assert phase == StudentPhase.MEMORIZATION_PLATEAU
    assert action == TeacherAction.WAIT_AND_OBSERVE

    # Step 500 and 600: healthy compression (40.0 -> 35.0 drop = 5.0, 36.0 -> 35.0 drop = 1.0)
    teacher.evaluate_step(500, 0.05, 1.0, 0.02, 35.0, optimizer)
    teacher.evaluate_step(600, 0.05, 1.0, 0.02, 35.0, optimizer)

    # Step 700: Window is [35.0, 35.0, 35.0], delta = 0.0 < 0.1 -> Boost Regularization!
    phase, action, meta = teacher.evaluate_step(700, 0.05, 1.0, 0.02, 35.0, optimizer)
    assert action == TeacherAction.BOOST_REGULARIZATION
    assert optimizer.param_groups[0]["weight_decay"] == 1.5

    # Step 750: Cooldown active -> must NOT double-boost immediately!
    phase, action, meta = teacher.evaluate_step(750, 0.05, 1.0, 0.02, 35.0, optimizer)
    assert action == TeacherAction.WAIT_AND_OBSERVE
    assert optimizer.param_groups[0]["weight_decay"] == 1.5  # preserved, not 2.25

    # 3. Grokking Transition and Promotion
    teacher.evaluate_step(800, 0.01, 1.0, 0.99, 20.0, optimizer)
    teacher.evaluate_step(900, 0.01, 1.0, 0.99, 20.0, optimizer)
    phase, action, meta = teacher.evaluate_step(1000, 0.01, 1.0, 0.99, 20.0, optimizer)
    assert phase == StudentPhase.GENERALIZED
    assert action == TeacherAction.PROMOTE_NEXT_LEVEL
    assert teacher.current_level == 2

    print("All FSM transition tests have passed successfully!")


def test_equilibrium_trap_detection_and_recovery():
    dummy_cfg = {
        "curriculum": {
            "train_memorized_threshold": 0.95,
            "val_transition_threshold": 0.15,
            "val_generalized_threshold": 0.95,
            "patience": 3,
            "plateau": {
                "stagnation_window": 3,
                "min_norm_drop_rate": 0.1,
                "max_steps_in_plateau": 5000
            },
            "interventions": {
                "weight_decay_multiplier": 1.5,
                "max_weight_decay": 5.0,
                "lr_decay_on_stagnation": 0.8,
                "cooldown_steps": 2
            },
            "equilibrium_trap": {
                "stagnation_window": 3,
                "loss_epsilon": 0.01,
                "em_epsilon": 0.05
            }
        }
    }

    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=2.0)
    teacher = FSMTeacher(dummy_cfg, initial_level=1, max_levels=3)
    teacher.base_weight_decay = 1.0  # base is 1.0, but optimizer currently has 2.0 (boosted earlier)

    # Stagnant at ~75% EM with loss ~0.45 (Equilibrium Trap)
    teacher.evaluate_step(100, 0.452, 0.75, 0.04, 28.0, optimizer)
    teacher.evaluate_step(200, 0.451, 0.76, 0.04, 28.0, optimizer)
    phase, action, meta = teacher.evaluate_step(300, 0.450, 0.75, 0.04, 28.0, optimizer)

    assert phase == StudentPhase.EQUILIBRIUM_TRAP
    assert action == TeacherAction.REDUCE_REGULARIZATION
    assert optimizer.param_groups[0]["weight_decay"] == 1.0  # Reset back to base wd to allow memorization
    print("Equilibrium trap detection and recovery test passed successfully!")


def test_norm_floor_and_smooth_step_down():
    dummy_cfg = {
        "curriculum": {
            "train_memorized_threshold": 0.95,
            "val_transition_threshold": 0.15,
            "val_generalized_threshold": 0.90,
            "patience": 3,
            "plateau": {
                "stagnation_window": 3,
                "min_norm_drop_rate": 0.1,
                "min_norm_floor": 26.0,
                "max_steps_in_plateau": 5000
            },
            "interventions": {
                "weight_decay_multiplier": 1.3,
                "max_weight_decay": 2.0,
                "lr_decay_on_stagnation": 0.8,
                "cooldown_steps": 2
            },
            "equilibrium_trap": {
                "stagnation_window": 3,
                "loss_epsilon": 0.01,
                "em_epsilon": 0.05
            }
        }
    }

    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0)
    teacher = FSMTeacher(dummy_cfg, initial_level=1, max_levels=3)

    # 1. Memorized at norm 25.0 (below min_norm_floor 26.0)
    for step in [100, 200, 300, 400]:
        phase, action, meta = teacher.evaluate_step(step, 0.02, 0.99, 0.04, 25.0, optimizer)

    # Even though delta norm is 0, since norm is 25.0 <= 26.0, it should NOT boost!
    assert action == TeacherAction.WAIT_AND_OBSERVE
    assert meta.get("status") == "in_grokking_norm_zone"
    assert optimizer.param_groups[0]["weight_decay"] == 1.0

    # 2. Suppose weight decay had been boosted to 1.80, and train EM drops to 75%
    optimizer.param_groups[0]["weight_decay"] = 1.80
    teacher.had_memorized = True
    phase, action, meta = teacher.evaluate_step(500, 0.15, 0.75, 0.04, 25.0, optimizer)

    assert phase == StudentPhase.EQUILIBRIUM_TRAP
    assert action == TeacherAction.REDUCE_REGULARIZATION
    # Stepped down smoothly: 1.80 / 1.3 = 1.38, NOT hard reset to 1.00
    assert optimizer.param_groups[0]["weight_decay"] == 1.38
    assert teacher.had_memorized is True  # preserved since train_em >= 0.50
    print("Norm floor and smooth step-down test passed successfully!")


if __name__ == "__main__":
    test_fsm_transitions()
    test_equilibrium_trap_detection_and_recovery()
    test_norm_floor_and_smooth_step_down()