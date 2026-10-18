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
                "lr_decay_on_stagnation": 0.8
            }
        }
    }

    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0)
    teacher = FSMTeacher(dummy_cfg, initial_level=1, max_levels=3)

    phase, action, _ = teacher.evaluate_step(100, 2.5, 0.40, 0.01, 50.0, optimizer)
    assert phase == StudentPhase.UNDERFITTING
    assert action == TeacherAction.CONTINUE

    teacher.evaluate_step(200, 0.05, 1.0, 0.02, 45.0, optimizer)
    teacher.evaluate_step(300, 0.05, 1.0, 0.02, 40.0, optimizer)
    phase, action, meta = teacher.evaluate_step(400, 0.05, 1.0, 0.02, 35.0, optimizer)
    assert phase == StudentPhase.MEMORIZATION_PLATEAU
    assert action == TeacherAction.WAIT_AND_OBSERVE

    teacher.evaluate_step(500, 0.05, 1.0, 0.02, 35.0, optimizer)
    teacher.evaluate_step(600, 0.05, 1.0, 0.02, 35.0, optimizer)
    phase, action, meta = teacher.evaluate_step(700, 0.05, 1.0, 0.02, 35.0, optimizer)
    assert action == TeacherAction.BOOST_REGULARIZATION
    assert optimizer.param_groups[0]["weight_decay"] == 1.5

    teacher.evaluate_step(800, 0.01, 1.0, 0.99, 20.0, optimizer)
    teacher.evaluate_step(900, 0.01, 1.0, 0.99, 20.0, optimizer)
    phase, action, meta = teacher.evaluate_step(1000, 0.01, 1.0, 0.99, 20.0, optimizer)
    assert phase == StudentPhase.GENERALIZED
    assert action == TeacherAction.PROMOTE_NEXT_LEVEL
    assert teacher.current_level == 2

    print("All FSM transition tests have passed successfully!")

if __name__ == "__main__":
    test_fsm_transitions()