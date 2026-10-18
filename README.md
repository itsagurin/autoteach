# 🎓 AutoTeach: Autonomous Curriculum Grokking Engine

**AutoTeach** is an autonomous curriculum learning and grokking research framework for micro-transformers. It trains small causal language models to transition from memorization to algorithmic generalization across a progressive ladder of mathematical tasks.

---

## 🌟 Key Features

* **MicroNanoGPT Architecture**: Efficient 2-layer, 4-head causal Transformer with weight tying, configurable context, and fast Mac MPS / CUDA support.
* **4-Level Curriculum Ladder**:
  1. **Level 1 (`MODULAR_ADD_P97`)**: Modular addition ($a + b \pmod{97}$) with stochastic mini-batch grokking.
  2. **Level 2 (`MODULAR_MULT_P97`)**: Modular multiplication ($a \times b \pmod{97}$) with full-batch discrete Fourier representation emergence.
  3. **Level 3 (`MULTI_DIGIT_ADDITION`)**: Multi-digit addition with carry propagation (2–3 digits).
  4. **Level 4 (`SCRATCHPAD_COMPOSITE_ARITHMETIC`)**: Multi-step composite arithmetic ($a \times b + c$) using Chain-of-Thought (CoT) scratchpads.
* **FSM Teacher (Finite State Machine)**:
  - Real-time phase tracking: `UNDERFITTING` ➔ `MEMORIZATION_PLATEAU` ➔ `GROKKING_TRANSITION` ➔ `GENERALIZED`.
  - **Equilibrium Trap Resilience**: Detects and breaks out of stochastic mini-batch equilibrium traps.
  - **Dynamic Regularization**: Adaptive weight decay with norm floor sweet-spot protection ($\|W\|_2 \le 26.0$) and smooth step-downs to prevent training collapse.
* **Anti-Catastrophic Forgetting**:
  - `ExperienceReplayBuffer` mixes historical task replay into active training.
  - Real-time retention scoring across validation splits of all previously mastered levels.

---

## 📁 Project Structure

```text
autoteach/
├── configs/
│   ├── student_config.yaml       # Model and optimizer parameters
│   ├── curriculum_config.yaml    # FSM thresholds, patience, and level configs
│   └── optimizer_config.yaml     # Learning rate & scheduler rules
├── src/
│   ├── curriculum/
│   │   └── fsm_teacher.py        # FSM Teacher monitoring & interventions
│   ├── engine/
│   │   ├── trainer.py            # StudentTrainer & EpochBatchSampler
│   │   └── masking.py            # Prompt/answer loss masking & batching
│   ├── models/
│   │   ├── nanogpt.py            # MicroNanoGPT causal transformer
│   │   └── tokenizer.py          # Character-level MathTokenizer
│   ├── tasks/
│   │   ├── modular.py            # Modular addition and multiplication
│   │   ├── multi_digit.py        # Multi-digit addition
│   │   ├── scratchpad.py         # Composite scratchpad arithmetic
│   │   └── replay_buffer.py      # Experience replay & retention evaluation
│   └── telemetry/
│       ├── grok_detector.py      # Grokking & equilibrium trap telemetry
│       └── metrics.py            # Weight norm, Exact Match (EM), RollingMetric
├── scripts/
│   ├── run_loop.py               # Main autonomous curriculum loop
│   ├── eval_checkpoint.py        # Checkpoint evaluation script
│   └── manual_grok_check.py      # Quick manual grokking diagnostic
└── tests/
    ├── test_fsm.py               # FSM state transitions & trap recovery tests
    ├── test_model.py             # Model forward pass, loss, and masking tests
    └── test_tasks.py             # Data integrity, tokenization, & replay tests
```

---

## 🚀 Quickstart

### 1. Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Autonomous Curriculum Loop

Start training from Level 1:
```bash
python scripts/run_loop.py
```

### 3. Resume from Checkpoints

Resume automatically from the latest saved level checkpoint (e.g., jump directly into Level 2):
```bash
python scripts/run_loop.py --resume
```

Or specify an exact checkpoint file:
```bash
python scripts/run_loop.py --checkpoint checkpoints/checkpoint_level_1.pt
```

### 4. Run Test Suite

```bash
python tests/test_tasks.py
python tests/test_model.py
python tests/test_fsm.py
```

---

## 📊 Telemetry Output

During execution, `run_loop.py` prints live telemetry:

```text
Step    | Level | Phase                | Train EM  | Val EM    | ||W||_2 | Action         
-------------------------------------------------------------------------------------
1400    | L1    | UNDERFITTING         | 16.87%    | 13.58%    | 29.92   | CONTINUE       
1600    | L1    | GROKKING_TRANSITION  | 40.92%    | 35.33%    | 27.32   | WAIT_AND_OBSERVE
7600    | L1    | GENERALIZED          | 96.39%    | 93.20%    | 21.65   | PROMOTE_NEXT_LEVEL
  └─► [EXPERIENCE REPLAY RETENTION] L1: 94.2%
```

---

## 📜 License

MIT License.
