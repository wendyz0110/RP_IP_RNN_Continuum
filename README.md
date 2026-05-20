# RP_IP_RNN_Continuum

An extension of the study ["A minimal recurrent neural network models the robustness of interleaved practice on motor sequence learning"](https://github.com/YJ-0000/RNN_during_RP_and_IP).

This project investigates how **practice structure** (repetitive/blocked vs. interleaved/random) affects motor sequence learning in a minimal RNN, and extends the original work by exploring a continuum of practice structures between these two extremes, as well as the effect of experience replay.

---

## Overview

A small RNN (7 input → 7 hidden → 4 output) is trained on synthetic 7-action motor sequences under two practice regimes:

- **Repetitive Practice (RP):** sequences presented in a blocked order (AAA BBB CCC)
- **Interleaved Practice (IP):** sequences presented in a random order (ABC BCA CAB)

After training, models are evaluated on retention of trained sequences and generalization to novel sequences, and tested for robustness to noise, weight pruning, and interference.

The project then extends this binary RP/IP comparison to a **continuum** of practice structures, and adds **replay** conditions.

---

## Repository Structure

```
helpers/                        # RNN model, data generator, training utilities
results_lr_0_02/                # Saved outputs for RP vs IP experiments (lr=0.02)
results_replay_lr_0_02/         # Saved outputs for replay experiments
results_block_continuum_*/      # Saved outputs for continuum experiments

main01_Experiment_Runner.py     # Run 100 RP vs IP experiments in parallel; save results
main02_plotting.py              # Plot RP vs IP results
main03_Replay.py                # Run replay experiments
main04_Replay_plotting.py       # Plot replay results
main05_Practice_Structure_Continuum.py   # Run continuum-of-practice experiments
main06_Continuum_Plotting.py    # Plot continuum results
main07_advanced_comtinuum.py    # Extended continuum experiments
main08_advanced_continuum_plotting.py    # Plot extended continuum results
```

---

## Requirements

- Python 3.x
- PyTorch
- NumPy
- joblib
- matplotlib (for plotting scripts)

Install dependencies:

```bash
pip install torch numpy joblib matplotlib
```

---

## Usage

Run experiments in order:

```bash
# 1. RP vs IP comparison (100 parallel runs)
python main01_Experiment_Runner.py

# 2. Plot RP vs IP results
python main02_plotting.py

# 3. Replay experiments
python main03_Replay.py
python main04_Replay_plotting.py

# 4. Practice structure continuum
python main05_Practice_Structure_Continuum.py
python main06_Continuum_Plotting.py

# 5. Advanced continuum
python main07_advanced_comtinuum.py
python main08_advanced_continuum_plotting.py
```

Results are saved as `.npy` files in the corresponding `results_*` directories.

---

## Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `lr` | `0.02` | SGD learning rate |
| `hidden_size` | `7` | RNN hidden units |
| `num_training_sequences` | `3` | Number of motor sequences trained |
| `num_pre_training_sequences` | `10` | Sequences for pretraining phase |
| `num_test_sequences` | `100` | Novel sequences for generalization test |
