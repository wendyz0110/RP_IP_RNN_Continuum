"""
Longer sequences → stronger temporal credit assignment demands: 12
Forces the RNN to rely more on hidden state structure
Amplifies interference effects (your core phenomenon)

Increase the number of base sequences (aka tasks learned): 6 
Each sequence more actions to choose from: 6
Keep 2000 samples per sequence

Model structure:
    Input 12
    Hidden: 14
    Output: 6
"""
# reuse original programs
from helpers.DataGenerator import generate_synthetic_data
from helpers.NetworkFunction import MotorLearningRNN
from helpers.TorchFunctions import train_evaluate_model

import os
import torch
import torch.nn as nn
import torch.optim as optim 
import numpy as np
from copy import deepcopy 
from joblib import Parallel, delayed 

# change in parameters

input_size = 12
hidden_size = 14
output_size = 6
batch_size = 20

num_training_sequences = 6
samples_per_sequence = 2000
num_pre_training_sequences = 14
num_test_sequences = 200

def make_block_schedule_indices(
    num_sequences=6,
    samples_per_sequence=2000,
    block_size=100,
    shuffle_within_sequence=True,
    random_block_order=False,
    seed=None
):
    rng = np.random.default_rng(seed)
    seq_indices = []

    for s in range(num_sequences):
        start = s * samples_per_sequence
        end = (s + 1) * samples_per_sequence
        idx = np.arange(start, end)
        if shuffle_within_sequence:
            rng.shuffle(idx)
        seq_indices.append(idx)

    pointers = np.zeros(num_sequences, dtype=int)
    schedule = []
    base_order = list(range(num_sequences))

    while np.any(pointers < samples_per_sequence):
        order = base_order.copy()
        if random_block_order:
            rng.shuffle(order)
        for s in order:
            start = pointers[s]
            end = min(start + block_size, samples_per_sequence)
            if start < samples_per_sequence:
                schedule.extend(seq_indices[s][start:end])
                pointers[s] = end

    return np.array(schedule)

def make_block_schedule_dataset(
    X_repetitive,
    y_repetitive,
    num_sequences=6,
    samples_per_sequence=2000,
    block_size=100,
    shuffle_within_sequence=True,
    random_block_order=False,
    seed=None
):
    schedule_idx = make_block_schedule_indices(
        num_sequences=num_sequences,
        samples_per_sequence=samples_per_sequence,
        block_size=block_size,
        shuffle_within_sequence=shuffle_within_sequence,
        random_block_order=random_block_order,
        seed=seed)

    return (
        X_repetitive[schedule_idx],
        y_repetitive[schedule_idx],
        schedule_idx)

# because we have 6 sequences now so sequence level retention is expanded
def sequence_level_retention(
    loss_retention_array,
    samples_per_sequence=2000,
    num_sequences=6
):
    """
    Compute average retention loss separately
    for each trained sequence.

    Assumes retention dataset is ordered:
        AAAA BBBB CCCC DDDD...

    Returns:
        seq_means:
            shape (num_sequences,)
    """

    seq_means = []

    for s in range(num_sequences):
        start = s * samples_per_sequence
        end = (s + 1) * samples_per_sequence
        seq_mean = np.mean(loss_retention_array[start:end])
        seq_means.append(seq_mean)

    return np.array(seq_means)

# =================================================
#                Model Runner
# =================================================

lr = 0.02

"""
The run_model_continuum() method performs serial block test and random block test at once
"""
def run_model_continuum(seed):

    np.random.seed(seed)
    torch.manual_seed(seed)

    input_size = 12
    hidden_size = 14
    output_size = 6
    batch_size = 20

    num_training_sequences = 6
    samples_per_sequence = 2000
    num_pre_training_sequences = 14
    num_test_sequences = 200

    block_sizes = [2000, 1000, 500, 250, 125, 50, 1]

    schedule_types = {
        "serial": False,         
        "random_block": True    
    }

    print(f"\n================ Seed {seed} ================")

    # --------------------------------------------------
    # Generate datasets
    # --------------------------------------------------

    # main training dataset (repetitive base dataset, schedule applied later)
    X_repetitive, y_repetitive, _, _ = generate_synthetic_data(
        num_sequences=num_training_sequences,
        samples_per_sequence=samples_per_sequence,
        sequence_length=input_size,
        num_actions=output_size,
        add_input_noise=True
    )

    # pre-training dataset
    X_pre, y_pre, _, _ = generate_synthetic_data(
        num_sequences=num_pre_training_sequences,
        samples_per_sequence=100,
        sequence_length=input_size,
        num_actions=output_size,
        add_input_noise=True
    )

    # test dataset: for novel sequences, repetitive vs interleaved does not matter
    _, _, X_test, y_test = generate_synthetic_data(
        num_sequences=num_test_sequences,
        samples_per_sequence=100,
        sequence_length=input_size,
        num_actions=output_size,
        add_input_noise=True
    )

    # --------------------------------------------------
    # Pre-train base model
    # --------------------------------------------------
    base_model = MotorLearningRNN(input_size, hidden_size, output_size, num_outputs=input_size)
    criterion = nn.CrossEntropyLoss()
    optimizer_pre = optim.SGD(base_model.parameters(), lr=lr, momentum=0.0)

    _, _, loss_test_pre, _, loss_test_array_pre = train_evaluate_model(
        X_pre, y_pre, X_pre, y_pre, X_test, y_test, base_model, criterion, optimizer_pre, batch_size=batch_size, is_dislplay_loss=False
    )

    # --------------------------------------------------
    # Run schedule conditions
    # --------------------------------------------------
    condition_results = {}

    for schedule_type, random_block_order_setting in schedule_types.items():

        print(f"\n======== Schedule type: {schedule_type} ========")

        condition_results[schedule_type] = {}

        for block_size in block_sizes:

            print(f"\nTraining {schedule_type}, block_size = {block_size}")

            # copy same pretrained model for each condition
            model = deepcopy(base_model)
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.0)

            # schedule seed offset
            schedule_seed = seed + block_size + (0 if schedule_type == "serial" else 100000)

            # generate scheduled datasets
            X_train_sched, y_train_sched, schedule_idx = make_block_schedule_dataset(
                X_repetitive, y_repetitive, num_sequences=num_training_sequences, samples_per_sequence=samples_per_sequence,
                block_size=block_size, shuffle_within_sequence=True, random_block_order=random_block_order_setting, seed=schedule_seed
            )

            loss_array, loss_retention, loss_test, loss_retention_array, loss_test_array = train_evaluate_model(
                X_train_sched, y_train_sched, X_repetitive, y_repetitive, X_test, y_test,
                model, criterion, optimizer, batch_size=batch_size, is_dislplay_loss=False
            )

            seq_retention_means = sequence_level_retention(
                loss_retention_array, samples_per_sequence=samples_per_sequence, num_sequences=num_training_sequences
            )

            retention_imbalance = np.std(seq_retention_means)
            
            """
            # --------------------------------------------------
            # Optional vulnerability tests at continuum endpoints
            # --------------------------------------------------
            if block_size in [2000, 1]:
                optimizer_vuln = optim.SGD(model.parameters(), lr=lr, momentum=0.0)

                loss_retention_noisy_array, loss_test_noisy_array, \
                loss_retention_pruned_array, loss_test_pruned_array, \
                loss_retention_interf_array, loss_test_interf_array = vulnerability_test(
                    X_repetitive,
                    y_repetitive,
                    X_test,
                    y_test,
                    model,
                    criterion,
                    optimizer_vuln,
                    num_repeat_noisy=30,
                    num_repeat_pruned=30,
                    num_interference_steps=30
                )
            else:
                loss_retention_noisy_array = None
                loss_test_noisy_array = None
                loss_retention_pruned_array = None
                loss_test_pruned_array = None
                loss_retention_interf_array = None
                loss_test_interf_array = None
            """
            
            condition_results[schedule_type][block_size] = {
                "loss_array": loss_array,
                "loss_retention": loss_retention,
                "loss_test": loss_test,
                "loss_retention_array": loss_retention_array,
                "loss_test_array": loss_test_array,
                "seq_retention_means": seq_retention_means,
                "retention_imbalance": retention_imbalance,
                "schedule_idx": schedule_idx
            }
                
    return {
        "seed": seed,
        "loss_test_pre": loss_test_pre,
        "loss_test_array_pre": loss_test_array_pre,
        "condition_results": condition_results
    }


# =================================================
#                Run in Parallel
# =================================================
num_runs = 10
results = Parallel(n_jobs=-1)(delayed(run_model_continuum)(1000 - i) for i in range(num_runs))

# =================================================
#                Aggregate Results
# =================================================

block_sizes = [2000, 1000, 500, 250, 125, 50, 1]
schedule_types = ["serial", "random_block"]

# pre-training results
loss_test_pre_list = [r["loss_test_pre"] for r in results]
loss_test_array_pre_list = [r["loss_test_array_pre"] for r in results]

loss_test_pre_array = np.array(loss_test_pre_list)
loss_test_array_pre_array = np.stack(loss_test_array_pre_list, axis=0)

# top-level dictionaries by schedule type
loss_array_by_schedule = {}
loss_retention_by_schedule = {}
loss_test_by_schedule = {}
loss_retention_array_by_schedule = {}
loss_test_array_by_schedule = {}
seq_retention_means_by_schedule = {}
retention_imbalance_by_schedule = {}
schedule_idx_by_schedule = {}

for schedule_type in schedule_types:

    loss_array_by_schedule[schedule_type] = {}
    loss_retention_by_schedule[schedule_type] = {}
    loss_test_by_schedule[schedule_type] = {}
    loss_retention_array_by_schedule[schedule_type] = {}
    loss_test_array_by_schedule[schedule_type] = {}
    seq_retention_means_by_schedule[schedule_type] = {}
    retention_imbalance_by_schedule[schedule_type] = {}
    schedule_idx_by_schedule[schedule_type] = {}

    for block_size in block_sizes:

        loss_array_list = [r["condition_results"][schedule_type][block_size]["loss_array"] for r in results]
        loss_retention_list = [r["condition_results"][schedule_type][block_size]["loss_retention"] for r in results]
        loss_test_list = [r["condition_results"][schedule_type][block_size]["loss_test"] for r in results]
        loss_retention_array_list = [r["condition_results"][schedule_type][block_size]["loss_retention_array"] for r in results]
        loss_test_array_list = [r["condition_results"][schedule_type][block_size]["loss_test_array"] for r in results]
        seq_retention_means_list = [r["condition_results"][schedule_type][block_size]["seq_retention_means"] for r in results]
        retention_imbalance_list = [r["condition_results"][schedule_type][block_size]["retention_imbalance"] for r in results]
        schedule_idx_list = [r["condition_results"][schedule_type][block_size]["schedule_idx"] for r in results]

        loss_array_by_schedule[schedule_type][block_size] = np.stack(loss_array_list, axis=0)
        loss_retention_by_schedule[schedule_type][block_size] = np.array(loss_retention_list)
        loss_test_by_schedule[schedule_type][block_size] = np.array(loss_test_list)
        loss_retention_array_by_schedule[schedule_type][block_size] = np.stack(loss_retention_array_list, axis=0)
        loss_test_array_by_schedule[schedule_type][block_size] = np.stack(loss_test_array_list, axis=0)
        seq_retention_means_by_schedule[schedule_type][block_size] = np.stack(seq_retention_means_list, axis=0)
        retention_imbalance_by_schedule[schedule_type][block_size] = np.array(retention_imbalance_list)
        schedule_idx_by_schedule[schedule_type][block_size] = np.stack(schedule_idx_list, axis=0)

# =================================================
#                Save Results
# =================================================

results_folder = f"results_advanced_continuum_lr_{str(lr).replace('.', '_')}"

if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# save raw results
np.save(results_folder + "/raw_results.npy", np.array(results, dtype=object), allow_pickle=True)

# save pre-training results
np.save(results_folder + "/loss_test_pre_array.npy", loss_test_pre_array)
np.save(results_folder + "/loss_test_array_pre_array.npy", loss_test_array_pre_array)

# save schedule-type dictionaries
np.save(results_folder + "/loss_array_by_schedule.npy", loss_array_by_schedule, allow_pickle=True)
np.save(results_folder + "/loss_retention_by_schedule.npy", loss_retention_by_schedule, allow_pickle=True)
np.save(results_folder + "/loss_test_by_schedule.npy", loss_test_by_schedule, allow_pickle=True)
np.save(results_folder + "/loss_retention_array_by_schedule.npy", loss_retention_array_by_schedule, allow_pickle=True)
np.save(results_folder + "/loss_test_array_by_schedule.npy", loss_test_array_by_schedule, allow_pickle=True)
np.save(results_folder + "/seq_retention_means_by_schedule.npy", seq_retention_means_by_schedule, allow_pickle=True)
np.save(results_folder + "/retention_imbalance_by_schedule.npy", retention_imbalance_by_schedule, allow_pickle=True)
np.save(results_folder + "/schedule_idx_by_schedule.npy", schedule_idx_by_schedule, allow_pickle=True)

print(f"\nResults saved to: {results_folder}")





















