
#=================================================================
#                   Practice Structure Continuum
#    "The original study compares two endpoints: RP, where one task 
# is practiced repeatedly before switching, and IP, where tasks switch 
# frequently. This continuum extension asks what happens between these 
# two end points.

# The continuum is achieved by manipulating a block size parameter:
# block_size = number of consecutive trials from the same sequence before switching
#=================================================================

# try pair replay with different block size, see if offline replay
# compensate for online schedule differences
# consider point out the 3 base sequences to check if they are similar?

import os
import torch
import torch.nn as nn
import torch.optim as optim 
import numpy as np
from copy import deepcopy 
from helpers.NetworkFunction import MotorLearningRNN
from helpers.DataGenerator import generate_synthetic_data
from helpers.TorchFunctions import train_evaluate_model
from joblib import Parallel, delayed 

 # =================================================
 #                Helper Functions
 # =================================================
 
"""
make_block_schedule_indices:
    
    Generate indices for shuffling the X_repetitive and y_repetitive datasets 
    according to block based practice schedules.
    This avoids the need to generate new dataset per block size
"""
def make_block_schedule_indices(
    num_sequences=3,
    samples_per_sequence=1000,
    block_size=100,
    shuffle_within_sequence=True,
    random_block_order=False, # controls blocks are serial or random
    seed=None
):
    # random number generator
    rng = np.random.default_rng(seed)

    # indices for each sequence 
    seq_indices = []
    for s in range(num_sequences):
        start = s * samples_per_sequence
        end = (s + 1) * samples_per_sequence
        idx = np.arange(start, end)

        if shuffle_within_sequence: # shuffle noisy variants for robustness
            rng.shuffle(idx)

        seq_indices.append(idx) # store each sequence's index list

    # pointer into each sequence's index list
    pointers = np.zeros(num_sequences, dtype=int) # [0, 0, 0] tracks num of samples used from each sequence
    schedule = [] # store final training order

    base_order = list(range(num_sequences)) # 1 then 2, then 3

    while np.any(pointers < samples_per_sequence):
        order = base_order.copy()

        if random_block_order: # if false, serial 1 2 3 1 2 3, otherwise random
            rng.shuffle(order)

        for s in order:
            start = pointers[s]
            end = min(start + block_size, samples_per_sequence) # prevent out of range

            if start < samples_per_sequence:
                # # add one block from each sequence based on order
                schedule.extend(seq_indices[s][start:end])
                pointers[s] = end

    return np.array(schedule)

"""
make_block_schedule_dataset:
    Wrapper function around make_block_schedule_indices
    Based on generated indices, generate the blocked training dataset
"""
def make_block_schedule_dataset(
    X_repetitive,
    y_repetitive,
    num_sequences=3,
    samples_per_sequence=1000,
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
        seed=seed
    )

    # reorder and return
    return X_repetitive[schedule_idx], y_repetitive[schedule_idx], schedule_idx

"""
Sequence level retention helper:
    Computes average retention loss separately for each trained sequence
    Assumes retention array is ordered as seq0, seq1, seq2
"""
def sequence_level_retention(
        loss_retention_array, 
        samples_per_sequence=1000, 
        num_sequences=3
):
    seq_means = []

    for s in range(num_sequences):
        start = s * samples_per_sequence
        end = (s + 1) * samples_per_sequence
        seq_means.append(np.mean(loss_retention_array[start:end]))

    return np.array(seq_means)

 # =================================================
 #                Model Runner
 # =================================================

lr = 0.02

"""
The run_model_continuum( ) method performs serial block test and random block test at once
"""
def run_model_continuum(seed):
    
    np.random.seed(seed)
    torch.manual_seed(seed)

    input_size = 7
    hidden_size = 7
    output_size = 4
    batch_size = 20

    num_training_sequences = 3
    samples_per_sequence = 1000
    num_pre_training_sequences = 10
    num_test_sequences = 100

    block_sizes = [1000, 500, 200, 100, 50, 20, 10, 5, 1]

    schedule_types = {"serial": False,        # fixed A-B-C-A-B-C order
                      "random_block": True}    # randomize block order within each cycle

    print(f"\n================ Seed {seed} ================")

    # generate training dataset (no schedule yet)
    X_repetitive, y_repetitive, _, _ = generate_synthetic_data(num_sequences=num_training_sequences,samples_per_sequence=samples_per_sequence,add_input_noise=True)
    # pre train dataset
    X_pre, y_pre, _, _ = generate_synthetic_data(num_sequences=num_pre_training_sequences,samples_per_sequence=100,add_input_noise=True)
    # test dataset
    _, _, X_test, y_test = generate_synthetic_data(num_sequences=num_test_sequences,add_input_noise=True)

    # pretrain model
    base_model = MotorLearningRNN(input_size,hidden_size,output_size,num_outputs=input_size)
    criterion = nn.CrossEntropyLoss()
    optimizer_pre = optim.SGD(base_model.parameters(), lr=lr, momentum=0.0)
    # pre train the model
    _, _, loss_test_pre, _, loss_test_array_pre = train_evaluate_model(X_pre, y_pre,X_pre, y_pre,X_test, y_test,base_model,criterion,optimizer_pre,is_dislplay_loss=False)

    # condition_results has two top-level keys:
    # condition_results["serial"]
    # condition_results["random_block"]
    condition_results = {}

    for schedule_type, random_block_order_setting in schedule_types.items():

        print(f"\n======== Schedule type: {schedule_type} ========")

        condition_results[schedule_type] = {}

        for block_size in block_sizes:
            print(f"\nTraining {schedule_type}, block_size = {block_size}")

            # copy the same pretrained model for each condition
            model = deepcopy(base_model)
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.0)

            # seed offset
            schedule_seed = seed + block_size + (0 if schedule_type == "serial" else 100000)

            X_train_sched, y_train_sched, schedule_idx = make_block_schedule_dataset(
                X_repetitive,
                y_repetitive,
                num_sequences=num_training_sequences,
                samples_per_sequence=samples_per_sequence,
                block_size=block_size,
                shuffle_within_sequence=True, 
                random_block_order=random_block_order_setting, # depending on schedule_type
                seed=schedule_seed
            )

            loss_array, loss_retention, loss_test, loss_retention_array, loss_test_array = train_evaluate_model(
                X_train_sched,
                y_train_sched,
                X_repetitive,
                y_repetitive,
                X_test,
                y_test,
                model,
                criterion,
                optimizer,
                is_dislplay_loss=False
            )

            seq_retention_means = sequence_level_retention(
                loss_retention_array,
                samples_per_sequence=samples_per_sequence,
                num_sequences=num_training_sequences
            )

            retention_imbalance = np.std(seq_retention_means)

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

# parallelize
num_runs = 20
results = Parallel(n_jobs=-1)(delayed(run_model_continuum)(1000 - i) for i in range(num_runs))

# =================================================
#                Aggregate Results
# =================================================

block_sizes = [1000, 500, 200, 100, 50, 20, 10, 5, 1]
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

        # list to array
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

results_folder = f"results_block_continuum_serial_and_random_lr_{str(lr).replace('.', '_')}"

if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# save raw results
np.save(results_folder + "/raw_results.npy", np.array(results, dtype=object), allow_pickle=True)

# save pre training results
np.save(results_folder + "/loss_test_pre_array.npy", loss_test_pre_array)
np.save(results_folder + "/loss_test_array_pre_array.npy", loss_test_array_pre_array)

# save schedule type dictionaries
np.save(results_folder + "/loss_array_by_schedule.npy", loss_array_by_schedule, allow_pickle=True)
np.save(results_folder + "/loss_retention_by_schedule.npy", loss_retention_by_schedule, allow_pickle=True)
np.save(results_folder + "/loss_test_by_schedule.npy", loss_test_by_schedule, allow_pickle=True)
np.save(results_folder + "/loss_retention_array_by_schedule.npy", loss_retention_array_by_schedule, allow_pickle=True)
np.save(results_folder + "/loss_test_array_by_schedule.npy", loss_test_array_by_schedule, allow_pickle=True)
np.save(results_folder + "/seq_retention_means_by_schedule.npy", seq_retention_means_by_schedule, allow_pickle=True)
np.save(results_folder + "/retention_imbalance_by_schedule.npy", retention_imbalance_by_schedule, allow_pickle=True)
np.save(results_folder + "/schedule_idx_by_schedule.npy", schedule_idx_by_schedule, allow_pickle=True)

print(f"\nResults saved to: {results_folder}")

"""
For 20 parallel runs: 360 models trained and evaluated (20 x 9 x 2)
"""






