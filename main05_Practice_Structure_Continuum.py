
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

import torch
import torch.nn as nn
import torch.optim as optim 
import numpy as np
from copy import deepcopy 
from helpers.NetworkFunction import MotorLearningRNN
from helpers.DataGenerator import generate_synthetic_data, sample_balanced_replay, sample_selected_replay
from helpers.TorchFunctions import train_evaluate_model, vulnerability_test
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
    """
    1000: pure RP
    1: pure, serial interleaved (not fully random IP, ABCABCABCABC..)
    """

    print(f"\n================ Seed {seed} ================")

    # generate data
    X_repetitive, y_repetitive, _, _ = generate_synthetic_data(num_sequences=num_training_sequences,samples_per_sequence=samples_per_sequence,add_input_noise=True)
    X_pre, y_pre, _, _ = generate_synthetic_data(num_sequences=num_pre_training_sequences,samples_per_sequence=100,add_input_noise=True)
    _, _, X_test, y_test = generate_synthetic_data(num_sequences=num_test_sequences,add_input_noise=True)

    # pre train model
    base_model = MotorLearningRNN(input_size,hidden_size,output_size,num_outputs=input_size)
    criterion = nn.CrossEntropyLoss()
    optimizer_pre = optim.SGD(base_model.parameters(), lr=lr, momentum=0.0)

    _, _, loss_test_pre, _, loss_test_array_pre = train_evaluate_model(X_pre, y_pre,X_pre, y_pre,X_test, y_test,base_model,criterion,optimizer_pre,is_dislplay_loss=False)

    # train one deep copied model per block size
    condition_results = {}

    for block_size in block_sizes:
        print(f"\nTraining block_size = {block_size}")

        # copy the pre trained model
        model = deepcopy(base_model)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.0)
        
        # generate scheduled datasets      ***** serial *****
        X_train_sched, y_train_sched, schedule_idx = make_block_schedule_dataset(
            X_repetitive,
            y_repetitive,
            num_sequences=num_training_sequences,
            samples_per_sequence=samples_per_sequence,
            block_size=block_size, # block size here
            shuffle_within_sequence=True,
            random_block_order=False,
            seed=seed + block_size
        )

        # evaluation metric A and B, retention and generalization
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

        """
        Generate retention performance for each trained sequence separately
        Then measure how uneven the model's retention is across the 3 sequencecs
        """
        seq_retention_means = sequence_level_retention(loss_retention_array,samples_per_sequence=samples_per_sequence,num_sequences=num_training_sequences)
        retention_imbalance = np.std(seq_retention_means) # std measures how unequal the 3 losses are

        condition_results[block_size] = {
            "loss_array": loss_array, # training loss during scheduled learning (3-element vector per epoch)
            "loss_retention": loss_retention, # scalar per schedule, average retention loss
            "loss_test": loss_test, #scalar, average loss on the novel generalization set
            "loss_retention_array": loss_retention_array, # per sequence sample retention loss
            "loss_test_array": loss_test_array, # per sequence sample generalization loss
            "seq_retention_means": seq_retention_means, #3-element vector per model, average retention loss per ABC sequences
            "retention_imbalance": retention_imbalance, # scalar per model
            "schedule_idx": schedule_idx # keeps track of the 3 sequences in the scheduled dataset
        }

    return {
        "seed": seed,
        "loss_test_pre": loss_test_pre,
        "loss_test_array_pre": loss_test_array_pre,
        "condition_results": condition_results
    }

'''
debug_result = run_model_continuum(seed=1000)

for block_size, res in debug_result["condition_results"].items():
    print("\nBlock size:", block_size)
    print("Retention loss:", res["loss_retention"])
    print("Generalization loss:", res["loss_test"])
    print("Seq retention means:", res["seq_retention_means"])
    print("Retention imbalance:", res["retention_imbalance"])
'''

# parallelize
num_runs = 20
results = Parallel(n_jobs=-1)(delayed(run_model_continuum)(1000 - i) for i in range(num_runs))

# =================================================
#                Aggregate Results
# =================================================

# same block sizes used inside run_model_continuum()
block_sizes = [1000, 500, 200, 100, 50, 20, 10, 5, 1]

# pre raining results
loss_test_pre_list = [r["loss_test_pre"] for r in results]
loss_test_array_pre_list = [r["loss_test_array_pre"] for r in results]
loss_test_pre_array = np.array(loss_test_pre_list)
loss_test_array_pre_array = np.stack(loss_test_array_pre_list, axis=0)

# dictionaries storing one array per block size
loss_array_by_block = {}
loss_retention_by_block = {}
loss_test_by_block = {}
loss_retention_array_by_block = {}
loss_test_array_by_block = {}
seq_retention_means_by_block = {}
retention_imbalance_by_block = {}
schedule_idx_by_block = {}

for block_size in block_sizes:

    # collect lists
    loss_array_list = [r["condition_results"][block_size]["loss_array"] for r in results]
    loss_retention_list = [r["condition_results"][block_size]["loss_retention"] for r in results]
    loss_test_list = [r["condition_results"][block_size]["loss_test"] for r in results]
    loss_retention_array_list = [r["condition_results"][block_size]["loss_retention_array"] for r in results]
    loss_test_array_list = [r["condition_results"][block_size]["loss_test_array"] for r in results]
    seq_retention_means_list = [r["condition_results"][block_size]["seq_retention_means"] for r in results]
    retention_imbalance_list = [r["condition_results"][block_size]["retention_imbalance"] for r in results]
    schedule_idx_list = [r["condition_results"][block_size]["schedule_idx"] for r in results]

    # convert lists to arrays
    loss_array_by_block[block_size] = np.stack(loss_array_list, axis=0)
    loss_retention_by_block[block_size] = np.array(loss_retention_list)
    loss_test_by_block[block_size] = np.array(loss_test_list)
    loss_retention_array_by_block[block_size] = np.stack(loss_retention_array_list, axis=0)
    loss_test_array_by_block[block_size] = np.stack(loss_test_array_list, axis=0)
    seq_retention_means_by_block[block_size] = np.stack(seq_retention_means_list, axis=0)
    retention_imbalance_by_block[block_size] = np.array(retention_imbalance_list)
    schedule_idx_by_block[block_size] = np.stack(schedule_idx_list, axis=0)
    
# =================================================
#                Save Results
# =================================================

import os

results_folder = f"results_block_continuum_lr_{str(lr).replace('.', '_')}"

if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# save raw results first, useful for debugging
np.save(results_folder + "/raw_results.npy", np.array(results, dtype=object), allow_pickle=True)

# save pre-training results
np.save(results_folder + "/loss_test_pre_array.npy", loss_test_pre_array)
np.save(results_folder + "/loss_test_array_pre_array.npy", loss_test_array_pre_array)

# save block-size dictionaries
np.save(results_folder + "/loss_array_by_block.npy", loss_array_by_block, allow_pickle=True)
np.save(results_folder + "/loss_retention_by_block.npy", loss_retention_by_block, allow_pickle=True)
np.save(results_folder + "/loss_test_by_block.npy", loss_test_by_block, allow_pickle=True)
np.save(results_folder + "/loss_retention_array_by_block.npy", loss_retention_array_by_block, allow_pickle=True)
np.save(results_folder + "/loss_test_array_by_block.npy", loss_test_array_by_block, allow_pickle=True)
np.save(results_folder + "/seq_retention_means_by_block.npy", seq_retention_means_by_block, allow_pickle=True)
np.save(results_folder + "/retention_imbalance_by_block.npy", retention_imbalance_by_block, allow_pickle=True)
np.save(results_folder + "/schedule_idx_by_block.npy", schedule_idx_by_block, allow_pickle=True)

print(f"\nResults saved to: {results_folder}")

"""
For 20 parallel runs: 180 models trained and evaluated (20 x 9)
"""






