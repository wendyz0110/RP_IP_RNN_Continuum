
#=================================================================
#                         OFFLINE REPLAY
#    "Offline" in the sense that the replay phase is separated from
# initial training and model fitting phase. First, the model will be 
# trained as usual using the training data. Then during the replay
# phase, the model is trained again on replay data.

# Before: pretrain → RP/IP training → evaluation
# After: pretrain → RP/IP training → OFFLINE REPLAY → evaluation
#=================================================================

import torch
import torch.nn as nn
import torch.optim as optim 
import numpy as np
from copy import deepcopy 
from helpers.NetworkFunction import MotorLearningRNN
from helpers.DataGenerator import generate_synthetic_data, sample_balanced_replay, sample_selected_replay
from helpers.TorchFunctions import train_evaluate_model, vulnerability_test
from joblib import Parallel, delayed 

lr = 0.02
def run_model(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    input_size = 7 
    hidden_size = 7
    output_size = 4
    batch_size = 20
    num_training_sequences = 3
    num_pre_training_sequences = 10
    num_test_sequences = 100 
    print('Learning Rate:', lr)
    
    X_repetitive, y_repetitive, X_interleaved, y_interleaved = generate_synthetic_data(
        num_training_sequences, samples_per_sequence=4000, add_input_noise=True)
    print(X_repetitive.shape)
    print(y_repetitive.shape)

    # pre train data
    X_pre, y_pre, _, _ = generate_synthetic_data(num_pre_training_sequences, samples_per_sequence=100, add_input_noise=True)
    
    # generalization testing data
    _, _, X_test, y_test = generate_synthetic_data(num_test_sequences, add_input_noise=True)
    
    # instantiate pretraining model
    model1 = MotorLearningRNN(input_size, hidden_size, output_size, num_outputs=input_size)
    criterion = nn.CrossEntropyLoss()
    optimizer1 = optim.SGD(model1.parameters(), lr=lr, momentum=0.0) # pure vanilla SGD
    
    total_num_epochs_pre = X_pre.shape[0] // batch_size
    print(f"Total number of epochs for pre-training: {total_num_epochs_pre}")
    
    # pretraining call, only generalization loss is retained
    _, _, loss_test_pre, _, loss_test_array_pre = train_evaluate_model(X_pre, y_pre, X_pre, y_pre, X_test, y_test,
                                                                       model1, criterion, optimizer1,
                                                                       is_dislplay_loss=False)
    # copy pretrained model for IP
    model2 = deepcopy(model1)
    # create new optimizers
    optimizer1 = optim.SGD(model1.parameters(), lr=lr, momentum=0.0)
    optimizer2 = optim.SGD(model2.parameters(), lr=lr, momentum=0.0)
    
    # Repetitive Practice (RP) Training 
    
    print("\nRepetitive Practice:")
    loss_array_rp, loss_retention_rp, loss_test_rp, loss_retention_array_rp, loss_test_array_rp = train_evaluate_model(
    X_repetitive, y_repetitive, X_repetitive, y_repetitive, X_test, y_test, model1, criterion, optimizer1, is_dislplay_loss=False)
    
    # Interleaved Practice (IP) Training
    print("\nInterleaved Practice:")
    
    # notice: retention evaluation sets are repetitive, not interleaved
    loss_array_ip, loss_retention_ip, loss_test_ip, loss_retention_array_ip, loss_test_array_ip = train_evaluate_model(
    X_interleaved, y_interleaved, X_repetitive, y_repetitive, X_test, y_test, model2, criterion, optimizer2, is_dislplay_loss=False)
        
    # testing schemes A and B were done as part of the training and testing phase
    
    # Because vulnerability test changes the model (weight pruning etc), we copy the models here
    model1_copy = deepcopy(model1)
    model2_copy = deepcopy(model2)

    # Vulnerability Testing Phase 
    
    print("\nVulnerability Test for Repetitive Practice:")
    optimizer_rp = optim.SGD(model1_copy.parameters(), lr=lr, momentum=0.0)
    loss_retention_noisy_array_rp, loss_test_noisy_array_rp, loss_retention_pruned_array_rp, loss_test_pruned_array_rp, loss_retention_interf_array_rp, loss_test_interf_array_rp = vulnerability_test(
    X_repetitive, y_repetitive, X_test, y_test, model1_copy, criterion, optimizer_rp,
    num_repeat_noisy=60, num_repeat_pruned=60, num_interference_steps=60)
    
    print("\nVulnerability Test for Interleaved Practice:")
    optimizer_ip = optim.SGD(model2_copy.parameters(), lr=lr, momentum=0.0)
    loss_retention_noisy_array_ip, loss_test_noisy_array_ip, loss_retention_pruned_array_ip, loss_test_pruned_array_ip, loss_retention_interf_array_ip, loss_test_interf_array_ip = vulnerability_test(
        X_repetitive, y_repetitive, X_test, y_test, model2_copy, criterion, optimizer_ip, num_repeat_noisy=60, 
        num_repeat_pruned=60, num_interference_steps=60)
    
    # ================= Balanced OFFLINE REPLAY (Identical RP and IP Replay) ====================
    # RP and IP will share the same Replay Dataset (randomly chosen, shuffled past sequences)
    # the replay algorithm modifies model1 and model2
    
    # dataset size: 200*3=600, ~5% of the original training set
    # =====================================================================================

    # generate replay dataset
    X_replay, y_replay = sample_balanced_replay(X_repetitive, y_repetitive,
                                                num_per_seq=4000, k=200) # 600 in total
    
    # copy the models for balanced Offline Replay
    model_RP_BalancedReplay = deepcopy(model1)
    model_IP_BalancedReplay = deepcopy(model2)
    
    print("\nOffline Replay for RP:")
    
    optimizer_BalancedReplay_rp = optim.SGD(model_RP_BalancedReplay.parameters(), lr=lr, momentum=0.0)
    
    results_rp_replay_balanced = train_evaluate_model(
        X_replay, y_replay,  
        X_repetitive, y_repetitive,
        X_test, y_test,
        model_RP_BalancedReplay,
        criterion,
        optimizer_BalancedReplay_rp,
        is_dislplay_loss=False
    )
    
    # original names + replay
    loss_array_rp_replay_balanced = results_rp_replay_balanced[0]
    loss_retention_rp_replay_balanced = results_rp_replay_balanced[1]
    loss_test_rp_replay_balanced = results_rp_replay_balanced[2]
    loss_retention_array_rp_replay_balanced = results_rp_replay_balanced[3]
    loss_test_array_rp_replay_balanced = results_rp_replay_balanced[4]
    
    print("\nOffline Replay for IP:")
    
    optimizer_BalancedReplay_ip = optim.SGD(model_IP_BalancedReplay.parameters(), lr=lr, momentum=0.0)
    
    results_ip_replay_balanced = train_evaluate_model(
        X_replay, y_replay,  
        X_repetitive, y_repetitive,
        X_test, y_test,
        model_IP_BalancedReplay,
        criterion,
        optimizer_BalancedReplay_ip,
        is_dislplay_loss=False
    )
    
    # original names + replay
    loss_array_ip_replay_balanced = results_ip_replay_balanced[0]
    loss_retention_ip_replay_balanced = results_ip_replay_balanced[1]
    loss_test_ip_replay_balanced = results_ip_replay_balanced[2]
    loss_retention_array_ip_replay_balanced = results_ip_replay_balanced[3]
    loss_test_array_ip_replay_balanced = results_ip_replay_balanced[4]
    
    # copy models again and do vulnerability tests again
    model1_BalancedReplay_copy = deepcopy(model_RP_BalancedReplay) # model1 = after replay
    model2_BalancedReplay_copy = deepcopy(model_IP_BalancedReplay)
    
    print("\nVulnerability Test AFTER Replay for RP:")

    optimizer_rp_replay_vuln = optim.SGD(model1_BalancedReplay_copy.parameters(), lr=lr, momentum=0.0)
    
    loss_retention_noisy_array_rp_replay, loss_test_noisy_array_rp_replay, \
    loss_retention_pruned_array_rp_replay, loss_test_pruned_array_rp_replay, \
    loss_retention_interf_array_rp_replay, loss_test_interf_array_rp_replay = vulnerability_test(
        X_repetitive, y_repetitive, X_test, y_test,
        model1_BalancedReplay_copy,
        criterion,
        optimizer_rp_replay_vuln,
        num_repeat_noisy=60,
        num_repeat_pruned=60,
        num_interference_steps=60)
    
    print("\nVulnerability Test AFTER Replay for IP:")

    optimizer_ip_replay_vuln = optim.SGD(model2_BalancedReplay_copy.parameters(), lr=lr, momentum=0.0)
    
    loss_retention_noisy_array_ip_replay, loss_test_noisy_array_ip_replay, \
    loss_retention_pruned_array_ip_replay, loss_test_pruned_array_ip_replay, \
    loss_retention_interf_array_ip_replay, loss_test_interf_array_ip_replay = vulnerability_test(
        X_repetitive, y_repetitive, X_test, y_test,
        model2_BalancedReplay_copy,
        criterion,
        optimizer_ip_replay_vuln,
        num_repeat_noisy=60,
        num_repeat_pruned=60,
        num_interference_steps=60)
    
    # ================= Selected Replay (RP Only) ====================
    # Based on human research, "Human hippocampal replay during rest prioritizes
    # weakly learned information and predicts memory performance
    # So we try replay more of sequence 1 and 2 for RP, see if RP performance improved even more
    # =====================================================================================
    
    print("\nSelected Replay for RP:")

    # generate selected replay dataset (biased toward A and B)
    X_replay_selected, y_replay_selected = sample_selected_replay(
        X_repetitive, y_repetitive, num_per_seq=4000, k_A=300, k_B=250, k_C=50
    )
    
    # copy RP model 
    model_RP_SelectedReplay = deepcopy(model1)
    optimizer_SelectedReplay_rp = optim.SGD(model_RP_SelectedReplay.parameters(), lr=lr, momentum=0.0)

    results_rp_replay_selected = train_evaluate_model(
        X_replay_selected, y_replay_selected,
        X_repetitive, y_repetitive,
        X_test, y_test,
        model_RP_SelectedReplay,
        criterion,
        optimizer_SelectedReplay_rp,
        is_dislplay_loss=False
    )

    loss_array_rp_replay_selected = results_rp_replay_selected[0]
    loss_retention_rp_replay_selected = results_rp_replay_selected[1]
    loss_test_rp_replay_selected = results_rp_replay_selected[2]
    loss_retention_array_rp_replay_selected = results_rp_replay_selected[3]
    loss_test_array_rp_replay_selected = results_rp_replay_selected[4]
    
    # vulnerability test
    print("\nVulnerability Test AFTER Selected Replay for RP:")
    
    model_RP_SelectedReplay_copy = deepcopy(model_RP_SelectedReplay)
    
    optimizer_rp_selected_replay_vuln = optim.SGD(
        model_RP_SelectedReplay_copy.parameters(), lr=lr, momentum=0.0
    )
    
    loss_retention_noisy_array_rp_replay_selected, \
    loss_test_noisy_array_rp_replay_selected, \
    loss_retention_pruned_array_rp_replay_selected, \
    loss_test_pruned_array_rp_replay_selected, \
    loss_retention_interf_array_rp_replay_selected, \
    loss_test_interf_array_rp_replay_selected = vulnerability_test(
        X_repetitive, y_repetitive,
        X_test, y_test,
        model_RP_SelectedReplay_copy,
        criterion,
        optimizer_rp_selected_replay_vuln,
        num_repeat_noisy=60,
        num_repeat_pruned=60,
        num_interference_steps=60
        )
     
    return (loss_test_pre, 
            loss_array_rp, 
            loss_retention_rp, 
            loss_test_rp, 
            loss_array_ip, 
            loss_retention_ip, 
            loss_test_ip, 
            loss_retention_array_rp, 
            loss_test_array_rp, 
            loss_retention_array_ip, 
            loss_test_array_ip, 
            loss_test_array_pre, 
            loss_retention_noisy_array_rp, 
            loss_test_noisy_array_rp, 
            loss_retention_noisy_array_ip, 
            loss_test_noisy_array_ip, 
            loss_retention_pruned_array_rp, 
            loss_test_pruned_array_rp, 
            loss_retention_pruned_array_ip, 
            loss_test_pruned_array_ip, 
            loss_retention_interf_array_rp, 
            loss_test_interf_array_rp, 
            loss_retention_interf_array_ip, 
            loss_test_interf_array_ip,
            
           # ==================== BALANCED REPLAY (RP) ====================
            loss_array_rp_replay_balanced,
            loss_retention_rp_replay_balanced,
            loss_test_rp_replay_balanced,
            loss_retention_array_rp_replay_balanced,
            loss_test_array_rp_replay_balanced,
            
            # ==================== BALANCED REPLAY (IP) ====================
            loss_array_ip_replay_balanced,
            loss_retention_ip_replay_balanced,
            loss_test_ip_replay_balanced,
            loss_retention_array_ip_replay_balanced,
            loss_test_array_ip_replay_balanced,
            
            # ==================== REPLAY VULNERABILITY (RP) ====================
            loss_retention_noisy_array_rp_replay,
            loss_test_noisy_array_rp_replay,
            loss_retention_pruned_array_rp_replay,
            loss_test_pruned_array_rp_replay,
            loss_retention_interf_array_rp_replay,
            loss_test_interf_array_rp_replay,
            
            # ==================== REPLAY VULNERABILITY (IP) ====================
            loss_retention_noisy_array_ip_replay,
            loss_test_noisy_array_ip_replay,
            loss_retention_pruned_array_ip_replay,
            loss_test_pruned_array_ip_replay,
            loss_retention_interf_array_ip_replay,
            loss_test_interf_array_ip_replay,
            
            # ==================== SELECTED REPLAY (RP) ====================
            loss_array_rp_replay_selected,
            loss_retention_rp_replay_selected,
            loss_test_rp_replay_selected,
            loss_retention_array_rp_replay_selected,
            loss_test_array_rp_replay_selected,
            
            # ==================== SELECTED REPLAY VULNERABILITY (RP) ====================
            loss_retention_noisy_array_rp_replay_selected,
            loss_test_noisy_array_rp_replay_selected,
            loss_retention_pruned_array_rp_replay_selected,
            loss_test_pruned_array_rp_replay_selected,
            loss_retention_interf_array_rp_replay_selected,
            loss_test_interf_array_rp_replay_selected
            )

# set random seed per run_model() iteration
results = Parallel(n_jobs=-1)(delayed(run_model)(1000 - i) for i in range(20))

"""
Blocked = rp
Interleaved = ip
"""

loss_test_pre_list = [results[i][0] for i in range(len(results))]
loss_array_blocked_list = [results[i][1] for i in range(len(results))]
loss_retention_blocked_list = [results[i][2] for i in range(len(results))]
loss_test_blocked_list = [results[i][3] for i in range(len(results))]
loss_array_random_list = [results[i][4] for i in range(len(results))]
loss_retention_random_list = [results[i][5] for i in range(len(results))]
loss_test_random_list = [results[i][6] for i in range(len(results))]
loss_retention_array_blocked_list = [results[i][7] for i in range(len(results))]
loss_test_array_blocked_list = [results[i][8] for i in range(len(results))]
loss_retention_array_random_list = [results[i][9] for i in range(len(results))]
loss_test_array_random_list = [results[i][10] for i in range(len(results))]
loss_test_array_pre_list = [results[i][11] for i in range(len(results))]
loss_retention_noisy_array_blocked_list = [results[i][12] for i in range(len(results))]
loss_test_noisy_array_blocked_list = [results[i][13] for i in range(len(results))]
loss_retention_noisy_array_random_list = [results[i][14] for i in range(len(results))]
loss_test_noisy_array_random_list = [results[i][15] for i in range(len(results))]
loss_retention_pruned_array_blocked_list = [results[i][16] for i in range(len(results))]
loss_test_pruned_array_blocked_list = [results[i][17] for i in range(len(results))]
loss_retention_pruned_array_random_list = [results[i][18] for i in range(len(results))]
loss_test_pruned_array_random_list = [results[i][19] for i in range(len(results))]
loss_retention_interf_array_blocked_list = [results[i][20] for i in range(len(results))]
loss_test_interf_array_blocked_list = [results[i][21] for i in range(len(results))]
loss_retention_interf_array_random_list = [results[i][22] for i in range(len(results))]
loss_test_interf_array_random_list = [results[i][23] for i in range(len(results))]

# add replay lists:
loss_array_rp_replay_balanced_list = [r[24] for r in results]
loss_retention_rp_replay_balanced_list = [r[25] for r in results]
loss_test_rp_replay_balanced_list = [r[26] for r in results]
loss_retention_array_rp_replay_balanced_list = [r[27] for r in results]
loss_test_array_rp_replay_balanced_list = [r[28] for r in results]
loss_array_ip_replay_balanced_list = [r[29] for r in results]
loss_retention_ip_replay_balanced_list = [r[30] for r in results]
loss_test_ip_replay_balanced_list = [r[31] for r in results]
loss_retention_array_ip_replay_balanced_list = [r[32] for r in results]
loss_test_array_ip_replay_balanced_list = [r[33] for r in results]

loss_retention_noisy_array_rp_replay_list = [r[34] for r in results]
loss_test_noisy_array_rp_replay_list = [r[35] for r in results]
loss_retention_pruned_array_rp_replay_list = [r[36] for r in results]
loss_test_pruned_array_rp_replay_list = [r[37] for r in results]
loss_retention_interf_array_rp_replay_list = [r[38] for r in results]
loss_test_interf_array_rp_replay_list = [r[39] for r in results]
loss_retention_noisy_array_ip_replay_list = [r[40] for r in results]
loss_test_noisy_array_ip_replay_list = [r[41] for r in results]
loss_retention_pruned_array_ip_replay_list = [r[42] for r in results]
loss_test_pruned_array_ip_replay_list = [r[43] for r in results]
loss_retention_interf_array_ip_replay_list = [r[44] for r in results]
loss_test_interf_array_ip_replay_list = [r[45] for r in results]

# selected replay lists
loss_array_rp_replay_selected_list = [r[46] for r in results]
loss_retention_rp_replay_selected_list = [r[47] for r in results]
loss_test_rp_replay_selected_list = [r[48] for r in results]
loss_retention_array_rp_replay_selected_list = [r[49] for r in results]
loss_test_array_rp_replay_selected_list = [r[50] for r in results]
loss_retention_noisy_array_rp_replay_selected_list = [r[51] for r in results]
loss_test_noisy_array_rp_replay_selected_list = [r[52] for r in results]
loss_retention_pruned_array_rp_replay_selected_list = [r[53] for r in results]
loss_test_pruned_array_rp_replay_selected_list = [r[54] for r in results]
loss_retention_interf_array_rp_replay_selected_list = [r[55] for r in results]
loss_test_interf_array_rp_replay_selected_list = [r[56] for r in results]

# change list to numpy array
loss_test_pre_array = np.array(loss_test_pre_list)
loss_array_blocked_array = np.stack(loss_array_blocked_list, axis=0)
loss_retention_blocked_array = np.array(loss_retention_blocked_list)
loss_test_blocked_array = np.array(loss_test_blocked_list)
loss_array_random_array = np.stack(loss_array_random_list, axis=0)
loss_retention_random_array = np.array(loss_retention_random_list)
loss_test_random_array = np.array(loss_test_random_list)
loss_retention_array_blocked_array = np.stack(loss_retention_array_blocked_list, axis=0)
loss_test_array_blocked_array = np.stack(loss_test_array_blocked_list, axis=0)
loss_retention_array_random_array = np.stack(loss_retention_array_random_list, axis=0)
loss_test_array_random_array = np.stack(loss_test_array_random_list, axis=0)
loss_test_array_pre_array = np.stack(loss_test_array_pre_list, axis=0)
loss_retention_noisy_array_blocked = np.stack(loss_retention_noisy_array_blocked_list, axis=0)
loss_test_noisy_array_blocked = np.stack(loss_test_noisy_array_blocked_list, axis=0)
loss_retention_noisy_array_random = np.stack(loss_retention_noisy_array_random_list, axis=0)
loss_test_noisy_array_random = np.stack(loss_test_noisy_array_random_list, axis=0)
loss_retention_pruned_array_blocked = np.stack(loss_retention_pruned_array_blocked_list, axis=0)
loss_test_pruned_array_blocked = np.stack(loss_test_pruned_array_blocked_list, axis=0)
loss_retention_pruned_array_random = np.stack(loss_retention_pruned_array_random_list, axis=0)
loss_test_pruned_array_random = np.stack(loss_test_pruned_array_random_list, axis=0)
loss_retention_interf_array_blocked = np.stack(loss_retention_interf_array_blocked_list, axis=0)
loss_test_interf_array_blocked = np.stack(loss_test_interf_array_blocked_list, axis=0)
loss_retention_interf_array_random = np.stack(loss_retention_interf_array_random_list, axis=0)
loss_test_interf_array_random = np.stack(loss_test_interf_array_random_list, axis=0)

loss_array_rp_replay_balanced_array = np.stack(loss_array_rp_replay_balanced_list, axis=0)
loss_retention_rp_replay_balanced_array = np.array(loss_retention_rp_replay_balanced_list)
loss_test_rp_replay_balanced_array = np.array(loss_test_rp_replay_balanced_list)
loss_retention_array_rp_replay_balanced_array = np.stack(loss_retention_array_rp_replay_balanced_list, axis=0)
loss_test_array_rp_replay_balanced_array = np.stack(loss_test_array_rp_replay_balanced_list, axis=0)
loss_array_ip_replay_balanced_array = np.stack(loss_array_ip_replay_balanced_list, axis=0)
loss_retention_ip_replay_balanced_array = np.array(loss_retention_ip_replay_balanced_list)
loss_test_ip_replay_balanced_array = np.array(loss_test_ip_replay_balanced_list)
loss_retention_array_ip_replay_balanced_array = np.stack(loss_retention_array_ip_replay_balanced_list, axis=0)
loss_test_array_ip_replay_balanced_array = np.stack(loss_test_array_ip_replay_balanced_list, axis=0)

loss_retention_noisy_array_rp_replay = np.stack(loss_retention_noisy_array_rp_replay_list, axis=0)
loss_test_noisy_array_rp_replay = np.stack(loss_test_noisy_array_rp_replay_list, axis=0)
loss_retention_pruned_array_rp_replay = np.stack(loss_retention_pruned_array_rp_replay_list, axis=0)
loss_test_pruned_array_rp_replay = np.stack(loss_test_pruned_array_rp_replay_list, axis=0)
loss_retention_interf_array_rp_replay = np.stack(loss_retention_interf_array_rp_replay_list, axis=0)
loss_test_interf_array_rp_replay = np.stack(loss_test_interf_array_rp_replay_list, axis=0)
loss_retention_noisy_array_ip_replay = np.stack(loss_retention_noisy_array_ip_replay_list, axis=0)
loss_test_noisy_array_ip_replay = np.stack(loss_test_noisy_array_ip_replay_list, axis=0)
loss_retention_pruned_array_ip_replay = np.stack(loss_retention_pruned_array_ip_replay_list, axis=0)
loss_test_pruned_array_ip_replay = np.stack(loss_test_pruned_array_ip_replay_list, axis=0)
loss_retention_interf_array_ip_replay = np.stack(loss_retention_interf_array_ip_replay_list, axis=0)
loss_test_interf_array_ip_replay = np.stack(loss_test_interf_array_ip_replay_list, axis=0)

# selected replay for RP
loss_array_rp_replay_selected_array = np.stack(loss_array_rp_replay_selected_list, axis=0)
loss_retention_rp_replay_selected_array = np.array(loss_retention_rp_replay_selected_list)
loss_test_rp_replay_selected_array = np.array(loss_test_rp_replay_selected_list)
loss_retention_array_rp_replay_selected_array = np.stack(loss_retention_array_rp_replay_selected_list, axis=0)
loss_test_array_rp_replay_selected_array = np.stack(loss_test_array_rp_replay_selected_list, axis=0)
loss_retention_noisy_array_rp_replay_selected = np.stack(loss_retention_noisy_array_rp_replay_selected_list, axis=0)
loss_test_noisy_array_rp_replay_selected = np.stack(loss_test_noisy_array_rp_replay_selected_list, axis=0)
loss_retention_pruned_array_rp_replay_selected = np.stack(loss_retention_pruned_array_rp_replay_selected_list, axis=0)
loss_test_pruned_array_rp_replay_selected = np.stack(loss_test_pruned_array_rp_replay_selected_list, axis=0)
loss_retention_interf_array_rp_replay_selected = np.stack(loss_retention_interf_array_rp_replay_selected_list, axis=0)
loss_test_interf_array_rp_replay_selected = np.stack(loss_test_interf_array_rp_replay_selected_list, axis=0)

# change a new folder
results_folder = f'results_replay_lr_{str(lr).replace(".", "_")}'

# create a folder to save the results
import os
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# save the results
np.save(results_folder + '/loss_test_pre_array.npy', loss_test_pre_array)
np.save(results_folder + '/loss_array_blocked_array.npy', loss_array_blocked_array)
np.save(results_folder + '/loss_retention_blocked_array.npy', loss_retention_blocked_array)
np.save(results_folder + '/loss_test_blocked_array.npy', loss_test_blocked_array)
np.save(results_folder + '/loss_array_random_array.npy', loss_array_random_array)
np.save(results_folder + '/loss_retention_random_array.npy', loss_retention_random_array)
np.save(results_folder + '/loss_test_random_array.npy', loss_test_random_array)
np.save(results_folder + '/loss_retention_array_blocked_array.npy', loss_retention_array_blocked_array)
np.save(results_folder + '/loss_test_array_blocked_array.npy', loss_test_array_blocked_array)
np.save(results_folder + '/loss_retention_array_random_array.npy', loss_retention_array_random_array)
np.save(results_folder + '/loss_test_array_random_array.npy', loss_test_array_random_array)
np.save(results_folder + '/loss_test_array_pre_array.npy', loss_test_array_pre_array)
np.save(results_folder + '/loss_retention_noisy_array_blocked.npy', loss_retention_noisy_array_blocked)
np.save(results_folder + '/loss_test_noisy_array_blocked.npy', loss_test_noisy_array_blocked)
np.save(results_folder + '/loss_retention_noisy_array_random.npy', loss_retention_noisy_array_random)
np.save(results_folder + '/loss_test_noisy_array_random.npy', loss_test_noisy_array_random)
np.save(results_folder + '/loss_retention_pruned_array_blocked.npy', loss_retention_pruned_array_blocked)
np.save(results_folder + '/loss_test_pruned_array_blocked.npy', loss_test_pruned_array_blocked)
np.save(results_folder + '/loss_retention_pruned_array_random.npy', loss_retention_pruned_array_random)
np.save(results_folder + '/loss_test_pruned_array_random.npy', loss_test_pruned_array_random)
np.save(results_folder + '/loss_retention_interf_array_blocked.npy', loss_retention_interf_array_blocked)
np.save(results_folder + '/loss_test_interf_array_blocked.npy', loss_test_interf_array_blocked)
np.save(results_folder + '/loss_retention_interf_array_random.npy', loss_retention_interf_array_random)
np.save(results_folder + '/loss_test_interf_array_random.npy', loss_test_interf_array_random)

np.save(results_folder + '/loss_array_rp_replay_balanced.npy', loss_array_rp_replay_balanced_array)
np.save(results_folder + '/loss_retention_rp_replay_balanced.npy', loss_retention_rp_replay_balanced_array)
np.save(results_folder + '/loss_test_rp_replay_balanced.npy', loss_test_rp_replay_balanced_array)
np.save(results_folder + '/loss_retention_array_rp_replay_balanced.npy', loss_retention_array_rp_replay_balanced_array)
np.save(results_folder + '/loss_test_array_rp_replay_balanced.npy', loss_test_array_rp_replay_balanced_array)
np.save(results_folder + '/loss_array_ip_replay_balanced.npy', loss_array_ip_replay_balanced_array)
np.save(results_folder + '/loss_retention_ip_replay_balanced.npy', loss_retention_ip_replay_balanced_array)
np.save(results_folder + '/loss_test_ip_replay_balanced.npy', loss_test_ip_replay_balanced_array)
np.save(results_folder + '/loss_retention_array_ip_replay_balanced.npy', loss_retention_array_ip_replay_balanced_array)
np.save(results_folder + '/loss_test_array_ip_replay_balanced.npy', loss_test_array_ip_replay_balanced_array)

np.save(results_folder + '/loss_retention_noisy_array_rp_replay.npy', loss_retention_noisy_array_rp_replay)
np.save(results_folder + '/loss_test_noisy_array_rp_replay.npy', loss_test_noisy_array_rp_replay)
np.save(results_folder + '/loss_retention_pruned_array_rp_replay.npy', loss_retention_pruned_array_rp_replay)
np.save(results_folder + '/loss_test_pruned_array_rp_replay.npy', loss_test_pruned_array_rp_replay)
np.save(results_folder + '/loss_retention_interf_array_rp_replay.npy', loss_retention_interf_array_rp_replay)
np.save(results_folder + '/loss_test_interf_array_rp_replay.npy', loss_test_interf_array_rp_replay)
np.save(results_folder + '/loss_retention_noisy_array_ip_replay.npy', loss_retention_noisy_array_ip_replay)
np.save(results_folder + '/loss_test_noisy_array_ip_replay.npy', loss_test_noisy_array_ip_replay)
np.save(results_folder + '/loss_retention_pruned_array_ip_replay.npy', loss_retention_pruned_array_ip_replay)
np.save(results_folder + '/loss_test_pruned_array_ip_replay.npy', loss_test_pruned_array_ip_replay)
np.save(results_folder + '/loss_retention_interf_array_ip_replay.npy', loss_retention_interf_array_ip_replay)
np.save(results_folder + '/loss_test_interf_array_ip_replay.npy', loss_test_interf_array_ip_replay)

# selected replay
np.save(results_folder + '/loss_array_rp_replay_selected.npy', loss_array_rp_replay_selected_array)
np.save(results_folder + '/loss_retention_rp_replay_selected.npy', loss_retention_rp_replay_selected_array)
np.save(results_folder + '/loss_test_rp_replay_selected.npy', loss_test_rp_replay_selected_array)
np.save(results_folder + '/loss_retention_array_rp_replay_selected.npy', loss_retention_array_rp_replay_selected_array)
np.save(results_folder + '/loss_test_array_rp_replay_selected.npy', loss_test_array_rp_replay_selected_array)
np.save(results_folder + '/loss_retention_noisy_array_rp_replay_selected.npy', loss_retention_noisy_array_rp_replay_selected)
np.save(results_folder + '/loss_test_noisy_array_rp_replay_selected.npy', loss_test_noisy_array_rp_replay_selected)
np.save(results_folder + '/loss_retention_pruned_array_rp_replay_selected.npy', loss_retention_pruned_array_rp_replay_selected)
np.save(results_folder + '/loss_test_pruned_array_rp_replay_selected.npy', loss_test_pruned_array_rp_replay_selected)
np.save(results_folder + '/loss_retention_interf_array_rp_replay_selected.npy', loss_retention_interf_array_rp_replay_selected)
np.save(results_folder + '/loss_test_interf_array_rp_replay_selected.npy', loss_test_interf_array_rp_replay_selected)









