import numpy as np
import torch
import matplotlib.pyplot as plt
from copy import deepcopy
from helpers.DataGenerator import generate_synthetic_data

def train_evaluate_model(X, y, X_retention, y_retention, X_test, y_test ,model, criterion, 
                         optimizer, batch_size=20, is_dislplay_loss=True):

    """
    Trains and evaluates the model
    Parameters:
        - Train the model on (X, y)
        - Evaluate retention on (X_retention, y_retention), which are the same as (X, y)
        - Then test generalization performance on new generated X_test, y_test
        - model: motor sequence RNN
        - criterion: loss function (cross entropy loss)
        - optimizer: SGD with momentum = 0.0
        
    Evalaution: Performs following two metric calculation
        - (A) Training set evaluation (retention)
        - (B) Generalization set evaluation (X_test y_test)
    """
    
    # check if training data exists
    if X.shape[0] != 0 or y.shape[0] != 0:
        
        # convert training and testing NumPy to PyTorch tensors
        X_train = torch.tensor(X, dtype=torch.float32)
        y_train = torch.tensor(y, dtype=torch.float32)
        X_retention = torch.tensor(X_retention, dtype=torch.float32)
        y_retention = torch.tensor(y_retention, dtype=torch.float32)

        # training set up
        loss_array = [] # stores loss per batch
        num_batch = len(X_train) // batch_size

        print(f'Length of X_train: {len(X_train)}, Batch size: {batch_size}, Number of batches: {num_batch}')

        # mini-batch training loop
        for i in range(0, len(X_train), batch_size):
            # extract the batch
            X_batch = X_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]
            # forward pass
            outputs = model(X_batch) # raw output before softmax

            # flatten y_batch and output, treat each time step as independent classification
            y_batch_view = y_batch.view(-1, y_batch.shape[2])
            outputs_view = outputs.view(-1, outputs.shape[2])

            # then convert y_batch vectors to class labels, [0010] = 2
            y_batch_class = torch.argmax(y_batch_view, dim=1)

            # compute the loss
            loss = criterion(outputs_view, y_batch_class)
                # this step does: softmax(output) + log + -log likelihood
            # store the loss
            loss_array.append(loss.item())
            
            # backpropagation
            optimizer.zero_grad() # clear gradients
            loss.backward() # BP through time compute gradients
            optimizer.step() # update weights

        # ***** Rentention / Training Set evaluation *****
        with torch.no_grad(): # disable gradient tracking
            # forward pass (but this time with updated weights)
            outputs = model(X_retention)
            # same as above...
            y_retention_view = y_retention.view(-1, y_retention.shape[2])
            y_retention_class = torch.argmax(y_retention_view, dim=1)
            outputs_view = outputs.view(-1, outputs.shape[2])
            
            # compute retention loss: how well old sequences are remembered
            loss_retention = criterion(outputs_view, y_retention_class)
            print(f'Training Set Evaluation Loss: {loss_retention.item():.4f}')
            
            # compute retention loss per-sequence
            print('y_retention shape:', y_retention.shape)
            loss_retention_array = []
            
            for i in range(y_retention.shape[0]):
                loss_retention_i = criterion(outputs[i, :, :], y_retention[i, :, :])
                loss_retention_array.append(loss_retention_i.item())
            # check the average of loss_retention_array is almost same as loss_retention
            print(f'(A) Training Set Evaluation Loss (each output): {loss_retention:.4f}, Average: {np.mean(loss_retention_array):.4f}')
    else:
        loss_array = []
        loss_retention = torch.tensor(0.0)
        loss_retention_array = []
        print('No training data provided.')

    # Test the model using Generalization data 
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(X_test)

        y_test_view = y_test.view(-1, y_test.shape[2])
        y_test_class = torch.argmax(y_test_view, dim=1)
        outputs_view = outputs.view(-1, outputs.shape[2])
        # compute loss
        loss_test = criterion(outputs_view, y_test_class)
        print(f'(B) Generalization Loss: {loss_test.item():.4f}')
        # compute loss for each sequence separately
        loss_test_array = []
        for i in range(y_test.shape[0]):
            # evaluate on 100 novel sequences
            loss_test_i = criterion(outputs[i, :, :], y_test[i, :, :])
            loss_test_array.append(loss_test_i.item())
            
        # check the average of loss_test_array is almost same as loss_test
        print(f'(B) Generalization Loss (each output): {loss_test:.4f}, Average: {np.mean(loss_test_array):.4f}')

    if is_dislplay_loss:
        loss_array = np.array(loss_array)
        plt.plot(loss_array) 
        plt.show()

    return loss_array, loss_retention.item(), loss_test.item(), loss_retention_array, loss_test_array

    """
    loss_array: training cross entropy loss per batch, training dynamics
    loss_retention_array: cross entropy loss per sequence in the training set, training set evaluation
        Represents memory quality for each sequence (shows forgetting for earlier sequences)
    loss_test_array: loss computed per sequence in the generalization set, generalization set evaluation
    """

def vulnerability_test(X_retention, y_retention, X_test, y_test, model, criterion, optimizer,
                       num_repeat_noisy=10, num_repeat_pruned=10, num_interference_steps=100,
                       batch_size=20):
    
    """
    Performs the rest of the 3 evaluation schemes:
        (C) Noise vulnerability test: add noise to model weights
        (D) Adversarial pruning test: remove model weights (=0)
        (E) Interference vulnerability test: update weights on novel sequence
        
    Parameters:
        - X_retention, y_retention: training sets, or previously learned sequences
            - X_repetitive, y_repetitive were passed here
        - X_test, y_test: novel sequences unseen during training
        - num_repeat_noisy: number of times to evaluate after cumulative noise injection
        - num_repeat_pruned: number of times to evaluate after cumulative pruning of weights
        - num_interference_steps: number of mini-batches of a new interfering sequence to train on
            - Controls the pressure / strength of interference from learning new sequence
    """
    # keeps only the third sequence, i.e. the last trained sequence
    X_retention = torch.tensor(X_retention[- (X_retention.shape[0] // 3):], dtype=torch.float32)
    y_retention = torch.tensor(y_retention[- (y_retention.shape[0] // 3):], dtype=torch.float32)
    
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    
    # print shapes of retention and test data
    print(f'X_retention shape: {X_retention.shape}, y_retention shape: {y_retention.shape}')

# (C) Noise vulnerability test (noisy model weights)
    
    # copy the model
    model_noisy = deepcopy(model)
    # noise parameter
    noise_std = 0.1
    loss_retention_noisy_array = []
    loss_test_noisy_array = []
    
    # the noise loop
    for repeat in range(num_repeat_noisy):
        if repeat > 0:
            # when repeat == 0, the firs entry in the loss arrays becomes the unperturbed baseline
            with torch.no_grad():
                for param in model_noisy.parameters():
                    # generate gaussian noise & add noise to each parameter
                    noise = torch.randn_like(param) * noise_std
                    param.add_(noise) # cumulative
                    
        # evaluate the model using retention data
        with torch.no_grad():
            # forward pass
            outputs = model_noisy(X_retention)
            # same as before
            y_retention_view = y_retention.view(-1, y_retention.shape[2])
            y_retention_class = torch.argmax(y_retention_view, dim=1)
            outputs_view = outputs.view(-1, outputs.shape[2])
            # compute loss
            loss_retention_noisy = criterion(outputs_view, y_retention_class)
            loss_retention_noisy_array.append(loss_retention_noisy.item())
            
        # evaluate the model using test data (generalization)
        with torch.no_grad():
            outputs = model_noisy(X_test)
            y_test_view = y_test.view(-1, y_test.shape[2])
            y_test_class = torch.argmax(y_test_view, dim=1)
            outputs_view = outputs.view(-1, outputs.shape[2])
            loss_test_noisy = criterion(outputs_view, y_test_class)
            loss_test_noisy_array.append(loss_test_noisy.item())

    # print all the noisy retention losses and test losses
    print(f'(C) Noisy Vulnerability Retention Losses: {loss_retention_noisy_array}')
    print(f'(C) Noisy Vulnerability Generalization Losses: {loss_test_noisy_array}')

# (D) Adversarial Prining Test (randomly set weights to 0)

    # copy the model
    model_pruned = deepcopy(model)
    # parameters
    pruning_percent = 0.05
    loss_retention_pruned_array = []
    loss_test_pruned_array = []
    
    # pruning loop
    for repeat in range(num_repeat_pruned):
        if repeat > 0:
            # again save the first entry in loss arrays as baseline before pruning
            with torch.no_grad():
                for param in model_pruned.parameters():
                    # randomly set 5% of weights to zero, cumulatively
                    mask = torch.rand_like(param) > pruning_percent
                    param.mul_(mask)
                    
        # evaluate the model using retention data
        with torch.no_grad():
            outputs = model_pruned(X_retention)
            y_retention_view = y_retention.view(-1, y_retention.shape[2])
            y_retention_class = torch.argmax(y_retention_view, dim=1)
            outputs_view = outputs.view(-1, outputs.shape[2])

            loss_retention_pruned = criterion(outputs_view, y_retention_class)
            loss_retention_pruned_array.append(loss_retention_pruned.item())
            
        # evaluate the model using generalization data
        with torch.no_grad():
            outputs = model_pruned(X_test)
            y_test_view = y_test.view(-1, y_test.shape[2])
            y_test_class = torch.argmax(y_test_view, dim=1)
            outputs_view = outputs.view(-1, outputs.shape[2])
            
            loss_test_pruned = criterion(outputs_view, y_test_class)
            loss_test_pruned_array.append(loss_test_pruned.item())

    # print all the pruned retention losses and test losses
    print(f'(D) Adversarial Pruning Retention Losses: {loss_retention_pruned_array}')
    print(f'(D) Adversarial Pruning Generalization Losses: {loss_test_pruned_array}')

# (E) Interference Vulnerability Test (update model weights using new tasks)

    # generate one new sequence type to interfere with old learning
    _, _, X_interference, y_interference = generate_synthetic_data(num_sequences=1, 
                                            samples_per_sequence=batch_size*num_interference_steps, add_input_noise=True)
    loss_retention_interference_array = []
    loss_test_interference_array = []
    
    for i in range(-batch_size, len(X_interference), batch_size):
        if i >= 0:
            # when i=-20 at beginning, the first entry in the loss array is again baseline
            X_batch = torch.tensor(X_interference[i:i + batch_size], dtype=torch.float32)
            y_batch = torch.tensor(y_interference[i:i + batch_size], dtype=torch.float32)
            
            # forward pass
            outputs = model(X_batch) # run with the **original, unpruned unnoised model**
            y_batch_view = y_batch.view(-1, y_batch.shape[2])
            outputs_view = outputs.view(-1, outputs.shape[2])
            y_batch_class = torch.argmax(y_batch_view, dim=1)

            # compute loss and back-propagate to update weights
            # note model is not copied before training
            loss = criterion(outputs_view, y_batch_class)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # evaluate the updatedmodel using retention data
        with torch.no_grad():
            
            # forward pass
            outputs = model(X_retention)
            y_retention_view = y_retention.view(-1, y_retention.shape[2])
            y_retention_class = torch.argmax(y_retention_view, dim=1)
            outputs_view = outputs.view(-1, outputs.shape[2])
            # compute loss
            loss_retention_interference = criterion(outputs_view, y_retention_class)
            loss_retention_interference_array.append(loss_retention_interference.item())
            
        # evaluate the model using transfer data
        with torch.no_grad():
            outputs = model(X_test)
            y_test_view = y_test.view(-1, y_test.shape[2])
            y_test_class = torch.argmax(y_test_view, dim=1)
            outputs_view = outputs.view(-1, outputs.shape[2])
          
            loss_test_interference = criterion(outputs_view, y_test_class)
            loss_test_interference_array.append(loss_test_interference.item())

    print(f'(E) Interference Vulnerability Retention Losses: {loss_retention_interference_array}')
    print(f'(E) Interference Vulnerability Generalization Losses: {loss_test_interference_array}')

    return loss_retention_noisy_array, loss_test_noisy_array, loss_retention_pruned_array, loss_test_pruned_array, loss_retention_interference_array, loss_test_interference_array


















