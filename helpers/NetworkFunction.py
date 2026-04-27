import torch
import torch.nn as nn

class MotorLearningRNN(nn.Module):
    
    """
    Learning Goal:
        - Conceptually, the model learns a mapping from current sequence context to next action (one hot vector)
        - Computationally, "The [model] was designed to serially generate seven one-hot encoded output vectors from a 
          single injection of the input sequence at the beginning."
          
    Network Structure:
        - Standard Elman RNN recurrence (hidden -> hidden) + output -> input feedback loop
        
        - Elman-type RNN: external input at every timepoint, learning to map temporal inputs to temporal outputs
        - MotorLearning RNN:
            - Only initial sequence injection, then the model generates the rest autonomously by feeding outputs as inputs
            - Hence given a cue, the model as to generate the full sequence internally, one-to-many RNN
            
        - To achieve output-to-input feedback loop, a linear layer was added to learn a linear projection to map
          4-dimensional one hot output vector to 7-dimensional input vector

    """
    
    # Constructor
    def __init__(self, input_size, hidden_size, output_size, num_outputs):
        super(MotorLearningRNN, self).__init__() # initialize parent class
        
    # Attributes
        self.hidden_size = hidden_size
        self.num_outputs = num_outputs # num_outputs = sequence length (7)
        # set up Elman RNN, h_t = tanh(W_ih x_t + W_hh h_{t-1})
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        # <hidden> to <output> mapping
        self.fc = nn.Linear(hidden_size, output_size)
        # <output> to <input> mapping 
        # Dimension: 4 to 7
        self.map_back_to_input = nn.Linear(output_size, input_size)
    
    # Methods
    def forward(self, x):
        # initialize hidden state (set to 0)
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        # forward pass with input = external input sequence
        out, hn = self.rnn(x, h0)
       
        # **iterative sequence generation**
        outputs = [] # stores predictions
        for i in range(self.num_outputs):
            # hidden to output
            out = self.fc(out)
            # store the predicted outputs
            outputs.append(out)
            # ** output to input mapping **
            out = self.map_back_to_input(out)
            # feed input back to RNN
            out, hn = self.rnn(out, hn)
        
        # stack outputs together
        outputs = torch.concatenate(outputs, dim=1)
        
        return outputs    # raw output (Ot) before softmax 


    
    
    
    
    
    