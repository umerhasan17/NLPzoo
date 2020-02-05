# https://github.com/spro/practical-pytorch

import torch
import torch.nn as nn
from torch.autograd import Variable

class VanillaRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(VanillaRNN, self).__init__()
        # define params
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        # define layers
        self.rnn = nn.RNN(input_size, hidden_size, self.n_layers)
        self.fc = nn.Linear(hidden_size, output_size)

        # self.inputToHidden = nn.Linear(input_size + hidden_size, hidden_size)
        # self.inputToOutput = nn.Linear(input_size + hidden_size, output_size)
        # self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, input, hidden):
        # combined = torch.stack((input, hidden), 1)
        # hidden = self.i2h(combined)
        # output = self.i2o(combined)
        # output = self.softmax(output)
        # batch_size = input.size(0)
        # hidden = self.init_hidden()
        output, hidden = self.rnn(input, hidden)
        output = output.contiguous().view(-1, self.hidden_size)
        output = self.fc(output)
        return output, hidden

    def init_hidden(self):
        hidden = torch.zeros(self.n_layers, self.hidden_size)


class GruRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(GruRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        
        self.encoder = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)
    
    def forward(self, input, hidden):
        input = self.encoder(input.view(1, -1))
        output, hidden = self.gru(input.view(1, 1, -1), hidden)
        output = self.decoder(output.view(1, -1))
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(self.n_layers, 1, self.hidden_size))