# __author__ = sushil(sushil79g@gmail.com)

#I understand therefore i can code

import torch
import torch.nn as nn
import torch.nn.parameter as parameter

class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.input_size_ = input_size
        self.hidden_size_ = hidden_size

        #Gate for input
        self.wii = parameter(torch.Tensor(input_size, hidden_size))
        self.whi = parameter(torch.Tensor(hidden_size,hidden_size))
        self.bi  = parameter(torch.Tensor(hidden_size))

        #gate for output
        self.wio = parameter(torch.Tensor(input_size, hidden_size))
        self.who = parameter(torch.Tensor(hidden_size,hidden_size))
        self.bo  = parameter(torch.Tensor(hidden_size))

        #gate for forget
        self.wif = parameter(torch.Tensor(input_size, hidden_size))
        self.whf = parameter(torch.Tensor(hidden_size,hidden_size))
        self.bf  = parameter(torch.Tensor(hidden_size))

        #info carried
        self.wig = parameter(torch.Tensor(input_size,hidden_size))
        self.whg = parameter(torch.Tensor(hidden_size,hidden_size))
        self.bg  = parameter(torch.Tensor(hidden_size))

    def forward(self, x):
        #lets say input x have shape of batch*sequence*feature
        batch_size, sequenxe_size, feature  = x.shape()
        hidden_sequence = []
        ht, ct = torch.zeros(self.hidden_size_), torch.zeros(self.hidden_size_)

        for inx in range(sequenxe_size):
            xt = x[:,1,:]
            it = torch.sigmoid(xt@self.wii + ht@self.whi + self.bi)
            ft = torch.sigmoid(xt@self.wif + ht@self.whf + self.bf)
            gt = torch.tanh(xt@self.wig + ht@self.whg + self.bg)
            ot = torch.sigmoid(xt@self.wio + ht@self.who + self.bo)
            ct = ft*ct + it*gt
            ht = ot*torch.tanh(ct)
            hidden_sequence.append(ht.unsqueeze(0))

        hidden_sequence = torch.cat(hidden_sequence, dim=0)
        hidden_sequence = hidden_sequence.transpose(0,1).contiguous()

        return hidden_sequence, (ht, ct)