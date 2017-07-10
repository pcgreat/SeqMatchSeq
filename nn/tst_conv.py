import copy
import math
import pdb

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.legacy.nn import TemporalConvolution

window_sizes = [1, 2, 3, 4, 5]
cov_dim = 150
mem_dim = 150


class MyTemporalConvoluation(nn.Module):
    def __init__(self, cov_dim, mem_dim, window_size):
        super(MyTemporalConvoluation, self).__init__()
        self.conv1 = nn.Conv1d(cov_dim, mem_dim, window_size)

    def forward(self, input):
        myinput = input.view(1, input.size()[0], input.size()[1]).transpose(1, 2)  # 1, 150, 56
        output = self.conv1(myinput)[0].transpose(0, 1)  # 56, 150
        return output


class MyTemporalConvoluation2(nn.Module):
    def __init__(self, cov_dim, mem_dim, window_size):
        super(MyTemporalConvoluation2, self).__init__()
        self.inp, self.outp, self.kw, self.dw = cov_dim, mem_dim, window_size, 1

        self.weight = Variable(torch.Tensor(self.outp, self.inp * self.kw), requires_grad=False)
        self.bias = Variable(torch.Tensor(self.outp), requires_grad=False)
        self.reset()

    def reset(self, stdv=None):
        if stdv is not None:
            stdv = stdv * math.sqrt(3)
        else:
            stdv = 1. / math.sqrt(self.kw * self.inp)

        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        weights = self.weight.view(self.inp, self.outp * self.kw)  # weights applied to all
        bias = self.bias
        nOutputFrame = int((input.size(0) - self.kw) / self.dw + 1)

        output = Variable(torch.FloatTensor(nOutputFrame, self.outp))
        for i in range(input.size(0)):  # do -- for each sequence element
            element = input[i]  # ; -- features of ith sequence element
            output[i] = torch.dot(element, weights) + bias
        return output

window_size = 1
input = Variable(torch.FloatTensor(158, 150))
input.data.uniform_(-1, 1)
myinput = copy.deepcopy(input)
temp1 = TemporalConvolution(cov_dim, mem_dim, window_size).forward(myinput.data)
temp2 = MyTemporalConvoluation2(cov_dim, mem_dim, window_size)(myinput)
print(temp1.size())
print(temp2.size())

pdb.set_trace()

class NewConvModule(nn.Module):
    def __init__(self, window_sizes, cov_dim, mem_dim):
        super(NewConvModule, self).__init__()
        self.window_sizes = window_sizes
        self.cov_dim = cov_dim
        self.mem_dim = mem_dim

        self.linear1 = nn.Linear(len(window_sizes) * mem_dim, mem_dim)
        self.relu1 = nn.ReLU()
        self.tanh1 = nn.Tanh()

    def forward(self, input, sizes):
        conv = [None] * len(self.window_sizes)
        pool = [None] * len(self.window_sizes)
        for i, window_size in enumerate(self.window_sizes):
            tempconv = TemporalConvoluation(window_size, self.cov_dim, self.mem_dim)(input)
            conv[i] = self.relu1(tempconv)
            pool[i] = DMax(dimension=0, windowSize=window_size, gpu=gpu)(conv[i], sizes)
        concate = torch.cat(pool, 1)  # JoinTable(2).updateOutput(pool)
        linear1 = self.linear1(concate)
        output = self.tanh1(linear1)
        return output
