import torch
import torch.nn as nn
from torch.autograd import Variable


class DMax(nn.Module):
    def __init__(self, dimension, windowSize, gpu):
        super(DMax, self).__init__()
        self.dimension = dimension
        self.windowSize = windowSize
        self.gpu = gpu
        self.max_modules = []
        self.gradInput = [torch.FloatTensor()]

    def forward(self, input, sizes):
        # input, sizes = inputs
        while len(self.max_modules) < sizes.size()[0]:
            self.max_modules.append(None)

        output = Variable(torch.FloatTensor(sizes.size()[0], input.size()[1]))
        start_idx = 0
        for i in range(0, sizes.size()[0]):
            max_module = self.max_modules[i]
            if max_module is None:
                if self.gpu:
                    self.max_modules[i] = lambda x: torch.max(x, dim=self.dimension)[0].cuda()
                else:  # max will return two vecs, one for value, the other for index, and then (1,N) => N
                    self.max_modules[i] = lambda x: torch.max(x, dim=self.dimension)[0]
                max_module = self.max_modules[i]
            output[i] = max_module(input[start_idx: start_idx + sizes[i] - self.windowSize + 1])
            start_idx = start_idx + sizes[i]
        return output

        # def updateGradInput(self, inputs, gradOutput):
        #     input, sizes = inputs
        #     self.gradInput[0].resizeAs(input).zero()
        #     self.gradInput[1] = self.gradInput[1].resizeAs(sizes).zero() \
        #         if self.gradInput[0] \
        #         else sizes.new().resizeAs(sizes).zero()
        #     start_idx = 0
        #     for i in range(0, len(sizes)):
        #         max_module = self.max_modules[i]
        #         assert max_module is not None
        #         # TODO: figure out what is here
        #         # self.gradInput[0]
        #         start_idx += sizes[i]
        #     return self.gradInput
