import copy
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._C import randperm
from torch.autograd import Variable
from tqdm import tqdm

import util.loadFiles as tr
from nn.DMax import DMax
from util.utils import MAP, MRR


class compAggWikiqa(nn.Module):
    def __init__(self, args):
        super(compAggWikiqa, self).__init__()

        self.mem_dim = args.mem_dim
        # self.att_dim = args.att_dim
        self.cov_dim = args.cov_dim
        self.learning_rate = args.learning_rate
        self.batch_size = args.batch_size
        self.emb_dim = args.wvecDim
        self.task = args.task
        self.numWords = args.numWords
        self.dropoutP = args.dropoutP
        self.grad = args.grad
        self.visualize = args.visualize
        self.emb_lr = args.emb_lr
        self.emb_partial = args.emb_partial
        self.comp_type = args.comp_type
        self.window_sizes = args.window_sizes
        self.window_large = args.window_large
        self.gpu = args.gpu

        self.best_score = 0

        self.emb_vecs = nn.Embedding(self.numWords, self.emb_dim)
        self.emb_vecs.weight.data = tr.loadVacab2Emb(self.task)

        # self.proj_module_master = self.new_proj_module()
        self.att_module_master = self.new_att_module()

        if self.comp_type == "mul":
            self.sim_sg_module = self.new_sim_mul_module()
        else:
            Exception("The word matching method is not provided")

        self.conv_module = self.new_conv_module()

        mem_dim = self.mem_dim

        class TempNet(nn.Module):
            def __init__(self, mem_dim):
                super(TempNet, self).__init__()
                self.layer1 = nn.Linear(mem_dim, 1)

            def forward(self, input):
                var1 = self.layer1(input)
                var1 = var1.view(-1)
                out = F.log_softmax(var1)
                return out

        self.soft_module = TempNet(mem_dim)

        # self.join_module = lambda x: torch.cat(x, 0)

        self.optim_state = {"learningRate": self.learning_rate}
        self.criterion = nn.KLDivLoss()

        # That's a bug in pytorch, Container.py's parameters()
        def get_container_parameters(container):
            w = []
            gw = []
            for module in container.modules:
                mparam = list(module.parameters())
                if mparam:
                    w.extend(mparam[0])
                    gw.extend(mparam[1])
            if not w:
                return
            return w, gw

        # self.params, self.grad_params = get_container_parameters(modules)
        # print("Parameter size: %s" % self.params[0].size())
        # self.best_params = self.params.copy()  # TODO: revisit
        self.dropout_modules = [None] * 2
        self.proj_modules = [None] * 2
        # self.att_modules = [None] * 2

        self.proj_modules = self.new_proj_module()
        self.dropout_modules = nn.Dropout(self.dropoutP)
        # for i in range(2):
        #     self.proj_modules[i] = new_proj_mod
        #     self.dropout_modules[i] = dropout_mod

    def new_proj_module(self):
        emb_dim = self.emb_dim
        mem_dim = self.mem_dim

        class NewProjModule(nn.Module):
            def __init__(self, emb_dim, mem_dim):
                super(NewProjModule, self).__init__()
                self.emb_dim = emb_dim
                self.mem_dim = mem_dim
                self.linear1 = nn.Linear(self.emb_dim, self.mem_dim)
                self.linear2 = nn.Linear(self.emb_dim, self.mem_dim)

            def forward(self, input):
                i = nn.Sigmoid()(self.linear1(input))
                u = nn.Tanh()(self.linear2(input))
                out = i.mul(u)  # CMulTable().updateOutput([i, u])
                return out

        module = NewProjModule(emb_dim, mem_dim)

        # if getattr(self, "proj_module_master", None):  # share parameters
        #     for (tar_param, src_param) in zip(module.parameters(), self.proj_module_master.parameters()):
        #         tar_param.grad.data = src_param.grad.data.clone()

        return module

    def new_att_module(self):

        class NewAttModule(nn.Module):
            def __init__(self):
                super(NewAttModule, self).__init__()

            def forward(self, linput, rinput):
                self.lPad = linput.view(-1, linput.size(0), linput.size(1))

                self.lPad = linput  # self.lPad = Padding(0, 0)(linput) TODO: figureout why padding?
                self.M_r = torch.mm(self.lPad, rinput.t())
                self.alpha = F.softmax(self.M_r.transpose(0, 1))
                self.Yl = torch.mm(self.alpha, self.lPad)
                return self.Yl

        att_module = NewAttModule()
        if getattr(self, "att_module_master", None):
            for (tar_param, src_param) in zip(att_module.parameters(), self.att_module_master.parameters()):
                tar_param.grad.data = src_param.grad.data.clone()
        return att_module

    def new_conv_module(self):
        window_sizes = self.window_sizes
        cov_dim = self.cov_dim
        mem_dim = self.mem_dim
        gpu = self.gpu

        class TemporalConvoluation(nn.Module):
            def __init__(self, cov_dim, mem_dim, window_size):
                super(TemporalConvoluation, self).__init__()
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
                weights = self.weight.view(-1, self.inp)  # weights applied to all
                bias = self.bias
                nOutputFrame = int((input.size(0) - self.kw) / self.dw + 1)

                output = Variable(torch.FloatTensor(nOutputFrame, self.outp))
                for i in range(input.size(0)):  # do -- for each sequence element
                    element = input[i]  # ; -- features of ith sequence element
                    output[i] = element.mm(weights) + bias
                return output

        class NewConvModule(nn.Module):
            def __init__(self, window_sizes, cov_dim, mem_dim):
                super(NewConvModule, self).__init__()
                self.window_sizes = window_sizes
                self.cov_dim = cov_dim
                self.mem_dim = mem_dim

                self.d_tempconv = {}
                self.d_dmax = {}
                for window_size in self.window_sizes:
                    self.d_tempconv[window_size] = TemporalConvoluation(self.cov_dim, self.mem_dim, window_size)
                    self.d_dmax[window_size] = DMax(dimension=0, windowSize=window_size, gpu=gpu)
                self.linear1 = nn.Linear(len(window_sizes) * mem_dim, mem_dim)
                self.relu1 = nn.ReLU()
                self.tanh1 = nn.Tanh()

            def forward(self, input, sizes):
                conv = [None] * len(self.window_sizes)
                pool = [None] * len(self.window_sizes)
                for i, window_size in enumerate(self.window_sizes):
                    tempconv = self.d_tempconv[window_size](input)
                    conv[i] = self.relu1(tempconv)
                    pool[i] = self.d_dmax[window_size](conv[i], sizes)
                concate = torch.cat(pool, 1)  # JoinTable(2).updateOutput(pool)
                linear1 = self.linear1(concate)
                output = self.tanh1(linear1)
                return output

        conv_module = NewConvModule(window_sizes, cov_dim, mem_dim)
        return conv_module

    def new_sim_mul_module(self):
        class NewSimMulModule(nn.Module):
            def __init__(self):
                super(NewSimMulModule, self).__init__()

            def forward(self, inputa, inputh):  # actually it's a_j vs h_j, element-wise mul
                return inputa.mul(inputh)  # return CMulTable().updateOutput([inputq, inputa])

        return NewSimMulModule()

    def forward(self, data_q, data_as):
        data_as_len = torch.IntTensor(len(data_as))
        for k in range(len(data_as)):
            data_as_len[k] = data_as[k].size()[0]
            if data_as_len[k] < self.window_large:
                as_tmp = torch.LongTensor(self.window_large).fill_(0)
                data_as[k] = as_tmp
                data_as_len[k] = self.window_large

        data_as_word = torch.cat(data_as, 0)
        inputs_a_emb = self.emb_vecs.forward(
            Variable(data_as_word.type(torch.LongTensor),
                     requires_grad=False))  # TODO: why LongTensor would convert to Float
        inputs_q_emb = self.emb_vecs.forward(Variable(data_q, requires_grad=False))

        # inputs_a_emb = self.dropout_modules[0].forward(inputs_a_emb)
        # inputs_q_emb = self.dropout_modules[1].forward(inputs_q_emb)
        inputs_a_emb = self.dropout_modules.forward(inputs_a_emb)
        inputs_q_emb = self.dropout_modules.forward(inputs_q_emb)

        # projs_a_emb = self.proj_modules[0].forward(inputs_a_emb)
        # projs_q_emb = self.proj_modules[1].forward(inputs_q_emb)
        projs_a_emb = self.proj_modules.forward(inputs_a_emb)
        projs_q_emb = self.proj_modules.forward(inputs_q_emb)

        if data_q.size()[0] == 1:
            projs_q_emb = projs_q_emb.resize(1, self.mem_dim)

        att_output = self.att_module_master.forward(projs_q_emb, projs_a_emb)
        sim_output = self.sim_sg_module.forward(projs_a_emb, att_output)

        conv_output = self.conv_module.forward(sim_output, data_as_len)
        soft_output = self.soft_module.forward(conv_output)
        # soft_output = nn.Linear(150, len(data_as))(conv_output).sum(0)[0]

        return soft_output

    def predict(self, data_raw):
        data_q, data_as, label = data_raw
        # label = Variable(label, requires_grad=False)

        data_as_len = torch.IntTensor(len(data_as))

        for k in range(len(data_as)):
            data_as_len[k] = data_as[k].size()[0]
            if data_as_len[k] < self.window_large:
                as_tmp = torch.LongTensor(self.window_large).fill_(0)
                as_tmp[0:data_as_len[k]] = copy.deepcopy(data_as[k])
                data_as[k] = as_tmp
                data_as_len[k] = self.window_large

        data_as_word = torch.cat(data_as, 0)
        inputs_a_emb = self.emb_vecs.forward(
            Variable(data_as_word.type(torch.LongTensor),
                     requires_grad=False))
        inputs_q_emb = self.emb_vecs.forward(Variable(data_q, requires_grad=False))

        # inputs_a_emb = self.dropout_modules[0].forward(inputs_a_emb)
        # inputs_q_emb = self.dropout_modules[1].forward(inputs_q_emb)
        inputs_a_emb = self.dropout_modules.forward(inputs_a_emb)
        inputs_q_emb = self.dropout_modules.forward(inputs_q_emb)

        # projs_a_emb = self.proj_modules[0].forward(inputs_a_emb)
        # projs_q_emb = self.proj_modules[1].forward(inputs_q_emb)
        projs_a_emb = self.proj_modules.forward(inputs_a_emb)
        projs_q_emb = self.proj_modules.forward(inputs_q_emb)

        if data_q.size()[0] == 1:
            projs_q_emb.resize(1, self.mem_dim)  # revisit

        att_output = self.att_module_master.forward(projs_q_emb, projs_a_emb)

        sim_output = self.sim_sg_module.forward(projs_a_emb, att_output)

        conv_output = self.conv_module.forward(sim_output, data_as_len)
        soft_output = self.soft_module.forward(conv_output)
        map = MAP(label, soft_output.data)
        mrr = MRR(label, soft_output.data)
        return map, mrr

    def predict_dataset(self, dataset):

        self.proj_modules.eval()
        self.dropout_modules.eval()
        # for i in range(2):
        #     self.proj_modules[i].eval()
        #     self.dropout_modules[i].eval()

        self.emb_vecs.eval()
        self.conv_module.eval()
        res = [0., 0.]
        dataset_size = len(dataset)
        for i in range(dataset_size):
            prediction = self.predict(dataset[i])
            res[0] = res[0] + prediction[0]
            res[1] = res[1] + prediction[1]

        res[0] = res[0] / dataset_size
        res[1] = res[1] / dataset_size

        return res

    def save(self, path, config, result, epoch):
        assert os.path.isdir(path)
        paraPath = path + config.task + str(config.expIdx)
        paraBestPath = path + config.task + str(config.expIdx) + '_best'
        recPath = path + config.task + str(config.expIdx) + 'Record.txt'

        file = open(recPath, 'a')
        if epoch == 0:
            for name, val in vars(config).items():
                file.write(name + '\t' + str(val) + '\n')

        file.write(config.task + ': ' + str(epoch) + ': ')
        for i, vals in enumerate(result):
            for _, val in enumerate(vals):
                file.write('%s, ' % val)

            if i == 0:
                print("Dev: MAP: %s, MRR: %s" % (vals[0], vals[1]))
            elif i == 1:
                print("Test: MAP: %s, MRR: %s" % (vals[0], vals[1]))
            else:
                print("Train: MAP: %s, MRR: %s" % (vals[0], vals[1]))

        file.write('\n')
        file.close()

        # if result[0][0] > self.best_score:
        #     self.best_score = result[0][0]
        #     self.best_params = copy.deepcopy(self.params)
        #     torch.save({"params": self.params, "config": config}, paraBestPath)
        #
        # torch.save({"params": self.params, "config": config}, paraBestPath)


def train(model: compAggWikiqa, dataset: list):
    model.proj_modules.train()
    model.dropout_modules.train()
    # for i in range(2):
    #     model.proj_modules[i].train()
    #     model.dropout_modules[i].train()

    model.emb_vecs.train()
    model.conv_module.train()

    dataset_size = len(dataset)
    indices = randperm(dataset_size)
    indices = [667] + [x for x in range(667)] + [x for x in range(668, 873)]  # TODO: remove me

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for i in tqdm(range(0, dataset_size, model.batch_size)):  # TODO change me for debug: # 1, model.batch_size):
        batch_size = min(model.batch_size,
                         dataset_size - i)  # min(i + model.batch_size - 1, dataset_size) - i + 1  # TODO: why?

        loss = 0.
        for j in range(0, batch_size):
            idx = indices[i + j]
            data_raw = dataset[idx]
            data_q, data_as, label = data_raw
            label = Variable(label, requires_grad=False)
            soft_output = model(data_q, data_as)

            example_loss = model.criterion(soft_output, label)

            loss += example_loss
        loss = loss / batch_size

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print(i, loss.data[0])
