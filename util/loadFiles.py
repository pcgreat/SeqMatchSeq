import os
import pdb

import numpy as np
import torch


def init(opt):
    if not os.path.isfile("./data/" + opt.task + "/vocab_py.t7"):
        buildVocab(opt.task)

    if not os.path.isfile("./data/" + opt.task + "/sequence/train_py.t7"):
        if opt.task == 'snli':
            buildData('dev', opt.task)
            buildData('test', opt.task)
            buildData('train', opt.task)
        else:
            buildData('all', opt.task)

    if not os.path.isfile("./data/" + opt.task + "/initEmb_py.t7"):
        buildVacab2Emb(opt)


def buildVocab(task):
    print("Building vocab dict ...")
    if task == 'snli':
        pass  # TODO: add back
    elif task == 'squad':
        pass  # TODO: add back
    elif task == 'wikiqa':
        vocab = {}
        ivocab = {}
        vocab['NULL'] = 0
        ivocab[0] = 'NULL'
        filenames = ["./data/" + task + "/WikiQACorpus/WikiQA-train.txt",
                     "./data/" + task + "/WikiQACorpus/WikiQA-dev.txt",
                     "./data/" + task + "/WikiQACorpus/WikiQA-test.txt"]
        for filename in filenames:
            for line in open(filename):
                divs = line.split('\t')
                for div_ in divs[:2]:  # e.g. divs = [question, label, 0/1]
                    words = div_.lower().split(' ')
                    for word in words:
                        if word not in vocab:
                            curr_idx = len(vocab)
                            vocab[word] = curr_idx
                            ivocab[curr_idx] = word

        print(len(ivocab))
        torch.save(vocab, "./data/" + task + "/vocab_py.t7")
        torch.save(ivocab, "./data/" + task + "/ivocab_py.t7")
    else:
        raise Exception('The specified task is not supported yet!')


def loadVocab(task):
    return torch.load("./data/%s/vocab_py.t7" % task)


def loadiVocab(task):
    return torch.load("./data/" + task + "/ivocab_py.t7")


def loadVacab2Emb(task):
    print("Loading embedding ...")
    return torch.load("./data/" + task + "/initEmb_py.t7")


def loadData(filename, task):
    print("Loading data " + filename + "...")
    return torch.load("./data/" + task + "/sequence/" + filename + "_py.t7")


def buildVacab2Emb(opt):
    vocab = loadVocab(opt.task)
    ivocab = loadiVocab(opt.task)

    emb = torch.randn(len(ivocab), opt.wvecDim) * 0.05
    if opt.task != 'snli':
        pass  # emb.zero()  TODO: not sure what it's doing

    print("Loading " + opt.preEmb + " ...")
    if opt.preEmb == 'glove':
        if opt.wvecDim in [300]:  # TODO: support 50, 100, 200 as well
            file = open("./data/" + opt.preEmb + "/glove.840B.%sd.txt" % opt.wvecDim, 'r')
        else:
            raise Exception("Glove doesn't have %s dimension embeddings" % opt.wvecDim)
    else:
        raise Exception("PreEmb not available: %s" % opt.preEmb)

    embRec = {}  # w_idx: 1
    for line in file.readlines():
        vals = line.rstrip("\n").split(' ')
        word = vals.pop(0)
        assert len(vals) == opt.wvecDim, pdb.set_trace()

        if word in vocab:
            emb[vocab[word], :] = torch.FloatTensor([float(v) for v in vals])

            embRec[vocab[word]] = 1

    print("Number of words not appear in " + opt.preEmb + ": " + str(len(vocab) - len(embRec)))
    if opt.task == 'snli':
        pass  # TODO
    torch.save(emb, "./data/" + opt.task + "/initEmb_py.t7")
    torch.save(embRec, "./data/" + opt.task + "/unUpdateVocab_py.t7")


def buildData(filename, task):
    dataset = {}
    vocab = loadVocab(task)
    print("Building " + task + " " + filename + " data ...")

    if task == 'snli':
        pass  # TODO
    elif task == 'squad':
        pass  # TODO
    elif task == 'wikiqa':
        filenames = {
            "train": './data/' + task + '/WikiQACorpus/WikiQA-train.txt',
            "dev": './data/' + task + '/WikiQACorpus/WikiQA-dev.txt',
            "test": './data/' + task + '/WikiQACorpus/WikiQA-test.txt'
        }
        for folder, filename in filenames.items():
            data = []
            # Start a fresh instance, candidates, labels
            question, candidates, labels = [], [], []

            for line in open(filename):
                divs = line.rstrip('\n').lower().split('\t')
                q_div, a_div, label = divs

                # parse q_div, a_div and label
                curr_q_words = [vocab[w] for w in q_div.split(' ')]
                curr_a_words = [vocab[w] for w in a_div.split(' ')]
                curr_label = float(label)

                if curr_q_words == question:  # the candidates this line is same with previous
                    pass
                else:
                    if not all(labels):
                        pass
                    if sum(
                            labels) > 0:  # if True, put previous instance into data; you need at least one positive label
                        question_ = torch.LongTensor(question)
                        candidates_ = [torch.LongTensor(words) for words in candidates]
                        labels_ = torch.FloatTensor(np.array(labels) / np.sum(labels))
                        data.append((question_, candidates_, labels_))

                    # Start a fresh instance, candidates, labels
                    question, candidates, labels = [], [], []
                    # add parsed q_div into question
                    question = curr_q_words

                # add parsed a_div, label into candidates, labels
                candidates.append(curr_a_words)
                labels.append(curr_label)

            torch.save(data, "./data/" + task + "/sequence/" + folder + '_py.t7')

    return dataset
