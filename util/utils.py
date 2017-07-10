import torch


def MAP(ground_label: torch.FloatTensor, predict_label: torch.FloatTensor):
    map = 0
    map_idx = 0
    extracted = {}

    for idx_, glab in enumerate(ground_label):
        if ground_label[idx_] != 0:
            extracted[idx_] = 1

    val, key = torch.sort(predict_label, 0, True)
    for i, idx_ in enumerate(key):
        if idx_ in extracted:
            map_idx += 1
            map += map_idx / (i + 1)

    assert (map_idx != 0)
    map = map / map_idx
    return map


def MRR(ground_label: torch.FloatTensor, predict_label: torch.FloatTensor):
    mrr = 0
    map_idx = 0
    extracted = {}

    for idx_, glab in enumerate(ground_label):
        if ground_label[idx_] != 0:
            extracted[idx_] = 1

    val, key = torch.sort(predict_label, 0, True)
    for i, idx_ in enumerate(key):
        if idx_ in extracted:
            mrr = 1.0 / (i + 1)
            break

    assert (mrr != 0)
    return mrr
