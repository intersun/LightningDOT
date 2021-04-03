"""
minimal running script of distributed training
"""
import argparse
import random

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch import optim

from horovod import torch as hvd

# communication operations
from utils.distributed import all_reduce_and_rescale_tensors, all_gather_list


class DataLoader(object):
    def __init__(self, vocab_size, n_class, batch_size=8, lengths=(5, 10)):
        self.vsize = vocab_size
        self.ncls = n_class
        self.bs = batch_size
        self.lengths = lengths

    def __iter__(self):
        while True:
            input_, target = self._random_batch()
            yield input_, target

    def _random_batch(self):
        inputs = []
        targets = []
        for _ in range(self.bs):
            i, t = self._random_inputs()
            inputs.append(i)
            targets.append(t)
        input_ = pad_sequence(inputs)
        targets = torch.LongTensor(targets)
        return input_, targets

    def _random_inputs(self):
        len_ = random.randint(*self.lengths)
        inputs = [random.randint(0, self.vsize-1) for _ in range(len_)]
        target = random.randint(0, self.ncls-1)
        return torch.LongTensor(inputs), target


class Model(nn.Module):
    def __init__(self, vsize, ncls):
        super().__init__()
        self.emb = nn.Embedding(vsize, 100)
        self.rnn = nn.LSTM(100, 100, 1)
        self.proj = nn.Linear(100, ncls)

    def forward(self, input_):
        emb_out = self.emb(input_)
        _, (h, c) = self.rnn(emb_out)
        output = self.proj(h[-1])
        return output

class InputExample(object):
    def __init__(self, input, target):
        self.input = input
        self.target = target

def main():
    vsize = 200
    ncls = 10
    accum = 4
    total_step = 100
    seed = 777

    # distributed initialization
    hvd.init()
    n_gpu = hvd.size()
    device = torch.device("cuda", hvd.local_rank())
    local_rank = hvd.rank()

    random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    #loader = DataLoader(vsize, ncls)
    model = Model(vsize, ncls).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    global_step = 0

    print ("local_rank" + str(local_rank))
    examples = torch.load('data/test_data/input'+str(local_rank)+'.txt')

    for step, ie in enumerate(examples):
        input_ = ie.input
        target = ie.target
        input_ = input_.to(device)
        target = target.to(device)
        logit = model(input_)
        loss = F.cross_entropy(logit, target, reduction='sum')
        losses = all_gather_list(loss.item())
        #losses = [loss.item()]
        loss.backward()
        if (step+1) % accum == 0:
            if local_rank != -1:
                grads = [p.grad.data for p in model.parameters()
                         if p is not None and p.requires_grad]
                all_reduce_and_rescale_tensors(grads, 1)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            if local_rank <= 0:
                print(f'step: {global_step}; loss: {sum(losses)}')
        if global_step >= total_step:
            break


if __name__ == '__main__':
    main()
