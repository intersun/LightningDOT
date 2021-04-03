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
	total_step = 100

	random.seed(seed)
	torch.manual_seed(seed)
	global_step = 0
	loader = DataLoader(vsize, ncls)
	examples = []
	print ("example generating")
	for step, (input_, target) in enumerate(loader):
		print ("example appended" + str(step))
		examples.append(InputExample(input=input_, target = target))
		global_step += 1
		if global_step >= total_step:
			break
	print ("saving torch.save")
	torch.save(examples, 'data/test_data/input0.txt')

	examples = torch.load('data/test_data/input.txt')
	for step, ie in enumerate(examples):
		print (step)
		print (ie.input)
		print (ie.target)

if __name__ == '__main__':
	main()


