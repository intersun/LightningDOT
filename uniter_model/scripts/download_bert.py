"""
Download and extract PyTorch pretrained BERT model
python scripts/download_bert.py bert-base-cased /pretrain/bert-base-cased.pt
"""
import sys

import torch
from pytorch_pretrained_bert import BertForPreTraining

bert, output = sys.argv[1:]
model = BertForPreTraining.from_pretrained(bert)
torch.save(model.state_dict(), output)
