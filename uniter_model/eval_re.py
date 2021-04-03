# coding=utf-8
# copied from hugginface github
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc.
# team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT for Referring Expression Comprehension Evaluation"""
import argparse
import json
import os
from os.path import exists
from time import time

import torch
from torch.utils.data import DataLoader

# to be deprecated once upgraded to 1.2
# from torch.utils.data.distributed import DistributedSampler
from data import DistributedSampler

from apex import amp
from horovod import torch as hvd

from data import (ReImageFeatDir, ReferringExpressionEvalDataset,
                  re_eval_collate, PrefetchLoader)
from model import BertForReferringExpressionComprehension

from utils.logger import LOGGER
from utils.distributed import all_gather_list
from utils.misc import Struct


def main(opts):

    hvd.init()
    n_gpu = hvd.size()
    device = torch.device("cuda", hvd.local_rank())
    torch.cuda.set_device(hvd.local_rank())
    rank = hvd.rank()
    opts.rank = rank
    LOGGER.info(f"device: {device}, n_gpu: {n_gpu}, rank: {hvd.rank()}, "
                f"16-bits training: {opts.fp16}")

    hps_file = f'{opts.output_dir}/log/hps.json'
    model_opts = json.load(open(hps_file))
    if 'mlp' not in model_opts:
        model_opts['mlp'] = 1
    model_opts = Struct(model_opts)

    # Prepro txt_dbs
    txt_dbs = opts.txt_db.split(':')

    # Prepro model
    if exists(opts.checkpoint):
        ckpt_file = torch.load(opts.checkpoint)
    else:
        ckpt_file = f'{opts.output_dir}/ckpt/model_epoch_{opts.checkpoint}.pt'
    checkpoint = torch.load(ckpt_file)
    bert_model = json.load(open(f'{txt_dbs[0]}/meta.json'))['bert']
    model = BertForReferringExpressionComprehension.from_pretrained(
                bert_model, img_dim=2048, mlp=model_opts.mlp,
                state_dict=checkpoint
            )
    if model_opts.cut_bert != -1:
        # cut some layers of BERT
        model.bert.encoder.layer = torch.nn.ModuleList(
            model.bert.encoder.layer[:opts.cut_bert]
        )
    model.to(device)

    if opts.fp16:
        model = amp.initialize(model, enabled=opts.fp16, opt_level='O2')

    # load DBs and image dirs
    eval_img_dir = ReImageFeatDir(opts.img_dir)
    for txt_db in txt_dbs:
        print(f'Evaluating {txt_db}')
        eval_dataset = ReferringExpressionEvalDataset(txt_db, eval_img_dir,
                                                      max_txt_len=-1)
        eval_sampler = DistributedSampler(eval_dataset, num_replicas=n_gpu,
                                          rank=rank, shuffle=False)
        eval_dataloader = DataLoader(eval_dataset,
                                     sampler=eval_sampler,
                                     batch_size=opts.batch_size,
                                     num_workers=opts.n_workers,
                                     pin_memory=opts.pin_mem,
                                     collate_fn=re_eval_collate)
        eval_dataloader = PrefetchLoader(eval_dataloader)

        # evaluate
        val_log, results = validate(model, eval_dataloader)

        # save
        result_dir = f'{opts.output_dir}/results_test'
        if not exists(result_dir) and rank == 0:
            os.makedirs(result_dir)

        # dummy sync
        _ = None
        all_gather_list(_)
        db_split = txt_db.split('/')[-1].split('-')[0]  # refcoco+_val_large
        img_dir = opts.img_dir.split('/')[-1]  # visual_grounding_coco_gt
        if n_gpu > 1:
            with open(f'{opts.output_dir}/results_test/'
                      f'results_{opts.checkpoint}_{db_split}_on_{img_dir}'
                      f'_rank{rank}.json',
                      'w') as f:
                json.dump(results, f)
            # dummy sync
            _ = None
            all_gather_list(_)

        # join results
        if n_gpu > 1:
            results = []
            for rank in range(n_gpu):
                results.extend(json.load(open(
                    f'{opts.output_dir}/results_test/'
                    f'results_{opts.checkpoint}_{db_split}_on_{img_dir}'
                    f'_rank{rank}.json')))
        if rank == 0:
            with open(f'{opts.output_dir}/results_test/'
                      f'results_{opts.checkpoint}_{db_split}_on_{img_dir}'
                      f'_all.json', 'w') as f:
                json.dump(results, f)

        # print
        print(f'{opts.output_dir}/results_test')


@torch.no_grad()
def validate(model, val_dataloader):
    LOGGER.info(f"start running evaluation.")
    model.eval()
    tot_score = 0
    n_ex = 0
    st = time()
    predictions = []
    for i, batch in enumerate(val_dataloader):
        # inputs
        (*batch_inputs, tgt_box_list, obj_boxes_list, sent_ids) = batch

        # scores (n, max_num_bb)
        scores = model(*batch_inputs, targets=None, compute_loss=False)
        ixs = torch.argmax(scores, 1).cpu().detach().numpy()  # (n, )

        # pred_boxes
        for ix, obj_boxes, tgt_box, sent_id in \
                zip(ixs, obj_boxes_list, tgt_box_list, sent_ids):
            pred_box = obj_boxes[ix]
            predictions.append({'sent_id': sent_id,
                                'pred_box': pred_box.tolist(),
                                'tgt_box': tgt_box.tolist()})
            if (val_dataloader.loader.dataset.computeIoU(pred_box, tgt_box)
                    > .5):
                tot_score += 1
            n_ex += 1

    tot_time = time()-st
    tot_score = sum(all_gather_list(tot_score))
    n_ex = sum(all_gather_list(n_ex))
    val_acc = tot_score / n_ex
    val_log = {'valid/acc': val_acc, 'valid/ex_per_s': n_ex/tot_time}
    model.train()
    LOGGER.info(f"validation ({n_ex} sents) finished in "
                f"{int(tot_time)} seconds"
                f", accuracy: {val_acc*100:.2f}%")

    # summarizae
    results = {'acc': val_acc, 'predictions': predictions}
    return val_log, results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Requited parameters
    parser.add_argument('--txt_db',
                        default=None, type=str,
                        help="The input train corpus. (LMDB)")
    parser.add_argument('--img_dir',
                        default=None, type=str,
                        help="The input train images.")
    parser.add_argument('--checkpoint',
                        default=None, type=str,
                        help="pretrained model (can take 'google-bert')")
    parser.add_argument('--batch_size',
                        default=256, type=int,
                        help="number of sentences per batch")
    parser.add_argument('--output_dir',
                        default=None, type=str,
                        help="The output directory where the model contains "
                             "the model checkpoints will be written.")

    # Device parameters
    parser.add_argument('--fp16',
                        action='store_true',
                        help="whether to use fp-16 float precision instead of "
                             "32 bit")
    parser.add_argument('--n_workers', type=int, default=4,
                        help="number of data workers")
    parser.add_argument('--pin_mem', action='store_true',
                        help="pin memory")

    args = parser.parse_args()

    main(args)
