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
"""BERT for Referring Expression Comprehension"""
import argparse
import json
import os
from os.path import exists, join
import random
from time import time

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam, Adamax
from torch.utils.data import DataLoader

# to be deprecated once upgraded to 1.2
# from torch.utils.data.distributed import DistributedSampler
from data import DistributedSampler

from apex import amp
from horovod import torch as hvd

import numpy as np
from tqdm import tqdm

from data import (ReImageFeatDir, ReferringExpressionDataset,
                  ReferringExpressionEvalDataset, re_collate, re_eval_collate,
                  PrefetchLoader)
from model import BertForReferringExpressionComprehension
from optim import warmup_linear, noam_schedule, AdamW

from utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file
from utils.distributed import (all_reduce_and_rescale_tensors, all_gather_list,
                               broadcast_tensors)
from utils.save import ModelSaver, save_training_meta
from utils.misc import NoOp, parse_with_config


def main(opts):

    hvd.init()
    n_gpu = hvd.size()
    device = torch.device("cuda", hvd.local_rank())
    torch.cuda.set_device(hvd.local_rank())
    rank = hvd.rank()
    opts.rank = rank
    LOGGER.info(f"device: {device}, n_gpu: {n_gpu}, rank: {hvd.rank()}, "
                f"16-bits training: {opts.fp16}")

    if opts.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, "
                         "should be >= 1".format(
                            opts.gradient_accumulation_steps))

    random.seed(opts.seed)
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(opts.seed)

    # train_samples = None
    LOGGER.info(f"Loading Train Dataset {opts.train_txt_db}, "
                f"{opts.train_img_dir}")

    # load DBs and image dirs
    train_img_dir = ReImageFeatDir(opts.train_img_dir)
    train_dataset = ReferringExpressionDataset(
                        opts.train_txt_db, train_img_dir,
                        max_txt_len=opts.max_txt_len)
    val_img_dir = ReImageFeatDir(opts.val_img_dir)
    val_dataset = ReferringExpressionEvalDataset(
                        opts.val_txt_db, val_img_dir,
                        max_txt_len=opts.max_txt_len)

    # Prepro model
    if opts.checkpoint and opts.checkpoint != 'scratch':
        if opts.checkpoint == 'google-bert':
            # from google-bert
            checkpoint = None
        else:
            checkpoint = torch.load(opts.checkpoint)
    else:
        # from scratch
        checkpoint = {}
    bert_model = json.load(open(f'{opts.train_txt_db}/meta.json'))['bert']
    model = BertForReferringExpressionComprehension.from_pretrained(
                bert_model, img_dim=2048,
                loss=opts.train_loss,
                margin=opts.margin,
                hard_ratio=opts.hard_ratio,
                mlp=opts.mlp,
                state_dict=checkpoint
            )
    if opts.cut_bert != -1:
        # cut some layers of BERT
        model.bert.encoder.layer = torch.nn.ModuleList(
            model.bert.encoder.layer[:opts.cut_bert]
        )
    del checkpoint
    for name, module in model.named_modules():
        # we may want to tune dropout for smaller dataset
        if isinstance(module, torch.nn.Dropout):
            if module.p != opts.dropout:
                module.p = opts.dropout
                LOGGER.info(f'{name} set to {opts.dropout}')
    model.to(device)

    # make sure every process has same model params in the beginning
    broadcast_tensors([p.data for p in model.parameters()], 0)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': opts.weight_decay},
        {'params': [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    # currently Adam only
    if opts.optim == 'adam':
        OptimCls = Adam
    elif opts.optim == 'adamax':
        OptimCls = Adamax
    elif opts.optim == 'adamw':
        OptimCls = AdamW
    else:
        raise ValueError('invalid optimizer')
    optimizer = OptimCls(optimizer_grouped_parameters,
                         lr=opts.learning_rate, betas=opts.betas)
    model, optimizer = amp.initialize(model, optimizer, enabled=opts.fp16,
                                      opt_level='O2')

    global_step = 0
    LOGGER.info("***** Running training *****")
    LOGGER.info("  Num examples = %d", len(train_dataset))
    LOGGER.info("  Batch size = %d", opts.train_batch_size)
    LOGGER.info("  Accumulate steps = %d", opts.gradient_accumulation_steps)
    LOGGER.info("  Num steps = %d", opts.num_train_steps)

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=n_gpu, rank=rank, shuffle=False)
    train_dataloader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  batch_size=opts.train_batch_size,
                                  num_workers=opts.n_workers,
                                  pin_memory=opts.pin_mem,
                                  collate_fn=re_collate)
    train_dataloader = PrefetchLoader(train_dataloader)

    val_sampler = DistributedSampler(
        val_dataset, num_replicas=n_gpu, rank=rank, shuffle=False)
    val_dataloader = DataLoader(val_dataset,
                                sampler=val_sampler,
                                batch_size=opts.val_batch_size,
                                num_workers=opts.n_workers,
                                pin_memory=opts.pin_mem,
                                collate_fn=re_eval_collate)
    val_dataloader = PrefetchLoader(val_dataloader)

    if rank == 0:
        save_training_meta(opts)
        TB_LOGGER.create(join(opts.output_dir, 'log'))
        pbar = tqdm(total=opts.num_train_steps)
        model_saver = ModelSaver(join(opts.output_dir, 'ckpt'), 'model_epoch')
        os.makedirs(join(opts.output_dir, 'results'))  # store ITM predictions
        add_log_to_file(join(opts.output_dir, 'log', 'log.txt'))
    else:
        LOGGER.disabled = True
        pbar = NoOp()
        model_saver = NoOp()
    running_loss = RunningMeter(opts.train_loss)
    n_examples = 0
    n_epoch = 0
    best_val_acc, best_epoch = None, None
    start = time()
    # quick hack for amp delay_unscale bug
    optimizer.zero_grad()
    optimizer.step()

    while True:
        model.train()
        for step, batch in enumerate(train_dataloader):
            if global_step >= opts.num_train_steps:
                break

            *_, targets = batch
            n_examples += targets.size(0)
            loss = model(*batch, compute_loss=True)
            loss = loss.sum()  # sum over vectorized loss TODO: investigate
            delay_unscale = (step+1) % opts.gradient_accumulation_steps != 0
            with amp.scale_loss(loss, optimizer, delay_unscale=delay_unscale
                                ) as scaled_loss:
                scaled_loss.backward()
                if not delay_unscale:
                    # gather gradients from every processes
                    # do this before unscaling to make sure every process uses
                    # the same gradient scale
                    grads = [p.grad.data for p in model.parameters()
                             if p.requires_grad and p.grad is not None]
                    all_reduce_and_rescale_tensors(grads, float(1))

            running_loss(loss.item())
            if (step + 1) % opts.gradient_accumulation_steps == 0:
                global_step += 1

                # learning rate scheduling
                if opts.decay == 'linear':
                    lr_this_step = opts.learning_rate * warmup_linear(
                        global_step, opts.warmup_steps, opts.num_train_steps)
                elif opts.decay == 'invsqrt':
                    lr_this_step = opts.learning_rate * noam_schedule(
                        global_step, opts.warmup_steps)
                elif opts.decay == 'constant':
                    lr_this_step = opts.learning_rate
                if lr_this_step < 0:
                    # save guard for possible miscalculation of train steps
                    lr_this_step = 1e-8
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                TB_LOGGER.add_scalar('lr', lr_this_step, global_step)

                # log loss
                losses = all_gather_list(running_loss)
                running_loss = RunningMeter(
                    opts.train_loss, sum(l.val for l in losses)/len(losses))
                TB_LOGGER.add_scalar('loss_'+opts.train_loss, running_loss.val,
                                     global_step)
                TB_LOGGER.step()

                # update model params
                if opts.grad_norm != -1:
                    grad_norm = clip_grad_norm_(amp.master_params(optimizer),
                                                opts.grad_norm)
                    TB_LOGGER.add_scalar('grad_norm', grad_norm, global_step)
                optimizer.step()
                optimizer.zero_grad()
                pbar.update(1)

                if global_step % 5 == 0:
                    torch.cuda.empty_cache()
                if global_step % 100 == 0:
                    # monitor training throughput
                    tot_ex = sum(all_gather_list(n_examples))
                    ex_per_sec = int(tot_ex / (time()-start))
                    LOGGER.info(f'{tot_ex} examples trained at '
                                f'{ex_per_sec} ex/s')
                    TB_LOGGER.add_scalar('perf/ex_per_s',
                                         ex_per_sec, global_step)
        # evaluate after each epoch
        val_log, _ = validate(model, val_dataloader)
        TB_LOGGER.log_scaler_dict(val_log)

        # save model
        n_epoch += 1
        model_saver.save(model, n_epoch)
        LOGGER.info(f"finished {n_epoch} epochs")

        # save best model
        if best_val_acc is None or val_log['valid/acc'] > best_val_acc:
            best_val_acc = val_log['valid/acc']
            best_epoch = n_epoch
            model_saver.save(model, 'best')

        # shuffle training data for the next epoch
        train_dataloader.loader.dataset.shuffle()

        # is training finished?
        if global_step >= opts.num_train_steps:
            break

    val_log, results = validate(model, val_dataloader)
    with open(f'{opts.output_dir}/results/'
              f'results_{global_step}_'
              f'rank{rank}_final.json', 'w') as f:
        json.dump(results, f)
    TB_LOGGER.log_scaler_dict(val_log)
    model_saver.save(model, f'{global_step}_final')

    # print best model
    LOGGER.info(f'best_val_acc = {best_val_acc*100:.2f}% '
                f'at epoch {best_epoch}.')


@torch.no_grad()
def validate(model, val_dataloader):
    LOGGER.info(f"start running evaluation.")
    model.eval()
    tot_score = 0
    n_ex = 0
    st = time()
    predictions = {}
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
            predictions['sent_id'] = {'pred_box': pred_box.tolist(),
                                      'tgt_box': tgt_box.tolist()}
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
    return val_log, predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--train_txt_db",
                        default=None, type=str,
                        help="The input train corpus. (LMDB)")
    parser.add_argument("--train_img_dir",
                        default=None, type=str,
                        help="The input train images.")
    parser.add_argument("--val_txt_db",
                        default=None, type=str,
                        help="The input validation corpus. (LMDB)")
    parser.add_argument("--val_img_dir",
                        default=None, type=str,
                        help="The input validation images.")
    parser.add_argument('--img_format', default='npz',
                        choices=['npz', 'lmdb', 'lmdb-compress'],
                        help='format of image feature')
    parser.add_argument("--checkpoint",
                        default=None, type=str,
                        help="pretrained model (can take 'google-bert') ")
    parser.add_argument("--cut_bert", default=-1, type=int,
                        help="reduce BERT layers (-1 for original depth)")
    parser.add_argument("--mlp", default=1, type=int,
                        help="number of MLP layers for RE output")

    parser.add_argument(
        "--output_dir", default=None, type=str,
        help="The output directory where the model checkpoints will be "
             "written.")

    # Prepro parameters
    parser.add_argument('--max_txt_len', type=int, default=60,
                        help='max number of tokens in text (BERT BPE)')

    # training parameters
    parser.add_argument("--train_batch_size",
                        default=128, type=int,
                        help="Total batch size for training. "
                             "(batch by examples)")
    parser.add_argument("--val_batch_size",
                        default=256, type=int,
                        help="Total batch size for validation. "
                             "(batch by tokens)")
    parser.add_argument("--train_loss",
                        default="cls", type=str,
                        choices=['cls', 'rank'],
                        help="loss to used during training")
    parser.add_argument("--margin",
                        default=0.2, type=float,
                        help="margin of ranking loss")
    parser.add_argument("--hard_ratio",
                        default=0.3, type=float,
                        help="sampling ratio of hard negatives")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=16,
                        help="Number of updates steps to accumualte before "
                             "performing a backward/update pass.")
    parser.add_argument("--learning_rate",
                        default=3e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_steps",
                        default=32000,
                        type=int,
                        help="Total number of training updates to perform.")
    parser.add_argument("--optim", default='adam',
                        choices=['adam', 'adamax', 'adamw'],
                        help="optimizer")
    parser.add_argument("--betas", default=[0.9, 0.98], nargs='+', type=float,
                        help="beta for adam optimizer")
    parser.add_argument("--decay", default='linear',
                        choices=['linear', 'invsqrt', 'constant'],
                        help="learning rate decay method")
    parser.add_argument("--dropout",
                        default=0.1,
                        type=float,
                        help="tune dropout regularization")
    parser.add_argument("--weight_decay",
                        default=0.0,
                        type=float,
                        help="weight decay (L2) regularization")
    parser.add_argument("--grad_norm",
                        default=0.25,
                        type=float,
                        help="gradient clipping (-1 for no clipping)")
    parser.add_argument("--warmup_steps",
                        default=4000,
                        type=int,
                        help="Number of training steps to perform linear "
                             "learning rate warmup for. (invsqrt decay)")

    # device parameters
    parser.add_argument('--seed',
                        type=int,
                        default=24,
                        help="random seed for initialization")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead "
                             "of 32-bit")
    parser.add_argument('--n_workers', type=int, default=4,
                        help="number of data workers")
    parser.add_argument('--pin_mem', action='store_true',
                        help="pin memory")

    # can use config files
    parser.add_argument('--config', help='JSON config files')

    args = parse_with_config(parser)

    if exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not "
                         "empty.".format(args.output_dir))

    # options safe guard
    main(args)
