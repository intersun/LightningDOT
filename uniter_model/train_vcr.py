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
"""BERT pre-training runner."""
import argparse
import json
import os
from os.path import exists, join
import random
from time import time

import torch
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam, Adamax
from torch.utils.data import DataLoader, ConcatDataset

from apex import amp
from horovod import torch as hvd

import numpy as np
from tqdm import tqdm

from data import (DistributedTokenBucketSampler,
                  DetectFeatLmdb, VcrDataset, VcrEvalDataset,
                  vcr_collate, vcr_eval_collate,
                  PrefetchLoader)
from model import BertForVisualCommonsenseReasoning
from optim import warmup_linear, noam_schedule, vqa_schedule, AdamW
from torch.utils.data.distributed import DistributedSampler

from utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file
from utils.distributed import (all_reduce_and_rescale_tensors, all_gather_list,
                               broadcast_tensors)
from utils.save import ModelSaver, save_training_meta
from utils.misc import NoOp, parse_with_config
NUM_SPECIAL_TOKENS = 81


def load_img_feat(dir_list, path2imgdir, opts):
    dir_ = dir_list.split(";")
    assert len(dir_) <= 2, "More than two img_dirs found"
    img_dir_gt, img_dir = None, None
    gt_dir_path, dir_path = "", ""
    for d in dir_:
        if "gt" in d:
            gt_dir_path = d
        else:
            dir_path = d
    if gt_dir_path != "":
        img_dir_gt = path2imgdir.get(gt_dir_path, None)
        if img_dir_gt is None:
            img_dir_gt = DetectFeatLmdb(gt_dir_path, -1,
                                        opts.max_bb, opts.min_bb, 100,
                                        opts.compressed_db)
            path2imgdir[gt_dir_path] = img_dir_gt
    if dir_path != "":
        img_dir = path2imgdir.get(dir_path, None)
        if img_dir is None:
            img_dir = DetectFeatLmdb(dir_path, opts.conf_th,
                                     opts.max_bb, opts.min_bb, opts.num_bb,
                                     opts.compressed_db)
            path2imgdir[dir_path] = img_dir
    return img_dir, img_dir_gt, path2imgdir


def main(opts):
    hvd.init()
    n_gpu = hvd.size()
    device = torch.device("cuda", hvd.local_rank())
    torch.cuda.set_device(hvd.local_rank())
    rank = hvd.rank()
    opts.rank = rank
    LOGGER.info("device: {} n_gpu: {}, rank: {}, "
                "16-bits training: {}".format(
                    device, n_gpu, hvd.rank(), opts.fp16))

    if opts.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, "
                         "should be >= 1".format(
                            opts.gradient_accumulation_steps))

    random.seed(opts.seed)
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(opts.seed)

    # train_examples = None
    LOGGER.info(f"Loading Train Dataset {opts.train_txt_db}, "
                f"{opts.train_img_dir}")

    # load DBs and image dirs
    train_txt_dbs = opts.train_txt_db.split(':')
    train_img_dirs = opts.train_img_dir.split(':')
    path2imgdir = {}
    train_datasets = []
    for db, dir_list in zip(train_txt_dbs, train_img_dirs):
        img_dir, img_dir_gt, path2imgdir = load_img_feat(
            dir_list, path2imgdir, opts)
        train_datasets.append(VcrDataset(opts.mask_prob, db, img_dir_gt,
                                         img_dir,
                                         opts.max_txt_len, task="qa"))
        train_datasets.append(VcrDataset(opts.mask_prob, db, img_dir_gt,
                                         img_dir,
                                         opts.max_txt_len, task="qar"))
    train_dataset = ConcatDataset(train_datasets)
    train_lens = [l for dset in train_datasets for l in dset.lens]
    val_img_dir, val_img_dir_gt, path2imgdir = load_img_feat(
            opts.val_img_dir, path2imgdir, opts)
    val_dataset = VcrEvalDataset("val",  opts.val_txt_db,
                                 val_img_dir_gt, val_img_dir,
                                 max_txt_len=-1)
    val_final_dataset = VcrEvalDataset("test",  opts.val_txt_db,
                                       val_img_dir_gt, val_img_dir,
                                       max_txt_len=-1)

    # Prepare model
    train_txt_db = train_txt_dbs[0]
    emb_file = f'{train_txt_db}/embedding.pt'

    if opts.checkpoint and opts.checkpoint_from == "pretrain":
        if opts.checkpoint == 'google-bert':
            checkpoint = None
        else:
            checkpoint = torch.load(opts.checkpoint)
    else:
        checkpoint = {}
    bert_model = json.load(open(f'{train_txt_db}/meta.json'))['bert']
    if 'bert' not in bert_model:
        bert_model = 'bert-large-cased'  # quick hack for glove exp
    model = BertForVisualCommonsenseReasoning.from_pretrained(
        bert_model, img_dim=2048, obj_cls=False,
        state_dict=checkpoint)
    model.init_type_embedding()
    model.init_word_embedding(NUM_SPECIAL_TOKENS)
    if opts.checkpoint_from == "vcr":
        checkpoint = torch.load(opts.checkpoint)
        state_dict = checkpoint.get('model_state', checkpoint)
        matched_state_dict = {}
        unexpected_keys = set()
        missing_keys = set()
        for name, param in model.named_parameters():
            missing_keys.add(name)
        for key, data in state_dict.items():
            if key in missing_keys:
                matched_state_dict[key] = data
                missing_keys.remove(key)
            else:
                unexpected_keys.add(key)
        print("Unexpected_keys:", list(unexpected_keys))
        print("Missing_keys:", list(missing_keys))
        model.load_state_dict(matched_state_dict, strict=False)
    if opts.cut_bert != -1:
        # cut some layers of BERT
        model.bert.encoder.layer = torch.nn.ModuleList(
            model.bert.encoder.layer[:opts.cut_bert])
    if exists(emb_file) and not opts.checkpoint:
        glove = torch.load(f'{train_txt_db}/embedding.pt')
        vsize = glove.size(0)
        hid_size = model.config.hidden_size
        model.bert.embeddings.word_embeddings = torch.nn.Embedding(
            vsize, hid_size)
        mul_ = hid_size // 300 + 1
        model.bert.embeddings.word_embeddings.weight.data = glove.repeat(
            1, mul_)[:, :hid_size]
        LOGGER.info('using GloVe for BERT')
    del checkpoint
    for name, module in model.named_modules():
        # we might want to tune dropout for smaller dataset
        if isinstance(module, torch.nn.Dropout):
            if module.p != opts.dropout:
                module.p = opts.dropout
                LOGGER.info(f'{name} set to {opts.dropout}')
    model.to(device)
    if rank != -1:
        # make sure every process has same model parameters in the beginning
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
    model, optimizer = amp.initialize(model, optimizer,
                                      enabled=opts.fp16, opt_level='O2')

    train_sampler = DistributedTokenBucketSampler(
        n_gpu, rank, train_lens, bucket_size=8192,
        batch_size=opts.train_batch_size, droplast=True)
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=n_gpu, rank=rank)
    val_final_sampler = DistributedSampler(
        val_final_dataset, num_replicas=n_gpu, rank=rank)
    train_dataloader = DataLoader(train_dataset,
                                  batch_sampler=train_sampler,
                                  num_workers=opts.n_workers,
                                  pin_memory=opts.pin_mem,
                                  collate_fn=vcr_collate)
    train_dataloader = PrefetchLoader(train_dataloader)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=opts.val_batch_size*3,
                                sampler=val_sampler,
                                num_workers=opts.n_workers,
                                pin_memory=opts.pin_mem,
                                collate_fn=vcr_eval_collate)
    val_final_dataloader = DataLoader(val_final_dataset,
                                      batch_size=opts.val_batch_size,
                                      sampler=val_final_sampler,
                                      num_workers=opts.n_workers,
                                      pin_memory=opts.pin_mem,
                                      collate_fn=vcr_eval_collate)
    val_dataloader = PrefetchLoader(val_dataloader)
    val_final_dataloader = PrefetchLoader(val_final_dataloader)

    global_step = 0
    if rank == 0:
        save_training_meta(opts)
        TB_LOGGER.create(join(opts.output_dir, 'log'))
        pbar = tqdm(total=opts.num_train_steps)
        model_saver = ModelSaver(join(opts.output_dir, 'ckpt'))
        os.makedirs(join(opts.output_dir, 'results'))  # store VQA predictions
        add_log_to_file(join(opts.output_dir, 'log', 'log.txt'))
    else:
        LOGGER.disabled = True
        pbar = NoOp()
        model_saver = NoOp()

    LOGGER.info(f"***** Running training with {n_gpu} GPUs *****")
    LOGGER.info("  Num examples = %d", len(train_dataset))
    LOGGER.info("  Batch size = %d", opts.train_batch_size)
    LOGGER.info("  Accumulate steps = %d", opts.gradient_accumulation_steps)
    LOGGER.info("  Num steps = %d", opts.num_train_steps)

    running_vcr_loss = RunningMeter('vcr_loss')
    running_obj_loss = RunningMeter('obj_cls_loss')
    running_loss = RunningMeter('loss')
    model.train()
    n_examples = 0
    n_epoch = 0
    start = time()
    # quick hack for amp delay_unscale bug
    optimizer.zero_grad()
    optimizer.step()
    while True:
        for step, batch in enumerate(train_dataloader):
            *_, targets = batch
            n_examples += targets.size(0)

            vcr_loss, obj_cls_loss = model(*batch, compute_loss=True)
            # loss = loss.mean()
            loss = vcr_loss + obj_cls_loss
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
            running_vcr_loss(vcr_loss.item())
            running_obj_loss(obj_cls_loss.item())

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
                elif opts.decay == 'vqa':
                    lr_this_step = opts.learning_rate * vqa_schedule(
                        global_step, opts.warm_int, opts.decay_int,
                        opts.decay_st, opts.decay_rate)
                if lr_this_step < 0:
                    # save guard for possible miscalculation of train steps
                    lr_this_step = 1e-8
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                TB_LOGGER.add_scalar('lr', lr_this_step, global_step)

                # log loss
                losses = all_gather_list(running_loss)
                running_loss = RunningMeter(
                    'loss', sum(l.val for l in losses)/len(losses))
                TB_LOGGER.add_scalar('loss', running_loss.val, global_step)

                vcr_losses = all_gather_list(running_vcr_loss)
                running_vcr_loss = RunningMeter(
                    'vcr_loss', sum(l.val for l in vcr_losses)/len(vcr_losses))
                TB_LOGGER.add_scalar('vcr_loss', running_vcr_loss.val,
                                     global_step)

                obj_losses = all_gather_list(running_obj_loss)
                running_obj_loss = RunningMeter(
                    'obj_cls_loss',
                    sum(l.val for l in obj_losses)/len(obj_losses))
                TB_LOGGER.add_scalar('obj_cls_loss', running_obj_loss.val,
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
                if global_step % opts.valid_steps == 0:
                    val_log, results = validate(
                        model, val_dataloader)
                    TB_LOGGER.log_scaler_dict(val_log)
                    model_saver.save(model, global_step)
            if global_step >= opts.num_train_steps:
                break
        if global_step >= opts.num_train_steps:
            break
        n_epoch += 1
        LOGGER.info(f"finished {n_epoch} epochs")
    val_log, results = validate(
        model, val_final_dataloader)
    with open(f'{opts.output_dir}/results/'
              f'results_{global_step}_'
              f'rank{rank}.json', 'w') as f:
        json.dump(results, f)
    TB_LOGGER.log_scaler_dict(val_log)
    model_saver.save(model, f'{global_step}_final')


def compute_accuracies(out_qa, labels_qa, out_qar, labels_qar):
    outputs_qa = out_qa.max(dim=-1)[1]
    outputs_qar = out_qar.max(dim=-1)[1]
    matched_qa = outputs_qa.squeeze() == labels_qa.squeeze()
    matched_qar = outputs_qar.squeeze() == labels_qar.squeeze()
    matched_joined = matched_qa & matched_qar
    n_correct_qa = matched_qa.sum().item()
    n_correct_qar = matched_qar.sum().item()
    n_correct_joined = matched_joined.sum().item()
    return n_correct_qa, n_correct_qar, n_correct_joined


@torch.no_grad()
def validate(model, val_loader):
    if hvd.rank() == 0:
        val_pbar = tqdm(total=len(val_loader))
    else:
        val_pbar = NoOp()
        LOGGER.info(f"start running evaluation ...")
    model.eval()
    val_qa_loss, val_qar_loss = 0, 0
    tot_qa_score, tot_qar_score, tot_score = 0, 0, 0
    n_ex = 0
    st = time()
    results = {}
    for i, batch in enumerate(val_loader):
        qids, *inputs, qa_targets, qar_targets, _ = batch
        scores = model(
            *inputs, targets=None, compute_loss=False)
        scores = scores.view(len(qids), -1)
        vcr_qa_loss = F.cross_entropy(
                scores[:, :4], qa_targets.squeeze(-1), reduction="sum")
        if scores.shape[1] > 8:
            qar_index = [4+answer_ind.item()*4+i for answer_ind in qa_targets
                         for i in range(4)]
            qar_scores = scores[:, qar_index]
        else:
            qar_scores = scores[:, 4:]
        vcr_qar_loss = F.cross_entropy(
            qar_scores, qar_targets.squeeze(-1), reduction="sum")
        val_qa_loss += vcr_qa_loss.item()
        val_qar_loss += vcr_qar_loss.item()
        curr_qa_score, curr_qar_score, curr_score = compute_accuracies(
            scores[:, :4], qa_targets, qar_scores, qar_targets)
        tot_qar_score += curr_qar_score
        tot_qa_score += curr_qa_score
        tot_score += curr_score
        for qid, score in zip(qids, scores):
            results[qid] = score.cpu().tolist()
        n_ex += len(qids)
        val_pbar.update(1)
    val_qa_loss = sum(all_gather_list(val_qa_loss))
    val_qar_loss = sum(all_gather_list(val_qar_loss))
    tot_qa_score = sum(all_gather_list(tot_qa_score))
    tot_qar_score = sum(all_gather_list(tot_qar_score))
    tot_score = sum(all_gather_list(tot_score))
    n_ex = sum(all_gather_list(n_ex))
    tot_time = time()-st
    val_qa_loss /= n_ex
    val_qar_loss /= n_ex
    val_qa_acc = tot_qa_score / n_ex
    val_qar_acc = tot_qar_score / n_ex
    val_acc = tot_score / n_ex
    val_log = {f'valid/vcr_qa_loss': val_qa_loss,
               f'valid/vcr_qar_loss': val_qar_loss,
               f'valid/acc_qa': val_qa_acc,
               f'valid/acc_qar': val_qar_acc,
               f'valid/acc': val_acc,
               f'valid/ex_per_s': n_ex/tot_time}
    model.train()
    LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                f"score_qa: {val_qa_acc*100:.2f} "
                f"score_qar: {val_qar_acc*100:.2f} "
                f"score: {val_acc*100:.2f} ")
    return val_log, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--task",
                        default="qa", type=str,
                        choices=['qa', 'qar'],
                        help="VCR tasks: qa or qar")
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
    parser.add_argument("--checkpoint_from",
                        default='pretrain', type=str,
                        choices=['pretrain', 'vcr'],
                        help="which setting is checkpoint from")
    parser.add_argument("--cut_bert", default=-1, type=int,
                        help="reduce BERT layers (-1 for original depth)")

    parser.add_argument(
        "--output_dir", default=None, type=str,
        help="The output directory where the model checkpoints will be "
             "written.")

    # Prepro parameters
    parser.add_argument('--max_txt_len', type=int, default=60,
                        help='max number of tokens in text (BERT BPE)')
    parser.add_argument('--conf_th', type=float, default=0.2,
                        help='threshold for dynamic bounding boxes '
                             '(-1 for fixed)')
    parser.add_argument('--max_bb', type=int, default=100,
                        help='max number of bounding boxes')
    parser.add_argument('--min_bb', type=int, default=10,
                        help='min number of bounding boxes')
    parser.add_argument('--num_bb', type=int, default=36,
                        help='static number of bounding boxes')

    # training parameters
    parser.add_argument("--train_batch_size",
                        default=4096, type=int,
                        help="Total batch size for training. "
                             "(batch by tokens)")
    parser.add_argument("--val_batch_size",
                        default=4096, type=int,
                        help="Total batch size for validation. "
                             "(batch by tokens)")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=16,
                        help="Number of updates steps to accumualte before "
                             "performing a backward/update pass.")
    parser.add_argument("--learning_rate",
                        default=3e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--valid_steps",
                        default=1000,
                        type=int,
                        help="Run validation every X steps")
    parser.add_argument("--num_train_steps",
                        default=100000,
                        type=int,
                        help="Total number of training updates to perform.")
    parser.add_argument('--mask_prob', default=0.15, type=float,
                        help='probability to mask in MRC training')
    parser.add_argument("--optim", default='adam',
                        choices=['adam', 'adamax', 'adamw'],
                        help="optimizer")
    parser.add_argument("--betas", default=[0.9, 0.98], nargs='+',
                        help="beta for adam optimizer")
    parser.add_argument("--decay", default='linear',
                        choices=['linear', 'invsqrt', 'constant', 'vqa'],
                        help="learning rate decay method")
    parser.add_argument("--decay_int", default=2000, type=int,
                        help="interval between VQA lr decy")
    parser.add_argument("--warm_int", default=2000, type=int,
                        help="interval for VQA lr warmup")
    parser.add_argument("--decay_st", default=20000, type=int,
                        help="when to start decay")
    parser.add_argument("--decay_rate", default=0.2, type=float,
                        help="ratio of lr decay")
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
                        default=42,
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
    # TODO

    if args.conf_th == -1:
        assert args.max_bb + args.max_txt_len + 2 <= 512
    else:
        assert args.num_bb + args.max_txt_len + 2 <= 512

    main(args)
