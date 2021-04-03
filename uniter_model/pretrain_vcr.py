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
from torch.utils.data import DataLoader
from data.data import ConcatDetectFeatBertTokDataset as ConcatDataset

from apex import amp
from horovod import torch as hvd

import numpy as np
from tqdm import tqdm

from data import (DistributedTokenBucketSampler,
                  DetectFeatLmdb, MlmDatasetForVCR, mlm_collate_for_vcr,
                  MrmDatasetForVCR, mrm_collate_for_vcr,
                  MrcDatasetForVCR, mrc_collate_for_vcr,
                  MetaLoader, PrefetchLoader)
from model import BertForImageTextPretrainingForVCR
from optim import warmup_linear, noam_schedule, vqa_schedule, AdamW

from utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file
from utils.distributed import (all_reduce_and_rescale_tensors, all_gather_list,
                               broadcast_tensors)
from utils.save import ModelSaver, save_training_meta
from utils.misc import NoOp, parse_with_config
NUM_SPECIAL_TOKENS = 81
IMG_DIM = 2048
IMG_LABEL_DIM = 1601


def parse_tasks(datasets):
    task_names = []
    dset_paths = []
    mix_ratio = []
    for i, dset in enumerate(datasets):
        assert len(dset['db']) == len(dset['img'])
        if 'mix_ratio' in dset:
            assert len(dset['tasks']) == len(dset['mix_ratio'])
            mix_ratio.extend(dset['mix_ratio'])
        task_names.extend(f'{t}_{dset["name"]}' for t in dset['tasks'])
        n_task = len(dset['tasks'])
        dset_paths.extend([(dset['db'], dset['img'])] * n_task)

    assert len(task_names) == len(set(task_names)) == len(dset_paths)
    if mix_ratio:
        assert len(task_names) == len(mix_ratio)
        return task_names, dset_paths, mix_ratio
    else:
        return task_names, dset_paths


def build_sampler(lens, batch_size, eval_, bucket_size=8192):
    droplast = not eval_
    sampler = DistributedTokenBucketSampler(
        hvd.size(), hvd.rank(), lens,
        bucket_size=bucket_size, batch_size=batch_size, droplast=droplast)
    return sampler


def build_mlm_train_dataloader(txt_db, img_dir_gt, img_dir,
                               n_gpu, opts):
    LOGGER.info(f"Loading MLM Train Dataset {txt_db}, "
                f"{[i.img_dir for i in img_dir]}"
                f"{[i.img_dir for i in img_dir_gt]}")
    train_datasets = [MlmDatasetForVCR(
                        db, dir_gt_, dir_, opts.max_txt_len, task=t)
                      for db, dir_gt_, dir_ in zip(txt_db, img_dir_gt, img_dir)
                      for t in opts.vcr_task]
    train_dataset = ConcatDataset(train_datasets)
    train_sampler = build_sampler(train_dataset.lens,
                                  opts.train_batch_size, eval_=False)
    train_dataloader = DataLoader(train_dataset,
                                  batch_sampler=train_sampler,
                                  num_workers=opts.n_workers,
                                  pin_memory=opts.pin_mem,
                                  collate_fn=mlm_collate_for_vcr)
    LOGGER.info(f"{len(train_dataset)} samples loaded")
    return train_dataloader


def build_mrm_train_dataloader(txt_db, img_dir_gt, img_dir,
                               n_gpu, opts):
    LOGGER.info(f"Loading MRM Train Dataset {txt_db}, "
                f"{[i.img_dir for i in img_dir]}"
                f"{[i.img_dir for i in img_dir_gt]}")

    train_datasets = [MrmDatasetForVCR(
                        opts.mrm_prob, db, dir_gt_,
                        dir_, opts.max_txt_len, task=t)
                      for db, dir_gt_, dir_ in zip(txt_db, img_dir_gt, img_dir)
                      for t in opts.vcr_task]
    train_dataset = ConcatDataset(train_datasets)
    train_sampler = build_sampler(train_dataset.lens,
                                  opts.train_batch_size, eval_=False)
    train_dataloader = DataLoader(train_dataset,
                                  batch_sampler=train_sampler,
                                  num_workers=opts.n_workers,
                                  pin_memory=opts.pin_mem,
                                  collate_fn=mrm_collate_for_vcr)
    LOGGER.info(f"{len(train_dataset)} samples loaded")
    return train_dataloader


def build_mrc_train_dataloader(txt_db, img_dir_gt, img_dir,
                               n_gpu, opts):
    LOGGER.info(f"Loading MRC Train Dataset {txt_db}, "
                f"{[i.img_dir for i in img_dir]}"
                f"{[i.img_dir for i in img_dir_gt]}")
    train_datasets = [MrcDatasetForVCR(
                        opts.mrc_prob, db, dir_gt_,
                        dir_, opts.max_txt_len, task=t)
                      for db, dir_gt_, dir_ in zip(txt_db, img_dir_gt, img_dir)
                      for t in opts.vcr_task]
    train_dataset = ConcatDataset(train_datasets)
    train_sampler = build_sampler(train_dataset.lens,
                                  opts.train_batch_size, eval_=False)
    train_dataloader = DataLoader(train_dataset,
                                  batch_sampler=train_sampler,
                                  num_workers=opts.n_workers,
                                  pin_memory=opts.pin_mem,
                                  collate_fn=mrc_collate_for_vcr)
    LOGGER.info(f"{len(train_dataset)} samples loaded")
    return train_dataloader


def build_mlm_val_dataloader(txt_db, img_dir_gt, img_dir,
                             n_gpu, opts):
    LOGGER.info(f"Loading MLM Val Dataset {txt_db}, "
                f"{img_dir_gt.img_dir}, {img_dir.img_dir}")
    val_datasets = [MlmDatasetForVCR(
                        txt_db, img_dir_gt, img_dir, -1, task=t)
                    for t in opts.vcr_task]
    val_dataset = ConcatDataset(val_datasets)
    val_sampler = build_sampler(val_dataset.lens,
                                opts.val_batch_size, eval_=True)
    val_dataloader = DataLoader(val_dataset,
                                batch_sampler=val_sampler,
                                num_workers=opts.n_workers,
                                pin_memory=opts.pin_mem,
                                collate_fn=mlm_collate_for_vcr)
    LOGGER.info(f"{len(val_dataset)} samples loaded")
    return val_dataloader


def build_mrm_val_dataloader(txt_db, img_dir_gt, img_dir,
                             n_gpu, opts):
    LOGGER.info(f"Loading MRM Val Dataset {txt_db}, "
                f"{img_dir_gt.img_dir}, {img_dir.img_dir}")
    val_datasets = [MrmDatasetForVCR(
                        opts.mrm_prob, txt_db, img_dir_gt,
                        img_dir, -1, task=t)
                    for t in opts.vcr_task]
    val_dataset = ConcatDataset(val_datasets)
    val_sampler = build_sampler(val_dataset.lens,
                                opts.val_batch_size, eval_=True)
    val_dataloader = DataLoader(val_dataset,
                                batch_sampler=val_sampler,
                                num_workers=opts.n_workers,
                                pin_memory=opts.pin_mem,
                                collate_fn=mrm_collate_for_vcr)
    LOGGER.info(f"{len(val_dataset)} samples loaded")
    return val_dataloader


def build_mrc_val_dataloader(txt_db, img_dir_gt, img_dir,
                             n_gpu, opts):
    LOGGER.info(f"Loading MRC Val Dataset {txt_db}, "
                f"{img_dir_gt.img_dir}, {img_dir.img_dir}")
    val_datasets = [MrcDatasetForVCR(
                        opts.mrc_prob, txt_db, img_dir_gt,
                        img_dir, -1, task=t)
                    for t in opts.vcr_task]
    val_dataset = ConcatDataset(val_datasets)
    val_sampler = build_sampler(val_dataset.lens,
                                opts.val_batch_size, eval_=True)
    val_dataloader = DataLoader(val_dataset,
                                batch_sampler=val_sampler,
                                num_workers=opts.n_workers,
                                pin_memory=opts.pin_mem,
                                collate_fn=mrc_collate_for_vcr)
    LOGGER.info(f"{len(val_dataset)} samples loaded")
    return val_dataloader


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

    if rank == 0:
        save_training_meta(opts)
        TB_LOGGER.create(join(opts.output_dir, 'log'))
        pbar = tqdm(total=opts.num_train_steps)
        model_saver = ModelSaver(join(args.output_dir, 'ckpt'))
        add_log_to_file(join(opts.output_dir, 'log', 'log.txt'))
    else:
        LOGGER.disabled = True
        pbar = NoOp()
        model_saver = NoOp()

    all_dbs = [db for datasets in [opts.train_datasets, opts.val_datasets]
               for dset in datasets for db in dset['db']]
    bert_model = json.load(open(f'{all_dbs[0]}/meta.json'))['bert']
    assert all(bert_model == json.load(open(f'{db}/meta.json'))['bert']
               for db in all_dbs)

    train_tasks, train_data_paths, mix_ratio = parse_tasks(opts.train_datasets)
    train_dataloaders = []
    path2imgdir = {}
    for (dbs, dirs), task in zip(train_data_paths, train_tasks):
        img_dirs = []
        img_gt_dirs = []
        for db, dir_list in zip(dbs, dirs):
            img_dir, img_dir_gt, path2imgdir = load_img_feat(
                dir_list, path2imgdir, opts)
            img_dirs.append(img_dir)
            img_gt_dirs.append(img_dir_gt)
        if task.startswith('mlm'):
            loader = build_mlm_train_dataloader(dbs, img_gt_dirs, img_dirs,
                                                n_gpu, opts)
        elif task.startswith('mrm'):
            loader = build_mrm_train_dataloader(dbs, img_gt_dirs, img_dirs,
                                                n_gpu, opts)
        elif task.startswith('mrc'):
            loader = build_mrc_train_dataloader(dbs, img_gt_dirs, img_dirs,
                                                n_gpu, opts)
        else:
            raise ValueError(f'Undefined task {task}')
        train_dataloaders.append(loader)
    val_tasks, val_data_paths = parse_tasks(opts.val_datasets)
    val_dataloaders = []
    for (db, dir_), task in zip(val_data_paths, val_tasks):
        assert len(db) == len(dir_) == 1
        db = db[0]
        dir_ = dir_[0]
        img_dir, img_dir_gt, path2imgdir = load_img_feat(
            dir_, path2imgdir, opts)
        if task.startswith('mlm'):
            loader = build_mlm_val_dataloader(db, img_dir_gt, img_dir, n_gpu, opts)
        elif task.startswith('mrm'):
            loader = build_mrm_val_dataloader(db, img_dir_gt, img_dir, n_gpu, opts)
        elif task.startswith('mrc'):
            loader = build_mrc_val_dataloader(db, img_dir_gt, img_dir, n_gpu, opts)
        else:
            raise ValueError(f'Undefined task {task}')
        val_dataloaders.append(PrefetchLoader(loader))
    meta_loader = MetaLoader(train_dataloaders,
                             mix_ratio=mix_ratio, names=train_tasks,
                             accum_steps=opts.gradient_accumulation_steps,
                             distributed=n_gpu > 1)
    meta_loader = PrefetchLoader(meta_loader)
    named_val_loaders = list(zip(val_tasks, val_dataloaders))

    # Prepare model

    if opts.checkpoint:
        if opts.checkpoint == 'google-bert':
            checkpoint = None
        else:
            checkpoint = torch.load(opts.checkpoint)
    else:
        checkpoint = {}
    model = BertForImageTextPretrainingForVCR.from_pretrained(
        bert_model, img_dim=IMG_DIM, img_label_dim=IMG_LABEL_DIM,
        state_dict=checkpoint)
    model.init_type_embedding()
    model.init_word_embedding(NUM_SPECIAL_TOKENS)
    model.pad_vocab()  # tensor core padding for vocabulary
    if opts.cut_bert != -1:
        # cut some layers of BERT
        model.bert.encoder.layer = torch.nn.ModuleList(
            model.bert.encoder.layer[:opts.cut_bert])

    for name, module in model.named_modules():
        # we might want to tune dropout for smaller dataset
        if isinstance(module, torch.nn.Dropout):
            if module.p != opts.dropout:
                module.p = opts.dropout
                LOGGER.info(f'{name} set to {opts.dropout}')
    model.to(device)
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
    LOGGER.info("  Batch size = %d", opts.train_batch_size)
    LOGGER.info("  Accumulate steps = %d", opts.gradient_accumulation_steps)
    LOGGER.info("  Num steps = %d", opts.num_train_steps)

    task2loss = {task: RunningMeter(f'loss/{task}') for task in train_tasks}
    model.train()
    n_examples = 0
    n_epoch = 0
    start = time()
    # quick hack for amp delay_unscale bug
    optimizer.zero_grad()
    optimizer.step()
    while True:
        for step, (name, batch) in enumerate(meta_loader):
            input_ids, *_ = batch
            n_examples += input_ids.size(0)
            task = name.split('_')[0]
            loss = model(*batch, task=task, compute_loss=True)
            loss = loss.mean()  # loss is not normalized
            if task == 'mrckl':
                # MRCkl normalization; safeguard fp16 overflow
                loss = loss.float() * IMG_LABEL_DIM
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

            task2loss[name](loss.item())

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
                for t, l in task2loss.items():
                    loss = sum(v for v in all_gather_list(l.val)
                               if v is not None) / hvd.size()
                    task2loss[t] = RunningMeter(f'loss/{t}', loss)
                TB_LOGGER.log_scaler_dict({l.name: l.val
                                           for l in task2loss.values()
                                           if l.val is not None})
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
                    validate(model, named_val_loaders)
                    model_saver.save(model, global_step)
            if global_step >= opts.num_train_steps:
                break
        if global_step % opts.valid_steps != 0:
            validate(model, named_val_loaders)
            model_saver.save(model, global_step)


def validate(model, named_val_loaders):
    model.eval()
    for task, loader in named_val_loaders:
        LOGGER.info(f"validate on {task} task")
        if task.startswith('mlm'):
            val_log = validate_mlm(model, loader)
        elif task.startswith('mrm'):
            val_log = validate_mrm(model, loader)
        elif task.startswith('mrc'):
            val_log = validate_mrc(model, loader, task)
        else:
            raise ValueError(f'Undefined task {task}')
        val_log = {f'{task}_{k}': v for k, v in val_log.items()}
        TB_LOGGER.log_scaler_dict(
            {f'valid_{task}/{k}': v for k, v in val_log.items()})
    model.train()


@torch.no_grad()
def validate_mrc(model, val_loader, task):
    LOGGER.info("start running MRC validation...")
    val_loss = 0
    n_feat = 0
    st = time()
    tot_score = 0
    for i, batch in enumerate(val_loader):
        *_, label = batch
        feat_mask, label_targets = label
        prediction_soft_label = model(
            *batch, task=task, compute_loss=False)
        if "kl" in task:
            prediction_soft_label = F.log_softmax(
                prediction_soft_label, dim=-1)
            loss = F.kl_div(
                prediction_soft_label, label_targets, reduction='sum')
            tot_score += compute_accuracy_for_mrc(
                prediction_soft_label, label_targets)
        else:
            cls_label_targets = label_targets.max(dim=-1)[1]  # argmax
            loss = F.cross_entropy(
                prediction_soft_label, cls_label_targets,
                ignore_index=0, reduction='sum')
            tot_score += compute_accuracy_for_mrc(
                prediction_soft_label[:, 1:], label_targets[:, 1:])
        val_loss += loss.item()
        n_feat += feat_mask.sum().item()
    val_loss = sum(all_gather_list(val_loss))
    tot_score = sum(all_gather_list(tot_score))
    n_feat = sum(all_gather_list(n_feat))
    tot_time = time()-st
    val_loss /= n_feat
    val_acc = tot_score / n_feat
    val_log = {'loss': val_loss,
               'acc': val_acc,
               'feat_per_s': n_feat/tot_time}
    LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                f"score: {val_acc*100:.2f}")
    return val_log


@torch.no_grad()
def validate_mrm(model, val_loader):
    LOGGER.info("start running MRM validation...")
    val_loss = 0
    n_feat = 0
    st = time()
    for i, batch in enumerate(val_loader):
        *_, feat_mask = batch
        loss = model(*batch, task='mrm', compute_loss=True)
        val_loss += loss.sum().item()
        n_feat += feat_mask.sum().item()
    val_loss = sum(all_gather_list(val_loss))
    n_feat = sum(all_gather_list(n_feat))
    tot_time = time()-st
    val_loss /= (n_feat * IMG_DIM)
    val_log = {'loss': val_loss,
               'feat_per_s': n_feat/tot_time}
    LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                f"loss: {val_loss:.2f}")
    return val_log


@torch.no_grad()
def validate_mlm(model, val_loader):
    LOGGER.info(f"start running MLM validation ...")
    val_loss = 0
    n_correct = 0
    n_word = 0
    st = time()
    for i, batch in enumerate(val_loader):
        *inputs, txt_labels = batch
        loss = model.forward(*batch, task='mlm', compute_loss=True)
        # loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1,
        #                                      reduction='sum')
        # loss = loss_fct(scores, txt_labels)
        loss = loss.mean()
        val_loss += loss.item()
        # n_correct += accuracy_count(scores, txt_labels)
        n_word += txt_labels.numel()
    val_loss = sum(all_gather_list(val_loss))
    n_correct = sum(all_gather_list(n_correct))
    n_word = sum(all_gather_list(n_word))
    tot_time = time()-st
    val_loss /= n_word
    acc = n_correct / n_word
    val_log = {'loss': val_loss,
               'acc': acc,
               'tok_per_s': n_word/tot_time}
    LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                f"acc: {acc*100:.2f}"
                f"loss: {val_loss}")
    return val_log


def compute_accuracy_for_mrc(out, labels):
    outputs = out.max(dim=-1)[1]
    labels = labels.max(dim=-1)[1]  # argmax
    n_correct = (outputs == labels).sum().item()
    return n_correct


def accuracy_count(out, labels):
    outputs = out.max(dim=-1)[1]
    mask = labels != -1
    n_correct = (outputs == labels).masked_select(mask).sum().item()
    return n_correct


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    # NOTE: train tasks and val tasks cannot take command line arguments
    parser.add_argument('--compressed_db', action='store_true',
                        help='use compressed LMDB')
    parser.add_argument("--vcr_task",
                        default=["qar"], type=str, nargs='+',
                        choices=['qa', 'qar'],
                        help="VCR tasks: qa or qar")
    parser.add_argument('--tasks', default=None, type=str, nargs='+',
                        help="specify pretraining tasks")
    parser.add_argument('--mrm_prob', default=0.15, type=float,
                        help='probability to mask in MRM training')
    parser.add_argument('--mrc_prob', default=0.15, type=float,
                        help='probability to mask in MRC training')
    parser.add_argument("--checkpoint",
                        default=None, type=str,
                        help="pretrained model (can take 'google-bert') ")
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
    assert len(args.vcr_task) > 0, "Must choose at least one vcr task"
    main(args)
