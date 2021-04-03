"""UNITER pre-training runner."""
import warnings
warnings.filterwarnings("ignore")

import argparse
import glob
from collections import defaultdict
import json
import math
import os
from os.path import exists, join
from time import time

import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

from apex import amp
from horovod import torch as hvd

from tqdm import tqdm

from dvl.utils import is_main_process, print_args
from dvl.models.bi_encoder import BiEncoderForPretraining, load_biencoder_checkpoint
from dvl.data.mlm import mlm_collate, mlm_blind_collate, MlmDataset, BlindMlmDataset
from dvl.data.mrm import mrc_collate, MrcDataset, MrfrDataset, mrfr_collate
from dvl.data.itm_pre import ItmDataset, itm_collate, itm_ot_collate

from uniter_model.data import (TokenBucketSampler, TokenBucketSamplerForItm,
                  MetaLoader, PrefetchLoader,
                  TxtTokLmdb, ImageLmdbGroup, ConcatDatasetWithLens)
from uniter_model.data.mrm_nce import NegativeImageSampler, MrmNceDataset, mrm_nce_collate

from uniter_model.model import UniterForPretraining
from uniter_model.optim import get_lr_sched
from uniter_model.optim.misc import build_optimizer

from uniter_model.utils.logger import LOGGER, RunningMeter, add_log_to_file
from uniter_model.utils.distributed import (all_reduce_and_rescale_tensors, all_gather_list,
                               broadcast_tensors)
from uniter_model.utils.save import ModelSaver, save_training_meta
from uniter_model.utils.misc import NoOp, parse_with_config, set_dropout, set_random_seed
from uniter_model.utils.const import IMG_DIM, IMG_LABEL_DIM, BUCKET_SIZE



WARM_STEP = 500


def build_dataloader(dataset, collate_fn, is_train, opts):
    if is_train:
        batch_size = opts.train_batch_size
    else:
        batch_size = opts.val_batch_size
    sampler = TokenBucketSampler(dataset.lens, bucket_size=BUCKET_SIZE,
                                 batch_size=batch_size, droplast=is_train)
    loader = DataLoader(dataset, batch_sampler=sampler,
                        num_workers=opts.n_workers, pin_memory=opts.pin_mem,
                        collate_fn=collate_fn)
    return loader


def build_dataloader_itm(dataset, collate_fn, is_train, opts):
    if is_train:
        batch_size = opts.train_batch_size
    else:
        batch_size = opts.val_batch_size
    sampler = TokenBucketSamplerForItm(
        dataset, bucket_size=BUCKET_SIZE,
        batch_size=batch_size, droplast=is_train)
    loader = DataLoader(dataset, batch_sampler=sampler,
                        num_workers=opts.n_workers, pin_memory=opts.pin_mem,
                        collate_fn=collate_fn)
    return loader


def build_mlm_dataset(txt_db, img_db, blind, is_train, opts):
    if is_train:
        if blind:
            collate_fn = mlm_blind_collate
            datasets = [BlindMlmDataset(t) for t in txt_db]
        else:
            collate_fn = mlm_collate
            datasets = [MlmDataset(t, i) for t, i in zip(txt_db, img_db)]
        dataset = ConcatDatasetWithLens(datasets)
    else:
        if blind:
            collate_fn = mlm_blind_collate
            dataset = BlindMlmDataset(txt_db)
        else:
            collate_fn = mlm_collate
            dataset = MlmDataset(txt_db, img_db)

    return dataset, collate_fn


def build_mrfr_dataset(txt_db, img_db, only_i, is_train, opts):
    collate_fn = (mrfr_only_img_collate if only_i
                  else mrfr_collate)
    if is_train:
        if only_i:
            datasets = [OnlyImgMrfrDataset(opts.mrm_prob, i) for i in img_db]
        else:
            datasets = [MrfrDataset(opts.mrm_prob, t, i)
                        for t, i in zip(txt_db, img_db)]
        dataset = ConcatDatasetWithLens(datasets)
    else:
        if only_i:
            dataset = OnlyImgMrfrDataset(opts.mrm_prob, img_db)
        else:
            dataset = MrfrDataset(opts.mrm_prob, txt_db, img_db)

    return dataset, collate_fn


def build_mrm_nce_dataset(txt_db, img_db, only_i, is_train, opts):
    assert not only_i
    neg_sampler = NegativeImageSampler(img_db, opts.neg_size)
    collate_fn = mrm_nce_collate(neg_sampler)
    if is_train:
        datasets = [MrmNceDataset(opts.mrm_prob, t, i)
                    for t, i in zip(txt_db, img_db)]
        dataset = ConcatDatasetWithLens(datasets)
    else:
        dataset = MrmNceDataset(opts.mrm_prob, txt_db, img_db)

    return dataset, collate_fn


def build_mrc_dataset(txt_db, img_db, only_i, is_train, opts):
    collate_fn = (mrc_only_img_collate if only_i
                  else mrc_collate)
    if is_train:
        if only_i:
            datasets = [OnlyImgMrcDataset(opts.mrm_prob, i) for i in img_db]
        else:
            datasets = [MrcDataset(opts.mrm_prob, t, i)
                        for t, i in zip(txt_db, img_db)]
        dataset = ConcatDatasetWithLens(datasets)
    else:
        if only_i:
            dataset = OnlyImgMrcDataset(opts.mrm_prob, img_db)
        else:
            dataset = MrcDataset(opts.mrm_prob, txt_db, img_db)

    return dataset, collate_fn


def build_itm_dataset(txt_db, img_db, is_train, opts):
    if is_train:
        datasets = [ItmDataset(t, i, opts.itm_neg_prob)
                    for t, i in zip(txt_db, img_db)]
        dataset = ConcatDatasetWithLens(datasets)
    else:
        dataset = ItmDataset(txt_db, img_db, opts.itm_neg_prob)
    collate_fn = itm_ot_collate if opts.itm_ot_lambda > 0 else itm_collate
    return dataset, collate_fn


def create_dataloaders(datasets, is_train, opts, all_img_dbs=None):
    if all_img_dbs is None:
        all_img_dbs = ImageLmdbGroup(opts.conf_th, opts.max_bb, opts.min_bb,
                                     opts.num_bb, opts.compressed_db)
    dataloaders = {}
    for dset in datasets:
        if is_train:
            assert len(dset['db']) == len(dset['img'])
            assert len(dset['tasks']) == len(dset['mix_ratio'])
            img_db = [all_img_dbs[path] for path in dset['img']]
        else:
            assert len(dset['db']) == len(dset['img']) == 1
            img_db = all_img_dbs[dset['img'][0]]

        for i, t in enumerate(dset['tasks']):
            task = f'{t}_{dset["name"]}'

            if is_train:
                LOGGER.info(f"Loading {task} train dataset "
                            f"{dset['db']}, {[img.img_dir for img in img_db]}")
                txt_db = [TxtTokLmdb(path, opts.max_txt_len)
                          for path in dset['db']]
            else:
                LOGGER.info(f"Loading {task} validation dataset, "
                            f"{dset['db']}, {img_db.img_dir}")
                txt_db = TxtTokLmdb(dset['db'][0], -1)

            if task.startswith('mlm'):
                blind = 'blind' in task
                dataset = build_mlm_dataset(txt_db, img_db,
                                            blind, is_train, opts)
            elif task.startswith('mrfr'):
                only_i = 'only_i' in task
                dataset = build_mrfr_dataset(txt_db, img_db,
                                             only_i, is_train, opts)
            elif task.startswith('mrm-nce'):
                only_i = 'only_i' in task
                dataset = build_mrm_nce_dataset(txt_db, img_db,
                                                only_i, is_train, opts)
            elif task.startswith('mrc'):
                only_i = 'only_i' in task
                dataset = build_mrc_dataset(txt_db, img_db,
                                            only_i, is_train, opts)
            elif task.startswith('itm'):
                dataset = build_itm_dataset(txt_db, img_db, is_train, opts)
            else:
                raise ValueError(f'Undefined task {task}')

            LOGGER.info(f"{len(dataset[0])*hvd.size()} samples loaded")
            if task.startswith('itm'):
                # itm handles distributed training in dset not sampler
                loader = build_dataloader_itm(*dataset, is_train, opts)
            else:
                loader = build_dataloader(*dataset, is_train, opts)
            if is_train:
                ratio = dset['mix_ratio'][i]
                dataloaders[task] = (loader, ratio)
            else:
                dataloaders[task] = PrefetchLoader(loader)
    return dataloaders, all_img_dbs


def batch_2_teacher(batch):
    from uniter_model.data.loader import move_to_cuda
    teacher_batch = dict()

    # copy label and extra information from student to teacher
    for k in batch:
        if k in ['txts', 'imgs', 'teacher']:
            continue
        teacher_batch[k] = move_to_cuda(batch[k])
    # update 'attn_masks'
    teacher_batch['input_ids'] = move_to_cuda(batch['txts']['input_ids'])
    teacher_batch['position_ids'] = move_to_cuda(batch['txts']['position_ids'])
    teacher_batch['img_feat'] = move_to_cuda(batch['imgs']['img_feat'])
    teacher_batch['img_pos_feat'] = move_to_cuda(batch['imgs']['img_pos_feat'])
    teacher_batch['img_masks'] = move_to_cuda(batch['imgs']['img_masks'])
    teacher_batch['gather_index'] = move_to_cuda(batch['teacher']['gather_index'])
    teacher_batch['attn_masks'] = move_to_cuda(batch['teacher']['attn_masks'])
    if 'img_mask_tgt' in batch['teacher']:
        teacher_batch['img_mask_tgt'] = batch['teacher']['img_mask_tgt']
    return teacher_batch


def main(args):
    hvd.init()
    n_gpu = hvd.size()
    device = torch.device("cuda", hvd.local_rank())
    torch.cuda.set_device(hvd.local_rank())
    rank = hvd.rank()
    args.rank = rank
    LOGGER.info("device: {} n_gpu: {}, rank: {}, "
                "16-bits training: {}".format(
                    device, n_gpu, hvd.rank(), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, "
                         "should be >= 1".format(
                            args.gradient_accumulation_steps))

    if is_main_process():
        """
        # if you want to use comet, please modify code below
        from comet_ml import Experiment
        experiment = Experiment(api_key='your api key', workspace='your workspace name', project_name=args.project_name)
        experiment.log_parameters({
            'train_batch_size': args.train_batch_size,
            'learning_rate': args.learning_rate,
            'gradient_accumulation_steps': args.gradient_accumulation_steps,
            'num_train_steps': args.num_train_steps,
            'warmup_steps': args.warmup_steps,
            'seed': args.seed,
            'output_dir': args.output_dir,
        })
        experiment.set_name(f'{args.train_batch_size}-{args.learning_rate}-{args.gradient_accumulation_steps}-'
                            f'{args.num_train_steps}')
        """
        experiment = None
        print_args(args)
    else:
        experiment = None
    set_random_seed(args.seed)

    if rank == 0:
        save_training_meta(args)
        pbar = tqdm(total=args.num_train_steps)
        model_saver = ModelSaver(join(args.output_dir, 'ckpt'))
        add_log_to_file(join(args.output_dir, 'log', 'log.txt'))
    else:
        LOGGER.disabled = True
        pbar = NoOp()
        model_saver = NoOp()

    all_dbs = [db for datasets in [args.train_datasets, args.val_datasets]
               for dset in datasets for db in dset['db']]

    tokenizer = json.load(open(f'{all_dbs[0]}/meta.json'))['bert']
    assert all(tokenizer == json.load(open(f'{db}/meta.json'))['bert']
               for db in all_dbs)

    # if not is_main_process():
        # make sure only the first process is downloading the model data
   #     hvd.allreduce(torch.tensor(0), name='barrier')

    # prepare model
    if False:
        model = UniterForPretraining.from_pretrained(
            args.model_config, checkpoint,
            img_dim=IMG_DIM, img_label_dim=IMG_LABEL_DIM,
            nce_temp=args.nce_temp, ot_pos_only=args.ot_pos_only)
    else:
        model = BiEncoderForPretraining(args.model_config, args, args.project_dim, IMG_DIM, IMG_LABEL_DIM,
                                        experiment=experiment)

    # if is_main_process():
    #    hvd.allreduce(torch.tensor(0), name='barrier')

    # model.pad_vocab()  # tensor core padding for vocabulary ???
    if args.train_state is not None:
        model.load_state_dict(torch.load(args.train_state['model_file'], map_location='cpu'), strict=True)
        global_step = args.train_state['step']
        pbar.update(global_step)
        LOGGER.info('loading model and optimizer from ' + args.train_state['model_file'])
        LOGGER.info(f'new global_step = {global_step}')
    else:
        LOGGER.info('no model checkpont provide, global step = 0')
        global_step = 0

    if args.biencoder_checkpoint is not None:
        model.load_state_dict(torch.load(args.biencoder_checkpoint, map_location='cpu'), strict=True)
    model.to(device)
    model.train()

    if args.teacher_checkpoint is not None and len(args.teacher_checkpoint) > 0:
        teacher_model = UniterForPretraining.from_pretrained(
                        args.teacher_config, torch.load(args.teacher_checkpoint, map_location='cpu'),
                        img_dim=IMG_DIM, img_label_dim=IMG_LABEL_DIM,
                        nce_temp=args.nce_temp, ot_pos_only=args.ot_pos_only)
        teacher_model.to(device)
        teacher_model.eval()
    else:
        teacher_model = None

    # make sure every process has same model parameters in the beginning
    broadcast_tensors([p.data for p in model.parameters()], 0)
    set_dropout(model, args.dropout)

    # Prepare optimizer
    optimizer = build_optimizer(model, args)
    if args.train_state is not None:
        optimizer.load_state_dict(torch.load(args.train_state['state_file'], map_location='cpu')['optimizer'])
    else:
        # quick hack for amp delay_unscale bug
        optimizer.zero_grad()
        optimizer.step()

    # build data loaders
    train_dataloaders, all_img_dbs = create_dataloaders(
        args.train_datasets, True, args)
    val_dataloaders, _ = create_dataloaders(
        args.val_datasets, False, args, all_img_dbs)
    meta_loader = MetaLoader(train_dataloaders,
                             accum_steps=args.gradient_accumulation_steps,
                             distributed=n_gpu > 1)
    meta_loader = PrefetchLoader(meta_loader)

    task2scaler = {t: i for i, t in enumerate(train_dataloaders.keys())}
    model, optimizer = amp.initialize(model, optimizer,
                                      num_losses=len(task2scaler),
                                      enabled=args.fp16, opt_level='O2')

    LOGGER.info(f"***** Running training with {n_gpu} GPUs *****")
    LOGGER.info("  Batch size = %d", args.train_batch_size)
    LOGGER.info("  Accumulate steps = %d", args.gradient_accumulation_steps)
    LOGGER.info("  Num steps = %d", args.num_train_steps)

    # to compute training statistics
    task2loss = {task: RunningMeter(f'loss/{task}')
                 for task in train_dataloaders.keys()}
    # ITM w/ OT
    n_examples = defaultdict(int)
    n_in_units = defaultdict(int)
    n_loss_units = defaultdict(int)
    n_neg_nce = defaultdict(int)
    grad_norm = 0

    start = time()
    # loss_record = defaultdict(list)
    for step, (name, batch) in enumerate(meta_loader):
        # forward pass
        assert all(name == n for n in all_gather_list(name))
        n_examples[name] += batch['txts']['input_ids'].size(0)
        n_in_units[name] += (batch['txts']['attention_mask'] == 1).sum().item()

        if 'nce' in name:
            n_neg_nce[name] += batch['neg_feats'].size(0)
        task = name.split('_')[0]
        loss_task, logits_student = model(batch, task=task, compute_loss=True)
        if task.startswith('itm'):
            itm_loss = loss_task
            n_loss_units[name] += itm_loss.size(0)     # ???
            itm_loss = itm_loss.mean()
            loss_task = itm_loss
            loss = loss_task
        else:
            n_loss_units[name] += loss_task.size(0)
            loss_task = loss_task.mean()  # loss is not normalized in model
            if teacher_model is not None:
                batch_teacher = batch_2_teacher(batch)

                with torch.no_grad():
                    loss_teacher, logits_teacher = teacher_model(batch_teacher, task, compute_loss=True)
                    # "bs" * num_class

                # mrfr --> feature regression
                # others --> logits
                if 'mrfr' in task:
                    loss_kd = args.kd_loss_weight * F.mse_loss(logits_teacher/args.T, logits_student/args.T)
                else:
                    loss_kd = nn.KLDivLoss()(
                        F.log_softmax(logits_student / args.T, dim=1),
                        F.softmax(logits_teacher / args.T, dim=1)) * (args.kd_loss_weight * args.T * args.T)

                # loss_record[f'{task}.teacher'].append(loss_teacher.mean().item())
                # loss_record[f'{task}.student'].append(loss_task.item())
                # loss_record[f'{task}.kd'].append(loss_kd.item())
                loss = loss_task + loss_kd.mean()
            else:
                loss = loss_task

        if experiment is not None and is_main_process():
            experiment.log_metric(name, loss.item())
            experiment.log_metric(name.split('_')[0], loss.item())
            try:
                experiment.log_metric(name + '.kd', loss_kd.item())
            except NameError:
                pass

        # backward pass
        delay_unscale = (step+1) % args.gradient_accumulation_steps != 0
        with amp.scale_loss(loss, optimizer, delay_unscale=delay_unscale,
                            loss_id=task2scaler[name]) as scaled_loss:
            scaled_loss.backward()
            if not delay_unscale:
                # gather gradients from every processes
                # do this before unscaling to make sure every process uses
                # the same gradient scale
                grads = [p.grad.data for p in model.parameters()
                         if p.requires_grad and p.grad is not None]
                all_reduce_and_rescale_tensors(grads, float(1))
        task2loss[name](loss.item())

        # optimizer update and logging
        if (step + 1) % args.gradient_accumulation_steps == 0:
            if experiment is not None and is_main_process():
                experiment.log_metric('global_step', global_step)
            global_step += 1

            # learning rate scheduling
            lr_this_step = get_lr_sched(global_step, args)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_step
            if experiment is not None and is_main_process():
                experiment.log_metric('lr', lr_this_step)

            # log loss
            for t, l in task2loss.items():
                loss = sum(v for v in all_gather_list(l.val)
                           if v is not None) / hvd.size()
                task2loss[t] = RunningMeter(f'loss/{t}', loss)
            if experiment is not None and is_main_process():
                experiment.log_metrics({l.name: l.val for l in task2loss.values() if l.val is not None})

            # update model params
            if args.grad_norm != -1:
                '''
                if global_step % 10 == 0 and not args.fp16:
                    bias = model.bert.img_embeddings.img_linear.bias
                    weight = model.bert.img_embeddings.img_linear.weight
                    print(f"bnorm: {bias.norm()}")
                    print(f"wnorm: {weight.norm()}")
                    print(f"bgnorm: {bias.grad.norm()}")
                    print(f"wgnorm: {weight.grad.norm()}")

                    mask = model.bert.img_embeddings.mask_embedding.weight
                    print(f"mnorm: {mask.norm()}")
                    print(f"mgnorm: {mask.grad.norm()}")

                    print([(n, p.grad.norm().item())
                           for n, p in model.named_parameters()
                           if p.grad is not None
                              and p.grad.norm().item() > grad_norm/10])
                '''
                grad_norm = clip_grad_norm_(amp.master_params(optimizer),
                                            args.grad_norm)
                if experiment is not None and is_main_process():
                    experiment.log_metric('grad_norm', grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            pbar.update(1)

            if global_step % 100 == 0:
                # monitor training throughput
                LOGGER.info(f'==============Step {global_step}===============')
                for t in train_dataloaders.keys():
                    assert all(tt == t for tt in all_gather_list(t))
                    tot_ex = sum(all_gather_list(n_examples[t]))
                    ex_per_sec = int(tot_ex / (time()-start))
                    tot_in = sum(all_gather_list(n_in_units[t]))
                    in_per_sec = int(tot_in / (time()-start))
                    tot_l = sum(all_gather_list(n_loss_units[t]))
                    l_per_sec = int(tot_l / (time()-start))
                    LOGGER.info(f'{t}: {tot_ex} examples trained at '
                                f'{ex_per_sec} ex/s')
                    if experiment is not None and is_main_process():
                        experiment.log_metric(f'perf/{t}_ex_per_s', ex_per_sec)
                        experiment.log_metric(f'perf/{t}_in_per_s', in_per_sec)
                        experiment.log_metric(f'perf/{t}_loss_per_s', l_per_sec)
                    if 'nce' in t:
                        avg_neg = sum(all_gather_list(n_neg_nce[t])
                                      ) / hvd.size() // step
                        LOGGER.info(f'{t}: averaging '
                                    f'{avg_neg} negative samples')
                LOGGER.info(f'===============================================')

            if global_step % args.valid_steps == 0:
                LOGGER.info(f'Step {global_step}: start validation')
                validate(model, val_dataloaders, experiment)
                model_saver.save(model, global_step, optimizer)
        if global_step >= args.num_train_steps:
            break
    if global_step % args.valid_steps != 0:
        LOGGER.info(f'Step {global_step}: start validation')
        model_saver.save(model, global_step)
        validate(model, val_dataloaders, experiment)


def validate(model, val_dataloaders, experiment=None):
    model.eval()
    for task, loader in val_dataloaders.items():
        LOGGER.info(f"validate on {task} task")
        if task.startswith('mlm'):
            val_log = validate_mlm(model, loader)
        elif task.startswith('mrfr'):
            val_log = validate_mrfr(model, loader)
        elif task.startswith('mrm-nce'):
            val_log = validate_mrm_nce(model, loader)
        elif task.startswith('mrc'):
            val_log = validate_mrc(model, loader, task)
        elif task.startswith('itm'):
            val_log = validate_itm(model, loader)
        else:
            raise ValueError(f'Undefined task {task}')
        val_log = {f'{task}_{k}': v for k, v in val_log.items()}
        if experiment is not None and is_main_process():
            experiment.log_metrics({f'valid_{task}/{k}': v for k, v in val_log.items()})
    model.train()


@torch.no_grad()
def validate_mlm(model, val_loader):
    LOGGER.info("start running MLM validation...")
    val_loss = 0
    n_correct = 0
    n_word = 0
    st = time()
    for i, batch in enumerate(val_loader):
        _, scores = model(batch, task='mlm', compute_loss=False)
        labels = batch['txt_labels']
        labels = labels[labels != -1]
        loss = F.cross_entropy(scores, labels, reduction='sum')
        val_loss += loss.item()
        n_correct += (scores.max(dim=-1)[1] == labels).sum().item()
        n_word += labels.numel()
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
                f"acc: {acc*100:.2f}")
    return val_log


@torch.no_grad()
def validate_mlm_old(model, val_loader):
    LOGGER.info("start running MLM validation...")
    val_loss = 0
    n_correct = 0
    n_word = 0
    st = time()
    for i, batch in enumerate(val_loader):
        scores = model.forward(batch, task='mlm', compute_loss=False)
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1,
                                             reduction='sum')
        scores = scores.contiguous().view(-1, model.config.vocab_size)
        labels = batch['txt_labels'].contiguous().view(-1)
        loss = loss_fct(scores, labels)
        val_loss += loss.item()
        n_correct += accuracy_count(scores, labels)
        n_word += batch['txt_labels'].numel()
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
                f"acc: {acc*100:.2f}")
    return val_log


def accuracy_count(out, labels):
    outputs = out.max(dim=-1)[1]
    mask = labels != -1
    n_correct = (outputs == labels).masked_select(mask).sum().item()
    return n_correct


@torch.no_grad()
def validate_mrfr(model, val_loader):
    LOGGER.info("start running MRFR validation...")
    val_loss = 0
    n_feat = 0
    st = time()
    for i, batch in enumerate(val_loader):
        loss, _ = model(batch, task='mrfr', compute_loss=True)
        val_loss += loss.sum().item() / IMG_DIM
        n_feat += batch['img_mask_tgt'].sum().item()
    val_loss = sum(all_gather_list(val_loss))
    n_feat = sum(all_gather_list(n_feat))
    tot_time = time()-st
    val_loss /= n_feat
    val_log = {'loss': val_loss,
               'feat_per_s': n_feat/tot_time}
    LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                f"loss: {val_loss:.2f}")
    return val_log


@torch.no_grad()
def validate_mrm_nce(model, val_loader):
    LOGGER.info("start running MRM-NCE validation...")
    val_loss = 0
    val_l2 = 0
    n_correct = 0
    cosine = 0
    n_feat = 0
    n_neg = 0
    st = time()
    for i, batch in enumerate(val_loader):
        feats, pos_feats, neg_feats = model(batch, task='mrm-nce',
                                            compute_loss=False)
        logits = model.mrm_nce(feats, pos_feats, neg_feats,
                               compute_loss=False)
        targets = torch.arange(0, logits.size(0),
                               dtype=torch.long, device=logits.device)
        val_loss += F.cross_entropy(logits, targets, reduction='sum').item()
        val_l2 += F.mse_loss(feats, pos_feats, reduction='sum'
                             ).item() / feats.size(-1)
        n_correct += (logits.max(dim=-1)[1] == targets).sum().item()
        cosine += F.cosine_similarity(feats, pos_feats, dim=-1).sum().item()
        nf = batch['img_mask_tgt'].sum().item()
        n_feat += nf
        n_neg += neg_feats.size(0) * nf
    val_loss = sum(all_gather_list(val_loss))
    val_l2 = sum(all_gather_list(val_l2))
    n_correct = sum(all_gather_list(n_correct))
    cosine = sum(all_gather_list(cosine))
    n_feat = sum(all_gather_list(n_feat))
    n_neg = sum(all_gather_list(n_neg))
    tot_time = time()-st
    val_loss /= n_feat
    val_acc = n_correct / n_feat
    val_log = {'loss': val_loss,
               'acc': val_acc,
               'l2': val_l2 / n_feat,
               'cosine': cosine / n_feat,
               'feat_per_s': n_feat/tot_time}
    LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                f"loss: {val_loss:.2f}, acc: {val_acc*100:.2f} "
                f"(average {n_neg/n_feat:.0f} negatives)")
    return val_log


@torch.no_grad()
def validate_mrc(model, val_loader, task):
    LOGGER.info("start running MRC validation...")
    val_loss = 0
    n_feat = 0
    st = time()
    tot_score = 0
    for i, batch in enumerate(val_loader):
        _, prediction_soft_label = model(
            batch, task=task, compute_loss=False)
        if "kl" in task:
            prediction_soft_label = F.log_softmax(
                prediction_soft_label, dim=-1)
            label_targets = batch['label_targets']
            loss = F.kl_div(
                prediction_soft_label, label_targets, reduction='sum')
            tot_score += compute_accuracy_for_soft_targets(
                prediction_soft_label, label_targets)
        else:
            # background class should not be the target
            cls_label_targets = label_targets[:, 1:].max(dim=-1)[1] + 1
            loss = F.cross_entropy(
                prediction_soft_label, cls_label_targets,
                ignore_index=0, reduction='sum')
            tot_score += compute_accuracy_for_soft_targets(
                prediction_soft_label[:, 1:], label_targets[:, 1:])
        val_loss += loss.item()
        n_feat += batch['img_mask_tgt'].sum().item()
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


def compute_accuracy_for_soft_targets(out, labels):
    outputs = out.max(dim=-1)[1]
    labels = labels.max(dim=-1)[1]  # argmax
    n_correct = (outputs == labels).sum().item()
    return n_correct


@torch.no_grad()
def validate_itm(model, val_loader):
    LOGGER.info("start running ITM validation...")
    val_loss = 0
    tot_ot_loss = 0
    tot_ot_pos = 0
    tot_ot_neg = 0
    tot_score = 0
    n_ex = 0
    st = time()
    for i, batch in enumerate(val_loader):
        loss, ot_loss, is_correct = model(batch, task='itm', compute_loss=False)
        if ot_loss is not None:
            if isinstance(ot_loss, tuple):
                ot_pos, ot_neg = ot_loss
                ot_pos = ot_pos.sum().item()
                ot_neg = ot_neg.sum().item()
                tot_ot_pos += ot_pos
                tot_ot_neg += ot_neg
                tot_ot_loss += ot_pos - ot_neg
            else:
                tot_ot_loss += ot_loss.sum().item()
        targets = batch['targets']
        val_loss += loss.mean().item()

        tot_score += is_correct.sum().item()
        n_ex += len(targets)
    val_loss = sum(all_gather_list(val_loss))
    tot_score = sum(all_gather_list(tot_score))
    n_ex = sum(all_gather_list(n_ex))
    tot_time = time()-st
    val_loss /= n_ex
    val_acc = tot_score / n_ex
    val_log = {'valid/loss': val_loss,
               'valid/acc': val_acc,
               'valid/ex_per_s': n_ex/tot_time}

    if ot_loss is not None:
        tot_ot_loss = sum(all_gather_list(tot_ot_loss))
        tot_ot_pos = sum(all_gather_list(tot_ot_pos))
        tot_ot_neg = sum(all_gather_list(tot_ot_neg))
        val_log['valid/ot_loss'] = tot_ot_loss / n_ex
        val_log['valid/ot_pos'] = tot_ot_pos / n_ex
        val_log['valid/ot_neg'] = tot_ot_neg / n_ex

    LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                f"score: {val_acc*100:.2f}")
    return val_log


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    # NOTE: train tasks and val tasks cannot take command line arguments
    parser.add_argument('--compressed_db', action='store_true',
                        help='use compressed LMDB')

    parser.add_argument("--model_config", type=str,
                        help="path to model structure config json")
    parser.add_argument("--checkpoint", default=None, type=str,
                        help="path to model checkpoint (*.pt)")

    parser.add_argument(
        "--output_dir", default=None, type=str,
        help="The output directory where the model checkpoints will be "
             "written.")

    parser.add_argument('--mrm_prob', default=0.15, type=float,
                        help='probability to mask in MRM training')
    parser.add_argument('--neg_size', default=128, type=int,
                        help='negative image size for NCE')
    parser.add_argument('--nce_temp', default=1.0, type=float,
                        help='softmax temperature for NCE')
    parser.add_argument('--itm_neg_prob', default=0.5, type=float,
                        help='probability to make negative examples'
                             'in ITM training')
    parser.add_argument('--itm_ot_lambda', default=0.0, type=float,
                        help='weight of OT (optimal transport) loss')
    parser.add_argument('--ot_pos_only', action='store_true',
                        help='use OT distance of positive pairs only')

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
    parser.add_argument("--train_batch_size", default=4096, type=int,
                        help="Total batch size for training. "
                             "(batch by tokens)")
    parser.add_argument("--val_batch_size", default=4096, type=int,
                        help="Total batch size for validation. "
                             "(batch by tokens)")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16,
                        help="Number of updates steps to accumualte before "
                             "performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--valid_steps", default=1000, type=int,
                        help="Run validation every X steps")
    parser.add_argument("--num_train_steps", default=100000, type=int,
                        help="Total number of training updates to perform.")
    parser.add_argument("--optim", default='adamw',
                        choices=['adam', 'adamax', 'adamw'],
                        help="optimizer")
    parser.add_argument("--betas", default=[0.9, 0.98], nargs='+',
                        help="beta for adam optimizer")
    parser.add_argument("--decay", default='linear',
                        choices=['linear', 'invsqrt'],
                        help="learning rate decay method")
    parser.add_argument("--dropout", default=0.1, type=float,
                        help="tune dropout regularization")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="weight decay (L2) regularization")
    parser.add_argument("--grad_norm", default=2.0, type=float,
                        help="gradient clipping (-1 for no clipping)")
    parser.add_argument("--warmup_steps", default=10000, type=int,
                        help="Number of training steps to perform linear "
                             "learning rate warmup for. (invsqrt decay)")

    # device parameters
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead "
                             "of 32-bit")
    parser.add_argument('--n_workers', type=int, default=4,
                        help="number of data workers")
    parser.add_argument('--pin_mem', action='store_true', help="pin memory")

    # can use config files
    parser.add_argument('--config', required=True, help='JSON config files')


    # added parameters
    parser.add_argument('--project_name', default='debug', type=str, help="")
    parser.add_argument('--project_dim', default=0, type=int, help="")
    parser.add_argument('--txt_model_config', default='bert-base', type=str, help="")
    parser.add_argument('--txt_checkpoint', default=None, type=str, help="")
    parser.add_argument('--img_model_config', default='uniter-base', type=str, help="")
    parser.add_argument('--img_checkpoint', default=None, type=str, help="")
    parser.add_argument('--biencoder_checkpoint', default=None, type=str, help="")
    parser.add_argument('--cls_concat', default="", type=str, help="")
    parser.add_argument('--fix_txt_encoder', action='store_true', help='')
    parser.add_argument('--fix_img_encoder', action='store_true', help='')

    # KD parameters
    parser.add_argument('--teacher_checkpoint', default=None, type=str, help="")
    parser.add_argument('--T', default=1.0, type=float, help="")
    parser.add_argument('--kd_loss_weight', default=0.5, type=float, help="")
    parser.add_argument('--normalized_logits', action='store_true', help="")

    args = parse_with_config(parser)

    if exists(args.output_dir) and os.listdir(args.output_dir):
        model_pts = glob.glob(os.path.join(args.output_dir, 'ckpt', 'model_step*.pt'))
        if len(model_pts) == 0:
            args.train_state = None
        else:
            max_step = max([int(os.path.basename(pt).split('_')[-1][:-3]) for pt in model_pts])
            last_pt = os.path.join(args.output_dir, 'ckpt', f'model_step_{max_step}.pt')
            train_state = os.path.join(args.output_dir, 'ckpt', f'train_state_{max_step}.pt')
            assert os.path.isfile(last_pt) and os.path.isfile(train_state), f'last train state/model pt does not exist, step = {max_step}'
            args.train_state = {'model_file': last_pt, 'state_file': train_state, 'step': max_step}
    else:
        args.train_state = None

    # options safe guard
    if args.conf_th == -1:
        assert args.max_bb + args.max_txt_len + 2 <= 512
    else:
        assert args.num_bb + args.max_txt_len + 2 <= 512

    main(args)
