
import argparse
import os
import logging
import torch
import json
import random
import itertools
import time
import collections
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

from horovod import torch as hvd

from GLOBAL_VARIABLES import N_EXAMPLES_TEACHER
from uniter_model.data import ImageLmdbGroup
from uniter_model.data.loader import PrefetchLoader
from uniter_model.model.itm import UniterForImageTextRetrieval
from transformers.tokenization_bert import BertTokenizer

from dvl.options import default_params, add_itm_params, add_logging_params, add_kd_params, parse_with_config, map_db_dirs
from dvl.data.itm import TxtTokLmdb, ItmFastDataset, ItmValDataset, itm_fast_collate, itm_fast_collate_kd
from dvl.models.bi_encoder import BiEncoder, get_optimizer, setup_for_distributed_mode, \
    BiEncoderNllLoss, get_schedule_linear, load_biencoder_checkpoint
from dvl.utils import print_args, num_of_parameters, _calc_loss, is_main_process
from dvl.hn import random_hard_neg, get_img_txt_mappings, sampled_hard_negatives
from dvl.const import IMG_DIM
from dvl.trainer import build_dataloader, _save_checkpoint, eval_model_on_dataloader, load_dataset
from dvl.indexer.faiss_indexers import DenseFlatIndexer


DEBUG_FLAG = False
torch.set_num_threads(4)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)


def train_parser(parser):
    default_params(parser)
    add_itm_params(parser)
    add_logging_params(parser)
    add_kd_params(parser)
    return parser


parser = argparse.ArgumentParser()
parser = train_parser(parser)
if DEBUG_FLAG:
    args = parse_with_config(parser, [
        '--config', './config/coco_ft_config_bert_debug.json',
        '--sample_init_hard_negatives'
    ])
    args.retrieval_mode = 'both'
else:
    args = parse_with_config(parser)

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir, exist_ok=True)
# options safe guard
if args.conf_th == -1:
    assert args.max_bb + args.max_txt_len + 2 <= 512
else:
    assert args.num_bb + args.max_txt_len + 2 <= 512


hvd.init()
torch.cuda.set_device(hvd.local_rank())
args.device = torch.device("cuda", hvd.local_rank())
args.local_rank = hvd.rank()
args.n_gpu = hvd.size()
args.fp16_opt_level = 'O2'  # for now let us assume always set opt level as O2
args.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
if args.project_dim > 0:
    args.vector_size = args.project_dim
else:
    args.vector_size = 768


if args.teacher_checkpoint is not None:
    logger.info('teacher checkpoint is provided, using KD framework')
    teacher_model = UniterForImageTextRetrieval.from_pretrained(os.path.join(args.teacher_checkpoint, 'config.json'),
                                                                state_dict=torch.load(os.path.join(args.teacher_checkpoint, 'model.pt')),
                                                                img_dim=IMG_DIM,
                                                                margin=0.2)
    collate_func = itm_fast_collate_kd
else:
    teacher_model = None
    collate_func = itm_fast_collate


if is_main_process():
    experiment = None
    # options for DEBUG
    print_args(args)
    if experiment is not None:
        experiment.log_parameters({
            'train_batch_size': args.train_batch_size,
            'learning_rate': args.learning_rate,
            'num_hard_negatives': args.num_hard_negatives,
            'hard_negatives_sampling': args.hard_negatives_sampling,
            'caption_score_weight': args.caption_score_weight,
            'kd_score_weight': args.kd_loss_weight,
            'Temperature': args.T,
            'project_dim': args.project_dim,
            'retrieval_mode': args.retrieval_mode
        })

if args.itm_global_file is not None:
    with open(args.itm_global_file) as f:
        args.img_meta = json.load(f)
else:
    args.img_meta = None

# Init Model
bi_encoder = BiEncoder(args, args.fix_img_encoder, args.fix_txt_encoder, args.project_dim)
load_biencoder_checkpoint(bi_encoder, args.biencoder_checkpoint)
optimizer = get_optimizer(bi_encoder, args.learning_rate)

logger.info(f'total #params in img model = {num_of_parameters(bi_encoder.img_model)}, '
            f'in txt model = {num_of_parameters(bi_encoder.txt_model)}')

logger.info(f'total #params in biencoder model = {num_of_parameters(bi_encoder, requires_grad=True)}')

bi_encoder, optimizer = setup_for_distributed_mode(bi_encoder, optimizer, args.device, args.n_gpu,
                                                   # args.local_rank,
                                                   -1,
                                                   args.fp16,
                                                   args.fp16_opt_level,
                                                   teacher_model=teacher_model)

# Load Data
all_img_dbs = ImageLmdbGroup(args.conf_th, args.max_bb, args.min_bb, args.num_bb, args.compressed_db)

# img2txt and txt2img mapping
train_img2txt, train_txt2img, train_img2set, train_txt2set, train_set2img, train_set2txt = \
    get_img_txt_mappings(args.train_txt_dbs)

if args.sample_init_hard_negatives:
    assert args.num_hard_negatives > 0, 'for init, num hard negatives has to > 0'
    hard_neg_txt, hard_neg_img = sampled_hard_negatives(all_img_dbs, args, collate_func, bi_encoder, train_img2txt, train_txt2img)
else:
    # hard_negatives = random_hard_neg(train_img2txt, args.num_hard_negatives, train_img2set, train_set2img)
    if args.num_hard_negatives > 0:
        raise NotImplementedError('random init hard negatives not impelmented yet')
    else:
        hard_neg_txt, hard_neg_img = None, None

val_img2txt = json.load(open(os.path.join(args.val_txt_db, 'img2txts.json')))

# load train and dev
logger.info(f"Loading Train Dataset "
            f"{args.train_txt_dbs}, {args.train_img_dbs}")
train_dataset = load_dataset(all_img_dbs, args.train_txt_dbs, args.train_img_dbs, args, True)
for dset in train_dataset.datasets:
    dset.new_epoch(hard_neg_img, hard_neg_txt)
train_dataloader = build_dataloader(train_dataset, collate_func, True, args)
logger.info(f'train dataset len = {len(train_dataset)}, dataloader len = {len(train_dataloader)}')

val_dataset = load_dataset(all_img_dbs, args.val_txt_db, args.val_img_db, args, is_train=False)
val_dataset.new_epoch()
val_dataloader = build_dataloader(val_dataset, collate_func, False, args)
logger.info(f'dev dataset len = {len(val_dataset)}, dataloader len = {len(val_dataloader)}')

updates_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
total_updates = updates_per_epoch * args.num_train_epochs
warmup_steps = int(0.1 * total_updates)
scheduler = get_schedule_linear(optimizer, warmup_steps, total_updates)
best_eval_metric = 0.0

if teacher_model:
    # teacher model will always be in eval mode
    teacher_model.eval()

for epoch in range(args.num_train_epochs):
    epoch_loss, epoch_correct_predictions, rolling_train_loss = 0, 0, 0.0
    bi_encoder.train()
    logger.info('*' * 70)
    if experiment is not None:
        experiment.log_metric('epoch', epoch)

    for dset in train_dataset.datasets:
        dset.new_epoch(hard_neg_img, hard_neg_txt)
    for step, batch in enumerate(train_dataloader):
        model_out = bi_encoder(batch)
        txt_vector, img_vectors, caption_vectors = model_out

        loss_function = BiEncoderNllLoss()
        bs = batch['sample_size']
        if args.num_hard_negatives > 0:
            loss_nce_txt, is_correct_txt, scores_txt = _calc_loss(args, loss_function, img_vectors[:bs], txt_vector, caption_vectors,
                                                          batch['pos_ctx_indices'], batch['neg_ctx_indices'], experiment)
            loss_nce_img, is_correct_img, scores_img = _calc_loss(args, loss_function, txt_vector[:bs], img_vectors, caption_vectors,
                                                          batch['pos_ctx_indices'], batch['neg_ctx_indices'], experiment)
        else:
            loss_nce_txt, is_correct_txt, scores_txt = _calc_loss(args, loss_function, img_vectors, txt_vector,
                                                                  caption_vectors,
                                                                  batch['pos_ctx_indices'], batch['neg_ctx_indices'],
                                                                  experiment)
            loss_nce_img, is_correct_img, scores_img = _calc_loss(args, loss_function, txt_vector, img_vectors,
                                                                  caption_vectors,
                                                                  batch['pos_ctx_indices'], batch['neg_ctx_indices'],
                                                                  experiment)
        if args.retrieval_mode in['txt_only']:
            raise ValueError('not supported anymore')
            is_correct, scores = is_correct_txt.sum().item(), scores_txt
            loss_nce = loss_nce_txt
        elif args.retrieval_mode in ['img_only']:
            raise ValueError('not supported anymore')
            is_correct, scores = is_correct_img.sum().item(), scores_img
            loss_nce = loss_nce_img
        else:
            is_correct = (is_correct_txt.sum().item() + is_correct_img.sum().item()) / 2
            loss_nce = 0.5 * loss_nce_txt + 0.5 * loss_nce_img
            scores = scores_txt * 0.5 + scores_img * 0.5

        if teacher_model:
            batch_new = {'gather_index': None}
            for k in batch:
                if 'teacher' in k:
                    new_k = k.replace('_teacher', '')
                    batch_new[new_k] = batch[k]

            with torch.no_grad():
                teacher_scores = teacher_model(batch_new, compute_loss=False).reshape(len(batch['txt_ids']), -1).T
                assert teacher_scores.shape[0] == N_EXAMPLES_TEACHER, 'number of teacher example does not match'
            # teacher_prob = teacher_prob.softmax(dim=1)

            # KD loss
            loss_kd = nn.KLDivLoss()(F.log_softmax(scores[:N_EXAMPLES_TEACHER] / args.T, dim=1),
                                     F.softmax(teacher_scores / args.T, dim=1)) * args.T * args.T
            loss = loss_nce + args.kd_loss_weight * loss_kd
        else:
            loss = loss_nce

        if args.n_gpu > 1:
            loss = loss.mean()
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        epoch_correct_predictions += is_correct
        epoch_loss += loss.item()
        rolling_train_loss += loss.item()

        if args.fp16:
            from apex import amp

            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
        else:
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(bi_encoder.parameters(), args.max_grad_norm)

        if (step+1) % args.log_result_step == 0 and is_main_process():
            lr = optimizer.param_groups[0]['lr']
            if args.teacher_checkpoint:
                logger.info(
                    'Epoch: %d: Step: %d/%d, loss=%f, loss_nce=%f, loss_kd=%f, lr=%f', epoch, step,
                    len(train_dataloader), loss.item(), loss_nce.item(), loss_kd.item(), lr)
            else:
                logger.info(
                    'Epoch: %d: Step: %d/%d, loss=%f, loss_nce=%f, loss_kd=0.0, lr=%f', epoch, step,
                    len(train_dataloader), loss.item(), loss_nce.item(), lr)

            if experiment is not None:
                experiment.log_metric('step', step)
                experiment.log_metric('lr', lr)
                experiment.log_metric('loss_train', loss.item())
                experiment.log_metric('loss_nce', loss_nce.item())
                experiment.log_metric('loss_nce_txt', loss_nce_txt.item())
                experiment.log_metric('loss_nce_img', loss_nce_img.item())

                if args.teacher_checkpoint:
                    experiment.log_metric('loss_kd', loss_kd.item())

        if (step + 1) % args.gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            bi_encoder.zero_grad()

        rolling_loss_step = args.log_result_step
        if (step + 1) % rolling_loss_step == 0:
            latest_rolling_train_av_loss = rolling_train_loss / rolling_loss_step
            # logger.info('Train batch %d', step)
            # logger.info('Avg. loss per last %d batches: %f', rolling_loss_step, latest_rolling_train_av_loss)
            rolling_train_loss = 0.0
            if experiment is not None and is_main_process():
                experiment.log_metric('rolling_loss', latest_rolling_train_av_loss)

        if step == 5 and DEBUG_FLAG:
            logger.info('break for debug')
            break

    epoch_loss = (epoch_loss / len(train_dataloader)) if len(train_dataloader) > 0 else 0
    total_samples = max(len(train_dataloader) * args.train_batch_size * 1, 1)
    correct_ratio = float(epoch_correct_predictions / total_samples)
    if experiment is not None and is_main_process():
        logger.info(f'Av Loss per epoch = {epoch_loss}, epoch total correct predictions = {correct_ratio}')
        experiment.log_metric('epoch_loss', epoch_loss)
        experiment.log_metric('correct_number_train', epoch_correct_predictions)
        experiment.log_metric('correct_ratio_train', correct_ratio)

    # eval and save
    bi_encoder.eval()
    img2txt = dict(collections.ChainMap(*[json.load(open(os.path.join(db_folder, 'img2txts.json'))) for db_folder in [args.val_txt_db]]))
    loss_val, correct_ratio_val, indexer_val, recall_both, _ = eval_model_on_dataloader(bi_encoder, val_dataloader, args, img2txt=img2txt)

    recall_val = dict()
    for t in recall_both[0]:
        recall_val[t] = (recall_both[0][t] + recall_both[1][t]) / 2

    current_eval_metric = np.mean(list(recall_val.values()))
    if experiment is not None and is_main_process():
        logger.info(f'val loss = {loss_val}. val correct prediction ratio = {correct_ratio_val}, recall = {recall_val}')
        experiment.log_metric('total_valid_loss', loss_val)
        experiment.log_metric('correct_ratio_valid', correct_ratio_val)
        experiment.log_metric('R@1', recall_val[1])
        experiment.log_metric('R@5', recall_val[5])
        experiment.log_metric('R@10', recall_val[10])
        experiment.log_metric('R@mean', current_eval_metric)

        experiment.log_metric('img_R@1', recall_both[0][1])
        experiment.log_metric('img_R@5', recall_both[0][5])
        experiment.log_metric('img_R@10', recall_both[0][10])
        experiment.log_metric('img_R@mean', np.mean(list(recall_both[0].values())))

        experiment.log_metric('txt_R@1', recall_both[1][1])
        experiment.log_metric('txt_R@5', recall_both[1][5])
        experiment.log_metric('txt_R@10', recall_both[1][10])
        experiment.log_metric('txt_R@mean', np.mean(list(recall_both[1].values())))


    if current_eval_metric > best_eval_metric and is_main_process():
        _save_checkpoint(args, bi_encoder, optimizer, scheduler, epoch, 0, 'best')
    if is_main_process():
        _save_checkpoint(args, bi_encoder, optimizer, scheduler, epoch, 0, 'last')

    if is_main_process() and args.save_all_epochs:
        _save_checkpoint(args, bi_encoder, optimizer, scheduler, epoch, 0)

    # sample hard negative in here
    if args.num_hard_negatives > 0:
        hard_neg_txt, hard_neg_img = sampled_hard_negatives(all_img_dbs, args, collate_func, bi_encoder, train_img2txt,
                                                            train_txt2img)
    else:
        # no hard negative sampling
        hard_neg_txt, hard_neg_img = None, None
        assert args.hard_negatives_sampling == 'none', f'sampleing method {args.hard_negatives_sampling} is not none'


if args.test_txt_db:
    test_dataset = load_dataset(all_img_dbs, args.test_txt_db, args.test_img_db, args, is_train=False)
    test_dataset.new_epoch()
    test_dataloader = build_dataloader(test_dataset, collate_func, False, args)
    logger.info(f'test dataset len = {len(test_dataset)}, dataloader len = {len(test_dataloader)}')
    bi_encoder.eval()
    img2txt = dict(collections.ChainMap(*[json.load(open(os.path.join(db_folder, 'img2txts.json'))) for db_folder in [args.test_txt_db]]))
    loss_test, correct_ratio_test, indexer_test, (recall_img, recall_txt), _ = eval_model_on_dataloader(bi_encoder, test_dataloader, args,
                                                                                           args.txt_retrieval, img2txt)
    recall_mean = np.mean(list(recall_img.values()))
    if experiment is not None:
        experiment.log_metric('test_img:R@1', "{:.4f}".format(round(recall_img[1], 4)))
        experiment.log_metric('test_img:R@5', "{:.4f}".format(round(recall_img[5], 4)))
        experiment.log_metric('test_img:R@10', "{:.4f}".format(round(recall_img[10], 4)))
        experiment.log_metric('test_img:R@mean', "{:.4f}".format(round(recall_mean, 4)))

    recall_mean = np.mean(list(recall_txt.values()))

    if experiment is not None:
        experiment.log_metric('test_txt:R@1', "{:.4f}".format(round(recall_txt[1], 4)))
        experiment.log_metric('test_txt:R@5', "{:.4f}".format(round(recall_txt[5], 4)))
        experiment.log_metric('test_txt:R@10', "{:.4f}".format(round(recall_txt[10], 4)))
        experiment.log_metric('test_txt:R@mean', "{:.4f}".format(round(recall_mean, 4)))

        experiment.log_metric('correct_ratio_test', "{:.4f}".format(round(correct_ratio_test, 4)))
        experiment.log_metric('loss_test', "{:.4f}".format(loss_test))
        experiment.log_metric('n_image_test', len(indexer_test.index_id_to_db_id))
