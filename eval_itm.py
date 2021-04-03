import argparse
import sys
import logging
import torch
import time
import json
import os
import collections
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

from horovod import torch as hvd

from uniter_model.data import ImageLmdbGroup
from uniter_model.data.loader import PrefetchLoader

from dvl.options import default_params, add_itm_params, parse_with_config, add_logging_params
from dvl.data.itm import TxtTokLmdb, ItmFastDataset, ItmValDataset, itm_fast_collate
from dvl.models.bi_encoder import BertEncoder, UniterEncoder, BiEncoder, get_optimizer, setup_for_distributed_mode, \
    BiEncoderNllLoss, get_schedule_linear
from dvl.indexer.faiss_indexers import DenseFlatIndexer
from dvl.utils import print_args, num_of_parameters, _calc_loss, compare_models
from dvl.trainer import load_saved_state, load_states_from_checkpoint, load_dataset, eval_model_on_dataloader, build_dataloader
from transformers.tokenization_bert import BertTokenizer


def EVAL_MODEL(config_file, bi_encoder_checkpoint, project_name='itm-eval-dev'):
    SEARCH_MODE = 'approx'
    DEBUG = True


    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if (logger.hasHandlers()):
        logger.handlers.clear()
    console = logging.StreamHandler()
    logger.addHandler(console)

    parser = argparse.ArgumentParser()
    default_params(parser)
    add_itm_params(parser)
    add_logging_params(parser)
    if DEBUG:
        args = parse_with_config(parser, [
            '--config', config_file,
            '--biencoder_checkpoint', bi_encoder_checkpoint,
            '--project_name', project_name,
            '--expr_name_prefix', 'UNITER-',
        ])
    else:
        args = parse_with_config(parser)

    args.tokenizer = BertTokenizer.from_pretrained(args.txt_model_config)
    parsed_args = os.path.basename(os.path.dirname(args.biencoder_checkpoint)).split('_')
    try:
        args.learning_rate, args.train_batch_size, args.num_hard_negatives, args.hard_negatives_sampling, \
        args.caption_score_weight= parsed_args[1:-1]
        args.caption_score_weight = float(args.caption_score_weight)
    except ValueError:
        args.learning_rate, args.train_batch_size, args.num_hard_negatives, args.hard_negatives_sampling, \
        args.caption_score_weight = 0, 0, 0, 0, 0

    if len(parsed_args) >= 4:
        args.hard_negatives_sampling = parsed_args[3]

    # options safe guard
    if args.conf_th == -1:
        assert args.max_bb + args.max_txt_len + 2 <= 512
    else:
        assert args.num_bb + args.max_txt_len + 2 <= 512
    # assert (args.hard_neg_size <= args.hard_neg_pool_size <= args.inf_minibatch_size)
    if args.steps_per_hard_neg != -1:
        assert args.hard_neg_size > 0

    hvd.init()
    n_gpu = hvd.size()
    args.device = torch.device("cuda", hvd.local_rank())
    args.local_rank = hvd.rank()
    args.n_gpu = hvd.size()
    args.inf_minibatch_size = 400
    args.vector_size = args.project_dim
    torch.cuda.set_device(hvd.local_rank())
    print_args(args)

    if args.itm_global_file is not None:
        with open(args.itm_global_file) as f:
            args.img_meta = json.load(f)
    else:
        args.img_meta = None

    experiment = None
    # experiment = Experiment(api_key='your api key', workspace='your workspace name', project_name=args.project_name)
    # experiment.set_name(f'{args.expr_name_prefix}{args.train_batch_size}-{args.learning_rate}-{args.num_hard_negatives}')

    bi_encoder = BiEncoder(args, args.fix_img_encoder, args.fix_txt_encoder, project_dim=args.project_dim)
    state_dict = torch.load(args.biencoder_checkpoint, map_location='cpu')
    try:
        bi_encoder.load_state_dict(state_dict['model_dict'])
    except KeyError:
        logger.info('loading from pre-trained model instead')
        for k in list(state_dict.keys()):
            if k.startswith('bert.'):
                state_dict[k[5:]] = state_dict.pop(k)
            else:
                state_dict.pop(k)
        bi_encoder.load_state_dict(state_dict, strict=True)

    print(f'total #params in img model = {num_of_parameters(bi_encoder.img_model)}, '
          f'in txt model = {num_of_parameters(bi_encoder.txt_model)}')

    img_model, txt_model = bi_encoder.img_model, bi_encoder.txt_model
    img_model.to(args.device)
    txt_model.to(args.device)

    img_model, _ = setup_for_distributed_mode(img_model, None, args.device, args.n_gpu, -1, args.fp16, args.fp16_opt_level)
    img_model.eval()

    txt_model, _ = setup_for_distributed_mode(txt_model, None, args.device, args.n_gpu, -1, args.fp16, args.fp16_opt_level)
    txt_model.eval()

    # Load Data
    all_img_dbs = ImageLmdbGroup(args.conf_th, args.max_bb, args.min_bb, args.num_bb, args.compressed_db)
    if experiment is not None:
        experiment.log_metric('lr', args.learning_rate)
        experiment.log_metric('bsz', args.train_batch_size)
        experiment.log_metric('num_hard_neg', args.num_hard_negatives)
        experiment.log_metric('hard_neg_sampling', args.hard_negatives_sampling)

    for partition, txt_db, img_db in zip(['dev', 'test'],
                                         [args.val_txt_db, args.test_txt_db],
                                         [args.val_img_db, args.test_img_db]):
        if partition in ['train']:
            continue

        if txt_db is None:
            continue
        print('*'*100)
        print('for set', partition)
        dataset = load_dataset(all_img_dbs, txt_db, img_db, args, is_train=False)
        dataset.new_epoch()
        dataloader = build_dataloader(dataset, itm_fast_collate, False, args)
        logger.info(f'dataset len = {len(dataset)}, dataloader len = {len(dataloader)}')

        img2txt = dict(collections.ChainMap(*[json.load(open(os.path.join(db_folder, 'img2txts.json'))) for db_folder in [txt_db]]))
        start_time = time.time()
        loss_val, correct_ratio_val, (indexer_img, indexer_txt), (recall_img, recall_txt), _ = eval_model_on_dataloader(bi_encoder, dataloader, args, img2txt=img2txt)
        print(f'time cost = {time.time() - start_time}s')
        print(f'average loss = {loss_val}, accuracy = {correct_ratio_val}')
        print('indexed ', len(indexer_img.index_id_to_db_id), 'data')
        print('image retrieval recall =', recall_img)
        print('txt retrieval recall =', recall_txt)

        recall_mean_img = np.mean(list(recall_img.values()))
        recall_mean_txt = np.mean(list(recall_txt.values()))
        if experiment is not None:
            experiment.log_metric('img_R@1_'+partition, "{:.2f}".format(round(recall_img[1]*100, 2)))
            experiment.log_metric('img_R@5_'+partition, "{:.2f}".format(round(recall_img[5]*100, 2)))
            experiment.log_metric('img_R@10_'+partition, "{:.2f}".format(round(recall_img[10]*100, 2)))
            experiment.log_metric('img_R@mean_'+partition, "{:.2f}".format(round(recall_mean_img*100, 2)))

            experiment.log_metric('img_R@1_'+partition, "{:.2f}".format(round(recall_txt[1]*100, 2)))
            experiment.log_metric('img_R@5_'+partition, "{:.2f}".format(round(recall_txt[5]*100, 2)))
            experiment.log_metric('img_R@10_'+partition, "{:.2f}".format(round(recall_txt[10]*100, 2)))
            experiment.log_metric('img_R@mean_'+partition, "{:.2f}".format(round(recall_mean_txt*100, 2)))
            experiment.log_metric('correct_ratio_'+partition, "{:.2f}".format(round(correct_ratio_val*100, 2)))

            experiment.log_metric('loss_test_'+partition, "{:.4f}".format(loss_val))
            experiment.log_metric('n_image_'+partition, len(indexer_img.index_id_to_db_id))


# Zero-shot evaluation for LightningDot on Flickr30k
# EVAL_MODEL('./config/flickr30k_eval_config.json', './data/model/LightningDot.pt')

# Fine-tune on Flickr30k, evaluate on Flickr30k
# EVAL_MODEL('./config/flickr30k_eval_config.json', './data/model/flickr-ft.pt')

# Fine-tune on COCO, evaluate on COCO
# EVAL_MODEL('./config/coco_eval_config.json', './data/model/coco-ft.pt')

EVAL_MODEL(sys.argv[1], sys.argv[2])

