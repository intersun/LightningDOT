import argparse
import json
import sys
import os
import logging
import torch
import random
import socket
import numpy as np


logger = logging.getLogger()


def default_params(parser: argparse.ArgumentParser):
    parser.add_argument('--txt_model_type', default='bert-base', type=str, help="")
    parser.add_argument('--txt_model_config', default='bert-base', type=str, help="")
    parser.add_argument('--txt_checkpoint', default=None, type=str, help="")
    parser.add_argument('--img_model_type', default='uniter-base', type=str, help="")
    parser.add_argument('--img_model_config', default='./config/img_base.json', type=str, help="")
    parser.add_argument('--img_checkpoint', default=None, type=str, help="")
    parser.add_argument('--biencoder_checkpoint', default=None, type=str, help="")
    parser.add_argument('--seperate_caption_encoder', action='store_true', help="")

    parser.add_argument('--train_batch_size', default=80, type=int, help="")
    parser.add_argument('--valid_batch_size', default=80, type=int, help="")
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, help="")
    parser.add_argument('--learning_rate', default=1e-5, type=float, help="")
    parser.add_argument('--max_grad_norm', default=2.0, type=float, help="")
    parser.add_argument('--warmup_steps', default=500, type=int, help="")
    parser.add_argument('--valid_steps', default=500, type=int, help="")
    parser.add_argument('--num_train_steps', default=5000, type=int, help="")
    parser.add_argument('--num_train_epochs', default=0, type=int, help="")

    parser.add_argument('--fp16', action='store_true', help="")
    parser.add_argument('--seed', default=42, type=int, help="")
    parser.add_argument('--output_dir', default='./', type=str, help="")
    parser.add_argument('--max_txt_len', default=64, type=int, help="")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument('--config', default=None, type=str, help="")
    parser.add_argument('--itm_global_file', default=None, type=str, help="")
    parser.add_argument("--no_cuda", action='store_true', help="Whether not to use CUDA when available")
    parser.add_argument('--n_workers', type=int, default=2, help="number of data workers")
    parser.add_argument('--pin_mem', action='store_true', help="pin memory")   # ???
    parser.add_argument('--hnsw_index', action='store_true', help="")
    parser.add_argument('--fp16_opt_level', type=str, default='O1', help="")
    parser.add_argument('--img_meta', type=str, default=None, help="")


def add_itm_params(parser: argparse.ArgumentParser):
    parser.add_argument('--conf_th', default=0.2, type=float, help="")
    parser.add_argument('--caption_score_weight', default=0.0, type=float, help="")
    parser.add_argument('--negative_size', default=10, type=int, help="")
    parser.add_argument('--num_hard_negatives', default=0, type=int, help="")
    parser.add_argument('--sample_init_hard_negatives', action='store_true', help="")
    parser.add_argument('--hard_negatives_sampling', default='none', type=str,
                        choices=['none', 'random', 'top', 'top-random', '10-20', '20-30'], help="")
    parser.add_argument('--max_bb', default=100, type=int, help="")
    parser.add_argument('--min_bb', default=10, type=int, help="")
    parser.add_argument('--num_bb', default=36, type=int, help="")
    parser.add_argument('--train_txt_dbs', default=None, type=str, help="")
    parser.add_argument('--train_img_dbs', default=None, type=str, help="")

    parser.add_argument('--txt_db_mapping', default=None, type=str, help="")
    parser.add_argument('--img_db_mapping', default=None, type=str, help="")
    parser.add_argument('--pretrain_mapping', default=None, type=str, help="")

    parser.add_argument('--val_txt_db', default=None, type=str, help="")
    parser.add_argument('--val_img_db', default=None, type=str, help="")
    parser.add_argument('--test_txt_db', default=None, type=str, help="")
    parser.add_argument('--test_img_db', default=None, type=str, help="")
    parser.add_argument('--steps_per_hard_neg', default=-1, type=int, help="")
    parser.add_argument('--inf_minibatch_size', default=400, type=int, help="")
    parser.add_argument('--project_dim', default=0, type=int, help='')
    parser.add_argument('--cls_concat', default="", type=str, help='')
    parser.add_argument('--fix_txt_encoder', action='store_true', help='')
    parser.add_argument('--fix_img_encoder', action='store_true', help='')
    parser.add_argument('--compressed_db', action='store_true', help='use compressed LMDB')
    parser.add_argument('--retrieval_mode', default="both",
                        choices=['img_only', 'txt_only', 'both'], type=str, help="")


def add_logging_params(parser: argparse.ArgumentParser):
    parser.add_argument('--log_result_step', default=4, type=int, help="")
    parser.add_argument('--project_name', default='itm', type=str, help="")
    parser.add_argument('--expr_name_prefix', default='', type=str, help="")
    parser.add_argument('--save_all_epochs', action='store_true', help="")


def add_kd_params(parser: argparse.ArgumentParser):
    parser.add_argument('--teacher_checkpoint', default=None, type=str, help="")
    parser.add_argument('--T', default=1.0, type=float, help="")
    parser.add_argument('--kd_loss_weight', default=1.0, type=float, help="")


def parse_with_config(parser, cmds=None):
    if cmds is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(cmds)

    if args.config is not None:
        config_args = json.load(open(args.config))
        override_keys = {arg[2:].split('=')[0] for arg in sys.argv[1:]
                         if arg.startswith('--')}
        for k, v in config_args.items():
            if k not in override_keys:
                setattr(args, k, v)
    return args


def map_db_dirs(args):
    # map img db
    for k in args.__dict__:
        if not isinstance(args.__dict__[k], str):
            continue
        if args.__dict__[k].startswith('/pretrain') and args.pretrain_mapping:
            print('pretrain', k, args.__dict__[k])
            args.__dict__[k] = args.__dict__[k].replace('/pretrain', args.pretrain_mapping)
        if args.__dict__[k].startswith('/db') and args.txt_db_mapping:
            print('db', k, args.__dict__[k])
            args.__dict__[k] = args.__dict__[k].replace('/db', args.txt_db_mapping)
        if args.__dict__[k].startswith('/img') and args.img_db_mapping:
            print('img', k, args.__dict__[k])
            args.__dict__[k] = args.__dict__[k].replace('/img', args.img_db_mapping)

    if args.img_db_mapping:
        for i in range(len(args.train_img_dbs)):
            args.train_img_dbs[i] = args.train_img_dbs[i].replace('/img', args.img_db_mapping)
    if args.txt_db_mapping:
        for i in range(len(args.train_txt_dbs)):
            args.train_txt_dbs[i] = args.train_txt_dbs[i].replace('/db', args.txt_db_mapping)




def print_args(args):
    logger.info(" **************** CONFIGURATION **************** ")
    for key, val in sorted(vars(args).items()):
        keystr = "{}".format(key) + (" " * (30 - len(key)))
        logger.info("%s -->   %s", keystr, val)
    logger.info(" **************** END CONFIGURATION **************** ")


def set_seed(args):
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def setup_args_gpu(args):
    """
     Setup arguments CUDA, GPU & distributed training
    """
    if args.local_rank == -1 or args.no_cuda:  # single-node multi-gpu (or cpu) mode
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # distributed mode
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device
    ws = os.environ.get('WORLD_SIZE')

    args.distributed_world_size = int(ws) if ws else 1

    logger.info(
        'Initialized host %s as d.rank %d on device=%s, n_gpu=%d, world size=%d', socket.gethostname(),
        args.local_rank, device,
        args.n_gpu,
        args.distributed_world_size)
    logger.info("16-bits training: %s ", args.fp16)
