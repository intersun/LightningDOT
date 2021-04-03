import argparse
import torch
import json
import os
import pickle

from horovod import torch as hvd
from torch.utils.data import DataLoader, ConcatDataset
from collections import ChainMap

from horovod import torch as hvd

from uniter_model.data import ImageLmdbGroup
from transformers.tokenization_bert import BertTokenizer

from dvl.options import default_params, add_itm_params, add_logging_params, parse_with_config
from dvl.data.itm import itm_fast_collate
from dvl.models.bi_encoder import BiEncoder, setup_for_distributed_mode, load_biencoder_checkpoint
from dvl.utils import print_args, num_of_parameters, is_main_process, get_model_encoded_vecs, retrieve_query, display_img
from dvl.trainer import build_dataloader, load_dataset
from dvl.indexer.faiss_indexers import DenseFlatIndexer


from dvl.options import default_params, add_itm_params, add_logging_params, parse_with_config


def get_db_names(data_name, Full):
    if data_name == 'flickr':
        cmd = '--config ./config/flickr30k_eval_config.json '\
              '--biencoder_checkpoint  /good_models/flickr_two-stream-add/biencoder.last.pt ' \
              '--teacher_checkpoint /pretrain/uniter_teacher_flickr.pt ' \
              '--img_meta /db/meta/flickr_meta.json'
        if Full:
            txt_dbs = [
                "/db/itm_flickr30k_train_base-cased.db",
                "/db/itm_flickr30k_val_base-cased.db",
                "/db/itm_flickr30k_test_base-cased.db",
            ]
            img_dbs = [
                "/img/flickr30k/",
                "/img/flickr30k/",
                "/img/flickr30k/",
            ]
        else:
            txt_dbs, img_dbs = '/db/itm_flickr30k_test_base-cased.db', '/img/flickr30k/'
    else:
        cmd = '--config ./config/coco_eval_config.json '\
              '--biencoder_checkpoint  /good_models/coco_two-stream-add/biencoder.last.pt ' \
              '--teacher_checkpoint /pretrain/uniter_teacher_coco.pt ' \
              '--img_meta /db/meta/coco_meta.json'
        if Full:
            txt_dbs = [
                "/db/itm_coco_train_base-cased.db",
                "/db/itm_coco_restval_base-cased.db",
                "/db/itm_coco_val_base-cased.db",
                "/db/itm_coco_test_base-cased.db"
            ]
            img_dbs = [
                "/img/coco_train2014/",
                "/img/coco_val2014",
                "/img/coco_val2014/",
                "/img/coco_val2014/"
            ]

        else:
            txt_dbs, img_dbs = '/db/itm_coco_test_base-cased.db', '/img/coco_val2014'
    return cmd, txt_dbs, img_dbs


def train_parser(parser):
    default_params(parser)
    add_itm_params(parser)
    add_logging_params(parser)
    parser.add_argument('--teacher_checkpoint', default=None, type=str, help="")
    return parser


def init_model(cmd):
    parser = argparse.ArgumentParser()
    parser = train_parser(parser)
    args = parse_with_config(parser, cmd.split())

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
    args.vector_size = 768
    args.tokenizer = BertTokenizer.from_pretrained(args.txt_model_config)
    print_args(args)

    with open(args.itm_global_file) as f:
        args.img_meta = json.load(f)

    # Init Model
    bi_encoder = BiEncoder(args, args.fix_img_encoder, args.fix_txt_encoder, project_dim=args.project_dim)
    load_biencoder_checkpoint(bi_encoder, args.biencoder_checkpoint)

    img_model, txt_model = bi_encoder.img_model, bi_encoder.txt_model
    img_model.to(args.device)
    txt_model.to(args.device)

    img_model, _ = setup_for_distributed_mode(img_model, None, args.device, args.n_gpu, -1, args.fp16, args.fp16_opt_level)
    img_model.eval()

    txt_model, _ = setup_for_distributed_mode(txt_model, None, args.device, args.n_gpu, -1, args.fp16, args.fp16_opt_level)
    txt_model.eval()
    return bi_encoder, args


def load_embedding(bi_encoder, EMBEDDED_FILE, args, txt_dbs, img_dbs, Full):
    if not os.path.isfile(EMBEDDED_FILE):
        # Load Data
        print('embedded file', EMBEDDED_FILE, 'not exist, creating one...')
        FILE_MAPPER = {
            'train': [args.train_txt_dbs, args.train_img_dbs, True],
            'dev': [args.val_txt_db, args.val_img_db, False],
            'test': [args.test_txt_db, args.test_img_db, False]
        }
        all_img_dbs = ImageLmdbGroup(args.conf_th, args.max_bb, args.min_bb, args.num_bb, args.compressed_db)

        if Full:
            dataset = load_dataset(all_img_dbs, txt_dbs, img_dbs, args, True)
            for d in dataset.datasets:
                d.new_epoch()
            dataloader = build_dataloader(dataset, itm_fast_collate, False, args, batch_size=512)
            img2txt = dict(ChainMap(*[json.load(open(os.path.join(db_folder, 'img2txts.json'))) for db_folder in txt_dbs]))
        else:
            dataset = load_dataset(all_img_dbs, txt_dbs, img_dbs, args, is_train=False)
            dataset.new_epoch()
            dataloader = build_dataloader(dataset, itm_fast_collate, False, args, batch_size=512)
            img2txt = dict(ChainMap(*[json.load(open(os.path.join(db_folder, 'img2txts.json'))) for db_folder in [txt_db]]))

        print(f'dataset len = {len(dataset)}, dataloader len = {len(dataloader)}')

        img_embedding = dict()
        caption_embedding = dict()
        labels_img_name = []
        embeds = get_model_encoded_vecs(bi_encoder, dataloader)
        with open(EMBEDDED_FILE, 'wb') as f:
            pickle.dump(embeds, f)
    else:
        print('embedded file found, loading...')
        with open(EMBEDDED_FILE, 'rb') as f:
            embeds = pickle.load(f)

    return embeds
