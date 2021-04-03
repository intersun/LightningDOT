###########################################################################################################################
# Re-Ranker
###########################################################################################################################
# Will also do speed test in this script

import pickle
import logging
import argparse
import os
import torch
import json
import collections
import time
import tqdm
import itertools

import numpy as np

from horovod import torch as hvd
from transformers.tokenization_bert import BertTokenizer

from uniter_model.model.itm import UniterForImageTextRetrieval
from uniter_model.data import ImageLmdbGroup
from uniter_model.data.loader import move_to_cuda
from dvl.options import default_params, add_itm_params, parse_with_config, add_logging_params, print_args, add_kd_params
from dvl.models.bi_encoder import BiEncoder, setup_for_distributed_mode, load_biencoder_checkpoint
from dvl.utils import num_of_parameters
from dvl.trainer import load_dataset, eval_model_on_dataloader, build_dataloader, get_indexer
from dvl.data.itm import itm_fast_collate
from dvl.const import IMG_DIM
from dvl.data.itm import pad_tensors


SEARCH_MODE = 'approx'

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
add_kd_params(parser)

data_name, Full = 'coco', False

if data_name == 'flickr':
    cmd = '--config ./config/flickr30k_eval_config.json --biencoder_checkpoint  /good_models/flickr_two-stream-add/biencoder.last.pt ' \
           '--teacher_checkpoint /pretrain/uniter_teacher_flickr.pt'
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
        txt_db, img_db = '/db/itm_flickr30k_test_base-cased.db', '/img/flickr30k/'
else:
    cmd = '--config ./config/coco_eval_config.json --biencoder_checkpoint  /good_models/coco_two-stream-add/biencoder.last.pt ' \
          '--teacher_checkpoint /pretrain/uniter_teacher_coco.pt'
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
        txt_db, img_db = '/db/itm_coco_test_base-cased.db', '/img/coco_val2014'


args = parse_with_config(parser, cmd.split())

args.tokenizer = BertTokenizer.from_pretrained(args.txt_model_config)
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
args.vector_size = 768
args.img_meta = None
args.num_tops = 100
args.valid_batch_size = 256
torch.cuda.set_device(hvd.local_rank())
print_args(args)


bi_encoder = BiEncoder(args, args.fix_img_encoder, args.fix_txt_encoder, project_dim=args.project_dim)
load_biencoder_checkpoint(bi_encoder, args.biencoder_checkpoint)

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

if Full:
    dataset = load_dataset(all_img_dbs, txt_dbs, img_dbs, args, True)
    for d in dataset.datasets:
        d.new_epoch()
    dataloader = build_dataloader(dataset, itm_fast_collate, False, args)
    img2txt = dict(collections.ChainMap(*[json.load(open(os.path.join(db_folder, 'img2txts.json'))) for db_folder in txt_dbs]))
else:
    dataset = load_dataset(all_img_dbs, txt_db, img_db, args, is_train=False)
    dataset.new_epoch()
    dataloader = build_dataloader(dataset, itm_fast_collate, False, args)
    img2txt = dict(collections.ChainMap(*[json.load(open(os.path.join(db_folder, 'img2txts.json'))) for db_folder in [txt_db]]))

logger.info(f'dataset len = {len(dataset)}, dataloader len = {len(dataloader)}')
txt2img = dict(itertools.chain(*[[(v, k) for v in vals] for k, vals in img2txt.items()]))
# build indexer, cache data
# args.hnsw_index = True
_, _, (indexer_img, indexer_txt), (recall_img, recall_txt), _ = eval_model_on_dataloader(bi_encoder, dataloader, args,
                                                                                         img2txt=img2txt, no_eval=True)

###########################################################################################################################
# Re-Ranker
###########################################################################################################################
txt_db_test, img_db_test = args.test_txt_db, args.test_img_db
dataset_test = load_dataset(all_img_dbs, txt_db_test, img_db_test, args, is_train=False)
dataset_test.new_epoch()
dataloader_test = build_dataloader(dataset_test, itm_fast_collate, False, args, batch_size=400)

feats_dict = {'imgs':  collections.defaultdict(dict), 'txts': collections.defaultdict(dict)}
recall_img2 = {1: 0, 5: 0, 10: 0, 20: 0, 50: 0, 100: 0}
recall_txt2 = {1: 0, 5: 0, 10: 0, 20: 0, 50: 0, 100: 0}
error_msg = []
total_len = 0
ranking_res_img = dict()
ranking_res_txt = dict()
sanity = 0
for i, batch in enumerate(tqdm.tqdm(dataloader_test)):
    batch['txts_fname'] = batch['txt_index']
    batch['imgs_fname'] = batch['img_fname']

    # record all features
    for sets in ['imgs', 'txts']:
        for k in batch[sets]:
            for idx, img_name in enumerate(batch[f'{sets}_fname']):
                try:
                    feats_dict[sets][img_name][k] = batch[sets][k][idx]
                except IndexError:
                    assert len(batch[sets][k]) == 1, 'should be same for whole batch'
                    feats_dict[sets][img_name][k] = batch[sets][k][0]
                except TypeError:
                    assert batch[sets][k] is None, f'feat for {k} is None'
                    feats_dict[sets][img_name][k] = None

    # batch.pop('imgs', None)
    batch.pop('caps', None)

    # recrod time for inference and search only
    with torch.no_grad():
        txt_vec, img_vec, caps_vec = bi_encoder(batch)
    res_img = [i[0] for i in indexer_img.search_knn(txt_vec.detach().cpu().numpy(), max(recall_img2.keys()))]
    res_txt = [i[0] for i in indexer_txt.search_knn(img_vec.detach().cpu().numpy(), max(recall_txt2.keys()))]

    total_len += len(res_img)

    for r, txt_index in zip(res_img, batch['txt_index']):
        ranking_res_img[txt_index] = r
        for top in recall_img2:
            recall_img2[top] += txt2img[txt_index] in r[:top]

    for r, img_index in zip(res_txt, batch['img_fname']):
        ranking_res_txt[img_index] = r
        for top in recall_txt2:
            recall_txt2[top] += any([txt_id in r[:top] for txt_id in img2txt[img_index]])

print('img retrieval results')
for top in recall_img2:
    print(f'R@{top} =', recall_img2[top] / total_len, end="\t")
print()

print('txt retrieval results')
for top in recall_txt2:
    print(f'R@{top} =', recall_txt2[top] / total_len, end="\t")
print()


if Full:
    sufix = '_large'
    # sufix = ''
    with open(f'/pretrain/{data_name}_all{sufix}/ir.bin', 'rb') as f:
        scores_ir = pickle.load(f)
        txt_ids = list(scores_ir.keys())
    with open(f'/pretrain/{data_name}_all{sufix}/tr.bin', 'rb') as f:
        scores_tr = pickle.load(f)
        img_ids = list(scores_tr.keys())
else:
    with open('/pretrain/itm_flickr_large_result/results.bin', 'rb') as f:
        scores_tuple = pickle.load(f)

    # score_mat : num_txt * num_img
    txt_ids, img_ids, scores_mat = scores_tuple[1], scores_tuple[2], scores_tuple[0]
    txt_ids_mapping = {t:i for i, t in enumerate(txt_ids)}
    imgs_ids_mapping ={t:i for i, t in enumerate(img_ids)}


# Re-ranker for OSCAR
if False:
    with open('/pretrain/oscar_score.pkl', 'rb') as f:
        scores_tuple = pickle.load(f)
        # scores are img_id x txt_id

    img_ids = list(scores_tuple.keys())
    diff_ids = list(set(indexer_txt.index_id_to_db_id) - set(list(scores_tuple[img_ids[0]].keys())))
    txt_ids =  list(set(indexer_txt.index_id_to_db_id))
    txt_ids_mapping, imgs_ids_mapping = {t:i for i, t in enumerate(txt_ids)}, {img:i for i, img in enumerate(img_ids)}
    scores_mat = np.zeros((len(txt_ids), len(img_ids)))
    for img_id in img_ids:
        for txt_id in txt_ids:
            if txt_id in diff_ids:
                scores_mat[ txt_ids_mapping[txt_id] ][ imgs_ids_mapping[img_id] ] = 0.0
            else:
                scores_mat[ txt_ids_mapping[txt_id] ][ imgs_ids_mapping[img_id] ] = scores_tuple[img_id][txt_id]



# for image retrieval
for threshold in [10, 20, 50, 100]:
    recall_rerank = {1: 0, 5: 0, 10: 0}
    for txt_id in txt_ids:
        if Full:
            scores = torch.Tensor([scores_ir[txt_id].get(img_id, -1000) for img_id in ranking_res_img[txt_id][:threshold]])
        else:
            scores = torch.Tensor([scores_mat[txt_ids_mapping[txt_id]][imgs_ids_mapping[img_id]] for img_id in ranking_res_img[txt_id][:threshold]])
        idx = scores.topk(10, 0)
        uniter_ids = [ranking_res_img[txt_id][i.item()] for i in idx[1]]
        for top in recall_rerank:
            recall_rerank[top] += txt2img[txt_id] in uniter_ids[:top]

    for top in recall_rerank:
        print(threshold, f'R@{top} =', recall_rerank[top] / total_len, end="\t")
    print()

# for text retrieval
for threshold in [10, 20, 50, 100]:
    recall_rerank = {1: 0, 5: 0, 10: 0}
    for img_id in img_ids:
        if Full:
            scores = torch.Tensor([scores_tr[img_id].get(txt_id, -1000) for txt_id in ranking_res_txt[img_id][:threshold]])
        else:
            scores = torch.Tensor([scores_mat[txt_ids_mapping[txt_id]][imgs_ids_mapping[img_id]] for txt_id in ranking_res_txt[img_id][:threshold]])
        idx = scores.topk(10, 0)
        uniter_ids = [ranking_res_txt[img_id][i.item()] for i in idx[1]]
        for top in recall_rerank:

            recall_rerank[top] += any([txt_id in uniter_ids[:top] for txt_id in img2txt[img_id]])
            # recall_rerank[top] += txt2img[txt_id] in uniter_ids[:top]

    for top in recall_rerank:
        print(threshold, f'R@{top} =', recall_rerank[top] / len(img_ids), end="\t")
    print()


