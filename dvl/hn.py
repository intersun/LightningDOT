import random
import logging
import collections
import json
import os
import itertools
import numpy as np
from collections import ChainMap


from dvl.trainer import build_dataloader, _save_checkpoint, eval_model_on_dataloader, load_dataset


logger = logging.getLogger()


def random_hard_neg(fname2id, num_hard_negatives, id2set, set2id):
    # num_hard_negatives must be very small
    hard_negs = dict()
    for i in fname2id:
        while True:
            hard_neg = random.choices(set2id[id2set[i]], k=num_hard_negatives)
            if fname2id[i] not in hard_neg:
                    break
        hard_negs[i] = hard_neg
    return hard_negs


def get_img_txt_mappings(train_txt_dbs):
    train_img2txt = dict(ChainMap(*[json.load(open(os.path.join(db_folder, 'img2txts.json'))) for db_folder in train_txt_dbs]))
    train_txt2img = dict(itertools.chain(*[[(v, k) for v in vals] for k, vals in train_img2txt.items()]))

    train_json = [json.load(open(os.path.join(db_folder, 'img2txts.json'))) for db_folder in train_txt_dbs]
    train_img2set = dict(ChainMap(*[{k:v for k in tj } for tj, v in zip(train_json, train_txt_dbs)]))
    train_txt2set = {txt_id: train_img2set[img_id] for txt_id, img_id in train_txt2img.items()}

    train_set2img, train_set2txt = collections.defaultdict(list), collections.defaultdict(list)
    for img_id, set_id in train_img2set.items():
        train_set2img[set_id].append(img_id)
        train_set2txt[set_id] += train_img2txt[img_id]

    return train_img2txt, train_txt2img, train_img2set, train_txt2set, train_set2img, train_set2txt


def sampled_hard_negatives(all_img_dbs, args, collate_func, bi_encoder, train_img2txt, train_txt2img):
    train_dataset_eval = load_dataset(all_img_dbs, args.train_txt_dbs, args.train_img_dbs, args, True)
    hard_negs_txt_all, hard_negs_img_all = [], []
    for dset in train_dataset_eval.datasets:
        dset.new_epoch()
        train_dataloader_hn = build_dataloader(dset, collate_func, True, args, args.valid_batch_size)
        logger.info(f'eval for train dataloader len (for hn) = {len(train_dataloader_hn)}')

        num_hard_sampled = min(max(args.num_hard_negatives * 2 + 10, 50), 1000)
        loss_hard, correct_ratio_hard, indexer_hard, recall_hard, (hard_neg_img, hard_neg_txt) = \
            eval_model_on_dataloader(bi_encoder, train_dataloader_hn, args, train_img2txt, num_hard_sampled)

        [v.remove(train_txt2img[k]) for k, v in hard_neg_img.items() if train_txt2img[k] in v]
        hard_neg_txt = {k: list(set(v)  - set(train_img2txt[k])) for k, v in hard_neg_txt.items()}


        # remove self in hard negatives as they are labels
        hard_negs_txt_all.append({k: random.sample(v, args.num_hard_negatives) for k, v in hard_neg_txt.items()})
        hard_negs_img_all.append({k: random.sample(v, args.num_hard_negatives) for k, v in hard_neg_img.items()})
    hard_negs_txt_all = dict(collections.ChainMap(*hard_negs_txt_all))
    hard_negs_img_all = dict(collections.ChainMap(*hard_negs_img_all))
    return hard_negs_txt_all, hard_negs_img_all
