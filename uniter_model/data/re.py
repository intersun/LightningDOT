"""
Referring Expression Comprehension dataset
"""
import sys
import json
import random
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from toolz.sandbox import unzip

from .data import TxtLmdb


class ReImageFeatDir(object):
    def __init__(self, img_dir):
        self.img_dir = img_dir

    def __getitem__(self, file_name):
        img_dump = np.load(f'{self.img_dir}/{file_name}', allow_pickle=True)
        img_feat = torch.tensor(img_dump['features'])
        img_bb = torch.tensor(img_dump['norm_bb'])
        return img_feat, img_bb


class ReDetectFeatDir(object):
    def __init__(self, img_dir, conf_th=0.2, max_bb=100, min_bb=10, num_bb=36,
                 format_='npz'):
        assert format_ == 'npz', 'only support npz for now.'
        assert isinstance(img_dir, str), 'img_dir is path, not db.'
        self.img_dir = img_dir
        self.conf_th = conf_th
        self.max_bb = max_bb
        self.min_bb = min_bb
        self.num_bb = num_bb

    def _compute_num_bb(self, img_dump):
        num_bb = max(self.min_bb, (img_dump['conf'] > self.conf_th).sum())
        num_bb = min(self.max_bb, num_bb)
        return num_bb

    def __getitem__(self, file_name):
        # image input features
        img_dump = np.load(f'{self.img_dir}/{file_name}', allow_pickle=True)
        num_bb = self._compute_num_bb(img_dump)
        img_feat = torch.tensor(img_dump['features'][:num_bb, :])
        img_bb = torch.tensor(img_dump['norm_bb'][:num_bb, :])
        return img_feat, img_bb


class ReferringExpressionDataset(Dataset):
    def __init__(self, db_dir, img_dir, max_txt_len=60):
        assert isinstance(img_dir, ReImageFeatDir) or \
               isinstance(img_dir, ReDetectFeatDir)
        self.img_dir = img_dir

        # load refs = [{ref_id, sent_ids, ann_id, image_id, sentences, split}]
        refs = json.load(open(f'{db_dir}/refs.json', 'r'))
        self.ref_ids = [ref['ref_id'] for ref in refs]
        self.Refs = {ref['ref_id']: ref for ref in refs}

        # load annotations = [{id, area, bbox, image_id, category_id}]
        anns = json.load(open(f'{db_dir}/annotations.json', 'r'))
        self.Anns = {ann['id']: ann for ann in anns}

        # load categories = [{id, name, supercategory}]
        categories = json.load(open(f'{db_dir}/categories.json', 'r'))
        self.Cats = {cat['id']: cat['name'] for cat in categories}

        # load images = [{id, file_name, ann_ids, height, width}]
        images = json.load(open(f'{db_dir}/images.json', 'r'))
        self.Images = {img['id']: img for img in images}

        # id2len: sent_id -> sent_len
        id2len = json.load(open(f'{db_dir}/id2len.json', 'r'))
        self.id2len = {int(_id): _len for _id, _len in id2len.items()}
        self.max_txt_len = max_txt_len
        self.sent_ids = self._get_sent_ids()

        # db[str(sent_id)] =
        # {sent_id, sent, ref_id, ann_id, image_id,
        #  bbox, input_ids, toked_sent}
        self.db = TxtLmdb(db_dir, readonly=True)

        # meta
        meta = json.load(open(f'{db_dir}/meta.json', 'r'))
        self.cls_ = meta['CLS']
        self.sep = meta['SEP']
        self.mask = meta['MASK']
        self.v_range = meta['v_range']

    def shuffle(self):
        # we shuffle ref_ids and make sent_ids according to ref_ids
        random.shuffle(self.ref_ids)
        self.sent_ids = self._get_sent_ids()

    def _get_sent_ids(self):
        sent_ids = []
        for ref_id in self.ref_ids:
            for sent_id in self.Refs[ref_id]['sent_ids']:
                sent_len = self.id2len[sent_id]
                if self.max_txt_len == -1 or sent_len < self.max_txt_len:
                    sent_ids.append(sent_id)
        return sent_ids

    def _get_img_feat(self, fname):
        img_feat, bb = self.img_dir[fname]
        img_bb = torch.cat([bb, bb[:, 4:5]*bb[:, 5:]], dim=-1)
        num_bb = img_feat.size(0)
        return img_feat, img_bb, num_bb

    def __len__(self):
        return len(self.sent_ids)

    def __getitem__(self, i):
        """
        Return:
        :input_ids     : (L, ), i.e., [cls, wd, wd, ..., sep, 0, 0]
        :position_ids  : range(L)
        :img_feat      : (num_bb, d)
        :img_pos_feat  : (num_bb, 7)
        :attn_masks    : (L+num_bb, ), i.e., [1, 1, ..., 0, 0, 1, 1]
        :obj_masks     : (num_bb, ) all 0's
        :target        : (1, )
        """
        # {sent_id, sent, ref_id, ann_id, image_id,
        #  bbox, input_ids, toked_sent}
        sent_id = self.sent_ids[i]
        txt_dump = self.db[str(sent_id)]
        image_id = txt_dump['image_id']
        fname = f'visual_grounding_coco_gt_{int(image_id):012}.npz'
        img_feat, img_pos_feat, num_bb = self._get_img_feat(fname)

        # text input
        input_ids = txt_dump['input_ids']
        input_ids = [self.cls_] + input_ids + [self.sep]
        attn_masks = [1] * len(input_ids)
        position_ids = list(range(len(input_ids)))
        attn_masks += [1] * num_bb

        input_ids = torch.tensor(input_ids)
        position_ids = torch.tensor(position_ids)
        attn_masks = torch.tensor(attn_masks)

        # target bbox
        img = self.Images[image_id]
        assert len(img['ann_ids']) == num_bb, \
            'Please use visual_grounding_coco_gt'
        target = img['ann_ids'].index(txt_dump['ann_id'])
        target = torch.tensor([target])

        # obj_masks, to be padded with 1, for masking out non-object prob.
        obj_masks = torch.tensor([0]*len(img['ann_ids'])).bool()

        return (input_ids, position_ids, img_feat, img_pos_feat, attn_masks,
                obj_masks, target)


def re_collate(inputs):
    """
    Return:
    :input_ids     : (n, max_L) padded with 0
    :position_ids  : (n, max_L) padded with 0
    :txt_lens      : list of [txt_len]
    :img_feat      : (n, max_num_bb, feat_dim)
    :img_pos_feat  : (n, max_num_bb, 7)
    :num_bbs       : list of [num_bb]
    :attn_masks    : (n, max_{L+num_bb}) padded with 0
    :obj_masks     : (n, max_num_bb) padded with 1
    :targets       : (n, )
    """
    (input_ids, position_ids, img_feats, img_pos_feats, attn_masks, obj_masks,
     targets) = map(list, unzip(inputs))

    txt_lens = [i.size(0) for i in input_ids]
    num_bbs = [f.size(0) for f in img_feats]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = pad_sequence(position_ids,
                                batch_first=True, padding_value=0)
    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
    targets = torch.cat(targets, dim=0)
    obj_masks = pad_sequence(obj_masks,
                             batch_first=True, padding_value=1).bool()

    batch_size = len(img_feats)
    num_bb = max(num_bbs)
    feat_dim = img_feats[0].size(1)
    pos_dim = img_pos_feats[0].size(1)
    img_feat = torch.zeros(batch_size, num_bb, feat_dim)
    img_pos_feat = torch.zeros(batch_size, num_bb, pos_dim)
    for i, (im, pos) in enumerate(zip(img_feats, img_pos_feats)):
        len_ = im.size(0)
        img_feat.data[i, :len_, :] = im.data
        img_pos_feat.data[i, :len_, :] = pos.data

    return (input_ids, position_ids, txt_lens,
            img_feat, img_pos_feat, num_bbs,
            attn_masks, obj_masks, targets)


class ReferringExpressionEvalDataset(ReferringExpressionDataset):
    def __getitem__(self, i):
        """
        Return:
        :input_ids     : (L, ), i.e., [cls, wd, wd, ..., sep, 0, 0]
        :position_ids  : range(L)
        :img_feat      : (num_bb, d)
        :img_pos_feat  : (num_bb, 7)
        :attn_masks    : (L+num_bb, ), i.e., [1, 1, ..., 0, 0, 1, 1]
        :obj_masks     : (num_bb, ) all 0's
        :tgt_box       : ndarray (4, ) xywh
        :obj_boxes     : ndarray (num_bb, 4) xywh
        :sent_id
        """
        # {sent_id, sent, ref_id, ann_id, image_id,
        #  bbox, input_ids, toked_sent}
        sent_id = self.sent_ids[i]
        txt_dump = self.db[str(sent_id)]
        image_id = txt_dump['image_id']
        if isinstance(self.img_dir, ReImageFeatDir):
            if '_gt' in self.img_dir.img_dir:
                fname = f'visual_grounding_coco_gt_{int(image_id):012}.npz'
            elif '_det' in self.img_dir.img_dir:
                fname = f'visual_grounding_det_coco_{int(image_id):012}.npz'
        elif isinstance(self.img_dir, ReDetectFeatDir):
            fname = f'coco_train2014_{int(image_id):012}.npz'
        else:
            sys.exit('%s not supported.' % self.img_dir)
        img_feat, img_pos_feat, num_bb = self._get_img_feat(fname)

        # image info
        img = self.Images[image_id]
        im_width, im_height = img['width'], img['height']

        # object boxes, img_pos_feat (xyxywha) -> xywh
        obj_boxes = np.stack([img_pos_feat[:, 0]*im_width,
                              img_pos_feat[:, 1]*im_height,
                              img_pos_feat[:, 4]*im_width,
                              img_pos_feat[:, 5]*im_height], axis=1)
        obj_masks = torch.tensor([0]*num_bb).bool()

        # target box
        tgt_box = np.array(txt_dump['bbox'])  # xywh

        # text input
        input_ids = txt_dump['input_ids']
        input_ids = [self.cls_] + input_ids + [self.sep]
        attn_masks = [1] * len(input_ids)
        position_ids = list(range(len(input_ids)))
        attn_masks += [1] * num_bb

        input_ids = torch.tensor(input_ids)
        position_ids = torch.tensor(position_ids)
        attn_masks = torch.tensor(attn_masks)

        return (input_ids, position_ids, img_feat, img_pos_feat, attn_masks,
                obj_masks, tgt_box, obj_boxes, sent_id)

    # IoU function
    def computeIoU(self, box1, box2):
        # each box is of [x1, y1, w, h]
        inter_x1 = max(box1[0], box2[0])
        inter_y1 = max(box1[1], box2[1])
        inter_x2 = min(box1[0]+box1[2]-1, box2[0]+box2[2]-1)
        inter_y2 = min(box1[1]+box1[3]-1, box2[1]+box2[3]-1)

        if inter_x1 < inter_x2 and inter_y1 < inter_y2:
            inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
        else:
            inter = 0
        union = box1[2]*box1[3] + box2[2]*box2[3] - inter
        return float(inter)/union


def re_eval_collate(inputs):
    """
    Return:
    :input_ids     : (n, max_L)
    :position_ids  : (n, max_L)
    :txt_lens      : list of [txt_len]
    :img_feat      : (n, max_num_bb, d)
    :img_pos_feat  : (n, max_num_bb, 7)
    :num_bbs       : list of [num_bb]
    :attn_masks    : (n, max{L+num_bb})
    :obj_masks     : (n, max_num_bb)
    :tgt_box       : list of n [xywh]
    :obj_boxes     : list of n [[xywh, xywh, ...]]
    :sent_ids      : list of n [sent_id]
    """
    (input_ids, position_ids, img_feats, img_pos_feats, attn_masks, obj_masks,
     tgt_box, obj_boxes, sent_ids) = map(list, unzip(inputs))

    txt_lens = [i.size(0) for i in input_ids]
    num_bbs = [f.size(0) for f in img_feats]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = pad_sequence(position_ids,
                                batch_first=True, padding_value=0)
    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
    obj_masks = pad_sequence(obj_masks,
                             batch_first=True, padding_value=1).bool()

    batch_size = len(img_feats)
    num_bb = max(num_bbs)
    feat_dim = img_feats[0].size(1)
    pos_dim = img_pos_feats[0].size(1)
    img_feat = torch.zeros(batch_size, num_bb, feat_dim)
    img_pos_feat = torch.zeros(batch_size, num_bb, pos_dim)
    for i, (im, pos) in enumerate(zip(img_feats, img_pos_feats)):
        len_ = im.size(0)
        img_feat.data[i, :len_, :] = im.data
        img_pos_feat.data[i, :len_, :] = pos.data

    return (input_ids, position_ids, txt_lens,
            img_feat, img_pos_feat, num_bbs,
            attn_masks, obj_masks, tgt_box, obj_boxes, sent_ids)
