"""
MLM datasets
"""
import math
import random

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from toolz.sandbox import unzip

from .data import (DetectFeatTxtTokDataset, TxtTokLmdb,
                   get_ids_and_lens, pad_tensors, get_gather_index)


def random_word(tokens, vocab_range, mask):
    """
    Masking some random tokens for Language Model task with probabilities as in
        the original BERT paper.
    :param tokens: list of int, tokenized sentence.
    :param vocab_range: for choosing a random word
    :return: (list of int, list of int), masked tokens and related labels for
        LM prediction
    """
    output_label = []

    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[i] = mask

            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[i] = random.choice(list(range(*vocab_range)))

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later)
            output_label.append(token)
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)
    if all(o == -1 for o in output_label):
        # at least mask 1
        output_label[0] = tokens[0]
        tokens[0] = mask

    return tokens, output_label


class MlmDataset(DetectFeatTxtTokDataset):
    def __init__(self, txt_db, img_db):
        assert isinstance(txt_db, TxtTokLmdb)
        super().__init__(txt_db, img_db)

    def __getitem__(self, i):
        """
        Return:
        - input_ids    : (L, ), i.e., [cls, wd, wd, ..., sep, 0, 0], 0s padded
        - img_feat     : (num_bb, d)
        - img_pos_feat : (num_bb, 7)
        - attn_masks   : (L + num_bb, ), ie., [1, 1, ..., 0, 0, 1, 1]
        - txt_labels   : (L, ), [-1, -1, wid, -1, -1, -1]
        0's padded so that (L + num_bb) % 8 == 0
        """
        example = super().__getitem__(i)

        # text input
        input_ids, txt_labels = self.create_mlm_io(example['input_ids'])

        # img input
        img_feat, img_pos_feat, num_bb = self._get_img_feat(
            example['img_fname'])

        attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)

        return input_ids, img_feat, img_pos_feat, attn_masks, txt_labels

    def create_mlm_io(self, input_ids):
        input_ids, txt_labels = random_word(input_ids,
                                            self.txt_db.v_range,
                                            self.txt_db.mask)
        input_ids = torch.tensor([self.txt_db.cls_]
                                 + input_ids
                                 + [self.txt_db.sep])
        txt_labels = torch.tensor([-1] + txt_labels + [-1])
        return input_ids, txt_labels


def mlm_collate(inputs):
    """
    Return:
    :input_ids    (n, max_L) padded with 0
    :position_ids (n, max_L) padded with 0
    :txt_lens     list of [txt_len]
    :img_feat     (n, max_num_bb, feat_dim)
    :img_pos_feat (n, max_num_bb, 7)
    :num_bbs      list of [num_bb]
    :attn_masks   (n, max_{L + num_bb}) padded with 0
    :txt_labels   (n, max_L) padded with -1
    """
    (input_ids, img_feats, img_pos_feats, attn_masks, txt_labels
     ) = map(list, unzip(inputs))

    # text batches
    txt_lens = [i.size(0) for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    txt_labels = pad_sequence(txt_labels, batch_first=True, padding_value=-1)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)

    # image batches
    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)

    bs, max_tl = input_ids.size()
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_pos_feat': img_pos_feat,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'txt_labels': txt_labels}
    return batch


class BlindMlmDataset(Dataset):
    def __init__(self, txt_db):
        assert isinstance(txt_db, TxtTokLmdb)
        self.txt_db = txt_db
        self.lens, self.ids = get_ids_and_lens(txt_db)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        id_ = self.ids[i]
        example = self.txt_db[id_]
        input_ids, txt_labels = self.create_mlm_io(example['input_ids'])
        attn_masks = torch.ones(len(input_ids), dtype=torch.long)

        return input_ids, attn_masks, txt_labels


def mlm_blind_collate(inputs):
    input_ids, attn_masks, txt_labels = map(list, unzip(inputs))

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)
    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
    txt_labels = pad_sequence(txt_labels, batch_first=True, padding_value=-1)

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'attn_masks': attn_masks,
             'txt_labels': txt_labels}
    return batch


def eval_mask(len_, num_samples=7):
    """ build the mask for evaluating MLM
    circularly mask 1 word out of every x words
    """
    # build the random masks
    if len_ <= num_samples:
        masks = torch.eye(len_).bool()
        num_samples = len_
    else:
        mask_inds = [list(range(i, len_, num_samples))
                     for i in range(num_samples)]
        masks = torch.zeros(num_samples, len_).bool()
        for i, indices in enumerate(mask_inds):
            for j in indices:
                masks.data[i, j] = 1
    assert (masks.sum(dim=0) != torch.ones(len_).long()).sum().item() == 0
    assert masks.sum().item() == len_
    return masks


def eval_gather_inds(len_, num_samples=7):
    """ get the gather indices """
    inds = torch.arange(0, num_samples, dtype=torch.long)
    mul = math.ceil(len_ / num_samples)
    output = inds.repeat(mul)[:len_]
    return output


def stack_pad_tensors(tensors, lens=None, ns=None, pad=0):
    """N x [B_i, T, ...]"""
    if ns is None:
        ns = [t.size(0) for t in tensors]
    if lens is None:
        lens = [t.size(1) for t in tensors]
    max_len = max(lens)
    bs = sum(ns)
    hid_dims = tensors[0].size()[2:]
    dtype = tensors[0].dtype
    output = torch.zeros(bs, max_len, *hid_dims, dtype=dtype)
    if pad:
        output.data.fill_(pad)
    i = 0
    for t, l, n in zip(tensors, lens, ns):
        output.data[i:i+n, :l, ...] = t.data
        i += n
    return output


def expand_tensors(tensors, ns):
    return [t.unsqueeze(0).expand(n, *tuple([-1]*t.dim()))
            for t, n in zip(tensors, ns)]


class MlmEvalDataset(DetectFeatTxtTokDataset):
    """ For evaluating MLM training task """
    def __init__(self, txt_db, img_db):
        assert isinstance(txt_db, TxtTokLmdb)
        super().__init__(txt_db, img_db)

    def __getitem__(self, i):
        example = super().__getitem__(i)

        # text input
        (input_ids, txt_labels, gather_inds
         ) = self.create_mlm_eval_io(example['input_ids'])

        # img input
        img_feat, img_pos_feat, num_bb = self._get_img_feat(
            example['img_fname'])

        attn_masks = torch.ones(input_ids.size(1) + num_bb, dtype=torch.long)

        return (input_ids, img_feat, img_pos_feat, attn_masks,
                txt_labels, gather_inds)

    def create_mlm_eval_io(self, input_ids):
        txt_labels = torch.tensor(input_ids)
        masks = eval_mask(len(input_ids))
        n_mask = masks.size(0)
        masks = torch.cat([torch.zeros(n_mask, 1).bool(),
                           masks,
                           torch.zeros(n_mask, 1).bool()],
                          dim=1)
        input_ids = torch.tensor([[self.txt_db.cls_]
                                  + input_ids
                                  + [self.txt_db.sep]
                                  for _ in range(n_mask)])
        input_ids.data.masked_fill_(masks, self.txt_db.mask)
        gather_inds = eval_gather_inds(len(txt_labels))
        return input_ids, txt_labels, gather_inds


def _batch_gather_tgt(gather_inds, n_masks):
    gather_tgts = []
    offset = 0
    for g, n in zip(gather_inds, n_masks):
        gather_tgts.append(g + offset)
        offset += n
    gather_tgt = pad_sequence(gather_tgts, batch_first=True, padding_value=0)
    return gather_tgt


def mlm_eval_collate(inputs):
    (input_ids, img_feats, img_pos_feats, attn_masks, txt_labels, gather_inds
     ) = map(list, unzip(inputs))

    # sizes
    n_masks, txt_lens = map(list, unzip(i.size() for i in input_ids))

    # text batches
    input_ids = stack_pad_tensors(input_ids, txt_lens, n_masks)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)
    txt_labels = pad_sequence(txt_labels, batch_first=True, padding_value=-1)
    gather_tgt = _batch_gather_tgt(gather_inds, n_masks)

    # image batches
    num_bbs = [f.size(0) for f in img_feats]
    img_feat = stack_pad_tensors(expand_tensors(img_feats, n_masks),
                                 num_bbs, n_masks)
    img_pos_feat = stack_pad_tensors(expand_tensors(img_pos_feats, n_masks),
                                     num_bbs, n_masks)

    bs, max_tl = input_ids.size()
    attn_masks = stack_pad_tensors(expand_tensors(attn_masks, n_masks),
                                   None, n_masks)
    out_size = attn_masks.size(1)
    # repeat txt_lens, num_bbs
    txt_lens = [l for l, n in zip(txt_lens, n_masks) for _ in range(n)]
    num_bbs = [b for b, n in zip(num_bbs, n_masks) for _ in range(n)]
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_pos_feat': img_pos_feat,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'gather_tgt': gather_tgt,
             'txt_labels': txt_labels}
    return batch


class BlindMlmEvalDataset(Dataset):
    def __init__(self, txt_db):
        assert isinstance(txt_db, TxtTokLmdb)
        self.txt_db = txt_db
        self.lens, self.ids = get_ids_and_lens(txt_db)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        id_ = self.ids[i]
        example = self.txt_db[id_]
        input_ids = example['input_ids']

        # text input
        input_ids = example['input_ids']
        (input_ids, txt_labels, gather_inds
         ) = self.txt_db.create_mlm_eval_io(input_ids)

        attn_masks = torch.ones(len(input_ids), dtype=torch.long)

        return input_ids, attn_masks, txt_labels, gather_inds


def mlm_blind_eval_collate(inputs):
    (input_ids, position_ids, attn_masks, txt_labels, gather_inds
     ) = map(list, unzip(inputs))

    # sizes
    n_masks, txt_lens = map(list, unzip(i.size() for i in input_ids))

    # text batches
    input_ids = stack_pad_tensors(input_ids, txt_lens, n_masks)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)
    attn_masks = stack_pad_tensors(expand_tensors(attn_masks, n_masks),
                                   None, n_masks)
    txt_labels = pad_sequence(txt_labels, batch_first=True, padding_value=-1)
    gather_tgt = _batch_gather_tgt(gather_inds, n_masks)

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'attn_masks': attn_masks,
             'gather_tgt': gather_tgt,
             'txt_labels': txt_labels}
    return batch
