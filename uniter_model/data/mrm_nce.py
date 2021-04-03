"""
MRM Datasets (contrastive learning version)
"""
import torch
from torch.nn.utils.rnn import pad_sequence
from toolz.sandbox import unzip
from cytoolz import curry

from .data import (DetectFeatLmdb, DetectFeatTxtTokDataset,
                   pad_tensors, get_gather_index)
from .mrm import _get_img_mask, _get_img_tgt_mask, _get_feat_target
from .itm import sample_negative


# FIXME diff implementation from mrfr, mrc
def _mask_img_feat(img_feat, img_masks, neg_feats,
                   noop_prob=0.1, change_prob=0.1):
    rand = torch.rand(*img_masks.size())
    noop_mask = rand < noop_prob
    change_mask = ~noop_mask & (rand < (noop_prob+change_prob)) & img_masks
    img_masks_in = img_masks & ~noop_mask & ~change_mask

    img_masks_ext = img_masks_in.unsqueeze(-1).expand_as(img_feat)
    img_feat_masked = img_feat.data.masked_fill(img_masks_ext, 0)

    n_neg = change_mask.sum().item()
    feat_dim = neg_feats.size(-1)
    index = torch.arange(0, change_mask.numel(), dtype=torch.long
                         ).masked_select(change_mask.view(-1))
    index = index.unsqueeze(-1).expand(-1, feat_dim)
    img_feat_out = img_feat_masked.view(-1, feat_dim).scatter(
        dim=0, index=index, src=neg_feats[:n_neg]).view(*img_feat.size())

    return img_feat_out, img_masks_in


class MrmNceDataset(DetectFeatTxtTokDataset):
    def __init__(self, mask_prob, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_prob = mask_prob

    def __getitem__(self, i):
        example = super().__getitem__(i)
        # text input
        input_ids = example['input_ids']
        input_ids = self.txt_db.combine_inputs(input_ids)

        # image input features
        img_feat, img_pos_feat, num_bb = self._get_img_feat(
            example['img_fname'])
        img_mask = _get_img_mask(self.mask_prob, num_bb)
        img_mask_tgt = _get_img_tgt_mask(img_mask, len(input_ids))

        attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)

        return (input_ids, img_feat, img_pos_feat,
                attn_masks, img_mask, img_mask_tgt,
                example['img_fname'])


class NegativeImageSampler(object):
    def __init__(self, img_dbs, neg_size, size_mul=8):
        if not isinstance(img_dbs, list):
            assert isinstance(img_dbs, DetectFeatLmdb)
            img_dbs = [img_dbs]
        self.neg_size = neg_size
        self.img_db = JoinedDetectFeatLmdb(img_dbs)
        all_imgs = []
        for db in img_dbs:
            all_imgs.extend(db.name2nbb.keys())
        self.all_imgs = all_imgs

    def sample_negative_feats(self, pos_imgs):
        neg_img_ids = sample_negative(self.all_imgs, pos_imgs, self.neg_size)
        all_neg_feats = torch.cat([self.img_db[img][0] for img in neg_img_ids],
                                  dim=0)
        # only use multiples of 8 for tensorcores
        n_cut = all_neg_feats.size(0) % 8
        if n_cut != 0:
            return all_neg_feats[:-n_cut]
        else:
            return all_neg_feats


class JoinedDetectFeatLmdb(object):
    def __init__(self, img_dbs):
        assert all(isinstance(db, DetectFeatLmdb) for db in img_dbs)
        self.img_dbs = img_dbs

    def __getitem__(self, file_name):
        for db in self.img_dbs:
            if file_name in db:
                return db[file_name]
        raise ValueError("image does not exists")


@curry
def mrm_nce_collate(neg_sampler, inputs):
    (input_ids, img_feats, img_pos_feats, attn_masks, img_masks, img_mask_tgts,
     positive_imgs) = map(list, unzip(inputs))

    txt_lens = [i.size(0) for i in input_ids]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)

    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)
    neg_feats = neg_sampler.sample_negative_feats(positive_imgs)

    # mask features
    img_masks = pad_sequence(img_masks, batch_first=True, padding_value=0)
    feat_targets = _get_feat_target(img_feat, img_masks)
    img_feat, img_masks_in = _mask_img_feat(img_feat, img_masks, neg_feats)
    img_mask_tgt = pad_sequence(img_mask_tgts,
                                batch_first=True, padding_value=0)

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
             'feat_targets': feat_targets,
             'img_masks': img_masks,
             'img_masks_in': img_masks_in,
             'img_mask_tgt': img_mask_tgt,
             'neg_feats': neg_feats}
    return batch
