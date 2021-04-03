import torch
import numpy as np
import itertools

from torch.nn.utils.rnn import pad_sequence
from uniter_model.data.itm import DetectFeatTxtTokDataset, TxtTokLmdb, DetectFeatLmdb, get_ids_and_lens
from uniter_model.data.data import get_gather_index
from toolz.sandbox import unzip
from cytoolz import concat
from GLOBAL_VARIABLES import N_EXAMPLES_TEACHER


def pad_tensors(tensors, lens=None, pad=0):
    """B x [T, ...]"""
    if lens is None:
        lens = [t.size(0) for t in tensors]
    max_len = max(lens)
    bs = len(tensors)
    hid = tensors[0].size(-1)
    dtype = tensors[0].dtype
    output = torch.zeros(bs, max_len, hid, dtype=dtype)
    if pad:
        output.data.fill_(pad)
    for i, (t, l) in enumerate(zip(tensors, lens)):
        output.data[i, :l, ...] = t.data
    return output


# for ITM task
class ItmFastDataset(DetectFeatTxtTokDataset):
    """ NOTE this Dataset handles distributed training itself
    (for more efficient negative sampling) """
    def __init__(self, txt_db, img_db, num_hard_negatives=0, img_meta=None, tokenizer=None):
        assert isinstance(txt_db, TxtTokLmdb)
        assert isinstance(img_db, DetectFeatLmdb)

        self.txt_db = txt_db
        self.img_db = img_db

        self.txt_lens, self.ids = get_ids_and_lens(txt_db)
        self.ids_2_idx = {idx:i for i, idx in enumerate(self.ids)}
        self.all_imgs = list(set(txt_db[id_]['img_fname'] for id_ in self.ids))

        self.num_hard_negatives = num_hard_negatives
        self.img_meta = img_meta
        self.tokenizer = tokenizer
        self.train_imgs = None
        self.neg_imgs = None
        # self.new_epoch(hard_negatives)

    def new_epoch(self, hard_negatives_img=None, hard_negatives_txt=None):
        """ should be called every epoch for more randomness"""
        self.lens = []
        self.train_imgs, self.neg_imgs = [], []
        self.train_txts, self.neg_txts = [], []
        for i, (id_, tl) in enumerate(zip(self.ids, self.txt_lens)):
            img_fname = super().__getitem__(i)['img_fname']
            self.train_imgs.append(img_fname)
            self.train_txts.append(id_)
            if hard_negatives_img is not None and self.num_hard_negatives > 0:
                self.neg_imgs.append(hard_negatives_img[id_][:self.num_hard_negatives])
                self.neg_txts.append(hard_negatives_txt[img_fname][:self.num_hard_negatives])
            else:
                self.neg_imgs.append(None)
                self.neg_txts.append(None)
            self.lens.append(tl + self.img_db.name2nbb[img_fname])

    def __getitem__(self, i):
        example = super().__getitem__(i)
        # labels and negative images should be sampled every epoch
        img_fname, hard_neg_imgs = self.train_imgs[i], self.neg_imgs[i]
        txt_fname, hard_neg_txts = self.ids[i], self.neg_txts[i]

        img_input_ids = torch.Tensor([101]).long()
        img_feat, img_pos_feat, num_bb = self._get_img_feat(img_fname)
        attn_masks_img = torch.ones(num_bb+1, dtype=torch.long)

        # text input
        input_ids = example['input_ids']
        input_ids = self.txt_db.combine_inputs(input_ids)
        attn_masks = torch.ones(len(input_ids), dtype=torch.long)

        if hard_neg_imgs is not None:
            # TODO: add hard negative here
            neg_imgs = dict({'img_input_ids': [], 'img_feat': [], 'img_pos_feat': [], 'num_bb': [], 'attn_masks_img': [],
                         'caption_ids': [], 'attn_masks_captions': []})
            for neg_id in hard_neg_imgs:
                neg_imgs['img_input_ids'].append(torch.Tensor([101]).long())
                t = self._get_img_feat(neg_id)
                neg_imgs['img_feat'].append(t[0])
                neg_imgs['img_pos_feat'].append(t[1])
                neg_imgs['num_bb'].append(t[2])
                neg_imgs['attn_masks_img'].append(torch.ones(t[2]+1, dtype=torch.long))
                if self.img_meta is not None:
                    tmp = [self.tokenizer.encode(i, add_special_tokens=False) + [self.tokenizer.sep_token_id]
                                   for i in self.img_meta[neg_id]['caption_multiple']]
                    neg_imgs['caption_ids'].append(torch.tensor([self.tokenizer.cls_token_id] + sum(tmp, []),
                                               dtype=input_ids.dtype, device=input_ids.device))
                    neg_imgs['attn_masks_captions'].append(torch.ones(len(neg_imgs['caption_ids'][-1]), dtype=torch.long))
                    # debug = [self.tokenizer.encode(a) for a in self.img_meta[img_fname]['annotation']]
            neg_txts = dict({'input_ids':[], 'position_ids':[], 'attention_mask':[]})
            for neg_id in hard_neg_txts:
                ei = super().__getitem__(self.ids_2_idx[neg_id])
                input_ids_ei = ei['input_ids']
                neg_txts['input_ids'].append(self.txt_db.combine_inputs(input_ids_ei))
                neg_txts['attention_mask'].append(torch.ones(len(neg_txts['input_ids'][-1]), dtype=torch.long))
        else:
            neg_imgs = None
            neg_txts = None

        if self.img_meta is not None:
            caption_ids = [self.tokenizer.encode(i, add_special_tokens=False) + [self.tokenizer.sep_token_id] for i in self.img_meta[img_fname]['caption_multiple']]
            caption_ids = torch.tensor([self.tokenizer.cls_token_id] + sum(caption_ids, []), dtype=input_ids.dtype, device=input_ids.device)
            attn_masks_captions = torch.ones(len(caption_ids), dtype=torch.long)
            # debug = [self.tokenizer.encode(a) for a in self.img_meta[img_fname]['annotation']]
        else:
            caption_ids = None
            attn_masks_captions = None

        # target = torch.Tensor(1).long()
        # target.data.fill_(ground_truth_label)
        return input_ids, img_feat, img_pos_feat, img_input_ids, attn_masks, attn_masks_img, self.ids[i], img_fname, neg_imgs, neg_txts, caption_ids, attn_masks_captions


def itm_fast_collate_kd(inputs):
    input_ids, img_feats, img_pos_feats, img_input_ids, attn_masks_text, attn_masks_img, idx, img_fname, negs, caption_ids, attn_masks_captions = map(list, unzip(inputs))

    # txt_lens = [i.size(0) for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    captions_ids = pad_sequence(caption_ids, batch_first=True, padding_value=0) if caption_ids[0] is not None else None

    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long).unsqueeze(0)
    position_ids_captions = torch.arange(0, captions_ids.size(1), dtype=torch.long).unsqueeze(0) if caption_ids[0] is not None else None

    if not None in negs:
        num_bbs_neg = list(itertools.chain(*[n['num_bb'] for n in negs]))
        img_feats_neg = list(itertools.chain(*[n['img_feat'] for n in negs]))
        img_input_ids_neg = list(itertools.chain(*[n['img_input_ids'] for n in negs]))
        img_pos_feat_neg = list(itertools.chain(*[n['img_pos_feat'] for n in negs]))
        attn_masks_img_neg = list(itertools.chain(*[n['attn_masks_img'] for n in negs]))
    else:
        num_bbs_neg = []
        img_feats_neg = []
        img_input_ids_neg = []
        img_pos_feat_neg = []
        attn_masks_img_neg = []

    num_bbs = [f.size(0) for f in img_feats] + num_bbs_neg
    img_feat = pad_tensors(img_feats+img_feats_neg, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats+img_pos_feat_neg, num_bbs)

    img_input_ids = pad_sequence(img_input_ids+img_input_ids_neg, batch_first=True, padding_value=0)
    img_position_ids = torch.arange(0, img_input_ids.size(1), dtype=torch.long).unsqueeze(0)

    attn_masks_text = pad_sequence(attn_masks_text, batch_first=True, padding_value=0)
    attn_masks_captions = pad_sequence(attn_masks_captions, batch_first=True, padding_value=0) if attn_masks_captions[0] is not None else None
    attn_masks_img = pad_sequence(attn_masks_img+attn_masks_img_neg, batch_first=True, padding_value=0)
    sample_size = len(inputs[0])
    assert all(sample_size == len(i) for i in inputs)

    bs, max_tl = input_ids.size()
    out_size = attn_masks_img.size(1)
    gather_index = get_gather_index([1]*bs, num_bbs, bs, 1, out_size)

    img_feat_teacher = img_feat[:N_EXAMPLES_TEACHER].repeat(bs, 1, 1)
    img_pos_feat_teacher = img_pos_feat[:N_EXAMPLES_TEACHER].repeat(bs, 1, 1)
    attn_masks_img_teacher = attn_masks_img[:N_EXAMPLES_TEACHER].repeat(bs, 1)[:, 1:]

    input_ids_teacher = input_ids.unsqueeze(1).repeat(1, 10, 1).view(bs*N_EXAMPLES_TEACHER, -1)
    position_ids_teacher = position_ids
    attn_masks_text_teacher = attn_masks_text.unsqueeze(1).repeat(1, 10, 1).view(bs*N_EXAMPLES_TEACHER, -1)

    attn_masks_teacher = torch.cat([attn_masks_text_teacher, attn_masks_img_teacher], dim=1)

    batch = {
        'txt_ids': input_ids,
        'img_ids': img_feat,
        'caption_ids': captions_ids,
        'txt_pos_ids': position_ids,
        'img_pos_ids': img_pos_feat,
        'caption_pos_ids': position_ids_captions,
        'txt_attn_masks': attn_masks_text,
        'img_attn_masks': attn_masks_img,
        'caption_attn_masks': attn_masks_captions,
        'img_txt_ids': img_input_ids,
        'img_txt_pos_ids': img_position_ids,
        'gather_index': gather_index,
        'sample_size': sample_size,
        'pos_ctx_indices': list(range(bs)),
        'neg_ctx_indices': list(range(bs, len(num_bbs))),
        'txt_index': idx,
        'img_fname': img_fname,

        'img_feat_teacher': img_feat_teacher,
        'img_pos_feat_teacher': img_pos_feat_teacher,
        'input_ids_teacher': input_ids_teacher,
        'position_ids_teacher': position_ids_teacher,
        'attn_masks_teacher': attn_masks_teacher
    }
    return batch


def itm_fast_collate(inputs):
    input_ids, img_feats, img_pos_feats, img_input_ids, attn_masks_text, attn_masks_img, idx, img_fname, neg_imgs, neg_txts, caption_ids, attn_masks_captions = map(list, unzip(inputs))
    bs = len(input_ids)
    # txt_lens = [i.size(0) for i in input_ids]

    if not None in neg_imgs:
        num_bbs_neg = list(itertools.chain(*[n['num_bb'] for n in neg_imgs]))
        img_feats_neg = list(itertools.chain(*[n['img_feat'] for n in neg_imgs]))
        img_input_ids_neg = list(itertools.chain(*[n['img_input_ids'] for n in neg_imgs]))
        img_pos_feat_neg = list(itertools.chain(*[n['img_pos_feat'] for n in neg_imgs]))
        attn_masks_img_neg = list(itertools.chain(*[n['attn_masks_img'] for n in neg_imgs]))
        caption_ids_neg = list(itertools.chain(*[n['caption_ids'] for n in neg_imgs]))
        attn_masks_captions_neg = list(itertools.chain(*[n['attn_masks_captions'] for n in neg_imgs]))

        input_ids_neg = list(itertools.chain(*[n['input_ids'] for n in neg_txts]))
        attn_masks_text_neg = list(itertools.chain(*[n['attention_mask'] for n in neg_txts]))
    else:
        num_bbs_neg = []
        img_feats_neg = []
        img_input_ids_neg = []
        img_pos_feat_neg = []
        attn_masks_img_neg = []
        caption_ids_neg = []
        attn_masks_captions_neg = []

        input_ids_neg = []
        attn_masks_text_neg = []

    input_ids = pad_sequence(input_ids+input_ids_neg, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long).unsqueeze(0)

    captions_ids = pad_sequence(caption_ids+caption_ids_neg, batch_first=True, padding_value=0) if caption_ids[0] is not None else None
    position_ids_captions = torch.arange(0, captions_ids.size(1), dtype=torch.long).unsqueeze(0) if caption_ids[0] is not None else None

    num_bbs = [f.size(0) for f in img_feats] + num_bbs_neg
    img_feat = pad_tensors(img_feats+img_feats_neg, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats+img_pos_feat_neg, num_bbs)

    img_input_ids = pad_sequence(img_input_ids+img_input_ids_neg, batch_first=True, padding_value=0)
    img_position_ids = torch.arange(0, img_input_ids.size(1), dtype=torch.long).unsqueeze(0)

    attn_masks_text = pad_sequence(attn_masks_text+attn_masks_text_neg, batch_first=True, padding_value=0)
    attn_masks_captions = pad_sequence(attn_masks_captions+attn_masks_captions_neg, batch_first=True, padding_value=0) if attn_masks_captions[0] is not None else None
    attn_masks_img = pad_sequence(attn_masks_img+attn_masks_img_neg, batch_first=True, padding_value=0)
    sample_size = bs
    # assert all(sample_size == len(i) for i in inputs)

    max_tl = input_ids.shape[1]
    out_size = attn_masks_img.size(1)
    gather_index = get_gather_index([1]*bs, num_bbs, bs, 1, out_size)

    batch = {
        'txts': {
            'input_ids': input_ids,
            'position_ids': position_ids,
            'attention_mask': attn_masks_text,
            'img_feat': None,
            'img_pos_feat': None,
            'img_masks': None,
            'gather_index': None
        },
        'imgs': {
            'input_ids': img_input_ids,
            'position_ids': img_position_ids,
            'attention_mask': attn_masks_img,
            'img_feat': img_feat,
            'img_pos_feat': img_pos_feat,
            'img_masks': None,
            'gather_index': gather_index
        },
        'caps': {
            'input_ids': captions_ids,
            'position_ids': position_ids_captions,
            'attention_mask': attn_masks_captions,
            'img_feat': None,
            'img_pos_feat': None,
            'img_masks': None,
            'gather_index': None
        },
        'sample_size': sample_size,
        'pos_ctx_indices': list(range(bs)),
        'neg_ctx_indices': list(range(bs, len(num_bbs))),
        'txt_index': idx,
        'img_fname': img_fname
    }
    return batch


class ItmValDataset(DetectFeatTxtTokDataset):
    """ For evaluating Image-Text-Retrieval task """
    def __init__(self, db_dir, img_dir, mini_batch_size=400):
        super().__init__(db_dir, img_dir)
        del self.lens
        self.txt2img = self.txt_db.txt2img
        self.img2txts = self.txt_db.img2txts
        self.all_img_ids = list(self.img2txts.keys())

        assert len(self.img2txts) >= mini_batch_size > 0
        self.bs = mini_batch_size

    def _get_batch_ids(self, i):
        gt_txt_id = self.ids[i]
        gt_img_id = self.txt2img[gt_txt_id]

        # sample fixed negatives for each gt image
        i = self.all_img_ids.index(gt_img_id)
        neg_st = i+1
        neg_end = neg_st+self.bs-1
        if neg_end > len(self.all_img_ids):
            # warp around
            neg_end -= len(self.all_img_ids)
            neg_img_ids = (self.all_img_ids[neg_st:]
                           + self.all_img_ids[:neg_end])
        else:
            neg_img_ids = self.all_img_ids[neg_st:neg_end]

        assert len(neg_img_ids) == (self.bs - 1),\
            "Did not sample enough neg samples"

        return gt_img_id, neg_img_ids

    def __getitem__(self, i):
        """ this returns list of mini-batches """
        gt_img_id, neg_img_ids = self._get_batch_ids(i)
        # NOTE 1st one is gt img
        batch = self.get_batch(i, [gt_img_id] + neg_img_ids)
        return batch

    def get_batch(self, i, img_ids):
        example = super().__getitem__(i)

        input_ids = example['input_ids']
        input_ids = self.txt_db.combine_inputs(input_ids)
        input_ids = input_ids.unsqueeze(0).expand(len(img_ids), -1).clone()
        position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                    ).unsqueeze(0)

        # process image features (gt always first)
        img_feats, img_pos_feats, num_bbs = map(
            list, unzip(map(self._get_img_feat, img_ids)))
        img_feat = pad_tensors(img_feats, num_bbs)
        img_pos_feat = pad_tensors(img_pos_feats, num_bbs)

        tl = input_ids.size(1)
        attn_masks_text = torch.ones(len(img_ids), tl).long()
        # attn_masks_text = torch.ones(1, tl).long()
        attn_masks_img = torch.zeros(len(img_ids), max(num_bbs)).long()
        for i, nbb in enumerate(num_bbs):
            attn_masks_img.data[i, :nbb].fill_(1)

        # out_size = attn_masks.size(1)
        gather_index = None  #get_gather_index([tl]*len(img_ids), num_bbs, len(img_ids), tl, out_size)

        batch = {'input_ids': input_ids,
                 'position_ids': position_ids,
                 'img_feat': img_feat,
                 'img_pos_feat': img_pos_feat,
                 'attn_masks_text': attn_masks_text,
                 'attn_masks_img': attn_masks_img,
                 'gather_index': gather_index}
        return batch


# for VQA