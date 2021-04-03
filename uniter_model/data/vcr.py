"""
VCR dataset
"""
import json
import copy
import random

import torch
from torch.nn.utils.rnn import pad_sequence
from toolz.sandbox import unzip
from torch.utils.data import Dataset

from .data import DetectFeatLmdb, TxtLmdb, random_word
from .mrc import DetectFeatDir_for_mrc


class ImageTextDataset(Dataset):
    def __init__(self, db_dir, img_dir_gt=None, img_dir=None,
                 max_txt_len=120, task="qa"):
        self.txt_lens = []
        self.ids = []
        self.task = task
        for id_, len_ in json.load(open(f'{db_dir}/id2len_{task}.json')
                                   ).items():
            if max_txt_len == -1 or len_ <= max_txt_len:
                self.txt_lens.append(len_)
                self.ids.append(id_)

        self.db = TxtLmdb(db_dir, readonly=True)
        self.img_dir = img_dir
        self.img_dir_gt = img_dir_gt

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        id_ = self.ids[i]
        txt_dump = self.db[id_]
        img_dump_gt, img_dump = None, None
        img_fname_gt, img_fname = txt_dump['img_fname']
        if self.img_dump_gt:
            img_dump_gt = self.img_dump_gt[img_fname_gt]
        if self.img_dir:
            img_dump = self.img_dir[img_fname]
        return img_dump_gt, img_dump, txt_dump


class DetectFeatBertTokDataset(ImageTextDataset):
    def __init__(self, db_dir, img_dir_gt=None, img_dir=None,
                 max_txt_len=60, task="qa"):
        assert not (img_dir_gt is None and img_dir is None),\
            "image_dir_gt and img_dir cannot all be None"
        assert task == "qa" or task == "qar",\
            "VCR only allow two tasks: qa or qar"
        assert img_dir_gt is None or isinstance(img_dir_gt, DetectFeatLmdb)
        assert img_dir is None or isinstance(img_dir, DetectFeatLmdb)

        super().__init__(db_dir, img_dir_gt, img_dir, max_txt_len, task)
        txt2img = json.load(open(f'{db_dir}/txt2img.json'))
        if self.img_dir and self.img_dir_gt:
            self.lens = [tl+self.img_dir_gt.name2nbb[txt2img[id_][0]] +
                         self.img_dir.name2nbb[txt2img[id_][1]]
                         for tl, id_ in zip(self.txt_lens, self.ids)]
        elif self.img_dir:
            self.lens = [tl+self.img_dir.name2nbb[txt2img[id_][1]]
                         for tl, id_ in zip(self.txt_lens, self.ids)]
        else:
            self.lens = [tl+self.img_dir_gt.name2nbb[txt2img[id_][0]]
                         for tl, id_ in zip(self.txt_lens, self.ids)]

        meta = json.load(open(f'{db_dir}/meta.json', 'r'))
        self.cls_ = meta['CLS']
        self.sep = meta['SEP']
        self.mask = meta['MASK']
        self.v_range = meta['v_range']

    def _get_img_feat(self, fname_gt, fname):
        if self.img_dir and self.img_dir_gt:
            img_feat_gt, bb_gt = self.img_dir_gt[fname_gt]
            img_bb_gt = torch.cat([bb_gt, bb_gt[:, 4:5]*bb_gt[:, 5:]], dim=-1)

            img_feat, bb = self.img_dir[fname]
            img_bb = torch.cat([bb, bb[:, 4:5]*bb[:, 5:]], dim=-1)

            img_feat = torch.cat([img_feat_gt, img_feat], dim=0)
            img_bb = torch.cat([img_bb_gt, img_bb], dim=0)
            num_bb = img_feat.size(0)
        elif self.img_dir:
            img_feat, bb = self.img_dir[fname]
            img_bb = torch.cat([bb, bb[:, 4:5]*bb[:, 5:]], dim=-1)
            num_bb = img_feat.size(0)
        elif self.img_dir_gt:
            img_feat, bb = self.img_dir_gt[fname_gt]
            img_bb = torch.cat([bb, bb[:, 4:5]*bb[:, 5:]], dim=-1)
            num_bb = img_feat.size(0)
        return img_feat, img_bb, num_bb


class VcrDataset(DetectFeatBertTokDataset):
    def __init__(self, mask_prob, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_prob = mask_prob
        del self.txt_lens

    def _get_input_ids(self, txt_dump):
        # text input
        input_ids_q = txt_dump['input_ids']
        type_ids_q = [0]*len(input_ids_q)
        input_ids_as = txt_dump['input_ids_as']
        if self.task == "qar":
            input_ids_rs = txt_dump['input_ids_rs']
            answer_label = txt_dump['qa_target']
            assert answer_label >= 0, "answer_label < 0"
            input_ids_gt_a = [self.sep] + copy.deepcopy(
                input_ids_as[answer_label])
            type_ids_gt_a = [2] * len(input_ids_gt_a)
            type_ids_q += type_ids_gt_a
            input_ids_q += input_ids_gt_a
            input_ids_for_choices = input_ids_rs
        else:
            input_ids_for_choices = input_ids_as
        return input_ids_q, input_ids_for_choices, type_ids_q

    def __getitem__(self, i):
        id_ = self.ids[i]
        txt_dump = self.db[id_]
        img_feat, img_pos_feat, num_bb = self._get_img_feat(
            txt_dump['img_fname'][0], txt_dump['img_fname'][1])
        object_targets = txt_dump["object_ids"]
        input_ids_q, input_ids_for_choices, type_ids_q = self._get_input_ids(
            txt_dump)
        label = txt_dump['%s_target' % (self.task)]

        choice_num_bbs, choice_img_feats, choice_img_pos_feats = (
            [], [], [])
        (choice_txt_lens, choice_input_ids, choice_txt_type_ids,
         choice_attn_masks, choice_position_ids, choice_targets) = (
             [], [], [], [], [], [])
        choice_obj_targets, choice_img_masks = ([], [])

        for index, input_ids_a in enumerate(input_ids_for_choices):
            if index == label:
                target = torch.tensor([1]).long()
            else:
                target = torch.tensor([0]).long()
            input_ids = [self.cls_] + copy.deepcopy(input_ids_q) +\
                [self.sep] + input_ids_a + [self.sep]
            type_id_for_choice = 3 if type_ids_q[-1] == 2 else 2
            txt_type_ids = [0] + type_ids_q + [type_id_for_choice]*(
                len(input_ids_a)+2)
            attn_masks = [1] * len(input_ids)
            position_ids = list(range(len(input_ids)))
            attn_masks += [1] * num_bb

            input_ids = torch.tensor(input_ids)
            position_ids = torch.tensor(position_ids)
            attn_masks = torch.tensor(attn_masks)
            txt_type_ids = torch.tensor(txt_type_ids)

            choice_txt_lens.append(len(input_ids))
            choice_input_ids.append(input_ids)
            choice_attn_masks.append(attn_masks)
            choice_position_ids.append(position_ids)
            choice_txt_type_ids.append(txt_type_ids)

            choice_num_bbs.append(num_bb)
            choice_img_feats.append(img_feat)
            choice_img_pos_feats.append(img_pos_feat)
            choice_targets.append(target)

            # mask image input features
            num_gt_bb = len(object_targets)
            num_det_bb = num_bb - num_gt_bb
            # only mask gt features
            img_mask = [random.random() < self.mask_prob
                        for _ in range(num_gt_bb)]
            if not any(img_mask):
                # at least mask 1
                img_mask[0] = True
            img_mask += [False]*num_det_bb
            img_mask = torch.tensor(img_mask)
            object_targets += [0]*num_det_bb
            obj_targets = torch.tensor(object_targets)

            choice_obj_targets.append(obj_targets)
            choice_img_masks.append(img_mask)

        return (choice_input_ids, choice_position_ids, choice_txt_lens,
                choice_txt_type_ids,
                choice_img_feats, choice_img_pos_feats, choice_num_bbs,
                choice_attn_masks, choice_targets, choice_obj_targets,
                choice_img_masks)


def vcr_collate(inputs):
    (input_ids, position_ids, txt_lens, txt_type_ids, img_feats,
     img_pos_feats, num_bbs, attn_masks, targets,
     obj_targets, img_masks) = map(list, unzip(inputs))

    all_num_bbs, all_img_feats, all_img_pos_feats = (
        [], [], [])
    all_txt_lens, all_input_ids, all_attn_masks,\
        all_position_ids, all_txt_type_ids = (
            [], [], [], [], [])
    all_obj_targets = []
    all_targets = []
    # all_targets = targets
    all_img_masks = []
    for i in range(len(num_bbs)):
        all_input_ids += input_ids[i]
        all_position_ids += position_ids[i]
        all_txt_lens += txt_lens[i]
        all_txt_type_ids += txt_type_ids[i]
        all_img_feats += img_feats[i]
        all_img_pos_feats += img_pos_feats[i]
        all_num_bbs += num_bbs[i]
        all_attn_masks += attn_masks[i]
        all_obj_targets += obj_targets[i]
        all_img_masks += img_masks[i]
        all_targets += targets[i]

    all_input_ids = pad_sequence(all_input_ids,
                                 batch_first=True, padding_value=0)
    all_position_ids = pad_sequence(all_position_ids,
                                    batch_first=True, padding_value=0)
    all_txt_type_ids = pad_sequence(all_txt_type_ids,
                                    batch_first=True, padding_value=0)
    all_attn_masks = pad_sequence(all_attn_masks,
                                  batch_first=True, padding_value=0)
    all_img_masks = pad_sequence(all_img_masks,
                                 batch_first=True, padding_value=0)
    # all_targets = pad_sequence(all_targets,
    #                            batch_first=True, padding_value=0)
    all_targets = torch.stack(all_targets, dim=0)

    batch_size = len(all_img_feats)
    num_bb = max(all_num_bbs)
    feat_dim = all_img_feats[0].size(1)
    pos_dim = all_img_pos_feats[0].size(1)
    all_img_feat = torch.zeros(batch_size, num_bb, feat_dim)
    all_img_pos_feat = torch.zeros(batch_size, num_bb, pos_dim)
    all_obj_target = torch.zeros(batch_size, num_bb)
    for i, (im, pos, label) in enumerate(zip(
            all_img_feats, all_img_pos_feats, all_obj_targets)):
        len_ = im.size(0)
        all_img_feat.data[i, :len_, :] = im.data
        all_img_pos_feat.data[i, :len_, :] = pos.data
        all_obj_target.data[i, :len_] = label.data

    obj_targets = all_obj_target[all_img_masks].contiguous()
    return (all_input_ids, all_position_ids, all_txt_lens,
            all_txt_type_ids,
            all_img_feat, all_img_pos_feat, all_num_bbs,
            all_attn_masks, all_targets, obj_targets, all_img_masks)


class VcrEvalDataset(DetectFeatBertTokDataset):
    def __init__(self, split, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.split = split
        del self.txt_lens

    def _get_input_ids(self, txt_dump):
        # text input
        input_ids_for_choices = []
        type_ids_for_choices = []
        input_ids_q = txt_dump['input_ids']
        type_ids_q = [0]*len(input_ids_q)
        input_ids_as = txt_dump['input_ids_as']
        input_ids_rs = txt_dump['input_ids_rs']
        for index, input_ids_a in enumerate(input_ids_as):
            curr_input_ids_qa = [self.cls_] + copy.deepcopy(input_ids_q) +\
                [self.sep] + input_ids_a + [self.sep]
            curr_type_ids_qa = [0] + type_ids_q + [2]*(
                len(input_ids_a)+2)
            input_ids_for_choices.append(curr_input_ids_qa)
            type_ids_for_choices.append(curr_type_ids_qa)
        for index, input_ids_a in enumerate(input_ids_as):
            curr_input_ids_qa = [self.cls_] + copy.deepcopy(input_ids_q) +\
                [self.sep] + input_ids_a + [self.sep]
            curr_type_ids_qa = [0] + type_ids_q + [2]*(
                len(input_ids_a)+1)
            if (self.split == "val" and index == txt_dump["qa_target"]) or\
                    self.split == "test":
                for input_ids_r in input_ids_rs:
                    curr_input_ids_qar = copy.deepcopy(curr_input_ids_qa) +\
                        input_ids_r + [self.sep]
                    curr_type_ids_qar = copy.deepcopy(curr_type_ids_qa) +\
                        [3]*(len(input_ids_r)+2)
                    input_ids_for_choices.append(curr_input_ids_qar)
                    type_ids_for_choices.append(curr_type_ids_qar)
        return input_ids_for_choices, type_ids_for_choices

    def __getitem__(self, i):
        qid = self.ids[i]
        id_ = self.ids[i]
        txt_dump = self.db[id_]
        img_feat, img_pos_feat, num_bb = self._get_img_feat(
            txt_dump['img_fname'][0], txt_dump['img_fname'][1])
        object_targets = txt_dump["object_ids"]
        input_ids_for_choices, type_ids_for_choices = self._get_input_ids(
            txt_dump)
        qa_target = torch.tensor([int(txt_dump["qa_target"])])
        qar_target = torch.tensor([int(txt_dump["qar_target"])])

        choice_num_bbs, choice_img_feats, choice_img_pos_feats = (
            [], [], [])
        (choice_txt_lens, choice_input_ids, choice_attn_masks,
         choice_position_ids, choice_txt_type_ids) = (
             [], [], [], [], [])
        choice_obj_targets = []
        for index, input_ids in enumerate(input_ids_for_choices):
            txt_type_ids = type_ids_for_choices[index]
            attn_masks = [1] * len(input_ids)
            position_ids = list(range(len(input_ids)))
            attn_masks += [1] * num_bb

            input_ids = torch.tensor(input_ids)
            position_ids = torch.tensor(position_ids)
            attn_masks = torch.tensor(attn_masks)
            txt_type_ids = torch.tensor(txt_type_ids)

            choice_txt_lens.append(len(input_ids))
            choice_input_ids.append(input_ids)
            choice_attn_masks.append(attn_masks)
            choice_position_ids.append(position_ids)
            choice_txt_type_ids.append(txt_type_ids)

            choice_num_bbs.append(num_bb)
            choice_img_feats.append(img_feat)
            choice_img_pos_feats.append(img_pos_feat)

            obj_targets = torch.tensor(object_targets)
            choice_obj_targets.append(obj_targets)

        return (qid, choice_input_ids, choice_position_ids, choice_txt_lens,
                choice_txt_type_ids,
                choice_img_feats, choice_img_pos_feats, choice_num_bbs,
                choice_attn_masks, qa_target, qar_target, choice_obj_targets)


def vcr_eval_collate(inputs):
    (qids, input_ids, position_ids, txt_lens, txt_type_ids,
     img_feats, img_pos_feats,
     num_bbs, attn_masks, qa_targets, qar_targets,
     obj_targets) = map(list, unzip(inputs))

    all_num_bbs, all_img_feats, all_img_pos_feats = (
        [], [], [])
    all_txt_lens, all_input_ids, all_attn_masks, all_position_ids,\
        all_txt_type_ids = (
            [], [], [], [], [])
    # all_qa_targets = qa_targets
    # all_qar_targets = qar_targets
    all_obj_targets = []
    for i in range(len(num_bbs)):
        all_input_ids += input_ids[i]
        all_position_ids += position_ids[i]
        all_txt_lens += txt_lens[i]
        all_img_feats += img_feats[i]
        all_img_pos_feats += img_pos_feats[i]
        all_num_bbs += num_bbs[i]
        all_attn_masks += attn_masks[i]
        all_txt_type_ids += txt_type_ids[i]
        all_obj_targets += obj_targets[i]

    all_input_ids = pad_sequence(all_input_ids,
                                 batch_first=True, padding_value=0)
    all_position_ids = pad_sequence(all_position_ids,
                                    batch_first=True, padding_value=0)
    all_txt_type_ids = pad_sequence(all_txt_type_ids,
                                    batch_first=True, padding_value=0)
    all_attn_masks = pad_sequence(all_attn_masks,
                                  batch_first=True, padding_value=0)
    all_obj_targets = pad_sequence(all_obj_targets,
                                   batch_first=True, padding_value=0)
    all_qa_targets = torch.stack(qa_targets, dim=0)
    all_qar_targets = torch.stack(qar_targets, dim=0)

    batch_size = len(all_img_feats)
    num_bb = max(all_num_bbs)
    feat_dim = all_img_feats[0].size(1)
    pos_dim = all_img_pos_feats[0].size(1)
    all_img_feat = torch.zeros(batch_size, num_bb, feat_dim)
    all_img_pos_feat = torch.zeros(batch_size, num_bb, pos_dim)
    for i, (im, pos) in enumerate(zip(
            all_img_feats, all_img_pos_feats)):
        len_ = im.size(0)
        all_img_feat.data[i, :len_, :] = im.data
        all_img_pos_feat.data[i, :len_, :] = pos.data

    return (qids, all_input_ids, all_position_ids, all_txt_lens,
            all_txt_type_ids,
            all_img_feat, all_img_pos_feat, all_num_bbs,
            all_attn_masks, all_qa_targets, all_qar_targets, all_obj_targets)


class MlmDatasetForVCR(DetectFeatBertTokDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.txt_lens

    def _get_input_ids(self, txt_dump, mask=True):
        # text input
        input_ids_q = txt_dump['input_ids']
        type_ids_q = [0]*len(input_ids_q)
        if mask:
            input_ids_q, txt_labels_q = random_word(
                input_ids_q, self.v_range, self.mask)
        else:
            txt_labels_q = input_ids_q

        answer_label = txt_dump['qa_target']
        assert answer_label >= 0, "answer_label < 0"

        input_ids_a = txt_dump['input_ids_as'][answer_label]
        type_ids_a = [2]*len(input_ids_a)
        if mask:
            input_ids_a, txt_labels_a = random_word(
                input_ids_a, self.v_range, self.mask)
        else:
            txt_labels_a = input_ids_a

        input_ids = input_ids_q + [self.sep] + input_ids_a
        type_ids = type_ids_q + [0] + type_ids_a
        txt_labels = txt_labels_q + [-1] + txt_labels_a

        if self.task == "qar":
            rationale_label = txt_dump['qar_target']
            assert rationale_label >= 0, "rationale_label < 0"

            input_ids_r = txt_dump['input_ids_rs'][rationale_label]
            type_ids_r = [3]*len(input_ids_r)
            if mask:
                input_ids_r, txt_labels_r = random_word(
                    input_ids_r, self.v_range, self.mask)
            else:
                txt_labels_r = input_ids_r

            input_ids += [self.sep] + input_ids_r
            type_ids += [2] + type_ids_r
            txt_labels += [-1] + txt_labels_r
        return input_ids, type_ids, txt_labels

    def __getitem__(self, i):
        id_ = self.ids[i]
        txt_dump = self.db[id_]
        img_feat, img_pos_feat, num_bb = self._get_img_feat(
            txt_dump['img_fname'][0], txt_dump['img_fname'][1])

        # txt inputs
        input_ids, type_ids, txt_labels = self._get_input_ids(txt_dump)
        input_ids = [self.cls_] + input_ids + [self.sep]
        txt_labels = [-1] + txt_labels + [-1]
        type_ids = [type_ids[0]] + type_ids + [type_ids[-1]]
        attn_masks = [1] * len(input_ids)
        position_ids = list(range(len(input_ids)))
        attn_masks += [1] * num_bb
        input_ids = torch.tensor(input_ids)
        position_ids = torch.tensor(position_ids)
        attn_masks = torch.tensor(attn_masks)
        txt_labels = torch.tensor(txt_labels)
        type_ids = torch.tensor(type_ids)

        return (input_ids, position_ids, type_ids, img_feat, img_pos_feat,
                attn_masks, txt_labels)


def mlm_collate_for_vcr(inputs):
    (input_ids, position_ids, type_ids, img_feats, img_pos_feats, attn_masks,
     txt_labels) = map(list, unzip(inputs))

    txt_lens = [i.size(0) for i in input_ids]
    num_bbs = [f.size(0) for f in img_feats]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    type_ids = pad_sequence(type_ids, batch_first=True, padding_value=0)
    position_ids = pad_sequence(position_ids,
                                batch_first=True, padding_value=0)
    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
    txt_labels = pad_sequence(txt_labels, batch_first=True, padding_value=-1)

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

    return (input_ids, position_ids, type_ids, txt_lens,
            img_feat, img_pos_feat, num_bbs,
            attn_masks, txt_labels)


class MrmDatasetForVCR(DetectFeatBertTokDataset):
    def __init__(self, mask_prob, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_prob = mask_prob
        del self.txt_lens

    def _get_input_ids(self, txt_dump, mask=True):
        # text input
        input_ids_q = txt_dump['input_ids']
        type_ids_q = [0]*len(input_ids_q)

        answer_label = txt_dump['qa_target']
        assert answer_label >= 0, "answer_label < 0"

        input_ids_a = txt_dump['input_ids_as'][answer_label]
        type_ids_a = [2]*len(input_ids_a)

        input_ids = input_ids_q + [self.sep] + input_ids_a
        type_ids = type_ids_q + [0] + type_ids_a

        if self.task == "qar":
            rationale_label = txt_dump['qar_target']
            assert rationale_label >= 0, "rationale_label < 0"

            input_ids_r = txt_dump['input_ids_rs'][rationale_label]
            type_ids_r = [3]*len(input_ids_r)

            input_ids += [self.sep] + input_ids_r
            type_ids += [2] + type_ids_r
        return input_ids, type_ids

    def __getitem__(self, i):
        id_ = self.ids[i]
        txt_dump = self.db[id_]
        img_feat, img_pos_feat, num_bb = self._get_img_feat(
            txt_dump['img_fname'][0], txt_dump['img_fname'][1])

        # image input features
        img_mask = [random.random() < self.mask_prob for _ in range(num_bb)]
        if not any(img_mask):
            # at least mask 1
            img_mask[0] = True
        img_mask = torch.tensor(img_mask)

        # text input
        input_ids, type_ids = self._get_input_ids(txt_dump)
        input_ids = [self.cls_] + input_ids + [self.sep]
        type_ids = [type_ids[0]] + type_ids + [type_ids[-1]]
        attn_masks = [1] * len(input_ids)
        position_ids = list(range(len(input_ids)))
        attn_masks += [1] * num_bb
        input_ids = torch.tensor(input_ids)
        position_ids = torch.tensor(position_ids)
        attn_masks = torch.tensor(attn_masks)
        type_ids = torch.tensor(type_ids)

        return (input_ids, position_ids, type_ids, img_feat, img_pos_feat,
                attn_masks, img_mask)


def mrm_collate_for_vcr(inputs):
    (input_ids, position_ids, type_ids, img_feats, img_pos_feats,
     attn_masks, img_masks) = map(list, unzip(inputs))

    txt_lens = [i.size(0) for i in input_ids]
    num_bbs = [f.size(0) for f in img_feats]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = pad_sequence(position_ids,
                                batch_first=True, padding_value=0)
    type_ids = pad_sequence(type_ids, batch_first=True, padding_value=0)
    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
    img_masks = pad_sequence(img_masks, batch_first=True, padding_value=0)

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

    return (input_ids, position_ids, type_ids, txt_lens,
            img_feat, img_pos_feat, num_bbs,
            attn_masks, img_masks)


class DetectFeatBertTokDataset_for_mrc_vcr(DetectFeatBertTokDataset):
    def __init__(self, db_dir, img_dir_gt=None, img_dir=None,
                 max_txt_len=60, task="qa"):
        assert not (img_dir_gt is None and img_dir is None),\
            "image_dir_gt and img_dir cannot all be None"
        assert task == "qa" or task == "qar",\
            "VCR only allow two tasks: qa or qar"
        assert img_dir_gt is None or isinstance(img_dir_gt, DetectFeatLmdb)
        assert img_dir is None or isinstance(img_dir, DetectFeatLmdb)
        super().__init__(db_dir, img_dir_gt, img_dir, max_txt_len, task)
        if self.img_dir:
            self.img_dir = DetectFeatDir_for_mrc(img_dir)
        if self.img_dir_gt:
            self.img_dir_gt = DetectFeatDir_for_mrc(img_dir_gt)

    def _get_img_feat(self, fname_gt, fname):
        if self.img_dir and self.img_dir_gt:
            img_feat_gt, bb_gt,\
                img_soft_labels_gt = self.img_dir_gt[fname_gt]
            img_bb_gt = torch.cat([bb_gt, bb_gt[:, 4:5]*bb_gt[:, 5:]], dim=-1)

            img_feat, bb, img_soft_labels = self.img_dir[fname]
            img_bb = torch.cat([bb, bb[:, 4:5]*bb[:, 5:]], dim=-1)

            img_feat = torch.cat([img_feat_gt, img_feat], dim=0)
            img_bb = torch.cat([img_bb_gt, img_bb], dim=0)
            img_soft_labels = torch.cat(
                [img_soft_labels_gt, img_soft_labels], dim=0)
            num_bb = img_feat.size(0)
        elif self.img_dir:
            img_feat, bb, img_soft_labels = self.img_dir[fname]
            img_bb = torch.cat([bb, bb[:, 4:5]*bb[:, 5:]], dim=-1)
            num_bb = img_feat.size(0)
        elif self.img_dir_gt:
            img_feat, bb, img_soft_labels = self.img_dir_gt[fname_gt]
            img_bb = torch.cat([bb, bb[:, 4:5]*bb[:, 5:]], dim=-1)
            num_bb = img_feat.size(0)
        return img_feat, img_bb, img_soft_labels, num_bb


class MrcDatasetForVCR(DetectFeatBertTokDataset_for_mrc_vcr):
    def __init__(self, mask_prob, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_prob = mask_prob
        del self.txt_lens

    def _get_input_ids(self, txt_dump, mask=True):
        # text input
        input_ids_q = txt_dump['input_ids']
        type_ids_q = [0]*len(input_ids_q)

        answer_label = txt_dump['qa_target']
        assert answer_label >= 0, "answer_label < 0"

        input_ids_a = txt_dump['input_ids_as'][answer_label]
        type_ids_a = [2]*len(input_ids_a)

        input_ids = input_ids_q + [self.sep] + input_ids_a
        type_ids = type_ids_q + [0] + type_ids_a

        if self.task == "qar":
            rationale_label = txt_dump['qar_target']
            assert rationale_label >= 0, "rationale_label < 0"

            input_ids_r = txt_dump['input_ids_rs'][rationale_label]
            type_ids_r = [3]*len(input_ids_r)

            input_ids += [self.sep] + input_ids_r
            type_ids += [2] + type_ids_r
        return input_ids, type_ids

    def __getitem__(self, i):
        id_ = self.ids[i]
        txt_dump = self.db[id_]
        img_feat, img_pos_feat, img_soft_labels, num_bb = self._get_img_feat(
            txt_dump['img_fname'][0], txt_dump['img_fname'][1])

        # image input features
        img_mask = [random.random() < self.mask_prob for _ in range(num_bb)]
        if not any(img_mask):
            # at least mask 1
            img_mask[0] = True
        img_mask = torch.tensor(img_mask)

        # text input
        input_ids, type_ids = self._get_input_ids(txt_dump)
        input_ids = [self.cls_] + input_ids + [self.sep]
        type_ids = [type_ids[0]] + type_ids + [type_ids[-1]]
        attn_masks = [1] * len(input_ids)
        position_ids = list(range(len(input_ids)))
        attn_masks += [1] * num_bb
        input_ids = torch.tensor(input_ids)
        position_ids = torch.tensor(position_ids)
        attn_masks = torch.tensor(attn_masks)
        type_ids = torch.tensor(type_ids)

        return (input_ids, position_ids, type_ids, img_feat, img_pos_feat,
                img_soft_labels, attn_masks, img_mask)


def mrc_collate_for_vcr(inputs):
    (input_ids, position_ids, type_ids, img_feats, img_pos_feats,
     img_soft_labels, attn_masks, img_masks
     ) = map(list, unzip(inputs))

    txt_lens = [i.size(0) for i in input_ids]
    num_bbs = [f.size(0) for f in img_feats]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = pad_sequence(position_ids,
                                batch_first=True, padding_value=0)
    type_ids = pad_sequence(type_ids, batch_first=True, padding_value=0)
    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
    img_masks = pad_sequence(img_masks, batch_first=True, padding_value=0)

    batch_size = len(img_feats)
    num_bb = max(num_bbs)
    feat_dim = img_feats[0].size(1)
    soft_label_dim = img_soft_labels[0].size(1)
    pos_dim = img_pos_feats[0].size(1)
    img_feat = torch.zeros(batch_size, num_bb, feat_dim)
    img_pos_feat = torch.zeros(batch_size, num_bb, pos_dim)
    img_soft_label = torch.zeros(batch_size, num_bb, soft_label_dim)
    for i, (im, pos, label) in enumerate(zip(img_feats,
                                             img_pos_feats,
                                             img_soft_labels)):
        len_ = im.size(0)
        img_feat.data[i, :len_, :] = im.data
        img_pos_feat.data[i, :len_, :] = pos.data
        img_soft_label.data[i, :len_, :] = label.data

    img_masks_ext_for_label = img_masks.unsqueeze(-1).expand_as(img_soft_label)
    label_targets = img_soft_label[img_masks_ext_for_label].contiguous().view(
        -1, soft_label_dim)
    return (input_ids, position_ids, type_ids, txt_lens,
            img_feat, img_pos_feat, num_bbs,
            attn_masks, (img_masks, label_targets))
