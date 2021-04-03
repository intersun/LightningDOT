#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


"""
 Pipeline to train DPR Biencoder
"""

import argparse
import glob
import logging
import math
import os
import random
import time
import json
import csv
import re

import torch
import numpy as np

from typing import Tuple
from collections import defaultdict
from torch import nn
from torch import Tensor as T
from torch.optim.lr_scheduler import LambdaLR

import torch.nn.functional as F
from torch.nn import LayerNorm
from transformers import BertModel, BertConfig, BertPreTrainedModel
from transformers.optimization import AdamW
from uniter_model.model.model import UniterPreTrainedModel, UniterModel, UniterConfig
from dvl.const import IMG_DIM
from dvl.indexer.faiss_indexers import DenseIndexer
from uniter_model.model.layer import GELU, BertOnlyMLMHead, BertPooler
from uniter_model.model.model import RegionClassification, RegionFeatureRegression, pad_tensor_to_mul

from typing import List


logger = logging.getLogger()
logger.setLevel(logging.INFO)
if (logger.hasHandlers()):
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)


def dot_product_scores(q_vectors: T, ctx_vectors: T, cosine=False) -> T:
    """
    calculates q->ctx scores for every row in ctx_vector
    :param q_vector:
    :param ctx_vector:
    :return:
    """
    # q_vector: n1 x D, ctx_vectors: n2 x D, result n1 x n2
    r = torch.matmul(q_vectors, torch.transpose(ctx_vectors, 0, 1))
    if cosine:
        n1 = torch.norm(q_vectors, dim=-1)
        n2 = torch.norm(ctx_vectors, dim=-1)
        n_out = torch.ger(n1, n2)
        return r / n_out
    return r


def cosine_scores(q_vector: T, ctx_vectors: T):
    # q_vector: n1 x D, ctx_vectors: n2 x D, result n1 x n2
    return F.cosine_similarity(q_vector, ctx_vectors, dim=1)


class BertEncoder(BertPreTrainedModel):
    def __init__(self, config, project_dim: int = 0):
        super().__init__(config)
        self.bert = BertModel(config)
        assert config.hidden_size > 0, 'Encoder hidden_size can\'t be zero'
        # self.encode_proj = nn.Linear(config.hidden_size, project_dim) if project_dim != 0 else None
        if project_dim > 0:
            self.encode_proj = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size * 2),
                GELU(),
                LayerNorm(config.hidden_size * 2, eps=1e-12),
                nn.Linear(config.hidden_size * 2, project_dim)
            )
        else:
            self.encode_proj = None
        self.init_weights()

    @classmethod
    def init_encoder(cls, cfg_name: str, checkpoint_path: str, project_dim: int = 0, dropout: float = 0.1, **kwargs)\
            -> BertModel:
        cfg = BertConfig.from_pretrained(cfg_name if cfg_name else 'bert-base-uncased')
        if dropout != 0:
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout

        if checkpoint_path is not None and len(checkpoint_path) > 0:
            state_dict = torch.load(checkpoint_path)
            return cls.from_pretrained(cfg_name, config=cfg, project_dim=project_dim, state_dict=state_dict, **kwargs)
        else:
            return cls.from_pretrained(cfg_name, config=cfg, project_dim=project_dim, **kwargs)

    def forward(self, input_ids, attention_mask, position_ids,
                img_feat=None, img_pos_feat=None, img_masks=None, gather_index=None):
        if self.config.output_hidden_states:
            sequence_output, pooled_output, hidden_states = self.bert(input_ids=input_ids,
                                                                      token_type_ids=None,
                                                                      attention_mask=attention_mask,
                                                                      position_ids=position_ids)
        else:
            hidden_states = None
            sequence_output, pooled_output = self.bert(input_ids=input_ids,
                                                       token_type_ids=None,
                                                       attention_mask=attention_mask,
                                                       position_ids=position_ids)
        pooled_output = sequence_output[:, 0, :]
        if self.encode_proj:
            pooled_output = self.encode_proj(pooled_output)
        return sequence_output, pooled_output, hidden_states

    def get_out_size(self):
        if self.encode_proj:
            return self.encode_proj.out_features
        return self.config.hidden_size


class UniterEncoder(UniterPreTrainedModel):
    def __init__(self, config, project_dim: int = 0):
        super().__init__(config)
        self.bert = UniterModel(config, IMG_DIM)
        assert config.hidden_size > 0, 'Encoder hidden_size can\'t be zero'
        # self.encode_proj = nn.Linear(config.hidden_size, project_dim) if project_dim != 0 else None   # Yen-Chun
        if project_dim > 0:
            self.encode_proj = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size * 2),
                GELU(),
                LayerNorm(config.hidden_size * 2, eps=1e-12),
                nn.Linear(config.hidden_size * 2, project_dim)
            )
        else:
            self.encode_proj = None
        self.apply(self.init_weights)

    @classmethod
    def init_encoder(cls, cfg_name: str, checkpoint_path: str, project_dim: int = 0, dropout: float = 0.1, **kwargs)\
            -> UniterModel:
        cfg = BertConfig.from_pretrained(cfg_name if cfg_name else 'bert-base-uncased')
        if dropout != 0:
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout
        if checkpoint_path is not None and len(checkpoint_path) > 0 and checkpoint_path.lower() != 'none':
            logger.info(f'load from {checkpoint_path} for uniter encoder')
            state_dict = torch.load(checkpoint_path)
        else:
            logger.info('no checkpoint, random initialization for img encoder')
            state_dict = dict()
        return cls.from_pretrained(cfg_name, state_dict=state_dict, project_dim=project_dim, **kwargs)

    def forward(self, input_ids, attention_mask, position_ids,
                img_feat, img_pos_feat,  img_masks, gather_index=None) -> Tuple[T, ...]:
        if self.config.output_hidden_states:
            sequence_output, pooled_output, hidden_states = self.bert(input_ids=input_ids,
                                                                      position_ids=position_ids,
                                                                      attention_mask=attention_mask,
                                                                      img_feat=img_feat,
                                                                      img_pos_feat=img_pos_feat,
                                                                      img_masks=img_masks,
                                                                      img_type_ids=None,
                                                                      gather_index=gather_index,
                                                                      output_all_encoded_layers=True
                                                                      )
        else:
            hidden_states = None
            sequence_output = self.bert(input_ids=input_ids,
                                        position_ids=position_ids,
                                        attention_mask=attention_mask,
                                        img_feat=img_feat,
                                        img_pos_feat=img_pos_feat,
                                        img_masks=img_masks,
                                        img_type_ids=None,
                                        gather_index=gather_index,
                                        output_all_encoded_layers=False)
        # pooled_output = self.bert.pooler(sequence_output)
        pooled_output = sequence_output[:, 0, :]
        if self.encode_proj:
            pooled_output = self.encode_proj(pooled_output)
        return sequence_output, pooled_output, hidden_states

    def get_out_size(self):
        if self.encode_proj:
            return self.encode_proj.out_features
        return self.config.hidden_size


class BiEncoder(nn.Module):
    """ Bi-Encoder model component. Encapsulates query/question and context/passage encoders.
    """

    def __init__(self, args, fix_img_encoder: bool = False, fix_txt_encoder: bool = False, project_dim: int = 0):
        super(BiEncoder, self).__init__()
        logger.info('*'*100)
        logger.info('loading img model')
        if args.img_model_type == 'uniter-base':
            self.img_model = UniterEncoder.init_encoder(args.img_model_config, checkpoint_path=args.img_checkpoint, project_dim=project_dim)
        else:
            raise ValueError(f'image encoder does not support other types ({args.img_model_type}) for now')

        logger.info('*' * 100)
        logger.info('loading txt model')
        if args.txt_model_type == 'bert-base':
            self.txt_model = BertEncoder.init_encoder(args.txt_model_config, checkpoint_path=args.txt_checkpoint, project_dim=project_dim)
        elif args.txt_model_type == 'uniter-base':
            self.txt_model = UniterEncoder.init_encoder(args.txt_model_config, checkpoint_path=args.txt_checkpoint, project_dim=project_dim)
        else:
            raise ValueError(f'txt encoder does not support other types ({args.txt_model_type}) for now')

        self.fix_img_encoder = fix_img_encoder
        self.fix_txt_encoder = fix_txt_encoder
        self.project_dim = project_dim
        if fix_txt_encoder:
            for param in self.txt_model.parameters():
                param.requires_grad = False
        if fix_img_encoder:
            for param in self.img_model.parameters():
                param.requires_grad = False

    @staticmethod
    def get_representation(sub_model, input_ids, attention_mask, position_ids, img_feat, img_pos_feat, img_masks,
                           gather_index=None, fix_encoder=False):
        if fix_encoder:
            with torch.no_grad():
                sequence_output, pooled_output, hidden_states = sub_model(input_ids, attention_mask, position_ids,
                                                                          img_feat, img_pos_feat, img_masks,
                                                                          gather_index)
        else:
            sequence_output, pooled_output, hidden_states = sub_model(input_ids, attention_mask, position_ids,
                                                                      img_feat, img_pos_feat, img_masks,
                                                                      gather_index)

        if sub_model.training:
            sequence_output.requires_grad_(requires_grad=True)
            pooled_output.requires_grad_(requires_grad=True)

        return sequence_output, pooled_output, hidden_states

    def forward(self, batch, output_all_encoded_layers=False):
        # batch keys
        #   imgs
        #   txts
        #   caps
        batch = defaultdict(lambda: None, batch)

        if 'txts' in batch:
            sb = batch['txts']
            txt_seq, txt_pooled, txt_hidden = self.get_representation(self.txt_model,  sb['input_ids'],
                                                                      sb['attention_mask'], sb['position_ids'],
                                                                      sb['img_feat'], sb['img_pos_feat'],
                                                                      sb['img_masks'],
                                                                      sb['gather_index'], self.fix_txt_encoder)
        else:
            txt_seq, txt_pooled = None, None

        if 'imgs' in batch:
            sb = batch['imgs']
            img_seq, img_pooled, img_hidden = self.get_representation(self.img_model, sb['input_ids'],
                                                                      sb['attention_mask'], sb['position_ids'],
                                                                      sb['img_feat'], sb['img_pos_feat'],
                                                                      sb['img_masks'],
                                                                      sb['gather_index'], self.fix_txt_encoder)
        else:
            img_seq, img_pooled = None, None

        if 'caps' in batch and batch['caps']['input_ids'] is not None:
            sb = batch['caps']
            cap_seq, cap_pooled, cap_hidden = self.get_representation(self.txt_model, sb['input_ids'],
                                                                      sb['attention_mask'], sb['position_ids'],
                                                                      sb['img_feat'], sb['img_pos_feat'],
                                                                      sb['img_masks'],
                                                                      sb['gather_index'], self.fix_txt_encoder)
        else:
            cap_seq, cap_pooled = None, None

        if output_all_encoded_layers:
            return txt_seq, img_seq, cap_seq
        else:
            return txt_pooled, img_pooled, cap_pooled


class BiEncoderForPretraining(nn.Module):
    """ MLM + MRM """
    def __init__(self, config_file, args, project_dim, img_dim, img_label_dim, nce_temp=1, ot_pos_only=False,
                 experiment=None):
        super().__init__()
        config = UniterConfig.from_json_file(config_file)
        self.bert = BiEncoder(args, project_dim=project_dim)
        self.cls = BertOnlyMLMHead(
            config, self.bert.img_model.bert.embeddings.word_embeddings.weight)    # ???
        self.feat_regress = RegionFeatureRegression(
            config.hidden_size, img_dim,
            self.bert.img_model.bert.img_embeddings.img_linear.weight)
        self.region_classifier = RegionClassification(
            config.hidden_size, img_label_dim)
        self.itm_output = nn.Linear(config.hidden_size, 2)
        self.cls_concat = args.cls_concat
        '''
        self.nce_output = BertPredictionHeadTransform(config)
        self.nce_output = nn.Sequential(BertPredictionHeadTransform(config),
                                        nn.Linear(config.hidden_size, img_dim))
        self.nce_norm = LayerNorm(config.hidden_size, eps=1e-12)
        self.nce_temp = nce_temp  # temperature
        '''
        self.ot_pos_only = ot_pos_only
        # self.apply(self.init_weights)
        self.vocab_pad = 0
        self.experiment = experiment

    def pad_vocab(self):
        # FIXME better padding after integrating huggingface ???
        emb_w = self.bert.embeddings.word_embeddings.weight.data
        padded_emb_w, n_pad = pad_tensor_to_mul(emb_w)
        padded_emb_w = nn.Parameter(padded_emb_w)
        self.bert.embeddings.word_embeddings.weight = padded_emb_w
        self.cls.predictions.decoder.weight = padded_emb_w
        self.vocab_pad = n_pad

    def forward(self, batch, task, compute_loss=True):
        batch = defaultdict(lambda: None, batch)
        if task == 'mlm':
            txt_labels = batch['txt_labels']
            return self.forward_mlm(batch, txt_labels, compute_loss)
        elif task == 'mrfr':
            img_mask_tgt = batch['img_mask_tgt']
            img_masks = batch['img_masks']
            mrfr_feat_target = batch['feat_targets']
            return self.forward_mrfr(batch, img_masks, img_mask_tgt, mrfr_feat_target, compute_loss)
        elif task == 'mrm-nce':
            raise NotImplementedError('nce does not work')
            img_mask_tgt = batch['img_mask_tgt']
            img_masks = batch['img_masks']
            img_masks_in = batch['img_masks_in']
            feat_target = batch['feat_targets']
            neg_feats = batch['neg_feats']
            return self.forward_mrm_nce(batch,
                                        img_masks_in, img_masks, img_mask_tgt,
                                        feat_target, neg_feats, compute_loss)
        elif task == 'itm':
            targets = batch['targets']
            ot_inputs = batch['ot_inputs']
            return self.forward_itm(batch,
                                    targets, ot_inputs, compute_loss)
        elif task.startswith('mrc'):
            img_mask_tgt = batch['img_mask_tgt']
            img_masks = batch['img_masks']
            mrc_label_target = batch['label_targets']
            return self.forward_mrc(batch,
                                    img_masks, img_mask_tgt,
                                    mrc_label_target, task, compute_loss)
        else:
            raise ValueError('invalid task')

    # MLM
    def forward_mlm(self, batch, txt_labels, compute_loss=True):
        txt_seq, img_seq, cap_seq = self.bert(batch, output_all_encoded_layers=True)
        # get only the text part

        img_cls = img_seq[:, 0:1, :].repeat(1, txt_seq.shape[1], 1)
        if self.cls_concat == 'add':
            sequence_output = txt_seq + img_cls
        elif self.cls_concat == 'multiply':
            sequence_output = txt_seq * img_cls
        elif len(self.cls_concat) == 0:
            sequence_output = txt_seq
        else:
            raise NotImplementedError(f'{self.cls_concat} not implemented yet')
        # only compute masked tokens for better efficiency
        masked_output = self._compute_masked_hidden(sequence_output,
                                                    txt_labels != -1)
        prediction_scores = self._pad_layer_unpad(masked_output, self.cls)
        if self.vocab_pad:
            prediction_scores = prediction_scores[:, :-self.vocab_pad]

        masked_lm_loss = F.cross_entropy(prediction_scores,
                                             txt_labels[txt_labels != -1],
                                             reduction='none')
        return masked_lm_loss, prediction_scores

    def _compute_masked_hidden(self, hidden, mask):
        """ get only the masked region (don't compute unnecessary hiddens) """
        mask = mask.unsqueeze(-1).expand_as(hidden)
        hidden_masked = hidden[mask].contiguous().view(-1, hidden.size(-1))
        return hidden_masked

    def _pad_layer_unpad(self, input_, layer):
        input_, n_pad = pad_tensor_to_mul(input_)
        output = layer(input_)
        if n_pad:
            output = output[:-n_pad, :]
        return output

    def mlm_eval(self, batch, gather_tgt):
        raise ValueError('Do not use this')
        sequence_output = self.bert(batch, output_all_encoded_layers=False)
        # get only the text part (excluding [CLS], [SEP])
        sequence_output = sequence_output[:, 1:input_ids.size(1)-1, :]
        # only compute masked tokens for better efficiency
        index = gather_tgt.unsqueeze(-1).expand(
            -1, -1, self.config.hidden_size)
        masked_output = torch.gather(sequence_output, dim=0, index=index)
        prediction_scores = self.cls(masked_output)
        if self.vocab_pad:
            prediction_scores = prediction_scores[..., :-self.vocab_pad]
        return prediction_scores

    # MRFR
    def forward_mrfr(self, batch, img_masks, img_mask_tgt,
                     feat_targets, compute_loss=True):
        txt_seq, img_seq, cap_seq = self.bert(batch, output_all_encoded_layers=True)
        txt_cls = txt_seq[:, 0:1, :].repeat(1, img_seq.shape[1], 1)
        if self.cls_concat == 'add':
            sequence_output = img_seq + txt_cls
        elif self.cls_concat == 'multiply':
            sequence_output = img_seq * txt_cls
        elif len(self.cls_concat) == 0:
            sequence_output = img_seq
        else:
            raise NotImplementedError(f'{self.cls_concat} not implemented yet')
        # only compute masked tokens for better efficiency
        masked_output = self._compute_masked_hidden(sequence_output,
                                                    img_mask_tgt)
        prediction_feat = self._pad_layer_unpad(masked_output,
                                                self.feat_regress)

        mrfr_loss = F.mse_loss(prediction_feat, feat_targets,
                               reduction='none')
        return mrfr_loss, prediction_feat

    # MRM-NCE
    def forward_mrm_nce(self,batch,
                        img_masks_in, img_masks, img_mask_tgt,
                        feat_targets, neg_feats, compute_loss=True):
        sequence_output = self.bert(batch,
                                    output_all_encoded_layers=False,
                                    img_masks=img_masks_in)

        # only compute masked tokens for better efficiency
        masked_output = self._compute_masked_hidden(sequence_output,
                                                    img_mask_tgt)

        masked_output = self._pad_layer_unpad(masked_output, self.nce_output)
        # neg within batch
        batch_neg = self._compute_masked_hidden(img_feat, ~img_masks)
        neg_feats, _ = pad_tensor_to_mul(
            torch.cat([neg_feats, batch_neg], dim=0))

        # shared image linear transform
        neg_output = self.nce_norm(
            self.bert.img_embeddings.img_linear(neg_feats))
        pos_output = self._pad_layer_unpad(feat_targets,
                                           self.bert.img_embeddings.img_linear)
        pos_output = self.nce_norm(pos_output)

        mrm_nce_loss = self.mrm_nce(masked_output, pos_output,
                                    neg_output, compute_loss=True)
        return mrm_nce_loss, masked_output

    def mrm_nce(self, masked_output, pos_output, neg_output,
                compute_loss=True):
        # dot product of ground truth feature
        masked_score = masked_output.matmul(pos_output.t())
        # dot product of neative samples
        neg_score = masked_output.matmul(neg_output.t())

        logits = torch.cat([masked_score, neg_score], dim=1).float()
        targets = torch.arange(0, masked_output.size(0),
                               dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits/self.nce_temp, targets,
                               reduction='none')
        return loss, logits

    def forward_itm(self, batch, targets, ot_inputs,
                    compute_loss=True):
        txt_seq, img_seq, cap_seq = self.bert(batch, output_all_encoded_layers=False)
        # OT loss
        if ot_inputs is not None:
            ot_scatter = ot_inputs['ot_scatter']

            b = sequence_output.size(0)
            tl = input_ids.size(1)
            il = img_feat.size(1)
            max_l = max(ot_inputs['scatter_max'] + 1, tl+il)

            ot_scatter = ot_scatter.unsqueeze(-1).expand_as(sequence_output)
            ctx_emb = torch.zeros(b, max_l, self.config.hidden_size,
                                  dtype=sequence_output.dtype,
                                  device=sequence_output.device
                                  ).scatter_(dim=1, index=ot_scatter,
                                             src=sequence_output)
            txt_emb = ctx_emb[:, :tl, :]
            img_emb = ctx_emb[:, tl:tl+il, :]

            txt_pad = ot_inputs['txt_pad']
            img_pad = ot_inputs['img_pad']
            ot_dist = optimal_transport_dist(txt_emb, img_emb,
                                             txt_pad, img_pad)
            if self.ot_pos_only:
                ot_loss = ot_dist.masked_select(targets == 1)
            else:
                ot_pos_dist = ot_dist.masked_select(targets == 1)
                ot_neg_dist = ot_dist.masked_select(targets == 0)
                ot_loss = (ot_pos_dist, ot_neg_dist)
        else:
            ot_loss = None

        loss_function = BiEncoderNllLoss()
        itm_loss1, is_correct1, scores1 = loss_function.calc(txt_seq, img_seq, cap_seq,
                                                          batch['pos_ctx_indices'],
                                                          batch['neg_ctx_indices'],
                                                          0.0, self.experiment, 'none')
        itm_loss2, is_correct2, scores2 = loss_function.calc(img_seq, txt_seq, cap_seq,
                                                          batch['pos_ctx_indices'],
                                                          batch['neg_ctx_indices'],
                                                          0.0, self.experiment, 'none')
        if compute_loss:
            return itm_loss1*0.5 + itm_loss2*0.5, ot_loss
        else:
            return itm_loss1*0.5 + itm_loss2*0.5, ot_loss, is_correct1*0.5 + is_correct2*0.5

    # MRC
    def forward_mrc(self, batch, img_masks, img_mask_tgt,
                    label_targets, task, compute_loss=True):
        txt_seq, img_seq, cap_seq = self.bert(batch, output_all_encoded_layers=True)
        txt_cls = txt_seq[:, 0:1, :].repeat(1, img_seq.shape[1], 1)
        if self.cls_concat == 'add':
            sequence_output = img_seq + txt_cls
        elif self.cls_concat == 'multiply':
            sequence_output = img_seq * txt_cls
        elif len(self.cls_concat) == 0:
            sequence_output = img_seq
        else:
            raise NotImplementedError(f'{self.cls_concat} not implemented yet')

        # sequence_output = torch.cat([txt_seq, img_seq], dim=1)
        # only compute masked regions for better efficiency
        masked_output = self._compute_masked_hidden(sequence_output, img_mask_tgt)
        prediction_soft_label = self._pad_layer_unpad(masked_output,
                                                      self.region_classifier)

        if "kl" in task:
            prediction_soft_label = F.log_softmax(
                prediction_soft_label, dim=-1)
            mrc_loss = F.kl_div(
                prediction_soft_label, label_targets, reduction='none')
        else:
            # background class should not be the target
            label_targets = torch.max(label_targets[:, 1:], dim=-1)[1] + 1
            mrc_loss = F.cross_entropy(
                prediction_soft_label, label_targets,
                ignore_index=0, reduction='none')
        return mrc_loss, prediction_soft_label


def get_optimizer(model: nn.Module, learning_rate: float = 1e-5, adam_eps: float = 1e-8,
                  weight_decay: float = 0.0, ) -> torch.optim.Optimizer:
    no_decay = ['bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_eps)
    return optimizer


def setup_for_distributed_mode(model: nn.Module, optimizer: torch.optim.Optimizer, device: object, n_gpu: int = 1,
                               local_rank: int = -1,
                               fp16: bool = False,
                               fp16_opt_level: str = "O1",
                               teacher_model = None) -> (nn.Module, torch.optim.Optimizer):
    model.to(device)
    if teacher_model is not None:
        teacher_model.to(device)
    if fp16:
        try:
            import apex
            from apex import amp
            apex.amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        if optimizer is None:
            if teacher_model is None:
                model = amp.initialize(model, optimizer, opt_level=fp16_opt_level)
            else:
                model, teacher_model = amp.initialize([model, teacher_model], optimizer, opt_level=fp16_opt_level)
        else:
            model, optimizer = amp.initialize(model, optimizer, opt_level=fp16_opt_level)

    #if n_gpu > 1:
    #    model = torch.nn.DataParallel(model)

    # if local_rank != -1:
    #   model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
    #                                                      output_device=local_rank,
    #                                                      find_unused_parameters=True)
    return model, optimizer


class BiEncoderNllLoss(object):

    def calc(self, q_vectors: T, ctx_vectors: T, caption_vectors: T, positive_idx_per_question: list,
             hard_negatice_idx_per_question: list = None, caption_score_weight: float = 0.1,
             experiment=None, reduction='mean'):
        """
        Computes nll loss for the given lists of question and ctx vectors.
        Note that although hard_negatice_idx_per_question in not currently in use, one can use it for the
        loss modifications. For example - weighted NLL with different factors for hard vs regular negatives.
        :return: a tuple of loss value and amount of correct predictions per batch
        """
        scores_img = self.get_scores(q_vectors, ctx_vectors)
        if caption_vectors is not None and caption_score_weight != 0:
            scores_caption = self.get_scores(q_vectors, caption_vectors)
            scores = (1 - caption_score_weight) * scores_img + caption_score_weight * scores_caption
        else:
            scores = scores_img

        if experiment is not None:
            experiment.log_metric('score_img_diag_mean', torch.diag(scores_img).mean().item())
            experiment.log_metric('score_img_offdiag_mean', (scores_img.sum() - torch.diag(scores_img).sum()) /
                                  (torch.numel(scores_img)-len(torch.diag(scores_img))))

            experiment.log_metric('score_diag_mean', torch.diag(scores).mean().item())
            experiment.log_metric('score_offdiag_mean', (scores.sum() - torch.diag(scores).sum()) /
                                  (torch.numel(scores) - len(torch.diag(scores))))

            if caption_vectors is not None and caption_score_weight != 0:
                experiment.log_metric('score_caption_diag_mean', torch.diag(scores_caption).mean().item())
                experiment.log_metric('score_caption_offdiag_mean', (scores_caption.sum() - torch.diag(scores_caption).sum()) /
                                      (torch.numel(scores_caption) - len(torch.diag(scores_caption))))

        if len(q_vectors.size()) > 1:
            q_num = q_vectors.size(0)
            scores = scores.view(q_num, -1)

        softmax_scores = F.log_softmax(scores, dim=1)

        loss = F.nll_loss(softmax_scores, torch.tensor(positive_idx_per_question).to(softmax_scores.device),
                          reduction=reduction)

        max_score, max_idxs = torch.max(softmax_scores, 1)
        correct_predictions_count = (max_idxs == torch.tensor(positive_idx_per_question).to(max_idxs.device)).sum()
        return loss, correct_predictions_count, scores

    @staticmethod
    def get_scores(q_vector: T, ctx_vectors: T) -> T:
        f = BiEncoderNllLoss.get_similarity_function()
        return f(q_vector, ctx_vectors)

    @staticmethod
    def get_similarity_function():
        return dot_product_scores


def get_schedule_linear(optimizer, warmup_steps, training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            0.0, float(training_steps - current_step) / float(max(1, training_steps - warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


class BiEncoderForVisualQuestionAnswering(nn.Module):
    """ Finetune multi-modal BERT for VQA
    """
    def __init__(self, args, fix_img_encoder: bool = False, fix_txt_encoder: bool = False,
                 seperate_caption_encoder: bool = False,
                 project_dim: int = 0,
                 hidden_size: int = 0, num_answer: int = 0, intersection=False):
        super(BiEncoderForVisualQuestionAnswering, self).__init__()
        self.biencoder = BiEncoder(args, fix_img_encoder, fix_txt_encoder, project_dim)
        self.intersection = intersection
        if self.intersection:
            hidden_size *= 2
        self.vqa_output = nn.Sequential(
            nn.Linear(hidden_size, hidden_size*2),
            GELU(),
            LayerNorm(hidden_size*2, eps=1e-12),
            nn.Linear(hidden_size*2, num_answer)
        )
        self.init_weights(self.vqa_output)

    def forward(self, batch, compute_loss=True, targets=None) -> Tuple[T, T]:

        q_pooled, ctx_pooled, caption_pooled = self.biencoder(batch)

        if self.intersection:
            pooled_output = torch.cat([q_pooled, ctx_pooled, q_pooled*ctx_pooled, q_pooled + ctx_pooled], dim=1)
        else:
            pooled_output = torch.cat([q_pooled, ctx_pooled], dim=1)

        answer_scores = self.vqa_output(pooled_output)

        if compute_loss:
            vqa_loss = F.binary_cross_entropy_with_logits(
                answer_scores, targets, reduction='none')
            return vqa_loss
        else:
            return answer_scores

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses
            # truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0,
                                       std=0.02)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


def load_biencoder_checkpoint(bi_encoder, biencoder_checkpoint):
    if biencoder_checkpoint is not None and len(biencoder_checkpoint) > 0 and biencoder_checkpoint.lower() != 'none':
        logger.info(f'loading ckpt from {biencoder_checkpoint}')
        state_dict = torch.load(biencoder_checkpoint, map_location='cpu')
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
    else:
        logger.info('no checkpoint provided, pass')
