"""
Bert for VCR model
"""
from torch import nn
from torch.nn import functional as F
from pytorch_pretrained_bert.modeling import (
    BertOnlyMLMHead)
from .model import (BertForImageTextPretraining,
                    _get_image_hidden,
                    mask_img_feat,
                    RegionFeatureRegression,
                    mask_img_feat_for_mrc,
                    RegionClassification)
import torch
import random


class BertForImageTextPretrainingForGQA(BertForImageTextPretraining):
    def init_type_embedding(self):
        new_emb = nn.Embedding(3, self.bert.config.hidden_size)
        new_emb.apply(self.init_bert_weights)
        for i in [0, 1]:
            emb = self.bert.embeddings.token_type_embeddings.weight.data[i, :]
            new_emb.weight.data[i, :].copy_(emb)
        emb = self.bert.embeddings.token_type_embeddings.weight.data[0, :]
        new_emb.weight.data[2, :].copy_(emb)
        self.bert.embeddings.token_type_embeddings = new_emb

    def forward(self, input_ids, position_ids, txt_type_ids, txt_lens,
                img_feat, img_pos_feat, num_bbs,
                attention_mask, labels, task, compute_loss=True):
        if task == 'mlm':
            txt_labels = labels
            return self.forward_mlm(input_ids, position_ids, txt_type_ids,
                                    txt_lens,
                                    img_feat, img_pos_feat, num_bbs,
                                    attention_mask, txt_labels, compute_loss)
        elif task == 'mrm':
            img_mask = labels
            return self.forward_mrm(input_ids, position_ids, txt_type_ids,
                                    txt_lens,
                                    img_feat, img_pos_feat, num_bbs,
                                    attention_mask, img_mask, compute_loss)
        elif task.startswith('mrc'):
            img_mask, mrc_label_target = labels
            return self.forward_mrc(input_ids, position_ids, txt_type_ids,
                                    txt_lens,
                                    img_feat, img_pos_feat, num_bbs,
                                    attention_mask, img_mask,
                                    mrc_label_target, task, compute_loss)
        else:
            raise ValueError('invalid task')

    # MLM
    def forward_mlm(self, input_ids, position_ids, txt_type_ids, txt_lens,
                    img_feat, img_pos_feat, num_bbs,
                    attention_mask, txt_labels, compute_loss=True):
        sequence_output = self.bert(input_ids, position_ids, txt_lens,
                                    img_feat, img_pos_feat, num_bbs,
                                    attention_mask,
                                    output_all_encoded_layers=False,
                                    txt_type_ids=txt_type_ids)
        # get only the text part
        sequence_output = sequence_output[:, :input_ids.size(1), :]
        # only compute masked tokens for better efficiency
        prediction_scores = self.masked_compute_scores(
            sequence_output, txt_labels != -1)
        if self.vocab_pad:
            prediction_scores = prediction_scores[:, :-self.vocab_pad]

        if compute_loss:
            masked_lm_loss = F.cross_entropy(prediction_scores,
                                             txt_labels[txt_labels != -1],
                                             reduction='none')
            return masked_lm_loss
        else:
            return prediction_scores

    # MRM
    def forward_mrm(self, input_ids, position_ids, txt_type_ids, txt_lens,
                    img_feat, img_pos_feat, num_bbs,
                    attention_mask, img_masks, compute_loss=True):
        img_feat, feat_targets = mask_img_feat(img_feat, img_masks)
        sequence_output = self.bert(input_ids, position_ids, txt_lens,
                                    img_feat, img_pos_feat, num_bbs,
                                    attention_mask,
                                    output_all_encoded_layers=False,
                                    txt_type_ids=txt_type_ids)
        # get only the text part
        sequence_output = _get_image_hidden(sequence_output, txt_lens, num_bbs)
        # only compute masked tokens for better efficiency
        prediction_feat = self.masked_compute_feat(
            sequence_output, img_masks)

        if compute_loss:
            mrm_loss = F.mse_loss(prediction_feat, feat_targets,
                                  reduction='none')
            return mrm_loss
        else:
            return prediction_feat

    # MRC
    def forward_mrc(self, input_ids, position_ids, txt_type_ids, txt_lens,
                    img_feat, img_pos_feat, num_bbs,
                    attention_mask, img_masks,
                    label_targets, task, compute_loss=True):
        img_feat = mask_img_feat_for_mrc(img_feat, img_masks)
        sequence_output = self.bert(input_ids, position_ids, txt_lens,
                                    img_feat, img_pos_feat, num_bbs,
                                    attention_mask,
                                    output_all_encoded_layers=False,
                                    txt_type_ids=txt_type_ids)
        # get only the image part
        sequence_output = _get_image_hidden(sequence_output, txt_lens, num_bbs)
        # only compute masked tokens for better efficiency
        prediction_soft_label = self.masked_predict_labels(
            sequence_output, img_masks)

        if compute_loss:
            if "kl" in task:
                prediction_soft_label = F.log_softmax(
                    prediction_soft_label, dim=-1)
                mrc_loss = F.kl_div(
                    prediction_soft_label, label_targets, reduction='none')
            else:
                label_targets = torch.max(
                    label_targets, -1)[1]  # argmax
                mrc_loss = F.cross_entropy(
                    prediction_soft_label, label_targets,
                    ignore_index=0, reduction='none')
            return mrc_loss
        else:
            return prediction_soft_label
