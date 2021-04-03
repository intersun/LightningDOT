"""
Bert for VCR model
"""
from torch import nn
from torch.nn import functional as F
from pytorch_pretrained_bert.modeling import (
    BertPreTrainedModel, BertEmbeddings, BertEncoder, BertLayerNorm,
    BertPooler, BertOnlyMLMHead)
from .model import (BertTextEmbeddings, BertImageEmbeddings,
                    BertForImageTextMaskedLM,
                    BertVisionLanguageEncoder,
                    BertForImageTextPretraining,
                    _get_image_hidden,
                    mask_img_feat,
                    RegionFeatureRegression,
                    mask_img_feat_for_mrc,
                    RegionClassification)
import torch
import random


class BertVisionLanguageEncoderForVCR(BertVisionLanguageEncoder):
    """ Modification for Joint Vision-Language Encoding
    """
    def __init__(self, config, img_dim, num_region_toks):
        BertPreTrainedModel.__init__(self, config)
        self.embeddings = BertTextEmbeddings(config)
        self.img_embeddings = BertImageEmbeddings(config, img_dim)
        self.num_region_toks = num_region_toks
        self.region_token_embeddings = nn.Embedding(
            num_region_toks,
            config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, position_ids, txt_lens,
                img_feat, img_pos_feat, num_bbs,
                attention_mask, output_all_encoded_layers=True,
                txt_type_ids=None, img_type_ids=None, region_tok_ids=None):
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self._compute_img_txt_embeddings(
            input_ids, position_ids, txt_lens,
            img_feat, img_pos_feat, num_bbs, attention_mask.size(1),
            txt_type_ids, img_type_ids)
        if region_tok_ids is not None:
            region_tok_embeddings = self.region_token_embeddings(
                region_tok_ids)
            embedding_output += region_tok_embeddings
            embedding_output = self.LayerNorm(embedding_output)
            embedding_output = self.dropout(embedding_output)
        encoded_layers = self.encoder(
            embedding_output, extended_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers


class BertForVisualCommonsenseReasoning(BertPreTrainedModel):
    """ Finetune multi-modal BERT for ITM
    """
    def __init__(self, config, img_dim, obj_cls=True, img_label_dim=81):
        super().__init__(config, img_dim)
        self.bert = BertVisionLanguageEncoder(
            config, img_dim)
        # self.vcr_output = nn.Linear(config.hidden_size, 1)
        # self.vcr_output = nn.Linear(config.hidden_size, 2)
        self.vcr_output = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size*2),
            nn.ReLU(),
            BertLayerNorm(config.hidden_size*2, eps=1e-12),
            nn.Linear(config.hidden_size*2, 2)
        )
        self.apply(self.init_bert_weights)
        self.obj_cls = obj_cls
        if self.obj_cls:
            self.region_classifier = RegionClassification(
                config.hidden_size, img_label_dim)

    def init_type_embedding(self):
        new_emb = nn.Embedding(4, self.bert.config.hidden_size)
        new_emb.apply(self.init_bert_weights)
        for i in [0, 1]:
            emb = self.bert.embeddings.token_type_embeddings.weight.data[i, :]
            new_emb.weight.data[i, :].copy_(emb)
        emb = self.bert.embeddings.token_type_embeddings.weight.data[0, :]
        new_emb.weight.data[2, :].copy_(emb)
        new_emb.weight.data[3, :].copy_(emb)
        self.bert.embeddings.token_type_embeddings = new_emb

    def init_word_embedding(self, num_special_tokens):
        orig_word_num = self.bert.embeddings.word_embeddings.weight.size(0)
        new_emb = nn.Embedding(
            orig_word_num + num_special_tokens, self.bert.config.hidden_size)
        new_emb.apply(self.init_bert_weights)
        emb = self.bert.embeddings.word_embeddings.weight.data
        new_emb.weight.data[:orig_word_num, :].copy_(emb)
        self.bert.embeddings.word_embeddings = new_emb

    def masked_predict_labels(self, sequence_output, mask):
        # only compute masked outputs
        mask = mask.unsqueeze(-1).expand_as(sequence_output)
        sequence_output_masked = sequence_output[mask].contiguous().view(
            -1, self.config.hidden_size)
        prediction_soft_label = self.region_classifier(sequence_output_masked)

        return prediction_soft_label

    def forward(self, input_ids, position_ids, txt_lens, txt_type_ids,
                img_feat, img_pos_feat, num_bbs,
                attention_mask, targets, obj_targets=None, img_masks=None,
                region_tok_ids=None, compute_loss=True):
        sequence_output = self.bert(input_ids, position_ids, txt_lens,
                                    img_feat, img_pos_feat, num_bbs,
                                    attention_mask,
                                    output_all_encoded_layers=False,
                                    txt_type_ids=txt_type_ids)
        pooled_output = self.bert.pooler(sequence_output)
        rank_scores = self.vcr_output(pooled_output)
        # rank_scores = rank_scores.reshape((-1, 4))

        if self.obj_cls and img_masks is not None:
            img_feat = mask_img_feat_for_mrc(img_feat, img_masks)
            masked_sequence_output = self.bert(
                input_ids, position_ids, txt_lens,
                img_feat, img_pos_feat, num_bbs,
                attention_mask,
                output_all_encoded_layers=False,
                txt_type_ids=txt_type_ids)
            # get only the image part
            img_sequence_output = _get_image_hidden(
                masked_sequence_output, txt_lens, num_bbs)
            # only compute masked tokens for better efficiency
            predicted_obj_label = self.masked_predict_labels(
                img_sequence_output, img_masks)

        if compute_loss:
            vcr_loss = F.cross_entropy(
                    rank_scores, targets.squeeze(-1),
                    reduction='mean')
            if self.obj_cls:
                obj_cls_loss = F.cross_entropy(
                    predicted_obj_label, obj_targets.long(),
                    ignore_index=0, reduction='mean')
            else:
                obj_cls_loss = torch.tensor([0.], device=vcr_loss.device)
            return vcr_loss, obj_cls_loss
        else:
            rank_scores = rank_scores[:, 1:]
            return rank_scores


class BertForImageTextPretrainingForVCR(BertForImageTextPretraining):
    def init_type_embedding(self):
        new_emb = nn.Embedding(4, self.bert.config.hidden_size)
        new_emb.apply(self.init_bert_weights)
        for i in [0, 1]:
            emb = self.bert.embeddings.token_type_embeddings.weight.data[i, :]
            new_emb.weight.data[i, :].copy_(emb)
        emb = self.bert.embeddings.token_type_embeddings.weight.data[0, :]
        new_emb.weight.data[2, :].copy_(emb)
        new_emb.weight.data[3, :].copy_(emb)
        self.bert.embeddings.token_type_embeddings = new_emb

    def init_word_embedding(self, num_special_tokens):
        orig_word_num = self.bert.embeddings.word_embeddings.weight.size(0)
        new_emb = nn.Embedding(
            orig_word_num + num_special_tokens, self.bert.config.hidden_size)
        new_emb.apply(self.init_bert_weights)
        emb = self.bert.embeddings.word_embeddings.weight.data
        new_emb.weight.data[:orig_word_num, :].copy_(emb)
        self.bert.embeddings.word_embeddings = new_emb
        self.cls = BertOnlyMLMHead(
            self.bert.config, self.bert.embeddings.word_embeddings.weight)

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
