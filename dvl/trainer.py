import collections
import os
import torch
import tqdm
import logging
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, ChainDataset
from uniter_model.data.loader import PrefetchLoader

from dvl.data.itm import TxtTokLmdb, ItmFastDataset, ItmValDataset, itm_fast_collate
from dvl.models.bi_encoder import BiEncoderNllLoss
from dvl.utils import _calc_loss
from dvl.indexer.faiss_indexers import DenseFlatIndexer, DenseHNSWFlatIndexer


logger = logging.getLogger()
CheckpointState = collections.namedtuple("CheckpointState",
                                         ['model_dict', 'optimizer_dict', 'scheduler_dict', 'offset', 'epoch',
                                          'encoder_params'])


class BiEncoderTrainer:
    def __init__(self, args):
        pass


def build_dataloader(dataset, collate_fn, is_train, opts, batch_size=None):
    if batch_size is None:
        batch_size = opts.train_batch_size if is_train else opts.valid_batch_size

    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=is_train, drop_last=False,
                            num_workers=opts.n_workers,
                            pin_memory=opts.pin_mem, collate_fn=collate_fn)
    dataloader = PrefetchLoader(dataloader)
    return dataloader


def get_model_obj(model: nn.Module):
    return model.module if hasattr(model, 'module') else model


def _save_checkpoint(args, biencoder, optimizer, scheduler, epoch: int, offset: int, cp_name: str = None) -> str:
    model_to_save = get_model_obj(biencoder)
    if cp_name is None:
        cp = os.path.join(args.output_dir, 'biencoder.' + str(epoch) + ('.' + str(offset) if offset > 0 else ''))
    else:
        cp = os.path.join(args.output_dir, 'biencoder.' + cp_name)
    cp += '.pt'


    meta_params = None

    state = CheckpointState(model_to_save.state_dict(),
                            optimizer.state_dict(),
                            scheduler.state_dict(),
                            offset,
                            epoch, meta_params
                            )
    torch.save(state._asdict(), cp)
    logger.info('Saved checkpoint at %s', cp)
    return cp


def load_saved_state(biencoder, optimizer=None, scheduler=None, saved_state: CheckpointState = ''):
    epoch = saved_state.epoch
    offset = saved_state.offset
    if offset == 0:  # epoch has been completed
        epoch += 1
    logger.info('Loading checkpoint @ batch=%s and epoch=%s', offset, epoch)

    model_to_load = get_model_obj(biencoder)
    logger.info('Loading saved model state ...')
    model_to_load.load_state_dict(saved_state.model_dict)  # set strict=False if you use extra projection

    if saved_state.optimizer_dict and optimizer is not None:
        logger.info('Loading saved optimizer state ...')
        optimizer.load_state_dict(saved_state.optimizer_dict)

    if saved_state.scheduler_dict and scheduler is not None:
        scheduler_state = saved_state.scheduler_dict
        scheduler.load_state_dict(scheduler_state)


def load_states_from_checkpoint(model_file: str) -> CheckpointState:
    logger.info('Reading saved model from %s', model_file)
    state_dict = torch.load(model_file, map_location='cpu')
    logger.info('model_state_dict keys %s', state_dict.keys())
    return CheckpointState(**state_dict)


def get_indexer(bi_encoder, eval_dataloader, args, hnsw_index, img_retrieval=True):
    bi_encoder.eval()
    img_embedding = dict()

    if hnsw_index:
        indexer_img = DenseHNSWFlatIndexer(args.vector_size)   # modify in future
    else:
        indexer_img = DenseFlatIndexer(args.vector_size)  # modify in future
    for i, batch in enumerate(tqdm.tqdm(eval_dataloader)):
        with torch.no_grad():
            model_out = bi_encoder(batch)
        local_q_vector, local_ctx_vectors, local_caption_vectors = model_out
        if img_retrieval:
            img_embedding.update({img_id: img_vec.detach().cpu().numpy() for img_id, img_vec in zip(batch['img_fname'], local_ctx_vectors)})
        else:
            img_embedding.update({img_id: txt_vec.detach().cpu().numpy() for img_id, txt_vec in zip(batch['txt_index'], local_q_vector)})
    indexer_img.index_data(list(img_embedding.items()))
    return indexer_img


def eval_model_on_dataloader(bi_encoder, eval_dataloader, args, img2txt=None, num_tops=100, no_eval=False):
    total_loss = 0.0
    bi_encoder.eval()
    total_correct_predictions = 0
    batches, total_samples = 0, 0
    labels_img_name = []
    labels_txt_name = []
    img_embedding = dict()
    txt_embedding = dict()
    if args.hnsw_index:
        indexer_img = DenseHNSWFlatIndexer(args.vector_size)   # modify in future
        indexer_txt = DenseHNSWFlatIndexer(args.vector_size)   # modify in future
    else:
        indexer_img = DenseFlatIndexer(args.vector_size)  # modify in future
        indexer_txt = DenseFlatIndexer(args.vector_size)  # modify in future
    query_txt, query_txt_id = [], []
    query_img, query_img_id = [], []
    for i, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            model_out = bi_encoder(batch)
        local_q_vector, local_ctx_vectors, local_caption_vectors = model_out

        query_txt.extend([out.view(-1).detach().cpu().numpy() for out in local_q_vector])
        query_txt_id.extend(batch['txt_index'])

        query_img.extend([out.view(-1).detach().cpu().numpy() for out in local_ctx_vectors])
        query_img_id.extend(batch['img_fname'])

        loss_function = BiEncoderNllLoss()

        loss, correct_cnt, score = _calc_loss(args, loss_function, local_q_vector, local_ctx_vectors, local_caption_vectors,
                                       list(range(len(local_q_vector))), None)

        total_loss += loss.item()
        total_correct_predictions += correct_cnt.sum().item()
        batches += 1
        total_samples += batch['txts']['input_ids'].shape[0]

        img_embedding.update({img_id: img_vec.detach().cpu().numpy() for img_id, img_vec in zip(batch['img_fname'], local_ctx_vectors)})
        txt_embedding.update({img_id: txt_vec.detach().cpu().numpy() for img_id, txt_vec in zip(batch['txt_index'], local_q_vector)})
        labels_img_name.extend(batch['img_fname'])
        labels_txt_name.extend(batch['txt_index'])

    total_loss = total_loss / batches
    correct_ratio = total_correct_predictions / float(total_samples)

    query_txt_np = np.array(query_txt)
    indexer_img.index_data(list(img_embedding.items()))
    query_img_np = np.array(query_img)
    indexer_txt.index_data(list(txt_embedding.items()))

    if no_eval:
        return total_loss, correct_ratio, (indexer_img, indexer_txt), (None, None), (None, None)
    else:
        res_txt = indexer_img.search_knn(query_txt_np, num_tops)
        rank_txt_res = {query_txt_id[i]: r[0] for i, r in enumerate(res_txt)}

        res_img = indexer_txt.search_knn(query_img_np, num_tops)
        rank_img_res = {query_img_id[i]: r[0] for i, r in enumerate(res_img)}

        recall_txt = {1: 0, 5: 0, 10: 0}
        for i, q in enumerate(query_txt_id):
            for top in recall_txt:
                recall_txt[top] += labels_img_name[i] in rank_txt_res[q][:top]

        for top in recall_txt:
            recall_txt[top] = recall_txt[top] / len(rank_txt_res)

        recall_img = {1: 0, 5: 0, 10: 0}
        for i, q in enumerate(np.unique(query_img_id)):
            for top in recall_img:
                # recall_img[top] += any([txt_id in rank_img_res[q][:top] for txt_id in img2txt[q]])
                recall_img[top] += any([txt_id in rank_img_res[q][:top] for txt_id in img2txt[q]])

        for top in recall_img:
            recall_img[top] = recall_img[top] / len(rank_img_res)

        return total_loss, correct_ratio, (indexer_img, indexer_txt), (recall_txt, recall_img), (rank_txt_res, rank_img_res)


def load_dataset(all_img_dbs, txt_dbs, img_dbs, args, is_train):
    if is_train:
        # train datasets
        datasets = []
        for txt_path, img_path in zip(txt_dbs, img_dbs):
            img_db = all_img_dbs[img_path]
            txt_db = TxtTokLmdb(txt_path, args.max_txt_len)
            datasets.append(ItmFastDataset(txt_db, img_db, args.num_hard_negatives, args.img_meta, args.tokenizer))

        datasets = ConcatDataset(datasets)  #
    else:
        # eval or test
        img_db = all_img_dbs[img_dbs]
        txt_db = TxtTokLmdb(txt_dbs, -1)
        datasets = ItmFastDataset(txt_db, img_db, args.inf_minibatch_size, args.img_meta, args.tokenizer)

    return datasets
