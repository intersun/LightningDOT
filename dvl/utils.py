import logging
import random
import tqdm
import torch
import pickle

import torch.distributed as dist

from collections import defaultdict
from horovod import torch as hvd
from torch import Tensor as T
from typing import Tuple


logger = logging.getLogger()


def get_rank():
    return hvd.rank()


def get_world_size():
    return hvd.size()


def print_args(args):
    logger.info(" **************** CONFIGURATION **************** ")
    for key, val in sorted(vars(args).items()):
        keystr = "{}".format(key) + (" " * (30 - len(key)))
        logger.info("%s -->   %s", keystr, val)
    logger.info(" **************** CONFIGURATION **************** ")


def num_of_parameters(model, requires_grad=False):
    if requires_grad:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def get_default_group():
    return dist.group.WORLD


def all_reduce(tensor, group=None):
    if group is None:
        group = get_default_group()
    return dist.all_reduce(tensor, group=group)


def all_gather_list(data, group=None, max_size=16384):
    """Gathers arbitrary data from all nodes into a list.
    Similar to :func:`~torch.distributed.all_gather` but for arbitrary Python
    data. Note that *data* must be picklable.
    Args:
        data (Any): data from the local worker to be gathered on other workers
        group (optional): group of the collective
    """
    SIZE_STORAGE_BYTES = 4  # int32 to encode the payload size

    enc = pickle.dumps(data)
    enc_size = len(enc)

    if enc_size + SIZE_STORAGE_BYTES > max_size:
        raise ValueError(
            'encoded data exceeds max_size, this can be fixed by increasing buffer size: {}'.format(enc_size))

    rank = get_rank()
    world_size = get_world_size()
    buffer_size = max_size * world_size

    if not hasattr(all_gather_list, '_buffer') or \
            all_gather_list._buffer.numel() < buffer_size:
        all_gather_list._buffer = torch.cuda.ByteTensor(buffer_size)
        all_gather_list._cpu_buffer = torch.ByteTensor(max_size).pin_memory()

    buffer = all_gather_list._buffer
    buffer.zero_()
    cpu_buffer = all_gather_list._cpu_buffer

    assert enc_size < 256 ** SIZE_STORAGE_BYTES, 'Encoded object size should be less than {} bytes'.format(
        256 ** SIZE_STORAGE_BYTES)

    size_bytes = enc_size.to_bytes(SIZE_STORAGE_BYTES, byteorder='big')

    cpu_buffer[0:SIZE_STORAGE_BYTES] = torch.ByteTensor(list(size_bytes))
    cpu_buffer[SIZE_STORAGE_BYTES: enc_size + SIZE_STORAGE_BYTES] = torch.ByteTensor(list(enc))

    start = rank * max_size
    size = enc_size + SIZE_STORAGE_BYTES
    buffer[start: start + size].copy_(cpu_buffer[:size])

    all_reduce(buffer, group=group)

    try:
        result = []
        for i in range(world_size):
            out_buffer = buffer[i * max_size: (i + 1) * max_size]
            size = int.from_bytes(out_buffer[0:SIZE_STORAGE_BYTES], byteorder='big')
            if size > 0:
                result.append(pickle.loads(bytes(out_buffer[SIZE_STORAGE_BYTES: size + SIZE_STORAGE_BYTES].tolist())))
        return result
    except pickle.UnpicklingError:
        raise Exception(
            'Unable to unpickle data from other workers. all_gather_list requires all '
            'workers to enter the function together, so this error usually indicates '
            'that the workers have fallen out of sync somehow. Workers can fall out of '
            'sync if one of them runs out of memory, or if there are other conditions '
            'in your training script that can cause one worker to finish an epoch '
            'while other workers are still iterating over their portions of the data.'
        )


def _calc_loss(args, loss_function, local_q_vector, local_ctx_vectors, local_caption_vectors, local_positive_idxs,
               local_hard_negatives_idxs: list = None, experiment=None
               ):
    """
    Calculates In-batch negatives schema loss and supports to run it in DDP mode by exchanging the representations
    across all the nodes.
    """
    distributed_world_size = 1  # args.distributed_world_size or 1
    if distributed_world_size > 1:
        # TODO: Add local_caption_vectors
        q_vector_to_send = torch.empty_like(local_q_vector).cpu().copy_(local_q_vector).detach_()
        ctx_vector_to_send = torch.empty_like(local_ctx_vectors).cpu().copy_(local_ctx_vectors).detach_()

        global_question_ctx_vectors = all_gather_list(
            [q_vector_to_send, ctx_vector_to_send, local_positive_idxs, local_hard_negatives_idxs],
            max_size=args.global_loss_buf_sz)

        global_q_vector = []
        global_ctxs_vector = []

        # ctxs_per_question = local_ctx_vectors.size(0)
        positive_idx_per_question = []
        hard_negatives_per_question = []

        total_ctxs = 0

        for i, item in enumerate(global_question_ctx_vectors):
            q_vector, ctx_vectors, positive_idx, hard_negatives_idxs = item

            if i != args.local_rank:
                global_q_vector.append(q_vector.to(local_q_vector.device))
                global_ctxs_vector.append(ctx_vectors.to(local_q_vector.device))
                positive_idx_per_question.extend([v + total_ctxs for v in positive_idx])
                hard_negatives_per_question.extend([[v + total_ctxs for v in l] for l in hard_negatives_idxs])
            else:
                global_q_vector.append(local_q_vector)
                global_ctxs_vector.append(local_ctx_vectors)
                positive_idx_per_question.extend([v + total_ctxs for v in local_positive_idxs])
                hard_negatives_per_question.extend([[v + total_ctxs for v in l] for l in local_hard_negatives_idxs])
            total_ctxs += ctx_vectors.size(0)

        global_q_vector = torch.cat(global_q_vector, dim=0)
        global_ctxs_vector = torch.cat(global_ctxs_vector, dim=0)

    else:
        global_q_vector = local_q_vector
        global_ctxs_vector = local_ctx_vectors
        global_caption_vector = local_caption_vectors
        positive_idx_per_question = local_positive_idxs
        hard_negatives_per_question = local_hard_negatives_idxs

    loss, is_correct, scores = loss_function.calc(global_q_vector, global_ctxs_vector, global_caption_vector,
                                                  positive_idx_per_question, hard_negatives_per_question,
                                                  args.caption_score_weight, experiment)

    return loss, is_correct, scores


def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')


def is_main_process():
    return hvd.rank() == 0


def display_img(img_meta, name, img_only=False):
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    img = mpimg.imread(img_meta[name]['img_file'])
    plt.imshow(img)
    plt.show()
    if not img_only:
        print('annotation')
        print('\t' + '\n\t'.join(img_meta[name]['annotation']))
        print('caption')
        print('\t' + img_meta[name]['caption'][0])


def retrieve_query(model, query, indexer, args, top=10):
    input_ids = args.tokenizer.encode(query)
    input_ids = torch.LongTensor(input_ids).to(args.device).unsqueeze(0)
    attn_mask = torch.ones(len(input_ids[0]), dtype=torch.long, device=args.device).unsqueeze(0)
    pos_ids = torch.arange(len(input_ids[0]), dtype=torch.long, device=args.device).unsqueeze(0)
    _, query_vector, _ = model.txt_model(input_ids=input_ids,attention_mask=attn_mask, position_ids=pos_ids)
    res = indexer.search_knn(query_vector.detach().cpu().numpy(), 100)
    return res


def get_model_encoded_vecs(model, dataloader):
    img_embedding, caption_embedding, query_embedding = dict(), dict(), defaultdict(list)
    labels_img_name = []
    # for i, batch in enumerate(dataloader):
    for i, batch in enumerate(tqdm.tqdm(dataloader)):
        with torch.no_grad():
            model_out = model(batch)
        local_q_vectors, local_ctx_vectors, local_caption_vectors = model_out

        img_embedding.update({img_id: img_vec.detach().cpu().numpy() for img_id, img_vec in zip(batch['img_fname'], local_ctx_vectors)})
        caption_embedding.update({img_id: cap_vec.detach().cpu().numpy() for img_id, cap_vec in zip(batch['img_fname'], local_caption_vectors)})
        query_embedding.update({img_id: cap_vec.detach().cpu().numpy() for img_id, cap_vec in zip(batch['txt_index'], local_q_vectors)})

    labels_img_name.extend(batch['img_fname'])
    return {
        'img_embed': img_embedding,
        'caption_embed': caption_embedding,
        'txt_embed': query_embedding,
        'img_name': labels_img_name
    }

