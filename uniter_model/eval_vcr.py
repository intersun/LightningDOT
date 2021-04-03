"""run inference of VCR for submission"""
import argparse
import json
import os
from os.path import exists
from time import time

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from apex import amp
from horovod import torch as hvd

from data import (DetectFeatLmdb, VcrEvalDataset, vcr_eval_collate,
                  PrefetchLoader)
from torch.utils.data.distributed import DistributedSampler
from model import BertForVisualCommonsenseReasoning

from utils.logger import LOGGER
from utils.distributed import all_gather_list
from utils.misc import NoOp, Struct
NUM_SPECIAL_TOKENS = 81


def load_img_feat(dir_list, path2imgdir, opts):
    dir_ = dir_list.split(";")
    assert len(dir_) <= 2, "More than two img_dirs found"
    img_dir_gt, img_dir = None, None
    gt_dir_path, dir_path = "", ""
    for d in dir_:
        if "gt" in d:
            gt_dir_path = d
        else:
            dir_path = d
    if gt_dir_path != "":
        img_dir_gt = path2imgdir.get(gt_dir_path, None)
        if img_dir_gt is None:
            img_dir_gt = DetectFeatLmdb(gt_dir_path, -1,
                                        opts.max_bb, opts.min_bb, 100,
                                        opts.compressed_db)
            path2imgdir[gt_dir_path] = img_dir_gt
    if dir_path != "":
        img_dir = path2imgdir.get(dir_path, None)
        if img_dir is None:
            img_dir = DetectFeatLmdb(dir_path, opts.conf_th,
                                     opts.max_bb, opts.min_bb, opts.num_bb,
                                     opts.compressed_db)
            path2imgdir[dir_path] = img_dir
    return img_dir, img_dir_gt, path2imgdir


def main(opts):
    hvd.init()
    n_gpu = hvd.size()
    device = torch.device("cuda", hvd.local_rank())
    torch.cuda.set_device(hvd.local_rank())
    rank = hvd.rank()
    opts.rank = rank
    LOGGER.info("device: {} n_gpu: {}, rank: {}, "
                "16-bits training: {}".format(
                    device, n_gpu, hvd.rank(), opts.fp16))

    hps_file = f'{opts.output_dir}/log/hps.json'
    model_opts = Struct(json.load(open(hps_file)))

    path2imgdir = {}
    # load DBs and image dirs
    val_img_dir, val_img_dir_gt, path2imgdir = load_img_feat(
            opts.img_dir, path2imgdir, model_opts)
    eval_dataset = VcrEvalDataset("test", opts.txt_db,
                                  val_img_dir_gt, val_img_dir,
                                  max_txt_len=-1)

    # Prepare model
    bert_model = json.load(open(f'{opts.txt_db}/meta.json'))['bert']
    model = BertForVisualCommonsenseReasoning.from_pretrained(
        bert_model, img_dim=2048, obj_cls=False,
        state_dict={})
    model.init_type_embedding()
    model.init_word_embedding(NUM_SPECIAL_TOKENS)
    if exists(opts.checkpoint):
        ckpt_file = opts.checkpoint
    else:
        ckpt_file = f'{opts.output_dir}/ckpt/model_step_{opts.checkpoint}.pt'
    checkpoint = torch.load(ckpt_file)
    state_dict = checkpoint.get('model_state', checkpoint)
    matched_state_dict = {}
    unexpected_keys = set()
    missing_keys = set()
    for name, param in model.named_parameters():
        missing_keys.add(name)
    for key, data in state_dict.items():
        if key in missing_keys:
            matched_state_dict[key] = data
            missing_keys.remove(key)
        else:
            unexpected_keys.add(key)
    print("Unexpected_keys:", list(unexpected_keys))
    print("Missing_keys:", list(missing_keys))
    model.load_state_dict(matched_state_dict, strict=False)
    if model_opts.cut_bert != -1:
        # cut some layers of BERT
        model.bert.encoder.layer = torch.nn.ModuleList(
            model.bert.encoder.layer[:model_opts.cut_bert])
    model.to(device)
    if opts.fp16:
        model = amp.initialize(model, enabled=opts.fp16, opt_level='O2')

    sampler = DistributedSampler(
        eval_dataset, num_replicas=n_gpu, rank=rank)
    eval_dataloader = DataLoader(eval_dataset,
                                 batch_size=opts.batch_size,
                                 sampler=sampler,
                                 num_workers=opts.n_workers,
                                 pin_memory=opts.pin_mem,
                                 collate_fn=vcr_eval_collate)
    eval_dataloader = PrefetchLoader(eval_dataloader)

    val_log, results = evaluate(model, eval_dataloader)
    result_dir = f'{opts.output_dir}/results_{opts.split}'
    if not exists(result_dir) and rank == 0:
        os.makedirs(result_dir)
    # dummy sync
    _ = None
    all_gather_list(_)
    if n_gpu > 1:
        with open(f'{opts.output_dir}/results_test/'
                  f'results_{opts.checkpoint}_rank{rank}.json',
                  'w') as f:
            json.dump(results, f)
        # dummy sync
        _ = None
        all_gather_list(_)
    # join results
    if n_gpu > 1:
        results = []
        for rank in range(n_gpu):
            results.extend(json.load(open(
                f'{opts.output_dir}/results_test/'
                f'results_{opts.checkpoint}_rank{rank}.json')))
    if rank == 0:
        with open(f'{opts.output_dir}/results_test/'
                  f'results_{opts.checkpoint}_all.json', 'w') as f:
            json.dump(results, f)


def compute_accuracies(out_qa, labels_qa, out_qar, labels_qar):
    outputs_qa = out_qa.max(dim=-1)[1]
    outputs_qar = out_qar.max(dim=-1)[1]
    matched_qa = outputs_qa.squeeze() == labels_qa.squeeze()
    matched_qar = outputs_qar.squeeze() == labels_qar.squeeze()
    matched_joined = matched_qa & matched_qar
    n_correct_qa = matched_qa.sum().item()
    n_correct_qar = matched_qar.sum().item()
    n_correct_joined = matched_joined.sum().item()
    return n_correct_qa, n_correct_qar, n_correct_joined


@torch.no_grad()
def evaluate(model, val_loader):
    if hvd.rank() == 0:
        val_pbar = tqdm(total=len(val_loader))
    else:
        val_pbar = NoOp()
        LOGGER.info(f"start running evaluation ...")
    model.eval()
    val_qa_loss, val_qar_loss = 0, 0
    tot_qa_score, tot_qar_score, tot_score = 0, 0, 0
    n_ex = 0
    st = time()
    results = {}
    for i, batch in enumerate(val_loader):
        qids, *inputs, qa_targets, qar_targets, _ = batch
        scores = model(
            *inputs, targets=None, compute_loss=False)
        scores = scores.view(len(qids), -1)
        if torch.max(qa_targets) > -1:
            vcr_qa_loss = F.cross_entropy(
                scores[:, :4], qa_targets.squeeze(-1), reduction="sum")
            if scores.shape[1] > 8:
                qar_scores = []
                for batch_id in range(scores.shape[0]):
                    answer_ind = qa_targets[batch_id].item()
                    qar_index = [4+answer_ind*4+i
                                 for i in range(4)]
                    qar_scores.append(scores[batch_id, qar_index])
                qar_scores = torch.stack(qar_scores, dim=0)
            else:
                qar_scores = scores[:, 4:]
            vcr_qar_loss = F.cross_entropy(
                qar_scores, qar_targets.squeeze(-1), reduction="sum")
            val_qa_loss += vcr_qa_loss.item()
            val_qar_loss += vcr_qar_loss.item()

            curr_qa_score, curr_qar_score, curr_score = compute_accuracies(
                scores[:, :4], qa_targets, qar_scores, qar_targets)
            tot_qar_score += curr_qar_score
            tot_qa_score += curr_qa_score
            tot_score += curr_score
        for qid, score in zip(qids, scores):
            results[qid] = score.cpu().tolist()
        n_ex += len(qids)
        val_pbar.update(1)
    val_qa_loss = sum(all_gather_list(val_qa_loss))
    val_qar_loss = sum(all_gather_list(val_qar_loss))
    tot_qa_score = sum(all_gather_list(tot_qa_score))
    tot_qar_score = sum(all_gather_list(tot_qar_score))
    tot_score = sum(all_gather_list(tot_score))
    n_ex = sum(all_gather_list(n_ex))
    tot_time = time()-st
    val_qa_loss /= n_ex
    val_qar_loss /= n_ex
    val_qa_acc = tot_qa_score / n_ex
    val_qar_acc = tot_qar_score / n_ex
    val_acc = tot_score / n_ex
    val_log = {f'valid/vcr_qa_loss': val_qa_loss,
               f'valid/vcr_qar_loss': val_qar_loss,
               f'valid/acc_qa': val_qa_acc,
               f'valid/acc_qar': val_qar_acc,
               f'valid/acc': val_acc,
               f'valid/ex_per_s': n_ex/tot_time}
    model.train()
    LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                f"score_qa: {val_qa_acc*100:.2f} "
                f"score_qar: {val_qar_acc*100:.2f} "
                f"score: {val_acc*100:.2f} ")
    return val_log, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--txt_db",
                        default=None, type=str,
                        help="The input train corpus. (LMDB)")
    parser.add_argument("--img_dir",
                        default=None, type=str,
                        help="The input train images.")
    parser.add_argument('--compressed_db', action='store_true',
                        help='use compressed LMDB')
    parser.add_argument("--split",
                        default="test", type=str,
                        help="The input split")
    parser.add_argument("--checkpoint",
                        default=None, type=str,
                        help="pretrained model (can take 'google-bert') ")
    parser.add_argument("--batch_size",
                        default=10, type=int,
                        help="number of tokens in a batch")
    parser.add_argument(
        "--output_dir", default=None, type=str,
        help="The output directory where the model checkpoints will be "
             "written.")
    # device parameters
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead "
                             "of 32-bit")
    parser.add_argument('--n_workers', type=int, default=4,
                        help="number of data workers")
    parser.add_argument('--pin_mem', action='store_true',
                        help="pin memory")

    args = parser.parse_args()

    main(args)
