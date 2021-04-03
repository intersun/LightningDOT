""" Image Text Retrieval evaluation helper """
import torch


@torch.no_grad()
def itm_eval(score_matrix, txt_ids, img_ids, txt2img, img2txts):
    # image retrieval
    img2j = {i: j for j, i in enumerate(img_ids)}
    _, rank_txt = score_matrix.topk(10, dim=1)
    gt_img_j = torch.LongTensor([img2j[txt2img[txt_id]]
                                 for txt_id in txt_ids],
                                ).to(rank_txt.device
                                     ).unsqueeze(1).expand_as(rank_txt)
    rank = (rank_txt == gt_img_j).nonzero()
    if rank.numel():
        ir_r1 = (rank < 1).sum().item() / len(txt_ids)
        ir_r5 = (rank < 5).sum().item() / len(txt_ids)
        ir_r10 = (rank < 10).sum().item() / len(txt_ids)
    else:
        ir_r1, ir_r5, ir_r10 = 0, 0, 0

    # text retrieval
    txt2i = {t: i for i, t in enumerate(txt_ids)}
    _, rank_img = score_matrix.topk(10, dim=0)
    tr_r1, tr_r5, tr_r10 = 0, 0, 0
    for j, img_id in enumerate(img_ids):
        gt_is = [txt2i[t] for t in img2txts[img_id]]
        ranks = [(rank_img[:, j] == i).nonzero() for i in gt_is]
        rank = min([10] + [r.item() for r in ranks if r.numel()])
        if rank < 1:
            tr_r1 += 1
        if rank < 5:
            tr_r5 += 1
        if rank < 10:
            tr_r10 += 1
    tr_r1 /= len(img_ids)
    tr_r5 /= len(img_ids)
    tr_r10 /= len(img_ids)

    tr_mean = (tr_r1 + tr_r5 + tr_r10) / 3
    ir_mean = (ir_r1 + ir_r5 + ir_r10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_log = {'txt_r1': tr_r1,
                'txt_r5': tr_r5,
                'txt_r10': tr_r10,
                'txt_r_mean': tr_mean,
                'img_r1': ir_r1,
                'img_r5': ir_r5,
                'img_r10': ir_r10,
                'img_r_mean': ir_mean,
                'r_mean': r_mean}
    return eval_log
