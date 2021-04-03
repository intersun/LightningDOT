"""
preprocess COCO annotations into LMDB
"""
import argparse
from collections import defaultdict
import json
import os
from os.path import basename, exists
import pickle
import re

from cytoolz import curry
from tqdm import tqdm
from pytorch_pretrained_bert import BertTokenizer

from utils.vqa import compute_target
from utils.visual_entailment import compute_target as compute_target_ve
from data.data import open_lmdb


IN_WORD = '@@'


@curry
def bert_tokenize(tokenizer, text):
    """ reconstructable tokenization for possible generation """
    if text == ('this house is leaning out to wards '
                'the road taken in cambridge@ @@@@'):
        # SBU special case
        text = text.replace('@@', '')
    assert IN_WORD not in text
    ids = []
    words = []
    for word in text.strip().split():
        ws = tokenizer.tokenize(word)
        if not ws:
            # some special char in conceptual caption
            continue
        words.append(ws[0])
        for w in ws[1:]:
            words.append(f'{IN_WORD}{w}')
        ids.extend(tokenizer.convert_tokens_to_ids(ws))
    return ids, words


@curry
def bert_tokenize_for_vcr(tokenizer, special_tokens, text, txt_region_tokens):
    """ reconstructable tokenization for possible generation """
    assert IN_WORD not in text
    ids = []
    words = []
    special_tokens_dict = {val: ind for ind, val in enumerate(special_tokens)}
    toked_txt_region_tokens = []
    index = 0
    for word in text.strip().split():
        if word in special_tokens_dict:
            words.append(word)
            ids.extend([len(tokenizer.vocab)+special_tokens_dict[word]])
            toked_txt_region_tokens.append(txt_region_tokens[index])
        else:
            ws = tokenizer.tokenize(word)
            words.append(ws[0])
            toked_txt_region_tokens.append(txt_region_tokens[index])
            for w in ws[1:]:
                words.append(f'{IN_WORD}{w}')
                toked_txt_region_tokens.append(txt_region_tokens[index])
            ids.extend(tokenizer.convert_tokens_to_ids(ws))
        index += 1
    return ids, words, toked_txt_region_tokens


def _norm_text(text):
    norm_text = re.sub(r"([.,'!?\"()*#:;])", '', text.lower()
                       ).replace('-', ' ').replace('/', ' ')
    return norm_text


def make_word2id(texts):
    word2id = {'PAD': 0, 'UNK': 1}
    for text in texts:
        for w in _norm_text(text).split():
            if w not in word2id:
                word2id[w] = len(word2id)
    return word2id


def gen_vqa_texts(annotation):
    questions = json.load(open(annotation))['questions']
    for q in questions:
        yield q['question']


def gen_ve_texts(annotation):
    contents = open(annotation, "r").read()
    hypotheses = [json.loads(str(item))
                  for item in contents.strip().split('\n')]
    for h in hypotheses:
        yield h['sentence2']


def gen_itm_texts(annotation):
    data = json.load(open(annotation))
    for q in data:
        for s in q["sentences"]:
            yield s['raw']


@curry
def _get_coco_fname(id_, split):
    fname = f'coco_{split}_{id_:012}.npz'
    return fname


def _get_vg_fname(id_):
    fname = f'vg_{int(id_):012}.npz'
    return fname


def _get_gqa_fname(id_):
    if "n" not in id_:
        fname = f'gqa_{int(id_):012}.npz'
    else:
        fname = f'gqa_{id_}.npz'
    return fname


def _get_flickr_fname(id_):
    fname = f'flickr30k_{id_:012}.npz'
    return fname


def _get_vcr_fname(id_, split):
    fname_gt = f'vcr_gt_{split}_{id_}.npz'
    fname = f'vcr_{split}_{id_}.npz'
    return fname_gt, fname


def process_vqa(questions, answers, ans2label, db, tokenizer, split):
    """
    Inputs:
    - questions : [{image_id, question, question_id}]
    - answers   : [{answers, image_id, question_id,
                    question_type, answer_type}]
    - ans2label : ans -> ans_id
    - db
    - tokenizer
    - split
    Return:
    - id2len   : qid -> tokenized question length
    - txt2img  : qid -> img(feature) filename
    - img2txts : img(feature) filename -> [qid]
    Besides, we write into db[qid]:
    - toked_question : [tokens]
    - input_ids      : [wd_ids]
    - img_fname      : img(feature) filename
    - target         : {labels, scores}
    """
    id2len = {}
    txt2img = {}
    img2txts = defaultdict(list)
    if split == 'vg':
        get_img_fname = _get_vg_fname
    elif split == 'gqa':
        get_img_fname = _get_gqa_fname
    else:
        get_img_fname = _get_coco_fname(split=split)
    for q in tqdm(questions, desc='processing VQA questions'):
        qid = str(q['question_id'])
        input_ids, toked_question = tokenizer(q['question'])
        id2len[qid] = len(input_ids)
        img_fname = get_img_fname(q['image_id'])
        txt2img[qid] = img_fname
        img2txts[img_fname].append(qid)
        q['toked_question'] = toked_question
        q['input_ids'] = input_ids
        q['img_fname'] = img_fname
        db[qid] = q
    if answers is not None:
        for a in tqdm(answers, desc='processing VQA answers'):
            qid = str(a['question_id'])
            q = db[qid]
            assert q['question_id'] == a['question_id']
            assert q['image_id'] == a['image_id']
            for k, v in a.items():
                q[k] = v
            q['target'] = compute_target(a['answers'], ans2label)
            db[qid] = q
    return id2len, txt2img, img2txts


def process_referring_expressions(refs, instances, iid_to_ann_ids,
                                  db, tokenizer, split):
    """
    Inputs:
    - refs: [ref_id, ann_id, image_id, split, sent_ids, sentences]
    - instances: {images, annotations, categories}
    - iid_to_ann_ids: image_id -> ann_ids ordered by extracted butd features
    Return:
    - id2len : sent_id -> tokenized question length
    - images : [{id, file_name, ann_ids, height, width} ]
    - annotations: [{id, area, bbox, image_id, category_id, iscrowd}]
    - categories : [{id, name, supercategory}]
    """
    # images within split
    image_set = set([ref['image_id'] for ref in refs if ref['split'] == split])
    images = []
    for img in instances['images']:
        if img['id'] in image_set:
            images.append({'id': img['id'], 'file_name': img['file_name'],
                           'ann_ids': iid_to_ann_ids[str(img['id'])],
                           'height': img['height'], 'width': img['width']})
    # anns within split
    annotations = []
    for ann in instances['annotations']:
        if ann['image_id'] in image_set:
            annotations.append({
                'id': ann['id'], 'area': ann['area'], 'bbox': ann['bbox'],
                'image_id': ann['image_id'], 'category_id': ann['category_id'],
                'iscrowd': ann['iscrowd']
            })
    Anns = {ann['id']: ann for ann in annotations}
    # category info
    categories = instances['categories']
    # refs within split
    refs = [ref for ref in refs if ref['split'] == split]
    id2len = {}
    for ref in tqdm(refs, desc='processing referring expressions'):
        ref_id = ref['ref_id']
        ann_id = ref['ann_id']
        image_id = ref['image_id']
        for sent in ref['sentences']:
            sent_id = sent['sent_id']
            input_ids, toked_sent = tokenizer(sent['sent'])
            id2len[str(sent_id)] = len(input_ids)
            db[str(sent_id)] = {
                'sent_id': sent_id, 'sent': sent['sent'],
                'ref_id': ref_id, 'ann_id': ann_id, 'image_id': image_id,
                'bbox': Anns[ann_id]['bbox'],
                'input_ids': input_ids, 'toked_sent': toked_sent}
    return id2len, images, annotations, categories, refs


def process_gqa(questions, db, tokenizer, split):
    id2len = {}
    txt2img = {}
    img2txts = defaultdict(list)
    get_img_fname = _get_gqa_fname
    for qid, q in tqdm(questions.items(),
                       desc=f'processing GQA_{split} questions'):
        input_ids, toked_question = tokenizer(q['question'])
        id2len[qid] = len(input_ids)
        img_fname = get_img_fname(q['imageId'])
        txt2img[qid] = img_fname
        img2txts[img_fname].append(qid)
        q['toked_question'] = toked_question
        q['input_ids'] = input_ids
        q['img_fname'] = img_fname
        input_ids_a, toked_a = tokenizer(q['fullAnswer'])
        id2len[qid] += len(input_ids_a)
        q['input_ids_a'] = input_ids_a
        q['toked_answers'] = toked_a
        db[qid] = q
    return id2len, txt2img, img2txts


def process_nlvr2(jsonl, db, tokenizer, imgs=None):
    id2len = {}
    txt2img = {}  # not sure if useful
    img2txts = defaultdict(list)  # not sure if useful
    for line in tqdm(jsonl, desc='processing NLVR2'):
        example = json.loads(line)
        id_ = example['identifier']
        img_id = '-'.join(id_.split('-')[:-1])
        img_fname = (f'nlvr2_{img_id}-img0.npz', f'nlvr2_{img_id}-img1.npz')
        if imgs is not None:
            if not all(img in imgs for img in img_fname):
                continue
        input_ids, toked_question = tokenizer(example['sentence'])
        target = 1 if example['label'] == 'True' else 0
        id2len[id_] = len(input_ids)
        txt2img[id_] = img_fname
        for fname in img_fname:
            img2txts[fname].append(id_)
        example['toked_question'] = toked_question
        example['input_ids'] = input_ids
        example['img_fname'] = img_fname
        example['target'] = target
        db[id_] = example
    return id2len, txt2img, img2txts


def process_visual_entailment(hypotheses, ans2label, db, tokenizer):
    id2len = {}
    txt2img = {}
    img2txts = defaultdict(list)
    for h in tqdm(hypotheses, desc='processing visaul entailment hypotheses'):
        hid = h['pairID']
        h['image_id'] = int(h["Flikr30kID"].split(".")[0])
        input_ids, toked_hypothesis = tokenizer(h['sentence2'])
        id2len[hid] = len(input_ids)
        img_fname = _get_flickr_fname(h['image_id'])
        txt2img[hid] = img_fname
        img2txts[img_fname].append(hid)
        h['toked_hypothesis'] = toked_hypothesis
        h['input_ids'] = input_ids
        h['target'] = compute_target_ve([h['gold_label']], ans2label)
        h['img_fname'] = img_fname
        db[hid] = h

    return id2len, txt2img, img2txts


def process_caption(data, db, tokenizer, split):
    id2len = {}
    txt2img = {}
    img2txts = defaultdict(list)
    for q in tqdm(data['annotations'], desc='processing COCO captions'):
        id_ = str(q['id'])
        input_ids, toked_caption = tokenizer(q['caption'])
        id2len[id_] = len(input_ids)
        img_fname = _get_coco_fname(q['image_id'], split)
        txt2img[id_] = img_fname
        img2txts[img_fname].append(id_)
        q['toked_caption'] = toked_caption
        q['input_ids'] = input_ids
        q['img_fname'] = img_fname
        db[id_] = q
    return id2len, txt2img, img2txts


def process_conceptual_caption(tsv, imgs, db, tokenizer, split):
    id2len = {}
    txt2img = {}
    img2txts = defaultdict(list)
    for line in tqdm(tsv, desc='processing conceptual captions'):
        fields = line.strip().split('\t')
        assert len(fields) == 4
        id_, _, caption, success = fields
        if success == 'fail':
            continue
        assert success == 'success'
        input_ids, toked_caption = tokenizer(caption)
        assert input_ids  # safeguard for empty text
        img_fname = f'gcc_{split}_{int(id_):012}.npz'
        if img_fname not in imgs:
            continue
        id2len[id_] = len(input_ids)
        txt2img[id_] = img_fname
        img2txts[img_fname].append(id_)
        db[id_] = {'id': id_,
                   'toked_caption': toked_caption,
                   'input_ids': input_ids,
                   'img_fname': img_fname}
    return id2len, txt2img, img2txts


def process_sbu_caption(data, db, tokenizer):
    id2len = {}
    txt2img = {}
    img2txts = defaultdict(list)
    for ex in tqdm(data, desc='processing SBU captions'):
        if ex['file_path'] == '0347/565.jpg':
            # special case for corrupted image
            continue
        id_ = ex['iid']
        input_ids, toked_caption = tokenizer(ex['sent'])
        assert input_ids  # safeguard for empty text
        try:
            # FIXME sbu feature extraction bug
            id_ = str(int(id_))
        except ValueError:
            pass
        img_fname = f'sbu_{id_}.npz'
        id2len[id_] = len(input_ids)
        txt2img[id_] = img_fname
        img2txts[img_fname].append(id_)
        db[id_] = {'id': id_,
                   'toked_caption': toked_caption,
                   'input_ids': input_ids,
                   'img_fname': img_fname}
    return id2len, txt2img, img2txts


def process_image_text_retrieval(data, db, tokenizer, dataset, split):
    id2len = {}
    txt2img = {}
    img2txts = defaultdict(list)
    if dataset == 'coco':
        _get_img_fname = _get_coco_fname(split=split)
    elif dataset == 'flickr':
        _get_img_fname = _get_flickr_fname
    else:
        raise ValueError('unrecognized data')
    for q in tqdm(data, desc=f'processing image_text_retrieval for {split}'):
        filename = q["filename"].split(".jpg")[0]
        image_id = (int(filename.split("_")[-1]) if re.search('[a-zA-Z]',
                                                              filename)
                    else int(filename))
        img_fname = _get_img_fname(image_id)
        for s in q["sentences"]:
            s['image_id'] = image_id
            id_ = str(s['sentid'])
            txt2img[id_] = img_fname
            img2txts[img_fname].append(id_)
            input_ids, toked_caption = tokenizer(s['raw'])
            id2len[id_] = len(input_ids)
            s['toked_caption'] = toked_caption
            s['input_ids'] = input_ids
            s['img_fname'] = img_fname
            db[id_] = s
    return id2len, txt2img, img2txts


def process_caption_licheng_cleaned(data, db, tokenizer, split="COCO"):
    """
    Inputs:
    - data      : [{id, dataset, split, sent, bbox,
                    dataset_image_id, file_path}]
    - db
    - tokenizer
    - split
    Return:
    - id2len    : id -> tokenized caption length
    - txt2img   : id -> img(feature) filenamee
    - img2txts  : img(feature) filename -> id(s)
    We will also write to db[id]:
    - image_id
    - toked_caption : [tokens]
    - input_ids     : [wd_ids]
    - img_fname     : img(feature) filename
    """
    id2len = {}
    txt2img = {}
    img2txts = defaultdict(list)
    for q in tqdm(data, desc='processing licheng collected captions '
                             f'for split: {split}'):
        id_ = str(q['id'])
        input_ids, toked_caption = tokenizer(q['sent'])
        id2len[id_] = len(input_ids)
        if q['dataset'] == 'vg':
            img_fname = _get_vg_fname(q['dataset_image_id'])
        else:
            assert q['dataset'] == 'coco'
            img_split = basename(q['file_path']).split('_')[1]
            img_fname = _get_coco_fname(q['dataset_image_id'], img_split)
        txt2img[id_] = img_fname
        img2txts[img_fname].append(id_)
        q['image_id'] = q['dataset_image_id']
        q['toked_caption'] = toked_caption
        q['input_ids'] = input_ids
        q['img_fname'] = img_fname
        db[id_] = q
    return id2len, txt2img, img2txts


def process_vcr_text(tokened_txt, objects, special_tokens):
    text_region_tokens = []
    image_region_tokens = [0]*len(objects)
    words = []
    for w in tokened_txt:
        if isinstance(w, str):
            word_splits = w.split(" ")
            for splited_w in word_splits:
                words.append(splited_w)
                text_region_tokens.append(0)
        else:
            for index in w:
                text_region_tokens.append(index+1)
                image_region_tokens[index] = index+1
                object_name = objects[index]
                if "person" in object_name:
                    object_name = f"{object_name}_{index}"
                    if object_name not in special_tokens:
                        special_tokens.append(object_name)
                words.append(object_name)
    return " ".join(words), image_region_tokens, text_region_tokens


def process_vcr_obj_categories(objects, object2ids):
    output_ids = []
    for obj in objects:
        output_ids.append(object2ids[obj]+1)
    return output_ids


def process_vcr(data, db, tokenizer, split, object2ids):
    id2len_qa = {}
    id2len_qar = {}
    txt2img = {}
    img2txts = defaultdict(list)
    special_tokens = [f"person_{i}" for i in range(81)]
    for q in tqdm(data, desc='processing VCR %s questions' % split):
        filename, file_extension = os.path.splitext(
            q["img_fn"].split("/")[-1])
        q["image_id"] = filename
        q['qa_target'] = q["answer_label"] if "answer_label" in q else -1
        q["qar_target"] = q["rationale_label"] \
            if "rationale_label" in q else -1
        qid = str(q['annot_id'])
        q["raw_q"], image_region_tokens,  txt_region_tokens = process_vcr_text(
            q["question"], q["objects"], special_tokens)
        q["image_region_tokens"] = image_region_tokens
        input_ids, toked_question, toked_txt_region_tokens = tokenizer(
            special_tokens, q["raw_q"], txt_region_tokens)
        object_ids = process_vcr_obj_categories(q["objects"], object2ids)
        q["object_ids"] = object_ids
        q['toked_question'] = toked_question
        q['input_ids'] = input_ids
        q['toked_txt_region_tokens'] = toked_txt_region_tokens
        q["raw_as"] = []
        q["raw_rs"] = []
        img_fname_gt, img_fname = _get_vcr_fname(q['image_id'], split)
        txt2img[qid] = [img_fname_gt, img_fname]
        img2txts[img_fname].append(qid)
        img2txts[img_fname_gt].append(qid)

        input_ids_as = []
        toked_as = []
        input_ids_rs = []
        toked_rs = []
        toked_txt_region_tokens_a = []
        toked_txt_region_tokens_r = []
        max_qa_len = 0
        for ans in q["answer_choices"]:
            raw_ans, _, txt_region_tokens = process_vcr_text(
                ans, q["objects"], special_tokens)
            q["raw_as"].append(raw_ans)
            input_ids_a, toked_a, toked_txt_region_tokens = tokenizer(
                special_tokens, raw_ans, txt_region_tokens)
            if len(input_ids_a) > max_qa_len:
                max_qa_len = len(input_ids_a)
            input_ids_as.append(input_ids_a)
            toked_as.append(toked_a)
            toked_txt_region_tokens_a.append(toked_txt_region_tokens)
        id2len_qa[qid] = (len(input_ids)+max_qa_len)*4

        max_r_len = 0
        for r in q["rationale_choices"]:
            raw_r, _, txt_region_tokens = process_vcr_text(
                r, q["objects"], special_tokens)
            q["raw_rs"].append(raw_r)
            input_ids_r, toked_r, toked_txt_region_tokens = tokenizer(
                special_tokens, raw_r, txt_region_tokens)
            if len(input_ids_r) > max_r_len:
                max_r_len = len(input_ids_r)
            input_ids_rs.append(input_ids_r)
            toked_rs.append(toked_r)
            toked_txt_region_tokens_r.append(toked_txt_region_tokens)
        id2len_qar[qid] = id2len_qa[qid]+max_r_len
        q['img_fname'] = [img_fname_gt, img_fname]
        q['toked_as'] = toked_as
        q['toked_txt_region_tokens_a'] = toked_txt_region_tokens_a
        q['input_ids_as'] = input_ids_as
        q['toked_rs'] = toked_rs
        q['input_ids_rs'] = input_ids_rs
        q['toked_txt_region_tokens_r'] = toked_txt_region_tokens_r
        db[qid] = q
    return id2len_qa, id2len_qar, txt2img, img2txts, special_tokens


def _get_img_split(annotation):
    for split in ['train2014', 'val2014', 'test2015', 'test-dev2015']:
        if split in annotation:
            img_split = split
            break
    else:
        if ('vg' in annotation.lower()
                or 'genome' in annotation.lower()):
            img_split = 'vg'
        elif 'gqa' in annotation.lower():
            if ('test' in annotation.lower()
                    or 'submission' in annotation.lower()):
                img_split = 'gqa'
            else:
                img_split = 'vg'
        elif 'val' in annotation.lower():
            img_split = 'val2014'
        elif 'train' in annotation.lower():
            img_split = 'train2014'
        else:
            raise ValueError('cannot identify split')
    if img_split == 'test-dev2015':
        img_split = 'test2015'
    return img_split


def main(opts):
    if not exists(opts.output):
        os.makedirs(opts.output)
    else:
        raise ValueError('Found existing DB. Please explicitly remove '
                         'for re-processing')
    meta = vars(opts)
    toker = BertTokenizer.from_pretrained(
        opts.bert, do_lower_case='uncased' in opts.bert)
    tokenizer = bert_tokenize(toker)
    meta['UNK'] = toker.convert_tokens_to_ids(['[UNK]'])[0]
    meta['CLS'] = toker.convert_tokens_to_ids(['[CLS]'])[0]
    meta['SEP'] = toker.convert_tokens_to_ids(['[SEP]'])[0]
    meta['MASK'] = toker.convert_tokens_to_ids(['[MASK]'])[0]
    meta['v_range'] = (toker.convert_tokens_to_ids('!')[0],
                       len(toker.vocab))
    with open(f'{opts.output}/meta.json', 'w') as f:
        json.dump(vars(opts), f, indent=4)

    output_field_name = ['id2len', 'txt2img', 'img2txts']
    with open_lmdb(opts.output, readonly=False) as db:
        if opts.task == 'vqa':
            questions = json.load(open(opts.annotations[0]))['questions']
            if len(opts.annotations) == 3:
                answers = json.load(open(opts.annotations[1]))['annotations']
                ans2label = pickle.load(open(opts.annotations[2], 'rb'))
                with open(f'{opts.output}/ans2label.pkl', 'wb') as f:
                    pickle.dump(ans2label, f)
            else:
                answers = None
                ans2label = None

            # train2014, val2014
            img_split = _get_img_split(opts.annotations[0])
            jsons = process_vqa(questions, answers, ans2label,
                                db, tokenizer, img_split)
        elif opts.task == 've':
            contents = open(opts.annotations[0], "r").read()
            hypotheses = [json.loads(str(item))
                          for item in contents.strip().split('\n')]
            from utils.misc import VE_ENT2IDX
            ans2label = VE_ENT2IDX
            jsons = process_visual_entailment(
                hypotheses, ans2label, db, tokenizer)
        elif opts.task == 'caption':
            data = json.load(open(opts.annotations[0]))
            img_split = _get_img_split(opts.annotations[0])
            jsons = process_caption(data, db, tokenizer, img_split)
        elif opts.task == 'conceptual':
            split = 'train' if 'train' in opts.annotations[0] else 'val'
            imgs = set(json.load(open(opts.annotations[1])))
            with open(opts.annotations[0]) as tsv:
                jsons = process_conceptual_caption(tsv, imgs,
                                                   db, tokenizer, split)
        elif opts.task == 'sbu':
            data = json.load(open(opts.annotations[0]))
            jsons = process_sbu_caption(data, db, tokenizer)
        elif opts.task == 'itm':
            data = json.load(open(opts.annotations[0]))
            if 'coco' in opts.annotations[0].lower():
                dataset = 'coco'
                if 'train' in opts.annotations[0].lower():
                    split = 'train2014'
                elif ('val' in opts.annotations[0].lower()
                      or 'test' in opts.annotations[0].lower()):
                    split = 'val2014'
                else:
                    raise ValueError()
            elif 'flickr' in opts.annotations[0].lower():
                dataset = 'flickr'
                split = None
            else:
                raise ValueError()
            jsons = process_image_text_retrieval(
                data, db, tokenizer, dataset, split)
        elif opts.task == 'licheng_cleaned':
            data = json.load(open(opts.annotations[0]))
            jsons = process_caption_licheng_cleaned(
                data, db, tokenizer,
                split=opts.annotations[0].split(".")[0].split("/")[-1])
        elif opts.task == 'gqa':
            data = json.load(open(opts.annotations[0]))
            data_split = opts.annotations[0].split(".")[0].split("/")[-1]
            data_split = data_split.split("_")[0]
            jsons = process_gqa(
                data, db, tokenizer,
                split=data_split)
        elif opts.task == 'vcr':
            data = []
            with open(opts.annotations[0], "r") as f:
                for line in f:
                    data.append(json.loads(line))
            img_split = opts.annotations[0].split("/")[-1].split(".")[0]
            tokenizer = bert_tokenize_for_vcr(toker)
            ann_folder = "/".join(opts.annotations[0].split("/")[:-1])
            object_categories_path = ann_folder+"/object_categories.json"
            object_categories = json.load(open(object_categories_path, "r"))
            jsons = process_vcr(data, db, tokenizer,
                                img_split, object_categories)
            output_field_name = ['id2len_qa', 'id2len_qar', 'txt2img',
                                 'img2txts', 'special_tokens']
        elif opts.task == 'nlvr2':
            with open(opts.annotations[0]) as ann:
                if len(opts.annotations) == 2:
                    imgs = set(json.load(open(opts.annotations[1])))
                else:
                    imgs = None
                jsons = process_nlvr2(ann, db, tokenizer, imgs)
        elif opts.task == 're':
            data = []
            refs = pickle.load(open(opts.annotations[0], 'rb'))
            instances = json.load(open(opts.annotations[1], 'r'))
            iid_to_ann_ids = json.load(open(opts.annotations[2],
                                            'r'))['iid_to_ann_ids']
            # dirs/refcoco_testA_bert-base-cased.db -> testA
            img_split = opts.output.split('/')[-1].split('_')[1]
            jsons = process_referring_expressions(
                refs, instances, iid_to_ann_ids, db, tokenizer, img_split)
            output_field_name = ['id2len', 'images', 'annotations',
                                 'categories', 'refs']
        else:
            raise ValueError()

    for dump, name in zip(jsons, output_field_name):
        with open(f'{opts.output}/{name}.json', 'w') as f:
            json.dump(dump, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations', required=True, nargs='+',
                        help='annotation JSON')
    parser.add_argument('--output', required=True,
                        help='output dir of DB')
    parser.add_argument('--task', required=True,
                        choices=['vqa', 'caption',
                                 've', "itm", "licheng_cleaned",
                                 'vcr', 'nlvr2', 're', 'gqa',
                                 'conceptual', 'sbu'])
    parser.add_argument('--bert', default='bert-base-cased')
    args = parser.parse_args()
    if args.task == 'vqa':
        assert len(args.annotations) == 3 or len(args.annotations) == 1
    elif args.task == 'gqa':
        assert len(args.annotations) == 1
    elif args.task == 've':
        assert len(args.annotations) == 1
    elif args.task == 'itm':
        assert len(args.annotations) == 1
    elif args.task == 'licheng_cleaned':
        assert len(args.annotations) == 1
    elif args.task == 'caption':
        assert len(args.annotations) == 1
    elif args.task == 'vcr':
        assert len(args.annotations) == 1
    elif args.task == 'nlvr2':
        assert len(args.annotations) == 1 or len(args.annotations) == 2
    elif args.task == 'conceptual':
        assert len(args.annotations) == 2 or len(args.annotations) == 1
    elif args.task == 'sbu':
        assert len(args.annotations) == 1
    elif args.task == 're':
        assert len(args.annotations) == 3
    main(args)
