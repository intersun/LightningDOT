import json
import os
import glob

from argparse import Namespace
from collections import ChainMap, defaultdict
from uniter_model.data import ImageLmdbGroup


NAME_PREFIX, NAME_SURFIX = '', '.npz'
caption_files_pattern = '/ssd2/siqi/Projects/offline/model_compression/raw/captions/flicker*.captions'
caption_logs_pattern = '/ssd2/siqi/Projects/offline/model_compression/raw/captions/logs/flicker*'
annotation_file = '/ssd2/siqi/Projects/offline/model_compression/raw/Flickr30K/annotations/results_20130124.token'
img_folder = '/ssd2/siqi/Projects/offline/model_compression/raw/Flickr30K/flickr30k-images'
output_file = '/ssd2/siqi/Projects/offline/model_compression/raw/flicker_meta.json'
args = Namespace(
    conf_th=0.2, max_bb=100, min_bb=10, num_bb=36, compressed_db=False,
    train_txt_dbs=["/ssd2/siqi/Projects/model_compression/data/db/itm_flickr30k_train_base-cased.db"],
    val_txt_db="/ssd2/siqi/Projects/model_compression/data/db/itm_flickr30k_val_base-cased.db",
    test_txt_db="/ssd2/siqi/Projects/model_compression/data/db/itm_flickr30k_test_base-cased.db"
 )


caption_files_pattern = '/ssd2/siqi/Projects/offline/model_compression/raw/captions/coco*.captions'
caption_logs_pattern = '/ssd2/siqi/Projects/offline/model_compression/raw/captions/logs/coco*'
annotation_files = [
    '/ssd2/siqi/Projects/offline/model_compression/raw/COCO_annotation/captions_val2014.json',
    '/ssd2/siqi/Projects/offline/model_compression/raw/COCO_annotation/captions_train2014.json',
]

img_folder = [
    '/ssd2/siqi/Projects/offline/model_compression/raw/Images/train2014',
    '/ssd2/siqi/Projects/offline/model_compression/raw/Images/val2014'
    ]
output_file = '/ssd2/siqi/Projects/offline/model_compression/raw/coco_meta.json'
args = Namespace(
    conf_th=0.2, max_bb=100, min_bb=10, num_bb=36, compressed_db=False,
    train_txt_dbs=["/db/itm_coco_train_base-cased.db", "/db/itm_coco_restval_base-cased.db"],
    val_txt_db="/db/itm_coco_val_base-cased.db",
    test_txt_db="/db/itm_coco_test_base-cased.db"
 )

MAX_LEN = 12


def annotation2json(annotation_file, format='flicker', prefix='', max_len = 12):

    res = defaultdict(list)
    if format == 'flicker':
        with open(annotation_file) as f:
            lines = [l.strip() for l in f.readlines()]
            for l in lines:
                # print(l, l.split('\t'))
                k, v = l.split('\t')
                k = k.split('.')[0]
                k = NAME_PREFIX + '0' * (max_len - len(k)) + k + NAME_SURFIX
                res[k].append(v)
    elif format == 'coco':
        # prefix = 'coco_val2014_'
        with open(annotation_file) as f:
            labels = json.load(f)['annotations']
            for l in labels:
                name = str(l['image_id'])
                name = prefix + '0' * (max_len - len(name)) + name + NAME_SURFIX
                # assert name in img2txt, f'error {name}'
                res[name].append(l['caption'])
    else:
        raise NotImplementedError()
    return res


def parse_rt_log(log_file, n_captions=5):
    # log_file, n_captions = '/ssd2/siqi/Projects/model_compression/data/raw/captions/logs/flicker.names.1', 5

    with open(log_file)as f:
        lines = [l.strip() for l in f.readlines()]
        idx = [i for i, l in enumerate(lines) if 'image 'in l and '.jpg:' in l]

    res = dict()
    for i in idx:
        captions = lines[(i-n_captions-1):(i-1)]
        name = (lines[i].split()[1]).split('.')[0]
        name = NAME_PREFIX + '0' * (MAX_LEN - len(name)) + name + NAME_SURFIX
        # print(captions, '\n', name, '\n', lines[i])
        res[name] = captions
    return res


all_img_dbs = ImageLmdbGroup(args.conf_th, args.max_bb, args.min_bb, args.num_bb, args.compressed_db)

train_img2txt = dict(ChainMap(*[json.load(open(os.path.join(db_folder, 'img2txts.json'))) for db_folder in args.train_txt_dbs]))
val_img2txt = dict(ChainMap(*[json.load(open(os.path.join(db_folder, 'img2txts.json'))) for db_folder in [args.val_txt_db]]))
test_img2txt = dict(ChainMap(*[json.load(open(os.path.join(db_folder, 'img2txts.json'))) for db_folder in [args.test_txt_db]]))

img2txt = dict()
for t in [train_img2txt, val_img2txt, test_img2txt]:
    img2txt.update(t)
txt2img = dict(sum([[(v, k) for v in vals] for k, vals in img2txt.items()], []))

caption_files = glob.glob(caption_files_pattern)
captions = dict()
total_counter = 0
for cf in caption_files:
    nf = '.'.join(cf.split('.')[:-1])
    print(cf, '\n', nf)
    with open(nf) as f:
        names = [f.strip().split('.')[0] for f in f.readlines()]
        # max_len = max([len(n) for n in names])
        names = [NAME_PREFIX + '0' * (MAX_LEN - len(n)) + n + NAME_SURFIX for n in names]
    with open(cf) as f:
        captions_tmp = [f.strip() for f in f.readlines()]

    captions.update({n: c for n, c in zip(names, captions_tmp)})

    assert len(names) == len(captions_tmp), f'lenght of names {len(names)} and {len(captions_tmp)} should be same'

    counter = 0
    for n in names:
        if n not in img2txt:
            counter += 1
            # print(n)
    # print(counter, len(names))
    total_counter += counter

captions_full = dict()
caption_log_files = glob.glob(caption_logs_pattern)
for log_file in caption_log_files:
    # print(log_file)
    captions_tmp = parse_rt_log(log_file)
    captions_full.update(captions_tmp)

for k in captions:
    assert captions[k] in captions_full[k][0], f'captions in {k} are different'
    # print('caption =', captions[k], '\ncaptions =', captions_full[k])

# annotations = annotation2json(annotation_file, 'flicker')
annotations = dict(ChainMap(*[
    annotation2json(annotation_files[0], 'coco', 'coco_val2014_', 12),
    annotation2json(annotation_files[1], 'coco', 'coco_train2014_', 12)
    ]))

everything = defaultdict(dict)
missing_annotation = []
missing_caption = []
for k in img2txt:
    k_name = 'COCO' + k[4:-4]
    try:
        everything[k]['annotation'] = annotations[k]
    except KeyError:
        everything[k]['annotation'] = []
        missing_annotation.append(k)

    fname = k_name + '.npz'
    try:
        everything[k]['caption'] = [captions[fname]]
        everything[k]['caption_multiple'] = captions_full[fname]
    except KeyError:
        everything[k]['caption'] = []
        everything[k]['caption_multiple'] = []
        missing_caption.append(k)

    everything[k]['index'] = img2txt[k]

    fname = k_name + '.jpg'
    # fname = ((k.split('.')[0]).split('_')[1]).lstrip('0')+'.jpg'
    if 'train' in fname:
        everything[k]['img_file'] = os.path.join(img_folder[0],  fname)
    else:
        everything[k]['img_file'] = os.path.join(img_folder[1],  fname)

    assert os.path.isfile(everything[k]['img_file']), f'file {fname} not exist in {img_folder}'

with open(output_file, 'w') as f:
   json.dump(everything, f, indent=2)
