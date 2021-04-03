"""
This code converts all RefCOCO(+/g) detections from Mask R-CNN 
(https://github.com/lichengunc/MAttNet)
to image_id -> [box], where each box is {box, category_id, category_name, score}
"""
import json
import os
import os.path as osp

dets_dir = 'datasets/refer/detections'
image_set = set()
dataset_names = ['refcoco_unc', 'refcoco+_unc', 'refcocog_umd']
Detections = {}
for dataset_name in dataset_names:
    dets_file = osp.join(dets_dir, dataset_name, 
                        'res101_coco_minus_refer_notime_dets.json')
    detections = json.load(open(dets_file, 'r'))
    for det in detections:
        image_set.add(det['image_id'])
    Detections[dataset_name] = detections
num_images = len(image_set)

iid_to_dets = {}
for dataset_name in dataset_names:
    detections = Detections[dataset_name]
    for det in detections:
        image_id = det['image_id']
        if image_id in image_set:
            box = {'box': det['box'], 
                    'category_id': det['category_id'], 
                    'category_name': det['category_name'], 
                    'score': det['score']}
            iid_to_dets[image_id] = iid_to_dets.get(image_id, []) + [box]
    for det in detections:
        image_id = det['image_id']
        if image_id in image_set:
            image_set.remove(image_id)

num_dets = sum([len(dets) for dets in iid_to_dets.values()])
print(f'{num_dets} detections in {num_images} images for {dataset_names}.')

# save
with open('index/iid_to_dets.json', 'w') as f:
    json.dump(iid_to_dets, f)

