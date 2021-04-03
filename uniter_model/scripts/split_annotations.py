import json
from os.path import join
import sys


def save_coco_train_val(data, output_dir):
    current_data = []
    rest_data = []
    for d in data:
        if not d['sent'].strip():
            # filter out empty sentence
            continue
        if (d['dataset'] == 'coco'
                and d['split'] == 'train'
                and 'val' in d['file_path']):
            current_data.append(d)
        else:
            rest_data.append(d)
    fileName = "pretrain_caption_coco_trainval.json"
    json.dump(current_data, open(join(output_dir, fileName), "w"))
    return rest_data


def save_by_dataset_and_split(data, dataset, split, output_dir):
    current_data = []
    rest_data = []
    for d in data:
        if not d['sent'].strip():
            # filter out empty sentence
            continue
        if split == 'trainval':
            if (d['dataset'] == 'coco'
                    and d['split'] == 'train'
                    and 'val' in d['file_path']):
                current_data.append(d)
            else:
                rest_data.append(d)
        elif d["dataset"] == dataset and d["split"] == split:
            current_data.append(d)
        else:
            rest_data.append(d)
    fileName = f"pretrain_caption_{dataset}_{split}.json"
    json.dump(current_data, open(join(output_dir, fileName), "w"))
    return rest_data


def main():
    input_file, output_dir = sys.argv[1:]
    data = json.load(open(input_file, "r"))
    data = save_coco_train_val(data, output_dir)
    for dataset in ["coco", "vg"]:
        for split in ["train", "val", "test"]:
            data = save_by_dataset_and_split(data, dataset, split, output_dir)


if __name__ == '__main__':
    main()
