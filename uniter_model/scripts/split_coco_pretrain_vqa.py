"""
split pretraining COCO VQA according to image directory
"""
import json
from os.path import dirname

ANNOTATION = '/ssd2/yenchun/ANNOTATIONS'
EXCLUDE_IID = f'{dirname(__file__)}/../index/excluded_coco_vg_iids.json'

COCO_TRAIN_QUESTION = (f'{ANNOTATION}/VQA/'
                       'v2_OpenEnded_mscoco_train2014_questions.json')
COCO_TRAIN_ANSWER = f'{ANNOTATION}/VQA/v2_mscoco_train2014_annotations.json'

COCO_VAL_QUESTION = (f'{ANNOTATION}/VQA/'
                     'v2_OpenEnded_mscoco_val2014_questions.json')
COCO_VAL_ANSWER = f'{ANNOTATION}/VQA/v2_mscoco_val2014_annotations.json'

OUT_DIR = f'{ANNOTATION}/latest_cleaned/'


def _filter_data(examples, exclude_iids):
    filtered = (ex for ex in examples if ex['image_id'] not in exclude_iids)
    return filtered


def main():
    ids = json.load(open(EXCLUDE_IID))
    train_exclude_iids = set(ids['flickr30k_coco_iids']
                             + ids['refer_val_coco_iids']
                             + ids['refer_test_coco_iids'])
    val_exclude_iids = set(ids['flickr30k_coco_iids']
                           + ids['karpathy_minival_iids']
                           + ids['karpathy_minitest_iids'])

    # split train
    questions = json.load(open(COCO_TRAIN_QUESTION))['questions']
    train_qs = _filter_data(questions, train_exclude_iids)
    with open(f'{OUT_DIR}/coco_vqa_trainsplit_questions.json', 'w') as f:
        json.dump({'questions': list(train_qs)}, f)
    answers = json.load(open(COCO_TRAIN_ANSWER))['annotations']
    train_as = _filter_data(answers, train_exclude_iids)
    with open(f'{OUT_DIR}/coco_vqa_trainsplit_annotations.json', 'w') as f:
        json.dump({'annotations': list(train_as)}, f)

    # split val
    questions = json.load(open(COCO_VAL_QUESTION))['questions']
    val_qs = _filter_data(questions, val_exclude_iids)
    with open(f'{OUT_DIR}/coco_vqa_valsplit_questions.json', 'w') as f:
        json.dump({'questions': list(val_qs)}, f)
    answers = json.load(open(COCO_VAL_ANSWER))['annotations']
    val_as = _filter_data(answers, val_exclude_iids)
    with open(f'{OUT_DIR}/coco_vqa_valsplit_annotations.json', 'w') as f:
        json.dump({'annotations': list(val_as)}, f)


if __name__ == '__main__':
    main()
