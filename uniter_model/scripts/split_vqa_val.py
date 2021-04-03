"""
split vqa val set for data augmentation
"""
import json


ANNOTATION = '/ssd2/yenchun/ANNOTATIONS'
# karpathy 5k test split
TEST_5K = f'{ANNOTATION}/Image-Text-Matching/coco_test.json'

# original VQA val data
VAL_QUESTION = f'{ANNOTATION}/VQA/v2_OpenEnded_mscoco_val2014_questions.json'
VAL_ANSWER = f'{ANNOTATION}/VQA/v2_mscoco_val2014_annotations.json'


def _get_img_id(img_name):
    img_name = img_name[:-4]
    id_ = int(img_name.split('_')[-1])
    return id_


def _get_test_ids():
    data = json.load(open(TEST_5K))
    ids = {_get_img_id(d['filename']) for d in data}
    return ids


def main():
    dev_ids = _get_test_ids()

    # process questions
    val_questions = json.load(open(VAL_QUESTION))['questions']
    dev_qs = []
    train_qs = []
    for q in val_questions:
        if q['image_id'] in dev_ids:
            dev_qs.append(q)
        else:
            train_qs.append(q)
    assert len(val_questions) == len(dev_qs) + len(train_qs)
    dev_q_name = VAL_QUESTION.replace('val', 'devval')
    json.dump({'questions': dev_qs}, open(dev_q_name, 'w'))
    train_q_name = VAL_QUESTION.replace('val', 'trainval')
    json.dump({'questions': train_qs}, open(train_q_name, 'w'))

    # process answers
    val_answers = json.load(open(VAL_ANSWER))['annotations']
    dev_as = []
    train_as = []
    for a in val_answers:
        if a['image_id'] in dev_ids:
            dev_as.append(a)
        else:
            train_as.append(a)
    assert len(dev_as) == len(dev_qs)
    assert len(train_as) == len(train_qs)
    dev_a_name = VAL_ANSWER.replace('val', 'devval')
    json.dump({'annotations': dev_as}, open(dev_a_name, 'w'))
    train_a_name = VAL_ANSWER.replace('val', 'trainval')
    json.dump({'annotations': train_as}, open(train_a_name, 'w'))


if __name__ == '__main__':
    main()
