"""
mcan vg annotation image id is COCO, need to map back to VG
"""
import json


ANNOTATION = '/ssd2/yenchun/ANNOTATIONS'
# karpathy 5k test split
TEST_5K = f'{ANNOTATION}/Image-Text-Matching/coco_test.json'

VG_QUESTION = f'{ANNOTATION}/VQA/VG_questions.json'
VG_ANSWER = f'{ANNOTATION}/VQA/VG_annotations.json'
VG_IMG_META = f'{ANNOTATION}/VQA/image_data.json'


def _get_img_id(img_name):
    img_name = img_name[:-4]
    id_ = int(img_name.split('_')[-1])
    return id_


def _get_test_ids():
    data = json.load(open(TEST_5K))
    ids = {_get_img_id(d['filename']) for d in data}
    return ids


def _get_coco2vg():
    data = json.load(open(VG_IMG_META))
    coco2vg = {d['coco_id']: d['image_id'] for d in data}
    return coco2vg


def filter_data(data, test_ids):
    filtered = (d for d in data if d['image_id'] not in test_ids)
    return filtered


def map_data(data, coco2vg):
    def gen_mapped():
        for d in data:
            coco_id = d['image_id']
            d['image_id'] = coco2vg[coco_id]
            yield d
    return gen_mapped()


def main():
    test_ids = _get_test_ids()
    coco2vg = _get_coco2vg()

    # process questions
    questions = json.load(open(VG_QUESTION))['questions']
    mapped_qs = list(map_data(filter_data(questions, test_ids), coco2vg))
    qname = f'{VG_QUESTION}.mapped'
    json.dump({'questions': mapped_qs}, open(qname, 'w'))
    del questions, mapped_qs

    # process answers
    answers = json.load(open(VG_ANSWER))['annotations']
    mapped_as = list(map_data(filter_data(answers, test_ids), coco2vg))
    aname = f'{VG_ANSWER}.mapped'
    json.dump({'annotations': mapped_as}, open(aname, 'w'))


if __name__ == '__main__':
    main()
