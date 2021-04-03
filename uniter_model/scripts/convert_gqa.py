"""
convert GQA jsons into VQA format
"""
import json

from toolz.sandbox import unzip

ANNOTATION = '/ssd2/yenchun/ANNOTATIONS/'
SPLITS = ['train', 'val', 'testdev']
VERSIONS = ['all', 'balanced']


def convert(item):
    qid, example = item
    q = {'image_id': example['imageId'],
         'question_id': qid,
         'question': example['question']}
    if 'answer' in example:
        a = {'image_id': example['imageId'],
             'question_id': qid,
             'answers': [{"answer": example['answer']}]}
    else:
        a = None
    return q, a


def convert_all(data):
    questions, answers = unzip(map(convert, data.items()))
    return questions, answers


def main():
    for split in SPLITS:
        for ver in VERSIONS:
            if split == 'train' and ver == 'all':
                data = {}
                for i in range(10):
                    for qid, ex in json.load(open(
                            f'{ANNOTATION}/GQA/train_all_questions/'
                            f'train_all_questions_{i}.json')).items():
                        for key in list(ex.keys()):
                            if key not in ['imageId', 'question', 'answer']:
                                del ex[key]
                        data[qid] = ex
            else:
                data = json.load(open(f'{ANNOTATION}/GQA/'
                                      f'{split}_{ver}_questions.json'))
            questions, answers = convert_all(data)
            json.dump({'questions': list(questions)},
                      open(f'{ANNOTATION}/GQA/'
                           f'gqa_{split}_{ver}_questions.vqa.json', 'w'))
            json.dump({'annotations': list(answers)},
                      open(f'{ANNOTATION}/GQA/'
                           f'gqa_{split}_{ver}_annotations.vqa.json', 'w'))
    data = json.load(open(f'{ANNOTATION}/GQA/submission_all_questions.json'))
    questions, _ = convert_all(data)
    json.dump({'questions': list(questions)},
              open(f'{ANNOTATION}/GQA/gqa_submission_questions.vqa.json', 'w'))


if __name__ == '__main__':
    main()
