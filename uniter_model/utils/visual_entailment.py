"""
NOTE: modified from ban-vqa
This code is slightly modified from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
import os
import sys
import pickle


def create_ans2label(path):
    """
    occurence: dict {answer -> whatever}
    name: dir of the output file
    """
    ans2label = {"contradiction": 0, "entailment":1 , "neutral": 2}
    label2ans = ["contradiction", "entailment", "neutral"]

    output_file = os.path.join(path, 'visual_entailment_ans2label.pkl')
    pickle.dump(ans2label, open(output_file, 'wb'))


def compute_target(answers, ans2label):
    answer_count = {}
    for answer in answers:
        answer_ = answer
        answer_count[answer_] = answer_count.get(answer_, 0) + 1

    labels = []
    scores = []
    for answer in answer_count:
        if answer not in ans2label:
            continue
        labels.append(ans2label[answer])
        score = answer_count[answer]/len(answers)
        scores.append(score)
    target = {'labels': labels, 'scores': scores}
    return target


if __name__ == '__main__':
    output = sys.argv[1:][0]
    print(output)
    if os.path.exists(f'{output}/visual_entailment_ans2label.pkl'):
        raise ValueError(f'{output} already exists')
    create_ans2label(output)
