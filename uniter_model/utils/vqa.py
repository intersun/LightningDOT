"""
NOTE: modified from ban-vqa
This code is slightly modified from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
import os
import json
import re
import sys
import pickle


CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve":
    "could've", "couldnt": "couldn't", "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've", "didnt": "didn't", "doesnt":
    "doesn't", "dont": "don't", "hadnt": "hadn't", "hadnt've":
    "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent":
    "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve":
    "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll",
    "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", "Im":
    "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've":
    "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's",
    "maam": "ma'am", "mightnt": "mightn't", "mightnt've":
    "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've",
    "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't",
    "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat":
    "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve":
    "she'd've", "she's": "she's", "shouldve": "should've", "shouldnt":
    "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve":
    "shouldn't've", "somebody'd": "somebodyd", "somebodyd've":
    "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll":
    "somebody'll", "somebodys": "somebody's", "someoned": "someone'd",
    "someoned've": "someone'd've", "someone'dve": "someone'd've",
    "someonell": "someone'll", "someones": "someone's", "somethingd":
    "something'd", "somethingd've": "something'd've", "something'dve":
    "something'd've", "somethingll": "something'll", "thats":
    "that's", "thered": "there'd", "thered've": "there'd've",
    "there'dve": "there'd've", "therere": "there're", "theres":
    "there's", "theyd": "they'd", "theyd've": "they'd've", "they'dve":
    "they'd've", "theyll": "they'll", "theyre": "they're", "theyve":
    "they've", "twas": "'twas", "wasnt": "wasn't", "wed've":
    "we'd've", "we'dve": "we'd've", "weve": "we've", "werent":
    "weren't", "whatll": "what'll", "whatre": "what're", "whats":
    "what's", "whatve": "what've", "whens": "when's", "whered":
    "where'd", "wheres": "where's", "whereve": "where've", "whod":
    "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl":
    "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll",
    "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve":
    "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll":
    "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd":
    "you'd", "youd've": "you'd've", "you'dve": "you'd've", "youll":
    "you'll", "youre": "you're", "youve": "you've"
}

MANUAL_MAP = {'none': '0',
              'zero': '0',
              'one': '1',
              'two': '2',
              'three': '3',
              'four': '4',
              'five': '5',
              'six': '6',
              'seven': '7',
              'eight': '8',
              'nine': '9',
              'ten': '10'}
ARTICLES = ['a', 'an', 'the']
PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
COMMA_STRIP = re.compile(r"(\d)(\,)(\d)")
PUNCT = [';', r"/", '[', ']', '"', '{', '}',
         '(', ')', '=', '+', '\\', '_', '-',
         '>', '<', '@', '`', ',', '?', '!']


# Notice that VQA score is the average of 10 choose 9 candidate answers cases
# See http://visualqa.org/evaluation.html
def get_score(occurences):
    if occurences == 0:
        return .0
    elif occurences == 1:
        return .3
    elif occurences == 2:
        return .6
    elif occurences == 3:
        return .9
    else:
        return 1.


def process_punctuation(inText):
    outText = inText
    for p in PUNCT:
        if (p + ' ' in inText
                or ' ' + p in inText
                or re.search(COMMA_STRIP, inText) is not None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    outText = PERIOD_STRIP.sub("", outText, re.UNICODE)
    return outText


def process_digit_article(inText):
    outText = []
    tempText = inText.lower().split()
    for word in tempText:
        word = MANUAL_MAP.setdefault(word, word)
        if word not in ARTICLES:
            outText.append(word)
        else:
            pass
    for wordId, word in enumerate(outText):
        if word in CONTRACTIONS:
            outText[wordId] = CONTRACTIONS[word]
    outText = ' '.join(outText)
    return outText


def preprocess_answer(answer):
    answer = process_digit_article(process_punctuation(answer))
    answer = answer.replace(',', '')
    return answer


def filter_answers(answers_dset, min_occurence):
    """This will change the answer to preprocessed version
    """
    occurence = {}

    for ans_entry in answers_dset:
        gtruth = ans_entry.get('multiple_choice_answer', None)
        if gtruth is None:
            gtruth = ans_entry['answers'][0]['answer']  # VG, GQA pretraining
        gtruth = preprocess_answer(gtruth)
        if gtruth not in occurence:
            occurence[gtruth] = set()
        occurence[gtruth].add(ans_entry['question_id'])
    for answer in list(occurence):
        if len(occurence[answer]) < min_occurence:
            occurence.pop(answer)

    print('Num of answers that appear >= %d times: %d' % (
        min_occurence, len(occurence)))
    return occurence


def create_ans2label(occurence, path):
    """
    occurence: dict {answer -> whatever}
    name: dir of the output file
    """
    ans2label = {}
    label2ans = []
    label = 0
    for answer in occurence:
        label2ans.append(answer)
        ans2label[answer] = label
        label += 1

    output_file = os.path.join(path, 'ans2label.pkl')
    pickle.dump(ans2label, open(output_file, 'wb'))


def compute_target(answers, ans2label):
    answer_count = {}
    if len(answers) == 1:
        # VG VQA, GQA
        answer_ = preprocess_answer(answers[0]['answer'])
        answer_count[answer_] = 10
    else:
        # COCO VQA
        for answer in answers:
            answer_ = preprocess_answer(answer['answer'])
            answer_count[answer_] = answer_count.get(answer_, 0) + 1

    labels = []
    scores = []
    for answer in answer_count:
        if answer not in ans2label:
            continue
        labels.append(ans2label[answer])
        score = get_score(answer_count[answer])
        scores.append(score)
    target = {'labels': labels, 'scores': scores}
    return target


if __name__ == '__main__':
    *answer_files, output = sys.argv[1:]
    answers = []
    for ans_file in answer_files:
        ans = json.load(open(ans_file))['annotations']
        answers.extend(ans)

    occurence = filter_answers(answers, 9)

    if os.path.exists(f'{output}/ans2label.pkl'):
        raise ValueError(f'{output} already exists')
    create_ans2label(occurence, output)
