import json
from xml import etree
import numpy as np
import xml.etree.ElementTree as ET
tree = ET.parse('copa-test.xml')
root = tree.getroot()

choice_test = []
for child in root.findall('item'):
    answer = int(child.get('most-plausible-alternative'))
    choice_test.append(answer)
choice_test = [x-1 for x in choice_test]
# question, answer, choice, type


# training part
question_train = []
answer_train_A = []
answer_train_B = []
choice_train = []
train_type = []

with open("train.jsonl",'r') as f:
    for line in f:
        train = json.loads(line)
        answer_train_1 = train.get('choice1')
        answer_train_2 = train.get('choice2')
        q_type = train.get('question')
        question = train.get('premise')
        choice = train.get('label')
        question_train.append(question)
        answer_train_A.append(answer_train_1)
        answer_train_B.append(answer_train_2)
        train_type.append(q_type)
        choice_train.append(choice)
answer_train = list(zip(answer_train_A, answer_train_B))


# val part

question_val = []
answer_val_A = []
answer_val_B = []
choice_val = []
val_type = []


with open("val.jsonl",'r') as f:
    for line in f:
        val = json.loads(line)
        answer_val_1 = val.get('choice1')
        answer_val_2 = val.get('choice2')
        q_val_type = val.get('question')
        question_2 = val.get('premise')
        choice_2 = val.get('label')
        question_val.append(question_2)
        answer_val_A.append(answer_val_1)
        answer_val_B.append(answer_val_2)
        val_type.append(q_val_type)
        choice_val.append(choice_2)
answer_val = list(zip(answer_val_A, answer_val_B))



# test part
question_test = []
answer_test_A = []
answer_test_B = []
test_type = []

with open("test.jsonl",'r') as f:
    for line in f:
        test = json.loads(line)
        answer_test_1 = test.get('choice1')
        answer_test_2 = test.get('choice2')
        q_test_type = test.get('question')
        question_3 = test.get('premise')
        question_test.append(question_3)
        answer_test_A.append(answer_test_1)
        answer_test_B.append(answer_test_2)
        test_type.append(q_test_type)

answer_test = list(zip(answer_test_A, answer_test_B))

