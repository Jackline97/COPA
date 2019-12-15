# CSI 5138 Group ProjectIntroduction to Deep Learning and Reinforcement Learning
# U Ottawa
#
# CSI 5138 Group Project
# Fall 2019
#
#
# Group 6
# Li, Yansong
# Qu, Shuzheng
# Su, Xuanyu
# Yang, Siyuan
# Linkletter, Maurice
#
#
# superGLUE COPA Task
#
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT COPA model runner."""
from os import path

import pandas as pd
import random
from tqdm import tqdm
import json
import numpy as np
import torch

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam

from modeling import BertForMultipleChoice




class CopaRecord(object):
    # A single copa data item

    def __init__(self, example_id, example_type, example_premise, example_hypothesis_1, example_hypothesis_2, example_label=None):
        self.id = example_id
        self.type = example_type
        self.premise = example_premise
        self.hypotheses = [
            example_hypothesis_1,
            example_hypothesis_2,
        ]
        self.label = example_label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        l = [
            f"id: {self.id}",
            f"type: {self.type}",
            f"premise: {self.premise}",
            f"hypothesis1: {self.hypotheses[0]}",
            f"hypothesis2: {self.hypotheses[1]}"
        ]

        if self.label is not None:
            l.append(f"label: {self.label}")

        return ", ".join(l)

class InputFeatures(object):
    def __init__(self, example_id, choices_features, label):
        self.example_id = example_id
        self.choices_features = [
            {
                'tokens': tokens,
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids
            }
            for tokens, input_ids, input_mask, segment_ids in choices_features
        ]
        self.label = label


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def convert_examples_to_features(examples, tokenizer, max_seq_length,is_training):
    features = []
    for example_index, example in enumerate(examples):
        premise_tokens = tokenizer.tokenize(example.premise)

        choices_features = []
        for hypothesis_index, hypothesis in enumerate(example.hypotheses):
            # We create a copy of the context tokens in order to be
            # able to shrink it according to ending_tokens
            premise_tokens_choice = premise_tokens[:]
            hypothesis_tokens = tokenizer.tokenize(hypothesis)

            _truncate_seq_pair(premise_tokens_choice, hypothesis_tokens, max_seq_length - 3)
            if example.type == 'cause':
                tokens = ["[CLS]"] + hypothesis_tokens + ["[SEP]"] + premise_tokens_choice + ["[SEP]"]
                segment_ids = [0] * (len(hypothesis_tokens) + 2) + [1] * (len(premise_tokens_choice) + 1)
            else:
                tokens = ["[CLS]"] + premise_tokens_choice + ["[SEP]"] + hypothesis_tokens + ["[SEP]"]
                segment_ids = [0] * (len(premise_tokens_choice) + 2) + [1] * (len(hypothesis_tokens) + 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            # Pad any sequence upto the max length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            choices_features.append((tokens, input_ids, input_mask, segment_ids))

        label = example.label

        features.append(
            InputFeatures(
                example_id=example.id,
                choices_features=choices_features,
                label=label
            )
        )

    return features

def convert_mnli_to_copa(entailment, neutral, contradiction):
    e_n = None
    e_c = None
    n_c = None
    skipped_count = 0
    try:
        e_pairId = str(entailment.pairID.item())
        e_sentence1 = str(entailment.sentence1.item())
        e_sentence2 = str(entailment.sentence2.item())

        n_pairId = str(neutral.pairID.item())
        n_sentence2 = str(neutral.sentence2.item())
        e_n = CopaRecord((e_pairId + '_' + n_pairId), 'effect', e_sentence1.lower(), e_sentence2.lower(), n_sentence2.lower(), 0)
    except:
        skipped_count += 1

    try:
        n_pairId = str(neutral.pairID.item())
        n_sentence2 = str(neutral.sentence2.item())
        c_pairId = str(contradiction.pairID.item())
        c_sentence2 = str(contradiction.sentence2.item())

        e_c = CopaRecord((e_pairId + '_' + c_pairId), 'effect', e_sentence1.lower(), c_sentence2.lower(), e_sentence2.lower(), 1)
        n_c = CopaRecord((n_pairId + '_' + c_pairId), 'effect', e_sentence1.lower(), n_sentence2.lower(), c_sentence2.lower(), 0)

    except:
        skipped_count += 1

    return e_n, e_c, n_c

def load_copa_data(path):
    records = []
    with open(path, 'r') as f:
        for line in f:
            train = json.loads(line)
            answer_train_1 = train.get('choice1')
            answer_train_2 = train.get('choice2')
            q_type = train.get('question')
            premise = train.get('premise')
            choice = train.get('label')
            idx = str(train.get('idx'))

            records.append(
                CopaRecord(idx, q_type, premise, answer_train_1, answer_train_2, choice)
            )

    return records

def load_copa_data_from_csv(path):
    records = []
    data = pd.read_csv(path, sep=',')

    for row in data.itertuples(index=True, name='Pandas'):
        idx = str(row[1])
        premise = str(row[2])
        answer_train_1 = str(row[3])
        answer_train_2 = str(row[4])
        q_type = str(row[5])
        choice = row[6]


        records.append(
                CopaRecord(idx, q_type, premise, answer_train_1, answer_train_2, choice)
            )

    return records

def load_mnli_data(mnli_path, max_seq_length):
    records = []
    mnli = pd.read_table(mnli_path, sep='\t', header=0)
    done = {}

    for index, row in mnli.iterrows():
        prompt_id = row.promptID

        if prompt_id in done:
            continue

        entailment_row = mnli.loc[mnli['pairID'] == str(prompt_id) + 'e']
        neutral_row = mnli.loc[mnli['pairID'] == str(prompt_id) + 'n']
        contradiction_row = mnli.loc[mnli['pairID'] == str(prompt_id) + 'c']

        (entail_neutral, entail_contradiction, neutral_contradiction) = convert_mnli_to_copa(entailment_row, neutral_row, contradiction_row)

        if entail_neutral is not None:
            records.append(entail_neutral)

        if entail_contradiction is not None:
            records.append(entail_contradiction)

        if neutral_contradiction is not None: 
            records.append(neutral_contradiction)

        done[prompt_id] = prompt_id

    return records

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def select_field(features, field):
    return [
        [
            choice[field]
            for choice in feature.choices_features
        ]
        for feature in features
    ]

def do_evaluation(model, eval_dataloader, is_training=False):

    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    logits_all = None
    with torch.no_grad():
        for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
            input_ids = input_ids.cuda()
            input_mask = input_mask.cuda()
            segment_ids = segment_ids.cuda()
            label_ids = label_ids.cuda()
            tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
            logits = model(input_ids, segment_ids, input_mask)
            logits = logits.detach().cpu().numpy()
            if logits_all is None:
                logits_all = logits.copy()
            else:
                logits_all = np.vstack((logits_all, logits))
            label_ids = label_ids.to('cpu').numpy()
            tmp_eval_accuracy = accuracy(logits, label_ids)
            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples

    model.zero_grad()
    return logits_all, eval_accuracy, eval_loss



def main():
    checkpoint_path = './checkpoint/bert_copa.pt'

    train_with_mnli = False
    train_with_copa = True

    load_checkpoint_if_exists = True
    save_checkpoint = True

    num_train_epochs = 3
    learning_rate = 3e-5
    max_seq_length = 50
    train_batch_size = 50

    warmup_proportion = 0.1

    seed = 1979
    gradient_accumulation_steps = 1
    l2_reg = 0.02

    use_margin_loss = True
    margin = .20# 0.37

    train_batch_size = int(train_batch_size / gradient_accumulation_steps)
    eval_batch_size = train_batch_size


    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    train_examples = []

    if train_with_mnli:
        mnli = load_mnli_data('./data/MNLI/dev_matched.tsv', max_seq_length)
        train_examples.extend(mnli)

    if train_with_copa:
        copa = load_copa_data('./data/COPA/train.jsonl')
        train_examples.extend(copa)


    num_train_steps = int(len(train_examples) / train_batch_size / gradient_accumulation_steps * num_train_epochs)

    eval_examples = load_copa_data('./data/COPA/val.jsonl')
    test_examples = load_copa_data_from_csv('./data/COPA/test.csv')

    # Prepare model
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    if use_margin_loss:
        model = BertForMultipleChoice.from_pretrained("bert-base-uncased", num_choices=2, margin=margin)
    else:
        model = BertForMultipleChoice.from_pretrained("bert-base-uncased", num_choices=2)

    if load_checkpoint_if_exists and path.exists(checkpoint_path):
        print(f'Loading existing checkpoint: {checkpoint_path} ')
        device = torch.device("cuda")
        model.load_state_dict(torch.load(checkpoint_path))
        model.to(device)


    model.cuda()

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())

    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': l2_reg},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    t_total = num_train_steps

    optimizer = BertAdam(optimizer_grouped_parameters, lr=learning_rate, warmup=warmup_proportion, t_total=t_total, b2=0.98)


    # Prep Eval Data
    eval_features = convert_examples_to_features(eval_examples, tokenizer, max_seq_length, True)
    eval_input_ids = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long)
    eval_input_mask = torch.tensor(select_field(eval_features, 'input_mask'), dtype=torch.long)
    eval_segment_ids = torch.tensor(select_field(eval_features, 'segment_ids'), dtype=torch.long)
    eval_label = torch.tensor([f.label for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(eval_input_ids, eval_input_mask, eval_segment_ids, eval_label)

    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)

    # Prep Test Data
    test_features = convert_examples_to_features(test_examples, tokenizer, max_seq_length, True)
    test_input_ids = torch.tensor(select_field(test_features, 'input_ids'), dtype=torch.long)
    test_input_mask = torch.tensor(select_field(test_features, 'input_mask'), dtype=torch.long)
    test_segment_ids = torch.tensor(select_field(test_features, 'segment_ids'), dtype=torch.long)
    test_label = torch.tensor([f.label for f in test_features], dtype=torch.long)
    test_data = TensorDataset(test_input_ids, test_input_mask, test_segment_ids, test_label)

    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=eval_batch_size)

    # Prep Training Data
    train_features = convert_examples_to_features(train_examples, tokenizer, max_seq_length, True)
    train_input_ids = torch.tensor(select_field(train_features, 'input_ids'), dtype=torch.long)
    train_input_mask = torch.tensor(select_field(train_features, 'input_mask'), dtype=torch.long)
    train_segment_ids = torch.tensor(select_field(train_features, 'segment_ids'), dtype=torch.long)
    train_label = torch.tensor([f.label for f in train_features], dtype=torch.long)
    train_data = TensorDataset(train_input_ids, train_input_mask, train_segment_ids, train_label)

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size)

    print("***** Running training *****")
    print("  Num examples = %d", len(train_examples))
    print("  Batch size = %d", train_batch_size)
    print("  Num steps = %d", num_train_steps)



    global_step = 0
    eval_acc_list = []
    for epoch in range(num_train_epochs):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc=(f'Epoch: {epoch}'))):
            model.train()
            batch = tuple(t.cuda() for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            loss = model(input_ids, segment_ids, input_mask, label_ids)

            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            loss.backward()
            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1

            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                model.zero_grad()
                global_step += 1

        logits_all, train_accuracy, train_loss = do_evaluation(model, train_dataloader, is_training=False)
        tqdm.write(f'Training Accuracy: {train_accuracy}')
        tqdm.write(f'Training Loss: {train_loss}')

        logits_all, eval_accuracy, eval_loss = do_evaluation(model, eval_dataloader, is_training=False)
        tqdm.write(f'Evaluation Accuracy: {eval_accuracy}')
        tqdm.write(f'Evaluation Loss: {eval_loss}')

        eval_acc_list.append(eval_accuracy)

    if save_checkpoint:
        torch.save(model.state_dict(), checkpoint_path)

    logits_all, test_acc, test_loss = do_evaluation(model, test_dataloader, is_training=False)
    print(f'Testing Accuracy: {test_acc}')
    print(f'Testing Loss: {test_loss}')



if __name__ == "__main__":
    main()
