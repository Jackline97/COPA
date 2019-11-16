"""
CSI 5138 Group Project


pip install tensorflow-gpu=1.14.0     (You really need to run this model on GPU or TPU)
pip install keras
pip install bert-tensorflow
pip install tensorflow-hub


Create a ./tmp directory for the checkpoints to be saved in.  If you change the model architecture you will
need to remove the existing checkpoints files.  Either delete thme or copy them else where.

- I have been able to get 65% accuracy on the validation set using this model.

- It uses BERT_base and it was trained on a GPU.  Training it on a CPU takes too long.

- I tried the BERT_Large model be I ran out of memory.  I think we need to try it on Google Colab using TPU.
There is a flag on line 213 to tell it to use TPU(s) or not.

- There is also a version of BERT called ALBERT that we will want to investigate.  https://tfhub.dev/google/albert_xxlarge/2



"""

from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from datetime import datetime

import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization

from tensorflow import keras
import os
import re

import json
from xml import etree
import numpy as np
import xml.etree.ElementTree as ET
import pandas as pd
tree = ET.parse('copa-test.xml')
root = tree.getroot()

choice_test = []
for child in root.findall('item'):
    answer = int(child.get('most-plausible-alternative'))
    choice_test.append(answer)
choice_test = [x-1 for x in choice_test]
# question, answer, choice, type



train_data = []
with open("train.jsonl",'r') as f:
    for line in f:
        train = json.loads(line)
        answer_train_1 = train.get('choice1')
        answer_train_2 = train.get('choice2')
        q_type = train.get('question')
        premise = train.get('premise')
        choice = train.get('label')
        idx = str(train.get('idx'))
        if q_type == 'effect':
            train_data.append([(idx + '_1'), answer_train_1.lower(), premise.lower(), (choice == 0)])
            train_data.append([(idx + '_2'), answer_train_2.lower(), premise.lower(), (choice != 0)])
        else:
            train_data.append([(idx + '_1'), premise.lower(), answer_train_1.lower(), (choice == 0)])
            train_data.append([(idx + '_2'), premise.lower(), answer_train_2.lower(), (choice != 0)])
training_df = pd.DataFrame(train_data, columns=['Id', 'sentence1', 'sentence2', 'label'])

val_data = []
with open("val.jsonl",'r') as f:
    for line in f:
        train = json.loads(line)
        answer_train_1 = train.get('choice1')
        answer_train_2 = train.get('choice2')
        q_type = train.get('question')
        premise = train.get('premise')
        choice = train.get('label')
        idx = str(train.get('idx'))
        if q_type == 'effect':
            val_data.append([(idx + '_1'), answer_train_1.lower(), premise.lower(), (choice == 0)])
            val_data.append([(idx + '_2'), answer_train_2.lower(), premise.lower(), (choice != 0)])
        else:
            val_data.append([(idx + '_1'), premise.lower(), answer_train_1.lower(), (choice == 0)])
            val_data.append([(idx + '_2'), premise.lower(), answer_train_2.lower(), (choice != 0)])
val_df = pd.DataFrame(val_data, columns=['Id', 'sentence1', 'sentence2', 'label'])



# Set the output directory for saving model file
OUTPUT_DIR = './tmp'


DATA_COLUMN1 = 'sentence1'
DATA_COLUMN2 = 'sentence2'
LABEL_COLUMN = 'label'

# label_list is the list of labels, i.e. True, False or 0, 1 or 'dog', 'cat'
label_list = [True, False]


# Use the InputExample class from BERT's run_classifier code to create examples from the data
train_InputExamples = training_df.apply(lambda x: bert.run_classifier.InputExample(guid=None, # Globally unique ID for bookkeeping, unused in this example
                                                                   text_a = x[DATA_COLUMN1],
                                                                   text_b = x[DATA_COLUMN2],
                                                                   label = x[LABEL_COLUMN]), axis = 1)

test_InputExamples = val_df.apply(lambda x: bert.run_classifier.InputExample(guid=None,
                                                                   text_a = x[DATA_COLUMN1],
                                                                   text_b = x[DATA_COLUMN2],
                                                                   label = x[LABEL_COLUMN]), axis = 1)

BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
#BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-24_H-1024_A-16/1"
#BERT_MODEL_HUB = "https://tfhub.dev/google/albert_xxlarge/2"
def create_tokenizer_from_hub_module():
    """Get the vocab file and casing info from the Hub module."""
    with tf.Graph().as_default():
        bert_module = hub.Module(BERT_MODEL_HUB)
        tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
        with tf.Session() as sess:
            vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                                  tokenization_info["do_lower_case"]])

    return bert.tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case)


tokenizer = create_tokenizer_from_hub_module()


# We'll set sequences to be at most 128 tokens long.
MAX_SEQ_LENGTH = 128
# Convert our train and test features to InputFeatures that BERT understands.
train_features = bert.run_classifier.convert_examples_to_features(train_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)
test_features = bert.run_classifier.convert_examples_to_features(test_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)


def create_model(is_predicting, input_ids, input_mask, segment_ids, labels, num_labels):
  """Creates a classification model."""

  bert_module = hub.Module(BERT_MODEL_HUB, trainable=True)

  bert_inputs = dict(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids)

  bert_outputs = bert_module(
      inputs=bert_inputs,
      signature="tokens",
      as_dict=True)

  # Use "pooled_output" for classification tasks on an entire sentence.
  # Use "sequence_outputs" for token-level output.
  output_layer = bert_outputs["pooled_output"]

  hidden_size = output_layer.shape[-1].value

  # Create our own layer to tune for politeness data.
  output_weights = tf.get_variable("output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable("output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):

    # Dropout helps prevent overfitting
    output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    # Convert labels into one-hot encoding
    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

    predicted_labels = tf.squeeze(tf.argmax(log_probs, axis=-1, output_type=tf.int32))
    # If we're predicting, we want predicted labels and the probabiltiies.
    if is_predicting:
      return (predicted_labels, log_probs)

    # If we're train/eval, compute loss between predicted and actual label
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)
    return (loss, predicted_labels, log_probs)


# model_fn_builder actually creates our model function
# using the passed parameters for num_labels, learning_rate, etc.
def model_fn_builder(num_labels, learning_rate, num_train_steps,  num_warmup_steps):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        is_predicting = (mode == tf.estimator.ModeKeys.PREDICT)

        # TRAIN and EVAL
        if not is_predicting:
            (loss, predicted_labels, log_probs) = create_model(is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

            train_op = bert.optimization.create_optimizer(loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)

            # Calculate evaluation metrics.
            def metric_fn(label_ids, predicted_labels):
                accuracy = tf.metrics.accuracy(label_ids, predicted_labels)
                f1_score = tf.contrib.metrics.f1_score(label_ids, predicted_labels)
                auc = tf.metrics.auc(label_ids, predicted_labels)
                recall = tf.metrics.recall(label_ids, predicted_labels)
                precision = tf.metrics.precision(label_ids, predicted_labels)
                true_pos = tf.metrics.true_positives(label_ids, predicted_labels)
                true_neg = tf.metrics.true_negatives(label_ids, predicted_labels)
                false_pos = tf.metrics.false_positives(label_ids, predicted_labels)
                false_neg = tf.metrics.false_negatives(label_ids, predicted_labels)
                return {
                    "eval_accuracy": accuracy,
                    "f1_score": f1_score,
                    "auc": auc,
                    "precision": precision,
                    "recall": recall,
                    "true_positives": true_pos,
                    "true_negatives": true_neg,
                    "false_positives": false_pos,
                    "false_negatives": false_neg
                }

            eval_metrics = metric_fn(label_ids, predicted_labels)

            if mode == tf.estimator.ModeKeys.TRAIN:
                return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
            else:
                return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metrics)
        else:
            (predicted_labels, log_probs) = create_model(is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

            predictions = {
                'probabilities': log_probs,
                'labels': predicted_labels
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Return the actual model function in the closure
    return model_fn


# Compute train and warmup steps from batch size
# These hyperparameters are copied from this colab notebook (https://colab.sandbox.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb)
BATCH_SIZE = 10 # 32
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 10.0
# Warmup is a period of time where hte learning rate
# is small and gradually increases--usually helps training.
WARMUP_PROPORTION = 0.1
# Model configs
SAVE_CHECKPOINTS_STEPS = 500
SAVE_SUMMARY_STEPS = 100

# Compute # train and warmup steps from batch size
num_train_steps = int(len(train_features) / BATCH_SIZE * NUM_TRAIN_EPOCHS)
num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)

run_config = tf.estimator.RunConfig(
    model_dir=OUTPUT_DIR,
    save_summary_steps=SAVE_SUMMARY_STEPS,
    save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS)



model_fn = model_fn_builder(
  num_labels=len(label_list),
  learning_rate=LEARNING_RATE,
  num_train_steps=num_train_steps,
  num_warmup_steps=num_warmup_steps)

estimator = tf.estimator.Estimator(
  model_fn=model_fn,
  config=run_config,
  params={"batch_size": BATCH_SIZE})



train_input_fn = bert.run_classifier.input_fn_builder(
    features=train_features,
    seq_length=MAX_SEQ_LENGTH,
    is_training=True,
    drop_remainder=False)


print(f'Beginning Training!')
current_time = datetime.now()
estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
print("Training took time ", datetime.now() - current_time)

test_input_fn = run_classifier.input_fn_builder(
    features=test_features,
    seq_length=MAX_SEQ_LENGTH,
    is_training=False,
    drop_remainder=False)

val_results = estimator.evaluate(input_fn=test_input_fn, steps=None)
print('Validation Results:')
print(val_results)

'''
Batch = 5 epoch = 10
{'auc': 0.6400001, 'eval_accuracy': 0.64, 'f1_score': 0.6666666, 'false_negatives': 32.0, 'false_positives': 40.0, 'loss': 1.8005009, 'precision': 0.6296296, 'recall': 0.68, 'true_negatives': 60.0, 'true_positives': 68.0, 'global_step': 1600}

Batch = 10 epoch = 10
{'auc': 0.65999997, 'eval_accuracy': 0.66, 'f1_score': 0.6666666, 'false_negatives': 42.0, 'false_positives': 26.0, 'loss': 1.6732528, 'precision': 0.6904762, 'recall': 0.58, 'true_negatives': 74.0, 'true_positives': 58.0, 'global_step': 800}

Batch = 16 epoch 10
{'auc': 0.65, 'eval_accuracy': 0.65, 'f1_score': 0.6666666, 'false_negatives': 43.0, 'false_positives': 27.0, 'loss': 1.6944106, 'precision': 0.6785714, 'recall': 0.57, 'true_negatives': 73.0, 'true_positives': 57.0, 'global_step': 800}

Batch = 10 epoch = 20
{'auc': 0.62, 'eval_accuracy': 0.62, 'f1_score': 0.6666666, 'false_negatives': 39.0, 'false_positives': 37.0, 'loss': 2.6343431, 'precision': 0.622449, 'recall': 0.61, 'true_negatives': 63.0, 'true_positives': 61.0, 'global_step': 1600}

10   100
{'auc': 0.60999995, 'eval_accuracy': 0.61, 'f1_score': 0.6666666, 'false_negatives': 41.0, 'false_positives': 37.0, 'loss': 3.6715355, 'precision': 0.6145833, 'recall': 0.59, 'true_negatives': 63.0, 'true_positives': 59.0, 'global_step': 8000}


'''

'''

def getPrediction(in_sentences):
  labels = ["Negative", "Positive"]
  input_examples = [run_classifier.InputExample(guid="", text_a = x, text_b = None, label = 0) for x in in_sentences] # here, "" is just a dummy label
  input_features = run_classifier.convert_examples_to_features(input_examples, label_list, MAX_SEQ_LENGTH, tokenizer)
  predict_input_fn = run_classifier.input_fn_builder(features=input_features, seq_length=MAX_SEQ_LENGTH, is_training=False, drop_remainder=False)
  predictions = estimator.predict(predict_input_fn)
  return [(sentence, prediction['probabilities'], labels[prediction['labels']]) for sentence, prediction in zip(in_sentences, predictions)]

pred_sentences = [
  "That movie was absolutely awful",
  "The acting was a bit lacking",
  "The film was creative and surprising",
  "Absolutely fantastic!"
]

predictions = getPrediction(pred_sentences)

predictions

'''