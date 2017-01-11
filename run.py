#!/usr/bin/env python3

import os

import tensorflow as tf
import json

from model import dmn
import data_utils

flags = tf.app.flags

# directories
flags.DEFINE_string('data_dir', 'data/babi/en', 'Data directory [data/babi/en]')
flags.DEFINE_string('save_dir', 'save/dmn/', 'Trained model saving directory [save/dmn/model]')
flags.DEFINE_string('train_summary_dir', 'summary/dmn/train/', 'Summary saving directory for train set [ummary/dmn/train]')
flags.DEFINE_string('dev_summary_dir', 'summary/dmn/dev/', 'Summary saving directory for dev set [ummary/dmn/train]')

# babi dataset params
flags.DEFINE_integer('task_id', 1, 'bAbi Task number [1]')
flags.DEFINE_float('val_ratio', 0.1, 'Validation data ratio to training data [0.1]')

# model parameters
flags.DEFINE_string('embed_type', 'glove', 'Type of word2vec - None,glove [glove]')
flags.DEFINE_integer('embed_size', 50, 'The dimension of word embedding - 50,100,200,300 [50]') 
flags.DEFINE_integer('hidden_size', 80, 'The number of neural of hidden layer in rnn [50]')
flags.DEFINE_integer('attention_hidden_size', 80, 'The number of neural of hidden layer in attention machanism [80]')
flags.DEFINE_integer('max_memory_pass', 3, 'Episodic Memory steps [3]')

# training parameters
flags.DEFINE_integer('epoch_size', 256, 'Number of training epochs [256]')
flags.DEFINE_integer('batch_size', 100, 'Batch size of mini-batch for training period [128]')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate at the start [0.001]')
flags.DEFINE_float('weight_decay', 0.1, 'Weight decay - 0 to turn off [0.001]')

# option
flags.DEFINE_integer('train_period',10,'Validation period for train set when traning [10]')
flags.DEFINE_integer('dev_period', 10, 'Validation period for develop set when training [40]')
flags.DEFINE_integer('save_period', 80, 'Saving period when training [80]')
flags.DEFINE_integer('print_period', 1, 'Printing period when training [1]')
flags.DEFINE_bool('train', True, 'True for train, False for test. [True]')
flags.DEFINE_boolean('load', False, 'Start training from saved model? [False]')

FLAGS = flags.FLAGS

def main(_):

     # Download data
    os.system('./download.sh')

    # Load data
    wordtable = data_utils.get_wordtable(FLAGS.embed_size, FLAGS.embed_type)
    train_data = data_utils.get_babi(FLAGS.data_dir, FLAGS.task_id, 'train', FLAGS.batch_size, wordtable)
    test_data =  data_utils.get_babi(FLAGS.data_dir, FLAGS.task_id, 'test', FLAGS.batch_size, wordtable)
    train_data, dev_data = train_data.split(FLAGS.val_ratio)

    # Padding data set
    FLAGS.vocab_size = wordtable.vocab_size
    FLAGS.max_sentence_len, FLAGS.max_question_len, FLAGS.max_facts_len = data_utils.get_max_len(train_data, dev_data, test_data)
    train_data, dev_data, test_data = data_utils.padding_datasets(FLAGS,train_data, dev_data, test_data)

    # Create save directory
    if not os.path.exists(FLAGS.save_dir):
        os.makedirs(FLAGS.save_dir)

    # Save the FlAGS
    path = FLAGS.save_dir + 'FLAGS'
    with open(path, 'w') as file:
        json.dump(FLAGS.__dict__, file)

    # Train model or use model
    model = dmn.DMN_Model(FLAGS, list(wordtable.vocab))
    init = tf.global_variables_initializer() #init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        
        if FLAGS.train:
            if FLAGS.load:
                model.load_model(sess)
            model.train(sess, train_data, dev_data)
        else:
            model.load_model(sess)
            model.test(sess, test_data)


if __name__ == '__main__':
    tf.app.run()


