# coding: utf-8
from __future__ import print_function

import argparse
import os
from collections import Counter

import tensorflow as tf
import tensorflow.contrib.keras as kr
from numpy.distutils.fcompiler import str2bool

import word2vec_helpers
from Model2 import TextCNN
from data_deal import read_and_clean_zh_file
import numpy as np

parser = argparse.ArgumentParser(description='LSTM for Classify')
parser.add_argument('--dev_sample_percentage', type=float, default=.1, help='Percentage of the training data to use for validation')
parser.add_argument('--positive_data_file', type=str, default='data/ham_100.utf8', help='train data source')
parser.add_argument('--negative_data_file', type=str, default='data/spam_100.utf8', help='test data source')
parser.add_argument('--num_classes', type=int, default='2', help='label')
parser.add_argument('--sequence_length', type=int, default='96', help='length')
parser.add_argument('--embedding_size', type=int, default=128, help='#sample of each minibatch')
parser.add_argument('--filter_sizes', type=str, default= "3,4,5", help='#')
parser.add_argument('--num_filters', type=int, default=128, help='#dim of hidden state')
parser.add_argument('--dropout_keep_prob', type=float, default=0.5, help='dropout keep_prob')

parser.add_argument('--l2_reg_lambda', type=float, default=0.0, help='L2 regularization lambda (default: 0.0)')
parser.add_argument('--batch_size', type=int, default=64, help='Batch Size (default: 64)')
parser.add_argument('--num_epochs', type=int, default=200, help='Number of training epochs (default: 200)')
parser.add_argument('--evaluate_every', type=int, default=100, help='Evalue model on dev set after this many steps (default: 100)')
parser.add_argument('--checkpoint_every', type=int, default=100, help='Save model after this many steps (defult: 100)')
parser.add_argument('--num_checkpoints', type=int, default=5, help='Number of checkpoints to store (default: 5)')
parser.add_argument('--allow_soft_placement', type=str2bool, default=True, help='Allow device soft device placement')
parser.add_argument('--log_device_placement', type=str2bool, default=False, help='Allow device soft device placement')
parser.add_argument('--save_model', type=str, default='best_model', help='train data source')
args = parser.parse_args()

save_dir = '/home/ivy/PycharmProjects/BasedCNN/best_model/1568855551/checkpoints/'
save_path = os.path.join(save_dir, 'model-12500')  # 最佳验证结果保存路径


def read_category():
    """读取分类目录，固定"""
    categories = [ '负向','正向']

    categories = [x for x in categories]

    cat_to_id = dict(zip(categories, range(len(categories))))

    return categories, cat_to_id

def padding_sentences(input_sentences, padding_token, padding_sentence_length = args.sequence_length):
    # print(sentence.split(' ') for sentence in input_sentences)
    # for sentence in input_sentences:
    #     print(sentence.split(' '))
    sentences = [sentence.split(' ') for sentence in input_sentences]
    max_sentence_length = padding_sentence_length if padding_sentence_length is not None else max([len(sentence) for sentence in sentences])
    for sentence in sentences:
        if len(sentence) > max_sentence_length:
            sentence = sentence[:max_sentence_length]
        else:
            sentence.extend([padding_token] * (max_sentence_length - len(sentence)))
    return (sentences, max_sentence_length)



class CnnModel:
    def __init__(self,args):
        # self.config = TCNNConfig()
        self.categories, self.cat_to_id = read_category()
        # self.words, self.word_to_id = read_vocab('text.txt')
        # self.vocab_size = len(self.words)
        self.model = TextCNN(args)
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=self.session, save_path=save_path)  # 读取保存的模型

    def predict(self, message):
        # 支持不论在python2还是python3下训练的模型都可以在2或者3的环境下运行
        data = message
        # data=list(content)
        sentences, max_document_length = padding_sentences(data, '<PADDING>')
        x = np.array(word2vec_helpers.embedding_sentences(sentences, embedding_size=args.embedding_size,file_to_load='./best_model/1568855551/trained_word2vec.model'))
        # print(x.shape)

        feed_dict = {
            self.model.input_x:x,
            self.model.dropout_keep_prob: 1.0
        }
        #最后一层输出
        y_pred_cls = self.session.run(self.model.predictions, feed_dict=feed_dict)
        # y_pred_cls = self.session.run(tf.nn.softmax(self.model.scores), feed_dict=feed_dict)
        y_prob = y_pred_cls.tolist()
        print(y_prob)
        return self.categories[y_pred_cls[0]]


if __name__ == '__main__':
    cnn_model = CnnModel(args)
    test_demo = ['雨后打开后备箱，盖子会一直滴水','简配过了点，没有防撞钢梁。','空间是最满意的，坐7个人很轻松','外观大气，内饰简洁，空间宽敞，安全稳定']

    for i in test_demo:
        print(cnn_model.predict(i))

    # build_vocab(args.positive_data_file,args.positive_data_file,'text.txt',vocab_size=10000)
