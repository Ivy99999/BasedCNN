import argparse
import datetime
import os
import time
import numpy as np
import tensorflow as tf

from numpy.distutils.fcompiler import str2bool

import data_deal
import word2vec_helpers
from Model import TextCNN

parser = argparse.ArgumentParser(description='LSTM for Classify')
parser.add_argument('--dev_sample_percentage', type=float, default=.1, help='Percentage of the training data to use for validation')
parser.add_argument('--positive_data_file', type=str, default='data/ham_100.utf8', help='train data source')
parser.add_argument('--negative_data_file', type=str, default='data/spam_100.utf8', help='test data source')
parser.add_argument('--num_labels', type=int, default='2', help='label')
parser.add_argument('--embedding_dim', type=int, default=128, help='#sample of each minibatch')
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

timestamp = str(int(time.time()))
out_dir = os.path.join(os.path.curdir, args.save_model, timestamp)
print("Writing to {}\n".format(out_dir))
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Load data
print("Loading data...")
positive_data_file = os.path.join('.', args.positive_data_file)
negative_data_file=os.path.join('.', args.negative_data_file)
# print(positive_data_file)

x_text, y = data_deal.load_positive_negative_data_files(positive_data_file, negative_data_file)

sentences, max_document_length = data_deal.padding_sentences(x_text, '<PADDING>')
x = np.array(word2vec_helpers.embedding_sentences(sentences, embedding_size = args.embedding_dim, file_to_save = os.path.join(out_dir, 'trained_word2vec.model')))

print("x.shape = {}".format(x.shape))
print("y.shape = {}".format(y.shape))

# Save params
training_params_file = os.path.join(out_dir, 'training_params.pickle')
params = {'num_labels' : args.num_labels, 'max_document_length' : max_document_length}
data_deal.saveDict(params, training_params_file)

# Shuffle data randomly
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/test set
# TODO: This is very crude, should use cross-validation
dev_sample_index = -1 * int(args.dev_sample_percentage * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

# Training
# =======================================================
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement = args.allow_soft_placement,
	log_device_placement = args.log_device_placement)
    sess = tf.Session(config = session_conf)
    with sess.as_default():
        cnn = TextCNN(
	    sequence_length = x_train.shape[1],
	    num_classes = y_train.shape[1],
	    embedding_size = args.embedding_dim,
	    filter_sizes = list(map(int, args.filter_sizes.split(","))),
	    num_filters = args.num_filters,
	    l2_reg_lambda = args.l2_reg_lambda)

	# Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=args.num_checkpoints)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: args.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

        # Generate batches
        batches = data_deal.batch_iter(
            list(zip(x_train, y_train)), args.batch_size, args.num_epochs)

        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % args.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_dev, y_dev, writer=dev_summary_writer)
                print("")
            if current_step % args.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))




# print(x_text)
