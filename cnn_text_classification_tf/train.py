#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import datetime
import shutil
import json

from cnn_text_classification_tf import data_helpers
from cnn_text_classification_tf.text_cnn import TextCNN
from tensorflow.contrib import learn

# Define parameters. Also shows up in comand line helper.
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("dict_of_files", "./data_filenames.json", "Json containing the paths to the data files by layer.")

tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "2,3", "Comma-separated filter sizes (default: '2,3')")
tf.flags.DEFINE_integer("num_filters", 4, "Number of filters per filter size (default: 4)")
tf.flags.DEFINE_integer("num_channels", 2, "Number of layers (default: 2)")
tf.flags.DEFINE_float("dropout_keep_prob", 1.0, "Dropout keep probability (default: 1.0)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

tf.flags.DEFINE_integer("batch_size", 256, "Batch Size (default: 256)")
tf.flags.DEFINE_integer("num_epochs", 800, "Number of training epochs (default: 800)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

def preprocess():
    data_files = FLAGS.dict_of_files
    with open(data_files) as f:
        d = json.load(f)
    
    # layer 1
    x_text_layer1, y = data_helpers.load_data_and_labels(
            data_files=d['layer1'],
            labels=[[0,0,1], # same as onehot(2)
                    [0,1,0], # same as onehot(1)
                    [1,0,0]] # same as onehot(0)
            )
     
    # layer 2:
    x_text_layer2, _ = data_helpers.load_data_and_labels(
            data_files=d['layer2'],
            labels=[[2], # Doesn't get used.
                    [1], # Doesn't get used.
                    [0]] # Doesn't get used.
            )
                        
     
    # Build vocabulary
    all_layers = x_text_layer1 + x_text_layer2
    max_document_length = max([len(x.split(" ")) for x in all_layers])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    vocab_processor.fit(all_layers)
     
    # Transform text data to integers with the above vocabulary
    x_layer1 = np.array(list(vocab_processor.transform(x_text_layer1)))
    x_layer2 = np.array(list(vocab_processor.transform(x_text_layer2)))
     
    # 3D arrays.  
    x = np.dstack([x_layer1, x_layer2])
     
    # Randomly shuffle data. This helps the model learn better and create a 
    # less biased dev set.
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]
    
    # Split train/dev set
    # The model will use the train set to learn the parameters and the dev
    # set can be used to tune the hyperparameters seen in the FLAGS. 
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    
    # Reshape to match tensorflows expected input shape
    # (height, num_channels, width)
    # x_train will also need to be reshaped but we'll save it for when we break
    # into batches (see function batch_iter() in data_helpers.py)
    x_dev = x_dev.reshape((x_dev.shape[0], x_dev.shape[2], x_dev.shape[1]))
    
    del x, y, x_shuffled, y_shuffled
     
    return x_train, y_train, vocab_processor, x_dev, y_dev


def train(x_train, y_train, vocab_processor, x_dev, y_dev):
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        
        with sess.as_default():
            
            cnn = TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                num_channels=FLAGS.num_channels,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(learning_rate=0.00001,
                                               beta1=0.9,
                                               beta2=0.999,
                                               epsilon=1e-8)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity.  These can be viewed
            # in the log files with the Tensorboard command.
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "mlmodels"))
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
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
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
            batches = data_helpers.batch_iter(
                x_train, y_train, FLAGS.batch_size, FLAGS.num_epochs)
            
            # Run a train step for each batch. Every so often, check the dev
            # set.
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    print("")
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))

def main(argv=None):
    # If there is a old version of the model stored, delete those model files
    # and create an empty directory. 
    # If there is no model directory, create one to store results for later.
    model_dir = os.path.abspath(os.path.join(os.path.curdir, "mlmodels"))
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir, ignore_errors=True)
        os.makedirs(model_dir)
    else:
        os.makedirs(model_dir)
        
    x_train, y_train, vocab_processor, x_dev, y_dev = preprocess()
    train(x_train, y_train, vocab_processor, x_dev, y_dev)

if __name__ == '__main__':
    tf.app.run()
    
