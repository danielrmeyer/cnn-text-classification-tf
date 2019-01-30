#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
from tensorflow.contrib import learn


def batch_data(layer1, layer2, batch_size=64):
    data_size = len(layer1)
    num_batches_per_epoch = int((data_size-1)/batch_size) + 1
    
    for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            
            batch_layer1_data = layer1[start_index:end_index]
            batch_layer2_data = layer2[start_index:end_index]
            
            yield zip(batch_layer1_data, batch_layer2_data)
            
            
def predict(layer1, layer2):
    
    # Map data into vocabulary
    vocab_path = os.path.join(os.path.curdir, "mlmodels", "vocab")
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    x_layer1 = np.array(list(vocab_processor.transform(layer1)))
    x_layer2 = np.array(list(vocab_processor.transform(layer2)))
     
    # 3D arrays
    x_test = np.dstack([x_layer1, x_layer2])
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[2], x_test.shape[1]))
    
    # Read in the model form the latest checkpoint. There are a lot of 
    # components to load in separately. 
    checkpoint_path = os.path.join(os.path.curdir, "mlmodels", "checkpoints")
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_path)
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=True,
          log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)
    
            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]
    
            # If dropout was used to regularize, get that information.
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
    
            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]
            
            all_predictions  = sess.run(predictions, {input_x: x_test, dropout_keep_prob: 1.0})
    
    return all_predictions
