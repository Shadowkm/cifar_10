
# coding: utf-8

# ## This notebook is modfied by [Udacity deeplearning courses github](https://github.com/udacity/deep-learning)

# # Image Classification


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm
import problem_unittests as tests
import tarfile


cifar10_dataset_folder_path = 'cifar-10-batches-py'

# Use Floyd's cifar-10 dataset if present
floyd_cifar10_location = '/input/cifar-10/python.tar.gz'
if isfile(floyd_cifar10_location):
    tar_gz_path = floyd_cifar10_location
else:
    tar_gz_path = 'cifar-10-python.tar.gz'

class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

if not isfile(tar_gz_path):
    with DLProgress(unit='B', unit_scale=True, miniters=1, desc='CIFAR-10 Dataset') as pbar:
        urlretrieve(
            'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
            tar_gz_path,
            pbar.hook)

if not isdir(cifar10_dataset_folder_path):
    with tarfile.open(tar_gz_path) as tar:
        tar.extractall()
        tar.close()


tests.test_folder_path(cifar10_dataset_folder_path)




get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format = 'retina'")

import helper
import numpy as np

# 更改 batch_id, sample_id 看不同的圖片資料
batch_id = 2
sample_id =10
helper.display_stats(cifar10_dataset_folder_path, batch_id, sample_id)


def normalize(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (32, 32, 3)
    : return: Numpy array of normalize data
    """
    # TODO: Implement Function
    return np.array(x / np.max(x))


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_normalize(normalize)



import pandas as pd
def one_hot_encode(x):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
    """
    # TODO: Implement Function
    # HINT: google "np.eye" or use label encoder from sklearn
    print(len(x))
    return np.eye(10)[x]


tests.test_one_hot_encode(one_hot_encode)



helper.preprocess_and_save_data(cifar10_dataset_folder_path, normalize, one_hot_encode)


import pickle
import problem_unittests as tests
import helper

valid_features, valid_labels = pickle.load(open('preprocess_validation.p', mode='rb'))


import tensorflow as tf

def neural_net_image_input(image_shape):

    image = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], image_shape[2]], name='x')
    return image 


def neural_net_label_input(n_classes):

    label = tf.placeholder(tf.float32, [None, n_classes], name='y')
    return label


def neural_net_keep_prob_input():

    keep_prob = tf.placeholder(tf.float32, name="keep_prob" )
    return keep_prob


tf.reset_default_graph()
tests.test_nn_image_inputs(neural_net_image_input)
tests.test_nn_label_inputs(neural_net_label_input)
tests.test_nn_keep_prob_inputs(neural_net_keep_prob_input)



def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):


    conv_layer = tf.layers.conv2d(x_tensor, conv_num_outputs, kernel_size=conv_ksize, strides=conv_strides, activation=tf.nn.relu)   
    conv_layer = tf.layers.max_pooling2d(conv_layer, pool_size=pool_ksize, strides=pool_strides)

    return conv_layer 


tests.test_con_pool(conv2d_maxpool)


def flatten(x_tensor):

    return tf.layers.flatten(x_tensor)


tests.test_flatten(flatten)


def fully_conn(x_tensor, num_outputs):
 
    return tf.layers.dense(x_tensor, num_outputs, activation=tf.nn.relu)


tests.test_fully_conn(fully_conn)



def output(x_tensor, num_outputs):

    return tf.layers.dense(x_tensor, num_outputs, activation=None)

tests.test_output(output)


def conv_net(x, keep_prob):

    conv_ksize = (3,3)
    conv_strides = (1,1)
    pool_ksize = (2,2)
    pool_strides = (2,2)
    n_filters_1 = 64
    n_filters_2 = 128
    
    conv_1 = conv2d_maxpool(x, n_filters_1, conv_ksize, conv_strides, pool_ksize, pool_strides)
    conv_2 = conv2d_maxpool(conv_1, n_filters_1, conv_ksize, conv_strides, pool_ksize, pool_strides)

    fc = flatten(conv_2)

    fc1 = fully_conn(fc, 300)
    
    # apply dropout
    fc2 = tf.nn.dropout(fc1, keep_prob=keep_prob)
 
    
    out = output(fc2, 10)
    
    # TODO: return output
    return out



tf.reset_default_graph()

# Inputs
x = neural_net_image_input((32, 32, 3))
y = neural_net_label_input(10)
keep_prob = neural_net_keep_prob_input()

# Model 的 output (注意！這時候還沒有經過 softmax function，這些 output 的值通常稱為 logits)
logits = conv_net(x, keep_prob)

# Name logits Tensor, so that is can be loaded from disk after training
logits = tf.identity(logits, name='logits')

# Loss and Optimizer
# 此時透過 softmax_cross_entropy_with_logits 將 model output 的值經過 softmax，再與我們的 label 計算 cross-entropy
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer().minimize(cost)

# Accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')



def train_neural_network(session, optimizer, keep_probability, feature_batch, label_batch):
    """
    Optimize the session on a batch of images and labels
    : session: Current TensorFlow session
    : optimizer: TensorFlow optimizer function
    : keep_probability: keep probability
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    """
    # TODO: Implement Function
    session.run(optimizer, feed_dict={
    x: feature_batch,
    y: label_batch,
    keep_prob: keep_probability})



"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_train_nn(train_neural_network)

def print_stats(session, feature_batch, label_batch, cost, accuracy):

    loss = session.run(cost, feed_dict = {
            x: feature_batch,
            y: label_batch,
            keep_prob: 1.
        })
    valid_acc = session.run(accuracy, feed_dict = {
            x: valid_features,
            y: valid_labels,
            keep_prob: 1.
        })
    print('Epoch {:>2}'
          'Loss: {:>10.4f} Validation Accuracy: {:.6f}'.format(
        epoch + 1,
        loss,
        valid_acc))




# TODO: Tune Parameters
epochs = 50
batch_size = 128
keep_probability = 0.5



"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
print('Checking the Training on a Single Batch...')
with tf.Session() as sess:
    # Initializing the variables
    sess.run(tf.global_variables_initializer())
    
    # Training cycle
    for epoch in range(epochs):
        batch_i = 1
        for batch_features, batch_labels in helper.load_preprocess_training_batch(batch_i, batch_size):
            train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
        print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
        print_stats(sess, batch_features, batch_labels, cost, accuracy)



"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
save_model_path = './image_classification'

print('Training...')
with tf.Session() as sess:
    # Initializing the variables
    sess.run(tf.global_variables_initializer())
    
    # Training cycle
    for epoch in range(epochs):
        # Loop over all batches
        n_batches = 5
        for batch_i in range(1, n_batches + 1):
            for batch_features, batch_labels in helper.load_preprocess_training_batch(batch_i, batch_size):
                train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
            print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
            print_stats(sess, batch_features, batch_labels, cost, accuracy)
            
    # Save Model
    saver = tf.train.Saver()
    save_path = saver.save(sess, save_model_path)
    print('-----model save ------')


"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format = 'retina'")

import tensorflow as tf
import pickle
import helper
import random

# Set batch size if not already set
try:
    if batch_size:
        pass
except NameError:
    batch_size = 64

save_model_path = './image_classification'
n_samples = 4
top_n_predictions = 3

def test_model():
    """
    Test the saved model against the test dataset
    """

    test_features, test_labels = pickle.load(open('preprocess_test.p', mode='rb'))
    loaded_graph = tf.Graph()

    with tf.Session(graph=loaded_graph) as sess:
        # Load model
        loader = tf.train.import_meta_graph(save_model_path + '.meta')
        loader.restore(sess, save_model_path)

        # Get Tensors from loaded model
        loaded_x = loaded_graph.get_tensor_by_name('x:0')
        loaded_y = loaded_graph.get_tensor_by_name('y:0')
        loaded_keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
        loaded_logits = loaded_graph.get_tensor_by_name('logits:0')
        loaded_acc = loaded_graph.get_tensor_by_name('accuracy:0')
        
        # Get accuracy in batches for memory limitations
        test_batch_acc_total = 0
        test_batch_count = 0
        
        for train_feature_batch, train_label_batch in helper.batch_features_labels(test_features, test_labels, batch_size):
            test_batch_acc_total += sess.run(
                loaded_acc,
                feed_dict={loaded_x: train_feature_batch, loaded_y: train_label_batch, loaded_keep_prob: 1.0})
            test_batch_count += 1

        print('Testing Accuracy: {}\n'.format(test_batch_acc_total/test_batch_count))

        # Print Random Samples
        random_test_features, random_test_labels = tuple(zip(*random.sample(list(zip(test_features, test_labels)), n_samples)))
        random_test_predictions = sess.run(
            tf.nn.top_k(tf.nn.softmax(loaded_logits), top_n_predictions),
            feed_dict={loaded_x: random_test_features, loaded_y: random_test_labels, loaded_keep_prob: 1.0})
        helper.display_image_predictions(random_test_features, random_test_labels, random_test_predictions)


test_model()



get_ipython().system('rm cifar-10-python.tar.gz')
get_ipython().system('rm preprocess_batch_1.p')
get_ipython().system('rm preprocess_batch_2.p')
get_ipython().system('rm preprocess_batch_3.p')
get_ipython().system('rm preprocess_batch_4.p')
get_ipython().system('rm preprocess_batch_5.p')
get_ipython().system('rm preprocess_validation.p')
get_ipython().system('rm preprocess_test.p')

