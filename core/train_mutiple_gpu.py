import tensorflow as tf
import numpy as np
import os
import sys
import re


def cnn_model_fn(features):
    # Input layer


input_layer = tf.reshape(features, [-1, 8, 8, 6])
with tf.variable_scope('conv1') as scope:
    # Convolutional Layer #1
conv1 = tf.layers.conv2d(
    inputs=input_layer,
    filters=192,
    kernel_size=[5, 5],
    padding='same',
    activation=tf.nn.relu
)
dp1 = tf.layers.dropout(conv1, rate=0.4)
bn1 = tf.layers.batch_normalization(dp1)

with tf.variable_scope('conv2') as scope:
    # Convolutional Layer #2
conv2 = tf.layers.conv2d(
    inputs=bn1,
    filters=192,
    kernel_size=[3, 3],
    padding='same',
    activation=tf.nn.relu)
dp2 = tf.layers.dropout(conv2, rate=0.4)
bn2 = tf.layers.batch_normalization(dp2)
with tf.variable_scope('conv3') as scope:
    # Convolutional Layer #3
conv3 = tf.layers.conv2d(
    inputs=bn2,
    filters=192,
    kernel_size=[3, 3],
    padding='same',
    activation=tf.nn.relu)
dp3 = tf.layers.dropout(conv3, rate=0.4)
bn3 = tf.layers.batch_normalization(dp3)
with tf.variable_scope('conv4') as scope:
    # Convolutional Layer #4
conv4 = tf.layers.conv2d(
    inputs=bn3,
    filters=192,
    kernel_size=[3, 3],
    padding='same',
    activation=tf.nn.relu)
dp4 = tf.layers.dropout(conv4, rate=0.4)
bn4 = tf.layers.batch_normalization(dp4)
with tf.variable_scope('conv5') as scope:
    # Convolutional Layer #5
conv5 = tf.layers.conv2d(
    inputs=bn4,
    filters=192,
    kernel_size=[3, 3],
    padding='same',
    activation=tf.nn.relu)
dp5 = tf.layers.dropout(conv5, rate=0.4)
bn5 = tf.layers.batch_normalization(dp5)
with tf.variable_scope('conv6') as scope:
    # Convolutional Layer #6
conv6 = tf.layers.conv2d(
    inputs=bn5,
    filters=192,
    kernel_size=[3, 3],
    padding='same',
    activation=tf.nn.relu)
dp6 = tf.layers.dropout(conv6, rate=0.4)
bn6 = tf.layers.batch_normalization(dp6)
with tf.variable_scope('conv7') as scope:
    # Convolutional Layer #7
conv7 = tf.layers.conv2d(
    inputs=bn6,
    filters=192,
    kernel_size=[3, 3],
    padding='same',
    activation=tf.nn.relu)
dp7 = tf.layers.dropout(conv7, rate=0.4)
bn7 = tf.layers.batch_normalization(dp7)
with tf.variable_scope('conv8') as scope:
    # Convolutional Layer #8
conv8 = tf.layers.conv2d(
    inputs=bn7,
    filters=192,
    kernel_size=[3, 3],
    padding='same',
    activation=tf.nn.relu)
dp8 = tf.layers.dropout(conv8, rate=0.4)
bn8 = tf.layers.batch_normalization(dp8)
# Convolutional Layer #9
# conv13 = tf.layers.conv2d(
# inputs=conv12,
# filters=1,
# kernel_size=[1, 1],
# use_bias=False,
# activation=newBiasAdd
# )
with tf.variable_scope('flatten_layer') as scope:
    # flatten tensor to create softmax layer
flatten_conv8 = tf.reshape(bn8, [-1, 8*8*192])
with tf.variable_scope('logit_layer') as scope:
    # Logits Layer
abstractMoves = tf.layers.dense(inputs=flatten_conv8, units=378)
return abstractMoves


def loss_f(scope, features, labels):


abstractMoves = cnn_model_fn(features)
onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.uint8), depth=378)
tf.add_n
crossL = tf.losses.softmax_cross_entropy(
    onehot_labels=onehot_labels, logits=abstractMoves, scope=scope)
tf.add_to_collection('losses', crossL)
loss = tf.get_collection('losses', scope=scope)
total_loss = tf.add_n(loss, name='total_loss')
for l in loss+[total_loss]:
loss_name = re.sub('tower_[0-9]*/', '', l.op.name)
tf.summary.scalar(loss_name, l)
return total_loss


def average_gradients(tower_grads):


average_grads = []
for grad_and_vars in zip(*tower_grads):
grads = []
for g, _ in grad_and_vars:
expand_g = tf.expand_dims(g, 0)
grads.append(expand_g)
grad = tf.concat(grads, 0)
grad = tf.reduce_mean(grad, 0)
v = grad_and_vars[0][1]
grad_and_var = (grad, v)
average_grads.append(grad_and_var)
return average_grads


def train():


with tf.Graph().as_default(), tf.device('/cpu:0'):
global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(
    0), trainable=False, dtype=tf.int32)
num_bataches_per_epoch = 10
decay_steps = 6
learningRate = tf.train.exponential_decay(learning_rate=0.003,
                                          global_step=tf.train.get_global_step(), decay_steps=decay_steps,
                                          decay_rate=0.5, staircase=True)
optimizer = tf.train.MomentumOptimizer(
    learning_rate=learningRate, momentum=0.9, use_nesterov=True)
tower_grad = []
DATAPATH = "../data/policynetwork/data_policy_one_float32.npz"
if DATAPATH != None:
with np.load(DATAPATH) as data:
data_features = data["feature"]
data_labels = data["label"]
assert data_features.shape[0] == data_labels.shape[0]
print(data_features.shape[0])
data_features = tf.convert_to_tensor(data_features)
data_labels = tf.convert_to_tensor(data_labels)
batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
    [data_features, data_labels], capacity=2*250)
with tf.variable_scope(tf.get_variable_scope()):
for i in range(2):
with tf.device('/gpu:{:d}'.format(i)):
with tf.name_scope('tower_{:d}'.format(i)) as scope:
feature_batch, label_batch = batch_queue.dequeue()
loss = loss_f(scope, feature_batch, label_batch)
tf.get_variable_scope().reuse_variables()
summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope=scope)
grads = optimizer.compute_gradients(loss)
tower_grad.append(grads)
summaries.append(tf.summary.scalar('learning_rate', learningRate))

grads = average_gradients(tower_grad)

for grad, var in grads:
if grad is not None:
summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)
for var in tf.trainable_variables():
summaries.append(tf.summary.histogram(var.op.name, var))
variable_averages = tf.train.ExponentialMovingAverage(0.9, global_step)

variable_average_op = variable_averages.apply(tf.trainable_variables())

train_op = tf.group(apply_gradient_op, variable_average_op)

saver = tf.train.Saver(tf.global_variables())

summary_op = tf.summary.merge(summaries)

init = tf.global_variables_initializer()

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

sess.run(init)

tf.train.start_queue_runners(sess=sess)
summary_writer = tf.summary.FileWriter(
    "./model/nn_on_multiGPU", graph=sess.graph)

for step in range(10):
_, loss_v = sess.run([train_op, loss])
print(loss_v)


def main(unused_argv):

    # read data
train()

if __name__ == "__main__":
tf.app.run()
