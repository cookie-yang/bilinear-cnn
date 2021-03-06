from __future__ import print_function
import tensorflow as tf
import numpy as np
#from scipy.misc import imread, imresize
import tflearn
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import os
from tflearn.data_utils import shuffle

import pickle 
from tflearn.data_utils import image_preloader
import h5py
import math
#import logging
import random
import time




def random_flip_right_to_left(image_batch):
    """
    This function will flip the images randomly.
    Input: batch of images [batch, height, width, channels]
    Output: batch of images flipped randomly [batch, height, width, channels]
    """
    result = []
    for n in range(image_batch.shape[0]):
        if bool(random.getrandbits(1)):     ## With 0.5 probability flip the image
            result.append(image_batch[n][:,::-1,:])
        else:
            result.append(image_batch[n])
    return result



class vgg16:
    def __init__(self, imgs, weights=None, sess=None):
        self.imgs = imgs
        self.last_layer_parameters = []     ## Parameters in this list will be optimized when only last layer is being trained 
        self.parameters = []                ## Parameters in this list will be optimized when whole BCNN network is finetuned
        self.convlayers(self.imgs)                   ## Create Convolutional layers
        self.fc_layers()                    ## Create Fully connected layer
        self.weight_file = weights          


    def convlayers(self,images):
        
        # zero-mean input
        with tf.name_scope('preprocess') as scope:
            # mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            images = images
            print('Adding Data Augmentation')


        # conv1_1
        with tf.variable_scope("conv1_1"):
            weights = tf.get_variable("W", [3,3,3,64], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
             # Create variable named "biases".
            biases = tf.get_variable("b", [64], initializer=tf.constant_initializer(0.1), trainable=True)
            conv = tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding='SAME')
            self.conv1_1 = tf.nn.relu(conv + biases)
            self.parameters += [weights, biases]


        # conv1_2
        with tf.variable_scope("conv1_2"):
            weights = tf.get_variable("W", [3,3,64,64], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
             # Create variable named "biases".
            biases = tf.get_variable("b", [64], initializer=tf.constant_initializer(0.1), trainable=True)
            conv = tf.nn.conv2d(self.conv1_1, weights, strides=[1, 1, 1, 1], padding='SAME')
            self.conv1_2 = tf.nn.relu(conv + biases)
            self.parameters += [weights, biases]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')

        # conv2_1
        with tf.variable_scope("conv2_1"):
            weights = tf.get_variable("W", [3,3,64,128], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
             # Create variable named "biases".
            biases = tf.get_variable("b", [128], initializer=tf.constant_initializer(0.1), trainable=True)
            conv = tf.nn.conv2d(self.pool1, weights, strides=[1, 1, 1, 1], padding='SAME')
            self.conv2_1 = tf.nn.relu(conv + biases)
            self.parameters += [weights, biases]



        # conv2_2
        with tf.variable_scope("conv2_2"):
            weights = tf.get_variable("W", [3,3,128,128], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
             # Create variable named "biases".
            biases = tf.get_variable("b", [128], initializer=tf.constant_initializer(0.1), trainable=True)
            conv = tf.nn.conv2d(self.conv2_1, weights, strides=[1, 1, 1, 1], padding='SAME')
            self.conv2_2 = tf.nn.relu(conv + biases)
            self.parameters += [weights, biases]


        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')

        # conv3_1
        with tf.variable_scope("conv3_1"):
            weights = tf.get_variable("W", [3,3,128,256], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
             # Create variable named "biases".
            biases = tf.get_variable("b", [256], initializer=tf.constant_initializer(0.1), trainable=True)
            conv = tf.nn.conv2d(self.pool2, weights, strides=[1, 1, 1, 1], padding='SAME')
            self.conv3_1 = tf.nn.relu(conv + biases)
            self.parameters += [weights, biases]


        # conv3_2
        with tf.variable_scope("conv3_2"):
            weights = tf.get_variable("W", [3,3,256,256], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
             # Create variable named "biases".
            biases = tf.get_variable("b", [256], initializer=tf.constant_initializer(0.1), trainable=True)
            conv = tf.nn.conv2d(self.conv3_1, weights, strides=[1, 1, 1, 1], padding='SAME')
            self.conv3_2 = tf.nn.relu(conv + biases)
            self.parameters += [weights, biases]

        # conv3_3
        with tf.variable_scope("conv3_3"):
            weights = tf.get_variable("W", [3,3,256,256], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
             # Create variable named "biases".
            biases = tf.get_variable("b", [256], initializer=tf.constant_initializer(0.1), trainable=True)
            conv = tf.nn.conv2d(self.conv3_2, weights, strides=[1, 1, 1, 1], padding='SAME')
            self.conv3_3 = tf.nn.relu(conv + biases)
            self.parameters += [weights, biases]


        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool3')

        # conv4_1
        with tf.variable_scope("conv4_1"):
            weights = tf.get_variable("W", [3,3,256,512], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
             # Create variable named "biases".
            biases = tf.get_variable("b", [512], initializer=tf.constant_initializer(0.1), trainable=True)
            conv = tf.nn.conv2d(self.pool3, weights, strides=[1, 1, 1, 1], padding='SAME')
            self.conv4_1 = tf.nn.relu(conv + biases)
            self.parameters += [weights, biases]


        # conv4_2
        with tf.variable_scope("conv4_2"):
            weights = tf.get_variable("W", [3,3,512,512], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
             # Create variable named "biases".
            biases = tf.get_variable("b", [512], initializer=tf.constant_initializer(0.1), trainable=True)
            conv = tf.nn.conv2d(self.conv4_1, weights, strides=[1, 1, 1, 1], padding='SAME')
            self.conv4_2 = tf.nn.relu(conv + biases)
            self.parameters += [weights, biases]


        # conv4_3
        with tf.variable_scope("conv4_3"):
            weights = tf.get_variable("W", [3,3,512,512], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
             # Create variable named "biases".
            biases = tf.get_variable("b", [512], initializer=tf.constant_initializer(0.1), trainable=True)
            conv = tf.nn.conv2d(self.conv4_2, weights, strides=[1, 1, 1, 1], padding='SAME')
            self.conv4_3 = tf.nn.relu(conv + biases)
            self.parameters += [weights, biases]

        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

        # conv5_1
        with tf.variable_scope("conv5_1"):
            weights = tf.get_variable("W", [3,3,512,512], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
             # Create variable named "biases".
            biases = tf.get_variable("b", [512], initializer=tf.constant_initializer(0.1), trainable=True)
            conv = tf.nn.conv2d(self.pool4, weights, strides=[1, 1, 1, 1], padding='SAME')
            self.conv5_1 = tf.nn.relu(conv + biases)
            self.parameters += [weights, biases]


        # conv5_2
        with tf.variable_scope("conv5_2"):
            weights = tf.get_variable("W", [3,3,512,512], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
             # Create variable named "biases".
            biases = tf.get_variable("b", [512], initializer=tf.constant_initializer(0.1), trainable=True)
            conv = tf.nn.conv2d(self.conv5_1, weights, strides=[1, 1, 1, 1], padding='SAME')
            self.conv5_2 = tf.nn.relu(conv + biases)
            self.parameters += [weights, biases]
            

        # conv5_3
        with tf.variable_scope("conv5_3"):
            weights = tf.get_variable("W", [3,3,512,512], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
             # Create variable named "biases".
            biases = tf.get_variable("b", [512], initializer=tf.constant_initializer(0.1), trainable=True)
            conv = tf.nn.conv2d(self.conv5_2, weights, strides=[1, 1, 1, 1], padding='SAME')
            self.conv5_3 = tf.nn.relu(conv + biases)

            self.parameters += [weights, biases]
            self.special_parameters = [weights,biases]
        self.conv5_3 = tf.transpose(self.conv5_3, perm=[0,3,1,2])
        
        self.conv5_3 = tf.reshape(self.conv5_3,[-1,512,784])

        
        conv5_3_T = tf.transpose(self.conv5_3, perm=[0,2,1])          
        self.phi_I = tf.matmul(self.conv5_3, conv5_3_T)                 #Matrix multiplication [batch_size,512,784] x [batch_size,784,512]

    
        self.phi_I = tf.reshape(self.phi_I,[-1,512*512])               
        #Reshape from [batch_size,512,512] to [batch_size, 512*512] ""
        print('Shape of phi_I after reshape', self.phi_I.get_shape())

        self.phi_I = tf.divide(self.phi_I,784.0)  
        print('Shape of phi_I after division', self.phi_I.get_shape())  

        self.y_ssqrt = tf.multiply(tf.sign(self.phi_I),tf.sqrt(tf.abs(self.phi_I)+1e-12))       
        #"""Take signed square root of phi_I"""
        print('Shape of y_ssqrt', self.y_ssqrt.get_shape())

        self.z_l2 = tf.nn.l2_normalize(self.y_ssqrt, dim=1)     
        #"""Apply l2 normalization"""
        print('Shape of z_l2', self.z_l2.get_shape())




    def fc_layers(self):

        with tf.variable_scope('fc-new') as scope:
            fc3w = tf.get_variable('W', [512*512, 5], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
            #fc3b = tf.Variable(tf.constant(1.0, shape=[100], dtype=tf.float32), name='biases', trainable=True)
            fc3b = tf.get_variable("b", [5], initializer=tf.constant_initializer(0.1), trainable=True)
            self.fc3l = tf.nn.bias_add(tf.matmul(self.z_l2, fc3w), fc3b)
            self.last_layer_parameters += [fc3w, fc3b]

    def load_initial_weights(self, session):

        """weight_dict contains weigths of VGG16 layers"""
        weights_dict = np.load(self.weight_file, encoding = 'bytes')

        
        """Loop over all layer names stored in the weights dict Load only conv-layers. Skip fc-layers in VGG16"""
        vgg_layers = ['conv1_1','conv1_2','conv2_1','conv2_2','conv3_1','conv3_2','conv3_3','conv4_1','conv4_2','conv4_3','conv5_1','conv5_2','conv5_3']
        
        for op_name in vgg_layers:
            with tf.variable_scope(op_name, reuse = True):
                
              # Loop over list of weights/biases and assign them to their corresponding tf variable
                # Biases
              
              var = tf.get_variable('b', trainable = True)
              print('Adding weights to',var.name)
              session.run(var.assign(weights_dict[op_name+'_b']))
                  
            # Weights
              var = tf.get_variable('W', trainable = True)
              print('Adding weights to',var.name)
              session.run(var.assign(weights_dict[op_name+'_W']))

        # with tf.variable_scope('fc-new', reuse = True):
        #     """
        #     Load fc-layer weights trained in the first step. 
        #     Use file .py to train last layer
        #     """
        #     last_layer_weights = np.load('last_layers_epoch_10.npz')
        #     print('Last layer weights: last_layers_epoch_10.npz')
        #     var = tf.get_variable('W', trainable = True)
        #     print('Adding weights to',var.name)
        #     session.run(var.assign(last_layer_weights['arr_0'][0]))
        #     var = tf.get_variable('b', trainable = True)
        #     print('Adding weights to',var.name)
        #     session.run(var.assign(last_layer_weights['arr_0'][1]))



def _parse_function(example_proto):
    keys_to_features = {
        "image/encoded":tf.VarLenFeature(tf.string),
        "image/height":tf.FixedLenFeature([],tf.int64),
        "image/width":tf.FixedLenFeature([],tf.int64),
        "image/class/label": tf.VarLenFeature(tf.string)
    }
    parsed_features = tf.parse_single_example(example_proto, keys_to_features)

    image_decoded = tf.image.decode_jpeg(
        parsed_features['image/encoded'].values[0], channels=3, try_recover_truncated=True, acceptable_fraction=0.3)
    # height = tf.cast(parsed_features["image/height"],tf.int32)
    # width = tf.cast(parsed_features["image/width"],tf.int32)
    size = tf.constant([448,448])
    image = tf.image.resize_images(
        image_decoded, size)
    label = label.decode('utf-8')
    index = label.index('y')
    depth = len(label)
    label = tf.one_hot(index, depth)
    return image,label


if __name__ == '__main__':
    batch_size = 16
    saving_interval = 2000
    initialize = True

    # data_path = "/data1/zhaoweiqiang/workspace/models/research/slim/dfdata/deepfasion/deepfasion_train.tfrecord"
    data_path = tf.placeholder(tf.string,shape=[None])
    train_dataset = tf.data.TFRecordDataset(data_path)
    train_dataset = train_dataset.map(_parse_function)
    train_dataset = train_dataset.apply(tf.contrib.data.shuffle_and_repeat(30000))
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(2)
    iterator = train_dataset.make_initializable_iterator()
    iterator1 = train_dataset.make_initializable_iterator()

    train_data_names = ["/data2/neonyang/fine-grained/data/fashionAI_train.tfrecord"]
    vali_data_names = ["/data2/neonyang/fine-grained/data/fashionAI_vali.tfrecord"]
    
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)     ## Start session to create training graph
    sess.run(iterator.initializer, feed_dict={data_path: train_data_names})
    sess.run(iterator1.initializer, feed_dict={data_path: vali_data_names})
    imgs,target = iterator.get_next()
    imgs_vali,target_vali = iterator1.get_next()

    #print 'Creating graph'
    vgg = vgg16(imgs, 'vgg16_weights.npz', sess)

    print('VGG network created')
    
    # # Defining other ops using Tensorflow
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=vgg.fc3l, labels=target))


    # print([_.name for _ in vgg.parameters])
    print([_.name for _ in tf.trainable_variables()])
    
    print([_.name for _ in vgg.parameters])

    


    fc_paramters = [x for x in tf.trainable_variables()
                    if x not in vgg.parameters]
    
    print([_.name for _ in fc_paramters])
    optimizer_fc = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9).minimize(loss,var_list=fc_paramters)
    optimizer_all = tf.train.MomentumOptimizer(
        learning_rate=0.001, momentum=0.9).minimize(loss)

    
    check_op = tf.add_check_numerics_ops()

    correct_prediction = tf.equal(tf.argmax(vgg.fc3l,1), tf.argmax(target,1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # num_correct_preds = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

    sess.run(tf.global_variables_initializer())

    if initialize:
        vgg.load_initial_weights(sess)
    
  
    saver = tf.train.Saver()
    print('Starting training')

    lr = 0.001
    finetune_step = 5000

    for step in range(1000000):
        avg_cost = 0.
        # print("Data shapes -- (train, val, test)", batch_xs.shape)
        cost = sess.run(loss)
        if step<= finetune_step:
                sess.run([optimizer_fc, check_op])
        else:
            sess.run([optimizer_all, check_op])
        if step % saving_interval == 0:
            print("saved model")
            saver.save(sess,'./model/blcnn.ckpt')
        accuracy1 = accuracy.eval(session=sess)
        vgg.imgs = imgs_vali
        accuracy = accuracy.eval(session=sess)
        print("Step:", '%03d' % step, "Loss:", str(cost), "Accuracy:", accuracy,"Accuracy1:", accuracy1,end='\r')
        vgg.imgs = imgs
                


