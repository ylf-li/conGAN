import numpy as np
import tensorflow as tf
import tensorlayer as tl
from utils import *
from tensorlayer.layers import *

class vgg16:
    def __init__(self, imgs):
        self.imgs = imgs
        self.convlayers()
        self.deconde()

    def convlayers(self):
        self.parameters = []

        with tf.name_scope('enconde') as scope:
            # zero-mean input
            with tf.name_scope('preprocess') as scope:
                mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
                images = self.imgs-mean

            # conv1_1
            with tf.name_scope('conv1_1') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,stddev=1e-1),name='weights')
                conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv1_1 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # conv1_2
            with tf.name_scope('conv1_2') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,stddev=1e-1),name='weights')
                conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv1_2 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # pool1

            self.pool1 = tf.nn.max_pool(self.conv1_2,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME',name='pool1')

            # conv2_1
            with tf.name_scope('conv2_1') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,stddev=1e-1),name='weights')
                conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv2_1 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # conv2_2
            with tf.name_scope('conv2_2') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,stddev=1e-1), name='weights')
                conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv2_2 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # pool2

            self.pool2 = tf.nn.max_pool(self.conv2_2,
                                   ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='SAME',
                                   name='pool2')

            # conv3_1
            with tf.name_scope('conv3_1') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,stddev=1e-1),name='weights')
                conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv3_1 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # conv3_2
            with tf.name_scope('conv3_2') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,stddev=1e-1),name='weights')
                conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv3_2 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # conv3_3
            with tf.name_scope('conv3_3') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,stddev=1e-1),name='weights')
                conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv3_3 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # pool3

            self.pool3 = tf.nn.max_pool(self.conv3_3,
                                   ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='SAME',
                                   name='pool3')

            # conv4_1
            with tf.name_scope('conv4_1') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,stddev=1e-1), name='weights')
                conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv4_1 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # conv4_2
            with tf.name_scope('conv4_2') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,stddev=1e-1),name='weights')
                conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv4_2 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # conv4_3
            with tf.name_scope('conv4_3') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,stddev=1e-1),name='weights')
                conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv4_3 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # pool4
            self.pool4 = tf.nn.max_pool(self.conv4_3,
                                   ksize=[1, 2, 2, 1],
                                   strides=[1, 1,1, 1],
                                   padding='SAME',
                                   name='pool4')

            # conv5_1
            with tf.name_scope('conv5_1') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,stddev=1e-1), name='weights')
                conv = tf.nn.atrous_conv2d(self.pool4, kernel,2, padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv5_1 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # conv5_2
            with tf.name_scope('conv5_2') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,stddev=1e-1),name='weights')
                conv = tf.nn.atrous_conv2d(self.conv5_1, kernel, 2, padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv5_2 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # conv5_3
            with tf.name_scope('conv5_3') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,stddev=1e-1), name='weights')
                conv = tf.nn.atrous_conv2d(self.conv5_2, kernel, 2, padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv5_3 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]


    def deconde(self):

        with tf.variable_scope("deconde"):

            # img_W=416 ;img_H=544
            # img_W=480;img_H=320
            img_W=224;img_H=224
            batch_size = tf.shape(self.conv5_3)[0]
            w_init = tf.contrib.layers.xavier_initializer_conv2d()
            # w_init = tf.contrib.keras.initializers.he_normal()
            #14

            conv5_1 = InputLayer(self.conv5_1,name='g/h0/conv5_1')
            conv5_1 = Conv2d(conv5_1, 32, (1,1), (1, 1),act=tf.identity,padding='SAME', W_init=w_init, name='g/h1/conv51')
            conv5_2 = InputLayer(self.conv5_2,name='g/h0/conv5_2')
            conv5_2 = Conv2d(conv5_2, 32, (1,1), (1, 1),act=tf.identity,padding='SAME', W_init=w_init, name='g/h1/conv52')
            conv5_3 = InputLayer(self.conv5_3,name='g/h0/conv5_3')
            conv5_3 = Conv2d(conv5_3, 32, (1,1), (1, 1),act=tf.identity,padding='SAME', W_init=w_init, name='g/h1/conv53')
			
            conv4_1 = InputLayer(self.conv4_1,name='g/h2/conv4_1')
            conv4_1 = Conv2d(conv4_1, 32, (1,1), (1, 1),act=tf.identity,padding='SAME', W_init=w_init, name='g/h2/conv41')
            conv4_2 = InputLayer(self.conv4_2,name='g/h2/conv4_2')
            conv4_2 = Conv2d(conv4_2, 32, (1,1), (1, 1),act=tf.identity,padding='SAME', W_init=w_init, name='g/h2/conv42')
            conv4_3 = InputLayer(self.conv4_3,name='g/h2/conv4_3')
            conv4_3 = Conv2d(conv4_3, 32, (1,1), (1, 1),act=tf.identity,padding='SAME', W_init=w_init, name='g/h2/conv43')
            concat=tl.layers.ConcatLayer([conv5_1,conv5_2,conv5_3,conv4_1,conv4_2,conv4_3],concat_dim=3,name='g/h2/concat')
            concat=Conv2d(concat, 512, (3,3), (1, 1),act=tf.nn.relu,padding='SAME', W_init=w_init, name='g/h12/conv')
			
            bilinear_init = deconv2d_bilinear_upsampling_initializer(shape=[3,3,1,1])
            self.deconv1=tl.layers.DeConv2dLayer(concat,shape = [3,3,256,512],output_shape = [batch_size,100,100,256],
                            strides=[1,2,2,1],W_init=w_init,padding='SAME',act=tf.identity, name='g/h1/decon2d')              
            self.deconv1=Conv2d(self.deconv1, 256, (3,3), (1, 1),act=tf.nn.relu,padding='SAME', W_init=w_init, name='g/h1/deconv')
            conv3_1 = InputLayer(self.conv3_1,name='g/h3/conv3_1')
            conv3_1 = Conv2d(conv3_1, 32, (1,1), (1, 1),act=tf.identity,padding='SAME', W_init=w_init, name='g/h3/conv31')
            conv3_2 = InputLayer(self.conv3_2,name='g/h3/conv3_2')
            conv3_2 = Conv2d(conv3_2, 32, (1,1), (1, 1),act=tf.identity,padding='SAME', W_init=w_init, name='g/h3/conv32')
            conv3_3 = InputLayer(self.conv3_3,name='g/h3/conv3_3')
            conv3_3 = Conv2d(conv3_3, 32, (1,1), (1, 1),act=tf.identity,padding='SAME', W_init=w_init, name='g/h3/conv33')
            concat=tl.layers.ConcatLayer([self.deconv1,conv3_1,conv3_2,conv3_3],concat_dim=3,name='g/h3/concat')
            concat=Conv2d(concat, 256, (3,3), (1, 1), act=tf.nn.relu,padding='SAME', W_init=w_init, name='g/h3/conv311')
			
            bilinear_init = deconv2d_bilinear_upsampling_initializer(shape=[3,3,1,1])
            self.deconv2=tl.layers.DeConv2dLayer(concat,shape = [3,3,128,256],output_shape = [batch_size,200,200,128],
                            strides=[1,2,2,1],W_init=w_init,padding='SAME',act=tf.identity, name='g/h2/decon2d')            
            self.deconv2=Conv2d(self.deconv2, 128, (3,3), (1, 1),act=tf.nn.relu,padding='SAME', W_init=w_init, name='g/h3/conv')
            conv2_1 = InputLayer(self.conv2_1,name='g/h4/conv2_1')
            conv2_1 = Conv2d(conv2_1, 32, (1,1), (1, 1),act=tf.identity,padding='SAME', W_init=w_init, name='g/h4/conv22')
            conv2_2 = InputLayer(self.conv2_2,name='g/h4/conv2_2')
            conv2_2 = Conv2d(conv2_2, 32, (1,1), (1, 1),act=tf.identity,padding='SAME', W_init=w_init, name='g/h4/conv21')
            concat=tl.layers.ConcatLayer([self.deconv2,conv2_1,conv2_2],concat_dim=3,name='g/h4/concat')
            concat=Conv2d(concat, 128, (3,3), (1, 1),act=tf.nn.relu,padding='SAME', W_init=w_init, name='g/h4/conv2')
			
            bilinear_init = deconv2d_bilinear_upsampling_initializer(shape=[3,3,1,1])			
            self.deconv3=tl.layers.DeConv2dLayer(concat,shape = [3,3,64,128],output_shape = [batch_size,400,400,64],
                            strides=[1,2,2, 1],W_init=w_init,padding='SAME',act=tf.identity, name='g/h4/decon2d')
            self.deconv3=Conv2d(self.deconv3, 64, (3,3), (1, 1),act=tf.nn.relu,padding='SAME', W_init=w_init, name='g/h2/conv')
            conv1_1 = InputLayer(self.conv1_1,name='g/h5/conv1_1')
            conv1_1 = Conv2d(conv1_1, 32, (1,1), (1, 1),act=tf.identity,padding='SAME', W_init=w_init, name='g/h5/conv12')
            conv1_2 = InputLayer(self.conv1_2,name='g/h5/conv1_2')
            conv1_2 = Conv2d(conv1_2, 32, (1,1), (1, 1),act=tf.identity,padding='SAME', W_init=w_init, name='g/h5/conv11')
            concat=tl.layers.ConcatLayer([self.deconv3,conv1_1,conv1_2],concat_dim=3,name='g/h5/concat')
            concat=Conv2d(concat, 32, (3,3), (1, 1),act=tf.nn.relu,padding='SAME', W_init=w_init, name='g/h5/conv1')

            self.pred=Conv2d(concat, 1, (1,1), (1, 1),padding='SAME', W_init=w_init, name='g/pre/conv')

