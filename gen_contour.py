import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import numpy as np
from utils import *
from model import *

from vgg16_inference import vgg16
import glob
import os
import sys
import random
import shutil
from scipy.misc import imread, imresize,imsave
import matplotlib.pyplot as plt


batch_size=1
img_size=256
crop_size=448
train=True
test=False

Img_path='data/BSD500/images/'

summaries_dir='summary'

image_name = '/opt/code/contour/data/BSD500/raw_images/374020.jpg'
# image_name = '/opt/code/contour/data/NYUD/raw_images/img_5632.png'
img_W=321;img_H=481

input_img=tf.placeholder(tf.float32,name='input_image')

vgg=vgg16(input_img,img_W,img_H)
Contour=tf.squeeze(vgg.pred.outputs)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
    print('Model restored')
    saver.restore(sess,"/opt/code/contour/models/model_BSD_448.ckpt")

    imgs_raw=imread(image_name,mode='RGB')
    imgs_raw=imgs_raw[0:320,0:480]

    pool1,pool2,pool3,pool4,pool5,res1,res2,res3,res4,res5,res= sess.run([vgg.pool1,vgg.pool2,vgg.pool3,vgg.pool4,vgg.pool5,\
    				vgg.net_h1.outputs,vgg.net_h2.outputs,vgg.net_h3.outputs,\
    				vgg.net_h4.outputs,vgg.net_h5.outputs,vgg.pred.outputs],feed_dict={vgg.imgs:[imgs_raw]})

    np.save(os.path.join('featuremaps',os.path.basename('pool1'+str(1))),pool1)
    np.save(os.path.join('featuremaps',os.path.basename('pool2'+str(2))),pool2)
    np.save(os.path.join('featuremaps',os.path.basename('pool3'+str(3))),pool3)
    np.save(os.path.join('featuremaps',os.path.basename('pool4'+str(4))),pool4)
    np.save(os.path.join('featuremaps',os.path.basename('pool5'+str(5))),pool5)

    np.save(os.path.join('featuremaps',os.path.basename('featuremaps'+str(1))),res1)
    np.save(os.path.join('featuremaps',os.path.basename('featuremaps'+str(2))),res2)
    np.save(os.path.join('featuremaps',os.path.basename('featuremaps'+str(3))),res3)
    np.save(os.path.join('featuremaps',os.path.basename('featuremaps'+str(4))),res4)
    np.save(os.path.join('featuremaps',os.path.basename('featuremaps'+str(5))),res5)
    imsave(os.path.join('featuremaps',os.path.basename('results.png')),1./(1.+np.exp(-res[0,:,:,0])))
