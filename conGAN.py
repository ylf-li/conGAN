
import glob
import os
import sys
import random
import shutil
import cv2
import numpy as np
from utils import *
from model import *
from vgg16_inference import vgg16

import tensorflow as tf
import tensorflow.contrib as tc
import tensorlayer as tl
from tensorlayer.layers import *
from scipy.misc import imread, imresize,imsave
import matplotlib.pyplot as plt


batch_size=8
epochs=50

start_learning_rate=1e-5
DGAN_learning_rate=1e-5
GGAN_learning_rate=1e-5
weight_decay=0.0002

crop_size_H=400
crop_size_W=400


Img_path='data/BSD500/images/'
GT_path='data/BSD500/gts/'

summaries_dir='summary'
shutil.rmtree('results')
os.mkdir('results')
shutil.rmtree('summary')
os.mkdir('summary')

GT_list=glob.glob(os.path.join(GT_path,'*.png'))
global_steps = tf.Variable(0, trainable=False)

input_img=tf.placeholder(tf.float32,[None,crop_size_H,crop_size_W,3],name='input_image')
GT = tf.placeholder(tf.float32,[None,crop_size_H,crop_size_W],name='GT')

pos_mask = tf.placeholder(tf.float32,[None,crop_size_H,crop_size_W],name='Pen_pos')
neg_mask = tf.placeholder(tf.float32,[None,crop_size_H,crop_size_W],name='Pen_neg')
pos_weight = tf.placeholder(tf.float32,[None,],name='pos_w')
neg_weight = tf.placeholder(tf.float32,[None,],name='neg_w')


print('loadind model VGG')
weights = np.load('vgg16_weights.npz')
keys = sorted(weights.keys())
vgg=vgg16(input_img)

Contour=tf.squeeze(vgg.pred.outputs)



# dis_G,logit_dis_G = discriminator_simplified_api(tf.concat(3,[tf.sigmoid(vgg.pred.outputs),input_img]),\
#                                 is_train=True,reuse=False)
# dis_GT,logit_dis_GT = discriminator_simplified_api(tf.concat(3,[tf.expand_dims(GT,-1),input_img]),\
#                                 is_train=True,reuse=True)

# dis_G,logit_dis_G = discriminator_simplified_api(tf.sigmoid(vgg.pred.outputs),\
#                                 is_train=True,reuse=False)
# dis_GT,logit_dis_GT = discriminator_simplified_api(tf.expand_dims(GT,-1),\
#                                 is_train=True,reuse=True)


vgg_vars = tl.layers.get_variables_with_name('enconde',True,True)
gen_vars = tl.layers.get_variables_with_name('deconde', True, True)
d_vars = tl.layers.get_variables_with_name('discriminator', True, True)

regularizer=tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(weight_decay),gen_vars+d_vars)
# Context loss
loss_context=tf.nn.weighted_cross_entropy_with_logits(Contour,GT,20)
loss_context = tf.reduce_mean(loss_context)+regularizer
tf.summary.scalar('loss_context', loss_context)

# Adversarial loss
# d_loss_real = tl.cost.sigmoid_cross_entropy(logit_dis_G, tf.zeros_like(logit_dis_G), name='dreal')
# d_loss_fake = tl.cost.sigmoid_cross_entropy(logit_dis_GT, tf.ones_like(logit_dis_GT), name='dfake')
# d_loss = d_loss_real + d_loss_fake
# tf.summary.scalar('d_loss', d_loss)

# g_gan_loss = tl.cost.sigmoid_cross_entropy(logit_dis_G, tf.ones_like(logit_dis_G), name='gfake')
# g_loss= g_gan_loss*1e-3+loss_context
# tf.summary.scalar('g_loss', g_loss)

learning_rate = tf.train.exponential_decay(start_learning_rate, global_steps,10000, 0.1, staircase=True)
loss_optim = tf.train.AdamOptimizer(learning_rate,0.9).minimize(loss_context, global_step=global_steps,var_list=gen_vars+vgg_vars)

# g_optim = tf.train.AdamOptimizer(learning_rate).minimize(g_loss, global_step=global_steps,var_list=gen_vars+vgg_vars)
# d_optim = tf.train.AdamOptimizer(learning_rate).minimize(d_loss, var_list=d_vars,global_step=global_steps)

num_batch=int(len(GT_list)/batch_size)
merged = tf.summary.merge_all()

iter_step=0

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.95

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
    train_writer = tf.summary.FileWriter('summary', sess.graph) 
    for i,k in enumerate(keys[:25]):
         sess.run(vgg.parameters[i].assign(weights[k]))
    # print('Model restored')
    # saver.restore(sess,"/opt/code/contour/models/BSDS/models_BSDS500_weighted.ckpt")
    all_names=[]
    for epoch in xrange(epochs):
        np.random.shuffle(GT_list)
        index_begin=0
        for index,batch_indx in enumerate(xrange(num_batch)):
            resall=[]
            GT_batch=GT_list[index_begin:index_begin+batch_size]
            index_begin=index_begin+batch_size

            img_names=[ os.path.basename(x).split('.png')[0] for x in GT_batch]
            gts_path =[os.path.join(GT_path,x+'.png')  for x in img_names ]
            img_path = [os.path.join(Img_path,x.replace('gt','data')+'.jpg')  for x in img_names ]

            imgs_raw=[cv2.imread(x) for x in img_path]
            gts_raw=[cv2.imread(x,0)/255.0 for x in gts_path]

            imgs=np.array(imgs_raw)
            gts=np.array(gts_raw)

            gts[gts>0.5]=1.0
            #gts[gts<0.5]=0.0

            iter_step=iter_step+1

            if(1):
                lr,g,gerarate_loss,res,summary,_ = sess.run([learning_rate,global_steps,loss_context,Contour,merged,loss_optim],\
                        feed_dict={GT:gts,vgg.imgs:imgs})
                resall.extend(res)
                train_writer.add_summary(summary,iter_step)
                save=[imsave(os.path.join('results','{}.png'.format(index)),\
                           1./(1+np.exp(-resall[i]))) for i in np.arange(0,len(resall))] 
                if((index+1)%500==0):
                    saver.save(sess,"/opt/code/contour/models/BSDS/models_BSDS500_weighted.ckpt")
                print("Epoch: [%2d/%2d] [%4d/%4d] learing_rate: %.8f global_step: %d  context_loss: %.6f" \
                    %(epoch, epochs, index, num_batch, lr,g,gerarate_loss))

            else:

                dis_loss,_,summary = sess.run([d_loss,d_optim,merged],\
                            feed_dict={GT:gts,vgg.imgs:imgs})

                for _ in xrange(3):
                	res,gerarate_loss,gen_loss,g_ganloss,summary,_= sess.run([Contour,loss_context,\
                    	g_loss,g_gan_loss,merged,g_optim],\
                        	feed_dict={GT:gts,vgg.imgs:imgs})

                resall.extend(res)
                train_writer.add_summary(summary,iter_step)
                save=[imsave(os.path.join('results','{}.png'.format(index)),\
                       1./(1+np.exp(-resall[i]))) for i in np.arange(0,len(resall))]  
                print("Epoch: [%2d/%2d] [%4d/%4d] d_loss: %.6f,g_loss: %.6f,gerarate_loss: %.6f" \
                    %(epoch, epochs, index, num_batch,dis_loss,gen_loss,gerarate_loss))
                if((index+1)%500==0):
                    saver.save(sess,"/opt/code/contour/models/BSDS/models_BSD500_GAN_weighted512.ckpt")