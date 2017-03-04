#!/usr/bin/env python
#WorldSmallestNet

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import random, os, shutil

#Define Graph
with tf.Graph().as_default():
	#Placeholder
	X1 = tf.placeholder(tf.float32,[None,1],name="Input")
	Y_ = tf.placeholder(tf.float32,[None,1],name="Label")
	#First Layer
	with tf.name_scope('Layer1'):
		w1 = tf.Variable(tf.constant(0.),name="Weights")
		b1 = tf.Variable(tf.constant(0.1),name="bias")
		X2 = tf.nn.tanh(tf.mul(X1,w1)+b1)
		tf.summary.scalar("w1",w1)
		tf.summary.scalar("b1",b1)
	#Second Layer
	with tf.name_scope('Layer2'):
		w2 = tf.Variable(tf.constant(0.),name="Weights")
		b2 = tf.Variable(tf.constant(0.1),name="bias")
		Y = tf.nn.sigmoid(tf.mul(X2,w2)+b2, name="Output")
		tf.summary.scalar("w2",w2)
		tf.summary.scalar("b2",b2)
	#Define Loss and Training Step
	with tf.name_scope('Train'):
		Loss = tf.sqrt(tf.abs(tf.reduce_mean(tf.sub(Y_,Y))),name="Loss")
		train_step = tf.train.AdamOptimizer(0.01).minimize(Loss)
		tf.summary.scalar("Loss",Loss)
	#Delete Older Saves
	if os.path.exists("SaveFiles"):
		shutil.rmtree("SaveFiles")
	#initialize Variables and start session
	summary = tf.summary.merge_all()
	init = tf.global_variables_initializer()
	saver = tf.train.Saver()
	sess = tf.Session()
	summary_writer = tf.summary.FileWriter("SaveFiles", graph=tf.get_default_graph())
	sess.run(init)
	#Train Models
	MaxStep = 5000
	for Step in range(MaxStep):
		# make RandomBatches with 0 and 1
		NumberBatch = []
		for i in range(50):
			Number = float(random.randint(0,1))
			NumberBatch.append(Number)
		NumberBatch = np.array(NumberBatch)
		NumberBatch = np.expand_dims(NumberBatch, axis=1)
		# run Training Step
		sess.run(train_step,feed_dict={Y_:NumberBatch,X1:NumberBatch})
		if Step%2 == 0:
			summary_str = sess.run(summary, feed_dict={Y_:NumberBatch,X1:NumberBatch})
			summary_writer.add_summary(summary_str, Step)
			summary_writer.flush()

		# Save Everything at last step
		if Step+1 == MaxStep:
			checkpoint_file = os.path.join("SaveFiles", 'model.ckpt')
			saver.save(sess, checkpoint_file, global_step=Step)
	# Show Results for 0 and 1
	InputTensor0 = [0]
	InputTensor1 = [1]
	InputTensor0 = np.array(InputTensor0)
	InputTensor1 = np.array(InputTensor1)
	InputTensor0 = np.expand_dims(InputTensor0, axis=1)
	InputTensor1 = np.expand_dims(InputTensor1, axis=1)
	print("Input: 0  | Output: ",sess.run(Y,feed_dict={Y_:InputTensor0,X1:InputTensor0}))
	print("Input: 1  | Output: ",sess.run(Y,feed_dict={Y_:InputTensor1,X1:InputTensor1}))
