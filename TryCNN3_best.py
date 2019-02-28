#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 01:03:20 2018

@author: yang
"""
import data
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

X_train, y_train, X_test, y_test, file_name = data.CIFAR10_getData()

#tensorflow sess
sess = tf.Session()

#data argument

rotation_list = np.random.choice(X_train.shape[0],10000)
X_rotation = tf.image.flip_left_right(X_train[rotation_list])
X_rotation = sess.run(X_rotation)
X_train = np.array([*X_train,*X_rotation])
y_train = np.array([*y_train,*y_train[rotation_list]])

"""
angle_list = np.random.choice(60000,10000)
X_angle = tf.contrib.image.rotate(X_train[angle_list],5)
X_angle = sess.run(X_angle)
X_train = np.array([*X_train,*X_angle])
y_train = np.array([*y_train,*y_train[angle_list]])

angle_list_ = np.random.choice(70000,10000)
X_angle_ = tf.contrib.image.rotate(X_train[angle_list_],360)
X_angle_ = sess.run(X_angle_)
X_train = np.array([*X_train,*X_angle_])
y_train = np.array([*y_train,*y_train[angle_list_]])
"""
"""
X_con = tf.image.random_contrast(X_train,lower=0.3, upper=1)
X_con = sess.run(X_con)
X_train = X_con
X_brigh = tf.image.random_brightness(X_train, max_delta=0.5)
X_brigh = sess.run(X_brigh)
X_train = X_brigh
"""

shif_list = np.random.choice(X_train.shape[0],10000)
X_shift = tf.contrib.image.translate(X_train[shif_list],[0.5,0.5])
X_shift = sess.run(X_shift)
X_train = np.array([*X_train,*X_shift])
y_train = np.array([*y_train,*y_train[shif_list]])


"""
j = 0
for i in shif_list:
    X_train[i] = X_shift[j]
    j += 1
"""
"""
X_train = (X_train-127.5) / 127.5
X_test = (X_test-127.5) / 127.5
"""

mean = np.mean(X_train,axis=(0,1,2,3))
std = np.std(X_train,axis=(0,1,2,3))
x_train = (X_train-mean)/(std+1e-7)
x_test = (X_test-mean)/(std+1e-7)
X_train = x_train.astype('float32')
X_test = x_test.astype('float32')


onehot = tf.one_hot(y_train, depth=10)
y_train_onehot = sess.run(onehot)
onehot1 = tf.one_hot(y_test, depth=10)
y_test_onehot = sess.run(onehot1)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x ,W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

#input layer
x = tf.placeholder(tf.float32,[None,32,32,3])

w_conv1 = weight_variable([3,3,3,32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.elu(conv2d(x, w_conv1) + b_conv1)
h_n1 = tf.layers.batch_normalization(h_conv1)

w_conv2 = weight_variable([3,3,32,32])
b_conv2 = bias_variable([32])
h_conv2 = tf.nn.elu(conv2d(h_n1, w_conv2) + b_conv2)
h_n2 = tf.layers.batch_normalization(h_conv2)

h_pool = max_pool_2x2(h_n2)

w_conv3 = weight_variable([3,3,32,64])
b_conv3 = bias_variable([64])
h_conv3 = tf.nn.elu(conv2d(h_pool, w_conv3) + b_conv3)
h_n3 = tf.layers.batch_normalization(h_conv3)

w_conv4 = weight_variable([3,3,64,64])
b_conv4 = bias_variable([64])
h_conv4 = tf.nn.elu(conv2d(h_n3, w_conv4) + b_conv4)
h_n4 = tf.layers.batch_normalization(h_conv4)

h_pool4 = max_pool_2x2(h_n4)
h_dropout4 = tf.nn.dropout(h_pool4, keep_prob=0.4)

w_conv5 = weight_variable([3,3,64,128])
b_conv5 = bias_variable([128])
h_conv5 = tf.nn.elu(conv2d(h_dropout4, w_conv5) + b_conv5)
h_n5 = tf.layers.batch_normalization(h_conv5)

w_conv6 = weight_variable([3,3,128,128])
b_conv6 = bias_variable([128])
h_conv6 = tf.nn.elu(conv2d(h_n5, w_conv6) + b_conv6)
h_n6 = tf.layers.batch_normalization(h_conv6)


h_pool6 = max_pool_2x2(h_n6)
h_dropout6 = tf.nn.dropout(h_pool6, keep_prob=0.5)


#FC layer
w_fc1 = weight_variable([4*4*128,1024])
b_fc1 = bias_variable([1024])
h_flat = tf.reshape(h_dropout6,[-1,4*4*128])
h_fc1 = tf.nn.elu(tf.matmul(h_flat,w_fc1) + b_fc1)

#FC out layer
w_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
h_fc2 = tf.matmul(h_fc1,w_fc2) + b_fc2

#out layer
y_label = tf.placeholder(tf.float32,[None,10])

#loss function
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=h_fc2, labels=y_label)
cross_entropy_v = tf.reduce_mean(cross_entropy)
#cross_entropy = tf.reduce_mean(tf.reduce_sum(tf.square(h_fc1 - y_label)))

#train optima
train_step = tf.train.AdamOptimizer(1e-3, beta1=0.9, beta2=0.99).minimize(cross_entropy)

#acc
correct_prediction = tf.equal(tf.argmax(h_fc2, 1), tf.argmax(y_label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#ini
sess.run(tf.global_variables_initializer())

epoch = 75
batch_size = 512
it = X_train.shape[0]//batch_size

acc_train = []
loss_train = []
acc_test = [0]
y_l = [0]
acc_test = [0]
loss_test = []

for i in range(epoch):
    for j in range(it):
        batch_x, batch_y = X_train[j*batch_size:((j+1)*batch_size)], y_train_onehot[j*batch_size:((j+1)*batch_size)]
        sess.run(train_step, feed_dict={x:batch_x, y_label:batch_y})
        if j % 10 == 0:
            acc, loss = sess.run(accuracy, feed_dict={x: batch_x, y_label: batch_y}), sess.run(cross_entropy_v, feed_dict={x: batch_x, y_label: batch_y})
            print('epoch : ',i,'/',epoch,' , step : ',j,'/',it,' , acc : ',acc,' , loss : ',loss)
            acc_train.append(acc)
            loss_train.append(loss)
            #summary = sess.run(merged, feed_dict={x: batch_x, y_label: batch_y})
            #write.add_summary(summary,(i*it)+j)
    if i % 1 == 0:
        if len(loss_test) == 0:
            loss_test.append(loss_train[0])
        acc_t = sess.run(accuracy, feed_dict={x: X_test, y_label: y_test_onehot})
        loss_t = sess.run(cross_entropy_v, feed_dict={x: X_test, y_label: y_test_onehot})
        print('val_acc : ', acc_t)
        print('val_loss : ', loss_t)
        y_l.append((i+1)*(it/10))
        acc_test.append(acc_t)
        loss_test.append(loss_t)


print('Test accuracy : ',sess.run(accuracy, feed_dict={x: X_test, y_label: y_test_onehot}))

plt.plot(acc_train, label='Train accuracy')
plt.plot(y_l, acc_test, label='Test accuracy')
plt.legend(loc='best')
plt.show()
plt.plot(loss_train, label='train loss')
plt.plot(y_l, loss_test, label='test loss')
plt.legend(loc='best')
plt.show()

#sess.close()

# drop  test acc : 0.6905
# no drop test acc : 0.691 , but overfitting is very critical
# now 0.8    loss : 0.87
