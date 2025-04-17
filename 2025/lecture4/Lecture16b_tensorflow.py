# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 2
#     language: python
#     name: python2
# ---

# %% deletable=true editable=true
# %matplotlib inline

import matplotlib
import matplotlib.pyplot as plt

#the above imports the plotting library matplotlib

# %% deletable=true editable=true
#standard imports
import time
import numpy as np
import h5py


# %% deletable=true editable=true
#We're not using the GPU here, so we set the 
#"CUDA_VISIBLE_DEVICES" environment variable to -1
#which tells tensorflow to only use the CPU

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
import tensorflow as tf

# %% [markdown] deletable=true editable=true
# ###  Tensors constants

# %% deletable=true editable=true
#let's define one-node computation graph -- namely, with a single constant
one = tf.constant(1)

# %% deletable=true editable=true
#it's a zero-dimensional tensor -- that is, a scalar
one

# %% deletable=true editable=true
#it's type is a tensorflow Tensor object
type(one)

# %% deletable=true editable=true
#it's datatype is integer
one.dtype

# %% deletable=true editable=true
#it has a tensor shape, just like numpy arrays or HDF5 datasets
one.shape

# %% deletable=true editable=true
#this is how we access the shape object as a python list
one.shape.as_list()

# %% deletable=true editable=true
#if we do this ...
oneinalist = tf.constant([1])

# %% deletable=true editable=true
#... then now we have a vector, with a nontrivial shape 
oneinalist.shape.as_list()
#more on vectors shortly

# %% deletable=true editable=true
#ok, now we're make a three-node computation graph, #
#with two input nodes (the "ones") and an output node (the "two")
two = one + one

# %% deletable=true editable=true
three = one + one + one

# %% deletable=true editable=true
six = tf.constant(6)
twelve = six + six

# %% deletable=true editable=true
thirtysix = twelve * three

# %% deletable=true editable=true
#also a tensor
two

# %% deletable=true editable=true
#still an integer
two.dtype

# %% deletable=true editable=true
#ok so now we want to actually look at the values
#in this compuation graph.   To do that we have to "run" the graph
#in a "session".   This is really just a formality. You can mostly
#ignore it's meaning. 

#so create a "session" object
sess = tf.Session()

# %% deletable=true editable=true
#now run the first tensor in the session
sess.run(one)
#it has value 1, just like we originally programmed it to

# %% deletable=true editable=true
#and "two" has the expected value as well
sess.run(two)

# %% deletable=true editable=true
#to be a little perverted ... 
three = two + two
sess.run(three)

# %% deletable=true editable=true
#we can do the same things with more complex tensors
#like this vector
testvec = tf.constant([1, 2, 3.3])

# %% deletable=true editable=true
#"testvec" is a float value since it had a decimal in its definition
#and it's shape is (3, ) because it is a length-3 vector
testvec

# %% deletable=true editable=true
#we can't quite do this:
testvec + one
#because of type mismatch

# %% deletable=true editable=true
#but we can "cast" the integer value to float and then add the variables:
newvec = testvec + tf.cast(one, tf.float32)

# %% deletable=true editable=true
sess.run(newvec)

# %% deletable=true editable=true
#here's a 2-d tensor (a matrix)
testmat = tf.constant([[1, 2, 3], [3.4, 4, 6]])
testmat

# %% deletable=true editable=true
#you can slice tensors in Tensorflow pretty much like in NumPy
#(though there are some differences)

#the first row of testmat
sess.run(testmat[0])

# %% deletable=true editable=true
#the first column of testmat
sess.run(testmat[:, 0])

# %% deletable=true editable=true
#now let's look at a 3-dimensional tensor 

randarray = np.random.uniform(size=(10, 4, 5), low=-1, high=1)
#see, we can use whatever values we want
testtensor = tf.constant(randarray, dtype=tf.float32)

testtensor

# %% deletable=true editable=true
#right, it's a random array, so .... 
sess.run(testtensor)[0]

# %% deletable=true editable=true
#and you can take its square .... 
sess.run(testtensor**2)[0]

# %% deletable=true editable=true
testtensor**2

# %% deletable=true editable=true
tf.reduce_sum(testtensor**2, axis=0)

# %% [markdown] deletable=true editable=true
# ### Tensor Operations

# %% deletable=true editable=true
#let's create some input data
x = tf.range(-10, 10, .1)

# %% deletable=true editable=true
#this is a tensor, of course
x

# %% deletable=true editable=true
#now let's compute the sine function on the input data
y = tf.sin(x)

# %% deletable=true editable=true
#output is also a tensor
y

# %% deletable=true editable=true
#actually get the concrete values
xvals = sess.run(x)
yvals = sess.run(y)  

#plot them
plt.plot(xvals, yvals)

# %% deletable=true editable=true
#we can do a more complex function easily
x = tf.range(-10, 10, .1)
y = tf.sin(x)
z = y**2 + 10
w = tf.log(z)
plt.plot(sess.run(x), sess.run(w))

# %% [markdown] deletable=true editable=true
# #### Matrix multiplication

# %% deletable=true editable=true
#let's recall a little about matrix multiplication in NumPy
rng = np.random.RandomState(0) #create some random data
randarr = rng.uniform(size=(10, 4))
randarr2 = rng.uniform(size=(10, 4))

multarr = randarr * randarr2 #<-- this is elementwise multiplication in numpy
multarr.shape

# %% deletable=true editable=true
#but matrix multiplication ("dot product") clearly doesn't work for these two matrices
multarr = np.dot(randarr, randarr2)
#... because of shape mismatch

# %% deletable=true editable=true
randarr3 = np.random.RandomState(0).uniform(size=(20, 10))

#this is matrix multiplication in numpy (like we've seen before)
multarr = np.dot(randarr3, randarr)  
multarr.shape

# %% deletable=true editable=true
tf.multiply

# %% deletable=true editable=true
#now let's do the same thing in tensorflow

mat = tf.constant(randarr)
mat2 = tf.constant(randarr2)
mat3 = tf.constant(randarr3)

mat * mat2  #element-wise multiplcation of the tensorflow objects


# %% deletable=true editable=true
sess.run(mat * mat2 - tf.multiply(mat, mat2))

# %% deletable=true editable=true
sess.run(mat + mat2 - tf.add(mat, mat2))

# %% deletable=true editable=true
#matrix multiplication in tensorflow is called "matmul"
tf.matmul(mat3, mat)

# %% [markdown] deletable=true editable=true
# #### Image convolution

# %% deletable=true editable=true
from PIL import Image

# %% deletable=true editable=true
#let's open our two bears image
im = Image.open('two bears.jpg')
im

# %% deletable=true editable=true
#let's make it into a float32 array
imarray = np.asarray(im).astype(np.float32)  #numpy
imtensor = tf.constant(imarray)   #now it's in tensorflow
imtensor.shape.as_list()

# %% deletable=true editable=true
#we're going to apply a constant filter to the image on all channels
#the filter is of the form (height, width, in_channels, out_channels)

k = 2  #filter size of 2

#we structure blocks of 3x3 with ones matrices on the diagonal
#and zeros on the off diagonal.   the 3x3 is due to the fcat 
#that images have 3 input channels and want to make an output that is also in image

filterarray = np.array([[np.ones((k, k)), np.zeros((k, k)), np.zeros((k, k))], 
                         [np.zeros((k, k)), np.ones((k, k)), np.zeros((k, k))], 
                         [np.zeros((k, k)), np.zeros((k, k)), np.ones((k, k))]]) / k**2
filterarray = filterarray.transpose(2, 3, 0, 1) #get the dimensions in the right order (height, width, inchannel, outchannel)

# %% deletable=true editable=true
filterarray.shape

# %% deletable=true editable=true
#this is a 2x2 constant filter
filterarray[:, :, 1, 1]

# %% deletable=true editable=true
filterarray[:, :, 0, 1]

# %% deletable=true editable=true
#ok let's apply 2d convolution using this filter
out = tf.nn.conv2d(imtensor[np.newaxis, :], 
                   filterarray,
                   strides=[1, 1, 1, 1],
                   padding='VALID')

#get the output value
outval = sess.run(out)[0]
print('shape = ', outval.shape)
#and look at it
plt.imshow(outval.astype(np.uint8))

# %% deletable=true editable=true
k = 10  #see, if we increase the size of the filter, the image gets blurrier

filterarray = np.array([[np.ones((k, k)), np.zeros((k, k)), np.zeros((k, k))], 
                         [np.zeros((k, k)), np.ones((k, k)), np.zeros((k, k))], 
                         [np.zeros((k, k)), np.zeros((k, k)), np.ones((k, k))]]) / k**2
filterarray = filterarray.transpose(2, 3, 0, 1)

out = tf.nn.conv2d(imtensor[np.newaxis], filterarray, strides=[1, 1, 1, 1], padding='VALID')
outval = sess.run(out)[0]
plt.imshow(outval.astype(np.uint8))

# %% deletable=true editable=true
#here we take minimum operation after the convolution -- look at the visual effect it has
k = 10
filterarray = np.array([[np.ones((k, k)), np.zeros((k, k)), np.zeros((k, k))], 
                         [np.zeros((k, k)), np.ones((k, k)), np.zeros((k, k))], 
                         [np.zeros((k, k)), np.zeros((k, k)), np.ones((k, k))]]) / k**2
filterarray = filterarray.transpose(2, 3, 0, 1)

out1 = tf.nn.conv2d(imtensor[np.newaxis], filterarray, strides=[1, 1, 1, 1], padding='VALID')
out2 = tf.minimum(out1, 90)
outval = sess.run(out2)[0]
plt.imshow(outval.astype(np.uint8))

#... in fact, this is (a little bit) like what the output of internal layers of 
#deep nets look like when operating on images

# %% deletable=true editable=true
#random filers make what looks like garbage
k = 10
filterarray = np.random.RandomState(0).uniform(size=(k, k, 3, 3))
out = tf.nn.conv2d(imtensor[np.newaxis], filterarray, strides=[1, 1, 1, 1], padding='VALID')
outval = sess.run(out)[0]
print('shape = ', outval.shape)
plt.imshow(outval.astype(np.uint8))

# %% deletable=true editable=true
#just for illustration purposes, let's do some matrix multiplication
#the red channel (first channel) is just a 2-d matrix
im_chnl0 = imtensor[:, :, 0]
print('Shape=', im_chnl0.shape.as_list())
plt.imshow(sess.run(im_chnl0), cmap='gray')

# %% deletable=true editable=true
#let's multiple our image matrix by a matrix of constant value
onemat = 1./100 * tf.ones(shape=(260, 4))
sess.run(onemat[0])

# %% deletable=true editable=true
#actually do the multiplication
outmat = tf.matmul(im_chnl0 , onemat)
outmat.shape.as_list()

# %% deletable=true editable=true
#adding 3 to show it can be done
outmat1 = tf.matmul(im_chnl0 , onemat) + 3

# %% deletable=true editable=true
sess.run(outmat[0])

# %% deletable=true editable=true
#see, we added 3....
sess.run(outmat1[0])

# %% [markdown] deletable=true editable=true
# ### Tensor Variables

# %% deletable=true editable=true
#construct a scalar (0-dimensional) variable with name "x"
eks = tf.get_variable('x', shape=(), dtype=tf.float32)

# %% deletable=true editable=true
#Yep, it's a variable
eks

# %% deletable=true editable=true
eks.name

# %% deletable=true editable=true
#we can do operations on variables just like we can on constant tensors
why = eks**2

# %% deletable=true editable=true
#"why" isn't a variable ... it's just a tensor
why
#this is because "why" doesn't care whether its input was a variable or a constant, it's
#still going to do the same operation regardless. 

# %% deletable=true editable=true
#this doesn't work yet because eks's value isn't actually specified anyhwere
sess.run(eks)

# %% deletable=true editable=true
#we specify it to the runner by using a "feed_dict"
sess.run(eks, feed_dict={eks: 8})

# %% deletable=true editable=true
#we can get anything computed from "eks" (such as "why") as long as we specify
#the value for eks.   this is like "feeding the roots" of the computational graph
#and then looking at some downstream leaf's value
sess.run(why, feed_dict={eks: 8})

# %% deletable=true editable=true
zee = why + 3

# %% deletable=true editable=true
sess.run(zee, feed_dict={eks: 8})

# %% deletable=true editable=true
sess.run(zee, feed_dict={why: 8})

# %% deletable=true editable=true
sess.run(zee, feed_dict={zee: 8})

# %% deletable=true editable=true
#not allowed to use the same variable name willy-nilly ("x" was already used above)
xarr = tf.get_variable('x', shape=(10, 4, 5), dtype=tf.float32)

# %% deletable=true editable=true
#so let's call it something else
x_arr = tf.get_variable('x_arr', shape=(10, 4, 5), dtype=tf.float32)
#... and make in a 3-tensor

# %% deletable=true editable=true
#of course, we can still compute the square
y_arr = x_arr**2

# %% deletable=true editable=true
#and get the actual value
randarray = np.random.uniform(size=(10, 4, 5), low=-1, high=1)

y_val = sess.run(y_arr, feed_dict = {x_arr: randarray})

# %% deletable=true editable=true
y_val[0]

# %% [markdown] deletable=true editable=true
# ### Let's construct SVM hinge loss 

# %% deletable=true editable=true
#ok let's load the neural data 
DATA_PATH = "/home/chengxuz/Class/psych253_2018/data/ventral_neural_data.hdf5"
Ventral_Dataset = h5py.File(DATA_PATH)

categories = Ventral_Dataset['image_meta']['category'][:]   #array of category labels for all images  --> shape == (5760,)
unique_categories = np.unique(categories)                #array of unique category labels --> shape == (8,)

Neural_Data = Ventral_Dataset['time_averaged_trial_averaged'][:]

num_neurons = Neural_Data.shape[1]
num_categories = 8 

# %% deletable=true editable=true
categories[[0, 1200, 2304]]

# %% deletable=true editable=true
#we'll construct 8 one-vs-all binary-valued vectors
category_matrix = np.array([categories == c for 
                             c in unique_categories]).T.astype(int)

# %% deletable=true editable=true
#... one for each category
category_matrix.shape

# %% deletable=true editable=true
#right, this first image is a fruit (5th category)
category_matrix[0]

# %% deletable=true editable=true
#we're not going to process all the images at once, so we batch them up
#... the size of the batches is going to be 256
batch_size = 256

# %% deletable=true editable=true
#let's set up some our key variables
weights = tf.get_variable('weights', 
                          shape=(num_neurons, num_categories),
                          dtype=tf.float32)

bias = tf.get_variable('bias', 
                       shape=(num_categories,),
                       dtype=tf.float32)

# %% deletable=true editable=true
#and placeholder variables as roots of the computation graph
#to receive the inputs
neural_data = tf.get_variable('neural_data',
                              shape=(batch_size, num_neurons),
                              dtype=tf.float32)

# %% deletable=true editable=true
category_labels = tf.get_variable('category_labels',
                                 shape=(batch_size, num_categories),
                                 dtype=tf.float32)

# %% deletable=true editable=true
#out margins formula is really simple
margins = tf.matmul(neural_data, weights) + bias

# %% deletable=true editable=true
#as is the SVM hinge loss
hinge_loss = tf.maximum(0., 1. - category_labels * margins)

# %% deletable=true editable=true
#let's actually compute it
#to do that we have to stick in some values for the weights, bias, and data
rng = np.random.RandomState(0)

initial_weights = rng.uniform(size=(num_neurons, num_categories),
                              low=-1,
                              high=1)

initial_bias = np.zeros((num_categories,))
                             
data_batch = Neural_Data[0: batch_size]
label_batch = category_matrix[0: batch_size]
        

# %% deletable=true editable=true
loss_val = sess.run(hinge_loss, feed_dict={weights: initial_weights,
                                           bias: initial_bias,
                                           neural_data: data_batch,
                                           category_labels: label_batch})
loss_val.shape
#ok it's at least the right shape (data_batch, num_categories)

# %% deletable=true editable=true
loss_val[0]

# %% [markdown] deletable=true editable=true
# ### Actually using trainable variables

# %% deletable=true editable=true
#as we'll see more next time its useful to separate the parameters
#of the model from the real "data" inputs and make the parameters
#(such as the weights and biases) initialize without feeding using the feed_dict

#let's use the tensorflow random uniform sampler to initialize weight balues
initial_weights = tf.random_uniform(shape=(num_neurons, num_categories),
                  minval=-1,
                  maxval=1,
                  seed=0)

# %% deletable=true editable=true
sess.run(initial_weights[0])

# %% deletable=true editable=true
#see if you run it twice without resetting the seed you get different values
sess.run(initial_weights[0])

# %% deletable=true editable=true
initial_bias = tf.zeros(shape=(num_categories,))

# %% deletable=true editable=true
#same idea as before -- we're initializing the weights but now with an 
#initializer as opposed to by hand
weights = tf.get_variable('weights_for_real', 
                           dtype=tf.float32,
                           initializer=initial_weights)
                         
bias = tf.get_variable('bias_for_real', 
                       dtype=tf.float32,
                       initializer=initial_bias)

# %% deletable=true editable=true
initialize_op = tf.global_variables_initializer()
sess.run(initialize_op)
sess.run(weights)

# %% deletable=true editable=true
sess.run(weights)

# %% deletable=true editable=true
#have to reconstruct the margins and loss now that "weights" and "bias" 
#have been redefined

margins = tf.matmul(neural_data, weights) + bias

hinge_loss = tf.maximum(0., 1. - category_labels * margins)

# %% deletable=true editable=true
#ok we should be able to run this without feeding the parameters

loss_val = sess.run(hinge_loss, feed_dict={neural_data: data_batch,
                                           category_labels: label_batch})
loss_val.shape

# %% [markdown] deletable=true editable=true
# ### Getting gradients

# %% deletable=true editable=true
#we want to get back the variable named "x" 
#that we defined before .... here's how to do it 
#without getting the "already define" error

with tf.variable_scope('', reuse=True):
    x = tf.get_variable('x')

# %% deletable=true editable=true
#define an operation on top of the variable "x"
y = x**2

# %% deletable=true editable=true
#let's compute the gradient of y wrt to x
grad = tf.gradients(y, x)

# %% deletable=true editable=true
grad[0]

# %% deletable=true editable=true
sess.run(grad, feed_dict = {x: 0})

# %% deletable=true editable=true
sess.run(grad, feed_dict = {x: 1})

# %% deletable=true editable=true
sess.run(grad, feed_dict = {x: 2})

# %% deletable=true editable=true
sess.run(grad, feed_dict = {x: 3})

# %% deletable=true editable=true
x_arr = tf.get_variable('x_arr2', shape=(100,), dtype=tf.float32)

# %% deletable=true editable=true
y_arr = x_arr**2

# %% deletable=true editable=true
grad_array = tf.gradients(y_arr, x_arr)

# %% deletable=true editable=true
grad_array

# %% [raw] deletable=true editable=true
#

# %% deletable=true editable=true
x_vals = np.arange(-5, 5, .1)
funcval = sess.run(y_arr, feed_dict = {x_arr: x_vals})
gradval = sess.run(grad_array, feed_dict = {x_arr: x_vals})[0]

# %% deletable=true editable=true
type([y_arr, grad_array])

# %% deletable=true editable=true
#or we could have equally well have done
funcval, gradval = sess.run([y_arr, grad_array], feed_dict={x_arr: x_vals})
gradval = gradval[0]

# %% deletable=true editable=true
#or we could have equally well have done
out = sess.run({'funcval': y_arr, 'gradval': grad_array}, feed_dict={x_arr: x_vals})
funcval = out['funcval']
gradval = out['gradval'][0]

# %% deletable=true editable=true
plt.plot(x_vals, funcval, label='the function')
plt.plot(x_vals, gradval, label='its derivative')
plt.legend(loc='lower right')

# %% deletable=true editable=true
#this is the tensorflow "model" definition
y_arr = 2*x_arr**3 - .5*x_arr**2 + 3*x_arr - 1

#and here's the model's derivative
grad_array = tf.gradients(y_arr, x_arr)

#now let's stick in some values
x_vals = np.arange(-5, 5, .1)
funcval = sess.run(y_arr, feed_dict = {x_arr: x_vals})
gradval = sess.run(grad_array, feed_dict = {x_arr: x_vals})[0]

#and plot it
plt.plot(x_vals, funcval, label='the function')
plt.plot(x_vals, gradval, label='its derivative')
plt.legend(loc='lower right')

# %% deletable=true editable=true
x_arr

# %% deletable=true editable=true
y = x_arr**.5
z = tf.log(tf.exp(tf.sin(y) + tf.cos(2*y)) + 1)
w = z / (y + tf.cos(x_arr))

grad_array = tf.gradients(w, x_arr)

x_vals = np.arange(1, 11, .1)
funcval, gradval = sess.run([w, grad_array], feed_dict = {x_arr: x_vals})
gradval = gradval[0]

plt.plot(x_vals, funcval, label='the function')
plt.plot(x_vals, gradval, label='its derivative')
plt.legend(loc='lower right')

# %% deletable=true editable=true

# %% deletable=true editable=true
y = x_arr**.5
z = tf.log(tf.exp(tf.sin(y) + tf.cos(2*y)) + 1)
w = z / (y + tf.cos(x_arr))

grad_array = tf.gradients(w, x_arr)
grad2_array = tf.gradients(grad_array, x_arr)

x_vals = np.arange(1, 11, .1)
funcval, gradval, grad2val = sess.run([w, grad_array, grad2_array], feed_dict = {x_arr: x_vals})
gradval = gradval[0]
grad2val = grad2val[0]

plt.plot(x_vals, funcval, label='the function')
plt.plot(x_vals, gradval, label='its derivative')
plt.plot(x_vals, grad2val, label='its second derivative')
plt.legend(loc='lower right')

# %% deletable=true editable=true
grad_array[0].name

# %% deletable=true editable=true
y = x_arr**(.5)
z = tf.log(tf.exp(tf.sin(y) + tf.cos(2*y)) + 1)
w = z / (y + tf.tan(x_arr))

grad_array = tf.gradients(w, x_arr)

x_vals = np.arange(0, 10, .1)
funcval = sess.run(w, feed_dict = {x_arr: x_vals})
gradval = sess.run(grad_array, feed_dict = {x_arr: x_vals})[0]

plt.plot(x_vals, funcval, label='the function')
plt.plot(x_vals, gradval, label='its derivative')
plt.legend(loc='lower left')

# %% deletable=true editable=true
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# %% deletable=true editable=true
xa = tf.get_variable('var_x', shape=(40000,), dtype=tf.float32)
ya = tf.get_variable('var_y', shape=(40000,), dtype=tf.float32)

# %% deletable=true editable=true
za = xa**2 + ya**2

# %% deletable=true editable=true
grad_array = tf.gradients(za, [xa, ya])

# %% deletable=true editable=true
#derivative wrt the first variable
grad_array[0]

# %% deletable=true editable=true
#derivative wrt the second variable
grad_array[1]

# %% deletable=true editable=true

# %% deletable=true editable=true
za = xa**2 + ya**2
grad_array = tf.gradients(za, [xa, ya])

x_vals = np.arange(-2, 2, .02)
y_vals = np.arange(-2, 2, .02)
xv, yv = np.meshgrid(x_vals, y_vals)
funcval = sess.run(za, feed_dict = {xa: xv.ravel(), ya: yv.ravel()})
gradval = sess.run(grad_array, feed_dict = {xa: xv.ravel(), ya: yv.ravel()})

fig = plt.figure(figsize=(14, 4))
ax = plt.subplot(1, 3, 1, projection='3d')
ax.plot_surface(xv, yv, funcval.reshape((200, 200)), cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
plt.title('function value')
ax = plt.subplot(1, 3, 2, projection='3d')
ax.plot_surface(xv, yv, gradval[0].reshape((200, 200)), cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
plt.title('x derivative')
ax = plt.subplot(1, 3, 3, projection='3d')
ax.plot_surface(xv, yv, gradval[1].reshape((200, 200)), cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
plt.title('y derivative')



# %% deletable=true editable=true
za = xa**2 + ya**2 * xa
grad_array = tf.gradients(za, [xa, ya])

x_vals = np.arange(-2, 2, .02)
y_vals = np.arange(-2, 2, .02)
xv, yv = np.meshgrid(x_vals, y_vals)
funcval = sess.run(za, feed_dict = {xa: xv.ravel(), ya: yv.ravel()})
gradval = sess.run(grad_array, feed_dict = {xa: xv.ravel(), ya: yv.ravel()})

fig = plt.figure(figsize=(14, 4))
ax = plt.subplot(1, 3, 1, projection='3d')
ax.plot_surface(xv, yv, funcval.reshape((200, 200)), cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
plt.title('function value')
ax = plt.subplot(1, 3, 2, projection='3d')
ax.plot_surface(xv, yv, gradval[0].reshape((200, 200)), cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
plt.title('x derivative')
ax = plt.subplot(1, 3, 3, projection='3d')
ax.plot_surface(xv, yv, gradval[1].reshape((200, 200)), cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
plt.title('y derivative')



# %% deletable=true editable=true
za = tf.cosh((3 + tf.cos(xa+2*ya)**3)**.5 * tf.sin(ya-xa))

grad_array = tf.gradients(za, [xa, ya])

x_vals = np.arange(-2, 2, .02)
y_vals = np.arange(-2, 2, .02)
xv, yv = np.meshgrid(x_vals, y_vals)
funcval = sess.run(za, feed_dict = {xa: xv.ravel(), ya: yv.ravel()})
gradval = sess.run(grad_array, feed_dict = {xa: xv.ravel(), ya: yv.ravel()})

fig = plt.figure(figsize=(14, 4))
ax = plt.subplot(1, 3, 1, projection='3d')
ax.plot_surface(xv, yv, funcval.reshape((200, 200)), cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
plt.title('function value')
ax = plt.subplot(1, 3, 2, projection='3d')
ax.plot_surface(xv, yv, gradval[0].reshape((200, 200)), cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
plt.title('x derivative')
ax = plt.subplot(1, 3, 3, projection='3d')
ax.plot_surface(xv, yv, gradval[1].reshape((200, 200)), cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
plt.title('y derivative')

