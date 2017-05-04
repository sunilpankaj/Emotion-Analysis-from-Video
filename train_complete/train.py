
import pandas as pd
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.cm as cm


# In[76]:

#data_original = pd.read_csv("fer2013.csv")
#data = data_original.head(100)
data = pd.read_csv("fer2013.csv")


# In[77]:

#check the number of images and each image data variable
print data.shape


# In[78]:

print data.head()


# In[79]:

np.unique(data["Usage"].values.ravel())


# In[80]:

print 'The number of training data set is %d'%(len(data[data.Usage == "Training"]))


# In[81]:

train_data = data
print len(train_data)


# In[82]:

pixels_values = train_data.pixels.str.split(" ").tolist()


# In[83]:

pixels_values = pd.DataFrame(pixels_values, dtype=int)


# In[84]:

images = pixels_values.values


# In[85]:

images = images.astype(np.float)


# In[86]:

images


# In[87]:

#Define a function to show image through 48*48 pixels
def show(img):
    show_image = img.reshape(48,48)

    #plt.imshow(show_image, cmap=cm.binary)
    #plt.imshow(show_image, cmap='gray')


# In[88]:

#show one image


# ## Image data pre-processing

# In[89]:

images = images - images.mean(axis=1).reshape(-1,1)


# In[90]:

images = np.multiply(images,100.0/255.0)


# In[91]:

each_pixel_mean = images.mean(axis=0)


# In[92]:

each_pixel_std = np.std(images, axis=0)


# In[93]:

images = np.divide(np.subtract(images,each_pixel_mean), each_pixel_std)


# In[94]:

print images.shape


# In[95]:

image_pixels = images.shape[1]
print 'Flat pixel values is %d'%(image_pixels)


# In[96]:

image_width = image_height = np.ceil(np.sqrt(image_pixels)).astype(np.uint8)


# In[97]:

print image_width


# In[98]:

labels_flat = train_data["emotion"].values.ravel()


# In[99]:

labels_count = np.unique(labels_flat).shape[0]


# In[100]:

print 'The number of different facial expressions is %d'%labels_count


# In[101]:

def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


# In[102]:

labels = dense_to_one_hot(labels_flat, labels_count)
labels = labels.astype(np.uint8)


# In[103]:

labels[0]


# In[104]:

# split data into training & validation
VALIDATION_SIZE = 1709


# In[105]:

validation_images = images[:VALIDATION_SIZE]
validation_labels = labels[:VALIDATION_SIZE]

train_images = images[VALIDATION_SIZE:]
train_labels = labels[VALIDATION_SIZE:]


# In[106]:

print 'The number of final training data: %d'%(len(train_images))


# ## Build Tensorflow CNN model

# In[107]:

# weight initialization
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=1e-4)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# In[108]:

# convolution
def conv2d(x, W, padd):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padd)


# In[109]:

# pooling
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')


# In[110]:

# input & output of NN

# images
x = tf.placeholder('float', shape=[None, image_pixels])
# labels
y_ = tf.placeholder('float', shape=[None, labels_count])


# In[111]:

# first convolutional layer 64
W_conv1 = weight_variable([5, 5, 1, 64])
b_conv1 = bias_variable([64])

# (27000, 2304) => (27000,48,48,1)
image = tf.reshape(x, [-1,image_width , image_height,1])
#print (image.get_shape()) # =>(27000,48,48,1)


h_conv1 = tf.nn.relu(conv2d(image, W_conv1, "SAME") + b_conv1)
#print (h_conv1.get_shape()) # => (27000,48,48,64)
h_pool1 = max_pool_2x2(h_conv1)
#print (h_pool1.get_shape()) # => (27000,24,24,1)
h_norm1 = tf.nn.lrn(h_pool1, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)


# In[112]:

# second convolutional layer
W_conv2 = weight_variable([5, 5, 64, 128])
b_conv2 = bias_variable([128])

h_conv2 = tf.nn.relu(conv2d(h_norm1, W_conv2, "SAME") + b_conv2)
#print (h_conv2.get_shape()) # => (27000,24,24,128)

h_norm2 = tf.nn.lrn(h_conv2, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)

h_pool2 = max_pool_2x2(h_norm2)


# In[113]:

# local layer weight initialization
def local_weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.04)
    return tf.Variable(initial)

def local_bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)


# In[114]:

# densely connected layer local 3
W_fc1 = local_weight_variable([12 * 12 * 128, 3072])
b_fc1 = local_bias_variable([3072])

# (27000, 12, 12, 128) => (27000, 12 * 12 * 128)
h_pool2_flat = tf.reshape(h_pool2, [-1, 12 * 12 * 128])

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#print (h_fc1.get_shape()) # => (27000, 1024)


# In[115]:

# densely connected layer local 4
W_fc2 = local_weight_variable([3072, 1536])
b_fc2 = local_bias_variable([1536])

# (40000, 7, 7, 64) => (40000, 3136)
h_fc2_flat = tf.reshape(h_fc1, [-1, 3072])

h_fc2 = tf.nn.relu(tf.matmul(h_fc2_flat, W_fc2) + b_fc2)
#print (h_fc1.get_shape()) # => (40000, 1024)


# In[116]:

# dropout
keep_prob = tf.placeholder('float')
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)


# In[117]:

# readout layer for deep net
W_fc3 = weight_variable([1536, labels_count])
b_fc3 = bias_variable([labels_count])

y = tf.nn.softmax(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)

#print (y.get_shape()) # => (40000, 10)


# In[118]:

# settings
LEARNING_RATE = 1e-4


# In[119]:

# cost function
cross_entropy = -tf.reduce_sum(y_*tf.log(y))


# optimisation function
train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)

# evaluation
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))


# In[120]:

# prediction function
#[0.1, 0.9, 0.2, 0.1, 0.1 0.3, 0.5, 0.1, 0.2, 0.3] => 1
predict = tf.argmax(y,1)


# In[121]:

# set to 3000 iterations
TRAINING_ITERATIONS = 3000

DROPOUT = 0.5
BATCH_SIZE = 50


# In[122]:

epochs_completed = 0
index_in_epoch = 0
num_examples = train_images.shape[0]

# serve data by batches
def next_batch(batch_size):

    global train_images
    global train_labels
    global index_in_epoch
    global epochs_completed

    start = index_in_epoch
    index_in_epoch += batch_size

    # when all trainig data have been already used, it is reorder randomly
    if index_in_epoch > num_examples:
        # finished epoch
        epochs_completed += 1
        # shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        train_images = train_images[perm]
        train_labels = train_labels[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return train_images[start:end], train_labels[start:end]


# In[123]:

# start TensorFlow session
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()

sess.run(init)


# In[124]:

# visualisation variables
train_accuracies = []
validation_accuracies = []
x_range = []

display_step=1


# In[125]:

for i in range(TRAINING_ITERATIONS):

    #get new batch
    batch_xs, batch_ys = next_batch(BATCH_SIZE)

    # check progress on every 1st,2nd,...,10th,20th,...,100th... step
    if i%display_step == 0 or (i+1) == TRAINING_ITERATIONS:

        train_accuracy = accuracy.eval(feed_dict={x:batch_xs,
                                                  y_: batch_ys,
                                                  keep_prob: 1.0})
        if(VALIDATION_SIZE):
            validation_accuracy = accuracy.eval(feed_dict={ x: validation_images[0:BATCH_SIZE],
                                                            y_: validation_labels[0:BATCH_SIZE],
                                                            keep_prob: 1.0})
            print('training_accuracy / validation_accuracy => %.2f / %.2f for step %d'%(train_accuracy, validation_accuracy, i))

            validation_accuracies.append(validation_accuracy)

        else:
             print('training_accuracy => %.4f for step %d'%(train_accuracy, i))
        train_accuracies.append(train_accuracy)
        x_range.append(i)

        # increase display_step
        if i%(display_step*10) == 0 and i and display_step<100:
            display_step *= 10
    # train on batch
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: DROPOUT})


init_op = tf.global_variables_initializer()
saver = tf.train.Saver()

save_path = saver.save(sess, "saved_model/model.ckpt")
print("Model saved in file: %s" % save_path)

