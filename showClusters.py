import datetime
import pickle

import matplotlib.pyplot as plt

import numpy as np
import tensorflow.compat.v1 as tf
import time

tf.disable_v2_behavior()
import utilities.params as par
import utilities.loadDataset as DS

"""# Reproducibility"""

tf.reset_default_graph()

np.random.seed(101)
tf.set_random_seed(101)

restore_checkpoint = True
# dsname = UTSig  / mnist / cifar10 / cedar /MCYT75 / SVHN /c-cube
dsname = 'mnist'

"""# Params"""
num_class, image_size1, image_size2, num_image_channel, \
checkpoint_path, alpha, n_epochs, m_plus, m_minus, lambda_, \
init_sigma, caps1_n_dims, caps2_n_dims, caps1_n_maps, \
primary_cap_size1, primary_cap_size2, num_cluster_per_class \
    = par.getParamCaps_Competitve(dsname)

checkpoint_path = checkpoint_path[0:len(checkpoint_path)-1]+"_U/"

caps1_n_caps = caps1_n_maps * primary_cap_size1 * primary_cap_size2  # 1152 primary capsules for mnist
ksize1 = (image_size1 - (primary_cap_size1) * 2 + 2) / 2
ksize2 = (image_size2 - (primary_cap_size2) * 2 + 2) / 2

ksize1 = int(ksize1)
ksize2 = int(ksize2)

caps2_n_caps = num_class * num_cluster_per_class

"""# Load Dataset"""
Train, Train_label, Test, Test_label = DS.loadDataset(dsname)

"""# Input Images"""

X = tf.placeholder(shape=[None, image_size1, image_size2, num_image_channel], dtype=tf.float32, name="X")
# plt.imshow(Test[200].reshape(image_size1,image_size2),cmap='gray')
# plt.show()

"""# Primary Capsules"""

conv1_params = {
    "filters": 256,
    "kernel_size": [ksize1, ksize2],
    "strides": 1,
    "padding": "valid",
    "activation": tf.nn.relu,
}

conv2_params = {
    "filters": caps1_n_maps * caps1_n_dims,  # 256 convolutional filters
    "kernel_size": [ksize1, ksize2],
    "strides": 2,
    "padding": "valid",
    "activation": tf.nn.relu
}

conv1 = tf.layers.conv2d(X, name="conv1", **conv1_params)
conv2 = tf.layers.conv2d(conv1, name="conv2", **conv2_params)

caps1_raw = tf.reshape(conv2, [-1, caps1_n_caps, caps1_n_dims],
                       name="caps1_raw")


def squash(s, axis=-1, epsilon=1e-7, name=None):
    with tf.name_scope(name, default_name="squash"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                     keep_dims=True)
        safe_norm = tf.sqrt(squared_norm + epsilon)
        squash_factor = squared_norm / (1. + squared_norm)
        unit_vector = s / safe_norm
        return squash_factor * unit_vector

caps1_output = squash(caps1_raw, name="caps1_output")

"""# Final Capsules

## Compute the Predicted Output Vectors
"""

W_init = tf.random_normal(
    shape=(1, caps1_n_caps, caps2_n_caps, caps2_n_dims, caps1_n_dims),
    stddev=init_sigma, dtype=tf.float32, name="W_init")
W = tf.Variable(W_init, name="W")

batch_size = tf.shape(X)[0]
W_tiled = tf.tile(W, [batch_size, 1, 1, 1, 1], name="W_tiled")

caps1_output_expanded = tf.expand_dims(caps1_output, -1,
                                       name="caps1_output_expanded")
caps1_output_tile = tf.expand_dims(caps1_output_expanded, 2,
                                   name="caps1_output_tile")
caps1_output_tiled = tf.tile(caps1_output_tile, [1, 1, caps2_n_caps, 1, 1],
                             name="caps1_output_tiled")

caps2_predicted = tf.matmul(W_tiled, caps1_output_tiled,
                            name="caps2_predicted")

"""## Routing by agreement

First let's initialize the raw routing weights $b_{i,j}$ to zero:
"""

raw_weights = tf.zeros([batch_size, caps1_n_caps, caps2_n_caps, 1, 1],
                       dtype=np.float32, name="raw_weights")

"""### Round 1"""

routing_weights = tf.nn.softmax(raw_weights, axis=2, name="routing_weights")

weighted_predictions = tf.multiply(routing_weights, caps2_predicted,
                                   name="weighted_predictions")
weighted_sum = tf.reduce_sum(weighted_predictions, axis=1, keep_dims=True,
                             name="weighted_sum")

caps2_output_round_1 = squash(weighted_sum, axis=-2,
                              name="caps2_output_round_1")

"""### Round 2"""

caps2_output_round_1_tiled = tf.tile(
    caps2_output_round_1, [1, caps1_n_caps, 1, 1, 1],
    name="caps2_output_round_1_tiled")

agreement = tf.matmul(caps2_predicted, caps2_output_round_1_tiled,
                      transpose_a=True, name="agreement")

raw_weights_round_2 = tf.add(raw_weights, agreement,
                             name="raw_weights_round_2")

routing_weights_round_2 = tf.nn.softmax(raw_weights_round_2,
                                        dim=2,
                                        name="routing_weights_round_2")
weighted_predictions_round_2 = tf.multiply(routing_weights_round_2,
                                           caps2_predicted,
                                           name="weighted_predictions_round_2")
weighted_sum_round_2 = tf.reduce_sum(weighted_predictions_round_2,
                                     axis=1, keep_dims=True,
                                     name="weighted_sum_round_2")
caps2_output_round_2 = squash(weighted_sum_round_2,
                              axis=-2,
                              name="caps2_output_round_2")

caps2_output = caps2_output_round_2

"""# Estimated Class Probabilities (Length)"""

def safe_norm(s, axis=-1, epsilon=1e-7, keep_dims=False, name=None):
    with tf.name_scope(name, default_name="safe_norm"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                     keep_dims=keep_dims)
        return tf.sqrt(squared_norm + epsilon)

y_proba = safe_norm(caps2_output, axis=-2, name="y_proba")

# @saeid
y_proba_reshape = tf.reshape(y_proba,[-1, num_cluster_per_class , num_class , 1])
# y_proba_smax = tf.nn.softmax(y_proba_reshape,axis = 1,)
# y_proba_smax_max = tf.reduce_max(y_proba_smax,axis = 1)
y_proba_smax_max = tf.reduce_max(y_proba_reshape,axis = 1)
y_proba_ver2 = tf.expand_dims(y_proba_smax_max,axis = 1)

# @saeid
y_proba_argmax = tf.argmax(y_proba_ver2, axis=2, name="y_proba_argmax")

y_pred = tf.squeeze(y_proba_argmax, axis=[1,2], name="y_pred")

"""# Labels"""

y = tf.placeholder(shape=[None], dtype=tf.int64, name="y")

"""# Margin loss"""

T = tf.one_hot(y, depth=num_class, name="T")

caps2_output_norm = safe_norm(caps2_output, axis=-2, keep_dims=True,
                              name="caps2_output_norm")
# @ saeid code
caps2_output_norm_reshape = tf.reshape(caps2_output_norm,[-1, num_cluster_per_class , num_class ,1, 1])
# caps2_output_norm_smax = tf.nn.softmax(caps2_output_norm_reshape,axis = 1,)
# caps2_output_norm_smax_max = tf.reduce_max(caps2_output_norm_smax,axis = 1)
caps2_output_norm_smax_max = tf.reduce_max(caps2_output_norm_reshape,axis = 1)

caps2_output_norm_ver2 = tf.expand_dims(caps2_output_norm_smax_max,axis = 1)

# @ saeid U
caps2_output_norm_min = tf.reduce_min(caps2_output_norm_reshape,axis = 1)
caps2_output_norm_min_ex = tf.expand_dims(caps2_output_norm_min,axis = 1)

# @saeid
present_error_raw = tf.square(tf.maximum(0., m_plus - caps2_output_norm_ver2),
                              name="present_error_raw")
present_error = tf.reshape(present_error_raw, shape=(-1, num_class),
                           name="present_error")

# @saeid
absent_error_raw = tf.square(tf.maximum(0., caps2_output_norm_ver2 - m_minus),
                             name="absent_error_raw")
absent_error = tf.reshape(absent_error_raw, shape=(-1, num_class),
                          name="absent_error")

# @saeid U
m_middle = 0.4
middle_error_raw = tf.square(tf.maximum(0., m_middle - caps2_output_norm_min_ex),
                              name="middle_error_raw")
middle_error = tf.reshape(middle_error_raw, shape=(-1, num_class),
                           name="middle_error")

L = tf.add(T * present_error, lambda_ * (1.0 - T) * absent_error,
           name="L")
gamma_ = 0.5
L_middle = tf.add(L , gamma_ * T * middle_error , name="L_middle")

margin_loss = tf.reduce_mean(tf.reduce_sum(L_middle, axis=1), name="margin_loss")

"""## Final Loss"""

# loss = tf.add(margin_loss, alpha * reconstruction_loss, name="loss")
regularizer = tf.nn.l2_loss(W)
alpha_ = 0.01
sigma_ = 0.1
beta = alpha_*(0.36 + 0.04 *lambda_*(num_class-1))/(np.sqrt(caps1_n_caps*caps2_n_caps*caps1_n_dims*caps2_n_dims*sigma_))
# beta = 0.000001
loss = tf.add(margin_loss, 2*beta * regularizer, name="loss")
# the above 2 is because of that l2_loss computes sum(t**2) / 2
print(beta)

"""# Final Touches

## Accuracy
"""

correct = tf.equal(y, y_pred, name="correct")
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

"""## Training Operations"""

optimizer = tf.train.AdamOptimizer()
training_op = optimizer.minimize(loss, name="training_op")

"""## Init and Saver"""

init = tf.global_variables_initializer()
saver = tf.train.Saver()


def getNextBatchTrain(batch_size):
    N = np.size(Train, 0)
    idx = np.random.randint(0, N, batch_size)
    batchLabel = Train_label[idx]
    return Train[idx, :], batchLabel.astype('uint8')


def getNextBatchTest(batch_size):
    N = np.size(Test, 0)
    idx = np.random.randint(0, N, batch_size)
    batchLabel = Test_label[idx]
    return Test[idx, :], batchLabel.astype('uint8')


"""# Training"""

batchSize = par.getBatchSize(dsname)
n_iterations_per_epoch = len(Train_label) // batchSize
n_iterations_validation = len(Test_label)

best_loss_val = np.infty

with tf.Session() as sess:
    saver.restore(sess, checkpoint_path)

    for iteration in range(1, n_iterations_validation + 1):
        X_batch = Test[iteration - 1:iteration]
        y_batch = Test_label[iteration - 1:iteration].astype('uint8')

        caps2_mat = sess.run(
            [caps2_output_norm_reshape],
            feed_dict={X: X_batch.reshape([-1, image_size1, image_size2, num_image_channel]),
                       y: y_batch})

        # print(np.shape(caps2_mat))
        caps2_mat = np.array(caps2_mat)
        caps2_mat = caps2_mat[0, 0, :, y_batch[0], 0, 0]
        # print(np.shape(caps2_mat))
        idx = np.argmax(caps2_mat)

        plt.imsave("./result/" + dsname + "/" + str(y_batch[0]) + "_" + str(idx) + "_" + str(iteration) + ".png",
                   np.matrix.transpose(X_batch.reshape(image_size1, image_size2)))

        print("\rEvaluating the model: {}/{} ({:.1f}%)".format(
            iteration, n_iterations_validation,
            iteration * 100 / n_iterations_validation),
            end=" " * 10)
