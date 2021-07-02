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
dsname = 'mnist'

"""# Params"""
num_class, image_size1, image_size2, num_image_channel, \
checkpoint_path, alpha, n_epochs, m_plus, m_minus, lambda_, \
init_sigma, caps1_n_dims, caps2_n_dims, caps1_n_maps, \
primary_cap_size1, primary_cap_size2, n_hidden1, n_hidden2 \
    = par.getParamCaps(dsname)
n_epochs = 120
primary_cap_size1 = 3
primary_cap_size2 = primary_cap_size1

caps1_n_caps = caps1_n_maps * primary_cap_size1 * primary_cap_size2  # 1152 primary capsules for mnist
ksize1 = (image_size1 - (primary_cap_size1) * 2 + 2) / 2
ksize2 = (image_size2 - (primary_cap_size2) * 2 + 2) / 2

ksize1 = int(ksize1)
ksize2 = int(ksize2)

caps2_n_caps = num_class

"""# Load Dataset"""
Train, Train_label, Test, Test_label = DS.loadDataset(dsname)

"""# Input Images"""

X = tf.placeholder(shape=[None, image_size1, image_size2, num_image_channel], dtype=tf.float32, name="X")

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

routing_weights = tf.nn.softmax(raw_weights, dim=2, name="routing_weights")

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

y_proba_argmax = tf.argmax(y_proba, axis=2, name="y_proba")

y_pred = tf.squeeze(y_proba_argmax, axis=[1, 2], name="y_pred")

"""# Labels"""

y = tf.placeholder(shape=[None], dtype=tf.int64, name="y")

"""# Margin loss"""

T = tf.one_hot(y, depth=caps2_n_caps, name="T")

caps2_output_norm = safe_norm(caps2_output, axis=-2, keep_dims=True,
                              name="caps2_output_norm")

present_error_raw = tf.square(tf.maximum(0., m_plus - caps2_output_norm),
                              name="present_error_raw")
present_error = tf.reshape(present_error_raw, shape=(-1, num_class),
                           name="present_error")

absent_error_raw = tf.square(tf.maximum(0., caps2_output_norm - m_minus),
                             name="absent_error_raw")
absent_error = tf.reshape(absent_error_raw, shape=(-1, num_class),
                          name="absent_error")

L = tf.add(T * present_error, lambda_ * (1.0 - T) * absent_error,
           name="L")

margin_loss = tf.reduce_mean(tf.reduce_sum(L, axis=1), name="margin_loss")

"""# Reconstruction

## Mask
"""

mask_with_labels = tf.placeholder_with_default(False, shape=(),
                                               name="mask_with_labels")

"""Now let's use `tf.cond()` to define the reconstruction targets as the labels `y` if `mask_with_labels` is `True`, or `y_pred` otherwise."""

reconstruction_targets = tf.cond(mask_with_labels,  # condition
                                 lambda: y,  # if True
                                 lambda: y_pred,  # if False
                                 name="reconstruction_targets")

reconstruction_mask = tf.one_hot(reconstruction_targets,
                                 depth=caps2_n_caps,
                                 name="reconstruction_mask")

reconstruction_mask_reshaped = tf.reshape(
    reconstruction_mask, [-1, 1, caps2_n_caps, 1, 1],
    name="reconstruction_mask_reshaped")

caps2_output_masked = tf.multiply(
    caps2_output, reconstruction_mask_reshaped,
    name="caps2_output_masked")

decoder_input = tf.reshape(caps2_output_masked,
                           [-1, caps2_n_caps * caps2_n_dims],
                           name="decoder_input")

"""## Decoder"""

n_output = image_size1 * image_size2 * num_image_channel

with tf.name_scope("decoder"):
    hidden1 = tf.layers.dense(decoder_input, n_hidden1,
                              activation=tf.nn.relu,
                              name="hidden1")
    hidden2 = tf.layers.dense(hidden1, n_hidden2,
                              activation=tf.nn.relu,
                              name="hidden2")
    decoder_output = tf.layers.dense(hidden2, n_output,
                                     activation=tf.nn.sigmoid,
                                     name="decoder_output")

"""## Reconstruction Loss"""

X_flat = tf.reshape(X, [-1, n_output], name="X_flat")
squared_difference = tf.square(X_flat - decoder_output,
                               name="squared_difference")
reconstruction_loss = tf.reduce_mean(squared_difference,
                                     name="reconstruction_loss")

"""## Final Loss"""

loss = tf.add(margin_loss, alpha * reconstruction_loss, name="loss")

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

batch_size = par.getBatchSize(dsname)
n_iterations_per_epoch = len(Train_label) // batch_size
n_iterations_validation = len(Test_label)

best_loss_val = np.infty
with tf.Session() as sess:
    if restore_checkpoint and tf.train.checkpoint_exists(checkpoint_path):
        saver.restore(sess, checkpoint_path)
        with open(checkpoint_path + 'OtherVars', 'rb') as f:
            acc_plot, loss_train_plot, loss_val_plot, time_per_epochs, start_epoch = pickle.load(f)
            start_epoch += 1
        print('\nStarting from epoch: %.0f\n' %(start_epoch + 1))
    else:
        print('\nCheck point not loaded\n')
        init.run()
        acc_plot = []
        loss_train_plot = []
        loss_val_plot = []
        time_per_epochs = []
        start_epoch = 0

    for epoch in range(start_epoch, n_epochs):
        startTime = time.time()
        for iteration in range(1, n_iterations_per_epoch + 1):
            X_batch, y_batch = getNextBatchTrain(batch_size)
            # Run the training operation and measure the loss:
            _, loss_train = sess.run(
                [training_op, loss],
                feed_dict={X: X_batch.reshape([-1, image_size1, image_size2, num_image_channel]),
                           y: y_batch,
                           mask_with_labels: True})
            print("\rIteration: {}/{} ({:.1f}%)  Loss: {:.5f}".format(
                iteration, n_iterations_per_epoch,
                iteration * 100 / n_iterations_per_epoch,
                loss_train),
                end="")
            loss_train_plot.append(loss_train)

        end_time = time.time()

        time_per_epochs.append(end_time - startTime)
        print('\nElapsed: %.1f' % (end_time - startTime))

        remainHour = (n_epochs - epoch) * (end_time - startTime) / 3600
        print('\nEstimated remaining time: %.1f hours' % remainHour)

        # At the end of each epoch,
        # measure the validation loss and accuracy:
        loss_vals = []
        acc_vals = []
        for iteration in range(1, n_iterations_validation + 1):
            X_batch = Test[iteration - 1:iteration]
            y_batch = Test_label[iteration - 1:iteration].astype('uint8')

            loss_val, acc_val = sess.run(
                [loss, accuracy],
                feed_dict={X: X_batch.reshape([-1, image_size1, image_size2, num_image_channel]),
                           y: y_batch})
            loss_vals.append(loss_val)
            acc_vals.append(acc_val)
            print("\rEvaluating the model: {}/{} ({:.1f}%)".format(
                iteration, n_iterations_validation,
                iteration * 100 / n_iterations_validation),
                end=" " * 10)
        loss_val = np.mean(loss_vals)
        acc_val = np.mean(acc_vals)
        print("\rEpoch: {}  Test accuracy: {:.4f}%  Loss: {:.6f}{}".format(
            epoch + 1, acc_val * 100, loss_val,
            " (improved)" if loss_val < best_loss_val else ""))

        acc_plot.append(acc_val)
        loss_val_plot.append(loss_val)

        # And save the model if it improved:
        # if loss_val < best_loss_val:
        save_path = saver.save(sess, checkpoint_path)
        best_loss_val = loss_val

        with open(checkpoint_path + 'OtherVars', 'wb') as f:
            start_epoch = epoch
            pickle.dump([acc_plot, loss_train_plot, loss_val_plot, time_per_epochs, start_epoch], f)


"""# Plot"""

plt.plot(acc_plot)
plt.show()

plt.plot(loss_train_plot)
plt.show()

plt.plot(loss_val_plot)
plt.show()

