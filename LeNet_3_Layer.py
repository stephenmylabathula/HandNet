import tensorflow as tf
import numpy as np
import os
import cv2
import sklearn.utils


# Weights Matrix
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# Bias Vector
def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)


# Depthwise Convolution
def depthwise_convolution_layer(x, kernel_size, stride_size, num_input_channels, channel_multiplier, activation="ReLU"):
    convolution_filter = weight_variable([kernel_size[0], kernel_size[1], num_input_channels, channel_multiplier])
    biases = bias_variable([num_input_channels * channel_multiplier])
    if activation == "ReLU":
        return tf.nn.relu(tf.add(tf.nn.depthwise_conv2d(x, convolution_filter, [1, stride_size[0], stride_size[1], 1],
                                                        padding='VALID'), biases))
    elif activation == "Sigmoid":
        return tf.nn.sigmoid(tf.add(tf.nn.depthwise_conv2d(x, convolution_filter, [1, stride_size[0], stride_size[1], 1],
                                                           padding='VALID'), biases))


# Max Pooling
def max_pool_layer(x, kernel_size, stride_size):
    return tf.nn.max_pool(x, ksize=[1, kernel_size[0], kernel_size[1], 1],
                          strides=[1, stride_size[0], stride_size[1], 1], padding='VALID')


# Create Classifier Model
def build_model(input_height, input_width, num_channels, num_labels):

    # Input/Output
    X = tf.placeholder(tf.float32, shape=[None, input_height, input_width, num_channels])
    Y = tf.placeholder(tf.float32, shape=[None, num_labels])

    # Build Model
    #   Convolutional Layers
    conv_layer_1 = depthwise_convolution_layer(X, kernel_size=(10, 10), stride_size=(1, 1), num_input_channels=num_channels, channel_multiplier=6)
    pool_layer_1 = max_pool_layer(conv_layer_1, kernel_size=(5, 5), stride_size=(4, 4))
    conv_layer_2 = depthwise_convolution_layer(pool_layer_1, kernel_size=(3, 3), stride_size=(1, 1), num_input_channels=num_channels*6, channel_multiplier=3)
    pool_layer_2 = max_pool_layer(conv_layer_2, kernel_size=(2, 2), stride_size=(2, 2))
    conv_layer_3 = depthwise_convolution_layer(pool_layer_2, kernel_size=(2, 2), stride_size=(1, 1), num_input_channels=num_channels*6*3, channel_multiplier=2)
    pool_layer_3 = max_pool_layer(conv_layer_3, kernel_size=(2, 2), stride_size=(2, 2))
    #   Dense Layers
    pool_layer_3_shape = pool_layer_3.get_shape().as_list()
    flattened_input = tf.reshape(pool_layer_3, [-1, pool_layer_3_shape[1] * pool_layer_3_shape[2] * pool_layer_3_shape[3]])
    hidden_layer_1_weights = weight_variable((pool_layer_3_shape[1] * pool_layer_3_shape[2] * pool_layer_3_shape[3], 500))
    hidden_layer_1_bias = bias_variable([500])
    dense_layer_1 = tf.nn.tanh(tf.add(tf.matmul(flattened_input, hidden_layer_1_weights), hidden_layer_1_bias))
    hidden_layer_2_weights = weight_variable((500, 500))
    hidden_layer_2_bias = bias_variable([500])
    dense_layer_2 = tf.nn.tanh(tf.add(tf.matmul(dense_layer_1, hidden_layer_2_weights), hidden_layer_2_bias))
    output_layer_weights = weight_variable((500, 6))
    output_layer_bias = bias_variable([6])
    model_output = tf.nn.tanh(tf.add(tf.matmul(dense_layer_2, output_layer_weights), output_layer_bias))
    #   Define Loss
    loss = tf.reduce_mean(tf.square(Y - model_output))
    # Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    return model_output, loss, optimizer, X, Y


images = []
points = np.zeros((1, 6))

for i in [1, 2, 3, 4, 5]:
    print "Loading Dataset " + str(i)
    dataset_path = '/usr/local/google/home/smylabathula/Desktop/hand_net_data/' + str(i) + '/'
    image_list = os.listdir(dataset_path + 'roi')
    image_list.sort()
    for image_file in image_list:
        img = cv2.imread(dataset_path + 'roi/' + image_file)
        img = img.astype(np.float32) / 255
        images.append(img)
    labels = np.loadtxt('/usr/local/google/home/smylabathula/Desktop/hand_net_data/' + str(i) + '/roi_joint_points.txt',
                        delimiter=',')
    labels[:,[1,3,5]] = labels[:,[1,3,5]] / 100 - 1
    labels[:,[2,4,6]] = -labels[:,[2,4,6]] / 100 + 1
    points = np.append(points, labels[:, 1:], axis=0)

points = points[1:]
images = np.array(images)

x, y = sklearn.utils.shuffle(images, points, random_state=0)
train_x = x[:5000]
train_y = y[:5000]
test_x = x[5000:]
test_y = y[5000:]

y_, loss, optimizer, X, Y = build_model(200, 200, 3, 6)
batch_size = 20
training_epochs = 50

# Start Tensorflow Session
session = tf.Session()
tf.global_variables_initializer().run(session=session)
#   Train
total_batches = train_x.shape[0] // batch_size
print "Training..."
for epoch in range(training_epochs):
    for batch in range(total_batches):
        offset = (batch * batch_size) % (train_y.shape[0] - batch_size)
        batch_x = train_x[offset:(offset + batch_size), :, :, :]
        batch_y = train_y[offset:(offset + batch_size), :]
        _, cost = session.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y})
    print "Epoch: " + str(epoch) + " - Training Loss: " + str(cost)

#   Test
total_test_batches = test_x.shape[0] // batch_size
for batch in range(total_test_batches):
    offset = (batch * batch_size) % (test_y.shape[0] - batch_size)
    batch_x = test_x[offset:(offset + batch_size), :, :, :]
    batch_y = test_y[offset:(offset + batch_size), :]
    test_prediction = np.array(session.run([y_], feed_dict={X: batch_x})).reshape([batch_size, 6])
    for i in range(batch_size):
        img = batch_x[i]
        # Retransform Points and Plot
        cv2.circle(img, (int(100 * (test_prediction[i][0] + 1)), int(-100 * (test_prediction[i][1] - 1))), 5, (0, 0, 255), -1)
        cv2.circle(img, (int(100 * (test_prediction[i][2] + 1)), int(-100 * (test_prediction[i][3] - 1))), 5, (0, 255, 0), -1)
        cv2.circle(img, (int(100 * (test_prediction[i][4] + 1)), int(-100 * (test_prediction[i][5] - 1))), 5, (255, 0, 0), -1)
        cv2.imshow("Image", img)
        cv2.waitKey(0)

saver = tf.train.Saver()
saver.save(session, '/usr/local/google/home/smylabathula/Desktop/hand_net_data/')

session.close()

