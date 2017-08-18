import tensorflow as tf


# TODO: Remove all num_channels parameters and replace with tf.shape(x)[-1]


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
    convolution_filter = weight_variable([kernel_size[0], kernel_size[1], num_input_channels,
                                          channel_multiplier])
    biases = bias_variable([num_input_channels * channel_multiplier])
    if activation == "ReLU":
        return tf.nn.relu(tf.add(tf.nn.depthwise_conv2d(x, convolution_filter, stride_size, padding='SAME'), biases))
    else:
        return tf.nn.sigmoid(tf.add(tf.nn.depthwise_conv2d(x, convolution_filter, stride_size, padding='SAME'), biases))


# Convolution Transpose
def convolution_transpose_layer(x, kernel_size, stride_size, output_size, num_input_channels, num_output_channels,
                                activation="ReLU"):
    convolution_transpose_filter = weight_variable([kernel_size[0], kernel_size[1],
                                                    num_output_channels, num_input_channels])
    biases = bias_variable([num_output_channels])
    if activation == "ReLU":
        return tf.nn.relu(tf.add(
            tf.nn.conv2d_transpose(x, convolution_transpose_filter,
                                   [tf.shape(x)[0], output_size[0], output_size[1], num_output_channels], stride_size),
            biases))
    else:
        return tf.nn.sigmoid(tf.add(
            tf.nn.conv2d_transpose(x, convolution_transpose_filter,
                                   [tf.shape(x)[0], output_size[0], output_size[1], num_output_channels], stride_size),
            biases))


# Max Pooling
def max_pool_layer(x, kernel_size, stride_size):
    return tf.nn.max_pool(x, ksize=[1, kernel_size[0], kernel_size[1], 1],
                          strides=[1, stride_size[0], stride_size[1], 1], padding='SAME')


def build_model(input_height, input_width, input_channels):
    # Input/Output Placeholders
    X = tf.placeholder(tf.float32, shape=[None, input_height, input_width, input_channels], name="X")
    Y = tf.placeholder(tf.float32, shape=[None, input_height, input_width], name="Y")

    # START ENCODER

    # Convolutional Layer 1
    convolution_layer_1_kernel_size = (5, 5)
    convolution_layer_1_strides = [1, 1, 1, 1]
    convolution_layer_1_channel_multiplier = 5
    convolution_layer_1 = depthwise_convolution_layer(X, convolution_layer_1_kernel_size, convolution_layer_1_strides,
                                                      input_channels, convolution_layer_1_channel_multiplier)

    # Max Pool Layer 1
    pooling_layer_1_kernel_size = (2, 2)
    pooling_layer_1_stride_size = (2, 2)
    pooling_layer_1 = max_pool_layer(convolution_layer_1, pooling_layer_1_kernel_size, pooling_layer_1_stride_size)

    # Convolutional Layer 2
    convolution_layer_2_kernel_size = (5, 5)
    convolution_layer_2_strides = [1, 1, 1, 1]
    convolution_layer_2_channel_multiplier = 10
    convolution_layer_2 = depthwise_convolution_layer(pooling_layer_1, convolution_layer_2_kernel_size,
                                                      convolution_layer_2_strides,
                                                      convolution_layer_1_channel_multiplier * input_channels,
                                                      convolution_layer_2_channel_multiplier)

    # Max Pooling Layer 2
    pooling_layer_2_kernel_size = (2, 2)
    pooling_layer_2_stride_size = (2, 2)
    pooling_layer_2 = max_pool_layer(convolution_layer_2, pooling_layer_2_kernel_size, pooling_layer_2_stride_size)

    # START DECODER

    # Convolutional Transpose Layer 1
    convolution_transpose_layer_1_kernel_size = (5, 5)
    convolution_transpose_layer_1_stride_size = [1, 2, 2, 1]
    convolution_transpose_layer_1_output_size = (120, 160)
    convolution_transpose_layer_1_output_channels = 20
    convolution_transpose_layer_1 = convolution_transpose_layer(pooling_layer_2,
                                                                convolution_transpose_layer_1_kernel_size,
                                                                convolution_transpose_layer_1_stride_size,
                                                                convolution_transpose_layer_1_output_size,
                                                                convolution_layer_2_channel_multiplier *
                                                                convolution_layer_1_channel_multiplier * input_channels,
                                                                convolution_transpose_layer_1_output_channels)

    # Convolutional Transpose Layer 2
    convolution_transpose_layer_2_kernel_size = (5, 5)
    convolution_transpose_layer_2_stride_size = [1, 2, 2, 1]
    convolution_transpose_layer_2_output_size = (240, 320)
    convolution_transpose_layer_2_output_channels = 1
    convolution_transpose_layer_2 = convolution_transpose_layer(convolution_transpose_layer_1,
                                                                convolution_transpose_layer_2_kernel_size,
                                                                convolution_transpose_layer_2_stride_size,
                                                                convolution_transpose_layer_2_output_size,
                                                                convolution_transpose_layer_1_output_channels,
                                                                convolution_transpose_layer_2_output_channels,)

    y_ = convolution_transpose_layer_2
    # cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.reshape(Y, [-1, 240 * 320]),
                                                                           # logits=tf.reshape(y_, [-1, 240 * 320])))
    # cross_entropy = tf.reduce_mean(-1. * tf.reshape(Y, [-1, 240 * 320]) * tf.log(tf.reshape(y_, [-1, 240 * 320])) - \
                    # (1. - tf.reshape(Y, [-1, 240 * 320])) * tf.log(1. - tf.reshape(y_, [-1, 240 * 320])))
    # -1 * tf.reduce_mean(tf.reshape(Y, [-1, 240 * 320]) * tf.log(tf.reshape(y_, [-1, 240 * 320])))
    l2_loss = tf.reduce_mean(tf.square(tf.reshape(Y, [-1, 240 * 320]) - tf.reshape(y_, [-1, 240 * 320])))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(l2_loss)
    return y_, l2_loss, optimizer, X, Y
