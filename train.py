import io
import os
import model
import data_provider
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Get Data
rgb_images, binary_images = data_provider.generate_rgb_and_binary_images(10000)
train_x = rgb_images[:9900]
train_y = binary_images[:9900]
test_x = rgb_images[9900:]
test_y = binary_images[9900:]

# Build Model
y_, loss, optimizer, X, Y = model.build_model(240, 320, 3)
batch_size = 20
training_epochs = 20

# Start Tensorflow Session
session = tf.Session()
tf.global_variables_initializer().run(session=session)

#   Generate Tensorboard Summary
summary_writer = tf.summary.FileWriter('tensorboard', session.graph)
if not os.path.exists('tensorboard'):
    os.makedirs('tensorboard')
with tf.name_scope('Loss'):
    tf.summary.scalar('Loss', loss)
merged_summary = tf.summary.merge_all()

#   Train
total_batches = train_x.shape[0] // batch_size
print "Training..."
for epoch in range(training_epochs):
    for batch in range(total_batches):
        offset = (batch * batch_size) % (train_y.shape[0] - batch_size)
        batch_x = train_x[offset:(offset + batch_size), :, :, :]
        batch_y = train_y[offset:(offset + batch_size), :]
        _, cost, summary = session.run([optimizer, loss, merged_summary], feed_dict={X: batch_x, Y: batch_y})
    print "Epoch: " + str(epoch) + " - Training Loss: " + str(cost)
    summary_writer.add_summary(summary, epoch)

#   Test
total_test_batches = test_x.shape[0] // batch_size
for batch in range(total_test_batches):
    offset = (batch * batch_size) % (test_y.shape[0] - batch_size)
    batch_x = test_x[offset:(offset + batch_size), :, :, :]
    batch_y = test_y[offset:(offset + batch_size), :]
    test_prediction = np.array(session.run([y_], feed_dict={X: batch_x})).reshape([batch_size, 240, 320])
    for i in range(len(test_prediction)):
        plt.figure()
        plt.subplot(131)
        plt.imshow(test_prediction[i])
        plt.subplot(132)
        plt.imshow(batch_y[i])
        plt.subplot(133)
        plt.imshow(batch_x[i])
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.clf()
        plt.close("all")
        prediction_images = tf.image.decode_png(buf.getvalue(), channels=4)
        prediction_images = tf.expand_dims(prediction_images, 0)
        image_summary_op = tf.summary.image("Ground Truth vs Prediction", prediction_images)
        summary_writer.add_summary(session.run(image_summary_op), batch * len(test_prediction) + i)
summary_writer.close()
