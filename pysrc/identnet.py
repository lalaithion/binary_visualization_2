import numpy as np  #NOQA
import tensorflow as tf
from sklearn.model_selection import train_test_split
import load
import os
import warnings


def cnn_2d_2f(features, labels, mode):
    input_layer = tf.cast(
        tf.reshape(features["x"], [-1, 256, 256, 1]),
        tf.float16)

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=16,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Convolutional Layer #3 and Pooling Layer #3
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool3_flat = tf.reshape(pool3, [-1, 32 * 32 * 32])
    dense = tf.layers.dense(
        inputs=pool3_flat,
        units=1024,
        activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense,
        rate=0.4,
        training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=2)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"]
        )}

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        eval_metric_ops=eval_metric_ops)


def get_data(dir, max=-1):
    features = []
    labels = []
    curr = 0

    for item in os.listdir(dir):
        *_, ext = item.rsplit(".")

        if ext.lower() == "png":
            labels.append(1)
        elif ext.lower() == "jpg":
            labels.append(0)
        else:
            warnings.warn("%s is not a valid extension at this time" % ext)
            continue

        feat = load.load_file(os.path.join(dir, item))
        features.append(feat)
        print("%s loaded!" % item)

        curr += 1
        if curr == max:
            break

    return np.array(features), np.array(labels)


def main(dir):
    data, labels = get_data(dir, max=100)

    print(labels, (labels.sum() / len(labels)))

    # Load training and eval data
    train_data, eval_data, train_labels, eval_labels = train_test_split(
        data,
        labels,
        test_size=0.20)

    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_2d_2f,
        model_dir="/tmp/mnist_convnet_model")

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log,
        every_n_iter=50)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=20,  # 20000,
        hooks=[logging_hook])

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)

    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


if __name__ == "__main__":
    main("../data/")
